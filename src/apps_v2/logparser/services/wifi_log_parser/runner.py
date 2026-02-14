from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, List, Sequence

from apps_v2.logparser.services.log_extractor.common.config import LLMExtractorSettings
from apps_v2.logparser.services.log_extractor.runner import ConnectionRecord, ExtractionOutput, LogExtractionRunner

from .runner_extract import WiFiRunnerExtractMixin
from .runner_induce import WiFiRunnerInduceMixin
from .runner_progress import WiFiRunnerProgressMixin
from .runner_scan import WiFiRunnerScanMixin
from .runner_types import (
    CLUSTER_SAMPLE_LIMIT,
    DEFAULT_RECORD_BATCH_SIZE,
    EXTRACT_WEIGHT,
    INDUCE_WEIGHT,
    SCAN_WEIGHT,
    TIMESTAMP_INFERENCE_MAX_LINES,
    ProgressCallback,
    WiFiLogParserStats,
)
from .template_compiler import TemplateCompiler
from .template_llm_parser import TemplateLLMParser
from .template_postprocessor import TemplatePostProcessor


logger = logging.getLogger(__name__)


_ANGLE_BRACKET_RE = re.compile(r"<\s*([^<>]*?)\s*>")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")


RecordBatchCallback = Callable[[list[ConnectionRecord]], None]


class WiFiLogParserRunner(
    WiFiRunnerExtractMixin,
    WiFiRunnerInduceMixin,
    WiFiRunnerScanMixin,
    WiFiRunnerProgressMixin,
    LogExtractionRunner,
):
    """LogExtractor runner with coverage-driven consolidation.

    Preserves LogExtractor's behaviour of skipping LLM induction for clusters
    below `min_cluster_size`, but allows those tiny clusters to be merged into
    an existing parser bucket when a validated regex covers them.
    """

    def __init__(self, settings: LLMExtractorSettings | None = None):
        super().__init__(settings=settings)
        # Only change the LLM output format for this engine.
        self.parser = TemplateLLMParser(
            self.settings.primary_model,
            self.settings,
            batch_size=self.settings.batch_size,
        )
        self._template_postprocessor = TemplatePostProcessor()
        self._template_compiler = TemplateCompiler()
        self._successful_templates: list[tuple[str, int, str]] = []
        self._standard_keys: list[str] = []

    @staticmethod
    def _preprocess_content(content: str) -> str:
        processed_log = _ANGLE_BRACKET_RE.sub(r"<\1>", str(content))
        processed_log = processed_log.replace("\t", " ")
        processed_log = _MULTI_SPACE_RE.sub(" ", processed_log)
        return processed_log.strip() + "|"

    def process_file(
        self,
        file_path: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> ExtractionOutput:
        """Process a log file in a streaming (chunk-safe) way.

        Unlike the base LogExtractionRunner, this implementation avoids reading
        the entire file into memory.
        """

        output, _ = self._process_file_impl(
            file_path,
            progress_callback=progress_callback,
            on_record_batch=None,
            batch_size=None,
        )
        return output

    def process_file_stream(
        self,
        file_path: Path,
        *,
        on_record_batch: RecordBatchCallback,
        progress_callback: ProgressCallback | None = None,
        batch_size: int | None = None,
    ) -> tuple[ExtractionOutput, int]:
        """Process a file and emit extracted records in batches."""

        return self._process_file_impl(
            file_path,
            progress_callback=progress_callback,
            on_record_batch=on_record_batch,
            batch_size=batch_size,
        )

    def process_logs(
        self,
        raw_logs: Sequence[str],
        progress_callback: ProgressCallback | None = None,
    ) -> ExtractionOutput:
        output, _ = self._process_logs_impl(
            raw_logs,
            progress_callback=progress_callback,
            on_record_batch=None,
            batch_size=None,
        )
        return output

    def process_logs_stream(
        self,
        raw_logs: Sequence[str],
        *,
        on_record_batch: RecordBatchCallback,
        progress_callback: ProgressCallback | None = None,
        batch_size: int | None = None,
    ) -> tuple[ExtractionOutput, int]:
        """Process in-memory logs and emit extracted records in batches."""

        return self._process_logs_impl(
            raw_logs,
            progress_callback=progress_callback,
            on_record_batch=on_record_batch,
            batch_size=batch_size,
        )

    def _process_file_impl(
        self,
        file_path: Path,
        *,
        progress_callback: ProgressCallback | None,
        on_record_batch: RecordBatchCallback | None,
        batch_size: int | None,
    ) -> tuple[ExtractionOutput, int]:
        self.timestamp_rule = None

        try:
            bytes_total = int(file_path.stat().st_size)
        except Exception:
            bytes_total = None

        chunk_size = max(1, int(getattr(self.settings, "chunk_size", 50_000) or 50_000))

        inference_limit = min(
            TIMESTAMP_INFERENCE_MAX_LINES,
            int(getattr(self.settings, "chunk_size", TIMESTAMP_INFERENCE_MAX_LINES) or TIMESTAMP_INFERENCE_MAX_LINES),
        )
        samples = self._sample_lines_from_file(file_path, max_lines=inference_limit)
        if not samples:
            return self._empty_output(lines_total=0, lines_kept=0), 0

        # Infer timestamp rule on a bounded sample (prevents OOM).
        self._ensure_timestamp_rule(samples)
        rule = self.timestamp_rule
        if rule is None:
            raise ValueError("Timestamp rule inference failed")

        standard_keys = self._infer_json_standard_keys(samples, rule)
        self._standard_keys = list(standard_keys)

        # Pass 1: scan file and build memory-bounded clusters.
        lines_total, lines_kept = self._scan_file_build_clusters(
            file_path,
            rule,
            standard_keys,
            cluster_sample_limit=CLUSTER_SAMPLE_LIMIT,
            progress_callback=progress_callback,
            bytes_total=bytes_total,
            chunk_size=chunk_size,
            file_fraction_base=0.0,
            file_fraction_weight=SCAN_WEIGHT,
        )
        logger.info(
            "WiFiLogParserRunner: scanned file lines_total=%s lines_kept=%s clusters=%s",
            lines_total,
            lines_kept,
            len(self.cache_manager.clusters),
        )

        if lines_kept <= 0:
            return (
                self._build_output(
                    records=[],
                    cluster_results={},
                    snapshot=None,
                    lines_total=lines_total,
                    lines_kept=0,
                    llm_calls=int(self.timestamp_agent.calls),
                    total_tokens=int(self.timestamp_agent.total_tokens),
                    llm_time=float(self.timestamp_agent.total_time),
                ),
                0,
            )

        # Pass 1b: induce parsers per eligible cluster + consolidation.
        cluster_results, snapshot = self._process_clusters(
            progress_callback=progress_callback,
            file_fraction_base=SCAN_WEIGHT,
            file_fraction_weight=INDUCE_WEIGHT,
        )

        records: list[ConnectionRecord] = []
        streamed_records = 0
        if on_record_batch is None:
            records = self._extract_records_from_file(
                file_path,
                rule,
                standard_keys,
                progress_callback=progress_callback,
                bytes_total=bytes_total,
                chunk_size=chunk_size,
                file_fraction_base=SCAN_WEIGHT + INDUCE_WEIGHT,
                file_fraction_weight=EXTRACT_WEIGHT,
            )
            streamed_records = len(records)
        else:
            streamed_records = self._stream_records(
                self._iter_records_from_file(
                    file_path,
                    rule,
                    standard_keys,
                    progress_callback=progress_callback,
                    bytes_total=bytes_total,
                    chunk_size=chunk_size,
                    file_fraction_base=SCAN_WEIGHT + INDUCE_WEIGHT,
                    file_fraction_weight=EXTRACT_WEIGHT,
                ),
                on_record_batch=on_record_batch,
                batch_size=batch_size,
            )

        return (
            self._build_output(
                records=records,
                cluster_results=cluster_results,
                snapshot=snapshot,
                lines_total=lines_total,
                lines_kept=lines_kept,
            ),
            int(streamed_records),
        )

    def _process_logs_impl(
        self,
        raw_logs: Sequence[str],
        *,
        progress_callback: ProgressCallback | None,
        on_record_batch: RecordBatchCallback | None,
        batch_size: int | None,
    ) -> tuple[ExtractionOutput, int]:
        if not raw_logs:
            return self._empty_output(lines_total=0, lines_kept=0), 0

        lines_total = len(raw_logs)
        logger.info("WiFiLogParserRunner: loaded %s raw lines", lines_total)

        chunk_size = max(1, int(getattr(self.settings, "chunk_size", 50_000) or 50_000))

        # Infer timestamp rule on a bounded prefix.
        inference_limit = min(
            TIMESTAMP_INFERENCE_MAX_LINES,
            int(getattr(self.settings, "chunk_size", TIMESTAMP_INFERENCE_MAX_LINES) or TIMESTAMP_INFERENCE_MAX_LINES),
        )
        sample_count = min(inference_limit, lines_total)
        samples = [str(raw_logs[i]) for i in range(sample_count) if str(raw_logs[i]).strip()]
        if not samples:
            return self._empty_output(lines_total=lines_total, lines_kept=0), 0

        self._ensure_timestamp_rule(samples)
        rule = self.timestamp_rule
        if rule is None:
            raise ValueError("Timestamp rule inference failed")

        standard_keys = self._infer_json_standard_keys(samples, rule)
        self._standard_keys = list(standard_keys)

        # Pass 1: scan + build bounded clusters.
        lines_kept = self._scan_logs_build_clusters(
            raw_logs,
            rule,
            standard_keys,
            cluster_sample_limit=CLUSTER_SAMPLE_LIMIT,
            progress_callback=progress_callback,
            chunk_size=chunk_size,
            file_fraction_base=0.0,
            file_fraction_weight=SCAN_WEIGHT,
        )
        logger.info(
            "WiFiLogParserRunner: scanned logs lines_total=%s lines_kept=%s clusters=%s",
            lines_total,
            lines_kept,
            len(self.cache_manager.clusters),
        )
        if lines_kept <= 0:
            return (
                self._build_output(
                    records=[],
                    cluster_results={},
                    snapshot=None,
                    lines_total=lines_total,
                    lines_kept=0,
                    llm_calls=int(self.timestamp_agent.calls),
                    total_tokens=int(self.timestamp_agent.total_tokens),
                    llm_time=float(self.timestamp_agent.total_time),
                ),
                0,
            )

        cluster_results, snapshot = self._process_clusters(
            progress_callback=progress_callback,
            file_fraction_base=SCAN_WEIGHT,
            file_fraction_weight=INDUCE_WEIGHT,
        )

        records: list[ConnectionRecord] = []
        streamed_records = 0
        if on_record_batch is None:
            records = self._extract_records_from_logs(
                raw_logs,
                rule,
                standard_keys,
                progress_callback=progress_callback,
                chunk_size=chunk_size,
                file_fraction_base=SCAN_WEIGHT + INDUCE_WEIGHT,
                file_fraction_weight=EXTRACT_WEIGHT,
            )
            streamed_records = len(records)
        else:
            streamed_records = self._stream_records(
                self._iter_records_from_logs(
                    raw_logs,
                    rule,
                    standard_keys,
                    progress_callback=progress_callback,
                    chunk_size=chunk_size,
                    file_fraction_base=SCAN_WEIGHT + INDUCE_WEIGHT,
                    file_fraction_weight=EXTRACT_WEIGHT,
                ),
                on_record_batch=on_record_batch,
                batch_size=batch_size,
            )

        return (
            self._build_output(
                records=records,
                cluster_results=cluster_results,
                snapshot=snapshot,
                lines_total=lines_total,
                lines_kept=lines_kept,
            ),
            int(streamed_records),
        )

    def _stream_records(
        self,
        records: Sequence[ConnectionRecord] | List[ConnectionRecord] | object,
        *,
        on_record_batch: RecordBatchCallback,
        batch_size: int | None,
    ) -> int:
        safe_batch_size = self._resolve_batch_size(batch_size)
        buffer: list[ConnectionRecord] = []
        emitted = 0

        for record in records:  # type: ignore[union-attr]
            buffer.append(record)
            if len(buffer) >= safe_batch_size:
                on_record_batch(buffer)
                emitted += len(buffer)
                buffer = []

        if buffer:
            on_record_batch(buffer)
            emitted += len(buffer)

        return int(emitted)

    @staticmethod
    def _resolve_batch_size(batch_size: int | None) -> int:
        if batch_size is None:
            return DEFAULT_RECORD_BATCH_SIZE
        try:
            return max(1, int(batch_size))
        except Exception:
            return DEFAULT_RECORD_BATCH_SIZE

    def _empty_output(self, *, lines_total: int, lines_kept: int) -> ExtractionOutput:
        return self._build_output(
            records=[],
            cluster_results={},
            snapshot=None,
            lines_total=lines_total,
            lines_kept=lines_kept,
            llm_calls=0,
            total_tokens=0,
            llm_time=0.0,
        )

    def _build_output(
        self,
        *,
        records: list[ConnectionRecord],
        cluster_results: dict[int, dict],
        snapshot: dict | None,
        lines_total: int,
        lines_kept: int,
        llm_calls: int | None = None,
        total_tokens: int | None = None,
        llm_time: float | None = None,
    ) -> ExtractionOutput:
        snap = snapshot or {}
        stats = WiFiLogParserStats(
            llm_calls=int(
                llm_calls
                if llm_calls is not None
                else (self.parser.get_call_count() + self.timestamp_agent.calls + self.repairer.stats.attempted)
            ),
            total_tokens=int(
                total_tokens
                if total_tokens is not None
                else (self.parser.get_total_tokens() + self.timestamp_agent.total_tokens + self.repairer.total_tokens)
            ),
            llm_time=float(
                llm_time
                if llm_time is not None
                else (self.parser.get_total_time() + self.timestamp_agent.total_time + self.repairer.total_time)
            ),
            fallback_attempted=self.repairer.stats.attempted,
            fallback_successful=self.repairer.stats.successful,
            fallback_failed=self.repairer.stats.failed,
            total_clusters=int(snap.get("total_clusters", 0) or 0),
            clusters_processed=int(snap.get("clusters_processed", 0) or 0),
            clusters_skipped=int(snap.get("clusters_skipped", 0) or 0),
            clusters_consolidated=int(snap.get("clusters_consolidated", 0) or 0),
            clusters_consolidated_eligible=int(snap.get("clusters_consolidated_eligible", 0) or 0),
            clusters_consolidated_too_small=int(snap.get("clusters_consolidated_too_small", 0) or 0),
            clusters_too_small_total=int(snap.get("clusters_too_small_total", 0) or 0),
            clusters_too_small_unmatched=int(snap.get("clusters_too_small_unmatched", 0) or 0),
            min_cluster_size=int(self.min_cluster_size),
            lines_total=int(lines_total),
            lines_kept=int(lines_kept),
        )
        return ExtractionOutput(records=records, cluster_details=cluster_results, stats=stats)
