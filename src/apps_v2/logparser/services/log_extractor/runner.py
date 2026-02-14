from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import logging
import time

import regex

from .common.config import LLMExtractorSettings, load_llm_settings
from .common.utils import normalise_mac, pick_first
from .fallback.llm_repairer import LLMRepairer
from .fallback.regex_validator import RegexValidator
from .llm.batch_loader import BatchLoader
from .llm.llm_parser import LLMParser
from .llm.postprocessor import PostProcessor
from .preprocessing.cache_manager import CacheManager
from .preprocessing.data_clusterer import Cluster, DataClusterer
from .preprocessing.data_loader import DataLoader
from .timestamp_agent import TimestampAgent, TimestampRule


logger = logging.getLogger(__name__)


@dataclass
class ConnectionRecord:
    ap_id: str
    client_id: str
    occurred_at: datetime
    event: str
    raw_log: str
    metadata: dict


@dataclass
class ExtractionStats:
    llm_calls: int
    total_tokens: int
    llm_time: float
    fallback_attempted: int
    fallback_successful: int
    fallback_failed: int
    total_clusters: int
    clusters_processed: int
    clusters_skipped: int
    lines_total: int = 0
    lines_kept: int = 0


@dataclass
class ExtractionOutput:
    records: List[ConnectionRecord]
    cluster_details: Dict[int, dict]
    stats: ExtractionStats


@dataclass(frozen=True)
class _ParsedLogLine:
    line_index: int
    raw_log: str
    content: str
    timestamp: datetime


class LogExtractionRunner:
    """Core orchestrator that adapts batch.py pipeline for the Django backend."""

    timestamp_rule: TimestampRule | None

    def __init__(self, settings: LLMExtractorSettings | None = None):
        self.settings = settings or load_llm_settings()
        self.loader = DataLoader()
        self.clusterer = DataClusterer()
        self.cache_manager = CacheManager(self.clusterer)
        self.batch_loader = BatchLoader()
        min_size = getattr(self.settings, "min_cluster_size", None) or self.settings.batch_size
        self.min_cluster_size = max(1, min_size)

        self.validator = RegexValidator()
        self.postprocessor = PostProcessor()
        self.parser = LLMParser(
            self.settings.primary_model,
            self.settings,
            batch_size=self.settings.batch_size,
        )
        self.repairer = LLMRepairer(self.settings.effective_fallback_models, self.settings)
        self.timestamp_agent = TimestampAgent(self.settings)
        self.timestamp_rule = None

    def process_file(
        self,
        file_path: Path,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> ExtractionOutput:
        self.timestamp_rule = None
        raw_lines = self.loader.load_lines(file_path)
        return self.process_logs(raw_lines, progress_callback=progress_callback)

    def process_logs(
        self,
        raw_logs: Sequence[str],
        progress_callback: Callable[[dict], None] | None = None,
    ) -> ExtractionOutput:
        raw_logs = list(raw_logs)
        logger.info("LogExtractionRunner: loaded %s raw lines", len(raw_logs))
        if not raw_logs:
            return ExtractionOutput(
                records=[],
                cluster_details={},
                stats=ExtractionStats(
                    llm_calls=0,
                    total_tokens=0,
                    llm_time=0.0,
                    fallback_attempted=0,
                    fallback_successful=0,
                    fallback_failed=0,
                    total_clusters=0,
                    clusters_processed=0,
                    clusters_skipped=0,
                ),
            )

        self._ensure_timestamp_rule(raw_logs)
        rule = self.timestamp_rule
        if rule is None:
            raise ValueError("Timestamp rule inference failed")

        parsed_lines: list[_ParsedLogLine] = []
        for line_index, raw_log in enumerate(raw_logs):
            raw_original = str(raw_log)
            dt, content = self.timestamp_agent.split(raw_original, rule)
            if dt is None or content is None:
                continue
            content = str(content).strip()
            if not content:
                continue
            parsed_lines.append(_ParsedLogLine(line_index=line_index, raw_log=raw_original, content=content, timestamp=dt))

        if not parsed_lines:
            stats = ExtractionStats(
                llm_calls=int(self.timestamp_agent.calls),
                total_tokens=int(self.timestamp_agent.total_tokens),
                llm_time=float(self.timestamp_agent.total_time),
                fallback_attempted=0,
                fallback_successful=0,
                fallback_failed=0,
                total_clusters=0,
                clusters_processed=0,
                clusters_skipped=0,
                lines_total=len(raw_logs),
                lines_kept=0,
            )
            return ExtractionOutput(records=[], cluster_details={}, stats=stats)

        normalized_logs = self.loader.preprocess_lines([item.content for item in parsed_lines])
        preprocessed_logs = self.clusterer.normalize_logs(normalized_logs)
        logger.info("LogExtractionRunner: normalization complete")
        log_indices = [item.line_index for item in parsed_lines]
        clusters = self.clusterer.cluster_logs(preprocessed_logs, log_indices)
        logger.info("LogExtractionRunner: %s clusters generated", len(clusters))
        self.cache_manager.merge_clusters(clusters)

        total_clusters = len(self.cache_manager.clusters)
        clusters_to_process = sorted(
            self.cache_manager.get_unprocessed_clusters(self.min_cluster_size),
            key=lambda c: c.size,
            reverse=True,
        )
        clusters_skipped = max(0, total_clusters - len(clusters_to_process))
        if progress_callback:
            progress_callback({
                'phase': 'start',
                'clusters_total': total_clusters,
                'clusters_processed': 0,
                'clusters_skipped': clusters_skipped,
            })

        clusters_processed = 0
        cluster_results: Dict[int, dict] = {}

        for cluster in clusters_to_process:
            logger.info(
                "LogExtractionRunner: processing cluster %s size=%s",
                cluster.cluster_id,
                cluster.size,
            )
            cluster_start = time.monotonic()
            result = self._process_cluster(cluster)
            logger.info(
                "LogExtractionRunner: cluster %s finished in %.2fs",
                cluster.cluster_id,
                time.monotonic() - cluster_start,
            )
            cluster_results[cluster.cluster_id or -1] = result
            clusters_processed += 1
            if progress_callback:
                progress_callback({
                    'phase': 'cluster_complete',
                    'clusters_total': total_clusters,
                    'clusters_processed': clusters_processed,
                    'clusters_skipped': clusters_skipped,
                    'cluster_id': cluster.cluster_id,
                    'cluster_size': cluster.size,
                    'connect_flag': result.get('connect_flag'),
                })

        logger.info(
            "LogExtractionRunner: clusters processed=%s skipped=%s",
            clusters_processed,
            clusters_skipped,
        )
        if progress_callback:
            progress_callback({
                'phase': 'complete',
                'clusters_total': total_clusters,
                'clusters_processed': clusters_processed,
                'clusters_skipped': clusters_skipped,
            })

        records: List[ConnectionRecord] = []
        for idx, item in enumerate(parsed_lines):
            preprocessed = preprocessed_logs[idx] if idx < len(preprocessed_logs) else item.content
            info = self.cache_manager.get_cluster_info_for_preprocessed_log(preprocessed)
            if not info:
                continue
            cluster_id, regex_pattern, connect_flag = info
            if connect_flag not in (1, -1):
                continue
            fields = self._extract_fields(regex_pattern, preprocessed)
            if not fields:
                # Cluster-level classification can be correct even when individual lines
                # do not match the inferred regex (e.g. headers, truncated lines).
                # Skipping here avoids hard-failing the whole extraction.
                continue

            ap_id = self._extract_ap_identifier(fields)
            client_id = self._extract_client_identifier(fields)
            if not (ap_id and client_id):
                continue

            event = "connect" if connect_flag > 0 else "disconnect"
            timestamp = item.timestamp
            metadata = {
                "connect_flag": connect_flag,
                "regex": regex_pattern,
                "fields": fields,
                "cluster_id": cluster_id,
                "line_index": item.line_index,
                "timestamp_source": "timestamp_agent",
            }
            record = ConnectionRecord(
                ap_id=ap_id,
                client_id=client_id,
                occurred_at=timestamp,
                event=event,
                raw_log=item.raw_log,
                metadata=metadata,
            )
            records.append(record)

        stats = ExtractionStats(
            llm_calls=int(self.parser.get_call_count() + self.timestamp_agent.calls + self.repairer.stats.attempted),
            total_tokens=int(self.parser.get_total_tokens() + self.timestamp_agent.total_tokens + self.repairer.total_tokens),
            llm_time=float(self.parser.get_total_time() + self.timestamp_agent.total_time + self.repairer.total_time),
            fallback_attempted=self.repairer.stats.attempted,
            fallback_successful=self.repairer.stats.successful,
            fallback_failed=self.repairer.stats.failed,
            total_clusters=total_clusters,
            clusters_processed=clusters_processed,
            clusters_skipped=clusters_skipped,
            lines_total=len(raw_logs),
            lines_kept=len(parsed_lines),
        )
        return ExtractionOutput(
            records=records,
            cluster_details=cluster_results,
            stats=stats,
        )

    def _ensure_timestamp_rule(self, raw_logs: Sequence[str]) -> None:
        if self.timestamp_rule is not None:
            return
        try:
            self.timestamp_rule = self.timestamp_agent.infer_rule(raw_logs)
        except Exception as exc:
            from apps_v2.llm.budget import BudgetExceeded

            if isinstance(exc, BudgetExceeded):
                raise
            logger.exception("TimestampAgent failed to infer timestamp header rule")
            raise ValueError("TimestampAgent failed to infer timestamp header rule") from exc

    def _process_cluster(self, cluster: Cluster) -> dict:
        template_examples = self._build_examples_text()
        sample_logs = self.batch_loader.sample_cluster(cluster, batch_size=min(self.settings.batch_size, cluster.size))
        logger.info(
            "LogExtractionRunner: invoking LLM for cluster %s sample_size=%s examples=%s",
            cluster.cluster_id,
            len(sample_logs),
            bool(template_examples),
        )
        start_time = time.monotonic()
        raw_output = self.parser.parse_batch(sample_logs, template_examples)
        logger.info(
            "LogExtractionRunner: LLM finished cluster %s in %.2fs",
            cluster.cluster_id,
            time.monotonic() - start_time,
        )
        regex_pattern, connect_flag = self.postprocessor.process_output(raw_output)
        self.cache_manager.update_cluster_flag(cluster, connect_flag)

        if not regex_pattern and sample_logs:
            logger.info(
                "LogExtractionRunner: retrying primary model for cluster %s due to missing regex",
                cluster.cluster_id,
            )
            retry_output = self.parser.parse_batch(sample_logs, template_examples)
            if retry_output:
                retry_regex, retry_flag = self.postprocessor.process_output(retry_output)
                if retry_regex:
                    regex_pattern = retry_regex
                if retry_flag != connect_flag:
                    connect_flag = retry_flag
                    self.cache_manager.update_cluster_flag(cluster, connect_flag)

        validation = self.validator.diagnose(regex_pattern, cluster.logs) if regex_pattern else (False, None, "Missing regex", "missing")
        is_valid, failed_log, failure_info, _ = validation

        if regex_pattern and is_valid:
            self.cache_manager.update_cluster_regex(cluster, regex_pattern)
            return {"regex": regex_pattern, "connect_flag": connect_flag}

        self.cache_manager.update_cluster_regex(cluster, None)
        if connect_flag == 0:
            return {"regex": None, "connect_flag": connect_flag}

        repaired_regex, repaired_flag = self.repairer.repair(
            cluster,
            sample_logs=sample_logs,
            broken_regex=regex_pattern,
            failed_log=failed_log,
            failure_info=failure_info,
            connect_flag=connect_flag,
        )

        if repaired_regex and self.validator.validate_regex(repaired_regex, cluster.logs):
            self.cache_manager.update_cluster_regex(cluster, repaired_regex)
            self.cache_manager.update_cluster_flag(cluster, repaired_flag)
            return {"regex": repaired_regex, "connect_flag": repaired_flag}

        return {"regex": regex_pattern, "connect_flag": connect_flag}

    def _build_examples_text(self) -> str:
        examples = []
        for cluster_id, (regex_pattern, connect_flag, sample_log) in self.cache_manager.cluster_to_regex.items():
            if not regex_pattern:
                continue
            examples.append(
                f"Log: `{sample_log}`\nregex: \"{regex_pattern}\"\nconnect_flag: {connect_flag}"
            )
            if len(examples) >= self.parser.successful_examples:
                break
        if not examples:
            return ""
        return "\n\nSUCCESSFUL EXAMPLES:\n" + "\n\n".join(examples)

    def _extract_fields(self, regex_pattern: str | None, preprocessed_log: str) -> dict:
        if not regex_pattern:
            return {}
        try:
            compiled = regex.compile(regex_pattern)
        except Exception:
            return {}
        try:
            match = compiled.match(preprocessed_log)
        except Exception:
            return {}
        if not match:
            return {}
        return {key: value for key, value in match.groupdict().items() if value}

    def _extract_ap_identifier(self, fields: dict) -> str | None:
        return pick_first(
            fields.get("ap_mac"),
            fields.get("ap_ip"),
            fields.get("ap_name"),
            fields.get("ap"),
            fields.get("ap_id"),
        )

    def _extract_client_identifier(self, fields: dict) -> str | None:
        mac = normalise_mac(
            pick_first(
                fields.get("client_mac"),
                fields.get("client_mac_1"),
                fields.get("client_mac_2"),
            )
        )
        if mac:
            return mac
        return pick_first(
            fields.get("client_name"),
            fields.get("client"),
            fields.get("client_id"),
        )
