from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, List, Sequence

from apps_v2.logparser.services.log_extractor.runner import ConnectionRecord

from .runner_types import PROGRESS_EMIT_EVERY_SECONDS, ProgressCallback


class WiFiRunnerExtractMixin:
    def _iter_records_from_file(
        self,
        file_path: Path,
        rule,
        standard_keys: Sequence[str],
        *,
        progress_callback: ProgressCallback | None,
        bytes_total: int | None,
        chunk_size: int,
        file_fraction_base: float,
        file_fraction_weight: float,
    ) -> Iterator[ConnectionRecord]:
        records_found = 0
        safe_chunk_size = max(1, int(chunk_size or 1))
        safe_bytes_total = int(bytes_total) if bytes_total is not None else None
        if safe_bytes_total is not None and safe_bytes_total < 0:
            safe_bytes_total = None

        last_emit = 0.0
        emit_every_seconds = PROGRESS_EMIT_EVERY_SECONDS

        if progress_callback:
            progress_callback(
                {
                    "pass": "extract",
                    "phase": "extract_start",
                    "file_fraction": self._weighted_file_fraction(base=file_fraction_base, weight=file_fraction_weight, progress=0.0),
                    "bytes_total": safe_bytes_total,
                    "bytes_processed": 0,
                    "records_found": 0,
                    "chunk_index": 0,
                    "chunk_size": safe_chunk_size,
                    "chunk_fill": 0,
                }
            )
            last_emit = time.monotonic()

        with file_path.open("rb") as handle:
            for line_index, raw in enumerate(handle):
                try:
                    raw_text = raw.decode("utf-8", errors="ignore")
                except Exception:
                    raw_text = ""
                raw_original = self._clean_raw_line(raw_text)
                dt, content = self.timestamp_agent.split(raw_original, rule)
                if dt is None or content is None:
                    last_emit = self._maybe_emit_progress(
                        progress_callback,
                        lambda: self._build_extract_payload(
                            phase="extract_progress",
                            file_fraction_base=file_fraction_base,
                            file_fraction_weight=file_fraction_weight,
                            bytes_total=safe_bytes_total,
                            bytes_processed=handle.tell(),
                            line_index=line_index,
                            records_found=records_found,
                            chunk_size=safe_chunk_size,
                        ),
                        last_emit,
                        emit_every_seconds=emit_every_seconds,
                    )
                    continue
                content = str(content).strip()
                if not content:
                    last_emit = self._maybe_emit_progress(
                        progress_callback,
                        lambda: self._build_extract_payload(
                            phase="extract_progress",
                            file_fraction_base=file_fraction_base,
                            file_fraction_weight=file_fraction_weight,
                            bytes_total=safe_bytes_total,
                            bytes_processed=handle.tell(),
                            line_index=line_index,
                            records_found=records_found,
                            chunk_size=safe_chunk_size,
                        ),
                        last_emit,
                        emit_every_seconds=emit_every_seconds,
                    )
                    continue

                preprocessed = self._preprocess_content(content)
                if standard_keys:
                    try:
                        preprocessed = self.clusterer._standardize_json_in_log(preprocessed, standard_keys)
                    except Exception:
                        pass

                info = self.cache_manager.get_cluster_info_for_preprocessed_log(preprocessed)
                if not info:
                    last_emit = self._maybe_emit_progress(
                        progress_callback,
                        lambda: self._build_extract_payload(
                            phase="extract_progress",
                            file_fraction_base=file_fraction_base,
                            file_fraction_weight=file_fraction_weight,
                            bytes_total=safe_bytes_total,
                            bytes_processed=handle.tell(),
                            line_index=line_index,
                            records_found=records_found,
                            chunk_size=safe_chunk_size,
                        ),
                        last_emit,
                        emit_every_seconds=emit_every_seconds,
                    )
                    continue
                assigned_cluster_id, regex_pattern, connect_flag = info
                if connect_flag not in (1, -1):
                    continue
                fields = self._extract_fields(regex_pattern, preprocessed)
                if not fields:
                    continue
                ap_id = self._extract_ap_identifier(fields)
                client_id = self._extract_client_identifier(fields)
                if not (ap_id and client_id):
                    continue

                event = "connect" if connect_flag > 0 else "disconnect"
                metadata = {
                    "connect_flag": connect_flag,
                    "regex": regex_pattern,
                    "fields": fields,
                    "cluster_id": assigned_cluster_id,
                    "line_index": int(line_index),
                    "timestamp_source": "timestamp_agent",
                }
                record = ConnectionRecord(
                    ap_id=ap_id,
                    client_id=client_id,
                    occurred_at=dt,
                    event=event,
                    raw_log=raw_original,
                    metadata=metadata,
                )
                records_found += 1

                last_emit = self._maybe_emit_progress(
                    progress_callback,
                    lambda: self._build_extract_payload(
                        phase="extract_progress",
                        file_fraction_base=file_fraction_base,
                        file_fraction_weight=file_fraction_weight,
                        bytes_total=safe_bytes_total,
                        bytes_processed=handle.tell(),
                        line_index=line_index,
                        records_found=records_found,
                        chunk_size=safe_chunk_size,
                    ),
                    last_emit,
                    emit_every_seconds=emit_every_seconds,
                )

                yield record

        if progress_callback:
            bytes_processed_final = safe_bytes_total
            if bytes_processed_final is None:
                try:
                    bytes_processed_final = int(file_path.stat().st_size)
                except Exception:
                    bytes_processed_final = None
            progress_callback(
                self._build_extract_payload(
                    phase="complete",
                    file_fraction_base=file_fraction_base,
                    file_fraction_weight=file_fraction_weight,
                    bytes_total=safe_bytes_total,
                    bytes_processed=bytes_processed_final,
                    line_index=None,
                    records_found=records_found,
                    chunk_size=safe_chunk_size,
                    force_progress=1.0,
                )
            )

    def _extract_records_from_file(
        self,
        file_path: Path,
        rule,
        standard_keys: Sequence[str],
        *,
        progress_callback: ProgressCallback | None,
        bytes_total: int | None,
        chunk_size: int,
        file_fraction_base: float,
        file_fraction_weight: float,
    ) -> List[ConnectionRecord]:
        return list(
            self._iter_records_from_file(
                file_path,
                rule,
                standard_keys,
                progress_callback=progress_callback,
                bytes_total=bytes_total,
                chunk_size=chunk_size,
                file_fraction_base=file_fraction_base,
                file_fraction_weight=file_fraction_weight,
            )
        )

    def _iter_records_from_logs(
        self,
        raw_logs: Sequence[str],
        rule,
        standard_keys: Sequence[str],
        *,
        progress_callback: ProgressCallback | None,
        chunk_size: int,
        file_fraction_base: float,
        file_fraction_weight: float,
    ) -> Iterator[ConnectionRecord]:
        records_found = 0
        lines_total = len(raw_logs)
        safe_chunk_size = max(1, int(chunk_size or 1))
        last_emit = 0.0
        emit_every_seconds = PROGRESS_EMIT_EVERY_SECONDS

        if progress_callback:
            progress_callback(
                {
                    "pass": "extract",
                    "phase": "extract_start",
                    "file_fraction": self._weighted_file_fraction(base=file_fraction_base, weight=file_fraction_weight, progress=0.0),
                    "lines_total": int(lines_total),
                    "lines_processed": 0,
                    "records_found": 0,
                    "chunk_index": 0,
                    "chunk_size": safe_chunk_size,
                    "chunk_fill": 0,
                }
            )
            last_emit = time.monotonic()

        for line_index, raw in enumerate(raw_logs):
            raw_original = str(raw)
            dt, content = self.timestamp_agent.split(raw_original, rule)
            if dt is None or content is None:
                last_emit = self._maybe_emit_progress(
                    progress_callback,
                    lambda: self._build_extract_payload(
                        phase="extract_progress",
                        file_fraction_base=file_fraction_base,
                        file_fraction_weight=file_fraction_weight,
                        bytes_total=None,
                        bytes_processed=None,
                        line_index=line_index,
                        lines_total=lines_total,
                        records_found=records_found,
                        chunk_size=safe_chunk_size,
                    ),
                    last_emit,
                    emit_every_seconds=emit_every_seconds,
                )
                continue
            content = str(content).strip()
            if not content:
                last_emit = self._maybe_emit_progress(
                    progress_callback,
                    lambda: self._build_extract_payload(
                        phase="extract_progress",
                        file_fraction_base=file_fraction_base,
                        file_fraction_weight=file_fraction_weight,
                        bytes_total=None,
                        bytes_processed=None,
                        line_index=line_index,
                        lines_total=lines_total,
                        records_found=records_found,
                        chunk_size=safe_chunk_size,
                    ),
                    last_emit,
                    emit_every_seconds=emit_every_seconds,
                )
                continue

            preprocessed = self._preprocess_content(content)
            if standard_keys:
                try:
                    preprocessed = self.clusterer._standardize_json_in_log(preprocessed, standard_keys)
                except Exception:
                    pass

            info = self.cache_manager.get_cluster_info_for_preprocessed_log(preprocessed)
            if not info:
                continue
            assigned_cluster_id, regex_pattern, connect_flag = info
            if connect_flag not in (1, -1):
                continue
            fields = self._extract_fields(regex_pattern, preprocessed)
            if not fields:
                continue
            ap_id = self._extract_ap_identifier(fields)
            client_id = self._extract_client_identifier(fields)
            if not (ap_id and client_id):
                continue

            event = "connect" if connect_flag > 0 else "disconnect"
            metadata = {
                "connect_flag": connect_flag,
                "regex": regex_pattern,
                "fields": fields,
                "cluster_id": assigned_cluster_id,
                "line_index": int(line_index),
                "timestamp_source": "timestamp_agent",
            }
            records_found += 1

            last_emit = self._maybe_emit_progress(
                progress_callback,
                lambda: self._build_extract_payload(
                    phase="extract_progress",
                    file_fraction_base=file_fraction_base,
                    file_fraction_weight=file_fraction_weight,
                    bytes_total=None,
                    bytes_processed=None,
                    line_index=line_index,
                    lines_total=lines_total,
                    records_found=records_found,
                    chunk_size=safe_chunk_size,
                ),
                last_emit,
                emit_every_seconds=emit_every_seconds,
            )

            yield ConnectionRecord(
                ap_id=ap_id,
                client_id=client_id,
                occurred_at=dt,
                event=event,
                raw_log=raw_original,
                metadata=metadata,
            )

        if progress_callback:
            progress_callback(
                self._build_extract_payload(
                    phase="complete",
                    file_fraction_base=file_fraction_base,
                    file_fraction_weight=file_fraction_weight,
                    bytes_total=None,
                    bytes_processed=None,
                    line_index=None,
                    lines_total=lines_total,
                    records_found=records_found,
                    chunk_size=safe_chunk_size,
                    force_progress=1.0,
                )
            )

    def _extract_records_from_logs(
        self,
        raw_logs: Sequence[str],
        rule,
        standard_keys: Sequence[str],
        *,
        progress_callback: ProgressCallback | None,
        chunk_size: int,
        file_fraction_base: float,
        file_fraction_weight: float,
    ) -> List[ConnectionRecord]:
        return list(
            self._iter_records_from_logs(
                raw_logs,
                rule,
                standard_keys,
                progress_callback=progress_callback,
                chunk_size=chunk_size,
                file_fraction_base=file_fraction_base,
                file_fraction_weight=file_fraction_weight,
            )
        )
