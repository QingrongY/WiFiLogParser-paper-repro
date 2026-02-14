from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

from .runner_types import PROGRESS_EMIT_EVERY_SECONDS, ProgressCallback, StreamingCluster


class WiFiRunnerScanMixin:
    @classmethod
    def _sample_lines_from_file(cls, file_path: Path, *, max_lines: int) -> list[str]:
        take = max(1, int(max_lines))
        samples: list[str] = []
        with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw in handle:
                line = cls._clean_raw_line(raw)
                if not line.strip():
                    continue
                samples.append(line)
                if len(samples) >= take:
                    break
        return samples

    def _infer_json_standard_keys(self, raw_samples: Sequence[str], rule) -> list[str]:
        if not raw_samples:
            return []

        # Cap JSON key inference to a small pool.
        cap = 2_000
        preprocessed: list[str] = []
        for raw in raw_samples:
            if len(preprocessed) >= cap:
                break
            dt, content = self.timestamp_agent.split(str(raw), rule)
            if dt is None or content is None:
                continue
            content = str(content).strip()
            if not content:
                continue
            preprocessed.append(self._preprocess_content(content))

        if not preprocessed:
            return []
        try:
            return list(self.clusterer._collect_json_keys(preprocessed))
        except Exception:
            return []

    def _get_or_create_streaming_cluster(
        self,
        template: str,
        *,
        cluster_sample_limit: int,
    ) -> StreamingCluster:
        template_key = str(template)
        existing_id = self.cache_manager.template_to_cluster.get(template_key)
        if existing_id is not None:
            existing = self.cache_manager.clusters.get(existing_id)
            if existing is not None:
                # If this is not a StreamingCluster (e.g., pre-populated by a different code path),
                # fall back to creating a new bounded cluster.
                if isinstance(existing, StreamingCluster):
                    return existing

        cluster_id = int(getattr(self.cache_manager, "next_cluster_id", 0) or 0)
        self.cache_manager.next_cluster_id = cluster_id + 1
        cluster = StreamingCluster(
            cluster_id=cluster_id,
            template=template_key,
            sample_limit=max(1, int(cluster_sample_limit)),
        )
        self.cache_manager.template_to_cluster[template_key] = cluster_id
        self.cache_manager.clusters[cluster_id] = cluster  # pyright: ignore[reportGeneralTypeIssues]
        return cluster

    def _scan_file_build_clusters(
        self,
        file_path: Path,
        rule,
        standard_keys: Sequence[str],
        *,
        cluster_sample_limit: int,
        progress_callback: ProgressCallback | None,
        bytes_total: int | None,
        chunk_size: int,
        file_fraction_base: float,
        file_fraction_weight: float,
    ) -> tuple[int, int]:
        lines_total = 0
        lines_kept = 0
        safe_chunk_size = max(1, int(chunk_size or 1))
        safe_bytes_total = int(bytes_total) if bytes_total is not None else None
        if safe_bytes_total is not None and safe_bytes_total < 0:
            safe_bytes_total = None

        last_emit = 0.0
        emit_every_seconds = PROGRESS_EMIT_EVERY_SECONDS

        if progress_callback:
            progress_callback(
                {
                    "pass": "scan",
                    "phase": "scan_start",
                    "file_fraction": self._weighted_file_fraction(base=file_fraction_base, weight=file_fraction_weight, progress=0.0),
                    "bytes_total": safe_bytes_total,
                    "bytes_processed": 0,
                    "lines_processed": 0,
                    "lines_kept": 0,
                    "clusters_found": 0,
                    "chunk_index": 0,
                    "chunk_size": safe_chunk_size,
                    "chunk_fill": 0,
                }
            )
            last_emit = time.monotonic()

        with file_path.open("rb") as handle:
            for line_index, raw in enumerate(handle):
                lines_total += 1

                try:
                    raw_text = raw.decode("utf-8", errors="ignore")
                except Exception:
                    raw_text = ""

                raw_original = self._clean_raw_line(raw_text)
                dt, content = self.timestamp_agent.split(raw_original, rule)
                if dt is None or content is None:
                    # Still emit a heartbeat so the UI doesn't look stuck.
                    last_emit = self._maybe_emit_progress(
                        progress_callback,
                        lambda: self._build_scan_payload(
                            phase="scan_progress",
                            file_fraction_base=file_fraction_base,
                            file_fraction_weight=file_fraction_weight,
                            bytes_total=safe_bytes_total,
                            bytes_processed=handle.tell(),
                            lines_processed=lines_total,
                            lines_kept=lines_kept,
                            clusters_found=len(self.cache_manager.clusters),
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
                        lambda: self._build_scan_payload(
                            phase="scan_progress",
                            file_fraction_base=file_fraction_base,
                            file_fraction_weight=file_fraction_weight,
                            bytes_total=safe_bytes_total,
                            bytes_processed=handle.tell(),
                            lines_processed=lines_total,
                            lines_kept=lines_kept,
                            clusters_found=len(self.cache_manager.clusters),
                            chunk_size=safe_chunk_size,
                        ),
                        last_emit,
                        emit_every_seconds=emit_every_seconds,
                    )
                    continue

                lines_kept += 1

                preprocessed = self._preprocess_content(content)
                if standard_keys:
                    try:
                        preprocessed = self.clusterer._standardize_json_in_log(preprocessed, standard_keys)
                    except Exception:
                        pass
                template = self.clusterer._normalize_log_for_template(preprocessed)
                cluster = self._get_or_create_streaming_cluster(template, cluster_sample_limit=cluster_sample_limit)
                cluster.add(preprocessed, index=line_index)

                last_emit = self._maybe_emit_progress(
                    progress_callback,
                    lambda: self._build_scan_payload(
                        phase="scan_progress",
                        file_fraction_base=file_fraction_base,
                        file_fraction_weight=file_fraction_weight,
                        bytes_total=safe_bytes_total,
                        bytes_processed=handle.tell(),
                        lines_processed=lines_total,
                        lines_kept=lines_kept,
                        clusters_found=len(self.cache_manager.clusters),
                        chunk_size=safe_chunk_size,
                    ),
                    last_emit,
                    emit_every_seconds=emit_every_seconds,
                )

        if progress_callback:
            bytes_processed_final = safe_bytes_total
            if bytes_processed_final is None:
                try:
                    bytes_processed_final = int(file_path.stat().st_size)
                except Exception:
                    bytes_processed_final = None
            progress_callback(
                self._build_scan_payload(
                    phase="scan_complete",
                    file_fraction_base=file_fraction_base,
                    file_fraction_weight=file_fraction_weight,
                    bytes_total=safe_bytes_total,
                    bytes_processed=bytes_processed_final,
                    lines_processed=lines_total,
                    lines_kept=lines_kept,
                    clusters_found=len(self.cache_manager.clusters),
                    chunk_size=safe_chunk_size,
                    force_progress=1.0,
                )
            )

        return int(lines_total), int(lines_kept)

    def _scan_logs_build_clusters(
        self,
        raw_logs: Sequence[str],
        rule,
        standard_keys: Sequence[str],
        *,
        cluster_sample_limit: int,
        progress_callback: ProgressCallback | None,
        chunk_size: int,
        file_fraction_base: float,
        file_fraction_weight: float,
    ) -> int:
        lines_total = len(raw_logs)
        lines_kept = 0
        safe_chunk_size = max(1, int(chunk_size or 1))
        last_emit = 0.0
        emit_every_seconds = PROGRESS_EMIT_EVERY_SECONDS

        if progress_callback:
            progress_callback(
                {
                    "pass": "scan",
                    "phase": "scan_start",
                    "file_fraction": self._weighted_file_fraction(base=file_fraction_base, weight=file_fraction_weight, progress=0.0),
                    "lines_total": int(lines_total),
                    "lines_processed": 0,
                    "lines_kept": 0,
                    "clusters_found": 0,
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
                    lambda: self._build_scan_payload(
                        phase="scan_progress",
                        file_fraction_base=file_fraction_base,
                        file_fraction_weight=file_fraction_weight,
                        bytes_total=None,
                        bytes_processed=None,
                        lines_processed=line_index + 1,
                        lines_total=lines_total,
                        lines_kept=lines_kept,
                        clusters_found=len(self.cache_manager.clusters),
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
                    lambda: self._build_scan_payload(
                        phase="scan_progress",
                        file_fraction_base=file_fraction_base,
                        file_fraction_weight=file_fraction_weight,
                        bytes_total=None,
                        bytes_processed=None,
                        lines_processed=line_index + 1,
                        lines_total=lines_total,
                        lines_kept=lines_kept,
                        clusters_found=len(self.cache_manager.clusters),
                        chunk_size=safe_chunk_size,
                    ),
                    last_emit,
                    emit_every_seconds=emit_every_seconds,
                )
                continue
            lines_kept += 1

            preprocessed = self._preprocess_content(content)
            if standard_keys:
                try:
                    preprocessed = self.clusterer._standardize_json_in_log(preprocessed, standard_keys)
                except Exception:
                    pass
            template = self.clusterer._normalize_log_for_template(preprocessed)
            cluster = self._get_or_create_streaming_cluster(template, cluster_sample_limit=cluster_sample_limit)
            cluster.add(preprocessed, index=line_index)

            last_emit = self._maybe_emit_progress(
                progress_callback,
                lambda: self._build_scan_payload(
                    phase="scan_progress",
                    file_fraction_base=file_fraction_base,
                    file_fraction_weight=file_fraction_weight,
                    bytes_total=None,
                    bytes_processed=None,
                    lines_processed=line_index + 1,
                    lines_total=lines_total,
                    lines_kept=lines_kept,
                    clusters_found=len(self.cache_manager.clusters),
                    chunk_size=safe_chunk_size,
                ),
                last_emit,
                emit_every_seconds=emit_every_seconds,
            )

        if progress_callback:
            progress_callback(
                self._build_scan_payload(
                    phase="scan_complete",
                    file_fraction_base=file_fraction_base,
                    file_fraction_weight=file_fraction_weight,
                    bytes_total=None,
                    bytes_processed=None,
                    lines_processed=lines_total,
                    lines_total=lines_total,
                    lines_kept=lines_kept,
                    clusters_found=len(self.cache_manager.clusters),
                    chunk_size=safe_chunk_size,
                    force_progress=1.0,
                )
            )

        return int(lines_kept)
