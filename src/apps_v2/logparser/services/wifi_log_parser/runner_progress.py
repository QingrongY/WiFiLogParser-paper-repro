from __future__ import annotations

import time
from typing import Callable

from .runner_types import ProgressCallback


class WiFiRunnerProgressMixin:
    @staticmethod
    def _clean_raw_line(raw: str) -> str:
        return str(raw).rstrip("\n").rstrip("\r")

    @staticmethod
    def _clamp_fraction(value: float | int | None) -> float:
        try:
            f = float(value or 0.0)
        except Exception:
            return 0.0
        if f < 0.0:
            return 0.0
        if f > 1.0:
            return 1.0
        return f

    @classmethod
    def _weighted_file_fraction(cls, *, base: float, weight: float, progress: float | int | None) -> float:
        base_val = float(base or 0.0)
        weight_val = float(weight or 0.0)
        prog = cls._clamp_fraction(progress)
        return cls._clamp_fraction(base_val + (weight_val * prog))

    @staticmethod
    def _maybe_emit_progress(
        progress_callback: ProgressCallback | None,
        payload_builder: Callable[[], dict],
        last_emit: float,
        *,
        emit_every_seconds: float,
    ) -> float:
        if not progress_callback:
            return last_emit
        now = time.monotonic()
        if last_emit and now - last_emit < float(emit_every_seconds):
            return last_emit
        progress_callback(payload_builder())
        return now

    def _build_scan_payload(
        self,
        *,
        phase: str,
        file_fraction_base: float,
        file_fraction_weight: float,
        bytes_total: int | None,
        bytes_processed: int | None,
        lines_processed: int,
        lines_kept: int,
        clusters_found: int,
        chunk_size: int,
        lines_total: int | None = None,
        force_progress: float | None = None,
    ) -> dict:
        safe_chunk_size = max(1, int(chunk_size or 1))
        progress = force_progress
        if progress is None:
            if bytes_total is not None and bytes_processed is not None and int(bytes_total) > 0:
                progress = float(bytes_processed) / float(bytes_total)
            elif lines_total is not None and int(lines_total) > 0:
                progress = float(lines_processed) / float(lines_total)
            else:
                progress = 0.0

        chunk_index = 0
        chunk_fill = 0
        if int(lines_processed or 0) > 0:
            chunk_index = (int(lines_processed) - 1) // safe_chunk_size
            chunk_fill = ((int(lines_processed) - 1) % safe_chunk_size) + 1

        phase_text = str(phase)
        if phase_text == "scan_progress" and chunk_fill and safe_chunk_size:
            phase_text = f"scan_progress (chunk {chunk_index + 1} {chunk_fill}/{safe_chunk_size})"

        payload: dict[str, object] = {
            "pass": "scan",
            "phase": phase_text,
            "file_fraction": self._weighted_file_fraction(
                base=file_fraction_base,
                weight=file_fraction_weight,
                progress=progress,
            ),
            "bytes_total": int(bytes_total) if bytes_total is not None else None,
            "bytes_processed": int(bytes_processed) if bytes_processed is not None else None,
            "lines_total": int(lines_total) if lines_total is not None else None,
            "lines_processed": int(lines_processed),
            "lines_kept": int(lines_kept),
            "clusters_found": int(clusters_found),
            "chunk_index": int(chunk_index),
            "chunk_size": int(safe_chunk_size),
            "chunk_fill": int(chunk_fill),
        }
        return {k: v for k, v in payload.items() if v is not None}

    def _build_extract_payload(
        self,
        *,
        phase: str,
        file_fraction_base: float,
        file_fraction_weight: float,
        bytes_total: int | None,
        bytes_processed: int | None,
        line_index: int | None,
        records_found: int,
        chunk_size: int,
        lines_total: int | None = None,
        force_progress: float | None = None,
    ) -> dict:
        safe_chunk_size = max(1, int(chunk_size or 1))
        lines_processed = (int(line_index) + 1) if line_index is not None else None

        progress = force_progress
        if progress is None:
            if bytes_total is not None and bytes_processed is not None and int(bytes_total) > 0:
                progress = float(bytes_processed) / float(bytes_total)
            elif lines_total is not None and lines_processed is not None and int(lines_total) > 0:
                progress = float(lines_processed) / float(lines_total)
            else:
                progress = 0.0

        chunk_index = 0
        chunk_fill = 0
        if lines_processed is not None and int(lines_processed or 0) > 0:
            chunk_index = (int(lines_processed) - 1) // safe_chunk_size
            chunk_fill = ((int(lines_processed) - 1) % safe_chunk_size) + 1

        phase_text = str(phase)
        if phase_text == "extract_progress" and chunk_fill and safe_chunk_size:
            phase_text = f"extract_progress (chunk {chunk_index + 1} {chunk_fill}/{safe_chunk_size})"

        payload: dict[str, object] = {
            "pass": "extract",
            "phase": phase_text,
            "file_fraction": self._weighted_file_fraction(
                base=file_fraction_base,
                weight=file_fraction_weight,
                progress=progress,
            ),
            "bytes_total": int(bytes_total) if bytes_total is not None else None,
            "bytes_processed": int(bytes_processed) if bytes_processed is not None else None,
            "lines_total": int(lines_total) if lines_total is not None else None,
            "line_index": int(line_index) if line_index is not None else None,
            "records_found": int(records_found),
            "chunk_index": int(chunk_index),
            "chunk_size": int(safe_chunk_size),
            "chunk_fill": int(chunk_fill),
        }
        return {k: v for k, v in payload.items() if v is not None}
