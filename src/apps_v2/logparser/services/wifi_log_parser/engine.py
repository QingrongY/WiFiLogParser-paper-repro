from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Sequence

from apps_v2.logparser.services.payload_adapter import record_to_payload, runner_output_to_payload, streamed_payload

from .runner import WiFiLogParserRunner


ProgressCallback = Callable[[dict], None]
RecordBatchCallback = Callable[[list[dict]], None]


class WiFiLogParserEngine:
    """Hybrid engine: LogExtractor + coverage-driven consolidation."""

    name = "wifi_log_parser"

    def __init__(self):
        self._runner = WiFiLogParserRunner()

    def process_file(self, file_path: Path, *, progress_callback: ProgressCallback | None = None) -> Dict[str, object]:
        output = self._runner.process_file(file_path, progress_callback=progress_callback)
        return runner_output_to_payload(output)

    def process_logs(
        self,
        raw_logs: Sequence[str],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> Dict[str, object]:
        output = self._runner.process_logs(raw_logs, progress_callback=progress_callback)
        return runner_output_to_payload(output)

    def process_file_stream(
        self,
        file_path: Path,
        *,
        on_record_batch: RecordBatchCallback,
        progress_callback: ProgressCallback | None = None,
        batch_size: int | None = None,
    ) -> Dict[str, object]:
        def _emit(batch):
            if not batch:
                return
            on_record_batch([record_to_payload(record) for record in batch])

        output, streamed = self._runner.process_file_stream(
            file_path,
            on_record_batch=_emit,
            progress_callback=progress_callback,
            batch_size=batch_size,
        )
        return streamed_payload(output, streamed_count=streamed)

    def process_logs_stream(
        self,
        raw_logs: Sequence[str],
        *,
        on_record_batch: RecordBatchCallback,
        progress_callback: ProgressCallback | None = None,
        batch_size: int | None = None,
    ) -> Dict[str, object]:
        def _emit(batch):
            if not batch:
                return
            on_record_batch([record_to_payload(record) for record in batch])

        output, streamed = self._runner.process_logs_stream(
            raw_logs,
            on_record_batch=_emit,
            progress_callback=progress_callback,
            batch_size=batch_size,
        )
        return streamed_payload(output, streamed_count=streamed)
