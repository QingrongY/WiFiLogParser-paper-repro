from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Protocol, Sequence

from apps_v2.logparser.services.log_extractor.runner import LogExtractionRunner
from apps_v2.logparser.services.payload_adapter import runner_output_to_payload
from apps_v2.logparser.services.wifi_log_parser.engine import WiFiLogParserEngine


ProgressCallback = Callable[[dict], None]


class ExtractionEngine(Protocol):
    """Common interface for pluggable log parsing algorithms."""

    name: str

    def process_file(
        self,
        file_path: Path,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> Dict[str, object]:
        ...

    def process_logs(
        self,
        raw_logs: Sequence[str],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> Dict[str, object]:
        ...


class LogExtractorEngine:
    """Baseline engine: skeleton clustering + LLM regex + fallback repair."""

    name = "log_extractor"

    def __init__(self):
        self._runner = LogExtractionRunner()

    def process_file(self, file_path: Path, *, progress_callback: ProgressCallback | None = None) -> Dict[str, object]:
        output = self._runner.process_file(file_path, progress_callback=progress_callback)
        return runner_output_to_payload(output)

    def process_logs(self, raw_logs: Sequence[str], *, progress_callback: ProgressCallback | None = None) -> Dict[str, object]:
        output = self._runner.process_logs(raw_logs, progress_callback=progress_callback)
        return runner_output_to_payload(output)


ENGINE_REGISTRY: dict[str, type[ExtractionEngine]] = {
    LogExtractorEngine.name: LogExtractorEngine,
    WiFiLogParserEngine.name: WiFiLogParserEngine,
}

DEFAULT_ENGINE_NAME = WiFiLogParserEngine.name


def get_engine_name() -> str:
    """Return the default engine name."""

    return DEFAULT_ENGINE_NAME


def list_supported_engines() -> tuple[str, ...]:
    return tuple(sorted(ENGINE_REGISTRY.keys()))


def get_engine(engine_name: str | None = None) -> ExtractionEngine:
    name = (engine_name or get_engine_name()).strip().lower()
    if not name:
        name = get_engine_name()

    engine_cls = ENGINE_REGISTRY.get(name)
    if engine_cls is None:
        supported = ", ".join(list_supported_engines())
        raise ValueError(f"Unsupported LLM parser engine: {name}. Supported engines: {supported}.")
    return engine_cls()
