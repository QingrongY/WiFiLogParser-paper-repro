from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class LLMExtractorSettings:
    """Configuration for the standalone WiFiLogParser experiments."""

    primary_model: str
    fallback_models: Sequence[str]
    batch_size: int = 8
    chunk_size: int = 50_000
    base_url: str | None = None
    api_key: str | None = None
    disable_fallback: bool = False
    min_cluster_size: int = 2
    request_timeout_seconds: float = 60.0
    max_retries: int = 1

    @property
    def effective_fallback_models(self) -> List[str]:
        if self.disable_fallback:
            return []
        if not self.fallback_models:
            return []
        return [model for model in self.fallback_models if model]


def _read_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _read_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return int(default)


def _read_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        return float(default)


def _read_models(name: str) -> Sequence[str]:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def load_llm_settings() -> LLMExtractorSettings:
    primary_model = str(os.getenv("LLM_PRIMARY_MODEL", "gemini-3.1-flash-lite")).strip()
    base_url = os.getenv("LLM_BASE_URL") or None
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or None

    fallback_models = _read_models("LLM_FALLBACK_MODELS")
    batch_size = max(1, _read_int("LLM_BATCH_SIZE", 8))
    chunk_size = max(1, _read_int("LLM_CHUNK_SIZE", 50_000))
    min_cluster_size = max(1, _read_int("LLM_MIN_CLUSTER_SIZE", 2))
    disable_fallback = _read_bool("LLM_DISABLE_FALLBACK", False)
    request_timeout_seconds = max(1.0, _read_float("LLM_REQUEST_TIMEOUT_SECONDS", 60.0))
    max_retries = max(0, _read_int("LLM_MAX_RETRIES", 1))

    return LLMExtractorSettings(
        primary_model=primary_model,
        fallback_models=fallback_models,
        batch_size=batch_size,
        chunk_size=chunk_size,
        base_url=base_url,
        api_key=api_key,
        disable_fallback=disable_fallback,
        min_cluster_size=min_cluster_size,
        request_timeout_seconds=request_timeout_seconds,
        max_retries=max_retries,
    )
