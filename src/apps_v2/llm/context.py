from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMContext:
    session_id: str | None = None
    user_id: str | None = None
    daily_quota_tokens: int | None = None
    session_quota_tokens: int | None = None


def get_llm_context() -> LLMContext | None:
    return None
