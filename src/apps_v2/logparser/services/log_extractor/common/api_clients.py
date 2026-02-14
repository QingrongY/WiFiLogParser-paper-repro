from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping, Optional

from openai import OpenAI

from apps_v2.llm.budget import commit_usage, release_reservation, reserve_tokens_for_call
from apps_v2.llm.context import get_llm_context
from apps_v2.llm.dates import usage_day

from .utils import count_message_tokens


@dataclass
class ChatResponse:
    content: str
    usage: Mapping[str, int] | None = None


class APIClient:
    """Thin wrapper around OpenAI-compatible chat endpoints."""

    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str,
        model: str,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ):
        if not api_key:
            raise ValueError("LLM API key must be configured before running the pipeline.")
        self.model = model

        timeout_value = None
        if timeout_seconds is not None:
            try:
                timeout_value = max(1.0, float(timeout_seconds))
            except (TypeError, ValueError):
                timeout_value = None

        retry_value = None
        if max_retries is not None:
            try:
                retry_value = max(0, int(max_retries))
            except (TypeError, ValueError):
                retry_value = None

        self._timeout_seconds = timeout_value

        client_kwargs = {
            'api_key': api_key,
            'base_url': base_url or None,
        }
        if timeout_value is not None:
            client_kwargs['timeout'] = timeout_value
        if retry_value is not None:
            client_kwargs['max_retries'] = retry_value

        try:
            self._client = OpenAI(**client_kwargs)
        except TypeError:
            self._client = OpenAI(api_key=api_key, base_url=base_url or None)

    def chat(self, messages: Iterable[Mapping[str, str]], *, temperature: float = 0.0,
             max_tokens: int = 1024) -> ChatResponse:
        messages_list = list(messages)

        ctx = get_llm_context()
        reservation = None
        if ctx and ctx.session_id:
            day = usage_day()
            prompt_estimate = 0
            if ctx.daily_quota_tokens is not None or ctx.session_quota_tokens is not None:
                prompt_estimate = count_message_tokens(messages_list, "gpt-4o-mini")
            reservation = reserve_tokens_for_call(
                session_id=ctx.session_id,
                user_id=ctx.user_id,
                day=day,
                prompt_tokens_estimate=prompt_estimate,
                max_tokens=max_tokens,
                daily_quota_tokens=ctx.daily_quota_tokens,
                session_quota_tokens=ctx.session_quota_tokens,
            )

        try:
            create = self._client.chat.completions.create
            if self._timeout_seconds is not None:
                try:
                    response = create(
                        model=self.model,
                        messages=messages_list,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                        timeout=self._timeout_seconds,
                    )
                except TypeError:
                    response = create(
                        model=self.model,
                        messages=messages_list,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                    )
            else:
                response = create(
                    model=self.model,
                    messages=messages_list,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                )
        except Exception:
            if reservation is not None:
                release_reservation(reservation)
            raise

        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        usage: Optional[MutableMapping[str, int]] = None
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }

        if reservation is not None:
            total_tokens = int((usage or {}).get("total_tokens") or 0)
            if total_tokens <= 0:
                # Fallback: if the provider doesn't return usage, count the
                # reservation (conservative) so quotas remain safe.
                total_tokens = max(reservation.reserved_session_tokens, reservation.reserved_daily_tokens, 0)
            commit_usage(reservation=reservation, actual_total_tokens=total_tokens)

        return ChatResponse(content=content, usage=usage)
