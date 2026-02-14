from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping, Sequence

import pandas as pd
import tiktoken


def count_message_tokens(messages: Sequence[Mapping[str, str]], model_name: str = "gpt-3.5-turbo") -> int:
    """Estimate token usage for OpenAI-compatible messages."""
    if model_name == "gpt-4o-mini":
        encoder = tiktoken.encoding_for_model("gpt-4o-mini")
    else:
        encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

    token_count = 0
    for message in messages:
        role_tokens = encoder.encode(message.get("role", ""))
        content_tokens = encoder.encode(message.get("content", ""))
        token_count += len(role_tokens) + len(content_tokens) + 4
    return token_count


def try_parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except Exception:
        try:
            parsed = pd.to_datetime(value, errors="coerce", utc=True)
        except Exception:
            return None
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()


def merge_datetime_parts(parts: Mapping[str, str | None]) -> datetime | None:
    """Combine partial datetime components into a timestamp if possible."""
    year = parts.get("year")
    month = parts.get("month")
    date = parts.get("date") or parts.get("day")
    time_part = parts.get("time")
    minute = parts.get("minute")
    second = parts.get("second")
    ampm = parts.get("ampm")
    timestamp = parts.get("timestamp")

    if timestamp:
        return try_parse_timestamp(timestamp)

    if time_part and minute and not second:
        time_part = f"{time_part}:{minute}"
        if parts.get("second"):
            time_part = f"{time_part}:{parts['second']}"
    elif time_part and second and ":" not in time_part:
        time_part = f"{time_part}:{second}"

    components = []
    if year:
        components.append(str(year))
    if month:
        components.append(str(month))
    if date:
        components.append(str(date))

    datetime_str = "-".join(components) if components else ""
    if time_part:
        datetime_str = f"{datetime_str} {time_part}".strip()
    if ampm:
        datetime_str = f"{datetime_str} {ampm}".strip()
    if not datetime_str:
        return None
    return try_parse_timestamp(datetime_str)


def to_naive_datetime(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def pick_first(*options: str | None) -> str | None:
    for value in options:
        if value:
            stripped = str(value).strip()
            if stripped:
                return stripped
    return None


def normalise_mac(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    cleaned = value.replace("-", ":").replace(".", "").replace(" ", "")
    cleaned = cleaned.lower()
    if len(cleaned) == 12 and ":" not in cleaned:
        return ":".join(cleaned[i:i + 2] for i in range(0, 12, 2))
    return cleaned
