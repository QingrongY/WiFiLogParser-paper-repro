from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class TemplateParse:
    template: str | None
    connect_flag: int


class TemplatePostProcessor:
    """Extract template/connect_flag JSON from LLM output."""

    def process_output(self, raw_output: str | None) -> TemplateParse:
        if not raw_output:
            return TemplateParse(template=None, connect_flag=0)

        data = _extract_first_json_object(raw_output)
        if not data:
            return TemplateParse(template=None, connect_flag=0)

        template = data.get("template")
        if template is not None:
            template = str(template)
        try:
            connect_flag = int(data.get("connect_flag", 0))
        except (TypeError, ValueError):
            connect_flag = 0
        if not template or not str(template).strip():
            return TemplateParse(template=None, connect_flag=connect_flag)
        return TemplateParse(template=str(template), connect_flag=connect_flag)


def _extract_first_json_object(text: str) -> dict | None:
    if not text:
        return None

    raw = str(text).strip()

    # Direct parse first.
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # Handle fenced blocks (```json ... ``` or ``` ... ```).
    fence = "```"
    if fence in raw:
        parts = raw.split(fence)
        for chunk in parts:
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.lower().startswith("json"):
                chunk = chunk[4:].strip()
            if not chunk:
                continue
            try:
                obj = json.loads(chunk)
                return obj if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                continue

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(raw[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None
