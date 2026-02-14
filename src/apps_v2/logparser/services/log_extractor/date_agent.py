from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Sequence

from .common.api_clients import APIClient
from .common.config import LLMExtractorSettings

logger = logging.getLogger(__name__)


@dataclass
class DateRule:
    pattern: re.Pattern[str]
    strftime: str
    has_year: bool
    base_year: int = 1970

    def parse(self, text: str) -> datetime:
        match = self.pattern.search(text)
        if not match:
            raise ValueError("timestamp pattern not found in log line")
        value = match.group('ts') if 'ts' in match.groupdict() else match.group(0)
        if '%z' in self.strftime and value.endswith('Z'):
            value = value[:-1] + '+0000'
        dt = datetime.strptime(value, self.strftime)
        if not self.has_year:
            dt = dt.replace(year=self.base_year)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt


class DateRuleAgent:
    SAMPLE_SIZE = 50

    def __init__(self, settings: LLMExtractorSettings):
        self.settings = settings
        self.api_client = None
        if settings.api_key:
            try:
                self.api_client = APIClient(
                    base_url=settings.base_url,
                    api_key=settings.api_key,
                    model=settings.primary_model,
                    timeout_seconds=settings.request_timeout_seconds,
                    max_retries=settings.max_retries,
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("Failed to initialise API client for DateRuleAgent")

    def build_rule(self, sample_logs: Sequence[str]) -> DateRule:
        logs = [line for line in sample_logs if line.strip()][: self.SAMPLE_SIZE]
        if not logs:
            raise ValueError("No log lines available for date rule inference")
        rule = self._infer_rule_with_llm(logs)
        if rule:
            logger.info("DateRuleAgent inferred rule format=%s has_year=%s from sample count=%s", rule.strftime, rule.has_year, len(logs))
            return rule
        raise ValueError("Unable to infer timestamp pattern from provided logs")

    def _infer_rule_with_llm(self, logs: Sequence[str]) -> DateRule | None:
        sample_text = "\n".join(logs[:10])
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a log timestamp extraction agent. Given sample log lines, "
                    "produce JSON describing how to extract the timestamp. "
                    "Use fields: regex (with named group ts), format (Python strptime), has_year (true/false)."
                ),
            },
            {
                "role": "user",
                "content": f"Sample logs:\n{sample_text}\nRespond with JSON.",
            },
        ]
        try:
            response = self.api_client.chat(messages)
        except Exception as exc:  # pragma: no cover - network failure
            from apps_v2.llm.budget import BudgetExceeded

            if isinstance(exc, BudgetExceeded):
                raise
            logger.exception("DateRuleAgent LLM call failed: %s", exc)
            return None
        content = response.content
        data = self._extract_json(content)
        if not data:
            return None
        regex = data.get('regex')
        strptime_fmt = data.get('format') or data.get('strftime')
        has_year = bool(data.get('has_year'))
        if not (regex and strptime_fmt):
            return None
        try:
            pattern = re.compile(regex)
            rule = DateRule(pattern=pattern, strftime=strptime_fmt, has_year=has_year)
            self._validate_rule(rule, logs)
            return rule
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to use LLM-generated date rule: %s", exc)
            return None

    def _validate_rule(self, rule: DateRule, logs: Sequence[str]) -> None:
        successes = 0
        for line in logs[:20]:
            try:
                rule.parse(line)
                successes += 1
            except Exception:
                continue
        if successes == 0:
            raise ValueError("Generated rule could not parse any samples")

    @staticmethod
    def _extract_json(content: str) -> dict | None:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None
        return None
