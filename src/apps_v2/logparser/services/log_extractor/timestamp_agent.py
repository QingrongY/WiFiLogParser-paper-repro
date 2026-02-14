from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence, cast

import numpy as np
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .common.api_clients import APIClient
from .common.config import LLMExtractorSettings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimestampRule:
    pattern: re.Pattern[str]
    date_strftime: str
    time_strftime: str
    has_year: bool
    base_year: int = 1970

    def search(self, text: str) -> re.Match[str] | None:
        return self.pattern.search(text)

    def parse_match(self, match: re.Match[str]) -> datetime:
        raw_date = match.group("date")
        raw_time = match.group("time")
        if raw_date is None or raw_time is None:
            raise ValueError("timestamp groups 'date' and 'time' missing")

        date_value = _clean_timestamp_value(raw_date, self.date_strftime)
        time_value = _clean_timestamp_value(raw_time, self.time_strftime)
        try:
            date_dt = datetime.strptime(date_value, self.date_strftime)
        except ValueError:
            cleaned = date_value.rstrip(" \t,;|")
            date_dt = datetime.strptime(cleaned, self.date_strftime)

        try:
            time_dt = datetime.strptime(time_value, self.time_strftime)
        except ValueError:
            cleaned = time_value.rstrip(" \t,;|")
            time_dt = datetime.strptime(cleaned, self.time_strftime)

        tzinfo = time_dt.tzinfo or date_dt.tzinfo
        dt = datetime.combine(date_dt.date(), time_dt.timetz())
        if tzinfo is not None and dt.tzinfo is None:
            dt = dt.replace(tzinfo=tzinfo)
        if not self.has_year:
            dt = dt.replace(year=self.base_year)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt


@dataclass(frozen=True)
class TimestampValidationReport:
    total: int
    success: int
    success_rate: float
    no_match: int
    parse_failed: int
    ts_too_far: int
    empty_content: int
    examples: dict[str, list[str]]


class TimestampAgent:
    """Infer a timestamp extraction rule and split logs into (timestamp, content)."""

    DIVERSE_SAMPLE_SIZE = 10
    CANDIDATE_POOL_SIZE = 2000
    SUCCESS_THRESHOLD = 0.90
    MAX_FIX_ATTEMPTS = 3
    MAX_TIMESTAMP_OFFSET = 200

    def __init__(self, settings: LLMExtractorSettings):
        self.settings = settings
        self._client: APIClient | None = None
        if settings.api_key:
            try:
                self._client = APIClient(
                    base_url=settings.base_url,
                    api_key=settings.api_key,
                    model=settings.primary_model,
                    timeout_seconds=settings.request_timeout_seconds,
                    max_retries=settings.max_retries,
                )
            except Exception:  # pragma: no cover - defensive
                logger.exception("TimestampAgent: failed to initialise API client")
        self.calls = 0
        self.total_tokens = 0
        self.total_time = 0.0
        self.last_report: TimestampValidationReport | None = None

    def infer_rule(self, raw_logs: Sequence[str]) -> TimestampRule:
        logs = [_normalise_line(line) for line in raw_logs]
        logs = [line for line in logs if line]
        if not logs:
            raise ValueError("No log lines available for timestamp inference")

        if self._client is None:
            raise ValueError("TimestampAgent: API client is not configured")

        diverse_samples = _pick_diverse_samples(
            logs,
            sample_size=self.DIVERSE_SAMPLE_SIZE,
            candidate_pool_size=self.CANDIDATE_POOL_SIZE,
        )
        if not diverse_samples:
            raise ValueError("TimestampAgent: unable to select samples")

        attempt = 0
        rule: TimestampRule | None = None
        report: TimestampValidationReport | None = None

        while True:
            if attempt == 0:
                candidate = self._infer_rule_with_llm(diverse_samples)
            else:
                if rule is None or report is None:
                    raise ValueError("TimestampAgent: internal error during repair")
                candidate = self._repair_rule_with_llm(diverse_samples, rule, report)

            report = self._validate_rule(candidate, logs)
            self.last_report = report
            logger.info(
                "TimestampAgent validation attempt=%s success_rate=%.3f total=%s",
                attempt + 1,
                report.success_rate,
                report.total,
            )
            if report.success_rate >= self.SUCCESS_THRESHOLD:
                return candidate

            rule = candidate
            attempt += 1
            if attempt > self.MAX_FIX_ATTEMPTS:
                raise ValueError(
                    "TimestampAgent: inferred rule did not reach required success rate "
                    f"{self.SUCCESS_THRESHOLD:.0%}. Last={report.success_rate:.1%} "
                    f"(success={report.success} total={report.total})."
                )

    def split(self, line: str, rule: TimestampRule) -> tuple[datetime | None, str | None]:
        cleaned = _normalise_line(line)
        if not cleaned:
            return None, None
        match = rule.search(cleaned)
        if not match:
            return None, None
        start_pos = min(match.start("date"), match.start("time"))
        if start_pos > self.MAX_TIMESTAMP_OFFSET:
            return None, None
        try:
            dt = rule.parse_match(match)
        except Exception:
            return None, None
        end_pos = max(match.end("date"), match.end("time"))
        content = cleaned[end_pos:]
        content = content.lstrip(" \t,|-:\"")
        return dt, content

    def _infer_rule_with_llm(self, sample_logs: Sequence[str]) -> TimestampRule:
        sample_text = "\n".join(f"Line {idx + 1}: {line}" for idx, line in enumerate(sample_logs))
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a log timestamp extraction agent. Given sample log lines, infer how to extract the "
                    "timestamp from the header. Respond with JSON only. "
                    "Output JSON fields: regex (MUST include named groups date and time), date_format (Python "
                    "datetime.strptime format for date), time_format (Python datetime.strptime format for time), "
                    "has_year (true/false). "
                    "The date/time groups must capture ONLY the date/time string values (no brackets, quotes, "
                    "or trailing separators). "
                    "Prefer regex patterns that tolerate variable spacing using \\s+."
                ),
            },
            {
                "role": "user",
                "content": f"Sample logs:\n{sample_text}\nRespond with JSON only.",
            },
        ]
        data = self._call_llm(messages)
        return _build_rule_from_json(data)

    def _repair_rule_with_llm(
        self,
        sample_logs: Sequence[str],
        rule: TimestampRule,
        report: TimestampValidationReport,
    ) -> TimestampRule:
        success_examples = (report.examples.get("success") or [])[:5]
        failure_blocks = []
        for key in ("no_match", "parse_failed", "ts_too_far", "empty_content"):
            examples = report.examples.get(key) or []
            if not examples:
                continue
            failure_blocks.append(f"{key}:\n" + "\n".join(f"- {line}" for line in examples[:5]))
        failures_text = "\n\n".join(failure_blocks) if failure_blocks else "(no examples)"

        sample_text = "\n".join(f"- {line}" for line in sample_logs)
        ok_text = "\n".join(f"- {line}" for line in success_examples)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a log timestamp extraction agent. Your job is to FIX the regex/format so that it "
                    "extracts the header timestamp across diverse log lines. Respond with JSON only. "
                    "Output fields: regex (named groups date and time), date_format, time_format, has_year."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Current rule:\n"
                    f"regex: {rule.pattern.pattern}\n"
                    f"date_format: {rule.date_strftime}\n"
                    f"time_format: {rule.time_strftime}\n"
                    f"has_year: {str(rule.has_year).lower()}\n\n"
                    f"Validation: success_rate={report.success_rate:.3f} success={report.success} total={report.total}\n\n"
                    "Diverse sample set:\n"
                    f"{sample_text}\n\n"
                    "Examples that should parse correctly:\n"
                    f"{ok_text or '(none)'}\n\n"
                    "Failure examples (grouped):\n"
                    f"{failures_text}\n\n"
                    "Return JSON only: {\"regex\": ..., \"date_format\": ..., \"time_format\": ..., \"has_year\": true/false}"
                ),
            },
        ]
        data = self._call_llm(messages)
        return _build_rule_from_json(data)

    def _call_llm(self, messages: Sequence[dict]) -> dict:
        start = time.monotonic()
        try:
            response = self._client.chat(messages, temperature=0.0, max_tokens=256)  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover - network failure
            from apps_v2.llm.budget import BudgetExceeded

            if isinstance(exc, BudgetExceeded):
                raise
            raise ValueError(f"TimestampAgent: LLM call failed: {exc}") from exc
        elapsed = time.monotonic() - start
        self.calls += 1
        self.total_time += elapsed
        if response.usage and response.usage.get("total_tokens") is not None:
            self.total_tokens += int(response.usage.get("total_tokens") or 0)
        data = _extract_json(response.content)
        if not data:
            raise ValueError("TimestampAgent: failed to parse JSON response")
        return data

    def _validate_rule(self, rule: TimestampRule, logs: Sequence[str]) -> TimestampValidationReport:
        examples: dict[str, list[str]] = {"success": [], "no_match": [], "parse_failed": [], "ts_too_far": [], "empty_content": []}
        no_match = 0
        parse_failed = 0
        ts_too_far = 0
        empty_content = 0
        success = 0

        for line in logs:
            match = rule.search(line)
            if not match:
                no_match += 1
                if len(examples["no_match"]) < 5:
                    examples["no_match"].append(line)
                continue
            start_pos = min(match.start("date"), match.start("time"))
            if start_pos > self.MAX_TIMESTAMP_OFFSET:
                ts_too_far += 1
                if len(examples["ts_too_far"]) < 5:
                    examples["ts_too_far"].append(line)
                continue
            try:
                dt = rule.parse_match(match)
            except Exception:
                parse_failed += 1
                if len(examples["parse_failed"]) < 5:
                    examples["parse_failed"].append(line)
                continue
            _ = dt
            end_pos = max(match.end("date"), match.end("time"))
            content = line[end_pos:].lstrip(" \t,|-:\"")
            if not content:
                empty_content += 1
                if len(examples["empty_content"]) < 5:
                    examples["empty_content"].append(line)
                continue
            success += 1
            if len(examples["success"]) < 5:
                examples["success"].append(line)

        total = len(logs)
        success_rate = (success / total) if total else 0.0
        return TimestampValidationReport(
            total=total,
            success=success,
            success_rate=success_rate,
            no_match=no_match,
            parse_failed=parse_failed,
            ts_too_far=ts_too_far,
            empty_content=empty_content,
            examples=examples,
        )


def _normalise_line(line: str) -> str:
    if line is None:
        return ""
    return str(line).replace("\t", " ").strip()


def _pick_diverse_samples(
    logs: Sequence[str],
    *,
    sample_size: int,
    candidate_pool_size: int,
) -> list[str]:
    if not logs:
        return []
    if len(logs) <= candidate_pool_size:
        candidates = list(dict.fromkeys(logs))
    else:
        step = max(1, len(logs) // candidate_pool_size)
        sampled = [logs[i] for i in range(0, len(logs), step)][:candidate_pool_size]
        candidates = list(dict.fromkeys(sampled))
    if len(candidates) <= sample_size:
        return candidates

    vectorizer = TfidfVectorizer()
    matrix = cast(Any, vectorizer.fit_transform(candidates)).toarray()
    similarity = cosine_similarity(matrix)
    indices = _dpp_sample(similarity, sample_size)
    return [candidates[i] for i in indices]


def _dpp_sample(similarity_matrix: np.ndarray, k: int) -> list[int]:
    n = similarity_matrix.shape[0]
    selected: set[int] = set()
    for _ in range(k):
        best_index = -1
        best_prob = -1.0
        for i in range(n):
            if i in selected:
                continue
            indices = list(selected) + [i]
            submatrix = similarity_matrix[np.ix_(indices, indices)]
            det = float(np.linalg.det(submatrix))
            prob = det / (1.0 + det)
            if prob > best_prob:
                best_prob = prob
                best_index = i
        if best_index == -1:
            remaining = [i for i in range(n) if i not in selected]
            if not remaining:
                break
            best_index = remaining[0]
        selected.add(best_index)
    return list(selected)


def _build_rule_from_json(data: dict) -> TimestampRule:
    regex_str = data.get("regex")
    date_fmt = data.get("date_format")
    time_fmt = data.get("time_format")
    has_year = bool(data.get("has_year"))
    if not (regex_str and date_fmt and time_fmt):
        raise ValueError("TimestampAgent: missing regex/date_format/time_format from model response")
    try:
        pattern = re.compile(str(regex_str))
    except Exception as exc:
        raise ValueError(f"TimestampAgent: failed to compile regex: {exc}") from exc
    if "date" not in pattern.groupindex or "time" not in pattern.groupindex:
        raise ValueError("TimestampAgent: regex must include named groups 'date' and 'time'")
    return TimestampRule(pattern=pattern, date_strftime=str(date_fmt), time_strftime=str(time_fmt), has_year=has_year)


def _clean_timestamp_value(value: str, fmt: str) -> str:
    cleaned = str(value).strip()
    cleaned = cleaned.strip('"')
    cleaned = cleaned.strip("[](){}<>")
    cleaned = cleaned.rstrip(",;|")
    if "%z" in fmt and cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+0000"
    return cleaned


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
