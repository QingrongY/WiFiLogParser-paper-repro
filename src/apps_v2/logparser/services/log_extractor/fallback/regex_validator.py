from __future__ import annotations

import logging
import re

import regex


logger = logging.getLogger(__name__)


class RegexValidator:
    """Rich validator that mirrors the original LogExtractor behaviour."""

    def validate_regex(self, regex_pattern: str | None, logs: list[str]) -> bool:
        valid, _, failure_info, _ = self._validate_regex_on_logs(regex_pattern, logs)
        if not valid and failure_info:
            logger.debug("Regex validation failed: %s", failure_info)
        return valid

    def diagnose(self, regex_pattern: str | None, logs: list[str]) -> tuple[bool, str | None, str | None, str | None]:
        """Return detailed diagnosis (valid, failed_log, failure_info, error_type)."""
        return self._validate_regex_on_logs(regex_pattern, logs)

    def _validate_regex_on_logs(
        self,
        regex_pattern: str | None,
        logs: list[str],
    ) -> tuple[bool, str | None, str | None, str | None]:
        if not regex_pattern or not logs:
            return False, None, "Empty regex or logs", "input_error"

        for log_line in logs:
            try:
                match = regex.match(regex_pattern, log_line)
                if not match:
                    failure_info = self._find_failure_point_by_fields(regex_pattern, log_line)
                    return False, log_line, failure_info, "match_fail"
            except Exception as exc:
                failure_info = self._find_failure_point_by_fields(regex_pattern, log_line)
                return False, log_line, f"{failure_info}\nError: {exc}", "syntax_error"
        return True, None, None, None

    def _find_failure_point_by_fields(self, regex_pattern: str, log: str) -> str:
        boundaries = [0]
        i = 0
        paren_depth = 0
        while i < len(regex_pattern):
            if regex_pattern.startswith("(?P<", i):
                paren_depth += 1
                i += 4
                while i < len(regex_pattern) and regex_pattern[i] != ">":
                    i += 1
                if i < len(regex_pattern):
                    i += 1
                continue
            if regex_pattern[i] == "(":
                paren_depth += 1
                i += 1
                continue
            if regex_pattern[i] == ")":
                paren_depth -= 1
                if paren_depth == 0:
                    boundaries.append(i + 1)
                i += 1
                continue
            if regex_pattern[i] == "\\":
                if i < len(regex_pattern) - 1 and regex_pattern[i + 1] == "s" and paren_depth == 0:
                    boundaries.append(i)
                i += 2
                continue
            if regex_pattern[i] == "[":
                i += 1
                while i < len(regex_pattern) and regex_pattern[i] != "]":
                    if regex_pattern[i] == "\\":
                        i += 2
                    else:
                        i += 1
                i += 1
                continue
            if regex_pattern[i] == " " and paren_depth == 0:
                boundaries.append(i)
                i += 1
                continue
            if regex_pattern[i] == "," and paren_depth == 0 and not self._in_quantifier(regex_pattern, i):
                boundaries.append(i)
                i += 1
                continue
            i += 1
        return self._test_by_boundaries(regex_pattern, log, boundaries, "segment")

    def _in_quantifier(self, regex_pattern: str, pos: int) -> bool:
        open_brace = -1
        close_brace = -1
        for i in range(pos - 1, -1, -1):
            if regex_pattern[i] == "}":
                close_brace = i
                break
            if regex_pattern[i] == "{":
                open_brace = i
                break
        return open_brace != -1 and close_brace == -1

    def _test_by_boundaries(
        self,
        regex_pattern: str,
        log: str,
        boundaries: list[int],
        unit_type: str,
        min_segment_length: int = 30,
    ) -> str:
        last_successful_end = 0
        last_regex_end = 0

        for idx in range(1, len(boundaries)):
            partial_regex = regex_pattern[:boundaries[idx]]
            try:
                regex.compile(partial_regex)
            except Exception as exc:
                prev = boundaries[idx - 1]
                failed = regex_pattern[prev: max(prev + min_segment_length, boundaries[idx])]
                return f"Syntax error in {unit_type}:\nRegex: '{failed.strip()}'\nError: {exc}"

            if not self._is_testable_regex(partial_regex):
                continue
            try:
                match = regex.match(partial_regex, log)
                if match:
                    last_successful_end = match.end()
                    last_regex_end = boundaries[idx]
                    continue
                failed_start = last_regex_end
                failed = regex_pattern[failed_start: max(failed_start + min_segment_length, boundaries[idx])]
                remaining = log[last_successful_end:].lstrip()
                context = remaining[:50] + "..." if len(remaining) > 50 else remaining
                return f"Match failed at {unit_type}:\nRegex: '{failed.strip()}'\nLog:   '{context}'"
            except Exception:
                continue

        failed_start = last_regex_end
        failed = regex_pattern[failed_start:]
        if len(failed) < min_segment_length:
            failed = regex_pattern[failed_start: failed_start + min_segment_length]
        try:
            regex.compile(regex_pattern)
        except Exception as exc:
            return f"Syntax error in final {unit_type}:\nRegex: '{failed.strip()}'\nError: {exc}"
        remaining = log[last_successful_end:].lstrip()
        context = remaining[:50] + "..." if len(remaining) > 50 else remaining
        return f"Match failed at final {unit_type}:\nRegex: '{failed.strip()}'\nLog:   '{context}'"

    def _is_testable_regex(self, partial_regex: str) -> bool:
        if not partial_regex:
            return False
        for open_sym, close_sym in (("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")):
            if partial_regex.count(open_sym) > partial_regex.count(close_sym):
                return False
        if partial_regex[-1] in {"{", "(", "[", "<", "\\"}:
            return False
        regex.compile(partial_regex)
        return True

    def extract_variables_with_regex(self, log: str, regex_pattern: str | None) -> dict | None:
        if not regex_pattern or not log:
            return None
        cleaned_log = re.sub(r"\s+", " ", log.strip())
        try:
            match = re.compile(regex_pattern).search(cleaned_log)
        except Exception:
            return None
        return match.groupdict() if match else None
