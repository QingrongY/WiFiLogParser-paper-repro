from __future__ import annotations

import json
import re


class PostProcessor:
    """Extract regex/connect flag JSON from LLM output."""

    def process_output(self, raw_output: str | None) -> tuple[str | None, int]:
        if not raw_output:
            return None, 0
        regex_pattern, connect_flag = self._extract_json_fields(raw_output)
        if not regex_pattern or not regex_pattern.strip():
            return None, connect_flag
        return regex_pattern, connect_flag

    def _extract_json_fields(self, response: str) -> tuple[str | None, int]:
        patterns = [
            r"```json\s*(.*?)\s*```",
            r"\{[^{}]*\"regex\"[^{}]*\}",
            r"\{.*?\"regex\".*?\}",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL)
            if not match:
                continue
            try:
                json_str = match.group(1) if "```" in pattern else match.group(0)
                data = json.loads(json_str)
                return data.get("regex"), int(data.get("connect_flag", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return self._direct_extract(response)

    def _direct_extract(self, response: str) -> tuple[str | None, int]:
        regex_pattern = None
        start = response.find('"regex"')
        if start != -1:
            colon = response.find(":", start)
            quote_start = response.find('"', colon + 1)
            if quote_start != -1:
                quote_start += 1
                quote_end = self._find_matching_quote(response, quote_start)
                if quote_end != -1:
                    raw_regex = response[quote_start:quote_end]
                    try:
                        regex_pattern = json.loads(f'"{raw_regex}"')
                    except json.JSONDecodeError:
                        regex_pattern = raw_regex
        flag_match = re.search(r'"connect_flag"\s*:\s*(-?\d+)', response)
        connect_flag = int(flag_match.group(1)) if flag_match else 0
        return regex_pattern, connect_flag

    def _find_matching_quote(self, text: str, start_pos: int) -> int:
        pos = start_pos
        while pos < len(text):
            if text[pos] == "\\":
                pos += 2
                continue
            if text[pos] == '"':
                return pos
            pos += 1
        return -1
