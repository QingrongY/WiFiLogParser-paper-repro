from __future__ import annotations

import time
from typing import Iterable

from ..common.api_clients import APIClient
from ..common.config import LLMExtractorSettings
from ..common.utils import count_message_tokens


TEMPLATE_MAX_TOKENS = 8192
TEMPLATE_REQUEST_TIMEOUT_SECONDS = 60.0

CONTENT_PARSING_PROMPT = r"""Task
Classify the client-AP event and generate a regex that matches the provided log lines.

Rules
- Set `event_label` to 1 for connection or connection attempts, including authentication, association, IP detection, and reconnection.
- Set `event_label` to -1 for disconnection events.
- Set `event_label` to 0 for all other events, including DHCP, failed connections, errors, and probe requests.
- Replace variable content with `.*?` while preserving fixed structure and delimiters.
- The regex must match the entire log line.
- Use named capture groups only for `client_ip`, `ap_ip`, `client_mac`, `ap_mac`, `client_name`, `ap_name`, and `ssid`.
- Extract only the client and AP involved in the event, not the logging source.
- Use numbered suffixes when the same field appears more than once.
- For JSON content, capture only relevant AP and client values with the appropriate named groups, and use `.*?` to skip unrelated key-value pairs, e.g., `\{.*?"ap":"(?P<ap_name>.*?)".*?"client":"(?P<client_mac>.*?)".*?\}`.

Input
Log lines: {sample_logs}

Output
Return only:
{"regex": "...", "event_label": "<integer: 1, -1, or 0>"}"""


class LLMParser:
    """LLM client wrapper with telemetry tracking."""

    def __init__(self, model: str, settings: LLMExtractorSettings, batch_size: int = 8,
                 successful_examples: int = 3):
        self.model = model
        self.batch_size = batch_size
        self.successful_examples = successful_examples
        self._settings = settings
        self.api_client = APIClient(
            base_url=settings.base_url,
            api_key=settings.api_key or "",
            model=model,
            timeout_seconds=TEMPLATE_REQUEST_TIMEOUT_SECONDS,
            max_retries=settings.max_retries,
            reasoning_effort=settings.reasoning_effort,
        )
        self._token_count = 0
        self._call_count = 0
        self._total_time = 0.0

    def parse_batch(self, batch_logs: Iterable[str], examples_text: str = "") -> str | None:
        logs = list(batch_logs)
        if not logs:
            return None
        logs_text = "\n".join(f"Log[{i + 1}]: `{log}`" for i, log in enumerate(logs))
        instruction = self._build_instruction(examples_text)
        if "{sample_logs}" in instruction:
            full_input = instruction.replace("{sample_logs}", logs_text)
        else:
            full_input = f"{instruction}\n\nLogs to analyze:\n{logs_text}"
        messages = [{"role": "user", "content": full_input}]

        start = time.monotonic()
        response = self.api_client.chat(messages, temperature=0.0, max_tokens=TEMPLATE_MAX_TOKENS)
        elapsed = time.monotonic() - start

        self._call_count += 1
        if response.usage and response.usage.get("total_tokens"):
            self._token_count += response.usage["total_tokens"]
        else:
            self._token_count += count_message_tokens(messages, "gpt-4o-mini")
        self._total_time += elapsed
        return response.content

    def _build_instruction(self, examples_text: str) -> str:
        del examples_text
        return CONTENT_PARSING_PROMPT

    def get_total_time(self) -> float:
        return self._total_time

    def get_total_tokens(self) -> int:
        return self._token_count

    def get_call_count(self) -> int:
        return self._call_count
