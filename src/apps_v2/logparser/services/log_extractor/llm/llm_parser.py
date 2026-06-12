from __future__ import annotations

import time
from typing import Iterable

from ..common.api_clients import APIClient
from ..common.config import LLMExtractorSettings
from ..common.utils import count_message_tokens


TEMPLATE_MAX_TOKENS = 2048
TEMPLATE_REQUEST_TIMEOUT_SECONDS = 10.0


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
        )
        self._token_count = 0
        self._call_count = 0
        self._total_time = 0.0

    def parse_batch(self, batch_logs: Iterable[str], examples_text: str) -> str | None:
        logs = list(batch_logs)
        if not logs:
            return None
        instruction = self._build_instruction(examples_text)
        logs_text = "\n".join(f"Log[{i + 1}]: `{log}`" for i, log in enumerate(logs))
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
        return f"""
You will be given WiFi logs. Your task is to classify the event type (connect_flag) and extract a unified log template.

Task 1:
Assign a connect_flag based on the type of client-AP interaction:
1 = The client connects or attempts to connect to the AP (includes authentication requests/responses, association requests/responses, IP detection, and reconnection attempts)
-1 = The client disconnects from the AP
0 = All other events (e.g., DHCP events, connect failed, error, probe requests, etc.)

Task 2:
Replace all variable content with .*? to extract a Regular Expression Template while keeping fixed structure and delimiters.
Your regular expression must match the entire log line from beginning to end.
You are allowed to define named capture groups (?P<name>.*?) only for the following fields:
- year/month/date/time/ampm: timestamp of the log
 - client_ip/ap_ip: IP address of the client or AP
 - client_mac/ap_mac: MAC address of the client or AP
 - client_name/ap_name: Name of the client or AP, may contain underscores and hyphens

IMPORTANT RULES:
- Use ONLY .*? for ALL content matching. Do not use \\d+, \\w+, [0-9]+, [a-zA-Z]+ or any other specific character classes.
- Extract only the client and AP involved in the actual event, not the logging source.
- When the same field appears again, use numbered suffixes for named groups: (?P<ap_name_1>...), (?P<client_ip_2>...)
- JSON handling: If logs contain JSON, capture the entire JSON as (?P<json_data>.*?).

OUTPUT FORMAT
You must respond with ONLY a valid JSON object. Do not include any explanations, code blocks, or additional text.
Return exactly this format:
{{"regex": "Regular expressions with named capture groups", "connect_flag": 1/-1/0}}

{examples_text}
""".strip()

    def get_total_time(self) -> float:
        return self._total_time

    def get_total_tokens(self) -> int:
        return self._token_count

    def get_call_count(self) -> int:
        return self._call_count
