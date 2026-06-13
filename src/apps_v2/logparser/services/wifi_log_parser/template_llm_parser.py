from __future__ import annotations

from apps_v2.logparser.services.log_extractor.llm.llm_parser import LLMParser


class TemplateLLMParser(LLMParser):
    """LLM parser that returns a template instead of a raw regex.

    Business logic is unchanged (connect_flag + full-line structure extraction).
    Only the OUTPUT FORMAT changes: the model returns a literal template string
    with placeholders that we compile into a regex.
    """

    def _build_instruction(self, examples_text: str) -> str:  # noqa: D401 - keep prompt text readable
        return f"""
You will be given WiFi logs. Your task is to classify the event type (connect_flag) and extract a unified log template.

Task 1:
Assign a connect_flag based on the type of client-AP interaction:
1 = The client connects or attempts to connect to the AP (includes authentication requests/responses, association requests/responses, IP detection, and reconnection attempts)
-1 = The client disconnects from the AP
0 = All other events (e.g., DHCP events, connect failed, error, probe requests, etc.)

Task 2:
Extract a single TEMPLATE that covers all log lines from beginning to end, preserving fixed structure and delimiters.
Replace all variable content spans with placeholders in the form {{{{name}}}}.

We will compile the template into a regular expression as follows:
- Every placeholder becomes a non-greedy wildcard (.*?).
- Placeholders that correspond to allowed schema fields become named capture groups (?P<name>.*?).

IMPORTANT RULES:
- Do NOT output a raw regex. Output ONLY the literal template with {{{{placeholders}}}}.
- Do NOT include any regex operators in the template (no .*?, no character classes, no anchors). Avoid backslashes unless a backslash appears literally in the log line.
- The compiled regex must match the entire log line from beginning to end.
- Extract only the client and AP involved in the event.
- You may use field-named placeholders ONLY for the following fields:
  - year/month/date/time/ampm: timestamp of the log
  - client_ip/ap_ip: IP address of the client or AP
  - client_mac/ap_mac: MAC address of the client or AP
  - client_name/ap_name: Name of the client or AP, may contain underscores and hyphens
  - ssid: Wi-Fi network name, not an AP
  - json_data: if logs contain JSON, capture the entire JSON
- JSON handling: If logs contain JSON, replace the entire JSON with {{{{json_data}}}}.
- When the same field appears again, use numbered suffixes: {{{{ap_name_1}}}}, {{{{client_ip_2}}}}.

OUTPUT FORMAT
You must respond with ONLY a valid JSON object. Do not include any explanations, code blocks, or additional text.
Return exactly this format:
{{"template": "Full-line template with placeholders", "connect_flag": 1/-1/0}}

{examples_text}
""".strip()
