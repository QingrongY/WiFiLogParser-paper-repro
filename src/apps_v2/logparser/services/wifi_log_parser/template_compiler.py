from __future__ import annotations

import re

import regex


_PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")

_ALLOWED_BASE_FIELDS: set[str] = {
    "year",
    "month",
    "date",
    "time",
    "ampm",
    "client_ip",
    "ap_ip",
    "client_mac",
    "ap_mac",
    "client_name",
    "ap_name",
    "json_data",
}


class TemplateCompiler:
    """Compile a literal template with placeholders into an anchored regex."""

    def compile(self, template: str) -> str:
        if template is None:
            raise ValueError("template is required")
        text = str(template)
        if not text.strip():
            raise ValueError("template is empty")

        parts: list[str] = []
        last = 0
        for match in _PLACEHOLDER_RE.finditer(text):
            literal = text[last: match.start()]
            if literal:
                parts.append(regex.escape(literal, special_only=True, literal_spaces=True))

            name = match.group(1)
            base = _strip_numeric_suffix(name)
            if base in _ALLOWED_BASE_FIELDS:
                parts.append(f"(?P<{name}>.*?)")
            else:
                parts.append(".*?")
            last = match.end()

        tail = text[last:]
        if tail:
            parts.append(regex.escape(tail, special_only=True, literal_spaces=True))

        body = "".join(parts)
        return f"\\A{body}\\Z"


def _strip_numeric_suffix(name: str) -> str:
    return re.sub(r"_\d+$", "", str(name))
