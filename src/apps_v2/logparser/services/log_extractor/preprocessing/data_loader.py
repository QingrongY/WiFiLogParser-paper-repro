from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import re


class DataLoader:
    """Mimic the original LogExtractor preprocessing pipeline."""

    def load_lines(self, file_path: str | Path | None, content: str | None = None) -> List[str]:
        if content is None and file_path is None:
            return []
        if content is None:
            with Path(file_path).open("r", encoding="utf-8", errors="ignore") as handle:
                content = handle.read()
        if not content:
            return []
        lines = [line.rstrip("\r") for line in content.splitlines()]
        return lines

    def preprocess_lines(self, logs: Iterable[str]) -> List[str]:
        preprocessed: List[str] = []
        for log in logs:
            if not isinstance(log, str):
                preprocessed.append(log)
                continue
            processed_log = re.sub(r'<\s*([^<>]*?)\s*>', r'<\1>', log)
            processed_log = processed_log.replace('\t', ' ')
            processed_log = re.sub(r'\s{2,}', ' ', processed_log)
            processed_log = processed_log.strip() + "|"
            preprocessed.append(processed_log)
        return preprocessed
