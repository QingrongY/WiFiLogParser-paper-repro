from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass
class Cluster:
    logs: List[str] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    cluster_id: int | None = None
    template: str | None = None

    @property
    def size(self) -> int:
        return len(self.logs)

    def append_log(self, log: str, index: int) -> None:
        self.logs.append(log)
        self.indices.append(index)

    def extend(self, other: "Cluster") -> None:
        self.logs.extend(other.logs)
        self.indices.extend(other.indices)


class DataClusterer:
    def normalize_logs(self, logs: Iterable[str]) -> List[str]:
        if not logs:
            return []
        logs = list(logs)
        standard_keys = self._collect_json_keys(logs)
        if standard_keys:
            standardized = []
            for log in logs:
                standardized.append(self._standardize_json_in_log(log, standard_keys))
            logs = standardized
        return logs

    def cluster_logs(self, logs: Iterable[str], log_indices: Iterable[int]) -> List[Cluster]:
        if not logs:
            return []
        clusters: dict[str, Cluster] = {}
        next_id = 0
        for log, index in zip(logs, log_indices):
            template = self._normalize_log_for_template(log)
            cluster = clusters.get(template)
            if not cluster:
                cluster = Cluster(cluster_id=next_id, template=template)
                clusters[template] = cluster
                next_id += 1
            cluster.append_log(log, index)
        return list(clusters.values())

    def _collect_json_keys(self, logs: Iterable[str]) -> List[str]:
        all_keys: set[str] = set()
        for log in logs:
            if not log:
                continue
            json_parts = re.findall(r"\{[^}]*\}", log)
            for json_str in json_parts:
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    keys = re.findall(r'"([^"]+)"\s*:', json_str)
                    if keys:
                        has_numbered = any(re.match(r"^[a-zA-Z_]+_\d+$", key) for key in keys)
                        if not has_numbered:
                            all_keys.update(keys)
                    continue
                keys = list(parsed.keys())
                has_numbered = any(re.match(r"^[a-zA-Z_]+_\d+$", key) for key in keys)
                if not has_numbered:
                    all_keys.update(keys)
        return sorted(all_keys)

    def _standardize_json_in_log(self, log_content: str, standard_keys: Iterable[str]) -> str:
        if not log_content or not standard_keys:
            return log_content
        if not re.search(r"\{[^}]*\}", log_content):
            return log_content

        def replace_json(match: re.Match[str]) -> str:
            json_str = match.group(0)
            try:
                parsed = json.loads(json_str)
                has_numbered = any(re.match(r"^[a-zA-Z_]+_\d+$", key) for key in parsed.keys())
                if has_numbered:
                    return json_str
                standardized = {key: parsed.get(key, "") for key in standard_keys}
                return json.dumps(standardized, separators=(",", ":"))
            except json.JSONDecodeError:
                return json_str

        return re.sub(r"\{[^}]*\}", replace_json, log_content)

    def _normalize_log_for_template(self, log_content: str) -> str:
        """Replicate the original skeleton normalisation logic to avoid cluster explosion."""
        common_tlds = "com|org|net|edu|gov|mil|int|co|io|me|tv|info|biz|name"
        normalized = re.sub(rf"(www\.)?[a-zA-Z0-9.-]+\.({common_tlds})", "", log_content)
        normalized = re.sub(r"(?=[A-Za-z0-9-]*-)[A-Za-z0-9-]+", "", normalized)
        normalized = re.sub(r"/[a-zA-Z0-9_./\\-]+(?=[|\s:,;!?]|$)", "", normalized)
        normalized = re.sub(r"[a-zA-Z]:\\[a-zA-Z0-9_.\\/-]+(?=[|\s:,;!?]|$)", "", normalized)
        normalized = re.sub(r"0x[0-9a-fA-F]+", "", normalized)
        normalized = re.sub(r"\b[0-9a-fA-F]{2}(?:[0-9a-fA-F]{2})*\b", "", normalized)
        normalized = re.sub(r"\d", "", normalized)
        normalized = re.sub(r"=", "", normalized)
        normalized = re.sub(r"\s+", "", normalized)
        return normalized.strip()
