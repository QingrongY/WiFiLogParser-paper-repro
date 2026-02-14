from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from .data_clusterer import Cluster, DataClusterer


@dataclass
class CacheStats:
    cache_hits: int
    cache_misses: int
    template_patterns: int
    total_clusters: int

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100


class CacheManager:
    """In-memory cache tracking cluster processing results."""

    def __init__(self, clusterer: DataClusterer):
        self.clusterer = clusterer
        self.template_to_cluster: Dict[str, int] = {}
        self.cluster_to_regex: Dict[int, Tuple[str | None, int, str]] = {}
        self.clusters: Dict[int, Cluster] = {}
        self.next_cluster_id = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def merge_clusters(self, new_clusters: Iterable[Cluster]) -> None:
        for cluster in new_clusters:
            template = cluster.template or ""
            cluster_id = self.template_to_cluster.get(template)
            if cluster_id is not None:
                existing = self.clusters[cluster_id]
                existing.extend(cluster)
                continue
            cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
            cluster.cluster_id = cluster_id
            self.template_to_cluster[template] = cluster_id
            self.clusters[cluster_id] = cluster

    def get_unprocessed_clusters(self, batch_size: int) -> list[Cluster]:
        pending: list[Cluster] = []
        for cluster in self.clusters.values():
            if cluster.size < batch_size:
                continue
            cache_entry = self.cluster_to_regex.get(cluster.cluster_id or -1)
            if cache_entry:
                regex, connect_flag, _ = cache_entry
                if connect_flag == 0:
                    continue
                if regex:
                    continue
            pending.append(cluster)
        return pending

    def update_cluster_flag(self, cluster: Cluster, connect_flag: int) -> None:
        cluster_id = cluster.cluster_id
        if cluster_id is None:
            return
        regex, _, example = self.cluster_to_regex.get(cluster_id, (None, 0, cluster.logs[0] if cluster.logs else ""))
        self.cluster_to_regex[cluster_id] = (regex, connect_flag, example)

    def update_cluster_regex(self, cluster: Cluster, regex_pattern: str | None) -> None:
        cluster_id = cluster.cluster_id
        if cluster_id is None:
            return
        _, connect_flag, example = self.cluster_to_regex.get(cluster_id, (None, 0, cluster.logs[0] if cluster.logs else ""))
        self.cluster_to_regex[cluster_id] = (regex_pattern, connect_flag, example)

    def get_cluster_info_for_preprocessed_log(self, preprocessed_log: str) -> tuple[int, str | None, int] | None:
        template = self.clusterer._normalize_log_for_template(preprocessed_log)
        cluster_id = self.template_to_cluster.get(template)
        if cluster_id is None:
            self.cache_misses += 1
            return None
        cache_entry = self.cluster_to_regex.get(cluster_id)
        if not cache_entry:
            self.cache_misses += 1
            return None
        self.cache_hits += 1
        regex, connect_flag, _ = cache_entry
        return cluster_id, regex, connect_flag

    def get_stats(self) -> CacheStats:
        return CacheStats(
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            template_patterns=len(self.cluster_to_regex),
            total_clusters=len(self.clusters),
        )
