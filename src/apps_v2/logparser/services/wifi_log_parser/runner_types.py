from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from apps_v2.logparser.services.log_extractor.preprocessing.data_clusterer import Cluster
from apps_v2.logparser.services.log_extractor.runner import ExtractionStats


# Limit how many lines we keep in memory to infer the timestamp rule.
# This bounds memory usage for huge files.
TIMESTAMP_INFERENCE_MAX_LINES = 20_000

# Per-cluster reservoir sample pool size.
# We validate/repair using only this sample, not all raw lines.
CLUSTER_SAMPLE_LIMIT = 200

# Emit progress heartbeats at most once per interval.
PROGRESS_EMIT_EVERY_SECONDS = 1.0

# File-level progress weighting across passes.
# These weights are used to compute a monotonic `file_fraction` (0..1) so the
# UI keeps moving during multi-pass processing.
SCAN_WEIGHT = 0.30
INDUCE_WEIGHT = 0.40
EXTRACT_WEIGHT = 0.30

# Streaming extraction record batch size (internal default).
DEFAULT_RECORD_BATCH_SIZE = 2_000


@dataclass
class WiFiLogParserStats(ExtractionStats):
    """ExtractionStats + consolidation telemetry."""

    # Total clusters consolidated into existing buckets.
    clusters_consolidated: int = 0
    # Consolidated clusters that were LLM-eligible (size >= min_cluster_size).
    clusters_consolidated_eligible: int = 0
    # Consolidated clusters that were skipped due to min-cluster-size.
    clusters_consolidated_too_small: int = 0
    # Clusters below min-cluster-size in the initial micro-grouping.
    clusters_too_small_total: int = 0
    # Tiny clusters that remained uncovered by any learned regex.
    clusters_too_small_unmatched: int = 0
    # Echo the configured min-cluster-size for reports.
    min_cluster_size: int = 0


ProgressCallback = Callable[[dict], None]


@dataclass
class StreamingCluster(Cluster):
    """Memory-bounded Cluster.

    `CacheManager.clusters` expects `Cluster` instances. This subclass keeps the
    same public attributes (logs/indices/cluster_id/template), but overrides the
    semantics:

    - `count` tracks the total number of lines assigned to the cluster.
    - `logs`/`indices` store a bounded reservoir sample only.
    - `size` reports the total count (not sample size).
    """

    sample_limit: int = 0
    count: int = 0

    @property
    def size(self) -> int:  # type: ignore[override]
        return int(self.count)

    def add(self, log: str, index: int) -> None:
        self.count += 1
        limit = int(self.sample_limit)
        if limit <= 0:
            return

        if len(self.logs) < limit:
            self.logs.append(log)
            self.indices.append(int(index))
            return

        j = random.randint(0, self.count - 1)
        if j < limit:
            self.logs[j] = log
            self.indices[j] = int(index)
