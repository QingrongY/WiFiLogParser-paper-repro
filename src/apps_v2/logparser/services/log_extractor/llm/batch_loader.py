from __future__ import annotations

import random
from collections import OrderedDict
from typing import List

import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class BatchLoader:
    """Sampling helper to pick representative logs from a cluster."""

    def __init__(self, max_dpp_logs: int = 2000):
        self.max_dpp_logs = max_dpp_logs

    def sample_cluster(self, cluster, batch_size: int = 8, sample_method: str = "dpp") -> List[str]:
        if not cluster.logs:
            return []
        unique_logs = list(OrderedDict.fromkeys(cluster.logs))
        if len(unique_logs) <= batch_size:
            return unique_logs
        method = sample_method
        if method == "dpp" and len(unique_logs) > self.max_dpp_logs:
            logger.info(
                "BatchLoader: cluster size %s exceeds max DPP size %s, using random sampling",
                len(unique_logs),
                self.max_dpp_logs,
            )
            method = "random"
        tfidf_matrix = None
        if method in {"dpp", "similar"}:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(unique_logs).toarray()
        if method == "dpp":
            similarity_matrix = cosine_similarity(tfidf_matrix)
            indices = self._dpp_sample(similarity_matrix, batch_size)
        elif method == "random":
            indices = random.sample(range(len(unique_logs)), batch_size)
        else:
            indices = self._cluster_sample(tfidf_matrix, batch_size)[0]
        return [unique_logs[i] for i in indices]

    def _dpp_sample(self, similarity_matrix: np.ndarray, k: int) -> List[int]:
        n = similarity_matrix.shape[0]
        selected: set[int] = set()
        for _ in range(k):
            best_index = -1
            best_prob = -1.0
            for i in range(n):
                if i in selected:
                    continue
                indices = list(selected) + [i]
                submatrix = similarity_matrix[np.ix_(indices, indices)]
                det = np.linalg.det(submatrix)
                prob = det / (1 + det)
                if prob > best_prob:
                    best_prob = prob
                    best_index = i
            if best_index == -1:
                remaining = [i for i in range(n) if i not in selected]
                if not remaining:
                    break
                best_index = random.choice(remaining)
            selected.add(best_index)
        return list(selected)

    def _cluster_sample(self, matrix: np.ndarray, batch_size: int):
        if matrix.shape[0] % batch_size:
            n_clusters = matrix.shape[0] // batch_size + 1
        else:
            n_clusters = matrix.shape[0] // batch_size
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(matrix)
        similarity_matrix = self._cosine_similarity(matrix, kmeans.cluster_centers_)
        rankings = np.argsort(-similarity_matrix, axis=1)

        groups = [[] for _ in range(n_clusters)]
        for sample_idx, label in enumerate(kmeans.labels_):
            groups[label].append(sample_idx)

        for group_idx, group in enumerate(groups):
            if len(group) > batch_size:
                group_sorted = sorted(group, key=lambda idx: similarity_matrix[idx, group_idx], reverse=True)
                groups[group_idx] = group_sorted[:batch_size]
                for sample_idx in group_sorted[batch_size:]:
                    for candidate_group in rankings[sample_idx]:
                        if len(groups[candidate_group]) < batch_size:
                            groups[candidate_group].append(sample_idx)
                            break
        return groups

    def _cosine_similarity(self, vector_a: np.ndarray, vector_b: np.ndarray) -> np.ndarray:
        num = np.dot(vector_a, vector_b.T)
        denom = np.linalg.norm(vector_a, axis=1).reshape(-1, 1) * np.linalg.norm(vector_b, axis=1)
        similarity_matrix = num / denom
        similarity_matrix[np.isneginf(similarity_matrix)] = 0
        similarity_matrix = 0.5 + 0.5 * similarity_matrix
        return similarity_matrix
