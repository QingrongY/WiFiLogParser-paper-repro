from __future__ import annotations

import logging
import time
from typing import Dict, Sequence

import regex

from apps_v2.logparser.services.log_extractor.preprocessing.data_clusterer import Cluster


logger = logging.getLogger(__name__)


class WiFiRunnerInduceMixin:
    def _process_clusters(
        self,
        *,
        progress_callback,
        file_fraction_base: float,
        file_fraction_weight: float,
    ) -> tuple[Dict[int, dict], dict]:
        total_clusters = len(self.cache_manager.clusters)
        clusters_sorted = sorted(self.cache_manager.clusters.values(), key=lambda c: c.size, reverse=True)

        too_small_ids: set[int] = set()
        for cluster in clusters_sorted:
            cluster_id = getattr(cluster, "cluster_id", None)
            if cluster_id is None:
                continue
            if int(getattr(cluster, "size", 0) or 0) < self.min_cluster_size:
                too_small_ids.add(int(cluster_id))

        processed_ids: set[int] = set()
        consolidated_ids: set[int] = set()

        def _snapshot() -> dict:
            consolidated_too_small = len(consolidated_ids & too_small_ids)
            consolidated_eligible = len(consolidated_ids - too_small_ids)
            processed_effective = len(processed_ids - consolidated_ids)
            clusters_skipped = len(too_small_ids) + consolidated_eligible
            too_small_unmatched = max(0, len(too_small_ids) - consolidated_too_small)
            return {
                "total_clusters": total_clusters,
                "clusters_processed": processed_effective,
                "clusters_skipped": clusters_skipped,
                "clusters_consolidated": len(consolidated_ids),
                "clusters_consolidated_eligible": consolidated_eligible,
                "clusters_consolidated_too_small": consolidated_too_small,
                "clusters_too_small_total": len(too_small_ids),
                "clusters_too_small_unmatched": too_small_unmatched,
            }

        if progress_callback:
            snapshot = _snapshot()
            progress_callback(
                {
                    "pass": "induce",
                    "phase": "start",
                    "file_fraction": self._weighted_file_fraction(
                        base=file_fraction_base,
                        weight=file_fraction_weight,
                        progress=(
                            (snapshot.get("clusters_processed", 0) + snapshot.get("clusters_skipped", 0))
                            / max(1, total_clusters)
                            if total_clusters
                            else 0.0
                        ),
                    ),
                    "clusters_total": total_clusters,
                    **{k: v for k, v in snapshot.items() if k != "total_clusters"},
                    "min_cluster_size": int(self.min_cluster_size),
                }
            )

        cluster_results: Dict[int, dict] = {}

        for cluster in clusters_sorted:
            cluster_id = getattr(cluster, "cluster_id", None)
            if cluster_id is None:
                continue
            cluster_id = int(cluster_id)

            template = getattr(cluster, "template", "") or ""
            mapped = self.cache_manager.template_to_cluster.get(str(template))
            if mapped is not None and int(mapped) != cluster_id:
                continue

            if int(getattr(cluster, "size", 0) or 0) < self.min_cluster_size:
                continue

            cache_entry = self.cache_manager.cluster_to_regex.get(cluster_id)
            if cache_entry:
                cached_regex, cached_flag, _ = cache_entry
                if cached_flag == 0:
                    continue
                if cached_regex:
                    continue

            logger.info(
                "WiFiLogParserRunner: processing canonical cluster %s size=%s",
                cluster_id,
                getattr(cluster, "size", 0),
            )

            cluster_start = time.monotonic()
            try:
                result = self._process_cluster(cluster)  # type: ignore[arg-type]
            except Exception as exc:
                logger.warning("WiFiLogParserRunner: cluster %s failed, skipping: %s", cluster_id, exc)
                result = {"template": None, "regex": None, "connect_flag": 0}
            processed_ids.add(cluster_id)
            elapsed = time.monotonic() - cluster_start
            logger.info("WiFiLogParserRunner: cluster %s finished in %.2fs", cluster_id, elapsed)

            cache_entry = self.cache_manager.cluster_to_regex.get(cluster_id)
            regex_pattern = cache_entry[0] if cache_entry else None
            connect_flag = int(cache_entry[1]) if cache_entry else int(result.get("connect_flag", 0) or 0)

            merged_count = 0
            if regex_pattern:
                merged_ids = self._consolidate_by_coverage(
                    canonical_cluster_id=cluster_id,
                    canonical_regex=regex_pattern,
                    too_small_ids=too_small_ids,
                    consolidated_ids=consolidated_ids,
                )
                merged_count = len(merged_ids)

            if merged_count:
                result = dict(result)
                result["consolidated_clusters"] = merged_count
            cluster_results[cluster_id] = result

            if progress_callback:
                snapshot = _snapshot()
                handled = (snapshot.get("clusters_processed", 0) or 0) + (snapshot.get("clusters_skipped", 0) or 0)
                progress_callback(
                    {
                        "pass": "induce",
                        "phase": "cluster_complete",
                        "file_fraction": self._weighted_file_fraction(
                            base=file_fraction_base,
                            weight=file_fraction_weight,
                            progress=(handled / max(1, total_clusters)) if total_clusters else 0.0,
                        ),
                        "clusters_total": total_clusters,
                        **{k: v for k, v in snapshot.items() if k != "total_clusters"},
                        "cluster_id": cluster_id,
                        "cluster_size": getattr(cluster, "size", 0),
                        "connect_flag": connect_flag,
                        "consolidated": merged_count,
                    }
                )

        if progress_callback:
            snapshot = _snapshot()
            handled = (snapshot.get("clusters_processed", 0) or 0) + (snapshot.get("clusters_skipped", 0) or 0)
            progress_callback(
                {
                    "pass": "induce",
                    "phase": "induce_complete",
                    "file_fraction": self._weighted_file_fraction(
                        base=file_fraction_base,
                        weight=file_fraction_weight,
                        progress=(handled / max(1, total_clusters)) if total_clusters else 1.0,
                    ),
                    "clusters_total": total_clusters,
                    **{k: v for k, v in snapshot.items() if k != "total_clusters"},
                    "min_cluster_size": int(self.min_cluster_size),
                }
            )

        return cluster_results, _snapshot()

    def _process_cluster(self, cluster: Cluster) -> dict:
        examples_text = self._build_template_examples_text()
        sample_logs = self.batch_loader.sample_cluster(cluster, batch_size=min(self.settings.batch_size, cluster.size))
        logger.info(
            "WiFiLogParserRunner: invoking LLM for cluster %s sample_size=%s examples=%s",
            cluster.cluster_id,
            len(sample_logs),
            bool(examples_text),
        )
        start_time = time.monotonic()
        raw_output = self.parser.parse_batch(sample_logs, examples_text)
        logger.info(
            "WiFiLogParserRunner: LLM finished cluster %s in %.2fs",
            cluster.cluster_id,
            time.monotonic() - start_time,
        )

        parsed = self._template_postprocessor.process_output(raw_output)
        template = parsed.template
        connect_flag = parsed.connect_flag
        self.cache_manager.update_cluster_flag(cluster, connect_flag)

        if not template and sample_logs:
            logger.info(
                "WiFiLogParserRunner: retrying primary model for cluster %s due to missing template",
                cluster.cluster_id,
            )
            retry_output = self.parser.parse_batch(sample_logs, examples_text)
            parsed_retry = self._template_postprocessor.process_output(retry_output)
            if parsed_retry.template:
                template = parsed_retry.template
            if parsed_retry.connect_flag != connect_flag:
                connect_flag = parsed_retry.connect_flag
                self.cache_manager.update_cluster_flag(cluster, connect_flag)

        regex_pattern: str | None = None
        if template:
            try:
                regex_pattern = self._template_compiler.compile(template)
            except Exception as exc:
                logger.info("WiFiLogParserRunner: template compile failed for cluster %s: %s", cluster.cluster_id, exc)
                regex_pattern = None

        validation = (
            self.validator.diagnose(regex_pattern, cluster.logs)
            if regex_pattern
            else (False, None, "Missing template/regex", "missing")
        )
        is_valid, failed_log, failure_info, _ = validation

        result_payload: dict = {"template": template, "regex": regex_pattern, "connect_flag": connect_flag}

        if regex_pattern and is_valid:
            self.cache_manager.update_cluster_regex(cluster, regex_pattern)
            if template and connect_flag in (1, -1) and cluster.logs:
                self._remember_successful_template(template=template, connect_flag=connect_flag, sample_log=cluster.logs[0])
            return result_payload

        self.cache_manager.update_cluster_regex(cluster, None)
        if connect_flag == 0:
            result_payload["regex"] = None
            return result_payload

        # Fallback: repair the compiled regex (still returns regex JSON). This is best-effort.
        broken = regex_pattern
        repaired_regex, repaired_flag = self.repairer.repair(
            cluster,
            sample_logs=sample_logs,
            broken_regex=broken,
            failed_log=failed_log,
            failure_info=failure_info,
            connect_flag=connect_flag,
        )
        repaired_regex = self._anchor_regex(repaired_regex) if repaired_regex else None

        if repaired_regex and self.validator.validate_regex(repaired_regex, cluster.logs):
            self.cache_manager.update_cluster_regex(cluster, repaired_regex)
            self.cache_manager.update_cluster_flag(cluster, repaired_flag)
            result_payload["regex"] = repaired_regex
            result_payload["connect_flag"] = repaired_flag
            result_payload["repaired"] = True
            return result_payload

        # Keep the template for inspection even if regex failed.
        result_payload["regex"] = None
        if failure_info:
            result_payload["error"] = str(failure_info)
        return result_payload

    def _build_template_examples_text(self) -> str:
        if not self._successful_templates:
            return ""
        examples = []
        take = getattr(self.parser, "successful_examples", 0) or 0
        if take <= 0:
            take = 3
        for template, connect_flag, sample_log in self._successful_templates[-take:]:
            examples.append(f"Log: `{sample_log}`\ntemplate: \"{template}\"\nconnect_flag: {connect_flag}")
        return "\n\nSUCCESSFUL EXAMPLES:\n" + "\n\n".join(examples)

    def _remember_successful_template(self, *, template: str, connect_flag: int, sample_log: str) -> None:
        if not template:
            return
        if connect_flag not in (1, -1):
            return
        self._successful_templates.append((template, int(connect_flag), str(sample_log)))

    @staticmethod
    def _anchor_regex(regex_pattern: str | None) -> str | None:
        if not regex_pattern:
            return None
        pattern = str(regex_pattern).strip()
        if not pattern:
            return None
        if not (pattern.startswith("\\A") or pattern.startswith("^")):
            pattern = "\\A" + pattern
        if not (pattern.endswith("\\Z") or pattern.endswith("$")):
            pattern = pattern + "\\Z"
        return pattern

    def _consolidate_by_coverage(
        self,
        *,
        canonical_cluster_id: int,
        canonical_regex: str,
        too_small_ids: set[int],
        consolidated_ids: set[int],
    ) -> list[int]:
        """Merge micro-groups into `canonical_cluster_id` if covered by `canonical_regex`.

        Implements paper Step 2 consolidation: representative check then full-group
        validation before mapping the signature to the canonical parser bucket.

        Returns the list of cluster ids newly consolidated.
        """

        try:
            compiled = regex.compile(canonical_regex)
        except Exception:
            return []

        merged: list[int] = []
        for candidate in self.cache_manager.clusters.values():
            candidate_id = candidate.cluster_id
            if candidate_id is None:
                continue
            candidate_id = int(candidate_id)
            if candidate_id == canonical_cluster_id:
                continue

            if candidate_id in consolidated_ids:
                continue

            template = candidate.template or ""
            mapped = self.cache_manager.template_to_cluster.get(template)
            if mapped is None or int(mapped) != candidate_id:
                # Only consolidate clusters that are still canonical for their signature.
                continue

            cache_entry = self.cache_manager.cluster_to_regex.get(candidate_id)
            if cache_entry and cache_entry[0]:
                # Candidate already has a learned regex; don't attempt to overwrite.
                continue

            if not candidate.logs:
                continue

            if not self._safe_compiled_fullmatch(compiled, candidate.logs[0]):
                continue

            if not self._compiled_fullmatches_all(compiled, candidate.logs):
                continue

            # Consolidate: map this signature to the canonical parser bucket.
            self.cache_manager.template_to_cluster[template] = canonical_cluster_id
            consolidated_ids.add(candidate_id)
            merged.append(candidate_id)

        return merged

    @staticmethod
    def _safe_compiled_fullmatch(compiled: regex.Pattern, text: str) -> bool:
        try:
            return bool(compiled.fullmatch(text))
        except Exception:
            return False

    @classmethod
    def _compiled_fullmatches_all(cls, compiled: regex.Pattern, logs: Sequence[str]) -> bool:
        for line in logs:
            if not cls._safe_compiled_fullmatch(compiled, line):
                return False
        return True
