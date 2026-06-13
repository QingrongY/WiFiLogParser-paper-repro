from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Sequence

from ..common.api_clients import APIClient
from ..common.config import LLMExtractorSettings
from ..common.utils import count_message_tokens
from ..llm.batch_loader import BatchLoader
from ..llm.postprocessor import PostProcessor
from ..preprocessing.data_clusterer import Cluster
from .regex_validator import RegexValidator


@dataclass
class ModelStats:
    calls: int = 0
    successful: int = 0
    failed: int = 0
    tokens: int = 0
    time: float = 0.0


@dataclass
class RepairStats:
    attempted: int = 0
    successful: int = 0
    failed: int = 0
    responses: list[str] = field(default_factory=list)


class LLMRepairer:
    """Single-attempt repair, optionally cycled across a fallback model list.

    Each group is granted one repair attempt per fallback model. The inner
    same-model retry loop is intentionally absent: empirically, a single shot
    with diagnostic feedback either succeeds or stagnates.
    """

    def __init__(self, models: Sequence[str], settings: LLMExtractorSettings, max_attempts: int = 3):
        self.models = [model for model in models if model]
        self.settings = settings
        self.max_attempts = max_attempts
        self.validator = RegexValidator()
        self.postprocessor = PostProcessor()
        self.stats = RepairStats()
        self.total_tokens = 0
        self.total_time = 0.0
        self.model_stats: dict[str, ModelStats] = {model: ModelStats() for model in self.models}

    def repair(
        self,
        cluster: Cluster,
        *,
        sample_logs: Iterable[str],
        broken_regex: str | None,
        failed_log: str | None,
        failure_info: str | None,
        connect_flag: int,
    ) -> tuple[str | None, int]:
        if not self.models:
            self.stats.failed += 1
            return None, connect_flag

        cluster_logs = list(cluster.logs)
        samples = self._prepare_samples(cluster_logs, failed_log, sample_logs)
        previous_regex = broken_regex
        same_regex_count = 0

        for attempt, model in enumerate(self.models[: self.max_attempts], start=1):
            client = self._build_client(model)
            if not client:
                continue
            fixed_regex = self._attempt_fallback_fix(
                client=client,
                model=model,
                broken_regex=broken_regex or "",
                failed_log=failed_log or (cluster_logs[0] if cluster_logs else ""),
                failure_info=failure_info or "Unknown failure",
                samples=samples,
            )
            if not fixed_regex:
                continue

            if fixed_regex == previous_regex:
                same_regex_count += 1
                if same_regex_count >= 2:
                    break
            else:
                same_regex_count = 0
            previous_regex = fixed_regex

            valid, _, _, _ = self.validator.diagnose(fixed_regex, cluster_logs)
            if valid:
                self.stats.successful += 1
                self.model_stats[model].successful += 1
                return fixed_regex, connect_flag
            self.model_stats[model].failed += 1

        self.stats.failed += 1
        return None, connect_flag

    def _prepare_samples(
        self,
        cluster_logs: List[str],
        failed_log: str | None,
        sample_logs: Iterable[str],
        sample_size: int = 5,
    ) -> List[str]:
        unique_logs = list(dict.fromkeys(sample_logs or cluster_logs))
        if not unique_logs:
            return []
        temp_cluster = Cluster()
        for idx, log in enumerate(unique_logs):
            temp_cluster.append_log(log, idx)
        sampled = BatchLoader().sample_cluster(temp_cluster, min(sample_size, temp_cluster.size or sample_size))
        if failed_log and failed_log not in sampled:
            if sampled:
                sampled[0] = failed_log
            else:
                sampled = [failed_log]
        return sampled

    def _attempt_fallback_fix(
        self,
        *,
        client: APIClient,
        model: str,
        broken_regex: str,
        failed_log: str,
        failure_info: str,
        samples: Sequence[str],
    ) -> str | None:
        self.stats.attempted += 1
        self.model_stats[model].calls += 1
        samples_text = "\n".join(f"Sample {idx + 1}: {log}" for idx, log in enumerate(samples))
        instructions = f"""
ORIGINAL BROKEN REGEX:
{broken_regex}

FAILED LOG:
{failed_log}

The regex failed near:
{failure_info}

You must fix the regex to match ALL samples below without changing named capture group semantics.
{samples_text}

Respond ONLY with JSON: {{"regex": "fixed pattern here"}}
""".strip()
        messages: Sequence[Mapping[str, str]] = [{"role": "user", "content": instructions}]
        start = time.monotonic()
        response = client.chat(messages)
        elapsed = time.monotonic() - start
        tokens = count_message_tokens(messages, "gpt-4o-mini")
        self.total_time += elapsed
        self.total_tokens += tokens
        self.model_stats[model].time += elapsed
        self.model_stats[model].tokens += tokens
        self.stats.responses.append(response.content)
        fixed_regex, _ = self.postprocessor.process_output(response.content)
        if not fixed_regex:
            self.model_stats[model].failed += 1
        return fixed_regex

    def _build_client(self, model: str) -> APIClient | None:
        api_key = self.settings.api_key or ""
        if not api_key:
            return None
        return APIClient(
            base_url=self.settings.base_url,
            api_key=api_key,
            model=model,
            timeout_seconds=self.settings.request_timeout_seconds,
            max_retries=self.settings.max_retries,
            reasoning_effort=self.settings.reasoning_effort,
        )
