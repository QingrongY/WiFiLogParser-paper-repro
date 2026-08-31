#!/usr/bin/env python3
"""Run pinned LILAC+/LogBatcher+ matched-prefix experiments.

The upstream parsers produce generic templates. This runner provides only an
OpenAI-compatible provider adapter and the shared, versioned two-stage `+`
post-processing required by the mobility-record task.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "baselines"
UPSTREAM_DIR = ROOT / ".baseline-work" / "upstream"
EVENTS = ("connect", "disconnect", "other")
MAC_RE = re.compile(r"(?i)(?<![0-9a-f])(?:[0-9a-f]{2}:){5}[0-9a-f]{2}(?![0-9a-f])")
GROUP_SUFFIX_RE = re.compile(r"^(ap|client)_(ip|mac|name)(?:_[12])?$")


def _sha256_bytes(chunks: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_bytes(iter(lambda: handle.read(1024 * 1024), b""))


def _sha256_file_prefix(path: Path, line_limit: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for _ in range(line_limit):
            line = handle.readline()
            if not line:
                raise ValueError(f"{path} has fewer than {line_limit} physical lines")
            digest.update(line)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _read_lines(path: Path, limit: int) -> list[str]:
    lines: list[str] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        for line in handle:
            lines.append(line.rstrip("\r\n"))
            if len(lines) == limit:
                break
    if len(lines) != limit:
        raise ValueError(f"{path} has {len(lines)} lines; expected at least {limit}")
    return lines


def _normalize_event(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"connect", "1", "+1"}:
        return "connect"
    if text in {"disconnect", "-1"}:
        return "disconnect"
    return "other"


def _normalize_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return ""
    compact = text.replace("-", ":").replace(".", "").replace(" ", "")
    if len(compact) == 12 and ":" not in compact and all(ch in "0123456789abcdef" for ch in compact):
        return ":".join(compact[i : i + 2] for i in range(0, 12, 2))
    return text


def _partial_match(predicted: str, expected: str) -> bool:
    pred = _normalize_id(predicted)
    gt = _normalize_id(expected)
    if not pred or not gt:
        return False
    if pred == gt or pred in gt or gt in pred:
        return True
    pred_compact = re.sub(r"[^a-z0-9]", "", pred)
    gt_compact = re.sub(r"[^a-z0-9]", "", gt)
    return bool(pred_compact and gt_compact and (pred_compact in gt_compact or gt_compact in pred_compact))


def _looks_like_mac(value: str) -> bool:
    return bool(MAC_RE.fullmatch(_normalize_id(value)))


def _pick_extracted_identifier(groups: dict[str, str | None], role: str) -> str:
    if role == "ap":
        order = (
            "ap_mac", "ap_mac_1", "ap_mac_2",
            "ap_ip", "ap_ip_1", "ap_ip_2",
            "ap_name", "ap_name_1", "ap_name_2",
            "ap_id",
        )
    else:
        order = (
            "client_mac_2", "client_mac_1", "client_mac",
            "client_ip_2", "client_ip_1", "client_ip",
            "client_name_2", "client_name_1", "client_name",
            "client_id",
        )
    for key in order:
        value = str(groups.get(key) or "").strip()
        if not value:
            continue
        if key.startswith("ap_mac") and not _looks_like_mac(value):
            continue
        return _normalize_id(value)
    return ""


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision and recall else 0.0


class TrackedLLM:
    def __init__(self, config: dict[str, Any], calls_path: Path):
        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            raise ValueError("LLM_API_KEY is not set; copy .env.example to .env")
        self.model = str(config["model"])
        self.temperature = float(config.get("temperature", 0.0))
        self.timeout = float(config.get("request_timeout_seconds", 60))
        self.max_retries = int(config.get("max_retries", 2))
        self.base_url = os.getenv("LLM_BASE_URL", "").strip() or None
        self.client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=self.timeout)
        self.calls_path = calls_path
        self.call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def complete(self, messages: list[dict[str, str]], context: dict[str, Any]) -> str:
        last_error: Exception | None = None
        started = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                text = (response.choices[0].message.content or "").strip()
                usage = getattr(response, "usage", None)
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                self.call_count += 1
                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens
                _append_jsonl(
                    self.calls_path,
                    {
                        **context,
                        "attempt": attempt,
                        "model": self.model,
                        "temperature": self.temperature,
                        "messages": messages,
                        "response": text,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "elapsed_seconds": round(time.monotonic() - started, 6),
                    },
                )
                return text
            except Exception as exc:  # provider error details are retained locally
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 4))
        raise RuntimeError(f"LLM request failed after {self.max_retries + 1} attempts: {last_error}")

    def stats(self) -> dict[str, Any]:
        return {
            "api_calls": self.call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


def _extract_backtick_template(response: str, fallback: str) -> tuple[str, bool]:
    selected = ""
    for line in response.splitlines():
        if "Log template:" in line:
            selected = line
            break
    if not selected:
        selected = next((line for line in response.splitlines() if "`" in line), "")
    if selected:
        start, end = selected.find("`"), selected.rfind("`")
        if 0 <= start < end:
            return selected[start + 1 : end], True
        start, end = selected.find('"'), selected.rfind('"')
        if 0 <= start < end:
            return selected[start + 1 : end], True
    return fallback, False


def _import_lilac() -> tuple[Any, Any]:
    benchmark = UPSTREAM_DIR / "LILAC" / "benchmark"
    runtime_root = ROOT / ".baseline-work" / "runtime" / "lilac_import"
    cwd = runtime_root / "a" / "b"
    cwd.mkdir(parents=True, exist_ok=True)
    (runtime_root / "openai_key.txt").write_text("https://example.invalid/v1\nplaceholder\n", encoding="utf-8")
    sys.path.insert(0, str(benchmark))
    previous = Path.cwd()
    try:
        os.chdir(cwd)
        lilac_module = importlib.import_module("logparser.LILAC.LILAC")
        cache_module = importlib.import_module("logparser.LILAC.parsing_cache")
    finally:
        os.chdir(previous)
    return lilac_module, cache_module


def _run_lilac(
    lines: list[str], parser_config: dict[str, Any], llm: TrackedLLM, raw_log: Path
) -> list[str]:
    lilac, cache_module = _import_lilac()
    original_postprocess = importlib.import_module("logparser.LILAC.gpt_query").post_process_template

    def query_compatible(
        log_message: str,
        regs_common: list[Any] | None = None,
        examples: list[dict[str, str]] | None = None,
        model: str = "",
    ) -> tuple[str, bool]:
        exemplars = examples or [
            {
                "query": "Log message: `try to connected to host: 172.16.254.1, finished.`",
                "answer": "Log template: `try to connected to host: {ip_address}, finished.`",
            }
        ]
        messages = [
            {"role": "system", "content": "You are an expert of log parsing, and now you will help to do log parsing."},
            {
                "role": "user",
                "content": (
                    "I want you to act like an expert of log parsing. I will give you a log message "
                    "delimited by backticks. You must identify and abstract all the dynamic variables in logs "
                    "with {placeholder} and output a static log template. Print the input log's template "
                    "delimited by backticks."
                ),
            },
            {"role": "assistant", "content": "Sure, I can help you with log parsing."},
        ]
        for exemplar in exemplars:
            messages.append({"role": "user", "content": exemplar["query"]})
            messages.append({"role": "assistant", "content": exemplar["answer"]})
        messages.append({"role": "user", "content": f"Log message: `{log_message}`"})
        response = llm.complete(messages, {"stage": "lilac_template_induction"})
        template, valid = _extract_backtick_template(response, log_message)
        if valid:
            template, valid = original_postprocess(template, regs_common or [])
        if not valid:
            template, valid = original_postprocess(log_message, regs_common or [])
        return template, valid

    lilac.query_template_from_gpt_with_check = query_compatible
    parser = lilac.LogParser(
        log_format="<Content>",
        shot=int(parser_config.get("shot", 0)),
        example_size=int(parser_config.get("example_size", 0)),
        selection_method=str(parser_config.get("selection_method", "LILAC")),
        model=llm.model,
    )
    cache = cache_module.ParsingCache()
    log_messages: list[Any] = []
    log_templates: list[Any] = []
    with raw_log.open("w", encoding="utf-8") as diagnostic, contextlib.redirect_stdout(diagnostic), contextlib.redirect_stderr(diagnostic):
        for idx, line in enumerate(lines):
            parser.process_log(cache, [line], log_messages, log_templates, idx, None, [], len(lines))
    by_index = {int(idx): cache.template_list[int(template_id)] for template_id, idx in log_templates}
    return [by_index[idx] for idx in range(len(lines))]


def _run_logbatcher(
    lines: list[str],
    dataset_name: str,
    parser_config: dict[str, Any],
    llm: TrackedLLM,
    work_dir: Path,
    raw_log: Path,
) -> list[str]:
    sampling_method = str(parser_config.get("sampling_method", "dpp"))
    if sampling_method != "dpp":
        raise ValueError(
            "the pinned single_dataset_paring API uses its upstream default DPP sampling; "
            f"unsupported configured value: {sampling_method}"
        )
    source = UPSTREAM_DIR / "LogBatcher"
    sys.path.insert(0, str(source))
    parsing_base = importlib.import_module("logbatcher.parsing_base")
    parser_module = importlib.import_module("logbatcher.parser")

    class CompatibleParser(parser_module.Parser):
        def __init__(self) -> None:
            self.model = llm.model
            self.theme = "wildash-matched-prefix"
            self.dataset = dataset_name
            self.token_list = [0, 0]
            self.time_consumption_llm = 0.0
            self.api_key = "provider-adapter"

        def chat(self, messages: list[dict[str, str]]) -> str:
            return llm.complete(messages, {"stage": "logbatcher_template_induction"})

    work_dir.mkdir(parents=True, exist_ok=True)
    with raw_log.open("w", encoding="utf-8") as diagnostic, contextlib.redirect_stdout(diagnostic), contextlib.redirect_stderr(diagnostic):
        parsing_base.single_dataset_paring(
            dataset=dataset_name,
            contents=lines,
            output_dir=str(work_dir) + os.sep,
            parser=CompatibleParser(),
            batch_size=int(parser_config.get("batch_size", 10)),
            chunk_size=int(parser_config.get("chunk_size", len(lines))),
            clustering_method=str(parser_config.get("clustering_method", "dbscan")),
            debug=False,
        )
    structured = work_dir / f"{dataset_name}_full.log_structured.csv"
    with structured.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(lines):
        raise ValueError(f"LogBatcher returned {len(rows)} rows for {len(lines)} input lines")
    return [str(row["EventTemplate"]) for row in rows]


def _representatives(indexes: list[int], lines: list[str], count: int) -> list[tuple[int, str]]:
    if len(indexes) <= count:
        chosen = indexes
    elif count <= 1:
        chosen = [indexes[0]]
    else:
        chosen = [indexes[round(i * (len(indexes) - 1) / (count - 1))] for i in range(count)]
    return [(idx, lines[idx]) for idx in dict.fromkeys(chosen)]


def _render_prompt(name: str, **values: str) -> str:
    text = (BASELINE_DIR / "prompts" / name).read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return text


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object in response: {text[:160]}")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("LLM response is not a JSON object")
    return value


def _postprocess_templates(
    baseline: str,
    dataset: str,
    lines: list[str],
    templates: list[str],
    config: dict[str, Any],
    llm: TrackedLLM,
    annotation_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_template: dict[str, list[int]] = defaultdict(list)
    for idx, template in enumerate(templates):
        by_template[template].append(idx)

    annotations: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = [
        {"line_index": idx, "event": "other", "ap_id": "", "client_id": "", "template_hash": ""}
        for idx in range(len(lines))
    ]
    representative_count = int(config.get("representative_lines", 3))
    repair_attempts = int(config.get("regex_repair_attempts", 1))

    ordered = sorted(by_template.items(), key=lambda item: (-len(item[1]), item[1][0]))
    for template_number, (template, indexes) in enumerate(ordered, start=1):
        template_hash = hashlib.sha256(template.encode("utf-8", errors="replace")).hexdigest()
        representatives = _representatives(indexes, lines, representative_count)
        examples = [line for _, line in representatives]
        examples_text = "\n".join(f"- `{line}`" for line in examples)
        classification_prompt = _render_prompt(
            "classify_template.txt", template=template, examples=examples_text
        )
        response = llm.complete(
            [{"role": "user", "content": classification_prompt}],
            {
                "stage": "classify_template",
                "baseline": baseline,
                "dataset": dataset,
                "template_hash": template_hash,
            },
        )
        try:
            parsed = _parse_json_object(response)
            event = _normalize_event(parsed.get("event"))
            reason = str(parsed.get("reason", ""))
        except Exception as exc:
            event, reason = "other", f"invalid classification response: {exc}"

        regex_text = ""
        regex_error = ""
        regex_metadata: dict[str, Any] = {}
        if event != "other":
            field_prompt = _render_prompt(
                "extract_fields.txt",
                event=event,
                template=template,
                examples=examples_text,
            )
            for attempt in range(repair_attempts + 1):
                prompt = field_prompt
                if attempt and regex_error:
                    prompt += (
                        "\n\nThe previous expression was invalid or failed representative lines. "
                        f"Correct it. Validation error: {regex_error}"
                    )
                field_response = llm.complete(
                    [{"role": "user", "content": prompt}],
                    {
                        "stage": "extract_fields" if attempt == 0 else "repair_field_regex",
                        "baseline": baseline,
                        "dataset": dataset,
                        "template_hash": template_hash,
                    },
                )
                try:
                    regex_metadata = _parse_json_object(field_response)
                    candidate = str(regex_metadata.get("regex", ""))
                    compiled = re.compile(candidate)
                    if not candidate.startswith("^") and not candidate.startswith("\\A"):
                        raise ValueError("regex is not start-anchored")
                    group_names = set(compiled.groupindex)
                    unexpected = {
                        name for name in group_names
                        if name not in {"ap_id", "client_id"} and not GROUP_SUFFIX_RE.fullmatch(name)
                    }
                    if unexpected:
                        raise ValueError(f"unsupported named groups: {sorted(unexpected)}")
                    if not any(name.startswith("ap_") for name in group_names):
                        raise ValueError("regex has no AP identifier group")
                    if not any(name.startswith("client_") for name in group_names):
                        raise ValueError("regex has no client identifier group")
                    failures = [line for line in examples if compiled.fullmatch(line) is None]
                    if failures:
                        raise ValueError(f"regex failed {len(failures)}/{len(examples)} representative lines")
                    regex_text, regex_error = candidate, ""
                    break
                except Exception as exc:
                    regex_error = str(exc)

        annotation = {
            "template_number": template_number,
            "template_hash": template_hash,
            "template": template,
            "occurrence": len(indexes),
            "representative_line_indexes": [idx for idx, _ in representatives],
            "event": event,
            "classification_reason": reason,
            "regex": regex_text,
            "regex_error": regex_error,
            "ap_source": str(regex_metadata.get("ap_source", "")),
            "client_source": str(regex_metadata.get("client_source", "")),
        }
        annotations.append(annotation)
        _append_jsonl(annotation_path, annotation)

        compiled = re.compile(regex_text) if regex_text else None
        for idx in indexes:
            ap_id = client_id = ""
            if event != "other" and compiled:
                match = compiled.fullmatch(lines[idx])
                if match:
                    groups = match.groupdict()
                    ap_id = _pick_extracted_identifier(groups, "ap")
                    client_id = _pick_extracted_identifier(groups, "client")
            predictions[idx] = {
                "line_index": idx,
                "event": event,
                "ap_id": ap_id,
                "client_id": client_id,
                "template_hash": template_hash,
            }
    return annotations, predictions


def _load_csv_ground_truth(path: Path, line_limit: int) -> dict[int, dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            try:
                idx = int(row.get("OriginalLineIdx", ""))
            except ValueError:
                continue
            if idx >= line_limit:
                continue
            event = _normalize_event(row.get("Action"))
            if event == "other":
                continue
            ap_candidates = {_normalize_id(row.get("ApId")), _normalize_id(row.get("BSSID"))} - {""}
            client_candidates = {_normalize_id(row.get("ClientId"))} - {""}
            if idx in by_index:
                raise ValueError(f"duplicate ground-truth OriginalLineIdx {idx} in {path}")
            by_index[idx] = {
                "event": event,
                "ap_candidates": ap_candidates,
                "client_candidates": client_candidates,
            }
    return by_index


HS_CONNECT_ACTIONS = {
    "auth-req",
    "auth-resp",
    "assoc-req",
    "assoc-resp",
    "client-ip-detected",
    "reassoc-req",
    "reassoc-resp",
    "client-authentication",
}
HS_DISCONNECT_ACTIONS = {"client-leave-wtp", "client-disconnected-by-wtp"}


def _hs_rule_row(row: list[str]) -> tuple[str, str, str, str] | None:
    if len(row) < 6:
        return None
    raw_action = row[2].strip()
    if raw_action in HS_CONNECT_ACTIONS:
        event = "connect"
    elif raw_action in HS_DISCONNECT_ACTIONS:
        event = "disconnect"
    else:
        return None
    macs = MAC_RE.findall(row[3])
    if len(macs) != 1:
        return None
    return row[1].strip(), event, _normalize_id(row[5]), _normalize_id(macs[0])


def _load_hs_ground_truth(
    raw_path: Path,
    gt_path: Path,
    lines: list[str],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for idx, line in enumerate(lines):
        parsed = _hs_rule_row(next(csv.reader([line])))
        if not parsed:
            continue
        _, event, ap_id, client_id = parsed
        by_index[idx] = {
            "event": event,
            "ap_candidates": {ap_id} if ap_id else set(),
            "client_candidates": {client_id} if client_id else set(),
        }

    # Validate the reconstructed historical rule against the saved GT without
    # relying on its per-day-reset OriginalLineIdx.
    raw_counter: Counter[tuple[str, str, str]] = Counter()
    with raw_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            parsed = _hs_rule_row(row)
            if parsed:
                timestamp, event, ap_id, _ = parsed
                raw_counter[(timestamp, event, ap_id)] += 1
    gt_counter: Counter[tuple[str, str, str]] = Counter()
    with gt_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            gt_counter[
                (
                    str(row.get("DateTime", "")).strip(),
                    _normalize_event(row.get("Action")),
                    _normalize_id(row.get("ApId")),
                )
            ] += 1
    if raw_counter != gt_counter:
        raise ValueError(
            "HS historical-rule reconstruction does not match saved GT: "
            f"raw-only={sum((raw_counter - gt_counter).values())}, "
            f"gt-only={sum((gt_counter - raw_counter).values())}"
        )
    audit = {
        "type": "hs_historical_rule",
        "prefix_event_lines": len(by_index),
        "full_rule_events": sum(raw_counter.values()),
        "saved_gt_events": sum(gt_counter.values()),
        "validation_key": "(timestamp, normalized event, AP ID) multiset",
        "full_multiset_equal": True,
        "client_evaluation": "raw MAC before the saved GT pseudonym transform",
    }
    return by_index, audit


def _evaluate(
    predictions: list[dict[str, Any]],
    gt: dict[int, dict[str, Any]],
    lines: list[str],
) -> dict[str, Any]:
    pred_by_idx = {int(row["line_index"]): row for row in predictions}
    class_metrics: dict[str, Any] = {}
    for label in EVENTS:
        tp = fp = fn = 0
        for idx in range(len(lines)):
            expected = gt.get(idx, {}).get("event", "other")
            predicted = _normalize_event(pred_by_idx.get(idx, {}).get("event"))
            tp += expected == label and predicted == label
            fp += expected != label and predicted == label
            fn += expected == label and predicted != label
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    macro_f1 = sum(class_metrics[label]["f1"] for label in EVENTS) / len(EVENTS)

    gt_event_pairs = {(idx, data["event"]) for idx, data in gt.items()}
    pred_event_pairs = {
        (idx, _normalize_event(row.get("event")))
        for idx, row in pred_by_idx.items()
        if _normalize_event(row.get("event")) != "other"
    }
    event_tp = len(gt_event_pairs & pred_event_pairs)
    event_fp = len(pred_event_pairs - gt_event_pairs)
    event_fn = len(gt_event_pairs - pred_event_pairs)
    legacy_precision = _safe_div(event_tp, event_tp + event_fp)
    legacy_recall = _safe_div(event_tp, event_tp + event_fn)

    strict_correct = {"ap": 0, "client": 0}
    legacy_correct = {"ap": 0, "client": 0, "event": event_tp}
    nonempty_totals = {"ap": 0, "client": 0}
    partial_only = {"ap": 0, "client": 0}
    for idx, data in gt.items():
        pred = pred_by_idx.get(idx, {})
        predicted_values = {
            "ap": _normalize_id(pred.get("ap_id")),
            "client": _normalize_id(pred.get("client_id")),
        }
        candidate_sets = {
            "ap": data["ap_candidates"],
            "client": data["client_candidates"],
        }
        for field in ("ap", "client"):
            candidates = candidate_sets[field]
            predicted = predicted_values[field]
            exact = any(predicted == candidate for candidate in candidates) if candidates else predicted == ""
            partial = any(_partial_match(predicted, candidate) for candidate in candidates) if candidates else predicted == ""
            strict_correct[field] += exact
            if candidates:
                nonempty_totals[field] += 1
                legacy_correct[field] += partial
            partial_only[field] += partial and not exact

    gt_total = len(gt)
    strict_ap = _safe_div(strict_correct["ap"], gt_total)
    strict_client = _safe_div(strict_correct["client"], gt_total)
    legacy_ap = _safe_div(legacy_correct["ap"], nonempty_totals["ap"])
    legacy_client = _safe_div(legacy_correct["client"], nonempty_totals["client"])
    flag_acc = _safe_div(event_tp, gt_total)
    return {
        "line_count": len(lines),
        "gt_event_lines": gt_total,
        "event_macro_f1_3class": macro_f1,
        "event_class_metrics": class_metrics,
        "legacy_event_set_f1": _f1(legacy_precision, legacy_recall),
        "legacy_event_precision": legacy_precision,
        "legacy_event_recall": legacy_recall,
        "strict_fea_paper_definition": (strict_ap + strict_client) / 2,
        "strict_ap_accuracy": strict_ap,
        "strict_client_accuracy": strict_client,
        "line_index_style_fea_with_event_and_partial": (legacy_ap + legacy_client + flag_acc) / 3,
        "partial_match_ap_accuracy": legacy_ap,
        "partial_match_client_accuracy": legacy_client,
        "event_flag_accuracy": flag_acc,
        "partial_only_matches": partial_only,
    }


def _write_templates(path: Path, lines: list[str], templates: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["line_index", "content", "event_template"])
        writer.writeheader()
        for idx, (line, template) in enumerate(zip(lines, templates)):
            writer.writerow({"line_index": idx, "content": line, "event_template": template})


def _read_templates(path: Path, lines: list[str]) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(lines):
        raise ValueError(f"template source {path} has {len(rows)} rows for {len(lines)} input lines")
    templates: list[str] = []
    for idx, (row, line) in enumerate(zip(rows, lines)):
        if int(row["line_index"]) != idx or row["content"] != line:
            raise ValueError(f"template source {path} does not match input at line {idx}")
        templates.append(str(row["event_template"]))
    return templates


def _run_one(
    baseline: str,
    dataset: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    reuse_templates_root: Path | None,
) -> dict[str, Any]:
    name = str(dataset["name"])
    line_limit = int(config["line_limit"])
    log_path = (ROOT / dataset["log_path"]).resolve()
    gt_path = (ROOT / dataset["gt_path"]).resolve()
    if not log_path.is_file() or not gt_path.is_file():
        raise FileNotFoundError(f"missing local data for {name}: {log_path} / {gt_path}")
    lines = _read_lines(log_path, line_limit)
    run_dir = output_root / baseline / name
    run_dir.mkdir(parents=True, exist_ok=True)
    calls_path = run_dir / "llm_calls.jsonl"
    if calls_path.exists():
        raise FileExistsError(f"refusing to append to existing call log: {calls_path}")
    llm = TrackedLLM(config["llm"], calls_path)
    started = time.monotonic()
    diagnostic = run_dir / "upstream_diagnostic.log"
    reused_path = None
    if reuse_templates_root:
        candidate = reuse_templates_root / baseline / name / "templates.csv"
        if candidate.is_file():
            templates = _read_templates(candidate, lines)
            reused_path = candidate
        elif baseline == "lilac" and name in {"Wilson", "University"}:
            raise FileNotFoundError(f"requested template reuse but source is missing: {candidate}")
    if reused_path:
        diagnostic.write_text(f"Reused generic templates from {reused_path}\n", encoding="utf-8")
    elif baseline == "lilac":
        templates = _run_lilac(
            lines, config["template_parsers"]["lilac"], llm, diagnostic
        )
    elif baseline == "logbatcher":
        templates = _run_logbatcher(
            lines,
            name,
            config["template_parsers"]["logbatcher"],
            llm,
            run_dir / "upstream_output",
            diagnostic,
        )
    else:
        raise ValueError(baseline)
    _write_templates(run_dir / "templates.csv", lines, templates)

    annotations, predictions = _postprocess_templates(
        baseline,
        name,
        lines,
        templates,
        config["postprocessing"],
        llm,
        run_dir / "template_annotations.jsonl",
    )
    for row in predictions:
        _append_jsonl(run_dir / "predictions.jsonl", row)
    if dataset.get("gt_loader") == "hs_historical_rule":
        gt, gt_audit = _load_hs_ground_truth(log_path, gt_path, lines)
    else:
        gt = _load_csv_ground_truth(gt_path, line_limit)
        gt_audit = {
            "type": "saved_csv_line_index",
            "prefix_event_lines": len(gt),
            "original_line_index_unique": True,
        }
    metrics = _evaluate(predictions, gt, lines)
    metrics["ground_truth_alignment"] = gt_audit
    _write_json(run_dir / "metrics.json", metrics)

    upstream_name = "LILAC" if baseline == "lilac" else "LogBatcher"
    metadata = {
        "baseline": baseline,
        "dataset": name,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "line_limit": line_limit,
        "sampling": "first N physical lines, zero-based line indexes",
        "raw_prefix_sha256": _sha256_file_prefix(log_path, line_limit),
        "ground_truth_sha256": _sha256_file(gt_path),
        "upstream_commit": _git_head(UPSTREAM_DIR / upstream_name),
        "model": llm.model,
        "temperature": llm.temperature,
        "base_url_host": re.sub(r"^(https?://[^/]+).*$", r"\1", llm.base_url or "default"),
        "template_count": len(set(templates)),
        "event_template_count": sum(item["event"] != "other" for item in annotations),
        "template_source": (
            {
                "mode": "reused",
                "path": str(reused_path.relative_to(ROOT)),
                "sha256": _sha256_file(reused_path),
            }
            if reused_path
            else {"mode": "generated_in_run"}
        ),
        "llm_usage": llm.stats(),
    }
    _write_json(run_dir / "run_metadata.json", metadata)
    return {**metadata, **metrics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/template_baselines_2000.json")
    parser.add_argument("--baseline", action="append", choices=("lilac", "logbatcher"))
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--run-id")
    parser.add_argument("--line-limit", type=int, help="temporary smoke-test override")
    parser.add_argument(
        "--reuse-templates-run",
        help="reuse validated generic templates from outputs/template_baselines/<run-id>",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.line_limit is not None:
        if args.line_limit <= 0:
            raise ValueError("--line-limit must be positive")
        config["line_limit"] = args.line_limit
    selected_baselines = args.baseline or ["lilac", "logbatcher"]
    selected_datasets = [
        item for item in config["datasets"] if not args.dataset or item["name"] in args.dataset
    ]
    if args.dataset and len(selected_datasets) != len(set(args.dataset)):
        known = {item["name"] for item in config["datasets"]}
        raise ValueError(f"unknown dataset(s): {sorted(set(args.dataset) - known)}")

    for name in ("LILAC", "LogBatcher"):
        if not (UPSTREAM_DIR / name / ".git").is_dir():
            raise FileNotFoundError("run scripts/prepare_template_baselines.sh first")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = ROOT / "outputs" / "template_baselines" / run_id
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(
        output_root / "experiment.json",
        {
            "config_path": str(config_path.relative_to(ROOT)),
            "config_sha256": _sha256_file(config_path),
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    summary: list[dict[str, Any]] = []
    reuse_root = (
        ROOT / "outputs" / "template_baselines" / args.reuse_templates_run
        if args.reuse_templates_run
        else None
    )
    for baseline in selected_baselines:
        for dataset in selected_datasets:
            print(f"Running {baseline}+ on {dataset['name']} ({config['line_limit']} lines)...", flush=True)
            summary.append(_run_one(baseline, dataset, config, output_root, reuse_root))
    _write_json(output_root / "summary.json", summary)
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "baseline",
            "dataset",
            "line_limit",
            "template_count",
            "event_template_count",
            "event_macro_f1_3class",
            "legacy_event_set_f1",
            "strict_fea_paper_definition",
            "line_index_style_fea_with_event_and_partial",
            "elapsed_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary)
    print(f"Results: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
