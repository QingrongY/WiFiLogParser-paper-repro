#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


EVENT_CONNECT = "connect"
EVENT_DISCONNECT = "disconnect"


@dataclass(frozen=True)
class EvalMetrics:
    matching: str
    event_precision: float
    event_recall: float
    event_f1: float
    field_extraction_accuracy: float
    event_tp: int
    event_fp: int
    event_fn: int
    field_tp: int
    gt_events: int
    pred_events: int


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _f1(precision: float, recall: float) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _normalize_event(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {EVENT_CONNECT, "1", "+1"}:
        return EVENT_CONNECT
    if text in {EVENT_DISCONNECT, "-1"}:
        return EVENT_DISCONNECT
    return ""


def _normalize_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return ""
    compact = text.replace("-", ":").replace(".", "").replace(" ", "")
    is_hex = all(ch in "0123456789abcdef" for ch in compact.replace(":", ""))
    if len(compact) == 12 and ":" not in compact and is_hex:
        return ":".join(compact[i : i + 2] for i in range(0, 12, 2))
    return text


def _compact_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _partial_field_match(predicted_value: str, gt_value: str) -> bool:
    pred = _normalize_id(predicted_value)
    gt = _normalize_id(gt_value)
    if not pred or not gt:
        return False
    if pred == gt:
        return True
    if gt in pred:
        return True
    if pred in gt or gt in pred:
        return True

    pred_compact = _compact_token(pred)
    gt_compact = _compact_token(gt)
    if not pred_compact or not gt_compact:
        return False
    if gt_compact in pred_compact:
        return True

    return (
        pred_compact == gt_compact
        or pred_compact in gt_compact
        or gt_compact in pred_compact
    )


def _max_partial_match_count(gt_values: list[str], pred_values: list[str]) -> int:
    if not gt_values or not pred_values:
        return 0

    used = [False] * len(pred_values)
    matched = 0

    order = sorted(range(len(gt_values)), key=lambda i: len(str(gt_values[i] or "")), reverse=True)
    for i in order:
        gt_value = gt_values[i]
        best_idx: int | None = None
        best_score = -1
        for j, pred_value in enumerate(pred_values):
            if used[j]:
                continue
            if not _partial_field_match(pred_value, gt_value):
                continue
            score = 2 if _normalize_id(pred_value) == _normalize_id(gt_value) else 1
            if score > best_score:
                best_score = score
                best_idx = j
                if score == 2:
                    break
        if best_idx is not None:
            used[best_idx] = True
            matched += 1

    return matched


def _max_partial_pair_match_count(
    gt_pairs: list[tuple[str, str]],
    pred_pairs: list[tuple[str, str]],
) -> int:
    if not gt_pairs or not pred_pairs:
        return 0

    used = [False] * len(pred_pairs)
    matched = 0

    order = sorted(
        range(len(gt_pairs)),
        key=lambda i: len(str(gt_pairs[i][0] or "")) + len(str(gt_pairs[i][1] or "")),
        reverse=True,
    )
    for i in order:
        gt_ap, gt_client = gt_pairs[i]
        best_idx: int | None = None
        best_score = -1
        for j, (pred_ap, pred_client) in enumerate(pred_pairs):
            if used[j]:
                continue
            if not _partial_field_match(pred_ap, gt_ap):
                continue
            if not _partial_field_match(pred_client, gt_client):
                continue
            exact_ap = _normalize_id(pred_ap) == _normalize_id(gt_ap)
            exact_client = _normalize_id(pred_client) == _normalize_id(gt_client)
            score = int(exact_ap) + int(exact_client)
            if score > best_score:
                best_score = score
                best_idx = j
                if score == 2:
                    break
        if best_idx is not None:
            used[best_idx] = True
            matched += 1

    return matched


def _normalize_timestamp(value: Any) -> str:
    if value is None:
        return ""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    if hasattr(ts, "tz") and ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _count_lines(file_path: Path) -> int:
    count = 0
    with file_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for _ in handle:
            count += 1
    return count


def _load_gt(gt_path: Path) -> pd.DataFrame:
    return pd.read_csv(gt_path, dtype=str, keep_default_na=False, engine="python", on_bad_lines="skip")


def _build_line_event_predictions(engine: Any, log_path: Path) -> list[dict[str, Any]]:
    runner = getattr(engine, "_runner", None)
    if runner is None:
        return []

    rule = getattr(runner, "timestamp_rule", None)
    if rule is None:
        return []

    standard_keys: list[str] = []
    try:
        standard_keys = list(getattr(runner, "_standard_keys", []) or [])
    except Exception:
        standard_keys = []

    event_rows: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_index, raw in enumerate(handle):
            raw_line = str(raw).rstrip("\n").rstrip("\r")
            dt, content = runner.timestamp_agent.split(raw_line, rule)
            if dt is None or content is None:
                continue
            content = str(content).strip()
            if not content:
                continue

            preprocessed = runner._preprocess_content(content)
            if standard_keys:
                try:
                    preprocessed = runner.clusterer._standardize_json_in_log(preprocessed, standard_keys)
                except Exception:
                    pass

            info = runner.cache_manager.get_cluster_info_for_preprocessed_log(preprocessed)
            connect_flag = int(info[2]) if info else 0
            event = _normalize_event(connect_flag)
            if event not in {EVENT_CONNECT, EVENT_DISCONNECT}:
                continue

            ts = _normalize_timestamp(dt)
            event_rows.append({"line_index": int(line_index), "timestamp": ts, "event": event})

    return event_rows


def _events_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        event = _normalize_event(rec.get("event"))
        if event not in {EVENT_CONNECT, EVENT_DISCONNECT}:
            continue
        metadata = rec.get("metadata") or {}
        line_index = metadata.get("line_index")
        try:
            idx = int(line_index) if line_index is not None else None
        except (TypeError, ValueError):
            idx = None
        rows.append(
            {
                "line_index": idx,
                "timestamp": _normalize_timestamp(rec.get("timestamp")),
                "event": event,
            }
        )
    return rows


def _eval_line_index(
    event_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    gt_df: pd.DataFrame,
    line_count: int,
    include_client_in_fea: bool = True,
) -> EvalMetrics:
    if "OriginalLineIdx" not in gt_df.columns:
        raise ValueError("Ground truth is missing column: OriginalLineIdx")

    idx_series = pd.to_numeric(gt_df["OriginalLineIdx"], errors="coerce")
    gt_df = gt_df[idx_series.notna()].copy()
    gt_df["line_index"] = idx_series[idx_series.notna()].astype(int)
    gt_df = gt_df[gt_df["line_index"] < int(line_count)]

    if bool(gt_df["line_index"].duplicated().any()):
        raise ValueError("line_index matching requires unique OriginalLineIdx in ground truth")

    gt_by_idx: dict[int, dict[str, Any]] = {}
    for row in gt_df.itertuples(index=False):
        event = _normalize_event(getattr(row, "Action", ""))
        if event not in {EVENT_CONNECT, EVENT_DISCONNECT}:
            continue
        idx = int(getattr(row, "line_index"))
        ap_candidates = {
            _normalize_id(getattr(row, "ApId", "")),
            _normalize_id(getattr(row, "BSSID", "")),
        }
        ap_candidates.discard("")
        client_candidates = {
            _normalize_id(getattr(row, "ClientId", "")),
        }
        client_candidates.discard("")
        gt_by_idx[idx] = {
            "event": event,
            "ap_candidates": ap_candidates,
            "client_candidates": client_candidates,
        }

    pred_by_idx: dict[int, tuple[str, str, str]] = {}
    for rec in records:
        metadata = rec.get("metadata") or {}
        line_index = metadata.get("line_index")
        if line_index is None:
            continue
        try:
            idx = int(line_index)
        except (TypeError, ValueError):
            continue
        event = _normalize_event(rec.get("event"))
        if event not in {EVENT_CONNECT, EVENT_DISCONNECT}:
            continue
        pred_by_idx[idx] = (event, _normalize_id(rec.get("ap_id")), _normalize_id(rec.get("client_id")))

    gt_events = {(idx, data["event"]) for idx, data in gt_by_idx.items()}
    pred_events: set[tuple[int, str]] = set()
    for row in event_rows:
        line_index = row.get("line_index")
        if line_index is None:
            continue
        try:
            idx = int(line_index)
        except (TypeError, ValueError):
            continue
        event = _normalize_event(row.get("event"))
        if event not in {EVENT_CONNECT, EVENT_DISCONNECT}:
            continue
        pred_events.add((idx, event))

    event_tp = len(gt_events & pred_events)
    event_fp = len(pred_events - gt_events)
    event_fn = len(gt_events - pred_events)

    precision = _safe_div(event_tp, event_tp + event_fp)
    recall = _safe_div(event_tp, event_tp + event_fn)
    event_f1 = _f1(precision, recall)

    pred_event_by_idx: dict[int, str] = {}
    for idx, event in pred_events:
        pred_event_by_idx[idx] = event

    flag_correct = 0
    for idx, gt_data in gt_by_idx.items():
        if pred_event_by_idx.get(idx) == gt_data["event"]:
            flag_correct += 1

    ap_total = 0
    ap_correct = 0
    client_total = 0
    client_correct = 0
    field_tp = 0
    for idx, gt_data in gt_by_idx.items():
        pred_data = pred_by_idx.get(idx)
        pred_event = pred_ap = pred_client = ""
        if pred_data:
            pred_event, pred_ap, pred_client = pred_data

        ap_candidates = gt_data["ap_candidates"]
        client_candidates = gt_data["client_candidates"]

        ap_ok = any(_partial_field_match(pred_ap, candidate) for candidate in ap_candidates)
        client_ok = any(_partial_field_match(pred_client, candidate) for candidate in client_candidates)

        if ap_candidates:
            ap_total += 1
            if ap_ok:
                ap_correct += 1

        if client_candidates:
            client_total += 1
            if client_ok:
                client_correct += 1

        if pred_event != gt_data["event"]:
            continue

        if include_client_in_fea:
            field_match_ok = bool(ap_candidates) and bool(client_candidates) and ap_ok and client_ok
        else:
            field_match_ok = bool(ap_candidates) and ap_ok

        if field_match_ok:
            field_tp += 1

    gt_total = len(gt_by_idx)
    flag_acc = _safe_div(flag_correct, gt_total)
    ap_acc = _safe_div(ap_correct, ap_total)
    client_acc = _safe_div(client_correct, client_total)
    if include_client_in_fea:
        fea = (ap_acc + client_acc + flag_acc) / 3.0
    else:
        fea = (ap_acc + flag_acc) / 2.0

    return EvalMetrics(
        matching="line_index",
        event_precision=precision,
        event_recall=recall,
        event_f1=event_f1,
        field_extraction_accuracy=fea,
        event_tp=event_tp,
        event_fp=event_fp,
        event_fn=event_fn,
        field_tp=field_tp,
        gt_events=gt_total,
        pred_events=len(pred_events),
    )


def _eval_record(
    event_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    gt_df: pd.DataFrame,
    include_client_in_fea: bool = True,
    field_metrics_on_extracted_only: bool = False,
) -> EvalMetrics:
    gt_event_counter: Counter[tuple[str, str]] = Counter()
    gt_ap_by_key: dict[tuple[str, str], list[str]] = {}
    gt_client_by_key: dict[tuple[str, str], list[str]] = {}
    gt_pair_by_key: dict[tuple[str, str], list[tuple[str, str]]] = {}

    for row in gt_df.itertuples(index=False):
        event = _normalize_event(getattr(row, "Action", ""))
        if event not in {EVENT_CONNECT, EVENT_DISCONNECT}:
            continue
        ts = _normalize_timestamp(getattr(row, "DateTime", ""))
        if not ts:
            continue
        ap_id = _normalize_id(getattr(row, "ApId", ""))
        client_id = _normalize_id(getattr(row, "ClientId", ""))
        key = (ts, event)
        gt_event_counter[key] += 1
        if ap_id:
            gt_ap_by_key.setdefault(key, []).append(ap_id)
        if client_id:
            gt_client_by_key.setdefault(key, []).append(client_id)
        if ap_id and client_id:
            gt_pair_by_key.setdefault(key, []).append((ap_id, client_id))

    pred_event_counter: Counter[tuple[str, str]] = Counter()
    pred_ap_by_key: dict[tuple[str, str], list[str]] = {}
    pred_client_by_key: dict[tuple[str, str], list[str]] = {}
    pred_pair_by_key: dict[tuple[str, str], list[tuple[str, str]]] = {}

    for row in event_rows:
        event = _normalize_event(row.get("event"))
        if event not in {EVENT_CONNECT, EVENT_DISCONNECT}:
            continue
        ts = _normalize_timestamp(row.get("timestamp"))
        if not ts:
            continue
        pred_event_counter[(ts, event)] += 1

    for rec in records:
        event = _normalize_event(rec.get("event"))
        if event not in {EVENT_CONNECT, EVENT_DISCONNECT}:
            continue
        ts = _normalize_timestamp(rec.get("timestamp"))
        if not ts:
            continue
        ap_id = _normalize_id(rec.get("ap_id"))
        client_id = _normalize_id(rec.get("client_id"))
        key = (ts, event)
        if ap_id:
            pred_ap_by_key.setdefault(key, []).append(ap_id)
        if client_id:
            pred_client_by_key.setdefault(key, []).append(client_id)
        if ap_id and client_id:
            pred_pair_by_key.setdefault(key, []).append((ap_id, client_id))

    event_tp = sum((gt_event_counter & pred_event_counter).values())
    event_fp = sum((pred_event_counter - gt_event_counter).values())
    event_fn = sum((gt_event_counter - pred_event_counter).values())

    precision = _safe_div(event_tp, event_tp + event_fp)
    recall = _safe_div(event_tp, event_tp + event_fn)
    event_f1 = _f1(precision, recall)

    field_tp = 0
    for key, gt_pairs in gt_pair_by_key.items():
        field_tp += _max_partial_pair_match_count(gt_pairs, pred_pair_by_key.get(key, []))

    gt_total = sum(gt_event_counter.values())
    flag_acc = _safe_div(event_tp, gt_total)
    if field_metrics_on_extracted_only:
        ap_total = sum(min(len(gt_values), len(pred_ap_by_key.get(key, []))) for key, gt_values in gt_ap_by_key.items())
        client_total = sum(
            min(len(gt_values), len(pred_client_by_key.get(key, []))) for key, gt_values in gt_client_by_key.items()
        )
    else:
        ap_total = sum(len(values) for values in gt_ap_by_key.values())
        client_total = sum(len(values) for values in gt_client_by_key.values())

    ap_correct = 0
    for key, gt_values in gt_ap_by_key.items():
        ap_correct += _max_partial_match_count(gt_values, pred_ap_by_key.get(key, []))

    client_correct = 0
    for key, gt_values in gt_client_by_key.items():
        client_correct += _max_partial_match_count(gt_values, pred_client_by_key.get(key, []))

    ap_acc = _safe_div(ap_correct, ap_total)
    client_acc = _safe_div(client_correct, client_total)
    if include_client_in_fea:
        fea = (ap_acc + client_acc + flag_acc) / 3.0
    else:
        fea = (ap_acc + flag_acc) / 2.0
        field_tp = ap_correct

    return EvalMetrics(
        matching="record",
        event_precision=precision,
        event_recall=recall,
        event_f1=event_f1,
        field_extraction_accuracy=fea,
        event_tp=event_tp,
        event_fp=event_fp,
        event_fn=event_fn,
        field_tp=field_tp,
        gt_events=gt_total,
        pred_events=sum(pred_event_counter.values()),
    )


def _set_env_defaults(config: dict[str, Any]) -> None:
    llm_cfg = config.get("llm") or {}
    preserve_if_present = {"primary_model", "base_url"}

    mapping = {
        "primary_model": "LLM_PRIMARY_MODEL",
        "base_url": "LLM_BASE_URL",
        "batch_size": "LLM_BATCH_SIZE",
        "chunk_size": "LLM_CHUNK_SIZE",
        "min_cluster_size": "LLM_MIN_CLUSTER_SIZE",
        "disable_fallback": "LLM_DISABLE_FALLBACK",
        "request_timeout_seconds": "LLM_REQUEST_TIMEOUT_SECONDS",
        "max_retries": "LLM_MAX_RETRIES",
    }
    for cfg_key, env_key in mapping.items():
        value = llm_cfg.get(cfg_key)
        if value is None:
            continue
        existing = str(os.getenv(env_key, "")).strip()
        if cfg_key in preserve_if_present and existing:
            continue
        os.environ[env_key] = str(value)

    fallback_models = llm_cfg.get("fallback_models")
    if isinstance(fallback_models, list):
        os.environ["LLM_FALLBACK_MODELS"] = ",".join(str(v) for v in fallback_models if str(v).strip())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WiFiLogParser main experiments.")
    parser.add_argument("--config", default="configs/main_experiment.json", help="Experiment config JSON path.")
    parser.add_argument("--output-root", default="outputs", help="Output directory root.")
    parser.add_argument("--dataset", action="append", dest="datasets", help="Dataset name filter (repeatable).")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    config_path = (ROOT / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    from apps_v2.logparser.services.wifi_log_parser.engine import WiFiLogParserEngine

    _set_env_defaults(config)

    if not (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise ValueError("LLM_API_KEY (or OPENAI_API_KEY) is required. Set it in .env or environment variables.")

    dataset_configs = list(config.get("datasets") or [])
    if args.datasets:
        keep = {name.strip() for name in args.datasets if name and name.strip()}
        dataset_configs = [d for d in dataset_configs if str(d.get("name", "")).strip() in keep]

    if not dataset_configs:
        raise ValueError("No dataset selected. Check --dataset names or config file.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (ROOT / args.output_root).resolve()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for dataset in dataset_configs:
        name = str(dataset.get("name", "")).strip()
        if not name:
            continue

        log_path = (ROOT / str(dataset["log_path"])).resolve()
        gt_path = (ROOT / str(dataset["gt_path"])).resolve()
        matching = str(dataset.get("matching", "line_index")).strip().lower()
        include_client_in_fea = not bool(dataset.get("exclude_client_in_fea", False))
        field_metrics_on_extracted_only = bool(dataset.get("field_metrics_on_extracted_only", False))

        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found for {name}: {log_path}")
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth not found for {name}: {gt_path}")

        dataset_out = run_dir / name
        dataset_out.mkdir(parents=True, exist_ok=True)

        line_count = _count_lines(log_path)
        print(f"[{name}] start | lines={line_count} | matching={matching}")

        started = time.perf_counter()
        engine = WiFiLogParserEngine()
        payload = engine.process_file(log_path)
        runtime_sec = time.perf_counter() - started

        records = list(payload.get("records", []))
        stats = dict(payload.get("stats", {}))
        event_rows = _build_line_event_predictions(engine, log_path)
        if not event_rows:
            event_rows = _events_from_records(records)

        gt_df = _load_gt(gt_path)
        if matching == "line_index":
            try:
                metrics = _eval_line_index(
                    event_rows,
                    records,
                    gt_df,
                    line_count,
                    include_client_in_fea=include_client_in_fea,
                )
            except ValueError:
                metrics = _eval_record(
                    event_rows,
                    records,
                    gt_df,
                    include_client_in_fea=include_client_in_fea,
                    field_metrics_on_extracted_only=field_metrics_on_extracted_only,
                )
        else:
            metrics = _eval_record(
                event_rows,
                records,
                gt_df,
                include_client_in_fea=include_client_in_fea,
                field_metrics_on_extracted_only=field_metrics_on_extracted_only,
            )

        _write_json(dataset_out / "stats.json", stats)
        _write_json(dataset_out / "metrics.json", asdict(metrics))
        _write_jsonl(dataset_out / "records.jsonl", records)

        row = {
            "dataset": name,
            "matching": metrics.matching,
            "event_f1": round(metrics.event_f1, 6),
            "event_precision": round(metrics.event_precision, 6),
            "event_recall": round(metrics.event_recall, 6),
            "field_extraction_accuracy": round(metrics.field_extraction_accuracy, 6),
            "gt_events": metrics.gt_events,
            "pred_events": metrics.pred_events,
            "event_tp": metrics.event_tp,
            "event_fp": metrics.event_fp,
            "event_fn": metrics.event_fn,
            "field_tp": metrics.field_tp,
            "predicted_events_total": len(event_rows),
            "records_total": len(records),
            "llm_calls": stats.get("llm_calls"),
            "total_tokens": stats.get("total_tokens"),
            "llm_time": stats.get("llm_time"),
            "runtime_sec": round(runtime_sec, 3),
            "log_path": str(log_path),
            "gt_path": str(gt_path),
        }
        summary_rows.append(row)
        print(
            f"[{name}] done | F1={row['event_f1']:.4f} | FEA={row['field_extraction_accuracy']:.4f} "
            f"| llm_calls={row['llm_calls']} | tokens={row['total_tokens']} | runtime={row['runtime_sec']}s"
        )

    if not summary_rows:
        raise ValueError("No experiments were executed. Check dataset names in config.")

    summary_csv = run_dir / "summary_main.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    _write_json(
        run_dir / "summary_main.json",
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "datasets": summary_rows,
        },
    )

    print(f"All done. Summary: {summary_csv}")


if __name__ == "__main__":
    main()
