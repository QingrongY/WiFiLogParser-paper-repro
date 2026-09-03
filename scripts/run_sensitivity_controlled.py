#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_single_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row in {path}, found {len(rows)}")
    return rows[0]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the controlled Table 8 sensitivity suite.")
    parser.add_argument("--config", default="configs/sensitivity_controlled_rerun.json")
    parser.add_argument("--output-root", default="outputs/sensitivity_controlled_rerun")
    parser.add_argument("--results-csv", default="results/sensitivity_controlled_rerun/summary.csv")
    parser.add_argument("--only", action="append", help="Run only the named configuration (repeatable).")
    parser.add_argument("--resume", action="store_true", help="Keep completed rows and run only missing configurations.")
    args = parser.parse_args()

    suite_path = (ROOT / args.config).resolve()
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    defaults = dict(suite["defaults"])
    dataset = dict(suite["dataset"])
    configurations = list(suite["configurations"])
    if args.only:
        selected = set(args.only)
        configurations = [item for item in configurations if item["id"] in selected]
    results_csv_path = (ROOT / args.results_csv).resolve()
    rows: list[dict[str, Any]] = []
    if args.resume and results_csv_path.exists():
        with results_csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        completed = {row["configuration"] for row in rows}
        configurations = [item for item in configurations if item["id"] not in completed]
    if not configurations:
        print("No pending sensitivity configurations.", flush=True)
        return

    suite_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = (ROOT / args.output_root / suite_run_id).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    temp_config_dir = output_root / "configs"
    temp_config_dir.mkdir(parents=True, exist_ok=True)

    for position, item in enumerate(configurations, start=1):
        effective = dict(defaults)
        effective.update({key: value for key, value in item.items() if key in defaults})
        model = str(effective["model"])
        config_id = str(item["id"])
        run_config = {
            "llm": {
                "primary_model": model,
                "fallback_models": [model],
                "base_url": "https://api.aimlapi.com/v1",
                "batch_size": int(effective["diversity_sample_size"]),
                "chunk_size": int(effective["chunk_size"]),
                "min_cluster_size": int(effective["min_cluster_size"]),
                "disable_fallback": bool(effective["disable_fallback"]),
                "request_timeout_seconds": int(effective["request_timeout_seconds"]),
                "max_retries": int(effective["max_retries"]),
            },
            "datasets": [dataset],
        }
        config_path = temp_config_dir / f"{config_id}.json"
        config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

        config_output_root = output_root / config_id
        env = os.environ.copy()
        env["LLM_PRIMARY_MODEL"] = model
        env["LLM_FALLBACK_MODELS"] = model
        reasoning = effective.get("reasoning_effort")
        if reasoning:
            env["LLM_REASONING_EFFORT"] = str(reasoning)
        else:
            env.pop("LLM_REASONING_EFFORT", None)

        started_at = _utc_now()
        print(f"[{position}/{len(configurations)}] {config_id} started at {started_at}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_main_experiments.py"),
                "--config",
                str(config_path.relative_to(ROOT)),
                "--output-root",
                str(config_output_root),
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        finished_at = _utc_now()
        summaries = sorted(config_output_root.glob("*/summary_main.csv"))
        if len(summaries) != 1:
            raise ValueError(f"Expected one run summary for {config_id}, found {len(summaries)}")
        source_row = _read_single_row(summaries[0])
        actual_run_id = summaries[0].parent.name
        rows.append(
            {
                "configuration": config_id,
                "parameter": item["parameter"],
                "setting": item["setting"],
                "run_id": actual_run_id,
                "started_at_utc": started_at,
                "finished_at_utc": finished_at,
                "model_id": model,
                "diversity_sample_size": effective["diversity_sample_size"],
                "chunk_size": effective["chunk_size"],
                "reasoning_effort": reasoning or "off",
                "temperature": effective["temperature"],
                "maximum_completion_tokens": effective["maximum_completion_tokens"],
                "event_f1": source_row["event_f1"],
                "field_extraction_accuracy": source_row["field_extraction_accuracy"],
                "llm_calls": source_row["llm_calls"],
                "total_tokens": source_row["total_tokens"],
                "runtime_sec": source_row["runtime_sec"],
                "input_lines": 50000,
                "raw_sha256": _sha256((ROOT / dataset["log_path"]).resolve()),
                "ground_truth_sha256": _sha256((ROOT / dataset["gt_path"]).resolve()),
                "artifact_directory": str(summaries[0].parent.relative_to(ROOT)),
            }
        )
        _write_csv(results_csv_path, rows)
        print(f"[{position}/{len(configurations)}] {config_id} finished at {finished_at}", flush=True)

    manifest = {
        "suite_run_id": suite_run_id,
        "suite_config": str(suite_path.relative_to(ROOT)),
        "suite_config_sha256": _sha256(suite_path),
        "runner": "scripts/run_main_experiments.py",
        "orchestrator": "scripts/run_sensitivity_controlled.py",
        "result_count": len(rows),
        "results_csv": args.results_csv,
    }
    manifest_path = results_csv_path.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Controlled sensitivity suite complete: {ROOT / args.results_csv}", flush=True)


if __name__ == "__main__":
    main()
