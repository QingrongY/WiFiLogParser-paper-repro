from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any


def record_to_payload(record: Any) -> dict[str, Any]:
    occurred_at = getattr(record, 'occurred_at', None)
    timestamp = _timestamp_to_iso(occurred_at)
    return {
        'ap_id': getattr(record, 'ap_id', None),
        'client_id': getattr(record, 'client_id', None),
        'timestamp': timestamp,
        'event': getattr(record, 'event', None),
        'raw_log': getattr(record, 'raw_log', None),
        'metadata': getattr(record, 'metadata', None),
    }


def runner_output_to_payload(output: Any) -> dict[str, Any]:
    records = [record_to_payload(record) for record in getattr(output, 'records', [])]
    stats_raw = getattr(output, 'stats', {})
    if isinstance(stats_raw, dict):
        stats = dict(stats_raw)
    else:
        try:
            stats = asdict(stats_raw)
        except Exception:
            stats = {}
    clusters = getattr(output, 'cluster_details', {})
    return {
        'records': records,
        'stats': stats,
        'clusters': clusters,
    }


def streamed_payload(output: Any, *, streamed_count: int) -> dict[str, Any]:
    payload = runner_output_to_payload(output)
    payload['records_streamed'] = int(streamed_count)
    payload['records'] = []
    return payload


def _timestamp_to_iso(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    isoformat = getattr(value, 'isoformat', None)
    if callable(isoformat):
        try:
            return str(isoformat())
        except Exception:
            return None
    return None
