from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bidkv.evidence.w1_decision_receipt import EvidenceError, build_receipt, load_trace


def _candidate(request_id: str, tokens: int, cost: float) -> dict:
    return {
        "request_id": request_id,
        "current_tokens": tokens,
        "priority": 0,
        "arrival_time_ms": 1.0,
        "num_prompt_tokens": 10,
        "num_computed_tokens": 20,
        "max_output_tokens": 20,
        "num_preemptions": 0,
        "observed_disruption_cost": cost,
    }


def _write_trace(tmp_path: Path) -> Path:
    log = tmp_path / "raw" / "pressure.jsonl"
    log.parent.mkdir(exist_ok=True)
    log.write_text("raw pressure event\n", encoding="utf-8")
    trace = {
        "schema_version": "bidkv.pressure-trace/v1",
        "trace_id": "real-run-1",
        "source": {
            "evidence_level": "real-serving",
            "repository_commit": "a" * 40,
            "runtime_commit": "b" * 40,
            "model": "Qwen3.5",
            "hardware": "Ascend 910B2 TP4",
            "raw_log": {
                "path": "raw/pressure.jsonl",
                "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            },
        },
        "events": [
            {
                "event_id": "event-1",
                "needed_tokens": 80,
                "candidates": [
                    _candidate("large-expensive", 100, 10.0),
                    _candidate("small-cheap", 80, 1.0),
                ],
            }
        ],
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace), encoding="utf-8")
    return path


def test_four_arms_share_candidate_universe_and_oracle_is_optimal(tmp_path: Path) -> None:
    path = _write_trace(tmp_path)
    trace, trace_hash = load_trace(path, root=tmp_path)

    receipt = build_receipt(trace, trace_hash)

    assert receipt["status"] == "VERIFIED"
    assert receipt["evidence_level"] == "decision-replay-only"
    event = receipt["events"][0]
    assert [arm["strategy"] for arm in event["arms"]] == [
        "stock",
        "static",
        "bidkv",
        "oracle",
    ]
    assert all(arm["capacity_satisfied"] for arm in event["arms"])
    oracle = next(arm for arm in event["arms"] if arm["strategy"] == "oracle")
    assert oracle["victims"] == ["small-cheap"]
    assert oracle["observed_disruption_cost"] == 1.0


def test_missing_or_changed_raw_log_fails_closed(tmp_path: Path) -> None:
    path = _write_trace(tmp_path)
    (tmp_path / "raw" / "pressure.jsonl").write_text("changed\n", encoding="utf-8")

    with pytest.raises(EvidenceError, match="missing or changed"):
        load_trace(path, root=tmp_path)


def test_synthetic_and_incomplete_candidate_data_are_rejected(tmp_path: Path) -> None:
    path = _write_trace(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source"]["evidence_level"] = "simulation"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvidenceError, match="real-serving"):
        load_trace(path, root=tmp_path)

    path = _write_trace(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["events"][0]["candidates"][0]["observed_disruption_cost"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    trace, trace_hash = load_trace(path, root=tmp_path)
    with pytest.raises(EvidenceError, match="candidate fields"):
        build_receipt(trace, trace_hash)


def test_duplicate_candidates_and_insufficient_capacity_are_rejected(tmp_path: Path) -> None:
    path = _write_trace(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["events"][0]["candidates"][1]["request_id"] = "large-expensive"
    path.write_text(json.dumps(payload), encoding="utf-8")
    trace, trace_hash = load_trace(path, root=tmp_path)
    with pytest.raises(EvidenceError, match="duplicate candidate"):
        build_receipt(trace, trace_hash)

    path = _write_trace(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["events"][0]["needed_tokens"] = 1000
    path.write_text(json.dumps(payload), encoding="utf-8")
    trace, trace_hash = load_trace(path, root=tmp_path)
    with pytest.raises(EvidenceError, match="required capacity"):
        build_receipt(trace, trace_hash)
