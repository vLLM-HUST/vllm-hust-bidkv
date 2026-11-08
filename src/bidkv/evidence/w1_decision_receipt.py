"""Fail-closed four-arm replay for the W1 KV-pressure decision contract."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

from bidkv.baselines.base import CompressionAction, RequestState
from bidkv.baselines.bidkv_strategy import BidKVStrategy
from bidkv.baselines.preempt_evict import PreemptEvictStrategy
from bidkv.baselines.static_random import StaticRandomStrategy

TRACE_SCHEMA = "bidkv.pressure-trace/v1"
RECEIPT_SCHEMA = "bidkv.w1-decision-receipt/v1"
HEX40 = frozenset("0123456789abcdef")


class EvidenceError(ValueError):
    """The frozen trace cannot be admitted as W1 decision evidence."""


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{name} must be an object")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{name} must be non-empty text")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise EvidenceError(f"{name} must be a positive integer")
    return value


def _finite_cost(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise EvidenceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise EvidenceError(f"{name} must be finite and non-negative")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit(value: Any, name: str) -> str:
    commit = _text(value, name)
    if len(commit) != 40 or any(character not in HEX40 for character in commit):
        raise EvidenceError(f"{name} must be lowercase 40-hex")
    return commit


def _source_path(root: Path, value: Any, name: str) -> Path:
    relative = Path(_text(value, name))
    if relative.is_absolute() or ".." in relative.parts:
        raise EvidenceError(f"{name} must be repository-relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise EvidenceError(f"{name} escapes the repository") from exc
    return path


def load_trace(path: Path, *, root: Path) -> tuple[dict[str, Any], str]:
    """Load and validate a real-serving pressure trace and its raw-log binding."""
    try:
        raw = path.read_bytes()
        trace = _object(json.loads(raw), "trace")
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read trace: {exc}") from exc
    if trace.get("schema_version") != TRACE_SCHEMA:
        raise EvidenceError("unsupported pressure-trace schema")
    _text(trace.get("trace_id"), "trace_id")
    source = _object(trace.get("source"), "source")
    expected_source = {
        "evidence_level",
        "repository_commit",
        "runtime_commit",
        "model",
        "hardware",
        "raw_log",
    }
    if set(source) != expected_source:
        raise EvidenceError(f"source fields must be exactly {sorted(expected_source)}")
    if source["evidence_level"] != "real-serving":
        raise EvidenceError("source.evidence_level must be real-serving")
    _commit(source["repository_commit"], "source.repository_commit")
    _commit(source["runtime_commit"], "source.runtime_commit")
    _text(source["model"], "source.model")
    _text(source["hardware"], "source.hardware")
    raw_log = _object(source["raw_log"], "source.raw_log")
    if set(raw_log) != {"path", "sha256"}:
        raise EvidenceError("source.raw_log must contain path and sha256")
    log_path = _source_path(root, raw_log["path"], "source.raw_log.path")
    expected_hash = _text(raw_log["sha256"], "source.raw_log.sha256")
    if len(expected_hash) != 64 or any(character not in HEX40 for character in expected_hash):
        raise EvidenceError("source.raw_log.sha256 must be lowercase 64-hex")
    if not log_path.is_file() or _sha256(log_path) != expected_hash:
        raise EvidenceError("raw pressure log is missing or changed")
    events = trace.get("events")
    if not isinstance(events, list) or not events:
        raise EvidenceError("events must be a non-empty array")
    return trace, hashlib.sha256(raw).hexdigest()


def _candidate(payload: Any, event_name: str) -> tuple[RequestState, float]:
    item = _object(payload, f"{event_name}.candidate")
    expected = {
        "request_id",
        "current_tokens",
        "priority",
        "arrival_time_ms",
        "num_prompt_tokens",
        "num_computed_tokens",
        "max_output_tokens",
        "num_preemptions",
        "observed_disruption_cost",
    }
    if set(item) != expected:
        raise EvidenceError(f"{event_name}.candidate fields must be exactly {sorted(expected)}")
    num_preemptions = item["num_preemptions"]
    if (
        not isinstance(num_preemptions, int)
        or isinstance(num_preemptions, bool)
        or num_preemptions < 0
    ):
        raise EvidenceError(f"{event_name}.num_preemptions must be a non-negative integer")
    request = RequestState(
        request_id=_text(item["request_id"], f"{event_name}.request_id"),
        current_tokens=_positive_int(item["current_tokens"], f"{event_name}.current_tokens"),
        priority=_finite_cost(item["priority"], f"{event_name}.priority"),
        arrival_time_ms=_finite_cost(item["arrival_time_ms"], f"{event_name}.arrival_time_ms"),
        num_prompt_tokens=_positive_int(
            item["num_prompt_tokens"], f"{event_name}.num_prompt_tokens"
        ),
        num_computed_tokens=_positive_int(
            item["num_computed_tokens"], f"{event_name}.num_computed_tokens"
        ),
        max_output_tokens=_positive_int(
            item["max_output_tokens"], f"{event_name}.max_output_tokens"
        ),
        num_preemptions=num_preemptions,
    )
    if request.num_computed_tokens < request.num_prompt_tokens:
        raise EvidenceError(f"{event_name}.num_computed_tokens is below prompt tokens")
    return request, _finite_cost(
        item["observed_disruption_cost"], f"{event_name}.observed_disruption_cost"
    )


def _oracle(
    candidates: list[RequestState], costs: dict[str, float], needed_tokens: int
) -> list[CompressionAction]:
    if len(candidates) > 20:
        raise EvidenceError("oracle replay supports at most 20 candidates per event")
    feasible: list[tuple[float, int, tuple[str, ...], tuple[RequestState, ...]]] = []
    for size in range(1, len(candidates) + 1):
        for subset in itertools.combinations(candidates, size):
            freed = sum(candidate.current_tokens for candidate in subset)
            if freed >= needed_tokens:
                ids = tuple(sorted(candidate.request_id for candidate in subset))
                feasible.append((sum(costs[item] for item in ids), freed, ids, subset))
    if not feasible:
        return []
    _, _, _, selected = min(feasible, key=lambda item: (item[0], item[1], item[2]))
    return [
        CompressionAction(
            request_id=candidate.request_id,
            action_type="evict",
            target_tokens=candidate.current_tokens,
            metadata={"strategy": "counterfactual-oracle"},
        )
        for candidate in selected
    ]


def _arm_receipt(
    name: str,
    actions: list[CompressionAction],
    candidates: list[RequestState],
    costs: dict[str, float],
    needed_tokens: int,
) -> dict[str, Any]:
    by_id = {candidate.request_id: candidate for candidate in candidates}
    action_ids = [action.request_id for action in actions]
    if len(action_ids) != len(set(action_ids)) or any(item not in by_id for item in action_ids):
        raise EvidenceError(f"{name} produced duplicate or unknown victims")
    freed = sum(action.target_tokens for action in actions)
    if freed < needed_tokens:
        raise EvidenceError(f"{name} did not free the required capacity")
    for action in actions:
        if action.target_tokens > by_id[action.request_id].current_tokens:
            raise EvidenceError(f"{name} freed more tokens than the candidate owns")
    disruption = sum(
        costs[action.request_id] * action.target_tokens / by_id[action.request_id].current_tokens
        for action in actions
    )
    return {
        "strategy": name,
        "victims": action_ids,
        "tokens_freed": freed,
        "observed_disruption_cost": disruption,
        "capacity_satisfied": True,
    }


def build_receipt(trace: dict[str, Any], trace_sha256: str) -> dict[str, Any]:
    """Replay all four arms against every exact candidate universe."""
    events: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for index, payload in enumerate(trace["events"]):
        item = _object(payload, f"events[{index}]")
        if set(item) != {"event_id", "needed_tokens", "candidates"}:
            raise EvidenceError(f"events[{index}] has unexpected fields")
        event_id = _text(item["event_id"], f"events[{index}].event_id")
        if event_id in seen_event_ids:
            raise EvidenceError(f"duplicate event_id: {event_id}")
        seen_event_ids.add(event_id)
        needed_tokens = _positive_int(item["needed_tokens"], f"events[{index}].needed_tokens")
        candidates_payload = item["candidates"]
        if not isinstance(candidates_payload, list) or not candidates_payload:
            raise EvidenceError(f"events[{index}].candidates must be non-empty")
        parsed = [_candidate(candidate, event_id) for candidate in candidates_payload]
        candidates = [candidate for candidate, _ in parsed]
        costs = {candidate.request_id: cost for candidate, cost in parsed}
        if len(costs) != len(candidates):
            raise EvidenceError(f"{event_id} has duplicate candidate request IDs")
        if sum(candidate.current_tokens for candidate in candidates) < needed_tokens:
            raise EvidenceError(f"{event_id} candidate pool cannot satisfy required capacity")
        pool_payload = [
            {"request_id": candidate.request_id, "current_tokens": candidate.current_tokens}
            for candidate in candidates
        ]
        pool_hash = hashlib.sha256(
            json.dumps(pool_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        strategies = {
            "stock": PreemptEvictStrategy().select_victims(candidates, needed_tokens),
            "static": StaticRandomStrategy(seed=index).select_victims(candidates, needed_tokens),
            "bidkv": BidKVStrategy().select_victims(candidates, needed_tokens),
            "oracle": _oracle(candidates, costs, needed_tokens),
        }
        arms = [
            _arm_receipt(name, actions, candidates, costs, needed_tokens)
            for name, actions in strategies.items()
        ]
        events.append(
            {
                "event_id": event_id,
                "candidate_pool_sha256": pool_hash,
                "needed_tokens": needed_tokens,
                "arms": arms,
            }
        )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "VERIFIED",
        "evidence_level": "decision-replay-only",
        "trace_id": trace["trace_id"],
        "trace_sha256": trace_sha256,
        "source": trace["source"],
        "event_count": len(events),
        "events": events,
        "claim_boundary": (
            "This receipt compares victim decisions on a frozen real-serving trace; "
            "it is not an end-to-end throughput, latency, or quality result."
        ),
    }
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    return receipt
