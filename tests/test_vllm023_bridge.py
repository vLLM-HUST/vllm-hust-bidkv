"""CPU contract and native-ledger regressions for the pinned measurement carrier."""

import ast
import gzip
import importlib.util
import json
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from bidkv.adapters.vllm023.bridge import (
    Official023PolicyBridge,
    PreemptionCandidate,
    PreemptionContext,
)


class PickFirst:
    vllm_preemption_policy_api_version = 1

    def select_victim(self, context):
        return context.candidates[0].request_id


class Abstain(PickFirst):
    def select_victim(self, _context):
        return None


class Invalid(PickFirst):
    def select_victim(self, _context):
        return []  # An unhashable contract violation must also fail over.


class Failing(PickFirst):
    def select_victim(self, _context):
        raise RuntimeError("test policy failure")


class Mutating(PickFirst):
    def select_victim(self, context):
        context.candidates[0].num_computed_tokens = 0


def context():
    return PreemptionContext(
        candidates=(
            PreemptionCandidate("first", 0, 1.0, 16, 2, 18, 0, 64),
            PreemptionCandidate("last", 0, 2.0, 32, 3, 35, 0, 64),
        ),
        scheduling_policy="fcfs",
        requesting_request_id="last",
        kv_cache_usage=1.0,
        now=10.0,
        builtin_victim_id="last",
    )


def config(policy=None, evidence_dir=None):
    values = {"policy": f"{__name__}.{policy.__name__}" if policy else None}
    if evidence_dir is not None:
        values["evidence_dir"] = str(evidence_dir)
    return SimpleNamespace(additional_config={"bidkv_official023": values})


@pytest.mark.parametrize(
    "policy,expected", [(None, "last"), (Abstain, "last"), (PickFirst, "first")]
)
def test_builtin_abstention_and_selection(policy, expected):
    bridge = Official023PolicyBridge(config(policy))
    assert bridge.select_victim(context()) == expected
    assert bridge.stats["decisions"] == 1
    assert bridge.stats["calls"] == int(policy is not None)


@pytest.mark.parametrize("policy", [Invalid, Failing, Mutating])
def test_fault_permanently_restores_builtin(policy):
    snapshot = context()
    bridge = Official023PolicyBridge(config(policy))
    assert bridge.select_victim(snapshot) == "last"
    assert bridge.select_victim(snapshot) == "last"
    assert bridge.stats["calls"] == bridge.stats["failures"] == 1
    assert bridge.stats["enabled"] is False
    assert snapshot.candidates[0].num_computed_tokens == 18
    assert bridge.stats["invalid_selections"] == int(policy is Invalid)


def test_snapshot_is_immutable():
    snapshot = context()
    with pytest.raises(FrozenInstanceError):
        snapshot.requesting_request_id = "first"
    with pytest.raises(FrozenInstanceError):
        snapshot.candidates[0].num_computed_tokens = 0


def test_wrong_scheduler_thread_is_rejected():
    bridge = Official023PolicyBridge(config(PickFirst))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(bridge.select_victim, context())
        with pytest.raises(RuntimeError, match="owning scheduler thread"):
            future.result()
    assert bridge.stats["decisions"] == 0


def test_journal_counts_every_decision_and_preserves_close(tmp_path):
    bridge = Official023PolicyBridge(config(PickFirst, tmp_path))
    bridge.select_victim(context())
    bridge.close()
    records = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert [row["event"] for row in records] == ["initialized", "preemption-decision", "closed"]
    assert records[-1]["stats"]["calls"] == records[-1]["stats"]["selections"] == 1
    assert records[1]["context"]["builtin_victim_id"] == "last"
    assert records[1]["selected_id"] == "first"
    assert all(row["wall_time_ns"] and row["monotonic_ns"] for row in records)


def preparer():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_official023_host.py"
    spec = importlib.util.spec_from_file_location("prepare_official023_host", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_patch_is_pinned_and_native_cleanup_is_preserved():
    module = preparer()
    original = gzip.decompress(
        (Path(__file__).parent / "fixtures/official023-scheduler.py.gz").read_bytes()
    )
    patched = module.patched_scheduler(original)

    def cleanup(raw):
        tree = ast.parse(raw)
        return next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_preempt_request"
        )

    assert ast.dump(cleanup(original)) == ast.dump(cleanup(patched))
    with pytest.raises(ValueError, match="exact official"):
        module.patched_scheduler(original + b"\n")


@pytest.mark.parametrize(
    "selected,expected_index,expected_budget,expected_encoder,remaining_scheduled",
    [
        ("skipped", 1, 10, 5, ["scheduled"]),
        ("scheduled", 1, 17, 8, []),
        ("current", 2, 10, 5, ["scheduled"]),
        ("future", 2, 10, 5, ["scheduled"]),
    ],
)
def test_native_rollback_for_arbitrary_victim(
    selected, expected_index, expected_budget, expected_encoder, remaining_scheduled
):
    # Execute the actual block inserted into the pinned scheduler. In particular,
    # removing an earlier skipped request must not skip the current request;
    # preempting scheduled work must restore both token and encoder budgets.
    requests = {
        name: SimpleNamespace(request_id=name, get_num_encoder_embeds=lambda _: 3)
        for name in ["skipped", "scheduled", "current", "future"]
    }
    host = SimpleNamespace(
        running=list(requests.values()),
        _select_bidkv023_victim=lambda *_: requests[selected],
    )
    state = {
        "self": host,
        "request": requests["current"],
        "scheduled_timestamp": 10.0,
        "req_index": 2,
        "scheduled_running_reqs": [requests["scheduled"]],
        "token_budget": 10,
        "encoder_compute_budget": 5,
        "num_scheduled_tokens": {"scheduled": 7},
        "req_to_new_blocks": {"scheduled": object()},
        "scheduled_spec_decode_tokens": {"scheduled": [1, 2]},
        "scheduled_encoder_inputs": {"scheduled": [0]},
    }
    exec(
        compile(textwrap.dedent(preparer().NEW_PREEMPTION), "native-preemption-block", "exec"),
        state,
    )
    assert state["req_index"] == expected_index
    assert state["token_budget"] == expected_budget
    assert state["encoder_compute_budget"] == expected_encoder
    assert [row.request_id for row in state["scheduled_running_reqs"]] == remaining_scheduled
    assert selected not in [row.request_id for row in host.running]
    for name in [
        "num_scheduled_tokens",
        "req_to_new_blocks",
        "scheduled_spec_decode_tokens",
        "scheduled_encoder_inputs",
    ]:
        assert ("scheduled" in state[name]) == (selected != "scheduled")
