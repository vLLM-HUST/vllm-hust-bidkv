"""Installed-distribution contract with vLLM's preemption-policy API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REQUIRE_INTEGRATION = os.environ.get("BIDKV_REQUIRE_VLLM_HUST_INTEGRATION") == "1"
REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = REPO_ROOT / "tests" / "fixtures" / "vllm_hust_scheduler_trace_v1.json"


def _integration_imports():
    try:
        from vllm.v1.core.sched.preemption import (
            PreemptionCandidate,
            PreemptionContext,
            PreemptionPolicyController,
        )

        from bidkv.adapters.vllm_hust.selector import BidkvPreemptionPolicy
    except (ImportError, OSError) as exc:
        if REQUIRE_INTEGRATION:
            pytest.fail(f"vLLM-HUST integration environment is incomplete: {exc}")
        pytest.skip(f"vLLM-HUST integration environment only: {exc}")
    return (
        BidkvPreemptionPolicy,
        PreemptionCandidate,
        PreemptionContext,
        PreemptionPolicyController,
    )


def _config(policy, additional_config: dict[str, Any]):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(preemption_policy=policy),
        additional_config=additional_config,
    )


def _context(step: dict[str, Any], candidate_cls, context_cls):
    candidates = tuple(
        candidate_cls(
            request_id=request["request_id"],
            priority=request["priority"],
            arrival_time=request["arrival_time"],
            num_prompt_tokens=0,
            num_output_tokens=request["output_tokens"],
            num_computed_tokens=request["num_computed_tokens"],
            num_preemptions=request["num_preemptions"],
            max_tokens=request["max_tokens"],
        )
        for request in step["running"]
    )
    return context_cls(
        candidates=candidates,
        scheduling_policy=step["policy"].lower(),
        requesting_request_id=candidates[0].request_id,
        kv_cache_usage=step["kv_utilization"],
        now=step["now_s"],
    )


def test_installed_policy_loads_through_vllm_controller() -> None:
    policy_cls, _, _, controller_cls = _integration_imports()
    controller = controller_cls(
        _config(
            policy_cls,
            {
                "enable_utility_victim_selection": True,
                "utility_strategy": "bidkv",
            },
        )
    )

    stats = controller.export_stats()
    assert stats["enabled"] is True
    assert stats["policy_name"].endswith(".BidkvPreemptionPolicy")


def test_native_policy_replays_recorded_trace() -> None:
    policy_cls, candidate_cls, context_cls, controller_cls = _integration_imports()
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    controllers = [
        controller_cls(_config(policy_cls, trace["additional_config"])) for _ in range(2)
    ]

    choices = [
        [
            controller.select_victim(_context(step, candidate_cls, context_cls))
            for step in trace["steps"]
        ]
        for controller in controllers
    ]

    assert choices[0] == choices[1]
    assert controllers[0].export_stats()["calls"] == len(trace["steps"])


def test_invalid_native_config_preserves_root_cause() -> None:
    policy_cls, _, _, controller_cls = _integration_imports()

    with pytest.raises(ValueError, match="utility_completion_weight"):
        controller_cls(
            _config(
                policy_cls,
                {
                    "enable_utility_victim_selection": True,
                    "utility_completion_weight": -1,
                },
            )
        )


def test_next_process_without_policy_restores_builtin() -> None:
    _, candidate_cls, context_cls, controller_cls = _integration_imports()
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    context = _context(trace["steps"][0], candidate_cls, context_cls)

    controller = controller_cls(_config(None, {}))

    assert controller.select_victim(context) == context.candidates[-1].request_id
    assert controller.export_stats()["policy_name"] == "builtin"
    assert controller.export_stats()["enabled"] is False
