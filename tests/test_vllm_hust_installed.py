"""Installed-distribution contract with the narrow vLLM selector seam."""

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
        from vllm.v1.core.sched.victim_selector import (
            NoOpVictimSelector,
            VictimSelectorMaterializationError,
            get_victim_selector,
        )

        from bidkv.adapters.vllm_hust.selector import BidkvVictimSelector
    except (ImportError, OSError) as exc:
        if REQUIRE_INTEGRATION:
            pytest.fail(f"vLLM-HUST integration environment is incomplete: {exc}")
        pytest.skip(f"vLLM-HUST integration environment only: {exc}")
    return (
        BidkvVictimSelector,
        NoOpVictimSelector,
        VictimSelectorMaterializationError,
        get_victim_selector,
    )


def _install_typed_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        from vllm.plugins.contracts import (
            ComponentIsolation,
            DomainContract,
            ExecutionPlane,
            ExtensionBundleDescriptor,
            ExtensionComponentDescriptor,
        )
        from vllm.plugins.snapshot import ExtensionStartupSnapshot
        from vllm.plugins.startup import ExtensionStartupResolution
    except (ImportError, OSError) as exc:
        if REQUIRE_INTEGRATION:
            pytest.fail(f"typed vLLM-HUST host contract is unavailable: {exc}")
        pytest.skip(f"typed vLLM-HUST integration environment only: {exc}")
    bundle = ExtensionBundleDescriptor(
        bundle_id="org.vllm-hust.bidkv",
        bundle_version="0.1.1",
        host_api_range=">=1,<2",
        components=(
            ExtensionComponentDescriptor(
                component_id="victim-selector",
                contracts=(DomainContract.SCHEDULER_POLICY_V1,),
                execution_planes=(ExecutionPlane.SCHEDULER,),
                isolation=ComponentIsolation.TRUSTED_IN_PROCESS,
                implementation_ref=(
                    "bidkv.adapters.vllm_hust.selector:BidkvVictimSelector"
                ),
            ),
        ),
    )
    resolution = ExtensionStartupResolution(
        snapshot=ExtensionStartupSnapshot.build((bundle,)),
        disabled_bundle_ids=(),
    )
    monkeypatch.setattr(
        "vllm.plugins.startup.get_configured_extension_startup",
        lambda: resolution,
    )


def _request(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=payload["request_id"],
        priority=payload["priority"],
        arrival_time=payload["arrival_time"],
        num_computed_tokens=payload["num_computed_tokens"],
        output_token_ids=list(range(payload["output_tokens"])),
        max_tokens=payload["max_tokens"],
        num_preemptions=payload["num_preemptions"],
    )


def _replay(selector, trace: dict[str, Any]) -> list[str]:
    from vllm.v1.core.sched.request_queue import SchedulingPolicy

    choices = []
    for step in trace["steps"]:
        victim = selector.pick_victim(
            [_request(request) for request in step["running"]],
            SchedulingPolicy[step["policy"]],
            kv_utilization=step["kv_utilization"],
            now_s=step["now_s"],
        )
        choices.append(victim.request_id)
    return choices


def test_installed_typed_component_loads_through_vllm_hust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector_cls, _, _, get_selector = _integration_imports()
    _install_typed_resolution(monkeypatch)
    selected = get_selector(
        SimpleNamespace(
            additional_config={
                "victim_selector_component": (
                    "org.vllm-hust.bidkv/victim-selector"
                ),
                "enable_utility_victim_selection": True,
            }
        )
    )
    assert isinstance(selected, selector_cls)


def test_native_selector_replays_recorded_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector_cls, _, _, get_selector = _integration_imports()
    _install_typed_resolution(monkeypatch)
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    config = {
        **trace["additional_config"],
        "victim_selector_component": "org.vllm-hust.bidkv/victim-selector",
    }

    selector = get_selector(SimpleNamespace(additional_config=config))
    replay_selector = get_selector(SimpleNamespace(additional_config=config))

    assert isinstance(selector, selector_cls)
    assert _replay(selector, trace) == _replay(replay_selector, trace)


def test_invalid_native_config_preserves_root_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, materialization_error, get_selector = _integration_imports()
    _install_typed_resolution(monkeypatch)
    invalid = {
        "victim_selector_component": "org.vllm-hust.bidkv/victim-selector",
        "enable_utility_victim_selection": True,
        "utility_completion_weight": -1,
    }

    with pytest.raises(materialization_error) as error:
        get_selector(SimpleNamespace(additional_config=invalid))

    assert error.value.__cause__ is not None


def test_emergency_disable_restores_noop_policy() -> None:
    _, noop_cls, _, get_selector = _integration_imports()

    disabled = get_selector(
        SimpleNamespace(additional_config={"victim_selector_plugin_disabled": True})
    )

    assert isinstance(disabled, noop_cls)
