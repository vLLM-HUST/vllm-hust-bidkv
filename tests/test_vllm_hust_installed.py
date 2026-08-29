"""Installed-distribution contract with the native vLLM-HUST selector seam."""

from __future__ import annotations

import json
import os
from importlib.metadata import entry_points
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REQUIRE_INTEGRATION = os.environ.get("BIDKV_REQUIRE_VLLM_HUST_INTEGRATION") == "1"
REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = REPO_ROOT / "tests" / "fixtures" / "vllm_hust_scheduler_trace_v1.json"


def _integration_imports():
    try:
        from bidkv.adapters.vllm_hust.selector import BidkvVictimSelector
        from vllm.v1.core.sched.victim_selector import (
            NoOpVictimSelector,
            VictimSelectorMaterializationError,
            get_victim_selector,
        )
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


def _configure_startup(monkeypatch: pytest.MonkeyPatch, manifests: list[str]) -> None:
    import vllm.envs as envs
    from vllm.plugins.startup import get_configured_extension_startup

    monkeypatch.setattr(envs, "VLLM_EXTENSION_MANIFESTS", manifests)
    monkeypatch.setattr(envs, "VLLM_EXTENSION_BUNDLES", None)
    monkeypatch.setattr(envs, "VLLM_EXTENSION_ALLOWED_PERMISSIONS", [])
    get_configured_extension_startup.cache_clear()


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


def _replay(selector, trace: dict[str, Any]) -> tuple[list[str], dict[str, Any], list]:
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
    return choices, selector.export_metrics(), selector.get_recent_snapshots(limit=20)


def test_installed_native_entry_point_loads_through_vllm_hust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector_cls, _, _, get_selector = _integration_imports()
    matches = [ep for ep in entry_points(group="vllm.victim_selector") if ep.name == "bidkv"]
    if not matches and not REQUIRE_INTEGRATION:
        pytest.skip("BidKV distribution metadata is not installed")

    assert len(matches) == 1
    assert matches[0].load() is selector_cls

    _configure_startup(monkeypatch, [])
    ambient_config = SimpleNamespace(additional_config={})
    assert isinstance(get_selector(ambient_config), selector_cls)

    selected_config = SimpleNamespace(
        additional_config={
            "victim_selector_plugin": "bidkv",
            "enable_utility_victim_selection": True,
        }
    )
    assert isinstance(get_selector(selected_config), selector_cls)


def test_real_manifest_typed_and_legacy_paths_replay_identically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector_cls, _, _, get_selector = _integration_imports()
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    config = dict(trace["additional_config"])
    config["victim_selector_plugin"] = "bidkv"

    _configure_startup(monkeypatch, [])
    legacy = get_selector(SimpleNamespace(additional_config=config))

    manifest = REPO_ROOT / "src" / "bidkv" / "manifests" / "vllm-hust-extension-v1.json"
    _configure_startup(monkeypatch, [str(manifest)])
    typed_config = {
        **config,
        "victim_selector_component": "org.vllm-hust.bidkv/victim-selector",
    }
    typed = get_selector(SimpleNamespace(additional_config=typed_config))

    assert isinstance(legacy, selector_cls)
    assert isinstance(typed, selector_cls)
    assert _replay(typed, trace) == _replay(legacy, trace)


def test_typed_and_explicit_legacy_config_failures_preserve_root_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, materialization_error, get_selector = _integration_imports()
    invalid = {
        "victim_selector_plugin": "bidkv",
        "enable_utility_victim_selection": True,
        "utility_completion_weight": -1,
    }

    _configure_startup(monkeypatch, [])
    with pytest.raises(materialization_error) as legacy_error:
        get_selector(SimpleNamespace(additional_config=invalid))

    manifest = REPO_ROOT / "src" / "bidkv" / "manifests" / "vllm-hust-extension-v1.json"
    _configure_startup(monkeypatch, [str(manifest)])
    with pytest.raises(materialization_error) as typed_error:
        get_selector(
            SimpleNamespace(
                additional_config={
                    **invalid,
                    "victim_selector_component": ("org.vllm-hust.bidkv/victim-selector"),
                }
            )
        )

    assert type(legacy_error.value.__cause__) is type(typed_error.value.__cause__)
    assert str(legacy_error.value.__cause__) == str(typed_error.value.__cause__)


def test_next_start_rollback_and_emergency_disable_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector_cls, noop_cls, _, get_selector = _integration_imports()
    manifest = REPO_ROOT / "src" / "bidkv" / "manifests" / "vllm-hust-extension-v1.json"
    base_config = {
        "victim_selector_plugin": "bidkv",
        "enable_utility_victim_selection": True,
    }

    _configure_startup(monkeypatch, [str(manifest)])
    typed = get_selector(
        SimpleNamespace(
            additional_config={
                **base_config,
                "victim_selector_component": ("org.vllm-hust.bidkv/victim-selector"),
            }
        )
    )

    _configure_startup(monkeypatch, [])
    rolled_back = get_selector(SimpleNamespace(additional_config=base_config))
    disabled = get_selector(
        SimpleNamespace(additional_config={"victim_selector_plugin_disabled": True})
    )

    assert isinstance(typed, selector_cls)
    assert isinstance(rolled_back, selector_cls)
    assert isinstance(disabled, noop_cls)
