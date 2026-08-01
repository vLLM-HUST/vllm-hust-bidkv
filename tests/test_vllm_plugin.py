"""Tests for the legacy vLLM general-plugin activation boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bidkv.adapters.vllm import plugin


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def reset_plugin_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(plugin, "_PATCHED", False)
    monkeypatch.delenv("BIDKV_STRATEGY", raising=False)
    monkeypatch.delenv("BIDKV_UTILITY_ENABLE", raising=False)


def test_installing_bidkv_does_not_patch_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(plugin, "_patch_scheduler_init", calls.append)

    plugin.register()

    assert calls == []
    assert plugin._PATCHED is True


def test_explicit_legacy_strategy_enables_scheduler_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setenv("BIDKV_STRATEGY", "preempt-evict")
    monkeypatch.setattr(plugin, "_patch_scheduler_init", calls.append)

    plugin.register()

    assert calls == ["preempt-evict"]
    assert plugin._PATCHED is True


def test_native_and_legacy_entry_points_are_declared_separately() -> None:
    config = (REPO_ROOT / "pyproject.toml").read_text()

    assert '[project.entry-points."vllm.general_plugins"]' in config
    assert 'bidkv = "bidkv.adapters.vllm.plugin:register"' in config
    assert '[project.entry-points."vllm.victim_selector"]' in config
    assert (
        'bidkv = "bidkv.adapters.vllm_hust.selector:BidkvVictimSelector"'
        in config
    )


def test_vllm_hust_optimization_manifest_matches_native_entry_point() -> None:
    manifest = json.loads(
        (REPO_ROOT / ".vllm-hust" / "optimization.json").read_text()
    )

    assert manifest["schema_version"] == 1
    assert manifest["id"] == "bidkv"
    assert manifest["entrypoint"] == {
        "group": "vllm.victim_selector",
        "name": "bidkv",
    }
    config = manifest["activation"]["extra_args"][1]
    assert manifest["activation"]["vllm_plugins"] == ["ascend"]
    assert config["victim_selector_plugin"] == "bidkv"
    assert config["enable_utility_victim_selection"] is True


def test_legacy_and_native_environment_switches_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BIDKV_STRATEGY", "bidkv")
    monkeypatch.setenv("BIDKV_UTILITY_ENABLE", "1")

    with pytest.raises(RuntimeError, match="cannot be combined"):
        plugin.register()


@pytest.mark.parametrize(
    "additional_config",
    [
        {"enable_utility_victim_selection": True},
        {"victim_selector_plugin": "bidkv"},
    ],
)
def test_legacy_hook_rejects_native_selector_configuration(
    additional_config: dict[str, object],
) -> None:
    scheduler = SimpleNamespace(
        vllm_config=SimpleNamespace(additional_config=additional_config)
    )

    with pytest.raises(RuntimeError, match="native BidKV victim selector"):
        plugin._install_bidkv(scheduler, "bidkv")
