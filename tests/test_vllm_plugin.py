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


def test_only_upstream_general_plugin_and_project_manifest_are_registered() -> None:
    config = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '[project.entry-points."vllm.general_plugins"]' in config
    assert 'bidkv = "bidkv.adapters.vllm.plugin:register"' in config
    assert '[project.entry-points."vllm.victim_selector"]' not in config
    assert '[project.entry-points."vllm_hust.extension_bundles"]' in config
    assert '"org.vllm-hust.bidkv" = "bidkv.manifests"' in config

    from bidkv.adapters.vllm_hust.selector import BidkvVictimSelector

    assert BidkvVictimSelector.vllm_victim_selector_api_version == 1


def test_experimental_extension_manifest_matches_native_selector() -> None:
    manifest_path = REPO_ROOT / "src" / "bidkv" / "manifests" / "vllm-hust-extension-v0.2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "0.2-experimental"
    assert manifest["extension_id"] == "org.vllm-hust.bidkv"
    assert manifest["kind"] == "scheduler_policy"
    assert manifest["host"] == {
        "provider": "vllm",
        "name": "vllm",
        "version_range": ">=0.18,<0.20",
        "api_range": ">=1,<2",
    }
    assert manifest["runtime"]["process_scope"] == "scheduler"
    assert manifest["lifecycle_owner"] == "vllm"
    assert manifest["requires_services"] == []
    assert manifest["implementation"] == [
        {
            "type": "python_module",
            "module": "bidkv.adapters.vllm_hust.selector",
            "object": "BidkvVictimSelector",
            "status": "legacy_unregistered",
        }
    ]
    assert manifest["components"] == [
        {
            "component_id": "victim-selector",
            "contracts": ["vllm.scheduler.policy.v1"],
            "execution_planes": ["scheduler"],
            "isolation": "trusted_in_process",
            "implementation_ref": ("bidkv.adapters.vllm_hust.selector:BidkvVictimSelector"),
            "permissions": [],
        }
    ]
    assert manifest["activation"] == {
        "entry_points": [],
        "environment": {
            "BIDKV_UTILITY_ENABLE": "1",
            "BIDKV_UTILITY_STRATEGY": "bidkv",
        },
        "additional_config": {
            "victim_selector_plugin": "bidkv",
            "enable_utility_victim_selection": True,
            "utility_strategy": "bidkv",
        },
    }
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'bidkv = ["manifests/*.json"]' in pyproject


def test_obsolete_dev_hub_activation_manifest_is_removed() -> None:
    assert not (REPO_ROOT / ".vllm-hust" / "optimization.json").exists()


def test_upstream_contract_gap_keeps_draft_and_release_boundaries_explicit() -> None:
    gap = (REPO_ROOT / "docs" / "upstream-scheduler-contract-gap.md").read_text(
        encoding="utf-8"
    )

    assert "f8b7db61e446911e0d62fcb8220f863d6098c471" in gap
    assert "minimum BidKV serving contract" in gap
    assert "does not register the private" in gap
    assert "No compatibility range may include the fresh official fork" in gap


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
    scheduler = SimpleNamespace(vllm_config=SimpleNamespace(additional_config=additional_config))

    with pytest.raises(RuntimeError, match="native BidKV victim selector"):
        plugin._install_bidkv(scheduler, "bidkv")
