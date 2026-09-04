"""Tests for typed packaging and the separately installed legacy adapter."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bidkv.adapters.vllm import plugin
from bidkv.experiments.vllm import serve

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


def test_main_wheel_registers_only_the_typed_project_manifest() -> None:
    config = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '[project.entry-points."vllm.general_plugins"]' not in config
    assert 'bidkv = "bidkv.adapters.vllm.plugin:register"' not in config
    assert '[project.entry-points."vllm.victim_selector"]' not in config
    assert '[project.entry-points."vllm_hust.extension_bundles"]' in config
    assert '"org.vllm-hust.bidkv" = "bidkv.manifests"' in config

    legacy_config = (REPO_ROOT / "legacy" / "vllm-general-plugin" / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert '[project.entry-points."vllm.general_plugins"]' in legacy_config
    assert 'bidkv = "bidkv.adapters.vllm.plugin:register"' in legacy_config

    from bidkv.adapters.vllm_hust.selector import BidkvPreemptionPolicy

    assert BidkvPreemptionPolicy.vllm_preemption_policy_api_version == 1


def test_legacy_experiment_fails_closed_without_separate_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class EmptyEntryPoints(tuple):
        def select(self, **_kwargs: object) -> tuple[object, ...]:
            return ()

    monkeypatch.setattr(serve, "entry_points", EmptyEntryPoints)

    assert serve._legacy_entry_point_installed() is False
    with pytest.raises(RuntimeError, match="bidkv-vllm-legacy"):
        serve.main()


def test_experimental_extension_manifest_matches_native_policy() -> None:
    manifest_path = REPO_ROOT / "src" / "bidkv" / "manifests" / "vllm-hust-extension-v0.2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "0.2-experimental"
    assert manifest["extension_id"] == "org.vllm-hust.bidkv"
    assert manifest["kind"] == "scheduler_policy"
    assert manifest["host"] == {
        "provider": "vllm",
        "name": "vllm",
        "version_range": ">=0.28.1rc1.dev319,<0.29",
        "api_range": ">=1,<2",
    }
    assert manifest["runtime"]["process_scope"] == "scheduler"
    assert manifest["lifecycle_owner"] == "vllm"
    assert manifest["requires_services"] == []
    assert manifest["implementation"] == [
        {
            "type": "python_module",
            "module": "bidkv.adapters.vllm_hust.selector",
            "object": "BidkvPreemptionPolicy",
            "status": "active",
        }
    ]
    assert manifest["components"] == [
        {
            "component_id": "victim-selector",
            "contracts": ["vllm.preemption-policy.v1"],
            "execution_planes": ["scheduler"],
            "isolation": "trusted_in_process",
            "implementation_ref": ("bidkv.adapters.vllm_hust.selector:BidkvPreemptionPolicy"),
            "permissions": [],
        }
    ]
    assert manifest["activation"] == {
        "entry_points": [],
        "environment": {
            "BIDKV_UTILITY_ENABLE": "1",
            "BIDKV_UTILITY_STRATEGY": "bidkv",
            "BIDKV_UTILITY_LIVENESS_PREEMPTIONS": "2",
            "BIDKV_UTILITY_CASCADE_GAIN_RATIO": "1.25",
        },
        "additional_config": {
            "_manager_runtime_qualification": {
                "accelerator": "ascend",
                    "execution_mode": "graph",
                    "model": "Qwen3.8-27B",
                    "tensor_parallel_size": 4,
                    "status": "incompatible",
                    "evidence": "docs/evidence/sage-mate-20260905-current-main-tp4-graph.md",
                },
        },
    }
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'bidkv = ["manifests/*.json"]' in pyproject


def test_dev_hub_activation_manifest_uses_typed_policy() -> None:
    manifest = json.loads(
        (REPO_ROOT / ".vllm-hust" / "optimization.json").read_text(encoding="utf-8")
    )

    assert manifest["entrypoint"] == {
        "group": "vllm_hust.extension_bundles",
        "name": "org.vllm-hust.bidkv",
    }
    assert manifest["activation"]["vllm_plugins"] == ["ascend"]
    assert manifest["activation"]["extra_args"][:2] == [
        "--preemption-policy",
        "bidkv.adapters.vllm_hust.selector.BidkvPreemptionPolicy",
    ]


def test_upstream_contract_gap_keeps_draft_and_release_boundaries_explicit() -> None:
    gap = (REPO_ROOT / "docs" / "upstream-scheduler-contract-gap.md").read_text(encoding="utf-8")

    assert "f8b7db61e446911e0d62fcb8220f863d6098c471" in gap
    assert "minimum BidKV serving contract" in gap
    assert "does not register the private" in gap
    assert "No compatibility range may include official vLLM" in gap


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
