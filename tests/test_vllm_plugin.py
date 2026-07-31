"""Tests for the legacy vLLM general-plugin activation boundary."""

from __future__ import annotations

import pytest

from bidkv.adapters.vllm import plugin


@pytest.fixture(autouse=True)
def reset_plugin_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(plugin, "_PATCHED", False)
    monkeypatch.delenv("BIDKV_STRATEGY", raising=False)


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
