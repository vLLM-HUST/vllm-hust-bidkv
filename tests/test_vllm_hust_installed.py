"""Installed-distribution contract with the native vLLM-HUST selector seam."""

from __future__ import annotations

import os
from importlib.metadata import entry_points
from types import SimpleNamespace

import pytest

REQUIRE_INTEGRATION = os.environ.get("BIDKV_REQUIRE_VLLM_HUST_INTEGRATION") == "1"


def _integration_imports():
    try:
        from vllm.v1.core.sched.victim_selector import (
            NoOpVictimSelector,
            get_victim_selector,
        )

        from bidkv.adapters.vllm_hust.selector import BidkvVictimSelector
    except (ImportError, OSError) as exc:
        if REQUIRE_INTEGRATION:
            pytest.fail(f"vLLM-HUST integration environment is incomplete: {exc}")
        pytest.skip(f"vLLM-HUST integration environment only: {exc}")
    return BidkvVictimSelector, NoOpVictimSelector, get_victim_selector


def test_installed_native_entry_point_loads_through_vllm_hust() -> None:
    selector_cls, noop_cls, get_selector = _integration_imports()
    matches = [ep for ep in entry_points(group="vllm.victim_selector") if ep.name == "bidkv"]
    if not matches and not REQUIRE_INTEGRATION:
        pytest.skip("BidKV distribution metadata is not installed")

    assert len(matches) == 1
    assert matches[0].load() is selector_cls

    ambient_config = SimpleNamespace(additional_config={})
    assert isinstance(get_selector(ambient_config), noop_cls)

    selected_config = SimpleNamespace(
        additional_config={
            "victim_selector_plugin": "bidkv",
            "enable_utility_victim_selection": True,
        }
    )
    assert isinstance(get_selector(selected_config), selector_cls)
