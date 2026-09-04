# SPDX-License-Identifier: Apache-2.0
"""Unit tests for bidkv.adapters.vllm_hust BidKV victim selector."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bidkv.adapters.vllm_hust.selector import BidkvSelectorConfig, BidkvVictimSelector


def _vllm_hust_importable() -> bool:
    try:
        from vllm.v1.core.sched.request_queue import SchedulingPolicy  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


def _make_request(
    request_id: str,
    *,
    priority: int = 0,
    arrival_time: float = 0.0,
    num_computed_tokens: int = 0,
    output_tokens: int = 0,
    max_tokens: int | None = 128,
    num_preemptions: int = 0,
):
    output_token_ids = list(range(output_tokens))
    return SimpleNamespace(
        request_id=request_id,
        priority=priority,
        arrival_time=arrival_time,
        num_computed_tokens=num_computed_tokens,
        output_token_ids=output_token_ids,
        num_output_tokens=output_tokens,
        max_tokens=max_tokens,
        num_preemptions=num_preemptions,
    )


# ---------------------------------------------------------------------------
# Config tests (no vllm import needed)
# ---------------------------------------------------------------------------


class TestBidkvSelectorConfig:
    def test_defaults(self):
        config = BidkvSelectorConfig()
        assert config.enable_utility_victim_selection is False
        assert config.utility_strategy == "bidkv"
        assert config.utility_completion_weight == 0.5
        assert config.utility_preempt_weight == 0.3
        assert config.utility_kv_gate == 0.0
        assert config.utility_liveness_preemptions == 2
        assert config.utility_cascade_gain_ratio == 1.25

    def test_validate_raises_on_negative_weights(self):
        with pytest.raises(ValueError):
            BidkvSelectorConfig(utility_completion_weight=-1).validate()
        with pytest.raises(ValueError):
            BidkvSelectorConfig(utility_preempt_weight=-1).validate()
        with pytest.raises(ValueError):
            BidkvSelectorConfig(utility_liveness_preemptions=-1).validate()
        with pytest.raises(ValueError):
            BidkvSelectorConfig(utility_cascade_gain_ratio=0.99).validate()

    def test_validate_raises_on_invalid_kv_gate(self):
        with pytest.raises(ValueError):
            BidkvSelectorConfig(utility_kv_gate=1.5).validate()
        with pytest.raises(ValueError):
            BidkvSelectorConfig(utility_kv_gate=-0.1).validate()

    def test_from_additional_config_preferred(self):
        config = BidkvSelectorConfig.from_additional_config(
            {"enable_utility_victim_selection": True, "utility_strategy": "largest-first"}
        )
        assert config.enable_utility_victim_selection is True
        assert config.utility_strategy == "largest-first"

    def test_from_additional_config_none_falls_back(self):
        config = BidkvSelectorConfig.from_additional_config(None)
        assert isinstance(config, BidkvSelectorConfig)
        assert config.enable_utility_victim_selection is False

    def test_to_additional_config_roundtrip(self):
        config = BidkvSelectorConfig(
            enable_utility_victim_selection=True,
            utility_strategy="bidkv",
            utility_completion_weight=0.6,
        )
        ac = config.to_additional_config()
        assert ac["enable_utility_victim_selection"] is True
        assert ac["utility_strategy"] == "bidkv"
        assert ac["utility_completion_weight"] == 0.6

    def test_from_vllm_config(self):
        vllm_config = SimpleNamespace(additional_config={"enable_utility_victim_selection": True})
        config = BidkvSelectorConfig.from_vllm_config(vllm_config)
        assert config.enable_utility_victim_selection is True


# ---------------------------------------------------------------------------
# Selector construction tests (no vllm runtime needed)
# ---------------------------------------------------------------------------


class TestBidkvVictimSelectorConstruction:
    def test_from_vllm_config_creates_selector(self):
        vllm_config = SimpleNamespace(additional_config={})
        selector = BidkvVictimSelector.from_vllm_config(vllm_config)
        assert isinstance(selector, BidkvVictimSelector)
        assert selector.config.enable_utility_victim_selection is False

    def test_enabled_selector(self):
        vllm_config = SimpleNamespace(additional_config={"enable_utility_victim_selection": True})
        selector = BidkvVictimSelector.from_vllm_config(vllm_config)
        assert selector._utility_enabled is True

    def test_kill_switch_disables(self):
        vllm_config = SimpleNamespace(
            additional_config={
                "enable_utility_victim_selection": True,
                "utility_kill_switch": True,
            }
        )
        selector = BidkvVictimSelector.from_vllm_config(vllm_config)
        assert selector._utility_enabled is False

    def test_export_metrics_initial(self):
        selector = BidkvVictimSelector.from_vllm_config(SimpleNamespace(additional_config={}))
        metrics = selector.export_metrics()
        assert metrics["total_preemptions"] == 0
        assert metrics["total_tokens_freed"] == 0
        assert metrics["kv_pressure_events"] == 0


def test_liveness_fallback_without_vllm_runtime() -> None:
    selector = BidkvVictimSelector(
        BidkvSelectorConfig(
            enable_utility_victim_selection=True,
            utility_liveness_preemptions=2,
        )
    )
    running = [
        _make_request("r1", num_computed_tokens=300, num_preemptions=2),
        _make_request("r2", num_computed_tokens=200, num_preemptions=4),
        _make_request("r3", num_computed_tokens=100, num_preemptions=3),
    ]

    victim = selector.pick_victim(running, "fcfs", kv_utilization=1.0)

    assert victim.request_id == "r3"
    assert selector.export_metrics()["liveness_fallback_hits"] == 1
    assert selector._last_decision["decision_reason"] == "liveness_fallback"


def test_liveness_guard_is_an_epoch_not_a_permanent_fallback() -> None:
    selector = BidkvVictimSelector(
        BidkvSelectorConfig(
            enable_utility_victim_selection=True,
            utility_liveness_preemptions=2,
        )
    )
    first_epoch = [
        _make_request("r1", num_computed_tokens=300, num_preemptions=2),
        _make_request("r2", num_computed_tokens=200, num_preemptions=2),
        _make_request("r3", num_computed_tokens=100, num_preemptions=2),
    ]
    context = SimpleNamespace(
        candidates=tuple(first_epoch),
        scheduling_policy="fcfs",
        requesting_request_id="r2",
        kv_cache_usage=1.0,
        now=1.0,
    )
    assert selector.select_victim(context) == "r2"
    assert selector.export_metrics()["liveness_fallback_hits"] == 1

    # Unchanged cumulative counts are below the new epoch-relative threshold,
    # so utility immediately becomes active again.
    selector.select_victim(SimpleNamespace(**{**vars(context), "now": 2.0}))
    metrics = selector.export_metrics()
    assert metrics["liveness_fallback_hits"] == 1
    assert metrics["utility_strategy_hits"] == 1

    second_epoch = [
        _make_request("r1", num_computed_tokens=300, num_preemptions=4),
        _make_request("r2", num_computed_tokens=200, num_preemptions=4),
        _make_request("r3", num_computed_tokens=100, num_preemptions=4),
    ]
    selector.select_victim(
        SimpleNamespace(**{**vars(context), "candidates": tuple(second_epoch), "now": 3.0})
    )
    metrics = selector.export_metrics()
    assert metrics["liveness_fallback_hits"] == 2
    assert metrics["liveness_epochs"] == 2


def test_requester_guard_avoids_marginal_multi_victim_cascade() -> None:
    selector = BidkvVictimSelector(
        BidkvSelectorConfig(
            enable_utility_victim_selection=True,
            utility_liveness_preemptions=0,
            utility_cascade_gain_ratio=1.25,
        )
    )
    running = (
        _make_request("largest", num_computed_tokens=1000),
        _make_request("requester", num_computed_tokens=900),
        _make_request("small", num_computed_tokens=100),
    )
    context = SimpleNamespace(
        candidates=running,
        scheduling_policy="fcfs",
        requesting_request_id="requester",
        kv_cache_usage=1.0,
        now=1.0,
    )

    assert selector.select_victim(context) == "requester"
    assert selector.export_metrics()["cascade_guard_hits"] == 1


def test_requester_guard_keeps_material_utility_gain() -> None:
    selector = BidkvVictimSelector(
        BidkvSelectorConfig(
            enable_utility_victim_selection=True,
            utility_liveness_preemptions=0,
            utility_cascade_gain_ratio=1.25,
        )
    )
    context = SimpleNamespace(
        candidates=(
            _make_request("largest", num_computed_tokens=2000),
            _make_request("requester", num_computed_tokens=900),
        ),
        scheduling_policy="fcfs",
        requesting_request_id="requester",
        kv_cache_usage=1.0,
        now=1.0,
    )

    assert selector.select_victim(context) == "largest"
    assert selector.export_metrics()["cascade_guard_hits"] == 0


def test_sustained_pressure_trace_stays_bounded_and_utility_active() -> None:
    selector = BidkvVictimSelector(
        BidkvSelectorConfig(
            enable_utility_victim_selection=True,
            utility_liveness_preemptions=2,
            utility_cascade_gain_ratio=1.25,
        )
    )
    counts = {"r0": 0, "r1": 0, "r2": 0}
    selected = {key: 0 for key in counts}
    for step in range(120):
        candidates = tuple(
            _make_request(
                request_id,
                num_computed_tokens=1000 - index * 25,
                num_preemptions=counts[request_id],
            )
            for index, request_id in enumerate(counts)
        )
        requesting = f"r{step % 3}"
        victim = selector.select_victim(
            SimpleNamespace(
                candidates=candidates,
                scheduling_policy="fcfs",
                requesting_request_id=requesting,
                kv_cache_usage=1.0,
                now=float(step),
            )
        )
        counts[victim] += 1
        selected[victim] += 1

    metrics = selector.export_metrics()
    assert metrics["utility_strategy_hits"] / metrics["total_preemptions"] >= 0.75
    assert metrics["liveness_fallback_hits"] > 0
    assert max(selected.values()) - min(selected.values()) <= 2


# ---------------------------------------------------------------------------
# Runtime tests (require vllm import)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _vllm_hust_importable(),
    reason="vLLM HUST internals not importable (transformer version mismatch)",
)
class TestBidkvVictimSelectorRuntime:
    def test_new_preemption_policy_returns_request_id(self):
        selector = BidkvVictimSelector.from_vllm_config(
            SimpleNamespace(additional_config={"enable_utility_victim_selection": True})
        )
        running = (
            _make_request("r1", num_computed_tokens=200),
            _make_request("r2", num_computed_tokens=100),
        )
        context = SimpleNamespace(
            candidates=running,
            scheduling_policy="fcfs",
            requesting_request_id="r1",
            kv_cache_usage=1.0,
            now=10.0,
        )

        assert selector.select_victim(context) == "r1"

    def test_default_non_priority_returns_tail(self):
        from vllm.v1.core.sched.request_queue import SchedulingPolicy

        selector = BidkvVictimSelector.from_vllm_config(SimpleNamespace(additional_config={}))
        running = [_make_request("r1"), _make_request("r2"), _make_request("r3")]
        victim = selector.pick_victim(running, SchedulingPolicy.FCFS)
        assert victim.request_id == "r3"

    def test_default_priority_returns_highest_priority(self):
        from vllm.v1.core.sched.request_queue import SchedulingPolicy

        selector = BidkvVictimSelector.from_vllm_config(SimpleNamespace(additional_config={}))
        running = [
            _make_request("r1", priority=1, arrival_time=1.0),
            _make_request("r2", priority=3, arrival_time=2.0),
            _make_request("r3", priority=2, arrival_time=3.0),
        ]
        victim = selector.pick_victim(running, SchedulingPolicy.PRIORITY)
        assert victim.request_id == "r2"

    def test_utility_mode_prefers_higher_u(self):
        from vllm.v1.core.sched.request_queue import SchedulingPolicy

        selector = BidkvVictimSelector.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    "enable_utility_victim_selection": True,
                    "utility_completion_weight": 0.5,
                    "utility_preempt_weight": 0.3,
                }
            )
        )
        running = [
            _make_request(
                "r1",
                num_computed_tokens=220,
                output_tokens=12,
                max_tokens=128,
                num_preemptions=0,
            ),
            _make_request(
                "r2",
                num_computed_tokens=260,
                output_tokens=120,
                max_tokens=128,
                num_preemptions=3,
            ),
            _make_request(
                "r3",
                num_computed_tokens=180,
                output_tokens=60,
                max_tokens=128,
                num_preemptions=1,
            ),
        ]
        victim = selector.pick_victim(running, SchedulingPolicy.FCFS)
        assert victim.request_id == "r1"

    def test_utility_mode_handles_missing_max_tokens(self):
        from vllm.v1.core.sched.request_queue import SchedulingPolicy

        selector = BidkvVictimSelector.from_vllm_config(
            SimpleNamespace(additional_config={"enable_utility_victim_selection": True})
        )
        running = [
            _make_request(
                "r1",
                num_computed_tokens=50,
                output_tokens=10,
                max_tokens=None,
                num_preemptions=0,
            ),
            _make_request(
                "r2",
                num_computed_tokens=70,
                output_tokens=20,
                max_tokens=0,
                num_preemptions=0,
            ),
        ]
        victim = selector.pick_victim(running, SchedulingPolicy.FCFS)
        assert victim.request_id in {"r1", "r2"}

    def test_liveness_fallback_restores_stable_default_order(self):
        from vllm.v1.core.sched.request_queue import SchedulingPolicy

        selector = BidkvVictimSelector.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    "enable_utility_victim_selection": True,
                    "utility_liveness_preemptions": 2,
                }
            )
        )
        running = [
            _make_request("r1", num_computed_tokens=300, num_preemptions=2),
            _make_request("r2", num_computed_tokens=200, num_preemptions=4),
            _make_request("r3", num_computed_tokens=100, num_preemptions=3),
        ]

        victim = selector.pick_victim(running, SchedulingPolicy.FCFS, kv_utilization=1.0)

        assert victim.request_id == "r3"
        metrics = selector.export_metrics()
        assert metrics["liveness_fallback_hits"] == 1
        assert metrics["utility_strategy_hits"] == 0
        assert metrics["default_strategy_hits"] == 1
        assert selector._last_decision["decision_reason"] == "liveness_fallback"

    def test_liveness_fallback_waits_until_every_candidate_repeated(self):
        from vllm.v1.core.sched.request_queue import SchedulingPolicy

        selector = BidkvVictimSelector.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    "enable_utility_victim_selection": True,
                    "utility_liveness_preemptions": 2,
                }
            )
        )
        running = [
            _make_request("r1", num_computed_tokens=300, num_preemptions=2),
            _make_request("r2", num_computed_tokens=200, num_preemptions=1),
            _make_request("r3", num_computed_tokens=100, num_preemptions=4),
        ]

        victim = selector.pick_victim(running, SchedulingPolicy.FCFS, kv_utilization=1.0)

        assert victim.request_id == "r1"
        assert selector.export_metrics()["liveness_fallback_hits"] == 0

    def test_kill_switch_falls_back_to_default(self):
        from vllm.v1.core.sched.request_queue import SchedulingPolicy

        selector = BidkvVictimSelector.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    "enable_utility_victim_selection": True,
                    "utility_kill_switch": True,
                }
            )
        )
        running = [_make_request("r1"), _make_request("r2")]
        victim = selector.pick_victim(running, SchedulingPolicy.FCFS)
        assert victim.request_id == "r2"

    def test_kv_gate_blocks_utility_when_usage_low(self):
        from vllm.v1.core.sched.request_queue import SchedulingPolicy

        selector = BidkvVictimSelector.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    "enable_utility_victim_selection": True,
                    "utility_kv_gate": 0.95,
                }
            )
        )
        running = [_make_request("r1"), _make_request("r2")]
        victim = selector.pick_victim(running, SchedulingPolicy.FCFS, kv_utilization=0.5)
        assert victim.request_id == "r2"

    def test_kv_gate_allows_utility_when_usage_high(self):
        from vllm.v1.core.sched.request_queue import SchedulingPolicy

        selector = BidkvVictimSelector.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    "enable_utility_victim_selection": True,
                    "utility_kv_gate": 0.8,
                }
            )
        )
        running = [
            _make_request("r1", num_computed_tokens=200, output_tokens=10, num_preemptions=0),
            _make_request("r2", num_computed_tokens=120, output_tokens=100, num_preemptions=2),
        ]
        victim = selector.pick_victim(running, SchedulingPolicy.FCFS, kv_utilization=0.9)
        assert victim.request_id == "r1"

    def test_cooldown_falls_back_to_default_within_window(self):
        from vllm.v1.core.sched.request_queue import SchedulingPolicy

        selector = BidkvVictimSelector.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    "enable_utility_victim_selection": True,
                    "utility_cooldown_s": 10.0,
                }
            )
        )
        running = [
            _make_request("r1", num_computed_tokens=200, output_tokens=10, num_preemptions=0),
            _make_request("r2", num_computed_tokens=120, output_tokens=100, num_preemptions=2),
            _make_request("r3", num_computed_tokens=90, output_tokens=5, num_preemptions=0),
        ]
        first = selector.pick_victim(
            running, SchedulingPolicy.FCFS, kv_utilization=1.0, now_s=100.0
        )
        second = selector.pick_victim(
            running, SchedulingPolicy.FCFS, kv_utilization=1.0, now_s=105.0
        )
        assert first.request_id == "r1"
        assert second.request_id == "r3"

    def test_export_metrics_tracks_hits_and_tokens(self):
        from vllm.v1.core.sched.request_queue import SchedulingPolicy

        selector = BidkvVictimSelector.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    "enable_utility_victim_selection": True,
                    "utility_kv_gate": 0.8,
                }
            )
        )
        running = [
            _make_request("r1", num_computed_tokens=220, output_tokens=5, num_preemptions=0),
            _make_request("r2", num_computed_tokens=100, output_tokens=90, num_preemptions=2),
            _make_request("r3", num_computed_tokens=120, output_tokens=120, num_preemptions=3),
        ]
        selector.pick_victim(running, SchedulingPolicy.FCFS, kv_utilization=0.9, now_s=10.0)
        selector.pick_victim(running, SchedulingPolicy.FCFS, kv_utilization=0.1, now_s=20.0)

        metrics = selector.export_metrics()
        assert metrics["total_preemptions"] == 2
        assert metrics["utility_strategy_hits"] == 1
        assert metrics["default_strategy_hits"] == 1
        assert metrics["strategy_hit_rate"] == 0.5
        assert metrics["total_tokens_freed"] == 340
        assert metrics["kv_pressure_events"] == 1

    def test_snapshot_records_counterfactual_candidates(self):
        from vllm.v1.core.sched.request_queue import SchedulingPolicy

        selector = BidkvVictimSelector.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    "enable_utility_victim_selection": True,
                    "utility_snapshot_enabled": True,
                    "utility_snapshot_top_k": 2,
                    "utility_snapshot_history_size": 4,
                }
            )
        )
        running = [
            _make_request("r1", num_computed_tokens=180, output_tokens=10, num_preemptions=0),
            _make_request("r2", num_computed_tokens=90, output_tokens=80, num_preemptions=2),
            _make_request("r3", num_computed_tokens=60, output_tokens=40, num_preemptions=1),
        ]
        victim = selector.pick_victim(
            running, SchedulingPolicy.FCFS, kv_utilization=1.0, now_s=42.0
        )
        assert victim.request_id == "r1"

        snapshots = selector.get_recent_snapshots(limit=1)
        assert len(snapshots) == 1
        snapshot = snapshots[0]
        assert snapshot["used_utility"] is True
        assert snapshot["selected_victim_id"] == "r1"
        assert len(snapshot["candidates"]) == 2
        assert snapshot["candidates"][0]["request_id"] == "r1"
        assert snapshot["candidates"][0]["selected"] is True

    def test_observability_log_records_selected_victim(self):
        from vllm.v1.core.sched.request_queue import SchedulingPolicy

        selector = BidkvVictimSelector.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    "enable_utility_victim_selection": True,
                    "utility_kv_gate": 0.8,
                }
            )
        )
        running = [
            _make_request("r1", num_computed_tokens=180, output_tokens=10),
            _make_request("r2", num_computed_tokens=90, output_tokens=80),
        ]
        selector.pick_victim(running, SchedulingPolicy.FCFS, kv_utilization=1.0, now_s=42.0)

        logger = Mock()
        selector.emit_observability_log(logger, "AsyncScheduler")

        assert logger.info.call_count == 2
        decision_format, *decision_args = logger.info.call_args_list[1].args
        assert "latest_decision victim=%s" in decision_format
        assert decision_args == [
            "AsyncScheduler",
            "r1",
            "r2",
            True,
            1.0,
            2,
            "FCFS",
        ]

    def test_largest_first_strategy(self):
        from vllm.v1.core.sched.request_queue import SchedulingPolicy

        selector = BidkvVictimSelector.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    "enable_utility_victim_selection": True,
                    "utility_strategy": "largest-first",
                }
            )
        )
        running = [
            _make_request("r1", num_computed_tokens=100),
            _make_request("r2", num_computed_tokens=300),
            _make_request("r3", num_computed_tokens=200),
        ]
        victim = selector.pick_victim(running, SchedulingPolicy.FCFS)
        assert victim.request_id == "r2"

    def test_static_random_strategy(self):
        from vllm.v1.core.sched.request_queue import SchedulingPolicy

        selector = BidkvVictimSelector.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    "enable_utility_victim_selection": True,
                    "utility_strategy": "static-random",
                }
            )
        )
        running = [_make_request("r1"), _make_request("r2"), _make_request("r3")]
        victim = selector.pick_victim(running, SchedulingPolicy.FCFS)
        assert victim.request_id in {"r1", "r2", "r3"}

    def test_pe_strategy_is_default(self):
        from vllm.v1.core.sched.request_queue import SchedulingPolicy

        selector = BidkvVictimSelector.from_vllm_config(
            SimpleNamespace(
                additional_config={
                    "enable_utility_victim_selection": True,
                    "utility_strategy": "pe",
                }
            )
        )
        running = [_make_request("r1"), _make_request("r2")]
        victim = selector.pick_victim(running, SchedulingPolicy.FCFS)
        assert victim.request_id == "r2"
