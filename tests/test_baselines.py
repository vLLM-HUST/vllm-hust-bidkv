"""bidkv baselines 单元测试。

测试覆盖：
- BaselineStrategy ABC / CompressionAction / RequestState 数据类型
- BaselineRegistry 注册与获取
- 7 个 baseline 各 ≥ 3 个测试
- Candidate-universe consistency（所有 baseline 使用同一候选池）
- Largest-First (was H2O-Style) ≠ PositionalScoring 区分验证
"""

from __future__ import annotations

import pytest

from bidkv.baselines import (
    BaselineRegistry,
    BaselineStrategy,
    BidKVStrategy,
    CompressionAction,
    LargestFirstStrategy,
    PreemptEvictSJFStrategy,
    PreemptEvictStrategy,
    RequestState,
    StaticRandomStrategy,
)
from bidkv.protocol.bid import CompressionBid
from bidkv.scoring import PositionalScoring

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidates(n: int = 5, tokens_per_req: int = 200) -> list[RequestState]:
    """生成 N 个候选请求。"""
    return [
        RequestState(
            request_id=f"req-{i}",
            current_tokens=tokens_per_req,
            priority=float(i),
            arrival_time_ms=float(i * 100),
            deadline_ms=float(10000 + i * 1000),
            token_ids=tuple(range(tokens_per_req)),
        )
        for i in range(n)
    ]


def _make_candidates_varied() -> list[RequestState]:
    """生成具有不同 token 数和优先级的候选请求。"""
    return [
        RequestState(
            "req-a",
            current_tokens=500,
            priority=1.0,
            arrival_time_ms=0.0,
            deadline_ms=20000.0,
            token_ids=tuple(range(500)),
        ),
        RequestState(
            "req-b",
            current_tokens=300,
            priority=3.0,
            arrival_time_ms=100.0,
            deadline_ms=15000.0,
            token_ids=tuple(range(300)),
        ),
        RequestState(
            "req-c",
            current_tokens=100,
            priority=2.0,
            arrival_time_ms=200.0,
            deadline_ms=25000.0,
            token_ids=tuple(range(100)),
        ),
        RequestState(
            "req-d",
            current_tokens=400,
            priority=0.5,
            arrival_time_ms=300.0,
            deadline_ms=12000.0,
            token_ids=tuple(range(400)),
        ),
    ]


def _make_bids_for_request(request_id: str, token_ids: tuple[int, ...]) -> list[CompressionBid]:
    """用 PositionalScoring 为请求生成 bids。"""
    scorer = PositionalScoring()
    return scorer.generate_bids(
        request_id,
        token_ids,
        [0.2, 0.4, 0.6],
    )


# ===========================================================================
# Data type tests
# ===========================================================================


class TestCompressionAction:
    """CompressionAction 数据类型测试。"""

    def test_valid_evict(self) -> None:
        action = CompressionAction(request_id="req-1", action_type="evict", target_tokens=100)
        assert action.request_id == "req-1"
        assert action.action_type == "evict"
        assert action.target_tokens == 100

    def test_valid_compress(self) -> None:
        action = CompressionAction(request_id="req-1", action_type="compress", target_tokens=50)
        assert action.action_type == "compress"

    def test_invalid_action_type(self) -> None:
        with pytest.raises(ValueError, match="action_type"):
            CompressionAction(request_id="req-1", action_type="delete", target_tokens=10)

    def test_invalid_target_tokens(self) -> None:
        with pytest.raises(ValueError, match="target_tokens"):
            CompressionAction(request_id="req-1", action_type="compress", target_tokens=0)

    def test_frozen(self) -> None:
        action = CompressionAction(request_id="req-1", action_type="compress", target_tokens=10)
        with pytest.raises(AttributeError):
            action.target_tokens = 20  # type: ignore[misc]


class TestRequestState:
    """RequestState 数据类型测试。"""

    def test_basic(self) -> None:
        req = RequestState(request_id="req-1", current_tokens=200)
        assert req.request_id == "req-1"
        assert req.current_tokens == 200
        assert req.priority == 0.0
        assert req.deadline_ms is None

    def test_with_deadline(self) -> None:
        req = RequestState(request_id="req-1", current_tokens=100, deadline_ms=5000.0)
        assert req.deadline_ms == 5000.0

    def test_frozen(self) -> None:
        req = RequestState(request_id="req-1", current_tokens=100)
        with pytest.raises(AttributeError):
            req.current_tokens = 300  # type: ignore[misc]


# ===========================================================================
# BaselineRegistry tests
# ===========================================================================


class TestBaselineRegistry:
    """BaselineRegistry 注册表测试。"""

    def test_register_and_get(self) -> None:
        registry = BaselineRegistry()
        strategy = PreemptEvictStrategy()
        registry.register(strategy)
        assert registry.get("preempt-evict") is strategy

    def test_duplicate_register_raises(self) -> None:
        registry = BaselineRegistry()
        registry.register(PreemptEvictStrategy())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(PreemptEvictStrategy())

    def test_get_unknown_raises(self) -> None:
        registry = BaselineRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list_strategies(self) -> None:
        registry = BaselineRegistry()
        registry.register(PreemptEvictStrategy())
        registry.register(StaticRandomStrategy())
        names = registry.list_strategies()
        assert names == ["preempt-evict", "static-random"]

    def test_create_default_registry(self) -> None:
        registry = BaselineRegistry()
        registry.create_default_registry()
        assert registry.count == 5
        # 验证所有 5 个策略都已注册
        for name in [
            "preempt-evict",
            "preempt-evict-sjf",
            "static-random",
            "largest-first",
            "bidkv",
        ]:
            assert isinstance(registry.get(name), BaselineStrategy)


# ===========================================================================
# Preempt-Evict tests
# ===========================================================================


class TestPreemptEvict:
    """Preempt-Evict baseline 测试。"""

    def test_name(self) -> None:
        assert PreemptEvictStrategy().name == "preempt-evict"

    def test_evicts_lowest_priority_first(self) -> None:
        candidates = _make_candidates_varied()
        strategy = PreemptEvictStrategy()
        actions = strategy.select_victims(candidates, needed_tokens=400)
        # req-d (priority=0.5) 应该最先被驱逐
        assert actions[0].request_id == "req-d"
        assert actions[0].action_type == "evict"

    def test_evicts_until_sufficient(self) -> None:
        candidates = _make_candidates_varied()
        strategy = PreemptEvictStrategy()
        actions = strategy.select_victims(candidates, needed_tokens=500)
        freed = sum(a.target_tokens for a in actions)
        assert freed >= 500

    def test_empty_candidates(self) -> None:
        strategy = PreemptEvictStrategy()
        actions = strategy.select_victims([], needed_tokens=100)
        assert actions == []

    def test_zero_needed(self) -> None:
        strategy = PreemptEvictStrategy()
        actions = strategy.select_victims(_make_candidates(), needed_tokens=0)
        assert actions == []

    def test_all_actions_are_evict(self) -> None:
        strategy = PreemptEvictStrategy()
        actions = strategy.select_victims(_make_candidates(3), needed_tokens=1000)
        for action in actions:
            assert action.action_type == "evict"


# ===========================================================================
# Static-Random tests
# ===========================================================================


class TestStaticRandom:
    """Static-Random baseline 测试。"""

    def test_name(self) -> None:
        assert StaticRandomStrategy().name == "static-random"

    def test_deterministic_with_seed(self) -> None:
        candidates = _make_candidates(5)
        s1 = StaticRandomStrategy(seed=42)
        s2 = StaticRandomStrategy(seed=42)
        a1 = s1.select_victims(candidates, needed_tokens=300)
        a2 = s2.select_victims(candidates, needed_tokens=300)
        assert [a.request_id for a in a1] == [a.request_id for a in a2]

    def test_fixed_compression_ratio(self) -> None:
        candidates = [RequestState("req-1", current_tokens=200)]
        strategy = StaticRandomStrategy(compression_ratio=0.5, seed=0)
        actions = strategy.select_victims(candidates, needed_tokens=50)
        assert len(actions) == 1
        assert actions[0].target_tokens == 100  # 200 * 0.5

    def test_all_actions_are_compress(self) -> None:
        strategy = StaticRandomStrategy(seed=0)
        actions = strategy.select_victims(_make_candidates(3), needed_tokens=100)
        for action in actions:
            assert action.action_type == "compress"

    def test_invalid_ratio(self) -> None:
        with pytest.raises(ValueError, match="compression_ratio"):
            StaticRandomStrategy(compression_ratio=0.0)

    def test_empty_candidates(self) -> None:
        strategy = StaticRandomStrategy(seed=0)
        assert strategy.select_victims([], needed_tokens=100) == []


# ===========================================================================
# Largest-First tests (was H2O-Style)
# ===========================================================================


class TestLargestFirst:
    """Largest-First baseline 测试。"""

    def test_name(self) -> None:
        assert LargestFirstStrategy().name == "largest-first"

    def test_largest_first_is_not_positional_scoring(self) -> None:
        """Largest-First ≠ PositionalScoring：确认是策略而非评分器。"""
        scorer = PositionalScoring()
        strategy = LargestFirstStrategy(scoring=scorer)
        # Largest-First 是 BaselineStrategy 实例
        assert isinstance(strategy, BaselineStrategy)
        # Largest-First 持有 PositionalScoring 实例
        assert strategy.scoring is scorer

    def test_compresses_with_scoring(self) -> None:
        candidates = _make_candidates_varied()
        strategy = LargestFirstStrategy()
        actions = strategy.select_victims(candidates, needed_tokens=100)
        assert len(actions) >= 1
        for action in actions:
            assert action.action_type == "compress"

    def test_respects_needed_tokens(self) -> None:
        candidates = _make_candidates_varied()
        strategy = LargestFirstStrategy()
        actions = strategy.select_victims(candidates, needed_tokens=100)
        freed = sum(a.target_tokens for a in actions)
        assert freed >= 100

    def test_empty_candidates(self) -> None:
        strategy = LargestFirstStrategy()
        assert strategy.select_victims([], needed_tokens=100) == []

    def test_with_scoring_states(self) -> None:
        """可以传入每个请求独立的 PositionalScoring 实例。"""
        candidates = [RequestState("req-1", current_tokens=100, token_ids=tuple(range(100)))]
        scorer = PositionalScoring()
        # 用 decode 数据更新 scorer
        scorer.update_from_decode_step([float(i) for i in range(100)])
        strategy = LargestFirstStrategy()
        actions = strategy.select_victims(
            candidates, needed_tokens=30, scoring_states={"req-1": scorer}
        )
        assert len(actions) >= 1


# ===========================================================================
# Preempt-Evict-SJF tests
# ===========================================================================


class TestPreemptEvictSJF:
    """Preempt-Evict-SJF ablation baseline 测试。"""

    def test_name(self) -> None:
        assert PreemptEvictSJFStrategy().name == "preempt-evict-sjf"

    def test_evicts_like_preempt_evict(self) -> None:
        """与 preempt-evict 相同的 eviction 逻辑（按优先级驱逐）。"""
        candidates = _make_candidates_varied()
        sjf = PreemptEvictSJFStrategy()
        base = PreemptEvictStrategy()
        actions_sjf = sjf.select_victims(candidates, needed_tokens=200)
        actions_base = base.select_victims(candidates, needed_tokens=200)
        # 相同的 victim 选择顺序
        assert [a.request_id for a in actions_sjf] == [a.request_id for a in actions_base]
        # 都是 evict 类型
        for a in actions_sjf:
            assert a.action_type == "evict"

    def test_empty_candidates(self) -> None:
        strategy = PreemptEvictSJFStrategy()
        assert strategy.select_victims([], needed_tokens=100) == []

    def test_metadata_contains_strategy_name(self) -> None:
        candidates = _make_candidates(2, tokens_per_req=100)
        strategy = PreemptEvictSJFStrategy()
        actions = strategy.select_victims(candidates, needed_tokens=50)
        for action in actions:
            assert action.metadata["strategy"] == "preempt-evict-sjf"


# ===========================================================================
# BidKV tests
# ===========================================================================


class TestBidKV:
    """BidKV 完整策略测试。"""

    def test_name(self) -> None:
        assert BidKVStrategy().name == "bidkv"

    def test_uses_full_pipeline(self) -> None:
        """BidKV 走完整 bid → pool → solver 流程（Mode A: evict 语义）。"""
        candidates = _make_candidates_varied()
        strategy = BidKVStrategy(delta_budget=0.5)
        actions = strategy.select_victims(candidates, needed_tokens=100)
        assert len(actions) >= 1
        for action in actions:
            assert action.action_type == "evict"
            assert "bid_id" in action.metadata
            assert "utility" in action.metadata

    def test_recompute_cost_aware_ordering(self) -> None:
        """freed（current_tokens）强主导：大 KV 占用且 completion 低的请求优先被驱逐。

        v8-frozen 公式: U = current_tokens / (1 + 0.5*completion + 0.3*P)，δ ∈ [1.0, 1.8]

        req-expensive: current_tokens=500, completion=10/256=0.039
                       quality_delta=1+0.5*0.039=1.019, U=500/1.019=491
        req-cheap:     current_tokens=400, completion=300/512=0.586
                       quality_delta=1+0.5*0.586=1.293, U=400/1.293=309
        req-expensive has higher U (larger KV footprint, low completion) → evicted first
        """
        candidates = [
            RequestState(
                "req-expensive-recompute",  # long prompt, tiny output, big KV
                current_tokens=500,
                token_ids=tuple(range(500)),
                num_prompt_tokens=450,
                num_computed_tokens=460,  # output_tokens=10
                max_output_tokens=256,
            ),
            RequestState(
                "req-cheap-recompute",  # short prompt, lots of output, smaller KV
                current_tokens=400,
                token_ids=tuple(range(400)),
                num_prompt_tokens=100,
                num_computed_tokens=400,  # output_tokens=300
                max_output_tokens=512,  # 300/512=58%, not near done
            ),
        ]
        strategy = BidKVStrategy()
        actions = strategy.select_victims(candidates, needed_tokens=100)
        assert len(actions) >= 1
        # v8 formula: freed (current_tokens) dominates → req-expensive evicted first
        assert actions[0].request_id == "req-expensive-recompute"

    def test_anti_starvation(self) -> None:
        """Previously preempted requests should be protected (higher delta)."""
        candidates = [
            RequestState(
                "req-fresh",
                current_tokens=500,
                token_ids=tuple(range(500)),
                num_prompt_tokens=400,
                num_computed_tokens=450,
                max_output_tokens=256,
                num_preemptions=0,
            ),
            RequestState(
                "req-starved",
                current_tokens=500,
                token_ids=tuple(range(500)),
                num_prompt_tokens=400,
                num_computed_tokens=450,
                max_output_tokens=256,
                num_preemptions=3,
            ),
        ]
        strategy = BidKVStrategy()
        actions = strategy.select_victims(candidates, needed_tokens=100)
        assert len(actions) >= 1
        # req-fresh has same completion but 0 preemptions → lower delta → first victim
        assert actions[0].request_id == "req-fresh"

    def test_output_tokens_dominant_over_small_output(self) -> None:
        """Large output_tokens dominate even with late_penalty from high completion.

        req-large-old: output=500 tokens, short prompt → high U despite late_penalty.
        req-small-young: output=5 tokens, long prompt → U≈0 (tiny freed, high recompute).
        """
        candidates = [
            RequestState(
                "req-small-young",
                current_tokens=200,
                token_ids=tuple(range(200)),
                num_prompt_tokens=190,
                num_computed_tokens=195,  # output_tokens=5
                max_output_tokens=256,
                num_preemptions=0,
            ),
            RequestState(
                "req-large-old",
                current_tokens=800,
                token_ids=tuple(range(800)),
                num_prompt_tokens=200,
                num_computed_tokens=700,  # output_tokens=500
                max_output_tokens=256,
                num_preemptions=0,
            ),
        ]
        strategy = BidKVStrategy()
        actions = strategy.select_victims(candidates, needed_tokens=100)
        assert len(actions) >= 1
        # req-large-old: output=500, recompute_norm=200/256=0.78,
        #   completion=500/256→1.0, late_penalty=1²*2=2.0,
        #   quality_delta=0.78+2.0=2.78, U=500/2.78=180
        # req-small-young: output=5, recompute_norm=190/256=0.742,
        #   quality_delta=0.742, U=5/0.742=6.7
        # Large output wins despite high completion penalty
        assert actions[0].request_id == "req-large-old"

    def test_empty_candidates(self) -> None:
        strategy = BidKVStrategy()
        assert strategy.select_victims([], needed_tokens=100) == []

    def test_zero_needed(self) -> None:
        strategy = BidKVStrategy()
        assert strategy.select_victims(_make_candidates(3), needed_tokens=0) == []


# ===========================================================================
# Candidate-universe consistency tests
# ===========================================================================


class TestCandidateUniverseConsistency:
    """验证所有 baseline 在同一 pressure event 使用同一候选池。"""

    def test_all_baselines_receive_same_candidates(self) -> None:
        """所有 baseline 接收完全相同的 candidates 列表。"""
        candidates = _make_candidates_varied()
        needed = 200

        strategies: list[BaselineStrategy] = [
            PreemptEvictStrategy(),
            PreemptEvictSJFStrategy(),
            StaticRandomStrategy(seed=42),
            LargestFirstStrategy(),
            BidKVStrategy(delta_budget=0.5),
        ]

        # 每个策略都接收同一个 candidates 对象
        results: dict[str, list[CompressionAction]] = {}
        for strategy in strategies:
            actions = strategy.select_victims(candidates, needed)
            results[strategy.name] = actions
            # 验证所有 action 的 request_id 都在候选池中
            candidate_ids = {r.request_id for r in candidates}
            for action in actions:
                assert action.request_id in candidate_ids, (
                    f"{strategy.name} produced action for {action.request_id} "
                    f"not in candidates: {candidate_ids}"
                )

    def test_all_baselines_produce_valid_actions(self) -> None:
        """所有 baseline 返回有效的 CompressionAction。"""
        candidates = _make_candidates(3, tokens_per_req=200)
        needed = 100

        strategies: list[BaselineStrategy] = [
            PreemptEvictStrategy(),
            PreemptEvictSJFStrategy(),
            StaticRandomStrategy(seed=0),
            LargestFirstStrategy(),
            BidKVStrategy(delta_budget=0.5),
        ]

        for strategy in strategies:
            actions = strategy.select_victims(candidates, needed)
            for action in actions:
                assert isinstance(action, CompressionAction)
                assert action.action_type in ("evict", "compress")
                assert action.target_tokens > 0


# ===========================================================================
# Registry integration test
# ===========================================================================


class TestRegistryIntegration:
    """Registry 集成测试 — 所有策略通过 registry 获取并运行。"""

    def test_all_strategies_via_registry(self) -> None:
        registry = BaselineRegistry()
        registry.create_default_registry()

        candidates = _make_candidates(3, tokens_per_req=200)

        for name in registry.list_strategies():
            strategy = registry.get(name)
            actions = strategy.select_victims(candidates, needed_tokens=100)
            assert isinstance(actions, list)
            for action in actions:
                assert isinstance(action, CompressionAction)
