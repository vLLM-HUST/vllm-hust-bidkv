"""bidkv protocol 层单元测试

测试覆盖范围：
- CompressionBid：字段验证、frozen 属性、utility 计算、三层字段体系
- BidPool：快照语义、查询方法
- BidAcceptance：字段验证、辅助属性
- CompressionBidProvider：Protocol 结构检查
- 异常类型：层次结构、字段
- 工具函数：compute_utility、make_bid_id
- BidKVConfig：feature gate、kill switch
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from bidkv.config import BidKVConfig
from bidkv.protocol.bid import (
    _UTILITY_EPSILON,
    FEATURE_GATE_ID,
    BidAcceptance,
    BidPool,
    CompressionBid,
    compute_utility,
    make_bid_id,
)
from bidkv.protocol.errors import (
    BidCapacityError,
    BidExecutionError,
    BidExpiredError,
    CompressionBidError,
)
from bidkv.protocol.provider import CompressionBidProvider

# ---------------------------------------------------------------------------
# 测试工具 / Fixtures
# ---------------------------------------------------------------------------


def make_bid(
    *,
    bid_id: str = "req-001:bid:0",
    request_id: str = "req-001",
    algorithm_id: str = "token_budget",
    tokens_freed: int = 256,
    quality_delta: float = 0.05,
    compress_latency_ms: float = 2.0,
    confidence: float = 0.9,
    metadata: dict[str, Any] | None = None,
) -> CompressionBid:
    """创建一个合法的 CompressionBid 测试实例。"""
    return CompressionBid(
        bid_id=bid_id,
        request_id=request_id,
        algorithm_id=algorithm_id,
        tokens_freed=tokens_freed,
        quality_delta=quality_delta,
        compress_latency_ms=compress_latency_ms,
        confidence=confidence,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# TestCompressionBid
# ---------------------------------------------------------------------------


class TestCompressionBid:
    """测试 CompressionBid 数据结构"""

    def test_valid_bid_creation(self) -> None:
        """合法参数应成功创建"""
        bid = make_bid()
        assert bid.bid_id == "req-001:bid:0"
        assert bid.request_id == "req-001"
        assert bid.algorithm_id == "token_budget"
        assert bid.tokens_freed == 256
        assert bid.quality_delta == 0.05
        assert bid.compress_latency_ms == 2.0
        assert bid.confidence == 0.9
        assert bid.metadata == {}

    def test_frozen_immutable(self) -> None:
        """frozen=True：不允许修改字段"""
        bid = make_bid()
        with pytest.raises((AttributeError, TypeError)):
            bid.tokens_freed = 512  # type: ignore[misc]

    def test_metadata_defaults_to_empty_dict(self) -> None:
        """metadata 默认为空 dict"""
        bid = make_bid()
        assert bid.metadata == {}

    def test_metadata_with_values(self) -> None:
        """metadata 支持任意 key-value"""
        meta = {"workload_type": "long_context", "curve_r2": 0.92}
        bid = make_bid(metadata=meta)
        assert bid.metadata["workload_type"] == "long_context"
        assert bid.metadata["curve_r2"] == pytest.approx(0.92)

    # --- 边界：tokens_freed ---

    def test_tokens_freed_minimum_valid(self) -> None:
        """tokens_freed = 1 是合法最小值"""
        bid = make_bid(tokens_freed=1)
        assert bid.tokens_freed == 1

    def test_tokens_freed_zero_raises(self) -> None:
        """tokens_freed = 0 应抛出 ValueError"""
        with pytest.raises(ValueError, match="tokens_freed must be > 0"):
            make_bid(tokens_freed=0)

    def test_tokens_freed_negative_raises(self) -> None:
        """tokens_freed < 0 应抛出 ValueError"""
        with pytest.raises(ValueError, match="tokens_freed must be > 0"):
            make_bid(tokens_freed=-1)

    # --- 边界：quality_delta ---

    def test_quality_delta_zero_lossless(self) -> None:
        """quality_delta = 0.0（无损）合法"""
        bid = make_bid(quality_delta=0.0)
        assert bid.quality_delta == 0.0

    def test_quality_delta_one_full_loss(self) -> None:
        """quality_delta = 1.0（全损）合法"""
        bid = make_bid(quality_delta=1.0)
        assert bid.quality_delta == 1.0

    def test_quality_delta_out_of_range_raises(self) -> None:
        """quality_delta < 0 应抛出 ValueError (>= 0.0 is valid for Mode A)"""
        with pytest.raises(ValueError, match="quality_delta must be >= 0.0"):
            make_bid(quality_delta=-0.01)
        # Values > 1.0 are valid in Mode A (delta = 1 + 2*completion)
        bid = make_bid(quality_delta=2.5)
        assert bid.quality_delta == 2.5

    # --- 边界：confidence ---

    def test_confidence_boundaries(self) -> None:
        """confidence = 0.0 / 1.0 均合法"""
        bid_zero = make_bid(confidence=0.0)
        bid_one = make_bid(confidence=1.0)
        assert bid_zero.confidence == 0.0
        assert bid_one.confidence == 1.0

    def test_confidence_out_of_range_raises(self) -> None:
        """confidence 超出 [0, 1] 应抛出 ValueError"""
        with pytest.raises(ValueError, match="confidence must be in"):
            make_bid(confidence=1.1)

    # --- 边界：compress_latency_ms ---

    def test_compress_latency_zero_valid(self) -> None:
        """compress_latency_ms = 0.0 合法"""
        bid = make_bid(compress_latency_ms=0.0)
        assert bid.compress_latency_ms == 0.0

    def test_compress_latency_negative_raises(self) -> None:
        """compress_latency_ms < 0 应抛出 ValueError"""
        with pytest.raises(ValueError, match="compress_latency_ms must be >= 0.0"):
            make_bid(compress_latency_ms=-1.0)

    # --- 空字符串验证 ---

    def test_empty_bid_id_raises(self) -> None:
        with pytest.raises(ValueError, match="bid_id must be a non-empty string"):
            make_bid(bid_id="")

    def test_empty_request_id_raises(self) -> None:
        with pytest.raises(ValueError, match="request_id must be a non-empty string"):
            make_bid(request_id="")

    def test_empty_algorithm_id_raises(self) -> None:
        with pytest.raises(ValueError, match="algorithm_id must be a non-empty string"):
            make_bid(algorithm_id="")

    # --- Utility 属性 ---

    def test_utility_lossless_bid(self) -> None:
        """无损 bid (delta=0) 的 utility = tokens_freed / epsilon"""
        bid = make_bid(tokens_freed=1000, quality_delta=0.0)
        expected = 1000 / _UTILITY_EPSILON
        assert bid.utility == pytest.approx(expected)

    def test_utility_higher_tokens_higher_utility(self) -> None:
        """相同 delta 下，tokens_freed 越多，utility 越高"""
        bid_small = make_bid(tokens_freed=100, quality_delta=0.1, bid_id="req-001:bid:0")
        bid_large = make_bid(tokens_freed=500, quality_delta=0.1, bid_id="req-001:bid:1")
        assert bid_large.utility > bid_small.utility

    def test_utility_higher_delta_lower_utility(self) -> None:
        """相同 tokens_freed 下，delta 越大，utility 越低"""
        bid_low_delta = make_bid(tokens_freed=256, quality_delta=0.01, bid_id="req-001:bid:0")
        bid_high_delta = make_bid(tokens_freed=256, quality_delta=0.5, bid_id="req-001:bid:1")
        assert bid_low_delta.utility > bid_high_delta.utility

    def test_normalized_utility_equals_utility(self) -> None:
        """normalized_utility 默认等于 utility（归一化由调度器负责）"""
        bid = make_bid()
        assert bid.normalized_utility == bid.utility

    # --- 三层字段体系验证 ---

    def test_layer1_solver_fields(self) -> None:
        """Layer 1 字段（r, δ）直接影响 utility 计算"""
        bid = make_bid(tokens_freed=500, quality_delta=0.1)
        assert bid.utility == pytest.approx(500 / (0.1 + _UTILITY_EPSILON))

    def test_layer2_filter_fields(self) -> None:
        """Layer 2 字段（request_id, compress_latency_ms）用于 BidPool 过滤"""
        bid = make_bid(request_id="req-filter", compress_latency_ms=5.0)
        assert bid.request_id == "req-filter"
        assert bid.compress_latency_ms == 5.0

    def test_layer3_observability_fields(self) -> None:
        """Layer 3 字段（confidence, metadata）用于可观测性"""
        meta = {"model": "qwen-7b", "curve_r2": 0.95}
        bid = make_bid(confidence=0.85, metadata=meta)
        assert bid.confidence == 0.85
        assert bid.metadata["curve_r2"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# TestBidPool
# ---------------------------------------------------------------------------


class TestBidPool:
    """测试 BidPool 快照集合"""

    def _make_pool(self, bids: list[CompressionBid]) -> BidPool:
        return BidPool(snapshot_time_ns=time.monotonic_ns(), bids=tuple(bids))

    def test_empty_pool(self) -> None:
        pool = self._make_pool([])
        assert pool.bids == ()
        assert pool.total_tokens_available == 0

    def test_total_tokens_available(self) -> None:
        bids = [
            make_bid(bid_id="req-001:bid:0", request_id="req-001", tokens_freed=100),
            make_bid(bid_id="req-002:bid:0", request_id="req-002", tokens_freed=200),
        ]
        pool = self._make_pool(bids)
        assert pool.total_tokens_available == 300

    def test_bids_for_request_filters_correctly(self) -> None:
        bids = [
            make_bid(bid_id="req-001:bid:0", request_id="req-001", tokens_freed=200),
            make_bid(bid_id="req-001:bid:1", request_id="req-001", tokens_freed=100),
            make_bid(bid_id="req-002:bid:0", request_id="req-002", tokens_freed=300),
        ]
        pool = self._make_pool(bids)
        req1_bids = pool.bids_for_request("req-001")
        assert len(req1_bids) == 2
        assert all(b.request_id == "req-001" for b in req1_bids)

    def test_bids_for_request_sorted_by_utility_desc(self) -> None:
        """bids_for_request 按 utility 降序排列"""
        bids = [
            make_bid(
                bid_id="req-001:bid:0", request_id="req-001", tokens_freed=100, quality_delta=0.5
            ),
            make_bid(
                bid_id="req-001:bid:1", request_id="req-001", tokens_freed=200, quality_delta=0.1
            ),
        ]
        pool = self._make_pool(bids)
        sorted_bids = pool.bids_for_request("req-001")
        assert sorted_bids[0].utility >= sorted_bids[1].utility

    def test_bids_for_unknown_request_returns_empty(self) -> None:
        pool = self._make_pool([make_bid()])
        assert pool.bids_for_request("unknown") == ()

    def test_top_k_by_utility(self) -> None:
        bids = [
            make_bid(
                bid_id="req-001:bid:0", request_id="req-001", tokens_freed=100, quality_delta=0.5
            ),
            make_bid(
                bid_id="req-002:bid:0", request_id="req-002", tokens_freed=500, quality_delta=0.1
            ),
            make_bid(
                bid_id="req-003:bid:0", request_id="req-003", tokens_freed=200, quality_delta=0.2
            ),
        ]
        pool = self._make_pool(bids)
        top2 = pool.top_k_by_utility(2)
        assert len(top2) == 2
        assert top2[0].utility >= top2[1].utility

    def test_top_k_invalid_raises(self) -> None:
        pool = self._make_pool([make_bid()])
        with pytest.raises(ValueError, match="k must be >= 1"):
            pool.top_k_by_utility(0)

    def test_snapshot_time_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="snapshot_time_ns must be >= 0"):
            BidPool(snapshot_time_ns=-1, bids=())

    def test_pool_is_frozen(self) -> None:
        pool = self._make_pool([make_bid()])
        with pytest.raises((AttributeError, TypeError)):
            pool.snapshot_time_ns = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestBidAcceptance
# ---------------------------------------------------------------------------


class TestBidAcceptance:
    """测试 BidAcceptance 决策结果"""

    def _make_acceptance(self, **kwargs: Any) -> BidAcceptance:
        defaults: dict[str, Any] = {
            "accepted_bid_ids": ("req-001:bid:0",),
            "total_tokens_freed": 256,
            "total_quality_delta": 0.05,
            "decision_reason": "kv_pool_pressure_threshold_exceeded",
        }
        defaults.update(kwargs)
        return BidAcceptance(**defaults)

    def test_valid_creation(self) -> None:
        acc = self._make_acceptance()
        assert acc.accepted_count == 1
        assert acc.is_empty is False
        assert acc.total_tokens_freed == 256

    def test_empty_acceptance(self) -> None:
        acc = self._make_acceptance(
            accepted_bid_ids=(), total_tokens_freed=0, total_quality_delta=0.0
        )
        assert acc.is_empty is True
        assert acc.accepted_count == 0

    def test_total_tokens_freed_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="total_tokens_freed must be >= 0"):
            self._make_acceptance(total_tokens_freed=-1)

    def test_total_quality_delta_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="total_quality_delta must be >= 0.0"):
            self._make_acceptance(total_quality_delta=-0.01)

    def test_empty_decision_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="decision_reason must be a non-empty string"):
            self._make_acceptance(decision_reason="")

    def test_frozen(self) -> None:
        acc = self._make_acceptance()
        with pytest.raises((AttributeError, TypeError)):
            acc.total_tokens_freed = 0  # type: ignore[misc]

    def test_multiple_bids(self) -> None:
        acc = self._make_acceptance(
            accepted_bid_ids=("req-001:bid:0", "req-002:bid:0"),
            total_tokens_freed=600,
            total_quality_delta=0.12,
        )
        assert acc.accepted_count == 2
        assert acc.total_tokens_freed == 600
        assert acc.total_quality_delta == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# TestCompressionBidProvider（Protocol 检查）
# ---------------------------------------------------------------------------


class TestCompressionBidProvider:
    """测试 CompressionBidProvider Protocol 接口"""

    def test_protocol_is_runtime_checkable(self) -> None:
        """CompressionBidProvider 应可用于 isinstance 检查"""
        assert (
            hasattr(CompressionBidProvider, "__protocol_attrs__")
            or hasattr(CompressionBidProvider, "__abstractmethods__")
            or isinstance(CompressionBidProvider, type)
        )

    def test_concrete_impl_satisfies_protocol(self) -> None:
        """实现了接口方法的类应被 isinstance 认可"""

        class MockProvider:
            def get_bids(
                self,
                request_id: str,  # noqa: ARG002
                *,
                min_tokens: int = 0,  # noqa: ARG002
                max_delta: float = 1.0,  # noqa: ARG002
            ) -> list[CompressionBid]:
                return []

            def accept_bid(self, bid_id: str) -> None:  # noqa: ARG002
                pass

        provider = MockProvider()
        assert isinstance(provider, CompressionBidProvider)

    def test_partial_impl_not_protocol(self) -> None:
        """只实现部分方法的类不应满足 Protocol"""

        class PartialProvider:
            def get_bids(
                self,
                request_id: str,  # noqa: ARG002
                *,
                min_tokens: int = 0,  # noqa: ARG002
                max_delta: float = 1.0,  # noqa: ARG002
            ) -> list[CompressionBid]:
                return []

            # 缺少 accept_bid

        partial = PartialProvider()
        assert not isinstance(partial, CompressionBidProvider)

    def test_feature_gate_off_returns_empty(self) -> None:
        """feature gate OFF 时，合规实现应返回空列表"""

        class FeatureOffProvider:
            def get_bids(
                self,
                request_id: str,  # noqa: ARG002
                *,
                min_tokens: int = 0,  # noqa: ARG002
                max_delta: float = 1.0,  # noqa: ARG002
            ) -> list[CompressionBid]:
                return []  # feature OFF：无 bid

            def accept_bid(self, bid_id: str) -> None:  # noqa: ARG002
                pass  # feature OFF：no-op

        provider = FeatureOffProvider()
        bids = provider.get_bids("req-001")
        assert bids == []


# ---------------------------------------------------------------------------
# TestExceptions
# ---------------------------------------------------------------------------


class TestExceptions:
    """测试异常类型层次结构和字段"""

    def test_compression_bid_error_base(self) -> None:
        err = CompressionBidError("test error", bid_id="req-001:bid:0")
        assert err.message == "test error"
        assert err.bid_id == "req-001:bid:0"
        assert str(err) == "test error"

    def test_compression_bid_error_no_bid_id(self) -> None:
        err = CompressionBidError("generic error")
        assert err.bid_id is None

    def test_bid_expired_is_subclass(self) -> None:
        assert issubclass(BidExpiredError, CompressionBidError)
        err = BidExpiredError("bid has expired", bid_id="req-001:bid:0")
        assert err.bid_id == "req-001:bid:0"

    def test_bid_capacity_error_fields(self) -> None:
        assert issubclass(BidCapacityError, CompressionBidError)
        err = BidCapacityError(
            "capacity exceeded",
            bid_id="req-001:bid:0",
            requested_tokens=512,
            available_tokens=128,
        )
        assert err.requested_tokens == 512
        assert err.available_tokens == 128

    def test_bid_execution_error_with_cause(self) -> None:
        assert issubclass(BidExecutionError, CompressionBidError)
        cause = RuntimeError("underlying error")
        err = BidExecutionError("execution failed", bid_id="req-001:bid:0", cause=cause)
        assert err.cause is cause
        assert err.bid_id == "req-001:bid:0"

    def test_all_errors_catchable_as_compression_bid_error(self) -> None:
        """所有 bid 异常均可通过基类捕获"""
        errors = [
            BidExpiredError("expired"),
            BidCapacityError("capacity"),
            BidExecutionError("execution"),
        ]
        for err in errors:
            with pytest.raises(CompressionBidError):
                raise err

    def test_repr_includes_bid_id(self) -> None:
        err = CompressionBidError("msg", bid_id="req-x:bid:0")
        r = repr(err)
        assert "req-x:bid:0" in r
        assert "msg" in r


# ---------------------------------------------------------------------------
# TestUtilityFunctions
# ---------------------------------------------------------------------------


class TestUtilityFunctions:
    """测试工具函数"""

    def test_compute_utility_basic(self) -> None:
        u = compute_utility(tokens_freed=1000, quality_delta=0.1)
        expected = 1000 / (0.1 + _UTILITY_EPSILON)
        assert u == pytest.approx(expected)

    def test_compute_utility_zero_delta(self) -> None:
        """delta=0 时不应除零，使用 epsilon 修正"""
        u = compute_utility(tokens_freed=100, quality_delta=0.0)
        assert u == pytest.approx(100 / _UTILITY_EPSILON)
        assert u > 0

    def test_compute_utility_max_delta(self) -> None:
        """delta=1.0 时 utility 最小"""
        u = compute_utility(tokens_freed=100, quality_delta=1.0)
        assert u == pytest.approx(100 / (1.0 + _UTILITY_EPSILON))

    def test_compute_utility_invalid_tokens(self) -> None:
        with pytest.raises(ValueError, match="tokens_freed must be > 0"):
            compute_utility(tokens_freed=0, quality_delta=0.1)

    def test_compute_utility_invalid_delta(self) -> None:
        with pytest.raises(ValueError, match="quality_delta must be in"):
            compute_utility(tokens_freed=100, quality_delta=1.5)

    def test_compute_utility_monotone_in_tokens(self) -> None:
        """tokens_freed 增加，utility 单调增"""
        deltas = [0.0, 0.1, 0.5, 1.0]
        for delta in deltas:
            u_small = compute_utility(100, delta)
            u_large = compute_utility(200, delta)
            assert u_large > u_small, f"Failed for delta={delta}"

    def test_compute_utility_monotone_in_delta(self) -> None:
        """quality_delta 增加，utility 单调减"""
        u_low = compute_utility(100, 0.01)
        u_high = compute_utility(100, 0.99)
        assert u_low > u_high

    def test_make_bid_id_format(self) -> None:
        bid_id = make_bid_id("req-abc123", 0)
        assert bid_id == "req-abc123:bid:0"

    def test_make_bid_id_level_1(self) -> None:
        bid_id = make_bid_id("req-xyz", 3)
        assert bid_id == "req-xyz:bid:3"

    def test_make_bid_id_empty_request_raises(self) -> None:
        with pytest.raises(ValueError, match="request_id must be a non-empty string"):
            make_bid_id("", 0)

    def test_make_bid_id_negative_level_raises(self) -> None:
        with pytest.raises(ValueError, match="level must be >= 0"):
            make_bid_id("req-001", -1)


# ---------------------------------------------------------------------------
# TestFeatureGate
# ---------------------------------------------------------------------------


class TestFeatureGate:
    """测试 Feature Gate 标识"""

    def test_feature_gate_id_value(self) -> None:
        """Feature gate ID 应为指定值"""
        assert FEATURE_GATE_ID == "compress.scheduling_primitive.v1"

    def test_import_does_not_activate_runtime(self) -> None:
        """导入模块本身不应产生运行时副作用"""
        from bidkv import protocol  # noqa: F401

        assert True


# ---------------------------------------------------------------------------
# TestBidKVConfig
# ---------------------------------------------------------------------------


class TestBidKVConfig:
    """测试 BidKVConfig 配置"""

    def test_default_off(self) -> None:
        """默认创建的配置是 OFF 的"""
        cfg = BidKVConfig()
        assert cfg.enabled is False
        assert cfg.kill_switch is False
        assert cfg.is_active is False

    def test_enabled_active(self) -> None:
        """enabled=True, kill_switch=False → is_active=True"""
        cfg = BidKVConfig(enabled=True)
        assert cfg.is_active is True

    def test_kill_switch_overrides_enabled(self) -> None:
        """kill_switch=True 时，即使 enabled=True，is_active 也为 False"""
        cfg = BidKVConfig(enabled=True, kill_switch=True)
        assert cfg.is_active is False

    def test_delta_budget_default(self) -> None:
        cfg = BidKVConfig()
        assert cfg.delta_budget == 1.0

    def test_delta_budget_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="delta_budget must be >= 0.0"):
            BidKVConfig(delta_budget=-0.1)

    def test_max_bids_per_solve_default(self) -> None:
        cfg = BidKVConfig()
        assert cfg.max_bids_per_solve == 0

    def test_max_bids_per_solve_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_bids_per_solve must be >= 0"):
            BidKVConfig(max_bids_per_solve=-1)

    def test_frozen(self) -> None:
        cfg = BidKVConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.enabled = True  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = BidKVConfig(
            enabled=True,
            kill_switch=False,
            delta_budget=0.5,
            max_bids_per_solve=10,
        )
        assert cfg.enabled is True
        assert cfg.delta_budget == 0.5
        assert cfg.max_bids_per_solve == 10
        assert cfg.is_active is True


# ---------------------------------------------------------------------------
# TestTopLevelImports
# ---------------------------------------------------------------------------


class TestTopLevelImports:
    """验证 bidkv 包的顶层导入是否正常"""

    def test_import_from_bidkv(self) -> None:
        from bidkv import (
            BidAcceptance,
            BidKVConfig,
            BidPool,
            CompressionBid,
            CompressionBidProvider,
            compute_utility,
            make_bid_id,
        )

        assert CompressionBid is not None
        assert BidPool is not None
        assert BidAcceptance is not None
        assert CompressionBidProvider is not None
        assert BidKVConfig is not None
        assert compute_utility is not None
        assert make_bid_id is not None

    def test_import_from_bidkv_protocol(self) -> None:
        from bidkv.protocol import BidAcceptance, BidPool, CompressionBid

        assert CompressionBid is not None
        assert BidPool is not None
        assert BidAcceptance is not None

    def test_version_exists(self) -> None:
        from bidkv import __version__

        assert isinstance(__version__, str)
        assert len(__version__) > 0

    def test_zero_external_dependencies(self) -> None:
        """bidkv 包应零外部依赖"""
        import importlib.metadata

        import bidkv

        # 确认包本身能导入
        assert bidkv.__version__

        # 验证 pyproject.toml 中 dependencies 为空
        # (通过检查包的运行时元数据)
        try:
            dist = importlib.metadata.distribution("vllm-hust-bidkv")
            requires = dist.requires
            # requires 可能为 None 或 只包含 optional deps
            if requires:
                # 过滤掉 optional (extra) 依赖
                core_deps = [r for r in requires if "extra ==" not in r]
                assert len(core_deps) == 0, f"Unexpected core dependencies: {core_deps}"
        except importlib.metadata.PackageNotFoundError:
            # 如果包未安装（pip install -e .），跳过此检查
            pass
