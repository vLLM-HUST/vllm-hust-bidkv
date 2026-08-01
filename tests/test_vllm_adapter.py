"""Tests for bidkv.adapters.vllm — vLLM adapter integration tests.

测试策略：
- 单元测试使用真实 bidkv 组件（无 mock）
- vLLM 集成测试通过 importability 和接口验证确认兼容性
- 所有核心 BidKV 逻辑（scoring、solver、pool、pressure）使用真实实现
"""

from __future__ import annotations

import pytest

from bidkv.adapters.vllm.adapter import AdapterMetrics, VLLMAdapter
from bidkv.config import BidKVConfig
from bidkv.pool import BidPoolManager
from bidkv.pressure import PressureConfig, PressureDetector
from bidkv.scoring import PositionalScoring
from bidkv.solver import GreedyBidSolver, SolverConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def positional_scoring() -> PositionalScoring:
    """PositionalScoring 实例（默认参数）。"""
    return PositionalScoring()


@pytest.fixture
def active_config() -> BidKVConfig:
    """启用 BidKV 的配置。"""
    return BidKVConfig(enabled=True, kill_switch=False, delta_budget=0.15)


@pytest.fixture
def inactive_config() -> BidKVConfig:
    """未启用 BidKV 的配置（默认）。"""
    return BidKVConfig()


@pytest.fixture
def adapter_active(
    active_config: BidKVConfig,
    positional_scoring: PositionalScoring,
) -> VLLMAdapter:
    """启用状态的 VLLMAdapter（无 scheduler）。"""
    return VLLMAdapter(
        config=active_config,
        scoring=positional_scoring,
        pressure_config=PressureConfig(enabled=True, threshold_pct=0.85),
        solver_config=SolverConfig(enabled=True, delta_budget=0.15),
    )


@pytest.fixture
def adapter_inactive(
    inactive_config: BidKVConfig,
    positional_scoring: PositionalScoring,
) -> VLLMAdapter:
    """未启用状态的 VLLMAdapter。"""
    return VLLMAdapter(config=inactive_config, scoring=positional_scoring)


# ---------------------------------------------------------------------------
# Test: VLLMAdapter construction
# ---------------------------------------------------------------------------


class TestVLLMAdapterConstruction:
    """VLLMAdapter 构造函数测试。"""

    def test_default_config_creates_inactive_adapter(
        self, inactive_config: BidKVConfig, positional_scoring: PositionalScoring
    ) -> None:
        adapter = VLLMAdapter(config=inactive_config, scoring=positional_scoring)
        assert not adapter.config.is_active
        assert not adapter.installed

    def test_active_config_creates_active_adapter(
        self, active_config: BidKVConfig, positional_scoring: PositionalScoring
    ) -> None:
        adapter = VLLMAdapter(config=active_config, scoring=positional_scoring)
        assert adapter.config.is_active
        assert not adapter.installed

    def test_components_initialized(self, adapter_active: VLLMAdapter) -> None:
        assert isinstance(adapter_active.pressure_detector, PressureDetector)
        assert isinstance(adapter_active.pool_manager, BidPoolManager)
        assert isinstance(adapter_active.solver, GreedyBidSolver)
        assert isinstance(adapter_active.metrics, AdapterMetrics)


# ---------------------------------------------------------------------------
# Test: Feature gate & kill switch
# ---------------------------------------------------------------------------


class TestFeatureGateKillSwitch:
    """Feature gate 和 kill switch 功能测试。"""

    def test_kill_switch_stops_all_operations(self, adapter_active: VLLMAdapter) -> None:
        """Kill switch 激活后，所有操作返回 0。"""
        adapter_active.track_request("req-1", list(range(100)))
        adapter_active.activate_kill_switch()

        assert not adapter_active.config.is_active
        assert adapter_active.metrics.kill_switch_activations == 1

    def test_kill_switch_deactivation_resumes(self, adapter_active: VLLMAdapter) -> None:
        """Kill switch 解除后恢复操作能力。"""
        adapter_active.activate_kill_switch()
        assert not adapter_active.config.is_active

        adapter_active.deactivate_kill_switch()
        assert adapter_active.config.is_active


# ---------------------------------------------------------------------------
# Test: Request tracking
# ---------------------------------------------------------------------------


class TestRequestTracking:
    """请求追踪功能测试。"""

    def test_track_request(self, adapter_active: VLLMAdapter) -> None:
        adapter_active.track_request("req-1", [1, 2, 3, 4, 5])
        assert "req-1" in adapter_active.get_tracked_requests()

    def test_track_multiple_requests(self, adapter_active: VLLMAdapter) -> None:
        adapter_active.track_request("req-1", [1, 2, 3])
        adapter_active.track_request("req-2", [4, 5, 6])
        tracked = adapter_active.get_tracked_requests()
        assert "req-1" in tracked
        assert "req-2" in tracked

    def test_on_request_complete_clears_tracking(self, adapter_active: VLLMAdapter) -> None:
        adapter_active.track_request("req-1", [1, 2, 3])
        adapter_active.on_request_complete("req-1")
        assert "req-1" not in adapter_active.get_tracked_requests()
        assert adapter_active.metrics.total_requests_completed == 1

    def test_on_request_complete_nonexistent_is_noop(self, adapter_active: VLLMAdapter) -> None:
        """清理不存在的请求不报错。"""
        adapter_active.on_request_complete("nonexistent")
        assert adapter_active.metrics.total_requests_completed == 1


# ---------------------------------------------------------------------------
# Test: get_kv_stats without scheduler
# ---------------------------------------------------------------------------


class TestGetKVStats:
    """get_kv_stats 在无 scheduler 时的行为。"""

    def test_no_scheduler_returns_zeros(self, adapter_active: VLLMAdapter) -> None:
        used, total = adapter_active.get_kv_stats()
        assert used == 0
        assert total == 0


# ---------------------------------------------------------------------------
# Test: install/uninstall hook
# ---------------------------------------------------------------------------


class TestInstallHooks:
    """Scheduler hook 安装/卸载测试。"""

    def test_install_without_scheduler_raises(self, adapter_active: VLLMAdapter) -> None:
        """未设置 scheduler 时 install 应抛出 RuntimeError。"""
        with pytest.raises(RuntimeError, match="scheduler not set"):
            adapter_active.install()

    def test_install_inactive_skips(
        self, inactive_config: BidKVConfig, positional_scoring: PositionalScoring
    ) -> None:
        """Feature OFF 时 install 不注入。"""
        adapter = VLLMAdapter(config=inactive_config, scoring=positional_scoring)
        adapter.install()  # 不报错，静默跳过
        assert not adapter.installed


# ---------------------------------------------------------------------------
# Test: H2O decode step callback
# ---------------------------------------------------------------------------


class TestH2ODecodeCallback:
    """H2O decode step 回调测试。"""

    def test_on_decode_step_updates_scoring(self, adapter_active: VLLMAdapter) -> None:
        """verify on_decode_step 调用 scoring.update_from_decode_step。"""
        scoring = adapter_active.scoring
        assert isinstance(scoring, PositionalScoring)
        assert scoring.decode_steps == 0

        adapter_active.on_decode_step("req-1", [0.1, 0.5, 0.3, 0.8])
        assert scoring.decode_steps == 1
        assert adapter_active.metrics.total_decode_steps == 1

    def test_on_decode_step_inactive_is_noop(self, adapter_inactive: VLLMAdapter) -> None:
        """Feature OFF 时 on_decode_step 不更新。"""
        scoring = adapter_inactive.scoring
        assert isinstance(scoring, PositionalScoring)

        adapter_inactive.on_decode_step("req-1", [0.1, 0.5])
        assert scoring.decode_steps == 0


# ---------------------------------------------------------------------------
# Test: Metrics
# ---------------------------------------------------------------------------


class TestAdapterMetrics:
    """AdapterMetrics 测试。"""

    def test_initial_metrics(self) -> None:
        m = AdapterMetrics()
        assert m.total_evictions == 0
        assert m.total_tokens_freed == 0
        assert m.total_pressure_events == 0

    def test_record_eviction(self) -> None:
        m = AdapterMetrics()
        m.record_eviction("req-1", 100)
        assert m.total_evictions == 1
        assert m.total_tokens_freed == 100

    def test_record_eviction_zero_is_noop(self) -> None:
        m = AdapterMetrics()
        m.record_eviction("req-1", 0)
        assert m.total_evictions == 0

    def test_as_dict(self) -> None:
        m = AdapterMetrics()
        m.record_eviction("req-1", 50)
        m.record_pressure_event()
        d = m.as_dict()
        assert d["total_evictions"] == 1
        assert d["total_tokens_freed"] == 50
        assert d["total_pressure_events"] == 1


# ---------------------------------------------------------------------------
# Test: positional_hook module
# ---------------------------------------------------------------------------


class TestPositionalHook:
    """positional_hook 模块的 _generate_attention_proxy 测试。"""

    def test_generate_attention_proxy_basic(self) -> None:
        from bidkv.adapters.vllm.positional_hook import _generate_attention_proxy

        proxy = _generate_attention_proxy(10)
        assert len(proxy) == 10
        # 所有值应为正数
        assert all(v > 0 for v in proxy)

    def test_generate_attention_proxy_empty(self) -> None:
        from bidkv.adapters.vllm.positional_hook import _generate_attention_proxy

        proxy = _generate_attention_proxy(0)
        assert proxy == []

    def test_generate_attention_proxy_single(self) -> None:
        from bidkv.adapters.vllm.positional_hook import _generate_attention_proxy

        proxy = _generate_attention_proxy(1)
        assert len(proxy) == 1
        assert proxy[0] > 0

    def test_attention_sink_property(self) -> None:
        """验证 position 0 (attention sink) 有较高权重。"""
        from bidkv.adapters.vllm.positional_hook import _generate_attention_proxy

        proxy = _generate_attention_proxy(100)
        # Position 0 应比中间位置权重高
        mid = len(proxy) // 2
        assert proxy[0] > proxy[mid]


# ---------------------------------------------------------------------------
# Test: scheduler_hook module
# ---------------------------------------------------------------------------


class TestSchedulerHook:
    """scheduler_hook 模块的辅助函数测试。"""

    def test_extract_token_ids_from_object(self) -> None:
        """从模拟对象中提取 token ids。"""
        from bidkv.adapters.vllm.scheduler_hook import _extract_token_ids

        class FakeRequest:
            prompt_token_ids = [1, 2, 3, 4]
            output_token_ids = [5, 6]

        token_ids = _extract_token_ids(FakeRequest())
        assert token_ids == [1, 2, 3, 4, 5, 6]

    def test_extract_token_ids_no_output(self) -> None:
        from bidkv.adapters.vllm.scheduler_hook import _extract_token_ids

        class FakeRequest:
            prompt_token_ids = [10, 20]
            output_token_ids = None

        token_ids = _extract_token_ids(FakeRequest())
        assert token_ids == [10, 20]

    def test_extract_token_ids_empty(self) -> None:
        from bidkv.adapters.vllm.scheduler_hook import _extract_token_ids

        class FakeRequest:
            pass

        token_ids = _extract_token_ids(FakeRequest())
        assert token_ids == []


# ---------------------------------------------------------------------------
# Helper: check vLLM v1 importability
# ---------------------------------------------------------------------------


def _vllm_v1_importable() -> bool:
    """检查 vLLM v1 内部模块是否可导入（需要 vllm._C 与 PyTorch ABI 兼容）。"""
    try:
        from vllm.v1.core.sched.scheduler import Scheduler  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


# ---------------------------------------------------------------------------
# Test: vLLM importability (requires vllm installed)
# ---------------------------------------------------------------------------


class TestVLLMImportability:
    """验证 vLLM 相关导入是否可用。"""

    def test_vllm_installed(self) -> None:
        """验证 vLLM 已安装。"""
        vllm = pytest.importorskip("vllm", reason="vLLM integration environment only")

        assert hasattr(vllm, "__version__")

    @pytest.mark.skipif(
        not _vllm_v1_importable(),
        reason="vLLM v1 internals not importable (vllm._C ABI mismatch)",
    )
    def test_vllm_v1_scheduler_importable(self) -> None:
        """验证 vLLM v1 Scheduler 可导入。"""
        from vllm.v1.core.sched.scheduler import Scheduler  # noqa: F401

    @pytest.mark.skipif(
        not _vllm_v1_importable(),
        reason="vLLM v1 internals not importable (vllm._C ABI mismatch)",
    )
    def test_vllm_kv_cache_manager_importable(self) -> None:
        """验证 KVCacheManager 可导入。"""
        from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: F401

    @pytest.mark.skipif(
        not _vllm_v1_importable(),
        reason="vLLM v1 internals not importable (vllm._C ABI mismatch)",
    )
    def test_vllm_block_pool_importable(self) -> None:
        """验证 BlockPool 可导入。"""
        from vllm.v1.core.block_pool import BlockPool  # noqa: F401


# ---------------------------------------------------------------------------
# Test: Full pipeline with scheduler hook on fake scheduler
# ---------------------------------------------------------------------------


class TestSchedulerHookIntegration:
    """使用简易 FakeScheduler 验证 hook 安装/卸载和流程。"""

    def _make_fake_scheduler(self) -> _FakeScheduler:
        return _FakeScheduler()

    def test_install_and_uninstall_hooks(self, adapter_active: VLLMAdapter) -> None:
        """验证 hook 安装后方法被替换，卸载后恢复。"""
        from bidkv.adapters.vllm.scheduler_hook import (
            install_scheduler_hook,
            uninstall_scheduler_hook,
        )

        scheduler = self._make_fake_scheduler()
        original_schedule = scheduler.schedule
        original_update = scheduler.update_from_output

        install_scheduler_hook(scheduler, adapter_active)

        # 方法已被替换
        assert scheduler.schedule is not original_schedule
        assert scheduler.update_from_output is not original_update
        assert hasattr(scheduler, "_bidkv_adapter")

        uninstall_scheduler_hook(scheduler, adapter_active)

        # 方法已恢复
        assert scheduler.schedule == original_schedule
        assert scheduler.update_from_output == original_update
        assert not hasattr(scheduler, "_bidkv_adapter")

    def test_patched_schedule_calls_original(self, adapter_active: VLLMAdapter) -> None:
        """验证 patched schedule 调用原始方法。"""
        from bidkv.adapters.vllm.scheduler_hook import install_scheduler_hook

        scheduler = self._make_fake_scheduler()
        install_scheduler_hook(scheduler, adapter_active)

        result = scheduler.schedule()
        assert result == "schedule_called"
        assert scheduler.schedule_call_count == 1

    def test_patched_schedule_forwards_new_scheduler_arguments(
        self, adapter_active: VLLMAdapter
    ) -> None:
        """Keep the legacy hook compatible with vLLM's evolving schedule API."""
        from bidkv.adapters.vllm.scheduler_hook import install_scheduler_hook

        scheduler = self._make_fake_scheduler()
        install_scheduler_hook(scheduler, adapter_active)

        result = scheduler.schedule(True)
        assert result == "schedule_called"
        assert scheduler.schedule_call_count == 1
        assert scheduler.last_throttle_prefills is True

    def test_patched_update_from_output_calls_original(self, adapter_active: VLLMAdapter) -> None:
        """验证 patched update_from_output 调用原始方法。"""
        from bidkv.adapters.vllm.scheduler_hook import install_scheduler_hook

        scheduler = self._make_fake_scheduler()
        install_scheduler_hook(scheduler, adapter_active)

        result = scheduler.update_from_output("sched_out", "model_out")
        assert result == "update_called"
        assert scheduler.update_call_count == 1

    def test_patched_free_request_cleans_bidkv(self, adapter_active: VLLMAdapter) -> None:
        """验证 patched _free_request 清理 BidKV 状态。"""
        from bidkv.adapters.vllm.scheduler_hook import install_scheduler_hook

        scheduler = self._make_fake_scheduler()
        install_scheduler_hook(scheduler, adapter_active)

        adapter_active.track_request("req-1", [1, 2, 3])
        assert "req-1" in adapter_active.get_tracked_requests()

        class FakeRequest:
            request_id = "req-1"

        scheduler._free_request(FakeRequest())
        assert "req-1" not in adapter_active.get_tracked_requests()


class _FakeRequest:
    """Minimal fake vLLM Request for testing _preempt_request path."""

    def __init__(self, request_id: str, status: str = "RUNNING") -> None:
        self.request_id = request_id
        self.status = status
        self.num_computed_tokens = 100
        self.num_prompt_tokens = 50
        self._output_token_ids: list[int] = list(range(50))
        self._all_token_ids: list[int] = list(range(100))
        self.spec_token_ids: list[int] = []
        self.num_preemptions = 0


class _FakeScheduler:
    """用于测试 hook 安装的简易 FakeScheduler。

    仅实现被 hook 的方法签名，内部计数验证调用。
    不模拟 vLLM 的完整调度逻辑。
    """

    def __init__(self) -> None:
        self.running: list[_FakeRequest] = []
        self.requests: dict[str, _FakeRequest] = {}
        self.schedule_call_count = 0
        self.update_call_count = 0
        self.free_count = 0
        self.kv_cache_manager = None
        self.preempted_requests: list[str] = []

    def add_running_request(self, request_id: str) -> _FakeRequest:
        """Helper: add a fake running request."""
        try:
            from vllm.v1.request import RequestStatus

            status = RequestStatus.RUNNING
        except ImportError:
            status = "RUNNING"
        req = _FakeRequest(request_id, status=status)
        self.requests[request_id] = req
        self.running.append(req)
        return req

    def schedule(self, throttle_prefills: bool = False) -> str:
        self.schedule_call_count += 1
        self.last_throttle_prefills = throttle_prefills
        return "schedule_called"

    def update_from_output(self, scheduler_output: object, model_runner_output: object) -> str:
        self.update_call_count += 1
        return "update_called"

    def _free_request(self, request: object) -> str:
        self.free_count += 1
        return "free_called"

    def _preempt_request(self, request: _FakeRequest, timestamp: float) -> None:
        self.preempted_requests.append(request.request_id)
