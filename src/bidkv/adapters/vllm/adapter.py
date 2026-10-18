"""VLLMAdapter — BidKV 在 vLLM 框架上的适配器。

vLLM v1（0.17+）调度架构：
- ``Scheduler`` 在 ``schedule()`` 中通过 ``kv_cache_manager.allocate_slots()`` 分配 KV
- 分配失败时 preempt 最低优先级 running request
- BidKV 在 preemption 前注入压缩尝试，减少不必要的 preemption

注入方式：Scheduler monkey-patch（``scheduler_hook.py``）。
vLLM v1 移除了 ``BlockSpaceManager`` 抽象和 ``--block-manager-class`` 参数，
因此 issue #044 原始 spec 中的 BlockManager 子类方案不再适用。
详见 ``__init__.py`` 中的 Architecture Decision 说明。

核心职责：
1. KV stats 获取：从 ``KVCacheManager.block_pool`` 读取 usage
2. Pressure interception：在 vLLM preempt 路径前注入请求调度优先级
3. Scoring 回调：decode step 后更新评分策略
4. Lifecycle 管理：请求完成时清理 bid 和内部状态
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from bidkv.adapters.base import BaseAdapterMetrics, FrameworkAdapter
from bidkv.baselines.base import BaselineStrategy
from bidkv.config import BidKVConfig
from bidkv.pool import BidPoolManager
from bidkv.pressure import PressureConfig, PressureDetector
from bidkv.scoring.base import ScoringStrategy
from bidkv.solver import GreedyBidSolver, SolverConfig

logger = logging.getLogger(__name__)

_DIAG_LOG = "/tmp/bidkv_diag.log"


def _diag(msg: str) -> None:
    """Write diagnostic message to a file (works in subprocesses)."""
    import os

    with open(_DIAG_LOG, "a") as f:
        f.write(f"[{os.getpid()}] adapter: {msg}\n")


class VLLMAdapter(FrameworkAdapter):
    """BidKV 在 vLLM 框架上的适配器。

    通过 monkey-patch vLLM 的 Scheduler，在 preemption 路径之前注入
    BidKV 压缩尝试。如果压缩能释放足够空间，则避免 preemption。

    Parameters
    ----------
    config:
        BidKV 全局配置（feature gate + kill switch）。
    scoring:
        评分策略实例。
    scheduler:
        vLLM 的 Scheduler 实例（``vllm.v1.core.sched.scheduler.Scheduler``）。
        若为 None，需在 ``install()`` 前通过 ``set_scheduler()`` 设置。
    pressure_config:
        PressureDetector 配置。若为 None 使用默认值（threshold 与 BidKV 对齐）。
    solver_config:
        GreedyBidSolver 配置。若为 None 使用默认值。

    """

    def __init__(
        self,
        config: BidKVConfig,
        scoring: ScoringStrategy,
        *,
        scheduler: Any = None,
        pressure_config: PressureConfig | None = None,
        solver_config: SolverConfig | None = None,
        experiment_strategy: BaselineStrategy | None = None,
        experiment_strategy_name: str = "bidkv",
    ) -> None:
        super().__init__(config, scoring)
        self._scheduler = scheduler

        # Experiment strategy routing
        self._experiment_strategy: BaselineStrategy | None = experiment_strategy
        self._experiment_strategy_name: str = experiment_strategy_name

        # BidKV 核心组件
        p_cfg = pressure_config or PressureConfig(enabled=config.is_active)
        s_cfg = solver_config or SolverConfig(
            enabled=config.is_active,
            delta_budget=config.delta_budget,
        )
        self._pressure_detector = PressureDetector(p_cfg)
        self._pool_manager = BidPoolManager(
            enabled=config.is_active,
            kill_switch=config.kill_switch,
        )
        self._solver = GreedyBidSolver(s_cfg)

        # 请求追踪
        # {request_id: list[int]} — 每个请求的 token ids
        self._request_tokens: dict[str, list[int]] = {}
        # {request_id: float} — 每个请求的到达时间 (monotonic ms)
        self._request_arrival_ms: dict[str, float] = {}
        # 已安装标记
        self._installed: bool = False
        # 原始方法备份（用于 uninstall）
        self._original_methods: dict[str, Any] = {}

        # Metrics
        self._metrics = AdapterMetrics()

        # Model executor reference (set by plugin.py after EngineCore init).
        self._model_executor: Any = None

        logger.info(
            "VLLMAdapter created: enabled=%s, kill_switch=%s",
            config.is_active,
            config.kill_switch,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pressure_detector(self) -> PressureDetector:
        """内部 PressureDetector 实例。"""
        return self._pressure_detector

    @property
    def pool_manager(self) -> BidPoolManager:
        """内部 BidPoolManager 实例。"""
        return self._pool_manager

    @property
    def solver(self) -> GreedyBidSolver:
        """内部 GreedyBidSolver 实例。"""
        return self._solver

    @property
    def metrics(self) -> AdapterMetrics:
        """适配器运行指标。"""
        return self._metrics

    @property
    def installed(self) -> bool:
        """是否已安装到 vLLM 框架。"""
        return self._installed

    # ------------------------------------------------------------------
    # FrameworkAdapter interface
    # ------------------------------------------------------------------

    def install(self) -> None:
        """将 bidkv 注入到 vLLM 调度路径。

        Monkey-patch vLLM Scheduler 的 schedule()、update_from_output()
        和 _free_request() 方法。

        Raises
        ------
        RuntimeError
            如果 scheduler 未设置。
        """
        if not self._config.is_active:
            logger.info("VLLMAdapter.install: BidKV not active, skipping injection")
            return

        if self._scheduler is None:
            raise RuntimeError(
                "VLLMAdapter.install: scheduler not set. Call set_scheduler() before install()."
            )

        from bidkv.adapters.vllm.scheduler_hook import install_scheduler_hook

        install_scheduler_hook(self._scheduler, self)
        self._installed = True
        logger.info("VLLMAdapter: installed into vLLM scheduler")

    def uninstall(self) -> None:
        """移除 bidkv 注入，恢复 vLLM 原始行为。"""
        if not self._installed:
            return

        from bidkv.adapters.vllm.scheduler_hook import uninstall_scheduler_hook

        uninstall_scheduler_hook(self._scheduler, self)
        self._installed = False
        logger.info("VLLMAdapter: uninstalled from vLLM scheduler")

    def set_scheduler(self, scheduler: Any) -> None:
        """设置 vLLM Scheduler 实例。

        Parameters
        ----------
        scheduler:
            vLLM v1 ``Scheduler`` 实例。
        """
        self._scheduler = scheduler

    def get_kv_stats(self) -> tuple[int, int]:
        """从 vLLM KVCacheManager 获取 KV 使用统计。

        通过 ``block_pool.get_usage()`` 和 ``block_pool.get_num_free_blocks()``
        计算 token 级别的使用量。

        Returns
        -------
        tuple[int, int]
            (used_tokens, max_tokens)。以 block 粒度对齐到 token 数。
        """
        if self._scheduler is None:
            return (0, 0)

        kv_cache_manager = getattr(self._scheduler, "kv_cache_manager", None)
        if kv_cache_manager is None:
            return (0, 0)

        block_pool = getattr(kv_cache_manager, "block_pool", None)
        if block_pool is None:
            return (0, 0)

        # vLLM v1: block_size stored on block_pool as hash_block_size
        block_size = getattr(block_pool, "hash_block_size", None)
        if block_size is None or block_size <= 0:
            return (0, 0)

        # 直接从 block_pool 获取 total blocks
        total_blocks = getattr(block_pool, "num_gpu_blocks", 0)
        if total_blocks <= 0:
            return (0, 0)

        usage = block_pool.get_usage()
        total_tokens = total_blocks * block_size
        used_tokens = int(total_tokens * usage)

        return (used_tokens, total_tokens)

    def on_request_complete(self, request_id: str) -> None:
        """请求完成时清理 bid 和内部状态。"""
        self._pool_manager.remove_by_request(request_id)
        self._request_tokens.pop(request_id, None)
        self._request_arrival_ms.pop(request_id, None)
        self._metrics.record_request_complete(request_id)
        logger.debug("on_request_complete: request=%s", request_id)

    # ------------------------------------------------------------------
    # Request tracking
    # ------------------------------------------------------------------

    def track_request(self, request_id: str, token_ids: list[int]) -> None:
        """开始追踪一个请求的 token。

        Parameters
        ----------
        request_id:
            请求 ID。
        token_ids:
            请求的 token ID 列表。
        """
        self._request_tokens[request_id] = list(token_ids)
        logger.debug(
            "track_request: request=%s, tokens=%d",
            request_id,
            len(token_ids),
        )

    def get_tracked_requests(self) -> list[str]:
        """返回当前追踪的所有请求 ID。"""
        return list(self._request_tokens.keys())

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def activate_kill_switch(self) -> None:
        """激活 kill switch，立即停止所有 BidKV 操作。"""
        self._config = BidKVConfig(
            enabled=self._config.enabled,
            kill_switch=True,
            delta_budget=self._config.delta_budget,
            max_bids_per_solve=self._config.max_bids_per_solve,
            truncation_ratio=self._config.truncation_ratio,
        )
        self._pool_manager.activate_kill_switch()
        self._solver.update_config(
            SolverConfig(
                enabled=self._solver._config.enabled,
                kill_switch=True,
                delta_budget=self._solver._config.delta_budget,
            )
        )
        self._pressure_detector.set_enabled(False)
        self._metrics.record_kill_switch()
        logger.warning("VLLMAdapter: KILL SWITCH activated")

    def deactivate_kill_switch(self) -> None:
        """解除 kill switch，恢复 BidKV 操作。"""
        self._config = BidKVConfig(
            enabled=self._config.enabled,
            kill_switch=False,
            delta_budget=self._config.delta_budget,
            max_bids_per_solve=self._config.max_bids_per_solve,
            truncation_ratio=self._config.truncation_ratio,
        )
        self._pool_manager.enable()
        self._solver.update_config(
            SolverConfig(
                enabled=True,
                kill_switch=False,
                delta_budget=self._config.delta_budget,
            )
        )
        self._pressure_detector.set_enabled(True)
        logger.info("VLLMAdapter: kill switch deactivated, BidKV resumed")

    # ------------------------------------------------------------------
    # Positional scoring decode step callback
    # ------------------------------------------------------------------

    def on_decode_step(self, request_id: str, attention_pattern: Sequence[float]) -> None:
        """decode step 完成后的回调，更新 PositionalScoring。

        由 positional_hook 在每个 decode step 后调用。

        Parameters
        ----------
        request_id:
            请求 ID。
        attention_pattern:
            当前 decode step 中 query token 对所有 KV token 的注意力权重。
        """
        if not self._config.is_active:
            return
        if hasattr(self._scoring, "update_from_decode_step"):
            self._scoring.update_from_decode_step(attention_pattern)
        self._metrics.record_decode_step(request_id)


class AdapterMetrics(BaseAdapterMetrics):
    """vLLM adapter 运行指标。

    继承 ``BaseAdapterMetrics`` 的 6 个跨框架共同字段，
    额外提供 vLLM 特有的 ``preemptions_avoided``，以及
    ``total_all_preemptions`` / ``total_all_tokens_freed``
    （记录所有 preemption，含 vLLM native LIFO，用于 Figure 3）。
    """

    def __init__(self) -> None:
        super().__init__()
        self.preemptions_avoided: int = 0
        self.total_all_preemptions: int = 0
        self.total_all_tokens_freed: int = 0

    def record_preemption_avoided(self) -> None:
        self.preemptions_avoided += 1

    def record_all_preemption(self, tokens_freed: int) -> None:
        """记录所有 preemption（native LIFO + proactive + SRPT），用于 Figure 3。"""
        self.total_all_preemptions += 1
        self.total_all_tokens_freed += max(0, tokens_freed)

    def as_dict(self) -> dict[str, int]:
        """返回所有指标的字典形式（含 vLLM 特有字段）。"""
        d = super().as_dict()
        d["preemptions_avoided"] = self.preemptions_avoided
        d["total_all_preemptions"] = self.total_all_preemptions
        d["total_all_tokens_freed"] = self.total_all_tokens_freed
        return d
