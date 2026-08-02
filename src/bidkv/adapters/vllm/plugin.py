"""Legacy vLLM general plugin for experiment-only scheduler injection.

Registered as a ``vllm.general_plugins`` entry point so that it executes
inside **every** vLLM process, including the EngineCore subprocess that is
spawned via ``multiprocessing.spawn``.

Flow
----
1. ``EngineCore.__init__`` calls ``load_general_plugins()`` at startup.
2. This plugin reads ``BIDKV_STRATEGY`` from the environment.
3. Only an explicitly configured legacy experiment monkey-patches
   ``Scheduler.__init__``. Installing BidKV by itself must not change vLLM.

Production serving uses the separate ``vllm.victim_selector`` entry point.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_PATCHED = False  # guard against double-patching


def register() -> None:
    """Entry point called by ``vllm.plugins.load_general_plugins()``."""
    global _PATCHED
    if _PATCHED:
        return

    strategy = os.environ.get("BIDKV_STRATEGY")
    if not strategy:
        logger.info(
            "BidKV legacy scheduler plugin is inactive; "
            "set BIDKV_STRATEGY for legacy experiments"
        )
        _PATCHED = True
        return

    if os.environ.get("BIDKV_UTILITY_ENABLE") == "1":
        raise RuntimeError(
            "BIDKV_STRATEGY enables the legacy scheduler hook and cannot be "
            "combined with BIDKV_UTILITY_ENABLE. Use the vllm.victim_selector "
            "integration for serving, or unset BIDKV_UTILITY_ENABLE for legacy "
            "experiments."
        )

    # ALL strategies install hooks for fair comparison.
    # preempt-evict hooks do FCFS (no reorder) — identical to vanilla vLLM
    # scheduling, but with the same infrastructure overhead for fairness.
    logger.info("BidKV plugin: patching Scheduler for strategy=%s", strategy)
    _patch_scheduler_init(strategy)

    # When num_gpu_blocks_override is used, relax the memory check that
    # prevents startup when other processes consume GPU memory.  The override
    # applies the exact block count regardless of detected available memory.
    if os.environ.get("BIDKV_SKIP_MEM_CHECK") == "1":
        try:
            import vllm.v1.core.kv_cache_utils as _kv_utils
            _kv_utils.check_enough_kv_cache_memory = lambda *_a, **_kw: None
            logger.info("BidKV plugin: patched check_enough_kv_cache_memory (BIDKV_SKIP_MEM_CHECK=1)")
        except ImportError:
            pass

    _PATCHED = True


def _patch_scheduler_init(strategy_name: str) -> None:
    """Monkey-patch Scheduler.__init__ to install BidKV hooks post-init."""
    from vllm.v1.core.sched.scheduler import Scheduler

    _original_init = Scheduler.__init__

    def _bidkv_patched_init(self: Scheduler, *args: object, **kwargs: object) -> None:  # type: ignore[no-untyped-def]
        _original_init(self, *args, **kwargs)  # type: ignore[misc]
        _install_bidkv(self, strategy_name)

    Scheduler.__init__ = _bidkv_patched_init  # type: ignore[method-assign]
    logger.info("BidKV plugin: Scheduler.__init__ patched")


def _install_bidkv(scheduler: object, strategy_name: str) -> None:
    """Install BidKV adapter + hooks on a live Scheduler instance."""
    vllm_config = getattr(scheduler, "vllm_config", None)
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    if (
        additional_config.get("enable_utility_victim_selection")
        or additional_config.get("victim_selector_plugin") == "bidkv"
    ):
        raise RuntimeError(
            "BIDKV_STRATEGY cannot be combined with the native BidKV victim "
            "selector. Remove BIDKV_STRATEGY for serving, or remove the victim "
            "selector configuration for legacy experiments."
        )

    from bidkv.adapters.vllm.adapter import VLLMAdapter
    from bidkv.baselines import BaselineRegistry
    from bidkv.config import BidKVConfig
    from bidkv.pressure import PressureConfig
    from bidkv.scoring.positional import PositionalScoring
    from bidkv.solver import SolverConfig

    config = BidKVConfig(
        enabled=True,
        kill_switch=False,
        delta_budget=0.30,
        truncation_ratio=float(os.environ.get("BIDKV_TRUNCATION_RATIO", "0.5")),
    )
    scoring = PositionalScoring()

    pressure_config = PressureConfig(
        enabled=True,
        threshold_pct=float(os.environ.get("BIDKV_PRESSURE_THRESHOLD", "0.98")),
        min_free_tokens=int(os.environ.get("BIDKV_MIN_FREE_TOKENS", "1024")),
    )
    solver_config = SolverConfig(
        enabled=True,
        delta_budget=config.delta_budget,
        max_bids_per_solve=200,
    )

    # Resolve baseline strategy from registry
    registry = BaselineRegistry()
    registry.create_default_registry()
    strategy = registry.get(strategy_name)

    adapter = VLLMAdapter(
        config=config,
        scoring=scoring,
        scheduler=scheduler,
        pressure_config=pressure_config,
        solver_config=solver_config,
        experiment_strategy=strategy,
        experiment_strategy_name=strategy_name,
    )

    # Install scheduler hooks
    adapter.install()

    # Store reference on scheduler for external access
    scheduler._bidkv_experiment_adapter = adapter  # type: ignore[attr-defined]

    # Register atexit + SIGTERM handler for final metrics dump
    import atexit
    import signal as _signal

    def _dump_final_metrics() -> None:
        from bidkv.adapters.vllm.scheduler_hook import _dump_metrics

        _dump_metrics(adapter)

    atexit.register(_dump_final_metrics)

    _original_sigterm = _signal.getsignal(_signal.SIGTERM)

    def _sigterm_handler(signum: int, frame: object) -> None:  # noqa: ARG001
        _dump_final_metrics()
        # Re-raise to allow normal shutdown
        import sys

        sys.exit(0)

    _signal.signal(_signal.SIGTERM, _sigterm_handler)

    logger.info(
        "BidKV installed on Scheduler in EngineCore subprocess: "
        "strategy=%s, pressure_threshold=%.0f%%",
        strategy_name,
        pressure_config.threshold_pct * 100,
    )
