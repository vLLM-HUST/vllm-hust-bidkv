"""Scheduler Hook — vLLM Scheduler monkey-patch 注入。

将 BidKV 智能 victim 选择注入到 vLLM v1 Scheduler 的调度路径中。

**Mode A 架构（v3 — pure native preemption）**：
策略控制 *WHO* gets preempted，执行机制是 vLLM native preempt+recompute。
不做 tail truncation / block-level 操作。

注入点：
- ``schedule()``：
  1. 定期刷新策略优先级缓存（每 3s，调用 strategy.select_victims()）
  2. 主动 preempt：KV > 阈值时，用 scheduler._preempt_request() 驱逐
     策略选中的 victim（单 victim + 5s cooldown 防 storm）
  3. 重排 running 列表：用缓存优先级影响 vLLM native preemption 的
     victim 选择（running.pop() FCFS）
- ``update_from_output()``：decode step 后更新 token tracking + positional scoring。
- ``_free_request()``：请求完成时清理 BidKV 状态。

设计原则：
- **纯 Mode A**：策略只做 `decision`（谁被 preempt），不做 `execution`
- **生效双路径**：proactive preempt（主动）+ reorder（影响 native preemption）
- **Feature OFF 零开销**：BidKV 未激活时直接调用原始方法
- **可逆**：``uninstall_scheduler_hook()`` 可恢复原始方法
"""

from __future__ import annotations

import functools
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bidkv.adapters.vllm.adapter import VLLMAdapter

logger = logging.getLogger(__name__)

# KV gate threshold for BidKV running-queue reorder — overridable via env var.
# BIDKV_KV_GATE: below this KV usage ratio, LIFO (vLLM default) is used.
_KV_GATE: float = float(os.environ.get("BIDKV_KV_GATE", "0.95"))

# 用于存储原始方法的属性名前缀
_ORIG_PREFIX = "_bidkv_orig_"


def install_scheduler_hook(scheduler: Any, adapter: VLLMAdapter) -> None:
    """将 BidKV 压缩逻辑注入到 vLLM Scheduler。

    Monkey-patch 三个方法：
    1. ``schedule()`` — preemption 前压缩
    2. ``update_from_output()`` — decode step 后 positional scoring 更新
    3. ``_free_request()`` — 请求完成时 cleanup

    Parameters
    ----------
    scheduler:
        vLLM v1 Scheduler 实例。
    adapter:
        VLLMAdapter 实例。
    """
    # 保存原始方法
    setattr(scheduler, f"{_ORIG_PREFIX}schedule", scheduler.schedule)
    setattr(scheduler, f"{_ORIG_PREFIX}update_from_output", scheduler.update_from_output)
    if hasattr(scheduler, "_free_request"):
        setattr(scheduler, f"{_ORIG_PREFIX}_free_request", scheduler._free_request)

    # Patch schedule()
    scheduler.schedule = functools.partial(_patched_schedule, scheduler, adapter)

    # Patch update_from_output()
    scheduler.update_from_output = functools.partial(
        _patched_update_from_output, scheduler, adapter
    )

    # Patch _free_request()
    if hasattr(scheduler, f"{_ORIG_PREFIX}_free_request"):
        scheduler._free_request = functools.partial(_patched_free_request, scheduler, adapter)

    # 保存 adapter 引用
    scheduler._bidkv_adapter = adapter

    # Wrap _preempt_request to count ALL preemptions (native LIFO + proactive + SRPT).
    # This provides total_all_preemptions / total_all_tokens_freed for Figure 3.
    orig_preempt = getattr(scheduler, "_preempt_request", None)
    if orig_preempt is not None:
        setattr(scheduler, f"{_ORIG_PREFIX}_preempt_request", orig_preempt)

        def _wrapped_preempt_request(  # type: ignore[misc]
            req: Any,
            now: float,
            _orig: Any = orig_preempt,
            _adapter: Any = adapter,
        ) -> Any:
            tokens = getattr(req, "num_computed_tokens", 0)
            result = _orig(req, now)
            _adapter._metrics.record_all_preemption(tokens)
            return result

        scheduler._preempt_request = _wrapped_preempt_request

    logger.info("BidKV scheduler hooks installed on vLLM Scheduler")


def uninstall_scheduler_hook(scheduler: Any, adapter: VLLMAdapter) -> None:  # noqa: ARG001
    """移除 BidKV 注入，恢复 vLLM 原始行为。

    Parameters
    ----------
    scheduler:
        vLLM v1 Scheduler 实例。
    adapter:
        VLLMAdapter 实例。
    """
    # 恢复原始方法
    orig_schedule = getattr(scheduler, f"{_ORIG_PREFIX}schedule", None)
    if orig_schedule is not None:
        scheduler.schedule = orig_schedule

    orig_update = getattr(scheduler, f"{_ORIG_PREFIX}update_from_output", None)
    if orig_update is not None:
        scheduler.update_from_output = orig_update

    orig_free = getattr(scheduler, f"{_ORIG_PREFIX}_free_request", None)
    if orig_free is not None:
        scheduler._free_request = orig_free

    orig_preempt = getattr(scheduler, f"{_ORIG_PREFIX}_preempt_request", None)
    if orig_preempt is not None:
        scheduler._preempt_request = orig_preempt

    # 清理属性
    for attr in list(vars(scheduler)):
        if attr.startswith(_ORIG_PREFIX) or attr == "_bidkv_adapter":
            delattr(scheduler, attr)

    # Close preemption logger if active (flushes and releases file handle)
    from bidkv.adapters.vllm import preemption_logger as _plogger

    _plogger.close_logger()

    logger.info("BidKV scheduler hooks removed from vLLM Scheduler")


_SCHEDULE_CALL_COUNT = 0
_DIAG_LOG = "/tmp/bidkv_diag.log"
_METRICS_FILE = "/tmp/bidkv_metrics_latest.json"


def _diag(msg: str) -> None:
    """Write diagnostic message to a file (works in subprocesses)."""
    import os

    with open(_DIAG_LOG, "a") as f:
        f.write(f"[{os.getpid()}] {msg}\n")


def _dump_metrics(adapter: VLLMAdapter) -> None:
    """Atomically dump adapter metrics to a well-known JSON file."""
    import json
    import os

    metrics = adapter.metrics.as_dict()
    # Tag with strategy + PID so runner can validate provenance
    metrics["_strategy"] = os.environ.get("BIDKV_STRATEGY", "")
    metrics["_pid"] = os.getpid()
    tmp = _METRICS_FILE + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(metrics, f)
    os.replace(tmp, _METRICS_FILE)


def _compute_keep_score(req: Any) -> float:
    """Compute keep score for a request: higher = more valuable to keep running.

    Combines completion progress, recompute-cost efficiency, and anti-starvation.
    """
    num_computed = getattr(req, "num_computed_tokens", 0)
    num_prompt = getattr(req, "num_prompt_tokens", 0)
    num_preemptions = getattr(req, "num_preemptions", 0)

    # Get max output tokens from sampling params
    max_tokens = 256  # conservative fallback
    sp = getattr(req, "sampling_params", None)
    if sp is not None:
        mt = getattr(sp, "max_tokens", None)
        if mt is not None and mt > 0:
            max_tokens = mt

    num_output = max(0, num_computed - num_prompt)
    remaining_output = max(1, max_tokens - num_output)
    completion = min(1.0, num_output / max_tokens) if max_tokens > 0 else 0.0
    efficiency = (num_computed + 1) / remaining_output
    starvation_mult = 1.0 + num_preemptions * 2.0
    keep_score = (0.1 + completion * completion) * efficiency * starvation_mult

    return keep_score


def _build_running_candidates(running: Any, adapter: VLLMAdapter) -> list[tuple[Any, Any]]:
    """Build (RequestState, vllm_request) pairs from scheduler.running.

    Returns a list so callers can map strategy decisions back to vLLM objects.
    """
    import time

    from bidkv.baselines.base import RequestState

    now_ms = time.monotonic() * 1000
    # SLO 截止时间 = 到达时间 + 120 秒 (request_timeout_s 默认值)
    slo_timeout_ms = 120_000.0

    pairs: list[tuple[Any, Any]] = []
    for req in running:
        rid = getattr(req, "request_id", None)
        if rid is None:
            continue
        token_ids = adapter._request_tokens.get(rid)

        if rid not in adapter._request_arrival_ms:
            adapter._request_arrival_ms[rid] = now_ms
        arrival_ms = adapter._request_arrival_ms[rid]

        num_prompt = getattr(req, "num_prompt_tokens", 0)
        num_computed = getattr(req, "num_computed_tokens", 0)
        num_preemptions = getattr(req, "num_preemptions", 0)
        max_output = 256  # conservative fallback
        sp = getattr(req, "sampling_params", None)
        if sp is not None:
            mt = getattr(sp, "max_tokens", None)
            if mt is not None and mt > 0:
                max_output = mt

        priority = -arrival_ms

        pairs.append(
            (
                RequestState(
                    request_id=rid,
                    current_tokens=len(token_ids) if token_ids else 0,
                    token_ids=tuple(token_ids) if token_ids else (),
                    priority=priority,
                    arrival_time_ms=arrival_ms,
                    deadline_ms=arrival_ms + slo_timeout_ms,
                    num_prompt_tokens=num_prompt,
                    num_computed_tokens=num_computed,
                    max_output_tokens=max_output,
                    num_preemptions=num_preemptions,
                ),
                req,
            )
        )
    return pairs


def _reorder_waiting_for_admission(scheduler: Any, adapter: VLLMAdapter) -> None:
    """Reorder scheduler.waiting for optimal admission under KV pressure.

    vLLM processes waiting[0] first and BREAKS on first allocation failure.
    Each strategy represents a COMPLETE scheduling approach with its own
    admission policy. The differentiation is in the key function:

    Strategy grouping by admission key:
    ┌─────────────────┬──────────────────────────────────────────────┐
    │ Strategy        │ Admission policy                             │
    ├─────────────────┼──────────────────────────────────────────────┤
    │ preempt-evict     │ FCFS — no reorder (true vLLM default)       │
    │ preempt-evict-sjf │ SJF by prompt_tokens (LIFO eviction ablation)│
    │ static-random     │ SJF by prompt_tokens                        │
    │ largest-first     │ SJF by prompt_tokens                        │
    │ bidkv             │ SJF by prompt_tokens (same as other SJF)    │
    └─────────────────┴──────────────────────────────────────────────┘

    BidKV's advantage is in quality-aware preemption via select_victims()
    using U = r / (δ + ε); not in admission ordering.
    preempt-evict is the true zero-intelligence baseline
    (FCFS waiting + LIFO preemption = vanilla vLLM behaviour).
    """
    strategy_name = adapter._experiment_strategy_name

    waiting = getattr(scheduler, "waiting", None)
    if waiting is None or len(waiting) <= 1:
        return

    if strategy_name == "preempt-evict":
        return

    else:
        # SJF by prompt_tokens (preempt-evict-sjf, static-random, largest-first, bidkv).
        waiting_list = list(waiting)
        waiting_list.sort(key=lambda r: getattr(r, "num_prompt_tokens", 0))
        waiting.clear()
        for req in waiting_list:
            waiting.append(req)


def _reorder_running_for_preemption(scheduler: Any, adapter: VLLMAdapter) -> None:
    """Reorder scheduler.running for native preemption victim selection.

    For FCFS policy, vLLM preempts with self.running.pop() — the last element.
    We put the MOST EXPENDABLE request at the END so it gets preempted first.

    Strategy differentiation through running-queue reorder:
    ┌─────────────────┬──────────────────────────────────────────────┐
    │ Strategy        │ Preemption ordering                          │
    ├─────────────────┼──────────────────────────────────────────────┤
    │ preempt-evict     │ NO reorder — pure vLLM default (LIFO)       │
    │ preempt-evict-sjf │ NO reorder — LIFO eviction (SJF ablation)   │
    │ static-random     │ select_victims(): random victim selection    │
    │ largest-first     │ select_victims(): capacity-greedy eviction   │

    │ bidkv             │ select_victims(): bid-informed priority      │
    └─────────────────┴──────────────────────────────────────────────┘
    """
    strategy_name = adapter._experiment_strategy_name

    if strategy_name == "preempt-evict":
        return

    if strategy_name == "preempt-evict-sjf":
        return

    running = getattr(scheduler, "running", None)
    if running is None or len(running) <= 1:
        return

    if strategy_name == "bidkv" and len(running) >= 2:
        # Long-context guard: avg_prompt > 500 → LIFO (empirically safer,
        # prevents target-fixation; validated in v14-v19 experiments).
        total_prompt = sum(getattr(r, "num_prompt_tokens", 0) for r in running)
        avg_prompt = total_prompt / len(running)
        if avg_prompt > 500:
            return

        # Pressure gate: below _KV_GATE, LIFO is used.
        kv_mgr = getattr(scheduler, "kv_cache_manager", None)
        if kv_mgr is not None:
            block_pool = getattr(kv_mgr, "block_pool", None)
            if block_pool is not None:
                usage = block_pool.get_usage()
                if usage < _KV_GATE:
                    return

    # Use strategy-specific cached priority from select_victims()
    cached = getattr(adapter, "_cached_preempt_priority", None)
    if cached:
        # Higher priority → FRONT (keep); lower priority → END (preempted by pop()).
        # Uncached requests default to float("inf") — protected until next refresh.
        scored = []
        for idx, req in enumerate(running):
            rid = getattr(req, "request_id", "")
            priority = cached.get(rid, float("inf"))
            scored.append((priority, idx, req))
        scored.sort(key=lambda x: (-x[0], x[1]))
        running.clear()
        for _, _, req in scored:
            running.append(req)
    else:
        # No cache yet: fall back to keep-score heuristic.
        scored = [((_compute_keep_score(req), idx), req) for idx, req in enumerate(running)]
        scored.sort(key=lambda x: -x[0][0])
        running.clear()
        for _, req in scored:
            running.append(req)


def _refresh_priority_cache(scheduler: Any, adapter: VLLMAdapter) -> None:
    """Refresh cached preemption priority using current strategy.

    Called every 3 seconds. Runs strategy.select_victims() to get the
    priority ordering, then caches it for _reorder_running_for_preemption.

    Strategy differentiation happens here: BidKV uses bid+quality scoring,
    largest-first uses positional scoring, bidkv uses full bid pipeline, etc.
    """
    import time

    refresh_interval_s = 3.0
    now = time.monotonic()
    last_refresh = getattr(adapter, "_last_priority_refresh", 0.0)
    if now - last_refresh < refresh_interval_s:
        return
    adapter._last_priority_refresh = now

    strategy = adapter._experiment_strategy
    strategy_name = adapter._experiment_strategy_name

    # preempt-evict: no priority cache needed
    if strategy_name in ("preempt-evict", "preempt-evict-sjf") or strategy is None:
        return

    running = getattr(scheduler, "running", None)
    if running is None or len(running) < 2:
        return

    pairs = _build_running_candidates(running, adapter)
    candidates = [p[0] for p in pairs]
    if len(candidates) < 2:
        return

    # Use total KV as needed_tokens to get full ordering (all candidates ranked)
    needed_all = sum(c.current_tokens for c in candidates) or 1

    try:
        actions = strategy.select_victims(candidates, needed_all)
    except Exception:  # noqa: BLE001
        return

    # Build priority: victims are expendable (low priority).
    # actions[0] = most expendable (first to be preempted).
    # Requests NOT in actions list are most valuable.
    victim_ids = [a.request_id for a in actions]
    priority: dict[str, float] = {}

    # Most expendable gets lowest score; later victims get slightly higher
    n_victims = len(victim_ids)
    for i, rid in enumerate(victim_ids):
        priority[rid] = float(i)  # 0 = most expendable

    # Requests not selected as victims: very high priority (keep)
    for pair in pairs:
        if pair[0].request_id not in priority:
            priority[pair[0].request_id] = float(n_victims + 100)

    adapter._cached_preempt_priority = priority


def _proactive_preempt(scheduler: Any, adapter: VLLMAdapter) -> None:
    """Mode A proactive preemption using vLLM native _preempt_request().

    When KV pressure exceeds threshold, selects a SINGLE victim via cached
    priority and preempts it using vLLM's native mechanism (abort + requeue
    with PREEMPTED status). The request will be rescheduled and recomputed.

    This is the core experimental variable: different strategies select
    DIFFERENT victims. BidKV picks the most efficient candidate; baselines
    use simpler heuristics.

    Guards:
    - KV utilization > 90%
    - Waiting queue non-empty (demand for blocks)
    - At least 3 running requests (keep 2+ running)
    - 5-second cooldown (prevent eviction storm)
    - preempt-evict: skip entirely (it IS the "no intervention" baseline)
    """
    import time

    strategy_name = adapter._experiment_strategy_name
    cooldown_s = 5.0

    now = time.monotonic()
    last_proactive = getattr(scheduler, "_bidkv_last_proactive", 0.0)
    if now - last_proactive < cooldown_s:
        return

    # Must have waiting requests (demand for blocks)
    waiting = getattr(scheduler, "waiting", None)
    if not waiting:
        return

    # Must have >= 3 running
    running = getattr(scheduler, "running", None)
    if running is None or len(running) < 3:
        return

    # Check KV utilization
    kv_mgr = getattr(scheduler, "kv_cache_manager", None)
    if kv_mgr is None:
        return
    block_pool = getattr(kv_mgr, "block_pool", None)
    if block_pool is None:
        return
    usage = block_pool.get_usage()
    if usage < 0.90:
        return

    # preempt-evict: skip — measures vanilla framework behavior
    if strategy_name in ("preempt-evict", "preempt-evict-sjf"):
        return

    # bidkv: skip proactive preempt — relies on running reorder (KV>95%) +
    # vLLM native preemption only. v9 experiment confirmed that enabling
    # proactive preempt for BidKV causes eviction storms (see issue #054).
    # This guard was accidentally deleted in commit fdd9a3f.
    if strategy_name == "bidkv":
        return

    cached = getattr(adapter, "_cached_preempt_priority", None)
    if not cached:
        return

    best_victim_req = None
    best_victim_idx = -1
    best_priority = float("inf")
    for i, req in enumerate(running):
        rid = getattr(req, "request_id", None)
        if rid is None:
            continue
        p = cached.get(rid, float("inf"))
        if p < best_priority:
            best_priority = p
            best_victim_req = req
            best_victim_idx = i

    if best_victim_req is None or best_victim_idx < 0:
        return

    victim_id = getattr(best_victim_req, "request_id", "")
    freed_estimate = getattr(best_victim_req, "num_computed_tokens", 0)

    preempt_fn = getattr(scheduler, "_preempt_request", None)
    if preempt_fn is None:
        return

    try:
        running.pop(best_victim_idx)
        preempt_fn(best_victim_req, now)
    except Exception:  # noqa: BLE001
        running.append(best_victim_req)
        return

    # Remove from prev_step_scheduled_req_ids so that when orig() picks up
    # this request from waiting as "resumed", the assertion at
    # scheduler.py:1051 (assert not scheduled_in_prev_step) won't fire.
    prev_ids = getattr(scheduler, "prev_step_scheduled_req_ids", None)
    if prev_ids is not None:
        prev_ids.discard(victim_id)

    scheduler._bidkv_last_proactive = now
    adapter._metrics.record_eviction(victim_id, freed_estimate)
    _diag(
        f"proactive PREEMPT: strategy={strategy_name} "
        f"victim={victim_id} usage={usage:.2f} freed~{freed_estimate}"
    )


_MODEL_EXECUTOR_RESOLVED = False


def _resolve_model_executor(scheduler: Any, adapter: VLLMAdapter) -> None:  # noqa: ARG001
    """Lazily discover model_executor via gc.get_referrers.

    Plugin cannot patch EngineCore.__init__ effectively because
    load_general_plugins() runs INSIDE the already-executing __init__.
    Instead, on the first schedule() call (after __init__ completes),
    walk gc.get_referrers(scheduler) to find EngineCore and grab its
    model_executor for block-table sync after truncation.
    """
    global _MODEL_EXECUTOR_RESOLVED
    if _MODEL_EXECUTOR_RESOLVED:
        return
    _MODEL_EXECUTOR_RESOLVED = True

    # gc.get_referrers returns 0 in vLLM's spawned EngineCore subprocess.
    # Walk the call stack instead: schedule() is called by EngineCore.step()
    # or similar, whose `self` has model_executor.
    import sys

    frame = sys._getframe(1)  # caller of _patched_schedule
    while frame is not None:
        self_obj = frame.f_locals.get("self")
        if self_obj is not None:
            me = getattr(self_obj, "model_executor", None)
            if me is not None:
                adapter._model_executor = me
                _diag(f"resolved model_executor via stack walk (frame={frame.f_code.co_name})")
                return
        frame = frame.f_back
    _diag("WARN: could not resolve model_executor via stack walk")


def _get_max_tokens_estimate(req: Any) -> int:
    """Estimate max output tokens from request's sampling_params."""
    sp = getattr(req, "sampling_params", None)
    if sp is not None:
        mt = getattr(sp, "max_tokens", None)
        if mt is not None and mt > 0:
            return mt
    return 256  # conservative fallback


def _proactive_srpt(scheduler: Any, adapter: VLLMAdapter) -> None:
    """Proactive SRPT: preempt high-remaining-cost running for low-cost waiting.

    Excluded: FCFS strategies (preempt-evict, preempt-evict-sjf).

    Guards: KV > 80%, waiting non-empty, ≥3 running, victim ≥10 output tokens,
    remaining(running) > 1.2× total(waiting), 1.5s cooldown.
    """
    import time

    strategy_name = adapter._experiment_strategy_name

    if strategy_name in ("preempt-evict", "preempt-evict-sjf"):
        return

    now = time.monotonic()
    last_srpt = getattr(scheduler, "_bidkv_last_srpt", 0.0)
    if now - last_srpt < 1.5:
        return

    running = getattr(scheduler, "running", None)
    waiting = getattr(scheduler, "waiting", None)
    if not running or not waiting or len(running) < 3:
        return

    # KV pressure check
    kv_mgr = getattr(scheduler, "kv_cache_manager", None)
    if kv_mgr is None:
        return
    block_pool = getattr(kv_mgr, "block_pool", None)
    if block_pool is None:
        return
    usage = block_pool.get_usage()
    if usage < 0.80:
        return

    # BidKV skips SRPT: recompute cost makes extra evictions counterproductive.
    # Running reorder (KV>95% gate) is the only BidKV intervention.
    if strategy_name == "bidkv":
        return

    # Find best waiting candidate (smallest total cost)
    best_waiting = None
    best_waiting_cost = float("inf")
    for req in waiting:
        prompt = getattr(req, "num_prompt_tokens", 0)
        max_out = _get_max_tokens_estimate(req)
        total = prompt + max_out
        if total < best_waiting_cost:
            best_waiting_cost = total
            best_waiting = req

    if best_waiting is None:
        return

    # Find worst running candidate (highest remaining cost, with min output guard)
    worst_running = None
    worst_remaining = 0
    worst_idx = -1
    for i, req in enumerate(running):
        prompt = getattr(req, "num_prompt_tokens", 0)
        computed = getattr(req, "num_computed_tokens", 0)
        output_so_far = max(0, computed - prompt)

        # Must have generated at least 10 output tokens
        if output_so_far < 10:
            continue

        # Anti-starvation: skip requests already preempted 2+ times
        if getattr(req, "num_preemptions", 0) >= 2:
            continue

        max_out = _get_max_tokens_estimate(req)
        remaining = max(0, max_out - output_so_far)

        if remaining > worst_remaining:
            worst_remaining = remaining
            worst_running = req
            worst_idx = i

    if worst_running is None:
        return

    # Benefit check: remaining(running) > 1.2× total(waiting)
    if worst_remaining < best_waiting_cost * 1.2:
        return

    preempt_fn = getattr(scheduler, "_preempt_request", None)
    if preempt_fn is None:
        return

    victim_id = getattr(worst_running, "request_id", "?")
    try:
        running.pop(worst_idx)
        preempt_fn(worst_running, now)
    except Exception:  # noqa: BLE001
        running.append(worst_running)
        return

    # Remove from prev_step_scheduled_req_ids so that when orig() picks up
    # this request from waiting as "resumed", the assertion at
    # scheduler.py:1051 (assert not scheduled_in_prev_step) won't fire.
    prev_ids = getattr(scheduler, "prev_step_scheduled_req_ids", None)
    if prev_ids is not None:
        prev_ids.discard(victim_id)

    scheduler._bidkv_last_srpt = now
    adapter._metrics.record_eviction(victim_id, worst_remaining)
    _diag(
        f"SRPT preempt: strategy={strategy_name} victim={victim_id} "
        f"remaining={worst_remaining} waiting_cost={best_waiting_cost} "
        f"usage={usage:.2f}"
    )


def _patched_schedule(
    scheduler: Any,
    adapter: VLLMAdapter,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Patched schedule() — Mode A: SJF admission + strategy-specific preemption.

    Flow:
    1. Sync request tracking (running + waiting)
    2. Track waiting arrival times
    3. Reorder waiting queue by strategy-specific SJF key
    4. Refresh preemption priority cache (select_victims)
    5. Proactive preemption via cached priority (all except preempt-evict)
    6. Proactive SRPT preemption (SJF strategies)
    7. Reorder running list (strategy-specific victim ordering)
    8. Call original schedule()

    Strategy differentiation hierarchy:
    - preempt-evict: FCFS admission + LIFO preemption (true vLLM default)
    - static-random/largest-first/bidkv: SJF(prompt) admission
      + strategy-specific select_victims() reorder + SRPT
    - preempt-evict-sjf: SJF(prompt) admission + LIFO preemption (ablation)
    - BidKV's edge: quality-aware U = r / (δ + ε) via completion-ratio δ
    """
    global _SCHEDULE_CALL_COUNT
    _SCHEDULE_CALL_COUNT += 1

    # Lazy-resolve model_executor on first call (after EngineCore.__init__)
    if adapter._model_executor is None:
        _resolve_model_executor(scheduler, adapter)
    if _SCHEDULE_CALL_COUNT <= 3 or _SCHEDULE_CALL_COUNT % 1000 == 0:
        used, total = adapter.get_kv_stats()
        tracked = len(adapter._request_tokens)
        running = len(getattr(scheduler, "running", []))
        pct = (used / total * 100) if total > 0 else 0.0
        _diag(
            f"schedule() #{_SCHEDULE_CALL_COUNT}: "
            f"kv={used}/{total} ({pct:.1f}%) "
            f"running={running} tracked={tracked}"
        )
        _dump_metrics(adapter)
    # Feature OFF 快速路径
    if not adapter.config.is_active:
        orig = getattr(scheduler, f"{_ORIG_PREFIX}schedule")
        return orig(*args, **kwargs)

    # 同步 request tracking
    _sync_request_tracking(scheduler, adapter)
    _track_waiting_arrival(scheduler, adapter)
    _reorder_waiting_for_admission(scheduler, adapter)
    _refresh_priority_cache(scheduler, adapter)

    # Motivation experiment: log preemption candidate snapshot (opt-in via env var).
    # Called BEFORE any reordering so all strategies see the same raw queue.
    # Activated only when BIDKV_LOG_PREEMPTION_EVENTS is set (zero overhead otherwise).
    from bidkv.adapters.vllm import preemption_logger as _plogger

    if _plogger.is_active():
        _plogger.log_event_if_enabled(scheduler, adapter)

    _proactive_preempt(scheduler, adapter)
    _proactive_srpt(scheduler, adapter)
    _reorder_running_for_preemption(scheduler, adapter)

    orig = getattr(scheduler, f"{_ORIG_PREFIX}schedule")
    result = orig(*args, **kwargs)

    return result


def _patched_update_from_output(
    scheduler: Any,
    adapter: VLLMAdapter,
    scheduler_output: Any,
    model_runner_output: Any,
) -> Any:
    """Patched update_from_output() — decode step 后更新 positional scoring。

    在调用原始 update_from_output 后，遍历 running requests，
    为每个请求更新 token tracking。
    """
    orig = getattr(scheduler, f"{_ORIG_PREFIX}update_from_output")
    result = orig(scheduler_output, model_runner_output)

    # Feature OFF 快速路径
    if not adapter.config.is_active:
        return result

    # 更新追踪信息：将新生成的 token 加入追踪
    _update_token_tracking_from_output(scheduler, adapter, model_runner_output)

    # 更新 positional scoring 累积注意力统计
    # Only run for strategies that use positional scoring (avoids wasting CPU
    # on strategies that don't benefit from cumulative attention data).
    # Note: BidKV Mode A uses completion-ratio δ, not positional token scores.
    # Sampled at 20% (every 5th step) to reduce CPU overhead.
    _POSITIONAL_STRATEGIES = ("largest-first",)
    if adapter._experiment_strategy_name in _POSITIONAL_STRATEGIES:
        positional_counter = getattr(scheduler, "_bidkv_positional_counter", 0) + 1
        scheduler._bidkv_positional_counter = positional_counter
        if positional_counter % 5 == 0:
            from bidkv.adapters.vllm.positional_hook import update_positional_from_output

            update_positional_from_output(adapter, scheduler, model_runner_output)

    return result


def _patched_free_request(scheduler: Any, adapter: VLLMAdapter, request: Any, **kwargs: Any) -> Any:
    """Patched _free_request() — 请求完成时清理 BidKV 状态。"""
    request_id = getattr(request, "request_id", None)
    if request_id is not None:
        # 记录最终 output token 数，用于前置实验 completion_ratio join
        from bidkv.adapters.vllm import preemption_logger as _plogger
        if _plogger.is_active():
            num_computed = getattr(request, "num_computed_tokens", 0)
            num_prompt = getattr(request, "num_prompt_tokens", 0)
            final_output = max(0, num_computed - num_prompt)
            _plogger.log_completion(request_id, final_output)

        adapter.on_request_complete(request_id)

    # 调用原始方法（透传所有额外参数，如 delay_free_blocks）
    orig = getattr(scheduler, f"{_ORIG_PREFIX}_free_request")
    return orig(request, **kwargs)


def _track_waiting_arrival(scheduler: Any, adapter: VLLMAdapter) -> None:
    """Record arrival time for waiting requests.

    Needed for anti-starvation (bidkv).
    Only records on first sight — does not overwrite.
    """
    import time

    waiting = getattr(scheduler, "waiting", None)
    if waiting is None:
        return

    now_ms = time.monotonic() * 1000
    for request in waiting:
        rid = getattr(request, "request_id", None)
        if rid is not None and rid not in adapter._request_arrival_ms:
            adapter._request_arrival_ms[rid] = now_ms


def _sync_request_tracking(scheduler: Any, adapter: VLLMAdapter) -> None:
    """同步 vLLM 的活跃请求到 adapter 的 tracking。

    遍历 scheduler.running，确保所有 running request 都被追踪。
    """
    running = getattr(scheduler, "running", [])
    tracked = set(adapter.get_tracked_requests())

    for request in running:
        req_id = getattr(request, "request_id", None)
        if req_id is None:
            continue
        if req_id not in tracked:
            # 新请求：从 request 中提取 token ids
            token_ids = _extract_token_ids(request)
            if token_ids:
                adapter.track_request(req_id, token_ids)


def _extract_token_ids(request: Any) -> list[int]:
    """从 vLLM Request 对象中提取 token IDs。

    vLLM v1 Request 有 ``prompt_token_ids`` 和 ``output_token_ids`` 属性。
    """
    prompt_ids = getattr(request, "prompt_token_ids", None)
    output_ids = getattr(request, "output_token_ids", None)

    token_ids: list[int] = []
    if prompt_ids is not None:
        token_ids.extend(prompt_ids)
    if output_ids is not None:
        token_ids.extend(output_ids)

    return token_ids


def _update_token_tracking_from_output(
    scheduler: Any,  # noqa: ARG001
    adapter: VLLMAdapter,
    model_runner_output: Any,
) -> None:
    """从 model_runner_output 中更新 token tracking。

    在每个 decode step 后，将新生成的 token 加入追踪。
    """
    sampled_token_ids = getattr(model_runner_output, "sampled_token_ids", None)
    if sampled_token_ids is None:
        return

    req_id_to_index = getattr(model_runner_output, "req_id_to_index", None)
    if req_id_to_index is None:
        return

    for req_id, req_index in req_id_to_index.items():
        tokens = adapter._request_tokens.get(req_id)
        if tokens is None:
            continue
        # 添加新 token
        if req_index < len(sampled_token_ids):
            new_token_ids = sampled_token_ids[req_index]
            if new_token_ids:
                tokens.extend(new_token_ids)
