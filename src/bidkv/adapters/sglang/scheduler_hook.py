"""SGLang Scheduler Hook — Request-level 调度注入（对称 vLLM Mode A）。

**Mode A 架构**：
策略控制 *WHO* gets preempted，执行机制是 SGLang 原生 eviction。
不做 token-level 部分释放。

注入点：``get_next_batch_to_run()``
1. 同步 request tracking（running + waiting）
2. 记录 waiting 到达时间
3. 重排 waiting queue（策略决定 admission 排序）
4. 刷新 preemption 优先级缓存（每 3s，调用 strategy.select_victims()）
5. Proactive SRPT（仅 SJF 策略：bidkv）
6. 重排 running 列表（影响 SGLang native eviction victim 选择）
7. Proactive preempt（KV > 90%，跳过 preempt-evict）
8. 调用原始 ``get_next_batch_to_run()``

策略分化表（3 策略）：

+-----------------------+-----------------+------------+----------+
| 层面                  | sglang_default  | static-random| bidkv    |
+=======================+=================+============+==========+
| Waiting 排序          | FCFS            | SJF(prompt)| SJF      |
| Running 排序          | no reorder      | cached     | cached   |
| select_victims        | N/A             | random     | U=r/(δ+ε)|
| SRPT 主动驱逐         | ❌              | ❌          | ✅        |
| Proactive             | ❌              | ✅          | ✅        |
+-----------------------+-----------------+------------+----------+

设计原则：
- **纯 Mode A**：策略只做 decision（谁被 preempt），不做 execution
- **生效双路径**：proactive preempt（主动）+ reorder（影响 native eviction）
- **Feature OFF 零开销**：BidKV 未激活时直接调用原始方法
- **可逆**：``uninstall_scheduler_hook()`` 可恢复原始方法
"""

from __future__ import annotations

import contextlib
import functools
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bidkv.adapters.sglang.adapter import SGLangAdapter

logger = logging.getLogger(__name__)

# 用于存储原始方法的属性名前缀
_ORIG_PREFIX = "_bidkv_orig_"

_SCHEDULE_CALL_COUNT = 0
_DIAG_LOG = "/tmp/bidkv_sglang_diag.log"
_METRICS_FILE = "/tmp/bidkv_sglang_metrics_latest.json"


def _diag(msg: str) -> None:
    """Write diagnostic message to a file (works in subprocesses)."""
    import os

    with open(_DIAG_LOG, "a") as f:
        f.write(f"[{os.getpid()}] {msg}\n")


def _dump_metrics(adapter: SGLangAdapter) -> None:
    """Atomically dump adapter metrics to a well-known JSON file."""
    import json
    import os

    metrics = adapter.metrics.as_dict()
    metrics["_strategy"] = adapter._experiment_strategy_name
    metrics["_pid"] = os.getpid()
    tmp = _METRICS_FILE + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(metrics, f)
    os.replace(tmp, _METRICS_FILE)


def install_scheduler_hook(scheduler: Any, adapter: SGLangAdapter) -> None:
    """将 BidKV 请求级调度逻辑注入到 SGLang Scheduler。

    Monkey-patch ``get_next_batch_to_run()`` 方法，在 native batch selection
    前执行 request-level 调度决策。

    ALL strategies install hooks for fair comparison.
    sglang_default hooks do FCFS (no reorder) — identical to vanilla SGLang
    scheduling, but with the same infrastructure overhead for fairness.

    Parameters
    ----------
    scheduler:
        SGLang ``Scheduler`` 实例。
    adapter:
        ``SGLangAdapter`` 实例。
    """
    if not hasattr(scheduler, "get_next_batch_to_run"):
        raise RuntimeError(
            "SGLang Scheduler does not have 'get_next_batch_to_run' method. "
            "This may indicate an incompatible SGLang version."
        )

    # 保存原始方法
    setattr(
        scheduler,
        f"{_ORIG_PREFIX}get_next_batch_to_run",
        scheduler.get_next_batch_to_run,
    )

    # Patch get_next_batch_to_run()
    scheduler.get_next_batch_to_run = functools.partial(
        _patched_get_next_batch_to_run, scheduler, adapter
    )

    # 保存 adapter 引用
    scheduler._bidkv_adapter = adapter

    logger.info("BidKV scheduler hooks installed on SGLang Scheduler")


def uninstall_scheduler_hook(scheduler: Any) -> None:
    """移除 BidKV 的 scheduler hook，恢复原始方法。

    Parameters
    ----------
    scheduler:
        SGLang ``Scheduler`` 实例。
    """
    orig = getattr(scheduler, f"{_ORIG_PREFIX}get_next_batch_to_run", None)
    if orig is not None:
        scheduler.get_next_batch_to_run = orig
        logger.info("SGLang scheduler hook uninstalled (get_next_batch_to_run)")

    # 清理属性
    for attr in list(vars(scheduler)):
        if attr.startswith(_ORIG_PREFIX) or attr == "_bidkv_adapter":
            with contextlib.suppress(AttributeError):
                delattr(scheduler, attr)

    logger.info("BidKV scheduler hooks removed from SGLang Scheduler")


# ---------------------------------------------------------------------------
# Core patched method
# ---------------------------------------------------------------------------


def _patched_get_next_batch_to_run(scheduler: Any, adapter: SGLangAdapter) -> Any:
    """Patched get_next_batch_to_run() — Mode A request-level scheduling.

    Flow (symmetric with vLLM scheduler_hook._patched_schedule):
    1. Sync request tracking (running + waiting)
    2. Track waiting arrival times
    3. Reorder waiting queue by strategy-specific key
    4. Refresh preemption priority cache (select_victims)
    5. Proactive SRPT preemption (bidkv only)
    6. Reorder running list (strategy-specific victim ordering)
    7. Proactive preempt (KV > 90%, skip sglang_default)
    8. Call original get_next_batch_to_run()
    """
    global _SCHEDULE_CALL_COUNT
    _SCHEDULE_CALL_COUNT += 1

    if _SCHEDULE_CALL_COUNT <= 3 or _SCHEDULE_CALL_COUNT % 1000 == 0:
        used, total = adapter.get_kv_stats()
        tracked = len(adapter._request_tokens)
        running_count = _get_running_count(scheduler)
        pct = (used / total * 100) if total > 0 else 0.0
        _diag(
            f"get_next_batch_to_run() #{_SCHEDULE_CALL_COUNT}: "
            f"kv={used}/{total} ({pct:.1f}%) "
            f"running={running_count} tracked={tracked}"
        )
        with contextlib.suppress(Exception):
            _dump_metrics(adapter)

    # Feature OFF 快速路径
    if not adapter.config.is_active:
        orig = getattr(scheduler, f"{_ORIG_PREFIX}get_next_batch_to_run")
        return orig()

    # 1. 同步 request tracking
    _sync_request_tracking(scheduler, adapter)

    # 2. Track arrival time for waiting requests
    _track_waiting_arrival(scheduler, adapter)

    # 3. Reorder waiting queue: strategy-specific key
    _reorder_waiting_for_admission(scheduler, adapter)

    # 4. Refresh strategy-specific preemption priority cache (every 3s)
    _refresh_priority_cache(scheduler, adapter)

    # 5. Proactive SRPT preemption (SJF strategies only)
    _proactive_srpt(scheduler, adapter)

    # 6. Reorder running list: strategy-specific victim selection
    _reorder_running_for_preemption(scheduler, adapter)

    # 7. Proactive preempt (KV > 90%)
    _proactive_preempt(scheduler, adapter)

    # 8. Call original get_next_batch_to_run()
    orig = getattr(scheduler, f"{_ORIG_PREFIX}get_next_batch_to_run")
    return orig()


# ---------------------------------------------------------------------------
# Request tracking
# ---------------------------------------------------------------------------


def _sync_request_tracking(scheduler: Any, adapter: SGLangAdapter) -> None:
    """同步 SGLang 的活跃请求到 adapter 的 tracking。"""
    running_reqs = _get_running_requests(scheduler)
    tracked = set(adapter.get_tracked_requests())

    for req in running_reqs:
        req_id = _get_request_id(req)
        if req_id is None:
            continue
        if req_id not in tracked:
            token_ids = _extract_token_ids(req)
            if token_ids:
                adapter.track_request(req_id, token_ids)


def _track_waiting_arrival(scheduler: Any, adapter: SGLangAdapter) -> None:
    """Record arrival time for waiting requests."""
    waiting = _get_waiting_requests(scheduler)
    if not waiting:
        return

    now_ms = time.monotonic() * 1000
    for req in waiting:
        rid = _get_request_id(req)
        if rid is not None and rid not in adapter._request_arrival_ms:
            adapter._request_arrival_ms[rid] = now_ms


# ---------------------------------------------------------------------------
# Waiting queue reorder
# ---------------------------------------------------------------------------


def _reorder_waiting_for_admission(scheduler: Any, adapter: SGLangAdapter) -> None:
    """Reorder waiting queue for optimal admission under KV pressure.

    +-------------------+-----------------------------------------------+
    | Strategy          | Admission policy                              |
    +===================+===============================================+
    | sglang_default    | FCFS — no reorder (true SGLang default)       |
    | static-random     | SJF by prompt_tokens                          |
    | bidkv             | SJF by prompt_tokens                          |
    +-------------------+-----------------------------------------------+
    """
    strategy_name = adapter._experiment_strategy_name

    waiting = _get_waiting_queue_ref(scheduler)
    if waiting is None or len(waiting) <= 1:
        return

    if strategy_name in (
        "sglang_default",
        "preempt-evict",
    ):
        # FCFS — no reorder.
        return

    else:
        # SJF strategies (static-random, bidkv): SJF by prompt_tokens.
        waiting_list = list(waiting)
        waiting_list.sort(
            key=lambda r: (
                getattr(r, "num_prompt_tokens", 0)
                or getattr(r, "origin_input_ids_unpadded_length", 0)
                or 0
            )
        )
        waiting.clear()
        for req in waiting_list:
            waiting.append(req)


# ---------------------------------------------------------------------------
# Running queue reorder
# ---------------------------------------------------------------------------


def _reorder_running_for_preemption(scheduler: Any, adapter: SGLangAdapter) -> None:
    """Reorder running list for native preemption victim selection.

    +-------------------+-----------------------------------------------+
    | Strategy          | Preemption ordering                           |
    +===================+===============================================+
    | sglang_default    | NO reorder — pure SGLang default              |
    | static-random     | select_victims(): random victim selection     |
    | bidkv             | select_victims(): bid-informed priority       |
    +-------------------+-----------------------------------------------+
    """
    strategy_name = adapter._experiment_strategy_name

    # sglang_default: NO reorder — measures pure SGLang default behavior.
    # preempt-evict-sjf: NO reorder — LIFO eviction, SJF admission only ablation.
    if strategy_name in ("sglang_default", "preempt-evict", "preempt-evict-sjf"):
        return

    running_ref = _get_running_batch_requests_ref(scheduler)
    if running_ref is None or len(running_ref) <= 1:
        return

    # Use strategy-specific cached priority from select_victims()
    cached = adapter._cached_preempt_priority
    if cached:
        scored = []
        for idx, req in enumerate(running_ref):
            rid = _get_request_id(req) or ""
            priority = cached.get(rid, float("inf"))
            scored.append((priority, idx, req))
        scored.sort(key=lambda x: (-x[0], x[1]))
        running_ref.clear()
        for _, _, req in scored:
            running_ref.append(req)
    else:
        scored = [((_compute_keep_score(req), idx), req) for idx, req in enumerate(running_ref)]
        scored.sort(key=lambda x: -x[0][0])
        running_ref.clear()
        for _, req in scored:
            running_ref.append(req)


# ---------------------------------------------------------------------------
# Priority cache refresh
# ---------------------------------------------------------------------------


def _refresh_priority_cache(scheduler: Any, adapter: SGLangAdapter) -> None:
    """Refresh cached preemption priority using current strategy.

    Called every 3 seconds. Runs strategy.select_victims() to get the
    priority ordering, then caches it for _reorder_running_for_preemption.
    """
    refresh_interval_s = 0.5
    now = time.monotonic()
    if now - adapter._last_priority_refresh < refresh_interval_s:
        return
    adapter._last_priority_refresh = now

    strategy = adapter._experiment_strategy
    strategy_name = adapter._experiment_strategy_name

    # sglang_default / preempt-evict-sjf: no priority cache needed
    if (
        strategy_name in ("sglang_default", "preempt-evict", "preempt-evict-sjf")
        or strategy is None
    ):
        return

    running_reqs = _get_running_requests(scheduler)
    if len(running_reqs) < 2:
        return

    pairs = _build_running_candidates(running_reqs, adapter, scheduler=scheduler)
    candidates = [p[0] for p in pairs]
    if len(candidates) < 2:
        return

    needed_all = (
        sum(c.private_tokens if c.private_tokens > 0 else c.current_tokens for c in candidates) or 1
    )

    try:
        actions = strategy.select_victims(candidates, needed_all)
    except Exception:  # noqa: BLE001
        return

    victim_ids = [a.request_id for a in actions]
    priority: dict[str, float] = {}

    n_victims = len(victim_ids)
    for i, rid in enumerate(victim_ids):
        priority[rid] = float(i)

    for pair in pairs:
        if pair[0].request_id not in priority:
            priority[pair[0].request_id] = float(n_victims + 100)

    adapter._cached_preempt_priority = priority


# ---------------------------------------------------------------------------
# Proactive preemption
# ---------------------------------------------------------------------------


def _proactive_preempt(scheduler: Any, adapter: SGLangAdapter) -> None:
    """Proactive preemption when KV pressure exceeds threshold.

    Selects 1 victim via cached priority and aborts it.
    When usage > 95%, selects 2 victims for faster pressure relief.

    Guards:
    - KV utilization > 85%
    - Waiting queue non-empty
    - At least 3 running requests
    - 2-second cooldown
    - sglang_default: skip entirely
    """
    cooldown_s = 2.0
    now = time.monotonic()
    last_proactive = getattr(scheduler, "_bidkv_last_proactive", 0.0)
    if now - last_proactive < cooldown_s:
        return

    waiting = _get_waiting_requests(scheduler)
    if not waiting:
        return

    running_reqs = _get_running_requests(scheduler)
    if len(running_reqs) < 3:
        return

    used, total = adapter.get_kv_stats()
    if total <= 0:
        return
    usage = used / total
    if usage < 0.85:
        return

    strategy_name = adapter._experiment_strategy_name

    if strategy_name in ("sglang_default", "preempt-evict", "preempt-evict-sjf"):
        return

    cached = adapter._cached_preempt_priority
    if not cached:
        return

    # Collect candidates sorted by priority (ascending = highest eviction priority)
    scored_reqs = []
    for req in running_reqs:
        rid = _get_request_id(req)
        if rid is None:
            continue
        p = cached.get(rid, float("inf"))
        scored_reqs.append((p, req))
    scored_reqs.sort(key=lambda x: x[0])

    if not scored_reqs:
        return

    # Only proceed if the top candidate has an actual victim priority (< 10).
    # Priorities 0–(n_victims-1) are assigned by select_victims(); 100+ are
    # catch-all non-victim assignments.  Without this guard, BidKV would
    # randomly evict a non-victim when select_victims() returns empty
    # (e.g., all running requests have output_tokens ≤ 2).
    if scored_reqs[0][0] >= 10.0:
        return  # no qualified victim in priority cache

    # Select 2 victims when KV usage is critical (>95%), otherwise 1
    n_victims = 2 if usage > 0.95 else 1
    n_victims = min(n_victims, len(scored_reqs), len(running_reqs) - 2)
    if n_victims <= 0:
        return

    aborted_any = False
    for _, victim_req in scored_reqs[:n_victims]:
        victim_id = _get_request_id(victim_req) or ""
        freed_estimate = getattr(victim_req, "num_computed_tokens", 0) or 0
        if _abort_request(scheduler, victim_id):
            aborted_any = True
            adapter._metrics.record_compression(victim_id, freed_estimate)
            _diag(
                f"proactive PREEMPT: strategy={strategy_name} "
                f"victim={victim_id} usage={usage:.2f} freed~{freed_estimate}"
            )

    if aborted_any:
        scheduler._bidkv_last_proactive = now


def _proactive_srpt(scheduler: Any, adapter: SGLangAdapter) -> None:
    """Proactive SRPT: preempt high-remaining-cost running for low-cost waiting.

    FCFS strategies (sglang_default) are excluded.

    Guards:
    - KV utilization > 75%
    - Waiting queue non-empty
    - At least 3 running requests
    - Running victim must have generated >= 10 output tokens
    - Clear benefit: remaining(running) > 1.2 × total(waiting)
    - 0.8-second cooldown
    """
    strategy_name = adapter._experiment_strategy_name

    if strategy_name in (
        "sglang_default",
        "preempt-evict",
        "preempt-evict-sjf",
    ):
        return

    now = time.monotonic()
    last_srpt = getattr(scheduler, "_bidkv_last_srpt", 0.0)
    if now - last_srpt < 0.8:
        return

    running_reqs = _get_running_requests(scheduler)
    waiting = _get_waiting_requests(scheduler)
    if not running_reqs or not waiting or len(running_reqs) < 3:
        return

    used, total = adapter.get_kv_stats()
    if total <= 0:
        return
    usage = used / total
    if usage < 0.75:
        return

    best_waiting_cost = float("inf")
    for req in waiting:
        prompt = (
            getattr(req, "num_prompt_tokens", 0)
            or getattr(req, "origin_input_ids_unpadded_length", 0)
            or 0
        )
        max_out = _get_max_tokens_estimate(req)
        total_cost = prompt + max_out
        if total_cost < best_waiting_cost:
            best_waiting_cost = total_cost

    if best_waiting_cost == float("inf"):
        return

    worst_running = None
    worst_remaining = 0
    for req in running_reqs:
        prompt = getattr(req, "num_prompt_tokens", 0) or 0
        computed = getattr(req, "num_computed_tokens", 0) or 0
        output_so_far = max(0, computed - prompt)

        if output_so_far < 10:
            continue
        if getattr(req, "num_preemptions", 0) >= 2:
            continue

        max_out = _get_max_tokens_estimate(req)
        remaining = max(0, max_out - output_so_far)

        if remaining > worst_remaining:
            worst_remaining = remaining
            worst_running = req

    if worst_running is None:
        return

    if worst_remaining < best_waiting_cost * 1.2:
        return

    victim_id = _get_request_id(worst_running) or ""
    aborted = _abort_request(scheduler, victim_id)
    if not aborted:
        return

    scheduler._bidkv_last_srpt = now
    adapter._metrics.record_eviction(victim_id, worst_remaining)
    _diag(
        f"SRPT preempt: strategy={strategy_name} victim={victim_id} "
        f"remaining={worst_remaining} waiting_cost={best_waiting_cost} "
        f"usage={usage:.2f}"
    )


# ---------------------------------------------------------------------------
# Helper: build RequestState candidates
# ---------------------------------------------------------------------------


def _build_running_candidates(
    running_reqs: list[Any],
    adapter: SGLangAdapter,
    scheduler: Any | None = None,  # noqa: ARG001
) -> list[tuple[Any, Any]]:
    """Build (RequestState, sglang_request) pairs from running requests.

    When ``scheduler`` is provided, computes ``private_tokens`` for each
    request. radix_hook.py was removed (Mode B cleaned up 2026-04-08), so
    ``private_tokens`` is always 0: Branch B (v8-frozen) of the bid formula
    is always used.
    """
    from bidkv.baselines.base import RequestState

    now_ms = time.monotonic() * 1000
    slo_timeout_ms = 120_000.0

    # radix_hook.py removed (Mode B cleaned up 2026-04-08); private_tokens = 0.

    pairs: list[tuple[Any, Any]] = []
    for req in running_reqs:
        rid = _get_request_id(req)
        if rid is None:
            continue
        token_ids = adapter._request_tokens.get(rid)

        if rid not in adapter._request_arrival_ms:
            adapter._request_arrival_ms[rid] = now_ms
        arrival_ms = adapter._request_arrival_ms[rid]

        num_prompt = (
            getattr(req, "num_prompt_tokens", 0)
            or getattr(req, "origin_input_ids_unpadded_length", 0)
            or 0
        )
        num_computed = getattr(req, "num_computed_tokens", 0) or 0
        num_preemptions = getattr(req, "num_preemptions", 0) or 0
        max_output = _get_max_tokens_estimate(req)

        priority = -arrival_ms

        private_tokens = 0

        pairs.append(
            (
                RequestState(
                    request_id=rid,
                    current_tokens=(
                        num_computed if num_computed > 0 else (len(token_ids) if token_ids else 0)
                    ),
                    token_ids=tuple(token_ids) if token_ids else (),
                    priority=priority,
                    arrival_time_ms=arrival_ms,
                    deadline_ms=arrival_ms + slo_timeout_ms,
                    num_prompt_tokens=num_prompt,
                    num_computed_tokens=num_computed,
                    max_output_tokens=max_output,
                    num_preemptions=num_preemptions,
                    private_tokens=private_tokens,
                ),
                req,
            )
        )
    return pairs


def _compute_keep_score(req: Any) -> float:
    """Compute keep score: higher = more valuable to keep running."""
    num_computed = getattr(req, "num_computed_tokens", 0) or 0
    num_prompt = (
        getattr(req, "num_prompt_tokens", 0)
        or getattr(req, "origin_input_ids_unpadded_length", 0)
        or 0
    )
    num_preemptions = getattr(req, "num_preemptions", 0) or 0

    max_tokens = _get_max_tokens_estimate(req)

    num_output = max(0, num_computed - num_prompt)
    remaining_output = max(1, max_tokens - num_output)

    completion = min(1.0, num_output / max_tokens) if max_tokens > 0 else 0.0
    efficiency = (num_computed + 1) / remaining_output
    starvation_mult = 1.0 + num_preemptions * 2.0

    return (0.1 + completion) * efficiency * starvation_mult


def _get_max_tokens_estimate(req: Any) -> int:
    """Estimate max output tokens from request's sampling_params.

    All strategies have equal access to max_tokens — it is a standard
    API parameter, NOT a bid signal.
    """
    sp = getattr(req, "sampling_params", None)
    if sp is not None:
        mt = getattr(sp, "max_tokens", None) or getattr(sp, "max_new_tokens", None)
        if mt is not None and mt > 0:
            return mt
    return 256


# ---------------------------------------------------------------------------
# SGLang-specific helpers: queue access
# ---------------------------------------------------------------------------


def _get_request_id(req: Any) -> str | None:
    """Extract request ID from a SGLang request object."""
    rid = getattr(req, "rid", None) or getattr(req, "request_id", None)
    return str(rid) if rid is not None else None


def _get_waiting_requests(scheduler: Any) -> list[Any]:
    """Get list of waiting requests from SGLang scheduler."""
    waiting = getattr(scheduler, "waiting_queue", None) or getattr(scheduler, "waiting", None)
    if waiting is None:
        return []
    return list(waiting)


def _get_waiting_queue_ref(scheduler: Any) -> Any | None:
    """Get a mutable reference to the waiting queue for in-place reorder."""
    for attr in ("waiting_queue", "waiting"):
        q = getattr(scheduler, attr, None)
        if q is not None and hasattr(q, "clear") and hasattr(q, "append"):
            return q
    return None


def _get_running_requests(scheduler: Any) -> list[Any]:
    """Get list of running requests from SGLang scheduler."""
    running_batch = getattr(scheduler, "running_batch", None)
    if running_batch is not None:
        reqs = getattr(running_batch, "reqs", None)
        if reqs is not None:
            return list(reqs)
    running = getattr(scheduler, "running", None)
    if running is not None:
        return list(running)
    return []


def _get_running_batch_requests_ref(scheduler: Any) -> Any | None:
    """Get a mutable reference to running batch reqs for in-place reorder."""
    running_batch = getattr(scheduler, "running_batch", None)
    if running_batch is not None:
        reqs = getattr(running_batch, "reqs", None)
        if reqs is not None and hasattr(reqs, "clear") and hasattr(reqs, "append"):
            return reqs
    running = getattr(scheduler, "running", None)
    if running is not None and hasattr(running, "clear") and hasattr(running, "append"):
        return running
    return None


def _get_running_count(scheduler: Any) -> int:
    """Get count of running requests."""
    running_batch = getattr(scheduler, "running_batch", None)
    if running_batch is not None:
        reqs = getattr(running_batch, "reqs", None)
        if reqs is not None:
            return len(reqs)
    running = getattr(scheduler, "running", None)
    if running is not None:
        return len(running)
    return 0


def _extract_token_ids(req: Any) -> list[int]:
    """Extract token IDs from a SGLang request object."""
    for attr in ("origin_input_ids", "input_ids", "prompt_token_ids"):
        ids = getattr(req, attr, None)
        if ids is not None:
            if hasattr(ids, "tolist"):
                return ids.tolist()
            return list(ids)
    return []


def _abort_request(scheduler: Any, request_id: str) -> bool:
    """Abort a running request via SGLang native mechanism.

    Tries multiple abort paths in order of preference:
    1. scheduler.abort_request(rid)
    2. scheduler.abort_requests([rid])
    """
    abort_fn = getattr(scheduler, "abort_request", None)
    if abort_fn is not None:
        try:
            abort_fn(request_id)
            return True
        except Exception:  # noqa: BLE001
            pass

    abort_fn2 = getattr(scheduler, "abort_requests", None)
    if abort_fn2 is not None:
        try:
            abort_fn2([request_id])
            return True
        except Exception:  # noqa: BLE001
            pass

    logger.debug("_abort_request: no abort mechanism available for %s", request_id)
    return False
