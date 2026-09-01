"""Preemption event logger — 用于论文前置实验（驱逐候选异质性）。

在同一次 vLLM 运行中，于每个 KV 压力事件点：
  1. 快照所有 running 请求的状态
  2. 同时调用 4 种策略的 select_victims()（候选集相同 → 公平对比）
  3. 将事件写入 JSONL 文件

激活方式（环境变量）：
  BIDKV_LOG_PREEMPTION_EVENTS=/path/to/log.jsonl

图坐标系设计：
  X 轴：completion_ratio = num_output_tokens / max_output_tokens
        （低 = 刚起步 = 质量代价低 → 左侧更好驱逐）
  Y 轴：tokens_freed  = num_computed_tokens
        （高 = 释放更多 KV 空间 → 上方更好驱逐）
  理想驱逐目标：左上角（高 Y + 低 X）

触发条件：KV 使用率 > LOG_KV_THRESHOLD，最高 MAX_EVENTS_PER_SEC 个事件/秒。
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bidkv.adapters.vllm.adapter import VLLMAdapter

logger = logging.getLogger(__name__)

# 触发阈值：KV 使用率超过此值才记录事件
LOG_KV_THRESHOLD: float = float(os.environ.get("BIDKV_LOG_KV_THRESHOLD", "0.80"))
# 最小记录间隔（秒）: 1.0 → 最多 1 event/s
_MIN_INTERVAL_S: float = 1.0
# 单次运行最大事件数（防止磁盘撑满）
_MAX_EVENTS: int = 2000

# 模块级状态
_log_path: str = ""
_log_fh: Any = None
_last_event_ts: float = 0.0
_event_count: int = 0
_initialized: bool = False

# Completion 日志（用于 post-hoc join 计算真实 completion_ratio）
_comp_log_path: str = ""
_comp_log_fh: Any = None

# 策略实例（懒加载，避免循环导入）
_strategies: dict[str, Any] | None = None


def _get_strategies() -> dict[str, Any]:
    """懒加载策略实例（首次调用时创建）。"""
    global _strategies
    if _strategies is not None:
        return _strategies
    from bidkv.baselines.bidkv_strategy import BidKVStrategy
    from bidkv.baselines.largest_first import LargestFirstStrategy
    from bidkv.baselines.preempt_evict import PreemptEvictStrategy
    from bidkv.baselines.preempt_evict_sjf import PreemptEvictSJFStrategy

    _strategies = {
        "pe-lifo": PreemptEvictStrategy(),
        "largest-first": LargestFirstStrategy(),
        "pe-sjf": PreemptEvictSJFStrategy(),
        "bidkv": BidKVStrategy(),
    }
    return _strategies


def init_if_needed() -> bool:
    """初始化日志文件（幂等）。返回 True 表示已激活。"""
    global _log_path, _log_fh, _initialized, _comp_log_path, _comp_log_fh
    if _initialized:
        return _log_fh is not None
    _initialized = True
    path = os.environ.get("BIDKV_LOG_PREEMPTION_EVENTS", "").strip()
    if not path:
        return False
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _log_path = path
    try:
        _log_fh = open(path, "a", buffering=1)  # line-buffered  # noqa: SIM115
        logger.info("BidKV preemption logger activated → %s", path)
    except OSError as exc:
        logger.warning("BidKV preemption logger: cannot open %s: %s", path, exc)
        return False

    # 同时打开 completion 日志（用于 post-hoc join）
    comp_path = os.environ.get("BIDKV_LOG_COMPLETIONS", "").strip()
    if not comp_path:
        # 默认放在同目录，文件名 completions.jsonl
        comp_path = os.path.join(os.path.dirname(os.path.abspath(path)), "completions.jsonl")
    _comp_log_path = comp_path
    try:
        _comp_log_fh = open(comp_path, "a", buffering=1)  # noqa: SIM115
        logger.info("BidKV completion logger activated → %s", comp_path)
    except OSError as exc:
        logger.warning("BidKV completion logger: cannot open %s: %s", comp_path, exc)
    return True


def is_active() -> bool:
    """快速检查日志是否已激活（避免每调度帧做文件 I/O）。"""
    return bool(_log_fh) or bool(os.environ.get("BIDKV_LOG_PREEMPTION_EVENTS", "").strip())


def close_logger() -> None:
    """关闭日志文件（可在 adapter teardown 时调用）。"""
    global _log_fh, _comp_log_fh, _initialized
    if _log_fh is not None:
        _log_fh.close()
        _log_fh = None
    if _comp_log_fh is not None:
        _comp_log_fh.close()
        _comp_log_fh = None
    _initialized = False


def log_completion(request_id: str, final_output_tokens: int) -> None:
    """记录请求完成时的最终 output token 数（用于 post-hoc join）。

    在 _patched_free_request() 中调用，此时 request 的 num_computed_tokens
    已包含全部 prompt + output tokens。
    """
    global _comp_log_fh
    if _comp_log_fh is None:
        return
    with suppress(OSError):
        _comp_log_fh.write(
            json.dumps({"request_id": request_id, "final_output_tokens": final_output_tokens})
            + "\n"
        )


# ---------------------------------------------------------------------------
# 主要对外接口
# ---------------------------------------------------------------------------


def log_event_if_enabled(scheduler: Any, adapter: VLLMAdapter) -> None:
    """在 _patched_schedule() 的「重排序前」时机记录一次事件快照。

    设计选择：在 running 队列被任何策略重排序之前拍照，
    保证所有策略看到的候选集完全相同 → 公平比较。

    Parameters
    ----------
    scheduler:
        vLLM Scheduler 实例。
    adapter:
        VLLMAdapter 实例。
    """
    global _last_event_ts, _event_count

    # 快速路径：检查是否激活
    if not is_active():
        return
    if not init_if_needed():
        return
    if _event_count >= _MAX_EVENTS:
        return

    # 速率限制
    now = time.monotonic()
    if now - _last_event_ts < _MIN_INTERVAL_S:
        return

    # 检查 KV 使用率
    kv_mgr = getattr(scheduler, "kv_cache_manager", None)
    if kv_mgr is None:
        return
    block_pool = getattr(kv_mgr, "block_pool", None)
    if block_pool is None:
        return
    try:
        usage = block_pool.get_usage()
    except Exception:  # noqa: BLE001
        return
    if usage < LOG_KV_THRESHOLD:
        return

    running = getattr(scheduler, "running", None)
    if not running or len(running) < 2:
        return

    _last_event_ts = now
    _event_count += 1

    # 构建候选快照
    candidates = _build_candidates(running, adapter, now_ms=now * 1000)
    if len(candidates) < 2:
        return

    # 对同一候选集运行所有策略（模拟决策）
    strategy_choices = _simulate_strategies(candidates, now_ms=now * 1000)

    event: dict[str, Any] = {
        "event_id": _event_count,
        "ts": round(now, 4),
        "kv_usage": round(usage, 4),
        "num_candidates": len(candidates),
        "candidates": candidates,
        "strategy_choices": strategy_choices,
    }

    try:
        _log_fh.write(json.dumps(event) + "\n")
    except OSError as exc:
        logger.warning("preemption logger write error: %s", exc)


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _build_candidates(running: Any, adapter: VLLMAdapter, now_ms: float) -> list[dict]:
    """将 running 队列的每个请求转为候选快照字典。

    axes:
      completion_ratio = num_output_tokens / max_output_tokens   (X 轴)
      tokens_freed     = num_computed_tokens                     (Y 轴)
    两个指标均以相同公式计算，与策略无关 → 公平。
    """
    result: list[dict] = []
    for req in running:
        rid = getattr(req, "request_id", None)
        if rid is None:
            continue

        num_prompt = getattr(req, "num_prompt_tokens", 0)
        num_computed = getattr(req, "num_computed_tokens", 0)
        num_output = max(0, num_computed - num_prompt)
        num_preemptions = getattr(req, "num_preemptions", 0)

        sp = getattr(req, "sampling_params", None)
        max_tokens = 256
        if sp is not None:
            mt = getattr(sp, "max_tokens", None)
            if mt is not None and mt > 0:
                max_tokens = mt

        # X 轴：完成度（0=刚开始，1=接近完成）
        completion_ratio = round(min(1.0, num_output / max_tokens) if max_tokens > 0 else 0.0, 4)

        # Y 轴：可释放 KV token 数（= 当前 KV cache 中的 token）
        tokens_freed = num_computed

        # BidKV utility（真实公式，与 BidKVStrategy.select_victims Mode A 对齐）
        # U = output_tokens / (recompute_norm + late_penalty + starvation)
        if num_output > 2:
            _recompute_norm = max(0.5, num_prompt / 256.0)
            _late_penalty = completion_ratio * 2.0
            _starvation = num_preemptions * 0.5
            _quality_delta = max(0.1, _recompute_norm + _late_penalty + _starvation)
            utility = round(num_output / _quality_delta, 2)
        else:
            # fallback: fallback 分支（output 太少，用 computed）
            utility = round(tokens_freed / max(0.1, 1.0 + num_preemptions * 0.5), 2)

        # 到达时间（用于 RequestState 构建，以及 LIFO priority）
        arrival_ms = adapter._request_arrival_ms.get(rid, now_ms)
        age_ms = round(now_ms - arrival_ms, 1)

        result.append(
            {
                "request_id": rid,
                # 图轴
                "completion_ratio": completion_ratio,
                "tokens_freed": tokens_freed,
                # 辅助字段
                "num_prompt_tokens": num_prompt,
                "num_output_tokens": num_output,
                "num_computed_tokens": num_computed,
                "max_output_tokens": max_tokens,
                "num_preemptions": num_preemptions,
                "utility_bidkv": utility,
                # 用于 RequestState
                "arrival_ms": arrival_ms,
                "age_ms": age_ms,
            }
        )
    return result


def _simulate_strategies(candidates: list[dict], now_ms: float) -> dict[str, str | None]:
    """对同一候选集运行所有策略，返回各策略选择的 request_id。

    候选集完全相同 → 策略间的差异只来自决策逻辑，而非信息差异。
    """
    from bidkv.baselines.base import RequestState

    slo_timeout_ms = 120_000.0

    # 构建 RequestState 列表（所有策略共享，不做副本）
    req_states: list[RequestState] = []
    for c in candidates:
        req_states.append(
            RequestState(
                request_id=c["request_id"],
                current_tokens=c["tokens_freed"],  # = num_computed_tokens
                priority=-c["arrival_ms"],  # 负到达时间 → 新请求优先级最低 = LIFO
                arrival_time_ms=c["arrival_ms"],
                deadline_ms=c["arrival_ms"] + slo_timeout_ms,
                token_ids=(),
                num_prompt_tokens=c["num_prompt_tokens"],
                num_computed_tokens=c["num_computed_tokens"],
                max_output_tokens=c["max_output_tokens"],
                num_preemptions=c["num_preemptions"],
            )
        )

    # needed_tokens = 全部 token（要全排序）
    needed = sum(c["tokens_freed"] for c in candidates) or 1

    strategies = _get_strategies()
    choices: dict[str, str | None] = {}

    for name, strategy in strategies.items():
        try:
            actions = strategy.select_victims(req_states, needed, now_ms=now_ms)
            choices[name] = actions[0].request_id if actions else None
        except Exception as exc:  # noqa: BLE001
            logger.debug("strategy %s simulation error: %s", name, exc)
            choices[name] = None

    return choices
