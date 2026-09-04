# SPDX-License-Identifier: Apache-2.0
"""BidKV policy for vLLM's immutable preemption extension API.

Utility formula:
    U = r / (delta + epsilon)

where:
    r       = tokens freed (num_computed_tokens)
    delta   = 1 + w_c * completion + w_p * preemptions
    epsilon = small constant to avoid division by zero
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from vllm.v1.core.sched.preemption import PreemptionContext


class _Candidate(Protocol):
    request_id: str
    priority: int
    arrival_time: float
    num_output_tokens: int
    num_computed_tokens: int
    num_preemptions: int
    max_tokens: int


_DEFAULT_MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Environment variables (plugin-owned, BIDKV_UTILITY_ prefix)
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: str = "0") -> bool:
    return bool(int(os.getenv(name, default)))


def _env_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _env_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BidkvSelectorConfig:
    """Immutable configuration for the BidKV victim selector."""

    enable_utility_victim_selection: bool = False
    utility_strategy: str = "bidkv"
    utility_kill_switch: bool = False
    utility_completion_weight: float = 0.5
    utility_preempt_weight: float = 0.3
    utility_kv_gate: float = 0.0
    utility_cooldown_s: float = 0.0
    utility_min_running: int = 1
    utility_liveness_preemptions: int = 2
    utility_cascade_gain_ratio: float = 1.25
    utility_snapshot_enabled: bool = False
    utility_snapshot_top_k: int = 3
    utility_snapshot_history_size: int = 32
    utility_epsilon: float = 1e-6
    utility_default_max_tokens: int = _DEFAULT_MAX_TOKENS

    @classmethod
    def from_env(cls) -> BidkvSelectorConfig:
        return cls(
            enable_utility_victim_selection=_env_bool("BIDKV_UTILITY_ENABLE"),
            utility_strategy=os.getenv("BIDKV_UTILITY_STRATEGY", "bidkv"),
            utility_kill_switch=_env_bool("BIDKV_UTILITY_KILL_SWITCH"),
            utility_completion_weight=_env_float("BIDKV_UTILITY_COMPLETION_WEIGHT", "0.5"),
            utility_preempt_weight=_env_float("BIDKV_UTILITY_PREEMPT_WEIGHT", "0.3"),
            utility_kv_gate=_env_float("BIDKV_UTILITY_KV_GATE", "0.0"),
            utility_cooldown_s=_env_float("BIDKV_UTILITY_COOLDOWN_S", "0.0"),
            utility_min_running=_env_int("BIDKV_UTILITY_MIN_RUNNING", "1"),
            utility_liveness_preemptions=_env_int(
                "BIDKV_UTILITY_LIVENESS_PREEMPTIONS", "2"
            ),
            utility_cascade_gain_ratio=_env_float(
                "BIDKV_UTILITY_CASCADE_GAIN_RATIO", "1.25"
            ),
            utility_snapshot_enabled=_env_bool("BIDKV_UTILITY_SNAPSHOT_ENABLED"),
            utility_snapshot_top_k=_env_int("BIDKV_UTILITY_SNAPSHOT_TOP_K", "3"),
            utility_snapshot_history_size=_env_int("BIDKV_UTILITY_SNAPSHOT_HISTORY_SIZE", "32"),
            utility_epsilon=_env_float("BIDKV_UTILITY_EPSILON", "1e-6"),
            utility_default_max_tokens=_env_int(
                "BIDKV_UTILITY_DEFAULT_MAX_TOKENS",
                str(_DEFAULT_MAX_TOKENS),
            ),
        )

    @classmethod
    def from_additional_config(
        cls,
        additional_config: dict[str, Any] | None,
    ) -> BidkvSelectorConfig:
        if additional_config is None:
            return cls.from_env()
        defaults = cls.from_env()
        config_data = additional_config or {}
        config = cls(
            enable_utility_victim_selection=bool(
                config_data.get(
                    "enable_utility_victim_selection",
                    defaults.enable_utility_victim_selection,
                )
            ),
            utility_strategy=str(config_data.get("utility_strategy", defaults.utility_strategy)),
            utility_kill_switch=bool(
                config_data.get("utility_kill_switch", defaults.utility_kill_switch)
            ),
            utility_completion_weight=float(
                config_data.get(
                    "utility_completion_weight",
                    defaults.utility_completion_weight,
                )
            ),
            utility_preempt_weight=float(
                config_data.get(
                    "utility_preempt_weight",
                    defaults.utility_preempt_weight,
                )
            ),
            utility_kv_gate=float(config_data.get("utility_kv_gate", defaults.utility_kv_gate)),
            utility_cooldown_s=float(
                config_data.get("utility_cooldown_s", defaults.utility_cooldown_s)
            ),
            utility_min_running=int(
                config_data.get("utility_min_running", defaults.utility_min_running)
            ),
            utility_liveness_preemptions=int(
                config_data.get(
                    "utility_liveness_preemptions",
                    defaults.utility_liveness_preemptions,
                )
            ),
            utility_cascade_gain_ratio=float(
                config_data.get(
                    "utility_cascade_gain_ratio",
                    defaults.utility_cascade_gain_ratio,
                )
            ),
            utility_snapshot_enabled=bool(
                config_data.get(
                    "utility_snapshot_enabled",
                    defaults.utility_snapshot_enabled,
                )
            ),
            utility_snapshot_top_k=int(
                config_data.get(
                    "utility_snapshot_top_k",
                    defaults.utility_snapshot_top_k,
                )
            ),
            utility_snapshot_history_size=int(
                config_data.get(
                    "utility_snapshot_history_size",
                    defaults.utility_snapshot_history_size,
                )
            ),
            utility_epsilon=float(config_data.get("utility_epsilon", defaults.utility_epsilon)),
            utility_default_max_tokens=int(
                config_data.get(
                    "utility_default_max_tokens",
                    defaults.utility_default_max_tokens,
                )
            ),
        )
        config.validate()
        return config

    @classmethod
    def from_vllm_config(cls, vllm_config) -> BidkvSelectorConfig:
        additional_config = getattr(vllm_config, "additional_config", None) or {}
        return cls.from_additional_config(additional_config)

    def validate(self) -> None:
        if self.utility_completion_weight < 0:
            raise ValueError("utility_completion_weight must be non-negative")
        if self.utility_preempt_weight < 0:
            raise ValueError("utility_preempt_weight must be non-negative")
        if self.utility_kv_gate < 0 or self.utility_kv_gate > 1:
            raise ValueError("utility_kv_gate must be in [0, 1]")
        if self.utility_cooldown_s < 0:
            raise ValueError("utility_cooldown_s must be non-negative")
        if self.utility_min_running <= 0:
            raise ValueError("utility_min_running must be positive")
        if self.utility_liveness_preemptions < 0:
            raise ValueError("utility_liveness_preemptions must be non-negative")
        if self.utility_cascade_gain_ratio < 1:
            raise ValueError("utility_cascade_gain_ratio must be at least 1")
        if self.utility_snapshot_top_k <= 0:
            raise ValueError("utility_snapshot_top_k must be positive")
        if self.utility_snapshot_history_size <= 0:
            raise ValueError("utility_snapshot_history_size must be positive")
        if self.utility_epsilon <= 0:
            raise ValueError("utility_epsilon must be positive")
        if self.utility_default_max_tokens <= 0:
            raise ValueError("utility_default_max_tokens must be positive")

    def to_additional_config(self) -> dict[str, Any]:
        return {
            "enable_utility_victim_selection": self.enable_utility_victim_selection,
            "utility_strategy": self.utility_strategy,
            "utility_kill_switch": self.utility_kill_switch,
            "utility_completion_weight": self.utility_completion_weight,
            "utility_preempt_weight": self.utility_preempt_weight,
            "utility_kv_gate": self.utility_kv_gate,
            "utility_cooldown_s": self.utility_cooldown_s,
            "utility_min_running": self.utility_min_running,
            "utility_liveness_preemptions": self.utility_liveness_preemptions,
            "utility_cascade_gain_ratio": self.utility_cascade_gain_ratio,
            "utility_snapshot_enabled": self.utility_snapshot_enabled,
            "utility_snapshot_top_k": self.utility_snapshot_top_k,
            "utility_snapshot_history_size": self.utility_snapshot_history_size,
            "utility_epsilon": self.utility_epsilon,
            "utility_default_max_tokens": self.utility_default_max_tokens,
        }


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UtilityCandidateScore:
    """Scoring record for a single candidate request."""

    request_id: str
    utility: float
    evict_score: float
    tokens_freed: int
    completion: float
    num_preemptions: int
    arrival_time: float


# ---------------------------------------------------------------------------
# BidKV Victim Selector
# ---------------------------------------------------------------------------


class BidkvPreemptionPolicy:
    """BidKV utility-based preemption policy for vLLM.

    Implements the immutable ``PreemptionPolicy`` protocol. When utility mode
    is enabled (and gating conditions are met), preempted victims are chosen
    by maximising U = r / (delta + epsilon).  Otherwise falls back to the
    default scheduler policy (FCFS tail or highest priority).

    Five strategies:
        pe              — FCFS / LIFO, same as upstream vLLM default
        pe-sjf          — same victim as PE (SJF is admission-side)
        static-random   — random victim from running queue
        largest-first   — max num_computed_tokens
        bidkv           — utility-ranked U = r/(δ+ε)
    """

    vllm_preemption_policy_api_version = 1

    def __init__(self, config: BidkvSelectorConfig) -> None:
        self.config = config
        self._last_utility_pick_ts = -math.inf
        snapshot_size = max(1, int(self.config.utility_snapshot_history_size))
        self._total_preemptions = 0
        self._total_tokens_freed = 0
        self._kv_pressure_events = 0
        self._utility_strategy_hits = 0
        self._default_strategy_hits = 0
        self._liveness_fallback_hits = 0
        self._liveness_epochs = 0
        self._cascade_guard_hits = 0
        self._liveness_preemption_offsets: dict[str, int] = {}
        self._consecutive_preemption_events = 0
        self._consecutive_preemption_checks = 0
        self._last_preempted_request_id: str | None = None
        self._preemptions_per_request: dict[str, int] = defaultdict(int)
        self._recent_preempted_req_ids: deque[str] = deque(maxlen=snapshot_size)
        self._last_decision: dict[str, Any] | None = None
        self._decision_snapshots: deque[dict[str, Any]] = deque(maxlen=snapshot_size)
        logging.getLogger("vllm").info(
            "[BidKV] INIT | enabled=%s | strategy=%s | kv_gate=%.2f | min_running=%d | "
            "completion_w=%.2f | preempt_w=%.2f | liveness_preemptions=%d | "
            "cascade_gain_ratio=%.2f",
            self.config.enable_utility_victim_selection,
            self.config.utility_strategy,
            self.config.utility_kv_gate,
            self.config.utility_min_running,
            self.config.utility_completion_weight,
            self.config.utility_preempt_weight,
            self.config.utility_liveness_preemptions,
            self.config.utility_cascade_gain_ratio,
        )

    # -- Factory (PreemptionPolicy protocol) ------------------------------

    @classmethod
    def from_vllm_config(cls, vllm_config) -> BidkvPreemptionPolicy:
        return cls(BidkvSelectorConfig.from_vllm_config(vllm_config))

    # -- Public API (PreemptionPolicy protocol) ----------------------------

    def select_victim(self, context: PreemptionContext) -> str:
        running = context.candidates
        if not running:
            raise ValueError("running is empty, cannot pick victim")

        strategy = self.config.utility_strategy if self._utility_enabled else "pe"
        now = self._resolve_now(context.now)
        policy = context.scheduling_policy
        kv_utilization = context.kv_cache_usage

        if strategy == "bidkv":
            victim = self._pick_bidkv_victim(
                running,
                policy,
                kv_utilization,
                now,
                getattr(context, "requesting_request_id", ""),
            )
        elif strategy == "static-random":
            victim = self._pick_random_victim(running, policy, kv_utilization, now)
        elif strategy == "largest-first":
            victim = self._pick_largest_first_victim(running, policy, kv_utilization, now)
        else:
            victim = self._pick_pe_victim(running, policy, kv_utilization, now)
        return victim.request_id

    def pick_victim(
        self,
        running: Sequence[_Candidate],
        policy: Any,
        *,
        kv_utilization: float | None = None,
        now_s: float | None = None,
    ) -> _Candidate:
        """Compatibility shim for offline traces from the pre-0.28 API."""
        policy_value = getattr(policy, "value", None)
        if policy_value is None:
            policy_value = str(getattr(policy, "name", policy)).lower()
        context = _LegacyContext(
            candidates=tuple(running),
            scheduling_policy=policy_value,
            kv_cache_usage=kv_utilization or 0.0,
            now=self._resolve_now(now_s),
        )
        selected_id = self.select_victim(context)  # type: ignore[arg-type]
        return next(candidate for candidate in running if candidate.request_id == selected_id)

    def emit_observability_log(self, logger, scheduler_name: str) -> None:
        metrics = self.export_metrics()
        if metrics["total_preemptions"] <= 0:
            return
        logger.info(
            "[BidKV][%s] total_preemptions=%d utility_hits=%d "
            "default_hits=%d hit_rate=%.3f tokens_freed=%d "
            "kv_pressure_events=%d consecutive_preempt_ratio=%.3f "
            "p95_preemptions_per_request=%.2f",
            scheduler_name,
            metrics["total_preemptions"],
            metrics["utility_strategy_hits"],
            metrics["default_strategy_hits"],
            metrics["strategy_hit_rate"],
            metrics["total_tokens_freed"],
            metrics["kv_pressure_events"],
            metrics["consecutive_preempt_ratio"],
            metrics["preemptions_per_request_p95"],
        )
        if self._last_decision is not None:
            decision = self._last_decision
            logger.info(
                "[BidKV][%s] latest_decision victim=%s default_victim=%s "
                "used_utility=%s kv_utilization=%s running=%d policy=%s",
                scheduler_name,
                decision["selected_victim_id"],
                decision["default_victim_id"],
                decision["used_utility"],
                decision["kv_utilization"],
                decision["running_size"],
                decision["policy"],
            )
        if self.config.utility_snapshot_enabled:
            snapshots = self.get_recent_snapshots(limit=1)
            if snapshots:
                logger.debug(
                    "[BidKV][%s] latest_snapshot=%s",
                    scheduler_name,
                    snapshots[0],
                )

    def export_metrics(self) -> dict[str, Any]:
        hit_rate = 0.0
        if self._total_preemptions > 0:
            hit_rate = self._utility_strategy_hits / self._total_preemptions

        consecutive_preempt_ratio = 0.0
        if self._consecutive_preemption_checks > 0:
            consecutive_preempt_ratio = (
                self._consecutive_preemption_events / self._consecutive_preemption_checks
            )

        return {
            "total_preemptions": self._total_preemptions,
            "total_tokens_freed": self._total_tokens_freed,
            "kv_pressure_events": self._kv_pressure_events,
            "consecutive_preempt_ratio": consecutive_preempt_ratio,
            "preemptions_per_request_p95": self._percentile(
                self._preemptions_per_request.values(), 95
            ),
            "preempted_req_ids": list(self._recent_preempted_req_ids),
            "strategy_hit_rate": hit_rate,
            "utility_strategy_hits": self._utility_strategy_hits,
            "default_strategy_hits": self._default_strategy_hits,
            "liveness_fallback_hits": self._liveness_fallback_hits,
            "liveness_epochs": self._liveness_epochs,
            "cascade_guard_hits": self._cascade_guard_hits,
        }

    def get_recent_snapshots(self, limit: int = 10) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return list(self._decision_snapshots)[-limit:]

    # ── Strategy implementations ──────────────────────────────────

    def _pick_pe_victim(self, running, policy, kv_utilization, now) -> _Candidate:
        victim = self._pick_default_victim(running, policy)
        self._record_preemption(
            victim=victim,
            used_utility=False,
            policy=policy,
            kv_utilization=kv_utilization,
            now_s=now,
            default_victim=victim,
            ranked_candidates=[],
            running_size=len(running),
        )
        return victim

    def _pick_random_victim(self, running, policy, kv_utilization, now) -> _Candidate:
        import random as _random

        victim = _random.choice(list(running))
        logging.getLogger("vllm").info(
            "[BidKV] RANDOM | victim=%s | kv_util=%.2f | running=%d",
            getattr(victim, "request_id", "?"),
            kv_utilization or 0.0,
            len(running),
        )
        self._record_preemption(
            victim=victim,
            used_utility=True,
            policy=policy,
            kv_utilization=kv_utilization,
            now_s=now,
            default_victim=self._pick_default_victim(running, policy),
            ranked_candidates=[],
            running_size=len(running),
        )
        return victim

    def _pick_largest_first_victim(self, running, policy, kv_utilization, now) -> _Candidate:
        victim = max(running, key=lambda r: getattr(r, "num_computed_tokens", 0))
        logging.getLogger("vllm").info(
            "[BidKV] LARGEST_FIRST | victim=%s | tokens=%d | kv_util=%.2f | running=%d",
            getattr(victim, "request_id", "?"),
            getattr(victim, "num_computed_tokens", 0),
            kv_utilization or 0.0,
            len(running),
        )
        self._record_preemption(
            victim=victim,
            used_utility=True,
            policy=policy,
            kv_utilization=kv_utilization,
            now_s=now,
            default_victim=self._pick_default_victim(running, policy),
            ranked_candidates=[],
            running_size=len(running),
        )
        return victim

    def _pick_bidkv_victim(
        self,
        running,
        policy,
        kv_utilization,
        now,
        requesting_request_id: str,
    ) -> _Candidate:
        if (
            self.config.utility_kv_gate > 0
            and kv_utilization is not None
            and kv_utilization >= self.config.utility_kv_gate
        ):
            self._kv_pressure_events += 1

        default_victim = self._pick_default_victim(running, policy)
        utility_enabled = self._can_use_utility(
            kv_utilization=kv_utilization,
            now_s=now,
            running_size=len(running),
        )

        if not utility_enabled:
            victim = default_victim
            reason_parts = ["main_switch_off"] if not self._utility_enabled else []
            if len(running) < self.config.utility_min_running:
                reason_parts.append(
                    f"running({len(running)})<min({self.config.utility_min_running})"
                )
            if (
                self.config.utility_kv_gate > 0
                and (kv_utilization or 0) < self.config.utility_kv_gate
            ):
                reason_parts.append(f"kv({kv_utilization:.2f})<gate({self.config.utility_kv_gate})")
            logging.getLogger("vllm").info(
                "[BidKV] FALLBACK | victim=%s (default) | reason=%s | kv_util=%.2f | running=%d",
                victim.request_id,
                ",".join(reason_parts) or "unknown",
                kv_utilization or 0.0,
                len(running),
            )
            self._record_preemption(
                victim=victim,
                used_utility=False,
                policy=policy,
                kv_utilization=kv_utilization,
                now_s=now,
                default_victim=default_victim,
                ranked_candidates=[],
                running_size=len(running),
            )
            return victim

        ranked_candidates, req_map = self._rank_candidates(running)
        liveness_threshold = self.config.utility_liveness_preemptions
        relative_preemptions = {
            candidate.request_id: max(
                0,
                candidate.num_preemptions
                - self._liveness_preemption_offsets.get(candidate.request_id, 0),
            )
            for candidate in ranked_candidates
        }
        if (
            liveness_threshold > 0
            and len(ranked_candidates) > 1
            and all(
                relative_preemptions[candidate.request_id] >= liveness_threshold
                for candidate in ranked_candidates
            )
        ):
            # Bound utility rotation without permanently disabling utility.
            # This is an epoch barrier: one self/default decision lets the
            # current scheduling pass stop cascading, then offsets advance so
            # subsequent pressure decisions return to utility ranking. A new
            # barrier is possible only after every still-runnable request has
            # paid another bounded number of preemptions.
            victim = req_map.get(requesting_request_id, default_victim)
            self._liveness_fallback_hits += 1
            self._liveness_epochs += 1
            for candidate in ranked_candidates:
                self._liveness_preemption_offsets[candidate.request_id] = (
                    candidate.num_preemptions
                )
            logging.getLogger("vllm").warning(
                "[BidKV] LIVENESS_FALLBACK | victim=%s (progress barrier) | "
                "threshold=%d | epoch=%d | min_relative_preemptions=%d | "
                "kv_util=%.2f | running=%d",
                victim.request_id,
                liveness_threshold,
                self._liveness_epochs,
                min(relative_preemptions.values()),
                kv_utilization or 0.0,
                len(running),
            )
            self._record_preemption(
                victim=victim,
                used_utility=False,
                policy=policy,
                kv_utilization=kv_utilization,
                now_s=now,
                default_victim=default_victim,
                ranked_candidates=ranked_candidates,
                running_size=len(running),
                decision_reason="liveness_fallback",
            )
            return victim

        top = ranked_candidates[0]
        cascade_guard_used = False
        requester = next(
            (
                candidate
                for candidate in ranked_candidates
                if candidate.request_id == requesting_request_id
            ),
            None,
        )
        if (
            requester is not None
            and requester.request_id != top.request_id
            and requester.tokens_freed * self.config.utility_cascade_gain_ratio
            >= top.tokens_freed
        ):
            # Preempting the request whose allocation failed stops the Core
            # loop after one victim. Prefer that bounded choice unless another
            # request frees materially more KV; this avoids multi-victim
            # cascades while retaining utility ranking when its gain is real.
            top = requester
            self._cascade_guard_hits += 1
            cascade_guard_used = True
        victim = req_map[top.request_id]
        self._last_utility_pick_ts = now
        logging.getLogger("vllm").info(
            "[BidKV] UTILITY_ACTIVE | victim=%s | U=%.4f | r=%d tok | "
            "completion=%.2f | preemptions=%d | kv_util=%.2f | running=%d | "
            "cascade_guard=%s",
            top.request_id,
            top.utility,
            top.tokens_freed,
            top.completion,
            top.num_preemptions,
            kv_utilization or 0.0,
            len(running),
            cascade_guard_used,
        )
        self._record_preemption(
            victim=victim,
            used_utility=True,
            policy=policy,
            kv_utilization=kv_utilization,
            now_s=now,
            default_victim=default_victim,
            ranked_candidates=ranked_candidates,
            running_size=len(running),
        )
        return victim

    # -- Internal helpers -------------------------------------------------

    @property
    def _utility_enabled(self) -> bool:
        return self.config.enable_utility_victim_selection and not self.config.utility_kill_switch

    @staticmethod
    def _pick_default_victim(running: Sequence[_Candidate], policy: str) -> _Candidate:
        if policy == "priority":
            return max(
                running,
                key=lambda request: (request.priority, request.arrival_time),
            )
        return running[-1]

    def _can_use_utility(
        self,
        *,
        kv_utilization: float | None,
        now_s: float | None,
        running_size: int,
    ) -> bool:
        if running_size < self.config.utility_min_running:
            return False
        if self.config.utility_kv_gate > 0 and (
            kv_utilization is None or kv_utilization < self.config.utility_kv_gate
        ):
            return False
        if self.config.utility_cooldown_s > 0 and self._last_utility_pick_ts > -math.inf:
            now = self._resolve_now(now_s)
            if now - self._last_utility_pick_ts < self.config.utility_cooldown_s:
                return False
        return True

    @staticmethod
    def _resolve_now(now_s: float | None) -> float:
        if now_s is not None:
            return float(now_s)
        return time.monotonic()

    def _rank_candidates(
        self, running: Sequence[_Candidate]
    ) -> tuple[list[UtilityCandidateScore], dict[str, _Candidate]]:
        req_map: dict[str, _Candidate] = {}
        candidates: list[UtilityCandidateScore] = []
        for request in running:
            request_id = str(getattr(request, "request_id", ""))
            req_map[request_id] = request
            candidates.append(self._score_request(request, request_id))

        candidates.sort(key=lambda c: (-c.utility, c.arrival_time, c.request_id))
        return candidates, req_map

    def _score_request(self, request: _Candidate, request_id: str) -> UtilityCandidateScore:
        tokens_freed = max(int(getattr(request, "num_computed_tokens", 0) or 0), 0)
        completion = self._compute_completion(request)
        num_preemptions = max(int(getattr(request, "num_preemptions", 0) or 0), 0)
        arrival_time = float(getattr(request, "arrival_time", 0.0) or 0.0)
        utility, evict_score = self._compute_utility(
            tokens_freed=tokens_freed,
            completion=completion,
            num_preemptions=num_preemptions,
        )
        return UtilityCandidateScore(
            request_id=request_id,
            utility=utility,
            evict_score=evict_score,
            tokens_freed=tokens_freed,
            completion=completion,
            num_preemptions=num_preemptions,
            arrival_time=arrival_time,
        )

    def _compute_utility(
        self,
        *,
        tokens_freed: int,
        completion: float,
        num_preemptions: int,
    ) -> tuple[float, float]:
        reward = max(float(tokens_freed), 0.0)
        preemptions = max(float(num_preemptions), 0.0)

        delta = (
            1.0
            + self.config.utility_completion_weight * completion
            + self.config.utility_preempt_weight * preemptions
        )
        utility = reward / max(delta + self.config.utility_epsilon, self.config.utility_epsilon)
        evict_score = utility
        return utility, evict_score

    def _compute_completion(self, request: _Candidate) -> float:
        output_tokens = self._output_tokens(request)
        max_tokens = getattr(request, "max_tokens", None)
        if not isinstance(max_tokens, (int, float)) or max_tokens <= 0:
            max_tokens = self.config.utility_default_max_tokens
        completion = float(output_tokens) / float(max_tokens)
        return min(max(completion, 0.0), 1.0)

    @staticmethod
    def _output_tokens(request: _Candidate) -> int:
        return int(getattr(request, "num_output_tokens", 0) or 0)

    def _record_preemption(
        self,
        *,
        victim: _Candidate,
        used_utility: bool,
        policy: str,
        kv_utilization: float | None,
        now_s: float,
        default_victim: _Candidate,
        ranked_candidates: Sequence[UtilityCandidateScore],
        running_size: int,
        decision_reason: str = "strategy",
    ) -> None:
        request_id = str(getattr(victim, "request_id", ""))
        tokens_freed = max(int(getattr(victim, "num_computed_tokens", 0) or 0), 0)

        self._total_preemptions += 1
        self._total_tokens_freed += tokens_freed
        if request_id and self._last_preempted_request_id is not None:
            self._consecutive_preemption_checks += 1
            if request_id == self._last_preempted_request_id:
                self._consecutive_preemption_events += 1

        if request_id:
            self._preemptions_per_request[request_id] += 1
            self._recent_preempted_req_ids.append(request_id)
            self._last_preempted_request_id = request_id

        if used_utility:
            self._utility_strategy_hits += 1
        else:
            self._default_strategy_hits += 1

        default_id = str(getattr(default_victim, "request_id", ""))
        self._last_decision = {
            "timestamp_s": round(now_s, 6),
            "policy": policy,
            "used_utility": used_utility,
            "kv_utilization": kv_utilization,
            "running_size": running_size,
            "selected_victim_id": request_id,
            "default_victim_id": default_id,
            "decision_reason": decision_reason,
        }
        if self.config.utility_snapshot_enabled:
            top_k = max(1, int(self.config.utility_snapshot_top_k))
            selected_id = request_id
            snapshot_candidates = [
                {
                    "rank": index + 1,
                    "request_id": candidate.request_id,
                    "utility": round(candidate.utility, 6),
                    "evict_score": round(candidate.evict_score, 6),
                    "tokens_freed": candidate.tokens_freed,
                    "completion": round(candidate.completion, 6),
                    "num_preemptions": candidate.num_preemptions,
                    "arrival_time": candidate.arrival_time,
                    "selected": candidate.request_id == selected_id,
                }
                for index, candidate in enumerate(ranked_candidates[:top_k])
            ]
            self._decision_snapshots.append(
                {**self._last_decision, "candidates": snapshot_candidates}
            )

    @staticmethod
    def _percentile(values: Sequence[int], percentile: int) -> float:
        data = sorted(int(v) for v in values if v is not None)
        if not data:
            return 0.0
        rank = math.ceil((percentile / 100.0) * len(data)) - 1
        rank = max(0, min(rank, len(data) - 1))
        return float(data[rank])


# Temporary import alias for callers migrating from the pre-0.28 HUST class
# name. It implements only the new PreemptionPolicy method.
BidkvVictimSelector = BidkvPreemptionPolicy


@dataclass(frozen=True)
class _LegacyContext:
    candidates: tuple[_Candidate, ...]
    scheduling_policy: str
    kv_cache_usage: float
    now: float
