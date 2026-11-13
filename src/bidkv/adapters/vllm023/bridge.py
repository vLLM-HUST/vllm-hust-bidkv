# SPDX-License-Identifier: Apache-2.0
"""Immutable victim-selection bridge for an explicitly patched vLLM 0.23 host.

The API follows vLLM-HUST's public preemption contract (source bad323ac8c66).
No Scheduler object or Request is supplied to the policy. This module does not
install a plugin, mutate framework classes, or claim general compatibility.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
PROFILE = "official-vllm-0.23-source-bridge/v1"


@dataclass(frozen=True, slots=True)
class PreemptionCandidate:
    request_id: str
    priority: int
    arrival_time: float
    num_prompt_tokens: int
    num_output_tokens: int
    num_computed_tokens: int
    num_preemptions: int
    max_tokens: int
    predicted_length: int | None = None


@dataclass(frozen=True, slots=True)
class PreemptionContext:
    candidates: tuple[PreemptionCandidate, ...]
    scheduling_policy: str
    requesting_request_id: str
    kv_cache_usage: float
    now: float
    builtin_victim_id: str


class Official023PolicyBridge:
    """One scheduler-thread controller, default-off with permanent fault fallback.

    An optional append-only measurement journal records every native decision,
    in both arms, with no prompt or output text. Journal I/O errors propagate:
    a run with missing required evidence must not become a successful result.
    """

    def __init__(self, vllm_config: Any):
        config = (vllm_config.additional_config or {}).get("bidkv_official023", {})
        if not isinstance(config, dict) or set(config) - {"policy", "evidence_dir"}:
            raise ValueError("Invalid official023 bridge configuration")
        configured = config.get("policy")
        self._owner_thread = threading.get_ident()
        self._policy = None
        self._fd = None
        self.stats: dict[str, Any] = {
            "profile": PROFILE,
            "policy": configured,
            "enabled": configured is not None,
            "decisions": 0,
            "calls": 0,
            "selections": 0,
            "abstentions": 0,
            "failures": 0,
            "invalid_selections": 0,
        }
        if configured is not None:
            if not isinstance(configured, str) or "." not in configured:
                raise ValueError("Policy must be a qualified class name")
            module, name = configured.rsplit(".", 1)
            implementation = getattr(importlib.import_module(module), name)
            if getattr(implementation, "vllm_preemption_policy_api_version", None) != 1:
                raise ValueError("Policy does not declare preemption API version 1")
            factory = getattr(implementation, "from_vllm_config", None)
            self._policy = factory(vllm_config) if callable(factory) else implementation()
            if not callable(getattr(self._policy, "select_victim", None)):
                raise TypeError("Policy must implement select_victim")
        if config.get("evidence_dir") is not None:
            folder = Path(config["evidence_dir"])
            if not folder.is_absolute():
                raise ValueError("Measurement journal directory must be absolute")
            folder.mkdir(parents=True, exist_ok=True)
            filename = folder / f"enginecore-{os.getpid()}.jsonl"
            self._fd = os.open(
                filename,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
        self._record("initialized", {})
        logger.info("BidKV official023 measurement bridge: %s", self.stats)

    def _record(self, event: str, details: dict[str, Any]) -> None:
        if self._fd is None:
            return
        data = {
            "event": event,
            "wall_time_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "stats": dict(self.stats),
            **details,
        }
        raw = (json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n").encode()
        if os.write(self._fd, raw) != len(raw):
            raise OSError("Incomplete preemption evidence write")

    def _disable(self, reason: str) -> None:
        self.stats["failures"] += 1
        self.stats["enabled"] = False
        self._policy = None
        logger.error("BidKV policy failed (%s); restoring native victim policy", reason)

    def select_victim(self, context: PreemptionContext) -> str:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Bridge must be used by its owning scheduler thread")
        candidate_ids = {item.request_id for item in context.candidates}
        if (
            not candidate_ids
            or len(candidate_ids) != len(context.candidates)
            or context.builtin_victim_id not in candidate_ids
            or context.requesting_request_id not in candidate_ids
        ):
            raise ValueError("Invalid native preemption snapshot")
        self.stats["decisions"] += 1
        selected = context.builtin_victim_id
        outcome = "builtin"
        if self._policy is not None:
            self.stats["calls"] += 1
            try:
                proposed = self._policy.select_victim(context)
            except Exception:
                self._disable("exception")
                outcome = "exception-fallback"
                logger.exception("Preemption policy exception")
            else:
                if proposed is None:
                    self.stats["abstentions"] += 1
                    outcome = "abstained"
                elif not isinstance(proposed, str) or proposed not in candidate_ids:
                    self.stats["invalid_selections"] += 1
                    self._disable("invalid selection")
                    outcome = "invalid-fallback"
                else:
                    selected = proposed
                    self.stats["selections"] += 1
                    outcome = "selected"
        self._record(
            "preemption-decision",
            {"context": asdict(context), "selected_id": selected, "outcome": outcome},
        )
        return selected

    def close(self) -> None:
        if self._fd is not None:
            self._record("closed", {})
            os.close(self._fd)
            self._fd = None
