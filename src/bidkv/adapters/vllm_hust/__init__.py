# SPDX-License-Identifier: Apache-2.0
"""Legacy vLLM-HUST victim-selector adapter for pinned contract replay.

The adapter remains importable for historical validation, but the main BidKV
distribution deliberately does not register the non-upstream
``vllm.victim_selector`` entry-point group.
"""

from bidkv.adapters.vllm_hust.selector import (
    BidkvSelectorConfig,
    BidkvVictimSelector,
    UtilityCandidateScore,
)

__all__ = [
    "BidkvSelectorConfig",
    "BidkvVictimSelector",
    "UtilityCandidateScore",
]
