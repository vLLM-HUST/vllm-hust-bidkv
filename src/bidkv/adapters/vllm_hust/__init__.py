# SPDX-License-Identifier: Apache-2.0
"""BidKV policy implementation for the vLLM-HUST typed scheduler contract.

The main distribution registers a static extension manifest rather than the
private legacy ``vllm.victim_selector`` entry-point group.
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
