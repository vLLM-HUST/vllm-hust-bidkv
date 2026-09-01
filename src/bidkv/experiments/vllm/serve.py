"""Custom vLLM API server with BidKV strategy injection.

Usage
-----
# Start with preempt-evict (no BidKV, vanilla vLLM):
BIDKV_STRATEGY=preempt-evict python -m bidkv.experiments.vllm.serve \
    --model meta-llama/Llama-3.1-8B-Instruct --enforce-eager --port 8000

# Start with BidKV largest-first strategy:
BIDKV_STRATEGY=largest-first python -m bidkv.experiments.vllm.serve \
    --model meta-llama/Llama-3.1-8B-Instruct --enforce-eager --port 8000

Hook mechanism
--------------
When the separate ``bidkv-vllm-legacy`` package is installed, BidKV hooks are
injected via its ``vllm.general_plugins`` entry point (implemented by
``bidkv.adapters.vllm.plugin``). The main ``bidkv`` wheel does not register
this automatic hook. The legacy plugin reads ``BIDKV_STRATEGY`` and patches
``Scheduler.__init__`` inside the EngineCore subprocess.

This module only needs to ensure ``BIDKV_STRATEGY`` is set before starting
the vLLM server.
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)

_STRATEGY_NAME: str = os.environ.get("BIDKV_STRATEGY", "preempt-evict")


def _legacy_entry_point_installed() -> bool:
    discovered = entry_points()
    if hasattr(discovered, "select"):
        candidates = discovered.select(group="vllm.general_plugins", name="bidkv")
    else:  # pragma: no cover - Python 3.10 compatibility
        candidates = [
            item
            for item in discovered.get("vllm.general_plugins", ())
            if item.name == "bidkv"
        ]
    return any(
        item.value == "bidkv.adapters.vllm.plugin:register" for item in candidates
    )


def main() -> None:
    """Entry point — starts vLLM server with BidKV strategy via plugin."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    strategy = _STRATEGY_NAME
    logger.info("BidKV experiment server starting (strategy=%s)", strategy)

    if not _legacy_entry_point_installed():
        raise RuntimeError(
            "legacy experiment replay requires the separate "
            "bidkv-vllm-legacy package; the main bidkv wheel intentionally "
            "does not register vllm.general_plugins"
        )

    if strategy == "preempt-evict":
        logger.info("preempt-evict: using vanilla vLLM (no BidKV)")
    else:
        logger.info(
            "BidKV strategy=%s: hooks will be injected via vllm.general_plugins "
            "entry point in the EngineCore subprocess",
            strategy,
        )

    # Ensure BIDKV_STRATEGY is in env for the subprocess to read
    os.environ["BIDKV_STRATEGY"] = strategy

    # Start the standard vLLM API server
    import runpy

    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
