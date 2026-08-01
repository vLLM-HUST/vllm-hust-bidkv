# BidKV

Framework-portable KV cache request scheduling primitive.

[中文文档](README_zh.md)

## Overview

`bidkv` is a **zero-dependency** Python package that addresses the **victim-selection problem** under KV cache pressure: when KV memory is exhausted, which request should be preempted?

The core idea is to evict the request that frees the most KV space per unit of quality loss, maximising utility:

$$U(r, \delta) = \frac{r}{\delta + \varepsilon}, \quad \varepsilon = 10^{-3}$$

where $r$ = tokens freed, $\delta$ = surrogate disruption estimate.

BidKV **does not compress tokens** — it only controls *who gets preempted*. The actual eviction is performed by the framework's native preempt + recompute path (vLLM) or RadixCache eviction (SGLang).

## Module Layout

| Module | Contents |
|--------|----------|
| `protocol/` | Core types: `CompressionBid`, `BidPool`, `BidAcceptance` |
| `scoring/` | `PositionalScoring` (attention-sink + recency heuristic) |
| `pool/` | `BidPoolManager` |
| `pressure/` | `PressureDetector` (KV pressure detection) |
| `solver/` | `GreedyBidSolver` (bid ranking + greedy selection) |
| `baselines/` | 6 baseline strategies + BidKV (see below) |
| `adapters/vllm/` | vLLM v1 adapter (scheduler hook + plugin) |
| `adapters/sglang/` | SGLang adapter (scheduler hook) |
| `experiments/` | Experiment runner, collector, analysis |

## Baseline Strategies

| Strategy name | Class | Scheduling logic |
|---------------|-------|------------------|
| `preempt-evict` | `PreemptEvictStrategy` | vLLM native FCFS admission + LIFO eviction |
| `preempt-evict-sjf` | `PreemptEvictSJFStrategy` | SJF admission + LIFO eviction |
| `static-random` | `StaticRandomStrategy` | Random victim selection |
| `largest-first` | `LargestFirstStrategy` | Capacity-greedy: evict largest KV occupant first |
| `bidkv` | `BidKVStrategy` | Quality-aware: maximise U = r / (δ + ε) |

## Configuration

```python
from bidkv import BidKVConfig

# Default: all bid logic bypassed (safe to import without activating)
config = BidKVConfig(enabled=False)

# Enable BidKV scheduling
config = BidKVConfig(enabled=True)
assert config.is_active

# Kill switch: immediately bypasses all logic even when enabled=True
config = BidKVConfig(enabled=True, kill_switch=True)
assert not config.is_active
```

## Adding a Custom Strategy

```python
from bidkv import (
    BaselineRegistry,
    BidKVStrategy,
    PreemptEvictStrategy, LargestFirstStrategy,
    StaticRandomStrategy, PreemptEvictSJFStrategy,
)

# Register all built-in strategies at once
registry = BaselineRegistry()
registry.create_default_registry()

# Or register selectively
registry2 = BaselineRegistry()
registry2.register(BidKVStrategy())
registry2.register(PreemptEvictStrategy())

strategy = registry2.get("bidkv")
print(strategy.name)              # "bidkv"
print(registry2.list_strategies())  # ["bidkv", "preempt-evict"]
```

## Running Experiments

```bash
# vLLM: 5 strategies × mixed workload × 3 rates × 3 runs
HF_HUB_OFFLINE=1 python -m bidkv.experiments.vllm.runner \
    --strategies "preempt-evict,preempt-evict-sjf,static-random,largest-first,bidkv" \
    --workloads mixed \
    --mixed-rates 2.0,3.8,5.7 \
    --runs 3 \
    --output-dir results/vllm_experiment \
    --gpu-memory-utilization 0.5 \
    --num-gpu-blocks-override 600 \
    --max-num-seqs 32

# SGLang: 3 strategies
HF_HUB_OFFLINE=1 python -m bidkv.experiments.sglang.runner \
    --strategies "sglang_default,slack_aware,bidkv" \
    --workloads mixed \
    --runs 3 \
    --output-dir results/sglang_experiment
```

## Framework Integration (vLLM)

For `vllm-hust`, install BidKV into the same Python environment. The runtime
discovers it through the `vllm.victim_selector` entry point. Installing the
package is inert by default; explicitly select and enable BidKV when serving:

```bash
python -m pip install -e . --no-deps

vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enforce-eager \
    --port 8000 \
    --additional-config '{
      "victim_selector_plugin": "bidkv",
      "enable_utility_victim_selection": true,
      "utility_strategy": "bidkv",
      "utility_kv_gate": 0.95
    }'
```

Verify discovery before launching:

```bash
python - <<'PY'
from importlib.metadata import entry_points

for ep in entry_points(group="vllm.victim_selector"):
    print(ep.name, "->", ep.value)
PY
```

Environment variables with the `BIDKV_UTILITY_` prefix provide equivalent
runtime configuration. `BIDKV_STRATEGY` belongs to the legacy experiment
adapter, which monkey-patches the scheduler. Do not combine it with the native
victim-selector integration.

### Legacy experiment adapter

Use this path only to reproduce the historical multi-strategy experiments:

```bash
BIDKV_STRATEGY=bidkv python -m bidkv.experiments.vllm.serve \
    --model meta-llama/Llama-3.1-8B-Instruct --enforce-eager --port 8000
```

## Zero Dependencies

`bidkv` depends only on the Python standard library — no torch, numpy, vllm, or sglang.

## Install

```bash
pip install -e .

# development mode
pip install -e ".[dev]"
```

## Testing

```bash
python -m pytest tests/ -v
```

## License

Apache-2.0
