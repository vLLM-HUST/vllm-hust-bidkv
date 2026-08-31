# BidKV

Framework-portable KV cache request scheduling primitive.

[中文文档](README_zh.md)

## Overview

`bidkv` is a **zero-dependency** Python package that addresses the **victim-selection problem** under KV cache pressure: when KV memory is exhausted, which request should be preempted?

The core idea is to evict the request that frees the most KV space per unit of quality loss, maximising utility:

$$U(r, \delta) = \frac{r}{\delta + \varepsilon}, \quad \varepsilon = 10^{-3}$$

where $r$ = tokens freed, $\delta$ = surrogate disruption estimate.

BidKV **does not compress tokens** — it only controls *who gets preempted*. The actual eviction is performed by the framework's native preempt + recompute path (vLLM) or RadixCache eviction (SGLang).

## Ecosystem classification

BidKV is a scheduler-local victim-selection policy component. It is not a KV
store, transport, connector, compression mechanism, or external state system.
The repository also contains framework adapters and experiment tooling, but
those delivery surfaces do not change the policy's runtime role.

See [`.vllm-hust/repository-profile.json`](./.vllm-hust/repository-profile.json)
for the machine-readable boundary and migration contract.

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

The historical HUST fork exposed `vllm.victim_selector`. BidKV retains that
adapter as an importable compatibility module for pinned replay, but the main
distribution no longer registers the non-upstream entry-point namespace. The
following launch shape applies only to a pinned legacy fork that already owns
that contract; it is not valid for a fresh official vLLM fork:

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

Verify the legacy module without claiming runtime discovery:

```bash
python - <<'PY'
from bidkv.adapters.vllm_hust.selector import BidkvVictimSelector
print(BidkvVictimSelector.vllm_victim_selector_api_version)
PY
```

Environment variables with the `BIDKV_UTILITY_` prefix belong to the same
pinned legacy contract. `BIDKV_STRATEGY` is a separate historical experiment
adapter that monkey-patches the scheduler. Neither path is a supported fresh
official-vLLM integration.

### vLLM-HUST Extension Manager path

BidKV also ships
`bidkv/manifests/vllm-hust-extension-v0.2.json` for the experimental
vLLM-HUST Extension Manifest 0.2 path. This manifest describes BidKV as an in-process scheduler
policy; it does not describe a KV store, KV connector, or external control
plane. The wheel registers the static manifest through
`vllm_hust.extension_bundles`; discovery neither imports BidKV nor enables
scheduling behavior.

Manifest 0.2 is not a compatibility promise and must not be published as a
stable Bundle v1 contract before all three host-provider acceptance gates pass.

> **Host contract warning:** `vllm.victim_selector` is a legacy HUST
> experimental hook and is absent from the fresh vLLM-HUST 0.23 fork. The
> upstream direction is RFC
> [#51608](https://github.com/vllm-project/vllm/issues/51608) and draft PR
> [#51601](https://github.com/vllm-project/vllm/pull/51601), whose target
> `vllm.scheduler_plugins`/PreemptionScore contract is not frozen. Do not add a
> competing private hook to the new core or claim current 0.23 compatibility.

The exact semantic mapping, draft code/design mismatch, and migration gates
are tracked in [the upstream scheduler contract gap](docs/upstream-scheduler-contract-gap.md).

```bash
pip install vllm-hust-ext bidkv
vllm-hust-ext extension inspect org.vllm-hust.bidkv
vllm-hust-ext extension validate org.vllm-hust.bidkv
vllm-hust-ext extension status org.vllm-hust.bidkv
```

On a fresh official vLLM installation, status must remain `incompatible` or
`degraded` and `run` refuses activation. A pinned legacy operator may provide
explicit host and `vllm.victim_selector` protocol evidence, but this is only a
replay path. The alpha gate requires migration to the stabilized upstream
Preemption contract, real scheduler invocation, conflict and failure tests,
and next-process rollback. Until then, there is no supported enable command.

For a legacy replay that was explicitly enabled, disable the saved intent and
start a fresh vLLM process to roll back:

```bash
vllm-hust-ext extension disable org.vllm-hust.bidkv
```

After the replacement vLLM process is running without BidKV, remove Manager
state before uninstalling the Python distribution. This prevents a later
reinstall from restoring stale enabled intent:

```bash
vllm-hust-ext extension forget org.vllm-hust.bidkv
pip uninstall bidkv
```

`forget` does not stop an existing vLLM process; process restart remains owned
by the vLLM operator.

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
