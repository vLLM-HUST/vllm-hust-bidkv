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

The Sage Mate target at vLLM-HUST `762f85b3` (`0.28.1rc1.dev319`) exposes the
immutable `vllm.preemption-policy.v1` contract. BidKV supplies
`BidkvPreemptionPolicy`; vLLM retains ownership of request state, preemption,
KV cleanup, reinsertion, and scheduling-budget rollback. The production path
does not monkey-patch `Scheduler`.

```bash
pip install vllm-hust-ext bidkv
vllm-hust-ext extension enable org.vllm-hust.bidkv
vllm-hust-ext run -- vllm serve /data/shared_models/Qwen/Qwen3.8-27B \
    --tensor-parallel-size 4 --port 8000
```

Verify the policy implementation API version:

```bash
python - <<'PY'
from bidkv.adapters.vllm_hust.selector import BidkvPreemptionPolicy
print(BidkvPreemptionPolicy.vllm_preemption_policy_api_version)
PY
```

`BIDKV_STRATEGY` belongs to a separate historical experiment adapter that
monkey-patches the scheduler. The main wheel does not register
`vllm.general_plugins`; installing `bidkv` therefore cannot auto-import the
legacy hook.

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

> **Host boundary:** the pinned vLLM-HUST target supports
> `vllm.preemption-policy.v1`; official vLLM does not yet expose this contract.
> The upstream direction is RFC
> [#51608](https://github.com/vllm-project/vllm/issues/51608) and draft PR
> [#51601](https://github.com/vllm-project/vllm/pull/51601), whose target
> `vllm.scheduler_plugins`/PreemptionScore contract is not frozen. The HUST
> interface stays minimal, generic, and free of BidKV names; it is not evidence
> of compatibility with official vLLM.

The exact semantic mapping, draft code/design mismatch, and migration gates
are tracked in [the upstream scheduler contract gap](docs/upstream-scheduler-contract-gap.md).

```bash
pip install vllm-hust-ext bidkv
vllm-hust-ext extension inspect org.vllm-hust.bidkv
vllm-hust-ext extension validate org.vllm-hust.bidkv
vllm-hust-ext extension status org.vllm-hust.bidkv
```

Manager validates the exact host protocol and renders
`--preemption-policy bidkv.adapters.vllm_hust.selector.BidkvPreemptionPolicy`.
The controller records calls, selections, abstentions, invalid selections, and
failures; an invalid result or runtime exception is logged once and permanently
restores the built-in victim policy for that engine process. Constructor or
configuration errors fail startup.

BidKV also has a forward-progress guard. If every runnable request has already
been preempted at least `BIDKV_UTILITY_LIVENESS_PREEMPTIONS` times (default 2),
it records `LIVENESS_FALLBACK` and uses vLLM's stable default victim order for
that decision. Set the value to `0` only to disable this guard for controlled
experiments.

Lifecycle labels are deliberately separate:

| Label | Meaning |
| --- | --- |
| installed | The pinned BidKV wheel and manifest are present. |
| configured | Manager validated the host version/protocol and rendered the launch option. |
| enabled | Saved operator intent requests BidKV on the next approved launch. |
| runtime effective | EngineCore logs the exact class and non-zero policy-call counters from a controlled online run. |

CPU contract and scheduler-path tests pass on the pinned source. The repaired
revision then completed a real Qwen3.8-27B Ascend TP4 graph-mode pressure run:
four concurrent requests with 12,517 input tokens and 2,048 requested output
tokens all returned HTTP 200. The selected policy was called 187 times with
zero failures; 6 decisions used BidKV utility selection and 181 deliberately
used the liveness fallback. Cancellation/recovery, output checks, emergency
disable, and next-process baseline rollback passed. This exact candidate lane
is therefore **compatible**. The verdict does not transfer to another model,
topology, artifact or host commit. See the
[qualification record](docs/evidence/sage-mate-20260904-tp4-graph.md).

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
pip install ./legacy/vllm-general-plugin
BIDKV_STRATEGY=bidkv python -m bidkv.experiments.vllm.serve \
    --model meta-llama/Llama-3.1-8B-Instruct --enforce-eager --port 8000
```

Do not install `bidkv-vllm-legacy` in a typed Extension Manager serving
environment.

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
