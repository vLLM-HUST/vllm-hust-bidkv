# BidKV

Framework-portable KV cache request scheduling primitive.

[中文文档](README_zh.md)

## Academic ownership and maintenance

- Academic owners: Chen Yanbo (`@cybber695`) and Wang Mingqi (`@MingqiWang-coder`)
- Administrative custodian: Shuhao Zhang (`@ShuhaoZhangTony`)
- Scope: maintenance of the accepted SC 2026 BidKV artifact; BidKV2 and BidKV-MultiNPU are separate follow-on projects

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

## Qwen3.5 TP2 development adaptation

This branch targets the inference-engine project's existing Qwen3.5-35B-A3B
BF16 / Ascend TP2 / PP1 / DP1 / FULL_DECODE_ONLY configuration. It remains a
**development candidate**, without a new functional or performance qualification.
The host, graph, cache and dataset settings remain owned by that project.

This adaptation is based on BidKV commit
`6f1fc75d694c6901051bdeae784d18b836c6fa70` and retains its Host 0.28 manifest
range. It is a separate branch from the Host 0.31 / PP2 / TP1 declaration now on
upstream main. The project's frozen HUST Core and Ascend source commits are
`ce5e49034a40d34fe88d78e4707c629449c3a5a2` and
`a9dd32033c5669ba28976c9d208a69d762796ba9`, respectively; use the project's
unified runtime instructions and frozen test profiles.

Prior local MMLU-Pro measurements covered three paired Native/BidKV starts,
200 performance questions and an overlapping 70-question quality subset.
Both arms scored 56/70 in every start, but policy calls and preemptions were
zero, and no stable cost reduction was established. Those measurements used
the earlier local runtime build, not the newly prepared unified SDK. They do
not qualify pressure behavior or gains on the frozen unified runtime.

The 0.2.2.dev3 manifest declares a plugin-owned `BIDKV_RUNTIME_CONTRACT`
environment value, rather than injecting unknown keys into Ascend additional_config. The policy factory
checks the effective model architecture, dtype, topology and graph mode and
rejects mismatches before accepting the policy. Manager still checks the host
version/API/protocol and exclusive scheduler slot; its implementation is not
modified. The former Qwen3.8 TP4 qualification is retained in
[`docs/evidence/qwen38-tp4-manifest-qualification.json`](docs/evidence/qwen38-tp4-manifest-qualification.json)
and its original evidence, instead of being claimed for this new target.
Manager admission is only permission to test the declared target: actual model
execution, policy calls, task quality and matched cost must be checked separately.

## Framework Integration (vLLM)

The TP4 commands and qualification below describe the historical target, not
the default Qwen3.5 TP2 development activation on this branch. Use the current
project profile and [`scripts/run_project_admission.py`](scripts/run_project_admission.py)
for finite admission; it requires an explicit frozen input profile and the
project's graph observer. Actual model launches remain under the project's
device admission and Extension Manager controls.

The Sage Mate target at vLLM-HUST `762f85b3` (`0.28.1rc1.dev319`) exposes the
immutable `vllm.preemption-policy.v1` contract. BidKV supplies
`BidkvPreemptionPolicy`; vLLM retains ownership of request state, preemption,
KV cleanup, reinsertion, and scheduling-budget rollback. The production path
does not monkey-patch `Scheduler`.

```bash
pip install vllm-hust-ext vllm-hust-bidkv
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
`bidkv/manifests/vllm-hust-extension-v0.3.json` for the experimental
vLLM-HUST Extension Manifest 0.3 path. This manifest describes BidKV as an in-process scheduler
policy; it does not describe a KV store, KV connector, or external control
plane. The wheel registers the static manifest through
`vllm_hust.extension_bundles`; discovery neither imports BidKV nor enables
scheduling behavior.

Manifest 0.3 is not a compatibility promise and must not be published as a
stable Bundle v1 contract before all three host-provider acceptance gates pass.

> **Host boundary:** the generic contract is maintained in the `vLLM-HUST`
> organization. Its abstention and built-in-victim semantics were hardened and
> merged through [vLLM-HUST/vllm-hust#11](https://github.com/vLLM-HUST/vllm-hust/pull/11).
> This qualification campaign intentionally did not submit to
> `vllm-project/vllm`; references to that project's scheduler work are context,
> not a publication or compatibility claim.

The exact semantic mapping, draft code/design mismatch, and migration gates
are tracked in [the upstream scheduler contract gap](docs/upstream-scheduler-contract-gap.md).

```bash
pip install vllm-hust-ext vllm-hust-bidkv
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

BidKV also has a bounded forward-progress guard. After every runnable request
has accumulated another `BIDKV_UTILITY_LIVENESS_PREEMPTIONS` preemptions
(default 2) within the current epoch, one `LIVENESS_FALLBACK` progress-barrier
decision preempts the requesting request (or the stable default if unavailable),
then advances per-request offsets and immediately re-enables utility selection.
This avoids the old permanent-default state. Utility selection also prefers the
request whose allocation failed when it frees within
`BIDKV_UTILITY_CASCADE_GAIN_RATIO` (default 1.25) of the top candidate, bounding
multi-victim cascades. Set the liveness threshold to `0` only for controlled
experiments.

Lifecycle labels are deliberately separate:

| Label | Meaning |
| --- | --- |
| installed | The pinned BidKV wheel and manifest are present. |
| configured | Manager validated the host version/protocol and rendered the launch option. |
| enabled | Saved operator intent requests BidKV on the next approved launch. |
| runtime effective | EngineCore logs the exact class and non-zero policy-call counters from a controlled online run. |

Qwen3.8-27B on the current vLLM-HUST/Ascend-HUST qualification pair has passed
the functional compatibility gate for Ascend TP4 `FULL_DECODE_ONLY` graph
mode. That statement is independent of whether BidKV is installed, configured,
enabled, or runtime-effective on any particular live instance. A five-cell
functional matrix and two alternating three-repeat cells completed every
request with no policy failure, invalid selection, graph failure, or traceback.
In the adversarial ascending-mixed cell, the two repeats that invoked the
policy made 63 calls each and matched the built-in arm's 63 preemptions; the old
-57.79% throughput collapse was eliminated. That cell is **inconclusive**
because one repeat did not invoke the policy. The interactive concurrency-eight
cell is **not-beneficial-in-tested-cell**: throughput delta mean -25.31% (95% CI
-26.66% to -23.96%) and P95 latency delta mean +34.57% (95% CI +31.96% to
+37.17%). These are scoped effectiveness results, not a whole-Mod verdict.
Short deterministic responses matched exactly;
long pressure output hashes are not an equality gate because repeated built-in
runs themselves diverged from the first generated tokens under TP4 graph batch
scheduling. Cancellation drained in one second and exact recovery passed. See
the [bounded-preemption matrix](docs/evidence/sage-mate-20260905-bounded-preemption-matrix.md),
the [earlier current-main requalification](docs/evidence/sage-mate-20260905-current-main-tp4-graph-r2.md),
the [superseded failed attempt](docs/evidence/sage-mate-20260905-current-main-tp4-graph.md),
and the [historical qualification](docs/evidence/sage-mate-20260904-tp4-graph.md).

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
pip uninstall vllm-hust-bidkv
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

## Canonical MOD metadata

Repository identity, directly responsible maintainers, advisor status, default-off
activation, rollback, scope, and evidence qualification are recorded in
[`MOD_METADATA.json`](MOD_METADATA.json). `advisor_status: unknown` is not the
same as confirmed `none`. Performance statements remain limited to the workloads
and evidence labels recorded there; they are not general online claims.
