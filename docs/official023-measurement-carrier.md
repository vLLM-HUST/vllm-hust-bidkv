# Experimental official-vLLM 0.23 measurement carrier

This carrier prepares an isolated copy of the exact official vLLM v0.23.0 package and its matching
prebuilt libraries. It adds an explicit native victim-selection call to the scheduler. It targets
the two-card Ascend measurement environment; it is separate from the maintained vLLM-HUST Extension
Manager integration and does not replace that host contract.

The immutable candidate/context schema, abstention and permanent fault fallback follow the
[public vLLM-HUST preemption API](https://github.com/vLLM-HUST/vllm-hust/blob/bad323ac8c667c8f107ac02f143a5b0441e6155c/vllm/v1/core/sched/preemption.py).
The scheduler retains request objects, block freeing, reinsertion, encoder accounting and
token-budget rollback. A policy sees frozen values and returns an eligible request ID. Removing any
earlier request also updates the native scan position, including an earlier request skipped in this
scheduling step.

## Prepare and select the carrier

```bash
python scripts/prepare_official023_host.py \
  --source /vllm-workspace/vllm \
  --out /workspace/new-measurement/runtime/official023
```

Preparation requires official commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` and its exact
scheduler SHA-256. It rejects tracked source changes and an existing destination. The original SDK
stays unchanged. The receipt records all copied source/library hashes, the prepared scheduler hash
and the adapter script hash; preparation makes no model requests.

Install this BidKV checkout in a separate environment using the matching official SDK. Place the
prepared package first on `PYTHONPATH` and verify its actual import paths before starting the
server. Both arms use the same tree. The baseline additional configuration is:

```json
{"bidkv_official023":{"policy":null,"evidence_dir":"/absolute/new/baseline/policy"}}
```

The candidate selects the existing BidKV implementation explicitly:

```json
{"bidkv_official023":{"policy":"bidkv.adapters.vllm_hust.selector.BidkvPreemptionPolicy","evidence_dir":"/absolute/new/candidate/policy"},"enable_utility_victim_selection":true,"utility_kill_switch":false}
```

Retain other SDK-specific settings identically in both arms. No legacy `vllm.general_plugins` wheel
or scheduler monkeypatch is installed by this carrier. A candidate installation or initialized
journal alone does not show runtime effectiveness: require actual policy calls, inspect selections
and abstentions, and reject policy failures/invalid selections. Zero-call cells remain valid
descriptions of that workload but cannot establish a selector benefit.

## Measurement evidence and gates

An optional, exclusive per-process journal records every native preemption decision in both arms,
its immutable snapshot, selected ID, exact cumulative counters and wall/monotonic timestamps. It
contains no prompt/output text. Journal failures propagate so incomplete required evidence cannot
silently produce a successful measurement. Both arms include this instrumentation in their measured
runtime; results must identify this experimental carrier.

The official Ascend 0.23 SDK pins Transformers 5.5.4. Keep its dependency contract; do not override
that pin merely to match a CPU scope-builder version. For the frozen five-dataset trace protocol,
preserve the original four tokenizer/config files and use its exact reference implementation to
freeze expected input IDs. Every actual service input must then pass `/tokenize` against those IDs
before generation. A mismatch stops that profile; no input recalculation, replacement or silent
version relabeling is permitted.

Run the focused host-only regressions with:

```bash
python -m pytest tests/test_vllm023_bridge.py -q
```

The compressed scheduler fixture is the byte-preserved Apache-2.0 official source at the pinned
commit, SHA-256 `9ae1491f57a6294e32416b93ef2ea68107fa0d65c3a4d7f745f1261012f735e9`. Tests cover
immutable state, selected/abstained/invalid/exception behavior, permanent fallback, thread
ownership, append-only counters, source drift and native budget/scan-position restoration. They do
not qualify model execution. Before dataset measurements, separately require the selected SDK's
dependency admission, real two-card graph-mode and continuous C8 qualification, complete input
admission and preserved cleanup/raw records. No new device-performance or five-dataset completion
claim is made by this source addition.
