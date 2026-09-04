# Sage Mate current-main Qwen3.8-27B TP4 graph requalification — 2026-09-05

## Verdict and immutable inputs

BidKV passes the functional compatibility gate on this exact lane and is
classified **runtime effective / performance neutral**:

- vLLM-HUST `a4d6aa022fb1885a25a802a6e29372c81eac6c9f`
  (`0.28.1rc1.dev391+ga4d6aa022`)
- vLLM-Ascend-HUST `2c8c722107a54127999a64c4eb0ec86139df8c26`
  (`0.25.1rc1+hust.20260903.4`)
- tested BidKV runtime `1462a17b3b5e59865957d7a2226fb2f0578eecb1`
- candidate image `sha256:c49d7c3949e11d17f78721bb9ae7ead7d3ea031c1012ef4210061c3731c8c9b9`
- baseline image `sha256:80f05c0d0c49c139f94922ae6057e3edb21251b8e8a332c1df35fb3d555d60d8`
- Qwen3.8-27B, Ascend NPU 0–3, TP4, `FULL_DECODE_ONLY` graph mode,
  explicit 1 GiB KV-cache pressure budget

The engine captured graph sizes 1/2/4/8, initialized all four HCCL ranks, and
logged `[BidKV] INIT | enabled=True` on its first managed launch. The production
profile propagated `BIDKV_UTILITY_*` through
`VLLM_ENGINE_EXTRA_ENV_PREFIXES`; no temporary export or second restart was
required. Core recorded 483 calls and 483 valid selections with zero
abstentions, failures, or invalid selections across the three pressure runs.

## Repeated matched A/B

Each run used four concurrent 12,425-token prompts and forced 2,048 output
tokens per request. Every request completed without starvation.

| Metric | Built-in mean (n=3) | BidKV mean (n=3) | Relative change |
| --- | ---: | ---: | ---: |
| Actual preemptions | 161.0 | 161.0 | 0.00% |
| Output throughput | 27.613 tok/s | 27.954 tok/s | +1.23% |
| Goodput | 0.013483 req/s | 0.013649 req/s | +1.23% |
| P95 TTFT | 217.409 s | 213.113 s | -1.98% |
| P95 TPOT | 126.088 ms | 125.818 ms | -0.21% |
| P95 latency | 292.388 s | 288.348 s | -1.38% |
| Jain output-rate fairness | 0.995896 | 0.995692 | -0.02% |

All 95% Student-t intervals overlap. The run therefore proves no measured
regression and real policy execution, but not a statistically separated
speedup. Local-compute prompt-token deltas exactly matched client prompt tokens
in all runs, so no additional recomputation was observed.

## Correctness, liveness, cancellation, and rollback

Four short deterministic outputs matched the built-in baseline SHA-256 values
exactly. Long forced output hashes are not used as a correctness equality gate:
two repeated built-in TP4 graph runs diverged in all four lanes, with first token
differences at indices 28, 41, 17, and 26. This is evidence of batch/numeric
nondeterminism in the baseline path, not a BidKV-specific divergence.

An additional heterogeneous 49,700-prompt-token concurrency run completed all
four requests with no starvation. Four concurrent streams cancelled after 12
seconds, drained to zero running/zero waiting within one second, and the next
request returned exactly `BIDKV_CANCEL_RECOVERY_OK` with HTTP 200.

The controller's fail-closed path permanently restores the built-in policy on
an exception or invalid request ID. Operator rollback removes the
`--preemption-policy` option and `BIDKV_UTILITY_*` environment, then starts a
fresh managed engine process. The exact production environment and image are
restored after qualification.

Raw redacted evidence is retained at
`/data/codex-build-artifacts/sage-mate-bidkv-main-requal-20260904T195737Z` on
the qualification host.
