# Sage Mate current-main Qwen3.8-27B TP4 graph qualification — 2026-09-05

## Superseded interpretation and immutable inputs

This record preserves the first single-run measurement, but its original
compatibility conclusion was invalid. The run proved functional execution,
cancellation/recovery and rollback. Its effectiveness classification is only
**not-beneficial-in-tested-cell** for this one homogeneous-long pressure cell
(`n=1`); it does not make BidKV incompatible and cannot be generalized to the
whole Mod. The later matched three-repeat qualification is the controlling
record:
[`sage-mate-20260905-current-main-tp4-graph-r2.md`](sage-mate-20260905-current-main-tp4-graph-r2.md).

- vLLM-HUST `a4d6aa022fb1885a25a802a6e29372c81eac6c9f`
  (`0.28.1rc1.dev391+ga4d6aa022`)
- vLLM-Ascend-HUST `2c8c722107a54127999a64c4eb0ec86139df8c26`
  (`0.25.1rc1+hust.20260903.4`)
- BidKV `5fb109be683f486dfdf45d50f88c6138e003637e`
- Candidate image ID
  `sha256:63e70bd9e5de1067b374c9587b0ae71c93b6a416f7f10c82499873c24b231a8d`
- Baseline image ID
  `sha256:80f05c0d0c49c139f94922ae6057e3edb21251b8e8a332c1df35fb3d555d60d8`
- Qwen3.8-27B, Ascend NPU 0–3, TP4, `FULL_DECODE_ONLY` graph mode,
  1 GiB explicit KV-cache pressure budget

The engine log proved all four HCCL ranks, `enforce_eager=False`, graph capture
sizes 1/2/4/8, and
`[BidKV] INIT | enabled=True | strategy=bidkv`. The managed launcher required
`VLLM_OPTIMIZATION_ENV_PREFIX=BIDKV_UTILITY_`; without that allowlist the class
was installed and configured but initialized with `enabled=False`.

## Matched A/B result

Each arm ran four concurrent requests containing 49,700 prompt tokens in total
and forced 8,192 output tokens in total. All requests completed.

| Metric | Built-in | BidKV | Relative change |
| --- | ---: | ---: | ---: |
| Actual preemptions | 161 | 164 | +1.86% |
| Output throughput | 27.879 tok/s | 27.223 tok/s | -2.35% |
| P95 TTFT | 214.327 s | 218.934 s | +2.15% |
| P95 TPOT | 126.415 ms | 129.934 ms | +2.78% |
| P95 latency | 289.406 s | 296.229 s | +2.36% |
| Jain output-rate fairness | 0.995808 | 0.995615 | -0.02% |

BidKV made 6 utility selections, followed by 158 liveness fallbacks. This
proves the implementation was invoked, but also shows that the fallback path
reproduced the built-in preemption churn for almost the entire pressure phase.
This selection mix is a configuration/workload diagnostic. It does not change
the functional compatibility verdict and must not be promoted to a public
whole-Mod effectiveness label.

Four short deterministic responses matched the built-in response hashes. The
four forced 2,048-token pressure responses did not. Later
baseline-versus-baseline repeatability controls diverged in all four lanes,
with first differences at token indices 28, 41, 17 and 26. The long-hash
difference is therefore retained as a TP4 graph determinism/semantic-correctness
investigation item, not classified as a BidKV compatibility failure.

Four concurrent streams were cancelled after 12 seconds. The server drained
from four running requests to zero running/zero waiting within one second and a
subsequent request returned exactly `BIDKV_CANCEL_RECOVERY_OK` with HTTP 200.
The next managed process removed the BidKV class and environment, restoring the
built-in policy.

The redacted raw evidence bundle is retained at
`/data/codex-build-artifacts/sage-mate-bidkv-main-requal-20260904T195737Z` on
the qualification host.
