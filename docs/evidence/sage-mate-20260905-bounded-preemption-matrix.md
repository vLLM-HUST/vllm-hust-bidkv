# Sage Mate bounded-preemption TP4 graph matrix — 2026-09-05

## Verdict and immutable inputs

BidKV runtime `199e0bdc6fc38fc9b14b626515efdcbf81de0b62` passes the functional compatibility gate for
Qwen3.8-27B on Ascend TP4 `FULL_DECODE_ONLY` graph. Effectiveness remains a
per-cell result: the repeated interactive cell is
`not-beneficial-in-tested-cell`, while the adversarial ascending-mixed cell is
`inconclusive`. Neither result changes the functional verdict or describes the
live state of another installation.

- vLLM-HUST base `a4d6aa022fb1885a25a802a6e29372c81eac6c9f`
  (`0.28.1rc1.dev391+ga4d6aa022`)
- vLLM-Ascend-HUST base `2c8c722107a54127999a64c4eb0ec86139df8c26`
  (`0.25.1rc1+hust.20260903.4`)
- requested Sage Mate baseline retained separately: Core `762f85b3`, Ascend
  `4e57439e`; its earlier qualification is in
  [`sage-mate-20260904-tp4-graph.md`](sage-mate-20260904-tp4-graph.md)
- BidKV source `199e0bdc6fc38fc9b14b626515efdcbf81de0b62`; clean wheel SHA-256
  `4abccc6a55e97d8c78cafd86366697f2adbcc106a4ef33ecc68157562a87eeb1`
- candidate image
  `sha256:a4e042e304507b3fa03f51c319098edb8173d32ebd5d5a5704ff842ef0a1ed77`
- baseline image
  `sha256:80f05c0d0c49c139f94922ae6057e3edb21251b8e8a332c1df35fb3d555d60d8`
- Qwen3.8-27B, NPU0-3, TP4, 1-GiB KV cache, graph capture/replay; NPU4-7
  excluded

The generic safe-abstention and built-in-victim contract was merged into the
`vLLM-HUST` organization through
[vLLM-HUST/vllm-hust#11](https://github.com/vLLM-HUST/vllm-hust/pull/11).
No PR was submitted to `vllm-project/vllm`.

## Stage-one functional matrix

Each row is one matched baseline/candidate pair. All rows completed without
starvation, captured and replayed graph mode on ranks 0-3, had no unexpected
traceback, and kept policy failure/invalid counters at zero.

| Cell | Candidate calls / selections / abstentions | Throughput delta | P95 latency delta | Functional result |
| --- | ---: | ---: | ---: | --- |
| homogeneous-long, c=4 | 211 / 1 / 210 | -0.14% | +0.11% | pass |
| interleaved mixed, c=4 | 0 / 0 / 0 | -0.17% | +0.16% | pass; policy not invoked |
| ascending mixed regression, c=4 | 0 / 0 / 0 | +129.45% | -58.50% | pass; policy not invoked |
| interactive batch, c=8 | 24 / 2 / 22 | -26.94% | +37.69% | pass |
| cancel/recovery burst, c=8 | 218 / 4 / 214 | -17.98% | +22.50% | pass |

The cancellation row cancelled four of eight streams, completed the other four,
reported no failed/starved request, and returned
`BIDKV_MATRIX_CANCEL_RECOVERY_OK` from the immediate recovery request. Its
single-run performance result remains `inconclusive`; it is retained as a
failure-recovery gate rather than promoted to a statistical claim.

## Three-repeat effectiveness matrix

The repeated cells alternate arm order. The analyzer uses paired Student-t 95%
intervals and refuses an effectiveness claim unless there are at least three
pairs, all functional gates pass, and the policy is invoked in every repeat.

| Cell | Calls by repeat | Throughput deltas | Throughput mean / 95% CI | P95 latency mean / 95% CI | Qualification |
| --- | --- | --- | --- | --- | --- |
| ascending mixed regression, c=4 | 63, 0, 63 | +0.33%, +138.38%, +0.12% | +46.28% / [-151.88%, +244.44%] | -20.12% / [-105.92%, +65.68%] | `inconclusive`; policy not invoked in every repeat |
| interactive batch, c=8 | 24, 24, 24 | -25.94%, -25.01%, -25.00% | -25.31% / [-26.66%, -23.96%] | +34.57% / [+31.96%, +37.17%] | `not-beneficial-in-tested-cell` |

In both ascending-mixed repeats that exercised BidKV, the baseline and candidate
each made 63 preemptions. BidKV selected twice and safely abstained 61 times;
throughput was +0.33% and +0.12%, and P95 latency was -0.27% and -0.09%. The old
implementation selected on all 63 calls in the same adversarial ordering and
collapsed to -57.79% throughput and +148.52% P95 latency. This is the direct
regression evidence that bounded abstention removes the requester
self-preemption cascade without disabling the selector.

The interactive result is intentionally not generalized. Its two utility picks
preempted a nearly complete large request before the remaining 22 calls safely
abstained; the tested default parameter cell is therefore not recommended for
that workload.

## API execution, graph, correctness, and rollback

- Runtime logs contain `UTILITY_ACTIVE` and `CASCADE_ABSTAIN`; Prometheus exports
  calls, selections, abstentions, failures, invalid selections, and the enabled
  gauge for the exact `BidkvPreemptionPolicy` class.
- Every candidate arm that reached pressure observed non-zero policy calls; the
  repeated interactive cell accumulated 72 calls, 6 selections, and 66
  abstentions with zero failures/invalid selections.
- Every arm records graph capture completion, at least one replay marker, and TP
  ranks `[0, 1, 2, 3]`.
- Short deterministic output hashes match. Long forced generations remain a
  non-deterministic TP4 graph scheduling probe: baseline-versus-baseline controls
  also diverge, so they are not misclassified as a BidKV correctness failure.
- The runner removes all manager overrides in `finally`, restarts the original
  `core-762f85b3-plugin-4e57439e-hybrid4-cann9.1` production image, waits for
  HTTP 200, and snapshots NPU state. NPU4-7 have no test process.

## Evidence custody

- build and exact-image contract:
  `/data/codex-build-artifacts/sage-mate-bidkv-fix-199e0bd-20260905`
- five-cell stage one:
  `/data/codex-build-artifacts/sage-mate-bidkv-fixed-199e0bd-stage1-20260905`
- alternating three-repeat stage two:
  `/data/codex-build-artifacts/sage-mate-bidkv-fixed-199e0bd-stage2-20260905`
- pre-fix diagnostic matrix:
  `/data/codex-build-artifacts/sage-mate-bidkv-config-matrix-20260905-stage1`

The source suite passed 385 tests with 25 skipped; focused contract tests passed
26 tests with 17 skipped, and Ruff passed. `installed`, `configured`, `enabled`,
and `runtimeEffective` remain live-instance observations and are not inferred
from this artifact evidence.
