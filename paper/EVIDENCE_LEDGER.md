# Evidence ledger

This ledger belongs to a post-acceptance maintenance qualification. It does not
re-evaluate, replace, or retract the accepted SC 2026 BidKV artifact and its
original results. Prospective regime-map and calibration claims below are
optional follow-up work, not acceptance debt.

| Claim | Evidence | Level | Boundary |
|---|---|---|---|
| Request-level utility selector, baselines, native plugin, kill switch, and runner exist | `src/bidkv/`, package metadata, and 385-passed/25-skipped source-suite result | host contract | No device-performance implication |
| Native graph-mode integration executes on the qualified configuration | `docs/evidence/sage-mate-20260905-bounded-preemption-matrix.md`, five functional cells | controlled real-device qualification | One Qwen3.8-27B model, Ascend TP4, `FULL_DECODE_ONLY`; not a production or cross-device claim |
| The current formula is beneficial in the repeated interactive cell | Three-repeat matched result: throughput -25.31% (95% CI [-26.66,-23.96]), p95 latency +34.57% (95% CI [+31.96,+37.17]) | negative result | Reject benefit/default-enablement for this cell; do not generalize to all regimes |
| BidKV improves the ascending-mixed cell | Three repeats with policy calls 63, 0, 63 | inconclusive | Inconsistent treatment exposure prevents an effectiveness estimate |
| BidKV frees required capacity through native preempt/recompute | qualified-run policy counters plus externally held raw receipts | bounded serving evidence | Checked-in summary does not permit independent raw-result reproduction |
| Disruption surrogate predicts real recompute/tail cost | none | 待验证假设 | `quality_delta` naming is not evidence of quality or prediction |
| BidKV improves SLOs over strongest victim baseline outside the negative interactive cell | none | 待验证假设 | Requires a preregistered regime, consistent policy invocation, matched distributions, and oracle pass |
| Result transfers between vLLM and SGLang | none | 待验证假设 | Full-footprint and private-token ownership must be analyzed separately |

This paper generated no new NPU/GPU experiment or result. It reports the
existing versioned qualification at its recorded evidence ceiling; raw custody
remains external. Future vLLM-HUST execution uses the official `.23` container
and scheduler allocation identity. Student owners retain implementation and
experiment responsibility.
