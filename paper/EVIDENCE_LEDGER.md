# Evidence ledger

| Claim | Evidence | Level | Boundary |
|---|---|---|---|
| Request-level utility selector, baselines, native plugin, kill switch, and runner exist | `src/bidkv/`, package metadata, and host tests | host contract | No serving or performance implication |
| BidKV frees required capacity through native preempt/recompute | no versioned serving result | 待验证假设 | Requires pressure-event, victim, release, and cache-accounting receipts |
| Disruption surrogate predicts real recompute/tail cost | none | 待验证假设 | `quality_delta` naming is not evidence of quality or prediction |
| BidKV improves SLOs over strongest victim baseline | none | 待验证假设 | Requires matched real serving distributions and oracle pass |
| Result transfers between vLLM and SGLang | none | 待验证假设 | Full-footprint and private-token ownership must be analyzed separately |

No NPU/GPU experiment or result was generated for this paper. Future
vLLM-HUST execution uses the official `.23` container and scheduler allocation
identity. Student owners retain implementation and experiment responsibility.
