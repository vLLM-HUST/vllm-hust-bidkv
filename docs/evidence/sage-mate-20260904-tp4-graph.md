# Sage Mate Qwen3.8-27B TP4 graph qualification — 2026-09-04

## Verdict and immutable inputs

BidKV is compatible with the exact lane below:

- vLLM-HUST `762f85b311fbab0bcf8921dd216f5093cd58b9b8`
  (`0.28.1rc1.dev319`), with preemption-policy API v1 candidate
  `7362232895e0a38bb5ef4ac11fc4b2e2aa3026dd`
- vLLM-Ascend-HUST `4e57439e58ed3d78e675f9fd7b4614fb183c5394`
  (`0.25.1rc1`)
- BidKV `463f798b209a33ff2d2f4e277b9aedb26d75fa29`
- Extension Manager `24036c11c894c3fe0736e59efd17159c5e307783`
- Image `sage-mate/mod-compat:bidkv-r004`, ID
  `sha256:2d427264d22cfeb5fcf211e322dea23daa704175d17bbf4f4749763c86a535a8`
- Model Qwen3.8-27B, Ascend NPU 0–3, TP4, graph mode

This result does not cover TP1, eager mode, legacy images, other commits or
other model families.

## Runtime proof

Four concurrent requests, each with a 12,517-token prompt and 2,048 requested
output tokens, completed with HTTP 200 in 336.584 seconds. Scheduler telemetry
recorded 187 policy calls, zero policy failures, 6 utility selections and 181
liveness-fallback decisions. The non-zero utility count proves the configured
BidKV implementation—not merely the built-in fallback—executed under real KV
pressure.

Cancellation and subsequent inference recovered correctly. Emergency disable
and a fresh service process restored the built-in victim policy. Output probes
passed before rollback and after production restoration.

The raw local evidence bundle is retained by the parent campaign at
`results/sage-mate-mod-compat-20260904T053524Z/bidkv-r004/`. Compatibility must
be requalified whenever any immutable input above changes.
