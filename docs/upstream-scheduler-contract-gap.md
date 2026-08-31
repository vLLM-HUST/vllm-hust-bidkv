# BidKV upstream scheduler contract gap

Status: blocked on an upstream contract, not blocked on the BidKV scoring
algorithm. This document is evidence for the release freeze; it is not an API
promise.

## Pinned upstream evidence

- vLLM RFC [#51608](https://github.com/vllm-project/vllm/issues/51608) was
  still open on 2026-09-01.
- Draft PR [#51601](https://github.com/vllm-project/vllm/pull/51601) was still
  open and draft at head `f8b7db61e446911e0d62fcb8220f863d6098c471`.
- The draft implementation exposes `QueueSortPlugin`, `FilterPlugin`,
  `ScorePlugin`, and one `PreemptionPlugin`. Its preemption method ranks live
  `Request` objects through `preemption_key(request, running_position)`.
- The design document in the same commit instead specifies composable,
  weighted, batched `PreemptionScore` over stable handles and read-only feature
  views.
- The RFC says trusted out-of-tree support comes after interface
  stabilization. The draft implementation has an in-process registry and
  `register_scheduler_plugin()` but no `importlib.metadata` entry-point
  discovery, even though its design document sketches a future
  `vllm.scheduler_plugins` descriptor entry point.

These are material contract differences. BidKV must not publish an adapter
against one draft shape and label it compatible with the other.

## Minimal BidKV mapping

| BidKV behavior | Upstream extension point | First migration decision |
|---|---|---|
| Rank already-approved running victims | Preemption / future PreemptionScore | Required; this is the minimum BidKV serving contract |
| Decide when KV pressure should trigger preemption | Scheduler-owned mechanism or a future explicit trigger contract | Not implemented by a policy calling private scheduler methods |
| Reorder waiting requests for admission | Score or dynamic QueueSort | Separate optional capability with its own workload evidence |
| Track request completion and prior preemptions | Scheduler-owned read-only features and batched lifecycle events | Required only when the stable interface provides them |
| Perform preemption, KV cleanup, reinsertion, or block mutation | vLLM core | Never owned by BidKV |

The initial upstream consumer should implement only victim ranking. Legacy
`schedule()`, `_preempt_request()`, waiting-list mutation, and private metrics
hooks are experiment code, not part of the new adapter.

## Activation and packaging rules

1. Installing `bidkv` remains inert.
2. The main distribution does not register the private
   `vllm.victim_selector` group.
3. The legacy selector remains import-only for a pinned archived fork.
4. A future adapter may register `vllm.scheduler_plugins` only after upstream
   actually implements and documents out-of-tree discovery.
5. Extension Manager must require explicit host and protocol evidence and fail
   before launch when either is missing or incompatible.
6. No compatibility range may include the fresh official fork until a real
   EngineCore loads the adapter and calls it.

## Acceptance gate for replacing the legacy manifest

- Pin a reviewed upstream commit whose code and design agree on naming,
  descriptor, data ownership, composition, and error behavior.
- Prove package installation does not load or activate BidKV.
- Prove explicit configuration loads exactly one intended implementation in
  EngineCore and duplicate names fail at startup.
- Replay deterministic victim decisions through the actual upstream interface.
- Run an online workload and report scheduler CPU overhead, throughput, TTFT,
  TPOT, fairness, preemption count, and KV/recompute effects.
- Inject invalid configuration, constructor failure, invalid score/index,
  runtime failure, and conflicting policy configuration; none may silently
  fall back to another policy.
- Disable BidKV, start a new vLLM process, and prove the built-in policy is
  restored without stale Manager intent.

Only after these gates pass should the manifest replace
`vllm.victim_selector` with the stabilized upstream protocol and change the
implementation carrier from `legacy_unregistered` to an active carrier.
