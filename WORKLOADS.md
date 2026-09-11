# Workload source

BidKV pins `intellistream/llm-serving-workloads` at
`ceef92d52e0c49f26ba1efc6706edd1f6df5d913` under
`workloads/llm-serving-workloads`. Families, presets, and replay metadata provide
matched request pressure for victim-selection baselines and treatments; the pin
does not choose an experiment or validate a scheduling benefit.

The private catalog has no repository license file and is internal-only. Every
comparison must freeze case IDs, arrival/replay identity, seeds, and split roles.
Updates require an explicit reviewed gitlink change and checker revalidation;
floating branches are not accepted.
