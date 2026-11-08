# W1 pressure-event decision receipt

Issue #5 requires stock, static, BidKV, and oracle decisions on one frozen
pressure-event trace. The repository now provides an admission gate for that
artifact; it does not manufacture a result from the four-event integration
fixture or from projected data.

The input must use schema `bidkv.pressure-trace/v1`, declare
`evidence_level=real-serving`, bind exact repository/runtime commits, model and
hardware, and bind the source pressure log by repository-relative path and
SHA-256. Every candidate needs the observed counterfactual disruption cost
from the matched collection campaign. Missing candidates, duplicate IDs,
non-finite costs, insufficient capacity, path escape, or source-log drift fail
closed.

Run the replay from an installed editable checkout:

```bash
python scripts/build_w1_decision_receipt.py evidence/w1/pressure-trace.json \
  --output evidence/w1/decision-receipt.json
```

All four arms consume the exact same candidate universe. The output is written
atomically with mode 0600 and is content-addressed. Its evidence level is
`decision-replay-only`: it does not establish throughput, latency, output
quality, or the portfolio-level 20%/30% performance gates. Those still require
matched real serving runs and benchmark custody.
