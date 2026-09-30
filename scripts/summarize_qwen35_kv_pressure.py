"""Print a CSV of every complete paired Qwen3.5 KV-pressure run.

Example:
  python scripts/summarize_qwen35_kv_pressure.py \
    results/qwen35-evoscientist-20260929 > comparisons.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


FIELDS = (
    "run",
    "repeat",
    "baseline_tps",
    "bidkv_tps",
    "delta_pct",
    "completed",
    "output_tokens",
    "baseline_preemptions",
    "bidkv_preemptions",
    "policy_calls",
    "policy_selections",
    "policy_failures",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()

    for summary_path in sorted(args.results_dir.glob("bidkv-*/summary.json")):
        records = json.loads(summary_path.read_text())
        for repeat in sorted({row["repeat"] for row in records}):
            pair = {row["arm"]: row for row in records if row["repeat"] == repeat}
            if set(pair) != {"baseline", "bidkv"}:
                continue
            baseline, bidkv = pair["baseline"], pair["bidkv"]
            if baseline["dataset_sha256"] != bidkv["dataset_sha256"]:
                raise ValueError(f"{summary_path}: repeat {repeat} used different inputs")
            for key in ("completed", "total_output_tokens"):
                if baseline[key] != bidkv[key]:
                    raise ValueError(f"{summary_path}: repeat {repeat} differs in {key}")
            if bidkv["policy_calls"] <= 0 or bidkv["policy_failures"]:
                raise ValueError(f"{summary_path}: repeat {repeat} has invalid policy activity")
            writer.writerow(
                {
                    "run": summary_path.parent.name,
                    "repeat": repeat,
                    "baseline_tps": baseline["output_throughput"],
                    "bidkv_tps": bidkv["output_throughput"],
                    "delta_pct": 100
                    * (bidkv["output_throughput"] / baseline["output_throughput"] - 1),
                    "completed": baseline["completed"],
                    "output_tokens": baseline["total_output_tokens"],
                    "baseline_preemptions": baseline["preemptions"],
                    "bidkv_preemptions": bidkv["preemptions"],
                    "policy_calls": bidkv["policy_calls"],
                    "policy_selections": bidkv["policy_selections"],
                    "policy_failures": bidkv["policy_failures"],
                }
            )


if __name__ == "__main__":
    main()
