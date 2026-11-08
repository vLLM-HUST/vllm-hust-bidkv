#!/usr/bin/env python3
"""Build a fail-closed W1 four-arm decision receipt."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from bidkv.evidence.w1_decision_receipt import EvidenceError, build_receipt, load_trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        trace, trace_hash = load_trace(args.trace, root=args.root)
        receipt = build_receipt(trace, trace_hash)
    except EvidenceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
