"""Check a declared development target against the effective host configuration.

This is startup admission, never a functional or performance qualification.
Historical adapters without a target contract retain their existing behavior.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any


def validate_runtime_contract(vllm_config: Any) -> dict[str, Any] | None:
    additional = getattr(vllm_config, "additional_config", None) or {}
    target = additional.get("bidkv_runtime_contract")
    raw = os.environ.get("BIDKV_RUNTIME_CONTRACT")
    if raw:
        try:
            environment_target = json.loads(raw)
        except (ValueError, TypeError) as error:
            raise ValueError("BidKV runtime contract environment is invalid JSON") from error
        if target is not None and target != environment_target:
            raise ValueError("BidKV runtime contracts disagree between environment and config")
        target = environment_target
    if target is None:
        return None
    if (not isinstance(target, dict) or type(target.get("schema_version")) is not int
            or target["schema_version"] != 1):
        raise ValueError("BidKV runtime contract must be a schema_version=1 object")
    fields = {
        "schema_version", "profile_id", "model_type", "architecture",
        "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
        "dtype", "enforce_eager", "cudagraph_mode", "qualification_status",
    }
    if set(target) != fields:
        raise ValueError("BidKV runtime contract fields are missing or unknown")
    if target["qualification_status"] != "unqualified-development-candidate":
        raise ValueError("BidKV development admission cannot declare qualification passed")
    if not isinstance(target["profile_id"], str) or not target["profile_id"]:
        raise ValueError("BidKV runtime contract requires a profile_id")
    for field in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
        if type(target[field]) is not int or target[field] < 1:
            raise ValueError(f"BidKV runtime contract requires a positive integer {field}")
    if target["enforce_eager"] is not False:
        raise ValueError("BidKV target admission requires enforce_eager=false")
    if target["cudagraph_mode"] != "FULL_DECODE_ONLY":
        raise ValueError("BidKV target admission requires FULL_DECODE_ONLY")
    model = getattr(vllm_config, "model_config", None)
    parallel = getattr(vllm_config, "parallel_config", None)
    compilation = getattr(vllm_config, "compilation_config", None)
    hf_config = getattr(model, "hf_config", None)
    mode = getattr(compilation, "cudagraph_mode", None)
    actual = {
        "model_type": getattr(hf_config, "model_type", None),
        "tensor_parallel_size": getattr(parallel, "tensor_parallel_size", None),
        "pipeline_parallel_size": getattr(parallel, "pipeline_parallel_size", None),
        "data_parallel_size": getattr(parallel, "data_parallel_size", None),
        "dtype": str(getattr(model, "dtype", None)).removeprefix("torch."),
        "enforce_eager": getattr(model, "enforce_eager", None),
        "cudagraph_mode": getattr(mode, "name", str(mode)),
    }
    mismatches = [
        f"{key}: expected {target[key]!r}, observed {value!r}"
        for key, value in actual.items()
        if value != target[key] or type(value) is not type(target[key])
    ]
    architectures = getattr(hf_config, "architectures", None) or []
    if target["architecture"] not in architectures:
        mismatches.append(
            f"architecture: expected {target['architecture']!r}, observed {architectures!r}"
        )
    if mismatches:
        raise ValueError("BidKV runtime target mismatch: " + "; ".join(mismatches))
    receipt = {"profile_id": target["profile_id"], **actual,
               "architecture": target["architecture"],
               "qualification_status": target["qualification_status"]}
    logging.getLogger("vllm").info(
        "[BidKV] TARGET_ADMITTED profile=%s model=%s TP=%s PP=%s graph=%s "
        "qualification=unqualified-development-candidate",
        receipt["profile_id"], receipt["model_type"], receipt["tensor_parallel_size"],
        receipt["pipeline_parallel_size"], receipt["cudagraph_mode"],
    )
    return receipt
