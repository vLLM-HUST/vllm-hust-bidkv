"""Finite Native/BidKV graph admission; all engine/request settings are inputs.

Manager appends --preemption-policy and --additional-config for the candidate.
This script never grants formal task-quality, mechanism or cost qualification.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import time
from pathlib import Path


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preemption-policy")
    parser.add_argument("--additional-config", type=json.loads, default={})
    args = parser.parse_args()
    assert not args.output.exists(), "Output must be new"
    args.output.mkdir(parents=True)
    frozen = json.loads(args.profile.read_text())
    central_path = args.project_root / frozen["central_profile"]
    assert digest(central_path) == frozen["central_profile_sha256"]
    central = json.loads(central_path.read_text())
    engine = copy.deepcopy(central["engine"])
    spec = importlib.util.spec_from_file_location(
        "shared_test_profile", args.project_root / "tools/shared_test_profile.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    packages = {name: importlib.metadata.version(name) for name in central["packages"]}
    # Engine validation and full SDK identity are separate facts. Preserve an
    # exact-version rejection; a local build is never relabeled as the release.
    checker.validate_engine(engine, central["packages"])
    package_differences = {
        name: {"reference": value, "actual": packages[name]}
        for name, value in central["packages"].items() if packages[name] != value
    }
    assert package_differences == frozen["declared_package_differences"]
    sdk_error = None
    try:
        checker.validate_engine(engine, packages)
    except ValueError as error:
        sdk_error = str(error)
    for path, expected in frozen["file_bindings"].items():
        assert digest(path) == expected, "Changed frozen input: " + path
    expected_bundles = "org.vllm-hust.bidkv" if args.preemption_policy else ""
    assert os.environ.get("VLLMHUST_EXT_ENABLED_BUNDLES", "") == expected_bundles
    if args.preemption_policy:
        assert args.preemption_policy == frozen["policy"]
        assert json.loads(os.environ["BIDKV_RUNTIME_CONTRACT"]) == frozen["target_contract"]
        assert args.additional_config == {}
    else:
        assert not args.additional_config
    engine["additional_config"].update(args.additional_config)
    engine.update(model=args.model, worker_cls=frozen["worker_cls"], disable_log_stats=False)
    if args.preemption_policy:
        engine["preemption_policy"] = args.preemption_policy
    save(args.output / "identity.json", {
        "profile_sha256": digest(args.profile), "runner_sha256": digest(__file__),
        "engine_arguments": engine, "packages": packages,
        "package_differences": package_differences,
        "shared_engine_parameters_match": True,
        "full_shared_contract_passed": sdk_error is None,
        "full_shared_contract_error": sdk_error,
        "qualification": "development admission only; no formal cost or task claim",
    })
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": row["question"]}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False) for row in frozen["requests"]]
    llm = None
    measurements = []
    try:
        llm = LLM(**engine)
        window = "bidkv-admission-" + str(time.time_ns())
        before = llm.collective_rpc("environment_snapshot", kwargs={"window": window})
        save(args.output / "workers-before.json", before)
        for phase, repeats in [("warmup", frozen["warmups"]), ("measured", frozen["repeats"])]:
            for repeat in range(repeats):
                started = time.perf_counter()
                outputs = llm.generate(
                    prompts, SamplingParams(**frozen["sampling"]), use_tqdm=False)
                seconds = time.perf_counter() - started
                rows = []
                assert len(outputs) == len(prompts)
                for request, output in zip(frozen["requests"], outputs, strict=True):
                    answer = output.outputs[0]
                    rows.append({"case_id": request["id"],
                                 "input_tokens": len(output.prompt_token_ids),
                                 "token_ids": list(answer.token_ids), "text": answer.text,
                                 "output_tokens": len(answer.token_ids),
                                 "finish_reason": answer.finish_reason,
                                 "expected": request["expected"],
                                 "answer_matches": answer.text.strip() == request["expected"]})
                measurements.append({"phase": phase, "repeat": repeat, "seconds": seconds,
                                     "input_tokens": sum(x["input_tokens"] for x in rows),
                                     "output_tokens": sum(x["output_tokens"] for x in rows),
                                     "responses": rows})
                save(args.output / "measurements.json", measurements)
        from prometheus_client import generate_latest
        (args.output / "metrics.prom").write_bytes(generate_latest())
        after = llm.collective_rpc("environment_snapshot")
        save(args.output / "workers-after.json", after)
        assert {row["rank"] for row in after} == {0, 1}
        before_by_rank = {row["rank"]: row for row in before}
        replays = {}
        for row in after:
            previous = before_by_rank[row["rank"]]
            assert row["worker_epoch"] == previous["worker_epoch"]
            assert row["enforce_eager"] is False
            assert row["cudagraph_mode"] == "FULL_DECODE_ONLY"
            assert row["eligible_decode_without_replay"] == 0
            replays[row["rank"]] = (row["decode_graph_replay_count"]
                                    - previous["decode_graph_replay_count"])
            assert replays[row["rank"]] > 0
        save(args.output / "result.json", {
            "success": True, "arm": "bidkv" if args.preemption_policy else "native",
            "graph_replays_by_rank": replays,
            "all_synthetic_answers_match": all(
                x["answer_matches"] for m in measurements for x in m["responses"]),
            "qualification": "finite synthetic admission; no dataset/benefit qualification",
        })
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
