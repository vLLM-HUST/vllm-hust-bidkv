"""Run paired Qwen3.5-35B BidKV benchmarks with a fixed benchmark-repo trace.

Example:
  python scripts/run_qwen35_kv_pressure.py \
    --model /path/to/Qwen3.5-35B-A3B --devices 0,1,2,3 \
    --output-dir /workspace/bidkv-qwen35-results

The script starts a fresh server for each arm, preserves raw results and logs,
and reports policy activity so a throughput delta cannot be mistaken for an
effective BidKV run when the selector was never called.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


POLICY = "bidkv.adapters.vllm_hust.selector.BidkvPreemptionPolicy"
POLICY_ENV = {
    "enable_utility_victim_selection": "BIDKV_UTILITY_ENABLE",
    "utility_strategy": "BIDKV_UTILITY_STRATEGY",
    "utility_kill_switch": "BIDKV_UTILITY_KILL_SWITCH",
    "utility_completion_weight": "BIDKV_UTILITY_COMPLETION_WEIGHT",
    "utility_preempt_weight": "BIDKV_UTILITY_PREEMPT_WEIGHT",
    "utility_kv_gate": "BIDKV_UTILITY_KV_GATE",
    "utility_cooldown_s": "BIDKV_UTILITY_COOLDOWN_S",
    "utility_min_running": "BIDKV_UTILITY_MIN_RUNNING",
    "utility_liveness_preemptions": "BIDKV_UTILITY_LIVENESS_PREEMPTIONS",
    "utility_cascade_gain_ratio": "BIDKV_UTILITY_CASCADE_GAIN_RATIO",
    "utility_snapshot_enabled": "BIDKV_UTILITY_SNAPSHOT_ENABLED",
    "utility_snapshot_top_k": "BIDKV_UTILITY_SNAPSHOT_TOP_K",
    "utility_snapshot_history_size": "BIDKV_UTILITY_SNAPSHOT_HISTORY_SIZE",
    "utility_epsilon": "BIDKV_UTILITY_EPSILON",
    "utility_default_max_tokens": "BIDKV_UTILITY_DEFAULT_MAX_TOKENS",
}
DEFAULT_TRACE = (
    Path(__file__).resolve().parents[2]
    / "vllm-hust-benchmark/scripts/traces/evoscientist-workload-custom.jsonl"
)


def http_get(url: str, timeout: float = 3) -> bytes:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return response.read()


def wait_for_server(process: subprocess.Popen, port: int, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup: {process.returncode}")
        try:
            http_get(f"http://127.0.0.1:{port}/health")
            return
        except (OSError, urllib.error.URLError):
            time.sleep(2)
    raise TimeoutError(f"server did not become healthy within {timeout}s")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def metric_value(metrics: str, name: str, event: str | None = None) -> float:
    values = []
    for line in metrics.splitlines():
        if not (line.startswith(f"{name}{{") or line.startswith(f"{name} ")):
            continue
        if event is not None and f'event="{event}"' not in line:
            continue
        match = re.search(r"\s([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$", line)
        if match:
            values.append(float(match.group(1)))
    return sum(values)


def git_head(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--devices", required=True, help="Explicit NPU IDs, e.g. 0,1,2,3")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--port", type=int, default=18581)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--start-repeat", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=268435456)
    parser.add_argument("--startup-timeout", type=int, default=900)
    parser.add_argument("--candidate-config", type=json.loads, default={})
    parser.add_argument("--arms", choices=("both", "baseline", "bidkv"), default="both")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.model.is_dir() or not (args.model / "config.json").is_file():
        parser.error("--model must point to a downloaded model directory")
    if not args.dry_run:
        index_path = args.model / "model.safetensors.index.json"
        if not index_path.is_file():
            parser.error("model weights are incomplete: missing safetensors index")
        weight_files = set(json.loads(index_path.read_text())["weight_map"].values())
        missing = sorted(name for name in weight_files if not (args.model / name).is_file())
        if missing:
            parser.error(f"model weights are incomplete: {len(missing)} shard(s) missing")
    if not args.trace.is_file():
        parser.error(f"trace not found: {args.trace}")
    devices = args.devices.split(",")
    if not devices or any(not item.isdecimal() for item in devices) or len(set(devices)) != len(devices):
        parser.error("--devices must be a comma-separated list of unique NPU IDs")
    if args.repeats < 1 or args.num_prompts < 1 or args.max_concurrency < 1:
        parser.error("repeats, num-prompts, and max-concurrency must be positive")
    if not 1 <= args.start_repeat <= args.repeats:
        parser.error("start-repeat must be between 1 and repeats")
    if not isinstance(args.candidate_config, dict):
        parser.error("--candidate-config must be a JSON object")
    vllm_executable = Path(sys.executable).with_name("vllm")
    if not vllm_executable.is_file():
        resolved = shutil.which("vllm")
        vllm_executable = Path(resolved) if resolved else vllm_executable
    if not vllm_executable.is_file():
        parser.error("vllm command not found in the active environment")

    candidate_config = {
        "enable_utility_victim_selection": True,
        **args.candidate_config,
    }
    if candidate_config["enable_utility_victim_selection"] is not True:
        parser.error("candidate_config must keep enable_utility_victim_selection=true")
    unknown_config = set(candidate_config) - POLICY_ENV.keys()
    if unknown_config:
        parser.error(f"unknown candidate_config keys: {sorted(unknown_config)}")
    candidate_env = {
        POLICY_ENV[key]: str(int(value)) if isinstance(value, bool) else str(value)
        for key, value in candidate_config.items()
    }
    dataset_sha = hashlib.sha256(args.trace.read_bytes()).hexdigest()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    index_path = args.model / "model.safetensors.index.json"
    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bidkv_commit": git_head(Path(__file__).resolve().parents[1]),
        "benchmark_script_sha256": script_sha,
        "benchmark_commit": git_head(args.trace.parent),
        "model_config_sha256": hashlib.sha256(
            (args.model / "config.json").read_bytes()
        ).hexdigest(),
        "model_index_sha256": (
            hashlib.sha256(index_path.read_bytes()).hexdigest()
            if index_path.is_file() else None
        ),
        "vllm_version": importlib.metadata.version("vllm"),
        "vllm_ascend_version": importlib.metadata.version("vllm-ascend"),
        "bidkv_version": importlib.metadata.version("vllm-hust-bidkv"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    records = json.loads(summary_path.read_text()) if summary_path.is_file() else []
    if args.arms == "bidkv":
        for repeat in range(args.start_repeat, args.repeats + 1):
            baseline = next(
                (row for row in records if row["repeat"] == repeat and row["arm"] == "baseline"),
                None,
            )
            if baseline is None or baseline["dataset_sha256"] != dataset_sha or baseline["devices"] != devices:
                parser.error(f"repeat {repeat}: matching saved baseline required for --arms bidkv")
    for repeat in range(args.start_repeat - 1, args.repeats):
        # Alternate order to reduce drift across paired runs.
        arms = ("baseline", "bidkv") if repeat % 2 == 0 else ("bidkv", "baseline")
        if args.arms != "both":
            arms = (args.arms,)
        for arm in arms:
            run_dir = args.output_dir / f"repeat-{repeat + 1}-{arm}"
            run_dir.mkdir(parents=True, exist_ok=True)
            server_cmd = [
                str(vllm_executable), "serve", str(args.model),
                "--host", "127.0.0.1", "--port", str(args.port),
                "--tensor-parallel-size", str(len(devices)),
                "--max-model-len", str(args.max_model_len),
                "--max-num-seqs", str(args.max_concurrency),
                "--kv-cache-memory-bytes", str(args.kv_cache_memory_bytes),
            ]
            if arm == "bidkv":
                server_cmd += [
                    "--preemption-policy", POLICY,
                ]
            bench_cmd = [
                str(vllm_executable), "bench", "serve",
                "--backend", "vllm", "--endpoint", "/v1/completions",
                "--host", "127.0.0.1", "--port", str(args.port),
                "--model", str(args.model),
                "--dataset-name", "custom", "--dataset-path", str(args.trace),
                "--custom-output-len", "-1", "--disable-shuffle", "--ignore-eos",
                "--num-prompts", str(args.num_prompts),
                "--max-concurrency", str(args.max_concurrency),
                "--request-rate", str(args.request_rate),
                "--seed", "0", "--save-result",
                "--result-dir", str(run_dir), "--result-filename", "result.json",
            ]
            record = {
                "arm": arm,
                "repeat": repeat + 1,
                "dataset_sha256": dataset_sha,
                "provenance": provenance,
                "server_command": server_cmd,
                "bench_command": bench_cmd,
                "devices": devices,
                "policy_environment": candidate_env if arm == "bidkv" else {},
            }
            (run_dir / "commands.json").write_text(json.dumps(record, indent=2) + "\n")
            if args.dry_run:
                print(json.dumps(record, indent=2))
                continue
            env = os.environ.copy()
            env["ASCEND_VISIBLE_DEVICES"] = args.devices
            env["ASCEND_RT_VISIBLE_DEVICES"] = args.devices
            for name in POLICY_ENV.values():
                env.pop(name, None)
            if arm == "bidkv":
                env.update(candidate_env)
            with (run_dir / "npu-before.txt").open("wb") as npu_log:
                subprocess.run(["npu-smi", "info"], stdout=npu_log, check=True)
            with (run_dir / "server.log").open("wb") as server_log:
                process = subprocess.Popen(
                    server_cmd, stdout=server_log, stderr=subprocess.STDOUT,
                    env=env, start_new_session=True,
                )
                try:
                    wait_for_server(process, args.port, args.startup_timeout)
                    with (run_dir / "bench.log").open("wb") as bench_log:
                        subprocess.run(
                            bench_cmd, stdout=bench_log, stderr=subprocess.STDOUT,
                            env=env, check=True,
                        )
                    metrics = http_get(f"http://127.0.0.1:{args.port}/metrics").decode()
                    (run_dir / "metrics.txt").write_text(metrics)
                    result = json.loads((run_dir / "result.json").read_text())
                    record.update({
                        "completed": result.get("completed"),
                        "output_throughput": result.get("output_throughput"),
                        "total_output_tokens": result.get("total_output_tokens"),
                        "preemptions": metric_value(metrics, "vllm:num_preemptions_total"),
                        "policy_calls": metric_value(
                            metrics, "vllm:preemption_policy_events", "calls"
                        ),
                        "policy_selections": metric_value(
                            metrics, "vllm:preemption_policy_events", "selections"
                        ),
                        "policy_failures": metric_value(
                            metrics, "vllm:preemption_policy_events", "failures"
                        ),
                    })
                    if record["completed"] != args.num_prompts:
                        raise RuntimeError(f"incomplete benchmark: {record['completed']} requests")
                    if not isinstance(record["output_throughput"], (int, float)) or not math.isfinite(
                        record["output_throughput"]
                    ) or record["output_throughput"] <= 0:
                        raise RuntimeError("benchmark reported invalid output throughput")
                    if arm == "bidkv" and record["policy_calls"] <= 0:
                        raise RuntimeError("BidKV policy was not invoked; increase KV pressure")
                    if arm == "bidkv" and record["policy_failures"] > 0:
                        raise RuntimeError("BidKV policy failed; inspect metrics and server log")
                    records.append(record)
                    summary_path.write_text(
                        json.dumps(records, indent=2) + "\n"
                    )
                    print(
                        f"{arm} repeat={repeat + 1} output_throughput="
                        f"{record['output_throughput']:.3f} tok/s "
                        f"policy_calls={record['policy_calls']:.0f}"
                    )
                finally:
                    with (run_dir / "npu-after.txt").open("wb") as npu_log:
                        subprocess.run(["npu-smi", "info"], stdout=npu_log, check=False)
                    stop_server(process)

    if args.dry_run:
        return
    for repeat in range(1, args.repeats + 1):
        pair = {row["arm"]: row for row in records if row["repeat"] == repeat}
        if len(pair) == 2:
            if pair["bidkv"]["total_output_tokens"] != pair["baseline"][
                "total_output_tokens"
            ]:
                raise RuntimeError(f"repeat {repeat}: output-token totals differ across arms")
            delta = 100 * (
                pair["bidkv"]["output_throughput"]
                / pair["baseline"]["output_throughput"] - 1
            )
            print(f"repeat={repeat} BidKV output throughput delta={delta:+.2f}%")


if __name__ == "__main__":
    main()
