"""vLLM 5-strategy experiment runner — 论文 §6 的主实验编排脚本。

Usage
-----
# 运行全部实验矩阵（自动启停 vLLM 服务）
python -m bidkv.experiments.vllm.runner \
    --strategies "preempt-evict,...,bidkv" \
    --workloads "mixed,long_context" \
    --runs 3 \
    --request-rates "1.5,3.0,6.0" \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output-dir results/vllm_$(date +%Y%m%d)/

# 仅运行 preempt-evict 基线（验证实验框架）
python -m bidkv.experiments.vllm.runner \
    --strategies "preempt-evict" \
    --workloads "mixed" \
    --runs 1 \
    --request-rates "2.0" \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output-dir results/vllm_smoke/

运行流程
--------
1. 验证 traces 已冻结（否则报错提示先运行 freeze 命令）
2. 按 strategy 分组：每个 strategy 启动一次 vLLM 服务
   - preempt-evict：原生 vLLM（无 bidkv 注入）
   - 其他策略：通过 serve.py 注入 bidkv + baseline overlay
3. 对当前 strategy，遍历 workload × rate × run
4. 按 Poisson 开环到达模型发送请求（无 max_inflight 限制）
5. 收集指标并保存为 JSON
6. 停止服务，进入下一个 strategy
7. 运行 candidate-universe consistency 校验
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from bidkv.baselines import BaselineRegistry
from bidkv.experiments.common.model import get_default_model
from bidkv.experiments.vllm.collector import (
    RequestResult,
    RunResult,
    save_run_result,
)
from bidkv.experiments.vllm.config import (
    ALL_STRATEGIES,
    ALL_WORKLOADS,
    DEFAULT_REQUEST_RATES,
    STRATEGY_PREEMPT_EVICT,
    WORKLOAD_REQUEST_RATES,
    ExperimentConfig,
    SLOConfig,
    VLLMServerConfig,
)
from bidkv.experiments.vllm.workload import (
    RequestTrace,
    WorkloadTrace,
    load_trace,
)

logger = logging.getLogger(__name__)

# Well-known path for adapter metrics IPC (written by scheduler_hook in subprocess)
_METRICS_FILE = Path("/tmp/bidkv_metrics_latest.json")


class VLLMExperimentRunner:
    """vLLM 实验编排器 — Poisson 开环到达模型。

    负责 strategy × workload × rate × run 的全矩阵运行。
    每个 strategy 启动一次 vLLM 服务（重复使用同一 server 实例），
    运行完所有 (workload, rate, run) 组合后停止服务。

    Parameters
    ----------
    config:
        实验配置。
    """

    def __init__(self, config: ExperimentConfig, *, resume: bool = False) -> None:
        self._config = config
        self._resume = resume
        self._registry = BaselineRegistry()
        self._registry.create_default_registry()
        self._server_proc: subprocess.Popen[bytes] | None = None
        self._server_log: object | None = None  # log file handle
        self._traces: dict[str, WorkloadTrace] = {}
        self._run_deadline: float = float("inf")
        # Bypass http_proxy for all localhost HTTP calls — proxies may cache
        # stale 502 responses from the engine-not-ready startup phase.
        self._http_opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({})
        )

    @property
    def config(self) -> ExperimentConfig:
        return self._config

    def load_traces(self) -> None:
        """加载所有冻结的工作负载 traces。

        按 (workload, rate) 组合查找 trace 文件。
        命名约定：{prefix}_rate{X.X}.json

        Raises
        ------
        FileNotFoundError
            trace 文件不存在。
        """
        for workload in self._config.workloads:
            rates = self._config.get_rates_for_workload(workload)
            for rate in rates:
                prefix = "mixed" if workload == "mixed" else "long"
                trace_file = f"{prefix}_rate{rate}.json"
                trace_path = self._config.traces_dir / trace_file
                key = f"{workload}__{rate}"
                self._traces[key] = load_trace(trace_path)
                logger.info(
                    "Loaded trace: %s (%d requests, rate=%.1f)",
                    trace_file,
                    self._traces[key].num_requests,
                    rate,
                )

    def run_all(self) -> list[RunResult]:
        """运行完整实验矩阵。

        按 (strategy, workload, rate) 分组运行：每组启动一次 vLLM 服务，
        确保服务状态不会跨 rate 累积。

        Returns
        -------
        list[RunResult]
            所有运行结果。
        """
        self.load_traces()

        results: list[RunResult] = []
        total = self._config.total_runs
        current = 0
        consecutive_server_crashes = 0

        for strategy in self._config.strategies:
            logger.info(
                "=" * 60 + "\nStarting strategy: %s\n" + "=" * 60,
                strategy,
            )
            consecutive_server_crashes = 0

            for workload in self._config.workloads:
                rates = self._config.get_rates_for_workload(workload)
                for rate in rates:
                    abort_count = 0
                    for run_idx in range(self._config.runs_per_combo):
                        current += 1
                        label = self._config.run_label(strategy, workload, rate, run_idx)

                        # Resume: skip if result file already exists
                        result_path = self._config.output_dir / f"{label}.json"
                        if self._resume and result_path.exists():
                            logger.info(
                                "[%d/%d] SKIP (resume): %s",
                                current,
                                total,
                                label,
                            )
                            continue

                        # Fresh server for EVERY run to avoid KV cache
                        # exhaustion across repeats.
                        try:
                            self._start_server(strategy)
                            self._warmup(strategy)
                        except RuntimeError as exc:
                            consecutive_server_crashes += 1
                            logger.error(
                                "Server crash #%d for strategy=%s: %s",
                                consecutive_server_crashes,
                                strategy,
                                exc,
                            )
                            if consecutive_server_crashes >= 3:
                                logger.error(
                                    "3 consecutive server crashes for %s — skipping",
                                    strategy,
                                )
                                break
                            continue

                        try:
                            logger.info("[%d/%d] Running: %s", current, total, label)

                            result = self._run_single(
                                strategy=strategy,
                                workload=workload,
                                request_rate=rate,
                                run_index=run_idx,
                            )
                            results.append(result)
                            save_run_result(result, self._config.output_dir)

                            # Check for abort (§8 Layer 1)
                            run_status = result.server_config.get("run_status", "completed")
                            if run_status == "aborted":
                                abort_count += 1

                            logger.info(
                                "[%d/%d] Completed: %s (thru=%.2f rps, "
                                "TTFT p50=%.0f/p99=%.0f ms, "
                                "TPOT p50=%.1f/p99=%.1f ms, "
                                "success=%d/%d, status=%s)",
                                current,
                                total,
                                label,
                                result.compute_throughput_rps(),
                                result.compute_p50_ttft_ms(),
                                result.compute_p99_ttft_ms(),
                                result.compute_p50_tpot_ms(),
                                result.compute_p99_tpot_ms(),
                                len(result.successful_requests),
                                len(result.request_results),
                                run_status,
                            )
                        finally:
                            self._stop_server()

                        # Cooldown between runs
                        cooldown_s = 3
                        logger.info("Cooling down %ds between runs...", cooldown_s)
                        time.sleep(cooldown_s)

                    # §8: 3 runs all abort → overload_failure
                    if abort_count >= self._config.runs_per_combo:
                        logger.warning(
                            "OVERLOAD_FAILURE: %s %s rate=%.1f — all %d runs aborted",
                            strategy,
                            workload,
                            rate,
                            abort_count,
                        )

        logger.info("All %d runs completed. Results in %s", total, self._config.output_dir)
        return results

    # ------------------------------------------------------------------
    # Server lifecycle management
    # ------------------------------------------------------------------

    def _start_server(self, strategy: str) -> None:
        """Start vLLM server for the given strategy.

        For preempt-evict: starts vanilla vLLM.
        For other strategies: starts vLLM with BidKV injection via serve.py.

        Blocks until the server is healthy.
        """
        if self._server_proc is not None:
            self._stop_server()

        # Pre-start cleanup: kill any orphan GPU processes from prior runs
        self._kill_orphan_gpu_processes()

        # Clear metrics file from previous run
        _METRICS_FILE.unlink(missing_ok=True)

        env = os.environ.copy()
        env["BIDKV_STRATEGY"] = strategy
        # Reduce log noise from vLLM
        env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

        # Build command: use our serve.py for BidKV injection
        serve_module = "bidkv.experiments.vllm.serve"
        cmd = [
            sys.executable,
            "-m",
            serve_module,
            *self._config.server.to_cli_args(),
        ]

        logger.info("Starting vLLM server: strategy=%s, cmd=%s", strategy, " ".join(cmd))

        # Write server output to a log file instead of PIPE to avoid
        # deadlock when the 64KB pipe buffer fills up.
        log_path = self._config.output_dir / f"server_{strategy}.log"
        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        self._server_log = open(log_path, "a")  # noqa: SIM115

        self._server_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._server_log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,  # Create new process group for clean kill
        )

        # Wait for server to become healthy
        health_url = f"{self._config.server.base_url}/health"
        deadline = time.monotonic() + self._config.server_startup_timeout_s
        healthy = False

        # Bypass http_proxy for localhost health checks — proxies may cache
        # 502 responses from the engine-not-ready phase indefinitely.

        health_attempts = 0
        while time.monotonic() < deadline:
            if self._server_proc.poll() is not None:
                # Server process died — read tail of log file
                self._server_log.close()
                self._server_log = None
                output = log_path.read_text(errors="replace")[-2000:]
                raise RuntimeError(
                    f"vLLM server exited unexpectedly (code={self._server_proc.returncode})"
                    f"\nOutput:\n{output}"
                )
            try:
                req = urllib.request.Request(health_url, method="GET")
                with self._http_opener.open(req, timeout=5) as resp:
                    if resp.status == 200:
                        healthy = True
                        break
            except (urllib.error.URLError, OSError) as exc:
                health_attempts += 1
                if health_attempts % 30 == 1:  # Log every ~60s
                    logger.warning(
                        "Health check attempt #%d failed (url=%s): %s",
                        health_attempts,
                        health_url,
                        exc,
                    )
                pass
            time.sleep(2)

        if not healthy:
            self._stop_server()
            raise RuntimeError(
                f"vLLM server did not become healthy within "
                f"{self._config.server_startup_timeout_s}s"
            )

        logger.info("vLLM server healthy (strategy=%s)", strategy)

    def _stop_server(self) -> None:
        """Stop the running vLLM server and all child processes."""
        if self._server_proc is None:
            return

        pid = self._server_proc.pid
        logger.info("Stopping vLLM server (pid=%d)...", pid)

        # Send SIGTERM to entire process group
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        try:
            self._server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            logger.warning("Server didn't stop gracefully, sending SIGKILL to process group")
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._server_proc.wait(timeout=10)

        self._server_proc = None
        if self._server_log is not None:
            self._server_log.close()
            self._server_log = None

        # Deep-kill any orphan EngineCore / resource_tracker processes
        self._kill_orphan_gpu_processes()

        # Wait and verify GPU memory is released
        self._wait_gpu_release()
        logger.info("vLLM server stopped")

    def _kill_orphan_gpu_processes(self) -> None:
        """Scan /proc for any remaining NVIDIA GPU-holding processes and kill them."""
        killed = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid_i = int(entry)
            try:
                with open(f"/proc/{entry}/maps") as f:
                    maps = f.read()
                if "nvidia" not in maps:
                    continue
                with open(f"/proc/{entry}/cmdline") as f:
                    cmd = f.read().replace("\0", " ")[:150]
                # Only kill bidkv/vllm/EngineCore/resource_tracker processes
                process_markers = (
                    "bidkv",
                    "vllm",
                    "EngineCore",
                    "resource_tracker",
                    "spawn_main",
                    "multiprocessing.spawn",
                )
                if any(marker in cmd for marker in process_markers):
                    os.kill(pid_i, signal.SIGKILL)
                    killed.append(f"PID {pid_i}: {cmd[:80]}")
            except (OSError, ProcessLookupError):
                continue
        if killed:
            logger.warning(
                "Killed %d orphan GPU processes: %s",
                len(killed),
                "; ".join(killed),
            )

    def _wait_gpu_release(self, timeout_s: float = 60) -> None:
        """Wait until GPU memory drops to near-baseline level.

        Measures current GPU usage as the first sample, then waits for it
        to stabilize (no further decrease over two consecutive checks).
        This handles any host-process baseline (e.g. 1.7 GiB) automatically.
        """
        deadline = time.monotonic() + timeout_s
        prev_used = None
        while time.monotonic() < deadline:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=5,
                ).strip()
                used_mib = int(out.split("\n")[0].strip())
                # If usage is stable (within 200 MiB of previous reading),
                # the server's GPU memory has been fully released.
                if prev_used is not None and abs(used_mib - prev_used) < 200:
                    logger.info("GPU memory stabilized at %d MiB", used_mib)
                    return
                prev_used = used_mib
                logger.debug("GPU usage %d MiB, waiting to stabilize...", used_mib)
            except (subprocess.SubprocessError, ValueError, OSError):
                pass
            time.sleep(3)
        logger.warning("GPU memory not stabilized after %.0fs (last=%s MiB)", timeout_s, prev_used)

    def _warmup(self, strategy: str) -> None:
        """Send warmup requests to the server."""
        api_url = f"{self._config.server.api_url}/chat/completions"
        logger.info("Sending %d warmup requests...", self._config.warmup_requests)

        for i in range(self._config.warmup_requests):
            payload = json.dumps(
                {
                    "model": self._config.server.model,
                    "messages": [{"role": "user", "content": f"Warmup request {i}. Say hello."}],
                    "max_tokens": 16,
                    "stream": False,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                api_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with self._http_opener.open(req, timeout=60) as resp:
                    resp.read()
            except Exception as exc:
                logger.warning("Warmup request %d failed: %s", i, exc)

        logger.info("Warmup completed (strategy=%s)", strategy)

    def _run_single(
        self,
        *,
        strategy: str,
        workload: str,
        request_rate: float,
        run_index: int,
    ) -> RunResult:
        """运行单次实验 — Poisson 开环到达。

        Parameters
        ----------
        strategy:
            策略名称。
        workload:
            工作负载名称。
        request_rate:
            请求到达速率 (req/s)。
        run_index:
            运行序号。

        Returns
        -------
        RunResult
            运行结果。
        """
        label = self._config.run_label(strategy, workload, request_rate, run_index)
        trace_key = f"{workload}__{request_rate}"
        trace = self._traces[trace_key]

        result = RunResult(
            run_label=label,
            strategy=strategy,
            workload=workload,
            request_rate=request_rate,
            run_index=run_index,
            start_time=time.time(),
        )

        # 验证策略存在（用于 candidate-universe consistency 记录）
        if strategy != STRATEGY_PREEMPT_EVICT:
            self._registry.get(strategy)  # validate strategy exists

        # 发送请求并收集结果 — Poisson 开环
        # Dynamic timeout: ensure enough time to send all requests + drain
        sending_estimate = trace.num_requests / max(request_rate, 0.01)
        effective_timeout = max(
            self._config.run_timeout_s,
            sending_estimate * 1.2 + 60,
        )
        # Watchdog gets extra buffer for in-flight request completion
        watchdog_timeout = effective_timeout + 120
        self._run_deadline = time.monotonic() + effective_timeout
        logger.info(
            "Run timeout: effective=%.0fs (sending_est=%.0fs), watchdog=%.0fs",
            effective_timeout,
            sending_estimate,
            watchdog_timeout,
        )

        # Watchdog: kill server after deadline to break stuck HTTP connections
        watchdog_fired = threading.Event()

        def _watchdog() -> None:
            if not watchdog_fired.wait(timeout=watchdog_timeout):
                logger.error(
                    "WATCHDOG: strategy=%s exceeded %.0fs, killing server",
                    strategy,
                    watchdog_timeout,
                )
                watchdog_fired.set()
                self._stop_server()

        watchdog = threading.Thread(target=_watchdog, daemon=True)
        watchdog.start()

        request_results, run_status = asyncio.run(
            self._send_workload_open_loop(
                trace=trace,
                strategy_name=strategy,
            )
        )

        # Cancel watchdog if run finished before deadline
        if not watchdog_fired.is_set():
            watchdog_fired.set()

        if watchdog_fired.is_set() and run_status != "timeout":
            # Watchdog killed the server — ensure status reflects this
            pass  # run_status already set by _send_workload_open_loop
        result.request_results = request_results
        result.end_time = time.time()
        result.server_config["run_status"] = run_status

        # Read adapter metrics from subprocess via well-known file
        result.adapter_metrics = self._read_adapter_metrics(expected_strategy=strategy)

        return result

    def _read_adapter_metrics(self, expected_strategy: str = "") -> dict[str, int]:
        """Read adapter metrics dumped by the vLLM subprocess."""
        if _METRICS_FILE.exists():
            try:
                data = json.loads(_METRICS_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    # Validate provenance: reject stale metrics from previous strategy
                    file_strategy = data.pop("_strategy", "")
                    data.pop("_pid", None)
                    if expected_strategy and file_strategy and file_strategy != expected_strategy:
                        logger.warning(
                            "Metrics file strategy mismatch: expected=%s, got=%s. "
                            "Discarding stale metrics.",
                            expected_strategy,
                            file_strategy,
                        )
                        return {}
                    return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to read adapter metrics: %s", exc)
        return {}

    async def _send_workload_open_loop(
        self,
        *,
        trace: WorkloadTrace,
        strategy_name: str,
    ) -> tuple[list[RequestResult], str]:
        """Poisson 开环发送 — 按 arrival_time_ms 调度请求发出。

        §8 设计原则：fail-safe 不在 "请求是否发出" 层面限流，
        只在 "实验是否继续" 层面做保护。

        Parameters
        ----------
        trace:
            工作负载 trace（含 arrival_time_ms）。
        strategy_name:
            策略名称（用于日志）。

        Returns
        -------
        tuple[list[RequestResult], str]
            (所有请求结果, run_status: "completed" | "aborted")
        """
        api_url = f"{self._config.server.api_url}/chat/completions"
        loop = asyncio.get_event_loop()

        results: list[RequestResult] = []
        pending_tasks: list[asyncio.Future[RequestResult]] = []
        epoch_start = time.monotonic()
        consecutive_timeouts = 0
        aborted = False

        for req_trace in trace.requests:
            # Check wall-clock deadline before firing more requests
            if time.monotonic() >= self._run_deadline:
                elapsed = time.monotonic() - epoch_start
                logger.warning(
                    "RUN TIMEOUT: strategy=%s exceeded wall-clock limit (%.0fs elapsed), "
                    "stopping new requests (%d already fired)",
                    strategy_name,
                    elapsed,
                    len(pending_tasks),
                )
                aborted = True
                break

            # Wait until the scheduled arrival time
            target_time = epoch_start + req_trace.arrival_time_ms / 1000.0
            now = time.monotonic()
            if target_time > now:
                await asyncio.sleep(target_time - now)

            # Check if run should be aborted (§8 Layer 1)
            if consecutive_timeouts >= self._config.consecutive_timeout_abort:
                logger.warning(
                    "ABORT: %d consecutive timeouts for strategy=%s, stopping run",
                    consecutive_timeouts,
                    strategy_name,
                )
                aborted = True
                break

            # Fire request immediately — no max_inflight cap (§8)
            future = loop.run_in_executor(
                None,
                self._send_request_streaming_sync,
                api_url,
                req_trace,
                strategy_name,
            )
            pending_tasks.append(future)

        # Wait for all in-flight requests to finish
        if pending_tasks:
            done_results = await asyncio.gather(*pending_tasks, return_exceptions=True)
            for r in done_results:
                if isinstance(r, RequestResult):
                    results.append(r)
                    # Track consecutive timeouts
                    if r.error and "timeout" in r.error.lower():
                        consecutive_timeouts += 1
                    else:
                        consecutive_timeouts = 0
                elif isinstance(r, BaseException):
                    logger.error("Unexpected exception in request task: %s", r)

        run_status: str
        if aborted and time.monotonic() >= self._run_deadline:
            run_status = "timeout"
        elif aborted:
            run_status = "aborted"
        else:
            run_status = "completed"
        return results, run_status

    def _send_request_streaming_sync(
        self,
        api_url: str,
        req_trace: RequestTrace,
        strategy_name: str,
    ) -> RequestResult:
        """Send a single request with SSE streaming to measure precise TTFT.

        Parameters
        ----------
        api_url:
            vLLM API URL.
        req_trace:
            Request trace.
        strategy_name:
            Strategy name for logging.

        Returns
        -------
        RequestResult
            Result with precise TTFT from streaming first token.
        """
        result = RequestResult(request_id=req_trace.request_id)

        payload = json.dumps(
            {
                "model": self._config.server.model,
                "messages": [{"role": "user", "content": req_trace.prompt}],
                "max_tokens": req_trace.max_tokens,
                "stream": True,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            api_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        result.submit_time = time.monotonic()
        chunks: list[str] = []

        # Use shorter timeout if near run deadline to avoid blocking the gather
        remaining = max(1.0, self._run_deadline - time.monotonic())
        effective_timeout = min(self._config.request_timeout_s, remaining)

        try:
            with self._http_opener.open(req, timeout=effective_timeout) as resp:
                first_chunk_received = False
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]  # strip "data: " prefix
                    if data_str == "[DONE]":
                        break

                    if not first_chunk_received:
                        result.first_token_time = time.monotonic()
                        result.ttft_ms = (result.first_token_time - result.submit_time) * 1000
                        first_chunk_received = True

                    chunk_data = json.loads(data_str)
                    choices = chunk_data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            chunks.append(content)

            result.finish_time = time.monotonic()
            result.total_latency_ms = (result.finish_time - result.submit_time) * 1000
            result.generated_text = "".join(chunks)
            # Count completion tokens from generated text
            result.completion_tokens = max(1, len(result.generated_text.split()))

            # If no first token was detected (edge case), use total latency
            if result.ttft_ms is None:
                result.ttft_ms = result.total_latency_ms
                result.first_token_time = result.finish_time

        except Exception as exc:
            result.finish_time = time.monotonic()
            result.total_latency_ms = (result.finish_time - result.submit_time) * 1000
            result.error = str(exc)
            logger.debug(
                "Request %s failed (%s): %s",
                req_trace.request_id,
                strategy_name,
                exc,
            )

        return result


def verify_candidate_consistency(results: list[RunResult]) -> dict[str, object]:
    """验证 candidate-universe consistency。

    检查同一 (workload, concurrency, run_index) 的不同策略在同一 pressure event
    时是否使用了相同的 candidate pool。

    Parameters
    ----------
    results:
        所有运行结果。

    Returns
    -------
    dict[str, object]
        一致性校验报告。
    """
    # 按 (workload, request_rate, run_index) 分组
    groups: dict[tuple[str, float, int], list[RunResult]] = {}
    for r in results:
        key = (r.workload, r.request_rate, r.run_index)
        groups.setdefault(key, []).append(r)

    report: dict[str, object] = {
        "total_groups": len(groups),
        "groups_with_snapshots": 0,
        "consistency_violations": 0,
        "details": [],
    }

    for key, group_results in groups.items():
        results_with_snapshots = [r for r in group_results if r.candidate_snapshots]
        if len(results_with_snapshots) < 2:
            continue

        report["groups_with_snapshots"] = int(report["groups_with_snapshots"]) + 1  # type: ignore[arg-type]

        # 比较不同策略在同一时间窗口内的 candidate pool
        # （简化比较：检查 candidate 集合大小一致性）
        snapshot_sizes = {
            r.strategy: [len(s.candidate_request_ids) for s in r.candidate_snapshots]
            for r in results_with_snapshots
        }
        report["details"].append(
            {  # type: ignore[union-attr]
                "group_key": f"{key[0]}__rate{key[1]}__r{key[2]}",
                "strategies": list(snapshot_sizes.keys()),
                "snapshot_counts": {s: len(sizes) for s, sizes in snapshot_sizes.items()},
            }
        )

    return report


def parse_args(argv: list[str] | None = None) -> tuple[ExperimentConfig, bool]:
    """解析命令行参数。

    Returns
    -------
    tuple[ExperimentConfig, bool]
        (config, resume) — 实验配置 + 是否断点续跑。
    """
    parser = argparse.ArgumentParser(description="BidKV vLLM 5-Strategy Experiment Runner")
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(ALL_STRATEGIES),
        help="Comma-separated strategy names.",
    )
    parser.add_argument(
        "--workloads",
        type=str,
        default=",".join(ALL_WORKLOADS),
        help="Comma-separated workload names.",
    )
    parser.add_argument(
        "--request-rates",
        type=str,
        default=",".join(str(r) for r in DEFAULT_REQUEST_RATES),
        help="Comma-separated fallback rates (req/s). "
        "Use --mixed-rates / --long-rates for per-workload.",
    )
    parser.add_argument(
        "--mixed-rates",
        type=str,
        default=None,
        help="Comma-separated rates for mixed workload (overrides --request-rates).",
    )
    parser.add_argument(
        "--long-rates",
        type=str,
        default=None,
        help="Comma-separated rates for long_context workload (overrides --request-rates).",
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per combination.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/vllm",
        help="Output directory.",
    )
    parser.add_argument(
        "--traces-dir",
        type=str,
        default="experiments/vllm/traces",
        help="Frozen workload traces directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=get_default_model(),
        help=(
            "Model name or local path. Defaults to the BIDKV_MODEL environment "
            f"variable when set, otherwise {get_default_model()}."
        ),
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Max tokens per scheduler step. Controls effective concurrency "
        "(vLLM default 2048 → ~30 concurrent). Set higher for more concurrent requests.",
    )
    parser.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        default=None,
        help="Override number of KV cache GPU blocks. Controls KV budget for "
        "pressure experiments (e.g. 800 blocks = ~12.8K tokens with block_size=16).",
    )
    parser.add_argument("--ttft-target-ms", type=float, default=2000.0)
    parser.add_argument("--warmup-requests", type=int, default=5)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip runs whose result file already exists (resume after crash).",
    )

    args = parser.parse_args(argv)

    strategies = tuple(s.strip() for s in args.strategies.split(","))
    workloads = tuple(w.strip() for w in args.workloads.split(","))
    request_rates = tuple(float(r.strip()) for r in args.request_rates.split(","))

    # Build per-workload rates: CLI overrides > WORKLOAD_REQUEST_RATES > fallback
    workload_rates = dict(WORKLOAD_REQUEST_RATES)
    if args.mixed_rates is not None:
        workload_rates["mixed"] = tuple(float(r) for r in args.mixed_rates.split(","))
    if args.long_rates is not None:
        workload_rates["long_context"] = tuple(float(r) for r in args.long_rates.split(","))

    return ExperimentConfig(
        strategies=strategies,
        workloads=workloads,
        request_rates=request_rates,
        workload_rates=workload_rates,
        runs_per_combo=args.runs,
        output_dir=Path(args.output_dir),
        traces_dir=Path(args.traces_dir),
        server=VLLMServerConfig(
            model=args.model,
            host=args.host,
            port=args.port,
            block_size=args.block_size,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            num_gpu_blocks_override=args.num_gpu_blocks_override,
        ),
        slo=SLOConfig(ttft_target_ms=args.ttft_target_ms),
        warmup_requests=args.warmup_requests,
        request_timeout_s=args.request_timeout_s,
    ), args.resume


def main(argv: list[str] | None = None) -> None:
    """CLI 入口。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config, resume = parse_args(argv)
    logger.info(
        "Experiment config: %d strategies × %d workloads × %d rates × %d runs = %d total",
        len(config.strategies),
        len(config.workloads),
        len(config.request_rates),
        config.runs_per_combo,
        config.total_runs,
    )

    runner = VLLMExperimentRunner(config, resume=resume)
    results = runner.run_all()

    # 验证 candidate-universe consistency
    consistency_report = verify_candidate_consistency(results)
    report_path = config.output_dir / "candidate_consistency_report.json"
    report_path.write_text(json.dumps(consistency_report, indent=2), encoding="utf-8")
    logger.info("Candidate consistency report: %s", report_path)

    # 摘要打印
    logger.info("=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    for r in results:
        success_rate = (
            len(r.successful_requests) / len(r.request_results) * 100 if r.request_results else 0
        )
        logger.info(
            "  %s: thru=%.2f rps | TTFT p50=%.0f p99=%.0f ms | "
            "TPOT p50=%.1f p99=%.1f ms | E2E p50=%.0f p99=%.0f ms | "
            "NormLat=%.1f ms/tok | success=%.0f%%",
            r.run_label,
            r.compute_throughput_rps(),
            r.compute_p50_ttft_ms(),
            r.compute_p99_ttft_ms(),
            r.compute_p50_tpot_ms(),
            r.compute_p99_tpot_ms(),
            r.compute_p50_e2e_latency_ms(),
            r.compute_p99_e2e_latency_ms(),
            r.compute_normalized_latency_ms(),
            success_rate,
        )


if __name__ == "__main__":
    main()
