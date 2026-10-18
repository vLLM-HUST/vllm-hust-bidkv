"""Experiment configuration for vLLM baseline experiment.

所有实验参数集中管理，确保 reproducibility。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from bidkv.experiments.common.model import get_default_model

# 策略名称常量 — 与 BaselineRegistry.name 对齐
STRATEGY_PREEMPT_EVICT = "preempt-evict"
STRATEGY_STATIC_RANDOM = "static-random"
STRATEGY_LARGEST_FIRST = "largest-first"
STRATEGY_PREEMPT_EVICT_SJF = "preempt-evict-sjf"
STRATEGY_BIDKV = "bidkv"

# Legacy name mapping: frozen result files use "h2o-style", code now uses "largest-first"
STRATEGY_LEGACY_NAMES: dict[str, str] = {
    "h2o-style": "largest-first",
}

ALL_STRATEGIES: tuple[str, ...] = (
    STRATEGY_PREEMPT_EVICT,
    STRATEGY_STATIC_RANDOM,
    STRATEGY_LARGEST_FIRST,
    STRATEGY_PREEMPT_EVICT_SJF,
    STRATEGY_BIDKV,
)

# 工作负载常量 — 按 §4 设计：Mixed + Long-context
WORKLOAD_MIXED = "mixed"
WORKLOAD_LONG_CONTEXT = "long_context"

ALL_WORKLOADS: tuple[str, ...] = (
    WORKLOAD_MIXED,
    WORKLOAD_LONG_CONTEXT,
)

# ⚠️ FROZEN — RULE RATE-FREEZE: 校准后冻结，不可基于策略表现调整
# 每个 workload 独立的冻结 rate 值（req/s）
# Calibration 依据（Issue #055）：
#   mixed:        tput 2.06→3.32→3.84(饱和), p99ttft 356→450→440ms
#   long_context: tput 0.50→0.63→0.67(饱和), p99ttft 3.2k→4.9k→10kms
WORKLOAD_REQUEST_RATES: dict[str, tuple[float, ...]] = {
    WORKLOAD_MIXED: (2.0, 3.8, 5.7),
    WORKLOAD_LONG_CONTEXT: (0.35, 0.5, 0.7),
}

# 向后兼容 fallback（不建议使用，优先用 WORKLOAD_REQUEST_RATES）
DEFAULT_REQUEST_RATES: tuple[float, ...] = (2.0, 3.8, 5.7)
DEFAULT_RUNS_PER_COMBO = 3

# 每个 workload 的默认请求数 — §4
WORKLOAD_NUM_REQUESTS: dict[str, int] = {
    WORKLOAD_MIXED: 1000,
    WORKLOAD_LONG_CONTEXT: 500,
}


@dataclass(frozen=True)
class VLLMServerConfig:
    """vLLM 服务端配置。

    Attributes
    ----------
    model:
        HuggingFace 模型名称或路径。
    block_size:
        KV cache block size（token 数/block）。
    max_num_seqs:
        最大并发请求数。
    gpu_memory_utilization:
        GPU 显存利用率上限。
    enforce_eager:
        是否禁用 CUDA graph（便于 positional scoring hook 工作）。
    host:
        服务监听地址。
    port:
        服务监听端口。
    """

    model: str = field(default_factory=get_default_model)
    block_size: int = 16
    max_num_seqs: int = 32
    gpu_memory_utilization: float = 0.85
    enforce_eager: bool = True
    disable_frontend_multiprocessing: bool = True
    max_model_len: int = 8192
    max_num_batched_tokens: int | None = None
    num_gpu_blocks_override: int | None = None
    enable_prefix_caching: bool = False  # Disabled: truncation frees blocks
    # without cleaning prefix cache index, causing stale hits → CUDA crash.
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/v1"

    def to_cli_args(self) -> list[str]:
        """生成 vllm serve 的命令行参数。"""
        args = [
            "--model",
            self.model,
            "--block-size",
            str(self.block_size),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.enforce_eager:
            args.append("--enforce-eager")
        if self.disable_frontend_multiprocessing:
            args.append("--disable-frontend-multiprocessing")
        if self.max_model_len:
            args.extend(["--max-model-len", str(self.max_model_len)])
        if self.max_num_batched_tokens is not None:
            args.extend(["--max-num-batched-tokens", str(self.max_num_batched_tokens)])
        if self.num_gpu_blocks_override is not None:
            args.extend(["--num-gpu-blocks-override", str(self.num_gpu_blocks_override)])
        if not self.enable_prefix_caching:
            args.append("--no-enable-prefix-caching")
        return args


@dataclass(frozen=True)
class SLOConfig:
    """SLO 配置。

    Attributes
    ----------
    ttft_target_ms:
        Time-To-First-Token SLO 目标（ms）。
    tpot_target_ms:
        Time-Per-Output-Token SLO 目标（ms）。
    """

    ttft_target_ms: float = 2000.0
    tpot_target_ms: float = 100.0


@dataclass(frozen=True)
class ExperimentConfig:
    """完整实验配置。

    Attributes
    ----------
    strategies:
        要运行的策略名称列表。
    workloads:
        要运行的工作负载列表。
    request_rates:
        请求到达速率列表 (req/s)，Phase 2 pilot 后冻结。
    runs_per_combo:
        每个 (策略, 工作负载, rate) 组合的独立运行次数。
    output_dir:
        结果输出目录。
    server:
        vLLM 服务端配置。
    slo:
        SLO 配置。
    traces_dir:
        冻结的工作负载 trace 目录。
    warmup_requests:
        预热请求数（不计入统计）。
    request_timeout_s:
        单个请求超时时间（秒）。
    server_startup_timeout_s:
        vLLM 服务启动超时（秒）。
    collect_candidate_snapshots:
        是否收集每个 pressure event 的 candidate pool snapshot
        （用于 candidate-universe consistency 验证）。
    consecutive_timeout_abort:
        连续 timeout 请求数达到此值时提前终止 run。
    run_timeout_s:
        单次 run 整体挂钟超时（秒）。超时后取消所有 in-flight 请求，
        run_status 标记为 "timeout"。默认 600（10 分钟）。
    """

    strategies: tuple[str, ...] = ALL_STRATEGIES
    workloads: tuple[str, ...] = ALL_WORKLOADS
    request_rates: tuple[float, ...] = DEFAULT_REQUEST_RATES
    workload_rates: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: dict(WORKLOAD_REQUEST_RATES),
    )
    runs_per_combo: int = DEFAULT_RUNS_PER_COMBO
    output_dir: Path = Path("results/vllm")
    server: VLLMServerConfig = field(default_factory=VLLMServerConfig)
    slo: SLOConfig = field(default_factory=SLOConfig)
    traces_dir: Path = Path("experiments/vllm/traces")
    warmup_requests: int = 5
    request_timeout_s: float = 120.0
    server_startup_timeout_s: float = float(os.environ.get("BIDKV_SERVER_STARTUP_TIMEOUT", "300"))
    collect_candidate_snapshots: bool = True
    consecutive_timeout_abort: int = 10
    run_timeout_s: float = 600.0

    def __post_init__(self) -> None:
        unknown = set(self.strategies) - set(ALL_STRATEGIES)
        if unknown:
            raise ValueError(f"Unknown strategies: {unknown}. Valid: {ALL_STRATEGIES}")
        unknown_wl = set(self.workloads) - set(ALL_WORKLOADS)
        if unknown_wl:
            raise ValueError(f"Unknown workloads: {unknown_wl}. Valid: {ALL_WORKLOADS}")
        if self.runs_per_combo < 1:
            raise ValueError(f"runs_per_combo must be >= 1, got {self.runs_per_combo}")

    def get_rates_for_workload(self, workload: str) -> tuple[float, ...]:
        """返回指定 workload 的冻结 rate 列表。"""
        return self.workload_rates.get(workload, self.request_rates)

    @property
    def total_runs(self) -> int:
        """总实验运行次数。"""
        total = 0
        for workload in self.workloads:
            rates = self.get_rates_for_workload(workload)
            total += len(self.strategies) * len(rates) * self.runs_per_combo
        return total

    def run_label(self, strategy: str, workload: str, request_rate: float, run_idx: int) -> str:
        """生成单次实验运行的标识符。"""
        return f"{strategy}__{workload}__rate{request_rate}__r{run_idx}"
