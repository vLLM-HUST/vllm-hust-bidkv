"""SGLang 可移植性实验配置 — 与 Issue-047 vLLM 实验口径对齐。

所有参数集中管理。与 vLLM 共享的参数（模型、SLO、rate 点）
直接复用 Issue-047 已冻结值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from bidkv.experiments.common.model import get_default_model

# ── 策略（v2.3 冻结版本，3 策略核心）────────────────────────────────
STRATEGY_SGLANG_DEFAULT = "sglang_default"  # SGLang native (= Preempt-Evict)
STRATEGY_RANDOM_EVICT = "static-random"  # Random victim selection (Random-Evict in paper)
STRATEGY_BIDKV = "bidkv"  # BidKV 完整 bid pipeline

# 扩展策略（验证性实验使用，非 v2.3 冻结 54-run 计划范围）
STRATEGY_PREEMPT_EVICT_SJF = "preempt-evict-sjf"  # SJF admission + LIFO eviction 消融

# v2.3 冻结正式策略（54-run 计划使用）
# 与论文 Table 4 一致：Vanilla SGLang / Random-Evict / BidKV
FROZEN_STRATEGIES: tuple[str, ...] = (
    STRATEGY_SGLANG_DEFAULT,
    STRATEGY_RANDOM_EVICT,
    STRATEGY_BIDKV,
)

# ALL_STRATEGIES = 冻结的正式评估策略（与论文 Table 4 一致：Vanilla SGLang / Random-Evict / BidKV）
ALL_STRATEGIES: tuple[str, ...] = FROZEN_STRATEGIES

# EXTENDED_STRATEGIES = 所有可运行策略（用于 __post_init__ 校验）
EXTENDED_STRATEGIES: tuple[str, ...] = (
    STRATEGY_SGLANG_DEFAULT,
    STRATEGY_RANDOM_EVICT,
    STRATEGY_BIDKV,
    STRATEGY_PREEMPT_EVICT_SJF,
)

# SGLang 策略名 → BaselineRegistry 内部名映射
STRATEGY_BASELINE_MAP: dict[str, str] = {
    STRATEGY_SGLANG_DEFAULT: "preempt-evict",
    STRATEGY_RANDOM_EVICT: "static-random",
    STRATEGY_BIDKV: "bidkv",
    STRATEGY_PREEMPT_EVICT_SJF: "preempt-evict-sjf",
}

# ── 工作负载 ──────────────────────────────────────────────────────
# 正式名称与 Issue-047 完全一致
WORKLOAD_MIXED = "mixed"
WORKLOAD_LONG_CONTEXT = "long_context"

ALL_WORKLOADS: tuple[str, ...] = (
    WORKLOAD_MIXED,
    WORKLOAD_LONG_CONTEXT,
)

# ── 默认 rate 点（与 vLLM 共享冻结 rates）──────────────────────────
# ⚠️ FROZEN — RULE RATE-FREEZE: 校准后冻结，不可基于策略表现调整
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


@dataclass(frozen=True)
class SLOConfig:
    """SLO 配置 — 与 Issue-047 完全一致。

    值来源：bidkv.experiments.vllm.config.SLOConfig 默认值。
    """

    ttft_target_ms: float = 2000.0
    tpot_target_ms: float = 100.0


@dataclass(frozen=True)
class SGLangServerConfig:
    """SGLang serving server 配置。

    Attributes
    ----------
    model:
        模型名称或本地路径。
    mem_fraction_static:
        GPU 内存分配比例（对应 --mem-fraction-static）。
    max_total_tokens:
        最大 KV 容量（对应 --max-total-tokens）。
    host:
        监听地址。
    port:
        监听端口。
    """

    model: str = field(default_factory=get_default_model)
    mem_fraction_static: float = 0.85
    max_total_tokens: int = 16384
    host: str = "127.0.0.1"
    port: int = 30000
    disable_cuda_graph: bool = False
    # Use triton backend to avoid flashinfer/sgl_kernel .so incompatibility with torch 2.6.0.
    # triton was the attention backend used in all previous sglang experiments (v1/v2/v3).
    attention_backend: str = "triton"

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/v1"

    def to_cli_args(self) -> list[str]:
        """生成 sglang serve 命令行参数。"""
        args = [
            "--model",
            self.model,
            "--mem-fraction-static",
            str(self.mem_fraction_static),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.max_total_tokens > 0:
            args += ["--max-total-tokens", str(self.max_total_tokens)]
        if self.disable_cuda_graph:
            args += ["--disable-cuda-graph"]
        if self.attention_backend is not None:
            args += ["--attention-backend", self.attention_backend]
        return args


@dataclass(frozen=True)
class SGLangExperimentConfig:
    """SGLang 可移植性实验完整配置。

    所有与 Issue-047 共享的参数直接复用其冻结值。
    """

    strategies: tuple[str, ...] = ALL_STRATEGIES
    workloads: tuple[str, ...] = ALL_WORKLOADS
    request_rates: tuple[float, ...] = DEFAULT_REQUEST_RATES
    workload_rates: dict[str, tuple[float, ...]] = field(
        default_factory=lambda: dict(WORKLOAD_REQUEST_RATES),
    )
    runs_per_combo: int = DEFAULT_RUNS_PER_COMBO
    output_dir: Path = Path("results/sglang")
    server: SGLangServerConfig = field(default_factory=SGLangServerConfig)
    slo: SLOConfig = field(default_factory=SLOConfig)
    traces_dir: Path = Path("experiments/vllm/traces")
    warmup_requests: int = 5
    request_timeout_s: float = 120.0
    server_startup_timeout_s: float = 300.0
    consecutive_timeout_abort: int = 10

    def get_rates_for_workload(self, workload: str) -> tuple[float, ...]:
        """返回指定 workload 的冻结 rate 列表。"""
        return self.workload_rates.get(workload, self.request_rates)

    @property
    def total_runs(self) -> int:
        total = 0
        for workload in self.workloads:
            rates = self.get_rates_for_workload(workload)
            total += len(self.strategies) * len(rates) * self.runs_per_combo
        return total

    def run_label(
        self,
        strategy: str,
        workload: str,
        rate: float,
        run_index: int,
    ) -> str:
        return f"sglang__{strategy}__{workload}__rate{rate}__run{run_index}"

    def __post_init__(self) -> None:
        unknown = set(self.strategies) - set(EXTENDED_STRATEGIES)
        if unknown:
            raise ValueError(
                f"Unknown strategies: {unknown}. Valid strategies: {EXTENDED_STRATEGIES}"
            )
