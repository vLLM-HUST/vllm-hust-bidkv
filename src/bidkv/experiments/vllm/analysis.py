"""Result analysis and paper figure generation for vLLM experiments.

生成论文 §6 需要的 Figure + Table 数据：

Figure 1 [Cat A]: SLO Attainment Rate vs KV 压力
Figure 2 [Cat A]: P99 TTFT vs 吞吐量（Pareto 前沿）
Figure 5 [Cat A]: 覆盖率与降级频率
Figure 6 [Cat B]: Δ budget 敏感性

依赖 matplotlib（可选）— 纯数据汇总不需要 matplotlib。
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from bidkv.experiments.vllm.collector import RunResult, load_all_run_results
from bidkv.experiments.vllm.config import (
    SLOConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyAggregation:
    """单个策略在特定 (workload, request_rate) 下的聚合指标。

    多次运行的均值和置信区间。业界标准指标体系
    （参照 vLLM SOSP'23, DistServe OSDI'24, SGLang, SARATHI-Serve）。
    """

    strategy: str
    workload: str
    request_rate: float
    runs: int = 0
    # -- 业界标准指标 --
    throughput_rps_mean: float = 0.0
    throughput_rps_ci95: float = 0.0
    p50_ttft_ms_mean: float = 0.0
    p50_ttft_ms_ci95: float = 0.0
    p99_ttft_ms_mean: float = 0.0
    p99_ttft_ms_ci95: float = 0.0
    p50_tpot_ms_mean: float = 0.0
    p50_tpot_ms_ci95: float = 0.0
    p99_tpot_ms_mean: float = 0.0
    p99_tpot_ms_ci95: float = 0.0
    p50_e2e_latency_ms_mean: float = 0.0
    p50_e2e_latency_ms_ci95: float = 0.0
    p99_e2e_latency_ms_mean: float = 0.0
    p99_e2e_latency_ms_ci95: float = 0.0
    normalized_latency_mean: float = 0.0
    normalized_latency_ci95: float = 0.0
    # -- SLO attainment（可选辅助指标）--
    slo_attainment_rate_mean: float = 0.0
    slo_attainment_rate_ci95: float = 0.0
    # -- 驱逐辅助指标 --
    eviction_coverage_mean: float = 0.0
    total_evictions_mean: float = 0.0
    total_tokens_freed_mean: float = 0.0
    pressure_events_mean: float = 0.0
    # -- 全量 preemption 指标（含 native LIFO，用于 Figure 3）--
    total_all_preemptions_mean: float = 0.0
    total_all_tokens_freed_mean: float = 0.0


def compute_ci95(values: list[float]) -> tuple[float, float]:
    """计算均值 ± 95% CI。

    使用 t 分布近似（n < 30 时 t_{n-1, 0.975}）。

    Parameters
    ----------
    values:
        数据点列表。

    Returns
    -------
    tuple[float, float]
        (mean, half_width_of_95_CI)
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (values[0], 0.0)

    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)

    # t 分布临界值（近似，n=3 → t=4.303, n=5 → t=2.776, n=10 → t=2.262）
    # 使用保守的查表值
    t_values = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}
    t_crit = t_values.get(n, 1.96)  # n >= 30 时近似 1.96

    ci_half = t_crit * std / math.sqrt(n)
    return (mean, ci_half)


def aggregate_results(
    results: list[RunResult],
    slo_config: SLOConfig | None = None,
) -> list[StrategyAggregation]:
    """将多次运行结果聚合为策略级统计。

    Parameters
    ----------
    results:
        所有运行结果。
    slo_config:
        SLO 配置（用于计算 violation rate）。

    Returns
    -------
    list[StrategyAggregation]
        按策略聚合的统计列表。
    """
    slo = slo_config or SLOConfig()

    # 按 (strategy, workload, request_rate) 分组
    groups: dict[tuple[str, str, float], list[RunResult]] = {}
    for r in results:
        key = (r.strategy, r.workload, r.request_rate)
        groups.setdefault(key, []).append(r)

    aggregations: list[StrategyAggregation] = []
    for (strategy, workload, request_rate), group in sorted(groups.items()):
        throughputs = [r.compute_throughput_rps() for r in group]
        p50_ttfts = [r.compute_p50_ttft_ms() for r in group]
        p99_ttfts = [r.compute_p99_ttft_ms() for r in group]
        p50_tpots = [r.compute_p50_tpot_ms() for r in group]
        p99_tpots = [r.compute_p99_tpot_ms() for r in group]
        p50_e2es = [r.compute_p50_e2e_latency_ms() for r in group]
        p99_e2es = [r.compute_p99_e2e_latency_ms() for r in group]
        norm_lats = [r.compute_normalized_latency_ms() for r in group]
        slo_attainments = [
            r.compute_slo_attainment_rate(slo.ttft_target_ms, slo.tpot_target_ms) for r in group
        ]

        # adapter metrics
        def _get_evictions(r: RunResult) -> float:
            am = r.adapter_metrics
            return float(am.get("total_evictions", am.get("total_compressions", 0)))

        total_evicts = [_get_evictions(r) for r in group]
        total_freed = [float(r.adapter_metrics.get("total_tokens_freed", 0)) for r in group]
        pressure_events = [float(r.adapter_metrics.get("total_pressure_events", 0)) for r in group]
        # 全量 preemption（含 native LIFO）——新字段，旧数据不存在则回落到 total_evictions
        total_all_preempts = [
            float(r.adapter_metrics.get("total_all_preemptions", _get_evictions(r)))
            for r in group
        ]
        total_all_freed = [
            float(r.adapter_metrics.get("total_all_tokens_freed",
                                        r.adapter_metrics.get("total_tokens_freed", 0)))
            for r in group
        ]

        tp_mean, tp_ci = compute_ci95(throughputs)
        p50_ttft_mean, p50_ttft_ci = compute_ci95(p50_ttfts)
        p99_ttft_mean, p99_ttft_ci = compute_ci95(p99_ttfts)
        p50_tpot_mean, p50_tpot_ci = compute_ci95(p50_tpots)
        p99_tpot_mean, p99_tpot_ci = compute_ci95(p99_tpots)
        p50_e2e_mean, p50_e2e_ci = compute_ci95(p50_e2es)
        p99_e2e_mean, p99_e2e_ci = compute_ci95(p99_e2es)
        norm_lat_mean, norm_lat_ci = compute_ci95(norm_lats)
        slo_mean, slo_ci = compute_ci95(slo_attainments)

        # Eviction coverage = requests with eviction / total requests
        coverages = []
        for r in group:
            total_reqs = len(r.successful_requests)
            # 简化：从 adapter metrics 推断
            evicts = _get_evictions(r)
            cov = evicts / total_reqs if total_reqs > 0 else 0.0
            coverages.append(min(1.0, cov))

        cov_mean, _ = compute_ci95(coverages)
        comp_mean, _ = compute_ci95(total_evicts)
        freed_mean, _ = compute_ci95(total_freed)
        pe_mean, _ = compute_ci95(pressure_events)
        all_preempts_mean, _ = compute_ci95(total_all_preempts)
        all_freed_mean, _ = compute_ci95(total_all_freed)

        aggregations.append(
            StrategyAggregation(
                strategy=strategy,
                workload=workload,
                request_rate=request_rate,
                runs=len(group),
                throughput_rps_mean=tp_mean,
                throughput_rps_ci95=tp_ci,
                p50_ttft_ms_mean=p50_ttft_mean,
                p50_ttft_ms_ci95=p50_ttft_ci,
                p99_ttft_ms_mean=p99_ttft_mean,
                p99_ttft_ms_ci95=p99_ttft_ci,
                p50_tpot_ms_mean=p50_tpot_mean,
                p50_tpot_ms_ci95=p50_tpot_ci,
                p99_tpot_ms_mean=p99_tpot_mean,
                p99_tpot_ms_ci95=p99_tpot_ci,
                p50_e2e_latency_ms_mean=p50_e2e_mean,
                p50_e2e_latency_ms_ci95=p50_e2e_ci,
                p99_e2e_latency_ms_mean=p99_e2e_mean,
                p99_e2e_latency_ms_ci95=p99_e2e_ci,
                normalized_latency_mean=norm_lat_mean,
                normalized_latency_ci95=norm_lat_ci,
                slo_attainment_rate_mean=slo_mean,
                slo_attainment_rate_ci95=slo_ci,
                eviction_coverage_mean=cov_mean,
                total_evictions_mean=comp_mean,
                total_tokens_freed_mean=freed_mean,
                pressure_events_mean=pe_mean,
                total_all_preemptions_mean=all_preempts_mean,
                total_all_tokens_freed_mean=all_freed_mean,
            )
        )

    return aggregations


def export_summary_json(
    aggregations: list[StrategyAggregation],
    output_dir: Path,
) -> Path:
    """导出聚合摘要为 JSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"

    data = {
        "aggregations": [
            {
                "strategy": a.strategy,
                "workload": a.workload,
                "request_rate": a.request_rate,
                "runs": a.runs,
                "throughput_rps": {
                    "mean": a.throughput_rps_mean,
                    "ci95": a.throughput_rps_ci95,
                },
                "ttft_ms_p50": {
                    "mean": a.p50_ttft_ms_mean,
                    "ci95": a.p50_ttft_ms_ci95,
                },
                "ttft_ms_p99": {
                    "mean": a.p99_ttft_ms_mean,
                    "ci95": a.p99_ttft_ms_ci95,
                },
                "tpot_ms_p50": {
                    "mean": a.p50_tpot_ms_mean,
                    "ci95": a.p50_tpot_ms_ci95,
                },
                "tpot_ms_p99": {
                    "mean": a.p99_tpot_ms_mean,
                    "ci95": a.p99_tpot_ms_ci95,
                },
                "e2e_latency_ms_p50": {
                    "mean": a.p50_e2e_latency_ms_mean,
                    "ci95": a.p50_e2e_latency_ms_ci95,
                },
                "e2e_latency_ms_p99": {
                    "mean": a.p99_e2e_latency_ms_mean,
                    "ci95": a.p99_e2e_latency_ms_ci95,
                },
                "normalized_latency_ms_per_token": {
                    "mean": a.normalized_latency_mean,
                    "ci95": a.normalized_latency_ci95,
                },
                "slo_attainment_rate": {
                    "mean": a.slo_attainment_rate_mean,
                    "ci95": a.slo_attainment_rate_ci95,
                },
                "eviction_coverage": a.eviction_coverage_mean,
                "total_evictions": a.total_evictions_mean,
                "total_tokens_freed": a.total_tokens_freed_mean,
                "pressure_events": a.pressure_events_mean,
                "total_all_preemptions": a.total_all_preemptions_mean,
                "total_all_tokens_freed": a.total_all_tokens_freed_mean,
            }
            for a in aggregations
        ],
    }

    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Exported summary to %s", path)
    return path


def generate_figures(
    aggregations: list[StrategyAggregation],
    output_dir: Path,
) -> list[Path]:
    """生成论文 Figure PDF 文件。

    需要 matplotlib。如果未安装，仅导出数据不生成图片。

    Parameters
    ----------
    aggregations:
        聚合统计。
    output_dir:
        图片输出目录。

    Returns
    -------
    list[Path]
        生成的 PDF 文件路径列表。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # 无头模式
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        logger.warning(
            "matplotlib not installed. Exporting data only, skipping figure generation. "
            "Install with: pip install matplotlib"
        )
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    generated: list[Path] = []

    # Figure 1 [Cat A]: SLO Attainment Rate — 按策略分组柱状图
    generated.extend(_plot_slo_attainment_bar(aggregations, figures_dir))

    # Figure 2 [Cat A]: P99 TTFT vs Throughput — Pareto 散点图
    generated.extend(_plot_pareto_front(aggregations, figures_dir))

    # Figure 5 [Cat A]: Eviction coverage
    generated.extend(_plot_eviction_coverage(aggregations, figures_dir))

    logger.info("Generated %d figures in %s", len(generated), figures_dir)
    return generated


def _plot_slo_attainment_bar(
    aggregations: list[StrategyAggregation],
    output_dir: Path,
) -> list[Path]:
    """Figure 1: SLO Attainment Rate bar chart."""
    import matplotlib.pyplot as plt

    paths: list[Path] = []

    # 按 workload 分别画图
    workloads = sorted({a.workload for a in aggregations})
    for workload in workloads:
        fig, ax = plt.subplots(figsize=(12, 6))
        wl_data = [a for a in aggregations if a.workload == workload]

        strategies = sorted({a.strategy for a in wl_data})
        rates = sorted({a.request_rate for a in wl_data})

        x_positions = range(len(rates))
        bar_width = 0.8 / max(1, len(strategies))

        for i, strategy in enumerate(strategies):
            means = []
            errors = []
            for rate in rates:
                match = [a for a in wl_data if a.strategy == strategy and a.request_rate == rate]
                if match:
                    means.append(match[0].slo_attainment_rate_mean * 100)
                    errors.append(match[0].slo_attainment_rate_ci95 * 100)
                else:
                    means.append(0)
                    errors.append(0)

            positions = [x + i * bar_width for x in x_positions]
            ax.bar(
                positions,
                means,
                bar_width,
                yerr=errors,
                label=strategy,
                capsize=3,
            )

        ax.set_xlabel("Request Rate (req/s)")
        ax.set_ylabel("SLO Attainment Rate (%)")
        ax.set_title(f"[Category A] SLO Attainment Rate — {workload}")
        ax.set_xticks([x + bar_width * len(strategies) / 2 for x in x_positions])
        ax.set_xticklabels([str(r) for r in rates])
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=0.3)

        path = output_dir / f"fig1_slo_attainment_{workload}.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def _plot_pareto_front(
    aggregations: list[StrategyAggregation],
    output_dir: Path,
) -> list[Path]:
    """Figure 2: P99 TTFT vs Throughput Pareto front."""
    import matplotlib.pyplot as plt

    paths: list[Path] = []

    workloads = sorted({a.workload for a in aggregations})
    for workload in workloads:
        fig, ax = plt.subplots(figsize=(10, 8))
        wl_data = [a for a in aggregations if a.workload == workload]

        strategies = sorted({a.strategy for a in wl_data})

        for strategy in strategies:
            s_data = [a for a in wl_data if a.strategy == strategy]
            throughputs = [a.throughput_rps_mean for a in s_data]
            ttfts = [a.p99_ttft_ms_mean for a in s_data]
            ax.scatter(throughputs, ttfts, label=strategy, s=60, zorder=5)
            # 连接同一策略不同并发度的点
            sorted_pts = sorted(zip(throughputs, ttfts, strict=False))
            if len(sorted_pts) > 1:
                ax.plot(
                    [p[0] for p in sorted_pts],
                    [p[1] for p in sorted_pts],
                    alpha=0.4,
                    linestyle="--",
                )

        ax.set_xlabel("Throughput (requests/sec)")
        ax.set_ylabel("P99 TTFT (ms)")
        ax.set_title(f"[Category A] Throughput vs Latency Pareto — {workload}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        path = output_dir / f"fig2_pareto_{workload}.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def _plot_eviction_coverage(
    aggregations: list[StrategyAggregation],
    output_dir: Path,
) -> list[Path]:
    """Figure 3: Reclamation event analysis — eviction count + tokens freed.

    使用 total_all_preemptions_mean（含 native LIFO + proactive + SRPT），
    回退到 total_evictions_mean（旧数据兼容）。
    优先绘制 mixed workload rate=3.8 快照；若无 mixed 则回退到 long_context 最高 rate。
    """
    import matplotlib.pyplot as plt

    PAPER_STRATEGIES = ["preempt-evict", "preempt-evict-sjf", "static-random",
                        "largest-first", "bidkv"]
    STRATEGY_LABELS = {
        "preempt-evict": "PE",
        "preempt-evict-sjf": "PE-SJF",
        "static-random": "Static-Random",
        "largest-first": "Largest-First",
        "bidkv": "BidKV",
    }
    COLORS = {
        "preempt-evict":     "#7f7f7f",
        "preempt-evict-sjf": "#aec7e8",
        "static-random":     "#ffbb78",
        "largest-first":     "#98df8a",
        "bidkv":             "#1f77b4",
    }

    # 优先取 mixed × rate=3.8；若无则回退到 long_context 最高 rate
    mixed_data = [a for a in aggregations if a.workload == "mixed"]
    lc_data = [a for a in aggregations if a.workload == "long_context"]

    if mixed_data:
        rates = sorted({a.request_rate for a in mixed_data})
        # 优先选 3.8，否则取最高 rate
        target_rate = 3.8 if 3.8 in rates else max(rates)
        rate_data = [a for a in mixed_data if a.request_rate == target_rate]
        workload_label = "mixed"
    elif lc_data:
        rates = sorted({a.request_rate for a in lc_data})
        target_rate = max(rates)
        rate_data = [a for a in lc_data if a.request_rate == target_rate]
        workload_label = "long_context"
    else:
        logger.warning("No mixed or long_context data for Figure 3 eviction plot")
        return []

    # 构建各策略数据
    evict_counts = []
    tokens_freed_k = []
    labels = []
    colors = []

    for strat in PAPER_STRATEGIES:
        match = [a for a in rate_data if a.strategy == strat]
        if match:
            a = match[0]
            # 优先用新字段（含 native LIFO），回退旧字段
            ev = a.total_all_preemptions_mean if a.total_all_preemptions_mean > 0 \
                else a.total_evictions_mean
            fr = a.total_all_tokens_freed_mean if a.total_all_tokens_freed_mean > 0 \
                else a.total_tokens_freed_mean
        else:
            ev = 0.0
            fr = 0.0
        evict_counts.append(ev)
        tokens_freed_k.append(fr / 1000.0)
        labels.append(STRATEGY_LABELS.get(strat, strat))
        colors.append(COLORS.get(strat, "#888888"))

    if all(v == 0 for v in evict_counts) and all(v == 0 for v in tokens_freed_k):
        logger.warning("Figure 3: all eviction counts are 0 — data may be from old format")
        return []

    paths: list[Path] = []
    x_pos = list(range(len(labels)))
    bar_w = 0.35

    # 单图双 Y 轴，每策略两柱（左实心=驱逐次数，右斜线=释放 token）
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    # Left bars: reclamation count（左 Y 轴，实心）
    bars1 = ax1.bar(
        [xi - bar_w / 2 for xi in x_pos], evict_counts, bar_w,
        color=colors, edgecolor="black", linewidth=0.7,
        label="Reclamation Count (All Paths)",
    )
    # Right bars: tokens freed ×1000（右 Y 轴，斜线纹理 + 半透明）
    bars2 = ax2.bar(
        [xi + bar_w / 2 for xi in x_pos], tokens_freed_k, bar_w,
        color=colors, edgecolor="black", linewidth=0.7,
        hatch="//", alpha=0.6,
        label="Tokens Freed (×1000)",
    )

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax1.set_xlabel("Strategy", fontsize=10)
    ax1.set_ylabel("Reclamation Count", fontsize=10)
    ax2.set_ylabel("Tokens Freed (×1000)", fontsize=10)
    ax1.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax1.set_axisbelow(True)

    # 数值标签
    for bar, val in zip(bars1, evict_counts, strict=False):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(evict_counts) * 0.01,
                     f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, tokens_freed_k, strict=False):
        if val > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(tokens_freed_k) * 0.01,
                f"{val:.0f}k",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # 合并图例（ax1 + ax2 的 handle）
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)

    fig.tight_layout()

    path = output_dir / f"fig5_compress_coverage_{workload_label}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    paths.append(path)
    logger.info("Figure 5 saved to %s", path)

    # 同步更新 paper/figures/fig5_compress_coverage.pdf（若目录存在）
    paper_figures = path.resolve().parent.parent.parent.parent.parent / "paper" / "figures"
    if paper_figures.is_dir():
        import shutil
        paper_dest = paper_figures / "fig5_compress_coverage.pdf"
        shutil.copy2(str(path), str(paper_dest))
        logger.info("Copied to paper figures: %s", paper_dest)

    return paths


def generate_table1_data(aggregations: list[StrategyAggregation]) -> list[dict[str, object]]:
    """生成 Table 1 数据：7 baseline 描述 + 特性矩阵。

    Returns
    -------
    list[dict]
        每个策略一行数据。
    """
    strategy_descriptions = {
        "preempt-evict": {
            "description": "Framework default preemption (no compression)",
            "has_scoring": False,
            "has_bid": False,
            "has_solver": False,
            "design_rationale": "Lower bound — zero compression baseline",
        },
        "static-random": {
            "description": "Random victim selection with fixed compression ratio",
            "has_scoring": False,
            "has_bid": False,
            "has_solver": False,
            "design_rationale": "Control group — isolates information value",
        },
        "largest-first": {
            "description": "Capacity-greedy victim selection (evicts largest KV)",
            "has_scoring": True,
            "has_bid": False,
            "has_solver": False,
            "design_rationale": "Isolates bid mechanism contribution from scoring",
        },
        "preempt-evict-sjf": {
            "description": "SJF admission + LIFO eviction (ablation baseline)",
            "has_scoring": False,
            "has_bid": False,
            "has_solver": False,
            "design_rationale": "Isolates SJF admission contribution from eviction",
        },
        "bidkv": {
            "description": "Full BidKV pipeline (positional scoring + bid + solver)",
            "has_scoring": True,
            "has_bid": True,
            "has_solver": True,
            "design_rationale": "Complete system with user-explicit quality preference",
        },
    }

    table_rows: list[dict[str, object]] = []
    for strategy, desc in strategy_descriptions.items():
        row: dict[str, object] = {
            "strategy": strategy,
            **desc,
        }
        # 附加聚合指标（如果存在）
        strategy_aggs = [a for a in aggregations if a.strategy == strategy]
        if strategy_aggs:
            avg_slo = sum(a.slo_attainment_rate_mean for a in strategy_aggs) / len(strategy_aggs)
            avg_throughput = sum(a.throughput_rps_mean for a in strategy_aggs) / len(strategy_aggs)
            row["avg_slo_attainment_rate"] = avg_slo
            row["avg_throughput_rps"] = avg_throughput

        table_rows.append(row)

    return table_rows


def run_analysis(results_dir: Path, output_dir: Path | None = None) -> None:
    """运行完整分析流程。

    Parameters
    ----------
    results_dir:
        RunResult JSON 文件所在目录。
    output_dir:
        分析结果输出目录。默认与 results_dir 相同。
    """
    if output_dir is None:
        output_dir = results_dir

    results = load_all_run_results(results_dir)
    if not results:
        logger.warning("No results found in %s", results_dir)
        return

    logger.info("Loaded %d run results", len(results))

    aggregations = aggregate_results(results)

    # 导出 JSON 摘要
    export_summary_json(aggregations, output_dir)

    # 生成 Table 1 数据
    table1 = generate_table1_data(aggregations)
    table1_path = output_dir / "table1_baselines.json"
    table1_path.write_text(json.dumps(table1, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Table 1 data exported to %s", table1_path)

    # Figure 6: Budget sensitivity (RULE FIG6-DEFAULT: surrogate budget sensitivity)
    budget_data = compute_budget_sensitivity(aggregations)
    if budget_data:
        budget_path = output_dir / "fig6_budget_sensitivity.json"
        budget_path.write_text(
            json.dumps(budget_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Figure 6 budget sensitivity data exported to %s", budget_path)

    # 生成 Figure PDF
    generated = generate_figures(aggregations, output_dir)

    # Figure 6 plot
    generated.extend(_plot_budget_sensitivity(budget_data, output_dir / "figures"))

    logger.info("Analysis complete. %d figures generated.", len(generated))


def compute_budget_sensitivity(
    aggregations: list[StrategyAggregation],
) -> list[dict[str, object]]:
    """计算 surrogate budget sensitivity（RULE FIG6-DEFAULT）。

    衡量不同 rate（压力代理）下 BidKV 的 SLO attainment 变化率。

    Returns
    -------
    list[dict]
        每个 (workload, rate) 点的 BidKV 与所有策略 SLO attainment 对比。
    """
    data: list[dict[str, object]] = []
    workloads = sorted({a.workload for a in aggregations})
    for workload in workloads:
        wl_data = [a for a in aggregations if a.workload == workload]
        rates = sorted({a.request_rate for a in wl_data})
        for rate in rates:
            point: dict[str, object] = {"workload": workload, "rate": rate}
            for a in wl_data:
                if a.request_rate == rate:
                    point[f"{a.strategy}_slo"] = a.slo_attainment_rate_mean
                    point[f"{a.strategy}_ci95"] = a.slo_attainment_rate_ci95
            data.append(point)
    return data


def _plot_budget_sensitivity(
    data: list[dict[str, object]],
    output_dir: Path,
) -> list[Path]:
    """Figure 6: Surrogate budget sensitivity — line plot per workload."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    if not data:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    workloads = sorted({d["workload"] for d in data})
    for workload in workloads:
        wl_data = [d for d in data if d["workload"] == workload]
        rates = sorted(d["rate"] for d in wl_data)

        # Extract all strategy names from data keys
        strategy_keys = set()
        for d in wl_data:
            for key in d:
                if key.endswith("_slo") and key != "rate":
                    strategy_keys.add(key.replace("_slo", ""))

        fig, ax = plt.subplots(figsize=(10, 6))
        for strategy in sorted(strategy_keys):
            slo_key = f"{strategy}_slo"
            ci_key = f"{strategy}_ci95"
            slos = []
            cis = []
            for rate in rates:
                match = [d for d in wl_data if d["rate"] == rate]
                if match and slo_key in match[0]:
                    slos.append(float(match[0][slo_key]) * 100)
                    cis.append(float(match[0].get(ci_key, 0)) * 100)
                else:
                    slos.append(0)
                    cis.append(0)
            ax.errorbar(rates, slos, yerr=cis, label=strategy, marker="o", capsize=3)

        ax.set_xlabel("Request Rate (req/s) — Surrogate KV Pressure")
        ax.set_ylabel("SLO Attainment Rate (%)")
        ax.set_title(f"[Category B] Budget Sensitivity — {workload}")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)

        path = output_dir / f"fig6_budget_sensitivity_{workload}.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def main(argv: list[str] | None = None) -> None:
    """CLI 入口：vLLM 分析脚本。"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="BidKV vLLM Result Analysis")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing RunResult JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis (default: results-dir/analysis).",
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"

    run_analysis(results_dir, output_dir)


if __name__ == "__main__":
    main()
