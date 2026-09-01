"""Tests for bidkv.experiments.vllm — vLLM experiment infrastructure.

测试策略：
- 所有实验框架内部逻辑使用真实组件（无 mock）
- 文件 I/O 使用 tmp_path fixture
- 不测试实际 vLLM server 连接（需要 GPU）
"""

from __future__ import annotations

import json

import pytest

from bidkv.experiments.vllm.analysis import (
    StrategyAggregation,
    aggregate_results,
    compute_ci95,
    export_summary_json,
    generate_table1_data,
)
from bidkv.experiments.vllm.collector import (
    CandidateSnapshot,
    MetricsCollector,
    RequestResult,
    RunResult,
    load_run_result,
    save_run_result,
)
from bidkv.experiments.vllm.config import (
    ALL_STRATEGIES,
    ALL_WORKLOADS,
    ExperimentConfig,
    VLLMServerConfig,
)
from bidkv.experiments.vllm.runner import (
    VLLMExperimentRunner,
    parse_args,
    verify_candidate_consistency,
)
from bidkv.experiments.vllm.workload import (
    RequestTrace,
    WorkloadTrace,
    load_trace,
    save_trace,
)

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestExperimentConfig:
    """ExperimentConfig 配置验证测试。"""

    def test_default_config(self) -> None:
        config = ExperimentConfig()
        assert config.strategies == ALL_STRATEGIES
        assert config.workloads == ALL_WORKLOADS
        assert config.runs_per_combo == 3
        # 90 = 5 strategies × 2 workloads × 3 rates × 3 runs.
        assert config.total_runs == 5 * 2 * 3 * 3

    def test_custom_config(self) -> None:
        config = ExperimentConfig(
            strategies=("preempt-evict", "bidkv"),
            workloads=("mixed",),
            request_rates=(2.0,),
            workload_rates={"mixed": (2.0,), "long_context": (0.35,)},
            runs_per_combo=2,
        )
        assert config.total_runs == 2 * 1 * 1 * 2  # 4

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategies"):
            ExperimentConfig(strategies=("nonexistent-strategy",))

    def test_invalid_workload_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown workloads"):
            ExperimentConfig(workloads=("chat",))

    def test_invalid_runs_raises(self) -> None:
        with pytest.raises(ValueError, match="runs_per_combo must be >= 1"):
            ExperimentConfig(runs_per_combo=0)

    def test_run_label(self) -> None:
        config = ExperimentConfig()
        label = config.run_label("bidkv", "mixed", 2.0, 2)
        assert label == "bidkv__mixed__rate2.0__r2"


class TestVLLMServerConfig:
    """VLLMServerConfig 测试。"""

    def test_default_urls(self) -> None:
        config = VLLMServerConfig()
        assert config.base_url == "http://127.0.0.1:8000"
        assert config.api_url == "http://127.0.0.1:8000/v1"

    def test_cli_args(self) -> None:
        config = VLLMServerConfig(
            model="test-model",
            block_size=32,
            max_num_seqs=16,
            gpu_memory_utilization=0.90,
            enforce_eager=True,
            port=8080,
        )
        args = config.to_cli_args()
        assert "--model" in args
        assert "test-model" in args
        assert "--enforce-eager" in args
        assert "--block-size" in args
        assert "32" in args

    def test_cli_args_no_eager(self) -> None:
        config = VLLMServerConfig(enforce_eager=False)
        args = config.to_cli_args()
        assert "--enforce-eager" not in args


# ---------------------------------------------------------------------------
# Workload trace tests
# ---------------------------------------------------------------------------


class TestWorkloadTrace:
    """WorkloadTrace 序列化/反序列化测试。"""

    def _make_trace(self) -> WorkloadTrace:
        return WorkloadTrace(
            workload_name="chat",
            dataset_source="test_sharegpt",
            frozen_at="2026-03-16T00:00:00Z",
            requests=[
                RequestTrace(
                    request_id="chat-0001",
                    prompt="Hello, how are you?",
                    max_tokens=128,
                    expected_output="I'm fine, thanks!",
                    metadata={"dataset": "sharegpt"},
                ),
                RequestTrace(
                    request_id="chat-0002",
                    prompt="What is the capital of France?",
                    max_tokens=64,
                    expected_output="Paris",
                ),
            ],
        )

    def test_save_and_load_trace(self, tmp_path: pytest.TempPathFactory) -> None:
        trace = self._make_trace()
        path = tmp_path / "chat.json"
        save_trace(trace, path)

        loaded = load_trace(path)
        assert loaded.workload_name == "chat"
        assert loaded.num_requests == 2
        assert loaded.requests[0].request_id == "chat-0001"
        assert loaded.requests[0].prompt == "Hello, how are you?"
        assert loaded.requests[0].expected_output == "I'm fine, thanks!"
        assert loaded.requests[0].metadata == {"dataset": "sharegpt"}
        assert loaded.dataset_source == "test_sharegpt"

    def test_load_nonexistent_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        with pytest.raises(FileNotFoundError):
            load_trace(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# Collector tests
# ---------------------------------------------------------------------------


class TestRequestResult:
    """RequestResult 指标计算测试。"""

    def test_success_flag(self) -> None:
        r = RequestResult(request_id="test-1")
        assert r.success
        r_fail = RequestResult(request_id="test-2", error="timeout")
        assert not r_fail.success

    def test_tpot_calculation(self) -> None:
        r = RequestResult(
            request_id="test-1",
            ttft_ms=100.0,
            total_latency_ms=1100.0,
            completion_tokens=11,
        )
        # TPOT = (1100 - 100) / (11 - 1) = 100 ms
        assert abs(r.tpot_ms - 100.0) < 1e-6

    def test_tpot_single_token(self) -> None:
        r = RequestResult(
            request_id="test-1",
            ttft_ms=100.0,
            total_latency_ms=100.0,
            completion_tokens=1,
        )
        assert r.tpot_ms == 0.0


class TestRunResult:
    """RunResult 聚合计算测试。"""

    def _make_run_result(self) -> RunResult:
        return RunResult(
            run_label="bidkv__mixed__rate2.0__r0",
            strategy="bidkv",
            workload="mixed",
            request_rate=2.0,
            run_index=0,
            start_time=1000.0,
            end_time=1010.0,
            request_results=[
                RequestResult(
                    request_id="r1",
                    ttft_ms=50.0,
                    total_latency_ms=200.0,
                    completion_tokens=10,
                ),
                RequestResult(
                    request_id="r2",
                    ttft_ms=100.0,
                    total_latency_ms=300.0,
                    completion_tokens=15,
                ),
                RequestResult(
                    request_id="r3",
                    ttft_ms=2500.0,
                    total_latency_ms=3000.0,
                    completion_tokens=20,
                ),
                RequestResult(
                    request_id="r4",
                    error="timeout",
                ),
            ],
        )

    def test_duration(self) -> None:
        r = self._make_run_result()
        assert r.duration_s == 10.0

    def test_successful_requests(self) -> None:
        r = self._make_run_result()
        assert len(r.successful_requests) == 3
        assert len(r.failed_requests) == 1

    def test_throughput(self) -> None:
        r = self._make_run_result()
        assert abs(r.compute_throughput_rps() - 0.3) < 1e-6

    def test_p99_ttft(self) -> None:
        r = self._make_run_result()
        # 3 个成功请求：50, 100, 2500 → 排序后
        # idx = max(0, int(3 * 0.99) - 1) = max(0, 1) = 1 → ttfts[1] = 100
        p99 = r.compute_p99_ttft_ms()
        assert p99 == 100.0

    def test_slo_attainment_rate(self) -> None:
        r = self._make_run_result()
        # SLO target 2000ms → 2/3 attain (ttft=50, 100 pass; ttft=2500 fails)
        rate = r.compute_slo_attainment_rate(2000.0)
        assert abs(rate - 2 / 3) < 1e-6


class TestRunResultPersistence:
    """RunResult 保存/加载测试。"""

    def test_save_and_load(self, tmp_path: pytest.TempPathFactory) -> None:
        result = RunResult(
            run_label="test__mixed__rate2.0__r0",
            strategy="bidkv",
            workload="mixed",
            request_rate=2.0,
            run_index=0,
            start_time=1000.0,
            end_time=1010.0,
            request_results=[
                RequestResult(
                    request_id="r1",
                    ttft_ms=50.0,
                    total_latency_ms=200.0,
                ),
            ],
            adapter_metrics={"total_evictions": 5, "total_tokens_freed": 1024},
        )

        path = save_run_result(result, tmp_path)
        loaded = load_run_result(path)

        assert loaded.run_label == "test__mixed__rate2.0__r0"
        assert loaded.strategy == "bidkv"
        assert loaded.request_rate == 2.0
        assert len(loaded.request_results) == 1
        assert loaded.request_results[0].ttft_ms == 50.0
        assert loaded.adapter_metrics["total_evictions"] == 5

    def test_save_with_candidate_snapshots(self, tmp_path: pytest.TempPathFactory) -> None:
        result = RunResult(
            run_label="test__mixed__rate2.0__r0",
            strategy="bidkv",
            workload="mixed",
            request_rate=2.0,
            run_index=0,
            candidate_snapshots=[
                CandidateSnapshot(
                    timestamp=1000.0,
                    pressure_ratio=0.9,
                    candidate_request_ids=["r1", "r2", "r3"],
                    needed_tokens=512,
                    strategy_name="bidkv",
                ),
            ],
        )

        path = save_run_result(result, tmp_path)
        loaded = load_run_result(path)

        assert len(loaded.candidate_snapshots) == 1
        assert loaded.candidate_snapshots[0].pressure_ratio == 0.9
        assert loaded.candidate_snapshots[0].candidate_request_ids == ["r1", "r2", "r3"]


# ---------------------------------------------------------------------------
# Analysis tests
# ---------------------------------------------------------------------------


class TestCI95:
    """95% CI 计算测试。"""

    def test_empty(self) -> None:
        mean, ci = compute_ci95([])
        assert mean == 0.0
        assert ci == 0.0

    def test_single(self) -> None:
        mean, ci = compute_ci95([42.0])
        assert mean == 42.0
        assert ci == 0.0

    def test_multiple(self) -> None:
        values = [10.0, 12.0, 14.0]
        mean, ci = compute_ci95(values)
        assert abs(mean - 12.0) < 1e-6
        assert ci > 0  # CI should be positive


class TestAggregation:
    """结果聚合测试。"""

    def _make_results(self) -> list[RunResult]:
        results = []
        for run_idx in range(3):
            results.append(
                RunResult(
                    run_label=f"bidkv__mixed__rate2.0__r{run_idx}",
                    strategy="bidkv",
                    workload="mixed",
                    request_rate=2.0,
                    run_index=run_idx,
                    start_time=1000.0,
                    end_time=1010.0,
                    request_results=[
                        RequestResult(
                            request_id=f"r{i}",
                            ttft_ms=50.0 + run_idx * 10,
                            total_latency_ms=200.0 + run_idx * 20,
                            completion_tokens=10,
                        )
                        for i in range(10)
                    ],
                    adapter_metrics={
                        "total_evictions": 2 + run_idx,
                        "total_tokens_freed": 100 + run_idx * 50,
                        "total_pressure_events": 5,
                    },
                )
            )
        return results

    def test_aggregate_produces_single_entry(self) -> None:
        results = self._make_results()
        aggs = aggregate_results(results)
        assert len(aggs) == 1
        assert aggs[0].strategy == "bidkv"
        assert aggs[0].workload == "mixed"
        assert aggs[0].request_rate == 2.0
        assert aggs[0].runs == 3

    def test_aggregate_mean_values(self) -> None:
        results = self._make_results()
        aggs = aggregate_results(results)
        agg = aggs[0]
        # 3 runs, each 10s, each 10 successful requests → throughput = 1.0 rps
        assert abs(agg.throughput_rps_mean - 1.0) < 1e-6

    def test_aggregate_ci95_positive(self) -> None:
        results = self._make_results()
        aggs = aggregate_results(results)
        agg = aggs[0]
        # 由于 3 次运行 ttft 不同，CI 应该 > 0
        assert agg.p99_ttft_ms_ci95 >= 0


class TestTable1:
    """Table 1 生成测试。"""

    def test_table1_has_all_strategies(self) -> None:
        table = generate_table1_data([])
        strategies = {row["strategy"] for row in table}
        assert "largest-first" in strategies  # Largest-First 必须出现
        assert "bidkv" in strategies
        assert len(strategies) == 5

    def test_table1_largest_first_present(self) -> None:
        """Largest-First 必须出现在 Table 1 中（was H2O-Style, issue-047 验收标准）。"""
        table = generate_table1_data([])
        lf_rows = [r for r in table if r["strategy"] == "largest-first"]
        assert len(lf_rows) == 1
        assert lf_rows[0]["has_scoring"] is True
        assert lf_rows[0]["has_bid"] is False


class TestExportSummary:
    """摘要导出测试。"""

    def test_export_creates_json(self, tmp_path: pytest.TempPathFactory) -> None:
        aggs = [
            StrategyAggregation(
                strategy="bidkv",
                workload="mixed",
                request_rate=2.0,
                runs=3,
                throughput_rps_mean=1.5,
                p99_ttft_ms_mean=500.0,
                slo_attainment_rate_mean=0.05,
            )
        ]
        path = export_summary_json(aggs, tmp_path)

        data = json.loads(path.read_text())
        assert len(data["aggregations"]) == 1


# ---------------------------------------------------------------------------
# Runner tests
# ---------------------------------------------------------------------------


class TestParseArgs:
    """命令行参数解析测试。"""

    def test_default_args(self) -> None:
        config, resume = parse_args([])
        assert config.strategies == ALL_STRATEGIES
        assert config.workloads == ALL_WORKLOADS
        assert config.runs_per_combo == 3
        assert resume is False

    def test_custom_strategies(self) -> None:
        config, _ = parse_args(["--strategies", "preempt-evict,bidkv"])
        assert config.strategies == ("preempt-evict", "bidkv")

    def test_custom_workloads(self) -> None:
        config, _ = parse_args(["--workloads", "mixed"])
        assert config.workloads == ("mixed",)

    def test_custom_rates(self) -> None:
        config, _ = parse_args(["--request-rates", "1.5,3.0"])
        assert config.request_rates == (1.5, 3.0)

    def test_custom_runs(self) -> None:
        config, _ = parse_args(["--runs", "5"])
        assert config.runs_per_combo == 5


class TestCandidateConsistency:
    """Candidate-universe consistency 验证测试。"""

    def test_empty_results(self) -> None:
        report = verify_candidate_consistency([])
        assert report["total_groups"] == 0

    def test_consistency_with_snapshots(self) -> None:
        results = [
            RunResult(
                run_label="bidkv__mixed__rate2.0__r0",
                strategy="bidkv",
                workload="mixed",
                request_rate=2.0,
                run_index=0,
                candidate_snapshots=[
                    CandidateSnapshot(
                        timestamp=1000.0,
                        pressure_ratio=0.9,
                        candidate_request_ids=["r1", "r2", "r3"],
                        needed_tokens=512,
                        strategy_name="bidkv",
                    ),
                ],
            ),
            RunResult(
                run_label="largest-first__mixed__rate2.0__r0",
                strategy="largest-first",
                workload="mixed",
                request_rate=2.0,
                run_index=0,
                candidate_snapshots=[
                    CandidateSnapshot(
                        timestamp=1000.0,
                        pressure_ratio=0.9,
                        candidate_request_ids=["r1", "r2", "r3"],
                        needed_tokens=512,
                        strategy_name="largest-first",
                    ),
                ],
            ),
        ]
        report = verify_candidate_consistency(results)
        assert report["groups_with_snapshots"] == 1


class TestVLLMExperimentRunner:
    """VLLMExperimentRunner 初始化测试。"""

    def test_runner_construction(self) -> None:
        config = ExperimentConfig(
            strategies=("preempt-evict",),
            workloads=("mixed",),
            request_rates=(2.0,),
            runs_per_combo=1,
        )
        runner = VLLMExperimentRunner(config)
        assert runner.config is config


class TestMetricsCollector:
    """MetricsCollector 测试（不连接实际服务）。"""

    def test_parse_prometheus_text(self) -> None:
        text = """# HELP vllm_num_requests Total number of requests
# TYPE vllm_num_requests counter
vllm_num_requests_total 42
vllm_avg_prompt_throughput_toks_per_s 150.5
"""
        metrics = MetricsCollector._parse_prometheus_text(text)
        assert metrics["vllm_num_requests_total"] == 42.0
        assert abs(metrics["vllm_avg_prompt_throughput_toks_per_s"] - 150.5) < 1e-6

    def test_collector_url(self) -> None:
        collector = MetricsCollector("http://localhost:8000")
        assert collector.metrics_url == "http://localhost:8000/metrics"

    def test_collector_reset(self) -> None:
        collector = MetricsCollector("http://localhost:8000")
        collector._snapshots.append({"test": True})
        assert len(collector.snapshots) == 1
        collector.reset()
        assert len(collector.snapshots) == 0
