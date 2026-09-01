"""SGLang 可移植性实验基础设施单元测试。

测试 config / collector / server / runner 的 CPU-only 逻辑。
不启动真实 SGLang server。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bidkv.experiments.sglang.collector import (
    RequestResult,
    RunResult,
    save_run_result,
    write_audit_entry,
)
from bidkv.experiments.sglang.config import (
    ALL_STRATEGIES,
    ALL_WORKLOADS,
    DEFAULT_REQUEST_RATES,
    STRATEGY_BIDKV,
    STRATEGY_SGLANG_DEFAULT,
    SGLangExperimentConfig,
    SGLangServerConfig,
    SLOConfig,
)
from bidkv.experiments.sglang.runner import SGLangExperimentRunner

# ===== Config Tests =====


class TestSLOConfig:
    def test_default_values(self) -> None:
        slo = SLOConfig()
        assert slo.ttft_target_ms == 2000.0
        assert slo.tpot_target_ms == 100.0

    def test_frozen(self) -> None:
        slo = SLOConfig()
        with pytest.raises(AttributeError):
            slo.ttft_target_ms = 999.0  # type: ignore[misc]


class TestSGLangServerConfig:
    def test_default_model(self) -> None:
        cfg = SGLangServerConfig()
        assert "Llama-3.1-8B-Instruct" in cfg.model

    def test_api_url(self) -> None:
        cfg = SGLangServerConfig(host="127.0.0.1", port=30000)
        assert cfg.api_url == "http://127.0.0.1:30000/v1"

    def test_base_url(self) -> None:
        cfg = SGLangServerConfig(host="0.0.0.0", port=8080)
        assert cfg.base_url == "http://0.0.0.0:8080"

    def test_cli_args(self) -> None:
        cfg = SGLangServerConfig(model="test-model", port=30000)
        args = cfg.to_cli_args()
        assert "--model" in args
        assert "test-model" in args
        assert "--port" in args
        assert "30000" in args


class TestSGLangExperimentConfig:
    def test_default_strategies(self) -> None:
        cfg = SGLangExperimentConfig()
        assert cfg.strategies == ALL_STRATEGIES
        assert len(cfg.strategies) == 3

    def test_default_workloads(self) -> None:
        cfg = SGLangExperimentConfig()
        assert cfg.workloads == ALL_WORKLOADS
        assert "mixed" in cfg.workloads
        assert "long_context" in cfg.workloads

    def test_default_rates(self) -> None:
        cfg = SGLangExperimentConfig()
        assert cfg.request_rates == DEFAULT_REQUEST_RATES
        assert cfg.request_rates == (2.0, 3.8, 5.7)

    def test_total_runs(self) -> None:
        cfg = SGLangExperimentConfig()
        expected = 3 * 2 * 3 * 3  # strategies * workloads * rates * runs
        assert cfg.total_runs == expected

    def test_total_runs_custom(self) -> None:
        cfg = SGLangExperimentConfig(
            strategies=(STRATEGY_BIDKV,),
            workloads=("mixed",),
            request_rates=(1.0, 2.0),
            workload_rates={"mixed": (1.0, 2.0), "long_context": (0.35, 0.5, 0.7)},
            runs_per_combo=2,
        )
        assert cfg.total_runs == 1 * 1 * 2 * 2

    def test_run_label(self) -> None:
        cfg = SGLangExperimentConfig()
        label = cfg.run_label("bidkv", "mixed", 2.0, 0)
        assert label == "sglang__bidkv__mixed__rate2.0__run0"

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategies"):
            SGLangExperimentConfig(strategies=("nonexistent",))

    def test_consecutive_timeout_abort(self) -> None:
        cfg = SGLangExperimentConfig()
        assert cfg.consecutive_timeout_abort == 10

    def test_slo_alignment_with_vllm(self) -> None:
        """SLO 值必须与 Issue-047 vLLM 实验一致。"""
        cfg = SGLangExperimentConfig()
        assert cfg.slo.ttft_target_ms == 2000.0
        assert cfg.slo.tpot_target_ms == 100.0


class TestStrategyConstants:
    def test_all_strategies_tuple(self) -> None:
        assert STRATEGY_SGLANG_DEFAULT in ALL_STRATEGIES
        assert STRATEGY_BIDKV in ALL_STRATEGIES
        assert len(ALL_STRATEGIES) == 3

    def test_strategy_count(self) -> None:
        assert len(ALL_STRATEGIES) == 3


# ===== Collector Tests =====


class TestRequestResult:
    def test_success_flag(self) -> None:
        r = RequestResult(request_id="r1")
        assert r.success is True

    def test_error_flag(self) -> None:
        r = RequestResult(request_id="r1", error="timeout")
        assert r.success is False

    def test_tpot_computation(self) -> None:
        r = RequestResult(
            request_id="r1",
            ttft_ms=100.0,
            total_latency_ms=600.0,
            completion_tokens=6,
        )
        # TPOT = (600 - 100) / (6 - 1) = 100.0
        assert r.tpot_ms == 100.0

    def test_tpot_single_token(self) -> None:
        r = RequestResult(
            request_id="r1",
            ttft_ms=100.0,
            total_latency_ms=200.0,
            completion_tokens=1,
        )
        assert r.tpot_ms == 0.0

    def test_tpot_zero_tokens(self) -> None:
        r = RequestResult(
            request_id="r1",
            ttft_ms=100.0,
            total_latency_ms=200.0,
            completion_tokens=0,
        )
        assert r.tpot_ms == 0.0


class TestRunResult:
    @staticmethod
    def _make_run_result() -> RunResult:
        return RunResult(
            run_label="test_run",
            strategy="bidkv",
            workload="mixed",
            request_rate=2.0,
            run_index=0,
            start_time=1000.0,
            end_time=1010.0,
            request_results=[
                RequestResult(request_id="r1", ttft_ms=100.0, total_latency_ms=500.0),
                RequestResult(request_id="r2", ttft_ms=200.0, total_latency_ms=600.0),
                RequestResult(request_id="r3", error="timeout"),
            ],
        )

    def test_duration(self) -> None:
        rr = self._make_run_result()
        assert rr.duration_s == 10.0

    def test_successful_requests(self) -> None:
        rr = self._make_run_result()
        assert len(rr.successful_requests) == 2

    def test_failed_requests(self) -> None:
        rr = self._make_run_result()
        assert len(rr.failed_requests) == 1

    def test_throughput(self) -> None:
        rr = self._make_run_result()
        # 2 successful / 10s = 0.2 rps
        assert rr.compute_throughput_rps() == pytest.approx(0.2)

    def test_p95_ttft(self) -> None:
        rr = self._make_run_result()
        p95 = rr.compute_p95_ttft_ms()
        assert p95 > 0.0

    def test_p99_ttft(self) -> None:
        rr = self._make_run_result()
        p99 = rr.compute_p99_ttft_ms()
        assert p99 > 0.0

    def test_slo_attainment(self) -> None:
        rr = self._make_run_result()
        # With ttft target 2000ms, both 100ms and 200ms pass
        rate = rr.compute_slo_attainment_rate(ttft_target_ms=2000.0)
        assert rate == 1.0

    def test_slo_attainment_partial(self) -> None:
        rr = self._make_run_result()
        # With ttft target 150ms, only r1 (100ms) passes, r2 (200ms) fails
        rate = rr.compute_slo_attainment_rate(ttft_target_ms=150.0)
        assert rate == 0.5

    def test_completion_rate(self) -> None:
        rr = self._make_run_result()
        # 2 success / 3 total
        assert rr.compute_completion_rate() == pytest.approx(2.0 / 3.0)


class TestSaveRunResult:
    def test_save_creates_json(self, tmp_path: Path) -> None:
        rr = RunResult(
            run_label="sglang__bidkv__mixed__rate2.0__run0",
            strategy="bidkv",
            workload="mixed",
            request_rate=2.0,
            run_index=0,
            start_time=1000.0,
            end_time=1010.0,
            request_results=[
                RequestResult(request_id="r1", ttft_ms=100.0, total_latency_ms=500.0),
            ],
        )
        path = save_run_result(rr, tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["strategy"] == "bidkv"
        assert data["summary"]["total_requests"] == 1
        assert data["summary"]["successful_requests"] == 1
        assert data["summary"]["ttft_ms_p95"] > 0


# ===== Runner Tests =====


class TestSGLangExperimentRunner:
    def test_construction(self) -> None:
        cfg = SGLangExperimentConfig()
        runner = SGLangExperimentRunner(cfg)
        assert runner.config.strategies == ALL_STRATEGIES


# ===== Audit Logging Tests =====


class TestWriteAuditEntry:
    def test_creates_jsonl_file(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "sglang_test" / "audit_test.jsonl"
        write_audit_entry(
            audit_path,
            candidate_count=3,
            candidate_request_ids=["r1", "r2", "r3"],
            strategy="bidkv",
            kv_usage_pct=0.92,
        )
        assert audit_path.exists()
        lines = audit_path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["event"] == "pressure_trigger"
        assert entry["candidate_count"] == 3
        assert entry["strategy"] == "bidkv"
        assert entry["kv_usage_pct"] == 0.92
        assert "candidate_list_hash" in entry
        assert "timestamp" in entry

    def test_appends_multiple_entries(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit.jsonl"
        for i in range(3):
            write_audit_entry(
                audit_path,
                candidate_count=i + 1,
                candidate_request_ids=[f"r{j}" for j in range(i + 1)],
                strategy="sglang_default",
                kv_usage_pct=0.8 + i * 0.05,
            )
        lines = audit_path.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_deterministic_hash(self, tmp_path: Path) -> None:
        """相同候选列表（不同顺序）应产生相同 hash。"""
        audit_path = tmp_path / "audit.jsonl"
        write_audit_entry(
            audit_path,
            candidate_count=2,
            candidate_request_ids=["r2", "r1"],
            strategy="bidkv",
            kv_usage_pct=0.9,
        )
        write_audit_entry(
            audit_path,
            candidate_count=2,
            candidate_request_ids=["r1", "r2"],
            strategy="bidkv",
            kv_usage_pct=0.9,
        )
        lines = audit_path.read_text().strip().split("\n")
        h1 = json.loads(lines[0])["candidate_list_hash"]
        h2 = json.loads(lines[1])["candidate_list_hash"]
        assert h1 == h2
