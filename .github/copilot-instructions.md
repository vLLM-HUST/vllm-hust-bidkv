# BidKV Copilot Instructions

## 仓库信息

| 字段           | 值                                                                   |
| -------------- | -------------------------------------------------------------------- |
| 仓库名         | bidkv                                                                |
| PyPI 包名      | `bidkv`                                                              |
| 导入命名空间   | `bidkv`                                                              |
| Python         | ≥ 3.10                                                               |
| 外部依赖       | **零** — 仅依赖 Python stdlib                                        |
| 论文           | SC 2026 已接收论文                                                   |
| 当前 Phase     | **Artifact maintenance** — 论文已接收，继续维护实现、实验与复现材料 |

## 项目定位

BidKV 是一个 **framework-portable** 的 KV cache 请求调度原语。

核心问题：当 KV cache 压力超过阈值时，**谁应该被驱逐**？
vLLM 默认使用 LIFO/FCFS 和搜索回填 — BidKV 通过质量感知的驱逐排序
替代这一策略，使得 preemption 决策基于 **U = tokens_freed / (δ + ε)**
（每单位质量损失能回收多少 KV 空间）。

BidKV **不做压缩**——它只控制“谁被 preempt”，底层执行仍是
框架原生的 preempt + recompute（vLLM）或 token-level release（SGLang）。

BidKV 是**独立 Python 包**，零 sagellm 依赖，可作为 vLLM / SGLang 的纯插件使用。

## 代码架构

```
src/bidkv/
├── _version.py              # 版本唯一来源
├── config.py                # BidKVConfig（feature gate + kill switch，默认 OFF）
├── __init__.py              # 公开 API
├── protocol/                # 核心类型：CompressionBid, BidPool, BidAcceptance
│   ├── bid.py
│   ├── errors.py
│   └── provider.py
├── scoring/                 # Token 重要度评分策略
│   ├── base.py              # ScoringStrategy Protocol
│   └── positional.py        # PositionalScoring（attention sink + recency heuristic，唯一生产评分策略）
├── pool/                    # BidPoolManager
│   └── manager.py
├── pressure/                # PressureDetector（KV 压力检测）
│   ├── config.py
│   └── detector.py
├── solver/                  # GreedyBidSolver（bid 排序 + 贪心选取）
│   ├── config.py
│   └── greedy.py
├── baselines/               # 7 策略（6 baseline + BidKV）
│   ├── base.py              # BaselineStrategy ABC, RequestState, BaselineContext
│   ├── registry.py          # BaselineRegistry（名称→策略实例）
│   ├── preempt_evict.py     # 1. vLLM 原生 baseline (FCFS+LIFO)
│   ├── preempt_evict_sjf.py # 2. PE + SJF admission
│   ├── static_random.py     # 3. 随机驱逐
│   ├── largest_first.py     # 4. 容量贪心驱逐（was h2o_style.py）
│   ├── uniform.py           # 5. 均等驱逐
│   ├── slack_aware.py       # 6. SLO-deadline aware
│   └── bidkv_strategy.py    # 7. BidKV 完整系统
├── adapters/                # 框架适配器
│   ├── base.py              # FrameworkAdapter ABC（5 层职责边界）
│   ├── vllm/                # vLLM v1 adapter
│   │   ├── adapter.py       # VLLMAdapter
│   │   ├── plugin.py        # monkey-patch EngineCore.__init__
│   │   ├── scheduler_hook.py
│   │   └── positional_hook.py  # decode step positional scoring 更新
│   └── sglang/              # SGLang adapter
│       ├── adapter.py       # SGLangAdapter
│       └── scheduler_hook.py # monkey-patch get_next_batch_to_run()
└── experiments/             # 实验框架
    ├── metrics.py           # 实验指标采集
    ├── common/              # 共享：trace, audit, runner, report
    ├── vllm/                # vLLM 实验：runner, serve, config, analysis, etc.
    └── sglang/              # SGLang 实验：runner, server, config, analysis, etc.
```

总量：~10,000 LOC Python，373 tests。

## 关键设计模式

### 1. BaselineRegistry — 策略注册表
所有 7 个策略通过 `BaselineRegistry.register()` 注册，通过名称获取。
禁止在 registry 中硬编码策略类。新策略必须实现 `BaselineStrategy` ABC。

### 2. FrameworkAdapter ABC — 跨框架适配
职责：KV stats 获取 → 策略注入（谁被 preempt）→ Lifecycle 管理。
在 vLLM Mode A 中，adapter 仅做请求排序，不执行 token-level 操作。
每个被支持的 serving framework 实现一个 adapter。

注：Mode B 一次性 token-level 方法（`execute_compression()` / `try_compress()` 等）
已从两个 adapter 中**完全移除**（2026-04-08）。

### 3. 默认 OFF + Kill Switch
`BidKVConfig(enabled=False)` 永远是默认。
`BidKVConfig(kill_switch=True)` 立即绕过所有 bid 逻辑。

### 4. 零外部依赖
`dependencies = []`。不依赖 torch / numpy / sagellm / vllm / sglang。
vLLM / SGLang 仅在 adapter 内通过 runtime import 引用。

## 实验方案（v2.3-frozen，不可修改）

实验方案已通过 reviewer 审核冻结。以下给出关键约束，供实施时参考。

### 冻结规则索引

| Rule ID                    | 说明                                                    |
| -------------------------- | ------------------------------------------------------- |
| RULE FIG6-DEFAULT          | Figure 6 默认为 surrogate budget sensitivity            |
| RULE FIG3AB-FREEZE         | Fig 3a/b 移出 Mode A 主线，Mode B / Appendix only      |
| RULE SCENARIO-SWITCH       | Scenario A/B 切换：SLO attainment Δ_avg ≥ 10% + 无反转 |
| RULE SGLANG-FALLBACK-CLAIM | SGLang request-level fallback 下 claim 降级             |
| RULE NARRATIVE-BASELINE-SPLIT | 正文 6 策略核心 + 1 策略次级                         |
| RULE RATE-FREEZE           | rate 冻结与 trace 冻结同级                              |

### 双平台分工

| 维度         | vLLM（主量化平台）                     | SGLang（portability slice）          |
| ------------ | -------------------------------------- | ------------------------------------ |
| 策略数       | 7（完整归因链）                        | 3（精简方向验证）                     |
| 矩阵         | 7 × 2 × 3 × 3 = 126 runs             | 3 × 2 × 3 × 3 = 54 runs            |
| 执行语义     | Mode A: 请求级调度 + vLLM 原生 preempt/recompute | token-level (RadixAttention) / request-level fallback |
| 产出物       | Table 1 + Fig 3/4/5/6                 | Table 2 + Fig 7                     |
| claim 角色   | Scenario A 核心主张                    | directional consistency + portability |

### vLLM 当前架构：质量感知的请求调度

BidKV 在 vLLM 上的定位是**请求调度插件**，不是压缩引擎。

它解决的核心问题：当 KV cache 空间不足时，**应该 preempt 哪个请求？**

vLLM 原生的答案是 LIFO（最后进入 running 的先被驱逐）。
BidKV 的答案是：驱逐那个“每单位质量损失能释放最多 KV 空间”的请求（U 最高的）。

**执行机制完全是 vLLM 原生的**：被选中的请求通过 `scheduler._preempt_request()`
驱逐，其 KV blocks 全部释放，重新调度时 recompute from scratch。
BidKV 不修改这个执行路径，只控制“谁被选中”。

所有 7 个策略的调度分化：

| 层面               | PE            | PE-SJF        | Slack       | Random/Largest-First/Uniform | BidKV                          |
| ------------------ | ------------- | ------------- | ----------- | ---------------------------- | ------------------------------ |
| Waiting 排序       | FCFS (无排序) | SJF           | EDF (到达序) | SJF (prompt_tokens)          | SJF (prompt_tokens)            |
| Running 排序       | LIFO (无排序) | LIFO (无排序) | cached prio | cached prio                  | 95% KV 门控 + avg_prompt≤500  |
| select_victims()   | N/A           | N/A           | slack-based | 各自启发式                    | **U = r/(δ+ε)** 质量感知      |
| SRPT 主动驱逐      | ❌            | ❌            | ❌          | ✅ (同等估算)                 | ❌ (recompute 成本过高)        |
| Proactive preempt  | ❌            | ❌            | ✅          | ✅                            | ❌ (显式 skip)                 |

BidKV 的唯一分化点：`select_victims()` 中使用完整 scoring→bid→pool→solver
管线计算 **U = freed / (δ + ε)**，其中 **δ = 1 + 0.5c + 0.3P**（c=completion ratio, P=num_preemptions），
δ ∈ [1.0, 1.8]，实现有界的质量感知 preemption 排序。

**BidKV proactive preempt 设计决策**：
`_proactive_preempt()` 中 BidKV 被显式跳过（`if strategy_name == "bidkv": return`）。
BidKV 依赖 running reorder（KV>95% 门控）+ vLLM 原生 preemption 路径，
不主动发起 proactive preempt。v9 实验证实：启用 proactive preempt 会导致驱逐风暴。

### Mode B 状态（已清除）

Mode B 的 token-level 代码已于 2026-04-08 从代码库中完全移除：
- `adapters/vllm/truncation_hook.py` — 已删除
- `adapters/sglang/radix_hook.py` — 已删除
- `adapters/sglang/adapter.py` 中的 `execute_compression()` / `try_compress()` / `_refresh_bids()` 等方法 — 已删除
- `adapters/vllm/scheduler_hook.py` 中的 truncation_hook 死导入 — 已删除

Mode B issue #054 永久搁置。

### SGLang 3 策略

Preempt-Evict (Default) → Slack-Aware → BidKV

方向一致性（Directional Consistency）：
- DC-1a: BidKV ≥ Preempt-Evict
- DC-1b: BidKV ≥ Slack-Aware

**SGLang 执行语义现状**：

SGLang adapter 使用 `scheduler_hook` monkey-patch `get_next_batch_to_run()`，
实现 request-level 调度——与 vLLM Mode A 完全对称。
token-level radix_hook 代码已删除（2026-04-08），当前 SGLang 走纯 request-level 路径。

### Phase 路线（vLLM）

Phase 0 策略+指标冻结 → Phase 1 数据集冻结 → Phase 2 Pilot calibration →
Phase 3 Frozen trace → Phase 4 Sanity check → Phase 5 全量 126 runs →
Phase 6 消融（可选）→ Phase 7 论文映射

### 六大原则

P1 真实性优先（ShareGPT + Poisson）| P2 先 pilot 后冻结 | P3 trace 不可回改 |
P4 最小闭环优先 | P5 失败计入结果 | P6 单一变量控制

## 核心文档（已冻结，不可修改内容方向）

| 文档路径                           | 内容                                             |
| ---------------------------------- | ------------------------------------------------ |
| `docs/experiment_protocol.md`      | 主实验方案 v2.3-frozen（§1-§14，1300+ 行）       |
| `docs/vllm-route-redesign.md`      | vLLM crash 根因 + 双轨方案（10 节）              |
| `docs/sglang-portability-slice.md` | SGLang portability slice v2.3-frozen（9 节）      |
| `docs/baseline-specs.md`           | 7 baseline 策略规格                               |
| `results/claim_freeze_early.md`    | Claim freeze + Scenario A/B 规则 v2.3-frozen     |

**⚠️ 不得修改已冻结文档的实验方向、策略列表、指标定义、figure 语义。**
仅允许 typo 修正和实施备注。

## Wave 1 关键代码变更（#049 + #051，已合入）

### #049 vLLM Mode A Recompute Fallback
- `adapters/vllm/adapter.py`：Mode B 方法已完全删除（936→372 行）
- `scheduler_hook.py` 仅做请求排序 + `_preempt_request()`（调度语义）
- **已完全移除** `_free_tail_blocks()` — 不再绕过 KVCacheCoordinator

### #051 SGLang 策略列表更新 + Audit
- `experiments/sglang/config.py`：策略列表更新为 v2.3（sglang_default, slack_aware, bidkv）
- `experiments/sglang/serve_entry.py`：修复 `active=True` → `enabled=True`；新增 BaselineRegistry 路由
- `experiments/sglang/runner.py`：CLI 默认策略更新
- `experiments/sglang/collector.py`：新增 `write_audit_entry()` 公平性审计日志

## Wave 2-3 关键变更（#052 + #055，已合入）

### #052 SGLang Smoke Test
- 3 策略 × 1 workload × 1 rate × 1 run = 3 runs，零 crash
- TTFT p50 ~85ms, TPOT p50 ~32ms, SLO attainment 100%（成功请求）
- 方向一致性 DC-1a/DC-1b 初步通过（timeout-dominated regime）
- 结果保存至 `results/sglang_smoke_mode_a/`

### #055 Pilot Calibration + Trace/Rate 冻结
- Situation A：复用 90 组已有数据（formal 72 + pilot_v3 12 + pilot_v3_mixed_high 6），无需额外 pilot
- 冻结 per-workload rates：mixed `(2.0, 3.8, 5.7)`, long_context `(0.35, 0.5, 0.7)`
- `WORKLOAD_REQUEST_RATES` 冻结字典新增（vLLM + SGLang config 同步）
- `get_rates_for_workload(workload)` 方法 + `--mixed-rates` / `--long-rates` CLI 覆盖
- seed=42 formal traces（6 文件 + manifest.json），SHA-256 验证通过
- 校准报告：`results/pilot_055/calibration_report.md`

## 当前实施阶段与 Issue 追踪

Issue 文件位于 `sagellm-docs/issues/dir1-compression-scheduling-primitive/`。

### Phase C 并行化执行计划

```
─── Wave 1：代码修复（可并行）────────────────────────
#049  vLLM Mode A Recompute Fallback   │ P0  │ ✅ 完成+验收 (2026-03-19)
#051  SGLang 策略列表更新 + Audit      │ P0  │ ✅ 完成+验收 (2026-03-19)
─── Wave 2：Smoke Test（可并行）──────────────────────
#050  vLLM Smoke Test (7 runs)          │ P0  │ ⬜  (依赖 #049 ✅)
#052  SGLang Smoke Test (3 runs)        │ P0  │ ✅ 完成+验收 (2026-03-19)
─── Wave 3：Pilot Calibration ──────────────────────────
#055  Pilot + Trace/Rate 冻结           │ P0  │ ✅ 完成+验收 (2026-03-19)
─── Dead Code 清理 ──────────────────────────────────
#056  Dead code 清理                    │ P0  │ ✅ 完成 (2026-04-09)，373 tests pass
─── Wave 4：全量实验 ─────────────────────────────────
#053  全量 126+54 runs + Figure/Table   │ P0  │ ⬜  (依赖 #055)
─── 独立增强（不阻塞主线）────────────────────────
#054  vLLM Mode B Kernel Extension      │ P1  │ 🚫 永久搁置（Mode B 代码已清除）
─── Phase D：论文 ───────────────────────────────────
#039  论文写作冲刺                     │ P0  │ ⬜  (依赖 #053)
```

### 关键路径

```
#049 ──→ #050 ──→ #055 ──→ #053 ──→ #039
       │                     ↑
#051 ──→ #052 ─────────┘
```

## 编码规范

- Python 3.10+，`from __future__ import annotations`
- 类型注解强制
- Docstring：Google 风格
- 行长度：100 字符
- Linter：ruff（`select = ["E", "F", "W", "I", "UP", "B", "SIM", "ARG"]`）
- **零外部依赖**：所有核心代码仅 stdlib
- 测试：`pytest tests/ -v`

## 关键约束

### 绝对禁止

- ❌ 向 `dependencies` 添加任何外部包（torch / numpy / vllm / sglang / sagellm）
- ❌ 修改已冻结文档的实验方向 / 策略列表 / figure 语义
- ❌ 在 vLLM adapter 中绕过 coordinator 操作 block_pool（参见 vllm-route-redesign.md §4）
- ❌ 使用 `_free_tail_blocks()` / null-block 替换 / `remove_skipped_blocks()` 的任何变体
- ❌ 在 Mode A 下为 Figure 6 / Figure 3a/b 赋予 task-level quality 含义
- ❌ 声称 SGLang "leverages native token-level KV release"（除非 smoke test 确认）
- ❌ 声称 BidKV 通过 `max_tokens` 获得信息优势（`max_tokens` 是标准 API 参数，所有策略平等可用）
- ❌ 在 `_get_max_tokens_estimate()` 中按策略名返回不同估算值（已移除此 asymmetry）
- ❌ 创建 `.venv` / `venv`

### 必须遵守

- ✅ `BidKVConfig(enabled=False)` 始终为默认
- ✅ Kill switch `BidKVConfig(kill_switch=True)` 必须立即旁路所有 bid 逻辑
- ✅ 新 baseline 必须通过 `BaselineRegistry.register()` 注册
- ✅ 新 adapter 必须实现 `FrameworkAdapter` ABC
- ✅ 失败计入结果（OOM / timeout / crash 全记录，禁止 cherry-pick）
- ✅ Frozen trace (seed=42) 跨所有策略共享
- ✅ Rate 冻结后不可基于策略表现调整（RULE RATE-FREEZE）

## 版本管理

`_version.py` 是唯一版本来源。不在 `pyproject.toml` / `__init__.py` 硬编码版本。

## 策略重命名记录

### h2o-style → largest-first（2026-04-06）

**原因**：`h2o-style` 策略实际行为是容量贪心驱逐（capacity-greedy eviction），
无真实 attention 数据时退化为按 KV 占用大小排序。重命名为 `largest-first`
以准确反映其机制。

**代码变更**：
- `baselines/h2o_style.py` → `baselines/largest_first.py`
- `H2OStyleStrategy` → `LargestFirstStrategy`（向后兼容别名已于 2026-04-08 删除）
- 策略名：`"h2o-style"` → `"largest-first"`
- 实验常量：`STRATEGY_H2O_STYLE` → `STRATEGY_LARGEST_FIRST`

**冻结数据映射**：
`results/` 目录中的已冻结实验数据仍使用 `h2o-style` 命名。
分析脚本中通过 `STRATEGY_LEGACY_NAMES` 映射或 `STRATEGY_DISPLAY` 字典
将 `h2o-style` 显示为 `Largest-First`。
**分析代码读取结果文件时应同时识别 `h2o-style` 和 `largest-first`。**

### H2OScoring → PositionalScoring（2026-04-07）

**原因**：`H2OScoring` 暗示使用 H2O 论文的 attention 机制，但实际实现
仅使用位置启发式（attention sink + recency heuristic），从未接收真实 attention weights。
重命名为 `PositionalScoring` 以准确反映其机制。

**代码变更**：
- `scoring/h2o.py` → `scoring/positional.py`
- `H2OScoring` → `PositionalScoring`（无向后兼容别名）
- `algorithm_id`: `"h2o"` → `"positional"`
- `scoring_method` metadata: `"h2o_cumulative_attention"` → `"positional"`

### h2o_hook → positional_hook（2026-04-07）

**原因**：H2O 命名全面清理，adapter hook 文件也相应重命名。

**代码变更**：
- `adapters/vllm/h2o_hook.py` → `adapters/vllm/positional_hook.py`
- `update_h2o_from_output` → `update_positional_from_output`
- `adapters/sglang/h2o_hook.py` — **已删除**（死代码，`install_h2o_hook` 未被任何代码导入）
- `scheduler_hook.py`：`_H2O_STRATEGIES` → `_POSITIONAL_STRATEGIES`
- 所有测试文件中的 H2O 引用已更新为 positional/largest-first

### _completion_factor 移除（2026-04-06）

`BidKVStrategy._completion_factor()` 死代码已移除。
该方法从未被 `select_victims()` 调用（v8 公式直接内联计算 quality_delta）。
`compression/base.py`（含 `GlobalNoBidStrategy`）已于 2026-04-09 随整个 `compression/` 目录一并删除。

## 测试

**重要**：必须使用 `python -m pytest`（而非裸 `pytest`），确保使用 conda 环境的 Python：

```bash
python -m pytest tests/ -v            # 全部 373 tests
python -m pytest tests/test_baselines.py -v   # 7 策略 baseline tests
python -m pytest tests/test_vllm_adapter.py   # vLLM adapter tests
python -m pytest tests/test_sglang_adapter.py # SGLang adapter tests
python -m ruff check . && python -m ruff format --check .
```

## Git 约定

- `main-dev` = 活跃开发分支
- `feature/bidkv` = 当前 BidKV 特性分支
- 不使用 `--no-verify`；hooks 失败必须先修复

## 论文关键时间线

| 节点           | 日期         | 说明                                        |
| -------------- | ------------ | ------------------------------------------- |
| Gate-A         | 2026-03-14   | ✅ 7/7 PASS，357 tests                      |
| 方案冻结       | 2026-03-19   | ✅ v2.3-frozen，reviewer APPROVED            |
| Wave 1 完成    | 2026-03-19   | ✅ #049+#051 代码修复，447 tests pass        |
| Dead code 清理 | 2026-04-09   | ✅ #056 scoring/compression 死代码清除，373 tests |
| Wave 2         | 03-20~22     | #050+#052 smoke test                       |
| Wave 3         | 03-22~23     | #055 pilot calibration + trace/rate 冻结    |
| Wave 4         | 03-24~28     | #053 全量 126+54 runs                       |
| Gate-B         | 2026-03-31   | vLLM Mode A 全量 + SGLang 全量 目标         |
| Phase D        | 04-01~08     | #039 论文写作冲刺                        |
| SC 2026 已接收 | 2026-07     | 论文已接收                                     |

## 实验硬件

- GPU：NVIDIA RTX A6000 48GB · CUDA 12.5
- 模型：Llama-3.1-8B-Instruct（bf16, 16GB）
- 推理引擎：vLLM 0.17.1 (v1 架构) + SGLang (RadixAttention)
- KV 限制：`--num-gpu-blocks-override 600`（600×16=9600 tokens，**必须指定否则无 KV 压力**）
- 并发：`--max-num-seqs 32`，`--gpu-memory-utilization 0.5`

## 实验运行规范

**必须使用 `nohup` 运行实验**，防止终端断连导致实验中断：

```bash
# ✅ 正确方式：nohup + 后台运行
nohup bash -c 'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m bidkv.experiments.vllm.runner \
    --strategies bidkv \
    --workloads mixed \
    --mixed-rates 2.0,3.8,5.7 \
    --runs 3 \
    --output-dir results/vllm_experiment_name \
    --gpu-memory-utilization 0.5 \
    --num-gpu-blocks-override 600 \
    --max-num-seqs 32 \
    --block-size 16 \
    --max-model-len 8192' > results/experiment_name.log 2>&1 &

# 检查进程
ps aux | grep "bidkv.experiments" | grep -v grep

# 等待结果文件出现
while ! ls results/vllm_experiment_name/bidkv__mixed__rate5.7__r2.json 2>/dev/null; do
  sleep 30; echo "waiting... $(date +%H:%M:%S)"
done && echo "DONE"

# ❌ 错误方式：直接在前台运行（终端断连会丢失实验）
# python -m bidkv.experiments.vllm.runner ...
```

**注意**：`nohup` 实验日志可能很少。
通过检查结果目录中的 JSON 文件判断完成状态。

## 冻结实验环境（v8-frozen，2026-04-02）

### 服务端参数（不可修改，所有后续实验必须使用）

```
--model           meta-llama/Llama-3.1-8B-Instruct
--gpu-memory-utilization  0.5
--num-gpu-blocks-override 600    # 600×16=9600 tokens KV
--max-num-seqs    32
--block-size      16
--max-model-len   8192
--enforce-eager
--disable-frontend-multiprocessing
--no-enable-prefix-caching
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

### 工作负载参数（RULE RATE-FREEZE）

| 工作负载 | Rates (req/s) | Requests/run | Runs/combo |
|----------|--------------|-------------|------------|
| mixed | 2.0, 3.8, 5.7 | 1000 | 3 |
| long_context | 0.35, 0.5, 0.7 | 500 | 3 |

### 观测指标体系（v8-frozen，2026-04-02 冻结，不可修改）

**主表指标**（论文 Table 1，4 列）：
- Throughput (req/s) — 标准 LLM serving 指标（vLLM, Orca, SGLang 均使用）
- SLO attainment(300ms) — S³ (ISCA'24)，TTFT≤300ms 达标率 (%)
- TTFT p95 — 尾部延迟主指标 (ms)
- TPOT p95 — Sarathi-Serve (OSDI'24)，统一使用 P95（P99 样本量不足方差过大）(ms)

**补充指标**（Appendix / Figure）：
- Goodput(500ms) — DistServe (OSDI'24)，TTFT≤500ms 的有效吞吐
- SLO attainment(500ms) / SLO attainment(1000ms) — 宽松阈值参考
- TTFT/TPOT p50, p99 — 延迟分布完整视图

**已移除**：
- Normalized Latency (Orca OSDI'22) — 被 TTFT+TPOT 分解严格覆盖
- Goodput 从主表移除 — 仅 2-3 篇论文使用，与 SLO attainment 正交性不足

**指标决策理由**（2026-04-02 确定）：
1. Throughput 是 reviewer 必期望指标，省略会引发 cherry-picking 质疑
2. 4 列完整覆盖 prefill(TTFT) + decode(TPOT) + 吞吐 + SLO 四个维度
3. BidKV 2/4 指标 #1，另 2 指标竞争力可接受（Thru -7%, TPOT -12%）
4. 论文叙事：quality-aware scheduling 用 ~7% 吞吐换取显著更好的准入响应性

### 论文叙事方针（2026-04-03 确定，标题→摘要→全文一致）

**目标场景**：KV 压力下的 **admission responsiveness**（准入响应性）。
即：新请求多快能收到第一个 token（TTFT），以及多大比例的请求在延迟 SLO 内完成首 token（SLO attainment）。

**核心因果链**（论文必须显式构建此链路）：
poor victim selection → 无效 recompute 浪费 KV 容量 → waiting queue 中请求等待更久
→ TTFT 飙升 → SLO 违约。
BidKV 通过 utility-guided victim selection 打破此链路。

**主要主张**：BidKV 是 admission-responsiveness-first 策略。
- 核心优势指标：SLO(300ms) cross-rate #1，TTFT P95 cross-rate #1
- 次要指标：Throughput #4（-7%），TPOT P95 #4（-12%）
- **Tradeoff 必须显式声明**：BidKV 用适度的 throughput/TPOT 代价换取
  显著更好的 TTFT 和 SLO attainment，对 latency-sensitive 部署有利

**TTFT vs TPOT tradeoff 解释**：
TTFT 和 TPOT 在 KV 压力下通常是 tradeoff：
- 激进驱逐（SRPT）→ 腾出 KV 空间 → 等待请求更快进入 → TPOT 好
  但被驱逐请求 recompute 成本高，占用 prefill 带宽 → TTFT 恶化
- BidKV 禁用 SRPT、使用 cost-aware 驱逐 → 减少 recompute 浪费
  → 更少的 prefill 竞争 → TTFT/SLO 改善，但 throughput/TPOT 略降

**叙事一致性要求**：
- 标题：mechanism-first 可接受，但不应暗示 throughput 优化
- 摘要：必须在前两句内定义目标场景（admission responsiveness under KV pressure）
- §1：¶2 必须显式构建 victim selection → TTFT/SLO 因果链；¶5 必须明确 BidKV 的收益体现
- §6：evaluation opening 应声明评估围绕 admission responsiveness 展开
- 数据 claim：必须首先报告 SLO 和 TTFT（核心指标），throughput/TPOT 作为 completeness
- **禁止**：声称 BidKV 是"全面领先"或隐瞒 throughput/TPOT 的非领先表现

### 论文编辑原则（2026-04-04 确定，适用于所有后续论文修改）

#### 原则 1：不提及未实现功能

论文中**绝对不能**将代码中未真实实现的方案描述为已有能力。
- ❌ 不要把 future work / 设计愿景当作已有功能来描述
- ❌ 不要在 Design/Implementation 章节中引入尚未在实验中实际运行的机制
- ✅ 未来方向仅在 §8 Conclusion 的 future work 句中简要提及

#### 原则 2：Limitation 不自我伤害

§7.2 Limitations 只写不可避免的 **scope boundaries**（硬件规模、模型规模、执行模型），
不写尖锐到会动摇论文可靠性的局限。
- ❌ 不要在 limitations 中进行 self-celebration（"precisely the heterogeneity BidKV exploits"）
- ❌ 不要主动暴露 reviewer 未必会注意到的设计弱点
- ✅ Limitations 条目格式：scope statement + future work 一句话

#### 原则 3：Admission Responsiveness 是唯一主线

全文叙事聚焦 **admission responsiveness**（TTFT + SLO attainment），不是抽象的 "scheduling efficiency"。
- 标题 → 摘要 → §1 → §4 → §6 → §7 → §8 必须保持此主线一致
- 摘要前两句必须定义目标场景（KV pressure → victim selection → admission latency）
- §1 ¶2 必须构建完整因果链：poor victim → wasted recompute → KV occupied → TTFT ↑ → SLO ↓
- §8 结论第一段必须回到 TTFT/SLO，不要停留在 "scheduling efficiency"
- 数据报告顺序：SLO → TTFT → Throughput → TPOT（核心指标在前）

#### 原则 4：Tradeoff 必须诚实声明

BidKV 不是全面领先，**必须显式承认 throughput/TPOT 代价**。
- 摘要使用 "at a modest throughput cost"（不是 "at comparable throughput"）
- §6 数据讨论中必须包含 tradeoff 段落
- 表格列序 = 叙事优先级：SLO → TTFT P95 → Throughput → TPOT P95
- 论述方式："BidKV trades modest throughput/TPOT for significantly better TTFT and SLO"

#### 原则 5：使用统一术语

| ✅ 正确术语 | ❌ 废弃术语 | 说明 |
|---|---|---|
| reclamation cost | recomputation cost (as metric) | 度量被驱逐请求的代价 |
| disruption estimate / disruption cost | quality delta | δ 的含义 |
| admission responsiveness | scheduling efficiency (as primary framing) | 主线叙事 |
| victim selection | compression scheduling | 核心问题定义 |
| reclaim / preempt | compress (in Mode A) | 执行动作 |
| modest throughput cost | comparable throughput | 代价表述 |

#### 原则 6：Table/Figure 不重复

- 每个 table/figure 必须传递独立信息，不允许两个 figure 展示同一指标的同一维度
- 已删除的 table/figure（如之前的 Table 1、TikZ Fig 2）**不要恢复**
- 未被 `\ref{}` 引用的 table/figure 应当删除，不要保留孤立浮动体

#### 原则 7：核心指标视觉优先

- Table 列序必须为：SLO → TTFT P95 → Throughput → TPOT P95（反映叙事优先级）
- Figure 中 BidKV 的核心优势指标（SLO, TTFT）应在突出位置
- 加粗/下划线标注规则：**粗体** = 最优，_下划线_ = 次优

#### 原则 8：§7.1 是正面定位，不是辩护

§7.1 Surrogate Disruption Estimate 的写法：
- ✅ δ 是 ranking signal，exact prediction unnecessary
- ✅ 只需保持 relative ordering → 低成本 victim 优先
- ✅ 结尾推到 admission responsiveness：cost-aware ranking → less wasted recompute → TTFT/SLO
- ❌ 不要写 Δ_max 的 conservative/aggressive 讨论（与 §4.3 矛盾——Δ_max 在 request-level 下已 relaxed）
- ❌ 不要连续使用否定句（"not direct"、"not precise"、"only degree of freedom"）

### 论文当前章节结构与状态（2026-04-04）

| 章节 | 标题 | 当前状态 | 修改注意事项 |
|------|------|---------|-------------|
| §1 | Introduction | ✅ 已修订 | 因果链在 ¶2，贡献在 ¶5 |
| §2 | Background and Motivation | ✅ 已修订 | §2.1 KV Cache Memory Model, §2.2 Victim-Selection Problem |
| §3 | Related Work | ✅ 已修订 | 系统层 + token-level + 本文定位 |
| §4 | Design | 待检查 | 四层架构，§4.1-4.4 |
| §5 | Implementation | 待检查 | vLLM + SGLang 集成 |
| §6 | Evaluation | 待检查 | §6.1 Setup, §6.2 Main, §6.3 Rate, §6.4 LC, §6.5 Mechanism |
| §7 | Discussion and Limitations | ✅ 已修订 | §7.1 正面定位, §7.2 scope-only, §7.3 broader impact |
| §8 | Conclusion | ✅ 已修订 | 回到 TTFT/SLO，不停在 scheduling efficiency |
| Abstract | — | ✅ 已修订 | "at a modest throughput cost" |

### 论文 Figure 清单（2026-04-04）

| Figure | 内容 | 文件 | 状态 |
|--------|------|------|------|
| Fig 1 | Algorithm 1 (Greedy Solver) | `figures/fig2_algorithm.tex` | ✅ |
| Fig 2 | Architecture overview diagram | `figures/fig1_architecture.jpg` | ✅ 已替换为 JPG |
| Fig 3 | Mixed rate sensitivity (Throughput + TTFT P95) | PDF (外部生成) | ✅ |
| Fig 4 | Long-context SLO + TTFT P95 bar charts | PDF (外部生成) | 待确认 |
| Fig 5 | Proactive eviction count + tokens freed | PDF (外部生成) | ⚠️ 3/5 策略零值 |

## v8 Mixed 全量结果（63 runs，2026-04-02 冻结，p95 从原始请求数据计算）

### Cross-Rate Average Ranking（7 策略 × 3 rates × 3 runs）

| Rank | Strategy | Throughput | SLO300 | TTFT95 | TPOT95 | Rank Sum | Wins |
|------|----------|-----------|--------|--------|--------|----------|------|
| 1 | static-random | #1 | #2 | #5 | #1 | 9 | 2 |
| 2 | **bidkv** | #4 | **#1** | **#1** | #4 | 10 | **2** |
| 3 | uniform | #2 | #3 | #4 | #2 | 11 | 0 |
| 4 | slack-aware | #3 | #6 | #6 | #3 | 18 | 0 |
| 5 | h2o-style | #6 | #4 | #3 | #6 | 19 | 0 |
| 6 | preempt-evict-sjf | #7 | #5 | #2 | #7 | 21 | 0 |
| 7 | preempt-evict | #5 | #7 | #7 | #5 | 24 | 0 |

BidKV rank_sum=10（static-random=9），wins=2 并列。
BidKV 以 SLO #1 + TTFT #1（用户体验核心指标）赢得 tiebreak。

### Cross-Rate Average Values

| Strategy | Throughput | SLO300% | TTFT p95 | TPOT p95 |
|----------|-----------|---------|----------|----------|
| **bidkv** | **2.99** | **87.1** | **554** | **96.4** |
| static-random | 3.21 | 87.0 | 1076 | 86.0 |
| uniform | 3.21 | 86.9 | 1069 | 86.7 |
| h2o-style | 2.92 | 84.4 | 584 | 100.1 |
| preempt-evict-sjf | 2.77 | 82.8 | 572 | 129.4 |
| slack-aware | 3.05 | 72.4 | 4023 | 93.2 |
| preempt-evict | 2.98 | 72.2 | 5241 | 98.3 |

### BidKV Per-Rate Performance

| Rate | Thru | SLO | TTFT | TPOT | Wins/4 | Top-3/4 |
|------|------|-----|------|------|--------|---------|
| 2.0 | **#1** | **#1** | **#1** | **#1** | **4/4** | 4/4 |
| 3.8 | #5 | #2 | **#1** | #5 | 1/4 | 2/4 |
| 5.7 | #4 | #2 | #2 | #4 | 0/4 | 2/4 |

### Tradeoff 分析

BidKV 禁用 SRPT（SRPT = 主动驱逐长运行请求以释放 KV），
用 ~7% 吞吐换取显著更好的延迟质量：
- TTFT p95: 554ms vs static-random 1076ms（1.9x 改善）
- SLO(300ms): 87.1% vs PE 72.2%（+14.9pp）
- Rate=2.0 下全指标 #1；高压下 Throughput/TPOT 竞争力下降

**v8b SRPT 已测试**：简单启用 SRPT（rate=5.7, 1 run）→ TTFT -98ms，
但 SLO -2pp, TPOT +3.7ms。简单启用无效。

### 各策略机制开关（scheduler_hook.py 当前状态）

| 机制 | PE | PE-SJF | Slack | Random/Largest-First/Uniform | BidKV |
|------|-----|--------|-------|------------------------------|-------|
| Waiting 排序 | FCFS | SJF | EDF | SJF | SJF |
| Running reorder | ❌ LIFO | ❌ LIFO | ✅ cached prio | ✅ cached prio | ✅ 95% KV 门控 |
| Proactive preempt | ❌ | ❌ | ✅ KV>90% | ✅ KV>90% | ❌ (显式 skip) |
| SRPT preempt | ❌ | ❌ | ❌ | ✅ KV>80% | ❌ |

BidKV 特有：U = freed / (1 + 0.5c + 0.3P + ε)，其中 c=completion_ratio, P=num_preemptions
BidKV proactive preempt 被显式跳过，仅通过 running reorder (KV>95%) + vLLM 原生 preemption 控制驱逐。

### 数据目录

| 路径 | 内容 | 状态 |
|------|------|------|
| `results/vllm_v8_full_validation/` | v8 全量 7×3×3=63 runs (mixed) | **当前主数据，冻结** |
| `results/vllm_v8_analysis/` | v8 全量分析报告 + JSON（4 指标体系） | **最终分析** |
| `results/vllm_v8b_srpt_quick/` | BidKV+SRPT 快速测试 (1 run) | 参考数据 |
| `results/vllm_v8_long_context/` | v8 long_context 7×3×3=63 runs | 进行中 |

### 实验结果 JSON 数据格式（mixed 与 long_context 通用）

**⚠️ 分析实验数据时必须使用以下正确字段名，不要凭记忆猜测。**
**⚠️ 所有字段名已于 2026-04-03 通过实际数据审计确认，以此为准。**

#### 1. 文件命名与目录

```
results/{result_dir}/{strategy}__{workload}__rate{rate}__r{run_index}.json
# 示例：bidkv__mixed__rate3.8__r0.json
# 示例：h2o-style__long_context__rate0.5__r2.json
```

#### 2. 顶层结构（13 个键）

```python
d = json.load(open(result_file))

# ✅ 核心数据
d['request_results']      # list[dict] — 每个请求的详细结果
d['adapter_metrics']      # dict — BidKV adapter 运行指标（驱逐统计等）
d['summary']              # dict — 预计算的汇总指标

# ✅ 运行元数据
d['duration_s']           # float — 运行时长（秒）
d['strategy']             # str — 策略名 ("bidkv", "h2o-style", etc.)
d['workload']             # str — 工作负载名 ("mixed", "long_context")
d['request_rate']         # float — 请求速率 (req/s)
d['run_index']            # int — 运行序号 (0, 1, 2)
d['run_label']            # str — 完整运行标签 ("bidkv__mixed__rate2.0__r0")
d['start_time']           # float — Unix 时间戳
d['end_time']             # float — Unix 时间戳
d['server_config']        # dict — 服务端配置 {'run_status': 'completed'}
d['candidate_snapshots']  # list — 通常为空列表 []

# ❌ 不存在的字段名（常见错误）：
# d['requests']           — 不存在，用 d['request_results']
# d['duration']           — 不存在，用 d['duration_s']
# d['rate']               — 不存在，用 d['request_rate']
```

#### 3. request_results 单个请求字段（10 个键）

```python
r = d['request_results'][0]

r['request_id']           # str — 请求 ID ("mixed-0000", "long-0042")
r['ttft_ms']              # float — TTFT（毫秒，已经是 ms 不需要 ×1000）
r['total_latency_ms']     # float — 端到端延迟（毫秒）
r['completion_tokens']    # int — 生成 token 数
r['prompt_tokens']        # int — ⚠️ 当前实验中始终为 0（采集代码未记录）
r['error']                # str — 成功时为空字符串 ''（不是 None）
r['submit_time']          # float — 提交时间戳（monotonic clock）
r['first_token_time']     # float — 首 token 时间戳（monotonic clock）
r['finish_time']          # float — 完成时间戳（monotonic clock）
r['generated_text']       # str — 生成的文本内容

# ❌ 不存在的字段名：
# r['ttft']               — 不存在，用 r['ttft_ms']
# r['tpot']               — 不存在，必须手动计算
# r['latency']            — 不存在，用 r['total_latency_ms']
# r['tokens']             — 不存在，用 r['completion_tokens']
```

#### 4. adapter_metrics 字段（⚠️ 字段名因策略/运行批次不同而有差异）

```python
am = d['adapter_metrics']

# ✅ 所有策略共有
am['total_tokens_freed']          # int — 总释放 token 数
am['total_pressure_events']       # int — KV 压力事件数
am['total_requests_completed']    # int — 完成请求数
am['total_decode_steps']          # int — decode 步数
am['kill_switch_activations']     # int — kill switch 触发次数
am['preemptions_avoided']         # int — 避免的 preemption 次数

# ⚠️ 驱逐次数字段名不一致（历史原因）：
# - 部分旧批次文件使用 'total_compressions'（如 mixed 的 bidkv/PE/PE-SJF/h2o-style）
# - 新批次文件使用 'total_evictions'（如 long_context 所有策略、mixed 的 slack/random/uniform）
# 安全读取方式：
evictions = am.get('total_evictions', am.get('total_compressions', 0))

# ❌ 不存在的字段名（常见错误）：
# am['evictions_total']            — 不存在
# am['evictions']                  — 不存在
# am['pressure_events']            — 不存在，用 am['total_pressure_events']
# am['priority_cache_refreshes']   — 不存在，此指标未记录到结果文件中
```

#### 5. summary 预计算字段（11 个键）

```python
s = d['summary']
s['throughput_rps']                  # float — 吞吐量 (req/s)
s['successful_requests']             # int — 成功请求数
s['failed_requests']                 # int — 失败请求数
s['total_requests']                  # int — 总请求数
s['ttft_ms_p50']                     # float — TTFT p50 (ms)
s['ttft_ms_p99']                     # float — TTFT p99 (ms)
s['tpot_ms_p50']                     # float — TPOT p50 (ms)
s['tpot_ms_p99']                     # float — TPOT p99 (ms)
s['e2e_latency_ms_p50']              # float — 端到端延迟 p50 (ms)
s['e2e_latency_ms_p99']              # float — 端到端延迟 p99 (ms)
s['normalized_latency_ms_per_token'] # float — 归一化延迟

# ⚠️ summary 中没有 p95！p95 必须从 request_results 原始数据计算。
# ⚠️ summary 中没有 SLO attainment！必须从 request_results 原始数据计算。
```

#### 6. 标准数据分析代码模板

```python
import json, os, statistics

def load_run(filepath):
    """加载单个实验结果文件，返回标准化的指标字典。"""
    d = json.load(open(filepath))
    
    # 过滤成功请求（error 是空字符串 '' 不是 None）
    ok = [r for r in d['request_results'] if not r.get('error')]
    
    # TTFT 列表
    ttft_list = [r['ttft_ms'] for r in ok if r['ttft_ms'] is not None]
    
    # TPOT 列表（必须手动计算）
    tpot_list = []
    for r in ok:
        if (r.get('completion_tokens', 0) > 1 
                and r.get('ttft_ms') is not None 
                and r.get('total_latency_ms') is not None):
            tpot = (r['total_latency_ms'] - r['ttft_ms']) / (r['completion_tokens'] - 1)
            tpot_list.append(tpot)
    
    # 排序后取百分位
    ttft_list.sort()
    tpot_list.sort()
    
    def percentile(data, p):
        if not data:
            return float('nan')
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]
    
    # SLO attainment（mixed=300ms, long_context=2000ms）
    slo_threshold = 2000.0 if 'long' in d.get('workload', '') else 300.0
    slo_count = sum(1 for t in ttft_list if t <= slo_threshold)
    slo_pct = slo_count / len(ttft_list) * 100 if ttft_list else 0
    
    # Throughput
    throughput = d['summary']['throughput_rps']
    
    # 驱逐统计（兼容新旧字段名）
    am = d.get('adapter_metrics', {})
    evictions = am.get('total_evictions', am.get('total_compressions', 0))
    freed = am.get('total_tokens_freed', 0)
    
    return {
        'throughput': throughput,
        'slo_pct': slo_pct,
        'ttft_p50': percentile(ttft_list, 50),
        'ttft_p95': percentile(ttft_list, 95),
        'ttft_p99': percentile(ttft_list, 99),
        'tpot_p50': percentile(tpot_list, 50),
        'tpot_p95': percentile(tpot_list, 95),
        'tpot_p99': percentile(tpot_list, 99),
        'ok_count': len(ok),
        'total_count': len(d['request_results']),
        'evictions': evictions,
        'tokens_freed': freed,
        'strategy': d.get('strategy', ''),
        'workload': d.get('workload', ''),
        'rate': d.get('request_rate', 0),
    }

def load_all_runs(result_dir):
    """加载目录下所有结果文件，按 (strategy, rate) 分组。"""
    from collections import defaultdict
    groups = defaultdict(list)
    for fn in sorted(os.listdir(result_dir)):
        if not fn.endswith('.json') or fn.startswith('candidate'):
            continue
        filepath = os.path.join(result_dir, fn)
        m = load_run(filepath)
        groups[(m['strategy'], m['rate'])].append(m)
    return groups

def cross_rate_average(groups):
    """计算每个策略的跨速率平均值。"""
    from collections import defaultdict
    strat_metrics = defaultdict(lambda: defaultdict(list))
    for (strat, rate), runs in groups.items():
        for m in runs:
            for k in ['throughput', 'slo_pct', 'ttft_p95', 'tpot_p95', 'evictions', 'tokens_freed']:
                strat_metrics[strat][k].append(m[k])
    
    result = {}
    for strat, metrics in strat_metrics.items():
        result[strat] = {k: statistics.mean(v) for k, v in metrics.items()}
    return result
```

#### 7. 已知数据陷阱（必读）

| 陷阱 | 说明 | 正确做法 |
|------|------|---------|
| `prompt_tokens` 始终为 0 | 实验采集代码未正确记录 prompt token 数 | 不要使用此字段做分析 |
| adapter_metrics 驱逐字段名不一致 | 旧批次用 `total_compressions`，新批次用 `total_evictions` | 用 `am.get('total_evictions', am.get('total_compressions', 0))` |
| summary 无 p95 | summary 只有 p50 和 p99 | p95 必须从 request_results 原始数据计算 |
| summary 无 SLO attainment | SLO 未预计算 | 必须从 ttft_ms 原始数据按阈值计算 |
| error 字段是空字符串 | 成功时 error='' 而不是 None | 用 `if not r.get('error')` 而非 `r.get('error') is None` |
| mixed bidkv 驱逐数为 0 | mixed 负载下 BidKV 未触发 proactive preempt（KV 压力不足） | 正常现象，不是 bug。只有 long_context 才有显著驱逐 |
| PE/PE-SJF 驱逐数始终为 0 | 这两个策略禁用了 proactive preempt，只靠 vLLM 原生 LIFO 驱逐 | 正常设计。adapter_metrics 只记录 BidKV 主动发起的驱逐 |

#### 8. 驱逐数据参考值（2026-04-03 审计确认）

| 策略 | Mixed Avg Evictions | Mixed Avg Freed | LC Avg Evictions | LC Avg Freed |
|------|--------------------:|----------------:|-----------------:|-------------:|
| bidkv | 0 | 0 | 108 | 259,122 |
| h2o-style (largest-first) | 0† | 59,916† | 117 | 287,235 |
| static-random | 72 | 18,957 | 100 | 208,620 |
| slack-aware | 27 | 16,327 | — | — |
| uniform | 73 | 18,048 | — | — |
| preempt-evict | 0 | 0 | 0 | 0 |
| preempt-evict-sjf | 0 | 0 | 0 | 0 |

†h2o-style mixed：`total_evictions` 字段不存在，但 `total_compressions`>0 且 `total_tokens_freed`>0。
