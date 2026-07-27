---
name: bidkv
description: "BidKV 仓库开发助手 — framework-portable KV cache 请求调度原语。用于实现 BidKV solver、baseline 策略、vLLM/SGLang 适配器、实验框架。零外部依赖，SC 2026 已接收论文 artifact。"
argument-hint: "描述 BidKV 相关任务：baseline 策略实现、adapter 开发、实验运行、solver 修改、冻结方案约束检查等。"
tools: ['vscode', 'execute', 'read', 'agent', 'edit', 'search', 'web', 'todo']
---

# BidKV Copilot Agent

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
│   ├── attention.py         # Attention-weight scoring
│   ├── h2o.py               # H2O decode-step scoring
│   ├── random_score.py
│   └── uniform.py
├── pool/                    # BidPoolManager
│   └── manager.py
├── pressure/                # PressureDetector（KV 压力检测）
│   ├── config.py
│   └── detector.py
├── compression/             # CompressionExecutor
│   └── base.py
├── solver/                  # GreedyBidSolver（bid 排序 + 贪心选取）
│   ├── config.py
│   ├── execution_result.py
│   └── greedy.py
├── baselines/               # 7 策略（6 baseline + BidKV）
│   ├── base.py              # BaselineStrategy ABC, RequestState, BaselineContext
│   ├── registry.py          # BaselineRegistry（名称→策略实例）
│   ├── preempt_evict.py     # 1. vLLM 原生 baseline (FCFS+LIFO)
│   ├── preempt_evict_sjf.py # 2. PE + SJF admission
│   ├── static_random.py     # 3. 随机驱逐
│   ├── h2o_style.py         # 4. attention-based heuristic
│   ├── uniform.py           # 5. 均等驱逐
│   ├── slack_aware.py       # 6. SLO-deadline aware
│   └── bidkv_strategy.py    # 7. BidKV 完整系统
├── adapters/                # 框架适配器
│   ├── base.py              # FrameworkAdapter ABC（5 层职责边界）
│   ├── vllm/                # vLLM v1 adapter
│   │   ├── adapter.py       # VLLMAdapter
│   │   ├── plugin.py        # monkey-patch EngineCore.__init__
│   │   ├── scheduler_hook.py
│   │   └── h2o_hook.py
│   └── sglang/              # SGLang adapter
│       ├── adapter.py       # SGLangAdapter
│       ├── scheduler_hook.py # monkey-patch get_next_batch_to_run()
│       ├── radix_hook.py    # token-level KV release via RadixAttention
│       └── h2o_hook.py
└── experiments/             # 实验框架
    ├── metrics.py           # 实验指标采集
    ├── common/              # 共享：trace, audit, runner, report
    ├── vllm/                # vLLM 实验：runner, serve, config, analysis, etc.
    └── sglang/              # SGLang 实验：runner, server, config, analysis, etc.
```

总量：~12,000 LOC Python，447+ tests。

## 关键设计模式

### 1. BaselineRegistry — 策略注册表
所有 7 个策略通过 `BaselineRegistry.register()` 注册，通过名称获取。
禁止在 registry 中硬编码策略类。新策略必须实现 `BaselineStrategy` ABC。

### 2. FrameworkAdapter ABC — 跨框架适配
职责：KV stats 获取 → 策略注入（谁被 preempt）→ Lifecycle 管理。
在 vLLM Mode A 中，adapter 仅做请求排序，不执行 token-level 操作。
每个被支持的 serving framework 实现一个 adapter。

注：adapter.py 中的 execute_compression() / try_compress() 等方法已标记为
**DEPRECATED (Mode B)** — 在 Mode A 实验中为死代码。

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

| 层面               | PE            | PE-SJF        | Slack       | Random/H2O/Uniform | BidKV                          |
| ------------------ | ------------- | ------------- | ----------- | ------------------- | ------------------------------ |
| Waiting 排序       | FCFS (无排序) | SJF           | EDF (到达序) | SJF (prompt_tokens) | SJF (prompt_tokens)            |
| Running 排序       | LIFO (无排序) | LIFO (无排序) | cached prio | cached prio         | 95% KV 门控 + avg_prompt≤500  |
| select_victims()   | N/A           | N/A           | slack-based | 各自启发式          | **U = r/(δ+ε)** 质量感知      |
| SRPT 主动驱逐      | ❌            | ❌            | ❌          | ✅ (同等估算)       | ❌ (recompute 成本过高)        |
| Proactive preempt  | ❌            | ❌            | ✅          | ✅                  | ✅                             |

BidKV 的唯一分化点：`select_victims()` 中使用完整 scoring→bid→pool→solver
管线计算 **U = tokens_freed / (quality_delta + ε)**，实现质量感知的 preemption 排序。

### Mode B 状态（已废弃）

adapter.py 中存在 Mode B 代码（`execute_compression()` → `_execute_tail_truncation()`），
`truncation_hook.py` 提供 token-level KV block 截断基础设施。
这些代码已全部标记为 **DEPRECATED (Mode B)**，在当前实验中为死代码，
保留用于潜在的 Mode B 未来扩展（issue #054）。

### SGLang 3 策略

Preempt-Evict (Default) → Slack-Aware → BidKV

方向一致性（Directional Consistency）：
- DC-1a: BidKV ≥ Preempt-Evict
- DC-1b: BidKV ≥ Slack-Aware

**SGLang 执行语义设计问题**：

SGLang 原生驱逐机制是 `RadixCache.evict()`，本质上做的是 **请求级 LRU 驱逐**
（驱逐整个 radix tree 节点），而不是在一个 running 请求中间释放部分 token。

当前 SGLang adapter 代码（`execute_compression()` → `radix_hook.free_kv_positions()`）
实现的是 **token-level 部分释放**——这比 SGLang 原生行为更激进，
且与 vLLM Mode A 的 request-level 调度语义不一致。

双平台一致性目标：SGLang 也应走 request-level 调度（控制 WHO gets evicted），
然后让 SGLang 原生 `RadixCache.evict()` 执行驱逐。这与 vLLM 完全对称。

当前状态：smoke test (#052) 中 KV 压力从未触发，token-level 路径从未实际执行。
SGLang adapter 的 token-level 代码有待重构为 request-level 语义，
或在全量实验中确认后再决定。

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
- `adapters/vllm/adapter.py`：`execute_compression()` 路由到 `_execute_tail_truncation()`（**已标记为 DEPRECATED Mode B**）
- `scheduler_hook.py` 仅做请求排序 + `_preempt_request()`（调度语义）
- `execute_abort()` 调用 `scheduler.abort_requests()`（**也已 DEPRECATED**）
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
─── Wave 4：全量实验 ─────────────────────────────────
#053  全量 126+54 runs + Figure/Table   │ P0  │ ⬜  (依赖 #055)
─── 独立增强（不阻塞主线）────────────────────────
#054  vLLM Mode B Kernel Extension      │ P1  │ ⬜  (可任何时候开始)
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
- ❌ 创建总结性文档（INTEGRATION.md、SUMMARY.md 等）

### 必须遵守

- ✅ `BidKVConfig(enabled=False)` 始终为默认
- ✅ Kill switch `BidKVConfig(kill_switch=True)` 必须立即旁路所有 bid 逻辑
- ✅ 新 baseline 必须通过 `BaselineRegistry.register()` 注册
- ✅ 新 adapter 必须实现 `FrameworkAdapter` ABC
- ✅ 失败计入结果（OOM / timeout / crash 全记录，禁止 cherry-pick）
- ✅ Frozen trace (seed=42) 跨所有策略共享
- ✅ Rate 冻结后不可基于策略表现调整（RULE RATE-FREEZE）
- ✅ 每次改动必须更新 `CHANGELOG.md`

## 版本管理

`_version.py` 是唯一版本来源。不在 `pyproject.toml` / `__init__.py` 硬编码版本。

## 测试

**重要**：必须使用 `python -m pytest`（而非裸 `pytest`），确保使用 conda 环境的 Python：

```bash
python -m pytest tests/ -v            # 全部 447+ tests
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
| Wave 2         | 03-20~22     | #050+#052 smoke test                       |
| Wave 3         | 03-22~23     | #055 pilot calibration + trace/rate 冻结    |
| Wave 4         | 03-24~28     | #053 全量 126+54 runs                       |
| Gate-B         | 2026-03-31   | vLLM Mode A 全量 + SGLang 全量 目标         |
| Phase D        | 04-01~08     | #039 论文写作冲刺                        |
| SC 2026 已接收 | 2026-07     | 论文已接收                                     |

## 实验结果 JSON 数据格式（mixed 与 long_context 通用）

**⚠️ 分析实验数据时必须使用以下正确字段名，不要凭记忆猜测。**
**⚠️ 所有字段名已于 2026-04-03 通过实际数据审计确认，以此为准。**

### 1. 文件命名与目录

```
results/{result_dir}/{strategy}__{workload}__rate{rate}__r{run_index}.json
# 示例：bidkv__mixed__rate3.8__r0.json
# 示例：h2o-style__long_context__rate0.5__r2.json
```

### 2. 顶层结构（13 个键）

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

### 3. request_results 单个请求字段（10 个键）

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

### 4. adapter_metrics 字段（⚠️ 字段名因策略/运行批次不同而有差异）

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

### 5. summary 预计算字段（11 个键）

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

### 6. 标准数据分析代码模板

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

### 7. 已知数据陷阱（必读）

| 陷阱 | 说明 | 正确做法 |
|------|------|---------|
| `prompt_tokens` 始终为 0 | 实验采集代码未正确记录 prompt token 数 | 不要使用此字段做分析 |
| adapter_metrics 驱逐字段名不一致 | 旧批次用 `total_compressions`，新批次用 `total_evictions` | 用 `am.get('total_evictions', am.get('total_compressions', 0))` |
| summary 无 p95 | summary 只有 p50 和 p99 | p95 必须从 request_results 原始数据计算 |
| summary 无 SLO attainment | SLO 未预计算 | 必须从 ttft_ms 原始数据按阈值计算 |
| error 字段是空字符串 | 成功时 error='' 而不是 None | 用 `if not r.get('error')` 而非 `r.get('error') is None` |
| mixed bidkv 驱逐数为 0 | mixed 负载下 BidKV 未触发 proactive preempt（KV 压力不足） | 正常现象，不是 bug。只有 long_context 才有显著驱逐 |
| PE/PE-SJF 驱逐数始终为 0 | 这两个策略禁用了 proactive preempt，只靠 vLLM 原生 LIFO 驱逐 | 正常设计。adapter_metrics 只记录 BidKV 主动发起的驱逐 |

### 8. 驱逐数据参考值（2026-04-03 审计确认）

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

## 实验硬件

- GPU：NVIDIA RTX A6000 48GB · CUDA 12.5
- 模型：Llama-3.1-8B-Instruct（bf16, 16GB）
- 推理引擎：vLLM 0.17.1 (v1 架构) + SGLang (RadixAttention)

## 🚫 NEVER_CREATE_DOT_VENV_MANDATORY

- 永远不要创建 `.venv` 或 `venv`（无任何例外）。
- 必须复用当前已配置的非-venv Python 环境（如现有 conda 环境）。
