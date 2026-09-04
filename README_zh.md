# BidKV

跨框架可移植的 KV cache 请求调度原语。

[English](README.md)

## 概述

`bidkv` 是一个**零外部依赖**的独立 Python 包，解决 KV cache 压力下的**受害者选择**问题：当 KV 空间不足时，应该 preempt 哪个请求？

核心思路：驱逐"每单位质量损失能释放最多 KV 空间"的请求，即最大化 utility：

$$U(r, \delta) = \frac{r}{\delta + \varepsilon}, \quad \varepsilon = 10^{-3}$$

其中 $r$ = 可释放 token 数，$\delta$ = surrogate 扰动代价估计。

BidKV **不执行压缩**——它只控制"谁被 preempt"，底层执行仍是框架原生的 preempt + recompute（vLLM）或 RadixCache 驱逐（SGLang）。

## 模块结构

| 模块 | 内容 |
|------|------|
| `protocol/` | 核心类型：`CompressionBid`, `BidPool`, `BidAcceptance` |
| `scoring/` | `PositionalScoring`（attention sink + recency 位置启发式） |
| `pool/` | `BidPoolManager` |
| `pressure/` | `PressureDetector`（KV 压力检测） |
| `solver/` | `GreedyBidSolver`（bid 排序 + 贪心选取） |
| `baselines/` | 6 个 baseline 策略 + BidKV（见下） |
| `adapters/vllm/` | vLLM v1 适配器（scheduler hook + plugin） |
| `adapters/sglang/` | SGLang 适配器（scheduler hook） |
| `experiments/` | 实验运行器、采集器、分析脚本 |

## Baseline 策略

| 策略名 | 类 | 调度逻辑 |
|--------|----|----------|
| `preempt-evict` | `PreemptEvictStrategy` | vLLM 原生：FCFS admission + LIFO 驱逐 |
| `preempt-evict-sjf` | `PreemptEvictSJFStrategy` | SJF admission + LIFO 驱逐 |
| `static-random` | `StaticRandomStrategy` | 随机受害者选择 |
| `largest-first` | `LargestFirstStrategy` | 容量贪心：优先驱逐 KV 占用最大的请求 |
| `bidkv` | `BidKVStrategy` | 质量感知：最大化 U = r / (δ + ε) |

## 配置

```python
from bidkv import BidKVConfig

# 默认：所有 bid 逻辑旁路（import 安全，不影响服务）
config = BidKVConfig(enabled=False)

# 启用 BidKV 调度
config = BidKVConfig(enabled=True)
assert config.is_active

# Kill switch：即使 enabled=True 也立即旁路所有逻辑
config = BidKVConfig(enabled=True, kill_switch=True)
assert not config.is_active
```

## 添加自定义策略

```python
from bidkv import (
    BaselineRegistry,
    BidKVStrategy,
    PreemptEvictStrategy, LargestFirstStrategy,
    StaticRandomStrategy, PreemptEvictSJFStrategy,
)

# 注册全部内置策略
registry = BaselineRegistry()
registry.create_default_registry()

# 或按需注册
registry2 = BaselineRegistry()
registry2.register(BidKVStrategy())
registry2.register(PreemptEvictStrategy())

strategy = registry2.get("bidkv")
print(strategy.name)              # "bidkv"
print(registry2.list_strategies())  # ["bidkv", "preempt-evict"]
```

## 运行实验

```bash
# vLLM：5 策略 × mixed 工作负载 × 3 速率 × 3 runs
HF_HUB_OFFLINE=1 python -m bidkv.experiments.vllm.runner \
    --strategies "preempt-evict,preempt-evict-sjf,static-random,largest-first,bidkv" \
    --workloads mixed \
    --mixed-rates 2.0,3.8,5.7 \
    --runs 3 \
    --output-dir results/vllm_experiment \
    --gpu-memory-utilization 0.5 \
    --num-gpu-blocks-override 600 \
    --max-num-seqs 32

# SGLang：3 策略
HF_HUB_OFFLINE=1 python -m bidkv.experiments.sglang.runner \
    --strategies "sglang_default,slack_aware,bidkv" \
    --workloads mixed \
    --runs 3 \
    --output-dir results/sglang_experiment
```

## 框架集成（vLLM）

Sage Mate 固定目标 vLLM-HUST `762f85b3`（`0.28.1rc1.dev319`）提供不可变的
`vllm.preemption-policy.v1` 契约。BidKV 只提供 `BidkvPreemptionPolicy`；请求状态、
抢占、KV 清理、重新入队以及调度预算回滚仍由 vLLM 管理。正式路径不 monkey patch
`Scheduler`。

```bash
pip install vllm-hust-ext bidkv
vllm-hust-ext extension enable org.vllm-hust.bidkv
vllm-hust-ext run -- vllm serve /data/shared_models/Qwen/Qwen3.8-27B \
    --tensor-parallel-size 4 --port 8000
```

可验证策略实现的 API 版本：

```bash
python - <<'PY'
from bidkv.adapters.vllm_hust.selector import BidkvPreemptionPolicy
print(BidkvPreemptionPolicy.vllm_preemption_policy_api_version)
PY
```

`BIDKV_STRATEGY` 是会 monkey patch Scheduler 的历史实验路径，不属于新的 Manager
启动链；安装包不会默认启用它。

### vLLM-HUST Extension Manager 路径

BidKV 现在随包提供 `bidkv/manifests/vllm-hust-extension-v0.2.json`，用于实验性的
vLLM-HUST Extension Manifest 0.2 路径。该 manifest 把 BidKV 描述为进程内
scheduler policy，而不是 KV store、KV connector 或外部 control plane。
wheel 会通过 `vllm_hust.extension_bundles` 静态注册 manifest；发现过程既不会
import BidKV，也不会启用调度行为。

Manifest 0.2 不构成兼容性承诺；三类 Host Provider 验收全部通过前，不能把它
作为稳定 Bundle v1 契约发布。

> **宿主边界：** 当前固定 vLLM-HUST 目标支持 `vllm.preemption-policy.v1`；
> 官方 vLLM 尚未提供该契约。上游方向是 RFC
> [#51608](https://github.com/vllm-project/vllm/issues/51608) 和 draft PR
> [#51601](https://github.com/vllm-project/vllm/pull/51601)，其目标
> `vllm.scheduler_plugins`/PreemptionScore 契约尚未冻结。HUST 的接口保持最小、
> 通用且不包含 BidKV 名称；不能据此宣称兼容官方 vLLM。

具体的语义映射、draft 代码与设计文档差异以及迁移门禁见
[上游 scheduler 契约差距](docs/upstream-scheduler-contract-gap.md)。

```bash
pip install vllm-hust-ext bidkv
vllm-hust-ext extension inspect org.vllm-hust.bidkv
vllm-hust-ext extension validate org.vllm-hust.bidkv
vllm-hust-ext extension status org.vllm-hust.bidkv
```

Manager 会校验精确宿主协议，并渲染
`--preemption-policy bidkv.adapters.vllm_hust.selector.BidkvPreemptionPolicy`。
控制器记录调用、选择、弃权、非法返回和故障计数；非法返回或运行时异常会明确记日志，
并在当前 Engine 进程内永久恢复 vLLM 内置策略。构造或配置错误直接阻止启动。

BidKV 还包含前向进展保护：当所有可运行请求都已至少被抢占
`BIDKV_UTILITY_LIVENESS_PREEMPTIONS` 次（默认 2 次）时，本次决策记录
`LIVENESS_FALLBACK`，并恢复使用 vLLM 稳定的默认 victim 顺序。只有受控实验才应把
该值设为 `0` 以关闭保护。

状态必须分开表达：

| 状态 | 含义 |
| --- | --- |
| installed | 固定版本 BidKV wheel 与 manifest 已存在。 |
| configured | Manager 已校验宿主版本/协议并生成启动参数。 |
| enabled | 已保存“下次获批启动启用 BidKV”的运维意图。 |
| runtime effective | 受控在线运行日志确认精确策略类，且调用计数非零。 |

历史固定通道通过了功能资格验证；但在当前 vLLM-HUST 与
vLLM-Ascend-HUST main 上重新执行真实硬件验收后，性能/正确性门槛没有通过，因此
Qwen3.8-27B TP4 graph 当前标记为 **incompatible**。策略确实 runtime effective
（6 次效用决策、158 次前进保护回退），取消/恢复通过，短确定性输出也与基线一致；
但实际抢占为 164 次，高于内置策略的 161 次，输出吞吐下降 2.35%，P95 TTFT
上升 2.15%，且强制 2,048-token 压力输出的哈希与基线不一致。详见
[当前 main 资格记录](docs/evidence/sage-mate-20260905-current-main-tp4-graph.md)与
[历史资格记录](docs/evidence/sage-mate-20260904-tp4-graph.md)。

如果旧回放曾被显式启用，回退时停用保存的意图并启动一个新的 vLLM 进程：

```bash
vllm-hust-ext extension disable org.vllm-hust.bidkv
```

确认新启动的 vLLM 进程已经回到未使用 BidKV 的路径后，应先清除 Manager 保存的
配置和启用意图，再卸载 Python 包，避免以后重装时恢复陈旧状态：

```bash
vllm-hust-ext extension forget org.vllm-hust.bidkv
pip uninstall bidkv
```

`forget` 不会停止已经运行的 vLLM 进程；进程重启仍由 vLLM 运维方负责。

### 旧版实验适配器

仅在复现历史多策略实验时使用：

```bash
BIDKV_STRATEGY=bidkv python -m bidkv.experiments.vllm.serve \
    --model meta-llama/Llama-3.1-8B-Instruct --enforce-eager --port 8000
```

## 零外部依赖

`bidkv` 仅依赖 Python stdlib，不依赖 torch / numpy / vllm / sglang。

## 安装

```bash
pip install -e .

# 开发模式
pip install -e ".[dev]"
```

## 测试

```bash
python -m pytest tests/ -v
```

## 许可证

Apache-2.0
