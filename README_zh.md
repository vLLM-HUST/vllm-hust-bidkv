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

历史 HUST fork 曾提供 `vllm.victim_selector`。BidKV 仍保留可导入的兼容模块，
用于固定旧版本的契约回放，但主发行包不再注册这个非上游 entry-point 命名空间。
下面的启动形态只适用于已经拥有该契约的固定旧 fork，不能用于新的官方 vLLM fork：

```bash
python -m pip install -e . --no-deps

vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --enforce-eager \
    --port 8000 \
    --additional-config '{
      "victim_selector_plugin": "bidkv",
      "enable_utility_victim_selection": true,
      "utility_strategy": "bidkv",
      "utility_kv_gate": 0.95
    }'
```

可验证旧适配器模块，但不能把这当成运行时已经支持自动发现：

```bash
python - <<'PY'
from bidkv.adapters.vllm_hust.selector import BidkvVictimSelector
print(BidkvVictimSelector.vllm_victim_selector_api_version)
PY
```

`BIDKV_UTILITY_` 前缀变量同样只属于这个固定旧契约。`BIDKV_STRATEGY` 是另一条
会 monkey patch Scheduler 的历史实验路径；两者都不是对新官方 vLLM 的支持。

### vLLM-HUST Extension Manager 路径

BidKV 现在随包提供 `bidkv/manifests/vllm-hust-extension-v0.2.json`，用于实验性的
vLLM-HUST Extension Manifest 0.2 路径。该 manifest 把 BidKV 描述为进程内
scheduler policy，而不是 KV store、KV connector 或外部 control plane。
wheel 会通过 `vllm_hust.extension_bundles` 静态注册 manifest；发现过程既不会
import BidKV，也不会启用调度行为。

Manifest 0.2 不构成兼容性承诺；三类 Host Provider 验收全部通过前，不能把它
作为稳定 Bundle v1 契约发布。

> **宿主契约警告：** `vllm.victim_selector` 是旧 HUST 实验 hook，新 fork 的
> vLLM-HUST 0.23 并不包含它。上游方向是 RFC
> [#51608](https://github.com/vllm-project/vllm/issues/51608) 和 draft PR
> [#51601](https://github.com/vllm-project/vllm/pull/51601)，其目标
> `vllm.scheduler_plugins`/PreemptionScore 契约尚未冻结。不要向新核心再加入一套
> 竞争的私有 hook，也不能宣称当前已兼容 0.23。

具体的语义映射、draft 代码与设计文档差异以及迁移门禁见
[上游 scheduler 契约差距](docs/upstream-scheduler-contract-gap.md)。

```bash
pip install vllm-hust-ext bidkv
vllm-hust-ext extension inspect org.vllm-hust.bidkv
vllm-hust-ext extension validate org.vllm-hust.bidkv
vllm-hust-ext extension status org.vllm-hust.bidkv
```

在新的官方 vLLM 环境中，状态必须保持 `incompatible` 或 `degraded`，Manager 的
`run` 会拒绝激活。固定旧 fork 的操作者可以显式提供宿主版本和
`vllm.victim_selector` 协议证据，但这只用于回放。alpha 门禁要求先迁移到稳定后的
上游 Preemption 契约，并完成真实 scheduler 调用、冲突/失败和下次进程回退测试；
在此之前没有受支持的启用命令。

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
