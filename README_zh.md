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

与同级目录的 `vllm-hust-dev-hub` 配合时，仓库清单可将启用流程缩减为一条
命令。dev-hub 会自动安装包、验证精确入口点并注入原生选择器配置：

```bash
cd ../vllm-hust-dev-hub
./manage.sh restart --optimization bidkv
```

在 `vllm-hust` 中使用时，需要把 BidKV 安装到同一个 Python 环境。
运行时通过 `vllm.victim_selector` 入口点自动发现它。仅安装不会改变调度行为；
启动服务时需要显式选择并启用 BidKV：

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

启动前可验证入口点：

```bash
python - <<'PY'
from importlib.metadata import entry_points

for ep in entry_points(group="vllm.victim_selector"):
    print(ep.name, "->", ep.value)
PY
```

也可以通过 `BIDKV_UTILITY_` 前缀的环境变量配置相同参数。
`BIDKV_STRATEGY` 属于旧版实验适配器，它会 monkey patch Scheduler，不能与新的
victim-selector 接入方式同时使用。

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
