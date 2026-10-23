# Qwen3.5 evidence custody

The original Qwen3.5 campaign, including negative, not-exercised and aborted runs,
is preserved in the [immutable benchmark archive](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/71c2d4ad5f34536d73f66c577e1017da59dba085/archive/mod-git-20261022/website-source-vllm-hust-bidkv).
Read the [original report and its scope](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/71c2d4ad5f34536d73f66c577e1017da59dba085/archive/mod-git-20261022/website-source-vllm-hust-bidkv/snapshot/results/qwen35-evoscientist-20260929/README.md).
This migration does not rerun or qualify the experiment and does not change runtime defaults.

The source repository retains implementation, analysis tools and the compatibility
patch. Datasets, metrics and run archives live in benchmark, not in source packages.
Historical Git commits remain unchanged. Other historical evidence documents and
paper assets remain a separate migration task; this page does not declare all
repository evidence migrated.

```bash
# Choose a new, empty directory outside this checkout.
python scripts/fetch_qwen35_evidence.py /tmp/bidkv-evidence-review
```

The downloader pins the benchmark commit and manifest SHA256, verifies every
original byte length and SHA256, and refuses repository-local or nonempty outputs.
It does not unpack archives, apply patches, run models, or reuse partial downloads.
Keep any failed download directory for inspection and use a new empty directory
when retrying. Inspect the original report before manually unpacking an archive;
the summarizer accepts an explicitly supplied external directory of extracted runs.

中文：原始输入、结果、失败记录及归档已迁至上方固定版本 benchmark 入口。
下载器逐文件回验，不执行归档或模型。迁移不授予新的性能资格，不覆盖历史证据；
其他历史文档和论文材料尚不属于本批清理范围。
