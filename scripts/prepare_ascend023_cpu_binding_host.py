# SPDX-License-Identifier: Apache-2.0
"""Prepare an isolated Ascend0.23 package with a guarded single-chip ID fix.

The installed upstream tree is read-only. This measurement compatibility fix
maps physical topology rows through the driver's existing logical-ID map; it
does not change CPU allocation, thread binding, memory migration or IRQ logic.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

BASE_COMMIT = "5cb98caaadeff42b5b62b996e34bb2aaa29d20fd"
FILE = "vllm_ascend/cpu_binding.py"
ORIGINAL_SHA256 = "71719e397ab003b31ba5e46e1a6da4da658ae7221eeead04311b65ad4d12d533"
OLD = "                    affinity[int(npu_match.group(1))] = self.expand_cpu_list(last_part)"
NEW = """                    physical_id = str(int(npu_match.group(1)))
                    chip_map = self.npu_map_info.get(physical_id)
                    if chip_map is None:
                        # A topology view may include unallocated physical NPUs.
                        # Never reinterpret such a row as a logical-ID identity.
                        logger.debug("Ignoring unmapped physical NPU topology row: %s", physical_id)
                        continue
                    if len(chip_map) != 1:
                        raise RuntimeError(
                            "CPU topology mapping requires one logical chip per physical NPU."
                        )
                    logical_text = next(iter(chip_map.values()))
                    if not isinstance(logical_text, str) or not logical_text.isdigit():
                        raise RuntimeError("CPU topology row has no valid logical NPU mapping.")
                    logical_id = int(logical_text)
                    cpus = self.expand_cpu_list(last_part)
                    if logical_id in affinity and affinity[logical_id] != cpus:
                        raise RuntimeError("Conflicting CPU topology for a logical NPU.")
                    affinity[logical_id] = cpus"""


def patched_source(raw):
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("CPU binding source differs from the exact official0.23 pin")
    text = raw.decode()
    if text.count(OLD) != 1:
        raise ValueError("Expected one original physical-ID topology assignment")
    result = text.replace(OLD, NEW).encode()
    compile(result, FILE, "exec")
    return result


def prepare(source, out):
    source, out = source.resolve(), out.resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Use a new isolated output outside the upstream tree")
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != BASE_COMMIT:
        raise ValueError("Expected the exact official Ascend0.23 source commit")
    changed = subprocess.check_output(
        ["git", "-C", str(source), "diff", "--name-only", "HEAD", "--", "vllm_ascend"]
    )
    if changed.strip():
        raise ValueError("Upstream source has tracked package changes")
    raw = (source / FILE).read_bytes()
    patched = patched_source(raw)
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < 4 * 1024**3:
        raise ValueError("Less than4GiB startup space on the output volume")
    tracked = (
        subprocess.check_output(["git", "-C", str(source), "ls-files", "-z", "--", "vllm_ascend"])
        .decode()
        .split("\0")
    )
    paths = {name for name in tracked if name}
    paths.update(
        path.relative_to(source).as_posix() for path in (source / "vllm_ascend").rglob("*.so")
    )
    if (source / "vllm_ascend/_version.py").is_file():
        paths.add("vllm_ascend/_version.py")
    out.mkdir(exist_ok=False)
    files = []
    for name in sorted(paths):
        path = source / name
        if not path.is_file():
            continue
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        data = target.read_bytes()
        files.append(
            {"path": name, "bytes": len(data), "source_sha256": hashlib.sha256(data).hexdigest()}
        )
    (out / FILE).write_bytes(patched)
    receipt = {
        "profile": "official-ascend0.23-single-chip-topology-compat/v1",
        "base_repository": "https://github.com/vllm-project/vllm-ascend",
        "base_commit": head,
        "source": str(source),
        "prepared_tree": str(out),
        "cpu_binding_original_sha256": ORIGINAL_SHA256,
        "cpu_binding_prepared_sha256": hashlib.sha256(patched).hexdigest(),
        "prepare_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_files": files,
        "upstream_source_modified": False,
        "model_generation_requests": 0,
        "scope": (
            "Single-chip physical/logical mapping only. "
            "Multi-chip or conflicting mappings safely reject; "
            "no CPU/IRQ/migration algorithm changes."
        ),
    }
    (out / "source-admission.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.out)
    print(json.dumps({key: value for key, value in result.items() if key != "source_files"}))
