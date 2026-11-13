# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of the actual pinned topology parser; no NPU/IRQ actions."""

import ast
import gzip
import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ascend023_prepare", ROOT / "scripts/prepare_ascend023_cpu_binding_host.py"
)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)
ORIGINAL = gzip.decompress((ROOT / "tests/fixtures/official023-cpu-binding.py.gz").read_bytes())


def parser(mapping, topology):
    tree = ast.parse(PREPARE.patched_source(ORIGINAL))
    device = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeviceInfo"
    )
    module = ast.Module(body=[device], type_ignores=[])
    namespace = {
        "re": re,
        "execute_command": lambda _command: (topology, 0),
        "logger": SimpleNamespace(debug=lambda *_args: None, warning=lambda *_args: None),
    }
    exec(compile(ast.fix_missing_locations(module), "pinned_device_info.py", "exec"), namespace)
    instance = object.__new__(namespace["DeviceInfo"])
    instance.npu_map_info = mapping
    return instance.parse_topo_affinity()


@pytest.mark.parametrize(
    ("mapping", "topology", "expected"),
    [
        (
            {"0": {"0": "0"}, "1": {"0": "1"}},
            "NPU0 X HCCS 0-3\nNPU1 HCCS X 4-7\n",
            {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]},
        ),
        (
            {"6": {"0": "0"}, "7": {"0": "1"}},
            "NPU6 X HCCS 96-119\nNPU7 HCCS X 48-71\n",
            {0: list(range(96, 120)), 1: list(range(48, 72))},
        ),
        (
            {"3": {"0": "0"}, "4": {"0": "1"}},
            "NPU3 X HCCS 0-23\nNPU4 HCCS X 96-119\n",
            {0: list(range(24)), 1: list(range(96, 120))},
        ),
    ],
)
def test_identity_and_actual_sparse_allocations(mapping, topology, expected):
    assert parser(mapping, topology) == expected


def test_unallocated_rows_never_overwrite_logical_zero_by_identity_guess():
    topology = "NPU0 X 0-23\nNPU6 X 96-119\nNPU7 X 48-71\nNPU9 X 144-167\n"
    assert parser({"6": {"0": "0"}, "7": {"0": "1"}}, topology) == {
        0: list(range(96, 120)),
        1: list(range(48, 72)),
    }
    assert parser({}, "NPU0 X 0-23\n") == {}


def test_multichip_and_invalid_mapping_safely_reject():
    with pytest.raises(RuntimeError, match="one logical chip"):
        parser({"6": {"0": "0", "1": "1"}}, "NPU6 X 0-23\n")
    with pytest.raises(RuntimeError, match="valid logical"):
        parser({"6": {"0": "-"}}, "NPU6 X 0-23\n")


def test_conflicting_physical_rows_cannot_rebind_one_logical_device():
    with pytest.raises(RuntimeError, match="Conflicting"):
        parser({"6": {"0": "0"}, "7": {"0": "0"}}, "NPU6 X 96-119\nNPU7 X 48-71\n")


def test_source_guard_and_no_changes_to_binding_irq_or_migration_algorithms():
    with pytest.raises(ValueError, match="exact official"):
        PREPARE.patched_source(ORIGINAL + b"\n")
    original, patched = ast.parse(ORIGINAL), ast.parse(PREPARE.patched_source(ORIGINAL))

    def functions(tree):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    before, after = functions(original), functions(patched)
    assert before.keys() == after.keys()
    assert [name for name in before if before[name] != after[name]] == ["parse_topo_affinity"]
