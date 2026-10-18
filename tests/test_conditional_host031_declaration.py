"""Declaration bounds are not runtime qualification or accepted effects."""

import json
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "src/bidkv/manifests/vllm-hust-extension-v0.3.json"


@pytest.mark.parametrize(
    ("version", "accepted"),
    [
        ("0.31.1.post1.dev19+gf7bc2a5e.empty", True),
        ("0.31.1.post1.dev18+g772b34583.empty", False),
        ("0.31.1.post1.dev20+gunknown.empty", False),
        ("0.31.1.post1", False),
        ("0.32.0", False),
        ("0.28.1rc1.dev319", False),
    ],
)
def test_exact_next_host_public_family_bound(version, accepted):
    manifest = json.loads(MANIFEST.read_bytes())
    assert (
        SpecifierSet(manifest["host"]["version_range"]).contains(version, prereleases=True)
        is accepted
    )


def test_new_scope_never_reuses_old_runtime_qualification():
    manifest = json.loads(MANIFEST.read_bytes())
    profile = manifest["activation"]["additional_config"]["_manager_runtime_qualification"]
    assert manifest["extension_version"] == "0.2.2.dev1"
    assert manifest["schema_version"] == "0.3-experimental"
    assert manifest["host"]["api_range"] == "==1.0"
    assert profile["host_commit"] == "f7bc2a5eebc505f28abfbf318734225e4c30c64c"
    assert profile["host_tree"] == "9ee4ce4dbcb4f685df8713fb1c33a53d7221ba27"
    assert profile["model"] == "/models/Qwen2.5-7B-Instruct"
    assert [
        profile[k] for k in ("pipeline_parallel_size", "tensor_parallel_size", "max_num_seqs")
    ] == [2, 1, 1]
    assert [profile[k] for k in ("warmup_requests", "measured_requests", "selected_requests")] == [
        16,
        256,
        32,
    ]
    assert (
        not {
            "status",
            "functional_status",
            "functional_compatibility",
            "recovery_status",
            "performance_status",
        }
        & profile.keys()
    )
