"""The new target checks actual model/topology, without asserting qualification."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bidkv.adapters.vllm_hust.runtime_contract import validate_runtime_contract
from bidkv.adapters.vllm_hust.selector import BidkvPreemptionPolicy


@pytest.fixture(autouse=True)
def clear_target_environment(monkeypatch):
    monkeypatch.delenv("BIDKV_RUNTIME_CONTRACT", raising=False)


def config():
    manifest = json.loads((Path(__file__).resolve().parents[1] / "src/bidkv/manifests/"
                           "vllm-hust-extension-v0.3.json").read_text())
    return SimpleNamespace(
        additional_config={"bidkv_runtime_contract": json.loads(
            manifest["activation"]["environment"]["BIDKV_RUNTIME_CONTRACT"])},
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3_5_moe",
                                     architectures=["Qwen3_5MoeForConditionalGeneration"]),
            dtype="torch.bfloat16", enforce_eager=False),
        parallel_config=SimpleNamespace(tensor_parallel_size=2, pipeline_parallel_size=1,
                                        data_parallel_size=1),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY")),
    )


def test_matching_target_is_admitted_but_unqualified():
    policy = BidkvPreemptionPolicy.from_vllm_config(config())
    assert policy.runtime_target["qualification_status"] == "unqualified-development-candidate"
    assert policy.runtime_target["tensor_parallel_size"] == 2


@pytest.mark.parametrize("section,field,value", [
    ("parallel_config", "tensor_parallel_size", 4),
    ("parallel_config", "pipeline_parallel_size", 2),
    ("parallel_config", "data_parallel_size", 2),
    ("model_config", "enforce_eager", True),
    ("model_config", "dtype", "torch.float16"),
    ("model_config", "hf_config", SimpleNamespace(model_type="qwen3", architectures=[])),
    ("compilation_config", "cudagraph_mode", SimpleNamespace(name="NONE")),
])
def test_mismatched_effective_host_fails_construction(section, field, value):
    c = config()
    setattr(getattr(c, section), field, value)
    with pytest.raises(ValueError, match="runtime target mismatch"):
        BidkvPreemptionPolicy.from_vllm_config(c)


@pytest.mark.parametrize("change", [
    {"schema_version": 2}, {"schema_version": True}, {"qualification_status": "passed"},
    {"enforce_eager": True}, {"tensor_parallel_size": True}, {"unknown": 1},
])
def test_bad_target_declarations_fail(change):
    c = config()
    c.additional_config["bidkv_runtime_contract"].update(change)
    with pytest.raises(ValueError):
        validate_runtime_contract(c)


def test_missing_host_fails_when_contract_is_declared():
    c = config()
    del c.model_config
    with pytest.raises(ValueError, match="runtime target mismatch"):
        validate_runtime_contract(c)


def test_contract_is_opt_in_for_historical_adapters():
    assert validate_runtime_contract(SimpleNamespace(additional_config={})) is None


def test_environment_contract_needs_no_host_additional_config(monkeypatch):
    c = config()
    target = c.additional_config.pop("bidkv_runtime_contract")
    monkeypatch.setenv("BIDKV_RUNTIME_CONTRACT", json.dumps(target))
    assert BidkvPreemptionPolicy.from_vllm_config(c).runtime_target["tensor_parallel_size"] == 2
    assert c.additional_config == {}


def test_conflicting_target_sources_fail(monkeypatch):
    c = config()
    other = copy.deepcopy(c.additional_config["bidkv_runtime_contract"])
    other["tensor_parallel_size"] = 4
    monkeypatch.setenv("BIDKV_RUNTIME_CONTRACT", json.dumps(other))
    with pytest.raises(ValueError, match="disagree"):
        validate_runtime_contract(c)


def test_invalid_environment_json_fails(monkeypatch):
    monkeypatch.setenv("BIDKV_RUNTIME_CONTRACT", "not-json")
    with pytest.raises(ValueError, match="invalid JSON"):
        validate_runtime_contract(config())


def test_manifest_additional_config_matches_installed_ascend_schema():
    from dataclasses import fields

    from vllm_ascend.ascend_config import AscendConfig

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "src/bidkv/manifests/vllm-hust-extension-v0.3.json").read_text())
    assert set(manifest["activation"]["additional_config"]) <= {
        field.name for field in fields(AscendConfig)
    }


def test_historical_qualification_is_retained_separately():
    root = Path(__file__).resolve().parents[1]
    old = json.loads((root / "docs/evidence/qwen38-tp4-manifest-qualification.json").read_text())
    assert old["qualification"]["model"] == "Qwen3.8-27B"
    assert old["qualification"]["tensor_parallel_size"] == 4
    manifest = json.loads((root / "src/bidkv/manifests/vllm-hust-extension-v0.3.json").read_text())
    assert "_manager_runtime_qualification" not in manifest["activation"]["additional_config"]
    assert manifest["activation"]["additional_config"] == {}
    target = json.loads(manifest["activation"]["environment"]["BIDKV_RUNTIME_CONTRACT"])
    assert target["qualification_status"] == "unqualified-development-candidate"
