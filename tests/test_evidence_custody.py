import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fetch_evidence", ROOT / "scripts/fetch_qwen35_evidence.py"
)
FETCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCH)


def test_checksum_and_size_fail_closed():
    raw = b"synthetic custody unit-test bytes"
    digest = hashlib.sha256(raw).hexdigest()
    assert FETCH.checked(raw, digest, len(raw)) == raw
    with pytest.raises(ValueError, match="checksum"):
        FETCH.checked(raw + b"corrupt", digest)
    with pytest.raises(ValueError, match="length"):
        FETCH.checked(raw, digest, len(raw) + 1)


def test_destination_outside_repo_and_empty(tmp_path):
    assert FETCH.destination(tmp_path) == tmp_path
    with pytest.raises(ValueError, match="outside"):
        FETCH.destination(ROOT / "results/new")
    (tmp_path / "existing").touch()
    with pytest.raises(ValueError, match="empty"):
        FETCH.destination(tmp_path)


@pytest.mark.parametrize("path", ["/results/x", "results/../x", "other/x"])
def test_unsafe_paths_rejected(path):
    with pytest.raises(ValueError, match="Unsafe"):
        FETCH.member(path)
