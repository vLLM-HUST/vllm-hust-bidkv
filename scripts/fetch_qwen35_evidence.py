"""Download immutable benchmark originals outside this repository; never extract or execute."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path, PurePosixPath
from urllib.error import URLError
from urllib.request import urlopen

COMMIT = "71c2d4ad5f34536d73f66c577e1017da59dba085"
CAPSULE = "archive/mod-git-20261022/website-source-vllm-hust-bidkv"
BASE = f"https://raw.githubusercontent.com/vLLM-HUST/vllm-hust-benchmark/{COMMIT}/{CAPSULE}"
MANIFEST_SHA256 = "2d9ba8f088e5b43b8b7ede5a2018264f405dcc575e887fd42de9c665134580e3"
ROOT = Path(__file__).resolve().parents[1]


def checked(raw: bytes, digest: str, size: int | None = None) -> bytes:
    if hashlib.sha256(raw).hexdigest() != digest or (size is not None and len(raw) != size):
        raise ValueError("Evidence checksum/length mismatch")
    return raw


def destination(path: Path) -> Path:
    target = path.resolve()
    if target == ROOT or ROOT in target.parents:
        raise ValueError("Evidence destination must be outside the source repository")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("Evidence destination must be an empty directory")
    return target


def member(path: str) -> Path:
    parts = PurePosixPath(path)
    if parts.is_absolute() or ".." in parts.parts or not path.startswith("results/"):
        raise ValueError("Unsafe evidence path")
    return Path(*parts.parts)


def download(url: str) -> bytes:
    for attempt in range(3):
        try:
            with urlopen(url, timeout=60) as response:
                return response.read()
        except (URLError, TimeoutError) as error:
            print(f"Download attempt {attempt + 1} failed: {url}: {error}", file=sys.stderr)
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    target = destination(args.destination)
    raw = checked(download(f"{BASE}/manifest.json"), MANIFEST_SHA256)
    manifest = json.loads(raw)
    files = manifest["files"]
    # Validate all destinations before writing anything.
    names = [member(entry["path"]) for entry in files]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate evidence path")
    target.mkdir(parents=True, exist_ok=True)
    for entry, name in zip(files, names, strict=True):
        payload = checked(
            download(f"{BASE}/snapshot/{entry['path']}"), entry["sha256"], entry["bytes"]
        )
        output = target / name
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(payload)
    with (target / "manifest.json").open("xb") as stream:
        stream.write(raw)
    print(f"Verified {len(files)} originals from benchmark {COMMIT}; no extraction or execution")


if __name__ == "__main__":
    main()
