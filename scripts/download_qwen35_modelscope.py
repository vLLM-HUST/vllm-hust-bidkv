"""Download one pinned Qwen3.5-35B-A3B snapshot from ModelScope with checksums.

Use an HF tree manifest for file sizes and LFS SHA-256 hashes. ModelScope serves
the same Qwen snapshot on machines where Hugging Face's weight CDN is blocked.
The downloader resumes partial files and verifies every completed file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote


MODEL_URL = "https://modelscope.cn/models/Qwen/Qwen3.5-35B-A3B/resolve/master"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path, size: int) -> str:
    digest = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(entry: dict, target: Path) -> str:
    relative = entry["path"]
    size = int(entry["size"])
    digest = entry.get("lfs", {}).get("oid")
    blob_oid = entry.get("oid") if digest is None else None
    target_path = target / relative
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and target_path.stat().st_size == size:
        if (digest is not None and sha256_file(target_path) == digest) or (
            blob_oid is not None and git_blob_sha1(target_path, size) == blob_oid
        ):
            return f"verified {relative}"

    partial = target_path.with_name(target_path.name + ".part")
    if partial.exists() and partial.stat().st_size > size:
        raise RuntimeError(f"partial file larger than expected: {partial}")
    url = f"{MODEL_URL}/{quote(relative, safe='/')}"
    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    command = [
        "curl", "--noproxy", "*", "--fail", "--location", "--silent", "--show-error",
        "--retry", "8", "--retry-delay", "3", "--connect-timeout", "20",
        "--speed-limit", "1024", "--speed-time", "120",
        "--continue-at", "-", "--output", str(partial), url,
    ]
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"download failed for {relative}: {result.stderr[-500:]}")
    actual_size = partial.stat().st_size
    if actual_size != size:
        raise RuntimeError(f"size mismatch for {relative}: {actual_size} != {size}")
    if digest is not None:
        actual_digest = sha256_file(partial)
        if actual_digest != digest:
            raise RuntimeError(f"checksum mismatch for {relative}: {actual_digest} != {digest}")
    elif blob_oid is not None:
        actual_oid = git_blob_sha1(partial, size)
        if actual_oid != blob_oid:
            raise RuntimeError(f"blob mismatch for {relative}: {actual_oid} != {blob_oid}")
    partial.replace(target_path)
    return f"downloaded {relative} ({size} bytes)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    entries = json.loads(args.manifest.read_text())
    if not isinstance(entries, list) or not entries:
        parser.error("manifest must be a nonempty HF tree list")
    args.target.mkdir(parents=True, exist_ok=True)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, entry, args.target) for entry in entries]
        for future in concurrent.futures.as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as error:
                failures.append(str(error))
                print(f"ERROR {error}", file=sys.stderr, flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} download(s) failed")


if __name__ == "__main__":
    main()
