#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare a separate source-pinned measurement host; leave its source untouched.

Requires the official vLLM v0.23.0 checkout and its matching prebuilt libraries.
This explicit native source adaptation is not the legacy scheduler monkeypatch
or a general Extension Manager compatibility claim. Both measured arms use the
same prepared tree, with the policy selected through additional configuration.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

BASE_COMMIT = "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"
SCHEDULER_SHA256 = "9ae1491f57a6294e32416b93ef2ea68107fa0d65c3a4d7f745f1261012f735e9"
SCHEDULER = "vllm/v1/core/sched/scheduler.py"

NEW_PREEMPTION = """                    # Selection receives immutable values; all state changes
                    # and scheduling-budget rollback remain native host work.
                    preempted_req = self._select_bidkv023_victim(
                        request, scheduled_timestamp
                    )
                    victim_index = self.running.index(preempted_req)
                    self.running.pop(victim_index)
                    if victim_index < req_index:
                        req_index -= 1
                    if preempted_req in scheduled_running_reqs:
                        preempted_req_id = preempted_req.request_id
                        scheduled_running_reqs.remove(preempted_req)
                        token_budget += num_scheduled_tokens.pop(preempted_req_id)
                        req_to_new_blocks.pop(preempted_req_id)
                        scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                        preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                            preempted_req_id, None
                        )
                        if preempted_encoder_inputs:
                            encoder_compute_budget += sum(
                                preempted_req.get_num_encoder_embeds(i)
                                for i in preempted_encoder_inputs
                            )
"""

SELECTOR_METHOD = """    def _select_bidkv023_victim(
        self, request: Request, timestamp: float
    ) -> Request:
        builtin = (
            max(self.running, key=lambda row: (row.priority, row.arrival_time))
            if self.policy == SchedulingPolicy.PRIORITY
            else self.running[-1]
        )
        context = PreemptionContext(
            candidates=tuple(
                PreemptionCandidate(
                    request_id=row.request_id,
                    priority=row.priority,
                    arrival_time=row.arrival_time,
                    num_prompt_tokens=row.num_prompt_tokens,
                    num_output_tokens=row.num_output_tokens,
                    num_computed_tokens=row.num_computed_tokens,
                    num_preemptions=row.num_preemptions,
                    max_tokens=row.max_tokens,
                    predicted_length=getattr(row, "predicted_length", None),
                )
                for row in self.running
            ),
            scheduling_policy=self.policy.value,
            requesting_request_id=request.request_id,
            kv_cache_usage=self.kv_cache_manager.usage,
            now=timestamp,
            builtin_victim_id=builtin.request_id,
        )
        return self.requests[self._bidkv023_policy.select_victim(context)]

"""


def patched_scheduler(raw):
    if hashlib.sha256(raw).hexdigest() != SCHEDULER_SHA256:
        raise ValueError("Scheduler differs from the exact official v0.23.0 source")
    text = raw.decode()
    marker = "from vllm.v1.core.sched.output import ("
    imports = """from bidkv.adapters.vllm023.bridge import (
    Official023PolicyBridge,
    PreemptionCandidate,
    PreemptionContext,
)
"""
    assert text.count(marker) == 1
    text = text.replace(marker, imports + marker)
    marker = "        self.running: list[Request] = []\n"
    assert text.count(marker) == 1
    text = text.replace(
        marker, marker + "        self._bidkv023_policy = Official023PolicyBridge(vllm_config)\n"
    )
    begin = text.index("                    # The request cannot be scheduled.\n")
    end = text.index(
        "                    self._preempt_request(preempted_req, scheduled_timestamp)", begin
    )
    text = text[:begin] + NEW_PREEMPTION + "\n" + text[end:]
    marker = "    def _preempt_request(self, request: Request, timestamp: float) -> None:\n"
    assert text.count(marker) == 1
    text = text.replace(marker, SELECTOR_METHOD + marker)
    compile(text, SCHEDULER, "exec")
    return text.encode()


def prepare(source, out):
    source, out = source.resolve(), out.resolve()
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Use a new, separate destination outside the source checkout")
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != BASE_COMMIT:
        raise ValueError("Source must be the exact official v0.23.0 commit")
    changed = subprocess.check_output(
        ["git", "-C", str(source), "diff", "--name-only", "HEAD", "--", "vllm"]
    )
    if changed.strip():
        raise ValueError("Source package has tracked modifications")
    original = (source / SCHEDULER).read_bytes()
    patched = patched_scheduler(original)
    out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out.parent).free < 4 * 1024**3:
        raise ValueError("Output volume has less than 4GiB free before preparation")
    tracked = (
        subprocess.check_output(["git", "-C", str(source), "ls-files", "-z", "--", "vllm"])
        .decode()
        .split("\0")
    )
    paths = {name for name in tracked if name}
    paths.update(path.relative_to(source).as_posix() for path in (source / "vllm").rglob("*.so"))
    if (source / "vllm/_version.py").is_file():
        paths.add("vllm/_version.py")
    out.mkdir(exist_ok=False)
    records = []
    for name in sorted(paths):
        path = source / name
        if not path.is_file():
            continue
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        raw = target.read_bytes()
        records.append(
            {"path": name, "bytes": len(raw), "source_sha256": hashlib.sha256(raw).hexdigest()}
        )
    (out / SCHEDULER).write_bytes(patched)
    receipt = {
        "profile": "official-vllm-0.23-source-bridge/v1",
        "base_repository": "https://github.com/vllm-project/vllm",
        "base_commit": BASE_COMMIT,
        "scheduler_original_sha256": hashlib.sha256(original).hexdigest(),
        "scheduler_prepared_sha256": hashlib.sha256(patched).hexdigest(),
        "prepare_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": str(source),
        "prepared_tree": str(out),
        "source_files": records,
        "source_modified": False,
        "model_inference_requests": 0,
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
