# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, no-inference OpenSandbox probe for the Gym 79901c5 ripgrep retry.

The first *real* provider upload completes, then this process discards its
acknowledgement by raising SandboxIOTimeoutError. The pinned agent must upload
again at a fresh path. This is a diagnostic, not an evaluation or a proof of
full-run reliability. Run only in a quiescent DeepSWE window after independent
capacity and no-writer checks. Without --execute it performs preflight only.

The SHA-pinned JSON manifest has these keys:

    efb_root, source_root, source_commit,
    source_file, source_sha256, task_list_file, task_list_sha256,
    resolved_run_file, resolved_run_sha256,
    ripgrep_file, ripgrep_sha256,
    provider_config, sandbox_spec

``provider_config`` contains the reviewed OpenSandbox endpoint and bounded
operation settings. ``sandbox_spec`` is the independently reviewed task-image
SandboxSpec with a finite TTL and no-egress network policy. The script cannot
derive these fields from shell-formatted run.sub, so --execute also requires an
explicit review attestation. Keep credentials out of the evidence file/logs.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from tempfile import TemporaryDirectory
from typing import Any

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.sandbox import AsyncSandbox, SandboxIOTimeoutError, SandboxSpec, create_provider
from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider
from responses_api_agents.opencode_sandboxed_agent import app as opencode_app
from responses_api_agents.opencode_sandboxed_agent.app import (
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
)


GYM_COMMIT = "79901c5761df43645056f7b5a482cfbc34c4d271"
EFB_COMMIT = "e4e71e5fa337ac58f0cdead6593a8c5e4019ade9"
SOURCE_COMMIT = "435ee89ec2f2e2289f33b0da4f992f0b7b7266b9"
SERVING_IMAGE_SHA256 = "c30b96ff5971e1af916dc280d528289bb090592042db347500d76de593cd02c4"
RIPGREP_SHA256 = "ebeaf56f8a25e102e9419933423738b3a2a613a444fd749d695e15eba53f71f2"
CHECKPOINT_REVISION = "72e1d323"
REMOTE_RG_PATTERN = re.compile(r"/tmp/nemo-gym-ripgrep-[0-9a-f]{32}\Z")
MAX_TTL_S = 1800
MAX_READY_TIMEOUT_S = 180
CLEANUP_TIMEOUT_S = 30


@dataclass(frozen=True)
class ProbeInputs:
    script_sha256: str
    manifest_sha256: str
    rg_path: Path
    resolved_run_sha256: str
    source_sha256: str
    task_list_sha256: str
    provider_config: dict[str, Any]
    sandbox_spec: SandboxSpec


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_file(path_value: object, expected_value: object, label: str) -> tuple[Path, str]:
    if not isinstance(path_value, str) or not isinstance(expected_value, str):
        raise ValueError(f"{label} needs a path and a SHA-256 digest")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_value):
        raise ValueError(f"{label} has an invalid SHA-256 digest")
    path = Path(path_value).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")
    actual = _sha256(path)
    if actual != expected_value:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_value}, got {actual}")
    return path, actual


def _git_head(path_value: object, label: str) -> str:
    if not isinstance(path_value, str):
        raise ValueError(f"{label} needs a checkout path")
    checkout = Path(path_value).resolve(strict=True)
    result = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def verify_preflight(manifest_path: Path, expected_manifest_sha256: str, expected_script_sha256: str) -> ProbeInputs:
    """Verify pinned bytes and the independently reviewed, restricted manifest."""
    _, script_sha256 = _verified_file(__file__, expected_script_sha256, "probe script")
    manifest_path, manifest_sha256 = _verified_file(manifest_path.as_posix(), expected_manifest_sha256, "manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")

    gym_root = Path(__file__).resolve().parents[2]
    if _git_head(str(gym_root), "Gym") != GYM_COMMIT:
        raise ValueError(f"Gym checkout must be exact {GYM_COMMIT}")
    for module, relative_path in (
        (opencode_app, "responses_api_agents/opencode_sandboxed_agent/app.py"),
        (opensandbox_provider, "nemo_gym/sandbox/providers/opensandbox/provider.py"),
    ):
        if Path(module.__file__).resolve() != gym_root / relative_path:
            raise ValueError(f"imported {module.__name__} is not from the pinned Gym checkout")
    if _git_head(manifest.get("efb_root"), "EFB") != EFB_COMMIT:
        raise ValueError(f"EFB checkout must be exact {EFB_COMMIT}")
    if (
        manifest.get("source_commit") != SOURCE_COMMIT
        or _git_head(manifest.get("source_root"), "task source") != SOURCE_COMMIT
    ):
        raise ValueError(f"task source must be exact {SOURCE_COMMIT}")

    source_file, source_sha256 = _verified_file(
        manifest.get("source_file"), manifest.get("source_sha256"), "task source"
    )
    if not source_file.is_relative_to(Path(manifest["source_root"]).resolve(strict=True)):
        raise ValueError("task source file must be inside the exact task source checkout")
    _, task_list_sha256 = _verified_file(manifest.get("task_list_file"), manifest.get("task_list_sha256"), "task list")
    resolved_run, resolved_run_sha256 = _verified_file(
        manifest.get("resolved_run_file"), manifest.get("resolved_run_sha256"), "resolved run"
    )
    rg_path, rg_sha256 = _verified_file(manifest.get("ripgrep_file"), manifest.get("ripgrep_sha256"), "ripgrep")
    if rg_sha256 != RIPGREP_SHA256:
        raise ValueError(f"ripgrep must be the pinned {RIPGREP_SHA256} binary")

    resolved_text = resolved_run.read_text(encoding="utf-8")
    for label, token in (
        ("Gym", GYM_COMMIT),
        ("vLLM 0.30 image", SERVING_IMAGE_SHA256),
        ("checkpoint", CHECKPOINT_REVISION),
        ("ripgrep mount", rg_path.as_posix()),
    ):
        if token not in resolved_text:
            raise ValueError(f"resolved run does not contain its pinned {label}")

    provider_config = manifest.get("provider_config")
    if not isinstance(provider_config, dict) or set(provider_config) != {"opensandbox"}:
        raise ValueError("provider_config must contain only opensandbox")
    provider_options = provider_config["opensandbox"]
    if not isinstance(provider_options, dict) or not set(provider_options) <= {"connection", "create", "operations"}:
        raise ValueError(
            "provider_config may contain only connection, create and operations; no setup or image rewrite"
        )
    if not isinstance(provider_options.get("connection"), dict) or not provider_options["connection"].get("domain"):
        raise ValueError("provider_config needs an explicit OpenSandbox connection domain")
    for key in ("create", "operations"):
        if key in provider_options and not isinstance(provider_options[key], dict):
            raise ValueError(f"provider_config {key} must be an object")
    if provider_options.get("create", {}).get("renew_interval_s") is not None:
        raise ValueError("probe sandbox TTL renewal is forbidden")
    spec_data = manifest.get("sandbox_spec")
    if not isinstance(spec_data, dict):
        raise ValueError("sandbox_spec must be a JSON object")
    if not set(spec_data) <= {"image", "ttl_s", "ready_timeout_s", "resources", "provider_options"}:
        raise ValueError("sandbox_spec cannot add commands, environment, mounts or metadata")
    spec = SandboxSpec(**spec_data)
    if not spec.image or not isinstance(spec.ttl_s, (int, float)) or not 60 <= spec.ttl_s <= MAX_TTL_S:
        raise ValueError(f"sandbox_spec needs an image and TTL from 60 to {MAX_TTL_S} seconds")
    if not isinstance(spec.ready_timeout_s, (int, float)) or not 1 <= spec.ready_timeout_s <= MAX_READY_TIMEOUT_S:
        raise ValueError(f"sandbox_spec needs a ready timeout from 1 to {MAX_READY_TIMEOUT_S} seconds")
    if spec.resources.gpu or spec.resources.gpu_type:
        raise ValueError("probe sandbox cannot request GPUs")
    if not isinstance(spec.resources.cpu, (int, float)) or not 0 < spec.resources.cpu <= 8:
        raise ValueError("probe sandbox needs a CPU limit from 0 to 8")
    if not isinstance(spec.resources.memory_mib, int) or not 512 <= spec.resources.memory_mib <= 49152:
        raise ValueError("probe sandbox needs a memory limit from 512 to 49152 MiB")
    if spec.resources.disk_gib is not None and not 1 <= spec.resources.disk_gib <= 50:
        raise ValueError("probe sandbox disk limit must be from 1 to 50 GiB")
    if set(spec.provider_options) != {"network_policy"}:
        raise ValueError("probe sandbox permits only a network policy provider option")
    network_policy = spec.provider_options.get("network_policy")
    if network_policy != {"defaultAction": "deny", "egress": []}:
        raise ValueError("probe sandbox needs default-deny network policy with no egress exceptions")

    return ProbeInputs(
        script_sha256=script_sha256,
        manifest_sha256=manifest_sha256,
        rg_path=rg_path,
        resolved_run_sha256=resolved_run_sha256,
        source_sha256=source_sha256,
        task_list_sha256=task_list_sha256,
        provider_config=provider_config,
        sandbox_spec=spec,
    )


def _agent_for_staging() -> OpenCodeSandboxedAgent:
    config = OpenCodeSandboxedAgentConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="",
        name="deepswe-rg-retry-probe",
        resources_server=ResourcesServerRef(type="resources_servers", name="unused"),
        model_server=ModelServerRef(type="responses_api_models", name="unused"),
        sandbox_provider="opensandbox",
        sandbox_config={},
        sandbox_timeout=60,
        opencode_max_context_window=1,
    )
    # No server or model client is created. model_construct still runs the
    # agent's own post-init and creates its real ripgrep upload semaphore.
    return OpenCodeSandboxedAgent.model_construct(config=config, server_client=None)


async def _stage_after_lost_ack(
    agent: OpenCodeSandboxedAgent, sandbox: AsyncSandbox, rg_path: Path, attempts: list[str]
) -> list[str]:
    """Discard only the first successful provider acknowledgement."""
    real_upload = sandbox.upload
    first_upload_completed = False

    async def upload_and_lose_first_ack(local_path: Path | str, remote_path: str) -> None:
        nonlocal first_upload_completed
        attempts.append(remote_path)
        await real_upload(local_path, remote_path)
        if len(attempts) == 1:
            first_upload_completed = True
            raise SandboxIOTimeoutError("injected lost acknowledgement after a completed provider upload")

    sandbox.upload = upload_and_lose_first_ack
    try:
        staged_path = await agent._stage_ripgrep(sandbox, rg_path)
    finally:
        sandbox.upload = real_upload

    if not first_upload_completed or len(attempts) != 2 or staged_path != attempts[1]:
        raise RuntimeError("Gym did not retry exactly once after the injected lost acknowledgement")
    if attempts[0] == attempts[1] or any(not REMOTE_RG_PATTERN.fullmatch(path) for path in attempts):
        raise RuntimeError("Gym reused an ambiguous ripgrep path or used an unexpected path")
    return attempts


async def _verify_remote_binary(sandbox: AsyncSandbox, attempts: list[str], expected_sha256: str) -> str:
    with TemporaryDirectory(prefix="deepswe-rg-probe-") as temp_dir:
        for index, remote_path in enumerate(attempts):
            local_path = Path(temp_dir) / f"rg-{index}"
            await sandbox.download(remote_path, local_path)
            if _sha256(local_path) != expected_sha256:
                raise RuntimeError(f"remote ripgrep attempt {index + 1} has a wrong checksum")

    install_command = (
        'mkdir -p "$HOME/.opencode/bin" '
        f'&& cp {quote(attempts[1])} "$HOME/.opencode/bin/rg" '
        '&& chmod 0755 "$HOME/.opencode/bin/rg" '
        '&& sha256sum "$HOME/.opencode/bin/rg" '
        '&& "$HOME/.opencode/bin/rg" --version'
    )
    result = await sandbox.exec(install_command, timeout_s=60)
    if result.return_code != 0 or result.error_type is not None:
        raise RuntimeError("ripgrep install or version check failed in the task sandbox")
    output = result.stdout or ""
    lines = output.splitlines()
    if not lines or lines[0].split(maxsplit=1)[0] != expected_sha256:
        raise RuntimeError("installed ripgrep has a wrong checksum")
    if not any(line.startswith("ripgrep ") for line in lines[1:]):
        raise RuntimeError("installed ripgrep did not report its version")
    return next(line for line in lines[1:] if line.startswith("ripgrep "))


def _disposable_provider_config(provider_config: dict[str, Any]) -> dict[str, Any]:
    """Keep the reviewed endpoint, while disabling implicit create/command retries."""
    disposable = copy.deepcopy(provider_config)
    disposable["opensandbox"].setdefault("create", {})["retries"] = 0
    disposable["opensandbox"].setdefault("operations", {})["command_retries"] = 0
    return disposable


async def run_live_probe(inputs: ProbeInputs, audit: dict[str, object] | None = None) -> dict[str, object]:
    """Start one short-lived real OpenSandbox sandbox and perform no inference."""
    if audit is None:
        audit = {}
    attempts: list[str] = []
    audit.update(
        status="failed",
        gym_commit=GYM_COMMIT,
        efb_commit=EFB_COMMIT,
        source_commit=SOURCE_COMMIT,
        script_sha256=inputs.script_sha256,
        manifest_sha256=inputs.manifest_sha256,
        resolved_run_sha256=inputs.resolved_run_sha256,
        source_sha256=inputs.source_sha256,
        task_list_sha256=inputs.task_list_sha256,
        ripgrep_sha256=RIPGREP_SHA256,
        configuration_binding="independently-reviewed-manifest",
        sandbox_id=None,
        upload_paths=attempts,
        cleanup_confirmed=False,
        inference_requests=0,
    )
    sandbox = AsyncSandbox(create_provider(_disposable_provider_config(inputs.provider_config)))
    stage = "create"
    try:
        await sandbox.start(inputs.sandbox_spec)
        sandbox_id = sandbox.sandbox_id
        audit["sandbox_id"] = sandbox_id
        if not sandbox_id:
            raise RuntimeError("OpenSandbox did not return a sandbox ID")
        stage = "upload_retry"
        await _stage_after_lost_ack(_agent_for_staging(), sandbox, inputs.rg_path, attempts)
        stage = "verify_binary"
        version = await _verify_remote_binary(sandbox, attempts, RIPGREP_SHA256)
        audit["installed_version"] = version
    except BaseException as error:
        audit["failure_stage"] = stage
        audit["failure_type"] = type(error).__name__
        raise
    finally:
        # Shield the one explicit close from the whole-route cancellation. A
        # failed or timed-out close remains a failure; the finite pod TTL is
        # the last bound if provider state cannot be confirmed.
        stop_task = asyncio.create_task(sandbox.stop())
        try:
            await asyncio.wait_for(asyncio.shield(stop_task), timeout=CLEANUP_TIMEOUT_S)
            if (
                audit["sandbox_id"] is None
                or sandbox.sandbox_id != audit["sandbox_id"]
                or not sandbox._stopped
                or sandbox._handle is None
            ):
                raise RuntimeError("cannot confirm provider teardown of the same probe sandbox")
            audit["cleanup_confirmed"] = True
        except BaseException as error:
            audit.setdefault("failure_stage", "cleanup")
            audit.setdefault("failure_type", type(error).__name__)
            audit["cleanup_error_type"] = type(error).__name__
            stop_task.cancel()
            raise
    audit["status"] = "passed"
    return audit


def _write_evidence(path: Path, audit: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="External resolved probe manifest JSON")
    parser.add_argument(
        "--reviewed-manifest-sha256", required=True, help="SHA-256 of the independently reviewed manifest"
    )
    parser.add_argument("--script-sha256", required=True, help="Independent SHA-256 of this exact probe script")
    parser.add_argument(
        "--attest-resolved-mapping-reviewed",
        action="store_true",
        help="Confirm manual comparison of manifest provider/task image against resolved run and canonical source",
    )
    parser.add_argument("--execute", action="store_true", help="Opt in to creating exactly one OpenSandbox pod")
    parser.add_argument("--evidence", type=Path, help="Required with --execute; contains no provider credentials")
    args = parser.parse_args()
    if args.execute and args.evidence is None:
        parser.error("--evidence is required with --execute")
    if args.execute and not args.attest_resolved_mapping_reviewed:
        parser.error("--execute requires --attest-resolved-mapping-reviewed")
    audit: dict[str, object] = {"status": "failed", "inference_requests": 0, "cleanup_confirmed": False}
    stage = "preflight"
    try:
        inputs = verify_preflight(args.manifest, args.reviewed_manifest_sha256, args.script_sha256)
        if not args.execute:
            print(json.dumps({"status": "preflight_passed", "manifest_sha256": inputs.manifest_sha256}))
            return
        stage = "sandbox_route"
        result = asyncio.run(asyncio.wait_for(run_live_probe(inputs, audit), timeout=inputs.sandbox_spec.ttl_s - 45))
    except BaseException as error:
        audit.setdefault("failure_stage", stage)
        audit.setdefault("failure_type", type(error).__name__)
        if args.evidence is not None:
            _write_evidence(args.evidence, audit)
        raise
    _write_evidence(args.evidence, result)
    print(json.dumps({"status": result["status"], "evidence": str(args.evidence)}))


if __name__ == "__main__":
    main()
