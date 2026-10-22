# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local, no-service checks for the opt-in DeepSWE rg retry diagnostic."""

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.diagnostics import deepswe_rg_retry_probe as probe

from nemo_gym.sandbox import SandboxBackendUnreachableError, SandboxIOTimeoutError, SandboxSpec
from responses_api_agents.opencode_sandboxed_agent import app as agent_app


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FileBackedSandbox:
    """Provider boundary that keeps actual uploaded bytes and records cleanup."""

    def __init__(
        self,
        *,
        fail_second_upload: bool = False,
        fail_first_with_provider_loss: bool = False,
        bad_installed_digest: bool = False,
        hang_on_stop: bool = False,
        change_id_on_stop: bool = False,
        skip_internal_stop: bool = False,
    ) -> None:
        self.sandbox_id = "probe-task-sandbox"
        self.files: dict[str, bytes] = {}
        self.upload_paths: list[str] = []
        self.exec_commands: list[str] = []
        self.started = False
        self.stopped = False
        self.fail_second_upload = fail_second_upload
        self.fail_first_with_provider_loss = fail_first_with_provider_loss
        self.bad_installed_digest = bad_installed_digest
        self.hang_on_stop = hang_on_stop
        self.change_id_on_stop = change_id_on_stop
        self.skip_internal_stop = skip_internal_stop
        self._handle = SimpleNamespace(sandbox_id=self.sandbox_id)
        self._stopped = False

    async def start(self, spec: SandboxSpec) -> None:
        assert spec.ttl_s == 600
        self.started = True

    async def upload(self, local_path: Path | str, remote_path: str) -> None:
        self.upload_paths.append(remote_path)
        if self.fail_first_with_provider_loss and len(self.upload_paths) == 1:
            raise SandboxBackendUnreachableError("provider outcome unknown")
        if self.fail_second_upload and len(self.upload_paths) == 2:
            raise SandboxIOTimeoutError("second provider upload failed")
        self.files[remote_path] = Path(local_path).read_bytes()

    async def download(self, remote_path: str, local_path: Path) -> None:
        local_path.write_bytes(self.files[remote_path])

    async def exec(self, command: str, *, timeout_s: int) -> SimpleNamespace:
        self.exec_commands.append(command)
        staged = self.upload_paths[-1]
        installed = self.files[staged]
        digest = "0" * 64 if self.bad_installed_digest else hashlib.sha256(installed).hexdigest()
        return SimpleNamespace(
            stdout=f"{digest}  /home/probe/.opencode/bin/rg\nripgrep 14.1.0\n",
            stderr="",
            return_code=0,
            error_type=None,
        )

    async def stop(self) -> None:
        if self.hang_on_stop:
            await asyncio.Future()
        if self.change_id_on_stop:
            self.sandbox_id = "different-sandbox"
        self.stopped = True
        if not self.skip_internal_stop:
            self._stopped = True


@pytest.fixture
def manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    source = tmp_path / "source.jsonl"
    source.write_text('{"task": "example"}\n', encoding="utf-8")
    task_list = tmp_path / "task-list.json"
    task_list.write_text('["example"]\n', encoding="utf-8")
    rg = tmp_path / "rg"
    rg.write_bytes(b"#!/bin/sh\necho ripgrep 14.1.0\n")
    monkeypatch.setattr(probe, "RIPGREP_SHA256", _digest(rg))
    resolved = tmp_path / "run.sub"
    resolved.write_text(
        f"{probe.GYM_COMMIT}\n{probe.SERVING_IMAGE_SHA256}\n{probe.CHECKPOINT_REVISION}\n{rg}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        probe,
        "_git_head",
        lambda path, label: {
            "Gym": probe.GYM_COMMIT,
            "EFB": probe.EFB_COMMIT,
            "task source": probe.SOURCE_COMMIT,
        }[label],
    )
    data = {
        "efb_root": str(tmp_path),
        "source_root": str(tmp_path),
        "source_commit": probe.SOURCE_COMMIT,
        "source_file": str(source),
        "source_sha256": _digest(source),
        "task_list_file": str(task_list),
        "task_list_sha256": _digest(task_list),
        "resolved_run_file": str(resolved),
        "resolved_run_sha256": _digest(resolved),
        "ripgrep_file": str(rg),
        "ripgrep_sha256": _digest(rg),
        "provider_config": {"opensandbox": {"connection": {"domain": "sandbox.example"}}},
        "sandbox_spec": {
            "image": "task-image:exact",
            "ttl_s": 600,
            "ready_timeout_s": 60,
            "resources": {"cpu": 2, "memory_mib": 8192},
            "provider_options": {"network_policy": {"defaultAction": "deny", "egress": []}},
        },
    }
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path, _digest(path)


def test_preflight_checks_pinned_bytes_and_refuses_changed_source(manifest: tuple[Path, str], tmp_path: Path) -> None:
    path, manifest_sha256 = manifest
    inputs = probe.verify_preflight(path, manifest_sha256, _digest(Path(probe.__file__)))
    assert inputs.sandbox_spec.image == "task-image:exact"
    assert inputs.rg_path.read_bytes().startswith(b"#!/bin/sh")

    (tmp_path / "source.jsonl").write_text('{"task": "changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="task source SHA-256 mismatch"):
        probe.verify_preflight(path, manifest_sha256, _digest(Path(probe.__file__)))


def test_preflight_pins_own_script_and_exact_source_commit(manifest: tuple[Path, str]) -> None:
    path, manifest_sha256 = manifest
    with pytest.raises(ValueError, match="probe script SHA-256 mismatch"):
        probe.verify_preflight(path, manifest_sha256, "0" * 64)

    data = json.loads(path.read_text(encoding="utf-8"))
    data["source_commit"] = "a" * 40
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="task source must be exact"):
        probe.verify_preflight(path, _digest(path), _digest(Path(probe.__file__)))


def test_default_cli_never_creates_a_provider(
    manifest: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path, manifest_sha256 = manifest
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--manifest",
            str(path),
            "--reviewed-manifest-sha256",
            manifest_sha256,
            "--script-sha256",
            _digest(Path(probe.__file__)),
        ],
    )

    def forbidden_provider(_config: object) -> None:
        raise AssertionError("dry run must not create a provider")

    monkeypatch.setattr(probe, "create_provider", forbidden_provider)
    probe.main()
    assert json.loads(capsys.readouterr().out)["status"] == "preflight_passed"


def test_preflight_refuses_a_different_imported_gym(
    manifest: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest_sha256 = manifest
    monkeypatch.setattr(probe.opencode_app, "__file__", "/tmp/other-gym/app.py")
    with pytest.raises(ValueError, match="not from the pinned Gym checkout"):
        probe.verify_preflight(path, manifest_sha256, _digest(Path(probe.__file__)))


@pytest.mark.parametrize(
    ("spec_change", "message"),
    [
        ({"ttl_s": 3600}, "TTL"),
        ({"provider_options": {}}, "network policy"),
        ({"resources": {"cpu": 2, "memory_mib": 8192, "disk_gib": 500}}, "disk limit"),
    ],
)
def test_preflight_refuses_unbounded_or_permissive_sandbox(
    manifest: tuple[Path, str], spec_change: dict[str, object], message: str
) -> None:
    path, _ = manifest
    data = json.loads(path.read_text(encoding="utf-8"))
    data["sandbox_spec"].update(spec_change)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        probe.verify_preflight(path, _digest(path), _digest(Path(probe.__file__)))


@pytest.mark.parametrize(
    ("section", "value", "message"),
    [
        ("provider", {"setup": {"commands": ["echo unexpected"]}}, "no setup or image rewrite"),
        ("provider", {"images": {"rewrites": [{"from": "x", "to": "y"}]}}, "no setup or image rewrite"),
        ("provider", {"create": {"renew_interval_s": 20}}, "TTL renewal is forbidden"),
        ("sandbox", {"entrypoint": ["sh", "-c", "echo unexpected"]}, "cannot add commands"),
        (
            "sandbox",
            {"provider_options": {"network_policy": {"defaultAction": "deny", "egress": [{"action": "allow"}]}}},
            "no egress exceptions",
        ),
    ],
)
def test_preflight_refuses_hidden_execution_or_egress(
    manifest: tuple[Path, str], section: str, value: dict[str, object], message: str
) -> None:
    path, _ = manifest
    data = json.loads(path.read_text(encoding="utf-8"))
    target = data["provider_config"]["opensandbox"] if section == "provider" else data["sandbox_spec"]
    target.update(value)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        probe.verify_preflight(path, _digest(path), _digest(Path(probe.__file__)))


@pytest.mark.asyncio
async def test_lost_ack_recovers_at_new_path_and_checks_actual_uploaded_bytes(
    manifest: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest_sha256 = manifest
    inputs = probe.verify_preflight(path, manifest_sha256, _digest(Path(probe.__file__)))
    sandbox = FileBackedSandbox()
    monkeypatch.setattr(probe, "AsyncSandbox", lambda provider: sandbox)
    seen_config: list[dict[str, object]] = []
    monkeypatch.setattr(probe, "create_provider", lambda config: seen_config.append(config) or config)
    monkeypatch.setattr(agent_app, "uniform", lambda *_: 0)

    result = await probe.run_live_probe(inputs)

    assert sandbox.started and sandbox.stopped
    assert result["status"] == "passed"
    assert result["inference_requests"] == 0
    assert sandbox.upload_paths == result["upload_paths"]
    assert len(set(sandbox.upload_paths)) == 2
    assert all(hashlib.sha256(contents).hexdigest() == probe.RIPGREP_SHA256 for contents in sandbox.files.values())
    assert len(sandbox.exec_commands) == 1
    assert sandbox.upload_paths[1] in sandbox.exec_commands[0]
    assert sandbox.upload_paths[0] not in sandbox.exec_commands[0]
    assert seen_config[0]["opensandbox"]["create"]["retries"] == 0
    assert seen_config[0]["opensandbox"]["operations"]["command_retries"] == 0
    assert "create" not in inputs.provider_config["opensandbox"]
    assert result["cleanup_confirmed"] is True
    assert result["script_sha256"] == _digest(Path(probe.__file__))


@pytest.mark.asyncio
async def test_exhausted_upload_fails_closed_and_stops_sandbox(
    manifest: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest_sha256 = manifest
    inputs = probe.verify_preflight(path, manifest_sha256, _digest(Path(probe.__file__)))
    sandbox = FileBackedSandbox(fail_second_upload=True)
    monkeypatch.setattr(probe, "AsyncSandbox", lambda provider: sandbox)
    monkeypatch.setattr(probe, "create_provider", lambda config: config)
    monkeypatch.setattr(agent_app, "uniform", lambda *_: 0)

    with pytest.raises(SandboxIOTimeoutError, match="second provider upload failed"):
        await probe.run_live_probe(inputs)

    assert sandbox.started and sandbox.stopped
    assert len(sandbox.upload_paths) == 2
    assert len(set(sandbox.upload_paths)) == 2
    assert sandbox.exec_commands == []


@pytest.mark.asyncio
async def test_wrong_installed_checksum_fails_closed_and_stops_sandbox(
    manifest: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest_sha256 = manifest
    inputs = probe.verify_preflight(path, manifest_sha256, _digest(Path(probe.__file__)))
    sandbox = FileBackedSandbox(bad_installed_digest=True)
    monkeypatch.setattr(probe, "AsyncSandbox", lambda provider: sandbox)
    monkeypatch.setattr(probe, "create_provider", lambda config: config)
    monkeypatch.setattr(agent_app, "uniform", lambda *_: 0)

    with pytest.raises(RuntimeError, match="installed ripgrep has a wrong checksum"):
        await probe.run_live_probe(inputs)

    assert sandbox.started and sandbox.stopped
    assert len(sandbox.upload_paths) == 2
    assert len(sandbox.exec_commands) == 1


@pytest.mark.asyncio
async def test_non_timeout_provider_loss_is_not_retried_and_stops_sandbox(
    manifest: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest_sha256 = manifest
    inputs = probe.verify_preflight(path, manifest_sha256, _digest(Path(probe.__file__)))
    sandbox = FileBackedSandbox(fail_first_with_provider_loss=True)
    monkeypatch.setattr(probe, "AsyncSandbox", lambda provider: sandbox)
    monkeypatch.setattr(probe, "create_provider", lambda config: config)

    with pytest.raises(SandboxBackendUnreachableError, match="provider outcome unknown"):
        await probe.run_live_probe(inputs)

    assert sandbox.started and sandbox.stopped
    assert len(sandbox.upload_paths) == 1
    assert sandbox.exec_commands == []


@pytest.mark.asyncio
async def test_route_cancel_still_confirms_same_sandbox_cleanup(
    manifest: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest_sha256 = manifest
    inputs = probe.verify_preflight(path, manifest_sha256, _digest(Path(probe.__file__)))
    sandbox = FileBackedSandbox()
    monkeypatch.setattr(probe, "AsyncSandbox", lambda provider: sandbox)
    monkeypatch.setattr(probe, "create_provider", lambda config: config)

    async def never_stage(*_args: object) -> list[str]:
        await asyncio.Future()

    monkeypatch.setattr(probe, "_stage_after_lost_ack", never_stage)
    audit: dict[str, object] = {}
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(probe.run_live_probe(inputs, audit), timeout=0.01)

    assert sandbox.started and sandbox.stopped
    assert audit["sandbox_id"] == sandbox.sandbox_id
    assert audit["cleanup_confirmed"] is True
    assert audit["failure_stage"] == "upload_retry"
    assert audit["status"] == "failed"


def test_cleanup_timeout_writes_failure_evidence_with_sandbox_id_and_paths(
    manifest: tuple[Path, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path, manifest_sha256 = manifest
    evidence_path = tmp_path / "evidence.json"
    sandbox = FileBackedSandbox(hang_on_stop=True)
    monkeypatch.setattr(probe, "AsyncSandbox", lambda provider: sandbox)
    monkeypatch.setattr(probe, "create_provider", lambda config: config)
    monkeypatch.setattr(agent_app, "uniform", lambda *_: 0)
    monkeypatch.setattr(probe, "CLEANUP_TIMEOUT_S", 0.01)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--manifest",
            str(path),
            "--reviewed-manifest-sha256",
            manifest_sha256,
            "--script-sha256",
            _digest(Path(probe.__file__)),
            "--attest-resolved-mapping-reviewed",
            "--execute",
            "--evidence",
            str(evidence_path),
        ],
    )

    with pytest.raises(TimeoutError):
        probe.main()

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["status"] == "failed"
    assert evidence["sandbox_id"] == "probe-task-sandbox"
    assert evidence["cleanup_confirmed"] is False
    assert evidence["failure_stage"] == "cleanup"
    assert evidence["cleanup_error_type"] == "TimeoutError"
    assert len(evidence["upload_paths"]) == 2
    assert "provider_config" not in evidence


@pytest.mark.asyncio
async def test_changed_sandbox_id_on_stop_fails_closed(
    manifest: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest_sha256 = manifest
    inputs = probe.verify_preflight(path, manifest_sha256, _digest(Path(probe.__file__)))
    sandbox = FileBackedSandbox(change_id_on_stop=True)
    monkeypatch.setattr(probe, "AsyncSandbox", lambda provider: sandbox)
    monkeypatch.setattr(probe, "create_provider", lambda config: config)
    monkeypatch.setattr(agent_app, "uniform", lambda *_: 0)
    audit: dict[str, object] = {}

    with pytest.raises(RuntimeError, match="cannot confirm provider teardown of the same probe sandbox"):
        await probe.run_live_probe(inputs, audit)

    assert audit["status"] == "failed"
    assert audit["cleanup_confirmed"] is False
    assert audit["failure_stage"] == "cleanup"


@pytest.mark.asyncio
async def test_stop_return_without_provider_teardown_fails_closed(
    manifest: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, manifest_sha256 = manifest
    inputs = probe.verify_preflight(path, manifest_sha256, _digest(Path(probe.__file__)))
    sandbox = FileBackedSandbox(skip_internal_stop=True)
    monkeypatch.setattr(probe, "AsyncSandbox", lambda provider: sandbox)
    monkeypatch.setattr(probe, "create_provider", lambda config: config)
    monkeypatch.setattr(agent_app, "uniform", lambda *_: 0)
    audit: dict[str, object] = {}

    with pytest.raises(RuntimeError, match="cannot confirm provider teardown of the same probe sandbox"):
        await probe.run_live_probe(inputs, audit)

    assert sandbox.stopped
    assert audit["status"] == "failed"
    assert audit["cleanup_confirmed"] is False
    assert audit["failure_stage"] == "cleanup"
