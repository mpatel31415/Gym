# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the DeepSWE OOM contract with an in-process provider boundary."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from fastapi import Request
from omegaconf import OmegaConf

import nemo_gym.rollout_collection as collection_module
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.failure_kinds import AGENT_RUN_ERROR, PROVIDER_OOM_KILLED
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper, _failures_path_for
from nemo_gym.rollout_observability import NG_AGENT_SANDBOX_OBSERVATION_KEY, SandboxObservation
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, SandboxHandle, SandboxOOMKilledError
from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.deepswe.app import (
    DeepSWEResourcesServer,
    DeepSWEResourcesServerConfig,
    DeepSWESeedSessionRequest,
    DeepSWEVerifyRequest,
    VerifierResult,
)
from responses_api_agents.opencode_sandboxed_agent.app import (
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
    OpenCodeSandboxedAgentRunRequest,
)


TASK_IMAGE = "public.example/project/example-task:v1.1"
SANDBOX_ID = "offline-agent-sandbox"


class LocalResponse:
    """Only the response surface used by the in-process transport and collector."""

    ok = True
    status = 200

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.cookies: dict[str, str] = {}

    async def json(self) -> dict:
        return self.payload

    async def read(self) -> bytes:
        return orjson.dumps(self.payload)

    def release(self) -> None:
        pass


def _guard_external_sandboxes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def forbidden(*args, **kwargs):
        raise AssertionError("offline OOM contract test reached a real sandbox boundary")

    for method in ("start", "start_with_setup", "connect"):
        monkeypatch.setattr(AsyncSandbox, method, forbidden)
    for method in ("create", "connect"):
        monkeypatch.setattr(OpenSandboxProvider, method, forbidden)


def _offline_system(task_assets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str):
    _guard_external_sandboxes(monkeypatch)
    command_log: list[str] = []

    async def exec_command(command: str, **kwargs) -> SandboxExecResult:
        command_log.append(command)
        if "Installed OpenCode" in command:
            return SandboxExecResult(stdout="Installed OpenCode", stderr="", return_code=0)
        if "opencode run --title" in command:
            assert "python3 -c" not in command  # No allocator can run from this fake boundary.
            if outcome in {"matched_oom", "mismatched_oom"}:
                error_sandbox = SANDBOX_ID if outcome == "matched_oom" else "different-sandbox"
                raise SandboxOOMKilledError(
                    f"OpenSandbox confirmed OOMKilled; sandbox_id={error_sandbox!r}", sandbox_id=error_sandbox
                )
            return SandboxExecResult(stdout="Shell: bash\n", stderr="", return_code=137)
        if "opencode session list" in command:
            if outcome == "child_exit_137_with_export":
                return SandboxExecResult(stdout='[{"id":"offline-session"}]', stderr="", return_code=0)
            return SandboxExecResult(stdout="", stderr="no export", return_code=1)
        if "opencode export" in command:
            assert outcome == "child_exit_137_with_export"
            return SandboxExecResult(stdout="", stderr="", return_code=0)
        if "opencode db" in command:
            assert outcome == "child_exit_137_with_export"
            return SandboxExecResult(stdout="", stderr="snapshot unavailable", return_code=1)
        raise AssertionError(f"unexpected fake sandbox command: {command[:100]}")

    async def download(_remote: str, local: Path) -> None:
        assert outcome == "child_exit_137_with_export"
        assert local.name == "export.json"
        export_fixture = (
            Path(__file__).parents[3]
            / "responses_api_agents/opencode_sandboxed_agent/tests/opencode_export_test_data.json"
        )
        local.write_bytes(export_fixture.read_bytes())

    sandbox = MagicMock()
    sandbox.exec = AsyncMock(side_effect=exec_command)
    sandbox.serialize = AsyncMock(return_value={"sandbox_id": SANDBOX_ID, "workdir": "/app"})
    sandbox.stop = AsyncMock()
    sandbox.upload = AsyncMock(side_effect=AssertionError("no upload in offline OOM contract test"))
    sandbox.download = AsyncMock(side_effect=download)
    sandbox._handle = SandboxHandle(sandbox_id=SANDBOX_ID, provider_name="opensandbox", raw=None)

    resource_client = MagicMock(spec=ServerClient)
    resource_client.post = AsyncMock(side_effect=AssertionError("offline test must not call a model server"))
    resource = DeepSWEResourcesServer(
        config=DeepSWEResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="offline_deepswe",
            tasks_dir=task_assets,
            expected_task_count=1,
            is_verifying_golden_patch=False,
            score_agent_sandbox_oom_as_zero=True,
            sandbox_provider="forbidden",
            sandbox_config={},
            logs_dir=tmp_path / "logs",
        ),
        server_client=resource_client,
    )
    monkeypatch.setattr(resource, "_create_sandbox", AsyncMock(return_value=sandbox))
    if outcome == "child_exit_137_with_export":
        monkeypatch.setattr(resource, "_collect_model_patch", AsyncMock(return_value=b"offline model patch\n"))
        monkeypatch.setattr(
            resource,
            "_run_verifier",
            AsyncMock(return_value=VerifierResult(evaluation_completed=True, reward=0.0, verifier_exit_code=0)),
        )
    else:
        monkeypatch.setattr(
            resource, "_collect_model_patch", AsyncMock(side_effect=AssertionError("OOM must not collect a patch"))
        )
        monkeypatch.setattr(
            resource, "_run_verifier", AsyncMock(side_effect=AssertionError("OOM must not run a verifier"))
        )

    agent_client = MagicMock(spec=ServerClient)
    agent_client.get = AsyncMock(side_effect=AssertionError("offline test must not call a model server"))
    agent = OpenCodeSandboxedAgent(
        config=OpenCodeSandboxedAgentConfig(
            host="0.0.0.0",
            port=8081,
            entrypoint="",
            name="offline_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="offline_deepswe"),
            model_server=ModelServerRef(type="responses_api_models", name="forbidden_model"),
            opencode_version="offline-test",
            opencode_max_context_window=32768,
            sandbox_provider="forbidden",
            sandbox_config={},
            sandbox_timeout=120,
            preinstalled_opencode=True,
            report_agent_sandbox_oom=True,
            execution_failure_reward_zero=True,
            artifacts_dir=str(tmp_path / "agent-artifacts"),
        ),
        server_client=agent_client,
    )
    monkeypatch.setattr(agent, "_start_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(agent, "_create_opencode_config", AsyncMock(return_value={}))

    async def resource_post(*, server_name: str, url_path: str, json: dict, cookies: dict, **kwargs) -> LocalResponse:
        assert server_name == "offline_deepswe"
        request = Request({"type": "http", "headers": [], "session": {SESSION_ID_KEY: "offline-session"}})
        if url_path == "/seed_session":
            assert cookies == {}
            result = await resource.seed_session(request, DeepSWESeedSessionRequest.model_validate(json))
        elif url_path == "/verify":
            assert cookies == {"sandbox_id": "offline-session"}
            result = await resource.verify(request, DeepSWEVerifyRequest.model_validate(json))
        else:
            raise AssertionError(f"unexpected resource route {url_path}")
        return LocalResponse(result.model_dump(mode="json"))

    agent_client.post = AsyncMock(side_effect=resource_post)
    body = OpenCodeSandboxedAgentRunRequest.model_validate(
        {
            "task_id": "example-task",
            "image": TASK_IMAGE,
            "verifier_metadata": {"task_id": "example-task"},
            "responses_create_params": {"input": [{"role": "user", "content": "Solve the task"}]},
            "capture_rollout_id": "offline-oom-contract",
        }
    )
    return SimpleNamespace(
        agent=agent, body=body, sandbox=sandbox, resource=resource, agent_client=agent_client, command_log=command_log
    )


async def test_confirmed_same_sandbox_oom_is_one_unmasked_collector_main(
    task_assets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = _offline_system(task_assets, tmp_path, monkeypatch, "matched_oom")
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps(system.body.model_dump(mode="json") | {"agent_ref": {"name": "offline_agent"}}) + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "output.jsonl"
    collector_client = MagicMock(spec=ServerClient)
    collector_client.global_config_dict = OmegaConf.create(
        {
            "offline_agent": {"responses_api_agents": {"opencode_sandboxed_agent": {}}},
            "offline_environment": {
                "environment_servers": {"legacy_agent": {"agent_server": {"name": "offline_agent"}}}
            },
        }
    )
    aggregates: list[list[dict]] = []

    async def collector_post(*, server_name: str, url_path: str, json, **kwargs) -> LocalResponse:
        if url_path == "/run":
            assert server_name == "offline_environment"
            request = Request({"type": "http", "headers": [], "session": {SESSION_ID_KEY: "offline-session"}})
            result = await system.agent.run(request, OpenCodeSandboxedAgentRunRequest.model_validate(json))
            return LocalResponse(result.model_dump(mode="json"))
        assert url_path == "/aggregate_metrics"
        aggregates.append([dict(row) for row in json.verify_responses])
        return LocalResponse(compute_aggregate_metrics(json.verify_responses).model_dump(mode="json"))

    collector_client.post = AsyncMock(side_effect=collector_post)
    monkeypatch.setattr(collection_module, "setup_server_client_utils", lambda *args, **kwargs: collector_client)
    monkeypatch.setattr(collection_module, "get_global_config_dict", lambda: {})
    await RolloutCollectionHelper().run_from_config(
        RolloutCollectionConfig(
            input_jsonl_fpath=str(input_path),
            output_jsonl_fpath=str(output_path),
            route_failures_to_sidecar=True,
            disable_health_check=True,
            require_complete=True,
        )
    )

    rows = [orjson.loads(line) for line in output_path.read_bytes().splitlines()]
    assert len(rows) == 1
    assert _failures_path_for(output_path).read_bytes() == b""
    assert len(aggregates) == 1 and len(aggregates[0]) == 1
    row = rows[0]
    assert row["reward"] == 0.0 and row["mask_sample"] is False
    assert row["failure_kind"] == PROVIDER_OOM_KILLED
    assert row["evaluation_completed"] is False
    assert row["verifier_exit_code"] is None and row["model_patch_bytes"] == 0
    assert row["opencode_error_type"] == SandboxOOMKilledError.__name__
    observation = SandboxObservation.model_validate_json(row["response"]["metadata"][NG_AGENT_SANDBOX_OBSERVATION_KEY])
    assert (
        observation.role,
        observation.phase,
        observation.outcome,
        observation.provider,
        observation.sandbox_id,
    ) == (
        "agent",
        "agent_execution",
        "oom",
        "opensandbox",
        SANDBOX_ID,
    )
    assert aggregates[0][0]["failure_kind"] == PROVIDER_OOM_KILLED
    assert system.resource._create_sandbox.await_count == 1
    system.resource._collect_model_patch.assert_not_awaited()
    system.resource._run_verifier.assert_not_awaited()
    assert len(system.command_log) == 2 and "opencode run --title" in system.command_log[1]
    system.sandbox.download.assert_not_awaited()
    assert system.resource._agent_sessions == {}
    assert system.sandbox.stop.await_count == 2  # Resource verification and agent-finally cleanup.


@pytest.mark.parametrize(
    ("outcome", "error_type", "message"),
    [
        ("mismatched_oom", ValueError, "does not match the seeded"),
        ("child_exit_137", RuntimeError, "Failed to list OpenCode"),
    ],
)
async def test_unconfirmed_agent_outcome_never_becomes_an_oom_zero(
    task_assets: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    error_type: type[Exception],
    message: str,
) -> None:
    system = _offline_system(task_assets, tmp_path, monkeypatch, outcome)
    request = Request({"type": "http", "headers": [], "session": {SESSION_ID_KEY: "offline-session"}})

    with pytest.raises(error_type, match=message):
        await system.agent.run(request, system.body)

    assert any("opencode run --title" in command for command in system.command_log)
    assert [call.kwargs["url_path"] for call in system.agent_client.post.await_args_list] == (
        ["/seed_session", "/verify"] if outcome == "mismatched_oom" else ["/seed_session"]
    )
    system.resource._collect_model_patch.assert_not_awaited()
    system.resource._run_verifier.assert_not_awaited()
    system.sandbox.download.assert_not_awaited()
    assert system.sandbox.stop.await_count >= 1


async def test_plain_child_exit_137_with_export_is_a_verified_agent_failure_not_provider_oom(
    task_assets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = _offline_system(task_assets, tmp_path, monkeypatch, "child_exit_137_with_export")
    request = Request({"type": "http", "headers": [], "session": {SESSION_ID_KEY: "offline-session"}})

    result = await system.agent.run(request, system.body)

    assert result.reward == 0.0 and result.mask_sample is False
    assert result.failure_kind == AGENT_RUN_ERROR
    assert result.evaluation_completed is True and result.verifier_exit_code == 0
    assert result.model_patch_bytes == len(b"offline model patch\n")
    assert result.opencode_exit_code == 137 and result.opencode_export_found is True
    assert NG_AGENT_SANDBOX_OBSERVATION_KEY not in (result.response.metadata or {})
    assert [call.kwargs["url_path"] for call in system.agent_client.post.await_args_list] == [
        "/seed_session",
        "/verify",
    ]
    system.resource._collect_model_patch.assert_awaited_once()
    system.resource._run_verifier.assert_awaited_once()
    assert system.sandbox.stop.await_count >= 1
