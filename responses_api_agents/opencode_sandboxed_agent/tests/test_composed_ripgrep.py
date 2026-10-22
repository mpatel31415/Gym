# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Local-only composition checks for ripgrep staging through the provider API."""

import hashlib
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import AsyncSandbox, SandboxBackendUnreachableError, SandboxHandle, SandboxIOTimeoutError
from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.opencode_sandboxed_agent import app as app_module
from responses_api_agents.opencode_sandboxed_agent.app import OpenCodeSandboxedAgent, OpenCodeSandboxedAgentConfig


class _LostAckFiles:
    """SDK file boundary: retain bytes before losing only the first acknowledgement."""

    def __init__(self, *, fail_all: bool = False) -> None:
        self.files: dict[str, bytes] = {}
        self.paths: list[str] = []
        self.fail_all = fail_all

    async def write_file(self, path: str, data: bytes) -> None:
        self.files[path] = bytes(data)
        self.paths.append(path)
        if self.fail_all or len(self.paths) == 1:
            raise TimeoutError("write_file acknowledgement lost")


def _server_and_sandbox(binary: Path, files: _LostAckFiles) -> tuple[OpenCodeSandboxedAgent, AsyncSandbox]:
    config = OpenCodeSandboxedAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="composed-ripgrep-test",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        model_server=ModelServerRef(type="responses_api_models", name=""),
        opencode_version="test",
        sandbox_provider="",
        sandbox_config={},
        sandbox_timeout=30,
        opencode_max_context_window=0,
        token_id_capture=True,
        preinstalled_opencode=True,
        report_agent_sandbox_oom=True,
        local_ripgrep_binary_path=binary,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": False}
    server = OpenCodeSandboxedAgent(config=config, server_client=client)
    provider = OpenSandboxProvider(operations={"retries": 0})
    sandbox = AsyncSandbox(provider, owns_provider=False)
    sandbox._handle = SandboxHandle(
        sandbox_id="local-composition-only", provider_name="opensandbox", raw=SimpleNamespace(files=files)
    )
    sandbox._stopped = False
    server._sandbox_id_to_sandbox["local-composition-only"] = sandbox
    return server, sandbox


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        cookies={"sandbox_id": "local-composition-only"},
        session={SESSION_ID_KEY: "local-composition-only"},
        state=SimpleNamespace(),
    )


@pytest.mark.asyncio
async def test_lost_ack_retries_real_provider_upload_and_installs_second_copy(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "rg"
    binary.write_bytes(b"#!/bin/sh\necho rg-composed-test\n")
    binary.chmod(0o755)
    files = _LostAckFiles()
    server, sandbox = _server_and_sandbox(binary, files)
    monkeypatch.setattr(server, "_create_opencode_config", AsyncMock(return_value={}))
    monkeypatch.setattr(app_module, "uniform", lambda *_: 0.0)

    home = tmp_path / "home"
    opencode_dir = home / ".opencode" / "bin"
    opencode_dir.mkdir(parents=True)
    opencode = opencode_dir / "opencode"
    opencode.write_text('#!/bin/sh\nif [ "$1" = --version ]; then echo test; fi\n')
    opencode.chmod(0o755)
    commands: list[str] = []

    async def execute(_handle: SandboxHandle, command: str, **_kwargs: object):
        commands.append(command)
        if len(commands) > 1:
            raise SandboxBackendUnreachableError("stop before any model execution")
        second_path = files.paths[1]
        second_copy = tmp_path / "second-upload"
        second_copy.write_bytes(files.files[second_path])
        setup_command = command.replace(second_path, shlex.quote(str(second_copy))).replace(
            "/tmp/nemo-gym-mcp-setup-error", shlex.quote(str(tmp_path / "mcp-setup-error"))
        )
        assert "/tmp/nemo-gym-mcp-setup-error" not in setup_command
        completed = subprocess.run(
            ["sh", "-c", setup_command],
            env={"HOME": str(home), "PATH": f"{opencode_dir}{os.pathsep}{os.defpath}"},
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert completed.returncode == 0, completed.stderr
        return SimpleNamespace(stdout=completed.stdout, stderr=completed.stderr, return_code=0, error_type=None)

    monkeypatch.setattr(sandbox._provider, "exec", execute)
    with pytest.raises(SandboxBackendUnreachableError, match="before any model"):
        await server.responses(
            _request(), NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Solve"}])
        )

    assert len(files.paths) == 2
    assert len(set(files.paths)) == 2
    assert all(path.startswith("/tmp/nemo-gym-ripgrep-") for path in files.paths)
    expected_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    assert all(hashlib.sha256(files.files[path]).hexdigest() == expected_hash for path in files.paths)
    installed = home / ".opencode" / "bin" / "rg"
    assert hashlib.sha256(installed.read_bytes()).hexdigest() == expected_hash
    assert files.paths[1] in commands[0]
    assert files.paths[0] not in commands[0]
    assert len(commands) == 2
    assert server._sandbox_id_to_run_result == {}


@pytest.mark.asyncio
async def test_two_provider_upload_timeouts_fail_before_agent_or_install(tmp_path: Path, monkeypatch) -> None:
    binary = tmp_path / "rg"
    binary.write_bytes(b"#!/bin/sh\necho rg-composed-test\n")
    files = _LostAckFiles(fail_all=True)
    server, sandbox = _server_and_sandbox(binary, files)
    monkeypatch.setattr(server, "_create_opencode_config", AsyncMock(return_value={}))
    monkeypatch.setattr(app_module, "uniform", lambda *_: 0.0)

    async def unexpected_exec(_handle: SandboxHandle, _command: str, **_kwargs: object):
        pytest.fail("agent execution began after exhausted ripgrep uploads")

    monkeypatch.setattr(sandbox._provider, "exec", unexpected_exec)
    with pytest.raises(SandboxIOTimeoutError, match="acknowledgement lost"):
        await server.responses(
            _request(), NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Solve"}])
        )

    assert len(files.paths) == 2
    assert len(set(files.paths)) == 2
    assert all(files.files[path] == binary.read_bytes() for path in files.paths)
    assert server._sandbox_id_to_run_result == {}
