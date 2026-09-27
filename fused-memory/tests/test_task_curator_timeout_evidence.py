"""TaskCurator LLM call sites: tool scoping and timeout evidence (task 3995).

Kept apart from ``test_task_curator.py`` so that file does not grow further;
the curator helpers are imported from it.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared.config_dir import CONFIG_DIR_PREFIX, TaskConfigDir
from shared.usage_gate import UsageGate
from test_task_curator import _agent_result, _make_config

from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig
from fused_memory.middleware.task_curator import CandidateTask, TaskCurator

_INVOKE = 'fused_memory.middleware.task_curator.invoke_with_cap_retry'

_SINGLE_OK = {'action': 'create', 'justification': 'x'}
_BATCH_OK = {'decisions': [
    {'candidate_index': 0, 'action': 'create', 'justification': 'x0'},
    {'candidate_index': 1, 'action': 'create', 'justification': 'x1'},
]}


async def _call_single(curator: TaskCurator, invoke: AsyncMock) -> None:
    with patch(_INVOKE, new=invoke):
        await curator._call_llm(
            CandidateTask(title='T'),
            pool=[],
            pool_sizes={'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0},
            start=0.0,
            project_id='p',
            project_root='/p',
        )


async def _call_batch(curator: TaskCurator, invoke: AsyncMock) -> None:
    with patch(_INVOKE, new=invoke):
        await curator._call_llm_batch(
            [CandidateTask(title='T0'), CandidateTask(title='T1')],
            pools=[[], []],
            pool_sizes_list=[{}, {}],
            start=0.0,
            project_id='p',
            project_root='/p',
        )


CallSite = Callable[[TaskCurator, AsyncMock], Awaitable[None]]

_CALL_SITES = [
    pytest.param(_call_single, _SINGLE_OK, id='single'),
    pytest.param(_call_batch, _BATCH_OK, id='batch'),
]


async def _successful_call_kwargs(
    drive: CallSite, structured: dict[str, Any], curator: TaskCurator,
) -> dict[str, Any]:
    invoke = AsyncMock(return_value=_agent_result(structured))
    await drive(curator, invoke)
    return dict(invoke.call_args.kwargs)


class TestCuratorMcpScoping:
    """Both call sites scope MCP to zero servers, strictly.

    Neither of the other two guards covers MCP. The ``'*'`` → ``--tools ''``
    substitution filters built-in and deferred tools only. The neutral cwd
    removes only the project ``.mcp.json``: at a neutral cwd an account-scoped
    claude.ai connector still connected and exposed
    ``mcp__claude_ai_Claude_Docs__create`` / ``update`` / ``delete``
    (measured on CLI 2.1.283).
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('drive', 'structured'), _CALL_SITES)
    async def test_passes_zero_server_strict_mcp_config(self, drive, structured):
        curator = TaskCurator(config=_make_config(), taskmaster=None)

        kwargs = await _successful_call_kwargs(drive, structured, curator)

        assert kwargs.get('mcp_config') == {'mcpServers': {}}
        assert kwargs.get('strict_mcp_config') is True


def _single_timeout(cfg: CuratorConfig) -> float:
    return cfg.timeout_seconds


def _batch_timeout(cfg: CuratorConfig) -> float:
    # _call_batch drives a batch of two, so one item of slack past the first.
    return min(
        cfg.timeout_seconds + cfg.per_item_slack_seconds,
        cfg.batch_timeout_cap_seconds,
    )


_CALL_SITES_WITH_TIMEOUT = [
    pytest.param(_call_single, _SINGLE_OK, _single_timeout, id='single'),
    pytest.param(_call_batch, _BATCH_OK, _batch_timeout, id='batch'),
]


def _gated_curator(config_dir_base: Path, config: FusedMemoryConfig | None = None) -> TaskCurator:
    return TaskCurator(
        config=config or _make_config(),
        taskmaster=None,
        usage_gate=MagicMock(spec=UsageGate),
        config_dir_base=config_dir_base,
    )


class TestCuratorTranscriptThreading:
    """A gated curator threads config_dir + session_id + startup_grace_secs.

    Both kwargs are what make ``transcript_turns`` stampable, and so what make
    ``is_zero_output_timeout`` transcript-authoritative instead of falling back
    to the empty-stdout ``turns==0 and cost_usd==0.0`` defaults.

    ``startup_grace_secs`` equals the call's own timeout because the curator's
    measured first-turn latency tail is longer than the 120s default. Time to
    the first assistant record over 5,484 curator transcripts: p99 117.3s,
    46 transcripts >= 120s, max 203.3s. The default would fast-kill ~0.84% of
    legitimate curator calls.

    A gate-less curator passes neither: ``invoke_with_cap_retry`` writes
    credentials into a config dir only on its gated branch, so an isolated
    ``CLAUDE_CONFIG_DIR`` there would turn every call into "Not logged in".
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('drive', 'structured', 'expected_timeout'), _CALL_SITES_WITH_TIMEOUT)
    async def test_gated_call_uses_one_per_process_config_dir(
        self, drive, structured, expected_timeout, tmp_path,
    ):
        curator = _gated_curator(tmp_path)

        kwargs = await _successful_call_kwargs(drive, structured, curator)

        config_dir = kwargs.get('config_dir')
        assert isinstance(config_dir, TaskConfigDir)
        assert config_dir.path.is_dir()
        assert config_dir.path.parent == tmp_path
        assert config_dir.path.name.startswith(CONFIG_DIR_PREFIX + 'fm-curator-')
        assert config_dir.path.name.endswith(f'-{os.getpid()}')

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('drive', 'structured', 'expected_timeout'), _CALL_SITES_WITH_TIMEOUT)
    async def test_gated_call_passes_a_uuid_session_id(
        self, drive, structured, expected_timeout, tmp_path,
    ):
        curator = _gated_curator(tmp_path)

        kwargs = await _successful_call_kwargs(drive, structured, curator)

        uuid.UUID(kwargs['session_id'])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('drive', 'structured', 'expected_timeout'), _CALL_SITES_WITH_TIMEOUT)
    async def test_gated_call_grace_equals_its_own_timeout(
        self, drive, structured, expected_timeout, tmp_path,
    ):
        config = _make_config()
        curator = _gated_curator(tmp_path, config)

        kwargs = await _successful_call_kwargs(drive, structured, curator)

        assert kwargs.get('startup_grace_secs') == kwargs['timeout_seconds']
        assert kwargs['timeout_seconds'] == expected_timeout(config.curator)

    @pytest.mark.asyncio
    async def test_successive_calls_share_the_dir_but_not_the_session(self, tmp_path):
        """A reused committed session id makes ``--session-id`` exit at once
        with 'already in use' (reify-3604), so every call needs a fresh one."""
        curator = _gated_curator(tmp_path)

        first = await _successful_call_kwargs(_call_single, _SINGLE_OK, curator)
        second = await _successful_call_kwargs(_call_single, _SINGLE_OK, curator)

        assert first['config_dir'] is second['config_dir']
        assert first['session_id'] != second['session_id']

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('drive', 'structured'), _CALL_SITES)
    async def test_gate_less_call_threads_nothing(self, drive, structured):
        curator = TaskCurator(config=_make_config(), taskmaster=None, usage_gate=None)

        kwargs = await _successful_call_kwargs(drive, structured, curator)

        assert kwargs.get('config_dir') is None
        assert kwargs.get('session_id') is None
        assert 'startup_grace_secs' not in kwargs
