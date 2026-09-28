"""TaskCurator LLM call sites: tool scoping and timeout evidence (task 3995).

Kept apart from ``test_task_curator.py`` so that file does not grow further;
the curator helpers are imported from it.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared.cli_invoke import AgentResult, TranscriptEvidence, transcript_evidence
from shared.config_dir import CONFIG_DIR_PREFIX, TaskConfigDir
from shared.usage_gate import UsageGate
from test_task_curator import _agent_result, _make_config, _pool_with_ids

from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig
from fused_memory.middleware.task_curator import (
    CandidateTask,
    CuratorDecision,
    CuratorFailureError,
    PoolWithheld,
    TaskCurator,
    _PoolEntry,
)

_CURATOR_MODULE = 'fused_memory.middleware.task_curator'
_INVOKE = f'{_CURATOR_MODULE}.invoke_with_cap_retry'
_EVIDENCE_READ = f'{_CURATOR_MODULE}.transcript_evidence_for_session'
_TRANSCRIPTS = Path(__file__).parent / 'fixtures' / 'curator_transcripts'

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


_DRIVES = [
    pytest.param(_call_single, id='single'),
    pytest.param(_call_batch, id='batch'),
]


def _fixture_lines(name: str) -> list[str]:
    return (_TRANSCRIPTS / name).read_text().splitlines()


def _schema_call_records(payload: Any) -> list[dict[str, Any]]:
    """A prompt, a thinking turn, then a StructuredOutput(*payload*) tool_use."""
    return [
        {'type': 'user', 'message': {'role': 'user', 'content': 'prompt'}},
        {'type': 'assistant', 'message': {'role': 'assistant', 'content': [
            {'type': 'thinking', 'thinking': 'redacted'},
        ]}},
        {'type': 'assistant', 'message': {'role': 'assistant', 'content': [
            {'type': 'tool_use', 'id': 'toolu_1', 'name': 'StructuredOutput', 'input': payload},
        ]}},
    ]


def _schema_tool_result(text: str, *, is_error: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {'type': 'tool_result', 'tool_use_id': 'toolu_1', 'content': text}
    if is_error:
        block['is_error'] = True
    return {'type': 'user', 'message': {'role': 'user', 'content': [block]}}


def _structured_output_lines(payload: Any) -> list[str]:
    """A two-turn transcript whose second turn's StructuredOutput(*payload*) the
    CLI accepted: tool_use, then the ``structured_output`` attachment, then the
    success tool_result."""
    records = _schema_call_records(payload) + [
        {'type': 'attachment', 'attachment': {
            'type': 'structured_output', 'data': payload, 'toolUseID': 'toolu_1',
        }},
        _schema_tool_result('Structured output provided successfully'),
    ]
    return [json.dumps(record) for record in records]


def _denied_structured_output_lines(payload: Any) -> list[str]:
    """The same two turns, but the CLI permission-denied the StructuredOutput
    call: an ``is_error`` tool_result and no acceptance record."""
    records = _schema_call_records(payload) + [
        _schema_tool_result(
            "The user doesn't want to proceed with this tool use. The tool use was rejected.",
            is_error=True,
        ),
    ]
    return [json.dumps(record) for record in records]


def _killed_run(subtype: str, transcript_turns: int | None) -> AgentResult:
    """The shape ``_parse_claude_output`` mints when stdout never arrived.

    ``turns`` and ``cost_usd`` are empty-stdout defaults; only
    ``transcript_turns`` (the watchdog's own read) is an observation.
    """
    return AgentResult(
        success=False,
        output='Agent produced no output',
        subtype=subtype,
        timed_out=True,
        duration_ms=181966,
        turns=0,
        cost_usd=0.0,
        transcript_turns=transcript_turns,
    )


def _plant_transcript(invoke_kwargs: Mapping[str, Any], lines: list[str]) -> None:
    project_dir = invoke_kwargs['config_dir'].path / 'projects' / 'neutral-cwd'
    project_dir.mkdir(parents=True, exist_ok=True)
    transcript = project_dir / f"{invoke_kwargs['session_id']}.jsonl"
    transcript.write_text('\n'.join(lines) + '\n')


def _scripted_invoker(*outcomes: tuple[list[str] | None, AgentResult]) -> AsyncMock:
    """Per call: plant the next transcript (if any) for the call's own session id,
    then return the next result."""
    remaining = iter(outcomes)

    async def invoke(**kwargs: Any) -> AgentResult:
        lines, result = next(remaining)
        if lines is not None:
            _plant_transcript(kwargs, lines)
        return result
    return AsyncMock(side_effect=invoke)


def _killed_run_invoker(
    transcript_lines: list[str] | None, *, subtype: str, transcript_turns: int | None,
) -> AsyncMock:
    return _scripted_invoker((transcript_lines, _killed_run(subtype, transcript_turns)))


async def _failure_of(drive: CallSite, curator: TaskCurator, invoke: AsyncMock) -> CuratorFailureError:
    with pytest.raises(CuratorFailureError) as excinfo:
        await drive(curator, invoke)
    return excinfo.value


def _leak_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if record.levelno == logging.WARNING and 'pure-classifier' in record.getMessage()
    ]


class TestCuratorFailureEvidence:
    """A failed gated call carries what its transcript recorded.

    ``turns`` / ``cost_usd`` on a killed run are empty-stdout defaults, so
    ``transcript_turns`` and the tool sequence are the only observations of what
    the run did: a genuine pre-turn stall (esc-curator-4) versus a run that
    wandered through tools and never answered (esc-curator-2).
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('drive', _DRIVES)
    async def test_tool_wandering_run_reports_its_tools(self, drive, tmp_path, caplog):
        curator = _gated_curator(tmp_path)
        invoke = _killed_run_invoker(
            _fixture_lines('esc_curator_2_tool_wandering.jsonl'),
            subtype='error_timeout_killed_with_progress',
            transcript_turns=6,
        )

        with caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE):
            err = await _failure_of(drive, curator, invoke)

        assert err.transcript_turns == 6
        assert err.tools_used == ('ToolSearch', 'TaskGet', 'ToolSearch')
        assert err.zero_output_timeout is False
        assert 'transcript_turns=6' in str(err)
        [leak] = _leak_warnings(caplog)
        assert 'ToolSearch' in leak
        assert 'TaskGet' in leak
        assert "--tools ''" in leak

    @pytest.mark.asyncio
    async def test_pre_turn_stall_is_a_zero_output_timeout(self, tmp_path, caplog):
        curator = _gated_curator(tmp_path)
        invoke = _killed_run_invoker(
            _fixture_lines('esc_curator_4_pre_turn_stall.jsonl'),
            subtype='error_empty_output',
            transcript_turns=0,
        )

        with caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE):
            err = await _failure_of(_call_single, curator, invoke)

        assert err.zero_output_timeout is True
        assert err.transcript_turns == 0
        assert err.tools_used == ()
        assert _leak_warnings(caplog) == []

    @pytest.mark.asyncio
    async def test_missing_transcript_leaves_tools_unknown(self, tmp_path):
        """Absence of a transcript is never reported as an empty tool list."""
        curator = _gated_curator(tmp_path)
        invoke = _killed_run_invoker(None, subtype='error_empty_output', transcript_turns=None)

        err = await _failure_of(_call_single, curator, invoke)

        assert err.tools_used is None
        assert err.transcript_turns is None
        assert err.timed_out is True
        assert err.subtype == 'error_empty_output'
        assert err.zero_output_timeout is True

    @pytest.mark.asyncio
    async def test_gate_less_curator_reads_no_evidence(self):
        curator = TaskCurator(config=_make_config(), taskmaster=None, usage_gate=None)
        invoke = _killed_run_invoker(None, subtype='error_empty_output', transcript_turns=None)

        with patch(_EVIDENCE_READ) as evidence_read:
            err = await _failure_of(_call_single, curator, invoke)

        evidence_read.assert_not_called()
        assert err.transcript_turns is None
        assert err.tools_used is None

    @pytest.mark.asyncio
    async def test_evidence_read_fault_never_replaces_the_llm_failure(self, tmp_path, caplog):
        curator = _gated_curator(tmp_path)
        invoke = _killed_run_invoker(
            None, subtype='error_timeout_killed_with_progress', transcript_turns=6,
        )

        with (
            patch(_EVIDENCE_READ, side_effect=OSError('transcript read fault')),
            caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE),
        ):
            err = await _failure_of(_call_single, curator, invoke)

        assert err.subtype == 'error_timeout_killed_with_progress'
        assert err.timed_out is True
        assert err.transcript_turns == 6
        assert err.tools_used is None
        assert any(record.levelno == logging.WARNING for record in caplog.records)


_EMPTY_POOL_SIZES = {'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0}
_SALVAGEABLE_DROP = {
    'action': 'drop',
    'target_id': '9001',
    'justification': 'j',
    'target_fingerprint': None,
    'rewritten_task': None,
}


async def _decide_single(
    curator: TaskCurator, invoke: AsyncMock, pool: list[_PoolEntry],
) -> CuratorDecision:
    with patch(_INVOKE, new=invoke):
        return await curator._call_llm(
            CandidateTask(title='T'),
            pool=pool,
            pool_sizes=_EMPTY_POOL_SIZES,
            start=0.0,
            project_id='p',
            project_root='/p',
        )


async def _decide_batch(
    curator: TaskCurator, invoke: AsyncMock, pools: list[list[_PoolEntry]],
) -> list[CuratorDecision]:
    with patch(_INVOKE, new=invoke):
        return await curator._call_llm_batch(
            [CandidateTask(title=f'T{i}') for i in range(len(pools))],
            pools=pools,
            pool_sizes_list=[_EMPTY_POOL_SIZES for _ in pools],
            start=0.0,
            project_id='p',
            project_root='/p',
        )


async def _curate(curator: TaskCurator, invoke: AsyncMock, title: str) -> CuratorDecision:
    async def corpus(*_args: Any, **_kwargs: Any):
        return _pool_with_ids(('9001', 'pending')), _EMPTY_POOL_SIZES, PoolWithheld()

    with patch.object(curator, '_build_corpus', side_effect=corpus), patch(_INVOKE, new=invoke):
        return await curator.curate(CandidateTask(title=title), project_id='p', project_root='/p')


def _breaker_config(threshold: int) -> FusedMemoryConfig:
    config = _make_config()
    config.curator.zero_output_breaker_threshold = threshold
    config.curator.zero_output_breaker_cooldown_seconds = 600.0
    return config


_ZOT = (None, _killed_run('error_empty_output', transcript_turns=0))
_SALVAGEABLE_KILL = _killed_run('error_timeout_killed_with_progress', transcript_turns=2)
_HEALTHY = (None, _agent_result(_SINGLE_OK))

_NO_COMPLETED_VERDICT = [
    pytest.param(
        _fixture_lines('esc_curator_2_tool_wandering.jsonl'), id='tools-but-no-structured-output',
    ),
    pytest.param(_fixture_lines('esc_curator_4_pre_turn_stall.jsonl'), id='no-assistant-turns'),
    pytest.param(_structured_output_lines('not a dict'), id='non-dict-structured-output'),
    pytest.param(None, id='no-transcript'),
]


class TestCuratorTranscriptSalvage:
    """A killed call whose transcript holds a StructuredOutput verdict the CLI
    accepted returns that verdict instead of raising.

    This is the transcript route, for a run whose stdout never arrived; the
    stdout route is ``TestCurateFallbacks::test_call_llm_salvages_schema_payload``.
    A salvaged verdict is held to exactly the validation a returned one is.
    """

    @pytest.mark.asyncio
    async def test_completed_verdict_is_salvaged(self, tmp_path, caplog):
        curator = _gated_curator(tmp_path)
        invoke = _scripted_invoker((_structured_output_lines(_SALVAGEABLE_DROP), _SALVAGEABLE_KILL))

        with caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE):
            decision = await _decide_single(curator, invoke, _pool_with_ids(('9001', 'pending')))

        assert decision.action == 'drop'
        assert decision.target_id == '9001'
        [salvage] = [
            record.getMessage() for record in caplog.records
            if record.levelno == logging.WARNING and 'salvag' in record.getMessage().lower()
        ]
        assert 'transcript_turns=2' in salvage
        assert 'drop' in salvage

    @pytest.mark.asyncio
    async def test_salvaged_verdict_meets_the_normal_validation(self, tmp_path):
        pool = _pool_with_ids(('42', 'pending'))
        returned = await _decide_single(
            _gated_curator(tmp_path / 'returned'),
            _scripted_invoker((None, _agent_result(_SALVAGEABLE_DROP))),
            pool,
        )

        salvaged = await _decide_single(
            _gated_curator(tmp_path / 'salvaged'),
            _scripted_invoker((_structured_output_lines(_SALVAGEABLE_DROP), _SALVAGEABLE_KILL)),
            pool,
        )

        assert salvaged.action == 'create'
        assert (salvaged.action, salvaged.target_id, salvaged.justification) == (
            returned.action, returned.target_id, returned.justification,
        )

    @pytest.mark.asyncio
    async def test_salvaged_verdict_resets_the_breaker(self, tmp_path):
        """ZOT, salvage, ZOT: had the salvage not reset the count, the second ZOT
        would reach the threshold of two and short-circuit the fourth call."""
        curator = _gated_curator(tmp_path, _breaker_config(threshold=2))
        invoke = _scripted_invoker(
            _ZOT,
            (_structured_output_lines(_SALVAGEABLE_DROP), _SALVAGEABLE_KILL),
            _ZOT,
            _HEALTHY,
        )

        decisions = [await _curate(curator, invoke, title) for title in 'ABCD']

        assert decisions[1].action == 'drop'
        assert invoke.await_count == 4
        assert 'zero-output-breaker' not in decisions[3].justification

    @pytest.mark.asyncio
    async def test_batch_verdict_is_salvaged(self, tmp_path):
        curator = _gated_curator(tmp_path)
        verdict = {'decisions': [
            {'candidate_index': 0, 'action': 'create', 'justification': 'j0'},
            {**_SALVAGEABLE_DROP, 'candidate_index': 1},
        ]}
        invoke = _scripted_invoker((_structured_output_lines(verdict), _SALVAGEABLE_KILL))

        decisions = await _decide_batch(curator, invoke, [[], _pool_with_ids(('9001', 'pending'))])

        assert [d.action for d in decisions] == ['create', 'drop']
        assert decisions[1].target_id == '9001'

    @pytest.mark.asyncio
    async def test_batch_salvage_closes_an_open_breaker(self, tmp_path):
        curator = _gated_curator(tmp_path, _breaker_config(threshold=1))
        verdict = {'decisions': [
            {'candidate_index': 0, 'action': 'create', 'justification': 'j0'},
            {'candidate_index': 1, 'action': 'create', 'justification': 'j1'},
        ]}
        invoke = _scripted_invoker(
            _ZOT,
            (_structured_output_lines(verdict), _SALVAGEABLE_KILL),
            _HEALTHY,
        )

        await _curate(curator, invoke, 'opens-the-breaker')
        await _decide_batch(curator, invoke, [[], []])
        after = await _curate(curator, invoke, 'after-salvage')

        assert invoke.await_count == 3
        assert 'zero-output-breaker' not in after.justification

    @pytest.mark.asyncio
    @pytest.mark.parametrize('transcript_lines', _NO_COMPLETED_VERDICT)
    async def test_no_completed_verdict_still_raises(self, transcript_lines, tmp_path):
        curator = _gated_curator(tmp_path)
        invoke = _scripted_invoker((transcript_lines, _SALVAGEABLE_KILL))

        with pytest.raises(CuratorFailureError):
            await _decide_single(curator, invoke, _pool_with_ids(('9001', 'pending')))

    @pytest.mark.asyncio
    async def test_gate_less_curator_never_salvages(self):
        curator = TaskCurator(config=_make_config(), taskmaster=None, usage_gate=None)
        invoke = _scripted_invoker((None, _SALVAGEABLE_KILL))
        salvageable = TranscriptEvidence(
            assistant_turns=2, accepted_schema_payload=_SALVAGEABLE_DROP, other_tool_uses=(),
        )

        with patch(_EVIDENCE_READ, return_value=salvageable), pytest.raises(CuratorFailureError):
            await _decide_single(curator, invoke, _pool_with_ids(('9001', 'pending')))


_CURATOR_DECISION_KEYS = {
    'action', 'justification', 'target_id', 'target_fingerprint', 'rewritten_task',
}


def _fixture_records(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in _fixture_lines(name)]


def _killed_run_for_fixture(name: str) -> AgentResult:
    """The killed-run result _parse_claude_output would mint for this transcript."""
    turns = sum(1 for record in _fixture_records(name) if record.get('type') == 'assistant')
    subtype = 'error_timeout_killed_with_progress' if turns > 0 else 'error_empty_output'
    return _killed_run(subtype, transcript_turns=turns)


def _salvage_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if record.levelno == logging.WARNING and 'salvag' in record.getMessage().lower()
    ]


class TestCitedTranscriptCorpus:
    """Each cited transcript, redacted into a fixture, lands on the branch it is
    Exhibit A for. A failure here is a real-transcript seam (interleaved
    attachment / queue-operation / last-prompt records) the synthetic cases
    missed."""

    @pytest.mark.parametrize(('name', 'turns', 'payload_keys', 'other_tools'), [
        pytest.param('esc_curator_4_pre_turn_stall.jsonl', 0, None, (), id='esc-curator-4'),
        pytest.param(
            'esc_curator_33_salvageable.jsonl', 2, _CURATOR_DECISION_KEYS, (), id='esc-curator-33',
        ),
        pytest.param(
            'esc_curator_2_tool_wandering.jsonl', 6, None, ('ToolSearch', 'TaskGet', 'ToolSearch'),
            id='esc-curator-2',
        ),
    ])
    def test_shared_evidence_reading(self, name, turns, payload_keys, other_tools):
        evidence = transcript_evidence(_fixture_records(name))

        assert evidence.assistant_turns == turns
        if payload_keys is None:
            assert evidence.accepted_schema_payload is None
        else:
            assert evidence.accepted_schema_payload is not None
            assert set(evidence.accepted_schema_payload) == payload_keys
        assert evidence.other_tool_uses == other_tools

    @pytest.mark.asyncio
    async def test_esc_curator_4_is_the_genuine_pre_turn_stall(self, tmp_path, caplog):
        name = 'esc_curator_4_pre_turn_stall.jsonl'
        invoke = _scripted_invoker((_fixture_lines(name), _killed_run_for_fixture(name)))

        with caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE):
            err = await _failure_of(_call_single, _gated_curator(tmp_path), invoke)

        assert err.zero_output_timeout is True
        assert err.transcript_turns == 0
        assert err.tools_used == ()
        assert _salvage_warnings(caplog) == []

    @pytest.mark.asyncio
    async def test_esc_curator_33_verdict_is_salvaged(self, tmp_path):
        name = 'esc_curator_33_salvageable.jsonl'
        verdict = transcript_evidence(_fixture_records(name)).accepted_schema_payload
        assert verdict is not None
        invoke = _scripted_invoker((_fixture_lines(name), _killed_run_for_fixture(name)))

        decision = await _decide_single(
            _gated_curator(tmp_path), invoke, _pool_with_ids((verdict['target_id'], 'pending')),
        )

        assert decision.action == verdict['action']
        assert decision.target_id == verdict['target_id']

    @pytest.mark.asyncio
    async def test_esc_curator_2_reports_its_tool_excursion(self, tmp_path, caplog):
        name = 'esc_curator_2_tool_wandering.jsonl'
        invoke = _scripted_invoker((_fixture_lines(name), _killed_run_for_fixture(name)))

        with caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE):
            err = await _failure_of(_call_single, _gated_curator(tmp_path), invoke)

        assert err.zero_output_timeout is False
        assert err.transcript_turns == 6
        assert err.tools_used == ('ToolSearch', 'TaskGet', 'ToolSearch')
        assert len(_leak_warnings(caplog)) == 1


def _schema_tool_denied() -> AgentResult:
    """The shape ``_parse_claude_output`` mints when the CLI denied StructuredOutput."""
    return AgentResult(
        success=False,
        output='StructuredOutput denied',
        structured_output=None,
        schema_tool_denied=True,
        timed_out=False,
        subtype='success',
    )


def _stdout_arrived_failure() -> AgentResult:
    """A failure that was NOT a kill: its stdout arrived and was already parsed."""
    return AgentResult(
        success=False,
        output='structured output retries exhausted',
        structured_output=None,
        timed_out=False,
        subtype='error_max_structured_output_retries',
        transcript_turns=2,
    )


_DENIED_TRANSCRIPTS = [
    pytest.param(
        _call_single, _fixture_lines('schema_tool_denied_rejected_only.jsonl'), id='single',
    ),
    pytest.param(_call_batch, _denied_structured_output_lines(_BATCH_OK), id='batch'),
]

_ACCEPTED_TRANSCRIPTS = [
    pytest.param(_call_single, _structured_output_lines(_SALVAGEABLE_DROP), id='single'),
    pytest.param(_call_batch, _structured_output_lines(_BATCH_OK), id='batch'),
]


class TestSalvageRequiresAKilledRunWithAnAcceptedVerdict:
    """Salvage is only for a run KILLED after the CLI accepted its verdict.

    A denied schema tool must reach the loud, un-suppressed
    ``curator_schema_tool_denied`` escalation whether or not the curator is
    gated, and a failure whose stdout arrived was already parsed by
    ``_parse_claude_output``, which owns stdout salvage.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('drive', 'transcript_lines'), _DENIED_TRANSCRIPTS)
    async def test_denied_schema_tool_is_raised_not_salvaged(
        self, drive, transcript_lines, tmp_path, caplog,
    ):
        invoke = _scripted_invoker((transcript_lines, _schema_tool_denied()))

        with caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE):
            err = await _failure_of(drive, _gated_curator(tmp_path), invoke)

        assert err.schema_tool_denied is True
        assert err.tools_used == ()
        assert _salvage_warnings(caplog) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('drive', 'transcript_lines'), _ACCEPTED_TRANSCRIPTS)
    async def test_denial_flag_alone_refuses_salvage(
        self, drive, transcript_lines, tmp_path, caplog,
    ):
        """Pins the result-flag guard independently of the acceptance rule."""
        denied_kill = replace(
            _killed_run('error_timeout_killed_with_progress', transcript_turns=2),
            schema_tool_denied=True,
        )
        invoke = _scripted_invoker((transcript_lines, denied_kill))

        with caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE):
            err = await _failure_of(drive, _gated_curator(tmp_path), invoke)

        assert err.schema_tool_denied is True
        assert _salvage_warnings(caplog) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('drive', 'transcript_lines'), _ACCEPTED_TRANSCRIPTS)
    async def test_failure_that_was_not_a_kill_is_not_salvaged(
        self, drive, transcript_lines, tmp_path, caplog,
    ):
        invoke = _scripted_invoker((transcript_lines, _stdout_arrived_failure()))

        with caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE):
            err = await _failure_of(drive, _gated_curator(tmp_path), invoke)

        assert err.subtype == 'error_max_structured_output_retries'
        assert _salvage_warnings(caplog) == []

    @pytest.mark.asyncio
    async def test_killed_run_with_only_a_rejected_verdict_is_raised(self, tmp_path, caplog):
        """A real CLI 2.1.283 transcript whose only StructuredOutput call failed the schema."""
        name = 'schema_rejected_only.jsonl'
        invoke = _scripted_invoker((_fixture_lines(name), _killed_run_for_fixture(name)))

        with caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE):
            err = await _failure_of(_call_single, _gated_curator(tmp_path), invoke)

        assert err.zero_output_timeout is False
        assert err.transcript_turns == 2
        assert err.tools_used == ()
        assert _salvage_warnings(caplog) == []


def _escalating_gated_curator(
    tmp_path: Path, config: FusedMemoryConfig | None = None,
) -> tuple[TaskCurator, AsyncMock]:
    escalator = AsyncMock()
    escalator.report_failure = AsyncMock(return_value=None)
    curator = TaskCurator(
        config=config or _make_config(),
        taskmaster=None,
        usage_gate=MagicMock(spec=UsageGate),
        config_dir_base=tmp_path,
        escalator=escalator,
    )
    return curator, escalator


# Each transcript carries a verdict (a valid ``drop`` of pool entry 9001) that
# a wrongful salvage would return; the acceptance record differs.
_DENIAL_TRANSCRIPTS_WITH_A_DROP = [
    pytest.param(_denied_structured_output_lines(_SALVAGEABLE_DROP), id='denied-attempt'),
    pytest.param(_structured_output_lines(_SALVAGEABLE_DROP), id='accepted-verdict'),
]


class TestGatedSchemaToolDenialEscalates:
    """The gated analogue of
    ``test_task_curator.py::test_schema_tool_denied_threads_to_report_failure``."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('transcript_lines', _DENIAL_TRANSCRIPTS_WITH_A_DROP)
    async def test_denial_reaches_report_failure(self, transcript_lines, tmp_path):
        curator, escalator = _escalating_gated_curator(tmp_path)
        invoke = _scripted_invoker((transcript_lines, _schema_tool_denied()))

        decision = await _curate(curator, invoke, 'T')

        assert decision.action == 'create'
        escalator.report_failure.assert_awaited_once()
        assert escalator.report_failure.await_args.kwargs['schema_tool_denied'] is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize('transcript_lines', _DENIAL_TRANSCRIPTS_WITH_A_DROP)
    async def test_denial_does_not_reset_the_breaker(self, transcript_lines, tmp_path):
        """ZOT, denial, ZOT: a denial is neither a ZOT nor a success, so the two
        ZOTs reach the threshold of two and the fourth call short-circuits."""
        curator, _ = _escalating_gated_curator(tmp_path, _breaker_config(threshold=2))
        invoke = _scripted_invoker(
            _ZOT, (transcript_lines, _schema_tool_denied()), _ZOT, _HEALTHY,
        )

        decisions = [await _curate(curator, invoke, title) for title in 'ABCD']

        assert decisions[1].action == 'create'
        assert invoke.await_count == 3
        assert decisions[3].justification == 'zero-output-breaker-open'
