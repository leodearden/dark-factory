"""TaskCurator LLM call sites: tool scoping and timeout evidence (task 3995).

Driven through ``curate`` / ``curate_batch`` with the corpus build and the LLM
call stubbed. A failed single call is observed where the curator reports it,
on its escalator. A failed batch is bisected rather than escalated, so it is
observed through its log and the calls its halves make.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _curator_helpers import agent_result, make_config, pool_with_ids
from _fm_helpers import LoopFreedomProbe
from shared.cli_invoke import AgentResult, TranscriptEvidence, transcript_evidence
from shared.config_dir import CONFIG_DIR_PREFIX, TaskConfigDir
from shared.usage_gate import UsageGate

from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig
from fused_memory.middleware.task_curator import (
    CandidateTask,
    CuratorDecision,
    PoolWithheld,
    TaskCurator,
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
_SALVAGEABLE_DROP = {
    'action': 'drop',
    'target_id': '9001',
    'justification': 'j',
    'target_fingerprint': None,
    'rewritten_task': None,
}

# (task_id, status) pairs; the default pool holds the drop target above.
Pool = tuple[tuple[str, str], ...]
_DROP_TARGET_POOL: Pool = (('9001', 'pending'),)
_EMPTY_POOL_SIZES = {'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0}


def _stub_corpus(curator: TaskCurator, pools_by_title: Mapping[str, Pool]):
    async def build(candidate: CandidateTask, *_args: Any, **_kwargs: Any):
        pool = pool_with_ids(*pools_by_title.get(candidate.title, ()))
        return pool, dict(_EMPTY_POOL_SIZES), PoolWithheld()
    return patch.object(curator, '_build_corpus', side_effect=build)


async def _curate(
    curator: TaskCurator, invoke: AsyncMock, title: str = 'T', pool: Pool = _DROP_TARGET_POOL,
) -> CuratorDecision:
    with _stub_corpus(curator, {title: pool}), patch(_INVOKE, new=invoke):
        return await curator.curate(CandidateTask(title=title), project_id='p', project_root='/p')


async def _curate_batch(
    curator: TaskCurator, invoke: AsyncMock, pools: Sequence[Pool] = ((), ()),
) -> list[CuratorDecision]:
    titles = [f'T{i}' for i in range(len(pools))]
    with _stub_corpus(curator, dict(zip(titles, pools, strict=True))), patch(_INVOKE, new=invoke):
        return await curator.curate_batch(
            [CandidateTask(title=title) for title in titles], project_id='p', project_root='/p',
        )


def _gated_curator(
    config_dir_base: Path, config: FusedMemoryConfig | None = None,
) -> tuple[TaskCurator, AsyncMock]:
    """A curator with a UsageGate, and the escalator its failures reach."""
    escalator = AsyncMock()
    escalator.report_failure = AsyncMock(return_value=None)
    curator = TaskCurator(
        config=config or make_config(),
        taskmaster=None,
        usage_gate=MagicMock(spec=UsageGate),
        config_dir_base=config_dir_base,
        escalator=escalator,
    )
    return curator, escalator


def _gate_less_curator() -> tuple[TaskCurator, AsyncMock]:
    escalator = AsyncMock()
    escalator.report_failure = AsyncMock(return_value=None)
    curator = TaskCurator(config=make_config(), taskmaster=None, usage_gate=None, escalator=escalator)
    return curator, escalator


def _reported_failure(escalator: AsyncMock) -> dict[str, Any]:
    escalator.report_failure.assert_awaited_once()
    return dict(escalator.report_failure.await_args.kwargs)


async def _single_call_kwargs(curator: TaskCurator) -> dict[str, Any]:
    invoke = AsyncMock(return_value=agent_result(_SINGLE_OK))
    await _curate(curator, invoke)
    return dict(invoke.call_args.kwargs)


async def _batch_call_kwargs(curator: TaskCurator) -> dict[str, Any]:
    invoke = AsyncMock(return_value=agent_result(_BATCH_OK))
    await _curate_batch(curator, invoke)
    invoke.assert_awaited_once()
    return dict(invoke.call_args.kwargs)


def _single_timeout(cfg: CuratorConfig) -> float:
    return cfg.timeout_seconds


def _batch_timeout(cfg: CuratorConfig) -> float:
    # _curate_batch sends a batch of two, so one item of slack past the first.
    return min(
        cfg.timeout_seconds + cfg.per_item_slack_seconds,
        cfg.batch_timeout_cap_seconds,
    )


_CALL_SITES = [
    pytest.param(_single_call_kwargs, _single_timeout, id='single'),
    pytest.param(_batch_call_kwargs, _batch_timeout, id='batch'),
]


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
    @pytest.mark.parametrize(('call_kwargs', 'expected_timeout'), _CALL_SITES)
    async def test_passes_zero_server_strict_mcp_config(self, call_kwargs, expected_timeout):
        curator, _ = _gate_less_curator()

        kwargs = await call_kwargs(curator)

        assert kwargs.get('mcp_config') == {'mcpServers': {}}
        assert kwargs.get('strict_mcp_config') is True


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

    A gate-less curator passes neither: no per-call OAuth token is passed
    there, so an isolated ``CLAUDE_CONFIG_DIR`` would turn every call into
    "Not logged in".
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('call_kwargs', 'expected_timeout'), _CALL_SITES)
    async def test_gated_call_uses_one_per_process_config_dir(
        self, call_kwargs, expected_timeout, tmp_path,
    ):
        curator, _ = _gated_curator(tmp_path)

        kwargs = await call_kwargs(curator)

        config_dir = kwargs.get('config_dir')
        assert isinstance(config_dir, TaskConfigDir)
        assert config_dir.path.is_dir()
        assert config_dir.path.parent == tmp_path
        assert config_dir.path.name.startswith(CONFIG_DIR_PREFIX + 'fm-curator-')
        assert config_dir.path.name.endswith(f'-{os.getpid()}')

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('call_kwargs', 'expected_timeout'), _CALL_SITES)
    async def test_gated_call_passes_a_uuid_session_id(
        self, call_kwargs, expected_timeout, tmp_path,
    ):
        curator, _ = _gated_curator(tmp_path)

        kwargs = await call_kwargs(curator)

        uuid.UUID(kwargs['session_id'])

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('call_kwargs', 'expected_timeout'), _CALL_SITES)
    async def test_gated_call_grace_equals_its_own_timeout(
        self, call_kwargs, expected_timeout, tmp_path,
    ):
        config = make_config()
        curator, _ = _gated_curator(tmp_path, config)

        kwargs = await call_kwargs(curator)

        assert kwargs.get('startup_grace_secs') == kwargs['timeout_seconds']
        assert kwargs['timeout_seconds'] == expected_timeout(config.curator)

    @pytest.mark.asyncio
    async def test_successive_calls_share_the_dir_but_not_the_session(self, tmp_path):
        """A reused committed session id makes ``--session-id`` exit at once
        with 'already in use' (reify-3604), so every call needs a fresh one."""
        curator, _ = _gated_curator(tmp_path)
        invoke = AsyncMock(return_value=agent_result(_SINGLE_OK))

        await _curate(curator, invoke, title='first')
        await _curate(curator, invoke, title='second')

        first, second = (call.kwargs for call in invoke.await_args_list)
        assert first['config_dir'] is second['config_dir']
        assert first['session_id'] != second['session_id']

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('call_kwargs', 'expected_timeout'), _CALL_SITES)
    async def test_gate_less_call_threads_nothing(self, call_kwargs, expected_timeout):
        curator, _ = _gate_less_curator()

        kwargs = await call_kwargs(curator)

        assert kwargs.get('config_dir') is None
        assert kwargs.get('session_id') is None
        assert 'startup_grace_secs' not in kwargs


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


def _warnings_mentioning(caplog: pytest.LogCaptureFixture, fragment: str) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if record.levelno == logging.WARNING and fragment in record.getMessage().lower()
    ]


def _leak_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return _warnings_mentioning(caplog, 'pure-classifier')


def _salvage_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return _warnings_mentioning(caplog, 'salvag')


_ZOT = (None, _killed_run('error_empty_output', transcript_turns=0))
_SALVAGEABLE_KILL = _killed_run('error_timeout_killed_with_progress', transcript_turns=2)
_HEALTHY = (None, agent_result(_SINGLE_OK))
_TOOL_WANDERING = 'esc_curator_2_tool_wandering.jsonl'
_WANDERED_TOOLS = ('ToolSearch', 'TaskGet', 'ToolSearch')


class TestCuratorFailureEvidence:
    """A failed gated call carries what its transcript recorded.

    ``turns`` / ``cost_usd`` on a killed run are empty-stdout defaults, so
    ``transcript_turns`` and the tool sequence are the only observations of what
    the run did: a genuine pre-turn stall (esc-curator-4) versus a run that
    wandered through tools and never answered (esc-curator-2).
    """

    @pytest.mark.asyncio
    async def test_tool_wandering_run_reports_its_tools(self, tmp_path, caplog):
        curator, escalator = _gated_curator(tmp_path)
        invoke = _killed_run_invoker(
            _fixture_lines(_TOOL_WANDERING),
            subtype='error_timeout_killed_with_progress',
            transcript_turns=6,
        )

        with caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE):
            await _curate(curator, invoke)

        report = _reported_failure(escalator)
        assert report['transcript_turns'] == 6
        assert report['tools_used'] == _WANDERED_TOOLS
        assert report['zero_output_timeout'] is False
        assert 'transcript_turns=6' in report['justification']
        [leak] = _leak_warnings(caplog)
        assert 'ToolSearch' in leak
        assert 'TaskGet' in leak
        assert "--tools ''" in leak

    @pytest.mark.asyncio
    async def test_tool_wandering_batch_logs_its_tools(self, tmp_path, caplog):
        """A failed batch is bisected, not escalated, so its log is its only report."""
        curator, escalator = _gated_curator(tmp_path)
        invoke = _scripted_invoker(
            (
                _fixture_lines(_TOOL_WANDERING),
                _killed_run('error_timeout_killed_with_progress', transcript_turns=6),
            ),
            _HEALTHY,
            _HEALTHY,
        )

        with caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE):
            decisions = await _curate_batch(curator, invoke)

        assert [d.action for d in decisions] == ['create', 'create']
        assert invoke.await_count == 3
        escalator.report_failure.assert_not_awaited()
        [leak] = _leak_warnings(caplog)
        assert 'ToolSearch' in leak
        assert 'TaskGet' in leak

    @pytest.mark.asyncio
    async def test_pre_turn_stall_is_a_zero_output_timeout(self, tmp_path, caplog):
        curator, escalator = _gated_curator(tmp_path)
        invoke = _killed_run_invoker(
            _fixture_lines('esc_curator_4_pre_turn_stall.jsonl'),
            subtype='error_empty_output',
            transcript_turns=0,
        )

        with caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE):
            await _curate(curator, invoke)

        report = _reported_failure(escalator)
        assert report['zero_output_timeout'] is True
        assert report['transcript_turns'] == 0
        assert report['tools_used'] == ()
        assert _leak_warnings(caplog) == []

    @pytest.mark.asyncio
    async def test_missing_transcript_leaves_tools_unknown(self, tmp_path):
        """Absence of a transcript is never reported as an empty tool list."""
        curator, escalator = _gated_curator(tmp_path)
        invoke = _killed_run_invoker(None, subtype='error_empty_output', transcript_turns=None)

        await _curate(curator, invoke)

        report = _reported_failure(escalator)
        assert report['tools_used'] is None
        assert report['transcript_turns'] is None
        assert report['timed_out'] is True
        assert report['subtype'] == 'error_empty_output'
        assert report['zero_output_timeout'] is True

    @pytest.mark.asyncio
    async def test_gate_less_curator_reads_no_evidence(self):
        curator, escalator = _gate_less_curator()
        invoke = _killed_run_invoker(None, subtype='error_empty_output', transcript_turns=None)

        with patch(_EVIDENCE_READ) as evidence_read:
            await _curate(curator, invoke)

        evidence_read.assert_not_called()
        report = _reported_failure(escalator)
        assert report['transcript_turns'] is None
        assert report['tools_used'] is None

    @pytest.mark.asyncio
    async def test_evidence_read_fault_never_replaces_the_llm_failure(self, tmp_path, caplog):
        curator, escalator = _gated_curator(tmp_path)
        invoke = _killed_run_invoker(
            None, subtype='error_timeout_killed_with_progress', transcript_turns=6,
        )

        with (
            patch(_EVIDENCE_READ, side_effect=OSError('transcript read fault')),
            caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE),
        ):
            await _curate(curator, invoke)

        report = _reported_failure(escalator)
        assert report['subtype'] == 'error_timeout_killed_with_progress'
        assert report['timed_out'] is True
        assert report['transcript_turns'] == 6
        assert report['tools_used'] is None
        assert _warnings_mentioning(caplog, 'could not read the transcript')

    @pytest.mark.asyncio
    async def test_evidence_read_leaves_the_event_loop_free(self, tmp_path):
        """A ZOT burst bisects into concurrent failed calls, each reading its
        transcript on the fused-memory server's loop unless offloaded."""
        probe = LoopFreedomProbe()

        def blocking_read(*_args: Any) -> None:
            probe.block()

        curator, _ = _gated_curator(tmp_path)
        invoke = _killed_run_invoker(None, subtype='error_empty_output', transcript_turns=0)

        with patch(_EVIDENCE_READ, side_effect=blocking_read):
            await _curate(curator, invoke)

        probe.assert_loop_stayed_free()


def _breaker_config(threshold: int) -> FusedMemoryConfig:
    config = make_config()
    config.curator.zero_output_breaker_threshold = threshold
    config.curator.zero_output_breaker_cooldown_seconds = 600.0
    return config


_NO_COMPLETED_VERDICT = [
    pytest.param(_fixture_lines(_TOOL_WANDERING), id='tools-but-no-structured-output'),
    pytest.param(_fixture_lines('esc_curator_4_pre_turn_stall.jsonl'), id='no-assistant-turns'),
    pytest.param(_structured_output_lines('not a dict'), id='non-dict-structured-output'),
    pytest.param(None, id='no-transcript'),
]


class TestCuratorTranscriptSalvage:
    """A killed call whose transcript holds a StructuredOutput verdict the CLI
    accepted returns that verdict instead of failing.

    This is the transcript route, for a run whose stdout never arrived; the
    stdout route is ``TestCurateFallbacks::test_call_llm_salvages_schema_payload``.
    A salvaged verdict is held to exactly the validation a returned one is.
    """

    @pytest.mark.asyncio
    async def test_completed_verdict_is_salvaged(self, tmp_path, caplog):
        curator, escalator = _gated_curator(tmp_path)
        invoke = _scripted_invoker((_structured_output_lines(_SALVAGEABLE_DROP), _SALVAGEABLE_KILL))

        with caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE):
            decision = await _curate(curator, invoke)

        assert decision.action == 'drop'
        assert decision.target_id == '9001'
        escalator.report_failure.assert_not_awaited()
        [salvage] = _salvage_warnings(caplog)
        assert 'transcript_turns=2' in salvage
        assert 'drop' in salvage

    @pytest.mark.asyncio
    async def test_salvaged_verdict_meets_the_normal_validation(self, tmp_path):
        pool: Pool = (('42', 'pending'),)
        returned_by, _ = _gated_curator(tmp_path / 'returned')
        returned = await _curate(
            returned_by, _scripted_invoker((None, agent_result(_SALVAGEABLE_DROP))), pool=pool,
        )

        salvaged_by, _ = _gated_curator(tmp_path / 'salvaged')
        salvaged = await _curate(
            salvaged_by,
            _scripted_invoker((_structured_output_lines(_SALVAGEABLE_DROP), _SALVAGEABLE_KILL)),
            pool=pool,
        )

        assert salvaged.action == 'create'
        assert (salvaged.action, salvaged.target_id, salvaged.justification) == (
            returned.action, returned.target_id, returned.justification,
        )

    @pytest.mark.asyncio
    async def test_salvaged_verdict_resets_the_breaker(self, tmp_path):
        """ZOT, salvage, ZOT: had the salvage not reset the count, the second ZOT
        would reach the threshold of two and short-circuit the fourth call."""
        curator, _ = _gated_curator(tmp_path, _breaker_config(threshold=2))
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
        curator, _ = _gated_curator(tmp_path)
        verdict = {'decisions': [
            {'candidate_index': 0, 'action': 'create', 'justification': 'j0'},
            {**_SALVAGEABLE_DROP, 'candidate_index': 1},
        ]}
        invoke = _scripted_invoker((_structured_output_lines(verdict), _SALVAGEABLE_KILL))

        decisions = await _curate_batch(curator, invoke, pools=((), _DROP_TARGET_POOL))

        assert [d.action for d in decisions] == ['create', 'drop']
        assert decisions[1].target_id == '9001'
        assert invoke.await_count == 1

    @pytest.mark.asyncio
    async def test_batch_salvage_resets_the_breaker(self, tmp_path):
        """ZOT, batch salvage, ZOT: as for a single salvage, the fourth call
        still reaches the LLM."""
        curator, _ = _gated_curator(tmp_path, _breaker_config(threshold=2))
        invoke = _scripted_invoker(
            _ZOT,
            (_structured_output_lines(_BATCH_OK), _SALVAGEABLE_KILL),
            _ZOT,
            _HEALTHY,
        )

        await _curate(curator, invoke, 'A')
        await _curate_batch(curator, invoke)
        await _curate(curator, invoke, 'C')
        after = await _curate(curator, invoke, 'D')

        assert invoke.await_count == 4
        assert 'zero-output-breaker' not in after.justification

    @pytest.mark.asyncio
    @pytest.mark.parametrize('transcript_lines', _NO_COMPLETED_VERDICT)
    async def test_no_completed_verdict_is_escalated(self, transcript_lines, tmp_path):
        curator, escalator = _gated_curator(tmp_path)
        invoke = _scripted_invoker((transcript_lines, _SALVAGEABLE_KILL))

        decision = await _curate(curator, invoke)

        assert decision.action == 'create'
        assert _reported_failure(escalator)['subtype'] == 'error_timeout_killed_with_progress'

    @pytest.mark.asyncio
    async def test_gate_less_curator_never_salvages(self):
        curator, escalator = _gate_less_curator()
        invoke = _scripted_invoker((None, _SALVAGEABLE_KILL))
        salvageable = TranscriptEvidence(
            assistant_turns=2, accepted_schema_payload=_SALVAGEABLE_DROP, other_tool_uses=(),
        )

        with patch(_EVIDENCE_READ, return_value=salvageable):
            decision = await _curate(curator, invoke)

        assert decision.action == 'create'
        escalator.report_failure.assert_awaited_once()


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


def _fixture_invoker(name: str) -> AsyncMock:
    return _scripted_invoker((_fixture_lines(name), _killed_run_for_fixture(name)))


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
        pytest.param(_TOOL_WANDERING, 6, None, _WANDERED_TOOLS, id='esc-curator-2'),
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
        curator, escalator = _gated_curator(tmp_path)

        with caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE):
            await _curate(curator, _fixture_invoker('esc_curator_4_pre_turn_stall.jsonl'))

        report = _reported_failure(escalator)
        assert report['zero_output_timeout'] is True
        assert report['transcript_turns'] == 0
        assert report['tools_used'] == ()
        assert _salvage_warnings(caplog) == []

    @pytest.mark.asyncio
    async def test_esc_curator_33_verdict_is_salvaged(self, tmp_path):
        name = 'esc_curator_33_salvageable.jsonl'
        verdict = transcript_evidence(_fixture_records(name)).accepted_schema_payload
        assert verdict is not None
        curator, _ = _gated_curator(tmp_path)

        decision = await _curate(
            curator, _fixture_invoker(name), pool=((verdict['target_id'], 'pending'),),
        )

        assert decision.action == verdict['action']
        assert decision.target_id == verdict['target_id']

    @pytest.mark.asyncio
    async def test_esc_curator_2_reports_its_tool_excursion(self, tmp_path, caplog):
        curator, escalator = _gated_curator(tmp_path)

        with caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE):
            await _curate(curator, _fixture_invoker(_TOOL_WANDERING))

        report = _reported_failure(escalator)
        assert report['zero_output_timeout'] is False
        assert report['transcript_turns'] == 6
        assert report['tools_used'] == _WANDERED_TOOLS
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


def _denied_kill() -> AgentResult:
    """A killed run flagged as denied, which pins the result-flag guard
    independently of the acceptance rule."""
    return replace(
        _killed_run('error_timeout_killed_with_progress', transcript_turns=2),
        schema_tool_denied=True,
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


# Each transcript carries a verdict a wrongful salvage would return.
_UNSALVAGEABLE_SINGLE = [
    pytest.param(
        _fixture_lines('schema_tool_denied_rejected_only.jsonl'), _schema_tool_denied(),
        id='denied-schema-tool',
    ),
    pytest.param(
        _structured_output_lines(_SALVAGEABLE_DROP), _denied_kill(), id='denial-flag-alone',
    ),
    pytest.param(
        _structured_output_lines(_SALVAGEABLE_DROP), _stdout_arrived_failure(), id='not-a-kill',
    ),
]

_UNSALVAGEABLE_BATCH = [
    pytest.param(
        _denied_structured_output_lines(_BATCH_OK), _schema_tool_denied(), id='denied-schema-tool',
    ),
    pytest.param(_structured_output_lines(_BATCH_OK), _denied_kill(), id='denial-flag-alone'),
    pytest.param(_structured_output_lines(_BATCH_OK), _stdout_arrived_failure(), id='not-a-kill'),
]


class TestSalvageRequiresAKilledRunWithAnAcceptedVerdict:
    """Salvage is only for a run KILLED after the CLI accepted its verdict.

    A denied schema tool must reach the loud, un-suppressed
    ``curator_schema_tool_denied`` escalation whether or not the curator is
    gated, and a failure whose stdout arrived was already parsed by
    ``_parse_claude_output``, which owns stdout salvage.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('transcript_lines', 'failure'), _UNSALVAGEABLE_SINGLE)
    async def test_single_failure_is_escalated_as_it_was(
        self, transcript_lines, failure, tmp_path, caplog,
    ):
        curator, escalator = _gated_curator(tmp_path)
        invoke = _scripted_invoker((transcript_lines, failure))

        with caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE):
            decision = await _curate(curator, invoke)

        assert decision.action == 'create'
        report = _reported_failure(escalator)
        assert report['schema_tool_denied'] is failure.schema_tool_denied
        assert report['subtype'] == failure.subtype
        assert report['tools_used'] == ()
        assert _salvage_warnings(caplog) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('transcript_lines', 'failure'), _UNSALVAGEABLE_BATCH)
    async def test_batch_failure_is_bisected(self, transcript_lines, failure, tmp_path, caplog):
        curator, _ = _gated_curator(tmp_path)
        invoke = _scripted_invoker((transcript_lines, failure), _HEALTHY, _HEALTHY)

        with caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE):
            await _curate_batch(curator, invoke)

        assert invoke.await_count == 3
        assert _salvage_warnings(caplog) == []

    @pytest.mark.asyncio
    async def test_killed_run_with_only_a_rejected_verdict_is_escalated(self, tmp_path, caplog):
        """A real CLI 2.1.283 transcript whose only StructuredOutput call failed the schema."""
        curator, escalator = _gated_curator(tmp_path)

        with caplog.at_level(logging.WARNING, logger=_CURATOR_MODULE):
            await _curate(curator, _fixture_invoker('schema_rejected_only.jsonl'))

        report = _reported_failure(escalator)
        assert report['zero_output_timeout'] is False
        assert report['transcript_turns'] == 2
        assert report['tools_used'] == ()
        assert _salvage_warnings(caplog) == []


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
        curator, escalator = _gated_curator(tmp_path)
        invoke = _scripted_invoker((transcript_lines, _schema_tool_denied()))

        decision = await _curate(curator, invoke)

        assert decision.action == 'create'
        assert _reported_failure(escalator)['schema_tool_denied'] is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize('transcript_lines', _DENIAL_TRANSCRIPTS_WITH_A_DROP)
    async def test_denial_does_not_reset_the_breaker(self, transcript_lines, tmp_path):
        """ZOT, denial, ZOT: a denial is neither a ZOT nor a success, so the two
        ZOTs reach the threshold of two and the fourth call short-circuits."""
        curator, _ = _gated_curator(tmp_path, _breaker_config(threshold=2))
        invoke = _scripted_invoker(
            _ZOT, (transcript_lines, _schema_tool_denied()), _ZOT, _HEALTHY,
        )

        decisions = [await _curate(curator, invoke, title) for title in 'ABCD']

        assert decisions[1].action == 'create'
        assert invoke.await_count == 3
        assert decisions[3].justification == 'zero-output-breaker-open'
