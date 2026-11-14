"""Orchestrator completion records are never write-triage children (task 6956).

Each task's completion record restates the same template as every other
task's, so triage would file task B's record as a child of task A's. Two routes
keep it standalone: the orchestrator declares a ``kind``, which
``write_triage.declares_attach_keys`` already force-stores, and a server-side
backstop force-stores a legacy write that carries only the completion source.
The kinded metadata is the orchestrator's own
(orchestrator/src/orchestrator/completion_record.py::completion_record_metadata),
so these tests check the value the writer actually emits.

Harness: ``tests/server/write_triage_harness.py`` and the
``server._tool_manager.call_tool('add_memory', ...)`` idiom of
``tests/server/test_write_triage_supersession.py``.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from orchestrator.completion_record import completion_record_metadata

from fused_memory import memory_metadata
from fused_memory.memory_metadata import CHILD_KINDS, ORCHESTRATOR_COMPLETION_SOURCE
from fused_memory.models.enums import MemoryCategory
from fused_memory.models.memory import MemoryResult
from fused_memory.server import write_triage
from fused_memory.server.grouped_read import PARENT_ID_KEY, is_contested_child
from fused_memory.server.tools import create_mcp_server
from fused_memory.server.write_triage import (
    CANONICAL_ID_KEY,
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    ROUTED_KEY,
    BandDecision,
    TriageFailOpenCounter,
    TriageJudgeVerdict,
    triage_write,
)
from server.write_triage_harness import configure_triage_config, search_candidate

_PROJECT_ID = 'dark_factory'
#: Patched at the consumer, as in test_write_triage_supersession.py.
_JUDGE_PATH = 'fused_memory.server.tools.judge_write'
_HIGH_BAND = 0.99
_JUDGE_BAND = 0.80
_BANDS = {'restate-band': _HIGH_BAND, 'judge-band': _JUDGE_BAND}

_TASK_A_COMPLETION = 'completion-4101'
_COMPLETION_CONTENT = (
    'Completed: Widen the done-task audit\n'
    'Steps completed: 3/3\n'
    'Modules: orchestrator/src/orchestrator'
)
_TASK_B_COMPLETION_METADATA = {
    'kinded': completion_record_metadata('4102'),
    'legacy-unkinded': {'task_id': '4102', 'source': ORCHESTRATOR_COMPLETION_SOURCE},
}

_TASK_X_GOTCHA = 'gotcha-4201'
_GOTCHA_CONTENT = (
    'Never run git stash in a dark-factory checkout: refs/stash is a single '
    'ref in the shared .git dir and is not per-worktree.'
)


class TestIsOrchestratorCompletionWrite:
    @pytest.mark.parametrize(
        'metadata', _TASK_B_COMPLETION_METADATA.values(), ids=_TASK_B_COMPLETION_METADATA,
    )
    def test_a_completion_write_counts(self, metadata) -> None:
        assert write_triage.is_orchestrator_completion_write(metadata) is True

    @pytest.mark.parametrize(
        'metadata',
        [None, {}, ORCHESTRATOR_COMPLETION_SOURCE, {'task_id': '4102'},
         {'task_id': '4102', 'source': 'agent_note'}],
        ids=['none', 'empty', 'not-a-dict', 'task-id-only', 'other-source'],
    )
    def test_anything_else_does_not_count(self, metadata) -> None:
        assert write_triage.is_orchestrator_completion_write(metadata) is False


async def _triage_a_completion_record(*, declares_kind: bool) -> tuple[AsyncMock, BandDecision]:
    mock_service = AsyncMock()
    configure_triage_config(mock_service)
    mock_service.search.return_value = [search_candidate('completion-4101', _HIGH_BAND)]
    counter = TriageFailOpenCounter(time_provider=lambda: 1000.0)
    decision = await triage_write(
        mock_service, content='c', project_id='p', counter=counter,
        caller_owns_attach_keys=declares_kind, caller_is_completion_record=True,
    )
    assert counter.live_count() == 0, 'a force-store is not a fail-open'
    return mock_service, decision


def _legacy_completion_lines(caplog) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if 'legacy completion record' in record.getMessage()
    ]


class TestTriageWriteForceStoresACompletionRecord:
    @pytest.mark.asyncio
    async def test_a_completion_record_stores_before_retrieval(self) -> None:
        mock_service, decision = await _triage_a_completion_record(declares_kind=False)

        assert decision.outcome == OUTCOME_STORED
        assert decision.canonical_id is None
        mock_service.search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_each_legacy_force_store_logs_one_line(self, caplog) -> None:
        """The line that measures when the source backstop can be retired."""
        with caplog.at_level(logging.INFO, logger=write_triage.logger.name):
            await _triage_a_completion_record(declares_kind=False)

        lines = _legacy_completion_lines(caplog)
        assert len(lines) == 1, f'{caplog.text!r}'
        assert 'project=p' in lines[0], f'{lines!r}'

    @pytest.mark.asyncio
    async def test_a_kinded_completion_record_is_not_counted_as_legacy(self, caplog) -> None:
        with caplog.at_level(logging.INFO, logger=write_triage.logger.name):
            _, decision = await _triage_a_completion_record(declares_kind=True)

        assert decision.outcome == OUTCOME_STORED
        assert _legacy_completion_lines(caplog) == [], f'{caplog.text!r}'


def _service(hit: MemoryResult) -> AsyncMock:
    """A triage-on service whose one search hit is *hit*."""
    mock_service = AsyncMock()
    configure_triage_config(mock_service, enabled=True)
    written = MagicMock()
    written.model_dump.return_value = {
        'memory_ids': ['new-id'],
        'category': 'observations_and_summaries',
        'stores_written': ['mem0'],
    }
    mock_service.add_memory.return_value = written
    mock_service.search.return_value = [hit]
    return mock_service


def _task_a_completion_record(score: float) -> MemoryResult:
    return search_candidate(
        _TASK_A_COMPLETION,
        score,
        content=_COMPLETION_CONTENT,
        category=MemoryCategory.observations_and_summaries,
        extra_metadata={'task_id': '4101', 'source': ORCHESTRATOR_COMPLETION_SOURCE},
    )


def _task_x_gotcha(score: float) -> MemoryResult:
    return search_candidate(
        _TASK_X_GOTCHA, score, content=_GOTCHA_CONTENT, extra_metadata={'task_id': '4201'},
    )


async def _call(
    mock_service: AsyncMock, *, content: str, category: str, agent_id: str, metadata: dict,
) -> dict:
    server = create_mcp_server(mock_service)
    return await server._tool_manager.call_tool('add_memory', {
        'content': content,
        'category': category,
        'agent_id': agent_id,
        'project_id': _PROJECT_ID,
        'metadata': metadata,
    })


async def _write_task_b_completion(mock_service: AsyncMock, metadata: dict) -> dict:
    return await _call(
        mock_service,
        content=_COMPLETION_CONTENT,
        category='observations_and_summaries',
        agent_id='orchestrator-task-4102',
        metadata=metadata,
    )


async def _write_task_y_gotcha(mock_service: AsyncMock) -> dict:
    return await _call(
        mock_service,
        content=_GOTCHA_CONTENT,
        category='procedural_knowledge',
        agent_id='claude-task-4202',
        metadata={'task_id': '4202'},
    )


def _persisted_metadata(mock_service: AsyncMock) -> dict:
    return mock_service.add_memory.await_args.kwargs['metadata'] or {}


class TestACompletionRecordForAnotherTaskLandsStandalone:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('score', _BANDS.values(), ids=_BANDS)
    @pytest.mark.parametrize(
        'metadata', _TASK_B_COMPLETION_METADATA.values(), ids=_TASK_B_COMPLETION_METADATA,
    )
    async def test_task_b_is_not_attached_under_task_a(
        self, score: float, metadata: dict,
    ) -> None:
        mock_service = _service(_task_a_completion_record(score))
        judge = AsyncMock(return_value=TriageJudgeVerdict(OUTCOME_CONTESTED, _TASK_A_COMPLETION))

        with patch(_JUDGE_PATH, new=judge):
            ack = await _write_task_b_completion(mock_service, metadata)

        assert ack[ROUTED_KEY] == OUTCOME_STORED, f'{ack!r}'
        assert CANONICAL_ID_KEY not in ack, f'{ack!r}'
        judge.assert_not_awaited()
        persisted = _persisted_metadata(mock_service)
        assert PARENT_ID_KEY not in persisted, f'{persisted!r}'
        assert is_contested_child(persisted) is False, f'{persisted!r}'
        assert persisted['task_id'] == '4102', f'{persisted!r}'
        assert persisted.get('kind') == metadata.get('kind'), (
            'the backstop routes; it does not stamp a kind'
        )

    @pytest.mark.asyncio
    async def test_the_same_write_under_another_source_is_attached(self) -> None:
        """The control: the force-store above is the completion source's doing."""
        mock_service = _service(_task_a_completion_record(_HIGH_BAND))

        ack = await _write_task_b_completion(
            mock_service, {'task_id': '4102', 'source': 'agent_note'},
        )

        assert ack[ROUTED_KEY] == OUTCOME_RESTATED, f'{ack!r}'
        assert ack[CANONICAL_ID_KEY] == _TASK_A_COMPLETION, f'{ack!r}'


class TestNoGenericTaskIdFilter:
    """An agent memory's task_id is provenance, not subject: it never blocks an attach."""

    @pytest.mark.asyncio
    async def test_a_restatement_from_another_task_still_attaches(self) -> None:
        mock_service = _service(_task_x_gotcha(_HIGH_BAND))

        ack = await _write_task_y_gotcha(mock_service)

        assert ack[ROUTED_KEY] == OUTCOME_RESTATED, f'{ack!r}'
        assert ack[CANONICAL_ID_KEY] == _TASK_X_GOTCHA, f'{ack!r}'
        assert _persisted_metadata(mock_service)[PARENT_ID_KEY] == _TASK_X_GOTCHA

    @pytest.mark.asyncio
    async def test_a_judged_amendment_from_another_task_still_attaches(self) -> None:
        mock_service = _service(_task_x_gotcha(_JUDGE_BAND))
        judge = AsyncMock(return_value=TriageJudgeVerdict(OUTCOME_AMENDED, _TASK_X_GOTCHA))

        with patch(_JUDGE_PATH, new=judge):
            ack = await _write_task_y_gotcha(mock_service)

        judge.assert_awaited_once()
        assert ack[ROUTED_KEY] == OUTCOME_AMENDED, f'{ack!r}'
        assert ack[CANONICAL_ID_KEY] == _TASK_X_GOTCHA, f'{ack!r}'


class TestTheOrchestratorsCompletionMetadata:
    """The writer's half of the contract matches fused-memory's half."""

    def test_it_validates_under_kind_enforcement(self) -> None:
        """If the kind registry is ever enforced, completion writes must not start failing."""
        metadata = completion_record_metadata('4102')

        assert memory_metadata.validate_memory_metadata(metadata, enforce_kind_registry=True) == []
        assert metadata['kind'] not in CHILD_KINDS

    def test_it_carries_the_source_fused_memory_owns(self) -> None:
        metadata = completion_record_metadata('4102')

        assert memory_metadata.is_orchestrator_completion_record(metadata) is True
        assert write_triage.declares_attach_keys(metadata) is True
