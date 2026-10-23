"""A write that declares ``supersedes`` is a correction, never a triage attach.

Triage attaches a restatement under the record it restates. A correction
restates the record it replaces too, so without a force-store the replacement
would be filed as a child of the very record it retires.

Harness: ``tests/server/write_triage_harness.py`` and the
``server._tool_manager.call_tool('add_memory', ...)`` idiom of
``tests/server/test_add_memory_write_triage_gate.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fused_memory.memory_metadata import SUPERSEDED_BY_METADATA_KEY
from fused_memory.server import write_triage
from fused_memory.server.grouped_read import PARENT_ID_KEY
from fused_memory.server.tools import create_mcp_server
from fused_memory.server.write_triage import (
    CANONICAL_ID_KEY,
    OUTCOME_AMENDED,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    ROUTED_KEY,
    TriageFailOpenCounter,
    TriageJudgeVerdict,
    triage_write,
)
from fused_memory.services.supersession_mark import SUPERSESSION_ACK_KEY
from server.write_triage_harness import configure_triage_config, search_candidate

_PROJECT_ID = 'dark_factory'
_AGENT_ID = 'claude-interactive'
OLD = '11111111-2222-3333-4444-555555555555'
OLD2 = '66666666-7777-4888-9999-aaaaaaaaaaaa'
NEW_ID = '0f6c2a4e-8b1d-4c3e-9a7f-5d2e1b0c9a8f'

_CONTENT = (
    'Never run git stash in a dark-factory checkout: refs/stash is a single '
    'ref in the shared .git dir and is not per-worktree.'
)

#: Patched at the consumer, as in test_add_memory_write_triage_gate.py.
_JUDGE_PATH = 'fused_memory.server.tools.judge_write'
_HIGH_BAND = 0.99
_MIDDLE_BAND = 0.80
#: With triage off the near-duplicate reject guard is live and refuses a
#: restatement before anything is written, so the triage-off write stays under it.
_BELOW_REJECT_GUARD = 0.50

_DECLARATION_SHAPES = {
    'scalar': OLD,
    'list': [OLD],
}


def _service(score: float, *, enabled: bool = True) -> AsyncMock:
    """A service whose one search hit is the record the write supersedes."""
    mock_service = AsyncMock()
    configure_triage_config(mock_service, enabled=enabled)
    written = MagicMock()
    written.memory_ids = [NEW_ID]
    written.model_dump.return_value = {
        'memory_ids': [NEW_ID],
        'category': 'procedural_knowledge',
        'stores_written': ['mem0'],
    }
    mock_service.add_memory.return_value = written
    mock_service.update_memory.return_value = {'status': 'updated', 'store': 'mem0', 'id': OLD}
    mock_service.search.return_value = [search_candidate(OLD, score)]
    return mock_service


async def _call(mock_service: AsyncMock, metadata: dict | None) -> dict:
    server = create_mcp_server(mock_service)
    args = {
        'content': _CONTENT,
        'category': 'procedural_knowledge',
        'agent_id': _AGENT_ID,
        'project_id': _PROJECT_ID,
    }
    if metadata is not None:
        args['metadata'] = metadata
    return await server._tool_manager.call_tool('add_memory', args)


def _persisted_metadata(mock_service: AsyncMock) -> dict:
    return mock_service.add_memory.await_args.kwargs['metadata'] or {}


class TestDeclaresSupersession:
    @pytest.mark.parametrize(
        'metadata',
        [
            {'supersedes': OLD},
            {'supersedes': [OLD]},
            {'supersedes': (OLD,)},
            {'supersedes': [OLD, OLD2]},
            {'supersedes': ['abc']},
            {'source': 'notes', 'supersedes': [OLD]},
        ],
        ids=['scalar', 'list', 'tuple', 'two', 'malformed-member', 'beside-other-keys'],
    )
    def test_a_non_empty_declaration_counts(self, metadata) -> None:
        assert write_triage.declares_supersession(metadata) is True

    @pytest.mark.parametrize(
        'metadata',
        [None, {}, {'supersedes': None}, {'supersedes': []}, {'source': 'notes'}, 'supersedes'],
        ids=['none', 'empty', 'null', 'empty-list', 'other-keys', 'not-a-dict'],
    )
    def test_nothing_declared_does_not_count(self, metadata) -> None:
        assert write_triage.declares_supersession(metadata) is False


class TestTriageWriteForceStoresASupersession:
    @pytest.mark.asyncio
    async def test_a_declared_supersession_stores_before_retrieval(self) -> None:
        mock_service = AsyncMock()
        configure_triage_config(mock_service)
        mock_service.search.return_value = [search_candidate(OLD, _HIGH_BAND)]
        counter = TriageFailOpenCounter(time_provider=lambda: 1000.0)

        decision = await triage_write(
            mock_service, content='c', project_id='p', counter=counter,
            caller_declares_supersession=True,
        )

        assert decision.outcome == OUTCOME_STORED
        assert decision.canonical_id is None
        mock_service.search.assert_not_awaited()
        assert counter.live_count() == 0, 'a force-store is not a fail-open'


class TestAddMemoryStoresASupersessionStandalone:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('declared', _DECLARATION_SHAPES.values(), ids=_DECLARATION_SHAPES)
    async def test_a_high_band_hit_on_the_superseded_record_is_not_attached(
        self, declared,
    ) -> None:
        mock_service = _service(_HIGH_BAND)

        ack = await _call(mock_service, {'supersedes': declared})

        assert ack[ROUTED_KEY] == OUTCOME_STORED, f'{ack!r}'
        assert CANONICAL_ID_KEY not in ack, f'{ack!r}'
        persisted = _persisted_metadata(mock_service)
        assert PARENT_ID_KEY not in persisted, f'{persisted!r}'
        assert 'kind' not in persisted, f'{persisted!r}'
        assert persisted['supersedes'] == declared, f'{persisted!r}'
        mock_service.search.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('declared', _DECLARATION_SHAPES.values(), ids=_DECLARATION_SHAPES)
    async def test_a_middle_band_write_never_asks_the_judge(self, declared) -> None:
        mock_service = _service(_MIDDLE_BAND)
        judge = AsyncMock(return_value=TriageJudgeVerdict(OUTCOME_AMENDED, OLD))

        with patch(_JUDGE_PATH, new=judge):
            ack = await _call(mock_service, {'supersedes': declared})

        judge.assert_not_awaited()
        assert ack[ROUTED_KEY] == OUTCOME_STORED, f'{ack!r}'
        assert PARENT_ID_KEY not in _persisted_metadata(mock_service)

    @pytest.mark.asyncio
    async def test_the_same_write_without_supersedes_is_routed(self) -> None:
        """The control: the force-store above is supersedes' doing, not the harness's."""
        mock_service = _service(_HIGH_BAND)

        ack = await _call(mock_service, None)

        assert ack[ROUTED_KEY] == OUTCOME_RESTATED, f'{ack!r}'
        assert ack[CANONICAL_ID_KEY] == OLD, f'{ack!r}'


class TestAddMemoryMarksTheSupersededRecord:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('triage_enabled', 'score'),
        [(True, _HIGH_BAND), (False, _BELOW_REJECT_GUARD)],
        ids=['triage-on', 'triage-off'],
    )
    async def test_the_superseded_record_is_marked_with_the_new_id(
        self, triage_enabled: bool, score: float,
    ) -> None:
        mock_service = _service(score, enabled=triage_enabled)

        ack = await _call(mock_service, {'supersedes': [OLD]})

        mock_service.update_memory.assert_awaited_once()
        kwargs = mock_service.update_memory.await_args.kwargs
        assert kwargs['memory_id'] == OLD
        assert kwargs['content'] is None
        assert kwargs['metadata_patch'] == {SUPERSEDED_BY_METADATA_KEY: NEW_ID}
        assert kwargs['project_id'] == _PROJECT_ID
        assert kwargs['agent_id'] == _AGENT_ID
        assert ack[SUPERSESSION_ACK_KEY] == {
            'superseded_by': NEW_ID, 'marked': [OLD], 'unmarked': [],
        }, f'{ack!r}'
        if triage_enabled:
            assert ack[ROUTED_KEY] == OUTCOME_STORED, f'{ack!r}'

    @pytest.mark.asyncio
    async def test_a_write_without_supersedes_marks_nothing(self) -> None:
        mock_service = _service(_HIGH_BAND)

        ack = await _call(mock_service, None)

        mock_service.update_memory.assert_not_awaited()
        assert SUPERSESSION_ACK_KEY not in ack, f'{ack!r}'

    @pytest.mark.asyncio
    async def test_a_refused_mark_is_disclosed_and_the_write_still_acks(self) -> None:
        mock_service = _service(_HIGH_BAND)
        mock_service.update_memory.return_value = {
            'error_type': 'MemoryNotFound', 'error': f'no memory {OLD}',
        }

        ack = await _call(mock_service, {'supersedes': [OLD]})

        assert 'error' not in ack, f'{ack!r}'
        assert ack['memory_ids'] == [NEW_ID], f'{ack!r}'
        body = ack[SUPERSESSION_ACK_KEY]
        assert body['marked'] == []
        [unmarked] = body['unmarked']
        assert unmarked['id'] == OLD
        assert unmarked['reason'] == 'patch_failed'
