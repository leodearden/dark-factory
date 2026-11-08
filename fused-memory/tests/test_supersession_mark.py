"""The supersession mark: its key's standing in the metadata vocabulary, and
``services/supersession_mark.py``'s two entry points at their public seams:
``mark_superseded`` (the tool path) and ``mark_superseded_after_retry`` (the
retry pass, once a deferred write lands)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from fused_memory.memory_metadata import (
    SUPERSEDED_BY_METADATA_KEY,
    classify_unknown_keys,
    validate_memory_metadata,
)
from fused_memory.models.enums import SourceStore
from fused_memory.models.memory import AddMemoryResponse
from fused_memory.services.mem0_add_outcome import Mem0AddOutcome
from fused_memory.services.supersession_mark import (
    SUPERSESSION_ACK_KEY,
    SUPERSESSION_RETRY_SOURCE,
    SupersessionMarks,
    UnmarkedReason,
    mark_superseded,
    mark_superseded_after_retry,
)

PROJECT_ID = 'dark_factory'
NEW_ID = '0f6c2a4e-8b1d-4c3e-9a7f-5d2e1b0c9a8f'
OLD = '11111111-2222-3333-4444-555555555555'
OLD2 = '66666666-7777-4888-9999-aaaaaaaaaaaa'
_NOT_FOUND = {'error_type': 'MemoryNotFound', 'error': 'no such memory'}


class TestTheMarkKeyIsQuietVocabulary:
    def test_the_mark_raises_no_violation_even_under_the_strict_registry(self):
        assert validate_memory_metadata(
            {SUPERSEDED_BY_METADATA_KEY: NEW_ID}, enforce_kind_registry=True
        ) == []

    def test_the_mark_is_not_an_unknown_key(self):
        assert classify_unknown_keys({SUPERSEDED_BY_METADATA_KEY: NEW_ID}) == []


def _service(**update_memory: Any) -> AsyncMock:
    service = AsyncMock()
    service.update_memory.return_value = {'status': 'updated', 'store': 'mem0'}
    for attr, value in update_memory.items():
        setattr(service.update_memory, attr, value)
    return service


def _written(*memory_ids: str) -> AddMemoryResponse:
    return AddMemoryResponse(memory_ids=list(memory_ids), stores_written=[SourceStore.mem0])


async def _mark(
    service: AsyncMock, metadata: Any, written: AddMemoryResponse | None = None,
) -> SupersessionMarks | None:
    return await mark_superseded(
        service,
        metadata=metadata,
        written=written if written is not None else _written(NEW_ID),
        project_id=PROJECT_ID,
        agent_id='claude-interactive',
        session_id=None,
        causation_id='cause-1',
        source='mcp_tool',
    )


def _patched_ids(service: AsyncMock) -> list[str]:
    return [call.kwargs['memory_id'] for call in service.update_memory.await_args_list]


def _fail_for(*failing: str):
    async def update_memory(**kwargs: Any) -> dict[str, Any]:
        if kwargs['memory_id'] in failing:
            return dict(_NOT_FOUND)
        return {'status': 'updated', 'store': 'mem0', 'id': kwargs['memory_id']}

    return update_memory


class TestMarkSuperseded:
    @pytest.mark.asyncio
    async def test_a_scalar_declaration_marks_its_record_with_the_new_id(self) -> None:
        service = _service()

        marks = await _mark(service, {'supersedes': OLD})

        service.update_memory.assert_awaited_once()
        kwargs = service.update_memory.await_args.kwargs
        assert kwargs['memory_id'] == OLD
        assert kwargs['content'] is None
        assert kwargs['metadata_mode'] == 'merge'
        assert kwargs['metadata_patch'] == {SUPERSEDED_BY_METADATA_KEY: NEW_ID}
        assert kwargs['project_id'] == PROJECT_ID
        assert marks is not None
        assert marks.superseded_by == NEW_ID
        assert marks.marked == (OLD,)
        assert marks.unmarked == ()

    @pytest.mark.asyncio
    async def test_each_declared_record_is_patched_once_in_declared_order(self) -> None:
        service = _service()

        marks = await _mark(service, {'supersedes': [OLD, OLD2, OLD]})

        assert _patched_ids(service) == [OLD, OLD2]
        assert marks is not None
        assert marks.marked == (OLD, OLD2)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'metadata',
        [None, {}, {'supersedes': []}, {'source': 'x'}],
        ids=['none', 'empty', 'empty-list', 'other-keys'],
    )
    async def test_nothing_declared_marks_nothing(self, metadata) -> None:
        service = _service()

        assert await _mark(service, metadata) is None
        service.update_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_member_that_is_not_a_full_uuid_is_reported_not_patched(self) -> None:
        service = _service()

        marks = await _mark(service, {'supersedes': ['abc', OLD]})

        assert _patched_ids(service) == [OLD]
        assert marks is not None
        assert marks.marked == (OLD,)
        assert [(u.memory_id, u.reason) for u in marks.unmarked] == [
            ('abc', UnmarkedReason.INVALID_MEMBER),
        ]

    @pytest.mark.asyncio
    async def test_an_unhashable_member_is_reported_not_raised(self) -> None:
        service = _service()

        marks = await _mark(service, {'supersedes': [{'id': OLD}, OLD]})

        assert marks is not None
        assert marks.marked == (OLD,)
        assert [u.reason for u in marks.unmarked] == [UnmarkedReason.INVALID_MEMBER]

    @pytest.mark.asyncio
    async def test_a_refused_patch_is_reported_and_the_rest_still_marked(self) -> None:
        service = _service(side_effect=_fail_for(OLD))

        marks = await _mark(service, {'supersedes': [OLD, OLD2]})

        assert marks is not None
        assert marks.marked == (OLD2,)
        [unmarked] = marks.unmarked
        assert unmarked.memory_id == OLD
        assert unmarked.reason == UnmarkedReason.PATCH_FAILED
        assert 'MemoryNotFound' in unmarked.error

    @pytest.mark.asyncio
    async def test_a_raising_patch_is_reported_not_raised(self) -> None:
        service = _service(side_effect=RuntimeError('qdrant timeout'))

        marks = await _mark(service, {'supersedes': [OLD]})

        assert marks is not None
        assert marks.marked == ()
        [unmarked] = marks.unmarked
        assert unmarked.reason == UnmarkedReason.PATCH_FAILED
        assert 'RuntimeError' in unmarked.error

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'written',
        [_written(), _written(NEW_ID, OLD2)],
        ids=['no-mem0-id', 'two-mem0-ids'],
    )
    async def test_without_exactly_one_new_record_nothing_is_patched(self, written) -> None:
        service = _service()

        marks = await _mark(service, {'supersedes': [OLD, 'abc']}, written)

        service.update_memory.assert_not_awaited()
        assert marks is not None
        assert marks.superseded_by is None
        assert marks.marked == ()
        reasons = {u.memory_id: u.reason for u in marks.unmarked}
        assert reasons == {
            OLD: UnmarkedReason.NO_SUPERSEDING_RECORD,
            'abc': UnmarkedReason.INVALID_MEMBER,
        }


class TestTheAckShape:
    @pytest.mark.asyncio
    async def test_to_ack_is_one_plain_key(self) -> None:
        service = _service(side_effect=_fail_for(OLD))
        marks = await _mark(service, {'supersedes': ['abc', OLD, OLD2]})
        assert marks is not None

        ack = marks.to_ack()

        assert set(ack) == {SUPERSESSION_ACK_KEY}
        body = ack[SUPERSESSION_ACK_KEY]
        assert body['superseded_by'] == NEW_ID
        assert body['marked'] == [OLD2]
        unmarked = {entry['id']: entry for entry in body['unmarked']}
        assert set(unmarked) == {'abc', OLD}
        assert unmarked[OLD] == {
            'id': OLD, 'reason': 'patch_failed', 'error': 'MemoryNotFound: no such memory',
        }
        assert unmarked['abc']['reason'] == 'invalid_supersedes_member'
        assert all(type(entry['reason']) is str for entry in body['unmarked'])
        assert all(type(entry['error']) is str for entry in body['unmarked'])
        assert body['pending_retry'] == []


def _pending_retry() -> AddMemoryResponse:
    return AddMemoryResponse(memory_ids=[], stores_pending_retry=[SourceStore.mem0])


class TestAWritePendingRetryDefersItsMarks:
    """A deferred write has no Mem0 id yet; the retry pass marks its targets
    when it lands (``mark_superseded_after_retry``), so they are neither
    patched now nor reported as a failure."""

    @pytest.mark.asyncio
    async def test_its_targets_are_reported_pending_and_nothing_is_patched(self) -> None:
        service = _service()

        marks = await _mark(service, {'supersedes': [OLD, 'abc', OLD]}, _pending_retry())

        service.update_memory.assert_not_awaited()
        assert marks is not None
        assert marks.superseded_by is None
        assert marks.marked == ()
        assert marks.pending_retry == (OLD,)
        assert [(u.memory_id, u.reason) for u in marks.unmarked] == [
            ('abc', UnmarkedReason.INVALID_MEMBER),
        ]

    @pytest.mark.asyncio
    async def test_the_ack_names_the_pending_targets(self) -> None:
        marks = await _mark(_service(), {'supersedes': [OLD]}, _pending_retry())
        assert marks is not None

        body = marks.to_ack()[SUPERSESSION_ACK_KEY]

        assert body == {
            'superseded_by': None, 'marked': [], 'unmarked': [], 'pending_retry': [OLD],
        }

    @pytest.mark.asyncio
    async def test_a_landed_write_reports_an_empty_pending_list(self) -> None:
        marks = await _mark(_service(), {'supersedes': [OLD]})
        assert marks is not None

        assert marks.pending_retry == ()
        assert marks.to_ack()[SUPERSESSION_ACK_KEY]['pending_retry'] == []


_RETRY_PROJECT = 'reify'
_RETRY_AGENT = 'claude-task-9'
_RETRY_SESSION = 'session-9'
_RETRY_CAUSATION = 'cause-9'


def _intent(metadata: dict[str, Any]) -> dict[str, Any]:
    """A ``mem0_intents`` row as the journal hands it back: metadata is a JSON column."""
    return {
        'id': 'intent-1',
        'write_op_id': 'op-1',
        'project_id': _RETRY_PROJECT,
        'agent_id': _RETRY_AGENT,
        'session_id': _RETRY_SESSION,
        'causation_id': _RETRY_CAUSATION,
        'category': 'procedural_knowledge',
        'content': 'the corrected claim',
        'metadata': json.dumps(metadata),
        'status': 'retrying',
    }


def _landed(*memory_ids: str) -> Mem0AddOutcome:
    return Mem0AddOutcome(memory_ids=memory_ids, empty_result_error=None)


class TestMarkSupersededAfterRetry:
    @pytest.mark.asyncio
    async def test_each_target_is_marked_with_the_landed_id_under_the_intents_attribution(
        self,
    ) -> None:
        service = _service()

        marks = await mark_superseded_after_retry(
            service,
            _intent({'supersedes': [OLD, OLD2], 'topic': 't'}),
            _landed(NEW_ID),
        )

        assert _patched_ids(service) == [OLD, OLD2]
        for call in service.update_memory.await_args_list:
            assert call.kwargs['metadata_patch'] == {SUPERSEDED_BY_METADATA_KEY: NEW_ID}
            assert call.kwargs['project_id'] == _RETRY_PROJECT
            assert call.kwargs['agent_id'] == _RETRY_AGENT
            assert call.kwargs['session_id'] == _RETRY_SESSION
            assert call.kwargs['causation_id'] == _RETRY_CAUSATION
            assert call.kwargs['_source'] == SUPERSESSION_RETRY_SOURCE
        assert marks == SupersessionMarks(
            superseded_by=NEW_ID, marked=(OLD, OLD2), unmarked=(), pending_retry=(),
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'metadata',
        [{}, {'supersedes': []}, {'topic': 't'}],
        ids=['empty', 'empty-list', 'other-keys'],
    )
    async def test_an_intent_declaring_no_supersession_marks_nothing(self, metadata) -> None:
        service = _service()

        assert await mark_superseded_after_retry(service, _intent(metadata), _landed(NEW_ID)) is None
        service.update_memory.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'outcome',
        [_landed(), _landed(NEW_ID, OLD2)],
        ids=['no-mem0-id', 'two-mem0-ids'],
    )
    async def test_without_exactly_one_landed_id_nothing_is_patched(self, outcome) -> None:
        service = _service()

        marks = await mark_superseded_after_retry(
            service, _intent({'supersedes': [OLD, OLD2]}), outcome
        )

        service.update_memory.assert_not_awaited()
        assert marks is not None
        assert marks.superseded_by is None
        assert marks.marked == ()
        assert {u.memory_id: u.reason for u in marks.unmarked} == {
            OLD: UnmarkedReason.NO_SUPERSEDING_RECORD,
            OLD2: UnmarkedReason.NO_SUPERSEDING_RECORD,
        }

    @pytest.mark.asyncio
    async def test_a_refused_patch_is_reported_and_the_rest_still_marked(self) -> None:
        service = _service(side_effect=_fail_for(OLD))

        marks = await mark_superseded_after_retry(
            service, _intent({'supersedes': [OLD, OLD2]}), _landed(NEW_ID)
        )

        assert marks is not None
        assert marks.marked == (OLD2,)
        [unmarked] = marks.unmarked
        assert unmarked.memory_id == OLD
        assert unmarked.reason == UnmarkedReason.PATCH_FAILED

    @pytest.mark.asyncio
    @pytest.mark.parametrize('project_id', [None, ''], ids=['missing', 'empty'])
    async def test_an_intent_naming_no_project_is_reported_not_marked_in_a_default_project(
        self, project_id, caplog
    ) -> None:
        service = _service()
        intent = {**_intent({'supersedes': [OLD, 'abc']}), 'project_id': project_id}

        with caplog.at_level('WARNING', logger='fused_memory.services.supersession_mark'):
            marks = await mark_superseded_after_retry(service, intent, _landed(NEW_ID))

        service.update_memory.assert_not_awaited()
        assert marks is not None
        assert marks.marked == ()
        assert {u.memory_id: u.reason for u in marks.unmarked} == {
            OLD: UnmarkedReason.NO_PROJECT_ID,
            'abc': UnmarkedReason.INVALID_MEMBER,
        }
        assert [r.levelname for r in caplog.records] == ['WARNING']
