"""The supersession mark: its key's standing in the metadata vocabulary, and
``services/supersession_mark.py::mark_superseded`` at its public seam."""

from __future__ import annotations

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
from fused_memory.services.supersession_mark import (
    SUPERSESSION_ACK_KEY,
    SupersessionMarks,
    UnmarkedReason,
    mark_superseded,
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
