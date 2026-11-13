"""One open trigger chain per (project, skill), at the submit boundary.

PRD plans/completion-driven-triggers-prd.md C2. ``trigger_chain_open_error``
reads every non-terminal ``trigger_chain`` carrier through a metadata lookup
source and refuses a chain task while another run of the same skill has an
open, unsuperseded chain. It fails open on a read error.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from _fake_metadata_lookup_source import FakeMetadataLookupSource
from shared.task_statuses import ACTIVE

from fused_memory.backends.task_backend_types import MetadataLookupRow
from fused_memory.middleware.trigger_chain_guard import trigger_chain_open_error

pytestmark = pytest.mark.asyncio

_LOGGER = 'fused_memory.middleware.trigger_chain_guard'


def _chain(role: str, run_id: str, *, skill: str = 'review', **fields: str) -> dict:
    return {'role': role, 'skill': skill, 'run_id': run_id, **fields}


def _row(task_id: int, status: str, matched_value: object) -> MetadataLookupRow:
    return {
        'task_id': task_id,
        'title': f'chain task {task_id}',
        'status': status,
        'priority': 'medium',
        'updated_at': '2026-11-02T00:00:00+00:00',
        'matched_value': matched_value,
    }


def _open_review_p1(**fields: str) -> list[MetadataLookupRow]:
    return [
        _row(11, 'pending', _chain('completion_gate', 'review-p-1', **fields)),
        _row(12, 'deferred', _chain('human_gate', 'review-p-1', **fields)),
    ]


def _submission(role: str = 'completion_gate', run_id: str = 'review-p-2', **chain: str) -> dict:
    return {'trigger_chain': _chain(role, run_id, **chain)}


async def _check(
    metadata: str | dict[str, Any] | None, source: FakeMetadataLookupSource, tag: str | None = None
) -> dict[str, Any] | None:
    return await trigger_chain_open_error(metadata, '/p', source=source, tag=tag)


async def test_an_open_chain_of_the_same_skill_refuses() -> None:
    result = await _check(_submission(), FakeMetadataLookupSource(_open_review_p1()))
    assert result is not None
    assert result['error_type'] == 'TriggerChainOpen'
    assert result['open_chain'] == [
        {'id': '11', 'role': 'completion_gate', 'status': 'pending', 'run_id': 'review-p-1'},
        {'id': '12', 'role': 'human_gate', 'status': 'deferred', 'run_id': 'review-p-1'},
    ]
    assert '11' in result['error']
    assert '12' in result['error']
    assert 'C5' in result['hint']
    assert 'superseded_by' in result['hint']


async def test_a_superseded_chain_does_not_refuse() -> None:
    source = FakeMetadataLookupSource(_open_review_p1(superseded_by='review-p-2'))
    assert await _check(_submission(), source) is None


async def test_the_runs_own_chain_does_not_refuse() -> None:
    source = FakeMetadataLookupSource(_open_review_p1()[:1])
    assert await _check(_submission('human_gate', 'review-p-1'), source) is None


async def test_another_skills_chain_does_not_refuse() -> None:
    rows = [_row(21, 'pending', _chain('completion_gate', 'hotspot-survey-p-1', skill='hotspot-survey'))]
    assert await _check(_submission(), FakeMetadataLookupSource(rows)) is None


@pytest.mark.parametrize('status', ['done', 'cancelled'])
async def test_a_terminal_row_the_source_returns_does_not_refuse(status: str) -> None:
    rows = [_row(13, status, _chain('completion_gate', 'review-p-1'))]
    assert await _check(_submission(), FakeMetadataLookupSource(rows)) is None


async def test_an_open_row_on_a_later_page_still_refuses() -> None:
    rows = [
        _row(11, 'pending', _chain('completion_gate', 'review-p-1', superseded_by='review-p-2')),
        _row(12, 'deferred', _chain('human_gate', 'review-p-1')),
    ]
    source = FakeMetadataLookupSource(rows, page_cap=1)
    result = await _check(_submission(), source)
    assert result is not None
    assert [entry['id'] for entry in result['open_chain']] == ['12']
    assert [call['offset'] for call in source.calls] == [0, 1]


async def test_a_submission_without_a_chain_never_reads() -> None:
    source = FakeMetadataLookupSource(_open_review_p1())
    assert await _check({'x_note': 'unrelated'}, source) is None
    assert await _check(None, source) is None
    assert source.calls == []


async def test_the_guard_owns_its_query() -> None:
    source = FakeMetadataLookupSource([])
    assert await _check(_submission(), source, tag='feature') is None
    assert len(source.calls) == 1
    call = source.calls[0]
    assert call['project_root'] == '/p'
    assert call['key'] == 'trigger_chain'
    assert call['value'] is None
    assert call['statuses'] == sorted(ACTIVE)
    assert call['include_value'] is True
    assert call['tag'] == 'feature'


async def test_a_read_error_fails_open_with_a_warning(caplog) -> None:
    source = FakeMetadataLookupSource([], error=RuntimeError('store unavailable'))
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert await _check(_submission(), source) is None
    assert 'fail-open' in caplog.text


async def test_a_page_the_guard_cannot_collate_raises_instead_of_failing_open() -> None:
    class _FieldRenamed(FakeMetadataLookupSource):
        async def find_tasks_by_metadata(self, *args: Any, **kwargs: Any) -> Any:
            page = await super().find_tasks_by_metadata(*args, **kwargs)
            kept = {key: value for key, value in page.items() if key != 'malformed_metadata_ids'}
            return {**kept, 'malformed_ids': page['malformed_metadata_ids']}

    with pytest.raises(KeyError, match='malformed_metadata_ids'):
        await _check(_submission(), _FieldRenamed(_open_review_p1()))


@pytest.mark.parametrize('bad', ['review', {'role': 'human_gate', 'run_id': 'review-p-1'}])
async def test_an_unreadable_stored_chain_is_skipped_with_a_warning(caplog, bad: object) -> None:
    rows = [_row(31, 'pending', bad), *_open_review_p1()]
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        result = await _check(_submission(), FakeMetadataLookupSource(rows))
    assert result is not None
    assert [entry['id'] for entry in result['open_chain']] == ['11', '12']
    assert '31' in caplog.text


async def test_malformed_metadata_ids_are_warned(caplog) -> None:
    source = FakeMetadataLookupSource([], malformed_ids=[41, 42])
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert await _check(_submission(), source) is None
    assert '41' in caplog.text
    assert '42' in caplog.text


async def test_an_incomplete_read_is_warned(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        assert await _check(_submission(), FakeMetadataLookupSource([], total=3)) is None
    assert 'incomplete' in caplog.text


async def test_json_string_metadata_is_judged_as_the_dict() -> None:
    source = FakeMetadataLookupSource(_open_review_p1())
    as_dict = await _check(_submission(), source)
    as_json = await _check(json.dumps(_submission()), source)
    assert as_dict is not None
    assert as_json == as_dict


async def test_a_malformed_submission_is_left_to_the_deterministic_guard() -> None:
    source = FakeMetadataLookupSource(_open_review_p1())
    assert await _check(_submission(skill='/review'), source) is None
    assert source.calls == []
