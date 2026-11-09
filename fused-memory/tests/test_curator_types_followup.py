"""The curator vocabulary's follow-up surface: the declared-scope band, the item count and the holder.

plans/follow-up-filing-shape-prd.md §4.2-§4.3, decisions 8, 10 and 13: the curator reads the stamp
and the item count of every candidate and pool entry, renders an absent stamp as no information, and
never offers a holder as a combine target.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from shared.declared_scope import DECLARED_SCOPE_METADATA_KEY, derive_declared_scope
from shared.followup_items import FOLLOWUP_ITEMS_METADATA_KEY, observed_item
from shared.task_statuses import TaskStatus

from fused_memory.middleware.curator_types import (
    UNSTAMPED_BAND,
    CandidateTask,
    PoolEntry,
    declared_scope_of,
    followup_items_of,
    is_combine_eligible,
    is_combine_eligible_status,
    item_count_of,
    render_declared_scope,
)
from fused_memory.middleware.followup_hold import FOLLOWUP_HOLDER_METADATA_KEY

_NARROW = derive_declared_scope(['pkg/a.py', 'pkg/b.py'], 'd' * 50, 'e' * 7, basis='submit')
_TWO_ITEMS = [
    observed_item(summary='first', files=['pkg/a.py'], origin_task='1').model_dump(mode='json'),
    observed_item(summary='second', files=['pkg/b.py'], origin_task='1').model_dump(mode='json'),
]


_CURATOR_TYPES_LOGGER = 'fused_memory.middleware.curator_types'


def _census_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The ``task_metadata.schema_warning`` census lines curator_types emitted."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == _CURATOR_TYPES_LOGGER and record.levelno == logging.WARNING
        and record.getMessage().startswith('task_metadata.schema_warning ')
    ]


@pytest.fixture
def census(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.WARNING, logger=_CURATOR_TYPES_LOGGER)
    return caplog


class TestDeclaredScopeOf:
    def test_a_stored_stamp_is_its_record(self) -> None:
        stamp = derive_declared_scope(['pkg/a.py'], 'd', '', basis='submit')
        assert declared_scope_of({DECLARED_SCOPE_METADATA_KEY: stamp.model_dump()}) == stamp

    def test_an_absent_stamp_reads_as_none_without_a_census_line(
        self, census: pytest.LogCaptureFixture,
    ) -> None:
        assert declared_scope_of({}) is None
        assert _census_lines(census) == []

    @pytest.mark.parametrize(
        'metadata',
        [
            pytest.param({DECLARED_SCOPE_METADATA_KEY: 'narrow'}, id='not-a-mapping'),
            pytest.param(
                {DECLARED_SCOPE_METADATA_KEY: {**_NARROW.model_dump(), 'band': 'broad'}},
                id='band-disagrees-with-counts',
            ),
        ],
    )
    def test_an_invalid_stamp_reads_as_none_with_a_census_line(
        self, metadata: dict[str, Any], census: pytest.LogCaptureFixture,
    ) -> None:
        assert declared_scope_of(metadata) is None
        [line] = _census_lines(census)
        assert DECLARED_SCOPE_METADATA_KEY in line


class TestFollowupItemsOf:
    def test_a_valid_list_is_its_items_and_its_count(self) -> None:
        metadata = {FOLLOWUP_ITEMS_METADATA_KEY: _TWO_ITEMS}
        assert [item.model_dump(mode='json') for item in followup_items_of(metadata)] == _TWO_ITEMS
        assert item_count_of(metadata) == 2

    @pytest.mark.parametrize(
        'metadata',
        [
            pytest.param({}, id='absent'),
            pytest.param(
                {'suggestion_hash': 'abc123', 'files': ['pkg/a.py'], 'spawned_from': '1'},
                id='legacy-single',
            ),
        ],
    )
    def test_no_list_reads_as_no_items_without_a_census_line(
        self, metadata: dict[str, Any], census: pytest.LogCaptureFixture,
    ) -> None:
        assert followup_items_of(metadata) == ()
        assert item_count_of(metadata) == 0
        assert _census_lines(census) == []

    @pytest.mark.parametrize(
        'metadata',
        [
            pytest.param({FOLLOWUP_ITEMS_METADATA_KEY: 'not-a-list'}, id='not-a-list'),
            pytest.param({FOLLOWUP_ITEMS_METADATA_KEY: [{'summary': 'no key'}]}, id='invalid-element'),
        ],
    )
    def test_a_malformed_list_reads_as_no_items_with_a_census_line(
        self, metadata: dict[str, Any], census: pytest.LogCaptureFixture,
    ) -> None:
        assert followup_items_of(metadata) == ()
        [line] = _census_lines(census)
        assert FOLLOWUP_ITEMS_METADATA_KEY in line


class TestRenderDeclaredScope:
    def test_a_stamp_renders_its_band_counts_and_basis(self) -> None:
        assert render_declared_scope(_NARROW) == 'narrow (files=2, text_chars=57, basis=submit)'

    def test_an_absent_stamp_renders_the_unstamped_word(self) -> None:
        assert render_declared_scope(None) == UNSTAMPED_BAND

    def test_an_undeclared_stamp_is_distinct_from_an_absent_one(self) -> None:
        undeclared = derive_declared_scope([], 'd', '', basis='combine')
        assert render_declared_scope(undeclared).startswith('undeclared (files=0, ')
        assert render_declared_scope(undeclared) != render_declared_scope(None)


def _entry(**overrides: Any) -> PoolEntry:
    fields: dict[str, Any] = {
        'task_id': '7', 'title': 't', 'description': 'd', 'details': 'x',
        'files_to_modify': ['pkg/a.py'], 'module_keys': [], 'status': 'pending',
        'priority': 'medium', 'source': 'module', 'combine_eligible': True,
    }
    return PoolEntry(**{**fields, **overrides})


class TestPoolEntryRender:
    def test_the_header_ends_with_the_item_count_and_a_band_line_follows_the_files(self) -> None:
        lines = _entry(item_count=2, declared_scope=_NARROW).render(400, 400).splitlines()
        assert lines[0].endswith(' followup_items=2')
        assert lines[4] == '  files_to_modify: pkg/a.py'
        assert lines[5] == f'  declared_scope: {render_declared_scope(_NARROW)}'

    def test_a_defaulted_entry_renders_zero_items_and_unstamped(self) -> None:
        lines = _entry().render(400, 400).splitlines()
        assert lines[0].endswith(' followup_items=0')
        assert lines[5] == '  declared_scope: unstamped'


_HOLDER_META = {FOLLOWUP_HOLDER_METADATA_KEY: {'project_id': 'p', 'minted_at': '2026-11-08'}}


class TestIsCombineEligible:
    def test_a_pending_non_holder_is_eligible(self) -> None:
        assert is_combine_eligible({'status': 'pending'})

    @pytest.mark.parametrize(
        'metadata', [_HOLDER_META, json.dumps(_HOLDER_META)], ids=['dict', 'json-string'],
    )
    def test_a_pending_holder_is_not_eligible(self, metadata: object) -> None:
        assert not is_combine_eligible({'status': 'pending', 'metadata': metadata})

    def test_a_done_task_is_not_eligible(self) -> None:
        assert not is_combine_eligible({'status': 'done'})

    @pytest.mark.parametrize('status', list(TaskStatus), ids=lambda status: status.value)
    def test_a_non_holder_agrees_with_the_status_axis(self, status: TaskStatus) -> None:
        task = {'status': status.value, 'metadata': {'files': ['pkg/a.py']}}
        assert is_combine_eligible(task) is is_combine_eligible_status(status.value)


def test_a_candidate_carries_its_stamp_outside_the_payload_hash() -> None:
    base = CandidateTask(title='t', description='d', files_to_modify=['pkg/a.py'])
    stamped = CandidateTask(
        title='t', description='d', files_to_modify=['pkg/a.py'], declared_scope=_NARROW,
    )
    assert stamped.declared_scope == _NARROW
    assert stamped.payload_hash() == base.payload_hash()
