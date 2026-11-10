"""Task 8938: a curator candidate carries one canonical spawned_from, whatever shape the store holds."""

from __future__ import annotations

import logging

import pytest

from fused_memory.middleware.curator_types import CandidateTask, spawned_from_id, spawned_from_of

_CURATOR_TYPES_LOGGER = 'fused_memory.middleware.curator_types'


def _census_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == _CURATOR_TYPES_LOGGER
        and record.getMessage().startswith('task_metadata.schema_warning ')
    ]


@pytest.mark.parametrize(('stored', 'canonical'), [
    (6592, '6592'),
    ('6592', '6592'),
    (' 6592 ', '6592'),
    ('task-x', 'task-x'),
    ('', None),
    ('   ', None),
    (None, None),
    (True, None),
    ({'id': 6592}, None),
])
def test_spawned_from_id_maps_each_stored_shape_to_its_canonical_text(stored, canonical):
    assert spawned_from_id(stored) == canonical


@pytest.mark.parametrize('value', [6592, ' 42 ', '', {'id': 42}, True])
def test_a_candidate_refuses_a_spawned_from_that_is_not_canonical(value):
    with pytest.raises(ValueError) as excinfo:
        CandidateTask(title='T', spawned_from=value)

    assert 'spawned_from' in str(excinfo.value)
    assert repr(value) in str(excinfo.value)


@pytest.mark.parametrize('value', ['42', 'task-x', None])
def test_a_candidate_accepts_a_canonical_spawned_from(value):
    assert CandidateTask(title='T', spawned_from=value).spawned_from == value


@pytest.mark.parametrize(('metadata', 'canonical'), [
    ({'spawned_from': 6592}, '6592'),
    ({'spawned_from': ' 6592 '}, '6592'),
    ({'spawned_from': None}, None),
    ({}, None),
])
def test_a_stored_spawned_from_naming_a_task_or_absent_reads_without_a_census_line(
    metadata, canonical, caplog,
):
    caplog.set_level(logging.WARNING, logger=_CURATOR_TYPES_LOGGER)

    assert spawned_from_of(metadata) == canonical
    assert _census_lines(caplog) == []


@pytest.mark.parametrize(('stored', 'type_name'), [
    ({'run_id': 'r', 'finding_id': 'f', 'related_task': 2042}, 'dict'),
    (True, 'bool'),
    (1.5, 'float'),
    ('  ', 'str'),
])
def test_a_stored_spawned_from_naming_no_task_reads_as_none_with_a_census_line(
    stored, type_name, caplog,
):
    caplog.set_level(logging.WARNING, logger=_CURATOR_TYPES_LOGGER)

    assert spawned_from_of({'spawned_from': stored}) is None
    [line] = _census_lines(caplog)
    assert 'key=spawned_from' in line
    assert f'type={type_name}' in line
