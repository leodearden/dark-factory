"""Task 8938: a curator candidate carries one canonical spawned_from, whatever shape the store holds."""

from __future__ import annotations

import pytest

from fused_memory.middleware.curator_types import CandidateTask, spawned_from_id


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


@pytest.mark.parametrize('value', [6592, ' 42 ', '', {'id': 42}])
def test_a_candidate_refuses_a_spawned_from_that_is_not_canonical(value):
    with pytest.raises(ValueError) as excinfo:
        CandidateTask(title='T', spawned_from=value)

    assert 'spawned_from' in str(excinfo.value)
    assert repr(value) in str(excinfo.value)


@pytest.mark.parametrize('value', ['42', 'task-x', None])
def test_a_candidate_accepts_a_canonical_spawned_from(value):
    assert CandidateTask(title='T', spawned_from=value).spawned_from == value
