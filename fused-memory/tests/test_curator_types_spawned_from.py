"""Task 8938: a curator candidate carries one canonical spawned_from, whatever shape the store holds."""

from __future__ import annotations

import pytest

from fused_memory.middleware.curator_types import spawned_from_id


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
