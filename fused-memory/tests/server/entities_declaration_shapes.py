"""The malformed ``entities`` declarations every write path must reject. (task 3669)

Every shape here must be rejected by the TOTAL ``InputValidationError`` contract
of ``utils/referent_resolution``. The pure-gate suite and the write-tool
boundary suite parametrize over this one corpus rather than two copies, so a
shape added for one cannot go silently unchecked by the other.

``FALSY_NON_LIST_SHAPES`` is spliced into ``MALFORMED_SHAPES`` and also
exported on its own, for a test that iterates it directly.

Import as ``from server.entities_declaration_shapes import ...``, as with
``server.write_triage_harness``.
"""

from __future__ import annotations

import pytest

#: The falsy NON-list shapes, as ``(label, value)`` pairs.
FALSY_NON_LIST_SHAPES = (('empty-str', ''), ('empty-dict', {}), ('zero', 0))

MALFORMED_SHAPES = (
    pytest.param('task 3127', id='a-str-not-a-list'),
    pytest.param([{'id': 'abc'}], id='non-digit-id'),
    pytest.param([{'id': 3127, 'projectId': 'reify'}], id='unrecognized-key'),
    pytest.param([{'id': True}], id='bool-id'),
    pytest.param([{'id': 3127, 'project_id': 'task'}], id='task-vocabulary-qualifier'),
    pytest.param([[1]], id='a-list-of-non-dicts'),
    pytest.param({'kind': 'task', 'id': 3127}, id='an-unwrapped-single-dict'),
    *(pytest.param(value, id=f'falsy-{label}') for label, value in FALSY_NON_LIST_SHAPES),
)
