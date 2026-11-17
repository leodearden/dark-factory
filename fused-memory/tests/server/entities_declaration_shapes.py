"""The malformed ``entities`` declarations, shared by two suites. (task 3669)

Every shape here must be rejected by the TOTAL ``InputValidationError`` contract
of ``utils/referent_resolution``. Two suites consume the one corpus:

1. ``test_entities_gate.py`` pins what the pure gate answers for each shape.
2. ``test_entities_gate_ingestion.py`` pins what an agent actually receives
   for the same shape at the write-tool boundary. Every shape reaches FastMCP
   only because both write tools annotate ``entities`` as ``Any``.

One corpus rather than two copies, so a shape added for one claim cannot go
silently unchecked for the other.

``FALSY_NON_LIST_SHAPES`` is exported on its own as well as spliced into
``MALFORMED_SHAPES``: ``test_entities_gate.py``'s falsy arm pins a different
property about it, that the gate short-circuits on ``entities is None`` and
never on ``not entities``.

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
