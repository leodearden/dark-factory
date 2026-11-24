"""The orchestrator's decision notes pass fused-memory's entities gate (task 10086).

orchestrator/src/orchestrator/decision_note.py::decision_note_fields is the
writer's half of a wire contract: the orchestrator cannot import fused_memory,
so these tests run what it emits through the real
fused_memory/server/entities_gate.py::entities_gate, as
tests/server/test_write_triage_completion_records.py does for
completion_record_metadata. decisions_and_rationale is Graphiti-primary, so the
gate records and conflict-checks the declaration.
"""

from __future__ import annotations

import pytest
from orchestrator.decision_note import decision_note_fields

from fused_memory.server.entities_gate import entities_gate

_PROJECT_ID = 'dark_factory'
_TASK_ID = '4102'
_RATIONALE = 'It already pages the store'
#: An adjacent sibling, one edit from the declared task: the typo axis's evidence.
_NAMES_A_SIBLING = 'Reuse the task 4103 helper for the sweep'
_DECISIONS = {
    'adjacent-sibling': _NAMES_A_SIBLING,
    'foreign-project-same-number': 'Mirror what reify task 4102 did for the sweep',
    'foreign-qualified-same-number': 'Mirror what reify:4102 did for the sweep',
}


@pytest.mark.parametrize('decision_text', _DECISIONS.values(), ids=_DECISIONS)
def test_a_decision_note_passes_the_gate(decision_text: str) -> None:
    """The note's text names the declared task, so the scan corroborates the declaration."""
    fields = decision_note_fields(_TASK_ID, {'decision': decision_text, 'rationale': _RATIONALE})

    assert fields['entities'] == [{'kind': 'task', 'id': _TASK_ID}]
    assert entities_gate(fields['entities'], content=fields['content'], group_id=_PROJECT_ID) is None


def test_the_bare_decision_text_would_have_its_declaration_rejected() -> None:
    """Premise: without the task named in the note, the sibling reads as a typo of the declaration."""
    block = entities_gate(
        [{'kind': 'task', 'id': _TASK_ID}], content=_NAMES_A_SIBLING, group_id=_PROJECT_ID,
    )

    assert block is not None
    assert block['error_type'] == 'DeclaredReferentConflictRejected', block
