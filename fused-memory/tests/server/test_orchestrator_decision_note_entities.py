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


#: Task numbers of each shape the orchestrator declares: one digit, a leading
#: zero, today's width, and a long run that a length cap would catch.
_DECLARED_IDS = ('7', '0132', '4102', '123456789012')
#: Ids the orchestrator declares nothing for.
_UNDECLARED_IDS = {'eval-run': 't', 'suffixed': '4102a', 'arabic-indic-digits': '٤١٠٢', 'empty': ''}
_NEUTRAL_DECISION = {'decision': 'Key the sweep on metadata', 'rationale': _RATIONALE}


@pytest.mark.parametrize('task_id', _DECLARED_IDS)
def test_every_task_number_the_note_declares_passes_the_gate(task_id: str) -> None:
    """The orchestrator's task-number test must accept nothing the gate refuses: a refused note is lost unlogged."""
    fields = decision_note_fields(task_id, _NEUTRAL_DECISION)

    assert fields['entities'] == [{'kind': 'task', 'id': task_id}]
    assert entities_gate(fields['entities'], content=fields['content'], group_id=_PROJECT_ID) is None


@pytest.mark.parametrize('task_id', _UNDECLARED_IDS.values(), ids=_UNDECLARED_IDS)
def test_the_gate_refuses_a_declaration_of_an_id_the_note_omits(task_id: str) -> None:
    """Premise for omitting the declaration: declaring the id would get the note refused."""
    fields = decision_note_fields(task_id, _NEUTRAL_DECISION)
    block = entities_gate(
        [{'kind': 'task', 'id': task_id}], content=fields['content'], group_id=_PROJECT_ID,
    )

    assert 'entities' not in fields
    assert block is not None
    assert block['error_type'] == 'ValidationError', block
