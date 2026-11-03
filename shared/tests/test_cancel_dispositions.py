"""shared.cancel_dispositions: the cancel-with-children contract.

Contract: plans/cancel-with-children-prd.md §4.2. Everything goes through the public names.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.cancel_dispositions import (
    CALLER_ACTIONS,
    CHILD_CANCELLING_ACTIONS,
    ENTRY_ONLY_ACTIONS,
    CancelAction,
    ChildDisposition,
    FoldDisposition,
    KeepDisposition,
    MootDisposition,
    RepointDisposition,
)

# --- the caller disposition shape (§4.2) -------------------------------------


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        pytest.param({'action': 'keep'}, KeepDisposition(), id='keep'),
        pytest.param(
            {'action': 'keep', 'reason': 'stands on main'},
            KeepDisposition(reason='stands on main'),
            id='keep-with-reason',
        ),
        pytest.param(
            {'action': 'repoint', 'onto': [8032]}, RepointDisposition(onto=(8032,)), id='repoint'
        ),
        pytest.param({'action': 'fold', 'into': 8032}, FoldDisposition(into=8032), id='fold'),
        pytest.param({'action': 'moot'}, MootDisposition(), id='moot'),
    ],
)
def test_each_documented_example_parses_to_its_variant(raw, expected):
    parsed = ChildDisposition.model_validate(raw).root

    assert type(parsed) is type(expected)
    assert parsed == expected


def test_repoint_onto_is_held_as_a_tuple():
    parsed = ChildDisposition.model_validate({'action': 'repoint', 'onto': [8032]}).root

    assert isinstance(parsed, RepointDisposition)
    assert parsed.onto == (8032,)


@pytest.mark.parametrize(
    ('disposition', 'field', 'value'),
    [
        pytest.param(KeepDisposition(), 'reason', 'changed', id='keep'),
        pytest.param(RepointDisposition(onto=(1,)), 'onto', (2,), id='repoint'),
        pytest.param(FoldDisposition(into=1), 'into', 2, id='fold'),
        pytest.param(MootDisposition(), 'reason', 'changed', id='moot'),
    ],
)
def test_every_variant_is_frozen(disposition, field, value):
    with pytest.raises(ValidationError):
        setattr(disposition, field, value)


def test_the_action_sets_partition_the_action_enum():
    assert CALLER_ACTIONS == {
        CancelAction.KEEP,
        CancelAction.REPOINT,
        CancelAction.FOLD,
        CancelAction.MOOT,
    }
    assert ENTRY_ONLY_ACTIONS == {CancelAction.CO_CANCELLED}
    assert CHILD_CANCELLING_ACTIONS == {CancelAction.FOLD, CancelAction.MOOT}
    assert CALLER_ACTIONS | ENTRY_ONLY_ACTIONS == set(CancelAction)
    assert not CALLER_ACTIONS & ENTRY_ONLY_ACTIONS
    assert CHILD_CANCELLING_ACTIONS <= CALLER_ACTIONS


@pytest.mark.parametrize(
    'raw',
    [
        pytest.param({'action': 'co_cancelled'}, id='entry-only-action'),
        pytest.param({'action': 'keep', 'onto': [1]}, id='extra-field'),
        pytest.param({'action': 'repoint', 'onto': []}, id='empty-onto'),
        pytest.param({'action': 'repoint', 'onto': [3, 3]}, id='duplicate-onto'),
        pytest.param({'action': 'fold', 'into': True}, id='bool-is-not-an-id'),
        pytest.param({'action': 'moot', 'reason': 'x' * 501}, id='reason-too-long'),
    ],
)
def test_child_disposition_rejects_malformed_input(raw):
    with pytest.raises(ValidationError):
        ChildDisposition.model_validate(raw)
