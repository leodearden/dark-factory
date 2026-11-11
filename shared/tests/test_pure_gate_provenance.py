"""``metadata.pure_gate_from``: the stamp naming what converted a submission into a pure gate."""

import pytest

from shared.pure_gate_provenance import PURE_GATE_FROM_KEY, PureGateTrigger, is_pure_gate_shape
from shared.task_metadata import parse_metadata


def test_stamp_is_census_clean_on_write():
    _, warnings = parse_metadata(
        {
            'pure_gate_from': 'execution_class=operational',
            'task_kind': 'deterministic',
            'always_escalates': True,
        },
        direction='write',
    )

    assert [w for w in warnings if w.field == 'pure_gate_from'] == []


def test_stored_value_vocabulary_is_the_wire_contract():
    assert PURE_GATE_FROM_KEY == 'pure_gate_from'
    assert {t.value for t in PureGateTrigger} == {
        'execution_class=operational',
        'execution_class=decision',
        'operational_ask_registry',
    }


@pytest.mark.parametrize(
    ('metadata', 'expected'),
    [
        ({'task_kind': 'deterministic', 'always_escalates': True}, True),
        (
            {
                'task_kind': 'deterministic',
                'always_escalates': True,
                'execution_class': 'operational',
                'operational_mode': 'gate',
            },
            True,
        ),
        ({}, False),
        ({'task_kind': 'normal', 'always_escalates': True}, False),
        ({'task_kind': 'deterministic'}, False),
        ({'task_kind': 'deterministic', 'always_escalates': 'true'}, False),
        (
            {
                'task_kind': 'deterministic',
                'always_escalates': True,
                'before_done': {'kind': 'script', 'script': 'x.sh'},
            },
            False,
        ),
    ],
    ids=[
        'pure_gate',
        'declared_operational_gate',
        'empty',
        'normal_kind',
        'no_always_escalates',
        'always_escalates_not_bool',
        'has_before_done',
    ],
)
def test_is_pure_gate_shape(metadata, expected):
    assert is_pure_gate_shape(metadata) is expected
