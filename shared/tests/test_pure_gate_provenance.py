"""``metadata.pure_gate_from``: the stamp naming what converted a submission into a pure gate."""

from shared.pure_gate_provenance import PURE_GATE_FROM_KEY, PureGateTrigger
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
