"""``metadata.pure_gate_from``: the stamp naming what converted a submission into a pure gate."""

import pytest

from shared.pure_gate_provenance import (
    PURE_GATE_FROM_KEY,
    PureGateTrigger,
    is_pure_gate_shape,
    pure_gate_stamped,
)
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


class TestPureGateStamped:
    def test_empty_metadata_becomes_a_stamped_pure_gate(self):
        assert pure_gate_stamped({}, PureGateTrigger.EXECUTION_CLASS_OPERATIONAL) == {
            'task_kind': 'deterministic',
            'always_escalates': True,
            'pure_gate_from': 'execution_class=operational',
        }

    def test_before_done_is_dropped_and_task_kind_overwritten(self):
        stamped = pure_gate_stamped(
            {'task_kind': 'normal', 'before_done': {'kind': 'script', 'script': 'x.sh'}},
            PureGateTrigger.OPERATIONAL_ASK_REGISTRY,
        )
        assert 'before_done' not in stamped
        assert stamped['task_kind'] == 'deterministic'

    def test_sibling_keys_survive_unchanged(self):
        stamped = pure_gate_stamped(
            {'files': ['a.py'], 'execution_class': 'operational'},
            PureGateTrigger.EXECUTION_CLASS_OPERATIONAL,
        )
        assert stamped['files'] == ['a.py']
        assert stamped['execution_class'] == 'operational'

    def test_metadata_already_in_the_pure_gate_shape_is_not_stamped(self):
        declared = {'task_kind': 'deterministic', 'always_escalates': True, 'operational_mode': 'gate'}
        assert pure_gate_stamped(declared, PureGateTrigger.EXECUTION_CLASS_OPERATIONAL) == declared

    def test_an_existing_stamp_on_a_pure_gate_is_kept(self):
        already_coerced = {
            'task_kind': 'deterministic',
            'always_escalates': True,
            'pure_gate_from': 'operational_ask_registry',
        }
        stamped = pure_gate_stamped(already_coerced, PureGateTrigger.EXECUTION_CLASS_DECISION)
        assert stamped['pure_gate_from'] == 'operational_ask_registry'

    def test_a_stale_stamp_on_a_rewritten_submission_names_the_new_trigger(self):
        retyped = {'task_kind': 'normal', 'pure_gate_from': 'operational_ask_registry'}
        stamped = pure_gate_stamped(retyped, PureGateTrigger.EXECUTION_CLASS_DECISION)
        assert stamped['pure_gate_from'] == 'execution_class=decision'

    def test_input_is_not_mutated_and_a_new_dict_is_returned(self):
        original = {'keep': 1, 'before_done': {'kind': 'script', 'script': 'x.sh'}}
        snapshot = {'keep': 1, 'before_done': {'kind': 'script', 'script': 'x.sh'}}
        stamped = pure_gate_stamped(original, PureGateTrigger.EXECUTION_CLASS_OPERATIONAL)
        assert stamped is not original
        assert original == snapshot

    @pytest.mark.parametrize(
        'metadata',
        [
            {},
            {'task_kind': 'normal'},
            {'task_kind': 'deterministic', 'always_escalates': 'true'},
            {'task_kind': 'deterministic', 'always_escalates': True},
            {'before_done': {'kind': 'script', 'script': 'x.sh'}, 'always_escalates': False},
            {'pure_gate_from': 'execution_class=decision', 'files': ['a.py']},
        ],
        ids=repr,
    )
    @pytest.mark.parametrize('trigger', list(PureGateTrigger))
    def test_result_is_always_in_the_pure_gate_shape(self, metadata, trigger):
        assert is_pure_gate_shape(pure_gate_stamped(metadata, trigger))
