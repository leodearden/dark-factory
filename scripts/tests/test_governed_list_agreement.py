"""scripts/governed_list_agreement.py: the register must read a declarations module as it evaluates."""

import pytest
from governed_list_agreement import assert_register_reads_as_evaluated
from shared.governed_exceptions import governed_exceptions


def test_a_module_declaring_nothing_agrees_with_a_namespace_holding_no_list() -> None:
    assert_register_reads_as_evaluated(__file__, {})


def test_a_list_built_inside_a_function_is_one_the_register_cannot_read() -> None:
    unread = governed_exceptions('probe.built_inside_a_function', 'A rule.', [])
    with pytest.raises(AssertionError, match='probe.built_inside_a_function'):
        assert_register_reads_as_evaluated(__file__, {'unread': unread})
