"""C3 boundary policy contract — the declared matrix, then the behaviour.

This file opens with assertions on the module's DATA, before any middleware
runs. That is deliberate and is what INV-1 asks for: the policy is a declared,
total matrix over outcome x tool class, so it can be checked as a table rather
than inferred by driving every path and hoping the cases were exhaustive.

No test here asserts on docstring or comment prose.
"""

from __future__ import annotations

import itertools
from typing import get_args

import pytest

from shared import uuid_prefix_guard as guard

# --- the resolution vocabulary is typed and closed ----------------------


def test_candidate_and_resolution_are_namedtuples() -> None:
    for declared in (guard.Candidate, guard.Resolution):
        assert issubclass(declared, tuple)
        assert hasattr(declared, '_fields')
        assert hasattr(declared, '_replace')


def test_candidate_fields() -> None:
    assert guard.Candidate._fields == ('namespace', 'id', 'preview')


def test_resolution_fields() -> None:
    assert guard.Resolution._fields == ('outcome', 'candidates')


def test_namespace_constants_are_derived_from_the_literal_type() -> None:
    """One source for the type and the constant, so they cannot drift apart."""
    assert get_args(guard.Namespace) == guard.NAMESPACES
    assert set(guard.NAMESPACES) == {'mem0', 'graphiti_node', 'graphiti_edge'}


def test_resolution_outcome_constants_are_derived_from_the_literal_type() -> None:
    assert get_args(guard.ResolutionOutcome) == guard.RESOLUTION_OUTCOMES
    assert set(guard.RESOLUTION_OUTCOMES) == {'unique', 'ambiguous', 'none'}


def test_resolver_unavailable_is_an_exception_carrying_the_failed_store() -> None:
    """INV-11: an outage must name which store failed, never fail soft silently."""
    assert issubclass(guard.ResolverUnavailable, Exception)
    error = guard.ResolverUnavailable('mem0')
    assert error.store == 'mem0'
    assert 'mem0' in str(error)


# --- the fact and tool-class vocabularies -------------------------------


def test_fact_outcome_vocabulary_is_exactly_the_four_declared_values() -> None:
    assert get_args(guard.FactOutcome) == guard.FACT_OUTCOMES
    assert set(guard.FACT_OUTCOMES) == {
        'expanded',
        'rejected',
        'forwarded_ambiguous',
        'resolver_unavailable',
    }


def test_tool_classes_are_exactly_the_three_declared_ones() -> None:
    assert {c.value for c in guard.ToolClass} == {'default', 'forward_on_ambiguity', 'exempt'}


def test_prefix_actions_are_exactly_the_five_declared_ones() -> None:
    assert {a.value for a in guard.PrefixAction} == {
        'substitute_and_forward',
        'reject_with_candidates',
        'forward_with_candidates',
        'forward_unchanged',
        'inert',
    }


# --- the matrix is TOTAL ------------------------------------------------

NON_EXEMPT_CLASSES = (guard.ToolClass.DEFAULT, guard.ToolClass.FORWARD_ON_AMBIGUITY)


@pytest.mark.parametrize(
    ('outcome', 'tool_class'),
    list(itertools.product(guard.RESOLUTION_OUTCOMES, NON_EXEMPT_CLASSES)),
)
def test_every_cell_is_declared(
    outcome: guard.ResolutionOutcome, tool_class: guard.ToolClass
) -> None:
    """No cell is reachable by falling off the end of a lookup (INV-1).

    Parametrized over the PRODUCT of the two vocabularies rather than over a
    hand-listed set of pairs, so adding an outcome or a tool class without
    declaring its cells fails here instead of at a call site in production.
    """
    assert (outcome, tool_class) in guard.POLICY_MATRIX
    assert isinstance(guard.POLICY_MATRIX[(outcome, tool_class)], guard.PrefixAction)


def test_matrix_declares_no_cell_beyond_the_product() -> None:
    expected = set(itertools.product(guard.RESOLUTION_OUTCOMES, NON_EXEMPT_CLASSES))
    assert set(guard.POLICY_MATRIX) == expected


def test_matrix_transcribes_the_prd_table() -> None:
    """PRD §4-C3's table, cell by cell."""
    m = guard.POLICY_MATRIX
    assert m[('unique', guard.ToolClass.DEFAULT)] is guard.PrefixAction.SUBSTITUTE_AND_FORWARD
    assert m[('unique', guard.ToolClass.FORWARD_ON_AMBIGUITY)] is guard.PrefixAction.SUBSTITUTE_AND_FORWARD
    assert m[('ambiguous', guard.ToolClass.DEFAULT)] is guard.PrefixAction.REJECT_WITH_CANDIDATES
    assert m[('ambiguous', guard.ToolClass.FORWARD_ON_AMBIGUITY)] is guard.PrefixAction.FORWARD_WITH_CANDIDATES
    assert m[('none', guard.ToolClass.DEFAULT)] is guard.PrefixAction.INERT
    assert m[('none', guard.ToolClass.FORWARD_ON_AMBIGUITY)] is guard.PrefixAction.INERT


def test_matrix_is_immutable() -> None:
    """The matrix IS the policy, so it must not be edited at runtime."""
    with pytest.raises(TypeError):
        guard.POLICY_MATRIX[('unique', guard.ToolClass.DEFAULT)] = guard.PrefixAction.INERT  # type: ignore[index]


def test_exempt_is_not_a_key_in_the_matrix() -> None:
    """An exemption is a declaration that this is not a repair site.

    It short-circuits ahead of resolution, so it has no outcome and therefore
    no row — a cell for it would imply the resolver had been consulted.
    """
    for outcome in guard.RESOLUTION_OUTCOMES:
        assert (outcome, guard.ToolClass.EXEMPT) not in guard.POLICY_MATRIX


# --- what is storm-counted, declared as data ----------------------------


def test_storm_counted_outcomes_are_exactly_the_two_fail_soft_ones() -> None:
    """INV-4, and the reason `expanded` is excluded is a measurement.

    `expanded` is the DESIGNED SUCCESS PATH. At ~10% of 124 writes/day it
    would fire the 3/3600 thresholds continuously and be ignored, which is how
    a storm escape stops being an escape.
    """
    assert frozenset(
        {'forwarded_ambiguous', 'resolver_unavailable'}
    ) == guard.STORM_COUNTED_OUTCOMES


def test_expanded_is_not_storm_counted() -> None:
    assert 'expanded' not in guard.STORM_COUNTED_OUTCOMES


def test_rejected_is_not_storm_counted() -> None:
    """A rejection is not fail-soft: the caller is told, so it needs no escape."""
    assert 'rejected' not in guard.STORM_COUNTED_OUTCOMES


def test_storm_counted_outcomes_are_all_real_fact_outcomes() -> None:
    assert set(guard.FACT_OUTCOMES) >= guard.STORM_COUNTED_OUTCOMES
