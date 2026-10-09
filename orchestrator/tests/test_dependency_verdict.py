"""The tri-state dependency answer in orchestrator.dependency_verdict.

Dispatch reads only ``.satisfied``, so an UNKNOWN verdict waits like an
UNSATISFIED one; park GC expires only on UNSATISFIED.  ``combine`` therefore
lets affirmative evidence from any arm win over another arm's ignorance, and
the kind of a non-satisfied verdict is rendered into ``reservation_expired``
reasons, so its wire values are behaviour.
"""

from __future__ import annotations

import pytest

from orchestrator.dependency_verdict import DepKind, DepState, DepsVerdict

SATISFIED = DepsVerdict(DepState.SATISFIED)


@pytest.mark.parametrize(
    ('verdicts', 'expected'),
    [
        (
            [DepsVerdict.unknown(DepKind.EXTERNAL), DepsVerdict.unsatisfied(DepKind.DELIVERED_CHECK)],
            DepsVerdict.unsatisfied(DepKind.DELIVERED_CHECK),
        ),
        (
            [DepsVerdict.unsatisfied(DepKind.LOCAL), DepsVerdict.unsatisfied(DepKind.EXTERNAL)],
            DepsVerdict.unsatisfied(DepKind.LOCAL),
        ),
        (
            [SATISFIED, DepsVerdict.unknown(DepKind.EXTERNAL), DepsVerdict.unknown(DepKind.DELIVERED_CHECK)],
            DepsVerdict.unknown(DepKind.EXTERNAL),
        ),
        ([SATISFIED, SATISFIED, SATISFIED], SATISFIED),
        ([], SATISFIED),
    ],
    ids=[
        'unsatisfied-beats-earlier-unknown',
        'first-unsatisfied-wins',
        'first-unknown-without-unsatisfied',
        'all-satisfied',
        'no-arms',
    ],
)
def test_combine_prefers_the_first_unsatisfied_then_the_first_unknown(verdicts, expected):
    assert DepsVerdict.combine(verdicts) == expected


def test_combine_accepts_any_iterable():
    assert DepsVerdict.combine(iter([DepsVerdict.unknown(DepKind.LOCAL)])) == DepsVerdict.unknown(
        DepKind.LOCAL
    )


@pytest.mark.parametrize(
    ('verdict', 'satisfied'),
    [
        (SATISFIED, True),
        (DepsVerdict.unknown(DepKind.EXTERNAL), False),
        (DepsVerdict.unsatisfied(DepKind.LOCAL), False),
    ],
)
def test_only_a_satisfied_verdict_reads_as_satisfied(verdict, satisfied):
    assert verdict.satisfied is satisfied


@pytest.mark.parametrize(
    ('state', 'kind'),
    [
        (DepState.SATISFIED, DepKind.LOCAL),
        (DepState.UNKNOWN, None),
        (DepState.UNSATISFIED, None),
    ],
)
def test_a_verdict_names_a_kind_exactly_when_it_is_not_satisfied(state, kind):
    with pytest.raises(ValueError) as excinfo:
        DepsVerdict(state, kind)
    assert str(state) in str(excinfo.value)
    assert str(kind) in str(excinfo.value)


def test_dependency_kind_wire_values():
    assert [kind.value for kind in DepKind] == ['local', 'external', 'delivered_check']
