"""The tri-state dependency answer in orchestrator.dependency_verdict.

Dispatch reads only ``.satisfied``, so an UNKNOWN verdict waits like an
UNSATISFIED one; park GC expires only on UNSATISFIED.  ``combine`` therefore
lets affirmative evidence from any arm win over another arm's ignorance, and
the kind of a non-satisfied verdict is rendered into ``reservation_expired``
reasons, so its wire values are behaviour.  ``deps_verdict`` must therefore
say UNSATISFIED only on affirmative evidence and UNKNOWN on absence of it.
"""

from __future__ import annotations

import pytest

from orchestrator.dependency_verdict import (
    DepKind,
    DepState,
    DepsVerdict,
    deps_verdict,
    task_external_deps,
)

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


EXTERNAL_DEP = 'other_project:7'


def _dependent(*local_deps: str, external_deps: tuple[str, ...] = ()) -> dict:
    return {
        'id': 'T',
        'dependencies': list(local_deps),
        'metadata': {'external_deps': list(external_deps)} if external_deps else {},
    }


@pytest.mark.parametrize(
    ('external_status_cache', 'resolver_failed', 'expected'),
    [
        (None, False, SATISFIED),
        ({EXTERNAL_DEP: 'done'}, False, SATISFIED),
        ({EXTERNAL_DEP: 'in-progress'}, False, DepsVerdict.unsatisfied(DepKind.EXTERNAL)),
        ({EXTERNAL_DEP: 'unknown_task'}, False, DepsVerdict.unsatisfied(DepKind.EXTERNAL)),
        ({}, False, DepsVerdict.unknown(DepKind.EXTERNAL)),
        ({EXTERNAL_DEP: 'in-progress'}, True, DepsVerdict.unknown(DepKind.EXTERNAL)),
    ],
    ids=[
        'inactive-without-a-cache',
        'done',
        'reported-live-status',
        'reported-sentinel',
        'absent-from-the-answer',
        'resolver-failed',
    ],
)
def test_the_external_arm_is_unsatisfied_only_on_a_reported_status(
    external_status_cache, resolver_failed, expected
):
    verdict = deps_verdict(
        _dependent(external_deps=(EXTERNAL_DEP,)),
        {},
        {},
        external_status_cache=external_status_cache,
        external_resolver_failed=resolver_failed,
    )
    assert verdict == expected


CHECKED_DEP = {'id': 'D', 'status': 'done', 'metadata': {'delivered_checks': ['check']}}


@pytest.mark.parametrize(
    ('delivered_check_cache', 'expected'),
    [
        (None, SATISFIED),
        ({'D': True}, SATISFIED),
        ({'D': False}, DepsVerdict.unsatisfied(DepKind.DELIVERED_CHECK)),
        ({}, DepsVerdict.unknown(DepKind.DELIVERED_CHECK)),
    ],
    ids=['inactive-without-a-cache', 'delivered', 'check-failed', 'not-evaluated'],
)
def test_the_delivered_check_arm_is_unsatisfied_only_on_a_failed_check(
    delivered_check_cache, expected
):
    verdict = deps_verdict(
        _dependent('D'),
        {'D': 'done'},
        {},
        delivered_check_cache=delivered_check_cache,
        terminal_dep_records={'D': CHECKED_DEP},
    )
    assert verdict == expected


def test_a_local_dependency_without_a_status_is_unsatisfied():
    assert deps_verdict(_dependent('D'), {}) == DepsVerdict.unsatisfied(DepKind.LOCAL)


def test_a_blocked_local_dependency_wins_over_an_unknown_external_one():
    verdict = deps_verdict(
        _dependent('D', external_deps=(EXTERNAL_DEP,)),
        {'D': 'pending'},
        {},
        external_status_cache={},
    )
    assert verdict == DepsVerdict.unsatisfied(DepKind.LOCAL)


class TestTaskExternalDeps:
    """``task_external_deps`` reads ``metadata.external_deps`` best-effort (task 2167 SEAM A).

    ``direction='read'`` never raises, so a corrupt historical metadata row
    cannot break a dispatch tick, and the deps stay in the canonical
    "project_id:task_id" wire STRING form (PRD Open Question #4).
    """

    def test_no_metadata_key_returns_empty_list(self):
        assert task_external_deps({'id': '1'}) == []

    def test_metadata_none_returns_empty_list(self):
        assert task_external_deps({'id': '2', 'metadata': None}) == []

    def test_dict_metadata_returns_external_deps(self):
        task = {'id': '3', 'metadata': {'external_deps': ['dark_factory:42', 'p:9']}}
        assert task_external_deps(task) == ['dark_factory:42', 'p:9']

    def test_json_string_metadata_is_parsed(self):
        """A JSON *string* is parsed; the old ``.get`` idiom raised AttributeError on it."""
        task = {'id': '4', 'metadata': '{"external_deps": ["a:1"]}'}
        assert task_external_deps(task) == ['a:1']

    def test_corrupt_metadata_string_returns_empty_list_no_raise(self):
        assert task_external_deps({'id': '5', 'metadata': 'not json{'}) == []

    def test_returned_elements_stay_string_wire_form(self):
        task = {'id': '6', 'metadata': {'external_deps': ['dark_factory:42', 'p:9']}}
        assert all(isinstance(dep, str) for dep in task_external_deps(task))
