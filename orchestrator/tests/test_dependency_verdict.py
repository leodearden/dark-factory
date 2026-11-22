"""The tri-state dependency answer in orchestrator.dependency_verdict.

Dispatch reads only ``.satisfied``, so an UNKNOWN verdict waits like an
UNSATISFIED one; park GC expires only on UNSATISFIED.  ``combine`` therefore
lets affirmative evidence from any arm win over another arm's ignorance, and
the kind of a non-satisfied verdict is rendered into ``reservation_expired``
reasons, so its wire values are behaviour.  ``deps_verdict`` must therefore
say UNSATISFIED only on affirmative evidence and UNKNOWN on absence of it.
"""

from __future__ import annotations

import logging

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
    assert [kind.value for kind in DepKind] == [
        'local',
        'external',
        'delivered_check',
        'scheduled_deploy',
    ]


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


@pytest.mark.parametrize(
    ('declaration', 'expected'),
    [
        (
            {'name': 'cap', 'kind': 'grep', 'pattern': 'X', 'expect': 'present'},
            DepsVerdict.unknown(DepKind.DELIVERED_CHECK),
        ),
        ({}, SATISFIED),
    ],
    ids=['bare-dict-is-still-gated', 'empty-dict-declares-nothing'],
)
def test_the_delivered_check_arm_gates_every_non_empty_declaration(declaration, expected):
    dep = {'id': 'D', 'status': 'done', 'metadata': {'delivered_checks': declaration}}
    verdict = deps_verdict(
        _dependent('D'),
        {'D': 'done'},
        {},
        delivered_check_cache={},
        terminal_dep_records={'D': dep},
    )
    assert verdict == expected


def test_a_local_dependency_without_a_status_is_unsatisfied():
    assert deps_verdict(_dependent('D'), {}) == DepsVerdict.unsatisfied(DepKind.LOCAL)


@pytest.mark.parametrize(
    'dependencies',
    [
        '12,13',
        '12, 13',
        [{'id': 12}, {'id': '13'}],
        [12, 13],
        ['12', '13'],
        [{'id': 12}, 13],
        (12, '13'),
        {12, 13},
    ],
    ids=[
        'csv-string',
        'csv-string-with-spaces',
        'dict-shaped-list',
        'int-list',
        'str-list',
        'mixed-list',
        'tuple',
        'set',
    ],
)
def test_the_local_arm_reads_two_dependency_ids_in_every_multi_dep_shape(dependencies):
    task = {'id': 'T', 'dependencies': dependencies}
    assert deps_verdict(task, {'12': 'done', '13': 'done'}) == SATISFIED
    assert deps_verdict(task, {'12': 'done', '13': 'pending'}) == DepsVerdict.unsatisfied(
        DepKind.LOCAL
    )
    assert deps_verdict(task, {'12': 'done'}) == DepsVerdict.unsatisfied(DepKind.LOCAL)


def test_the_local_arm_reads_a_bare_int_as_one_dependency_id():
    task = {'id': 'T', 'dependencies': 12}
    assert deps_verdict(task, {'12': 'done'}) == SATISFIED
    assert deps_verdict(task, {'12': 'pending'}) == DepsVerdict.unsatisfied(DepKind.LOCAL)


def test_a_blocked_local_dependency_wins_over_an_unknown_external_one():
    verdict = deps_verdict(
        _dependent('D', external_deps=(EXTERNAL_DEP,)),
        {'D': 'pending'},
        {},
        external_status_cache={},
    )
    assert verdict == DepsVerdict.unsatisfied(DepKind.LOCAL)


def _scheduled_dep(phase: str, provenance_kind: str = 'deterministic-deploy-scheduled') -> dict:
    return {
        'id': 'D',
        'status': 'done',
        'metadata': {
            'done_provenance': {
                'kind': provenance_kind,
                'unit': 'orchestrator-dark-factory.service',
                'transient_unit': 'orch-redeploy-restart-D.service',
                'fire_delay_secs': 60,
            },
            'before_done_scheduled_at': {
                'at': '2026-11-09T10:00:00+00:00',
                'transient_unit': 'orch-redeploy-restart-D.service',
                'fire_delay_secs': 60,
            },
            'deploy_state': {'phase': phase},
        },
    }


@pytest.mark.parametrize(
    ('phase', 'expected'),
    [
        ('scheduled', DepsVerdict.unsatisfied(DepKind.SCHEDULED_DEPLOY)),
        ('failed', DepsVerdict.unsatisfied(DepKind.SCHEDULED_DEPLOY)),
        ('verified', SATISFIED),
        ('done', SATISFIED),
    ],
)
def test_a_done_scheduled_self_deploy_holds_its_dependent_until_the_restart_is_verified(
    phase, expected
):
    verdict = deps_verdict(
        _dependent('D'),
        {'D': 'done'},
        {},
        terminal_dep_records={'D': _scheduled_dep(phase)},
    )
    assert verdict == expected


def test_a_scheduled_self_deploy_record_in_tasks_by_id_also_holds():
    verdict = deps_verdict(_dependent('D'), {'D': 'done'}, {'D': _scheduled_dep('scheduled')})
    assert verdict == DepsVerdict.unsatisfied(DepKind.SCHEDULED_DEPLOY)


@pytest.mark.parametrize('terminal_dep_records', [{}, None], ids=['absent', 'no-backfill'])
def test_a_done_dependency_without_a_record_does_not_hold(terminal_dep_records):
    verdict = deps_verdict(
        _dependent('D'), {'D': 'done'}, {}, terminal_dep_records=terminal_dep_records
    )
    assert verdict == SATISFIED


def test_a_done_dependency_the_backfill_deferred_before_caching_is_unknown():
    verdict = deps_verdict(
        _dependent('D'), {'D': 'done'}, {}, terminal_dep_records={}, deferred_terminal_deps={'D'}
    )
    assert verdict == DepsVerdict.unknown(DepKind.SCHEDULED_DEPLOY)


@pytest.mark.parametrize(
    ('phase', 'expected'),
    [
        ('scheduled', DepsVerdict.unsatisfied(DepKind.SCHEDULED_DEPLOY)),
        ('verified', SATISFIED),
    ],
)
def test_a_deferred_dependency_with_a_record_follows_its_record(phase, expected):
    verdict = deps_verdict(
        _dependent('D'),
        {'D': 'done'},
        {},
        terminal_dep_records={'D': _scheduled_dep(phase)},
        deferred_terminal_deps={'D'},
    )
    assert verdict == expected


def test_a_done_deploy_that_was_not_scheduled_does_not_hold():
    record = _scheduled_dep('scheduled', provenance_kind='deterministic-deploy')
    verdict = deps_verdict(_dependent('D'), {'D': 'done'}, {}, terminal_dep_records={'D': record})
    assert verdict == SATISFIED


def test_a_non_terminal_dependency_is_reported_by_the_local_arm_first():
    verdict = deps_verdict(
        _dependent('D'), {'D': 'pending'}, {}, terminal_dep_records={'D': _scheduled_dep('scheduled')}
    )
    assert verdict == DepsVerdict.unsatisfied(DepKind.LOCAL)


def test_the_scheduled_deploy_hold_is_logged(caplog):
    with caplog.at_level(logging.DEBUG, logger='orchestrator.scheduler'):
        deps_verdict(
            _dependent('D'), {'D': 'done'}, {}, terminal_dep_records={'D': _scheduled_dep('scheduled')}
        )
    held = [r.getMessage() for r in caplog.records if r.name == 'orchestrator.scheduler']
    assert any('Task T' in m and 'dep D' in m and "'scheduled'" in m for m in held), held


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


# --- The early-start dependency answer (PRD plans/early-start-on-predecessor-branch-prd.md section 7) ---


def _train_member(train_id: str, *local_deps: str) -> dict:
    return {'id': 'T', 'dependencies': list(local_deps), 'metadata': {'train': {'id': train_id}}}


def _merge_deferred(dep_id: str, train_id: str) -> dict:
    return {'id': dep_id, 'status': 'merge-deferred', 'metadata': {'train': {'id': train_id}}}


#: Answers recorded from deps_verdict before the waiting_predecessors input existed.
_RECORDED = [
    pytest.param(_dependent(), {}, {}, SATISFIED, id='no-deps'),
    pytest.param(_dependent('D', 'E'), {'D': 'done', 'E': 'done'}, {}, SATISFIED, id='all-done'),
    pytest.param(_dependent('D'), {'D': 'cancelled'}, {}, SATISFIED, id='cancelled-dep'),
    pytest.param(
        _dependent('P'), {'P': 'pending'}, {}, DepsVerdict.unsatisfied(DepKind.LOCAL),
        id='pending-dep',
    ),
    pytest.param(
        _dependent('P'), {}, {}, DepsVerdict.unsatisfied(DepKind.LOCAL), id='dep-without-status',
    ),
    pytest.param(
        _train_member('t1', 'P'),
        {'P': 'merge-deferred'},
        {'tasks_by_id': {'P': _merge_deferred('P', 't1')}},
        SATISFIED,
        id='merge-deferred-same-train',
    ),
    pytest.param(
        _train_member('t1', 'P'),
        {'P': 'merge-deferred'},
        {'tasks_by_id': {'P': _merge_deferred('P', 't2')}},
        DepsVerdict.unsatisfied(DepKind.LOCAL),
        id='merge-deferred-other-train',
    ),
    pytest.param(
        _dependent('D'),
        {'D': 'done'},
        {'terminal_dep_records': {'D': _scheduled_dep('scheduled')}},
        DepsVerdict.unsatisfied(DepKind.SCHEDULED_DEPLOY),
        id='held-scheduled-self-deploy',
    ),
    pytest.param(
        _dependent('D'),
        {'D': 'done'},
        {'terminal_dep_records': {}, 'deferred_terminal_deps': {'D'}},
        DepsVerdict.unknown(DepKind.SCHEDULED_DEPLOY),
        id='self-deploy-deferred-by-the-fetch-budget',
    ),
    pytest.param(
        _dependent(external_deps=(EXTERNAL_DEP,)),
        {},
        {'external_status_cache': {EXTERNAL_DEP: 'done'}},
        SATISFIED,
        id='external-done',
    ),
    pytest.param(
        _dependent(external_deps=(EXTERNAL_DEP,)),
        {},
        {'external_status_cache': {}},
        DepsVerdict.unknown(DepKind.EXTERNAL),
        id='external-missing-from-the-cache',
    ),
    pytest.param(
        _dependent(external_deps=(EXTERNAL_DEP,)),
        {},
        {'external_status_cache': {EXTERNAL_DEP: 'in-progress'}, 'external_resolver_failed': True},
        DepsVerdict.unknown(DepKind.EXTERNAL),
        id='external-resolver-failed',
    ),
    pytest.param(
        _dependent('D'),
        {'D': 'done'},
        {'tasks_by_id': {}, 'delivered_check_cache': {'D': True}, 'terminal_dep_records': {'D': CHECKED_DEP}},
        SATISFIED,
        id='delivered-check-true',
    ),
    pytest.param(
        _dependent('D'),
        {'D': 'done'},
        {'tasks_by_id': {}, 'delivered_check_cache': {'D': False}, 'terminal_dep_records': {'D': CHECKED_DEP}},
        DepsVerdict.unsatisfied(DepKind.DELIVERED_CHECK),
        id='delivered-check-false',
    ),
    pytest.param(
        _dependent('D'),
        {'D': 'done'},
        {'tasks_by_id': {}, 'delivered_check_cache': {}, 'terminal_dep_records': {'D': CHECKED_DEP}},
        DepsVerdict.unknown(DepKind.DELIVERED_CHECK),
        id='delivered-check-absent-from-the-cache',
    ),
    pytest.param(
        _dependent('P', 'Q'),
        {'P': 'pending', 'Q': 'pending'},
        {},
        DepsVerdict.unsatisfied(DepKind.LOCAL),
        id='two-pending-deps',
    ),
    pytest.param(
        _dependent('P', external_deps=(EXTERNAL_DEP,)),
        {'P': 'pending'},
        {'tasks_by_id': {}, 'external_status_cache': {}},
        DepsVerdict.unsatisfied(DepKind.LOCAL),
        id='pending-dep-and-unknown-external-dep',
    ),
]


@pytest.mark.parametrize(('task', 'status_map', 'kwargs', 'recorded'), _RECORDED)
class TestCallersWithoutTheInputGetTodaysAnswers:
    """Invariant 9: the input is opt-in, and it never turns an unfinished dependency satisfied."""

    def test_no_input_answers_as_recorded(self, task, status_map, kwargs, recorded):
        assert deps_verdict(task, status_map, **kwargs) == recorded

    def test_an_empty_input_answers_as_recorded(self, task, status_map, kwargs, recorded):
        verdict = deps_verdict(task, status_map, **kwargs, waiting_predecessors=frozenset())
        assert verdict == recorded

    def test_every_dep_waiting_keeps_the_recorded_state_and_kind(
        self, task, status_map, kwargs, recorded
    ):
        every_dep = frozenset(task['dependencies'])
        verdict = deps_verdict(task, status_map, **kwargs, waiting_predecessors=every_dep)
        assert (verdict.satisfied, verdict.state, verdict.kind) == (
            recorded.satisfied, recorded.state, recorded.kind,
        )


PLAIN_LOCAL = DepsVerdict.unsatisfied(DepKind.LOCAL)


def _checked(dep_id: str) -> dict:
    return {'id': dep_id, 'status': 'done', 'metadata': {'delivered_checks': ['check']}}


class TestTheEarlyStartAnswer:
    def test_the_only_unfinished_dep_waiting_names_its_predecessor(self):
        verdict = deps_verdict(_dependent('P'), {'P': 'in-progress'}, waiting_predecessors={'P'})

        assert verdict == DepsVerdict.waiting_on('P')
        assert not verdict.satisfied
        assert (verdict.state, verdict.kind) == (DepState.UNSATISFIED, DepKind.LOCAL)
        assert verdict.waiting_predecessor == 'P'

    def test_a_dep_outside_the_set_gets_the_plain_answer(self):
        verdict = deps_verdict(_dependent('P'), {'P': 'in-progress'}, waiting_predecessors={'Z'})

        assert verdict == PLAIN_LOCAL
        assert verdict.waiting_predecessor is None

    @pytest.mark.parametrize('waiting', [{'P'}, {'P', 'Q'}], ids=['one-waiting', 'both-waiting'])
    def test_two_unfinished_deps_get_the_plain_answer(self, waiting):
        verdict = deps_verdict(
            _dependent('P', 'Q'),
            {'P': 'in-progress', 'Q': 'in-progress'},
            waiting_predecessors=waiting,
        )

        assert verdict == PLAIN_LOCAL

    @pytest.mark.parametrize(
        ('task', 'kwargs'),
        [
            pytest.param(
                _dependent('P', 'D'),
                {'tasks_by_id': {}, 'delivered_check_cache': {'D': False},
                 'terminal_dep_records': {'D': _checked('D')}},
                id='done-dep-check-failed',
            ),
            pytest.param(
                _dependent('P', 'D'),
                {'tasks_by_id': {}, 'delivered_check_cache': {},
                 'terminal_dep_records': {'D': _checked('D')}},
                id='done-dep-check-absent-from-the-cache',
            ),
            pytest.param(
                _dependent('P', 'D', external_deps=(EXTERNAL_DEP,)),
                {'tasks_by_id': {}, 'external_status_cache': {}},
                id='external-dep-unknown',
            ),
            pytest.param(
                _dependent('P', 'D'),
                {'terminal_dep_records': {'D': _scheduled_dep('scheduled')}},
                id='done-dep-a-held-scheduled-self-deploy',
            ),
        ],
    )
    def test_another_arm_that_does_not_pass_gives_the_plain_answer(self, task, kwargs):
        verdict = deps_verdict(
            task, {'P': 'in-progress', 'D': 'done'}, **kwargs, waiting_predecessors={'P'},
        )

        assert verdict == PLAIN_LOCAL

    def test_every_other_arm_passing_names_the_predecessor(self):
        verdict = deps_verdict(
            _dependent('P', 'D', 'E', external_deps=(EXTERNAL_DEP,)),
            {'P': 'in-progress', 'D': 'done', 'E': 'done'},
            {},
            external_status_cache={EXTERNAL_DEP: 'done'},
            delivered_check_cache={'D': True, 'E': True},
            terminal_dep_records={'D': _checked('D'), 'E': _checked('E')},
            waiting_predecessors={'P'},
        )

        assert verdict == DepsVerdict.waiting_on('P')


class TestTheWaitingPredecessorInvariant:
    @pytest.mark.parametrize(
        ('state', 'kind'),
        [
            (DepState.SATISFIED, None),
            (DepState.UNKNOWN, DepKind.LOCAL),
            (DepState.UNSATISFIED, DepKind.EXTERNAL),
            (DepState.UNSATISFIED, DepKind.DELIVERED_CHECK),
        ],
    )
    def test_only_an_unsatisfied_local_verdict_names_a_predecessor(self, state, kind):
        with pytest.raises(ValueError) as excinfo:
            DepsVerdict(state, kind, waiting_predecessor='P')
        assert str(state) in str(excinfo.value)
        assert "'P'" in str(excinfo.value)

    def test_an_empty_predecessor_id_is_refused(self):
        with pytest.raises(ValueError):
            DepsVerdict.waiting_on('')
