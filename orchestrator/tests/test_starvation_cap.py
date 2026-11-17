"""The auto-pin cap L1: what it says about the waiting cohort, and when it is filed, refreshed or resolved."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.blocker_liveness import HolderState, LockWorld, NonLiveHolder
from orchestrator.pin_reservation import Blocker, BlockerKind
from orchestrator.starvation_cap import (
    CapAction,
    CapReport,
    CohortMember,
    cap_action,
    cap_report,
    cap_text,
    is_news,
    next_baseline,
    sync_cap_l1,
)
from orchestrator.starvation_state import CapBaseline, WatchdogState
from orchestrator.starvation_watchdog import (
    AUTO_PIN_CAP_SENTINEL,
    STARVATION_WATCHDOG_MARKER,
    STARVATION_WATCHDOG_ROLE,
    StarvationGate,
)

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)
TTL = timedelta(seconds=300)
REFUSED, DUAL = StarvationGate.LOCK_REFUSED, StarvationGate.DUAL
CAP = 3


def _live() -> dict[str, Any]:
    return {
        'status': 'in-progress', 'claimant_run_id': 'run-x',
        'heartbeat_at': (NOW - timedelta(seconds=30)).isoformat(),
    }


def _stale() -> dict[str, Any]:
    return {'status': 'in-progress', 'claimant_run_id': None, 'heartbeat_at': None}


def _pending() -> dict[str, Any]:
    return {'status': 'pending'}


def _held(module: str, owner: str) -> Blocker:
    return Blocker(module=module, owner=owner, kind=BlockerKind.HELD)


def _world(
    tasks: Mapping[str, Mapping[str, Any]], blockers: Mapping[str, Sequence[Blocker]],
) -> LockWorld:
    return LockWorld(
        tasks=tasks,
        blockers_of=lambda tid: tuple(blockers.get(tid, ())),
        now=NOW,
        claimant_ttl=TTL,
    )


def _report(
    waiting: Mapping[str, StarvationGate],
    blockers: Mapping[str, Sequence[Blocker]],
    holders: Mapping[str, Mapping[str, Any]],
) -> CapReport:
    tasks = {tid: _pending() for tid in waiting} | dict(holders)
    return cap_report(waiting, _world(tasks, blockers), CAP)


def _wedged(*members: str, module: str = 'hot/d.py', holder: str = '0') -> CapReport:
    """Every member waits on *module*, held by a holder with no live claimant."""
    return _report(
        {m: REFUSED for m in members},
        {m: [_held(module, holder)] for m in members},
        {holder: _stale()},
    )


def _all_live(*members: str) -> CapReport:
    return _report(
        {m: REFUSED for m in members},
        {m: [_held('hot/d.py', '0')] for m in members},
        {'0': _live()},
    )


def _cohort_of(size: int, module: str = 'hot/d.py') -> CapReport:
    return _wedged(*(f'T{i:04d}' for i in range(size)), module=module)


class TestCapReport:
    def test_members_carry_gate_and_first_blocker_module(self) -> None:
        report = _report(
            {'S4': DUAL, 'S3': REFUSED, 'S5': REFUSED},
            {
                'S3': [_held('hot/d.py', '0'), _held('hot/e.py', '0')],
                'S4': [_held('hot/e.py', '0')],
            },
            {'0': _live()},
        )

        assert report.members == (
            CohortMember('S3', REFUSED, 'hot/d.py'),
            CohortMember('S4', DUAL, 'hot/e.py'),
            CohortMember('S5', REFUSED, None),
        )
        assert report.cap == CAP
        assert report.modules == frozenset({'hot/d.py', 'hot/e.py'})

    def test_a_non_live_holder_on_the_second_blocker_is_a_wedge(self) -> None:
        second = _held('hot/e.py', 'W')
        report = _report(
            {'S3': REFUSED},
            {'S3': [_held('hot/d.py', 'L'), second]},
            {'L': _live(), 'W': _stale()},
        )

        assert report.alarming
        assert [w.holder for w in report.wedges] == [
            NonLiveHolder(second, HolderState.NO_LIVE_CLAIMANT),
        ]
        assert report.subjects == frozenset({'W'})

    def test_a_non_live_holder_is_listed_once_with_every_waiter(self) -> None:
        report = _wedged('S3', 'S4', 'S5')

        assert len(report.wedges) == 1
        assert report.wedges[0].waiters == ('S3', 'S4', 'S5')
        assert report.subjects == frozenset({'0'})

    def test_all_holders_live_is_not_alarming(self) -> None:
        report = _all_live('S3', 'S4')

        assert report.wedges == ()
        assert not report.alarming
        assert report.subjects == frozenset()

    def test_a_cohort_with_no_recorded_blocker_is_alarming_about_its_members(self) -> None:
        report = _report({'S3': REFUSED, 'S4': DUAL}, {}, {})

        assert report.unexplained
        assert report.alarming
        assert report.subjects == frozenset({'S3', 'S4'})

    def test_one_member_without_blockers_among_live_holders_is_not_alarming(self) -> None:
        report = _report(
            {'S3': REFUSED, 'S4': REFUSED},
            {'S3': [_held('hot/d.py', '0')]},
            {'0': _live()},
        )

        assert not report.unexplained
        assert not report.alarming

    def test_an_empty_cohort_is_not_alarming(self) -> None:
        report = _report({}, {}, {})

        assert report.members == ()
        assert not report.alarming


class TestChangeRule:
    BASE = _wedged('S3', 'S4').announced()

    def test_the_announcement_is_size_modules_and_subjects(self) -> None:
        assert self.BASE == CapBaseline(
            size=2, modules=frozenset({'hot/d.py'}), subjects=frozenset({'0'}),
        )

    def test_a_new_non_live_subject_is_news(self) -> None:
        report = _report(
            {'S3': REFUSED, 'S4': REFUSED},
            {'S3': [_held('hot/d.py', '0')], 'S4': [_held('hot/d.py', '9')]},
            {'0': _stale(), '9': _stale()},
        )

        assert is_news(report, self.BASE)
        assert cap_action(report, self.BASE, None) is CapAction.FILE

    def test_a_new_first_blocker_module_with_a_wedge_is_news(self) -> None:
        report = _report(
            {'S3': REFUSED, 'S5': REFUSED},
            {'S3': [_held('hot/d.py', '0')], 'S5': [_held('hot/e.py', '0')]},
            {'0': _stale()},
        )

        assert cap_action(report, self.BASE, None) is CapAction.FILE

    def test_a_standing_cohort_files_nothing(self) -> None:
        report = _wedged('S3', 'S4', 'S6')

        assert not is_news(report, self.BASE)
        assert cap_action(report, self.BASE, None) is CapAction.NONE

    @pytest.mark.parametrize(('base_size', 'news_size'), [(290, 435), (4, 14), (0, 10)])
    def test_growth_reaching_the_bound_is_news(self, base_size: int, news_size: int) -> None:
        base = CapBaseline(
            size=base_size, modules=frozenset({'hot/d.py'}), subjects=frozenset({'0'}),
        )

        assert cap_action(_cohort_of(news_size - 1), base, None) is CapAction.NONE
        assert cap_action(_cohort_of(news_size), base, None) is CapAction.FILE

    def test_news_that_is_not_alarming_files_nothing(self) -> None:
        assert cap_action(_all_live('S3', 'S4', 'S5'), CapBaseline.EMPTY, None) is CapAction.NONE

    def test_an_open_alarm_whose_rendering_changed_is_refreshed(self) -> None:
        _, open_detail = cap_text(_wedged('S3'))

        assert cap_action(_wedged('S3', 'S4'), self.BASE, open_detail) is CapAction.REFRESH

    def test_an_open_alarm_whose_rendering_is_unchanged_is_left(self) -> None:
        report = _wedged('S3', 'S4')
        _, open_detail = cap_text(report)

        assert cap_action(report, self.BASE, open_detail) is CapAction.NONE

    @pytest.mark.parametrize('report', [_all_live('S3', 'S4'), _report({}, {}, {})],
                             ids=['all-live', 'empty'])
    def test_an_open_alarm_that_is_no_longer_alarming_is_resolved(self, report: CapReport) -> None:
        assert cap_action(report, self.BASE, 'any open detail') is CapAction.RESOLVE


class TestNextBaseline:
    BASE = CapBaseline(size=7, modules=frozenset({'hot/x.py'}), subjects=frozenset({'X'}))

    @pytest.mark.parametrize('action', [CapAction.FILE, CapAction.REFRESH])
    def test_filing_or_refreshing_records_the_report(self, action: CapAction) -> None:
        report = _wedged('S3', 'S4')

        assert next_baseline(action, report, self.BASE) == report.announced()

    @pytest.mark.parametrize('action', [CapAction.RESOLVE, CapAction.NONE])
    def test_an_empty_cohort_ends_the_episode(self, action: CapAction) -> None:
        assert next_baseline(action, _report({}, {}, {}), self.BASE) == CapBaseline.EMPTY

    def test_resolving_an_all_live_cohort_keeps_the_baseline(self) -> None:
        assert next_baseline(CapAction.RESOLVE, _all_live('S3'), self.BASE) == self.BASE

    def test_leaving_a_standing_cohort_keeps_the_baseline(self) -> None:
        assert next_baseline(CapAction.NONE, _wedged('S3'), self.BASE) == self.BASE


class TestCapText:
    def test_summary_names_the_non_live_holder(self) -> None:
        summary, _ = cap_text(_wedged('S3', 'S4', holder='5519'))

        assert summary.startswith(STARVATION_WATCHDOG_MARKER)
        assert len(summary) <= 200
        assert '5519' in summary

    def test_summary_of_a_large_cohort_is_bounded(self) -> None:
        holders = {f'H{i}': _stale() for i in range(60)}
        report = _report(
            {f'S{i}': REFUSED for i in range(60)},
            {f'S{i}': [_held(f'hot/{i}.py', f'H{i}')] for i in range(60)},
            holders,
        )
        summary, _ = cap_text(report)

        assert summary.startswith(STARVATION_WATCHDOG_MARKER)
        assert len(summary) <= 200

    def test_detail_names_members_by_module_holders_with_reasons_and_the_cap(self) -> None:
        report = _report(
            {'S3': REFUSED, 'S4': DUAL, 'S5': REFUSED},
            {'S3': [_held('hot/d.py', '0')], 'S4': [_held('hot/e.py', '7')]},
            {'0': _stale(), '7': {'status': 'blocked', 'claimant_run_id': 'r'}},
        )
        _, detail = cap_text(report)

        for name in ('S3', 'S4', 'S5', 'hot/d.py', 'hot/e.py', 'task 0', 'task 7'):
            assert name in detail
        assert HolderState.NO_LIVE_CLAIMANT.value in detail
        assert HolderState.BLOCKED.value in detail
        assert f'auto_pin_max_concurrent={CAP}' in detail
        d_section = detail.index('hot/d.py:')
        e_section = detail.index('hot/e.py:')
        assert d_section < detail.index('S3', d_section) < e_section < detail.index('S4', e_section)

    def test_an_unexplained_cohort_says_so(self) -> None:
        summary, detail = cap_text(_report({'S3': REFUSED}, {}, {}))

        assert summary.startswith(STARVATION_WATCHDOG_MARKER)
        assert 'S3' in detail
        assert 'no recorded blocker' in detail

    def test_the_text_carries_no_ages_or_counts(self) -> None:
        _, detail = cap_text(_wedged('S3'))

        for volatile in ('eligible_age_secs', 'skip_count', 'lock_refusals'):
            assert volatile not in detail


def _queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


def _open_cap_l1s(queue: EscalationQueue) -> list[Escalation]:
    return queue.get_by_task(AUTO_PIN_CAP_SENTINEL, status='pending', level=1)


def _pre_deploy_cap_l1(queue: EscalationQueue) -> Escalation:
    esc = Escalation(
        id=queue.make_id(AUTO_PIN_CAP_SENTINEL),
        task_id=AUTO_PIN_CAP_SENTINEL,
        agent_role='orchestrator-scheduler',
        severity='blocking',
        category='risk_identified',
        summary=f'{STARVATION_WATCHDOG_MARKER}: 290 starving task(s) not auto-pinned',
        detail='the old standing-backlog text',
        level=1,
    )
    queue.submit(esc)
    return esc


class _RefusingQueue(EscalationQueue):
    def submit(self, escalation: Escalation) -> str:
        raise OSError('disk full')


class TestSyncCapL1:
    def test_file_submits_one_l1_and_records_the_baseline(self, tmp_path: Path) -> None:
        queue, state = _queue(tmp_path), WatchdogState(None, '/proj')
        report = _wedged('S3', 'S4')

        assert sync_cap_l1(queue, report, state)

        [esc] = _open_cap_l1s(queue)
        assert esc.agent_role == STARVATION_WATCHDOG_ROLE
        assert esc.level == 1
        assert (esc.summary, esc.detail) == cap_text(report)
        assert state.baseline() == report.announced()

    def test_refresh_keeps_the_record_and_rewrites_its_detail(self, tmp_path: Path) -> None:
        queue, state = _queue(tmp_path), WatchdogState(None, '/proj')
        sync_cap_l1(queue, _wedged('S3'), state)
        [filed] = _open_cap_l1s(queue)
        grown = _wedged('S3', 'S4')

        assert sync_cap_l1(queue, grown, state)

        [refreshed] = _open_cap_l1s(queue)
        assert refreshed.id == filed.id
        assert refreshed.detail == cap_text(grown)[1]
        assert refreshed.dedupe_count == filed.dedupe_count + 1
        assert state.baseline() == grown.announced()

    def test_resolve_closes_every_open_cap_l1(self, tmp_path: Path) -> None:
        queue, state = _queue(tmp_path), WatchdogState(None, '/proj')
        sync_cap_l1(queue, _wedged('S3'), state)
        [filed] = _open_cap_l1s(queue)

        assert sync_cap_l1(queue, _all_live('S3'), state)

        assert _open_cap_l1s(queue) == []
        closed = queue.get(filed.id)
        assert closed is not None
        assert closed.status == 'resolved'
        assert closed.resolved_by == STARVATION_WATCHDOG_ROLE

    def test_a_pre_deploy_cap_l1_is_refreshed_not_duplicated(self, tmp_path: Path) -> None:
        queue, state = _queue(tmp_path), WatchdogState(None, '/proj')
        old = _pre_deploy_cap_l1(queue)

        assert sync_cap_l1(queue, _wedged('S3'), state)

        [esc] = _open_cap_l1s(queue)
        assert esc.id == old.id
        assert esc.detail == cap_text(_wedged('S3'))[1]

    def test_a_pre_deploy_cap_l1_is_resolved_once_nothing_alarms(self, tmp_path: Path) -> None:
        queue, state = _queue(tmp_path), WatchdogState(None, '/proj')
        old = _pre_deploy_cap_l1(queue)

        assert sync_cap_l1(queue, _all_live('S3'), state)

        assert _open_cap_l1s(queue) == []
        closed = queue.get(old.id)
        assert closed is not None
        assert closed.resolved_by == STARVATION_WATCHDOG_ROLE

    def test_a_human_close_of_a_standing_cohort_files_nothing(self, tmp_path: Path) -> None:
        queue, state = _queue(tmp_path), WatchdogState(None, '/proj')
        sync_cap_l1(queue, _wedged('S3', 'S4'), state)
        [filed] = _open_cap_l1s(queue)
        queue.resolve(filed.id, 'wait', resolved_by='human')

        assert sync_cap_l1(queue, _wedged('S3', 'S4', 'S6'), state)

        assert _open_cap_l1s(queue) == []

    def test_no_queue_records_nothing(self) -> None:
        state = WatchdogState(None, '/proj')

        assert sync_cap_l1(None, _wedged('S3'), state)

        assert state.baseline() == CapBaseline.EMPTY

    def test_a_failed_filing_leaves_the_baseline_unrecorded(self, tmp_path: Path) -> None:
        state = WatchdogState(None, '/proj')

        assert not sync_cap_l1(_RefusingQueue(tmp_path / 'esc'), _wedged('S3'), state)

        assert state.baseline() == CapBaseline.EMPTY
