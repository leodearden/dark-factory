"""The early-start selection at the Scheduler boundary (early start, PRD sections 5.1-5.4).

``plans/early-start-on-predecessor-branch-prd.md``, task T5a: step (a), the
shortlist of dependents that qualify to start on a waiting predecessor's
branch, and step (c), the uninterrupted claim. P is dispatched by the ordinary
pass first, so it really holds its keys, and is then offered as in-progress.

The Scheduler runs on the code defaults: pydantic-settings would otherwise merge
the operational yaml's own ``module_overrides`` into the test's holder limits.
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _orch_helpers import HermeticMcpSession
from _park_test_helpers import make_task
from shared.psi import PsiSample

from orchestrator.config import (
    EarlyStartConfig,
    OrchestratorConfig,
    PsiAdmissionConfig,
    WarmBaseHardDownConfig,
)
from orchestrator.delivered_checks import ChecksAtCommit
from orchestrator.dispatch_row import FreshRow
from orchestrator.early_start_selection import (
    DependentBranch,
    EarlyStartHolds,
    EarlyStartRefusal,
    EarlyStartRefusalReason,
    EarlyStartSelection,
    EarlyStartShortlist,
    PairReads,
    StepReads,
)
from orchestrator.eligibility_clock import EligibilityClockStore
from orchestrator.event_store import EventStore, EventType
from orchestrator.overrides import OverrideStore
from orchestrator.pause_policy import PauseClass
from orchestrator.pin_reservation import Blocker, BlockerKind
from orchestrator.scheduler import Scheduler, SchedulerCallbacks
from orchestrator.starvation_state import StarvationStateStore
from orchestrator.task_picture import TaskPicture

pytestmark = [pytest.mark.usefixtures('code_default_config'), pytest.mark.asyncio]

FIXED_DT = datetime(2026, 11, 22, 0, 0, 0, tzinfo=UTC)
P_FILES = ['a.py', 'b.py']
D_FILES = ['a.py', 'c.py']
TIP = 'a' * 40

Reason = EarlyStartRefusalReason


class _RecordingSession(HermeticMcpSession):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    async def call_tool(self, name: str, arguments: dict, timeout: float = 30) -> dict:
        self.calls.append(name)
        return await super().call_tool(name, arguments, timeout)


@dataclass(frozen=True)
class _Rig:
    scheduler: Scheduler
    store: EventStore
    overrides: OverrideStore
    session: _RecordingSession
    status: AsyncMock
    project_root: str

    def offer(self, *tasks: dict) -> None:
        self.scheduler.get_tasks = AsyncMock(return_value=[dict(t) for t in tasks])

    async def picture(self, *tasks: dict) -> TaskPicture:
        self.offer(*tasks)
        picture = await self.scheduler.build_task_picture()
        assert picture is not None
        return picture

    async def acquire(self) -> str:
        """The task the ordinary pass starts next; it must start one."""
        assignment = await self.scheduler.acquire_next()
        assert assignment is not None
        return assignment.task_id

    async def dispatch(self, task: dict) -> None:
        """Dispatch *task* through the ordinary pass, so it really holds its keys."""
        self.offer(task)
        assignment = await self.scheduler.acquire_next()
        assert assignment is not None and assignment.task_id == task['id']

    def shortlist(self, picture: TaskPicture, *waiting: str) -> EarlyStartShortlist:
        return self.scheduler.early_start_shortlist(picture, waiting_predecessors=set(waiting or ('P',)))

    async def select(
        self, picture: TaskPicture, reads: StepReads, *,
        waiting: frozenset[str] = frozenset({'P'}), holds: EarlyStartHolds | None = None,
    ) -> EarlyStartSelection:
        return await self.scheduler.select_early_start(
            picture, reads, waiting, clear_holds() if holds is None else holds,
        )

    def events(self) -> list[dict]:
        rows = [row for event_type in EventType for row in self.store.fetch_events_by_type(event_type)]
        return sorted(rows, key=lambda row: row['id'])

    def assert_untouched(self, *task_ids: str) -> None:
        for tid in task_ids:
            assert not self.scheduler.is_dispatched(tid), tid
            assert not self.scheduler.lock_table.is_held(tid), tid
            assert self.scheduler.lock_table.successor_relation(tid) is None, tid
        assert self.scheduler.lock_table.snapshot_successor_of() == {}


def _rig(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
    callbacks: SchedulerCallbacks | None = None,
    early_start: EarlyStartConfig | None = None,
    psi_admission: PsiAdmissionConfig | None = None,
    warm_base_hard_down: WarmBaseHardDownConfig | None = None,
    announced_starving: tuple[str, ...] = (),
) -> _Rig:
    """*announced_starving* are tasks a previous process announced to the starvation watchdog."""
    config = OrchestratorConfig(
        max_per_module=1, lock_depth=12, module_overrides={'two.py': 2},
        early_start=EarlyStartConfig(enabled=True) if early_start is None else early_start,
        psi_admission=PsiAdmissionConfig() if psi_admission is None else psi_admission,
        warm_base_hard_down=(
            WarmBaseHardDownConfig() if warm_base_hard_down is None else warm_base_hard_down
        ),
        project_root=tmp_path,
    )
    store = EventStore(tmp_path / 'runs.db', 'run-1')
    overrides = OverrideStore(tmp_path / 'overrides.db')
    session = _RecordingSession()
    clock_store = EligibilityClockStore(tmp_path / 'eligibility.db') if announced_starving else None
    for tid in announced_starving:
        StarvationStateStore(tmp_path / 'eligibility.db').announce(str(config.project_root), tid)
    scheduler = Scheduler(
        config,
        store,
        mcp_session=session,
        wall_time_source=lambda: FIXED_DT,
        override_store=overrides,
        state_snapshot_path=tmp_path / 'scheduler_state.json',
        branch_work_reader=dict,
        callbacks=callbacks,
        eligibility_clock_store=clock_store,
    )
    scheduler.finish_startup()
    status = AsyncMock()
    monkeypatch.setattr(scheduler, 'set_task_status', status)
    return _Rig(scheduler, store, overrides, session, status, str(config.project_root))


def task(
    tid: str, priority: str = 'medium', files: list[str] | None = None, *,
    status: str = 'pending', deps: tuple[str, ...] = (), **metadata: object,
) -> dict:
    row = make_task(tid, priority, D_FILES if files is None else files, status=status)
    row['dependencies'] = list(deps)
    row['metadata'].update(metadata)
    return row


def p_running(files: list[str] | None = None) -> dict:
    return task('P', files=P_FILES if files is None else files, status='in-progress')


async def _with_p_running(tmp_path, monkeypatch, **rig_overrides) -> _Rig:
    rig = _rig(tmp_path, monkeypatch, **rig_overrides)
    await rig.dispatch(task('P', files=P_FILES))
    return rig


def _refusal(reason: Reason, dependent: str = 'D', predecessor: str = 'P') -> EarlyStartRefusal:
    return EarlyStartRefusal(reason, dependent=dependent, predecessor=predecessor)


def clear_holds() -> EarlyStartHolds:
    return EarlyStartHolds(
        review_checkpoint=False, merge_queue_halted=False, usage_paused=False,
        cost_ceiling_breached=False,
    )


def pair_reads(
    picture: TaskPicture, dependent: str = 'D', predecessor: str = 'P', *,
    tip: str = TIP, checks: ChecksAtCommit = ChecksAtCommit.PASSED,
    branch: DependentBranch = DependentBranch.ABSENT, row: dict | None = None,
    dep_statuses: dict[str, str] | None = None,
) -> tuple[str, PairReads]:
    fresh = FreshRow(
        task=picture.tasks_by_id[dependent] if row is None else row,
        dep_statuses=dep_statuses or {},
    )
    return dependent, PairReads(
        predecessor=predecessor, predecessor_tip=tip, predecessor_checks=checks,
        dependent_branch=branch, dependent_row=fresh,
    )


def reads_of(*pairs: tuple[str, PairReads], statuses: dict[str, str] | None = None) -> StepReads:
    return StepReads(statuses=statuses or {}, pairs=dict(pairs))


def reads_for(picture: TaskPicture, dependent: str = 'D', predecessor: str = 'P', **kw) -> StepReads:
    statuses = kw.pop('statuses', None)
    return reads_of(pair_reads(picture, dependent, predecessor, **kw), statuses=statuses)


def _acquired_rows(store: EventStore, task_id: str) -> list[dict]:
    return [
        event['data'] for event in store.fetch_events_by_type(EventType.lock_acquired)
        if event['task_id'] == task_id
    ]


def _held(module: str, *owners: str) -> list[Blocker]:
    return [Blocker(module, owner, BlockerKind.HELD) for owner in owners]


class TestShortlistQualifyingDependent:
    async def test_a_pending_dependent_of_only_the_predecessor_is_the_one_candidate(
        self, tmp_path, monkeypatch,
    ):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(p_running(), task('D', deps=('P',)))

        shortlist = rig.shortlist(picture)

        (candidate,) = shortlist.candidates
        assert (candidate.dependent, candidate.predecessor) == ('D', 'P')
        assert candidate.keys == ('a.py', 'c.py')
        assert candidate.priority == picture.priority_of('D')
        assert candidate.task['id'] == 'D'
        assert shortlist.refusals == ()

    async def test_a_dependent_whose_other_dependency_is_done_still_qualifies(
        self, tmp_path, monkeypatch,
    ):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(
            p_running(), task('Q', files=['q.py'], status='done'), task('D', deps=('P', 'Q')),
        )

        shortlist = rig.shortlist(picture)

        assert [c.dependent for c in shortlist.candidates] == ['D']
        assert shortlist.refusals == ()

    async def test_a_task_waiting_on_no_waiting_predecessor_is_neither_candidate_nor_refusal(
        self, tmp_path, monkeypatch,
    ):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(
            p_running(), task('Q', files=['q.py'], status='in-progress'), task('E', deps=('Q',)),
        )

        assert rig.shortlist(picture) == EarlyStartShortlist(candidates=(), refusals=())

    async def test_the_ordinary_pass_still_does_not_start_the_dependent(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(p_running(), task('D', deps=('P',)))
        assert [c.dependent for c in rig.shortlist(picture).candidates] == ['D']

        assert await rig.scheduler.acquire_next() is None
        assert not rig.scheduler.is_dispatched('D')


class TestShortlistRefusals:
    async def test_row_9_a_human_gate_is_not_a_code_task(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        gate = task('D', deps=('P',), task_kind='deterministic', always_escalates=True)
        picture = await rig.picture(p_running(), gate)

        shortlist = rig.shortlist(picture)

        assert shortlist.candidates == ()
        assert shortlist.refusals == (_refusal(Reason.NOT_CODE_TASK),)

    async def test_row_9_a_milestone_task_is_refused(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        fired = {'mode': 'dated', 'at': '2026-11-01T00:00:00+00:00'}
        picture = await rig.picture(p_running(), task('D', deps=('P',), milestone=fired))

        shortlist = rig.shortlist(picture)

        assert shortlist.candidates == ()
        assert shortlist.refusals == (_refusal(Reason.HAS_MILESTONE),)

    async def test_another_unfinished_dependency_is_not_only_dependency(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(
            p_running(), task('Q', files=['q.py'], status='in-progress'), task('D', deps=('P', 'Q')),
        )

        shortlist = rig.shortlist(picture)

        assert shortlist.candidates == ()
        assert shortlist.refusals == (_refusal(Reason.NOT_ONLY_DEPENDENCY),)

    async def test_a_dependent_in_requeue_cooldown_is_not_eligible(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        rig.scheduler.release('D', requeued=True)
        picture = await rig.picture(p_running(), task('D', deps=('P',)))

        shortlist = rig.shortlist(picture)

        assert shortlist.candidates == ()
        assert shortlist.refusals == (_refusal(Reason.NOT_ELIGIBLE),)

    async def test_a_predecessor_with_a_successor_refuses_every_dependent(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        assert rig.scheduler.acquire_as_successor('D0', ['z.py'], predecessor='P', priority='medium')
        picture = await rig.picture(
            p_running(),
            task('D0', files=['z.py'], status='in-progress', deps=('P',)),
            task('D', deps=('P',)),
            task('D3', files=['d3.py'], deps=('P',)),
        )

        shortlist = rig.shortlist(picture)

        assert shortlist.candidates == ()
        assert set(shortlist.refusals) == {
            _refusal(Reason.PREDECESSOR_HAS_DEPENDENT, 'D'),
            _refusal(Reason.PREDECESSOR_HAS_DEPENDENT, 'D3'),
        }

    async def test_a_predecessor_that_is_itself_a_successor_refuses_its_dependents(
        self, tmp_path, monkeypatch,
    ):
        rig = _rig(tmp_path, monkeypatch)
        await rig.dispatch(task('P0', files=['p0.py']))
        assert rig.scheduler.acquire_as_successor('P', P_FILES, predecessor='P0', priority='medium')
        picture = await rig.picture(
            task('P0', files=['p0.py'], status='in-progress'),
            task('P', files=P_FILES, status='in-progress', deps=('P0',)),
            task('D', deps=('P',)),
        )

        shortlist = rig.shortlist(picture)

        assert shortlist.candidates == ()
        assert shortlist.refusals == (_refusal(Reason.PREDECESSOR_HAS_DEPENDENT),)


class TestShortlistRanking:
    """Row 46 set-up: candidates come in the one tier order, then id order."""

    async def _ranked(self, rig: _Rig, *dependents: dict) -> list[str]:
        picture = await rig.picture(p_running(), *dependents)
        return [c.dependent for c in rig.shortlist(picture).candidates]

    async def test_the_higher_tier_comes_first(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)

        ranked = await self._ranked(rig, task('D2', 'medium', deps=('P',)), task('D1', 'high', deps=('P',)))

        assert ranked == ['D1', 'D2']

    async def test_between_equal_tiers_the_pinned_one_comes_first(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        rig.overrides.set_override(rig.project_root, 'D2', pinned=True, pin_order=1)

        ranked = await self._ranked(rig, task('D1', deps=('P',)), task('D2', deps=('P',)))

        assert ranked == ['D2', 'D1']

    async def test_between_two_pins_the_lower_pin_order_comes_first(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        rig.overrides.set_override(rig.project_root, 'D1', pinned=True, pin_order=2)
        rig.overrides.set_override(rig.project_root, 'D2', pinned=True, pin_order=1)

        ranked = await self._ranked(rig, task('D1', deps=('P',)), task('D2', deps=('P',)))

        assert ranked == ['D2', 'D1']

    async def test_then_numeric_id_order(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)

        ranked = await self._ranked(rig, task('12', deps=('P',)), task('9', deps=('P',)))

        assert ranked == ['9', '12']


class TestClaim:
    async def test_a_qualifying_pair_is_claimed_with_an_ordinary_starts_bookkeeping(
        self, tmp_path, monkeypatch,
    ):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(
            p_running(), task('D', deps=('P',)), task('H', 'high', ['b.py']),
        )

        selection = await rig.select(picture, reads_for(picture))

        claim = selection.claim
        assert claim is not None and selection.refusals == ()
        assert (claim.dependent, claim.predecessor) == ('D', 'P')
        assert claim.keys == ('a.py', 'c.py')
        assert claim.stacked_commit == TIP
        assert claim.tier == 'medium'
        assert claim.task == picture.tasks_by_id['D']
        scheduler = rig.scheduler
        assert scheduler.is_dispatched('D')
        relation = scheduler.lock_table.successor_relation('D')
        assert relation is not None and relation.predecessor == 'P'
        assert scheduler.lock_table.held_modules('D') == {'a.py', 'c.py'}
        assert scheduler.lock_table.held_modules('P') == set(P_FILES)
        assert _acquired_rows(rig.store, 'D') == [
            {'modules': ['a.py', 'c.py'], 'priority': 'medium', 'successor_of': 'P'},
        ]
        assert scheduler.get_state_snapshot().successor_of == {'D': 'P'}
        assert rig.store.fetch_events_by_type(EventType.task_skipped) == []

    async def test_the_claim_carries_the_store_row(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(p_running(), task('D', deps=('P',)))
        fresh = task('D', deps=('P',), stamped_since_the_picture=True)

        selection = await rig.select(picture, reads_for(picture, row=fresh))

        assert selection.claim is not None
        assert selection.claim.task['metadata']['stamped_since_the_picture'] is True

    @pytest.mark.parametrize(('fresh', 'dep_statuses', 'reason'), [
        (task('D', deps=('P', 'X')), {'X': 'pending'}, Reason.NOT_ONLY_DEPENDENCY),
        (
            task('D', deps=('P',), milestone={'mode': 'dated', 'at': '2026-11-01T00:00:00+00:00'}),
            {}, Reason.HAS_MILESTONE,
        ),
        (task('D', deps=('P',), status='blocked'), {}, Reason.NOT_ELIGIBLE),
    ], ids=['new-dependency', 'milestone', 'blocked'])
    async def test_the_claim_judges_the_store_row_not_the_pictures(
        self, tmp_path, monkeypatch, fresh, dep_statuses, reason,
    ):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(p_running(), task('D', deps=('P',)))
        assert [c.dependent for c in rig.shortlist(picture).candidates] == ['D']

        selection = await rig.select(picture, reads_for(picture, row=fresh, dep_statuses=dep_statuses))

        assert selection == EarlyStartSelection(claim=None, refusals=(_refusal(reason),))
        rig.assert_untouched('D')

    async def test_reads_keyed_by_another_task_than_their_row_are_refused(
        self, tmp_path, monkeypatch,
    ):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(p_running(), task('D', deps=('P',)), task('E', deps=('P',)))
        _, pair = pair_reads(picture, 'E')

        with pytest.raises(ValueError, match=r"(?s)(?=.*'D')(?=.*'E')"):
            StepReads(statuses={}, pairs={'D': pair})

    async def test_an_unread_pair_is_neither_claimed_nor_refused(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(p_running(), task('D', deps=('P',)))

        assert await rig.select(picture, reads_of()) == EarlyStartSelection(None, ())
        rig.assert_untouched('D')

    async def test_reads_for_another_predecessor_are_not_this_pairs(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(p_running(), task('D', deps=('P',)))

        selection = await rig.select(picture, reads_for(picture, predecessor='Q'))

        assert selection == EarlyStartSelection(None, ())
        rig.assert_untouched('D')

    async def test_a_predecessor_no_longer_waiting_is_not_claimed_on(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(p_running(), task('D', deps=('P',)))

        selection = await rig.select(picture, reads_for(picture), waiting=frozenset())

        assert selection.claim is None
        rig.assert_untouched('D')

    async def test_a_predecessor_the_reads_report_done_leaves_its_dependents_unrefused(
        self, tmp_path, monkeypatch,
    ):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(p_running(), task('D', deps=('P',)), task('E', deps=('P',)))

        selection = await rig.select(picture, reads_for(picture, dep_statuses={'P': 'done'}))

        assert selection == EarlyStartSelection(None, ())
        rig.assert_untouched('D', 'E')

    async def test_a_claim_that_claims_nobody_writes_nothing(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(p_running(), task('D', deps=('P', 'X')))
        events_before, calls_before = rig.events(), list(rig.session.calls)

        selection = await rig.select(picture, reads_for(picture, dep_statuses={'X': 'pending'}))

        assert selection.claim is None
        assert rig.events() == events_before
        assert rig.session.calls == calls_before
        rig.status.assert_not_awaited()
        assert rig.scheduler.is_dispatched('P')
        rig.assert_untouched('D')


class TestTheSelectionFixesNoTasksKeys:
    """A task's lock keys are derived from its files at its own dispatch, not when a selection weighed it."""

    async def test_tasks_a_selection_weighed_dispatch_on_their_later_files(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(
            p_running(), task('E', files=['e_old.py'], deps=('P',)), task('T', files=['t_old.py']),
        )
        assert [c.dependent for c in rig.shortlist(picture).candidates] == ['E']
        assert await rig.select(picture, reads_of()) == EarlyStartSelection(None, ())

        rig.offer(
            task('P', files=P_FILES, status='done'),
            task('E', files=['e_new.py'], deps=('P',)),
            task('T', files=['t_new.py']),
        )
        assert {await rig.acquire(), await rig.acquire()} == {'E', 'T'}

        assert rig.scheduler.lock_table.held_modules('E') == {'e_new.py'}
        assert rig.scheduler.lock_table.held_modules('T') == {'t_new.py'}


class TestRow7Keys:
    async def test_a_two_holder_key_already_full_is_key_unavailable(self, tmp_path, monkeypatch):
        rig = _rig(tmp_path, monkeypatch)
        await rig.dispatch(task('P', files=['two.py']))
        rig.offer(task('P', files=['two.py'], status='in-progress'), task('U', files=['two.py']))
        assert await rig.acquire() == 'U'
        picture = await rig.picture(
            task('P', files=['two.py'], status='in-progress'),
            task('U', files=['two.py'], status='in-progress'),
            task('D', files=['two.py'], deps=('P',)),
        )

        selection = await rig.select(picture, reads_for(picture))

        assert selection == EarlyStartSelection(None, (_refusal(Reason.KEY_UNAVAILABLE),))
        rig.assert_untouched('D')

    async def test_a_two_holder_key_takes_the_successor_then_refuses_a_third(
        self, tmp_path, monkeypatch,
    ):
        rig = _rig(tmp_path, monkeypatch)
        await rig.dispatch(task('P', files=['two.py']))
        picture = await rig.picture(
            task('P', files=['two.py'], status='in-progress'),
            task('D', files=['two.py'], deps=('P',)),
        )

        assert (await rig.select(picture, reads_for(picture))).claim is not None

        rig.offer(
            task('P', files=['two.py'], status='in-progress'),
            task('D', files=['two.py'], status='in-progress', deps=('P',)),
            task('X', 'high', ['two.py']),
        )
        assert await rig.scheduler.acquire_next() is None
        assert rig.scheduler.lock_table.blockers('X', ['two.py']) == _held('two.py', 'D', 'P')

    async def test_a_foreign_reservation_is_key_reserved(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        rig.scheduler.lock_table.install_parks('Z', ['c.py'], 'low')
        picture = await rig.picture(
            p_running(), task('D', deps=('P',)), task('Z', 'low', ['c.py']),
        )

        selection = await rig.select(picture, reads_for(picture))

        assert selection.claim is None
        assert _refusal(Reason.KEY_RESERVED) in selection.refusals
        rig.assert_untouched('D')


class TestRow46OneDependentPerPredecessor:
    def _dependents(self) -> tuple[dict, dict]:
        return task('D1', 'high', ['a.py', 'c.py'], deps=('P',)), task('D2', 'medium', ['b.py', 'd.py'], deps=('P',))

    async def test_the_higher_ranked_is_claimed_and_the_other_refused(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        d1, d2 = self._dependents()
        picture = await rig.picture(p_running(), d2, d1)

        selection = await rig.select(picture, reads_of(pair_reads(picture, 'D1'), pair_reads(picture, 'D2')))

        assert selection.claim is not None and selection.claim.dependent == 'D1'
        assert selection.refusals == (_refusal(Reason.PREDECESSOR_HAS_DEPENDENT, 'D2'),)

        later = await rig.select(picture, reads_of(pair_reads(picture, 'D2')))
        assert later.claim is None
        assert _refusal(Reason.PREDECESSOR_HAS_DEPENDENT, 'D2') in later.refusals

        assert rig.scheduler.release_early_start_claim(selection.claim)
        after = await rig.select(picture, reads_of(pair_reads(picture, 'D2')))
        assert after.claim is not None and after.claim.dependent == 'D2'


class TestReleasePath:
    async def _claimed(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(p_running(), task('D', deps=('P',)))
        selection = await rig.select(picture, reads_for(picture))
        assert selection.claim is not None
        return rig, selection.claim

    async def test_releasing_a_live_claim_drops_keys_relation_and_started_mark(
        self, tmp_path, monkeypatch,
    ):
        rig, claim = await self._claimed(tmp_path, monkeypatch)

        assert rig.scheduler.release_early_start_claim(claim) is True

        rig.assert_untouched('D')
        assert rig.scheduler.lock_table.held_modules('P') == set(P_FILES)
        released = [e for e in rig.store.fetch_events_by_type(EventType.lock_released) if e['task_id'] == 'D']
        assert len(released) == 1

    async def test_a_second_release_or_a_stale_claim_changes_nothing(self, tmp_path, monkeypatch):
        rig, claim = await self._claimed(tmp_path, monkeypatch)
        assert rig.scheduler.release_early_start_claim(dataclasses.replace(claim, predecessor='Q')) is False
        assert rig.scheduler.is_dispatched('D')

        assert rig.scheduler.release_early_start_claim(claim) is True
        events = rig.events()
        assert rig.scheduler.release_early_start_claim(claim) is False
        assert rig.events() == events

    async def test_a_claim_whose_predecessor_released_is_still_released(self, tmp_path, monkeypatch):
        rig, claim = await self._claimed(tmp_path, monkeypatch)
        rig.scheduler.release('P')
        assert rig.scheduler.lock_table.successor_relation('D') is None
        assert rig.scheduler.lock_table.held_modules('D') == set(D_FILES)

        assert rig.scheduler.release_early_start_claim(claim) is True

        rig.assert_untouched('D')
        released = [e for e in rig.store.fetch_events_by_type(EventType.lock_released) if e['task_id'] == 'D']
        assert len(released) == 1

    async def test_a_claim_an_ordinary_release_ended_reads_as_stale(self, tmp_path, monkeypatch):
        rig, claim = await self._claimed(tmp_path, monkeypatch)
        rig.scheduler.release('D')
        events = rig.events()

        assert rig.scheduler.release_early_start_claim(claim) is False

        assert rig.events() == events

    async def test_an_earlier_claim_does_not_release_a_later_one(self, tmp_path, monkeypatch):
        rig, claim1 = await self._claimed(tmp_path, monkeypatch)
        assert rig.scheduler.release_early_start_claim(claim1) is True
        picture = await rig.picture(p_running(), task('D', deps=('P',)))
        claim2 = (await rig.select(picture, reads_for(picture))).claim
        assert claim2 is not None

        assert rig.scheduler.release_early_start_claim(claim1) is False
        assert rig.scheduler.is_dispatched('D')
        assert rig.scheduler.lock_table.held_modules('D') == set(D_FILES)

        assert rig.scheduler.release_early_start_claim(claim2) is True


class TestTheClaimsBookkeeping:
    """The claim is followed by awaited bookkeeping; a claim whose bookkeeping escapes is not kept."""

    async def test_a_claim_whose_bookkeeping_is_cancelled_is_released(self, tmp_path, monkeypatch):
        resolve = AsyncMock()
        rig = _rig(
            tmp_path, monkeypatch, callbacks=SchedulerCallbacks(on_starvation_resolve=resolve),
            announced_starving=('D',),
        )
        rig.offer(task('P', files=P_FILES), task('D', deps=('P',)))
        assert await rig.acquire() == 'P'
        resolve.side_effect = asyncio.CancelledError
        picture = await rig.picture(p_running(), task('D', deps=('P',)))

        with pytest.raises(asyncio.CancelledError):
            await rig.select(picture, reads_for(picture))

        resolve.assert_awaited_once_with('D')
        rig.assert_untouched('D')
        assert rig.scheduler.lock_table.held_modules('P') == set(P_FILES)
        released = [e for e in rig.store.fetch_events_by_type(EventType.lock_released) if e['task_id'] == 'D']
        assert len(released) == 1


class TestRow34NeverStartedTwice:
    async def test_a_claimed_dependent_is_not_started_again_by_the_ordinary_pass(
        self, tmp_path, monkeypatch,
    ):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(p_running(), task('D', deps=('P',)))
        assert (await rig.select(picture, reads_for(picture))).claim is not None

        rig.offer(task('P', files=P_FILES, status='cancelled'), task('D', deps=('P',)))
        assignment = await rig.scheduler.acquire_next()

        assert assignment is None or assignment.task_id != 'D'
        assert len(_acquired_rows(rig.store, 'D')) == 1

    async def test_a_dependent_the_ordinary_pass_started_is_not_claimed(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        rig.offer(task('P', files=P_FILES, status='done'), task('D', files=['c.py'], deps=('P',)))
        assert await rig.acquire() == 'D'
        picture = await rig.picture(p_running(), task('D', files=['c.py'], deps=('P',)))

        selection = await rig.select(picture, reads_for(picture))

        assert selection == EarlyStartSelection(None, (_refusal(Reason.NOT_ELIGIBLE),))
        assert rig.scheduler.lock_table.snapshot_successor_of() == {}


class TestRow8PredecessorChecks:
    async def test_checks_passed_at_the_tip_are_claimed(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(p_running(), task('D', deps=('P',)))

        selection = await rig.select(picture, reads_for(picture, checks=ChecksAtCommit.PASSED))

        assert selection.claim is not None

    @pytest.mark.parametrize(('checks', 'reason'), [
        (ChecksAtCommit.FAILED, Reason.DELIVERED_CHECK_FAILED),
        (ChecksAtCommit.SCRIPT_KIND, Reason.DELIVERED_CHECK_SCRIPT),
    ])
    async def test_checks_not_passed_refuse_the_pair(self, tmp_path, monkeypatch, checks, reason):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(p_running(), task('D', deps=('P',)))

        selection = await rig.select(picture, reads_for(picture, checks=checks))

        assert selection == EarlyStartSelection(None, (_refusal(reason),))
        rig.assert_untouched('D')


def _marked_dependent(mark: str | None) -> dict:
    row = task('D', deps=('P',))
    if mark is not None:
        row['metadata']['early_start'] = {'predecessor': mark}
    return row


class TestRow44TheBranchRule:
    @pytest.mark.parametrize(('branch', 'mark'), [
        (DependentBranch.UNSTACKED, None),
        (DependentBranch.STACKED, 'Q'),
        (DependentBranch.STACKED, None),
        (DependentBranch.UNREADABLE, None),
        (DependentBranch.UNREADABLE, 'P'),
    ], ids=['cut-from-main', 'stacked-on-another', 'stacked-unmarked', 'unreadable', 'unreadable-marked'])
    async def test_a_branch_not_stacked_on_this_predecessor_is_refused(
        self, tmp_path, monkeypatch, branch, mark,
    ):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(p_running(), _marked_dependent(mark))

        selection = await rig.select(picture, reads_for(picture, branch=branch))

        assert selection == EarlyStartSelection(None, (_refusal(Reason.HAS_BRANCH),))
        rig.assert_untouched('D')

    @pytest.mark.parametrize(('branch', 'mark'), [
        (DependentBranch.STACKED, 'P'),
        (DependentBranch.ABSENT, None),
    ], ids=['stacked-on-this-predecessor', 'no-branch'])
    async def test_no_branch_or_one_stacked_on_this_predecessor_is_claimed(
        self, tmp_path, monkeypatch, branch, mark,
    ):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(p_running(), _marked_dependent(mark))

        selection = await rig.select(picture, reads_for(picture, branch=branch))

        assert selection.claim is not None and selection.refusals == ()

    async def test_the_mark_is_read_from_the_store_row(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(p_running(), task('D', deps=('P',), early_start={'predecessor': 'P'}))
        unmarked = task('D', deps=('P',))

        selection = await rig.select(
            picture, reads_for(picture, branch=DependentBranch.STACKED, row=unmarked),
        )

        assert selection == EarlyStartSelection(None, (_refusal(Reason.HAS_BRANCH),))


def _saturated() -> PsiSample:
    return PsiSample(cpu_some10=0.0, mem_some10=99.0, mem_full10=0.0, io_some10=0.0, read_ok=True)


async def _held_by_the_switch(tmp_path, monkeypatch):
    return await _with_p_running(tmp_path, monkeypatch, early_start=EarlyStartConfig(enabled=False)), clear_holds()


async def _held_by_host_load(tmp_path, monkeypatch):
    rig = await _with_p_running(
        tmp_path, monkeypatch, psi_admission=PsiAdmissionConfig(enabled=True, min_inflight_floor=1),
    )
    assert rig.scheduler.is_dispatched('P')
    rig.scheduler._read_psi_sample = _saturated
    return rig, clear_holds()


async def _held_by_a_pause(tmp_path, monkeypatch):
    rig = await _with_p_running(tmp_path, monkeypatch)
    rig.scheduler.pause('operator says stop', PauseClass.OPERATOR)
    return rig, clear_holds()


async def _held_by_the_warm_base(tmp_path, monkeypatch):
    probe = AsyncMock(return_value='ok')
    rig = await _with_p_running(
        tmp_path, monkeypatch, callbacks=SchedulerCallbacks(warm_base_health_probe=probe),
        warm_base_hard_down=WarmBaseHardDownConfig(enabled=True),
    )
    probe.return_value = 'absent'
    assert await rig.scheduler.acquire_next() is None
    return rig, clear_holds()


def _held_by_the_harness(field: str):
    async def _setup(tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        return rig, dataclasses.replace(clear_holds(), **{field: True})
    return _setup


class TestRow11SystemHolds:
    @pytest.mark.parametrize(('setup', 'reason'), [
        (_held_by_the_switch, Reason.DISABLED),
        (_held_by_host_load, Reason.HOST_LOAD),
        (_held_by_a_pause, Reason.SCHEDULER_PAUSED),
        (_held_by_the_warm_base, Reason.WARM_BASE_DOWN),
        (_held_by_the_harness('review_checkpoint'), Reason.REVIEW_CHECKPOINT),
        (_held_by_the_harness('merge_queue_halted'), Reason.MERGE_QUEUE_HALTED),
        (_held_by_the_harness('usage_paused'), Reason.USAGE_PAUSED),
        (_held_by_the_harness('cost_ceiling_breached'), Reason.COST_CEILING),
    ], ids=[
        'disabled', 'host-load', 'paused', 'warm-base-down', 'review-checkpoint',
        'merge-queue-halted', 'usage-paused', 'cost-ceiling',
    ])
    async def test_a_hold_refuses_before_any_candidate_and_writes_nothing(
        self, tmp_path, monkeypatch, setup, reason,
    ):
        rig, holds = await setup(tmp_path, monkeypatch)
        picture = await rig.picture(p_running(), task('D', deps=('P', 'X')), task('E', deps=('P',)))
        events_before, calls_before = rig.events(), list(rig.session.calls)

        selection = await rig.select(
            picture, reads_of(pair_reads(picture, 'D'), pair_reads(picture, 'E')), holds=holds,
        )

        assert selection == EarlyStartSelection(None, (EarlyStartRefusal(reason),))
        rig.assert_untouched('D', 'E')
        assert rig.events() == events_before
        assert rig.session.calls == calls_before
        rig.status.assert_not_awaited()


def _higher_tier_waiting(blocking_task: str = 'T', dependent: str = 'D') -> EarlyStartRefusal:
    return EarlyStartRefusal(
        Reason.HIGHER_TIER_WAITING, dependent=dependent, predecessor='P', blocking_task=blocking_task,
    )


class TestTierCondition:
    """PRD section 5.1 "About the rest of the queue": a higher tier that could start once P lands wins.

    D ('medium' unless stated) needs a.py and c.py; P holds a.py.
    """

    async def _select(self, rig: _Rig, *others: dict, d: dict | None = None, **reads_kw):
        picture = await rig.picture(p_running(), d or task('D', deps=('P',)), *others)
        return await rig.select(picture, reads_for(picture, **reads_kw))

    async def test_row_3_a_higher_tier_that_could_start_on_a_shared_key_wins(
        self, tmp_path, monkeypatch,
    ):
        rig = await _with_p_running(tmp_path, monkeypatch)

        selection = await self._select(rig, task('T', 'high', ['c.py', 'e.py']))

        assert selection == EarlyStartSelection(None, (_higher_tier_waiting(),))
        rig.assert_untouched('D')

    @pytest.mark.parametrize('t_read', [True, False], ids=['t-read-unstacked', 't-unread'])
    async def test_row_4_a_higher_tier_waiting_on_the_same_predecessor_wins(
        self, tmp_path, monkeypatch, t_read,
    ):
        rig = await _with_p_running(tmp_path, monkeypatch)
        picture = await rig.picture(
            p_running(), task('D', deps=('P',)), task('T', 'high', ['a.py'], deps=('P',)),
        )
        pairs = [pair_reads(picture, 'D')]
        if t_read:
            pairs.append(pair_reads(picture, 'T', branch=DependentBranch.UNSTACKED))

        selection = await rig.select(picture, reads_of(*pairs))

        assert selection.claim is None
        assert _higher_tier_waiting() in selection.refusals
        rig.assert_untouched('D', 'T')

    async def test_row_5_a_higher_tier_also_held_by_an_unrelated_task_does_not_count(
        self, tmp_path, monkeypatch,
    ):
        rig = await _with_p_running(tmp_path, monkeypatch)
        rig.offer(p_running(), task('U', files=['u.py']))
        assert await rig.acquire() == 'U'

        selection = await self._select(
            rig, task('U', files=['u.py'], status='in-progress'),
            task('T', 'high', ['c.py', 'e.py', 'u.py']),
        )

        assert selection.claim is not None and selection.refusals == ()

    @pytest.mark.parametrize(('y_status', 'claimed'), [('done', False), ('in-progress', True)])
    async def test_row_6_the_statuses_just_read_decide_whether_a_higher_tier_could_start(
        self, tmp_path, monkeypatch, y_status, claimed,
    ):
        rig = await _with_p_running(tmp_path, monkeypatch)

        selection = await self._select(
            rig, task('Y', files=['y.py'], status='in-progress'),
            task('T', 'high', ['c.py'], deps=('Y',)),
            statuses={'Y': y_status},
        )

        if claimed:
            assert selection.claim is not None and selection.refusals == ()
        else:
            assert selection == EarlyStartSelection(None, (_higher_tier_waiting(),))

    async def test_a_higher_tier_kept_off_its_key_by_another_reservation_does_not_count(
        self, tmp_path, monkeypatch,
    ):
        rig = await _with_p_running(tmp_path, monkeypatch)
        rig.scheduler.lock_table.install_parks('Z', ['e.py'], 'low')

        selection = await self._select(rig, task('T', 'high', ['c.py', 'e.py']), task('Z', 'low', ['e.py']))

        assert selection.claim is not None and selection.refusals == ()

    @pytest.mark.parametrize('t_tier', ['medium', 'low'])
    async def test_a_same_or_lower_tier_does_not_count(self, tmp_path, monkeypatch, t_tier):
        rig = await _with_p_running(tmp_path, monkeypatch)

        selection = await self._select(rig, task('T', t_tier, ['c.py']))

        assert selection.claim is not None and selection.refusals == ()

    @pytest.mark.parametrize(('t_order', 'd_order', 'claimed'), [(1, 2, False), (2, 1, True)])
    async def test_between_pins_the_lower_pin_order_is_the_higher_tier(
        self, tmp_path, monkeypatch, t_order, d_order, claimed,
    ):
        rig = await _with_p_running(tmp_path, monkeypatch)
        rig.overrides.set_override(rig.project_root, 'T', pinned=True, pin_order=t_order)
        rig.overrides.set_override(rig.project_root, 'D', pinned=True, pin_order=d_order)

        selection = await self._select(rig, task('T', 'medium', ['c.py']))

        if claimed:
            assert selection.claim is not None and selection.refusals == ()
        else:
            assert selection == EarlyStartSelection(None, (_higher_tier_waiting(),))

    async def test_a_watchdog_pin_outranks_an_unpinned_critical(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        assert rig.overrides.create_watchdog_pin(
            rig.project_root, 'T', ttl_until=datetime.now(UTC) + timedelta(days=30),
        ) is not None

        selection = await self._select(
            rig, task('T', 'low', ['c.py']), d=task('D', 'critical', deps=('P',)),
        )

        assert selection == EarlyStartSelection(None, (_higher_tier_waiting(),))

    async def test_a_higher_tier_already_started_does_not_count(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)
        rig.offer(p_running(), task('T', 'high', ['two.py']))
        assert await rig.acquire() == 'T'

        selection = await self._select(
            rig, task('T', 'high', ['two.py']), d=task('D', files=['a.py', 'two.py'], deps=('P',)),
        )

        assert selection.claim is not None and selection.refusals == ()

    async def test_a_higher_tier_no_longer_pending_does_not_count(self, tmp_path, monkeypatch):
        rig = await _with_p_running(tmp_path, monkeypatch)

        selection = await self._select(
            rig, task('T', 'high', ['c.py']), statuses={'T': 'in-progress'},
        )

        assert selection.claim is not None and selection.refusals == ()
