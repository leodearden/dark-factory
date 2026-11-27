"""The early-start selection at the Scheduler boundary (early start, PRD sections 5.1-5.4).

``plans/early-start-on-predecessor-branch-prd.md``, task T5a: step (a), the
shortlist of dependents that qualify to start on a waiting predecessor's
branch, and step (c), the uninterrupted claim. P is dispatched by the ordinary
pass first, so it really holds its keys, and is then offered as in-progress.

The Scheduler runs on the code defaults: pydantic-settings would otherwise merge
the operational yaml's own ``module_overrides`` into the test's holder limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _orch_helpers import HermeticMcpSession
from _park_test_helpers import make_task

from orchestrator.config import OrchestratorConfig
from orchestrator.early_start_selection import (
    EarlyStartRefusal,
    EarlyStartRefusalReason,
    EarlyStartShortlist,
)
from orchestrator.event_store import EventStore
from orchestrator.overrides import OverrideStore
from orchestrator.scheduler import Scheduler
from orchestrator.task_picture import TaskPicture

pytestmark = [pytest.mark.usefixtures('code_default_config'), pytest.mark.asyncio]

FIXED_DT = datetime(2026, 11, 22, 0, 0, 0, tzinfo=UTC)
P_FILES = ['a.py', 'b.py']
D_FILES = ['a.py', 'c.py']

Reason = EarlyStartRefusalReason


@dataclass(frozen=True)
class _Rig:
    scheduler: Scheduler
    store: EventStore
    overrides: OverrideStore
    session: HermeticMcpSession
    status: AsyncMock
    project_root: str

    def offer(self, *tasks: dict) -> None:
        self.scheduler.get_tasks = AsyncMock(return_value=[dict(t) for t in tasks])

    async def picture(self, *tasks: dict) -> TaskPicture:
        self.offer(*tasks)
        picture = await self.scheduler.build_task_picture()
        assert picture is not None
        return picture

    async def dispatch(self, task: dict) -> None:
        """Dispatch *task* through the ordinary pass, so it really holds its keys."""
        self.offer(task)
        assignment = await self.scheduler.acquire_next()
        assert assignment is not None and assignment.task_id == task['id']

    def shortlist(self, picture: TaskPicture, *waiting: str) -> EarlyStartShortlist:
        return self.scheduler.early_start_shortlist(picture, waiting_predecessors=set(waiting or ('P',)))


def _rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **config_overrides: object) -> _Rig:
    config = OrchestratorConfig(
        max_per_module=1, lock_depth=12, module_overrides={'two.py': 2},
        early_start={'enabled': True}, project_root=tmp_path, **config_overrides,
    )
    store = EventStore(tmp_path / 'runs.db', 'run-1')
    overrides = OverrideStore(tmp_path / 'overrides.db')
    session = HermeticMcpSession()
    scheduler = Scheduler(
        config,
        store,
        mcp_session=session,
        wall_time_source=lambda: FIXED_DT,
        override_store=overrides,
        state_snapshot_path=tmp_path / 'scheduler_state.json',
        branch_work_reader=dict,
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


async def _with_p_running(tmp_path, monkeypatch, **config_overrides) -> _Rig:
    rig = _rig(tmp_path, monkeypatch, **config_overrides)
    await rig.dispatch(task('P', files=P_FILES))
    return rig


def _refusal(reason: Reason, dependent: str = 'D', predecessor: str = 'P') -> EarlyStartRefusal:
    return EarlyStartRefusal(reason, dependent=dependent, predecessor=predecessor)


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
