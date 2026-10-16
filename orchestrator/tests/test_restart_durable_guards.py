"""A fleet redeploy must re-arm none of the orchestrator's one-strike guards.

Four guards, one property.  Each exists to say "never do this again" and each
used to hold that verdict in process RAM, so the ~8-15h fleet redeploy turned
every one of them into "never again *until the next redeploy*" — the prevented
thing recurred on a fixed cadence forever.  They are tested together, in one
file, because the guarantee is one guarantee: reading the four instances side
by side is what makes it legible, and no single owning suite could state it.

The restart is simulated the way it actually happens — a SECOND owner built
against the same ``project_root``, with fresh process memory and the same
files on disk.

Covers:
  step-7: the steward's capped set and its three per-escalation counters.
  step-9: the merge worker's coalesce one-strike registry.
  task-6861: the offline lane's episode record — every red-path decision
             for a command in one record, read by whichever worker runs next.
  step-13: the scheduler's resurrection guard, which decides whether a
           re-pended task carries an age bonus it did not earn.
  step-19: the steward's four guards being scoped to the TASK, not to the
           process, so its two enumeration sites keep reading only its own
           marks.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import shared.safe_io as _safe_io
from _lane_episode_fakes import Clock, FakeLedgerClient
from _orch_helpers import make_placeholder_future, pydantic_spec
from escalation.models import Escalation

from orchestrator import guard_state
from orchestrator.config import GitConfig, LaneCommand, OrchestratorConfig
from orchestrator.merge_types import QueuedBranch
from orchestrator.offline_lane import OfflineLaneWorker
from orchestrator.offline_lane_episode_state import (
    EPISODE_RETENTION,
    EPISODES_FILE,
    CommandRecord,
    EpisodeStore,
)

# A round, arbitrary instant.  Every TTL assertion is relative to it, so no
# test here depends on the wall clock.
_T0 = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def guard_clock(monkeypatch):
    """Freeze the clock every guard store defaults to, and hand back the dial.

    ``guard_state`` never reads the wall clock internally, so replacing this
    one module-level function is enough to drive every guard's TTL — no guard
    needs a test-only clock parameter threaded through its owner.
    """
    frozen = [_T0]
    monkeypatch.setattr(guard_state, '_utc_now', lambda: frozen[0])
    return frozen


def _escalation(**overrides) -> Escalation:
    defaults: dict = dict(
        id='esc-5352-1',
        task_id='42',
        agent_role='orchestrator',
        severity='blocking',
        category='limit_exhausted',
        summary='execute limit exhausted',
    )
    defaults.update(overrides)
    return Escalation(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# step-7 — the steward's give-up survives its own restart
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestStewardCapSurvivesRedeploy:
    """``make_steward`` roots every build at the same ``tmp_path/project``, so
    a second build IS the redeployed steward: same task, same project root,
    empty memory."""

    async def test_a_capped_escalation_is_not_re_handled_after_a_restart(
        self, make_steward, caplog, guard_clock,
    ):
        """The 1,183,854-lines-in-20.5h incident, one redeploy later.

        On main the fresh steward re-adopts the record from scratch, so a
        permanently unhandleable escalation is re-handled — and its ladder
        re-burnt — once per redeploy, forever.
        """
        first = make_steward(config_overrides={'steward_max_attempts': 1})
        first._mark_capped('esc-5352-1')

        redeployed = make_steward(config_overrides={'steward_max_attempts': 1})
        with patch(
            'orchestrator.steward.invoke_agent', new_callable=AsyncMock,
        ) as mock_invoke, caplog.at_level('INFO'):
            await redeployed._handle_escalation(_escalation())

        mock_invoke.assert_not_called()
        assert 'handling escalation' not in caplog.text, (
            'a capped record must produce no further log lines — silence is '
            'the signal that distinguishes a healthy idle steward from the spin'
        )

    async def test_a_capped_escalation_is_filtered_from_the_pending_read(
        self, make_steward, guard_clock,
    ):
        first = make_steward()
        first._mark_capped('esc-5352-1')

        redeployed = make_steward()
        redeployed.escalation_queue.get_by_task.return_value = [_escalation()]
        with patch.object(
            redeployed, '_watch_for_escalation', new_callable=AsyncMock,
        ) as mock_watch:
            mock_watch.return_value = None
            assert await redeployed._next_escalation() is None

        # The watcher must still be consulted, so the loop blocks on inotify
        # instead of hot-returning the capped record.  (Mock assertion methods
        # take no message argument — a trailing `, (...)` would build a
        # discarded tuple rather than attach an explanation.)
        mock_watch.assert_awaited_once()

    async def test_a_capped_escalation_is_excluded_from_the_watcher_argv(
        self, make_steward, guard_clock,
    ):
        """Filtering the pending read alone is not enough: the watcher's
        initial scan would emit the still-pending capped record immediately,
        turning every call into a subprocess respawn instead of an inotify
        block."""
        first = make_steward()
        first._mark_capped('esc-5352-1')

        redeployed = make_steward()
        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec:
            proc = AsyncMock()
            proc.returncode = 0
            proc.communicate.return_value = (b'', b'')
            mock_exec.return_value = proc
            await redeployed._watch_for_escalation()

        cmd = list(mock_exec.call_args[0])
        pairs = [
            (cmd[i], cmd[i + 1]) for i in range(len(cmd) - 1)
            if cmd[i] == '--exclude-id'
        ]
        assert pairs == [('--exclude-id', 'esc-5352-1')], f'got {cmd!r}'

    @pytest.mark.parametrize(
        ('counter', 'limit_field', 'limit'),
        [
            ('_retry_counts', 'steward_max_attempts', 1),
            ('_timeout_counts', 'steward_max_timeouts_per_escalation', 3),
            ('_empty_output_counts', 'steward_max_empty_outputs_per_escalation', 2),
        ],
    )
    async def test_an_exhausted_ladder_is_not_re_spent_after_a_restart(
        self, make_steward, guard_clock, counter, limit_field, limit,
    ):
        """The capped set is one layer; the counters beneath it are the other.

        Leaving these process-local would let a redeploy hand the same
        escalation a fresh full budget — the ladder re-spent once per
        redeploy, which is the cost this task is filed against.
        """
        overrides = {limit_field: limit, 'steward_max_attempts': 1}
        first = make_steward(config_overrides=overrides)
        getattr(first, counter)['esc-5352-1'] = limit

        redeployed = make_steward(config_overrides=overrides)
        with patch(
            'orchestrator.steward.invoke_agent', new_callable=AsyncMock,
        ) as mock_invoke:
            await redeployed._handle_escalation(_escalation())

        mock_invoke.assert_not_called()
        assert 'esc-5352-1' in redeployed._capped_escalations, (
            'the guard must fire on the FIRST look, not after another full budget'
        )

    async def test_a_new_escalation_still_gets_a_full_budget(
        self, make_steward, guard_clock,
    ):
        """The bound that makes persisting a give-up safe: the keys are
        escalation ids, so only the byte-identical record that already
        exhausted the ladder is remembered."""
        first = make_steward(config_overrides={'steward_max_attempts': 1})
        first._retry_counts['esc-5352-1'] = 1
        first._mark_capped('esc-5352-1')

        redeployed = make_steward(config_overrides={'steward_max_attempts': 1})
        assert redeployed._retry_counts.get('esc-5352-2', 0) == 0
        assert 'esc-5352-2' not in redeployed._capped_escalations

    async def test_resolution_clears_the_counters_durably(
        self, make_steward, guard_clock,
    ):
        """A resolved escalation's counters must not haunt a later record."""
        first = make_steward()
        esc = _escalation()
        first._retry_counts[esc.id] = 1
        first._timeout_counts[esc.id] = 2
        first._empty_output_counts[esc.id] = 1

        first._dismiss_capped_l0(esc, 'attempt_cap')

        redeployed = make_steward()
        assert redeployed._retry_counts.get(esc.id, 0) == 0
        assert redeployed._timeout_counts.get(esc.id, 0) == 0
        assert redeployed._empty_output_counts.get(esc.id, 0) == 0

    async def test_the_cap_re_arms_once_the_ttl_elapses(
        self, make_steward, guard_clock,
    ):
        """The state must not outlive its subject: past the TTL the record is
        uncapped again and its ladder is full."""
        first = make_steward(config_overrides={'steward_max_attempts': 1})
        first._mark_capped('esc-5352-1')
        first._retry_counts['esc-5352-1'] = 1

        guard_clock[0] = _T0 + timedelta(days=8)

        redeployed = make_steward(config_overrides={'steward_max_attempts': 1})
        assert 'esc-5352-1' not in redeployed._capped_escalations
        assert redeployed._retry_counts.get('esc-5352-1', 0) == 0


def _steward_for(make_steward, task_id: str, **kwargs):
    return make_steward(
        task={'id': task_id, 'title': 't', 'description': 'd'}, **kwargs,
    )


@pytest.mark.asyncio
class TestStewardGuardsAreTaskScoped:
    """Durable is not the same as shared.

    The four steward guards are keyed by escalation id alone, so one file per
    counter is read by EVERY ``TaskSteward`` in the process: each one loads
    every other task's marks at construction.  Membership tests survive that
    (ids are unique), but the two ENUMERATION sites do not.

    ``_log_capped_idle_once`` is reached from ``_next_escalation`` whenever the
    filtered pending list is empty — the normal steady state for a HEALTHY
    steward — so its ``if not self._capped_escalations`` early return stops
    firing and the steward warns about a count and a list of ids belonging to
    other tasks.  That line is the load-bearing incident signal from task
    3170 whose success condition is SILENCE, and a shared file makes it fire
    falsely for essentially every steward in the fleet as soon as any task
    caps anything inside the TTL.  ``_watch_for_escalation`` meanwhile passes
    one ``--exclude-id`` per capped id in the whole process to every watcher
    subprocess, so its argv grows with a week of fleet-wide cap history for no
    benefit.

    This is the counterpart to ``TestStewardCapSurvivesRedeploy``: that class
    pins WHEN the state is forgotten, this one pins WHO can see it, and the
    last test here holds both at once.
    """

    async def test_another_task_s_cap_is_not_visible(self, make_steward, guard_clock):
        _steward_for(make_steward, '43')._mark_capped('esc-43-7')

        for_42 = _steward_for(make_steward, '42')

        assert 'esc-43-7' not in for_42._capped_escalations
        assert len(for_42._capped_escalations) == 0

    async def test_a_steward_holding_no_caps_of_its_own_stays_silent(
        self, make_steward, guard_clock, caplog,
    ):
        """Silence IS the signal, so the assertion is the absence of a record
        rather than anything about its text."""
        _steward_for(make_steward, '43')._mark_capped('esc-43-7')
        for_42 = _steward_for(make_steward, '42')

        with caplog.at_level(logging.WARNING, logger='orchestrator.steward'):
            for_42._log_capped_idle_once()

        assert caplog.records == [], (
            'a steward that has capped nothing must not announce the '
            'capped-only idle state'
        )

    async def test_the_watcher_argv_excludes_only_this_task_s_caps(
        self, make_steward, guard_clock,
    ):
        _steward_for(make_steward, '43')._mark_capped('esc-43-7')
        for_42 = _steward_for(make_steward, '42')
        for_42._mark_capped('esc-42-7')

        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as mock_exec:
            proc = AsyncMock()
            proc.returncode = 0
            proc.communicate.return_value = (b'', b'')
            mock_exec.return_value = proc
            await for_42._watch_for_escalation()

        cmd = list(mock_exec.call_args[0])
        excluded = [cmd[i + 1] for i in range(len(cmd) - 1) if cmd[i] == '--exclude-id']
        assert excluded == ['esc-42-7'], f'got {cmd!r}'

    async def test_another_task_s_counters_are_not_visible_either(
        self, make_steward, guard_clock,
    ):
        """Not merely "unaffected by" — a steward must not be able to READ
        another task's ladder at all."""
        _steward_for(make_steward, '43')._retry_counts['esc-43-1'] = 3

        for_42 = _steward_for(make_steward, '42')

        assert for_42._retry_counts.get('esc-43-1', 0) == 0
        assert len(for_42._retry_counts) == 0

    async def test_scoping_does_not_weaken_durability(self, make_steward, guard_clock):
        """The regression guard for the restart-durable property: narrowing
        WHO shares the state must not shorten how long it lasts."""
        for_42 = _steward_for(make_steward, '42')
        for_42._mark_capped('esc-42-7')
        for_42._retry_counts['esc-42-7'] = 3
        _steward_for(make_steward, '43')._mark_capped('esc-43-7')

        redeployed = _steward_for(make_steward, '42')

        assert 'esc-42-7' in redeployed._capped_escalations
        assert redeployed._retry_counts.get('esc-42-7', 0) == 3
        assert 'esc-43-7' not in redeployed._capped_escalations


# ---------------------------------------------------------------------------
# step-9 — the merge worker's one strike stays spent
# ---------------------------------------------------------------------------

def _merge_config(project_root: Path):
    from orchestrator.config import GitConfig, OrchestratorConfig
    return OrchestratorConfig(
        project_root=project_root,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
        ),
    )


def _merge_request(task_id: str, config):
    from orchestrator.merge_queue import MergeRequest
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(f'task/{task_id}', config.git.branch_prefix),
        worktree=config.project_root / f'wt-{task_id}',
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=make_placeholder_future(),
    )


def _merge_worker(project_root: Path | None, config):
    """A bare worker rooted at *project_root* (``None`` = no project root).

    ``git_ops.project_root`` is a GENUINE path rather than an auto-specced
    child mock on purpose: a ``MagicMock`` is a valid ``os.PathLike`` whose
    ``os.fspath()`` is repr-derived, so letting the attribute auto-spec would
    have the worker materialise a real ``MagicMock/`` tree the moment it
    persisted anything — the task-3223 incident conftest's
    ``_no_mock_derived_stray_dirs`` fence exists for.
    """
    from orchestrator.merge_lane.types import TrainCallbacks
    from orchestrator.merge_queue import SpeculativeMergeWorker

    git_ops = MagicMock()
    git_ops.config = config
    if project_root is None:
        del git_ops.project_root
    else:
        git_ops.project_root = project_root

    def _factory(train_id: str) -> TrainCallbacks:
        return TrainCallbacks(
            status_check=AsyncMock(return_value={}),
            mark_member_done=AsyncMock(),
        )

    # event_store=None short-circuits signal 2 (the blocked-history arm) by its
    # own `is not None` guard, so these assertions isolate signal 1.
    return SpeculativeMergeWorker(
        git_ops,
        asyncio.Queue(),
        event_store=None,
        train_callback_factory=_factory,
    )


@pytest.mark.asyncio
class TestCoalesceDerailSurvivesRedeploy:
    """A train that derailed once must not re-form after a restart.

    On main the registry is a bare ``set()`` rebuilt at construction, which is
    why ``train_derailed`` reads 19-in-30d / 16-in-14d: the identical poison
    train re-forms on the redeploy cadence, forever.
    """

    async def test_a_derailed_member_is_still_excluded_after_a_restart(
        self, tmp_path: Path, guard_clock,
    ):
        config = _merge_config(tmp_path)
        _merge_worker(tmp_path, config)._mark_coalesce_derailed(['4001', '4002'])

        redeployed = _merge_worker(tmp_path, config)
        for task_id in ('4001', '4002'):
            assert redeployed._default_coalesce_exclusion_reason(
                _merge_request(task_id, config),
            ) == 'coalesce_derailed_one_strike'

    async def test_a_task_that_never_derailed_is_unaffected(
        self, tmp_path: Path, guard_clock,
    ):
        config = _merge_config(tmp_path)
        _merge_worker(tmp_path, config)._mark_coalesce_derailed(['4001', '4002'])

        redeployed = _merge_worker(tmp_path, config)
        assert redeployed._default_coalesce_exclusion_reason(
            _merge_request('4003', config),
        ) is None

    async def test_the_strike_expires(self, tmp_path: Path, guard_clock):
        """A task is not excluded from train formation forever — the marker
        decays with the TTL, which IS the decay policy the old comment
        anticipated."""
        config = _merge_config(tmp_path)
        _merge_worker(tmp_path, config)._mark_coalesce_derailed(['4001'])

        guard_clock[0] = _T0 + timedelta(days=8)

        assert _merge_worker(tmp_path, config)._default_coalesce_exclusion_reason(
            _merge_request('4001', config),
        ) is None

    async def test_a_worker_without_a_project_root_still_works(
        self, tmp_path: Path, guard_clock,
    ):
        """The bare-worker case degrades to memory and writes nothing — the
        same property ``test_merge_queue_multihost_wiring.py`` pins for the
        sibling state paths."""
        config = _merge_config(tmp_path)
        worker = _merge_worker(None, config)
        worker._mark_coalesce_derailed(['4001'])

        assert worker._default_coalesce_exclusion_reason(
            _merge_request('4001', config),
        ) == 'coalesce_derailed_one_strike'
        assert not (tmp_path / 'data').exists(), 'nothing may reach disk'


# ---------------------------------------------------------------------------
# task-6861 — the offline lane's episode record
# ---------------------------------------------------------------------------

_LANE_COMMAND = 'integration'
_LANE_NODE = 'tests/test_a.py::test_one'
_LANE_HEAD = 'a1' * 20
_LANE_PASS_BOUND_SECS = 30.0

# A lane command red on one test every run, in pytest's own summary form.
_RED_SCRIPT = (
    f'echo "FAILED {_LANE_NODE} - AssertionError: still red"\n'
    'echo "1 failed in 0.01s"\n'
    'exit 1\n'
)


class _ProcessDied(BaseException):
    """The orchestrator process ends here: ``OfflineLaneWorker.run`` lets anything but an ``Exception`` through."""


class _Stopped(BaseException):
    """The process is stopped at its next look at main, after the pass it was woken for."""


class _TaskStore(FakeLedgerClient):
    """The task store, which outlives every process; it can end its caller between submit and commit."""

    die_at_next_commit = False

    async def commit_planning(self, task_ids: list[str]) -> None:
        if self.die_at_next_commit:
            self.die_at_next_commit = False
            raise _ProcessDied
        await super().commit_planning(task_ids)


class _LaneHost:
    """One project root, its task store and its clock; each :meth:`process` is a redeploy's fresh worker."""

    def __init__(self, tmp_path: Path) -> None:
        self.project_root = tmp_path
        self.worktree = tmp_path / 'wt'
        self.worktree.mkdir()
        (self.worktree / 'red.sh').write_text(_RED_SCRIPT)
        self.store = _TaskStore([])
        self.clock = Clock()

    def process(self) -> _LaneProcess:
        return _LaneProcess(self)

    def record(self) -> CommandRecord:
        return EpisodeStore(self.project_root, clock=self.clock).load(_LANE_COMMAND)


class _LaneProcess:
    """A worker with empty memory on the host's project root, running the red command."""

    def __init__(self, host: _LaneHost) -> None:
        self._host = host
        self._git_ops = MagicMock()
        self._git_ops.reset_persistent_offline_deep_worktree = AsyncMock(return_value=host.worktree)
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.project_root = host.project_root
        config.git = GitConfig(
            offline_lane_legacy_numeric_enabled=False,
            offline_lane_commands=[LaneCommand(name=_LANE_COMMAND, command='sh red.sh')],
            offline_lane_poll_interval_secs=0.01,
        )
        self.worker = OfflineLaneWorker(
            self._git_ops,
            config,
            lock_path=host.project_root / 'offline_lane.lock',
            task_client=host.store,
            now=host.clock,
        )

    async def red_pass(self, *, minutes: float = 30) -> None:
        """Wake the worker *minutes* after the last pass and let ``run()`` make one pass, then stop it."""
        self._host.clock.advance(minutes=minutes)
        self._git_ops.get_main_sha = AsyncMock(side_effect=[_LANE_HEAD, _Stopped()])
        await self.worker.on_post_merge('task-1', _LANE_HEAD, _LANE_HEAD)
        with pytest.raises(_Stopped):
            await asyncio.wait_for(self.worker.run(), timeout=_LANE_PASS_BOUND_SECS)


@pytest.mark.asyncio
class TestOfflineLaneEpisodeRecordSurvivesRedeploy:
    """Every red-path decision lives in one record per command, in
    ``data/orchestrator/guards/offline_lane_episodes.json``: the held
    episodes, the covering task's binding and how far its filing got.  A
    redeploy's fresh worker reads that file, so it neither files a second
    task for a break already covered nor abandons a filing it was cut off
    in the middle of (red-path PRD §6 scenarios 19 and 21).
    """

    async def test_a_redeployed_worker_files_no_second_task_for_a_covered_break(self, tmp_path: Path):
        host = _LaneHost(tmp_path)
        first = host.process()
        await first.red_pass()
        await first.red_pass()
        assert len(host.store.submitted) == 1

        redeployed = host.process()
        await redeployed.red_pass()
        await redeployed.red_pass()

        assert len(host.store.submitted) == 1
        assert host.record().episodes[_LANE_NODE].covering_task_id == '7001'

    async def test_a_filing_cut_off_before_commit_planning_is_committed_not_refiled(self, tmp_path: Path):
        """Scenario 19: the submit landed, the process died before
        ``commit_planning``; the next process's next pass commits that task."""
        host = _LaneHost(tmp_path)
        first = host.process()
        await first.red_pass()
        host.store.die_at_next_commit = True
        with pytest.raises(_ProcessDied):
            await first.red_pass()
        stranded = host.record().episodes[_LANE_NODE]

        await host.process().red_pass()

        assert (stranded.covering_task_id, stranded.planning_committed, stranded.filing_due) == ('7001', False, True)
        assert host.store.tasks['7001']['status'] == 'pending'
        assert len(host.store.submitted) == 1
        episode = host.record().episodes[_LANE_NODE]
        assert (episode.planning_committed, episode.filing_due) == (True, False)

    async def test_an_unreadable_record_costs_at_most_one_duplicate_and_says_so(
        self, tmp_path: Path, caplog,
    ):
        """Scenario 21: the record is lost while its task is open."""
        host = _LaneHost(tmp_path)
        first = host.process()
        await first.red_pass()
        await first.red_pass()
        episodes_file = guard_state.guard_path(tmp_path, EPISODES_FILE)
        assert episodes_file is not None
        episodes_file.write_text('{ not json', encoding='utf-8')

        with caplog.at_level(logging.WARNING):
            redeployed = host.process()
        for _ in range(4):
            await redeployed.red_pass()

        assert len(host.store.submitted) == 2
        assert any(
            record.levelno >= logging.WARNING and EPISODES_FILE in record.getMessage()
            for record in caplog.records
        )

    @pytest.mark.parametrize(
        ('idle', 'submitted'),
        [(EPISODE_RETENTION - timedelta(minutes=1), 1), (EPISODE_RETENTION, 2)],
        ids=['kept', 'expired'],
    )
    async def test_the_record_expires_fourteen_days_after_its_last_write(
        self, tmp_path: Path, idle: timedelta, submitted: int,
    ):
        """Kept, the binding still covers the test, so two spaced reds file
        nothing; expired, the record is gone and they file a fresh task that
        names no prior one."""
        host = _LaneHost(tmp_path)
        first = host.process()
        await first.red_pass()
        await first.red_pass()

        host.clock.advance(seconds=idle.total_seconds())
        redeployed = host.process()
        await redeployed.red_pass(minutes=0)
        await redeployed.red_pass()

        assert len(host.store.submitted) == submitted
        assert host.store.submitted[-1]['metadata']['x_offline_lane']['prior_task_ids'] == []


# ---------------------------------------------------------------------------
# step-13 — the scheduler's resurrection guard
# ---------------------------------------------------------------------------

_MAX_ID = 5000


def _scheduler():
    """A started scheduler on this test's ``project_root``.

    conftest's autouse ``_isolate_orch_config`` pins that root to the test's
    ``tmp_path``, so a second build in the same test IS the redeployed
    scheduler: same guards directory, empty memory.
    """
    from orchestrator.config import OrchestratorConfig
    from orchestrator.scheduler import Scheduler

    scheduler = Scheduler(OrchestratorConfig(max_per_module=1))
    scheduler.finish_startup()
    return scheduler


_FILED = '2026-10-01T00:00:00+00:00'
_REOPENED = '2026-10-05T00:00:00+00:00'


def _task(task_id: str, status: str = 'pending', pending_since: str | None = None) -> dict:
    metadata: dict = {'files': [f'mod{task_id}']}
    if pending_since is not None:
        metadata['pending_since'] = pending_since
    return {
        'id': task_id,
        'title': f'Task {task_id}',
        'status': status,
        'priority': 'medium',
        'dependencies': [],
        'metadata': metadata,
    }


async def _tick(scheduler, tasks: list[dict]) -> str | None:
    scheduler.get_tasks = AsyncMock(return_value=tasks)
    scheduler.get_statuses = AsyncMock(return_value=({}, None))
    assignment = await scheduler.acquire_next()
    return assignment.task_id if assignment is not None else None


def _spy_on_guard_writes(monkeypatch) -> list:
    calls: list = []
    real = _safe_io.atomic_write_text

    def recorder(path, text, **kwargs):
        calls.append(path)
        return real(path, text, **kwargs)

    monkeypatch.setattr(_safe_io, 'atomic_write_text', recorder)
    return calls


class TestResurrectionAnchorSurvivesRedeploy:
    """``_update_age_anchors`` is driven directly, so the assertion is about
    the anchor and nothing else."""

    def test_a_resurrected_task_does_not_re_acquire_its_age_bonus(self, guard_clock):
        """On main the fresh scheduler anchors '100' to ``int('100')``, giving
        it an age of 4900 — worth ``age_alpha: 10.0 x 4900``.  It jumps the
        queue over genuinely-old pending tasks once per redeploy, and its
        victims are the starvation watchdog's own."""
        first = _scheduler()
        first._update_age_anchors([_task('100', pending_since=_FILED)], _MAX_ID)
        first._update_age_anchors([_task('100', pending_since=_REOPENED)], _MAX_ID)

        redeployed = _scheduler()
        redeployed._update_age_anchors([_task('100', pending_since=_REOPENED)], _MAX_ID)

        assert redeployed._compute_age('100', _MAX_ID) == 0

    def test_a_genuinely_old_pending_task_keeps_its_age(self, guard_clock):
        """The property the fix must not break."""
        redeployed = _scheduler()
        redeployed._update_age_anchors([_task('200')], _MAX_ID)

        assert redeployed._compute_age('200', _MAX_ID) == _MAX_ID - 200

    def test_a_non_numeric_id_anchors_to_max_id_either_way(self, guard_clock):
        """Pins the third branch across the change."""
        first = _scheduler()
        first._update_age_anchors([_task('epic-a', status='cancelled')], _MAX_ID)

        redeployed = _scheduler()
        redeployed._update_age_anchors([_task('epic-a')], _MAX_ID)
        redeployed._update_age_anchors([_task('epic-b')], _MAX_ID)

        assert redeployed._compute_age('epic-a', _MAX_ID) == 0
        assert redeployed._compute_age('epic-b', _MAX_ID) == 0

    @pytest.mark.asyncio
    async def test_a_cold_start_writes_once_and_a_steady_state_tick_not_at_all(
        self, guard_clock, monkeypatch,
    ):
        """The two halves of one property, measured from the same spy.

        The COLD START is a first tick that finds N stamps changed while the
        process was down, inline on the event loop.  Its N resets are
        batched, so it pays one write rather than N writes of an N-entry file.
        The STEADY STATE re-observes the same ids every ~15s, which is
        the difference between a quiet guard and a file rewritten ~5,760 times
        a day.  Spying from BEFORE the first call is what makes the cold start
        assertable at all.
        """
        ids = range(1000, 1200)
        first = _scheduler()
        await _tick(first, [_task(str(tid), 'in-progress', _FILED) for tid in ids])
        await _tick(first, [_task('99', 'in-progress')])

        cold_start = [_task(str(tid), pending_since=_REOPENED) for tid in ids]
        scheduler = _scheduler()

        writes = _spy_on_guard_writes(monkeypatch)
        scheduler._update_age_anchors(cold_start, _MAX_ID)

        assert len(writes) == 1, (
            f'a cold start of {len(cold_start)} observations must be one write,'
            f' got {len(writes)}'
        )

        writes.clear()
        scheduler._update_age_anchors(cold_start, _MAX_ID)

        assert writes == [], 'a re-observation of an unchanged observation must be silent'

    @pytest.mark.asyncio
    async def test_the_guard_expires(self, guard_clock):
        """A record that no write refreshed within the TTL expires, and the
        restart re-anchors the task to its own id.  That is the trade-off
        which bounds the file, and it takes a TTL with no age-guard write."""
        first = _scheduler()
        await _tick(first, [
            _task('100', 'in-progress', _FILED), _task('4000', 'in-progress'),
        ])
        await _tick(first, [_task('100', pending_since=_REOPENED), _task('4000', 'in-progress')])

        guard_clock[0] = _T0 + timedelta(days=31)

        redeployed = _scheduler()
        redeployed._update_age_anchors([_task('100', pending_since=_REOPENED)], _MAX_ID)
        assert redeployed._compute_age('100', _MAX_ID) == _MAX_ID - 100

    @pytest.mark.asyncio
    async def test_any_write_within_the_ttl_keeps_an_in_fetch_record_alive(self, guard_clock):
        """'1000''s departure on day 20 re-records '100''s reset, so on day 40
        the restart restores it: '900''s 150 beats the 50 accrued since."""
        first = _scheduler()
        await _tick(first, [_task('100', 'in-progress', _FILED), _task('1000', 'in-progress')])
        await _tick(first, [_task('100', 'in-progress', _REOPENED), _task('1000', 'in-progress')])
        guard_clock[0] = _T0 + timedelta(days=20)
        await _tick(first, [_task('100', 'in-progress', _REOPENED)])
        guard_clock[0] = _T0 + timedelta(days=40)

        tasks = [_task('100', pending_since=_REOPENED), _task('900'), _task('1050', 'in-progress')]

        assert await _tick(_scheduler(), tasks) == '900'


@pytest.mark.asyncio
class TestStaleSweepRecordingSurvivesRedeploy:
    """``_phase_stale_sweep`` is the other write site and must be covered too."""

    async def test_a_swept_anchor_is_restored_after_a_restart(self, guard_clock):
        first = _scheduler()
        first._pending_anchor['300'] = 5
        await _tick(first, [_task('99')])

        redeployed = _scheduler()
        redeployed._update_age_anchors([_task('300')], _MAX_ID)
        assert redeployed._compute_age('300', _MAX_ID) == _MAX_ID - 5, (
            'absence from the fetch is not a cancellation (D3)'
        )
