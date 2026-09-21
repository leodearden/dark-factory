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
  step-11: the offline lane's red-path state (open fix task, advance count,
           promoted blocker) — one coupled subject keyed by one fingerprint.
  step-13: the scheduler's resurrection guard, which decides whether a
           re-pended task carries an age bonus it did not earn.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import shared.safe_io as _safe_io
from _orch_helpers import make_placeholder_future
from escalation.models import Escalation

from orchestrator import guard_state
from orchestrator.merge_types import QueuedBranch

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

    async def test_two_tasks_stewards_do_not_clobber_each_other(
        self, make_steward, guard_clock,
    ):
        """Every steward in the process shares one file per counter, so the
        load-merge-write has to hold at the integration level too."""
        for_42 = make_steward(task={'id': '42', 'title': 't', 'description': 'd'})
        for_43 = make_steward(task={'id': '43', 'title': 't', 'description': 'd'})

        for_42._mark_capped('esc-42-7')
        for_43._mark_capped('esc-43-7')

        redeployed = make_steward(task={'id': '42', 'title': 't', 'description': 'd'})
        assert 'esc-42-7' in redeployed._capped_escalations
        assert 'esc-43-7' in redeployed._capped_escalations


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
    from orchestrator.merge_queue import SpeculativeMergeWorker, TrainCallbacks

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
# step-11 — the offline lane's red-path state
# ---------------------------------------------------------------------------

_CONFIRMED = ['t::a', 't::b']
_WORKTREE = Path('/tmp/_offline-deep')


def _fingerprint() -> str:
    from orchestrator.workflow import compute_failing_test_set_fingerprint
    return compute_failing_test_set_fingerprint(_CONFIRMED)


def _offline_config(tmp_path: Path, **git_overrides):
    from _orch_helpers import pydantic_spec

    from orchestrator.config import GitConfig, OrchestratorConfig

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.project_root = tmp_path
    config.git = GitConfig(**git_overrides)
    return config


class _OfflineLane(NamedTuple):
    """One worker and the collaborator mocks its red path is measured through.

    They travel together because the worker declares them as optional
    Protocols, so reaching back through ``worker.task_client`` would be a
    read against ``... | None``.
    """

    worker: Any
    task_client: AsyncMock
    escalation_queue: MagicMock

    async def advance(self, head: str) -> None:
        await self.worker._handle_red_run(_WORKTREE, head)

    def blockers(self) -> list:
        return [
            call.args[0] for call in self.escalation_queue.submit.call_args_list
            if call.args[0].level == 2
        ]


def _offline_lane(
    tmp_path: Path, config, *, fix_task_id: str = 'fix-1', status: str = 'in-progress',
) -> _OfflineLane:
    """A worker rooted at *config.project_root*, with its red-path collaborators.

    ``escalation_queue.get_by_task`` returns nothing, so the pending-escalation
    dedup can never be what suppresses a second blocker — only the worker's own
    idempotency guard can, which is what these tests are about.
    """
    from orchestrator.offline_lane import OfflineLaneWorker

    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value='headsha')
    git_ops.reset_persistent_offline_deep_worktree = AsyncMock(return_value=_WORKTREE)

    task_client = AsyncMock()
    task_client.submit_fix_task = AsyncMock(return_value=fix_task_id)
    task_client.get_status = AsyncMock(return_value=status)

    escalation_queue = MagicMock()
    escalation_queue.get_by_task.return_value = []
    escalation_queue.make_id.return_value = 'esc-offline-1'

    worker = OfflineLaneWorker(
        git_ops,
        config,
        lock_path=tmp_path / 'offline_lane.lock',
        confirmation_runner=AsyncMock(return_value=list(_CONFIRMED)),
        task_client=task_client,
        escalation_queue=escalation_queue,
    )
    worker._last_green_head = 'GREEN'
    return _OfflineLane(worker, task_client, escalation_queue)


@pytest.mark.asyncio
class TestOfflineLaneRedStateSurvivesRedeploy:
    """The three red-path attributes are ONE subject keyed by one fingerprint.

    Persisting the counts without ``open_fix_tasks`` would be inert, not
    merely incomplete: ``_handle_red_run`` branches on it, so an empty map
    routes a restarted worker into ``_file_new_fix_task``, which files a
    DUPLICATE task for a break already tracked and resets the advance count
    to 1 before it can ever reach the promotion threshold.
    """

    async def test_a_restart_does_not_file_a_second_task_for_the_same_break(
        self, tmp_path: Path, guard_clock,
    ):
        config = _offline_config(tmp_path)
        first = _offline_lane(tmp_path, config)
        await first.advance('HEAD1')
        first.task_client.submit_fix_task.assert_awaited_once()

        redeployed = _offline_lane(tmp_path, config)
        await redeployed.advance('HEAD2')

        redeployed.task_client.submit_fix_task.assert_not_awaited()
        redeployed.task_client.append_suspect_range.assert_awaited_once_with(
            'fix-1', 'GREEN..HEAD2',
        )

    async def test_the_advance_count_accumulates_across_restarts(
        self, tmp_path: Path, guard_clock,
    ):
        """Three advances, three different workers: on main the count never
        reaches the threshold if a redeploy lands between advances."""
        config = _offline_config(tmp_path, offline_lane_red_advances_before_blocker=3)

        await _offline_lane(tmp_path, config).advance('HEAD1')
        second = _offline_lane(tmp_path, config)
        await second.advance('HEAD2')
        assert second.blockers() == [], 'two advances must not promote'

        third = _offline_lane(tmp_path, config)
        await third.advance('HEAD3')

        assert len(third.blockers()) == 1

    async def test_a_promoted_blocker_is_not_re_filed_after_a_restart(
        self, tmp_path: Path, guard_clock,
    ):
        """The symptom an operator cannot tell from a genuine recurrence."""
        config = _offline_config(tmp_path, offline_lane_red_advances_before_blocker=2)

        first = _offline_lane(tmp_path, config)
        await first.advance('HEAD1')
        await first.advance('HEAD2')
        assert len(first.blockers()) == 1

        redeployed = _offline_lane(tmp_path, config)
        await redeployed.advance('HEAD3')

        assert redeployed.blockers() == []

    async def test_the_done_clear_arm_is_durable(self, tmp_path: Path, guard_clock):
        """A landed fix task must free the fingerprint on disk, so a genuine
        LATER recurrence files a FRESH task rather than deduping against
        state that no longer describes anything."""
        config = _offline_config(tmp_path)
        await _offline_lane(tmp_path, config).advance('HEAD1')
        await _offline_lane(tmp_path, config, status='done').advance('HEAD2')

        later = _offline_lane(tmp_path, config, fix_task_id='fix-2')
        await later.advance('HEAD3')
        later.task_client.submit_fix_task.assert_awaited_once()

    async def test_an_empty_task_id_records_nothing_on_disk_either(
        self, tmp_path: Path, guard_clock,
    ):
        """The task-2016 amendment, across the persistence change: a bogus
        empty id must not become a key a later genuine recurrence dedups
        against."""
        config = _offline_config(tmp_path)
        await _offline_lane(tmp_path, config, fix_task_id='').advance('HEAD1')

        redeployed = _offline_lane(tmp_path, config).worker
        assert _fingerprint() not in redeployed.open_fix_tasks
        assert _fingerprint() not in redeployed._red_advance_counts

    async def test_the_red_state_expires(self, tmp_path: Path, guard_clock):
        """Past the TTL a fresh worker files a new task rather than deduping
        against a fix task that has long since stopped being relevant."""
        config = _offline_config(tmp_path)
        await _offline_lane(tmp_path, config).advance('HEAD1')

        guard_clock[0] = _T0 + timedelta(days=15)

        redeployed = _offline_lane(tmp_path, config, fix_task_id='fix-2')
        await redeployed.advance('HEAD2')
        redeployed.task_client.submit_fix_task.assert_awaited_once()


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


def _task(task_id: str, status: str = 'pending') -> dict:
    return {
        'id': task_id,
        'title': f'Task {task_id}',
        'status': status,
        'priority': 'medium',
        'dependencies': [],
        'metadata': {'files': [f'mod{task_id}']},
    }


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
        first._update_age_anchors([_task('100', status='cancelled')], _MAX_ID)
        first._update_age_anchors([_task('100')], _MAX_ID)

        redeployed = _scheduler()
        redeployed._update_age_anchors([_task('100')], _MAX_ID)

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

    def test_a_cold_start_writes_once_and_a_steady_state_tick_not_at_all(
        self, guard_clock, monkeypatch,
    ):
        """The two halves of one property, measured from the same spy.

        The COLD START is the first ``acquire_next`` tick after every fleet
        redeploy, inline on the event loop: it observes every active
        non-pending task at once — 373 in the live store today, 345 of them
        ``deferred``, a status that accumulates monotonically — so a
        per-element write would make that tick pay N writes of an N-entry
        file.  The STEADY STATE re-observes the same ids every ~15s, which is
        the difference between a quiet guard and a file rewritten ~5,760 times
        a day.  Spying from BEFORE the first call is what makes the cold start
        assertable at all.
        """
        cold_start = [
            _task(str(tid), status='deferred') for tid in range(1000, 1200)
        ]
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

    def test_the_guard_expires(self, guard_clock):
        """The mark cannot outlive the task it was recorded for."""
        first = _scheduler()
        first._update_age_anchors([_task('100', status='cancelled')], _MAX_ID)

        guard_clock[0] = _T0 + timedelta(days=31)

        redeployed = _scheduler()
        redeployed._update_age_anchors([_task('100')], _MAX_ID)
        assert redeployed._compute_age('100', _MAX_ID) == _MAX_ID - 100


@pytest.mark.asyncio
class TestStaleSweepRecordingSurvivesRedeploy:
    """``_phase_stale_sweep`` is the other write site and must be covered too."""

    async def test_a_swept_id_is_still_marked_after_a_restart(self, guard_clock):
        first = _scheduler()
        first._pending_anchor['300'] = 5
        first.get_tasks = AsyncMock(return_value=[_task('99')])
        first.get_statuses = AsyncMock(return_value=({}, None))

        await first.acquire_next()
        assert '300' in first._was_non_pending

        redeployed = _scheduler()
        redeployed._update_age_anchors([_task('300')], _MAX_ID)
        assert redeployed._compute_age('300', _MAX_ID) == 0
