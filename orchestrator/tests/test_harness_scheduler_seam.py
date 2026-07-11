"""Tests for the Harness<->Scheduler constructor-injected callback seam
(task 2235, W10-alpha).

Replaces the nine post-construction ``scheduler._on_*`` monkey-patches
(installed in ``Harness.__init__`` after the Scheduler is built) with a
single ``SchedulerCallbacks`` bundle passed to ``Scheduler(config,
callbacks=...)`` at construction time.  Also covers the three liveness
reach-ins (``_dispatched`` / ``lock_table._held`` / ``_requeue_history``)
and the startup gate (``finish_startup`` / ``started``).

Covers:
  (a) callback-install grep-guard -- harness.py must not post-construction
      ASSIGN any of the nine retired scheduler callback attributes.
  (b) constructed-with-callbacks -- a freshly-built Harness's scheduler has
      all 9 hooks wired (non-None) with no half-wired window, and driving a
      representative trigger (park-stop trip) actually routes to the
      harness's own ``pause_scheduler`` method.
  (c) liveness-read grep-guard -- harness.py must not reach into
      ``scheduler._dispatched`` / ``scheduler.lock_table._held`` /
      ``scheduler._requeue_history`` directly -- it must use the public
      accessors instead.
  (d) behavioural -- the mid-run stranded sweep consults
      ``scheduler.is_actively_held`` (folding dispatched/held/recent-cancel)
      and the retry-cap path reads requeue history via
      ``scheduler.requeue_history``.
  (e) startup -- ``scheduler.finish_startup()`` is invoked after the startup
      reconcile sweeps and strictly before the first ``acquire_next()``.

Step-9 (callback-install guard, part b) is RED against the current
harness.py: the nine hooks are still installed as post-construction
attribute assignments (dead writes -- the Scheduler no longer reads them,
per task 2235 steps 1-8) and the Harness never passes ``callbacks=`` to the
Scheduler constructor.  Step-10 migrates harness.py to close this gap.

Step-11 (liveness-read guard + behavioural + startup, parts c-e) is RED
against harness.py post-step-10: the three liveness reach-ins are still
direct attribute access and ``run()`` never calls ``finish_startup()``.
Step-12 migrates harness.py to close this gap.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness, TaskReport
from orchestrator.run_store import RunStore
from orchestrator.workflow import WorkflowOutcome

_HARNESS_SRC_PATH = Path(__file__).parent.parent / 'src' / 'orchestrator' / 'harness.py'

# Assignment-form (LHS ``self.scheduler.<attr> =``) reach-in patterns for the
# nine retired scheduler callback attributes. Deliberately scoped to
# assignment form (trailing ``\s*=``) so this guard does NOT false-positive
# on the retired-pattern docstrings (~4177-4458) that mention e.g.
# ``self.scheduler._on_external_dep_block`` in backticked prose without a
# trailing ``=`` -- those are reworded by step-10, not deleted, and a naive
# whole-file substring scan would wrongly flag them.
_CALLBACK_ASSIGNMENT_PATTERNS = (
    re.compile(r'self\.scheduler\._on_\w+\s*='),
    re.compile(r'self\.scheduler\._warm_base_health_probe\s*='),
    re.compile(r'self\.scheduler\._suppress_blocked_write\s*='),
)


def _find_callback_assignment_reachins(content: str) -> list[str]:
    """Return ``"LINENO: text"`` for each line in *content* that assigns
    directly to one of the nine retired scheduler callback attributes.

    Line-based (not a single ``re.MULTILINE`` scan) so failures report an
    actionable line number instead of just "somewhere in this 9000-line file".
    """
    hits = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        if any(pattern.search(line) for pattern in _CALLBACK_ASSIGNMENT_PATTERNS):
            hits.append(f'{lineno}: {line.strip()}')
    return hits


class TestCallbackInstallGrepGuard:
    """(a) harness.py must not post-construction-assign the nine retired
    scheduler callback attributes -- they must flow through the
    SchedulerCallbacks constructor seam instead (task 2235)."""

    def test_no_assignment_form_reachins(self) -> None:
        content = _HARNESS_SRC_PATH.read_text()
        hits = _find_callback_assignment_reachins(content)
        assert not hits, (
            'harness.py still post-construction-assigns retired Scheduler '
            'callback attributes; task 2235 migrates these to '
            f'Scheduler(config, callbacks=SchedulerCallbacks(...)): {hits}'
        )

    def test_excluded_post_construction_gates_survive_untouched(self) -> None:
        """Sanity check: the two intentionally-excluded post-construction
        gates (``_landed_outbox_gate`` task 2156, ``_already_landed_gate``
        task 2313) are NOT nine-callback reach-ins, so this guard must not
        (and does not) flag them -- they legitimately stay post-construction.
        """
        content = _HARNESS_SRC_PATH.read_text()
        assert 'self.scheduler._landed_outbox_gate = ' in content, (
            'expected the excluded _landed_outbox_gate install (task 2156) '
            'to still be present in harness.py'
        )
        assert 'self.scheduler._already_landed_gate = ' in content, (
            'expected the excluded _already_landed_gate install (task 2313) '
            'to still be present in harness.py'
        )
        hits = _find_callback_assignment_reachins(content)
        assert not any('_landed_outbox_gate' in hit for hit in hits)
        assert not any('_already_landed_gate' in hit for hit in hits)


class TestConstructedWithCallbacks:
    """(b) A freshly-built Harness wires all 9 hooks into the Scheduler's
    SchedulerCallbacks bundle AT CONSTRUCTION TIME -- no half-wired window
    where the Scheduler exists but its callbacks are unset."""

    def test_all_nine_hooks_non_none_after_construction(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)

        callbacks = harness.scheduler._callbacks
        for field in dataclasses.fields(callbacks):
            assert getattr(callbacks, field.name) is not None, (
                f'SchedulerCallbacks.{field.name} is None immediately after '
                'Harness construction -- task 2235 requires the Harness to '
                'build the Scheduler with callbacks=SchedulerCallbacks(...) '
                'at construction time (no post-construction install window).'
            )

    @pytest.mark.asyncio
    async def test_park_stop_trip_routes_to_harness_pause_scheduler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Driving the park-stop trip through the Scheduler must route to
        the Harness's OWN ``pause_scheduler`` -- observed via its RunStore
        persistence side effect, not merely ``scheduler.is_paused`` (which
        the scheduler's synchronous latch sets on its own, regardless of
        whether the trip callback is wired at all -- see
        ``_maybe_fire_park_stop_trip``'s docstring)."""
        config = OrchestratorConfig(
            project_root=tmp_path,
            max_per_module=1,
            park_stop_parked_threshold=1,
            park_stop_parked_window_hours=1.0,
        )
        harness = Harness(config)

        mock_run_store = MagicMock(spec=RunStore)
        harness._run_store = mock_run_store
        harness._run_id = 'run-test-0001'
        harness.event_store = EventStore(tmp_path / 'events.db', 'run-test-0001')

        monkeypatch.setattr(
            'orchestrator.scheduler.mcp_call', AsyncMock(return_value={}),
        )

        await harness.scheduler.set_task_status('T1', 'blocked')
        # Yield to the event loop so the ensure_future'd trip callback runs.
        await asyncio.sleep(0)

        mock_run_store.save_scheduler_pause.assert_called_once()


# ---------------------------------------------------------------------------
# (c) Liveness-read grep-guard (step-11/12)
# ---------------------------------------------------------------------------

# Reach-in patterns for the three Scheduler-private liveness attributes.
# Unlike the callback-install guard above, these are READ patterns -- any
# appearance (expression or docstring prose) indicates either a live
# reach-in or a stale doc-comment; task 2235 step-12 eliminates the literal
# substring from harness.py entirely (rewording prose mentions), so a plain
# scan -- not restricted to assignment form -- is the correct guard here.
_LIVENESS_REACHIN_PATTERNS = (
    re.compile(r'self\.scheduler\._dispatched\b'),
    re.compile(r'self\.scheduler\.lock_table\._held\b'),
    re.compile(r'self\.scheduler\._requeue_history\b'),
)


def _find_liveness_reachins(content: str) -> list[str]:
    """Return ``"LINENO: text"`` for each line in *content* that reaches
    into Scheduler-private liveness state directly."""
    hits = []
    for lineno, line in enumerate(content.splitlines(), start=1):
        if any(pattern.search(line) for pattern in _LIVENESS_REACHIN_PATTERNS):
            hits.append(f'{lineno}: {line.strip()}')
    return hits


class TestLivenessReadGrepGuard:
    """(c) harness.py must not reach into the Scheduler's private liveness
    state (``_dispatched`` / ``lock_table._held`` / ``_requeue_history``) --
    it must use the public ``is_dispatched()`` / ``is_actively_held()`` /
    ``requeue_history()`` accessors instead (task 2235)."""

    def test_no_liveness_reachins(self) -> None:
        content = _HARNESS_SRC_PATH.read_text()
        hits = _find_liveness_reachins(content)
        assert not hits, (
            'harness.py still reaches into Scheduler-private liveness state '
            '(_dispatched / lock_table._held / _requeue_history) directly; '
            f'task 2235 migrates these to the public accessors: {hits}'
        )


# ---------------------------------------------------------------------------
# (d) Behavioural: mid-run sweep + retry-cap accessor usage (step-11/12)
# ---------------------------------------------------------------------------

class TestMidRunSweepUsesIsActivelyHeld:
    """(d1) The mid-run stranded-in-progress sweep consults
    ``scheduler.is_actively_held()`` (folding dispatched / module-lock-held /
    recent-cancel into one Scheduler-owned check) rather than reaching into
    the three liveness signals individually."""

    @pytest.mark.asyncio
    async def test_actively_held_task_skipped_other_processed(
        self, tmp_path: Path,
    ) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        harness = Harness(config)
        harness.scheduler.get_statuses = AsyncMock(  # type: ignore[method-assign]
            return_value=({'held': 'in-progress', 'free': 'in-progress'}, None),
        )

        held_queries: list[str] = []

        def _is_actively_held(tid: str) -> bool:
            held_queries.append(tid)
            return tid == 'held'

        harness.scheduler.is_actively_held = _is_actively_held  # type: ignore[method-assign]

        processed: list[str] = []

        async def _fake_reconcile_one(tid, status, *, mid_run):
            processed.append(tid)
            return None

        harness._reconcile_one_stranded = _fake_reconcile_one  # type: ignore[method-assign]

        cleared: list[str] = []
        harness.scheduler.clear_workflow_cancel = cleared.append  # type: ignore[method-assign]

        await harness._reconcile_stranded_in_progress(mid_run=True)

        assert set(held_queries) == {'held', 'free'}, (
            'is_actively_held must be consulted for every mid-run sweep candidate'
        )
        assert processed == ['free'], (
            'a task for which is_actively_held() is True must be skipped as '
            f'actively held; the other must be processed. processed={processed}'
        )
        assert cleared == ['free'], (
            'clear_workflow_cancel (lazy prune) must run only for tasks that '
            f'pass the is_actively_held check. cleared={cleared}'
        )


class TestRetryCapUsesRequeueHistoryAccessor:
    """(d2) ``_apply_retry_cap`` reads requeue history via
    ``scheduler.requeue_history()``, not ``scheduler._requeue_history``
    directly (task 2235)."""

    @pytest.mark.asyncio
    async def test_apply_retry_cap_reads_via_accessor(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path, requeue_cap=1)
        harness = Harness(config)
        harness._run_id = 'run-test'
        harness._escalation_queue = None
        harness.scheduler.trigger_retry_cap_exhausted = AsyncMock()  # type: ignore[method-assign]

        calls: list[str] = []
        real_requeue_history = harness.scheduler.requeue_history

        def _spy(tid: str):
            calls.append(tid)
            return real_requeue_history(tid)

        harness.scheduler.requeue_history = _spy  # type: ignore[method-assign]

        report = TaskReport(
            task_id='t1',
            title='T1',
            outcome=WorkflowOutcome.REQUEUED,
            cost_usd=1.0,
            steward_cost_usd=0.0,
            block_reason='architect returned empty output',
            block_detail='see iteration log',
            block_phase='plan',
        )

        await harness._apply_retry_cap('t1', report, requeued=True)

        assert calls == ['t1'], (
            '_apply_retry_cap must read requeue history via the public '
            f'scheduler.requeue_history() accessor; spy calls={calls}'
        )


# ---------------------------------------------------------------------------
# (e) Startup: finish_startup() precedes the first acquire_next() (step-11/12)
# ---------------------------------------------------------------------------

class TestFinishStartupCalledBeforeFirstAcquireNext:
    """(e) ``scheduler.finish_startup()`` is invoked after the startup
    reconcile sweeps and strictly before the first ``acquire_next()`` call."""

    @pytest.mark.asyncio
    async def test_finish_startup_precedes_first_acquire_next(
        self, tmp_path: Path, mock_orch_config,
    ) -> None:
        mock_orch_config.max_concurrent_tasks = 2
        mock_orch_config.fused_memory.project_id = 'test'

        with patch('orchestrator.harness.McpLifecycle') as mock_mcp_cls, \
             patch('orchestrator.harness.Scheduler'), \
             patch('orchestrator.harness.BriefingAssembler'):
            h = Harness(mock_orch_config)

        mock_mcp = mock_mcp_cls.return_value
        mock_mcp.start = AsyncMock()
        mock_mcp.stop = AsyncMock()

        h.git_ops = MagicMock()
        h.git_ops.has_dirty_working_tree = AsyncMock(return_value=None)
        h.git_ops.worktree_base = tmp_path / '.worktrees'

        h._start_escalation_server = AsyncMock()  # type: ignore[method-assign]
        h._start_merge_worker = AsyncMock()  # type: ignore[method-assign]
        h._dismiss_stale_escalations = AsyncMock()  # type: ignore[method-assign]
        h._start_orphan_l0_reaper = MagicMock()  # type: ignore[method-assign]
        h._start_terminal_status_watcher = MagicMock()  # type: ignore[method-assign]
        h._tag_task_modules = AsyncMock()  # type: ignore[method-assign]
        h._recover_crashed_tasks = AsyncMock()  # type: ignore[method-assign]
        h._reconcile_lane_checkouts = AsyncMock()  # type: ignore[method-assign]
        h._reconcile_stranded_in_progress = AsyncMock()  # type: ignore[method-assign]
        h._tag_prd_metadata = AsyncMock()  # type: ignore[method-assign]

        h.scheduler = MagicMock()
        h.scheduler.get_tasks = AsyncMock(return_value=[])
        h.scheduler.get_statuses = AsyncMock(return_value=({}, None))
        h.scheduler.set_task_status = AsyncMock()

        order: list[str] = []
        h.scheduler.finish_startup = MagicMock(
            side_effect=lambda: order.append('finish_startup'),
        )

        def _acquire_next_side_effect(*args, **kwargs):
            order.append('acquire_next')
            raise RuntimeError('__loop_reached_sentinel__')

        h.scheduler.acquire_next = AsyncMock(side_effect=_acquire_next_side_effect)

        with pytest.raises(RuntimeError, match='__loop_reached_sentinel__'):
            await h.run()

        assert order == ['finish_startup', 'acquire_next'], (
            'scheduler.finish_startup() must be called after the startup '
            'reconcile sweeps and strictly before the first acquire_next(); '
            f'got {order}'
        )
