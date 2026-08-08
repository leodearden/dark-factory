"""Claimant-field projection through the dashboard's task seams (task 3543 / PRD ι).

PRD ``plans/task-escalation-state-graph-prd.md`` task ι (spec S8/E12, D9): the
strand datum already exists on every MCP ``get_tasks`` row
(``claimant_run_id``/``heartbeat_at``, written by the orchestrator's dispatch
claim), but the dashboard DROPS it at ``_shape_task``.  This module pins the
projection at its two seams:

* :func:`dashboard.data.tasks._shape_task` — carries the two raw columns
  through to the dashboard wire shape.
* :func:`dashboard.data.tasks.task_is_stranded` — the single dashboard-side
  strand predicate, a thin wrapper that binds ``STRANDED_HEARTBEAT_TTL`` and
  ``resolve_now`` onto :func:`shared.task_claimant.is_stranded`.
* :func:`dashboard.data.active_tasks._build_task_row` — stamps the two raw
  columns plus the computed ``stranded`` boolean onto every task row.

The delegation is asserted by patching the shared predicate, so the dashboard's
notion of "stranded" can never silently fork from the shared one (INV-5).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from dashboard.data import active_tasks as active_tasks_mod
from dashboard.data import tasks as tasks_mod
from dashboard.data.active_tasks import _build_task_row
from dashboard.data.tasks import STRANDED_HEARTBEAT_TTL, _shape_task, task_is_stranded

_NOW = datetime(2026, 8, 8, 12, 0, 0, tzinfo=UTC)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _row(**overrides) -> dict:
    """A minimally-valid MCP ``get_tasks`` row."""
    row = {
        'id': '7',
        'title': 'a task',
        'description': '',
        'details': '',
        'status': 'in-progress',
        'priority': 'medium',
        'dependencies': [],
        'metadata': {},
        'updatedAt': _iso(_NOW),
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# (a) _shape_task carries the claimant columns
# ---------------------------------------------------------------------------


class TestShapeTaskCarriesClaimantFields:
    def test_preserves_claimant_run_id_and_heartbeat_at(self):
        shaped = _shape_task(
            _row(claimant_run_id='run-1/sess-1/pid=42', heartbeat_at=_iso(_NOW))
        )

        assert shaped is not None
        assert shaped['claimant_run_id'] == 'run-1/sess-1/pid=42'
        assert shaped['heartbeat_at'] == _iso(_NOW)

    def test_missing_columns_surface_as_none_not_keyerror(self):
        """Pre-migration rows (or an older fused-memory) omit both columns.

        The shaped dict must still carry the keys — a downstream ``.get()``
        returning None is a legible "no claimant", whereas an absent key would
        force every consumer to guard.
        """
        shaped = _shape_task(_row())

        assert shaped is not None
        assert shaped['claimant_run_id'] is None
        assert shaped['heartbeat_at'] is None

    def test_existing_shape_keys_are_unchanged(self):
        shaped = _shape_task(_row(claimant_run_id='x', heartbeat_at=_iso(_NOW)))

        assert shaped is not None
        # The nine pre-existing keys must survive verbatim.
        for key in (
            'id',
            'title',
            'description',
            'details',
            'status',
            'priority',
            'dependencies',
            'metadata',
            'updated_at',
        ):
            assert key in shaped
        assert shaped['id'] == 7
        assert shaped['updated_at'] == _iso(_NOW)


# ---------------------------------------------------------------------------
# (b) task_is_stranded truth table
# ---------------------------------------------------------------------------


class TestTaskIsStrandedTruthTable:
    def test_ttl_is_a_timedelta(self):
        assert isinstance(STRANDED_HEARTBEAT_TTL, timedelta)
        assert timedelta(0) < STRANDED_HEARTBEAT_TTL

    def test_in_progress_with_null_claimant_is_stranded(self):
        task = {'status': 'in-progress', 'claimant_run_id': None, 'heartbeat_at': None}
        assert task_is_stranded(task, now=_NOW) is True

    def test_in_progress_with_blank_claimant_is_stranded(self):
        task = {
            'status': 'in-progress',
            'claimant_run_id': '   ',
            'heartbeat_at': _iso(_NOW),
        }
        assert task_is_stranded(task, now=_NOW) is True

    def test_in_progress_with_fresh_heartbeat_is_not_stranded(self):
        task = {
            'status': 'in-progress',
            'claimant_run_id': 'run/sess/pid=1',
            'heartbeat_at': _iso(_NOW - STRANDED_HEARTBEAT_TTL / 2),
        }
        assert task_is_stranded(task, now=_NOW) is False

    def test_in_progress_with_stale_heartbeat_is_stranded(self):
        task = {
            'status': 'in-progress',
            'claimant_run_id': 'run/sess/pid=1',
            'heartbeat_at': _iso(_NOW - STRANDED_HEARTBEAT_TTL - timedelta(seconds=1)),
        }
        assert task_is_stranded(task, now=_NOW) is True

    def test_in_progress_with_missing_heartbeat_is_stranded(self):
        task = {'status': 'in-progress', 'claimant_run_id': 'run/sess/pid=1'}
        assert task_is_stranded(task, now=_NOW) is True

    def test_in_progress_with_unparseable_heartbeat_is_stranded(self):
        task = {
            'status': 'in-progress',
            'claimant_run_id': 'run/sess/pid=1',
            'heartbeat_at': 'not-a-timestamp',
        }
        assert task_is_stranded(task, now=_NOW) is True

    @pytest.mark.parametrize('status', ['pending', 'blocked', 'done', 'review', 'cancelled'])
    def test_non_in_progress_status_is_never_stranded(self, status):
        """The shared predicate hard-gates on status == 'in-progress'.

        Notably 'review' — which the burndown zone map folds into the
        in_progress zone — can never be stranded, no matter how dead its
        claimant looks.
        """
        task = {'status': status, 'claimant_run_id': None, 'heartbeat_at': None}
        assert task_is_stranded(task, now=_NOW) is False

    def test_infra_hold_metadata_carves_out_of_stranded(self):
        task = {
            'status': 'in-progress',
            'claimant_run_id': None,
            'heartbeat_at': None,
            'metadata': {'infra_hold': True},
        }
        assert task_is_stranded(task, now=_NOW) is False


class TestTaskIsStrandedDelegates:
    """The dashboard predicate must not fork from the shared one."""

    def test_delegates_to_shared_is_stranded_with_bound_ttl(self):
        task = {'status': 'in-progress', 'claimant_run_id': None}

        with patch.object(tasks_mod, 'is_stranded', return_value=True) as shared:
            assert task_is_stranded(task, now=_NOW) is True

        shared.assert_called_once_with(task, _NOW, STRANDED_HEARTBEAT_TTL)

    def test_does_not_use_has_live_claimant(self):
        """``not has_live_claimant(...)`` is a DIFFERENT predicate.

        It carries neither the in-progress status gate nor the infra_hold
        carve-out, so substituting it would over-report strands.  Patching it
        to a wrong answer must not change the wrapper's result.
        """
        task = {'status': 'done', 'claimant_run_id': None, 'heartbeat_at': None}
        with patch.object(
            tasks_mod, 'has_live_claimant', create=True, return_value=False
        ):
            assert task_is_stranded(task, now=_NOW) is False

    def test_none_now_resolves_through_resolve_now(self):
        """Clock discipline: the wrapper reads the clock only via resolve_now."""
        task = {'status': 'in-progress', 'claimant_run_id': None}

        with patch.object(tasks_mod, 'resolve_now', return_value=_NOW) as resolver:
            assert task_is_stranded(task) is True

        resolver.assert_called_once_with(None)

    def test_explicit_now_is_threaded_unchanged(self):
        task = {'status': 'in-progress', 'claimant_run_id': None}
        other = _NOW - timedelta(days=3)

        with patch.object(tasks_mod, 'is_stranded', return_value=False) as shared:
            task_is_stranded(task, now=other)

        assert shared.call_args.args[1] == other


# ---------------------------------------------------------------------------
# (c) active_tasks._build_task_row projects the strand onto every task row
# ---------------------------------------------------------------------------


def _task(**overrides) -> dict:
    """A dashboard-shaped task row as ``_shape_task`` emits it."""
    task = {
        'id': 42,
        'title': 'a task',
        'description': '',
        'details': '',
        'status': 'in-progress',
        'priority': 'medium',
        'dependencies': [],
        'metadata': {},
        'updated_at': _iso(_NOW),
        'claimant_run_id': None,
        'heartbeat_at': None,
    }
    task.update(overrides)
    return task


def _rt(agent: str | None = None) -> dict:
    """A ``_runtime_fields``-shaped dict; ``agent`` is the worktree-presence signal."""
    return {
        'agent': agent,
        'loops': 0,
        'attempts': 0,
        'started': 0,
        'lane': None,
        'phase': None,
        'lane_state': None,
        'runtime_offline': False,
    }


class TestBuildTaskRowStrandProjection:
    def test_row_carries_claimant_fields_and_stranded(self):
        row = _build_task_row(
            'p',
            _task(claimant_run_id='run/sess/pid=1', heartbeat_at=_iso(_NOW)),
            42,
            _rt(),
            'p/T-42',
            now=_NOW,
        )

        assert row['claimant_run_id'] == 'run/sess/pid=1'
        assert row['heartbeat_at'] == _iso(_NOW)
        assert row['stranded'] is False

    def test_worktree_agent_with_null_claimant_is_stranded(self):
        """THE defect this task removes.

        ``agent`` is set purely from ``entry.has_worktree`` — a leftover
        worktree directory, not evidence that anything is running.  A task
        showing ``claude-task-42`` with a dead claimant is exactly the
        strand that previously read as healthy.  The two signals must be
        independent fields: the row keeps its ``agent`` AND reports
        ``stranded=True``.
        """
        row = _build_task_row(
            'p',
            _task(claimant_run_id=None, heartbeat_at=None),
            42,
            _rt(agent='claude-task-42'),
            'p/T-42',
            now=_NOW,
        )

        assert row['agent'] == 'claude-task-42'
        assert row['stranded'] is True

    def test_stranded_is_not_derived_from_agent(self):
        """A live claimant with NO worktree is not stranded."""
        row = _build_task_row(
            'p',
            _task(claimant_run_id='run/sess/pid=1', heartbeat_at=_iso(_NOW)),
            42,
            _rt(agent=None),
            'p/T-42',
            now=_NOW,
        )

        assert row['agent'] is None
        assert row['stranded'] is False

    def test_stale_heartbeat_is_stranded(self):
        row = _build_task_row(
            'p',
            _task(
                claimant_run_id='run/sess/pid=1',
                heartbeat_at=_iso(_NOW - STRANDED_HEARTBEAT_TTL - timedelta(minutes=1)),
            ),
            42,
            _rt(agent='claude-task-42'),
            'p/T-42',
            now=_NOW,
        )

        assert row['stranded'] is True

    @pytest.mark.parametrize('status', ['blocked', 'done', 'pending', 'cancelled'])
    def test_terminal_and_non_in_progress_rows_are_not_stranded(self, status):
        row = _build_task_row(
            'p', _task(status=status), 42, _rt(agent='claude-task-42'), 'p/T-42', now=_NOW
        )

        assert row['stranded'] is False

    def test_missing_claimant_keys_degrade_to_none_not_keyerror(self):
        """A pre-migration row lacks both columns entirely."""
        task = _task()
        del task['claimant_run_id']
        del task['heartbeat_at']

        row = _build_task_row('p', task, 42, _rt(), 'p/T-42', now=_NOW)

        assert row['claimant_run_id'] is None
        assert row['heartbeat_at'] is None
        # No claimant evidence is not evidence of liveness.
        assert row['stranded'] is True

    def test_existing_row_keys_are_unchanged(self):
        row = _build_task_row('p', _task(), 42, _rt(), 'p/T-42', now=_NOW)

        for key in (
            'id', 'project', 'title', 'description', 'details', 'status',
            'agent', 'loops', 'attempts', 'lane', 'phase', 'lane_state',
            'runtime_offline', 'meta_files', 'train', 'external_deps', 'prd',
        ):
            assert key in row


class TestBuildTaskRowClockDiscipline:
    def test_threads_the_caller_supplied_now(self):
        task = _task(claimant_run_id='run/sess/pid=1', heartbeat_at=_iso(_NOW))

        with patch.object(
            active_tasks_mod, 'task_is_stranded', return_value=False
        ) as predicate:
            _build_task_row('p', task, 42, _rt(), 'p/T-42', now=_NOW)

        predicate.assert_called_once_with(task, now=_NOW)

    def test_caller_now_beats_the_live_clock(self):
        """A heartbeat that is fresh relative to the SUPPLIED now — but ancient
        relative to the live clock — must read as live.

        Proves the row's verdict derives from the threaded instant, not from a
        clock read inside the row builder.
        """
        historical = datetime(2020, 1, 1, tzinfo=UTC)
        row = _build_task_row(
            'p',
            _task(claimant_run_id='run/sess/pid=1', heartbeat_at=_iso(historical)),
            42,
            _rt(),
            'p/T-42',
            now=historical + timedelta(seconds=5),
        )

        assert row['stranded'] is False
