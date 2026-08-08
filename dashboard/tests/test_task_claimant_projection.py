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

The delegation is asserted by patching the shared predicate, so the dashboard's
notion of "stranded" can never silently fork from the shared one (INV-5).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from dashboard.data import tasks as tasks_mod
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
        assert STRANDED_HEARTBEAT_TTL > timedelta(0)

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
