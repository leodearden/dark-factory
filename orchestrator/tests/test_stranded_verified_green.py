"""Tests for the verified-green stranded remediation detector.

Covers ``orchestrator.stranded_verified_green`` — the pure helpers and the
async ``detect_verified_green`` shape check that the stranded-blocked reaper
consults before submitting a lane branch directly to the merge queue
(stranding-remediation-scheduler-ergonomics-prd.md leaf α, §2.1).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.event_store import EventStore, EventType


def _emit_verify(
    store: EventStore, task_id: str, *, passed: bool, tip_sha: str | None,
) -> None:
    """Emit a single workflow_verify row (mirrors workflow.py:2084 shape)."""
    data: dict = {'passed': passed}
    if tip_sha is not None:
        data['tip_sha'] = tip_sha
    store.emit(EventType.workflow_verify, task_id=task_id, data=data)


class TestLastVerifiedGreenTip:
    """Unit tests for ``last_verified_green_tip(event_store, task_id)``."""

    def test_latest_passed_with_tip_wins(self, tmp_path: Path) -> None:
        """The LATEST passed row carrying a tip_sha wins over an earlier one."""
        from orchestrator.stranded_verified_green import last_verified_green_tip

        store = EventStore(tmp_path / 'runs.db', 'run-1')
        _emit_verify(store, '7', passed=True, tip_sha='aaa111')
        _emit_verify(store, '7', passed=True, tip_sha='bbb222')

        assert last_verified_green_tip(store, '7') == 'bbb222'

    def test_later_failed_row_does_not_erase_earlier_green(
        self, tmp_path: Path,
    ) -> None:
        """A later FAILED re-verify does not erase the latest passed-with-tip."""
        from orchestrator.stranded_verified_green import last_verified_green_tip

        store = EventStore(tmp_path / 'runs.db', 'run-1')
        _emit_verify(store, '7', passed=True, tip_sha='green9')
        _emit_verify(store, '7', passed=False, tip_sha='red9')

        assert last_verified_green_tip(store, '7') == 'green9'

    def test_later_passed_without_tip_falls_back_to_earlier_with_tip(
        self, tmp_path: Path,
    ) -> None:
        """A later passed row lacking a tip_sha does not shadow an earlier
        passed-WITH-tip: the latest passed-WITH-tip wins."""
        from orchestrator.stranded_verified_green import last_verified_green_tip

        store = EventStore(tmp_path / 'runs.db', 'run-1')
        _emit_verify(store, '7', passed=True, tip_sha='withtip')
        _emit_verify(store, '7', passed=True, tip_sha=None)

        assert last_verified_green_tip(store, '7') == 'withtip'

    def test_none_when_no_passed_row_carries_a_tip(self, tmp_path: Path) -> None:
        """passed rows without a tip_sha (and empty-string tips) yield None."""
        from orchestrator.stranded_verified_green import last_verified_green_tip

        store = EventStore(tmp_path / 'runs.db', 'run-1')
        _emit_verify(store, '7', passed=True, tip_sha=None)
        _emit_verify(store, '7', passed=True, tip_sha='')
        _emit_verify(store, '7', passed=False, tip_sha='red')

        assert last_verified_green_tip(store, '7') is None

    def test_none_when_no_rows(self, tmp_path: Path) -> None:
        """No workflow_verify rows for the task → None."""
        from orchestrator.stranded_verified_green import last_verified_green_tip

        store = EventStore(tmp_path / 'runs.db', 'run-1')
        _emit_verify(store, 'other', passed=True, tip_sha='x')

        assert last_verified_green_tip(store, '7') is None

    def test_none_when_event_store_none(self) -> None:
        """event_store=None → None (fail-safe)."""
        from orchestrator.stranded_verified_green import last_verified_green_tip

        assert last_verified_green_tip(None, '7') is None

    def test_none_when_task_id_none(self, tmp_path: Path) -> None:
        """task_id=None → None (fail-safe)."""
        from orchestrator.stranded_verified_green import last_verified_green_tip

        store = EventStore(tmp_path / 'runs.db', 'run-1')
        assert last_verified_green_tip(store, None) is None

    def test_reads_cross_run(self, tmp_path: Path) -> None:
        """A prior-run green is visible via fetch_events_by_type_all_runs.

        The strand can span an orchestrator restart, so the green may live
        under a PRIOR run_id — a run-scoped read would miss it.
        """
        from orchestrator.stranded_verified_green import last_verified_green_tip

        db_path = tmp_path / 'runs.db'
        prior = EventStore(db_path, 'run-1')
        _emit_verify(prior, '7', passed=True, tip_sha='priorgreen')

        current = EventStore(db_path, 'run-2')
        assert last_verified_green_tip(current, '7') == 'priorgreen'
