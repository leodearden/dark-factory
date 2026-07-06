"""Tests for LandedOutbox, LandedRow, and MergeProvenance (task 2153 / W1 α).

Covers:
  step-1:  record() + re-open (simulated restart) + lookup() round-trip;
           lookup() of an unknown task_id returns None; LandedRow is frozen.
  step-3:  all() returns every landed row; record() is last-write-wins by
           task_id (WA-2); the overwrite persists across a re-open.
  step-5:  consume() idempotently prunes a row; a pruned row does not
           resurrect after a re-open (durable prune).
  step-7:  record() fsyncs both the row file and its parent directory before
           returning (WA-1) — executable proxy for crash-durability.
  step-9:  MergeProvenance.lookup() is a fail-safe façade over a bound
           LandedOutbox; returns None when unbound.
  step-11: SpeculativeMergeWorker constructs and holds a LandedOutbox
           (None-safe when git_ops has no project_root) and binds it to
           MergeProvenance.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from orchestrator.landed_outbox import LandedOutbox, LandedRow

# ---------------------------------------------------------------------------
# step-1 — record() + lookup() round-trip across a simulated restart
# ---------------------------------------------------------------------------


class TestLandedOutboxRecordLookupRoundtrip:
    """record() persists a row that survives a simulated restart (re-open)."""

    def test_record_survives_reopen_and_lookup(self, tmp_path: Path) -> None:
        path = tmp_path / 'data' / 'orchestrator' / 'landed_outbox.json'
        outbox = LandedOutbox(path)
        row = LandedRow(task_id='42', branch_tip_sha='aaa', advanced_sha='bbb', landed_at=123.0)

        outbox.record(row)

        # Simulated restart: a SECOND LandedOutbox instance at the same path
        # must observe the row purely from disk.
        reopened = LandedOutbox(path)
        looked_up = reopened.lookup('42')
        assert looked_up == row, f'Expected {row!r} to round-trip; got {looked_up!r}'

    def test_lookup_unknown_task_id_returns_none(self, tmp_path: Path) -> None:
        outbox = LandedOutbox(tmp_path / 'landed_outbox.json')
        assert outbox.lookup('nope') is None

    def test_landed_row_is_frozen(self) -> None:
        row = LandedRow(task_id='1', branch_tip_sha='a', advanced_sha='b', landed_at=1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            row.task_id = '2'  # type: ignore[misc]
