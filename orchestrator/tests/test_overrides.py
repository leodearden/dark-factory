"""Tests for the OverrideStore SQLite persistence layer."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest


class TestSchema:
    def test_creates_overrides_table_and_index_with_wal(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        db_path = tmp_path / 'sub' / 'scheduler_overrides.db'
        _store = OverrideStore(db_path)

        # Parent dirs must have been auto-created
        assert db_path.parent.exists()
        assert db_path.exists()

        conn = sqlite3.connect(str(db_path))
        try:
            # Table exists
            names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert 'overrides' in names

            # Index exists
            index_names = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            }
            assert 'idx_overrides_pinned' in index_names

            # WAL mode
            mode = conn.execute('PRAGMA journal_mode').fetchone()[0]
            assert mode == 'wal'
        finally:
            conn.close()


class TestSetGet:
    def test_set_boost_then_get_overrides_returns_row(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideRow, OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'task-A', boost_tier='high')

        result = store.get_overrides('proj')
        assert result == {
            'task-A': OverrideRow(
                boost_tier='high',
                pinned=False,
                pin_order=None,
                reserve_now=False,
                ttl_until=None,
            )
        }

    def test_get_overrides_does_not_return_other_projects(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj-A', 'task-1', boost_tier='low')
        store.set_override('proj-B', 'task-2', boost_tier='critical')

        result_a = store.get_overrides('proj-A')
        assert 'task-1' in result_a
        assert 'task-2' not in result_a

        result_b = store.get_overrides('proj-B')
        assert 'task-2' in result_b
        assert 'task-1' not in result_b

    def test_set_override_rejects_unknown_boost_tier(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        with pytest.raises(ValueError, match='boost_tier'):
            store.set_override('proj', 'A', boost_tier='urgent')

    def test_set_override_accepts_valid_boost_tiers(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        for tier in ('critical', 'high', 'medium', 'low', 'polish'):
            store.set_override('proj', f'task-{tier}', boost_tier=tier)
            result = store.get_overrides('proj')
            assert result[f'task-{tier}'].boost_tier == tier


class TestPinning:
    def test_pin_auto_assigns_pin_order_max_plus_one(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')

        # Pin A and B without explicit pin_order
        store.set_override('proj', 'A', pinned=True)
        store.set_override('proj', 'B', pinned=True)

        overrides = store.get_overrides('proj')
        assert overrides['A'].pin_order == 1
        assert overrides['B'].pin_order == 2

        # Pin C with explicit pin_order=5
        store.set_override('proj', 'C', pinned=True, pin_order=5)
        overrides = store.get_overrides('proj')
        assert overrides['C'].pin_order == 5

        # Pin D without explicit pin_order — should get MAX(5)+1 = 6
        store.set_override('proj', 'D', pinned=True)
        overrides = store.get_overrides('proj')
        assert overrides['D'].pin_order == 6

    def test_pin_order_collision_raises_with_both_ids(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore, PinOrderCollision

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', pinned=True, pin_order=3)

        with pytest.raises(PinOrderCollision) as exc_info:
            store.set_override('proj', 'B', pinned=True, pin_order=3)

        msg = str(exc_info.value)
        assert 'A' in msg
        assert 'B' in msg
        assert '3' in msg

        # Store must be left untouched — B was not written
        overrides = store.get_overrides('proj')
        assert 'B' not in overrides

    def test_repin_without_explicit_order_is_idempotent(self, tmp_path: Path) -> None:
        """Regression: re-pinning an already-pinned task must NOT shift its pin_order.

        The auto-assign query previously included the task's OWN row in MAX(),
        so calling set_override(pinned=True) a second time would shift A from 5 → 6.
        """
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        # A is the sole pinned task at pin_order=5.
        store.set_override('proj', 'A', pinned=True, pin_order=5)

        # Re-pin A (e.g. to also add a boost) — pin_order must NOT change.
        store.set_override('proj', 'A', pinned=True, boost_tier='high')

        overrides = store.get_overrides('proj')
        assert overrides['A'].pin_order == 5, (
            'Re-pinning an already-pinned task must preserve its existing pin_order'
        )
        assert overrides['A'].boost_tier == 'high'

    def test_pin_order_without_pinned_raises(self, tmp_path: Path) -> None:
        """pin_order may only be supplied together with pinned=True."""
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')

        # pin_order with no pinned kwarg at all (None)
        with pytest.raises(ValueError, match='pinned=True'):
            store.set_override('proj', 'A', pin_order=5)

        # pin_order with pinned=False
        with pytest.raises(ValueError, match='pinned=True'):
            store.set_override('proj', 'A', pinned=False, pin_order=5)

        # Store is clean — no rows created
        assert store.get_overrides('proj') == {}

    def test_concurrent_auto_pin_assigns_distinct_pin_orders(self, tmp_path: Path) -> None:
        """Two concurrent set_override(pinned=True) calls must produce {1,2}.

        Pre-fix (no BEGIN IMMEDIATE): both threads can read MAX=0 simultaneously
        and both compute pin_order=1, producing a PinOrderCollision or duplicate
        pin_order values.  Post-fix (BEGIN IMMEDIATE): SQLite serializes the two
        write transactions so they assign 1 and 2 respectively.
        """
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        barrier = threading.Barrier(2)
        errors: list[Exception] = []

        def pin_task(task_id: str) -> None:
            try:
                barrier.wait()  # both threads enter set_override at the same time
                store.set_override('proj', task_id, pinned=True)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        t1 = threading.Thread(target=pin_task, args=('A',))
        t2 = threading.Thread(target=pin_task, args=('B',))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == [], f'Unexpected exceptions: {errors}'

        overrides = store.get_overrides('proj')
        assert 'A' in overrides, 'Task A must be present after concurrent pin'
        assert 'B' in overrides, 'Task B must be present after concurrent pin'

        pin_orders = {overrides['A'].pin_order, overrides['B'].pin_order}
        assert pin_orders == {1, 2}, (
            f'Expected distinct pin_orders {{1, 2}}; got {pin_orders}'
        )

    def test_set_override_pinned_false_clears_pin_order(self, tmp_path: Path) -> None:
        """set_override(pinned=False) must also zero out pin_order.

        Docstring contract: 'Passing pinned=False is an explicit write that
        zeroes pinned AND pin_order.'  Pre-fix the COALESCE leaves a stale
        pin_order on the row when no pin_order kwarg is supplied.
        """
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')

        # Setup: pin A with an explicit pin_order
        store.set_override('proj', 'A', pinned=True, pin_order=2)
        overrides = store.get_overrides('proj')
        assert overrides['A'].pinned is True
        assert overrides['A'].pin_order == 2

        # Action: un-pin without supplying a pin_order
        store.set_override('proj', 'A', pinned=False)

        # Assertion: pinned=False AND pin_order=None
        overrides = store.get_overrides('proj')
        assert overrides['A'].pinned is False
        assert overrides['A'].pin_order is None, (
            'pin_order must be cleared when pinned=False is set explicitly'
        )


class TestPinQueue:
    def test_get_pin_queue_returns_pinned_rows_in_pin_order_asc(
        self, tmp_path: Path
    ) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', pinned=True, pin_order=3)
        store.set_override('proj', 'B', pinned=True, pin_order=1)
        store.set_override('proj', 'C', pinned=True, pin_order=2)
        # D is boost-only, not pinned
        store.set_override('proj', 'D', boost_tier='high')

        queue = store.get_pin_queue('proj')
        task_ids = [tid for tid, _ in queue]
        assert task_ids == ['B', 'C', 'A']

        # D must not appear in the pin queue
        assert all(tid != 'D' for tid, _ in queue)


class TestClear:
    def test_clear_override_none_deletes_row(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', boost_tier='high', pinned=True)

        result = store.clear_override('proj', 'A')
        assert result is True
        assert 'A' not in store.get_overrides('proj')

    def test_clear_override_field_boost_tier_keeps_row(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', boost_tier='high', pinned=True)

        result = store.clear_override('proj', 'A', field='boost_tier')
        assert result is True

        overrides = store.get_overrides('proj')
        assert 'A' in overrides
        assert overrides['A'].boost_tier is None
        assert overrides['A'].pinned is True

    def test_clear_override_field_pinned_clears_pin_order_too(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', pinned=True, pin_order=2)

        store.clear_override('proj', 'A', field='pinned')

        overrides = store.get_overrides('proj')
        assert 'A' in overrides
        assert overrides['A'].pinned is False
        assert overrides['A'].pin_order is None

    def test_clear_override_missing_row_returns_false(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        result = store.clear_override('proj', 'nonexistent')
        assert result is False


class TestReorder:
    def test_reorder_pin_queue_rewrites_pin_order(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', pinned=True, pin_order=1)
        store.set_override('proj', 'B', pinned=True, pin_order=2)
        store.set_override('proj', 'C', pinned=True, pin_order=3)

        store.reorder_pin_queue('proj', ['C', 'A', 'B'])

        queue = store.get_pin_queue('proj')
        task_ids = [tid for tid, _ in queue]
        assert task_ids == ['C', 'A', 'B']

    def test_reorder_pin_queue_accepts_csv_string(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', pinned=True, pin_order=1)
        store.set_override('proj', 'B', pinned=True, pin_order=2)
        store.set_override('proj', 'C', pinned=True, pin_order=3)

        store.reorder_pin_queue('proj', 'C,A,B')

        queue = store.get_pin_queue('proj')
        task_ids = [tid for tid, _ in queue]
        assert task_ids == ['C', 'A', 'B']

    def test_reorder_pin_queue_rejects_missing_or_extra_ids(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', pinned=True, pin_order=1)
        store.set_override('proj', 'B', pinned=True, pin_order=2)
        store.set_override('proj', 'C', pinned=True, pin_order=3)

        # Omitting B
        with pytest.raises(ValueError):
            store.reorder_pin_queue('proj', ['C', 'A'])

        # Including a non-pinned / unknown task
        with pytest.raises(ValueError):
            store.reorder_pin_queue('proj', ['C', 'A', 'B', 'D'])

    def test_reorder_pin_queue_rejects_duplicate_ids(self, tmp_path: Path) -> None:
        """Duplicate task IDs in the input must raise ValueError before writing.

        Pre-fix: ['A', 'A', 'B'] passes set-equality against {A,B} (wrong set)
        and writes A→1, A→2 silently.  The duplicate check must fire BEFORE the
        set-equality check so the error message clearly says 'duplicate'.
        """
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', pinned=True, pin_order=1)
        store.set_override('proj', 'B', pinned=True, pin_order=2)
        store.set_override('proj', 'C', pinned=True, pin_order=3)

        # List form: duplicate A, only 3 elements so set-equality would be
        # {A,B} != {A,B,C} — but the duplicate check must fire first
        with pytest.raises(ValueError, match='duplicate'):
            store.reorder_pin_queue('proj', ['A', 'A', 'B'])

        # CSV form: same check via string path
        with pytest.raises(ValueError, match='duplicate'):
            store.reorder_pin_queue('proj', 'A,A,B')

        # After both raises, pin_order values must be unchanged
        queue = store.get_pin_queue('proj')
        task_ids = [tid for tid, _ in queue]
        assert task_ids == ['A', 'B', 'C'], 'pin_order must be unchanged after rejected reorder'


class TestSweeps:
    def test_clear_terminal_deletes_named_owners(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        store.set_override('proj', 'A', boost_tier='high')
        store.set_override('proj', 'B', boost_tier='medium')
        store.set_override('proj', 'C', boost_tier='low')

        cleared = store.clear_terminal('proj', {'A', 'C'})
        assert sorted(cleared) == ['A', 'C']

        remaining = store.get_overrides('proj')
        assert list(remaining.keys()) == ['B']

    def test_clear_expired_deletes_past_ttl(self, tmp_path: Path) -> None:
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        now = datetime(2026, 5, 14, tzinfo=UTC)
        past = now - timedelta(hours=1)
        future = now + timedelta(hours=1)

        store.set_override('proj', 'A', ttl_until=past)
        store.set_override('proj', 'B', ttl_until=future)
        store.set_override('proj', 'C')  # no TTL

        cleared = store.clear_expired('proj', now)
        assert cleared == ['A']

        remaining = store.get_overrides('proj')
        assert 'A' not in remaining
        assert 'B' in remaining
        assert 'C' in remaining

    def test_clear_expired_handles_non_utc_ttl(self, tmp_path: Path) -> None:
        """TTLs stored with non-UTC offsets must still expire correctly.

        Pre-fix: ttl_until is stored verbatim as ISO string, so lexicographic
        compare of "2026-05-14T05:00:00+05:00" vs "2026-05-14T01:00:00+00:00"
        incorrectly puts the TTL *after* now (05 > 01).

        Post-fix: set_override normalises to UTC before storing, so the stored
        value is "2026-05-14T00:00:00+00:00" and clear_expired correctly sees
        it has expired when now = 01:00 UTC.
        """
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        # +05:00 time that is 00:00 UTC absolute (i.e. already expired vs 01:00 UTC)
        ttl_plus5 = datetime(2026, 5, 14, 5, 0, tzinfo=timezone(timedelta(hours=5)))
        now_utc = datetime(2026, 5, 14, 1, 0, tzinfo=UTC)  # 1 hour later in absolute time

        store.set_override('proj', 'A', ttl_until=ttl_plus5)
        cleared = store.clear_expired('proj', now_utc)
        assert cleared == ['A'], (
            'Entry with ttl_plus5 (==00:00 UTC absolute) must be cleared when now=01:00 UTC'
        )

    def test_set_override_rejects_naive_ttl(self, tmp_path: Path) -> None:
        """set_override must reject naive (tz-unaware) ttl_until datetimes."""
        from orchestrator.overrides import OverrideStore

        store = OverrideStore(tmp_path / 'scheduler_overrides.db')
        with pytest.raises(ValueError, match='timezone'):
            store.set_override('proj', 'B', ttl_until=datetime(2026, 5, 14))


class TestSeparateDB:
    def test_override_survives_simulated_set_task_status_cycle(
        self, tmp_path: Path
    ) -> None:
        """Regression guard: override rows survive unrelated DB writes.

        This test encodes the invariant that overrides live in a *separate*
        SQLite file from taskmaster state.  If someone ever consolidates the
        storage, this test will fail and force a conscious decision.
        """
        from orchestrator.overrides import OverrideRow, OverrideStore

        db_path = tmp_path / 'scheduler_overrides.db'
        store = OverrideStore(db_path)
        store.set_override('proj', 'A', boost_tier='high', pinned=True)

        # Simulate what set_task_status / a reify cycle does: create or
        # overwrite an unrelated SQLite file in the same directory.
        taskmaster_db = tmp_path / 'taskmaster.db'
        import sqlite3

        conn = sqlite3.connect(str(taskmaster_db))
        conn.execute('CREATE TABLE tasks (id TEXT PRIMARY KEY, status TEXT)')
        conn.execute("INSERT INTO tasks VALUES ('A', 'in_progress')")
        conn.commit()
        conn.close()

        # The override must still be intact.  Re-open from the same path to
        # exercise the connect-per-call pattern rather than any cached state.
        store2 = OverrideStore(db_path)
        overrides = store2.get_overrides('proj')

        assert 'A' in overrides
        row = overrides['A']
        assert row == OverrideRow(
            boost_tier='high',
            pinned=True,
            pin_order=1,
            reserve_now=False,
            ttl_until=None,
        )
