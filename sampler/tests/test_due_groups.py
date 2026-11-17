"""Due-time groups (PRD D4): the store's group records and the tick that drives them."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from sampler.metrics import SampleKey

ROW = SampleKey('occt_queue_depth')
GROUP = 'consumers'
OTHER = 'verify_scopes'


# ---------------------------------------------------------------------------
# The store's connection seam, observed
# ---------------------------------------------------------------------------


class _Connection:
    """A real connection whose ``commit`` can be made to raise.

    A proxy, because ``sqlite3.Connection`` is a C type whose ``commit`` cannot
    be replaced on an instance. Closing it without a commit rolls back, which is
    exactly what a process dying mid-tick does to an open transaction.
    """

    def __init__(self, conn: sqlite3.Connection, spy: Spy):
        self._conn = conn
        self._spy = spy

    def commit(self) -> None:
        if self._spy.commit_fails:
            raise sqlite3.OperationalError('injected: commit failed')
        self._conn.commit()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


class Spy:
    """An injectable ``connect`` over ``connect_durably`` that counts what it opens."""

    def __init__(self) -> None:
        self.opened = 0
        self.commit_fails = False

    def __call__(self, db_path: Path, *, autocommit: bool = False) -> Any:
        from sampler.store import connect_durably

        self.opened += 1
        return _Connection(connect_durably(db_path, autocommit=autocommit), self)


def _store(tmp_path: Path, connect: Any = None):
    from sampler.store import LoadSampleStore

    db_path = tmp_path / 'db.sqlite'
    if connect is None:
        return LoadSampleStore(db_path)
    return LoadSampleStore(db_path, connect=connect)


def _rows(db_path: Path) -> list[tuple]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute('SELECT ts, metric, leaf, value FROM samples ORDER BY ts').fetchall()
    finally:
        conn.close()


def _meta_keys(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {row[0] for row in conn.execute('SELECT key FROM meta')}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Store: group records travel with the tick's rows
# ---------------------------------------------------------------------------


class TestGroupRecordsAreWrittenWithTheTick:
    def test_a_fresh_store_reads_back_what_the_tick_wrote(self, tmp_path: Path):
        from sampler.store import GroupRecord

        _store(tmp_path).write_tick(
            100,
            unwindowed={ROW: 1.0},
            windowed={},
            records={GROUP: GroupRecord(attempted_at=100, state='{"v": 1}', log_throttle='{}')},
        )

        records = _store(tmp_path).read_group_records([GROUP])

        assert records[GROUP] == GroupRecord(
            attempted_at=100, state='{"v": 1}', log_throttle='{}'
        )
        assert _rows(tmp_path / 'db.sqlite') == [(100, 'occt_queue_depth', None, 1.0)]

    def test_a_name_never_written_reads_as_an_empty_record(self, tmp_path: Path):
        from sampler.store import GroupRecord

        records = _store(tmp_path).read_group_records([GROUP, OTHER])

        assert records == {GROUP: GroupRecord(), OTHER: GroupRecord()}

    def test_a_field_left_none_leaves_the_stored_value_unchanged(self, tmp_path: Path):
        from sampler.store import GroupRecord

        store = _store(tmp_path)
        store.write_tick(
            100,
            unwindowed={},
            windowed={},
            records={GROUP: GroupRecord(attempted_at=100, state='first', log_throttle='quiet')},
        )
        store.write_tick(130, unwindowed={}, windowed={}, records={GROUP: GroupRecord(attempted_at=130)})

        assert store.read_group_records([GROUP])[GROUP] == GroupRecord(
            attempted_at=130, state='first', log_throttle='quiet'
        )

    def test_a_tick_with_no_rows_still_writes_its_records(self, tmp_path: Path):
        from sampler.store import GroupRecord

        store = _store(tmp_path)
        store.write_tick(100, unwindowed={}, windowed={}, records={GROUP: GroupRecord(attempted_at=100)})

        assert store.read_group_records([GROUP])[GROUP].attempted_at == 100
        assert _rows(tmp_path / 'db.sqlite') == []

    def test_records_of_two_groups_do_not_mix(self, tmp_path: Path):
        from sampler.store import GroupRecord

        store = _store(tmp_path)
        store.write_tick(
            100,
            unwindowed={},
            windowed={},
            records={
                GROUP: GroupRecord(attempted_at=100, state='mine'),
                OTHER: GroupRecord(attempted_at=90, state='theirs'),
            },
        )

        records = store.read_group_records([GROUP, OTHER])

        assert records[GROUP] == GroupRecord(attempted_at=100, state='mine')
        assert records[OTHER] == GroupRecord(attempted_at=90, state='theirs')

    def test_existing_callers_without_records_keep_working(self, tmp_path: Path):
        from sampler.store import GroupRecord

        store = _store(tmp_path)
        store.write_tick(100, unwindowed={ROW: 2.0}, windowed={})

        assert _rows(tmp_path / 'db.sqlite') == [(100, 'occt_queue_depth', None, 2.0)]
        assert store.read_group_records([GROUP]) == {GROUP: GroupRecord()}


class TestATickIsAtomicAcrossRowsAndRecords:
    def test_a_failed_commit_leaves_neither_rows_nor_records(self, tmp_path: Path):
        from sampler.store import GroupRecord

        spy = Spy()
        store = _store(tmp_path, spy)
        before = _meta_keys(tmp_path / 'db.sqlite')
        spy.commit_fails = True

        with pytest.raises(sqlite3.OperationalError, match='injected'):
            store.write_tick(
                100,
                unwindowed={ROW: 1.0},
                windowed={},
                records={GROUP: GroupRecord(attempted_at=100, state='lost')},
            )

        assert _rows(tmp_path / 'db.sqlite') == []
        assert _meta_keys(tmp_path / 'db.sqlite') == before
        spy.commit_fails = False
        assert store.read_group_records([GROUP]) == {GROUP: GroupRecord()}


class TestReadingRecordsIsOneConnection:
    def test_every_name_is_read_on_one_connection(self, tmp_path: Path):
        spy = Spy()
        store = _store(tmp_path, spy)
        spy.opened = 0

        store.read_group_records([GROUP, OTHER, 'load'])

        assert spy.opened == 1


class TestIsGroupDue:
    DUE = 30

    def _attempted_at(self, tmp_path: Path, ts: int):
        from sampler.store import GroupRecord

        store = _store(tmp_path)
        store.write_tick(ts, unwindowed={}, windowed={}, records={GROUP: GroupRecord(attempted_at=ts)})
        return store

    def test_a_group_never_attempted_is_due(self, tmp_path: Path):
        assert _store(tmp_path).is_group_due(GROUP, 1_000, self.DUE)

    def test_inside_the_interval_it_is_not_due(self, tmp_path: Path):
        store = self._attempted_at(tmp_path, 1_000)

        assert not store.is_group_due(GROUP, 1_000, self.DUE)
        assert not store.is_group_due(GROUP, 1_029, self.DUE)

    def test_at_exactly_the_interval_it_is_due(self, tmp_path: Path):
        assert self._attempted_at(tmp_path, 1_000).is_group_due(GROUP, 1_030, self.DUE)

    def test_a_stamp_in_the_future_is_due(self, tmp_path: Path):
        assert self._attempted_at(tmp_path, 5_000).is_group_due(GROUP, 1_000, self.DUE)

    def test_another_groups_stamp_does_not_count(self, tmp_path: Path):
        assert self._attempted_at(tmp_path, 1_000).is_group_due(OTHER, 1_010, self.DUE)
