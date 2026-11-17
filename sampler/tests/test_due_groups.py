"""Due-time groups (PRD D4): the store's group records and the tick that drives them."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from sampler.metrics import GroupName, MetricSpec, SampleKey

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


# ---------------------------------------------------------------------------
# Tick: the group contract
# ---------------------------------------------------------------------------

DUE = 30
T0 = 1_000_000
INCREMENT = MetricSpec('fixture_counter_increment')
LEAF_FIGURE = MetricSpec('fixture_leaf_figure', per_leaf=True)


def _due_group(collect: Any, *, due_seconds: int | None = DUE, logs_rows: bool = True):
    from sampler.metrics import MetricGroup

    return MetricGroup(
        GroupName.LOAD,
        collect=collect,
        metrics=(INCREMENT, LEAF_FIGURE),
        windowed=False,
        due_seconds=due_seconds,
        logs_rows=logs_rows,
    )


def _increments(db_path: Path) -> list[tuple[int, float]]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            'SELECT ts, value FROM samples WHERE metric = ? ORDER BY ts', (INCREMENT.name,)
        ).fetchall()
    finally:
        conn.close()


def _tick(store, now: int, groups):
    from sampler.sampler import collect_tick, run_tick

    return run_tick(store, now, collect_tick(store, now, groups))


class _Recording:
    """A stateful collector that records what it was handed and returns a scripted state."""

    def __init__(self, *, samples: Any = None, raises: list[BaseException] | None = None):
        self.calls: list[tuple[int, str | None]] = []
        self.samples = {} if samples is None else samples
        self.raises = list(raises or [])

    def __call__(self, now: int, previous: str | None):
        from sampler.metrics import Collected

        self.calls.append((now, previous))
        if self.raises:
            raise self.raises.pop(0)
        return Collected(self.samples, state=f'state@{now}')


class _Counter:
    """A cumulative counter recorded as increments against the collector's own state."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __call__(self, now: int, previous: str | None):
        import json

        from sampler.metrics import Collected

        state = json.dumps(self.value)
        if previous is None:
            return Collected({}, state=state)
        return Collected({INCREMENT.key(): float(self.value - json.loads(previous))}, state=state)


class TestTheGroupContract:
    @pytest.mark.parametrize('due_seconds', [0, -30])
    def test_due_seconds_must_be_positive(self, due_seconds: int):
        from sampler.metrics import every_tick

        with pytest.raises(ValueError, match='due_seconds') as raised:
            _due_group(every_tick(dict), due_seconds=due_seconds)
        assert str(GroupName.LOAD) in str(raised.value)
        assert str(due_seconds) in str(raised.value)

    def test_a_due_group_must_be_unwindowed(self):
        from sampler.metrics import MetricGroup, every_tick

        with pytest.raises(ValueError, match='windowed') as raised:
            MetricGroup(
                GroupName.LOAD,
                collect=every_tick(dict),
                metrics=(INCREMENT,),
                windowed=True,
                due_seconds=DUE,
            )
        assert str(GroupName.LOAD) in str(raised.value)
        assert str(DUE) in str(raised.value)

    def test_an_every_tick_group_may_be_windowed_and_logs_its_rows_by_default(self):
        from sampler.metrics import MetricGroup, every_tick

        group = MetricGroup(
            GroupName.LOAD, collect=every_tick(dict), metrics=(INCREMENT,), windowed=True
        )

        assert group.due_seconds is None
        assert group.logs_rows is True

    def test_every_tick_adapts_a_zero_argument_collector(self):
        from sampler.metrics import Collected, every_tick

        collect = every_tick(lambda: {INCREMENT.key(): 1.0})

        assert collect(T0, 'whatever came before') == Collected({INCREMENT.key(): 1.0}, state=None)


class TestCollectTickHonoursEachGroupsCadence:
    def test_an_every_tick_group_is_collected_on_every_call(self, tmp_path: Path):
        collector = _Recording()
        group = _due_group(collector, due_seconds=None)
        store = _store(tmp_path)

        _tick(store, T0, [group])
        _tick(store, T0 + 1, [group])
        _tick(store, T0 + 2, [group])

        assert [now for now, _previous in collector.calls] == [T0, T0 + 1, T0 + 2]

    def test_a_due_group_waits_out_its_interval(self, tmp_path: Path):
        from sampler.metrics import Collected
        from sampler.sampler import Skipped, collect_tick

        group = _due_group(_Recording())
        store = _store(tmp_path)

        assert isinstance(collect_tick(store, T0, [group])[group], Collected)
        _tick(store, T0, [group])
        assert collect_tick(store, T0 + DUE - 1, [group]) == {group: Skipped.NOT_DUE}
        assert isinstance(collect_tick(store, T0 + DUE, [group])[group], Collected)

    def test_the_collector_receives_the_state_the_last_written_tick_persisted(self, tmp_path: Path):
        collector = _Recording()
        group = _due_group(collector)
        store = _store(tmp_path)

        _tick(store, T0, [group])
        _tick(store, T0 + DUE, [group])

        assert collector.calls == [(T0, None), (T0 + DUE, f'state@{T0}')]


class TestRunTickPersistsEachGroupsRecord:
    def test_state_rows_and_stamp_land_together(self, tmp_path: Path):
        from sampler.store import GroupRecord

        group = _due_group(_Recording(samples={INCREMENT.key(): 4.0}))
        store = _store(tmp_path)

        _tick(store, T0, [group])

        assert _increments(tmp_path / 'db.sqlite') == [(T0, 4.0)]
        assert store.read_group_records([GroupName.LOAD]) == {
            GroupName.LOAD: GroupRecord(attempted_at=T0, state=f'state@{T0}')
        }

    def test_a_group_dropped_at_admission_keeps_its_state_but_advances_its_stamp(
        self, tmp_path: Path, caplog
    ):
        from sampler.sampler import Skipped

        store = _store(tmp_path)
        _tick(store, T0, [_due_group(_Recording())])
        bad = _due_group(_Recording(samples={SampleKey('fixture_misspelt'): 1.0}))

        with caplog.at_level(logging.ERROR):
            dispositions = _tick(store, T0 + DUE, [bad])

        assert dispositions == {bad: Skipped.DROPPED}
        record = store.read_group_records([GroupName.LOAD])[GroupName.LOAD]
        assert record.state == f'state@{T0}'
        assert record.attempted_at == T0 + DUE
        assert _increments(tmp_path / 'db.sqlite') == []
        (error,) = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert str(GroupName.LOAD) in error
        assert 'fixture_misspelt' in error


def _tracebacks(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.ERROR and r.exc_info]


class TestAFailedCollection:
    def test_it_writes_nothing_keeps_its_state_and_advances_its_stamp(self, tmp_path: Path):
        from sampler.sampler import Failed

        db_path = tmp_path / 'db.sqlite'
        _tick(_store(tmp_path), T0, [_due_group(_Recording(samples={INCREMENT.key(): 1.0}))])
        failing = _due_group(_Recording(raises=[RuntimeError('cgroupfs went away')]))

        dispositions = _tick(_store(tmp_path), T0 + DUE, [failing])

        (outcome,) = dispositions.values()
        assert isinstance(outcome, Failed) and outcome.kind == 'RuntimeError'
        assert _increments(db_path) == [(T0, 1.0)]
        record = _store(tmp_path).read_group_records([GroupName.LOAD])[GroupName.LOAD]
        assert record.state == f'state@{T0}'
        assert record.attempted_at == T0 + DUE

    def test_its_traceback_is_logged_once_per_kind_per_hour_across_processes(
        self, tmp_path: Path, caplog
    ):
        def fail_with(exc: BaseException, now: int) -> int:
            caplog.clear()
            with caplog.at_level(logging.ERROR):
                _tick(_store(tmp_path), now, [_due_group(_Recording(raises=[exc]))])
            return len(_tracebacks(caplog))

        assert fail_with(RuntimeError('first'), T0) == 1
        assert str(GroupName.LOAD) in _tracebacks(caplog)[0].getMessage()
        assert fail_with(RuntimeError('again'), T0 + 30) == 0
        assert fail_with(OSError('another kind'), T0 + 60) == 1
        assert fail_with(RuntimeError('still'), T0 + 3600 - DUE) == 0
        assert fail_with(RuntimeError('an hour on'), T0 + 3600) == 1

    def test_the_throttle_round_trips_and_a_corrupt_one_reads_empty(self):
        from sampler.sampler import LogThrottle

        throttle = LogThrottle().logged('RuntimeError', T0)

        assert LogThrottle.from_json(throttle.to_json()) == throttle
        assert not throttle.should_log('RuntimeError', T0 + 10)
        assert throttle.should_log('OSError', T0 + 10)
        assert LogThrottle.from_json('{not json') == LogThrottle()
        assert LogThrottle.from_json(None) == LogThrottle()


class TestACrashMidTickLosesNothingAndCountsNothingTwice:
    """PRD §7: rows and state commit together, so a lost tick is re-spanned, not dropped."""

    def test_a_process_that_dies_before_writing(self, tmp_path: Path):
        from sampler.sampler import collect_tick

        counter = _Counter(100)
        group = _due_group(counter)

        _tick(_store(tmp_path), T0, [group])
        counter.value = 130
        collect_tick(_store(tmp_path), T0 + DUE, [group])
        counter.value = 175
        _tick(_store(tmp_path), T0 + 2 * DUE, [group])

        assert sum(value for _ts, value in _increments(tmp_path / 'db.sqlite')) == 175 - 100

    def test_a_tick_whose_commit_fails(self, tmp_path: Path):
        counter = _Counter(100)
        group = _due_group(counter)
        spy = Spy()
        store = _store(tmp_path, spy)

        _tick(store, T0, [group])
        counter.value = 130
        spy.commit_fails = True
        with pytest.raises(sqlite3.OperationalError, match='injected'):
            _tick(store, T0 + DUE, [group])
        spy.commit_fails = False
        counter.value = 175
        _tick(store, T0 + 2 * DUE, [group])

        assert sum(value for _ts, value in _increments(tmp_path / 'db.sqlite')) == 175 - 100


# ---------------------------------------------------------------------------
# Tick: the journal line
# ---------------------------------------------------------------------------


def _main(monkeypatch, tmp_path: Path, groups, caplog) -> str:
    import sampler.__main__ as entry

    monkeypatch.setenv('DARK_FACTORY_ROOT', str(tmp_path))
    caplog.clear()
    with caplog.at_level(logging.INFO):
        entry.main(groups=groups)
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith('tick ')]
    return line


class TestTheTickLine:
    LEAVES = ('service@orchestrator-reify.service', 'verify@x-1234abcd', 'kernel')

    def _summarised(self):
        samples = {LEAF_FIGURE.key(leaf): 1.0 for leaf in self.LEAVES}
        return _due_group(_Recording(samples={**samples, INCREMENT.key(): 2.0}), logs_rows=False)

    def test_a_group_that_does_not_log_its_rows_renders_one_summary_token(
        self, monkeypatch, tmp_path: Path, caplog
    ):
        line = _main(monkeypatch, tmp_path, [self._summarised()], caplog)

        assert f'{GroupName.LOAD}=rows:4 leaves:3' in line, line
        for leaf in self.LEAVES:
            assert leaf not in line, line
        assert LEAF_FIGURE.name not in line, line

    def test_a_group_that_logs_its_rows_renders_them(self, monkeypatch, tmp_path: Path, caplog):
        from sampler.metrics import every_tick

        group = _due_group(every_tick(lambda: {INCREMENT.key(): 2.0}), due_seconds=None)

        line = _main(monkeypatch, tmp_path, [group], caplog)

        assert INCREMENT.name in line, line

    def test_a_group_not_yet_due_renders_not_due(self, monkeypatch, tmp_path: Path, caplog):
        group = self._summarised()

        _main(monkeypatch, tmp_path, [group], caplog)
        line = _main(monkeypatch, tmp_path, [group], caplog)

        assert f'{GroupName.LOAD}=not-due' in line, line

    def test_a_failed_group_renders_its_exception_class(self, monkeypatch, tmp_path: Path, caplog):
        group = _due_group(_Recording(raises=[PermissionError('denied')]))

        line = _main(monkeypatch, tmp_path, [group], caplog)

        assert f'{GroupName.LOAD}=failed:PermissionError' in line, line

    def test_a_dropped_group_renders_dropped(self, monkeypatch, tmp_path: Path, caplog):
        group = _due_group(_Recording(samples={SampleKey('fixture_misspelt'): 1.0}))

        line = _main(monkeypatch, tmp_path, [group], caplog)

        assert f'{GroupName.LOAD}=dropped' in line, line
