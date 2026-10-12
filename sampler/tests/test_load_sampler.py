"""Tests for sampler.sampler.run_tick — the per-tick orchestration function."""

from __future__ import annotations

import dataclasses
import logging
import sqlite3
from collections.abc import Collection, Mapping
from pathlib import Path
from types import MappingProxyType

import pytest

from sampler.metrics import (
    METRIC_GROUPS,
    GroupName,
    MetricGroup,
    MetricSpec,
    SampleKey,
    Samples,
)

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

FAKE_PSI = {
    SampleKey('psi_cpu_some_avg10'): 2.50,
    SampleKey('psi_cpu_full_avg10'): 0.30,
    SampleKey('psi_mem_some_avg10'): 1.23,
    SampleKey('psi_mem_full_avg10'): 0.00,
    SampleKey('psi_io_some_avg10'): 0.75,
    SampleKey('psi_io_full_avg10'): 0.45,
}

OCCT_QUEUE_DEPTH = SampleKey('occt_queue_depth')
VERIFY_CONCURRENCY = SampleKey('verify_concurrency')
VERIFY_RSS_TOTAL_BYTES = SampleKey('verify_rss_total_bytes')
RUNQUEUE_RATIO = SampleKey('runqueue_ratio')
REIFY = 'orchestrator-reify.service'

FAKE_PROCESS_METRICS = {
    OCCT_QUEUE_DEPTH: 3.0,
    VERIFY_CONCURRENCY: 2.0,
    VERIFY_RSS_TOTAL_BYTES: 1048576.0,
}

_GROUPS = {group.name: group for group in METRIC_GROUPS}
_NOTHING: Samples = MappingProxyType({})


def _collected(
    *, psi: Samples = _NOTHING, process: Samples = _NOTHING, load: Samples = _NOTHING
) -> dict[MetricGroup, Samples]:
    """One tick's collection over the live table, a degraded group being ``{}``."""
    return {
        _GROUPS[GroupName.PSI]: psi,
        _GROUPS[GroupName.PROCESS]: process,
        _GROUPS[GroupName.LOAD]: load,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunTick:
    def test_writes_exactly_9_rows_for_tick(self, tmp_path: Path):
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000

        run_tick(store, now, _collected(psi=FAKE_PSI, process=FAKE_PROCESS_METRICS))

        conn = sqlite3.connect(str(tmp_path / 'db.sqlite'))
        count = conn.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
        conn.close()
        assert count == 9

    def test_nine_distinct_metrics_in_tick(self, tmp_path: Path):
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000

        run_tick(store, now, _collected(psi=FAKE_PSI, process=FAKE_PROCESS_METRICS))

        conn = sqlite3.connect(str(tmp_path / 'db.sqlite'))
        distinct = conn.execute(
            'SELECT COUNT(DISTINCT metric) FROM samples WHERE ts = ?', (now,)
        ).fetchone()[0]
        conn.close()
        # >= 7 satisfies the task's live integration signal
        assert distinct == 9

    def test_psi_rows_have_null_windows(self, tmp_path: Path):
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000

        run_tick(store, now, _collected(psi=FAKE_PSI, process=FAKE_PROCESS_METRICS))

        conn = sqlite3.connect(str(tmp_path / 'db.sqlite'))
        psi_rows = conn.execute(
            "SELECT window_mean, window_max FROM samples"
            " WHERE metric LIKE 'psi_%'"
        ).fetchall()
        conn.close()

        assert len(psi_rows) == 6
        for window_mean, window_max in psi_rows:
            assert window_mean is None, f'Expected NULL window_mean, got {window_mean}'
            assert window_max is None, f'Expected NULL window_max, got {window_max}'

    def test_non_psi_rows_have_non_null_windows(self, tmp_path: Path):
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000

        run_tick(store, now, _collected(psi=FAKE_PSI, process=FAKE_PROCESS_METRICS))

        conn = sqlite3.connect(str(tmp_path / 'db.sqlite'))
        proc_rows = conn.execute(
            "SELECT metric, window_mean, window_max FROM samples"
            " WHERE metric NOT LIKE 'psi_%'"
        ).fetchall()
        conn.close()

        assert len(proc_rows) == 3
        for metric, window_mean, window_max in proc_rows:
            assert window_mean is not None, f'{metric}: window_mean should not be NULL'
            assert window_max is not None, f'{metric}: window_max should not be NULL'

    def test_non_psi_windows_equal_trailing_window(self, tmp_path: Path):
        """On first tick (no prior history), window_mean == window_max == current value."""
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000

        run_tick(store, now, _collected(psi=FAKE_PSI, process=FAKE_PROCESS_METRICS))

        conn = sqlite3.connect(str(tmp_path / 'db.sqlite'))
        row = conn.execute(
            "SELECT value, window_mean, window_max FROM samples"
            " WHERE metric = 'occt_queue_depth'"
        ).fetchone()
        conn.close()

        value, window_mean, window_max = row
        assert value == pytest.approx(3.0)
        # No prior rows -> trailing_window returns (current, current)
        assert window_mean == pytest.approx(3.0)
        assert window_max == pytest.approx(3.0)

    def test_second_tick_uses_prior_history_for_windows(self, tmp_path: Path):
        """On the second tick, window should incorporate first tick's value."""
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')

        # First tick
        run_tick(store, 1_000_000, _collected(psi=FAKE_PSI, process=FAKE_PROCESS_METRICS))

        # Second tick with different process metrics
        second_metrics = {
            OCCT_QUEUE_DEPTH: 5.0,
            VERIFY_CONCURRENCY: 4.0,
            VERIFY_RSS_TOTAL_BYTES: 2_097_152.0,
        }
        run_tick(store, 1_000_005, _collected(psi=FAKE_PSI, process=second_metrics))

        conn = sqlite3.connect(str(tmp_path / 'db.sqlite'))
        row = conn.execute(
            "SELECT value, window_mean, window_max FROM samples"
            " WHERE metric = 'occt_queue_depth' AND ts = 1000005"
        ).fetchone()
        conn.close()

        value, window_mean, window_max = row
        assert value == pytest.approx(5.0)
        # window = [3.0 (prior), 5.0 (current)] -> mean=4.0, max=5.0
        assert window_mean == pytest.approx(4.0)
        assert window_max == pytest.approx(5.0)

    def test_second_tick_triggers_cleanup(self, tmp_path: Path):
        """run_tick calls cleanup_old so stale rows are removed."""
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')

        # Pre-insert a very old row (older than the 30-day retention window,
        # widened from 24h by task 3592 step-14).
        very_old_ts = 1_000_000 - 2_592_001
        store.insert_sample(very_old_ts, SampleKey('psi_cpu_some_avg10'), 0.0)

        now = 1_000_000
        run_tick(store, now, _collected(psi=FAKE_PSI, process=FAKE_PROCESS_METRICS))

        conn = sqlite3.connect(str(tmp_path / 'db.sqlite'))
        old_row = conn.execute(
            'SELECT COUNT(*) FROM samples WHERE ts = ?', (very_old_ts,)
        ).fetchone()[0]
        conn.close()
        assert old_row == 0, 'Very old row should have been cleaned up'


# ---------------------------------------------------------------------------
# Task 3592 step-11: run_tick's third collection group
# ---------------------------------------------------------------------------

FAKE_LOAD_METRICS = {
    RUNQUEUE_RATIO: 4.0625,
    SampleKey('runqueue_read_ok'): 1.0,
    SampleKey('own_cpu_some10', REIFY): 1.77,
    SampleKey('own_read_ok', REIFY): 1.0,
}


def _rows(db_path: Path, key: SampleKey) -> list[tuple]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            'SELECT ts, value, window_mean, window_max FROM samples'
            ' WHERE metric = ? AND leaf IS ? ORDER BY ts',
            (key.metric, key.leaf),
        ).fetchall()
    finally:
        conn.close()


class TestRunTickLoadGroup:
    """Load metrics are SAMPLER-windowed, unlike the kernel-windowed PSI rows."""

    def test_load_rows_carry_populated_windows_while_psi_rows_stay_null(
        self, tmp_path: Path
    ):
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)

        run_tick(
            store,
            1_000_000,
            _collected(psi=FAKE_PSI, process=FAKE_PROCESS_METRICS, load=FAKE_LOAD_METRICS),
        )

        for key in FAKE_LOAD_METRICS:
            (_ts, _value, window_mean, window_max), = _rows(db_path, key)
            assert window_mean is not None, f'{key} window_mean is NULL'
            assert window_max is not None, f'{key} window_max is NULL'
        (_ts, _v, psi_mean, psi_max), = _rows(db_path, SampleKey('psi_cpu_some_avg10'))
        assert psi_mean is None and psi_max is None

    def test_second_tick_window_reflects_both_samples(self, tmp_path: Path):
        """Proves the row went through store.trailing_window, not a NULL write."""
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)

        run_tick(store, 1_000_000, _collected(load={RUNQUEUE_RATIO: 2.0}))
        run_tick(store, 1_000_005, _collected(load={RUNQUEUE_RATIO: 6.0}))

        _first, (_ts, value, window_mean, window_max) = _rows(db_path, RUNQUEUE_RATIO)
        assert value == pytest.approx(6.0)
        assert window_mean == pytest.approx(4.0)
        assert window_max == pytest.approx(6.0)

    def test_a_per_leaf_key_lands_in_the_metric_and_leaf_columns(self, tmp_path: Path):
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)

        run_tick(store, 1_000_000, _collected(load={SampleKey('own_cpu_some10', REIFY): 1.77}))

        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute('SELECT metric, leaf, value FROM samples').fetchall()
        finally:
            conn.close()
        assert rows == [('own_cpu_some10', REIFY, pytest.approx(1.77))]

    def test_degraded_load_group_writes_zero_load_rows(self, tmp_path: Path):
        """{} is the degraded group's value — zero rows, not a fabricated 0.0."""
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)

        run_tick(store, 1_000_000, _collected(psi=FAKE_PSI, process=FAKE_PROCESS_METRICS))

        conn = sqlite3.connect(str(db_path))
        try:
            total = conn.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
            loadish = conn.execute(
                "SELECT COUNT(*) FROM samples"
                " WHERE metric LIKE 'runqueue%' OR metric LIKE 'own_%'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert loadish == 0
        assert total == 9, 'the other two groups must still write their rows'


class TestRunTickAdmitsEachGroupAgainstItself:
    """An unadmitted key drops its own group for the tick, never the whole tick."""

    def test_an_unadmitted_key_drops_only_its_own_group(self, tmp_path: Path, caplog):
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        load = {SampleKey('runqueu_ratio'): 1.0, RUNQUEUE_RATIO: 2.0}

        with caplog.at_level(logging.ERROR):
            run_tick(
                LoadSampleStore(db_path),
                1_000_000,
                _collected(psi=FAKE_PSI, process=FAKE_PROCESS_METRICS, load=load),
            )

        written = _metrics_written(db_path)
        assert written == set(FAKE_PSI) | set(FAKE_PROCESS_METRICS)
        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1, errors
        assert 'load' in errors[0]
        assert 'runqueu_ratio' in errors[0]

    def test_every_offender_is_named(self, tmp_path: Path, caplog):
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        process = {
            SampleKey('occt_queue_dept'): 1.0,
            SampleKey('verify_concurrency', 'leaf.service'): 1.0,
            VERIFY_RSS_TOTAL_BYTES: 1.0,
        }

        with caplog.at_level(logging.ERROR):
            run_tick(LoadSampleStore(tmp_path / 'db.sqlite'), 1_000_000, _collected(process=process))

        (error,) = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert 'process' in error
        assert 'occt_queue_dept' in error
        assert 'verify_concurrency:leaf.service' in error

    @pytest.mark.parametrize('windowed', [False, True])
    def test_window_mode_comes_from_the_group(self, tmp_path: Path, windowed: bool):
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        spec = MetricSpec('fixture_metric')
        group = MetricGroup(GroupName.LOAD, collect=dict, metrics=(spec,), windowed=windowed)
        db_path = tmp_path / 'db.sqlite'

        run_tick(LoadSampleStore(db_path), 1_000_000, {group: {spec.key(): 3.5}})

        ((_ts, value, window_mean, window_max),) = _rows(db_path, spec.key())
        assert value == pytest.approx(3.5)
        if windowed:
            assert (window_mean, window_max) == (pytest.approx(3.5), pytest.approx(3.5))
        else:
            assert (window_mean, window_max) == (None, None)

    def test_it_returns_the_groups_it_wrote(self, tmp_path: Path):
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        collected = _collected(psi=FAKE_PSI, load={SampleKey('runqueu_ratio'): 1.0})

        written = run_tick(LoadSampleStore(tmp_path / 'db.sqlite'), 1_000_000, collected)

        assert written == {
            _GROUPS[GroupName.PSI]: FAKE_PSI,
            _GROUPS[GroupName.PROCESS]: {},
        }

    def test_every_admitted_group_lands_in_one_write(self, tmp_path: Path):
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore, connect_durably

        opened: list[int] = []

        def counting_connect(db_path: Path, *, autocommit: bool = False):
            opened.append(1)
            return connect_durably(db_path, autocommit=autocommit)

        store = LoadSampleStore(tmp_path / 'db.sqlite', connect=counting_connect)
        three_groups = _collected(
            psi=FAKE_PSI, process=FAKE_PROCESS_METRICS, load=FAKE_LOAD_METRICS
        )
        one_group = {_GROUPS[GroupName.LOAD]: FAKE_LOAD_METRICS}
        run_tick(store, 999_995, three_groups)

        opened.clear()
        run_tick(store, 1_000_000, one_group)
        one = len(opened)
        opened.clear()
        run_tick(store, 1_000_005, three_groups)
        three = len(opened)

        assert one >= 1, 'the connect spy never fired — the seam has detached'
        assert three == one, f'a 1-group tick opened {one} connections, a 3-group tick {three}'


# ---------------------------------------------------------------------------
# __main__.main: one degrade handler per group of an injected table
# ---------------------------------------------------------------------------

FAKE_SAMPLES: dict[GroupName, Samples] = {
    GroupName.PSI: FAKE_PSI,
    GroupName.PROCESS: FAKE_PROCESS_METRICS,
    GroupName.LOAD: FAKE_LOAD_METRICS,
}


def _metrics_written(db_path: Path) -> set[SampleKey]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            SampleKey(*row) for row in conn.execute('SELECT DISTINCT metric, leaf FROM samples')
        }
    finally:
        conn.close()


def _table(
    *,
    samples: Mapping[GroupName, Samples] = FAKE_SAMPLES,
    raising: Collection[GroupName] = (),
) -> tuple[MetricGroup, ...]:
    """METRIC_GROUPS with each collector replaced by a fake returning *samples* or raising."""

    def collector(name: GroupName):
        def collect() -> Samples:
            if name in raising:
                raise RuntimeError(f'{name} is down')
            return samples[name]
        return collect

    return tuple(
        dataclasses.replace(group, collect=collector(group.name)) for group in METRIC_GROUPS
    )


def _run_main(monkeypatch, tmp_path: Path, table: tuple[MetricGroup, ...]) -> Path:
    """Drive sampler.__main__.main over *table*, rooted at *tmp_path*."""
    import sampler.__main__ as entry

    monkeypatch.setenv('DARK_FACTORY_ROOT', str(tmp_path))
    entry.main(groups=table)
    return tmp_path / 'data/load-samples.db'


def _tick_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith('tick ')]


def _errors(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


class TestMainDegradesEachGroupIndependently:
    """One loud degrade point per group, no shared fate.

    The groups read unrelated kernel surfaces, so one failing must not discard
    another's rows. A failing group falls back to {} so run_tick writes ZERO
    rows for it, which is the shape that makes a fabricated healthy 0.0
    impossible.
    """

    def test_every_group_is_written_when_healthy(self, monkeypatch, tmp_path: Path):
        written = _metrics_written(_run_main(monkeypatch, tmp_path, _table()))

        for samples in FAKE_SAMPLES.values():
            assert set(samples) <= written

    @pytest.mark.parametrize('down', list(GroupName))
    def test_a_failing_group_keeps_the_other_groups_rows(
        self, monkeypatch, tmp_path: Path, caplog, down: GroupName
    ):
        with caplog.at_level(logging.ERROR):
            db_path = _run_main(monkeypatch, tmp_path, _table(raising={down}))

        written = _metrics_written(db_path)
        for name, samples in FAKE_SAMPLES.items():
            if name is not down:
                assert set(samples) <= written, name
        assert not [key for key in written if _GROUPS[down].admits(key)], (
            f'a failed {down} group must write zero rows, not a 0.0-valued one'
        )
        assert any(str(down) in message for message in _errors(caplog)), _errors(caplog)

    def test_two_failing_groups_still_write_the_third(self, monkeypatch, tmp_path: Path):
        """Only store-construction failure justifies a non-zero exit."""
        table = _table(raising={GroupName.LOAD, GroupName.PROCESS})

        assert set(FAKE_PSI) <= _metrics_written(_run_main(monkeypatch, tmp_path, table))

    def test_an_unadmitted_key_drops_only_its_group(self, monkeypatch, tmp_path: Path, caplog):
        bad = {**FAKE_SAMPLES, GroupName.LOAD: {**FAKE_LOAD_METRICS, SampleKey('runqueu_ratio'): 1.0}}

        with caplog.at_level(logging.ERROR):
            db_path = _run_main(monkeypatch, tmp_path, _table(samples=bad))

        assert _metrics_written(db_path) == set(FAKE_PSI) | set(FAKE_PROCESS_METRICS)
        (error,) = _errors(caplog)
        assert 'load' in error
        assert 'runqueu_ratio' in error

    def test_a_healthy_tick_also_drives_the_vacuum_gate(
        self, monkeypatch, tmp_path: Path
    ):
        """``main()`` is the SOLE caller of maybe_vacuum, so it is the only guard.

        run_tick enforces retention, and the compaction is wired one level up
        in ``__main__.main``, so deleting the call there would leave nothing
        red and a 30-day corpus that is never compacted again.

        The stamp is read back against the TICK's own ts rather than merely
        asserted present: ``__main__`` reads the clock once and hands the same
        ``now`` to run_tick and maybe_vacuum, so the two agreeing is what says
        the vacuum gate ran as part of THIS tick.
        """
        db_path = _run_main(monkeypatch, tmp_path, _table())

        from sampler.store import LoadSampleStore

        conn = sqlite3.connect(str(db_path))
        try:
            tick_ts = conn.execute('SELECT MAX(ts) FROM samples').fetchone()[0]
        finally:
            conn.close()

        # _is_due reads "due" as elapsed outside [0, interval), so this boundary
        # pair holds only for a stamp of exactly tick_ts.
        store = LoadSampleStore(db_path)
        assert store.should_vacuum(tick_ts + 86399) is False, (
            'main() left the vacuum clock unset, so maybe_vacuum was never called'
        )
        assert store.should_vacuum(tick_ts + 86400) is True

    def test_main_runs_the_live_table_by_default(self):
        """The systemd ExecStart path calls main() with no argument."""
        import inspect

        import sampler.__main__ as entry

        assert inspect.signature(entry.main).parameters['groups'].default is METRIC_GROUPS


class TestTheTickLogLine:
    """An operator watching journalctl sees every group the table names."""

    def test_it_names_exactly_the_tables_groups(self, monkeypatch, tmp_path: Path, caplog):
        table = _table()[:2]

        with caplog.at_level(logging.INFO):
            _run_main(monkeypatch, tmp_path, table)

        (line,) = _tick_lines(caplog)
        for group in table:
            assert f'{group.name}=' in line, line
        assert f'{GroupName.LOAD}=' not in line, line

    def test_it_renders_key_labels(self, monkeypatch, tmp_path: Path, caplog):
        with caplog.at_level(logging.INFO):
            _run_main(monkeypatch, tmp_path, _table())

        (line,) = _tick_lines(caplog)
        assert 'runqueue_ratio' in line
        assert f'own_cpu_some10:{REIFY}' in line
        assert 'SampleKey(' not in line

    def test_a_dropped_group_shows_as_dropped(self, monkeypatch, tmp_path: Path, caplog):
        bad = {**FAKE_SAMPLES, GroupName.LOAD: {SampleKey('runqueu_ratio'): 1.0}}

        with caplog.at_level(logging.INFO):
            _run_main(monkeypatch, tmp_path, _table(samples=bad))

        (line,) = _tick_lines(caplog)
        assert f'{GroupName.LOAD}=dropped' in line, line


# ---------------------------------------------------------------------------
# Review suggestion 1: a tick's write cost must not scale with its metric count
# ---------------------------------------------------------------------------


class TestTickCostIsFlatInTheMetricCount:
    """run_tick must keep handing the store ONE tick, not N rows.

    The store-level measurement and reasoning live on
    ``LoadSampleStore.write_tick``; this is the end of the chain that stops
    run_tick quietly going back to a per-row loop. The leaf count is what makes
    it matter: ``discover_pressure_cgroups`` returns however many cgroup leaves
    exist at collection time, so the row count per tick is unbounded here.
    """

    def _tick(self, store, now: int, leaves: int) -> None:
        from sampler.sampler import run_tick

        run_tick(
            store,
            now,
            _collected(
                psi={SampleKey('psi_cpu_some_avg10'): 1.0},
                process={VERIFY_CONCURRENCY: 2.0},
                load={
                    RUNQUEUE_RATIO: 0.5,
                    **{SampleKey('own_cpu_some10', f'leaf{i}'): float(i) for i in range(leaves)},
                },
            ),
        )

    def test_one_leaf_and_a_hundred_leaves_cost_the_same_connections(self, tmp_path: Path):
        from sampler.store import LoadSampleStore, connect_durably

        opened: list[int] = []

        def counting_connect(db_path: Path, *, autocommit: bool = False):
            opened.append(1)
            return connect_durably(db_path, autocommit=autocommit)

        store = LoadSampleStore(tmp_path / 'db.sqlite', connect=counting_connect)
        # Prime the retention clock first. On a VIRGIN store the first tick
        # also runs the interval-gated cleanup, which opens its own
        # connections — comparing a cleanup tick against a steady-state one
        # would measure the retention gate rather than the write path, and
        # this test went red on exactly that before the priming tick existed.
        self._tick(store, 999_995, leaves=1)

        opened.clear()
        self._tick(store, 1_000_000, leaves=1)
        few = len(opened)
        opened.clear()
        self._tick(store, 1_000_005, leaves=100)
        many = len(opened)

        # A FLOOR before the equality, because both sides are spy counts: if the
        # injected connect ever stops being the one write_tick uses — a cached
        # connection, a pool — both stay 0 and `0 == many` holds while
        # measuring nothing.
        assert few >= 1, 'the connect spy never fired — the seam has detached'
        assert few == many, (
            f'a 4-metric tick opened {few} connections and a 103-metric tick '
            f'opened {many}'
        )
