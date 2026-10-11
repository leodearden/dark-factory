"""Tests for sampler.sampler.run_tick — the per-tick orchestration function."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pytest

from sampler.metrics import SampleKey

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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunTick:
    def test_writes_exactly_9_rows_for_tick(self, tmp_path: Path):
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000

        run_tick(
            store, now,
            psi=FAKE_PSI, process_metrics=FAKE_PROCESS_METRICS, load_metrics={},
        )

        conn = sqlite3.connect(str(tmp_path / 'db.sqlite'))
        count = conn.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
        conn.close()
        assert count == 9

    def test_nine_distinct_metrics_in_tick(self, tmp_path: Path):
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000

        run_tick(
            store, now,
            psi=FAKE_PSI, process_metrics=FAKE_PROCESS_METRICS, load_metrics={},
        )

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

        run_tick(
            store, now,
            psi=FAKE_PSI, process_metrics=FAKE_PROCESS_METRICS, load_metrics={},
        )

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

        run_tick(
            store, now,
            psi=FAKE_PSI, process_metrics=FAKE_PROCESS_METRICS, load_metrics={},
        )

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

        run_tick(
            store, now,
            psi=FAKE_PSI, process_metrics=FAKE_PROCESS_METRICS, load_metrics={},
        )

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
        run_tick(
            store, 1_000_000,
            psi=FAKE_PSI, process_metrics=FAKE_PROCESS_METRICS, load_metrics={},
        )

        # Second tick with different process metrics
        second_metrics = {
            OCCT_QUEUE_DEPTH: 5.0,
            VERIFY_CONCURRENCY: 4.0,
            VERIFY_RSS_TOTAL_BYTES: 2_097_152.0,
        }
        run_tick(
            store, 1_000_005,
            psi=FAKE_PSI, process_metrics=second_metrics, load_metrics={},
        )

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
        run_tick(
            store, now,
            psi=FAKE_PSI, process_metrics=FAKE_PROCESS_METRICS, load_metrics={},
        )

        conn = sqlite3.connect(str(tmp_path / 'db.sqlite'))
        old_row = conn.execute(
            'SELECT COUNT(*) FROM samples WHERE ts = ?', (very_old_ts,)
        ).fetchone()[0]
        conn.close()
        assert old_row == 0, 'Very old row should have been cleaned up'


# ---------------------------------------------------------------------------
# Task 3592 step-9 / task 6772: the exact-or-per-leaf key guard (detail B)
# ---------------------------------------------------------------------------


class TestUnexpectedKeys:
    """The guard's purpose is unchanged: a typo must still fail.

    A per-leaf metric is recorded once per cgroup leaf DISCOVERED at
    collection time, so no fixed set can list its keys. The guard admits a
    host-level key by exact metric name with no leaf, and a per-leaf key by
    metric name plus a non-empty leaf; the leaf itself is opaque data.
    """

    EXACT = frozenset({'runqueue_ratio', 'runqueue_read_ok'})
    PER_LEAF = frozenset({'own_cpu_some10', 'own_read_ok'})

    def _unexpected(self, *keys: SampleKey) -> set[SampleKey]:
        from sampler.sampler import unexpected_keys

        return unexpected_keys(set(keys), exact=self.EXACT, per_leaf=self.PER_LEAF)

    def test_exact_name_accepted(self):
        assert self._unexpected(RUNQUEUE_RATIO, SampleKey('runqueue_read_ok')) == set()

    @pytest.mark.parametrize(
        'key',
        [
            SampleKey('own_cpu_some10', 'orchestrator-dark-factory.service'),
            SampleKey('own_read_ok', 'df-dark_factory.slice'),
            SampleKey('own_cpu_some10', 'df-reify.slice'),
        ],
    )
    def test_per_leaf_metric_with_a_leaf_accepted(self, key):
        assert self._unexpected(key) == set()

    def test_per_leaf_metric_without_a_leaf_rejected(self):
        """A per-leaf metric with no leaf is evidence about no cgroup."""
        assert self._unexpected(SampleKey('own_cpu_some10')) == {SampleKey('own_cpu_some10')}

    def test_per_leaf_metric_with_an_empty_leaf_rejected(self):
        """An empty leaf names no cgroup either."""
        key = SampleKey('own_read_ok', '')
        assert self._unexpected(key) == {key}

    def test_misspelled_metric_rejected(self):
        key = SampleKey('own_cpu_some_10', 'leaf')
        assert self._unexpected(key) == {key}

    def test_exact_metric_carrying_a_leaf_rejected(self):
        """A host-level metric is not per-leaf, so a leaf on it is a collector bug."""
        key = SampleKey('runqueue_ratio', 'leaf.service')
        assert self._unexpected(key) == {key}

    def test_a_leaf_containing_a_colon_is_accepted(self):
        """The leaf is opaque data: nothing parses it, so ':' in it is just a character."""
        assert self._unexpected(SampleKey('own_read_ok', 'weird:leaf.slice')) == set()

    def test_empty_per_leaf_set_degenerates_to_exact_membership(self):
        """PSI keeps its current strictness: no per-leaf metric exists there."""
        from sampler.sampler import unexpected_keys

        psi_exact = frozenset({'psi_cpu_some_avg10'})
        assert unexpected_keys(
            {SampleKey('psi_cpu_some_avg10')}, exact=psi_exact, per_leaf=frozenset()
        ) == set()
        assert unexpected_keys(
            {SampleKey('psi_cpu_some_avg10x')}, exact=psi_exact, per_leaf=frozenset()
        ) == {SampleKey('psi_cpu_some_avg10x')}
        # A per-leaf-shaped key cannot sneak past an empty per-leaf set either.
        per_leaf_shaped = SampleKey('psi_cpu_some_avg10', 'leaf')
        assert unexpected_keys(
            {per_leaf_shaped}, exact=psi_exact, per_leaf=frozenset()
        ) == {per_leaf_shaped}

    def test_every_offender_is_reported_not_just_the_first(self):
        assert self._unexpected(
            RUNQUEUE_RATIO,
            SampleKey('runqueu_ratio'),
            SampleKey('own_read_ok', ''),
            SampleKey('own_read_ok', 'leaf'),
        ) == {SampleKey('runqueu_ratio'), SampleKey('own_read_ok', '')}

    def test_empty_input_is_accepted(self):
        """A degraded collection group hands run_tick {} — never an error."""
        assert self._unexpected() == set()


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
            psi=FAKE_PSI,
            process_metrics=FAKE_PROCESS_METRICS,
            load_metrics=FAKE_LOAD_METRICS,
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

        run_tick(
            store, 1_000_000, psi={}, process_metrics={},
            load_metrics={RUNQUEUE_RATIO: 2.0},
        )
        run_tick(
            store, 1_000_005, psi={}, process_metrics={},
            load_metrics={RUNQUEUE_RATIO: 6.0},
        )

        _first, (_ts, value, window_mean, window_max) = _rows(db_path, RUNQUEUE_RATIO)
        assert value == pytest.approx(6.0)
        assert window_mean == pytest.approx(4.0)
        assert window_max == pytest.approx(6.0)

    def test_a_per_leaf_key_lands_in_the_metric_and_leaf_columns(self, tmp_path: Path):
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)

        run_tick(
            store, 1_000_000, psi={}, process_metrics={},
            load_metrics={SampleKey('own_cpu_some10', REIFY): 1.77},
        )

        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute('SELECT metric, leaf, value FROM samples').fetchall()
        finally:
            conn.close()
        assert rows == [('own_cpu_some10', REIFY, pytest.approx(1.77))]

    def test_unexpected_load_key_raises_naming_it(self, tmp_path: Path):
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')

        with pytest.raises(AssertionError, match='runqueu_ratio'):
            run_tick(
                store, 1_000_000, psi={}, process_metrics={},
                load_metrics={SampleKey('runqueu_ratio'): 1.0},
            )

    def test_degraded_load_group_writes_zero_load_rows(self, tmp_path: Path):
        """{} is the degraded group's value — zero rows, not a fabricated 0.0."""
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)

        run_tick(
            store, 1_000_000,
            psi=FAKE_PSI, process_metrics=FAKE_PROCESS_METRICS, load_metrics={},
        )

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


# ---------------------------------------------------------------------------
# Task 3592 step-17: __main__'s third independent degrade point
# ---------------------------------------------------------------------------


def _metrics_written(db_path: Path) -> set[SampleKey]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            SampleKey(*row) for row in conn.execute('SELECT DISTINCT metric, leaf FROM samples')
        }
    finally:
        conn.close()


def _run_main(monkeypatch, tmp_path: Path, **raising: bool):
    """Drive sampler.__main__.main with each collector optionally raising."""
    import sampler.__main__ as entry

    monkeypatch.setenv('DARK_FACTORY_ROOT', str(tmp_path))

    def collector(name: str, value: dict[SampleKey, float]):
        def collect(**_kwargs):
            if raising.get(name):
                raise RuntimeError(f'{name} is down')
            return value
        return collect

    monkeypatch.setattr(entry, 'collect_psi', collector('psi', FAKE_PSI))
    monkeypatch.setattr(
        entry, 'collect_process_metrics', collector('process', FAKE_PROCESS_METRICS)
    )
    monkeypatch.setattr(
        entry, 'collect_load_metrics', collector('load', FAKE_LOAD_METRICS)
    )
    entry.main()
    return tmp_path / 'data/load-samples.db'


class TestMainDegradesEachGroupIndependently:
    """Three collection groups, three loud degrade points, no shared fate.

    The groups read unrelated kernel surfaces — /proc/pressure, the psutil
    process scan, and /proc/stat + cgroupfs — so one failing must not discard
    another's rows. Each falls back to {} so run_tick writes ZERO rows for it,
    which is the shape that makes a fabricated healthy 0.0 impossible.
    """

    def test_all_three_groups_written_when_healthy(self, monkeypatch, tmp_path: Path):
        db_path = _run_main(monkeypatch, tmp_path)

        written = _metrics_written(db_path)
        assert set(FAKE_PSI) <= written
        assert set(FAKE_PROCESS_METRICS) <= written
        assert set(FAKE_LOAD_METRICS) <= written

    def test_load_failure_keeps_psi_and_process_rows(self, monkeypatch, tmp_path: Path, caplog):
        with caplog.at_level(logging.ERROR):
            db_path = _run_main(monkeypatch, tmp_path, load=True)

        written = _metrics_written(db_path)
        assert set(FAKE_PSI) <= written
        assert set(FAKE_PROCESS_METRICS) <= written
        assert not [m for m in written if m.metric.startswith(('runqueue', 'own_'))], (
            'a failed load group must write zero rows, not a 0.0-valued one'
        )
        messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert any('load' in m.lower() for m in messages), messages

    def test_process_failure_keeps_load_rows(self, monkeypatch, tmp_path: Path, caplog):
        with caplog.at_level(logging.ERROR):
            db_path = _run_main(monkeypatch, tmp_path, process=True)

        written = _metrics_written(db_path)
        assert set(FAKE_LOAD_METRICS) <= written
        assert set(FAKE_PSI) <= written
        assert not set(FAKE_PROCESS_METRICS) & written

    def test_psi_failure_keeps_load_rows(self, monkeypatch, tmp_path: Path, caplog):
        with caplog.at_level(logging.ERROR):
            db_path = _run_main(monkeypatch, tmp_path, psi=True)

        written = _metrics_written(db_path)
        assert set(FAKE_LOAD_METRICS) <= written
        assert set(FAKE_PROCESS_METRICS) <= written
        assert not set(FAKE_PSI) & written

    def test_a_failing_group_does_not_abort_the_tick(self, monkeypatch, tmp_path: Path):
        """Only store-construction failure justifies a non-zero exit."""
        db_path = _run_main(monkeypatch, tmp_path, load=True, process=True)

        assert set(FAKE_PSI) <= _metrics_written(db_path)

    def test_a_healthy_tick_also_drives_the_vacuum_gate(
        self, monkeypatch, tmp_path: Path
    ):
        """``main()`` is the SOLE caller of maybe_vacuum, so it is the only guard.

        ``sampler.sampler``'s module docstring records the split deliberately:
        run_tick enforces retention, and the compaction is wired one level up
        in ``__main__.main`` — which means deleting the call here leaves
        nothing red and a 30-day corpus that is never compacted again.

        The stamp is read back against the TICK's own ts rather than merely
        asserted present: ``__main__`` reads the clock once and hands the same
        ``now`` to run_tick and maybe_vacuum, so the two agreeing is what says
        the vacuum gate ran as part of THIS tick.
        """
        db_path = _run_main(monkeypatch, tmp_path)

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

    def test_the_tick_log_line_reports_the_load_group(self, monkeypatch, tmp_path: Path, caplog):
        """An operator watching journalctl must be able to see the new group."""
        with caplog.at_level(logging.INFO):
            _run_main(monkeypatch, tmp_path)

        tick_lines = [
            r.getMessage() for r in caplog.records if r.getMessage().startswith('tick ')
        ]
        assert tick_lines, [r.getMessage() for r in caplog.records]
        assert any('runqueue_ratio' in line for line in tick_lines), tick_lines
        # The journal renders key labels, not NamedTuple reprs.
        assert any(f'own_cpu_some10:{REIFY}' in line for line in tick_lines), tick_lines
        assert not [line for line in tick_lines if 'SampleKey(' in line], tick_lines


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
            psi={SampleKey('psi_cpu_some_avg10'): 1.0},
            process_metrics={VERIFY_CONCURRENCY: 2.0},
            load_metrics={
                RUNQUEUE_RATIO: 0.5,
                **{SampleKey('own_cpu_some10', f'leaf{i}'): float(i) for i in range(leaves)},
            },
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
