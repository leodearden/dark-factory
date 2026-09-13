"""Tests for sampler.sampler.run_tick — the per-tick orchestration function."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fake data
# ---------------------------------------------------------------------------

FAKE_PSI = {
    'psi_cpu_some_avg10': 2.50,
    'psi_cpu_full_avg10': 0.30,
    'psi_mem_some_avg10': 1.23,
    'psi_mem_full_avg10': 0.00,
    'psi_io_some_avg10': 0.75,
    'psi_io_full_avg10': 0.45,
}

FAKE_PROCESS_METRICS = {
    'occt_queue_depth': 3.0,
    'verify_concurrency': 2.0,
    'verify_rss_total_bytes': 1048576.0,
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

        run_tick(store, now, psi=FAKE_PSI, process_metrics=FAKE_PROCESS_METRICS)

        conn = sqlite3.connect(str(tmp_path / 'db.sqlite'))
        count = conn.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
        conn.close()
        assert count == 9

    def test_nine_distinct_metrics_in_tick(self, tmp_path: Path):
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')
        now = 1_000_000

        run_tick(store, now, psi=FAKE_PSI, process_metrics=FAKE_PROCESS_METRICS)

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

        run_tick(store, now, psi=FAKE_PSI, process_metrics=FAKE_PROCESS_METRICS)

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

        run_tick(store, now, psi=FAKE_PSI, process_metrics=FAKE_PROCESS_METRICS)

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

        run_tick(store, now, psi=FAKE_PSI, process_metrics=FAKE_PROCESS_METRICS)

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
        run_tick(store, 1_000_000, psi=FAKE_PSI, process_metrics=FAKE_PROCESS_METRICS)

        # Second tick with different process metrics
        second_metrics = {
            'occt_queue_depth': 5.0,
            'verify_concurrency': 4.0,
            'verify_rss_total_bytes': 2_097_152.0,
        }
        run_tick(store, 1_000_005, psi=FAKE_PSI, process_metrics=second_metrics)

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

        # Pre-insert a very old row (older than 24h)
        very_old_ts = 1_000_000 - 86401
        store.insert_sample(very_old_ts, 'psi_cpu_some_avg10', 0.0)

        now = 1_000_000
        run_tick(store, now, psi=FAKE_PSI, process_metrics=FAKE_PROCESS_METRICS)

        conn = sqlite3.connect(str(tmp_path / 'db.sqlite'))
        old_row = conn.execute(
            'SELECT COUNT(*) FROM samples WHERE ts = ?', (very_old_ts,)
        ).fetchone()[0]
        conn.close()
        assert old_row == 0, 'Very old row should have been cleaned up'


# ---------------------------------------------------------------------------
# Task 3592 step-9: the exact-or-stem metric-name guard (detail B)
# ---------------------------------------------------------------------------


class TestUnexpectedMetricNames:
    """The guard's purpose is unchanged: a typo must still fail.

    What changed is that it must now also admit names with a DYNAMIC tail —
    ``own_cpu_some10:<cgroup-leaf>`` is generated per discovered cgroup, so no
    fixed frozenset can ever list them. A stem set is the smallest extension
    that admits the tail while keeping a misspelling rejected.
    """

    EXACT = frozenset({'runqueue_ratio', 'runqueue_read_ok'})
    STEMS = frozenset({'own_cpu_some10', 'own_read_ok'})

    def _unexpected(self, *names):
        from sampler.sampler import unexpected_metric_names

        return unexpected_metric_names(set(names), exact=self.EXACT, stems=self.STEMS)

    def test_exact_name_accepted(self):
        assert self._unexpected('runqueue_ratio', 'runqueue_read_ok') == set()

    @pytest.mark.parametrize(
        'name',
        [
            'own_cpu_some10:orchestrator-dark-factory.service',
            'own_read_ok:df-dark_factory.slice',
            'own_cpu_some10:df-reify.slice',
        ],
    )
    def test_stem_with_a_tail_accepted(self, name):
        assert self._unexpected(name) == set()

    def test_bare_stem_without_a_tail_rejected(self):
        """A stem is not itself a metric — nothing ever emits a bare one."""
        assert self._unexpected('own_cpu_some10') == {'own_cpu_some10'}

    def test_misspelled_stem_rejected(self):
        assert self._unexpected('own_cpu_some_10:leaf') == {'own_cpu_some_10:leaf'}

    def test_stem_with_an_empty_tail_rejected(self):
        """``own_read_ok:`` names no cgroup, so it is evidence about nothing."""
        assert self._unexpected('own_read_ok:') == {'own_read_ok:'}

    def test_only_the_first_colon_splits_the_stem(self):
        """A leaf name may itself contain ':' — the tail is everything after."""
        assert self._unexpected('own_read_ok:weird:leaf.slice') == set()

    def test_empty_stem_set_degenerates_to_exact_membership(self):
        """PSI keeps its current strictness: no dynamic tail is possible there."""
        from sampler.sampler import unexpected_metric_names

        psi_exact = frozenset({'psi_cpu_some_avg10'})
        assert unexpected_metric_names(
            {'psi_cpu_some_avg10'}, exact=psi_exact, stems=frozenset()
        ) == set()
        assert unexpected_metric_names(
            {'psi_cpu_some_avg10x'}, exact=psi_exact, stems=frozenset()
        ) == {'psi_cpu_some_avg10x'}
        # A colon name cannot sneak past an empty stem set either.
        assert unexpected_metric_names(
            {'psi_cpu_some_avg10:leaf'}, exact=psi_exact, stems=frozenset()
        ) == {'psi_cpu_some_avg10:leaf'}

    def test_every_offender_is_reported_not_just_the_first(self):
        assert self._unexpected(
            'runqueue_ratio', 'runqueu_ratio', 'own_read_ok:', 'own_read_ok:leaf'
        ) == {'runqueu_ratio', 'own_read_ok:'}

    def test_empty_input_is_accepted(self):
        """A degraded collection group hands run_tick {} — never an error."""
        assert self._unexpected() == set()


# ---------------------------------------------------------------------------
# Task 3592 step-11: run_tick's third collection group
# ---------------------------------------------------------------------------

FAKE_LOAD_METRICS = {
    'runqueue_ratio': 4.0625,
    'runqueue_read_ok': 1.0,
    'own_cpu_some10:orchestrator-reify.service': 1.77,
    'own_read_ok:orchestrator-reify.service': 1.0,
}


def _rows(db_path: Path, metric: str) -> list[tuple]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            'SELECT ts, value, window_mean, window_max FROM samples'
            ' WHERE metric = ? ORDER BY ts',
            (metric,),
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

        for metric in FAKE_LOAD_METRICS:
            (_ts, _value, window_mean, window_max), = _rows(db_path, metric)
            assert window_mean is not None, f'{metric} window_mean is NULL'
            assert window_max is not None, f'{metric} window_max is NULL'
        (_ts, _v, psi_mean, psi_max), = _rows(db_path, 'psi_cpu_some_avg10')
        assert psi_mean is None and psi_max is None

    def test_second_tick_window_reflects_both_samples(self, tmp_path: Path):
        """Proves the row went through store.trailing_window, not a NULL write."""
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)

        run_tick(
            store, 1_000_000, psi={}, process_metrics={},
            load_metrics={'runqueue_ratio': 2.0},
        )
        run_tick(
            store, 1_000_005, psi={}, process_metrics={},
            load_metrics={'runqueue_ratio': 6.0},
        )

        _first, (_ts, value, window_mean, window_max) = _rows(db_path, 'runqueue_ratio')
        assert value == pytest.approx(6.0)
        assert window_mean == pytest.approx(4.0)
        assert window_max == pytest.approx(6.0)

    def test_dynamic_keys_round_trip_into_the_metric_column_verbatim(
        self, tmp_path: Path
    ):
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        db_path = tmp_path / 'db.sqlite'
        store = LoadSampleStore(db_path)
        metric = 'own_cpu_some10:orchestrator-reify.service'

        run_tick(
            store, 1_000_000, psi={}, process_metrics={},
            load_metrics={metric: 1.77},
        )

        (_ts, value, _mean, _max), = _rows(db_path, metric)
        assert value == pytest.approx(1.77)

    def test_unexpected_load_key_raises_naming_it(self, tmp_path: Path):
        from sampler.sampler import run_tick
        from sampler.store import LoadSampleStore

        store = LoadSampleStore(tmp_path / 'db.sqlite')

        with pytest.raises(AssertionError, match='runqueu_ratio'):
            run_tick(
                store, 1_000_000, psi={}, process_metrics={},
                load_metrics={'runqueu_ratio': 1.0},
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
