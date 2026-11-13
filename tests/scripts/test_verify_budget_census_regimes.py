"""by_load_band, band_pressure, load_movement, cold_separability and
merge_gate_budget: the regime a duration was measured under, reported rather
than inferred.

The synthetic-corpus norm lives in tests/scripts/verify_budget_census_fixture.py.
"""
from __future__ import annotations

import json
from typing import Any

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from verify_budget_census_fixture import (
    FULL_SUITE,
    command_entry,
    run_main,
    summary_payload,
    write_corpus,
    write_module_yaml,
    write_worktree_record,
)

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

# "Whatever the start instant reads" — a sentinel, because `None` is itself a
# MEANINGFUL end reading here (a degraded component) and cannot double as the
# default.
_MIRROR_START: Any = object()


def _load_record(
    cpu: float | None = 1.0,
    cpu60: float | None = None,
    runqueue: float = 0.5,
    end_cpu: float | None = _MIRROR_START,
    end_cpu60: float | None = _MIRROR_START,
):
    """One `load` record as `verify._load_sample`/`_xdist_workers` emit it.

    The end instant MIRRORS the start unless a case says otherwise, which is
    the shape a stable host leaves behind.
    """
    end = cpu if end_cpu is _MIRROR_START else end_cpu
    return {
        'start': {
            'cpu_some10': cpu,
            'cpu_some60': cpu if cpu60 is None else cpu60,
            'runqueue_ratio': runqueue,
        },
        'end': {
            'cpu_some10': end,
            'cpu_some60': end if end_cpu60 is _MIRROR_START else end_cpu60,
            'runqueue_ratio': runqueue,
        },
        'xdist': {'n_flag': None, 'auto_num_workers': None},
    }


class TestPsiBanding:
    """Durations bucketed by the host pressure their command actually ran under.

    This is what deliverable 1 exists for: "measure on a quiet host" stops
    being a precondition, because every record says how quiet its host was.
    """

    def _banded(self, tmp_path, *entries):
        from verify_budget_census import (  # noqa: PLC0415
            by_load_band,
            load_records,
            select_full_suite_legs,
        )

        write_corpus(tmp_path, *entries)
        legs = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
        ).legs
        return by_load_band(legs)

    def test_a_quiet_and_a_busy_run_land_in_different_bands(self, tmp_path):
        bands = self._banded(
            tmp_path,
            command_entry(duration_secs=3000.0, load=_load_record(cpu=1.0)),
            command_entry(duration_secs=4600.0, load=_load_record(cpu=75.0)),
        )

        assert bands['unstamped']['durations']['n'] == 0
        banded = {
            name: row['durations']['max']
            for name, row in bands.items()
            if name != 'unstamped' and row['durations']['n']
        }
        assert sorted(banded.values()) == [3000.0, 4600.0]
        assert len(banded) == 2, 'a quiet and a heavily-loaded run must not share a band'

    def test_the_bands_are_named_constants_not_inline_numbers(self):
        """Band edges carry a rationale, so a later reader can argue with them."""
        from verify_budget_census import PSI_BANDS  # noqa: PLC0415

        assert PSI_BANDS
        for band in PSI_BANDS:
            assert band.name
            assert band.why, f'band {band.name} has no stated rationale'

    def test_the_bands_tile_the_range_without_gaps_or_overlap(self):
        """Every possible pressure reading lands in exactly one band."""
        from verify_budget_census import PSI_BANDS, band_for  # noqa: PLC0415

        for value in (0.0, 0.01, 4.9, 5.0, 24.9, 25.0, 49.9, 50.0, 99.9, 100.0):
            matches = [b.name for b in PSI_BANDS if b.lo <= value < b.hi]
            assert len(matches) == 1, f'{value} matched {matches}'
            assert band_for(value) == matches[0]


class TestUnstampedIsItsOwnRow:
    """The honesty property that matters most for the HISTORICAL corpus.

    Measured while building this: ZERO records in the live corpus carry a load
    stamp, because deliverable 1 has not merged. So `unstamped` is not an edge
    case — today it is the entire corpus, and every figure the budget is
    currently derived from sits in it.

    Putting those in a zero-pressure band would be the worst available lie: it
    would file the busiest historical runs in the IDLE band and then invite the
    conclusion that the suite is slow even on a quiet host.
    """

    def _banded(self, tmp_path, *entries):
        from verify_budget_census import (  # noqa: PLC0415
            by_load_band,
            load_records,
            select_full_suite_legs,
        )

        write_corpus(tmp_path, *entries)
        legs = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
        ).legs
        return by_load_band(legs)

    def test_a_record_predating_the_stamp_is_unstamped(self, tmp_path):
        bands = self._banded(tmp_path, command_entry(duration_secs=4991.0, load=None))

        assert bands['unstamped']['durations'] == {
            'n': 1, 'p50': 4991.0, 'p90': 4991.0, 'max': 4991.0,
        }

    def test_a_degraded_psi_read_is_unstamped_not_zero_pressure(self, tmp_path):
        """A null cpu_some10 means "not knowable", never "the host was idle"."""
        bands = self._banded(
            tmp_path, command_entry(duration_secs=4991.0, load=_load_record(cpu=None)),
        )

        assert bands['unstamped']['durations']['n'] == 1
        assert bands[_idle_band_name()]['durations']['n'] == 0

    def test_a_genuinely_idle_host_is_banded_not_unstamped(self, tmp_path):
        """The complement — 0.0 is a real reading and must not read as missing."""
        bands = self._banded(
            tmp_path, command_entry(duration_secs=3000.0, load=_load_record(cpu=0.0)),
        )

        assert bands['unstamped']['durations']['n'] == 0
        assert bands[_idle_band_name()]['durations']['n'] == 1

    def test_the_bands_reconcile_against_the_selected_n(self, tmp_path):
        """Never silently dropped: banded + unstamped accounts for every leg."""
        bands = self._banded(
            tmp_path,
            command_entry(duration_secs=3000.0, load=_load_record(cpu=0.0)),
            command_entry(duration_secs=3100.0, load=_load_record(cpu=40.0)),
            command_entry(duration_secs=4991.0, load=None),
            command_entry(duration_secs=7200.0, load=None, timed_out=True, rc=-9),
        )

        accounted = sum(
            row['durations']['n'] + row['timed_out'] + row['failed']
            for row in bands.values()
        )
        assert accounted == 4


def _idle_band_name():
    from verify_budget_census import PSI_BANDS  # noqa: PLC0415

    return PSI_BANDS[0].name


def _heavy_band_name():
    from verify_budget_census import band_for  # noqa: PLC0415

    return band_for(75.0)


class TestTheBandStatisticSpansTheRun:
    """A leg is banded by the HIGHER of its two readings, not by the start alone.

    The start sample is taken immediately BEFORE the command. For the population
    being censused — full-suite runs with a p50 near 3000s — that 10-second
    window describes the host during the QUEUE WAIT, not during the run. The end
    sample is stamped precisely to cover the other end of the interval, so
    banding that read nothing from it rested the report's central claim ("a
    duration figure, banded by the load it ran under") on its weakest available
    reading.
    """

    def _banded(self, tmp_path, *entries):
        from verify_budget_census import (  # noqa: PLC0415
            by_load_band,
            load_records,
            select_full_suite_legs,
        )

        write_corpus(tmp_path, *entries)
        legs = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
        ).legs
        return by_load_band(legs)

    def test_a_run_that_started_quiet_and_ended_busy_bands_busy(self, tmp_path):
        """The case start-banding got wrong: an hour of contention filed as idle."""
        bands = self._banded(
            tmp_path,
            command_entry(duration_secs=4600.0, load=_load_record(cpu=1.0, end_cpu=75.0)),
        )

        assert bands[_idle_band_name()]['durations']['n'] == 0
        assert bands[_heavy_band_name()]['durations']['max'] == 4600.0

    def test_the_peak_counts_whichever_end_it_falls_on(self, tmp_path):
        """A run that started busy and ended quiet is not a quiet run either."""
        bands = self._banded(
            tmp_path,
            command_entry(duration_secs=4600.0, load=_load_record(cpu=75.0, end_cpu=1.0)),
        )

        assert bands[_idle_band_name()]['durations']['n'] == 0
        assert bands[_heavy_band_name()]['durations']['max'] == 4600.0

    @pytest.mark.parametrize(
        ('start', 'end'), [(75.0, None), (None, 75.0)], ids=['end-null', 'start-null'],
    )
    def test_one_knowable_reading_still_bands_the_leg(self, tmp_path, start, end):
        """A half-degraded record carries a real reading; discarding it would be
        the silent-fail-soft shape `unstamped` exists to avoid in reverse."""
        bands = self._banded(
            tmp_path,
            command_entry(duration_secs=4600.0, load=_load_record(cpu=start, end_cpu=end)),
        )

        assert bands['unstamped']['durations']['n'] == 0
        assert bands[_heavy_band_name()]['durations']['max'] == 4600.0

    def test_the_statistic_is_the_max_of_the_two_instants(self):
        """Stated directly, since the whole report rests on this one choice.

        The MAX, not the mean: a band says "measured under AT LEAST this much
        contention", and a mean would let a run that started quiet and ended
        saturated file as merely moderate.
        """
        from verify_budget_census import band_pressure  # noqa: PLC0415

        assert band_pressure(_load_record(cpu=1.0, end_cpu=75.0)) == 75.0
        assert band_pressure(_load_record(cpu=75.0, end_cpu=1.0)) == 75.0
        assert band_pressure(_load_record(cpu=None, end_cpu=None)) is None
        assert band_pressure(None) is None

    def test_the_smoothed_window_is_not_folded_into_the_band(self):
        """`cpu_some60` is a DIFFERENT averaging window, and the band edges were
        argued for avg10 — mixing them would make the edges mean something
        nobody stated. It is reported by `load_movement` instead."""
        from verify_budget_census import band_pressure  # noqa: PLC0415

        load = _load_record(cpu=1.0, cpu60=90.0, end_cpu=2.0, end_cpu60=95.0)

        assert band_pressure(load) == 2.0


class TestHostMovementIsReported:
    """The end reading reaches a reader, not just the banding statistic.

    A band alone cannot show that a run began idle and ended saturated, and that
    movement is what tells an operator whether a duration is a statement about
    the suite or about the fleet's occupancy while it ran.
    """

    def _movement(self, tmp_path, *entries):
        from verify_budget_census import (  # noqa: PLC0415
            load_movement,
            load_records,
            select_full_suite_legs,
        )

        write_corpus(tmp_path, *entries)
        legs = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
        ).legs
        return load_movement(legs)

    def test_a_rise_across_bands_is_counted_as_a_rise(self, tmp_path):
        movement = self._movement(
            tmp_path, command_entry(load=_load_record(cpu=1.0, end_cpu=75.0)),
        )

        assert movement['rose'] == 1
        assert movement['held'] == movement['fell'] == 0

    def test_a_fall_across_bands_is_counted_as_a_fall(self, tmp_path):
        movement = self._movement(
            tmp_path, command_entry(load=_load_record(cpu=75.0, end_cpu=1.0)),
        )

        assert movement['fell'] == 1
        assert movement['held'] == movement['rose'] == 0

    def test_a_move_inside_one_band_is_held(self, tmp_path):
        """Counted in BANDS, because a 3-point rise inside one band is not a fact
        anyone would act on."""
        movement = self._movement(
            tmp_path, command_entry(load=_load_record(cpu=60.0, end_cpu=63.0)),
        )

        assert movement['held'] == 1
        assert movement['rose'] == movement['fell'] == 0

    def test_a_half_degraded_record_is_not_knowable_yet_still_banded(self, tmp_path):
        """The two counts differ DELIBERATELY: a movement needs both instants, a
        band needs only one. Asserted together so neither drifts into the other."""
        from verify_budget_census import (  # noqa: PLC0415
            by_load_band,
            load_movement,
            load_records,
            select_full_suite_legs,
        )

        write_corpus(
            tmp_path,
            command_entry(duration_secs=4600.0, load=_load_record(cpu=75.0, end_cpu=None)),
        )
        legs = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
        ).legs

        assert load_movement(legs)['not_knowable'] == 1
        assert by_load_band(legs)['unstamped']['durations']['n'] == 0

    def test_the_smoothed_end_of_run_reading_is_surfaced(self, tmp_path):
        """`cpu_some60` at the END is the one reading describing an INTERVAL of
        the run itself rather than an instant before or after it."""
        movement = self._movement(
            tmp_path,
            command_entry(load=_load_record(cpu=1.0, end_cpu=70.0, end_cpu60=40.0)),
            command_entry(load=_load_record(cpu=1.0, end_cpu=70.0, end_cpu60=60.0)),
        )

        assert movement['end_cpu_some60']['n'] == 2
        assert movement['end_cpu_some60']['max'] == 60.0

    def test_a_degraded_smoothed_reading_is_absent_not_zero(self, tmp_path):
        movement = self._movement(
            tmp_path, command_entry(load=_load_record(cpu=1.0, end_cpu=70.0, end_cpu60=None)),
        )

        assert movement['end_cpu_some60'] == {
            'n': 0, 'p50': None, 'p90': None, 'max': None,
        }

    def test_it_rides_in_the_report_and_prints(self, tmp_path, capsys):
        write_corpus(tmp_path, command_entry(load=_load_record(cpu=1.0, end_cpu=75.0)))

        _rc, out, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator', '--json',
        )
        assert json.loads(out)['load_movement']['rose'] == 1

        _rc, text, _err = run_main(
            capsys, '--root', str(tmp_path), '--module', 'orchestrator',
        )
        assert 'HOST MOVEMENT DURING THE RUN' in text
        assert 'rose 1' in text


class TestTheProducerAndTheConsumerAgreeOnTheLoadShape:
    """The only seam in this deliverable that crosses a module boundary.

    Everything else on both sides hand-builds the `load` dict — `_load_record`
    here, `TestCheckRunLoad` and `TestLoadReachesTheSummaryPayload._load` in
    `orchestrator/tests/test_verify_load_stamp.py` — so neither suite can see a
    RENAME. Renesting a key in `verify._load_sample`, or in the
    `{'start','end','xdist'}` composition at its single call site, would leave
    all of them green while the census silently booked every run under
    `unstamped` — the bucket whose whole purpose is to never be confused with a
    reading. That is the same whitelist-drift failure `segments` actually
    suffered, at the one place no single-module test can reach.

    So this drives the REAL `run_verification` into a corpus-shaped tmp root and
    reads the summary.json it WRITES with the REAL census.
    """

    _START_CPU = 1.0
    _END_CPU = 75.0
    _END_CPU60 = 40.0

    def _legs_from_a_real_verify(self, tmp_path, monkeypatch):
        import asyncio  # noqa: PLC0415
        from unittest.mock import patch  # noqa: PLC0415

        from orchestrator.config import ModuleConfig, OrchestratorConfig  # noqa: PLC0415
        from shared.psi import PsiSample  # noqa: PLC0415
        from verify_budget_census import (  # noqa: PLC0415
            load_records,
            select_full_suite_legs,
        )

        from orchestrator import verify  # noqa: PLC0415

        # `OrchestratorConfig(project_root=...)` reads an ambient
        # ORCH_CONFIG_PATH, so without this the "default" config is whichever
        # project the runner happens to point at — the same leak
        # `_target_subprocess_env` scrubs for child processes (task 2957).
        # Pointed at a guaranteed-ABSENT file rather than deleted, which is the
        # spelling orchestrator/tests/conftest.py's `code_default_config` uses
        # (that fixture is not visible from this directory): an absent path makes
        # the project layer skip itself and leaves `defaults.yaml` loading, while
        # an unset var falls back to the RELATIVE `config.yaml` and so depends on
        # the runner's cwd.
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(tmp_path / 'no-such-config.yaml'))

        def sample(cpu, cpu60):
            return PsiSample(
                cpu_some10=cpu, cpu_some60=cpu60, mem_some10=0.0, mem_full10=0.0,
                io_some10=0.0, read_ok=True, runqueue_ratio=0.25,
                runqueue_read_ok=True,
            )

        # Exactly two readings, start then end. A third call exhausts the
        # iterator, which `_load_sample`'s never-raise wrapper turns into null
        # PSI fields — and the assertions below then fail loudly rather
        # than passing on a fabricated reading.
        readings = iter((
            sample(self._START_CPU, 0.5),
            sample(self._END_CPU, self._END_CPU60),
        ))
        real_load_sample = verify._load_sample

        async def fake_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **_kw):
            return 0, 'ok', False

        # The worktree IS the corpus path: `<root>/.worktrees/<lane>/` is where
        # `load_records` looks, and `run_verification` persists into
        # `<worktree>/.task/verify/` on its own.
        worktree = tmp_path / '.worktrees' / '3353'
        (worktree / '.task').mkdir(parents=True, exist_ok=True)
        write_module_yaml(tmp_path)

        with patch.object(verify, '_run_cmd', side_effect=fake_run_cmd), \
             patch.object(
                 verify, '_load_sample',
                 side_effect=lambda: real_load_sample(read=lambda: next(readings)),
             ):
            asyncio.run(
                verify.run_verification(
                    worktree,
                    OrchestratorConfig(
                        project_root=tmp_path, verify_admission_enabled=False,
                    ),
                    ModuleConfig(prefix='orchestrator', test_command=FULL_SUITE),
                    attempt_id=1,
                    task_id='3353',
                    max_retries=0,
                ),
            )

        selection = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
        )
        assert len(selection.legs) == 1, (
            f'the real producer wrote nothing the census selects: '
            f'{selection.rejected_entries} / {selection.rejected_records}'
        )
        return selection.legs

    def test_the_census_reads_a_pressure_out_of_a_real_record(
        self, tmp_path, monkeypatch,
    ):
        """`band_pressure` indexes the producer's own keys, not a literal."""
        from verify_budget_census import band_pressure  # noqa: PLC0415

        legs = self._legs_from_a_real_verify(tmp_path, monkeypatch)

        assert band_pressure(legs[0].load) == self._END_CPU

    def test_a_real_record_bands_instead_of_landing_in_unstamped(
        self, tmp_path, monkeypatch,
    ):
        """The failure this guard exists for: a renamed key books every run as
        "load not knowable" while every single-module test stays green."""
        from verify_budget_census import by_load_band  # noqa: PLC0415

        bands = by_load_band(self._legs_from_a_real_verify(tmp_path, monkeypatch))

        assert bands['unstamped']['durations']['n'] == 0
        assert bands[_heavy_band_name()]['durations']['n'] == 1

    def test_the_movement_section_reads_both_instants_of_a_real_record(
        self, tmp_path, monkeypatch,
    ):
        from verify_budget_census import load_movement  # noqa: PLC0415

        movement = load_movement(self._legs_from_a_real_verify(tmp_path, monkeypatch))

        assert movement['rose'] == 1
        assert movement['not_knowable'] == 0
        assert movement['end_cpu_some60']['max'] == self._END_CPU60


class TestColdSeparabilityIsReportedNotInferred:
    """D17's ruled fallback, asserted as a property of the report.

    A summary.json carries no is-cold flag. The only available inference —
    "attempt-1 in a worktree with no prior verify dir is cold" — would mix warm
    reruns into a cold distribution and then get frozen into a budget as if
    measured. D17 anticipates this and rules the fallback: label the cold value
    INTERIM with its basis. So the census says it CANNOT separate them, counts
    the records it could only have guessed at, and labels no series "cold".
    """

    def _finding(self, tmp_path, *entries):
        from verify_budget_census import (  # noqa: PLC0415
            cold_separability,
            load_records,
            select_full_suite_legs,
        )

        write_corpus(tmp_path, *entries)
        legs = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
        ).legs
        return cold_separability(legs)

    def test_the_finding_says_not_separable(self, tmp_path):
        finding = self._finding(tmp_path, command_entry())

        assert finding['cold_separable'] is False
        assert finding['basis'], 'the finding must state WHY it cannot separate'

    def test_the_guessable_records_are_counted(self, tmp_path):
        """attempt-1 records are the ones a cold inference would have claimed."""
        from verify_budget_census import (  # noqa: PLC0415
            cold_separability,
            load_records,
            select_full_suite_legs,
        )

        write_corpus(tmp_path, command_entry(), lane='100')
        second = write_worktree_record(tmp_path, '200', 'attempt-3.orchestrator.summary.json')
        second.write_text(json.dumps(summary_payload(command_entry())), encoding='utf-8')

        legs = select_full_suite_legs(
            load_records([tmp_path]), expected=FULL_SUITE, prefix='orchestrator',
        ).legs
        finding = cold_separability(legs)

        assert finding['first_attempt_records'] == 1
        assert finding['later_attempt_records'] == 1

    def test_no_series_is_labelled_cold(self, tmp_path):
        """The assertion that stops a guess becoming a measurement."""
        finding = self._finding(tmp_path, command_entry())

        assert 'cold' not in {k.lower() for k in finding} - {'cold_separable'}
        assert not any(
            isinstance(v, dict) and 'p50' in v for v in finding.values()
        ), 'a duration series here would be a cold distribution built on a guess'


class TestTheMergeGateBudgetIsReported:
    """Reported, never changed. The merge gate is ALWAYS cold.

    It verifies a freshly-created worktree every time, so its budget is a
    different question from the task lane's — and this census does not answer
    it. Saying so in the report is what stops a reader applying a warm-derived
    figure to a strictly costlier path.
    """

    def test_the_project_yaml_value_is_read(self, tmp_path):
        from verify_budget_census import merge_gate_budget  # noqa: PLC0415

        (tmp_path / 'dark-factory-orchestrator.yaml').write_text(
            'merge_verify_cold_command_timeout_secs: 7200\n', encoding='utf-8',
        )

        readout = merge_gate_budget(tmp_path)

        assert readout['merge_verify_cold_command_timeout_secs'] == 7200
        assert readout['source'].endswith('dark-factory-orchestrator.yaml')

    def test_the_defaults_yaml_is_the_fallback(self, tmp_path):
        """Unset in the project yaml is the LIVE state — the default governs."""
        from verify_budget_census import merge_gate_budget  # noqa: PLC0415

        (tmp_path / 'dark-factory-orchestrator.yaml').write_text(
            'max_concurrent_tasks: 4\n', encoding='utf-8',
        )
        defaults = tmp_path / 'orchestrator' / 'src' / 'orchestrator' / 'defaults.yaml'
        defaults.parent.mkdir(parents=True)
        defaults.write_text(
            'merge_verify_cold_command_timeout_secs: 7200\n', encoding='utf-8',
        )

        readout = merge_gate_budget(tmp_path)

        assert readout['merge_verify_cold_command_timeout_secs'] == 7200
        assert readout['source'].endswith('defaults.yaml')

    def test_an_unresolvable_budget_is_null_not_guessed(self, tmp_path):
        from verify_budget_census import merge_gate_budget  # noqa: PLC0415

        readout = merge_gate_budget(tmp_path)

        assert readout['merge_verify_cold_command_timeout_secs'] is None
        assert readout['source'] is None

    def test_the_readout_states_the_merge_gate_is_always_cold(self, tmp_path):
        from verify_budget_census import merge_gate_budget  # noqa: PLC0415

        readout = merge_gate_budget(tmp_path)

        assert readout['always_cold'] is True
        assert 'census changes nothing' in readout['note'].lower()
