"""scripts/verify_budget_census.py's CPU AND PEAK MEMORY section: what the clean
full-suite legs cost, read from each command's C4 ``resources`` record.

The record's shape and the meaning of each capture state live in
orchestrator/src/orchestrator/verify_command_resources.py; the census reads the
summary JSON and imports none of it, so the records here are built by hand.
Only the pin on the census's restated capture states imports the record's own.

The synthetic-corpus norm lives in tests/scripts/verify_budget_census_fixture.py.
"""
from __future__ import annotations

import json
from typing import Any

import pytest
import verify_budget_census
from verify_budget_census_fixture import (
    command_entry,
    run_main,
    summary_payload,
    write_corpus,
    write_worktree_record,
)

WINDOW = '2026-09-12T00:00:00+00:00..2026-09-14T12:00:00+00:00'
GIB = 1024 ** 3


def _resources(capture: str = 'exit', **overrides: Any) -> dict[str, Any]:
    """A C4 record as ``CommandResources.to_dict`` writes it."""
    measured = capture != 'unavailable'
    record: dict[str, Any] = {
        'capture': capture,
        'capture_reason': f'fixture {capture}',
        'relocated_scopes': ['run-u1.scope'] if capture == 'relocated' else [],
        'cpu_usec': 600_000_000 if measured else None,
        'user_usec': 500_000_000 if measured else None,
        'system_usec': 100_000_000 if measured else None,
        'memory_peak_bytes': GIB if measured else None,
        'swap_peak_bytes': 0 if measured else None,
        'memory_high_events': 0 if measured else None,
        'memory_max_events': 0 if measured else None,
        'oom_kills': 0 if measured else None,
        'cpu_stall_some_usec': 1_000 if measured else None,
        'memory_stall_some_usec': 0 if measured else None,
        'memory_stall_full_usec': 0 if measured else None,
        'io_stall_some_usec': 0 if measured else None,
        'scope_unit': 'df-verify-dark-factory-c889f090-a3825b3fefa0.scope',
        'host': 'fixture-host',
        'project': 'dark_factory',
        'role': 'task',
        'task_id': '3353',
        'module': 'orchestrator',
        'leg': 'test',
        'segment': None,
        'plan_id': None,
        'attempt': 1,
        'started_at': '2026-09-13T04:00:00+00:00',
        'ended_at': '2026-09-13T04:55:00+00:00',
    }
    record.update(overrides)
    return record


def _cost(capsys: pytest.CaptureFixture[str], *argv: str) -> dict[str, Any]:
    rc, out, _err = run_main(capsys, *argv, '--json', '--window', WINDOW)
    assert rc == 0
    return json.loads(out)['resources']


def _three_exit_legs() -> tuple[dict[str, Any], ...]:
    return (
        command_entry(resources=_resources(cpu_usec=600_000_000, memory_peak_bytes=GIB)),
        command_entry(resources=_resources(cpu_usec=1_200_000_000, memory_peak_bytes=2 * GIB)),
        command_entry(
            resources=_resources(cpu_usec=1_800_000_000, memory_peak_bytes=7 * GIB // 2),
        ),
    )


class TestTheCaptureStates:

    def test_the_census_copy_of_the_states_matches_the_record(self):
        """A state the census does not know reads as malformed."""
        from orchestrator.verify_command_resources import Capture

        assert set(verify_budget_census.C4_CAPTURE_STATES) == {c.value for c in Capture}


class TestCleanLegsCapturedAtExitMakeTheSeries:

    def test_cpu_seconds_and_peak_bytes_are_series_over_exit_captures(self, tmp_path, capsys):
        """cpu secs [600, 1200, 1800]: p50 = 1200, p90 = 1200 + 0.8 * 600 = 1680.
        peaks [1, 2, 3.5] GiB: p50 = 2 GiB, p90 = 2 + 0.8 * 1.5 = 3.2 GiB."""
        write_corpus(tmp_path, *_three_exit_legs())

        cost = _cost(capsys, '--root', str(tmp_path))

        assert cost['cpu_secs'] == {'n': 3, 'p50': 1200.0, 'p90': 1680.0, 'max': 1800.0}
        assert cost['memory_peak_bytes'] == {
            'n': 3, 'p50': 2.0 * GIB, 'p90': 3.2 * GIB, 'max': 3.5 * GIB,
        }
        assert cost['captures'] == {'exit': 3}

    def test_every_clean_leg_is_counted_under_exactly_one_capture(self, tmp_path, capsys):
        write_corpus(
            tmp_path,
            *_three_exit_legs(),
            command_entry(resources=_resources('relocated')),
            command_entry(resources=None),
        )
        rc, out, _err = run_main(capsys, '--root', str(tmp_path), '--json', '--window', WINDOW)
        report = json.loads(out)

        assert rc == 0
        assert sum(report['resources']['captures'].values()) == (
            report['overall']['durations']['n']
        )


class TestWhichLegsCount:

    def test_relocated_and_unavailable_are_counted_but_not_measured(self, tmp_path, capsys):
        write_corpus(
            tmp_path,
            command_entry(resources=_resources(cpu_usec=600_000_000)),
            command_entry(resources=_resources('relocated', cpu_usec=9_000_000_000)),
            command_entry(resources=_resources('unavailable')),
        )

        cost = _cost(capsys, '--root', str(tmp_path))

        assert cost['captures'] == {'exit': 1, 'relocated': 1, 'unavailable': 1}
        assert cost['cpu_secs']['n'] == 1
        assert cost['cpu_secs']['max'] == 600.0
        assert cost['memory_peak_bytes']['n'] == 1

    def test_a_leg_without_a_record_is_unrecorded(self, tmp_path, capsys):
        """``resources: null`` and a record written before task 6355 (no key)."""
        write_corpus(
            tmp_path,
            command_entry(resources=None),
            command_entry(),
        )

        cost = _cost(capsys, '--root', str(tmp_path))

        assert cost['captures'] == {'unrecorded': 2}
        assert cost['cpu_secs'] == {'n': 0, 'p50': None, 'p90': None, 'max': None}
        assert cost['memory_peak_bytes']['n'] == 0

    def test_timed_out_and_failed_legs_are_left_out_entirely(self, tmp_path, capsys):
        """The same partition as the overall durations: a timed-out leg's cost
        is the budget's, a failed leg's is a stopped run's."""
        write_corpus(
            tmp_path,
            command_entry(resources=_resources(cpu_usec=600_000_000)),
            command_entry(
                rc=-9, timed_out=True,
                resources=_resources('before_kill', cpu_usec=9_000_000_000),
            ),
            command_entry(rc=1, resources=_resources(cpu_usec=9_000_000_000)),
        )

        cost = _cost(capsys, '--root', str(tmp_path))

        assert cost['captures'] == {'exit': 1}
        assert cost['cpu_secs']['max'] == 600.0


class TestMalformedRecordsNeverCrashTheCensus:

    @pytest.mark.parametrize(
        'resources',
        [
            _resources(cpu_usec=True),
            _resources(cpu_usec=-1),
            _resources(memory_peak_bytes='12'),
            _resources('teleported'),
            {'cpu_usec': 600_000_000},
            'exit',
            [600_000_000],
        ],
        ids=[
            'bool-cpu', 'negative-cpu', 'string-peak', 'unknown-capture',
            'no-capture', 'string', 'list',
        ],
    )
    def test_a_malformed_record_is_counted_and_the_rest_still_report(
        self, tmp_path, capsys, resources,
    ):
        write_corpus(
            tmp_path,
            command_entry(resources=_resources(cpu_usec=600_000_000)),
            command_entry(resources=resources),
        )

        cost = _cost(capsys, '--root', str(tmp_path))

        assert cost['captures'] == {'exit': 1, 'malformed': 1}
        assert cost['cpu_secs'] == {'n': 1, 'p50': 600.0, 'p90': 600.0, 'max': 600.0}


class TestTheRoleFilterAppliesToTheSection:

    def test_a_merge_record_is_left_out_of_a_task_census(self, tmp_path, capsys):
        write_corpus(tmp_path, command_entry(resources=_resources(cpu_usec=600_000_000)))
        merge = write_worktree_record(
            tmp_path, '_merge-abc123', 'attempt-1.orchestrator.summary.json',
        )
        merge.write_text(
            json.dumps(summary_payload(
                command_entry(resources=_resources(cpu_usec=9_000_000_000)),
            )),
            encoding='utf-8',
        )

        cost = _cost(capsys, '--root', str(tmp_path), '--role', 'task')

        assert cost['captures'] == {'exit': 1}
        assert cost['cpu_secs']['max'] == 600.0


def _cost_section(text: str) -> list[str]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith('CPU AND PEAK MEMORY'))
    end = next((i for i in range(start, len(lines)) if not lines[i].strip()), len(lines))
    return lines[start:end]


class TestTheTextReport:

    def test_the_section_shows_seconds_gib_captures_and_the_not_zero_note(
        self, tmp_path, capsys,
    ):
        write_corpus(
            tmp_path,
            *_three_exit_legs(),
            command_entry(resources=_resources('unavailable')),
            command_entry(),
        )

        rc, out, _err = run_main(capsys, '--root', str(tmp_path), '--window', WINDOW)
        section = _cost_section(out)

        assert rc == 0
        assert 'orchestrator/src/orchestrator/verify_command_resources.py' in section[0]
        cpu = next(line for line in section if 'CPU-seconds' in line)
        assert 'p50=   1200' in cpu and 'max=   1800' in cpu, cpu
        peak = next(line for line in section if 'peak GiB' in line)
        assert 'p50=    2.0' in peak and 'p90=    3.2' in peak and 'max=    3.5' in peak, peak
        captures = next(line for line in section if 'captures' in line)
        for token in ('exit=3', 'unavailable=1', 'unrecorded=1'):
            assert token in captures, captures
        note = ' '.join(section)
        assert 'NOT zero' in note, section

    def test_an_unmeasured_series_prints_dashes_not_zero(self, tmp_path, capsys):
        write_corpus(tmp_path, command_entry())

        _rc, out, _err = run_main(capsys, '--root', str(tmp_path), '--window', WINDOW)
        section = _cost_section(out)

        cpu = next(line for line in section if 'CPU-seconds' in line)
        assert 'max=      -' in cpu, cpu
