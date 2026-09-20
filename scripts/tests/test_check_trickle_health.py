"""Tests for scripts/legibility/check_trickle_health.py — the thing that
finally RUNS the two trickle probes.

The defect this module closes is not a wrong verdict, it is an ABSENT
CALLER: no systemd unit, cron entry or config bound either
``check_trickle_progress.py`` or ``check_trickle_liveness.sh``, and the
only bindings either ever had were the one-shot ``before_done`` milestone
predicates on tasks 2587/2615 — both ``done``, and a completed milestone
predicate never runs again. So the tests that matter most here are the
ones pinning that this module EXECUTES the sibling scripts by path.

Every test injects BOTH probe runners. None may shell out to a real
``systemctl`` or run the real progress probe against the operator's live
state; ``scripts/tests/conftest.py::_isolate_legibility_trickle_state`` is
the backstop if one ever slips.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from legibility import check_trickle_health, trickle_state

SCRIPT = Path(__file__).parent.parent / 'legibility' / 'check_trickle_health.py'


def _ok(*_args, **_kwargs):
    return 0, 'OK: probe passed'


def _fails(message):
    def _runner(*_args, **_kwargs):
        return 1, message
    return _runner


def _raises(exc):
    def _runner(*_args, **_kwargs):
        raise exc
    return _runner


def _check(**kwargs):
    """Run the aggregate with both runners injected and escalation off."""
    kwargs.setdefault('project_id', 'dark_factory')
    kwargs.setdefault('progress_runner', _ok)
    kwargs.setdefault('liveness_runner', _ok)
    kwargs.setdefault('poster', lambda url, envelope: None)
    return check_trickle_health.run_health_check(**kwargs)


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), (
        f'Expected {SCRIPT} to be executable (os.X_OK); it is not. A systemd '
        f'ExecStart and every sibling probe in this directory rely on it. '
        f'Run: chmod +x {SCRIPT}'
    )


class TestAggregateVerdict:
    """ONE verdict for the whole invocation, never one per probe — the same
    rule ``run_nightly`` states as "ONE escalation for the whole night"."""

    def test_both_probes_passing_is_healthy(self):
        result = _check()

        assert result.exit_code == 0
        assert result.progress_ok is True
        assert result.liveness_ok is True
        assert 'progress' in result.reason.lower()
        assert 'liveness' in result.reason.lower()

    def test_a_failing_progress_probe_fails_the_aggregate(self):
        result = _check(progress_runner=_fails('ERROR: barren streak'))

        assert result.exit_code == 1
        assert result.progress_ok is False
        assert result.liveness_ok is True

    def test_a_failing_liveness_probe_fails_the_aggregate(self):
        """The mitigation ``nightly::_record_trickle_progress`` claims is
        only real if this fails LOUD."""
        result = _check(liveness_runner=_fails('ERROR: Result=failed'))

        assert result.exit_code == 1
        assert result.progress_ok is True
        assert result.liveness_ok is False

    def test_both_probes_run_even_when_the_first_fails(self):
        """Never short-circuits: the operator needs BOTH answers in one
        place, and a progress failure is often explained by the liveness
        one."""
        result = _check(
            progress_runner=_fails('ERROR: progress detail here'),
            liveness_runner=_fails('ERROR: liveness detail here'),
        )

        assert result.exit_code == 1
        assert result.progress_ok is False
        assert result.liveness_ok is False
        assert 'progress detail here' in result.progress_output
        assert 'liveness detail here' in result.liveness_output

    @pytest.mark.parametrize('exc', [
        pytest.param(FileNotFoundError('no such probe'), id='FileNotFoundError'),
        pytest.param(
            subprocess.TimeoutExpired(cmd=['probe'], timeout=1.0),
            id='TimeoutExpired',
        ),
    ])
    def test_a_probe_that_cannot_RUN_is_a_failed_probe(self, exc):
        """A probe that cannot run is not evidence of health. Never a pass,
        never an uncaught traceback."""
        result = _check(progress_runner=_raises(exc))

        assert result.exit_code == 1
        assert result.progress_ok is False
        assert type(exc).__name__ in result.progress_output, (
            'the reason a probe could not run must be named, not swallowed'
        )


class TestTheProbesItActuallyTargets:
    """THE GAP-2 regression tests. The defect was that nothing referenced
    either probe, so pinning the argv BY CONSTRUCTION is what keeps a
    rename from silently orphaning one again."""

    def test_progress_argv_targets_the_sibling_script(self):
        argv = check_trickle_health.progress_argv(
            'dark_factory', max_barren_runs=3, max_age_hours=72,
            max_failed_runs=2,
        )

        assert argv == [
            sys.executable,
            str(SCRIPT.parent / 'check_trickle_progress.py'),
            'dark_factory', '3', '72', '2',
        ]

    def test_liveness_argv_targets_the_sibling_script(self):
        argv = check_trickle_health.liveness_argv('dark_factory', hours=72)

        assert argv == [
            'bash',
            str(SCRIPT.parent / 'check_trickle_liveness.sh'),
            'dark_factory', '72',
        ]

    def test_both_probe_paths_exist(self):
        assert check_trickle_health.PROGRESS_SCRIPT.is_file()
        assert check_trickle_health.LIVENESS_SCRIPT.is_file()


class TestCli:
    """``main(argv) -> exit-code``, the shape a systemd ExecStart runs."""

    def test_exactly_one_of_config_or_project_id_is_required(self, capsys):
        with pytest.raises(SystemExit):
            check_trickle_health.main([])
        assert 'project-id' in capsys.readouterr().err

    def test_defaults_are_read_from_the_module_not_hardcoded(self):
        """The two halves cannot drift: the probe's default thresholds ARE
        ``trickle_state``'s constants, by reference."""
        parser = check_trickle_health.build_parser()
        defaults = vars(parser.parse_args(['--project-id', 'dark_factory']))

        assert defaults['max_barren_runs'] == trickle_state.DEFAULT_MAX_BARREN_RUNS
        assert defaults['max_failed_runs'] == trickle_state.DEFAULT_MAX_FAILED_RUNS
        assert defaults['max_age_hours'] == 72

    def test_the_reason_goes_to_stderr_on_a_non_zero_exit(
        self, capsys, monkeypatch
    ):
        monkeypatch.setattr(
            check_trickle_health, 'run_health_check',
            lambda **kw: check_trickle_health.HealthResult(
                exit_code=1, progress_ok=False, liveness_ok=True,
                progress_output='', liveness_output='', escalated=False,
                reason='synthetic failure reason',
            ),
        )

        assert check_trickle_health.main(['--project-id', 'dark_factory']) == 1

        captured = capsys.readouterr()
        assert 'synthetic failure reason' in captured.err
        assert 'synthetic failure reason' not in captured.out

    def test_the_reason_goes_to_stdout_when_healthy(self, capsys, monkeypatch):
        monkeypatch.setattr(
            check_trickle_health, 'run_health_check',
            lambda **kw: check_trickle_health.HealthResult(
                exit_code=0, progress_ok=True, liveness_ok=True,
                progress_output='', liveness_output='', escalated=False,
                reason='synthetic healthy reason',
            ),
        )

        assert check_trickle_health.main(['--project-id', 'dark_factory']) == 0

        captured = capsys.readouterr()
        assert 'synthetic healthy reason' in captured.out
        assert 'synthetic healthy reason' not in captured.err
