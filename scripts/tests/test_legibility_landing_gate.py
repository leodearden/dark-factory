"""Tests for scripts/legibility/landing_gate.py: a project's landing-gate
command, run in its checkout, says whether a census output may land there
(task 7448). Real subprocesses through ``sys.executable -c``, ``tmp_path`` as
the checkout."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from legibility import config, landing_gate

GateStatus = landing_gate.GateStatus


def _spec(code: str, *, timeout_secs: int = 60) -> config.LandingGate:
    return config.LandingGate(argv=(sys.executable, "-c", code), timeout_secs=timeout_secs)


def test_a_command_that_exits_zero_passes(tmp_path: Path) -> None:
    verdict = landing_gate.run_gate(tmp_path, _spec("pass"))

    assert verdict.status is GateStatus.PASSED
    assert verdict.passed is True
    assert verdict.returncode == 0


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_a_command_that_exits_non_zero_fails_with_its_output(tmp_path: Path, stream: str) -> None:
    code = f"import sys; print('boom', file=sys.{stream}); sys.exit(1)"

    verdict = landing_gate.run_gate(tmp_path, _spec(code))

    assert verdict.status is GateStatus.FAILED
    assert verdict.passed is False
    assert verdict.returncode == 1
    assert "boom" in verdict.output_tail


def test_the_command_runs_in_the_checkout(tmp_path: Path) -> None:
    verdict = landing_gate.run_gate(tmp_path, _spec("import os; print(os.getcwd())"))

    assert verdict.passed is True
    assert verdict.output_tail.strip() == str(tmp_path.resolve())


def test_a_command_that_outruns_its_timeout_did_not_run(tmp_path: Path) -> None:
    verdict = landing_gate.run_gate(tmp_path, _spec("import time; time.sleep(5)", timeout_secs=1))

    assert verdict.status is GateStatus.DID_NOT_RUN
    assert verdict.passed is False
    assert verdict.returncode is None
    assert "timed out" in verdict.describe()


def test_a_timeout_keeps_the_output_written_before_it(tmp_path: Path) -> None:
    def timing_out(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output=b"collected 3", stderr=b"still running")

    verdict = landing_gate.run_gate(tmp_path, _spec("pass"), runner=timing_out)

    assert verdict.status is GateStatus.DID_NOT_RUN
    assert "collected 3" in verdict.output_tail
    assert "still running" in verdict.output_tail


def test_a_command_that_cannot_be_launched_did_not_run(tmp_path: Path) -> None:
    spec = config.LandingGate(argv=(str(tmp_path / "no-such-executable"),))

    verdict = landing_gate.run_gate(tmp_path, spec)

    assert verdict.status is GateStatus.DID_NOT_RUN
    assert verdict.passed is False
    assert verdict.returncode is None
    assert "could not be run" in verdict.describe()


def test_no_gate_configured_is_ungated_and_launches_nothing(tmp_path: Path) -> None:
    def poison(*args, **kwargs):
        raise AssertionError("an ungated landing launched a command")

    verdict = landing_gate.run_gate(tmp_path, None, runner=poison)

    assert verdict.status is GateStatus.UNGATED
    assert verdict.passed is True


def test_long_output_keeps_only_its_end(tmp_path: Path) -> None:
    size = landing_gate.OUTPUT_TAIL_CHARS * 3
    code = f"import sys; sys.stdout.write('a' * {size} + 'THE-END'); sys.exit(1)"

    verdict = landing_gate.run_gate(tmp_path, _spec(code))

    assert len(verdict.output_tail) <= landing_gate.OUTPUT_TAIL_CHARS
    assert verdict.output_tail.endswith("THE-END")


def test_a_failed_verdict_describes_its_command_and_exit_code(tmp_path: Path) -> None:
    spec = _spec("import sys; sys.exit(3)")

    described = landing_gate.run_gate(tmp_path, spec).describe()

    assert "exited 3" in described
    assert sys.executable in described
