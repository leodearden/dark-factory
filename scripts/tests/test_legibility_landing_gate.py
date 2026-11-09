"""Tests for scripts/legibility/landing_gate.py: a project's landing-gate
command, run in its checkout, says whether a census output may land there
(task 7448). Real subprocesses (mostly ``sys.executable -c``), a one-commit
git repo under ``tmp_path`` as the checkout."""
from __future__ import annotations

import contextlib
import os
import shlex
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from legibility import config, landing_gate

from df_pytest_isolation import read_leaked_pid, wait_pid_gone

GateStatus = landing_gate.GateStatus


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.hooksPath=/dev/null"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "init"], check=True)
    return tmp_path


def _head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
    ).stdout.strip()


def _spec(code: str, *, timeout_secs: int = 60) -> config.LandingGate:
    return config.LandingGate(argv=(sys.executable, "-c", code), timeout_secs=timeout_secs)


def _heads(*shas: str) -> Callable[[Path | str], str]:
    """A HEAD reader that answers *shas* in turn, one per read."""
    answers = iter(shas)
    return lambda repo: next(answers)


class _CountingRunner:
    def __init__(self, returncode: int = 0) -> None:
        self.calls = 0
        self.returncode = returncode

    def __call__(self, argv, **kwargs) -> subprocess.CompletedProcess[str]:
        self.calls += 1
        return subprocess.CompletedProcess(argv, self.returncode, "", "")


def test_a_command_that_exits_zero_passes(checkout: Path) -> None:
    verdict = landing_gate.run_gate(checkout, _spec("pass"))

    assert verdict.status is GateStatus.PASSED
    assert verdict.passed is True
    assert verdict.returncode == 0


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_a_command_that_exits_non_zero_fails_with_its_output(checkout: Path, stream: str) -> None:
    code = f"import sys; print('boom', file=sys.{stream}); sys.exit(1)"

    verdict = landing_gate.run_gate(checkout, _spec(code))

    assert verdict.status is GateStatus.FAILED
    assert verdict.passed is False
    assert verdict.returncode == 1
    assert "boom" in verdict.output_tail


def test_the_command_runs_in_the_checkout(checkout: Path) -> None:
    verdict = landing_gate.run_gate(checkout, _spec("import os; print(os.getcwd())"))

    assert verdict.passed is True
    assert verdict.output_tail.strip() == str(checkout.resolve())


def test_a_command_that_outruns_its_timeout_did_not_run(checkout: Path) -> None:
    verdict = landing_gate.run_gate(checkout, _spec("import time; time.sleep(5)", timeout_secs=1))

    assert verdict.status is GateStatus.DID_NOT_RUN
    assert verdict.passed is False
    assert verdict.returncode is None
    assert "timed out" in verdict.describe()


def test_a_timeout_keeps_the_output_written_before_it(checkout: Path) -> None:
    def timing_out(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"], output=b"collected 3", stderr=b"still running")

    verdict = landing_gate.run_gate(checkout, _spec("pass"), runner=timing_out)

    assert verdict.status is GateStatus.DID_NOT_RUN
    assert "collected 3" in verdict.output_tail
    assert "still running" in verdict.output_tail


def test_a_command_that_cannot_be_launched_did_not_run(checkout: Path) -> None:
    spec = config.LandingGate(argv=(str(checkout / "no-such-executable"),))

    verdict = landing_gate.run_gate(checkout, spec)

    assert verdict.status is GateStatus.DID_NOT_RUN
    assert verdict.passed is False
    assert verdict.returncode is None
    assert "could not be run" in verdict.describe()


def test_no_gate_configured_is_ungated_and_launches_nothing(checkout: Path) -> None:
    def poison(*args, **kwargs):
        raise AssertionError("an ungated landing launched a command")

    verdict = landing_gate.run_gate(checkout, None, runner=poison)

    assert verdict.status is GateStatus.UNGATED
    assert verdict.passed is True


def test_long_output_keeps_only_its_end(checkout: Path) -> None:
    size = landing_gate.OUTPUT_TAIL_CHARS * 3
    code = f"import sys; sys.stdout.write('a' * {size} + 'THE-END'); sys.exit(1)"

    verdict = landing_gate.run_gate(checkout, _spec(code))

    assert len(verdict.output_tail) <= landing_gate.OUTPUT_TAIL_CHARS
    assert verdict.output_tail.endswith("THE-END")


def test_a_failed_verdict_describes_its_command_and_exit_code(checkout: Path) -> None:
    spec = _spec("import sys; sys.exit(3)")

    described = landing_gate.run_gate(checkout, spec).describe()

    assert "exited 3" in described
    assert sys.executable in described


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_a_timeout_kills_every_process_the_command_started(checkout: Path, tmp_path_factory) -> None:
    """A timeout kills every process the command started, not just the command. The grandchild
    holds none of the gate's pipes and the foreground is exec'd, so killing only the direct child
    would leave the grandchild running here rather than stall the gate on pipe EOF."""
    pidfile = tmp_path_factory.mktemp("grandchild") / "pid"
    command = f"sleep 300 >/dev/null 2>&1 & echo $! > {shlex.quote(str(pidfile))}; exec sleep 60"
    spec = config.LandingGate(argv=("/bin/sh", "-c", command), timeout_secs=3)

    verdict = landing_gate.run_gate(checkout, spec)

    assert verdict.status is GateStatus.DID_NOT_RUN
    grandchild = read_leaked_pid(pidfile)
    try:
        assert wait_pid_gone(grandchild, timeout=5), (
            f"pid {grandchild}, a grandchild the gate command started, is still running after the gate "
            "timed out: the timeout did not kill the command's whole process group"
        )
    finally:
        if not wait_pid_gone(grandchild, timeout=0):
            with contextlib.suppress(ProcessLookupError):
                os.kill(grandchild, signal.SIGKILL)


def test_output_that_is_not_utf_8_still_fails_with_its_output(checkout: Path) -> None:
    code = "import sys; sys.stdout.buffer.write(b'\\xff\\xfe boom'); sys.exit(1)"

    verdict = landing_gate.run_gate(checkout, _spec(code))

    assert verdict.status is GateStatus.FAILED
    assert verdict.returncode == 1
    assert "boom" in verdict.output_tail


def test_a_verdict_names_the_head_it_ran_at(checkout: Path) -> None:
    verdict = landing_gate.run_gate(checkout, _spec("pass"))

    assert verdict.head == _head(checkout)


def test_a_head_that_moves_during_a_run_runs_the_command_again(checkout: Path) -> None:
    runner = _CountingRunner()

    verdict = landing_gate.run_gate(checkout, _spec("pass"), runner=runner, head_of=_heads("a", "b", "b", "b"))

    assert runner.calls == 2
    assert verdict.status is GateStatus.PASSED
    assert verdict.head == "b"


def test_a_head_that_moves_during_every_run_refuses_the_landing(checkout: Path) -> None:
    runner = _CountingRunner()
    shas = [str(read) for read in range(2 * landing_gate.GATE_ATTEMPTS)]

    verdict = landing_gate.run_gate(checkout, _spec("pass"), runner=runner, head_of=_heads(*shas))

    assert runner.calls == landing_gate.GATE_ATTEMPTS
    assert verdict.status is GateStatus.HEAD_MOVED
    assert verdict.passed is False
    assert verdict.head is None
    assert "HEAD moved" in verdict.describe()


def test_a_run_that_did_not_finish_is_not_run_again_when_head_moves(checkout: Path) -> None:
    calls = []

    def timing_out(argv, **kwargs):
        calls.append(argv)
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    verdict = landing_gate.run_gate(checkout, _spec("pass"), runner=timing_out, head_of=_heads("a", "b"))

    assert len(calls) == 1
    assert verdict.status is GateStatus.DID_NOT_RUN
    assert "timed out" in verdict.describe()


def test_a_head_that_cannot_be_read_did_not_run_and_launches_nothing(tmp_path: Path) -> None:
    runner = _CountingRunner()

    def unreadable(repo):
        raise RuntimeError("git rev-parse HEAD failed")

    verdict = landing_gate.run_gate(tmp_path, _spec("pass"), runner=runner, head_of=unreadable)

    assert runner.calls == 0
    assert verdict.status is GateStatus.DID_NOT_RUN
    assert "could not read HEAD" in verdict.describe()
