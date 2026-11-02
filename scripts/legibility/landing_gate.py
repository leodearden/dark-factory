"""Run a project's configured landing-gate command in its checkout, and say
whether a machine-written legibility output may land there (task 7448).

A verdict holds for the one HEAD its command ran at: a run during which HEAD
moved (the merge worker landing on the same checkout) is run again, up to
GATE_ATTEMPTS times, so the tree that lands is the tree that was gated."""
from __future__ import annotations

import contextlib
import enum
import os
import shlex
import signal
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from legibility import census_identity, config

OUTPUT_TAIL_CHARS = 4000
GATE_ATTEMPTS = 3


class GateStatus(enum.StrEnum):
    UNGATED = "ungated"
    PASSED = "passed"
    FAILED = "failed"
    DID_NOT_RUN = "did-not-run"
    HEAD_MOVED = "head-moved"


_LANDING_STATUSES = frozenset({GateStatus.UNGATED, GateStatus.PASSED})


@dataclass(frozen=True)
class GateVerdict:
    """One landing-gate run: its *status*, the command, its exit code (None
    unless it ran to an exit), the end of its stdout and stderr, why it did
    not run, when it did not, and the *head* the verdict holds for (None when
    ungated or when HEAD never held still)."""

    status: GateStatus
    argv: tuple[str, ...] = ()
    returncode: int | None = None
    output_tail: str = ""
    not_run_reason: str | None = None
    head: str | None = None

    @property
    def passed(self) -> bool:
        return self.status in _LANDING_STATUSES

    def describe(self) -> str:
        if self.status is GateStatus.UNGATED:
            return "no landing gate configured"
        gate = f"landing gate `{shlex.join(self.argv)}`"
        if self.status is GateStatus.PASSED:
            return f"{gate} passed"
        if self.status is GateStatus.FAILED:
            return f"{gate} exited {self.returncode}"
        if self.status is GateStatus.HEAD_MOVED:
            return f"{gate} was run {GATE_ATTEMPTS} times and HEAD moved during each run"
        return f"{gate} {self.not_run_reason}"


def _text(stream: str | bytes | None) -> str:
    # An injected runner's TimeoutExpired may carry bytes.
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream or ""


def _tail(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    joined = "\n".join(part for part in (_text(stdout), _text(stderr)) if part)
    return joined[-OUTPUT_TAIL_CHARS:]


def _kill_group(process: subprocess.Popen[str]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)


def _run_in_own_session(argv: list[str], *, cwd: str, timeout: int) -> subprocess.CompletedProcess[str]:
    """``subprocess.run`` for a command that may start its own children (``uv``
    starts ``pytest``): on timeout, or on any exception while waiting, the
    command's whole process group is killed, so nothing it started outlives
    the gate in the checkout."""
    with subprocess.Popen(
        argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        encoding="utf-8", errors="replace", start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(argv, timeout, output=stdout, stderr=stderr) from None
        except BaseException:
            _kill_group(process)
            raise
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _run_once(run: Callable[..., Any], repo: Path | str, argv: tuple[str, ...], timeout_secs: int) -> GateVerdict:
    try:
        completed = run(list(argv), cwd=str(repo), timeout=timeout_secs)
    except subprocess.TimeoutExpired as exc:
        return GateVerdict(
            GateStatus.DID_NOT_RUN, argv, None, _tail(exc.stdout, exc.stderr),
            f"timed out after {timeout_secs}s",
        )
    except (OSError, ValueError) as exc:
        return GateVerdict(GateStatus.DID_NOT_RUN, argv, None, "", f"could not be run: {exc}")
    status = GateStatus.PASSED if completed.returncode == 0 else GateStatus.FAILED
    return GateVerdict(status, argv, completed.returncode, _tail(completed.stdout, completed.stderr))


def _run_at_a_still_head(
    run: Callable[..., Any],
    repo: Path | str,
    argv: tuple[str, ...],
    timeout_secs: int,
    head_of: Callable[[Path | str], str],
) -> GateVerdict:
    verdict = GateVerdict(GateStatus.HEAD_MOVED, argv)
    for _ in range(GATE_ATTEMPTS):
        head = head_of(repo)
        verdict = _run_once(run, repo, argv, timeout_secs)
        if verdict.status is GateStatus.DID_NOT_RUN or head_of(repo) == head:
            return replace(verdict, head=head)
    return replace(verdict, status=GateStatus.HEAD_MOVED)


def run_gate(
    repo: Path | str,
    spec: config.LandingGate | None,
    *,
    runner: Callable[..., Any] | None = None,
    head_of: Callable[[Path | str], str] = census_identity.resolve_as_of_sha,
) -> GateVerdict:
    """*spec*'s command run with cwd *repo*, as a GateVerdict for the HEAD it
    ran at. Only exit 0 passes; a timeout, a launch failure or a HEAD that
    cannot be read did not run, and a HEAD that moved during every attempt
    refuses: each fails closed. With no *spec* the landing is ungated and
    nothing is launched. Never raises."""
    if spec is None:
        return GateVerdict(GateStatus.UNGATED)
    argv = tuple(spec.argv)
    try:
        return _run_at_a_still_head(runner or _run_in_own_session, repo, argv, spec.timeout_secs, head_of)
    except RuntimeError as exc:  # head_of's failure; _run_once raises nothing
        return GateVerdict(GateStatus.DID_NOT_RUN, argv, None, "", f"could not read HEAD: {exc}")
