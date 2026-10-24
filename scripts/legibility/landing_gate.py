"""Run a project's configured landing-gate command in its checkout, and say
whether a machine-written legibility output may land there (task 7448)."""
from __future__ import annotations

import enum
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from legibility import config

OUTPUT_TAIL_CHARS = 4000


class GateStatus(enum.StrEnum):
    UNGATED = "ungated"
    PASSED = "passed"
    FAILED = "failed"
    DID_NOT_RUN = "did-not-run"


_LANDING_STATUSES = frozenset({GateStatus.UNGATED, GateStatus.PASSED})


@dataclass(frozen=True)
class GateVerdict:
    """One landing-gate run: its *status*, the command, its exit code (None
    unless it ran to an exit), the end of its stdout and stderr, and why it
    did not run, when it did not."""

    status: GateStatus
    argv: tuple[str, ...] = ()
    returncode: int | None = None
    output_tail: str = ""
    not_run_reason: str | None = None

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
        return f"{gate} {self.not_run_reason}"


def _text(stream: str | bytes | None) -> str:
    # TimeoutExpired carries bytes even under text=True.
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream or ""


def _tail(stdout: str | bytes | None, stderr: str | bytes | None) -> str:
    joined = "\n".join(part for part in (_text(stdout), _text(stderr)) if part)
    return joined[-OUTPUT_TAIL_CHARS:]


def run_gate(
    repo: Path | str,
    spec: config.LandingGate | None,
    *,
    runner: Callable[..., Any] | None = None,
) -> GateVerdict:
    """*spec*'s command run with cwd *repo*, as a GateVerdict. Only exit 0
    passes; a timeout or a launch failure did not run, and fails closed.
    With no *spec* the landing is ungated and nothing is launched. Never
    raises."""
    if spec is None:
        return GateVerdict(GateStatus.UNGATED)
    argv = tuple(spec.argv)
    try:
        completed = (runner or subprocess.run)(
            list(argv), cwd=str(repo), capture_output=True, text=True, timeout=spec.timeout_secs,
        )
    except subprocess.TimeoutExpired as exc:
        return GateVerdict(
            GateStatus.DID_NOT_RUN, argv, None, _tail(exc.stdout, exc.stderr),
            f"timed out after {spec.timeout_secs}s",
        )
    except (OSError, ValueError) as exc:
        return GateVerdict(GateStatus.DID_NOT_RUN, argv, None, "", f"could not be run: {exc}")
    status = GateStatus.PASSED if completed.returncode == 0 else GateStatus.FAILED
    return GateVerdict(status, argv, completed.returncode, _tail(completed.stdout, completed.stderr))
