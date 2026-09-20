#!/usr/bin/env python3
"""scripts/legibility/check_trickle_health.py — the thing that RUNS the two
trickle probes (task 4514).

THE WHOLE POINT IS THE CALLER, NOT THE VERDICT. Both probes already
existed and both were correct; what did not exist was anything that ran
them. Verified in task 4514: no systemd unit, cron entry or config bound
``check_trickle_progress.py`` or ``check_trickle_liveness.sh``, and the
only bindings either ever had were the one-shot ``before_done`` milestone
predicates on tasks 2587/2615 — both ``done``, and a completed milestone
predicate never runs again. That absence is why the ``classify_run``
vocabulary hole this same task closes stayed latent so long: nothing was
reading the verdict that would have shown it.

THE ADJACENT PRECEDENT, and the reason deploying this is tracked
SEPARATELY from shipping it: ``legibility-transcript-check@.{service,
timer}`` and its installer shipped under task 2901 ("wire the
transcript-persistence detector to run periodically"), that task is
``done``, and the timer is STILL not installed on this host. Shipping an
installer is demonstrably not the same as binding a probe.

IT EXECUTES THE PROBES; IT NEVER RE-IMPLEMENTS EITHER VERDICT. That is
what keeps the no-lockstep-duplication rule (INV-5) true BY
CONSTRUCTION — the exact drift that let a suppressed night read like a
quiet one for 14 nights (2026-07-16..29) — lets
``check_trickle_liveness.sh`` stay byte-identical as its own header
comment requires, and means the argv this module builds IS the contract a
future ``before_done`` predicate would use.

A SEPARATE UNIT, NOT A SECOND ``ExecStart`` ON ``legibility-trickle@``.
A failing probe must not flip ``legibility-trickle@<project>.service`` to
``Result=failed``, which would invert ``check_trickle_liveness.sh`` into a
permanent false alarm — the exact trade
``scripts/legibility/nightly.py::_escalate_barren_streak`` refuses in
writing.

NO STATE PATH IS PASSED TO THE PROGRESS PROBE, and it does not need one:
since task 4514 both this reader and the nightly writer resolve
``scripts/legibility/trickle_state.py::trickle_state_path`` from the
invoking user's passwd entry, independently of environment. That property
is what makes binding this timer safe at all — before it, a reader whose
ambient environment disagreed with the writer's read ``missing`` and
failed permanently.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Self-bootstrap for a bare `ExecStart=... check_trickle_health.py` run: a
# direct script invocation puts only scripts/legibility/ on sys.path, not
# scripts/, so `from legibility import ...` would not resolve. Skipped under
# pytest/normal package import, where __name__ is
# 'legibility.check_trickle_health'. Mirrors check_transcript_persistence.py's
# identical guard (this module needs no orchestrator/src entry).
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legibility import trickle_state  # noqa: E402

logger = logging.getLogger('legibility.check_trickle_health')

PROGRESS_SCRIPT = Path(__file__).resolve().parent / 'check_trickle_progress.py'
LIVENESS_SCRIPT = Path(__file__).resolve().parent / 'check_trickle_liveness.sh'

DEFAULT_MAX_AGE_HOURS = 72
"""Freshness window handed to the progress probe, matching PRD task kappa's.

Deliberately loose enough that a ``Persistent=true`` catch-up firing
BEFORE the nightly's own catch-up cannot false-alarm."""

PROBE_TIMEOUT_SECONDS = 120.0
"""Per-probe subprocess timeout. A probe that hangs must become a FAILED
verdict rather than wedging the unit until systemd kills it."""


@dataclass(frozen=True)
class HealthResult:
    """The outcome of one :func:`run_health_check` pass.

    ``exit_code`` is the authoritative loud signal (0 healthy, 1 = at
    least one probe failed) that a systemd unit surfaces. It is NEVER
    affected by whether the escalation POST landed: ``escalated`` records
    that separately, because a down escalation server must not be able to
    mask a verdict.
    """

    exit_code: int
    progress_ok: bool
    liveness_ok: bool
    progress_output: str
    liveness_output: str
    escalated: bool
    reason: str


def progress_argv(
    project_id: str, *, max_barren_runs: int, max_age_hours: int,
    max_failed_runs: int,
) -> list[str]:
    """Build the argv that RUNS the progress probe.

    Separated from the runner so the target can be asserted without
    executing anything — the path is pinned by construction, which is what
    stops a rename from silently orphaning the probe the way GAP 2 found
    it."""
    return [
        sys.executable, str(PROGRESS_SCRIPT), project_id,
        str(max_barren_runs), str(max_age_hours), str(max_failed_runs),
    ]


def liveness_argv(project_id: str, *, hours: int) -> list[str]:
    """Build the argv that RUNS the liveness probe. Under ``bash`` rather
    than ``sys.executable``: it is a shell script, and its own header
    comment forbids changing its semantics."""
    return ['bash', str(LIVENESS_SCRIPT), project_id, str(hours)]


def _run(argv: Sequence[str]) -> tuple[int, str]:
    """Execute *argv*, returning ``(returncode, stdout + stderr)``.

    A probe that cannot RUN is not evidence of health, so
    ``TimeoutExpired``/``OSError`` becomes a FAILED verdict carrying the
    exception's name — never re-raised into an uncaught traceback, and
    never swallowed into a pass."""
    try:
        completed = subprocess.run(
            list(argv), capture_output=True, text=True,
            timeout=PROBE_TIMEOUT_SECONDS, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return 1, (
            f'{type(exc).__name__}: could not run {argv[0]!r} '
            f'({exc}) — a probe that cannot run is not evidence of health'
        )
    return completed.returncode, (completed.stdout or '') + (completed.stderr or '')


def _default_progress_runner(project_id: str, **kwargs) -> tuple[int, str]:
    return _run(progress_argv(project_id, **kwargs))


def _default_liveness_runner(project_id: str, **kwargs) -> tuple[int, str]:
    return _run(liveness_argv(project_id, **kwargs))


def _invoke(runner, project_id: str, **kwargs) -> tuple[bool, str]:
    """Call one probe *runner*, absorbing a raise into a failed verdict.

    The default runners already convert their own subprocess failures; this
    guard covers an INJECTED runner and any exception class the default
    pair does not anticipate. Same reasoning either way: a probe that
    cannot run is not a probe that passed."""
    try:
        returncode, output = runner(project_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 — deliberately total
        return False, (
            f'{type(exc).__name__}: probe runner raised ({exc}) — treated as '
            f'a FAILED probe, never as a pass'
        )
    return returncode == 0, output


def _build_reason(progress_ok: bool, liveness_ok: bool, project_id: str) -> str:
    """One line naming BOTH probes' verdicts, always — an operator reading
    a journal entry should never have to run the other one to find out."""
    progress = 'ok' if progress_ok else 'FAILED'
    liveness = 'ok' if liveness_ok else 'FAILED'
    headline = (
        'legibility trickle health OK' if progress_ok and liveness_ok
        else 'legibility trickle health DEGRADED'
    )
    return (
        f'{headline} for {project_id}: progress={progress} '
        f'liveness={liveness}'
    )


def run_health_check(
    *,
    project_id: str | None = None,
    config_path: str | Path | None = None,
    max_barren_runs: int = trickle_state.DEFAULT_MAX_BARREN_RUNS,
    max_age_hours: int = DEFAULT_MAX_AGE_HOURS,
    max_failed_runs: int = trickle_state.DEFAULT_MAX_FAILED_RUNS,
    liveness_hours: int = DEFAULT_MAX_AGE_HOURS,
    progress_runner=None,
    liveness_runner=None,
    poster=None,
) -> HealthResult:
    """Run BOTH probes and aggregate them into ONE verdict.

    NEVER SHORT-CIRCUITS. Both probes run even when the first fails,
    because the operator needs both answers in one place and a progress
    failure is very often EXPLAINED by the liveness one. The aggregation
    into a single ``HealthResult`` rather than one per probe mirrors
    ``run_nightly``'s "ONE escalation for the whole night, not one per
    record" rule.

    *progress_runner* / *liveness_runner* are the dependency-injection
    seams, alongside *poster*. Each takes ``(project_id, **kwargs)`` and
    returns ``(returncode, output)``.
    """
    del config_path, poster  # escalation wiring arrives in a later step

    progress_ok, progress_output = _invoke(
        progress_runner if progress_runner is not None
        else _default_progress_runner,
        project_id or '',
        max_barren_runs=max_barren_runs,
        max_age_hours=max_age_hours,
        max_failed_runs=max_failed_runs,
    )
    liveness_ok, liveness_output = _invoke(
        liveness_runner if liveness_runner is not None
        else _default_liveness_runner,
        project_id or '',
        hours=liveness_hours,
    )

    return HealthResult(
        exit_code=0 if (progress_ok and liveness_ok) else 1,
        progress_ok=progress_ok,
        liveness_ok=liveness_ok,
        progress_output=progress_output,
        liveness_output=liveness_output,
        escalated=False,
        reason=_build_reason(progress_ok, liveness_ok, project_id or ''),
    )


def build_parser() -> argparse.ArgumentParser:
    """The CLI surface, built separately so its DEFAULTS are assertable.

    Every threshold default is read FROM ``trickle_state`` by reference,
    never spelled as a literal here, so the writer's constants and this
    probe's cannot drift."""
    parser = argparse.ArgumentParser(
        prog='check_trickle_health.py',
        description=(
            'Run both legibility trickle probes and report one verdict '
            '(task 4514).'
        ),
    )
    parser.add_argument(
        '--config', default=None, help="Path to the project's legibility.yaml.",
    )
    parser.add_argument(
        '--project-id', default=None,
        help='Project id, resolved via resolve_config_path.',
    )
    parser.add_argument(
        '--max-barren-runs', type=int,
        default=trickle_state.DEFAULT_MAX_BARREN_RUNS,
        help='Consecutive barren runs the progress probe tolerates.',
    )
    parser.add_argument(
        '--max-age-hours', type=int, default=DEFAULT_MAX_AGE_HOURS,
        help='Freshness window for the recorded run state.',
    )
    parser.add_argument(
        '--max-failed-runs', type=int,
        default=trickle_state.DEFAULT_MAX_FAILED_RUNS,
        help='Consecutive failed runs the progress probe tolerates.',
    )
    parser.add_argument(
        '--liveness-hours', type=int, default=DEFAULT_MAX_AGE_HOURS,
        help='Window handed to check_trickle_liveness.sh.',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint — what a systemd ``ExecStart`` runs.

    Prints the one-line reason to stderr on a non-zero exit and stdout
    otherwise, mirroring
    ``scripts/legibility/check_transcript_persistence.py::main``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.config and not args.project_id:
        parser.error('requires --config or --project-id')

    result = run_health_check(
        project_id=args.project_id,
        config_path=args.config,
        max_barren_runs=args.max_barren_runs,
        max_age_hours=args.max_age_hours,
        max_failed_runs=args.max_failed_runs,
        liveness_hours=args.liveness_hours,
    )
    print(result.reason, file=sys.stderr if result.exit_code else sys.stdout)
    return result.exit_code


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
