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
from dataclasses import dataclass, replace
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


def _unresolved_result(config_path) -> HealthResult:
    """The verdict when no project id could be resolved AT ALL.

    A THIRD outcome, not a fabricated probe failure, and worded distinctly
    on purpose: an operator reading the journal must be able to tell a
    MISCONFIGURED INVOCATION from a broken pipeline. ``_build_reason`` is
    for the two-probe case, so this line is composed separately rather
    than widening that function with a third state.

    Non-zero because ``_invoke``'s own rule — a probe that cannot run is
    not evidence of health — applies a fortiori to a probe that was never
    invoked."""
    return HealthResult(
        exit_code=1,
        progress_ok=False,
        liveness_ok=False,
        progress_output='',
        liveness_output='',
        escalated=False,
        reason=(
            f'legibility trickle health UNRESOLVED: no project id could be '
            f'resolved from config {config_path} and none was given, so '
            f'NEITHER probe ran. Pass --project-id, or point --config at a '
            f'readable legibility.yaml.'
        ),
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

    RESOLUTION. The project id comes from *project_id* if given, else from
    the config's own ``project_id``; *project_id* wins when both are
    present. GAP 3's identity argument applies here too — a probe that
    GUESSES at the project id reads a different state file than the writer
    wrote, which is the same divergence class task 4514 closed at the path
    layer. Measured before this resolution existed: ``--config`` alone sent
    ``''`` to both probes, yielding a guaranteed DEGRADED verdict and a
    spurious ``escalate_info`` for a perfectly healthy pipeline.

    *progress_runner* / *liveness_runner* are the dependency-injection
    seams, alongside *poster*. Each takes ``(project_id, **kwargs)`` and
    returns ``(returncode, output)``.
    """
    # Resolved ONCE, here, so every use below is a plain `project_id` — the
    # invariant is established at the boundary rather than re-defaulted at
    # six call sites.
    #
    # CONDITIONAL, NOT UNCONDITIONAL, and a later reader's first instinct
    # will be to "simplify" that away. The common production path is
    # `--project-id %i` from legibility-trickle-health@.service; on it the
    # config is needed only for `escalation_port`, and resolving it goes
    # through `nightly.resolve_config_path`, which imports `nightly` and
    # with it the whole trickle pipeline (census, codebook, coder, digest).
    # That load stays where it was: lazy, after `_should_escalate`, and
    # non-fatal. Only the `project_id is None` branch resolves early, and it
    # reaches `config.load_config` directly without touching `nightly`.
    cfg = None
    if project_id is None:
        cfg = _load_config(config_path=config_path, project_id=None)
        if cfg is None:
            return _unresolved_result(config_path)
        project_id = cfg.project_id

    progress_ok, progress_output = _invoke(
        progress_runner if progress_runner is not None
        else _default_progress_runner,
        project_id,
        max_barren_runs=max_barren_runs,
        max_age_hours=max_age_hours,
        max_failed_runs=max_failed_runs,
    )
    liveness_ok, liveness_output = _invoke(
        liveness_runner if liveness_runner is not None
        else _default_liveness_runner,
        project_id,
        hours=liveness_hours,
    )

    result = HealthResult(
        exit_code=0 if (progress_ok and liveness_ok) else 1,
        progress_ok=progress_ok,
        liveness_ok=liveness_ok,
        progress_output=progress_output,
        liveness_output=liveness_output,
        escalated=False,
        reason=_build_reason(progress_ok, liveness_ok, project_id),
    )

    # The escalation outcome NEVER changes exit_code: the non-zero exit is
    # the authoritative loud signal whether or not the POST landed.
    if not _load_and_decide(
        project_id, progress_ok, max_barren_runs, max_age_hours,
    ):
        return result
    # SPOT: one load per invocation. On the `--config`-only path the yaml
    # was already read above to resolve the project id; reuse it rather
    # than paying for a second parse of the same file.
    cfg = cfg if cfg is not None else _load_config(
        config_path=config_path, project_id=project_id,
    )
    if cfg is None:
        return result
    rerun = (
        progress_argv(
            project_id, max_barren_runs=max_barren_runs,
            max_age_hours=max_age_hours, max_failed_runs=max_failed_runs,
        ),
        liveness_argv(project_id, hours=liveness_hours),
    )
    return replace(
        result,
        escalated=post_health_finding(cfg, result, rerun, poster=poster),
    )


# ---------------------------------------------------------------------------
# Escalation — a best-effort mirror, copied by CONVENTION not by import
# ---------------------------------------------------------------------------
#
# The shape comes from
# `scripts/legibility/check_transcript_persistence.py`, which is itself a
# by-convention mirror of `scripts/legibility/nightly.py::post_escalation`.
# `nightly` is deliberately NOT imported: it pulls the whole trickle
# pipeline -- census, codebook, coder, digest -- at module load, which a
# probe that runs every night at 04:30 has no business paying for.
#
# `census_trigger.post_mcp_envelope` IS imported rather than re-derived, so
# the 400 -> `initialize` -> `notifications/initialized` handshake retry
# stays single-sourced. Task 3644 added it after every legibility
# escalation was being silently dropped by a stateful streamable-HTTP
# server that rejects a session-less `tools/call` at the transport layer.

_ESCALATION_TOOL_NAME = 'escalate_info'
_ESCALATION_AGENT_ROLE = 'legibility-trickle-health'
_ESCALATION_CATEGORY = 'infra_issue'
_ESCALATION_SEVERITY = 'info'


def _default_poster(url: str, envelope: dict) -> None:
    """Post *envelope* to *url* over the MCP streamable-HTTP transport."""
    from legibility import census_trigger  # lazy: keeps the probe path light

    census_trigger.post_mcp_envelope(url, envelope, timeout=10.0)


def _build_escalation_arguments(cfg, result: HealthResult, rerun) -> dict:
    """Build the ``escalate_info`` arguments for ONE invocation.

    The ``task_id`` is synthetic (``legibility-trickle-health-<project_id>``)
    because this is a timer-driven probe, not a Taskmaster task — the same
    reasoning ``nightly.py`` and ``check_transcript_persistence.py`` already
    record. It is deliberately DISTINCT from nightly's
    ``legibility-trickle-<project_id>`` so the two escalation histories stay
    separately readable.

    ONE envelope for the whole invocation, never one per probe: both
    probes' captured output goes into a single detail, because which probe
    failed is rarely the interesting question — the pair is.

    *rerun* is the ``(progress_argv, liveness_argv)`` pair THIS invocation
    actually ran, so the commands in the detail reproduce this verdict
    exactly rather than approximating it with stock thresholds."""
    failed = [
        name for name, ok in
        (('progress', result.progress_ok), ('liveness', result.liveness_ok))
        if not ok
    ]
    project_id = cfg.project_id
    return {
        'task_id': f'legibility-trickle-health-{project_id}',
        'agent_role': _ESCALATION_AGENT_ROLE,
        'category': _ESCALATION_CATEGORY,
        'severity': _ESCALATION_SEVERITY,
        'summary': (
            f'legibility trickle health: {" and ".join(failed)} probe(s) '
            f'FAILED for project {project_id}'
        ),
        'detail': (
            f'{result.reason}\n\n'
            f'--- check_trickle_progress.py output ---\n'
            f'{result.progress_output.strip() or "(no output)"}\n\n'
            f'--- check_trickle_liveness.sh output ---\n'
            f'{result.liveness_output.strip() or "(no output)"}\n\n'
            f'Re-run by hand (the exact commands this probe ran):\n'
            f'  {" ".join(rerun[0])}\n'
            f'  {" ".join(rerun[1])}'
        ),
    }


def post_health_finding(cfg, result: HealthResult, rerun, *, poster=None) -> bool:
    """Best-effort ``escalate_info`` POST. NEVER raises.

    Any failure is logged as ONE warning and swallowed, returning
    ``False``: a down escalation server must not mask the finding, because
    :attr:`HealthResult.exit_code` is the authoritative loud signal
    whether or not this POST landed."""
    poster_fn = poster if poster is not None else _default_poster
    try:
        envelope = {
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'tools/call',
            'params': {
                'name': _ESCALATION_TOOL_NAME,
                'arguments': _build_escalation_arguments(cfg, result, rerun),
            },
        }
        poster_fn(f'http://localhost:{cfg.escalation_port}/mcp', envelope)
        return True
    except Exception as exc:  # noqa: BLE001 — best-effort, never propagate
        logger.warning(
            'legibility trickle health: escalation post failed (best-effort, '
            'the run still exits non-zero): %s', exc,
        )
        return False


def _should_escalate(
    progress_ok: bool, doc, max_barren_runs: int, max_age_hours: int,
) -> bool:
    """Decide whether this invocation POSTS, as opposed to merely exiting
    non-zero. Narrower than the exit-code predicate, deliberately.

    ``False`` WHEN ``progress_ok``. A liveness-only failure means the unit
    RAN and FAILED, which is already owned twice for that run: by the
    nightly's own decision-8 escalation, and by
    ``check_trickle_liveness.sh``'s ``Result != success`` gate now that
    something finally runs it. Posting here is the double-alarm this task
    forbids. NO COVERAGE IS LOST, and the enumeration is what makes that
    checkable rather than merely asserted — a unit that NEVER RAN leaves
    the state ``missing``; one that STOPPED FIRING leaves ``recorded_at``
    stale; one that FAILS REPEATEDLY becomes a ``consecutive_failed_runs``
    streak. All three fail the PROGRESS probe and do post, and
    ``DEFAULT_MAX_FAILED_RUNS = 2`` rather than 1 is exactly what keeps
    night one out of the post set while night two is in it.

    ``False`` WHEN the recorded doc shows ``outcome == barren``,
    ``consecutive_barren_runs == max_barren_runs``, AND THE RECORD IS
    FRESH. ``nightly::_escalate_barren_streak`` is EDGE-triggered by exact
    equality and fired for THIS run; posting would duplicate it. At
    ``> max_barren_runs`` the nightly is silent by design and this probe
    takes over.

    THE FRESHNESS CLAUSE IS LOAD-BEARING, and it is where the defect
    lived. The suppression's entire justification is "the nightly fired
    for THIS run, so posting duplicates it" — and that justification holds
    only while the recorded run IS this run. Unscoped, it suppressed on
    the recorded document alone: measured, a ``['barren'] * 3`` document
    frozen 30 days ago failed the progress probe's STALENESS branch every
    night while never once posting, indefinitely. That is precisely the
    stopped-firing case the paragraph above claims "fails the PROGRESS
    probe and does post"; the enumeration was aspirational, and scoping
    the suppression by freshness is what makes it true. ``max_age_hours``
    is reused rather than given its own window BY DESIGN: it is already
    the window the progress probe uses to decide a record still describes
    a running pipeline, so the suppression becomes incapable of outliving
    the nightly's edge-trigger by construction.

    ``missing``/``malformed`` are POST-WORTHY verdicts, not suppression
    grounds — that is the recorder itself having stopped. An absent or
    unparseable ``recorded_at`` (``age is None``) takes the same posture
    and does NOT suppress: a record whose freshness cannot be assessed is
    never evidence that anyone alarmed.

    This reads three FIELDS off the recorded document. It does not
    re-derive either probe's verdict, so INV-5 stays intact."""
    if progress_ok:
        return False
    if not isinstance(doc, dict):
        return True
    if doc.get('outcome') != trickle_state.OUTCOME_BARREN:
        return True
    if doc.get('consecutive_barren_runs') != max_barren_runs:
        return True
    age = trickle_state.recorded_age_hours(doc)
    return age is None or age > max_age_hours


def _load_config(*, config_path, project_id):
    """Resolve the project's :class:`LegibilityConfig`, or ``None``.

    An unreadable or absent ``legibility.yaml`` must never crash the probe
    into a traceback: the verdict is already computed and already loud, and
    losing only the ESCALATION to a config problem degrades strictly better
    than losing the verdict too. ``nightly.resolve_config_path`` is imported
    lazily so the common ``--config`` path never pulls the trickle
    pipeline."""
    from legibility import config as legibility_config

    try:
        if config_path is not None:
            return legibility_config.load_config(config_path)
        from legibility import nightly  # lazy: avoid pulling the pipeline

        return legibility_config.load_config(
            nightly.resolve_config_path(project_id)
        )
    except Exception as exc:  # noqa: BLE001 — deliberately total
        logger.warning(
            'legibility trickle health: could not load config for project=%s '
            '(%s: %s); the verdict stands, the escalation is skipped',
            project_id, type(exc).__name__, exc,
        )
        return None


def _load_and_decide(
    project_id: str, progress_ok: bool, max_barren_runs: int,
    max_age_hours: int,
) -> bool:
    """Read the recorded doc and apply :func:`_should_escalate` to it."""
    _status, doc = trickle_state.load_state(
        trickle_state.trickle_state_path(project_id)
    )
    return _should_escalate(
        progress_ok, doc, max_barren_runs, max_age_hours,
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
