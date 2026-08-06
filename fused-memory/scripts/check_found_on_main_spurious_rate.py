#!/usr/bin/env python3
"""Predicate wrapper: found_on_main spurious-rate check (task 2676 / PRD label ι).

Thin, read-only wrapper around ``audit_found_on_main_provenance.py``
(exercised live by task 2648's audit run) for use as a ``before_done``
``kind='predicate'`` script under the DeterministicRunner convention (see
CLAUDE.md "Predicate exit-code contract"). It runs the existing audit, then
considers only the ``found_on_main`` done-provenance stamps belonging to
tasks whose ``updatedAt`` is strictly AFTER ``--since``, and asks whether
any of *those* were flagged ``misattributed`` or ``deliverable_absent`` by
the audit's classifier. Older stamps (already covered by a prior run, or
predating the fix this predicate soaks) are not re-flagged on every
invocation.

Freshness caveat: ``updatedAt`` is an approximation, not a dedicated
stamp-write timestamp — ``shared.task_metadata.DoneProvenance`` carries no
such field for ``kind='found_on_main'``. Any subsequent write to the task
(a re-tag, an unrelated metadata annotation, a dependency edit, ...) also
bumps ``updatedAt``, so a task whose found_on_main stamp actually predates
``--since`` but was touched afterwards for an unrelated reason can still
surface here. Because this predicate re-runs on resume and escalates on
any non-zero exit, a repeat escalation for an already-known/fixed task is
possible and may be benign — it does not necessarily mean a fresh
regression.

Contract (exit code only — this script parses no output, per the
DeterministicRunner predicate convention):

  - Exit 0: zero found_on_main stamps updated after ``--since`` were
    flagged ``misattributed``/``deliverable_absent``.
  - Exit 1: one or more were flagged (see the freshness caveat above — a
    repeat flag on an already-known/fixed task is possible and may be
    benign, not necessarily a fresh regression) — OR the task backend is
    not configured (an infra/config problem). This predicate's contract is
    intentionally coarse: non-zero-for-any-reason means "did not pass"
    (fail loud/closed rather than silently degrade), with no separate exit
    code distinguishing a check failure from an infra failure. A
    structured stdout summary is printed for the check-failure case — one
    line per offending task (task_id, cited commit sha, flag class) — for
    human/log triage; the DeterministicRunner itself never parses it, only
    the exit code.
  - Exit 2: ``--since`` could not be parsed — a caller usage error, kept
    on its own code so it is never mistaken for "offenders found" (1) by
    a caller branching on exit code alone.

Two orthogonal signals, deliberately kept separate (task 3576):

  - ``stamped_at`` supplies FRESHNESS scoping (``stamp_class``), and is
    what gates the exit code.
  - Commit FAN-OUT supplies the sub-pattern A/B split (``sub_pattern``),
    and is reporting only.

They answer different questions and neither substitutes for the other. A
timestamp cannot distinguish legitimate named-sibling attribution from
spurious bare-merge-commit sharing — those differ in citation SHAPE, not
in time; two stamps written in the same second can be one legitimate and
one spurious. Conversely fan-out says nothing about when a stamp was
written. See :func:`classify_sub_patterns` for the fan-out rule.

Deliberately narrower than the audit's own ``_FLAGGED_VERDICTS``: this
predicate's contract (PRD decomposition label ι) is specifically
misattribution/deliverable-absence, the two verdict classes this PRD's
fixes (Face A) target. ``commit_not_on_main``/``reverted`` are a Face-B/
retrospective concern the audit still reports but this predicate does not
gate on.

Read-only: this script never calls ``update_task``/``set_task_status`` —
it only reads tasks and gathers git facts (both already read-only in the
wrapped audit module).

Usage
-----
  python scripts/check_found_on_main_spurious_rate.py \\
      --project-root /path/to/project --since 2026-07-16T00:00:00Z

  # Optional: audit a non-default ref, or point at a specific config.
  python scripts/check_found_on_main_spurious_rate.py \\
      --project-root /path/to/project --since 2026-07-16T00:00:00+00:00 \\
      --ref origin/main --config /path/to/fused-memory-config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from shared.task_metadata import parse_metadata

logger = logging.getLogger('check_found_on_main_spurious_rate')

# Verdicts this predicate gates on — a strict subset of the audit module's
# own _FLAGGED_VERDICTS (which also includes commit_not_on_main/reverted;
# see module docstring for why those are out of scope here).
_SPURIOUS_VERDICTS = frozenset({'misattributed', 'deliverable_absent'})


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def parse_since(value: str) -> datetime:
    """Parse ``--since`` into an aware UTC ``datetime``.

    Accepts any ``datetime.fromisoformat``-parseable string (``Z`` suffix
    normalised to ``+00:00`` first, since ``fromisoformat`` on Python <3.11
    rejects a bare ``Z``). A naive value (no offset) is treated as already
    UTC — the documented contract is "ISO-8601 UTC" — rather than silently
    guessing a different zone; an aware value is converted to UTC. Raises
    ``ValueError`` on an unparseable string; ``main()`` catches this and
    maps it to exit code 2 — a usage error, distinct from the 0/1
    business-logic exit codes (see module docstring "Contract").
    """
    normalized = value.strip()
    if normalized.endswith('Z'):
        normalized = normalized[:-1] + '+00:00'
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _parse_task_timestamp(value: Any) -> datetime | None:
    """Parse a task's ``updatedAt`` value to an aware UTC ``datetime``, or
    ``None`` if it is missing, non-string, or unparseable — never raises.

    ``value`` comes from an untyped raw task dict (``dict[str, Any]``), so
    it is not statically known to be a ``str`` — guard with ``isinstance``
    rather than assuming the shape, consistent with the "never raises"
    contract.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return parse_since(value)
    except ValueError:
        return None


def _id_sort_key(task_id: str) -> int:
    """Numeric sort key for a task id; non-numeric ids fall back to 0
    (mirrors ``audit_found_on_main_provenance._id_as_int``)."""
    return int(task_id) if str(task_id).isdecimal() else 0


def _stamped_at_of(task: dict[str, Any]) -> str | None:
    """Read ``metadata.done_provenance.stamped_at`` off a raw task dict, or
    ``None`` when absent/unreadable — never raises.

    ``parse_metadata`` accepts metadata as either a dict or a JSON string
    (both shapes come back from ``get_tasks()``) and is itself warn-and-omit
    rather than raising on malformed input, so a corrupt blob degrades to
    ``None`` here and the caller falls back to ``updatedAt``.
    """
    try:
        meta, _warnings = parse_metadata(task.get('metadata'), direction='read')
    except Exception:  # noqa: BLE001 - never-raises contract; any parse
        return None      # failure just means "no usable stamp".
    provenance = meta.done_provenance
    if provenance is None:
        return None
    return getattr(provenance, 'stamped_at', None)


def find_spurious_since(
    report: dict[str, Any], tasks: list[dict[str, Any]], since: datetime,
) -> list[dict[str, Any]]:
    """Cross-reference *report*'s per-task verdicts with *tasks*' ``updatedAt``
    stamps, returning offending records updated strictly after *since* and
    flagged ``misattributed``/``deliverable_absent``.

    *report* is ``audit_found_on_main_provenance.build_audit_report``'s
    return value; *tasks* is the raw task list already fetched for that
    call (same ``get_tasks()`` shape), reused here rather than refetched —
    ``build_audit_report``'s ``tasks`` detail entries don't carry
    ``updatedAt`` through. A task whose freshness timestamp is missing or
    unparseable is conservatively excluded (timing can't be confirmed, so it
    is never silently counted as "after") rather than raising.

    Freshness (task 3576): the timestamp compared against *since* is
    ``metadata.done_provenance.stamped_at`` — the dedicated stamp-*write*
    instant recorded server-side by fused-memory's
    ``_validate_done_provenance`` chokepoint — falling back to ``updatedAt``
    only when no usable ``stamped_at`` is present. That fallback is what
    ``updatedAt`` alone used to do for every task, and it over-approximates:
    ANY write to a task bumps ``updatedAt`` (a re-tag, an unrelated metadata
    annotation, a dependency edit, the audit's own ``--apply`` annotation),
    so an old stamp touched later for an unrelated reason re-entered the
    window on every run. ``stamped_at`` answers the question this predicate
    actually asks — "was this attribution asserted after *since*?" — exactly
    rather than by approximation.

    Each record is annotated with ``stamp_class``: ``'fresh'`` when its
    freshness came from ``stamped_at``, ``'legacy'`` when it fell back to
    ``updatedAt``. See :func:`gating_offenders` for what that gates.

    Returns records sorted by ``int(task_id)`` for deterministic output.
    """
    by_id = {str(t.get('id', '')): t for t in tasks}
    # Fan-out is computed over the FULL report, before the flagged-subset
    # filter below — see classify_sub_patterns for why that matters.
    sub_pattern_by_commit = classify_sub_patterns(report)
    offenders: list[dict[str, Any]] = []
    for detail in report.get('tasks', []):
        verdict = detail.get('verdict')
        if verdict not in _SPURIOUS_VERDICTS:
            continue
        task_id = detail['task_id']
        task = by_id.get(task_id)
        if task is None:
            continue

        updated_at = task.get('updatedAt')
        stamped_at = _stamped_at_of(task)
        # Prefer the dedicated stamp-write instant; fall back to updatedAt
        # when it is absent (a pre-3576 blob) or unparseable (corrupt).
        freshness_dt = _parse_task_timestamp(stamped_at)
        stamp_class = 'fresh'
        if freshness_dt is None:
            freshness_dt = _parse_task_timestamp(updated_at)
            stamp_class = 'legacy'
        if freshness_dt is None or freshness_dt <= since:
            continue

        commit = detail.get('commit')
        offenders.append({
            'task_id': task_id,
            'commit': commit,
            'verdict': verdict,
            'updated_at': updated_at,
            'stamped_at': stamped_at,
            'stamp_class': stamp_class,
            'sub_pattern': sub_pattern_by_commit.get(commit) if commit else None,
        })
    offenders.sort(key=lambda o: _id_sort_key(o['task_id']))
    return offenders


def classify_sub_patterns(report: dict[str, Any]) -> dict[str, str]:
    """Map each cited commit to its sub-pattern, by fan-out over *report*.

    Fan-out = how many DISTINCT found_on_main task ids cite a given commit:

      - ``>= 2`` -> ``'shared_bare_merge'``: several unrelated tasks all
        pointing at one bare "Merge task/X into main" commit. This is the
        spurious sharing sub-pattern (8 of recon's 13 cases; corroborated
        by esc-3398-1 for task 3301).
      - ``== 1`` -> ``'named_sibling'``: one task citing one sibling's
        commit — the legitimate attribution sub-pattern (the other 5).

    Counted over the FULL ``report['tasks']`` list — every audited
    found_on_main task, whatever its verdict — not just the flagged subset.
    A spurious task sharing a commit with an ``ok`` task is still
    ``shared_bare_merge``; the sharing is the signal, and counting only
    flagged entries would misclassify it as ``named_sibling``.

    Details with no ``commit`` are skipped rather than grouped under a
    ``None`` key. Pure function; never raises.
    """
    tasks_by_commit: dict[str, set[str]] = {}
    for detail in report.get('tasks', []):
        commit = detail.get('commit')
        if not commit:
            continue
        tasks_by_commit.setdefault(commit, set()).add(str(detail.get('task_id', '')))
    return {
        commit: ('shared_bare_merge' if len(task_ids) >= 2 else 'named_sibling')
        for commit, task_ids in tasks_by_commit.items()
    }


def gating_offenders(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter *records* to the ones that gate the exit code — the fresh ones.

    A record is ``stamp_class='legacy'`` when its freshness had to fall back
    to ``updatedAt`` because the blob carries no usable
    ``done_provenance.stamped_at``. Absence of that field is sound proof the
    stamp predates task 3576: fused-memory's ``_validate_done_provenance``
    is the single server-side write site for ``kind='found_on_main'``
    provenance, and every write through it since 3576 landed populates the
    field — so a blob lacking it cannot have been written since. (That
    chokepoint is not bypassable by callers: ``update_task`` refuses to let
    ``metadata.done_provenance`` be written around it — task 2201/C1.)

    Legacy records are excluded from the GATE but never from the REPORT:
    :func:`find_spurious_since` still returns them and ``_run`` still prints
    them with their ``stamp_class``. That is deliberate. The residual risk
    this design carries is a future found_on_main write path that bypasses
    ``_validate_done_provenance`` — its stamps would carry no ``stamped_at``,
    be classed legacy, and not gate. Printing them loudly rather than
    dropping them is the mitigation: an operator still sees the offender in
    the log, so the failure is visible rather than silent.

    Pure filter — never mutates *records*, and preserves their order (which
    :func:`find_spurious_since` already sorted by numeric task id).
    """
    return [r for r in records if r.get('stamp_class') == 'fresh']


def format_summary(offenders: list[dict[str, Any]]) -> list[str]:
    """One structured stdout line per offending task: task_id, cited sha,
    flag class, plus the two task-3576 annotations — ``stamp_class``
    (fresh|legacy, which decides whether the record gates) and
    ``sub_pattern`` (shared_bare_merge|named_sibling, which is reporting
    only). Human/log-triage only — the DeterministicRunner predicate
    contract is exit-code-only and parses no output (see module docstring).

    Every offender is rendered, legacy ones included: a legacy record stops
    gating but must never stop being visible. Annotations absent from a
    record render as ``None`` rather than raising — this is triage output,
    never a source of exceptions.
    """
    return [
        f'task_id={o["task_id"]} commit={o["commit"]} flag_class={o["verdict"]} '
        f'stamp_class={o.get("stamp_class")} sub_pattern={o.get("sub_pattern")}'
        for o in offenders
    ]


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

async def _run(args: argparse.Namespace, since: datetime) -> int:
    # `since` arrives already parsed by main() — this function never calls
    # parse_since itself. That keeps main()'s ValueError/exit-2 usage-error
    # mapping scoped tightly around just the parse_since(args.since) call:
    # a ValueError raised anywhere in here (build_audit_report, git parsing,
    # the backend, ...) is a genuine internal failure and must propagate
    # uncaught, never be mislabeled as "invalid --since".
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )

    import os  # noqa: PLC0415

    # Sibling import: audit_found_on_main_provenance.py lives alongside this
    # script in fused-memory/scripts/, which is sys.path[0] when this file
    # is invoked directly. Deferred to runtime (never a module-level import)
    # so the test suite can importlib-load this module in isolation without
    # the scripts/ directory on sys.path — mirrors
    # correct_found_on_main_backlog.py's identical deferred-import pattern.
    from audit_found_on_main_provenance import GitFacts, build_audit_report  # noqa: PLC0415

    from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend  # noqa: PLC0415
    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    config = FusedMemoryConfig()
    if config.taskmaster is None:
        # Fail loud/closed: an unconfigured task backend is an infra
        # problem, not a clean result, and this predicate's exit-code
        # contract has no separate channel for infra vs. check-failure
        # (see module docstring "Contract") — non-zero means "did not
        # pass" either way, by design.
        logger.error('Task backend not configured in fused-memory config')
        return 1

    backend = SqliteTaskBackend(config.taskmaster)
    await backend.start()
    try:
        raw = await backend.get_tasks(args.project_root)
        tasks = raw.get('tasks') or []
        logger.info('Fetched %d task(s) from task backend', len(tasks))

        # Read-only: gathers git facts + reads tasks; never mutates state.
        git = GitFacts(args.project_root)
        report = await build_audit_report(tasks, git, ref=args.ref)

        offenders = find_spurious_since(report, tasks, since)
        summary_lines = format_summary(offenders)

        logger.info(
            'found_on_main spurious-rate check: %d found_on_main task(s) audited, '
            '%d flagged misattributed/deliverable_absent since %s',
            report.get('total', 0), len(offenders), since.isoformat(),
        )
        for line in summary_lines:
            print(line)

        return 1 if offenders else 0
    finally:
        await backend.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--since', required=True,
        help=(
            'ISO-8601 UTC timestamp; only found_on_main stamps updated after '
            'this instant are considered (a naive value is treated as UTC).'
        ),
    )
    parser.add_argument(
        '--project-root', required=True,
        help='Absolute filesystem path to the project (required by Taskmaster + git)',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to fused-memory config file (sets CONFIG_PATH env var)',
    )
    parser.add_argument(
        '--ref', default='main',
        help='Git ref to audit commit lineage against (default: main)',
    )
    args = parser.parse_args()

    # Parse --since here, in main(), BEFORE any backend/report work, and
    # scope the ValueError catch tightly around just this call — not around
    # the whole run. A malformed --since is a caller usage error, kept on
    # its own exit code (2) rather than propagating as an uncaught
    # traceback exiting 1, which would be indistinguishable from a genuine
    # "offenders found" failure under this predicate's exit-code-only
    # contract (see module docstring "Contract"). Deliberately NOT wrapping
    # asyncio.run(_run(...)) in this same try/except: a ValueError raised
    # later — e.g. from build_audit_report, git parsing, or the backend —
    # is a genuine internal failure and must propagate as such, never get
    # mislabeled as "invalid --since".
    try:
        since = parse_since(args.since)
    except ValueError as exc:
        print(f'error: invalid --since {args.since!r}: {exc}', file=sys.stderr)
        return 2

    return asyncio.run(_run(args, since))


if __name__ == '__main__':
    sys.exit(main())
