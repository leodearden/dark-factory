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
    ``updatedAt`` through. A task with a missing or unparseable
    ``updatedAt`` is conservatively excluded (timing can't be confirmed,
    so it is never silently counted as "after") rather than raising.

    Freshness caveat: ``updatedAt`` over-approximates "the found_on_main
    stamp was (re)written after *since*" — any write to the task bumps it,
    so a task touched for an unrelated reason after *since* can still be
    included even if its stamp itself predates *since* (see module
    docstring "Freshness caveat"; ``done_provenance`` carries no dedicated
    stamp-write timestamp to key off of instead).

    Returns records sorted by ``int(task_id)`` for deterministic output.
    """
    updated_at_by_id = {str(t.get('id', '')): t.get('updatedAt') for t in tasks}
    offenders: list[dict[str, Any]] = []
    for detail in report.get('tasks', []):
        verdict = detail.get('verdict')
        if verdict not in _SPURIOUS_VERDICTS:
            continue
        task_id = detail['task_id']
        updated_dt = _parse_task_timestamp(updated_at_by_id.get(task_id))
        if updated_dt is None or updated_dt <= since:
            continue
        offenders.append({
            'task_id': task_id,
            'commit': detail.get('commit'),
            'verdict': verdict,
            'updated_at': updated_at_by_id.get(task_id),
        })
    offenders.sort(key=lambda o: _id_sort_key(o['task_id']))
    return offenders


def format_summary(offenders: list[dict[str, Any]]) -> list[str]:
    """One structured stdout line per offending task: task_id, cited sha,
    flag class. Human/log-triage only — the DeterministicRunner predicate
    contract is exit-code-only and parses no output (see module docstring)."""
    return [
        f'task_id={o["task_id"]} commit={o["commit"]} flag_class={o["verdict"]}'
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
