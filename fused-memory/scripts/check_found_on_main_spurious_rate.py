#!/usr/bin/env python3
"""Predicate wrapper: found_on_main spurious-rate check (task 2676 / PRD label ι).

Thin, read-only wrapper around ``audit_found_on_main_provenance.py``
(exercised live by task 2648's audit run) for use as a ``before_done``
``kind='predicate'`` script under the DeterministicRunner convention (see
CLAUDE.md "Predicate exit-code contract"). It runs the existing audit, then
considers only the ``found_on_main`` done-provenance stamps WRITTEN
strictly after ``--since``, and asks whether any of *those* were flagged
``misattributed`` or ``deliverable_absent`` by the audit's classifier.
Older stamps (already covered by a prior run, or predating the fix this
predicate soaks) are reported but do not gate, so they are not re-escalated
on every invocation.

Freshness (task 3576): the timestamp compared against ``--since`` is
``metadata.done_provenance.stamped_at`` — the dedicated stamp-*write*
instant recorded server-side by fused-memory's
``_validate_done_provenance`` chokepoint — not ``updatedAt``. ``updatedAt``
remains only as a fallback for blobs that carry no usable ``stamped_at``,
and those records are marked ``stamp_class='legacy'``.

That distinction is the point. ``updatedAt`` is bumped by ANY write to a
task (a re-tag, an unrelated metadata annotation, a dependency edit, the
sibling audit's own ``--apply`` annotation), so keying off it meant a
stamp written months ago looked "fresh" forever and the same known
offenders re-entered the window on every single run — guaranteeing a
permanent EXIT 1 that said nothing about whether anything new had gone
wrong. ``stamped_at`` answers the question this predicate actually asks:
"was this attribution asserted after ``--since``?"

A blob with no ``stamped_at`` provably predates task 3576, because every
write through the (non-bypassable) server chokepoint since then populates
it. Such records are LEGACY BACKLOG by construction: they are still
reported — printed with ``stamp_class=legacy`` — but they do not gate.

That "absence proves it predates 3576" argument covers an ABSENT field
only. A field that is PRESENT but unparseable proves the opposite — it
came from a post-3576 write — so it is classed ``stamp_class=corrupt``,
logged at WARNING, and GATES. Silently demoting a corrupt stamp to
non-gating would be exactly the fail-open this predicate exists to
prevent. Its freshness still falls back to ``updatedAt`` (the never-raises
contract), so a corrupt stamp on an untouched-since-``--since`` task is
still excluded on timing.

Two orthogonal signals, deliberately kept separate (task 3576):

  - ``stamped_at`` supplies FRESHNESS scoping (``stamp_class``), and is
    what gates the exit code.
  - Commit FAN-OUT supplies the sub-pattern A/B split (``sub_pattern``),
    and is reporting only.

They answer different questions and neither substitutes for the other. A
timestamp cannot distinguish one-task-one-commit attribution from
spurious bare-merge-commit sharing — those differ in citation SHAPE, not
in time; two stamps written in the same second can be one legitimate and
one spurious. Conversely fan-out says nothing about when a stamp was
written. See :func:`classify_sub_patterns` for the fan-out rule.

Contract (the DeterministicRunner branches on the exit code alone, per
the predicate convention; stdout is human/log triage plus one machine
-readable trailing line — see "Trailing JSON summary" below):

  - Exit 0: no NEW spurious found_on_main stamp was written since
    ``--since`` — i.e. zero flagged records with a GATING
    ``stamp_class`` (``fresh`` or ``corrupt``). Legacy (pre-3576)
    offenders may still be present and printed; exit 0 means "the
    freshness window no longer re-admits legacy rows", NOT "the backlog
    was corrected".
  - Exit 1: at least one flagged record gates — genuinely fresh (its
    ``stamped_at`` is after ``--since``) or corrupt (a present-but-
    unparseable ``stamped_at``, which proves a post-3576 write) — OR the
    task backend is not configured (an infra/config problem). This
    predicate's contract is intentionally coarse:
    non-zero-for-any-reason means "did not pass" (fail loud/closed rather
    than silently degrade), with no separate exit code distinguishing a
    check failure from an infra failure. A structured stdout summary is
    printed for EVERY flagged record — gating or not — one line per
    offending task (task_id, cited commit sha, flag class,
    ``stamp_class``, ``sub_pattern``) for human/log triage, followed by a
    single trailing JSON counts object.
  - Exit 2: ``--since`` could not be parsed — a caller usage error, kept
    on its own code so it is never mistaken for "offenders found" (1) by
    a caller branching on exit code alone.

Trailing JSON summary
---------------------

The LAST stdout line is always a compact JSON counts object
(``{"gating": N, "legacy": N, "total_flagged": N, "stamp_classes": {...},
"sub_patterns": {...}}``). It exists for the orchestrator's provenance
note, not for this script's own contract.

``DeterministicRunner._summarize_predicate_output`` folds a passing
predicate's output into ``done_provenance.note``, which fused-memory's
reconciliation ``_format_outcome_echo`` then INGESTS INTO MEMORY. Its
extractor runs two tiers: tier 1 parses a trailing JSON block and is a
true structural allowlist; tier 2 keeps the last non-log-shaped line as a
best-effort heuristic. Before task 3576 this script only ever printed
offender lines on the exit-1 path, so a passing run's last line was a
timestamped log line that tier 2's denylist rejected. Now a PASSING run
routinely prints legacy offender lines — which carry neither timestamp
nor level token, so tier 2 would keep whichever offender happened to sort
last, stamping the self-contradictory note ``predicate check passed
(rc=0): task_id=705 ... stamp_class=legacy``. Emitting the JSON object
last makes tier 1 win, so the note carries the aggregate verdict instead
of an arbitrary offender row.

The object is deliberately COUNTS ONLY — no task ids, no shas. That keeps
it bounded by the fixed class vocabulary (3 integers plus two small
mappings, ~200 chars at worst) rather than by the offender count, so it
can never trip the extractor's 400-char cap, which elides an oversized
payload WHOLESALE and would silently degrade the note back to the bare
verdict on precisely the runs with the most to report.

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
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# Deliberately exempt from this file's deferred-import convention (see _run,
# which defers `os`, the sibling audit module, and the fused_memory backend/
# config imports). `dark-factory-shared` is a hard install dependency of the
# fused-memory package — declared in fused-memory/pyproject.toml, not an
# optional or path-sensitive one — so importing it at module level adds no
# environment requirement that running this script did not already have. The
# sibling audit script (audit_found_on_main_provenance.py) imports it at
# module level from this same directory for the same reason. The deferred
# imports below exist for a DIFFERENT reason: they are either sys.path[0]-
# sensitive (the sibling script) or heavyweight, and the test suite
# importlib-loads this module in isolation.
from shared.task_metadata import parse_metadata

logger = logging.getLogger('check_found_on_main_spurious_rate')

# Verdicts this predicate gates on — a strict subset of the audit module's
# own _FLAGGED_VERDICTS (which also includes commit_not_on_main/reverted;
# see module docstring for why those are out of scope here).
_SPURIOUS_VERDICTS = frozenset({'misattributed', 'deliverable_absent'})

# stamp_class values that GATE the exit code. `fresh` is a stamp written
# after --since. `corrupt` is a present-but-unparseable stamp: unlike an
# ABSENT one it proves a post-3576 write, so the "predates the field"
# argument that excuses `legacy` does not cover it — fail closed. Only
# `legacy` (absent stamp, freshness fell back to updatedAt) is non-gating.
_GATING_STAMP_CLASSES = frozenset({'fresh', 'corrupt'})


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

    Each record is annotated with ``stamp_class``:

      - ``'fresh'`` — freshness came from a parseable ``stamped_at``.
      - ``'corrupt'`` — a ``stamped_at`` is PRESENT but unparseable, so
        freshness fell back to ``updatedAt``. Logged at WARNING and still
        gating: a present stamp proves a post-3576 write, so the "absence
        proves it predates 3576" argument that excuses ``'legacy'`` does
        not apply.
      - ``'legacy'`` — no ``stamped_at`` at all; freshness fell back to
        ``updatedAt``. Non-gating.

    See :func:`gating_offenders` for what that gates.

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
            # ABSENT vs. PRESENT-BUT-UNPARSEABLE are NOT the same signal and
            # must not collapse to one class. Only absence supports the
            # "predates task 3576" argument that makes a record non-gating;
            # a corrupt stamp is a post-3576 write whose freshness merely
            # cannot be read, so it gates and is logged rather than silently
            # demoted. Warn here — before the since-filter — so a corrupt
            # stamp is never swallowed by the timing exclusion below either.
            if stamped_at is None:
                stamp_class = 'legacy'
            else:
                stamp_class = 'corrupt'
                logger.warning(
                    'task %s: done_provenance.stamped_at is present but '
                    'unparseable (%r); falling back to updatedAt for '
                    'freshness and classing it stamp_class=corrupt (gating)',
                    task_id, stamped_at,
                )
            freshness_dt = _parse_task_timestamp(updated_at)
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
        pointing at one commit — in recon's sample, a bare "Merge task/X
        into main". Sharing IS positive evidence of the spurious
        sub-pattern (8 of recon's 13 cases; corroborated by esc-3398-1 for
        task 3301), since a commit cannot legitimately be the deliverable
        of several unrelated tasks.
      - ``== 1`` -> ``'sole_citer'``: no other audited found_on_main task
        cites this commit. Named for exactly what is measured and NOTHING
        more. Recon's other 5 cases were legitimate one-task-one-sibling
        attributions, but fan-out of 1 does not establish that: a single
        task citing a bare merge commit that no other task happens to cite
        also lands here. The label must not read as a verdict — an
        operator sees it on the stdout triage line, and the earlier name
        ``named_sibling`` asserted a legitimacy this measurement cannot
        support.

    Reporting only — ``sub_pattern`` never gates the exit code (see
    :func:`gating_offenders`), so a fan-out-1 record being merely
    "not shared" rather than "known good" cannot mis-gate; it can only
    mislead triage, which is why the name was narrowed to the measurement.

    Counted over the FULL ``report['tasks']`` list — every audited
    found_on_main task, whatever its verdict — not just the flagged subset.
    A spurious task sharing a commit with an ``ok`` task is still
    ``shared_bare_merge``; the sharing is the signal, and counting only
    flagged entries would misclassify it as ``sole_citer``.

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
        commit: ('shared_bare_merge' if len(task_ids) >= 2 else 'sole_citer')
        for commit, task_ids in tasks_by_commit.items()
    }


def gating_offenders(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter *records* to the ones that gate the exit code.

    Gating classes are ``'fresh'`` and ``'corrupt'``; only ``'legacy'`` is
    excluded.

    A record is ``stamp_class='legacy'`` when its freshness had to fall back
    to ``updatedAt`` because the blob carries NO
    ``done_provenance.stamped_at`` at all. Absence of that field is sound
    proof the stamp predates task 3576: fused-memory's
    ``_validate_done_provenance`` is the single server-side write site for
    ``kind='found_on_main'`` provenance, and every write through it since
    3576 landed populates the field — so a blob lacking it cannot have been
    written since. (That chokepoint is not bypassable by callers:
    ``update_task`` refuses to let ``metadata.done_provenance`` be written
    around it — task 2201/C1.)

    That argument turns entirely on ABSENCE, so it does NOT extend to a
    ``stamped_at`` that is present but unparseable — such a blob provably
    came from a post-3576 write, and only its freshness is unreadable.
    Those are classed ``'corrupt'`` and GATE (and are logged at WARNING
    where detected, in :func:`find_spurious_since`). Folding them into
    ``'legacy'`` would be a silent fail-open in exactly the direction this
    predicate exists to catch: a corrupt stamp would stop gating without
    anyone being told.

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
    return [r for r in records if r.get('stamp_class') in _GATING_STAMP_CLASSES]


def format_summary(offenders: list[dict[str, Any]]) -> list[str]:
    """One structured stdout line per offending task: task_id, cited sha,
    flag class, plus the two task-3576 annotations — ``stamp_class``
    (fresh|corrupt|legacy, which decides whether the record gates) and
    ``sub_pattern`` (shared_bare_merge|sole_citer, which is reporting
    only). Human/log-triage only — the DeterministicRunner branches on the
    exit code, never on these lines (see module docstring). The aggregate
    counts the orchestrator's provenance note DOES pick up are emitted
    separately, as the trailing JSON line ``_run`` prints after these.

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
        gating = gating_offenders(offenders)
        summary_lines = format_summary(offenders)

        sub_pattern_counts: dict[str, int] = {}
        stamp_class_counts: dict[str, int] = {}
        for o in offenders:
            sub_key = str(o.get('sub_pattern'))
            sub_pattern_counts[sub_key] = sub_pattern_counts.get(sub_key, 0) + 1
            stamp_key = str(o.get('stamp_class'))
            stamp_class_counts[stamp_key] = stamp_class_counts.get(stamp_key, 0) + 1

        # Report the non-gating count as its own class rather than deriving
        # it as (total - gating): with `corrupt` now gating alongside
        # `fresh`, that subtraction would silently relabel corrupt records
        # as legacy in the log.
        legacy_count = stamp_class_counts.get('legacy', 0)
        logger.info(
            'found_on_main spurious-rate check: %d found_on_main task(s) audited, '
            '%d flagged misattributed/deliverable_absent since %s '
            '(%d gating [fresh|corrupt], %d legacy [pre-3576, reported only]); '
            'stamp classes: %s; sub-patterns: %s',
            report.get('total', 0), len(offenders), since.isoformat(),
            len(gating), legacy_count,
            ', '.join(f'{k}={v}' for k, v in sorted(stamp_class_counts.items())) or 'none',
            ', '.join(f'{k}={v}' for k, v in sorted(sub_pattern_counts.items())) or 'none',
        )
        # Print ALL offenders, not just the gating ones: a legacy record
        # stops gating but must never stop being visible (see
        # gating_offenders for the residual-risk argument).
        for line in summary_lines:
            print(line)

        # The trailing JSON counts object MUST be the final stdout line —
        # see the module docstring's "Trailing JSON summary" section. It is
        # what DeterministicRunner._summarize_predicate_output's tier-1
        # (structural, allowlist) extractor picks up for
        # done_provenance.note; without it, tier 2's heuristic would keep
        # whichever offender line sorted last and stamp a note reading
        # "predicate check passed (rc=0): task_id=... stamp_class=legacy".
        # Printed unconditionally, including when there are no offenders at
        # all, so the note shape does not depend on the result.
        print(json.dumps({
            'gating': len(gating),
            'legacy': legacy_count,
            'total_flagged': len(offenders),
            'stamp_classes': stamp_class_counts,
            'sub_patterns': sub_pattern_counts,
        }, sort_keys=True))

        return 1 if gating else 0
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
