"""Shared plumbing for the READ-ONLY tasks.db sweep scripts.

Home of the tasks.db discovery ladder and the leak-scanner CLI skeleton that
were previously copied into ``scan_task_toolcall_leaks.py``,
``scan_provenance_note_log_leaks.py`` and ``audit_wiped_metadata_files.py``
(task 3336, closing task 3286's "~134 identical lines" finding). Every
function body here was lifted from those copies verbatim, so "pure extraction,
no behaviour change" stays checkable by diff.

This module is NOT a CLI in its own right — the leading underscore marks it as
importable-by-sibling-scripts only. It hosts two tiers:

* **Tier 1, discovery** (``_DEFAULT_PROJECT_ROOTS``, :func:`tasks_db_path`,
  :func:`resolve_project_roots`, :func:`discover_project_roots`,
  :func:`discover_db_paths`) — adopted by ALL THREE sweep scripts.
* **Tier 2, leak-scanner CLI plumbing** — adopted by the two leak scanners
  only. ``audit_wiped_metadata_files.py`` deliberately keeps its own
  ``format_report``/``format_json``/``_build_parser``/``main``: its
  ``--min-fidelity`` filter, its object-shaped rather than array-shaped JSON
  and its different no-roots message are genuinely different behaviour, not
  duplication. Folding them in here would silently delete those semantics.
  Its exit 3 is NOT among those differences any more — task 3474 gave
  :func:`run_scan_cli` the same "nothing was scanned/audited, so this is not
  a clean run" semantics, so ``docs/legibility/design-invariants.md``'s
  no-silent-fail-soft rule is now honoured by BOTH tiers rather than only by
  the script that opted out of the shared plumbing.

IMPORT-RESOLUTION CONTRACT — read before moving this file.
This module MUST stay a flat sibling at ``scripts/_task_db_scan.py``. The
sweep scripts' CLI tests drive ``main()`` by shelling out
(``subprocess.run([python, <script path>, ...])``), and
``scripts/tests/conftest.py``'s sys.path insertion does not reach those child
processes. They resolve ``import _task_db_scan`` solely because a
DIRECTLY-EXECUTED script places its own directory at ``sys.path[0]``.
Therefore: never relocate this module into a package, and never invoke any of
the three sweep scripts via ``python -m`` — either change breaks every CLI
test at once. (In-process pytest resolution is separately handled by
``scripts/tests/conftest.py``, which already puts ``scripts/`` on sys.path.)
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

# ---------------------------------------------------------------------------
# Tier 1 — tasks.db discovery (adopted by all three sweep scripts).
# ---------------------------------------------------------------------------

# Multi-project discovery fallback, mirroring
# scripts/migrate_metadata_modules_to_files.py's DEFAULT_ROOTS.
_DEFAULT_PROJECT_ROOTS = ("/home/leo/src/dark-factory",)


def tasks_db_path(project_root: str) -> Path:
    """``<root>/.taskmaster/tasks/tasks.db`` — the live task store."""
    return Path(project_root) / ".taskmaster" / "tasks" / "tasks.db"


def resolve_project_roots(
    project_roots: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Resolve the list of project roots, WITHOUT filtering for existence.

    Precedence (first supplied wins): *project_roots* >
    ``DASHBOARD_KNOWN_PROJECT_ROOTS`` (comma-separated, read from *env*,
    defaulting to the real ``os.environ`` when *env* is None) > the
    dark-factory default root. Env entries are stripped, empties are dropped,
    and order is preserved.

    Existence filtering is deliberately left to the callers, because they
    filter on different things: :func:`discover_db_paths` drops a missing
    ``tasks.db``, while :func:`discover_project_roots` drops the ROOT owning
    one.
    """
    if project_roots is not None:
        return list(project_roots)

    environ = env if env is not None else os.environ
    roots_env = (environ.get("DASHBOARD_KNOWN_PROJECT_ROOTS") or "").strip()
    if roots_env:
        return [r.strip() for r in roots_env.split(",") if r.strip()]
    return list(_DEFAULT_PROJECT_ROOTS)


def discover_project_roots(
    project_roots: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Resolve the list of project roots to audit.

    Precedence (first supplied wins): *project_roots* >
    ``DASHBOARD_KNOWN_PROJECT_ROOTS`` (comma-separated, read from *env*,
    defaulting to the real ``os.environ``) > the dark-factory default root.

    A root whose ``tasks.db`` does not exist is silently dropped — this never
    raises on a missing or not-yet-set-up project.
    """
    return [
        root
        for root in resolve_project_roots(project_roots=project_roots, env=env)
        if tasks_db_path(root).exists()
    ]


def discover_db_paths(
    explicit_dbs: list[str] | None = None,
    project_roots: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Resolve the list of tasks.db paths to scan.

    Precedence (first supplied wins): *explicit_dbs* > *project_roots* >
    ``DASHBOARD_KNOWN_PROJECT_ROOTS`` (read from *env*, defaulting to the
    real ``os.environ`` when *env* is None) > the dark-factory default root.

    A resolved db path that does not exist on disk is silently skipped —
    this never raises on a missing/not-yet-set-up project.
    """
    if explicit_dbs is not None:
        candidates = list(explicit_dbs)
    else:
        candidates = [
            str(tasks_db_path(root))
            for root in resolve_project_roots(project_roots=project_roots, env=env)
        ]

    return [path for path in candidates if os.path.exists(path)]


# ---------------------------------------------------------------------------
# Tier 2 — leak-scanner CLI plumbing (the two leak scanners only).
#
# audit_wiped_metadata_files.py deliberately does NOT adopt this tier: its
# main() carries a --min-fidelity filter, object-shaped rather than
# array-shaped JSON, and a different no-roots message.
#
# Its exit 3 (roots resolved but every one failed to audit, kept distinct
# from 0 per docs/legibility/design-invariants.md's no-silent-fail-soft rule)
# used to be listed here too. It is no longer a differentiator: run_scan_cli
# returns 3 for the same condition as of task 3474, which is what makes this
# shared plumbing consistent with the invariant rather than an exception to
# it.
# ---------------------------------------------------------------------------

# The exit-2 signal, emitted verbatim by both leak scanners.
NO_DB_RESOLVED_MESSAGE = (
    "no tasks.db resolvable (checked --db / --project-root / "
    "DASHBOARD_KNOWN_PROJECT_ROOTS / the dark-factory default)"
)


def format_json(matches: Sequence[NamedTuple]) -> str:
    """Render *matches* as a JSON array, carrying the FULL untruncated field."""
    return json.dumps([m._asdict() for m in matches])


def truncate(text: str, max_len: int) -> str:
    """Cap *text* at *max_len* characters, marking elision with ``...``.

    A string at or below the limit is returned unchanged (no ellipsis), so a
    report line is only ever marked as truncated when it actually was.
    """
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def group_matches_by_db(matches: Sequence[Any]) -> dict[str, list[Any]]:
    """Group *matches* by their ``db_path`` field for a per-database report.

    Keys iterate in sorted order (a report's db sections are stable regardless
    of scan order); within a key, matches keep their original scan order.
    """
    by_db: dict[str, list[Any]] = {}
    for m in matches:
        by_db.setdefault(m.db_path, []).append(m)
    return {db_path: by_db[db_path] for db_path in sorted(by_db)}


def add_db_discovery_args(parser: argparse.ArgumentParser, *, json_help: str) -> None:
    """Add the shared ``--db`` / ``--project-root`` / ``--json`` arguments.

    ``--json``'s help is a parameter because the two scanners name different
    payloads ("fragments" vs "leak lines"); the other two are byte-identical
    between them.
    """
    parser.add_argument(
        "--db", dest="dbs", action="append",
        help="Explicit tasks.db path to scan. May be repeated.",
    )
    parser.add_argument(
        "--project-root", dest="project_roots", action="append",
        help=(
            "Project root to scan (maps to <root>/.taskmaster/tasks/tasks.db). "
            "May be repeated."
        ),
    )
    parser.add_argument("--json", action="store_true", help=json_help)


def sweep_databases(
    db_paths: Sequence[str],
    scan_fn: Callable[[str], list[Any]],
) -> tuple[list[Any], list[str]]:
    """Run *scan_fn* over every path in *db_paths*, returning (matches, unreadable).

    A single unreadable database (e.g. a stale/corrupt file, or a transient
    "database is locked"/"file is not a database" condition) does not abort the
    sweep: it is logged to stderr and skipped so every other resolvable
    database is still scanned and reported. Only ``sqlite3.Error`` is caught —
    any other exception propagates, so a real bug in *scan_fn* surfaces instead
    of being silently downgraded to "unreadable database".

    Writes the per-database warnings during the loop and the aggregate
    "results below are incomplete" warning after it, so a caller printing its
    report afterwards keeps the warnings above the results.
    """
    matches: list[Any] = []
    unreadable: list[str] = []
    for db_path in db_paths:
        try:
            matches.extend(scan_fn(db_path))
        except sqlite3.Error as exc:
            print(f"warning: skipping unreadable database {db_path}: {exc}", file=sys.stderr)
            unreadable.append(db_path)

    if unreadable:
        print(
            f"warning: {len(unreadable)} database(s) skipped due to read errors "
            "(see warnings above); results below are incomplete",
            file=sys.stderr,
        )

    return matches, unreadable


def run_scan_cli(
    argv: list[str] | None,
    *,
    parser: argparse.ArgumentParser,
    scan_fn: Callable[[str], list[Any]],
    render: Callable[[list[Any], argparse.Namespace], str],
) -> int:
    """Shared leak-scanner ``main()`` body.

    Exit codes: 0 = clean, 1 = at least one match found, 2 = no tasks.db could
    be resolved from --db / --project-root / DASHBOARD_KNOWN_PROJECT_ROOTS /
    the dark-factory default, 3 = every resolved database was unreadable, so
    NOTHING was scanned (never treat 3 as a clean run).

    A SINGLE unreadable database among several is not exit 3: it stays a
    warn-and-continue skip, and the readable remainder decides 0 vs 1.

    STDOUT ON EXIT 3 STILL LOOKS CLEAN — read this before writing a consumer.
    The report is rendered before the exit-3 branch (mirroring
    ``audit_wiped_metadata_files.py``), so a total-failure sweep still emits a
    well-formed EMPTY payload: an empty ``[]`` under ``--json``, or the
    scanner's ordinary "nothing found" line otherwise. A pipeline that reads
    only stdout — ``... --json | jq -e 'length == 0'`` and friends — therefore
    CANNOT distinguish "scanned everything, found nothing" from "scanned
    NOTHING at all". Every consumer MUST branch on the exit code (and/or read
    the stderr error line); treating a parseable empty payload as a clean
    result is exactly the false green this exit code exists to kill.

    *render* receives the collected matches and the parsed Namespace and
    returns the exact text to print — that is where each scanner's
    JSON-vs-report choice and its own truncation flag live.
    """
    args = parser.parse_args(argv)

    db_paths = discover_db_paths(explicit_dbs=args.dbs, project_roots=args.project_roots)
    if not db_paths:
        print(NO_DB_RESOLVED_MESSAGE, file=sys.stderr)
        return 2

    matches, unreadable = sweep_databases(db_paths, scan_fn)

    print(render(matches, args))

    # Gate on the COUNT, not on "unreadable and not matches": the second
    # element here is MATCHES, not successfully-scanned databases (unlike
    # audit_wiped_metadata_files.py's `audits`, whose condition does not port
    # verbatim). One unreadable db beside a readable CLEAN one yields
    # matches=[] too, and that partial failure must stay 0. db_paths is
    # non-empty by the early return above, and every path either succeeds or
    # lands in `unreadable`, so this equality is exactly "every resolved
    # database was unreadable".
    if len(unreadable) == len(db_paths):
        # Nothing was scanned at all — never report that as a clean sweep.
        print(
            "error: every resolved database was unreadable; NOTHING was "
            "scanned (this is not a clean result)",
            file=sys.stderr,
        )
        return 3

    return 1 if matches else 0


# ---------------------------------------------------------------------------
# Tier 3 — audit-script CLI plumbing (the two audit scripts).
#
# Adopted by audit_wiped_metadata_files.py and audit_combine_gate_marker_loss.py
# (task 3616, closing task 3286's follow-on finding that the two audit main()s
# were ~65 near-identical lines). This tier sweeps PROJECT ROOTS rather than db
# paths, and its per-root callback returns exactly ONE audit object per root —
# see sweep_project_roots' contract, which is what the exit-3 gate rests on.
#
# Each script keeps what genuinely differs: its own --project-root/--json
# parser and epilog wording, its own report/JSON renderers, its own
# "is this dirty?" predicate, and (for wiped) the --min-fidelity filter.
# ---------------------------------------------------------------------------

# The exit-2 signal, emitted verbatim by both audit scripts. Named to parallel
# NO_DB_RESOLVED_MESSAGE above, which carries the db-flavoured spelling.
NO_PROJECT_ROOT_RESOLVED_MESSAGE = (
    "no project root resolvable with a readable tasks.db (checked "
    "--project-root / DASHBOARD_KNOWN_PROJECT_ROOTS / the "
    "dark-factory default)"
)

# Named rather than spelled inline so each script's epilog, its main()
# docstring and run_audit_cli's returns can never drift into disagreeing about
# what a number means — the property audit_combine_gate_marker_loss.py's
# per-script EXIT_* names were introduced for, preserved here now that the
# returns live behind a module boundary. Each denotes EXACTLY ONE outcome;
# that is the whole reason 3 exists rather than being folded into 0.
AUDIT_EXIT_OK = 0                # audited; nothing dirty found
AUDIT_EXIT_FINDINGS = 1          # at least one dirty result (is_dirty fired)
AUDIT_EXIT_NO_ROOT = 2           # no project root resolved to a readable tasks.db
AUDIT_EXIT_NOTHING_AUDITED = 3   # roots resolved but EVERY one failed to audit


def sweep_project_roots(
    roots: Sequence[str],
    audit_fn: Callable[[str], Any],
) -> tuple[list[Any], list[str]]:
    """Run *audit_fn* over every root in *roots*, returning (audits, unreadable).

    CONTRACT — :func:`run_audit_cli`'s exit-3 gate depends on it: *audit_fn*
    returns EXACTLY ONE audit object per root, or raises ``sqlite3.Error``.
    Under that contract ``len(audits) + len(unreadable) == len(roots)`` always
    holds, so "no audits but some unreadable" is precisely "every root failed".
    An *audit_fn* that instead returned None to "skip" a root would break that
    equality and silently re-open the false green exit 3 exists to close
    (docs/legibility/design-invariants.md, no-silent-fail-soft).

    A single unreadable project (e.g. a corrupt/locked tasks.db, or a transient
    "database is locked"/"file is not a database" condition) does not abort the
    sweep: it is logged to stderr and skipped so every other resolvable project
    is still audited and reported. Only ``sqlite3.Error`` is caught — any other
    exception propagates, so a real bug in *audit_fn* surfaces instead of being
    silently downgraded to "unreadable project".

    Writes the per-project warnings during the loop and the aggregate "results
    below are incomplete" warning after it, so a caller printing its report
    afterwards keeps the warnings above the results.

    Structurally parallel to :func:`sweep_databases` on purpose: keeping the
    two shapes aligned is what makes "pure extraction, no behaviour change"
    checkable by diff.
    """
    audits: list[Any] = []
    unreadable: list[str] = []
    for root in roots:
        try:
            audits.append(audit_fn(root))
        except sqlite3.Error as exc:
            print(f"warning: skipping unreadable project {root}: {exc}", file=sys.stderr)
            unreadable.append(root)

    if unreadable:
        print(
            f"warning: {len(unreadable)} project(s) skipped due to read errors "
            "(see warnings above); results below are incomplete",
            file=sys.stderr,
        )

    return audits, unreadable


def run_audit_cli(
    argv: list[str] | None,
    *,
    parser: argparse.ArgumentParser,
    audit_fn: Callable[[str, argparse.Namespace], Any],
    render: Callable[[list[Any], argparse.Namespace], str],
    is_dirty: Callable[[list[Any]], bool],
    on_roots: Callable[[list[str], argparse.Namespace], None] | None = None,
) -> int:
    """Shared audit-script ``main()`` body.

    Exit codes: :data:`AUDIT_EXIT_OK` (0) = audited, nothing dirty;
    :data:`AUDIT_EXIT_FINDINGS` (1) = *is_dirty* fired;
    :data:`AUDIT_EXIT_NO_ROOT` (2) = no project root resolved to a readable
    tasks.db; :data:`AUDIT_EXIT_NOTHING_AUDITED` (3) = roots resolved but EVERY
    one failed to audit, so NOTHING was audited (never treat 3 as a clean run).

    A SINGLE unreadable project among several is not exit 3: it stays a
    warn-and-continue skip, and the readable remainder decides 0 vs 1.

    STDOUT ON EXIT 3 STILL LOOKS CLEAN — read this before writing a consumer.
    The report is rendered before the exit-3 branch (preserving both audit
    scripts' pre-extraction ordering), so a total-failure sweep still emits a
    well-formed EMPTY payload: an empty projects list under ``--json``, or the
    script's ordinary "nothing found" report otherwise. A pipeline that reads
    only stdout — ``... --json | jq -e '.projects | length == 0'`` and friends
    — therefore CANNOT distinguish "audited everything, found nothing" from
    "audited NOTHING at all". Every consumer MUST branch on the exit code
    (and/or read the stderr error line); treating a parseable empty payload as
    a clean result is exactly the false green this exit code exists to kill.

    WHY THE EXIT-3 GATE IS SPELLED ``unreadable and not audits`` here, unlike
    :func:`run_scan_cli`'s ``len(unreadable) == len(db_paths)``: that tier's
    first return element is MATCHES, so one unreadable db beside a readable
    CLEAN one would wrongly trip a "nothing found" gate. THIS tier's first
    element IS the successfully-audited list, so the condition is exactly
    "every root failed" — but only while :func:`sweep_project_roots`'
    one-audit-per-root contract holds. An *audit_fn* that returned None to skip
    a root would silently break it.

    *audit_fn* takes ``(root, args)`` because both adopters need a parsed flag
    inside the per-root call that the calling script cannot close over (argv is
    parsed here): ``--min-fidelity`` for one, ``--project-id`` for the other.
    It is bound to a one-argument closure at the :func:`sweep_project_roots`
    call below so that function keeps a signature parallel to
    :func:`sweep_databases`. *is_dirty* deliberately does NOT take *args* —
    neither adopter's predicate reads a flag.

    *on_roots*, when supplied, is called with the RESOLVED root list BEFORE the
    empty-roots return, so a hook can observe the empty case. That is where
    ``audit_combine_gate_marker_loss.py``'s multi-root ``--project-id`` warning
    has always sat: a once-per-run warning computed from the resolved roots,
    which fits neither *audit_fn* (per-root) nor *render* (post-sweep).
    """
    args = parser.parse_args(argv)

    roots = discover_project_roots(project_roots=args.project_roots)

    if on_roots is not None:
        on_roots(roots, args)

    if not roots:
        print(NO_PROJECT_ROOT_RESOLVED_MESSAGE, file=sys.stderr)
        return AUDIT_EXIT_NO_ROOT

    audits, unreadable = sweep_project_roots(roots, lambda root: audit_fn(root, args))

    print(render(audits, args))

    if unreadable and not audits:
        # Nothing was audited at all — never report that as a clean sweep.
        print(
            "error: every resolved project was unreadable; NOTHING was "
            "audited (this is not a clean result)",
            file=sys.stderr,
        )
        return AUDIT_EXIT_NOTHING_AUDITED

    return AUDIT_EXIT_FINDINGS if is_dirty(audits) else AUDIT_EXIT_OK
