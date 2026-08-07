"""Shared plumbing for the READ-ONLY tasks.db sweep scripts.

Home of the tasks.db discovery ladder, the leak-scanner CLI skeleton and the
audit-script CLI skeleton that were previously copied across
``scan_task_toolcall_leaks.py``, ``scan_provenance_note_log_leaks.py``,
``audit_wiped_metadata_files.py`` and ``audit_combine_gate_marker_loss.py``
(tasks 3336 and 3616, closing task 3286's "~134 identical lines" finding).
Every function body here was lifted from those copies verbatim, so "pure
extraction, no behaviour change" stays checkable by diff.

This module is NOT a CLI in its own right — the leading underscore marks it as
importable-by-sibling-scripts only. It hosts three tiers:

* **Tier 1, discovery** (``_DEFAULT_PROJECT_ROOTS``, :func:`tasks_db_path`,
  :func:`resolve_project_roots`, :func:`discover_project_roots`,
  :func:`discover_db_paths`) — adopted by ALL FOUR sweep scripts.
* **Tier 2, leak-scanner CLI plumbing** (:func:`sweep_databases`,
  :func:`run_scan_cli`, :func:`add_db_discovery_args`,
  :data:`NO_DB_RESOLVED_MESSAGE`, :func:`format_json`, :func:`truncate`,
  :func:`group_matches_by_db`) — adopted by the two LEAK SCANNERS only. It
  sweeps DB PATHS and collects MATCHES; the audit scripts sweep PROJECT ROOTS
  and collect one audit per root, which is Tier 3.
* **Tier 3, audit-script CLI plumbing** (:func:`run_audit_cli`,
  :func:`sweep_project_roots`, the ``AUDIT_EXIT_*`` codes,
  :data:`NO_PROJECT_ROOT_RESOLVED_MESSAGE`, :func:`format_kv_line`,
  :func:`format_coverage_block`) — adopted by ``audit_wiped_metadata_files.py``
  and ``audit_combine_gate_marker_loss.py`` (task 3616).

Both audit scripts deliberately keep their own ``format_report`` /
``format_json`` / ``_build_parser``. What remains genuinely per-script, and
must NOT be folded in here: the two ``--project-root``/``--json`` parsers and
their two epilogs (one speaks of candidates, the other of ACTIONABLE findings
and terminal-row suppression); wiped's ``--min-fidelity`` filter; both scripts'
object-shaped rather than array-shaped JSON; combine's ``--project-id``
handling; and each COVERAGE block's caveat PROSE and label set — only the
block's alignment is shared, because those labels say what each script could
not see. Folding any of them in would silently delete those semantics.

Exit 3 is NOT among the differences: task 3474 gave :func:`run_scan_cli` the
same "nothing was scanned/audited, so this is not a clean run" semantics that
the audit scripts had, so ``docs/legibility/design-invariants.md``'s
no-silent-fail-soft rule is honoured by every tier here. Tiers 2 and 3 spell
the exit-3 GATE differently on purpose — see :func:`run_audit_cli`'s docstring
for why the audit spelling does not port to Tier 2 and vice versa.

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
# The AUDIT scripts do not adopt THIS tier, and should not: it sweeps db
# PATHS and accumulates MATCHES, while they sweep project ROOTS and collect
# exactly one audit object per root. That difference is load-bearing rather
# than cosmetic — it is why the two tiers' exit-3 gates are spelled
# differently (see run_scan_cli's comment below and run_audit_cli's
# docstring). Their shared main() body is Tier 3 instead.
#
# Exit 3 (every resolved target failed, kept distinct from 0 per
# docs/legibility/design-invariants.md's no-silent-fail-soft rule) is common
# to both tiers as of task 3474, which is what makes this shared plumbing
# consistent with the invariant rather than an exception to it.
#
# WHY THIS TIER RETURNS BARE 0/1/2/3 while Tier 3 returns named AUDIT_EXIT_*
# constants: the two tiers' NUMBERS are identical and must stay in lockstep
# (0 clean / 1 found something / 2 nothing resolved / 3 resolved but nothing
# scanned-or-audited), but each number's SENTENCE is per-tier -- here 2 is "no
# tasks.db resolvable" and 3 is "every resolved DATABASE was unreadable",
# where Tier 3 speaks of project ROOTS. run_scan_cli's four returns all sit
# inside one ~30-line body next to the comment that names each outcome, so the
# literals stay readable in place; Tier 3 needed names because
# audit_combine_gate_marker_loss.py ALIASES them into its own EXIT_* vocabulary
# across a module boundary, where a re-spelled literal could silently drift
# from what is actually returned. Left inline deliberately, not overlooked --
# a shared tier-neutral set would have to be vague enough to cover both
# wordings, losing exactly the precision AUDIT_EXIT_* carries.
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

    PARSER CONTRACT — *parser* MUST define the roots under
    ``dest="project_roots"`` (an ``action="append"`` ``--project-root``, as
    both adopters spell it); that attribute is read off the parsed Namespace
    below. Unlike Tier 2, where :func:`add_db_discovery_args` owns the flag
    definitions outright, this tier leaves each adopter to hand-roll its own
    ``--project-root``/``--json`` because their help text names different
    resolved paths — so the dest name is a convention the two sides must agree
    on rather than something the shared code guarantees. A third adopter that
    spells ``dest="roots"`` gets an ``AttributeError`` from inside this
    function; the fix is the dest, not this function.

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

    # dest="project_roots" is the PARSER CONTRACT above, not an accident.
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


# --- Tier 3 reporting primitives (LAYOUT only, never the prose) -------------

# Width of the coverage block's label column. Every label in both audit scripts
# is <= 31 characters, so all of their values land at column 40 — this constant
# is the MEASURED, byte-exact reproduction of both existing blocks, not a new
# choice. A label at or past the gutter falls back to a single separating
# space rather than colliding with its value.
_COVERAGE_LABEL_WIDTH = 36


def format_kv_line(pairs: Sequence[tuple[str, Any]], *, indent: str = "  ") -> str:
    """Render *pairs* as one indented ``key=value key=value`` line.

    The enforced form of the "precedent's ``key=value`` style" that
    ``audit_combine_gate_marker_loss.py``'s ``_format_finding_line`` docstring
    previously only asserted in prose. Values go through ``str()``, exactly as
    the f-strings this replaces did, so an int, a str and a ``len()`` result
    all render identically to before.

    The FIELD SET stays per-script: the two adopters name different columns,
    and several of their keys deliberately differ from the underlying attribute
    names (``source`` for ``expected_source``, and so on).
    """
    return indent + " ".join(f"{key}={value}" for key, value in pairs)


def format_coverage_block(
    caveat: str | Sequence[str],
    rows: Sequence[tuple[str, Any]],
    details: Sequence[str] = (),
) -> list[str]:
    """Render an always-printed COVERAGE block: caveat, aligned rows, details.

    *caveat* is emitted verbatim above the rows and accepts either a single
    pre-joined string (``audit_combine_gate_marker_loss.py``'s
    ``_COVERAGE_CAVEAT`` constant) or a sequence of lines
    (``audit_wiped_metadata_files.py``'s three-line literal). Each row renders
    as ``f"    {label:<36}{value}"``; each detail as ``f"      - {detail}"``,
    with the detail section omitted entirely when *details* is empty.

    ONLY THE ALIGNMENT IS SHARED, deliberately. The 36-character gutter is the
    measured, byte-exact reproduction of BOTH scripts' existing coverage blocks
    (every label in both files is <= 31 chars, so every value already lands at
    column 40), and a drifting gutter is the failure mode this helper exists to
    prevent. The caveat PROSE and the label/field sets stay per-script: wiped's
    caveat is about unrecoverable plan scope, combine's about combine targets
    with no comparison source, and folding them into one string here would
    delete exactly the semantics each block carries.
    """
    lines = [caveat] if isinstance(caveat, str) else list(caveat)

    for label, value in rows:
        if len(label) < _COVERAGE_LABEL_WIDTH:
            lines.append(f"    {label:<{_COVERAGE_LABEL_WIDTH}}{value}")
        else:
            # Past the gutter: keep a separator rather than rendering an
            # unreadable "label:value" collision.
            lines.append(f"    {label} {value}")

    # NAME what is missing, never just count it — a count alone tells an
    # operator that coverage is incomplete but not where to look.
    for detail in details:
        lines.append(f"      - {detail}")

    return lines
