#!/usr/bin/env python3
"""Retroactive, STATUS-AWARE sweep over checked-in ``delivered_checks`` (task 3500).

READ-ONLY / REPORT-ONLY: this module and its CLI never mutate a task record or
a manifest file. Every database connection it opens is a read-only SQLite URI
(``sqlite3.connect(f"file:{path}?mode=ro", uri=True)``), so the sweep is
structurally incapable of writing to the live WAL database the running
orchestrator holds open. Manifest YAML on disk and git history are only ever
read. There is no ``--apply`` flag and no MCP client is ever constructed.
REMEDIATION IS A SEPARATE, REVIEWED FOLLOW-UP — repairing a descriptor from
this report is never done by this script (the audit/repair split of tasks
3146 / 3329).

WHAT IT ANSWERS. ``shared.delivered_check_polarity`` gates NEW descriptors at
authoring time, where the reference tree is free: ``commit_planning`` runs
before the task is implemented, so HEAD *is* the pre-task tree and "already
green" is decisive. That gate cannot see the descriptors already committed.
This script sweeps those, and it needs one extra input to do it.

WHY STATUS IS THE AXIS. Evaluating a checked-in descriptor against a tree
yields a BIT, not a verdict. "``expect: present`` and it matches" is the
SUCCESS state of a landed producer AND a never-fires vacuous gate on a live
one; the two are indistinguishable from the descriptor alone. Measured over
the checked-in corpus, applying the authoring-time 2x2 with the reference tree
set to main-today flags 313/548 descriptors (57%) — overwhelmingly correctly
delivered work (esc-3500-1). The producer's STATUS is what turns the bit into
a disposition, and status lives in ``.taskmaster/tasks/tasks.db``. That is why
this is a script rather than a test: the shared-suite ratchet
(``shared/tests/test_capability_manifest.py::TestCheckedInGrepDescriptorHygiene``)
must run in any checkout, so it sweeps only the STRUCTURAL codes — which are
statements about a descriptor's shape and stay true whatever tree they are
measured against — and deliberately leaves the vacuity direction to this tool.

THE REFERENCE POINT IS MAIN TODAY, NOT THE TASK'S DONE-TIME SHA. That ruling
is measured, not assumed (fold-in esc-4545-2): cross-tabbing done-time-vs-today
surfaces 39 checks that FAILED at their producer's done stamp and DELIVER now,
and every one is a merge-timing artifact — 36 of the 39 land within 2h of the
done stamp and none beyond 24h, i.e. the check was evaluated against a main
that had not yet absorbed the very merge it was checking for. A done-time rule
would carry all 39 as false positives. Main today is also the tree an operator
can actually act on: a descriptor that is broken TODAY is broken for every
dependent dispatched from now on, whatever it did at merge time.

EVALUATION GOES THROUGH THE RUNTIME PRIMITIVE.
``shared.delivered_check_polarity.evaluate_grep_at_tree`` — which
``orchestrator.delivered_checks._run_grep_check`` also delegates to — so the
sweep cannot disagree with the scheduler about ERE-vs-Python regex, the ``-e``
separator, pathspec placement, or the ``rc >= 2 -> ERRORED`` boundary. A sweep
that used its own grep would invent findings the gate does not see, and miss
the ones it does.

EPISTEMIC HONESTY. ``ERRORED`` is its own disposition, never folded into
"never delivered"; a descriptor whose task is absent from tasks.db is reported
under COVERAGE rather than dropped; and SUPERSESSION is reported as its own
disposition rather than as a defect (the measured case: task 3618's
``expect: absent`` gzip checks fail on main today only because task 3578
deliberately restored gzip reading afterwards). Presenting a partial sweep as
complete, or a legitimately-undone descriptor as an authoring defect, would be
exactly the no-silent-fail-soft violation in
docs/legibility/design-invariants.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# Tier 1 (tasks.db discovery). The flat-sibling import contract the exemplar
# records applies verbatim: _task_db_scan.py must stay a flat sibling in
# scripts/, and this script must NEVER be invoked via `python -m` — the CLI
# tests shell out to the script path and resolve this import solely because a
# DIRECTLY-EXECUTED script puts its own directory at sys.path[0].
from _task_db_scan import (
    AUDIT_EXIT_FINDINGS,
    AUDIT_EXIT_NO_ROOT,
    AUDIT_EXIT_NOTHING_AUDITED,
    AUDIT_EXIT_OK,
    format_coverage_block,
    format_kv_line,
    run_audit_cli,
    tasks_db_path,
)

# Bind `shared` to the SAME checkout as this script via a __file__-relative
# path, never a hardcoded absolute. An editable install puts the MAIN
# checkout's shared/src on sys.path for a bare `python3`, so without this a
# copy of this script running from a worktree would evaluate descriptors using
# the MAIN checkout's primitive. Same reasoning and same form as
# audit_combine_gate_marker_loss.py:97-104. The shared.* imports below MUST
# stay after this insert.
_SHARED_SRC = Path(__file__).resolve().parent.parent / "shared" / "src"
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))

from shared.capability_manifest import load_capability_manifest  # noqa: E402
from shared.delivered_check_polarity import (  # noqa: E402
    CheckOutcome,
    evaluate_grep_at_tree,
    extract_delivered_checks,
)

MANIFEST_SUFFIX = ".capability-manifest.yaml"

#: Statuses at which a producer has stopped promising anything. `done` and
#: `cancelled` are BOTH terminal but mean opposite things here, which is why
#: they are not one bucket: a done task's failing check is a defect, a
#: cancelled one's is inert.
DONE_STATUSES = ("done",)
INERT_STATUSES = ("cancelled",)

# --- Dispositions ----------------------------------------------------------
#
# Seven, and the split is load-bearing. `delivered` vs `vacuous_live_gate`
# reads the SAME evaluation bit — they are told apart only by status, which is
# the whole reason this tool exists. `unevaluable` and `no_task` exist so that
# "we could not tell" is never rendered as "we checked and it is fine".

DISPOSITION_DELIVERED = "delivered"                    # done + passes: success
DISPOSITION_BROKEN = "broken"                          # done + fails: never delivered
DISPOSITION_SUPERSEDED = "superseded"                  # done + fails, later task undid it
DISPOSITION_VACUOUS_LIVE_GATE = "vacuous_live_gate"    # live + already passes: gates nothing
DISPOSITION_HEALTHY = "healthy"                        # live + fails: normal, forward-looking
DISPOSITION_INERT = "inert"                            # cancelled producer: promises nothing
DISPOSITION_UNEVALUABLE = "unevaluable"                # git could not answer
DISPOSITION_NO_TASK = "no_task"                        # descriptor reached no task row

#: The dispositions an operator must act on. `superseded` is deliberately NOT
#: here: the descriptor was correct and delivered, and later work legitimately
#: undid it — filing it beside real defects is what makes a report get ignored.
DEFECT_DISPOSITIONS = (DISPOSITION_BROKEN, DISPOSITION_VACUOUS_LIVE_GATE)


class DescriptorRow(NamedTuple):
    """One ``kind: grep`` delivered_check, joined to its producer's status.

    ``status`` is ``None`` when the descriptor came from a sidecar naming a
    task this project's tasks.db does not carry — the COVERAGE case, kept as a
    row rather than dropped so the report can name what it could not classify.

    ``source`` distinguishes the two places a descriptor lives: ``'metadata'``
    (``tasks.db``'s ``metadata.delivered_checks``, the copy the runtime gate
    actually evaluates) and ``'manifest'`` (the checked-in sidecar, the
    authoring original). They can disagree — a sidecar capability whose task
    was never stamped has no metadata copy at all — and the source is what
    tells an operator which artifact to repair.
    """

    task_id: int
    tag: str
    status: str | None
    name: str
    kind: str
    pattern: str | None
    expect: str | None
    paths: tuple[str, ...]
    source: str
    manifest: str | None
    task_label: str | None


class Finding(NamedTuple):
    """One classified descriptor.

    ``superseded_by`` is set only when the supersession probe attributed a
    later task to the pattern's return.

    ``open_dependents`` names the still-open tasks that depend on this
    descriptor's producer — the tasks a defective check actually holds up. It
    is populated only for ACTIONABLE dispositions, because it answers "who is
    stuck", and nobody is stuck behind a healthy or a delivered check. Both
    trailing fields carry defaults so a synthetic Finding in a test can be
    built from the two that define it.
    """

    row: DescriptorRow
    disposition: str
    superseded_by: str | None = None
    open_dependents: tuple[int, ...] = ()


class AuditCoverage(NamedTuple):
    """What the sweep could NOT classify — always printed, never inferred.

    A report that shows only findings lets a reader mistake a partial sweep
    for a complete one.
    """

    descriptors_total: int
    descriptors_without_task: int
    unevaluable: int
    sidecars_unloadable: int


class ProjectAudit(NamedTuple):
    project_root: str
    findings: list[Finding]
    coverage: AuditCoverage


# ---------------------------------------------------------------------------
# The pure core
# ---------------------------------------------------------------------------


def classify_descriptor(
    outcome: CheckOutcome,
    *,
    status: str | None,
    superseded_by: str | None = None,
) -> str:
    """Disposition for one descriptor, from its outcome and its producer's status.

    PURE — no git, no database, no clock. Every input that could vary is an
    argument, which is what makes the whole classification testable against
    synthetic rows and what keeps the two sources of truth (evaluation and
    status) visibly separate.

    The ordering of the guards is the semantics:

    1. ``ERRORED`` first. It is not ``FAIL``. Folding "git could not answer"
       into "the capability was never delivered" is the exact confusion the
       runtime gate's own ``rc >= 2 -> ERRORED`` boundary exists to prevent.
    2. No status at all — the descriptor reached no task row — is a COVERAGE
       fact and cannot be a verdict about delivery.
    3. A cancelled producer is inert in BOTH evaluation cells: it promised
       nothing (so a failing check is not a defect) and it gates nothing (so a
       passing one is not a vacuous live gate).
    4. Done: ``PASS`` is the success state; ``FAIL`` is the defect, unless a
       later task is known to have reintroduced the pattern.
    5. Live: ``PASS`` is the vacuous gate this sweep exists to surface, and
       ``FAIL`` is the ordinary forward-looking majority.

    *superseded_by* only ever explains away a would-be DEFECT. It is ignored
    on every other cell, so a supersession signal can never launder a real
    disposition into a footnote.
    """
    if outcome is CheckOutcome.ERRORED:
        return DISPOSITION_UNEVALUABLE
    if status is None:
        return DISPOSITION_NO_TASK
    if status in INERT_STATUSES:
        return DISPOSITION_INERT
    if status in DONE_STATUSES:
        if outcome is CheckOutcome.PASS:
            return DISPOSITION_DELIVERED
        return DISPOSITION_SUPERSEDED if superseded_by else DISPOSITION_BROKEN
    if outcome is CheckOutcome.PASS:
        return DISPOSITION_VACUOUS_LIVE_GATE
    return DISPOSITION_HEALTHY


def evaluate_row(row: DescriptorRow, *, repo_root: str, ref: str = "HEAD") -> CheckOutcome:
    """Evaluate one descriptor against *ref*, through the RUNTIME primitive.

    Deliberately a one-line delegation rather than its own grep:
    :func:`shared.delivered_check_polarity.evaluate_grep_at_tree` is the same
    function ``orchestrator.delivered_checks._run_grep_check`` delegates to, so
    this sweep and the scheduler cannot disagree about what a check means.
    """
    return evaluate_grep_at_tree(
        row.pattern or "",
        list(row.paths),
        expect=row.expect,
        repo_root=repo_root,
        ref=ref,
    )


# ---------------------------------------------------------------------------
# Source 1 — tasks.db
# ---------------------------------------------------------------------------


def _grep_rows_from_checks(
    checks: list[dict],
    *,
    task_id: int,
    tag: str,
    status: str | None,
    source: str,
    manifest: str | None = None,
    task_label: str | None = None,
) -> list[DescriptorRow]:
    """Filter *checks* down to evaluable grep descriptors and shape them.

    Only ``kind == 'grep'`` survives: the sweep is a statement about grep
    POLARITY against a tree, and a script check has no pattern to evaluate
    (its own guard is TestCheckedInScriptCheckTargets). ``kind == 'path'`` is
    not swept either: :func:`find_superseding_task` is a pickaxe over a
    PATTERN and this tool has no path-history counterpart to it, so a done
    path check that fails could not be told apart from a superseded one. A grep entry with
    no pattern is a SCHEMA defect the corpus validator owns, not a delivery
    one.
    """
    rows = []
    for check in checks:
        if not isinstance(check, dict) or check.get("kind") != "grep":
            continue
        pattern = check.get("pattern")
        name = check.get("name")
        if not isinstance(pattern, str) or not pattern:
            continue
        if not isinstance(name, str) or not name:
            continue
        paths = check.get("paths")
        rows.append(
            DescriptorRow(
                task_id=task_id,
                tag=tag,
                status=status,
                name=name,
                kind="grep",
                pattern=pattern,
                expect=check.get("expect"),
                paths=tuple(p for p in (paths or []) if isinstance(p, str)),
                source=source,
                manifest=manifest,
                task_label=task_label,
            )
        )
    return rows


def load_metadata_checks(db_path: str) -> list[DescriptorRow]:
    """Every grep ``metadata.delivered_checks`` entry in *db_path*, with status.

    Read-only URI — the guarantee is structural, not a convention. Malformed
    metadata is SKIPPED rather than raised: a single undecodable row must not
    abort a whole-project sweep, and ``extract_delivered_checks`` already
    implements exactly that benign-absent contract (the same one
    ``lock_charter_guard.extract_files`` uses at the wire boundary).
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("SELECT tag, id, status, metadata FROM tasks")
        rows: list[DescriptorRow] = []
        for record in cursor:
            checks = extract_delivered_checks(record["metadata"])
            if not checks:
                continue
            rows.extend(
                _grep_rows_from_checks(
                    checks,
                    task_id=int(record["id"]),
                    tag=record["tag"],
                    status=record["status"],
                    source="metadata",
                )
            )
        return rows
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Source 2 — the checked-in sidecars
# ---------------------------------------------------------------------------


def manifest_paths(project_root: str) -> list[Path]:
    """Every tracked ``*.capability-manifest.yaml`` under *project_root*.

    ``git ls-files``, never ``rglob`` — the same anti-worktree reasoning
    ``shared/tests/capability_manifest_corpus.py`` records: ``.worktrees/`` is
    gitignored and holds thousands of other tasks' checkouts, so an
    rglob-based sweep would audit unrelated working copies. A non-checkout
    yields ``[]`` rather than raising: the tasks.db half of the join is still
    worth reporting.
    """
    try:
        result = subprocess.run(
            ["git", "-C", project_root, "ls-files", "-z", "--", f"*{MANIFEST_SUFFIX}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [Path(project_root) / rel for rel in sorted(set(result.stdout.split("\0"))) if rel]


def load_manifest_checks(
    project_root: str, statuses: dict[int, tuple[str, str]]
) -> tuple[list[DescriptorRow], int]:
    """Sidecar grep descriptors, joined to *statuses*; plus the unloadable count.

    *statuses* maps ``task_id -> (tag, status)``. A capability whose task is
    absent gets ``status=None``, which :func:`classify_descriptor` renders as
    ``no_task`` — reported under COVERAGE rather than dropped, because a
    sidecar naming a task this store does not carry is exactly the kind of gap
    a silent sweep would hide.

    The unloadable count is returned rather than raised for the same reason
    ``iter_script_checks`` swallows: an unloadable sidecar is a manifest-shape
    defect that ``TestCheckedInManifestCorpus`` already reports with a better
    attributed message, and re-reporting it here would misattribute it. It is
    still COUNTED, so the coverage block can say how much of the corpus this
    run could not read.
    """
    rows: list[DescriptorRow] = []
    unloadable = 0
    for path in manifest_paths(project_root):
        try:
            doc = load_capability_manifest(path)
        except Exception:  # noqa: BLE001 - counted, never silent; see docstring
            unloadable += 1
            continue
        rel = str(path.relative_to(project_root))
        for task in doc.tasks:
            task_id = getattr(task, "task_id", None)
            if task_id is None:
                continue
            tag, status = statuses.get(int(task_id), ("master", None))
            for cap in task.capabilities:
                check = cap.delivered_check
                if check is None or check.kind != "grep":
                    continue
                rows.extend(
                    _grep_rows_from_checks(
                        [
                            {
                                "name": cap.name,
                                "kind": "grep",
                                "pattern": check.pattern,
                                "expect": check.expect,
                                "paths": list(check.paths),
                            }
                        ],
                        task_id=int(task_id),
                        tag=tag,
                        status=status,
                        source="manifest",
                        manifest=rel,
                        task_label=task.label,
                    )
                )
    return rows, unloadable


# ---------------------------------------------------------------------------
# Source 3 — git history, for supersession only
# ---------------------------------------------------------------------------

#: The merge-commit subject convention the merge lane writes. Deriving a task
#: id from anything else would be guessing; a commit this cannot attribute
#: simply yields no supersession claim, which degrades toward reporting the
#: row as the defect it appears to be rather than explaining it away.
#: The two subject spellings that ATTRIBUTE a commit to a task explicitly: the
#: merge-lane convention (`Merge task/3578 into main`, also `task-3578`) and the
#: direct-to-main trailing parenthetical (`... (task 3578)`). MEASURED against
#: this repo's real log, and deliberately narrow at both edges. It does NOT
#: match `tasks 3618-3621` (the `s` breaks the character class), which is how a
#: PRD/manifest commit naming a RANGE of tasks stays unattributed; and it does
#: not match the in-lane `impl(3540):` / `amend(4776):` prefixes, which are the
#: producer's own pre-merge commits rather than a later undoing — those still
#: reach us through their `Merge task/3540 into main` commit. Widening this is
#: how a false positive gets in, and a false positive here silently downgrades
#: a REAL defect to a footnote.
_TASK_IN_SUBJECT_RE = re.compile(r"task[\s/-]#?(\d+)")

#: Sidecars quote a descriptor's pattern VERBATIM on their `pattern:` line, so
#: the commit that introduced a manifest changes the pickaxe count for its own
#: pattern and would be attributed as a superseding task. Excluded for the same
#: reason the authoring gate carries a `self_referential` code: a descriptor
#: matching its own declaration is evidence about nothing.
_SUPERSESSION_EXCLUDE_PATHSPEC = f":(exclude)*{MANIFEST_SUFFIX}"


def find_superseding_task(
    row: DescriptorRow, *, repo_root: str, since: str | None
) -> str | None:
    r"""A LATER task that reintroduced *row*'s pattern, or ``None``.

    Only ever consulted for a would-be ``broken`` row, and only ever able to
    DOWNGRADE it — so a false negative costs a footnote and a false positive
    costs a missed defect. That asymmetry is why the attribution is strict:
    the task id must come from the merge-commit subject convention, never
    inferred from the diff or the author.

    *since* is the producer's ``updated_at``; commits at or before it are the
    producer's own work, not a later undoing. Absent, no claim is made at all
    rather than scanning all of history and attributing the producer's own
    merge to it.

    ``--pickaxe-regex`` IS LOAD-BEARING, not a refinement. A bare ``-S`` is a
    LITERAL-string pickaxe, but every pattern here is the POSIX ERE that
    ``git grep -E`` evaluates — so without it the probe searches for the
    characters ``^import gzip`` rather than for the line they describe, and
    silently finds nothing for any pattern carrying a metacharacter. That is
    not a rare corner: it is exactly the measured exemplar this disposition
    exists for (task 3618's ``^import gzip`` / ``gzip\.open`` checks, undone by
    task 3578), which a literal pickaxe reports as ``broken`` — the precise
    false positive the module docstring promises not to emit. Verified on this
    repo: literal ``-S`` returns nothing for both patterns, ``--pickaxe-regex``
    returns 3578's commits for both.
    """
    if not since or not row.pattern:
        return None
    argv = [
        "git", "-C", repo_root, "log", "--format=%s",
        f"--since={since}", "-S", row.pattern, "--pickaxe-regex",
    ]
    # A pathspec of nothing-but-exclusions applies the exclusion to the
    # implicit "everything" — verified against this repo's git rather than
    # assumed, since older git treats it as matching nothing.
    argv += ["--", *row.paths, _SUPERSESSION_EXCLUDE_PATHSPEC]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for subject in result.stdout.splitlines():
        match = _TASK_IN_SUBJECT_RE.search(subject)
        if match and match.group(1) != str(row.task_id):
            return match.group(1)
    return None


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------


#: A dependent in one of these statuses is no longer waiting on anything, so a
#: defective producer check is not holding it up. `deferred` is deliberately
#: NOT here: a deferred task still needs the capability, it is just not being
#: dispatched today, so it belongs in the "who is stuck" list.
CLOSED_DEPENDENT_STATUSES = ("done", "cancelled")


def load_open_dependents(db_path: str) -> dict[int, tuple[int, ...]]:
    """``{producer_task_id: (open dependent ids, ...)}``, read-only.

    THE POINT OF SCOPE ITEM 5. A report naming a broken descriptor without
    naming who is stuck behind it does not let an operator triage — the
    originating complaint is precisely that a dependent sits blocked on a
    capability that can never be delivered.

    A MISSING ``dependencies`` TABLE IS NOT AN ERROR HERE, and the swallow is
    narrow and deliberate. ``sweep_project_roots`` catches ``sqlite3.Error``
    and turns it into "unreadable project"; an ``OperationalError`` escaping
    this function would therefore demote a perfectly readable project to a
    skip, and — if it were the only root — turn a healthy sweep into a false
    exit 3, the exact silent-fail-soft that exit code exists to prevent. Only
    the "no such table" shape is swallowed; every other ``sqlite3.Error``
    propagates so a genuinely corrupt database still surfaces.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT d.depends_on AS producer, d.task_id AS dependent "
                "FROM dependencies d JOIN tasks t "
                "  ON t.id = d.task_id AND t.tag = d.tag "
                f"WHERE t.status NOT IN ({','.join('?' * len(CLOSED_DEPENDENT_STATUSES))})",
                CLOSED_DEPENDENT_STATUSES,
            )
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc):
                raise
            return {}
        out: dict[int, list[int]] = {}
        for record in cursor:
            out.setdefault(int(record["producer"]), []).append(int(record["dependent"]))
        return {producer: tuple(sorted(set(v))) for producer, v in out.items()}
    finally:
        conn.close()


def _load_statuses(db_path: str) -> tuple[dict[int, tuple[str, str]], dict[int, str]]:
    """``{task_id: (tag, status)}`` and ``{task_id: updated_at}``, read-only."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        statuses: dict[int, tuple[str, str]] = {}
        stamps: dict[int, str] = {}
        for record in conn.execute("SELECT tag, id, status, updated_at FROM tasks"):
            statuses[int(record["id"])] = (record["tag"], record["status"])
            stamps[int(record["id"])] = record["updated_at"]
        return statuses, stamps
    finally:
        conn.close()


def audit_project(project_root: str, ref: str = "HEAD") -> ProjectAudit:
    """Join tasks.db, the sidecars and the tree; classify every descriptor.

    Raises ``sqlite3.Error`` on an unreadable database, which is the contract
    ``_task_db_scan.sweep_project_roots`` depends on: exactly one audit object
    per root, or a raise. Returning a sentinel to "skip" would silently break
    the exit-3 gate that stops a total-failure sweep reading as clean.

    Deduplication is by ``(task_id, name)`` with the METADATA copy winning:
    that is the artifact the runtime gate actually evaluates, so when the
    sidecar and the stamped metadata disagree, the stamped one is the live
    behaviour and the sidecar row would be a phantom.
    """
    db = str(tasks_db_path(project_root))
    statuses, stamps = _load_statuses(db)
    open_dependents = load_open_dependents(db)
    metadata_rows = load_metadata_checks(db)
    manifest_rows, unloadable = load_manifest_checks(project_root, statuses)

    seen = {(r.task_id, r.name) for r in metadata_rows}
    rows = metadata_rows + [r for r in manifest_rows if (r.task_id, r.name) not in seen]

    findings: list[Finding] = []
    for row in rows:
        outcome = evaluate_row(row, repo_root=project_root, ref=ref)
        superseded_by = None
        if outcome is CheckOutcome.FAIL and row.status in DONE_STATUSES:
            superseded_by = find_superseding_task(
                row, repo_root=project_root, since=stamps.get(row.task_id)
            )
        disposition = classify_descriptor(
            outcome, status=row.status, superseded_by=superseded_by
        )
        findings.append(
            Finding(
                row=row,
                disposition=disposition,
                superseded_by=superseded_by,
                # Only for a defect: "who is stuck" is a triage question, and
                # nobody is stuck behind a delivered or forward-looking check.
                open_dependents=(
                    open_dependents.get(row.task_id, ())
                    if disposition in DEFECT_DISPOSITIONS
                    else ()
                ),
            )
        )

    return ProjectAudit(
        project_root=project_root,
        findings=findings,
        coverage=AuditCoverage(
            descriptors_total=len(findings),
            descriptors_without_task=sum(
                1 for f in findings if f.disposition == DISPOSITION_NO_TASK
            ),
            unevaluable=sum(
                1 for f in findings if f.disposition == DISPOSITION_UNEVALUABLE
            ),
            sidecars_unloadable=unloadable,
        ),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

#: LIVE defects first, TERMINAL ones after — the report's whole triage claim.
#: A vacuous gate on a live producer is holding dependents up RIGHT NOW and can
#: be repaired by editing a task that is still open; a broken check on a closed
#: producer is historical debt whose repair is a follow-up. Printing the
#: history first is how a report stops being read.
_LIVE_SECTIONS = (
    ("VACUOUS LIVE GATES", DISPOSITION_VACUOUS_LIVE_GATE),
)

#: SUPERSEDED sits below BROKEN and is deliberately in this group rather than
#: among the defects: it is a correctly-authored descriptor that later work
#: legitimately undid, and filing it beside real defects is what makes a report
#: get ignored.
_TERMINAL_SECTIONS = (
    ("BROKEN", DISPOSITION_BROKEN),
    ("SUPERSEDED", DISPOSITION_SUPERSEDED),
    ("UNEVALUABLE", DISPOSITION_UNEVALUABLE),
)

_SECTIONS = _LIVE_SECTIONS + _TERMINAL_SECTIONS

#: One line per disposition saying WHY, in the reader's terms. A disposition
#: name alone is a verdict without an argument: it tells an operator what the
#: classifier concluded but not whether to act, which forces them back into
#: this file to re-derive the rule. Keyed by disposition so a new one cannot be
#: added without noticing the reason is missing.
_REASONS = {
    DISPOSITION_BROKEN: (
        "the producer is done, but this check does not pass against main "
        "today — whatever it was meant to gate was never gated, and any "
        "dependent still waiting on it waits forever"
    ),
    DISPOSITION_VACUOUS_LIVE_GATE: (
        "the check ALREADY passes on main while its producer is still open, "
        "so landing the producer cannot change the verdict — it gates "
        "nothing, and a dependent is released for a reason unrelated to the "
        "capability"
    ),
    DISPOSITION_SUPERSEDED: (
        "correctly authored and delivered, then legitimately undone by later "
        "work — REPORTED, NOT ACTIONABLE; repairing it would revert the "
        "superseding change"
    ),
    DISPOSITION_UNEVALUABLE: (
        "git could not answer for this descriptor (rc >= 2, a missing ref, or "
        "no repository) — ERRORED is not FAIL, so nothing is claimed about "
        "delivery either way"
    ),
    DISPOSITION_NO_TASK: (
        "no row in this project's tasks.db carries this task id, so the "
        "producer's status is unknown and the descriptor cannot be classified"
    ),
}

_COVERAGE_CAVEAT = (
    "COVERAGE — what this sweep could NOT classify. A descriptor whose task is\n"
    "  absent from tasks.db, or whose sidecar would not load, is counted here\n"
    "  rather than dropped: a report that showed only findings would let a\n"
    "  partial sweep read as a complete one. The orphan class is real and\n"
    "  measured, not hypothetical — of 543 mechanical sidecar capabilities, 35\n"
    "  reach no task at all; 34 of those sit on tasks carrying no\n"
    "  delivered_checks whatsoever, and 10 of THOSE are still open."
)


def _format_finding(finding: Finding) -> list[str]:
    """One finding as a key=value line plus its indented ``reason:``.

    ``format_kv_line`` owns the indent and the separator and nothing else, so
    the field set and the column ORDER stay here — deliberately, per that
    helper's docstring: what it buys is that this script's finding lines cannot
    drift in SHAPE from the other audit scripts', not that they share fields.
    """
    row = finding.row
    pairs: list[tuple[str, object]] = [
        ("task_id", row.task_id),
        ("status", row.status),
        ("name", row.name),
        ("expect", row.expect),
        ("pattern", repr(row.pattern)),
        ("source", row.source),
    ]
    if row.manifest:
        pairs.append(("manifest", row.manifest))
    if finding.superseded_by:
        pairs.append(("superseded_by", finding.superseded_by))
    if finding.disposition in DEFECT_DISPOSITIONS:
        # ALWAYS emitted for a defect, including when empty: "nothing is stuck
        # behind this one" is itself a triage answer, and omitting the field
        # would make it indistinguishable from "we did not look".
        pairs.append(
            (
                "open_dependents",
                ",".join(str(d) for d in finding.open_dependents) or "none",
            )
        )
    lines = [format_kv_line(pairs)]
    reason = _REASONS.get(finding.disposition)
    if reason:
        lines.append(f"      reason: {reason}")
    return lines


def _coverage_details(audit: ProjectAudit) -> list[str]:
    """NAME the unclassified descriptors, never just count them.

    A bare count tells an operator that coverage is incomplete but not where to
    look, which is a half-measure the shared block's own docstring calls out.
    """
    details = []
    for finding in audit.findings:
        if finding.disposition == DISPOSITION_NO_TASK:
            details.append(
                f"no task row for task_id={finding.row.task_id} "
                f"(capability {finding.row.name!r} in {finding.row.manifest})"
            )
    return sorted(details)


def format_report(audits: list[ProjectAudit]) -> str:
    """Human-readable report: one block per project, sections then COVERAGE."""
    lines: list[str] = []
    for audit in audits:
        lines.append(f"=== {audit.project_root}")
        for label, disposition in _SECTIONS:
            rows = [f for f in audit.findings if f.disposition == disposition]
            lines.append(f"  {label} ({len(rows)})")
            for finding in rows:
                lines.extend(_format_finding(finding))
        lines.append("")
        lines.extend(format_coverage_block(
            _COVERAGE_CAVEAT,
            (
                ("descriptors swept", audit.coverage.descriptors_total),
                ("descriptors with no task row", audit.coverage.descriptors_without_task),
                ("descriptors git could not evaluate", audit.coverage.unevaluable),
                ("sidecars that would not load", audit.coverage.sidecars_unloadable),
            ),
            _coverage_details(audit),
        ))
        lines.append("")
    if not audits:
        # Even an empty sweep names its sections and its caveat, so a reader
        # can tell "no defects" from "this tool does not report that". See
        # run_audit_cli's STDOUT ON EXIT 3 warning: this payload is well-formed
        # and empty on a total-failure sweep too, so only the exit code
        # distinguishes them.
        lines.append("no projects audited")
        for label, _ in _SECTIONS:
            lines.append(f"  {label} (0)")
        lines.append("  " + _COVERAGE_CAVEAT)
    return "\n".join(lines)


def format_json(audits: list[ProjectAudit]) -> str:
    """A JSON OBJECT, never a bare array.

    An array has nowhere to hang COVERAGE, so a consumer reading it could not
    tell a complete sweep from a partial one — and per run_audit_cli's contract
    a consumer must branch on the EXIT CODE anyway, because an empty
    ``projects`` list is also what a total-failure sweep emits.
    """
    return json.dumps(
        {
            "projects": [
                {
                    "project_root": audit.project_root,
                    "findings": [
                        {
                            **finding.row._asdict(),
                            "paths": list(finding.row.paths),
                            "disposition": finding.disposition,
                            "superseded_by": finding.superseded_by,
                            "open_dependents": list(finding.open_dependents),
                            "reason": _REASONS.get(finding.disposition),
                        }
                        for finding in audit.findings
                    ],
                    "coverage": audit.coverage._asdict(),
                }
                for audit in audits
            ]
        },
        indent=2,
        sort_keys=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ALIASES, never re-spellings. The returns live in _task_db_scan.run_audit_cli,
# so a local `EXIT_OK = 0` could drift from what actually gets returned while
# the epilog below kept promising 0 — exactly the drift
# test_exit_constants_alias_the_shared_tier_3_codes exists to catch, here and
# in audit_combine_gate_marker_loss.py.
EXIT_OK = AUDIT_EXIT_OK                        # audited; no ACTIONABLE defect
EXIT_DEFECTS = AUDIT_EXIT_FINDINGS             # a broken or vacuous-live-gate
                                               # descriptor was found
EXIT_NO_ROOT = AUDIT_EXIT_NO_ROOT              # no project root resolved to a
                                               # readable tasks.db
EXIT_NOTHING_AUDITED = AUDIT_EXIT_NOTHING_AUDITED  # roots resolved but EVERY
                                                   # one failed to audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "READ-ONLY, STATUS-AWARE audit of checked-in delivered_checks: "
            "reports every kind=grep descriptor whose polarity against main "
            "today disagrees with its producer's status — a closed producer "
            "whose capability is nowhere on main (broken), or an open one "
            "whose check already passes and therefore gates nothing (vacuous "
            "live gate). Reporting only — never mutates a task or a manifest. "
            "Remediation is a separate, individually-reviewed follow-up."
        ),
        epilog=(
            "exit codes: 0 = audited, no ACTIONABLE defect; 1 = at least one "
            "actionable defect (broken, or a vacuous live gate); 2 = no "
            "project root resolved to a readable tasks.db; 3 = roots resolved "
            "but every one failed to audit, so NOTHING was swept (never treat "
            "3 as a clean run). Superseded, unevaluable, delivered, inert and "
            "forward-looking rows are reported in full but never affect the "
            "exit code."
        ),
    )
    parser.add_argument(
        "--project-root", dest="project_roots", action="append",
        help=(
            "Project root to audit (resolves <root>/.taskmaster/tasks/tasks.db "
            "and the tracked *.capability-manifest.yaml sidecars, evaluated "
            "against that checkout's HEAD). May be repeated."
        ),
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit a JSON object (findings plus coverage) instead of a report.",
    )
    return parser


def _audit_root(root: str, args: argparse.Namespace) -> ProjectAudit:
    """Audit ONE project root. Exactly one audit, or ``sqlite3.Error``.

    That is :func:`_task_db_scan.sweep_project_roots`' one-audit-per-root
    contract, and the exit-3 gate depends on it — a sentinel return to "skip"
    a root would silently re-open the false green.
    """
    del args  # this script takes no flag that varies the per-root audit
    return audit_project(root)


def _render(audits: list[ProjectAudit], args: argparse.Namespace) -> str:
    return format_json(audits) if args.json else format_report(audits)


def _is_dirty(audits: list[ProjectAudit]) -> bool:
    """Exit 1 keys ONLY on the two ACTIONABLE dispositions.

    Everything else is reported in full and counted in COVERAGE but stays out
    of the exit code, for the reason the exemplar's terminal-row suppression
    records: a detector that is permanently red is a detector nobody reads.
    Specifically excluded, each for its own reason —

    * ``superseded``: correctly delivered work that later work undid. Acting on
      it means reverting the superseding change.
    * ``inert`` (a cancelled producer): promised nothing, gates nothing.
    * ``delivered`` / ``healthy``: the success and the ordinary
      forward-looking states, i.e. the overwhelming majority of the corpus.
    * ``unevaluable`` / ``no_task``: coverage facts, not verdicts. Letting
      "git could not answer" exit 1 would make an infrastructure failure
      indistinguishable from a real defect.
    """
    return any(
        f.disposition in DEFECT_DISPOSITIONS for a in audits for f in a.findings
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Exit codes: 0 = audited, no actionable defect; 1 = at least one ``broken``
    or ``vacuous_live_gate`` descriptor; 2 = no project root resolved to a
    readable tasks.db; 3 = roots resolved but every one failed to audit, so
    NOTHING was swept (never treat 3 as a clean run).

    A thin delegation to :func:`_task_db_scan.run_audit_cli` (Tier 3), shared
    with audit_combine_gate_marker_loss.py and audit_wiped_metadata_files.py.
    What stays here is what genuinely differs: this script's parser and epilog,
    its report and JSON shapes, and its actionable-disposition predicate.
    Nothing in this file returns a bare integer.
    """
    return run_audit_cli(
        argv,
        parser=_build_parser(),
        audit_fn=_audit_root,
        render=_render,
        is_dirty=_is_dirty,
    )


if __name__ == "__main__":
    sys.exit(main())
