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
from _task_db_scan import tasks_db_path

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
    """One classified descriptor. ``superseded_by`` is set only when the
    supersession probe attributed a later task to the pattern's return."""

    row: DescriptorRow
    disposition: str
    superseded_by: str | None = None


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
    (its own guard is TestCheckedInScriptCheckTargets). A grep entry with no
    pattern is a SCHEMA defect the corpus validator owns, not a delivery one.
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
_TASK_IN_SUBJECT_RE = re.compile(r"task[/-](\d+)")


def find_superseding_task(
    row: DescriptorRow, *, repo_root: str, since: str | None
) -> str | None:
    """A LATER task that reintroduced *row*'s pattern, or ``None``.

    Only ever consulted for a would-be ``broken`` row, and only ever able to
    DOWNGRADE it — so a false negative costs a footnote and a false positive
    costs a missed defect. That asymmetry is why the attribution is strict:
    the task id must come from the merge-commit subject convention, never
    inferred from the diff or the author.

    *since* is the producer's ``updated_at``; commits at or before it are the
    producer's own work, not a later undoing. Absent, no claim is made at all
    rather than scanning all of history and attributing the producer's own
    merge to it.
    """
    if not since or not row.pattern:
        return None
    argv = [
        "git", "-C", repo_root, "log", "--format=%s",
        f"--since={since}", "-S", row.pattern,
    ]
    if row.paths:
        argv += ["--", *row.paths]
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
        findings.append(
            Finding(
                row=row,
                disposition=classify_descriptor(
                    outcome, status=row.status, superseded_by=superseded_by
                ),
                superseded_by=superseded_by,
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
# Reporting (the CLI itself lands in step-24)
# ---------------------------------------------------------------------------

#: Section order is severity order, and SUPERSEDED sits below the defects
#: deliberately: it is a correctly-authored descriptor that later work undid,
#: and filing it beside real defects is what makes a report get ignored.
_SECTIONS = (
    ("BROKEN", DISPOSITION_BROKEN),
    ("VACUOUS LIVE GATES", DISPOSITION_VACUOUS_LIVE_GATE),
    ("SUPERSEDED", DISPOSITION_SUPERSEDED),
    ("UNEVALUABLE", DISPOSITION_UNEVALUABLE),
)

_COVERAGE_CAVEAT = (
    "COVERAGE — what this sweep could NOT classify. A descriptor whose task is\n"
    "  absent from tasks.db, or whose sidecar would not load, is counted here\n"
    "  rather than dropped: a report that showed only findings would let a\n"
    "  partial sweep read as a complete one."
)


def _format_finding(finding: Finding) -> str:
    row = finding.row
    parts = [
        f"task_id={row.task_id}",
        f"name={row.name}",
        f"expect={row.expect}",
        f"pattern={row.pattern!r}",
        f"source={row.source}",
    ]
    if finding.superseded_by:
        parts.append(f"superseded_by={finding.superseded_by}")
    return "  " + " ".join(parts)


def format_report(audits: list[ProjectAudit]) -> str:
    """Human-readable report: one block per project, sections then COVERAGE."""
    lines: list[str] = []
    for audit in audits:
        lines.append(f"=== {audit.project_root}")
        for label, disposition in _SECTIONS:
            rows = [f for f in audit.findings if f.disposition == disposition]
            lines.append(f"  {label} ({len(rows)})")
            lines.extend(_format_finding(f) for f in rows)
        lines.append("")
        lines.append("  " + _COVERAGE_CAVEAT)
        for label, value in (
            ("descriptors swept", audit.coverage.descriptors_total),
            ("descriptors with no task row", audit.coverage.descriptors_without_task),
            ("descriptors git could not evaluate", audit.coverage.unevaluable),
            ("sidecars that would not load", audit.coverage.sidecars_unloadable),
        ):
            lines.append(f"    {label:<36}{value}")
        lines.append("")
    if not audits:
        # Even an empty sweep names its sections, so a reader can tell "no
        # defects" from "this tool does not report that".
        lines.append("no projects audited")
        for label, _ in _SECTIONS:
            lines.append(f"  {label} (0)")
        lines.append("  " + _COVERAGE_CAVEAT)
    return "\n".join(lines)


def format_json(audits: list[ProjectAudit]) -> str:
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
