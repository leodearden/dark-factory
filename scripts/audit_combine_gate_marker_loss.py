#!/usr/bin/env python3
"""Audit the blast radius of the curator-combine ``metadata`` wipe.

READ-ONLY / REPORT-ONLY: this module and its CLI never mutate a task record,
a ticket record, or a manifest file. Every database connection it opens is a
read-only SQLite URI (``sqlite3.connect(f"file:{path}?mode=ro", uri=True)``),
so the sweep is structurally incapable of writing to the live WAL databases
the running orchestrator holds open. Manifest YAML on disk is only ever read.
There is no ``--apply`` flag and no MCP client is ever constructed.
REMEDIATION IS A SEPARATE, REVIEWED FOLLOW-UP — backfilling a lost key from
this report is never done by this script (the audit/repair split of tasks
3146 / 3329).

Background (task 3591): the curator's combine path
(fused-memory/src/fused_memory/middleware/task_interceptor.py, combine block
~2158) writes exactly
``{'curator_action': 'combine', 'curator_justification', 'combined_at'}``, and
it USED TO pass ``metadata_mode='replace'`` — so every OTHER key the surviving
task carried was DROPPED rather than merged. The load-bearing casualty is
``metadata.delivered_checks``: orchestrator/src/orchestrator/delivered_checks.py
defines both ``gate_mark_done_on_delivered_checks`` and
``verify_delivered_checks_on_main`` off that key, so wiping it SILENTLY REMOVES
a mark-done gate — the task still closes, just without the check that was
supposed to hold it. ``prd_path``/``prd_task_label`` (provenance) and
``task_kind`` (dispatch) are the other consumers. This script enumerates every
task bearing the combine signature whose pre-combine metadata can be
reconstructed from an independent source, and reports the keys that are gone.

WHAT A LIVE HIT MEANS, MEASURED RATHER THAN ASSUMED. Task 3446's fix HAS
LANDED: merge commit ``f70877e327`` ("Merge task/3446 into main") is an
ancestor of this branch, and the combine path now passes
``metadata_mode='merge'`` (fused-memory/src/fused_memory/middleware/
task_interceptor.py:2208, under the "Do NOT 'restore' replace" comment). The
leak is therefore CLOSED, and the reading is the regression-signal one: a
FRESH actionable finding (gate_removing / provenance / dispatch) on a
non-terminal task means the fix REGRESSED and must be investigated. Rows that
predate the fix are HISTORICAL DAMAGE, not evidence of a live leak — the 24
then-live victims were hand-remediated on 2026-08-03 and the ~67 terminal
combine targets were deliberately left alone, which is why terminal findings
never drive the exit code (see :func:`main`).

EPISTEMIC HONESTY. Findings are ``ticket-evidenced``, never certain. The
creating ticket is a SUBMIT-TIME snapshot, so it cannot see a key legitimately
added between submit and combine; that is why this script reports key LOSS
(``expected_keys - live_keys``) and never value drift. Most combine targets
have no reconstructable pre-combine state at all, so the report always carries
a COVERAGE block naming how many targets had no comparison source whatsoever.
Presenting a partial sweep as complete would be exactly the
no-silent-fail-soft violation in docs/legibility/design-invariants.md.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import NamedTuple

# Tier 1 (tasks.db discovery) ONLY, imported as a flat sibling. This module
# deliberately keeps its own format_report/format_json/_build_parser/main, for
# the same reasons audit_wiped_metadata_files.py does: a fourth exit code
# (3 = roots resolved but every one failed to audit, kept distinct from 0 per
# docs/legibility/design-invariants.md's no-silent-fail-soft rule) and
# object-shaped rather than array-shaped JSON.
#
# IMPORT-RESOLUTION CONTRACT: _task_db_scan.py must stay a flat sibling in
# scripts/, and this script must NEVER be invoked via `python -m` — the CLI
# tests shell out to the script path and resolve this import solely because a
# DIRECTLY-EXECUTED script puts its own directory at sys.path[0].
from _task_db_scan import (  # noqa: F401  (tasks_db_path/discover_project_roots re-exported)
    AUDIT_EXIT_FINDINGS,
    AUDIT_EXIT_NO_ROOT,
    AUDIT_EXIT_NOTHING_AUDITED,
    AUDIT_EXIT_OK,
    discover_project_roots,
    run_audit_cli,
    tasks_db_path,
)

# Bind `shared` to the SAME checkout as this script via a __file__-relative
# path, never a hardcoded absolute. An editable install puts the MAIN
# checkout's shared/src on sys.path for a bare `python3`, so without this a
# copy of this script running from a worktree would validate manifests using
# the MAIN checkout's schema. Same reasoning and same form as
# repair_wiped_metadata_files.py:65-75 (tasks 2881/2882/3329). The
# shared.capability_manifest import below MUST stay after this insert.
_SHARED_SRC = Path(__file__).resolve().parent.parent / "shared" / "src"
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))

from shared.capability_manifest import load_capability_manifest  # noqa: E402

# The curator verdict this audit is about. The sibling verdict is 'create',
# which files a NEW task and wipes nothing.
CURATOR_ACTION_COMBINE = "combine"


class CombineTarget(NamedTuple):
    """One task bearing the curator-combine signature, as stored in tasks.db.

    ``metadata_keys`` is the CURRENT live key set. On an unremediated combine
    target it is exactly the three keys the combine path writes
    (``curator_action``, ``curator_justification``, ``combined_at``); every
    key a comparison source expected but that is ABSENT from this tuple is a
    finding.

    A NamedTuple rather than a dataclass, following the precedent: ``_asdict()``
    feeds the JSON writer and ``_replace()`` feeds filtering.
    """

    tag: str
    task_id: int
    status: str
    metadata_keys: tuple[str, ...]


def _decode_metadata(raw: object) -> dict:
    """Decode a raw ``metadata`` blob into a dict, degrading to ``{}``.

    Mirrors :func:`audit_wiped_metadata_files._decode_files`. Degrades for
    NULL, an empty string, malformed JSON, or a payload that decodes to
    anything other than a dict (a list, a bare scalar, ``null``). A corrupt
    metadata blob is data to be skipped, never a reason to abort a sweep over
    thousands of tasks.
    """
    if not raw or not isinstance(raw, (str, bytes)):
        return {}
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def load_combine_targets(tasks_db_path: str) -> dict[tuple[str, int], CombineTarget]:
    """Load every curator-combined task from *tasks_db_path*, keyed by ``(tag, id)``.

    Selects only rows whose decoded ``metadata.curator_action`` is the string
    ``'combine'`` — a wrong-typed value is corrupt data and is skipped, and
    the sibling ``'create'`` verdict wipes nothing so it is not a target.

    Keyed by the full ``(tag, id)`` primary key — the live DB uses a single
    ``master`` tag, but the schema permits the same numeric id under two tags
    and collapsing them would silently merge two distinct tasks.

    Opens the database via a read-only URI (``mode=ro``) so the load is
    structurally incapable of mutating live task records even while
    fused-memory holds the same file open in WAL mode. Closed in a
    ``try/finally`` and never a ``with`` block — a sqlite3 ``with`` is a
    TRANSACTION, not a close.
    """
    targets: dict[tuple[str, int], CombineTarget] = {}
    conn = sqlite3.connect(f"file:{tasks_db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute("SELECT tag, id, status, metadata FROM tasks")
        for tag, task_id, status, metadata in cursor:
            payload = _decode_metadata(metadata)
            if payload.get("curator_action") != CURATOR_ACTION_COMBINE:
                continue
            targets[(tag, task_id)] = CombineTarget(
                tag=tag,
                task_id=task_id,
                status=status,
                metadata_keys=tuple(payload),
            )
    finally:
        conn.close()
    return targets


# ---------------------------------------------------------------------------
# Comparison source (1) — the creating ticket.
#
# The curator files every task through a reconciliation ticket, and the
# ticket's candidate payload is a SUBMIT-TIME SNAPSHOT of the metadata the
# task was created with. A status='created' row is therefore the closest thing
# to a pre-combine record of what the task's metadata held.
# ---------------------------------------------------------------------------

# Ticket status meaning "the curator actually filed this task". The other
# statuses ('pending', 'combined', 'dropped', ...) never produced the task
# whose metadata this audit is reconstructing.
TICKET_STATUS_CREATED = "created"


def tickets_db_path(project_root: str) -> Path:
    """``<root>/data/reconciliation/tickets.db`` — the curator's ticket store.

    NOTE: tickets.db lives in the MAIN checkout, not in a task worktree.
    """
    return Path(project_root) / "data" / "reconciliation" / "tickets.db"


def load_ticket_expectations(tickets_db_path: str, project_id: str) -> dict[str, dict]:
    """Load submit-time metadata per task id from *tickets_db_path*.

    Returns ``{str(task_id): submit_metadata_dict}`` built from each created
    ticket's ``candidate_json['metadata']``.

    THE ``project_id`` PREDICATE IS LOAD-BEARING — DO NOT DROP IT AS
    REDUNDANT. Measured read-only on the live store: task_id ``'3157'`` has a
    ``project_id='reify'`` created-row sitting right next to a
    ``project_id='dark_factory'`` row for the same id. A task_id-only query
    would silently import ANOTHER PROJECT's submit payload and then report
    every dark_factory key that payload lacks as LOST — a wall of confident
    false positives. Bound as a parameter, never string-formatted.

    ``task_id`` is a TEXT column, so results are keyed by ``str(task_id)``;
    callers joining from tasks.db's INTEGER id must convert. A row is skipped
    (never raised on) when its ``task_id`` is NULL, its ``candidate_json`` is
    malformed or decodes to a non-dict, or its ``metadata`` is absent or not a
    dict — one corrupt ticket cannot abort a sweep.

    DUPLICATE CREATED-ROWS ARE RESOLVED EXPLICITLY, NEWEST WINS. Nothing in
    the schema forbids two ``status='created'`` rows for one
    ``(project_id, task_id)``, and measured read-only on the live store
    (2026-08-04) 32 such groups exist across 4281 created rows — e.g.
    ``autopilot_video/579`` and a long tail under ``reify`` (dark_factory
    itself currently has none, but this script audits any root passed to
    ``--project-root``). The winning row decides which keys get reported LOST,
    so an unordered scan would make the finding list differ run-to-run.
    ``ORDER BY created_at, ticket_id`` (both TEXT, ISO-8601 so lexicographic
    order IS chronological; ticket_id breaks a same-timestamp tie) plus the
    unconditional overwrite below means the NEWEST created row wins — it is
    the closest surviving record of what the task was filed with.

    Returns ``{}`` rather than raising when the file is absent; the caller
    records that as a coverage fact, not as a clean result.
    """
    if not Path(tickets_db_path).exists():
        return {}

    expectations: dict[str, dict] = {}
    conn = sqlite3.connect(f"file:{tickets_db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute(
            "SELECT task_id, candidate_json FROM tickets "
            "WHERE project_id = ? AND status = ? "
            "ORDER BY created_at, ticket_id",
            (project_id, TICKET_STATUS_CREATED),
        )
        for task_id, candidate_json in cursor:
            if task_id is None:
                continue
            candidate = _decode_metadata(candidate_json)
            metadata = candidate.get("metadata")
            if not isinstance(metadata, dict):
                continue
            expectations[str(task_id)] = metadata
    finally:
        conn.close()
    return expectations


# ---------------------------------------------------------------------------
# Comparison source (2) — the capability manifests.
#
# WHY A GLOBBED REVERSE INDEX RATHER THAN FOLLOWING metadata.prd_path.
# prd_path is ITSELF one of the wiped keys. Resolving a task's manifest by
# reading its live metadata.prd_path would therefore fail on exactly the
# victims this detector exists to find: a task that lost prd_path would
# resolve to no manifest and be filed under "no comparison source" instead of
# being reported as a delivered_checks loss. So the index is built by globbing
# EVERY manifest and keying on the stamped task_id, with nothing read from
# tasks.db at all.
#
# Measured: 31 live manifests bind 250 task_ids with ZERO task_id bound by
# more than one manifest, so the reverse index is unambiguous in practice. The
# ambiguous case is still detected and ACTED ON rather than silently
# first-wins: see _expected_from_manifest (which withholds delivered_checks
# from a contested binding) and AuditCoverage.ambiguous_manifest_bindings.
# ---------------------------------------------------------------------------

# Both manifest homes, swept in this order. Sorted within each glob so the
# report is deterministic run-to-run.
_MANIFEST_GLOBS = (
    ("plans", "*.capability-manifest.yaml"),
    ("docs/prds", "*.capability-manifest.yaml"),
)

# The delivered_check kinds that commit_planning actually copies into
# metadata.delivered_checks. THE REAL RULE, read off the stamping site:
# fused-memory/src/fused_memory/server/manifest_stamping.py:311 is
# `if check is None or check.kind not in ('grep', 'script'): continue`, i.e.
# BOTH mechanical kinds are copied and only 'manual' is dropped (corroborated
# by DeliveredCheckMeta.kind: Literal['grep', 'script']). A grep-only filter
# here would under-count the expected entries and produce FALSE NEGATIVES on
# the one severity class that removes a mark-done gate.
MECHANICAL_CHECK_KINDS = ("grep", "script")


class ManifestExpectation(NamedTuple):
    """What a capability manifest says a task's metadata should carry.

    ``delivered_check_names`` holds the capability names whose
    ``delivered_check`` is mechanical, i.e. exactly the entries
    ``commit_planning`` would have stamped into
    ``metadata.delivered_checks``. Non-empty means the task was gated.

    ``ambiguous`` is True when more than one manifest binds this task_id, in
    which case ``bound_by`` names every one of them and the other fields come
    from the first in sorted order. Flagged rather than silently resolved: a
    first-wins choice would attribute another PRD's checks to the task.
    """

    task_id: str
    manifest_path: str
    prd_path: str
    label: str
    delivered_check_names: tuple[str, ...]
    ambiguous: bool
    bound_by: tuple[str, ...]


def _manifest_paths(project_root: str) -> list[Path]:
    """Every capability-manifest sidecar under *project_root*, sorted."""
    root = Path(project_root)
    paths: list[Path] = []
    for subdir, pattern in _MANIFEST_GLOBS:
        paths.extend(sorted((root / subdir).glob(pattern)))
    return paths


def build_manifest_index(
    project_root: str,
    parse_failures: list[str] | None = None,
) -> dict[str, ManifestExpectation]:
    """Build the task_id -> :class:`ManifestExpectation` reverse index.

    Globs both manifest homes (``<root>/plans`` and ``<root>/docs/prds``) and
    keys on each manifest task's STAMPED ``task_id``. A block whose
    ``task_id`` is still ``None`` (authoring time, before ``commit_planning``
    stamps it) binds nothing and is skipped.

    *parse_failures* is an optional accumulator: a manifest that fails to read,
    parse, or validate is appended to it as a ``"<path>: <error>"`` string and
    skipped, so one malformed sidecar cannot abort the sweep AND the count
    still reaches the coverage block. Recorded rather than swallowed — a sweep
    that could not read half the manifests must never read as complete
    (docs/legibility/design-invariants.md, no-silent-fail-soft).

    Reuses ``shared.capability_manifest``'s validated pydantic models rather
    than re-implementing YAML shape validation, so the mechanical-kind filter
    is a one-line test against an already-parsed ``Literal`` kind.
    """
    index: dict[str, ManifestExpectation] = {}
    for path in _manifest_paths(project_root):
        try:
            doc = load_capability_manifest(path)
        except Exception as exc:  # noqa: BLE001 — see docstring: recorded, not swallowed
            # Deliberately broad: load_capability_manifest raises OSError,
            # yaml.YAMLError and pydantic.ValidationError, and a future schema
            # change could add more. Every one means the same thing here —
            # this sidecar is unreadable — and it is RECORDED, so breadth
            # loses no information.
            if parse_failures is not None:
                parse_failures.append(f"{path}: {exc}")
            continue

        for task in doc.tasks:
            if task.task_id is None:
                continue
            key = str(task.task_id)
            names = tuple(
                cap.name
                for cap in task.capabilities
                if cap.delivered_check is not None
                and cap.delivered_check.kind in MECHANICAL_CHECK_KINDS
            )
            existing = index.get(key)
            if existing is not None:
                # Second binding: keep the first (sorted) manifest's fields but
                # flag the collision and name every binder.
                index[key] = existing._replace(
                    ambiguous=True,
                    bound_by=existing.bound_by + (str(path),),
                )
                continue
            index[key] = ManifestExpectation(
                task_id=key,
                manifest_path=str(path),
                prd_path=doc.prd,
                label=task.label,
                delivered_check_names=names,
                ambiguous=False,
                bound_by=(str(path),),
            )
    return index


# ---------------------------------------------------------------------------
# Severity — RANKED BY CONSUMER, NOT BY GAP SIZE.
#
# Every constant below names the CONCRETE CODE that reads the lost key, with a
# file citation, because that consumer is the entire justification for the
# rank. Ranking by how often a key is lost would invert the report: task_kind
# is the most common gap (all 24 live victims lost it) yet is load-bearing
# only in the rare deterministic case, so a frequency ranking would bury the
# two delivered_checks losses — the only ones that silently remove a mark-done
# gate — under two dozen benign rows.
# ---------------------------------------------------------------------------

# Consumer: orchestrator/src/orchestrator/delivered_checks.py —
# gate_mark_done_on_delivered_checks (:377) and verify_delivered_checks_on_main
# (:198) both read metadata.delivered_checks. Wiping it does not fail the gate,
# it DELETES the gate: the task closes with no check ever run.
SEVERITY_GATE_REMOVING = "gate_removing"

# Consumer: PRD provenance. prd_path / prd_task_label are how a task is traced
# back to the PRD and manifest label that authored it (and how the manifest
# stamper re-finds its sidecar). Losing them orphans the task from its PRD;
# nothing silently stops gating.
SEVERITY_PROVENANCE = "provenance"

# Consumer: Scheduler.is_deterministic (orchestrator/src/orchestrator/
# scheduler.py:2046) — the single source of truth routing a task to
# DeterministicRunner, to the no-lock module path, and to the restart
# stamp-clear. Only applies when the expected value IS 'deterministic'.
SEVERITY_DISPATCH = "dispatch"

# Real provenance loss with no gating or dispatch consumer.
SEVERITY_INFORMATIONAL = "informational"

# NO BEHAVIOURAL CONSEQUENCE AT ALL. Scheduler.is_deterministic (scheduler.py:
# 2046) is exactly `metadata.get('task_kind') == 'deterministic'`, so an ABSENT
# task_kind is byte-identical in behaviour to task_kind='normal'. A lost
# non-deterministic task_kind therefore changes nothing, and is demoted BELOW
# informational so it cannot crowd out a finding that matters.
SEVERITY_BENIGN = "benign"

# Highest-consequence FIRST. This tuple IS the report's sort order.
_SEVERITY_PRECEDENCE = (
    SEVERITY_GATE_REMOVING,
    SEVERITY_PROVENANCE,
    SEVERITY_DISPATCH,
    SEVERITY_INFORMATIONAL,
    SEVERITY_BENIGN,
)

# The severities that denote a REAL consumer losing something: a removed
# mark-done gate, an orphaned PRD provenance chain, a mis-dispatched task.
#
# THIS TUPLE IS THE EXIT CODE'S DEFINITION, and it exists because the severity
# ladder above would otherwise rank consumers for a reader while nothing
# downstream acted on the ranking. SEVERITY_INFORMATIONAL and SEVERITY_BENIGN
# are deliberately EXCLUDED: benign is by construction a no-op (an absent
# task_kind is behaviourally identical to the non-deterministic value lost),
# and informational is real-but-inert provenance. Measured on the live store
# (2026-08-03), keying the exit on "any non-terminal finding" made the script
# exit 1 on 9 live rows of which ZERO were gate_removing or provenance — i.e.
# red on day one for noise, which is exactly the permanently-red-and-therefore-
# ignored failure the terminal-suppression rule already refuses to accept.
# Excluded rows are still REPORTED in full and still counted in COVERAGE.
_ACTIONABLE_SEVERITIES = (
    SEVERITY_GATE_REMOVING,
    SEVERITY_PROVENANCE,
    SEVERITY_DISPATCH,
)

# The task_kind value that is actually load-bearing (scheduler.py:2046).
TASK_KIND_DETERMINISTIC = "deterministic"

# Keys whose loss is provenance-grade.
_PROVENANCE_KEYS = ("prd_path", "prd_task_label")

# The gate-removing key.
_GATE_KEY = "delivered_checks"

_BENIGN_TASK_KIND_REASON = (
    "Scheduler.is_deterministic() (scheduler.py:2046) tests "
    "metadata.get('task_kind') == 'deterministic', so an ABSENT task_kind is "
    "behaviourally identical to the non-deterministic value that was lost; "
    "nothing dispatches differently"
)


def _severity_rank(severity: str) -> int:
    """Position of *severity* in :data:`_SEVERITY_PRECEDENCE`, gate-removing 0.

    Fails soft on an unknown value by returning ``len(_SEVERITY_PRECEDENCE)``,
    so a severity added later sorts LAST instead of raising mid-report —
    mirroring ``_source_rank`` in ``audit_wiped_metadata_files.py``.
    """
    try:
        return _SEVERITY_PRECEDENCE.index(severity)
    except ValueError:
        return len(_SEVERITY_PRECEDENCE)


def classify_gap(key: str, expected_value: object) -> tuple[str, str]:
    """Classify one lost metadata *key* as ``(severity, reason)``.

    *expected_value* is what the comparison source says the key held. It is
    INSPECTED, not merely carried: the ``task_kind`` branch resolves to two
    different severities for the same key depending on whether the lost value
    was ``'deterministic'``. Reporting on the key name alone would rank all 24
    live task_kind losses as dispatch-affecting when only a handful are.

    An unforeseen key falls through to :data:`SEVERITY_INFORMATIONAL` rather
    than raising — the metadata vocabulary grows (docs/task-authoring.md's
    Tier-C ``x_`` namespace is open by design), and a sweep must not abort on
    a key this script has not been taught about.
    """
    if key == _GATE_KEY:
        return (
            SEVERITY_GATE_REMOVING,
            "removes the gate_mark_done_on_delivered_checks / "
            "verify_delivered_checks_on_main mark-done gate "
            "(orchestrator/src/orchestrator/delivered_checks.py); the task "
            "closes with the check never run",
        )
    if key in _PROVENANCE_KEYS:
        return (
            SEVERITY_PROVENANCE,
            "orphans the task from the PRD and manifest label that authored "
            "it; the manifest stamper can no longer re-find its sidecar",
        )
    if key == "task_kind":
        if expected_value == TASK_KIND_DETERMINISTIC:
            return (
                SEVERITY_DISPATCH,
                "Scheduler.is_deterministic() (scheduler.py:2046) no longer "
                "routes this task to DeterministicRunner, the no-lock module "
                "path, or the restart stamp-clear",
            )
        return (SEVERITY_BENIGN, _BENIGN_TASK_KIND_REASON)
    return (
        SEVERITY_INFORMATIONAL,
        "provenance/bookkeeping key with no gating or dispatch consumer",
    )


# ---------------------------------------------------------------------------
# project_id resolution and the mismatch guard.
# ---------------------------------------------------------------------------


def resolve_project_id(project_root: str, override: str | None = None) -> str:
    """Derive the tickets.db ``project_id`` for *project_root*.

    ``/home/leo/src/dark-factory`` -> ``dark_factory``: the directory name with
    ``-`` mapped to ``_``. *override* wins outright when supplied, which is the
    escape hatch for a project whose directory name does not match its id.
    """
    if override:
        return override
    return Path(project_root).name.replace("-", "_")


def _distinct_ticket_project_ids(tickets_db_path: str) -> set[str]:
    """Every distinct ``project_id`` holding a created-row, read-only.

    Returns an empty set when the file is absent, so a caller cannot tell a
    missing store from an empty one by exception handling alone — that
    distinction is drawn by the guard below, deliberately.
    """
    if not Path(tickets_db_path).exists():
        return set()
    conn = sqlite3.connect(f"file:{tickets_db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute(
            "SELECT DISTINCT project_id FROM tickets WHERE status = ?",
            (TICKET_STATUS_CREATED,),
        )
        return {row[0] for row in cursor if row[0]}
    finally:
        conn.close()


def _project_id_mismatch(
    tickets_db_path: str,
    project_id: str,
    ticket_expectations: dict[str, dict],
) -> str | None:
    """Detect a mis-derived *project_id*, returning a message or ``None``.

    WHY THIS GUARD EXISTS. :func:`resolve_project_id` derives the id
    HEURISTICALLY from a directory name. A wrong derivation yields zero ticket
    expectations, which makes every combine target look totally wiped and
    produces a wall of confident FALSE POSITIVES an operator would read as a
    real incident. Silently reporting that would be exactly the
    no-silent-fail-soft violation named in
    docs/legibility/design-invariants.md, so the mismatch is stated loudly and
    names both the resolved id and the ids actually present.

    Deliberately INERT when the tickets.db is absent, or holds no created-rows
    at all: those are honest "no comparison source", not misconfiguration, and
    firing there would cry wolf on every fresh project.
    """
    if ticket_expectations:
        return None
    present = _distinct_ticket_project_ids(tickets_db_path)
    if not present or project_id in present:
        return None
    return (
        f"PROJECT_ID MISMATCH: resolved project_id={project_id!r} matches ZERO "
        f"created tickets, but the ticket store holds created-rows for "
        f"{sorted(present)}. Every finding below is therefore unevidenced and "
        f"probably a FALSE POSITIVE -- re-run with --project-id set correctly."
    )


# ---------------------------------------------------------------------------
# The audit itself.
# ---------------------------------------------------------------------------

# Where a lost key's expectation came from. 'none' is a FIRST-CLASS value, not
# an absence: a combine target no source can speak for is still reported, so a
# partial sweep can never read as complete.
SOURCE_TICKET = "ticket"
SOURCE_MANIFEST = "manifest"
SOURCE_NONE = "none"

# Statuses meaning the task is finished and will not be dispatched again.
#
# MEASURED SPLIT that validates this constant exactly (live store, 2026-08-03,
# 91 combine targets): done=63 + cancelled=4 = the 67 terminal targets that
# were deliberately left un-remediated; pending=21 + in-progress=2 + blocked=1
# = the 24 live victims that were hand-remediated the same day. Non-terminal
# is NECESSARY but not SUFFICIENT for a finding to drive the exit code — see
# :func:`is_actionable`, which also requires a consumer-bearing severity. If
# the terminal backlog drove the exit, this detector would be permanently red
# and therefore ignored, destroying its value as a re-runnable regression
# check. Terminal findings are suppressed from the EXIT CODE, never from the
# report.
TERMINAL_STATUSES = ("done", "cancelled")

# The pseudo-key used for a target no comparison source can speak for. It is
# reported as a finding rather than dropped, because "we could not check this
# task" and "this task is fine" are opposite outcomes.
NO_COMPARISON_SOURCE_KEY = "(no comparison source)"

_NO_COMPARISON_SOURCE_REASON = (
    "no created ticket and no capability manifest binds this task, so its "
    "pre-combine metadata cannot be reconstructed at all; this row asserts "
    "UNKNOWN, neither clean nor damaged"
)


class Finding(NamedTuple):
    """One metadata key a comparison source expected but live metadata lacks.

    ``expected_source`` is per-KEY, not per-task: a single task can carry a
    manifest-sourced ``delivered_checks`` finding beside a ticket-sourced
    ``source`` finding, and the two claims rest on different evidence.

    ``terminal`` mirrors :data:`TERMINAL_STATUSES` and is what keeps the
    long-lived terminal backlog out of the exit code.
    """

    tag: str
    task_id: int
    status: str
    key: str
    severity: str
    expected_source: str
    terminal: bool
    reason: str


def is_actionable(finding: Finding) -> bool:
    """Whether *finding* should drive :data:`EXIT_LIVE_FINDINGS`.

    THREE independent exclusions, each for a different reason:

    * ``terminal`` — done/cancelled tasks will never be dispatched again, so
      the loss can no longer change any behaviour. The ~67-row terminal
      backlog was deliberately left un-remediated; letting it set the exit
      code would pin this detector permanently red.
    * severity not in :data:`_ACTIONABLE_SEVERITIES` — informational and
      benign rows name no consumer that lost anything it acts on.
    * :data:`NO_COMPARISON_SOURCE_KEY` — that row asserts UNKNOWN, *neither
      clean nor damaged*. Exiting non-zero on it would report "we could not
      check this task" as "this task is damaged", which is a distinct and
      louder claim than the evidence supports.

    Excluded findings are still printed in full and still counted in the
    COVERAGE block; only the exit code ignores them.
    """
    return (
        not finding.terminal
        and finding.key != NO_COMPARISON_SOURCE_KEY
        and finding.severity in _ACTIONABLE_SEVERITIES
    )


class AuditCoverage(NamedTuple):
    """How much of the combine population the audit could actually see.

    ALWAYS reported, including when zero findings are produced. Measured
    live: 25 of 91 combine targets have no created ticket, so the finding
    list is an OBSERVABLE SUBSET and never "the damaged population".
    Presenting it as complete would be a no-silent-fail-soft violation
    (docs/legibility/design-invariants.md).

    ``manifest_parse_failure_details`` carries the ``"<path>: <error>"``
    strings themselves, not just the count: an operator told only that "2
    manifests failed to parse" cannot find out WHICH, which would swallow the
    failures at exactly the reporting boundary no-silent-fail-soft is about.
    ``ambiguous_manifest_bindings`` counts combine targets whose manifest
    binding was contested (see :func:`_expected_from_manifest`). Both fields
    are defaulted so they are purely additive to positional construction.
    """

    total_combine_targets: int
    targets_with_ticket: int
    targets_with_manifest: int
    targets_without_comparison_source: int
    manifest_parse_failures: int
    manifest_parse_failure_details: tuple[str, ...] = ()
    ambiguous_manifest_bindings: int = 0


class ProjectAudit(NamedTuple):
    """One project's audit result: what was found, and what could be seen."""

    project_root: str
    project_id: str
    findings: list[Finding]
    coverage: AuditCoverage
    # Non-None means the resolved project_id matched NOTHING in a tickets.db
    # that plainly holds created-rows for other projects — see
    # _project_id_mismatch. Defaulted so the field is purely additive.
    project_id_mismatch: str | None = None


def _finding_sort_key(finding: Finding) -> tuple[int, int, str, str]:
    """Gate-removing first, then NUMERIC task id so 100 follows 20.

    Numeric-safe with the precedent's try/except fallback
    (``_candidate_sort_key``): a non-numeric task id sorts under 0 by its
    string form rather than aborting the sort mid-report.
    """
    rank = _severity_rank(finding.severity)
    try:
        return (rank, int(finding.task_id), "", finding.key)
    except (TypeError, ValueError):
        return (rank, 0, str(finding.task_id), finding.key)


def _expected_from_manifest(expectation: ManifestExpectation) -> dict[str, object]:
    """The metadata keys a manifest binding implies, for the union below.

    ``delivered_checks`` is contributed ONLY when the manifest binds at least
    one mechanical check, because ``commit_planning`` skips the stamp entirely
    when the mechanical list is empty (manifest_stamping.py's
    ``if not mechanical:`` guard). Contributing it unconditionally would
    manufacture a GATE-REMOVING false positive for every manual-only manifest.

    AN AMBIGUOUS BINDING ALSO WITHHOLDS ``delivered_checks``. When two
    manifests bind one task_id, the check names here come from whichever
    manifest sorted first — i.e. they may belong to the OTHER PRD entirely.
    Claiming a lost mark-done gate on that basis would emit the single
    highest-severity row, and drive exit 1, off a coin flip. The contested
    binding is instead surfaced as provenance (``prd_path`` /
    ``prd_task_label``, which the reason string annotates with every binder)
    and counted in :attr:`AuditCoverage.ambiguous_manifest_bindings`, so the
    collision is visible without being asserted as damage.
    """
    expected: dict[str, object] = {
        "prd_path": expectation.prd_path,
        "prd_task_label": expectation.label,
    }
    if expectation.delivered_check_names and not expectation.ambiguous:
        expected["delivered_checks"] = list(expectation.delivered_check_names)
    return expected


_AMBIGUOUS_BINDING_NOTE = (
    "CONTESTED MANIFEST BINDING -- this task_id is bound by more than one "
    "capability manifest ({binders}), so the manifest-sourced values above "
    "come from the first in sorted order and may belong to another PRD; "
    "delivered_checks is withheld entirely rather than asserted"
)


def audit_project(project_root: str, project_id: str | None = None) -> ProjectAudit:
    """Audit one project root for curator-combine metadata loss.

    Reads the task store, the ticket store and every capability manifest, all
    READ-ONLY, unions the two comparison sources with PER-KEY provenance, and
    reports each key a source expected but live metadata no longer carries.

    Only ``expected_keys - live_keys`` is ever claimed — never value drift.
    The ticket is a SUBMIT-TIME snapshot, so a key legitimately added between
    submit and combine is invisible to it and comparing values would surface
    those benign post-submit edits as findings.

    Where both sources speak, the TICKET wins shared keys (it is the closer
    record of what the task actually held) except ``delivered_checks``, which
    the manifest owns — a submit payload rarely carries it, since
    ``commit_planning`` stamps it after the fact.

    A missing tickets.db or an unreadable manifest degrades to the remaining
    source rather than aborting; a target no source covers is still emitted,
    carrying :data:`SOURCE_NONE`.
    """
    project_id = resolve_project_id(project_root, override=project_id)
    targets = load_combine_targets(str(tasks_db_path(project_root)))
    ticket_expectations = load_ticket_expectations(
        str(tickets_db_path(project_root)), project_id
    )
    parse_failures: list[str] = []
    manifest_index = build_manifest_index(project_root, parse_failures=parse_failures)

    findings: list[Finding] = []
    with_ticket = 0
    with_manifest = 0
    without_source = 0
    ambiguous_bindings = 0

    for target in targets.values():
        key_id = str(target.task_id)
        ticket = ticket_expectations.get(key_id)
        manifest = manifest_index.get(key_id)
        if ticket is not None:
            with_ticket += 1
        if manifest is not None:
            with_manifest += 1

        terminal = target.status in TERMINAL_STATUSES
        if ticket is None and manifest is None:
            without_source += 1
            findings.append(
                Finding(
                    tag=target.tag,
                    task_id=target.task_id,
                    status=target.status,
                    key=NO_COMPARISON_SOURCE_KEY,
                    severity=SEVERITY_INFORMATIONAL,
                    expected_source=SOURCE_NONE,
                    terminal=terminal,
                    reason=_NO_COMPARISON_SOURCE_REASON,
                )
            )
            continue

        # Union with per-key provenance. Manifest first, then the ticket
        # overwrites the keys it also speaks for, then delivered_checks is
        # restored to the manifest — see this function's docstring.
        #
        # The restore reads back from `manifest_expected` rather than
        # re-deriving from `manifest.delivered_check_names`, so
        # _expected_from_manifest stays the SINGLE place that decides whether
        # the manifest speaks for delivered_checks at all. Re-deriving here
        # silently bypassed both of that helper's guards (the empty-mechanical
        # one and the contested-binding one) and re-introduced exactly the
        # gate_removing false positive they exist to prevent.
        manifest_expected = (
            _expected_from_manifest(manifest) if manifest is not None else {}
        )
        expected: dict[str, object] = {}
        source_of: dict[str, str] = {}
        for key, value in manifest_expected.items():
            expected[key] = value
            source_of[key] = SOURCE_MANIFEST
        if ticket is not None:
            for key, value in ticket.items():
                expected[key] = value
                source_of[key] = SOURCE_TICKET
        if _GATE_KEY in manifest_expected:
            expected[_GATE_KEY] = manifest_expected[_GATE_KEY]
            source_of[_GATE_KEY] = SOURCE_MANIFEST

        # A contested binding annotates every MANIFEST-sourced reason with the
        # binders, so the collision travels with the row an operator reads
        # rather than being visible only as an aggregate count.
        ambiguous_note = ""
        if manifest is not None and manifest.ambiguous:
            ambiguous_bindings += 1
            ambiguous_note = _AMBIGUOUS_BINDING_NOTE.format(
                binders=", ".join(sorted(manifest.bound_by))
            )

        live = set(target.metadata_keys)
        for key in expected.keys() - live:
            severity, reason = classify_gap(key, expected[key])
            if ambiguous_note and source_of[key] == SOURCE_MANIFEST:
                reason = f"{reason}; {ambiguous_note}"
            findings.append(
                Finding(
                    tag=target.tag,
                    task_id=target.task_id,
                    status=target.status,
                    key=key,
                    severity=severity,
                    expected_source=source_of[key],
                    terminal=terminal,
                    reason=reason,
                )
            )

    findings.sort(key=_finding_sort_key)
    return ProjectAudit(
        project_root=project_root,
        project_id=project_id,
        findings=findings,
        project_id_mismatch=_project_id_mismatch(
            str(tickets_db_path(project_root)), project_id, ticket_expectations
        ),
        coverage=AuditCoverage(
            total_combine_targets=len(targets),
            targets_with_ticket=with_ticket,
            targets_with_manifest=with_manifest,
            targets_without_comparison_source=without_source,
            manifest_parse_failures=len(parse_failures),
            manifest_parse_failure_details=tuple(parse_failures),
            ambiguous_manifest_bindings=ambiguous_bindings,
        ),
    )


# ---------------------------------------------------------------------------
# Reporting.
#
# Both caveats are SINGLE module-level constants rather than inline prose, so
# the tests pin the constant and an editor improving the wording changes one
# place instead of drifting two copies apart.
# ---------------------------------------------------------------------------

_FIDELITY_CAVEAT = (
    "  FIDELITY: findings are TICKET-EVIDENCED, not certain. The creating "
    "ticket is a SUBMIT-TIME snapshot, so it cannot see a key legitimately "
    "added between submit and combine; only key LOSS is claimed, never value "
    "drift. Measured agreement over 400 never-combined dark_factory tasks: "
    "367/368 on task_kind and 368/368 on source, the single task_kind "
    "mismatch (task 3295) proving the field IS mutable post-creation. "
    "Manifest-sourced rows rest on the capability sidecar instead, and are "
    "labelled source=manifest -- they carry their OWN assumption: the sidecar "
    "is stamped with task_id in manifest_stamping.py step 4, which COMMITS TO "
    "DISK BEFORE step 5 copies delivered_checks onto the task, and step 5's "
    "per-label body is individually guarded (it logs a warning and continues "
    "on an exception, and again when update_task returns a structured "
    "rejection). A label whose step-5 write failed therefore leaves a stamped "
    "sidecar beside a task that NEVER carried delivered_checks, which is "
    "indistinguishable here from a combine wipe. Before treating a "
    "source=manifest delivered_checks row as combine damage, grep the "
    "fused-memory log for 'failed to build/write delivered_checks' or "
    "'rejected delivered_checks write' naming that label."
)

_COVERAGE_CAVEAT = (
    "  COVERAGE (the findings above are an OBSERVABLE SUBSET, not the full "
    "damaged population - a combine target with no comparison source is "
    "UNKNOWN, neither clean nor damaged):"
)


def _format_finding_line(finding: Finding) -> str:
    """One finding, in the precedent's ``key=value`` style."""
    return (
        f"  task_id={finding.task_id} tag={finding.tag} "
        f"status={finding.status} key={finding.key} "
        f"severity={finding.severity} source={finding.expected_source}"
    )


def _format_coverage(coverage: AuditCoverage) -> list[str]:
    """Render the always-printed coverage block.

    Never omitted and never abbreviated when there are no findings: the whole
    point is that the finding list is an observable subset, and a reader must
    be told the size of the unobservable remainder.
    """
    lines = [
        _COVERAGE_CAVEAT,
        f"    combine targets scanned:            {coverage.total_combine_targets}",
        f"    with a creating ticket:             {coverage.targets_with_ticket}",
        f"    with a capability manifest:         {coverage.targets_with_manifest}",
        f"    with NO comparison source:          {coverage.targets_without_comparison_source}",
        f"    contested manifest bindings:        {coverage.ambiguous_manifest_bindings}",
        f"    manifests that failed to parse:     {coverage.manifest_parse_failures}",
    ]
    # NAME the unreadable sidecars, never just count them. A count alone tells
    # an operator that coverage is incomplete but not where to look, which
    # swallows the failure at the reporting boundary (no-silent-fail-soft).
    for detail in coverage.manifest_parse_failure_details:
        lines.append(f"      - {detail}")
    return lines


def _format_section(label: str, findings: list[Finding]) -> list[str]:
    lines = [f"  -- {label} ({len(findings)}) --"]
    for finding in findings:
        lines.append(_format_finding_line(finding))
        lines.append(f"      reason: {finding.reason}")
    return lines


def format_report(audits: list[ProjectAudit]) -> str:
    """Render *audits* as a grouped, human-readable report.

    LIVE findings render BEFORE terminal ones: those are the rows an operator
    can still act on. Terminal findings are printed IN FULL regardless — they
    are suppressed from the exit code, never from the report.

    NOTE that "live" and "exit-code-driving" are NOT the same set: a live row
    can still be informational, benign, or '(no comparison source)', none of
    which turn the exit red. The trailing summary line therefore prints the
    ACTIONABLE count alongside the live count (see :func:`is_actionable`).

    Both the FIDELITY and COVERAGE blocks are emitted for every project,
    INCLUDING projects with zero findings.

    Pure: returns the text and never prints. ``main()`` does the single print.
    """
    lines: list[str] = []
    total_live = 0
    total_terminal = 0
    total_actionable = 0
    for audit in audits:
        lines.append(f"{audit.project_root} (project_id={audit.project_id}):")
        if audit.project_id_mismatch:
            # ABOVE the findings on purpose: a reader must see that the rows
            # below are unevidenced before reading a single one of them.
            lines.append(f"  {audit.project_id_mismatch}")

        live = [f for f in audit.findings if not f.terminal]
        terminal = [f for f in audit.findings if f.terminal]
        total_live += len(live)
        total_terminal += len(terminal)

        total_actionable += sum(1 for f in audit.findings if is_actionable(f))

        lines.extend(_format_section("live findings", live))
        lines.extend(_format_section("terminal findings (reported, but not "
                                     "counted toward the exit code)", terminal))
        lines.append(_FIDELITY_CAVEAT)
        lines.extend(_format_coverage(audit.coverage))

    # The actionable count is spelled out because it, NOT the live count, is
    # what the exit code keys on: a live row that is informational, benign, or
    # '(no comparison source)' exits 0. Printing only "N live" beside a 0 exit
    # would read as a contradiction.
    lines.append(
        f"{total_live + total_terminal} finding(s) across {len(audits)} "
        f"project(s): {total_live} live ({total_actionable} actionable, "
        f"i.e. exit-code-driving), {total_terminal} terminal"
    )
    return "\n".join(lines)


def format_json(audits: list[ProjectAudit]) -> str:
    """Render *audits* as a JSON OBJECT carrying findings AND coverage.

    An object rather than a bare array so the coverage counts and the fidelity
    caveat travel with the findings — a consumer that received only an array
    could not tell a clean project from an unobservable one.

    Pure: returns the text and never prints.
    """
    return json.dumps(
        {
            "fidelity": _FIDELITY_CAVEAT,
            "projects": [
                {
                    "project_root": audit.project_root,
                    "project_id": audit.project_id,
                    "project_id_mismatch": audit.project_id_mismatch,
                    "coverage": audit.coverage._asdict(),
                    "findings": [f._asdict() for f in audit.findings],
                }
                for audit in audits
            ],
        }
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

# Named rather than spelled inline so the epilog, main()'s docstring and the
# returns can never drift into disagreeing about what a number means. Each
# denotes EXACTLY ONE outcome -- that is the whole reason 3 exists rather than
# being folded into 0.
#
# The per-script NAMES survive because the epilog wording is per-script (this
# one is about ACTIONABLE findings and terminal-row suppression; wiped's is
# about candidates), but since task 3616 the VALUES have ONE home: the returns
# now live in _task_db_scan.run_audit_cli, so a local re-spelling would drift
# from what actually gets returned. test_exit_constants_alias_the_shared_
# tier_3_codes is what keeps these aliases honest.
EXIT_OK = AUDIT_EXIT_OK                       # audited; no ACTIONABLE findings
                                              # and no id mismatch
EXIT_LIVE_FINDINGS = AUDIT_EXIT_FINDINGS      # an actionable finding
                                              # (is_actionable), or the
                                              # project_id guard fired
EXIT_NO_ROOT = AUDIT_EXIT_NO_ROOT             # no project root resolved to a
                                              # readable tasks.db
EXIT_NOTHING_SCANNED = AUDIT_EXIT_NOTHING_AUDITED  # roots resolved but EVERY
                                                   # one failed to audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "READ-ONLY audit of curator-combine metadata loss: reports every "
            "task bearing the combine signature whose pre-combine metadata "
            "keys can be shown missing against an independent source (its "
            "creating ticket, or a capability manifest). Reporting only -- "
            "never mutates a task, ticket, or manifest. Remediation is a "
            "separate, individually-reviewed follow-up."
        ),
        epilog=(
            "exit codes: 0 = audited, no ACTIONABLE findings; 1 = at least "
            "one actionable finding -- a NON-TERMINAL loss whose severity is "
            "gate_removing, provenance or dispatch -- or the project-id "
            "mismatch guard fired; 2 = no project root resolved to a readable "
            "tasks.db; 3 = roots resolved but every one failed to audit, so "
            "NOTHING was scanned (never treat 3 as a clean run). Terminal "
            "(done/cancelled) findings, informational/benign findings, and "
            "'(no comparison source)' rows are always reported in full but "
            "never affect the exit code."
        ),
    )
    parser.add_argument(
        "--project-root", dest="project_roots", action="append",
        help=(
            "Project root to audit (resolves <root>/.taskmaster/tasks/tasks.db, "
            "<root>/data/reconciliation/tickets.db, <root>/plans and "
            "<root>/docs/prds). May be repeated."
        ),
    )
    parser.add_argument(
        "--project-id", dest="project_id", default=None,
        help=(
            "Override the tickets.db project_id (default: %(default)s, i.e. "
            "derived from the project root's directory name with '-' mapped "
            "to '_'). A wrong id matches zero tickets and would make every "
            "combine target look wiped, so a mismatch is reported loudly. "
            "INTENDED FOR SINGLE-ROOT RUNS: this is ONE id applied to EVERY "
            "resolved root, and a project_id is inherently per-project, so "
            "combining it with several roots is warned about on stderr."
        ),
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit a JSON object (findings plus coverage) instead of a report.",
    )
    return parser


def _audit_root(root: str, args: argparse.Namespace) -> ProjectAudit:
    """Audit ONE project root under the (possibly overridden) project_id.

    Raises ``sqlite3.Error`` for an unreadable project, which is what
    :func:`_task_db_scan.sweep_project_roots` turns into a warn-and-skip; every
    other exception propagates. Returns exactly one audit, per that function's
    one-audit-per-root contract.
    """
    return audit_project(root, args.project_id)


def _render(audits: list[ProjectAudit], args: argparse.Namespace) -> str:
    return format_json(audits) if args.json else format_report(audits)


def _is_dirty(audits: list[ProjectAudit]) -> bool:
    """Exit 1 keys on ACTIONABLE findings, or a fired project-id guard.

    See :func:`main`'s docstring for why the excluded rows are excluded -- they
    are still printed in full and still counted in COVERAGE.
    """
    actionable = any(is_actionable(f) for a in audits for f in a.findings)
    mismatch = any(a.project_id_mismatch for a in audits)
    return actionable or mismatch


def _warn_project_id_across_roots(roots: list[str], args: argparse.Namespace) -> None:
    """Warn once, up front, that ONE --project-id is being applied to MANY roots.

    Runs on the RESOLVED root list, before the empty-roots exit-2 return -- the
    position it has always occupied, preserved verbatim through the Tier-3
    extraction (task 3616). It fits neither the per-root audit callback nor the
    post-sweep renderer, which is why run_audit_cli takes an on_roots hook.
    """
    if len(roots) > 1 and args.project_id:
        print(
            f"warning: --project-id {args.project_id!r} is applied to ALL "
            f"{len(roots)} resolved project roots, but a project_id is "
            "per-project; any root whose real id differs will be audited "
            "against the wrong tickets.db rows (and a root with no created "
            "tickets will silently report '(no comparison source)' instead of "
            "flagging the mismatch). Prefer one --project-id run per root.",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Exit codes: 0 = audited, no ACTIONABLE findings; 1 = at least one
    actionable finding (see :func:`is_actionable`) or a fired project-id
    mismatch guard; 2 = no project root resolved to a readable tasks.db;
    3 = roots resolved but EVERY one failed to audit, so NOTHING was scanned.

    3 exists because 0 would otherwise be returned for two opposite outcomes --
    "audited everything, found nothing" and "audited nothing at all" -- and a
    CI/cron consumer reading only the exit code would take a total failure for
    a clean run. That is exactly the silent fail-soft this module refuses to do
    (see the module docstring).

    1 keys on ACTIONABLE findings only -- non-terminal AND severity in
    {gate_removing, provenance, dispatch} AND not the '(no comparison source)'
    pseudo-row. Everything excluded is excluded for the SAME reason the ~67
    terminal combine targets are: a row that names no consumer which lost
    something it acts on cannot be acted upon, and if such rows drove the exit
    code this detector would be permanently red and therefore ignored,
    destroying its value as a re-runnable regression check. Measured on the
    live store (2026-08-03) that was not hypothetical: 9 non-terminal rows,
    zero of them gate_removing or provenance. Every excluded row is still
    printed in full and still counted in COVERAGE.

    A single unreadable project (a corrupt/locked tasks.db) does NOT abort the
    sweep: it is logged to stderr and skipped so every other project is still
    audited, and a trailing warning states that the results are incomplete.

    ``--project-id`` is ONE id applied to EVERY resolved root, but a
    project_id is inherently per-project, so pairing it with several roots
    queries each root's tickets.db for another project's rows.
    :func:`_project_id_mismatch` catches the loud version of that (an id
    matching zero created-rows in a store that plainly holds others), but it
    is deliberately INERT for a root with no tickets.db or no created-rows at
    all -- there the misuse would otherwise degrade silently into "(no
    comparison source)" rows that look like honest missing evidence. So the
    combination is warned about on stderr up front. Warned, not rejected: the
    override is legitimate for a multi-root sweep over same-id checkouts (a
    main checkout plus its worktrees), which is a real use, and this script
    reports rather than gatekeeps.

    The roots loop, the warn-and-continue skip and the exit-code ladder are
    :func:`_task_db_scan.run_audit_cli` (Tier 3, task 3616), shared with
    audit_wiped_metadata_files.py. What stays here is what genuinely differs:
    this script's parser and epilog, its ``--project-id`` handling
    (:func:`_audit_root` and :func:`_warn_project_id_across_roots`), its
    object-shaped JSON, its report and its actionable-findings predicate
    (:func:`_is_dirty`). Nothing in this file returns a bare integer any more.
    """
    return run_audit_cli(
        argv,
        parser=_build_parser(),
        audit_fn=_audit_root,
        render=_render,
        is_dirty=_is_dirty,
        on_roots=_warn_project_id_across_roots,
    )


if __name__ == "__main__":
    sys.exit(main())
