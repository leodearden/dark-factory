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
(fused-memory/src/fused_memory/server/task_interceptor.py:2100) writes exactly
``{'curator_action': 'combine', 'curator_justification', 'combined_at'}`` and
passes ``metadata_mode='replace'``, so every OTHER key the surviving task
carried is DROPPED rather than merged. The load-bearing casualty is
``metadata.delivered_checks``: orchestrator/src/orchestrator/delivered_checks.py
defines both ``gate_mark_done_on_delivered_checks`` and
``verify_delivered_checks_on_main`` off that key, so wiping it SILENTLY REMOVES
a mark-done gate — the task still closes, just without the check that was
supposed to hold it. ``prd_path``/``prd_task_label`` (provenance) and
``task_kind`` (dispatch) are the other consumers. This script enumerates every
task bearing the combine signature whose pre-combine metadata can be
reconstructed from an independent source, and reports the keys that are gone.

WHAT A LIVE HIT MEANS, MEASURED RATHER THAN ASSUMED. As of main tip
9cec63e10e (2026-08-03) the combine path STILL reads ``metadata_mode='replace'``
(task_interceptor.py:2130); task 3446's fix is unmerged, on branch ``task/3446``
(commits 556e720c3d, 8750bbaaaa). So a NON-TERMINAL finding from this script
means the fix HAS NOT LANDED YET — not that it regressed. Once 3446 merges,
that reading inverts and a fresh live hit becomes a regression signal.

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
from _task_db_scan import discover_project_roots, tasks_db_path  # noqa: F401

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
            "WHERE project_id = ? AND status = ?",
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
# ambiguous case is still detected and flagged rather than silently first-wins.
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
