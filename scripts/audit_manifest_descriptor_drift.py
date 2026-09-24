#!/usr/bin/env python3
"""Audit capability-manifest sidecars against the task store, in two directions:
``delivered_check`` descriptors that have drifted from their producer task's
``metadata.delivered_checks`` entry (sidecar -> task), and task labels their
sidecar does not declare (task -> sidecar).

READ-ONLY / REPORT-ONLY, in both directions: this module and its CLI never
mutate a task record or a manifest file. Every database connection it opens is
a read-only SQLite URI (``sqlite3.connect(f"file:{path}?mode=ro", uri=True)``),
so the sweep is structurally incapable of writing to the live WAL database the
running orchestrator holds open. Manifest YAML on disk is only ever read. There
is no ``--apply`` flag and no MCP client is ever constructed. RESYNCING A
DRIFTED SIDECAR, OR FIXING AN UNBOUND LABEL, IS A SEPARATE, REVIEWED EDIT —
never done from this report by this script.

THE DRIFT DIRECTION (task 4545). ``metadata.delivered_checks`` is copied
exactly ONE WAY, sidecar -> task record, at ``commit_planning``
(fused-memory/src/fused_memory/server/manifest_stamping.py step 5, the
``for task in doc.tasks:`` block that builds a ``DeliveredCheckMeta`` per
mechanical capability and writes the list through ``update_task``). NOTHING
EVER SYNCS BACK. So when a delivered_check is repaired by hand on the task
record — because the sidecar's pattern was wrong, or matched the wrong file, or
had its ``expect`` inverted — the sidecar keeps the stale spelling, and the two
descriptors silently disagree from then on.

THAT IS A REGENERATION HAZARD, NOT A LIVE BLOCK, and the distinction is the
whole reason this is an audit rather than a gate. The δ gate
(orchestrator/src/orchestrator/delivered_checks.py) reads the TASK METADATA,
which is already the repaired copy — so no dependent is blocked today. The
exposure is that a re-decompose of the same PRD re-runs the stamper and
re-writes the STALE sidecar spelling over the repair, silently reverting it.
This sweep is the regression check that keeps the two spellings in agreement so
that re-stamp is a no-op.

CORRECTING A DESCRIPTOR FORCES RE-EVALUATION WITHOUT OPERATOR ACTION:
orchestrator/src/orchestrator/scheduler.py::_delivered_checks_descriptor_digest
folds the whole-list digest into the delivered-checks cache key, so an edit at a
fixed main sha is a cache MISS by design. Nothing here has to invalidate
anything.

THERE IS NO COMMIT-ORDERING PREMISE HERE. An earlier framing of this defect
supposed the drift came from sidecars committed before/after their task records;
that half was FALSIFIED and lives on in task 3500. This sweep makes no claim
about when either side was written — only that the two spellings, as they stand
right now, disagree.

AND IT DOES NOT JUDGE WHETHER A CHECK PASSES. A row here means the two
descriptors DISAGREE, nothing more. Both spellings may fail (the transcript
preservation seam's gz-consumer row is exactly that: correct-and-superseded,
task 3578 restored gzip reading after 3618 removed it), both may pass, or one of
each. Whether a check is satisfied on main is the δ gate's question and
``verify_delivered_checks_on_main``'s, not this script's.

THE LABEL-BINDING DIRECTION (task 4907). The drift walk is keyed on each
sidecar's STAMPED task_id, so it cannot see a task whose ``prd_task_label``
matches no sidecar entry at all. ``commit_planning``'s stamper binds nothing
for such a task and copies it no ``delivered_checks``. It does name the label,
in the ``missing_labels`` of the ``manifest_stamping`` report that
``commit_planning`` returns (manifest_stamping.py step 4b), but only once, at
planning time, for that batch only, and nothing re-surfaces it afterwards. This
direction is the retrospective sweep over the whole store: it walks from every
task the stamper would admit to the sidecar it would open, and reports each
label that tracked sidecar does not declare as an UNBOUND LABEL, in its own
list. It came out of the task-4590 stamp-coverage audit and task 4907's
adjudication of the rows that audit left.

AN UNBOUND LABEL IS NOT BY ITSELF A DEFECT. A sidecar is scoped, not a per-task
registry (plans/capability-delivered-checks-prd.md §"Sketch of approach",
"Coverage caveat (scope)"), and may legitimately omit a label. The adjudicated
cases are recorded in the ``.capability-manifest.md`` twins, each under
"Unbound task labels (task 4907 adjudication)". Only a LIVE row, whose task is
not yet done or cancelled, makes a run dirty. Fixing one is a separate,
reviewed edit; which edit is stated once, in ``_UNBOUND_CAVEAT``, which the
report and the JSON both carry.
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

# Tier 1 (tasks.db discovery) and Tier 3 (audit-script CLI plumbing: the roots
# loop, the warn-and-continue skip, the exit-code ladder and the two reporting
# layout primitives), imported as a flat sibling and shared with
# audit_combine_gate_marker_loss.py / audit_wiped_metadata_files.py. Tier 2 is
# the LEAK-SCANNER skeleton and does not apply here — it sweeps db paths and
# accumulates matches, not one audit per project root.
#
# IMPORT-RESOLUTION CONTRACT: _task_db_scan.py must stay a flat sibling in
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
# copy of this script running from a worktree would validate manifests using
# the MAIN checkout's schema — and this script is EXPECTED to run from a
# worktree (that is what --manifest-root is for). Same reasoning and same form
# as audit_combine_gate_marker_loss.py (tasks 2881/2882/3329). The shared.*
# imports below MUST stay after this insert.
_SHARED_SRC = Path(__file__).resolve().parent.parent / "shared" / "src"
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))

from shared.capability_manifest import (  # noqa: E402
    DeliveredCheckMeta,
    load_capability_manifest,
)
from shared.task_statuses import TERMINAL  # noqa: E402

# The kinds the stamper actually copies. manifest_stamping.py step 5 reads
# `if check is None or check.kind not in ('grep', 'script'): continue`, so a
# 'manual' capability never reaches metadata.delivered_checks at all and can
# never drift. Comparing one would emit a permanent false positive on every
# manual-checked capability in the corpus.
MECHANICAL_CHECK_KINDS = ("grep", "script")

# The sidecar filename suffix: what the stamper appends to a task's prd_path,
# and what `git ls-files` matches.
_MANIFEST_SUFFIX = ".capability-manifest.yaml"
_MANIFEST_GLOB = f"*{_MANIFEST_SUFFIX}"

_GIT_TIMEOUT_SECS = 30


class ManifestDiscoveryUnavailable(RuntimeError):
    """`git ls-files` could not enumerate the manifest corpus.

    Raised rather than degraded to an empty list on purpose: an empty corpus
    and a clean corpus are INDISTINGUISHABLE in the finding count, and only one
    of them is good news. Swallowing this would let a non-checkout, a broken
    git, or a permissions failure render as a confident zero
    (docs/legibility/design-invariants.md, no-silent-fail-soft).
    """


def _tracked_manifest_paths(manifest_root: str) -> list[str]:
    """Every TRACKED ``*.capability-manifest.yaml`` under *manifest_root*.

    Returns sorted, unique, repo-relative paths — repo-relative because that is
    what a reader can act on against a checkout, whereas an absolute path from
    somebody else's worktree is noise in a report.

    TRACKED rather than globbed, deliberately. The stamper only ever reads what
    is committed, and an untracked scratch copy of a sidecar in a working tree
    is not part of the corpus a re-decompose would re-stamp from. This also
    makes the sweep agree with shared/tests/test_capability_manifest.py's
    checked-in-corpus family, which is likewise git-driven.

    Raises :class:`ManifestDiscoveryUnavailable` on any failure to run git or a
    non-zero return — see that class for why this is never degraded to ``[]``.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(manifest_root), "ls-files", "-z", "--", _MANIFEST_GLOB],
            capture_output=True, text=True, timeout=_GIT_TIMEOUT_SECS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ManifestDiscoveryUnavailable(
            f"could not run `git ls-files` in {manifest_root}: {exc}"
        ) from exc

    if completed.returncode != 0:
        raise ManifestDiscoveryUnavailable(
            f"`git ls-files` failed in {manifest_root} (rc={completed.returncode}): "
            f"{completed.stderr.strip() or 'no stderr'}"
        )

    return sorted({p for p in completed.stdout.split("\0") if p})


def _decode_metadata(raw: object) -> dict:
    """Decode a raw ``metadata`` blob into a dict, degrading to ``{}``.

    Copied from :func:`audit_combine_gate_marker_loss._decode_metadata`.
    Degrades for NULL, an empty string, malformed JSON, or a payload that
    decodes to anything other than a dict (a list, a bare scalar, ``null``). A
    corrupt metadata blob is data to be skipped, never a reason to abort a
    sweep over thousands of tasks.
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


class ManifestBinding(NamedTuple):
    """A task the stamper would try to bind to a sidecar label.

    ``label`` is the task's ``prd_task_label``, verbatim. ``manifest`` is the
    sidecar path exactly as the stamper's step 1 DERIVES it from ``prd_path``
    (see :func:`_manifest_binding`): never read from anywhere, and not yet
    resolved against a project root, which is step 2
    (:func:`_contained_sidecar_relpath`).
    """

    task_id: int
    status: str
    label: str
    manifest: str


class TaskStoreScan(NamedTuple):
    """Everything this audit reads from one tasks.db, taken in ONE scan.

    ``row_ids`` and ``delivered_checks`` feed the drift direction, and they are
    separate because they answer two different coverage questions that must not
    be conflated: a manifest binding a task_id with NO ROW AT ALL is a
    stale/unstamped binding, while a task that exists but carries no same-named
    ``delivered_checks`` entry is an un-stamped or metadata-wiped gate.
    Collapsing them would misattribute 6 live rows to a population of 32 that a
    different audit already owns.

    A task absent from ``delivered_checks`` is a COVERAGE row, never a finding:
    the drift direction reports descriptors that DISAGREE, and a missing entry
    is not a disagreement — it is an absence, whose dominant live cause is the
    curator-combine ``metadata`` wipe that
    ``scripts/audit_combine_gate_marker_loss.py`` (tasks 3146/3329) owns.

    ``manifest_bindings`` feeds the label-binding direction: every task the
    stamper would admit, sorted by numeric task id.
    """

    row_ids: frozenset[int]
    delivered_checks: dict[int, dict[str, dict]]
    manifest_bindings: tuple[ManifestBinding, ...]


def _delivered_check_entries(metadata: dict) -> dict[str, dict]:
    """A task's ``metadata.delivered_checks`` keyed by capability name.

    An entry that is not a dict, or has no string ``name``, is dropped: it
    names no capability, so there is nothing to pair it with.
    """
    checks = metadata.get("delivered_checks")
    if not isinstance(checks, list):
        return {}
    return {
        entry["name"]: entry
        for entry in checks
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }


def _manifest_binding(task_id: int, status: str, metadata: dict) -> ManifestBinding | None:
    """The stamper's admission gate and sidecar derivation, in one place.

    Mirrors
    fused-memory/src/fused_memory/server/manifest_stamping.py::_stamp_capability_manifests_impl
    step 1 exactly, so the label-binding direction's population IS the
    stamper's. A task is admitted only when its metadata carries a non-empty
    ``prd_path`` AND ``prd_task_label`` (the stamper's falsy gate), and its
    sidecar is derived strictly as ``re.sub(r'\\.md$', '', prd_path)`` plus the
    sidecar suffix. It is never read from ``metadata.capability_manifest``,
    which the task-4590 investigation found pointing at the ``.md`` twin.

    ONE DELIBERATE DIVERGENCE: a truthy NON-STRING in either key is not
    admitted. The stamper would admit it, then fail to derive a path from a
    non-string ``prd_path`` or never match a non-string label against the
    sidecar's string labels — so such a row can be neither bound nor usefully
    reported. No live row has that shape.

    Step 2 resolves the derived path against the project root, which this
    function does not know; that is :func:`_contained_sidecar_relpath`, applied
    by the sweep.
    """
    prd_path = metadata.get("prd_path")
    label = metadata.get("prd_task_label")
    if not prd_path or not label:
        return None
    if not isinstance(prd_path, str) or not isinstance(label, str):
        return None
    return ManifestBinding(
        task_id=task_id,
        status=status,
        label=label,
        manifest=re.sub(r"\.md$", "", prd_path) + _MANIFEST_SUFFIX,
    )


def _contained_sidecar_relpath(project_root: str, derived: str) -> str | None:
    """The stamper's step 2 for one derived sidecar path: the root-relative
    POSIX path it resolves to, or ``None`` when the stamper would open nothing
    there.

    Resolved and containment-checked as
    fused-memory/src/fused_memory/server/manifest_stamping.py::_stamp_capability_manifests_impl
    does it, so a ``./``-prefixed, ``..``-bearing or absolute in-root
    ``prd_path`` reaches the sidecar the stamper would open, and one that
    resolves outside the root is refused, as the stamper refuses it. The stamper
    then checks that the file exists; the sweep instead looks the relpath up
    among the TRACKED sidecars (see :func:`_tracked_manifest_paths`).

    A path that cannot be resolved at all (an embedded NUL raises
    ``ValueError``) is ``None`` too, never an abort: the stamper opens nothing
    for it either, and one corrupt row must not end a sweep over thousands.
    """
    root = Path(project_root).resolve()
    try:
        resolved = (root / derived).resolve()
    except ValueError:
        return None
    if not resolved.is_relative_to(root):
        return None
    return resolved.relative_to(root).as_posix()


def load_task_store_scan(tasks_db_path: str) -> TaskStoreScan:
    """Read everything both directions need from tasks.db, in ONE scan.

    ONE SCAN, NOT ONE PER DIRECTION, for correctness rather than tidiness. The
    store is live and in WAL mode with the orchestrator writing to it, so a
    second connection would read a second snapshot, and the drift and
    label-binding halves of one report could straddle a write and contradict
    each other. Each row's metadata is decoded ONCE and feeds both
    ``delivered_checks`` and the manifest binding.

    ``tag`` is pinned to ``'master'`` because that is the tag the stamper writes
    under and the only tag the live store uses; the schema permits the same
    numeric id under a second tag, and a manifest's stamped ``task_id`` carries
    no tag, so an unpinned query would let an unrelated same-id row from another
    tag masquerade as the producer.

    Opens the database via a read-only URI (``mode=ro``) so the load is
    structurally incapable of mutating live task records even while fused-memory
    holds the same file open in WAL mode. Closed in a ``try/finally`` and never
    a ``with`` block — a sqlite3 ``with`` is a TRANSACTION, not a close.

    Its own open rather than :func:`_task_db_scan.connect_ro`, deliberately:
    that helper raises ``TaskDbUnreadable``, which is not a ``sqlite3.Error`` —
    the only exception :func:`_task_db_scan.sweep_project_roots` catches — so
    adopting it would turn a skipped unreadable project into an aborted sweep.
    """
    row_ids: set[int] = set()
    delivered_checks: dict[int, dict[str, dict]] = {}
    bindings: list[ManifestBinding] = []
    conn = sqlite3.connect(f"file:{tasks_db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute("SELECT id, status, metadata FROM tasks WHERE tag = 'master'")
        for task_id, status, metadata in cursor:
            try:
                tid = int(task_id)
            except (TypeError, ValueError):
                continue
            row_ids.add(tid)
            decoded = _decode_metadata(metadata)
            entries = _delivered_check_entries(decoded)
            if entries:
                delivered_checks[tid] = entries
            binding = _manifest_binding(tid, status, decoded)
            if binding is not None:
                bindings.append(binding)
    finally:
        conn.close()
    bindings.sort(key=lambda binding: binding.task_id)
    return TaskStoreScan(
        row_ids=frozenset(row_ids),
        delivered_checks=delivered_checks,
        manifest_bindings=tuple(bindings),
    )


class DescriptorDrift(NamedTuple):
    """One capability whose sidecar and task-record descriptors disagree.

    ``manifest`` is repo-relative; ``differing_fields`` names the normalized
    :class:`DeliveredCheckMeta` fields that differ, sorted; ``sidecar_check``
    and ``task_check`` carry BOTH full normalized descriptors so a reader never
    has to open two files to see what drifted.

    A NamedTuple rather than a dataclass, following the sibling audits' stated
    precedent: ``_asdict()`` feeds the JSON writer.
    """

    manifest: str
    task_id: int
    label: str
    capability: str
    differing_fields: tuple[str, ...]
    sidecar_check: dict
    task_check: dict


def _expected_meta(capability_name: str, check: object) -> dict:
    """The normalized descriptor a re-decompose WOULD stamp for this capability.

    Constructed field-for-field the way manifest_stamping.py step 5 constructs
    it, so the comparison is against exactly what the stamper would write —
    not against an approximation of it. Normalizing through
    :class:`DeliveredCheckMeta` on this side and the task side both is what
    makes an ABBREVIATED task entry (one omitting the defaulted ``script`` /
    ``args`` / ``timeout_secs`` keys) compare equal to a full one, which is the
    difference between the 8 real drift rows and 22 absent-vs-default artifacts.

    MAY RAISE, and the caller GUARDS it symmetrically with the task-record
    side. Today it cannot: a validated sidecar grep/script ``DeliveredCheck``
    shares ``_check_kind_conditional_fields`` with :class:`DeliveredCheckMeta`
    and ``ManifestCapability.name`` carries ``min_length=1``, so every
    conversion succeeds. But that is an IMPLICIT coupling between two models
    that are free to diverge, and an unguarded ``ValidationError`` here would
    escape :func:`audit_project` into
    :func:`_task_db_scan.sweep_project_roots`, which catches only
    ``sqlite3.Error`` — aborting every REMAINING project root over one bad
    sidecar. That is the exact fail-loud-but-fail-everything mode the
    ``git_discovery_failed`` handling was written to avoid, so this degrades to
    a coverage row instead.
    """
    return DeliveredCheckMeta(
        name=capability_name,
        kind=check.kind,  # type: ignore[union-attr]
        pattern=check.pattern,  # type: ignore[union-attr]
        expect=check.expect,  # type: ignore[union-attr]
        paths=check.paths,  # type: ignore[union-attr]
        script=check.script,  # type: ignore[union-attr]
        args=check.args,  # type: ignore[union-attr]
        timeout_secs=check.timeout_secs,  # type: ignore[union-attr]
    ).model_dump()


class AuditCoverage(NamedTuple):
    """How much of the corpus the sweep could actually compare.

    ALWAYS reported, including on a zero-finding sweep. The finding list is a
    comparison of MATCHED PAIRS only, and several classes never reach a
    comparison at all: a capability whose producer task carries no same-named
    entry, a manifest binding a task_id with no tasks.db row, a task-record
    entry that will not validate, a sidecar descriptor that will not convert,
    and a sidecar that would not parse. Presenting the finding list as the
    whole corpus would be a no-silent-fail-soft violation
    (docs/legibility/design-invariants.md).

    SEEN AND COMPARED ARE TWO DIFFERENT NUMBERS, and both are reported because
    conflating them overstates the one figure that states comparison VOLUME.
    ``mechanical_capabilities_seen`` counts every mechanical capability the
    sweep reached — the eligible population. ``mechanical_capabilities_compared``
    counts only those that actually reached a descriptor comparison, i.e. that
    paired with a task-record entry AND normalized on both sides. The
    difference is exactly accounted for by the four skip counters, so::

        seen == compared
               + capabilities_without_task_entry
               + malformed_task_entries
               + unconvertible_sidecar_descriptors

    A reader must never have to derive the true matched-pair count by
    subtracting other rows, in a report whose whole thesis is that the finding
    list is not the whole corpus.

    ``task_entries_with_no_sidecar_capability`` is the REVERSE direction, and
    it exists because the sidecar->task walk alone is blind to two real drift
    shapes: a hand-repair that RENAMED a capability on the task record (the old
    name lands in ``capabilities_without_task_entry``, a bucket this report
    attributes to a different owner, so genuine drift would be misfiled as
    somebody else's problem), and a sidecar capability changed grep->manual
    while a stale mechanical entry remains on the record (which a re-decompose
    would LEAVE in place, since manifest_stamping step 5 does
    ``if not mechanical: continue`` rather than clearing). A rename shows up as
    the two rows TOGETHER on the same task — that is its signature, and the
    details name the manifest so it can be read as one.

    ``manifest_parse_failure_details`` and ``uncomparable_details`` carry the
    strings themselves, not just counts: an operator told only that "2 manifests
    failed to parse" cannot find out WHICH, which swallows the failures at
    exactly the reporting boundary the rule is about.

    ``git_discovery_failed`` marks a run whose manifest corpus could not be
    enumerated at all — the one case where a zero finding count means nothing.
    Every field after the counts is defaulted so it is purely additive to
    positional construction.

    THE LABEL-BINDING DIRECTION CLOSES ITS OWN IDENTITY, for the same reason:
    ``ProjectAudit.unbound_labels`` must never read as the whole population it
    came from. ``manifest_bearing_tasks`` counts every task the stamper would
    admit, and each lands in exactly one class::

        manifest_bearing_tasks == tasks_bound_to_a_declared_label
                                  + len(ProjectAudit.unbound_labels)
                                  + tasks_without_a_tracked_sidecar
                                  + tasks_on_an_unparseable_sidecar

    A sidecar that is not in the tracked corpus at all (absent, untracked,
    outside the project root, or unresolvable) and one that is tracked but will
    not parse are separate terms, because only the second IS tracked. Both
    classes are COUNTED and never itemized in ``uncomparable_details``. The
    first is a stamper no-op that numbers in the hundreds live, and would bury
    the rows the report exists to show. The second's sidecar is already named in
    ``manifest_parse_failure_details``.
    """

    manifests_swept: int
    mechanical_capabilities_compared: int
    capabilities_without_task_entry: int
    manifest_tasks_without_db_row: int
    malformed_task_entries: int
    manifest_parse_failures: int
    mechanical_capabilities_seen: int = 0
    unconvertible_sidecar_descriptors: int = 0
    task_entries_with_no_sidecar_capability: int = 0
    manifest_parse_failure_details: tuple[str, ...] = ()
    uncomparable_details: tuple[str, ...] = ()
    git_discovery_failed: bool = False
    manifest_bearing_tasks: int = 0
    tasks_bound_to_a_declared_label: int = 0
    tasks_without_a_tracked_sidecar: int = 0
    tasks_on_an_unparseable_sidecar: int = 0


class UnboundLabel(NamedTuple):
    """A manifest-bearing task whose label its tracked sidecar does not declare.

    The stamper binds nothing for such a task and copies it no
    ``delivered_checks``. ``declared_labels`` are the labels the sidecar DOES
    declare, in the sidecar's own order, so a reader never has to open the file
    to see what the task's label failed to match.

    A NamedTuple, like :class:`DescriptorDrift`, because ``_asdict()`` feeds the
    JSON writer.
    """

    task_id: int
    label: str
    status: str
    manifest: str
    declared_labels: tuple[str, ...]

    @property
    def is_live(self) -> bool:
        """Whether a future ``commit_planning`` can still touch this task.

        The hazard an unbound label carries is that a planning batch touching
        its task stamps and copies nothing. A done or cancelled task can no
        longer be touched that way, so its row is historical: still REPORTED,
        but not dirty. A live row can still cost something, and while it is
        live its label is still cheap to fix.

        Defined as ``status not in shared.task_statuses.TERMINAL``, compared as
        the raw string tasks.db holds (``TaskStatus`` is a ``StrEnum``) and never
        converted with ``TaskStatus(status)``, which raises on an unknown value.
        So an unknown or empty status is live by construction: the check fails
        toward reporting, never toward a confident zero.

        A property is not a NamedTuple field, so ``_asdict()`` omits it; the JSON
        writer carries it explicitly.
        """
        return self.status not in TERMINAL


class ProjectAudit(NamedTuple):
    """One project's audit: what drifted, which labels bind nothing, and what
    could be seen.

    ``findings`` is the drift direction (sidecar -> task) and ``unbound_labels``
    the label-binding direction (task -> sidecar); they are separate lists
    because they are separate defects with separate remedies.

    ``project_root`` owns the tasks.db; ``manifest_root`` owns the sidecars.
    They are the SAME path by default and differ only under ``--manifest-root``
    — which exists because ``.taskmaster/`` is gitignored and lives only in the
    primary checkout, so a sweep of a task WORKTREE's sidecars must still read
    the primary checkout's task store. Both are recorded so a decoupled run is
    unambiguous in the report.
    """

    project_root: str
    manifest_root: str
    findings: list[DescriptorDrift]
    unbound_labels: list[UnboundLabel]
    coverage: AuditCoverage


def _drift_sort_key(drift: DescriptorDrift) -> tuple[str, int, str]:
    """Manifest path, then NUMERIC task id, then capability name.

    A report whose row order depends on filesystem or sqlite iteration order
    cannot be diffed between runs.
    """
    return (drift.manifest, drift.task_id, drift.capability)


def _unbound_sort_key(row: UnboundLabel) -> tuple[str, int]:
    """Manifest path, then NUMERIC task id — for the reason given on
    :func:`_drift_sort_key`."""
    return (row.manifest, row.task_id)


class LabelBindingSweep(NamedTuple):
    """The label-binding direction's rows, and the population they came from.

    The bound, no-tracked-sidecar and unparseable-sidecar counts, together with
    ``len(unbound_labels)``, partition ``manifest_bearing_tasks`` — see
    :class:`AuditCoverage`.
    """

    unbound_labels: list[UnboundLabel]
    manifest_bearing_tasks: int
    tasks_bound_to_a_declared_label: int
    tasks_without_a_tracked_sidecar: int
    tasks_on_an_unparseable_sidecar: int


def _sweep_label_bindings(
    bindings: tuple[ManifestBinding, ...],
    project_root: str,
    declared_by_manifest: dict[str, tuple[str, ...]],
    unparseable_manifests: set[str],
) -> LabelBindingSweep:
    """Classify every binding, and list those whose tracked, parsed sidecar
    does not declare their label.

    Mirrors the stamper's own matching in
    fused-memory/src/fused_memory/server/manifest_stamping.py::_stamp_capability_manifests_impl.
    Step 2 opens only a sidecar that resolves inside *project_root* and exists
    (:func:`_contained_sidecar_relpath`), so a task whose sidecar is not among
    the tracked ones is promised nothing and is never a row. Step 4 matches on
    LABEL, by exact string equality, so a declared label is bound whatever its
    entry's ``task_id`` says. A task on a sidecar that failed to parse is never
    a row either: that sidecar declares nothing KNOWN, which is not the same as
    declaring nothing, and the parse failure is already named in the coverage
    details.

    A row is not by itself a defect. What a row does and does not mean, and the
    reviewed edits that fix one, are stated once, in ``_UNBOUND_CAVEAT``.
    """
    rows: list[UnboundLabel] = []
    bound = without_tracked_sidecar = on_unparseable_sidecar = 0
    for binding in bindings:
        manifest = _contained_sidecar_relpath(project_root, binding.manifest)
        if manifest in unparseable_manifests:
            on_unparseable_sidecar += 1
            continue
        declared = None if manifest is None else declared_by_manifest.get(manifest)
        if manifest is None or declared is None:
            without_tracked_sidecar += 1
        elif binding.label in declared:
            bound += 1
        else:
            rows.append(UnboundLabel(
                task_id=binding.task_id,
                label=binding.label,
                status=binding.status,
                manifest=manifest,
                declared_labels=declared,
            ))
    rows.sort(key=_unbound_sort_key)
    return LabelBindingSweep(
        unbound_labels=rows,
        manifest_bearing_tasks=len(bindings),
        tasks_bound_to_a_declared_label=bound,
        tasks_without_a_tracked_sidecar=without_tracked_sidecar,
        tasks_on_an_unparseable_sidecar=on_unparseable_sidecar,
    )


def audit_project(project_root: str, manifest_root: str | None = None) -> ProjectAudit:
    """Run both directions for one project: compare every mechanical sidecar
    descriptor against its task-record twin, and list every task label its
    tracked sidecar does not declare (:func:`_sweep_label_bindings`).

    *manifest_root* defaults to *project_root*, so a single-checkout run and a
    multi-project sweep behave exactly as they would without the flag.

    Raises ``sqlite3.Error`` for an unreadable task store, which
    :func:`_task_db_scan.sweep_project_roots` turns into a warn-and-skip.
    :class:`ManifestDiscoveryUnavailable` is caught HERE and recorded as
    ``git_discovery_failed`` rather than allowed to escape, because that helper
    only catches ``sqlite3.Error`` and an escaping traceback would abort the
    whole multi-root sweep over one bad manifest root.
    """
    root = str(project_root)
    manifests_root = str(manifest_root) if manifest_root is not None else root

    scan = load_task_store_scan(str(tasks_db_path(root)))

    try:
        relpaths = _tracked_manifest_paths(manifests_root)
    except ManifestDiscoveryUnavailable as exc:
        return ProjectAudit(
            project_root=root,
            manifest_root=manifests_root,
            findings=[],
            unbound_labels=[],
            coverage=AuditCoverage(
                manifests_swept=0,
                mechanical_capabilities_compared=0,
                capabilities_without_task_entry=0,
                manifest_tasks_without_db_row=0,
                malformed_task_entries=0,
                manifest_parse_failures=0,
                mechanical_capabilities_seen=0,
                unconvertible_sidecar_descriptors=0,
                task_entries_with_no_sidecar_capability=0,
                uncomparable_details=(str(exc),),
                git_discovery_failed=True,
                manifest_bearing_tasks=0,
                tasks_bound_to_a_declared_label=0,
                tasks_without_a_tracked_sidecar=0,
                tasks_on_an_unparseable_sidecar=0,
            ),
        )

    findings: list[DescriptorDrift] = []
    manifests_swept = 0
    seen = 0
    compared = 0
    without_entry = 0
    without_db_row = 0
    malformed = 0
    unconvertible = 0
    orphaned_entries = 0
    parse_failure_details: list[str] = []
    uncomparable_details: list[str] = []
    # Gathered here so the label-binding direction parses no sidecar twice.
    declared_by_manifest: dict[str, tuple[str, ...]] = {}
    unparseable_manifests: set[str] = set()

    for relpath in relpaths:
        try:
            doc = load_capability_manifest(Path(manifests_root) / relpath)
        except Exception as exc:  # noqa: BLE001 — recorded, never swallowed
            # NAMED, not merely counted: a sweep that could not read half the
            # corpus must never read as complete (no-silent-fail-soft).
            parse_failure_details.append(f"{relpath}: {exc}")
            unparseable_manifests.add(relpath)
            continue

        manifests_swept += 1
        declared_by_manifest[relpath] = tuple(task.label for task in doc.tasks)
        for task in doc.tasks:
            if task.task_id is None:
                # Authoring time, before commit_planning stamps the id. It
                # binds no producer, so there is nothing to compare against.
                continue
            try:
                task_id = int(task.task_id)
            except (TypeError, ValueError):
                continue

            if task_id not in scan.row_ids:
                without_db_row += 1
                continue
            entries = scan.delivered_checks.get(task_id, {})

            mechanical_names: set[str] = set()
            for capability in task.capabilities:
                check = capability.delivered_check
                if check is None or check.kind not in MECHANICAL_CHECK_KINDS:
                    continue
                mechanical_names.add(capability.name)
                # SEEN, not compared: this capability is merely ELIGIBLE for a
                # comparison. `compared` is incremented below, only once both
                # sides have actually normalized. See AuditCoverage.
                seen += 1

                entry = entries.get(capability.name)
                if entry is None:
                    without_entry += 1
                    continue

                try:
                    expected = _expected_meta(capability.name, check)
                except Exception as exc:  # noqa: BLE001 — recorded, never swallowed
                    # SYMMETRIC with the task-record guard below, and for the
                    # same reason: an unconvertible descriptor is a coverage
                    # row, never a finding and never an abort. See
                    # _expected_meta for why this cannot fire today and why it
                    # is guarded anyway.
                    unconvertible += 1
                    uncomparable_details.append(
                        f"{relpath} task {task_id} capability {capability.name!r}: "
                        f"unconvertible sidecar descriptor: {exc}"
                    )
                    continue

                try:
                    actual = DeliveredCheckMeta(**entry).model_dump()
                except Exception as exc:  # noqa: BLE001 — recorded, never swallowed
                    # NAMED, never silently dropped. A task-record entry
                    # that will not validate cannot be compared, and is a
                    # different defect from a drifted one — it goes to the
                    # coverage details, not the finding list.
                    malformed += 1
                    uncomparable_details.append(
                        f"task {task_id} capability {capability.name!r}: "
                        f"unvalidatable task-record entry: {exc}"
                    )
                    continue

                compared += 1
                if expected == actual:
                    continue

                differing = tuple(sorted(k for k in expected if expected[k] != actual[k]))
                findings.append(DescriptorDrift(
                    manifest=relpath,
                    task_id=task_id,
                    label=task.label,
                    capability=capability.name,
                    differing_fields=differing,
                    sidecar_check=expected,
                    task_check=actual,
                ))

            # THE OTHER DIRECTION. Everything above walks sidecar -> task, so a
            # task-record entry with no same-named mechanical capability is
            # invisible to it — and two real drift shapes live exactly there
            # (a renamed capability, and a grep->manual sidecar leaving a stale
            # mechanical entry behind). Reported as its OWN coverage class, not
            # folded into capabilities_without_task_entry: that bucket is
            # attributed to a different owner and explicitly never remediated
            # from this report, so absorbing a rename into it would misfile
            # genuine drift as somebody else's problem. Coverage rather than a
            # finding because a finding here means two spellings DISAGREE, and
            # an orphaned entry is an absence, not a disagreement.
            for orphan in sorted(set(entries) - mechanical_names):
                orphaned_entries += 1
                uncomparable_details.append(
                    f"task {task_id} capability {orphan!r}: task-record entry "
                    f"with no same-named mechanical capability in {relpath} "
                    f"(a renamed capability shows up here AND in the "
                    f"no-task-entry count, on the same task)"
                )

    findings.sort(key=_drift_sort_key)
    label_bindings = _sweep_label_bindings(
        scan.manifest_bindings, root, declared_by_manifest, unparseable_manifests)
    return ProjectAudit(
        project_root=root,
        manifest_root=manifests_root,
        findings=findings,
        unbound_labels=label_bindings.unbound_labels,
        coverage=AuditCoverage(
            manifests_swept=manifests_swept,
            mechanical_capabilities_compared=compared,
            capabilities_without_task_entry=without_entry,
            manifest_tasks_without_db_row=without_db_row,
            malformed_task_entries=malformed,
            manifest_parse_failures=len(parse_failure_details),
            mechanical_capabilities_seen=seen,
            unconvertible_sidecar_descriptors=unconvertible,
            task_entries_with_no_sidecar_capability=orphaned_entries,
            manifest_parse_failure_details=tuple(parse_failure_details),
            uncomparable_details=tuple(uncomparable_details),
            manifest_bearing_tasks=label_bindings.manifest_bearing_tasks,
            tasks_bound_to_a_declared_label=label_bindings.tasks_bound_to_a_declared_label,
            tasks_without_a_tracked_sidecar=label_bindings.tasks_without_a_tracked_sidecar,
            tasks_on_an_unparseable_sidecar=label_bindings.tasks_on_an_unparseable_sidecar,
        ),
    )


_COVERAGE_CAVEAT = (
    "  COVERAGE (the findings above are a comparison of MATCHED PAIRS only, "
    "not the whole corpus - SEEN is the eligible population and COMPARED the "
    "matched pairs, and the rows between them account for the difference: a "
    "capability with no same-named task-record entry, a manifest binding a "
    "task_id with no tasks.db row, an unvalidatable task entry, an "
    "unconvertible sidecar descriptor and an unparseable sidecar are all "
    "counted here and are NONE of them drift; the missing-entry class is owned "
    "by audit_combine_gate_marker_loss.py and is never remediated from this "
    "report. A task entry with no capability is the REVERSE direction - a "
    "RENAMED capability appears as that row AND a no-task-entry row on the "
    "SAME task, and that pair IS drift even though neither row alone says so):"
)

_UNBOUND_CAVEAT = (
    "  UNBOUND LABELS (a row is a manifest-bearing task whose prd_task_label "
    "its tracked sidecar does not declare, so commit_planning's stamper binds "
    "nothing for it and copies it no delivered_checks. A row is NOT by itself "
    "a defect - a sidecar may deliberately omit a label, and the cases task "
    "4907 adjudicated are recorded in the sidecars' .md twins under 'Unbound "
    "task labels (task 4907 adjudication)'. A historical row's task is done or "
    "cancelled, so no planning batch can touch it again; a LIVE row's task can "
    "still be touched. Fixing a row is a separate, reviewed edit: realign a "
    "misspelled or transliterated label on the task, clear prd_task_label "
    "(keeping prd_path) on a task filed from outside the PRD's decomposition "
    "plan, or complete the sidecar for a label that plan declares - never "
    "invent a sidecar entry for a label the plan does not declare):"
)

# Printed ABOVE a git-discovery-failed project's rows, because a reader must
# see that the zero below means "nothing was enumerated" before reading it as
# "nothing was wrong".
_DISCOVERY_FAILED_NOTICE = (
    "  WARNING: the manifest corpus could not be enumerated -- this is NOT a "
    "clean result, it is an UNKNOWN one (an empty corpus and a clean corpus "
    "are indistinguishable in the finding count below)"
)


def _format_finding_line(drift: DescriptorDrift) -> str:
    """One drift row, in the sibling audits' ``key=value`` style.

    The field set and order are this script's; format_kv_line supplies only the
    indent and separator. ``fields`` is deliberately shorter than the
    ``differing_fields`` attribute it carries.
    """
    return format_kv_line([
        ("manifest", drift.manifest),
        ("task_id", drift.task_id),
        ("capability", drift.capability),
        ("fields", ",".join(drift.differing_fields)),
    ])


def _format_unbound_line(row: UnboundLabel) -> str:
    """One unbound-label row, in the sibling audits' ``key=value`` style.

    ``hazard`` tells a LIVE row from a historical one at a glance and to a
    grep; see :attr:`UnboundLabel.is_live`.
    """
    return format_kv_line([
        ("manifest", row.manifest),
        ("task_id", row.task_id),
        ("label", row.label),
        ("status", row.status),
        ("hazard", "LIVE" if row.is_live else "historical"),
    ])


def _format_unbound_section(rows: list[UnboundLabel]) -> list[str]:
    """Render the label-binding direction: header, caveat, then every row.

    Printed even when *rows* is empty, and every row is NAMED, historical ones
    included: a reader must be able to tell "looked and found none" from
    "never looked", and a count alone would not say where to look. Each row's
    continuation line lists what its sidecar DOES declare.
    """
    live = sum(row.is_live for row in rows)
    lines = [f"  -- unbound task labels ({len(rows)}, {live} live) --", _UNBOUND_CAVEAT]
    for row in rows:
        lines.append(_format_unbound_line(row))
        lines.append(f"      declared labels: {', '.join(row.declared_labels) or '(none)'}")
    return lines


def _format_coverage(coverage: AuditCoverage) -> list[str]:
    """Render the always-printed coverage block.

    Never omitted and never abbreviated when there are no findings: the point
    is that the finding list compares matched pairs only, and a reader must be
    told the size of the unpaired remainder.

    The details NAME the unreadable sidecars and the unvalidatable task
    entries, never just count them. A count alone tells an operator that
    coverage is incomplete but not where to look, which swallows the failure at
    exactly the reporting boundary no-silent-fail-soft is about.

    Only the ALIGNMENT is shared with the sibling audits (via
    format_coverage_block); _COVERAGE_CAVEAT and the labels below are this
    script's, and say what THIS script could not see.
    """
    return format_coverage_block(
        _COVERAGE_CAVEAT,
        [
            ("manifests swept:", coverage.manifests_swept),
            # SEEN is the eligible population; COMPARED is the matched-pair
            # count. Both are printed, adjacent, so the volume figure cannot be
            # read as larger than it is and no reader has to derive one by
            # subtracting the skip rows below. See AuditCoverage.
            ("mechanical capabilities seen:", coverage.mechanical_capabilities_seen),
            ("mechanical capabilities compared:", coverage.mechanical_capabilities_compared),
            ("capabilities with no task entry:", coverage.capabilities_without_task_entry),
            ("task entries with no capability:",
             coverage.task_entries_with_no_sidecar_capability),
            ("manifest tasks with no db row:", coverage.manifest_tasks_without_db_row),
            ("unvalidatable task entries:", coverage.malformed_task_entries),
            ("unconvertible sidecar descriptors:",
             coverage.unconvertible_sidecar_descriptors),
            ("manifests that failed to parse:", coverage.manifest_parse_failures),
            # The label-binding direction. With the unbound-label count these
            # partition the manifest-bearing tasks; see AuditCoverage.
            ("manifest-bearing tasks:", coverage.manifest_bearing_tasks),
            ("tasks bound to a declared label:", coverage.tasks_bound_to_a_declared_label),
            ("tasks with no tracked sidecar:", coverage.tasks_without_a_tracked_sidecar),
            ("tasks on an unparseable sidecar:", coverage.tasks_on_an_unparseable_sidecar),
        ],
        details=(*coverage.manifest_parse_failure_details, *coverage.uncomparable_details),
    )


def format_report(audits: list[ProjectAudit]) -> str:
    """Render *audits* as a human-readable report.

    Every project gets both sections, drifted descriptors and unbound task
    labels, and its COVERAGE block, INCLUDING projects with nothing to report —
    see :func:`_format_unbound_section` and :func:`_format_coverage`. The
    trailing total counts both dimensions, and how many unbound rows are live.

    Both roots are named on the header line so a ``--manifest-root`` run is
    unambiguous: a reader never has to guess which manifest tree was compared
    against which task store.

    Pure: returns the text and never prints. ``main()`` does the single print.
    """
    lines: list[str] = []
    drifted = unbound = live = 0
    for audit in audits:
        if audit.manifest_root == audit.project_root:
            lines.append(f"{audit.project_root}:")
        else:
            lines.append(
                f"{audit.project_root} (manifests from {audit.manifest_root}):"
            )
        if audit.coverage.git_discovery_failed:
            # ABOVE both sections on purpose — see _DISCOVERY_FAILED_NOTICE.
            lines.append(_DISCOVERY_FAILED_NOTICE)

        drifted += len(audit.findings)
        lines.append(f"  -- drifted descriptors ({len(audit.findings)}) --")
        for drift in audit.findings:
            lines.append(_format_finding_line(drift))
            for field in drift.differing_fields:
                lines.append(
                    f"      {field}: sidecar={drift.sidecar_check[field]!r} "
                    f"task={drift.task_check[field]!r}"
                )
        unbound += len(audit.unbound_labels)
        live += sum(row.is_live for row in audit.unbound_labels)
        lines.extend(_format_unbound_section(audit.unbound_labels))
        lines.extend(_format_coverage(audit.coverage))

    lines.append(
        f"{drifted} drifted descriptor(s), {unbound} unbound task label(s) "
        f"({live} live) across {len(audits)} project(s)"
    )
    return "\n".join(lines)


def format_json(audits: list[ProjectAudit]) -> str:
    """Render *audits* as a JSON OBJECT (never a bare array).

    An object because the coverage block must travel WITH the findings: a
    machine consumer handed a bare finding array could read it as the whole
    corpus, which is the exact false-completeness the coverage block exists to
    prevent. The caveat prose ships in the payload for the same reason, each
    dimension's under its own key.

    Each unbound row carries ``is_live`` explicitly: it is a property, not a
    NamedTuple field, so ``_asdict()`` alone would drop it.
    """
    return json.dumps({
        "caveat": _COVERAGE_CAVEAT,
        "unbound_labels_caveat": _UNBOUND_CAVEAT,
        "projects": [
            {
                "project_root": audit.project_root,
                "manifest_root": audit.manifest_root,
                "coverage": audit.coverage._asdict(),
                "findings": [drift._asdict() for drift in audit.findings],
                "unbound_labels": [
                    {**row._asdict(), "is_live": row.is_live}
                    for row in audit.unbound_labels
                ],
            }
            for audit in audits
        ],
    })


def _is_dirty(audits: list[ProjectAudit]) -> bool:
    """Exit 1 keys on findings, a failed manifest discovery, OR a LIVE unbound
    label.

    The second disjunct is not belt-and-braces. An empty corpus and a clean
    corpus are INDISTINGUISHABLE in the finding count, and only one of them is
    good news — so a run that could not enumerate the corpus must never exit 0
    (docs/legibility/design-invariants.md, no-silent-fail-soft).

    The third is deliberately ASYMMETRIC. A historical unbound row is reported
    but not dirty, because a done or cancelled task can no longer be touched by
    a future ``commit_planning``. A live one can, and while it is live is the
    only time its label is still cheap to fix. See :attr:`UnboundLabel.is_live`.
    """
    return any(
        a.findings
        or a.coverage.git_discovery_failed
        or any(row.is_live for row in a.unbound_labels)
        for a in audits
    )


# The per-script NAMES survive because this script's epilog wording is its own,
# but the VALUES have ONE home: the returns live in _task_db_scan.run_audit_cli,
# so a local re-spelling would drift from what actually gets returned. The
# label-binding direction changed what makes a run dirty (_is_dirty), not these
# values. test_exit_constants_alias_the_shared_tier_3_codes keeps them honest.
EXIT_OK = AUDIT_EXIT_OK                            # swept; no drift and no
                                                   # LIVE unbound label
EXIT_DRIFT = AUDIT_EXIT_FINDINGS                   # a drifted descriptor, a
                                                   # LIVE unbound label, or a
                                                   # failed manifest discovery
EXIT_NO_ROOT = AUDIT_EXIT_NO_ROOT                  # no project root resolved to
                                                   # a readable tasks.db
EXIT_NOTHING_AUDITED = AUDIT_EXIT_NOTHING_AUDITED  # roots resolved but EVERY
                                                   # one failed to audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "READ-ONLY sweep of capability-manifest sidecars against the task "
            "store, in two directions. DESCRIPTOR DRIFT (sidecar -> task): a "
            "delivered_check descriptor that has drifted from the "
            "metadata.delivered_checks entry on its producer task. "
            "delivered_checks is copied one way, sidecar -> task, at "
            "commit_planning and never syncs back, so a hand-repaired task "
            "record sits beside a stale sidecar that any re-decompose would "
            "re-stamp over the repair. LABEL BINDING (task -> sidecar): a "
            "task whose prd_task_label its tracked sidecar does not declare, "
            "so commit_planning binds nothing for it. Reporting only -- never "
            "mutates a task record or a manifest. Resyncing a drifted sidecar "
            "or fixing an unbound label is a separate, individually-reviewed "
            "edit."
        ),
        epilog=(
            "exit codes: 0 = swept, every compared descriptor agrees and no "
            "unbound label is live; 1 = at least one drifted descriptor, OR a "
            "LIVE unbound label (its task not yet done or cancelled; a "
            "historical one is reported but does not count), OR the manifest "
            "corpus could not be enumerated (an empty corpus and a clean "
            "corpus are indistinguishable in the finding count, and only one "
            "is good news); 2 = no project root resolved to a readable "
            "tasks.db; 3 = roots resolved but every one failed to audit, so "
            "NOTHING was swept (never treat 3 as a clean run). A drift row "
            "means the two spellings DISAGREE -- it is not a claim that either "
            "one passes. An unbound label is not by itself a defect -- a "
            "sidecar may deliberately omit one."
        ),
    )
    parser.add_argument(
        "--project-root", dest="project_roots", action="append",
        help=(
            "Project root to audit (resolves <root>/.taskmaster/tasks/tasks.db, "
            "and its own tracked *.capability-manifest.yaml files unless "
            "--manifest-root says otherwise). May be repeated."
        ),
    )
    parser.add_argument(
        "--manifest-root", dest="manifest_root", default=None,
        help=(
            "Git checkout to read manifest sidecars FROM (default: the project "
            "root itself). Exists because .taskmaster/ is gitignored and lives "
            "only in the primary checkout, so auditing a task WORKTREE's "
            "sidecars means reading manifests from the worktree while reading "
            "the task store from the primary root. INTENDED FOR SINGLE-ROOT "
            "RUNS: this is ONE manifest tree applied to EVERY resolved root, "
            "so combining it with several roots is warned about on stderr."
        ),
    )
    parser.add_argument(
        "--json", action="store_true",
        help=(
            "Emit a JSON object (findings, unbound labels and coverage) instead "
            "of a report."
        ),
    )
    return parser


def _audit_root(root: str, args: argparse.Namespace) -> ProjectAudit:
    """Audit ONE project root under the (possibly overridden) manifest root.

    Raises ``sqlite3.Error`` for an unreadable task store, which
    :func:`_task_db_scan.sweep_project_roots` turns into a warn-and-skip; every
    other exception propagates. Returns exactly one audit, per that function's
    one-audit-per-root contract.
    """
    return audit_project(root, args.manifest_root)


def _render(audits: list[ProjectAudit], args: argparse.Namespace) -> str:
    return format_json(audits) if args.json else format_report(audits)


def _warn_manifest_root_across_roots(
    roots: list[str], args: argparse.Namespace
) -> None:
    """Warn once, up front, that ONE --manifest-root is applied to MANY roots.

    Runs on the RESOLVED root list, before the empty-roots exit-2 return — the
    position run_audit_cli's on_roots hook occupies. Warned, not rejected: one
    manifest tree over several same-corpus task stores is a legitimate use, and
    this script reports rather than gatekeeps.
    """
    if len(roots) > 1 and args.manifest_root:
        print(
            f"warning: --manifest-root {args.manifest_root!r} is applied to ALL "
            f"{len(roots)} resolved project roots, so every root is audited "
            "against the SAME manifest tree rather than its own. Any root whose "
            "sidecars differ will be compared against another checkout's "
            "spellings. Prefer one --manifest-root run per root.",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: sweep both directions, descriptor drift (sidecar -> task)
    and label binding (task -> sidecar), for every resolved project root.

    Exit codes: 0 = swept, every compared descriptor agrees and no unbound
    label is live; 1 = a drifted descriptor, a LIVE unbound label, OR a
    manifest corpus that could not be enumerated; 2 = no project root resolved
    to a readable tasks.db; 3 = roots resolved but EVERY one failed to audit,
    so NOTHING was swept. A historical unbound label (its task done or
    cancelled) is reported but never makes the run dirty — see
    :func:`_is_dirty`.

    3 exists because 0 would otherwise be returned for two opposite outcomes —
    "swept everything, found nothing" and "swept nothing at all" — and a
    CI/cron consumer reading only the exit code would take a total failure for
    a clean run. That is exactly the silent fail-soft this module refuses to do
    (see the module docstring).

    1 covers the failed-discovery case for the SAME reason, one level down: a
    root whose `git ls-files` failed produces zero findings, and zero findings
    from an unenumerable corpus is an UNKNOWN result, not a clean one.

    A single unreadable project (a corrupt/locked tasks.db) does NOT abort the
    sweep: it is logged to stderr and skipped so every other project is still
    audited, and a trailing warning states that the results are incomplete.

    The roots loop, the warn-and-continue skip and the exit-code ladder are
    :func:`_task_db_scan.run_audit_cli` (Tier 3). What stays here is what
    genuinely differs: this script's parser and epilog, its ``--manifest-root``
    handling (:func:`_audit_root` and :func:`_warn_manifest_root_across_roots`),
    its object-shaped JSON, its report and its :func:`_is_dirty` predicate.
    Nothing in this file returns a bare integer.
    """
    return run_audit_cli(
        argv,
        parser=_build_parser(),
        audit_fn=_audit_root,
        render=_render,
        is_dirty=_is_dirty,
        on_roots=_warn_manifest_root_across_roots,
    )


if __name__ == "__main__":
    sys.exit(main())
