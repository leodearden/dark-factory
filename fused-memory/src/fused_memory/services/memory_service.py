"""Core orchestration layer — owns backends, classifier, router, durable queue."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import time
import uuid as uuid_mod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from graphiti_core.nodes import EpisodeType

from fused_memory.backends.graphiti_client import GraphitiBackend
from fused_memory.backends.mem0_client import (
    _FUSED_MEMORY_OWNED_METADATA_KEYS,
    Mem0Backend,
    split_managed_metadata,
)
from fused_memory.config.schema import FusedMemoryConfig, MemoryMetadataConfig
from fused_memory.memory_metadata import (
    PARENT_ID_DEAD_CODE,
    PARENT_ID_UNAVAILABLE_CODE,
    CanonicalUniquenessViolation,
    MemoryMetadataValidationError,
    MetadataViolation,
    ParentHasChildrenError,
    is_valid_topic_slug,
    parent_liveness_violation,
    validate_memory_metadata,
)
from fused_memory.middleware.mem0_update_storm_escalator import Mem0UpdateStormEscalator
from fused_memory.models.enums import (
    GRAPHITI_PRIMARY,
    MEM0_PRIMARY,
    MemoryCategory,
    SourceStore,
)
from fused_memory.models.memory import (
    AddEpisodeResponse,
    AddMemoryResponse,
    EpisodeStatus,
    MemoryResult,
    ReadRouteResult,
)
from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)
from fused_memory.models.scope import Scope
from fused_memory.reconciliation.recon_pool_map import (
    CYCLE_SUMMARY_STAGE_TO_RECON_POOL as _CYCLE_SUMMARY_STAGE_TO_RECON_POOL,
)
from fused_memory.reconciliation.standing_decision_constants import (
    EXPIRY_REASON_MERGE,
    STATE_ACTIVE,
)
from fused_memory.reconciliation.standing_decision_writer import (
    expire_entity_standing_decision,
)
from fused_memory.routing.classifier import WriteClassifier
from fused_memory.routing.router import ReadRouter
from fused_memory.server.storm_counter import StormCounter
from fused_memory.services.durable_queue import DurableWriteQueue
from fused_memory.services.memory_metadata_census import (
    UnknownKeyStormDetector,
    emit_schema_warnings,
    file_unknown_key_storm_escalation,
)
from fused_memory.utils.async_utils import gather_collect, gather_or_raise
from fused_memory.utils.task_naming import canonicalize_task_node_name
from fused_memory.utils.validation import require_full_uuid

if TYPE_CHECKING:
    from fused_memory.backends.task_backend_protocol import TaskBackendProtocol
    from fused_memory.reconciliation.event_buffer import EventBuffer
    from fused_memory.reconciliation.recon_ledger import ReconLedgerStore
    from fused_memory.services.planned_episode_registry import PlannedEpisodeRegistry
    from fused_memory.services.write_journal import WriteJournal

logger = logging.getLogger(__name__)

# Per-sub-close timeout used by MemoryService._safe_close (task 2701). A healthy
# FalkorDB/Qdrant localhost driver teardown completes in well under 1s; 3s gives
# headroom under load while capping a hung network-driver close so no single
# backend can consume the whole shutdown budget or starve the durable-flush
# SQLite closes (_write_journal/_event_buffer) that run after it. The paired
# outer step budget lives in server/main.py as _MEMORY_CLOSE_STEP_TIMEOUT and
# must dominate 6 * _SUBCLOSE_TIMEOUT (guarded by TestShutdownBudgetArithmetic).
_SUBCLOSE_TIMEOUT = 3.0

# Canonical relational verb for dependency facts (mirrors routing/classifier.py:19).
# Used by _restore_superseded_dependency_edges to identify edges that should
# never be superseded by LLM edge-resolution.
_DEPENDENCY_FACT_RE = re.compile(r'\bdepends on\b', re.I)

# Mem0Backend.add() (backends/mem0_client.py) currently pins infer=False
# unconditionally for every write, which guarantees a successful mem0.add()
# always returns exactly one result with an id — so an empty result is
# always anomalous (task 1974; see the empty-result WARNING below). Flip
# this to False in lockstep if that pin is ever lifted or made
# configurable, so the WARNING doesn't degenerate into a recurring,
# non-actionable per-write log line under a legitimate infer-driven
# dedup/no-op.
_MEM0_ADD_INFER_PINNED_FALSE = True

# Canonical order for extracting a memory's human-readable content string from a
# raw Qdrant payload: mem0 stores the verbatim content under 'data', but older
# / alternately-written points use 'memory' or 'content'. First non-empty string
# wins. Kept package-internal here (a deliberate minor duplication of scripts/
# clear_malformed_empty_memory.py:_CONTENT_KEYS and audit_duplicate_memories.py)
# rather than importing from scripts/, so the service has no scripts/ dependency.
_MEM0_CONTENT_KEYS = ('data', 'memory', 'content')


def _is_dependency_fact(fact: str | None) -> bool:
    """Return True when *fact* expresses a dependency relationship.

    Uses the canonical ``depends on`` phrasing (case-insensitive, word-boundary
    anchored) that matches the project's relational verb in classifier.py.
    """
    return bool(fact) and _DEPENDENCY_FACT_RE.search(fact) is not None  # type: ignore[arg-type]


# Priority-override facts carry several independent sub-attributes (e.g.
# boost_tier, pinned, reserve_now, TTL) that legitimately coexist as distinct
# valid edges on the same entity. Matching on the "priority override" phrase
# alone would treat all of those sub-attributes as one single-valued
# predicate and wrongly invalidate one when another is written — recreating
# the over-invalidation failure mode task 2111 fixed. Requiring BOTH tokens
# restricts this classifier to the genuinely single-valued TTL scalar named
# in the task 2265 incident, used by
# ``_invalidate_stale_superseded_ttl_edges`` to identify same-subject
# contradictions that Graphiti's upstream edge-resolver under-invalidated.
_PRIORITY_OVERRIDE_TTL_FACT_RE = re.compile(
    r'\bpriority[-\s]+override\b.*\bTTL\b|\bTTL\b.*\bpriority[-\s]+override\b',
    re.I | re.S,
)


def _is_priority_override_ttl_fact(fact: str | None) -> bool:
    """Return True when *fact* expresses a priority-override TTL value.

    Requires BOTH a ``priority[-\\s]+override`` phrase AND a ``TTL`` token
    (case-insensitive, in either order) — see
    ``_PRIORITY_OVERRIDE_TTL_FACT_RE`` for why both are required. The
    separator allows one-or-more hyphen/whitespace characters (not just a
    single hyphen or ASCII space) so LLM-generated free text with double
    spaces, newlines, or other Unicode whitespace between the two words is
    still classified correctly.
    """
    return (
        bool(fact) and _PRIORITY_OVERRIDE_TTL_FACT_RE.search(fact) is not None  # type: ignore[arg-type]
    )


# Sibling of _PRIORITY_OVERRIDE_TTL_FACT_RE (task 2351, follow-up to 2319,
# esc-2319-8). reserve_now is the SAME class of single-valued scalar as TTL:
# the scheduler overrides table (server/tools.py) stores it as a single
# INTEGER column per task_id row, written/cleared via COALESCE the same way
# boost_tier/pinned/ttl_until are — never a list of co-existing values for one
# subject. So a fresh "reserve_now = true" (or false/cleared) fact leaving a
# pre-existing "reserve_now" edge with the opposite value still valid on the
# same subject is the identical under-invalidation risk task 2319 fixed for
# TTL — a reader (or the scheduler dispatch loop) could believe a reservation
# is active when it was cleared, or vice versa. Requiring BOTH the
# "priority override" phrase AND a reserve_now token (accepting the
# underscore/hyphen/space separator variants an LLM-generated fact might use,
# mirroring _PRIORITY_OVERRIDE_TTL_FACT_RE's own separator tolerance) keeps
# this matcher from collapsing other override sub-attributes (boost_tier,
# pinned, TTL) into the reserve_now predicate.
_PRIORITY_OVERRIDE_RESERVE_NOW_FACT_RE = re.compile(
    r'\bpriority[-\s]+override\b.*\breserve[-\s_]+now\b'
    r'|\breserve[-\s_]+now\b.*\bpriority[-\s]+override\b',
    re.I | re.S,
)


def _is_priority_override_reserve_now_fact(fact: str | None) -> bool:
    """Return True when *fact* expresses a priority-override reserve_now value.

    Sibling of ``_is_priority_override_ttl_fact`` (task 2351) for the
    reserve_now boolean scalar — same single-valued-per-subject shape and the
    same under-invalidation risk; see
    ``_PRIORITY_OVERRIDE_RESERVE_NOW_FACT_RE`` for why both tokens are
    required.
    """
    return (
        bool(fact)
        and _PRIORITY_OVERRIDE_RESERVE_NOW_FACT_RE.search(fact) is not None  # type: ignore[arg-type]
    )


def _is_priority_override_scalar_fact(fact: str | None) -> bool:
    """Return True when *fact* matches any recognized single-valued
    priority-override scalar shape (TTL or reserve_now).

    Convenience combinator for callers that only need a single "is this fact
    SOME priority-override scalar shape" boolean, with no need to know which
    one. ``_invalidate_stale_superseded_ttl_edges`` (task 2319, extended by
    task 2351) does NOT call this function — it always needs to know WHICH
    class(es) a fact matches, to keep TTL and reserve_now from being
    collapsed into one predicate, so it calls
    ``_priority_override_scalar_predicates`` directly for both the fire
    pre-filter and candidate matching. Using this union matcher for either of
    those would be unsafe: it cannot distinguish a fresh reserve_now write
    from a still-valid TTL edge on the same subject, which would
    re-invalidate the TTL edge (and vice versa) — see
    ``_priority_override_scalar_predicates`` and the hook body.
    """
    return _is_priority_override_ttl_fact(fact) or _is_priority_override_reserve_now_fact(fact)


# The set of distinct single-valued priority-override predicate classes a fact
# can carry. TTL and reserve_now are DISTINCT, legitimately-coexisting
# sub-attributes on the same subject (see the module comment at the top of
# this section) — a fresh reserve_now write does NOT contradict a valid TTL
# edge, and vice versa. The stale-superseded hook uses this per-edge predicate
# set (never the union matcher) to decide which candidates a fresh edge
# actually supersedes: only same-predicate candidates are invalidated. Using
# the union matcher for candidate matching would collapse TTL and reserve_now
# into one class and re-introduce the cross-predicate over-invalidation the
# two-token matchers (task 2111 lineage) exist to prevent.
_PRIORITY_OVERRIDE_SCALAR_PREDICATES: tuple[tuple[str, Any], ...] = (
    ('ttl', _is_priority_override_ttl_fact),
    ('reserve_now', _is_priority_override_reserve_now_fact),
)


def _priority_override_scalar_predicates(fact: str | None) -> frozenset[str]:
    """Return the set of single-valued priority-override predicate classes
    *fact* matches (a subset of ``{'ttl', 'reserve_now'}``).

    A fact normally matches exactly one class, but the return type is a set so
    a (rare) fact mentioning both scalars is handled without silently picking
    one. The stale-superseded hook invalidates a candidate edge only when its
    predicate set INTERSECTS the fresh edge's predicate set for the same
    subject — the discrimination that keeps a reserve_now write from
    invalidating a valid TTL edge (and vice versa).
    """
    if not fact:
        return frozenset()
    return frozenset(
        name for name, matcher in _PRIORITY_OVERRIDE_SCALAR_PREDICATES if matcher(fact)
    )


# Canonical stage -> recon_pool map for per-cycle reconciliation summaries
# (metadata.kind == 'cycle_summary'), imported above from the leaf module
# reconciliation/recon_pool_map.py (task 2140) so this map and the per-stage
# _STAGE1_CYCLE_SUMMARY_RECON_POOL / _STAGE2_CYCLE_SUMMARY_RECON_POOL
# constants in reconciliation/stages/*.py are literally the same object —
# see recon_pool_map.py for why importing it here doesn't recreate the
# circular import that used to force a duplicated dict.


def _infer_recon_pool(meta: dict) -> str | None:
    """Infer the recon_pool tag for a cycle_summary write from metadata.stage.

    Returns None when meta['kind'] != 'cycle_summary' (non-cycle-summary
    writes are never touched) or when meta['stage'] is missing/unknown (the
    pool cannot be inferred; callers must not clobber any caller-supplied
    recon_pool in that case).
    """
    if meta.get('kind') != 'cycle_summary':
        return None
    stage = meta.get('stage')
    if not isinstance(stage, str):
        return None
    return _CYCLE_SUMMARY_STAGE_TO_RECON_POOL.get(stage)


_REQUIRED_CYCLE_SUMMARY_KEYS: tuple[str, ...] = ('stage', 'run_id')


def _missing_cycle_summary_keys(meta: dict) -> list[str]:
    """Return which required keys among _REQUIRED_CYCLE_SUMMARY_KEYS are
    missing/invalid on a cycle_summary write; [] for non-cycle_summary kinds
    (and whenever all required keys are present and valid).

    stage is invalid when absent, non-str, or not a known key in
    _CYCLE_SUMMARY_STAGE_TO_RECON_POOL — the same "known stage" check
    _infer_recon_pool uses, so there is one source of truth for the stage
    set. run_id is invalid when absent, non-str, or empty/whitespace-only
    (an empty run_id is as useless to the Path-2 triple-filter
    count_memories_by_metadata({kind, run_id, stage}) pre-check as an absent
    one). Order is stable: 'stage' before 'run_id'.
    """
    if meta.get('kind') != 'cycle_summary':
        return []

    missing: list[str] = []
    stage = meta.get('stage')
    if not isinstance(stage, str) or stage not in _CYCLE_SUMMARY_STAGE_TO_RECON_POOL:
        missing.append('stage')
    run_id = meta.get('run_id')
    if not isinstance(run_id, str) or not run_id.strip():
        missing.append('run_id')
    return missing


def _cycle_summary_run_id_backfill(meta: dict, causation_id: str | None) -> str | None:
    """Return a backfill value for a missing/invalid cycle_summary run_id, or
    None when there is nothing to repair (task 2109).

    run_id is LLM-supplied (see the reconciliation stage1/stage2 prompts) and
    empirically gets dropped under prompt-compliance failures. Rather than
    only warning (the task-2094 behavior), repair it server-side from an
    authoritative causation id — mirrors the _infer_recon_pool precedent
    (task 2077): derive/repair authoritatively rather than trust the LLM.

    Two candidate sources are checked, in order:
      1. meta['_causation_id'] — set by a direct in-process caller (and the
         literal source the task names).
      2. the causation_id parameter — the production MCP-boundary path:
         server/tools.py::_extract_causation pops '_causation_id' out of the
         metadata dict into this parameter before MemoryService.add_memory
         runs, so on the real reconciliation stage-agent path
         meta['_causation_id'] is always absent and the run_id value lives
         only in this parameter. Checking meta alone would never fire in
         production.

    Returns None (nothing to backfill) when: kind != 'cycle_summary'; run_id
    is already a non-empty string (must not clobber a valid value); or
    neither candidate source is a non-empty string.

    KNOWN LIMITATION (flagged in task 2109 amendment review): the
    causation_id parameter is authoritative BY CONVENTION, not by proof.
    server/tools.py::_extract_causation never actually passes None through —
    when '_causation_id' is absent from metadata it synthesizes a fresh
    str(uuid4()) on the spot (tools.py ~398-399). This function cannot tell
    that call-scoped synthetic UUID apart from a genuine run-scoped
    causation id (reconciliation/harness.py generates run_id the same way,
    via str(uuid4())), so a cycle_summary write that drops BOTH run_id and
    '_causation_id' before reaching the MCP boundary still gets a
    plausible-looking run_id backfilled here — silently suppressing the
    fallback warning that would otherwise flag the fully-dropped case, and
    handing the Path-2 triple-filter count_memories_by_metadata pre-check a
    run_id that will never match any sibling write from the same real run.
    Closing this gap requires _extract_causation to flag whether it
    synthesized (vs. received) the id and, for a synthesized id, skip this
    backfill in favor of the warning — a change to
    fused_memory/server/tools.py, outside this task's locked module scope.
    See TestCycleSummaryRunIdBackfillToolsBoundary in
    tests/test_memory_service.py, which pins today's behavior so this stays
    a tracked, visible follow-up rather than silent debt.
    """
    if meta.get('kind') != 'cycle_summary':
        return None
    run_id = meta.get('run_id')
    if isinstance(run_id, str) and run_id.strip():
        return None
    for candidate in (meta.get('_causation_id'), causation_id):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _normalize_task_id_metadata(meta: dict) -> None:
    """Coerce ``meta['task_id']`` to ``str`` in place, when present and non-None.

    Shared by add_memory, add_system_record, count_memories_by_metadata, and
    get_memories_by_metadata (task 2620 and its amendment rounds, closing
    gaps flagged in the task-2620 review — add_system_record is a parallel
    Mem0-only write path sharing add_memory's exact-match read filters, and
    count_memories_by_metadata/get_memories_by_metadata are the read side of
    those same filters) so every path that can produce or consume a
    task_id-keyed marker gets the same handling: those read filters are
    exact-match, and the project-wide convention is a string task_id
    (recon_ledger's task_id column is TEXT; every reader queries with
    str(task_id)) — an int-typed value written without this coercion
    silently false-negatives against a str-typed query filter, making a
    write invisible to its own gate (e.g. the stage2_suppress
    completion-guard). Coercing at the write boundary closes that gap for
    every future write instead of relying on each LLM-prompt-driven writer
    to remember the convention (task 2620, sibling of task 2454's
    flag_dedup-specific fix).

    On the read side (count_memories_by_metadata/get_memories_by_metadata),
    callers pass their own ``filters`` dict; those two methods copy it
    before calling this helper so the caller's original dict is never
    mutated in place. This direction only protects a caller who queries
    with an int-typed task_id filter (forgetting the str convention)
    against the now-str-normalized data the write side produces — it does
    NOT retroactively make historical int-typed task_id values, or
    anything written by a path that bypasses add_memory/add_system_record,
    matchable. Qdrant's payload filter is type-sensitive, so a str-coerced
    query can only ever match str-typed stored data; reaching legacy
    int-typed rows needs a separate backfill/migration, not read-side
    coercion.

    Assumes a scalar (int/str) task_id, matching today's single-task-id
    write convention on this path. A list/tuple value would str()-coerce to
    a Python repr (e.g. ``'[5040, 5149]'``), not a filter-compatible
    canonical form — unlike ``server/recon_report._canonicalize_task_id_string``'s
    comma-joined-string canonicalization used elsewhere for multi-task dedup
    signatures. No current caller passes a non-scalar task_id on this path;
    if one ever does, canonicalize consistently with that helper instead of
    relying on this bare ``str()`` (task-2620 review, noted as a latent edge
    case rather than a live defect).
    """
    if 'task_id' in meta and meta['task_id'] is not None:
        meta['task_id'] = str(meta['task_id'])


async def _apply_memory_metadata_validation(
    meta: dict,
    *,
    project_id: str,
    agent_id: str | None,
    config: MemoryMetadataConfig,
    storm_detector: UnknownKeyStormDetector,
    project_root: str,
    parent_lookup: Callable[[str, str], Awaitable[dict | None]],
    count_canonical: Callable[[str, dict], Awaitable[int]],
    find_canonical: Callable[..., Awaitable[list[dict]]],
) -> None:
    """Validate the Mem0 metadata vocabulary at the write boundary, in place.

    Task 3195 (leaf β of ``docs/prds/memory-metadata-vocabulary.md``).  The
    third of this module's shared in-place metadata helpers, alongside
    :func:`_normalize_task_id_metadata` and
    :func:`_apply_cycle_summary_metadata_tagging`, and shared by
    ``add_memory`` and ``add_system_record`` for the same reason the
    task-2222 amendment made the cycle-summary tagging shared: PRD D8/§2 pin
    enforcement at the SERVICE seam precisely because ``add_system_record``
    is a second write path that a tools-layer validator would leak past.
    Two call sites with drifting behaviour would reopen that hole.

    Discharges five obligations:

    1. **Normalize + shape-check** via ``validate_memory_metadata`` (the only
       in-place mutation is ``supersedes`` scalar→list, PRD D2).
    2. **Resolve ``parent_id`` LIVENESS** (task 3197, leaf δ) — see below.
    3. **Census** every violation, fatal or not, so warn-mode leaves a trace.
    4. **Reject** — but ONLY when ``enforce`` is on AND at least one
       violation is fatal.  Unknown keys are never fatal, so flipping
       ``enforce`` cannot turn the 1,627-key long tail into an outage.
    5. **Re-check ``canonical`` UNIQUENESS** (task 3198, leaf ε) via
       :func:`_check_canonical_uniqueness`, which raises its own
       :class:`CanonicalUniquenessViolation`.  It runs AFTER the reject arm
       above: malformed metadata is refused on shape before any live-state
       probe is spent on it.

    LIVENESS IS HERE, NOT IN THE REGISTRY, on purpose.  Leaf β made
    ``validate_memory_metadata`` a pure synchronous function taking only a
    dict, so it structurally *cannot* perform a store lookup — a boundary
    its docstring states explicitly so a later leaf "cannot accidentally
    grow a second implementation of it in here (INV-5)".  This helper is
    the nearest layer that can reach a store, and it is already the SINGLE
    shared home for both write paths, so putting liveness here gets
    ``add_system_record`` covered by construction.  Only the CODES and the
    message wording stay in the registry, behind
    :func:`~fused_memory.memory_metadata.parent_liveness_violation`, so the
    rule still has exactly one normative home.

    The lookup fires only when ``parent_id`` is PRESENT *and* already
    shape-valid: the common write path (no ``parent_id`` at all — leaf α
    measured zero live records carrying one) pays no round-trip, and an id
    no store could resolve is never spent on.  Liveness ADDS a violation
    rather than opening a second rejection path: because
    ``parent_liveness_violation`` is ``fatal=True``, warn mode censuses and
    proceeds while ``enforce`` rejects, both through the same arms below.

    A lookup that FAILS is a different fact from a parent that is gone, and
    gets its own code (``parent_id_liveness_unavailable``).
    ``Mem0Backend.get_point_by_id`` propagates a Qdrant read-timeout rather
    than collapsing it into ``None`` precisely to preserve that
    distinction; folding both into ``dead_parent_id`` would discard it here
    and tell an operator a live parent is dead.  That code is fatal too, so
    ``enforce`` fails CLOSED on it — INV-3 read literally: an actor that
    cannot corroborate must not act.  The blast radius of failing closed is
    confined to writes that actually carry ``parent_id``, a population leaf
    α measured at zero live records, and only while ``enforce`` is on.

    ``parent_lookup`` is REQUIRED and takes no ``None`` default.  A
    defaultable resolver would let a future third write path construct this
    helper without one and silently skip liveness — reintroducing the exact
    silent-orphan class leaf δ exists to close, and doing it invisibly.

    The enforce flags are read PER CALL off the shared config object rather
    than captured, so a config edit takes effect on the next write.

    Raises :class:`MemoryMetadataValidationError` when enforcing.  Everything
    else here — the census line, the storm detector, the escalation filing —
    is strictly best-effort and structurally cannot raise, because it runs on
    the live memory write path where a raise would fail the write because the
    *complaint about* the write failed.

    ASYNC ON PURPOSE — do not re-inline the escalation hop.  Validation,
    census and detection are pure CPU and stay inline, but
    ``file_unknown_key_storm_escalation`` does blocking filesystem I/O
    (``EscalationQueue`` construction, a queue-directory scan, a durable
    fsync-flushed write).  Called directly from these coroutines it would run
    that I/O ON the event loop and stall every other concurrent memory write
    for its duration.  The ported precedent
    (``middleware/candidate_key_escalation``) is invoked from a synchronous
    SQLite-migration path, so its never-raises contract transfers but its
    sync-context assumption does not.  ``asyncio.to_thread`` is awaited rather
    than fire-and-forgotten so the call can never outlive the write or be
    dropped by task GC; it yields the loop, which is the property that
    matters.
    """
    violations = validate_memory_metadata(
        meta, enforce_kind_registry=config.enforce_kind_registry
    )

    # parent_id LIVENESS (leaf δ). Gated on the SHAPE check having passed —
    # `validate_memory_metadata` emits `invalid_parent_id_shape` under the
    # same key, so any parent_id-keyed violation means the id is malformed
    # and no store could resolve it in that spelling.
    if 'parent_id' in meta and not any(v.key == 'parent_id' for v in violations):
        try:
            parent = await parent_lookup(project_id, meta['parent_id'])
        except Exception as exc:
            # `Exception`, never `BaseException`: CancelledError,
            # KeyboardInterrupt and SystemExit must keep propagating, per
            # the repo's cancellation convention.
            #
            # The exception TYPE is logged so the raw backend cause is
            # degraded, not discarded — the census code says only "could
            # not be checked", and an operator debugging a burst of
            # `parent_id_liveness_unavailable` needs something to correlate
            # against.
            logger.warning(
                'memory_metadata: parent_id liveness lookup failed for '
                'project_id=%r parent_id=%r: %s: %s',
                project_id, meta['parent_id'], type(exc).__name__, exc,
            )
            liveness_code = PARENT_ID_UNAVAILABLE_CODE
        else:
            liveness_code = None if parent is not None else PARENT_ID_DEAD_CODE
        if liveness_code is not None:
            violations.append(
                parent_liveness_violation(meta['parent_id'], code=liveness_code)
            )

    # NOTE: this is `if violations:`, not an early `return` — the canonical
    # uniqueness re-check below must still run for metadata that is
    # perfectly well-formed, which is the overwhelmingly common case for a
    # canonical write.  An early return here would make the whole check
    # dead code that still looked wired up.
    if violations:
        emit_schema_warnings(violations, project_id=project_id, agent_id=agent_id)

        unknown_keys = [v.key for v in violations if v.code == 'unknown_key']
        if unknown_keys and storm_detector.record(project_id, agent_id, unknown_keys):
            if project_root:
                await asyncio.to_thread(
                    file_unknown_key_storm_escalation,
                    project_root,
                    project_id=project_id,
                    agent_id=agent_id,
                    keys=unknown_keys,
                )
            else:
                # No configured taskmaster.project_root means no project queue to
                # file into. The census lines are already out, so the signal is
                # not lost — only the escalation.
                logger.debug(
                    'memory_metadata: unknown-key storm from project_id=%r '
                    'agent_id=%r but no project_root is configured; not escalating',
                    project_id, agent_id,
                )

        if config.enforce and any(v.fatal for v in violations):
            raise MemoryMetadataValidationError([v for v in violations if v.fatal])

    await _check_canonical_uniqueness(
        meta,
        project_id=project_id,
        agent_id=agent_id,
        config=config,
        count_canonical=count_canonical,
        find_canonical=find_canonical,
    )


#: Returned as the incumbent id when the count says an incumbent exists but
#: the follow-up scroll comes back empty (a concurrent delete between the two
#: round-trips).  A structured rejection with an unresolvable id beats an
#: IndexError on the live write path.
_CANONICAL_INCUMBENT_UNKNOWN = '<unknown>'


async def _check_canonical_uniqueness(
    meta: dict,
    *,
    project_id: str,
    agent_id: str | None,
    config: MemoryMetadataConfig,
    count_canonical: Callable[[str, dict], Awaitable[int]],
    find_canonical: Callable[..., Awaitable[list[dict]]],
) -> None:
    """Enforce <=1 canonical memory per ``(project, topic)`` (PRD V1, INV-3).

    The live half of ``canonical``.  It lives HERE rather than in
    :func:`~fused_memory.memory_metadata.validate_memory_metadata` because
    it needs store state, and that validator is pure by construction — the
    shape half (``canonical_without_topic``) stays there.

    Collaborators are injected as bound callables rather than as ``self``,
    matching how this seam already takes ``storm_detector``/``config``: the
    module-level helper stays decoupled from ``MemoryService`` and trivially
    stubbable.

    SCOPE — THE INVARIANT IS MEM0-SCOPED (3198 amendment, stated because
    the silence read as coverage).  Both probes go to Mem0/Qdrant payload
    filters, but this seam deliberately runs BEFORE the
    ``write_graphiti``/``write_mem0`` branching so that no write path can
    bypass the vocabulary rules.  The consequence, spelled out rather than
    left to be discovered: for a Graphiti-primary category
    (``entities_and_relations``, ``temporal_facts``,
    ``decisions_and_rationale``) a ``canonical: True`` record never lands
    in Mem0, so the count cannot see a previously-written Graphiti-primary
    canonical and the <=1-per-``(project, topic)`` rule does NOT hold for
    those categories.  This matches the PRD, whose whole vocabulary is
    framed as the Mem0 metadata vocabulary.

    The probe is nonetheless issued for every canonical write rather than
    skipped for Graphiti-primary ones, deliberately: ``dual_write`` can
    route any category into Mem0 too, so a category-based skip would be
    wrong exactly when it mattered, and a Graphiti-primary canonical that
    DOES have a Mem0 twin still gets caught.  The cost is one count that
    can only return 0 on a Graphiti-only canonical write — a rare write on
    a rare key.  ``TestCanonicalUniquenessAtSeam`` pins this behaviour for
    ``decisions_and_rationale`` so a later reader cannot mistake it for
    coverage.  Closing the gap properly needs a Graphiti-side count, which
    the PRD does not specify — do not fake it here.

    PROBE FAILURE (3198 amendment).  Both probes talk to Qdrant and
    ``Mem0Backend.count_by_metadata`` propagates a read timeout by
    contract, so the probe can fail for reasons that have nothing to do
    with the write — including for a Graphiti-primary write that would
    never have touched Mem0 at all.  Explicitly decided, not incidental:

    * a failure is ALWAYS censused, under ``code``
      ``canonical_uniqueness_check_unavailable`` — degradation is loud, and
      an operator can grep the same census stream they already watch;
    * ``enforce = False`` (the shipped default) → the write PROCEEDS.  Warn
      mode's whole contract is "census the violation and let the write
      through"; failing a valid write because the *complaint machinery* was
      unavailable would be strictly worse than the duplicate it is trying to
      prevent, and would contradict this seam's promise that everything but
      the enforce-raise is best-effort;
    * ``enforce = True`` → FAIL CLOSED: the original backend error is
      re-raised.  An operator who turned enforcement on asked for the
      invariant to hold; admitting an unverifiable canonical would be the
      silent fail-soft the house norm forbids.  The error surfaces as
      itself (bare ``raise``, traceback intact) rather than being dressed up
      as a :class:`CanonicalUniquenessViolation`, because "the store was
      unreachable" and "a duplicate exists" are different facts and a caller
      must be able to tell them apart.

    Ordering and guards are load-bearing — the ORDINARY write path must
    issue ZERO extra round-trips:

    1. not an asserted canonical → return.  Every non-canonical write, i.e.
       almost all of them, stops here having done no I/O.
    2. ``topic`` missing or not slug-valid → return.  The shape violation
       was already reported by the pure validator, and we must never build
       a query on a malformed key (``count_by_metadata`` also rejects an
       empty filter).
    3. count == 0 → return.  The happy path pays exactly one exact Qdrant
       count and never scrolls.
    4. otherwise resolve the incumbent's id and reject.

    WHY COUNT THEN SCROLL: V1 contract-fixes ``count_memories_by_metadata``
    as the INV-3 mechanism, but also requires the error to name the existing
    canonical's id — which an ``int`` structurally cannot carry.  Counting
    first honours both, and confines the second round-trip to a path that is
    already failing the write, where its cost is irrelevant.

    WHY THE EXISTING ``enforce`` FLAG AND NOT A NEW ONE: measured, not
    assumed.  When this was written the live ``dark_factory`` corpus held
    exactly ONE ``canonical: true`` record, and its ``topic`` was
    ``eval_worktree_plan_tools_missing`` — snake_case, which fails
    ``TOPIC_SLUG_RE``.  Enforcing uniqueness on day one over a topic key
    whose own live values still only warn would be the census-refuted-premise
    outage the warn default exists to prevent, and would make uniqueness the
    single fatal check that ignores the flag every sibling check honours.

    THAT SPECIFIC HAZARD IS NOW CLEARED, AND θ WAS NOT ACTUALLY THE GATE
    (measured 2026-08-04).  Two corrections to the paragraph above, both
    recorded because the original wording sent a reader to the wrong check:

    * ``retro_stamp_topics.py`` has now been RUN (``--apply``), and the
      residual records outside its id-bounded manifest were normalized too,
      so ``legacy_topic_spelling_remains`` is empty and every live
      ``canonical: true`` topic conforms in BOTH projects.  Note the trap
      this sentence used to set: task 3201 was ``done`` for months meaning
      the SCRIPT had landed, while the sweep had never been applied
      (``stamped_total: 0``).  "Has θ landed?" is the wrong question —
      re-run the script and read ``legacy_topic_spelling_remains``.
    * θ was necessary bookkeeping but nearly irrelevant to blast radius.
      ``enforce`` rejects WRITES and never re-validates the corpus, so
      normalizing records at rest moved the measured false-rejection rate
      by ~1/week (~20 → ~19).  Rejections come from NEW writes by writers
      who were never told the rule: ``_MEMORY_INSTRUCTIONS`` still carries
      no slug guidance.  THE REAL PRECONDITION is leaf ι (task 3202).

    Task 3626 is the gate that re-measures and decides the flip; it carries
    the full model and the re-measurement recipes.  Do not flip from this
    docstring alone.

    RESIDUAL — this check is inherently TOCTOU-windowed: two concurrent
    first-canonical writes for one topic can both observe 0 and both
    succeed.  The PRD specifies no locking and Qdrant has no unique
    constraint, so the window is stated plainly rather than papered over
    with an implication of atomicity.  ``pick_survivor``
    (``fused-memory/scripts/audit_duplicate_memories.py``) remains the
    after-the-fact backstop that resolves a duplicate pair.  Do NOT add
    locking here — that would be an unreviewed scope expansion.
    """
    if meta.get('canonical') is not True:
        return

    topic = meta.get('topic')
    if not isinstance(topic, str) or not is_valid_topic_slug(topic):
        return

    filters = {'topic': topic, 'canonical': True}

    def _census(code: str, message: str) -> None:
        """Emit one uniqueness census line.

        Routed through the SAME ``emit_schema_warnings`` path the shape
        violations use, so every code this check can produce is
        grep-anchored in the one census format operators already know
        (V1: "grep-anchored, never renamed").
        """
        emit_schema_warnings(
            [MetadataViolation(key='canonical', code=code, message=message, fatal=True)],
            project_id=project_id,
            agent_id=agent_id,
        )

    try:
        if await count_canonical(project_id, filters) == 0:
            return
        records = await find_canonical(project_id, filters, limit=1)
    except Exception as exc:
        # The PROBE failed, not the write — a Qdrant timeout/outage, which
        # `count_by_metadata` propagates by contract.  Always censused
        # (degradation is loud); fail-open under the shipped warn default
        # because failing a valid write when only the complaint machinery
        # broke is worse than the duplicate it guards against; fail-closed
        # under `enforce` because an unverifiable canonical must not be
        # admitted silently.  Re-raised bare so the caller sees the real
        # backend error rather than a CanonicalUniquenessViolation that
        # would assert a duplicate we never actually observed.  Full
        # reasoning under PROBE FAILURE in the docstring.
        _census(
            'canonical_uniqueness_check_unavailable',
            f'could not verify <=1 canonical per (project, topic) for '
            f'project_id={project_id!r} topic={topic!r}: '
            f'{type(exc).__name__}: {exc}',
        )
        if config.enforce:
            raise
        return

    incumbent_id = (
        records[0].get('id', _CANONICAL_INCUMBENT_UNKNOWN)
        if records
        else _CANONICAL_INCUMBENT_UNKNOWN
    )
    error = CanonicalUniquenessViolation(
        project_id=project_id, topic=topic, incumbent_id=incumbent_id
    )

    # `config.enforce` is read PER CALL off the shared config object, never
    # captured, so a config edit takes effect on the next write — the same
    # note this seam already makes about the shape-check enforce flag.
    if config.enforce:
        raise error

    # Warn mode: census the violation and let the write proceed.
    _census('canonical_uniqueness_violation', str(error))


def _apply_cycle_summary_metadata_tagging(
    meta: dict,
    causation_id: str | None,
    *,
    project_id: str,
) -> None:
    """Apply server-side cycle_summary metadata tagging to ``meta`` in place.

    Shared by add_memory and add_system_record (task 2222 amendment) so
    every write path that can carry a cycle_summary payload gets the same
    authoritative treatment: recon_pool auto-tag from metadata.stage (task
    2077) — recon_pool is the only key the pool-cap trim and
    prune_recon_cycle_summaries.py filter on; run_id auto-backfill from the
    causation id (task 2109) — run_id drives the Path-2 triple-filter
    verification pre-check; and a WARNING for whatever remains
    missing/invalid after the backfill (task 2094/2109), so an untagged or
    unverifiable cycle_summary write is observable instead of silently
    piling up unbounded or dropping out of the Path-2 pre-check. No-op
    (and no warning) for any meta['kind'] != 'cycle_summary'.
    """
    inferred_recon_pool = _infer_recon_pool(meta)
    if inferred_recon_pool is not None:
        meta['recon_pool'] = inferred_recon_pool

    backfilled_run_id = _cycle_summary_run_id_backfill(meta, causation_id)
    if backfilled_run_id is not None:
        meta['run_id'] = backfilled_run_id

    missing_keys = _missing_cycle_summary_keys(meta)
    if missing_keys:
        logger.warning(
            'MemoryService: cycle_summary write missing required '
            'metadata key(s) %s — stage drives recon_pool (pool-cap trim); '
            'run_id drives the Path-2 triple-filter verification pre-check',
            missing_keys,
            extra={
                'project_id': project_id,
                'stage': meta.get('stage'),
                'run_id': meta.get('run_id'),
                'caller_recon_pool': meta.get('recon_pool'),
                'causation_id': causation_id,
                'missing_cycle_summary_keys': missing_keys,
            },
        )


class MemoryNotFoundError(Exception):
    """Raised when a mem0 memory id is not found."""


def _serialize_temporal(
    valid_at: Any,
    invalid_at: Any,
) -> dict[str, str | None] | None:
    """Serialize valid_at/invalid_at to an ISO 8601 dict or None.

    Returns None when both values are None (common case — no temporal context).
    Uses .isoformat() when available, falls back to str() for pre-serialized strings
    or other types.
    """
    if valid_at is None and invalid_at is None:
        return None

    def _to_iso(v: Any) -> str | None:
        if v is None:
            return None
        if hasattr(v, 'isoformat'):
            return v.isoformat()
        return str(v)

    return {
        'valid_at': _to_iso(valid_at),
        'invalid_at': _to_iso(invalid_at),
    }


def _created_at_to_utc_iso(created_at: datetime | None) -> str | None:
    """Serialize an episode's created_at to canonical UTC ISO-8601, or None.

    str(created_at) preserves the stored UTC offset (and uses a space
    separator), so emitted-string order can diverge from instant order
    when episodes carry differing offsets. Normalizing to UTC ISO-8601
    makes the emitted strings sort lexically iff the instants do. Naive
    (tzinfo-less) values are assumed to already be UTC, matching
    retrieve_episodes' own naive-datetime handling.
    """
    if created_at is None:
        return None
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    return created_at.astimezone(UTC).isoformat()


def _node_to_dict(n: Any) -> dict:
    """Normalize a Graphiti node into get_entity's node dict shape.

    Accepts either a dict (exact-match path, from graphiti.get_nodes_by_exact_name)
    or an object with attributes (fuzzy-match path, from graphiti.search_nodes).
    Missing/None 'labels' default to [] in both cases.
    """
    if isinstance(n, dict):
        uuid, name, summary, labels = n.get('uuid'), n.get('name'), n.get('summary'), n.get('labels')
    else:
        uuid = getattr(n, 'uuid', None)
        name = getattr(n, 'name', None)
        summary = getattr(n, 'summary', None)
        labels = getattr(n, 'labels', None)
    return {'uuid': uuid, 'name': name, 'summary': summary, 'labels': labels or []}


def _edge_to_dict(e: Any) -> dict:
    """Normalize a Graphiti edge into get_entity's edge dict shape.

    Accepts either a dict (exact-match path, from graphiti.get_valid_edges_for_node —
    an EdgeDict {uuid, fact, name} with no temporal fields) or an object with
    attributes (fuzzy-match path, from graphiti.search — a fact-search result
    that may carry valid_at/invalid_at). Dict inputs yield temporal=None today
    since EdgeDict has no valid_at/invalid_at keys; _serialize_temporal is used
    regardless so this stays forward-compatible if EdgeDict is later enriched.
    """
    if isinstance(e, dict):
        return {
            'uuid': e.get('uuid'),
            'fact': e.get('fact', ''),
            'temporal': _serialize_temporal(e.get('valid_at'), e.get('invalid_at')),
        }
    return {
        'uuid': getattr(e, 'uuid', None),
        'fact': getattr(e, 'fact', str(e)),
        'temporal': _serialize_temporal(
            getattr(e, 'valid_at', None),
            getattr(e, 'invalid_at', None),
        ),
    }


# Defensive: resolve the embedding-provider's RateLimitError class at module
# load. If the openai SDK is absent or its exception hierarchy is renamed,
# _is_rate_limit_or_quota_error still works via its duck-typed fallback checks
# below rather than raising an ImportError.
try:
    from openai import RateLimitError as _OpenAIRateLimitError
except Exception:
    _OpenAIRateLimitError = None


def _is_rate_limit_or_quota_error(exc: BaseException) -> bool:
    """Return True if *exc* represents an embedding-provider rate-limit or quota error.

    Matches ANY ``openai.RateLimitError`` — this covers both hard quota
    exhaustion (``insufficient_quota``, non-retryable) and transient
    too-many-requests rate limiting (retryable); the two are not
    distinguished. Also matches duck-typed equivalents — a
    ``status_code == 429`` attribute (regardless of code/message), or
    ``'insufficient_quota'`` appearing in the error's ``code`` attribute —
    so callers (get_entity's degraded fallback) can classify this condition
    without a hard dependency on the openai SDK's exact exception hierarchy
    (e.g. if Graphiti wraps/re-raises the error, or a different embedding
    provider is configured). As a last resort, ``'insufficient_quota'``
    appearing in the exception's string message also matches, but ONLY when
    the exception carries neither a ``status_code`` nor a ``code`` attribute
    — a concrete (even if non-matching, e.g. ``status_code=500``) status or
    code classification is trusted over the fuzzy message-substring guess,
    so a wrapped/log-echo error that merely quotes an upstream
    'insufficient_quota' message inside an otherwise-unrelated, already-
    classified error does not get swallowed into the degraded fallback
    (task 2448 review).

    Callers that need to tell hard quota exhaustion apart from a
    retryable transient rate limit must not rely on this predicate alone —
    it deliberately treats both as one condition. get_entity's degraded
    fallback is fine with that because it has no retry loop of its own: on
    a match it degrades immediately rather than retrying, for either cause.

    Never matches a bare BaseException (e.g. ``asyncio.CancelledError``,
    ``KeyboardInterrupt``, ``SystemExit``) — those are structured-concurrency
    shutdown signals, not application-level errors, and must always
    propagate unchanged.
    """
    if not isinstance(exc, Exception):
        return False
    if _OpenAIRateLimitError is not None and isinstance(exc, _OpenAIRateLimitError):
        return True
    status_code = getattr(exc, 'status_code', None)
    if status_code == 429:
        return True
    code = str(getattr(exc, 'code', '') or '')
    if 'insufficient_quota' in code.lower():
        return True
    if status_code is None and not code:
        return 'insufficient_quota' in str(exc).lower()
    return False


def _graphiti_degraded_entity_result() -> dict:
    """Return get_entity's degraded-fallback superset dict.

    Centralizes the {'nodes': [], 'edges': [], 'degraded': True,
    'failed_stores': [...]} shape so get_entity's exact-match and
    fuzzy-fallback degrade sites (see get_entity) cannot drift apart.
    Deliberately NOT shared with search()'s degraded convention
    (memory_service.py's SearchResults / search() failed_stores
    construction): that path returns a SearchResults list-subclass, not a
    plain dict, and the two shapes are intentionally different types.
    """
    return {
        'nodes': [],
        'edges': [],
        'degraded': True,
        'failed_stores': [SourceStore.graphiti.value],
    }


def _degrade_or_reraise(exc: Exception, name: str) -> dict:
    """Classify *exc*: return get_entity's degraded superset dict, or re-raise.

    Shared by get_entity's exact-match and fuzzy-fallback ``except`` blocks
    (task 2448 review) so the classification guard, warning log, and
    degraded-dict return can't drift apart between the two call sites.
    Re-raises *exc* unchanged (same object, so callers still see the
    original type/traceback) when ``_is_rate_limit_or_quota_error(exc)`` is
    False.

    The warning log deliberately does not hardcode "429": the
    message-substring fallback branch of ``_is_rate_limit_or_quota_error``
    can match an error that carries no ``status_code`` attribute at all, so
    asserting 429 unconditionally would misdescribe that classification path
    (task 2448 review). It logs the exception's actual
    ``getattr(exc, 'status_code', None)`` instead.
    """
    if not _is_rate_limit_or_quota_error(exc):
        raise exc
    logger.warning(
        'get_entity degraded: rate-limit/quota error for %r (status_code=%s): %s',
        name,
        getattr(exc, 'status_code', None),
        exc,
    )
    return _graphiti_degraded_entity_result()


def _store_failure_diagnostics(
    store: SourceStore,
    exc: BaseException | None,
    *,
    query: str,
    project_id: str,
    reason: str,
) -> dict:
    """Build a structured failure-diagnostics dict for a degraded search() store.

    Called from search() for both root-cause variants a selected store can hit:
    ``reason='exception'`` when the store's search task raised (any exception other
    than the inner GraphitiBackend.search TimeoutError swallow — see search()'s
    per-task except block), and ``reason='timeout'`` when the store's task was
    still pending when the OUTER ``search_timeout_seconds`` asyncio.wait deadline
    elapsed and was cancelled (there, *exc* is None — there is no exception object,
    only the fact of the timeout).

    This is the diagnosability fix for task 2653: search()'s prior degraded-path
    WARNING carried only ``{'store': ..., 'error': str(e)}`` — no exception type, no
    query shape, no rate-limit/quota classification — which left a recurring
    degradation unattributable even though get_status/`/health` reported the store
    as connected (a query-execution failure, not a connectivity loss). Returns a
    plain dict (not raised, not logged) so callers can both log it and collect it
    into SearchResults.failure_diagnostics without doing either twice.

    Args:
        store: Which store failed.
        exc: The raised exception, or None for the outer-timeout variant (there,
            error_type/error describe the timeout itself rather than a real
            exception object).
        query: The search query text — only its length is recorded (``query_len``),
            not its content, matching the write-journal's existing
            query[:200]-truncation-not-full-body convention.
        project_id: The project scope the search ran under. Deliberately embedded
            in every per-store dict (even though one search() call shares a single
            project_id across all its diagnostics) so each entry is independently
            self-describing — a log shipper or downstream consumer reading one
            failure_diagnostics entry (e.g. off a WARNING's ``extra``) never needs
            to join back against the parent SearchResults or the enclosing
            search() call's scope to know which project it came from.
        reason: ``'exception'`` or ``'timeout'`` — which degrade variant produced
            this diagnostic.

    Returns:
        dict with keys: store, reason, error_type, error, rate_limit_or_quota,
        query_len, project_id.
    """
    return {
        'store': store.value,
        'reason': reason,
        'error_type': type(exc).__name__ if exc is not None else 'TimeoutError',
        'error': (str(exc)[:500] if exc is not None else 'search_timeout'),
        'rate_limit_or_quota': _is_rate_limit_or_quota_error(exc) if exc is not None else False,
        'query_len': len(query),
        # Intentionally repeated per-entry (not deduped onto SearchResults) so
        # each diagnostic stays self-contained for independent log consumption —
        # see the `project_id` Args note above.
        'project_id': project_id,
    }


class SearchResults(list):
    """list subclass returned by MemoryService.search carrying in-band degrade metadata.

    All ~11 internal list-consuming callers (context_assembler, targeted, flag_dedup,
    mem0_dedup, task_knowledge_sync) keep working unchanged — only tools.py reads the
    extra attributes (task 1812).

    Attributes:
        degraded: True when one or more selected stores raised or timed out.
        failed_stores: List of store name strings (SourceStore.value) that failed.
        failure_diagnostics: List of structured failure-diagnostic dicts (task 2653),
            one per failed store — see _store_failure_diagnostics. Empty when
            degraded is False.

    .. warning::
        The `degraded`, `failed_stores`, and `failure_diagnostics` metadata do
        **not** survive list-returning operations (slicing, sorted(), concatenation,
        list comprehensions).  Those operations return a plain ``list``, silently
        dropping the degrade metadata. Callers that need the metadata after a
        transform should read the attributes *before* the transform, or pass the
        SearchResults object directly without intermediate list operations.
    """

    def __init__(
        self,
        iterable=(),
        *,
        degraded: bool = False,
        failed_stores=None,
        failure_diagnostics=None,
    ):
        super().__init__(iterable)
        self.degraded = degraded
        self.failed_stores: list[str] = failed_stores if failed_stores is not None else []
        self.failure_diagnostics: list[dict] = (
            failure_diagnostics if failure_diagnostics is not None else []
        )


@dataclass
class ReconcileStats:
    """Aggregated counts from one ``_reconcile_episode_identity`` run.

    Returned to the caller and logged for observability — NOT wired into the
    durable write-journal schema (extending that schema is out of scope for
    task 2202 / W6-β). Each field mirrors the int return of the
    correspondingly-named post-write sweep — including
    ``stale_ttl_edges_invalidated`` (task 2319), the under-invalidation-
    direction counterpart of ``sibling_edges_restored``. ``errors`` collects
    the label of any sub-pass that raised (task 2202 step-4's best-effort
    guard); an all-zeros, empty-errors instance signals a fully-converged,
    idempotent reconcile.
    """

    edges_deduped: int = 0
    dependency_edges_restored: int = 0
    sibling_edges_restored: int = 0
    stale_ttl_edges_invalidated: int = 0
    nodes_resolved: int = 0
    task_names_normalized: int = 0
    errors: list[str] = field(default_factory=list)


class DescendantScan(NamedTuple):
    """What a cascade WOULD destroy — and whether that answer is complete.

    ``truncated`` is not decoration: a read-only walk cannot page past
    ``MemoryService._CHILD_SCAN_LIMIT``, so the id list can genuinely be a
    subset. Carrying that as data forces a caller gating an irreversible
    multi-record delete to decide what to do about "I could not see all of
    them" instead of reading a partial set as complete.
    """

    ids: list[str]
    truncated: bool


class MemoryService:
    """Central orchestration — fused read/write across Graphiti + Mem0."""

    def __init__(self, config: FusedMemoryConfig):
        self.config = config
        self.graphiti = GraphitiBackend(config)
        self.mem0 = Mem0Backend(config)
        self.classifier = WriteClassifier(config)
        self.router = ReadRouter(config)
        self.durable_queue: DurableWriteQueue | None = None
        self._event_buffer: EventBuffer | None = None
        self._write_journal: WriteJournal | None = None
        self.taskmaster: TaskBackendProtocol | None = None
        self.planned_episode_registry: PlannedEpisodeRegistry | None = None
        self.recon_ledger: ReconLedgerStore | None = None
        # {project_id: project_root} registry snapshot (task 3088). Injected by
        # set_known_projects at server startup — MemoryService is constructed
        # before build_known_projects_map runs, so it cannot arrive by
        # constructor. Used to resolve an escalation queue's filesystem root
        # from the project_id update_memory carries.
        self._known_projects: dict[str, str] = {}
        # INV-4 storm escape for update_memory's silent-rewrite primitive (task
        # 3088). Both are constructed UNCONDITIONALLY — never obtained from
        # ReconciliationHarness (built behind `if config.reconciliation ...
        # enabled:` in server/main.py) or curator_escalator (built after this
        # service). An alarm bound to either would vanish in exactly the
        # degraded configuration where an unattended rewrite loop is least
        # likely to be noticed any other way.
        self._mem0_update_storm_counters: dict[str, StormCounter] = {}
        self._mem0_update_storm_escalator = Mem0UpdateStormEscalator()
        # Test seam for the injectable-clock convention: a 3600s window has to
        # be exercised by advancing a fake clock, not by sleeping.
        self._mem0_update_storm_time_provider: Callable[[], float] = time.time
        # Process-start baselines for uptime reporting
        self._started_at: datetime = datetime.now(UTC)
        self._start_monotonic: float = time.monotonic()
        # Mem0 metadata unknown-key storm detector (task 3195, leaf β).
        # Constructed ONCE so warn counts survive across writes but never leak
        # between processes; a per-write detector would reset its window every
        # time and could never reach the threshold, leaving the escape hatch
        # as dead code that still looked wired up. This is also why the two
        # storm-tuning config leaves are restart-only rather than
        # hot-reloadable — see MemoryMetadataConfig's docstring.
        self._metadata_storm_detector = UnknownKeyStormDetector(
            threshold=config.memory_metadata.unknown_key_storm_threshold,
            window_seconds=config.memory_metadata.unknown_key_storm_window_seconds,
        )

    def _memory_metadata_project_root(self) -> str:
        """Project root the unknown-key storm escalation would be filed into.

        Reuses the established resolution (``config.taskmaster.project_root``
        or empty, see ``reconciliation/harness.py`` and ``server/main.py``)
        rather than introducing a second notion of "this project's root".
        Returns ``''`` when unconfigured, which the caller treats as
        "census only, do not escalate" — there is no queue to file into.
        """
        raw = self.config.taskmaster.project_root if self.config.taskmaster else ''
        return os.path.expanduser(raw) if raw else ''

    def set_event_buffer(self, buffer: EventBuffer) -> None:
        """Wire the reconciliation event buffer into the service."""
        self._event_buffer = buffer

    def set_write_journal(self, journal: WriteJournal) -> None:
        """Wire the write journal for durable auditing."""
        self._write_journal = journal

    def set_planned_registry(self, registry: PlannedEpisodeRegistry) -> None:
        """Wire the planned episode registry into the service."""
        self.planned_episode_registry = registry

    def set_recon_ledger(self, store: ReconLedgerStore) -> None:
        """Wire the recon ledger store into the service."""
        self.recon_ledger = store

    def set_known_projects(self, known_projects: Mapping[str, str] | None) -> None:
        """Wire the ``{project_id: project_root}`` registry snapshot (task 3088).

        The same snapshot ``server/main.py`` builds with
        ``build_known_projects_map`` and already hands to ``ReconciliationHarness``
        and ``TicketJanitor``. Injected rather than derived so this stays pure
        data with no lifetime coupling to any conditionally-constructed
        component — the ``update_memory`` storm alarm must keep working with
        reconciliation disabled, which is exactly the degraded configuration
        where an unattended rewrite loop is least likely to be noticed.

        Copied on entry so a later mutation of the caller's map cannot change
        resolution out from under an in-flight escalation.
        """
        self._known_projects = dict(known_projects or {})
        # Forward to the storm escalator, which is where project_id →
        # project_root resolution actually happens.
        self._mem0_update_storm_escalator.set_known_projects(self._known_projects)

    async def _emit_event(self, event: ReconciliationEvent) -> None:
        if self._event_buffer:
            await self._event_buffer.push(event)

    async def initialize(self) -> None:
        """Initialize backends and the durable write queue."""
        await self.graphiti.initialize()

        qcfg = self.config.queue

        # Initialize planned episode registry (co-located with durable queue data).
        # If set_planned_registry() was called before initialize(), honour the external
        # registry instead of creating a new one (preventing the lifecycle conflict).
        if self.planned_episode_registry is None:
            from fused_memory.services.planned_episode_registry import PlannedEpisodeRegistry
            self.planned_episode_registry = PlannedEpisodeRegistry(data_dir=qcfg.data_dir)
            await self.planned_episode_registry.initialize()

        self.durable_queue = DurableWriteQueue(
            data_dir=qcfg.data_dir,
            execute_write=self._execute_durable_write,
            workers_per_group=qcfg.workers_per_group,
            semaphore_limit=qcfg.semaphore_limit,
            max_attempts=qcfg.max_attempts,
            retry_base_seconds=qcfg.retry_base_seconds,
            retry_max_delay_seconds=qcfg.retry_max_delay_seconds,
            write_timeout_seconds=qcfg.write_timeout_seconds,
            transient_max_attempts=qcfg.transient_max_attempts,
            transient_error_names=qcfg.transient_error_names,
            on_terminal=self._record_queue_terminal_outcome,
        )
        self.durable_queue.register_callback(
            'dual_write_episode', self._dual_write_callback
        )
        self.durable_queue.register_callback(
            'refresh_entity_summaries', self._refresh_summaries_callback
        )
        await self.durable_queue.initialize()

        logger.info('MemoryService initialized')

    async def _safe_close(
        self, label: str, resource: Any, timeout: float | None = None
    ) -> None:
        """Close one resource, time-boxed and logging any failure without re-raising.

        Each close is bounded by ``timeout`` (default: the module-level
        ``_SUBCLOSE_TIMEOUT``, read at call time so it can be monkeypatched) so a
        hung network-driver teardown can neither consume the whole shutdown budget
        nor starve the durable-flush closes that run after it. The bound applies
        UNIFORMLY to every resource, including the durable-flush SQLite closes
        (durable_queue/write_journal/event_buffer), not just the graphiti/mem0
        network drivers — see the cancel-branch comment below for why hard-bounding
        (and, on overrun, cancelling) even a durable close drops no committed data.
        Both the timeout and the generic-failure paths swallow the error so close()
        continues through every resource.
        """
        if timeout is None:
            timeout = _SUBCLOSE_TIMEOUT
        start = time.monotonic()
        try:
            close_task = asyncio.ensure_future(resource.close())
        except Exception:
            # resource.close() raised synchronously, or returned a
            # non-awaitable (e.g. a bare Mock in tests, or a driver whose
            # close() is a plain function). Match the pre-time-boxing
            # `await resource.close()` behaviour: log and swallow so close()
            # continues through every remaining resource.
            logger.exception('MemoryService.close: %s.close failed', label)
            return
        # asyncio.wait signals a budget overrun structurally (via `pending`),
        # never by raising — so a TimeoutError the resource's own close() raises
        # stays a generic failure (ERROR) instead of being misread as an overrun.
        _, pending = await asyncio.wait({close_task}, timeout=timeout)
        if pending:
            # Budget expired: the close is still running. Cancel it and move on
            # so a hung backend can't starve the durable-flush closes that follow.
            # Cancelling is data-safe for EVERY resource this bounds, not only the
            # network drivers:
            #  * graphiti/mem0 closes only tear down sockets; their writes are
            #    already durable via durable_queue / synchronous commits, and the
            #    OS reclaims the sockets on process exit.
            #  * durable_queue/write_journal/event_buffer are aiosqlite (WAL mode,
            #    full-durability pragmas): every row commits synchronously during
            #    normal operation, so it is durable in the WAL BEFORE close() runs.
            #    Their close() only drains already-persisted in-flight work and runs
            #    a best-effort wal_checkpoint(TRUNCATE) + connection close — pure
            #    housekeeping. Cancelling abandons that compaction, never a committed
            #    row (SQLite replays the WAL on next open); and aiosqlite's
            #    per-connection worker thread finishes any in-flight statement rather
            #    than tearing it mid-write.
            # The WARNING names the label + budget so the restart journal identifies
            # the culprit.
            close_task.cancel()
            with contextlib.suppress(BaseException):
                await close_task
            logger.warning(
                'MemoryService.close: %s.close exceeded %.1fs budget '
                '(elapsed %.2fs) — cancelled to avoid starving later closes',
                label,
                timeout,
                time.monotonic() - start,
            )
            return
        try:
            close_task.result()
        except Exception:
            logger.exception('MemoryService.close: %s.close failed', label)
        else:
            logger.debug(
                'MemoryService.close: %s.close completed in %.2fs',
                label,
                time.monotonic() - start,
            )

    async def close(self) -> None:
        if self.durable_queue:
            await self._safe_close('durable_queue', self.durable_queue)
        await self._safe_close('graphiti', self.graphiti)
        await self._safe_close('mem0', self.mem0)
        if self._write_journal:
            await self._safe_close('write_journal', self._write_journal)
        if self._event_buffer:
            await self._safe_close('event_buffer', self._event_buffer)
        if self.planned_episode_registry:
            await self._safe_close('planned_episode_registry', self.planned_episode_registry)

    # ------------------------------------------------------------------
    # Journal helper
    # ------------------------------------------------------------------

    async def _journaled_backend_call(
        self,
        write_op_id: str | None,
        causation_id: str | None,
        backend: str,
        operation: str,
        payload: dict[str, Any],
        coro: Any,
    ) -> Any:
        """Execute a backend call and log to write journal."""
        result = None
        try:
            result = await coro
            if self._write_journal:
                await self._write_journal.log_backend_op(
                    write_op_id=write_op_id,
                    causation_id=causation_id,
                    backend=backend,
                    operation=operation,
                    payload=payload,
                    result_summary=str(result)[:500] if result else None,
                    success=True,
                )
            return result
        except Exception as e:
            if self._write_journal:
                await self._write_journal.log_backend_op(
                    write_op_id=write_op_id,
                    causation_id=causation_id,
                    backend=backend,
                    operation=operation,
                    payload=payload,
                    success=False,
                    error=str(e),
                )
            raise

    async def _record_queue_terminal_outcome(
        self,
        write_op_id: str,
        terminal_status: str,
        error: str | None = None,
    ) -> None:
        """Write a durable-queue terminal outcome back onto its write_op row.

        Passed to ``DurableWriteQueue(on_terminal=...)`` in ``initialize()``.
        Because it lives at the queue seam, every enqueue site inherits the
        write-back: ``add_episode``, ``add_memory``'s Graphiti leg, and the
        retained ``mem0_add`` dispatcher alike.

        This is a BOUND METHOD rather than a captured ``WriteJournal``
        reference so ``self._write_journal`` is resolved at CALL time — the
        same lazy-collaborator idiom as ``execute_write=self._execute_durable_write``.
        It has to be: ``server/main.py`` calls ``memory_service.initialize()``
        (which constructs the queue) BEFORE ``set_write_journal()``, so the
        journal is still ``None`` when the hook is wired.
        """
        journal = self._write_journal
        if journal is None:
            return
        await journal.record_terminal_outcome(
            write_op_id=write_op_id,
            terminal_status=terminal_status,
            terminal_error=error,
        )

    @staticmethod
    def _mem0_payload_digest(
        content: str,
        project_id: str | None,
        agent_id: str | None,
        session_id: str | None,
        category: str | None,
        metadata: dict | None,
    ) -> str:
        """sha256 over the canonical mem0 write payload — audit/idempotency key.

        Used to stamp the write-ahead ``mem0_intent`` (task 2710) so a
        dead-lettered intent carries a stable fingerprint of exactly what
        would have been written, for audit and manual replay.
        """
        canonical = json.dumps(
            {
                'content': content,
                'project_id': project_id,
                'agent_id': agent_id,
                'session_id': session_id,
                'category': category,
                'metadata': metadata or {},
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    # ------------------------------------------------------------------
    # Durable queue: execute write dispatcher
    # ------------------------------------------------------------------

    async def _execute_durable_write(
        self, operation: str, payload: dict[str, Any]
    ) -> Any:
        """Route a queued write to the appropriate backend handler."""
        if operation == 'mem0_add':
            return await self._execute_mem0_write(payload)
        if operation == 'mem0_classify_and_add':
            return await self._execute_mem0_classify_and_add(payload)
        return await self._execute_graphiti_write(operation, payload)

    # ------------------------------------------------------------------
    # Dedup: remove duplicate edges created by a single add_episode call
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_fact(text: str) -> str:
        """Normalize a fact string for dedup comparison.

        Replicates graphiti_core's internal normalization:
        lowercase + collapse whitespace.  Two facts that differ only in
        capitalisation or spacing are treated as the same edge.
        """
        return re.sub(r'\s+', ' ', text.lower()).strip()

    async def _dedup_episode_edges(self, result: Any, *, group_id: str) -> int:
        """Remove duplicate edges produced by a single add_episode call.

        Graphiti's LLM extraction pipeline can emit multiple edges that
        express the same fact (same source_node_uuid, target_node_uuid,
        and normalised fact text).  This method groups the edges returned
        in *result* by that triple and deletes all but the first edge in
        each group via ``bulk_remove_edges``.

        Args:
            result: The value returned by ``add_episode`` (typically an
                    AddEpisodeResults object with an ``edges`` attribute).
                    Handles ``None`` and objects with empty/missing edges
                    gracefully.

        Returns:
            Number of duplicate edges removed (0 when nothing to do).
        """
        if result is None:
            return 0

        edges = getattr(result, 'edges', None) or getattr(result, 'entity_edges', None) or []
        if not edges:
            return 0

        # Group edges by (source_node_uuid, target_node_uuid, normalized_fact)
        seen: dict[tuple[str, str, str], str] = {}   # key → first uuid
        duplicates: list[str] = []

        for edge in edges:
            src_uuid = getattr(edge, 'source_node_uuid', '') or ''
            tgt_uuid = getattr(edge, 'target_node_uuid', '') or ''
            fact_norm = self._normalize_fact(getattr(edge, 'fact', '') or '')
            edge_uuid = getattr(edge, 'uuid', '') or ''
            key = (src_uuid, tgt_uuid, fact_norm)

            if key in seen:
                duplicates.append(edge_uuid)
            else:
                seen[key] = edge_uuid

        if not duplicates:
            return 0

        logger.info('Deduplicating %d edge(s) after add_episode', len(duplicates))
        return await self.graphiti.bulk_remove_edges(duplicates, group_id=group_id)

    async def _dedup_episode_nodes(self, result: Any, *, group_id: str) -> int:
        """Resolve/collapse exact-name duplicate entity nodes touched by one add_episode call.

        graphiti_core's ingestion-time entity resolution (resolve_extracted_nodes)
        only resolves each extracted node against candidates surfaced by hybrid
        embedding+BM25 search — there is no guaranteed exact-name Cypher lookup.
        When that search misses an existing canonical node, ingestion mints a
        brand-new node even though a node with the exact same name already
        exists. This sweep re-checks each entity name this episode touched by
        delegating to α's (task 2198) write-time-identity chokepoint
        ``GraphitiBackend._resolve_or_create_entity``, which resolves a lone
        match, no-ops on zero matches, and collapses >=2 matches into a
        canonical survivor via ``merge_entities``.

        Modelled on ``_dedup_episode_edges``; handles None / empty result the
        same way. Each resolve is best-effort (mirrors
        ``_restore_superseded_dependency_edges``): a transient backend error
        resolving one name must not fail an already-committed episode write,
        and must not stop subsequent names from being processed. Unresolved
        names simply survive to be healed the next time an episode touches
        that name.

        Callers (``_reconcile_episode_identity``) MUST run this under α's
        ``_identity_lock_for(group_id)`` — ``_resolve_or_create_entity``
        performs no locking of its own.

        Args:
            result: The value returned by ``add_episode`` (typically an
                    AddEpisodeResults object with a ``nodes`` attribute).
                    Handles ``None`` and objects with empty/missing nodes
                    gracefully.

        Returns:
            Number of distinct names that resolved to a non-None uuid (0
            when nothing to do). This is a resolve count, not a merge
            count — a name can resolve without any collapse having been
            necessary.
        """
        if result is None:
            return 0

        nodes = getattr(result, 'nodes', None) or []
        if not nodes:
            return 0

        names: list[str] = []
        seen: set[str] = set()
        for node in nodes:
            name = getattr(node, 'name', '') or ''
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)

        resolved = 0
        failed = 0
        for name in names:
            try:
                uuid = await self.graphiti._resolve_or_create_entity(name, group_id=group_id)
                if uuid is not None:
                    resolved += 1
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                # Best-effort: a transient backend error (lock contention,
                # write timeout) must not fail an already-committed episode
                # write.  Log and continue so the episode reports success.
                logger.exception(
                    'Failed to resolve exact-name entity %r after add_episode; '
                    'will retry on next episode',
                    name,
                )
                failed += 1

        if resolved > 0:
            logger.info('Resolved %d exact-name entity/entities after add_episode', resolved)
        if failed > 0:
            logger.warning(
                'Failed to resolve %d exact-name entity/entities after add_episode',
                failed,
            )
        return resolved

    async def _restore_superseded_dependency_edges(
        self, result: Any, *, group_id: str
    ) -> int:
        """Undo false dependency-edge invalidations caused by LLM edge-resolution.

        Graphiti's upstream ``add_episode`` LLM pipeline can falsely supersede
        existing "X depends on Y" edges when a new dependency fact is added for
        a hub entity (dependencies are additive, so supersession is wrong for
        them). This method scans the edges returned in *result* — exactly the
        edges this episode touched — and clears ``invalid_at`` for any that are
        both (a) invalidated AND (b) express a dependency fact.

        Modelled on ``_dedup_episode_edges``; handles None / empty result the
        same way. Legitimate dependency removals flow through
        ``remove_dependency`` (a different code path) and are unaffected.

        Args:
            result: The value returned by ``add_episode`` (typically an
                    AddEpisodeResults object with an ``edges`` attribute).
                    Handles ``None`` and objects with empty/missing edges
                    gracefully.

        Returns:
            Number of dependency edges whose invalidation was reversed (0 when
            nothing to do).
        """
        if result is None:
            return 0

        edges = (
            getattr(result, 'edges', None)
            or getattr(result, 'entity_edges', None)
            or []
        )
        if not edges:
            return 0

        restored = 0
        failed = 0
        for edge in edges:
            if getattr(edge, 'invalid_at', None) is None:
                continue
            fact = getattr(edge, 'fact', '') or ''
            if not _is_dependency_fact(fact):
                continue
            edge_uuid = getattr(edge, 'uuid', '') or ''
            try:
                await self.graphiti.update_edge(
                    edge_uuid, group_id=group_id, clear_invalid_at=True
                )
                restored += 1
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                # Best-effort: a transient backend error (lock contention,
                # write timeout) must not fail an already-committed episode
                # write.  Log and continue so the episode reports success.
                logger.exception(
                    'Failed to restore dependency edge %s; will retry on next episode',
                    edge_uuid,
                )
                failed += 1

        if restored > 0:
            logger.info(
                'Restored %d falsely-superseded dependency edge(s) after add_episode',
                restored,
            )
        if failed > 0:
            logger.warning(
                'Failed to restore %d dependency edge(s) after add_episode',
                failed,
            )
        return restored

    async def _restore_falsely_superseded_sibling_edges(
        self, result: Any, *, group_id: str
    ) -> int:
        """Undo false sibling-edge invalidations caused by LLM edge-resolution.

        Graphiti's upstream ``add_episode`` LLM pipeline can falsely supersede
        pre-existing edges that merely SHARE an entity node with a newly-added
        fact, even when the new fact does not contradict them (an
        over-triggering "most-recent-edge-wins per node" heuristic). A
        legitimate temporal supersession always replaces a fact between the
        SAME two entities with a newer fact between those same two entities,
        so graphiti both sets ``invalid_at`` on the old edge AND adds a new
        valid edge on that same node-pair. This method scans the edges
        returned in *result* — exactly the edges this episode touched — and
        clears ``invalid_at`` for any invalidated edge whose
        ``(source_node_uuid, target_node_uuid)`` pair is NOT restated by any
        surviving (``invalid_at is None``) edge in this same write: such an
        edge cannot be a legitimate supersession, so its invalidation is
        provably the sibling-invalidation bug.

        Known limitation: an invalidated edge whose node-pair IS restated by
        a surviving valid edge (a same-node-pair invalidation) is left
        untouched, since it may be a legitimate contradiction the LLM
        correctly resolved — that case is left to the LLM's judgment
        (documented follow-up, not handled here). The opposite-direction
        gap — a pre-existing edge the LLM failed to invalidate at all,
        left coexisting with a fresh contradictory edge — is handled for
        the narrow priority-override/TTL fact-shape by
        ``_invalidate_stale_superseded_ttl_edges`` (task 2319); it remains
        unhandled here for other fact shapes.

        Modelled on ``_restore_superseded_dependency_edges``; handles None /
        empty result the same way. Dependency-fact edges (``_is_dependency_fact``)
        are excluded from restore candidates here even when their node-pair
        is unrestated: ``_restore_superseded_dependency_edges`` already
        restores every invalidated dependency edge unconditionally
        (regardless of node-pair) earlier in the same post-write chain, so
        this hook would otherwise issue a redundant second ``update_edge``
        call for the same edge. This keeps the two hooks' contracts
        non-overlapping without relying on any in-place mutation of *result*
        (the edge objects returned by ``add_episode`` are not refreshed by a
        subsequent ``update_edge`` call, so ``invalid_at`` here always
        reflects the pre-restore state regardless of hook order).

        Args:
            result: The value returned by ``add_episode`` (typically an
                    AddEpisodeResults object with an ``edges`` attribute).
                    Handles ``None`` and objects with empty/missing edges
                    gracefully.

        Returns:
            Number of sibling edges whose false invalidation was reversed (0
            when nothing to do).
        """
        if result is None:
            return 0

        edges = (
            getattr(result, 'edges', None)
            or getattr(result, 'entity_edges', None)
            or []
        )
        if not edges:
            return 0

        restated_pairs: set[tuple[str, str]] = set()
        for edge in edges:
            if getattr(edge, 'invalid_at', None) is not None:
                continue
            src = getattr(edge, 'source_node_uuid', '') or ''
            tgt = getattr(edge, 'target_node_uuid', '') or ''
            restated_pairs.add((src, tgt))

        restored = 0
        failed = 0
        for edge in edges:
            if getattr(edge, 'invalid_at', None) is None:
                continue
            if _is_dependency_fact(getattr(edge, 'fact', '') or ''):
                # Exclusively _restore_superseded_dependency_edges's domain —
                # it already restores every invalidated dependency edge
                # unconditionally, earlier in the same post-write chain.
                continue
            src = getattr(edge, 'source_node_uuid', '') or ''
            tgt = getattr(edge, 'target_node_uuid', '') or ''
            if (src, tgt) in restated_pairs:
                # Same-node-pair invalidation: possibly a legitimate
                # contradiction. Leave to the LLM's judgment.
                continue
            edge_uuid = getattr(edge, 'uuid', '') or ''
            try:
                await self.graphiti.update_edge(
                    edge_uuid, group_id=group_id, clear_invalid_at=True
                )
                restored += 1
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                # Best-effort: a transient backend error (lock contention,
                # write timeout) must not fail an already-committed episode
                # write.  Log and continue so the episode reports success.
                logger.exception(
                    'Failed to restore sibling edge %s; will retry on next episode',
                    edge_uuid,
                )
                failed += 1

        if restored > 0:
            logger.info(
                'Restored %d falsely-superseded sibling edge(s) after add_episode',
                restored,
            )
        if failed > 0:
            logger.warning(
                'Failed to restore %d sibling edge(s) after add_episode',
                failed,
            )
        return restored

    async def _invalidate_stale_superseded_ttl_edges(
        self, result: Any, *, group_id: str
    ) -> int:
        """Invalidate pre-existing stale priority-override scalar edges left
        behind when Graphiti's upstream LLM edge-resolver under-invalidates.

        Covers every predicate class recognized by
        ``_priority_override_scalar_predicates`` — the TTL scalar (task 2319)
        and the reserve_now boolean scalar (task 2351, follow-up to 2319 via
        esc-2319-8) — since both are genuinely single-valued-per-subject
        fields on the scheduler override model and share the identical
        under-invalidation risk. The two classes are tracked and matched
        SEPARATELY throughout (see steps 1 and 3 below); this hook never
        collapses them via the ``_is_priority_override_scalar_fact`` union
        matcher, which would re-invalidate a still-valid TTL edge on a fresh
        reserve_now write (or vice versa). The rest of this docstring refers
        to "priority-override/TTL" for historical continuity with task 2319,
        but every step applies equally to reserve_now facts.

        Mirror-image (under-invalidation direction) counterpart of
        ``_restore_falsely_superseded_sibling_edges`` (task 2111), which
        fixes the OVER-invalidation direction and documents this exact gap
        in its "Known limitation" note. Graphiti's ``dedupe_edges``/
        ``resolve_edge_contradictions`` pipeline can fail to set
        ``invalid_at`` on a pre-existing edge that a freshly-written edge
        contradicts, leaving two-or-more ``invalid_at is None`` edges
        asserting contradictory values for the same single-valued scalar
        (task 2265's TTL incident: 10800 and 86400 both valid for ~2h10m).

        Unlike the sibling hook, which scans only ``result``'s edges (because
        graphiti TOUCHES the edges it falsely invalidates, so they carry the
        fresh ``invalid_at`` and appear in the result), this hook must
        re-query live graph state: the pre-existing stale edge was NOT
        touched by this episode, so it is generally absent from
        ``result.edges``. It therefore:

        1. Scans *result* for the "authoritative fresh" set — valid
           (``invalid_at is None``) edges whose
           ``_priority_override_scalar_predicates`` predicate-class set is
           non-empty — and records, per subject, which classes it wrote. If
           no edge yields a non-empty set, returns 0 without any graph query
           — this scopes the hook to fire only when the current episode
           actually wrote a priority-override/TTL or reserve_now fact.
        2. For each distinct SUBJECT (``source_node_uuid``) of an
           authoritative-fresh edge — never the object/target node — queries
           ``graphiti.get_valid_edges_for_node`` for every currently-valid
           edge on that node — including pre-existing stale edges this
           episode never touched. The target/object of a TTL fact is a
           generic value/concept node ("TTL", "X seconds") shared across
           every task that ever mentioned a TTL; because
           ``get_valid_edges_for_node`` is UNDIRECTED, querying that shared
           node would return — and this hook would then invalidate — the
           priority-override/TTL edges of OTHER subjects, silently destroying
           valid, current facts belonging to unrelated entities (the
           cross-entity over-invalidation failure mode this hook must NOT
           reintroduce). The subject Task node is on BOTH the stale and the
           fresh edge, so querying it alone still catches the genuine
           same-subject stale edge; the target query is both harmful and
           unnecessary. This is what enforces the docstring invariant that
           only *same-subject* single-valued scalars are superseded.
        3. Invalidates every returned edge that is not itself one of the
           authoritative-fresh edges AND whose own
           ``_priority_override_scalar_predicates`` classes INTERSECT the
           fresh classes recorded for that same subject in step 1: it is a
           same-subject, SAME-PREDICATE contradiction of the fact just
           written. A candidate of a DIFFERENT scalar class — e.g. a
           still-valid TTL edge on a subject whose fresh write this episode
           was reserve_now — does not intersect and is left untouched, even
           though both are "priority-override scalar" facts; this
           intersection check is what keeps the two classes from being
           collapsed into one (see ``_priority_override_scalar_predicates``).
           The invalidation timestamp is computed PER SUBJECT — the newest
           ``valid_at`` among *that subject's own* authoritative-fresh
           edges (falling back to ``datetime.now(UTC)`` only when that
           subject's fresh edge(s) carried no ``valid_at``), so the stale
           fact's supersession is stamped as of the moment that subject's
           own new fact became valid. A single episode may write fresh TTL
           facts for two different subjects at different ``valid_at``
           times; using a single global max across every subject instead
           would stamp an earlier subject's stale edge with a later
           subject's timestamp, reopening a same-subject overlap window
           between the two — the exact defect this hook exists to close.
           A processed-uuid set deduplicates edges reachable from more than
           one subject node (an undirected per-node query returns an edge
           spanning two queried subjects under both endpoints), so each
           stale edge is invalidated at most once. Each invalidation attempt is
           individually best-effort — mirroring the sibling hook's per-edge
           guard — so a transient backend failure for one stale edge is
           logged and counted but does not stop the remaining edges from
           being processed or fail the already-committed episode.

        This enforces the invariant that the newest write for a given
        (entity, predicate)-shape is the only ``invalid_at is None`` edge of
        that shape. It is idempotent: a re-run sees only the single
        surviving fresh edge and no-ops. It structurally cannot invalidate
        the fresh edge itself, since authoritative-fresh uuids are excluded
        from the invalidation candidates.

        Args:
            result: The value returned by ``add_episode`` (typically an
                    AddEpisodeResults object with an ``edges`` attribute).
                    Handles ``None`` and objects with empty/missing edges
                    gracefully.
            group_id: The project graph to query/write.

        Returns:
            Number of stale priority-override/TTL edges invalidated (0 when
            nothing to do).
        """
        if result is None:
            return 0

        edges = (
            getattr(result, 'edges', None)
            or getattr(result, 'entity_edges', None)
            or []
        )
        if not edges:
            return 0

        keep_uuids: set[str] = set()
        subject_node_uuids: set[str] = set()
        # PER-SUBJECT predicate classes freshly written by THIS episode. A
        # candidate stale edge is invalidated only when its own predicate
        # class(es) intersect the fresh classes recorded for its subject —
        # NOT merely because it matches the union scalar matcher. TTL and
        # reserve_now legitimately coexist on one subject; a fresh reserve_now
        # write must not invalidate a still-valid TTL edge (and vice versa).
        subject_predicates: dict[str, set[str]] = {}
        # PER-SUBJECT supersession stamps, not a single global max: see the
        # docstring's step 3 for why using one global max across every
        # subject touched by this episode would be imprecise when the
        # episode writes fresh TTL facts for multiple distinct subjects at
        # different valid_at times.
        subject_stamps: dict[str, datetime] = {}
        for edge in edges:
            if getattr(edge, 'invalid_at', None) is not None:
                continue
            fresh_predicates = _priority_override_scalar_predicates(
                getattr(edge, 'fact', '') or ''
            )
            if not fresh_predicates:
                continue
            edge_uuid = getattr(edge, 'uuid', '') or ''
            if edge_uuid:
                keep_uuids.add(edge_uuid)
            # SUBJECT-SCOPED: collect ONLY the subject/source node, never the
            # object/target. For a fact "Task N priority override TTL of X
            # seconds" the source is the subject Task node and the target is a
            # generic value/concept node ("TTL", "X seconds") shared across
            # every task that ever mentioned a TTL. get_valid_edges_for_node is
            # UNDIRECTED, so querying that shared target node would return the
            # priority-override/TTL edges of OTHER subjects and invalidate their
            # valid, current facts — the cross-entity over-invalidation failure
            # mode. The subject Task node is on BOTH the stale and the fresh
            # edge, so querying it alone still catches the genuine same-subject
            # stale edge; the target query is both harmful and unnecessary.
            src = getattr(edge, 'source_node_uuid', '') or ''
            valid_at = getattr(edge, 'valid_at', None)
            if src:
                subject_node_uuids.add(src)
                subject_predicates.setdefault(src, set()).update(fresh_predicates)
                if valid_at is not None and (
                    src not in subject_stamps or valid_at > subject_stamps[src]
                ):
                    subject_stamps[src] = valid_at

        if not keep_uuids:
            # This episode did not write a fresh priority-override/TTL fact —
            # nothing to supersede, and no graph query needed.
            return 0

        invalidated = 0
        failed = 0
        processed_uuids: set[str] = set()
        for node_uuid in subject_node_uuids:
            candidates = await self.graphiti.get_valid_edges_for_node(
                node_uuid, group_id=group_id,
            )
            # This subject's own stamp — the newest valid_at among its
            # authoritative-fresh edges, falling back to now(UTC) only when
            # unavailable for THIS subject (never another subject's stamp).
            stamp = subject_stamps.get(node_uuid) or datetime.now(UTC)
            # Predicate classes freshly written for THIS subject. Only a
            # candidate carrying one of these same classes is a genuine
            # supersession; a candidate of a different scalar class (e.g. a
            # valid TTL edge when the fresh fact was reserve_now) coexists
            # legitimately and must be left untouched.
            fresh_classes = subject_predicates.get(node_uuid, set())
            for candidate in candidates:
                candidate_uuid = candidate.get('uuid', '') or ''
                if not candidate_uuid or candidate_uuid in keep_uuids:
                    continue
                candidate_classes = _priority_override_scalar_predicates(
                    candidate.get('fact', '') or ''
                )
                if not (candidate_classes & fresh_classes):
                    continue
                if candidate_uuid in processed_uuids:
                    # Undirected per-node query: an edge shared between two
                    # touched nodes is returned once per endpoint. Invalidate
                    # it at most once.
                    continue
                processed_uuids.add(candidate_uuid)
                try:
                    await self.graphiti.update_edge(
                        candidate_uuid, group_id=group_id, invalid_at=stamp,
                    )
                    invalidated += 1
                except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    # Best-effort: a transient backend error (lock contention,
                    # write timeout) must not fail an already-committed episode
                    # write.  Log and continue so the episode reports success.
                    logger.exception(
                        'Failed to invalidate stale priority-override/TTL edge %s; '
                        'will retry on next episode',
                        candidate_uuid,
                    )
                    failed += 1

        if invalidated > 0:
            logger.info(
                'Invalidated %d stale superseded priority-override/TTL edge(s) '
                'after add_episode',
                invalidated,
            )
        if failed > 0:
            logger.warning(
                'Failed to invalidate %d stale priority-override/TTL edge(s) after add_episode',
                failed,
            )
        return invalidated

    async def _normalize_task_node_names(self, result: Any, *, group_id: str) -> int:
        """Canonicalize non-canonical task-entity node names to 'Task N'.

        graphiti_core's LLM entity extraction sometimes mints task-entity nodes
        with non-canonical names (e.g. 'task 132', 'tasks 153') instead of the
        canonical 'Task N' form. This method scans each distinct entity name
        this episode touched via ``canonicalize_task_node_name`` (task 2110)
        and, for every name that canonicalizes to something other than itself,
        corrects the live node(s):

        - If a canonical 'Task N' node already exists, every bad-named node is
          MERGED into it via ``merge_entities`` — renaming instead would recreate
          the exact-name duplicate ``_dedup_episode_nodes`` exists to resolve.
          Any *other* pre-existing canonical duplicates (e.g. left over from
          before this hook existed, which this episode never touched) are
          folded into the same survivor too, so a single hook run fully
          collapses the canonical-name group rather than only fixing the
          bad-named arrival.
        - Otherwise, the bad-named survivor (most valid edges, then oldest, then
          uuid — same canonical ordering as ``find_duplicate_entity_nodes``) is
          RENAMED to the canonical name via ``rename_entity_node``, and any
          remaining bad-named duplicates are merged into it.

        Modelled on ``_dedup_episode_nodes``; handles None / empty result the
        same way. Each rename/merge is best-effort (mirrors
        ``_dedup_episode_nodes``): a transient backend error for one name must
        not fail an already-committed episode write, and must not stop
        subsequent names from being processed. Untouched bad names simply
        survive to be healed the next time an episode touches that name.

        Args:
            result: The value returned by ``add_episode`` (typically an
                    AddEpisodeResults object with a ``nodes`` attribute).
                    Handles ``None`` and objects with empty/missing nodes
                    gracefully.

        Returns:
            Number of nodes successfully renamed or merged into a canonical
            survivor (0 when nothing to do).
        """
        if result is None:
            return 0

        nodes = getattr(result, 'nodes', None) or []
        if not nodes:
            return 0

        canonical_by_bad_name: dict[str, str] = {}
        for node in nodes:
            name = getattr(node, 'name', '') or ''
            if not name or name in canonical_by_bad_name:
                continue
            canonical = canonicalize_task_node_name(name)
            if canonical is None or canonical == name:
                continue
            canonical_by_bad_name[name] = canonical

        fixed = 0
        failed = 0
        for bad_name, canonical in canonical_by_bad_name.items():
            try:
                bad_matches = await self.graphiti.find_duplicate_entity_nodes(
                    bad_name, group_id=group_id,
                )
                if not bad_matches:
                    continue
                canon_matches = await self.graphiti.find_duplicate_entity_nodes(
                    canonical, group_id=group_id,
                )
                if canon_matches:
                    canon_survivor = canon_matches[0]['uuid']
                    for dup in bad_matches:
                        await self.graphiti.merge_entities(
                            dup['uuid'], canon_survivor, group_id=group_id,
                        )
                        fixed += 1
                    # Pre-existing canonical duplicates this episode never
                    # touched (e.g. left behind before this hook existed)
                    # would otherwise only get fixed if some future episode
                    # happens to touch them again — fold them in now too.
                    for dup in canon_matches[1:]:
                        await self.graphiti.merge_entities(
                            dup['uuid'], canon_survivor, group_id=group_id,
                        )
                        fixed += 1
                else:
                    survivor = bad_matches[0]['uuid']
                    await self.graphiti.rename_entity_node(
                        survivor, canonical, group_id=group_id,
                    )
                    fixed += 1
                    for dup in bad_matches[1:]:
                        await self.graphiti.merge_entities(
                            dup['uuid'], survivor, group_id=group_id,
                        )
                        fixed += 1
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                # Best-effort: a transient backend error (lock contention,
                # write timeout) must not fail an already-committed episode
                # write. Log and continue so the episode reports success.
                logger.exception(
                    'Failed to normalize task node name %r -> %r after '
                    'add_episode; will retry on next episode',
                    bad_name, canonical,
                )
                failed += 1

        if fixed > 0:
            logger.info(
                'Normalized %d task-entity node name(s) after add_episode', fixed,
            )
        if failed > 0:
            logger.warning(
                'Failed to normalize %d task-entity node name(s) after add_episode',
                failed,
            )
        return fixed

    async def _reconcile_episode_identity(
        self, result: Any, *, group_id: str
    ) -> ReconcileStats:
        """Fold the six post-write identity/dedup sweeps into one call.

        Task 2202 (W6-β): the single reconcile step ``_execute_graphiti_write``
        runs immediately after ``add_episode``, inside α's (task 2198)
        per-group identity lock — making entity identity a write-time
        guarantee instead of a best-effort post-hoc sweep. This obsoletes the
        recurring "duplicate Graphiti node -> manual FalkorDB merge" operator
        runbook (tasks 2073/2081/2110/2118, the /unblock Graphiti-dedup
        protocol): those incidents arose because the sweeps ran outside any
        lock and could race with a concurrent same-group write; folding them
        into one locked reconcile closes that race.

        Runs the six sub-passes in their pre-existing chain order —
        dependency-restore before sibling-restore, matching the ordering
        this replaces at the ``_execute_graphiti_write`` call site (a
        dependency edge must be un-superseded before the sibling-restore
        pass considers it, so it is correctly skipped there rather than
        double-processed) — and aggregates each sub-pass's int return into
        the matching ``ReconcileStats`` field. ``_invalidate_stale_superseded_
        ttl_edges`` (task 2319) runs immediately after
        ``_restore_falsely_superseded_sibling_edges``, grouping the two
        edge-temporal passes together, and before ``_dedup_episode_nodes``:
        it is the mirror-image, under-invalidation-direction counterpart of
        the sibling-restore pass.

        Each sub-pass runs under its own best-effort guard: a generic
        ``Exception`` is logged and recorded as that sub-pass's label in
        ``ReconcileStats.errors`` (leaving its count at 0), and the
        remaining sub-passes still run — a single sub-pass failure must
        never fail the already-committed episode write.
        ``CancelledError``/``KeyboardInterrupt``/``SystemExit`` are never
        swallowed; they propagate immediately and skip any later sub-passes.

        Args:
            result: The value returned by ``add_episode`` (typically an
                AddEpisodeResults object), forwarded verbatim to every
                sub-pass.
            group_id: The project graph this episode was written to.

        Returns:
            A ReconcileStats aggregating every sub-pass's count (and any
            per-sub-pass failure labels).
        """
        stats = ReconcileStats()

        async def _run_pass(label: str, coro: Any) -> int:
            try:
                return await coro
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                # Best-effort: one sub-pass's transient backend error must
                # not fail the others, nor the already-committed episode
                # write. Log, record the label, and continue.
                logger.exception(
                    'Sub-pass %s failed during _reconcile_episode_identity; '
                    'continuing with the remaining sub-passes',
                    label,
                )
                stats.errors.append(label)
                return 0

        stats.edges_deduped = await _run_pass(
            '_dedup_episode_edges',
            self._dedup_episode_edges(result, group_id=group_id),
        )
        stats.dependency_edges_restored = await _run_pass(
            '_restore_superseded_dependency_edges',
            self._restore_superseded_dependency_edges(result, group_id=group_id),
        )
        stats.sibling_edges_restored = await _run_pass(
            '_restore_falsely_superseded_sibling_edges',
            self._restore_falsely_superseded_sibling_edges(result, group_id=group_id),
        )
        stats.stale_ttl_edges_invalidated = await _run_pass(
            '_invalidate_stale_superseded_ttl_edges',
            self._invalidate_stale_superseded_ttl_edges(result, group_id=group_id),
        )
        stats.nodes_resolved = await _run_pass(
            '_dedup_episode_nodes',
            self._dedup_episode_nodes(result, group_id=group_id),
        )
        stats.task_names_normalized = await _run_pass(
            '_normalize_task_node_names',
            self._normalize_task_node_names(result, group_id=group_id),
        )
        return stats

    async def _execute_graphiti_write(
        self, operation: str, payload: dict[str, Any]
    ) -> Any:
        """Dispatch a queued write to the Graphiti backend."""
        source_str = payload.get('source', 'text')
        try:
            episode_type = EpisodeType[source_str]
        except (KeyError, AttributeError):
            episode_type = EpisodeType.text

        # Extract journal metadata from payload (injected at enqueue time)
        causation_id = payload.pop('_causation_id', None)
        write_op_id = payload.pop('_write_op_id', None)
        temporal_context = payload.pop('temporal_context', None)
        reference_time_iso = payload.pop('reference_time', None)
        reference_time = None
        if reference_time_iso is not None:
            try:
                reference_time = datetime.fromisoformat(reference_time_iso)
            except (ValueError, TypeError):
                logger.warning(
                    'Invalid reference_time %r in queue payload; treating as None',
                    reference_time_iso,
                )

        # Write-time identity gate (task 2202 / W6-β): add_episode and the
        # folded post-write reconcile run as ONE critical section under α's
        # (task 2198) per-group_id identity lock, so entity identity is a
        # write-time guarantee rather than a best-effort post-hoc race. This
        # is what obsoletes the recurring "duplicate Graphiti node -> manual
        # FalkorDB merge" operator runbook (tasks 2073/2081/2110/2118, the
        # /unblock Graphiti-dedup protocol). This method is the fallthrough
        # dispatch target in _execute_durable_write (:474) for every queued
        # operation other than 'mem0_add'/'mem0_classify_and_add' — so both
        # 'add_episode' AND 'add_memory_graphiti' writes acquire this lock,
        # meaning ALL Graphiti writes for a given group_id fully serialize,
        # not just add_episode (intended per B1; distinct group_ids/projects
        # still proceed concurrently). Graphiti-only critical section —
        # _execute_mem0_write never acquires this lock (B3).
        async with self.graphiti._identity_lock_for(payload['group_id']):
            result = await self._journaled_backend_call(
                write_op_id=write_op_id,
                causation_id=causation_id,
                backend='graphiti',
                operation='add_episode',
                payload={'content': payload['content'][:200], 'group_id': payload.get('group_id')},
                coro=self.graphiti.add_episode(
                    name=payload.get('name', ''),
                    content=payload['content'],
                    source=episode_type,
                    group_id=payload['group_id'],
                    source_description=payload.get('source_description', ''),
                    uuid=payload.get('uuid'),
                    temporal_context=temporal_context,
                    reference_time=reference_time,
                ),
            )
            reconcile_stats = await self._reconcile_episode_identity(
                result, group_id=payload['group_id'],
            )
            logger.debug(
                'Reconciled episode identity for group_id=%r: %r',
                payload['group_id'], reconcile_stats,
            )

        # Register planning episodes so they can be filtered from search results
        if temporal_context == 'planning' and self.planned_episode_registry is not None:
            episode_uuid = payload.get('uuid')
            group_id = payload.get('group_id')
            if episode_uuid and group_id:
                await self.planned_episode_registry.register(episode_uuid, group_id)
            elif episode_uuid and not group_id:
                logger.warning(
                    'Skipping planned episode registration: group_id missing from payload '
                    'for episode %s',
                    episode_uuid,
                )

        return result

    async def _execute_mem0_write(self, payload: dict[str, Any]) -> Any:
        """Execute a queued Mem0 add operation."""
        causation_id = payload.pop('_causation_id', None)
        write_op_id = payload.pop('_write_op_id', None)
        scope = Scope(
            project_id=payload['project_id'],
            agent_id=payload.get('agent_id'),
            session_id=payload.get('session_id'),
        )
        metadata = payload.get('metadata', {})
        # Resolved once, outside the try, so every attempt journals against a
        # stable id when the payload carries no '_write_op_id'.
        journal_write_op_id = write_op_id or str(uuid_mod.uuid4())

        result = None
        error_msg = None
        try:
            result = await self._journaled_backend_call(
                write_op_id=write_op_id,
                causation_id=causation_id,
                backend='mem0',
                operation='add',
                payload={'content': payload['content'][:200]},
                coro=self.mem0.add(
                    content=payload['content'], scope=scope, metadata=metadata
                ),
            )
            return result
        except Exception as e:
            error_msg = f'{type(e).__name__}: {e}'
            raise
        finally:
            # Log Layer 1 for the queued write on BOTH paths (task 3582). This
            # used to run only after a successful await, so a mem0_add that
            # dead-lettered never produced a write_ops row at all — leaving the
            # queue's terminal write-back nothing well-formed to land on, and
            # the failure invisible to the journal. The `finally` mirrors
            # add_episode's, and log_write_op is an upsert, so the retries of a
            # single item converge on one row whose last attempt wins.
            if self._write_journal:
                await self._write_journal.log_write_op(
                    write_op_id=journal_write_op_id,
                    causation_id=causation_id,
                    source='durable_queue',
                    operation='add_memory',
                    project_id=payload['project_id'],
                    agent_id=payload.get('agent_id'),
                    session_id=payload.get('session_id'),
                    params={
                        'content': payload['content'][:200],
                        'category': metadata.get('category', ''),
                    },
                    result_summary=str(result)[:500] if result else None,
                    success=error_msg is None,
                    error=error_msg,
                )

    async def _execute_mem0_classify_and_add(
        self, payload: dict[str, Any]
    ) -> Any:
        """Classify a fact extracted from an episode and write to Mem0 if appropriate."""
        fact_text = payload['fact_text']
        causation_id = payload.get('_causation_id')
        temporal_context = payload.get('temporal_context')
        write_op_id = str(uuid_mod.uuid4())
        scope = Scope(
            project_id=payload.get('project_id', 'main'),
            agent_id=payload.get('agent_id'),
            session_id=payload.get('session_id'),
        )

        classification = await self.classifier.classify(fact_text)
        if classification.primary not in MEM0_PRIMARY and classification.secondary is None:
            return None  # Not Mem0-bound

        metadata = {
            'category': classification.primary.value,
            'source': 'episode_extraction',
            'confidence': classification.confidence,
        }
        if classification.secondary:
            metadata['secondary_category'] = classification.secondary.value
        if temporal_context == 'planning':
            metadata['planned'] = True

        result = await self._journaled_backend_call(
            write_op_id=write_op_id,
            causation_id=causation_id,
            backend='mem0',
            operation='add',
            payload={'content': fact_text[:200]},
            coro=self.mem0.add(content=fact_text, scope=scope, metadata=metadata),
        )

        # Log Layer 1 for the derived write
        if self._write_journal:
            await self._write_journal.log_write_op(
                write_op_id=write_op_id,
                causation_id=causation_id,
                source='dual_write',
                provenance='derived',
                operation='add_memory',
                project_id=scope.project_id,
                agent_id=scope.agent_id,
                session_id=scope.session_id,
                params={
                    'content': fact_text[:200],
                    'category': classification.primary.value,
                },
                result_summary=str(result)[:500] if result else None,
                success=True,
            )

        logger.debug(f'Durable dual-wrote fact to Mem0: {fact_text[:80]}')
        return result

    async def _refresh_entity_summaries_from_result(
        self, result: Any, group_id: str
    ) -> None:
        """Best-effort post-ingestion summary refresh for edge endpoints in *result*.

        Graphiti's ``add_episode`` LLM extraction pipeline invalidates/supersedes/
        dedups edges internally as a side effect of ingestion, without any
        fused-memory code observing which edges changed. This helper closes that
        gap: it reads the edges returned by the ingestion call (mirrors the
        ``result.edges`` / ``result.entity_edges`` idiom used elsewhere in this
        file), collects the deduplicated set of non-empty source/target node
        uuids, and calls ``refresh_entity_summary`` for each — so the entity
        nodes touched by this write get a freshly rebuilt summary.

        Args:
            result: The value returned by ``add_episode`` (or equivalent),
                    typically carrying an ``edges`` (or ``entity_edges``)
                    attribute. ``None`` and empty/missing edges are a no-op.
            group_id: Graphiti graph id to refresh within.
        """
        if result is None:
            return

        edges = getattr(result, 'edges', None) or getattr(result, 'entity_edges', None) or []
        if not edges:
            return

        uuids: dict[str, None] = {}  # ordered dedup set
        for edge in edges:
            src_uuid = getattr(edge, 'source_node_uuid', '') or ''
            tgt_uuid = getattr(edge, 'target_node_uuid', '') or ''
            if src_uuid:
                uuids[src_uuid] = None
            if tgt_uuid:
                uuids[tgt_uuid] = None

        for node_uuid in uuids:
            try:
                await self.graphiti.refresh_entity_summary(node_uuid, group_id=group_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    'post-ingest refresh_entity_summary failed for uuid=%s group=%s: %s',
                    node_uuid, group_id, exc,
                )

    async def _dual_write_callback(
        self, callback_type: str, result: Any, payload: dict[str, Any]
    ) -> None:
        """Post-process callback: extract facts and enqueue each for durable Mem0
        write, then trigger a best-effort post-ingestion entity-summary refresh.

        Instead of writing directly to Mem0 (fire-and-forget), we batch-enqueue
        each extracted fact as a ``mem0_classify_and_add`` queue item so it gets
        independent retry / dead-letter handling. After the Mem0 enqueue, we also
        refresh the summary of every entity node touched by an edge endpoint in
        this result — closing the ingestion-time staleness gap where graphiti_core
        invalidates/supersedes/dedups edges internally without any fused-memory
        code observing which nodes changed.
        """
        if result is None:
            return

        edges = getattr(result, 'edges', None) or getattr(result, 'entity_edges', None) or []

        if edges:
            project_id = payload.get('project_id', 'main')
            group_id = f'mem0_{project_id}'

            batch = [
                {
                    'group_id': group_id,
                    'operation': 'mem0_classify_and_add',
                    'payload': {
                        'fact_text': getattr(edge, 'fact', None) or str(edge),
                        'project_id': project_id,
                        'agent_id': payload.get('agent_id'),
                        'session_id': payload.get('session_id'),
                        '_causation_id': payload.get('_causation_id'),
                        'temporal_context': payload.get('temporal_context'),
                    },
                }
                for edge in edges
            ]

            assert self.durable_queue is not None
            await self.durable_queue.enqueue_batch(batch)

        refresh_group_id = payload.get('group_id') or payload.get('project_id') or 'main'
        await self._refresh_entity_summaries_from_result(result, group_id=refresh_group_id)

    async def _refresh_summaries_callback(
        self, callback_type: str, result: Any, payload: dict[str, Any]
    ) -> None:
        """Refresh-only post-ingestion callback: no Mem0 enqueue.

        Used by ingestion paths that already handle Mem0 elsewhere —
        ``add_memory`` writes Mem0 directly and ``replay_from_store`` only
        re-ingests into Graphiti — so, unlike ``_dual_write_callback``, this
        callback must not also batch-enqueue a ``mem0_classify_and_add`` for
        each edge (that would double-write Mem0). It only triggers the
        best-effort post-ingestion entity-summary refresh.
        """
        group_id = payload.get('group_id') or payload.get('project_id') or 'main'
        await self._refresh_entity_summaries_from_result(result, group_id=group_id)

    # ------------------------------------------------------------------
    # Write: add_episode
    # ------------------------------------------------------------------

    async def add_episode(
        self,
        content: str,
        source: str = 'text',
        project_id: str = 'main',
        agent_id: str | None = None,
        session_id: str | None = None,
        reference_time: datetime | None = None,
        source_description: str = '',
        causation_id: str | None = None,
        temporal_context: str | None = None,
        _source: str = 'mcp_tool',
    ) -> AddEpisodeResponse:
        """Full ingestion pipeline — durably enqueue episode, return immediately."""
        scope = Scope(project_id=project_id, agent_id=agent_id, session_id=session_id)
        episode_id = str(uuid_mod.uuid4())
        write_op_id = str(uuid_mod.uuid4())

        # Parse source type name for storage
        try:
            source_name = EpisodeType[source.lower()].name
        except (KeyError, AttributeError):
            source_name = 'text'

        assert self.durable_queue is not None

        success = True
        error_msg = None
        try:
            await self.durable_queue.enqueue(
                group_id=scope.graphiti_group_id,
                operation='add_episode',
                payload={
                    'uuid': episode_id,
                    'name': f'episode_{episode_id[:8]}',
                    'content': content,
                    'source': source_name,
                    'group_id': scope.graphiti_group_id,
                    'source_description': source_description,
                    # Scope fields for callback reconstruction
                    'project_id': project_id,
                    'agent_id': agent_id,
                    'session_id': session_id,
                    # Journal metadata (popped by _execute_graphiti_write)
                    '_causation_id': causation_id,
                    '_write_op_id': write_op_id,
                    'temporal_context': temporal_context,
                    'reference_time': reference_time.isoformat() if reference_time is not None else None,
                },
                callback_type='dual_write_episode',
            )
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            if self._write_journal:
                await self._write_journal.log_write_op(
                    write_op_id=write_op_id,
                    causation_id=causation_id,
                    source=_source,
                    operation='add_episode',
                    project_id=project_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    params={'content': content[:200], 'source': source},
                    result_summary={'episode_id': episode_id, 'status': 'queued'} if success else None,
                    success=success,
                    error=error_msg,
                )

        await self._emit_event(ReconciliationEvent(
            id=str(uuid_mod.uuid4()),
            type=EventType.episode_added,
            source=EventSource.agent,
            project_id=project_id,
            timestamp=datetime.now(UTC),
            payload={'episode_id': episode_id, 'content_preview': content[:200]},
            agent_id=agent_id,
        ))

        return AddEpisodeResponse(
            episode_id=episode_id,
            status=EpisodeStatus.queued,
            message=f'Episode queued for processing in project {project_id}',
        )

    # ------------------------------------------------------------------
    # Write: add_memory
    # ------------------------------------------------------------------

    async def add_memory(
        self,
        content: str,
        category: str | MemoryCategory | None = None,
        project_id: str = 'main',
        agent_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        dual_write: bool = False,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
    ) -> AddMemoryResponse:
        """Lightweight classified write — skip extraction pipeline."""
        scope = Scope(project_id=project_id, agent_id=agent_id, session_id=session_id)
        write_op_id = str(uuid_mod.uuid4())

        # Resolve category
        if category is None:
            classification = await self.classifier.classify(content)
            resolved_category = classification.primary
        elif isinstance(category, str):
            resolved_category = MemoryCategory(category)
        else:
            resolved_category = category

        memory_ids: list[str] = []
        stores_written: list[SourceStore] = []
        meta = dict(metadata or {})
        meta['category'] = resolved_category.value

        # Normalize metadata.task_id to str at this shared write boundary
        # (task 2620, sibling of task 2454's flag_dedup-specific fix; shared
        # with add_system_record below per the task-2620 amendment review —
        # see _normalize_task_id_metadata's docstring for the full
        # rationale).
        _normalize_task_id_metadata(meta)

        # Server-side cycle_summary metadata tagging (recon_pool auto-tag
        # task 2077, run_id auto-backfill task 2109, missing-key warning
        # task 2094/2109) — factored into a shared helper (task 2222
        # amendment) so add_system_record gets the identical authoritative
        # treatment. See _apply_cycle_summary_metadata_tagging's docstring.
        _apply_cycle_summary_metadata_tagging(meta, causation_id, project_id=project_id)

        # Mem0 metadata vocabulary validation (task 3195, leaf β). Placement is
        # load-bearing at BOTH ends:
        #   AFTER the two tagging helpers above, so category/recon_pool/run_id
        #   are already in `meta` and get classified as server-stamped rather
        #   than censused as unknown keys (the alternative is a second copy of
        #   the server-stamped key list — an INV-5 violation);
        #   BEFORE the write_graphiti/write_mem0 branching below, the
        #   write-ahead mem0 intent, and every backend call, so a rejection can
        #   never leave a pending intent for recover_mem0_intents to reconcile
        #   or a half-written Graphiti twin. Being before the branching is also
        #   what makes this cover Graphiti-primary writes, which never reach
        #   Mem0 at all — V1 covers the seam, not just the Mem0 branch.
        await _apply_memory_metadata_validation(
            meta,
            project_id=project_id,
            agent_id=agent_id,
            config=self.config.memory_metadata,
            storm_detector=self._metadata_storm_detector,
            project_root=self._memory_metadata_project_root(),
            # Bound methods, not `self`: the module-level helper stays
            # decoupled from MemoryService and trivially stubbable, matching
            # how it already takes storm_detector/config as collaborators.
            parent_lookup=self.get_memory_by_id,
            count_canonical=self.count_memories_by_metadata,
            find_canonical=self.get_memories_by_metadata,
        )

        write_graphiti = (
            resolved_category in GRAPHITI_PRIMARY or dual_write
        )
        write_mem0 = (
            resolved_category in MEM0_PRIMARY or dual_write
        )

        _graphiti_error = None
        _mem0_error = None

        # Graphiti: enqueue via durable queue (async, but durably persisted)
        if write_graphiti:
            try:
                assert self.durable_queue is not None
                await self.durable_queue.enqueue(
                    group_id=scope.graphiti_group_id,
                    operation='add_memory_graphiti',
                    payload={
                        'name': f'memory_{resolved_category.value}',
                        'content': content,
                        'source': 'text',
                        'group_id': scope.graphiti_group_id,
                        'source_description': f'add_memory:{resolved_category.value}',
                        '_causation_id': causation_id,
                        '_write_op_id': write_op_id,
                    },
                    callback_type='refresh_entity_summaries',
                )
                # Durably persisted to SQLite — report as written
                stores_written.append(SourceStore.graphiti)
            except Exception as e:
                logger.error(f'Graphiti enqueue failed: {e}')
                _graphiti_error = f'{type(e).__name__}: {e}'

        # Mem0: direct synchronous call so memory_ids are returned to the caller.
        # The durable-queue path cannot return server-assigned IDs because Mem0
        # assigns IDs server-side and the queue worker has no path back to the caller.
        # Durability is retained via write_journal (log_backend_op captures every call).
        # The _execute_durable_write 'mem0_add' dispatcher is kept intact for backward
        # compat — any in-flight queue items from before this fix still drain correctly.
        if write_mem0:
            # Write-ahead intent (task 2710): durably journal the intended
            # mem0 write BEFORE the risky await, so a crash mid-mem0-write
            # leaves a 'pending' trace instead of a silently orphaned Graphiti
            # twin. recover_mem0_intents reconciles any pending intent at
            # startup. Keyed to the existing write_op_id so the reconciler can
            # correlate with the per-call mem0 backend_op as evidence.
            intent_id: str | None = None
            if self._write_journal:
                intent_id = str(uuid_mod.uuid4())
                payload_digest = self._mem0_payload_digest(
                    content,
                    project_id,
                    agent_id,
                    session_id,
                    resolved_category.value,
                    meta,
                )
                await self._write_journal.log_mem0_intent(
                    intent_id=intent_id,
                    write_op_id=write_op_id,
                    causation_id=causation_id,
                    project_id=project_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    category=resolved_category.value,
                    content=content,
                    metadata=meta,
                    payload_digest=payload_digest,
                )
            try:
                mem0_result = await self._journaled_backend_call(
                    write_op_id=write_op_id,
                    causation_id=causation_id,
                    backend='mem0',
                    operation='add',
                    payload={'content': content[:200]},
                    coro=self.mem0.add(content=content, scope=scope, metadata=meta),
                )
                mem0_ids = [
                    r['id']
                    for r in (mem0_result or {}).get('results', [])
                    if isinstance(r, dict) and 'id' in r
                ]
                memory_ids.extend(mem0_ids)
                stores_written.append(SourceStore.mem0)
                # In-request call resolved without raising → stamp terminal.
                if self._write_journal and intent_id:
                    await self._write_journal.resolve_mem0_intent(
                        intent_id, 'completed'
                    )
                if not mem0_ids and _MEM0_ADD_INFER_PINNED_FALSE:
                    # mem0.add() did not raise but returned no results — a silent
                    # dedup/infer no-op drop (task 1974). Under the pinned
                    # infer=False, a successful write always returns exactly one
                    # result with an id, so this is always anomalous.
                    logger.warning(
                        'MemoryService.add_memory: mem0 add returned zero memory_ids '
                        '(silent empty-result drop; possible dedup/infer no-op)',
                        extra={
                            'project_id': project_id,
                            'category': resolved_category.value,
                            'causation_id': causation_id,
                            'kind': meta.get('kind'),
                        },
                    )
            except Exception as e:
                logger.error(f'Mem0 write failed: {e}')
                _mem0_error = str(e)
                # add() provably raised → mem0 did NOT persist. Stamp the
                # intent 'failed' (the reconciler treats a failed-only
                # backend_op as safe to re-issue).
                if self._write_journal and intent_id:
                    await self._write_journal.resolve_mem0_intent(
                        intent_id, 'failed', reason=str(e)
                    )

        # Layer 1 journal entry
        if self._write_journal:
            await self._write_journal.log_write_op(
                write_op_id=write_op_id,
                causation_id=causation_id,
                source=_source,
                operation='add_memory',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params={'content': content[:200], 'category': resolved_category.value},
                result_summary={
                    'memory_ids': memory_ids,
                    'stores': [s.value for s in stores_written],
                },
                success=not (_graphiti_error or _mem0_error),
                error=_graphiti_error or _mem0_error,
            )

        await self._emit_event(ReconciliationEvent(
            id=str(uuid_mod.uuid4()),
            type=EventType.memory_added,
            source=EventSource.agent,
            project_id=project_id,
            timestamp=datetime.now(UTC),
            payload={
                'memory_ids': memory_ids,
                'category': resolved_category.value,
                'content_preview': content[:200],
            },
            agent_id=agent_id,
        ))

        msg = f'Memory queued for {[s.value for s in stores_written]}'
        if _graphiti_error:
            msg += f' [graphiti_error: {_graphiti_error}]'
        if _mem0_error:
            msg += f' [mem0_error: {_mem0_error}]'

        return AddMemoryResponse(
            memory_ids=memory_ids,
            stores_written=stores_written,
            category=resolved_category,
            message=msg,
        )

    # ------------------------------------------------------------------
    # Recovery: reconcile crash-mid-write mem0 intents (task 2710)
    # ------------------------------------------------------------------

    async def recover_mem0_intents(self) -> dict[str, int]:
        """Reconcile every crash-mid-write mem0 intent still ``pending``.

        A ``pending`` intent means the process died between the write-ahead
        intent (committed before the mem0 await) and its terminal stamp.
        Classify each by the existing per-call mem0 ``backend_op`` keyed on
        ``write_op_id`` — reusing durable evidence rather than a second
        bookkeeping channel:

        - a SUCCESS mem0 backend_op → the write provably landed (only the
          completion stamp was lost) → mark ``completed``; NOT an orphan, no
          re-issue.
        - only FAILED mem0 backend_op(s) → ``add()`` raised. In the common
          (clean, pre-persist) failure mem0 did not land, so re-issue is safe
          (0 prior writes + 1 = 1): rebuild ``Scope`` + metadata and call
          ``mem0.add``; ``completed`` on success, ``dead`` on error.
          RESIDUAL DUPLICATE RISK (accepted, documented): a failure raised
          AFTER mem0 committed but at/near the response (e.g. a read-timeout
          on an otherwise-successful add) ALSO records a FAILED backend_op
          while the write actually landed — re-issuing then mints a duplicate
          twin, since the pinned ``infer=False`` add is non-idempotent. We
          accept this narrow post-send-failure risk to heal the far more
          common clean-failure case; the fully UNKNOWN no-beop case below is
          the one we refuse to auto-re-issue.
        - NO mem0 backend_op → outcome UNKNOWN (killed before/inside the
          await) → dead-letter with a structured reason. ``mem0.add`` pins
          ``infer=False`` and is non-idempotent, so a blind re-issue risks a
          duplicate twin; the goal is that a partial write is never SILENT,
          not automatic healing. Manual replay remains available.

        Every outcome is a durable row (readable via ``get_mem0_intents``)
        plus a loud log line. Idempotent — no pending intents ⇒ zeroed
        summary — so it is safe to run on every startup.
        """
        summary = {'scanned': 0, 'reconciled': 0, 'reissued': 0, 'dead_lettered': 0}
        if self._write_journal is None:
            return summary

        pending = await self._write_journal.get_incomplete_mem0_intents()
        for intent in pending:
            summary['scanned'] += 1
            intent_id = intent['id']
            write_op_id = intent['write_op_id']
            payload_digest = intent.get('payload_digest')

            beops = await self._write_journal.get_backend_ops_for_write_op(
                write_op_id
            )
            mem0_beops = [b for b in beops if b.get('backend') == 'mem0']
            any_success = any(b.get('success') for b in mem0_beops)

            if any_success:
                # Write landed; only the completion stamp was lost.
                await self._write_journal.resolve_mem0_intent(
                    intent_id,
                    'completed',
                    reason='reconciled: mem0 backend_op confirms write landed',
                )
                summary['reconciled'] += 1
                logger.info(
                    'recover_mem0_intents: intent %s reconciled completed — '
                    'mem0 backend_op confirms write landed (write_op_id=%s)',
                    intent_id,
                    write_op_id,
                )
            elif mem0_beops:
                # Only failed backend_op(s) → add() raised → re-issue. Heals the
                # common pre-persist failure; see the RESIDUAL DUPLICATE RISK
                # note in this method's docstring for the narrow
                # post-commit-failure case where this can mint a duplicate twin.
                try:
                    scope = Scope(
                        project_id=intent.get('project_id') or 'main',
                        agent_id=intent.get('agent_id'),
                        session_id=intent.get('session_id'),
                    )
                    metadata = json.loads(intent.get('metadata') or '{}')
                    await self.mem0.add(
                        content=intent.get('content') or '',
                        scope=scope,
                        metadata=metadata,
                    )
                    await self._write_journal.resolve_mem0_intent(
                        intent_id,
                        'completed',
                        reason='reconciled: re-issued after failed-only mem0 backend_op',
                    )
                    summary['reissued'] += 1
                    logger.warning(
                        'recover_mem0_intents: intent %s re-issued and completed — '
                        'prior mem0 add failed (write_op_id=%s)',
                        intent_id,
                        write_op_id,
                    )
                except Exception as e:
                    reason = (
                        f're-issue failed: {type(e).__name__}: {e} '
                        f'(write_op_id={write_op_id}, payload_digest={payload_digest})'
                    )
                    await self._write_journal.resolve_mem0_intent(
                        intent_id, 'dead', reason=reason
                    )
                    summary['dead_lettered'] += 1
                    logger.error(
                        'recover_mem0_intents: intent %s dead-lettered — re-issue '
                        'raised: %s',
                        intent_id,
                        e,
                    )
            else:
                # No backend_op → UNKNOWN outcome → dead-letter (never silent).
                reason = (
                    'dead-lettered: no mem0 backend_op for '
                    f'write_op_id={write_op_id} — outcome UNKNOWN (killed '
                    'before/inside the mem0 await). '
                    f'payload_digest={payload_digest} '
                    f'project_id={intent.get("project_id")} '
                    f'agent_id={intent.get("agent_id")} '
                    f'session_id={intent.get("session_id")}. '
                    'Not auto-re-issued: infer=False add is non-idempotent and '
                    'would risk a duplicate twin; manual replay available.'
                )
                await self._write_journal.resolve_mem0_intent(
                    intent_id, 'dead', reason=reason
                )
                summary['dead_lettered'] += 1
                logger.warning(
                    'recover_mem0_intents: intent %s dead-lettered — no mem0 '
                    'backend_op, outcome UNKNOWN (write_op_id=%s, payload_digest=%s)',
                    intent_id,
                    write_op_id,
                    payload_digest,
                )

        if summary['scanned']:
            logger.info('recover_mem0_intents complete: %s', summary)
        return summary

    async def add_system_record(
        self,
        content: str,
        *,
        project_id: str,
        agent_id: str | None,
        category: str | MemoryCategory,
        metadata: dict | None = None,
        causation_id: str | None = None,
        session_id: str | None = None,
        _source: str = 'mcp_tool',
    ) -> AddMemoryResponse:
        """Dedup-exempt, Mem0-only system write (task 2222 / W5-δ).

        Always writes through ``Mem0Backend.add_system_record`` — never the
        general ``add()`` — so the write is structurally exempt from Mem0's
        (future) dedup behaviour regardless of any change to the general
        add path. Never routes to Graphiti, regardless of ``category``:
        category is stamped into the metadata as a tag only, not used for
        store routing. Models the Mem0 branch of :meth:`add_memory` minus
        the Graphiti half and minus the general-add dedup-dependent path.
        """
        scope = Scope(project_id=project_id, agent_id=agent_id, session_id=session_id)
        write_op_id = str(uuid_mod.uuid4())

        resolved_category = MemoryCategory(category) if isinstance(category, str) else category

        meta = dict(metadata or {})
        meta['category'] = resolved_category.value

        # Same task_id normalization add_memory applies (task 2620 amendment
        # review): this Mem0-only path shares add_memory's exact-match read
        # filters (count_memories_by_metadata/get_memories_by_metadata), so
        # a task_id-keyed marker written here needs the same str convention.
        # See _normalize_task_id_metadata's docstring for the full rationale.
        _normalize_task_id_metadata(meta)

        # Same authoritative cycle_summary tagging add_memory applies (task
        # 2222 amendment): the tool docstring names the cycle-summary Mem0
        # mirror as the intended caller, and recon_pool/run_id are the keys
        # the pool-cap trim and Path-2 triple-filter pre-check rely on — a
        # system-record cycle_summary must not go untagged just because it
        # bypassed add_memory.
        _apply_cycle_summary_metadata_tagging(meta, causation_id, project_id=project_id)

        # Same vocabulary validation add_memory applies (task 3195, leaf β).
        # PRD D8/§2 name add_system_record as the second unguarded write path
        # that a tools-layer validator would leak past, so it shares the very
        # same helper rather than getting a parallel implementation that could
        # drift. Placed after the tagging helpers and before the
        # _journaled_backend_call below, for the reasons spelled out at
        # add_memory's call site.
        await _apply_memory_metadata_validation(
            meta,
            project_id=project_id,
            agent_id=agent_id,
            config=self.config.memory_metadata,
            storm_detector=self._metadata_storm_detector,
            project_root=self._memory_metadata_project_root(),
            # Bound methods, not `self`: the module-level helper stays
            # decoupled from MemoryService and trivially stubbable, matching
            # how it already takes storm_detector/config as collaborators.
            parent_lookup=self.get_memory_by_id,
            count_canonical=self.count_memories_by_metadata,
            find_canonical=self.get_memories_by_metadata,
        )

        mem0_result = None
        mem0_ids: list[str] = []
        _mem0_error: str | None = None
        try:
            mem0_result = await self._journaled_backend_call(
                write_op_id=write_op_id,
                causation_id=causation_id,
                backend='mem0',
                operation='add',
                payload={'content': content[:200]},
                coro=self.mem0.add_system_record(content=content, scope=scope, metadata=meta),
            )
            mem0_ids = [
                r['id']
                for r in (mem0_result or {}).get('results', [])
                if isinstance(r, dict) and 'id' in r
            ]
        except Exception as e:
            # Mirrors add_memory's try/except around its Mem0 branch (task
            # 2222 amendment): without this, a raised exception would skip
            # BOTH the Layer-1 write-journal entry and the memory_added
            # event below, and propagate a bare error to the tool boundary
            # with no audit trail — exactly what this guaranteed-persistence
            # system-write path exists to rule out.
            logger.error(f'Mem0 system-record write failed: {e}')
            _mem0_error = str(e)

        # Mem0Backend.add_system_record pins infer=False LOCALLY and
        # unconditionally (never inherited from the general add() pin — see
        # its docstring; MUST NEVER pass infer=True), so a successful write
        # always returns exactly one result with an id. An empty result is
        # therefore always a silent-drop anomaly (mirrors the add_memory
        # empty-result check, task 1974) — and silence is exactly what this
        # guaranteed-persistence system-write path exists to rule out, so it
        # must not be journaled as an unconditional success.
        _empty_result = not mem0_ids
        if _empty_result and not _mem0_error:
            logger.warning(
                'MemoryService.add_system_record: mem0 add_system_record '
                'returned zero memory_ids (silent empty-result drop on the '
                'dedup-exempt system-write path)',
                extra={
                    'project_id': project_id,
                    'category': resolved_category.value,
                    'causation_id': causation_id,
                    'kind': meta.get('kind'),
                },
            )

        if self._write_journal:
            await self._write_journal.log_write_op(
                write_op_id=write_op_id,
                causation_id=causation_id,
                source=_source,
                operation='add_system_record',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params={'content': content[:200], 'category': resolved_category.value},
                result_summary={
                    'memory_ids': mem0_ids,
                    'stores': [SourceStore.mem0.value],
                },
                success=not _empty_result,
                error=(
                    _mem0_error if _mem0_error else
                    ('empty_result: mem0 add_system_record returned zero memory_ids'
                     if _empty_result else None)
                ),
            )

        # An empty result (whether from a caught exception above or a
        # silent zero-id return) is tagged directly on the event payload —
        # not just the Layer-1 journal — so a reconciliation consumer
        # keying off memory_added alone can still distinguish a real write
        # from a drop (task 2222 amendment).
        _event_payload = {
            'memory_ids': mem0_ids,
            'category': resolved_category.value,
            'content_preview': content[:200],
        }
        if _empty_result:
            _event_payload['empty_result'] = True
            if _mem0_error:
                _event_payload['error'] = _mem0_error

        await self._emit_event(ReconciliationEvent(
            id=str(uuid_mod.uuid4()),
            type=EventType.memory_added,
            source=EventSource.agent,
            project_id=project_id,
            timestamp=datetime.now(UTC),
            payload=_event_payload,
            agent_id=agent_id,
        ))

        msg = f'Memory queued for {[SourceStore.mem0.value]}'
        if _mem0_error:
            msg += f' [mem0_error: {_mem0_error}]'

        return AddMemoryResponse(
            memory_ids=mem0_ids,
            stores_written=[SourceStore.mem0],
            category=resolved_category,
            message=msg,
        )

    # ------------------------------------------------------------------
    # Replay: re-ingest Mem0 memories into Graphiti
    # ------------------------------------------------------------------

    async def replay_from_store(
        self,
        source_project_id: str,
        target_project_id: str | None = None,
        limit: int | None = None,
    ) -> int:
        """Fetch memories from Mem0 and enqueue each for Graphiti write.

        Args:
            limit: Max memories to replay. None = all (up to 1000).

        Returns the count of items queued.
        """
        target = target_project_id or source_project_id
        scope = Scope(project_id=source_project_id)
        fetch_limit = limit if limit else 1000
        all_mems = await self.mem0.get_all(scope, limit=fetch_limit)
        memories = all_mems.get('results', [])
        if not memories:
            return 0

        assert self.durable_queue is not None
        batch = []
        for mem in memories:
            content = mem.get('memory', '')
            if not content:
                continue
            meta = mem.get('metadata', {}) or {}
            category = meta.get('category', 'observations_and_summaries')
            batch.append({
                'group_id': target,
                'operation': 'add_memory_graphiti',
                'payload': {
                    'name': f'replay_{category}',
                    'content': content,
                    'source': 'text',
                    'group_id': target,
                    'source_description': f'replay_from_mem0:{category}',
                },
                'callback_type': 'refresh_entity_summaries',
            })

        if batch:
            await self.durable_queue.enqueue_batch(batch)
        return len(batch)

    # ------------------------------------------------------------------
    # Read: search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        project_id: str = 'main',
        categories: list[str] | None = None,
        stores: list[str] | None = None,
        limit: int = 10,
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        include_planned: bool = False,
    ) -> list[MemoryResult]:
        """Unified search across both stores with automatic fan-out.

        When include_planned=False (default), edges and memories from planning
        episodes (temporal_context='planning') are excluded.  Set include_planned=True
        to include them — useful for reconciliation and auditing.
        """
        scope = Scope(project_id=project_id, agent_id=agent_id, session_id=session_id)

        # Determine routing
        stores_override = [SourceStore(s) for s in stores] if stores else None
        route: ReadRouteResult = await self.router.route(query, stores_override)

        # Fan out to stores in parallel with timeout
        search_timeout = self.config.queue.search_timeout_seconds
        store_list: list[SourceStore] = []
        task_list: list[asyncio.Task] = []

        if SourceStore.graphiti in route.stores:
            store_list.append(SourceStore.graphiti)
            task_list.append(asyncio.create_task(
                self._search_graphiti(query, scope, limit, include_planned=include_planned)
            ))
        if SourceStore.mem0 in route.stores:
            store_list.append(SourceStore.mem0)
            task_list.append(asyncio.create_task(
                self._search_mem0(query, scope, limit, include_planned=include_planned, categories=categories)
            ))

        results: list[MemoryResult] = []
        failed_stores: list[SourceStore] = []
        failure_diagnostics: list[dict] = []
        if task_list:
            done, pending = await asyncio.wait(
                task_list, timeout=search_timeout, return_when=asyncio.ALL_COMPLETED
            )
            for t in pending:
                t.cancel()

            timed_out_stores = [
                store_list[i] for i, t in enumerate(task_list) if t in pending
            ]
            if timed_out_stores:
                logger.warning(
                    f'Search timed out for stores: {[s.value for s in timed_out_stores]}'
                )
            failed_stores.extend(timed_out_stores)
            failure_diagnostics.extend(
                _store_failure_diagnostics(
                    store, None, query=query, project_id=project_id, reason='timeout'
                )
                for store in timed_out_stores
            )

            for i, t in enumerate(task_list):
                if t not in done:
                    continue
                try:
                    store_results = t.result()
                    results.extend(store_results)
                except Exception as e:
                    diag = _store_failure_diagnostics(
                        store_list[i], e, query=query, project_id=project_id, reason='exception'
                    )
                    logger.warning('search.store_failed', extra=diag)
                    failed_stores.append(store_list[i])
                    failure_diagnostics.append(diag)

        # Sort: primary store results first, then by relevance score
        def sort_key(r: MemoryResult) -> tuple[int, float]:
            is_primary = 0 if r.source_store == route.primary_store else 1
            return (is_primary, -r.relevance_score)

        results.sort(key=sort_key)

        # Graphiti results: filter by category and infer category for ambiguous
        # nodes.  Mem0 results are already category-scoped server-side via the
        # pushdown in Mem0Backend.search (see task 1083), so this block is
        # redundant-but-harmless for mem0 and acts as defence-in-depth.
        if categories:
            cat_set = {MemoryCategory(c) for c in categories}
            graphiti_overlap = cat_set & GRAPHITI_PRIMARY
            results = [
                r for r in results
                if r.category in cat_set
                or (
                    r.source_store == SourceStore.graphiti
                    and r.category is None
                    and graphiti_overlap
                )
            ]
            # Assign inferred category to Graphiti results when unambiguous
            if len(graphiti_overlap) == 1:
                inferred = next(iter(graphiti_overlap))
                for r in results:
                    if r.source_store == SourceStore.graphiti and r.category is None:
                        r.category = inferred

        degraded = bool(failed_stores)
        final = results[:limit]

        # Log search when causation_id is present (recon paths)
        if causation_id and self._write_journal:
            await self._write_journal.log_write_op(
                write_op_id=str(uuid_mod.uuid4()),
                causation_id=causation_id,
                source='mcp_tool',
                operation='search',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                kind='read',
                params={'query': query[:200], 'limit': limit},
                result_summary={
                    'count': len(final),
                    'failed_stores': [s.value for s in failed_stores],
                },
                success=not degraded,
            )

        return SearchResults(
            final,
            degraded=degraded,
            failed_stores=[s.value for s in failed_stores],
            failure_diagnostics=failure_diagnostics,
        )

    async def _search_graphiti(
        self, query: str, scope: Scope, limit: int, include_planned: bool = False
    ) -> list[MemoryResult]:
        """Search Graphiti and convert results to MemoryResult.

        When include_planned=False (default), edges whose entire provenance is
        composed of planned-only episodes are excluded.  When include_planned=True,
        those edges are returned and marked with metadata['planned'] = True.
        """
        edges = await self.graphiti.search(
            query=query,
            group_ids=[scope.graphiti_group_id],
            num_results=int(limit * 1.5) + 1,
        )

        # Fetch planned UUIDs once (avoid per-edge DB hits).
        planned_uuids: set[str] = set()
        if self.planned_episode_registry is not None:
            planned_uuids = await self.planned_episode_registry.get_planned_uuids(
                scope.graphiti_group_id
            )

        results = []
        for i, edge in enumerate(edges):
            fact = getattr(edge, 'fact', str(edge))
            valid_at = getattr(edge, 'valid_at', None)
            invalid_at = getattr(edge, 'invalid_at', None)

            # Skip superseded edges (invalid_at set means the fact has been
            # replaced by a newer edge).  Check this before anything else to
            # avoid unnecessary work on edges that will be discarded.
            if invalid_at is not None:
                continue

            temporal = _serialize_temporal(valid_at, invalid_at)

            # Extract entity names from source/target nodes
            entities = []
            source_node = getattr(edge, 'source_node', None)
            target_node = getattr(edge, 'target_node', None)
            if source_node and hasattr(source_node, 'name'):
                entities.append(source_node.name)
            if target_node and hasattr(target_node, 'name'):
                entities.append(target_node.name)

            # Episode provenance
            episodes = getattr(edge, 'episodes', None) or []
            provenance = [str(ep) for ep in episodes]

            # Determine whether this edge is purely aspirational (all episodes planned).
            is_planned_edge = bool(provenance) and all(
                ep in planned_uuids for ep in provenance
            )

            if is_planned_edge and not include_planned:
                # Skip planning-only edges in normal search results.
                continue

            # Score: rank-based (no explicit score from Graphiti search)
            score = max(0.0, 1.0 - (i * 0.05))

            metadata: dict[str, Any] = {}
            if is_planned_edge:
                metadata['planned'] = True

            results.append(MemoryResult(
                id=getattr(edge, 'uuid', str(i)),
                content=fact,
                category=None,
                source_store=SourceStore.graphiti,
                relevance_score=score,
                provenance=provenance,
                temporal=temporal,
                entities=entities,
                metadata=metadata,
            ))
        # Truncate to the original limit (over-fetch may have produced extras).
        return results[:limit]

    async def _search_mem0(
        self,
        query: str,
        scope: Scope,
        limit: int,
        include_planned: bool = False,
        categories: list[str] | None = None,
    ) -> list[MemoryResult]:
        """Search Mem0 and convert results to MemoryResult.

        When include_planned=False (default), results tagged with planned=True
        in their metadata are excluded.  When include_planned=True they are returned.
        """
        # Forward categories so Mem0Backend pushes the filter down to Qdrant
        # (task 1083: prevents false-negatives caused by post-filtering on
        # an already-truncated top-N that excludes low-ranked matching memories).
        response = await self.mem0.search(query=query, scope=scope, limit=limit, categories=categories)
        mem0_results = response.get('results', [])
        results = []
        for item in mem0_results:
            content = item.get('memory', '')
            score = float(item.get('score', 0.0))
            meta = item.get('metadata', {}) or {}

            # Filter out planning-tagged results unless explicitly requested.
            if meta.get('planned') is True and not include_planned:
                continue

            category = None
            if 'category' in meta:
                with contextlib.suppress(ValueError):
                    category = MemoryCategory(meta['category'])

            results.append(MemoryResult(
                id=item.get('id', ''),
                content=content,
                category=category,
                source_store=SourceStore.mem0,
                relevance_score=min(score, 1.0),
                metadata=meta,
                created_at=item.get('created_at'),
            ))
        return results

    # ------------------------------------------------------------------
    # Read: get_entity
    # ------------------------------------------------------------------

    async def get_entity(
        self,
        name: str,
        project_id: str = 'main',
        *,
        edge_limit: int = 10,
    ) -> dict:
        """Entity lookup in Graphiti — returns nodes + edges.

        Tries an exact, case-sensitive name match first (via
        graphiti.get_nodes_by_exact_name): canonical labels like "Task 115" resolve to
        the exact node instead of scattering across fuzzy neighbours. On an exact hit,
        fuzzy node search (search_nodes) is skipped entirely, and edges are fetched
        from each resolved node's uuid via graphiti.get_valid_edges_for_node (a uuid
        traversal) instead of a semantic fact search — this keeps edges consistent
        with the resolved node(s) rather than scattering across unrelated nodes that
        happen to be textually similar. When multiple nodes share the exact name
        (duplicate-name pathology), edges are fetched for every matched uuid and
        unioned, deduped by edge uuid, so `edges` stays consistent with the full
        `nodes` array. Only the fuzzy fallback below uses the semantic edge search.
        On no exact match, falls back to the fuzzy gather path.

        Concurrent Graphiti calls — both the exact-match branch's per-uuid
        get_valid_edges_for_node fetches and the fuzzy fallback's node/edge
        search — run via asyncio.gather(return_exceptions=True). This ensures no
        call becomes an orphaned background task in the error path: gather()
        awaits every coroutine to settlement before returning, even when one (or
        more) raise an exception. If any call fails, all exceptions are logged
        (warning) then the first exception is re-raised.

        Performance trade-off: the exact-match lookup below is a serial round-trip
        that runs in front of the fuzzy gather on every call — its result (hit vs.
        miss) gates which branch runs, so it cannot be folded into the concurrent
        gather(). This adds one extra round-trip of latency even to calls that end
        up on the fuzzy path. Accepted because canonical-label lookups (e.g.
        "Task 115") are this fast path's primary target; gating it behind a
        name-pattern regex was considered and rejected (see plan design_decisions)
        as added complexity that would leave non-numeric exact lookups (e.g.
        "Auth Service") still exposed to fuzzy neighbours for no benefit.

        Edge cap: `edge_limit` (default 10) applies ONLY to the fuzzy-fallback
        branch, where edges are gathered via a single graphiti.search() call
        passing edge_limit as num_results. graphiti.search returns edges in
        RELEVANCE-ranked order, so when an entity has more than `edge_limit`
        valid edges, the lowest-ranked ones are silently dropped — raise
        edge_limit to fetch more, or cross-check completeness via a direct
        search() call. The exact-match branch does NOT use edge_limit: its
        get_valid_edges_for_node traversal returns every valid edge for the
        resolved node uuid(s), uncapped. Both branches exclude invalidated
        edges (invalid_at set): the exact branch at the Cypher level (WHERE
        e.invalid_at IS NULL), and the fuzzy branch via an explicit
        invalid_at filter applied to the search() results, mirroring
        search()'s own drop (task 312) — so neither branch surfaces
        superseded facts (task 2726).

        EXACT vs. FUZZY edge sets are fundamentally different — TOPOLOGICAL
        vs. SEMANTIC — and comparing them across branches will produce
        false-positive "missing edges" findings. The exact branch's
        get_valid_edges_for_node (graphiti_client.py) runs a plain, uncached
        Cypher MATCH — `MATCH (n:Entity {uuid: $uuid})-[e:RELATES_TO]-()
        WHERE e.invalid_at IS NULL` — with no LIMIT and no ranking: it
        returns every RELATES_TO edge that is graph-connected to the
        resolved node uuid(s), full stop. The fuzzy branch's graphiti.search()
        instead does a semantic/relevance search over edge fact TEXT, which
        can surface edges whose fact merely mentions the entity's name (or is
        contextually related) without that edge being RELATES_TO-incident on
        this node at all. So an edge that appears in a search() result for
        this entity's name but is ABSENT from get_entity's exact-branch
        `edges` is not necessarily an omission bug — it may simply not be
        topologically connected to the resolved node. Before flagging
        get_entity as "silently omitting valid edges," confirm which branch
        was actually compared: the topological get_valid_edges_for_node set,
        or a semantic search() set. (This exact confusion produced a
        cross-project false positive: solar_challenge_platform's Stage-1
        reconciliation flagged get_entity for `review/briefing.yaml` as
        omitting edges found via search(); it became dark_factory task 2404,
        cancelled 2026-07-09 once the architect confirmed it was expected
        topological-vs-semantic divergence, not a bug.)

        Degraded fallback: if an embedding-provider rate-limit or quota error
        (openai.RateLimitError or a duck-typed equivalent — see
        _is_rate_limit_or_quota_error) is raised by either the exact-match or
        fuzzy-fallback Graphiti calls, this returns the degraded superset
        dict {'nodes': [], 'edges': [], 'degraded': True, 'failed_stores':
        ['graphiti']} instead of raising. The RESPONSE SHAPE mirrors
        search()'s degraded/failed_stores convention, but the TRIGGER is
        narrower than search()'s: search() degrades on ANY per-store
        Exception (task 1812), whereas get_entity degrades ONLY on a
        classified rate-limit/quota error — a Graphiti connection failure,
        timeout, or other backend error still propagates as a hard
        exception here, unlike search(). All other errors (including
        asyncio.CancelledError, a BaseException that never reaches this
        except clause) propagate unchanged.
        """
        # NOTE: each try/except below wraps ONLY the awaited Graphiti calls, not
        # the local dict-building that follows a successful gather (_node_to_dict/
        # _edge_to_dict, the edge-dedup loop). Keeping that pure data-transformation
        # code outside the guarded region means a bug there raises normally instead
        # of risking mis-classification as a degraded quota error (see plan review).
        try:
            # See "Performance trade-off" above: this call is intentionally serial —
            # its 0/1/many result decides whether the fuzzy gather runs at all.
            exact = await self.graphiti.get_nodes_by_exact_name(name, group_id=project_id)
            if exact:
                uuids = [uuid for n in exact if (uuid := n.get('uuid'))]
                # Two-tier check via gather_or_raise (fused_memory.utils.async_utils),
                # mirroring the fuzzy fallback below: return_exceptions=True ensures a
                # transient failure in one concurrent get_valid_edges_for_node call
                # cannot leave sibling coroutines running as orphaned background
                # tasks. Pass 1 re-raises structured-cancellation signals; Pass 2
                # logs every exception, then raises the first (get_entity's local
                # Pass-2 semantics — see async_utils docstring).
                edge_results = await gather_or_raise(
                    (self.graphiti.get_valid_edges_for_node(u, group_id=project_id) for u in uuids),
                    label='get_entity: get_valid_edges_for_node failed',
                    logger=logger,
                )
        except Exception as exc:
            return _degrade_or_reraise(exc, name)

        if exact:
            edge_lists = cast(list, edge_results)
            # Duplicate-name matches each contribute their own edges; dedup by
            # edge uuid so `edges` stays consistent with the (possibly
            # multi-node) `nodes` array instead of double-counting shared edges.
            seen: set = set()
            edges = []
            for edge_list in edge_lists:
                for e in edge_list:
                    edge_uuid = e.get('uuid')
                    if edge_uuid is not None:
                        if edge_uuid in seen:
                            continue
                        seen.add(edge_uuid)
                    edges.append(e)
            node_data = [_node_to_dict(n) for n in exact]
            edge_data = [_edge_to_dict(e) for e in edges]
            return {'nodes': node_data, 'edges': edge_data}

        try:
            # Two-tier check via gather_or_raise (fused_memory.utils.async_utils).
            # Both coroutines settle before either is inspected (no orphans).
            # Pass 1: re-raises structured-cancellation signals (CancelledError,
            #   KeyboardInterrupt, SystemExit) before any per-call logging —
            #   cancellation takes precedence over application-level failures
            #   regardless of position in the results list.
            # Pass 2: logs each captured Exception and raises the first — these
            #   are application-level failures from the Graphiti backend.
            results = await gather_or_raise(
                (
                    self.graphiti.search_nodes(
                        query=name,
                        group_ids=[project_id],
                        max_nodes=5,
                    ),
                    self.graphiti.search(
                        query=name,
                        group_ids=[project_id],
                        num_results=edge_limit,
                    ),
                ),
                label='get_entity: Graphiti call failed',
                logger=logger,
            )
        except Exception as exc:
            return _degrade_or_reraise(exc, name)

        nodes = cast(list, results[0])
        edges = cast(list, results[1])

        node_data = [_node_to_dict(n) for n in nodes]
        # Drop superseded edges (invalid_at set) so the fuzzy/semantic branch
        # presents the same valid-edge semantics as the exact/topological branch
        # (get_valid_edges_for_node's Cypher WHERE invalid_at IS NULL) and as
        # search() (task 312, ~line 2588) — otherwise an agent resolving via the
        # fuzzy fallback could silently act on a stale fact (task 2726).
        edge_data = [_edge_to_dict(e) for e in edges if getattr(e, 'invalid_at', None) is None]

        return {'nodes': node_data, 'edges': edge_data}

    # ------------------------------------------------------------------
    # Read: get_edge (thin wrapper for cite_edge — task β)
    # ------------------------------------------------------------------

    async def get_edge(self, edge_uuid: str, project_id: str) -> dict:
        """Fetch a single edge by UUID from Graphiti.

        Returns {uuid, name, fact} on success.  Raises EdgeNotFoundError on miss.
        group_id is mapped from project_id to match the Graphiti storage convention.
        """
        name, fact = await self.graphiti.get_edge_text(edge_uuid, group_id=project_id)
        return {'uuid': edge_uuid, 'name': name, 'fact': fact}

    # ------------------------------------------------------------------
    # Read: get_entity_by_uuid (topological direct-UUID diagnostic — task 2086)
    # ------------------------------------------------------------------

    async def get_entity_by_uuid(self, entity_uuid: str, project_id: str) -> dict:
        """Fetch a single Entity node by UUID from Graphiti (topological readback).

        Unlike get_entity (name-based, with fuzzy fallback and a semantic edge
        gather), this is a direct UUID lookup with no edge gather — intended as
        a diagnostic for confirming node identity without the fuzzy/semantic
        matching that can mask duplicate-node pathology.

        Returns {uuid, name, summary} on success.  Raises NodeNotFoundError on
        miss.  group_id is mapped from project_id to match the Graphiti storage
        convention.
        """
        name, summary = await self.graphiti.get_node_text(entity_uuid, group_id=project_id)
        return {'uuid': entity_uuid, 'name': name, 'summary': summary}

    # ------------------------------------------------------------------
    # Read: get_memory (thin dispatcher for cite_memory — task β)
    # ------------------------------------------------------------------

    async def get_memory(self, memory_id: str, store: str, project_id: str) -> dict:
        """Fetch a memory by id from either Graphiti or Mem0.

        Returns a minimal metadata fingerprint dict:
          {category, agent_id, created_at} for mem0;
          {name, fact_snippet} for graphiti.
        Raises EdgeNotFoundError (graphiti) or ValueError (mem0 not found).
        """
        if store == 'graphiti':
            name, fact = await self.graphiti.get_edge_text(memory_id, group_id=project_id)
            return {'name': name, 'fact_snippet': fact[:120]}

        # mem0 path
        from fused_memory.models.scope import Scope

        scope = Scope(project_id=project_id)
        rec = await self.mem0.get(memory_id, scope)
        if rec is None:
            raise MemoryNotFoundError(memory_id)
        metadata = rec.get('metadata') or {}
        return {
            'category': rec.get('category'),
            'agent_id': metadata.get('agent_id'),
            'created_at': rec.get('created_at'),
        }

    # ------------------------------------------------------------------
    # Read: get_episodes
    # ------------------------------------------------------------------

    async def get_episodes(
        self,
        project_id: str = 'main',
        last_n: int = 10,
    ) -> list[dict]:
        """Retrieve raw episodes from Graphiti."""
        episodes = await self.graphiti.retrieve_episodes(
            group_ids=[project_id],
            last_n=last_n,
        )
        return [
            {
                'uuid': getattr(ep, 'uuid', None),
                'name': getattr(ep, 'name', None),
                'content': getattr(ep, 'content', None),
                'created_at': _created_at_to_utc_iso(getattr(ep, 'created_at', None)),
                'source': getattr(ep, 'source', None),
                'group_id': getattr(ep, 'group_id', None),
            }
            for ep in episodes
        ]

    # ------------------------------------------------------------------
    # Read: get_episode_content
    # ------------------------------------------------------------------

    async def get_episode_content(self, episode_uuid: str, project_id: str) -> str | None:
        """Fetch a single episode's original source text by UUID.

        Thin passthrough to graphiti.get_episode_by_uuid. Unlike get_episodes
        (recent-N only) or search results (edge/fact text only), this is the
        by-UUID episode-body lookup needed by the reconciliation promotion-time
        batch-plan gate (task 2033) to run is_batch_plan_framing on the
        episode's actual content. Returns None when the episode is not found
        (or on a fail-safe timeout inside get_episode_by_uuid).
        """
        node = await self.graphiti.get_episode_by_uuid(episode_uuid, group_id=project_id)
        return getattr(node, 'content', None) if node is not None else None

    # ------------------------------------------------------------------
    # Read: deterministic metadata count
    # ------------------------------------------------------------------

    async def count_memories_by_metadata(
        self,
        project_id: str,
        filters: dict,
    ) -> int:
        """Deterministic Mem0 count for the given metadata equality filters.

        Use this when you need a reliable key-equality lookup instead of
        semantic search — e.g. counting persistence markers or escalation
        markers keyed by ``flag_id``.  Goes through ``Mem0Backend.count_by_metadata``
        which talks to Qdrant's count API directly with a payload filter, so
        the result is exact rather than top-N-bounded.

        A ``task_id`` filter is normalized to str here too (task 2620
        amendment), symmetric with the add_memory/add_system_record
        write-side coercion — see ``_normalize_task_id_metadata``'s
        docstring. This protects a caller that queries with an int-typed
        task_id filter (forgetting the str convention) against the
        now-str-normalized data those write paths produce — it does NOT
        retroactively make historical int-typed task_id values, or
        anything written by a path that bypasses add_memory/
        add_system_record, matchable. Qdrant's payload filter is
        type-sensitive, so a str-coerced query can only ever match
        str-typed stored data; reaching legacy int-typed rows needs a
        separate backfill/migration, not this read-side coercion.
        """
        scope = Scope(project_id=project_id)
        filters = dict(filters)
        _normalize_task_id_metadata(filters)
        return await self.mem0.count_by_metadata(scope, filters)

    async def scan_memory_content(
        self,
        project_id: str,
        needles: list[str] | None = None,
        *,
        filters: dict | None = None,
        exhaustive: bool = False,
        limit: int | None = None,
    ) -> dict:
        """Literal substring scan over Mem0 payload TEXT (task 3083, WORK b).

        Thin passthrough to ``Mem0Backend.scan_payload_text``. Neither semantic
        (``search``) nor metadata equality
        (``count_memories_by_metadata``/``get_memories_by_metadata``) — it
        matches the memory TEXT itself, which is the capability whose absence
        made the tool-call XML leak corpus unsweepable: a leaked serialized
        fragment carries almost no semantic signal, so a live 2026-07-26
        semantic probe for it returned zero.

        *needles* and *filters* of ``None`` are passed through AS ``None``;
        the backend supplies the default needle set from
        ``fused_memory.utils.toolcall_xml_leak.PREFILTER_NEEDLES`` so the
        sentinels are defined in exactly one place. The caller's collections
        are copied before use and never mutated.

        A ``task_id`` filter is normalized to str on the COPY, exactly as
        ``count_memories_by_metadata``/``get_memories_by_metadata`` do — see
        ``_normalize_task_id_metadata``'s docstring. The backend turns every
        filter entry into a ``MatchValue`` equality condition and Qdrant's
        payload filter is TYPE-SENSITIVE, so without this an int ``task_id``
        would match nothing and return an empty scan with no error: a
        silently-wrong clean verdict, which is the exact failure class this
        tool exists to eliminate.

        Returns ``{'matches': [...], 'scanned': int, 'truncated': bool}``.

        A Qdrant read-timeout is PROPAGATED (raises ``TimeoutError``), not
        returned as an empty match list — a timed-out scan must never be
        mistaken for a clean corpus — and is surfaced at the MCP boundary as
        ``{'error', 'error_type': 'TimeoutError'}`` by ``@mcp_tool_errors``.
        """
        scope = Scope(project_id=project_id)
        scan_filters = None
        if filters is not None:
            scan_filters = dict(filters)
            _normalize_task_id_metadata(scan_filters)
        return await self.mem0.scan_payload_text(
            scope=scope,
            needles=list(needles) if needles is not None else None,
            filters=scan_filters,
            exhaustive=exhaustive,
            limit=limit,
        )

    async def get_memories_by_metadata(
        self,
        project_id: str,
        filters: dict,
        limit: int = 1000,
    ) -> list[dict]:
        """Deterministic (non-semantic) enumeration of Mem0 memories matching *filters*.

        Counterpart to ``count_memories_by_metadata``: where that returns only
        an int, this returns the full list of matching memory dicts so callers
        can inspect IDs, timestamps, and metadata for GC or pool-cap enforcement.

        Goes through ``Mem0Backend.scroll_by_metadata`` which talks to Qdrant's
        scroll API directly with a payload filter — NOT semantic search.  Using
        semantic search for pool GC is the failure mode that caused the
        stage2_cycle_summary pool to grow unboundedly (tasks 20e8c2f1, 45489c2b,
        db2ea69e).

        Returns a list of ``{'id', 'created_at', 'metadata'}`` dicts.

        A ``task_id`` filter is normalized to str here too — see
        ``count_memories_by_metadata`` above and ``_normalize_task_id_metadata``'s
        docstring (task 2620 amendment).

        A Qdrant read-timeout is PROPAGATED (raises ``TimeoutError``), not
        returned as ``[]`` — aligning with ``count_memories_by_metadata`` — and
        is surfaced at the MCP boundary as ``{'error', 'error_type':
        'TimeoutError'}`` via the ``@mcp_tool_errors`` decorator.
        """
        scope = Scope(project_id=project_id)
        filters = dict(filters)
        _normalize_task_id_metadata(filters)
        return await self.mem0.scroll_by_metadata(scope, filters, limit)

    async def get_memory_by_id(
        self,
        project_id: str,
        memory_id: str,
    ) -> dict | None:
        """Raw Mem0 point-id read (content + full payload), non-semantic.

        Fetches a single Mem0 record by its raw Qdrant point-id (the memory
        UUID) via ``Mem0Backend.get_point_by_id`` — bypassing BOTH semantic
        ranking (``search``) and metadata-equality filtering
        (``count_memories_by_metadata`` / ``get_memories_by_metadata``). Distinct
        from the fingerprint-only :meth:`get_memory` (which returns only
        {category, agent_id, created_at} through the mem0 layer and raises on a
        miss): this returns the full raw payload plus a ready-to-read content
        string.

        Returns ``{'id', 'content', 'metadata'}`` — where ``content`` is the
        first non-empty string among the canonical ``_MEM0_CONTENT_KEYS``
        (``data`` → ``memory`` → ``content``) and ``metadata`` is the FULL
        unprocessed Qdrant payload — or ``None`` on a genuine not-found.

        A Qdrant read-timeout is PROPAGATED (raises ``TimeoutError``), NOT
        collapsed into ``None`` — so the caller can distinguish "memory
        genuinely absent" from "backend timed out" (no-silent-fail invariant);
        surfaced at the MCP boundary as ``{'error', 'error_type': 'TimeoutError'}``.
        """
        scope = Scope(project_id=project_id)
        payload = await self.mem0.get_point_by_id(memory_id, scope)
        if payload is None:
            return None
        content = ''
        for key in _MEM0_CONTENT_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value:
                content = value
                break
        return {'id': memory_id, 'content': content, 'metadata': payload}

    async def get_mem0_deletion_tombstone(
        self,
        project_id: str,
        memory_id: str,
    ) -> dict | None:
        """Why a recon sweep deleted Mem0 record *memory_id*, or ``None``.

        Reads the ``mem0_tombstone`` ledger row written by
        :func:`~fused_memory.reconciliation.mem0_tombstone.record_mem0_deletion_tombstone`
        after every confirmed recon-initiated Mem0 delete, and returns its
        decoded payload: which sweep took the record (``deleter``), which run
        performed the deletion (``deleting_run_id``), when (``deleted_at``),
        and the victim's identifying metadata (``kind``, ``record_type``,
        ``source``, ``recon_pool``, ``run_id``, ``created_at``).

        The row's own timestamps are added as ``tombstone_created_at`` /
        ``tombstone_expires_at`` rather than merged bare, because the payload
        already carries a ``created_at`` — the VICTIM's, i.e. how old the
        evicted record was — while the row's is when the tombstone was
        written. Flattening them together would clobber the former with the
        latter, reproducing exactly the kind of run/timestamp conflation that
        made the original recon-gate-165 report unreadable.

        **Strictly additive** (task 3041): this is a sibling of
        :meth:`get_memory_by_id`, which is deliberately left untouched. Its
        ``None``-on-miss contract is load-bearing for at least three
        in-process callers (``reconciliation/citation_verifier.py``,
        reconciliation stage1, and ``server/recon_report.py``'s
        ``cite_memory``), all of which branch on ``is None``; widening it to
        return a dict-with-tombstone would silently flip every one of them.
        The tombstone is instead surfaced at the MCP boundary, on
        ``server/tools.py``'s ``get_memory_by_id`` not-found branch, so the
        exact query that dead-ended for the audit now self-explains.

        Fail-safe throughout — a tombstone is diagnostic, so a problem
        reading one must never be worse than not having it. No ledger wired
        (``recon_ledger_enabled=False``, same
        ``getattr(self, 'recon_ledger', None)`` precedent as
        :meth:`get_cycle_summary_presence`), no row, a payload that is
        undecodable or not a JSON object, and a *raising* store read (ledger
        not initialized, SQLite locked/corrupt, aiosqlite thread error) all
        return ``None``.

        The two FAULT cases — malformed payload and a raising store read —
        each log one WARNING (the latter with ``exc_info``); the two ordinary
        states (no ledger, no row) log nothing. That split is the point: a
        broken tombstone store must not be indistinguishable from "no
        tombstone exists", which is the same undiscoverability class task 3041
        was filed to fix (loud-over-silent / no-silent-fail-soft, see
        ``docs/legibility/design-invariants.md``). The store guard lives HERE
        rather than only at the MCP boundary so that "fail-safe throughout"
        holds for every caller, not just the one that happens to wrap it
        (reviewer finding robustness, task 3041 amendment pass).

        ``None`` therefore means "no readable tombstone", which covers both
        "never deliberately deleted" and "the tombstone expired past
        :data:`~fused_memory.reconciliation.mem0_tombstone.MEM0_TOMBSTONE_TTL_DAYS`".
        A tombstone proves deliberate deletion; its absence does not prove
        the converse.
        """
        ledger = getattr(self, 'recon_ledger', None)
        if ledger is None:
            return None
        try:
            record = await ledger.get_mem0_tombstone(project_id, memory_id)
        except Exception:
            # A FAULT, not an ordinary state — the caller cannot tell this
            # apart from "no tombstone exists" by the return value alone, so
            # it must be loud in the log even though the return degrades.
            logger.warning(
                'get_mem0_deletion_tombstone: tombstone store read FAILED for '
                'memory_id=%s in project=%s; reporting no tombstone',
                memory_id,
                project_id,
                exc_info=True,
                extra={'project_id': project_id, 'memory_id': memory_id},
            )
            return None
        if record is None:
            return None
        try:
            payload = json.loads(record.payload_json)
        except (TypeError, ValueError):
            payload = None
        if not isinstance(payload, dict):
            logger.warning(
                'get_mem0_deletion_tombstone: unreadable tombstone payload for '
                'memory_id=%s in project=%s; reporting no tombstone',
                memory_id,
                project_id,
                extra={'project_id': project_id, 'memory_id': memory_id},
            )
            return None
        return {
            **payload,
            'tombstone_created_at': record.created_at,
            'tombstone_expires_at': record.expires_at,
        }

    # ------------------------------------------------------------------
    # Read: cycle_summary ledger presence (task 2436, τ1)
    # ------------------------------------------------------------------

    async def get_cycle_summary_presence(
        self,
        project_id: str,
        run_id: str,
        stage: str,
    ) -> dict[str, Any]:
        """Report whether the AUTHORITATIVE cycle_summary ReconLedgerStore row exists.

        Thin read against ``ReconLedgerStore.get_by_identity``, mapping
        ``stage`` to the ledger's ``flag_type`` column to disambiguate the
        Stage 1 (``memory_consolidator``) vs Stage 2 (``task_knowledge_sync``)
        rows written under the same ``run_id`` (see
        ``summary_pool.write_cycle_summary``). This is the definitive presence
        check Stage 3 uses instead of relying solely on the best-effort Mem0
        mirror.

        Returns an INCONCLUSIVE ``{'present': False, 'ledger_available':
        False, ...}`` when no ledger is wired — mirrors
        ``write_cycle_summary`` returning ``False`` when unwired. Callers
        must not read that as a definitive absence.

        The returned dict also carries a ``remediation`` field (task 2652)
        sourced from the row's ``payload_json['remediation']`` marker (see
        ``summary_pool.write_cycle_summary``): ``True``/``False`` for a
        present row written under this change, or ``None`` when the row is
        absent, the ledger is unwired, the row predates this change and
        lacks the key (legacy), or the key is present but holds a non-bool
        value (corrupted/hand-edited row) — lets Stage 3 disambiguate a
        remediation run's expected missing Stage 1 (``memory_consolidator``)
        cycle_summary — Stage 1 still runs a focused turn on such a pass and
        may still emit findings; it only skips its own per-cycle summary
        write, by design (task 2652) — from a genuine Stage 1 write failure.
        """
        ledger = getattr(self, 'recon_ledger', None)
        if ledger is None:
            return {
                'present': False,
                'ledger_available': False,
                'project_id': project_id,
                'run_id': run_id,
                'stage': stage,
                'remediation': None,
            }
        # Presence is intentionally state-agnostic here: any row matching the
        # five-part identity counts as present, regardless of `record.state`.
        # This is safe because cycle_summary rows are always upserted with a
        # fixed state='active' by write_cycle_summary — no writer ever flips
        # a cycle_summary row to a different state — and expiry is a hard
        # DELETE via ReconLedgerStore.gc(), not a soft-delete/supersede. If a
        # future writer introduces a non-active cycle_summary state, revisit
        # this to filter on `record.state == 'active'`.
        record = await ledger.get_by_identity(
            project_id,
            record_kind='cycle_summary',
            task_id='',
            flag_type=stage,
            run_id=run_id,
        )
        remediation: bool | None = None
        if record is not None:
            # Guard ONLY this parse — not the get_by_identity read above — so
            # a malformed payload degrades to remediation=None rather than
            # crashing presence detection, while a genuine ledger read error
            # still propagates uncaught (test_ledger_read_error_is_not_swallowed_as_definitive_absent).
            try:
                payload = json.loads(record.payload_json)
            except (TypeError, ValueError):
                payload = None
            raw_remediation = payload.get('remediation') if isinstance(payload, dict) else None
            # write_cycle_summary always stamps a bool, so this should
            # already be True/False/absent — but coerce defensively: a
            # corrupted/hand-edited row with a non-bool value (e.g. the
            # string "yes") must degrade to None (report-as-missing), not
            # be trusted as a suppression signal for Stage 3 (task 2652
            # amendment).
            remediation = raw_remediation if isinstance(raw_remediation, bool) else None
        return {
            'present': record is not None,
            'ledger_available': True,
            'project_id': project_id,
            'run_id': run_id,
            'stage': stage,
            'remediation': remediation,
        }

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    #: How many child ids :meth:`delete_memory` lists in one scroll.
    #:
    #: The refusal message has to be READABLE — an unbounded listing of a
    #: pathological fan-out would produce an error string no agent or
    #: operator can act on, and the scroll fetches full payloads.  When the
    #: live count exceeds what the scroll returned, the listing is marked
    #: ``truncated`` ("at least N") rather than silently reading as
    #: exhaustive.  A CASCADE is not bounded by this: it re-scrolls until a
    #: pass yields no unvisited children.
    _CHILD_SCAN_LIMIT = 100

    async def _count_children(self, memory_id: str, *, project_id: str) -> int:
        """Live count of records whose ``metadata.parent_id`` is *memory_id*.

        The cheap exact primitive (Qdrant's count API), read fresh at every
        call — INV-3: corroborate against the store, never against
        remembered state.  A child can be written between two deletes, so a
        gate trusting a cached "childless" answer would be checking history.
        """
        return await self.count_memories_by_metadata(
            project_id, {'parent_id': memory_id}
        )

    async def _list_children(self, memory_id: str, *, project_id: str) -> list[str]:
        """Ids of *memory_id*'s children, bounded by ``_CHILD_SCAN_LIMIT``."""
        rows = await self.get_memories_by_metadata(
            project_id, {'parent_id': memory_id}, limit=self._CHILD_SCAN_LIMIT
        )
        return [row['id'] for row in rows]

    async def list_descendant_ids(
        self, memory_id: str, *, project_id: str
    ) -> DescendantScan:
        """Every descendant of *memory_id*, deepest-first — WITHOUT deleting.

        The read-only twin of :meth:`_cascade_delete_children`: same
        primitives (:meth:`_count_children` / :meth:`_list_children`), same
        visited-set termination for self-parent records and cycles, same
        deepest-first order, no new backend call and no second tree-walk
        (INV-5).  The enumeration a caller GATES on and the traversal the
        cascade PERFORMS therefore cannot disagree about the shape of the
        tree — a disagreement would mean checking one set and destroying
        another.

        Public and side-effect-free on purpose.  The citation-repoint gate
        lives at the MCP tool layer, which needs to ask "what would this
        cascade destroy?" *before* anything is destroyed; a hook that
        mutated would turn look-before-you-leap into the leap.

        ONE deliberate divergence from the cascade, surfaced as data rather
        than hidden: ``truncated``.  ``_cascade_delete_children`` re-scrolls
        past ``_CHILD_SCAN_LIMIT`` only because DELETING a page is what
        makes the next one visible; a non-mutating walk has no such lever,
        so a fan-out wider than the bound genuinely cannot be fully seen
        here.  Do not "fix" this by copying the cascade's ``while`` loop
        into this context — it would spin on the same page forever.  Say so
        instead, and let the caller refuse.

        Returns:
            DescendantScan: ``ids`` deepest-first (the order the cascade
            would destroy them in), excluding *memory_id* itself; and
            ``truncated``, true when any visited node reported more children
            than the bounded scroll returned.
        """
        # Seeded with the target so a record that is its own parent, or a
        # cycle leading back to the target, terminates instead of recursing.
        visited = {memory_id}
        ordered: list[str] = []
        truncated = False

        async def walk(node: str) -> None:
            nonlocal truncated
            # Count first, scroll only on a non-zero count — the same cheap
            # ordering the refusal gate uses, so a leaf costs one exact
            # count and no payload fetch.
            count = await self._count_children(node, project_id=project_id)
            if not count:
                return
            children = await self._list_children(node, project_id=project_id)
            if len(children) < count:
                truncated = True
            for child in children:
                if child in visited:
                    continue
                visited.add(child)
                await walk(child)
                # Appended AFTER its own subtree: post-order is what makes
                # the listing deepest-first.
                ordered.append(child)

        await walk(memory_id)
        return DescendantScan(ids=ordered, truncated=truncated)

    async def refuse_if_children(self, memory_id: str, *, project_id: str) -> None:
        """Raise ``ParentHasChildrenError`` if *memory_id* still has children.

        PUBLIC and side-effect-free (it either raises or returns), for the
        same reason :meth:`list_descendant_ids` is: the MCP tool layer needs
        to ask "would this delete be refused?" BEFORE it runs the citation
        gate, whose repoint pass mutates live task metadata.  Without that
        pre-flight a delete of a cited PARENT rewrote every citation to the
        replacement and only then hit this refusal — mutation left behind by
        an operation that reported failure, and the exact asymmetry the
        cascade path avoids by enumerating before it gates.  Exposing the
        one gate (rather than a count the caller re-wraps in its own error)
        keeps the refusal's construction — ids, count, ``truncated``,
        registry pointer — with exactly one home (INV-5).

        Count FIRST, scroll only on a non-zero count: the count is exact and
        cheap while the scroll fetches full payloads, and ``delete_memory``
        has six in-repo recon callers (including bulk pool GC) that would
        otherwise pay for a listing nobody reads.

        A scroll returning FEWER ids than the count — the bound above, or a
        concurrent write between the two reads — still refuses, marked
        ``truncated``.  Downgrading a disagreement to "no children" would be
        precisely the silent orphan this gate exists to prevent; presenting
        a partial list as exhaustive would understate it.
        """
        child_count = await self._count_children(memory_id, project_id=project_id)
        if child_count == 0:
            return
        child_ids = await self._list_children(memory_id, project_id=project_id)
        raise ParentHasChildrenError(
            parent_id=memory_id,
            child_ids=child_ids,
            truncated=len(child_ids) < child_count,
        )

    async def _cascade_delete_children(
        self,
        memory_id: str,
        *,
        project_id: str,
        agent_id: str | None,
        session_id: str | None,
        causation_id: str | None,
        _source: str,
        visited: set[str] | None,
    ) -> list[str]:
        """Delete *memory_id*'s subtree, depth-first, and return EVERY id it took.

        The return value is the whole destroyed set — grandchildren
        included, deepest-first — not just the direct children.  It is what
        the caller reports as ``cascaded_child_ids`` on the result, the
        journal row and the ``memory_deleted`` event, and an MCP caller
        never sees the server's journal: naming only the direct children
        would tell them a SMALLER set was destroyed than actually was, and
        leave them to reconstruct the rest from a log they cannot read.

        CHILDREN FIRST, parent last — the caller deletes the parent only
        after this returns.  Parent-first would re-open precisely the orphan
        window this gate closes: a crash between the two leaves live
        children pointing at a dead uuid, still recognised as children,
        still suppressed from grouped search, content unreachable while
        remaining in Qdrant.  Children-first fails safe: the surviving state
        is "parent alive, some children gone", which the refusal gate still
        protects and an operator can retry.

        Each child is deleted by RE-ENTERING :meth:`delete_memory` rather
        than by a local ``mem0.delete`` loop, so every child gets its own
        write-journal row, its own reconciliation event and its own child
        gate for free — no second, unguarded delete implementation to drift
        (INV-5).

        The ``await`` loop is SEQUENTIAL on purpose: an ``asyncio.gather``
        would destroy the ordering the contract depends on (and trip the
        repo's gather-convention guard).

        *visited* terminates self-parent records and parent cycles: it is
        seeded with the parent id and carries every id the chain has
        committed to deleting, so a cycle's second visit is filtered out
        instead of recursing.  The loop re-scrolls until a pass yields
        nothing unvisited, so a fan-out wider than ``_CHILD_SCAN_LIMIT`` is
        fully covered — the id LISTING is bounded, the cascade is not.
        :meth:`list_descendant_ids` walks the same tree read-only and
        therefore CANNOT re-scroll like this (deleting a page is what makes
        the next one visible), which is why it reports ``truncated`` where
        this loop simply keeps going.

        Then CORROBORATE (INV-3, after acting): re-count the children and
        raise rather than delete the parent if any survived.  Survivors are
        measured against the ENCLOSING frames' in-flight set only, never
        against the ids this frame just deleted — otherwise a child whose
        delete silently did not take would be filtered out as "already
        handled", which is exactly the partial-failure this re-read exists
        to catch.  Without it the operation reports success while leaving
        an orphan behind.
        """
        # Records an ENCLOSING frame is already committed to deleting. They
        # are excluded from the corroboration below — an ancestor still in
        # flight is not an orphan-to-be. Ids THIS frame deletes are
        # deliberately NOT added here, so a delete that silently did not
        # take resurfaces as a survivor instead of being explained away.
        in_flight = set(visited) if visited else set()
        in_flight.add(memory_id)
        visited = set(in_flight)
        deleted: list[str] = []

        while await self._count_children(memory_id, project_id=project_id):
            child_ids = await self._list_children(memory_id, project_id=project_id)
            fresh = [cid for cid in child_ids if cid not in visited]
            if not fresh:
                break
            for child_id in fresh:
                visited.add(child_id)
                child_result = await self.delete_memory(
                    memory_id=child_id,
                    store='mem0',
                    project_id=project_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    causation_id=causation_id,
                    _source=_source,
                    cascade=True,
                    _visited=visited,
                    _cascade_parent=memory_id,
                )
                # The child's OWN subtree went first, so its ids precede it
                # here — the same deepest-first order the deletes actually
                # ran in. Dropping this frame's return value would report
                # A→B→C as having destroyed only B.
                deleted.extend(child_result.get('cascaded_child_ids') or [])
                deleted.append(child_id)

        if await self._count_children(memory_id, project_id=project_id):
            survivors = [
                cid
                for cid in await self._list_children(
                    memory_id, project_id=project_id
                )
                if cid not in in_flight
            ]
            if survivors:
                raise ParentHasChildrenError(
                    parent_id=memory_id, child_ids=survivors
                )

        return deleted

    async def delete_memory(
        self,
        memory_id: str,
        store: str,
        project_id: str = 'main',
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
        *,
        cascade: bool = False,
        _visited: set[str] | None = None,
        _cascade_parent: str | None = None,
    ) -> dict:
        """Delete a memory from the specified store.

        REFUSES to orphan children (task 3197, leaf δ; PRD V3's lifecycle
        contract — "no operation may silently orphan a child or dangle a
        pointer it could have seen").  Deleting a Mem0 record that other
        records point at via ``metadata.parent_id`` raises
        :class:`~fused_memory.memory_metadata.ParentHasChildrenError`
        listing the child ids, BEFORE any backend call, journal row or
        reconciliation event — so a refused delete leaves nothing claiming a
        deletion happened.  The caller's explicit way out is
        ``cascade=True``.

        The gate is UNCONDITIONAL — deliberately not behind
        ``memory_metadata.enforce``, unlike the V1 shape checks.  It is a
        lifecycle safety gate, not a vocabulary check: behind a default-off
        flag the orphan hole would stay open exactly as long as the flag
        stayed off, i.e. the machinery would ship and none of the
        protection.  Shipping it on is safe because leaf α measured
        ``metadata.parent_id`` at zero live corpus footprint — there are no
        existing children, so no live delete (including the six in-repo
        recon callers) can regress.

        The child check is a LIVE re-read per INV-3, never cached state, and
        it is charged only where the relationship can exist: ``parent_id``
        is a Mem0 payload key, so the graphiti arm keeps its current
        zero-extra-round-trip cost.  On the common childless path the cost
        is ONE exact Qdrant count and ZERO scrolls; the payload scroll is
        paid only when there is something to list, because the error
        contract needs the child *ids* and a count cannot supply them.

        ``cascade=True`` is the caller's explicit opt-in: it deletes the
        CHILDREN FIRST and the parent last, then re-checks.  See
        :meth:`_cascade_delete_children`.  The result's
        ``cascaded_child_ids`` — and the journal row and ``memory_deleted``
        event that carry it — name EVERY record the cascade destroyed,
        grandchildren included, deepest-first.  ``cascade`` is Mem0-only:
        ``store='graphiti'`` with ``cascade=True`` raises ``ValueError``
        rather than performing a silent plain delete (see below).

        ``memory_id`` is validated for SHAPE ONLY: it must be a canonical
        36-character UUID. A truncated id (e.g. an 8-char hex prefix lifted out
        of a search-result snippet) raises rather than silently no-opping —
        both backends treat a miss as "already deleted", so without this guard
        such a call got a confirming ``{'status': 'deleted'}`` envelope, a
        ``success=True`` journal entry and a ``memory_deleted`` event while
        nothing was removed.

        EXISTENCE IS NOT CHECKED, and the difference is user-visible: a
        well-formed UUID that no longer resolves — a stale id copied out of an
        old report, a survivor id from an earlier consolidation — still reports
        ``deleted``, for exactly the same backend reason. Closing that half
        needs a per-store existence read: ``update_memory`` below already does
        it for its Qdrant arm (see the §5(c) read-leg comment there), while the
        Graphiti arm additionally needs a ``remove_edge`` that distinguishes
        not-found from already-deleted. Deliberately out of scope here — task
        3132 closes the malformed-shape half only.

        The guard sits above the store branch so ONE check covers both the
        Graphiti and Mem0 paths, and above the journal write and event emission
        so a rejected delete leaves no false audit trail. It sits BELOW
        ``SourceStore(store)`` so a call that is wrong in both ways reports the
        bad store first — the same store-then-shape precedence the MCP boundary
        gives agents, rather than the inverse for internal callers.

        Raises:
            ParentHasChildrenError: the target still has children and
                ``cascade`` was not requested — or a child SURVIVED a
                requested cascade, in which case the parent is left in
                place too.
            ValueError: ``cascade=True`` was combined with a non-Mem0
                store, which no store branch can honour.
        """
        scope = Scope(project_id=project_id)
        source = SourceStore(store)
        # `cascade` is MEM0-ONLY, and an unhonourable request is refused
        # rather than dropped. The graphiti arm has no `metadata.parent_id`
        # to recurse on, so tolerating the flag there meant a plain delete
        # returning a bare {'status': 'deleted'} — while the `memory_deleted`
        # event still carried `cascade: True` with an empty child list,
        # recording a cascade as requested-and-satisfied when nothing
        # recursive ever ran. Refusing keeps the audit trail unable to lie
        # (loud-over-silent-degradation).
        #
        # Placed with the store check and BEFORE `require_full_uuid` so this
        # layer and the MCP boundary agree on precedence: store validity,
        # then store/cascade compatibility, then id shape.
        if cascade and source != SourceStore.mem0:
            raise ValueError(
                f'cascade=True is not supported for store={store!r}: parent/child '
                'links are the Mem0 payload key metadata.parent_id, so a '
                f'{store} record has no children to cascade to. Retry without '
                'cascade if a plain delete of this record is what was meant.'
            )
        require_full_uuid(memory_id, field_name='memory_id')

        write_op_id = str(uuid_mod.uuid4())
        cascaded_child_ids: list[str] = []

        if source == SourceStore.graphiti:
            await self._journaled_backend_call(
                write_op_id=write_op_id,
                causation_id=causation_id,
                backend='graphiti',
                operation='remove_edge',
                payload={'memory_id': memory_id},
                coro=self.graphiti.remove_edge(memory_id, group_id=project_id),
            )
            result = {'status': 'deleted', 'store': 'graphiti', 'id': memory_id}
        else:
            # Child gate — BEFORE the backend call, the journal row and the
            # event, so a refused delete leaves no trace claiming a
            # deletion. `parent_id` is a Mem0 payload key, which is why this
            # is in the mem0 arm only.
            if cascade:
                cascaded_child_ids = await self._cascade_delete_children(
                    memory_id,
                    project_id=project_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    causation_id=causation_id,
                    _source=_source,
                    visited=_visited,
                )
            else:
                await self.refuse_if_children(memory_id, project_id=project_id)

            del_result = await self._journaled_backend_call(
                write_op_id=write_op_id,
                causation_id=causation_id,
                backend='mem0',
                operation='delete',
                payload={'memory_id': memory_id},
                coro=self.mem0.delete(memory_id, scope),
            )
            result = {'status': 'deleted', 'store': 'mem0', 'id': memory_id, **del_result}
            if cascaded_child_ids:
                result['cascaded_child_ids'] = cascaded_child_ids

        if self._write_journal:
            await self._write_journal.log_write_op(
                write_op_id=write_op_id,
                causation_id=causation_id,
                source=_source,
                operation='delete_memory',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params={
                    'memory_id': memory_id,
                    'store': store,
                    'cascade': cascade,
                    # Whose cascade took this record. Without it a cascaded
                    # delete is indistinguishable from a direct one in the
                    # journal, and the PRD's "children deleted too,
                    # journalled" signal is only half legible.
                    'cascade_parent_id': _cascade_parent,
                },
                result_summary=result,
                success=True,
            )

        await self._emit_event(ReconciliationEvent(
            id=str(uuid_mod.uuid4()),
            type=EventType.memory_deleted,
            source=EventSource.agent,
            project_id=project_id,
            timestamp=datetime.now(UTC),
            payload={
                'memory_id': memory_id,
                'store': store,
                'cascade': cascade,
                'cascade_parent_id': _cascade_parent,
                'cascaded_child_ids': cascaded_child_ids,
            },
        ))

        return result

    def _record_content_amend(self, project_id: str, agent_id: str | None) -> None:
        """Count one in-place content amendment; escalate on a burst (INV-4).

        Post-write and never blocking: this is a monitoring alarm, not a rate
        limiter. Crossing the threshold must not reject the write that crossed
        it, or a legitimate large consolidation cycle would fail mid-run over
        its own success count.

        Counts the CONTENT arm only. A metadata patch is cheap to notice and
        cheap to correct; counting patches would drown the signal that a silent
        content-rewrite loop is running.

        One counter per ``agent_id``, so two independently-busy agents cannot
        sum into a false alarm. The threshold and window are read LIVE off the
        shared config and passed into ``record()`` per call — captured once,
        they would make both green-tier leaves restart-only in disguise.
        """
        label = agent_id or '<unattributed>'
        counter = self._mem0_update_storm_counters.get(label)
        if counter is None:
            counter = StormCounter(time_provider=self._mem0_update_storm_time_provider)
            self._mem0_update_storm_counters[label] = counter

        cfg = getattr(self.config, 'mem0_update', None)
        threshold = getattr(cfg, 'storm_threshold', None)
        window_seconds = getattr(cfg, 'storm_window_seconds', None)
        if not isinstance(threshold, int) or not isinstance(window_seconds, int | float):
            return

        storm = counter.record(
            threshold=threshold,
            window_seconds=float(window_seconds),
            label=label,
        )

        # Evict counters whose window has gone empty. Each counter self-prunes
        # its own deque, but nothing would drop the counter OBJECT, and
        # ``agent_id`` is caller-supplied and unbounded in cardinality — the
        # gate is a self-reported prefix match, so a widened prefix admits
        # arbitrary suffixes (``recon-stage-1-run-<uuid>`` mints a fresh key
        # every run). A server designed to run for weeks between restarts would
        # otherwise accumulate one dead counter per agent it ever saw.
        #
        # Runs on EVERY amend, not just a breach: the leak is on the common
        # path. It is O(live agents) because the sweep is itself what keeps
        # that from becoming O(agents ever seen). See StormCounter.prune on why
        # dropping an empty counter is behaviour-preserving.
        for other, dormant in list(self._mem0_update_storm_counters.items()):
            if other != label and dormant.prune(float(window_seconds)) == 0:
                del self._mem0_update_storm_counters[other]

        if storm is None:
            return

        # Never let the alarm's own failure reach the caller: the write already
        # landed, and turning a completed amendment into an exception would be
        # strictly worse than losing the signal. The escalator is itself
        # never-raise; this is the belt to its braces.
        try:
            self._mem0_update_storm_escalator.report_storm(
                project_id=project_id,
                agent_id=label,
                count=storm['count'],
                threshold=storm['threshold'],
                window_seconds=storm['window_seconds'],
            )
        except Exception:
            logger.exception(
                'update_memory storm escalation failed for agent %r in project %r '
                '(count=%s); the amendment itself succeeded',
                label, project_id, storm['count'],
            )

    @staticmethod
    def _apply_metadata_delta(
        existing_custom: dict[str, Any],
        *,
        metadata_patch: dict | None,
        metadata_delete_keys: list[str] | None,
        metadata_mode: str,
    ) -> dict[str, Any]:
        """Apply an ``update_memory`` metadata delta to a record's CUSTOM subset.

        The single home for merge / replace / delete semantics (task 3088). Both
        the metadata-only routes and the combined content+metadata fold call
        this, so a caller gets the same resulting metadata whether or not it
        also amended the content — semantics that drifted between the two arms
        would be invisible to any test that exercised only one of them (INV-5).

        *existing_custom* is the mem0-owned-key-stripped subset from
        :func:`split_managed_metadata`; mem0-owned keys never reach here, which
        is why nothing below has to defend against clobbering them.

        ``metadata_mode='replace'`` replaces the custom subset with exactly what
        *metadata_patch* supplies, PLUS any ``_FUSED_MEMORY_OWNED_METADATA_KEYS``
        carried over from *existing_custom* — never the whole Qdrant payload.
        The carry-through is what stops a routine re-tag from silently evicting
        the record from every category-scoped search (``category`` is a Qdrant
        payload filter, so losing it has no symptom at all); a *metadata_patch*
        that names the key explicitly still wins, so deliberate
        re-categorization needs no special case. Deletions apply after the
        merge. Returns a fresh dict; the input is not mutated.
        """
        if metadata_mode == 'replace':
            # Seed with the protected subset rather than {}: replace still means
            # replace for ordinary custom keys, but a key nothing restores must
            # not be destroyable by omission.
            new_custom = {
                k: v for k, v in existing_custom.items()
                if k in _FUSED_MEMORY_OWNED_METADATA_KEYS
            }
        else:
            new_custom = dict(existing_custom)
        new_custom.update(metadata_patch or {})
        for key in metadata_delete_keys or ():
            new_custom.pop(key, None)
        return new_custom

    async def update_memory(
        self,
        memory_id: str,
        project_id: str = 'main',
        content: str | None = None,
        metadata_patch: dict | None = None,
        metadata_delete_keys: list[str] | None = None,
        metadata_mode: str = 'merge',
        reason: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        emit_event: bool = False,
        _source: str = 'mcp_tool',
    ) -> dict:
        """Amend a Mem0 record's content and/or patch its metadata IN PLACE.

        Task 3088; contract in ``plans/mem0-in-place-update-decision.md`` §3.
        The Qdrant point id is preserved, so the record keeps its identity and
        every reference to it stays valid — which is the whole point, and also
        why the tool sits behind an authorization gate and a storm alarm: an
        in-place amendment is invisible to every downstream reader.

        Argument validation (arm presence, reserved-key rejection, contradictory
        key lists, ``metadata_mode`` values) belongs to the MCP tool layer, which
        fails those loud before dispatching here — mirroring how ``update_edge``
        splits its boundary checks from its write path.

        Returns the ``{'status': 'updated', 'store': 'mem0', 'id': memory_id,
        ...}`` envelope on success, or a structured ``{'error_type': ...}``
        rejection. The id is echoed so a caller can assert identity stability
        straight from the response instead of re-fetching.

        *emit_event* forces a ``memory_updated`` event on a metadata-only route,
        which is otherwise silent (a patch leaves the record saying the same
        thing). It is deliberately INTERNAL: no MCP-level argument surfaces it
        in this ship, because no concrete consumer needs it yet and an
        unexercised knob on the event channel is one more thing to get wrong.
        A content amend always emits, flag or not.
        """
        write_op_id = str(uuid_mod.uuid4())

        # Journal params: truncated copies for the audit row only. The full
        # values go to the backend — same convention as update_edge's fact.
        params: dict[str, Any] = {'memory_id': memory_id, 'metadata_mode': metadata_mode}
        if content is not None:
            params['content'] = content[:200]
        if metadata_patch:
            params['metadata_patch'] = metadata_patch
        if metadata_delete_keys:
            params['metadata_delete_keys'] = list(metadata_delete_keys)
        if reason:
            params['reason'] = reason[:200]

        # §5(c) read leg — runs FIRST, in EVERY arm, before any write.
        #
        # Not merely a convenience read for the metadata-reforwarding dance: it
        # is the existence check. Qdrant's set_payload/delete_payload return
        # acknowledged/completed for an UNKNOWN point id rather than an error,
        # so the metadata-only fast paths would otherwise emit a success
        # envelope AND a journal row for a write that touched nothing.
        #
        # A TimeoutError from here PROPAGATES untouched. Mem0Backend.
        # get_point_by_id deliberately does not swallow it (unlike get()), which
        # is what keeps "genuinely absent" distinguishable from "backend timed
        # out"; catching both into one MemoryNotFound outcome would throw that
        # distinction away at the one layer that still has it.
        existing = await self.get_memory_by_id(project_id=project_id, memory_id=memory_id)
        if existing is None:
            return {
                'error': (
                    f'Memory {memory_id!r} does not exist in mem0 for project '
                    f'{project_id!r}; nothing was updated.'
                ),
                'error_type': 'MemoryNotFound',
                'store': 'mem0',
                'id': memory_id,
            }

        # The FULL raw Qdrant payload — mem0-owned keys and custom provenance
        # keys alike. Copied so the arms below can compute a delta against it
        # without mutating the value the read leg returned.
        existing_payload: dict[str, Any] = dict(existing.get('metadata') or {})
        managed, existing_custom = split_managed_metadata(existing_payload)

        # ONE delta computation, consumed by the two routes that must construct
        # a full metadata dict themselves: the content arm's ``mem0.update``
        # (whose backend starts from a FRESH payload) and the metadata-only
        # ``overwrite_payload`` route. Sharing it is what keeps merge / replace
        # / delete semantics from drifting between the combined path and the
        # metadata-only path — a caller must get the same resulting metadata
        # whether or not it also amended the content.
        #
        # The set_payload / delete_payload fast paths deliberately do NOT read
        # it: they hand the raw patch / key list to Qdrant and let it apply
        # merge and delete SERVER-side, which is the entire reason those routes
        # can skip a read-modify-write. So the INV-5 single-home claim is
        # narrower than "every arm calls _apply_metadata_delta": merge and
        # delete semantics have two implementations that have to agree — this
        # one and Qdrant's primitives. ``TestMetadataFastPathEquivalence`` pins
        # that agreement, so a new rule added here (a second protected key,
        # say) fails a test instead of silently splitting the routes apart.
        new_custom = self._apply_metadata_delta(
            existing_custom,
            metadata_patch=metadata_patch,
            metadata_delete_keys=metadata_delete_keys,
            metadata_mode=metadata_mode,
        )

        scope = Scope(project_id=project_id)

        if content is not None:
            # Journal the PRIOR text beside the new one, so the audit row is a
            # genuine before/after rather than a record of what the text was
            # rewritten TO. The read leg above already fetched it, so this costs
            # nothing; without it the storm escalation's "inspect the affected
            # records" instruction sends an operator to a journal with nothing
            # to diff against, which is exactly the forensic evidence a
            # silent-rewrite alarm exists to make reachable. Truncated at 200
            # like `content`, same convention update_edge uses for `fact`.
            params['content_before'] = (existing.get('content') or '')[:200]

            # Content-amend arm, folding in any metadata delta rather than
            # issuing a second write for it — a combined call must never leave
            # the record carrying new content with stale metadata.
            #
            # Forward ONLY the custom subset as metadata=: mem0's
            # _update_memory starts a FRESH payload from deepcopy(metadata) and
            # re-attaches just its own nine keys, so anything custom that is not
            # forwarded here is destroyed. This is the read-modify-forward dance
            # tag_cgl_eta_rehome_scope.apply_tags already had to solve; the
            # mem0-owned keys are deliberately NOT forwarded because mem0
            # restores or recomputes each of them itself.
            result_data = await self._journaled_backend_call(
                write_op_id=write_op_id,
                causation_id=causation_id,
                backend='mem0',
                operation='update_memory',
                payload=params,
                coro=self.mem0.update(
                    memory_id, content, scope, metadata=new_custom,
                ),
            )
            result: dict[str, Any] = {
                'status': 'updated',
                'store': 'mem0',
                'id': memory_id,
                'content_amended': True,
                'metadata_patched': bool(metadata_patch or metadata_delete_keys),
            }
            if isinstance(result_data, dict):
                result.update(result_data)
                # Re-stamp the envelope keys the backend response must not be
                # able to overwrite — 'id' above all, since the whole contract
                # is that the caller can read identity stability off it.
                result['status'] = 'updated'
                result['store'] = 'mem0'
                result['id'] = memory_id
        else:
            # Metadata-only arm — §5(b)'s decision table. Deliberately routes
            # AROUND mem0's Memory.update, which would re-embed the content,
            # rewrite updated_at and append a history row for what may be a
            # purely cosmetic tag.
            #
            # The three primitives are not interchangeable: set_payload and
            # delete_payload are native PARTIAL operations, so the new payload
            # need not be computed from the old one; overwrite_payload replaces
            # the ENTIRE point payload and therefore requires the mem0-owned
            # subset re-attached underneath, or the point loses its own
            # data/hash/created_at and becomes unreadable by mem0's get/search.
            wants_replace = metadata_mode == 'replace'
            if (metadata_patch and metadata_delete_keys) or wants_replace:
                # One read-modify-overwrite_payload write. Chosen over
                # set_payload-then-delete_payload because two round-trips have
                # no ordering guarantee, no atomicity and no rollback: a failed
                # second call leaves the record half-patched while the journal
                # row claims the whole edit landed.
                #
                # overwrite_payload replaces the ENTIRE point payload, so the
                # mem0-owned subset rides along underneath — omit it and the
                # point loses its own data/hash/created_at and stops being
                # readable by mem0's own get/search.
                new_payload = {**managed, **new_custom}
                operation = 'update_memory_overwrite_payload'
                coro = self.mem0.overwrite_payload(memory_id, new_payload, scope)
            elif metadata_patch:
                # Qdrant merges server-side, so unlisted pre-existing keys
                # survive without this layer reconstructing the whole payload.
                operation = 'update_memory_set_payload'
                coro = self.mem0.set_payload(memory_id, dict(metadata_patch), scope)
            else:
                operation = 'update_memory_delete_payload'
                coro = self.mem0.delete_payload(
                    memory_id, list(metadata_delete_keys or ()), scope,
                )

            await self._journaled_backend_call(
                write_op_id=write_op_id,
                causation_id=causation_id,
                backend='mem0',
                operation=operation,
                payload=params,
                coro=coro,
            )
            result = {
                'status': 'updated',
                'store': 'mem0',
                'id': memory_id,
                'content_amended': False,
                'metadata_patched': True,
            }

        if self._write_journal:
            await self._write_journal.log_write_op(
                write_op_id=write_op_id,
                causation_id=causation_id,
                source=_source,
                operation='update_memory',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params=params,
                result_summary=result,
                success=True,
            )

        # Event on the content arm always; on a metadata-only route only when a
        # caller explicitly opts in. A metadata patch leaves the record saying
        # exactly what it said before, so there is nothing for a downstream
        # consolidator to re-read — announcing it would be noise on a channel
        # whose consumers act on changed CONTENT.
        if content is not None or emit_event:
            await self._emit_event(ReconciliationEvent(
                id=str(uuid_mod.uuid4()),
                type=EventType.memory_updated,
                source=EventSource.agent,
                project_id=project_id,
                timestamp=datetime.now(UTC),
                payload={'memory_id': memory_id, 'store': 'mem0'},
            ))

        if content is not None:
            self._record_content_amend(project_id=project_id, agent_id=agent_id)

        return result

    async def update_edge(
        self,
        edge_uuid: str,
        fact: str | None = None,
        project_id: str = 'main',
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
        invalid_at: datetime | None = None,
        clear_invalid_at: bool = False,
    ) -> dict:
        """Update an existing Graphiti edge's fact text and/or invalidate it.

        At least one of ``fact``, ``invalid_at``, or ``clear_invalid_at`` must
        be supplied. Setting ``invalid_at`` marks the edge as superseded as of
        that moment (used by reconciliation to retire contradicted facts without
        destroying their audit trail). Setting ``clear_invalid_at=True`` resets
        ``invalid_at`` to ``None``, restoring the edge to active status.

        **Guard-2 verification (TOCTOU note):** when *fact* is supplied, a
        ``get_edge_text`` readback is performed after the save to confirm
        persistence. When *clear_invalid_at* is supplied, a separate
        ``get_edge_invalid_at`` readback confirms the edge was actually
        restored to active status — this is independent of the fact
        verdict, so a combined ``fact`` + ``clear_invalid_at`` call is
        ``verified`` only when *both* readbacks confirm. A pure
        ``invalid_at=<timestamp>`` supersede (no fact, no clear) is not
        readback-verified — it persists reliably via the same write path
        used for ``fact``. There is a small TOCTOU window between each save
        and its readback — if a concurrent writer updates the same edge in
        that window (another reconciliation cycle, an interactive agent, or
        a retry) the readback may return a different value and ``verified``
        will be ``False`` even though *our* save did persist.  This is a
        known false-negative mode; reconciliation is mostly single-writer
        per edge so it is rare in practice.  Undercounting ``edges_updated``
        (the conservative outcome) is safer than overcounting, so the
        behaviour is intentional.  Both readbacks (``get_edge_text`` and
        ``get_edge_invalid_at``) resolve their Cypher through the same
        per-``group_id`` graph object (``GraphitiBackend._graph_for``) that
        the backend's explicit ``clear_invalid_at`` write uses, so
        read-after-write consistency for *our own* write is expected within
        that connection — the TOCTOU risk above is about a *different*
        writer racing us, not about our own write lagging its readback.
        """
        if fact is None and invalid_at is None and not clear_invalid_at:
            raise ValueError('update_edge requires fact, invalid_at, or clear_invalid_at to be set')
        write_op_id = str(uuid_mod.uuid4())

        params: dict[str, Any] = {'edge_uuid': edge_uuid}
        if fact is not None:
            # Truncated copy for journal/payload logging only.  The full fact
            # is passed to graphiti.update_edge and used for the Guard-2
            # readback comparison.  A verified=False journal entry will show
            # params['fact'] (truncated) as the 'expected' value — the actual
            # equality check was against the un-truncated string.
            params['fact'] = fact[:200]
        if invalid_at is not None:
            params['invalid_at'] = invalid_at.isoformat()
        if clear_invalid_at:
            params['clear_invalid_at'] = True

        result_data = await self._journaled_backend_call(
            write_op_id=write_op_id,
            causation_id=causation_id,
            backend='graphiti',
            operation='update_edge',
            payload=params,
            coro=self.graphiti.update_edge(
                edge_uuid, fact, group_id=project_id, invalid_at=invalid_at,
                clear_invalid_at=clear_invalid_at,
            ),
        )
        result = {'status': 'updated', 'store': 'graphiti', **result_data}

        # Guard 2: post-write persistence verification.
        # Only when fact text was supplied — invalid_at-only updates have nothing to compare.
        if fact is not None:
            try:
                _name, readback_fact = await self.graphiti.get_edge_text(
                    edge_uuid, group_id=project_id
                )
                result['verified'] = readback_fact == fact
                if not result['verified']:
                    result['verification_error'] = (
                        f'Readback fact mismatch: expected {fact!r}, got {readback_fact!r}'
                    )
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                # Log unexpected exceptions (e.g. AttributeError from a future
                # signature change) so they remain observable in logs even
                # though we don't re-raise — the save succeeded and must be
                # reported; only the verification signal is lost.
                logger.exception(
                    'update_edge verification readback failed for edge %s',
                    edge_uuid,
                    exc_info=e,
                )
                result['verified'] = False
                result['verification_error'] = f'{type(e).__name__}: {e}'
        else:
            # No fact supplied — nothing to compare yet. May be revised below
            # by the clear_invalid_at readback; a pure invalid_at=<timestamp>
            # supersede (no fact, no clear) stays trivially verified here.
            result['verified'] = True

        # Guard 2b: post-write clear_invalid_at verification, independent of
        # the fact verdict above. Combines via AND — a fact+clear call is
        # verified only when both readbacks confirm.
        if clear_invalid_at:
            try:
                readback_invalid_at = await self.graphiti.get_edge_invalid_at(
                    edge_uuid, group_id=project_id
                )
                cleared = readback_invalid_at is None
                result['verified'] = bool(result.get('verified', True)) and cleared
                if not cleared:
                    error_msg = f'invalid_at not cleared: readback={readback_invalid_at!r}'
                    existing_error = result.get('verification_error')
                    result['verification_error'] = (
                        f'{existing_error}; {error_msg}' if existing_error else error_msg
                    )
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.exception(
                    'update_edge clear_invalid_at verification readback failed for edge %s',
                    edge_uuid,
                    exc_info=e,
                )
                result['verified'] = False
                error_msg = f'{type(e).__name__}: {e}'
                existing_error = result.get('verification_error')
                result['verification_error'] = (
                    f'{existing_error}; {error_msg}' if existing_error else error_msg
                )

        if self._write_journal:
            await self._write_journal.log_write_op(
                write_op_id=write_op_id,
                causation_id=causation_id,
                source=_source,
                operation='update_edge',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params=params,
                result_summary=result,
                success=True,
            )

        await self._emit_event(ReconciliationEvent(
            id=str(uuid_mod.uuid4()),
            type=EventType.memory_updated,
            source=EventSource.agent,
            project_id=project_id,
            timestamp=datetime.now(UTC),
            payload={'edge_uuid': edge_uuid, 'store': 'graphiti'},
        ))

        return result

    async def reassign_edge(
        self,
        edge_uuid: str,
        new_endpoint_uuid: str,
        which_end: str,
        project_id: str = 'main',
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
    ) -> dict:
        """Re-point one Graphiti edge's endpoint to a different Entity node, losslessly.

        Delegates to GraphitiBackend.reassign_edge(), which moves ONE end
        (``which_end='source'`` or ``'target'``) of the edge onto
        ``new_endpoint_uuid`` via an atomic uuid-preserving CREATE-new +
        DELETE-old, preserving the fact, fact_embedding, valid_at/invalid_at/
        expired_at, created_at, group_id, episodes, and the edge uuid, then
        refreshes the two affected endpoint summaries. Journals the operation
        and emits a memory_updated event (mirroring update_edge).

        Args:
            edge_uuid: UUID of the RELATES_TO edge to reassign.
            new_endpoint_uuid: UUID of the Entity node the endpoint moves onto.
            which_end: Which end to move — ``'source'`` or ``'target'``.
            project_id: Project scope (graph key + journal logging).
            agent_id: Which agent is calling (optional).
            session_id: Session context (optional).
            causation_id: Reconciliation causation ID (optional).
            _source: Source label for the journal entry.

        Returns:
            ``{'status': 'reassigned', 'store': 'graphiti', **audit}`` where
            audit is the backend's dict (uuid, which_end, old/new/unchanged
            endpoint uuids, moved, refreshed_nodes).
        """
        write_op_id = str(uuid_mod.uuid4())
        success = True
        error_msg = None
        result: dict = {}
        try:
            result = await self.graphiti.reassign_edge(
                edge_uuid, new_endpoint_uuid, which_end=which_end, group_id=project_id,
            )
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            if self._write_journal:
                try:
                    await self._write_journal.log_write_op(
                        write_op_id=write_op_id,
                        causation_id=causation_id,
                        source=_source,
                        operation='reassign_edge',
                        project_id=project_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        params={
                            'edge_uuid': edge_uuid,
                            'new_endpoint_uuid': new_endpoint_uuid,
                            'which_end': which_end,
                        },
                        result_summary=result if success else None,
                        success=success,
                        error=error_msg,
                    )
                except Exception as journal_exc:
                    logger.warning(
                        'reassign_edge: journal log_write_op failed: %s',
                        journal_exc,
                    )

        # Reached only on a SUCCESSFUL reassign (a backend failure re-raises
        # through the finally above, never landing here). Emit the
        # memory_updated event ONLY when the edge actually moved: a no-op
        # reassign (moved=False — the new endpoint already equals the current
        # one) changed nothing in the graph, so emitting would trigger spurious
        # downstream reconciliation for an edge that did not change. The journal
        # still records the (successful) no-op call as an accurate audit trail.
        if result.get('moved'):
            await self._emit_event(ReconciliationEvent(
                id=str(uuid_mod.uuid4()),
                type=EventType.memory_updated,
                source=EventSource.agent,
                project_id=project_id,
                timestamp=datetime.now(UTC),
                payload={'edge_uuid': edge_uuid, 'store': 'graphiti'},
            ))

        return {'status': 'reassigned', 'store': 'graphiti', **result}

    async def delete_episode(
        self,
        episode_id: str,
        project_id: str = 'main',
        cascade: bool = True,
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
    ) -> dict:
        """Delete a Graphiti episode."""
        write_op_id = str(uuid_mod.uuid4())

        await self._journaled_backend_call(
            write_op_id=write_op_id,
            causation_id=causation_id,
            backend='graphiti',
            operation='remove_episode',
            payload={'episode_id': episode_id},
            coro=self.graphiti.remove_episode(episode_id, group_id=project_id),
        )

        if self._write_journal:
            await self._write_journal.log_write_op(
                write_op_id=write_op_id,
                causation_id=causation_id,
                source=_source,
                operation='delete_episode',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params={'episode_id': episode_id, 'cascade': cascade},
                result_summary={'status': 'deleted'},
                success=True,
            )

        return {'status': 'deleted', 'episode_id': episode_id, 'cascade': cascade}

    async def redact_episode_content(
        self,
        episode_uuid: str,
        new_content: str,
        project_id: str = 'main',
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
    ) -> dict:
        """Replace one Graphiti episode's raw content in place, preserving its edges.

        The non-destructive counterpart to ``delete_episode`` for an episode
        whose text carries a leaked serialized tool-call fragment (task 3083).
        ``delete_episode(cascade=True)`` would destroy the entities and edges
        exclusively sourced from that episode — which for the known residual
        ``d12b0eb4`` includes demonstrably-valid collateral — so the leak is
        neutralised in the raw text and the extracted knowledge is left alone.

        See ``GraphitiBackend.redact_episode_content`` for the full rationale
        and for the loud refusals (blank replacement, or a replacement that
        still carries a leak, or an absent episode uuid).

        Returns:
            ``{status, store, uuid, old_content, new_content}``.
        """
        write_op_id = str(uuid_mod.uuid4())

        result_data = await self._journaled_backend_call(
            write_op_id=write_op_id,
            causation_id=causation_id,
            backend='graphiti',
            operation='redact_episode_content',
            payload={'episode_uuid': episode_uuid},
            coro=self.graphiti.redact_episode_content(
                episode_uuid, group_id=project_id, new_content=new_content,
            ),
        )

        if self._write_journal:
            await self._write_journal.log_write_op(
                write_op_id=write_op_id,
                causation_id=causation_id,
                source=_source,
                operation='redact_episode_content',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                # Truncated copies for the journal only — the full strings are
                # returned to the caller for audit.
                params={
                    'episode_uuid': episode_uuid,
                    'new_content': new_content[:200],
                },
                result_summary={'status': 'redacted'},
                success=True,
            )

        return {
            'status': 'redacted',
            'store': 'graphiti',
            **(result_data or {}),
        }

    async def refresh_entity_summary(
        self,
        entity_uuid: str | None = None,
        project_id: str = 'main',
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
        entity_name: str | None = None,
    ) -> dict:
        """Regenerate a Graphiti entity node's summary from its valid edges.

        Accepts either *entity_uuid* (canonical identifier) or *entity_name*
        (resolved via an exact name lookup).  When both are supplied, entity_uuid
        takes precedence.  Raises ValueError if neither is provided.

        Delegates to GraphitiBackend.refresh_entity_summary(), which queries
        remaining valid edges, deduplicates their facts, and writes back a
        clean summary. Logs the operation via write journal if available.

        Args:
            entity_uuid: UUID of the Entity node to refresh (optional when entity_name is given).
            entity_name: Exact entity name to resolve to a UUID (optional when entity_uuid is given).
            project_id: Project scope (for journal logging).
            agent_id: Which agent is calling (optional).
            session_id: Session context (optional).
            causation_id: Reconciliation causation ID (optional).
            _source: Source label for journal entry.

        Returns:
            Dict from backend: {uuid, name, old_summary, new_summary, edge_count}.

        Raises:
            ValueError: if neither entity_uuid nor entity_name is provided.
        """
        if entity_uuid is None and entity_name is None:
            raise ValueError('Either entity_uuid or entity_name must be provided')

        # Resolve entity_name → UUID when UUID is not directly supplied
        if entity_uuid is None:
            assert entity_name is not None  # guaranteed by the ValueError check above
            entity_uuid = await self.graphiti.resolve_entity_by_name(
                entity_name, group_id=project_id
            )

        write_op_id = str(uuid_mod.uuid4())
        success = True
        error_msg = None
        result: dict = {}
        journal_params: dict = {'entity_uuid': entity_uuid}
        if entity_name is not None:
            journal_params['entity_name'] = entity_name
        try:
            result = await self.graphiti.refresh_entity_summary(entity_uuid, group_id=project_id)
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            if self._write_journal:
                try:
                    await self._write_journal.log_write_op(
                        write_op_id=write_op_id,
                        causation_id=causation_id,
                        source=_source,
                        operation='refresh_entity_summary',
                        project_id=project_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        params=journal_params,
                        result_summary=result if success else None,
                        success=success,
                        error=error_msg,
                    )
                except Exception as journal_exc:
                    logger.warning(
                        'refresh_entity_summary: journal log_write_op failed: %s',
                        journal_exc,
                    )

        return result

    async def set_entity_summary(
        self,
        entity_uuid: str | None = None,
        summary: str | None = None,
        project_id: str = 'main',
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
        entity_name: str | None = None,
    ) -> dict:
        """Overwrite a Graphiti entity node's summary with explicit text, verbatim.

        Unlike refresh_entity_summary (which regenerates the summary from the
        entity's currently-valid edges), this writes *summary* exactly as given —
        it never reads or derives from edges. An empty string clears the summary
        entirely. This is the operator/reconciliation escape hatch for stale
        narrative text that edge-derived regeneration cannot remove.

        Accepts either *entity_uuid* (canonical identifier) or *entity_name*
        (resolved via an exact name lookup). When both are supplied, entity_uuid
        takes precedence. Raises ValueError if neither is provided, or if
        summary is not provided.

        Args:
            entity_uuid: UUID of the Entity node to overwrite (optional when entity_name is given).
            summary: Exact text to write as the new summary. May be '' to clear.
            entity_name: Exact entity name to resolve to a UUID (optional when entity_uuid is given).
            project_id: Project scope (for journal logging).
            agent_id: Which agent is calling (optional).
            session_id: Session context (optional).
            causation_id: Reconciliation causation ID (optional).
            _source: Source label for journal entry.

        Returns:
            Dict from backend: {uuid, name, old_summary, new_summary}.

        Raises:
            ValueError: if summary is None, or if neither entity_uuid nor entity_name is provided.
        """
        if summary is None:
            raise ValueError('summary must be provided')
        if entity_uuid is None and entity_name is None:
            raise ValueError('Either entity_uuid or entity_name must be provided')

        # Resolve entity_name → UUID when UUID is not directly supplied
        if entity_uuid is None:
            assert entity_name is not None  # guaranteed by the ValueError check above
            entity_uuid = await self.graphiti.resolve_entity_by_name(
                entity_name, group_id=project_id
            )

        write_op_id = str(uuid_mod.uuid4())
        success = True
        error_msg = None
        result: dict = {}
        journal_params: dict = {'entity_uuid': entity_uuid}
        if entity_name is not None:
            journal_params['entity_name'] = entity_name
        try:
            result = await self.graphiti.set_entity_summary(entity_uuid, summary, group_id=project_id)
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            if self._write_journal:
                try:
                    await self._write_journal.log_write_op(
                        write_op_id=write_op_id,
                        causation_id=causation_id,
                        source=_source,
                        operation='set_entity_summary',
                        project_id=project_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        params=journal_params,
                        result_summary=result if success else None,
                        success=success,
                        error=error_msg,
                    )
                except Exception as journal_exc:
                    logger.warning(
                        'set_entity_summary: journal log_write_op failed: %s',
                        journal_exc,
                    )

        return result

    async def rename_entity(
        self,
        new_name: str | None = None,
        entity_uuid: str | None = None,
        entity_name: str | None = None,
        project_id: str = 'main',
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
    ) -> dict:
        """Rename a Graphiti entity node to *new_name*, verbatim.

        Operator-facing escape hatch that corrects a mis-cased/mis-pluralized
        task-entity node name (e.g. 'task 132' -> 'Task 132') minted by
        graphiti-core's LLM entity extraction, for pre-existing bad nodes an
        episode may never re-touch (task 2110). Mirrors set_entity_summary's
        identifier-resolution, validation, and journal-logging contract.

        Accepts either *entity_uuid* (canonical identifier) or *entity_name*
        (resolved via an exact name lookup). When both are supplied, entity_uuid
        takes precedence. Raises ValueError if neither is provided, or if
        new_name is not provided or is empty.

        Args:
            new_name: Exact new name to write. Must be non-empty.
            entity_uuid: UUID of the Entity node to rename (optional when entity_name is given).
            entity_name: Exact entity name to resolve to a UUID (optional when entity_uuid is given).
            project_id: Project scope (for journal logging and group_id).
            agent_id: Which agent is calling (optional).
            session_id: Session context (optional).
            causation_id: Reconciliation causation ID (optional).
            _source: Source label for journal entry.

        Returns:
            Dict from backend: {uuid, old_name, new_name}.

        Raises:
            ValueError: if new_name is None/empty, or if neither entity_uuid nor
                        entity_name is provided.
        """
        if not new_name:
            raise ValueError('new_name must be provided')
        if entity_uuid is None and entity_name is None:
            raise ValueError('Either entity_uuid or entity_name must be provided')

        # Resolve entity_name → UUID when UUID is not directly supplied
        if entity_uuid is None:
            assert entity_name is not None  # guaranteed by the ValueError check above
            entity_uuid = await self.graphiti.resolve_entity_by_name(
                entity_name, group_id=project_id
            )

        write_op_id = str(uuid_mod.uuid4())
        success = True
        error_msg = None
        result: dict = {}
        journal_params: dict = {'entity_uuid': entity_uuid, 'new_name': new_name}
        if entity_name is not None:
            journal_params['entity_name'] = entity_name
        try:
            result = await self.graphiti.rename_entity_node(entity_uuid, new_name, group_id=project_id)
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            if self._write_journal:
                try:
                    await self._write_journal.log_write_op(
                        write_op_id=write_op_id,
                        causation_id=causation_id,
                        source=_source,
                        operation='rename_entity',
                        project_id=project_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        params=journal_params,
                        result_summary=result if success else None,
                        success=success,
                        error=error_msg,
                    )
                except Exception as journal_exc:
                    logger.warning(
                        'rename_entity: journal log_write_op failed: %s',
                        journal_exc,
                    )

        return result

    async def rebuild_entity_summaries(
        self,
        project_id: str = 'main',
        force: bool = False,
        dry_run: bool = False,
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
        entity_uuids: list[str] | None = None,
    ) -> dict:
        """Batch-rebuild Entity node summaries from their current valid edges.

        Orchestrates the rebuild pipeline:
        1. Target selection — entity_uuids (targeted, bypasses detection) takes
           precedence over detect_stale_with_edges (force=False) or
           list_entity_nodes (force=True).
        2. Fan-out — asyncio.Semaphore(20) + gather_collect (fused_memory.utils.async_utils)
           calling graphiti.rebuild_entity_from_edges for each target.
        3. Error accumulation — two-tier: gather_collect's Pass 1 (cancellation
           propagation) then per-entity dict accumulation (Pass 2).

        Logs the operation via write journal if available.

        Args:
            project_id: Project scope (determines FalkorDB graph).
            force: Rebuild every entity regardless of staleness.
            dry_run: Detect stale entities but do not write any summaries.
            agent_id: Which agent is calling (optional).
            session_id: Session context (optional).
            causation_id: Reconciliation causation ID (optional).
            _source: Source label for journal entry.
            entity_uuids: When provided (non-empty), force-regenerate exactly
                these entities from their currently-valid edges, bypassing
                staleness detection entirely (takes precedence over force).

        Returns:
            Dict with keys: total_entities, stale_entities, rebuilt, skipped,
            errors, details.
        """
        write_op_id = str(uuid_mod.uuid4())
        success = True
        error_msg = None
        result: dict = {}
        try:
            # --- Target selection ---
            targets: list[dict] = []
            all_edges: dict[str, list] = {}
            total_entities: int = 0
            not_found_details: list[dict] = []

            if entity_uuids is not None and len(entity_uuids) > 0:
                requested = list(dict.fromkeys(entity_uuids))  # dedupe, preserve order
                all_entities = await self.graphiti.list_entity_nodes(group_id=project_id)
                by_uuid = {e['uuid']: e for e in all_entities}
                targets = [
                    {'uuid': u, 'name': by_uuid[u]['name'], 'old_summary': by_uuid[u]['summary']}
                    for u in requested
                    if u in by_uuid
                ]
                not_found_details = [
                    {'uuid': u, 'name': None, 'status': 'not_found'}
                    for u in requested
                    if u not in by_uuid
                ]
                total_entities = len(targets)
                if not dry_run:
                    all_edges = await self.graphiti.get_all_valid_edges(group_id=project_id)
            elif entity_uuids is not None:
                # entity_uuids == [] — explicit zero-count no-op, no backend calls.
                pass
            elif force:
                all_entities = await self.graphiti.list_entity_nodes(group_id=project_id)
                targets = [
                    {'uuid': e['uuid'], 'name': e['name'], 'old_summary': e['summary']}
                    for e in all_entities
                ]
                total_entities = len(all_entities)
                if not dry_run:
                    all_edges = await self.graphiti.get_all_valid_edges(group_id=project_id)
            else:
                if dry_run:
                    stale, total_entities = await self.graphiti.detect_stale_dry_run(
                        group_id=project_id
                    )
                else:
                    detect_result = await self.graphiti.detect_stale_with_edges(
                        group_id=project_id
                    )
                    stale = detect_result.stale
                    all_edges = detect_result.all_edges
                    total_entities = detect_result.total_count
                targets = [
                    {'uuid': s['uuid'], 'name': s['name'], 'old_summary': s['summary']}
                    for s in stale
                ]

            stale_entities = len(targets)
            rebuilt = 0
            skipped = 0
            errors = 0
            details: list[dict] = []

            if dry_run:
                skipped = stale_entities
                for t in targets:
                    details.append({
                        'uuid': t['uuid'],
                        'name': t['name'],
                        'status': 'skipped_dry_run',
                    })
            else:
                sem = asyncio.Semaphore(20)

                async def _rebuild_one(t: dict) -> dict:
                    async with sem:
                        edges = all_edges.get(t['uuid'], [])
                        return await self.graphiti.rebuild_entity_from_edges(
                            t['uuid'], t['name'], edges,
                            group_id=project_id,
                            old_summary=t['old_summary'],
                        )

                # Pass 1: propagate CancelledError before per-entity accumulation.
                try:
                    gather_results = await gather_collect(_rebuild_one(t) for t in targets)
                except BaseException as e:
                    if not isinstance(e, Exception):
                        logger.warning(
                            'rebuild_entity_summaries: cancellation signal received '
                            'group=%s rebuilt_so_far=%d errors_so_far=%d; propagating',
                            project_id, rebuilt, errors,
                        )
                    raise

                # Pass 2: per-entity accumulation.
                for t, r in zip(targets, gather_results, strict=True):
                    if isinstance(r, Exception):
                        errors += 1
                        logger.error(
                            'rebuild_entity_summaries: failed to rebuild node=%s name=%r: %s',
                            t['uuid'], t['name'], r,
                        )
                        details.append({
                            'uuid': t['uuid'],
                            'name': t['name'],
                            'status': 'error',
                            'error': str(r),
                        })
                    else:
                        if not isinstance(r, dict):
                            raise TypeError(
                                f'rebuild_entity_summaries: rebuild_entity_from_edges returned '
                                f'unexpected type {type(r).__name__!r} for node={t["uuid"]} '
                                f'name={t["name"]!r}'
                            )
                        rebuilt += 1
                        details.append({
                            'uuid': t['uuid'],
                            'name': t['name'],
                            'status': 'rebuilt',
                            'old_summary': r.get('old_summary', ''),
                            'new_summary': r.get('new_summary', ''),
                            'edge_count': r.get('edge_count', 0),
                        })

            # Requested UUIDs absent from the graph are reported alongside the
            # rebuilt/skipped/error details in both dry_run and write modes,
            # without affecting rebuilt/skipped/errors counts.
            details.extend(not_found_details)

            logger.info(
                'rebuild_entity_summaries: group=%s total=%d stale=%d rebuilt=%d '
                'skipped=%d errors=%d dry_run=%s force=%s',
                project_id, total_entities, stale_entities, rebuilt, skipped, errors,
                dry_run, force,
            )
            result = {
                'total_entities': total_entities,
                'stale_entities': stale_entities,
                'rebuilt': rebuilt,
                'skipped': skipped,
                'errors': errors,
                'details': details,
            }
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            if self._write_journal:
                try:
                    await self._write_journal.log_write_op(
                        write_op_id=write_op_id,
                        causation_id=causation_id,
                        source=_source,
                        operation='rebuild_entity_summaries',
                        project_id=project_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        params={'force': force, 'dry_run': dry_run, 'entity_uuids': entity_uuids},
                        result_summary={
                            'total_entities': result.get('total_entities', 0),
                            'stale_entities': result.get('stale_entities', 0),
                            'rebuilt': result.get('rebuilt', 0),
                            'skipped': result.get('skipped', 0),
                            'errors': result.get('errors', 0),
                        } if success else None,
                        success=success,
                        error=error_msg,
                    )
                except Exception as journal_exc:
                    logger.warning(
                        'rebuild_entity_summaries: journal log_write_op failed: %s',
                        journal_exc,
                    )

        return result

    async def merge_entities(
        self,
        deprecated_uuid: str,
        surviving_uuid: str,
        project_id: str = 'main',
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
    ) -> dict:
        """Merge two Graphiti entity nodes by redirecting edges and deleting the deprecated.

        Delegates to GraphitiBackend.merge_entities(), which validates both nodes,
        redirects all edges from the deprecated node to the surviving node, deletes
        the deprecated node, and refreshes the surviving node's summary.
        Logs the operation via write journal if available.

        Args:
            deprecated_uuid: UUID of the entity node to be deleted.
            surviving_uuid: UUID of the entity node that absorbs the edges.
            project_id: Project scope (for journal logging).
            agent_id: Which agent is calling (optional).
            session_id: Session context (optional).
            causation_id: Reconciliation causation ID (optional).
            _source: Source label for journal entry.

        Returns:
            Audit dict from backend: {surviving_uuid, surviving_name, deprecated_uuid,
            deprecated_name, edges_redirected, surviving_summary}.
        """
        write_op_id = str(uuid_mod.uuid4())
        success = True
        error_msg = None
        result: dict = {}
        try:
            result = await self.graphiti.merge_entities(deprecated_uuid, surviving_uuid, group_id=project_id)
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            if self._write_journal:
                try:
                    await self._write_journal.log_write_op(
                        write_op_id=write_op_id,
                        causation_id=causation_id,
                        source=_source,
                        operation='merge_entities',
                        project_id=project_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        params={
                            'deprecated_uuid': deprecated_uuid,
                            'surviving_uuid': surviving_uuid,
                        },
                        result_summary=result if success else None,
                        success=success,
                        error=error_msg,
                    )
                except Exception as journal_exc:
                    logger.warning(
                        'merge_entities: journal log_write_op failed: %s',
                        journal_exc,
                    )

        # ζ (task 2899) — best-effort standing-decision invalidation. Reached
        # only on a SUCCESSFUL merge (a backend failure re-raises through the
        # finally above, never landing here). The post-merge entity is a new
        # subject, so any ACTIVE decision on either merged uuid no longer
        # applies and is flipped to expired/merge. merge_entities is the
        # authoritative operation; this secondary consequence must NEVER break
        # it — a hook failure is logged and swallowed (the row simply stays
        # ACTIVE, caught later by TTL or the growth sweep).
        try:
            await self._expire_standing_decisions_for_merge(
                project_id, deprecated_uuid, surviving_uuid
            )
        except Exception as hook_exc:
            logger.warning(
                'merge_entities: standing-decision invalidation hook failed for '
                'deprecated=%s surviving=%s project_id=%s (merge already '
                'committed; row left ACTIVE for TTL/growth sweep): %s',
                deprecated_uuid,
                surviving_uuid,
                project_id,
                hook_exc,
            )

        return result

    async def _expire_standing_decisions_for_merge(
        self,
        project_id: str,
        deprecated_uuid: str,
        surviving_uuid: str,
    ) -> int:
        """Expire ACTIVE ``entity_standing_decision`` rows on EITHER merged uuid.

        ζ's ``merge_entities`` invalidation hook (task 2899). A merge fuses two
        entity nodes into one new subject, so a prior "this class of complaint
        about entity X was dismissed" decision on either the deprecated or the
        surviving uuid no longer applies — re-deriving the complaint once
        against the merged entity is correct (PRD §Staleness; also the first fix
        for the dangling-uuid hazard, research fact 9).

        Enumerates ACTIVE standing rows via ``list_entity_standing_decisions``
        (state=active) and filters to those whose ``entity_uuid`` is one of the
        two merged uuids, flipping each through the single-source
        :func:`~fused_memory.reconciliation.standing_decision_writer.expire_entity_standing_decision`
        primitive with ``reason='merge'``. List+filter is used (NOT
        ``get_active_entity_standing_decision``, which raises on >1 active
        grounds for one entity) so this stays robust to the future
        multiple-active-grounds-per-entity case.

        Returns the number of rows expired. An unwired ledger
        (``self.recon_ledger is None``) is a no-op returning ``0`` (did-not-run,
        never a spurious miss) — matching the ledger-None-returns-0 convention
        of the ζ growth sweep and the writer's guard.
        """
        ledger = getattr(self, 'recon_ledger', None)
        if ledger is None:
            return 0
        rows = await ledger.list_entity_standing_decisions(project_id, state=STATE_ACTIVE)
        merged_uuids = {deprecated_uuid, surviving_uuid}
        expired = 0
        for row in rows:
            if row.entity_uuid not in merged_uuids:
                continue
            # Per-row fail-safe: a single malformed row (e.g. a payload missing
            # edge_count_at_decision, which the flip helper reads directly) must
            # NOT block the sibling merged uuid's flip. Leave the bad row ACTIVE
            # (re-caught later by TTL or the growth sweep) and continue — the
            # same per-row guard _sweep_entity_standing_decision_growth uses.
            try:
                await expire_entity_standing_decision(
                    ledger, row, reason=EXPIRY_REASON_MERGE
                )
                expired += 1
            except Exception:
                logger.warning(
                    '_expire_standing_decisions_for_merge: flip to expired/merge '
                    'failed for entity_uuid=%s project_id=%s (left active for '
                    'TTL/growth sweep)',
                    row.entity_uuid,
                    project_id,
                    exc_info=True,
                )
        return expired

    async def delete_entity(
        self,
        entity_uuid: str,
        project_id: str = 'main',
        force: bool = False,
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
    ) -> dict:
        """Delete a Graphiti entity node by UUID, refreshing connected neighbours.

        Delegates to GraphitiBackend.delete_entity(), which validates the node
        exists, guards against accidental deletion of nodes with active edges
        (unless force=True), collects neighbours before deletion, performs
        DETACH DELETE, then refreshes each neighbour's summary.
        Logs the operation via write journal if available.

        Args:
            entity_uuid: UUID of the entity node to delete.
            project_id: Project scope (for Graphiti group_id and journal logging).
            force: When True, bypass the active-edges guard.
            agent_id: Which agent is calling (optional).
            session_id: Session context (optional).
            causation_id: Reconciliation causation ID (optional).
            _source: Source label for journal entry.

        Returns:
            Audit dict from backend: {deleted_uuid, deleted_name, active_edge_count,
            forced, connected_refreshed}.

        Raises:
            NodeNotFoundError: if the entity does not exist.
            ActiveEdgesError: if the node has valid active edges and force=False.
        """
        write_op_id = str(uuid_mod.uuid4())
        success = True
        error_msg = None
        result: dict = {}
        try:
            result = await self.graphiti.delete_entity(entity_uuid, group_id=project_id, force=force)
        except Exception as e:
            success = False
            error_msg = str(e)
            raise
        finally:
            if self._write_journal:
                try:
                    await self._write_journal.log_write_op(
                        write_op_id=write_op_id,
                        causation_id=causation_id,
                        source=_source,
                        operation='delete_entity',
                        project_id=project_id,
                        agent_id=agent_id,
                        session_id=session_id,
                        params={
                            'entity_uuid': entity_uuid,
                            'force': force,
                        },
                        result_summary=result if success else None,
                        success=success,
                        error=error_msg,
                    )
                except Exception as journal_exc:
                    logger.warning(
                        'delete_entity: journal log_write_op failed: %s',
                        journal_exc,
                    )

        return result

    # ------------------------------------------------------------------
    # Management
    # ------------------------------------------------------------------

    async def get_status(self, project_id: str | None = None) -> dict:
        """Health check and per-project statistics for both backends."""
        status: dict[str, Any] = {}

        # Graphiti connectivity + project discovery
        graphiti_counts: dict[str, int] = {}
        try:
            graphs = await self.graphiti.list_graphs()
            for graph_name in graphs:
                try:
                    graphiti_counts[graph_name] = await self.graphiti.node_count(graph_name)
                except Exception:
                    graphiti_counts[graph_name] = -1
            status['graphiti'] = {'connected': True}
        except Exception as e:
            status['graphiti'] = {'connected': False, 'error': str(e)}

        # Mem0 connectivity + project discovery
        mem0_counts: dict[str, int] = {}
        try:
            mem0_projects = await self.mem0.list_projects()
            for pid, _collection_name in mem0_projects:
                try:
                    scope = Scope(project_id=pid)
                    mem0_counts[pid] = await self.mem0.count(scope)
                except Exception:
                    mem0_counts[pid] = -1
            status['mem0'] = {'connected': True}
        except Exception as e:
            status['mem0'] = {'connected': False, 'error': str(e)}

        # Merge into per-project dict
        all_project_ids = sorted(set(graphiti_counts) | set(mem0_counts))
        if project_id:
            all_project_ids = [p for p in all_project_ids if p == project_id]
        projects: dict[str, dict] = {}
        for pid in all_project_ids:
            projects[pid] = {
                'graphiti_nodes': graphiti_counts.get(pid, 0),
                'mem0_memories': mem0_counts.get(pid, 0),
            }
        status['projects'] = projects

        # Queue stats — scoped to project_id when given so dead-letter counts
        # reflect only this project's rows, not the global write_queue table.
        if self.durable_queue:
            try:
                status['queue'] = await self.durable_queue.get_stats(group_id=project_id)
            except Exception as e:
                status['queue'] = {'error': str(e)}

        # Server uptime fields (informational; dashboard reads uptime_seconds as primary)
        status['started_at'] = self._started_at.isoformat()
        status['uptime_seconds'] = int(time.monotonic() - self._start_monotonic)

        return status

    def get_consolidation_tools(self) -> dict:
        """Return the restricted tool set for the consolidation agent."""
        return {
            'search': self.search,
            'add_memory': self.add_memory,
            'delete_memory': self.delete_memory,
            'get_episodes': self.get_episodes,
            'get_status': self.get_status,
        }
