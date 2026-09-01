"""Core orchestration layer — owns backends, classifier, router, durable queue."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
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
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar, cast

from graphiti_core.nodes import EpisodeType

from fused_memory.backends.graphiti_client import ActiveEdgesError, GraphitiBackend
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
from fused_memory.middleware.referent_repair_storm_escalator import (
    emit_referent_repair_storm_escalation,
)
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
from fused_memory.services.topic_anchor import (
    _ANCHOR_SCROLL_LIMIT,
    _MAX_ANCHOR_TOPICS,
    extract_anchor_topics,
    resolve_topic_anchor_enabled,
    select_canonical_payload,
)
from fused_memory.utils.async_utils import gather_collect, gather_or_raise
from fused_memory.utils.canonical_labels import Referent, parse_node_name, scan_content
from fused_memory.utils.referent_resolution import (
    REFERENT_SOURCES,
    ReferentResolution,
    ReferentSet,
    local_referent,
    resolve_referents,
)
from fused_memory.utils.task_naming import canonicalize_task_node_name
from fused_memory.utils.validation import _safe_repr, require_full_uuid

if TYPE_CHECKING:
    from fused_memory.backends.task_backend_protocol import TaskBackendProtocol
    from fused_memory.reconciliation.event_buffer import EventBuffer
    from fused_memory.reconciliation.recon_ledger import ReconLedgerStore
    from fused_memory.services.planned_episode_registry import PlannedEpisodeRegistry
    from fused_memory.services.write_journal import WriteJournal

logger = logging.getLogger(__name__)

#: Return type of one ``_reconcile_episode_identity`` sub-pass. Generic so ONE
#: best-effort guard covers both the six int-returning sweeps and task 3671's
#: ``ReferentStats``-returning verification pass — see ``_run_pass``.
_T = TypeVar('_T')

# Per-sub-close timeout used by MemoryService._safe_close (task 2701). A healthy
# FalkorDB/Qdrant localhost driver teardown completes in well under 1s; 3s gives
# headroom under load while capping a hung network-driver close so no single
# backend can consume the whole shutdown budget or starve the durable-flush
# SQLite closes (_write_journal/_event_buffer) that run after it. The paired
# outer step budget lives in server/main.py as _MEMORY_CLOSE_STEP_TIMEOUT and
# must dominate 6 * _SUBCLOSE_TIMEOUT (guarded by TestShutdownBudgetArithmetic).
_SUBCLOSE_TIMEOUT = 3.0

# Reciprocal Rank Fusion constant for the cross-store merge in
# MemoryService.search (task 3658, PRD D4 — deliberately a module constant, not
# config: it is part of the documented read contract, not an operator knob).
#
# The consequence worth internalizing: because K dominates the ranks in play
# (limit is typically <= 20), the fused value is an ORDINAL, never a similarity.
# Every possible score lives in the narrow band 1/(K+1) .. 1/(K+limit) — for
# K=60, roughly 0.0164 down to 0.0125. Do not read a fused score as "how
# similar"; per-store truth lives in metadata['store_score'].
RRF_K = 60


def _rrf_score(rank: int) -> float:
    """Reciprocal Rank Fusion score for a 1-based per-store rank (task 3658).

    The value is ORDINAL, never a similarity: rank-1 scores 1/(RRF_K + 1) =
    1/61 ~ 0.0164 and rank-2 scores 1/62 ~ 0.0161, regardless of how good
    either result actually is.  Consumers must not compare it across API
    versions or treat it as a distance; the honest per-store signal is
    ``metadata['store_score']`` (the Mem0 cosine; ``None`` for Graphiti, which
    exposes no scores at all — the very reason RRF was chosen over score
    calibration).

    The PRD writes fusion as ``Σ over stores of 1/(K + rank_store(r))``, but
    that sum degenerates to this single term for every result here: Graphiti
    results are keyed by edge uuid and Mem0 results by memory id, and there is
    no cross-store dedup anywhere in the pipeline, so no result is ever
    contributed by more than one store.  A real multi-term accumulator would be
    dead code no input can exercise.

    That degeneracy is exactly what fixes the Mem0 shut-out: with one term per
    result, the merged order becomes a rank INTERLEAVE (graphiti-1, mem0-1,
    graphiti-2, mem0-2, ...) rather than one store's results wholesale
    preceding the other's.
    """
    return 1.0 / (RRF_K + rank)


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


def _mem0_content(payload: dict) -> str:
    """The human-readable text of a RAW Qdrant payload, or ``''``.

    First non-empty string among :data:`_MEM0_CONTENT_KEYS`.  Shared by every
    caller that turns a raw payload into text — ``get_memory_by_id`` and the
    topic-anchored pin in :meth:`MemoryService.search` — so the fallback ORDER
    lives in exactly one place.  A second inline copy of this loop would be a
    place for the two paths to disagree about which key holds the body (INV-5).

    Note this is the RAW-PAYLOAD path, distinct from the Mem0 *search item*
    path in ``_search_mem0``: a search item and a scroll payload do not put the
    text under the same key, which is exactly why guessing one key is wrong.
    """
    for key in _MEM0_CONTENT_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ''


def _mem0_category(meta: dict) -> MemoryCategory | None:
    """The MemoryCategory a raw Mem0 payload declares, or ``None``.

    ``None`` for an absent key AND for an unrecognised value: a payload's
    ``category`` is a plain string with no read-time schema enforcement, so an
    unknown value degrades to "no category" rather than raising mid-search.
    Shared by ``_search_mem0`` and the topic-anchored pin so both paths agree
    on what an unrecognised category means (INV-5).
    """
    if 'category' not in meta:
        return None
    with contextlib.suppress(ValueError):
        return MemoryCategory(meta['category'])
    return None


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
    baseline: dict | None = None,
) -> None:
    """Validate the Mem0 metadata vocabulary at the write boundary, in place.

    Task 3195 (leaf β of ``docs/prds/memory-metadata-vocabulary.md``).  The
    third of this module's shared in-place metadata helpers, alongside
    :func:`_normalize_task_id_metadata` and
    :func:`_apply_cycle_summary_metadata_tagging`, and shared by ALL THREE
    Mem0 write paths — ``add_memory``, ``add_system_record`` and (task 3523)
    ``update_memory`` — for the same reason the task-2222 amendment made the
    cycle-summary tagging shared: PRD D8/§2 pin enforcement at the SERVICE
    seam precisely because ``add_system_record`` is a second write path that
    a tools-layer validator would leak past.  Call sites with drifting
    behaviour would reopen that hole.

    ``update_memory`` is the third such path and reproduced exactly that
    leak until task 3523: a patch could set any ``topic`` spelling or a
    second ``canonical`` for a taken topic without ever reaching this
    function.  D8/§2 enumerated only the two add paths, and that silence
    read as coverage.  If a FOURTH write path appears, it belongs here too —
    the enumeration above is the checkable list.

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

    ``baseline`` — JUDGE THE DELTA, NEVER THE CORPUS (task 3523).  The two
    add paths CREATE a record, so there is no pre-image: they pass no
    ``baseline`` and every obligation above applies to the whole dict,
    bit-identically to before this parameter existed.  ``update_memory``
    AMENDS one, and passes the record's pre-image custom subset.  When it is
    supplied, obligations 2 through 5 are reduced to what this write actually
    CHANGED — a violation the record already carried (on a key this write
    left alone) is neither re-censused nor re-rejected, the ``parent_id``
    liveness probe fires only for a parent this write ASSERTS, and the
    uniqueness probe fires only for a ``canonical``/``topic`` claim the
    record does not already hold.

    That reduction has THREE implementation sites, not one, because the
    rules reach live state differently.  Obligations 3 and 4 are reduced by
    the ``(key, code)`` subtraction below; obligation 5 by
    :func:`_check_canonical_uniqueness`'s guard 3; obligation 2 by its own
    claim-is-NEW gate on the liveness block, because the subtraction
    structurally cannot see liveness codes (the pure validator cannot
    produce them) and would let both survive every patch.  A fourth rule
    that reads live state needs its own gate too — the subtraction will not
    cover it.

    That reduction is not a leniency knob; it is what keeps ``enforce``
    meaning "reject WRITES" instead of quietly becoming "re-validate the
    corpus".  Both PRD §9 leaf ε's 2026-08-04 amendment and
    :func:`_check_canonical_uniqueness` state that model in prose, and task
    3626's decision to flip ``enforce`` on is measured against it (~20 → ~19
    false rejections/week).  Validating the full effective dict on every
    patch would silently invalidate that measurement: legacy records are
    known fatal-invalid today (``scripts/sweep_toolcall_xml_leak.py``
    enumerates the classes — unknown ``kind``, malformed ``supersedes``,
    non-bool ``canonical``), so re-tagging exactly those records would start
    failing the moment the flip landed.  ``scripts/retro_stamp_topics.py``
    is the in-repo bulk re-tagger that would hit it: it stamps ``topic``
    onto legacy records through THIS path, one metadata-only patch each.
    (That sweep is cited for the enumeration and for the re-tagging
    exposure, NOT as a caller of this arm — it repairs by delete + re-add
    through ``add_memory`` and pre-checks with ``validate_memory_metadata``
    itself, so it never reaches ``update_memory``.)  Costs one extra PURE
    synchronous
    ``validate_memory_metadata`` call on a shallow copy, and zero I/O; see
    the block comment at the subtraction for the two-halves forgiveness rule
    and its ordering constraints.

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

    The lookup fires only when ``parent_id`` is PRESENT, already
    shape-valid, *and* (task 3523, when a ``baseline`` is supplied) actually
    ASSERTED by this write: the common write path (no ``parent_id`` at all —
    leaf α measured zero live records carrying one) pays no round-trip, an
    id no store could resolve is never spent on, and a patch that leaves an
    existing ``parent_id`` untouched is answerable for neither the lookup
    nor its verdict.  Liveness ADDS a violation
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
    #
    # ALSO gated on the parent_id claim being NEW (task 3523), mirroring
    # `_check_canonical_uniqueness`'s guard 3 in shape and for the same
    # reason. Liveness is the ONE rule the (key, code) subtraction below
    # structurally cannot delta-scope: the baseline set is built by the PURE
    # `validate_memory_metadata`, which cannot produce `dead_parent_id` or
    # `parent_id_liveness_unavailable`, so those codes would survive the
    # subtraction on EVERY patch — including one that never mentions
    # parent_id. A record whose parent was later deleted would then become
    # permanently un-patchable under `enforce` (and census a `dead_parent_id`
    # line per patch under the shipped warn mode), which is exactly the
    # "`enforce` re-validates the corpus" failure the delta rule exists to
    # prevent. So the scoping happens HERE, at the source, instead.
    #
    # Fail-CLOSED is preserved for every write that ASSERTS a parent: a new
    # or CHANGED parent_id still pays the round-trip and still rejects under
    # `enforce`. Only an untouched pre-existing one is forgiven — the same
    # value-unchanged half the shape rules use below. Compared raw rather
    # than against the normalized copy because `validate_memory_metadata`'s
    # only in-place mutation is `supersedes`; `parent_id` is never rewritten.
    if (
        'parent_id' in meta
        and (baseline is None or baseline.get('parent_id') != meta['parent_id'])
        and not any(v.key == 'parent_id' for v in violations)
    ):
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

    # DELTA SCOPING (task 3523) — judge what this write CHANGED, never the
    # record at rest.  Supplied only by `update_memory`, which is amending an
    # existing record; the two add paths create one and so have no pre-image,
    # pass no baseline, and are bit-identical to before.
    #
    # Reducing the set here, ONCE, is what makes the rule uniform: all three
    # arms below — census, storm detector, enforce-reject — then operate on
    # NEW violations only.  Re-censusing a pre-existing violation on every
    # patch would inflate the census stream the task-3626 flip is measured
    # from and trip false unknown-key storms off a long tail that was already
    # counted; re-rejecting one would quietly restate `enforce` from "rejects
    # WRITES" to "re-validates the corpus", which is the model both
    # `_check_canonical_uniqueness`'s docstring and PRD §9 leaf ε's
    # 2026-08-04 amendment state and measure against.
    #
    # AFTER the liveness block on purpose: that block gates its round-trip on
    # `parent_id` carrying no shape violation, so subtracting first would let
    # a pre-existing `invalid_parent_id_shape` spend a lookup on an id no
    # store could resolve and then census `dead_parent_id` for it — blaming
    # the wrong rule, which leaf δ explicitly forbids.
    #
    # THIS SUBTRACTION DOES NOT DELTA-SCOPE LIVENESS, and cannot: `already`
    # comes from the PURE `validate_memory_metadata`, which structurally
    # cannot emit `dead_parent_id` / `parent_id_liveness_unavailable`, so
    # `(v.key, v.code) not in already` is unconditionally True for both and
    # they would survive every patch. Liveness is delta-scoped at its SOURCE
    # instead — see the claim-is-NEW gate on the block above. Do not "unify"
    # the two by deleting that gate and relying on this list comprehension:
    # it would silently reinstate corpus re-validation for exactly one rule.
    #
    # Forgiven only when BOTH halves hold: the baseline already carried this
    # (key, code) AND the write left that key's value alone.  (key, code)
    # alone is not enough — swapping one bad slug for a DIFFERENT bad slug
    # repeats the pair while being entirely this write's doing, and would
    # earn a free pass.  "Judge what the write CHANGED" is about the KEY's
    # value, not about which rule happens to fire.
    #
    # Compared against the NORMALIZED baseline copy, not the raw pre-image:
    # `validate_memory_metadata` mutates in place, so a record whose stored
    # `supersedes` is a legacy scalar would otherwise read as "changed" on
    # every patch that never mentioned it.
    #
    # One extra PURE synchronous call on a shallow COPY, and zero I/O.
    if baseline is not None:
        _unset = object()
        before = dict(baseline)
        already = {
            (v.key, v.code)
            for v in validate_memory_metadata(
                before, enforce_kind_registry=config.enforce_kind_registry
            )
        }
        violations = [
            v for v in violations
            if (v.key, v.code) not in already
            or meta.get(v.key, _unset) != before.get(v.key, _unset)
        ]

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
        baseline=baseline,
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
    baseline: dict | None = None,
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
    3. *baseline* supplied and the effective ``(canonical, topic)`` claim
       EQUALS the baseline's → return (task 3523).  This write asserts no
       claim the record does not already hold, so there is nothing new to
       check, and ε's contracted zero-extra-round-trips property must hold
       for a no-op too.  Only ``update_memory`` supplies a baseline; the two
       add paths pass none, keep this guard inert, and so keep today's exact
       guard order and round-trip count.

       THIS IS WHAT MAKES SELF-INCUMBENCY STRUCTURALLY IMPOSSIBLE — do not
       "fix" it later by adding an ``exclude_id``.  The probe now runs only
       when the record is ACQUIRING a claim, and a record that does not yet
       hold the claim in the store cannot appear in the store-side count.
       An ``exclude_id`` would instead cost an extra round-trip on every
       canonical patch, need ``limit=2`` to filter self out of the scroll,
       add a parameter to both injected collaborators, and risk
       over-excluding a genuine duplicate — while STILL needing this guard
       to avoid probing on a no-op.

       Compared as a PAIR, not on ``canonical`` alone: a canonical record
       re-homed from topic T to topic U changes no ``canonical`` value but
       is acquiring a claim at U, where it genuinely is not the incumbent.
    4. count == 0 → return.  The happy path pays exactly one exact Qdrant
       count and never scrolls.
    5. otherwise resolve the incumbent's id and reject.

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

    STILL TRUE AFTER TASK 3523, and deliberately so.  Wiring this seam into
    ``update_memory`` added a third write path, but its enforcement is
    DELTA-scoped: a patch is judged only on the violations and claims it
    introduces, so amending a record never re-validates what that record
    already carried.  Had it been full-dict instead, every patch of a legacy
    record would have become a rejection under ``enforce`` and the ~19/week
    figure above — the number 3626 flips against — would have silently
    stopped describing the system.  Guard 3 is the uniqueness half of that
    rule; see :func:`_apply_memory_metadata_validation`'s ``baseline`` for
    the shape half.

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

    # Guard 3 (task 3523) — no NEW claim, no probe.  See the numbered list in
    # the docstring: this is what makes the record's own presence in the
    # store irrelevant, so no `exclude_id` is needed anywhere below.
    if baseline is not None and (
        (baseline.get('canonical'), baseline.get('topic')) == (True, topic)
    ):
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


def _encode_referents(resolution: ReferentResolution) -> dict[str, Any]:
    """Encode a resolved referent set for the durable-queue payload.

    THE WIRE CONTRACT (task 3670, PRD leaf epsilon).  One additional key,
    ``'referents'``, on the EXISTING ``add_episode`` / ``add_memory_graphiti``
    payloads::

        {'source': <one of REFERENT_SOURCES>,
         'refs': [{'kind': ..., 'project_id': ..., 'number': ...}, ...]}

    Deliberately NO ``payload_version``, no unknown-operation guard and no
    migration (PRD "Queue compatibility is free here").  An OLD consumer
    draining a new row ignores exactly one unknown key; a NEW consumer draining
    an old row finds the key absent and treats it as "no referents" — which is
    today's behaviour exactly.  A new queue OPERATION would have needed all
    three; one additional key on an existing payload needs none of them.

    Nesting everything under a single key (rather than flat ``referent_source``
    + ``referent_refs``) keeps the back-compat story to one presence test and
    gives :func:`_decode_referents` exactly one thing to validate.

    Emits PLAIN JSON SCALARS ONLY, never the frozen :class:`Referent` dataclass
    itself: the queue persists payloads as JSON TEXT in SQLite, so a
    non-serializable value here would surface only in production.

    AMBIGUITY IS DELIBERATELY NOT THREADED — READ THIS BEFORE WRITING ZETA.
    ``ReferentResolution.ambiguous`` (and ``.conflicts``) are dropped here; only
    ``.source`` and ``.referents`` ride the wire.  That matters because gamma
    excludes ambiguous referents from ``.referents`` on purpose ("recorded, not
    guessed"), so a consumer that reads ONLY ``refs`` sees an ambiguous endpoint
    as a plain non-member of the set — indistinguishable from a genuine
    conflation.  Leaf zeta must therefore NOT treat "endpoint not in the decoded
    set" as sufficient grounds for leaf eta to repoint the edge, or an ambiguous
    reference gets destructively repaired instead of recorded and left alone
    (PRD boundary-test table: "Ambiguous scan | ref routed to ``.ambiguous``;
    treated as undeclared; recorded, not guessed").

    Zeta re-derives it rather than reading it off the wire.  ``.ambiguous`` is
    ``scan_content(content, group_id=group_id).ambiguous`` verbatim on EVERY
    precedence path — a pure function of ``(content, group_id)``, independent of
    ``declared``/``metadata`` (referent_resolution.py: "`.ambiguous` is the
    scan's verbatim answer on every path").  ``_execute_graphiti_write`` holds
    both ``payload['content']`` and ``payload['group_id']``, so zeta can recover
    the producer's exact ambiguity set from data already on the payload.

    That re-derivation is a SECOND SCAN SITE, which gamma's own comment flags as
    the INV-5 lockstep duplication canonical_labels exists to prevent — so
    carrying ``'ambiguous'`` as a third key is the better long-term shape and is
    filed as follow-up work.  It is not done here because this leaf's frozen
    contract is the two-key blob and widening it changes this function's return
    arity and the wire shape every test in
    tests/test_referent_queue_threading.py pins.  Extending it later is
    additive and needs no migration, exactly as adding ``'referents'`` did.
    """
    return {
        'source': resolution.source,
        'refs': [
            {'kind': r.kind, 'project_id': r.project_id, 'number': r.number}
            for r in resolution.referents
        ],
    }


def _decode_referents(payload: dict[str, Any]) -> tuple[ReferentSet, str]:
    """Pop and decode the ``'referents'`` blob :func:`_encode_referents` wrote.

    Returns ``(referents, source)``.  An ABSENT key decodes to ``((), 'none')``
    — an old-format queue row executes byte-identically to today.

    POPS the key, matching how ``_execute_graphiti_write`` already treats
    ``temporal_context`` / ``unverified_claim`` / ``reference_time``.  Safe
    because ``DurableWriteQueue._process_item`` hands the executor one
    ``parsed_payload()`` and the registered callback a SECOND, FRESH one,
    precisely so the executor can pop what the callbacks read back.

    Each entry is rebuilt through the ``Referent(...)`` constructor rather than
    kept as a bare dict, so the frozen type's kind-registry validation runs on
    untrusted wire data too — which is also what makes an unregistered ``kind``
    on the wire fall into the degradation path below instead of minting a bogus
    referent.  That constructor validates ``kind`` ONLY, so ``number`` and
    ``project_id`` are type-checked here before it runs; see the inline comment
    in the decode loop for the three distinct ways an unchecked field escapes.

    DEGRADATION IS ALL-OR-NOTHING.  Any unreadable element — a non-dict blob, a
    ``source`` outside :data:`REFERENT_SOURCES`, a non-list ``refs``, or a
    SINGLE malformed entry — degrades the WHOLE blob to ``((), 'none')``, never
    a partial set.  A partial set is worse than no set for the consumer this
    exists to serve: leaf zeta's set-membership check reads "endpoint not in
    the referent set" as a conflation and leaf eta repairs it by repointing the
    edge, so a referent silently dropped by a lenient decoder would manufacture
    a false conflation and drive destructive edge surgery onto the wrong node.
    Referents are therefore accumulated into a local list and only frozen into
    a tuple on FULL success, so a partial set cannot escape by construction.

    DEGRADES RATHER THAN RAISES, deliberately.  This runs inside the queue
    executor: raising would route the item to ``_handle_failure`` and
    eventually dead-letter it, LOSING the memory over a telemetry field.
    Degrading is safe here only BECAUSE the anomaly lands in the 'none' bucket
    that ``_execute_graphiti_write``'s counter makes loud — the INV-4 escape,
    not a silent fallthrough.  The ABSENT key is the one case that does NOT
    warn: it is the load-bearing back-compat path (every row written before
    task 3670), not an anomaly, and warning on it would drown the log during a
    drain of a pre-feature queue.  It is still COUNTED, in the same bucket.

    Loud-and-degrade mirrors the invalid-``reference_time`` arm already in
    ``_execute_graphiti_write``, so this file has one idiom, not two.
    """
    blob = payload.pop('referents', None)
    if blob is None:
        return (), 'none'

    def _degrade(reason: str) -> tuple[ReferentSet, str]:
        # _safe_repr, not a bare %r: the blob is arbitrary decoded JSON from a
        # queue row and this warning fires on EVERY retry attempt of that item,
        # so an oversized corrupt value would otherwise dump its full repr into
        # the log repeatedly. Matches how the sibling module this codec is
        # written against (utils/referent_resolution.py) renders every one of
        # its untrusted-value rejection messages.
        logger.warning(
            "Unreadable 'referents' payload key (%s); treating the write as "
            'having no referents. Blob: %s',
            reason, _safe_repr(blob),
        )
        return (), 'none'

    if not isinstance(blob, dict):
        return _degrade(f'expected a dict, got {type(blob).__name__}')
    source = blob.get('source')
    if source not in REFERENT_SOURCES:
        return _degrade(f'source {source!r} is not one of {list(REFERENT_SOURCES)}')
    refs = blob.get('refs')
    if not isinstance(refs, list):
        return _degrade(f"'refs' must be a list, got {type(refs).__name__}")

    decoded: list[Referent] = []
    for entry in refs:
        if not isinstance(entry, dict):
            return _degrade(f'entry {_safe_repr(entry)} is not a dict')
        # `Referent.__post_init__` validates `kind` against the kind registry
        # but NOT `number`/`project_id` — those two fields accept any object at
        # all, so the constructor alone does NOT harden this boundary. Each
        # unchecked type is a distinct downstream failure:
        #   - a non-str `number` (e.g. 3127) mints a Referent that compares
        #     UNEQUAL to its string twin, so leaf zeta's set-membership check
        #     would read a legitimate endpoint as a conflation and leaf eta
        #     would repoint the edge destructively — the same false-conflation
        #     failure the all-or-nothing rule above exists to prevent, arriving
        #     through a mistyped field instead of a dropped one;
        #   - a None `number`/`project_id` mints a referent whose `node_name`
        #     is the literal string 'Task None';
        #   - an UNHASHABLE `number` (a list) mints a Referent that raises
        #     TypeError the moment a consumer puts it in a set — a raise inside
        #     the queue executor, i.e. exactly the dead-letter-and-lose-the-
        #     memory outcome degrade-rather-than-raise exists to prevent.
        # `_encode_referents` only ever emits strings, so this is reachable
        # today only from a corrupt or hand-edited SQLite row — but this
        # function is the wire-hardening boundary, so it hardens the fields
        # that matter rather than assuming its own encoder wrote the row.
        number = entry.get('number')
        project_id = entry.get('project_id', '')
        if not isinstance(number, str) or not isinstance(project_id, str):
            return _degrade(
                f'entry {_safe_repr(entry)} has a non-string number/project_id'
            )
        try:
            decoded.append(Referent(
                kind=entry.get('kind', 'task'),
                project_id=project_id,
                number=number,
            ))
        except (KeyError, TypeError, ValueError) as e:
            return _degrade(f'entry {_safe_repr(entry)} is not a valid Referent: {e}')

    return tuple(decoded), source


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
    #: The verification sub-pass's findings (task 3671, PRD leaf zeta). A
    #: STRUCTURED RECORD SET rather than an int, deliberately: zeta's
    #: postcondition is INV-2 structured-facts-at-failure — "which edge, which
    #: end, which check, what should it have pointed at" — and a count cannot
    #: carry any of that, which is exactly why the logger.debug-only int shape
    #: above is not enough for it. Leaf eta reads the findings off this field.
    referent_stats: ReferentStats = field(default_factory=lambda: ReferentStats())
    #: The repair sub-pass's records (task 3672, PRD leaf eta) — what was
    #: actually DONE about the findings on ``referent_stats``, including the
    #: two dispositions where the right answer was to do nothing. A structured
    #: record set for the same reason zeta's is: "we refused to guess at this
    #: edge end" is a fact an operator has to act on, and no count can carry
    #: it. Eta is the only WRITING consumer of zeta's detect-only output.
    repair_stats: ReferentRepairStats = field(
        default_factory=lambda: ReferentRepairStats()
    )
    errors: list[str] = field(default_factory=list)


#: THE closed vocabulary of verification checks (task 3671, PRD leaf zeta).
#: The single normative site for the "which check fired" field the PRD's
#: §Contract requires on every repair record — a check name must be REGISTERED
#: here, never spelled as a bare string at a call site, or the two consumers
#: (leaf eta's repair, leaf iota's rate) key off vocabularies that drift.
#:
#: The C' post-LLM veto is deliberately NOT a third member. Post-write,
#: "extracted Task N was merged onto the Task M node" is observationally
#: IDENTICAL to "an edge about Task M is attached to a Task N node" — which
#: these two checks already detect. A separate veto mechanism would be two
#: sites that must agree byte-for-byte, i.e. exactly the INV-5 lockstep
#: duplication utils/canonical_labels.py exists to prevent. The PRD says so
#: outright: it "folds in", and is "not a distinct leaf".
REFERENT_CHECKS: tuple[str, ...] = ('set-membership', 'per-edge-pairing')


@dataclass(frozen=True, kw_only=True)
class ReferentFinding:
    """One edge END that landed on a node the write was not about.

    The structured record INV-2 requires, carrying every field the PRD
    §Contract names — "edge uuid, old endpoint uuid, new endpoint uuid,
    referent set, which check fired" — plus what leaf eta needs to act:

    ==========================  ====================================
    PRD field                   Attribute
    ==========================  ====================================
    edge uuid                   :attr:`edge_uuid` (+ :attr:`which_end`)
    old endpoint uuid           :attr:`old_endpoint_uuid`
    new endpoint uuid           :attr:`new_endpoint_uuid`
    referent set                :attr:`referent_set`
    which check fired           :attr:`check`
    ==========================  ====================================

    FROZEN, and its collection field is a TUPLE, for the reason
    :class:`~fused_memory.utils.canonical_labels.Referent` and ``LabelScan``
    are: a finding is evidence for DESTRUCTIVE edge surgery, and ``frozen=True``
    blocks attribute rebinding only — a list field would leave
    ``finding.referent_set.append(...)`` open, letting a consumer quietly widen
    the set that justified the repair it is about to perform.

    Keyword-only because eleven fields, seven of them strings, is exactly the
    shape where a positional argument silently lands in the wrong slot.

    ``new_endpoint_uuid is None`` means "the node does not exist yet, or its
    name keys a duplicate-name group" — leaf eta resolves-or-mints via
    ``ensure_entity_node``, which handles both identically. It does NOT mean
    unrepairable; that is :attr:`resolvable`, which zeta defaults to ``False``
    so "recorded and left alone, never guessed at" is the structural default
    rather than something every construction site must remember.
    """

    #: The edge whose endpoint is wrong.
    edge_uuid: str
    #: Which end: ``'source'`` or ``'target'``. With :attr:`edge_uuid` this is
    #: the identity of the finding — at most one finding per (edge, end).
    which_end: str
    #: Which check fired; one of :data:`REFERENT_CHECKS`.
    check: str
    #: The node the edge is attached to today.
    old_endpoint_uuid: str
    #: That node's name as this episode's result reported it. Recorded for the
    #: operator log; the VERDICT is keyed off :attr:`endpoint_referent`, since a
    #: spelling can have been normalized out from under this string.
    old_endpoint_name: str
    #: The parsed referent that name denotes — the thing actually compared.
    endpoint_referent: Referent
    #: The declared referent set, as canonical node names, that the endpoint
    #: was tested against.
    referent_set: tuple[str, ...]
    #: The referent the edge SHOULD hang off, when exactly one candidate
    #: survives. ``None`` whenever :attr:`resolvable` is False.
    intended_referent: Referent | None = None
    #: The uuid of :attr:`intended_referent`'s node, when it resolves to
    #: exactly one live node. See the class docstring for what ``None`` means.
    new_endpoint_uuid: str | None = None
    #: Whether a correct target was determined. Defaults False — fail-closed.
    resolvable: bool = False
    #: Why not, when :attr:`resolvable` is False. Empty on a resolvable
    #: finding.
    reason: str = ''

    def __post_init__(self) -> None:
        if self.check not in REFERENT_CHECKS:
            raise ValueError(
                f'unregistered referent check {self.check!r}; registered checks '
                f'are {list(REFERENT_CHECKS)}. Add it to '
                'memory_service.REFERENT_CHECKS rather than recording a finding '
                'no consumer can key off.'
            )

    def to_dict(self) -> dict[str, Any]:
        """A plain, JSON-safe dict keyed exactly by this record's field names.

        The payload the operator warning carries. Referents render as their
        canonical ``node_name`` rather than as a dataclass repr, so the log
        line and any future durable row read as graph names — the same thing
        an operator would type into a query.
        """
        return {
            'edge_uuid': self.edge_uuid,
            'which_end': self.which_end,
            'check': self.check,
            'old_endpoint_uuid': self.old_endpoint_uuid,
            'old_endpoint_name': self.old_endpoint_name,
            'endpoint_referent': self.endpoint_referent.node_name,
            'referent_set': list(self.referent_set),
            'intended_referent': (
                self.intended_referent.node_name
                if self.intended_referent is not None
                else None
            ),
            'new_endpoint_uuid': self.new_endpoint_uuid,
            'resolvable': self.resolvable,
            'reason': self.reason,
        }

def _endpoint_referent(endpoint_name: str, *, group_id: str) -> Referent | None:
    """The referent an edge ENDPOINT's node name denotes, or ``None``.

    ``None`` for an empty name (the endpoint this episode's result does not
    name), and for a name that is not a canonical task label at all
    ('MergeWorker') or merely MENTIONS one ('Task 42 orchestrator' —
    ``parse_node_name`` is anchored).

    A named function rather than an inline expression so the SOURCE-INVARIANT
    reclassification cannot be dropped by a later edit that only means to
    re-order the tuple this feeds: the bare ``parse_node_name`` it replaces read
    as complete, which is precisely how ζ came to be the one referent path that
    skipped the rule. See
    :func:`~fused_memory.utils.referent_resolution.local_referent`.
    """
    if not endpoint_name:
        return None
    referent = parse_node_name(endpoint_name)
    if referent is None:
        return None
    return local_referent(referent, group_id=group_id)


def _candidate_pool(
    *,
    referents: frozenset[Referent],
    cited: frozenset[Referent],
    endpoint: Referent,
    ambiguous: frozenset[Referent],
    source: str,
) -> frozenset[Referent]:
    """The evidence rule, before either endpoint is subtracted.

    ``cited & referents``, falling back to the whole of ``referents`` ONLY when
    the fact says nothing about where this edge belongs. The edge's own fact is
    the sharpest evidence available about which node THIS edge belongs on, so a
    citation the declaration corroborates wins; the whole declared set is the
    fallback for when the fact cites nothing the declaration also names.
    INTERSECTING rather than unioning is what keeps a repair target from ever
    originating outside the referent set — an LLM-restated fact naming a task
    the write never declared itself to be about must not become a target.
    Fact-scoping is also what keeps mode (iii) repairable: with referents
    {3074, 3075} the whole-set fallback would see two candidates and abandon a
    repair the fact unambiguously determines.

    THE CORROBORATION GUARD (``endpoint in cited``) is what makes the fallback
    safe on the SET-MEMBERSHIP arm, and it is the membership-arm counterpart of
    the pairing arm's ``cited_declared`` guard — same principle, same
    fail-closed direction. A fact that NAMES the very node its edge landed on is
    the strongest possible evidence the attachment is CORRECT, so it must not be
    read as "the fact is silent, fall back to the declared set". Without the
    guard the dominant legitimate write shape becomes a repair instruction:
    ``resolve_referents`` derives ``source='metadata'`` from the write's ambient
    ``task_id``, and its own docstring names the mismatch as deliberately NOT a
    conflict — "An agent working on task 3668 legitimately writes memories about
    Task 2500". With referents {3668} and an edge whose fact reads "Task 2500
    was completed by the merge worker" hanging off the ``Task 2500`` node, the
    unguarded fallback yields the sole candidate ``Task 3668`` and hands leaf eta
    a ``resolvable=True`` instruction to repoint a CORRECT edge onto the task the
    agent merely happened to be working on — manufacturing the exact
    misattribution this PRD exists to prevent, and polluting the rate leaf iota
    samples with a finding that has no observable defect.

    THE CORROBORATION GUARD IS NOT SUFFICIENT ON ITS OWN, and two further vetoes
    sit beside it because it closes only the SUBSET of that shape where the fact
    happens to name the endpoint. An LLM-paraphrased fact that restates no task
    number at all ("the merge worker completed it") is the routine extraction
    outcome, and on such a fact ``cited`` is empty, so the corroboration guard
    cannot fire and the unguarded fallback re-manufactures exactly the
    ``resolvable=True``-onto-the-ambient-task instruction described above.

    VETO 1 — AN AMBIGUOUS ENDPOINT (PRD boundary row "Ambiguous scan | ref routed
    to ``.ambiguous``; treated as undeclared; recorded, not guessed"). γ routes a
    number claimed by BOTH a bare own-project mention and a foreign-qualified
    reference to ``LabelScan.ambiguous`` and EXCLUDES it from ``.referents``, on
    purpose. ε then drops ``.ambiguous`` from the wire, so a consumer reading only
    the decoded set sees an ambiguous endpoint as a plain non-member —
    indistinguishable from a genuine conflation (``_encode_referents``:
    "AMBIGUITY IS DELIBERATELY NOT THREADED — READ THIS BEFORE WRITING ZETA").
    ζ therefore RE-DERIVES the producer's ambiguity set from ``content`` and
    suppresses the pool for any endpoint in it: an ambiguous reference must be
    RECORDED and LEFT ALONE, never handed to eta as destructive repair surgery.
    Tested FIRST, ahead of even the corroboration guard, because it is the
    strongest "do not touch this" signal available and must hold whatever the
    fact happens to cite.

    VETO 2 — A ``source='metadata'`` FALLBACK. ``resolve_referents`` ranks ambient
    ``metadata['task_id']`` ABOVE the content-derived scan, and its own docstring
    names the resulting mismatch as deliberately NOT a conflict: "An agent working
    on task 3668 legitimately writes memories about Task 2500". A referent set
    bridged from the task an agent merely HAPPENS to be dispatched on is not a
    claim about which node any particular edge belongs on, so it must not become a
    repair target by default. The whole-declared-set fallback is therefore
    suppressed for ``source='metadata'``; ``'declared'`` (the caller stated its
    referents) and ``'derived'`` (they were scanned out of this very content) keep
    the fallback, because there the declared set genuinely IS evidence about the
    content. The ``cited & referents`` intersection survives on every source: a
    fact that names a declared referent is per-EDGE evidence regardless of how the
    declaration was sourced.

    THE CORROBORATION GUARD IS TESTED FIRST OF THE TWO CITATION RULES, AND THAT
    ORDER IS LOAD-BEARING — not stylistic.
    Behind the intersection short-circuit the guard is UNREACHABLE for every
    fact that cites the endpoint AND some declared referent, which is not an
    exotic shape but the same legitimate ambient-task write one sentence longer:
    "Task 2500 was completed as part of task 3668 by the merge worker" cites
    {2500, 3668}, so ``cited & referents`` is ``{3668}`` — non-empty — and an
    intersection-first order returns it, re-manufacturing the exact
    ``resolvable=True``-onto-Task-3668 instruction the paragraph above exists to
    prevent, on a fact that literally asserts the edge is about Task 2500. A
    citation of the endpoint therefore suppresses the pool UNCONDITIONALLY:
    corroboration is not merely a fallback the intersection can outrank, it is a
    veto. Nothing this test shadows is lost on the pairing arm, which is only
    reached when ``endpoint_referent not in cited`` — the guard can never fire
    there, so mode (iii) still resolves through the intersection below.

    Corroborated findings are still RECORDED — they are just recorded with an
    empty pool, which becomes ``resolvable=False`` plus a reason at the caller.
    That is this pass's stated postcondition: recorded and left alone, never
    guessed at.

    Extracted so the rule lives at ONE site that both :func:`_candidate_targets`
    and :func:`_unresolvable_reason` read. Without it the reason builder would
    have to RECOMPUTE the pool to explain itself — a second copy that must agree
    with the first byte-for-byte, which is exactly the INV-5 lockstep
    duplication this PRD exists to avoid.

    Args:
        referents: The set the write declared itself to be about.
        cited: The referents this edge's own fact mentions.
        endpoint: The referent the flagged endpoint currently parses as — read
            ONLY to ask whether the fact corroborates it, and whether it was
            ambiguous. The subtraction of the endpoint from the pool stays in
            :func:`_candidate_targets`, so this function remains "which referents
            is there evidence for", not "which targets survive".
        ambiguous: The referents the EPISODE CONTENT was ambiguous about, as
            re-derived by :meth:`MemoryService._verify_episode_referents` from
            ``scan_content(content, group_id=...).ambiguous``. Already through
            :func:`~fused_memory.utils.referent_resolution.local_referent`, so it
            compares equal to *endpoint* on a self-qualified spelling.
        source: The ``ReferentSource`` :func:`_decode_referents` read off the
            queue payload — one of :data:`REFERENT_SOURCES`. Read ONLY to decide
            whether the whole-declared-set fallback is licensed (veto 2 above).
    """
    if endpoint in ambiguous:
        # VETO 1. The episode content itself could not say which project's task
        # this number denotes, so there is nothing here to repair TOWARDS.
        # Ahead of the corroboration guard deliberately: an ambiguous endpoint is
        # unrepairable whatever the fact happens to cite.
        return frozenset()
    if endpoint in cited:
        # The fact names the node this edge end is already on. It is evidence
        # FOR the current attachment, never for repointing it elsewhere, so the
        # pool is suppressed rather than allowed to nominate a target the fact
        # does not support.
        #
        # FIRST, ahead of the intersection: a fact citing BOTH the endpoint and
        # a declared referent ("Task 2500 was completed as part of task 3668")
        # has a non-empty intersection, so testing the intersection first would
        # short-circuit past this guard entirely and return the ambient task as
        # a repair target. See the docstring — this is a veto, not a fallback.
        return frozenset()
    corroborated_citations = cited & referents
    if corroborated_citations:
        return corroborated_citations
    if source == 'metadata':
        # VETO 2. The fact cites no declared referent, and the declaration is
        # only the task this agent happened to be dispatched on — ambient
        # context, not an assertion about where this edge belongs. Falling back
        # to it here is what would manufacture the misattribution this PRD
        # exists to prevent on the dominant legitimate write shape.
        return frozenset()
    return referents


def _candidate_targets(
    *,
    referents: frozenset[Referent],
    cited: frozenset[Referent],
    endpoint: Referent,
    other_endpoint: Referent | None,
    ambiguous: frozenset[Referent],
    source: str,
) -> tuple[Referent, ...]:
    """Which referent could this misattached edge end correctly point at?

    A pure module-level function — no ``self``, no I/O — so the rule that
    decides whether leaf eta may perform destructive edge surgery is directly
    unit-testable in isolation from the walk that drives it.

    The rule, in order:

    1. ``pool = _candidate_pool(...)`` — the fact-cited intersection when it is
       non-empty; else the whole declared set, UNLESS one of three vetoes
       empties it: the fact cites the endpoint itself (corroboration), the
       endpoint referent was AMBIGUOUS in the episode content, or the
       declaration came from ambient ``source='metadata'`` and the fact cites no
       declared referent. See that function for why each is load-bearing on the
       dominant legitimate write shape.
    2. Subtract *endpoint*, the referent this finding is ABOUT. A "repair" onto
       the node the edge is already attached to is not a repair — and is not
       even a harmless no-op, because :meth:`_intended_endpoint_uuid` resolves
       the CANONICAL name: with a non-canonical endpoint spelling
       (``'task #3074'``) and a canonical ``'Task 3074'`` node both present it
       yields a DIFFERENT uuid, and eta would perform real edge surgery on an
       endpoint that was already correct.

       On the SET-MEMBERSHIP arm this subtraction is provably a NO-OP: the pool
       is always a subset of ``referents``, and membership fires precisely when
       the endpoint is NOT in ``referents``, so the endpoint can never be in the
       pool. It is therefore a STRUCTURAL GUARANTEE at the single site that
       decides targets rather than a behaviour change on the dominant path —
       which is the point: a future third check cannot silently reintroduce a
       self-targeting repair by forgetting to guard for it.

       The invariant is deliberately NOT additionally enforced by a raising
       ``ReferentFinding.__post_init__`` validator. This pass runs inside an
       already-committed write's identity-lock critical section, where raising
       is strictly worse than recording: the write has landed either way, and an
       exception would destroy the very evidence eta needs.
    3. Subtract *other_endpoint*. Not defensive ceremony: ``reassign_edge``
       (graphiti_client.py) explicitly refuses a move that would fold the edge
       into a self-loop, so a "target" equal to the edge's other end is not a
       repair eta could perform. This subtraction is precisely what turns the
       live Task 2519/2520 case — referents {2519}, endpoints (Task 2519,
       Task 2520), a fact unary about 2519 — into the zero-candidate row the PRD
       names as explicitly unrepairable.
    4. Return in a deterministic order.

    Exactly one survivor means the correct target is DETERMINED. Zero or more
    than one means it is not, and the caller records the finding with
    ``resolvable=False`` and a reason: RECORDED AND LEFT ALONE — never silently
    dropped, and never guessed at. That is why
    :attr:`ReferentFinding.resolvable` defaults to ``False`` rather than
    ``True``: the fail-closed direction is structural rather than a matter of
    every construction site remembering to say so.

    Args:
        referents: The set the write declared itself to be about.
        cited: The referents this edge's own fact mentions
            (``scan_content(...).refs``, which already excludes ambiguity).
        endpoint: The referent the flagged endpoint currently parses as.
            Non-optional: a finding is only ever built for an endpoint that
            PARSED, so the flagged referent is always known — encoded in the
            type rather than accepting a ``None`` no call site can produce.
        other_endpoint: The referent at the edge's OTHER end, or ``None`` when
            that end is not a task node at all.
        ambiguous: The referents the EPISODE CONTENT was ambiguous about.
            Forwarded verbatim to :func:`_candidate_pool` (veto 1).
        source: The ``ReferentSource`` the declaration came from. Forwarded
            verbatim to :func:`_candidate_pool` (veto 2).

    Returns:
        The surviving candidates, sorted by ``(kind, project_id, number)``.
        Sorted rather than kept in the caller's first-seen order because the
        inputs are FROZENSETS, whose iteration order is not stable across
        processes under hash randomization — and a finding must be stable across
        runs and diffable in eta's audit.
    """
    # `other_endpoint` may be None; None is simply not a member of a
    # frozenset[Referent], and typeshed types `frozenset.__sub__` as accepting
    # AbstractSet[_T_co | None], so no explicit `- {None}` branch is needed.
    pool = _candidate_pool(
        referents=referents, cited=cited, endpoint=endpoint,
        ambiguous=ambiguous, source=source,
    ) - {endpoint, other_endpoint}
    return tuple(sorted(pool, key=lambda r: (r.kind, r.project_id, r.number)))

def _unresolvable_reason(
    candidates: tuple[Referent, ...],
    *,
    pool: frozenset[Referent],
    cited: frozenset[Referent],
    endpoint: Referent,
    other_endpoint: Referent | None,
    ambiguous: frozenset[Referent],
    source: str,
) -> str:
    """Why :func:`_candidate_targets` could not determine a correct target.

    Carried on the finding so "recorded and left alone" is legible as a REASON
    rather than as an absence — a reader must be able to tell an unrepairable
    row from a row nobody looked at, and to tell "the check had nothing to point
    at but the node it was already on" from "the only target would form a
    self-loop".

    Args:
        candidates: What :func:`_candidate_targets` returned.
        pool: The PRE-subtraction pool from :func:`_candidate_pool`. Membership
            is tested here rather than inferred from ``other_endpoint is None``
            precisely so the message stays HONEST when BOTH subtractions apply.
        cited: The referents this edge's own fact mentions — the same set the
            pool was computed from, so the corroboration branch reads the SAME
            input :func:`_candidate_pool` decided on rather than re-deriving it
            from the empty pool it produced (which is indistinguishable from an
            empty declared set).
        endpoint: The referent the flagged endpoint currently parses as.
        other_endpoint: The referent at the edge's other end, or ``None``.
        ambiguous: The referents the EPISODE CONTENT was ambiguous about — the
            same set :func:`_candidate_pool` vetoed on, read here for the SAME
            reason ``cited`` is: an emptied pool cannot say WHICH veto emptied
            it, and the three vetoes need three different explanations.
        source: The ``ReferentSource`` the declaration came from, likewise.
    """
    if len(candidates) > 1:
        return (
            'more than one candidate target survives '
            f'({[c.node_name for c in candidates]}) and the edge fact does not '
            'discriminate between them; recorded, not guessed at'
        )
    # Zero candidates. Either `_candidate_pool` returned nothing (the
    # corroboration branch below), or one of the two subtractions emptied it.
    #
    # Ordered FIRST among the zero-candidate branches because corroboration is
    # the most specific thing that can be said about a finding: when the fact
    # names the endpoint, "there was no target" is true but uninformative, and
    # "the fact says this edge belongs where it is" is the reason an operator
    # (and leaf eta) actually needs. It also cannot be inferred from `pool`,
    # which the guard deliberately empties.
    #
    # AMBIGUITY OUTRANKS CORROBORATION here, mirroring the veto order in
    # `_candidate_pool`: when the content could not say which project's task the
    # number denotes, that is the fact about this row an operator (and eta) most
    # needs, and it holds whatever the edge fact happens to cite.
    if endpoint in ambiguous:
        return (
            f'the endpoint referent {endpoint.node_name!r} was AMBIGUOUS in the '
            'episode content — claimed by both a bare own-project mention and a '
            'foreign-qualified reference — so it is treated as undeclared rather '
            'than as a conflation; recorded, not guessed at'
        )
    if endpoint in cited:
        return (
            f"the edge's own fact cites {endpoint.node_name!r}, the endpoint it "
            'landed on, which corroborates the current attachment; the declared '
            'referent set is not evidence for repointing it, so this is '
            'recorded, not repaired'
        )
    if source == 'metadata' and not pool:
        # Veto 2. Reached only when neither veto above fired and the fact cited
        # no declared referent, so the ONLY thing that could have supplied a
        # target was the whole-declared-set fallback the source suppresses.
        return (
            "the write's referent set was bridged from ambient "
            "metadata['task_id'] rather than declared or derived from the "
            'content, and this edge\'s fact cites no declared referent; the task '
            'an agent happens to be dispatched on is not evidence about which '
            'node this edge belongs on, so this is recorded, not repaired'
        )
    if endpoint in pool:
        return (
            f'the only candidate target {endpoint.node_name!r} is the node this '
            'edge end is already attached to, so there is nothing to repoint '
            'to; recorded, not repaired'
        )
    if other_endpoint is not None:
        # The live Task 2519/2520 row.
        return (
            f'the only candidate target {other_endpoint.node_name!r} is this '
            "edge's other endpoint, so repointing would form the self-loop "
            'reassign_edge refuses; there is no correct target'
        )
    # Defensive: unreachable while the caller no-ops on an empty referent set.
    # Kept so a future relaxation records a reason rather than an empty string
    # that reads as "resolvable".
    return 'no candidate target could be determined from the declared referents'



@dataclass
class ReferentStats:
    """What one ``_verify_episode_referents`` run looked at, and what it found.

    The in-process half of INV-2's structured record: leaf eta reads
    :attr:`findings` off the return value, inside the same identity-lock
    critical section, and acts on it. (The process-lifetime half leaf iota
    reads is ``MemoryService.referent_finding_counts``.)

    The three summary counts are ``@property`` comprehensions over
    :attr:`findings` rather than fields precisely so they CANNOT drift from the
    list they summarize — the same property-not-field discipline
    :attr:`Referent.node_name` follows. A stored count is a second site that
    must be incremented in lockstep with every append.

    :attr:`endpoints_unresolved` exists so this pass's one blind spot — an edge
    endpoint uuid that this episode's ``result.nodes`` does not name, so its
    name is unknown and it cannot be checked at all — is COUNTED rather than
    silently skipped. A skipped endpoint is a check that did not run, and a
    verification pass that cannot say how often it declined to look is not a
    verification pass.
    """

    edges_scanned: int = 0
    endpoints_checked: int = 0
    endpoints_unresolved: int = 0
    findings: list[ReferentFinding] = field(default_factory=list)

    @property
    def set_membership_findings(self) -> int:
        """Findings from the SET MEMBERSHIP check."""
        return sum(1 for f in self.findings if f.check == 'set-membership')

    @property
    def pairing_findings(self) -> int:
        """Findings from the PER-EDGE PAIRING check."""
        return sum(1 for f in self.findings if f.check == 'per-edge-pairing')

    @property
    def unresolvable_findings(self) -> int:
        """Findings recorded with no determinable correct target — left alone."""
        return sum(1 for f in self.findings if not f.resolvable)


#: THE closed vocabulary of repair dispositions (task 3672, PRD leaf eta).
#: The single normative site for the "what happened to this finding" field on
#: every repair record — an outcome must be REGISTERED here, never spelled as a
#: bare string at a call site, or the operator surface and leaf iota's rate key
#: off vocabularies that drift.
#:
#: The four are genuinely different answers to the operator's question, and the
#: split is load-bearing:
#:
#: * ``'repaired'``    — the ensure -> reassign sequence ran. ``moved=False``
#:   inside this outcome is ``reassign_edge``'s own corroborate-before-acting
#:   no-op (the edge was ALREADY correct), which is why the streak counts
#:   ``moved=True`` records rather than this outcome alone.
#: * ``'unrepairable'``— zeta could not determine a correct target and eta
#:   REFUSED TO GUESS. Working as designed.
#: * ``'degenerate'``  — both ends of one edge would land on one node; the edge
#:   was skipped WHOLE. Also a refusal, which is why it shares
#:   ``flagged_unrepairable`` with the row above.
#: * ``'failed'``      — we tried and the backend did not cooperate. An
#:   INFRASTRUCTURE signal, deliberately not folded into the refusal bucket:
#:   conflating them would let a FalkorDB outage read as a scanner regression
#:   in leaf iota's rate, and would feed a false repair-storm streak.
REFERENT_REPAIR_OUTCOMES: tuple[str, ...] = (
    'repaired', 'unrepairable', 'degenerate', 'failed',
)

#: CONSECUTIVE episodes whose repair pass moved at least one edge endpoint
#: before the INV-4 storm alarm fires (task 3672, PRD leaf eta).
#:
#: TEN, which is this leaf's resolution of PRD OPEN QUESTION 1, taking the
#: PRD's own suggested value. Chosen against the measured base rate: ~0.22% of
#: live task-mentioning edges carry a conflated endpoint, arriving at random.
#: Ten consecutive episodes each needing a repair is not that distribution by
#: any reading, so the threshold trades a vanishing false-positive rate for a
#: detection delay of at most ten writes — the right side of that trade for an
#: alarm whose subject is a REGRESSION IN THE PRODUCER, which by definition
#: persists until someone fixes it.
#:
#: A LOWER value would page on ordinary clustering (a burst of writes about one
#: task family legitimately repairs several times running); a HIGHER one buys
#: nothing, because the condition does not self-heal.
_REFERENT_REPAIR_STREAK_THRESHOLD: int = 10


def _episode_uuid_of(result: Any) -> str:
    """The uuid of the episode *result* describes, or ``''`` if unreadable.

    WHY THE EXTRACTION IS DEFENSIVE rather than a plain ``result.episode.uuid``.
    Its one caller is the EIGHTH sub-pass of ``_reconcile_episode_identity``,
    which runs AFTER the episode write is already durable —
    :meth:`MemoryService._reconcile_episode_identity`'s ``_run_pass`` guard
    exists precisely so a sub-pass failure never fails that committed write.
    The other seven sub-passes already treat ``result`` as duck-typed. Raising
    an ``AttributeError`` while merely READING an identifier would forfeit the
    repair for a reason wholly unrelated to repairing, which is an absurd way
    to lose it.

    WHAT ``''`` MEANS DOWNSTREAM. Not "no exclusion is needed" but "we could not
    establish which episode this is, so exclude nothing and refuse more
    deletions" — the fail-closed direction, consistent with
    :attr:`ReferentFinding.resolvable` defaulting to ``False``. It flows into
    :meth:`GraphitiBackend.count_foreign_relationships`, whose ``episode_uuid``
    argument narrows a delete guard; dropping the exclusion can only ever make
    that guard STRICTER.

    A non-``str`` uuid is rejected along with an absent one. That is not
    hypothetical tidiness: a ``MagicMock`` result — the shape most of this
    module's tests hand it — auto-creates ``.episode.uuid`` as a child mock,
    and passing that into a Cypher parameter would be a silently wrong query
    rather than a clean refusal.

    Args:
        result: Whatever ``graphiti.add_episode`` returned, however malformed.

    Returns:
        A non-empty ``str`` uuid, or ``''``.
    """
    try:
        uuid = getattr(getattr(result, 'episode', None), 'uuid', None)
    except Exception:
        # A property that raises is still a shape we must survive; see above.
        return ''
    return uuid if isinstance(uuid, str) and uuid else ''


@dataclass(frozen=True, kw_only=True)
class ReferentRepair:
    """What eta DID about one :class:`ReferentFinding` — the audit record.

    The eta half of INV-2 structured-facts-at-failure. zeta's finding says
    "this edge end is wrong and here is what it should point at"; this record
    says what was then done about it, including the two dispositions where the
    right answer was to do NOTHING. A dropped finding and a repaired one are
    indistinguishable to a downstream rate; an ``'unrepairable'`` record is the
    evidence a human uses to decide the case by hand.

    FROZEN, and its collection field is a TUPLE, for exactly the reason
    :class:`ReferentFinding` is: this is evidence for DESTRUCTIVE edge surgery,
    and ``frozen=True`` blocks attribute rebinding only — a list field would
    leave ``record.summaries_refreshed.append(...)`` open, letting a consumer
    quietly widen the claim about what this pass actually refreshed. The
    emptied-node stamp is applied with :func:`dataclasses.replace`, never
    mutation, for the same reason.

    Keyword-only because twelve fields, seven of them strings, is exactly the
    shape where a positional argument silently lands in the wrong slot.
    """

    #: The edge this record is about.
    edge_uuid: str
    #: Which end: ``'source'`` or ``'target'``. With :attr:`edge_uuid` this is
    #: the identity of the finding this record answers.
    which_end: str
    #: What happened; one of :data:`REFERENT_REPAIR_OUTCOMES`.
    outcome: str
    #: The node the edge end was attached to when the finding was recorded.
    old_endpoint_uuid: str
    #: Which zeta check produced the finding, carried through unchanged so a
    #: reader never has to join this record back to a ``ReferentStats`` to
    #: learn why the repair happened.
    check: str
    #: The node the end now points at. ``''`` when nothing was targeted (every
    #: outcome other than ``'repaired'``).
    new_endpoint_uuid: str = ''
    #: The canonical ``node_name`` of the referent that target denotes. ``''``
    #: when none — a finding zeta left unresolvable names no intended referent.
    intended_referent: str = ''
    #: Whether ``ensure_entity_node`` MINTED the target rather than resolving
    #: an existing node.
    minted: bool = False
    #: Whether ``reassign_edge`` actually moved the endpoint. ``False`` on its
    #: corroborated no-op arm — the edge was already correct.
    moved: bool = False
    #: Node uuids whose summary is known to reflect the post-repair edge set:
    #: the union of what ``reassign_edge`` reported refreshing itself and what
    #: eta's backstop then re-refreshed. Says what is true of the GRAPH, not
    #: what this method happened to call.
    summaries_refreshed: tuple[str, ...] = ()
    #: The uuid of an emptied node this pass deleted, when the narrow
    #: three-condition cleanup fired. ``''`` when nothing was deleted.
    deleted_emptied_node: str = ''
    #: Why, for the outcomes that did not repair. Carried VERBATIM from
    #: ``ReferentFinding.reason`` on the ``'unrepairable'`` arm — the operator
    #: must see zeta's own explanation, not an eta-authored paraphrase — and
    #: carrying the exception text on the ``'failed'`` arm.
    reason: str = ''

    def __post_init__(self) -> None:
        if self.outcome not in REFERENT_REPAIR_OUTCOMES:
            raise ValueError(
                f'unregistered repair outcome {self.outcome!r}; registered '
                f'outcomes are {list(REFERENT_REPAIR_OUTCOMES)}. Add it to '
                'memory_service.REFERENT_REPAIR_OUTCOMES rather than recording '
                'a disposition no consumer can key off.'
            )

    def to_dict(self) -> dict[str, Any]:
        """A plain, JSON-safe dict keyed exactly by this record's field names.

        The payload the operator warning and the storm escalation's detail
        carry. :attr:`summaries_refreshed` renders as a LIST because a tuple is
        not JSON, and the escalation detail is serialized.
        """
        return {
            'edge_uuid': self.edge_uuid,
            'which_end': self.which_end,
            'outcome': self.outcome,
            'old_endpoint_uuid': self.old_endpoint_uuid,
            'new_endpoint_uuid': self.new_endpoint_uuid,
            'intended_referent': self.intended_referent,
            'check': self.check,
            'minted': self.minted,
            'moved': self.moved,
            'summaries_refreshed': list(self.summaries_refreshed),
            'deleted_emptied_node': self.deleted_emptied_node,
            'reason': self.reason,
        }


@dataclass
class ReferentRepairStats:
    """What one ``_repair_episode_referents`` run did, and what it declined to do.

    The in-process half of eta's INV-2 record: returned on
    ``ReconcileStats.repair_stats`` and carried verbatim into the repair-storm
    escalation's detail, so the alarm ships the EVIDENCE rather than a count.

    Every summary count is a ``@property`` comprehension over :attr:`repairs`
    rather than a field, precisely so it CANNOT drift from the list it
    summarizes — the same property-not-field discipline
    :class:`ReferentStats` follows. A stored count is a second site that must
    be incremented in lockstep with every append.
    """

    repairs: list[ReferentRepair] = field(default_factory=list)

    @property
    def repaired(self) -> int:
        """Endpoints this pass actually MOVED.

        ``moved=True`` specifically, not merely ``outcome == 'repaired'``: a
        ``moved=False`` result is ``reassign_edge``'s corroborate-before-acting
        no-op, meaning the edge was already correct. Counting it would make the
        INV-4 storm streak fire on a graph that needed no repairs at all.
        """
        return sum(
            1 for r in self.repairs if r.outcome == 'repaired' and r.moved
        )

    @property
    def flagged_unrepairable(self) -> int:
        """Findings RECORDED AND LEFT ALONE — the NEVER GUESS disposition.

        Deliberately folds ``'unrepairable'`` and ``'degenerate'`` into ONE
        bucket: the task's NEVER GUESS rule assigns both the same disposition
        (recorded, not acted on), so the operator reads one number for "we
        refused to act". ``'failed'`` is excluded on purpose — see
        :data:`REFERENT_REPAIR_OUTCOMES`.
        """
        return sum(
            1 for r in self.repairs
            if r.outcome in ('unrepairable', 'degenerate')
        )

    @property
    def degenerate_edges(self) -> int:
        """EDGES skipped whole, not records — a degenerate edge yields one
        record per end, and the operator's question is how many edges."""
        return len({r.edge_uuid for r in self.repairs if r.outcome == 'degenerate'})

    @property
    def failed(self) -> int:
        """Findings whose repair was ATTEMPTED and did not complete."""
        return sum(1 for r in self.repairs if r.outcome == 'failed')

    @property
    def nodes_minted(self) -> int:
        """Repair targets ``ensure_entity_node`` had to create."""
        return sum(1 for r in self.repairs if r.minted)

    @property
    def nodes_deleted(self) -> int:
        """DISTINCT emptied nodes the cleanup phase deleted.

        Distinct, because two repairs moving endpoints off the same node stamp
        that one deletion onto both records.
        """
        return len({
            r.deleted_emptied_node for r in self.repairs if r.deleted_emptied_node
        })


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
        # INV-4 storm escape for the referent-set queue channel (task 3670, PRD
        # leaf epsilon). `_decode_referents` degrades an unreadable or absent
        # blob to ('none') rather than raising — losing the memory over a
        # telemetry field would be worse — so that degradation MUST be counted
        # rather than silently fallen through. Constructed UNCONDITIONALLY, for
        # the same reason the two counters above are: an alarm that only exists
        # when `_write_journal` is configured would vanish in exactly the
        # degraded configuration where a referent-less write storm is least
        # likely to be noticed any other way.
        #
        # Bucketed by ALL FOUR sources, not just 'none', because leaf iota needs
        # a DENOMINATOR: "sustained 100% none" is a rate, and an absolute
        # none-count alone cannot distinguish a broken producer from a quiet
        # system. Keyed off gamma's exported REFERENT_SOURCES so the vocabulary
        # lives at ONE site (that constant's stated purpose) and a fifth source
        # cannot escape the counter.
        #
        # Bounded by that four-member closed vocabulary, so unlike the per-agent
        # storm counters above it needs no pruning.
        self._referent_source_counts: dict[str, int] = dict.fromkeys(REFERENT_SOURCES, 0)
        # INV-4 storm escape for the verification sub-pass (task 3671, PRD leaf
        # zeta), and leaf iota's read side. Copies the shape directly above for
        # the same stated reasons: constructed UNCONDITIONALLY — never obtained
        # from `_write_journal` or the ReconciliationHarness — so it does not go
        # dark in exactly the degraded configuration where a finding storm is
        # least likely to be noticed any other way; and bounded by a closed
        # vocabulary, so unlike the per-agent storm counters above it needs no
        # pruning.
        #
        # Keyed off REFERENT_CHECKS so the check vocabulary lives at ONE site
        # and a third check cannot escape the counter. The extra 'unresolvable'
        # bucket is a SECOND, ORTHOGONAL axis (whether a finding can be acted on)
        # rather than a third check, so the buckets deliberately do not sum to
        # the finding total.
        self._referent_finding_counts: dict[str, int] = dict.fromkeys(
            (*REFERENT_CHECKS, 'unresolvable'), 0,
        )
        # INV-4 storm escape for the REPAIR sub-pass (task 3672, PRD leaf eta):
        # the counter half of the alarm whose fire half is
        # `middleware/referent_repair_storm_escalator`. Both dicts are
        # constructed UNCONDITIONALLY, for the reason the two counters above
        # give and which applies with more force here: this is the escape hatch
        # for a pass that WRITES to the graph, and an alarm sourced from
        # `_write_journal` or the ReconciliationHarness would go dark in exactly
        # the degraded configuration where a repair storm is least likely to be
        # noticed any other way.
        #
        # `_referent_repair_streaks` maps group_id -> CONSECUTIVE episodes whose
        # repair pass moved at least one endpoint. Per group_id because the
        # escalation queue is per-project: a regression in one project's graph
        # must not be masked by another project's clean writes, which is exactly
        # what a single global streak would do (any interleaved clean project
        # would reset it, and the alarm would be unreachable under concurrency).
        #
        # Bounded by the LIVE PROJECT SET — `group_id == project_id`, of which
        # there are a handful — so unlike the per-agent mem0 storm counters
        # above it needs no pruning; there is no unbounded key space to leak.
        self._referent_repair_streaks: dict[str, int] = {}
        # Process-lifetime totals, monotonic and never reset, keyed by a closed
        # five-member vocabulary so a reader never distinguishes "zero" from
        # "absent". Deliberately SEPARATE from the streaks: the streak is a live
        # gauge that resets on a clean pass, these are counters a reader samples
        # and differences (the uptime-baseline convention above).
        self._referent_repair_counts: dict[str, int] = dict.fromkeys(
            ('repaired', 'flagged_unrepairable', 'failed',
             'nodes_minted', 'nodes_deleted'), 0,
        )
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

    async def _verify_episode_referents(
        self, result: Any, *, group_id: str, referents: ReferentSet,
        content: str = '', referent_source: str = 'derived',
    ) -> ReferentStats:
        """Verify each edge hangs off a node this write is actually ABOUT.

        The write-time verification sub-pass (task 3671, PRD leaf zeta). Leaf
        gamma resolves WHICH referents a write is about and leaf epsilon threads
        that set through the durable queue to ``_execute_graphiti_write``; this
        pass closes the loop by walking the committed episode's edges and asking,
        per endpoint, whether the node the edge landed on is one of them.

        DETECTS AND RECORDS ONLY — it performs no writes of any kind. The repair
        (``ensure_entity_node`` -> ``reassign_edge`` -> ``refresh_entity_summary``)
        and the repair-storm streak escalation are leaf ETA's; this pass's output
        is the structured evidence eta acts on. A finding whose correct target
        cannot be determined is RECORDED and LEFT ALONE, never guessed at.

        RUNS LAST in ``_reconcile_episode_identity``, and that ordering is
        load-bearing in one direction: any repair eta performs, and the
        ``new_endpoint_uuid`` this pass resolves, must describe POST-normalization
        topology. Minting a 'Task N' node before ``_normalize_task_node_names``
        ran would create exactly the duplicate that pass exists to collapse.

        The DETECTION verdict, by contrast, is deliberately invariant across that
        normalization, because endpoint names are keyed through
        :func:`~fused_memory.utils.canonical_labels.parse_node_name` rather than
        compared as raw strings: ``parse_node_name('task #3127')`` and
        ``parse_node_name('Task 3127')`` yield the IDENTICAL frozen
        :class:`Referent`. That is what makes the in-memory ``result.nodes`` names
        sufficient even though a rename or merge may have just moved out from
        under them — and it is why this pass costs zero extra backend round-trips
        on the clean path, rather than one ``get_node_text`` per endpoint
        serialized inside the per-group identity lock.

        Runs INSIDE ``_identity_lock_for`` (see ``_execute_graphiti_write``), so
        no wrongly-attached state is ever externally visible between the write
        and its verification.

        THE C' POST-LLM VETO FOLDS IN HERE; it is not a second mechanism, and its
        absence is not an oversight. Post-write, "extracted Task N was merged onto
        the Task M node" is observationally IDENTICAL to "an edge about Task M is
        attached to a Task N node" — which the checks below already detect. A
        separate veto would be two sites that must agree byte-for-byte, the INV-5
        lockstep duplication utils/canonical_labels.py exists to prevent. The PRD
        says so outright: it is "not a distinct leaf".

        SET MEMBERSHIP: an endpoint whose parsed referent is not in *referents* is
        a node this write never declared itself to be about. This catches the
        dominant measured live shape, whose defining signature is that the
        landed-on number is never named by the fact at all.

        PER-EDGE PAIRING: if the edge's own FACT cites at least one task
        referent and the endpoint's referent is not among them, the fact talks
        about ``Task M`` while the edge landed on ``Task N``. This is what
        catches mode (iii), where BOTH numbers are legitimately declared and
        membership therefore cannot fire — the live case being an episode that
        minted 'Task 3074' and 'Task 3075' 60us apart. Resolved decision 7:
        neither check alone is sufficient, so both run.

        The two are ORDERED, not additive. Membership is evaluated first and
        wins the label, so an endpoint failing both is reported exactly ONCE.

        RESOLVABILITY is decided by :func:`_candidate_targets`, whose rule
        reproduces every row of the PRD's boundary-test sketch. Exactly one
        surviving candidate means the correct target is determined; zero or more
        than one means it is not, and the finding is recorded with
        ``resolvable=False`` and a reason rather than dropped or guessed at.

        AMBIGUITY IS RE-DERIVED HERE, NOT READ OFF THE WIRE, and that is by
        epsilon's explicit instruction (``_encode_referents``: "AMBIGUITY IS
        DELIBERATELY NOT THREADED — READ THIS BEFORE WRITING ZETA"). Gamma routes
        a number claimed by BOTH a bare own-project mention and a
        foreign-qualified reference in the same content to
        ``LabelScan.ambiguous`` and EXCLUDES it from ``.referents`` — "recorded,
        not guessed" — and epsilon's two-key blob carries only ``.source`` and
        ``.refs``. A consumer reading the decoded set alone therefore cannot tell
        an AMBIGUOUS endpoint from a genuine conflation: both are simply
        non-members. Since ``.ambiguous`` is
        ``scan_content(content, group_id=group_id).ambiguous`` verbatim on every
        precedence path — a pure function of ``(content, group_id)``, independent
        of source — this pass recovers the producer's exact set from
        ``payload['content']``, which ``_execute_graphiti_write`` already holds.
        An endpoint in that set is still DETECTED and RECORDED (it really is
        outside the declared set), but never made ``resolvable``: the PRD's
        boundary row is "treated as undeclared; recorded, not guessed". The
        veto itself lives in :func:`_candidate_pool` beside its two siblings, so
        all three read at ONE site (INV-5).

        The re-derivation is a SECOND SCAN SITE, which gamma's own comment flags
        as the kind of lockstep duplication canonical_labels exists to prevent.
        Carrying ``'ambiguous'`` as a third wire key is the better long-term
        shape and is epsilon's filed follow-up; it is not done here because it
        would widen a frozen contract every test in
        tests/test_referent_queue_threading.py pins. Scanned ONCE per episode,
        after the edgeless early-out, so the clean path pays for it only when
        there is something to check.

        An EMPTY *referents* makes the whole pass a no-op, honouring the contract
        ``resolve_referents`` publishes in its own docstring ("an EMPTY
        ``.referents`` carries nothing to test membership against, so a downstream
        verifier must no-op on it regardless of ``.source``") and the PRD's
        ``source='none'`` boundary row ("no repair attempted").

        Args:
            result: The value returned by ``add_episode`` (typically an
                AddEpisodeResults object). ``None``, edgeless, and
                missing-attribute results are all handled the same way the
                sibling post-write sweeps handle them.
            group_id: The project graph this episode was written to.
            referents: The referent set leaf epsilon decoded off the queue
                payload — what this write DECLARED itself to be about.
            content: The episode body, threaded from ``payload['content']``, so
                this pass can RE-DERIVE the producer's ambiguity set (see the
                AMBIGUITY paragraph above). Defaults to ``''`` — no content, no
                ambiguity — which is the pre-threading behaviour exactly.
            referent_source: The ``ReferentSource`` leaf epsilon decoded
                alongside *referents*, one of :data:`REFERENT_SOURCES`. Read only
                by :func:`_candidate_pool`, to decide whether the
                whole-declared-set fallback is licensed. Defaults to
                ``'derived'``, the source on which that rule is unchanged.

        Returns:
            A :class:`ReferentStats` recording what was walked and every finding.
            All-zero with no findings means "every checkable endpoint agreed".
        """
        stats = ReferentStats()
        if result is None or not referents:
            return stats
        # The `edges or entity_edges` idiom every sibling post-write sweep in
        # this file uses (2378, 2518, 2621, 2795, 3832, 3871). Reading `edges`
        # alone would make this pass a silent, TOTAL no-op on a result that
        # exposes only `entity_edges` -- a shape the other six still walk, and
        # one this method's own docstring promises parity on.
        edges = (
            getattr(result, 'edges', None)
            or getattr(result, 'entity_edges', None)
            or []
        )
        if not edges:
            return stats

        # The producer's ambiguity set, re-derived from the episode body — see
        # the AMBIGUITY paragraph above for why it is re-derived rather than read
        # off the wire. THROUGH `local_referent`, for the same reason the
        # endpoint parse below is: `scan_content` preserves the qualifier it
        # read, so a self-qualified ambiguous mention ('dark_factory:2500') would
        # otherwise compare unequal to the locally-classified endpoint referent
        # and the veto would silently miss.
        #
        # PERMISSIVE mode (no `known_project_ids`), matching gamma's own choice —
        # the producer scanned in that mode too, and this must recover the
        # producer's set, not a differently-parameterized one.
        ambiguous = frozenset(
            local_referent(ref, group_id=group_id)
            for ref in scan_content(content, group_id=group_id).ambiguous
        ) if content else frozenset()

        # The episode's own node names, which is all the detection needs — see
        # the parse_node_name invariance note above. Same defensive
        # `getattr(..., '') or ''` idiom the sibling sweeps use, since a mocked
        # or partially-populated result must degrade rather than raise inside an
        # already-committed write's critical section.
        names_by_uuid: dict[str, str] = {}
        for node in getattr(result, 'nodes', None) or []:
            node_uuid = getattr(node, 'uuid', '') or ''
            node_name = getattr(node, 'name', '') or ''
            if node_uuid and node_name:
                names_by_uuid[node_uuid] = node_name

        # Membership is tested on Referent OBJECTS (frozen => hashable, equality
        # on the (kind, project_id, number) triple), never on rendered names.
        # `referent_names` is the human-readable rendering carried on the record.
        referent_set = frozenset(referents)
        referent_names = tuple(r.node_name for r in referents)

        for edge in edges:
            stats.edges_scanned += 1
            edge_uuid = getattr(edge, 'uuid', '') or ''
            # Scanned ONCE per edge, not once per endpoint. The FACT is what
            # pairing reads — not the episode content — because the fact is the
            # per-edge assertion whose subject must match the endpoint it landed
            # on; the episode body is about the write as a whole and cannot
            # discriminate between two edges of the same episode.
            #
            # PERMISSIVE mode (no `known_project_ids`), matching the choice
            # gamma made and documented in `resolve_referents`. Threading
            # `self._known_projects` here would fork that decision mid-PRD, and
            # would DROP a foreign reference the fact genuinely makes — turning
            # a true negative into a false pairing finding.
            #
            # `scan.refs` already excludes `scan.ambiguous`, so nothing further
            # is filtered out here: an ambiguous reference is deliberately
            # invisible to this check rather than evidence for it.
            cited = frozenset(
                scan_content(
                    getattr(edge, 'fact', '') or '', group_id=group_id,
                ).refs
            )
            # The referents this edge's fact names that the write also DECLARED
            # itself to be about — i.e. the concrete alternatives this edge
            # could actually belong on. Computed once per edge, beside `cited`,
            # because both endpoints test against it.
            cited_declared = cited & referent_set
            # BOTH ends are resolved before EITHER is checked: the candidate
            # rule needs the OTHER end's referent (a target equal to it would be
            # the self-loop `reassign_edge` refuses), which is only knowable once
            # both names are parsed.
            ends: list[tuple[str, str, str, Referent | None]] = []
            for which_end, attr in (
                ('source', 'source_node_uuid'), ('target', 'target_node_uuid'),
            ):
                endpoint_uuid = getattr(edge, attr, '') or ''
                endpoint_name = names_by_uuid.get(endpoint_uuid, '')
                if not endpoint_name:
                    # This episode's result does not name the node this edge end
                    # points at, so its name is unknown and it cannot be checked.
                    # COUNTED rather than skipped silently: a check that did not
                    # run is not a check that passed.
                    stats.endpoints_unresolved += 1
                # THROUGH `local_referent`, never a bare `parse_node_name`.
                # The parser preserves the qualifier it read, so in group
                # 'dark_factory' an endpoint node named 'dark_factory:3127'
                # parses FOREIGN while `scan_content` and `resolve_referents`
                # both answer the LOCAL 'Task 3127' for that same spelling.
                # Comparing the two directly made a self-qualified endpoint
                # unequal to the very referent the write DECLARED itself to be
                # about: set-membership fired, and the corroboration veto could
                # not save it either, because `endpoint in cited` compares a
                # foreign-qualified endpoint against a locally-classified
                # citation. The self-qualified reclassification is
                # SOURCE-INVARIANT by contract (referent_resolution's module
                # docstring) and this is its fourth consumer.
                ends.append((
                    which_end,
                    endpoint_uuid,
                    endpoint_name,
                    _endpoint_referent(endpoint_name, group_id=group_id),
                ))

            for index, end in enumerate(ends):
                which_end, endpoint_uuid, endpoint_name, endpoint_referent = end
                if endpoint_referent is None:
                    # Unresolvable (counted above), not a task label at all
                    # ('MergeWorker'), or a name that merely MENTIONS one
                    # ('Task 42 orchestrator' — parse_node_name is anchored).
                    continue
                stats.endpoints_checked += 1
                # `1 - index` is the other end of a two-element list.
                other_referent = ends[1 - index][3]
                # ORDERED, NOT ADDITIVE. Membership is evaluated FIRST and
                # wins the label, so an endpoint failing BOTH checks reports the
                # stronger, more specific signal exactly once. Two findings
                # naming the same (edge_uuid, which_end) would hand eta two
                # repair instructions for one edge end that it would have to
                # reconcile before acting, and would double-count in iota's
                # rate. "Both checks run" is a coverage claim, not a licence to
                # emit two findings for one wrong endpoint.
                #
                # `if cited_declared` on the pairing arm is LOAD-BEARING, not a
                # micro-optimization: a fact citing no task number is
                # UNINFORMATIVE about which node its edge belongs on, never
                # contradictory (resolved decision 8; gamma's
                # `_conflicting_referents` choice 4, one level down). A scanner
                # blind spot — bare digits, a reference by title, a hard-wrapped
                # qualified ref — must never manufacture evidence for
                # destructive edge surgery.
                #
                # The guard is the DECLARATION-CORROBORATED citation, not merely
                # a non-empty one, because per-edge pairing is a discriminator
                # AMONG DECLARED REFERENTS (resolved decision 7; the PRD's mode
                # (iii) row repairs to `Task 3075`, itself declared) — not a
                # general "does the fact mention some other number" test. Two
                # consequences make that fail-closed. An endpoint reaching this
                # arm is itself declared (membership passed), so it ALREADY
                # satisfies the PRD's first postcondition; and a citation
                # outside the declared set can never become a target anyway
                # (`_candidate_pool`'s intersection rule forbids it). Firing
                # there would emit a finding unactionable BY CONSTRUCTION —
                # polluting the rate leaf iota samples and raising an operator
                # WARNING for an endpoint with no observable defect. Same
                # rationale as the empty-scan guard it subsumes (non-empty
                # `cited_declared` implies non-empty `cited`), one step out.
                #
                # Because this arm is only reached when the endpoint's referent
                # IS declared and is NOT cited, a non-empty `cited_declared`
                # necessarily names a DIFFERENT declared referent — exactly the
                # mode (iii) shape. It also means the pool on this arm is
                # `cited & referents`, which never contains the endpoint, so
                # `_candidate_targets`' endpoint subtraction is now unreachable
                # from EITHER arm. It is retained deliberately as a structural
                # invariant at the single site that decides targets, so a future
                # third check cannot reintroduce a self-targeting repair.
                #
                # The MEMBERSHIP arm carries the SAME fail-closed principle, but
                # one layer down, in `_candidate_pool`: a fact that cites the
                # endpoint it landed on corroborates that attachment, so it never
                # falls back to the declared set for a target. That guard lives
                # with the evidence rule rather than here so both arms read ONE
                # site (INV-5) — see `_candidate_pool` for why the dominant
                # `source='metadata'` write shape depends on it.
                if endpoint_referent not in referent_set:
                    check = 'set-membership'
                elif cited_declared and endpoint_referent not in cited:
                    check = 'per-edge-pairing'
                else:
                    continue

                candidates = _candidate_targets(
                    referents=referent_set,
                    cited=cited,
                    endpoint=endpoint_referent,
                    other_endpoint=other_referent,
                    ambiguous=ambiguous,
                    source=referent_source,
                )
                resolvable = len(candidates) == 1
                stats.findings.append(ReferentFinding(
                    edge_uuid=edge_uuid,
                    which_end=which_end,
                    check=check,
                    old_endpoint_uuid=endpoint_uuid,
                    old_endpoint_name=endpoint_name,
                    endpoint_referent=endpoint_referent,
                    referent_set=referent_names,
                    intended_referent=candidates[0] if resolvable else None,
                    resolvable=resolvable,
                    reason='' if resolvable else _unresolvable_reason(
                        candidates,
                        pool=_candidate_pool(
                            referents=referent_set, cited=cited,
                            endpoint=endpoint_referent,
                            ambiguous=ambiguous, source=referent_source,
                        ),
                        cited=cited,
                        endpoint=endpoint_referent,
                        other_endpoint=other_referent,
                        ambiguous=ambiguous,
                        source=referent_source,
                    ),
                ))

        # SECOND PASS, deliberately: the lookup below is deferred to the
        # findings alone so the ~99.8% clean path issues ZERO extra queries
        # inside the per-group identity lock. One query per DISTINCT intended
        # referent, cached for this call.
        uuid_by_name: dict[str, str | None] = {}
        for index, finding in enumerate(stats.findings):
            if finding.intended_referent is None:
                continue
            name = finding.intended_referent.node_name
            if name not in uuid_by_name:
                uuid_by_name[name] = await self._intended_endpoint_uuid(
                    name, group_id=group_id,
                )
            # `dataclasses.replace`, never mutation: the record is frozen
            # because it is evidence for destructive edge surgery.
            stats.findings[index] = dataclasses.replace(
                finding, new_endpoint_uuid=uuid_by_name[name],
            )

        for finding in stats.findings:
            # The two INV-2 surfaces no consumer has to parse a log for: the
            # process-lifetime counter leaf iota reads, and the return value
            # leaf eta reads in-process inside this same critical section.
            self._referent_finding_counts[finding.check] += 1
            if not finding.resolvable:
                self._referent_finding_counts['unresolvable'] += 1
            # WARNING, not DEBUG. The task calls out today's `logger.debug`-only
            # ReconcileStats shape as unacceptable here, and a misattached edge
            # is a correctness defect an operator should see. This line is the
            # OPERATOR surface ONLY — it carries the structured payload for
            # legibility, but nothing parses it.
            logger.warning(
                'Referent verification finding: %s', finding.to_dict(),
            )

        return stats

    async def _intended_endpoint_uuid(
        self, name: str, *, group_id: str
    ) -> str | None:
        """The uuid of the node *name* denotes, or ``None`` — never a write.

        ``get_nodes_by_exact_name`` SPECIFICALLY, because it is documented
        ``ro_query``-only and never raises on zero-or-many. zeta detects and must
        not write, which rules out ``ensure_entity_node`` (MINTS) and
        ``_resolve_or_create_entity`` (COLLAPSES) despite both being available
        and both being what leaf eta will use.

        ``len(rows) != 1`` yields ``None``, collapsing ABSENT and
        DUPLICATE-NAME-GROUP to the same answer on purpose: the PRD measured 38
        live name keys carrying more than one node, and zeta picking a survivor
        from such a group would pre-empt the identity-lock-held collapse that is
        ``_resolve_or_create_entity``'s job — while eta's ``ensure_entity_node``
        handles both cases identically anyway. ``None`` therefore means "eta
        resolves-or-mints", NEVER "unrepairable"; that is
        :attr:`ReferentFinding.resolvable`.

        Best-effort, mirroring ``_normalize_task_node_names``: a transient
        backend error degrades the uuid to ``None`` rather than losing the
        finding. Detection is the primary result and the uuid is an audit
        convenience, so a lookup failure must not cost the evidence.
        """
        try:
            rows = await self.graphiti.get_nodes_by_exact_name(
                name, group_id=group_id,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logger.warning(
                'Failed to resolve intended referent %r to a node uuid during '
                'referent verification; recording the finding without one',
                name, exc_info=True,
            )
            return None
        return rows[0]['uuid'] if len(rows) == 1 else None

    async def _repair_episode_referents(
        self, stats: ReferentStats, *, group_id: str, episode_uuid: str = ''
    ) -> ReferentRepairStats:
        """REPAIR the edge endpoints leaf zeta found on the wrong node.

        The write-time repair sub-pass (task 3672, PRD leaf eta). zeta
        (``_verify_episode_referents``) detects and records; this pass is the
        only WRITER, and it acts on nothing but zeta's structured findings — it
        re-derives no verdict of its own. Per resolvable finding:

        1. ``ensure_entity_node(intended.node_name)`` — resolve-or-mint the
           node the edge should hang off.
        2. ``reassign_edge(edge_uuid, <that uuid>, which_end=...)`` — move the
           one endpoint, losslessly and atomically.
        3. ``refresh_entity_summary`` as a BACKSTOP over step 2's own internal
           refresh (see the loop body).

        ALPHA LOCK CONTRACT. ``ensure_entity_node``'s docstring states that
        callers MUST hold ``_identity_lock_for(group_id)`` — it performs no
        locking of its own, and a concurrent same-group writer between its
        resolve and its mint produces exactly the duplicate-name pair the
        identity gate exists to prevent. This pass satisfies that by PLACEMENT:
        it runs inside ``_reconcile_episode_identity``, whose whole chain
        ``_execute_graphiti_write`` wraps in ``async with
        self.graphiti._identity_lock_for(payload['group_id'])``. That placement
        is load-bearing, not incidental — moving this pass out of the chain, or
        calling it from anywhere that does not already hold the lock, would
        reintroduce the race alpha's contract forbids.

        ``ensure_entity_node`` IS CALLED UNCONDITIONALLY, for every resolvable
        finding, and ITS RETURN — never ``finding.new_endpoint_uuid`` — is the
        uuid handed to ``reassign_edge``. Three reasons:

        * It is IDEMPOTENT: once the node exists, every later call takes the
          resolve path and mints nothing. A branch on ``new_endpoint_uuid is
          None`` would buy no round-trips worth having and would create a
          SECOND site that can disagree about what the edge should point at.
        * zeta returns ``None`` for BOTH "absent" and "duplicate-name group"
          (``_intended_endpoint_uuid``: ``len(rows) != 1`` yields ``None``, so
          zeta never pre-empts the identity-lock-held collapse), and
          ``ensure_entity_node`` handles the two identically — it
          resolves-or-collapses-or-mints through ``_resolve_or_create_entity``.
          Branching would have to re-derive the distinction zeta explicitly
          declined to make.
        * It re-reads from the graph under the lock, so the target is
          corroborated at WRITE time rather than taken from a lookup made a few
          statements earlier. zeta's ``new_endpoint_uuid`` is demoted to what
          its own docstring already calls it: an audit convenience.

        INV-3 CORROBORATE-BEFORE-ACTING is preserved by DELEGATION, not by a
        second check here. ``reassign_edge`` re-reads BOTH endpoints from
        topology (a directed MATCH) and returns ``moved=False`` without issuing
        any write when the end already points at the target — so its report
        outranks zeta's in-memory snapshot, and the record's
        ``old_endpoint_uuid`` is read back from it rather than copied from the
        finding. This pass must therefore never pre-read or second-guess an
        endpoint itself: a second read would be a second answer that can
        disagree with the one the write actually used.

        WHY THE PASS NEEDS THE EPISODE'S IDENTITY. The emptied-node cleanup's
        guard claims to distinguish "the node was MINTED by this very episode
        out of the mis-resolved reference alone" from "the project GENUINELY
        OWNS the task". Until it was given ``episode_uuid`` it had no datum
        capable of making that distinction — it inferred mintedness from
        valid-edge emptiness, which is a different and much weaker proposition
        (see :meth:`_cleanup_emptied_nodes`, condition 4). The identifier is
        threaded straight down to that guard and used nowhere else.

        ``episode_uuid`` is KEYWORD-ONLY and DEFAULTED so the parameter is
        purely additive: existing direct callers are unaffected, and anything
        that forgets to pass it degrades to the STRICT, more-conservative
        predicate rather than a looser one. ``''`` does not mean "no exclusion
        is needed"; it means "we could not establish which episode this is, so
        exclude nothing and refuse more deletions" — the fail-closed direction,
        consistent with :attr:`ReferentFinding.resolvable` defaulting to
        ``False``. See :func:`_episode_uuid_of` for how the chain extracts it.

        Args:
            stats: zeta's ``ReferentStats`` for this episode, consumed verbatim.
            group_id: The project graph, which is also the project_id.
            episode_uuid: UUID of the episode whose write is in flight, used
                ONLY by the emptied-node cleanup's fourth condition. ``''``
                (the default) is the fail-closed value.

        Returns:
            A ``ReferentRepairStats`` carrying one record per finding —
            including the ones deliberately left alone.
        """
        repair_stats = ReferentRepairStats()

        # Grouped by edge FIRST, so the degenerate predicate is evaluated once
        # per edge BEFORE any backend call for that edge. dict preserves
        # insertion order, so findings are still processed in zeta's order.
        findings_by_edge: dict[str, list[ReferentFinding]] = {}
        for finding in stats.findings:
            findings_by_edge.setdefault(finding.edge_uuid, []).append(finding)

        # uuid -> the name this episode's result reported for that node, the
        # only thing the cleanup's canonical-label guard has to parse. Keyed by
        # the FINDING's uuid: when `reassign_edge`'s topology re-read disagrees
        # with it, we have no name for the node that actually lost the edge,
        # and no name means no deletion (fail closed).
        endpoint_names: dict[str, str] = {
            f.old_endpoint_uuid: f.old_endpoint_name for f in stats.findings
        }

        for edge_uuid, edge_findings in findings_by_edge.items():
            if self._is_degenerate_edge(edge_findings):
                # Skip the edge WHOLE — never half-move it. One warning naming
                # the edge and BOTH endpoint node uuids, because "which edge,
                # which nodes" is the whole of what a human needs to decide
                # this case by hand (NEVER GUESS: recorded, left alone).
                logger.warning(
                    'Referent repair: skipping degenerate edge %s WHOLE — both '
                    'ends would sit on one node (endpoints %s); repairing '
                    'either end alone would leave the edge half-attributed, '
                    'and repairing both would fold it into a self-referential '
                    'RELATES_TO. Recorded and left alone: %s',
                    edge_uuid,
                    sorted({f.old_endpoint_uuid for f in edge_findings}),
                    [f.to_dict() for f in edge_findings],
                )
                for finding in edge_findings:
                    repair_stats.repairs.append(ReferentRepair(
                        edge_uuid=finding.edge_uuid,
                        which_end=finding.which_end,
                        outcome='degenerate',
                        old_endpoint_uuid=finding.old_endpoint_uuid,
                        check=finding.check,
                        reason=(
                            'both ends of this edge would sit on one node; '
                            'skipped whole rather than half-moved'
                        ),
                    ))
                continue

            await self._repair_edge_findings(
                edge_findings, repair_stats, group_id=group_id,
            )

        await self._cleanup_emptied_nodes(
            repair_stats, endpoint_names,
            group_id=group_id, episode_uuid=episode_uuid,
        )

        streak = self._record_referent_repair_pass(
            stats, repair_stats, group_id=group_id,
        )
        if streak is not None and streak >= _REFERENT_REPAIR_STREAK_THRESHOLD:
            await self._escalate_referent_repair_storm(
                repair_stats, group_id=group_id, streak=streak,
            )

        return repair_stats

    def _record_referent_repair_pass(
        self,
        stats: ReferentStats,
        repair_stats: ReferentRepairStats,
        *,
        group_id: str,
    ) -> int | None:
        """Fold one completed repair pass into the INV-4 escape hatch.

        Two independent things, deliberately kept apart:

        THE PROCESS-LIFETIME TOTALS accumulate unconditionally — monotonic
        counters leaf iota samples and differences.

        THE PER-GROUP STREAK counts CONSECUTIVE EPISODES that needed a repair,
        and its update rule is a three-way branch, not a two-way one:

        * ``repaired >= 1`` -> ``+1``, by exactly one however many endpoints
          moved. The streak's unit is the EPISODE; the per-episode repair count
          rides separately in :attr:`ReferentRepairStats.repaired` and is
          carried into the escalation's evidence. Making the increment
          proportional would let a single wide episode breach the threshold on
          its own, which is precisely the "one bad write" case this alarm is
          not for.
        * zero repairs but ``endpoints_checked > 0`` -> ``0``. THE POSITIVE
          HEALTH SIGNAL: we looked at this episode's edge endpoints and none
          needed moving.
        * zero repairs and ``endpoints_checked == 0`` -> UNCHANGED, and the key
          is not even created.

        THE ASYMMETRY IS THE POINT, and it is copied from
        ``MergeWorker._record_runner_recovered``, which pops a host's
        unavailability entry only on a SUCCESSFUL probe — never on the mere
        absence of a failure. Here the analogue of "no failure observed" is an
        episode that declared no referents, produced no edges, or yielded no
        findings: the pass ran, looked at nothing, and returned. That is no
        evidence of health, and clearing the streak on it would make the alarm
        structurally UNREACHABLE — the measured base rate is ~0.22% of live
        task-mentioning edges, so the overwhelming majority of episodes check
        nothing, and any one of them interleaved between two repairs would zero
        a streak that can then never reach ten.

        Only ``repaired`` (``outcome == 'repaired' and moved``) counts toward
        the streak. A ``moved=False`` result is ``reassign_edge``'s
        corroborate-before-acting no-op — the edge was ALREADY correct — and
        ``'unrepairable'``/``'degenerate'``/``'failed'`` are a refusal, a
        refusal, and an infrastructure fault respectively. None of the four is
        evidence that the resolver is producing wrong endpoints, and counting
        any of them would let a FalkorDB outage page as a scanner regression.

        Returns:
            The group's streak AFTER the increment, when this pass incremented
            it — i.e. the reading the storm gate thresholds. ``None`` on the
            other two arms. Returning the value rather than having the caller
            re-read the dict is what keeps the gate from firing on a pass that
            merely LEFT a breached streak in place: an episode that checked
            nothing performs no repair, so it must not re-page an alarm it
            produced no evidence for.
        """
        self._referent_repair_counts['repaired'] += repair_stats.repaired
        self._referent_repair_counts['flagged_unrepairable'] += (
            repair_stats.flagged_unrepairable
        )
        self._referent_repair_counts['failed'] += repair_stats.failed
        self._referent_repair_counts['nodes_minted'] += repair_stats.nodes_minted
        self._referent_repair_counts['nodes_deleted'] += repair_stats.nodes_deleted

        if repair_stats.repaired:
            streak = self._referent_repair_streaks.get(group_id, 0) + 1
            self._referent_repair_streaks[group_id] = streak
            return streak
        if stats.endpoints_checked:
            self._referent_repair_streaks[group_id] = 0
        return None

    async def _escalate_referent_repair_storm(
        self,
        repair_stats: ReferentRepairStats,
        *,
        group_id: str,
        streak: int,
    ) -> None:
        """Fire the INV-4 repair-storm alarm for *group_id*.

        THE PREDICATE IS ``streak >= threshold``, NOT ``==``, and the caller
        spells it inline. Every subsequent breach re-enters here; collapsing a
        sustained storm to ONE operator entry is the ESCALATOR's job, via its
        pending-anchor dedupe fold. That is the same division of labour
        ``merge_liveness`` uses — ``_record_runner_unavailable`` returns
        ``streak >= threshold`` on every episode and the filing side folds — and
        it is the robust arrangement: a counter that fired only on the exact
        boundary would go permanently silent if one breach were ever missed
        (a filing failure, a restart, a config change to the threshold), and
        nothing downstream would notice the alarm had been disarmed.

        ``orchestrator.critical_gate.critical_filing_gate`` is the canonical
        spelling of this predicate and is deliberately NOT imported: fused-memory
        depends on ``dark-factory-shared`` only, and the sole orchestrator
        imports anywhere under ``fused_memory/`` are lazy, optional and confined
        to ``reconciliation/sandbox_guard.py``. Taking a hard dependency on the
        orchestrator package to reuse a one-line comparison would invert that
        layering for no gain. INV-5's no-lockstep-duplication concern does not
        bite here either, because ``>=`` is not a VOCABULARY — there is nothing
        two sites must agree byte-for-byte about, unlike
        :data:`REFERENT_REPAIR_OUTCOMES` above, which is exactly why that one
        does live at a single normative site.

        ``asyncio.to_thread`` IS LOAD-BEARING — do not re-inline the escalation
        hop. ``EscalationQueue`` construction, its queue-directory scan and its
        fsync-flushed write are blocking filesystem I/O; called directly from
        this coroutine they would run ON the event loop and stall every other
        concurrent memory write for their duration. That warning and its
        precedent are already recorded on ``_check_memory_metadata`` in this
        file ("ASYNC ON PURPOSE — do not re-inline the escalation hop"), and the
        ported module (``middleware/candidate_key_escalation``) carries its
        never-raise contract over but not its synchronous-caller assumption.
        It is AWAITED rather than fire-and-forgotten so the call can never
        outlive the write or be dropped by task GC.

        The per-group IDENTITY LOCK IS STILL HELD across that hop, which is
        accepted rather than overlooked. It is tolerable because this path runs
        only at ``streak >= 10`` — an already-anomalous condition, not the
        steady state — and because it folds to a single ``get_by_task`` read
        the moment one escalation is open, so a sustained storm does not
        repeatedly pay for a full filing.

        PROJECT ROOT resolution is ``self._known_projects[group_id]`` and has NO
        FALLBACK. graphiti's ``group_id`` IS the ``project_id`` (models/scope.py),
        so that map is the correct and only answer. Falling back to
        ``config.taskmaster.project_root`` is explicitly forbidden by the mem0
        storm escalator's docstring and for the reason it gives: that field
        defaults to ``'.'``, so the fallback files into the SERVER'S CWD, where
        no operator is watching, and reports success doing it — a silent
        misfile is strictly worse than a logged refusal, because it also
        destroys the evidence that the alarm ever fired.

        NEVER RAISES a generic exception. The repairs this is complaining about
        have already committed to the graph, and this runs inside the reconcile
        chain of an episode whose write is already durable — failing that chain
        because the COMPLAINT about it failed would turn a heads-up into an
        outage. Belt to the escalator's own never-raise braces, matching the
        guard discipline every best-effort hop in this file uses;
        ``CancelledError`` / ``KeyboardInterrupt`` / ``SystemExit`` still
        propagate.
        """
        project_root = self._known_projects.get(group_id)
        if not project_root:
            # A REFUSAL, never a guess. Structured so the operator who finds
            # this line has everything the escalation would have carried.
            logger.warning(
                'referent repair storm in group_id=%r (streak=%d, threshold=%d, '
                '%d repair(s) this episode) could NOT be escalated: the group '
                'is absent from `_known_projects`, so no project queue can be '
                'resolved. Not falling back to config.taskmaster.project_root '
                '(it defaults to the server cwd, where nobody is watching). '
                'Repairs continue. Records: %s',
                group_id, streak, _REFERENT_REPAIR_STREAK_THRESHOLD,
                repair_stats.repaired,
                [r.to_dict() for r in repair_stats.repairs],
            )
            return

        try:
            await asyncio.to_thread(
                emit_referent_repair_storm_escalation,
                project_root,
                project_id=group_id,
                streak=streak,
                threshold=_REFERENT_REPAIR_STREAK_THRESHOLD,
                repairs=repair_stats.repaired,
                records=[r.to_dict() for r in repair_stats.repairs],
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logger.warning(
                'referent repair storm alarm for group_id=%r (streak=%d) could '
                'not be filed; the repairs themselves already committed and are '
                'unaffected',
                group_id, streak, exc_info=True,
            )

    async def _cleanup_emptied_nodes(
        self,
        repair_stats: ReferentRepairStats,
        endpoint_names: Mapping[str, str],
        *,
        group_id: str,
        episode_uuid: str = '',
    ) -> None:
        """Delete the nodes THIS PASS emptied — under four conditions, all required.

        Runs AFTER every reassignment for the episode, because a node is only
        empty once every edge that was going to leave it has left.

        THE TWO SITUATIONS THE NODE CAN BE IN, and why the guard is this narrow:

        * The project GENUINELY OWNS the task. The node keeps its pre-existing
          edges after the repair, so the emptiness check fails and it is never
          touched. Deleting a real task entity to fix an attribution error
          would be a far more damaging bug than the one being fixed.
        * The node was MINTED by this very episode out of the mis-resolved
          reference alone. The repair leaves it edgeless and semantically
          wrong, and it would otherwise persist as a phantom entity that future
          writes keep re-colliding with — leaving the fix only half-working
          and, per the PRD's "coupling" section, keeping the duplicate-name key
          that disables ``dedup_helpers``' deterministic exact-match protection.

        THE FOUR CONDITIONS, in the order they are evaluated — the two free
        local checks first, then the cheap typed query, then the new untyped
        degree query, so the ~99.8% clean path pays nothing extra inside the
        identity lock:

        1. This pass moved at least one endpoint OFF the node — an
           ``outcome='repaired'`` record with ``moved=True``. A ``moved=False``
           no-op emptied nothing, so it is not even a candidate and neither
           query is issued for it. The node that GAINED the edge is never a
           candidate either; deleting the repair target would undo the repair.
        2. Its name parses as a canonical task label —
           :func:`~fused_memory.utils.canonical_labels.parse_node_name`, the
           single normative label vocabulary (PRD leaf beta), reused here
           rather than respelled as a regex at the delete site. That function
           is ANCHORED by design, so a node merely MENTIONING a task
           ('Task 42 orchestrator') returns ``None`` and survives. This is the
           guard where a drifting pattern would delete a real entity, which is
           exactly the duplication INV-5 exists to prevent.
        3. ``get_valid_edges_for_node`` returns empty.
        4. :meth:`GraphitiBackend.count_foreign_relationships` returns 0 — the
           node has NO relationship of ANY type or validity other than THIS
           episode's own ``MENTIONS`` link.

        WHY CONDITION 4 EXISTS, AND WHAT CONDITION 3 ALONE COULD NOT
        DISTINGUISH. Conditions 1-3 do not test the proposition this guard's
        stated intent claims, and the gap is a confirmed data-loss defect, not
        a theoretical one:

        * ``get_valid_edges_for_node`` is ``MATCH (n:Entity {uuid:$uuid})
          -[e:RELATES_TO]-() WHERE e.invalid_at IS NULL``. It sees ONLY
          currently-valid ``RELATES_TO``, and is blind both to INVALIDATED
          ``RELATES_TO`` history and to ``MENTIONS`` from Episodic nodes.
        * ``delete_entity_node`` issues a bare ``MATCH (n:Entity {uuid:$uuid})
          DETACH DELETE n``, destroying EVERY relationship — all of the above
          included.
        * ``parse_node_name`` cannot help, because a genuine ``Task N`` node
          passes it BY CONSTRUCTION: it is exactly a canonical task label.

        The exposure is reachable inside ONE critical section.
        :meth:`_invalidate_stale_superseded_ttl_edges` is sub-pass FOUR of the
        same ``_reconcile_episode_identity`` chain and exists precisely to
        invalidate same-subject superseded edges; this cleanup is sub-pass
        EIGHT. So a real, project-owned ``Task N`` node whose facts were
        TTL-invalidated at pass 4 reads as "empty" at pass 8 the moment this
        pass moves its one remaining valid edge away — and would be
        irreversibly deleted with its full temporal history.

        WHAT ``force=False`` DOES AND DOES NOT SUPPLY. The delete goes through
        :meth:`GraphitiBackend.delete_entity` with ``force=False``, never the
        lower-level ``delete_entity_node``, and that argument still guards a
        genuine RACE on VALID edges — a node that gains a live edge between our
        read and the delete — turning the lost race into an
        ``ActiveEdgesError`` refusal. That is worth keeping.

        It is NOT, however, "a second, INDEPENDENT check" of the emptiness this
        cleanup is predicated on, as this docstring previously claimed:
        ``delete_entity`` re-checks by calling ``get_valid_edges_for_node`` —
        the SAME query — so it is blind to exactly what condition 4 exists to
        see. A same-query recheck cannot see what the first query missed.
        Condition 4 is the only thing standing between this cleanup and
        destroying a real entity's temporal history.

        ``delete_entity_node`` issues a bare DETACH DELETE with no edge guard
        and no neighbour refresh, so using it would mean re-implementing both.

        CONDITION 4 IS THE CONSERVATIVE DIRECTION BY CONSTRUCTION. Its two
        imprecisions both only ever REFUSE a delete: an undirected self-loop
        double-counts, and an unresolved ``episode_uuid`` of ``''`` drops the
        exclusion entirely (see :func:`_episode_uuid_of`). If it ever fires
        more often than expected, the correct response is to INVESTIGATE THE
        NODE — something is attached to it that nobody expected — never to
        relax the predicate.

        The stamp back onto the matching records uses
        :func:`dataclasses.replace`, never mutation — the record is frozen
        because it is evidence for destructive edge surgery.
        """
        candidates: list[str] = []
        for record in repair_stats.repairs:
            if record.outcome != 'repaired' or not record.moved:
                continue
            uuid = record.old_endpoint_uuid
            if uuid in candidates:
                continue
            name = endpoint_names.get(uuid, '')
            if not name or parse_node_name(name) is None:
                continue
            candidates.append(uuid)

        for uuid in candidates:
            try:
                remaining = await self.graphiti.get_valid_edges_for_node(
                    uuid, group_id=group_id,
                )
                if remaining:
                    continue
                # CONDITION 4, and the only thing standing between this cleanup
                # and destroying a real entity's temporal history. Runs LAST
                # because it is the most expensive: an untyped degree query,
                # issued only for candidates that already passed the two free
                # local checks and the cheap typed one.
                foreign = await self.graphiti.count_foreign_relationships(
                    uuid, group_id=group_id, episode_uuid=episode_uuid,
                )
                if foreign:
                    logger.info(
                        'Referent repair: not deleting emptied node %s — it '
                        'still has %d relationship(s) that a DETACH DELETE '
                        'would destroy (invalidated RELATES_TO history, or '
                        'MENTIONS from an episode other than %r). The endpoint '
                        'move stands; the node does too',
                        uuid, foreign, episode_uuid,
                    )
                    continue
                await self.graphiti.delete_entity(
                    uuid, group_id=group_id, force=False,
                )
            except ActiveEdgesError:
                # A LOST RACE, not a failure: the node gained an edge between
                # our read and the delete, and force=False refused. Exactly
                # what that argument is for. INFO, because the outcome is
                # correct — the node is still there and still has an edge.
                logger.info(
                    'Referent repair: not deleting emptied node %s — it '
                    'gained a valid edge between the emptiness check and the '
                    'delete, and delete_entity(force=False) refused',
                    uuid,
                )
                continue
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                # PER-CANDIDATE, and guarded SEPARATELY from the reassignments
                # above rather than sharing the outer `_run_pass`. `_run_pass`
                # substitutes an EMPTY ReferentRepairStats on a raise, which
                # would discard the structured record of every reassignment
                # that had ALREADY COMMITTED to the graph — reporting zero
                # repairs for an episode that performed several. That is the
                # exact silent-degradation shape INV-2 and the
                # no-silent-fail-soft invariant rule out. Cleanup is
                # opportunistic hygiene; the reassignment is the correctness
                # fix, and their failure domains must not be shared.
                #
                # Per-candidate so one node whose emptiness cannot be read does
                # not strand every other phantom node behind it.
                logger.warning(
                    'Referent repair: emptied-node cleanup failed for node %s; '
                    'the endpoint move(s) that emptied it already committed '
                    'and stand, but the node may remain as a phantom entity',
                    uuid, exc_info=True,
                )
                continue
            logger.info(
                'Referent repair: deleted emptied node %s (%r) — this pass '
                'moved its last edge off it',
                uuid, endpoint_names.get(uuid, ''),
            )
            for index, record in enumerate(repair_stats.repairs):
                if (
                    record.outcome == 'repaired' and record.moved
                    and record.old_endpoint_uuid == uuid
                ):
                    repair_stats.repairs[index] = dataclasses.replace(
                        record, deleted_emptied_node=uuid,
                    )

    @staticmethod
    def _is_degenerate_edge(findings: list[ReferentFinding]) -> bool:
        """Would repairing this edge's findings fold it into a self-loop?

        ONE predicate, evaluated once per edge BEFORE any backend call for that
        edge — deliberately NOT an ``except ValueError`` around
        ``reassign_edge``. That primitive does refuse a move whose new endpoint
        equals the endpoint left in place, but by the time it raises, the FIRST
        end has already committed: the half-attributed edge the rule exists to
        prevent has already happened, and the exception arrives too late to be
        a guard. Deciding before any write is the only formulation that can
        "skip the edge whole".

        Two shapes, both reachable, both producing the identical outcome:

        LITERAL — two findings share an ``old_endpoint_uuid``. Both ends
        already sit on the node being repaired away from, i.e. a self-loop on
        the wrong node. Not hypothetical:
        :meth:`GraphitiBackend.get_valid_edges_for_node`'s own docstring
        documents that an A->A ``RELATES_TO`` edge exists in this graph and
        double-matches its undirected query.

        PROJECTED — two RESOLVABLE findings share an ``intended_referent``. The
        two moves would converge both ends onto one node. Equally reachable
        through zeta's rules: ``_candidate_targets`` subtracts ``endpoint`` and
        ``other_endpoint``, neither of which removes a THIRD referent that both
        ends would legitimately move onto. This is the SAME predicate evaluated
        on post-repair endpoints, which is why it lives here as one extra
        comparison rather than as a second guard that could drift.

        Comparing the intended ``node_name`` rather than a resolved uuid is
        equivalent and available BEFORE the mint: ``ensure_entity_node`` keys
        on the NAME, so two findings converge on one node exactly when their
        referents render the same ``node_name``.

        THE COMPARISON IS ON THE RENDERED NAME, NOT ON THE ``Referent``.
        Those are not the same test. :attr:`Referent.node_name` returns
        ``f'{project_id}:{number}'` whenever ``project_id`` is non-empty,
        IGNORING ``kind`` entirely — so two ``Referent``s differing only in
        ``kind`` render the SAME node name while comparing unequal as frozen
        dataclasses. Comparing the objects would therefore miss a convergence
        the mint will then actually perform. It happens to be unobservable
        today only because ``canonical_labels._KIND_LABELS`` has exactly one
        entry, and this is a pre-write safety gate for destructive edge
        surgery — the post-hoc ``reassign_edge`` ValueError arrives too late to
        substitute for it — so it must not silently depend on that.

        Resolvable findings only, on the projected arm: a finding eta will
        never act on cannot converge with anything.
        """
        if len(findings) < 2:
            return False
        endpoints = [f.old_endpoint_uuid for f in findings]
        if len(set(endpoints)) < len(endpoints):
            return True
        targets = [
            f.intended_referent.node_name for f in findings
            if f.resolvable and f.intended_referent is not None
        ]
        return len(set(targets)) < len(targets)

    async def _repair_edge_findings(
        self,
        findings: list[ReferentFinding],
        repair_stats: ReferentRepairStats,
        *,
        group_id: str,
    ) -> None:
        """Run the repair sequence for one non-degenerate edge's findings.

        Split out from :meth:`_repair_episode_referents` so the pre-write
        degenerate predicate reads as a gate on the whole edge rather than as
        another branch inside the per-finding loop.

        TWO GUARDS PER FINDING, NOT ONE, split at the commit point. The first
        wraps ``ensure_entity_node`` + ``reassign_edge`` — everything that can
        fail with NOTHING written — and its ``except`` records ``'failed'``.
        The second wraps only the post-write summary backstop, whose failures
        cannot un-write the move that already landed and therefore must not be
        able to book it as ``'failed'``. Sharing one ``except`` across the
        commit point is what would let a cosmetic post-write problem report an
        episode's real repair as an infrastructure fault that did nothing.
        """
        for finding in findings:
            intended = finding.intended_referent
            if not finding.resolvable or intended is None:
                # NEVER GUESS, as the STRUCTURAL default rather than a branch
                # this pass chose to add. `ReferentFinding.resolvable` already
                # DEFAULTS to False — zeta made fail-closed structural, so a
                # finding is unrepairable unless something positively
                # determined a target — and this arm is what honours it.
                #
                # The PRD's boundary row, verbatim: "Unary fact, no correct
                # target: flagged, recorded, left unrepaired". Its live case is
                # the fact "Umbrella task 2519 was filed and then cancelled to
                # avoid orphaning its vector" sitting on the `Task 2520` node —
                # the fact names exactly ONE task and it is not the one the
                # edge landed on, so there is no second candidate to move it
                # to and any target eta picked would be invented.
                #
                # The RECORD EXISTING AT ALL is the point. A dropped finding
                # and a repaired one are indistinguishable to leaf iota's rate
                # and to an operator reading the audit; an 'unrepairable'
                # record carrying zeta's own `reason` is the evidence a human
                # uses to decide the case by hand. The reason is copied
                # VERBATIM rather than paraphrased — a paraphrase is a second
                # site that can drift from the rule that actually fired.
                #
                # `intended is None` is folded in here rather than crashed on:
                # zeta forbids that shape (a resolvable finding always names a
                # referent) but the TYPE permits it, and fail-closed means an
                # impossible shape becomes a refusal, never a guess.
                repair_stats.repairs.append(ReferentRepair(
                    edge_uuid=finding.edge_uuid,
                    which_end=finding.which_end,
                    outcome='unrepairable',
                    old_endpoint_uuid=finding.old_endpoint_uuid,
                    check=finding.check,
                    reason=finding.reason,
                ))
                continue

            # PER-FINDING containment, not per-pass: each finding names a
            # distinct (edge, end), so one failure carries NO information about
            # the others, and aborting the batch would leave the graph in a
            # state neither zeta's findings nor eta's records describe. A batch
            # of independent edge repairs must be able to make partial
            # progress.
            #
            # `'failed'` is a THIRD disposition, distinct from BOTH
            # `'repaired'` and `'unrepairable'`: unrepairable means we REFUSED
            # TO GUESS, failed means we TRIED and the backend did not
            # cooperate. Conflating them would let a FalkorDB outage read as a
            # scanner regression in leaf iota's rate, and would feed a false
            # repair-storm streak whose whole claim is that the scanner or the
            # resolver has REGRESSED.
            try:
                target_uuid = await self.graphiti.ensure_entity_node(
                    intended.node_name, group_id=group_id,
                )
                result = await self.graphiti.reassign_edge(
                    finding.edge_uuid, target_uuid,
                    which_end=finding.which_end, group_id=group_id,
                )
                moved = bool(result.get('moved'))
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                logger.warning(
                    'Referent repair FAILED for one edge end; it is left '
                    'unrepaired and recorded as such (%s): %s',
                    exc, finding.to_dict(), exc_info=True,
                )
                repair_stats.repairs.append(ReferentRepair(
                    edge_uuid=finding.edge_uuid,
                    which_end=finding.which_end,
                    outcome='failed',
                    old_endpoint_uuid=finding.old_endpoint_uuid,
                    check=finding.check,
                    intended_referent=intended.node_name,
                    reason=f'{type(exc).__name__}: {exc}',
                ))
                continue

            # STEP 3 SITS OUTSIDE THE WRITE GUARD, in its own, deliberately.
            # By here the endpoint move has ALREADY COMMITTED to the graph, so
            # the two steps' failures mean opposite things and must not share
            # an `except`: a raise from the backstop caught by the write guard
            # above would record `outcome='failed'` with `moved` never set —
            # a committed repair booked as an infrastructure failure that did
            # nothing. That would un-count it in `ReferentRepairStats.repaired`,
            # suppress its `_referent_repair_streaks` increment, and drop the
            # emptied node from `_cleanup_emptied_nodes` candidacy: the exact
            # "report zero repairs for an episode that performed one" shape
            # that `_backstop_endpoint_summaries` and the cleanup's own
            # independent guard both already exist to rule out. Same reasoning,
            # same remedy, third site.
            #
            # `_backstop_endpoint_summaries` swallows per node, so only a
            # malformed `result` reaches this arm — and when it does, the
            # fallback reports what `reassign_edge` said IT refreshed, read
            # totally so the recovery path cannot itself raise.
            refreshed: tuple[str, ...] = ()
            if moved:
                try:
                    refreshed = await self._backstop_endpoint_summaries(
                        result, group_id=group_id,
                    )
                except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    logger.warning(
                        'Referent repair: the backstop summary refresh failed '
                        'wholesale for edge %s; the endpoint move already '
                        'committed and STANDS, and is recorded as a repair. '
                        'Only the summary regeneration is in doubt',
                        finding.edge_uuid, exc_info=True,
                    )
                    reported = (
                        result.get('refreshed_nodes')
                        if isinstance(result, dict) else None
                    )
                    refreshed = (
                        tuple(reported)
                        if isinstance(reported, (list, tuple)) else ()
                    )
            repair_stats.repairs.append(ReferentRepair(
                edge_uuid=finding.edge_uuid,
                which_end=finding.which_end,
                outcome='repaired',
                # From reassign_edge's topology re-read, NOT from the finding.
                old_endpoint_uuid=result.get(
                    'old_endpoint_uuid', finding.old_endpoint_uuid,
                ),
                check=finding.check,
                new_endpoint_uuid=target_uuid,
                intended_referent=intended.node_name,
                # zeta found no single pre-existing node keyed by this name
                # moments earlier under this same lock, so ensure_entity_node
                # took its mint-or-collapse path. The two are deliberately not
                # distinguished here — zeta collapses them to one `None` on
                # purpose, and telling them apart would cost a second query
                # inside the identity lock to sharpen a telemetry count.
                minted=finding.new_endpoint_uuid is None,
                moved=moved,
                summaries_refreshed=refreshed,
            ))

    async def _backstop_endpoint_summaries(
        self, result: dict[str, Any], *, group_id: str
    ) -> tuple[str, ...]:
        """Step 3 of the repair sequence — a BACKSTOP, not a third unconditional call.

        ``reassign_edge`` already refreshes the two AFFECTED endpoint summaries
        after a real move (the OLD endpoint, which lost the edge, and the NEW
        one, which gained it) — but per-node try/except that LOGS AND SWALLOWS,
        reporting only what actually succeeded in ``refreshed_nodes``. That
        leaves the two naive options both wrong:

        * An unconditional third call DOUBLES the summary regeneration on the
          ~100% happy path, inside the per-group identity lock where every
          extra round-trip serializes same-group writes.
        * Omitting step 3 entirely leaves the PRD's stated user-observable
          signal — "the ``Task N±1`` summary no longer contains it" — silently
          degradable whenever that swallowed exception fires.

        Reading ``refreshed_nodes`` and re-refreshing only the REMAINDER is the
        only formulation that makes the signal a guarantee at zero happy-path
        cost: it is the primitive's own report of what it actually achieved,
        and it is the only way to tell the two apart from out here.

        Per-node try/except that logs and swallows a generic ``Exception``: the
        topology move has ALREADY COMMITTED by the time this runs, so a failed
        summary regeneration must never un-count the repair — that would report
        zero repairs for an episode that performed one, the exact
        silent-degradation shape INV-2 rules out.
        ``CancelledError``/``KeyboardInterrupt``/``SystemExit`` propagate.

        :meth:`GraphitiBackend.set_entity_summary` (task 2057) remains the
        documented MANUAL escape hatch for a stale sentence still carried by a
        genuinely VALID edge, and is deliberately never invoked from here:
        overwriting a summary verbatim is not a decision a write-time pass may
        take unattended.

        Returns:
            The union of what ``reassign_edge`` reported refreshing and what
            this backstop then refreshed — what is true of the GRAPH, not what
            was called.
        """
        already = list(result.get('refreshed_nodes') or ())
        affected = [
            result.get('old_endpoint_uuid'), result.get('new_endpoint_uuid'),
        ]
        refreshed = list(already)
        for node_uuid in affected:
            if not node_uuid or node_uuid in refreshed:
                continue
            try:
                await self.graphiti.refresh_entity_summary(
                    node_uuid, group_id=group_id,
                )
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                logger.warning(
                    'Referent repair: backstop summary refresh failed for node '
                    '%s (edge %s); the endpoint move already committed and '
                    'stands, but that node\'s summary may still name the '
                    'referent the edge no longer hangs off',
                    node_uuid, result.get('uuid'), exc_info=True,
                )
                continue
            refreshed.append(node_uuid)
        return tuple(refreshed)
    async def _reconcile_episode_identity(
        self, result: Any, *, group_id: str, referents: ReferentSet = (),
        content: str = '', referent_source: str = 'derived',
    ) -> ReconcileStats:
        """Fold the eight post-write identity/verification/repair sweeps into one call.

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

        ``_verify_episode_referents`` (task 3671, PRD leaf zeta) runs LAST,
        after ``_normalize_task_node_names``, and that ordering is load-bearing:
        it keys on the canonical ``Task N`` names the normalization pass
        produces, and the ``new_endpoint_uuid`` it resolves — plus any repair
        leaf eta performs off its findings — must describe POST-normalization
        topology. Minting a ``Task N`` node before normalization ran would
        create exactly the duplicate that pass exists to collapse. (Should task
        3335's cross-project split ever land, zeta must run after that too, for
        the same reason.) It is also the one sub-pass that performs no writes:
        it DETECTS and records, and its ``ReferentStats`` is the structured
        evidence eta acts on.

        ``_repair_episode_referents`` (task 3672, PRD leaf eta) is the EIGHTH
        and last, and the ONLY WRITING CONSUMER of zeta's detect-only output —
        it re-derives no verdict of its own, acting on nothing but the findings
        handed to it. Its placement after zeta is a data dependency (it takes
        zeta's ``ReferentStats`` as its argument), and its placement after
        ``_normalize_task_node_names`` is load-bearing in the SAME direction
        zeta's is, only more so: eta MINTS ``Task N`` nodes and REPOINTS edges
        onto them, so minting before normalization ran would create exactly the
        duplicate-name pair that pass exists to collapse — the repair would
        introduce the disease it was called to cure.

        Running inside this chain is also what satisfies alpha's
        ``ensure_entity_node`` LOCK CONTRACT: ``_execute_graphiti_write`` wraps
        this whole call in ``async with _identity_lock_for(group_id)``, and eta
        performs no locking of its own. That placement is load-bearing, not
        incidental.

        One repair shape is deliberately NOT automated: a stale sentence still
        carried by a genuinely VALID edge. ``refresh_entity_summary`` regenerates
        a summary from the edges that remain, so it cannot remove a sentence
        whose edge is correct; ``GraphitiBackend.set_entity_summary`` (task 2057)
        stays the documented MANUAL escape hatch for that case. Overwriting a
        summary verbatim is not a decision a write-time pass may take unattended.

        Each sub-pass runs under its own best-effort guard: a generic
        ``Exception`` is logged and recorded as that sub-pass's label in
        ``ReconcileStats.errors`` (leaving its count at its default — ``0`` for
        the six int passes, an empty ``ReferentStats`` for zeta, an empty
        ``ReferentRepairStats`` for eta), and the remaining sub-passes still
        run — a single sub-pass failure must never fail the already-committed
        episode write. That guarantee is worth most at eta, the one pass that
        WRITES: its failure is the likeliest to be real, and it arrives after
        the episode is already durable.
        ``CancelledError``/``KeyboardInterrupt``/``SystemExit`` are never
        swallowed; they propagate immediately and skip any later sub-passes.

        Args:
            result: The value returned by ``add_episode`` (typically an
                AddEpisodeResults object), forwarded verbatim to every
                sub-pass.
            group_id: The project graph this episode was written to.
            referents: The referent set the write DECLARED itself to be about,
                forwarded to the verification sub-pass. Defaults to empty, which
                makes that pass a no-op — so every caller predating task 3671 is
                unchanged.

        Returns:
            A ReconcileStats aggregating every sub-pass's count, the
            verification pass's findings, the repair pass's structured records,
            and any per-sub-pass failure labels.
        """
        stats = ReconcileStats()

        async def _run_pass(label: str, coro: Awaitable[_T], default: _T) -> _T:
            # ONE best-effort guard, not two. Task 3671's verification sub-pass
            # is the first to return a dataclass rather than an int, so the
            # caller supplies the DEFAULT its own type demands — a parallel
            # object-returning guard would duplicate the swallow/propagate rule
            # at two sites that must stay identical (the INV-5 lockstep
            # duplication this PRD exists to push back on), and returning `0`
            # where a ReferentStats belongs would blow up every consumer on
            # exactly the error path this guard exists to survive.
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
                return default

        stats.edges_deduped = await _run_pass(
            '_dedup_episode_edges',
            self._dedup_episode_edges(result, group_id=group_id),
            0,
        )
        stats.dependency_edges_restored = await _run_pass(
            '_restore_superseded_dependency_edges',
            self._restore_superseded_dependency_edges(result, group_id=group_id),
            0,
        )
        stats.sibling_edges_restored = await _run_pass(
            '_restore_falsely_superseded_sibling_edges',
            self._restore_falsely_superseded_sibling_edges(result, group_id=group_id),
            0,
        )
        stats.stale_ttl_edges_invalidated = await _run_pass(
            '_invalidate_stale_superseded_ttl_edges',
            self._invalidate_stale_superseded_ttl_edges(result, group_id=group_id),
            0,
        )
        stats.nodes_resolved = await _run_pass(
            '_dedup_episode_nodes',
            self._dedup_episode_nodes(result, group_id=group_id),
            0,
        )
        stats.task_names_normalized = await _run_pass(
            '_normalize_task_node_names',
            self._normalize_task_node_names(result, group_id=group_id),
            0,
        )
        stats.referent_stats = await _run_pass(
            '_verify_episode_referents',
            self._verify_episode_referents(
                result, group_id=group_id, referents=referents,
                content=content, referent_source=referent_source,
            ),
            ReferentStats(),
        )
        # EIGHTH, and constructed sequentially rather than alongside the seven
        # above, because the coroutine's ARGUMENT is zeta's result: eta consumes
        # `stats.referent_stats`, which does not exist until the line above has
        # awaited. That data dependency is the whole ordering constraint, and
        # writing it this way makes it un-reorderable by accident.
        #
        # `stats.referent_stats` is passed even on zeta's failure path, where
        # `_run_pass` substituted an empty `ReferentStats` — eta then walks no
        # findings and no-ops, which is the correct degradation: a repair pass
        # must never act on evidence its detector failed to produce.
        stats.repair_stats = await _run_pass(
            '_repair_episode_referents',
            self._repair_episode_referents(
                stats.referent_stats, group_id=group_id,
                episode_uuid=_episode_uuid_of(result),
            ),
            ReferentRepairStats(),
        )
        return stats

    def referent_source_counts(self) -> dict[str, int]:
        """How many Graphiti write ATTEMPTS resolved to each referent source.

        ATTEMPTS, not completed writes, and the distinction is load-bearing for
        anyone building an alert on the rate.  The increment sits at the TOP of
        ``_execute_graphiti_write``, which ``DurableWriteQueue._process_item``
        re-invokes on every RETRY of an item with a freshly parsed payload — so
        a retry storm on one group inflates whichever bucket that item lands in,
        and an item that eventually dead-letters is still counted.  Retries are
        in the numerator AND the denominator; the skew is roughly uniform across
        buckets in the common case (a row's source does not change between its
        own attempts), so a "sustained 100% none" reading survives it, but a
        per-bucket ABSOLUTE count must not be read as a count of memories.

        The increment deliberately stays at the top rather than moving after the
        successful backend call: counting only successes would make the escape
        go dark during a backend outage — exactly when a referent-less write
        storm is least likely to be noticed any other way — and would decouple
        it from the single decode the journal stamp also reads.

        The INV-4 storm escape for the referent-set queue channel (task 3670,
        PRD leaf epsilon), and the read side of ``_referent_source_counts``.

        Emitted at the CONSUMER (``_execute_graphiti_write``), not at the three
        producers, deliberately: the regression this exists to detect is "the
        plumbing breaks, every row arrives referent-less, and the feature
        no-ops in total silence", and that failure lives on the PRODUCER side —
        a counter emitted there would go dark in exactly that scenario. Only
        the consumer sees both new-format and old-format rows.

        Buckets ALL FOUR sources rather than only 'none', because leaf iota
        needs a denominator: "sustained 100% none" is a RATE, and an absolute
        none-count alone cannot distinguish a broken producer from a quiet
        system.

        Returns a COPY, so a caller cannot mutate the escape hatch's own state.
        Process-lifetime totals, never reset — a monotonic counter a reader
        samples and differences, matching the uptime-baseline convention above.
        """
        return dict(self._referent_source_counts)

    def referent_finding_counts(self) -> dict[str, int]:
        """How many verification findings each check has produced, ever.

        The read side of ``_referent_finding_counts`` and the INV-4 storm escape
        for the verification sub-pass (task 3671, PRD leaf zeta) — and the
        surface leaf IOTA reads to turn findings into a rate. Deliberately
        mirrors :meth:`referent_source_counts` rather than inventing a second
        idiom in this file.

        THE BUCKETS ARE TWO ORTHOGONAL AXES, NOT A PARTITION.
        ``'set-membership'`` and ``'per-edge-pairing'`` answer "which check
        fired" and do partition the findings between them (they are ordered, so
        an endpoint failing both is counted once, under membership).
        ``'unresolvable'`` answers the independent question "could a correct
        target be determined at all", and increments ALONGSIDE whichever check
        fired. So the three counts intentionally do not sum to the finding
        total, and ``unresolvable`` is a numerator over the other two, not a
        third category.

        Every bucket exists from construction, so a reader never has to
        distinguish "zero" from "absent". Returns a COPY, so a caller cannot
        mutate the escape hatch's own state. Process-lifetime totals, never
        reset — a monotonic counter a reader samples and differences, matching
        the convention above.
        """
        return dict(self._referent_finding_counts)

    def referent_repair_counts(self) -> dict[str, Any]:
        """What the REPAIR sub-pass has done, ever, plus its live streaks.

        The read side of ``_referent_repair_counts`` / ``_referent_repair_streaks``
        and the INV-4 storm escape for the repair sub-pass (task 3672, PRD leaf
        eta) — the third in the series, following
        :meth:`referent_source_counts` and :meth:`referent_finding_counts`
        rather than inventing a fourth idiom in this file.

        IT DOES NOT MIRROR THEIR RETURN TYPE, and a caller must not assume it
        does.  Both siblings return ``dict[str, int]``; this returns
        ``dict[str, Any]``, because the repair sub-pass is the only one of the
        three that has a GAUGE to report as well as counters, and eta specifies
        its read side as ONE accessor carrying both.  Two consequences a
        consumer has to know, because they are the ones that bite:

        - ``sum(counts.values())``, and "emit every key as a gauge", work on
          both siblings and BREAK here: ``'streaks'`` is a nested
          ``dict[str, int]``, not an int.  Read the totals as
          ``{k: v for k, v in counts.items() if k != 'streaks'}`` and iterate
          the gauge separately.
        - ``'streaks'`` shares the flat key space with the outcome buckets, so
          the name is RESERVED.  A future sixth bucket called 'streaks' would
          silently shadow the gauge rather than collide loudly.

        THE FIVE TOTALS ARE PROCESS-LIFETIME AND MONOTONIC, never reset: a
        reader samples and differences them. They partition eta's dispositions
        the way :data:`REFERENT_REPAIR_OUTCOMES` describes —
        ``flagged_unrepairable`` folds the two REFUSALS ('unrepairable',
        'degenerate') while ``failed`` stays separate, because conflating an
        infrastructure fault with a refusal would let a FalkorDB outage read as
        a scanner regression in leaf iota's rate. ``repaired`` counts endpoints
        actually MOVED, so it is a strict measure of what changed in the graph.

        ``'streaks'`` IS A GAUGE, NOT A COUNTER, and the only entry here that
        can go DOWN: group_id -> consecutive episodes whose repair pass moved
        something, cleared by a pass that looked and found nothing. It is the
        value the storm gate thresholds, exposed so an operator can see how
        close a project is to the alarm rather than only learning at the breach.
        A group appears only once a pass has produced evidence about it, so an
        absent key means "no repair pass has drawn a conclusion here yet" —
        distinct from a present ``0``, which means "checked, clean".

        Returns a COPY at BOTH levels — the nested streaks dict included — so a
        caller cannot mutate the escape hatch's own state. A read-only escape
        hatch a consumer can write through is not an escape hatch.
        """
        counts: dict[str, Any] = dict(self._referent_repair_counts)
        counts['streaks'] = dict(self._referent_repair_streaks)
        return counts

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
        # task 3142: rides the same payload channel temporal_context does, so
        # the tag reaches the persisted episodic node (and, via
        # _dual_write_callback, every fact derived from it).
        unverified_claim = bool(payload.pop('unverified_claim', False))
        # task 3670: the referent set resolved at the write boundary, popped on
        # the same channel. An ABSENT key decodes to ((), 'none'), so a queue
        # row written before this feature executes byte-identically to today.
        #
        # `referents` is handed to `_reconcile_episode_identity` below, which
        # forwards it to leaf zeta's `_verify_episode_referents` (task 3671) as
        # the LAST sub-pass — INSIDE the identity-lock critical section,
        # deliberately, so no wrongly-attached state is ever externally visible
        # between the write and its verification. Nothing the BACKEND sees
        # changes, which is what keeps an old-format row byte-identical.
        referents: ReferentSet
        referents, referent_source = _decode_referents(payload)
        # INV-4 escape: EVERY Graphiti write is bucketed, so the absent and
        # degraded paths are counted rather than silently falling through. See
        # `_referent_source_counts` in __init__ for why this is unconditional
        # and why all four sources are bucketed. Leaf iota reads it.
        self._referent_source_counts[referent_source] += 1
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
                # referent_source/referent_count (task 3670) are the DURABLE
                # half of the telemetry split, and come from the ONE decode
                # above — never re-derived, so the durable channel and the
                # in-process counter cannot disagree. The counter is the
                # unconditional INV-4 escape (it exists even when
                # `_write_journal` is None); this row is what gives leaf iota
                # per-project, time-windowed data, through a journal row that
                # already exists — no new schema, no new table, no new write.
                # That resolves epsilon's half of PRD open question 2 (which
                # suggested `write_ops.params`) without pre-empting iota's
                # read-path choice.
                payload={
                    'content': payload['content'][:200],
                    'group_id': payload.get('group_id'),
                    'referent_source': referent_source,
                    'referent_count': len(referents),
                },
                coro=self.graphiti.add_episode(
                    name=payload.get('name', ''),
                    content=payload['content'],
                    source=episode_type,
                    group_id=payload['group_id'],
                    source_description=payload.get('source_description', ''),
                    uuid=payload.get('uuid'),
                    temporal_context=temporal_context,
                    reference_time=reference_time,
                    unverified_claim=unverified_claim,
                ),
            )
            reconcile_stats = await self._reconcile_episode_identity(
                result, group_id=payload['group_id'], referents=referents,
                # BOTH halves of what zeta needs beyond the decoded set:
                # `content` so it can re-derive the producer's ambiguity set
                # (epsilon drops `.ambiguous` from the wire on purpose), and
                # `referent_source` so an ambient `metadata['task_id']`
                # declaration is never mistaken for evidence about which node an
                # edge belongs on. The FULL content, not the 200-char journal
                # excerpt above -- a truncated body would silently lose the
                # second half of an ambiguity pair.
                content=payload['content'],
                referent_source=referent_source,
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
        unverified_claim = bool(payload.get('unverified_claim', False))
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
        # Omitted entirely when untagged (task 3142) rather than set False, so
        # no existing record shape changes and a metadata filter for the key
        # matches only genuinely flagged facts.
        if unverified_claim:
            metadata['unverified_claim'] = True

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
                        # task 3142: the Mem0 half rides its own payload
                        # channel, so the episode's tag must be copied onto
                        # every derived fact explicitly — those facts ARE the
                        # artefacts the incident produced.
                        'unverified_claim': payload.get('unverified_claim', False),
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
        unverified_claim: bool = False,
        _source: str = 'mcp_tool',
    ) -> AddEpisodeResponse:
        """Full ingestion pipeline — durably enqueue episode, return immediately.

        ``unverified_claim`` (task 3142) marks an episode carrying a completion
        claim that could not be confirmed against its live authority. It is a
        LABEL, never a rejection: the episode is ingested either way, and the
        flag follows the same payload -> backend path as ``temporal_context``
        so both the Graphiti episodic node and every derived Mem0 fact carry
        it.
        """
        scope = Scope(project_id=project_id, agent_id=agent_id, session_id=session_id)
        episode_id = str(uuid_mod.uuid4())
        write_op_id = str(uuid_mod.uuid4())

        # Parse source type name for storage
        try:
            source_name = EpisodeType[source.lower()].name
        except (KeyError, AttributeError):
            source_name = 'text'

        assert self.durable_queue is not None

        # Resolve WHICH referents this episode is about (task 3670, PRD leaf
        # epsilon) BEFORE the try below, for the same loud-over-silent reason
        # add_memory's call sits outside its own try: gamma raises
        # InputValidationError on a structural wiring bug, and that must not be
        # absorbed by an enqueue-failure handler.
        #
        # metadata=None is not an oversight: add_episode deliberately never
        # persists a metadata argument — the same fact that forced task 3142's
        # `unverified_claim` onto this payload channel — so the bridge has
        # nothing to read and the derived scan is the only live source here.
        #
        # declared=None: leaf delta owns the `entities` parameter; this is the
        # seam it fills.
        resolution = resolve_referents(
            declared=None,
            metadata=None,
            content=content,
            group_id=scope.graphiti_group_id,
        )

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
                    'unverified_claim': unverified_claim,
                    'reference_time': reference_time.isoformat() if reference_time is not None else None,
                    'referents': _encode_referents(resolution),
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
            # Resolve WHICH referents this write is about (task 3670, PRD leaf
            # epsilon), so leaf zeta can verify the resulting edges against it.
            #
            # Placement is load-bearing at BOTH ends:
            #   INSIDE `if write_graphiti:` — a Mem0-only write never reaches
            #   Graphiti, so it pays for no scan;
            #   OUTSIDE the `try:` below, which degrades to `_graphiti_error`
            #   and DROPS the Graphiti write. gamma raises InputValidationError
            #   on structural inputs (a non-str content or group_id) precisely
            #   so a wiring bug is loud; resolving inside that try would
            #   convert that loud signal into a silently skipped Graphiti
            #   write.
            #
            # `meta` is read AFTER _normalize_task_id_metadata has coerced
            # task_id to a scalar str, which is the contract gamma's metadata
            # bridge documents itself against.
            #
            # declared=None: leaf delta owns the `entities` parameter and its
            # `_entities_gate`, and THIS CALL is the single seam it fills. No
            # declared referents can exist until it lands.
            resolution = resolve_referents(
                declared=None,
                metadata=meta,
                content=content,
                group_id=scope.graphiti_group_id,
            )
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
                        # Popped and decoded by _execute_graphiti_write.
                        'referents': _encode_referents(resolution),
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
            if not isinstance(content, str):
                # `resolve_referents` (task 3670) raises InputValidationError on
                # a truthy non-str content, deliberately — but this call sits in
                # a per-memory loop whose `enqueue_batch` only runs AFTER the
                # loop completes, so letting it propagate would abort the WHOLE
                # replay and enqueue nothing over ONE malformed Mem0 record.
                # Skipping the record keeps the blast radius at one row, which
                # is what it was before referents were threaded here. Loud
                # rather than silent, unlike the empty-content skip above: an
                # empty memory is ordinary, a non-str one is a Mem0 anomaly.
                logger.warning(
                    'Skipping replay of a Mem0 record whose memory is not a '
                    'string (got %s): %s',
                    type(content).__name__, _safe_repr(content),
                )
                continue
            meta = mem.get('metadata', {}) or {}
            category = meta.get('category', 'observations_and_summaries')
            # The THIRD and last producer of add_memory_graphiti rows (task
            # 3670, PRD leaf epsilon). Threaded even though the PRD named only
            # the two primary write-boundary sites: replayed rows carry real
            # prose whose referents the derived scanner can see, so leaving
            # them on the absent path would stamp them 'none' and inflate leaf
            # iota's undeclared bucket with writes that were plainly derivable
            # — a false regression signal in the very counter this task exists
            # to make trustworthy. They also produce real graph edges leaf zeta
            # will want to verify.
            #
            # Unlike add_episode, this loop DOES hold a metadata dict (the Mem0
            # record's own), so the bridge is live here. declared=None: leaf
            # delta's seam, as at the other two producers.
            #
            # add_system_record is deliberately NOT threaded — it is Mem0-only
            # and never routes to Graphiti.
            resolution = resolve_referents(
                declared=None, metadata=meta, content=content, group_id=target,
            )
            batch.append({
                'group_id': target,
                'operation': 'add_memory_graphiti',
                'payload': {
                    'name': f'replay_{category}',
                    'content': content,
                    'source': 'text',
                    'group_id': target,
                    'source_description': f'replay_from_mem0:{category}',
                    'referents': _encode_referents(resolution),
                },
                'callback_type': 'refresh_entity_summaries',
            })

        if batch:
            await self.durable_queue.enqueue_batch(batch)
        return len(batch)

    # ------------------------------------------------------------------
    # Read: search
    #
    # Cross-store merge is Reciprocal Rank Fusion with RRF_K (task 3658, PRD
    # D4). The router's primary_store is a TIEBREAK, not precedence: it used to
    # order results wholesale, which let one store fill `limit` and made the
    # other structurally unreachable no matter how well it matched. Graphiti
    # emits no synthesized scores any more — it has none of its own to report,
    # which is why fusion is by rank rather than by calibrated score. See
    # `_rrf_score` and the `search` docstring for the full consumer contract.
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
        anchor_topics: bool = True,
    ) -> list[MemoryResult]:
        """Unified search across both stores with automatic fan-out.

        When include_planned=False (default), edges and memories from planning
        episodes (temporal_context='planning') are excluded.  Set include_planned=True
        to include them — useful for reconciliation and auditing.

        Ordering (task 3658, PRD D4).  Each responding store ranks its own
        results — Mem0 by cosine descending, Graphiti by its backend rank — and
        the two are merged by Reciprocal Rank Fusion with ``K = RRF_K``, ties
        broken by (router primary store, store-internal rank).  The router's
        primary store is a TIEBREAK ONLY; it is no longer precedence, so
        neither store can fill ``limit`` and shut the other out.

        Scores.  ``relevance_score`` is the fused RRF value and is **ORDINAL,
        never a similarity**: single-store rank-1 is 1/61 ~ 0.0164 no matter
        how good the match is.  Do not threshold it, do not compare it across
        API versions, and do not compare it to a cosine.  Per-store truth lives
        in ``metadata``:

          - ``store_rank`` — int, 1-based rank within the store that returned
            it (over that store's surviving results; deliberately not
            renumbered by the category filter below, since it is a per-store
            telemetry fact rather than a position in the merged output).
          - ``store_score`` — the Mem0 cosine, verbatim; ``None`` for Graphiti,
            whose public search() exposes no scores at all.

        ``degraded`` / ``failed_stores`` / ``failure_diagnostics`` and
        per-store error absorption are unchanged: a store that raises or times
        out is absorbed, and the surviving store's results are still returned.

        ``anchor_topics`` (task 3111).  Topic-anchored recall is a PROMOTION,
        not an addition: the returned window stays exactly ``limit`` long, so
        each pinned canonical evicts the lowest-ranked genuine hit.  That trade
        is right for an AGENT-FACING read, where surfacing a cluster's
        canonical is worth one marginal tail result, and WRONG for a
        correctness-critical machine consumer that treats the window as a
        candidate set.  Pass ``anchor_topics=False`` from any caller which:

          - thresholds or post-filters the window and would silently lose a
            genuine candidate to displacement — the procedural_knowledge
            near-duplicate WRITE guard (``server/tools.py``, which searches at
            ``limit=5``) is the sharp case.  A pinned canonical deliberately
            carries NO ``metadata['store_score']`` (see the score contract
            below), so it can never qualify in ``find_near_duplicate_memory``:
            every pin is a candidate slot spent on a record the guard must
            ignore, shrinking an effective 5-candidate set toward 2 on exactly
            the consolidated topics the guard exists to protect.
          - is an IDEMPOTENCY check, where a displaced prior record reads as
            "absent" and causes a duplicate WRITE
            (``reconciliation/mem0_dedup.find_prior_memories``).
          - sweeps for markers, where a displaced marker is silently "not
            swept this cycle"
            (``reconciliation/stages/task_knowledge_sync._query_stage2_flags``).

        The default stays ``True`` because the agent-facing seams are the
        majority of call sites and are the ones this task exists to fix; the
        flag is an explicit OPT-OUT so a future machine consumer that forgets
        it degrades to today's already-shipped behaviour rather than to a
        silent correctness bug.  The pin is NOT relocated to the MCP boundary:
        a non-suppressing pin is contractually placed at this seam (task 3111
        scope note) so that ``stores=['mem0']`` / category-scoped agent
        searches are anchored too, unlike the SUPPRESSING grouped read, which
        PRD V2 bars from this seam precisely because it would blind these same
        machine consumers with no way to opt out.
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

        # Merge by Reciprocal Rank Fusion (task 3658, PRD D4).  Primary sort is
        # the fused score descending; the router's primary store is only a
        # TIEBREAK — it used to be outright precedence, which let one store
        # fill `limit` and made the other structurally unreachable.  Store rank
        # is the final tiebreak, which makes the ordering total and
        # deterministic (is_primary already distinguishes the only two stores
        # that can tie on score).  Read store_rank defensively so a result from
        # a future code path that lacks the key can never raise here.
        def sort_key(r: MemoryResult) -> tuple[float, int, int]:
            primary_rank = 0 if r.source_store == route.primary_store else 1
            return (-r.relevance_score, primary_rank, r.metadata.get('store_rank', 0))

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

        # ---- Topic-anchored canonical pin (task 3111) -------------------
        # PLACEMENT IS LOAD-BEARING, in both directions:
        #   * AFTER the category filter above, because that comprehension
        #     REBINDS `results`, and a pinned mem0 row whose category fell
        #     outside `cat_set` would simply be dropped there (only Graphiti
        #     rows with category None get the escape hatch).
        #   * AFTER the sort at the top of this tail, because `sort_key` orders
        #     by -relevance_score and every RRF score is <= 1/61 — a pin with
        #     no synthetic score could not survive a re-sort, and giving it one
        #     is forbidden (see the store_score contract below).
        #   * BEFORE `final = results[:limit]`, so the pinned record lands
        #     INSIDE the returned window and the limit contract is preserved by
        #     construction rather than by arithmetic.
        # GATE: read LIVE off the shared config object on every call, never
        # captured at construction — that live read is what makes the knob
        # genuinely green-tier hot-reloadable (config/reload.py's reload-safety
        # rule); a construction-captured value would not observe an in-place
        # reload and would have to stay restart-only.
        # GATE 1 (per-call): `anchor_topics=False` is the caller asserting it
        # reads this window as a CANDIDATE SET, not as a presentation — see the
        # docstring.  Checked FIRST and cheaply, so an opted-out caller pays
        # neither the config read nor the harvest.  Gate 2 is the live,
        # green-tier config knob.
        if anchor_topics and resolve_topic_anchor_enabled(self):
            try:
                # HARVEST FROM THE WINDOW THE CALLER WILL ACTUALLY SEE, not from the
                # full merged list.  `results` here still holds every merged hit, and
                # the slice to `limit` happens below — so harvesting from all of it
                # would let a topic carried ONLY by an out-of-window record pull in a
                # canonical that then DISPLACES an in-window record the caller would
                # otherwise have been shown.  That inverts the contract both
                # agent-facing docstrings state ("finding any member of a consolidated
                # cluster also surfaces that topic's canonical"): the agent would pay
                # the displacement for a cluster it never found a member of.  It also
                # spends a Qdrant round-trip per invisible topic on the hot path.
                # NAMED `topics_to_anchor`, not `anchor_topics`: the latter is now
                # the boolean OPT-OUT PARAMETER of this method, and reusing it here
                # would shadow it — silently making the gate above unreachable to any
                # later read of the flag within this block.
                topics_to_anchor = extract_anchor_topics(
                    results[:limit], max_topics=_MAX_ANCHOR_TOPICS
                )
                # FAN OUT, don't serialize.  The lookups are fully independent reads
                # (`pinned_ids` is a post-hoc dedup and `pin_at` is pure ordering —
                # neither is an input to any lookup), and this seam is the hottest read
                # path in the system: every agent search runs it.  (The
                # procedural_knowledge near-dup pre-check does NOT — it opts out via
                # `anchor_topics=False` — so it pays none of this.)  Serialized,
                # the cap would add up to
                # _MAX_ANCHOR_TOPICS round-trips of latency instead of one round-trip's
                # worth.  No semaphore, unlike grouped_read._bounded_gather: that
                # module's fan-out is sized by the CALLER's result list (up to the
                # search tool's 1000-hit clamp), whereas this one is bounded at 3 by
                # _MAX_ANCHOR_TOPICS before it starts.
                payload_sets = await gather_collect(
                    self.get_memories_by_metadata(
                        project_id,
                        {'topic': topic, 'canonical': True},
                        limit=_ANCHOR_SCROLL_LIMIT,
                    )
                    for topic in topics_to_anchor
                )
                for payloads in payload_sets:
                    # gather_collect CAPTURES per-item exceptions rather than raising
                    # (its documented Pass-2 "caller classifies" contract).  Re-raise
                    # the first so the two-tier band below classifies a fan-out failure
                    # exactly as it classified a sequential await: a wiring bug stays
                    # loud, a backend fault still fails open.
                    if isinstance(payloads, Exception):
                        raise payloads

                # BUILD INTO A LOCAL COPY and rebind only on FULL success, so the
                # fail-open below can honestly promise the un-pinned list.  Mutating
                # `results` in place would leave the first topic's pin applied when the
                # second one raised — a partially-transformed list is not the "returning
                # un-pinned results" the WARNING claims.
                pinned = list(results)
                # Two distinct topics can legitimately resolve to the SAME canonical, so
                # pinned ids are tracked across the loop: without this the second topic
                # would pin an already-pinned record a second time.
                pinned_ids: set[str] = set()
                # Flags are applied only once the whole loop has succeeded: setting
                # `topic_anchored` on a moved result mutates an object SHARED with
                # `results`, so an eager write would survive the fail-open rebind and
                # leak a half-applied pin into the "untouched" list.
                to_flag: list[MemoryResult] = []
                # Pins accumulate in TOPIC-RANK order: each lands just after the
                # previously-pinned ones rather than at index 0, so the canonical of the
                # highest-ranked topic stays ahead of the next topic's.  A plain
                # insert(0) would emit them in reverse.  Every slot in [0, pin_at) is a
                # pin whose id is already in `pinned_ids`, so the move-to-front branch
                # below can only ever find `existing >= pin_at` and its pop cannot
                # disturb them.
                pin_at = 0
                for payloads in payload_sets:
                    canonical = select_canonical_payload(
                        payloads,
                        allowed_categories=set(categories) if categories else None,
                        include_planned=include_planned,
                    )
                    if canonical is None:
                        continue
                    canonical_id = canonical.get('id', '')
                    if canonical_id in pinned_ids:
                        continue
                    pinned_ids.add(canonical_id)

                    # MOVE, don't rebuild, when the canonical is already among the
                    # cosine results — which is the measured inversion itself: it IS a
                    # genuine match, just the worst one in its own cluster, so it sits
                    # at the bottom of the window or just outside it.  Its
                    # relevance_score and its metadata (including the real
                    # metadata['store_score'] cosine `_search_mem0` stamped) are honest
                    # measurements; replacing them with a freshly-built zero-scored
                    # result would destroy signal every score-reading consumer depends
                    # on, the near-duplicate write guard first among them.  Only the
                    # ORDER changes, plus the topic_anchored flag.
                    existing = next(
                        (i for i, r in enumerate(pinned) if r.id == canonical_id), None
                    )
                    if existing is not None:
                        moved = pinned.pop(existing)
                        to_flag.append(moved)
                        pinned.insert(pin_at, moved)
                        pin_at += 1
                        continue

                    payload_meta = canonical.get('metadata') or {}
                    # WIRE SHAPE: partition the raw payload and put only the CUSTOM half
                    # on the wire, exactly as grouped_read._promoted_parent does for the
                    # structurally identical raw-payload -> search-hit conversion.
                    # split_managed_metadata is the DECIDED HOME for that partition
                    # (INV-5), so this reuses it rather than re-deriving the key set.
                    # Forwarding the raw payload would give ONE canonical two different
                    # wire shapes depending on how it was reached — pinned vs. direct
                    # hit — leaking hash/user_id/role to the MCP consumer and emitting
                    # the body text TWICE (once as `content`, once as metadata['data']),
                    # which also works against this arm's measured 1070 tokens/query on
                    # the hottest read path in the system.
                    #
                    # `content` still reads the RAW payload: 'data' is a mem0-MANAGED
                    # key, so it lands in `managed`, never in the custom half that goes
                    # on the wire.  `created_at` comes off the scroll dict's top-level
                    # key — get_memories_by_metadata's documented return shape — which
                    # scroll_by_metadata lifts from that same managed payload key.
                    managed, custom = split_managed_metadata(dict(payload_meta))
                    created_at = canonical.get('created_at')
                    if not isinstance(created_at, str):
                        created_at = managed.get('created_at')
                    # SCORE CONTRACT: the injected anchor must never gain a
                    # 'store_score' — note the split above cannot introduce one,
                    # since a raw scroll payload has no such key and only
                    # _search_mem0 ever stamps it.  The
                    # write-time near-duplicate guard reads the cosine from
                    # metadata['store_score'] and qualifies on `>= threshold`
                    # (near_duplicate_guard.find_near_duplicate_memory :114-121, via
                    # _cosine_of :71-84); a MISSING cosine means "not comparable" and can
                    # never qualify at any threshold, while a synthetic one would
                    # hard-block EVERY procedural_knowledge write on a consolidated
                    # topic — turning a retrieval fix into a write outage on precisely
                    # the topics it exists to help.  relevance_score is NOT the cosine
                    # since task 3658 (it is an ordinal RRF value, rank-1 ~ 0.0164), so
                    # setting it is not a substitute either.  The pin is by ORDER ONLY.
                    #
                    # Order alone SUFFICES because nothing downstream re-sorts it: the
                    # sort in this method already ran, above, and the only other
                    # transform between here and the agent —
                    # grouped_read.group_search_results at the MCP boundary — is
                    # append-only, with no sort() and no truncation of its own.  A
                    # record placed at index 0 here is still at index 0 when the agent
                    # reads it.  (Its one way to lose slot 0 is _suppress_child folding
                    # a CHILD-shaped hit into its parent, which is why
                    # select_canonical_payload excludes child-shaped payloads outright
                    # rather than trusting writes to be well-formed.)
                    pinned.insert(pin_at, MemoryResult(
                        id=canonical_id,
                        content=_mem0_content(payload_meta),
                        category=_mem0_category(custom),
                        source_store=SourceStore.mem0,
                        relevance_score=0.0,
                        metadata=custom,
                        created_at=created_at if isinstance(created_at, str) else None,
                        topic_anchored=True,
                    ))
                    pin_at += 1

                # COMMIT POINT.  Nothing above this line has touched `results` or any
                # object reachable from it, so every `raise` between here and the `try`
                # leaves the un-pinned list exactly as the sort/filter tail produced it.
                for result in to_flag:
                    result.topic_anchored = True
                results = pinned
            except (TypeError, AttributeError, NameError):
                # A wiring/programming bug — e.g. a future signature change to
                # get_memories_by_metadata or to the topic_anchor selectors —
                # rather than a transient backend failure.  Re-raise so it
                # surfaces loudly with a traceback instead of being absorbed
                # into fail-open, which would quietly leave the transform inert
                # in production behind nothing but a WARNING log.  Same two-tier
                # idiom as the near-duplicate pre-check in server/tools.py.
                raise
            except Exception:
                # FAIL-OPEN. Anchoring is a retrieval ENRICHMENT, so a failure
                # to enrich must never become a failure to RETRIEVE.  This seam
                # is shared by every MemoryService.search call site, so without
                # this one Qdrant read timeout would break every search in the
                # system — and get_memories_by_metadata genuinely PROPAGATES a
                # TimeoutError (unlike Mem0Backend.search, which swallows into
                # {}), so that is a live path, not a hypothetical.
                #
                # `results` is left exactly as the sort/filter tail produced it
                # — including its ORDER and every result's topic_anchored flag —
                # because the pins are built into a local copy and both the copy
                # and the flags are committed in one step at the end of the
                # `try`.  An ALL-OR-NOTHING pin, not a partial one: a list
                # carrying the first topic's pin and not the second's would be a
                # third state neither this WARNING nor any caller expects.
                logger.warning(
                    'topic-anchored recall failed; returning un-pinned results',
                    exc_info=True,
                )

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

        Results are ranked by Graphiti's own backend ordering and carry, in
        metadata (task 3658):

          - ``store_rank``: 1-based rank, contiguous over the SURVIVING edges
            (an edge dropped for ``invalid_at`` or planned-only provenance does
            not consume a rank).
          - ``store_score``: always ``None`` — Graphiti's public ``search()``
            exposes no scores, and synthesizing one would be a lie the
            cross-store merge would then act on.

        ``relevance_score`` is the ordinal ``_rrf_score(store_rank)``, not a
        similarity.
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
        for edge in edges:
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

            # Rank over SURVIVORS only (task 3658): the raw enumerate index
            # counted edges skipped above, and since RRF maps rank directly to
            # score, a gap would silently penalize Graphiti for facts the
            # caller never sees.
            rank = len(results) + 1

            metadata: dict[str, Any] = {'store_rank': rank, 'store_score': None}
            if is_planned_edge:
                metadata['planned'] = True

            results.append(MemoryResult(
                id=getattr(edge, 'uuid', str(rank)),
                content=fact,
                category=None,
                source_store=SourceStore.graphiti,
                relevance_score=_rrf_score(rank),
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

        Results are ranked by Mem0's own cosine-descending ordering and carry,
        in metadata (task 3658):

          - ``store_rank``: 1-based rank, contiguous over the SURVIVING results
            (a result dropped for ``planned`` does not consume a rank).
          - ``store_score``: Mem0's raw cosine, verbatim and un-clamped — the
            honest per-store signal for the E1 retrieval probe and the task
            3212 telemetry.

        ``relevance_score`` is the ordinal ``_rrf_score(store_rank)``: the
        cosine no longer reaches it, so it is no longer comparable to a
        similarity.
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

            category = _mem0_category(meta)

            # Rank over SURVIVORS only (task 3658) — a result skipped above must
            # not consume a rank, since RRF maps rank directly to score.
            rank = len(results) + 1

            # COPY before stamping: `meta` is the dict object handed back by
            # Mem0Backend.search, i.e. the caller's own response structure.
            # Stamping into it would mutate that response in place.  The cosine
            # is stored raw and un-clamped — store_score is a plain dict value
            # with no pydantic bound, and clamping would corrupt the honest
            # per-store signal.  (The old min(score, 1.0) existed only to
            # satisfy MemoryResult.relevance_score's le=1.0; the RRF value is
            # <= 1/61, so that clamp is no longer needed there either.)
            metadata = dict(meta)
            metadata['store_rank'] = rank
            metadata['store_score'] = score

            results.append(MemoryResult(
                id=item.get('id', ''),
                content=content,
                category=category,
                source_store=SourceStore.mem0,
                relevance_score=_rrf_score(rank),
                metadata=metadata,
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
        return {'id': memory_id, 'content': _mem0_content(payload), 'metadata': payload}

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

    async def list_child_ids(
        self, memory_id: str, *, project_id: str
    ) -> DescendantScan:
        """*memory_id*'s DIRECT children — one level deep, WITHOUT deleting.

        Deliberately not :meth:`list_descendant_ids`' transitive post-order
        walk.  The consolidate op re-points a victim's immediate children
        onto the new canonical and then deletes only that victim; its
        grandchildren keep a living parent throughout and are never touched.
        Enumerating them here would invite reparenting records whose own
        parent is still alive — a pointer rewrite nothing asked for.

        Public and side-effect-free for the same reason
        :meth:`refuse_if_children` is: the tool layer has to pre-flight
        "what would I have to reparent?" BEFORE the destructive part of the
        operation starts.

        Built from the same :meth:`_count_children` / :meth:`_list_children`
        primitives as every other child read (INV-5) — no new scroll — so
        the count-then-scroll ordering, the ``_CHILD_SCAN_LIMIT`` bound and
        the truncation semantics keep exactly one home.  Count first: the
        count is exact and cheap while the scroll fetches full payloads, and
        a childless victim (the common case) must not pay for a listing
        with nothing in it.

        Returns:
            DescendantScan: ``ids`` in scroll order, bounded by
            ``_CHILD_SCAN_LIMIT``; and ``truncated``, true when the live
            count exceeds what the scroll returned — the bound above, or a
            concurrent write landing between the two reads.  A caller that
            reparents a truncated listing and then deletes the parent
            silently orphans everything it could not see, so the
            disagreement is carried as data rather than reconciled toward
            the smaller answer.
        """
        count = await self._count_children(memory_id, project_id=project_id)
        if not count:
            return DescendantScan(ids=[], truncated=False)
        ids = await self._list_children(memory_id, project_id=project_id)
        return DescendantScan(ids=ids, truncated=len(ids) < count)

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

        METADATA VOCABULARY is the exception, and belongs HERE (task 3523).  A
        patch runs :func:`_apply_memory_metadata_validation` at this seam, the
        same one ``add_memory`` and ``add_system_record`` use, for the reason
        PRD D8 pins enforcement at the service layer: a tools-layer validator
        leaks past every additional write path, and this was the third one.
        Placed after the §5(c) existence check and before every journaled
        backend call, so a rejection cannot leave a journal row, a partial
        write, or a pending mem0 intent behind.

        Two properties of that check are deliberate and easy to "simplify" away:

        * It is DELTA-scoped — only violations and ``canonical`` claims NEW
          relative to the record's pre-image are judged.  Amending a record
          never re-validates the record.  See ``baseline`` on the seam.
        * A CONTENT-ONLY amend does not run it at all.  Such a write leaves the
          metadata byte-identical, so there is nothing it is responsible for.
          That reason stands unaided; the consequence of getting it wrong is
          that a legacy record's TEXT would become uncorrectable under
          ``enforce`` because of metadata the amend never touched.

        Returns the ``{'status': 'updated', 'store': 'mem0', 'id': memory_id,
        ...}`` envelope on success, or a structured ``{'error_type': ...}``
        rejection. The id is echoed so a caller can assert identity stability
        straight from the response instead of re-fetching.

        A vocabulary rejection is the one outcome that does NOT use that
        envelope: :class:`MemoryMetadataValidationError` and
        :class:`CanonicalUniquenessViolation` PROPAGATE from here, exactly as
        they do from ``add_memory``.  PRD V1 keeps the two deliberately
        distinguishable at an ``except`` (neither subclasses the other), and
        flattening them into ``error_type`` strings at this layer would discard
        their structured fields — the incumbent id a caller needs in order to
        act.  The MCP tool above converts every exception to an
        ``{'error', 'error_type'}`` envelope via ``@mcp_tool_errors()``, and
        does so identically for all three write paths.

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
        # The set_payload / delete_payload fast paths read it for VALUES only
        # (task 3523 — so the seam's `supersedes` scalar→list normalization is
        # not lost on this route), never for merge / delete SEMANTICS: they
        # still name only the patch keys / key list and let Qdrant apply the
        # merge and the delete SERVER-side, which is the entire reason those
        # routes can skip a read-modify-write. So the INV-5 single-home claim is
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

        # Mem0 metadata vocabulary validation on the THIRD write path (task
        # 3523). PRD D8/§2 pin enforcement at this seam precisely because a
        # second write path leaks past a tools-layer validator; update_memory
        # is a third one and reproduced exactly that leak.
        #
        # Placement mirrors add_memory's (see the note at its call site):
        # AFTER the §5(c) read leg's existence check and the delta, so the
        # EFFECTIVE post-patch custom subset is what gets judged; BEFORE
        # `scope` and every _journaled_backend_call below, so a rejection can
        # never leave a journal row or a half-applied patch behind.
        #
        # GATED ON A METADATA DELTA EXISTING. A content-only amend leaves the
        # record's metadata byte-identical, so this write is responsible for
        # none of it; validating it anyway would be corpus re-validation by
        # another name. That first-principles reason is the whole
        # justification and stands unaided — do not prop it up with a named
        # repair sweep: no in-repo sweep drives this arm (grepped — the only
        # callers of MemoryService.update_memory are the MCP tool and
        # scripts/retro_stamp_topics.py, and the latter never amends
        # content). The consequence of getting it wrong is nonetheless real:
        # under `enforce` a legacy record's TEXT would become uncorrectable
        # because of unrelated legacy metadata, and `enforce` would quietly
        # restate from "rejects WRITES" to "re-validates the corpus", the
        # model task 3626's flip measurement depends on. It also keeps the
        # seam's cost off the one arm that already pays for a re-embed.
        if metadata_patch or metadata_delete_keys:
            await _apply_memory_metadata_validation(
                new_custom,
                project_id=project_id,
                agent_id=agent_id,
                config=self.config.memory_metadata,
                storm_detector=self._metadata_storm_detector,
                project_root=self._memory_metadata_project_root(),
                parent_lookup=self.get_memory_by_id,
                count_canonical=self.count_memories_by_metadata,
                find_canonical=self.get_memories_by_metadata,
                # The record's PRE-IMAGE, free from the §5(c) read leg above.
                # Only violations NEW relative to it are this write's problem.
                baseline=existing_custom,
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
                #
                # The VALIDATED values for the patch keys, not the raw patch
                # (task 3523): the vocabulary seam normalizes in place —
                # `supersedes` scalar→list, PRD D2 — and writing the raw patch
                # here would persist the legacy scalar on this route while the
                # overwrite and content arms persisted a list. Restricted to
                # the patch keys, so the fast path keeps its whole point:
                # Qdrant still merges server-side and no pre-image key the
                # caller did not name is rewritten.
                #
                # UNFILTERED on purpose, and LOUD if that ever stops holding.
                # This branch is reached only for `metadata_mode == 'merge'`
                # with no delete keys, so `_apply_metadata_delta`'s
                # `new_custom.update(metadata_patch)` puts every patch key in
                # and the seam only ever ASSIGNS (`meta['supersedes'] =
                # members`), never pops — so `missing` is unreachable today.
                # Were a future normalizer to drop a key, an `if k in
                # new_custom` filter would silently skip it here while
                # set_payload merged server-side and LEFT THE OLD VALUE in
                # Qdrant, whereas the overwrite and content arms would drop
                # it: the three-route split this whole change closed,
                # reopened silently. Raising costs the write and names the
                # divergence, which is the house's loud-over-silent norm; it
                # happens before the coroutine is built, so no journal row
                # and no un-awaited coroutine are left behind.
                missing = [k for k in metadata_patch if k not in new_custom]
                if missing:
                    raise RuntimeError(
                        'update_memory: the metadata vocabulary seam removed '
                        f'patch key(s) {missing!r} from the effective metadata; '
                        'the set_payload fast path cannot express a key REMOVAL '
                        '(Qdrant merges server-side), so this write would leave '
                        'the stale value in place while the overwrite and '
                        'content arms would drop it. Route key-removing '
                        'normalization through the overwrite arm instead.'
                    )
                operation = 'update_memory_set_payload'
                coro = self.mem0.set_payload(
                    memory_id,
                    {k: new_custom[k] for k in metadata_patch},
                    scope,
                )
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
