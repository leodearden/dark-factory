"""Core orchestration layer — owns backends, classifier, router, durable queue."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
import uuid as uuid_mod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from graphiti_core.nodes import EpisodeType

from fused_memory.backends.graphiti_client import GraphitiBackend
from fused_memory.backends.mem0_client import Mem0Backend
from fused_memory.config.schema import FusedMemoryConfig
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
from fused_memory.routing.classifier import WriteClassifier
from fused_memory.routing.router import ReadRouter
from fused_memory.services.durable_queue import DurableWriteQueue
from fused_memory.utils.async_utils import gather_collect, gather_or_raise
from fused_memory.utils.task_naming import canonicalize_task_node_name

if TYPE_CHECKING:
    from fused_memory.backends.task_backend_protocol import TaskBackendProtocol
    from fused_memory.reconciliation.event_buffer import EventBuffer
    from fused_memory.reconciliation.recon_ledger import ReconLedgerStore
    from fused_memory.services.planned_episode_registry import PlannedEpisodeRegistry
    from fused_memory.services.write_journal import WriteJournal

logger = logging.getLogger(__name__)

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


class SearchResults(list):
    """list subclass returned by MemoryService.search carrying in-band degrade metadata.

    All ~11 internal list-consuming callers (context_assembler, targeted, flag_dedup,
    mem0_dedup, task_knowledge_sync) keep working unchanged — only tools.py reads the
    extra attributes (task 1812).

    Attributes:
        degraded: True when one or more selected stores raised or timed out.
        failed_stores: List of store name strings (SourceStore.value) that failed.

    .. warning::
        The `degraded` and `failed_stores` metadata do **not** survive list-returning
        operations (slicing, sorted(), concatenation, list comprehensions).  Those
        operations return a plain ``list``, silently dropping the degrade metadata.
        Callers that need the metadata after a transform should read the attributes
        *before* the transform, or pass the SearchResults object directly without
        intermediate list operations.
    """

    def __init__(self, iterable=(), *, degraded: bool = False, failed_stores=None):
        super().__init__(iterable)
        self.degraded = degraded
        self.failed_stores: list[str] = failed_stores if failed_stores is not None else []


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
        # Process-start baselines for uptime reporting
        self._started_at: datetime = datetime.now(UTC)
        self._start_monotonic: float = time.monotonic()

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
        )
        self.durable_queue.register_callback(
            'dual_write_episode', self._dual_write_callback
        )
        self.durable_queue.register_callback(
            'refresh_entity_summaries', self._refresh_summaries_callback
        )
        await self.durable_queue.initialize()

        logger.info('MemoryService initialized')

    async def _safe_close(self, label: str, resource: Any) -> None:
        """Close one resource, logging any failure without re-raising.

        Lets close() continue through every resource even when one fails.
        """
        try:
            await resource.close()
        except Exception:
            logger.exception('MemoryService.close: %s.close failed', label)

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

        # Log Layer 1 for the queued write
        if self._write_journal:
            await self._write_journal.log_write_op(
                write_op_id=write_op_id or str(uuid_mod.uuid4()),
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
                success=True,
            )

        return result

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

        # Server-side cycle_summary metadata tagging (recon_pool auto-tag
        # task 2077, run_id auto-backfill task 2109, missing-key warning
        # task 2094/2109) — factored into a shared helper (task 2222
        # amendment) so add_system_record gets the identical authoritative
        # treatment. See _apply_cycle_summary_metadata_tagging's docstring.
        _apply_cycle_summary_metadata_tagging(meta, causation_id, project_id=project_id)

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

        # Same authoritative cycle_summary tagging add_memory applies (task
        # 2222 amendment): the tool docstring names the cycle-summary Mem0
        # mirror as the intended caller, and recon_pool/run_id are the keys
        # the pool-cap trim and Path-2 triple-filter pre-check rely on — a
        # system-record cycle_summary must not go untagged just because it
        # bypassed add_memory.
        _apply_cycle_summary_metadata_tagging(meta, causation_id, project_id=project_id)

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

            for i, t in enumerate(task_list):
                if t not in done:
                    continue
                try:
                    store_results = t.result()
                    results.extend(store_results)
                except Exception as e:
                    logger.warning(
                        'search.store_failed',
                        extra={'store': store_list[i].value, 'error': str(e)},
                    )
                    failed_stores.append(store_list[i])

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
        resolved node uuid(s), uncapped.

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
        edge_data = [_edge_to_dict(e) for e in edges]

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
        """
        scope = Scope(project_id=project_id)
        return await self.mem0.count_by_metadata(scope, filters)

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
        """
        scope = Scope(project_id=project_id)
        return await self.mem0.scroll_by_metadata(scope, filters, limit)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_memory(
        self,
        memory_id: str,
        store: str,
        project_id: str = 'main',
        agent_id: str | None = None,
        session_id: str | None = None,
        causation_id: str | None = None,
        _source: str = 'mcp_tool',
    ) -> dict:
        """Delete a memory from the specified store."""
        scope = Scope(project_id=project_id)
        source = SourceStore(store)
        write_op_id = str(uuid_mod.uuid4())

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
            del_result = await self._journaled_backend_call(
                write_op_id=write_op_id,
                causation_id=causation_id,
                backend='mem0',
                operation='delete',
                payload={'memory_id': memory_id},
                coro=self.mem0.delete(memory_id, scope),
            )
            result = {'status': 'deleted', 'store': 'mem0', 'id': memory_id, **del_result}

        if self._write_journal:
            await self._write_journal.log_write_op(
                write_op_id=write_op_id,
                causation_id=causation_id,
                source=_source,
                operation='delete_memory',
                project_id=project_id,
                agent_id=agent_id,
                session_id=session_id,
                params={'memory_id': memory_id, 'store': store},
                result_summary=result,
                success=True,
            )

        await self._emit_event(ReconciliationEvent(
            id=str(uuid_mod.uuid4()),
            type=EventType.memory_deleted,
            source=EventSource.agent,
            project_id=project_id,
            timestamp=datetime.now(UTC),
            payload={'memory_id': memory_id, 'store': store},
        ))

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

        return result

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
