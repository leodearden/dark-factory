"""Stale priority-override / pin-queue Graphiti edge sweep — task 2781.

Stage 1 (``MemoryConsolidator``) has no deterministic sweep that keeps VALID
(``invalid_at IS NULL``) priority-override / pin-queue Graphiti temporal_facts
edges (boost tier, TTL, pin order) in sync with live scheduler-override state.
When a task's override is consumed (task dispatched -> row cleared by the
scheduler's ``clear_terminal``) or expires (``clear_expired``), the task drops
out of the live override table but its "task N has boost/TTL/pin override" edge
silently persists as a stale valid_at edge until a human catches it — this has
now required TWO one-time manual backfills (run 2d59c7de, finding 3852bd07:
c583e636/task 5166 boost, f4833e2d/task 4940 TTL, b58b3b23/task 4079 pin
order). Distinct from the task 2319/2351 WRITE-TIME under-invalidation guard
(``_invalidate_stale_superseded_ttl_edges`` in memory_service.py), which only
fires when a NEW conflicting episode is written — a consumed/expired override
with no new write never trips it. This module is the recurring deterministic
sweep that closes that gap, mirroring task 2613's
``stale_status_snapshot_edge_sweep`` precedent exactly (pure lexical
extractor(s) + pure decision core + best-effort async orchestrator).

Design decisions (captured in plan.json):

- Live-state source is the FULL ``scheduler_overrides.db`` table (all rows),
  NOT ``get_pin_queue``'s ``pinned=1`` projection that the task literally
  names. ``get_pin_queue`` returns only ``WHERE pinned=1`` rows, so a
  non-pinned boost_tier override (``set_task_priority_override(boost_tier=...)``
  with pinned unset) would be wrongly seen as "absent from the pin queue" and
  its LIVE edge invalidated — over-invalidation that violates the
  never-retire-a-valid-edge invariant of this edge-class lineage
  (2111/2319/2351/2613). Reading the full overrides table is a strict superset
  of ``get_pin_queue`` and makes "task absent from ALL live overrides" a
  positive, fail-safe consumed/cleared signal (``clear_terminal`` /
  ``clear_expired`` delete consumed/expired rows, so absence is authoritative).
- The TTL-expiry check uses the live override row's absolute ``ttl_until``, not
  the edge's ``valid_at + ttl_secs``: ``get_all_valid_edges`` returns edge
  dicts with only ``{uuid, fact, name}`` (no valid_at), and the overrides row
  already stores ``ttl_until`` as the authoritative absolute expiry the
  scheduler's ``clear_expired`` honors.
- Staleness is positively-determinable-only: (task absent from live overrides)
  OR (TTL edge whose live ``ttl_until`` has elapsed). A task still present with
  any live override never selects — conservative under-invalidation
  (self-heals next cycle) is the correct bias for an irreversible
  invalidation, mirroring 2613's invalidate-only-on-positively-terminal
  fail-safe.
- Subject task_id is extracted lexically from LLM-generated edge fact text:
  gate on the "priority override" phrase, then a SINGLE ``TASK_REF_RE``
  subject id; multi-subject and no-subject facts are never candidates
  (mirrors 2613's count-only exclusion).
- The live overrides-DB read is replicated in-module via
  ``shared.async_sqlite_base`` plumbing (no orchestrator import, no
  ``server.tools`` import — that would create an import cycle), injectable
  into the orchestrator for hermetic tests (mirroring how 2613 injects
  ``taskmaster``).

Best-effort throughout (modelled on
``stale_status_snapshot_edge_sweep.sweep_stale_status_snapshot_edges``): a
transient read failure aborts the cycle no-op (self-heals next cycle); a
per-edge ``update_edge`` failure does NOT abort the loop;
``asyncio.CancelledError`` / ``KeyboardInterrupt`` / ``SystemExit`` are
re-raised unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from shared.async_sqlite_base import apply_full_durability_pragmas, connect_daemon

from fused_memory.models.scope import resolve_main_checkout
from fused_memory.reconciliation.stale_status_snapshot_edge_sweep import (
    _SWEEP_AGENT_ID,
    flatten_dedup_edges,
)
from fused_memory.reconciliation.task_filter import TASK_REF_RE

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# extract_priority_override_task_id — pure lexical extraction
# --------------------------------------------------------------------------- #

# Gate: a priority-override edge must contain the "priority override" phrase
# (separator-tolerant: one-or-more hyphen/whitespace chars between the two
# words, so LLM-generated free text with double spaces/newlines still
# matches). A fact without this phrase is not a priority-override edge this
# sweep concerns itself with at all. Source of truth for the phrase grammar:
# services/memory_service.py::_PRIORITY_OVERRIDE_TTL_FACT_RE (the write-time
# guard anchors on the identical "priority[-\s]+override" phrase).
_PRIORITY_OVERRIDE_GATE_RE: re.Pattern[str] = re.compile(
    r'\bpriority[-\s]+override\b',
    re.IGNORECASE,
)


def extract_priority_override_task_id(fact: str) -> int | None:
    """Return the single subject task_id *fact* asserts a priority override for.

    Returns None (never a candidate for invalidation) when:
    - *fact* contains no "priority override" phrase at all (e.g. 'Task 5 is
      done', 'Reordered pin queue: [1, 2, 3]') — the gate fails; or
    - *fact* has no single distinct extractable subject: either zero
      ``TASK_REF_RE`` ids, or two-or-more distinct ids (a multi-subject event
      record). Requiring exactly one distinct subject excludes multi-id
      records from candidacy, mirroring 2613's count-only exclusion.

    Algorithm:
      1. Gate on ``_PRIORITY_OVERRIDE_GATE_RE`` against the raw fact text;
         short-circuit to None when absent.
      2. Collect ``TASK_REF_RE`` subject ids ('task N' / 'df N' / '#N' — the
         shared task-reference grammar 2613 and task_filter already use, so
         this stays in sync if that grammar changes). Return the id iff
         exactly one distinct id is present, else None. Bare digits without a
         task/df/# prefix (dates, TTL seconds, pin orders) are never
         subjects.

    Pure: no I/O, no side effects.
    """
    fact = fact or ''
    if not _PRIORITY_OVERRIDE_GATE_RE.search(fact):
        return None

    ids = {int(m.group(1)) for m in TASK_REF_RE.finditer(fact)}
    if len(ids) != 1:
        return None
    return next(iter(ids))


# --------------------------------------------------------------------------- #
# is_ttl_override_fact — pure classifier
# --------------------------------------------------------------------------- #

# Requires BOTH a "priority[-\s]+override" phrase AND a "TTL" token, in either
# order (re.I | re.S so the two tokens may be separated by newlines and the
# phrase's own separator tolerates double spaces / newlines). Parallel matcher
# to — source of truth — services/memory_service.py::_PRIORITY_OVERRIDE_TTL_FACT_RE
# (the write-time under-invalidation guard). Kept as a module-local mirror
# rather than cross-importing the private service helper into reconciliation,
# so the TTL-fact grammar stays consistent with the write-time guard without a
# layering violation. Requiring both tokens restricts this classifier to the
# genuinely single-valued TTL scalar — a boost/pin edge (no "TTL" token) or a
# bare-TTL non-override fact (no phrase) never matches.
_TTL_OVERRIDE_FACT_RE: re.Pattern[str] = re.compile(
    r'\bpriority[-\s]+override\b.*\bTTL\b|\bTTL\b.*\bpriority[-\s]+override\b',
    re.I | re.S,
)


def is_ttl_override_fact(fact: str) -> bool:
    """Return True iff *fact* expresses a priority-override TTL value.

    Requires BOTH a ``priority[-\\s]+override`` phrase AND a ``TTL`` token
    (case-insensitive, either order, separator-tolerant) — see
    ``_TTL_OVERRIDE_FACT_RE``. A boost-tier or pin-order priority-override
    fact (no ``TTL`` token) and an unrelated bare-``TTL`` fact (no phrase)
    both return False.

    Pure: no I/O, no side effects.
    """
    return bool(fact) and _TTL_OVERRIDE_FACT_RE.search(fact) is not None


# --------------------------------------------------------------------------- #
# select_stale_priority_override_edges — pure decision core
# --------------------------------------------------------------------------- #


def select_stale_priority_override_edges(
    edges: list[dict],
    live_overrides: dict[str, dict],
    *,
    now: datetime,
) -> list[dict]:
    """Return the subset of *edges* whose priority override is now stale.

    An edge is selected (stale) iff its extracted subject task (via
    ``extract_priority_override_task_id``) satisfies EITHER:
    - the task is ABSENT from *live_overrides* — its override row was consumed
      (dispatch -> ``clear_terminal``) or expired-and-cleared
      (``clear_expired``), so the edge no longer describes any live override;
      OR
    - the edge is a TTL edge (``is_ttl_override_fact``) whose live row has an
      absolute ``ttl_until`` that has already elapsed (``now >= ttl_until``).

    Positively-determinable-only / conservative under-invalidation: a task
    still present with a live override (and a null or not-yet-elapsed
    ``ttl_until``) is never selected. A transient read gap or still-live
    override can only UNDER-select (self-heals next cycle), never wrongly
    retire a valid edge — the correct fail-safe bias for an irreversible
    invalidation, mirroring 2613's invalidate-only-on-positively-terminal.

    Args:
        edges: Candidate edges (as returned by ``flatten_dedup_edges``).
        live_overrides: ``{task_id_str: {'ttl_until': datetime | None, ...}}``
            live override state (as returned by ``read_live_override_state``).
        now: Tz-aware UTC instant the TTL-elapsed comparison is made against.

    Pure: no I/O, no side effects.
    """
    selected: list[dict] = []
    for edge in edges:
        fact = edge.get('fact') or ''
        tid = extract_priority_override_task_id(fact)
        if tid is None:
            continue
        row = live_overrides.get(str(tid))
        if row is None or (
            is_ttl_override_fact(fact)
            and row.get('ttl_until') is not None
            and now >= row['ttl_until']
        ):
            selected.append(edge)
    return selected


# --------------------------------------------------------------------------- #
# read_live_override_state — live scheduler-override reader
# --------------------------------------------------------------------------- #


def _overrides_db_path(project_root: str) -> Path:
    """Return the canonical scheduler_overrides.db path for *project_root*.

    Source of truth: ``server/tools.py::_overrides_db_path`` — hand-mirrored
    here (not imported) because importing ``server.tools`` into
    ``reconciliation`` would create an import cycle (``server.tools`` imports
    ``reconciliation``), and fused-memory has no orchestrator dependency.
    """
    return Path(project_root) / 'data' / 'orchestrator' / 'scheduler_overrides.db'


def _canonical_project_root(project_root: str) -> str:
    """Normalize *project_root* to the canonical form the override WRITE path
    stores rows under.

    ``set_task_priority_override`` funnels ``project_root`` through
    ``server/tools.py::_normalize_project_root`` -> ``resolve_main_checkout``
    before BOTH opening ``scheduler_overrides.db`` and writing each row's
    ``project_root`` column, so a persisted row's key is always the resolved
    *main*-checkout path — never a worktree subpath, trailing-slash, or
    symlinked spelling. This reader MUST normalize identically: a mismatched
    ``WHERE project_root=?`` filter (or DB-file path) silently returns zero
    rows, ``read_live_override_state`` returns ``{}``, and
    ``select_stale_priority_override_edges`` then treats EVERY valid
    boost/pin/TTL edge as "task absent" — the exact mass over-invalidation of
    valid edges (an irreversible write) this module exists to prevent.

    Mirrors ``_normalize_project_root``: prefer ``resolve_main_checkout`` (the
    write path's exact normalizer — it maps any worktree subpath to its main
    checkout, the key the row was actually written under). Falls back to
    ``str(Path(project_root).resolve())`` when ``resolve_main_checkout``
    cannot run (``ValueError``: the path is not inside a git working tree, or
    ``git`` is unavailable — e.g. a hermetic temp-dir test), which still
    canonicalizes a trailing slash, a ``.``/``..`` segment, or a symlinked
    ancestor. Pure apart from the ``git`` subprocess ``resolve_main_checkout``
    may spawn (cached there by resolved input path).
    """
    try:
        return resolve_main_checkout(project_root)
    except ValueError:
        return str(Path(project_root).resolve())


async def read_live_override_state(project_root: str) -> dict[str, dict]:
    """Read live scheduler-override state from ``scheduler_overrides.db``.

    Reads the FULL overrides table for *project_root* — deliberately WITHOUT
    ``get_pin_queue``'s ``pinned=1`` filter (see the module docstring's
    live-state-source design decision) — so a task absent from the returned
    map is a positive, fail-safe "override consumed/cleared" signal rather
    than merely "not pinned".

    *project_root* is first normalized via ``_canonical_project_root`` so both
    the DB-file lookup and the ``WHERE project_root=?`` filter key on the same
    resolved main-checkout path the override write path persisted rows under —
    guarding against the mass over-invalidation a non-canonical root (worktree
    subpath, trailing slash, symlink) would otherwise trigger (see that
    helper's docstring).

    Returns ``{}`` when the DB file does not exist (no overrides recorded
    yet). Otherwise returns ``{str(task_id): {'boost_tier', 'pinned',
    'pin_order', 'reserve_now', 'ttl_until'}}`` where ``ttl_until`` is parsed
    to a tz-aware ``datetime`` (or ``None`` when the column is NULL).

    Opens the DB via ``connect_daemon`` + ``apply_full_durability_pragmas``
    (the same ``shared.async_sqlite_base`` plumbing ``server/tools.py`` uses),
    closing the connection in ``finally``. May raise on a transient DB error;
    the caller (``sweep_stale_priority_override_edges``) treats that
    best-effort — aborting the cycle no-op, self-healing next cycle.
    """
    project_root = _canonical_project_root(project_root)
    db_path = _overrides_db_path(project_root)
    if not db_path.exists():
        return {}

    db = await connect_daemon(str(db_path))
    try:
        await apply_full_durability_pragmas(db, busy_timeout_ms=5000)
        cursor = await db.execute(
            'SELECT task_id, boost_tier, pinned, pin_order, reserve_now, ttl_until '
            'FROM overrides WHERE project_root=?',
            (project_root,),
        )
        rows = await cursor.fetchall()
    finally:
        await db.close()

    live: dict[str, dict] = {}
    for task_id, boost_tier, pinned, pin_order, reserve_now, ttl_until in rows:
        live[str(task_id)] = {
            'boost_tier': boost_tier,
            'pinned': bool(pinned),
            'pin_order': pin_order,
            'reserve_now': bool(reserve_now),
            'ttl_until': datetime.fromisoformat(ttl_until) if ttl_until else None,
        }
    return live


# --------------------------------------------------------------------------- #
# sweep_stale_priority_override_edges — async orchestrator
# --------------------------------------------------------------------------- #


async def sweep_stale_priority_override_edges(
    memory_service,
    project_id: str,
    project_root: str,
    *,
    run_id: str,
    read_live: Callable[[str], Awaitable[dict[str, dict]]] | None = None,
    now: datetime | None = None,
    log: logging.Logger = logger,
) -> dict:
    """Enumerate valid priority-override edges and invalidate the stale ones.

    Enumerates ALL currently-valid Graphiti edges for *project_id* via
    ``memory_service.graphiti.get_all_valid_edges`` (a deterministic bulk
    query — never the LLM's semantic search), lexically extracts each edge's
    single subject task_id, reads LIVE override state via *read_live*, and
    invalidates (``memory_service.update_edge(..., invalid_at=...)``) every
    edge whose task is absent from the live override table OR whose (TTL)
    edge's live ``ttl_until`` has elapsed.

    Args:
        memory_service: Object exposing ``.graphiti.get_all_valid_edges`` and
            ``.update_edge``.
        project_id: Graphiti group_id to enumerate and invalidate within.
        project_root: Project root whose ``scheduler_overrides.db`` supplies
            live override state.
        run_id: Reconciliation run id, threaded through as update_edge's
            causation_id.
        read_live: Async live-override reader; defaults to
            ``read_live_override_state``. Injected in tests (mirroring how
            2613 injects ``taskmaster``). Resolved at call time (not as a
            bound default) so a monkeypatch of the module-level
            ``read_live_override_state`` is honored by the default path.
        now: Invalidation + TTL-comparison timestamp; defaults to
            ``datetime.now(UTC)``.
        log: Logger to use (default: this module's logger).

    Returns:
        dict with int counts: ``scanned`` (edges enumerated after dedup),
        ``candidate_edges`` (edges that passed the priority-override lexical
        gate — the pre-selection candidate pool), ``stale_selected`` (that
        subset ``select_stale_priority_override_edges`` judged stale),
        ``invalidated`` (successful update_edge calls), ``errors`` (caught
        failures). The scanned -> candidate_edges -> stale_selected ->
        invalidated funnel is monotonically narrowing, so each is a distinct,
        self-describing quantity.
    """
    stats = {
        'scanned': 0,
        'candidate_edges': 0,
        'stale_selected': 0,
        'invalidated': 0,
        'errors': 0,
    }

    if not project_root:
        return stats

    if read_live is None:
        read_live = read_live_override_state

    # Enumeration is a single bulk call the rest of the cycle depends on: a
    # failure here ends this cycle's sweep early with the stats gathered so
    # far (self-heals next cycle). CancelledError/KeyboardInterrupt/SystemExit
    # are never swallowed.
    try:
        grouped = await memory_service.graphiti.get_all_valid_edges(group_id=project_id)
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        log.exception(
            'stale_priority_override_edge_sweep: get_all_valid_edges failed for group_id=%s',
            project_id,
        )
        stats['errors'] += 1
        return stats

    edges = flatten_dedup_edges(grouped)
    stats['scanned'] = len(edges)

    candidate_edges = [
        edge
        for edge in edges
        if extract_priority_override_task_id(edge.get('fact') or '') is not None
    ]
    # True pre-selection candidate-pool size — set on EVERY exit path (incl.
    # the no-candidates short-circuit and a subsequent read_live failure), so
    # the stat always describes "priority-override edges seen", never the
    # narrower selected-as-stale count (that is `stale_selected`, below).
    stats['candidate_edges'] = len(candidate_edges)

    # No priority-override edges at all -> no live-state read needed.
    if not candidate_edges:
        return stats

    # Reading live override state is the other bulk call the invalidation
    # decision depends on: a failure aborts this cycle no-op (self-heals next
    # cycle). CancelledError/KeyboardInterrupt/SystemExit are never swallowed.
    try:
        live = await read_live(project_root)
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        log.exception(
            'stale_priority_override_edge_sweep: read_live failed for project_root=%s',
            project_root,
        )
        stats['errors'] += 1
        return stats

    invalidate_at = now or datetime.now(UTC)
    stale = select_stale_priority_override_edges(candidate_edges, live, now=invalidate_at)
    stats['stale_selected'] = len(stale)

    for edge in stale:
        try:
            await memory_service.update_edge(
                edge['uuid'],
                invalid_at=invalidate_at,
                project_id=project_id,
                agent_id=_SWEEP_AGENT_ID,
                causation_id=run_id,
            )
            stats['invalidated'] += 1
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            log.exception(
                'stale_priority_override_edge_sweep: update_edge failed for uuid=%s',
                edge['uuid'],
            )
            stats['errors'] += 1

    return stats
