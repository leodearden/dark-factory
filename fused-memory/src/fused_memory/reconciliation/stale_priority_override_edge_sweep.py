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

import re
from datetime import datetime

from fused_memory.reconciliation.task_filter import TASK_REF_RE

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
        if row is None:
            selected.append(edge)
        elif (
            is_ttl_override_fact(fact)
            and row.get('ttl_until') is not None
            and now >= row['ttl_until']
        ):
            selected.append(edge)
    return selected
