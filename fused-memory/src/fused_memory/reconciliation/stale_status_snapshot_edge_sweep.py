"""Stale task-status snapshot Graphiti edge sweep — task 2613.

Retires VALID (``invalid_at IS NULL``) Graphiti edges whose asserted
active/pending/blocked/stalled/in-progress status is contradicted by the
task's live status, read by a direct ``taskmaster.get_statuses`` lookup,
never by semantic search. Which ids an edge asserts, and under which marker,
is read by ``status_snapshot_extraction.py``.

Fail-safe direction: under-selection self-heals next cycle, while
over-selection is unrecoverable, since an invalidated Graphiti edge is never
re-validated. So an edge is retired only on a positively known status.

Design rationale, residuals and measurements:
``docs/stale-status-snapshot-sweep.md``.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from datetime import UTC, datetime
from typing import NamedTuple

from shared.task_statuses import TaskStatus

from fused_memory.backends.graphiti_client import apply_incompleteness_policy
from fused_memory.reconciliation.status_snapshot_extraction import (
    extract_blocked_assertion_task_ids,
    extract_snapshot_edge_task_ids,
    extract_snapshot_edge_task_ids_by_marker_class,
)
from fused_memory.reconciliation.task_filter import INACTIVE_TASK_STATUSES
from fused_memory.utils.validation import canonicalize_project_id

logger = logging.getLogger(__name__)

# The closed status vocabulary, frozen once at import: what "positively known"
# means for the blocked-assertion rule. A value outside it is UNKNOWN, never a
# contradiction.
_KNOWN_TASK_STATUSES: frozenset[str] = frozenset(TaskStatus)


# --------------------------------------------------------------------------- #
# build_supersede_fact — the superseding temporal_fact's wording
# --------------------------------------------------------------------------- #

# RESULTING-STATE-ONLY, stamped 'As of <date>'. Never pair it with the prior
# state: 'was blocked and is now pending' re-extracts as a blocked assertion
# and would churn forever, and 'is no longer blocked' is prior-state framing
# that no write guard catches. The sweep writes in-process, bypassing the MCP
# boundary's framing guards, so this template and TestBuildSupersedeFact are
# the only defence. See docs/stale-status-snapshot-sweep.md
# §"The superseding write".
_SUPERSEDE_FACT_TEMPLATE = 'As of {date}, task {task_id} has status {status}.'


# Per-cycle ceiling on superseding add_memory ATTEMPTS, which are serial
# network writes inside the invalidation loop. The excess is tallied into
# stats['supersede_skipped'] and logged; those facts are lost, not deferred.
# See docs/stale-status-snapshot-sweep.md §"The per-cycle ceiling".
_MAX_SUPERSEDE_WRITES_PER_CYCLE = 50


def build_supersede_fact(task_id: int, status: str, now: datetime) -> str:
    """Render the resulting-state-only temporal_fact that supersedes a retired
    blocked assertion.

    Args:
        task_id: The task whose blocked assertion was contradicted.
        status: Its CURRENT census status — the resulting state, stated on its
            own. Never paired with the prior state; see
            ``_SUPERSEDE_FACT_TEMPLATE``.
        now: The timestamp to stamp the observation with: the same instant
            the sweep used as the edge's ``invalid_at``. Always passed in,
            never read from a clock.

    Pure: no I/O, no side effects.
    """
    return _SUPERSEDE_FACT_TEMPLATE.format(
        date=now.strftime('%Y-%m-%d'), task_id=task_id, status=status,
    )


# --------------------------------------------------------------------------- #
# flatten_dedup_edges
# --------------------------------------------------------------------------- #


def flatten_dedup_edges(grouped: dict[str, list[dict]]) -> list[dict]:
    """Flatten get_all_valid_edges' dict[entity_uuid, list[EdgeDict]] shape.

    ``GraphitiBackend.get_all_valid_edges``'s undirected MATCH pattern
    double-attributes each directed edge under both its source and target
    entity UUID. This flattens the grouping and dedups by ``edge['uuid']``,
    keeping the first-seen occurrence, so each edge is considered exactly
    once by the rest of the sweep.

    Pure: no I/O, no side effects.
    """
    seen: set[str] = set()
    result: list[dict] = []
    for edges in grouped.values():
        for edge in edges:
            uuid = edge['uuid']
            if uuid in seen:
                continue
            seen.add(uuid)
            result.append(edge)
    return result


# --------------------------------------------------------------------------- #
# select_stale_status_snapshot_edges — pure decision core
# --------------------------------------------------------------------------- #


def is_blocked_assertion_contradicted(status: str | None) -> bool:
    """Does *status* positively contradict a 'blocked' assertion?

    True iff *status* is a POSITIVELY KNOWN member of the closed
    ``shared.task_statuses.TaskStatus`` vocabulary and is not 'blocked'. An
    unrecognised value, or ``None`` (id absent from the census), is unknown,
    never a contradiction. See ``docs/stale-status-snapshot-sweep.md``
    §"Selection rules".

    Pure: no I/O, no side effects.
    """
    return status in _KNOWN_TASK_STATUSES and status != TaskStatus.BLOCKED


def select_stale_status_snapshot_edges(
    edges: list[dict],
    statuses: dict[str, str],
    *,
    edge_ids: dict[str, set[int]] | None = None,
    blocked_edge_ids: dict[str, set[int]] | None = None,
) -> list[dict]:
    """Return the subset of *edges* whose asserted status is now contradicted.

    TWO independent selection rules; an edge is selected (stale) when EITHER
    fires.

    1. TERMINAL rule. ``extract_snapshot_edge_task_ids`` returns at least one
       id AND any one of those ids resolves (via *statuses*) to a
       positively-terminal status (``INACTIVE_TASK_STATUSES`` —
       done/cancelled). This rule reads the UNION of every status marker.

    2. BLOCKED-ASSERTION rule. ``extract_blocked_assertion_task_ids`` returns
       at least one id AND any one of those ids resolves to a positively-known
       status that is not 'blocked' (``is_blocked_assertion_contradicted``).

    In both rules an aggregate/plural edge referencing several ids is selected
    as a whole the moment ANY single referenced id contradicts it. An id
    absent from *statuses*, or mapped outside the closed ``TaskStatus``
    vocabulary, never contributes to selection. Why rule 2 is a separate
    rule: ``docs/stale-status-snapshot-sweep.md`` §"Selection rules".

    Args:
        edges: Candidate edges (as returned by ``flatten_dedup_edges``).
        statuses: ``{task_id_str: status_str}`` census.
        edge_ids: Optional precomputed ``{edge['uuid']: extract_snapshot_edge_task_ids(fact)}``
            mapping, so a caller that already extracted ids does not re-run
            the extraction pipeline. When omitted, ids are computed from each
            edge's fact.
        blocked_edge_ids: Optional precomputed
            ``{edge['uuid']: extract_blocked_assertion_task_ids(fact)}``
            mapping, for rule 2, with the same semantics and the same default
            as *edge_ids*.

    Pure: no I/O, no side effects.
    """
    selected: list[dict] = []
    for edge in edges:
        # Each fall-back extracts independently, so a caller that precomputes
        # BOTH maps costs zero extraction-pipeline entries here.
        if edge_ids is not None:
            ids = edge_ids.get(edge['uuid'], set())
        else:
            ids = extract_snapshot_edge_task_ids(edge.get('fact'))
        if blocked_edge_ids is not None:
            blocked_ids = blocked_edge_ids.get(edge['uuid'], set())
        else:
            blocked_ids = extract_blocked_assertion_task_ids(edge.get('fact'))

        # Rule 1 — terminal, over UNION ids.
        if ids and any(statuses.get(str(i)) in INACTIVE_TASK_STATUSES for i in ids):
            selected.append(edge)
            continue
        # Rule 2 — blocked assertion contradicted, over BLOCKED-scoped ids only.
        if blocked_ids and any(
            is_blocked_assertion_contradicted(statuses.get(str(i))) for i in blocked_ids
        ):
            selected.append(edge)
    return selected


# --------------------------------------------------------------------------- #
# sweep_stale_status_snapshot_edges — async orchestrator
# --------------------------------------------------------------------------- #

# Written to update_edge's agent_id — identifies this sweep's writes in the
# write journal / audit trail, mirroring degenerate_task_node_sweep's
# convention of stamping a stable, identifiable actor.
_SWEEP_AGENT_ID = 'recon-stage-memory_consolidator'

# The ``report.stats`` names of this sweep's completeness pair, once
# MemoryConsolidator has prefixed the short keys ``enumeration_complete`` /
# ``enumeration_incomplete_kind`` that the sweep itself returns. Whole
# literals, not prefix concatenations, so a key stays greppable from a log
# line back to here.

STATUS_SNAPSHOT_ENUMERATION_COMPLETE_STAT_KEY: str = (
    'stale_status_snapshot_edges_enumeration_complete'
)
"""Key under Stage 1's ``report.stats``: this sweep's TRI-STATE read verdict.

``True`` = the edge enumeration was proven whole, ``False`` = a corpus was
observed and found incomplete, ``None`` = no corpus was observed. The only
safe predicate is ``is True``; ``is not False`` would admit the UNKNOWN case,
letting a cycle that never looked pass as one that looked and found
everything.

Conditional presence: ABSENT when the sweep itself raised (the stage swallows
that best-effort and sets NONE of its stats), which is honest in a way
``False`` would not be — ``False`` claims a corpus was observed. Read via
``report.stats.get(...)``, never direct indexing.
"""

STATUS_SNAPSHOT_ENUMERATION_INCOMPLETE_KIND_STAT_KEY: str = (
    'stale_status_snapshot_edges_enumeration_incomplete_kind'
)
"""Key under Stage 1's ``report.stats``: WHICH WAY the corpus was partial.

Carries the backend's stable ``INCOMPLETE_*`` discriminator (never
``PagedRead.reason``, whose wording the backend documents as deliberately
unstable), or ``None`` when there is nothing to discriminate. Conditional
presence exactly as for
:data:`STATUS_SNAPSHOT_ENUMERATION_COMPLETE_STAT_KEY` above.
"""


async def sweep_stale_status_snapshot_edges(
    memory_service,
    taskmaster,
    project_id: str,
    project_root: str,
    *,
    run_id: str,
    now: datetime | None = None,
    log: logging.Logger = logger,
) -> dict:
    """Enumerate valid status-snapshot edges and invalidate the stale ones.

    Enumerates ALL currently-valid Graphiti edges for *project_id* via
    ``memory_service.graphiti.enumerate_all_valid_edges``, extracts the task
    ids each edge asserts, cross-references those ids' CURRENT status via
    ``taskmaster.get_statuses``, and invalidates
    (``memory_service.update_edge(..., invalid_at=...)``) every edge
    ``select_stale_status_snapshot_edges`` selects. After a successful
    blocked-rule invalidation it also writes ONE superseding
    resulting-state-only ``temporal_facts`` memory per contradicted task per
    cycle (``build_supersede_fact``), capped at
    ``_MAX_SUPERSEDE_WRITES_PER_CYCLE``.

    Args:
        memory_service: Object exposing ``.graphiti.enumerate_all_valid_edges``,
            ``.update_edge`` and ``.add_memory``.
        taskmaster: Object exposing ``.get_statuses``. A falsy value (e.g.
            unavailable backend) short-circuits to all-zero stats.
        project_id: Graphiti group_id to enumerate and invalidate within.
        project_root: Taskmaster project root for the status lookup. A
            falsy value short-circuits to all-zero stats.
        run_id: Reconciliation run id, threaded through as update_edge's
            causation_id.
        now: Invalidation timestamp; defaults to ``datetime.now(UTC)``.
        log: Logger to use (default: this module's logger).

    Best-effort: an exception from ``enumerate_all_valid_edges``,
    ``get_statuses`` or ``update_edge`` is caught, logged and tallied into
    ``stats['errors']``. An enumeration or cross-reference failure ends this
    cycle's sweep and returns the stats so far; an ``update_edge`` failure
    does not stop the loop. An ``add_memory`` failure is caught, logged and
    tallied into ``stats['supersede_errors']`` and does not stop the loop
    either; it is retried in-cycle from the task's next stale edge, never
    next cycle. ``asyncio.CancelledError``/``KeyboardInterrupt``/
    ``SystemExit`` are re-raised unchanged from every path. See
    ``docs/stale-status-snapshot-sweep.md`` §"Sweep execution".

    Returns:
        dict with int counts: ``scanned`` (edges enumerated after dedup),
        ``candidate_edges`` (edges selected as stale by
        ``select_stale_status_snapshot_edges``), ``invalidated`` (successful
        update_edge calls), ``errors`` (caught enumerate/cross-reference/
        invalidate failures), ``superseded`` (successful superseding
        temporal_fact writes — blocked rule only), ``supersede_errors``
        (caught superseding-write failures), ``supersede_skipped`` (distinct
        contradicted tasks left without a superseding fact because
        ``_MAX_SUPERSEDE_WRITES_PER_CYCLE`` was reached).

        Plus two keys describing the READ that produced ``scanned`` rather
        than counting anything:

        ``enumeration_complete`` (``bool | None``) — TRI-STATE. ``True`` =
        the read was PROVEN complete; ``False`` = a corpus was observed and
        it was INCOMPLETE; ``None`` = NO corpus was observed at all, either
        because this call short-circuited on a falsy
        *taskmaster*/*project_root* or because the enumeration itself failed
        (``errors`` tells those two apart). The ONLY predicate a caller may
        gate on is ``is True``.

        ``enumeration_incomplete_kind`` (``str | None``) — the backend's
        ``INCOMPLETE_*`` constant (``graphiti_client.py::INCOMPLETE_PAGE_CAP``
        and siblings), the documented STABLE discriminator. Callers should
        branch on membership in
        ``graphiti_client.py::INCOMPLETE_STRUCTURAL_KINDS`` rather than on a
        specific kind. ``PagedRead.reason`` is deliberately NOT surfaced.

        ``invalidated == candidate_edges - errors`` holds exactly: every
        selected edge is either invalidated or tallied into ``errors`` by the
        ONE counted invalidation loop, and nothing else is folded into
        ``errors``. The two read keys and an empirical incompleteness leave
        it untouched.

    Empty *taskmaster*/*project_root* short-circuits to all-zero stats with
    no backend calls. When enumeration yields no candidate ids at all,
    ``taskmaster.get_statuses`` is never called.
    """
    return await _sweep_status_snapshot_edges(
        memory_service, taskmaster, project_id, project_root,
        task_id=None, run_id=run_id, now=now, log=log,
    )


async def sweep_stale_status_snapshot_edges_for_task(
    memory_service,
    taskmaster,
    project_id: str,
    project_root: str,
    task_id: int,
    *,
    run_id: str,
    now: datetime | None = None,
    log: logging.Logger = logger,
) -> dict:
    """Sweep only the status-snapshot edges that name *task_id*. (task 4851)

    A LATENCY complement to ``sweep_stale_status_snapshot_edges``, not a
    coverage one: the same selection rules and the same fail-safe, retiring
    an edge only on a status positively known from a LIVE
    ``taskmaster.get_statuses`` read, never from a transition payload. What
    it misses, the next periodic cycle still retires.

    The read is ``enumerate_valid_edges_mentioning(str(task_id))``, a
    digit-substring SUPERSET ('1420', dates), made exact by keeping only the
    edges whose extracted ids contain *task_id*. An aggregate edge naming
    *task_id* is judged on ALL its ids, exactly as the periodic sweep would
    judge it. A short or date-like id makes that read large; see
    ``docs/stale-status-snapshot-sweep.md`` §"The per-task sweep".

    Arguments, the best-effort contract and the returned stats are those of
    ``sweep_stale_status_snapshot_edges``, except that ``scanned`` counts the
    edges the narrowed read returned.
    """
    return await _sweep_status_snapshot_edges(
        memory_service, taskmaster, project_id, project_root,
        task_id=task_id, run_id=run_id, now=now, log=log,
    )


class _CorpusRead(NamedTuple):
    """What one sweep read observed. ``edges`` is None when nothing may be swept."""

    edges: list[dict] | None
    complete: bool | None
    incomplete_kind: str | None


async def _read_corpus(
    memory_service, project_id: str, task_id: int | None, *, log: logging.Logger,
) -> _CorpusRead:
    """THE READ: every valid edge, or only those whose fact contains *task_id*.

    ``enumerate_*`` never raises, so the fail-closed structural guard
    (``apply_incompleteness_policy``) is applied here, inside the same try,
    so a structural incompleteness lands in the one handler. An EMPIRICAL
    incompleteness only warns, through the injected *log*, and the sweep
    proceeds on what it got. Completeness is reported even when the policy
    aborts the read. See ``docs/stale-status-snapshot-sweep.md``
    §"Read completeness".
    """
    method = (
        'enumerate_all_valid_edges' if task_id is None
        else 'enumerate_valid_edges_mentioning'
    )
    paged = None
    try:
        graphiti = memory_service.graphiti
        if task_id is None:
            grouped, paged = await graphiti.enumerate_all_valid_edges(group_id=project_id)
        else:
            grouped, paged = await graphiti.enumerate_valid_edges_mentioning(
                str(task_id), group_id=project_id,
            )
        apply_incompleteness_policy(
            paged,
            method=method,
            group_id=project_id,
            returned_count=len(grouped),
            noun='entities',
            consequence='must not drive a staleness verdict',
            log=log,
        )
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        log.exception(
            'stale_status_snapshot_edge_sweep: %s failed for group_id=%s',
            method, project_id,
        )
        if paged is None:
            return _CorpusRead(None, None, None)
        return _CorpusRead(None, paged.complete, paged.incomplete_kind)
    return _CorpusRead(flatten_dedup_edges(grouped), paged.complete, paged.incomplete_kind)


# Per event loop, per canonical project: the lock that keeps two sweeps of one
# project from overlapping. See ``_sweep_lock_for``.
_SWEEP_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
] = weakref.WeakKeyDictionary()


def _sweep_lock_for(project_id: str) -> asyncio.Lock:
    """The lock every sweep of *project_id* holds, on the running loop.

    Each sweep must READ after the previous one's retirements, or two sweeps
    retire the same edge and both write its superseding facts. Keyed by the
    canonical project id and by event loop, because an ``asyncio.Lock`` binds
    to the first loop that waits on it. Serializes within one process only.
    See ``docs/stale-status-snapshot-sweep.md`` §"The per-project lock".
    """
    locks = _SWEEP_LOCKS.setdefault(asyncio.get_running_loop(), {})
    key = canonicalize_project_id(project_id)
    lock = locks.get(key)
    if lock is None:
        lock = locks[key] = asyncio.Lock()
    return lock


def _seed_stats() -> dict:
    return {
        'scanned': 0, 'candidate_edges': 0, 'invalidated': 0, 'errors': 0,
        'superseded': 0, 'supersede_errors': 0, 'supersede_skipped': 0,
        # Seeded UNKNOWN, not True: nothing is proven about the corpus until
        # the enumeration has returned.
        'enumeration_complete': None, 'enumeration_incomplete_kind': None,
    }


async def _sweep_status_snapshot_edges(
    memory_service,
    taskmaster,
    project_id: str,
    project_root: str,
    *,
    task_id: int | None,
    run_id: str,
    now: datetime | None,
    log: logging.Logger,
) -> dict:
    """The one core behind both public sweeps, one sweep per project at a time.

    *task_id* is consulted in exactly two places: THE READ
    (``_read_corpus``) and THE POST-FILTER right after the single extraction
    pass. Everything else — the status census, selection, the counted
    invalidation loop, the supersede ceiling and the stats — is shared, and
    runs under ``_sweep_lock_for(project_id)``.
    """
    if not taskmaster or not project_root:
        return _seed_stats()
    async with _sweep_lock_for(project_id):
        return await _sweep_under_lock(
            memory_service, taskmaster, project_id, project_root,
            task_id=task_id, run_id=run_id, now=now, log=log,
        )


async def _sweep_under_lock(
    memory_service,
    taskmaster,
    project_id: str,
    project_root: str,
    *,
    task_id: int | None,
    run_id: str,
    now: datetime | None,
    log: logging.Logger,
) -> dict:
    stats = _seed_stats()
    corpus = await _read_corpus(memory_service, project_id, task_id, log=log)
    stats['enumeration_complete'] = corpus.complete
    stats['enumeration_incomplete_kind'] = corpus.incomplete_kind
    if corpus.edges is None:
        stats['errors'] += 1
        return stats
    stats['scanned'] = len(corpus.edges)

    # ONE by-class extraction per edge, reused by the post-filter, the census
    # and selection.
    by_class = {
        edge['uuid']: extract_snapshot_edge_task_ids_by_marker_class(edge.get('fact'))
        for edge in corpus.edges
    }
    edges = corpus.edges
    if task_id is not None:
        # THE POST-FILTER: the narrowed read is a digit-substring superset, so
        # keep only edges that really name the task, judged from the SAME
        # extraction pass.
        edges = [edge for edge in edges if task_id in by_class[edge['uuid']].all_ids]
    edge_ids: dict[str, set[int]] = {
        edge['uuid']: by_class[edge['uuid']].all_ids for edge in edges
    }
    blocked_edge_ids: dict[str, set[int]] = {
        edge['uuid']: by_class[edge['uuid']].blocked_ids for edge in edges
    }

    # The census covers BOTH maps' ids, so it stays correct even if
    # blocked-subset-of-union ever stopped holding.
    candidate_ids: set[int] = set()
    for ids in edge_ids.values():
        candidate_ids |= ids
    for ids in blocked_edge_ids.values():
        candidate_ids |= ids

    if not candidate_ids:
        return stats

    try:
        statuses = await taskmaster.get_statuses(
            project_root, ids=[str(i) for i in sorted(candidate_ids)],
        )
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        log.exception(
            'stale_status_snapshot_edge_sweep: get_statuses failed for project_root=%s',
            project_root,
        )
        stats['errors'] += 1
        return stats

    stale = select_stale_status_snapshot_edges(
        edges, statuses, edge_ids=edge_ids, blocked_edge_ids=blocked_edge_ids,
    )
    stats['candidate_edges'] = len(stale)

    invalidate_at = now or datetime.now(UTC)
    # One superseding fact per contradicted TASK per cycle, not per edge.
    #
    # Membership means WRITTEN, not ATTEMPTED: a task's sibling edges are the
    # only retry its fact can get, so a failed write must not consume the
    # task's one write this cycle.
    superseded_ids: set[int] = set()
    # Distinct tasks the per-cycle write ceiling left without a fact, and the
    # number of add_memory calls issued so far. The ceiling bounds attempts,
    # not successes.
    supersede_skipped_ids: set[int] = set()
    supersede_attempts = 0
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
                'stale_status_snapshot_edge_sweep: update_edge failed for uuid=%s',
                edge['uuid'],
            )
            stats['errors'] += 1
            continue

        # Only after a SUCCESSFUL invalidation, and only for ids the BLOCKED
        # rule contributed. Gating on success bounds the write to one per task
        # over the fact's lifetime: the retired edge never re-enumerates.
        for contradicted_id in sorted(blocked_edge_ids.get(edge['uuid'], set())):
            status = statuses.get(str(contradicted_id))
            if (
                not is_blocked_assertion_contradicted(status)
                or contradicted_id in superseded_ids
            ):
                continue
            if supersede_attempts >= _MAX_SUPERSEDE_WRITES_PER_CYCLE:
                supersede_skipped_ids.add(contradicted_id)
                continue
            supersede_attempts += 1
            try:
                await memory_service.add_memory(
                    content=build_supersede_fact(contradicted_id, status, invalidate_at),
                    category='temporal_facts',
                    project_id=project_id,
                    agent_id=_SWEEP_AGENT_ID,
                    causation_id=run_id,
                )
                stats['superseded'] += 1
                # Marked as done only HERE — see superseded_ids' comment.
                superseded_ids.add(contradicted_id)
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                log.exception(
                    'stale_status_snapshot_edge_sweep: add_memory supersede failed '
                    'for task_id=%s (edge uuid=%s already invalidated)',
                    contradicted_id, edge['uuid'],
                )
                stats['supersede_errors'] += 1

    stats['supersede_skipped'] = len(supersede_skipped_ids)
    if supersede_skipped_ids:
        # No silent caps: say how much work was dropped and that it is not
        # coming back next cycle (the edges are already invalidated).
        log.warning(
            'stale_status_snapshot_edge_sweep: per-cycle superseding-write ceiling '
            '(%d) reached; %d contradicted task(s) left without a superseding fact '
            'this cycle and will not be retried (their edges are already '
            'invalidated). task_ids=%s',
            _MAX_SUPERSEDE_WRITES_PER_CYCLE,
            len(supersede_skipped_ids),
            sorted(supersede_skipped_ids),
        )

    return stats
