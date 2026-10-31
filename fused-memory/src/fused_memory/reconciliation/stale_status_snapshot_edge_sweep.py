"""Stale task-status snapshot Graphiti edge sweep — task 2613.

Stage 1 (``MemoryConsolidator``) has no deterministic sweep that keeps VALID
(``invalid_at IS NULL``) task-status-snapshot Graphiti edges in sync with the
tasks they describe. A status-snapshot edge asserts that some task_id is
currently active/pending/blocked/stalled/in-progress; once that task reaches
a terminal status (done/cancelled), the edge is stale but nothing retires
it. The prior
approach — a manual LLM `search` sweep — was unreliable (run b1408864 found
only 17 of an estimated 23+ stale edges). This module replaces it with a
deterministic direct-lookup sweep, mirroring the deterministic-lookup-over-
semantic-search precedent set by task 1680 (count_memories_by_metadata) and
task 2107 (degenerate_task_node_sweep).

Which task ids an edge fact asserts, and under which marker, is read by the
pure lexical layer in ``status_snapshot_extraction.py``; its docstring holds
the extraction paths, their invariants and their residuals.

Standing invariants of the selection and sweep layer:

- Fail-safe direction, throughout the module: under-selection self-heals
  next cycle (or is caught by Stage 2); over-selection is unrecoverable,
  since an invalidated Graphiti edge is never re-validated. Every
  extraction path is deliberately biased toward missing a snapshot over
  wrongly retiring one.
- Invalidate-only-on-positively-KNOWN, and an aggregate/plural edge is
  invalidated as a whole the moment any ONE referenced id contradicts it —
  see ``select_stale_status_snapshot_edges``. Two rules qualify an edge:
  the TERMINAL rule (a union-marker id reached done/cancelled) and the
  BLOCKED-ASSERTION rule (a blocked-asserted id has a positively-known
  status other than 'blocked'). Neither ever fires on an id whose census
  value is absent or outside the closed
  ``shared.task_statuses.TaskStatus`` vocabulary.  The GENERAL rule
  (asserted status != live status, for every marker) was measured by task
  4851 and NOT adopted — verdict in
  ``plans/stale-status-snapshot-sweep-report.md``.  'active' and 'stalled'
  have no ``TaskStatus`` counterpart, hence no contradiction test, so they
  stay terminal-rule-only.
- The sweep is best-effort throughout (mirrors
  ``degenerate_task_node_sweep``): an edge-enumeration
  (``get_all_valid_edges``) or status cross-reference (``get_statuses``)
  failure ends this cycle's sweep early (self-heals next cycle); a
  per-edge invalidation failure does not — see
  ``sweep_stale_status_snapshot_edges``.
- 'blocked' is deliberately NOT added to ``INACTIVE_TASK_STATUSES`` (stays
  ``{done, cancelled}``), which is what preserves
  invalidate-only-on-positively-terminal for a genuinely-still-blocked
  task.  AMENDED, task 3037: a stale blocked assertion IS now retired
  before its task goes terminal — but by a SEPARATE second rule keyed on
  the blocked-SCOPED id set (``extract_blocked_assertion_task_ids``), NOT
  by widening ``INACTIVE_TASK_STATUSES``.  Widening it would make the
  terminal rule retire every ACTIVE/PENDING snapshot whose task merely
  became blocked, and — since that rule reads UNION ids — could fire on an
  id some other marker contributed.  The two rules stay disjoint in what
  they read: rule 1 union ids against ``INACTIVE_TASK_STATUSES``, rule 2
  blocked-asserted ids against "positively known and != 'blocked'".

Two hypotheses were investigated and RULED OUT for the task-2613 miss
rate; do not re-open them:

1. Edge traversal/scope: ``GraphitiBackend.get_all_valid_edges`` runs an
   unfiltered ``MATCH`` (no entity-type, label, or episode filter),
   reproduced end-to-end scanning every missed edge and dropping it at
   extraction — every gap was lexical, not a query-scope gap.

   AMENDED, task 4340 (2026-08-17).  The lexical findings above STAND and
   are unaffected; what does not stand is the scope of the ruling-out.
   Those grounds covered query FILTERS only and were silent about
   server-side TRUNCATION.  Task 4340 measured a FalkorDB
   ``RESULTSET_SIZE`` cap silently truncating that very query to roughly
   HALF the valid-edge corpus on dark_factory (exact figures:
   ``plans/falkordb-resultset-cap-audit.md``, which is the one place they
   are recorded).  So at the time of the task-2613
   investigation this sweep saw about half the edges, and the miss rate was
   computed against a truncated denominator.  ``get_all_valid_edges`` is
   paginated as of task 4340 and the truncation is gone.

   RE-MEASURED, task 4851, against the complete corpus.  The metric's
   definition and every figure live in
   ``plans/stale-status-snapshot-sweep-report.md`` and nowhere else.  The
   durable finding: at steady state the shipped rules select approximately
   nothing, because the sweep keeps up.  The facts that still name a
   now-terminal task are an UPPER bound on misses, and by-eye triage finds
   them mostly NOT status snapshots at all.  The genuine misses are the
   fail-safe lexical classes listed under Known residuals.  Do not re-open
   the LEXICAL hypothesis; do not treat the task-2613 rate as measured on a
   whole corpus.
2. A shared blind spot with ``task_count_verification``: that function is
   an aggregate census-vs-tree consistency check (``task_filter``), not a
   per-task edge sweep at all — its 'healthy' report was correct, not
   blind.

``sweep_stale_status_snapshot_edges_for_task`` (task 4851) shares the
periodic sweep's core, so the one-pass extraction discipline (see
``status_snapshot_extraction.py``'s docstring) holds for
it too. Its narrowed read is cheap only for a selective id; see its docstring.
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

# The closed status vocabulary, frozen once at import. Defines "positively
# known" for the blocked-assertion selection rule — see
# ``is_blocked_assertion_contradicted``. A census value outside it is treated
# as UNKNOWN, never as a contradiction. (task 3037)
_KNOWN_TASK_STATUSES: frozenset[str] = frozenset(TaskStatus)


# --------------------------------------------------------------------------- #
# build_supersede_fact — the superseding temporal_fact's wording (task 3037)
# --------------------------------------------------------------------------- #

# RESULTING-STATE-ONLY, and every word of it is load-bearing. Two independent
# hazards rule out the wordings a reader reaches for first; both were measured
# on this module's own extractor and on task_filter's write guards, and both
# are pinned by TestBuildSupersedeFact rather than left as a claim here.
#
# (1) 'Task {N} was blocked and is now pending' — FORBIDDEN, twice over.
#     This module's OWN extractor re-extracts it (measured -> {2848}), so the
#     superseding fact would satisfy the very selection rule that produced it:
#     next cycle the sweep would retire it and write a fresh copy, forever.
#     It is also MIXED TEMPORAL FRAMING, which prompts/stage1.py L326-346
#     forbids for temporal_facts because Graphiti atomization can re-emit the
#     stale 'was blocked' fragment as a co-current edge — reintroducing
#     exactly the assertion this sweep just retired.
#
# (2) 'Task {N} is no longer blocked' — also forbidden, but for only ONE of
#     those reasons, which is why it is worth naming separately: it is
#     extractor-SAFE (measured -> set()), so an author checking only the
#     self-churn hazard would conclude it is fine. It is PRIOR-STATE framing,
#     which the same resulting-state-only rule refuses.
#     Note what that means operationally: unlike (1), NOTHING executable
#     rejects this wording — all four of task_filter's write-norm predicates
#     pass it (asserted in
#     test_prior_state_negation_is_extractor_safe_but_still_not_the_template).
#     The prohibition is a prose norm, so this template's compliance with it
#     rests on review, not on a guard that would catch a regression.
#
# The chosen template states the CURRENT state and nothing else, and carries
# an explicit 'As of <date>' stamp so it reads as a point-in-time observation
# rather than a standing claim about a live task-table field (the distinction
# task_filter.frames_live_task_status_as_current_fact enforces).
#
# Note the sweep writes in-process via memory_service.add_memory and so
# BYPASSES the MCP-boundary enforcement at server/tools.py:2856/3061 that
# would otherwise reject a recon-stage-* writer's malformed framing. The
# module's compliance therefore rests entirely on this template plus its
# tests — there is no second line of defence downstream.
_SUPERSEDE_FACT_TEMPLATE = 'As of {date}, task {task_id} has status {status}.'


# Per-cycle ceiling on superseding add_memory calls (amendment,
# reviewer_comprehensive performance finding, task 3037).
#
# Each superseding write is a real Graphiti write plus an embedding round-trip,
# awaited SERIALLY inside the per-stale-edge invalidation loop. Its count is
# bounded only by how many distinct blocked-asserted tasks are contradicted in
# the cycle — and on the FIRST cycle after this ships, every blocked assertion
# accumulated in the corpus since the sweep was written becomes eligible at
# once. This module treats a ~3.3s edge enumeration as a whole-cycle LIVENESS
# property and carries performance tests to bound regex cost; adding an
# unbounded number of sequential network writes to the same loop would blow
# that budget by minutes on the drain cycle.
#
# 50 keeps the worst-case added latency in the same order as the enumeration
# cost the module already budgets for, while sitting far above steady state:
# once the backlog has drained, a cycle supersedes roughly one fact per task
# unblocked since the previous cycle — single digits.
#
# NO SILENT CAP: truncation is tallied into stats['supersede_skipped'] and
# logged at WARNING with the count. And it IS a loss, not a deferral — the
# edges are still invalidated, so they do not re-enumerate next cycle and the
# skipped facts are never written. That is the same trade the module already
# takes on a supersede FAILURE (see sweep_stale_status_snapshot_edges'
# best-effort paragraph): the correctness-bearing half of the step — retiring
# the contradicted assertion — always completes; only the record of what
# replaced it is best-effort.
#
# The ceiling counts ATTEMPTS, not successes, so a cycle whose writes are all
# failing cannot spin through an unbounded number of them via the in-cycle
# retry path either.
_MAX_SUPERSEDE_WRITES_PER_CYCLE = 50


def build_supersede_fact(task_id: int, status: str, now: datetime) -> str:
    """Render the resulting-state-only temporal_fact that supersedes a retired
    blocked assertion.

    Args:
        task_id: The task whose blocked assertion was contradicted.
        status: Its CURRENT census status — the resulting state, stated on its
            own. Never paired with the prior state; see
            ``_SUPERSEDE_FACT_TEMPLATE`` for why.
        now: The timestamp to stamp the observation with. Always passed in,
            never read from a module-level clock, so the fact carries the same
            instant the sweep used as the edge's ``invalid_at`` and a replayed
            or back-dated run stays reproducible.

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
    """Does *status* positively contradict a 'blocked' assertion? (task 3037)

    True iff *status* is a POSITIVELY KNOWN member of the closed
    ``shared.task_statuses.TaskStatus`` vocabulary and is not 'blocked'.

    The vocabulary check is what preserves this module's
    invalidate-only-on-positively-known fail-safe in the same direction the
    terminal rule already has it. A naive ``status != 'blocked'`` would read
    every unrecognised value as a contradiction, so a malformed, renamed or
    sentinel census entry ('unknown', '', a future status this build has not
    heard of) would RETIRE a live edge — the unrecoverable direction, since an
    invalidated Graphiti edge is never re-validated. Treating it as unknown
    instead costs at most one cycle of under-selection.

    ``None`` (id absent from the census entirely) is likewise unknown, which
    is what makes a transient ``get_statuses`` gap self-healing.

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

    1. TERMINAL rule (task 2613). ``extract_snapshot_edge_task_ids`` returns
       at least one id AND any one of those ids resolves (via *statuses*) to
       a positively-terminal status (``INACTIVE_TASK_STATUSES`` —
       done/cancelled). This rule reads the UNION of every status marker, so
       it covers active/pending/in-progress/stalled assertions too.

    2. BLOCKED-ASSERTION rule (task 3037).
       ``extract_blocked_assertion_task_ids`` returns at least one id AND any
       one of those ids resolves to a positively-known status that is not
       'blocked' (``is_blocked_assertion_contradicted``). A blocked assertion
       is contradicted by ANY non-blocked status, not merely a terminal one:
       before this rule, 'Task 2848 remains blocked as of 2026-07-22'
       survived a blocked->pending unblock and stayed live until the task
       eventually reached done/cancelled.

    In both rules an aggregate/plural edge referencing several ids is selected
    as a whole the moment ANY single referenced id contradicts it — the
    snapshot as asserted no longer holds.

    Why rule 2 is a SEPARATE rule keyed on a blocked-SCOPED id set, rather
    than 'blocked' being added to ``INACTIVE_TASK_STATUSES``: that widening
    would make rule 1 retire every ACTIVE/PENDING snapshot whose task merely
    became blocked, and — because rule 1 reads union ids — a blocked-status
    trigger there could fire on an id some OTHER marker contributed. Rule 2
    reads only the ids the fact asserts as blocked, which is exactly its
    scope. The general rule (any asserted status contradicted by the live
    one) was measured by task 4851 and not adopted; 'active' and 'stalled'
    have no ``TaskStatus`` counterpart to contradict, so they remain
    terminal-rule-only — see ``plans/stale-status-snapshot-sweep-report.md``.

    Invalidate-only-on-positively-known, throughout: an id absent from
    *statuses*, or mapped to a value outside the closed
    ``shared.task_statuses.TaskStatus`` vocabulary, never contributes to
    selection under either rule — a transient census gap or a
    genuinely-still-blocked task can only under-select (self-heals next
    cycle), never wrongly select a valid edge.

    Args:
        edges: Candidate edges (as returned by ``flatten_dedup_edges``).
        statuses: ``{task_id_str: status_str}`` census.
        edge_ids: Optional precomputed ``{edge['uuid']: extract_snapshot_edge_task_ids(fact)}``
            mapping. A caller that already extracted ids for its own purposes
            (e.g. ``sweep_stale_status_snapshot_edges``, which needs them to
            build its candidate-id set) passes this to avoid re-running the
            extraction regex pipeline a second time per edge. When omitted
            (the default), ids are computed directly from each edge's fact —
            unchanged standalone behavior. (amendment, reviewer_comprehensive
            efficiency finding, task 2613)
        blocked_edge_ids: Optional precomputed
            ``{edge['uuid']: extract_blocked_assertion_task_ids(fact)}``
            mapping, for rule 2. Exactly the same semantics and the same
            fall-back-to-internal-extraction default as *edge_ids*, so
            standalone behaviour is unchanged when it is omitted. (task 3037)

    Pure: no I/O, no side effects.
    """
    selected: list[dict] = []
    for edge in edges:
        # Each fall-back extracts independently — so a caller that
        # precomputes BOTH maps (the sweep does) costs zero
        # extraction-pipeline entries here. Pinned by
        # test_extraction_runs_exactly_once_per_edge.
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

# The names this sweep's completeness pair carries once MemoryConsolidator has
# prefixed it onto ``report.stats``.  Exported as symbols rather than left as
# inline literals at each of producer / consumer / test, matching the
# :data:`~fused_memory.reconciliation.task_count_snapshot_cadence.SNAPSHOT_PRUNE_ENUMERATION_OK_STAT_KEY`
# precedent for exactly this stat family — otherwise a rename has to be found
# by grep across five files, which is how a stat key ends up half-renamed and
# a consumer starts silently reading `None`.  (amendment,
# reviewer_comprehensive pattern-consistency finding, task 4386)
#
# NOTE the two layers: the sweep's own returned dict uses the SHORT keys
# ``enumeration_complete`` / ``enumeration_incomplete_kind`` (documented on
# ``sweep_stale_status_snapshot_edges`` below), and the consolidator projects
# them onto ``report.stats`` under this module's prefix.  These constants name
# the ``report.stats`` layer — the one that crosses module boundaries and is
# read by the ledger, the journal and the judge.  Spelled out as whole
# literals rather than built by concatenating a prefix, so the shipped key
# name stays greppable from a log line back to here.

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
    ``memory_service.graphiti.enumerate_all_valid_edges`` (a deterministic bulk
    query — never the LLM's semantic search), extracts the specific task ids
    each edge asserts as active/pending/blocked/stalled/in-progress, cross-references those
    ids' CURRENT status via ``taskmaster.get_statuses`` (a direct status
    lookup, not semantic search), and invalidates
    (``memory_service.update_edge(..., invalid_at=...)``) every edge whose
    asserted status is now contradicted — either by a terminal
    (done/cancelled) task, or (task 3037) by a blocked assertion whose task
    now has any other positively-known status. After a successful
    blocked-rule invalidation it also writes ONE superseding
    resulting-state-only ``temporal_facts`` memory per contradicted task per
    cycle (``build_supersede_fact``), so the graph records what replaced the
    retired assertion instead of merely losing it.

    Args:
        memory_service: Object exposing ``.graphiti.enumerate_all_valid_edges``
            and
            ``.update_edge``.
        taskmaster: Object exposing ``.get_statuses``. A falsy value (e.g.
            unavailable backend) short-circuits to all-zero stats.
        project_id: Graphiti group_id to enumerate and invalidate within.
        project_root: Taskmaster project root for the status lookup. A
            falsy value short-circuits to all-zero stats.
        run_id: Reconciliation run id, threaded through as update_edge's
            causation_id.
        now: Invalidation timestamp; defaults to ``datetime.now(UTC)``.
        log: Logger to use (default: this module's logger).

    Best-effort (mirrors
    ``degenerate_task_node_sweep.sweep_degenerate_task_nodes``): a transient
    backend error enumerating (``enumerate_all_valid_edges``), cross-referencing
    (``get_statuses``), or invalidating (``update_edge``) is caught, logged,
    and tallied into ``stats['errors']``. An enumeration or cross-reference
    failure aborts the rest of this cycle's sweep (there is nothing left to
    act on without them) and returns the stats gathered so far; a per-edge
    ``update_edge`` failure does NOT abort the loop — the remaining stale
    edges are still attempted. A superseding-write (``add_memory``) failure
    is likewise caught, logged, and tallied — but into ``stats
    ['supersede_errors']``, and it too leaves the loop running. It degrades
    only OBSERVABILITY: by the time it runs the edge is already invalidated
    (the correctness-bearing half of the step), so the graph is left
    consistent, just without the fact recording what replaced the retired
    assertion. The missing fact is not retried next cycle — the invalidated
    edge no longer enumerates — so this is a permanently missing superseding
    fact rather than a corrupted edge or a repeating failure.
    ``asyncio.CancelledError``/``KeyboardInterrupt``/``SystemExit`` are
    re-raised unchanged from BOTH write paths (never swallowed as
    best-effort).

    Because the missing fact cannot be retried NEXT cycle, the one retry it
    can still get is in-cycle: the per-task dedupe marker is set on SUCCESS
    only, so a task named by several stale edges gets a fresh attempt from
    each until one lands. (amendment, reviewer_comprehensive robustness
    finding, task 3037)

    The superseding writes are additionally capped per cycle at
    ``_MAX_SUPERSEDE_WRITES_PER_CYCLE`` so an unbounded backlog cannot turn
    the invalidation loop into minutes of serial network writes; the excess
    is tallied into ``stats['supersede_skipped']`` and logged at WARNING
    rather than dropped silently.

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
        than counting anything (task 4386):

        ``enumeration_complete`` (``bool | None``) — TRI-STATE, because three
        outcomes are genuinely distinct and collapsing any pair loses
        information the reader needs. ``True`` = the read was PROVEN
        complete; ``False`` = a corpus was observed and it was INCOMPLETE;
        ``None`` = NO corpus was observed at all, either because this call
        short-circuited on a falsy *taskmaster*/*project_root* or because the
        enumeration itself failed (``errors`` tells those two apart). The
        ONLY predicate a caller may gate on is ``is True``: an ``is not
        False`` test would admit both UNKNOWN cases, which would let a cycle
        that never looked pass as one that looked and found everything.

        ``enumeration_incomplete_kind`` (``str | None``) — the backend's
        ``INCOMPLETE_*`` constant (``graphiti_client.py::INCOMPLETE_PAGE_CAP``
        and siblings), which is the documented STABLE discriminator. Callers
        should branch on membership in
        ``graphiti_client.py::INCOMPLETE_STRUCTURAL_KINDS`` rather than on a
        specific kind. ``PagedRead.reason`` is deliberately NOT surfaced: the
        backend documents its wording as diagnostic prose and an unstable
        interface, so projecting it into stats would invite consumers to
        parse it.

        Both keys describe the read, NOT the counted funnel, so they leave
        the ``invalidated == candidate_edges - errors`` identity below
        exactly true. An EMPIRICAL incompleteness in particular does not
        touch ``errors``: the sweep proceeds on what it fetched, and the
        cycle merely says the corpus was partial.

        ``errors`` stays scoped to the enumerate / cross-reference /
        INVALIDATE paths, which is what keeps the identity
        ``invalidated == candidate_edges - errors`` exactly true: every
        selected edge is either invalidated or tallied there, by the ONE
        counted invalidation loop. That identity is the structural guard
        against a future second invalidation site going uncounted (it is
        pinned by a test), so a NON-invalidation failure mode must never be
        folded into ``errors`` — the superseding write gets its own
        ``supersede_errors`` key for exactly that reason. (task 3037)

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
    judge it.

    Its cost depends on the id. A multi-digit id reads a handful of rows; a
    one- or two-digit id, or one that also occurs in dates, reads a sizeable
    fraction of the corpus (row counts in plans/falkordb-resultset-cap-audit.md).
    That is accepted because it is bounded by the periodic sweep's full read,
    runs off any request path, and never overlaps another sweep of the
    project (``_sweep_lock_for``), so a cascade of transitions queues rather
    than stacks.

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

    Completeness is reported even when the policy aborts the read, and that
    is load-bearing: ``apply_incompleteness_policy`` RAISES on a structural
    incompleteness, and an aborted cycle that reported only errors=1 would
    leave the operator reconstructing the cause from logs — the exact
    reconstruction the signal exists to remove. (task 4386)

    ``enumerate_*`` NEVER raises — it reports incompleteness as a value — so
    the fail-closed structural guard has to be applied here, or a page-capped
    read is taken for the whole corpus and every edge the missing pages carry
    is scanned as absent: a silently clean cycle that retires nothing. It is
    deliberately inside the same try, so a structural incompleteness lands in
    the one handler (``IncompleteEnumerationError`` subclasses ``Exception``
    precisely so it is caught). An EMPIRICAL incompleteness only warns —
    through the sweep's injected ``log``, so the message surfaces with the
    rest of the cycle's diagnostics — and the sweep proceeds on what it got.
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
    """The lock every sweep of *project_id* holds, on the running loop. (task 4851)

    One superseding fact per contradicted task holds only if each sweep READS
    after the previous one's retirements: a retired edge stops enumerating,
    and that is the whole dedup. Overlapping sweeps of one project would both
    read the same still-valid edge, both retire it and both write its facts —
    a cancellation cascade starts one targeted sweep per task without awaiting
    any, and a full cycle's Stage-1 sweep can start beside them.

    Keyed by the canonical project id, as the graph itself is, and by event
    loop, because an ``asyncio.Lock`` binds to the first loop that waits on
    it. Serializes within one process only, which is where both callers run.
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
        # Seeded UNKNOWN, not True: neither key is a count, and until the
        # enumeration has actually returned nothing has been proven about the
        # corpus. Every early return therefore reports the honest 'no corpus
        # observed' rather than a fabricated clean read. (task 4386)
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
    runs under ``_sweep_lock_for(project_id)``. (task 4851)
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

    # Extract each edge's ids exactly once — ONE by-class pass per edge
    # yielding BOTH id sets, reused below to build candidate_ids and (via
    # select_stale_status_snapshot_edges's two precomputation kwargs) for the
    # final selection, instead of re-running the extraction pipeline per
    # edge per marker class. (amendment, reviewer_comprehensive efficiency
    # finding, task 2613; extended to the blocked family, task 3037)
    by_class = {
        edge['uuid']: extract_snapshot_edge_task_ids_by_marker_class(edge.get('fact'))
        for edge in corpus.edges
    }
    edges = corpus.edges
    if task_id is not None:
        # THE POST-FILTER: the narrowed read is a digit-substring superset, so
        # keep only edges that really name the task, judged from the SAME
        # extraction pass. (task 4851)
        edges = [edge for edge in edges if task_id in by_class[edge['uuid']].all_ids]
    edge_ids: dict[str, set[int]] = {
        edge['uuid']: by_class[edge['uuid']].all_ids for edge in edges
    }
    blocked_edge_ids: dict[str, set[int]] = {
        edge['uuid']: by_class[edge['uuid']].blocked_ids for edge in edges
    }

    # The census must cover BOTH maps' ids. blocked_ids is a subset of
    # all_ids by construction, so this is currently the union of all_ids
    # alone — but taking it over both is what keeps the census correct if
    # that containment ever stops holding, rather than silently
    # cross-referencing a blocked-asserted id against a status never looked
    # up (which would read as absent, i.e. under-select).
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
    # One superseding fact per contradicted TASK per cycle, not per edge: a
    # contradicted task typically has several blocked-assertion edges (one per
    # phrasing Graphiti extracted), and writing per edge would emit the
    # identical fact several times in a single cycle.
    #
    # Membership means WRITTEN, not ATTEMPTED. Those coincide except when a
    # write fails, and there the distinction is the whole point: the sibling
    # edges of the same task are the only retry this fact can ever get (an
    # invalidated edge does not re-enumerate next cycle), so a transient
    # failure must not consume the task's one-write-per-cycle budget.
    # (amendment, reviewer_comprehensive robustness finding, task 3037)
    superseded_ids: set[int] = set()
    # Distinct tasks the per-cycle write ceiling left without a fact, and the
    # number of add_memory calls issued so far — see
    # _MAX_SUPERSEDE_WRITES_PER_CYCLE. Attempts, not successes, are what the
    # ceiling bounds: they are what costs wall-clock.
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

        # Second half of the deterministic step, gated on a SUCCESSFUL
        # invalidation and scoped to ids the BLOCKED rule contributed.
        #
        # Gating on success is also what bounds the write to exactly ONE per
        # contradicted task over the fact's whole lifetime: an invalidated
        # edge drops out of get_all_valid_edges, so it cannot re-present next
        # cycle. Writing on FAILURE would instead repeat the same fact every
        # cycle for as long as the invalidation kept failing — and would
        # meanwhile leave the graph asserting both that the task is blocked
        # (the edge is still valid) and that it is not.
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
