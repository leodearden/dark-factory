"""Stale task-status snapshot Graphiti edge sweep — task 2613.

Stage 1 (``MemoryConsolidator``) has no deterministic sweep that keeps VALID
(``invalid_at IS NULL``) task-status-snapshot Graphiti edges in sync with the
tasks they describe. A status-snapshot edge asserts that some task_id is
currently active/pending/in-progress; once that task reaches a terminal
status (done/cancelled), the edge is stale but nothing retires it. The prior
approach — a manual LLM `search` sweep — was unreliable (run b1408864 found
only 17 of an estimated 23+ stale edges). This module replaces it with a
deterministic direct-lookup sweep, mirroring the deterministic-lookup-over-
semantic-search precedent set by task 1680 (count_memories_by_metadata) and
task 2107 (degenerate_task_node_sweep).

Design decisions (captured in plan.json):

- extract_snapshot_edge_task_ids returns the empty set for pure count-only
  snapshots with no specific task-id reference (e.g. "There are 8 tasks in
  progress", "1505 done / 148 cancelled") — these are OUT OF SCOPE,
  stale-by-design audit trail per Snapshot Discipline, and must never be
  invalidated by this sweep.
- extract_snapshot_edge_task_ids anchors the status marker directly to its
  own task reference (INDIVIDUAL_SNAPSHOT_RE) rather than gating on
  whole-fact marker presence, so an incidental status word describing
  something else in the same fact (e.g. "Task 142 landed on the active
  branch" — the active BRANCH, not task 142) is never wrongly attributed to
  the reference. (amendment, reviewer_comprehensive precision finding, task
  2613)
- Invalidate-only-on-positively-terminal: an unknown/missing/still-active
  status for a referenced id never triggers invalidation (fail-safe,
  mirrors flag_dedup.filter_terminal_metadata_flags) — a transient census
  hiccup can only under-invalidate (self-heals next cycle), never wrongly
  retire a valid edge.
- An aggregate snapshot edge ("the active pending tasks are [A, B, C]") is
  invalidated as a whole the moment ANY one referenced id is now terminal —
  the snapshot as asserted no longer holds.
- Best-effort throughout (modelled on
  ``degenerate_task_node_sweep.sweep_degenerate_task_nodes``): every caught
  failure is logged and tallied into ``stats['errors']`` rather than
  raised. ``get_all_valid_edges``/``get_statuses`` are single bulk calls the
  rest of the cycle depends on, so a failure there ends this cycle's sweep
  early with the stats gathered so far (self-heals next cycle). A per-edge
  ``update_edge`` failure, by contrast, does NOT abort the loop — the
  remaining stale edges are still attempted, exactly like the per-item
  find/delete calls in ``sweep_degenerate_task_nodes``.
- 'blocked' is now a recognized non-terminal snapshot status marker. Its
  absence made the gate short-circuit
  (``if not SNAPSHOT_STATUS_RE.search(fact): return set()``) fire before
  the ``INACTIVE_TASK_STATUSES`` cross-reference could ever run, so
  blocked-worded edges were structurally invisible regardless of the
  referenced task's real status (task 2885 repro: 4 stale edges survived a
  blocked->done transition; scanned=5868/invalidated=0). (amendment, task
  3042)
- The single ``_STATUS_MARKER_ALT`` constant now feeds the gate, the
  anchored individual regex, and the phrase regex, so the gate can never
  again be narrower than the anchored matchers — that drift (a marker
  present in one but not the other) is the exact bug class task 3042
  fixes. (amendment, task 3042)
- Two anchoring paths with an explicit precision contract: closed-class
  connective only (copula / article — NOT the preposition 'in', which
  would let the marker bind to a noun the task is merely located in) for
  the general individual form (``INDIVIDUAL_SNAPSHOT_RE``); a lazy
  ``{0,3}``-word open-class gap and an 'in' preposition permitted ONLY
  when the marker is immediately followed by the literal noun 'status'
  (``SNAPSHOT_STATUS_PHRASE_RE``). That status-noun requirement is what
  keeps 'Task N is in the active branch' / 'is in the pending merge
  queue' out while still reaching 'Task N is in a blocked status ...'.
  (amendment, task 3042)
- 'blocked' is deliberately NOT added to ``INACTIVE_TASK_STATUSES`` — that
  frozenset stays ``{done, cancelled}`` — which is what preserves
  invalidate-only-on-positively-terminal (a genuinely-still-blocked task is
  never selected). (amendment, task 3042)
- Known residual (task 3042): a blocked-status assertion that neither uses
  a closed-class connective nor the literal 'status' noun (e.g. "Task N
  remains parked awaiting adjudication") is still not extracted. This is
  the deliberate fail-safe direction — under-selection self-heals or is
  caught by Stage 2, whereas over-selection would wrongly retire true
  facts.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime

from fused_memory.reconciliation.task_filter import INACTIVE_TASK_STATUSES, TASK_REF_RE

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# extract_snapshot_edge_task_ids — pure lexical extraction
# --------------------------------------------------------------------------- #

# Status markers this sweep treats as non-terminal snapshot assertions.
# Hoisted into a single constant so the gate (SNAPSHOT_STATUS_RE) and the
# anchored individual-form regex (INDIVIDUAL_SNAPSHOT_RE) can never drift
# apart again — before task 3042 they were hand-copied duplicates, and a
# marker present in one but not the other silently suppressed a whole class
# of edges (the gate short-circuits below, before the INACTIVE_TASK_STATUSES
# cross-reference in select_stale_status_snapshot_edges ever runs).
_STATUS_MARKER_ALT = r'(?:active|pending|blocked|in[-\s]?progress)'

# Gate: a status-snapshot edge must assert one of these non-terminal
# statuses somewhere in its fact text, else it is not a status-snapshot edge
# this sweep concerns itself with at all (e.g. 'Task 5 is done' — already
# terminal at write time, not what this sweep targets).
SNAPSHOT_STATUS_RE: re.Pattern[str] = re.compile(
    r'\b' + _STATUS_MARKER_ALT + r'\b',
    re.IGNORECASE,
)

# NOTE (amendment, reviewer_comprehensive correctness-recall finding, task
# 2613): individual-form extraction used to run against a COUNT_QUANTITY_RE-
# stripped copy of the WHOLE fact, so a verb-less snapshot like 'Task 5 in
# progress' or 'Task 5 pending' had its own digit stripped out first (it
# matches '\d+\s+in[-\s]?progress' / '\d+\s+pending') before the id
# extraction ever ran — a silent false negative. INDIVIDUAL_SNAPSHOT_RE
# below anchors the status marker directly to its own task reference
# instead (see its docstring), so COUNT_QUANTITY_RE is now only ever
# applied to an already list-scoped aggregate segment, never to the whole
# fact — see the aggregate-form loop in extract_snapshot_edge_task_ids.

# Strips 'N <count-noun>' spans (e.g. '8 tasks', '148 cancelled', '3 pending')
# from an aggregate list segment before bare-digit extraction, so an
# embedded count phrase (e.g. '...tasks: 142, 148, and 200 total remain')
# never contributes a spurious id.
COUNT_QUANTITY_RE: re.Pattern[str] = re.compile(
    r'\b\d+\s+(?:tasks?|active|pending|in[-\s]?progress|done|cancell?ed|'
    r'blocked|deferred|review|total|merge[-\s]?deferred)\b',
    re.IGNORECASE,
)

# NOTE (amendment, reviewer_comprehensive correctness-precision finding, task
# 2613): the individual form used to extract a task id via TASK_REF_RE
# anywhere in the fact as long as SNAPSHOT_STATUS_RE matched ANYWHERE else in
# the same fact — so 'Task 142 landed on the active branch' (the BRANCH is
# active, not task 142) wrongly yielded {142}. INDIVIDUAL_SNAPSHOT_RE anchors
# the status marker directly to its own task reference: 'task N' (or '#N' /
# 'df N') optionally followed by a copula ('is'/'are'/'was'/'were') and/or an
# article ('a'/'an'), then the status marker itself — with nothing else (e.g.
# an intervening verb like 'landed') allowed in between. Built directly from
# TASK_REF_RE.pattern (not a hand-copied duplicate) so the two stay in sync
# if the shared task-reference grammar ever changes.
#
# NOTE (amendment, task 3042): two further changes on top of the above.
# (a) 'blocked' is a non-terminal snapshot status (a blocked task has not
#     reached a terminal outcome), and its prior absence from
#     _STATUS_MARKER_ALT made every blocked-worded edge invisible to this
#     sweep regardless of the referenced task's real status: the gate
#     (SNAPSHOT_STATUS_RE) short-circuits before the INACTIVE_TASK_STATUSES
#     cross-reference in select_stale_status_snapshot_edges ever runs.
# (b) The connective here is DELIBERATELY still the task-2613 closed-class
#     form — copula and/or 'a'/'an' only. An earlier revision of this task
#     also admitted the preposition 'in' and the article 'the' here, to
#     reach the repro fact 'Task N is in a blocked status ...'. That was
#     reverted (amendment, reviewer_comprehensive correctness-precision
#     finding, task 3042): 'in'/'the' turn the connective into a full
#     prepositional noun phrase, so the marker binds to a NOUN the task is
#     merely located in rather than to the task's status — empirically
#     'Task 5 is in the active branch', 'Task 7 is in the pending merge
#     queue', 'Task 5 is in a blocked directory' and 'Task 5 is the pending
#     review owner' all wrongly yielded an id. That is the over-selection
#     direction the module docstring forbids (a fact like 'Task 142 is in
#     the active sprint retrospective' stays true forever, yet the sweep
#     would retire it once 142 went done). The repro facts are reached
#     instead by SNAPSHOT_STATUS_PHRASE_RE below, whose 'in' path requires
#     the literal noun 'status' after the marker.
# (c) The precision invariant task 2613 established is therefore unchanged:
#     ONLY closed-class function words (copula / article) may sit between
#     the reference and the marker — never an open-class verb or noun, and
#     never a preposition introducing a noun phrase. So 'Task 142 landed on
#     the blocked branch' still yields set().
INDIVIDUAL_SNAPSHOT_RE: re.Pattern[str] = re.compile(
    TASK_REF_RE.pattern
    + r'\s*(?:is|are|was|were)?\s*(?:an?\s+)?'
    + _STATUS_MARKER_ALT + r'\b',
    re.IGNORECASE,
)

# NOTE (new regex, task 3042): SNAPSHOT_STATUS_PHRASE_RE admits a second,
# more permissive anchoring path that INDIVIDUAL_SNAPSHOT_RE deliberately
# refuses: an open-class gap of up to 3 words between the task reference
# and the preposition 'in' (e.g. 'is deliberately parked in ...', where
# 'deliberately parked' is a verb phrase, not a closed-class connective).
# This gap is admitted ONLY when the marker is immediately followed by the
# literal noun 'status' — that noun is what turns the span into an
# explicit status assertion about the referenced task rather than an
# incidental mention, so the open-class gap costs no precision (preserves
# the task-2613/3042 precision invariant documented on INDIVIDUAL_SNAPSHOT_RE
# above). The gap is lazy (``{0,3}?``) and bounded, and — because '.', ',',
# ';' are not ``\w`` and the gap's own leading ``\s+`` cannot absorb them —
# it can never cross a clause boundary: 'Task 5 is done. Task 9 is in
# blocked status' anchors only to task 9, never dragging in task 5.
SNAPSHOT_STATUS_PHRASE_RE: re.Pattern[str] = re.compile(
    TASK_REF_RE.pattern
    + r'(?:\s+\w+){0,3}?\s+in\s+(?:an?\s+|the\s+)?'
    + _STATUS_MARKER_ALT + r'\s+status\b',
    re.IGNORECASE,
)

# Detects the start of an aggregate list segment: '...tasks [' or
# '...tasks are [' or '...tasks:' or '...tasks are:'. The 'open' group
# records which delimiter opened the segment ('[' vs ':') so
# _list_segment can decide how to find the segment's end.
LIST_INTRODUCER_RE: re.Pattern[str] = re.compile(
    r'\btasks?\b\s*(?:are|is|were)?\s*(?P<open>[:\[])',
    re.IGNORECASE,
)

# Clause-ish boundary used to close a colon-introduced list segment (no
# closing delimiter of its own) and as a fallback for an unterminated
# bracket segment. Mirrors task_filter._CLAUSE_SPLIT_RE.
_CLAUSE_BOUNDARY_RE: re.Pattern[str] = re.compile(r'[.;\n!?]')

# Bare digit token — used only within an already-detected list segment, so
# a bare '\d+' never contributes a candidate id outside that scoped context
# (dates/commit-hashes/ports in ordinary prose are never list segments).
_BARE_DIGIT_RE: re.Pattern[str] = re.compile(r'\b\d+\b')


def _list_segment(text: str, start: int, open_char: str) -> str:
    """Return the list-segment substring of *text* beginning at *start*.

    A bracket-opened segment (``open_char == '['``) runs to the matching
    ``']'`` when present, else falls back to the next clause boundary / end
    of string (mirrors the colon case) rather than running away to the end
    of a long text. A colon-opened segment has no closing delimiter of its
    own, so it always runs to the next clause boundary ([.;\\n!?]) or end of
    string.
    """
    if open_char == '[':
        close = text.find(']', start)
        if close != -1:
            return text[start:close]

    boundary = _CLAUSE_BOUNDARY_RE.search(text, start)
    end = boundary.start() if boundary else len(text)
    return text[start:end]


def extract_snapshot_edge_task_ids(fact: str) -> set[int]:
    """Return the task ids *fact* asserts as active/pending/blocked/in-progress.

    Returns the empty set (never a candidate for invalidation) when:
    - *fact* contains no active/pending/blocked/in-progress status marker at
      all (e.g. 'Task 5 is done', 'Task 7 landed as merge commit'); or
    - *fact* is a pure count-only snapshot with no specific task-id
      reference (e.g. 'There are 8 tasks in progress', '1505 done / 148
      cancelled') — out of scope per Snapshot Discipline; or
    - a task id is only incidentally near an unrelated status word (e.g.
      'Task 142 landed on the active branch' — the active BRANCH, not task
      142 — see INDIVIDUAL_SNAPSHOT_RE).

    Algorithm:
      1. Gate on SNAPSHOT_STATUS_RE against the raw fact text; short-circuit
         to the empty set when absent.
      2. Individual form: extract ids via INDIVIDUAL_SNAPSHOT_RE, which
         anchors 'task N' / '#N' / 'df N' (TASK_REF_RE's own grammar)
         directly to an adjacent status marker (only an optional copula
         and/or article may sit in between — deliberately NOT the
         preposition 'in') — so an incidental status word elsewhere in
         the fact is never wrongly attributed to the reference.
      3. Status-phrase form: extract ids via SNAPSHOT_STATUS_PHRASE_RE,
         which additionally admits a bounded open-class gap (e.g. 'is
         deliberately parked in ...') between the reference and the
         marker, but only when the marker is immediately followed by the
         literal noun 'status' — the token that makes the span an
         explicit status assertion rather than an incidental mention.
      4. Aggregate form: for each detected list segment ('tasks are
         [...]' / 'tasks: ...'), strip COUNT_QUANTITY_RE spans from that
         segment only (so an embedded count phrase doesn't contribute a
         spurious id) and collect its bare digit tokens.

    Pure: no I/O, no side effects.
    """
    fact = fact or ''
    if not SNAPSHOT_STATUS_RE.search(fact):
        return set()

    ids: set[int] = {int(m.group(1)) for m in INDIVIDUAL_SNAPSHOT_RE.finditer(fact)}
    ids |= {int(m.group(1)) for m in SNAPSHOT_STATUS_PHRASE_RE.finditer(fact)}

    for intro in LIST_INTRODUCER_RE.finditer(fact):
        segment = COUNT_QUANTITY_RE.sub(' ', _list_segment(fact, intro.end(), intro.group('open')))
        ids.update(int(tok) for tok in _BARE_DIGIT_RE.findall(segment))

    return ids


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


def select_stale_status_snapshot_edges(
    edges: list[dict],
    statuses: dict[str, str],
    *,
    edge_ids: dict[str, set[int]] | None = None,
) -> list[dict]:
    """Return the subset of *edges* whose asserted status is now contradicted.

    An edge is selected (stale) iff ``extract_snapshot_edge_task_ids`` returns
    at least one id AND any one of those ids resolves (via *statuses*) to a
    positively-terminal status (``INACTIVE_TASK_STATUSES`` — done/cancelled).
    An aggregate edge referencing several ids is selected as a whole the
    moment ANY single referenced id is terminal — the snapshot as asserted no
    longer holds.

    Invalidate-only-on-positively-terminal: an id absent from *statuses*, or
    mapped to any non-terminal/unknown value, never contributes to
    selection — a transient census gap or genuinely-still-active task can
    only under-select (self-heals next cycle), never wrongly select a valid
    edge.

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

    Pure: no I/O, no side effects.
    """
    selected: list[dict] = []
    for edge in edges:
        if edge_ids is not None:
            ids = edge_ids.get(edge['uuid'], set())
        else:
            ids = extract_snapshot_edge_task_ids(edge.get('fact') or '')
        if not ids:
            continue
        if any(statuses.get(str(i)) in INACTIVE_TASK_STATUSES for i in ids):
            selected.append(edge)
    return selected


# --------------------------------------------------------------------------- #
# sweep_stale_status_snapshot_edges — async orchestrator
# --------------------------------------------------------------------------- #

# Written to update_edge's agent_id — identifies this sweep's writes in the
# write journal / audit trail, mirroring degenerate_task_node_sweep's
# convention of stamping a stable, identifiable actor.
_SWEEP_AGENT_ID = 'recon-stage-memory_consolidator'


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
    ``memory_service.graphiti.get_all_valid_edges`` (a deterministic bulk
    query — never the LLM's semantic search), extracts the specific task ids
    each edge asserts as active/pending/blocked/in-progress, cross-references those
    ids' CURRENT status via ``taskmaster.get_statuses`` (a direct status
    lookup, not semantic search), and invalidates
    (``memory_service.update_edge(..., invalid_at=...)``) every edge whose
    asserted status is now contradicted by a terminal (done/cancelled) task.

    Args:
        memory_service: Object exposing ``.graphiti.get_all_valid_edges`` and
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
    backend error enumerating (``get_all_valid_edges``), cross-referencing
    (``get_statuses``), or invalidating (``update_edge``) is caught, logged,
    and tallied into ``stats['errors']``. An enumeration or cross-reference
    failure aborts the rest of this cycle's sweep (there is nothing left to
    act on without them) and returns the stats gathered so far; a per-edge
    ``update_edge`` failure does NOT abort the loop — the remaining stale
    edges are still attempted. ``asyncio.CancelledError``/``KeyboardInterrupt``/
    ``SystemExit`` are re-raised unchanged (never swallowed as best-effort).

    Returns:
        dict with int counts: ``scanned`` (edges enumerated after dedup),
        ``candidate_edges`` (edges selected as stale by
        ``select_stale_status_snapshot_edges``), ``invalidated`` (successful
        update_edge calls), ``errors`` (caught enumerate/cross-reference/
        invalidate failures).

    Empty *taskmaster*/*project_root* short-circuits to all-zero stats with
    no backend calls. When enumeration yields no candidate ids at all,
    ``taskmaster.get_statuses`` is never called.
    """
    stats = {'scanned': 0, 'candidate_edges': 0, 'invalidated': 0, 'errors': 0}

    if not taskmaster or not project_root:
        return stats

    try:
        grouped = await memory_service.graphiti.get_all_valid_edges(group_id=project_id)
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        log.exception(
            'stale_status_snapshot_edge_sweep: get_all_valid_edges failed for group_id=%s',
            project_id,
        )
        stats['errors'] += 1
        return stats

    edges = flatten_dedup_edges(grouped)
    stats['scanned'] = len(edges)

    # Extract each edge's ids exactly once — reused below both to build
    # candidate_ids and (via select_stale_status_snapshot_edges's edge_ids
    # kwarg) for the final selection, instead of re-running the extraction
    # regex pipeline a second time per edge. (amendment,
    # reviewer_comprehensive efficiency finding, task 2613)
    edge_ids: dict[str, set[int]] = {
        edge['uuid']: extract_snapshot_edge_task_ids(edge.get('fact') or '') for edge in edges
    }

    candidate_ids: set[int] = set()
    for ids in edge_ids.values():
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

    stale = select_stale_status_snapshot_edges(edges, statuses, edge_ids=edge_ids)
    stats['candidate_edges'] = len(stale)

    invalidate_at = now or datetime.now(UTC)
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

    return stats
