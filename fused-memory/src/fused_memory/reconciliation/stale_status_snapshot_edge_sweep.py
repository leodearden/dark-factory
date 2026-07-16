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
- Invalidate-only-on-positively-terminal: an unknown/missing/still-active
  status for a referenced id never triggers invalidation (fail-safe,
  mirrors flag_dedup.filter_terminal_metadata_flags) — a transient census
  hiccup can only under-invalidate (self-heals next cycle), never wrongly
  retire a valid edge.
- An aggregate snapshot edge ("the active pending tasks are [A, B, C]") is
  invalidated as a whole the moment ANY one referenced id is now terminal —
  the snapshot as asserted no longer holds.
- Best-effort throughout (modelled on
  ``degenerate_task_node_sweep.sweep_degenerate_task_nodes``): a transient
  backend error enumerating, cross-referencing, or invalidating one edge
  must not abort the sweep for the remaining edges — it is tallied into
  ``stats['errors']`` and the sweep continues.
"""

from __future__ import annotations

import logging
import re

from fused_memory.reconciliation.task_filter import TASK_REF_RE

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# extract_snapshot_edge_task_ids — pure lexical extraction
# --------------------------------------------------------------------------- #

# Gate: a status-snapshot edge must assert one of these non-terminal
# statuses somewhere in its fact text, else it is not a status-snapshot edge
# this sweep concerns itself with at all (e.g. 'Task 5 is done' — already
# terminal at write time, not what this sweep targets).
SNAPSHOT_STATUS_RE: re.Pattern[str] = re.compile(
    r'\b(?:active|pending|in[-\s]?progress)\b',
    re.IGNORECASE,
)

# Strips 'N <count-noun>' spans (e.g. '8 tasks', '148 cancelled', '3 pending')
# BEFORE id extraction, so a count operand is never mistaken for a task id —
# this is what keeps pure task-COUNT snapshots ('There are 8 tasks in
# progress') from ever contributing a candidate id.
COUNT_QUANTITY_RE: re.Pattern[str] = re.compile(
    r'\b\d+\s+(?:tasks?|active|pending|in[-\s]?progress|done|cancell?ed|'
    r'blocked|deferred|review|total|merge[-\s]?deferred)\b',
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
    """Return the task ids *fact* asserts as active/pending/in-progress.

    Returns the empty set (never a candidate for invalidation) when:
    - *fact* contains no active/pending/in-progress status marker at all
      (e.g. 'Task 5 is done', 'Task 7 landed as merge commit'); or
    - *fact* is a pure count-only snapshot with no specific task-id
      reference (e.g. 'There are 8 tasks in progress', '1505 done / 148
      cancelled') — out of scope per Snapshot Discipline.

    Algorithm:
      1. Gate on SNAPSHOT_STATUS_RE against the raw fact text; short-circuit
         to the empty set when absent.
      2. Strip COUNT_QUANTITY_RE spans so a count operand ('8' in '8
         tasks') is never treated as an id.
      3. Extract ids via the union of:
         - TASK_REF_RE matches (individual form: 'task N' / '#N' / 'df N'),
           imported from task_filter for consistency with the rest of the
           reconciliation detector family; and
         - bare digit tokens found ONLY inside a detected aggregate list
           segment (introduced by 'tasks are [...]' / 'tasks: ...').

    Pure: no I/O, no side effects.
    """
    fact = fact or ''
    if not SNAPSHOT_STATUS_RE.search(fact):
        return set()

    stripped = COUNT_QUANTITY_RE.sub(' ', fact)

    ids: set[int] = {int(m) for m in TASK_REF_RE.findall(stripped)}

    for intro in LIST_INTRODUCER_RE.finditer(stripped):
        segment = _list_segment(stripped, intro.end(), intro.group('open'))
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
