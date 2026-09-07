"""Post-write sanity check for extracted dependency-DIRECTION facts (task 3770).

THE MECHANISM THIS TARGETS. Graphiti's extraction LLM, reading an episode that
mentions several adjacent task numbers, does not invent task numbers — every id
it names is real and genuinely nearby. What it gets wrong is the DIRECTION of
the relation between them, in two recurring shapes:

  FLATTENING — two tasks that are merely PARALLEL (siblings sharing a dependent,
  or sharing a dependency) are re-stated as SEQUENTIAL. Live example from
  Graphiti episode f3d18584-4041-4397-9faa-d4a14c01f71d: "Task 3727 waits behind
  task 3619" — 3727's only dependency is 3256; 3619 and 3727 merely share the
  DEPENDENT 3578. And "Task 3730 needs task 3733" — neither depends on the
  other; they share the DEPENDENCIES 3728 and 3578.

  INVERSION — a TRANSITIVE chain is stated backwards. Same episode: "Task 3618
  waits behind task 3578", where in fact 3618 has no dependencies at all and
  3578 reaches 3618 transitively (3578 -> 3619 -> 3618). Note this one is
  INVISIBLE to a direct-edge check: 3618 is not in direct(3578) either. Only the
  transitive closure exposes it.

Because every number is real and adjacent, the resulting fact READS as plausible
and a planning read accepts it without friction — which makes this strictly more
dangerous than random hallucination, and is the whole reason for checking.

SCOPE. Compact multi-task chain/parallel dependency shorthand ONLY. The gate is
``extract_dependency_assertions`` itself: a fact carrying no direction-bearing
phrase between two task references yields ``[]``, and the caller then performs
NO ground-truth read at all. A blanket check on every write would cost far more
for far less yield — the common path here is one regex scan per edge and zero
I/O.

PURE LEAF. No I/O, stdlib plus the two shared regexes from
``fused_memory.reconciliation.task_filter``. The ground-truth graph is passed IN
as data (``build_dependency_index``), so the entire decision core is testable
with a plain dict and the only I/O lives in the thin ``MemoryService`` adapter.
Same convention as ``middleware/candidate_key.py`` and
``middleware/recon_claim_verification_guard.py``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from fused_memory.reconciliation.task_filter import (
    STRICT_CLAUSE_BOUNDARY_RE,
    TASK_REF_RE,
)

# ── Phrase vocabulary ──────────────────────────────────────────────────────
#
# ONE alternation constant per direction, and both feed BOTH the cheap gate
# (`_PHRASE_GATE_RE`) and the positional binder (`_PHRASE_GAP_RE`). That is
# deliberate, and it is the structure `stale_status_snapshot_edge_sweep`'s
# task-3042 amendment exists to enforce: when a gate is spelled independently of
# the matchers it admits, the two drift, and the gate silently starts refusing
# facts the matcher would have classified. A phrase added below is added to the
# gate and to the binder in the same edit, because there is only one spelling.

#: Phrases where the FIRST task reference is the DEPENDENT — the thing that
#: waits — and the second is the DEPENDENCY.
_FORWARD_PHRASE_ALT = (
    r'(?:'
    r'waits?\s+behind'
    r'|waiting\s+behind'
    r'|needs?'
    r'|requires?'
    r'|depends?\s+(?:up)?on'
    r'|dependent\s+(?:up)?on'
    r'|blocked\s+(?:by|on)'
    r'|waits?\s+(?:on|for)'
    r'|waiting\s+(?:on|for)'
    r')'
)

#: Phrases where the direction is REVERSED relative to word order — the FIRST
#: reference is the DEPENDENCY and the second is the DEPENDENT. Direction is
#: read from the PHRASE, never from which id was written first; that is the
#: whole point of keeping two alternations rather than one.
_INVERSE_PHRASE_ALT = (
    r'(?:'
    r'blocks?'
    r'|blocking'
    r'|gates?'
    r'|gating'
    r'|unblocks?'
    r'|a\s+(?:hard\s+|blocking\s+)?dependency\s+(?:of|for)'
    r'|a\s+(?:hard\s+|blocking\s+)?prerequisite\s+(?:of|for)'
    r'|depended\s+(?:up)?on\s+by'
    r')'
)

#: Optional copula, so "is blocked by" / "is a dependency of" bind through the
#: same two alternations as their bare forms rather than needing their own arms.
_COPULA_ALT = r'(?:(?:is|are|was|were|remains?|stays?|sits?)\s+)?'

#: Closed-class adverbs that may sit between the copula and the phrase without
#: changing its meaning ("is still blocked by", "directly blocks").
_ADVERB_ALT = r'(?:(?:currently|still|now|already|directly|transitively)\s+)*'

#: The cheap whole-fact gate. Built from the SAME two alternations the binder
#: uses, so it can never be narrower than what the binder would accept.
_PHRASE_GATE_RE: re.Pattern[str] = re.compile(
    r'\b(?:' + _FORWARD_PHRASE_ALT + r'|' + _INVERSE_PHRASE_ALT + r')',
    re.IGNORECASE,
)

#: The positional binder. Matched against the WHOLE GAP between two ADJACENT
#: task references (anchored \A ... \Z), so a phrase only binds a pair it sits
#: literally between — no long-range association across intervening prose, and
#: no binding to a reference that is not its immediate neighbour.
_PHRASE_GAP_RE: re.Pattern[str] = re.compile(
    r'\A\s*' + _COPULA_ALT + _ADVERB_ALT
    + r'(?:(?P<forward>' + _FORWARD_PHRASE_ALT + r')'
    + r'|(?P<inverse>' + _INVERSE_PHRASE_ALT + r'))\s*\Z',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DependencyAssertion:
    """One directional dependency claim read out of an extracted fact.

    ``dependent`` waits for ``dependency`` — always in that normalised
    orientation, whichever way round the source phrase wrote it. ``phrase`` is
    the matched connective VERBATIM, kept so a finding can quote the extraction's
    own wording rather than a paraphrase of it.

    Frozen: an assertion is evidence that may justify retiring a Graphiti edge,
    so it must not be widenable after construction.
    """

    dependent: int
    dependency: int
    phrase: str


def extract_dependency_assertions(fact: object) -> list[DependencyAssertion]:
    """Return the directional dependency claims *fact* makes, or ``[]``.

    THIS IS THE SCOPE GATE. An empty return means "this write is out of scope",
    and the caller must then perform no ground-truth read at all — the check is
    restricted to compact multi-task dependency shorthand precisely so the
    overwhelmingly common write costs one regex scan and zero I/O.

    Algorithm:

    1. Cheap whole-fact gate on ``_PHRASE_GATE_RE``. No direction-bearing phrase
       anywhere -> ``[]`` without splitting or scanning for references.
    2. Split on ``task_filter.STRICT_CLAUSE_BOUNDARY_RE`` and bind only WITHIN a
       clause. A phrase never reaches across a sentence boundary to a reference
       in a different clause.
    3. Within a clause, collect ``TASK_REF_RE`` matches positionally and test
       each ADJACENT pair's gap against ``_PHRASE_GAP_RE``, anchored end to end.
       Adjacency is what makes a comma-chained fact ("A waits behind B, C needs
       D") yield exactly the two intended pairs and not the four a
       cross-product would produce.
    4. A ``forward`` gap yields ``(first, second)``; an ``inverse`` gap yields
       the SWAPPED ``(second, first)``.

    WHY ``STRICT_CLAUSE_BOUNDARY_RE`` AND NOT ``_CLAUSE_SPLIT_RE``: the
    canonical rationale for that divergence lives beside the constant itself.
    This module's stake in it is the same fail-safe direction
    ``stale_status_snapshot_edge_sweep._list_segment`` documents — an assertion
    read out here can end in ``memory_service.update_edge(invalid_at=...)``, a
    real Graphiti edge retirement, so a LONGER clause absorbing more incidental
    material is over-selection on a path where over-selection wrongly retires a
    true fact while under-selection merely self-heals on the next write. The
    accepted cost is the dotted-technical-token shatter task 3403 documents.

    ``TASK_REF_RE`` is imported, never re-spelled: it already carries the
    ``task N`` / ``task #N`` / ``task/N`` / ``df N`` / ``#N`` vocabulary and the
    anchoring that keeps ``get_task``, ``task_knowledge_sync`` and the plural
    ``.worktrees/tasks/94/`` path segment from reading as references. It has
    exactly ONE capture group and this module preserves that by using
    ``finditer`` + ``.group(1)`` plus match spans rather than ``findall``.

    Never raises: ``None``, a non-string, or an empty/whitespace fact returns
    ``[]``. A checker that can throw on malformed input is a checker that can
    fail an already-committed episode write.
    """
    if not isinstance(fact, str) or not fact.strip():
        return []
    if not _PHRASE_GATE_RE.search(fact):
        return []

    assertions: list[DependencyAssertion] = []
    for clause in STRICT_CLAUSE_BOUNDARY_RE.split(fact):
        refs = [
            (int(m.group(1)), m.start(), m.end())
            for m in TASK_REF_RE.finditer(clause)
        ]
        for (first, _, first_end), (second, second_start, _) in zip(
            refs, refs[1:], strict=False
        ):
            gap = clause[first_end:second_start]
            match = _PHRASE_GAP_RE.match(gap)
            if match is None:
                continue
            phrase = (match.group('forward') or match.group('inverse') or '').strip()
            if match.group('forward'):
                assertions.append(DependencyAssertion(first, second, phrase))
            else:
                assertions.append(DependencyAssertion(second, first, phrase))
    return assertions


@dataclass(frozen=True)
class DependencyIndex:
    """The ground-truth dependency graph, in the three shapes the classifier needs.

    ``direct``     — ``{task: frozenset(its own declared dependencies)}``
    ``dependents`` — the REVERSE adjacency, ``{task: frozenset(tasks that
                     depend on it)}``
    ``closure``    — the TRANSITIVE reachability set, ``{task: frozenset(every
                     task it transitively waits for)}``

    Every node appearing in EITHER edge column is a key in all three maps, so a
    leaf that nothing depends on is still a KNOWN node — that is what keeps the
    classifier's unknown-id fail-safe from firing on a perfectly ordinary leaf.

    Values are ``frozenset`` and the dataclass is frozen: a finding produced
    against this index may retire a Graphiti edge, so the graph it was justified
    against must not be widenable after the fact.
    """

    direct: Mapping[int, frozenset[int]]
    dependents: Mapping[int, frozenset[int]]
    closure: Mapping[int, frozenset[int]]


def build_dependency_index(edges: Mapping[int, Iterable[int]]) -> DependencyIndex:
    """Build a :class:`DependencyIndex` from a ``{task_id: [depends_on, ...]}`` map.

    Accepts the exact shape ``SqliteTaskBackend.get_dependency_edges`` returns —
    including its inherited contract that a task with NO dependencies is simply
    ABSENT from the map rather than present with an empty list. Nodes are
    therefore seeded from BOTH edge columns, so such a task is still a known
    node with an empty ``closure`` and a populated ``dependents``.

    The closure is computed by a memoized DFS carrying an on-stack set.
    Cycle-safety is a real requirement, not defensive padding: ``add_dependency``
    rejects only self-loops, not longer cycles, and there is no other
    transitive-closure or cycle-detection code over task dependencies anywhere in
    the repo (every existing consumer is strictly 1-hop). A malformed graph must
    TERMINATE the walk rather than blow the stack inside the Graphiti write-path
    identity lock. When a cycle is present each node on it reports the whole
    cycle as reachable, itself included — the honest answer for a graph that
    should not exist.

    Pure: no I/O, and the caller's mapping is never mutated.
    """
    direct: dict[int, set[int]] = {}
    dependents: dict[int, set[int]] = {}
    for task_id, deps in (edges or {}).items():
        node = int(task_id)
        direct.setdefault(node, set())
        dependents.setdefault(node, set())
        for raw_dep in deps or ():
            dep = int(raw_dep)
            direct[node].add(dep)
            # Seed from the VALUE column too: a task nothing depends on must
            # still be a known node, or the classifier's unknown-id fail-safe
            # would fire on every leaf.
            direct.setdefault(dep, set())
            dependents.setdefault(dep, set()).add(node)

    closure: dict[int, frozenset[int]] = {}
    #: Nodes currently being expanded, mapped to their DFS stack depth. This is
    #: the on-stack set that makes the walk cycle-safe; the depth is what makes
    #: memoization safe in the presence of one.
    on_stack: dict[int, int] = {}

    def _closure_of(node: int, depth: int) -> tuple[frozenset[int], int]:
        # Returns (reachable set, lowlink) where lowlink is the SHALLOWEST
        # stack depth any back-edge in this subtree pointed at. A frame whose
        # lowlink is its own depth saw no back-edge to a STRICT ANCESTOR, so
        # its set is complete and may be memoized; a frame truncated by such a
        # back-edge is correct only for this particular walk (its ancestor's
        # own frame supplies the rest) and must NOT be cached as that node's
        # answer. Without this distinction a cyclic graph memoizes a truncated
        # closure and the classifier then reads a real dependency as absent —
        # under-flagging silently instead of terminating loudly.
        cached = closure.get(node)
        if cached is not None:
            return cached, depth
        on_stack[node] = depth
        reachable: set[int] = set()
        lowlink = depth
        for dep in direct.get(node, ()):
            reachable.add(dep)
            ancestor_depth = on_stack.get(dep)
            if ancestor_depth is not None:
                # Back edge: this arm is already being expanded further up the
                # stack. Stop rather than recurse — that is what terminates a
                # malformed cyclic graph.
                lowlink = min(lowlink, ancestor_depth)
                continue
            sub, sub_lowlink = _closure_of(dep, depth + 1)
            reachable |= sub
            lowlink = min(lowlink, sub_lowlink)
        del on_stack[node]
        result = frozenset(reachable)
        if lowlink >= depth:
            closure[node] = result
        return result, lowlink

    for node in direct:
        if node not in closure:
            _closure_of(node, 0)

    return DependencyIndex(
        direct={k: frozenset(v) for k, v in direct.items()},
        dependents={k: frozenset(v) for k, v in dependents.items()},
        closure=dict(closure),
    )


#: THE closed classification vocabulary — the single normative site. A
#: classification is REGISTERED here, never spelled as a bare string at a call
#: site, so the finding record, the log line and every test read one vocabulary
#: rather than three that are free to drift.
#: An extraction that inverted a transitive chain: the ground truth runs the
#: OTHER way (``dependency`` reaches ``dependent``).
REVERSED = 'reversed'
#: An extraction that FLATTENED a parallel relation into a sequential one: the
#: two tasks are co-siblings, neither waits for the other.
SIBLING_SEQUENTIAL = 'sibling_sequential'
#: Both ids are known, no path runs either way, and they are not co-siblings —
#: the ground truth supports no relation between them at all.
UNSUPPORTED = 'unsupported'


def classify_dependency_assertion(
    assertion: DependencyAssertion, index: DependencyIndex
) -> str | None:
    """Classify *assertion* against ground truth, or ``None`` when it is fine.

    ``None`` means "do not flag" — either the claim is SUPPORTED, or it cannot
    be adjudicated and the fail-safe direction is to leave it alone.

    THE ORDER BELOW IS LOAD-BEARING. It is behaviour pinned by its own test, not
    an implementation detail a refactor may reorder:

    1. Either id absent from the index, or ``dependent == dependency`` -> ``None``.
       An unknown id means UNKNOWN, not WRONG. This mirrors
       ``stale_status_snapshot_edge_sweep``'s invalidate-only-on-positively-
       terminal doctrine: a transient or partial ground-truth read must be able
       to UNDER-flag (self-healing on the next write) rather than wrongly retire
       a true fact.
    2. ``dependency in closure[dependent]`` -> ``None``. SUPPORTED, directly or
       transitively.
    3. ``dependent in closure[dependency]`` -> ``REVERSED``. The chain runs the
       other way. Checked against the CLOSURE, not ``direct``, because the live
       inversion this exists to catch is invisible at one hop: "Task 3618 waits
       behind task 3578" is wrong precisely because 3578 reaches 3618 (via 3619)
       while 3618 depends on nothing.
    4. Co-sibling in EITHER DAG direction -> ``SIBLING_SEQUENTIAL``.
    5. Otherwise -> ``UNSUPPORTED``.

    WHY STEP 4 SPANS BOTH DAG DIRECTIONS. The escalation summarises the sibling
    case as "two tasks that share a dependent". Measured against the graph as of
    the bad extraction, that literal predicate catches 1 of the 3 known errors,
    not the 2 required: 3619/3727 do share the dependent 3578, but 3730/3733
    shared NO dependent at all — they shared the DEPENDENCIES {3728, 3578}. The
    task description's own prose gives both shapes explicitly ("both are
    dependencies of 3578" vs. "both depend on 3728 and 3578"), so covering both
    honours the stated intent rather than departing from it.

    WHY STEP 2 MUST PRECEDE STEP 4. ``direct[3730] & direct[3728] == {3727}``,
    so 3730 and 3728 ARE co-siblings under the shared-dependency arm — and
    ``3730 -> 3728`` is nonetheless a REAL direct edge. Evaluating the sibling
    arm first would flag a true fact, which is exactly the false-positive class
    the task's companion assertion forbids. Supported always wins.
    """
    dependent, dependency = assertion.dependent, assertion.dependency
    if dependent == dependency:
        return None
    if dependent not in index.closure or dependency not in index.closure:
        return None
    if dependency in index.closure[dependent]:
        return None
    if dependent in index.closure[dependency]:
        return REVERSED
    shares_dependent = index.dependents.get(dependent, frozenset()) & index.dependents.get(
        dependency, frozenset()
    )
    shares_dependency = index.direct.get(dependent, frozenset()) & index.direct.get(
        dependency, frozenset()
    )
    if shares_dependent or shares_dependency:
        return SIBLING_SEQUENTIAL
    return UNSUPPORTED
