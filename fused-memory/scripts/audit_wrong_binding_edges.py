#!/usr/bin/env python3
"""Retrospective read-only sweep for WRONG-BINDING extracted edges in Graphiti.

Background
----------
Escalation ``esc-4639-1`` split extraction defects into two families, and they
need different fixes:

* fact-CONTENT overreach — the ``fact`` text itself asserts more than the
  source episode said. ``scripts/audit_unverified_completion_claims.py``
  answers the retrospective half of that family.
* fact-PLACEMENT wrong-binding — the fact is a faithful restatement, but the
  edge is ATTACHED to the wrong entity. Reading it off the node it hangs from
  therefore attributes a true statement to the wrong subject.

This script answers the second family. The canonical specimen: reify node
``Task 6165`` carries five live ``RELATES_TO`` edges (``63fa5c78``,
``9a8e780b``, ``317da2e2``, ``6a79d29b``, ``9135f049``) whose facts are every
one of them about task **6164**. Their single source episode ``779b7b7d``
never mentions 6165 at all, and no node named ``Task 6164`` exists in that
graph. Anyone who read "the Task 6165 ruling" out of the graph was reading
task 6164's ruling.

Detection, and its exact relationship to the LIVE write-time guard
------------------------------------------------------------------
``MemoryService._verify_episode_referents`` already performs this check
post-write, as a ``set-membership`` test (memory_service.py:3524-3529): an
endpoint whose name is a task label must be among the referents its fact
names. It computes the correction and logs a warning; leaf ETA
(``ensure_entity_node`` -> ``reassign_edge`` -> ``refresh_entity_summary``)
that would ACT on it has no production wiring.

This sweep is the RETROSPECTIVE counterpart of that check, and it IMPORTS the
same detection vocabulary rather than re-deriving it — see
:func:`fact_referents` and :func:`endpoint_referent`, both thin adapters over
``fused_memory.utils.canonical_labels``. That is deliberate and structural:
the retrospective and live views of one defect must not be able to drift into
two different answers about what a task label is. INV-5 / task 3667 makes
"one compiled description of the vocabulary" an invariant, and
``tests/test_audit_wrong_binding_edges.py::TestNoSecondVocabulary`` pins it
for this script over the AST.

Safety
------
This script has NO mutation path at all. There is no ``--apply``, no
``--invalidate``, no ``--delete``, no ``--repair`` and no ``--reassign``.
Every edge the report indicts is left exactly as it is, for human
adjudication — 192 flags on the planning population, of which some are
legitimate cross-task relations ("BOOKMARK task 4043 tracks the ... work
surfaced by esc-3437-13" on node ``task 3443``), so auto-reassigning on a
regex verdict is precisely what the task's scope note forbids.
``TestReadOnlyByConstruction`` enforces the absence mechanically so a later
editor cannot quietly relax it. That test's forbidden-call list deliberately
names ``reassign_edge`` and ``merge_entities``: they are the lossless
remediation primitives a later editor would reach for on THIS defect class,
and remediation stays human-gated and out of scope here.

The graph is read exclusively over ``GRAPH.RO_QUERY``, where read-only is
SERVER-enforced rather than client-promised — never through ``MemoryService``
or ``GraphitiBackend``, whose handles can write, and never through
``graphiti_core.driver.falkordb_driver.FalkorDriver``, whose ``__init__``
fire-and-forgets ``build_indices_and_constraints()``.
"""
from __future__ import annotations

import difflib
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from fused_memory.backends.graphiti_client import (
    _DEFAULT_READ_PAGE_SIZE,
    _RESULTSET_SIZE,
    PagedRead,
    _paged_ro_query,
)
from fused_memory.utils.canonical_labels import (
    Referent,
    parse_node_name,
    scan_content,
)

# --------------------------------------------------------------------------- #
# The detection vocabulary — IMPORTED, never re-derived
# --------------------------------------------------------------------------- #


def fact_referents(fact: str | None, graph: str) -> frozenset[Referent]:
    """Every task referent *fact* NAMES, as a set.

    A thin adapter over ``canonical_labels.scan_content`` — the UNANCHORED
    half of the shared vocabulary, which answers "what does this prose refer
    to". The graph name doubles as the ``group_id``, which is what lets a
    self-qualified reference ('reify:5181' read inside the reify graph) be
    recognised as LOCAL while a genuinely foreign one ('dark_factory:2500')
    keeps its qualifier. Referents are compared in full, never by bare
    number, so those two can never be confused — the cross-project collapse
    ``utils/cross_project_refs.py`` exists to detect.

    KNOWN BLIND SPOTS, inherited verbatim from the shared scanner, which is
    documented "precision over recall" because its other consumers perform
    destructive edge surgery:

    * a node or reference written as BARE DIGITS ('1251') is invisible;
    * a reference made by task TITLE rather than by number is invisible;
    * Greek-letter and codename ALIASES are invisible;
    * a genuine project-qualified reference split across lines by HARD
      WRAPPING is missed ('dark_factory:\\n2500'), because the qualified
      pattern's '[ \\t]' padding cannot span a newline;
    * the HYPHEN spelling 'task-1836' is missed, because the separator
      alternation admits only whitespace, '#' and ':';
    * the BARE-HASH spelling '#4262' is missed, because a mention must carry
      the literal word 'task'.

    WIDENING ANY OF THESE IS OUT OF SCOPE HERE. They are properties of
    ``canonical_labels`` at its single site, and that module is on the LIVE
    memory write path — editing a pattern to suit a retrospective survey
    would change what the write-time guard does to production writes. The
    last two blind spots are the ones that actually bite this detector (they
    make an endpoint that IS named in the fact look unnamed), and they are
    absorbed by :func:`bare_id_present`, which mints no referent and can only
    SUPPRESS a flag.

    ``ambiguous`` referents are deliberately NOT returned. A referent lands
    there only when its number was claimed BOTH by a bare mention and by a
    foreign-qualified one, so the bare spelling guarantees the digits appear
    literally in the fact — which means :func:`bare_id_present` already
    covers exactly that population, in the conservative direction.

    Args:
        fact: The edge's ``r.fact`` text. NULL/empty yields an empty set — a
            NULL column must not abort a whole-corpus sweep.
        graph: The graph being swept, used as the scan's ``group_id``.
    """
    if not fact:
        return frozenset()
    return frozenset(scan_content(fact, group_id=graph).refs)


def endpoint_referent(node_name: str | None) -> Referent | None:
    """The task referent an edge ENDPOINT's name denotes, or None.

    A one-line delegation to the IMPORTED ``canonical_labels.parse_node_name``
    — the ANCHORED half of the shared vocabulary, which answers "is this
    entity NAME a task label?" rather than "does this text mention a task".
    Anchoring is what keeps a name that merely CONTAINS a reference ('Task 42
    orchestrator', 'reify task 12') out of the population entirely, so no such
    node can ever be flagged.

    A FOREIGN-qualified name ('reify:132') yields a referent that keeps its
    qualifier and is therefore NOT equal to a local ``Task 132``. The detector
    compares full :class:`Referent` objects, never bare numbers, so the
    cross-project collapse ``utils/cross_project_refs.py`` exists to detect
    can never be introduced here by the comparison itself.

    The falsy guard is not decoration: this sweep reads EVERY live
    ``RELATES_TO`` row in a graph, and a node with a NULL ``name`` arrives
    here as None. ``parse_node_name`` would raise ``TypeError`` on it, aborting
    a whole-corpus read over one odd historical row.
    """
    if not node_name:
        return None
    return parse_node_name(node_name)


def bare_id_present(referent: Referent, fact: str | None) -> bool:
    """Does *referent*'s already-parsed id appear as a standalone digit run?

    THE ONE ID CHECK THIS SCRIPT PERFORMS ITSELF, and deliberately the
    narrowest one that closes the gap. Read the next paragraph before
    concluding it violates INV-5.

    IT IS NOT A SECOND VOCABULARY. It compiles no task-label pattern — no
    'task', no separator alternation, no qualifier rule — and it can MINT no
    referent: the id it looks for was produced by
    :func:`endpoint_referent`, i.e. by the shared parser. It is a pure
    containment question about digits. Its only effect on the detector is to
    SUPPRESS a flag, never to create one, which is the conservative direction
    for a report whose every row a human adjudicates by hand.
    ``tests/test_audit_wrong_binding_edges.py::TestNoSecondVocabulary`` pins
    the distinction structurally: the pattern below contains no 'task'.

    WHY IT IS NEEDED. ``scan_content`` is documented "precision over recall",
    and two of its blind spots bite this detector specifically: the bare-hash
    spelling ('#4262') and the hyphen spelling ('task-1836'). Both are cases
    where the fact DOES name the endpoint and the shared scanner cannot see
    it, so without this check the endpoint would be flagged as mis-bound —
    a false positive in exactly the direction that costs a reader time. The
    fix is NOT to widen ``canonical_labels`` (that module is on the live
    memory write path; a survey's convenience must not change what the
    write-time guard does to production writes) and NOT to add a second
    mention pattern here (that IS the INV-5 violation).

    Word-boundary matched, so a digit run that is a SUBSTRING of a longer id
    never counts: an endpoint ``Task 616`` is not "named" by a fact about task
    6165. Without that, the suppression would silently swallow real findings
    in exactly the near-miss population this sweep exists to measure.

    A FOREIGN referent is matched on its NUMBER alone, since containment
    cannot see projects. That makes 'reify:132' look named by a fact saying
    'task 132' — a suppression, never an invented flag, and the precise
    comparison still happens in the set-membership test upstream.
    """
    if not fact:
        return False
    return re.search(rf'\b{re.escape(referent.number)}\b', fact) is not None


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Finding:
    """One endpoint the fact hanging off it does not name.

    Frozen for the same reason :class:`Referent` is: a finding is evidence a
    human adjudicates and, if they act, evidence for destructive edge surgery.
    A consumer must not be able to rewrite which edge or which end it names.

    ``end`` is ``'subject'`` or ``'object'`` — WHICH endpoint is mis-bound,
    not merely that the edge is. Specimen ``8a51e13b`` is why the distinction
    is a first-class column: its subject is correctly named and its OBJECT is
    the mis-bound end, so a report that only said "this edge is suspect"
    would send a reader to the wrong node.

    ``fact_referents`` is a sorted TUPLE, not the frozenset
    :func:`fact_referents` returns: tuples are ordered (so the report is
    byte-stable across runs) and hashable (so ``frozen=True`` is not
    advertising an immutability the field does not have).

    ``episodes`` carries ``r.episodes`` verbatim. It is the re-derivation
    path, and load-bearing: anyone who read "the Task 6165 ruling" out of the
    graph was reading task 6164's ruling, so re-deriving the truth means
    going back to the SOURCE episode rather than to the node.

    The three CAUSE-ATTRIBUTION columns default to None because the pure
    layer cannot compute them: ``proximity``/``nearest_id`` need the fact's
    id set (available, but computed by :func:`id_proximity` as a separate,
    separately-testable concern) and ``correct_node_present`` needs the
    graph's whole task-node census, which only the reader can supply. None
    means NOT COMPUTED, and is distinct from a computed 'unrelated'/False.
    """

    edge_uuid: str
    graph: str
    end: str
    node_name: str
    node_referent: Referent
    fact_referents: tuple[Referent, ...]
    fact: str
    episodes: tuple[str, ...] = ()
    proximity: str | None = None
    nearest_id: str | None = None
    correct_node_present: bool | None = None

    def to_json(self) -> dict[str, Any]:
        """The report row: referents rendered as their canonical node names.

        ``node_referent`` serializes as BOTH its rendered name and its bare
        number: the name is what a reader greps the graph for, the number is
        what :func:`id_proximity` compares. Rendering only one would make the
        artifact answer only half the questions asked of it.
        """
        return {
            'edge_uuid': self.edge_uuid,
            'graph': self.graph,
            'end': self.end,
            'node_name': self.node_name,
            'node_task_id': self.node_referent.number,
            'node_referent': self.node_referent.node_name,
            'fact_referents': [r.node_name for r in self.fact_referents],
            'fact': self.fact,
            'episodes': list(self.episodes),
            'proximity': self.proximity,
            'nearest_id': self.nearest_id,
            'correct_node_present': self.correct_node_present,
        }


def vars_of(finding: Finding) -> dict[str, Any]:
    """Field dict for a slots dataclass (``vars()`` does not work on slots).

    The same accommodation ``audit_unverified_completion_claims.py`` makes,
    and for the same reason: the enrichment pass rebuilds each finding with
    ``Finding(**{**vars_of(f), 'proximity': ...})``, which needs a field dict.
    """
    return {
        f: getattr(finding, f)
        for f in Finding.__dataclass_fields__  # type: ignore[attr-defined]
    }


def _sorted_referents(referents: frozenset[Referent]) -> tuple[Referent, ...]:
    """Referents in a deterministic order, so the report diffs cleanly.

    Sorted on (project_id, number-as-int-when-numeric, number) rather than on
    the rendered name: 'Task 132' and 'Task 6165' sort the wrong way as
    strings, and a report a human diffs run to run should read in id order.
    """
    return tuple(
        sorted(referents, key=lambda r: (r.project_id, len(r.number), r.number))
    )


def classify_edge(
    subject_name: str | None,
    object_name: str | None,
    fact: str | None,
    edge_uuid: str,
    graph: str,
    *,
    episodes: list[str] | tuple[str, ...] | None = None,
) -> list[Finding]:
    """Every endpoint of one edge that the *fact* fails to name.

    THE RULE, mirroring the write-time guard's ``set-membership`` check at
    ``memory_service.py:3524-3529`` deliberately, so the retrospective and
    live views of one defect cannot drift into two different verdicts: for
    EACH endpoint whose name parses as a task label, if the fact names at
    least one referent and that endpoint's referent is NOT among them (and
    its bare id does not appear in the fact either), emit a Finding for that
    end.

    THREE PROPERTIES A READER MUST NOT MISREAD:

    (a) A fact naming ZERO referents is UNVERIFIABLE, not clean. There is
        nothing to compare the endpoint against, so no verdict is possible in
        either direction. :func:`build_report` excludes that population from
        the DENOMINATOR for the same reason — folding it in would understate
        the rate by dividing by a population most of which was never
        adjudicated.

    (b) BOTH endpoints are checked, not just the subject. Specimen
        ``8a51e13b`` is ``(Task 6080) -> (Task 6128)`` with a fact naming
        6126 and 6080: the subject is correctly named and the OBJECT is the
        mis-bound end. A subject-only rule — the one the task description
        proposes — reports that edge clean.

    (c) Referents are compared in FULL, never by bare number, so a foreign
        ``reify:132`` never satisfies a local ``Task 132``. Collapsing them
        is the cross-project bug ``utils/cross_project_refs.py`` exists to
        detect, and a detector that made the same collapse could not see it.

    Subject is examined before object so a two-finding edge lists its ends in
    graph order rather than in whatever order a set iterated.
    """
    named = fact_referents(fact, graph)
    if not named:
        return []

    sorted_named = _sorted_referents(named)
    episode_uuids = tuple(episodes or ())
    findings: list[Finding] = []
    for end, name in (('subject', subject_name), ('object', object_name)):
        referent = endpoint_referent(name)
        if referent is None:
            continue
        if referent in named:
            continue
        if bare_id_present(referent, fact):
            continue
        findings.append(
            Finding(
                edge_uuid=edge_uuid,
                graph=graph,
                end=end,
                node_name=str(name),
                node_referent=referent,
                fact_referents=sorted_named,
                fact=fact or '',
                episodes=episode_uuids,
            )
        )
    return findings


# --------------------------------------------------------------------------- #
# Cause attribution
# --------------------------------------------------------------------------- #

PROXIMITY_BUCKETS: tuple[str, ...] = (
    'one_digit_diff',
    'prefix',
    'similar',
    'unrelated',
)
"""The four buckets, in PRECEDENCE order (closest first).

Load-bearing rather than documentation: :func:`id_proximity` ranks candidates
by index into this tuple, and :func:`build_report` seeds ``by_proximity`` from
it so every bucket is present with a 0 rather than absent when empty.
"""

_SIMILAR_RATIO = 0.75
"""``difflib.SequenceMatcher`` ratio above which a pair counts as 'similar'.

STRICTLY ABOVE, not at-or-above, and that is a measured decision rather than
a style choice. ``SequenceMatcher(None, '3443', '4043').ratio()`` is EXACTLY
0.75, and that pair is a live specimen — "BOOKMARK task 4043 tracks the ...
work surfaced by esc-3437-13" on node ``task 3443`` — which is a legitimate
cross-task relation, not a near-miss id neighbour. A ``>=`` comparison would
sweep it, and the rest of the equal-length two-digits-apart population, into
the near-miss buckets and inflate the very cause evidence this column exists
to supply. The planning figure ("13 more at >=0.75") was measured before that
boundary case was adjudicated.
"""


def _id_sort_key(task_id: str) -> tuple[int, str]:
    """Numeric-then-lexicographic order, for deterministic tie-breaking.

    Ids are digit strings, so ``int()`` is the order a human means by "the
    lowest id" — '99' before '100', which string order gets backwards. The
    string tail keeps the key TOTAL, so '0132' and '132' (different referents
    by design; digits are never int-normalized) still order deterministically.
    Non-numeric input cannot arrive from the shared parser, but is ordered
    last rather than raising into a whole-corpus sweep.
    """
    return (int(task_id) if task_id.isdigit() else 1 << 62, task_id)


def _pair_bucket(node_task_id: str, candidate: str) -> str:
    """Which bucket ONE (mis-bound id, named id) pair falls in."""
    if len(node_task_id) == len(candidate):
        if sum(a != b for a, b in zip(node_task_id, candidate, strict=True)) == 1:
            return 'one_digit_diff'
    elif node_task_id.startswith(candidate) or candidate.startswith(node_task_id):
        return 'prefix'
    if difflib.SequenceMatcher(None, node_task_id, candidate).ratio() > _SIMILAR_RATIO:
        return 'similar'
    return 'unrelated'


def id_proximity(
    node_task_id: str, named_ids: Collection[str]
) -> tuple[str, str]:
    """How close is the MIS-BOUND id to the nearest id the fact names?

    Returns ``(bucket, nearest_id)``. The buckets, in precedence order:

    * ``one_digit_diff`` — equal length, exactly ONE differing character.
      The signature of a near-miss neighbour, e.g. ``Task 6165`` carrying a
      fact about task 6164.
    * ``prefix`` — one id is a strict prefix of the other ('430' / '4302').
      A DIFFERENT mechanism from one_digit_diff (a truncated id rather than a
      mistyped digit), which is why the two are not collapsed.
    * ``similar`` — ``difflib.SequenceMatcher`` ratio strictly above
      :data:`_SIMILAR_RATIO`; read that constant before changing the
      comparison.
    * ``unrelated`` — everything else, including legitimate cross-task
      relations that merely happen to hang off a task-shaped node.

    WHY THIS LIVES IN THE SCRIPT rather than in a one-off notebook: it is the
    EVIDENCE that separates "resolution grabbed a near-miss id neighbour"
    from "unrelated mis-attachment", and the report's whole cause argument
    rests on its distribution. The planning measurement put 120/192 (62.5%)
    of flags in the one_digit_diff + prefix + similar buckets — against a
    chance baseline near zero over ~2090 (reify) / ~1452 (dark_factory)
    task-shaped nodes. A number that load-bearing has to be reproducible by
    re-running the committed script, not by trusting a transcript.

    Ties are broken on the numerically LOWEST id, so the report is
    byte-stable across runs and cannot depend on set iteration order. An
    empty *named_ids* yields ``('unrelated', '')`` — defensive only:
    :func:`classify_edge` never emits a finding for a fact naming nothing.
    """
    best: tuple[int, tuple[int, str], str] | None = None
    for candidate in named_ids:
        bucket = _pair_bucket(node_task_id, candidate)
        key = (PROXIMITY_BUCKETS.index(bucket), _id_sort_key(candidate), candidate)
        if best is None or key < best:
            best = key
    if best is None:
        return ('unrelated', '')
    return (PROXIMITY_BUCKETS[best[0]], best[2])


def correct_node_present(nearest_id: str, task_node_ids: Collection[str]) -> bool:
    """Does a node for the id the fact ACTUALLY names already exist?

    True when *nearest_id* is among the task ids harvested from this graph's
    ``Entity`` node names (built by :meth:`EdgeReader.read_task_node_ids`).

    THE MEASUREMENT THIS COLUMN EXISTS FOR: it separates "the correct node
    was missing, so resolution had nothing right to pick" from ACTIVE
    mis-resolution — resolution choosing a wrong node over an available
    correct one. The planning sweep found the correct node ALREADY PRESENT in
    124/194 (64%) of endpoint checks, which is what makes the cause a
    resolution defect rather than a coverage gap.

    The canonical specimen is the OTHER 36%: node ``Task 6164`` does not
    exist in reify, while ``Task 6165`` — created three days before the
    episode — does. So the specimen everyone quotes is not representative of
    the population, and the report has to carry both counts for that to be
    visible.

    An empty *nearest_id* names no node and is False: nothing can be present.
    """
    if not nearest_id:
        return False
    return nearest_id in task_node_ids


# --------------------------------------------------------------------------- #
# The read seam — GRAPH.RO_QUERY, paged, no graphiti driver
# --------------------------------------------------------------------------- #

DEFAULT_GRAPH = 'dark_factory'
"""The graph swept when --graph is not given."""

RO_COMMAND = 'GRAPH.RO_QUERY'
"""The only FalkorDB command this script is permitted to issue.

Read-only here is SERVER-enforced, not client-promised: a ``CREATE`` issued
through this command path is refused by a live FalkorDB and materializes
nothing. For a sweep whose entire premise is "do not reassign an edge on a
regex verdict", that is the difference between a promise and a guarantee.

Which is why this reads over Cypher rather than through ``MemoryService`` /
``GraphitiBackend`` (both reach the store through a handle that CAN write) and
never constructs ``graphiti_core.driver.falkordb_driver.FalkorDriver``, whose
``__init__`` fire-and-forgets ``build_indices_and_constraints()`` — a WRITE,
issued before a read-only sweep had read anything.
"""

_EDGE_MATCH = (
    'MATCH (a)-[r:RELATES_TO]->(b) '
    'WHERE r.invalid_at IS NULL AND r.expired_at IS NULL '
)

EDGE_PAGE_CYPHER = (
    _EDGE_MATCH
    + 'RETURN a.name, b.name, r.uuid, r.fact, r.episodes '
    'ORDER BY r.uuid SKIP {skip} LIMIT {limit}'
)
"""One page of LIVE ``RELATES_TO`` edges.

``r.fact_embedding`` is deliberately NOT projected: ~1500 floats per edge over
15256 edges, for a text detector that never looks at them.

``ORDER BY r.uuid`` is a TOTAL order over the matched population, and is
load-bearing rather than cosmetic — see ``_paged_ro_query``'s docstring: each
page is a separate query, so without a total order SKIP n on page 2 can skip
rows page 1 never returned, silently and permanently.
"""

EDGE_CENSUS_CYPHER = _EDGE_MATCH + 'RETURN count(*)'
"""The identical MATCH/WHERE as a single-row count, so the two numbers
describe the same population. A single-row aggregate can never be truncated by
the row cap it is being used to detect."""

_NODE_MATCH = 'MATCH (n:Entity) '

NODE_PAGE_CYPHER = _NODE_MATCH + 'RETURN n.name ORDER BY n.name SKIP {skip} LIMIT {limit}'
"""One page of Entity names, filtered to task labels in Python.

The filter cannot run in Cypher: "is this name a task label" is the imported
anchored parser's question, and re-expressing it as a Cypher predicate would
be exactly the second vocabulary INV-5 forbids.
"""

NODE_CENSUS_CYPHER = _NODE_MATCH + 'RETURN count(*)'


class EdgeReader:
    """Reads one graph's live ``RELATES_TO`` edges and task-node ids.

    Constructed either with an explicit *graph* handle (the tests pass a
    double) or with a uri + graph name, in which case a ``falkordb.asyncio``
    client is opened lazily on first use. ``falkordb.FalkorDB`` — the DB
    client — is categorically distinct from graphiti's ``FalkorDriver``; see
    :data:`RO_COMMAND` for why the latter must never be constructed here.

    BOTH READS ARE PAGED, and that is a DELIBERATE DIVERGENCE from
    ``audit_unverified_completion_claims.py``, which issues one unpaginated
    ``MATCH (e:Episodic)``. That is correct THERE — its population (2976
    dark_factory / 4547 reify) sits under the server's 10000-row
    ``RESULTSET_SIZE``. It is wrong HERE: reify holds 15256 live RELATES_TO
    rows, so an unpaginated read returns exactly 10000 of them, silently, and
    every denominator in the report would be wrong. Verified during planning
    against the live store — a bare MATCH returned 10000 while
    ``_paged_ro_query`` returned ``rows_seen=15256 expected_rows=15256
    complete=True``.

    *page_size* and *resultset_size* are injectable so a test can exercise the
    paging behaviour against a small cap; the defaults are the module-level
    constants ``_paged_ro_query`` itself uses, so production runs cannot drift
    from the shared primitive's tuning.
    """

    def __init__(
        self,
        *,
        graph: Any | None = None,
        graph_name: str = DEFAULT_GRAPH,
        uri: str | None = None,
        page_size: int = _DEFAULT_READ_PAGE_SIZE,
        resultset_size: int = _RESULTSET_SIZE,
    ) -> None:
        self._graph = graph
        self.graph_name = graph_name
        self.uri = uri
        self.page_size = page_size
        self.resultset_size = resultset_size

    @staticmethod
    def assert_read_only_command(command: str) -> None:
        """Raise unless *command* is :data:`RO_COMMAND`.

        A client-side guard layered on the server-side one, so a violation is
        a typed error at the seam that owns the guarantee rather than a redis
        error surfacing three layers down.
        """
        if command != RO_COMMAND:
            raise RuntimeError(
                f'this sweep may only issue {RO_COMMAND}, refused {command!r} '
                f'— it is strictly read-only'
            )

    def _resolve_graph(self) -> Any:
        """Return the graph handle, opening a client on first use.

        The import is local so that merely importing this module — which the
        tests do, with a double in hand — never requires falkordb to be
        installed or a store to be running.
        """
        if self._graph is None:
            from falkordb.asyncio import FalkorDB  # noqa: PLC0415

            client = FalkorDB.from_url(self.uri) if self.uri else FalkorDB()
            self._graph = client.select_graph(self.graph_name)
        return self._graph

    async def _read(self, page: str, census: str) -> PagedRead:
        self.assert_read_only_command(RO_COMMAND)
        return await _paged_ro_query(
            self._resolve_graph(),
            page,
            census,
            page_size=self.page_size,
            resultset_size=self.resultset_size,
        )

    async def fetch_edges(self) -> tuple[list[list], PagedRead]:
        """Every live ``RELATES_TO`` row, plus the completeness proof.

        The PagedRead is returned ALONGSIDE the rows rather than folded into
        them, so a caller can never infer completeness from a row count. An
        incomplete read is a fact the report must publish (``truncated_by``),
        not an exception — a census disagreeing by a handful of rows is the
        expected signature of a live graph being written to mid-read.
        """
        read = await self._read(EDGE_PAGE_CYPHER, EDGE_CENSUS_CYPHER)
        return list(read.rows), read

    async def read_task_node_ids(self) -> tuple[set[str], PagedRead]:
        """The task ids named by this graph's ``Entity`` nodes.

        Feeds :func:`correct_node_present`. Names are parsed with the IMPORTED
        anchored parser, so a FOREIGN 'reify:132' node inside dark_factory
        contributes NOTHING to the local id set — harvesting its bare number
        would make ``correct_node_present`` claim a local ``Task 132`` exists
        when it does not.

        Paged for the same reason the edge read is: dark_factory measured
        16083 Entity nodes and reify 23616 (2026-08-17), both far above the
        10000 cap, and a truncated node census makes this column answer False
        for every node past the cut — manufacturing exactly the "the correct
        node was missing" conclusion it exists to test.
        """
        read = await self._read(NODE_PAGE_CYPHER, NODE_CENSUS_CYPHER)
        ids: set[str] = set()
        for row in read.rows:
            referent = endpoint_referent(row[0] if row else None)
            if referent is not None and not referent.project_id:
                ids.add(referent.number)
        return ids, read
