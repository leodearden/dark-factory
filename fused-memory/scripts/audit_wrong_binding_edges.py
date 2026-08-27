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

Usage
-----
The two-graph sweep this task's artifact was produced by::

    uv run python scripts/audit_wrong_binding_edges.py \\
        --graph dark_factory --graph reify --json \\
        --out-dir ../docs/wrong-binding-edge-sweep-2026-08-27

As a cron gate, exiting non-zero when anything is found::

    uv run python scripts/audit_wrong_binding_edges.py --fail-on-finding

Exit codes: ``0`` ran, ``1`` infra failure (NOTHING is emitted — a truncated
report that looks complete is worse than none), ``2`` ``--fail-on-finding``
with at least one finding.
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import logging
import os
import re
import sys
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
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


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #

KNOWN_GAPS: tuple[dict[str, Any], ...] = (
    {
        'sub_class': 'direction_reversal',
        'description': (
            'The edge points the wrong way: BOTH endpoints are correctly '
            'named in the fact, only the subject/object roles are swapped.'
        ),
        'specimen_edge_uuids': ['1cf19488', '01e3ff5d'],
        'specimens': [
            '1cf19488 (Task 6346)->(Task 6347): "The recurring-attention task '
            '#6347 depends on task #6346."',
            '01e3ff5d (Task 5997)->(Task 6014): "Task 6014 carries task 5997 '
            'as a hard dependency."',
        ],
        'why_out_of_reach': (
            'Set-membership is SATISFIED at both ends, so no text rule can '
            'separate a reversal from correct grammatical voice.'
        ),
        'cheap_heuristic_measured_and_rejected': {
            'rule': 'leftmost task id named in the fact == object id != subject id',
            'flagged': 85,
            'population': 7131,
            'verdict': 'NOT SHIPPED — precision far too low',
            'worked_false_positives': [
                '"Task 2660 depends on Task 2659 landing" on edge (2659)->(2660)',
                '"Task 846 is the companion task to Task 839" on edge (839)->(846)',
                '"Themes addressed by follow-up Task 2083 are related to Task '
                '394" on edge (394)->(2083)',
            ],
            'what_would_be_needed': (
                'The AUTHORITATIVE task dependency graph. The two true '
                'specimens are indistinguishable from the benign majority by '
                'fact text alone, because both of their endpoints are named.'
            ),
        },
    },
    {
        'sub_class': 'fact_contradicts_source_episode',
        'description': (
            'The fact is bound to plausible endpoints and names them both, '
            'but disagrees with the episode it was extracted from.'
        ),
        'specimen_edge_uuids': ['993a9a7b'],
        'specimens': [
            '993a9a7b (Task 6004)->(Task 5997): "Task 6004\'s rulings were '
            'ported verbatim into task 5997."',
        ],
        'why_out_of_reach': (
            'Reachable by NO text or topology rule: adjudicating it requires '
            're-reading the source episode body and comparing meaning. That '
            'is the fact-CONTENT family esc-4639-1 separates from this one.'
        ),
    },
)
"""The two sub-classes this Class-A detector provably does NOT cover.

Shipped as a first-class report key, not as a comment, because the artifact
travels: a reader who finds 192 findings and no statement of scope will read
"the sweep found no reversals" as "the corpus holds no reversals". The
measured refutation of the cheap direction heuristic is recorded here so
nobody re-proposes it from first principles — it looks obviously correct and
is not.
"""

CAVEATS: tuple[str, ...] = (
    'CANDIDATES, NOT VERDICTS: every finding is a candidate for HUMAN '
    'adjudication. Some are legitimate cross-task relations that merely hang '
    'off a task-shaped node — e.g. "BOOKMARK task 4043 tracks the ... work '
    'surfaced by esc-3437-13" on node "task 3443". Auto-reassigning on a '
    'regex verdict is exactly what this task\'s scope note forbids, and this '
    'script has no path to do it.',
    'DETECTION BOUND, RECALL: endpoints and facts are read with the SHARED '
    'vocabulary in fused_memory.utils.canonical_labels, which is documented '
    'precision-over-recall. A node named with bare digits, a reference made '
    'by task TITLE, an alias/codename, and a hard-wrapped qualified ref are '
    'all invisible by design. The rate reported here is therefore a LOWER '
    'BOUND on Class A.',
    'DENOMINATOR: "rate" divides by "population" — edges with at least one '
    'task-shaped endpoint AND at least one task id named in the fact. Edges '
    'whose fact names no task id are UNVERIFIABLE, not clean, and are '
    'counted separately; they are never folded into the denominator and '
    'never summed with the findings.',
    'COVERAGE, CLASS: this sweep covers ONE sub-class — an endpoint the fact '
    'does not name. See known_gaps for the two it provably does not cover.',
    'LIVE CORPUS: both graphs are written to continuously, so every count '
    'here is a snapshot at swept_at, not a fixed population.',
)
"""The report's caveats, as DATA rather than docstring prose — they travel
with the committed artifact into investigation.md and into whatever reads it
next. Deliberately not pinned word for word by a test: that would be
tautological, and substring-matching prose pins cosmetics."""


def _tally(findings: list[Finding], key, seed: tuple[str, ...]) -> dict[str, int]:
    """Count *findings* by *key*, with every *seed* value present at 0.

    The seed is the no-silent-absence half: a bucket that appeared only when
    non-zero would read as NOT MEASURED rather than as measured-and-none.
    """
    counts = dict.fromkeys(seed, 0)
    for finding in findings:
        k = key(finding)
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items()))


def build_report(
    findings: list[Finding],
    *,
    swept_at: str,
    graphs: list[str],
    scanned: int,
    population: int,
    unverifiable: int = 0,
    reads: object = (),
    limit_listing: int | None = None,
) -> dict[str, Any]:
    """Build the machine-readable report.

    THE DENOMINATOR IS ``population``, NOT ``scanned``. An edge whose fact
    names no task id offers nothing to compare an endpoint against, so it is
    UNVERIFIABLE — a third state, neither finding nor clean. Dividing by
    ``scanned`` would fold that whole population into the denominator and
    understate the rate by roughly the ratio of the two. ``unverifiable`` is
    reported alongside and is never summed with ``findings``.

    NO SILENT CAPS, in both directions. An incomplete :class:`PagedRead`
    publishes its own ``reason`` VERBATIM into ``truncated_by`` and
    ``caveats`` — re-wording it would hide the store's own account of what
    went wrong, and every denominator here depends on the read being
    complete, so an incomplete one does not footnote the numbers, it
    invalidates them. A bounded listing likewise names what it withheld while
    still COUNTING it.

    ``families`` (nodes carrying more than one finding) is a first-class
    summary key because it is the strongest single signal in the artifact:
    five findings on one node is a systematic mis-resolution event, and five
    findings on five nodes is not, and a reader must not have to recompute
    that distinction by hand.

    Pure: *swept_at* is an argument rather than a ``datetime.now()`` call,
    which is what makes the output byte-stable across two runs on one input.

    Args:
        reads: ``(graph, kind, PagedRead)`` triples — one per read performed.
    """
    findings = sorted(
        findings,
        key=lambda f: (f.graph, _id_sort_key(f.node_referent.number), f.edge_uuid),
    )

    incomplete = [
        {
            'graph': graph,
            'kind': kind,
            'rows_seen': read.rows_seen,
            'expected_rows': read.expected_rows,
            'incomplete_kind': read.incomplete_kind,
            'reason': read.reason,
        }
        for graph, kind, read in reads  # type: ignore[misc]
        if not read.complete
    ]

    listed = findings if limit_listing is None else findings[:limit_listing]
    withheld = len(findings) - len(listed)

    truncated_by: dict[str, Any] | None = None
    if incomplete or withheld:
        truncated_by = {}
        if incomplete:
            truncated_by['incomplete_reads'] = incomplete
        if withheld:
            truncated_by['listing'] = {
                'withheld': withheld,
                'flag': '--limit-listing',
                'note': (
                    f'{withheld} finding(s) were COUNTED in summary.findings '
                    f'but not listed. Re-run without --limit-listing to see '
                    f'them.'
                ),
            }

    # A node carrying MORE THAN ONE finding — the Task 6165 shape.
    by_node: dict[tuple[str, str], list[str]] = {}
    for finding in findings:
        by_node.setdefault((finding.graph, finding.node_name), []).append(
            finding.edge_uuid
        )
    families = [
        {'graph': graph, 'node_name': name, 'findings': len(uuids),
         'edge_uuids': sorted(uuids)}
        for (graph, name), uuids in sorted(by_node.items())
        if len(uuids) > 1
    ]

    return {
        'swept_at': swept_at,
        'graphs': list(graphs),
        'scanned': scanned,
        'population': population,
        'unverifiable': unverifiable,
        'summary': {
            'findings': len(findings),
            'rate': (len(findings) / population) if population else 0.0,
            'by_graph': _tally(findings, lambda f: f.graph, tuple(graphs)),
            'by_end': _tally(findings, lambda f: f.end, ('subject', 'object')),
            'by_proximity': _tally(
                findings, lambda f: f.proximity or 'unrelated', PROXIMITY_BUCKETS
            ),
            'correct_node_present': _tally(
                findings,
                lambda f: 'true' if f.correct_node_present else 'false',
                ('true', 'false'),
            ),
            'families': families,
        },
        'truncated_by': truncated_by,
        # The store's own account of any incomplete read, verbatim, so the
        # caveats a reader skims already carry it.
        'caveats': list(CAVEATS) + [e['reason'] for e in incomplete if e['reason']],
        'known_gaps': [dict(gap) for gap in KNOWN_GAPS],
        'findings': [f.to_json() for f in listed],
    }


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #

logger = logging.getLogger('audit_wrong_binding_edges')


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Factored out precisely so ``TestReadOnlyByConstruction`` can enumerate
    ``parser._actions`` and assert that NO mutation affordance exists. There
    is deliberately no ``--apply``, no ``--invalidate``, no ``--delete``, no
    ``--repair`` and no ``--reassign``: this script reports, and a human
    adjudicates.
    """
    parser = argparse.ArgumentParser(
        prog='audit_wrong_binding_edges',
        description=(
            'Read-only retrospective sweep for wrong-binding RELATES_TO edges '
            '(task 4717, esc-4639-1). Reports only — it never mutates the graph.'
        ),
    )
    parser.add_argument(
        '--graph', action='append', metavar='GRAPH',
        help=(
            'Graph to sweep. Repeatable — the artifact run sweeps both '
            f'dark_factory and reify. Default: {DEFAULT_GRAPH}.'
        ),
    )
    parser.add_argument(
        '--graph-uri', default=None,
        help='FalkorDB URI (redis://host:port). Default: env, then config.',
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Emit the full JSON report on stdout instead of a short summary.',
    )
    parser.add_argument(
        '--out-dir', default=None, metavar='DIR',
        help='Also write DIR/report.json (byte-identical to the --json output).',
    )
    parser.add_argument(
        '--include-unverifiable', action='store_true',
        help=(
            'List the edges whose fact names NO task id. They are always '
            'COUNTED under the top-level "unverifiable" key; this adds the '
            'listing.'
        ),
    )
    parser.add_argument(
        '--limit-listing', type=int, default=None, metavar='N',
        help=(
            'List at most N findings. All are still COUNTED, and '
            'truncated_by names what was withheld.'
        ),
    )
    parser.add_argument(
        '--fail-on-finding', action='store_true',
        help='Exit 2 when at least one finding is found (for a CI/cron gate).',
    )
    return parser


def _resolve_uri(args: argparse.Namespace) -> str | None:
    """Resolve the FalkorDB URI from --graph-uri, then the env, then config."""
    if args.graph_uri:
        return str(args.graph_uri)
    env_uri = os.environ.get('FALKORDB_URI')
    if env_uri:
        return env_uri
    try:
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

        config = FusedMemoryConfig()
        return getattr(getattr(config.graphiti, 'falkordb', None), 'uri', None)
    except Exception:
        logger.warning(
            'could not resolve a FalkorDB uri from config; falling back to '
            'the client default', exc_info=True,
        )
        return None


async def _sweep_graph(
    reader: Any, graph: str
) -> tuple[list[Finding], int, int, int, list[dict[str, Any]], list[tuple]]:
    """Sweep one graph. Returns findings, scanned, population, unverifiable,
    unverifiable rows, and the (graph, kind, PagedRead) triples."""
    node_ids, node_read = await reader.read_task_node_ids()
    rows, edge_read = await reader.fetch_edges()
    logger.info(
        '%s: %d live RELATES_TO row(s) (complete=%s), %d task node(s)',
        graph, edge_read.rows_seen, edge_read.complete, len(node_ids),
    )

    findings: list[Finding] = []
    population = 0
    unverifiable = 0
    unverifiable_rows: list[dict[str, Any]] = []
    for row in rows:
        subject, obj, uuid, fact, episodes = (list(row) + [None] * 5)[:5]
        # An edge with no task-shaped endpoint asks no question this sweep can
        # answer, so it is outside BOTH the population and the unverifiable
        # count — it is simply not about tasks.
        if endpoint_referent(subject) is None and endpoint_referent(obj) is None:
            continue
        if not fact_referents(fact, graph):
            unverifiable += 1
            unverifiable_rows.append(
                {'edge_uuid': uuid, 'graph': graph, 'subject': subject,
                 'object': obj, 'fact': fact}
            )
            continue
        population += 1
        findings.extend(
            classify_edge(subject, obj, fact, uuid, graph, episodes=episodes)
        )

    # Cause attribution, once the graph's whole task-node census is in hand.
    enriched: list[Finding] = []
    for finding in findings:
        named = {r.number for r in finding.fact_referents if not r.project_id}
        bucket, nearest = id_proximity(finding.node_referent.number, named)
        enriched.append(
            Finding(**{
                **vars_of(finding),
                'proximity': bucket,
                'nearest_id': nearest,
                'correct_node_present': correct_node_present(nearest, node_ids),
            })
        )
    reads = [(graph, 'nodes', node_read), (graph, 'edges', edge_read)]
    return enriched, len(rows), population, unverifiable, unverifiable_rows, reads


async def _run(
    args: argparse.Namespace, *, reader_factory: object | None = None
) -> int:
    """Sweep every requested graph and emit one report.

    Exit codes: ``0`` ran, ``1`` infra failure (NOTHING is emitted — a
    truncated report that looks complete is worse than none), ``2``
    ``--fail-on-finding`` and at least one finding was found.
    """
    if reader_factory is None:
        logging.basicConfig(
            level=logging.INFO, format='%(levelname)s %(name)s: %(message)s',
            stream=sys.stderr,
        )
    graphs = list(args.graph) if args.graph else [DEFAULT_GRAPH]

    try:
        if reader_factory is None:
            uri = _resolve_uri(args)

            def make_reader(name: str) -> Any:
                return EdgeReader(graph_name=name, uri=uri)
        else:
            make_reader = reader_factory  # type: ignore[assignment]

        findings: list[Finding] = []
        scanned = population = unverifiable = 0
        unverifiable_rows: list[dict[str, Any]] = []
        reads: list[tuple] = []
        for graph in graphs:
            found, n, pop, unv, unv_rows, gr = await _sweep_graph(
                make_reader(graph), graph
            )
            findings.extend(found)
            scanned += n
            population += pop
            unverifiable += unv
            unverifiable_rows.extend(unv_rows)
            reads.extend(gr)
    except Exception:
        logger.error(
            'could not read the edge population — no report is emitted rather '
            'than a partial one that would read as complete', exc_info=True,
        )
        return 1

    report = build_report(
        findings,
        swept_at=datetime.now(UTC).isoformat(),
        graphs=graphs,
        scanned=scanned,
        population=population,
        unverifiable=unverifiable,
        reads=reads,
        limit_listing=args.limit_listing,
    )
    if args.include_unverifiable:
        report['unverifiable_edges'] = unverifiable_rows

    blob = json.dumps(report, indent=2, sort_keys=False, default=str)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'report.json').write_text(blob + '\n')
        logger.info('wrote %s', out / 'report.json')

    if args.json:
        print(blob)
    else:
        summary = report['summary']
        print(
            f"scanned={report['scanned']} population={report['population']} "
            f"unverifiable={report['unverifiable']} "
            f"findings={summary['findings']} rate={summary['rate']:.4f} "
            f"truncated_by={'yes' if report['truncated_by'] else 'no'}"
        )

    if args.fail_on_finding and report['summary']['findings']:
        return 2
    return 0


def main() -> int:
    """Entry point: parse argv and run the sweep."""
    return asyncio.run(_run(_build_parser().parse_args()))


if __name__ == '__main__':
    sys.exit(main())
