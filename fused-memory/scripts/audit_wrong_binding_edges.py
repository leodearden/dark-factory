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

import re
from dataclasses import dataclass
from typing import Any

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
