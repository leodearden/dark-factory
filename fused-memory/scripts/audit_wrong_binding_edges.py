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

from fused_memory.utils.canonical_labels import Referent, scan_content

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
