"""Tests for docstring accuracy of get_all_valid_edges in graphiti_client.py.

Task 446: Fix misleading docstring — DISTINCT scope and double-attribution semantics.

Three test/impl pairs:
  step-1/2: DISTINCT rationale correction (self-loop-only scope, remove misleading claims)
  step-3/4: double-attribution Returns note (each directed edge appears under both endpoints)
  step-5/6: directed MATCH alternative note (-[e:RELATES_TO]-> gives single-appearance)
"""
from __future__ import annotations

import re

from fused_memory.backends.graphiti_client import GraphitiBackend


# ---------------------------------------------------------------------------
# step-1: DISTINCT rationale must be corrected
# ---------------------------------------------------------------------------


class TestGetAllValidEdgesDistinctRationale:
    """get_all_valid_edges docstring must accurately describe the DISTINCT scope.

    Asserts:
      (a) docstring is not None and non-empty
      (b) misleading phrase 'can produce duplicate rows' is absent
      (c) misleading cross-reference 'matching the pattern in get_valid_edges_for_node' is absent
      (d) corrected rationale mentions 'self-loop' as the case where DISTINCT matters
    """

    def _doc(self) -> str:
        doc = GraphitiBackend.get_all_valid_edges.__doc__
        assert doc is not None, "get_all_valid_edges must have a docstring"
        assert doc.strip(), "get_all_valid_edges docstring must not be blank"
        return doc

    def test_docstring_not_none(self) -> None:
        """Docstring exists and is non-empty."""
        self._doc()

    def test_misleading_duplicate_rows_phrase_absent(self) -> None:
        """Old misleading phrase 'can produce duplicate rows' must be removed."""
        doc = self._doc()
        assert 'can produce duplicate rows' not in doc, (
            "Misleading phrase 'can produce duplicate rows' must be removed; "
            "DISTINCT does not prevent directed A→B duplication (n.uuid differs per row)"
        )

    def test_misleading_cross_reference_absent(self) -> None:
        """Misleading cross-reference to get_valid_edges_for_node must be removed."""
        doc = self._doc()
        assert 'matching the pattern in get_valid_edges_for_node' not in doc, (
            "Misleading cross-reference must be removed; get_valid_edges_for_node returns "
            "DISTINCT e.uuid (no n.uuid), so DISTINCT has different effective scope there"
        )

    def test_self_loop_rationale_present(self) -> None:
        """Corrected DISTINCT rationale must mention 'self-loop' as the relevant case."""
        doc = self._doc()
        assert 'self-loop' in doc, (
            "Docstring must mention 'self-loop' to explain when DISTINCT actually matters: "
            "only for A→A edges where both traversal directions produce identical rows"
        )


# ---------------------------------------------------------------------------
# step-3: Returns section must document double-attribution semantics
# ---------------------------------------------------------------------------


class TestGetAllValidEdgesReturnsDoubleAttribution:
    """get_all_valid_edges docstring Returns section must document double-attribution.

    Because the undirected MATCH pattern matches each directed edge from both
    endpoints, a directed A→B edge appears in the result under both A's UUID and
    B's UUID.  The Returns section must communicate this intentional behavior.

    Asserts:
      (a) Returns section mentions 'both' source and target (or equivalent phrasing)
      (b) Returns section mentions 'appears' in context of the double-attribution
          OR uses phrasing like 'twice' / 'each endpoint' to describe the behavior
    """

    def _doc(self) -> str:
        doc = GraphitiBackend.get_all_valid_edges.__doc__
        assert doc is not None
        return doc

    def test_returns_mentions_both_endpoints(self) -> None:
        """Returns section must note each directed edge appears under both endpoints."""
        doc = self._doc()
        # Accept any phrasing that communicates the double-attribution:
        # 'both its source and target', 'both endpoints', 'both the source', etc.
        has_both = (
            'both' in doc
            or re.search(r'appear[s]?\s+\w+\s+(source|target|endpoint)', doc) is not None
            or 'twice' in doc
        )
        assert has_both, (
            "Returns section must document double-attribution: each directed edge "
            "appears under both its source and target entity UUID"
        )

    def test_returns_section_double_attribution_in_returns_block(self) -> None:
        """Double-attribution note must appear in or near the Returns section."""
        doc = self._doc()
        # Find the Returns section
        returns_idx = doc.find('Returns:')
        assert returns_idx != -1, "Docstring must have a Returns: section"
        returns_text = doc[returns_idx:]
        # The double-attribution note should be in the Returns block or nearby
        has_note = (
            'both' in returns_text
            or 'twice' in returns_text
            or 'double' in returns_text
            or 'each endpoint' in returns_text
            or 'source and target' in returns_text
        )
        assert has_note, (
            "Double-attribution note must appear in the Returns section; "
            "callers need to know each directed edge appears under both endpoints"
        )


# ---------------------------------------------------------------------------
# step-5: docstring must note the directed MATCH alternative
# ---------------------------------------------------------------------------


class TestGetAllValidEdgesDirectedMatchNote:
    """get_all_valid_edges docstring must mention the directed MATCH alternative.

    A Note section should inform future maintainers that using
    (n:Entity)-[e:RELATES_TO]->() (directed arrow) would give single-appearance
    semantics per edge if ever needed.

    Asserts:
      (a) docstring contains the directed arrow syntax '-[e:RELATES_TO]->'
      (b) docstring mentions 'single-appearance' or 'single appearance' as the
          alternative semantics
    """

    def _doc(self) -> str:
        doc = GraphitiBackend.get_all_valid_edges.__doc__
        assert doc is not None
        return doc

    def test_directed_arrow_syntax_present(self) -> None:
        """Docstring must mention the directed MATCH pattern with arrow syntax."""
        doc = self._doc()
        assert '-[e:RELATES_TO]->' in doc, (
            "Docstring must include '-[e:RELATES_TO]->' to document the directed "
            "MATCH alternative that gives single-appearance semantics"
        )

    def test_single_appearance_semantics_mentioned(self) -> None:
        """Docstring must describe the directed alternative as single-appearance."""
        doc = self._doc()
        has_single = (
            'single-appearance' in doc
            or 'single appearance' in doc
        )
        assert has_single, (
            "Docstring must mention 'single-appearance' or 'single appearance' "
            "to describe the semantics of the directed MATCH alternative"
        )
