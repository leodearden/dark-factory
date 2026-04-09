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
