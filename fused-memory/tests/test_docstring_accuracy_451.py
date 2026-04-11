"""Tests for docstring accuracy in graphiti_client.py.

Task 451: Fix docstrings — detect_stale_summaries missing 'summary' key +
cross-reference _rebuild_entity_from_edges vs refresh_entity_summary.

Three test/impl pairs:
  step-1/2: detect_stale_summaries Returns section must list 'summary' key
  step-3/4: _rebuild_entity_from_edges docstring must routing-note refresh_entity_summary
  step-5/6: refresh_entity_summary docstring must routing-note _rebuild_entity_from_edges
"""
from __future__ import annotations

import re

from fused_memory.backends.graphiti_client import GraphitiBackend

# ---------------------------------------------------------------------------
# Helper: _keywords_in_proximity
# ---------------------------------------------------------------------------


class TestKeywordsInProximityHelper:
    """Unit tests for the _keywords_in_proximity module-level helper.

    Verifies proximity-based keyword co-occurrence detection, specifically
    targeting the 'e.g.' splitter regression case that motivated the helper.

    Asserts:
      (a) returns True when keywords are adjacent
      (b) returns True within 200 chars across an 'e.g.' boundary (regression case)
      (c) returns False when keywords are > 200 chars apart
      (d) returns False when one keyword is absent
      (e) regression baseline against actual refresh_entity_summary docstring
    """

    def test_adjacent_keywords_return_true(self) -> None:
        """Keywords directly adjacent are within any positive distance."""
        assert _keywords_in_proximity('foo bar', 'foo', 'bar') is True

    def test_within_200_across_eg_boundary(self) -> None:
        """Keywords within 200 chars around an 'e.g.' boundary are found.

        This is the regression case: re.split(r'\\.\\s+|\\.\\$') splits
        'e.g. _rebuild_entity_from_edges' at the period, destroying co-occurrence.
        The proximity approach is immune because it never tokenises into sentences.
        """
        doc = 'For bulk use (e.g. _rebuild_entity_from_edges) supply edges'
        assert _keywords_in_proximity(doc, 'bulk', '_rebuild_entity_from_edges') is True

    def test_far_keywords_return_false(self) -> None:
        """Keywords separated by more than 200 chars return False."""
        doc = 'bulk ' + 'x' * 300 + ' _rebuild_entity_from_edges'
        assert _keywords_in_proximity(doc, 'bulk', '_rebuild_entity_from_edges') is False

    def test_missing_keyword_returns_false(self) -> None:
        """Returns False when one keyword is not present in the doc."""
        assert _keywords_in_proximity('foo bar baz', 'foo', 'qux') is False

    def test_regression_baseline_refresh_docstring(self) -> None:
        """Regression baseline: 'bulk' and '_rebuild_entity_from_edges' co-occur
        within 200 chars in the actual refresh_entity_summary docstring."""
        doc = GraphitiBackend.refresh_entity_summary.__doc__
        assert doc is not None, 'refresh_entity_summary must have a docstring'
        assert _keywords_in_proximity(doc, 'bulk', '_rebuild_entity_from_edges'), (
            "'bulk' and '_rebuild_entity_from_edges' must be within 200 chars in "
            "refresh_entity_summary docstring"
        )


# ---------------------------------------------------------------------------
# step-1: detect_stale_summaries Returns section must include 'summary' key
# ---------------------------------------------------------------------------


class TestDetectStaleSummariesReturnsIncludesSummaryKey:
    """detect_stale_summaries docstring Returns section must list 'summary'.

    The underlying _detect_stale_summaries_with_edges builds stale dicts that
    include a 'summary' key (the current entity summary text).  The public API
    docstring omitted it, leaving callers unaware they can diff summary text
    without a second DB query.  The sibling _detect_stale_summaries_dry_run
    already lists 'summary' correctly.

    Asserts:
      (a) docstring is not None and non-empty
      (b) Returns section lists 'summary' key alongside the six existing keys
      (c) all six pre-existing keys remain documented in the Returns section
    """

    def _doc(self) -> str:
        doc = GraphitiBackend.detect_stale_summaries.__doc__
        assert doc is not None, 'detect_stale_summaries must have a docstring'
        assert doc.strip(), 'detect_stale_summaries docstring must not be blank'
        return doc

    def test_docstring_not_none(self) -> None:
        """Docstring exists and is non-empty."""
        self._doc()

    def test_returns_section_lists_summary_key(self) -> None:
        """Returns section must include 'summary' as a standalone listed key.

        Uses a word-boundary check to avoid false positives from 'summary_line_count'
        which also contains 'summary' as a prefix.
        """
        doc = self._doc()
        returns_idx = doc.find('Returns:')
        assert returns_idx != -1, "Docstring must have a 'Returns:' section"
        returns_text = doc[returns_idx:]
        # Match 'summary' as a standalone key — not as a prefix of 'summary_line_count'
        # The negative lookahead (?!_) rejects 'summary_line_count', 'summary_lines', etc.
        has_standalone_summary = re.search(r'\bsummary\b(?!_)', returns_text) is not None
        assert has_standalone_summary, (
            "Returns section must list 'summary' as a standalone key (not just as part of "
            "'summary_line_count'); each stale entity dict includes the current summary text "
            "so callers can diff it against the canonical fact set without a second DB query"
        )

    def test_returns_section_retains_existing_keys(self) -> None:
        """All six pre-existing keys must remain in the Returns section."""
        doc = self._doc()
        returns_idx = doc.find('Returns:')
        assert returns_idx != -1, "Docstring must have a 'Returns:' section"
        returns_text = doc[returns_idx:]
        for key in ('uuid', 'name', 'duplicate_count', 'stale_line_count',
                    'valid_fact_count', 'summary_line_count'):
            assert key in returns_text, (
                f"Returns section must retain pre-existing key '{key}'"
            )


# ---------------------------------------------------------------------------
# step-3: _rebuild_entity_from_edges must routing-note refresh_entity_summary
# ---------------------------------------------------------------------------


class TestRebuildEntityFromEdgesCrossReferencesRefresh:
    """_rebuild_entity_from_edges docstring must have a routing note for single-entity use.

    The two methods are an intentional fork.  The existing TOCTOU note already
    mentions refresh_entity_summary for consistency reasons; this class checks
    for a USE-CASE routing note — a sentence that pairs 'single-entity' with
    'refresh_entity_summary' so maintainers know where to go for non-bulk use.

    Asserts:
      (a) docstring is not None and non-empty
      (b) 'single-entity' appears in the docstring
      (c) 'single-entity' and 'refresh_entity_summary' appear in the same sentence
    """

    def _doc(self) -> str:
        doc = GraphitiBackend._rebuild_entity_from_edges.__doc__
        assert doc is not None, '_rebuild_entity_from_edges must have a docstring'
        assert doc.strip(), '_rebuild_entity_from_edges docstring must not be blank'
        return doc

    def test_docstring_not_none(self) -> None:
        """Docstring exists and is non-empty."""
        self._doc()

    def test_single_entity_phrase_present(self) -> None:
        """Docstring must contain 'single-entity' to signal non-bulk use routing."""
        doc = self._doc()
        assert 'single-entity' in doc, (
            "Docstring must mention 'single-entity' to route non-bulk callers to "
            "refresh_entity_summary"
        )

    def test_single_entity_and_refresh_in_same_sentence(self) -> None:
        """'single-entity' and 'refresh_entity_summary' must appear in the same sentence."""
        doc = self._doc()
        # Split on sentence boundaries (period followed by whitespace or end of string)
        sentences = re.split(r'\.\s+|\.$', doc)
        found = any(
            'single-entity' in sentence and 'refresh_entity_summary' in sentence
            for sentence in sentences
        )
        assert found, (
            "A sentence in the docstring must contain both 'single-entity' and "
            "'refresh_entity_summary' to provide a use-case routing note (not just the "
            "existing TOCTOU consistency note)"
        )


# ---------------------------------------------------------------------------
# step-5: refresh_entity_summary must routing-note _rebuild_entity_from_edges
# ---------------------------------------------------------------------------


class TestRefreshEntitySummaryCrossReferencesRebuild:
    """refresh_entity_summary docstring must have a routing note for bulk use.

    The method already mentions _rebuild_entity_from_edges once (in the name
    parameter doc as an example caller that supplies name/old_summary).  That
    existing mention is not in a sentence containing 'bulk', so this class
    specifically tests for a USE-CASE routing note that pairs 'bulk' with
    '_rebuild_entity_from_edges'.

    Asserts:
      (a) docstring is not None and non-empty
      (b) '_rebuild_entity_from_edges' appears in the docstring
      (c) 'bulk' and '_rebuild_entity_from_edges' appear in the same sentence
    """

    def _doc(self) -> str:
        doc = GraphitiBackend.refresh_entity_summary.__doc__
        assert doc is not None, 'refresh_entity_summary must have a docstring'
        assert doc.strip(), 'refresh_entity_summary docstring must not be blank'
        return doc

    def test_docstring_not_none(self) -> None:
        """Docstring exists and is non-empty."""
        self._doc()

    def test_rebuild_reference_present(self) -> None:
        """Docstring must reference '_rebuild_entity_from_edges'."""
        doc = self._doc()
        assert '_rebuild_entity_from_edges' in doc, (
            "Docstring must mention '_rebuild_entity_from_edges' to cross-reference "
            "the bulk-optimised counterpart"
        )

    def test_bulk_and_rebuild_in_same_sentence(self) -> None:
        """'bulk' and '_rebuild_entity_from_edges' must appear in the same sentence."""
        doc = self._doc()
        sentences = re.split(r'\.\s+|\.$', doc)
        found = any(
            'bulk' in sentence and '_rebuild_entity_from_edges' in sentence
            for sentence in sentences
        )
        assert found, (
            "A sentence in the docstring must contain both 'bulk' and "
            "'_rebuild_entity_from_edges' to provide a use-case routing note for "
            "callers rebuilding many entities at once"
        )
