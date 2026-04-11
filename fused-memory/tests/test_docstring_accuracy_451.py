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


def _keywords_in_proximity(doc: str, kw1: str, kw2: str, max_distance: int = 200) -> bool:
    """Return True iff kw1 and kw2 appear within max_distance chars of each other in doc.

    Uses re.finditer to locate all occurrences of each keyword; returns True iff
    the minimum pairwise character distance is strictly less than max_distance.
    Immune to sentence-tokenisation artefacts like 'e.g.' or 'i.e.'.
    """
    pos1 = [m.start() for m in re.finditer(re.escape(kw1), doc)]
    pos2 = [m.start() for m in re.finditer(re.escape(kw2), doc)]
    if not pos1 or not pos2:
        return False
    return min(abs(p1 - p2) for p1 in pos1 for p2 in pos2) < max_distance


def _returns_section_text(doc: str) -> str | None:
    """Return the text of the Returns: section, bounded by the next section header.

    Finds 'Returns:' in doc, then clips the slice at the next Google/numpy-style
    section header (blank line + capitalised identifier + colon, e.g. 'Raises:',
    'Note:', 'Example:', 'Args:').  Returns None if no 'Returns:' header exists.
    """
    returns_idx = doc.find('Returns:')
    if returns_idx == -1:
        return None
    after_returns = doc[returns_idx + len('Returns:'):]
    match = re.search(r'\n\s*\n\s*[A-Z]\w+:', after_returns)
    if match:
        return doc[returns_idx:returns_idx + len('Returns:') + match.start()]
    return doc[returns_idx:]


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
# Helper: _returns_section_text
# ---------------------------------------------------------------------------


class TestReturnsSectionTextHelper:
    """Unit tests for the _returns_section_text module-level helper.

    Verifies that the helper correctly bounds the Returns section to avoid the
    scope-leak where doc[returns_idx:] bleeds into subsequent Raises:/Note:/Example:
    sections.

    Asserts:
      (a) returns the full Returns section when no following section header exists
      (b) returns bounded section when followed by Raises: (scope-leak regression case)
      (c) returns bounded section when followed by Note:
      (d) returns bounded section when followed by Example:
      (e) returns None when no Returns: header exists
      (f) regression baseline: returns non-empty string for detect_stale_summaries docstring
    """

    def test_returns_only_section(self) -> None:
        """Returns the full Returns section when no following section header exists."""
        doc = 'Docstring intro.\n\nReturns:\n    foo: The foo value.\n'
        result = _returns_section_text(doc)
        assert result is not None
        assert 'foo' in result

    def test_bounded_by_raises(self) -> None:
        """Returns section is bounded before a following Raises: section.

        This is the scope-leak regression case: the old doc[returns_idx:] approach
        would include 'ValueError' from the Raises section, masking a regression
        where the key was removed from Returns.
        """
        doc = 'Intro.\n\nReturns:\n    foo: bar\n\nRaises:\n    ValueError: something.\n'
        result = _returns_section_text(doc)
        assert result is not None
        assert 'foo' in result
        assert 'ValueError' not in result

    def test_bounded_by_note(self) -> None:
        """Returns section is bounded before a following Note: section."""
        doc = 'Intro.\n\nReturns:\n    foo: bar\n\nNote:\n    Some note.\n'
        result = _returns_section_text(doc)
        assert result is not None
        assert 'foo' in result
        assert 'Some note' not in result

    def test_bounded_by_example(self) -> None:
        """Returns section is bounded before a following Example: section."""
        doc = 'Intro.\n\nReturns:\n    foo: bar\n\nExample:\n    example_code()\n'
        result = _returns_section_text(doc)
        assert result is not None
        assert 'foo' in result
        assert 'example_code' not in result

    def test_none_when_no_returns_section(self) -> None:
        """Returns None when the docstring has no Returns: header."""
        doc = 'Intro.\n\nArgs:\n    x: some arg\n'
        result = _returns_section_text(doc)
        assert result is None

    def test_regression_baseline_detect_stale_summaries(self) -> None:
        """Regression baseline: returns non-empty string for detect_stale_summaries docstring."""
        doc = GraphitiBackend.detect_stale_summaries.__doc__
        assert doc is not None, 'detect_stale_summaries must have a docstring'
        result = _returns_section_text(doc)
        assert result is not None, "detect_stale_summaries docstring must have a Returns: section"
        assert result.strip() != '', 'Returns section must not be blank'


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
        returns_text = _returns_section_text(self._doc())
        assert returns_text is not None, "Docstring must have a 'Returns:' section"
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
        returns_text = _returns_section_text(self._doc())
        assert returns_text is not None, "Docstring must have a 'Returns:' section"
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
        """'single-entity' and 'refresh_entity_summary' must appear within 200 characters."""
        doc = self._doc()
        assert _keywords_in_proximity(doc, 'single-entity', 'refresh_entity_summary'), (
            "Docstring must contain 'single-entity' and 'refresh_entity_summary' within "
            "200 characters of each other to provide a use-case routing note (not just the "
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
        """'bulk' and '_rebuild_entity_from_edges' must appear within 200 characters."""
        doc = self._doc()
        assert _keywords_in_proximity(doc, 'bulk', '_rebuild_entity_from_edges'), (
            "Docstring must contain 'bulk' and '_rebuild_entity_from_edges' within "
            "200 characters of each other to provide a use-case routing note for "
            "callers rebuilding many entities at once"
        )
