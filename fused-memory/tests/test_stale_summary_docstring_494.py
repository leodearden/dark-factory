"""Tests for StaleSummaryResult class docstring alignment with named-access convention.

Task 494: Align StaleSummaryResult class docstring with named-access convention.

The StaleSummaryResult NamedTuple docstring must:
  (a) exist and be non-empty
  (b) document the named-access pattern by referencing .stale, .all_edges, .total_count
  (c) NOT contain the old positional-unpacking example line
      ``stale, all_edges, total = await``

Task 438/465 already migrated every real call site to named access; the docstring
must reflect that convention rather than advertising the old positional pattern.
"""
from __future__ import annotations

from fused_memory.backends.graphiti_client import StaleSummaryResult


class TestStaleSummaryResultDocstring:
    """StaleSummaryResult class docstring must document named-access and drop unpacking example."""

    def _doc(self) -> str:
        doc = StaleSummaryResult.__doc__
        assert doc is not None, 'StaleSummaryResult must have a class docstring'
        assert doc.strip(), 'StaleSummaryResult class docstring must not be blank'
        return doc

    def test_docstring_exists(self) -> None:
        """Docstring exists and is non-empty."""
        self._doc()

    def test_docstring_documents_named_access_fields(self) -> None:
        """Docstring must reference .stale, .all_edges, and .total_count by name.

        The named-access pattern (result.stale, result.all_edges, result.total_count)
        is the canonical idiom after Task 438/465 migrated every call site away from
        positional unpacking.  The docstring must guide future readers to that pattern.
        """
        doc = self._doc()
        for field in ('.stale', '.all_edges', '.total_count'):
            assert field in doc, (
                f"StaleSummaryResult docstring must reference the named field '{field}' "
                f"so readers see the canonical access pattern (result{field}) rather than "
                f"positional unpacking"
            )

    def test_docstring_does_not_promote_positional_unpacking_example(self) -> None:
        """Docstring must NOT contain the old positional-unpacking example line.

        The literal substring 'stale, all_edges, total = await' is the specific
        example the task targets for removal.  Its presence actively advertises a
        pattern the project moved away from in Task 438/465, creating design tension.

        Note: a brief prose note that positional unpacking *still works* (NamedTuple
        guarantee) is permitted — this assertion only blocks the concrete example line.
        """
        doc = self._doc()
        assert 'stale, all_edges, total = await' not in doc, (
            "StaleSummaryResult docstring must not contain the positional-unpacking example "
            "'stale, all_edges, total = await ...'; that pattern was deprecated in Task 438/465 "
            "and every real call site now uses named attribute access"
        )
