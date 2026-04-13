"""Test hygiene pins for Task 704.

Task 704: Two documentary hygiene items flagged during post-merge review of
Task 449's TestGetAllValidEdges relocation.

  Item 1 — fix stale module docstring in test_rebuild_entity_summaries.py
    The tracker line for GraphitiBackend.get_all_valid_edges() still uses the
    old arrow-only phrasing ("→ see test_refresh_entity_summary.py"), which
    misleads readers into thinking the method is not exercised here at all.
    The correct phrasing documents both facts: unit tests live in
    test_refresh_entity_summary.py AND get_all_valid_edges is used here as a
    mocked dependency.

  Item 2 — clarify the `or []` rationale in test_none_result_set_returns_empty_dict
    The existing two-line comment in test_refresh_entity_summary.py explains the
    mechanics of the mutation but does NOT tie the test back to the defensive
    `result.result_set or []` guard in GraphitiBackend.get_all_valid_edges().
    A rationale comment naming that guard ensures a future reader who removes it
    will see this test fail with context.

Both meta-tests read target file source via pathlib.Path.read_text() — more
robust than importing the test module (which would perturb pytest collection).
"""
from __future__ import annotations

import pathlib


def _read_sibling(name: str) -> str:
    """Return the raw source text of a sibling test file."""
    return (pathlib.Path(__file__).parent / name).read_text(encoding='utf-8')


# ---------------------------------------------------------------------------
# Item 1 — module docstring in test_rebuild_entity_summaries.py
# ---------------------------------------------------------------------------


class TestModuleDocstring704:
    """Module docstring in test_rebuild_entity_summaries.py must use accurate phrasing.

    After Task 449 moved TestGetAllValidEdges to test_refresh_entity_summary.py,
    the tracker line must document that:
      (a) unit tests live in test_refresh_entity_summary.py, AND
      (b) get_all_valid_edges is still used here as a mocked dependency.

    Asserts:
      (a) the substring 'unit tests in test_refresh_entity_summary.py' is present
      (b) the substring 'used here as a mocked dependency' is present
      (c) the old arrow-only phrasing '→ see test_refresh_entity_summary.py' is NOT present
    """

    def _source(self) -> str:
        return _read_sibling('test_rebuild_entity_summaries.py')

    def test_rebuild_entity_summaries_docstring_marks_get_all_valid_edges_as_mocked_dependency(
        self,
    ) -> None:
        """Module docstring must use the accurate two-part phrasing for get_all_valid_edges.

        The tracker line must convey that unit tests live in
        test_refresh_entity_summary.py AND that get_all_valid_edges is used
        here as a mocked dependency — not merely redirect with '→ see ...'.
        """
        src = self._source()

        assert 'unit tests in test_refresh_entity_summary.py' in src, (
            "test_rebuild_entity_summaries.py module docstring must contain "
            "'unit tests in test_refresh_entity_summary.py' to accurately convey "
            "where the unit tests for get_all_valid_edges live"
        )

        assert 'used here as a mocked dependency' in src, (
            "test_rebuild_entity_summaries.py module docstring must contain "
            "'used here as a mocked dependency' to clarify that get_all_valid_edges "
            "is still exercised (as a mock) in this file"
        )

        assert '\u2192 see test_refresh_entity_summary.py' not in src, (
            "test_rebuild_entity_summaries.py module docstring must NOT contain "
            "the stale arrow-only phrasing '\u2192 see test_refresh_entity_summary.py'; "
            "replace it with the accurate two-part phrasing that notes both the "
            "unit-test location and the mocked-dependency usage"
        )


# ---------------------------------------------------------------------------
# Item 2 — rationale comment in test_none_result_set_returns_empty_dict
# ---------------------------------------------------------------------------


class TestNoneResultSetRationale704:
    """Comment in test_none_result_set_returns_empty_dict must cite the `or []` guard.

    The test exercises get_all_valid_edges()'s defensive
    `for row in (result.result_set or []):` guard by wiring
    `graph.ro_query.return_value.result_set = None`.  The comment immediately
    preceding the mutation must name the `or []` guard and anchor it to
    get_all_valid_edges so a future reader who removes the guard understands
    why this test exists.

    Asserts:
      (a) the substring 'or []' appears in the comment lines within ~5 lines
          before the `result_set = None` assignment inside
          test_none_result_set_returns_empty_dict
      (b) the comment also references 'get_all_valid_edges' by name
    """

    def _source(self) -> str:
        return _read_sibling('test_refresh_entity_summary.py')

    def _lines_before_mutation(self) -> list[str]:
        """Return up to 5 comment/blank lines immediately before `result_set = None`."""
        src = self._source()
        lines = src.splitlines()
        mutation_marker = 'graph.ro_query.return_value.result_set = None'
        # Find the mutation line inside test_none_result_set_returns_empty_dict
        for idx, line in enumerate(lines):
            if mutation_marker in line:
                # Return up to 5 lines immediately preceding the mutation
                start = max(0, idx - 5)
                return lines[start:idx]
        return []

    def test_none_result_set_test_cites_or_guard(self) -> None:
        """Comment before result_set=None mutation must cite the `or []` guard expression.

        The defensive `result.result_set or []` guard in get_all_valid_edges()
        is the production code that this test pins down.  Naming it in a comment
        ensures that if the guard is ever removed, the test failure message
        gives a reader immediate context about what changed.
        """
        preceding = self._lines_before_mutation()
        assert preceding, (
            "Could not find 'graph.ro_query.return_value.result_set = None' in "
            "test_refresh_entity_summary.py — has the mutation line been renamed?"
        )
        combined = '\n'.join(preceding)

        assert 'or []' in combined, (
            "The comment lines immediately before "
            "'graph.ro_query.return_value.result_set = None' must contain the "
            "literal substring 'or []' to cite the defensive guard being tested. "
            f"Found preceding lines:\n{combined}"
        )

        assert 'get_all_valid_edges' in combined, (
            "The comment lines immediately before "
            "'graph.ro_query.return_value.result_set = None' must reference "
            "'get_all_valid_edges' by name to anchor the rationale to the "
            f"production call site. Found preceding lines:\n{combined}"
        )
