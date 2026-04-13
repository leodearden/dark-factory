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

import ast
import pathlib


def _read_sibling(name: str) -> str:
    """Return the raw source text of a sibling test file."""
    return (pathlib.Path(__file__).parent / name).read_text(encoding='utf-8')


def _extract_module_docstring(src: str) -> str:
    """Return the module-level docstring of the given source, or '' if absent.

    Scopes substring checks to the docstring alone — excludes comment text,
    string literals in function bodies, and any other non-docstring content.
    Use this instead of raw-source substring search when the intent is to pin
    wording that belongs in the module docstring specifically.
    """
    return ast.get_docstring(ast.parse(src)) or ''


def test_extract_module_docstring_returns_only_the_docstring() -> None:
    """_extract_module_docstring must return only the module docstring, not comments.

    Constructs a synthetic source string where 'in_docstring_marker' appears
    in the module docstring and 'in_comment_marker' appears in a line comment
    (but NOT in the docstring).  Proves that the helper scopes substring checks
    to the docstring alone, excluding comment text and other string literals.
    """
    synthetic_src = (
        '"""Module docstring containing in_docstring_marker here."""\n'
        '# in_comment_marker appears only in this comment, not the docstring\n'
        'x = 1\n'
    )
    result = _extract_module_docstring(synthetic_src)
    assert 'in_docstring_marker' in result, (
        "_extract_module_docstring must return the module docstring text, "
        "which contains 'in_docstring_marker'"
    )
    assert 'in_comment_marker' not in result, (
        "_extract_module_docstring must NOT include comment text — "
        "'in_comment_marker' appears only in a comment, not in the docstring"
    )


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

    NOTE: These assertions match literal substrings. If the pinned wording is ever
    rephrased, update the assertions in this class to match the new text.
    """

    def test_rebuild_entity_summaries_docstring_marks_get_all_valid_edges_as_mocked_dependency(
        self,
    ) -> None:
        """Module docstring must use the accurate two-part phrasing for get_all_valid_edges.

        The tracker line must convey that unit tests live in
        test_refresh_entity_summary.py AND that get_all_valid_edges is used
        here as a mocked dependency — not merely redirect with '→ see ...'.

        Assertions are scoped to the module docstring (via _extract_module_docstring)
        rather than the raw source, so they cannot spuriously pass or fail due to
        matching text in comments, helper-class docstrings, or other string literals.
        """
        src = _read_sibling('test_rebuild_entity_summaries.py')
        docstring = _extract_module_docstring(src)

        assert 'unit tests in test_refresh_entity_summary.py' in docstring, (
            "test_rebuild_entity_summaries.py module docstring must contain "
            "'unit tests in test_refresh_entity_summary.py' to accurately convey "
            "where the unit tests for get_all_valid_edges live"
        )

        assert 'used here as a mocked dependency' in docstring, (
            "test_rebuild_entity_summaries.py module docstring must contain "
            "'used here as a mocked dependency' to clarify that get_all_valid_edges "
            "is still exercised (as a mock) in this file"
        )

        assert '→ see test_refresh_entity_summary.py' not in docstring, (
            "test_rebuild_entity_summaries.py module docstring must NOT contain "
            "the stale arrow-only phrasing '→ see test_refresh_entity_summary.py'; "
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
      (a) the substring 'or []' appears in the contiguous comment/blank lines
          immediately before the `result_set = None` assignment inside
          test_none_result_set_returns_empty_dict
      (b) the comment also references 'get_all_valid_edges' by name

    NOTE: These assertions match literal substrings. If the pinned wording is ever
    rephrased, update the assertions in this class to match the new text.
    """

    @staticmethod
    def _lines_before_mutation() -> list[str]:
        """Return contiguous comment/blank lines immediately before the mutation.

        Scoped to the body of test_none_result_set_returns_empty_dict — finds the
        function definition first and searches forward within it for the mutation
        marker, so a second occurrence of the marker added elsewhere in the file
        will not silently anchor the check to the wrong site.

        Walks backward from the mutation collecting contiguous comment/blank lines
        rather than using a fixed-size window, so future additions to the comment
        block (extra context sentences, blank-line separators) don't push existing
        lines outside the slice and cause a false failure.
        """
        src = _read_sibling('test_refresh_entity_summary.py')
        lines = src.splitlines()
        fn_marker = 'def test_none_result_set_returns_empty_dict'
        mutation_marker = 'graph.ro_query.return_value.result_set = None'

        # Locate the enclosing function definition
        fn_idx: int | None = None
        for i, line in enumerate(lines):
            if fn_marker in line:
                fn_idx = i
                break
        if fn_idx is None:
            return []

        # Search forward within the function body for the mutation marker
        mutation_idx: int | None = None
        for i in range(fn_idx + 1, len(lines)):
            if mutation_marker in lines[i]:
                mutation_idx = i
                break
        if mutation_idx is None:
            return []

        # Walk backward collecting contiguous comment/blank lines
        result: list[str] = []
        for i in range(mutation_idx - 1, fn_idx, -1):
            stripped = lines[i].strip()
            if stripped.startswith('#') or stripped == '':
                result.insert(0, lines[i])
            else:
                break

        return result

    def test_lines_before_mutation_is_declared_staticmethod(self) -> None:
        """_lines_before_mutation must be decorated @staticmethod.

        The helper never references self, so declaring it as a regular instance
        method is a style error.  Inspects via __dict__ to see the raw descriptor
        rather than going through attribute access (which would unwrap the
        staticmethod descriptor and return the underlying function, hiding its type).
        """
        descriptor = TestNoneResultSetRationale704.__dict__['_lines_before_mutation']
        assert isinstance(descriptor, staticmethod), (
            "_lines_before_mutation never references self and must be decorated "
            "@staticmethod; found descriptor type: "
            f"{type(descriptor).__name__}"
        )

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
            "test_none_result_set_returns_empty_dict of test_refresh_entity_summary.py "
            "— has the mutation line or the enclosing function been renamed?"
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
