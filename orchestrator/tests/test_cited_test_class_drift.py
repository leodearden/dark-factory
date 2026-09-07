"""Static guard: a test class cited by name in a src docstring/comment must exist.

(Module docstring completed in step-4 — see plan.json step-4.)
"""
from __future__ import annotations

import tokenize

import pytest


class TestCitedNames:
    """Unit tests for the pure extractor ``_cited_names``.

    Each positive case mirrors a citation form that occurs for real under the
    workspace members' ``src/`` trees; each negative case is a string that
    LOOKS like a citation and must not be treated as one.
    """

    def test_backticked_docstring_citation_is_extracted(self):
        """The double-backtick form, e.g. as used around
        ``orchestrator.verify_classify``'s pins."""
        source = '"""Ordering is pinned by ``TestOrderingIsPreserved``."""\n'
        assert _cited_names(source) == {'TestOrderingIsPreserved': 1}

    def test_bare_comment_citation_in_path_colon_colon_form_is_extracted(self):
        """The bare ``file.py::Name`` idiom — the majority form 4240's
        backtick-only extractor could not see; used by e.g.
        ``fused_memory.services.completion_claim_gate``."""
        source = '# see tests/test_x.py::TestClauseBoundaryIsolation for the pin\n'
        assert _cited_names(source) == {'TestClauseBoundaryIsolation': 1}

    def test_line_is_the_name_s_line_not_the_token_s_first_line(self):
        """A multi-line STRING token starts many lines above its citation, so
        the reported line must be recovered from the offset WITHIN the token.
        The repeat on the last line also pins first-occurrence-wins."""
        source = '"""Summary.\n\nPinned by ``TestOrderingIsPreserved``.\n\nAnd again: ``TestOrderingIsPreserved``.\n"""\n'
        assert _cited_names(source) == {'TestOrderingIsPreserved': 3}

    def test_two_distinct_names_in_one_token_are_both_extracted(self):
        source = '"""Both ``TestOrderingIsPreserved`` and TestClauseBoundaryIsolation."""\n'
        assert _cited_names(source) == {
            'TestOrderingIsPreserved': 1,
            'TestClauseBoundaryIsolation': 1,
        }

    def test_english_prose_words_are_not_citations(self):
        """``Test``/``Tests``/``Tested`` carry zero uppercase segments."""
        assert _cited_names('"""Test the thing. Tests run. Tested already."""\n') == {}

    def test_single_segment_names_are_excluded_by_shape(self):
        """``TestClient`` (starlette, cited in ``dashboard.app.lifespan``'s
        docstring) and the ``TestBase``/``TestA`` placeholders in
        ``orchestrator.pytest_markers``' prose stay out of the corpus with NO
        carve-out table in existence — the shape rule alone excludes them."""
        source = '"""Uses ``TestClient``; cf. TestBase, TestCase, TestA."""\n'
        assert _cited_names(source) == {}

    def test_code_identifiers_are_not_prose_citations(self):
        """Only COMMENT and STRING tokens are read: writing a name in CODE is
        not a claim of coverage, so an import or a call yields nothing."""
        source = 'from helpers import TestFooBarBaz\n\nTestFooBarBaz()\n'
        assert _cited_names(source) == {}

    def test_untokenizable_source_raises_instead_of_reporting_no_citations(self):
        """Loud, not fail-soft: a silent empty dict makes the guard vacuous for
        exactly the file it could not read."""
        with pytest.raises((tokenize.TokenError, SyntaxError)):
            _cited_names('"""unterminated ``TestOrderingIsPreserved``\n')
