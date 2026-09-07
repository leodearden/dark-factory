"""Static guard: a test class cited by name in a src docstring/comment must exist.

(Module docstring completed in step-4 — see plan.json step-4.)
"""
from __future__ import annotations

import re
import tokenize

import pytest

#: A citation is ``Test`` followed by AT LEAST TWO CamelCase segments.  That
#: one shape predicate covers the backticked, bare and ``file.py::Name`` forms
#: alike, and is why no allowlist, carve-out dict or import-resolution table
#: exists anywhere in this module.
_CITATION_RE = re.compile(r'\bTest(?:[A-Z][a-z0-9_]*){2,}')


def _cited_names(source: str) -> dict[str, int]:
    """``{cited class name: 1-indexed line of its first occurrence}`` in *source*.

    Reads COMMENT and STRING tokens ONLY, never code. A name written in code —
    an import, a call, a base class — is a use, not a claim that a test class
    of that name covers something; only prose makes that claim, so only prose
    is swept. (This is also what keeps the guard from tripping over its own
    unit tests' synthetic sources.)

    The shape rule is ``Test`` + >=2 CamelCase segments, and it has two
    deliberate consequences. It needs no backticks, so the bare and
    ``file.py::Name`` forms are covered on equal footing. And it deliberately
    does NOT see single-segment ``Test<Word>`` names — which is precisely how
    the third-party ``TestClient`` cited in ``dashboard.app.lifespan``'s
    docstring, and the ``TestBase``/``TestA`` placeholders in
    ``orchestrator.pytest_markers``' prose, stay out of the corpus with no
    carve-out table in existence to maintain or go stale. A genuine
    single-segment citation would be invisible to this guard: that residual
    gap is a RULE, stated here, not a measured count.

    The line reported is the line the NAME sits on, not the token's first
    line: a multi-line docstring token starts many lines above its citation.

    Raises on an unreadable/untokenizable *source* (``TokenError``,
    ``SyntaxError``, ``IndentationError`` all propagate) — a silent empty dict
    would make the guard vacuous for exactly the file it could not read.
    """
    cited: dict[str, int] = {}
    tokens = tokenize.generate_tokens(iter(source.splitlines(keepends=True)).__next__)
    for tok in tokens:
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        for match in _CITATION_RE.finditer(tok.string):
            line = tok.start[0] + tok.string[: match.start()].count('\n')
            cited.setdefault(match.group(), line)
    return cited


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
