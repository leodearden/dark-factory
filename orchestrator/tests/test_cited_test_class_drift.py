"""Static guard: a test class CITED BY NAME in a src docstring or comment must exist.

A src docstring saying "pinned by ``TestFooBarBaz``" is a load-bearing claim
of coverage, and nothing checks it. ``orchestrator.verify_classify`` carried
one for TEN DAYS naming a class that existed nowhere in the repo (citation
landed 2026-08-05; the class was written 2026-08-15). A reader who trusts such
a claim does not go and write the test.

WHAT THIS ASSERTS, and nothing else: every cited IDENTIFIER RESOLVES to a real
``class Test...`` under one of the workspace's test trees.

WHAT IT DELIBERATELY DOES NOT DO:

* It is NOT a wording pin. It asserts nothing whatever about the prose around
  a citation — rewording a citing sentence is a no-op by construction, which
  ``TestRewordInvariance`` below demonstrates executably rather than by
  assertion. Do not extend this into a wording pin; same scope discipline as
  ``tests/scripts/test_setup_host_unit_installation.py::test_setup_md_disable_block_covers_every_foreign_orchestrator_unit``.
* There is NO allowlist and no carve-out table of any kind, so there is
  nothing here that can go stale.
* There is NO staleness assertion in EITHER direction: no "this name must
  still be cited" arm, and no assertion that any src file mentions any
  particular string. That direction makes another package's PROSE a
  merge-blocking gate on this module. It is the detector task 3554 removed on
  review, and the standing prohibition on reintroducing it — explicitly
  including "as a word-boundary or regex variant" — lives in
  ``tests/scripts/test_skills_module_config_decision.py``'s module docstring,
  on the removed ``test_no_unlisted_skills_mentioning_test_escapes_triage``.
  Here no carve-out entries exist at all, so that failure mode is
  structurally unreachable rather than merely unimplemented.

COVERAGE RULE — a rule, deliberately not a measured fraction: every ``Test`` +
>=2-CamelCase-segment name appearing in a COMMENT or STRING token of any
``<member>/src`` file is checked, in the backticked, bare, ``file.py::Name``
and line-wrapped forms alike. Out of scope by shape: single-segment
``Test<Word>`` names; citations in non-Python files; and names appearing in
CODE rather than prose, which are uses, not coverage claims.

Built in two halves, in that order — the layout of the precedent
``orchestrator/tests/test_marker_registration_drift.py``: unit tests of the
pure helpers against synthetic strings and ``tmp_path`` trees FIRST, then the
wired guard against the REAL tree. The real tree is green on arrival, so the
synthetic half carries the whole burden of proving the mechanism can FAIL.
"""
from __future__ import annotations

import ast
import re
import tokenize
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import cast

import pytest
from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT

# This file rglob('*.py')s both corpora, so it belongs to the whole-tree-scan
# family. The ceiling's derivation lives at its single canonical home,
# _orch_helpers.py::WHOLE_TREE_SCAN_TEST_TIMEOUT; the module-level mark is
# REQUIRED by test_whole_tree_scan_timeout_guard.py, which recomputes its
# scanner census from source on every run and checks this mark's VALUE.
pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

# Resolved from THIS FILE, never from the process CWD: merge-verify runs pytest
# from orchestrator/ while a plain run starts at the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

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


class TestRewordInvariance:
    """The executable policy discriminator: this guard pins IDENTIFIERS, never prose.

    The module docstring's reword-invariance claim points HERE. A wording pin
    — the shape ``roles.py``'s ARCHITECT rule 5 forbids, and that task 3554
    removed on review under the standing prohibition in
    ``tests/scripts/test_skills_module_config_decision.py`` — goes red on the
    first case below by definition. This guard cannot: rewording a citing
    sentence is a no-op by construction.

    If this module is ever blocked as a docstring meta-test, escalate citing
    the task description's policy paragraph and the two landed precedents
    (``orchestrator/tests/test_marker_registration_drift.py``,
    ``tests/scripts/test_setup_host_unit_installation.py``) — do NOT harden
    the regex.
    """

    _TERSE = '"""Pinned by ``TestOrderingIsPreserved``."""\n'
    _REWRITTEN = (
        '"""Summary line, rewritten from scratch.\n'
        '\n'
        '    Ordering across the whole batch is what this function actually\n'
        '        guarantees; the property is exercised end to end by\n'
        '        ``TestOrderingIsPreserved``, which is the only reason a\n'
        '        caller may rely on it.\n'
        '    """\n'
    )

    def test_rewording_the_sentence_around_a_citation_changes_nothing(self):
        """Different words, different clauses, different indentation and line
        breaks — same identifier, so the same extracted set."""
        assert set(_cited_names(self._TERSE)) == {'TestOrderingIsPreserved'}
        assert set(_cited_names(self._REWRITTEN)) == {'TestOrderingIsPreserved'}

    def test_moving_a_citation_from_a_docstring_into_a_comment_changes_nothing(self):
        comment = '# Ordering is pinned by TestOrderingIsPreserved.\n'
        assert set(_cited_names(comment)) == set(_cited_names(self._TERSE))

    def test_renaming_the_identifier_does_change_the_extracted_set(self):
        """The converse. Without it the invariance above would be vacuous."""
        renamed = self._REWRITTEN.replace('TestOrderingIsPreserved', 'TestOrderingIsStable')
        assert set(_cited_names(renamed)) == {'TestOrderingIsStable'}

    def test_prose_citing_no_test_class_extracts_nothing_however_long(self):
        assert _cited_names('"""' + 'Prose naming no identifier at all. ' * 40 + '"""\n') == {}


def _defined_test_classes(test_dirs: Iterable[Path], wanted: Collection[str]) -> set[str]:
    """The subset of *wanted* that is actually DEFINED as a class under *test_dirs*.

    Two stages, and only the second one decides. A bytes-level line-anchored
    prefilter (``^[ \\t]*class[ \\t]+<name>\\b``, MULTILINE, one alternation over
    the whole wanted set) is a CHEAP NARROWING that reads each test file once
    without decoding it; ``ast.parse`` then confirms — or refuses — every file
    that survives. Accepting a prefilter hit on its own would let a ``class
    TestX`` sitting inside a triple-quoted synthetic source string resolve a
    genuinely dangling citation, i.e. fail PERMISSIVE and hollow the guard out
    silently (``test_a_class_inside_a_string_literal_is_not_resolved`` pins
    this; ``test_marker_registration_drift.py::_applied_marker_names`` made the
    same AST-not-grep choice). The narrowing is what keeps that soundness
    affordable: parsing only the candidates costs a small fraction of parsing
    the whole test corpus, which was measured to dominate the sweep.

    Files are visited in sorted order so failure messages are deterministic
    under xdist. A file that cannot be read or parsed is re-raised as an
    ``AssertionError`` naming it — never swallowed, never skipped, because a
    skipped file is a file this guard is silently vacuous for.
    """
    if not wanted:
        return set()
    prefilter = re.compile(
        (r'^[ \t]*class[ \t]+(?:' + '|'.join(map(re.escape, sorted(wanted))) + r')\b').encode(),
        re.MULTILINE,
    )
    found: set[str] = set()
    for test_dir in test_dirs:
        for path in sorted(test_dir.rglob('*.py')):
            try:
                raw = path.read_bytes()
                if not prefilter.search(raw):
                    continue
                tree = ast.parse(raw)
            except (SyntaxError, ValueError, OSError) as exc:
                raise AssertionError(
                    f'{path} could not be read/parsed while resolving cited test '
                    f'class names: {exc!r}. Fix the file — a silently skipped '
                    f'file would make this guard vacuous for it.'
                ) from exc
            found.update(
                node.name
                for node in ast.walk(tree)
                if isinstance(node, ast.ClassDef) and node.name in wanted
            )
    return found


class TestDefinedTestClasses:
    """Unit tests for the resolver ``_defined_test_classes``, on synthetic trees."""

    @staticmethod
    def _tree(root: Path, files: dict[str, str]) -> Path:
        for name, text in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return root

    def test_a_class_in_a_nested_subdirectory_resolves_and_a_txt_sibling_does_not(self, tmp_path):
        self._tree(tmp_path, {
            'deep/nested/test_a.py': 'class TestFooBarBaz:\n    pass\n',
            'deep/nested/notes.txt': 'class TestQuuxCorgeGrault:\n',
        })
        wanted = {'TestFooBarBaz', 'TestQuuxCorgeGrault'}
        assert _defined_test_classes([tmp_path], wanted) == {'TestFooBarBaz'}

    def test_a_name_defined_nowhere_is_absent_from_the_result(self, tmp_path):
        self._tree(tmp_path, {'test_a.py': 'class TestFooBarBaz:\n    pass\n'})
        assert _defined_test_classes([tmp_path], {'TestNotDefinedAnywhere'}) == set()

    def test_a_class_inside_a_string_literal_is_not_resolved(self, tmp_path):
        """THE soundness case. A ``class Test...`` at column 0 inside a
        triple-quoted synthetic source matches the line-anchored prefilter, so
        a regex-only resolver would let a genuinely dangling citation resolve —
        the permissive direction, which hollows the guard out silently. This
        module's own unit tests embed exactly such sources, so the trap is
        self-inflicted, not hypothetical; it is the one
        ``test_marker_registration_drift.py::test_marker_name_inside_a_string_literal_is_ignored``
        guards for markers.
        """
        self._tree(tmp_path, {'test_a.py': 'SOURCE = """\nclass TestFooBarBaz:\n    pass\n"""\n'})
        assert _defined_test_classes([tmp_path], {'TestFooBarBaz'}) == set()

    def test_a_class_nested_inside_another_class_still_resolves(self, tmp_path):
        """``ast.walk``, not ``tree.body`` — nesting depth is not a reason to
        call a real class absent."""
        self._tree(tmp_path, {'test_a.py': 'class Outer:\n    class TestFooBarBaz:\n        pass\n'})
        assert _defined_test_classes([tmp_path], {'TestFooBarBaz'}) == {'TestFooBarBaz'}

    def test_empty_wanted_short_circuits_without_scanning(self):
        """Nothing is read at all — asserted directly, by handing the resolver a
        root that raises if it is ever swept."""

        class _ExplodingRoot:
            def rglob(self, pattern: str):
                raise AssertionError(f'swept for {pattern} despite an empty wanted set')

        assert _defined_test_classes([cast(Path, _ExplodingRoot())], set()) == set()

    def test_an_unparseable_file_matching_the_prefilter_raises_naming_it(self, tmp_path):
        self._tree(tmp_path, {'test_broken.py': 'class TestFooBarBaz:\n    def f(:\n        pass\n'})
        with pytest.raises(AssertionError, match='test_broken.py'):
            _defined_test_classes([tmp_path], {'TestFooBarBaz'})

    def test_an_unreadable_file_raises_naming_it(self, tmp_path):
        """A dangling symlink is yielded by ``rglob`` and raises ``OSError`` on
        read. Never swallowed, never silently skipped: a skipped file is a file
        this guard is vacuous for."""
        (tmp_path / 'test_dangling.py').symlink_to(tmp_path / 'missing.py')
        with pytest.raises(AssertionError, match='test_dangling.py'):
            _defined_test_classes([tmp_path], {'TestFooBarBaz'})
