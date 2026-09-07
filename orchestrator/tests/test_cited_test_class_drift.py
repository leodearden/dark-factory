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


def _wrapped_candidates(source: str, name: str) -> set[str]:
    """Names *name* might be, if its citation was WRAPPED across a line break.

    A CANDIDATE GENERATOR ONLY, and deliberately unsound. Joining line-wraps
    during EXTRACTION was measured to invent names that exist nowhere: in
    ``shared.locking`` a comment line ends with ``...::TestFileExtensionsDriftGuard``
    and the next begins ``# Drift guard (...)``, which this join turns into
    ``TestFileExtensionsDriftGuardDrift``
    (``test_the_false_positive_that_decided_the_design`` pins exactly that).
    Applied as an extraction rule it would red-wall the guard on arrival with
    pure fabrications.

    What makes an unsound generator safe is the CALLER, not the generator:
    ``_dangling_citations`` invokes this only for a name that has ALREADY
    failed to resolve, and honours a candidate only when the joined string is
    itself an AST-confirmed test class. The fabricated name above can never be
    produced there, because its bare first half resolves on its own and the
    join is therefore never attempted. Do not "simplify" this back into an
    extraction rule, and never call it during extraction.
    """
    joiner = re.compile(re.escape(name) + r'-?[ \t]*\n[ \t]*#?[ \t]*([A-Z][A-Za-z0-9_]*)')
    return {name + match.group(1) for match in joiner.finditer(source)}


class TestWrappedCandidates:
    """Unit tests for the line-wrap JOIN generator ``_wrapped_candidates``.

    Both wrap shapes below are drawn from citations that exist for real in the
    tree, so the fallback is fitted to the population it must recover, not to
    an invented one.
    """

    def test_hyphen_wrap_joins(self):
        """The shape in ``orchestrator.verify``."""
        source = (
            '                # double-add. Pinned by TestRunScopedVerificationReverse-\n'
            '                # DependencyGuards (test_verify_reverse_dep.py, task 2607\n'
        )
        assert _wrapped_candidates(source, 'TestRunScopedVerificationReverse') == {
            'TestRunScopedVerificationReverseDependencyGuards'
        }

    def test_no_hyphen_wrap_joins(self):
        """The shape in ``orchestrator.workflow`` — wrapped with no hyphen at
        all, which is why the join cannot simply key on a trailing dash."""
        source = (
            '                    # the REAL _mark_blocked (TestAlreadyLandedLadderWith\n'
            '                    # RealMarkBlocked) rather than a stub:\n'
        )
        assert _wrapped_candidates(source, 'TestAlreadyLandedLadderWith') == {
            'TestAlreadyLandedLadderWithRealMarkBlocked'
        }

    def test_wrap_inside_a_docstring_without_a_comment_prefix_joins(self):
        source = '"""Pinned by TestAlreadyLandedLadderWith\n    RealMarkBlocked.\n    """\n'
        assert _wrapped_candidates(source, 'TestAlreadyLandedLadderWith') == {
            'TestAlreadyLandedLadderWithRealMarkBlocked'
        }

    def test_a_name_not_at_end_of_line_yields_nothing(self):
        assert _wrapped_candidates('# Pinned by TestFooBarBaz today.\n', 'TestFooBarBaz') == set()

    def test_a_lowercase_continuation_yields_nothing(self):
        source = '# Pinned by TestFooBarBaz\n# and by nothing else.\n'
        assert _wrapped_candidates(source, 'TestFooBarBaz') == set()

    def test_the_false_positive_that_decided_the_design(self):
        """THE reason this helper is a candidate GENERATOR and not an
        extraction rule. In ``shared.locking`` a comment line ends with
        ``...::TestFileExtensionsDriftGuard`` and the NEXT line begins ``# Drift
        guard (...)``, so this pure helper invents a name that exists nowhere.
        Asserted plainly rather than papered over: the helper alone is UNSOUND,
        and its safety comes entirely from the caller in ``_dangling_citations``
        invoking it ONLY for names that already failed to resolve, and honouring
        a candidate ONLY when the joined name is itself an AST-confirmed class.
        """
        source = (
            '# Drift guard (this shared copy): shared/tests/test_locking.py::TestFileExtensionsDriftGuard\n'
            '# Drift guard (the other copy): fused-memory/tests/test_lock_charter_guard.py\n'
        )
        assert _wrapped_candidates(source, 'TestFileExtensionsDriftGuard') == {
            'TestFileExtensionsDriftGuardDrift'
        }


class TestDanglingCitations:
    """THE efficacy proof: evidence that the mechanism can actually FAIL.

    The real tree is green on arrival, so the wired guard below can only ever
    demonstrate that it passes. Everything that shows this guard is not vacuous
    lives here, on synthetic trees — the same division of labour as
    ``test_marker_registration_drift.py::TestUnregisteredMarkers``.
    """

    @staticmethod
    def _corpus(root: Path, src: dict[str, str], tests: dict[str, str]):
        src_dir, test_dir = root / 'pkg' / 'src', root / 'pkg' / 'tests'
        for target, files in ((src_dir, src), (test_dir, tests)):
            for name, text in files.items():
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
        return [src_dir], [test_dir]

    _UNRELATED = {'test_a.py': 'class TestSomethingElseEntirely:\n    pass\n'}

    def test_a_planted_dangling_citation_is_reported_with_its_file_and_line(self, tmp_path):
        src_dirs, test_dirs = self._corpus(
            tmp_path,
            {'mod.py': '"""Summary.\n\nPinned by ``TestNoSuchClassAnywhere``.\n"""\n'},
            self._UNRELATED,
        )
        assert _dangling_citations(src_dirs, test_dirs) == {
            'TestNoSuchClassAnywhere': f'{src_dirs[0] / "mod.py"}:3'
        }

    def test_a_citation_naming_a_real_class_is_not_reported(self, tmp_path):
        src_dirs, test_dirs = self._corpus(
            tmp_path,
            {'mod.py': '"""Pinned by ``TestSomethingElseEntirely``."""\n'},
            self._UNRELATED,
        )
        assert _dangling_citations(src_dirs, test_dirs) == {}

    def test_the_historical_defect_reproduced_in_miniature(self, tmp_path):
        """The exact shape ``orchestrator.verify_classify`` carried for ten
        days: a backticked pin naming a class nobody had written yet."""
        src_dirs, test_dirs = self._corpus(
            tmp_path,
            {'verify_classify.py': (
                '"""Pinned by ``TestAnchoredSlotTimeoutWithCollateralIsEnvTransient``,\n'
                'which exercises the classifier end to end."""\n'
            )},
            self._UNRELATED,
        )
        assert set(_dangling_citations(src_dirs, test_dirs)) == {
            'TestAnchoredSlotTimeoutWithCollateralIsEnvTransient'
        }

    def test_both_wrap_shapes_resolve_when_the_joined_name_exists(self, tmp_path):
        src_dirs, test_dirs = self._corpus(
            tmp_path,
            {'mod.py': (
                '# double-add. Pinned by TestRunScopedVerificationReverse-\n'
                '# DependencyGuards (the sibling suite).\n'
                '# the REAL _mark_blocked (TestAlreadyLandedLadderWith\n'
                '# RealMarkBlocked) rather than a stub:\n'
            )},
            {'test_a.py': (
                'class TestRunScopedVerificationReverseDependencyGuards:\n    pass\n\n'
                'class TestAlreadyLandedLadderWithRealMarkBlocked:\n    pass\n'
            )},
        )
        assert _dangling_citations(src_dirs, test_dirs) == {}

    def test_a_wrap_looking_pair_whose_bare_name_resolves_is_not_mangled(self, tmp_path):
        """The ``shared.locking`` false positive stays absent end to end: the
        bare name resolves, so the join is never even attempted."""
        src_dirs, test_dirs = self._corpus(
            tmp_path,
            {'locking.py': (
                '# Drift guard (this copy): tests/test_locking.py::TestFileExtensionsDriftGuard\n'
                '# Drift guard (other copy): tests/test_charter.py::TestExtensionlessNamesDriftGuard\n'
            )},
            {'test_a.py': (
                'class TestFileExtensionsDriftGuard:\n    pass\n\n'
                'class TestExtensionlessNamesDriftGuard:\n    pass\n'
            )},
        )
        assert _dangling_citations(src_dirs, test_dirs) == {}

    def test_a_wrapped_citation_whose_joined_name_is_absent_is_still_reported(self, tmp_path):
        src_dirs, test_dirs = self._corpus(
            tmp_path,
            {'mod.py': '# Pinned by TestNoSuchClassAnywhere-\n# ButWrapped (see below).\n'},
            self._UNRELATED,
        )
        dangling = _dangling_citations(src_dirs, test_dirs)
        assert set(dangling) == {'TestNoSuchClassAnywhere'}
        reported = dangling['TestNoSuchClassAnywhere']
        assert f'{src_dirs[0] / "mod.py"}:1' in reported
        assert 'TestNoSuchClassAnywhereButWrapped' in reported

    def test_the_same_resolving_citation_reworded_is_empty_both_ways(self, tmp_path):
        """The reword discriminator at GUARD level, not just extractor level."""
        tests = {'test_a.py': 'class TestOrderingIsPreserved:\n    pass\n'}
        terse = self._corpus(
            tmp_path / 'terse', {'mod.py': '"""Pinned by ``TestOrderingIsPreserved``."""\n'}, tests
        )
        verbose = self._corpus(
            tmp_path / 'verbose',
            {'mod.py': (
                '"""A completely different summary sentence.\n\n'
                '    What this function guarantees is ordering across the whole\n'
                '        batch, and that property is exercised end to end by\n'
                '        ``TestOrderingIsPreserved``.\n'
                '    """\n'
            )},
            tests,
        )
        assert _dangling_citations(*terse) == {}
        assert _dangling_citations(*verbose) == {}

    def test_a_src_tree_with_no_citations_at_all_raises(self, tmp_path):
        """LIVENESS. A sweep that finds nothing is broken, not clean — the one
        outcome that would let this guard pass forever while checking nothing.
        """
        src_dirs, test_dirs = self._corpus(
            tmp_path, {'mod.py': '"""Prose citing no test class at all."""\n'}, self._UNRELATED
        )
        with pytest.raises(AssertionError, match='no cited test class names'):
            _dangling_citations(src_dirs, test_dirs)
