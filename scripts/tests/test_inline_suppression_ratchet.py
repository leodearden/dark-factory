"""Boundary tests for ``scripts/inline_suppressions.py`` — PRD γ1, scenarios 1-11.

WHAT IS UNDER TEST.  The inline-suppression scanner: the kind table and its
COMMENT-token scan, D7's content-addressed multiset key, D8's consumer model,
D9's ratified classes, and the ``--check`` / ``--seed`` / ``--tighten`` /
``--json`` verbs with the 0/1/2 exit ladder the PRD Contract fixes.
``plans/inv12-exceptions-owned-or-ratified-prd.md``, boundary-test sketch rows
1-11.

HOW IT IS TESTED, and why the shape differs between families.  Almost every
test builds a throwaway git repository under ``tmp_path`` and calls
``inline_suppressions.main([...])`` IN-PROCESS, capturing stdout/stderr with
``capsys``: the verbs' whole contract is an exit code plus rendered lines, and
an in-process call gets both without paying a subprocess per scenario.  The two
exceptions are deliberate and each is a case an in-process call cannot prove:

* the import-failure test (Contract: "an ImportError is 2, never 1") needs a
  real interpreter, because the fault it pins happens before ``main`` exists;
* the first-party code drift guard runs
  ``fused-memory/scripts/check_bare_magicmock_config.py`` as a subprocess and
  reads the violation messages it EMITS, rather than importing its private
  ``_RULE_A_CODE`` / ``_RULE_B_CODE`` constants — ``docs/code-quality.md``'s
  Tests stance (a test that reads another module's private attributes pins
  implementation rather than behaviour).

A REAL GIT REPOSITORY, not a bare directory.  The scanner enumerates its corpus
from ``git ls-files``, so trackedness is a property the fixtures must exercise
rather than one the tests assert by reading the code — the same argument
``scripts/tests/test_design_invariants_consistency.py::_write_scan_tree`` gives.

NO WALL-CLOCK ASSERTION APPEARS IN THIS MODULE, on purpose.  The PRD's ≤10 s
scan budget is enforced here as COUNTED WORK (``files_enumerated`` versus
``files_tokenized``), never as ``assert elapsed < N``.
``orchestrator/tests/test_merge_lane_ratchet.py`` already ruled on exactly this
for the sibling ratchet — "count WORK, never wall-clock" — and the measurement
behind that ruling holds here: this scan costs ~5 CPU-seconds but took 6.5-6.8 s
of wall clock on a box at load 105, and the merge gate runs on that same box.
Subprocesses therefore get a generous ``timeout=`` rather than a clock guard.

XDIST-SAFE.  The merge gate runs ``pytest … -n auto --dist loadgroup``, so every
test here keeps its mutable state inside ``tmp_path`` and changes no process-wide
state outside ``monkeypatch``.
"""

import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import inline_suppressions
import pytest
from shared.governed_exceptions import INLINE_MARKER_FORMS, Policy

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / 'scripts' / 'inline_suppressions.py'

#: Where a fixture tree's baseline lives.  The real one is
#: ``scripts/inline_suppression_baseline.json`` and is κ1's to seed; fixtures
#: keep theirs inside ``tmp_path`` so no test can reach the committed file.
_BASELINE_NAME = 'inline_suppression_baseline.json'

#: Generous, and not a performance assertion — see the module docstring.  A
#: subprocess that blows through this has hung, which is worth a loud
#: ``TimeoutExpired`` rather than a silent wait.
_SUBPROCESS_TIMEOUT_SECS = 120


def _scrubbed_git_env() -> dict[str, str]:
    """``os.environ`` with every ``GIT_*`` override removed.

    ``GIT_DIR``, ``GIT_WORK_TREE``, ``GIT_INDEX_FILE`` and
    ``GIT_CEILING_DIRECTORIES`` are inherited by default, and any one of them
    silently retargets a git invocation at a different repository than its
    ``cwd`` implies — ``git -C <path>`` READS as "act on <path>" but only
    changes directory, while ``GIT_DIR`` skips repository discovery outright.
    A fixture that lost that race would ``git add`` into the live checkout.
    Spelled as in ``test_design_invariants_consistency.py::_scrubbed_git_env``;
    ``df_pytest_isolation._df_git_env_hermetic`` is the suite-wide second line
    of the same defence.
    """
    return {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}


def _run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run ``git`` *args* against *cwd* under :func:`_scrubbed_git_env`."""
    return subprocess.run(
        ['git', *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        timeout=_SUBPROCESS_TIMEOUT_SECS,
        env=_scrubbed_git_env(),
    )


def _write_files(root: Path, files: Mapping[str, str]) -> None:
    """Write every *files* entry under *root*, creating parents as needed.

    Split out of :func:`_write_fixture_tree` because the consumer-model tests
    need a tree of ``pyproject.toml`` files and no git at all: the nearest-config
    walk reads the filesystem, so making those tests pay for a repository would
    be ceremony that tests nothing.
    """
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')


def _write_fixture_tree(
    root: Path, files: Mapping[str, str], *, baseline: bool = False
) -> Path:
    """Build a git repo at *root* holding *files*, and return its baseline path.

    *files* maps a repo-relative path to its whole content — source modules and
    ``pyproject.toml`` alike, since the scanner reads the nearest pyproject as
    ordinary tracked content and a second parameter for it would only be a
    second way to write a file.

    Every path is TRACKED: ``git init -q`` then ``git add -A -f``.  No commit,
    and no ``user.name`` / ``user.email`` — ``git ls-files`` reads the INDEX,
    not history, so a commit would be ceremony the scanner never looks at.  The
    ``-f`` defeats any ambient global gitignore.  A test that needs an UNTRACKED
    file writes it itself after this call.

    With ``baseline=True`` the scanner's own ``--seed`` verb seeds the returned
    path from this very tree, so a fixture baseline can never drift from the key
    format the scanner emits.  The path is returned either way: the tests that
    need an ABSENT baseline, or a deliberately corrupt one, need somewhere to
    point ``--baseline`` at just as much as the seeded ones do.
    """
    _write_files(root, files)

    for args in (['init', '-q'], ['add', '-A', '-f']):
        _run_git(args, cwd=root)

    baseline_path = root / _BASELINE_NAME
    if baseline:
        seeded = inline_suppressions.main(
            ['--seed', '--root', str(root), '--baseline', str(baseline_path)]
        )
        assert seeded == 0, f'fixture --seed failed with exit {seeded}'
    return baseline_path


def _check(root: Path, baseline_path: Path, *paths: str) -> int:
    """Run ``--check`` over *root*, optionally scoped to *paths*."""
    return inline_suppressions.main(
        ['--check', '--root', str(root), '--baseline', str(baseline_path), *paths]
    )


def _python_env_without_shared() -> dict[str, str]:
    """An environment in which ``import shared`` cannot resolve.

    ``PYTHONPATH`` is emptied AND ``PYTHONNOUSERSITE`` is set, because the
    scanner is meant to fail on a missing ``shared`` and not on the ambient
    editable install that ``sys.path`` would otherwise supply.
    """
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    env['PYTHONNOUSERSITE'] = '1'
    return env


def _run_script(args: list[str], *, cwd: Path, script: Path, env: dict[str, str] | None = None):
    """Run *script* as a real subprocess with ``sys.executable``."""
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_SECS,
        env=env,
    )


def _sites(source: str, *, path: str = 'm.py') -> list:
    """Every Site in *source*, flattened out of the comments that hold them."""
    return [
        site
        for comment in inline_suppressions.scan_source(source, path=path)
        for site in comment.sites
    ]


# ---------------------------------------------------------------------------
# Layer 1 — the kind table and the COMMENT-token scan.


def test_scan_finds_every_kind_the_table_declares():
    """All five kinds, so a row nobody detects turns this red.

    Driven off ``Kind`` itself rather than a hand-written list: a sixth row
    added to the table (the PRD names ``pytest.mark.skip`` and ``shellcheck
    disable`` as the expected extensions) with no detection pattern would
    otherwise pass a list that never mentioned it.
    """
    source = '\n'.join(
        [
            'a = 1  # type: ignore[arg-type]',
            'b = 2  # noqa: E402',
            'c = 3  # pyright: ignore[reportArgumentType]',
            'd = 4  # pragma: no cover',
            'e = 5  # nosec',
        ]
    )

    assert {site.kind for site in _sites(source)} == set(inline_suppressions.Kind)


def test_site_carries_path_line_kind_codes_and_the_stripped_physical_line():
    source = 'x = 1\n\n    # noqa: E402\n'

    (site,) = _sites(source, path='pkg/mod.py')

    assert site.path == 'pkg/mod.py'
    assert site.line == 3
    assert site.kind is inline_suppressions.Kind.NOQA
    assert site.codes == ('E402',)
    assert site.text == '# noqa: E402'


def test_bracketed_codes_are_extracted_in_source_order():
    """Order is the SOURCE's here; sorting is the key's job, not the scan's."""
    (mypy_site,) = _sites('a = 1  # type: ignore[attr-defined, arg-type]')
    (pyright_site,) = _sites('a = 1  # pyright: ignore[reportArgumentType]')

    assert mypy_site.codes == ('attr-defined', 'arg-type')
    assert pyright_site.codes == ('reportArgumentType',)


def test_noqa_codes_are_extracted_with_or_without_a_space_after_the_colon():
    (listed,) = _sites('a = 1  # noqa: E402,F401')
    (tight,) = _sites('a = 1  # noqa:E402')

    assert listed.codes == ('E402', 'F401')
    assert tight.codes == ('E402',)


def test_noqa_code_extraction_stops_at_the_first_token_that_is_not_a_code():
    """The live tree's dominant shape: a code, then a dash, then prose.

    Measured over this repository 2026-09-19 — 232 distinct ``noqa`` tails, of
    which the prose-carrying ones (``# noqa: F401  — the binding IS the
    wiring``, ``# noqa: E402  (import after path fix)``) are common enough that
    splitting the whole tail on whitespace manufactures 'the', 'never', 'a' and
    'IS' as rule codes.  Stopping at the first non-code token reproduces ruff's
    own reading and yields 40 distinct codes over the tree, every one of them
    real.
    """
    (dashed,) = _sites('a = 1  # noqa: F401  — the binding IS the wiring')
    (parenthesised,) = _sites('a = 1  # noqa: E402  (import after path fix)')

    assert dashed.codes == ('F401',)
    assert parenthesised.codes == ('E402',)


def test_a_kebab_case_code_is_a_code_only_in_first_position():
    """The union of the two consumers' own grammars, and nothing wider.

    ruff recognises a ``<letters><digits>`` code anywhere in the list;
    ``fused-memory/scripts/check_bare_magicmock_config.py`` anchors its
    kebab-case code immediately after ``noqa:``.  Admitting kebab-case
    ANYWHERE would read the tail of ``# noqa: F401  re-export shim`` as a
    second code, which is neither consumer's rule.
    """
    (first_party,) = _sites('a = 1  # noqa: bare-magicmock — deliberate')
    (invented,) = _sites('a = 1  # noqa: bare-something-else')
    (prose,) = _sites('a = 1  # noqa: F401  re-export shim')

    assert first_party.codes == ('bare-magicmock',)
    assert invented.codes == ('bare-something-else',)
    assert prose.codes == ('F401',)


def test_bare_markers_carry_no_codes():
    """Absence of codes is an empty tuple, never None — a Site always has a
    codes tuple, so no consumer of it needs a null check."""
    bare_noqa, bare_ignore, pragma, nosec = _sites(
        '\n'.join(
            [
                'a = 1  # noqa',
                'b = 2  # type: ignore',
                'c = 3  # pragma: no cover',
                'd = 4  # nosec',
            ]
        )
    )

    assert bare_noqa.codes == ()
    assert bare_ignore.codes == ()
    assert pragma.codes == ()
    assert nosec.codes == ()


def test_a_suppression_inside_a_string_literal_is_not_a_site():
    """THE REASON THIS IS ``tokenize`` AND NOT A REGEX.

    A regex over source text cannot tell a comment from a string that merely
    mentions one, and this repository is full of the latter — every test that
    asserts on a marker, and this very module.  Same argument as
    ``scripts/merge_lane_metrics.py::_comment_lines``.
    """
    source = '\n'.join(
        [
            'DOC = "# type: ignore[arg-type]"',
            "OTHER = '# noqa: E402'",
            'TRIPLE = """',
            '# pyright: ignore[reportAny]',
            '"""',
        ]
    )

    assert _sites(source) == []


def test_the_word_nanosecond_does_not_register_as_nosec():
    """``nosec`` is a word, not a substring — 'nanosecond' contains it."""
    assert _sites('a = 1  # 5 nanoseconds is the budget') == []
    assert _sites('a = 1  # nanosecond') == []
    assert _sites('a = 1  # nosecret') == []


def test_one_comment_carrying_a_suppression_and_a_disposition_is_one_site():
    """D6's shape: the disposition rides in the SAME comment token.

    The Site's text is the whole stripped physical line — the disposition
    included — because that is what D7 digests, so editing the disposition
    changes the key just as editing the code does.
    """
    (site,) = _sites('    value = call()  # type: ignore[attr-defined]  # debt: task 5601')

    assert site.kind is inline_suppressions.Kind.TYPE_IGNORE
    assert site.codes == ('attr-defined',)
    assert site.text == 'value = call()  # type: ignore[attr-defined]  # debt: task 5601'


def test_one_comment_carrying_two_kinds_yields_one_site_per_kind():
    comment_sites = _sites('a = 1  # type: ignore[arg-type]  # noqa: E402')

    assert [(site.kind, site.codes) for site in comment_sites] == [
        (inline_suppressions.Kind.TYPE_IGNORE, ('arg-type',)),
        (inline_suppressions.Kind.NOQA, ('E402',)),
    ]


def test_scan_source_yields_every_comment_not_only_the_suppressing_ones():
    """The disposition-without-a-suppression violation (scenario 7's second
    half) is a property of a comment that produced NO Site, so the scan has to
    hand back the plain comments too."""
    comments = inline_suppressions.scan_source(
        'a = 1  # debt: task 5\nb = 2  # noqa: E402\n', path='m.py'
    )

    assert [(comment.line, comment.text, len(comment.sites)) for comment in comments] == [
        (1, '# debt: task 5', 0),
        (2, '# noqa: E402', 1),
    ]


# ---------------------------------------------------------------------------
# Layer 1 — tracked-file enumeration, the byte prefilter, and the failure
# polarity that makes an unreadable file loud instead of invisible.


def test_only_tracked_files_are_scanned(tmp_path: Path):
    """An untracked marker-bearing file is invisible.

    ``git ls-files``, never a filesystem walk: a scratch copy, a gitignored
    watcher artifact or a half-written module must not be able to turn the gate
    red on nothing but local working-tree state.  The cost is stated rather
    than hidden — a new file that has not been ``git add``ed is not scanned
    either, and an author running this locally before staging gets a green
    verdict on content the gate will see once it is staged.
    """
    _write_fixture_tree(tmp_path, {'tracked.py': 'a = 1  # noqa: E402\n'})
    (tmp_path / 'untracked.py').write_text('b = 2  # noqa: F401\n', encoding='utf-8')

    scan = inline_suppressions.scan_tree(tmp_path)

    assert [site.path for site in scan.sites] == ['tracked.py']


def test_scan_counts_the_files_it_enumerated_and_the_files_it_tokenized(tmp_path: Path):
    """THE COUNTED WORK THE ≤10 s BUDGET RESTS ON.

    The prefilter is the whole reason the scan fits the budget, and the only
    honest way to assert it on a loaded box is to count files rather than
    seconds — see the module docstring.  A file with no marker byte substring
    is ENUMERATED and not TOKENIZED.
    """
    _write_fixture_tree(
        tmp_path,
        {
            'marked.py': 'a = 1  # noqa: E402\n',
            'plain.py': 'b = 2\nc = 3\n',
            'also_plain.py': '"""Just a docstring."""\n',
        },
    )

    scan = inline_suppressions.scan_tree(tmp_path)

    assert scan.files_enumerated == 3
    assert scan.files_tokenized == 1


def test_a_tracked_file_that_cannot_be_tokenized_is_an_instrument_failure(tmp_path: Path):
    """BOUNDARY SCENARIO 8 — never skipped, and never a violation.

    A file the scanner cannot read is not evidence about that file's
    suppressions; it is evidence that the instrument is broken.  Skipping it
    would make the count read LOW, which is the direction that reports a breach
    as a clean tree (INV-11 ``no-silent-fail-soft``).  Exit 2, and the message
    names the file so the operator knows which one to open.
    """
    _write_fixture_tree(tmp_path, {'broken.py': 'value = (  # noqa: E402\n'})

    with pytest.raises(inline_suppressions.InstrumentFailure) as caught:
        inline_suppressions.scan_tree(tmp_path)

    assert 'broken.py' in str(caught.value)


def test_a_tracked_file_with_broken_indentation_is_an_instrument_failure(tmp_path: Path):
    """The trio is ``(TokenError, IndentationError, SyntaxError)``, not
    ``TokenError`` alone — a file whose dedent matches no outer level raises
    the second and would otherwise escape as a bare traceback.

    A mismatched DEDENT specifically, not merely a surprising indent: measured
    against this interpreter, ``tokenize`` accepts an over-indented line
    without complaint (the parser rejects it later, but the tokenizer does not),
    so a fixture built on one would assert nothing.
    """
    _write_fixture_tree(
        tmp_path, {'bad_indent.py': 'if True:\n      a = 1  # noqa: E402\n   b = 2\n'}
    )

    with pytest.raises(inline_suppressions.InstrumentFailure) as caught:
        inline_suppressions.scan_tree(tmp_path)

    assert 'bad_indent.py' in str(caught.value)


def test_a_tracked_file_that_is_not_utf8_is_an_instrument_failure(tmp_path: Path):
    """Decoding is its own step with its own catch, ahead of tokenize."""
    _write_fixture_tree(tmp_path, {'placeholder.py': 'a = 1\n'})
    (tmp_path / 'undecodable.py').write_bytes(b'a = 1  # noqa: E402\nb = "\xff\xfe"\n')
    _run_git(['add', '-A', '-f'], cwd=tmp_path)

    with pytest.raises(inline_suppressions.InstrumentFailure) as caught:
        inline_suppressions.scan_tree(tmp_path)

    assert 'undecodable.py' in str(caught.value)


def test_a_broken_file_with_no_marker_substring_is_never_read(tmp_path: Path):
    """The prefilter's REACH, stated rather than discovered later.

    A file carrying no marker byte substring cannot hold a suppression, so it
    is never decoded and never tokenized — and therefore a syntactically broken
    one is not an instrument failure either.  That is honest: this scanner
    never claimed to parse the tree, only to find its markers, and the bytes it
    did read prove there is nothing here to find.
    """
    _write_fixture_tree(tmp_path, {'broken.py': 'value = (\n'})

    scan = inline_suppressions.scan_tree(tmp_path)

    assert scan.files_enumerated == 1
    assert scan.files_tokenized == 0
    assert scan.sites == ()


def test_enumeration_refuses_rather_than_returning_an_empty_corpus(tmp_path: Path):
    """An empty corpus and a clean corpus are INDISTINGUISHABLE downstream.

    Only one of them is good news, so a failed ``git ls-files`` raises instead
    of degrading to ``[]`` — the argument
    ``scripts/audit_manifest_descriptor_drift.py::ManifestDiscoveryUnavailable``
    makes, and it is sharper here because an empty scan would seed or tighten a
    baseline to nothing and silently open the gate.
    """
    not_a_repo = tmp_path / 'loose'
    not_a_repo.mkdir()

    with pytest.raises(inline_suppressions.InstrumentFailure):
        inline_suppressions.scan_tree(not_a_repo)


def test_enumeration_refuses_when_git_cannot_be_run_at_all(tmp_path: Path, monkeypatch):
    """A missing ``git`` is an OSError, not a non-zero return — a refusal that
    only checked the return code would let this one through as a clean scan."""
    _write_fixture_tree(tmp_path, {'a.py': 'x = 1  # noqa: E402\n'})
    monkeypatch.setenv('PATH', str(tmp_path / 'empty-bin'))

    with pytest.raises(inline_suppressions.InstrumentFailure):
        inline_suppressions.scan_tree(tmp_path)


def test_a_tracked_path_whose_worktree_file_is_gone_is_passed_over(tmp_path: Path):
    """``git ls-files`` reads the INDEX, so it lists a file deleted from the
    worktree.  That is an ordinary mid-edit state, not a broken instrument."""
    _write_fixture_tree(tmp_path, {'kept.py': 'a = 1  # noqa: E402\n', 'gone.py': 'b = 2\n'})
    (tmp_path / 'gone.py').unlink()

    scan = inline_suppressions.scan_tree(tmp_path)

    assert [site.path for site in scan.sites] == ['kept.py']


# ---------------------------------------------------------------------------
# D7 — the multiset key: (kind, sorted codes, digest of the stripped line),
# and deliberately NO path.


def _keys(source: str, *, path: str = 'm.py') -> list:
    return [inline_suppressions.key_for(site) for site in _sites(source, path=path)]


def test_the_same_marker_line_in_two_files_yields_one_key():
    """NO PATH IN THE KEY, which is what makes D7 a multiset and not a
    per-file count."""
    line = 'value = call()  # type: ignore[attr-defined]\n'

    assert _keys(line, path='pkg/a.py') == _keys(line, path='other/deeply/nested/b.py')


def test_renaming_and_splitting_a_file_changes_no_key(tmp_path: Path):
    """BOUNDARY SCENARIO 4 — 40 grandfathered markers survive a rename and a
    split, at the multiset level.

    Under the per-file counts the brief originally proposed, splitting this
    repository's 391-marker file would make every moved marker NEW and
    regenerating the baseline the only exit — the one operation the whole
    ratchet exists to prevent anybody performing casually.
    """
    del tmp_path
    lines = [f'v{index} = call()  # type: ignore[attr-defined]' for index in range(40)]

    before = Counter(_keys('\n'.join(lines) + '\n', path='big.py'))
    after = Counter(
        _keys('\n'.join(lines[:17]) + '\n', path='renamed/part_one.py')
        + _keys('\n'.join(lines[17:]) + '\n', path='renamed/part_two.py')
    )

    assert before == after
    assert sum(before.values()) == 40


def test_editing_the_marker_line_changes_its_key():
    """BOUNDARY SCENARIO 3 — a touch is visible, and editing only the CODE is
    a touch.

    This is D7's whole conversion mechanism: a grandfathered marker stays
    grandfathered until somebody edits the line it sits on, at which point it
    becomes a new key and needs a disposition.
    """
    original = _keys('value = call()  # type: ignore[arg-type]')
    recoded = _keys('value = call()  # type: ignore[attr-defined]')
    renamed = _keys('other = call()  # type: ignore[arg-type]')

    assert original != recoded
    assert original != renamed


def test_editing_an_unrelated_line_of_the_same_statement_does_not():
    """The stated LIMIT of "touched", rather than a hidden one.

    The key digests one physical line, so a multi-line call whose marker rides
    on the closing line is untouched by an edit to its first argument.  D7
    records this as a cost accepted knowingly, so it belongs in a test rather
    than in a reader's discovery.
    """
    before = _keys('value = call(\n    first,\n)  # type: ignore[arg-type]\n')
    after = _keys('value = call(\n    SECOND,\n)  # type: ignore[arg-type]\n')

    assert before == after


def test_reindenting_the_marker_line_does_not_change_its_key():
    """The digest is of the STRIPPED line — moving a statement into an ``if``
    is not an edit to the suppression."""
    flush = _keys('value = call()  # type: ignore[arg-type]')
    indented = _keys('if True:\n        value = call()  # type: ignore[arg-type]\n')

    assert flush == indented


def test_codes_are_sorted_at_construction():
    """``[b, a]`` and ``[a, b]`` are the same suppression.

    Asserted on the constructor rather than on two source lines, because two
    source lines spelling the codes in different orders differ in their DIGEST
    too — which would make the test pass for the wrong reason.
    """
    digest = 'abcdef012345'
    kind = inline_suppressions.Kind.TYPE_IGNORE

    one = inline_suppressions.SuppressionKey(kind=kind, codes=('b', 'a'), digest=digest)
    other = inline_suppressions.SuppressionKey(kind=kind, codes=('a', 'b'), digest=digest)

    assert one == other
    assert one.codes == ('a', 'b')


def test_two_kinds_on_one_line_are_two_distinct_keys():
    """Same digest, different kind — so a line silencing two tools owes two
    dispositions' worth of accounting, not one."""
    first, second = _keys('a = 1  # type: ignore[arg-type]  # noqa: E402')

    assert first != second
    assert first.digest == second.digest
    assert {first.kind, second.kind} == {
        inline_suppressions.Kind.TYPE_IGNORE,
        inline_suppressions.Kind.NOQA,
    }


def test_a_rendered_key_is_a_string_and_the_interface_offers_no_way_back():
    """``render()`` exists because JSON object keys are strings by the
    format's definition; a reader that parsed one back would not.

    ``shared.ratchet`` treats every key as an opaque identity token, so an
    inverse would be exactly the ad-hoc parser of an internal value heuristic
    12 forbids — and it would quietly make the rendering a wire format that
    could never be changed again.  Pinned as an INTERFACE assertion: the public
    surface of the key is its three fields and ``render``, nothing else.
    """
    key = inline_suppressions.key_for(_sites('a = 1  # noqa: E402')[0])

    assert isinstance(key.render(), str)
    assert {name for name in dir(key) if not name.startswith('_')} == {
        'kind',
        'codes',
        'digest',
        'render',
    }


# ---------------------------------------------------------------------------
# D8 — the consumer model: which tool, if any, actually reads this marker.

#: What all eight of this repository's pyproject.toml files declare, verbatim.
_RUFF_CONFIG = '[tool.ruff.lint]\nselect = ["E", "F", "UP", "B", "SIM", "I"]\nignore = ["E501"]\n'


def _consumer_of(
    root: Path,
    *,
    path: str = 'pkg/mod.py',
    kind=None,
    codes: tuple[str, ...] = (),
):
    """The consumer *root*'s model resolves for one synthetic site."""
    site = inline_suppressions.Site(
        path=path,
        line=1,
        kind=kind if kind is not None else inline_suppressions.Kind.NOQA,
        codes=codes,
        text='x = 1',
    )
    return inline_suppressions.ConsumerModel(root).consumer_for(site)


def test_type_ignore_and_pyright_ignore_resolve_to_pyright_whatever_the_code(tmp_path: Path):
    """No config is consulted for these two: pyright runs over every package as
    a declared gate, so the marker is read wherever it sits."""
    _write_files(tmp_path, {'pkg/mod.py': 'x = 1\n'})

    for kind in (
        inline_suppressions.Kind.TYPE_IGNORE,
        inline_suppressions.Kind.PYRIGHT_IGNORE,
    ):
        for codes in ((), ('arg-type',), ('reportArgumentType',), ('not-a-real-code',)):
            assert _consumer_of(tmp_path, kind=kind, codes=codes) is (
                inline_suppressions.Consumer.PYRIGHT
            ), (kind, codes)


def test_pragma_no_cover_and_nosec_resolve_to_no_consumer(tmp_path: Path):
    """D8's finding: nothing in this repository reads either one today.

    No coverage gate runs ``# pragma: no cover``, and bandit is not installed —
    the live ``nosec`` count is zero, which is what makes that kind's row a
    statement about tools rather than about code.
    """
    _write_files(tmp_path, {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': _RUFF_CONFIG})

    for kind in (
        inline_suppressions.Kind.PRAGMA_NO_COVER,
        inline_suppressions.Kind.NOSEC,
    ):
        assert _consumer_of(tmp_path, kind=kind) is inline_suppressions.Consumer.NONE, kind


def test_a_noqa_code_the_nearest_config_selects_is_consumed_by_ruff(tmp_path: Path):
    _write_files(tmp_path, {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': _RUFF_CONFIG})

    for code in ('E402', 'F401', 'B006', 'SIM102', 'I001', 'UP038'):
        assert _consumer_of(tmp_path, codes=(code,)) is inline_suppressions.Consumer.RUFF, code


def test_a_noqa_code_the_nearest_config_does_not_select_has_no_consumer(tmp_path: Path):
    """D8's whole point, and its largest single inflow.

    ``PLC0415`` alone accounts for 963 of this tree's markers and no
    ``pyproject.toml`` here selects ``PL``; every one of them is dead.
    Reporting them as ruff-consumed would leave the biggest source of new
    markers entirely unpoliced.
    """
    _write_files(tmp_path, {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': _RUFF_CONFIG})

    for code in ('PLC0415', 'ANN001', 'ARG002', 'A002', 'N802'):
        assert _consumer_of(tmp_path, codes=(code,)) is inline_suppressions.Consumer.NONE, code


def test_a_selector_matches_by_linter_and_number_never_by_string_prefix(tmp_path: Path):
    """THE LOAD-BEARING CASE — ``select = ["B"]`` does NOT select ``BLE001``.

    Measured against real ruff, not assumed: ``ruff check --isolated --select
    B`` does not flag ``BLE001`` while ``--select BLE`` does.  A selector
    resolves to a (linter, code-prefix) PAIR, so linter ``B``
    (flake8-bugbear) never reaches linter ``BLE`` (flake8-blind-except).  This
    tree carries 202 ``# noqa: BLE001`` markers — the second-largest
    population — and a naive ``code.startswith(selector)`` would silently
    report every one of them as ruff-consumed, defeating D8 for them.

    Parsing both sides at the boundary into a typed ``(linter, number)`` pair
    is also exactly what heuristic 12 prescribes, so the correct behaviour and
    the cited heuristic coincide here.
    """
    _write_files(
        tmp_path,
        {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': '[tool.ruff.lint]\nselect = ["B"]\n'},
    )

    assert _consumer_of(tmp_path, codes=('B006',)) is inline_suppressions.Consumer.RUFF
    assert _consumer_of(tmp_path, codes=('BLE001',)) is inline_suppressions.Consumer.NONE


def test_a_partial_selector_matches_on_the_number_prefix(tmp_path: Path):
    """Within one linter, a selector IS a numeric prefix: ``E4`` selects E402
    and ``E5`` does not."""
    _write_files(
        tmp_path,
        {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': '[tool.ruff.lint]\nselect = ["E4"]\n'},
    )

    assert _consumer_of(tmp_path, codes=('E402',)) is inline_suppressions.Consumer.RUFF
    assert _consumer_of(tmp_path, codes=('E501',)) is inline_suppressions.Consumer.NONE


def test_an_ignored_code_has_no_consumer_even_though_a_selector_matches(tmp_path: Path):
    """All eight pyprojects here set ``ignore = ["E501"]``, so ruff provably
    never emits E501 and every ``# noqa: E501`` in the tree is dead.  Reading
    ``ignore`` as well as ``select`` is the same tomllib read and is strictly
    more honest."""
    _write_files(tmp_path, {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': _RUFF_CONFIG})

    assert _consumer_of(tmp_path, codes=('E501',)) is inline_suppressions.Consumer.NONE


def test_a_bare_noqa_is_consumed_when_the_nearest_config_selects_anything(tmp_path: Path):
    """A bare ``# noqa`` silences whatever ruff would have said, so it is
    consumed exactly when ruff has something to say at all."""
    _write_files(
        tmp_path,
        {
            'pkg/mod.py': 'x = 1\n',
            'pyproject.toml': _RUFF_CONFIG,
            'bare/mod.py': 'x = 1\n',
            'bare/pyproject.toml': '[tool.ruff.lint]\nselect = []\n',
        },
    )

    assert _consumer_of(tmp_path) is inline_suppressions.Consumer.RUFF
    assert _consumer_of(tmp_path, path='bare/mod.py') is inline_suppressions.Consumer.NONE


def test_the_nearest_pyproject_wins_and_is_not_merged_with_the_root(tmp_path: Path):
    """Ruff takes the NEAREST applicable config without merging — stated in
    this repository's own root ``pyproject.toml``, and modelled here."""
    _write_files(
        tmp_path,
        {
            'pyproject.toml': _RUFF_CONFIG,
            'member/pyproject.toml': '[tool.ruff.lint]\nselect = ["ANN"]\n',
            'member/mod.py': 'x = 1\n',
            'top.py': 'x = 1\n',
        },
    )

    assert _consumer_of(tmp_path, path='member/mod.py', codes=('ANN001',)) is (
        inline_suppressions.Consumer.RUFF
    )
    assert _consumer_of(tmp_path, path='member/mod.py', codes=('E402',)) is (
        inline_suppressions.Consumer.NONE
    )
    assert _consumer_of(tmp_path, path='top.py', codes=('E402',)) is (
        inline_suppressions.Consumer.RUFF
    )


def test_a_pyproject_with_no_ruff_section_is_skipped_and_the_walk_continues(tmp_path: Path):
    """Ruff skips a ``pyproject.toml`` carrying no ``[tool.ruff]`` at all, so a
    packaging-only manifest must not shadow the config above it."""
    _write_files(
        tmp_path,
        {
            'pyproject.toml': _RUFF_CONFIG,
            'member/pyproject.toml': '[project]\nname = "member"\nversion = "0"\n',
            'member/mod.py': 'x = 1\n',
        },
    )

    assert _consumer_of(tmp_path, path='member/mod.py', codes=('E402',)) is (
        inline_suppressions.Consumer.RUFF
    )


def test_a_file_with_no_pyproject_above_it_has_no_ruff_consumer(tmp_path: Path):
    """The walk stops at the scan ROOT, never climbing out of the tree under
    measurement — otherwise a scan of a fixture tree would silently read this
    repository's own config."""
    _write_files(tmp_path, {'pkg/mod.py': 'x = 1\n'})

    assert _consumer_of(tmp_path, codes=('E402',)) is inline_suppressions.Consumer.NONE


def test_a_config_key_that_could_widen_the_selected_set_is_an_instrument_failure(
    tmp_path: Path,
):
    """THE SPLIT IS BY DIRECTION OF ERROR, which is the only thing that matters
    for a gate.

    Under-reading the selected set makes the scanner reject a marker ruff
    genuinely honours — a false red on a legitimate suppression, the expensive
    failure — so it refuses to guess rather than proceeding on a config it does
    not fully model.  Exit 2, naming the file AND the key.
    """
    _write_files(
        tmp_path,
        {
            'pkg/mod.py': 'x = 1\n',
            'pyproject.toml': '[tool.ruff.lint]\nselect = ["E"]\nextend-select = ["ANN"]\n',
        },
    )

    with pytest.raises(inline_suppressions.InstrumentFailure) as caught:
        _consumer_of(tmp_path, codes=('E402',))

    assert 'pyproject.toml' in str(caught.value)
    assert 'extend-select' in str(caught.value)


def test_config_keys_that_only_ever_subtract_are_tolerated(tmp_path: Path):
    """Over-reading the selected set only grandfathers a dead marker — the
    cheap failure — so it is tolerated with the reason recorded rather than
    modelled.

    ``per-file-ignores`` is the concrete case: present only in
    ``orchestrator/pyproject.toml`` (``tests/**/*.py: ["F811"]``), affecting 13
    grandfathered markers, and modelling it would need path-glob machinery for
    no change in the direction that can hurt.
    """
    _write_files(
        tmp_path,
        {
            'pkg/mod.py': 'x = 1\n',
            'pyproject.toml': (
                '[tool.ruff.lint]\n'
                'select = ["E", "F"]\n'
                'extend-ignore = ["E731"]\n'
                '[tool.ruff.lint.per-file-ignores]\n'
                '"tests/**/*.py" = ["F811"]\n'
            ),
        },
    )

    assert _consumer_of(tmp_path, codes=('E402',)) is inline_suppressions.Consumer.RUFF


def test_a_ruff_section_that_declares_no_select_is_an_instrument_failure(tmp_path: Path):
    """The same direction-of-error rule as ``extend-select``, applied to an
    ABSENT key rather than an unmodelled one.

    A ``[tool.ruff]`` section with no ``select`` anywhere does not mean "ruff
    checks nothing" — it means ruff applies its BUILT-IN default rule set,
    which is wider than the nothing this model would otherwise infer and which
    drifts with the ruff version.  Reading it as empty is precisely the
    under-read that rejects a marker ruff genuinely honours.  This repository's
    own root ``pyproject.toml`` records having been in that state, with two
    gated directories reporting "All checks passed!" while running a rule set
    nobody chose.

    Nothing in this tree reaches the refusal: all eight pyprojects declare
    ``select`` explicitly.
    """
    _write_files(
        tmp_path,
        {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': '[tool.ruff]\nline-length = 100\n'},
    )

    with pytest.raises(inline_suppressions.InstrumentFailure) as caught:
        _consumer_of(tmp_path, codes=('E402',))

    assert 'pyproject.toml' in str(caught.value)
    assert 'select' in str(caught.value)


# ---------------------------------------------------------------------------
# D8, first-party half — the codes no ruff selector can ever match.

#: The checker whose private rule-code constants this scanner necessarily
#: copies.  Its path is named once here, and the guard below runs it.
_FIRST_PARTY_CHECKER = REPO_ROOT / 'fused-memory' / 'scripts' / 'check_bare_magicmock_config.py'

#: One fixture per rule, each written to trigger exactly one of them, so the
#: guard below sees every code the checker can name.  Kept as SOURCE rather
#: than as a list of expected codes: the point is to read the codes back out of
#: what the checker emits, never to restate them.
_FIRST_PARTY_TRIGGERS: Mapping[str, str] = {
    'rule_a.py': 'from unittest.mock import MagicMock\n\nconfig = MagicMock()\n',
    'rule_b.py': (
        'from unittest.mock import MagicMock\n\ndouble = MagicMock(passed=True, summary=\'ok\')\n'
    ),
    'rule_c.py': (
        'import asyncio\n\n\nasync def run(req):\n'
        '    return await asyncio.wait_for(req.result, timeout=5)\n'
    ),
}


def test_a_first_party_code_resolves_to_the_first_party_consumer(tmp_path: Path):
    """These are not ruff codes at all, so no selector can ever match them —
    and the ``(linter, number)`` split must tolerate a kebab-case code, which
    has no numeric run."""
    _write_files(tmp_path, {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': _RUFF_CONFIG})

    for code in inline_suppressions.FIRST_PARTY_CODES:
        assert _consumer_of(tmp_path, codes=(code,)) is (
            inline_suppressions.Consumer.FIRST_PARTY
        ), code


def test_an_invented_neighbouring_code_has_no_consumer(tmp_path: Path):
    """The table is a closed set, not a kebab-case shape test: a code that
    merely LOOKS like a first-party one is read by nobody."""
    _write_files(tmp_path, {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': _RUFF_CONFIG})

    assert _consumer_of(tmp_path, codes=('bare-something-else',)) is (
        inline_suppressions.Consumer.NONE
    )


def test_the_first_party_table_matches_the_codes_that_checker_actually_emits(tmp_path: Path):
    """THE DRIFT GUARD, and it is BEHAVIOURAL IN BOTH DIRECTIONS.

    The source of truth is a private constant in another package's
    stdlib-only script, so there is no public seam to import and a second copy
    is unavoidable.  Heuristic 11 then asks that the copy's drift be made loud
    rather than that the copy be hidden.

    It asserts on the violation messages the checker EMITS, never on its
    private ``_RULE_A_CODE`` / ``_RULE_B_CODE`` / ``_RULE_C_CODE`` — reading
    those would pin implementation rather than behaviour
    (``docs/code-quality.md``, Tests stance), and the emitted message is
    strictly the stronger assertion: a checker that renamed its constant while
    still emitting the old code would remain correctly modelled.

    SET EQUALITY, not containment, and that direction is load-bearing.  A
    containment check ("every code in our table is real") passes happily while
    the table MISSES a code the checker honours — which is exactly the defect
    this test found: ``wall-clock-deadline`` (Rule C, task 4246) is a live
    first-party code that both the PRD's D8 prose and this task's plan omit.
    Missing a code is the expensive direction: the scanner would tell an author
    to delete a marker a live checker genuinely reads.
    """
    _write_files(tmp_path, _FIRST_PARTY_TRIGGERS)

    emitted = set()
    for name in _FIRST_PARTY_TRIGGERS:
        completed = _run_script(
            [str(tmp_path / name)], cwd=tmp_path, script=_FIRST_PARTY_CHECKER
        )
        emitted.update(re.findall(r'#\s*noqa:\s*([a-z0-9]+(?:-[a-z0-9]+)+)', completed.stdout))

    assert emitted, 'the checker emitted no suppression remedy at all — fixtures stopped triggering'
    assert emitted == set(inline_suppressions.FIRST_PARTY_CODES)


# ---------------------------------------------------------------------------
# Layer 3 — classification: owned, ratified by class, or unowned.


def _classify(tmp_path: Path, files: Mapping[str, str]):
    """Scan and classify a fixture tree in one step."""
    _write_fixture_tree(tmp_path, files)
    scan = inline_suppressions.scan_tree(tmp_path)
    return inline_suppressions.classify(scan, inline_suppressions.ConsumerModel(tmp_path))


def test_the_shipped_class_table_is_empty(tmp_path: Path):
    """D9 — the valve is the OPERATOR's, and it ships shut.

    The rows are ruled by δ and applied by ζb; an implementer adding one here
    would be ratifying a blanket exception on the operator's behalf, which is
    the one thing D9 reserves.
    """
    del tmp_path
    assert dict(inline_suppressions.RATIFIED_SUPPRESSION_CLASSES) == {}


def test_a_site_with_a_well_formed_disposition_is_owned(tmp_path: Path):
    """BOUNDARY SCENARIO 2 — all three of D6's forms, and none contributes a
    key to the unowned multiset."""
    result = _classify(
        tmp_path,
        {
            'pyproject.toml': _RUFF_CONFIG,
            'm.py': (
                'a = 1  # noqa: E402  # debt: task 5601\n'
                'b = 2  # noqa: F401  # debt: ticket tkt_0RTCC80EM92A7WD08D6RF6ZZPY\n'
                'c = 3  # noqa: B006  # ratified: inv12-day-one-test-doubles\n'
            ),
        },
    )

    assert result.violations == ()
    assert result.counts == {}
    assert [entry.ownership for entry in result.classified] == [
        inline_suppressions.Ownership.DEBT,
        inline_suppressions.Ownership.DEBT,
        inline_suppressions.Ownership.POLICY,
    ]


def test_a_site_with_no_disposition_is_unowned_and_contributes_a_key(tmp_path: Path):
    result = _classify(
        tmp_path, {'pyproject.toml': _RUFF_CONFIG, 'm.py': 'a = 1  # noqa: E402\n'}
    )

    assert [entry.ownership for entry in result.classified] == [
        inline_suppressions.Ownership.UNOWNED
    ]
    assert sum(result.counts.values()) == 1


def test_a_marker_that_does_not_parse_is_a_violation_naming_its_comment(tmp_path: Path):
    """BOUNDARY SCENARIO 7, first half.

    Present-but-broken is the case worth being loud about: the author plainly
    meant to disposition something, and the entry is silently undisposed until
    somebody is told.  Exit 1 and not 2 — it is a fault at the SITE, fixed by
    an agent, which is what ``MalformedDisposition``'s own docstring records.
    """
    result = _classify(
        tmp_path,
        {'pyproject.toml': _RUFF_CONFIG, 'm.py': 'a = 1  # noqa: E402  # debt: soon\n'},
    )

    (violation,) = result.violations
    assert violation.path == 'm.py'
    assert violation.line == 1
    rendered = violation.render()
    for form in INLINE_MARKER_FORMS:
        assert form in rendered


def test_a_disposition_on_a_line_with_no_suppression_is_a_violation(tmp_path: Path):
    """BOUNDARY SCENARIO 7, second half — and it CANNOT come from the parser.

    ``parse_disposition_marker`` returns a perfectly valid ``Debt(TaskRef(5))``
    here; its docstring says so outright and delegates this violation to the
    scanner, because deciding it needs the kind table.  So the detection is
    "this comment yielded a Disposition and no Site", never a parser outcome.
    """
    result = _classify(
        tmp_path, {'pyproject.toml': _RUFF_CONFIG, 'm.py': 'x = 1  # debt: task 5\n'}
    )

    (violation,) = result.violations
    assert violation.path == 'm.py'
    assert violation.line == 1
    assert result.counts == {}


def test_a_site_matching_a_ratified_class_is_policy_by_reference(tmp_path: Path, monkeypatch):
    """BOUNDARY SCENARIO 10 — outside the unowned multiset, and COUNTED under
    its class so blanket policy stays visible in the report."""
    row = inline_suppressions.SuppressionClass(
        kind=inline_suppressions.Kind.NOQA,
        code='E402',
        scope=inline_suppressions.Scope.ANY,
    )
    monkeypatch.setattr(
        inline_suppressions, 'RATIFIED_SUPPRESSION_CLASSES', {row: Policy('inv12-day-one')}
    )

    result = _classify(
        tmp_path, {'pyproject.toml': _RUFF_CONFIG, 'm.py': 'a = 1  # noqa: E402\n'}
    )

    (entry,) = result.classified
    assert entry.ownership is inline_suppressions.Ownership.CLASS
    assert entry.suppression_class == row
    assert result.counts == {}


def test_class_scope_matches_src_tests_or_any(tmp_path: Path, monkeypatch):
    """A path is ``tests`` iff one of its COMPONENTS is ``tests``.

    Verified complete for this repository: every tracked test module lives
    under a ``tests`` component, so the rule needs no filename pattern beside
    it.  ``src`` is then simply "not tests", which keeps the two scopes a
    partition rather than two independent predicates that could both miss.
    """
    files = {
        'pyproject.toml': _RUFF_CONFIG,
        'pkg/mod.py': 'a = 1  # noqa: E402\n',
        'pkg/tests/test_mod.py': 'b = 2  # noqa: E402\n',
    }
    for scope, expected in (
        (inline_suppressions.Scope.ANY, {'pkg/mod.py', 'pkg/tests/test_mod.py'}),
        (inline_suppressions.Scope.SRC, {'pkg/mod.py'}),
        (inline_suppressions.Scope.TESTS, {'pkg/tests/test_mod.py'}),
    ):
        row = inline_suppressions.SuppressionClass(
            kind=inline_suppressions.Kind.NOQA, code='E402', scope=scope
        )
        monkeypatch.setattr(
            inline_suppressions, 'RATIFIED_SUPPRESSION_CLASSES', {row: Policy('inv12-day-one')}
        )
        tree = tmp_path / scope.value
        tree.mkdir()
        result = _classify(tree, files)

        covered = {
            entry.site.path
            for entry in result.classified
            if entry.ownership is inline_suppressions.Ownership.CLASS
        }
        assert covered == expected, scope


def test_a_suppression_class_renders_d9s_published_key():
    row = inline_suppressions.SuppressionClass(
        kind=inline_suppressions.Kind.NOQA,
        code='E402',
        scope=inline_suppressions.Scope.TESTS,
    )

    assert row.render() == 'noqa[E402]@tests'
