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

import json
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
from shared.ratchet import BASELINE_README, SCHEMA_VERSION, load

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

    ``PYTHONPATH`` is emptied and ``PYTHONNOUSERSITE`` is set — and NEITHER IS
    SUFFICIENT, which is why :data:`_NO_SITE` exists beside this.  Measured from
    ``/tmp`` with both applied: ``import shared`` still resolved, to
    ``<worktree>/shared/src/shared/__init__.py``, because the workspace member is
    installed EDITABLE and the ``.pth`` entry that does it lives in the venv's own
    ``site-packages``, which neither knob touches.
    """
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}
    env['PYTHONNOUSERSITE'] = '1'
    return env


#: The interpreter flag that actually hides an editable install: ``-S`` skips
#: site processing altogether, so no ``.pth`` file is read.  Measured: the same
#: subprocess is rc=0 ``RESOLVED`` without it and rc=1
#: ``ModuleNotFoundError: No module named 'shared'`` with it.  The scanner only
#: ever imports stdlib plus ``shared``, so nothing else is lost.
_NO_SITE = ('-S',)


def _run_script(
    args: list[str],
    *,
    cwd: Path,
    script: Path,
    env: dict[str, str] | None = None,
    flags: tuple[str, ...] = (),
):
    """Run *script* as a real subprocess with ``sys.executable`` and *flags*."""
    return subprocess.run(
        [sys.executable, *flags, str(script), *args],
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

    THE PREFILTER'S REACH BOUNDS THIS FINDING, and the bound is asserted rather
    than left to be discovered.  A file carrying no marker byte substring at
    all is never decoded, so a stray disposition alone in such a file is
    invisible.  That is the honest limit and it lands on the harmless side: in
    a file with no suppressions there is nothing anywhere for the marker to
    answer for, so it is inert prose.  The dangerous case — an author who
    believes a REAL marker in this file is now dispositioned when it is not —
    is exactly the case that IS caught, because that file carries a marker.
    """
    result = _classify(
        tmp_path,
        {
            'pyproject.toml': _RUFF_CONFIG,
            'm.py': 'a = 1  # noqa: E402  # debt: task 5601\nx = 2  # debt: task 5\n',
            'markerless.py': 'y = 3  # debt: task 7\n',
        },
    )

    (violation,) = result.violations
    assert violation.path == 'm.py'
    assert violation.line == 2
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


# ---------------------------------------------------------------------------
# Layer 4 — the ratchet and the verbs.


def _revise(root: Path, files: Mapping[str, str], *, removing: tuple[str, ...] = ()) -> None:
    """Evolve an already-tracked fixture tree and re-stage the result.

    The ratchet's whole subject matter is a tree that CHANGED after a baseline
    was seeded, so every scenario from here down needs a second edit against a
    live index.  ``git add -A`` stages deletions too, which is what makes the
    rename half of scenario 4 a real rename rather than a copy.
    """
    for relative in removing:
        (root / relative).unlink()
    _write_files(root, files)
    _run_git(['add', '-A', '-f'], cwd=root)


def _grandfathered(count: int) -> str:
    """*count* distinct lines, each carrying one undisposed suppression.

    Distinct rather than identical on purpose: identical lines collapse to one
    D7 key with a multiplicity, which is a different scenario (5 and 9) from
    the many-keys-survive-a-rename one this feeds.
    """
    return ''.join(f'a{index} = {index}  # type: ignore[arg-type]\n' for index in range(count))


def test_a_new_undisposed_suppression_is_one_violation_line(tmp_path: Path, capsys):
    """BOUNDARY SCENARIO 1 — the gate's headline case.

    The line has to carry everything an agent needs to act without opening the
    PRD: where it is, what kind it is, its codes, why it is a finding, and how
    to spell the fix.  The accepted forms are asserted against
    ``INLINE_MARKER_FORMS`` itself rather than against retyped strings, so a
    later edit to the published grammar cannot leave this message behind.
    """
    baseline = _write_fixture_tree(tmp_path, {'m.py': 'a = 1\n'}, baseline=True)
    _revise(tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\n'})
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 1

    (violation,) = capsys.readouterr().err.strip().splitlines()
    assert violation.startswith('m.py:1:')
    assert 'type: ignore' in violation
    assert 'arg-type' in violation
    for form in INLINE_MARKER_FORMS:
        assert form in violation


def test_the_same_suppression_with_a_disposition_is_green(tmp_path: Path):
    """BOUNDARY SCENARIO 2 — the escape hatch the gate exists to push authors
    towards, asserted on the very marker scenario 1 rejects."""
    baseline = _write_fixture_tree(tmp_path, {'m.py': 'a = 1\n'}, baseline=True)
    _revise(tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]  # debt: task 5601\n'})

    assert _check(tmp_path, baseline) == 0


def test_editing_a_grandfathered_markers_code_is_a_new_violation(tmp_path: Path, capsys):
    """BOUNDARY SCENARIO 3 — D7's conversion mechanism, at the gate.

    The marker is not new and the file is not new; only the CODE on the line
    changed, which is exactly the edit that ought to make an author own what
    they are silencing.  The digest is of the whole stripped line, so the old
    key stops being claimed and the new one is in excess.
    """
    baseline = _write_fixture_tree(
        tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\n'}, baseline=True
    )
    _revise(tmp_path, {'m.py': 'a = 1  # type: ignore[attr-defined]\n'})
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 1
    assert 'attr-defined' in capsys.readouterr().err


def test_renaming_and_splitting_a_file_of_grandfathered_markers_is_green(tmp_path: Path):
    """BOUNDARY SCENARIO 4 — the pathless key, end to end.

    Forty grandfathered markers move to two new filenames and the gate does not
    notice, because no key mentions a path.  This is the property that keeps an
    ordinary refactor out of the baseline: a key set that moved with the code
    needs no diff to the committed file at all.
    """
    lines = _grandfathered(40).splitlines(keepends=True)
    baseline = _write_fixture_tree(tmp_path, {'big.py': ''.join(lines)}, baseline=True)
    _revise(
        tmp_path,
        {'moved/first.py': ''.join(lines[:17]), 'moved/second.py': ''.join(lines[17:])},
        removing=('big.py',),
    )

    assert _check(tmp_path, baseline) == 0


def test_several_new_suppressions_print_one_violation_each(tmp_path: Path, capsys):
    baseline = _write_fixture_tree(tmp_path, {'m.py': 'a = 1\n'}, baseline=True)
    _revise(
        tmp_path,
        {
            'm.py': 'a = 1  # type: ignore[arg-type]\n',
            'n.py': 'b = 2  # pyright: ignore[reportArgumentType]\n',
        },
    )
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 1

    lines = capsys.readouterr().err.strip().splitlines()
    assert len(lines) == 2
    assert {line.split(':')[0] for line in lines} == {'m.py', 'n.py'}


def test_a_clean_whole_tree_run_labels_its_green_clean(tmp_path: Path, capsys):
    """Three greens exist and they are not interchangeable, so the report says
    which one this is.  ``clean`` is the only one that means the whole tree was
    measured against a real baseline and nothing was in excess."""
    baseline = _write_fixture_tree(
        tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\n'}, baseline=True
    )
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 0

    report = capsys.readouterr().out
    assert 'clean' in report
    assert 'partial' not in report


def _dead_marker_violation(tmp_path: Path, capsys, source: str) -> str:
    """Seed a marker-free tree, add *source*, and return its one violation line."""
    baseline = _write_fixture_tree(
        tmp_path, {'pyproject.toml': _RUFF_CONFIG, 'm.py': 'a = 1\n'}, baseline=True
    )
    _revise(tmp_path, {'m.py': source})
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 1

    (violation,) = capsys.readouterr().err.strip().splitlines()
    return violation


@pytest.mark.parametrize(
    'source',
    [
        'a = 1  # noqa: PLC0415\n',
        'a = 1  # pragma: no cover\n',
    ],
)
def test_a_new_marker_no_tool_reads_is_rejected_for_deletion(
    tmp_path: Path, capsys, source: str
):
    """BOUNDARY SCENARIO 9 — D8's rejection arm.

    Both halves of the population D8 names: a ``noqa`` whose code the nearest
    config does not select (PLC0415 is the tree's largest such inflow) and a kind
    no configured tool reads at all.  The remedy is DELETION, so the line says
    so and publishes no disposition forms — offering an author a fix that does
    not work is worse than offering none.
    """
    violation = _dead_marker_violation(tmp_path, capsys, source)

    assert 'delete this marker' in violation
    assert 'no tool reads it' in violation
    for form in INLINE_MARKER_FORMS:
        assert form not in violation


@pytest.mark.parametrize(
    'source',
    [
        'a = 1  # noqa: PLC0415  # debt: task 5601\n',
        'a = 1  # pragma: no cover  # debt: task 5601\n',
    ],
)
def test_a_disposition_does_not_rescue_a_marker_no_tool_reads(
    tmp_path: Path, capsys, source: str
):
    """BOUNDARY SCENARIO 9's sharp edge — D8 accepts NO disposition.

    The exit code and the reason are both unchanged from the undispositioned
    case, which is the whole content of the claim: a dead marker is not debt to
    be owned, it is a line to be removed, and an author who dispositions one has
    answered a question nobody asked.  This falls out of the fixed
    consumer-first classification order rather than from a special case.
    """
    violation = _dead_marker_violation(tmp_path, capsys, source)

    assert 'delete this marker' in violation
    assert 'no tool reads it' in violation


def test_a_ratified_class_does_not_rescue_a_marker_no_tool_reads(
    tmp_path: Path, capsys, monkeypatch
):
    """BOUNDARY SCENARIO 9 against D9's valve, which is the other thing that
    could plausibly rescue a site and equally does not.

    The operator's valve rules on suppressions a tool HONOURS; a row matching a
    marker nothing reads would ratify a no-op, so the consumer check running
    first makes the row inert rather than making it a widening.
    """
    monkeypatch.setattr(
        inline_suppressions,
        'RATIFIED_SUPPRESSION_CLASSES',
        {
            inline_suppressions.SuppressionClass(
                kind=inline_suppressions.Kind.NOQA,
                code='PLC0415',
                scope=inline_suppressions.Scope.ANY,
            ): Policy('inv12-day-one')
        },
    )

    violation = _dead_marker_violation(tmp_path, capsys, 'a = 1  # noqa: PLC0415\n')

    assert 'no tool reads it' in violation


def test_a_grandfathered_dead_marker_is_green_and_still_counted(
    tmp_path: Path, capsys
):
    """D8's "grandfathered dead markers stay counted" — the complement that
    keeps the gate a RATCHET rather than a sweep.

    Rejecting every dead marker outright would red the whole tree on the day the
    gate lands, so the baseline holds the existing ones.  What must NOT happen is
    that they vanish from the report: the population is the thing D11's sweep
    tightens, and a number nobody can see never shrinks.
    """
    baseline = _write_fixture_tree(
        tmp_path,
        {'pyproject.toml': _RUFF_CONFIG, 'm.py': 'a = 1  # noqa: PLC0415\n'},
        baseline=True,
    )
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 0

    report = capsys.readouterr().out
    assert '1 unowned in 1 key' in report


def test_a_second_copy_of_a_grandfathered_dead_line_is_in_excess(tmp_path: Path, capsys):
    """The multiset is a multiset: the baseline permits ONE of that line.

    Identical lines share one D7 key, so the only thing separating the
    grandfathered copy from a new one is multiplicity — which is exactly what
    ``excess`` measures, and why the key carries a count rather than a flag.
    """
    baseline = _write_fixture_tree(
        tmp_path,
        {'pyproject.toml': _RUFF_CONFIG, 'm.py': 'a = 1  # noqa: PLC0415\n'},
        baseline=True,
    )
    _revise(tmp_path, {'m.py': 'a = 1  # noqa: PLC0415\na = 1  # noqa: PLC0415\n'})
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 1

    lines = capsys.readouterr().err.strip().splitlines()
    assert [line.split(':')[1] for line in lines] == ['1', '2']


def test_with_no_baseline_yet_a_whole_tree_of_undisposed_markers_is_advisory(
    tmp_path: Path, capsys
):
    """BOUNDARY SCENARIO 11 — the pre-κ1 state, which is every tree until the
    cutover lands.

    D12 makes baseline ABSENCE a legitimate state, so the gate is green and says
    it is enforcing nothing.  The label has to be its own word rather than a
    silent ``clean``: a reader who sees ``clean`` over a tree of undisposed
    markers concludes the scanner is broken, and a reader who sees nothing at all
    concludes the gate is live when it is not.  The message names the step that
    seeds the baseline, so the next question is answered in the same line.
    """
    baseline = _write_fixture_tree(tmp_path, {'m.py': _grandfathered(5)})
    capsys.readouterr()

    assert not baseline.exists()
    assert _check(tmp_path, baseline) == 0

    report = capsys.readouterr().out
    assert 'advisory' in report
    assert 'κ1' in report


def test_a_scoped_check_over_a_clean_scope_is_partial_and_ignores_the_rest(
    tmp_path: Path, capsys
):
    """The label exists because a scoped green is a WEAKER claim, and the one
    place that matters is the finding it did not look for.

    Scoped ``--check`` is sound in one direction only, which is why D12 keeps it:
    dropping sites can lower a key's current count and so only ever UNDER-reports
    excess — it can never manufacture a violation.  Slack is the mirror and is
    therefore UNSOUND from a partial view (every unscanned baseline key reads as
    headroom), so the report declines to put a number on it rather than printing
    one that is wrong.
    """
    baseline = _write_fixture_tree(
        tmp_path, {'pkg/kept.py': 'a = 1  # type: ignore[arg-type]\n'}, baseline=True
    )
    _revise(tmp_path, {'other/fresh.py': 'b = 2  # type: ignore[attr-defined]\n'})
    capsys.readouterr()

    assert _check(tmp_path, baseline, 'pkg') == 0

    report = capsys.readouterr()
    assert 'partial' in report.out
    assert 'clean' not in report.out
    assert 'slack n/a' in report.out
    assert report.err == ''

    assert _check(tmp_path, baseline) == 1


@pytest.mark.parametrize('damage', ['truncated', 'not-json', 'wrong-schema'])
def test_a_baseline_that_exists_but_cannot_be_read_is_never_green(
    tmp_path: Path, capsys, damage: str
):
    """THE ANTI-FAIL-SOFT CASE, and the whole reason absence is detected by an
    explicit existence check instead of by catching ``BaselineUnusable``.

    ``shared.ratchet.load`` collapses absent, undecodable, unparseable, misshapen
    and wrong-schema into ONE refusal, because to the kernel's callers they mean
    one thing.  This consumer is the one place where they do not: absence is a
    legitimate pre-κ1 state and everything else is a broken instrument.  Reaching
    the advisory path by catching that refusal would report a corrupt baseline as
    a clean tree — exactly the silent fail-soft the kernel's own docstring says an
    empty baseline causes (INV-11).
    """
    baseline = _write_fixture_tree(
        tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\n'}, baseline=True
    )
    seeded = baseline.read_text(encoding='utf-8')
    if damage == 'truncated':
        baseline.write_text(seeded[: len(seeded) // 2], encoding='utf-8')
    elif damage == 'not-json':
        baseline.write_text('this is not a baseline\n', encoding='utf-8')
    else:
        baseline.write_text(seeded.replace('"schema_version": 1', '"schema_version": 99'))
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 2

    report = capsys.readouterr()
    assert 'advisory' not in report.out
    assert 'clean' not in report.out
    assert str(baseline) in report.err


def _seed(root: Path, baseline_path: Path, *paths: str) -> int:
    """Run ``--seed`` over *root*, optionally scoped to *paths*."""
    return inline_suppressions.main(
        ['--seed', '--root', str(root), '--baseline', str(baseline_path), *paths]
    )


def test_seed_writes_the_unowned_multiset_under_the_kernels_preamble(tmp_path: Path):
    """``--seed`` is κ1's one call, so what it writes has to be reviewable.

    The kernel owns the file's shape and its preamble — the paragraph stating
    that the only legal diff is a DELETION is re-emitted on every write, which is
    the only handle a reviewer has on a file keyed by content digests.  What this
    scanner owns is the CONTENT: exactly the unowned multiset, and a params block
    naming the kinds it swept and the digest width D7 fixes.
    """
    files = {
        'pyproject.toml': _RUFF_CONFIG,
        'm.py': _grandfathered(3) + 'b = 2  # noqa: E402  # debt: task 5601\n',
    }
    baseline = _write_fixture_tree(tmp_path, files)

    assert _seed(tmp_path, baseline) == 0

    raw = json.loads(baseline.read_text(encoding='utf-8'))
    assert raw['_README'] == BASELINE_README
    assert raw['schema_version'] == SCHEMA_VERSION
    written = load(baseline)
    scan = inline_suppressions.scan_tree(tmp_path)
    expected = inline_suppressions.classify(
        scan, inline_suppressions.ConsumerModel(tmp_path)
    ).counts
    assert dict(written.counts) == expected
    assert len(written.counts) == 3
    assert set(written.params) == {'kinds', 'key_scheme', 'digest_hex'}
    assert written.params['kinds'] == tuple(kind.value for kind in inline_suppressions.Kind)
    assert written.params['digest_hex'] == 12
    assert written.complete is True


def test_a_baseline_just_seeded_makes_the_same_tree_clean(tmp_path: Path, capsys):
    """The round trip, which is the only thing that proves the two halves agree.

    A key the seed writes and a key the check computes are produced by the same
    code, so equality is unsurprising; what this catches is a params block or a
    schema version that differs between the write and the read, which turns every
    later run into an exit 2 nobody can explain.
    """
    baseline = _write_fixture_tree(tmp_path, {'m.py': _grandfathered(4)})
    assert _seed(tmp_path, baseline) == 0
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 0
    assert 'clean' in capsys.readouterr().out


def test_seeding_over_an_existing_baseline_is_refused_and_changes_nothing(
    tmp_path: Path, capsys
):
    """BOUNDARY SCENARIO 6 — and it is asserted on BYTES, not on mtime.

    ``shared.ratchet.dump`` is deliberately unpoliced against whatever already
    sits at its path: seeding a new baseline and carrying an honestly incomplete
    one across a file boundary are both legitimate, and neither survives a writer
    that refuses unfamiliar keys.  So regenerating an existing baseline from a
    fresh scan is the one call that widens the gate, and closing that hole is this
    consumer's job rather than the kernel's.
    """
    baseline = _write_fixture_tree(
        tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\n'}, baseline=True
    )
    before = baseline.read_bytes()
    _revise(tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\nb = 2  # nosec\n'})
    capsys.readouterr()

    assert _seed(tmp_path, baseline) == 2

    assert baseline.read_bytes() == before
    assert str(baseline) in capsys.readouterr().err


def test_a_scoped_seed_is_refused_before_any_scan_work(tmp_path: Path, capsys):
    """A baseline seeded from part of a tree makes every unscanned suppression a
    fresh violation, so the scope is refused rather than honoured.

    REFUSED BEFORE THE SCAN, which is asserted the only way it can be asserted
    from outside: the tree also holds a file that cannot be tokenized, and that
    file is an exit 2 of its own with a message naming it.  A refusal that came
    after the scan would report the file instead of the scope.
    """
    baseline = _write_fixture_tree(
        tmp_path,
        {'pkg/m.py': 'a = 1  # type: ignore[arg-type]\n', 'broken.py': 'a = (  # nosec\n'},
    )
    capsys.readouterr()

    assert _seed(tmp_path, baseline, 'pkg') == 2

    error = capsys.readouterr().err
    assert 'scoped' in error
    assert 'broken.py' not in error
    assert not baseline.exists()


def _tighten(root: Path, baseline_path: Path, *paths: str) -> int:
    """Run ``--tighten`` over *root*, optionally scoped to *paths*."""
    return inline_suppressions.main(
        ['--tighten', '--root', str(root), '--baseline', str(baseline_path), *paths]
    )


def test_removing_a_marker_becomes_slack_that_tighten_takes_away(tmp_path: Path, capsys):
    """BOUNDARY SCENARIO 5 — the ratchet's forward click.

    Slack is not a cosmetic figure: because two identical lines share one D7 key,
    an un-tightened baseline lets an identical line straight back in where one was
    removed.  The headroom is a standing invitation nobody meant to leave open,
    which is why ``--check`` reports it and D11's sweep files a task to spend it.
    """
    lines = _grandfathered(3).splitlines(keepends=True)
    baseline = _write_fixture_tree(tmp_path, {'m.py': ''.join(lines)}, baseline=True)
    _revise(tmp_path, {'m.py': ''.join(lines[:2])})
    capsys.readouterr()

    assert _check(tmp_path, baseline) == 0
    assert 'slack 1' in capsys.readouterr().out

    before = set(load(baseline).counts)
    assert _tighten(tmp_path, baseline) == 0

    reported = capsys.readouterr().out
    (gone,) = before - set(load(baseline).counts)
    assert gone in reported

    assert _check(tmp_path, baseline) == 0
    assert 'slack 0' in capsys.readouterr().out


def test_a_second_tighten_over_an_unchanged_tree_writes_the_same_bytes(
    tmp_path: Path, capsys
):
    """Idempotent, and asserted on BYTES so a rewrite that reordered keys or
    re-rendered a float would show up.

    ``tighten`` is the pointwise minimum, so this follows from the arithmetic
    rather than from a guard — which is the property worth pinning, because a
    baseline that churned on every run would make its own review impossible.
    """
    lines = _grandfathered(3).splitlines(keepends=True)
    baseline = _write_fixture_tree(tmp_path, {'m.py': ''.join(lines)}, baseline=True)
    _revise(tmp_path, {'m.py': ''.join(lines[:2])})

    assert _tighten(tmp_path, baseline) == 0
    once = baseline.read_bytes()
    assert _tighten(tmp_path, baseline) == 0
    capsys.readouterr()

    assert baseline.read_bytes() == once


def test_tighten_never_adds_a_key_so_it_is_not_a_way_to_go_green(tmp_path: Path):
    """The no-add-key property, from the consumer's side.

    ``tighten``'s result is a pointwise minimum, so a key the baseline does not
    hold has multiplicity 0 there and the minimum of anything and 0 is 0 — the
    property is structural rather than a check anyone can forget.  What this test
    pins is the consequence that matters at the gate: an agent facing a red run
    cannot clear it by tightening.
    """
    baseline = _write_fixture_tree(
        tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\n'}, baseline=True
    )
    _revise(
        tmp_path,
        {'m.py': 'a = 1  # type: ignore[arg-type]\nb = 2  # type: ignore[attr-defined]\n'},
    )
    before = set(load(baseline).counts)

    assert _tighten(tmp_path, baseline) == 0

    assert set(load(baseline).counts) == before
    assert _check(tmp_path, baseline) == 1


def test_a_scoped_tighten_is_refused_and_leaves_the_baseline_alone(tmp_path: Path, capsys):
    """The widening the kernel's arithmetic cannot refuse, refused here.

    A scoped scan is honestly ``complete=True`` for its scope, so it passes
    ``_require_comparable`` and the pointwise minimum goes through — writing
    ``baseline ∩ scope`` and silently deleting every key outside the paths.  That
    is a one-command gate-widening, so the verb refuses the scope outright.
    """
    baseline = _write_fixture_tree(
        tmp_path,
        {
            'pkg/kept.py': 'a = 1  # type: ignore[arg-type]\n',
            'other/kept.py': 'b = 2  # type: ignore[attr-defined]\n',
        },
        baseline=True,
    )
    before = baseline.read_bytes()
    capsys.readouterr()

    assert _tighten(tmp_path, baseline, 'pkg') == 2

    assert baseline.read_bytes() == before
    assert 'scoped' in capsys.readouterr().err


#: One tree exercising every block the report publishes: an unowned marker of
#: each pyright spelling, a debt, a policy by inline ratification, both flavours
#: of dead marker, and a file the prefilter drops.
_REPORT_TREE = {
    'pyproject.toml': _RUFF_CONFIG,
    'plain.py': 'nothing = "here"\n',
    'm.py': (
        'a = 1  # type: ignore[arg-type]\n'
        'b = 2  # noqa: E402  # debt: task 5601\n'
        'c = 3  # noqa: F401  # ratified: inv12-day-one-test-doubles\n'
        'd = 4  # pragma: no cover\n'
        'e = 5  # noqa: PLC0415\n'
        'g = 7  # pyright: ignore[reportArgumentType]\n'
    ),
}


def _json_text(root: Path, baseline_path: Path, capsys, *paths: str) -> str:
    """Run ``--json`` and return its raw stdout, asserting the exit code is 0.

    The buffer is drained first: a fixture seeded through ``--seed`` has already
    printed its own report line, and ``json.loads`` of the two concatenated fails
    on the first character with nothing to say about why.
    """
    capsys.readouterr()
    code = inline_suppressions.main(
        ['--json', '--root', str(root), '--baseline', str(baseline_path), *paths]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    return captured.out


def _json_report(root: Path, baseline_path: Path, capsys, *paths: str) -> dict:
    """The parsed ``--json`` report."""
    return json.loads(_json_text(root, baseline_path, capsys, *paths))


def test_json_publishes_every_block_the_register_reads(tmp_path: Path, capsys):
    """``--json`` is γ2's only data source, so nothing downstream re-scans.

    The totals are asserted as WHOLE collections rather than by spot-checking a
    key, because the failure this guards against is a block that quietly stops
    being emitted — which a membership test passes right up until the consumer
    reads a missing key.  Every kind appears even at zero, for the same reason:
    ``nosec`` reading 0 is a measurement, and a schema where it vanishes makes
    "no bandit markers" and "the scanner forgot bandit" the same output.
    """
    baseline = _write_fixture_tree(tmp_path, _REPORT_TREE)
    report = _json_report(tmp_path, baseline, capsys)

    assert report['schema_version'] == inline_suppressions.REPORT_SCHEMA_VERSION == 1
    assert set(report['params']) == {'kinds', 'key_scheme', 'digest_hex'}
    assert report['status'] == 'advisory'
    assert report['files_enumerated'] == 2
    assert report['files_tokenized'] == 1
    assert report['sites'] == 6
    assert report['kind_totals'] == {
        'noqa': 3,
        'nosec': 0,
        'pragma: no cover': 1,
        'pyright: ignore': 1,
        'type: ignore': 1,
    }
    assert report['consumers'] == {'first-party': 0, 'none': 2, 'pyright': 2, 'ruff': 2}
    assert report['ownership'] == {'class': 0, 'debt': 1, 'policy': 1, 'unowned': 4}
    assert report['by_kind_code'] == [
        {'kind': 'noqa', 'code': 'E402', 'sites': 1},
        {'kind': 'noqa', 'code': 'F401', 'sites': 1},
        {'kind': 'noqa', 'code': 'PLC0415', 'sites': 1},
        {'kind': 'pragma: no cover', 'code': None, 'sites': 1},
        {'kind': 'pyright: ignore', 'code': 'reportArgumentType', 'sites': 1},
        {'kind': 'type: ignore', 'code': 'arg-type', 'sites': 1},
    ]


def test_json_names_every_debt_owner_and_every_ratified_id_with_its_sites(
    tmp_path: Path, capsys
):
    """The two blocks that exist for a CONSUMER rather than for a reader.

    γ2's closed-world check asks whether every inline ``ratified:`` id names a
    real ratification row, so it needs the ids AND the sites citing each one — an
    id with no sites to point at is a finding it cannot report.  The debt block
    is the same shape for the same reason: the owner has to be followable back to
    a task or a ticket without re-scanning the tree.
    """
    baseline = _write_fixture_tree(tmp_path, _REPORT_TREE)
    report = _json_report(tmp_path, baseline, capsys)

    assert report['debt'] == [
        {'path': 'm.py', 'line': 2, 'kind': 'noqa', 'codes': ['E402'], 'owner': 'task 5601'}
    ]
    assert report['ratified'] == [
        {
            'id': 'inv12-day-one-test-doubles',
            'sites': [{'path': 'm.py', 'line': 3, 'kind': 'noqa', 'codes': ['F401']}],
        }
    ]


def test_json_publishes_the_resolved_ruff_lists_rather_than_the_params_block(
    tmp_path: Path, capsys
):
    """Decision 3's audit trail, and decision 8's placement of it.

    The consumer model's answer for a ``noqa`` depends entirely on these two
    lists, so a reader has to be able to see what the model actually read —
    including the known meta-prefix limit, which is only visible as an absence.
    They are NOT in ``params``: a params mismatch is exit 2 whose only exit is
    re-seeding, so putting them there would punish an ordinary reviewable
    pyproject edit by demanding the one operation nobody should perform casually.
    """
    files = dict(_REPORT_TREE)
    files['pkg/pyproject.toml'] = '[tool.ruff.lint]\nselect = ["F"]\nignore = []\n'
    files['pkg/mod.py'] = 'h = 8  # noqa: F401\n'
    baseline = _write_fixture_tree(tmp_path, files)

    report = _json_report(tmp_path, baseline, capsys)

    assert report['ruff_config'] == [
        {'pyproject': 'pkg/pyproject.toml', 'select': ['F'], 'ignore': []},
        {
            'pyproject': 'pyproject.toml',
            'select': ['E', 'F', 'UP', 'B', 'SIM', 'I'],
            'ignore': ['E501'],
        },
    ]
    assert 'select' not in report['params']


def test_json_counts_a_class_ratified_site_under_its_class(tmp_path: Path, capsys, monkeypatch):
    """BOUNDARY SCENARIO 10 in the report — blanket policy stays VISIBLE.

    A class row moves sites out of the unowned multiset, which is exactly the
    move that could hide a growing population behind one operator ruling.  So the
    report counts them under the rendered class key, and D11's sweep can see how
    much each row is carrying.
    """
    row = inline_suppressions.SuppressionClass(
        kind=inline_suppressions.Kind.TYPE_IGNORE,
        code='arg-type',
        scope=inline_suppressions.Scope.ANY,
    )
    monkeypatch.setattr(
        inline_suppressions, 'RATIFIED_SUPPRESSION_CLASSES', {row: Policy('inv12-day-one')}
    )
    baseline = _write_fixture_tree(tmp_path, _REPORT_TREE)

    report = _json_report(tmp_path, baseline, capsys)

    assert report['classes'] == [
        {
            'class': 'type: ignore[arg-type]@any',
            'sites': [{'path': 'm.py', 'line': 1, 'kind': 'type: ignore', 'codes': ['arg-type']}],
        }
    ]
    assert report['ownership']['class'] == 1


def test_json_carries_the_verdict_a_check_would_reach(tmp_path: Path, capsys):
    """The report reader sees the gate's answer without running the gate.

    Without this a consumer would have to invoke ``--check`` as well, and then
    reconcile two scans of a tree that may have changed between them.
    """
    baseline = _write_fixture_tree(
        tmp_path,
        {
            'm.py': _grandfathered(3),
            'n.py': 'y = 8  # type: ignore[arg-type]  # debt: soon\n',
        },
        baseline=True,
    )
    _revise(tmp_path, {'m.py': _grandfathered(2) + 'z = 9  # type: ignore[no-any-return]\n'})

    report = _json_report(tmp_path, baseline, capsys)

    assert report['status'] == 'clean'
    assert set(report['baseline']) == {'path', 'present', 'excess', 'slack'}
    assert report['baseline']['path'] == str(baseline)
    assert report['baseline']['present'] is True
    assert list(report['baseline']['excess'].values()) == [1]
    assert list(report['baseline']['slack'].values()) == [1]
    located = [(entry['path'], entry['line']) for entry in report['violations']]
    assert located == [('n.py', 1), ('m.py', 3)]


def test_json_exits_zero_even_when_the_gate_would_be_red(tmp_path: Path, capsys):
    """A REPORT verb, not a gate: a consumer parsing the report must not also be
    gated by it.

    Only a broken instrument makes ``--json`` non-zero, which is what lets γ2
    read a tree that is currently in breach — the state it most needs to read.
    """
    baseline = _write_fixture_tree(tmp_path, {'m.py': 'a = 1\n'}, baseline=True)
    _revise(tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\n'})

    report = _json_report(tmp_path, baseline, capsys)

    assert report['violations'] != []
    assert _check(tmp_path, baseline) == 1


@pytest.mark.parametrize('verb', ['--seed', '--tighten'])
def test_json_is_mutually_exclusive_with_the_writing_verbs(tmp_path: Path, capsys, verb: str):
    """An argparse error rather than a silent precedence rule.

    ``--json --seed`` has two defensible readings (report then write; write then
    report) and no way for the caller to say which they meant, so the CLI refuses
    instead of picking one.
    """
    baseline = _write_fixture_tree(tmp_path, _REPORT_TREE)

    with pytest.raises(SystemExit) as raised:
        inline_suppressions.main(
            ['--json', verb, '--root', str(tmp_path), '--baseline', str(baseline)]
        )

    assert raised.value.code == 2
    assert 'not allowed with' in capsys.readouterr().err


def test_two_json_runs_over_one_tree_emit_identical_bytes(tmp_path: Path, capsys):
    """Determinism, asserted on BYTES, plus the sort that makes it hold.

    A report that reordered between runs would make every consumer's diff noise,
    and the reason it cannot is structural: every mapping is dumped with sorted
    keys and every list is sorted before it is dumped.
    """
    baseline = _write_fixture_tree(tmp_path, _REPORT_TREE)

    first = _json_text(tmp_path, baseline, capsys)
    second = _json_text(tmp_path, baseline, capsys)

    assert first == second
    report = json.loads(first)
    assert list(report) == sorted(report)
    assert list(report['kind_totals']) == sorted(report['kind_totals'])
    assert report['by_kind_code'] == sorted(
        report['by_kind_code'], key=lambda row: (row['kind'], row['code'] or '')
    )


def test_a_missing_shared_import_is_exit_two_and_never_exit_one(tmp_path: Path):
    """The Contract's "an ImportError is 2, never 1", which only a real process
    can prove.

    Every other test here calls ``main([...])`` in-process, and none of them could
    catch this: the fault it pins happens before ``main`` is reachable.  A
    top-level ``from shared… import …`` raises while the module is still
    executing, so ``main`` is never defined, the ``__main__`` block never runs,
    and Python's own uncaught-exception exit is 1 — the exact code the Contract
    reserves for a FINDING.  A gate that reported a broken environment as an
    INV-12 breach would send an agent to fix code that was never the problem.

    The script is copied ALONE into ``tmp_path``, so the ``<repo>/shared/src`` its
    bootstrap resolves from ``__file__`` does not exist either, and the run gets
    ``-S`` because the workspace member is editable-installed in the venv and
    ``PYTHONPATH`` alone cannot hide it.  A generous subprocess ``timeout``
    rather than any wall-clock assertion, per this module's docstring.
    """
    copied = tmp_path / SCRIPT.name
    copied.write_text(SCRIPT.read_text(encoding='utf-8'), encoding='utf-8')

    completed = _run_script(
        ['--check', '--root', str(tmp_path), '--baseline', str(tmp_path / _BASELINE_NAME)],
        cwd=tmp_path,
        script=copied,
        env=_python_env_without_shared(),
        flags=_NO_SITE,
    )

    assert completed.returncode == 2, completed.stderr
    assert 'shared' in completed.stderr
    assert 'uv run --project shared' in completed.stderr
    assert 'Traceback' not in completed.stderr


# ---------------------------------------------------------------------------
# The live tree — the merge gate's actual enforcement point.

#: Where κ1 seeds the committed baseline.  Absent until that cutover lands,
#: which is what makes the enforcing guard below a skip rather than a red.
LIVE_BASELINE = REPO_ROOT / 'scripts' / _BASELINE_NAME


def _live_report(capsys) -> dict:
    """The ``--json`` report for THIS repository."""
    capsys.readouterr()
    code = inline_suppressions.main(['--json', '--root', str(REPO_ROOT)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    return json.loads(captured.out)


def test_the_live_tree_reports_the_signal_this_scanner_exists_to_produce(capsys):
    """The NON-VACUITY FLOOR, because "nothing was read" and "nothing was wrong"
    are otherwise the same output.

    Four kinds are non-zero in this repository and the fifth is zero, and that
    exact shape is the task's user-observable deliverable: pyright and ruff are
    both declared gates here, no coverage gate is configured, and bandit is not
    installed — so ``nosec`` reading 0 is a measurement about TOOLS rather than a
    scanner that stopped looking.  Asserting the zero beside the four is what
    makes a broken enumeration loud: a scan that silently read nothing would
    satisfy the zero and fail every other assertion here.
    """
    report = _live_report(capsys)

    assert report['files_enumerated'] >= 1000
    assert report['sites'] >= 1000
    for kind in ('type: ignore', 'noqa', 'pragma: no cover', 'pyright: ignore'):
        assert report['kind_totals'][kind] > 0, kind
    assert report['kind_totals']['nosec'] == 0
    assert report['consumers']['none'] > 0
    assert report['ruff_config'] != []


def test_the_live_scan_tokenizes_exactly_the_marker_bearing_files(capsys):
    """THE ≤10 s BUDGET, PINNED AS COUNTED WORK AND NEVER AS A CLOCK.

    The budget fits only because of the prefilter, so the honest guard is that
    the prefilter is doing its job: strictly fewer files are tokenized than
    enumerated, and the ones that are are exactly those whose raw bytes carry a
    marker substring.  An ``assert elapsed < 10`` would instead be a new flake on
    the box the merge gate runs on — measured, this scan costs about 5 CPU-seconds
    and took 6.5-6.8 s of wall clock at load 105 on 32 cores.

    The expected count is re-derived from the PUBLIC kind table rather than read
    off the scanner's private prefilter constant, so the two can disagree; a
    marker added to the table with no prefilter byte is exactly what that catches.
    A tracked path whose worktree file is gone is passed over here for the same
    reason the scanner passes over it: ``git ls-files`` reads the index.
    """
    markers = tuple(spec.marker.encode('utf-8') for spec in inline_suppressions.KIND_SPECS.values())
    listed = _run_git(['ls-files', '-z', '--', '*.py'], cwd=REPO_ROOT).stdout
    tracked = {path for path in listed.split('\0') if path}
    carrying = 0
    for relative in tracked:
        try:
            raw = (REPO_ROOT / relative).read_bytes()
        except FileNotFoundError:
            continue
        carrying += any(marker in raw for marker in markers)

    report = _live_report(capsys)

    assert report['files_enumerated'] == len(tracked)
    assert report['files_tokenized'] < report['files_enumerated']
    assert report['files_tokenized'] == carrying


def test_the_live_tree_passes_the_gate_once_the_baseline_is_seeded(capsys):
    """The merge gate's actual assertion — ENFORCING iff the baseline exists.

    It skips rather than reds before the κ1 cutover because D12 makes baseline
    absence a legitimate state, and a guard that failed on the pre-cutover tree
    would block every task until κ1 landed.  Everything about the gate MECHANISM
    is covered hermetically by the fixture-tree tests above, so the skip loses no
    coverage of this module's behaviour — only of this repository's compliance,
    which is not yet a thing to be compliant with.
    """
    if not LIVE_BASELINE.exists():
        pytest.skip(
            f'{LIVE_BASELINE.relative_to(REPO_ROOT)} does not exist yet: the baseline is '
            'seeded once, on main, by the operator step κ1. Until then every run is '
            'advisory by design (D12), and the gate mechanism is covered hermetically by '
            'the fixture-tree tests in this module.'
        )
    capsys.readouterr()

    assert inline_suppressions.main(['--check', '--root', str(REPO_ROOT)]) == 0, (
        capsys.readouterr().err
    )
