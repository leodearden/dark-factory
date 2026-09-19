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
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import inline_suppressions

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
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')

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
