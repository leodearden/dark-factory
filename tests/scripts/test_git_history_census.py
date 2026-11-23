"""The git-history census, pinned on synthetic sources and on a checkout planted under tmp_path.

Every source here is text parsed in memory or written to disk; no git process runs.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import git_history_census
import pytest
from git_history_census import (
    MARKER,
    GateTestRoot,
    HistoryCall,
    Sweep,
    SweepSource,
    UnreadableGateFile,
    applied_marker_names,
    history_calls,
    imported_module_stems,
    main,
    may_hold_a_history_call_or_marker,
    sweep,
    sweep_sources,
    text_source,
    unaccounted_reader_failure,
)

_GIT_PROGRAM = 'git'

# ─── The detector, pinned on synthetic sources ──────────────────────────────


@pytest.mark.parametrize(
    ('source', 'subcommand'),
    [
        pytest.param(
            "subprocess.run(['git', '-C', str(tmp_path), 'log', '-1'])", 'log',
            id='fixture-repo-argv',
        ),
        pytest.param(
            "subprocess.run(('git', '-C', str(REPO_ROOT), 'merge-base', 'HEAD', 'main'))",
            'merge-base', id='checkout-argv',
        ),
        pytest.param("_git(repo, 'rev-list', '--count', 'HEAD')", 'rev-list', id='git-helper'),
        pytest.param("repo.git('describe', '--tags')", 'describe', id='git-method'),
        pytest.param("_run_git(root, 'log', '--format=%H')", 'log', id='suffixed-helper'),
        pytest.param("subprocess.run('git log -1 --format=%H', shell=True)", 'log', id='shelled'),
        pytest.param("run(f'git -C {root} merge-base HEAD main')", 'merge-base', id='f-string'),
    ],
)
def test_a_history_call_is_detected(source: str, subcommand: str) -> None:
    assert [call.subcommand for call in history_calls(ast.parse(source))] == [subcommand]


@pytest.mark.parametrize(
    'source',
    [
        pytest.param('"""Reads git log against the checkout."""', id='docstring'),
        pytest.param('x = 1  # git log --oneline', id='comment'),
        pytest.param("subprocess.run(['git', 'commit', '-m', 'log'])", id='message-argv'),
        pytest.param("_git(repo, 'commit', '-m', 'merge-base')", id='message-helper'),
        pytest.param("subprocess.run(['git', 'status'])", id='other-subcommand'),
        pytest.param("subprocess.run(['git', *args])", id='unknowable-argv'),
        pytest.param("MESSAGE = 'git log is a history read'", id='string-not-passed'),
    ],
)
def test_a_non_history_call_is_not_detected(source: str) -> None:
    assert history_calls(ast.parse(source)) == []


def test_a_detected_call_carries_its_line() -> None:
    source = "import subprocess\n\n\nsubprocess.run(['git', 'rev-list', 'HEAD'])\n"
    assert history_calls(ast.parse(source)) == [HistoryCall(line=4, subcommand='rev-list')]


def test_the_marker_is_seen_on_one_class_among_several() -> None:
    source = (
        'import pytest\n'
        'class TestTree:\n'
        '    def test_a(self): ...\n'
        f'@pytest.mark.{MARKER}\n'
        'class TestHistory:\n'
        '    def test_b(self): ...\n'
    )
    assert MARKER in applied_marker_names(ast.parse(source))


def test_the_marker_is_seen_in_a_pytestmark_list() -> None:
    source = f'import pytest\npytestmark = [pytest.mark.timeout(60), pytest.mark.{MARKER}]\n'
    assert MARKER in applied_marker_names(ast.parse(source))


def test_a_marker_named_only_in_a_string_is_not_applied() -> None:
    assert MARKER not in applied_marker_names(ast.parse(f"NOTE = 'pytest.mark.{MARKER}'\n"))


@pytest.mark.parametrize(
    'source',
    [
        pytest.param('from _lme_artifacts import load', id='from-module'),
        pytest.param('import _lme_artifacts', id='import'),
        pytest.param('from . import _lme_artifacts', id='relative-submodule'),
        pytest.param('from tests.arm_harness._lme_artifacts import load', id='dotted'),
        pytest.param('def f():\n    import _lme_artifacts\n', id='nested'),
    ],
)
def test_a_helper_import_is_resolved_to_its_stem(source: str) -> None:
    assert '_lme_artifacts' in imported_module_stems(ast.parse(source))


# ─── The sweep's accounting, on synthetic sources ───────────────────────────

_SYNTHETIC_ROOT = GateTestRoot(path='pkg/tests', pytest_config='pkg/pyproject.toml')
_HISTORY_HELPER_SOURCE = "def base(root):\n    return _git(root, 'merge-base', 'main', 'HEAD')\n"


def _swept(*sources: SweepSource) -> Sweep:
    return sweep_sources(frozenset(source.root for source in sources), sources)


def test_a_file_that_never_spells_git_is_still_found_as_an_importer() -> None:
    helper = text_source('pkg/tests/_history.py', _SYNTHETIC_ROOT, _HISTORY_HELPER_SOURCE)
    importer = text_source(
        'pkg/tests/test_versions.py', _SYNTHETIC_ROOT, 'from _history import versions\n',
    )
    synthetic = _swept(helper, importer)
    swept_helper, swept_importer = synthetic.files

    assert swept_importer.calls == ()
    assert synthetic.importing_test_files(swept_helper) == (swept_importer,)


def test_a_marked_importer_of_a_history_helper_does_not_vouch_for_an_unmarked_one() -> None:
    other_root = GateTestRoot(path='other/tests', pytest_config=None)
    synthetic = _swept(
        text_source('pkg/tests/_history.py', _SYNTHETIC_ROOT, _HISTORY_HELPER_SOURCE),
        text_source(
            'pkg/tests/test_reads.py', _SYNTHETIC_ROOT,
            f'import pytest\nfrom _history import base\npytestmark = pytest.mark.{MARKER}\n',
        ),
        text_source(
            'pkg/tests/test_also_reads.py', _SYNTHETIC_ROOT, 'from _history import base\n',
        ),
        text_source('other/tests/test_other.py', other_root, 'from _history import base\n'),
    )
    files = synthetic.by_path()
    helper, unmarked = files['pkg/tests/_history.py'], files['pkg/tests/test_also_reads.py']
    elsewhere = files['other/tests/test_other.py']

    assert synthetic.ways_accounted_for(helper) == ['a helper imported by a marked test file']
    assert synthetic.unaccounted_helper_importers() == [(helper, unmarked)]
    assert synthetic.reaches_a_history_call(unmarked)
    assert not synthetic.reaches_a_history_call(elsewhere)


def test_the_importers_of_a_helper_nothing_accounts_for_wait_on_the_helper() -> None:
    synthetic = _swept(
        text_source('pkg/tests/_history.py', _SYNTHETIC_ROOT, _HISTORY_HELPER_SOURCE),
        text_source('pkg/tests/test_reads.py', _SYNTHETIC_ROOT, 'from _history import base\n'),
    )
    helper = synthetic.by_path()['pkg/tests/_history.py']

    assert synthetic.ways_accounted_for(helper) == []
    assert synthetic.unaccounted_helper_importers() == []


def test_importers_are_resolved_only_for_a_module_that_makes_a_history_call() -> None:
    synthetic = _swept(
        text_source('pkg/tests/_plain.py', _SYNTHETIC_ROOT, 'VALUE = 1\n'),
        text_source('pkg/tests/test_uses.py', _SYNTHETIC_ROOT, 'from _plain import VALUE\n'),
    )
    with pytest.raises(AssertionError, match='makes no history call'):
        synthetic.importing_test_files(synthetic.by_path()['pkg/tests/_plain.py'])


# ─── The prefilter must not blind the census ────────────────────────────────

_SPELLINGS_OF_A_HISTORY_CALL = [
    pytest.param("subprocess.run(['git', 'log', '-1'])", 'log', id='argv-list'),
    pytest.param("subprocess.run(('git', '-C', str(root), 'merge-base', 'a', 'b'))", 'merge-base', id='argv-tuple'),
    pytest.param("_git(root, 'rev-list', 'HEAD')", 'rev-list', id='underscore-helper'),
    pytest.param("def f(self):\n    return self._git('describe', '--tags')\n", 'describe', id='method'),
    pytest.param("run_git(root, 'log')", 'log', id='suffixed-helper'),
    pytest.param("git(root, 'log')", 'log', id='bare-helper'),
    pytest.param("run('git log -1 --format=%H')", 'log', id='shelled'),
    pytest.param("run(f'git -C {root} merge-base HEAD main')", 'merge-base', id='shelled-in-f-string'),
    pytest.param("\uff47\uff49\uff54(root, 'log')", 'log', id='full-width-helper-identifier'),
    pytest.param("run_\uff47\uff49\uff54(root, 'log')", 'log', id='full-width-helper-suffix'),
    pytest.param(r"subprocess.run(['\x67it', 'log'])", 'log', id='escaped-argv-program'),
    pytest.param(r"subprocess.run(['\u0067it', 'log'])", 'log', id='unicode-escaped-argv-program'),
    pytest.param(r"subprocess.run(['\147it', 'log'])", 'log', id='octal-escaped-argv-program'),
    pytest.param(r"subprocess.run(['\N{LATIN SMALL LETTER G}it', 'log'])", 'log', id='named-escape'),
    pytest.param("subprocess.run(['g' 'it', 'log'])", 'log', id='adjacent-literals-argv'),
    pytest.param("subprocess.run(['gi'\n    't', 'log'])", 'log', id='adjacent-literals-across-lines'),
    pytest.param(
        "subprocess.run(['g'  # the program\n    'i'\n    't', 'log'])", 'log',
        id='adjacent-literals-around-a-comment',
    ),
    pytest.param("run('g' 'it log -1')", 'log', id='shelled-across-adjacent-literals'),
    pytest.param("run(f'g' f'it log {x}')", 'log', id='shelled-across-adjacent-f-strings'),
    pytest.param("run('g\\\nit log -1')", 'log', id='shelled-across-a-line-continuation'),
    pytest.param("run(r'g' 'it log')", 'log', id='raw-prefixed-adjacent-literal'),
]


@pytest.mark.parametrize(('source', 'subcommand'), _SPELLINGS_OF_A_HISTORY_CALL)
def test_the_real_sweep_reports_a_history_call_however_it_is_spelled(
    source: str, subcommand: str, tmp_path: Path,
) -> None:
    (tmp_path / 'pkg/tests').mkdir(parents=True)
    (tmp_path / 'pkg/tests/_reads.py').write_text(source)

    (swept,) = sweep(frozenset({_SYNTHETIC_ROOT}), repo_root=tmp_path).files

    assert [call.subcommand for call in swept.calls] == [subcommand]
    assert swept.calls == tuple(history_calls(ast.parse(source)))


def _escapes_spelling(letter: str) -> list[str]:
    code = ord(letter)
    return [
        f'\\x{code:02x}', f'\\u{code:04x}', f'\\U{code:08x}', f'\\{code:o}',
        f'\\N{{LATIN SMALL LETTER {letter.upper()}}}',
    ]


@pytest.mark.parametrize('position', range(len(_GIT_PROGRAM)))
def test_the_real_sweep_reports_a_call_whose_program_is_escaped_in_any_one_letter(
    position: int, tmp_path: Path,
) -> None:
    (tmp_path / 'pkg/tests').mkdir(parents=True)
    for index, escape in enumerate(_escapes_spelling(_GIT_PROGRAM[position])):
        program = _GIT_PROGRAM[:position] + escape + _GIT_PROGRAM[position + 1:]
        source = f"subprocess.run(['{program}', 'log'])\n"
        (tmp_path / f'pkg/tests/_reads_{index}.py').write_text(source)

    swept = sweep(frozenset({_SYNTHETIC_ROOT}), repo_root=tmp_path)

    assert [[call.subcommand for call in file.calls] for file in swept.files] == [['log']] * 5


@pytest.mark.parametrize(
    'marker_spelling',
    [
        pytest.param(f'pytest.mark.{MARKER}', id='attribute'),
        pytest.param(f'pytest.mark.{MARKER}()', id='call'),
        pytest.param(f'pytest.mark.{MARKER.replace("r", chr(0xFF52), 1)}', id='full-width-identifier'),
    ],
)
def test_the_real_sweep_sees_the_marker_in_a_file_that_makes_no_call(
    marker_spelling: str, tmp_path: Path,
) -> None:
    (tmp_path / 'pkg/tests').mkdir(parents=True)
    (tmp_path / 'pkg/tests/test_marked.py').write_text(
        f'import pytest\npytestmark = {marker_spelling}\n'
    )

    (swept,) = sweep(frozenset({_SYNTHETIC_ROOT}), repo_root=tmp_path).files

    assert swept.calls == ()
    assert swept.carries_marker


def test_the_real_sweep_finds_every_importer_of_a_history_helper(tmp_path: Path) -> None:
    sources = {
        '_history.py': _HISTORY_HELPER_SOURCE,
        'test_from_import.py': 'from _history import base\n',
        'test_plain_import.py': 'import _history\n',
        'test_dotted_import.py': 'import pkg.tests._history as history\n',
        'test_nested_import.py': 'def test_it():\n    from . import _history\n',
        'test_full_width_import.py': 'from _\uff48istory import base\n',
        'test_other_helper.py': 'from _other import base\n',
        'test_mentions_only.py': '# _history is not imported here\nNAME = "_history"\n',
    }
    (tmp_path / 'pkg/tests').mkdir(parents=True)
    for name, source in sources.items():
        (tmp_path / 'pkg/tests' / name).write_text(source)

    swept = sweep(frozenset({_SYNTHETIC_ROOT}), repo_root=tmp_path)
    helper = swept.by_path()['pkg/tests/_history.py']

    assert sorted(importer.stem for importer in swept.importing_test_files(helper)) == [
        'test_dotted_import', 'test_from_import', 'test_full_width_import',
        'test_nested_import', 'test_plain_import',
    ]


def test_a_file_with_no_git_in_it_yields_no_call_and_is_not_parsed(tmp_path: Path) -> None:
    (tmp_path / 'pkg/tests').mkdir(parents=True)
    (tmp_path / 'pkg/tests/test_never_parsed.py').write_text('def broken(:\n')

    (swept,) = sweep(frozenset({_SYNTHETIC_ROOT}), repo_root=tmp_path).files

    assert swept.calls == ()
    assert not swept.carries_marker


def test_an_unparseable_file_that_may_hold_a_history_call_fails_the_sweep(tmp_path: Path) -> None:
    (tmp_path / 'pkg/tests').mkdir(parents=True)
    (tmp_path / 'pkg/tests/_reads.py').write_text("def broken(:\n    _git(root, 'log')\n")

    with pytest.raises(UnreadableGateFile, match='_reads.py could not be read or parsed'):
        sweep(frozenset({_SYNTHETIC_ROOT}), repo_root=tmp_path)


def test_an_unparseable_test_file_naming_a_history_helper_fails_the_sweep(tmp_path: Path) -> None:
    (tmp_path / 'pkg/tests').mkdir(parents=True)
    (tmp_path / 'pkg/tests/_history.py').write_text(_HISTORY_HELPER_SOURCE)
    (tmp_path / 'pkg/tests/test_uses.py').write_text('from _history import (\n')

    with pytest.raises(UnreadableGateFile, match='test_uses.py could not be read or parsed'):
        sweep(frozenset({_SYNTHETIC_ROOT}), repo_root=tmp_path)


@pytest.mark.parametrize(
    'source',
    [
        pytest.param('x = 1\n', id='nothing'),
        pytest.param('import os\nprint("a\\nb", os.sep)\n', id='harmless-escape'),
        pytest.param('names = ["msg", "item"]\n', id='strings-separated-by-a-comma'),
        pytest.param(r"print('\x1b[0m \u2014 \n')" + '\n', id='escapes-spelling-no-letter-of-git'),
        pytest.param('a = "msg"\nb = "item"\n', id='strings-on-separate-statements'),
        pytest.param('import pytest\npytestmark = pytest.mark.slow\n', id='another-marker'),
    ],
)
def test_a_source_that_cannot_spell_git_is_not_admitted(source: str) -> None:
    assert not may_hold_a_history_call_or_marker(source)


# ─── The throwaway mapping is an argument ───────────────────────────────────

_SEEDS_ITS_REPO = 'pkg/tests/test_seeds_its_repo.py'
_SEEDS_ITS_REPO_SOURCE = (
    'import subprocess\n'
    '\n'
    '\n'
    'def test_counts_commits(tmp_path):\n'
    "    subprocess.run(['git', '-C', str(tmp_path), 'rev-list', '--count', 'HEAD'], check=True)\n"
)
_SEEDS_ITS_REPO_CALL_LINE = 5


def _plant(checkout: Path, rel_path: str, text: str) -> None:
    path = checkout / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_a_file_listed_in_the_throwaway_mapping_is_accounted_for(tmp_path: Path) -> None:
    _plant(tmp_path, _SEEDS_ITS_REPO, _SEEDS_ITS_REPO_SOURCE)
    roots = frozenset({_SYNTHETIC_ROOT})

    listed = sweep(roots, repo_root=tmp_path, throwaway={_SEEDS_ITS_REPO: 'reason'})
    unlisted = sweep(roots, repo_root=tmp_path, throwaway={})

    assert unaccounted_reader_failure(listed) is None
    assert unaccounted_reader_failure(unlisted) is not None


# ─── The command line ───────────────────────────────────────────────────────


def _census(checkout: Path, *roots: str) -> int:
    return main(['--checkout', str(checkout), *roots])


def test_an_unaccounted_reader_of_a_tmp_repo_fails_the_census_naming_its_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _plant(tmp_path, _SEEDS_ITS_REPO, _SEEDS_ITS_REPO_SOURCE)

    code = _census(tmp_path, 'pkg/tests')
    out = capsys.readouterr().out

    assert code == 1
    assert f'{_SEEDS_ITS_REPO}:{_SEEDS_ITS_REPO_CALL_LINE}' in out
    assert 'git rev-list' in out


def test_a_marked_reader_passes_the_census_silently(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _plant(
        tmp_path, _SEEDS_ITS_REPO,
        f'import pytest\npytestmark = pytest.mark.{MARKER}\n{_SEEDS_ITS_REPO_SOURCE}',
    )

    code = _census(tmp_path, 'pkg/tests')
    captured = capsys.readouterr()

    assert (code, captured.out, captured.err) == (0, '', '')


def test_an_unmarked_importer_of_a_marked_files_helper_fails_the_census_naming_both(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _plant(tmp_path, 'pkg/tests/_history.py', _HISTORY_HELPER_SOURCE)
    _plant(
        tmp_path, 'pkg/tests/test_reads.py',
        f'import pytest\nfrom _history import base\npytestmark = pytest.mark.{MARKER}\n',
    )
    _plant(tmp_path, 'pkg/tests/test_also_reads.py', 'from _history import base\n')

    code = _census(tmp_path, 'pkg/tests')
    out = capsys.readouterr().out

    assert code == 1
    assert 'pkg/tests/test_also_reads.py' in out
    assert 'pkg/tests/_history.py' in out


@pytest.mark.parametrize(
    'root',
    [
        pytest.param('no/such/dir', id='missing'),
        pytest.param(_SEEDS_ITS_REPO, id='a-file'),
        pytest.param('..', id='outside-the-checkout'),
    ],
)
def test_a_root_that_is_not_a_directory_under_the_checkout_is_a_usage_error(
    root: str, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _plant(tmp_path, _SEEDS_ITS_REPO, _SEEDS_ITS_REPO_SOURCE)

    code = _census(tmp_path, 'pkg/tests', root)
    captured = capsys.readouterr()

    assert code == 2
    assert repr(root) in captured.err
    assert captured.out == ''


def test_an_admitted_file_the_parser_rejects_is_an_incomplete_census_not_a_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    _plant(tmp_path, 'pkg/tests/_reads.py', "def broken(:\n    _git(root, 'log')\n")

    code = _census(tmp_path, 'pkg/tests')
    captured = capsys.readouterr()

    assert code == 2
    assert 'pkg/tests/_reads.py' in captured.err
    assert captured.out == ''


def test_the_census_runs_as_a_script(tmp_path: Path) -> None:
    _plant(tmp_path, _SEEDS_ITS_REPO, _SEEDS_ITS_REPO_SOURCE)
    script = Path(git_history_census.__file__).resolve()

    proc = subprocess.run(
        [sys.executable, str(script), '--checkout', str(tmp_path), 'pkg/tests'],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )

    assert proc.returncode == 1, proc.stderr
    assert f'{_SEEDS_ITS_REPO}:{_SEEDS_ITS_REPO_CALL_LINE}' in proc.stdout
