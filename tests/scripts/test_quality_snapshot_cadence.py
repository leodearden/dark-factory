"""Tests of ``scripts/quality_snapshot_cadence.py``, the pinned, identity-keyed snapshot runner.

Every fixture is a small REAL single-package git repository measured by the
real instrument in a subprocess; a mid-run event is simulated by a ``--python``
shim that acts and then execs the real interpreter, so no module is patched.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import quality_metrics_snapshot as snapshot
import quality_snapshot_cadence as cadence
from git_listing import git

REPO_ROOT = Path(__file__).parents[2]

_FILES: dict[str, str] = {
    'pyproject.toml': '[project]\nname = "pkg"\n',
    'src/pkg/__init__.py': '',
    'src/pkg/mod.py': 'def f(a):\n    if a:\n        return 1\n    return 0\n',
    'tests/test_mod.py': 'def test_x():\n    assert True\n',
}

_IDENTITY = ('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid')


def _commit(root: Path, message: str) -> str:
    git(root, 'add', '-A')
    git(root, *_IDENTITY, 'commit', '--no-verify', '--allow-empty', '-q', '-m', message)
    return git(root, 'rev-parse', 'HEAD').strip()


def _repo(tmp_path: Path, files: dict[str, str] = _FILES) -> Path:
    """A committed repo under *tmp_path* whose one branch is main."""
    root = tmp_path / 'repo'
    for relpath, text in files.items():
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
    git(root, 'init', '-q')
    _commit(root, 'base')
    git(root, 'branch', '-M', 'main')
    return root


@pytest.fixture(autouse=True)
def state_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / 'state'
    monkeypatch.setenv('XDG_STATE_HOME', str(home))
    return home


def _pinned_trees_home(state_home: Path) -> Path:
    return state_home / 'dark-factory' / 'quality-snapshot' / 'fixture'


def _cadence(root: Path, data: Path, *extra: str, python: str = sys.executable) -> int:
    return cadence.main([
        '--root', str(root), '--project-id', 'fixture', '--python', python,
        '--data-dir', str(data), *extra,
    ])


def _main_sha(root: Path) -> str:
    return git(root, 'rev-parse', 'main').strip()


def _worktrees(root: Path) -> list[Path]:
    listing = git(root, 'worktree', 'list', '--porcelain').splitlines()
    return [Path(line.removeprefix('worktree ')) for line in listing if line.startswith('worktree ')]


def _below_run_id_and_cost(path: Path) -> list[str]:
    """*path*'s lines without the two a re-run changes: run_id and the wall clock."""
    return [
        line
        for line in path.read_text(encoding='utf-8').splitlines()
        if not line.startswith(('  "run_id": ', '    "wall_clock_s": '))
    ]


def _shim(tmp_path: Path, body: str) -> Path:
    """An executable ``--python`` that runs *body*, then execs the real interpreter on its argv."""
    shim = tmp_path / 'bin' / 'python-shim'
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text(
        f'#!{sys.executable}\nimport os\nimport subprocess\nimport sys\n\n{body}\n'
        'os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n',
        encoding='utf-8',
    )
    shim.chmod(0o755)
    return shim


#: A merge landing on main while the instrument runs.
_MOVES_MAIN = '''if '--run-id' in sys.argv:
    subprocess.run(
        ['git', '-C', os.environ['SHIM_ROOT'], '-c', 'user.name=f', '-c', 'user.email=f@e.invalid',
         'commit', '--allow-empty', '--no-verify', '-q', '-m', 'moved'],
        check=True,
    )'''


class TestAFirstRun:
    def test_it_measures_main_into_the_data_dir_and_removes_the_pinned_tree(
        self, tmp_path: Path, capfd: pytest.CaptureFixture[str], state_home: Path
    ) -> None:
        root = _repo(tmp_path)
        data = tmp_path / 'data'
        sha = _main_sha(root)
        assert _cadence(root, data) == 0
        out = data / 'fixture' / f'{sha}.json'
        assert snapshot.load_snapshot(out)['as_of_sha'] == sha
        assert f'measured: {sha} -> {out}' in capfd.readouterr().out.splitlines()
        assert _worktrees(root) == [root.resolve()]
        assert list(_pinned_trees_home(state_home).iterdir()) == []


class TestTheIdentitySkip:
    def test_a_second_run_at_one_identity_is_already_measured(
        self, tmp_path: Path, capfd: pytest.CaptureFixture[str]
    ) -> None:
        root = _repo(tmp_path)
        data = tmp_path / 'data'
        sha = _main_sha(root)
        assert _cadence(root, data) == 0
        out = data / 'fixture' / f'{sha}.json'
        written, written_at = out.read_bytes(), out.stat().st_mtime_ns
        capfd.readouterr()
        assert _cadence(root, data) == 0
        params = snapshot.load_snapshot(out)['params']
        printed = capfd.readouterr().out
        assert printed.splitlines() == [f'already measured: {sha} {json.dumps(params, sort_keys=True)}']
        assert (out.read_bytes(), out.stat().st_mtime_ns) == (written, written_at)
        assert 'wrote ' not in printed

    def test_force_re_measures_byte_identically_below_run_id_and_cost(
        self, tmp_path: Path, capfd: pytest.CaptureFixture[str]
    ) -> None:
        # The PRD's sketch row 16.
        root = _repo(tmp_path)
        data = tmp_path / 'data'
        assert _cadence(root, data) == 0
        out = data / 'fixture' / f'{_main_sha(root)}.json'
        first = tmp_path / 'first.json'
        shutil.copyfile(out, first)
        assert _cadence(root, data, '--force') == 0
        assert 'already measured' not in capfd.readouterr().out
        assert _below_run_id_and_cost(out) == _below_run_id_and_cost(first)
        assert snapshot.load_snapshot(out)['run_id'] != snapshot.load_snapshot(first)['run_id']

    def test_a_changed_params_re_measures(
        self, tmp_path: Path, capfd: pytest.CaptureFixture[str]
    ) -> None:
        root = _repo(tmp_path)
        data = tmp_path / 'data'
        assert _cadence(root, data) == 0
        out = data / 'fixture' / f'{_main_sha(root)}.json'
        older = snapshot.load_snapshot(out)
        older['params']['complexipy_version'] = '0.0.0'
        out.write_text(snapshot.render_snapshot(older), encoding='utf-8')
        capfd.readouterr()
        assert _cadence(root, data) == 0
        assert 'already measured' not in capfd.readouterr().out
        assert snapshot.load_snapshot(out)['params'] == snapshot.measurement_params()


class TestAnExplicitShaList:
    def test_each_sha_is_measured_and_named_by_its_full_sha(self, tmp_path: Path) -> None:
        root = _repo(tmp_path)
        first = _main_sha(root)
        (root / 'src' / 'pkg' / 'mod.py').write_text('def f(a):\n    return a\n', encoding='utf-8')
        second = _commit(root, 'second')
        data = tmp_path / 'data'
        assert _cadence(root, data, '--sha', first, second[:10]) == 0
        for sha in (first, second):
            assert snapshot.load_snapshot(data / 'fixture' / f'{sha}.json')['as_of_sha'] == sha


class TestThePinnedTree:
    def test_main_moving_mid_run_is_measured_at_the_sha_it_had(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state_home: Path
    ) -> None:
        # The PRD's sketch row 18. The live checkout is also dirty, which the
        # instrument would refuse there: only the pinned tree is measured.
        root = _repo(tmp_path)
        before = _main_sha(root)
        (root / 'src' / 'pkg' / 'mod.py').write_text('CHANGED = 1\n', encoding='utf-8')
        monkeypatch.setenv('SHIM_ROOT', str(root))
        data = tmp_path / 'data'
        assert _cadence(root, data, python=str(_shim(tmp_path, _MOVES_MAIN))) == 0
        assert snapshot.load_snapshot(data / 'fixture' / f'{before}.json')['as_of_sha'] == before
        assert _main_sha(root) != before
        assert _worktrees(root) == [root.resolve()]
        assert list(_pinned_trees_home(state_home).iterdir()) == []


# ---------------------------------------------------------------------------
# Refusals: exit 2, the cause named on stderr, the pinned tree removed.


def _refused(
    root: Path, data: Path, capfd: pytest.CaptureFixture[str], *extra: str, python: str = sys.executable
) -> str:
    """Run, assert the refusal and that no worktree is left behind, and return stderr."""
    assert _cadence(root, data, *extra, python=python) == 2
    assert _worktrees(root) == [root.resolve()]
    return capfd.readouterr().err


#: Ask the real instrument for its params, then report a complexipy version it does not record.
_MISPREDICTS = '''if '--params' in sys.argv:
    import json
    asked = subprocess.run([sys.executable, *sys.argv[1:]], capture_output=True, text=True, check=True)
    params = json.loads(asked.stdout)
    params['complexipy_version'] = '0.0.0'
    print(json.dumps(params))
    sys.exit(0)'''


#: The instrument failing after locking its pinned tree, so that removing the tree fails too.
_LOCKS_AND_FAILS = '''if '--run-id' in sys.argv:
    tree = sys.argv[sys.argv.index('--root') + 1]
    subprocess.run(['git', '-C', tree, 'worktree', 'lock', tree], check=True)
    print('shim: the measurement failed', file=sys.stderr)
    sys.exit(1)'''


class TestRefusals:
    def test_an_instrument_failure_names_its_cause_and_the_sha(
        self, tmp_path: Path, capfd: pytest.CaptureFixture[str], state_home: Path
    ) -> None:
        root = _repo(tmp_path, {**_FILES, 'pyproject.toml': '[tool.ruff]\nline-length = 100\n'})
        data = tmp_path / 'data'
        err = _refused(root, data, capfd)
        assert '[project].name' in err
        assert _main_sha(root) in err
        assert not list(data.rglob('*.json'))
        assert list(_pinned_trees_home(state_home).iterdir()) == []

    def test_an_unknown_sha_is_named_and_nothing_is_measured(
        self, tmp_path: Path, capfd: pytest.CaptureFixture[str]
    ) -> None:
        root = _repo(tmp_path)
        data = tmp_path / 'data'
        assert '0123456789abcdef0123' in _refused(root, data, capfd, '--sha', '0123456789abcdef0123')
        assert not data.exists()

    def test_a_missing_interpreter_is_named(
        self, tmp_path: Path, capfd: pytest.CaptureFixture[str]
    ) -> None:
        root = _repo(tmp_path)
        data = tmp_path / 'data'
        missing = tmp_path / 'missing-python'
        assert str(missing) in _refused(root, data, capfd, python=str(missing))
        assert not data.exists()

    def test_an_unreadable_existing_snapshot_is_named_and_left_as_it_was(
        self, tmp_path: Path, capfd: pytest.CaptureFixture[str]
    ) -> None:
        root = _repo(tmp_path)
        data = tmp_path / 'data'
        existing = data / 'fixture' / f'{_main_sha(root)}.json'
        existing.parent.mkdir(parents=True)
        existing.write_text('not json\n', encoding='utf-8')
        assert str(existing) in _refused(root, data, capfd)
        assert existing.read_text(encoding='utf-8') == 'not json\n'

    def test_a_recorded_identity_unlike_the_prediction_names_both(
        self, tmp_path: Path, capfd: pytest.CaptureFixture[str]
    ) -> None:
        root = _repo(tmp_path)
        data = tmp_path / 'data'
        err = _refused(root, data, capfd, python=str(_shim(tmp_path, _MISPREDICTS)))
        recorded = snapshot.load_snapshot(data / 'fixture' / f'{_main_sha(root)}.json')['params']
        assert '"complexipy_version": "0.0.0"' in err
        assert json.dumps(recorded, sort_keys=True) in err

    def test_a_state_home_that_cannot_hold_the_pinned_tree_is_named(
        self, tmp_path: Path, capfd: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _repo(tmp_path)
        data = tmp_path / 'data'
        blocker = tmp_path / 'state-is-a-file'
        blocker.write_text('', encoding='utf-8')
        monkeypatch.setenv('XDG_STATE_HOME', str(blocker))
        assert str(blocker) in _refused(root, data, capfd)
        assert not data.exists()

    def test_a_tree_left_behind_is_named_below_the_instruments_cause(
        self, tmp_path: Path, capfd: pytest.CaptureFixture[str], state_home: Path
    ) -> None:
        root = _repo(tmp_path)
        data = tmp_path / 'data'
        assert _cadence(root, data, python=str(_shim(tmp_path, _LOCKS_AND_FAILS))) == 2
        err = capfd.readouterr().err
        [left_behind] = _pinned_trees_home(state_home).iterdir()
        assert err.index('shim: the measurement failed') < err.index(str(left_behind))

    def test_a_project_id_that_is_not_a_path_segment_is_an_argparse_exit_2(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(SystemExit) as raised:
            cadence.main(['--root', str(tmp_path), '--project-id', '../x'])
        assert raised.value.code == 2


def test_the_real_entry_point_measures_and_writes(tmp_path: Path, state_home: Path) -> None:
    root = _repo(tmp_path)
    data = tmp_path / 'data'
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith('GIT_') or key == 'GIT_CEILING_DIRECTORIES'
    }
    env['XDG_STATE_HOME'] = str(state_home)
    proc = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / 'scripts' / 'quality_snapshot_cadence.py'),
            '--root', str(root), '--project-id', 'fixture', '--python', sys.executable,
            '--data-dir', str(data),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (data / 'fixture' / f'{_main_sha(root)}.json').is_file()
