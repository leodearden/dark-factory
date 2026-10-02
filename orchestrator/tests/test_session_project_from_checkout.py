"""SessionRecord.project's cwd default resolves to the enclosing git checkout's MAIN working tree -- task 5181."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

from orchestrator import session_hooks as sh
from orchestrator import session_registry as sr

_SPAWN_ENV_VARS = (
    'CLAUDE_SPAWN_ROLE',
    'CLAUDE_SPAWN_PROJECT',
    'CLAUDE_SPAWN_TASK_ID',
    'CLAUDE_SPAWN_ESCALATION_ID',
    'CLAUDE_SPAWN_TITLE',
    'CLAUDE_SPAWN_PROMPT',
    'CLAUDE_SPAWN_SESSION_ID',
    'CLAUDE_SPAWN_PARENT_ID',
    'CLAUDE_SPAWN_LAUNCHER_PID',
    'CLAUDE_SPAWN_CWD',
)


@pytest.fixture(autouse=True)
def _hermetic_spawn_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Clear inherited spawn identity and pin the fleet root to this test.

    Module-local on purpose: test_session_registry.py exercises the unset
    fleet-root default, so this must not live in conftest.py.
    """
    for name in _SPAWN_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path / 'fleet'))


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True)


def _init_repo(path: Path) -> Path:
    """``git init`` a fresh repo at *path* with one commit.  Creates its own
    target, so it cannot escape into an enclosing repo.
    """
    path.mkdir(parents=True)
    _git(path, 'init', '-q', '-b', 'main')
    _git(path, '-c', 'user.name=t', '-c', 'user.email=t@example.com', 'commit', '-q', '--allow-empty', '-m', 'init')
    return path


def _add_worktree(repo: Path, path: Path, branch: str) -> Path:
    _git(repo, 'worktree', 'add', '-q', str(path), '-b', branch)
    return path


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cwd_project(cwd: Path | str) -> str:
    return sr.parse_spawn_identity(env={}, title='', prompt='', cwd=str(cwd)).project


@pytest.fixture
def acme(tmp_path: Path) -> Path:
    return _init_repo(tmp_path / 'acme')


# ---------------------------------------------------------------------------
# The cwd default names the enclosing checkout's main working tree
# ---------------------------------------------------------------------------


def test_repo_subdirectory_resolves_to_repo(acme: Path) -> None:
    assert _cwd_project(_mkdir(acme / 'orchestrator' / 'src')) == 'acme'


def test_repo_root_resolves_to_repo(acme: Path) -> None:
    assert _cwd_project(acme) == 'acme'


def test_nested_linked_worktree_root_resolves_to_main_checkout(acme: Path) -> None:
    worktree = _add_worktree(acme, acme / '.worktrees' / '4924', 'task/4924')
    assert _cwd_project(worktree) == 'acme'


def test_nested_linked_worktree_subdirectory_resolves_to_main_checkout(acme: Path) -> None:
    worktree = _add_worktree(acme, acme / '.worktrees' / '4924', 'task/4924')
    assert _cwd_project(_mkdir(worktree / 'shared')) == 'acme'


def test_sibling_linked_worktree_resolves_to_main_checkout(acme: Path, tmp_path: Path) -> None:
    worktree = _add_worktree(acme, tmp_path / 'acme-fix10', 'fix10')
    assert _cwd_project(worktree) == 'acme'


def test_directory_outside_any_repo_keeps_its_basename(tmp_path: Path) -> None:
    assert _cwd_project(_mkdir(tmp_path / 'scratch' / 'outside')) == 'outside'


@pytest.mark.parametrize(
    'gitfile_content',
    [
        pytest.param('not a gitfile\n', id='garbled'),
        pytest.param('gitdir: {tmp_path}/gone/.git/worktrees/x\n', id='missing-gitdir'),
    ],
)
def test_unresolvable_gitfile_falls_back_to_the_checkout_holding_it(tmp_path: Path, gitfile_content: str) -> None:
    odd = _mkdir(tmp_path / 'odd')
    (odd / '.git').write_text(gitfile_content.format(tmp_path=tmp_path))
    assert _cwd_project(_mkdir(odd / 'sub')) == 'odd'


def test_relative_cwd_is_not_resolved_against_the_process_cwd(acme: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mkdir(acme / 'orchestrator')
    monkeypatch.chdir(acme)
    assert _cwd_project('orchestrator') == 'orchestrator'


def test_env_and_title_still_win_over_the_checkout(acme: Path) -> None:
    cwd = str(_mkdir(acme / 'orchestrator'))
    from_env = sr.parse_spawn_identity(env={'CLAUDE_SPAWN_PROJECT': 'other'}, title='', prompt='', cwd=cwd)
    from_title = sr.parse_spawn_identity(env={}, title='prd:df attention-rail', prompt='', cwd=cwd)
    assert from_env.project == 'other'
    assert from_title.project == 'df'


# ---------------------------------------------------------------------------
# Both write paths file the session under the repo
# ---------------------------------------------------------------------------


def test_launching_names_the_record_for_the_main_checkout(
    acme: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    worktree = _add_worktree(acme, acme / '.worktrees' / '4924', 'task/4924')
    monkeypatch.setenv('CLAUDE_SPAWN_CWD', str(_mkdir(worktree / 'shared')))
    monkeypatch.setenv('CLAUDE_SPAWN_LAUNCHER_PID', '4242')
    fleet = tmp_path / 'fleet'

    rc = sr.main(['launching'])

    assert rc == 0
    slug = sr.build_session_slug('session', 'acme', None, 4242)
    assert Path(capsys.readouterr().out.strip()).name == slug
    assert sr.read_record(slug, root=fleet).project == 'acme'


def test_hand_launched_hooks_converge_on_one_record_named_for_the_repo(acme: Path, tmp_path: Path) -> None:
    fleet = tmp_path / 'fleet'
    start_cwd = _mkdir(acme / 'orchestrator')
    stop_cwd = _mkdir(acme / 'shared')

    sh.run_session_start({'session_id': 'sess-5181', 'cwd': str(start_cwd)}, env={}, root=fleet)
    sh.run_stop({'session_id': 'sess-5181', 'cwd': str(stop_cwd)}, env={}, root=fleet)

    slug = sr.build_session_slug('session', 'acme', None, 'sess-5181')  # type: ignore[arg-type]
    assert [entry.name for entry in sr.sessions_dir(fleet).iterdir()] == [slug]
    record = sr.read_record(slug, root=fleet)
    assert record.project == 'acme'
    assert record.status == sr.Status.IDLE
