"""Tests for orchestrator.agents.agent_run_home, the redirected-HOME mechanics shared by the third-party CLI backends."""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
from _operator_home_fixture import OPERATOR_EMAIL, OPERATOR_IGNORED_FILE, write_operator_home

from orchestrator.agents.agent_run_home import (
    private_run_home,
    redirected_home_env,
    require_operator_home,
    write_owner_only_json,
)

OPERATOR_HOME = Path('/home/operator')


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class TestRequireOperatorHome:

    def test_the_operator_home_is_the_inherited_home(self):
        assert require_operator_home(
            {'HOME': '/home/operator', 'PATH': '/bin'}, backend='mimo',
        ) == OPERATOR_HOME

    @pytest.mark.parametrize('backend', ['mimo', 'zcode'])
    @pytest.mark.parametrize('base_env', [{}, {'HOME': ''}, {'HOME': '   '}])
    def test_a_missing_or_blank_home_is_refused_naming_home_and_the_backend(
        self, base_env, backend,
    ):
        with pytest.raises(ValueError) as excinfo:
            require_operator_home(base_env, backend=backend)
        assert 'HOME' in str(excinfo.value)
        assert backend in str(excinfo.value)


class TestPrivateRunHome:

    def test_the_home_is_a_private_directory_with_tmp_and_a_gitconfig(self, tmp_path):
        with private_run_home(backend='mimo', operator_home=tmp_path / 'op') as root:
            assert root.is_dir()
            assert _mode(root) == 0o700
            assert root.name.startswith('mimo-home-')
            assert (root / 'tmp').is_dir()
            assert (root / '.gitconfig').exists()

    def test_each_invocation_gets_a_fresh_home(self):
        with (
            private_run_home(backend='mimo', operator_home=OPERATOR_HOME) as first,
            private_run_home(backend='mimo', operator_home=OPERATOR_HOME) as second,
        ):
            assert first != second

    def test_the_home_is_removed_after_the_block(self):
        with private_run_home(backend='mimo', operator_home=OPERATOR_HOME) as root:
            pass
        assert not root.exists()

    def test_the_home_is_removed_when_the_block_raises(self):
        with (
            pytest.raises(RuntimeError, match='body failed'),
            private_run_home(backend='mimo', operator_home=OPERATOR_HOME) as root,
        ):
            raise RuntimeError('body failed')
        assert not root.exists()

    @pytest.mark.skipif(os.geteuid() == 0, reason='root ignores directory permissions')
    def test_a_home_that_cannot_be_removed_is_a_warning_not_an_error(self, caplog):
        locked: Path | None = None
        try:
            with (
                caplog.at_level(logging.WARNING, logger='orchestrator.agents.agent_run_home'),
                private_run_home(backend='mimo', operator_home=OPERATOR_HOME) as root,
            ):
                locked = root / 'locked'
                locked.mkdir()
                (locked / 'held').write_text('x')
                locked.chmod(0o500)
            assert str(root) in caplog.text
        finally:
            if locked is not None and locked.exists():
                locked.chmod(0o700)
                shutil.rmtree(locked.parent, ignore_errors=True)


class TestWriteOwnerOnlyJson:

    def test_writes_the_document_readable_by_the_owner_only(self, tmp_path):
        path = tmp_path / 'a' / 'b' / 'doc.json'
        write_owner_only_json(path, {'k': ['v', 1]})
        assert json.loads(path.read_text()) == {'k': ['v', 1]}
        assert _mode(path) == 0o600

    def test_an_existing_file_is_refused(self, tmp_path):
        path = tmp_path / 'doc.json'
        path.write_text('{}')
        with pytest.raises(FileExistsError):
            write_owner_only_json(path, {'k': 'v'})


class TestRedirectedHomeEnv:

    def test_home_and_tmpdir_point_into_the_root_and_path_is_kept(self, tmp_path):
        env = redirected_home_env(
            {'PATH': '/usr/local/bin:/usr/bin', 'HOME': '/home/operator'},
            root=tmp_path, operator_home=OPERATOR_HOME,
        )
        assert env['HOME'] == str(tmp_path)
        assert env['USERPROFILE'] == str(tmp_path)
        assert env['TMPDIR'] == str(tmp_path / 'tmp')
        assert env['PATH'] == '/usr/local/bin:/usr/bin'

    def test_the_operator_tool_homes_default_under_the_operator_home(self, tmp_path):
        env = redirected_home_env(
            {'HOME': '/home/operator'}, root=tmp_path, operator_home=OPERATOR_HOME,
        )
        assert env['XDG_CONFIG_HOME'] == '/home/operator/.config'
        assert env['XDG_CACHE_HOME'] == '/home/operator/.cache'
        assert env['XDG_DATA_HOME'] == '/home/operator/.local/share'
        assert env['XDG_STATE_HOME'] == '/home/operator/.local/state'
        assert env['CARGO_HOME'] == '/home/operator/.cargo'
        assert env['RUSTUP_HOME'] == '/home/operator/.rustup'

    def test_an_inherited_tool_home_wins_and_a_blank_one_falls_back(self, tmp_path):
        env = redirected_home_env(
            {'HOME': '/home/operator', 'CARGO_HOME': '/opt/cargo', 'XDG_CONFIG_HOME': ''},
            root=tmp_path, operator_home=OPERATOR_HOME,
        )
        assert env['CARGO_HOME'] == '/opt/cargo'
        assert env['XDG_CONFIG_HOME'] == '/home/operator/.config'

    def test_every_secret_is_dropped(self, tmp_path):
        secrets = {
            'FOO_API_KEY': 'k1', 'FOO_TOKEN': 'k2', 'CLAUDE_CODE_OAUTH_TOKEN': 'k3',
            'CLAUDE_OAUTH_TOKEN_A': 'k4', 'ANTHROPIC_BASE_URL': 'https://x.test/anthropic',
            'ANTHROPIC_AUTH_TOKEN': 'k5',
        }
        env = redirected_home_env(
            {'HOME': '/home/operator', 'PATH': '/usr/bin', **secrets},
            root=tmp_path, operator_home=OPERATOR_HOME,
        )
        for name, value in secrets.items():
            assert name not in env
            assert value not in env.values()


def _git(repo: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['git', *args], cwd=repo, env=env, capture_output=True, text=True, check=False,
    )


def _repo(root: Path) -> Path:
    root.mkdir()
    subprocess.run(['git', 'init', '-q', str(root)], check=True, env={
        'PATH': os.environ['PATH'], 'HOME': str(root), 'GIT_CONFIG_NOSYSTEM': '1',
    })
    (root / 'kept.txt').write_text('kept\n')
    (root / OPERATOR_IGNORED_FILE).write_text('dropped\n')
    return root


def _operator_env(op: Path) -> dict[str, str]:
    return {'PATH': os.environ['PATH'], 'HOME': str(op), 'GIT_CONFIG_NOSYSTEM': '1'}


class TestRedirectedHomeGit:

    @pytest.mark.parametrize('home_name', ['operator', 'op#1;x', 'op"q\\z'])
    def test_git_sees_the_operator_identity_and_excludes(self, tmp_path, home_name):
        op = write_operator_home(tmp_path / home_name)
        repo = _repo(tmp_path / 'repo')
        with private_run_home(backend='mimo', operator_home=op) as root:
            env = redirected_home_env(_operator_env(op), root=root, operator_home=op)
            email = _git(repo, env, 'config', 'user.email')
            staged = _git(repo, env, 'add', '-A', '--dry-run')
        assert email.returncode == 0, email.stderr
        assert email.stdout.strip() == OPERATOR_EMAIL
        assert staged.returncode == 0, staged.stderr
        assert 'kept.txt' in staged.stdout
        assert OPERATOR_IGNORED_FILE not in staged.stdout

    def test_an_operator_home_without_a_gitconfig_is_not_an_error(self, tmp_path):
        op = write_operator_home(tmp_path / 'operator', with_gitconfig=False)
        repo = _repo(tmp_path / 'repo')
        with private_run_home(backend='mimo', operator_home=op) as root:
            status = _git(
                repo, redirected_home_env(_operator_env(op), root=root, operator_home=op), 'status',
            )
        assert status.returncode == 0, status.stderr
