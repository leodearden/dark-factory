"""Every agent invocation that requests the sandbox runs its git with rerere off.

rerere writes resolutions into the common ``.git/rr-cache``, which the sandbox
write set leaves read-only; half-on rerere leaves a dangling MERGE_RR record and
a stale MERGE_RR.lock behind (task 5844, docs/shared-repo-git-maintenance.md).

The assertions read what real git resolves, never the GIT_CONFIG_* spellings:
those names are git's contract, so a plausible but wrong spelling must fail here.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

from orchestrator.agents.sandbox_git_config import SANDBOX_GIT_CONFIG, sandbox_git_config_env


def _git_ok(repo: Path, *args: str) -> None:
    assert_isolated_git_repo(repo)
    proc = subprocess.run(
        ['git', *args], cwd=str(repo), capture_output=True, text=True,
        env=git_env_with_ceiling(repo),
    )
    assert proc.returncode == 0, f'git {" ".join(args)} failed: {proc.stderr}'


def _git_config_get(repo: Path, key: str, extra_env: Mapping[str, str]) -> str:
    assert_isolated_git_repo(repo)
    proc = subprocess.run(
        ['git', 'config', '--get', key], cwd=str(repo), capture_output=True, text=True,
        env={**git_env_with_ceiling(repo), **extra_env},
    )
    return proc.stdout.strip()


@pytest.fixture
def rerere_repo(tmp_path: Path) -> Path:
    """An isolated repo whose own config turns rerere fully on."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    init = subprocess.run(
        ['git', 'init', '-q', '-b', 'main', '.'],
        cwd=str(repo), capture_output=True, text=True,
        env=git_env_with_ceiling(repo),
    )
    assert init.returncode == 0, f'git init failed: {init.stderr}'
    assert_isolated_git_repo(repo)

    _git_ok(repo, 'config', 'rerere.enabled', 'true')
    _git_ok(repo, 'config', 'rerere.autoupdate', 'true')
    _git_ok(repo, 'config', 'user.name', 'Repo Name')
    _git_ok(repo, 'config', 'user.email', 'sandbox-git-config@test.invalid')
    return repo


class TestSandboxGitConfigEnv:
    def test_sandbox_config_overrides_repo_local_rerere(self, rerere_repo: Path) -> None:
        entries = sandbox_git_config_env({})

        assert _git_config_get(rerere_repo, 'rerere.enabled', entries) == 'false'
        assert _git_config_get(rerere_repo, 'rerere.autoupdate', entries) == 'false'
        for key, value in SANDBOX_GIT_CONFIG:
            assert _git_config_get(rerere_repo, key, entries) == value, (
                f'git must read {key}={value} from the entries built for SANDBOX_GIT_CONFIG'
            )

    def test_inherited_git_config_entries_survive(self, rerere_repo: Path) -> None:
        inherited = {
            'GIT_CONFIG_COUNT': '1',
            'GIT_CONFIG_KEY_0': 'user.name',
            'GIT_CONFIG_VALUE_0': 'Inherited Name',
        }
        env = {**inherited, **sandbox_git_config_env(inherited)}

        assert _git_config_get(rerere_repo, 'user.name', env) == 'Inherited Name', (
            'an inherited GIT_CONFIG_COUNT entry must be appended after, never overwritten'
        )
        assert _git_config_get(rerere_repo, 'rerere.enabled', env) == 'false'

    @pytest.mark.parametrize('bad_count', ['two', '-1'])
    def test_malformed_inherited_count_is_refused(self, bad_count: str) -> None:
        with pytest.raises(ValueError) as excinfo:
            sandbox_git_config_env({'GIT_CONFIG_COUNT': bad_count})

        assert 'GIT_CONFIG_COUNT' in str(excinfo.value)
        assert bad_count in str(excinfo.value)

    def test_inherited_mapping_not_mutated(self) -> None:
        inherited = {
            'GIT_CONFIG_COUNT': '1',
            'GIT_CONFIG_KEY_0': 'user.name',
            'GIT_CONFIG_VALUE_0': 'Inherited Name',
        }
        before = dict(inherited)

        sandbox_git_config_env(inherited)

        assert inherited == before
