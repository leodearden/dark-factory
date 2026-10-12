"""Every agent invocation that requests the sandbox runs its git with rerere off.

rerere writes resolutions into the common ``.git/rr-cache``, which the sandbox
write set leaves read-only; half-on rerere leaves a dangling MERGE_RR record and
a stale MERGE_RR.lock behind (task 5844, docs/shared-repo-git-maintenance.md).

The assertions read what real git resolves, never the GIT_CONFIG_* spellings:
those names are git's contract, so a plausible but wrong spelling must fail here.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest
from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

from orchestrator.agents import sandbox_dispatch
from orchestrator.agents.invoke import invoke_agent
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


_SUCCESS_JSON = json.dumps({
    'result': 'ok',
    'is_error': False,
    'subtype': 'success',
    'cost_usd': 0.0,
    'duration_ms': 1,
    'num_turns': 1,
    'session_id': 's',
})


@pytest.fixture
def fake_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[[], list[str]]:
    """Put claude/codex/gemini/pi on PATH, each recording the rerere config git reads."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    dump = tmp_path / 'git-dump'
    # No stdin read: gemini and pi inherit stdin, so a `cat` could block.
    body = (
        '#!/bin/sh\n'
        '{ git config --get rerere.enabled; git config --get rerere.autoupdate; }'
        ' > "$FAKE_AGENT_GIT_DUMP"\n'
        f"printf '%s\\n' '{_SUCCESS_JSON}'\n"
    )
    for name in ('claude', 'codex', 'gemini', 'pi'):
        script = bin_dir / name
        script.write_text(body)
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ.get("PATH", "")}')
    monkeypatch.setenv('FAKE_AGENT_GIT_DUMP', str(dump))
    for var in ('DF_AGENT_CPU_NICE', 'DF_AGENT_CPU_GOVERN'):
        monkeypatch.delenv(var, raising=False)

    return lambda: dump.read_text().splitlines()


@contextmanager
def _backend_none() -> Iterator[None]:
    """Sandbox requested, but the backend resolves to none: argv runs unwrapped."""
    sandbox_dispatch.set_backend('none')
    yield


@contextmanager
def _identity_sandbox() -> Iterator[None]:
    """Reach the sandboxed claude sub-path through public sandbox_dispatch names only."""
    with (
        patch('orchestrator.agents.sandbox_dispatch.resolve_active_backend', return_value='bwrap'),
        patch(
            'orchestrator.agents.sandbox_dispatch.wrap_command',
            side_effect=lambda cmd, *args, **kwargs: cmd,
        ),
    ):
        yield


_REPO_CONFIG = ['true', 'true']
_SANDBOX_CONFIG = ['false', 'false']


class TestSandboxedDispatchGitConfig:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('backend', 'sandbox_modules', 'context', 'expected'),
        [
            pytest.param('claude', None, nullcontext, _REPO_CONFIG, id='claude-unsandboxed-control'),
            pytest.param('claude', [], _backend_none, _SANDBOX_CONFIG, id='claude-backend-none'),
            pytest.param('claude', [], _identity_sandbox, _SANDBOX_CONFIG, id='claude-sandboxed'),
            pytest.param('codex', [], _backend_none, _SANDBOX_CONFIG, id='codex'),
            pytest.param('gemini', [], _backend_none, _SANDBOX_CONFIG, id='gemini'),
            pytest.param('pi', [], _backend_none, _SANDBOX_CONFIG, id='pi'),
        ],
    )
    async def test_agent_git_reads_rerere_off_only_when_sandbox_requested(
        self,
        fake_agents: Callable[[], list[str]],
        rerere_repo: Path,
        backend: str,
        sandbox_modules: list[str] | None,
        context: Callable[[], AbstractContextManager[None]],
        expected: list[str],
    ) -> None:
        overrides = {'X_PROBE': '1'} if backend in ('claude', 'pi') else None
        overrides_before = dict(overrides) if overrides is not None else None

        with context():
            result = await invoke_agent(
                prompt='hi',
                system_prompt='s',
                cwd=rerere_repo,
                backend=backend,
                model='opus',
                max_turns=1,
                max_budget_usd=0.01,
                sandbox_modules=sandbox_modules,
                env_overrides=overrides,
            )

        if backend == 'claude':
            assert result.success, f'fake claude dispatch did not succeed: {result!r}'
        assert fake_agents() == expected, (
            f'{backend} with sandbox_modules={sandbox_modules!r}: the agent child must read '
            f'rerere {expected} — off whenever the sandbox is requested, the repo config '
            'otherwise. Apply sandbox_git_config_env in orchestrator/src/orchestrator/'
            'agents/invoke.py at every backend that reaches wrap_command.'
        )
        assert 'GIT_CONFIG_COUNT' not in os.environ, (
            "the policy belongs to the child env only; it must not leak into the "
            "orchestrator's own process environment."
        )
        assert overrides == overrides_before, "the caller's env_overrides must not be mutated"
