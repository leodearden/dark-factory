"""A predecessor's delivered checks evaluated at one commit (early-start PRD §5.1, choice 1).

Driven against a real git repository: ``main`` plus a branch whose tip adds a marker file.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest
from _delivered_check_fixtures import install_delivered_check_script, script_check

from orchestrator.delivered_checks import ChecksAtCommit, evaluate_delivered_checks_at

pytestmark = pytest.mark.asyncio

_TOKEN = 'EARLY_START_MARKER_TOKEN_V1'
_MARKER = 'src/marker.py'


def _git_env() -> dict[str, str]:
    return {
        **os.environ,
        'GIT_CONFIG_GLOBAL': os.devnull,
        'GIT_CONFIG_NOSYSTEM': '1',
        'GIT_AUTHOR_NAME': 'Test',
        'GIT_AUTHOR_EMAIL': 'test@example.com',
        'GIT_COMMITTER_NAME': 'Test',
        'GIT_COMMITTER_EMAIL': 'test@example.com',
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(root), *args],
        check=True, capture_output=True, text=True, env=_git_env(),
    ).stdout.strip()


class _Repo:
    def __init__(self, root: Path) -> None:
        subprocess.run(
            ['git', 'init', '-b', 'main', str(root)],
            check=True, capture_output=True, text=True, env=_git_env(),
        )
        (root / 'README.md').write_text('# fixture\n', encoding='utf-8')
        _git(root, 'add', 'README.md')
        _git(root, 'commit', '-m', 'initial')
        self.root = root
        self.main_sha = _git(root, 'rev-parse', 'HEAD')
        _git(root, 'checkout', '-b', 'task/P')
        marker = root / _MARKER
        marker.parent.mkdir(parents=True)
        marker.write_text(f'{_TOKEN}\n', encoding='utf-8')
        _git(root, 'add', _MARKER)
        _git(root, 'commit', '-m', 'add marker')
        self.tip_sha = _git(root, 'rev-parse', 'HEAD')
        _git(root, 'checkout', 'main')


@pytest.fixture
def repo(tmp_path: Path) -> _Repo:
    return _Repo(tmp_path / 'repo')


def _grep(pattern: str, *, name: str = 'cap') -> dict:
    return {'name': name, 'kind': 'grep', 'pattern': pattern, 'expect': 'present'}


def _path(path: str) -> dict:
    return {'name': 'file', 'kind': 'path', 'paths': [path], 'expect': 'present'}


async def _evaluate(repo: _Repo, metadata: dict | None, *, commit: str | None = None, **kw):
    return await evaluate_delivered_checks_at(
        metadata,
        commit=commit or repo.tip_sha,
        project_root=repo.root,
        check_timeout_secs=kw.pop('check_timeout_secs', 30.0),
        **kw,
    )


@pytest.mark.parametrize('metadata', [None, {}, {'delivered_checks': []}])
async def test_no_declared_checks_passes(repo: _Repo, metadata: dict | None) -> None:
    assert await _evaluate(repo, metadata) is ChecksAtCommit.PASSED


async def test_a_grep_check_is_read_at_the_given_commit(repo: _Repo) -> None:
    metadata = {'delivered_checks': [_grep(_TOKEN)]}

    assert await _evaluate(repo, metadata) is ChecksAtCommit.PASSED
    assert await _evaluate(repo, metadata, commit=repo.main_sha) is not ChecksAtCommit.PASSED


async def test_a_grep_check_that_does_not_match_at_the_tip_fails(repo: _Repo) -> None:
    metadata = {'delivered_checks': [_grep('NOT_ANYWHERE_IN_THE_TREE')]}

    assert await _evaluate(repo, metadata) is ChecksAtCommit.FAILED


async def test_a_path_check_present_at_the_tip_passes(repo: _Repo) -> None:
    metadata = {'delivered_checks': [_path(_MARKER)]}

    assert await _evaluate(repo, metadata) is ChecksAtCommit.PASSED


async def test_a_script_check_is_never_run(repo: _Repo) -> None:
    ran = repo.root / 'script-ran'
    install_delivered_check_script(repo.root, 'checks/probe.sh', f'#!/bin/sh\ntouch {ran}\n')
    metadata = {
        'delivered_checks': [_grep(_TOKEN), script_check('probe', 'checks/probe.sh'), _path(_MARKER)],
    }

    assert await _evaluate(repo, metadata) is ChecksAtCommit.SCRIPT_KIND
    assert not ran.exists()


async def test_an_unreadable_entry_fails(repo: _Repo) -> None:
    metadata = {'delivered_checks': [_grep(_TOKEN), 'not-a-descriptor']}

    assert await _evaluate(repo, metadata) is ChecksAtCommit.FAILED


async def test_a_commit_git_cannot_resolve_fails(repo: _Repo) -> None:
    metadata = {'delivered_checks': [_grep(_TOKEN)]}

    assert await _evaluate(repo, metadata, commit='f' * 40) is ChecksAtCommit.FAILED


@pytest.mark.timing(reason='check_timeout_secs bounds a hung delivered check; ERRORED/timeout must read as FAILED')
async def test_a_check_that_outruns_its_timeout_fails(repo: _Repo) -> None:
    async def _hung_runner(argv, **_kwargs) -> tuple[int, str, str]:
        await asyncio.Event().wait()
        return (0, '', '')

    metadata = {'delivered_checks': [_grep(_TOKEN)]}

    verdict = await _evaluate(repo, metadata, check_timeout_secs=0.05, runner=_hung_runner)

    assert verdict is ChecksAtCommit.FAILED
