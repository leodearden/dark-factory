"""Tests for shared.git_read_rules, including the real-git facts its table rests on."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from shared.git_read_rules import GIT_FILE_WRITING_FLAGS, GitReadRules, git_read_rules
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_REVISION_ARGS = {
    'log': ['-1'],
    'show': ['-1'],
    'diff': ['HEAD~1'],
    'status': [],
    'rev-parse': ['HEAD'],
    'ls-files': [],
}

_MUTATING_SUBCOMMANDS = ['commit', 'branch', 'checkout', 'reset', 'push', 'config', 'tag', 'stash']


class TestRules:
    def test_allow_and_deny_follow_argument_order(self) -> None:
        rules = git_read_rules('log', 'show')
        assert rules == GitReadRules(
            allow=('Bash(git log:*)', 'Bash(git show:*)'),
            deny=('Bash(git log *--output*)', 'Bash(git show *--output*)'),
        )

    def test_subcommands_without_write_flags_deny_nothing(self) -> None:
        assert git_read_rules('status', 'rev-parse', 'ls-files').deny == ()

    @pytest.mark.parametrize('subcommand', _MUTATING_SUBCOMMANDS)
    def test_unknown_subcommand_is_refused_by_name(self, subcommand: str) -> None:
        with pytest.raises(ValueError) as exc_info:
            git_read_rules('log', subcommand)
        assert subcommand in str(exc_info.value)
        assert 'log' in str(exc_info.value)

    def test_rules_are_frozen(self) -> None:
        rules = git_read_rules('log')
        with pytest.raises(dataclasses.FrozenInstanceError):
            cast(Any, rules).allow = ()

    def test_flag_table_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            cast(Any, GIT_FILE_WRITING_FLAGS)['commit'] = ()


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)


@pytest.fixture
def two_commit_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    subprocess.run(['git', 'init', '-q', str(repo)], check=True, capture_output=True)
    for key, value in (
        ('user.email', 'git-read-rules-test@example.com'),
        ('user.name', 'Git Read Rules Test'),
        ('commit.gpgsign', 'false'),
    ):
        _git(repo, 'config', key, value)
    for n in (1, 2):
        (repo / 'f.txt').write_text(f'{n}\n')
        _git(repo, 'add', 'f.txt')
        _git(repo, 'commit', '-q', '-m', f'c{n}')
    return repo


class TestGitPremises:
    @pytest.mark.parametrize(
        ('subcommand', 'flag'),
        [(s, f) for s, flags in GIT_FILE_WRITING_FLAGS.items() for f in flags],
    )
    def test_listed_flag_writes_a_file(self, two_commit_repo: Path, tmp_path: Path, subcommand: str, flag: str) -> None:
        target = tmp_path / 'written'
        _git(two_commit_repo, subcommand, *_REVISION_ARGS[subcommand], f'{flag}={target}')
        assert target.exists()

    @pytest.mark.parametrize(
        ('subcommand', 'abbreviation'),
        [(s, f[:k]) for s, flags in GIT_FILE_WRITING_FLAGS.items() for f in flags for k in range(3, len(f))],
    )
    def test_abbreviated_flag_is_refused(
        self, two_commit_repo: Path, tmp_path: Path, subcommand: str, abbreviation: str,
    ) -> None:
        target = tmp_path / 'written'
        result = _git(two_commit_repo, subcommand, *_REVISION_ARGS[subcommand], f'{abbreviation}={target}')
        assert result.returncode != 0
        assert not target.exists()

    @pytest.mark.parametrize('subcommand', [s for s, flags in GIT_FILE_WRITING_FLAGS.items() if not flags])
    def test_unlisted_subcommand_does_not_write_through_output(
        self, two_commit_repo: Path, tmp_path: Path, subcommand: str,
    ) -> None:
        target = tmp_path / 'written'
        _git(two_commit_repo, subcommand, *_REVISION_ARGS[subcommand], f'--output={target}')
        assert not target.exists()
