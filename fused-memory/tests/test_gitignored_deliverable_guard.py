"""Unit tests for fused_memory.middleware.gitignored_deliverable_guard (task 3611)."""

from __future__ import annotations

import shutil
import subprocess

import pytest
from _fm_helpers import _init_git_repo

from fused_memory.middleware.gitignored_deliverable_guard import make_gitignore_probe


def _require_git() -> None:
    if shutil.which('git') is None:
        pytest.skip('git is not available')


@pytest.fixture
def gitignore_repo(tmp_path):
    """A real repo ignoring tasks.db, *.log and .taskmaster/, plus a TRACKED tracked.log."""
    _require_git()
    _init_git_repo(tmp_path)
    (tmp_path / '.gitignore').write_text('tasks.db\n*.log\n.taskmaster/\n')
    (tmp_path / 'tracked.log').write_text('tracked despite the *.log rule\n')
    subprocess.run(
        ['git', '-C', str(tmp_path), 'add', '-f', '.gitignore', 'tracked.log'],
        check=True,
    )
    subprocess.run(
        [
            'git', '-C', str(tmp_path),
            '-c', 'user.email=t@e.example', '-c', 'user.name=T',
            'commit', '-q', '-m', 'ignore rules and a force-tracked log',
        ],
        check=True,
    )
    return tmp_path


class TestMakeGitignoreProbe:
    """The one impure adapter, pinned against real ``git check-ignore``."""

    def test_ignored_absent_path_is_reported(self, gitignore_repo):
        probe = make_gitignore_probe(gitignore_repo)
        assert probe(['tasks.db']) == frozenset({'tasks.db'})

    def test_unignored_path_yields_empty_set_not_none(self, gitignore_repo):
        result = make_gitignore_probe(gitignore_repo)(['src/foo.py'])
        assert result is not None
        assert result == frozenset()

    def test_mixed_declaration_reports_only_the_ignored_subset(self, gitignore_repo):
        result = make_gitignore_probe(gitignore_repo)(['tasks.db', 'src/foo.py'])
        assert result == frozenset({'tasks.db'})

    def test_tracked_path_matching_a_rule_counts_as_committable(self, gitignore_repo):
        result = make_gitignore_probe(gitignore_repo)(['tracked.log'])
        assert result == frozenset()

    def test_nested_and_absolute_spellings_echo_verbatim(self, gitignore_repo):
        absolute = str(gitignore_repo / 'tasks.db')
        result = make_gitignore_probe(gitignore_repo)(
            ['.taskmaster/tasks/tasks.db', absolute],
        )
        assert result == frozenset({'.taskmaster/tasks/tasks.db', absolute})

    def test_non_git_directory_fails_open_with_none(self, tmp_path):
        _require_git()
        not_a_repo = tmp_path / 'plain'
        not_a_repo.mkdir()
        assert make_gitignore_probe(not_a_repo)(['tasks.db']) is None

    def test_missing_directory_fails_open_with_none(self, tmp_path):
        _require_git()
        assert make_gitignore_probe(tmp_path / 'does-not-exist')(['tasks.db']) is None

    def test_path_outside_repo_fails_open_despite_partial_stdout(self, gitignore_repo):
        result = make_gitignore_probe(gitignore_repo)(['tasks.db', '/etc/hosts'])
        assert result is None

    def test_empty_declaration_yields_empty_set(self, gitignore_repo):
        assert make_gitignore_probe(gitignore_repo)([]) == frozenset()
