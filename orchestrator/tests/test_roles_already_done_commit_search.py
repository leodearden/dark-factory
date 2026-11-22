"""Real-git pins for the architect's already-done commit search (task 5836).

The ARCHITECT "Work is already on main" exit used to prescribe
``git log --all --oneline -- <path>`` and "grepping recent merge commits",
neither scoped to main nor to the task.  The prescribed search is now one
named constant, exercised here against real git.  The prompt's prose is
deliberately not pinned; see ``orchestrator/tests/test_roles_ancestry_check.py``'s
module docstring.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

import pytest
from _ladder_git import commit, git, git_rc, new_repo

from orchestrator.agents.roles import ALREADY_DONE_COMMIT_SEARCH, ARCHITECT

SYMBOL = 'def wanted'
PATH = 'mod.py'


@dataclass(frozen=True)
class SearchHistory:
    """A main whose ``mod.py`` gained SYMBOL twice, checked out on an unmerged sibling."""

    root: Path
    sha_side: str
    sha_feat: str
    sha_later: str


@pytest.fixture
def history(tmp_path: Path) -> SearchHistory:
    root = tmp_path / 'repo'
    new_repo(root)
    commit(root, 'base', PATH)

    git(root, 'checkout', '-b', 'task/8')
    sha_side = commit(root, f'side: {SYMBOL}', PATH)

    git(root, 'checkout', 'main')
    git(root, 'checkout', '-b', 'task/7')
    sha_feat = commit(root, f'feat: {SYMBOL}', PATH)

    git(root, 'checkout', 'main')
    git(root, 'merge', '--no-ff', '-m', 'Merge task/7 into main', 'task/7')

    (root / PATH).write_text(f'{SYMBOL}\n{SYMBOL}\n')
    git(root, 'add', PATH)
    git(root, 'commit', '-m', 'later: a second definition')
    sha_later = git(root, 'rev-parse', 'HEAD')

    git(root, 'checkout', 'task/8')
    return SearchHistory(root=root, sha_side=sha_side, sha_feat=sha_feat, sha_later=sha_later)


def _run_search(root: Path) -> list[str]:
    rendered = ALREADY_DONE_COMMIT_SEARCH.replace('<symbol>', SYMBOL).replace('<path>', PATH)
    output = git(root, *shlex.split(rendered)[1:])
    return [line.split()[0] for line in output.splitlines()]


def test_search_lists_only_main_commits_oldest_first(history: SearchHistory) -> None:
    listed = _run_search(history.root)

    assert listed == [history.sha_feat, history.sha_later]
    assert history.sha_side not in listed
    for sha in listed:
        assert git_rc(history.root, 'merge-base', '--is-ancestor', sha, 'main') == 0


def test_all_refs_listing_surfaces_the_unmerged_sibling_commit(history: SearchHistory) -> None:
    listed = git(history.root, 'log', '--all', '--format=%H', '--', PATH).splitlines()

    assert history.sha_side in listed


def test_architect_prompt_prescribes_the_search() -> None:
    assert ALREADY_DONE_COMMIT_SEARCH in ARCHITECT.system_prompt
