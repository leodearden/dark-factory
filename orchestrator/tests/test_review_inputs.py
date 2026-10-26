"""The context-complete reviewer's inputs, gathered from a real checkout.

Every test drives ``orchestrator/review_inputs.py`` against real git
repositories under ``tmp_path``: the subject is git's own answer, so a fake
would only test the fake.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from orchestrator.agents.review_briefing import ChangedFile, ReviewConvention
from orchestrator.review_inputs import (
    collect_review_change,
    gather_review_subject,
    load_review_conventions,
    review_fork_point,
)


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ['git', *args], cwd=repo, check=True, capture_output=True, text=True,
    )
    return done.stdout.strip()


def _commit(repo: Path, files: dict[str, str | bytes], message: str) -> str:
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '-m', message)
    return _git(repo, 'rev-parse', 'HEAD')


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'main')
    _git(repo, 'config', 'user.email', 'reviewer@example.com')
    _git(repo, 'config', 'user.name', 'Reviewer')
    _git(repo, 'config', 'commit.gpgsign', 'false')
    _commit(repo, {'a.py': 'a = 1\n', 'm1.py': 'm1 = 1\n', 'm2.py': 'm2 = 1\n'}, 'base')
    return repo


@pytest.mark.asyncio
class TestFreshForkPoint:
    """The 5695 shape: main merged into the branch after its creation-time base was stored."""

    def _branch_with_main_merged_in(self, repo: Path, tmp_path: Path) -> tuple[Path, str, str]:
        creation_sha = _git(repo, 'rev-parse', 'HEAD')
        worktree = tmp_path / 'wt'
        _git(repo, 'worktree', 'add', '-q', '-b', 'task/x', str(worktree))
        _commit(worktree, {'a.py': 'a = 2\n'}, 'task work')
        main_tip = _commit(repo, {'m1.py': 'm1 = 2\n', 'm2.py': 'm2 = 2\n'}, 'main advances')
        _git(worktree, 'merge', '-q', '--no-edit', 'main')
        return worktree, creation_sha, main_tip

    async def test_the_fork_point_is_the_merged_main_tip(self, repo: Path, tmp_path: Path):
        worktree, _, main_tip = self._branch_with_main_merged_in(repo, tmp_path)

        assert await review_fork_point(worktree, 'main') == main_tip

    async def test_the_change_from_the_fork_point_is_the_task_work_alone(
        self, repo: Path, tmp_path: Path,
    ):
        worktree, _, _ = self._branch_with_main_merged_in(repo, tmp_path)
        fork_point = await review_fork_point(worktree, 'main')
        assert fork_point is not None

        change = await collect_review_change(worktree, fork_point)

        assert [changed.path for changed in change.files] == ['a.py']
        assert 'm1.py' not in change.diff
        assert 'm2.py' not in change.diff
        assert change.base_sha == fork_point

    async def test_the_stale_creation_base_would_review_mains_churn(
        self, repo: Path, tmp_path: Path,
    ):
        worktree, creation_sha, _ = self._branch_with_main_merged_in(repo, tmp_path)

        change = await collect_review_change(worktree, creation_sha)

        assert {'m1.py', 'm2.py'} <= {changed.path for changed in change.files}

    async def test_an_unresolvable_main_has_no_fork_point(self, repo: Path):
        assert await review_fork_point(repo, 'no-such-branch') is None


@pytest.mark.asyncio
class TestCollectReviewChange:
    async def test_numstat_counts_binary_and_spaced_paths(self, repo: Path):
        base = _git(repo, 'rev-parse', 'HEAD')
        _commit(repo, {
            'a.py': 'a = 10\nb = 2\nc = 3\n',
            'dir with space/file name.py': 'x = 1\ny = 2\n',
            'image.bin': b'\x00\x01\x02binary\x00payload',
        }, 'change')

        change = await collect_review_change(repo, base)

        assert set(change.files) == {
            ChangedFile('a.py', 3, 1),
            ChangedFile('dir with space/file name.py', 2, 0),
            ChangedFile('image.bin', None, None),
        }
        assert '+b = 2' in change.diff

    async def test_an_unknown_base_names_the_base_and_gits_stderr(self, repo: Path):
        base = 'f' * 40
        git_stderr = subprocess.run(
            ['git', 'diff', '--no-renames', f'{base}...HEAD'],
            cwd=repo, capture_output=True, text=True,
        ).stderr.strip()
        assert git_stderr

        with pytest.raises(RuntimeError) as raised:
            await collect_review_change(repo, base)

        assert base in str(raised.value)
        assert git_stderr in str(raised.value)


class TestLoadReviewConventions:
    def test_a_missing_file_has_no_conventions(self, tmp_path: Path):
        assert load_review_conventions(tmp_path / 'absent.yaml') == ()

    def test_conventions_load_in_order(self, tmp_path: Path):
        path = tmp_path / 'briefing.yaml'
        path.write_text(
            'project: demo\n'
            'conventions:\n'
            '  - rule: "Cite refs as path::symbol"\n'
            '    why: "Line pins drift."\n'
            '  - rule: "Never run git stash"\n'
            '    why: "refs/stash is shared."\n',
        )

        assert load_review_conventions(path) == (
            ReviewConvention('Cite refs as path::symbol', 'Line pins drift.'),
            ReviewConvention('Never run git stash', 'refs/stash is shared.'),
        )

    def test_a_file_without_conventions_has_none(self, tmp_path: Path):
        path = tmp_path / 'briefing.yaml'
        path.write_text('project: demo\nsubprojects: {}\n')

        assert load_review_conventions(path) == ()

    @pytest.mark.parametrize('text', [
        'conventions: [unclosed\n',
        'conventions: "a single string"\n',
        '- just\n- a list\n',
    ])
    def test_a_malformed_file_names_its_path(self, tmp_path: Path, text: str):
        path = tmp_path / 'briefing.yaml'
        path.write_text(text)

        with pytest.raises(ValueError) as raised:
            load_review_conventions(path)

        assert str(path) in str(raised.value)

    def test_an_entry_missing_its_rule_names_its_index(self, tmp_path: Path):
        path = tmp_path / 'briefing.yaml'
        path.write_text(
            'conventions:\n'
            '  - rule: "fine"\n'
            '    why: "fine"\n'
            '  - why: "no rule here"\n',
        )

        with pytest.raises(ValueError) as raised:
            load_review_conventions(path)

        assert str(path) in str(raised.value)
        assert 'conventions[1]' in str(raised.value)


@pytest.mark.asyncio
async def test_gather_review_subject_assembles_every_input(repo: Path):
    base = _git(repo, 'rev-parse', 'HEAD')
    _commit(repo, {'a.py': 'a = 3\n'}, 'task work')
    (repo / 'review').mkdir()
    (repo / 'review' / 'briefing.yaml').write_text(
        'conventions:\n  - rule: "Keep it small"\n    why: "Reviews stay cheap."\n',
    )
    task = {'id': '7', 'title': 'Bump a'}

    subject = await gather_review_subject(
        repo, base,
        task=task, plan=None, plan_absent_reason='no plan on disk',
        conventions_relpath='review/briefing.yaml',
    )

    assert subject.task == task
    assert subject.plan is None
    assert subject.plan_absent_reason == 'no plan on disk'
    assert subject.change.base_sha == base
    assert [changed.path for changed in subject.change.files] == ['a.py']
    assert subject.conventions == (ReviewConvention('Keep it small', 'Reviews stay cheap.'),)
