"""TaskWorkflow._review dispatches the configured reviewer generation.

The worktree is a real git checkout in the 5695 shape: a task branch with
main merged into it after its creation-time base was stored. The
context-complete generation must brief from the branch's own change (a
fresh fork point), the plan on disk and the checkout's conventions; the
incumbent keeps today's stored-base diff.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _workflow_helpers import FakeBriefing, _Fixture, _make
from test_reviewer_verdict_routing import _invoke_writes_review_verdict

from orchestrator.agents.review_briefing import ReviewBriefing, ReviewConvention, ReviewSubject
from orchestrator.agents.roles import reviewers_for

CONVENTIONS_RELPATH = 'review/briefing.yaml'


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ['git', *args], cwd=repo, check=True, capture_output=True, text=True,
    )
    return done.stdout.strip()


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '-m', message)
    return _git(repo, 'rev-parse', 'HEAD')


def _branch_with_main_merged_in(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'main')
    _git(repo, 'config', 'user.email', 'reviewer@example.com')
    _git(repo, 'config', 'user.name', 'Reviewer')
    _git(repo, 'config', 'commit.gpgsign', 'false')
    _commit(repo, {
        'a.py': 'a = 1\n',
        'm1.py': 'm1 = 1\n',
        CONVENTIONS_RELPATH: (
            'conventions:\n'
            '  - rule: "Keep reviews cheap"\n'
            '    why: "Every finding costs a cycle."\n'
        ),
    }, 'base')
    worktree = tmp_path / 'wt'
    _git(repo, 'worktree', 'add', '-q', '-b', 'task/50', str(worktree))
    _commit(worktree, {'a.py': 'a = 2\n'}, 'task work')
    _commit(repo, {'m1.py': 'm1 = 2\n'}, 'main advances')
    _git(worktree, 'merge', '-q', '--no-edit', 'main')
    return worktree


class _Spied:
    def __init__(self, f: _Fixture):
        fake = FakeBriefing()
        self.context_complete = AsyncMock(side_effect=fake.build_context_complete_reviewer_prompt)
        self.diff_only = AsyncMock(side_effect=fake.build_reviewer_prompt)
        fake.build_context_complete_reviewer_prompt = self.context_complete  # type: ignore[method-assign]
        fake.build_reviewer_prompt = self.diff_only  # type: ignore[method-assign]
        f.wf.briefing = fake

    def subject(self) -> ReviewSubject:
        assert self.context_complete.await_args is not None
        return self.context_complete.await_args.args[1]


def _setup(tmp_path: Path, briefing: ReviewBriefing) -> tuple[_Fixture, _Spied]:
    worktree = _branch_with_main_merged_in(tmp_path)
    f = _make(worktree=worktree, project_root=tmp_path / 'proj')
    config = f.wf.config
    config.git.main_branch = 'main'
    config.review.briefing_path = CONVENTIONS_RELPATH
    config.review.reviewer_briefing = briefing
    config.reviewer_stagger_secs = 0
    config.max_reviewer_retries = 0
    f.wf.git_ops.get_diff_from_base = AsyncMock(return_value='STORED-BASE DIFF')
    f.wf.git_ops.get_diff_from_main = AsyncMock(return_value='MAIN DIFF')
    f.wf._invoke = AsyncMock(  # type: ignore[method-assign]
        side_effect=_invoke_writes_review_verdict(f, verdict='PASS', summary='ok'),
    )
    return f, _Spied(f)


def _mirror(f: _Fixture) -> dict:
    return json.loads((f.artifacts.root / 'reviews' / 'reviewer_comprehensive.json').read_text())


@pytest.mark.asyncio
class TestContextCompleteDispatch:
    async def test_the_subject_is_the_branch_s_own_change(self, tmp_path: Path):
        f, spied = _setup(tmp_path, ReviewBriefing.CONTEXT_COMPLETE)
        f.artifacts.write_plan({'task_id': '50', 'title': 'T', 'analysis': 'Bump a.', 'steps': []})

        await f.wf._review()

        assert spied.context_complete.await_count == len(reviewers_for(ReviewBriefing.CONTEXT_COMPLETE))
        assert spied.diff_only.await_count == 0
        subject = spied.subject()
        assert [changed.path for changed in subject.change.files] == ['a.py']
        assert subject.task == f.wf.task
        assert subject.plan == f.artifacts.read_plan()
        assert subject.conventions == (
            ReviewConvention('Keep reviews cheap', 'Every finding costs a cycle.'),
        )
        assert _mirror(f)['reviewer'] == 'reviewer_comprehensive'
        assert _mirror(f)['verdict'] == 'PASS'

    async def test_a_malformed_conventions_file_degrades_loudly(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        f, spied = _setup(tmp_path, ReviewBriefing.CONTEXT_COMPLETE)
        assert f.wf.worktree is not None
        conventions_file = f.wf.worktree / CONVENTIONS_RELPATH
        conventions_file.write_text('conventions: [unclosed\n')

        with caplog.at_level(logging.WARNING):
            await f.wf._review()

        assert spied.subject().conventions == ()
        assert any(
            record.levelno == logging.WARNING and str(conventions_file) in record.getMessage()
            for record in caplog.records
        )

    async def test_an_absent_plan_is_stated(self, tmp_path: Path):
        f, spied = _setup(tmp_path, ReviewBriefing.CONTEXT_COMPLETE)

        await f.wf._review()

        subject = spied.subject()
        assert subject.plan is None
        assert subject.plan_absent_reason


@pytest.mark.asyncio
class TestDiffOnlyDispatch:
    async def test_the_incumbent_reviews_the_stored_base_diff(self, tmp_path: Path):
        f, spied = _setup(tmp_path, ReviewBriefing.DIFF_ONLY)

        await f.wf._review()

        assert spied.context_complete.await_count == 0
        assert spied.diff_only.await_count == len(reviewers_for(ReviewBriefing.DIFF_ONLY))
        assert spied.diff_only.await_args is not None
        args, _ = spied.diff_only.await_args
        assert args == ('reviewer_comprehensive', 'STORED-BASE DIFF')


def test_the_verdict_cache_fingerprint_follows_the_generation(tmp_path: Path):
    f, _ = _setup(tmp_path, ReviewBriefing.DIFF_ONLY)
    diff_only = f.wf._reviewer_config_fingerprint()

    f.wf.config.review.reviewer_briefing = ReviewBriefing.CONTEXT_COMPLETE
    context_complete = f.wf._reviewer_config_fingerprint()

    assert diff_only is not None
    assert context_complete is not None
    assert diff_only != context_complete
