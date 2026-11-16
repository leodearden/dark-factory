"""TaskWorkflow._review dispatches the configured reviewer generation.

The worktree is a real git checkout in the 5695 shape: a task branch with
main merged into it after its creation-time base was stored. The
context-complete generation must brief from the branch's own change (a
fresh fork point), the plan on disk and the checkout's conventions; the
incumbent diff-only generation briefs from the same fresh fork point.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _merged_in_branch import MergedInBranch, branch_with_main_merged_in, diffed_paths
from _workflow_helpers import FakeBriefing, _Fixture, _make, _make_git_ops
from test_reviewer_verdict_routing import _invoke_writes_review_verdict

from orchestrator.agents.review_briefing import ReviewBriefing, ReviewConvention, ReviewSubject
from orchestrator.agents.roles import reviewers_for

CONVENTIONS_RELPATH = 'review/briefing.yaml'


CONVENTIONS_YAML = (
    'conventions:\n'
    '  - rule: "Keep reviews cheap"\n'
    '    why: "Every finding costs a cycle."\n'
)


class _Spied:
    def __init__(self, f: _Fixture, monkeypatch: pytest.MonkeyPatch):
        fake = FakeBriefing()
        self.context_complete = AsyncMock(side_effect=fake.build_context_complete_reviewer_prompt)
        self.diff_only = AsyncMock(side_effect=fake.build_reviewer_prompt)
        monkeypatch.setattr(fake, 'build_context_complete_reviewer_prompt', self.context_complete)
        monkeypatch.setattr(fake, 'build_reviewer_prompt', self.diff_only)
        f.wf.briefing = fake

    def subject(self) -> ReviewSubject:
        assert self.context_complete.await_args is not None
        return self.context_complete.await_args.args[1]


def _setup(
    tmp_path: Path, briefing: ReviewBriefing, monkeypatch: pytest.MonkeyPatch,
) -> tuple[_Fixture, _Spied, MergedInBranch]:
    merged = branch_with_main_merged_in(
        tmp_path, base_files={CONVENTIONS_RELPATH: CONVENTIONS_YAML},
    )
    f = _make(worktree=merged.worktree, project_root=tmp_path / 'proj')
    config = f.wf.config
    config.review.briefing_path = CONVENTIONS_RELPATH
    config.review.reviewer_briefing = briefing
    config.reviewer_stagger_secs = 0
    config.max_reviewer_retries = 0
    monkeypatch.setattr(
        f.wf, '_invoke',
        AsyncMock(side_effect=_invoke_writes_review_verdict(f, verdict='PASS', summary='ok')),
    )
    return f, _Spied(f, monkeypatch), merged


def _mirror(f: _Fixture) -> dict:
    return json.loads((f.artifacts.artifact_path('reviews') / 'reviewer_comprehensive.json').read_text())


@pytest.mark.asyncio
class TestContextCompleteDispatch:
    async def test_the_subject_is_the_branch_s_own_change(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        f, spied, _ = _setup(tmp_path, ReviewBriefing.CONTEXT_COMPLETE, monkeypatch)
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
        monkeypatch: pytest.MonkeyPatch,
    ):
        f, spied, _ = _setup(tmp_path, ReviewBriefing.CONTEXT_COMPLETE, monkeypatch)
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

    async def test_a_failed_gather_is_an_error_verdict_naming_the_cause(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        f, spied, _ = _setup(tmp_path, ReviewBriefing.CONTEXT_COMPLETE, monkeypatch)
        f.wf.config.git.main_branch = 'no-such-branch'
        stored_base = f.artifacts.read_base_commit()
        assert stored_base

        reviews = await f.wf._review()

        roster = [role.name for role in reviewers_for(ReviewBriefing.CONTEXT_COMPLETE)]
        assert sorted(reviews.reviewer_errors) == sorted(roster)
        assert spied.context_complete.await_count == 0
        assert spied.diff_only.await_count == 0
        mirror = _mirror(f)
        assert mirror['verdict'] == 'ERROR'
        assert stored_base in mirror['summary']

    async def test_an_absent_plan_is_stated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        f, spied, _ = _setup(tmp_path, ReviewBriefing.CONTEXT_COMPLETE, monkeypatch)

        await f.wf._review()

        subject = spied.subject()
        assert subject.plan is None
        assert subject.plan_absent_reason


@pytest.mark.asyncio
class TestDiffOnlyDispatch:
    async def test_the_incumbent_reviews_the_branch_s_own_change(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        f, spied, merged = _setup(tmp_path, ReviewBriefing.DIFF_ONLY, monkeypatch)
        f.artifacts.update_base_commit(merged.creation_sha)
        f.wf.git_ops.get_diff_from_base = _make_git_ops(merged.repo).get_diff_from_base

        await f.wf._review()

        assert spied.context_complete.await_count == 0
        assert spied.diff_only.await_count == len(reviewers_for(ReviewBriefing.DIFF_ONLY))
        assert spied.diff_only.await_args is not None
        role, diff = spied.diff_only.await_args.args
        assert role == 'reviewer_comprehensive'
        assert diffed_paths(diff) == {'a.py'}
        assert 'm1.py' not in diff


def test_the_verdict_cache_fingerprint_follows_the_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    f, _, _ = _setup(tmp_path, ReviewBriefing.DIFF_ONLY, monkeypatch)
    diff_only = f.wf._reviewer_config_fingerprint()

    f.wf.config.review.reviewer_briefing = ReviewBriefing.CONTEXT_COMPLETE
    context_complete = f.wf._reviewer_config_fingerprint()

    assert diff_only is not None
    assert context_complete is not None
    assert diff_only != context_complete
