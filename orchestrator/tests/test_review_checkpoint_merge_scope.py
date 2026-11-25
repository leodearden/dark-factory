"""A review checkpoint's merge scope is fixed when the review starts.

Merges that land while a review runs belong to the next review.  Drives a real
``ReviewCheckpoint`` only through its public seams (``record_merge``,
``should_trigger``, ``should_run_full``, ``run_focused``, ``run_full``) with a
real config, and Phase 1 and the deep reviewer replaced by fakes.  The reviewer
fake records the merges a test queued as landing mid-review before it answers.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _review_checkpoint_root import review_scratch_root
from shared.cli_invoke import AgentResult, AllAccountsCappedException

from orchestrator.config import OrchestratorConfig, ReviewConfig
from orchestrator.review_checkpoint import ReviewCheckpoint, ReviewOutcome
from orchestrator.verify import VerifyResult

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures('code_default_config')]


class _Reviewer:
    """The deep reviewer: lands the merges queued for this review, then answers or caps."""

    def __init__(self, checkpoint: ReviewCheckpoint) -> None:
        self._checkpoint = checkpoint
        self._landing: list[list[str]] = []
        self._caps_next = False

    def lands_during_next_review(self, *merges: list[str]) -> None:
        self._landing.extend(merges)

    def caps_next_review(self) -> None:
        self._caps_next = True

    async def __call__(self, **_: object) -> AgentResult:
        landing, self._landing = self._landing, []
        for modules in landing:
            self._checkpoint.record_merge(modules)
        if self._caps_next:
            self._caps_next = False
            raise AllAccountsCappedException(
                retries=1, elapsed_secs=1.0, label='Review checkpoint [x]',
            )
        return AgentResult(
            success=True, output='', subtype='success',
            structured_output={'findings': [], 'summary': 'ok'},
        )


@pytest.fixture
def checkpoint(tmp_path: Path) -> Iterator[ReviewCheckpoint]:
    with review_scratch_root('df-review-merge-scope-') as root:
        config = OrchestratorConfig(
            project_root=root,
            review=ReviewConfig(
                interval=2,
                full_review_min_interval_secs=0.0,
                full_review_min_tasks=2,
                reports_dir=str(tmp_path / 'reports'),
            ),
        )
        mcp = MagicMock()
        mcp.mcp_config_json.return_value = {'mcpServers': {}}
        yield ReviewCheckpoint(config, mcp=mcp, usage_gate=None)


@pytest.fixture
def reviewer(checkpoint: ReviewCheckpoint, monkeypatch: pytest.MonkeyPatch) -> _Reviewer:
    reviewer = _Reviewer(checkpoint)
    monkeypatch.setattr('orchestrator.review_checkpoint.invoke_with_cap_retry', reviewer)
    monkeypatch.setattr(
        'orchestrator.review_checkpoint.run_full_verification',
        AsyncMock(return_value=VerifyResult(
            passed=True, summary='OK', test_output='', lint_output='', type_output='',
        )),
    )
    return reviewer


async def test_a_merge_that_lands_during_a_review_counts_toward_the_next(
    checkpoint: ReviewCheckpoint, reviewer: _Reviewer,
) -> None:
    checkpoint.record_merge(['pkg/a'])
    checkpoint.record_merge(['pkg/a'])
    assert checkpoint.should_trigger()
    reviewer.lands_during_next_review(['pkg/late'])

    report = await checkpoint.run_focused()

    assert report.ran
    assert report.changed_modules == ['pkg/a']
    assert not checkpoint.should_trigger()
    checkpoint.record_merge(['pkg/b'])
    assert checkpoint.should_trigger()


async def test_the_next_focused_review_covers_modules_merged_during_the_last(
    checkpoint: ReviewCheckpoint, reviewer: _Reviewer,
) -> None:
    checkpoint.record_merge(['pkg/a'])
    checkpoint.record_merge(['pkg/a'])
    reviewer.lands_during_next_review(['pkg/late'])
    assert (await checkpoint.run_focused()).ran
    checkpoint.record_merge(['pkg/b'])

    report = await checkpoint.run_focused()

    assert report.ran
    assert report.changed_modules == ['pkg/late', 'pkg/b']


async def test_a_module_merged_again_during_a_review_stays_in_scope(
    checkpoint: ReviewCheckpoint, reviewer: _Reviewer,
) -> None:
    checkpoint.record_merge(['pkg/a'])
    checkpoint.record_merge(['pkg/a'])
    reviewer.lands_during_next_review(['pkg/a'])
    assert (await checkpoint.run_focused()).ran
    checkpoint.record_merge(['pkg/b'])

    report = await checkpoint.run_focused()

    assert report.changed_modules == ['pkg/a', 'pkg/b']


async def test_a_merge_during_a_full_review_counts_toward_the_next_full_review(
    checkpoint: ReviewCheckpoint, reviewer: _Reviewer,
) -> None:
    checkpoint.record_merge(['pkg/a'])
    checkpoint.record_merge(['pkg/b'])
    assert checkpoint.should_run_full(time.monotonic())
    reviewer.lands_during_next_review(['pkg/late'])

    report = await checkpoint.run_full()

    assert report.ran
    assert not checkpoint.should_run_full(time.monotonic())
    checkpoint.record_merge(['pkg/c'])
    assert checkpoint.should_run_full(time.monotonic())
    assert checkpoint.should_trigger()


async def test_a_capped_review_leaves_every_merge_unreviewed(
    checkpoint: ReviewCheckpoint, reviewer: _Reviewer,
) -> None:
    checkpoint.record_merge(['pkg/a'])
    checkpoint.record_merge(['pkg/a'])
    reviewer.lands_during_next_review(['pkg/late'])
    reviewer.caps_next_review()

    capped = await checkpoint.run_focused()

    assert capped.outcome is ReviewOutcome.SKIPPED_CAPPED
    report = await checkpoint.run_focused()
    assert report.ran
    assert report.changed_modules == ['pkg/a', 'pkg/late']
