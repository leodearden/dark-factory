"""A degraded review checkpoint says so: a cap skip keeps its trigger, a missing briefing is announced.

Drives ``ReviewCheckpoint`` only through its public trigger and run seams
(``record_merge``, ``should_trigger``, ``should_run_full``, ``run_focused``,
``run_full``) against a real scratch directory, with Phase 1 and the deep
reviewer replaced by fakes.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec, stamp_stock_routing_config
from _recording_event_store import _RecordingEventStore
from shared.cli_invoke import AgentResult, AllAccountsCappedException

from orchestrator.config import OrchestratorConfig, ReviewConfig
from orchestrator.event_store import EventStore
from orchestrator.review_checkpoint import (
    NO_BRIEFING_NOTICE,
    ReviewCheckpoint,
    ReviewOutcome,
)
from orchestrator.verify import VerifyResult

_BRIEFING_PATH = 'review/briefing.yaml'
_YAML_FENCE = '```yaml'


@pytest.fixture
def review_root() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix='df-review-degraded-') as scratch:
        root = Path(scratch) / 'project'
        root.mkdir()
        assert '/tmp/pytest' not in str(root), (
            f'{root} contains "/tmp/pytest", which ReviewCheckpoint._run_review refuses '
            'as a project_root; this fixture must live outside pytest basetemp'
        )
        yield root


def _make_checkpoint(review_root: Path, tmp_path: Path) -> ReviewCheckpoint:
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.project_root = review_root
    config.module_configs_or_empty = {}
    config.review = ReviewConfig(
        interval=2,
        full_review_min_interval_secs=86400.0,
        full_review_min_tasks=2,
        briefing_path=_BRIEFING_PATH,
        reports_dir=str(tmp_path / 'reports'),
    )
    config.models.deep_reviewer = 'opus'
    config.max_turns.deep_reviewer = 50
    config.budgets.deep_reviewer = 10.0
    config.effort.deep_reviewer = 'max'
    config.backends.deep_reviewer = 'claude'
    config.fused_memory.project_id = 'test-project'
    config.escalation.host = 'localhost'
    config.escalation.port = 9999
    stamp_stock_routing_config(config)

    mcp = MagicMock()
    mcp.mcp_config_json.return_value = {'mcpServers': {}}

    cp = ReviewCheckpoint(config, mcp=mcp, usage_gate=None)
    cp.event_store = cast(EventStore, _RecordingEventStore())
    return cp


@pytest.fixture
def checkpoint(review_root: Path, tmp_path: Path) -> ReviewCheckpoint:
    return _make_checkpoint(review_root, tmp_path)


@pytest.fixture
def reviewer(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    reviewer = AsyncMock(return_value=AgentResult(
        success=True, output='', subtype='success',
        structured_output={'findings': [], 'summary': 'ok'},
    ))
    monkeypatch.setattr('orchestrator.review_checkpoint.invoke_with_cap_retry', reviewer)
    monkeypatch.setattr(
        'orchestrator.review_checkpoint.run_full_verification',
        AsyncMock(return_value=VerifyResult(
            passed=True, summary='OK', test_output='', lint_output='', type_output='',
        )),
    )
    return reviewer


def _cap(reviewer: AsyncMock) -> None:
    reviewer.side_effect = AllAccountsCappedException(
        retries=1, elapsed_secs=1.0, label='Review checkpoint [x]',
    )


def _uncap(reviewer: AsyncMock) -> None:
    reviewer.side_effect = None


def _prompt(reviewer: AsyncMock) -> str:
    assert reviewer.await_args is not None, 'the deep reviewer was never invoked'
    return reviewer.await_args.kwargs['prompt']


def _saved_report(tmp_path: Path, review_id: str) -> dict:
    return json.loads((tmp_path / 'reports' / f'review-{review_id}.json').read_text())


def _write_briefing(review_root: Path, content: str) -> None:
    path = review_root / _BRIEFING_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.mark.asyncio
class TestCapSkippedCheckpointIsDeferred:
    async def test_capped_focused_review_is_skipped_and_keeps_its_trigger(
        self, checkpoint: ReviewCheckpoint, reviewer: AsyncMock,
    ) -> None:
        checkpoint.record_merge(['pkg/a'])
        checkpoint.record_merge(['pkg/a'])
        assert checkpoint.should_trigger()
        _cap(reviewer)

        report = await checkpoint.run_focused()

        assert report.outcome is ReviewOutcome.SKIPPED_CAPPED
        assert checkpoint.should_trigger()

    async def test_next_focused_review_carries_the_skipped_modules(
        self, checkpoint: ReviewCheckpoint, reviewer: AsyncMock,
    ) -> None:
        checkpoint.record_merge(['pkg/a'])
        checkpoint.record_merge(['pkg/a'])
        _cap(reviewer)
        await checkpoint.run_focused()

        checkpoint.record_merge(['pkg/b'])
        _uncap(reviewer)
        report = await checkpoint.run_focused()

        assert report.outcome is ReviewOutcome.RAN
        assert report.changed_modules == ['pkg/a', 'pkg/b']
        assert 'pkg/a' in _prompt(reviewer)
        assert not checkpoint.should_trigger()

    async def test_capped_full_review_keeps_the_full_review_gate_open(
        self, checkpoint: ReviewCheckpoint, reviewer: AsyncMock,
    ) -> None:
        checkpoint.record_merge(['pkg/a'])
        checkpoint.record_merge(['pkg/b'])
        assert checkpoint.should_run_full(time.monotonic())
        _cap(reviewer)

        skipped = await checkpoint.run_full()

        assert skipped.outcome is ReviewOutcome.SKIPPED_CAPPED
        assert checkpoint.should_run_full(time.monotonic())

        _uncap(reviewer)
        ran = await checkpoint.run_full()

        assert ran.outcome is ReviewOutcome.RAN
        assert not checkpoint.should_run_full(time.monotonic())

    async def test_a_review_that_ran_persists_its_outcome(
        self, checkpoint: ReviewCheckpoint, reviewer: AsyncMock, tmp_path: Path,
    ) -> None:
        report = await checkpoint.run_focused()

        assert _saved_report(tmp_path, report.review_id)['outcome'] == 'ran'


@pytest.mark.asyncio
class TestMissingBriefingIsAnnounced:
    async def _assert_announced_as_missing(
        self, checkpoint: ReviewCheckpoint, reviewer: AsyncMock, tmp_path: Path,
    ) -> None:
        report = await checkpoint.run_focused()

        prompt = _prompt(reviewer)
        assert '## Review Briefing' in prompt
        assert NO_BRIEFING_NOTICE in prompt
        assert _YAML_FENCE not in prompt
        assert report.briefing_missing is True
        assert _saved_report(tmp_path, report.review_id)['briefing_missing'] is True

    async def test_absent_briefing_is_announced_outside_any_fence(
        self, checkpoint: ReviewCheckpoint, reviewer: AsyncMock, tmp_path: Path,
    ) -> None:
        await self._assert_announced_as_missing(checkpoint, reviewer, tmp_path)

    async def test_present_briefing_is_fenced_as_yaml(
        self, checkpoint: ReviewCheckpoint, reviewer: AsyncMock, review_root: Path,
        tmp_path: Path,
    ) -> None:
        _write_briefing(review_root, 'conventions: [x]\n')

        report = await checkpoint.run_focused()

        prompt = _prompt(reviewer)
        assert f'{_YAML_FENCE}\nconventions: [x]' in prompt
        assert NO_BRIEFING_NOTICE not in prompt
        assert report.briefing_missing is False
        assert _saved_report(tmp_path, report.review_id)['briefing_missing'] is False

    async def test_unreadable_briefing_is_treated_as_missing(
        self, checkpoint: ReviewCheckpoint, reviewer: AsyncMock, review_root: Path,
        tmp_path: Path,
    ) -> None:
        (review_root / _BRIEFING_PATH).mkdir(parents=True)

        await self._assert_announced_as_missing(checkpoint, reviewer, tmp_path)

    async def test_whitespace_only_briefing_is_treated_as_missing(
        self, checkpoint: ReviewCheckpoint, reviewer: AsyncMock, review_root: Path,
        tmp_path: Path,
    ) -> None:
        _write_briefing(review_root, '  \n\t\n')

        await self._assert_announced_as_missing(checkpoint, reviewer, tmp_path)

    async def test_capped_review_without_a_briefing_still_flags_it_missing(
        self, checkpoint: ReviewCheckpoint, reviewer: AsyncMock,
    ) -> None:
        _cap(reviewer)

        report = await checkpoint.run_focused()

        assert report.outcome is ReviewOutcome.SKIPPED_CAPPED
        assert report.briefing_missing is True
