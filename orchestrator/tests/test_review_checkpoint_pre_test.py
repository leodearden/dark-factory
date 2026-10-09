"""review.pre_test_commands: the config knobs and their wiring into the review checkpoint.

Fixtures are kept MODULE-LOCAL (not conftest.py) — a conftest.py edit trips
verify.py's has_conftest and forces the merge-time verify to fall back to
running the full owning-package suite instead of a scoped subset.
"""

from __future__ import annotations

import json
import logging
import shlex
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec, stamp_stock_routing_config
from _recording_event_store import _RecordingEventStore
from pydantic import ValidationError
from shared.cli_invoke import AgentResult, AllAccountsCappedException

from orchestrator.config import OrchestratorConfig, ReviewConfig, apply_reload
from orchestrator.review_checkpoint import ReviewCheckpoint, ReviewReport
from orchestrator.verify import VerifyResult


class TestReviewPreTestConfig:
    def test_defaults(self):
        cfg = ReviewConfig()
        assert cfg.pre_test_commands == []
        assert cfg.pre_test_timeout_secs == 1800.0

    def test_commands_preserve_order(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig(review=ReviewConfig(pre_test_commands=['a', 'b']))
        assert cfg.review.pre_test_commands == ['a', 'b']

    @pytest.mark.parametrize('bad', [0, -1])
    def test_non_positive_timeout_rejected(self, bad):
        with pytest.raises(ValidationError):
            ReviewConfig(pre_test_timeout_secs=bad)

    def test_apply_reload_hot_applies_pre_test_knobs(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig()
        fresh = OrchestratorConfig(review=ReviewConfig(
            pre_test_commands=['npm run build'], pre_test_timeout_secs=600.0,
        ))

        report = apply_reload(live, fresh)

        assert report['reloaded'] is True
        for path in ('review.pre_test_commands', 'review.pre_test_timeout_secs'):
            assert path in report['applied']
            assert path not in report['restart_required']
        assert live.review.pre_test_commands == ['npm run build']
        assert live.review.pre_test_timeout_secs == 600.0


_PASSING_PHASE1 = VerifyResult(
    passed=True, summary='OK', test_output='', lint_output='', type_output='',
)


def _make_checkpoint(tmp_path: Path, commands: list[str]) -> ReviewCheckpoint:
    """A ReviewCheckpoint over a spec_set config mock, mirroring test_review_checkpoint_cap.py.

    project_root stays Path('/tmp') because _run_review refuses '/tmp/pytest*'
    roots; reports land under the absolute tmp_path dir and the briefing path
    names a file that does not exist.
    """
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.project_root = Path('/tmp')
    config.models.deep_reviewer = 'opus'
    config.max_turns.deep_reviewer = 50
    config.budgets.deep_reviewer = 10.0
    config.effort.deep_reviewer = 'max'
    config.backends.deep_reviewer = 'claude'
    config.fused_memory.project_id = 'test-project'
    config.escalation.host = 'localhost'
    config.escalation.port = 9999
    config.verify_env = {}
    config.review.pre_test_commands = commands
    config.review.pre_test_timeout_secs = 30.0
    config.review.briefing_path = 'df-6020-no-such-briefing.yaml'
    config.review.reports_dir = str(tmp_path / 'reports')
    stamp_stock_routing_config(config)

    mcp = MagicMock()
    mcp.mcp_config_json.return_value = {'mcpServers': {}}

    cp = ReviewCheckpoint(config, mcp=mcp, usage_gate=None)
    cp.event_store = _RecordingEventStore()  # type: ignore[assignment]
    return cp


def _patch_reviewer(monkeypatch) -> AsyncMock:
    reviewer = AsyncMock(return_value=AgentResult(
        success=True, output='', subtype='success',
        structured_output={'findings': [], 'summary': 'ok'},
    ))
    monkeypatch.setattr('orchestrator.review_checkpoint.invoke_with_cap_retry', reviewer)
    return reviewer


def _patch_phase1(monkeypatch, phase1: AsyncMock) -> AsyncMock:
    monkeypatch.setattr('orchestrator.review_checkpoint.run_full_verification', phase1)
    return phase1


def _prompt_of(reviewer: AsyncMock) -> str:
    assert reviewer.await_args is not None
    return reviewer.await_args.kwargs['prompt']


def _saved_report(tmp_path: Path, report: ReviewReport) -> dict:
    path = tmp_path / 'reports' / f'review-{report.review_id}.json'
    return json.loads(path.read_text())


@pytest.mark.asyncio
class TestReviewCheckpointPreTest:
    async def test_commands_run_in_order_before_phase1_in_project_root(
        self, monkeypatch, tmp_path,
    ):
        log = tmp_path / 'setup.log'
        quoted = shlex.quote(str(log))
        commands = [
            f"printf 'one %s\\n' \"$PWD\" >> {quoted}",
            f"printf 'two\\n' >> {quoted}",
        ]
        snapshots: list[list[str]] = []

        async def _phase1(*_args, **_kwargs) -> VerifyResult:
            snapshots.append(log.read_text().splitlines() if log.exists() else [])
            return _PASSING_PHASE1

        _patch_reviewer(monkeypatch)
        _patch_phase1(monkeypatch, AsyncMock(side_effect=_phase1))

        report = await _make_checkpoint(tmp_path, commands).run_focused()

        assert snapshots == [['one /tmp', 'two']]
        assert len(report.pre_test) == 2

    async def test_empty_commands_is_a_noop(self, monkeypatch, tmp_path):
        reviewer = _patch_reviewer(monkeypatch)
        phase1 = _patch_phase1(monkeypatch, AsyncMock(return_value=_PASSING_PHASE1))

        report = await _make_checkpoint(tmp_path, []).run_focused()

        phase1.assert_awaited_once()
        assert report.pre_test == ()
        assert 'pre_test_commands' not in _saved_report(tmp_path, report)
        assert 'Pre-test' not in _prompt_of(reviewer)

    async def test_failing_command_does_not_abort_review_and_is_reported(
        self, monkeypatch, tmp_path, caplog,
    ):
        command = 'echo setup-broke-tail; exit 7'
        reviewer = _patch_reviewer(monkeypatch)
        phase1 = _patch_phase1(monkeypatch, AsyncMock(return_value=VerifyResult(
            passed=False, summary='e2e smokes red', test_output='stale bundle',
            lint_output='', type_output='',
        )))

        with caplog.at_level(logging.WARNING, logger='orchestrator.review_pre_test'):
            report = await _make_checkpoint(tmp_path, [command]).run_focused()

        phase1.assert_awaited_once()
        reviewer.assert_awaited_once()
        prompt = _prompt_of(reviewer)
        assert (
            prompt.index('## Phase 1 Results')
            < prompt.index('setup-broke-tail')
            < prompt.index('## Review Briefing')
        )
        assert command in prompt
        assert 'FAILED' in prompt

        saved = _saved_report(tmp_path, report)['pre_test_commands']
        assert len(saved) == 1
        assert saved[0]['command'] == command
        assert saved[0]['returncode'] == 7
        assert saved[0]['timed_out'] is False
        assert 'setup-broke-tail' in saved[0]['output_tail']

        assert any(
            command in r.getMessage()
            for r in caplog.records if r.levelno >= logging.WARNING
        )

    async def test_cap_exhausted_review_still_carries_pre_test(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(
            'orchestrator.review_checkpoint.invoke_with_cap_retry',
            AsyncMock(side_effect=AllAccountsCappedException(
                retries=1, elapsed_secs=1.0, label='x',
            )),
        )
        _patch_phase1(monkeypatch, AsyncMock(return_value=_PASSING_PHASE1))

        report = await _make_checkpoint(tmp_path, ['true']).run_focused()

        assert len(report.pre_test) == 1
        assert report.pre_test[0].failed is False
