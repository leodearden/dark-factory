"""The reviewer-trial ``campaign`` command owns its account-pool gate: one
gate built for the whole campaign, threaded to every trial, and torn down
even when a trial raises (task 3288). A ``--pool`` request that yields no gate
refuses to run rather than silently running ungated.

Both ``UsageGate`` bindings a construction could read are patched to one
factory, so the pins hold whichever route builds the gate.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from shared.testing import make_gate_mock

import orchestrator.evals.reviewer_trial.__main__ as trial_cli

_NOT_CALLED = object()


class _CampaignHarness:
    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, usage_cap_enabled: bool = True,
    ):
        self.gate = make_gate_mock()
        self.factory = MagicMock(return_value=self.gate)
        monkeypatch.setattr('shared.usage_gate.UsageGate', self.factory)
        monkeypatch.setattr('orchestrator.evals.runner.UsageGate', self.factory)
        monkeypatch.setattr(trial_cli, '_load_corpus', lambda: SimpleNamespace(diffs=[]))
        self.trial_saw = _NOT_CALLED

        async def failing_trial(*_args, usage_gate=None, **_kwargs):
            self.trial_saw = usage_gate
            raise RuntimeError('trial blew up')

        monkeypatch.setattr('orchestrator.evals.reviewer_trial.runner.run_trial', failing_trial)
        self.config_path = tmp_path / 'orchestrator.yaml'
        self.config_path.write_text(
            f'project_root: {tmp_path}\nusage_cap:\n  enabled: {str(usage_cap_enabled).lower()}\n',
        )
        self.results_dir = tmp_path / 'results'

    def invoke(self, pool_flag: str):
        return CliRunner().invoke(trial_cli.cli, [
            'campaign', '--results-dir', str(self.results_dir), '--trials', '1',
            '--split', 'all', pool_flag, '--config', str(self.config_path),
        ])


def test_pooled_campaign_tears_its_gate_down_when_a_trial_raises(monkeypatch, tmp_path):
    h = _CampaignHarness(monkeypatch, tmp_path)

    result = h.invoke('--pool')

    h.factory.assert_called_once()
    h.gate.check_at_startup.assert_awaited_once()
    assert h.trial_saw is h.gate, f'the trial was handed {h.trial_saw!r}, not the campaign gate'
    assert isinstance(result.exception, RuntimeError), (
        f'expected the trial failure to surface, got {result.exception!r}\n{result.output}'
    )
    h.gate.shutdown.assert_awaited_once()


def test_unpooled_campaign_builds_no_gate(monkeypatch, tmp_path):
    h = _CampaignHarness(monkeypatch, tmp_path)

    result = h.invoke('--no-pool')

    h.factory.assert_not_called()
    assert h.trial_saw is None, f'the trial was handed {h.trial_saw!r}, expected an explicit None'
    assert isinstance(result.exception, RuntimeError), (
        f'expected the trial failure to surface, got {result.exception!r}\n{result.output}'
    )


def _assert_refused_to_run_ungated(h: _CampaignHarness, result) -> None:
    assert isinstance(result.exception, SystemExit) and result.exit_code == 1, (
        f'expected a usage error, got exit {result.exit_code} / {result.exception!r}\n{result.output}'
    )
    assert '--no-pool' in result.output, f'the refusal does not name the remedy:\n{result.output}'
    assert h.trial_saw is _NOT_CALLED, 'an ungated trial ran although --pool was requested'


def test_a_pool_request_whose_gate_fails_to_build_refuses_to_run(monkeypatch, tmp_path):
    h = _CampaignHarness(monkeypatch, tmp_path)
    h.factory.side_effect = RuntimeError('pool unavailable')

    _assert_refused_to_run_ungated(h, h.invoke('--pool'))


def test_a_pool_request_against_a_disabled_usage_cap_refuses_to_run(monkeypatch, tmp_path):
    h = _CampaignHarness(monkeypatch, tmp_path, usage_cap_enabled=False)

    _assert_refused_to_run_ungated(h, h.invoke('--pool'))
    h.factory.assert_not_called()
