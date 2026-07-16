"""Tests for the chronic-flake auto-file feature (task 2358).

Policy (Leo, 2026-07-08, verify-flakiness survey follow-up): after a verify
completes, detect a chronic pool-infra flake from reify's ``run_all.sh``
FLAKY substrate (reify task 5142) and auto-file a medium-priority De-flake
fix task into the project's task tree — non-blocking (the gate stays
green), with dedup + a 7-day rate limit.

Covers:
- OrchestratorConfig ChronicFlakeConfig submodel defaults + defaults.yaml
  round-trip (step-1/step-2)
- CHRONIC-FLAKY marker line-anchored parsing (step-3/step-4)
- Flaky ledger read + chronic-test computation (step-5/step-6)
- De-flake fix-task argument builder (step-7/step-8)
- FilingLedger rate-limit persistence (step-9/step-10)
- maybe_file_chronic_flake_tasks happy-path/dedup/rate-limit (step-11/step-12)
- Non-blocking guarantee (step-13/step-14)
- SchedulerChronicFlakeTaskClient adapter (step-15/step-16)
"""

from __future__ import annotations

from importlib import resources as pkg_resources

import pytest
import yaml

from orchestrator.config import OrchestratorConfig


def _load_package_defaults() -> dict:
    """Read the shipped defaults.yaml so tests stay in sync automatically."""
    defaults_file = pkg_resources.files('orchestrator') / 'defaults.yaml'
    return yaml.safe_load(defaults_file.read_text())


# ── Step-1 / Step-2: Config default tests ─────────────────────────────────────


class TestChronicFlakeConfigDefaults:
    """OrchestratorConfig exposes a ``ChronicFlakeConfig`` submodel with the
    reify-sourced defaults, shipped OFF (``enabled: false``) until reify:5142
    lands and is confirmed on the target project's main."""

    def test_pydantic_default_enabled_is_false(self):
        from orchestrator.config import ChronicFlakeConfig
        field_info = ChronicFlakeConfig.model_fields['enabled']
        assert field_info.default is False

    def test_pydantic_default_threshold(self):
        from orchestrator.config import ChronicFlakeConfig
        field_info = ChronicFlakeConfig.model_fields['threshold']
        assert field_info.default == 3

    def test_pydantic_default_window(self):
        from orchestrator.config import ChronicFlakeConfig
        field_info = ChronicFlakeConfig.model_fields['window']
        assert field_info.default == 20

    def test_pydantic_default_rate_limit_days(self):
        from orchestrator.config import ChronicFlakeConfig
        field_info = ChronicFlakeConfig.model_fields['rate_limit_days']
        assert field_info.default == 7

    def test_pydantic_default_ledger_relpath(self):
        from orchestrator.config import ChronicFlakeConfig
        field_info = ChronicFlakeConfig.model_fields['ledger_relpath']
        assert field_info.default == 'data/verify-logs/flaky-ledger.jsonl'

    def test_reachable_as_orchestrator_config_attribute(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        assert config.chronic_flake.enabled is False
        assert config.chronic_flake.threshold == 3
        assert config.chronic_flake.window == 20
        assert config.chronic_flake.rate_limit_days == 7
        assert config.chronic_flake.ledger_relpath == 'data/verify-logs/flaky-ledger.jsonl'

    def test_defaults_yaml_block_round_trips(self, monkeypatch, tmp_path):
        """The shipped defaults.yaml declares the same chronic_flake: block
        explicitly (including enabled: false) so the feature is discoverable
        and retunable in orchestrator.yaml without guessing at Pydantic
        defaults, mirroring the git:/psi_admission: precedent."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        defaults = _load_package_defaults()
        assert 'chronic_flake' in defaults
        block = defaults['chronic_flake']
        assert block['enabled'] is False
        assert config.chronic_flake.enabled == block['enabled']
        assert config.chronic_flake.threshold == block['threshold']
        assert config.chronic_flake.window == block['window']
        assert config.chronic_flake.rate_limit_days == block['rate_limit_days']
        assert config.chronic_flake.ledger_relpath == block['ledger_relpath']
