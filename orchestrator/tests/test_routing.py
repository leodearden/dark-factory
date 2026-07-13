"""Task beta: model allowlist + fail-fast validation + per-account
availability probe (routing.py, config.py).

Fixtures are kept MODULE-LOCAL (not conftest.py) -- a conftest.py edit trips
verify.py's has_conftest and forces the merge-time verify to fall back to
running the full owning-package suite instead of a scoped subset (mirrors
test_config_psi_admission_reload.py's stated rationale).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchestrator.config import OrchestratorConfig, RoutingConfig
from orchestrator.routing import DEFAULT_ALLOWED_MODELS


class TestRoutingConfigDefaults:
    """RoutingConfig is attached to OrchestratorConfig with the routing.py
    allowlist default (mirrors the ModelsConfig/PsiAdmissionConfig submodel
    pattern, config.py:141/473)."""

    def test_default_allowed_models_attached_on_orchestrator_config(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig()
        assert isinstance(cfg.routing, RoutingConfig)
        assert cfg.routing.allowed_models == list(DEFAULT_ALLOWED_MODELS)


class TestAllowlistFailFastValidation:
    """A configured model string outside routing.allowed_models must raise a
    structured, field-named pydantic.ValidationError at load (mirrors
    _validate_steward_timeout_invariant, config.py:2633)."""

    def test_model_outside_allowlist_raises_naming_field_and_value(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        with pytest.raises(ValidationError) as exc_info:
            OrchestratorConfig(models={'architect': 'sonnett'})
        message = str(exc_info.value)
        assert 'architect' in message
        assert 'sonnett' in message

    def test_all_in_allowlist_config_constructs_ok(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig(models={'architect': 'haiku'})
        assert cfg.models.architect == 'haiku'

    def test_unblock_auto_model_outside_allowlist_raises(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        with pytest.raises(ValidationError) as exc_info:
            OrchestratorConfig(unblock_auto={'model': 'bogus-model-9'})
        message = str(exc_info.value)
        assert 'unblock_auto' in message or 'model' in message
        assert 'bogus-model-9' in message


class TestNonClaudeBackendScopeBoundary:
    """The allowlist validator is SCOPED to claude-backend roles: a role
    running on a non-claude backend (the harness-backend axis) must never be
    rejected against the claude-centric allowlist."""

    def test_non_claude_backend_model_is_not_checked_against_allowlist(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig(
            backends={'reviewer': 'gemini'},
            models={'reviewer': 'gemini-2.5-pro'},
        )
        assert cfg.models.reviewer == 'gemini-2.5-pro'
        assert cfg.backends.reviewer == 'gemini'
