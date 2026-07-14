"""Task beta: model allowlist + fail-fast validation + per-account
availability probe (routing.py, config.py).

Fixtures are kept MODULE-LOCAL (not conftest.py) -- a conftest.py edit trips
verify.py's has_conftest and forces the merge-time verify to fall back to
running the full owning-package suite instead of a scoped subset (mirrors
test_config_psi_admission_reload.py's stated rationale).
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from orchestrator.config import (
    RELOADABLE_FIELDS,
    ModelsConfig,
    OrchestratorConfig,
    RoutingConfig,
    apply_reload,
)
from orchestrator.routing import DEFAULT_ALLOWED_MODELS, FABLE_CANDIDATE_MODEL, probe_models
from shared.cli_invoke import AgentResult
from shared.config_models import AccountConfig


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


class TestRoutingAllowlistReloadDisposition:
    """routing.allowed_models is green-tier: hot-reloadable without a process
    restart (mirrors TestPsiAdmissionReloadDisposition in
    test_config_psi_admission_reload.py)."""

    @pytest.mark.parametrize('leaf', list(RoutingConfig.model_fields))
    def test_every_leaf_is_reloadable(self, leaf):
        assert f'routing.{leaf}' in RELOADABLE_FIELDS, (
            f'routing.{leaf!r} is expected to be green-tier reloadable but is '
            f'missing from RELOADABLE_FIELDS'
        )

    def test_widening_allowlist_applies(self, monkeypatch, tmp_path):
        """A reload that WIDENS routing.allowed_models (adds back a model
        already in use) applies cleanly: reloaded=True, the leaf lands in
        applied, live is updated in place."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig(
            models=ModelsConfig(architect='opus'),
            routing=RoutingConfig(allowed_models=['sonnet', 'opus']),
        )
        fresh = OrchestratorConfig(
            models=ModelsConfig(architect='opus'),
            routing=RoutingConfig(allowed_models=['haiku', 'sonnet', 'opus']),
        )
        report = apply_reload(live, fresh)
        assert report['reloaded'] is True
        assert report['error'] is None
        assert report['applied']['routing.allowed_models'] == {
            'old': ['sonnet', 'opus'], 'new': ['haiku', 'sonnet', 'opus'],
        }
        assert live.routing.allowed_models == ['haiku', 'sonnet', 'opus']

    def test_tightening_allowlist_below_in_use_model_rolls_back(
        self, monkeypatch, tmp_path
    ):
        """I5 hybrid-invariant rollback for the NEW cross-field invariant
        (_validate_models_in_allowlist), mirroring
        TestApplyReloadHybridRollback in test_config.py.

        Uses a SYNTHETIC allowlist containing ONLY 'routing.allowed_models'
        (omitting 'models.architect', which travels with it under the real
        RELOADABLE_FIELDS -- both are whole-submodel green-tier groups, so an
        end-to-end reload normally updates them together and never produces
        this hybrid). This is the routing analogue of PRD boundary scenario
        5 -- unreachable end-to-end under v1, so it is exercised here at the
        unit level via the injectable `allowlist` param, exactly like the
        steward-timeout precedent.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        # live is valid on its own: architect='opus' is in ['haiku','sonnet','opus'].
        live = OrchestratorConfig(
            models=ModelsConfig(architect='opus'),
            routing=RoutingConfig(allowed_models=['haiku', 'sonnet', 'opus']),
        )
        # fresh is valid on its own: EVERY ModelsConfig role (not just
        # architect) must be pinned to 'sonnet', the sole entry in
        # fresh.routing.allowed_models -- ModelsConfig has other roles that
        # default to 'opus' (implementer, debugger, merger, steward,
        # deep_reviewer), which would otherwise fail the allowlist
        # invariant during construction of `fresh` itself, before
        # apply_reload is ever reached.
        fresh = OrchestratorConfig(
            models=ModelsConfig(**{role: 'sonnet' for role in ModelsConfig.model_fields}),
            routing=RoutingConfig(allowed_models=['sonnet']),
        )
        live_dump_before = live.model_dump()
        # Applying ONLY routing.allowed_models (drop 'opus') while
        # models.architect stays at live's 'opus' produces an invalid hybrid
        # ('opus' no longer allowed) that OrchestratorConfig.model_validate
        # must reject.
        report = apply_reload(live, fresh, allowlist=frozenset({'routing.allowed_models'}))
        assert report['reloaded'] is False
        assert report['error'].startswith('hybrid-invariant')
        assert 'architect' in report['error']
        assert report['applied'] == {}
        # Rolled back byte-for-byte: routing.allowed_models restored;
        # models.architect was never touched (categorized restart_required
        # under the synthetic allowlist).
        assert live.model_dump() == live_dump_before
        assert live.routing.allowed_models == ['haiku', 'sonnet', 'opus']
        assert live.models.architect == 'opus'

    def test_typo_in_fresh_config_fails_closed_before_apply_reload_is_reached(
        self, monkeypatch, tmp_path
    ):
        """The 'typo'd model reload returns a structured error naming the
        field' user-observable signal (PRD boundary-test-11 shape).

        A real reload builds `fresh` by re-running load_config() against the
        (edited) on-disk YAML -- i.e. constructing a fresh OrchestratorConfig
        -- before ever calling apply_reload(). When an operator's edit
        introduces a models.<role> typo with routing.allowed_models
        unchanged, THAT construction itself raises ValidationError naming the
        field (see TestAllowlistFailFastValidation), so a hot-reload fails
        closed (I1) and never reaches apply_reload/live at all -- `live` is
        provably untouched because it is never even passed to apply_reload
        in this path. Harness.reload_config() (task gamma, harness.py:9667)
        is the thin orchestration layer that wraps exactly this raise into
        the {reloaded: False, error: ...} shape via a bare try/except; that
        wrapping is exercised by that module's own tests, not re-tested here.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        with pytest.raises(ValidationError) as exc_info:
            OrchestratorConfig(models=ModelsConfig(architect='sonnett'))
        assert 'architect' in str(exc_info.value)


# ---------------------------------------------------------------------------
# probe_models: per-account x per-model availability probe. invoke_fn and
# token_resolver are dependency-injected (mirrors invoke_with_cap_retry's
# invoke_fn= seam) so every scenario below runs network-free.
# ---------------------------------------------------------------------------


class _ScriptedProbeCli:
    """Keyed fake invoke_fn standing in for invoke_claude_agent: looks up a
    canned AgentResult by (model, oauth_token), defaulting to a successful
    result when no override is scripted for that pair -- so a test only
    needs to script the (model, token) combinations it actually cares
    about. Records every call's kwargs, in order, for the call-count /
    argument assertions below."""

    def __init__(self, overrides: dict[tuple[str, str], AgentResult] | None = None) -> None:
        self._overrides = overrides or {}
        self.calls: list[dict] = []

    async def __call__(self, **kwargs: object) -> AgentResult:
        self.calls.append(kwargs)
        key = (kwargs.get('model'), kwargs.get('oauth_token'))
        return self._overrides.get(key, AgentResult(success=True, output='ok'))


class TestProbeModelsTargetSet:
    """The probe's default target model set is dedup(allowed_models +
    [claude-fable-5]), order-preserving -- fable is probed even though it is
    NOT admitted to the runtime allowlist (task beta is the G3 gate that
    produces the fable-availability data task xi's admission gate later
    consumes; see this task's plan design_decisions)."""

    def test_default_target_set_is_allowed_models_plus_fable(self):
        accounts = [AccountConfig(name='max-x', oauth_token_env='MAX_X_TOKEN')]
        cli = _ScriptedProbeCli()

        report = asyncio.run(probe_models(
            accounts, ['haiku', 'sonnet'],
            invoke_fn=cli,
            token_resolver={'MAX_X_TOKEN': 'tok-x'}.get,
        ))

        assert report.models == ['haiku', 'sonnet', FABLE_CANDIDATE_MODEL]

    def test_explicit_models_argument_overrides_default(self):
        accounts = [AccountConfig(name='max-x', oauth_token_env='MAX_X_TOKEN')]
        cli = _ScriptedProbeCli()

        report = asyncio.run(probe_models(
            accounts, ['haiku', 'sonnet'],
            models=['opus'],
            invoke_fn=cli,
            token_resolver={'MAX_X_TOKEN': 'tok-x'}.get,
        ))

        assert report.models == ['opus']
        assert {call['model'] for call in cli.calls} == {'opus'}


class TestProbeModelsStatusMappingAndDispatch:
    """Per-account x per-model status uses classify_invocation's outcome --
    OK->available, ModelNotFound->unavailable, AuthFailed->auth_error,
    CapHit->capped -- and invoke_fn is dispatched accounts x models times
    with the right model and resolved token."""

    def test_status_mapping_and_call_count_and_arguments(self):
        accounts = [
            AccountConfig(name='max-x', oauth_token_env='MAX_X_TOKEN'),
            AccountConfig(name='max-y', oauth_token_env='MAX_Y_TOKEN'),
        ]
        token_map = {'MAX_X_TOKEN': 'tok-x', 'MAX_Y_TOKEN': 'tok-y'}
        not_found_body = (
            '{"type":"error","error":{"type":"not_found_error",'
            '"message":"model: sonnet"}}'
        )
        overrides = {
            ('sonnet', 'tok-x'): AgentResult(
                success=False, api_error_status=404, output=not_found_body,
            ),
            ('haiku', 'tok-y'): AgentResult(
                success=False, api_error_status=401, output='unauthorized',
            ),
            ('sonnet', 'tok-y'): AgentResult(
                success=False,
                output="You've hit your usage limit. Your plan resets in 3h.",
            ),
        }
        cli = _ScriptedProbeCli(overrides)

        report = asyncio.run(probe_models(
            accounts, ['haiku', 'sonnet'],
            invoke_fn=cli,
            token_resolver=token_map.get,
        ))

        assert report.accounts['max-x']['haiku'] == 'available'
        assert report.accounts['max-x']['sonnet'] == 'unavailable'
        assert report.accounts['max-x'][FABLE_CANDIDATE_MODEL] == 'available'
        assert report.accounts['max-y']['haiku'] == 'auth_error'
        assert report.accounts['max-y']['sonnet'] == 'capped'
        assert report.accounts['max-y'][FABLE_CANDIDATE_MODEL] == 'available'

        # accounts (2) x models (haiku, sonnet, claude-fable-5 = 3) == 6.
        assert len(cli.calls) == 6
        for call in cli.calls:
            assert call['model'] in {'haiku', 'sonnet', FABLE_CANDIDATE_MODEL}
            assert call['oauth_token'] in {'tok-x', 'tok-y'}


class TestProbeModelsMissingToken:
    """An account whose env token cannot be resolved yields a distinct
    'no_token' status for every target model, without ever calling
    invoke_fn for that account."""

    def test_missing_token_account_yields_no_token_status_without_invoking(self):
        accounts = [
            AccountConfig(name='max-x', oauth_token_env='MAX_X_TOKEN'),
            AccountConfig(name='max-missing', oauth_token_env='MAX_MISSING_TOKEN'),
        ]
        cli = _ScriptedProbeCli()

        report = asyncio.run(probe_models(
            accounts, ['haiku'],
            invoke_fn=cli,
            token_resolver={'MAX_X_TOKEN': 'tok-x'}.get,  # MAX_MISSING_TOKEN absent
        ))

        assert report.accounts['max-missing']['haiku'] == 'no_token'
        assert report.accounts['max-missing'][FABLE_CANDIDATE_MODEL] == 'no_token'
        # Only max-x's calls were dispatched (2 models: haiku + fable).
        assert len(cli.calls) == 2
        assert all(call['oauth_token'] == 'tok-x' for call in cli.calls)
