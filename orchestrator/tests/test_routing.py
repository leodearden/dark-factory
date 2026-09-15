"""Task beta: model allowlist + fail-fast validation + per-account
availability probe (routing.py, config.py).

Fixtures are kept MODULE-LOCAL (not conftest.py) -- a conftest.py edit trips
verify.py's has_conftest and forces the merge-time verify to fall back to
running the full owning-package suite instead of a scoped subset (mirrors
test_config_psi_admission_reload.py's stated rationale).
"""

from __future__ import annotations

import asyncio
from typing import cast

import pytest
import yaml
from click.testing import CliRunner
from pydantic import ValidationError
from shared.cli_invoke import AgentResult
from shared.config_models import AccountConfig, UsageCapConfig

import orchestrator.routing as routing_module
from orchestrator.cli import main
from orchestrator.config import (
    RELOADABLE_FIELDS,
    BackendsConfig,
    ModelsConfig,
    OrchestratorConfig,
    RoutingConfig,
    UnblockAutoConfig,
    apply_reload,
)
from orchestrator.routing import (
    DEFAULT_ALLOWED_MODELS,
    DEFAULT_PROBE_BUDGET_USD,
    FABLE_CANDIDATE_MODEL,
    ProbeReport,
    probe_models,
    render_probe_artifact,
)


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
            OrchestratorConfig(models=ModelsConfig(architect='sonnett'))
        message = str(exc_info.value)
        assert 'architect' in message
        assert 'sonnett' in message

    def test_all_in_allowlist_config_constructs_ok(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig(models=ModelsConfig(architect='haiku'))
        assert cfg.models.architect == 'haiku'

    def test_unblock_auto_model_outside_allowlist_raises(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        with pytest.raises(ValidationError) as exc_info:
            OrchestratorConfig(unblock_auto=UnblockAutoConfig(model='bogus-model-9'))
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
            backends=BackendsConfig(reviewer='gemini'),
            models=ModelsConfig(reviewer='gemini-2.5-pro'),
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
        # module_tagger defaults to 'haiku' (task 2540 flip); pin it to an
        # already-allowed model in `live` so this config -- whose allowlist
        # deliberately omits 'haiku' before the widening reload -- is valid.
        live = OrchestratorConfig(
            models=ModelsConfig(architect='opus', module_tagger='sonnet'),
            routing=RoutingConfig(allowed_models=['sonnet', 'opus']),
        )
        fresh = OrchestratorConfig(
            models=ModelsConfig(architect='opus', module_tagger='sonnet'),
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
    argument assertions below.

    An override may also be an exception (instance or class): the call
    raises it instead of returning, standing in for a real invoke_fn crash
    (network error, subprocess crash, etc.) -- see
    TestProbeModelsInvokeErrorIsolation.
    """

    def __init__(
        self,
        overrides: dict[tuple[str, str], AgentResult | BaseException | type[BaseException]]
        | None = None,
    ) -> None:
        self._overrides = overrides or {}
        self.calls: list[dict] = []

    async def __call__(self, **kwargs: object) -> AgentResult:
        self.calls.append(kwargs)
        key = cast('tuple[str, str]', (kwargs.get('model'), kwargs.get('oauth_token')))
        outcome = self._overrides.get(key, AgentResult(success=True, output='ok'))
        if isinstance(outcome, BaseException) or (
            isinstance(outcome, type) and issubclass(outcome, BaseException)
        ):
            raise outcome
        return outcome


class TestProbeModelsTargetSet:
    """The probe's default target model set is dedup(allowed_models +
    [FABLE_CANDIDATE_MODEL]), order-preserving -- so the fable candidate is
    exercised even where a config has not admitted it, and exactly once
    where a config has. The artifact this produces is the per-(account,
    model) availability evidence an admission decision consumes."""

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

    def test_fable_candidate_is_the_admitted_literal_and_is_dispatched(self):
        """The one place the fable model string is pinned. Both admission
        rulings name 'claude-fable-5-1', so a future reader can re-check the
        coupling rather than guess: D5 admitted it to the eval arm
        (``orchestrator.evals.reviewer_trial.variants::VARIANT_FABLE51_SOLO``,
        ``model='claude-fable-5-1'``) and D6 admitted it to a live runtime
        allowlist (``dark-factory-orchestrator.yaml``'s
        ``routing.allowed_models``, commit 526e0eba99). Spelled as a LITERAL
        here on purpose -- the rest of this module asserts symbolically
        against the constant, so this test is what would catch the constant
        being repointed away from admission again.
        """
        assert FABLE_CANDIDATE_MODEL == 'claude-fable-5-1'

        accounts = [AccountConfig(name='max-x', oauth_token_env='MAX_X_TOKEN')]
        cli = _ScriptedProbeCli()

        report = asyncio.run(probe_models(
            accounts, ['haiku', 'sonnet'],
            invoke_fn=cli,
            token_resolver={'MAX_X_TOKEN': 'tok-x'}.get,
        ))

        assert report.models == ['haiku', 'sonnet', 'claude-fable-5-1']
        # Genuinely dispatched, not merely listed in report.models.
        assert 'claude-fable-5-1' in {call['model'] for call in cli.calls}

    def test_config_that_already_admits_the_candidate_gets_no_phantom_row(self):
        """A config whose allowlist already carries the candidate must be
        probed for it exactly once -- _dedup_preserve_order collapses the
        union to a no-op.

        WHY this is asserted: while the constant is stale, this same call
        yields a FIFTH trailing entry that every account probes as
        'unavailable', writing a phantom always-red row into the committed
        artifact -- re-telling the very lie this task exists to stop. The
        allowlist below is dark-factory's own live one
        (dark-factory-orchestrator.yaml, routing.allowed_models).
        """
        accounts = [AccountConfig(name='max-x', oauth_token_env='MAX_X_TOKEN')]
        cli = _ScriptedProbeCli()

        report = asyncio.run(probe_models(
            accounts, ['haiku', 'sonnet', 'opus', 'claude-fable-5-1'],
            invoke_fn=cli,
            token_resolver={'MAX_X_TOKEN': 'tok-x'}.get,
        ))

        assert report.models == ['haiku', 'sonnet', 'opus', 'claude-fable-5-1']


class TestProbeModelsStatusMappingAndDispatch:
    """Per-account x per-model status uses classify_invocation's outcome --
    OK->available, ModelNotFound->unavailable, AuthFailed->auth_error,
    CapHit->capped, else (a classified but otherwise-unrecognized Failure)
    ->error -- and invoke_fn is dispatched accounts x models times with the
    right model and resolved token."""

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
        overrides: dict[tuple[str, str], AgentResult | BaseException | type[BaseException]] = {
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
            # Classifies to Failure(kind='unclassified') -- not auth, not a
            # 404/marker model-not-found, not a cap/near-cap prefix, not a
            # CliLocalError marker, not a timed-out wedge -- so it exercises
            # classify_probe_outcome's generic 'error' catch-all branch.
            (FABLE_CANDIDATE_MODEL, 'tok-y'): AgentResult(
                success=False, output='something unexpected went wrong',
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
        assert report.accounts['max-y'][FABLE_CANDIDATE_MODEL] == 'error'

        # accounts (2) x models (haiku, sonnet, the fable candidate = 3) == 6.
        assert len(cli.calls) == 6
        for call in cli.calls:
            assert call['model'] in {'haiku', 'sonnet', FABLE_CANDIDATE_MODEL}
            assert call['oauth_token'] in {'tok-x', 'tok-y'}


class TestProbeModelsBudgetExhaustion:
    """A probe turn aborted by the local ``--max-budget-usd`` ceiling records
    the distinct ``'budget_too_low'`` status, never the generic ``'error'``.

    WHY the budget subtype outranks the catch-all: an
    ``error_max_budget_usd`` result is positive, structured evidence that
    the Anthropic API ACCEPTED the request and consumed real tokens --
    exactly the semantics ``shared/src/shared/usage_gate.py::
    _probe_hit_local_budget_cap`` already states. So the model string DID
    resolve for that account and the account was live; only the probe's own
    ceiling stopped the turn. Letting it fall through to the unclassified
    catch-all is the defect: a mis-sized budget then masquerades as
    unavailability and the committed artifact reports a model broken on
    every account when it is in fact present.
    """

    def test_budget_abort_is_budget_too_low_and_leaves_the_catch_all_intact(self):
        accounts = [AccountConfig(name='max-x', oauth_token_env='MAX_X_TOKEN')]
        overrides: dict[tuple[str, str], AgentResult | BaseException | type[BaseException]] = {
            ('sonnet', 'tok-x'): AgentResult(
                success=False, subtype='error_max_budget_usd', output='',
                turns=1, cost_usd=0.05,
            ),
            # Classifies to Failure(kind='unclassified') -- the generic
            # 'error' branch, asserted in the SAME run so the new budget
            # branch is shown to narrow nothing.
            (FABLE_CANDIDATE_MODEL, 'tok-x'): AgentResult(
                success=False, output='something unexpected went wrong',
            ),
        }
        cli = _ScriptedProbeCli(overrides)

        report = asyncio.run(probe_models(
            accounts, ['haiku', 'sonnet'],
            invoke_fn=cli,
            token_resolver={'MAX_X_TOKEN': 'tok-x'}.get,
        ))

        assert report.accounts['max-x']['sonnet'] == 'budget_too_low'
        assert report.accounts['max-x'][FABLE_CANDIDATE_MODEL] == 'error'
        # Unscripted -> the fake's default success: an ordinary probe is
        # untouched by the new branch.
        assert report.accounts['max-x']['haiku'] == 'available'

    def test_budget_abort_outranks_a_cap_like_body(self):
        """Precedence is evidence-based, not incidental ordering: a result
        carrying BOTH the budget subtype AND a body that reads like an
        account-level cap hit still records 'budget_too_low'.

        The local ``--max-budget-usd`` ceiling firing is NOT an account cap
        -- the distinction ``_probe_hit_local_budget_cap`` draws -- so the
        structured subtype must outrank every string heuristic below it,
        including the cap tier.
        """
        accounts = [AccountConfig(name='max-x', oauth_token_env='MAX_X_TOKEN')]
        overrides: dict[tuple[str, str], AgentResult | BaseException | type[BaseException]] = {
            ('sonnet', 'tok-x'): AgentResult(
                success=False, subtype='error_max_budget_usd',
                output="You've hit your usage limit. Your plan resets in 3h.",
                turns=1, cost_usd=0.05,
            ),
        }
        cli = _ScriptedProbeCli(overrides)

        report = asyncio.run(probe_models(
            accounts, ['sonnet'],
            models=['sonnet'],
            invoke_fn=cli,
            token_resolver={'MAX_X_TOKEN': 'tok-x'}.get,
        ))

        assert report.accounts['max-x']['sonnet'] == 'budget_too_low'


class TestProbeModelsBudgetForwarding:
    """The per-invocation budget probe_models forwards as ``max_budget_usd``
    defaults to the named ``DEFAULT_PROBE_BUDGET_USD`` constant, and an
    explicit ``budget_usd=`` still overrides it.

    Asserted through what actually REACHES invoke_fn (the fake's recorded
    calls), never through ``inspect.signature`` -- a declared default that
    some layer then overwrites would still be a defect, and the forwarded
    value is the thing the probe's behaviour depends on.
    """

    def test_default_budget_is_the_named_constant(self):
        # The single pin of the chosen number. Basis: one turn still pays for
        # the CLI's own preamble, and one fable turn measures ~$0.15-0.25 that
        # way, so $1.00 clears the most expensive probed model with ~4x margin
        # -- and the old $0.05 did not, which is the defect (task 5404).
        assert DEFAULT_PROBE_BUDGET_USD == 1.0

        accounts = [AccountConfig(name='max-x', oauth_token_env='MAX_X_TOKEN')]
        cli = _ScriptedProbeCli()

        asyncio.run(probe_models(
            accounts, ['haiku'],
            invoke_fn=cli,
            token_resolver={'MAX_X_TOKEN': 'tok-x'}.get,
        ))

        assert cli.calls, 'expected at least one probe invocation'
        for call in cli.calls:
            assert call['max_budget_usd'] == DEFAULT_PROBE_BUDGET_USD
            # Pinned in the same loop: this stays a ONE-turn probe, which is
            # the premise that makes 'budget_too_low' mean "the ceiling is
            # mis-sized" rather than "the agent ran long".
            assert call['max_turns'] == 1

    def test_explicit_budget_argument_is_forwarded(self):
        accounts = [AccountConfig(name='max-x', oauth_token_env='MAX_X_TOKEN')]
        cli = _ScriptedProbeCli()

        asyncio.run(probe_models(
            accounts, ['haiku'],
            invoke_fn=cli,
            token_resolver={'MAX_X_TOKEN': 'tok-x'}.get,
            budget_usd=0.25,
        ))

        assert cli.calls, 'expected at least one probe invocation'
        assert all(call['max_budget_usd'] == 0.25 for call in cli.calls)


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


# ---------------------------------------------------------------------------
# render_probe_artifact: pure, deterministic YAML serializer for the
# probe-models report -- safe to commit and diff (see this task's plan
# design_decisions).
# ---------------------------------------------------------------------------


def _sample_probe_report() -> ProbeReport:
    return ProbeReport(
        models=['haiku', 'sonnet', FABLE_CANDIDATE_MODEL],
        accounts={
            'max-x': {
                'haiku': 'available',
                'sonnet': 'unavailable',
                FABLE_CANDIDATE_MODEL: 'available',
            },
            'max-y': {
                'haiku': 'auth_error',
                'sonnet': 'capped',
                FABLE_CANDIDATE_MODEL: 'no_token',
            },
        },
    )


class TestRenderProbeArtifact:
    """render_probe_artifact(report, generated_at) is a pure, deterministic
    YAML serializer -- safe to commit and diff."""

    def test_yaml_round_trip_contains_models_generated_at_and_account_statuses(self):
        report = _sample_probe_report()
        artifact = render_probe_artifact(report, generated_at='2026-07-13T00:00:00Z')

        parsed = yaml.safe_load(artifact)

        assert parsed['generated_at'] == '2026-07-13T00:00:00Z'
        assert parsed['models'] == ['haiku', 'sonnet', FABLE_CANDIDATE_MODEL]
        assert FABLE_CANDIDATE_MODEL in parsed['models']

        for account_name, statuses in report.accounts.items():
            for model, status in statuses.items():
                assert parsed['accounts'][account_name][model] == status
            # The evidence an admission decision consumes: a fable-candidate
            # row present per account.
            assert FABLE_CANDIDATE_MODEL in parsed['accounts'][account_name]

    def test_is_pure_and_deterministic(self):
        report = _sample_probe_report()
        first = render_probe_artifact(report, generated_at='2026-07-13T00:00:00Z')
        second = render_probe_artifact(report, generated_at='2026-07-13T00:00:00Z')
        assert first == second


# ---------------------------------------------------------------------------
# `orchestrator probe-models` CLI: loads config, drives routing.probe_models
# (network-free via a monkeypatched fake on the routing module, mirroring
# how test_cli.py patches orchestrator.harness.Harness for the `run`
# command), and writes the rendered artifact to --output.
# ---------------------------------------------------------------------------


def _fake_probe_cli_config(monkeypatch, tmp_path) -> OrchestratorConfig:
    """A real, hermetically-constructed OrchestratorConfig for the CLI
    tests below -- ORCH_CONFIG_PATH/cwd are neutralised first (mirrors
    every other direct-construction test in this module) so this
    OrchestratorConfig() call can never pick up a stray real config.yaml
    from the actual environment."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')
    # Default RoutingConfig() (haiku/sonnet/opus) -- ModelsConfig's own
    # defaults (e.g. architect='opus') must stay inside routing.allowed_models
    # or OrchestratorConfig's fail-fast allowlist validator (step-2) rejects
    # construction; this test only cares that config.routing.allowed_models
    # is forwarded to probe_models, not that the list is unusual.
    return OrchestratorConfig(
        project_root=tmp_path,
        usage_cap=UsageCapConfig(
            accounts=[AccountConfig(name='max-x', oauth_token_env='MAX_X_TOKEN')],
        ),
        routing=RoutingConfig(),
    )


def _install_capturing_probe_models(monkeypatch, report: ProbeReport) -> list[dict]:
    """Monkeypatch routing.probe_models with a fake that records each call's
    keyword arguments and returns *report*, and hand back the list it records
    into.

    Same stub shape as the two tests above, widened to capture ``budget_usd``.
    Written once so the budget tests below all read the forwarded value the
    same way -- including the parse-time rejection test, whose whole assertion
    is that this list stays EMPTY.
    """
    calls: list[dict] = []

    async def fake_probe_models(accounts, allowed_models, *, models=None, **kwargs):
        calls.append({
            'accounts': accounts, 'allowed_models': allowed_models, 'models': models,
            'budget_usd': kwargs.get('budget_usd'),
        })
        return report

    monkeypatch.setattr(routing_module, 'probe_models', fake_probe_models)
    return calls


class TestProbeModelsCliBudgetOption:
    """`probe-models --budget-usd` is the operator control the probe was
    missing: the CLI forwarded nothing, so every run silently took
    probe_models's own default. It must forward the routing constant by
    default, honour an override, and reject a non-positive ceiling at parse
    time -- with the new classification, a zero or negative ceiling would
    make EVERY (account, model) pair abort and record 'budget_too_low',
    committing an artifact that is uniformly and plausibly wrong.
    """

    def _run(self, monkeypatch, tmp_path, extra_args: list[str]):
        fake_config = _fake_probe_cli_config(monkeypatch, tmp_path)
        monkeypatch.setattr('orchestrator.cli.load_config', lambda _path: fake_config)
        probe_calls = _install_capturing_probe_models(
            monkeypatch,
            ProbeReport(models=['haiku'], accounts={'max-x': {'haiku': 'available'}}),
        )

        cfg_file = tmp_path / 'config.yaml'
        cfg_file.write_text('')

        result = CliRunner().invoke(main, [
            'probe-models',
            '--config', str(cfg_file),
            '--output', str(tmp_path / 'model-availability.yaml'),
            *extra_args,
        ])
        return result, probe_calls

    def test_default_budget_is_the_routing_constant(self, monkeypatch, tmp_path):
        result, probe_calls = self._run(monkeypatch, tmp_path, [])

        assert result.exit_code == 0, result.output
        assert [call['budget_usd'] for call in probe_calls] == [DEFAULT_PROBE_BUDGET_USD]

    def test_budget_option_is_forwarded(self, monkeypatch, tmp_path):
        result, probe_calls = self._run(monkeypatch, tmp_path, ['--budget-usd', '2.5'])

        assert result.exit_code == 0, result.output
        assert [call['budget_usd'] for call in probe_calls] == [2.5]

    def test_non_positive_budget_is_rejected_before_probing(self, monkeypatch, tmp_path):
        result, probe_calls = self._run(monkeypatch, tmp_path, ['--budget-usd', '0'])

        assert result.exit_code != 0
        assert '--budget-usd' in result.output
        # A VALUE rejection, not an unknown-option one: click's 'Invalid
        # value' prefix is what proves the FloatRange constraint fired rather
        # than the option simply not existing (which is how this same test
        # would pass vacuously before the option is added).
        assert 'Invalid value' in result.output
        assert probe_calls == [], 'a rejected ceiling must never reach probe_models'


class TestProbeModelsCli:
    def test_writes_artifact_with_fable_row_and_exits_zero(self, monkeypatch, tmp_path):
        fake_config = _fake_probe_cli_config(monkeypatch, tmp_path)
        monkeypatch.setattr('orchestrator.cli.load_config', lambda _path: fake_config)

        scripted_report = ProbeReport(
            models=['haiku', 'sonnet', FABLE_CANDIDATE_MODEL],
            accounts={
                'max-x': {
                    'haiku': 'available',
                    'sonnet': 'available',
                    FABLE_CANDIDATE_MODEL: 'available',
                },
            },
        )
        probe_calls: list[dict] = []

        async def fake_probe_models(accounts, allowed_models, *, models=None, **kwargs):
            probe_calls.append({
                'accounts': accounts, 'allowed_models': allowed_models, 'models': models,
            })
            return scripted_report

        monkeypatch.setattr(routing_module, 'probe_models', fake_probe_models)

        cfg_file = tmp_path / 'config.yaml'
        cfg_file.write_text('')
        output_file = tmp_path / 'model-availability.yaml'

        result = CliRunner().invoke(main, [
            'probe-models',
            '--config', str(cfg_file),
            '--output', str(output_file),
        ])

        assert result.exit_code == 0, result.output
        assert output_file.exists(), 'expected the artifact file to be written at --output'
        parsed = yaml.safe_load(output_file.read_text())
        assert FABLE_CANDIDATE_MODEL in parsed['models']
        assert parsed['accounts']['max-x'][FABLE_CANDIDATE_MODEL] == 'available'
        assert parsed['accounts']['max-x']['haiku'] == 'available'

        assert len(probe_calls) == 1
        assert probe_calls[0]['models'] is None
        assert probe_calls[0]['allowed_models'] == list(DEFAULT_ALLOWED_MODELS)
        assert [a.name for a in probe_calls[0]['accounts']] == ['max-x']

    def test_models_option_is_forwarded_to_probe_models(self, monkeypatch, tmp_path):
        fake_config = _fake_probe_cli_config(monkeypatch, tmp_path)
        monkeypatch.setattr('orchestrator.cli.load_config', lambda _path: fake_config)

        probe_calls: list[list[str] | None] = []

        async def fake_probe_models(accounts, allowed_models, *, models=None, **kwargs):
            probe_calls.append(models)
            return ProbeReport(models=models or [], accounts={})

        monkeypatch.setattr(routing_module, 'probe_models', fake_probe_models)

        cfg_file = tmp_path / 'config.yaml'
        cfg_file.write_text('')
        output_file = tmp_path / 'model-availability.yaml'

        result = CliRunner().invoke(main, [
            'probe-models',
            '--config', str(cfg_file),
            '--output', str(output_file),
            '--models', 'opus,haiku',
        ])

        assert result.exit_code == 0, result.output
        assert probe_calls == [['opus', 'haiku']]
