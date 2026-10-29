"""The pure open policy of the shadow coordinator (plans/live-shadow-eval-prd.md C2).

Sampling is a pure function of (task_id, candidate, shape, stratum, seed, rate),
so both projects' coordinators cut identically (PRD decision 8); the gate walk
is one ordered precedence, sampling first among the resource gates.
"""

from __future__ import annotations

import dataclasses
import random

import pytest
from shared.psi import PsiSample

from orchestrator.config import PsiAdmissionConfig
from orchestrator.evals.configs import EvalConfig, get_config_by_name
from orchestrator.event_store import ShadowSkipReason
from orchestrator.shadow_eval import (
    HostVerdict,
    OpenDecision,
    OpenInputs,
    budget_exhausted,
    candidate_has_credentials,
    cost_basis_of,
    decide_open,
    host_verdict,
    sample_point,
    sampled_in,
    shapes_triggered_by,
)
from orchestrator.shadow_types import CostBasis, ShadowShape
from orchestrator.workflow_types import WorkflowState

_STRATUM = ('df', 'feature', 'full')
_R = ShadowSkipReason


def _open_inputs(**overrides) -> OpenInputs:
    """Inputs that pass every gate; each test fails exactly the gates it names."""
    fields = {
        'task_id': '5387',
        'shape': ShadowShape.IMPLEMENTER,
        'candidate': 'glm-5.3-endpoint',
        'stratum': _STRATUM,
        'base_sha': 'a' * 40,
        'seed': 0,
        'sample_rate': 1.0,
        'enabled': True,
        'enabled_shapes': frozenset({ShadowShape.IMPLEMENTER}),
        'end_to_end_candidates': frozenset(),
        'paused': False,
        'already_open': False,
        'has_credentials': True,
        'isolation_available': True,
        'credits_exhausted_recently': False,
        'spent_usd': 0.0,
        'daily_budget_usd': 50.0,
        'held_cells': 0,
        'max_concurrent': 1,
        'host': HostVerdict.CLEAR,
    }
    fields.update(overrides)
    return OpenInputs(**fields)


def _random_inputs(rng: random.Random) -> OpenInputs:
    shapes = list(ShadowShape)
    return _open_inputs(
        task_id=str(rng.randrange(10_000)),
        shape=rng.choice(shapes),
        candidate=rng.choice(['glm-5.3-endpoint', 'minimax-m3-endpoint', 'claude-sonnet-max']),
        stratum=(rng.choice(['df', 'reify']), rng.choice(['feature', 'fix']), 'full'),
        base_sha=rng.choice(['a' * 40, None]),
        seed=rng.randrange(100),
        sample_rate=rng.random(),
        enabled=rng.random() < 0.9,
        enabled_shapes=frozenset(rng.sample(shapes, rng.randrange(len(shapes) + 1))),
        end_to_end_candidates=frozenset(rng.sample(['glm-5.3-endpoint', 'x'], rng.randrange(3))),
        paused=rng.random() < 0.1,
        already_open=rng.random() < 0.1,
        has_credentials=rng.random() < 0.9,
        isolation_available=rng.random() < 0.9,
        credits_exhausted_recently=rng.random() < 0.1,
        spent_usd=rng.uniform(0, 60),
        daily_budget_usd=50.0,
        held_cells=rng.randrange(3),
        max_concurrent=rng.randrange(1, 3),
        host=rng.choice(list(HostVerdict)),
    )


class TestSamplePoint:
    def test_points_are_unit_interval_and_repeatable(self):
        rng = random.Random(5387)
        for _ in range(300):
            args = (
                str(rng.randrange(10_000)),
                rng.choice(['glm-5.3-endpoint', 'minimax-m3-endpoint']),
                rng.choice(list(ShadowShape)),
                (rng.choice(['df', 'reify']), rng.choice(['feature', 'fix']), 'full'),
                rng.randrange(100),
            )
            point = sample_point(*args)
            assert 0.0 <= point < 1.0
            assert sample_point(*args) == point

    def test_decide_open_is_a_pure_function_of_its_inputs(self):
        rng = random.Random(5387)
        for _ in range(300):
            inputs = _random_inputs(rng)
            assert decide_open(inputs) == decide_open(dataclasses.replace(inputs))

    def test_the_golden_point_pins_cross_process_reproducibility(self):
        # sha256 over the canonical json, first 8 bytes / 2**64. A salted
        # hash() would differ per process and so per project's coordinator.
        point = sample_point('5387', 'glm-5.3-endpoint', ShadowShape.IMPLEMENTER, _STRATUM, 0)
        assert point == 0.571306983043451

    def test_raising_the_rate_only_ever_adds_tasks(self):
        rates = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        for task in range(200):
            point = sample_point(str(task), 'c', ShadowShape.IMPLEMENTER, _STRATUM, 3)
            admitted = [sampled_in(point, rate) for rate in rates]
            first_in = admitted.index(True)
            assert all(admitted[first_in:])
            assert admitted[0] is False
            assert admitted[-1] is True

    def test_half_rate_selects_about_half_and_the_seed_changes_the_set(self):
        def selected(seed: int) -> set[int]:
            return {
                task
                for task in range(400)
                if sampled_in(sample_point(str(task), 'c', ShadowShape.IMPLEMENTER, _STRATUM, seed), 0.5)
            }

        assert 0.35 <= len(selected(0)) / 400 <= 0.65
        assert selected(0) != selected(1)


def _sampled_out_point_task() -> str:
    """A task id whose point lies above 0.5 at seed 0, so rate 0.5 samples it out."""
    return next(
        str(task)
        for task in range(1000)
        if sample_point(str(task), 'glm-5.3-endpoint', ShadowShape.IMPLEMENTER, _STRATUM, 0) >= 0.5
    )


# Each gate alone, failed, in the precedence order of decide_open.
_GATE_FAILURES: list[tuple[ShadowSkipReason, dict]] = [
    (_R.SHAPE_DISABLED, {'enabled': False}),
    (_R.SHAPE_DISABLED, {'enabled_shapes': frozenset({ShadowShape.ARCHITECT})}),
    (
        _R.SHAPE_DISABLED,
        {
            'shape': ShadowShape.END_TO_END,
            'enabled_shapes': frozenset({ShadowShape.END_TO_END}),
            'end_to_end_candidates': frozenset({'some-other-candidate'}),
        },
    ),
    (_R.PAUSED, {'paused': True}),
    (_R.ALREADY_OPEN, {'already_open': True}),
    (_R.SAMPLED_OUT, {'sample_rate': 0.0}),
    (_R.NO_BASE_SHA, {'base_sha': None}),
    (_R.NO_BASE_SHA, {'base_sha': '  '}),
    (_R.NO_CREDENTIALS, {'has_credentials': False}),
    (_R.ISOLATION_UNAVAILABLE, {'isolation_available': False}),
    (_R.CREDITS_EXHAUSTED, {'credits_exhausted_recently': True}),
    (_R.BUDGET_EXHAUSTED, {'spent_usd': 50.0}),
    (_R.CONCURRENCY_FULL, {'held_cells': 1}),
    (_R.HOST_SATURATED, {'host': HostVerdict.SATURATED}),
]

_PRECEDENCE = [
    _R.SHAPE_DISABLED,
    _R.PAUSED,
    _R.ALREADY_OPEN,
    _R.SAMPLED_OUT,
    _R.NO_BASE_SHA,
    _R.NO_CREDENTIALS,
    _R.ISOLATION_UNAVAILABLE,
    _R.CREDITS_EXHAUSTED,
    _R.BUDGET_EXHAUSTED,
    _R.CONCURRENCY_FULL,
    _R.HOST_SATURATED,
]


class TestGatePrecedence:
    def test_all_gates_passing_opens(self):
        assert decide_open(_open_inputs()) == OpenDecision(open=True, reason=None, host=HostVerdict.CLEAR)

    @pytest.mark.parametrize(
        'reason,failure',
        _GATE_FAILURES,
        ids=[f'{reason.value}-{i}' for i, (reason, _) in enumerate(_GATE_FAILURES)],
    )
    def test_each_gate_alone_reports_its_reason(self, reason, failure):
        decision = decide_open(_open_inputs(**failure))

        assert decision.open is False
        assert decision.reason is reason

    def test_a_sampled_out_task_never_reports_a_load_reason(self):
        decision = decide_open(
            _open_inputs(
                task_id=_sampled_out_point_task(),
                sample_rate=0.5,
                spent_usd=99.0,
                held_cells=5,
                host=HostVerdict.SATURATED,
            ),
        )
        assert decision.reason is _R.SAMPLED_OUT

    @pytest.mark.parametrize(
        'earlier,later',
        [(a, b) for i, a in enumerate(_PRECEDENCE) for b in _PRECEDENCE[i + 1:]],
        ids=lambda r: r.value,
    )
    def test_two_failing_gates_report_the_earlier(self, earlier, later):
        failures = {reason: failure for reason, failure in reversed(_GATE_FAILURES)}
        decision = decide_open(_open_inputs(**{**failures[later], **failures[earlier]}))

        assert decision.reason is earlier

    def test_the_budget_gate_is_the_shared_budget_rule(self):
        assert budget_exhausted(50.0, 50.0) is True
        assert budget_exhausted(49.99, 50.0) is False
        assert decide_open(_open_inputs(spent_usd=49.99)).open is True


class TestShapeTriggers:
    def test_architect_is_enabled_by_the_consequence_shape_alone(self):
        decision = decide_open(
            _open_inputs(
                shape=ShadowShape.ARCHITECT,
                enabled_shapes=frozenset({ShadowShape.ARCHITECT_CONSEQUENCE}),
            ),
        )
        assert decision.open is True

    def test_end_to_end_opens_for_a_listed_end_to_end_candidate(self):
        decision = decide_open(
            _open_inputs(
                shape=ShadowShape.END_TO_END,
                enabled_shapes=frozenset({ShadowShape.END_TO_END}),
                end_to_end_candidates=frozenset({'glm-5.3-endpoint'}),
            ),
        )
        assert decision.open is True

    def test_plan_entry_triggers_the_live_architect_shapes(self):
        assert shapes_triggered_by(WorkflowState.PLAN, entering=True) == {
            ShadowShape.ARCHITECT,
            ShadowShape.END_TO_END,
        }

    def test_plan_exit_triggers_the_implementer(self):
        assert shapes_triggered_by(WorkflowState.PLAN, entering=False) == {ShadowShape.IMPLEMENTER}

    def test_the_consequence_leg_is_never_phase_triggered(self):
        for phase in WorkflowState:
            for entering in (True, False):
                assert ShadowShape.ARCHITECT_CONSEQUENCE not in shapes_triggered_by(phase, entering)

    def test_other_phases_trigger_nothing(self):
        assert shapes_triggered_by(WorkflowState.EXECUTE, entering=True) == frozenset()


class TestOpenDecision:
    def test_an_open_decision_carries_no_reason(self):
        with pytest.raises(ValueError, match='reason'):
            OpenDecision(open=True, reason=_R.PAUSED, host=HostVerdict.CLEAR)

    def test_a_skip_names_its_reason(self):
        with pytest.raises(ValueError, match='reason'):
            OpenDecision(open=False, reason=None, host=HostVerdict.CLEAR)


def _sample(*, read_ok: bool, pressure: float) -> PsiSample:
    return PsiSample(
        cpu_some10=pressure,
        mem_some10=pressure,
        mem_full10=pressure,
        io_some10=pressure,
        read_ok=read_ok,
    )


class TestHostVerdict:
    def test_an_unreadable_sample_is_its_own_fail_open_arm(self):
        cfg = PsiAdmissionConfig()
        assert host_verdict(_sample(read_ok=False, pressure=99.0), cfg) is HostVerdict.UNREADABLE

    def test_a_saturated_readable_sample(self):
        cfg = PsiAdmissionConfig()
        assert host_verdict(_sample(read_ok=True, pressure=99.0), cfg) is HostVerdict.SATURATED

    def test_a_calm_readable_sample(self):
        cfg = PsiAdmissionConfig()
        assert host_verdict(_sample(read_ok=True, pressure=0.0), cfg) is HostVerdict.CLEAR

    def test_a_disabled_gate_is_clear(self):
        cfg = PsiAdmissionConfig(enabled=False)
        assert host_verdict(_sample(read_ok=True, pressure=99.0), cfg) is HostVerdict.CLEAR

    def test_an_unreadable_host_opens_and_the_decision_says_so(self):
        decision = decide_open(_open_inputs(host=HostVerdict.UNREADABLE))

        assert decision.open is True
        assert decision.host is HostVerdict.UNREADABLE


class TestCandidateFacts:
    def test_a_blank_endpoint_token_is_no_credentials(self, monkeypatch):
        monkeypatch.setenv('ZAI_API_KEY', '')
        config = get_config_by_name('glm-5.3-endpoint')
        assert config is not None
        assert candidate_has_credentials(config) is False

    def test_a_populated_endpoint_token_is_credentials(self, monkeypatch):
        monkeypatch.setenv('ZAI_API_KEY', 'zai-test-key')
        config = get_config_by_name('glm-5.3-endpoint')
        assert config is not None
        assert candidate_has_credentials(config) is True

    def test_a_blank_api_key_is_no_credentials(self):
        config = EvalConfig('x', 'claude', 'm', 'high', env_overrides={'ANTHROPIC_API_KEY': ' '})
        assert candidate_has_credentials(config) is False

    @pytest.mark.parametrize('name', ['claude-sonnet-max', 'claude-opus-high'])
    def test_a_native_config_with_no_credential_key_has_credentials(self, name):
        config = get_config_by_name(name)
        assert config is not None
        assert candidate_has_credentials(config) is True

    @pytest.mark.parametrize('name', ['glm-5.3-endpoint', 'glm-5.3-flash-endpoint'])
    def test_glm_endpoint_candidates_are_subscription(self, name):
        config = get_config_by_name(name)
        assert config is not None
        assert cost_basis_of(config) is CostBasis.SUBSCRIPTION

    @pytest.mark.parametrize('name', ['minimax-m3-endpoint', 'claude-sonnet-max'])
    def test_minimax_and_native_configs_are_metered(self, name):
        config = get_config_by_name(name)
        assert config is not None
        assert cost_basis_of(config) is CostBasis.METERED
