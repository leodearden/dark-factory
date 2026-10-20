"""TaskVerifyDispatcher: one routing point for a task-role verify, local or remote.

The value layer (policy, eligibility, the decision record) is pure.
"""

from __future__ import annotations

import dataclasses

import pytest

from orchestrator.task_verify_dispatch import (
    CrossCheck,
    DispatchDecision,
    RedConfirmation,
    TaskVerifyPolicy,
    eligible,
)

_SIDECAR_KEYS = {
    'decision', 'runner', 'reason', 'tip_sha', 'base_sha',
    'remote_wall_secs', 'lease_held_secs', 'cross_check', 'red_confirmed_locally',
}


def _decision(reason, **overrides) -> DispatchDecision:
    fields = {
        'runner': 'local',
        'reason': reason,
        'tip_sha': 'a' * 40,
        'base_sha': 'b' * 40,
        'remote_wall_secs': None,
        'lease_held_secs': None,
        'cross_check': None,
        'red_confirmed_locally': None,
    }
    fields.update(overrides)
    return DispatchDecision(**fields)


class TestTaskVerifyPolicy:
    def test_defaults_route_everything_locally(self) -> None:
        policy = TaskVerifyPolicy()

        assert policy.host_policy == 'local'
        assert policy.remote_modules == ()
        assert policy.cross_check_every_n == 0
        assert policy.red_confirms_locally is False

    def test_is_frozen(self) -> None:
        pytest.raises(
            dataclasses.FrozenInstanceError, setattr, TaskVerifyPolicy(), 'host_policy', 'remote_all',
        )

    def test_remote_modules_policy_needs_modules(self) -> None:
        with pytest.raises(ValueError, match='host_policy') as excinfo:
            TaskVerifyPolicy(host_policy='remote_modules', remote_modules=())

        assert 'remote_modules' in str(excinfo.value)

    def test_negative_cross_check_cadence_is_refused(self) -> None:
        with pytest.raises(ValueError, match='cross_check_every_n'):
            TaskVerifyPolicy(cross_check_every_n=-1)

    @pytest.mark.parametrize('host_policy', ['remote', 'LOCAL', ''])
    def test_unknown_host_policy_is_refused(self, host_policy) -> None:
        with pytest.raises(ValueError, match='host_policy'):
            TaskVerifyPolicy(host_policy=host_policy)

    @pytest.mark.parametrize('remote_modules', [['orchestrator'], 'orchestrator'])
    def test_remote_modules_must_be_a_tuple(self, remote_modules) -> None:
        with pytest.raises(ValueError, match='remote_modules'):
            TaskVerifyPolicy(host_policy='remote_modules', remote_modules=remote_modules)


class TestEligible:
    _POLICIES = [
        TaskVerifyPolicy(),
        TaskVerifyPolicy(host_policy='remote_all'),
        TaskVerifyPolicy(host_policy='remote_modules', remote_modules=('orchestrator',)),
    ]

    @pytest.mark.parametrize('executed', [set(), {'orchestrator'}, {'fused_memory'}])
    def test_local_policy_is_never_eligible(self, executed: set[str]) -> None:
        assert eligible(
            TaskVerifyPolicy(), executed, force_workspace=False, per_module_overrides_present=False,
        ) is False

    @pytest.mark.parametrize('executed', [set(), {'orchestrator'}, {'fused_memory', 'shared'}])
    def test_remote_all_is_eligible_for_any_executed_set(self, executed: set[str]) -> None:
        assert eligible(
            TaskVerifyPolicy(host_policy='remote_all'), executed,
            force_workspace=False, per_module_overrides_present=False,
        ) is True

    @pytest.mark.parametrize(('executed', 'expected'), [
        ({'orchestrator'}, True),
        ({'orchestrator', 'fused_memory'}, True),
        ({'fused_memory'}, False),
        (set(), False),
    ])
    def test_remote_modules_needs_a_listed_module_executed(
        self, executed: set[str], expected: bool,
    ) -> None:
        policy = TaskVerifyPolicy(host_policy='remote_modules', remote_modules=('orchestrator',))

        assert eligible(
            policy, executed, force_workspace=False, per_module_overrides_present=False,
        ) is expected

    @pytest.mark.parametrize('policy', _POLICIES)
    def test_force_workspace_is_never_eligible(self, policy: TaskVerifyPolicy) -> None:
        assert eligible(
            policy, {'orchestrator'}, force_workspace=True, per_module_overrides_present=False,
        ) is False

    @pytest.mark.parametrize('policy', _POLICIES)
    def test_per_module_overrides_are_never_eligible(self, policy: TaskVerifyPolicy) -> None:
        assert eligible(
            policy, {'orchestrator'}, force_workspace=False, per_module_overrides_present=True,
        ) is False


class TestDispatchDecision:
    @pytest.mark.parametrize('reason', ['local', 'REMOTE', 'busy'])
    def test_reason_outside_the_vocabulary_is_refused(self, reason) -> None:
        with pytest.raises(ValueError, match=f"'{reason}'"):
            _decision(reason)

    @pytest.mark.parametrize(('reason', 'expected'), [
        ('remote', 'remote'),
        ('not_eligible', 'local'),
        ('merge_demand', 'local'),
        ('cross_check', 'local'),
        ('red_confirm', 'local'),
        ('dispatcher_error', 'local'),
    ])
    def test_decision_is_derived_from_the_reason(self, reason, expected: str) -> None:
        assert _decision(reason).decision == expected

    def test_to_dict_has_exactly_the_sidecar_keys(self) -> None:
        rendered = _decision('remote', runner='laptop', remote_wall_secs=1.5).to_dict()

        assert set(rendered) == _SIDECAR_KEYS
        assert rendered['decision'] == 'remote'
        assert rendered['runner'] == 'laptop'
        assert rendered['reason'] == 'remote'
        assert rendered['tip_sha'] == 'a' * 40
        assert rendered['base_sha'] == 'b' * 40
        assert rendered['remote_wall_secs'] == 1.5
        assert rendered['cross_check'] is None
        assert rendered['red_confirmed_locally'] is None

    @pytest.mark.parametrize(('remote_passed', 'local_passed', 'agree'), [
        (True, True, True),
        (False, True, False),
        (True, False, False),
        (False, False, True),
    ])
    def test_cross_check_renders_its_agreement(
        self, remote_passed: bool, local_passed: bool, agree: bool,
    ) -> None:
        cross = CrossCheck(remote_passed=remote_passed, local_passed=local_passed)

        assert cross.agree is agree
        assert _decision('cross_check', cross_check=cross).to_dict()['cross_check'] == {
            'remote_passed': remote_passed, 'local_passed': local_passed, 'agree': agree,
        }

    @pytest.mark.parametrize('local_passed', [True, False])
    def test_red_confirmation_renders_the_remote_red(self, local_passed: bool) -> None:
        confirmation = RedConfirmation(local_passed=local_passed)

        rendered = _decision('red_confirm', red_confirmed_locally=confirmation).to_dict()

        assert rendered['red_confirmed_locally'] == {
            'remote_passed': False, 'local_passed': local_passed,
        }
