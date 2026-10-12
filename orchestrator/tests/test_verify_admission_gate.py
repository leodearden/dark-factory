"""The in-process task-verify admission gate (plans/verify-admission-fifo-prd.md, task α).

Driven only through the gate's public API — ``gate_for(tmp_path)``, its value
types, ``admit`` and ``snapshot`` — with the REAL flock from
``shared.verify_admission`` taken inside every granted body. The gate is not
behind the autouse ``_neutralize_verify_admission`` seam, so no marker is needed.
"""

from __future__ import annotations

import dataclasses
from datetime import timedelta

import pytest
from shared.verify_admission import is_gated_role

from orchestrator.verify_admission_gate import (
    ROLE_CLASSES,
    AdmissionPlan,
    AdmissionPolicy,
    AdmissionRequest,
    gate_for,
)


class TestRoleClasses:
    @pytest.mark.parametrize('role', ['merge', 'task', 'background', 'offline', 'no-such-role'])
    def test_role_classes_agree_with_the_gated_role_authority(self, role):
        assert (role in ROLE_CLASSES) == is_gated_role(role)

    def test_task_ranks_ahead_of_background(self):
        assert ROLE_CLASSES['task'] < ROLE_CLASSES['background']


class TestAdmissionPlan:
    def test_begin_stamps_a_unique_plan(self):
        first = AdmissionPlan.begin(role='task', task_id='42')
        second = AdmissionPlan.begin(role='task', task_id='42')

        assert first.plan_id != second.plan_id
        assert first.arrival <= second.arrival
        assert first.arrived_at.utcoffset() == timedelta(0)
        assert first.role == 'task'
        assert first.task_id == '42'

    def test_requests_of_one_plan_share_its_identity_and_arrival(self):
        plan = AdmissionPlan.begin(role='background', task_id=None)

        a = plan.request(module='a', scope_kind=None)
        b = plan.request(module='b', scope_kind='file_scoped')

        assert a.plan_id == b.plan_id == plan.plan_id
        assert a.plan_arrival == b.plan_arrival == plan.arrival
        assert (a.role, a.module, a.task_id) == ('background', 'a', None)
        assert (b.module, b.scope_kind) == ('b', 'file_scoped')

    def test_an_ungated_role_cannot_begin_a_plan(self):
        with pytest.raises(ValueError, match=r'only gated verify roles.*merge'):
            AdmissionPlan.begin(role='merge', task_id='1')

    def test_an_ungated_role_cannot_build_a_request(self):
        with pytest.raises(ValueError, match=r'only gated verify roles.*merge'):
            AdmissionRequest(
                role='merge', plan_id='p', plan_arrival=0.0,
                task_id=None, module=None, scope_kind=None,
            )


class TestAdmissionPolicy:
    def test_a_valid_policy_constructs(self):
        policy = AdmissionPolicy(slots=1, order='arrival', promote_after_secs=0.0)
        assert (policy.slots, policy.order, policy.promote_after_secs) == (1, 'arrival', 0.0)

    def test_slots_below_one_is_rejected(self):
        with pytest.raises(ValueError, match=r'slots.*0'):
            AdmissionPolicy(slots=0, order='unordered', promote_after_secs=1.0)

    def test_an_unknown_order_is_rejected(self):
        with pytest.raises(ValueError, match=r'order.*fifo'):
            AdmissionPolicy(slots=1, order='fifo', promote_after_secs=1.0)  # pyright: ignore[reportArgumentType]

    def test_a_negative_promote_age_is_rejected(self):
        with pytest.raises(ValueError, match=r'promote_after_secs.*-1'):
            AdmissionPolicy(slots=1, order='arrival', promote_after_secs=-1.0)


class TestRegistry:
    def test_one_gate_per_slots_dir(self, tmp_path):
        assert gate_for(tmp_path) is gate_for(tmp_path)

    def test_the_key_is_the_resolved_path(self, tmp_path):
        assert gate_for(tmp_path) is gate_for(tmp_path / 'x' / '..')

    def test_distinct_dirs_get_distinct_gates(self, tmp_path):
        assert gate_for(tmp_path / 'a') is not gate_for(tmp_path / 'b')


class TestFreshSnapshot:
    def test_a_fresh_gate_is_empty(self, tmp_path):
        snap = gate_for(tmp_path).snapshot()

        assert snap.depth == 0
        assert snap.waiters == ()
        assert snap.holders == ()
        assert snap.oldest is None
        assert snap.oldest_wait_secs is None

    def test_the_snapshot_is_frozen(self, tmp_path):
        snap = gate_for(tmp_path).snapshot()

        with pytest.raises(dataclasses.FrozenInstanceError):
            snap.waiters = ()  # pyright: ignore[reportAttributeAccessIssue]
