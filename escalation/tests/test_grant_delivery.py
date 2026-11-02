"""Tests for the granted-files delivery vocabulary: escalation/grant_delivery.py.

``grant_delivery_for`` decides, synchronously and at the API, whether a
``resolve_issue`` scope grant is routed to a delivery path; ``shortfall_reason``
explains every value that is not. Task 5650.
"""

from __future__ import annotations

import pytest

from escalation.action_effects import ACTION_EFFECTS, WORKFLOW_RESUME
from escalation.grant_delivery import (
    GrantDelivery,
    grant_delivery_for,
    shortfall_reason,
)

_GRANT = ['pkg/a.py', 'pkg/b.py']
_TASK_ID = '3438'

_NON_RESUME_DISPOSITIONS = sorted(
    {
        effect.workflow_disposition
        for effect in ACTION_EFFECTS.values()
        if effect.workflow_disposition != WORKFLOW_RESUME
    }
)

_DELIVERED_OR_NOTHING_TO_DELIVER = {
    GrantDelivery.NONE_GRANTED,
    GrantDelivery.SCHEDULED,
}


def _probe_must_not_run() -> GrantDelivery | None:
    raise AssertionError('the harness probe must not be consulted on this path')


def _probe_returning(value: GrantDelivery | None):
    def probe() -> GrantDelivery | None:
        return value

    return probe


class TestGrantDeliveryPrecedence:
    @pytest.mark.parametrize('granted_files', [None, []])
    def test_no_grant_is_none_granted_without_probing(self, granted_files):
        delivery = grant_delivery_for(
            granted_files=granted_files,
            workflow_disposition=WORKFLOW_RESUME,
            applied=True,
            probe_blocker=_probe_must_not_run,
        )

        assert delivery is GrantDelivery.NONE_GRANTED

    def test_the_non_resume_dispositions_are_sourced_from_table_b(self):
        assert _NON_RESUME_DISPOSITIONS, 'Table B lost every non-resume action'

    @pytest.mark.parametrize('disposition', _NON_RESUME_DISPOSITIONS)
    def test_a_grant_on_a_non_resume_action_does_not_apply(self, disposition):
        delivery = grant_delivery_for(
            granted_files=_GRANT,
            workflow_disposition=disposition,
            applied=True,
            probe_blocker=_probe_must_not_run,
        )

        assert delivery is GrantDelivery.ACTION_DOES_NOT_APPLY_GRANTS

    def test_a_grant_on_an_unapplied_resume_is_record_not_open(self):
        delivery = grant_delivery_for(
            granted_files=_GRANT,
            workflow_disposition=WORKFLOW_RESUME,
            applied=False,
            probe_blocker=_probe_must_not_run,
        )

        assert delivery is GrantDelivery.RECORD_NOT_OPEN

    @pytest.mark.parametrize(
        'blocker',
        [GrantDelivery.NO_ORCHESTRATOR, GrantDelivery.NO_PLAN_TO_WIDEN],
    )
    def test_a_probe_blocker_is_returned(self, blocker):
        delivery = grant_delivery_for(
            granted_files=_GRANT,
            workflow_disposition=WORKFLOW_RESUME,
            applied=True,
            probe_blocker=_probe_returning(blocker),
        )

        assert delivery is blocker

    def test_an_unblocked_applied_resume_is_scheduled(self):
        delivery = grant_delivery_for(
            granted_files=_GRANT,
            workflow_disposition=WORKFLOW_RESUME,
            applied=True,
            probe_blocker=_probe_returning(None),
        )

        assert delivery is GrantDelivery.SCHEDULED


class TestShortfallReason:
    @pytest.mark.parametrize('delivery', sorted(_DELIVERED_OR_NOTHING_TO_DELIVER))
    def test_no_reason_when_nothing_fell_short(self, delivery):
        assert shortfall_reason(
            delivery, task_id=_TASK_ID, granted_files=_GRANT,
        ) is None

    @pytest.mark.parametrize(
        'delivery',
        [d for d in GrantDelivery if d not in _DELIVERED_OR_NOTHING_TO_DELIVER],
    )
    def test_every_shortfall_names_the_task_and_every_path(self, delivery):
        reason = shortfall_reason(
            delivery, task_id=_TASK_ID, granted_files=_GRANT,
        )

        assert isinstance(reason, str) and reason
        assert _TASK_ID in reason
        for path in _GRANT:
            assert path in reason
