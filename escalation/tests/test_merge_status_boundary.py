"""merge_status's boundary suite, rows B1-B9, across a simulated restart.

The executable form of ``plans/merge-status-durable-non-landed-prd.md``'s
Contract (B+H) and Boundary-test sketch, rows B1-B9.  B10 lives in
``orchestrator/tests/test_merge_status_boundary_b10.py`` because its subject is
EventStore's cross-run read.

Real git, real producers, real ``escalation.git_authority.probe_landing``: the
prior run's state is written by the orchestrator's own finalize and journal
writers, and the simulated restart is a fresh run_id plus a fresh journal store
over the prior run's files.  The unit-scope tier tests are
``escalation/tests/test_merge_status_tiers.py``.
"""
from __future__ import annotations

from typing import Any

import pytest
from orchestrator.merge_lane import MergeOutcome
from shared.merge_state import OUTCOME_STATES, MergeReason, MergeState

from escalation.merge_status_answer import hint_for

pytestmark = pytest.mark.asyncio

_BRANCH_KEY_DEFECT = (
    'esc-4834-3, follow-up tkt_0RVWJQ8EMTQQ4JX0F81NN4237P: every producer writes '
    "branch as QueuedBranch.bare_id while merge_status matches the branch key "
    "literally, so the 'task/<id>' form the merge-queue, unblock and "
    'unblock-low-risk skills document answers no_record/never_submitted for a '
    "prior run's row."
)


class TestB1PriorRunBlockedAfterRestart:
    """B1 is THE REGRESSION PIN — must not be waived: a prior run's blocked is never this run's state."""

    @pytest.mark.parametrize('key', ['task_id', 'request_id'])
    async def test_b1_a_prior_run_blocked_answers_stale_record_never_blocked(
        self, world: Any, key: str,
    ) -> None:
        _unmerged_branch(world, '101')
        prior = await _finalize_in_prior_run(
            world, '101', MergeOutcome('blocked', reason='post-merge verify failed'),
        )
        poll_key = {'task_id': '101', 'request_id': prior.request_id}[key]

        response = await _poll(_restart(world), **{key: poll_key})

        assert response['state'] != 'blocked'
        assert response['state'] not in OUTCOME_STATES
        assert response['state'] == MergeState.stale_record
        assert response['reason'] == MergeReason.prior_run_terminal
        assert response['source'] == 'event_store_cross_run'
        assert response['hint'] == hint_for(MergeReason.prior_run_terminal)
        last = response['last_outcome']
        assert last['state'] == 'blocked'
        assert last['is_current_run'] is False
        assert last['run_id'] == _RUN_PRIOR
        assert last['request_id'] == prior.request_id


class TestB4NeverSubmitted:
    async def test_b4_nothing_anywhere_answers_no_record_never_submitted(
        self, world: Any,
    ) -> None:
        response = await _poll(_restart(world), task_id='104')

        assert response['state'] == MergeState.no_record
        assert response['reason'] == MergeReason.never_submitted
        assert response['source'] == 'none'
        assert response['last_outcome'] is None
        assert response['hint'] == hint_for(MergeReason.never_submitted)


@pytest.mark.xfail(strict=True, reason=_BRANCH_KEY_DEFECT)
async def test_branch_key_in_the_skills_full_ref_form_resolves_a_prior_run_row(
    world: Any,
) -> None:
    _unmerged_branch(world, '110')
    prior = await _finalize_in_prior_run(
        world, '110', MergeOutcome('blocked', reason='post-merge verify failed'),
    )

    response = await _poll(_restart(world), branch='task/110')

    assert response['state'] == MergeState.stale_record
    assert response['last_outcome'] is not None
    assert response['last_outcome']['request_id'] == prior.request_id
