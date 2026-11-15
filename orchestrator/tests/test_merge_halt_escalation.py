"""The halt-owning L1 a merge outcome that halted the merge queue files (task 7277).

Public seams only: this file imports lane types, so the merge-lane ratchet
counts every private read and lane patch target in it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.merge_halt_escalation import (
    MERGE_HALT_CATEGORY_BY_STATUS,
    MERGE_HALT_STATUSES,
    build_merge_halt_escalation_text,
    submit_halt_owning_escalation,
)
from orchestrator.merge_lane import MergeOutcome

_TID = '40'

_EXPECTED_CATEGORY = {
    'wip_halted': 'wip_conflict',
    'done_wip_recovery': 'wip_conflict',
    'wip_recovery_no_advance': 'wip_conflict',
    'unmerged_state': 'unmerged_state',
    'stash_failed': 'stash_failed',
}

_HALT_OUTCOMES = {
    'wip_halted': MergeOutcome(status='wip_halted', overlap_files=['x.py']),
    'done_wip_recovery': MergeOutcome(status='done_wip_recovery', recovery_branch='wip/r'),
    'wip_recovery_no_advance': MergeOutcome(
        status='wip_recovery_no_advance', recovery_branch='wip/r',
    ),
    'unmerged_state': MergeOutcome(status='unmerged_state'),
    'stash_failed': MergeOutcome(status='stash_failed', dirty_files=['README.md']),
}


class FakeMergeHalt:
    def __init__(
        self, *, is_wip_halted: bool = True, halt_owner_esc_id: str | None = None,
        queue: EscalationQueue | None = None,
    ) -> None:
        self.is_wip_halted = is_wip_halted
        self.halt_owner_esc_id = halt_owner_esc_id
        self.queue = queue
        self.owners: list[str] = []
        self.unhalt_reasons: list[str | None] = []
        self.record_present_at_owner: list[bool] = []

    def set_halt_owner(self, esc_id: str) -> None:
        assert self.is_wip_halted
        assert self.halt_owner_esc_id is None
        if self.queue is not None:
            self.record_present_at_owner.append(self.queue.get(esc_id) is not None)
        self.halt_owner_esc_id = esc_id
        self.owners.append(esc_id)

    def unhalt_wip(self, reason: str | None = None) -> None:
        self.is_wip_halted = False
        self.unhalt_reasons.append(reason)


class RaisingQueue(EscalationQueue):
    def submit(self, escalation: Escalation) -> str:
        raise RuntimeError('disk full')


def _escalation(queue: EscalationQueue) -> Escalation:
    return Escalation(
        id=queue.make_id(_TID),
        task_id=_TID,
        agent_role='orchestrator',
        severity='blocking',
        category='stash_failed',
        summary='project_root park failed',
        detail='dirty tracked tree',
        suggested_action='manual_intervention',
        level=1,
    )


class TestHaltStatusesMapToTheirCategory:
    @pytest.mark.parametrize(('status', 'category'), sorted(_EXPECTED_CATEGORY.items()))
    def test_status_maps_to_category(self, status: str, category: str) -> None:
        text = build_merge_halt_escalation_text(status, _HALT_OUTCOMES[status], task_id=_TID)

        assert MERGE_HALT_CATEGORY_BY_STATUS[status] == category
        assert text.category == category

    def test_the_halt_statuses_are_exactly_the_five(self) -> None:
        assert frozenset(_EXPECTED_CATEGORY) == MERGE_HALT_STATUSES

    def test_every_category_is_one_a_restart_re_owns(self) -> None:
        # The set orchestrator/src/orchestrator/harness.py::Harness._rehydrate_merge_halt re-owns.
        assert set(MERGE_HALT_CATEGORY_BY_STATUS.values()) <= {
            'wip_conflict', 'unmerged_state', 'stash_failed',
        }


class TestTextCarriesTheOutcomesFacts:
    def test_stash_failed_names_the_dirty_files(self) -> None:
        text = build_merge_halt_escalation_text(
            'stash_failed', _HALT_OUTCOMES['stash_failed'], task_id=_TID,
        )

        assert 'README.md' in text.detail

    @pytest.mark.parametrize('status', ['done_wip_recovery', 'wip_recovery_no_advance'])
    def test_wip_recovery_names_the_recovery_branch(self, status: str) -> None:
        text = build_merge_halt_escalation_text(status, _HALT_OUTCOMES[status], task_id=_TID)

        assert 'wip/r' in text.detail

    def test_wip_halted_names_the_overlap_in_the_summary(self) -> None:
        text = build_merge_halt_escalation_text(
            'wip_halted', _HALT_OUTCOMES['wip_halted'], task_id=_TID,
        )

        assert 'x.py' in text.summary

    def test_text_unpacks_as_category_summary_detail(self) -> None:
        category, summary, detail = build_merge_halt_escalation_text(
            'unmerged_state', _HALT_OUTCOMES['unmerged_state'], task_id=_TID,
        )

        assert category == 'unmerged_state'
        assert summary
        assert _TID in detail


class TestSubmitHaltOwningEscalation:
    def test_the_record_is_submitted_before_the_halt_is_owned(self, tmp_path: Path) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        halt = FakeMergeHalt(queue=queue)
        esc = _escalation(queue)

        submit_halt_owning_escalation(queue, halt, esc)

        assert queue.get(esc.id) is not None
        assert halt.owners == [esc.id]
        assert halt.record_present_at_owner == [True]

    def test_a_failed_submit_releases_an_ownerless_halt(self, tmp_path: Path) -> None:
        halt = FakeMergeHalt()
        queue = RaisingQueue(tmp_path / 'esc')
        esc = _escalation(queue)

        with pytest.raises(RuntimeError, match='disk full'):
            submit_halt_owning_escalation(queue, halt, esc)

        assert halt.is_wip_halted is False
        assert halt.unhalt_reasons == ['halt_escalation_submit_failed']
        assert halt.owners == []

    def test_no_merge_halt_still_submits(self, tmp_path: Path) -> None:
        queue = EscalationQueue(tmp_path / 'esc')
        esc = _escalation(queue)

        submit_halt_owning_escalation(queue, None, esc)

        assert queue.get(esc.id) is not None
