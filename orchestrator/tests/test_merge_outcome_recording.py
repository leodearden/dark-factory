"""What a blocked or conflicted merge outcome records for the workflow and runs.db (task 7302)."""

from __future__ import annotations

import dataclasses

import pytest
from escalation.categories import EscalationCategory

from orchestrator.merge_lane.outcome_facts import (
    MAX_RECORDED_CONFLICTED_PATHS,
    ConflictEvidence,
    MergeBlockCategory,
)
from orchestrator.merge_lane.types import MergeOutcome


class TestOutcomeFacts:
    @pytest.mark.parametrize('member', list(MergeBlockCategory))
    def test_every_block_category_is_a_registered_escalation_category(
        self, member: MergeBlockCategory,
    ) -> None:
        assert member.value in EscalationCategory

    def test_needs_rebase_spelling(self) -> None:
        assert MergeBlockCategory.MERGE_NEEDS_REBASE == 'merge_needs_rebase'

    def test_unattributed_evidence_omits_counterparts(self) -> None:
        evidence = ConflictEvidence(conflicted_paths=('a.py', 'b.py'))

        assert evidence.to_event_payload() == {
            'conflicted_paths': ['a.py', 'b.py'],
            'conflicted_paths_total': 2,
        }

    @pytest.mark.parametrize(
        ('counterparts', 'recorded'),
        [((), []), (('7202',), ['7202'])],
    )
    def test_attributed_evidence_records_counterparts(
        self, counterparts: tuple[str, ...], recorded: list[str],
    ) -> None:
        evidence = ConflictEvidence(
            conflicted_paths=('a.py',), counterpart_task_ids=counterparts,
        )

        assert evidence.to_event_payload()['counterpart_task_ids'] == recorded

    def test_recorded_paths_are_bounded_and_the_true_count_kept(self) -> None:
        paths = tuple(f'f{i}.py' for i in range(MAX_RECORDED_CONFLICTED_PATHS + 5))

        payload = ConflictEvidence(conflicted_paths=paths).to_event_payload()

        assert payload['conflicted_paths'] == list(paths[:MAX_RECORDED_CONFLICTED_PATHS])
        assert payload['conflicted_paths_total'] == MAX_RECORDED_CONFLICTED_PATHS + 5

    def test_evidence_is_frozen(self) -> None:
        evidence = ConflictEvidence(conflicted_paths=('a.py',))

        pytest.raises(
            dataclasses.FrozenInstanceError, setattr, evidence, 'conflicted_paths', ('b.py',),
        )

    def test_bare_blocked_outcome_defaults(self) -> None:
        outcome = MergeOutcome('blocked', reason='x')

        assert outcome.block_category is MergeBlockCategory.MERGE_ERROR
        assert outcome.conflict_evidence is None
