"""Tests for the merge_status response model and the provenance invariants it enforces.

PRD ``plans/merge-status-durable-non-landed-prd.md`` D1/D2/D4/D8 and invariants
I1/I3.  Everything here is pure: no MCP server, no event store, no git.  The
tier walk that PRODUCES these answers is pinned through ``create_server`` in
``test_merge_status_tiers.py``.
"""
from __future__ import annotations

import dataclasses
import enum
from typing import Any

import pytest
from shared.merge_state import MergeReason, MergeState

from escalation.merge_status_answer import (
    PRESENT_TRUTH_SOURCES,
    LastOutcome,
    MergeStatusAnswer,
    MergeStatusSource,
    hint_for,
    map_terminal_state,
)

_PRESENT = (MergeStatusSource.live, MergeStatusSource.event_store, MergeStatusSource.git)
_HISTORICAL = (
    MergeStatusSource.journal,
    MergeStatusSource.event_store_cross_run,
    MergeStatusSource.none,
)


def _row(**overrides: Any) -> dict[str, Any]:
    """A row in ``EventStore.latest_merge_finalized``'s projection."""
    row: dict[str, Any] = {
        'request_id': 'mr-1',
        'task_id': '42',
        'branch': 'task/42',
        'state': 'wip_halted',
        'snapshot_tip': 'a' * 40,
        'merge_sha': None,
        'superseded_by': None,
        'reason': 'wip present on main checkout',
        'finished_at': '2026-10-09T12:00:00+00:00',
        'run_id': 'run-prior',
        'is_current_run': False,
        'absorbed_request_ids': None,
    }
    row.update(overrides)
    return row


class TestMergeStatusSource:
    def test_is_a_str_enum(self) -> None:
        assert issubclass(MergeStatusSource, enum.StrEnum)

    def test_exact_members(self) -> None:
        assert {s.value for s in MergeStatusSource} == {
            'live', 'event_store', 'git', 'journal', 'event_store_cross_run', 'none',
        }

    def test_present_truth_sources(self) -> None:
        assert frozenset(_PRESENT) == PRESENT_TRUTH_SOURCES


class TestMapTerminalState:
    @pytest.mark.parametrize(
        'raw,expected',
        [
            ('done', MergeState.done),
            ('done_wip_recovery', MergeState.done),
            ('already_merged', MergeState.done),
            ('conflict', MergeState.conflict),
            ('abandoned', MergeState.abandoned),
            ('superseded', MergeState.superseded),
            ('blocked', MergeState.blocked),
            ('wip_halted', MergeState.blocked),
            ('wip_recovery_no_advance', MergeState.blocked),
            ('unmerged_state', MergeState.blocked),
            ('stash_failed', MergeState.blocked),
            ('unknown_branch', MergeState.blocked),
            ('error', MergeState.blocked),
        ],
    )
    def test_collapse(self, raw: str, expected: MergeState) -> None:
        assert map_terminal_state(raw) is expected


class TestOutcomeAnswers:
    @pytest.mark.parametrize('source', _PRESENT)
    def test_renders_state_source_and_fields(self, source: MergeStatusSource) -> None:
        answer = MergeStatusAnswer.outcome(
            source=source,
            fields={
                'state': MergeState.done,
                'request_id': 'mr-1',
                'outcome': 'done',
                'finished_at': '2026-10-09T12:00:00+00:00',
            },
        )

        assert answer.to_response() == {
            'state': MergeState.done,
            'request_id': 'mr-1',
            'generation': 1,
            'source': source.value,
            'outcome': 'done',
            'finished_at': '2026-10-09T12:00:00+00:00',
            'last_outcome': None,
        }

    def test_event_store_free_text_reason_is_rendered_verbatim(self) -> None:
        response = MergeStatusAnswer.outcome(
            source=MergeStatusSource.event_store,
            fields={
                'state': MergeState.blocked,
                'request_id': 'mr-1',
                'outcome': 'wip_halted',
                'reason': 'wip present on main checkout',
            },
        ).to_response()

        assert response['reason'] == 'wip present on main checkout'
        assert 'hint' not in response

    def test_a_wrapped_tier_dict_at_the_rendered_generation_is_accepted(self) -> None:
        response = MergeStatusAnswer.outcome(
            source=MergeStatusSource.git,
            fields={
                'state': MergeState.done, 'request_id': None, 'generation': 1,
                'kind': 'found_on_main', 'merge_sha': 'c' * 40,
                'outcome': 'found_on_main',
            },
        ).to_response()

        assert response['generation'] == 1
        assert response['kind'] == 'found_on_main'
        assert response['merge_sha'] == 'c' * 40

    def test_a_wrapped_tier_dict_at_another_generation_is_refused(self) -> None:
        with pytest.raises(ValueError, match='generation'):
            MergeStatusAnswer.outcome(
                source=MergeStatusSource.git,
                fields={
                    'state': MergeState.done, 'request_id': None, 'generation': 2,
                    'kind': 'found_on_main', 'merge_sha': 'c' * 40,
                    'outcome': 'found_on_main',
                },
            )

    @pytest.mark.parametrize('source', _HISTORICAL)
    def test_an_outcome_state_from_history_violates_i1(
        self, source: MergeStatusSource
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            MergeStatusAnswer.outcome(
                source=source,
                fields={'state': MergeState.blocked, 'request_id': 'mr-1'},
            )

        message = str(excinfo.value)
        assert 'I1' in message
        assert 'blocked' in message
        assert source.value in message

    def test_an_unrecognised_live_passthrough_is_accepted(self) -> None:
        response = MergeStatusAnswer.outcome(
            source=MergeStatusSource.live,
            fields={'state': 'dispatching', 'request_id': 'mr-1', 'position': 0},
        ).to_response()

        assert response['state'] == 'dispatching'
        assert response['source'] == 'live'


class TestEpistemicAnswers:
    @pytest.mark.parametrize(
        'state,reason,source',
        [
            (MergeState.no_record, MergeReason.never_submitted, MergeStatusSource.none),
            (
                MergeState.stale_record, MergeReason.prior_run_terminal,
                MergeStatusSource.event_store_cross_run,
            ),
            (MergeState.journaled, MergeReason.journaled_not_live, MergeStatusSource.journal),
        ],
    )
    def test_each_record_state_pairs_with_its_source(
        self, state: MergeState, reason: MergeReason, source: MergeStatusSource
    ) -> None:
        response = MergeStatusAnswer.epistemic(
            state, reason=reason, request_id='mr-1', source=source,
        ).to_response()

        assert response['state'] is state
        assert response['source'] == source.value
        assert response['reason'] == reason.value
        assert response['hint'] == hint_for(reason)
        assert response['last_outcome'] is None

    @pytest.mark.parametrize(
        'state,reason',
        [
            (MergeState.no_record, MergeReason.never_submitted),
            (MergeState.stale_record, MergeReason.prior_run_terminal),
            (MergeState.journaled, MergeReason.journaled_not_live),
        ],
    )
    def test_the_paired_source_is_the_default(
        self, state: MergeState, reason: MergeReason
    ) -> None:
        answer = MergeStatusAnswer.epistemic(state, reason=reason, request_id='mr-1')

        assert answer.source in frozenset(MergeStatusSource) - PRESENT_TRUTH_SOURCES

    @pytest.mark.parametrize(
        'state,reason,wrong_source',
        [
            (MergeState.no_record, MergeReason.never_submitted, MergeStatusSource.journal),
            (MergeState.stale_record, MergeReason.prior_run_terminal, MergeStatusSource.none),
            (
                MergeState.journaled, MergeReason.journaled_not_live,
                MergeStatusSource.event_store_cross_run,
            ),
        ],
    )
    def test_a_mismatched_source_names_the_pairing(
        self, state: MergeState, reason: MergeReason, wrong_source: MergeStatusSource
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            MergeStatusAnswer.epistemic(
                state, reason=reason, request_id='mr-1', source=wrong_source,
            )

        message = str(excinfo.value)
        assert state.value in message
        assert wrong_source.value in message

    @pytest.mark.parametrize('source', list(MergeStatusSource))
    def test_unknown_accepts_any_failing_tier(self, source: MergeStatusSource) -> None:
        response = MergeStatusAnswer.epistemic(
            MergeState.unknown, reason=MergeReason.probe_failed,
            request_id='mr-1', source=source,
        ).to_response()

        assert response['state'] is MergeState.unknown
        assert response['source'] == source.value
        assert response['hint'] == hint_for(MergeReason.probe_failed)

    def test_unknown_must_name_the_failing_tier(self) -> None:
        with pytest.raises(ValueError):
            MergeStatusAnswer.epistemic(
                MergeState.unknown, reason=MergeReason.probe_failed, request_id='mr-1',
            )

    @pytest.mark.parametrize(
        'reason', [r for r in MergeReason if r is not MergeReason.probe_failed],
    )
    def test_unknown_requires_probe_failed(self, reason: MergeReason) -> None:
        with pytest.raises(ValueError) as excinfo:
            MergeStatusAnswer.epistemic(
                MergeState.unknown, reason=reason, request_id='mr-1',
                source=MergeStatusSource.git,
            )

        assert 'I3' in str(excinfo.value)

    @pytest.mark.parametrize(
        'state', [MergeState.no_record, MergeState.stale_record, MergeState.journaled],
    )
    def test_probe_failed_is_reserved_for_unknown(self, state: MergeState) -> None:
        with pytest.raises(ValueError) as excinfo:
            MergeStatusAnswer.epistemic(
                state, reason=MergeReason.probe_failed, request_id='mr-1',
            )

        assert 'I3' in str(excinfo.value)

    @pytest.mark.parametrize(
        'state,source',
        [
            (MergeState.no_record, MergeStatusSource.none),
            (MergeState.stale_record, MergeStatusSource.event_store_cross_run),
            (MergeState.journaled, MergeStatusSource.journal),
            (MergeState.unknown, MergeStatusSource.git),
        ],
    )
    def test_a_missing_reason_raises(
        self, state: MergeState, source: MergeStatusSource
    ) -> None:
        with pytest.raises(ValueError):
            MergeStatusAnswer.epistemic(
                state, reason=None, request_id='mr-1', source=source,
            )

    def test_landing_unattributable_qualifies_any_record_state(self) -> None:
        for state in (MergeState.no_record, MergeState.stale_record, MergeState.journaled):
            response = MergeStatusAnswer.epistemic(
                state, reason=MergeReason.landing_unattributable, request_id='mr-1',
            ).to_response()
            assert response['reason'] == 'landing_unattributable'

    def test_journaled_carries_its_extra_fields(self) -> None:
        response = MergeStatusAnswer.epistemic(
            MergeState.journaled, reason=MergeReason.journaled_not_live,
            request_id='mr-1', fields={'enqueued_at': 1_700_000_000.0},
        ).to_response()

        assert response['enqueued_at'] == 1_700_000_000.0

    def test_an_outcome_answer_carries_no_merge_reason(self) -> None:
        with pytest.raises(ValueError):
            MergeStatusAnswer(
                state=MergeState.done, source=MergeStatusSource.git, request_id='mr-1',
                reason=MergeReason.never_submitted, fields={}, last_outcome=None,
            )

    @pytest.mark.parametrize(
        'key', ['state', 'source', 'hint', 'last_outcome', 'reason'],
    )
    def test_epistemic_fields_may_not_shadow_rendered_keys(self, key: str) -> None:
        with pytest.raises(ValueError):
            MergeStatusAnswer.epistemic(
                MergeState.no_record, reason=MergeReason.never_submitted,
                request_id='mr-1', fields={key: 'x'},
            )

    @pytest.mark.parametrize('key', ['source', 'hint', 'last_outcome'])
    def test_outcome_fields_may_not_shadow_rendered_keys(self, key: str) -> None:
        with pytest.raises(ValueError):
            MergeStatusAnswer.outcome(
                source=MergeStatusSource.event_store,
                fields={'state': MergeState.done, 'request_id': 'mr-1', key: 'x'},
            )


class TestHints:
    def test_every_reason_has_a_hint(self) -> None:
        for reason in MergeReason:
            assert hint_for(reason).strip(), f'{reason} has an empty hint'

    def test_hints_are_pairwise_distinct(self) -> None:
        hints = [hint_for(reason) for reason in MergeReason]
        assert len(set(hints)) == len(hints)


class TestLastOutcome:
    def test_from_row_renders_the_labelled_projection(self) -> None:
        lo = LastOutcome.from_row(_row(superseded_by='coalesce-x-1', merge_sha='d' * 40))

        assert lo.to_wire() == {
            'state': MergeState.blocked,
            'run_id': 'run-prior',
            'finished_at': '2026-10-09T12:00:00+00:00',
            'is_current_run': False,
            'reason': 'wip present on main checkout',
            'merge_sha': 'd' * 40,
            'superseded_by': 'coalesce-x-1',
            'request_id': 'mr-1',
            'branch': 'task/42',
        }

    def test_state_is_coarse(self) -> None:
        assert LastOutcome.from_row(_row(state='already_merged')).state is MergeState.done

    def test_a_non_terminal_state_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            LastOutcome(
                state=MergeState.queued, run_id='run-prior',
                finished_at='2026-10-09T12:00:00+00:00', is_current_run=False,
                reason=None, merge_sha=None, superseded_by=None,
                request_id='mr-1', branch='task/42',
            )


class TestWithLastOutcome:
    def test_returns_a_new_answer_and_leaves_the_original_unchanged(self) -> None:
        original = MergeStatusAnswer.epistemic(
            MergeState.stale_record, reason=MergeReason.prior_run_terminal,
            request_id='mr-1',
        )
        lo = LastOutcome.from_row(_row())

        enriched = original.with_last_outcome(lo)

        assert enriched is not original
        assert original.last_outcome is None
        assert original.to_response()['last_outcome'] is None
        assert enriched.to_response()['last_outcome'] == lo.to_wire()

    def test_none_leaves_last_outcome_none(self) -> None:
        answer = MergeStatusAnswer.epistemic(
            MergeState.no_record, reason=MergeReason.never_submitted, request_id=None,
        ).with_last_outcome(None)

        assert answer.to_response()['last_outcome'] is None

    def test_is_frozen(self) -> None:
        answer = MergeStatusAnswer.epistemic(
            MergeState.no_record, reason=MergeReason.never_submitted, request_id=None,
        )

        field = 'state'
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(answer, field, MergeState.done)
