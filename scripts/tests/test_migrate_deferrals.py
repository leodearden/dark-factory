"""Tests for ``scripts/migrate_deferrals.py``: every unrecorded deferred row
mapped to the record it should carry, and re-stamped through the guarded write.

The mapping section is pure, over ``deferral_store.TaskRow.from_wire`` rows and
a fixed NOW. Records are built with ``shared.task_deferral``'s own models.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import migrate_deferrals
import pytest
from deferral_store import TaskRow
from migrate_deferrals import (
    AlreadyRecorded,
    Fallback,
    RowPlan,
    Source,
    carrier_reason,
    fallback_plan,
    plan_row,
)
from shared.task_deferral import (
    REASON_MAX_CHARS,
    DeferralInvalidReason,
    DeferralKind,
    NeedsHumanReason,
    PlanningDeferral,
    parse_request,
    record_to_json,
)
from shared.task_statuses import TaskStatus

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
HUMAN_DECOMPOSED_NOTE = 'human_decomposed: possibly an uncommitted planning birth'


def wire_row(task_id: int, *, status: str = 'deferred', metadata: object = None, **extra: Any) -> dict[str, Any]:
    return {
        'id': str(task_id),
        'title': f'task {task_id}',
        'status': status,
        'metadata': dict(extra) if metadata is None else metadata,
    }


def row(task_id: int = 1, **metadata: Any) -> TaskRow:
    return TaskRow.from_wire(wire_row(task_id, **metadata))


def planned(task_row: TaskRow) -> RowPlan:
    plan = plan_row(task_row, now=NOW)
    assert isinstance(plan, RowPlan)
    return plan


def planning_record() -> dict[str, Any]:
    return record_to_json(PlanningDeferral(deferred_at=NOW, stamped_at=NOW, stamped_by='claude'))


class TestPlanRow:
    @pytest.mark.parametrize('carrier', [5255, '5255'])
    def test_x_coalesced_into_maps_to_carried_by(self, carrier: object) -> None:
        plan = planned(row(7, x_coalesced_into=carrier))

        assert plan.request['kind'] == DeferralKind.CARRIED_BY
        assert plan.request['carrier_task_id'] == 5255
        assert plan.source == Source.X_COALESCED_INTO == 'x_coalesced_into'
        assert plan.fallback is None
        assert plan.request['reason'] == 'migrated from metadata.x_coalesced_into'
        assert (plan.task_id, plan.title) == (7, 'task 7')

    def test_a_deferred_watch_trigger_maps_to_until_condition(self) -> None:
        plan = planned(row(deferred_watch=True, trigger='the upstream API ships'))

        assert plan.request['kind'] == DeferralKind.UNTIL_CONDITION
        assert plan.request['condition'] == 'the upstream API ships'
        assert plan.source == Source.TRIGGER == 'trigger'

    def test_x_armed_by_maps_to_until_condition(self) -> None:
        plan = planned(row(x_armed_by='esc-4827 is ruled'))

        assert plan.request['kind'] == DeferralKind.UNTIL_CONDITION
        assert plan.request['condition'] == 'esc-4827 is ruled'
        assert plan.source == Source.X_ARMED_BY == 'x_armed_by'

    def test_x_coalesced_into_wins_over_trigger_and_x_armed_by(self) -> None:
        plan = planned(row(x_coalesced_into=5255, deferred_watch=True, trigger='t', x_armed_by='a'))

        assert plan.source == Source.X_COALESCED_INTO

    def test_trigger_wins_over_x_armed_by(self) -> None:
        plan = planned(row(deferred_watch=True, trigger='t', x_armed_by='a'))

        assert plan.source == Source.TRIGGER
        assert plan.request['condition'] == 't'

    @pytest.mark.parametrize(
        'metadata',
        [
            {'x_armed_by': None},
            {'deferred_watch': None, 'trigger': 't'},
            {'x_coalesced_into': None},
            {'deferred_watch': True},
            {'deferred_watch': True, 'trigger': None},
            {},
        ],
        ids=['armed-null', 'watch-null', 'coalesced-null', 'watch-no-trigger', 'trigger-null', 'bare'],
    )
    def test_values_not_keys_decide_and_nothing_maps_to_legacy_unknown(self, metadata: dict[str, Any]) -> None:
        plan = planned(row(**metadata))

        assert plan.request['kind'] == DeferralKind.LEGACY_UNKNOWN
        assert plan.source is None
        assert plan.fallback is None

    @pytest.mark.parametrize(('flag', 'noted'), [(True, True), ('yes', True), (False, False), (None, False)])
    def test_legacy_unknown_notes_a_human_decomposed_row(self, flag: object, noted: bool) -> None:
        plan = planned(row(human_decomposed=flag))

        assert (HUMAN_DECOMPOSED_NOTE in plan.request['reason']) is noted

    def test_a_mapped_request_carries_no_human_decomposed_note(self) -> None:
        plan = planned(row(x_coalesced_into=5255, human_decomposed=True))

        assert HUMAN_DECOMPOSED_NOTE not in plan.request['reason']

    def test_a_row_with_a_valid_record_is_already_recorded(self) -> None:
        plan = plan_row(row(8, deferral=planning_record(), x_coalesced_into=5255), now=NOW)

        assert plan == AlreadyRecorded(task_id=8, title='task 8', kind=DeferralKind.PLANNING)

    def test_a_row_with_an_invalid_record_is_planned(self) -> None:
        plan = planned(row(9, deferral={'kind': 'carried_by'}, x_coalesced_into=5255))

        assert plan.request['kind'] == DeferralKind.CARRIED_BY

    def test_unreadable_metadata_is_planned_legacy_unknown(self) -> None:
        plan = planned(TaskRow.from_wire(wire_row(10, metadata='[1]')))

        assert plan.request['kind'] == DeferralKind.LEGACY_UNKNOWN

    def test_an_oversized_condition_is_a_predicted_fallback(self) -> None:
        text = 'armed by ' + 'x' * 3000

        plan = planned(row(11, x_armed_by=text))

        assert plan.request['kind'] == DeferralKind.LEGACY_UNKNOWN
        assert plan.source == Source.X_ARMED_BY
        assert plan.fallback == Fallback(
            source=Source.X_ARMED_BY, value=text, reason_code=DeferralInvalidReason.TEXT_TOO_LONG
        )
        reason = plan.request['reason']
        assert len(reason) <= REASON_MAX_CHARS
        assert 'x_armed_by' in reason and 'text_too_long' in reason and text[:40] in reason

    def test_a_non_decimal_carrier_is_a_predicted_fallback(self) -> None:
        plan = planned(row(12, x_coalesced_into='soon'))

        assert plan.request['kind'] == DeferralKind.LEGACY_UNKNOWN
        assert plan.fallback is not None
        assert plan.fallback.source == Source.X_COALESCED_INTO
        assert '"soon"' in plan.request['reason']

    @pytest.mark.parametrize(
        'task_row',
        [
            row(x_coalesced_into=5255),
            row(x_coalesced_into='5255'),
            row(deferred_watch=True, trigger='t'),
            row(x_armed_by='a'),
            row(human_decomposed=True),
            row(x_armed_by='y' * 3000, human_decomposed=True),
            row(),
        ],
    )
    def test_every_planned_request_round_trips_the_shared_parser(self, task_row: TaskRow) -> None:
        request = planned(task_row).request

        parsed = parse_request(request, now=NOW)

        assert parsed.model_dump(mode='json', exclude_none=True) == request
        assert json.loads(json.dumps(request)) == request


class TestFallbackPlan:
    def test_a_server_refusal_falls_back_to_legacy_unknown_naming_source_and_value(self) -> None:
        plan = planned(row(13, x_coalesced_into=5255, human_decomposed=True))

        fallback = fallback_plan(plan, reason_code=DeferralInvalidReason.CARRIER_NOT_FOUND)

        assert (fallback.task_id, fallback.title, fallback.source) == (13, 'task 13', Source.X_COALESCED_INTO)
        assert fallback.fallback == Fallback(
            source=Source.X_COALESCED_INTO, value=5255, reason_code=DeferralInvalidReason.CARRIER_NOT_FOUND
        )
        assert fallback.request['kind'] == DeferralKind.LEGACY_UNKNOWN
        reason = fallback.request['reason']
        assert 'x_coalesced_into' in reason and '5255' in reason and 'carrier_not_found' in reason
        assert HUMAN_DECOMPOSED_NOTE in reason
        assert parse_request(fallback.request, now=NOW).model_dump(mode='json', exclude_none=True) == fallback.request

    def test_the_refused_value_is_the_source_value_as_read(self) -> None:
        plan = planned(row(14, deferred_watch=True, trigger='the API ships'))

        fallback = fallback_plan(plan, reason_code=DeferralInvalidReason.TEXT_TOO_LONG)

        assert fallback.fallback is not None
        assert fallback.fallback.value == 'the API ships'
        assert '"the API ships"' in fallback.request['reason']


class TestCarrierReason:
    STATUSES = {900: TaskStatus.DONE, 901: TaskStatus.CANCELLED, 902: TaskStatus.PENDING}

    @pytest.mark.parametrize(
        ('carrier', 'expected'),
        [
            (900, NeedsHumanReason.CLOSABLE),
            (901, NeedsHumanReason.ORPHANED),
            (903, NeedsHumanReason.CARRIER_MISSING),
            (902, None),
        ],
    )
    def test_a_carried_by_plan_reads_its_carriers_status(
        self, carrier: int, expected: NeedsHumanReason | None
    ) -> None:
        plan = planned(row(x_coalesced_into=carrier))

        assert carrier_reason(plan, self.STATUSES, now=NOW) is expected

    @pytest.mark.parametrize('task_row', [row(x_armed_by='a'), row()], ids=['until-condition', 'legacy'])
    def test_any_other_plan_has_no_carrier_reason(self, task_row: TaskRow) -> None:
        assert carrier_reason(planned(task_row), self.STATUSES, now=NOW) is None

    def test_the_migration_agent_id_is_named(self) -> None:
        assert migrate_deferrals.MIGRATION_AGENT_ID == 'migrate-deferrals'
