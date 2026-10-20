"""orchestrator/src/orchestrator/escalation_router.py — the applier half of the
info-L0 router (plans/info-l0-disposition-router-prd.md D5/D7/D8/D9/D12)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import pytest
from _curator_fake import FakeCurator
from escalation.disposition import ExitKind
from escalation.models import Escalation

from orchestrator.escalation_router import (
    ROUTER_ID,
    Converted,
    Failed,
    Refused,
    RouterHold,
    Undecided,
    classify_ticket,
    exit_kind_for,
    hold_note,
    read_hold,
    router_owns,
)
from orchestrator.workflow_types import WorkflowOutcome


def _record(
    *, severity: str = 'info', level: int = 0, esc_id: str = 'esc-42-1', task_id: str = '42',
    agent_role: str = 'implementer', summary: str = 'helper X is duplicated in two modules',
    detail: str = 'Seen while editing foo.py; bar.py carries a copy.',
    suggested_action: str = 'create_followup_task', category: str = 'design_concern',
    triaged_by: str | None = None, triage_note: str = '',
) -> Escalation:
    return Escalation(
        id=esc_id, task_id=task_id, agent_role=agent_role, severity=severity,
        category=category, summary=summary, detail=detail,
        suggested_action=suggested_action, level=level,
        triaged_by=triaged_by, triage_note=triage_note,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestRouterOwns:
    def test_enabled_info_l0_is_owned(self):
        assert router_owns(_record(), enabled=True)

    def test_padded_upper_case_info_counts_as_info(self):
        assert router_owns(_record(severity=' INFO '), enabled=True)

    @pytest.mark.parametrize('severity', ['blocking', 'critical'])
    def test_non_info_l0_is_not_owned(self, severity):
        assert not router_owns(_record(severity=severity), enabled=True)

    def test_info_l1_is_not_owned(self):
        assert not router_owns(_record(level=1), enabled=True)

    def test_disabled_router_owns_nothing(self):
        assert not router_owns(_record(), enabled=False)


class TestExitKindFor:
    @pytest.mark.parametrize('cancel_kind', ['hard', 'soft'])
    @pytest.mark.parametrize('outcome', [*WorkflowOutcome, None])
    def test_any_cancellation_is_a_cancelled_exit(self, cancel_kind, outcome):
        assert exit_kind_for(cancel_kind, outcome) is ExitKind.CANCELLED

    @pytest.mark.parametrize(('outcome', 'expected'), [
        (WorkflowOutcome.DONE, ExitKind.DONE),
        (WorkflowOutcome.MERGE_DEFERRED, ExitKind.MERGE_DEFERRED),
        (WorkflowOutcome.BLOCKED, ExitKind.BLOCKED),
        (WorkflowOutcome.ESCALATED, ExitKind.BLOCKED),
        (WorkflowOutcome.CANCELLED, ExitKind.CANCELLED),
        (WorkflowOutcome.SOFT_CANCELLED, ExitKind.CANCELLED),
        (WorkflowOutcome.REQUEUED, ExitKind.REQUEUED),
        (WorkflowOutcome.PLANNED, ExitKind.REQUEUED),
    ])
    def test_uncancelled_exit_follows_the_outcome(self, outcome, expected):
        assert exit_kind_for(None, outcome) is expected

    def test_unknown_exit_routes_nothing(self):
        assert exit_kind_for(None, None) is None


class TestRouterHold:
    FILED_AT = datetime(2026, 10, 20, 12, 30, 5, tzinfo=UTC)

    def test_hold_note_round_trips_through_read_hold(self):
        record = _record(triaged_by=ROUTER_ID, triage_note=hold_note('tkt_7', self.FILED_AT))
        assert read_hold(record) == RouterHold('tkt_7', self.FILED_AT)

    def test_unstamped_record_has_no_hold(self):
        assert read_hold(_record()) is None

    def test_record_triaged_by_someone_else_has_no_hold(self):
        record = _record(
            triaged_by='escalation-watcher-auto', triage_note=hold_note('tkt_7', self.FILED_AT),
        )
        assert read_hold(record) is None

    @pytest.mark.parametrize('note', [
        '',
        'verified: still relevant',
        'info-l0-router hold: not json',
        'info-l0-router hold: {"ticket": "tkt_7"}',
        'info-l0-router hold: {"ticket": 7, "filed_at": "2026-10-20T12:30:05+00:00"}',
        'info-l0-router hold: {"ticket": "tkt_7", "filed_at": "yesterday"}',
    ])
    def test_router_stamp_with_a_foreign_note_reads_as_no_hold_and_warns(self, note, caplog):
        record = _record(triaged_by=ROUTER_ID, triage_note=note)
        with caplog.at_level(logging.WARNING, logger='orchestrator.escalation_router'):
            assert read_hold(record) is None
        assert any(record.id in r.getMessage() for r in caplog.records)


class TestClassifyTicket:
    @staticmethod
    async def _answer(payload: dict) -> dict:
        curator = FakeCurator()
        curator.set_ticket('tkt_1', payload)
        return await curator('resolve_ticket', {'ticket': 'tkt_1'})

    @pytest.mark.parametrize('status', ['created', 'combined'])
    async def test_a_named_task_is_converted(self, status):
        body = await self._answer({'status': status, 'task_id': '9001'})
        assert classify_ticket(body) == Converted('9001')

    async def test_refused_carries_the_reason(self):
        body = await self._answer({'status': 'refused', 'reason': 'cancelled premise'})
        assert classify_ticket(body) == Refused('cancelled premise')

    @pytest.mark.parametrize('reason', ['worker_dead', 'unknown_ticket', 'server_restart'])
    async def test_terminal_failure_carries_the_reason(self, reason):
        body = await self._answer({'status': 'failed', 'reason': reason, 'task_id': None})
        assert classify_ticket(body) == Failed(reason)

    @pytest.mark.parametrize('payload', [
        {'status': 'failed', 'reason': 'timeout', 'task_id': None},
        {'status': 'failed', 'reason': 'server_closed', 'task_id': None},
        {'status': 'awaiting_curation', 'reason': 'pool exhausted'},
    ])
    async def test_a_ticket_the_curator_may_still_decide_is_undecided(self, payload):
        assert isinstance(classify_ticket(await self._answer(payload)), Undecided)

    @pytest.mark.parametrize('body', [
        {},
        {'jsonrpc': '2.0', 'id': 1, 'result': {'content': []}},
        {'jsonrpc': '2.0', 'id': 1, 'result': {'content': [{'type': 'text', 'text': 'nope'}]}},
        {'jsonrpc': '2.0', 'id': 1, 'result': {'content': [
            {'type': 'text', 'text': json.dumps({'error': 'bad', 'error_type': 'ValidationError'})},
        ]}},
    ])
    def test_an_unreadable_answer_is_undecided(self, body):
        assert isinstance(classify_ticket(body), Undecided)
