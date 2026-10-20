"""orchestrator/src/orchestrator/escalation_router.py — the applier half of the
info-L0 router (plans/info-l0-disposition-router-prd.md D5/D7/D8/D9/D12)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import pytest
from _curator_fake import FakeCurator
from escalation.disposition import (
    Addressed,
    ConvertViaCurator,
    CuratorCandidate,
    Defer,
    DeferReason,
    ExitKind,
    ObservationConsumed,
    PromoteReason,
    PromoteWithHint,
    StatusInfo,
)
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.escalation_router import (
    AGGREGATE_ROLE,
    FALLBACK_ACTION,
    ROUTER_ID,
    Converted,
    Failed,
    Refused,
    RouteContext,
    RouterHold,
    RouterPorts,
    Undecided,
    apply,
    classify_ticket,
    exit_kind_for,
    hold_note,
    read_hold,
    router_owns,
    status_info_sentinel,
)
from orchestrator.event_store import EventStore, EventType
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

    @pytest.mark.asyncio
    @pytest.mark.parametrize('status', ['created', 'combined'])
    async def test_a_named_task_is_converted(self, status):
        body = await self._answer({'status': status, 'task_id': '9001'})
        assert classify_ticket(body) == Converted('9001')

    @pytest.mark.asyncio
    async def test_refused_carries_the_reason(self):
        body = await self._answer({'status': 'refused', 'reason': 'cancelled premise'})
        assert classify_ticket(body) == Refused('cancelled premise')

    @pytest.mark.asyncio
    @pytest.mark.parametrize('reason', ['worker_dead', 'unknown_ticket', 'server_restart'])
    async def test_terminal_failure_carries_the_reason(self, reason):
        body = await self._answer({'status': 'failed', 'reason': reason, 'task_id': None})
        assert classify_ticket(body) == Failed(reason)

    @pytest.mark.asyncio
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


# ---------------------------------------------------------------------------
# The applier
# ---------------------------------------------------------------------------

FILER_RUN_ID = 'harness-filer-run'
CONTEXT = RouteContext(by='orphan-reaper', exit_kind=ExitKind.ORPHAN)
EVENT_KEYS = {'escalation_id', 'class', 'by', 'exit_kind', 'ticket', 'task_id', 'decided_at', 'reason'}


@pytest.fixture
def queue(tmp_path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'escalations')


@pytest.fixture
def events(tmp_path) -> EventStore:
    return EventStore(tmp_path / 'runs.db', run_id='run-test')


@pytest.fixture
def curator() -> FakeCurator:
    return FakeCurator()


@pytest.fixture
def ports(queue, events, curator) -> RouterPorts:
    return RouterPorts(
        queue=queue, event_store=events, dispatch_tool=curator,
        project_root='/srv/project', filing_claimant_run_id=FILER_RUN_ID,
    )


def _seed(queue: EscalationQueue, **fields) -> Escalation:
    record = _record(esc_id=queue.make_id(fields.get('task_id', '42')), **fields)
    queue.submit(record)
    return record


def _current(queue: EscalationQueue, esc_id: str) -> Escalation:
    record = queue.get(esc_id)
    assert record is not None
    return record


def _dispositions(events: EventStore) -> list[dict]:
    return events.fetch_events_by_type(EventType.info_l0_dispositioned)


def _router_l1s(queue: EscalationQueue, task_id: str) -> list[Escalation]:
    return queue.get_by_task(task_id, status='pending', level=1, agent_role=AGGREGATE_ROLE)


def _convert(record: Escalation) -> ConvertViaCurator:
    return ConvertViaCurator(CuratorCandidate(
        title=record.summary, description=f'Escalation: {record.id}',
        escalation_id=record.id, spawned_from=record.task_id,
    ))


class TestApplyCloses:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(('disposition', 'resolution_class'), [
        (Addressed(), 'addressed'),
        (ObservationConsumed(), 'observation-consumed'),
        (StatusInfo('orchestrator-starvation-watchdog'), 'status-info'),
    ])
    async def test_a_close_dismisses_with_the_variant_class_and_emits_one_row(
        self, queue, events, ports, disposition, resolution_class,
    ):
        record = _seed(queue)
        await apply(disposition, record, ports, CONTEXT)

        closed = _current(queue, record.id)
        assert (closed.status, closed.resolved_by, closed.resolution_class) == (
            'dismissed', ROUTER_ID, resolution_class,
        )
        (row,) = _dispositions(events)
        assert row['task_id'] == '42'
        assert set(row['data']) >= EVENT_KEYS
        assert row['data']['escalation_id'] == record.id
        assert row['data']['class'] == resolution_class
        assert (row['data']['by'], row['data']['exit_kind']) == ('orphan-reaper', 'orphan')
        assert row['data']['ticket'] is None
        assert row['data']['task_id'] == '42'
        assert datetime.fromisoformat(row['data']['decided_at']).tzinfo is not None

    @pytest.mark.asyncio
    async def test_a_status_info_close_names_its_class_aggregate(self, queue, ports):
        record = _seed(queue, agent_role='orchestrator-starvation-watchdog')
        await apply(StatusInfo('orchestrator-starvation-watchdog'), record, ports, CONTEXT)
        resolution = _current(queue, record.id).resolution or ''
        assert status_info_sentinel('orchestrator-starvation-watchdog') in resolution

    @pytest.mark.asyncio
    async def test_defer_writes_nothing_and_emits_a_defer_row(self, queue, events, ports):
        record = _seed(queue)
        await apply(Defer(DeferReason.REQUEUED_EXIT), record, ports, CONTEXT)
        after = _current(queue, record.id)
        assert (after.status, after.triaged_by, after.resolution) == ('pending', None, None)
        (row,) = _dispositions(events)
        assert (row['data']['class'], row['data']['reason']) == ('defer', 'requeued_exit')

    @pytest.mark.asyncio
    async def test_a_record_closed_since_it_was_read_gets_no_write_and_no_event(
        self, queue, events, ports,
    ):
        record = _seed(queue)
        queue.resolve(record.id, 'steward fixed it', resolved_by='claude-task-42-steward')
        await apply(Addressed(), record, ports, CONTEXT)
        after = _current(queue, record.id)
        assert (after.status, after.resolved_by) == ('resolved', 'claude-task-42-steward')
        assert _dispositions(events) == []


class TestApplyConverts:
    @pytest.mark.asyncio
    async def test_conversion_files_one_ticket_and_holds_the_record(
        self, queue, events, ports, curator,
    ):
        record = _seed(queue)
        await apply(_convert(record), record, ports, CONTEXT)

        (arguments,) = curator.calls_named('submit_task')
        assert arguments['project_root'] == '/srv/project'
        assert arguments['title'] == record.summary
        assert arguments['description'] == f'Escalation: {record.id}'
        assert arguments['planning_mode'] is False
        assert arguments['priority'] == 'low'
        metadata = arguments['metadata']
        assert {k: metadata[k] for k in ('source', 'spawn_context', 'escalation_id', 'spawned_from')} == {
            'source': ROUTER_ID, 'spawn_context': ROUTER_ID,
            'escalation_id': record.id, 'spawned_from': '42',
        }
        assert isinstance(metadata['suggestion_hash'], str) and metadata['suggestion_hash']

        held = _current(queue, record.id)
        assert (held.status, held.triaged_by) == ('pending', ROUTER_ID)
        hold = read_hold(held)
        assert hold is not None and hold.ticket == 'tkt_1'
        (row,) = _dispositions(events)
        assert (row['data']['class'], row['data']['ticket']) == ('convert-pending', 'tkt_1')

    @pytest.mark.asyncio
    async def test_the_suggestion_hash_is_stable_per_escalation_id(self, queue, ports, curator):
        first, second = _seed(queue), _seed(queue)
        await apply(_convert(first), first, ports, CONTEXT)
        await apply(_convert(second), second, ports, CONTEXT)
        await apply(_convert(first), _current(queue, first.id), ports, CONTEXT)
        hashes = [call['metadata']['suggestion_hash'] for call in curator.calls_named('submit_task')]
        assert hashes[0] == hashes[2]
        assert hashes[0] != hashes[1]

    @pytest.mark.asyncio
    async def test_a_combined_submission_converts_at_once(self, queue, events, ports, curator):
        curator.combine_submits('8123')
        record = _seed(queue)
        await apply(_convert(record), record, ports, CONTEXT)
        closed = _current(queue, record.id)
        assert (closed.status, closed.resolution_class) == ('dismissed', 'converted')
        assert '8123' in (closed.resolution or '')
        (row,) = _dispositions(events)
        assert row['data']['class'] == 'converted'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('break_curator', [
        lambda c: c.fail_submits(),
        lambda c: c.raise_on_submit(),
    ], ids=['failure-payload', 'dispatch-raises'])
    async def test_a_submission_the_curator_did_not_take_promotes_ticket_failed(
        self, queue, events, ports, curator, break_curator,
    ):
        break_curator(curator)
        record = _seed(queue)
        await apply(_convert(record), record, ports, CONTEXT)
        (l1,) = _router_l1s(queue, '42')
        assert 'ticket_failed' in l1.detail
        assert _current(queue, record.id).status == 'dismissed'
        (row,) = _dispositions(events)
        assert (row['data']['class'], row['data']['reason']) == ('promoted', 'ticket_failed')


class TestApplyPromotes:
    @pytest.mark.asyncio
    async def test_the_fallback_l1_preserves_the_author_and_hints_conversion(self, queue, ports):
        record = _seed(
            queue, severity='info', category='design_concern',
            suggested_action='split the helper into shared/', detail='Original author text.',
        )
        await apply(PromoteWithHint(PromoteReason.NO_CONVERTIBLE_CONTENT), record, ports, CONTEXT)

        (l1,) = _router_l1s(queue, '42')
        assert (l1.level, l1.task_id, l1.agent_role) == (1, '42', 'harness-info-l0-router')
        assert l1.suggested_action == FALLBACK_ACTION
        assert l1.suggested_action != 'manual_intervention'
        assert (l1.severity, l1.category) == ('info', 'design_concern')
        assert l1.detail.startswith('Original author text.')
        for fact in ('no_convertible_content', record.id, 'split the helper into shared/'):
            assert fact in l1.detail
        assert l1.filing_claimant_run_id == FILER_RUN_ID

        dismissed = _current(queue, record.id)
        assert dismissed.status == 'dismissed'
        assert l1.id in (dismissed.resolution or '')

    @pytest.mark.asyncio
    async def test_an_empty_detail_starts_the_l1_with_the_summary(self, queue, ports):
        record = _seed(queue, detail='', summary='just a summary line')
        await apply(PromoteWithHint(PromoteReason.CONVERSION_CAP), record, ports, CONTEXT)
        (l1,) = _router_l1s(queue, '42')
        assert l1.detail.startswith('just a summary line')


class TestApplyNeverRaises:
    @staticmethod
    def _break_resolve(queue: EscalationQueue, esc_id: str, monkeypatch) -> None:
        real_resolve = queue.resolve

        def resolve(escalation_id, *args, **kwargs):
            if escalation_id == esc_id:
                raise OSError('disk full')
            return real_resolve(escalation_id, *args, **kwargs)

        monkeypatch.setattr(queue, 'resolve', resolve)

    @pytest.mark.asyncio
    async def test_a_failing_queue_write_files_a_router_error_l1(
        self, queue, events, ports, monkeypatch, caplog,
    ):
        record = _seed(queue)
        self._break_resolve(queue, record.id, monkeypatch)
        with caplog.at_level(logging.ERROR, logger='orchestrator.escalation_router'):
            await apply(Addressed(), record, ports, CONTEXT)

        (l1,) = _router_l1s(queue, '42')
        assert 'router_error' in l1.detail
        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert any(
            record.id in message and 'Addressed' in message and 'orphan-reaper' in message
            for message in errors
        )

    @pytest.mark.asyncio
    async def test_when_filing_also_fails_apply_still_returns(
        self, queue, events, ports, monkeypatch,
    ):
        record = _seed(queue)
        self._break_resolve(queue, record.id, monkeypatch)

        def submit(_escalation):
            raise OSError('disk full')

        monkeypatch.setattr(queue, 'submit', submit)
        applied = await apply(Addressed(), record, ports, CONTEXT)
        assert applied.disposition_class == 'router-error'
        (row,) = _dispositions(events)
        assert row['data']['class'] == 'router-error'
