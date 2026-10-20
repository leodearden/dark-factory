"""orchestrator/src/orchestrator/escalation_router.py — the applier half of the
info-L0 router (plans/info-l0-disposition-router-prd.md D5/D7/D8/D9/D12)."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

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
    AGGREGATE_CATEGORY,
    AGGREGATE_ROLE,
    FALLBACK_ACTION,
    OVERFLOW_SENTINEL,
    ROUTER_ID,
    Converted,
    Failed,
    InfoL0Sweep,
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
        'info-l0-router hold: {"ticket": "tkt_7", "filed_at": "2026-10-20T12:30:05"}',
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


# ---------------------------------------------------------------------------
# The per-pass sweep
# ---------------------------------------------------------------------------

STARVATION = 'orchestrator-starvation-watchdog'
SKEW = 'orchestrator-merge-skew-tripwire'


def _sweep(ports: RouterPorts, *, max_conversions: int = 20) -> InfoL0Sweep:
    return InfoL0Sweep(ports, by='orphan-reaper', max_conversions=max_conversions)


async def _route_all(sweep: InfoL0Sweep, records: list[Escalation]) -> None:
    for record in records:
        await sweep.route(record, exit_kind=ExitKind.ORPHAN, task_status=None)


def _aggregates(queue: EscalationQueue, sentinel: str) -> list[Escalation]:
    return [
        esc for esc in queue.get_by_task(sentinel, status='pending', level=1, agent_role=AGGREGATE_ROLE)
        if esc.category == AGGREGATE_CATEGORY
    ]


def _hold(queue: EscalationQueue, record: Escalation, ticket: str, filed_at: datetime) -> Escalation:
    queue.stamp_triage(record.id, triaged_by=ROUTER_ID, triage_note=hold_note(ticket, filed_at))
    return _current(queue, record.id)


class TestSweepRoute:
    @pytest.mark.asyncio
    async def test_a_work_note_is_decided_by_route_info_l0_and_converted(self, queue, ports, curator):
        record = _seed(queue)
        sweep = _sweep(ports)
        await sweep.route(record, exit_kind=ExitKind.DONE, task_status='done')
        (arguments,) = curator.calls_named('submit_task')
        assert arguments['metadata']['escalation_id'] == record.id
        assert 'Subject status: done' in arguments['description']

    @pytest.mark.asyncio
    async def test_a_router_held_record_is_skipped(self, queue, events, ports, curator):
        record = _hold(queue, _seed(queue), 'tkt_77', datetime.now(UTC))
        sweep = _sweep(ports)
        await sweep.route(record, exit_kind=ExitKind.ORPHAN)
        assert curator.calls == []
        assert _dispositions(events) == []
        assert read_hold(_current(queue, record.id)) == read_hold(record)

    @pytest.mark.asyncio
    async def test_conversions_past_the_cap_are_promoted_and_aggregated(self, queue, ports, curator):
        records = [_seed(queue) for _ in range(5)]
        sweep = _sweep(ports, max_conversions=2)
        await _route_all(sweep, records)
        sweep.close()

        assert len(curator.calls_named('submit_task')) == 2
        overflow_ids = [r.id for r in records[2:]]
        assert [_current(queue, r.id).status for r in records] == ['pending'] * 2 + ['dismissed'] * 3
        promoted = _router_l1s(queue, '42')
        assert len(promoted) == 3
        assert all('conversion_cap' in l1.detail for l1 in promoted)

        (aggregate,) = _aggregates(queue, OVERFLOW_SENTINEL)
        assert (aggregate.level, aggregate.severity, aggregate.agent_role) == (1, 'info', AGGREGATE_ROLE)
        assert aggregate.filing_claimant_run_id == FILER_RUN_ID
        assert all(esc_id in aggregate.detail for esc_id in overflow_ids)
        assert not any(r.id in aggregate.detail for r in records[:2])

    @pytest.mark.asyncio
    async def test_after_one_unreachable_submit_later_conversions_promote_without_dispatch(
        self, queue, events, ports, curator,
    ):
        curator.raise_on_submit()
        records = [_seed(queue) for _ in range(3)]
        sweep = _sweep(ports)
        await _route_all(sweep, records)

        assert len(curator.calls_named('submit_task')) == 1
        reasons = [row['data']['reason'] for row in _dispositions(events)]
        assert reasons == ['ticket_failed'] * 3
        assert all(_current(queue, r.id).status == 'dismissed' for r in records)


class TestSweepStatusInfoAggregates:
    @staticmethod
    def _mechanical(queue: EscalationQueue, role: str, n: int) -> list[Escalation]:
        return [
            _seed(queue, task_id=str(500 + i), agent_role=role, category='infra_issue')
            for i in range(n)
        ]

    @pytest.mark.asyncio
    async def test_one_aggregate_per_class_lists_that_class_only(self, queue, ports, curator):
        starved = self._mechanical(queue, STARVATION, 5)
        skewed = self._mechanical(queue, SKEW, 2)
        sweep = _sweep(ports)
        await _route_all(sweep, starved + skewed)
        sweep.close()

        assert curator.calls == []
        for record in starved + skewed:
            assert _current(queue, record.id).resolution_class == 'status-info'
        for role, members, others in ((STARVATION, starved, skewed), (SKEW, skewed, starved)):
            (aggregate,) = _aggregates(queue, status_info_sentinel(role))
            assert aggregate.task_id == status_info_sentinel(role)
            assert aggregate.severity == 'info'
            assert all(r.id in aggregate.detail for r in members)
            assert not any(r.id in aggregate.detail for r in others)

    @pytest.mark.asyncio
    async def test_a_later_sweep_amends_the_still_open_aggregate(self, queue, ports):
        first = _sweep(ports)
        await _route_all(first, self._mechanical(queue, STARVATION, 2))
        first.close()
        (aggregate,) = _aggregates(queue, status_info_sentinel(STARVATION))
        queue.submit(Escalation(
            id=queue.make_id('l2'), task_id='l2', agent_role='escalation-watcher-auto',
            severity='info', category=AGGREGATE_CATEGORY, summary='cluster', level=2,
            members=[aggregate.id],
        ))

        later = [_seed(queue, task_id='600', agent_role=STARVATION, category='infra_issue')]
        second = _sweep(ports)
        await _route_all(second, later)
        second.close()

        (still,) = _aggregates(queue, status_info_sentinel(STARVATION))
        assert still.id == aggregate.id
        (amendment,) = still.amendments
        assert later[0].id in amendment['detail']
        assert amendment['agent_role'] == AGGREGATE_ROLE

    @pytest.mark.asyncio
    async def test_once_the_aggregate_is_resolved_the_next_sweep_files_a_fresh_one(self, queue, ports):
        first = _sweep(ports)
        await _route_all(first, self._mechanical(queue, STARVATION, 1))
        first.close()
        (old,) = _aggregates(queue, status_info_sentinel(STARVATION))
        queue.resolve(old.id, 'digest read', resolved_by='escalation-watcher-auto')

        second = _sweep(ports)
        await _route_all(second, [_seed(queue, task_id='700', agent_role=STARVATION, category='infra_issue')])
        second.close()
        (fresh,) = _aggregates(queue, status_info_sentinel(STARVATION))
        assert fresh.id != old.id

    def test_a_sweep_that_routed_nothing_files_nothing(self, queue, ports):
        _sweep(ports).close()
        assert queue.get_pending() == []


class TestSweepFinishHold:
    FILED_AT = datetime(2026, 10, 20, 9, 0, tzinfo=UTC)
    TIMEOUT_SECS = 900.0

    def _held(self, queue: EscalationQueue) -> tuple[Escalation, RouterHold]:
        record = _hold(queue, _seed(queue), 'tkt_5', self.FILED_AT)
        hold = read_hold(record)
        assert hold is not None
        return record, hold

    async def _finish(self, ports, record, hold, *, after_secs: float):
        sweep = _sweep(ports)
        return await sweep.finish_hold(
            record, hold, now=self.FILED_AT + timedelta(seconds=after_secs),
            timeout_secs=self.TIMEOUT_SECS,
        )

    @pytest.mark.asyncio
    async def test_a_created_ticket_converts_citing_task_and_ticket(self, queue, events, ports, curator):
        record, hold = self._held(queue)
        curator.set_ticket('tkt_5', {'status': 'created', 'task_id': '9001'})
        await self._finish(ports, record, hold, after_secs=60)

        closed = _current(queue, record.id)
        assert (closed.status, closed.resolution_class) == ('dismissed', 'converted')
        assert '9001' in (closed.resolution or '') and 'tkt_5' in (closed.resolution or '')
        (row,) = _dispositions(events)
        assert (row['data']['class'], row['data']['ticket']) == ('converted', 'tkt_5')
        (arguments,) = curator.calls_named('resolve_ticket')
        assert (arguments['ticket'], arguments['project_root']) == ('tkt_5', '/srv/project')
        assert 0 < arguments['timeout_seconds'] <= 5

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('answer', 'reason'), [
        ({'status': 'refused', 'reason': 'cancelled premise'}, 'ticket_refused'),
        ({'status': 'failed', 'reason': 'worker_dead', 'task_id': None}, 'ticket_failed'),
    ])
    async def test_a_ticket_the_curator_will_not_create_promotes(
        self, queue, ports, curator, answer, reason,
    ):
        record, hold = self._held(queue)
        curator.set_ticket('tkt_5', answer)
        await self._finish(ports, record, hold, after_secs=60)
        (l1,) = _router_l1s(queue, '42')
        assert reason in l1.detail
        assert l1.suggested_action == FALLBACK_ACTION
        assert _current(queue, record.id).status == 'dismissed'

    @pytest.mark.asyncio
    async def test_an_undecided_ticket_inside_its_bound_leaves_the_record_alone(
        self, queue, events, ports,
    ):
        record, hold = self._held(queue)
        assert await self._finish(ports, record, hold, after_secs=self.TIMEOUT_SECS - 1) is None
        after = _current(queue, record.id)
        assert (after.status, after.triage_note) == ('pending', record.triage_note)
        assert _router_l1s(queue, '42') == []
        assert _dispositions(events) == []

    @pytest.mark.asyncio
    async def test_an_undecided_ticket_past_its_bound_promotes_ticket_timeout(self, queue, ports):
        record, hold = self._held(queue)
        await self._finish(ports, record, hold, after_secs=self.TIMEOUT_SECS + 1)
        (l1,) = _router_l1s(queue, '42')
        assert 'ticket_timeout' in l1.detail
        assert l1.detail.startswith(record.detail)

    @pytest.mark.asyncio
    async def test_after_one_unreachable_lookup_later_holds_are_not_dispatched(
        self, queue, ports, curator, monkeypatch,
    ):
        first, first_hold = self._held(queue)
        second, second_hold = self._held(queue)

        async def unreachable(name: str, arguments: dict, *, timeout: float = 15) -> dict:
            curator.calls.append((name, dict(arguments)))
            raise ConnectionError('fused-memory unreachable')

        sweep = InfoL0Sweep(
            RouterPorts(queue, None, unreachable, '/srv/project', FILER_RUN_ID),
            by='orphan-reaper', max_conversions=20,
        )
        now = self.FILED_AT + timedelta(seconds=60)
        assert await sweep.finish_hold(first, first_hold, now=now, timeout_secs=self.TIMEOUT_SECS) is None
        assert await sweep.finish_hold(second, second_hold, now=now, timeout_secs=self.TIMEOUT_SECS) is None
        assert len(curator.calls_named('resolve_ticket')) == 1
