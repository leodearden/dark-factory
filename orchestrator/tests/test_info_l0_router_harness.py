"""The info-L0 router's harness call sites: the orphan reaper's info branch and
the startup amnesty (plans/info-l0-disposition-router-prd.md D7/D8/D9/D12)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _curator_fake import FakeCurator
from escalation.classify import DONE_STEP_COMMIT_ORPHAN_CLASS
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.escalation_router import (
    AGGREGATE_CATEGORY,
    AGGREGATE_ROLE,
    FALLBACK_ACTION,
    ROUTER_ID,
    hold_note,
    read_hold,
    status_info_sentinel,
)
from orchestrator.event_store import EventStore, EventType
from orchestrator.harness import Harness

TICKET_TIMEOUT_SECS = 900.0
AGED_SECS = 300.0
STARVATION = 'orchestrator-starvation-watchdog'
FILER_ID = 'run-old/sess-old/pid=111'
LIVE_ID = 'run-new/sess-new/pid=222'
EVAL_WORKTREE = '/home/leo/src/dark-factory/.eval-worktrees/2339/df_task_2339'


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config) -> Harness:
    """A real Harness over a real queue and event store; only the reaper and
    the amnesty are driven, so ``Harness.start()`` is bypassed and the
    scheduler is a mock whose ``dispatch_tool`` is the curator double."""
    mock_orch_config.orphan_l0_reaper_enabled = True
    mock_orch_config.orphan_l0_timeout_secs = 60.0
    mock_orch_config.orphan_l0_check_interval_secs = 1.0
    mock_orch_config.orphan_l0_dispatch_freshness_secs = 120.0
    mock_orch_config.info_l0_router_enabled = True
    mock_orch_config.info_l0_router_ticket_timeout_secs = TICKET_TIMEOUT_SECS
    mock_orch_config.info_l0_router_max_conversions_per_sweep = 20

    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    h.scheduler.is_actively_held = MagicMock(return_value=False)
    h.scheduler.get_task = AsyncMock(return_value={'status': 'in-progress', 'metadata': {}})
    h.scheduler.dispatch_tool = FakeCurator()
    h.event_store = EventStore(tmp_path / 'runs.db', run_id='run-test')
    h._escalation_queue = EscalationQueue(tmp_path / 'escalations')
    return h


def _queue(harness: Harness) -> EscalationQueue:
    queue = harness._escalation_queue
    assert queue is not None
    return queue


def _curator(harness: Harness) -> FakeCurator:
    curator = harness.scheduler.dispatch_tool
    assert isinstance(curator, FakeCurator)
    return curator


def _submit_aged(
    harness: Harness, task_id: str, *, seconds_ago: float = AGED_SECS,
    severity: str = 'info', level: int = 0, agent_role: str = 'implementer',
    category: str = 'design_concern', summary: str = 'helper X is duplicated',
    detail: str = 'Author text: bar.py carries a copy of foo.helper.',
    suggested_action: str = 'create_followup_task', worktree: str | None = None,
    filing_claimant_run_id: str | None = None,
) -> Escalation:
    queue = _queue(harness)
    esc = Escalation(
        id=queue.make_id(task_id), task_id=task_id, agent_role=agent_role,
        severity=severity, category=category, summary=summary, detail=detail,
        suggested_action=suggested_action, level=level, worktree=worktree,
        timestamp=(datetime.now(UTC) - timedelta(seconds=seconds_ago)).isoformat(),
        filing_claimant_run_id=filing_claimant_run_id,
    )
    queue.submit(esc)
    return esc


def _hold(harness: Harness, esc: Escalation, ticket: str, *, filed_secs_ago: float) -> Escalation:
    filed_at = datetime.now(UTC) - timedelta(seconds=filed_secs_ago)
    _queue(harness).stamp_triage(esc.id, triaged_by=ROUTER_ID, triage_note=hold_note(ticket, filed_at))
    return _current(harness, esc.id)


def _current(harness: Harness, esc_id: str) -> Escalation:
    esc = _queue(harness).get(esc_id)
    assert esc is not None
    return esc


def _make_live(harness: Harness, *, claimant_run_id: str | None) -> None:
    harness.scheduler.is_actively_held = MagicMock(return_value=True)
    harness.scheduler.get_task = AsyncMock(return_value={
        'status': 'in-progress', 'claimant_run_id': claimant_run_id, 'metadata': {},
    })


def _l1s(harness: Harness, *, agent_role: str) -> list[Escalation]:
    return [e for e in _queue(harness).get_pending() if e.level == 1 and e.agent_role == agent_role]


def _aggregates(harness: Harness, sentinel: str) -> list[Escalation]:
    return [
        e for e in _l1s(harness, agent_role=AGGREGATE_ROLE)
        if e.task_id == sentinel and e.category == AGGREGATE_CATEGORY
    ]


def _dispositions(harness: Harness) -> list[dict]:
    store = harness.event_store
    assert store is not None
    return [row['data'] for row in store.fetch_events_by_type(EventType.info_l0_dispositioned)]


@pytest.mark.asyncio
class TestReaperFinishesHolds:
    async def test_a_created_ticket_converts_the_held_record(self, harness: Harness):
        esc = _hold(harness, _submit_aged(harness, '42'), 'tkt_9', filed_secs_ago=120.0)
        _curator(harness).set_ticket('tkt_9', {'status': 'created', 'task_id': '9001'})

        await harness._reap_orphan_l0_escalations()

        closed = _current(harness, esc.id)
        assert (closed.status, closed.resolution_class) == ('dismissed', 'converted')
        assert 'converted:9001' in (closed.resolution or '')
        assert _l1s(harness, agent_role='harness-orphan-reaper') == []

    async def test_a_hold_is_finished_while_a_newer_workflow_is_live(self, harness: Harness):
        esc = _submit_aged(harness, '42', filing_claimant_run_id=FILER_ID)
        esc = _hold(harness, esc, 'tkt_9', filed_secs_ago=120.0)
        _curator(harness).set_ticket('tkt_9', {'status': 'created', 'task_id': '9001'})
        _make_live(harness, claimant_run_id=LIVE_ID)
        harness._escalation_events['42'] = MagicMock()

        await harness._reap_orphan_l0_escalations()

        assert _current(harness, esc.id).resolution_class == 'converted'

    @pytest.mark.parametrize(('answer', 'reason', 'filed_secs_ago'), [
        ({'status': 'refused', 'reason': 'cancelled premise'}, 'ticket_refused', 120.0),
        ({'status': 'failed', 'reason': 'timeout', 'task_id': None}, 'ticket_timeout',
         TICKET_TIMEOUT_SECS + 60.0),
    ], ids=['refused', 'timeout'])
    async def test_a_ticket_that_will_not_convert_promotes_with_the_hint(
        self, harness: Harness, answer, reason, filed_secs_ago,
    ):
        esc = _hold(harness, _submit_aged(harness, '42'), 'tkt_9', filed_secs_ago=filed_secs_ago)
        _curator(harness).set_ticket('tkt_9', answer)

        await harness._reap_orphan_l0_escalations()

        (l1,) = _l1s(harness, agent_role=AGGREGATE_ROLE)
        assert l1.suggested_action == FALLBACK_ACTION
        assert l1.detail.startswith(esc.detail)
        assert reason in l1.detail
        assert _l1s(harness, agent_role='harness-orphan-reaper') == []
        assert _current(harness, esc.id).status == 'dismissed'

    async def test_an_undecided_hold_inside_its_bound_stays_untouched(self, harness: Harness):
        esc = _hold(harness, _submit_aged(harness, '42'), 'tkt_9', filed_secs_ago=120.0)

        await harness._reap_orphan_l0_escalations()

        after = _current(harness, esc.id)
        assert (after.status, after.triage_note) == ('pending', esc.triage_note)
        assert [e for e in _queue(harness).get_pending() if e.level == 1] == []


def _seed_storm(harness: Harness, first_task: int, n: int) -> list[Escalation]:
    """*n* aged starvation-watchdog notes; durable writes are not under test here."""
    with patch('os.fsync'):
        return [
            _submit_aged(harness, str(first_task + i), agent_role=STARVATION, category='infra_issue')
            for i in range(n)
        ]


@pytest.mark.asyncio
class TestReaperRoutesInfoOrphans:
    async def test_a_mechanical_storm_closes_status_info_under_one_aggregate(self, harness: Harness):
        harness.event_store = None  # 489 durable event commits cost seconds and are not under test
        storm = _seed_storm(harness, 10_000, 489)

        await harness._reap_orphan_l0_escalations()

        assert all(_current(harness, e.id).resolution_class == 'status-info' for e in storm)
        (aggregate,) = _aggregates(harness, status_info_sentinel(STARVATION))
        assert all(e.id in aggregate.detail for e in storm)
        assert _l1s(harness, agent_role='harness-orphan-reaper') == []
        assert _curator(harness).calls == []

        later = _seed_storm(harness, 20_000, 3)
        await harness._reap_orphan_l0_escalations()

        (still,) = _aggregates(harness, status_info_sentinel(STARVATION))
        assert still.id == aggregate.id
        assert all(e.id in still.amendments[-1]['detail'] for e in later)

    async def test_an_unregistered_orchestrator_role_goes_to_the_curator(self, harness: Harness):
        esc = _submit_aged(harness, '42', agent_role='orchestrator-offline-lane', category='infra_issue')

        await harness._reap_orphan_l0_escalations()

        held = _current(harness, esc.id)
        assert held.status == 'pending'
        hold = read_hold(held)
        assert hold is not None and hold.ticket == 'tkt_1'
        (row,) = _dispositions(harness)
        assert (row['class'], row['by'], row['exit_kind']) == ('convert-pending', 'orphan-reaper', 'orphan')

    async def test_an_open_l1_of_another_category_does_not_swallow_an_info_note(
        self, harness: Harness,
    ):
        _submit_aged(harness, '42', severity='blocking', level=1, category='task_failure', seconds_ago=5.0)
        esc = _submit_aged(harness, '42', category='infra_issue')

        await harness._reap_orphan_l0_escalations()

        held = _current(harness, esc.id)
        assert held.status == 'pending' and read_hold(held) is not None
        assert len(_curator(harness).calls_named('submit_task')) == 1

    async def test_the_done_step_commit_class_closes_status_info(self, harness: Harness):
        esc = _submit_aged(
            harness, '42', agent_role='orchestrator', category='infra_issue',
            suggested_action='verify_wip_reconciliation',
        )

        await harness._reap_orphan_l0_escalations()

        assert _current(harness, esc.id).resolution_class == 'status-info'
        (aggregate,) = _aggregates(harness, status_info_sentinel(DONE_STEP_COMMIT_ORPHAN_CLASS))
        assert esc.id in aggregate.detail

    async def test_an_info_note_under_a_live_workflow_is_still_deferred(self, harness: Harness):
        esc = _submit_aged(harness, '42', filing_claimant_run_id=FILER_ID)
        _make_live(harness, claimant_run_id=LIVE_ID)

        await harness._reap_orphan_l0_escalations()

        after = _current(harness, esc.id)
        assert (after.status, after.triaged_by) == ('pending', None)
        assert _curator(harness).calls == []
        assert _dispositions(harness) == []

    async def test_an_eval_lane_info_note_is_still_dismissed_by_the_eval_rule(self, harness: Harness):
        esc = _submit_aged(harness, '2339', worktree=EVAL_WORKTREE)

        await harness._reap_orphan_l0_escalations()

        after = _current(harness, esc.id)
        assert (after.status, after.resolved_by) == ('dismissed', 'harness-orphan-reaper')
        assert _curator(harness).calls == []

    async def test_with_the_router_disabled_an_info_orphan_is_promoted_verbatim(
        self, harness: Harness,
    ):
        harness.config.info_l0_router_enabled = False
        esc = _submit_aged(harness, '42')

        assert await harness._reap_orphan_l0_escalations() == 1

        (l1,) = _l1s(harness, agent_role='harness-orphan-reaper')
        assert l1.suggested_action == 'manual_intervention'
        assert esc.summary in l1.summary
        assert _curator(harness).calls == []


def _write_legacy(harness: Harness, task_id: str, *, level: int, category: str) -> Escalation:
    """A record filed before its category was retired: on disk, never through submit."""
    queue = _queue(harness)
    esc = Escalation(
        id=queue.make_id(task_id), task_id=task_id, agent_role='implementer',
        severity='blocking', category=category, summary=f'legacy L{level}', detail='legacy detail',
        level=level, timestamp=(datetime.now(UTC) - timedelta(seconds=AGED_SECS)).isoformat(),
    )
    (queue.queue_dir / f'{esc.id}.json').write_text(esc.to_json())
    return esc


@pytest.mark.asyncio
class TestBlockingDedupIsCategoryScoped:
    """The unblock-4917 amendment: an open L1 covers a blocking orphan only
    when it is about the same category, and it is told what it absorbed."""

    async def test_an_l1_of_another_category_does_not_cover_a_blocking_orphan(self, harness: Harness):
        _submit_aged(harness, '42', severity='blocking', level=1, category='task_failure', seconds_ago=5.0)
        orphan = _submit_aged(harness, '42', severity='blocking', category='infra_issue')

        assert await harness._reap_orphan_l0_escalations() == 1

        (l1,) = _l1s(harness, agent_role='harness-orphan-reaper')
        assert (l1.suggested_action, l1.category) == ('manual_intervention', 'infra_issue')
        assert orphan.summary in l1.summary
        assert 'Auto-promoted to level 1' in (_current(harness, orphan.id).resolution or '')

    async def test_a_same_category_l1_covers_the_orphan_and_records_it(self, harness: Harness):
        covering = _submit_aged(
            harness, '42', severity='blocking', level=1, category='infra_issue', seconds_ago=5.0,
        )
        orphan = _submit_aged(
            harness, '42', severity='blocking', category='infra_issue', summary='disk quota hit',
        )

        assert await harness._reap_orphan_l0_escalations() == 0

        dismissed = _current(harness, orphan.id)
        assert (dismissed.status, dismissed.resolved_by) == ('dismissed', 'harness-orphan-reaper')
        assert 'open L1 already covers' in (dismissed.resolution or '')
        assert _l1s(harness, agent_role='harness-orphan-reaper') == []
        (amendment,) = _current(harness, covering.id).amendments
        assert amendment['agent_role'] == 'harness-orphan-reaper'
        carried = amendment['summary'] + amendment['detail']
        assert orphan.id in carried and 'disk quota hit' in carried

    async def test_a_legacy_category_does_not_abort_the_sweep(self, harness: Harness):
        _write_legacy(harness, '7', level=1, category='retired_category')
        legacy = _write_legacy(harness, '7', level=0, category='retired_category')
        later = _submit_aged(harness, '8', severity='blocking')

        await harness._reap_orphan_l0_escalations()

        assert _current(harness, legacy.id).status == 'dismissed'
        assert _current(harness, later.id).status == 'dismissed'
        assert [l1.task_id for l1 in _l1s(harness, agent_role='harness-orphan-reaper')] == ['8']


def _strand_events(harness: Harness) -> list[dict]:
    store = harness.event_store
    assert store is not None
    return store.fetch_events_by_type(EventType.stale_l0_strand_dismissed)


@pytest.mark.asyncio
class TestStartupAmnestyRoutesInfoNotes:
    """Boundary #16: an info L0 pending across a restart is routed, never
    dismissed unread; blocking strands are swept exactly as before."""

    async def test_a_work_note_is_routed_and_survives_the_sweep_held(self, harness: Harness):
        esc = _submit_aged(harness, '42', seconds_ago=30.0)

        await harness._dismiss_stale_escalations()

        held = _current(harness, esc.id)
        assert (held.status, held.resolved_by) == ('pending', None)
        hold = read_hold(held)
        assert hold is not None and hold.ticket == 'tkt_1'
        (row,) = _dispositions(harness)
        assert (row['class'], row['by'], row['exit_kind']) == ('convert-pending', 'startup-amnesty', 'restart')
        assert _strand_events(harness) == []

    async def test_a_mechanical_note_closes_status_info_under_one_aggregate(self, harness: Harness):
        esc = _submit_aged(harness, '42', agent_role=STARVATION, category='infra_issue')

        await harness._dismiss_stale_escalations()

        assert _current(harness, esc.id).resolution_class == 'status-info'
        (aggregate,) = _aggregates(harness, status_info_sentinel(STARVATION))
        assert esc.id in aggregate.detail
        assert _strand_events(harness) == []

    async def test_a_hold_from_before_the_restart_converts_when_its_ticket_was_created(
        self, harness: Harness,
    ):
        esc = _hold(harness, _submit_aged(harness, '42'), 'tkt_9', filed_secs_ago=120.0)
        _curator(harness).set_ticket('tkt_9', {'status': 'created', 'task_id': '9001'})

        await harness._dismiss_stale_escalations()

        assert _current(harness, esc.id).resolution_class == 'converted'
        (row,) = _dispositions(harness)
        assert (row['class'], row['exit_kind']) == ('converted', 'restart')

    async def test_an_undecided_hold_from_before_the_restart_stays_pending(self, harness: Harness):
        esc = _hold(harness, _submit_aged(harness, '42'), 'tkt_9', filed_secs_ago=120.0)

        await harness._dismiss_stale_escalations()

        after = _current(harness, esc.id)
        assert (after.status, after.triage_note) == ('pending', esc.triage_note)
        assert _curator(harness).calls_named('submit_task') == []

    async def test_a_blocking_strand_is_still_swept_with_its_event_and_aggregate(
        self, harness: Harness,
    ):
        strand = _submit_aged(harness, '5189', severity='blocking', category='task_failure')
        _submit_aged(harness, '42', seconds_ago=30.0)

        await harness._dismiss_stale_escalations()

        swept = _current(harness, strand.id)
        assert (swept.status, swept.resolved_by, swept.resolution_class) == (
            'dismissed', 'auto-dismissed', 'stale-strand',
        )
        (event,) = _strand_events(harness)
        assert event['data']['escalation_id'] == strand.id
        (aggregate,) = _l1s(harness, agent_role='harness-stale-l0-strand-sweep')
        assert strand.id in aggregate.detail

    async def test_an_eval_lane_info_note_is_dismissed_as_before(self, harness: Harness):
        esc = _submit_aged(harness, '2339', worktree=EVAL_WORKTREE, seconds_ago=30.0)

        await harness._dismiss_stale_escalations()

        assert (_current(harness, esc.id).status, _current(harness, esc.id).resolved_by) == (
            'dismissed', 'auto-dismissed',
        )
        assert _curator(harness).calls == []

    async def test_with_the_router_disabled_every_l0_is_dismissed_as_before(self, harness: Harness):
        harness.config.info_l0_router_enabled = False
        info = _submit_aged(harness, '42', seconds_ago=30.0)
        blocking = _submit_aged(harness, '43', severity='blocking', seconds_ago=30.0)

        await harness._dismiss_stale_escalations()

        for esc in (info, blocking):
            assert (_current(harness, esc.id).status, _current(harness, esc.id).resolved_by) == (
                'dismissed', 'auto-dismissed',
            )
        assert _curator(harness).calls == []
        assert _dispositions(harness) == []
