"""Unit tests for orchestrator.merge_lane.request_ledger (task MQ-invariants eta / 1992).

Covered here:
  TestRequestLedgerLifecycle    — arm / sweep / requeue lifecycle, the per-episode
                                  alarm count (mark_alarmed, task 5827) and discard
  TestMergeRequestStuckSentinel — the per-request dedup sentinel
  TestAlarmMergeRequestStuck    — the single L1 filing function
  TestReportStuckRequests       — the per-sweep reporter: one bounded alarm ladder
                                  per dequeue episode drives both the WARNING and
                                  the L1 (task 5827)

This module imports only orchestrator.merge_lane.types / orchestrator.config at
module scope (NOT orchestrator.merge_lane.worker): RequestLedger has no
git/GitOps dependency at all. Ledger names are imported LOCALLY inside each
test, so a not-yet-implemented name never breaks collection of the rest of the
file.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.merge_lane.types import MergeOutcome, MergeRequest, QueuedBranch

# ---------------------------------------------------------------------------
# Helpers (per-file duplication convention — see test_merge_queue_resolve_release.py
# / test_merge_queue_concurrent_verify.py)
# ---------------------------------------------------------------------------


def _make_request(
    task_id: str = 't1',
    branch: str = 'task/t1',
    *,
    request_id: str | None = None,
) -> MergeRequest:
    """Build a bare MergeRequest with a fresh Future for the running event loop.

    worktree/config/module_configs are irrelevant to RequestLedger, which only
    reads task_id/branch/request_id/result — dummy values keep this test
    module free of any git/GitOps dependency (pure data-structure unit tests).
    """
    kwargs: dict = {}
    if request_id is not None:
        kwargs['request_id'] = request_id
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(branch, 'task/'),
        worktree=Path('/tmp/unused'),
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=OrchestratorConfig(),
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
        **kwargs,
    )


# ---------------------------------------------------------------------------
# step-1 RED / step-2 GREEN: RequestLedger lifecycle + exactly-once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRequestLedgerLifecycle:
    """RequestLedger arm/sweep/requeue lifecycle (task 1992 step-1).

    RED until step-2 GREEN adds orchestrator.merge_lane.request_ledger.
    """

    async def test_on_dequeue_arms_the_ledger(self) -> None:
        from orchestrator.merge_lane.request_ledger import RequestLedger

        ledger = RequestLedger()
        req = _make_request('t1', 'task/t1')
        t0 = 1_000_000.0

        ledger.on_dequeue(req, now=t0)

        assert len(ledger) == 1
        assert req.request_id in ledger.open_request_ids()

    async def test_stuck_entries_respects_threshold(self) -> None:
        from orchestrator.merge_lane.request_ledger import RequestLedger

        ledger = RequestLedger()
        req = _make_request('t2', 'task/t2')
        t0 = 1_000_000.0
        ledger.on_dequeue(req, now=t0)

        assert ledger.stuck_entries(now=t0 + 100, threshold_s=1000) == []

        stuck = ledger.stuck_entries(now=t0 + 2000, threshold_s=1000)
        assert len(stuck) == 1
        entry = stuck[0]
        assert entry.request_id == req.request_id
        assert entry.task_id == 't2'
        assert entry.branch == 't2'  # ledger records QueuedBranch.bare_id
        assert entry.age_secs == pytest.approx(2000)

    async def test_passive_resolution_via_set_result_sweeps_entry(self) -> None:
        from orchestrator.merge_lane.request_ledger import RequestLedger

        ledger = RequestLedger()
        req = _make_request('t3', 'task/t3')
        t0 = 1_000_000.0
        ledger.on_dequeue(req, now=t0)

        req.result.set_result(MergeOutcome('done'))

        stuck = ledger.stuck_entries(now=t0 + 2000, threshold_s=1000)
        assert stuck == []
        assert len(ledger) == 0

    async def test_passive_resolution_via_cancel_sweeps_entry(self) -> None:
        """Abandoned (cancelled) Futures count as resolved too."""
        from orchestrator.merge_lane.request_ledger import RequestLedger

        ledger = RequestLedger()
        req = _make_request('t4', 'task/t4')
        t0 = 1_000_000.0
        ledger.on_dequeue(req, now=t0)

        req.result.cancel()  # abandoned

        stuck = ledger.stuck_entries(now=t0 + 2000, threshold_s=1000)
        assert stuck == []
        assert len(ledger) == 0

    async def test_on_requeued_removes_then_redequeue_restarts_age(self) -> None:
        from orchestrator.merge_lane.request_ledger import RequestLedger

        ledger = RequestLedger()
        req = _make_request('t5', 'task/t5')
        t0 = 1_000_000.0
        ledger.on_dequeue(req, now=t0)

        ledger.on_requeued(req.request_id)
        assert len(ledger) == 0
        assert not ledger.stuck_entries(now=t0 + 5000, threshold_s=1000)

        t2 = t0 + 10_000.0
        ledger.on_dequeue(req, now=t2)

        # Age measured from t2, not t0: not yet stuck just after re-dequeue.
        assert ledger.stuck_entries(now=t2 + 10, threshold_s=1000) == []
        stuck = ledger.stuck_entries(now=t2 + 2000, threshold_s=1000)
        assert len(stuck) == 1
        assert stuck[0].age_secs == pytest.approx(2000)

    async def test_double_on_dequeue_keeps_earliest_and_does_not_duplicate(self) -> None:
        from orchestrator.merge_lane.request_ledger import RequestLedger

        ledger = RequestLedger()
        req = _make_request('t6', 'task/t6')
        t0 = 1_000_000.0
        ledger.on_dequeue(req, now=t0)
        ledger.on_dequeue(req, now=t0 + 500)  # second dequeue — must not reset the clock

        assert len(ledger) == 1
        stuck = ledger.stuck_entries(now=t0 + 1500, threshold_s=1000)
        assert len(stuck) == 1
        assert stuck[0].age_secs == pytest.approx(1500)  # measured from t0, not t0+500

    async def test_double_on_requeued_is_idempotent(self) -> None:
        from orchestrator.merge_lane.request_ledger import RequestLedger

        ledger = RequestLedger()
        req = _make_request('t7', 'task/t7')
        ledger.on_dequeue(req, now=1_000_000.0)

        ledger.on_requeued(req.request_id)
        ledger.on_requeued(req.request_id)  # must not raise (no KeyError)

        assert ledger.is_empty()

    async def test_is_empty_reflects_state(self) -> None:
        from orchestrator.merge_lane.request_ledger import RequestLedger

        ledger = RequestLedger()
        assert ledger.is_empty()

        req = _make_request('t8', 'task/t8')
        ledger.on_dequeue(req, now=1_000_000.0)
        assert not ledger.is_empty()

        ledger.on_requeued(req.request_id)
        assert ledger.is_empty()

    # -----------------------------------------------------------------
    # Per-episode alarm count (task 5827) and discard (task 1992 review):
    #   * mark_alarmed counts the stuck alarms raised for an entry's CURRENT
    #     dequeue episode, so report_stuck_requests can walk a bounded
    #     ladder instead of re-alarming on every heartbeat poll;
    #   * discard is the eviction path for a truly leaked (never-resolved,
    #     never-requeued) entry.
    # -----------------------------------------------------------------

    async def test_mark_alarmed_counts_up_in_stuck_entries(self) -> None:
        from orchestrator.merge_lane.request_ledger import RequestLedger

        ledger = RequestLedger()
        req = _make_request('t9', 'task/t9')
        t0 = 1_000_000.0
        ledger.on_dequeue(req, now=t0)

        stuck = ledger.stuck_entries(now=t0 + 2000, threshold_s=1000)
        assert len(stuck) == 1
        assert stuck[0].alarms_raised == 0

        ledger.mark_alarmed(req.request_id)
        stuck = ledger.stuck_entries(now=t0 + 2100, threshold_s=1000)
        assert [s.alarms_raised for s in stuck] == [1]

        ledger.mark_alarmed(req.request_id)
        stuck = ledger.stuck_entries(now=t0 + 2200, threshold_s=1000)
        assert [s.alarms_raised for s in stuck] == [2]

    async def test_new_dequeue_generation_resets_alarms_raised(self) -> None:
        from orchestrator.merge_lane.request_ledger import RequestLedger

        ledger = RequestLedger()
        req = _make_request('t10', 'task/t10')
        t0 = 1_000_000.0
        ledger.on_dequeue(req, now=t0)
        ledger.mark_alarmed(req.request_id)

        ledger.on_requeued(req.request_id)
        t2 = t0 + 10_000.0
        ledger.on_dequeue(req, now=t2)

        stuck = ledger.stuck_entries(now=t2 + 2000, threshold_s=1000)
        assert [s.alarms_raised for s in stuck] == [0], (
            'on_requeued + a later on_dequeue is a new episode and must start uncounted'
        )

    async def test_idempotent_redequeue_keeps_alarms_raised(self) -> None:
        from orchestrator.merge_lane.request_ledger import RequestLedger

        ledger = RequestLedger()
        req = _make_request('t10b', 'task/t10b')
        t0 = 1_000_000.0
        ledger.on_dequeue(req, now=t0)
        ledger.mark_alarmed(req.request_id)

        ledger.on_dequeue(req, now=t0 + 500)

        stuck = ledger.stuck_entries(now=t0 + 2000, threshold_s=1000)
        assert [s.alarms_raised for s in stuck] == [1], (
            'a second on_dequeue of an armed request is the same episode and keeps its count'
        )

    async def test_mark_alarmed_is_a_noop_for_absent_request_id(self) -> None:
        from orchestrator.merge_lane.request_ledger import RequestLedger

        ledger = RequestLedger()
        ledger.mark_alarmed('no-such-request')

        assert ledger.is_empty()

    async def test_discard_removes_entry_regardless_of_unresolved_future(self) -> None:
        from orchestrator.merge_lane.request_ledger import RequestLedger

        ledger = RequestLedger()
        req = _make_request('t11', 'task/t11')
        t0 = 1_000_000.0
        ledger.on_dequeue(req, now=t0)
        assert not req.result.done()

        ledger.discard(req.request_id)

        assert ledger.is_empty()
        assert not req.result.done()  # discard never touches the Future

    async def test_discard_is_idempotent_for_absent_request_id(self) -> None:
        from orchestrator.merge_lane.request_ledger import RequestLedger

        ledger = RequestLedger()
        ledger.discard('no-such-request')  # must not raise
        ledger.discard('no-such-request')  # still must not raise


# ---------------------------------------------------------------------------
# step-3 RED / step-4 GREEN: _alarm_merge_request_stuck + _merge_request_stuck_sentinel
# ---------------------------------------------------------------------------


class _FakeEscalationQueue:
    """Minimal fake escalation queue (copied from
    test_merge_queue_multihost_wiring.py:1200 — per-file duplication
    convention).
    """

    def __init__(self, *, open_l1: bool = False):
        self._open_l1 = open_l1
        self._seq = 0
        self.submitted: list = []

    def has_open_l1(self, task_id: str) -> bool:  # noqa: ARG002
        return self._open_l1

    def make_id(self, task_id: str) -> str:
        self._seq += 1
        return f'esc-{self._seq}'

    def submit(self, esc) -> None:
        self.submitted.append(esc)

    def open_it(self):
        """Simulate a prior open L1 (for dedup tests)."""
        self._open_l1 = True

    def close_it(self):
        """Simulate a human closing the open L1 (the inverse of open_it)."""
        self._open_l1 = False


class _FakeEventStore:
    """Minimal fake event store (copied from test_merge_queue_multihost_wiring.py)."""

    def __init__(self):
        self.emitted: list = []

    def emit(self, event_type, *, task_id=None, phase=None, data=None, **kw):
        self.emitted.append({'event_type': event_type, 'task_id': task_id, 'data': data or {}})


def _make_stuck(
    request_id: str = 'mr-abc12345',
    task_id: str = 'stuck-task',
    branch: str = 'task/stuck-task',
    age_secs: float = 16201.0,
    phase: str = 'unowned',
):
    from orchestrator.merge_lane.request_ledger import StuckRequest

    return StuckRequest(
        request_id=request_id, task_id=task_id, branch=branch,
        age_secs=age_secs, phase=phase,
    )


class TestMergeRequestStuckSentinel:
    """_merge_request_stuck_sentinel (task 1992 step-3).

    RED until step-4 GREEN adds the function to merge_request_ledger.py.
    """

    def test_sentinel_is_per_request_id(self):
        from orchestrator.merge_lane.request_ledger import _merge_request_stuck_sentinel

        assert _merge_request_stuck_sentinel('mr-abc12345') == '__merge_request_stuck__mr-abc12345'


class TestAlarmMergeRequestStuck:
    """_alarm_merge_request_stuck module-level helper (task 1992 step-3).

    RED until step-4 GREEN adds the function to merge_request_ledger.py.
    """

    def _call(self, eq, stuck, *, event_store=None):
        from orchestrator.merge_lane.request_ledger import _alarm_merge_request_stuck
        _alarm_merge_request_stuck(eq, stuck, event_store=event_store)

    def test_none_queue_is_noop(self):
        """None escalation_queue -> returns silently, no raise."""
        self._call(None, _make_stuck())
        # No assertion needed — must not raise

    def test_first_call_submits_exactly_one_escalation(self):
        eq = _FakeEscalationQueue(open_l1=False)
        self._call(eq, _make_stuck())
        assert len(eq.submitted) == 1

    def test_escalation_has_level_1_and_blocking_severity(self):
        eq = _FakeEscalationQueue(open_l1=False)
        self._call(eq, _make_stuck())
        esc = eq.submitted[0]
        assert esc.level == 1
        assert esc.severity == 'blocking'

    def test_escalation_has_merge_request_stuck_category(self):
        eq = _FakeEscalationQueue(open_l1=False)
        self._call(eq, _make_stuck())
        esc = eq.submitted[0]
        assert esc.category == 'merge_request_stuck'

    def test_escalation_agent_role_starts_with_orchestrator(self):
        eq = _FakeEscalationQueue(open_l1=False)
        self._call(eq, _make_stuck())
        esc = eq.submitted[0]
        assert esc.agent_role.startswith('orchestrator-')

    def test_escalation_task_id_is_the_sentinel(self):
        eq = _FakeEscalationQueue(open_l1=False)
        stuck = _make_stuck(request_id='mr-deadbeef')
        self._call(eq, stuck)
        esc = eq.submitted[0]
        assert esc.task_id == '__merge_request_stuck__mr-deadbeef'

    def test_summary_and_detail_name_request_id_and_branch(self):
        eq = _FakeEscalationQueue(open_l1=False)
        stuck = _make_stuck(request_id='mr-namecheck', branch='task/namecheck-branch')
        self._call(eq, stuck)
        esc = eq.submitted[0]
        assert 'mr-namecheck' in esc.summary
        assert 'task/namecheck-branch' in esc.summary
        assert 'mr-namecheck' in esc.detail
        assert 'task/namecheck-branch' in esc.detail

    def test_detail_names_the_integer_age(self):
        eq = _FakeEscalationQueue(open_l1=False)
        stuck = _make_stuck(age_secs=16201.7)
        self._call(eq, stuck)
        esc = eq.submitted[0]
        assert '16201' in esc.detail

    def test_second_call_with_open_l1_is_deduped(self):
        eq = _FakeEscalationQueue(open_l1=True)  # alarm already open
        self._call(eq, _make_stuck())
        assert len(eq.submitted) == 0

    def test_event_store_emits_escalation_created_event(self):
        from orchestrator.event_store import EventType

        eq = _FakeEscalationQueue(open_l1=False)
        es = _FakeEventStore()
        stuck = _make_stuck(request_id='mr-evented')
        self._call(eq, stuck, event_store=es)

        assert len(es.emitted) >= 1
        types = [e['event_type'] for e in es.emitted]
        assert EventType.escalation_created in types
        events = [e for e in es.emitted if e['event_type'] == EventType.escalation_created]
        assert any('mr-evented' in str(e['data']) for e in events)



# ---------------------------------------------------------------------------
# task 5827: report_stuck_requests — one bounded alarm ladder per dequeue episode
# ---------------------------------------------------------------------------

_THRESHOLD_S = 1000.0
_T0 = 1_000_000.0


def _stuck_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records
        if r.levelno == logging.WARNING and r.name == 'orchestrator.merge_lane'
    ]


def _escalation_created_data(event_store: _FakeEventStore) -> list[dict]:
    from orchestrator.event_store import EventType

    return [
        e['data'] for e in event_store.emitted
        if e['event_type'] == EventType.escalation_created
    ]


def _no_pipeline_phases() -> dict[str, str]:
    return {}


@pytest.mark.asyncio
class TestReportStuckRequests:
    """report_stuck_requests drives the WARNING and the L1 off one per-episode ladder.

    The incident (reify mr-97cb9456, 2026-09-23): once a human closed the
    merge_request_stuck L1, every heartbeat re-filed it, because the only
    escalation dedup was ``has_open_l1``. Sweep ages stay below 3x the
    threshold, the ladder's re-alarm tier, except in tests about that tier.
    """

    def _sweep(self, ledger, at, *, escalation_queue, event_store=None,
               pipeline_phases=_no_pipeline_phases) -> None:
        from orchestrator.merge_lane.request_ledger import report_stuck_requests

        report_stuck_requests(
            ledger, now=at, threshold_s=_THRESHOLD_S, pipeline_phases=pipeline_phases,
            escalation_queue=escalation_queue, event_store=event_store,
        )

    def _armed(self, task_id: str, *, now: float = _T0):
        from orchestrator.merge_lane.request_ledger import RequestLedger

        ledger = RequestLedger()
        req = _make_request(task_id, f'task/{task_id}')
        ledger.on_dequeue(req, now=now)
        return ledger, req

    async def test_first_sweep_past_threshold_warns_and_alarms_once(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        ledger, req = self._armed('rs1')
        eq = _FakeEscalationQueue(open_l1=False)
        es = _FakeEventStore()

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
            self._sweep(ledger, _T0 + 2000, escalation_queue=eq, event_store=es)

        warnings = _stuck_warnings(caplog)
        assert len(warnings) == 1, caplog.text
        message = warnings[0].getMessage()
        assert req.request_id in message
        assert req.branch.bare_id in message
        assert '2000' in message
        assert [e.category for e in eq.submitted] == ['merge_request_stuck']
        assert [d['alarm_number'] for d in _escalation_created_data(es)] == [1]

    async def test_closed_alarm_is_not_refiled_while_the_same_dequeue_stays_stuck(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        ledger, _req = self._armed('rs2')
        eq = _FakeEscalationQueue(open_l1=False)  # every prior alarm already closed

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
            for age in (1500, 2000, 2500, 2900):
                self._sweep(ledger, _T0 + age, escalation_queue=eq)

        assert len(eq.submitted) == 1, [e.summary for e in eq.submitted]
        assert len(_stuck_warnings(caplog)) == 1, caplog.text

    async def test_a_new_dequeue_generation_alarms_again(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        ledger, req = self._armed('rs3')
        eq = _FakeEscalationQueue(open_l1=False)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
            self._sweep(ledger, _T0 + 1500, escalation_queue=eq)
            ledger.on_requeued(req.request_id)
            t2 = _T0 + 10_000
            ledger.on_dequeue(req, now=t2)

            self._sweep(ledger, t2 + 500, escalation_queue=eq)
            assert len(eq.submitted) == 1

            self._sweep(ledger, t2 + 1500, escalation_queue=eq)

        assert len(eq.submitted) == 2
        assert len(_stuck_warnings(caplog)) == 2, caplog.text

    async def test_an_open_l1_covers_the_episode_alarm_after_it_closes(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        ledger, _req = self._armed('rs4')
        eq = _FakeEscalationQueue(open_l1=False)
        eq.open_it()

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
            self._sweep(ledger, _T0 + 1500, escalation_queue=eq)
            assert len(eq.submitted) == 0
            assert len(_stuck_warnings(caplog)) == 1, caplog.text

            eq.close_it()
            self._sweep(ledger, _T0 + 2500, escalation_queue=eq)

        assert len(eq.submitted) == 0
        assert len(_stuck_warnings(caplog)) == 1, caplog.text

    async def test_no_escalation_queue_still_warns_once_per_episode(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        ledger, _req = self._armed('rs5')

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
            self._sweep(ledger, _T0 + 1500, escalation_queue=None)
            self._sweep(ledger, _T0 + 2500, escalation_queue=None)

        assert len(_stuck_warnings(caplog)) == 1, caplog.text

    async def test_phase_comes_from_the_pipeline_lookup(self) -> None:
        verifying_ledger, verifying_req = self._armed('rs6a')
        verifying_events = _FakeEventStore()
        self._sweep(
            verifying_ledger, _T0 + 1500, escalation_queue=_FakeEscalationQueue(),
            event_store=verifying_events,
            pipeline_phases=lambda: {verifying_req.request_id: 'verifying'},
        )

        leaked_ledger, _leaked_req = self._armed('rs6b')
        leaked_events = _FakeEventStore()
        self._sweep(
            leaked_ledger, _T0 + 1500, escalation_queue=_FakeEscalationQueue(),
            event_store=leaked_events,
            pipeline_phases=lambda: {'mr-someone-else': 'verifying'},
        )

        assert [d['phase'] for d in _escalation_created_data(verifying_events)] == ['verifying']
        assert [d['phase'] for d in _escalation_created_data(leaked_events)] == ['unowned']

    async def test_pipeline_lookup_is_consulted_only_when_an_alarm_is_due(self) -> None:
        ledger, req = self._armed('rs7')
        eq = _FakeEscalationQueue(open_l1=False)
        lookups: list[int] = []

        def recording_phases() -> dict[str, str]:
            lookups.append(1)
            return {req.request_id: 'verifying'}

        self._sweep(ledger, _T0 + 500, escalation_queue=eq, pipeline_phases=recording_phases)
        assert len(lookups) == 0, 'nothing is stuck yet'

        self._sweep(ledger, _T0 + 1500, escalation_queue=eq, pipeline_phases=recording_phases)
        assert len(lookups) == 1

        self._sweep(ledger, _T0 + 2500, escalation_queue=eq, pipeline_phases=recording_phases)
        assert len(lookups) == 1, 'the episode already alarmed and the next tier is not due'

    async def test_one_final_alarm_fires_past_the_last_ladder_multiple(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        ledger, req = self._armed('rs8')
        eq = _FakeEscalationQueue(open_l1=False)  # every prior alarm already closed
        es = _FakeEventStore()

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
            self._sweep(ledger, _T0 + 1500, escalation_queue=eq, event_store=es)
            self._sweep(ledger, _T0 + 2999, escalation_queue=eq, event_store=es)
            assert len(eq.submitted) == 1

            self._sweep(ledger, _T0 + 3001, escalation_queue=eq, event_store=es)
            assert len(eq.submitted) == 2
            assert len(_stuck_warnings(caplog)) == 2, caplog.text
            final = eq.submitted[1]
            assert [d['alarm_number'] for d in _escalation_created_data(es)] == [1, 2]
            assert '2 of 2' in final.detail
            assert req.request_id in final.summary
            assert req.branch.bare_id in final.summary

            self._sweep(ledger, _T0 + 10_000, escalation_queue=eq, event_store=es)
            self._sweep(ledger, _T0 + 100_000, escalation_queue=eq, event_store=es)

        assert len(eq.submitted) == 2
        assert len(_stuck_warnings(caplog)) == 2, caplog.text

    async def test_an_open_l1_at_the_final_tier_consumes_it(self) -> None:
        ledger, _req = self._armed('rs9')
        eq = _FakeEscalationQueue(open_l1=False)

        self._sweep(ledger, _T0 + 1500, escalation_queue=eq)
        assert len(eq.submitted) == 1
        eq.open_it()  # the human has not closed the first alarm

        self._sweep(ledger, _T0 + 3001, escalation_queue=eq)
        assert len(eq.submitted) == 1

        eq.close_it()
        self._sweep(ledger, _T0 + 5000, escalation_queue=eq)
        assert len(eq.submitted) == 1, 'the ladder is spent; a close does not re-fire it'

    async def test_a_spent_ladder_rearms_on_a_new_dequeue_generation(self) -> None:
        ledger, req = self._armed('rs10')
        eq = _FakeEscalationQueue(open_l1=False)
        es = _FakeEventStore()

        self._sweep(ledger, _T0 + 1500, escalation_queue=eq, event_store=es)
        self._sweep(ledger, _T0 + 3001, escalation_queue=eq, event_store=es)
        ledger.on_requeued(req.request_id)
        t2 = _T0 + 10_000
        ledger.on_dequeue(req, now=t2)

        self._sweep(ledger, t2 + 1500, escalation_queue=eq, event_store=es)

        assert [d['alarm_number'] for d in _escalation_created_data(es)] == [1, 2, 1]
