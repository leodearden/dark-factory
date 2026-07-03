"""Unit tests for orchestrator.merge_request_ledger (task MQ-invariants eta / 1992).

Steps covered:
  step-1 RED   — RequestLedger lifecycle + exactly-once
  step-2 GREEN — implement RequestLedger
  step-3 RED   — _alarm_merge_request_stuck + _merge_request_stuck_sentinel
  step-4 GREEN — implement the alarm + sentinel
  step-5 RED   — re-export shim identity + worker._request_ledger attribute

This module intentionally imports only orchestrator.merge_types /
orchestrator.config (NOT orchestrator.merge_queue) at module scope for the
pure data-structure tests, per the plan's "narrow unit-test surface" design
decision — RequestLedger has no git/GitOps dependency at all. The lone
exception is step-5's shim-identity/worker-wiring check, which imports
orchestrator.merge_queue LOCALLY inside its test methods (mirroring every
other RED-until-GREEN symbol in this file) so a not-yet-wired name never
breaks collection of the rest of the file.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_types import MergeOutcome, MergeRequest

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
        branch=branch,
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

    RED until step-2 GREEN adds orchestrator.merge_request_ledger.
    """

    async def test_on_dequeue_arms_the_ledger(self) -> None:
        from orchestrator.merge_request_ledger import RequestLedger

        ledger = RequestLedger()
        req = _make_request('t1', 'task/t1')
        t0 = 1_000_000.0

        ledger.on_dequeue(req, now=t0)

        assert len(ledger) == 1
        assert req.request_id in ledger.open_request_ids()

    async def test_stuck_entries_respects_threshold(self) -> None:
        from orchestrator.merge_request_ledger import RequestLedger

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
        assert entry.branch == 'task/t2'
        assert entry.age_secs == pytest.approx(2000)

    async def test_passive_resolution_via_set_result_sweeps_entry(self) -> None:
        from orchestrator.merge_request_ledger import RequestLedger

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
        from orchestrator.merge_request_ledger import RequestLedger

        ledger = RequestLedger()
        req = _make_request('t4', 'task/t4')
        t0 = 1_000_000.0
        ledger.on_dequeue(req, now=t0)

        req.result.cancel()  # abandoned

        stuck = ledger.stuck_entries(now=t0 + 2000, threshold_s=1000)
        assert stuck == []
        assert len(ledger) == 0

    async def test_on_requeued_removes_then_redequeue_restarts_age(self) -> None:
        from orchestrator.merge_request_ledger import RequestLedger

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
        from orchestrator.merge_request_ledger import RequestLedger

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
        from orchestrator.merge_request_ledger import RequestLedger

        ledger = RequestLedger()
        req = _make_request('t7', 'task/t7')
        ledger.on_dequeue(req, now=1_000_000.0)

        ledger.on_requeued(req.request_id)
        ledger.on_requeued(req.request_id)  # must not raise (no KeyError)

        assert ledger.is_empty()

    async def test_is_empty_reflects_state(self) -> None:
        from orchestrator.merge_request_ledger import RequestLedger

        ledger = RequestLedger()
        assert ledger.is_empty()

        req = _make_request('t8', 'task/t8')
        ledger.on_dequeue(req, now=1_000_000.0)
        assert not ledger.is_empty()

        ledger.on_requeued(req.request_id)
        assert ledger.is_empty()

    # -----------------------------------------------------------------
    # Amendment (post-verification review pass, task 1992): mark_warned /
    # discard — added to close two review findings:
    #   (1) the stuck WARNING log had no per-episode dedup of its own
    #       (only the escalation did), so a leaked request would log
    #       identically on every heartbeat poll forever;
    #   (4) a truly leaked (never-resolved, never-requeued) entry had no
    #       eviction path other than the process dying.
    # -----------------------------------------------------------------

    async def test_mark_warned_reflected_in_stuck_entries_already_warned(self) -> None:
        from orchestrator.merge_request_ledger import RequestLedger

        ledger = RequestLedger()
        req = _make_request('t9', 'task/t9')
        t0 = 1_000_000.0
        ledger.on_dequeue(req, now=t0)

        stuck = ledger.stuck_entries(now=t0 + 2000, threshold_s=1000)
        assert len(stuck) == 1
        assert stuck[0].already_warned is False

        ledger.mark_warned(req.request_id)

        stuck = ledger.stuck_entries(now=t0 + 3000, threshold_s=1000)
        assert len(stuck) == 1
        assert stuck[0].already_warned is True

    async def test_mark_warned_resets_on_requeue_and_redequeue(self) -> None:
        """A fresh episode (on_requeued then a later on_dequeue) must clear
        the previous episode's warned flag — mirrors the escalation's
        has_open_l1 dedup, which likewise doesn't dedup across episodes.
        """
        from orchestrator.merge_request_ledger import RequestLedger

        ledger = RequestLedger()
        req = _make_request('t10', 'task/t10')
        t0 = 1_000_000.0
        ledger.on_dequeue(req, now=t0)
        ledger.mark_warned(req.request_id)

        ledger.on_requeued(req.request_id)
        t2 = t0 + 10_000.0
        ledger.on_dequeue(req, now=t2)  # brand-new entry — fresh episode

        stuck = ledger.stuck_entries(now=t2 + 2000, threshold_s=1000)
        assert len(stuck) == 1
        assert stuck[0].already_warned is False, (
            'a fresh episode after on_requeued + re-on_dequeue must reset warned'
        )

    async def test_mark_warned_is_a_noop_for_absent_request_id(self) -> None:
        from orchestrator.merge_request_ledger import RequestLedger

        ledger = RequestLedger()
        ledger.mark_warned('no-such-request')  # must not raise

    async def test_discard_removes_entry_regardless_of_unresolved_future(self) -> None:
        from orchestrator.merge_request_ledger import RequestLedger

        ledger = RequestLedger()
        req = _make_request('t11', 'task/t11')
        t0 = 1_000_000.0
        ledger.on_dequeue(req, now=t0)
        assert not req.result.done()

        ledger.discard(req.request_id)

        assert ledger.is_empty()
        assert not req.result.done()  # discard never touches the Future

    async def test_discard_is_idempotent_for_absent_request_id(self) -> None:
        from orchestrator.merge_request_ledger import RequestLedger

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
    from orchestrator.merge_request_ledger import StuckRequest

    return StuckRequest(
        request_id=request_id, task_id=task_id, branch=branch,
        age_secs=age_secs, phase=phase,
    )


class TestMergeRequestStuckSentinel:
    """_merge_request_stuck_sentinel (task 1992 step-3).

    RED until step-4 GREEN adds the function to merge_request_ledger.py.
    """

    def test_sentinel_is_per_request_id(self):
        from orchestrator.merge_request_ledger import _merge_request_stuck_sentinel

        assert _merge_request_stuck_sentinel('mr-abc12345') == '__merge_request_stuck__mr-abc12345'


class TestAlarmMergeRequestStuck:
    """_alarm_merge_request_stuck module-level helper (task 1992 step-3).

    RED until step-4 GREEN adds the function to merge_request_ledger.py.
    """

    def _call(self, eq, stuck, *, event_store=None):
        from orchestrator.merge_request_ledger import _alarm_merge_request_stuck
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
# step-5 RED: re-export shim identity + worker._request_ledger attribute
# ---------------------------------------------------------------------------


class TestReexportShimAndWorkerLedgerAttribute:
    """merge_queue re-export shim identity + SpeculativeMergeWorker wiring
    (task 1992 step-5).

    RED until step-6 GREEN adds the re-export shim block to merge_queue.py
    and initialises ``self._request_ledger`` in
    ``SpeculativeMergeWorker.__init__``. merge_queue is imported LOCALLY here
    (not at module scope — see module docstring) so this being RED does not
    break collection of the rest of the file.
    """

    def test_shim_names_are_the_same_object_as_the_source_module(self):
        import orchestrator.merge_request_ledger as ledger_module
        from orchestrator.merge_queue import (
            RequestLedger,
            StuckRequest,
            _alarm_merge_request_stuck,
            _merge_request_stuck_sentinel,
        )

        assert RequestLedger is ledger_module.RequestLedger
        assert StuckRequest is ledger_module.StuckRequest
        assert _alarm_merge_request_stuck is ledger_module._alarm_merge_request_stuck
        assert _merge_request_stuck_sentinel is ledger_module._merge_request_stuck_sentinel

    def test_fresh_worker_has_an_empty_request_ledger(self, tmp_path: Path):
        from orchestrator.merge_queue import RequestLedger, SpeculativeMergeWorker

        git_config = GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
        )
        git_ops = GitOps(git_config, tmp_path)
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue())

        assert isinstance(worker._request_ledger, RequestLedger)
        assert worker._request_ledger.is_empty()
