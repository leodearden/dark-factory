"""Unit tests for orchestrator.merge_request_ledger (task MQ-invariants eta / 1992).

Steps covered:
  step-1 RED   — RequestLedger lifecycle + exactly-once
  step-2 GREEN — implement RequestLedger
  step-3 RED   — _alarm_merge_request_stuck + _merge_request_stuck_sentinel
  step-4 GREEN — implement the alarm + sentinel
  step-5 RED   — re-export shim identity + worker._request_ledger attribute

This module intentionally imports only orchestrator.merge_types /
orchestrator.config (NOT orchestrator.merge_queue) for the pure
data-structure tests, per the plan's "narrow unit-test surface" design
decision — RequestLedger has no git/GitOps dependency at all.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import OrchestratorConfig
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
