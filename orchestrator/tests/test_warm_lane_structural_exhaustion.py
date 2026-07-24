"""Tests for task 2988 — EXHAUSTED requeue-cap flip + structural-exhaustion
pool-level L2 (PRD ε / W3, plans/warm-lane-exhaustion-hardening-prd.md).

Two coupled mechanisms closing both failure poles of the 2026-07-22 incident:
  (1) Cap-semantics: a WarmLanePoolExhausted requeue no longer counts against
      the per-task requeue cap (threaded table-driven from the disposition
      flip through TerminalReport -> TaskReport -> Scheduler.record_requeue).
  (2) Structural-exhaustion L2: a pool-GLOBAL consecutive-EXHAUSTED counter on
      GitOps fires a harness-installed born-at-L2 escalation after N trips.

Test classes (added across steps 3/7/9/11 — the record_requeue route unit
test lives in test_retry_cap.py, step-5):
  step-03: TestStructuralExhaustionConfigKnob — the GitConfig threshold field
  step-07: test_warm_lane_pool_exhausted_does_not_count — end-to-end retry-cap
  step-09: TestConsecutiveExhaustedCounter — git_ops counter mechanics
  step-11: TestFileStructuralExhaustionL2 / TestStructuralExhaustionWiring —
           the harness filer, dedup, no-op, and callback install
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


# ---------------------------------------------------------------------------
# step-03: GitConfig.warm_lane_structural_exhaustion_l2_threshold knob
# ---------------------------------------------------------------------------


class TestStructuralExhaustionConfigKnob:
    """GitConfig.warm_lane_structural_exhaustion_l2_threshold (task 2988):
    green-tier/hot-reloadable, default 5 (PRD Open Q1), ge=1.  Mirrors
    warm_lane_drift_l2_threshold's Field conventions; defaults.yaml
    deliberately omits it (the Field default is the single source of truth).
    """

    def test_default_is_five(self):
        from orchestrator.config import GitConfig
        assert GitConfig().warm_lane_structural_exhaustion_l2_threshold == 5

    def test_zero_is_rejected_ge_one(self):
        from pydantic import ValidationError

        from orchestrator.config import GitConfig
        with pytest.raises(ValidationError):
            GitConfig(warm_lane_structural_exhaustion_l2_threshold=0)

    def test_negative_is_rejected_ge_one(self):
        from pydantic import ValidationError

        from orchestrator.config import GitConfig
        with pytest.raises(ValidationError):
            GitConfig(warm_lane_structural_exhaustion_l2_threshold=-1)

    def test_accepts_explicit_positive_override(self):
        from orchestrator.config import GitConfig
        cfg = GitConfig(warm_lane_structural_exhaustion_l2_threshold=2)
        assert cfg.warm_lane_structural_exhaustion_l2_threshold == 2


# ---------------------------------------------------------------------------
# step-07: end-to-end — a WarmLanePoolExhausted requeue does NOT burn the
# per-task retry cap.  The counts_against_requeue_cap flag is threaded, table-
# driven, from the disposition -> TerminalReport -> TaskReport -> record_requeue.
# ---------------------------------------------------------------------------


def _build_task_report(
    outcome,
    *,
    counts_against_requeue_cap: bool = True,
    block_reason: str = 'warm_lane_pool_exhausted',
    cost_usd: float = 1.0,
    steward_cost_usd: float = 0.5,
):
    from orchestrator.harness import TaskReport

    return TaskReport(
        task_id='t1',
        title='T1',
        outcome=outcome,
        cost_usd=cost_usd,
        steward_cost_usd=steward_cost_usd,
        block_reason=block_reason,
        block_detail='pool census: all lanes assigned',
        block_phase='plan',
        counts_against_requeue_cap=counts_against_requeue_cap,
    )


@pytest.mark.asyncio
class TestWarmLanePoolExhaustedDoesNotCount:
    """Task 2988 prescribed observable: driving requeue_cap+2 EXHAUSTED
    (counts_against_requeue_cap=False) REQUEUED reports through
    ``harness._apply_retry_cap`` leaves the scheduler's GENUINE requeue counter
    at 0, the transient counter at 0, and NEVER fires
    ``trigger_retry_cap_exhausted`` — so no retry-cap escalation, no
    reblock-guard arming (the 2026-07-22 incident's L2 storm).  A DONE report
    still clears, and a ``counts_against_requeue_cap=True`` REQUEUED still
    increments (the general retry-cap path is unbroken).
    """

    def _config(self, tmp_path):
        from orchestrator.config import OrchestratorConfig
        return OrchestratorConfig(
            max_per_module=1, requeue_cap=3, project_root=tmp_path,
        )

    async def test_warm_lane_pool_exhausted_does_not_count(self, tmp_path):
        from test_retry_cap import _make_harness

        from orchestrator.workflow import WorkflowOutcome

        config = self._config(tmp_path)
        harness = _make_harness(config)
        trigger = AsyncMock()
        harness.scheduler.trigger_retry_cap_exhausted = trigger  # type: ignore[method-assign]

        n = config.requeue_cap + 2
        results = []
        for _ in range(n):
            results.append(
                await harness._apply_retry_cap(
                    't1',
                    _build_task_report(
                        WorkflowOutcome.REQUEUED,
                        counts_against_requeue_cap=False,
                    ),
                    True,
                )
            )

        # No retry-cap escalation, no reblock arming: trigger never fired.
        assert trigger.await_count == 0
        # Neither the genuine nor the transient counter moved.
        assert harness.scheduler._requeue_counts.get('t1', 0) == 0
        assert harness.scheduler.transient_requeue_count('t1') == 0
        # The caller's requeued flag is preserved (cooldown path unaffected).
        assert all(r is True for r in results)
        # ...but the attempt timeline is preserved for a later genuine failure.
        assert len(harness.scheduler.requeue_history('t1')) == n

    async def test_done_clears_and_counting_requeue_still_increments(self, tmp_path):
        from test_retry_cap import _make_harness

        from orchestrator.workflow import WorkflowOutcome

        config = self._config(tmp_path)
        harness = _make_harness(config)
        harness.scheduler.trigger_retry_cap_exhausted = AsyncMock()  # type: ignore[method-assign]

        # A counts_against_requeue_cap=True REQUEUED DOES increment the genuine
        # counter (guard the general path).
        await harness._apply_retry_cap(
            't1',
            _build_task_report(
                WorkflowOutcome.REQUEUED,
                counts_against_requeue_cap=True,
                block_reason='architect returned empty output',
            ),
            True,
        )
        assert harness.scheduler._requeue_counts['t1'] == 1

        # A non-counting EXHAUSTED requeue interleaves WITHOUT shifting it.
        await harness._apply_retry_cap(
            't1',
            _build_task_report(
                WorkflowOutcome.REQUEUED, counts_against_requeue_cap=False,
            ),
            True,
        )
        assert harness.scheduler._requeue_counts['t1'] == 1

        # DONE clears everything.
        await harness._apply_retry_cap(
            't1', _build_task_report(WorkflowOutcome.DONE), False,
        )
        assert 't1' not in harness.scheduler._requeue_counts
        assert harness.scheduler.requeue_history('t1') == []
