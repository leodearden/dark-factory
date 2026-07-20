"""Tests for task μ's harness-side routing-tier bump machinery (task 2542,
plans/adaptive-model-routing-prd.md Phase 4).

Covers the harness seams that stamp ``metadata.routing.routing_tier`` (γ's
harness-owned, monotonic counter) so a re-dispatch after a failed dispatch
routes one ladder rung stronger via the retry-tier-up rule (see
test_routing_retry_tier_up.py):

  - ``_bumped_routing_dump``  -- the pure helper backing every bump site (step-3/4)
  - ``_maybe_bump_routing_tier`` -- the terminal-failure (BLOCKED) auto-bump in
    ``_run_slot``'s finally (step-5/6)
  - ``_maybe_auto_eval`` carrying the redo sibling at parent tier+1 (step-7/8)
  - ``_bump_routing_tier_by_id`` / ``pre_increment_routing_tier`` -- the
    escalate_model by-id bump scheduled off the FastMCP worker (step-9/10)

RED (step-3) until ``_bumped_routing_dump`` exists in ``orchestrator.harness``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from orchestrator.harness import Harness, _bumped_routing_dump
from orchestrator.workflow import WorkflowOutcome


def _valid_mirror(routing_tier: int = 0) -> dict:
    """A minimally-valid ``RoutingDecisionMirror`` dict (every required field
    set) so ``RoutingState.from_metadata`` reconstructs it rather than
    tolerant-degrading to a fresh default."""
    return {
        'role': 'implementer',
        'model': 'sonnet',
        'effort': 'high',
        'budget_usd': 10.0,
        'max_turns': 80,
        'source_layer': 'config',
        'rule_id': None,
        'rejected': [],
        'routing_tier': routing_tier,
        'decided_at': None,
    }


class TestBumpedRoutingDump:
    """``_bumped_routing_dump(metadata, by=1)``: read the RoutingState from
    ``metadata['routing']``, increment ``routing_tier`` by ``by``, and
    serialize -- preserving every other field (this helper only ever adds, so
    the monotonicity of invariant 8 holds by construction)."""

    def test_bumps_zero_to_one(self):
        dump = _bumped_routing_dump({'routing': {'routing_tier': 0}})
        assert dump['routing_tier'] == 1

    def test_preserves_latest_history_and_saturated(self):
        latest = _valid_mirror(routing_tier=2)
        metadata = {
            'routing': {
                'routing_tier': 2,
                'simple_saturated': True,
                'latest': latest,
                'history': [latest],
            }
        }

        dump = _bumped_routing_dump(metadata)

        assert dump['routing_tier'] == 3
        # Only the counter moved -- latest/history/simple_saturated ride along.
        assert dump['simple_saturated'] is True
        assert dump['latest'] is not None
        assert dump['latest']['model'] == 'sonnet'
        assert len(dump['history']) == 1

    def test_none_metadata_yields_fresh_tier_one(self):
        dump = _bumped_routing_dump(None)
        assert dump['routing_tier'] == 1
        assert dump['latest'] is None
        assert dump['history'] == []
        assert dump['simple_saturated'] is False

    def test_non_dict_metadata_yields_fresh_tier_one(self):
        dump = _bumped_routing_dump('not-a-dict')
        assert dump['routing_tier'] == 1

    def test_by_two_adds_two(self):
        dump = _bumped_routing_dump({'routing': {'routing_tier': 1}}, by=2)
        assert dump['routing_tier'] == 3


class TestMaybeBumpRoutingTier:
    """``Harness._maybe_bump_routing_tier(assignment, report)``: the
    terminal-failure auto-bump called from ``_run_slot``'s finally.

    Fires ONLY on ``report.outcome == BLOCKED`` -- the unambiguous
    terminal-failed dispatch that boundary test 5 exercises. DONE (the success
    path, boundary test 6) and REQUEUED (an in-process retry of the SAME work
    — deferred by design decision, since there is no clean per-requeue
    lost-work signal here) do NOT bump. Called as an unbound method against a
    duck-typed self so the unit stays isolated to ``self.scheduler.update_task``.
    """

    @staticmethod
    def _assignment(routing_tier: int = 0):
        return SimpleNamespace(
            task={'metadata': {'routing': {'routing_tier': routing_tier}}},
            task_id='42',
        )

    @staticmethod
    def _harness():
        return SimpleNamespace(scheduler=SimpleNamespace(update_task=AsyncMock()))

    @pytest.mark.asyncio
    async def test_blocked_bumps_via_merge(self):
        harness = self._harness()
        report = SimpleNamespace(outcome=WorkflowOutcome.BLOCKED)

        await Harness._maybe_bump_routing_tier(harness, self._assignment(0), report)

        harness.scheduler.update_task.assert_awaited_once()
        args, kwargs = harness.scheduler.update_task.call_args
        assert args[0] == '42'
        assert args[1]['routing']['routing_tier'] == 1
        assert kwargs['metadata_mode'] == 'merge'

    @pytest.mark.asyncio
    async def test_done_does_not_bump(self):
        harness = self._harness()
        report = SimpleNamespace(outcome=WorkflowOutcome.DONE)

        await Harness._maybe_bump_routing_tier(harness, self._assignment(0), report)

        harness.scheduler.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_requeued_does_not_bump(self):
        harness = self._harness()
        report = SimpleNamespace(outcome=WorkflowOutcome.REQUEUED)

        await Harness._maybe_bump_routing_tier(harness, self._assignment(0), report)

        harness.scheduler.update_task.assert_not_awaited()
