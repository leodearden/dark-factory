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

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from orchestrator.config import OrchestratorConfig
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


class TestAutoEvalRedoCarriesBumpedTier:
    """``_maybe_auto_eval``'s redo sibling starts at ``parent_tier + 1`` (task
    μ, trigger 1b): the redo carries a FRESH routing state
    (``{'routing_tier': parent+1}``, NOT a copy of the parent's decision
    history) alongside the existing ``force_full_path=True``. Harness-set at
    submit time = harness-stamped, not author-supplied (invariant 8).

    Uses a real Harness so ``_extract_task_id`` / ``git_ops`` are genuine; the
    budget + prior-redo lookups are stubbed and ``dispatch_tool`` returns ``{}``
    so ``_maybe_auto_eval`` returns right after capturing the submit_task call.
    """

    def _harness(self, tmp_path: Path) -> Harness:
        harness = Harness(OrchestratorConfig(project_root=tmp_path))
        # Budget OK + no prior redos → straight through to submit_task.
        harness._auto_eval_budget_used_24h = AsyncMock(return_value=0.0)
        harness._find_prior_auto_eval_redos = AsyncMock(return_value=[])
        # submit_task capture; {} → _extract_task_id None → early return before
        # any further dispatch_tool call, so call_args_list[0] IS submit_task.
        harness.scheduler.dispatch_tool = AsyncMock(return_value={})
        return harness

    @staticmethod
    def _assignment(routing_tier: int | None):
        routing = {} if routing_tier is None else {'routing': {'routing_tier': routing_tier}}
        return SimpleNamespace(
            task={
                'metadata': {'optimistic_path': 'lever-b', **routing},
                'title': 'do the thing',
                'priority': 'medium',
            },
            task_id='7',
        )

    def _submitted_metadata(self, harness: Harness) -> dict:
        call = harness.scheduler.dispatch_tool.call_args_list[0]
        assert call.args[0] == 'submit_task'
        return call.args[1]['metadata']

    @pytest.mark.asyncio
    async def test_redo_carries_parent_tier_plus_one(self, tmp_path: Path):
        harness = self._harness(tmp_path)
        report = SimpleNamespace(outcome=WorkflowOutcome.BLOCKED, block_phase='verify')

        await harness._maybe_auto_eval(self._assignment(routing_tier=2), report)

        submitted = self._submitted_metadata(harness)
        assert submitted.get('routing') == {'routing_tier': 3}
        assert submitted['force_full_path'] is True

    @pytest.mark.asyncio
    async def test_redo_from_parent_without_routing_starts_at_tier_one(self, tmp_path: Path):
        harness = self._harness(tmp_path)
        report = SimpleNamespace(outcome=WorkflowOutcome.BLOCKED, block_phase='verify')

        await harness._maybe_auto_eval(self._assignment(routing_tier=None), report)

        submitted = self._submitted_metadata(harness)
        assert submitted.get('routing') == {'routing_tier': 1}
        assert submitted['force_full_path'] is True


class TestBumpRoutingTierById:
    """``Harness._bump_routing_tier_by_id(task_id)``: the escalate_model by-id
    bump (task μ, trigger 3). Fetches the task's CURRENT metadata via
    ``scheduler.get_task`` and writes the +1 tier through
    ``metadata_mode='merge'``; no-ops when the task is absent (a synthetic
    sentinel id resolves to None)."""

    @pytest.mark.asyncio
    async def test_fetch_then_bump_via_merge(self):
        harness = SimpleNamespace(
            scheduler=SimpleNamespace(
                get_task=AsyncMock(return_value={'metadata': {'routing': {'routing_tier': 0}}}),
                update_task=AsyncMock(),
            ),
        )

        await Harness._bump_routing_tier_by_id(harness, '42')

        harness.scheduler.get_task.assert_awaited_once_with('42')
        harness.scheduler.update_task.assert_awaited_once()
        args, kwargs = harness.scheduler.update_task.call_args
        assert args[0] == '42'
        assert args[1]['routing']['routing_tier'] == 1
        assert kwargs['metadata_mode'] == 'merge'

    @pytest.mark.asyncio
    async def test_absent_task_is_noop(self):
        harness = SimpleNamespace(
            scheduler=SimpleNamespace(
                get_task=AsyncMock(return_value=None),
                update_task=AsyncMock(),
            ),
        )

        await Harness._bump_routing_tier_by_id(harness, 'nope')

        harness.scheduler.update_task.assert_not_awaited()


class TestPreIncrementRoutingTier:
    """``Harness.pre_increment_routing_tier(task_id)``: the SYNC shim the
    (sync, off-loop) FastMCP resolve_issue worker calls. It must NOT do the
    metadata write itself — it schedules the async ``_bump_routing_tier_by_id``
    coroutine onto the loop via ``_schedule_coro_threadsafe`` (keeping the
    metadata write harness-owned and on-loop)."""

    def test_schedules_bump_coroutine_onto_loop(self):
        # _bump_routing_tier_by_id + _schedule_coro_threadsafe are Mocked so no
        # real coroutine is created (nothing to leave un-awaited) and the
        # wiring is asserted directly: pre_increment builds the coro for the
        # given task_id and hands it, with a label, to the loop bridge.
        coro_sentinel = object()
        harness = SimpleNamespace(
            _bump_routing_tier_by_id=Mock(return_value=coro_sentinel),
            _schedule_coro_threadsafe=Mock(),
        )

        Harness.pre_increment_routing_tier(harness, '42')

        harness._bump_routing_tier_by_id.assert_called_once_with('42')
        harness._schedule_coro_threadsafe.assert_called_once()
        sched_args, sched_kwargs = harness._schedule_coro_threadsafe.call_args
        assert sched_args[0] is coro_sentinel
        assert 'label' in sched_kwargs
