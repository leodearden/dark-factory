"""Stage 1's full-cycle flag chain, driven through stage1_flag_chain's public seam.

The chain runs every drop filter and the two suppressing steps over a cycle's
flags, then acknowledges the markers of flags it dropped as resolved. These
tests build a FlagChainContext over AsyncMock services and assert on the
FlagChainOutcome alone: survivors, published stats and the Hook A result.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from fused_memory.reconciliation.flag_dedup import EntityStandingSuppressionResult
from fused_memory.reconciliation.stage1_flag_chain import (
    FlagChainContext,
    FlagChainOutcome,
    ResolvingFilter,
    SuppressionStormEscape,
    apply_resolving_filters,
    run_stage1_flag_chain,
)

CHAIN = 'fused_memory.reconciliation.stage1_flag_chain'

PRESENT_TASK = {'id': '3438', 'title': 'real task 3438'}


def _memory_without_ledger() -> AsyncMock:
    """A memory service whose recon_ledger is unset, so every ledger-backed
    step (Hook A, dedup_flags' suppression gate, acknowledgment) fails open."""
    memory = AsyncMock()
    memory.recon_ledger = None
    return memory


async def _no_storm_escape(result: EntityStandingSuppressionResult) -> None:
    return None


def _context(
    taskmaster: AsyncMock,
    memory: AsyncMock | None = None,
    escalate_suppression_storm: SuppressionStormEscape = _no_storm_escape,
) -> FlagChainContext:
    return FlagChainContext(
        memory_service=memory if memory is not None else _memory_without_ledger(),
        taskmaster=taskmaster,
        escalate_suppression_storm=escalate_suppression_storm,
        project_id='test_project',
        project_root='/tmp/test',
        known_projects={},
        run_id='run-flag-chain',
        events=[],
        active_tasks=None,
    )


def _taskmaster_returning(task: dict) -> AsyncMock:
    taskmaster = AsyncMock()
    taskmaster.get_task = AsyncMock(return_value=task)
    return taskmaster


def _flag(task_id: str, flag_type: str) -> dict:
    return {
        'task_id': task_id,
        'flag_type': flag_type,
        'description': f'Task {task_id}: {flag_type}',
    }


async def _run(flags: list[dict], ctx: FlagChainContext) -> FlagChainOutcome:
    return await run_stage1_flag_chain([dict(f) for f in flags], ctx)


@pytest.mark.asyncio
async def test_apply_resolving_filters_runs_in_order_and_counts_each_filters_drops():
    seen_by_second: list[list[dict]] = []

    async def drop_a(flags: list[dict], ctx: FlagChainContext) -> list[dict]:
        return [f for f in flags if f['task_id'] != 'a']

    async def drop_b(flags: list[dict], ctx: FlagChainContext) -> list[dict]:
        seen_by_second.append(list(flags))
        return [f for f in flags if f['task_id'] != 'b']

    flags = [_flag('a', 'x'), _flag('b', 'x'), _flag('c', 'x')]
    survivors, stats = await apply_resolving_filters(
        flags,
        _context(AsyncMock()),
        (ResolvingFilter('a_dropped', drop_a), ResolvingFilter('b_dropped', drop_b)),
    )

    assert seen_by_second == [[flags[1], flags[2]]]
    assert survivors == [flags[2]]
    assert stats == {'a_dropped': 1, 'b_dropped': 1}


@pytest.mark.asyncio
async def test_chain_publishes_terminal_metadata_drop_count():
    stale = _flag('7', 'stale_metadata')
    unrelated = _flag('100', 'missing_deliverable')

    outcome = await _run(
        [stale, unrelated], _context(_taskmaster_returning({'id': '7', 'status': 'done'})),
    )

    assert [f['flag_type'] for f in outcome.kept_flags] == ['missing_deliverable']
    assert outcome.stats['terminal_metadata_flags_dropped'] == 1


@pytest.mark.asyncio
async def test_chain_publishes_false_absence_drop_count():
    absent = _flag('3438', 'task_absent')
    unrelated = _flag('100', 'missing_deliverable')

    outcome = await _run([absent, unrelated], _context(_taskmaster_returning(PRESENT_TASK)))

    assert [f['flag_type'] for f in outcome.kept_flags] == ['missing_deliverable']
    assert outcome.stats['false_absence_flags_dropped'] == 1


@pytest.mark.asyncio
async def test_resolving_drops_are_acknowledged_and_dedup_suppressions_are_not():
    resolved = _flag('3438', 'task_absent')
    suppressed = _flag('200', 'missing_deliverable')
    kept = _flag('100', 'missing_deliverable')

    async def dedup_dropping_suppressed(**kwargs):
        return [f for f in kwargs['flags'] if f['task_id'] != suppressed['task_id']]

    acknowledge = AsyncMock(return_value=7)
    with (
        patch(f'{CHAIN}.dedup_flags', new=dedup_dropping_suppressed),
        patch(f'{CHAIN}.acknowledge_resolved_flags', new=acknowledge),
    ):
        outcome = await _run(
            [resolved, suppressed, kept], _context(_taskmaster_returning(PRESENT_TASK)),
        )

    assert acknowledge.await_args is not None
    acknowledged = acknowledge.await_args.kwargs['resolved_flags']
    assert resolved in acknowledged
    assert suppressed not in acknowledged
    assert kept not in acknowledged
    assert outcome.stats['stage1_flag_markers_acknowledged'] == 7
    assert outcome.kept_flags == [kept]


@pytest.mark.asyncio
async def test_hook_a_suppressions_are_not_acknowledged():
    resolved = _flag('3438', 'task_absent')
    held = _flag('200', 'missing_deliverable')
    kept = _flag('100', 'missing_deliverable')

    async def hook_a_holding_back_held(**kwargs) -> EntityStandingSuppressionResult:
        return EntityStandingSuppressionResult(
            kept_flags=[f for f in kwargs['flags'] if f['task_id'] != held['task_id']],
            suppressed_by_decision={'decision-entity': 1},
            grounds_by_decision={'decision-entity': 'standing decision covers task 200'},
            suppression_evaluated=True,
        )

    acknowledge = AsyncMock(return_value=1)
    with (
        patch(f'{CHAIN}.filter_entity_standing_decisions', new=hook_a_holding_back_held),
        patch(f'{CHAIN}.acknowledge_resolved_flags', new=acknowledge),
    ):
        outcome = await _run(
            [resolved, held, kept], _context(_taskmaster_returning(PRESENT_TASK)),
        )

    assert acknowledge.await_args is not None
    acknowledged = acknowledge.await_args.kwargs['resolved_flags']
    assert resolved in acknowledged
    assert held not in acknowledged
    assert kept not in acknowledged
    assert outcome.stats['entity_standing_decision_suppressed'] == 1
    assert [f['task_id'] for f in outcome.kept_flags] == [kept['task_id']]


@pytest.mark.asyncio
async def test_outcome_carries_hook_a_result_which_fails_open_without_a_ledger():
    flags = [_flag('100', 'missing_deliverable')]

    outcome = await _run(flags, _context(_taskmaster_returning(PRESENT_TASK)))

    assert outcome.standing_decision_result.suppressed_by_decision == {}
    assert outcome.standing_decision_result.suppression_evaluated is False
    assert outcome.standing_decision_result.kept_flags == flags
    assert outcome.stats['entity_standing_decision_suppressed'] == 0


@pytest.mark.asyncio
async def test_chain_hands_hook_a_result_to_the_injected_storm_escape_once():
    escaped: list[EntityStandingSuppressionResult] = []

    async def record_storm_escape(result: EntityStandingSuppressionResult) -> None:
        escaped.append(result)

    ctx = _context(
        _taskmaster_returning(PRESENT_TASK),
        memory=_memory_without_ledger(),
        escalate_suppression_storm=record_storm_escape,
    )

    outcome = await _run([_flag('100', 'missing_deliverable')], ctx)

    assert len(escaped) == 1
    assert escaped[0] is outcome.standing_decision_result
