"""Each best-effort step of a full Stage 1 cycle fails in isolation.

A failing step is logged and swallowed: ``MemoryConsolidator.run`` still
returns a report, the step publishes none of its success stats, and every
later step still runs, through to the cycle-summary write. Cancellation is
never swallowed.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from escalation.queue import EscalationQueue

from fused_memory.models.reconciliation import StageId, StageReport, Watermark
from fused_memory.reconciliation.consolidation_auto_executor import (
    auto_consolidation_initial_stats,
)
from fused_memory.reconciliation.stages.base import BaseStage
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.task_filter import FilteredTaskTree
from reconciliation.consolidator_fixtures import make_consolidator

CONSOLIDATOR = 'fused_memory.reconciliation.stages.memory_consolidator'
FLAG_CHAIN = 'fused_memory.reconciliation.stage1_flag_chain'

_CURATOR_GATE_ZEROS = dict.fromkeys(
    (
        'curator_gate_resolution_scanned',
        'curator_gate_resolution_flags_emitted',
        'curator_gate_resolution_errors',
    ),
    0,
)
_ORPHAN_SWEEP_ZEROS = dict.fromkeys(
    (
        'orphaned_recon_escalations_scanned',
        'orphaned_recon_escalations_terminal',
        'orphaned_recon_escalations_missing',
        'orphaned_recon_escalations_live',
        'orphaned_recon_escalations_ambiguous',
        'orphaned_recon_escalations_unresolvable',
        'orphaned_recon_escalations_errors',
        'orphaned_recon_escalations_flags_emitted',
    ),
    0,
)

# (callee patched in memory_consolidator, prefixes of that step's own stats,
#  those stats as they stand before the step runs)
BEST_EFFORT_STEPS = [
    pytest.param(
        'sweep_resolved_curator_gates', ('curator_gate_resolution_',), _CURATOR_GATE_ZEROS,
        id='curator-gate-sweep',
    ),
    pytest.param(
        'run_auto_consolidation', ('auto_consolidation',), auto_consolidation_initial_stats(),
        id='auto-consolidation',
    ),
    pytest.param(
        'sweep_orphaned_recon_escalations', ('orphaned_recon_escalations_',), _ORPHAN_SWEEP_ZEROS,
        id='orphan-sweep',
    ),
    pytest.param(
        'sweep_degenerate_task_nodes', ('degenerate_task_nodes_',), {},
        id='degenerate-task-node-sweep',
    ),
    pytest.param(
        'sweep_stale_status_snapshot_edges',
        ('stale_status_snapshot_edges_', 'stale_blocked_edges_'),
        {},
        id='status-snapshot-edge-sweep',
    ),
    pytest.param(
        'sweep_stale_priority_override_edges', ('stale_priority_override_edges_',), {},
        id='priority-override-edge-sweep',
    ),
]


def _stage(tmp_path: Path) -> MemoryConsolidator:
    """A stage on which every full-cycle step is reached."""
    stage = make_consolidator()
    stage.memory.recon_ledger = None
    stage._escalation_queue = EscalationQueue(tmp_path / 'escalations')
    stage.filtered_task_tree = FilteredTaskTree(
        active_tasks=[{'id': '100', 'status': 'pending'}],
        done_tasks=[{'id': '7', 'status': 'done'}],
    )
    return stage


def _one_flag_report() -> StageReport:
    return StageReport(
        stage=StageId.memory_consolidator,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        items_flagged=[
            {
                'task_id': '100',
                'flag_type': 'missing_deliverable',
                'description': 'Task 100 has no deliverable',
            },
        ],
        stats={},
    )


@contextlib.contextmanager
def _full_cycle(failing_callee: str, error: BaseException, cycle_summary: AsyncMock):
    """A one-flag full cycle in which *failing_callee* raises *error*; yields its mock."""
    failing = AsyncMock(side_effect=error)
    with (
        patch.object(BaseStage, 'run', new=AsyncMock(return_value=_one_flag_report())),
        patch(f'{FLAG_CHAIN}.dedup_flags', new=AsyncMock(side_effect=lambda **kw: kw['flags'])),
        patch(
            f'{FLAG_CHAIN}.filter_false_absence_flags',
            new=AsyncMock(side_effect=lambda taskmaster, project_root, flags: flags),
        ),
        patch(f'{FLAG_CHAIN}.acknowledge_resolved_flags', new=AsyncMock(return_value=0)),
        patch(f'{CONSOLIDATOR}.write_stage1_cycle_summary', new=cycle_summary),
        patch(f'{CONSOLIDATOR}.{failing_callee}', new=failing),
    ):
        yield failing


async def _run(stage: MemoryConsolidator) -> StageReport:
    return await stage.run(
        events=[],
        watermark=Watermark(project_id=stage.project_id),
        prior_reports=[],
        run_id='run-failure-isolation',
    )


def _own_stats(stats: dict[str, Any], prefixes: tuple[str, ...]) -> dict[str, Any]:
    return {key: value for key, value in stats.items() if key.startswith(prefixes)}


@pytest.mark.asyncio
@pytest.mark.parametrize(('callee', 'prefixes', 'pre_run_stats'), BEST_EFFORT_STEPS)
async def test_a_failing_step_is_isolated(tmp_path, callee, prefixes, pre_run_stats):
    stage = _stage(tmp_path)
    cycle_summary = AsyncMock(return_value=True)

    with _full_cycle(callee, RuntimeError(f'{callee} exploded'), cycle_summary) as failing:
        report = await _run(stage)

    failing.assert_awaited_once()
    assert isinstance(report, StageReport)
    assert _own_stats(report.stats, prefixes) == pre_run_stats
    cycle_summary.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(('callee', 'prefixes', 'pre_run_stats'), BEST_EFFORT_STEPS)
async def test_cancellation_is_never_swallowed(tmp_path, callee, prefixes, pre_run_stats):
    stage = _stage(tmp_path)

    with (
        _full_cycle(callee, asyncio.CancelledError(), AsyncMock(return_value=True)),
        pytest.raises(asyncio.CancelledError),
    ):
        await _run(stage)
