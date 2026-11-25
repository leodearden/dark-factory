"""A review checkpoint runs beside dispatch: the loop keeps ticking and dispatching while it runs.

Drives the real ``Harness.run`` loop over a real ``Scheduler`` through
``_run_loop_board``.  The harness holds a real ``ReviewCheckpoint`` whose
public ``run_focused`` is held until the test releases it, so a merge reaches
the checkpoint through the harness's own slot-report path and starts a
checkpoint that does not finish while the test watches the loop.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _run_loop_board import Board, build_board_harness, fleet_dir_env, plain, run_driven, until
from shared.project_layout import scheduler_state_json
from shared.scheduler_state import SchedulerStateSnapshot

from orchestrator.config import ReviewConfig
from orchestrator.harness import Harness
from orchestrator.review_checkpoint import ReviewCheckpoint, ReviewOutcome, ReviewReport
from orchestrator.verify import VerifyResult

pytestmark = pytest.mark.asyncio

_UNIT = 'orchestrator-review-checkpoint-dispatch-test.service'
_PHASE1_PASSED = VerifyResult(
    passed=True, summary='OK', test_output='', lint_output='', type_output='',
)


@pytest.fixture(autouse=True)
def _fleet_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fleet_dir_env(monkeypatch, tmp_path, _UNIT)


@dataclass(frozen=True)
class _Scene:
    harness: Harness
    board: Board
    entered: asyncio.Event
    release: asyncio.Event

    async def start_checkpoint(self) -> None:
        """Let task 1 merge, which starts a review checkpoint; return once a tick begun since then completes.

        A tick already in flight when the checkpoint starts could pick up work
        filed afterwards, so the test files only after a fresh tick.
        """
        await self.board.until(lambda: self.board.started('1'), 'task 1 dispatches')
        self.board.gates['1'].set()
        await until(self.entered.is_set, 'task 1 merging starts a review checkpoint')
        checkpoint_running = self.board.mark()
        await self.board.until(
            lambda: bool(self.board.ticks_since(checkpoint_running)),
            'the run loop ticks while the review checkpoint runs',
        )


def _held_checkpoint_scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Scene:
    h, board = build_board_harness(
        tmp_path, monkeypatch, general_slots=2, reserve=0, tasks=[plain('1')],
    )
    checkpoint = ReviewCheckpoint(
        h.config.model_copy(
            update={'review': ReviewConfig(interval=1, full_review_on_complete=False)},
        ),
        mcp=MagicMock(),
        usage_gate=None,
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def held_run_focused() -> ReviewReport:
        entered.set()
        await release.wait()
        return ReviewReport(
            review_id='held', mode='focused', changed_modules=[], phase1=_PHASE1_PASSED,
            outcome=ReviewOutcome.RAN, briefing_missing=False,
        )

    monkeypatch.setattr(checkpoint, 'run_focused', held_run_focused)
    h.review_checkpoint = checkpoint
    return _Scene(h, board, entered, release)


async def test_a_ready_task_dispatches_into_a_free_slot_while_a_checkpoint_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _held_checkpoint_scene(tmp_path, monkeypatch)
    board = scene.board

    async def drive() -> None:
        try:
            await scene.start_checkpoint()
            board.file(plain('2'))
            await board.until(
                lambda: board.started('2'), 'task 2 dispatches while the review checkpoint runs',
            )
            assert not scene.release.is_set()
        finally:
            scene.release.set()

    await run_driven(scene.harness, board, drive)


async def test_the_scheduler_snapshot_advances_while_a_checkpoint_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _held_checkpoint_scene(tmp_path, monkeypatch)
    board = scene.board
    snapshot_path = scheduler_state_json(tmp_path)

    def snapshot_shows_task_2_holding_since(checkpoint_started: datetime) -> bool:
        if not snapshot_path.exists():
            return False
        snapshot = SchedulerStateSnapshot.from_json(json.loads(snapshot_path.read_text()))
        return (
            '2' in snapshot.held_by_task
            and snapshot.snapshot_at is not None
            and snapshot.snapshot_at > checkpoint_started
        )

    async def drive() -> None:
        try:
            await scene.start_checkpoint()
            checkpoint_started = datetime.now(UTC)
            board.file(plain('2'))
            await board.until(
                lambda: snapshot_shows_task_2_holding_since(checkpoint_started),
                'the scheduler snapshot, written while the review checkpoint runs, shows task 2',
            )
        finally:
            scene.release.set()

    await run_driven(scene.harness, board, drive)
