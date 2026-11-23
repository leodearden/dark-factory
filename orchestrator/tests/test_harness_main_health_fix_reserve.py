"""The 2026-11-04 / 2026-11-15 deadlock replayed through the real ``Harness.run`` loop (task 8480).

Every general run slot holds a merge-waiter that cannot land until main is
healed.  A main-health fix filed in that state must dispatch on the reserved
slot at the next tick, while the loop picks nothing, and so locks nothing,
whenever every slot is held.  The replay also runs with the dispatch-admission
gate on and the host saturated, where the fix must still take the reserve.

The rig is ``_run_loop_board.py``: a REAL ``Scheduler`` over an in-test task
list, with the slot body stubbed at its one seam, ``_run_slot``.  Here each
slot is a merge-waiter that ends only when the test opens its gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from _run_loop_board import (
    Board,
    build_board_harness,
    fleet_dir_env,
    plain,
    run_driven,
    settle,
    task,
    until,
)
from shared.psi import PsiSample

from orchestrator.config import PsiAdmissionConfig
from orchestrator.run_slots import DispatchScope

_LONG_POLL = 3600.0
_GENERAL_SLOTS = 18
_UNIT = 'orchestrator-reserve-test.service'


def _fix(tid: str) -> dict:
    return task(tid, 'high', signature=f'sig-{tid}')


_IDLE_PSI = PsiSample(cpu_some10=0.0, mem_some10=0.0, mem_full10=0.0, io_some10=0.0, read_ok=True)
_SATURATED_PSI = PsiSample(cpu_some10=0.0, mem_some10=0.0, mem_full10=5.0, io_some10=0.0, read_ok=True)


@dataclass
class _HostPsi:
    """The host PSI reading every dispatch tick sees; a test changes it mid-run."""

    sample: PsiSample = _IDLE_PSI


def _merge_waiters() -> list[dict]:
    return [plain(str(n)) for n in range(1, _GENERAL_SLOTS + 1)]


@pytest.fixture(autouse=True)
def _fleet_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fleet_dir_env(monkeypatch, tmp_path, _UNIT)


async def _saturate(board: Board) -> None:
    await until(lambda: len(board.live) == _GENERAL_SLOTS, 'every general slot holds a merge-waiter')


@pytest.mark.asyncio
class TestMainHealthFixReserve:

    async def test_fix_filed_while_every_general_slot_holds_a_merge_waiter_dispatches_next_tick(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        h, board = build_board_harness(tmp_path, monkeypatch, general_slots=_GENERAL_SLOTS, reserve=1, tasks=_merge_waiters())

        async def drive() -> None:
            await _saturate(board)
            filed = board.file(plain('50', 'critical'), _fix('60'))

            await until(lambda: board.started('60'), 'the fix starts on the reserve')
            assert board.live_at_start['60'] == _GENERAL_SLOTS + 1
            first = board.ticks_since(filed)[0]
            assert (first.scope, first.picked) == (DispatchScope.MAIN_HEALTH_FIX, '60')

            await settle()
            assert not board.started('50')
            assert not board.holds_locks('50')

        await run_driven(h, board, drive)

    async def test_fix_filed_while_host_psi_is_saturated_still_dispatches_on_the_reserve(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        assert _SATURATED_PSI.saturated(PsiAdmissionConfig()), 'premise: the sample trips the gate'
        host = _HostPsi()
        monkeypatch.setattr('orchestrator.scheduler.read_psi_sample', lambda *, project_id=None: host.sample)
        h, board = build_board_harness(
            tmp_path, monkeypatch, general_slots=_GENERAL_SLOTS, reserve=1, tasks=_merge_waiters(), psi_admission=PsiAdmissionConfig(),
        )

        async def drive() -> None:
            await _saturate(board)
            host.sample = _SATURATED_PSI
            filed = board.file(plain('50', 'critical'), _fix('60'))

            await until(lambda: board.started('60'), 'the fix starts on the reserve under a PSI hold')
            assert board.live_at_start['60'] == _GENERAL_SLOTS + 1
            first = board.ticks_since(filed)[0]
            assert (first.scope, first.picked) == (DispatchScope.MAIN_HEALTH_FIX, '60')

            await settle()
            assert not board.started('50')
            assert not board.holds_locks('50')

        await run_driven(h, board, drive)

    async def test_loop_picks_nothing_while_every_slot_is_held(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        h, board = build_board_harness(tmp_path, monkeypatch, general_slots=_GENERAL_SLOTS, reserve=1, tasks=_merge_waiters())

        async def drive() -> None:
            await _saturate(board)
            board.file(_fix('60'))
            await until(lambda: board.started('60'), 'the fix takes the reserve')

            ticks_when_full = len(board.ticks)
            board.file(plain('70'))
            await settle()
            assert len(board.ticks) == ticks_when_full, 'no tick while every slot is held'
            assert not board.holds_locks('70')

            freed = board.mark()
            board.gates['1'].set()
            await until(lambda: board.started('70'), 'the next task starts on the freed slot')
            first = board.ticks_since(freed)[0]
            assert (first.scope, first.picked) == (DispatchScope.ANY, '70')

        await run_driven(h, board, drive)

    async def test_reserve_disabled_keeps_the_old_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        h, board = build_board_harness(tmp_path, monkeypatch, general_slots=_GENERAL_SLOTS, reserve=0, tasks=_merge_waiters())

        async def drive() -> None:
            await _saturate(board)
            board.file(_fix('60'))
            await settle()

            assert not board.started('60')
            assert not board.holds_locks('60')
            assert len(board.ticks) == _GENERAL_SLOTS, 'no tick while every slot is held'

        await run_driven(h, board, drive)

    async def test_a_second_fix_does_not_take_a_second_extra_slot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        h, board = build_board_harness(tmp_path, monkeypatch, general_slots=_GENERAL_SLOTS, reserve=1, tasks=_merge_waiters())

        async def drive() -> None:
            await _saturate(board)
            board.file(_fix('60'), _fix('61'))
            await until(lambda: board.started('60') or board.started('61'), 'one fix starts')
            await settle()

            [first_fix] = [tid for tid in ('60', '61') if board.started(tid)]
            second_fix = '61' if first_fix == '60' else '60'
            assert board.max_live == _GENERAL_SLOTS + 1
            assert not board.holds_locks(second_fix)

            freed = board.mark()
            board.gates[first_fix].set()
            await until(lambda: board.started(second_fix), 'the second fix takes the freed reserve')
            assert board.live_at_start[second_fix] == _GENERAL_SLOTS + 1
            first = board.ticks_since(freed)[0]
            assert (first.scope, first.picked) == (DispatchScope.MAIN_HEALTH_FIX, second_fix)

        await run_driven(h, board, drive)

    async def test_a_general_slot_freed_during_a_restricted_tick_is_refilled_without_waiting_a_poll(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        h, board = build_board_harness(
            tmp_path, monkeypatch, general_slots=_GENERAL_SLOTS, reserve=1, tasks=_merge_waiters(), idle_poll_secs=_LONG_POLL,
        )
        filed: list[int] = []

        async def end_a_merge_waiter_and_file_plain_work() -> None:
            filed.append(board.file(plain('70')))
            board.gates['1'].set()
            await until(lambda: board.started('1') and '1' not in board.live, 'merge-waiter 1 ends mid-tick')
            await settle()

        board.inside_next_restricted_tick(end_a_merge_waiter_and_file_plain_work)

        async def drive() -> None:
            await until(lambda: board.started('70'), 'the freed general slot is refilled before the next poll')
            [epoch] = filed
            first = board.ticks_since(epoch)[0]
            assert (first.scope, first.picked) == (DispatchScope.ANY, '70')

        await run_driven(h, board, drive)
