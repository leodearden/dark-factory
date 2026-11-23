"""The un-chained mark survives an orchestrator restart (task 11047).

A red deep-chain tip marks its owner and every link, so each verifies alone
before it can be batched again. The mark is persisted under the project's data
directory, so a worker built after a restart over the same data directory and
event journal still dispatches every marked request without members. The row
runs a first lane until the red tip has been followed by a solo verify, stops
it, then runs a second lane over the same scene and resubmits the same tasks.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest
from _merge_lane_fakes import healthy_verify_clock, make_lane
from test_merge_queue_deep_reservation import (
    _deep,
    _requests,
    _running,
    _Scene,
    _scene,
    _until,
)
from test_merge_queue_deep_unchained_mark import (
    _chain_items,
    _high,
    _outcome,
    _resubmission,
    _TreeKeyedVerifier,
)

from orchestrator.merge_lane import MergeRequest
from orchestrator.verify import VerifyResult

pytestmark = pytest.mark.asyncio

_MARKED = ('S', 'F1', 'F2', 'F3')


class _SoloVerifiesHang(_TreeKeyedVerifier):
    """A red tip verifies; a verify of one member alone never returns."""

    def __init__(self) -> None:
        super().__init__()
        self.solo_entered = asyncio.Event()

    async def run_scoped(
        self, worktree: Path, config: Any, module_configs: list[Any],
        task_files: list[str] | None = None, **options: Any,
    ) -> VerifyResult:
        files = frozenset(p.name for p in Path(worktree).glob('*.txt'))
        if len(files) == 1:
            self.solo_entered.set()
            await asyncio.Event().wait()
        return await super().run_scoped(worktree, config, module_configs, task_files, **options)


class TestMarksSurviveARestart:
    async def test_marked_requests_dispatch_without_members_after_a_restart(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """S, F1, F2 and F3 ride a red four-item tip, so all four are marked;
        the process is killed while the solo verifies are under way. A lane built
        afterwards, over the same data directory and event journal, is handed
        the same four tasks with G1-G3 chainable behind them. Each marked task
        still dispatches alone."""
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep(6))
        first = {
            'S': await _high(scene, 'S'), 'F1': await _high(scene, 'F1'),
            **await _requests(scene, 'F2', 'F3'),
        }
        before = _SoloVerifiesHang()
        await _killed_while_verifying_alone(scene, before, [first[name] for name in _MARKED])
        assert _chain_items(scene, 'S')[0] == 4, 'the red tip carried all four members'

        verified_before = {name: len(_chain_items(scene, name)) for name in _MARKED}
        await _run_second_process(scene, first)
        for name in _MARKED:
            after = _chain_items(scene, name)[verified_before[name]:]
            assert after and set(after) == {1}, (
                f'{name} was batched again after the restart: {after}'
            )


async def _killed_while_verifying_alone(
    scene: _Scene, verifier: _SoloVerifiesHang, requests: list[MergeRequest],
) -> None:
    """Run a lane until a marked member verifies alone, then kill it.

    The lane is cancelled rather than stopped: a graceful stop resolves every
    pending request and so retires the marks, which a killed process never does.
    """
    queue: asyncio.Queue[Any] = asyncio.Queue()
    lane = make_lane(
        scene.git_ops, queue, verifier=verifier, clock=healthy_verify_clock(),
        speculation_depth=2, remote_runners=[], event_store=scene.events,
    )
    run = asyncio.create_task(lane.run())
    for request in requests:
        queue.put_nowait(request)
    try:
        await _until(verifier.solo_entered.is_set, label='a marked member verifying alone')
    finally:
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run


async def _run_second_process(scene: _Scene, first: dict[str, Any]) -> None:
    resubmitted = {name: _resubmission(scene, first[name]) for name in _MARKED}
    unmarked = await _requests(scene, 'G1', 'G2', 'G3')
    async with _running(
        scene, verifier=_TreeKeyedVerifier(), remotes=[], speculation_depth=2,
    ) as run:
        run.put(*resubmitted.values(), *unmarked.values())
        for name, request in {**resubmitted, **unmarked}.items():
            await _outcome(request, name)
