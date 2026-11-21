"""A member of a red deep chain verifies un-chained until its verdict is applied (task 10723).

The mark is set by a red tip (owner and every link) or a chain-arm error
(owner). It lasts until a verdict for the task is applied: the request lands,
is blocked or is dropped. A verify whose result the head-failure cascade voids
is no applied verdict, so the re-dispatch must still verify un-chained; the
sequence measured on task 5715 is the regression these rows pin.

Live merger and verifier loops over a real tmp git repo, observed only through
the lane's public surface: request futures, ``snapshot()``, the event store
and the fakes' ledgers. Verdicts are keyed on the verified tree's content, so
a red is caused by a file and not by a call's position.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _merge_lane_fakes import FakeRemoteRunner, FakeVerifier, fails, passes
from _orch_helpers import MERGE_GATE_BARRIER_TIMEOUT, wait_responsive
from test_merge_queue_deep_reservation import (
    _deep,
    _lane_time_passes,
    _request,
    _requests,
    _reservations,
    _running,
    _Scene,
    _scene,
    _until,
    _verify_events_for,
)

from orchestrator.event_store import EventType
from orchestrator.git_ops import _run
from orchestrator.merge_lane import MergeLane, MergeOutcome, MergeRequest
from orchestrator.verify import VerifyResult

pytestmark = pytest.mark.asyncio

_RED_FILES = frozenset({'S.txt', 'F1.txt'})
"""S and F1 are genuinely red: any tree holding either fails its verify."""

TreeTest = Callable[[frozenset[str]], bool]
Chains = list[tuple[str, list[str]]]


def _verdict(files: frozenset[str]) -> VerifyResult:
    if files & _RED_FILES:
        return fails(category='test', summary=f'red: {sorted(files & _RED_FILES)}').result
    return passes().result


def _never(_files: frozenset[str]) -> bool:
    return False


def _chains(lane: MergeLane) -> Chains:
    """The live chain reservations, as ``(owner, links)``."""
    return [(r['owner_task_id'], r['link_task_ids']) for r in _reservations(lane)]


class _TreeKeyedVerifier(FakeVerifier):
    """The local host: red iff the worktree holds a red file.

    *raising* selects the trees whose verify raises instead; a test may swap
    it between rounds. Each call records its task, its tree and, once a test
    sets ``lane``, the chains reserved while it ran.
    """

    def __init__(self, *, raising: TreeTest = _never) -> None:
        super().__init__()
        self.raising = raising
        self.lane: MergeLane | None = None
        self.calls: list[tuple[str | None, frozenset[str], Chains]] = []

    async def run_scoped(
        self, worktree: Path, config: Any, module_configs: list[Any],
        task_files: list[str] | None = None, **options: Any,
    ) -> VerifyResult:
        task_id = options.get('task_id')
        self._note_entry(task_id)
        files = frozenset(p.name for p in Path(worktree).glob('*.txt'))
        self.calls.append((task_id, files, [] if self.lane is None else _chains(self.lane)))
        if self.raising(files):
            raise RuntimeError(f'verify infra exploded on {sorted(files)}')
        return _verdict(files)


class _TreeKeyedRemote(FakeRemoteRunner):
    """A remote host: red iff the merge commit's tree holds a red file.

    A verify whose tree *hold* selects waits for *release* first.
    """

    def __init__(
        self, name: str, repo: Path, *, hold: TreeTest, release: asyncio.Event,
        entered: asyncio.Event,
    ) -> None:
        super().__init__(name, release=release, entered=entered)
        self.repo = repo
        self.hold = hold

    async def run_merge_verify(self, merge_sha: str, spec: Any) -> VerifyResult:
        self.calls.append(merge_sha)
        _, listing, _ = await _run(['git', 'ls-tree', '--name-only', merge_sha], cwd=self.repo)
        files = frozenset(name for name in listing.split() if name.endswith('.txt'))
        if self.hold(files) and self.release is not None:
            self.live = True
            try:
                if self.entered is not None:
                    self.entered.set()
                await self.release.wait()
            finally:
                self.live = False
        return _verdict(files)


async def _high(scene: _Scene, name: str) -> MergeRequest:
    return dataclasses.replace(await _request(scene, name), lane='high')


def _resubmission(scene: _Scene, request: MergeRequest) -> MergeRequest:
    """A fresh request for *request*'s task and branch, as a resubmission makes."""
    return MergeRequest(
        task_id=request.task_id, branch=request.branch, worktree=request.worktree,
        pre_rebased=False, task_files=None, module_configs=[], config=scene.config,
        result=asyncio.get_running_loop().create_future(), lane='normal',
    )


def _chain_items(scene: _Scene, task_id: str) -> list[int]:
    """The ``chain_items`` of every verify *task_id* owned, in order."""
    return [e['data']['chain_items'] for e in _verify_events_for(scene, {task_id})]


def _cancelled_verifies(scene: _Scene, task_id: str) -> list[dict[str, Any]]:
    return [
        e for e in scene.events.fetch_events_by_type(EventType.merge_verify_cancelled)
        if e.get('task_id') == task_id
    ]


async def _outcome(request: MergeRequest, name: str) -> MergeOutcome:
    return await wait_responsive(request.result, label=f'{name} outcome')


def _f1_alone(files: frozenset[str]) -> bool:
    return 'F1.txt' in files and 'S.txt' not in files


class TestAVoidedVerifyKeepsTheMark:
    """Task 5715's sequence: verified alone, voided by a red entry ahead, re-dispatched."""

    async def test_a_voided_red_is_followed_by_an_unchained_verify_until_the_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """S owns a red four-item tip over F1, F2 and F3, so all four are
        marked and the halving state drops to 2. S and F1 ride the high lane,
        so F1 is the entry ahead of S: it verifies alone on the remote and is
        held, while S verifies alone behind it on local, red. F2 then waits
        for a host and G1-G3 wait in the lane. F1 goes red and the cascade
        voids S's result. S's re-dispatch, alone in the frontier with G1-G3
        chainable behind it, must still build no chain; S then blocks on its
        own verify, and that applied verdict lets a resubmission chain."""
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep(6))
        reqs = {
            'S': await _high(scene, 'S'), 'F1': await _high(scene, 'F1'),
            **await _requests(scene, 'F2', 'F3', 'G1', 'G2', 'G3'),
        }
        f1_entered, release_f1 = asyncio.Event(), asyncio.Event()
        remote = _TreeKeyedRemote(
            'ahead-host', scene.git_ops.project_root,
            hold=_f1_alone, release=release_f1, entered=f1_entered,
        )
        async with _running(
            scene, verifier=_TreeKeyedVerifier(), remotes=[remote], speculation_depth=2,
        ) as run:
            run.put(*(reqs[name] for name in ('S', 'F1', 'F2', 'F3')))
            await wait_responsive(
                f1_entered.wait(), timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='F1 verifying alone on the remote after the red tip',
            )
            await _until(
                lambda: len(_verify_events_for(scene, {'S'})) == 2,
                label='S verified alone behind F1',
            )
            assert _chain_items(scene, 'S') == [4, 1], 'the red tip, then S alone'
            run.put(reqs['G1'], reqs['G2'], reqs['G3'])
            await _until(
                lambda: run.lane.unfrozen_suffix() == tuple(
                    reqs[name].request_id for name in ('F3', 'G1', 'G2', 'G3')
                ),
                label='F2 waiting for a host, F3 and G1-G3 in the lane',
            )
            await _lane_time_passes(run, 60)

            release_f1.set()
            assert (await _outcome(reqs['F1'], 'F1')).status == 'blocked'
            s_outcome = await _outcome(reqs['S'], 'S')
            assert s_outcome.status == 'blocked', s_outcome
            assert _cancelled_verifies(scene, 'S') == [], (
                "precondition: S's alone verify returned before the cascade "
                'voided it, rather than being cancelled mid-verify'
            )
            assert _chain_items(scene, 'S') == [4, 1, 1], (
                'S re-dispatched after the void must verify un-chained, though '
                'G1-G3 were chainable behind it'
            )
            for name in ('F2', 'F3', 'G1', 'G2', 'G3'):
                assert (await _outcome(reqs[name], name)).status == 'done', name

            again, h1 = _resubmission(scene, reqs['S']), await _request(scene, 'H1')
            run.put(again, h1)
            assert (await _outcome(again, 'resubmitted S')).status == 'blocked'
            assert (await _outcome(h1, 'H1')).status == 'done'
        assert _chain_items(scene, 'S')[3] == 2, (
            "S's blocked verdict cleared the mark, so its resubmission chains H1"
        )


class _SecondSVerifyHangs(_TreeKeyedVerifier):
    """S's second verify hangs until it is cancelled; every other call is tree-keyed."""

    def __init__(self) -> None:
        super().__init__()
        self.hanging, self.torn_down = asyncio.Event(), asyncio.Event()
        self._s_calls = 0

    async def run_scoped(
        self, worktree: Path, config: Any, module_configs: list[Any],
        task_files: list[str] | None = None, **options: Any,
    ) -> VerifyResult:
        if options.get('task_id') == 'S':
            self._s_calls += 1
            if self._s_calls == 2:
                self.hanging.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.torn_down.set()
        return await super().run_scoped(worktree, config, module_configs, task_files, **options)


class TestADroppedRequestTakesItsMarkWithIt:
    """A drop applies no verdict but ends the request, so the mark leaves with it."""

    async def test_a_marked_request_cancelled_mid_verify_lets_its_resubmission_chain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """S owns a red tip over B1, so both are marked. S's un-chained verify
        is cancelled by its waiter and the request is dropped; S's
        resubmission is a new request and chains H1."""
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep())
        reqs = await _requests(scene, 'S', 'B1')
        verifier = _SecondSVerifyHangs()
        async with _running(scene, verifier=verifier, remotes=[], speculation_depth=1) as run:
            verifier.lane = run.lane
            run.put(reqs['S'], reqs['B1'])
            await wait_responsive(verifier.hanging.wait(), label="S's un-chained verify hanging")
            reqs['S'].result.cancel()
            await wait_responsive(verifier.torn_down.wait(), label="S's verify torn down")
            assert (await _outcome(reqs['B1'], 'B1')).status == 'done'

            again, h1 = _resubmission(scene, reqs['S']), await _request(scene, 'H1')
            run.put(again, h1)
            assert (await _outcome(again, 'resubmitted S')).status == 'blocked'
            assert (await _outcome(h1, 'H1')).status == 'done'
        s_chains = [chains for task_id, _files, chains in verifier.calls if task_id == 'S']
        assert s_chains[0] == [('S', ['B1'])], 'the red tip that marked S'
        assert s_chains[1] == [('S', ['H1'])], (
            "dropping S's request cleared the mark, so its resubmission chains H1"
        )


class TestBAChainArmErrorStillTerminates:
    """Task 3185's termination property: one deferred round, then a terminal block."""

    async def test_a_raised_tip_then_a_raised_alone_verify_blocks_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """S's tip over B1 raises, which marks S and requeues it. B1 lands on
        its own round. S's alone verify raises too, and since it was not
        chained that error is S's own: S blocks, after exactly two verifies.
        The applied block clears the mark, so a resubmission chains H1."""
        scene = _scene(tmp_path, monkeypatch, merge_deep=_deep())
        reqs = await _requests(scene, 'S', 'B1')
        verifier = _TreeKeyedVerifier(raising=lambda files: 'S.txt' in files)
        async with _running(scene, verifier=verifier, remotes=[], speculation_depth=2) as run:
            verifier.lane = run.lane
            run.put(reqs['S'], reqs['B1'])
            outcome = await _outcome(reqs['S'], 'S')
            assert outcome.status == 'blocked', outcome
            assert outcome.reason is not None
            assert outcome.reason.startswith('Verification error:'), outcome.reason
            assert (await _outcome(reqs['B1'], 'B1')).status == 'done'
            s_chains = [chains for task_id, _files, chains in verifier.calls if task_id == 'S']
            assert s_chains == [[('S', ['B1'])], []], (
                'one chained round that raised, then one un-chained round that blocks'
            )

            again, h1 = _resubmission(scene, reqs['S']), await _request(scene, 'H1')
            verifier.raising = _never
            run.put(again, h1)
            assert (await _outcome(again, 'resubmitted S')).status == 'blocked'
        s_chains = [chains for task_id, _files, chains in verifier.calls if task_id == 'S']
        assert s_chains[2] == [('S', ['H1'])], (
            "S's blocked verdict cleared the mark, so its resubmission chains H1"
        )
