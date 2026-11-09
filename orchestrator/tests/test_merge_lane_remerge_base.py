"""Where a recovery re-merge goes: the live frontier tip, or main (task 8524).

``choose_remerge_base`` is the pure decision behind every recovery re-merge
(plans/merge-lane-policy-prd.md D1): a replacement re-anchors on the frozen
tip only while that tip can still land, and otherwise falls to main with the
reason recorded.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Coroutine
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from _merge_lane_fakes import lane_scene_config

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import MergeResult
from orchestrator.merge_lane.dead_base_ledger import (
    DeadBaseKiller,
    DeadBaseLedger,
    DeadBaseSite,
)
from orchestrator.merge_lane.remerge_base import (
    RemergeClaim,
    choose_remerge_base,
    claim_remerge_base,
    frontier_tip,
)
from orchestrator.merge_lane.speculation_controller import PermitLedger
from orchestrator.merge_lane.types import (
    DecidedItem,
    InflightEntry,
    InflightStatus,
    InflightVerifyResult,
    LandingCandidate,
    MergeOutcome,
    MergeRequest,
    PendingGate,
    QueuedBranch,
    RealMergeItem,
    RemergeBase,
    RemergeReason,
    SpecPermit,
)

_MAIN = 'sha-main'


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    return lane_scene_config(tmp_path, GitConfig())


class Verifies:
    """Builds verify tasks in each state a frontier owner's verify can be in,
    and cancels whichever are still pending when the test ends."""

    def __init__(self) -> None:
        self._made: list[asyncio.Task[Any]] = []

    def pending(self) -> asyncio.Task[Any]:
        task = asyncio.create_task(asyncio.Event().wait())
        self._made.append(task)
        return task

    async def settled(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        await asyncio.wait([task])
        return task

    async def finished(self, result: InflightVerifyResult) -> asyncio.Task[Any]:
        return await self.settled(_returns(result))

    async def raised(self) -> asyncio.Task[Any]:
        return await self.settled(_raises())

    async def cancelled(self) -> asyncio.Task[Any]:
        task = self.pending()
        task.cancel()
        await asyncio.wait([task])
        return task

    async def close(self) -> None:
        for task in self._made:
            task.cancel()
        if self._made:
            await asyncio.wait(self._made)


async def _returns(result: InflightVerifyResult) -> InflightVerifyResult:
    return result


async def _raises() -> InflightVerifyResult:
    raise RuntimeError('verify blew up')


@pytest_asyncio.fixture
async def verifies() -> AsyncIterator[Verifies]:
    made = Verifies()
    yield made
    await made.close()


def _request(config: OrchestratorConfig, task_id: str) -> MergeRequest:
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(f'task/{task_id}', config.git.branch_prefix),
        worktree=config.project_root,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
    )


def _real_item(
    config: OrchestratorConfig, task_id: str, *, base: str, commit: str | None,
) -> RealMergeItem:
    return RealMergeItem(
        request=_request(config, task_id),
        merge_result=MergeResult(success=True, merge_commit=commit),
        merge_wt=Path('/fake/merge-wt'),
        base_sha=base,
        speculative=base != _MAIN,
    )


def _entry(
    config: OrchestratorConfig,
    task_id: str,
    *,
    base: str,
    commit: str | None,
    verify: asyncio.Task[Any] | None,
    status: InflightStatus | None = None,
    gate_task: asyncio.Task[Any] | None = None,
) -> InflightEntry:
    item = _real_item(config, task_id, base=base, commit=commit)
    return InflightEntry(
        item=item, lease=None, verify_task=verify, merge_wt=item.merge_wt,
        was_speculative=item.speculative, status=status,
        gate=None if gate_task is None else _gate(item, gate_task),
    )


def _passthrough(config: OrchestratorConfig, task_id: str) -> InflightEntry:
    outcome = MergeOutcome(status='conflict')
    item = DecidedItem(
        request=_request(config, task_id), immediate_outcome=outcome,
        base_sha=_MAIN, speculative=False,
    )
    return InflightEntry(
        item=item, lease=None, verify_task=None, merge_wt=None,
        was_speculative=False, passthrough_outcome=outcome,
    )


def _gate(item: RealMergeItem, task: asyncio.Task[Any]) -> PendingGate:
    commit = item.merge_result.merge_commit or ''
    return PendingGate(
        task=task,
        candidate=LandingCandidate(
            item=item, sha=commit, merge_wt=item.merge_wt, spec_warm=False,
            warm_results={}, verified_commit=commit,
        ),
        rebased_sha='sha-rebased',
        rebased_onto='sha-main-2',
    )


def _choose(
    frozen_tip: str,
    frontier: list[InflightEntry],
    *,
    dead: frozenset[str] | DeadBaseLedger = frozenset(),
    permit_available: bool = True,
) -> RemergeBase:
    return choose_remerge_base(
        _MAIN, frozen_tip, dead, frontier, permit_available=permit_available,
    )


class TestRemergeBaseInvariant:
    """A tip is set exactly when the decision is LIVE_TIP."""

    def test_live_tip_without_a_tip_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r'live_tip.*None|None.*live_tip'):
            RemergeBase(tip=None, reason=RemergeReason.LIVE_TIP)

    def test_a_tip_on_a_fall_to_main_reason_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"no_tip.*'t'|'t'.*no_tip"):
            RemergeBase(tip='t', reason=RemergeReason.NO_TIP)

    def test_speculative_iff_a_tip_is_set(self) -> None:
        assert RemergeBase(tip='t', reason=RemergeReason.LIVE_TIP).speculative is True
        assert RemergeBase(tip=None, reason=RemergeReason.TIP_DEAD).speculative is False

    def test_wire_names_the_reason_and_the_anchor(self) -> None:
        assert RemergeBase(tip='t', reason=RemergeReason.LIVE_TIP).to_wire() == {
            'anchor_reason': 'live_tip', 'anchor_base': 't',
        }
        assert RemergeBase(tip=None, reason=RemergeReason.NO_PERMIT).to_wire() == {
            'anchor_reason': 'no_permit', 'anchor_base': None,
        }


@pytest.mark.asyncio
class TestFrontierTip:
    """The tip is the newest real merge commit in the frontier."""

    async def test_newest_real_commit(
        self, config: OrchestratorConfig, verifies: Verifies,
    ) -> None:
        frontier = [
            _entry(config, 'a', base=_MAIN, commit='sha-a', verify=verifies.pending()),
            _entry(config, 'b', base='sha-a', commit='sha-b', verify=verifies.pending()),
        ]
        assert frontier_tip(frontier) == 'sha-b'

    async def test_passthroughs_and_commitless_merges_are_skipped(
        self, config: OrchestratorConfig, verifies: Verifies,
    ) -> None:
        frontier = [
            _entry(config, 'a', base=_MAIN, commit='sha-a', verify=verifies.pending()),
            _entry(config, 'b', base='sha-a', commit=None, verify=verifies.pending()),
            _passthrough(config, 'c'),
        ]
        assert frontier_tip(frontier) == 'sha-a'

    async def test_empty_frontier_has_no_tip(self) -> None:
        assert frontier_tip([]) is None


@pytest.mark.asyncio
class TestChooseRemergeBase:
    """The reason table, checked in the order the decision applies it."""

    async def test_tip_equal_to_main_is_no_tip(self) -> None:
        assert _choose(_MAIN, []) == RemergeBase(None, RemergeReason.NO_TIP)

    async def test_blank_tip_is_no_tip(self) -> None:
        assert _choose('', []) == RemergeBase(None, RemergeReason.NO_TIP)

    async def test_dead_tip(self, config: OrchestratorConfig, verifies: Verifies) -> None:
        frontier = [_entry(config, 'a', base=_MAIN, commit='sha-a', verify=verifies.pending())]
        chosen = _choose('sha-a', frontier, dead=frozenset({'sha-a'}))
        assert chosen == RemergeBase(None, RemergeReason.TIP_DEAD)

    async def test_dead_earlier_commit_breaks_the_tip_chain(
        self, config: OrchestratorConfig, verifies: Verifies,
    ) -> None:
        """The circularity case: the tip is stacked on the item being re-merged."""
        frontier = [
            _entry(config, 'x', base=_MAIN, commit='sha-x', verify=verifies.pending()),
            _entry(config, 'y', base='sha-x', commit='sha-y', verify=verifies.pending()),
        ]
        chosen = _choose('sha-y', frontier, dead=frozenset({'sha-x'}))
        assert chosen == RemergeBase(None, RemergeReason.TIP_DEAD)

    async def test_dead_earlier_base_breaks_the_tip_chain(
        self, config: OrchestratorConfig, verifies: Verifies,
    ) -> None:
        frontier = [
            _entry(config, 'x', base='sha-gone', commit='sha-x', verify=verifies.pending()),
            _entry(config, 'y', base='sha-x', commit='sha-y', verify=verifies.pending()),
        ]
        chosen = _choose('sha-y', frontier, dead=frozenset({'sha-gone'}))
        assert chosen == RemergeBase(None, RemergeReason.TIP_DEAD)

    async def test_a_base_on_main_is_never_dead(
        self, config: OrchestratorConfig, verifies: Verifies,
    ) -> None:
        frontier = [_entry(config, 'a', base=_MAIN, commit='sha-a', verify=verifies.pending())]
        chosen = _choose('sha-a', frontier, dead=frozenset({_MAIN}))
        assert chosen == RemergeBase('sha-a', RemergeReason.LIVE_TIP)

    @pytest.mark.parametrize('status', [InflightStatus.REQUEUED, InflightStatus.DROPPED])
    async def test_vacated_owner(
        self, config: OrchestratorConfig, verifies: Verifies, status: InflightStatus,
    ) -> None:
        frontier = [
            _entry(config, 'a', base=_MAIN, commit='sha-a', verify=verifies.pending(), status=status),
        ]
        assert _choose('sha-a', frontier) == RemergeBase(None, RemergeReason.TIP_VACATED)

    async def test_a_vacated_link_below_the_tip_vacates_it(
        self, config: OrchestratorConfig, verifies: Verifies,
    ) -> None:
        frontier = [
            _entry(
                config, 'v', base=_MAIN, commit='sha-v', verify=verifies.pending(),
                status=InflightStatus.DROPPED,
            ),
            _entry(config, 'w', base='sha-v', commit='sha-w', verify=verifies.pending()),
        ]
        assert _choose('sha-w', frontier) == RemergeBase(None, RemergeReason.TIP_VACATED)

    async def test_owner_without_a_verify(self, config: OrchestratorConfig) -> None:
        frontier = [_entry(config, 'a', base=_MAIN, commit='sha-a', verify=None)]
        assert _choose('sha-a', frontier) == RemergeBase(None, RemergeReason.TIP_VERIFY_DEAD)

    async def test_owner_verify_cancelled(
        self, config: OrchestratorConfig, verifies: Verifies,
    ) -> None:
        frontier = [_entry(config, 'a', base=_MAIN, commit='sha-a', verify=await verifies.cancelled())]
        assert _choose('sha-a', frontier) == RemergeBase(None, RemergeReason.TIP_VERIFY_DEAD)

    async def test_owner_verify_raised(
        self, config: OrchestratorConfig, verifies: Verifies,
    ) -> None:
        frontier = [_entry(config, 'a', base=_MAIN, commit='sha-a', verify=await verifies.raised())]
        assert _choose('sha-a', frontier) == RemergeBase(None, RemergeReason.TIP_VERIFY_DEAD)

    @pytest.mark.parametrize('result', [
        InflightVerifyResult(outcome=MergeOutcome(status='blocked'), merge_wt=None),
        InflightVerifyResult(outcome=None, merge_wt=None, status=InflightStatus.RUNNER_UNAVAILABLE),
        InflightVerifyResult(outcome=None, merge_wt=None, status=InflightStatus.DROPPED),
    ], ids=['failed', 'runner-unavailable', 'dropped'])
    async def test_owner_verify_finished_badly(
        self, config: OrchestratorConfig, verifies: Verifies, result: InflightVerifyResult,
    ) -> None:
        frontier = [_entry(config, 'a', base=_MAIN, commit='sha-a', verify=await verifies.finished(result))]
        assert _choose('sha-a', frontier) == RemergeBase(None, RemergeReason.TIP_VERIFY_DEAD)

    async def test_owner_parked_on_a_gate(
        self, config: OrchestratorConfig, verifies: Verifies,
    ) -> None:
        owner = _entry(
            config, 'a', base=_MAIN, commit='sha-a', verify=verifies.pending(),
            gate_task=verifies.pending(),
        )
        assert _choose('sha-a', [owner]) == RemergeBase(None, RemergeReason.TIP_VERIFY_DEAD)

    async def test_tip_with_no_owner(
        self, config: OrchestratorConfig, verifies: Verifies,
    ) -> None:
        frontier = [_entry(config, 'a', base=_MAIN, commit='sha-a', verify=verifies.pending())]
        assert _choose('sha-elsewhere', frontier) == RemergeBase(None, RemergeReason.TIP_VERIFY_DEAD)

    async def test_live_tip_without_a_permit(
        self, config: OrchestratorConfig, verifies: Verifies,
    ) -> None:
        frontier = [_entry(config, 'a', base=_MAIN, commit='sha-a', verify=verifies.pending())]
        chosen = _choose('sha-a', frontier, permit_available=False)
        assert chosen == RemergeBase(None, RemergeReason.NO_PERMIT)

    async def test_live_tip_with_a_pending_verify(
        self, config: OrchestratorConfig, verifies: Verifies,
    ) -> None:
        frontier = [
            _entry(config, 'a', base=_MAIN, commit='sha-a', verify=verifies.pending()),
            _entry(config, 'b', base='sha-a', commit='sha-b', verify=verifies.pending()),
        ]
        assert _choose('sha-b', frontier) == RemergeBase('sha-b', RemergeReason.LIVE_TIP)

    async def test_live_tip_with_a_passed_verify(
        self, config: OrchestratorConfig, verifies: Verifies,
    ) -> None:
        passed = await verifies.finished(InflightVerifyResult(outcome=None, merge_wt=None))
        frontier = [_entry(config, 'a', base=_MAIN, commit='sha-a', verify=passed)]
        assert _choose('sha-a', frontier) == RemergeBase('sha-a', RemergeReason.LIVE_TIP)

    async def test_dead_bases_may_be_the_ledger(
        self, config: OrchestratorConfig, verifies: Verifies,
    ) -> None:
        ledger = DeadBaseLedger(cap=8)
        frontier = [
            _entry(config, 'x', base=_MAIN, commit='sha-x', verify=verifies.pending()),
            _entry(config, 'y', base='sha-x', commit='sha-y', verify=verifies.pending()),
        ]
        assert _choose('sha-y', frontier, dead=ledger) == RemergeBase('sha-y', RemergeReason.LIVE_TIP)
        ledger.record('sha-x', DeadBaseKiller(DeadBaseSite.ADOPTION_VOID, 'x'))
        assert _choose('sha-y', frontier, dead=ledger) == RemergeBase(None, RemergeReason.TIP_DEAD)


def _ledger(depth: int) -> PermitLedger[SpecPermit]:
    return PermitLedger(asyncio.Semaphore(depth), depth)


def _claim(
    ledger: PermitLedger[SpecPermit],
    frontier: list[InflightEntry],
    *,
    held: SpecPermit | None,
    dead: frozenset[str] = frozenset(),
) -> RemergeClaim:
    return claim_remerge_base(
        main_sha=_MAIN, frozen_tip=frontier_tip(frontier) or _MAIN, dead_bases=dead,
        frontier=frontier, held_permit=held, ledger=ledger,
    )


class TestRemergeClaimInvariant:
    """A claim holds a permit exactly when its anchor is speculative."""

    def test_a_speculative_anchor_needs_a_permit(self) -> None:
        with pytest.raises(ValueError, match=r'live_tip.*None'):
            RemergeClaim(RemergeBase('t', RemergeReason.LIVE_TIP), None)

    def test_a_fall_to_main_holds_no_permit(self) -> None:
        with pytest.raises(ValueError, match=r'no_tip.*SpecPermit'):
            RemergeClaim(RemergeBase(None, RemergeReason.NO_TIP), SpecPermit())


@pytest.mark.asyncio
class TestClaimRemergeBase:
    """The permit a replacement leaves with, and what the ledger sees."""

    async def test_a_held_permit_rides_a_live_tip(
        self, config: OrchestratorConfig, verifies: Verifies,
    ) -> None:
        ledger = _ledger(2)
        held = await ledger.acquire()
        frontier = [_entry(config, 'a', base=_MAIN, commit='sha-a', verify=verifies.pending())]

        claim = _claim(ledger, frontier, held=held)

        assert claim == RemergeClaim(RemergeBase('sha-a', RemergeReason.LIVE_TIP), held)
        assert ledger.live == frozenset({held})
        assert ledger.slot_available == 1

    async def test_a_held_permit_is_released_on_a_fall_to_main(self) -> None:
        ledger = _ledger(2)
        held = await ledger.acquire()

        claim = _claim(ledger, [], held=held)

        assert claim == RemergeClaim(RemergeBase(None, RemergeReason.NO_TIP), None)
        assert held.released is True
        assert ledger.live == frozenset()
        assert ledger.slot_available == 2

    async def test_a_live_tip_takes_a_free_slot(
        self, config: OrchestratorConfig, verifies: Verifies,
    ) -> None:
        ledger = _ledger(2)
        frontier = [_entry(config, 'a', base=_MAIN, commit='sha-a', verify=verifies.pending())]

        claim = _claim(ledger, frontier, held=None)

        assert claim.anchor == RemergeBase('sha-a', RemergeReason.LIVE_TIP)
        assert claim.permit is not None
        assert ledger.live == frozenset({claim.permit})
        assert ledger.slot_available == 1

    async def test_a_live_tip_with_no_free_slot_falls_to_main(
        self, config: OrchestratorConfig, verifies: Verifies,
    ) -> None:
        ledger = _ledger(1)
        merger_permit = await ledger.acquire()
        frontier = [_entry(config, 'a', base=_MAIN, commit='sha-a', verify=verifies.pending())]

        claim = _claim(ledger, frontier, held=None)

        assert claim == RemergeClaim(RemergeBase(None, RemergeReason.NO_PERMIT), None)
        assert ledger.live == frozenset({merger_permit})
        assert ledger.slot_available == 0


_OWNER_KINDS = ('live', 'vacated', 'verify_dead', 'gate_parked', 'dead_chained')


def _random_frontier(
    rng: random.Random, config: OrchestratorConfig, verifies: Verifies, cascade: int,
) -> tuple[list[InflightEntry], frozenset[str]]:
    frontier: list[InflightEntry] = []
    dead: set[str] = set()
    base = _MAIN
    for i in range(rng.randint(0, 3)):
        kind = rng.choice(_OWNER_KINDS)
        commit = f'sha-{cascade}-{i}'
        frontier.append(_entry(
            config, f't{cascade}-{i}', base=base, commit=commit,
            verify=None if kind == 'verify_dead' else verifies.pending(),
            status=InflightStatus.REQUEUED if kind == 'vacated' else None,
            gate_task=verifies.pending() if kind == 'gate_parked' else None,
        ))
        if kind == 'dead_chained':
            dead.add(commit)
        base = commit
    return frontier, frozenset(dead)


def _assert_accounted(ledger: PermitLedger[SpecPermit], outstanding: list[RemergeClaim]) -> None:
    assert ledger.slot_available + len(ledger.live) == ledger.depth
    for claim in outstanding:
        if claim.anchor.speculative:
            assert claim.permit in ledger.live
        else:
            assert claim.permit is None


@pytest.mark.asyncio
async def test_permits_are_conserved_across_random_cascades(
    config: OrchestratorConfig, verifies: Verifies,
) -> None:
    """Boundary row 4 (plans/merge-lane-policy-prd.md §7): 100 seeded cascades.

    Other holders (the merger, other in-flight items) occupy a random share
    of the slots, so a live tip sometimes finds none free.
    """
    rng = random.Random(8524)
    seen: set[RemergeReason] = set()
    for cascade in range(100):
        depth = rng.randint(2, 4)
        ledger = _ledger(depth)
        others = [ledger.try_acquire() for _ in range(rng.randint(0, depth))]
        frontier, dead = _random_frontier(rng, config, verifies, cascade)
        outstanding: list[RemergeClaim] = []
        for _ in range(rng.randint(1, depth)):
            held = ledger.try_acquire() if rng.random() < 0.5 else None
            claim = _claim(ledger, frontier, held=held, dead=dead)
            seen.add(claim.anchor.reason)
            outstanding.append(claim)
            _assert_accounted(ledger, outstanding)
        rng.shuffle(outstanding)
        while outstanding:
            permit = outstanding.pop().permit
            if permit is not None:
                ledger.release(permit)
            _assert_accounted(ledger, outstanding)
        for permit in others:
            assert permit is not None
            ledger.release(permit)
        assert ledger.live == frozenset()
    assert seen == set(RemergeReason)
