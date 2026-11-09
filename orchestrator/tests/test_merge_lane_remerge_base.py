"""Where a recovery re-merge goes: the live frontier tip, or main (task 8524).

``choose_remerge_base`` is the pure decision behind every recovery re-merge
(plans/merge-lane-policy-prd.md D1): a replacement re-anchors on the frozen
tip only while that tip can still land, and otherwise falls to main with the
reason recorded.
"""

from __future__ import annotations

import asyncio
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
from orchestrator.merge_lane.remerge_base import choose_remerge_base, frontier_tip
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
    gate: PendingGate | None = None,
) -> InflightEntry:
    item = _real_item(config, task_id, base=base, commit=commit)
    return InflightEntry(
        item=item, lease=None, verify_task=verify, merge_wt=item.merge_wt,
        was_speculative=item.speculative, status=status, gate=gate,
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
        owner = _entry(config, 'a', base=_MAIN, commit='sha-a', verify=verifies.pending())
        assert isinstance(owner.item, RealMergeItem)
        owner.gate = _gate(owner.item, verifies.pending())
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
