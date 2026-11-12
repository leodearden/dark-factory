"""The gate-rebased head as an ordinary dispatch item (task 8525).

A head whose gate re-verify became a dispatch is rebuilt as an item carrying
its rebased commit, and the lane recognises work re-anchored on that commit
(plans/merge-lane-policy-prd.md D3).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from _merge_lane_fakes import lane_scene_config

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import MergeResult
from orchestrator.merge_lane.dead_base_ledger import DeadBaseKiller, DeadBaseSite
from orchestrator.merge_lane.rebased_head import (
    is_gate_reanchor,
    rebased_head_commits,
    rebased_head_item,
)
from orchestrator.merge_lane.types import (
    CapPermit,
    DecidedItem,
    InflightEntry,
    LandingCandidate,
    MergeOutcome,
    MergeRequest,
    QueuedBranch,
    RealMergeItem,
    RemergeBase,
    RemergeReason,
    SpecPermit,
)

_MAIN = 'sha-main'
_REBASED = 'sha-a-rebased'
_ONTO = 'sha-main-2'
_NEW_WT = Path('/fake/rebased-head-wt')
_OLD_WT = Path('/fake/merge-wt')


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    return lane_scene_config(tmp_path, GitConfig())


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


def _speculative_item(config: OrchestratorConfig, task_id: str, commit: str) -> RealMergeItem:
    return RealMergeItem(
        request=_request(config, task_id),
        merge_result=MergeResult(
            success=True, details='merged', merge_commit=commit,
            pre_merge_sha='sha-tip', merge_worktree=_OLD_WT,
        ),
        merge_wt=_OLD_WT,
        base_sha='sha-tip',
        speculative=True,
        started_monotonic=12.5,
        merged_branch_tip='sha-branch-tip',
        cap_permit=CapPermit(),
        permit=SpecPermit(),
        remerge_base=RemergeBase(tip='sha-tip', reason=RemergeReason.LIVE_TIP),
    )


def _candidate(item: RealMergeItem, sha: str) -> LandingCandidate:
    return LandingCandidate(
        item=item, sha=sha, merge_wt=item.merge_wt, spec_warm=True,
        warm_results={}, verified_commit=item.merge_result.merge_commit or '',
    )


def _entry(item: RealMergeItem | DecidedItem) -> InflightEntry:
    if isinstance(item, DecidedItem):
        return InflightEntry(
            item=item, lease=None, verify_task=None, merge_wt=None,
            was_speculative=False, passthrough_outcome=item.immediate_outcome,
        )
    return InflightEntry(
        item=item, lease=None, verify_task=None, merge_wt=item.merge_wt,
        was_speculative=item.speculative,
    )


@pytest.mark.asyncio
class TestRebasedHeadItem:
    async def test_the_head_becomes_a_main_based_item_on_its_rebased_commit(
        self, config: OrchestratorConfig,
    ) -> None:
        item = _speculative_item(config, 'a', 'sha-a')
        cap_permit = CapPermit()

        head = rebased_head_item(
            _candidate(item, 'sha-a'),
            rebased_sha=_REBASED, rebased_onto=_ONTO, merge_wt=_NEW_WT,
            cap_permit=cap_permit,
        )

        assert isinstance(head, RealMergeItem)
        assert (head.base_sha, head.merge_wt, head.speculative) == (_ONTO, _NEW_WT, False)
        assert head.merge_result == MergeResult(
            success=True, details='merged', merge_commit=_REBASED,
            pre_merge_sha='sha-tip', merge_worktree=_NEW_WT,
        )
        assert head.permit is None
        assert head.cap_permit is cap_permit
        assert head.remerge_base is None
        assert head.gate_rebase_of == 'sha-a'
        assert head.request is item.request
        assert (head.merged_branch_tip, head.started_monotonic) == ('sha-branch-tip', 12.5)

    async def test_no_cap_permit_is_carried_as_none(self, config: OrchestratorConfig) -> None:
        item = _speculative_item(config, 'a', 'sha-a')

        head = rebased_head_item(
            _candidate(item, 'sha-a'),
            rebased_sha=_REBASED, rebased_onto=_ONTO, merge_wt=_NEW_WT, cap_permit=None,
        )

        assert head.cap_permit is None

    async def test_the_superseded_commit_is_the_candidates_not_the_items_merge(
        self, config: OrchestratorConfig,
    ) -> None:
        item = _speculative_item(config, 'a', 'sha-a')

        head = rebased_head_item(
            _candidate(item, 'sha-a-skip-cleared'),
            rebased_sha=_REBASED, rebased_onto=_ONTO, merge_wt=_NEW_WT, cap_permit=None,
        )

        assert head.gate_rebase_of == 'sha-a-skip-cleared'

    async def test_the_candidate_item_is_left_untouched(
        self, config: OrchestratorConfig,
    ) -> None:
        item = _speculative_item(config, 'a', 'sha-a')

        rebased_head_item(
            _candidate(item, 'sha-a'),
            rebased_sha=_REBASED, rebased_onto=_ONTO, merge_wt=_NEW_WT, cap_permit=None,
        )

        assert (item.merge_result.merge_commit, item.base_sha, item.gate_rebase_of) == (
            'sha-a', 'sha-tip', None,
        )


@pytest.mark.asyncio
class TestRebasedHeadCommits:
    async def test_only_gate_rebased_entries_contribute_their_commit(
        self, config: OrchestratorConfig,
    ) -> None:
        item = _speculative_item(config, 'a', 'sha-a')
        head = rebased_head_item(
            _candidate(item, 'sha-a'),
            rebased_sha=_REBASED, rebased_onto=_ONTO, merge_wt=_NEW_WT, cap_permit=None,
        )
        ordinary = _speculative_item(config, 'b', 'sha-b')
        passthrough = DecidedItem(
            request=_request(config, 'c'), immediate_outcome=MergeOutcome(status='conflict'),
            base_sha=_MAIN, speculative=False,
        )

        commits = rebased_head_commits([_entry(head), _entry(ordinary), _entry(passthrough)])

        assert commits == frozenset({_REBASED})

    async def test_a_frontier_without_a_rebased_head_has_none(
        self, config: OrchestratorConfig,
    ) -> None:
        assert rebased_head_commits([_entry(_speculative_item(config, 'b', 'sha-b'))]) == (
            frozenset()
        )
        assert rebased_head_commits([]) == frozenset()


_LIVE = RemergeBase(tip=_REBASED, reason=RemergeReason.LIVE_TIP)


class TestIsGateReanchor:
    @pytest.mark.parametrize(
        'site', [DeadBaseSite.GATE_REBASED, DeadBaseSite.GATE_REANCHOR], ids=str,
    )
    def test_a_gate_killed_base_anchored_on_a_live_tip_is_a_reanchor(
        self, site: DeadBaseSite,
    ) -> None:
        assert is_gate_reanchor(DeadBaseKiller(site, 'a'), _LIVE) is True

    @pytest.mark.parametrize(
        'site',
        sorted(set(DeadBaseSite) - {DeadBaseSite.GATE_REBASED, DeadBaseSite.GATE_REANCHOR}),
        ids=str,
    )
    def test_any_other_killer_is_a_void(self, site: DeadBaseSite) -> None:
        assert is_gate_reanchor(DeadBaseKiller(site, 'a'), _LIVE) is False

    def test_an_unknown_killer_is_a_void(self) -> None:
        assert is_gate_reanchor(None, _LIVE) is False

    @pytest.mark.parametrize(
        'reason', sorted(set(RemergeReason) - {RemergeReason.LIVE_TIP}), ids=str,
    )
    def test_a_gate_killed_base_that_fell_to_main_is_a_void(
        self, reason: RemergeReason,
    ) -> None:
        killer = DeadBaseKiller(DeadBaseSite.GATE_REBASED, 'a')

        assert is_gate_reanchor(killer, RemergeBase(tip=None, reason=reason)) is False
