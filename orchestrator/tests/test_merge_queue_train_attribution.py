"""Tests for train-attribution additions to merge_queue.py.

Step-3 (RED): TRAIN_VERIFY_FAILED_REASON_PREFIX tagging in _do_train_merge.
Step-5 (RED): reverify_member_solo verify-only contract.

Both test classes use an AsyncMock git_ops + a MagicMock _TrainMergeHost so
that _do_train_merge (and reverify_member_solo) can be driven to the desired
code paths without real cargo / git.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import make_placeholder_future, pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.git_ops import MergeResult
from orchestrator.merge_queue import (
    MAIN_HEALTH_RED_REASON_PREFIX,
    TRAIN_REBASE_CONFLICT_REASON_PREFIX,
    GroupMergeRequest,
    MergeOutcome,
    _do_train_merge,
)

# ---------------------------------------------------------------------------
# Helpers shared by both test classes
# ---------------------------------------------------------------------------


def _make_git_ops_mock() -> MagicMock:
    """Return an AsyncMock git_ops with required methods pre-wired."""
    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value='deadbeef00')
    git_ops.rebase_onto_main = AsyncMock(return_value=True)
    git_ops.merge_to_main = AsyncMock(
        return_value=MergeResult(
            success=True,
            conflicts=False,
            merge_commit='aaaa1111merge',
            pre_merge_sha='deadbeef00',
            merge_worktree=Path('/tmp/fake-merge-wt'),
        )
    )
    git_ops.cleanup_merge_worktree = AsyncMock()
    git_ops.advance_main = AsyncMock(return_value='advanced')
    return git_ops


def _make_host(git_ops: MagicMock) -> MagicMock:
    """Return a fake _TrainMergeHost MagicMock satisfying the Protocol."""
    host = MagicMock()
    host._git_ops = git_ops
    host._event_store = None
    host._post_merge_verify_timeouts = {}
    host._post_merge_verify_enospc_retries = {}
    host._cas_retries = {}
    host.MAX_POST_MERGE_VERIFY_TIMEOUTS = 3
    host.MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES = 3
    host.halt_for_wip = MagicMock()
    host.unhalt_wip = MagicMock()
    host._abandon_outcome = MagicMock(return_value=MergeOutcome('blocked', reason='abandoned'))
    return host


def _make_group_req(
    *,
    task_id: str = 'tip99',
    train_id: str = 'train-test-1',
    member_task_ids: list[str] | None = None,
    worktree: Path = Path('/tmp/fake-tip-wt'),
) -> GroupMergeRequest:
    """Return a minimal GroupMergeRequest with AsyncMock callbacks."""
    members = member_task_ids or ['b1', 'b2', 'tip99']
    status_check = AsyncMock(return_value={m: 'merge-deferred' for m in members})
    mark_member_done = AsyncMock()
    try:
        future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
    except RuntimeError:
        future = make_placeholder_future()

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.merge_verify_workspace = False
    config.max_advance_attempts = 3

    return GroupMergeRequest(
        task_id=task_id,
        branch=task_id,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
        train_id=train_id,
        member_task_ids=members,
        tip_branch=task_id,
        tip_task_id=task_id,
        status_check=status_check,
        mark_member_done=mark_member_done,
    )


# ---------------------------------------------------------------------------
# Step-3: TRAIN_VERIFY_FAILED_REASON_PREFIX tagging tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTrainVerifyFailedReasonTagging:
    """_do_train_merge tags the reason prefix on interaction-candidate verify-red.

    Interaction candidate = status=='blocked' AND failure_category!='' AND
    reason does NOT start with MAIN_HEALTH_RED_REASON_PREFIX.
    """

    async def test_verify_red_with_category_gets_tagged(self) -> None:
        """Generic verify-red with failure_category set → reason starts with prefix."""
        from orchestrator.merge_queue import TRAIN_VERIFY_FAILED_REASON_PREFIX

        git_ops = _make_git_ops_mock()
        host = _make_host(git_ops)
        req = _make_group_req()

        original_reason = 'Post-merge verification failed: 3 tests failed'
        patched_outcome = MergeOutcome(
            'blocked',
            reason=original_reason,
            failure_category='cargo_test',
        )

        with patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            new=AsyncMock(return_value=patched_outcome),
        ):
            result = await _do_train_merge(host, req)

        assert result.status == 'blocked'
        assert result.reason.startswith(TRAIN_VERIFY_FAILED_REASON_PREFIX), (
            f'expected reason to start with {TRAIN_VERIFY_FAILED_REASON_PREFIX!r}, '
            f'got {result.reason!r}'
        )
        # Original summary must be preserved in the tagged reason
        assert original_reason in result.reason
        # failure_category preserved
        assert result.failure_category == 'cargo_test'

    async def test_main_health_red_not_tagged(self) -> None:
        """Main-health-red outcome must NOT be tagged (even though failure_category is set)."""
        from orchestrator.merge_queue import TRAIN_VERIFY_FAILED_REASON_PREFIX

        git_ops = _make_git_ops_mock()
        host = _make_host(git_ops)
        req = _make_group_req()

        mhr_reason = (
            f'{MAIN_HEALTH_RED_REASON_PREFIX} — test "foo" fails on bare main'
        )
        mhr_outcome = MergeOutcome(
            'blocked',
            reason=mhr_reason,
            failure_category='cargo_test',
        )

        with patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            new=AsyncMock(return_value=mhr_outcome),
        ):
            result = await _do_train_merge(host, req)

        assert not result.reason.startswith(TRAIN_VERIFY_FAILED_REASON_PREFIX), (
            f'main-health-red must NOT get tagged; got {result.reason!r}'
        )
        assert result.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX)

    async def test_transient_infra_empty_category_not_tagged(self) -> None:
        """Transient-infra outcome (failure_category='') must NOT be tagged."""
        from orchestrator.merge_queue import TRAIN_VERIFY_FAILED_REASON_PREFIX

        git_ops = _make_git_ops_mock()
        host = _make_host(git_ops)
        req = _make_group_req()

        transient_outcome = MergeOutcome(
            'blocked',
            reason='Post-merge verification failed: disk full',
            failure_category='',  # no category → transient/disk
        )

        with patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            new=AsyncMock(return_value=transient_outcome),
        ):
            result = await _do_train_merge(host, req)

        assert not result.reason.startswith(TRAIN_VERIFY_FAILED_REASON_PREFIX), (
            f'empty failure_category must NOT be tagged; got {result.reason!r}'
        )

    async def test_tip_rebase_conflict_not_tagged(self) -> None:
        """Tip-rebase conflict path produces TRAIN_REBASE_CONFLICT prefix, NOT the new prefix."""
        from orchestrator.merge_queue import TRAIN_VERIFY_FAILED_REASON_PREFIX

        git_ops = _make_git_ops_mock()
        git_ops.rebase_onto_main = AsyncMock(return_value=False)  # conflict!
        host = _make_host(git_ops)
        req = _make_group_req()

        with patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            new=AsyncMock(side_effect=AssertionError('should not reach verify')),
        ):
            result = await _do_train_merge(host, req)

        assert not result.reason.startswith(TRAIN_VERIFY_FAILED_REASON_PREFIX), (
            f'rebase-conflict must NOT use the new prefix; got {result.reason!r}'
        )
        assert result.reason.startswith(TRAIN_REBASE_CONFLICT_REASON_PREFIX)


# ---------------------------------------------------------------------------
# Step-5: reverify_member_solo verify-only contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReverifyMemberSoloContract:
    """reverify_member_solo wraps _run_post_merge_verify and never calls advance_main."""

    async def test_pass_returns_solo_verify_result_passed(self) -> None:
        """_run_post_merge_verify → None (pass): SoloVerifyResult.passed=True."""
        from orchestrator.merge_queue import SoloVerifyResult, reverify_member_solo

        git_ops = _make_git_ops_mock()
        solo_wt = Path('/tmp/fake-solo-wt')
        solo_branch = '_solo-b2'
        tip_sha = 'cafe1234solo'
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))

        with patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            new=AsyncMock(return_value=None),
        ):
            result = await reverify_member_solo(
                git_ops=git_ops,
                member_id='b2',
                solo_wt=solo_wt,
                solo_branch=solo_branch,
                tip_sha=tip_sha,
                config=config,
                task_files=None,
                module_configs=[],
            )

        assert isinstance(result, SoloVerifyResult)
        assert result.passed is True
        assert result.member_id == 'b2'
        assert result.merge_sha == tip_sha
        assert result.reason == ''

        # advance_main must NEVER be called (verify-only contract)
        git_ops.advance_main.assert_not_called()

    async def test_fail_returns_solo_verify_result_failed(self) -> None:
        """_run_post_merge_verify → blocked MergeOutcome: SoloVerifyResult.passed=False."""
        from orchestrator.merge_queue import SoloVerifyResult, reverify_member_solo

        git_ops = _make_git_ops_mock()
        solo_wt = Path('/tmp/fake-solo-wt')
        solo_branch = '_solo-b1'
        tip_sha = 'dead1234solo'
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))

        fail_outcome = MergeOutcome(
            'blocked',
            reason='Solo verify failed: 2 tests failed',
            failure_category='cargo_test',
        )

        with patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            new=AsyncMock(return_value=fail_outcome),
        ):
            result = await reverify_member_solo(
                git_ops=git_ops,
                member_id='b1',
                solo_wt=solo_wt,
                solo_branch=solo_branch,
                tip_sha=tip_sha,
                config=config,
                task_files=None,
                module_configs=[],
            )

        assert isinstance(result, SoloVerifyResult)
        assert result.passed is False
        assert result.member_id == 'b1'
        assert 'Solo verify failed' in result.reason

        # advance_main must NEVER be called
        git_ops.advance_main.assert_not_called()

    async def test_solo_worktree_cleaned_up_on_pass(self) -> None:
        """Pass-path: worktree+branch are RETAINED (handed off to _attribute_train_failure).

        Step-17 revision: the old contract (unconditional cleanup) is wrong for the pass
        path — advance_main needs the worktree alive.  cleanup_merge_worktree must NOT
        be called and delete_solo_branch must NOT be called on the pass path.
        solo_wt and solo_branch must be populated in the returned result.
        """
        from orchestrator.merge_queue import SoloVerifyResult, reverify_member_solo

        git_ops = _make_git_ops_mock()
        git_ops.delete_solo_branch = AsyncMock()
        solo_wt = Path('/tmp/fake-solo-wt-pass')
        solo_branch = '_solo-b2'
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))

        with patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            new=AsyncMock(return_value=None),
        ):
            result = await reverify_member_solo(
                git_ops=git_ops,
                member_id='b2',
                solo_wt=solo_wt,
                solo_branch=solo_branch,
                tip_sha='abc123',
                config=config,
                task_files=None,
                module_configs=[],
            )

        # Pass path: worktree and branch are left intact for advance_main
        git_ops.cleanup_merge_worktree.assert_not_called()
        git_ops.delete_solo_branch.assert_not_called()

        # solo_wt and solo_branch populated for handoff
        assert isinstance(result, SoloVerifyResult)
        assert result.passed is True
        assert result.solo_wt == solo_wt
        assert result.solo_branch == solo_branch

    async def test_solo_worktree_cleaned_up_on_fail(self) -> None:
        """Fail-path: cleanup_merge_worktree AND delete_solo_branch are both called;
        returned SoloVerifyResult has solo_wt=None and solo_branch=None to reflect
        that the paths are already torn down (not stale references)."""
        from orchestrator.merge_queue import SoloVerifyResult, reverify_member_solo

        git_ops = _make_git_ops_mock()
        git_ops.delete_solo_branch = AsyncMock()
        solo_wt = Path('/tmp/fake-solo-wt-fail')
        solo_branch = '_solo-b1'
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))

        fail_outcome = MergeOutcome('blocked', reason='tests failed', failure_category='x')

        with patch(
            'orchestrator.merge_queue._run_post_merge_verify',
            new=AsyncMock(return_value=fail_outcome),
        ):
            result = await reverify_member_solo(
                git_ops=git_ops,
                member_id='b1',
                solo_wt=solo_wt,
                solo_branch=solo_branch,
                tip_sha='xyz789',
                config=config,
                task_files=None,
                module_configs=[],
            )

        # Fail path: both worktree and branch are torn down
        git_ops.cleanup_merge_worktree.assert_called_once_with(solo_wt)
        git_ops.delete_solo_branch.assert_called_once_with(solo_branch)

        # Returned result must have solo_wt=None and solo_branch=None to avoid
        # callers operating on stale (already-deleted) paths.
        assert isinstance(result, SoloVerifyResult)
        assert result.passed is False
        assert result.solo_wt is None, (
            f'Expected solo_wt=None on fail path, got {result.solo_wt!r}'
        )
        assert result.solo_branch is None, (
            f'Expected solo_branch=None on fail path, got {result.solo_branch!r}'
        )
