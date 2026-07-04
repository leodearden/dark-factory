"""Behavioral exhaustive-handling regression pin for the MQ-refactor ο union
split (task 2000): ``RealMergeItem`` / ``DecidedItem`` replacing the single
``SpeculativeItem`` product dataclass.

These tests drive the real dispatch/finalize/verify-and-advance pipeline with
each variant directly constructed (not via the production factory sites) and
assert the two arms are handled exactly as the pre-split product type was:

  1. A ``RealMergeItem`` takes the real-verify path and lands on main.
  2. A ``DecidedItem`` dispatches as a passthrough (no verify task) and
     delivers its ``immediate_outcome`` in submission order, honoring
     ``already_delivered`` (no double ``set_result``).
  3. The train-handoff shape built at the ``_merger_loop`` train site
     (``DecidedItem(immediate_outcome=outcome, speculative=False)``) has no
     behavioral drift under the split: passthrough delivery, no main
     advancement of its own, no speculation-slot release.

All three pass against the step-2 union split; they would have been
unwritable (no variant types to construct) against the pre-split single
``SpeculativeItem`` product dataclass.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import (
    DecidedItem,
    GroupMergeRequest,
    MergeOutcome,
    MergeRequest,
    RealMergeItem,
    SpeculativeMergeWorker,
)

pytestmark = pytest.mark.asyncio

# ── Fixtures + helpers (per-file duplication convention — see
# test_merge_queue_resolve_release.py / test_merge_queue_concurrent_verify.py) ──


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


def _make_req(
    task_id: str, branch: str, worktree: Path, config: OrchestratorConfig,
) -> MergeRequest:
    """Build a minimal MergeRequest with a fresh event-loop future."""
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
    )


async def _make_branch_with_file(
    git_ops: GitOps, branch_name: str, filename: str, content: str,
) -> Path:
    """Create a worktree branch with one committed file and return its path."""
    worktree = (await git_ops.create_worktree(branch_name)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


# ── (1) RealMergeItem: real-verify path advances main ───────────────────────


async def test_real_merge_item_lands_via_verify_and_advance(
    git_ops: GitOps, config: OrchestratorConfig,
) -> None:
    """A RealMergeItem driven through _verify_and_advance takes the
    real-verify path (host lease → _run_inflight_verify → _finalize_inflight
    CAS loop) and advances main — the REAL arm of the union must still land
    merges exactly as the pre-split product type did.
    """
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker = SpeculativeMergeWorker(git_ops, queue)

    wt = await _make_branch_with_file(git_ops, 'union-real', 'real_arm.py', 'x = 1\n')
    req = _make_req('union-real', 'union-real', wt, config)

    merge_result = await git_ops.merge_to_main(wt, 'union-real')
    assert merge_result.success and merge_result.merge_commit is not None
    assert merge_result.merge_worktree is not None
    worker._register_owned_merge_worktree(merge_result.merge_worktree)
    base_sha = await git_ops.get_main_sha()

    item = RealMergeItem(
        request=req,
        merge_result=merge_result,
        merge_wt=merge_result.merge_worktree,
        base_sha=base_sha,
        speculative=False,
    )

    passing = MagicMock(passed=True, summary='', timed_out=False)
    with patch(
        'orchestrator.merge_queue.run_scoped_verification',
        AsyncMock(return_value=passing),
    ):
        advanced = await worker._verify_and_advance(item)

    assert advanced is True, 'RealMergeItem must take the real-verify path and advance main'
    outcome = req.result.result()
    assert outcome.status == 'done', f'expected done, got {outcome!r}'


# ── (2) DecidedItem: passthrough dispatch + submission-order delivery ───────


async def test_decided_item_dispatches_as_passthrough(
    git_ops: GitOps, config: OrchestratorConfig,
) -> None:
    """A DecidedItem (immediate_outcome set) must dispatch as a passthrough
    entry — no lease, no real verify task — carrying its own immediate_outcome,
    and _finalize_inflight must deliver it in submission order."""
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker = SpeculativeMergeWorker(git_ops, queue)

    req = _make_req('union-decided', 'union-decided', git_ops.project_root, config)
    outcome = MergeOutcome('conflict')
    item = DecidedItem(
        request=req, immediate_outcome=outcome, base_sha='deadbeef', speculative=False,
    )

    entry = await worker._dispatch_item(item)
    assert entry is not None
    assert entry.verify_task is None, 'DecidedItem must dispatch with no real verify task'
    assert entry.lease is None, 'DecidedItem passthrough must not hold a host lease'
    assert entry.passthrough_outcome is outcome, (
        'DecidedItem must dispatch as a passthrough carrying its own immediate_outcome'
    )

    delivered = await worker._finalize_inflight(entry)
    assert req.result.result() is outcome, (
        'passthrough must deliver the outcome in submission order'
    )
    assert delivered is False, "'conflict' is not in ('done', 'already_merged')"


async def test_decided_item_already_delivered_skips_double_set_result(
    git_ops: GitOps, config: OrchestratorConfig,
) -> None:
    """already_delivered=True must not raise InvalidStateError nor overwrite
    the out-of-band-delivered result — the verifier is an ordering-token
    only for an already-delivered DecidedItem."""
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker = SpeculativeMergeWorker(git_ops, queue)

    req = _make_req('union-already', 'union-already', git_ops.project_root, config)
    conflict_outcome = MergeOutcome('conflict')
    req.result.set_result(conflict_outcome)  # merger already resolved OOB

    item = DecidedItem(
        request=req, immediate_outcome=conflict_outcome, base_sha='deadbeef',
        speculative=False, already_delivered=True,
    )
    entry = await worker._dispatch_item(item)
    assert entry is not None
    assert entry.passthrough_outcome is conflict_outcome

    # Must not raise InvalidStateError (double set_result on an already-done Future).
    await worker._finalize_inflight(entry)
    assert req.result.result() is conflict_outcome


# ── (3) Train handoff: DecidedItem passthrough, no behavioral drift ─────────


async def test_train_handoff_decided_item_passthrough_no_drift(
    git_ops: GitOps, config: OrchestratorConfig,
) -> None:
    """Mirrors the _merger_loop train-handoff site exactly — a
    GroupMergeRequest wrapped in ``DecidedItem(immediate_outcome=outcome,
    speculative=False)`` — and pins that _do_train_merge's handoff has no
    behavioral drift under the RealMergeItem/DecidedItem split: it dispatches
    as a passthrough, advances nothing itself, and releases no speculation
    slot (trains are structurally exempt — DecidedItem carries no
    counts_against_cap / merge_wt for either mechanism to act on)."""
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker = SpeculativeMergeWorker(git_ops, queue)

    future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
    req = GroupMergeRequest(
        task_id='union-train',
        branch='union-train',
        worktree=git_ops.project_root,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
        train_id='union-train-id',
        member_task_ids=['union-train'],
        tip_branch='union-train',
        tip_task_id='union-train',
        status_check=AsyncMock(),
        mark_member_done=AsyncMock(),
    )
    done_outcome = MergeOutcome('done', merge_sha='a' * 40)
    actual_main = await git_ops.get_main_sha()

    # Exactly the _merger_loop train-handoff shape: DecidedItem(request=req,
    # base_sha=actual_main, speculative=False, immediate_outcome=outcome,
    # started_monotonic=t0) — no merge_result/merge_wt/counts_against_cap at all.
    item = DecidedItem(
        request=req,
        base_sha=actual_main,
        speculative=False,
        immediate_outcome=done_outcome,
        started_monotonic=time.monotonic(),
    )

    main_before = await git_ops.get_main_sha()
    slot_locked_before = worker._speculation_slot.locked()

    entry = await worker._dispatch_item(item)
    assert entry is not None
    assert entry.verify_task is None
    assert entry.was_speculative is False
    assert entry.passthrough_outcome is done_outcome

    advanced = await worker._finalize_inflight(entry)
    assert advanced is True, "'done' passthrough status counts as landed"
    assert req.result.result() is done_outcome

    main_after = await git_ops.get_main_sha()
    assert main_after == main_before, (
        'the passthrough delivery itself must not advance main — the train '
        'landing already happened inside _do_train_merge before this handoff'
    )
    assert worker._speculation_slot.locked() == slot_locked_before, (
        'train item has speculative=False; no speculation slot may be acquired '
        'or released by its passthrough delivery'
    )
