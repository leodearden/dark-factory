"""Unit tests for ``retire_cancelled_merge_request`` (PRD merge-worktree-
lifecycle-integrity §4 C3 last row / §5 D3 — task ε).

``retire_cancelled_merge_request`` FULLY RETIRES a cancelled live merge entry
before ``merge_cancel`` returns, so an immediate resubmit gets a FRESH entry
and never coalesces onto / observes the cancelled corpse:

  (i)   identity-guarded ``registry.release(branch, detach_waiters=True)`` —
        release only when the slot still belongs to the retiring request_id
        (a concurrent resubmit may have reclaimed the freed slot);
  (ii)  route the branch's in-flight worktree through the C1 primitive
        ``remove_merge_worktree_guarded(reason='merge_cancel_retire')`` and
        emit ``worktree_reaped``;
  (iii) YIELD the loop so the cancel's ``_on_finalized`` records the terminal
        ``'abandoned'`` outcome FIRST, then ``retention.forget`` clears it
        (yield-then-forget — a synchronous forget would be clobbered by the
        async re-record).

``retire_cancelled_merge_request`` is imported LOCALLY inside each test so the
not-yet-defined symbol never breaks collection during RED (step-3 → step-4).
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.merge_queue import enqueue_merge_request
from orchestrator.merge_types import (
    InFlightMergeRegistry,
    MergeRequest,
    QueuedBranch,
    TerminalOutcomeRecord,
    TerminalOutcomeRetention,
)


class _RetireSpy:
    """git_ops spy for retirement tests.

    ``find_inflight_merge_worktree`` returns the planted scratch path only
    WHILE it still exists on disk, so a post-removal re-scan finds nothing
    (mirrors the disk-scan/removal surface of the C3
    ``test_merge_queue_c3_submit_identity._ScratchSpy``).
    ``remove_merge_worktree_guarded`` records ``(path, reason)`` and deletes
    the dir, returning the ``'removed'`` outcome.
    """

    def __init__(self, scratch: Path | None) -> None:
        self._scratch = scratch
        self.removed: list[tuple[Path, str]] = []
        self.find_calls = 0

    async def find_inflight_merge_worktree(self, branch: str) -> Path | None:  # noqa: ARG002
        self.find_calls += 1
        if self._scratch is not None and self._scratch.exists():
            return self._scratch
        return None

    async def remove_merge_worktree_guarded(self, path: Path, *, reason: str) -> str:
        self.removed.append((path, reason))
        if path.exists():
            shutil.rmtree(path)
        return 'removed'


def _plant_worktree(tmp_path: Path, name: str) -> Path:
    wt = tmp_path / name
    wt.mkdir()
    (wt / '.git').write_text('gitdir: elsewhere\n')
    return wt


def _config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=tmp_path,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
        ),
    )


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
    *,
    request_id: str | None = None,
    snapshot_tip: str | None = None,
) -> MergeRequest:
    kwargs: dict[str, Any] = {}
    if request_id is not None:
        kwargs['request_id'] = request_id
    if snapshot_tip is not None:
        kwargs['snapshot_tip'] = snapshot_tip
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
        **kwargs,
    )


class TestRetireCancelledMergeRequest:
    """Behavior of the orchestrator-owned retirement helper."""

    @pytest.mark.asyncio
    async def test_basic_release_worktree_and_forget(self, tmp_path: Path) -> None:
        """(a) release the slot, remove the worktree via C1, forget the sticky."""
        from orchestrator.merge_queue import retire_cancelled_merge_request

        registry = InFlightMergeRegistry()
        retention = TerminalOutcomeRetention()
        fut = asyncio.get_running_loop().create_future()
        registry.acquire('B', 'T', fut, request_id='mr-a', snapshot_tip='sha-a')
        retention.record(TerminalOutcomeRecord(
            request_id='mr-a', branch='B', task_id='T', state='abandoned',
        ))
        wt = _plant_worktree(tmp_path, 'merge-B')
        spy = _RetireSpy(wt)

        outcome = await retire_cancelled_merge_request(
            request_id='mr-a', branch='B', task_id='T',
            registry=registry, retention=retention, git_ops=spy, event_store=None,
        )

        assert registry.is_inflight('B') is False
        assert retention.get_by_branch('B') is None
        assert retention.get_by_task('T') is None
        assert retention.get('mr-a') is None
        assert spy.removed == [(wt, 'merge_cancel_retire')]
        assert not wt.exists()
        assert outcome == 'removed'

    @pytest.mark.asyncio
    async def test_yield_then_forget_ordering(self, tmp_path: Path) -> None:
        """(b) forget runs AFTER the cancel's async _on_finalized re-record.

        Register _on_finalized via enqueue_merge_request, cancel the future
        (schedules the async 'abandoned' record), then retire.  If retire
        forgot BEFORE yielding, the async re-record would leave a live entry;
        a clean ring proves yield-then-forget.
        """
        from orchestrator.merge_queue import retire_cancelled_merge_request

        config = _config(tmp_path)
        registry = InFlightMergeRegistry()
        ring = TerminalOutcomeRetention()
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        wt = _plant_worktree(tmp_path, 'merge-bb')
        req = _make_request(
            'bb', 'bb', wt, config, request_id='mr-b', snapshot_tip='sha-b',
        )

        await enqueue_merge_request(queue, req, event_store=None, retention=ring)
        # Sanity: the request is now queued but not yet finalised.
        assert ring.get_by_branch('bb') is None

        req.result.cancel()  # schedules _on_finalized -> record(state='abandoned')

        await retire_cancelled_merge_request(
            request_id='mr-b', branch='bb', task_id='bb',
            registry=registry, retention=ring, git_ops=None, event_store=None,
        )

        # If the re-record had shadowed forget, these would be non-None.
        assert ring.get_by_branch('bb') is None
        assert ring.get_by_task('bb') is None
        assert ring.get('mr-b') is None

    @pytest.mark.asyncio
    async def test_identity_guard_does_not_drop_fresh_slot(self, tmp_path: Path) -> None:
        """(c) a slot reclaimed by a DIFFERENT request_id must NOT be released."""
        from orchestrator.merge_queue import retire_cancelled_merge_request

        registry = InFlightMergeRegistry()
        retention = TerminalOutcomeRetention()
        fut_a = asyncio.get_running_loop().create_future()
        registry.acquire('B', 'T', fut_a, request_id='mr-a', snapshot_tip='sha-a')
        # An immediate resubmit reclaims the freed branch slot for a new id.
        registry.release('B', detach_waiters=True)  # cancels fut_a
        fut_other = asyncio.get_running_loop().create_future()
        registry.acquire('B', 'T2', fut_other, request_id='mr-other', snapshot_tip='sha-o')

        await retire_cancelled_merge_request(
            request_id='mr-a', branch='B', task_id='T',
            registry=registry, retention=retention, git_ops=None, event_store=None,
        )

        entry = registry.entry('B')
        assert entry is not None
        assert entry.request_id == 'mr-other'  # fresh slot preserved
        assert not fut_other.done()  # its future was NOT cancelled
        fut_other.cancel()  # cleanup

    @pytest.mark.asyncio
    async def test_none_guards_degrade_to_noop(self, tmp_path: Path) -> None:
        """(d) git_ops=None / retention=None / registry=None never raise."""
        from orchestrator.merge_queue import retire_cancelled_merge_request

        outcome = await retire_cancelled_merge_request(
            request_id='mr-a', branch='B', task_id='T',
            registry=None, retention=None, git_ops=None, event_store=None,
        )
        assert outcome is None

    @pytest.mark.asyncio
    async def test_branch_none_still_forgets_without_worktree_scan(
        self, tmp_path: Path,
    ) -> None:
        """(d2) branch=None: forget still clears the sticky; no worktree scan."""
        from orchestrator.merge_queue import retire_cancelled_merge_request

        registry = InFlightMergeRegistry()
        retention = TerminalOutcomeRetention()
        retention.record(TerminalOutcomeRecord(
            request_id='mr-a', branch='B', task_id='T', state='abandoned',
        ))
        spy = _RetireSpy(_plant_worktree(tmp_path, 'merge-orphan'))

        outcome = await retire_cancelled_merge_request(
            request_id='mr-a', branch=None, task_id=None,
            registry=registry, retention=retention, git_ops=spy, event_store=None,
        )

        assert outcome is None
        assert retention.get('mr-a') is None  # forget still ran
        assert spy.find_calls == 0  # no branch -> no worktree scan
