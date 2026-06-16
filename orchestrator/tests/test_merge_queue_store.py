"""Tests for MergeQueueStore and related utilities (task 1772).

Covers:
  step-1:  record() + load() round-trips a single MergeRequest's identity.
  step-3:  remove() removes one entry; unknown id is a no-op.
  step-5:  load() robustness (non-existent path, empty file, corrupt file,
           idempotent record by request_id).
  step-7:  record() skips GroupMergeRequest.
  step-9:  reconstruct_merge_request() returns a fresh MergeRequest with
           correct identity.
  step-11: recover_pending_merges() selects survivors and drops already-landed
           / branch-gone records.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import make_placeholder_future

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.merge_queue import GroupMergeRequest, MergeOutcome, MergeRequest

# Import the module under test — will fail (ImportError) until step-2 creates it.
from orchestrator.merge_queue_store import (
    MergeQueueStore,
    PersistedMergeRequest,
    reconstruct_merge_request,
    recover_pending_merges,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _real_config(tmp_path: Path) -> OrchestratorConfig:
    """Minimal real OrchestratorConfig pointing at tmp_path."""
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


def _make_req(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
    *,
    snapshot_tip: str | None = 'abc123',
    generation: int = 1,
    lane: str = 'normal',
    task_files: list[str] | None = None,
    pre_rebased: bool = False,
) -> MergeRequest:
    """Build a MergeRequest with a placeholder future (safe outside a running loop)."""
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=pre_rebased,
        task_files=task_files,
        module_configs=[],
        config=config,
        result=make_placeholder_future(),
        snapshot_tip=snapshot_tip,
        generation=generation,
        lane=lane,  # type: ignore[arg-type]
    )


def _make_group_req(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
) -> GroupMergeRequest:
    """Build a GroupMergeRequest with dummy train fields and async callbacks."""
    async def _status_check(ids: list[str]) -> dict[str, str]:
        return {}

    async def _mark_done(tid: str, sha: str) -> None:
        pass

    return GroupMergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=make_placeholder_future(),
        train_id='train-001',
        member_task_ids=[task_id],
        tip_branch=branch,
        tip_task_id=task_id,
        status_check=_status_check,
        mark_member_done=_mark_done,
    )


# ---------------------------------------------------------------------------
# step-1 — record() + load() round-trip
# ---------------------------------------------------------------------------


class TestMergeQueueStoreRecordLoad:
    """record() + load() round-trips a single-task MergeRequest's serializable identity."""

    def test_record_load_roundtrip(self, tmp_path: Path) -> None:
        """Verify all serializable identity fields survive record + load."""
        store_path = tmp_path / 'data' / 'orchestrator' / 'merge_queue.json'
        store = MergeQueueStore(store_path)
        config = _real_config(tmp_path)
        worktree = tmp_path / '.worktrees' / 'task-42'
        worktree.mkdir(parents=True, exist_ok=True)

        req = _make_req(
            task_id='42',
            branch='42',
            worktree=worktree,
            config=config,
            snapshot_tip='deadbeef',
            generation=2,
            lane='high',
            task_files=['src/foo.py', 'tests/test_foo.py'],
            pre_rebased=True,
        )

        store.record(req)
        records = store.load()

        assert len(records) == 1, f'Expected 1 record, got {len(records)}'
        p = records[0]
        assert isinstance(p, PersistedMergeRequest)
        assert p.request_id == req.request_id
        assert p.task_id == '42'
        assert p.branch == '42'
        assert p.worktree == str(worktree)
        assert p.pre_rebased is True
        assert p.task_files == ['src/foo.py', 'tests/test_foo.py']
        assert p.snapshot_tip == 'deadbeef'
        assert p.generation == 2
        assert p.lane == 'high'
        assert p.enqueued_at == pytest.approx(req.enqueued_at, rel=1e-6)


# ---------------------------------------------------------------------------
# step-3 — remove()
# ---------------------------------------------------------------------------


class TestMergeQueueStoreRemove:
    """remove() deletes a single entry; unknown id is a no-op."""

    def test_remove_one_keeps_other(self, tmp_path: Path) -> None:
        store_path = tmp_path / 'merge_queue.json'
        store = MergeQueueStore(store_path)
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()

        req_a = _make_req('a', 'a', wt, config)
        req_b = _make_req('b', 'b', wt, config)

        store.record(req_a)
        store.record(req_b)

        store.remove(req_a.request_id)
        records = store.load()

        assert len(records) == 1
        assert records[0].request_id == req_b.request_id

    def test_remove_unknown_is_noop(self, tmp_path: Path) -> None:
        store_path = tmp_path / 'merge_queue.json'
        store = MergeQueueStore(store_path)
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()

        req = _make_req('x', 'x', wt, config)
        store.record(req)

        # Removing a nonexistent id must not raise or remove the existing entry.
        store.remove('mr-does-not-exist')
        records = store.load()
        assert len(records) == 1
        assert records[0].request_id == req.request_id


# ---------------------------------------------------------------------------
# step-5 — robustness: fail-open loads, idempotent record
# ---------------------------------------------------------------------------


class TestMergeQueueStoreRobustness:
    """load() is fail-open; record() is idempotent by request_id."""

    def test_load_nonexistent_path_returns_empty(self, tmp_path: Path) -> None:
        store = MergeQueueStore(tmp_path / 'nonexistent' / 'merge_queue.json')
        assert store.load() == []

    def test_load_empty_file_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / 'merge_queue.json'
        p.write_text('', encoding='utf-8')
        store = MergeQueueStore(p)
        assert store.load() == []

    def test_load_corrupt_file_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / 'merge_queue.json'
        p.write_text('not json {{{{', encoding='utf-8')
        store = MergeQueueStore(p)
        assert store.load() == []

    def test_record_idempotent_by_request_id(self, tmp_path: Path) -> None:
        """Recording the same request_id twice yields exactly one entry."""
        store_path = tmp_path / 'merge_queue.json'
        store = MergeQueueStore(store_path)
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()

        req = _make_req('t', 't', wt, config, generation=1)
        store.record(req)

        # Mutate generation on the SAME request object (same request_id) and record again.
        object.__setattr__(req, 'generation', 3) if hasattr(req, '__dataclass_fields__') else None
        req2 = MergeRequest(
            task_id=req.task_id,
            branch=req.branch,
            worktree=req.worktree,
            pre_rebased=req.pre_rebased,
            task_files=req.task_files,
            module_configs=[],
            config=config,
            result=make_placeholder_future(),
            request_id=req.request_id,  # SAME id
            snapshot_tip=req.snapshot_tip,
            generation=3,             # updated value
            lane=req.lane,
            enqueued_at=req.enqueued_at,
        )
        store.record(req2)

        records = store.load()
        assert len(records) == 1, f'Expected 1 (idempotent), got {len(records)}'
        assert records[0].generation == 3, 'Latest value should overwrite the previous'


# ---------------------------------------------------------------------------
# step-7 — GroupMergeRequest is NOT journaled
# ---------------------------------------------------------------------------


class TestMergeQueueStoreSkipsGroupMergeRequest:
    """record() returns early for GroupMergeRequest; load() stays empty."""

    def test_group_request_not_recorded(self, tmp_path: Path) -> None:
        store_path = tmp_path / 'merge_queue.json'
        store = MergeQueueStore(store_path)
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()

        group_req = _make_group_req('train-tip', 'train-tip', wt, config)
        store.record(group_req)

        assert store.load() == [], 'GroupMergeRequest must not be persisted'


# ---------------------------------------------------------------------------
# step-9 — reconstruct_merge_request()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReconstructMergeRequest:
    """reconstruct_merge_request returns a MergeRequest with correct identity."""

    async def test_reconstruct_preserves_identity(self, tmp_path: Path) -> None:
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()

        # Build a PersistedMergeRequest directly.
        original = _make_req(
            '99',
            '99',
            wt,
            config,
            snapshot_tip='cafebabe',
            generation=2,
            lane='high',
            task_files=['a.py'],
        )
        store_path = tmp_path / 'merge_queue.json'
        store = MergeQueueStore(store_path)
        store.record(original)
        [persisted] = store.load()

        reconstructed = reconstruct_merge_request(persisted, config)

        # Must be a MergeRequest (not a GroupMergeRequest).
        assert type(reconstructed) is MergeRequest

        # Fresh unresolved future.
        assert isinstance(reconstructed.result, asyncio.Future)
        assert not reconstructed.result.done(), 'Future should NOT be resolved yet'

        # Identity fields preserved.
        assert reconstructed.request_id == original.request_id
        assert reconstructed.task_id == '99'
        assert reconstructed.branch == '99'
        assert reconstructed.snapshot_tip == 'cafebabe'
        assert reconstructed.generation == 2
        assert reconstructed.lane == 'high'
        assert reconstructed.task_files == ['a.py']
        assert reconstructed.worktree == wt
        assert isinstance(reconstructed.worktree, Path)

        # Defaults for non-serializable fields.
        assert reconstructed.module_configs == []
        assert reconstructed.pre_rebased is False
        assert reconstructed.config is config


# ---------------------------------------------------------------------------
# step-11 — recover_pending_merges() with fake git_ops
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecoverPendingMerges:
    """recover_pending_merges selects survivors and drops already-landed or gone branches."""

    async def test_recover_selects_survivors(self, tmp_path: Path) -> None:
        """
        Three records:
          A — branch exists and NOT an ancestor of main → re-enqueued (recovered)
          B — is_ancestor True → already merged, dropped
          C — resolve_branch_sha None → branch gone, dropped
        """
        config = _real_config(tmp_path)
        wt = tmp_path / 'wt'
        wt.mkdir()

        store_path = tmp_path / 'merge_queue.json'
        store = MergeQueueStore(store_path)

        req_a = _make_req('a', 'a', wt, config)
        req_b = _make_req('b', 'b', wt, config)
        req_c = _make_req('c', 'c', wt, config)
        store.record(req_a)
        store.record(req_b)
        store.record(req_c)

        main_branch = 'main'
        branch_prefix = 'task/'

        # Fake git_ops:
        # - A: branch exists (resolve returns sha), NOT ancestor
        # - B: branch exists, IS ancestor
        # - C: branch gone (resolve returns None)
        async def fake_resolve(branch: str) -> str | None:
            mapping = {
                f'{branch_prefix}a': 'sha-a',
                f'{branch_prefix}b': 'sha-b',
                f'{branch_prefix}c': None,
            }
            return mapping.get(branch)

        async def fake_is_ancestor(ancestor: str, descendant: str) -> bool:
            return ancestor == f'{branch_prefix}b'

        fake_git_ops = MagicMock()
        fake_git_ops.resolve_branch_sha = fake_resolve
        fake_git_ops.is_ancestor = fake_is_ancestor

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()

        report = await recover_pending_merges(
            store,
            queue,
            fake_git_ops,
            config,
            event_store=None,
            main_branch=main_branch,
            branch_prefix=branch_prefix,
        )

        # --- queue assertions ---
        assert queue.qsize() == 1, f'Expected exactly 1 re-enqueued item; got {queue.qsize()}'
        recovered_req = queue.get_nowait()
        assert recovered_req.request_id == req_a.request_id, (
            f'Expected req_a to be recovered; got {recovered_req.request_id!r}'
        )
        assert not recovered_req.result.done(), 'Recovered request must have unresolved future'

        # --- drop assertions ---
        remaining = store.load()
        remaining_ids = {r.request_id for r in remaining}
        assert req_b.request_id not in remaining_ids, 'Already-merged B must be removed'
        assert req_c.request_id not in remaining_ids, 'Branch-gone C must be removed'

        # --- report ---
        assert report['recovered'] == 1
        assert report['dropped'] == 2
