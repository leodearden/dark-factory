"""Tests for _finalizing_head observability window (task 1741).

When SpeculativeMergeWorker._verifier_loop pops the head from _inflight and
awaits _finalize_inflight(head), the head is invisible to snapshot() and thus
to all MCP read APIs (get_merge_queue, merge_status Tier-1).  Under K≥2
(Lever C) this window is observable: the second entry appears as head-of-line
while the actual head is being finalised.

FIX: add self._finalizing_head side-field (mirrors _remerging_item / _inflight_req),
set/clear inside _finalize_inflight, and surface in snapshot() sections 0, occupancy,
and verify_in_progress.

Steps covered:
  step-1 RED  — snapshot() entries: finalizing head as head-of-line
  step-3 RED  — snapshot() occupancy: finalizing head's host in by_host
  step-5 RED  — snapshot() verify_in_progress: prefers finalizing head
  step-7 RED  — _finalize_inflight sets/clears _finalizing_head around verify await
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import (
    InflightEntry,
    InflightVerifyResult,
    MergeOutcome,
    MergeRequest,
    SpeculativeMergeWorker,
    SpeculativeItem,
)
from orchestrator.verify_runner import HostLease


# ── fixtures ─────────────────────────────────────────────────────────────────


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


# ── shared helpers ────────────────────────────────────────────────────────────


def _make_req(
    task_id: str,
    branch: str,
    config: OrchestratorConfig,
    git_repo: Path,
) -> MergeRequest:
    """Build a minimal MergeRequest with a fresh event-loop future."""
    loop = asyncio.get_event_loop()
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=git_repo,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=loop.create_future(),
        lane='normal',
    )


def _make_item(req: MergeRequest) -> SpeculativeItem:
    return SpeculativeItem(
        request=req,
        merge_result=None,
        merge_wt=None,
        base_sha='abc123',
        speculative=False,
        skip_verify=False,
    )


def _make_entry(
    req: MergeRequest,
    lease: HostLease | None,
    phase: str = 'verifying',
    started_at: float | None = None,
) -> InflightEntry:
    return InflightEntry(
        item=_make_item(req),
        lease=lease,
        verify_task=None,
        merge_wt=None,
        was_speculative=False,
        phase=phase,
        started_at=started_at,
    )


def _make_mock_allocator(host_names: list[str]) -> MagicMock:
    alloc = MagicMock()
    alloc.host_names = host_names
    alloc.release = AsyncMock()
    alloc.cancel_and_release = AsyncMock()
    return alloc


# ── step-1: snapshot() entries ────────────────────────────────────────────────


@pytest.mark.asyncio
class TestSnapshotFinalizingHeadEntries:
    """snapshot() must surface _finalizing_head as position-0 (head-of-line).

    RED until step-2 GREEN adds section-0 to snapshot().

    State: local_head in worker._finalizing_head (popped from _inflight),
    laptop_entry in worker._inflight.  snapshot() must show local first, then
    laptop, with depth==2 and head_of_line == local task_id.
    """

    async def test_finalizing_head_is_head_of_line(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """entries[0] must be the finalizing head, not _inflight[0].

        RED: snapshot() iterates only self._inflight, so entries[0] is the laptop
        entry and depth==1.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        req_local = _make_req('fhv-local', 'task/fhv-local', config, git_repo)
        req_laptop = _make_req('fhv-laptop', 'task/fhv-laptop', config, git_repo)

        local_lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        laptop_lease = HostLease(name='laptop', runner=MagicMock(), is_local=False)

        started = time.time() - 5.0
        local_head = _make_entry(req_local, local_lease, phase='verifying', started_at=started)
        laptop_entry = _make_entry(req_laptop, laptop_lease, phase='verifying')

        # The finalize window: local_head has been popped, laptop is still in _inflight.
        worker._inflight.append(laptop_entry)
        worker._finalizing_head = local_head  # type: ignore[attr-defined]

        snap = worker.snapshot()

        entry_task_ids = [e['task_id'] for e in snap['entries']]

        assert snap['depth'] == 2, (
            f"Expected depth==2 (local_head + laptop), got {snap['depth']}. "
            "RED: _finalizing_head not yet read by snapshot()."
        )
        assert snap['entries'][0]['task_id'] == 'fhv-local', (
            f"Expected entries[0]['task_id']=='fhv-local', got {snap['entries'][0]['task_id']!r}. "
            "RED: snapshot() reads _inflight only; laptop appears first."
        )
        assert snap['entries'][0]['host'] == 'local', (
            f"Expected host=='local', got {snap['entries'][0]['host']!r}."
        )
        assert snap['entries'][0]['state'] == 'verifying', (
            f"Expected state=='verifying', got {snap['entries'][0]['state']!r}."
        )
        assert snap['entries'][0]['verify_started_at'] is not None, (
            "Expected verify_started_at to be set on the finalizing head entry."
        )
        assert snap['head_of_line'] == 'fhv-local', (
            f"Expected head_of_line=='fhv-local', got {snap['head_of_line']!r}."
        )
        assert 'fhv-laptop' in entry_task_ids, (
            f"Laptop entry missing from snapshot entries: {entry_task_ids}."
        )
        # Laptop must be at position 1 (after the finalizing head).
        assert snap['entries'][1]['task_id'] == 'fhv-laptop', (
            f"Expected entries[1]['task_id']=='fhv-laptop', got {snap['entries'][1]['task_id']!r}."
        )

    async def test_no_finalizing_head_unchanged(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """When _finalizing_head is None, snapshot() is byte-identical to before.

        GREEN both before and after step-2 (regression guard).
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        req_a = _make_req('fhv-only', 'task/fhv-only', config, git_repo)
        laptop_lease = HostLease(name='laptop', runner=MagicMock(), is_local=False)
        entry_a = _make_entry(req_a, laptop_lease, phase='verifying', started_at=time.time() - 2)

        worker._inflight.append(entry_a)
        # _finalizing_head intentionally left at default (None after step-2 or unset before)

        snap = worker.snapshot()

        assert snap['depth'] == 1
        assert snap['head_of_line'] == 'fhv-only'
        assert snap['entries'][0]['task_id'] == 'fhv-only'


# ── step-3: snapshot() occupancy ─────────────────────────────────────────────


@pytest.mark.asyncio
class TestSnapshotFinalizingHeadOccupancy:
    """snapshot()['occupancy']['by_host'] must include the finalizing head's host.

    RED until step-4 GREEN patches the occupancy section.

    State: local_head in _finalizing_head, laptop_entry in _inflight.
    Allocator has host_names=['local','laptop'].
    by_host must contain both 'local' and 'laptop'; hosts_busy must be 2.
    """

    async def test_occupancy_includes_finalizing_host(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """by_host must map 'local' → local task_id when local is finalizing.

        RED: by_host built only from _inflight; 'local' absent; hosts_busy==1.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = _make_mock_allocator(['local', 'laptop'])

        req_local = _make_req('occ-local', 'task/occ-local', config, git_repo)
        req_laptop = _make_req('occ-laptop', 'task/occ-laptop', config, git_repo)

        local_lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        laptop_lease = HostLease(name='laptop', runner=MagicMock(), is_local=False)

        local_head = _make_entry(req_local, local_lease)
        laptop_entry = _make_entry(req_laptop, laptop_lease)

        worker._inflight.append(laptop_entry)
        worker._finalizing_head = local_head  # type: ignore[attr-defined]

        snap = worker.snapshot()
        occ = snap['occupancy']

        assert 'local' in occ['by_host'], (
            f"'local' missing from by_host={occ['by_host']}. "
            "RED: by_host built from _inflight only; finalizing head not included."
        )
        assert occ['by_host']['local'] == 'occ-local', (
            f"Expected by_host['local']=='occ-local', got {occ['by_host'].get('local')!r}."
        )
        assert 'laptop' in occ['by_host'], (
            f"'laptop' missing from by_host={occ['by_host']}."
        )
        assert occ['by_host']['laptop'] == 'occ-laptop', (
            f"Expected by_host['laptop']=='occ-laptop', got {occ['by_host'].get('laptop')!r}."
        )
        assert occ['hosts_busy'] == 2, (
            f"Expected hosts_busy==2, got {occ['hosts_busy']}. "
            "RED: hosts_busy==len(by_host) which is 1 before step-4."
        )
        assert occ['hosts_total'] == 2, (
            f"Expected hosts_total==2, got {occ['hosts_total']}."
        )

    async def test_occupancy_no_finalizing_head_unchanged(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """With no finalizing head, occupancy is unchanged (regression guard).

        GREEN before and after step-4.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = _make_mock_allocator(['local', 'laptop'])

        req_laptop = _make_req('occ-only', 'task/occ-only', config, git_repo)
        laptop_lease = HostLease(name='laptop', runner=MagicMock(), is_local=False)
        laptop_entry = _make_entry(req_laptop, laptop_lease)

        worker._inflight.append(laptop_entry)

        snap = worker.snapshot()
        occ = snap['occupancy']

        assert occ['hosts_busy'] == 1
        assert 'laptop' in occ['by_host']
        assert occ['by_host']['laptop'] == 'occ-only'


# ── step-5: snapshot() verify_in_progress ────────────────────────────────────


@pytest.mark.asyncio
class TestSnapshotFinalizingHeadVerifyInProgress:
    """snapshot()['verify_in_progress'] must reflect the finalizing head.

    RED until step-6 GREEN changes the head-selection logic.

    State: local_head in _finalizing_head (phase='verifying'),
    laptop_entry in _inflight (phase='verifying').
    verify_in_progress must report the finalizing head's task_id, not laptop's.
    """

    async def test_verify_in_progress_reports_finalizing_head(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """verify_in_progress['task_id'] must be the finalizing head, not _inflight[0].

        RED: verify_in_progress computed from _inflight[0] (laptop); reports laptop.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        req_local = _make_req('vip-local', 'task/vip-local', config, git_repo)
        req_laptop = _make_req('vip-laptop', 'task/vip-laptop', config, git_repo)

        local_lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        laptop_lease = HostLease(name='laptop', runner=MagicMock(), is_local=False)

        started = time.time() - 3.0
        local_head = _make_entry(req_local, local_lease, phase='verifying', started_at=started)
        laptop_entry = _make_entry(req_laptop, laptop_lease, phase='verifying')

        worker._inflight.append(laptop_entry)
        worker._finalizing_head = local_head  # type: ignore[attr-defined]

        snap = worker.snapshot()

        assert snap['verify_in_progress'] is not None, (
            "Expected verify_in_progress to be non-None (local head is verifying)."
        )
        assert snap['verify_in_progress']['task_id'] == 'vip-local', (
            f"Expected verify_in_progress['task_id']=='vip-local', "
            f"got {snap['verify_in_progress'].get('task_id')!r}. "
            "RED: verify_in_progress uses _inflight[0] (laptop) before step-6."
        )
        assert snap['verify_in_progress']['phase'] == 'verifying', (
            f"Expected phase=='verifying', got {snap['verify_in_progress'].get('phase')!r}."
        )

    async def test_verify_in_progress_no_finalizing_head_unchanged(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """With no finalizing head, verify_in_progress still comes from _inflight[0].

        GREEN before and after step-6 (regression guard).
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        req_a = _make_req('vip-only', 'task/vip-only', config, git_repo)
        local_lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry_a = _make_entry(req_a, local_lease, phase='verifying', started_at=time.time() - 1)

        worker._inflight.append(entry_a)

        snap = worker.snapshot()

        assert snap['verify_in_progress'] is not None
        assert snap['verify_in_progress']['task_id'] == 'vip-only'


# ── step-7: _finalize_inflight sets/clears _finalizing_head ──────────────────


@pytest.mark.asyncio
class TestFinalizingHeadLifecycle:
    """_finalize_inflight must set _finalizing_head at entry and clear in finally.

    RED until step-8 GREEN wraps _finalize_inflight body in a try/finally that
    sets/clears self._finalizing_head.

    Uses a gated verify_task (FAIL path, no CAS advance needed) to pause inside
    _finalize_inflight at the `await entry.verify_task` point, assert the field is
    set, then release the gate and assert it clears.
    """

    async def _make_merged_item(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        branch: str,
        filename: str,
        content: str,
    ):
        """Build a real merged item so _finalize_inflight's merge_result path works."""
        from orchestrator.git_ops import _run as _git_run
        from orchestrator.merge_queue import SpeculativeItem

        # Create branch with file
        worktree = (await git_ops.create_worktree(branch)).path
        (worktree / filename).write_text(content)
        await git_ops.commit(worktree, f'Add {filename}')

        loop = asyncio.get_event_loop()
        req = MergeRequest(
            task_id=branch,
            branch=branch,
            worktree=worktree,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
            lane='normal',
        )
        merge_result = await git_ops.merge_to_main(worktree, branch)
        assert merge_result.success
        base_sha = await git_ops.get_main_sha()
        item = SpeculativeItem(
            request=req,
            merge_result=merge_result,
            merge_wt=merge_result.merge_worktree,
            base_sha=base_sha,
            speculative=False,
            skip_verify=False,
        )
        return req, item

    async def test_finalizing_head_set_during_await_cleared_after(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """_finalizing_head is set during `await entry.verify_task` and cleared in finally.

        RED: _finalize_inflight never assigns self._finalizing_head, so mid-await
        assertion (worker._finalizing_head is entry) fails immediately.
        """
        gate = asyncio.Event()

        async def _gated_fail_verify() -> InflightVerifyResult:
            await gate.wait()
            return InflightVerifyResult(
                outcome=MergeOutcome('blocked', reason='gated-test'),
                merge_wt=None,
            )

        req, item = await self._make_merged_item(
            git_ops, config, 'fhl-gate', 'fhl.py', 'x = 1\n',
        )

        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        alloc = MagicMock()
        alloc.release = AsyncMock()
        alloc.cancel_and_release = AsyncMock()
        worker._host_allocator = alloc

        local_lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        verify_task = asyncio.ensure_future(_gated_fail_verify())
        entry = InflightEntry(
            item=item,
            lease=local_lease,
            verify_task=verify_task,
            merge_wt=None,
            was_speculative=False,
            phase='verifying',
            started_at=time.time() - 1.0,
        )

        # Launch _finalize_inflight; it will pause at `await entry.verify_task`
        fin = asyncio.ensure_future(worker._finalize_inflight(entry))
        await asyncio.sleep(0)   # yield to let _finalize_inflight reach the await

        assert worker._finalizing_head is entry, (  # type: ignore[attr-defined]
            f"Expected worker._finalizing_head is entry after yield; "
            f"got {worker._finalizing_head!r}. "  # type: ignore[attr-defined]
            "RED: _finalize_inflight never assigns _finalizing_head."
        )
        snap_mid = worker.snapshot()
        assert snap_mid['head_of_line'] == req.task_id, (
            f"Expected head_of_line=={req.task_id!r} mid-await; "
            f"got {snap_mid['head_of_line']!r}."
        )

        # Release the gate so _finalize_inflight can complete.
        gate.set()
        await fin

        assert worker._finalizing_head is None, (  # type: ignore[attr-defined]
            f"Expected worker._finalizing_head is None after _finalize_inflight; "
            f"got {worker._finalizing_head!r}. "  # type: ignore[attr-defined]
            "RED: finally clause not added yet."
        )
        assert req.result.done(), "Expected req.result to be resolved after FAIL path."
