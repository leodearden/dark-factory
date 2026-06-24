"""Tests for the η=1892 bounce/rebase layer of the two-layer merge queue.

Covers:
  pre-1        — scaffolding: fixtures + helpers (no test bodies yet)
  step-01 RED  — recompute probes suffix vs FROZEN TIP, not bare main
  step-03 RED  — bounce primitives exist (NEEDS_REBASE_REASON_PREFIX,
                 MERGE_BOUNCE_CAP, MergeBounceRegistry)
  step-05 RED  — _bounce_conflicting_suffix_items() clean-rebase path re-queues
  step-07 RED  — _bounce_conflicting_suffix_items() escalation paths
  step-09 RED  — _acquire_next_request() wires bounce at graph time
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.merge_queue import (
    InflightEntry,
    MergeRequest,
    SpeculativeItem,
    SpeculativeMergeWorker,
    SuffixConflictGraph,
)

# ── Module-level sentinel (mirrors test_merge_queue_frozen_prefix.py) ────────
_SENTINEL_VERIFY_TASK = object()  # noqa: PD901


# ── Fixtures ──────────────────────────────────────────────────────────────────


async def _setup_repo(repo: Path) -> None:
    """Initialise a bare git repo with shared.txt and README.md on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    (repo / 'shared.txt').write_text('line1\nline2\nline3\n')
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


# ── Module-level helpers ──────────────────────────────────────────────────────


def _make_req(
    task_id: str,
    branch: str,
    config: OrchestratorConfig,
    git_repo: Path,
    lane: Literal['normal', 'high'] = 'normal',
    *,
    merge_first_enqueued_at: float | None = 1000.0,
) -> MergeRequest:
    """Build a minimal MergeRequest with a fresh event-loop future.

    Must be called from within an async context (asyncio.get_running_loop()).
    """
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=git_repo,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane=lane,
        merge_first_enqueued_at=merge_first_enqueued_at,
    )


def _make_worker(git_ops: GitOps) -> SpeculativeMergeWorker:
    """Build a bare SpeculativeMergeWorker for unit tests (no harness wiring)."""
    return SpeculativeMergeWorker(git_ops, asyncio.Queue())


def _make_fake_item(
    task_id: str,
    *,
    base_sha: str = 'aaaa0000',
    merge_commit: str | None = 'bbbb1111',
    config: OrchestratorConfig,
    git_repo: Path,
) -> tuple[MergeRequest, SpeculativeItem]:
    """Build a (MergeRequest, SpeculativeItem) pair from fake SHAs.

    Does NOT create real git branches — suitable for pure-unit tests that
    only exercise frozen-prefix/frozen_prefix_tip accessor methods without
    real git operations. Must be called from an async context (for the future).
    """
    req = _make_req(task_id, f'task/{task_id}', config, git_repo)
    fake_merge_result = (
        MergeResult(
            success=True,
            merge_commit=merge_commit,
        )
        if merge_commit is not None
        else None
    )
    item = SpeculativeItem(
        request=req,
        merge_result=fake_merge_result,
        merge_wt=None,
        base_sha=base_sha,
        speculative=False,
        skip_verify=False,
    )
    return req, item


def _make_inflight_entry(
    item: SpeculativeItem,
    *,
    verifying: bool = True,
) -> InflightEntry:
    """Build an InflightEntry for unit tests.

    verifying=True  → verify_task set to _SENTINEL_VERIFY_TASK (any non-None
                       object; frozen_prefix() only checks `is not None`).
    verifying=False → verify_task=None (passthrough entry; excluded from the
                       frozen prefix by §5.3 definition).
    """
    return InflightEntry(
        item=item,
        lease=None,
        verify_task=_SENTINEL_VERIFY_TASK if verifying else None,  # type: ignore[arg-type]
        merge_wt=None,
        was_speculative=False,
        phase='verifying',
    )


async def _create_branch_editing(
    repo: Path,
    branch_name: str,
    filename: str,
    content: str,
    base_branch: str = 'main',
) -> str:
    """Create a branch editing filename with content; return the branch SHA."""
    await _run(['git', 'checkout', '-b', branch_name], cwd=repo)
    (repo / filename).write_text(content)
    await _run(['git', 'add', filename], cwd=repo)
    await _run(['git', 'commit', '-m', f'Edit {filename} in {branch_name}'], cwd=repo)
    _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    await _run(['git', 'checkout', base_branch], cwd=repo)
    return sha.strip()


async def _create_frozen_tip_commit(
    repo: Path,
    filename: str,
    content: str,
) -> str:
    """Commit a change to main (simulating a frozen-prefix merge commit).

    Directly commits on main (no branch switch needed — we're already on main
    after _setup_repo).  Returns the resulting commit SHA.
    """
    (repo / filename).write_text(content)
    await _run(['git', 'add', filename], cwd=repo)
    await _run(['git', 'commit', '-m', f'Frozen-tip commit: {filename}'], cwd=repo)
    _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    return sha.strip()


# ── step-01 RED: recompute probes suffix vs FROZEN TIP, not bare main ─────────


@pytest.mark.asyncio
class TestFrozenTipProbe:
    """recompute_suffix_conflict_graph() §7 must use frozen_prefix_tip() as the
    probe base, not bare main_sha.

    The RED case: frozen prefix has an entry whose merge_commit T is NOT the
    same as the current main HEAD (M0).  A suffix item S is clean vs M0 but
    conflicts with T.

    - Current (bare-main) probe: S vs M0 → clean → NOT in conflicts_with_main.
      The test asserts IN → FAILS (RED) until step-02 repoints the base.
    - After fix (frozen-tip) probe: S vs T → conflict → IN conflicts_with_main
      → PASSES (GREEN).

    CONTROL: empty frozen prefix → frozen_prefix_tip == main_sha == M0 → S
    clean vs M0 → NOT in conflicts_with_main (both before and after fix).

    RED until step-02 GREEN repoints the §7 probe base to frozen_prefix_tip().
    """

    async def test_suffix_item_conflicts_with_frozen_tip(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """S is clean vs bare main (M0) but conflicts with the frozen tip T.

        Setup:
          M0 = initial commit (shared.txt = 'line1\\nline2\\nline3\\n')
          T  = commit on a side branch 'frozen-tip' that edits shared.txt line2
               → 'FROZEN-LINE2'.  T's SHA is used as the frozen item's
               merge_commit.  main HEAD stays at M0.
          S  = suffix branch off M0 editing shared.txt line2 → 'SUFFIX-LINE2'.

        S vs M0 → CLEAN (only S changed line2; M0 has the original).
        S vs T  → CONFLICT (both S and T changed line2 to different values).
        """
        # Record the original main SHA (M0).
        _, m0_sha_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        m0_sha = m0_sha_raw.strip()

        # 1. Create T on a side branch (NOT on main) so main_sha stays == M0.
        await _run(['git', 'checkout', '-b', 'frozen-tip'], cwd=git_repo)
        (git_repo / 'shared.txt').write_text('line1\nFROZEN-LINE2\nline3\n')
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'Frozen-tip: edit line2'], cwd=git_repo)
        _, t_sha_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        frozen_tip_sha = t_sha_raw.strip()
        # Return to main — main HEAD is still M0.
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        # Sanity: main must still be at M0 (frozen-tip commit is NOT on main).
        _, cur_main_raw, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        assert cur_main_raw.strip() == m0_sha, 'main HEAD must stay at M0 for this test'

        # 2. Create suffix branch S from M0 that edits line2 to 'SUFFIX-LINE2'.
        #    S vs M0 = clean (no conflicting edit on M0).
        #    S vs T  = conflict (both S and T edited line2).
        await _run(['git', 'checkout', '-b', 'task/591'], cwd=git_repo)
        (git_repo / 'shared.txt').write_text('line1\nSUFFIX-LINE2\nline3\n')
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'S edits line2'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        # 3. Build worker; populate frozen prefix with T as merge_commit.
        worker = _make_worker(git_ops)
        _, item_t = _make_fake_item(
            't-frozen', merge_commit=frozen_tip_sha, config=config, git_repo=git_repo,
        )
        entry_t = _make_inflight_entry(item_t, verifying=True)
        worker._inflight.append(entry_t)

        # 4. Put suffix request S into the lane buffer.
        req_s = _make_req('591', '591', config, git_repo)
        worker._lane_buffers['normal'].append(req_s)

        # 5. Recompute.
        await worker.recompute_suffix_conflict_graph()

        # 6. S conflicts with the FROZEN TIP → must appear in conflicts_with_main.
        #    RED: current code probes S vs bare M0 → S is clean → assertion fails.
        #    GREEN: probe repointed to T → S conflicts → assertion passes.
        assert req_s.request_id in worker._suffix_conflict_graph.conflicts_with_main, (
            'Expected S to conflict with the frozen tip T but it was NOT in '
            'conflicts_with_main.  The §7 probe base must be frozen_prefix_tip(), '
            'not bare main_sha.  S is clean vs M0 (bare main) but conflicts with T.'
        )

    async def test_empty_frozen_prefix_uses_bare_main(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """CONTROL: empty frozen prefix → frozen_prefix_tip() == main_sha → a
        suffix item clean vs bare main is NOT in conflicts_with_main (both
        before and after the step-02 repoint).
        """
        # Create branch S editing only README.md — clean vs shared.txt on main.
        await _create_branch_editing(
            git_repo, 'task/591-clean', 'README.md', 'clean-suffix\n',
        )

        worker = _make_worker(git_ops)  # no _inflight → frozen prefix empty
        req_s = _make_req('591-clean', '591-clean', config, git_repo)
        worker._lane_buffers['normal'].append(req_s)

        await worker.recompute_suffix_conflict_graph()

        # S is clean vs bare main == frozen_prefix_tip (empty prefix).
        assert req_s.request_id not in worker._suffix_conflict_graph.conflicts_with_main, (
            'S edits only README.md and is clean vs bare main; with an empty '
            'frozen prefix, frozen_prefix_tip() == main_sha, so S must NOT be '
            'flagged in conflicts_with_main.'
        )
