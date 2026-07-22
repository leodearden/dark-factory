"""INV-3 chain-intact verdict enforcement (PRD plans/merge-verdict-integrity-prd.md leaf γ).

A verdict (PASS or FAIL) is adoptable iff, at adoption time, its verified
tree still equals ``current-main ∘ S`` for an ordered sequence S of
STILL-LIVE pipeline items (none failed/ejected/superseded/re-merged since
verify dispatch).  A broken-chain verdict is VOID: emit ``verdict_voided``
naming the dead link, re-merge + re-verify the item, and a FAIL from a dead
base NEVER blocks the task.  Deliberately verify-depth-agnostic: chain-intact,
NOT base==main.

This file is the boundary-test leaf (PRD §7).  It reuses the merge-queue unit
harness by bare module-name sibling import (no ``tests/__init__.py``):
  - ``_make_worker`` / ``_make_fake_item`` — pure builders (test_merge_queue_two_layer_integration).

Step-1 (this commit) covers the pure predicate + recorder foundation:
  * ``_chain_dead_link(item, main_sha)`` — the single shared chain-intact predicate.
  * ``_record_dead_base(*shas)`` — the bounded-FIFO dead-commit ledger.
Adoption / dispatch / cascade enforcement land in later steps.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run

# Bare sibling-import of the pure builders (no tests/__init__.py — tests dir is
# on sys.path; mirrors test_concurrent_verify_boundary importing from
# test_merge_queue_concurrent_verify).
from test_merge_queue_two_layer_integration import _make_fake_item, _make_worker

# ── Fixtures (mirrored from test_merge_queue_verify_base_invariant.py) ────────


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


# ── _chain_dead_link predicate ───────────────────────────────────────────────


@pytest.mark.asyncio
class TestChainDeadLinkPredicate:
    """The single shared chain-intact predicate: a chain is dead iff the item's
    ``base_sha`` is a known-dead commit AND is not current main.  Depth-agnostic:
    only a DEAD base breaks the chain; a live deep base does not."""

    async def test_dead_base_not_main_returns_base(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """base_sha in the dead set and != main → broken chain, return the dead link."""
        worker = _make_worker(git_ops)
        _, item = _make_fake_item(
            't1', base_sha='deadcommit', merge_commit='mc1', config=config, git_repo=git_repo,
        )
        worker._dead_base_commits.add('deadcommit')
        assert worker._chain_dead_link(item, 'livemain') == 'deadcommit'

    async def test_base_equals_main_returns_none(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """base_sha == main → the base LANDED, chain intact, even if once recorded dead."""
        worker = _make_worker(git_ops)
        _, item = _make_fake_item(
            't2', base_sha='mainsha', merge_commit='mc2', config=config, git_repo=git_repo,
        )
        worker._dead_base_commits.add('mainsha')
        assert worker._chain_dead_link(item, 'mainsha') is None

    async def test_live_deep_base_returns_none(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """Verify-depth-agnostic: a live predecessor merge_commit (NOT dead) is intact."""
        worker = _make_worker(git_ops)
        _, item = _make_fake_item(
            't3', base_sha='livedeep', merge_commit='mc3', config=config, git_repo=git_repo,
        )
        # livedeep is NOT in the dead set → a deep multi-merge chain is intact.
        assert worker._chain_dead_link(item, 'livemain') is None

    async def test_decided_item_returns_none(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A DecidedItem/passthrough has no verified tree to adopt → always None."""
        worker = _make_worker(git_ops)
        _, item = _make_fake_item(
            't4', base_sha='deadcommit', merge_commit=None, config=config, git_repo=git_repo,
        )
        worker._dead_base_commits.add('deadcommit')
        assert worker._chain_dead_link(item, 'livemain') is None


# ── _record_dead_base bounded FIFO ledger ────────────────────────────────────


class TestRecordDeadBase:
    """The dead-commit ledger is a bounded FIFO: past the cap the oldest recorded
    commit is evicted; empty/blank/duplicate SHAs are no-ops."""

    def test_record_and_bounded_fifo_eviction(self, git_ops: GitOps) -> None:
        worker = _make_worker(git_ops)
        worker._dead_base_commits_cap = 3
        # cap+1 distinct SHAs → oldest (c0) evicted; size clamps to cap.
        worker._record_dead_base('c0', 'c1', 'c2', 'c3')
        assert len(worker._dead_base_commits) == 3
        assert 'c0' not in worker._dead_base_commits
        assert {'c1', 'c2', 'c3'} <= worker._dead_base_commits

    def test_record_skips_empty_blank_and_dupes(self, git_ops: GitOps) -> None:
        worker = _make_worker(git_ops)
        # '' and '  ' (strips to empty) are skipped; the duplicate 'x' collapses.
        worker._record_dead_base('', 'x', 'x', '  ')
        assert worker._dead_base_commits == {'x'}
