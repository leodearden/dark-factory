"""Tests collapsing redundant phase encodings onto the ItemLifecycle registry
(merge-queue-reliability PRD scope-4 lambda / task 2173, depends on kappa=2169).

Task kappa (task 2169) wired ``ItemLifecycle`` — the request_id-keyed
lifecycle-state registry — at every put/pop call site in
``SpeculativeMergeWorker``. This task (lambda) deletes the two redundant
phase encodings that predate it:

  step-1  RED   — vestigial single-host ``_verify_item``/``_verify_phase``/
                  ``_verify_started_at`` worker fields are gone (API-surface
                  contract; the fields are write-only, zero reads anywhere)
  step-2  GREEN — the fields and their six write sites deleted (pure
                  dead-code removal)
  step-3  RED   — snapshot()/frozen_prefix() derive phase from the
                  ItemLifecycle registry, not a stored ``InflightEntry.phase``
  step-4  GREEN — ``SpeculativeMergeWorker._entry_phase()`` added; all 5
                  readers repointed onto the registry (``InflightEntry.phase``
                  kept as a fallback source — zero existing-test churn)
  step-5  RED   — ``InflightEntry`` no longer carries a free-form phase field
  step-6  GREEN — ``InflightEntry.phase`` field deleted; existing tests
                  reconciled to drive registry state instead

This module imports orchestrator.merge_queue LOCALLY inside each test method
(not at module scope) so a not-yet-implemented symbol never breaks collection
of the rest of the file during the RED steps — mirrors
test_merge_queue_lifecycle_registry.py's / test_merge_queue_request_liveness.py's
convention.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run

# ---------------------------------------------------------------------------
# Fixtures + helpers (per-file duplication convention — see
# test_merge_queue_lifecycle_registry.py; there is no shared worker conftest).
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Initialise a bare git repository with a single commit on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
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


def _make_worker(git_ops: GitOps):
    """Build a bare SpeculativeMergeWorker for unit tests (no harness wiring).

    Mirrors test_merge_queue_lifecycle_registry.py's / test_merge_queue_
    invariant_integration_gate.py:212's ``_make_worker``.
    """
    from orchestrator.merge_queue import SpeculativeMergeWorker

    return SpeculativeMergeWorker(git_ops, asyncio.Queue())


# ---------------------------------------------------------------------------
# step-1 RED / step-2 GREEN: vestigial single-host observability fields
# ---------------------------------------------------------------------------


class TestVestigialVerifyFieldsRemoved:
    """The single-host ``_verify_item``/``_verify_phase``/``_verify_started_at``
    fields are write-only after epsilon (merge_queue.py:5085-5088) with ZERO
    reads anywhere in the tree — deleting them is pure dead-code removal that
    structurally eliminates the stale-phantom-snapshot-entry class described
    at merge_queue.py:6900-6903 (task lambda / task 2173 step-1).

    RED until step-2 GREEN deletes the field declarations (currently
    merge_queue.py:5089-5091) and their six write sites.
    """

    def test_vestigial_fields_absent(self, git_ops: GitOps) -> None:
        worker = _make_worker(git_ops)

        assert not hasattr(worker, '_verify_item'), (
            'worker._verify_item must be deleted — it is write-only '
            '(zero reads anywhere in the tree).'
        )
        assert not hasattr(worker, '_verify_phase'), (
            'worker._verify_phase must be deleted — it is write-only '
            '(zero reads anywhere in the tree).'
        )
        assert not hasattr(worker, '_verify_started_at'), (
            'worker._verify_started_at must be deleted — it is write-only '
            '(zero reads anywhere in the tree).'
        )
