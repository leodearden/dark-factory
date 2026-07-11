"""Tests for promoting the ε=1890 verify-base guard into two_layer_invariants (I5, task 1999).

Covers:
  step-01 RED — _verify_base_frozen_tip_violations() composed into
                two_layer_invariants(): a frozen entry whose base_sha is not
                the expected frozen-tip base produces a violation, verified
                structurally (via a direct call to the helper, then asserting
                two_layer_invariants() and snapshot()['two_layer_invariants']
                compose that same output) rather than by matching violation
                message wording — the two checks are intentionally worded
                distinctly (see merge_queue.py docstrings) but that wording is
                cosmetic and free to change without breaking these tests. A
                HEALTHY multi-entry chained frozen prefix produces NO such
                violation (guards against a naive "every entry == newest tip"
                implementation).

See _warn_if_verify_base_not_frozen_tip (merge_queue.py) for the log-only
dispatch-time guard this promotes to snapshot granularity, and
check_frozen_prefix_invariant for the base-chain math this mirrors.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.merge_queue import (
    DecidedItem,
    InflightEntry,
    MergeOutcome,
    MergeRequest,
    RealMergeItem,
    SpeculativeItem,
    SpeculativeMergeWorker,
)

# ── Module-level sentinel for verify_task in pure unit tests ─────────────────
#
# frozen_prefix() only checks `e.verify_task is not None`, so any non-None
# object doubles as a verifying-entry marker (mirrors test_merge_queue_frozen_prefix.py).
_SENTINEL_VERIFY_TASK = object()  # noqa: PD901


# ── Fixtures (mirrored from test_merge_queue_frozen_prefix.py) ───────────────


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


# ── Module-level helpers (mirrored from test_merge_queue_frozen_prefix.py) ───


def _make_req(
    task_id: str,
    branch: str,
    config: OrchestratorConfig,
    git_repo: Path,
    lane: Literal['normal', 'high'] = 'normal',
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

    Does NOT create real git branches — suitable for pure-unit tests that only
    exercise the accessor/invariant methods, mirroring
    test_merge_queue_frozen_prefix.py's helper of the same name.
    """
    req = _make_req(task_id, f'task/{task_id}', config, git_repo)
    item: SpeculativeItem
    if merge_commit is not None:
        item = RealMergeItem(
            request=req,
            merge_result=MergeResult(success=True, merge_commit=merge_commit),
            merge_wt=Path('/fake/merge-wt'),
            base_sha=base_sha,
            speculative=False,
        )
    else:
        item = DecidedItem(
            request=req,
            immediate_outcome=MergeOutcome('conflict'),
            base_sha=base_sha,
            speculative=False,
        )
    return req, item


def _make_inflight_entry(
    item: SpeculativeItem,
    *,
    verifying: bool = True,
) -> InflightEntry:
    """Build an InflightEntry for unit tests (verifying=True → frozen)."""
    return InflightEntry(
        item=item,
        lease=None,
        verify_task=_SENTINEL_VERIFY_TASK if verifying else None,  # type: ignore[arg-type]
        merge_wt=None,
        was_speculative=False,
    )


# ── step-01 RED: _verify_base_frozen_tip_violations() composed into two_layer_invariants ──


@pytest.mark.asyncio
class TestVerifyBaseFrozenTipPromotion:
    """Promote the ε=1890 dispatch-time guard predicate to snapshot granularity (task 1999 I5).

    RED until step-02 GREEN adds _verify_base_frozen_tip_violations() and
    composes it into two_layer_invariants() as sub-check (iii).
    """

    async def test_stale_verify_base_produces_distinct_violation(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A frozen head entry whose base_sha is NOT main_sha → dedicated verify-base violation.

        Asserts a STRUCTURAL signal that the new sub-check fired — calling
        :meth:`_verify_base_frozen_tip_violations` directly — rather than
        inferring it from violation-message wording (regex-matching prose is
        fragile: it breaks on a legitimate reword and would pass even if the
        wrong check produced the matching words). :meth:`two_layer_invariants`
        and ``snapshot()['two_layer_invariants']`` (which reads
        _last_known_main_sha, per λ=1895's real-main-not-frozen-tip
        convention) are each checked to COMPOSE that same helper output
        verbatim, proving the wiring rather than the wording.
        """
        worker = _make_worker(git_ops)
        main_sha = 'M0'
        _, item = _make_fake_item(
            't-stale', base_sha='deadbeef', merge_commit='c1',
            config=config, git_repo=git_repo,
        )
        worker._inflight.append(_make_inflight_entry(item, verifying=True))
        rid = item.request.request_id

        # Structural signal: call the dedicated helper directly instead of
        # grepping combined output for wording that "looks like" the new check.
        direct_violations = worker._verify_base_frozen_tip_violations(main_sha)
        assert direct_violations, (
            f'expected _verify_base_frozen_tip_violations() to flag {rid!r} directly, '
            f'got: {direct_violations}'
        )
        assert all(rid in v for v in direct_violations), (
            f'expected every violation to name {rid!r}, got: {direct_violations}'
        )

        # two_layer_invariants() must compose the helper's own output verbatim —
        # a behavioral proof that sub-check (iii) is wired in, independent of
        # whatever wording either check happens to use.
        violations = worker.two_layer_invariants(main_sha)
        assert all(v in violations for v in direct_violations), (
            f'expected two_layer_invariants() to include every violation returned by '
            f'_verify_base_frozen_tip_violations(), got: {violations}'
        )

        # snapshot()['two_layer_invariants'] must surface the same violation(s) —
        # set _last_known_main_sha so snapshot() computes against real main.
        worker._last_known_main_sha = main_sha
        snap_violations = worker.snapshot()['two_layer_invariants']
        assert all(v in snap_violations for v in direct_violations), (
            f'expected snapshot()["two_layer_invariants"] to surface the same '
            f'verify-base violation(s), got: {snap_violations}'
        )

    async def test_healthy_chained_prefix_has_no_verify_base_violation(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A HEALTHY 2-entry chained frozen prefix must NOT trip the verify-base check.

        entry0.base==main_sha, entry0.merge_commit==C1; entry1.base==C1,
        entry1.merge_commit==C2.  Guards against a naive "every entry.base_sha
        == frozen_prefix_tip(main_sha)" implementation, which would falsely
        flag entry0 (base=main_sha != newest tip C2).
        """
        worker = _make_worker(git_ops)
        main_sha = 'M0'
        _, item_0 = _make_fake_item(
            't-0', base_sha=main_sha, merge_commit='C1',
            config=config, git_repo=git_repo,
        )
        _, item_1 = _make_fake_item(
            't-1', base_sha='C1', merge_commit='C2',
            config=config, git_repo=git_repo,
        )
        worker._inflight.append(_make_inflight_entry(item_0, verifying=True))
        worker._inflight.append(_make_inflight_entry(item_1, verifying=True))

        # Direct call isolates the dedicated helper from the other two
        # sub-checks composed into two_layer_invariants() below.
        assert worker._verify_base_frozen_tip_violations(main_sha) == [], (
            'expected _verify_base_frozen_tip_violations() to report no violations '
            'for a healthy chained frozen prefix'
        )

        violations = worker.two_layer_invariants(main_sha)
        assert violations == [], (
            f'expected a healthy chained frozen prefix to have no violations, got: {violations}'
        )
