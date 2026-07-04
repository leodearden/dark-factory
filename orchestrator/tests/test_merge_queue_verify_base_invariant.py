"""Tests for promoting the ε=1890 verify-base guard into two_layer_invariants (I5, task 1999).

Covers:
  step-01 RED — _verify_base_frozen_tip_violations() composed into
                two_layer_invariants(): a frozen entry whose base_sha is not
                the expected frozen-tip base produces a DISTINCTLY-worded
                violation (surfaced by both two_layer_invariants() directly
                and snapshot()['two_layer_invariants']); a HEALTHY multi-entry
                chained frozen prefix produces NO such violation (guards
                against a naive "every entry == newest tip" implementation).

See _warn_if_verify_base_not_frozen_tip (merge_queue.py) for the log-only
dispatch-time guard this promotes to snapshot granularity, and
check_frozen_prefix_invariant for the base-chain math this mirrors.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Literal

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.merge_queue import (
    InflightEntry,
    MergeOutcome,
    MergeRequest,
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
    fake_merge_result = MergeResult(
        success=True,
        merge_commit=merge_commit,
    ) if merge_commit is not None else None
    item = SpeculativeItem(
        request=req,
        merge_result=fake_merge_result,
        merge_wt=Path('/fake/merge-wt') if fake_merge_result is not None else None,
        base_sha=base_sha,
        speculative=False,
        skip_verify=False,
        immediate_outcome=None if fake_merge_result is not None else MergeOutcome('conflict'),
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
        phase='verifying',
    )


def _verify_base_violations(violations: list[str], rid: str) -> list[str]:
    """Filter *violations* to ones with the DISTINCT verify-base/frozen-tip wording naming *rid*.

    Matches /verify.?base/ (case-insensitive) AND 'frozen' AND the rid — a
    pattern the pre-existing 'frozen-prefix base-chain broken' string does not
    satisfy (no 'verify-base' wording), so a non-empty result here proves the
    NEW sub-check fired rather than the pre-existing base-chain check.
    """
    return [
        v for v in violations
        if re.search(r'verify.?base', v, re.IGNORECASE)
        and 'frozen' in v.lower()
        and rid in v
    ]


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
        """A frozen head entry whose base_sha is NOT main_sha → distinct verify-base violation.

        Surfaced both directly via two_layer_invariants() and via
        snapshot()['two_layer_invariants'] (which reads _last_known_main_sha,
        per λ=1895's real-main-not-frozen-tip convention).
        """
        worker = _make_worker(git_ops)
        main_sha = 'M0'
        _, item = _make_fake_item(
            't-stale', base_sha='deadbeef', merge_commit='c1',
            config=config, git_repo=git_repo,
        )
        worker._inflight.append(_make_inflight_entry(item, verifying=True))
        rid = item.request.request_id

        violations = worker.two_layer_invariants(main_sha)
        new_violations = _verify_base_violations(violations, rid)
        assert new_violations, (
            f'expected a distinct verify-base/frozen-tip violation naming {rid!r}, '
            f'got: {violations}'
        )
        assert not any('base-chain broken' in v for v in new_violations), (
            'the new verify-base violation must be distinctly worded from the '
            f'pre-existing base-chain check, got: {new_violations}'
        )

        # snapshot()['two_layer_invariants'] must surface the same violation —
        # set _last_known_main_sha so snapshot() computes against real main.
        worker._last_known_main_sha = main_sha
        snap_violations = worker.snapshot()['two_layer_invariants']
        assert _verify_base_violations(snap_violations, rid), (
            f'expected snapshot()["two_layer_invariants"] to surface the same '
            f'verify-base violation, got: {snap_violations}'
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

        violations = worker.two_layer_invariants(main_sha)
        assert violations == [], (
            f'expected a healthy chained frozen prefix to have no violations, got: {violations}'
        )
