"""Real-git tests for the deep merge-ahead chain builder (PRD β, task 3184).

Covers:
  step-01 RED  — ChainResult dataclass contract (fields, defaults, .depth)
  step-03 RED  — GitOps.merge_branch_into_worktree: caller-supplied worktree
  step-05 RED  — merge_branch_into_worktree conflict arm: abort + clean tip
  step-07 RED  — acquire_chain_build_lane / release_chain_build_lane seam
  step-09 RED  — SpeculativeMergeWorker.chain_snapshot() pure reader
  step-11 RED  — build_chain degenerate inputs touch zero worktrees
  step-13 RED  — build_chain builds a clean chain in ONE worktree
  step-15 RED  — build_chain truncation semantics + purity (decision #4)
                 (conflict / train / merge_error reasons, empty-prefix edge)

Reference: ``plans/deep-merge-ahead-prd.md``.

Harness notes (see plan pre-1):
  * ``orchestrator/pyproject.toml`` does NOT set ``asyncio_mode`` → pytest-asyncio
    runs STRICT, so ``@pytest.mark.asyncio`` is required on async test classes.
  * That same config turns "marked with @pytest.mark.asyncio but not an async
    function" into an ERROR — never put a sync ``test_*`` inside a marked class.
  * Default per-test ``timeout = 60``; real-git classes carry
    ``@pytest.mark.timeout(180)`` since a chain of 3 merges plus worktree
    creation can exceed it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import MergeRequest, SpeculativeMergeWorker
from orchestrator.merge_types import GroupMergeRequest, QueuedBranch

# ── repo fixtures (mirrors test_merge_queue_conflict_graph.py:34-50) ──────────


async def _setup_repo(repo: Path) -> None:
    """Init a repo with a 20-line shared.txt plus disjoint.txt.

    ``shared.txt`` gets **20** numbered lines rather than the 3 used by
    test_merge_queue_conflict_graph.py's own ``_setup_repo``: git's 3-line
    diff context window makes near-line edits in a tiny file conflict even
    when they touch different lines (gotcha documented verbatim at
    test_merge_queue_conflict_graph.py:454-460).  20 lines makes a line-1 vs
    line-15 edit pair genuinely non-conflicting, so this file can build both
    conflicting and non-conflicting fixtures from the same seed.
    """
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    (repo / 'shared.txt').write_text(''.join(f'line{i}\n' for i in range(1, 21)))
    (repo / 'disjoint.txt').write_text('aaa\nbbb\nccc\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _add_recording_seed_to_repo(repo: Path) -> None:
    """Commit a recording ``scripts/seed-warm-lane.sh`` into the repo at HEAD.

    Mirrors test_merge_speculation.py:225-248.  Without a COMMITTED seed
    script, ``acquire_spec_lane`` soft-degrades to a cold ephemeral worktree
    (``warm=False``), silently making every "warm ``_spec-`` lane" assertion
    vacuous.  conftest's autouse ``_isolate_warm_lane_script_dir`` pins
    ``ORCH_WARM_LANE_SCRIPT_DIR`` at an absent dir, but a repo-local
    ``scripts/seed-warm-lane.sh`` still resolves first — which is why
    committing it here works.
    """
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / 'seed-warm-lane.sh'
    script.write_text(
        '#!/usr/bin/env bash\n'
        '# argv: <base_target> <lane_dir> <mode>\n'
        'ARGV_FILE="$2/scripts/seed-warm-lane.sh.argv"\n'
        'echo "$@" >> "$ARGV_FILE"\n',
    )
    script.chmod(0o755)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'add recording seed-warm-lane.sh'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    asyncio.run(_add_recording_seed_to_repo(repo))
    return repo


# ── config / GitOps helpers (mirrors test_merge_speculation.py:161-176) ───────


def _make_spec_git_config(*, on: bool = True, **extra) -> GitConfig:
    """Build a GitConfig with ``merge_spec_warm_lane_pool=on``."""
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        merge_spec_warm_lane_pool=on,
        **extra,
    )


def _make_git_ops(repo: Path, *, pool: bool = True, size: int = 1) -> GitOps:
    """Build a GitOps over *repo* with (or without) a ``_spec-`` warm lane pool.

    ``merge_spec_warm_lane_pool_size`` is a **GitOps constructor kwarg**, not a
    GitConfig field (git_ops.py:2041).  The pool is only constructed when
    ``size > 0 AND config.merge_spec_warm_lane_pool`` (git_ops.py:2093);
    otherwise ``git_ops.spec_warm_lane_pool is None``.
    """
    return GitOps(
        _make_spec_git_config(on=pool), repo, merge_spec_warm_lane_pool_size=size,
    )


def _make_config(repo: Path, git_config: GitConfig | None = None) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=repo, git=git_config or _make_spec_git_config(),
    )


# ── MergeRequest helpers (mirrors test_merge_queue_conflict_graph.py:74-96) ───


def _make_req(
    task_id: str,
    branch: str,
    config: OrchestratorConfig,
    git_repo: Path,
    *,
    lane: Literal['normal', 'high'] = 'normal',
) -> MergeRequest:
    """Build a minimal MergeRequest with a fresh event-loop future.

    CAUTION: builds its future via ``asyncio.get_running_loop()``, so this can
    ONLY be called from inside an async test — never at module/fixture scope.
    *branch* is the BARE suffix (``'101'``), not the prefixed name.
    """
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        worktree=git_repo,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane=lane,
    )


def _make_group_req(
    task_id: str,
    branch: str,
    config: OrchestratorConfig,
    git_repo: Path,
    *,
    member_task_ids: list[str] | None = None,
    lane: Literal['normal', 'high'] = 'normal',
) -> GroupMergeRequest:
    """Build a minimal GroupMergeRequest (train) for the truncation test."""
    qb = QueuedBranch.parse(branch, config.git.branch_prefix)

    async def _status_check(_ids: list[str]) -> dict[str, str]:
        return {}

    async def _mark_member_done(_tid: str, _sha: str) -> None:
        return None

    return GroupMergeRequest(
        task_id=task_id,
        branch=qb,
        worktree=git_repo,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane=lane,
        train_id=f'train-{task_id}',
        member_task_ids=member_task_ids or [task_id],
        tip_branch=qb,
        tip_task_id=task_id,
        status_check=_status_check,
        mark_member_done=_mark_member_done,
    )


def _make_worker(git_ops: GitOps) -> SpeculativeMergeWorker:
    """Build a bare SpeculativeMergeWorker for unit tests (no harness wiring)."""
    return SpeculativeMergeWorker(git_ops, asyncio.Queue())


# ── git helpers ──────────────────────────────────────────────────────────────


async def _create_branch_editing(
    repo: Path,
    branch_name: str,
    filename: str,
    content: str,
    base_branch: str = 'main',
) -> str:
    """Create a branch that edits *filename* with *content*; return its SHA.

    Copied out of ``TestRecomputeFootprintEdges._create_branch_editing``
    (test_merge_queue_conflict_graph.py:323-338) as a module-level helper.
    Callers pass the FULL prefixed branch name (``'task/101'``), while
    ``_make_req`` takes the bare suffix (``'101'``).
    """
    await _run(['git', 'checkout', '-b', branch_name], cwd=repo)
    (repo / filename).write_text(content)
    await _run(['git', 'add', filename], cwd=repo)
    await _run(['git', 'commit', '-m', f'Edit {filename} in {branch_name}'], cwd=repo)
    _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    await _run(['git', 'checkout', base_branch], cwd=repo)
    return sha.strip()


async def _worktree_names(git_ops: GitOps) -> set[str]:
    """Return the set of registered worktree directory names.

    Copied from test_git_ops_train_solo.py:93-104.
    """
    _, out, _ = await _run(
        ['git', 'worktree', 'list', '--porcelain'],
        cwd=git_ops.project_root,
    )
    names: set[str] = set()
    for line in out.splitlines():
        if line.startswith('worktree '):
            wt_path = Path(line[len('worktree '):].strip())
            names.add(wt_path.name)
    return names


async def _rev_parse(cwd: Path, rev: str = 'HEAD') -> str:
    """Return the stripped SHA of *rev* resolved inside *cwd*."""
    _, sha, _ = await _run(['git', 'rev-parse', rev], cwd=cwd)
    return sha.strip()


def _shared_txt_with(line_no: int, text: str) -> str:
    """Return a 20-line shared.txt body with line *line_no* replaced by *text*."""
    lines = [f'line{i}\n' for i in range(1, 21)]
    lines[line_no - 1] = f'{text}\n'
    return ''.join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# step-01: RED — ChainResult contract
# ═══════════════════════════════════════════════════════════════════════════


class TestChainResultContract:
    """ChainResult carries the PRD β contract shape.

    Plain sync class (no @pytest.mark.asyncio) — none of these need a loop,
    and a sync test inside a marked class is an ERROR under this file's
    filterwarnings config.
    """

    def test_chain_result_importable_from_merge_types(self):
        """ChainResult lives in merge_types, beside the classify_and_merge sum type."""
        from orchestrator.merge_types import ChainResult

        assert ChainResult is not None

    def test_chain_result_reexported_from_merge_queue(self):
        """ChainResult is reachable as orchestrator.merge_queue.ChainResult.

        merge_queue is the module the PRD names and it already re-exports the
        merge_types names through its shim; a missing entry would break the
        `from orchestrator.merge_queue import X` convention every consumer uses.
        """
        from orchestrator.merge_queue import ChainResult as MQChainResult
        from orchestrator.merge_types import ChainResult as MTChainResult

        assert MQChainResult is MTChainResult

    def test_links_and_tip_are_the_only_required_fields(self):
        """ChainResult(links=[], tip='abc123') constructs with no other args."""
        from orchestrator.merge_types import ChainResult

        res = ChainResult(links=[], tip='abc123')
        assert res.links == []
        assert res.tip == 'abc123'

    def test_links_round_trip_task_id_sha_pairs(self):
        """links is a list[tuple[task_id, merge_commit]] in land order."""
        from orchestrator.merge_types import ChainResult

        pairs = [('101', 'sha1'), ('102', 'sha2')]
        res = ChainResult(links=list(pairs), tip='sha2')
        assert res.links == pairs

    def test_optional_field_defaults(self):
        """truncated_at/truncated_reason/lane default None; lane_warm defaults False."""
        from orchestrator.merge_types import ChainResult

        res = ChainResult(links=[], tip='abc123')
        assert res.truncated_at is None
        assert res.truncated_reason is None
        assert res.lane is None
        assert res.lane_warm is False

    def test_field_set_is_pinned(self):
        """The exact field tuple is pinned so a later addition is a deliberate edit."""
        import dataclasses

        from orchestrator.merge_types import ChainResult

        assert tuple(f.name for f in dataclasses.fields(ChainResult)) == (
            'links', 'tip', 'truncated_at', 'truncated_reason', 'lane', 'lane_warm',
        )

    def test_depth_is_link_count(self):
        """.depth == len(links) — the value γ will emit as chain_items."""
        from orchestrator.merge_types import ChainResult

        assert ChainResult(links=[], tip='x').depth == 0
        assert ChainResult(
            links=[('101', 'a'), ('102', 'b')], tip='b',
        ).depth == 2


# ═══════════════════════════════════════════════════════════════════════════
# step-03/05: RED — GitOps.merge_branch_into_worktree
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.timeout(180)
class TestMergeBranchIntoWorktree:
    """merge_branch_into_worktree merges into a CALLER-SUPPLIED worktree.

    The load-bearing divergence from ``merge_to_main``: that method calls
    ``_create_merge_worktree`` (git_ops.py:9319) on EVERY call, provisioning a
    fresh ``_merge-<hex8>`` worktree.  A ``for req in chain: merge_to_main(...)``
    loop would create k worktrees — directly violating this task's
    user-observable signal ("exactly one worktree used").
    """

    async def test_clean_merge_uses_the_supplied_worktree(self, git_repo: Path):
        """A clean merge lands in the passed worktree and creates none of its own."""
        git_ops = _make_git_ops(git_repo)
        await _create_branch_editing(git_repo, 'task/101', 'disjoint.txt', 'edit-101\n')
        main_sha = await _rev_parse(git_repo)

        before = await _worktree_names(git_ops)
        wt = await git_ops.create_throwaway_verify_worktree(main_sha)
        pre_head = await _rev_parse(wt)

        res = await git_ops.merge_branch_into_worktree(wt, '101')

        assert res.success is True
        assert res.conflicts is False
        assert res.merge_commit is not None
        # Stripped at the source — merge_to_main returns raw _run stdout with a
        # trailing newline, so this pins the divergence.
        assert res.merge_commit == res.merge_commit.strip()
        assert len(res.merge_commit) == 40
        # The SAME path passed in — not a fresh one.
        assert res.merge_worktree == wt
        assert res.pre_merge_sha == pre_head

        # ZERO worktrees created beyond the one the test itself made.
        assert await _worktree_names(git_ops) == before | {wt.name}

    async def test_clean_merge_is_a_two_parent_no_ff_merge(self, git_repo: Path):
        """HEAD == merge_commit, HEAD^1 == pre-merge tip, HEAD^2 == branch tip."""
        git_ops = _make_git_ops(git_repo)
        branch_tip = await _create_branch_editing(
            git_repo, 'task/101', 'disjoint.txt', 'edit-101\n',
        )
        main_sha = await _rev_parse(git_repo)
        wt = await git_ops.create_throwaway_verify_worktree(main_sha)
        pre_head = await _rev_parse(wt)

        res = await git_ops.merge_branch_into_worktree(wt, '101')

        assert await _rev_parse(wt) == res.merge_commit
        assert await _rev_parse(wt, 'HEAD^1') == pre_head
        assert await _rev_parse(wt, 'HEAD^2') == branch_tip
        assert (wt / 'disjoint.txt').read_text() == 'edit-101\n'

    async def test_merge_subject_is_canonical(self, git_repo: Path):
        """Subject comes from _merge_subject — find_merge_marker depends on it."""
        git_ops = _make_git_ops(git_repo)
        await _create_branch_editing(git_repo, 'task/101', 'disjoint.txt', 'edit-101\n')
        main_sha = await _rev_parse(git_repo)
        wt = await git_ops.create_throwaway_verify_worktree(main_sha)

        await git_ops.merge_branch_into_worktree(wt, '101')

        _, subject, _ = await _run(['git', 'log', '-1', '--format=%s'], cwd=wt)
        assert subject.strip() == 'Merge task/101 into main'

    async def test_sequential_merges_reuse_the_same_worktree(self, git_repo: Path):
        """Two merges into one worktree give a linear first-parent chain."""
        git_ops = _make_git_ops(git_repo)
        await _create_branch_editing(git_repo, 'task/101', 'a.txt', 'edit-101\n')
        await _create_branch_editing(git_repo, 'task/102', 'b.txt', 'edit-102\n')
        main_sha = await _rev_parse(git_repo)

        before = await _worktree_names(git_ops)
        wt = await git_ops.create_throwaway_verify_worktree(main_sha)

        res1 = await git_ops.merge_branch_into_worktree(wt, '101')
        assert res1.success is True
        res2 = await git_ops.merge_branch_into_worktree(wt, '102')
        assert res2.success is True
        assert res2.merge_worktree == wt

        # Linear first-parent chain in land order.
        assert await _rev_parse(wt, 'HEAD^1') == res1.merge_commit
        assert (wt / 'a.txt').read_text() == 'edit-101\n'
        assert (wt / 'b.txt').read_text() == 'edit-102\n'

        # Still exactly the one worktree the test created.
        assert await _worktree_names(git_ops) == before | {wt.name}
    # ── step-05: conflict arm ────────────────────────────────────────────────

    async def test_conflict_reports_and_leaves_worktree_clean_at_tip(
        self, git_repo: Path,
    ):
        """A textual conflict is reported AND the merge is aborted.

        Unlike ``merge_to_main`` (which RETAINS a conflicted worktree), the
        chain builder must leave the lane at the last clean tip so that tip is
        still verifiable — a residual conflicted index would poison it.
        """
        git_ops = _make_git_ops(git_repo)
        await _create_branch_editing(
            git_repo, 'task/101', 'shared.txt', _shared_txt_with(1, 'from-101'),
        )
        await _create_branch_editing(
            git_repo, 'task/102', 'shared.txt', _shared_txt_with(1, 'from-102'),
        )
        main_sha = await _rev_parse(git_repo)
        wt = await git_ops.create_throwaway_verify_worktree(main_sha)

        ok = await git_ops.merge_branch_into_worktree(wt, '101')
        assert ok.success is True
        tip = await _rev_parse(wt)
        assert tip == ok.merge_commit

        res = await git_ops.merge_branch_into_worktree(wt, '102')

        assert res.success is False
        assert res.conflicts is True
        assert res.merge_commit is None
        assert res.details
        assert 'shared.txt' in res.details
        assert res.merge_worktree == wt
        assert res.pre_merge_sha == tip

        # Worktree left clean AND at the tip.
        assert await _rev_parse(wt) == tip
        _, unmerged, _ = await _run(
            ['git', 'diff', '--name-only', '--diff-filter=U'], cwd=wt,
        )
        assert unmerged.strip() == ''
        rc, _, _ = await _run(['git', 'rev-parse', '--verify', '-q', 'MERGE_HEAD'], cwd=wt)
        assert rc != 0

    async def test_worktree_reusable_after_an_aborted_conflict(self, git_repo: Path):
        """A clean disjoint branch still merges after a conflict was aborted."""
        git_ops = _make_git_ops(git_repo)
        await _create_branch_editing(
            git_repo, 'task/101', 'shared.txt', _shared_txt_with(1, 'from-101'),
        )
        await _create_branch_editing(
            git_repo, 'task/102', 'shared.txt', _shared_txt_with(1, 'from-102'),
        )
        await _create_branch_editing(git_repo, 'task/103', 'c.txt', 'edit-103\n')
        main_sha = await _rev_parse(git_repo)
        wt = await git_ops.create_throwaway_verify_worktree(main_sha)

        first = await git_ops.merge_branch_into_worktree(wt, '101')
        conflicted = await git_ops.merge_branch_into_worktree(wt, '102')
        assert conflicted.success is False

        third = await git_ops.merge_branch_into_worktree(wt, '103')
        assert third.success is True
        assert third.merge_commit is not None
        assert await _rev_parse(wt, 'HEAD^1') == first.merge_commit
        assert (wt / 'c.txt').read_text() == 'edit-103\n'

    async def test_missing_ref_fails_loudly_without_conflict(self, git_repo: Path):
        """A non-conflict merge failure is loud, conflicts=False, HEAD unchanged."""
        git_ops = _make_git_ops(git_repo)
        main_sha = await _rev_parse(git_repo)
        wt = await git_ops.create_throwaway_verify_worktree(main_sha)
        tip = await _rev_parse(wt)

        res = await git_ops.merge_branch_into_worktree(wt, 'does-not-exist-999')

        assert res.success is False
        assert res.conflicts is False
        assert res.merge_commit is None
        assert res.details
        assert res.merge_worktree == wt
        assert await _rev_parse(wt) == tip


# ═══════════════════════════════════════════════════════════════════════════
# step-07: RED — acquire_chain_build_lane / release_chain_build_lane
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.timeout(180)
class TestChainBuildLane:
    """The chain-build worktree-routing seam in merge_liveness.py."""

    async def test_seam_is_reexported_through_merge_queue(self):
        """Both names are reachable via merge_queue's merge_liveness shim.

        merge_queue.py re-exports every merge_liveness name; a missing entry
        is a real regression for `from orchestrator.merge_queue import X`
        importers, which is the house convention.
        """
        from orchestrator.merge_liveness import (
            acquire_chain_build_lane,
            release_chain_build_lane,
        )
        from orchestrator.merge_queue import (
            acquire_chain_build_lane as mq_acquire,
        )
        from orchestrator.merge_queue import (
            release_chain_build_lane as mq_release,
        )

        assert mq_acquire is acquire_chain_build_lane
        assert mq_release is release_chain_build_lane

    async def test_warm_path_assigns_a_spec_lane(self, git_repo: Path):
        """Pool ON → a warm _spec-0 lane checked out at the base commit."""
        from orchestrator.merge_liveness import (
            acquire_chain_build_lane,
            release_chain_build_lane,
        )
        from orchestrator.warm_lane_pool import LaneState

        git_ops = _make_git_ops(git_repo, pool=True, size=1)
        config = _make_config(git_repo)
        head_sha = await _rev_parse(git_repo)

        lane, warm = await acquire_chain_build_lane(git_ops, config, head_sha)

        assert warm is True
        assert lane is not None
        assert lane == git_ops.worktree_base / '_spec-0'
        assert await git_ops._is_registered_worktree(lane)
        assert await _rev_parse(lane) == head_sha
        assert git_ops.spec_warm_lane_pool is not None
        assert git_ops.spec_warm_lane_pool._lanes[lane] == LaneState.ASSIGNED

        await release_chain_build_lane(git_ops, lane, warm=warm)

        # Warm release: pool FREE, worktree retained (target/ stays warm).
        assert git_ops.spec_warm_lane_pool._lanes[lane] == LaneState.FREE
        assert await git_ops._is_registered_worktree(lane)

    async def test_cold_path_when_pool_knob_off(self, git_repo: Path):
        """Pool OFF → an ephemeral _merge- worktree, warm=False, removed on release."""
        from orchestrator.merge_liveness import (
            acquire_chain_build_lane,
            release_chain_build_lane,
        )

        git_ops = _make_git_ops(git_repo, pool=False)
        config = _make_config(git_repo)
        assert git_ops.spec_warm_lane_pool is None
        head_sha = await _rev_parse(git_repo)

        lane, warm = await acquire_chain_build_lane(git_ops, config, head_sha)

        assert warm is False
        assert lane is not None
        assert lane.name.startswith('_merge-')
        assert await _rev_parse(lane) == head_sha

        await release_chain_build_lane(git_ops, lane, warm=False)
        assert lane.name not in await _worktree_names(git_ops)

    async def test_refuses_the_serial_merge_verify_lane(
        self, git_repo: Path, monkeypatch, caplog,
    ):
        """Fail-closed: never borrow the serial _merge-verify lane.

        Chain dispatch is spec-lane-side and structurally exempt from DF-3071's
        serial-head admission gate.  Silently borrowing _merge-verify would make
        the chain build contend with the serial head verify, and 3071's guard
        would then read the lane BUSY and defer the fleet.
        """
        import logging

        from orchestrator.git_ops import PERSISTENT_MERGE_WORKTREE_NAME
        from orchestrator.merge_liveness import acquire_chain_build_lane

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        head_sha = await _rev_parse(git_repo)
        serial = git_ops.worktree_base / PERSISTENT_MERGE_WORKTREE_NAME

        async def _fake_acquire(_merge_commit: str) -> tuple[Path, bool]:
            return serial, True

        monkeypatch.setattr(git_ops, 'acquire_spec_lane', _fake_acquire)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            lane, warm = await acquire_chain_build_lane(git_ops, config, head_sha)

        assert lane is None
        assert warm is False
        # Loud refusal, not a silent None.
        assert any(
            PERSISTENT_MERGE_WORKTREE_NAME in r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        )

    async def test_normal_spec_lane_does_not_trip_the_guard(self, git_repo: Path):
        """Negative control: a normal _spec-0 resolution is accepted."""
        from orchestrator.merge_liveness import (
            acquire_chain_build_lane,
            release_chain_build_lane,
        )

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        head_sha = await _rev_parse(git_repo)

        lane, warm = await acquire_chain_build_lane(git_ops, config, head_sha)

        assert lane is not None
        assert lane.name == '_spec-0'
        assert warm is True
        await release_chain_build_lane(git_ops, lane, warm=warm)

    async def test_acquire_never_raises(self, git_repo: Path, monkeypatch):
        """A lane hiccup degrades to (None, False), never into the dispatch loop."""
        from orchestrator.merge_liveness import acquire_chain_build_lane

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        head_sha = await _rev_parse(git_repo)

        async def _boom(_merge_commit: str) -> tuple[Path, bool]:
            raise RuntimeError('lane provisioning exploded')

        monkeypatch.setattr(git_ops, 'acquire_spec_lane', _boom)

        lane, warm = await acquire_chain_build_lane(git_ops, config, head_sha)

        assert lane is None
        assert warm is False

    async def test_release_of_none_is_a_noop(self, git_repo: Path):
        """release_chain_build_lane(None) lets callers release unconditionally."""
        from orchestrator.merge_liveness import release_chain_build_lane

        git_ops = _make_git_ops(git_repo)
        await release_chain_build_lane(git_ops, None, warm=False)
        await release_chain_build_lane(git_ops, None, warm=True)


# ═══════════════════════════════════════════════════════════════════════════
# step-09: RED — SpeculativeMergeWorker.chain_snapshot()
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.timeout(180)
class TestChainSnapshot:
    """chain_snapshot() returns queued items in pick order without mutating."""

    async def test_returns_merge_request_objects_not_ids(self, git_repo: Path):
        """Contrast with unfrozen_suffix(), which returns request_id strings."""
        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        worker = _make_worker(git_ops)
        reqs = [_make_req(str(i), str(i), config, git_repo) for i in (101, 102, 103)]
        worker._lane_buffers['normal'].extend(reqs)

        snap = worker.chain_snapshot()

        assert isinstance(snap, tuple)
        assert len(snap) == 3
        for got, want in zip(snap, reqs, strict=True):
            assert got is want

    async def test_is_the_merge_request_typed_twin_of_unfrozen_suffix(
        self, git_repo: Path,
    ):
        """The headline ordering invariant: same order as unfrozen_suffix()."""
        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        worker = _make_worker(git_ops)
        worker._lane_buffers['high'].extend([
            _make_req('201', '201', config, git_repo, lane='high'),
            _make_req('202', '202', config, git_repo, lane='high'),
        ])
        worker._lane_buffers['normal'].extend([
            _make_req('101', '101', config, git_repo),
            _make_req('102', '102', config, git_repo),
        ])

        assert tuple(
            r.request_id for r in worker.chain_snapshot()
        ) == worker.unfrozen_suffix()

    async def test_high_lane_precedes_normal_lane(self, git_repo: Path):
        """Lane priority matches MERGE_LANES = ('high', 'normal')."""
        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        worker = _make_worker(git_ops)
        hi = _make_req('201', '201', config, git_repo, lane='high')
        lo = _make_req('101', '101', config, git_repo)
        # Append the normal-lane item FIRST to prove lane order, not insert order.
        worker._lane_buffers['normal'].append(lo)
        worker._lane_buffers['high'].append(hi)

        assert [r.task_id for r in worker.chain_snapshot()] == ['201', '101']

    async def test_fifo_within_lane_not_aging_key_order(self, git_repo: Path):
        """Append order wins over merge_first_enqueued_at (the requeue case).

        `_note_requeue` re-appends at the deque TAIL while retaining an old
        `merge_first_enqueued_at`, so aging order genuinely diverges from FIFO.
        Sorting by `_aging_key` here would disagree with BOTH order authorities
        (unfrozen_suffix iterates the deque directly; _pop_next_pickable scans
        FIFO and degrades to buf[0] on an empty conflict graph) and would break
        the contiguous-prefix property δ's in-order CAS walk depends on.
        """
        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        worker = _make_worker(git_ops)
        first = _make_req('101', '101', config, git_repo)
        second = _make_req('102', '102', config, git_repo)
        # Opposite order to the append order: `second` looks OLDER by aging key.
        first.merge_first_enqueued_at = 2000.0
        second.merge_first_enqueued_at = 1000.0
        worker._lane_buffers['normal'].extend([first, second])

        assert [r.task_id for r in worker.chain_snapshot()] == ['101', '102']

    async def test_does_not_mutate_the_lane_buffers(self, git_repo: Path):
        """Nothing is popped — build_chain must be pure w.r.t. queue state."""
        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        worker = _make_worker(git_ops)
        worker._lane_buffers['normal'].extend(
            _make_req(str(i), str(i), config, git_repo) for i in (101, 102, 103)
        )
        before = [list(b) for b in worker._lane_buffers.values()]

        worker.chain_snapshot()

        assert [list(b) for b in worker._lane_buffers.values()] == before

    async def test_fires_no_lifecycle_transition(self, git_repo: Path):
        """Contrast with _pop_next_pickable, which fires LANE_BUFFERED→MERGING."""
        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        worker = _make_worker(git_ops)
        worker._lane_buffers['normal'].extend(
            _make_req(str(i), str(i), config, git_repo) for i in (101, 102)
        )
        before = dict(worker._lifecycle._states)

        worker.chain_snapshot()

        assert dict(worker._lifecycle._states) == before

    async def test_is_idempotent(self, git_repo: Path):
        """Two consecutive calls return equal tuples."""
        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        worker = _make_worker(git_ops)
        worker._lane_buffers['normal'].extend(
            _make_req(str(i), str(i), config, git_repo) for i in (101, 102)
        )

        assert worker.chain_snapshot() == worker.chain_snapshot()

    async def test_empty_queue_returns_empty_tuple(self, git_repo: Path):
        git_ops = _make_git_ops(git_repo)
        worker = _make_worker(git_ops)

        assert worker.chain_snapshot() == ()

    async def test_halted_lanes_are_excluded(self, git_repo: Path):
        """A halted lane's items are not dispatchable, so they cannot chain.

        Matches _pop_next_pickable's is_lane_halted skip.  This is the one
        deliberate divergence from unfrozen_suffix(), which reports the whole
        buffer region regardless of halt state — chaining an item the pipeline
        cannot land would build an unlandable tip.
        """
        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        worker = _make_worker(git_ops)
        worker._lane_buffers['high'].append(
            _make_req('201', '201', config, git_repo, lane='high'),
        )
        worker._lane_buffers['normal'].append(_make_req('101', '101', config, git_repo))
        worker.halt_lane('high', 'test halt')
        assert worker.is_lane_halted('high')

        assert [r.task_id for r in worker.chain_snapshot()] == ['101']


class TestChainSnapshotIsSync:
    """chain_snapshot is pure/synchronous — callable with no running loop.

    Unmarked (sync) class: a sync test_* inside an @pytest.mark.asyncio class
    is an ERROR under this file's filterwarnings config.  Uses function
    introspection plus an empty-buffer call rather than staging requests,
    because _make_req needs a running loop — which would defeat the point.
    """

    def test_chain_snapshot_is_not_a_coroutine_function(self):
        import asyncio as _asyncio

        from orchestrator.merge_queue import SpeculativeMergeWorker

        assert not _asyncio.iscoroutinefunction(
            SpeculativeMergeWorker.chain_snapshot,
        )

    def test_callable_with_no_running_event_loop(self, git_repo: Path):
        """An empty-buffer call returns () outside any event loop.

        Same contract as frozen_prefix / unfrozen_suffix / frozen_prefix_tip /
        _pop_next_pickable, so chain_snapshot stays callable from snapshot().
        """
        import asyncio as _asyncio

        with pytest.raises(RuntimeError):
            _asyncio.get_running_loop()

        worker = _make_worker(_make_git_ops(git_repo))

        assert worker.chain_snapshot() == ()


# ═══════════════════════════════════════════════════════════════════════════
# step-11: RED — build_chain degenerate inputs
# ═══════════════════════════════════════════════════════════════════════════


def _count_spec_lane_acquires(git_ops: GitOps, monkeypatch) -> list[str]:
    """Wrap acquire_spec_lane with a call recorder; return the record list."""
    calls: list[str] = []
    original = git_ops.acquire_spec_lane

    async def _recording(merge_commit: str):
        calls.append(merge_commit)
        return await original(merge_commit)

    monkeypatch.setattr(git_ops, 'acquire_spec_lane', _recording)
    return calls


@pytest.mark.asyncio
@pytest.mark.timeout(180)
class TestBuildChainDegenerate:
    """Degenerate inputs return an empty ChainResult and touch ZERO worktrees.

    The cap=0 case is the structural expression of α's kill switch at the
    builder level: no worktree is provisioned and no git command runs.
    """

    async def _assert_empty_and_untouched(
        self, git_ops: GitOps, res, head: str, before: set[str], calls: list[str],
    ) -> None:
        from orchestrator.warm_lane_pool import LaneState

        assert res.links == []
        assert res.tip == head
        assert res.truncated_at is None
        assert res.truncated_reason is None
        assert res.depth == 0
        # An empty result NEVER holds a lane, so a caller that skips the
        # release on the empty path cannot leak one.
        assert res.lane is None
        assert res.lane_warm is False
        assert await _worktree_names(git_ops) == before
        assert calls == []
        assert git_ops.spec_warm_lane_pool is not None
        assert all(
            s == LaneState.FREE
            for s in git_ops.spec_warm_lane_pool._lanes.values()
        )

    async def test_cap_zero_is_the_kill_switch(self, git_repo: Path, monkeypatch):
        """cap=0 (α's shipped default) provisions nothing and runs no git."""
        from orchestrator.merge_queue import build_chain

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        head = await _rev_parse(git_repo)
        before = await _worktree_names(git_ops)
        calls = _count_spec_lane_acquires(git_ops, monkeypatch)
        snap = (_make_req('101', '101', config, git_repo),)

        res = await build_chain(git_ops, snap, head, cap=0, target_depth=3)

        await self._assert_empty_and_untouched(git_ops, res, head, before, calls)

    async def test_target_depth_zero(self, git_repo: Path, monkeypatch):
        from orchestrator.merge_queue import build_chain

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        head = await _rev_parse(git_repo)
        before = await _worktree_names(git_ops)
        calls = _count_spec_lane_acquires(git_ops, monkeypatch)
        snap = (_make_req('101', '101', config, git_repo),)

        res = await build_chain(git_ops, snap, head, cap=6, target_depth=0)

        await self._assert_empty_and_untouched(git_ops, res, head, before, calls)

    async def test_empty_queue_snapshot(self, git_repo: Path, monkeypatch):
        from orchestrator.merge_queue import build_chain

        git_ops = _make_git_ops(git_repo)
        head = await _rev_parse(git_repo)
        before = await _worktree_names(git_ops)
        calls = _count_spec_lane_acquires(git_ops, monkeypatch)

        res = await build_chain(git_ops, (), head, cap=6, target_depth=3)

        await self._assert_empty_and_untouched(git_ops, res, head, before, calls)

    async def test_negative_cap_is_treated_as_zero(self, git_repo: Path, monkeypatch):
        """Defensive: α's validator is ge=0, but a direct call must not misbehave."""
        from orchestrator.merge_queue import build_chain

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        head = await _rev_parse(git_repo)
        before = await _worktree_names(git_ops)
        calls = _count_spec_lane_acquires(git_ops, monkeypatch)
        snap = (_make_req('101', '101', config, git_repo),)

        res = await build_chain(git_ops, snap, head, cap=-1, target_depth=3)

        await self._assert_empty_and_untouched(git_ops, res, head, before, calls)

    async def test_lane_acquisition_failure_returns_empty(
        self, git_repo: Path, monkeypatch,
    ):
        """acquire_chain_build_lane -> (None, False) yields the empty result."""
        from orchestrator.merge_queue import build_chain

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        head = await _rev_parse(git_repo)
        await _create_branch_editing(git_repo, 'task/101', 'a.txt', 'edit-101\n')
        before = await _worktree_names(git_ops)

        async def _boom(_merge_commit: str):
            raise RuntimeError('lane provisioning exploded')

        monkeypatch.setattr(git_ops, 'acquire_spec_lane', _boom)
        snap = (_make_req('101', '101', config, git_repo),)

        res = await build_chain(git_ops, snap, head, cap=6, target_depth=1)

        assert res.links == []
        assert res.tip == head
        assert res.lane is None
        assert res.lane_warm is False
        assert await _worktree_names(git_ops) == before

    async def test_depth_is_clamped_to_snapshot_length(self, git_repo: Path):
        """depth == min(len(snapshot), cap, target_depth) — a 1-item chain builds."""
        from orchestrator.merge_liveness import release_chain_build_lane
        from orchestrator.merge_queue import build_chain

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        await _create_branch_editing(git_repo, 'task/101', 'a.txt', 'edit-101\n')
        head = await _rev_parse(git_repo)
        snap = (_make_req('101', '101', config, git_repo),)

        res = await build_chain(git_ops, snap, head, cap=6, target_depth=99)

        assert [tid for tid, _ in res.links] == ['101']
        assert res.depth == 1
        assert res.tip == res.links[0][1]
        # A depth stop is not a truncation.
        assert res.truncated_at is None
        assert res.truncated_reason is None
        await release_chain_build_lane(git_ops, res.lane, warm=res.lane_warm)


# ═══════════════════════════════════════════════════════════════════════════
# step-13: RED — build_chain clean chain in ONE worktree
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
@pytest.mark.timeout(180)
class TestBuildChainClean:
    """A clean 3-item chain builds in exactly one worktree."""

    async def _three_disjoint_branches(self, git_repo: Path) -> list[str]:
        return [
            await _create_branch_editing(git_repo, 'task/101', 'a.txt', 'edit-101\n'),
            await _create_branch_editing(git_repo, 'task/102', 'b.txt', 'edit-102\n'),
            await _create_branch_editing(git_repo, 'task/103', 'c.txt', 'edit-103\n'),
        ]

    async def test_clean_chain_from_main_sha_head(self, git_repo: Path, monkeypatch):
        """head == main_sha (the frozen-prefix-empty variant of frozen_prefix_tip)."""
        from orchestrator.merge_liveness import release_chain_build_lane
        from orchestrator.merge_queue import build_chain
        from orchestrator.warm_lane_pool import LaneState

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        tips = await self._three_disjoint_branches(git_repo)
        head = await _rev_parse(git_repo)
        before = await _worktree_names(git_ops)
        calls = _count_spec_lane_acquires(git_ops, monkeypatch)
        snap = tuple(
            _make_req(str(i), str(i), config, git_repo) for i in (101, 102, 103)
        )

        res = await build_chain(git_ops, snap, head, cap=6, target_depth=3)

        # Order and identity.
        assert [tid for tid, _ in res.links] == ['101', '102', '103']
        shas = [sha for _, sha in res.links]
        assert len(set(shas)) == 3
        for sha in shas:
            assert len(sha) == 40
            assert sha == sha.strip()

        assert res.tip == shas[-1]
        assert res.depth == 3
        assert res.truncated_at is None
        assert res.truncated_reason is None

        # Exactly ONE worktree used — this task's headline signal.
        added = await _worktree_names(git_ops) - before
        assert added == {'_spec-0'}
        assert len(calls) == 1

        # Lane ownership on the success path: γ verifies the tip in place.
        assert res.lane is not None
        assert res.lane == git_ops.worktree_base / '_spec-0'
        assert res.lane_warm is True
        assert git_ops.spec_warm_lane_pool is not None
        assert git_ops.spec_warm_lane_pool._lanes[res.lane] == LaneState.ASSIGNED

        # Cumulative tree correctness — the chain is a real superset, which is
        # why ONE verify can authorise the whole prefix.
        assert await _rev_parse(res.lane) == res.tip
        assert (res.lane / 'a.txt').read_text() == 'edit-101\n'
        assert (res.lane / 'b.txt').read_text() == 'edit-102\n'
        assert (res.lane / 'c.txt').read_text() == 'edit-103\n'

        # Linear first-parent history in land order; second parent = branch tip.
        assert await _rev_parse(res.lane, f'{shas[2]}^1') == shas[1]
        assert await _rev_parse(res.lane, f'{shas[1]}^1') == shas[0]
        assert await _rev_parse(res.lane, f'{shas[0]}^1') == head
        for sha, tip in zip(shas, tips, strict=True):
            assert await _rev_parse(res.lane, f'{sha}^2') == tip

        # The documented release path works end to end.
        await release_chain_build_lane(git_ops, res.lane, warm=res.lane_warm)
        assert git_ops.spec_warm_lane_pool._lanes[res.lane] == LaneState.FREE

    async def test_clean_chain_from_a_frozen_merge_commit_head(self, git_repo: Path):
        """head == a prior merge commit (the frozen-prefix-non-empty variant)."""
        from orchestrator.merge_liveness import release_chain_build_lane
        from orchestrator.merge_queue import build_chain

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        await _create_branch_editing(git_repo, 'task/100', 'z.txt', 'edit-100\n')
        await self._three_disjoint_branches(git_repo)
        main_sha = await _rev_parse(git_repo)

        # Mimic frozen_prefix_tip: the newest frozen item's merge commit.
        seed_wt = await git_ops.create_throwaway_verify_worktree(main_sha)
        seeded = await git_ops.merge_branch_into_worktree(seed_wt, '100')
        assert seeded.success is True
        head = seeded.merge_commit
        assert head is not None
        await git_ops.cleanup_merge_worktree(seed_wt)

        snap = tuple(
            _make_req(str(i), str(i), config, git_repo) for i in (101, 102, 103)
        )

        res = await build_chain(git_ops, snap, head, cap=6, target_depth=3)

        assert [tid for tid, _ in res.links] == ['101', '102', '103']
        assert res.truncated_at is None
        assert res.lane is not None
        assert await _rev_parse(res.lane, f'{res.links[0][1]}^1') == head
        # The frozen head's own content is still present under the chain tip.
        assert (res.lane / 'z.txt').read_text() == 'edit-100\n'
        await release_chain_build_lane(git_ops, res.lane, warm=res.lane_warm)

    async def test_target_depth_stop_is_not_a_truncation(self, git_repo: Path):
        """cap=6, target_depth=2 over a 3-item snapshot → first two, no truncation."""
        from orchestrator.merge_liveness import release_chain_build_lane
        from orchestrator.merge_queue import build_chain

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        await self._three_disjoint_branches(git_repo)
        head = await _rev_parse(git_repo)
        snap = tuple(
            _make_req(str(i), str(i), config, git_repo) for i in (101, 102, 103)
        )

        res = await build_chain(git_ops, snap, head, cap=6, target_depth=2)

        assert [tid for tid, _ in res.links] == ['101', '102']
        assert res.tip == res.links[1][1]
        assert res.truncated_at is None
        assert res.truncated_reason is None
        await release_chain_build_lane(git_ops, res.lane, warm=res.lane_warm)

    async def test_cap_stop_is_not_a_truncation(self, git_repo: Path):
        """cap=2, target_depth=6 over a 3-item snapshot → first two, no truncation."""
        from orchestrator.merge_liveness import release_chain_build_lane
        from orchestrator.merge_queue import build_chain

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        await self._three_disjoint_branches(git_repo)
        head = await _rev_parse(git_repo)
        snap = tuple(
            _make_req(str(i), str(i), config, git_repo) for i in (101, 102, 103)
        )

        res = await build_chain(git_ops, snap, head, cap=2, target_depth=6)

        assert [tid for tid, _ in res.links] == ['101', '102']
        assert res.truncated_at is None
        assert res.truncated_reason is None
        await release_chain_build_lane(git_ops, res.lane, warm=res.lane_warm)


# ═══════════════════════════════════════════════════════════════════════════
# step-15: RED — build_chain truncation semantics + purity (decision #4)
# ═══════════════════════════════════════════════════════════════════════════


def _spy_merges(git_ops: GitOps, monkeypatch) -> list[str]:
    """Record the bare branch id of every merge ATTEMPT, in order.

    Proves the chain is a contiguous PREFIX: an item past the truncation
    point must never even be attempted, since a hole in ``links`` would
    break δ's in-order CAS walk.
    """
    calls: list[str] = []
    original = git_ops.merge_branch_into_worktree

    def _recording(worktree: Path, branch: str):
        calls.append(branch)
        return original(worktree, branch)

    monkeypatch.setattr(git_ops, 'merge_branch_into_worktree', _recording)
    return calls


def _spy_merge_attempt_events(monkeypatch) -> list[tuple[str, object]]:
    """Replace ``merge_queue._emit_merge_attempt`` with a recorder."""
    from orchestrator import merge_queue as _mq

    calls: list[tuple[str, object]] = []

    def _recording(_event_store, task_id, outcome, **_kw) -> None:
        calls.append((task_id, outcome))

    monkeypatch.setattr(_mq, '_emit_merge_attempt', _recording)
    return calls


def _spy_note_conflict_detected(monkeypatch) -> list[str]:
    """Replace ``SpeculativeMergeWorker._note_conflict_detected`` with a recorder.

    ``classify_and_merge`` calls it on its conflict arm; calling it from the
    chain builder would feed the conflict-graph / thrash machinery from a
    non-genuine conflict (one against an UNLANDED predecessor).
    """
    calls: list[str] = []

    def _recording(_self, request_id: str) -> None:
        calls.append(request_id)

    monkeypatch.setattr(
        SpeculativeMergeWorker, '_note_conflict_detected', _recording,
    )
    return calls


async def _is_ancestor(cwd: Path, maybe_ancestor: str, descendant: str) -> bool:
    rc, _, _ = await _run(
        ['git', 'merge-base', '--is-ancestor', maybe_ancestor, descendant],
        cwd=cwd,
    )
    return rc == 0


async def _assert_lane_clean_at(lane: Path, tip: str) -> None:
    """The lane is at *tip* with no unmerged paths and no in-progress merge."""
    assert await _rev_parse(lane) == tip
    _, unmerged, _ = await _run(
        ['git', 'diff', '--name-only', '--diff-filter=U'], cwd=lane,
    )
    assert unmerged.strip() == ''
    rc, _, _ = await _run(['git', 'rev-parse', '--verify', '-q', 'MERGE_HEAD'], cwd=lane)
    assert rc != 0


@pytest.mark.asyncio
@pytest.mark.timeout(180)
class TestBuildChainTruncation:
    """The task's headline signal: clean prefix, conflicted item UNTOUCHED, ONE worktree.

    "Untouched" is the load-bearing half (PRD decision #4).  A chain conflict
    at position j may be a conflict with an *unlanded* predecessor, so item j
    must take its normal sequential path later — which it cannot if the
    builder has already resolved its future, emitted a ``merge_attempt``, or
    fed the conflict-graph machinery on its behalf.
    """

    async def _conflicting_trio(self, git_repo: Path) -> list[str]:
        """101 and 102 both rewrite shared.txt line 1; 103 is disjoint/clean."""
        return [
            await _create_branch_editing(
                git_repo, 'task/101', 'shared.txt', _shared_txt_with(1, 'from-101'),
            ),
            await _create_branch_editing(
                git_repo, 'task/102', 'shared.txt', _shared_txt_with(1, 'from-102'),
            ),
            await _create_branch_editing(git_repo, 'task/103', 'c.txt', 'edit-103\n'),
        ]

    # ── A. textual-conflict truncation ───────────────────────────────────

    async def test_conflict_truncates_at_the_clean_prefix(
        self, git_repo: Path, monkeypatch,
    ):
        from orchestrator.merge_liveness import release_chain_build_lane
        from orchestrator.merge_queue import build_chain

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        tips = await self._conflicting_trio(git_repo)
        head = await _rev_parse(git_repo)
        before = await _worktree_names(git_ops)
        acquires = _count_spec_lane_acquires(git_ops, monkeypatch)
        merges = _spy_merges(git_ops, monkeypatch)
        snap = tuple(
            _make_req(str(i), str(i), config, git_repo) for i in (101, 102, 103)
        )

        res = await build_chain(git_ops, snap, head, cap=6, target_depth=3)

        # The clean prefix only.
        assert [tid for tid, _ in res.links] == ['101']
        assert res.tip == res.links[0][1]
        assert res.depth == 1
        assert res.truncated_at == '102'
        assert res.truncated_reason == 'conflict'

        # 103 was never ATTEMPTED — a contiguous prefix, not a gap-filling scan.
        assert merges == ['101', '102']
        assert res.lane is not None
        assert not await _is_ancestor(res.lane, tips[2], res.tip)
        assert await _is_ancestor(res.lane, tips[0], res.tip)

        # The lane is clean and verifiable AT the tip: the aborted 102 merge
        # left no unmerged index behind to poison γ's verify.
        await _assert_lane_clean_at(res.lane, res.tip)

        # Exactly ONE worktree for the whole chain.
        assert await _worktree_names(git_ops) - before == {'_spec-0'}
        assert len(acquires) == 1

        await release_chain_build_lane(git_ops, res.lane, warm=res.lane_warm)

    # ── B. purity: the conflicted item is UNTOUCHED ──────────────────────

    async def test_resolves_no_future(self, git_repo: Path, monkeypatch):
        """102's future stays pending — it may only conflict with UNLANDED 101."""
        from orchestrator.merge_liveness import release_chain_build_lane
        from orchestrator.merge_queue import build_chain

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        await self._conflicting_trio(git_repo)
        head = await _rev_parse(git_repo)
        snap = tuple(
            _make_req(str(i), str(i), config, git_repo) for i in (101, 102, 103)
        )

        res = await build_chain(git_ops, snap, head, cap=6, target_depth=3)

        assert res.truncated_at == '102'
        assert all(not r.result.done() for r in snap)
        await release_chain_build_lane(git_ops, res.lane, warm=res.lane_warm)

    async def test_emits_no_merge_attempt_event(self, git_repo: Path, monkeypatch):
        """Zero events of ANY type — an outcome event would be a false report."""
        from _recording_event_store import _RecordingEventStore

        from orchestrator.merge_liveness import release_chain_build_lane
        from orchestrator.merge_queue import build_chain

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        store = _RecordingEventStore()
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), store)  # type: ignore[arg-type]
        emitted = _spy_merge_attempt_events(monkeypatch)
        await self._conflicting_trio(git_repo)
        head = await _rev_parse(git_repo)
        worker._lane_buffers['normal'].extend(
            _make_req(str(i), str(i), config, git_repo) for i in (101, 102, 103)
        )

        res = await build_chain(
            git_ops, worker.chain_snapshot(), head, cap=6, target_depth=3,
        )

        assert res.truncated_at == '102'
        assert emitted == []
        assert store.events == []
        await release_chain_build_lane(git_ops, res.lane, warm=res.lane_warm)

    async def test_does_not_mutate_queue_state(self, git_repo: Path, monkeypatch):
        """Buffers, lifecycle states and the drift/conflict machinery untouched."""
        from orchestrator.merge_liveness import release_chain_build_lane
        from orchestrator.merge_queue import build_chain

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        worker = _make_worker(git_ops)
        noted = _spy_note_conflict_detected(monkeypatch)
        await self._conflicting_trio(git_repo)
        head = await _rev_parse(git_repo)
        worker._lane_buffers['normal'].extend(
            _make_req(str(i), str(i), config, git_repo) for i in (101, 102, 103)
        )
        buffers_before = {k: list(v) for k, v in worker._lane_buffers.items()}
        lifecycle_before = dict(worker._lifecycle._states)

        res = await build_chain(
            git_ops, worker.chain_snapshot(), head, cap=6, target_depth=3,
        )

        assert res.truncated_at == '102'
        # Object identity AND order, not just equality.
        after = {k: list(v) for k, v in worker._lane_buffers.items()}
        assert after.keys() == buffers_before.keys()
        for lane_name, items in buffers_before.items():
            assert len(after[lane_name]) == len(items)
            for got, want in zip(after[lane_name], items, strict=True):
                assert got is want
        assert dict(worker._lifecycle._states) == lifecycle_before
        assert noted == []
        await release_chain_build_lane(git_ops, res.lane, warm=res.lane_warm)

    async def test_rebuild_is_stable(self, git_repo: Path):
        """Invalidation is rebuild-per-round, so a repeat build must be stable.

        Compares task_ids and the tip TREE rather than raw shas: committer
        timestamps advance between rounds, so the merge shas legitimately
        differ while the content does not.
        """
        from orchestrator.merge_liveness import release_chain_build_lane
        from orchestrator.merge_queue import build_chain

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        await self._conflicting_trio(git_repo)
        head = await _rev_parse(git_repo)
        snap = tuple(
            _make_req(str(i), str(i), config, git_repo) for i in (101, 102, 103)
        )

        first = await build_chain(git_ops, snap, head, cap=6, target_depth=3)
        assert first.lane is not None
        tree_first = await _rev_parse(first.lane, f'{first.tip}^{{tree}}')
        await release_chain_build_lane(git_ops, first.lane, warm=first.lane_warm)

        second = await build_chain(git_ops, snap, head, cap=6, target_depth=3)
        assert second.lane is not None
        tree_second = await _rev_parse(second.lane, f'{second.tip}^{{tree}}')

        assert [tid for tid, _ in second.links] == [tid for tid, _ in first.links]
        assert second.truncated_at == first.truncated_at
        assert second.truncated_reason == first.truncated_reason
        assert tree_second == tree_first
        assert all(not r.result.done() for r in snap)
        await release_chain_build_lane(git_ops, second.lane, warm=second.lane_warm)

    # ── C. truncation at a train (GroupMergeRequest) ─────────────────────

    async def test_truncates_at_a_group_merge_request(
        self, git_repo: Path, monkeypatch,
    ):
        """A train is landed by _do_train_merge, never chained as a plain item.

        All three items here are textually CLEAN, so a missing guard shows up
        as a 3-link chain rather than as an incidental conflict.
        """
        from orchestrator.merge_liveness import release_chain_build_lane
        from orchestrator.merge_queue import build_chain

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        await _create_branch_editing(git_repo, 'task/101', 'a.txt', 'edit-101\n')
        await _create_branch_editing(git_repo, 'task/199', 'd.txt', 'edit-199\n')
        await _create_branch_editing(git_repo, 'task/103', 'c.txt', 'edit-103\n')
        head = await _rev_parse(git_repo)
        merges = _spy_merges(git_ops, monkeypatch)
        train = _make_group_req(
            '199', '199', config, git_repo, member_task_ids=['198', '199'],
        )
        snap = (
            _make_req('101', '101', config, git_repo),
            train,
            _make_req('103', '103', config, git_repo),
        )

        res = await build_chain(git_ops, snap, head, cap=6, target_depth=3)

        assert [tid for tid, _ in res.links] == ['101']
        assert res.truncated_at == '199'
        assert res.truncated_reason == 'train_request'
        # The train is not merged, and 103 behind it is not skipped past.
        assert merges == ['101']
        assert not train.result.done()
        assert all(not r.result.done() for r in snap)
        await release_chain_build_lane(git_ops, res.lane, warm=res.lane_warm)

    # ── D. non-conflict merge error is a DISTINCT reason ─────────────────

    async def test_missing_ref_truncates_as_merge_error(
        self, git_repo: Path, monkeypatch,
    ):
        """A broken ref must not be reported as a textual conflict.

        Conflating the two would hide a genuine fault behind the chain's most
        common benign outcome (loud-over-silent-degradation).
        """
        from orchestrator.merge_liveness import release_chain_build_lane
        from orchestrator.merge_queue import build_chain

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        await _create_branch_editing(git_repo, 'task/101', 'a.txt', 'edit-101\n')
        await _create_branch_editing(git_repo, 'task/103', 'c.txt', 'edit-103\n')
        head = await _rev_parse(git_repo)
        merges = _spy_merges(git_ops, monkeypatch)
        # No task/102 branch is ever created.
        snap = tuple(
            _make_req(str(i), str(i), config, git_repo) for i in (101, 102, 103)
        )

        res = await build_chain(git_ops, snap, head, cap=6, target_depth=3)

        assert [tid for tid, _ in res.links] == ['101']
        assert res.truncated_at == '102'
        assert res.truncated_reason == 'merge_error'
        assert merges == ['101', '102']
        assert res.lane is not None
        await _assert_lane_clean_at(res.lane, res.tip)
        assert all(not r.result.done() for r in snap)
        await release_chain_build_lane(git_ops, res.lane, warm=res.lane_warm)

    # ── E. the first item conflicts: empty prefix holds NO lane ──────────

    async def test_first_item_conflicts_yields_empty_and_releases_the_lane(
        self, git_repo: Path,
    ):
        """head already carries the edit 101 conflicts with → zero links."""
        from orchestrator.merge_queue import build_chain
        from orchestrator.warm_lane_pool import LaneState

        git_ops = _make_git_ops(git_repo)
        config = _make_config(git_repo)
        await _create_branch_editing(
            git_repo, 'task/100', 'shared.txt', _shared_txt_with(1, 'from-100'),
        )
        await _create_branch_editing(
            git_repo, 'task/101', 'shared.txt', _shared_txt_with(1, 'from-101'),
        )
        main_sha = await _rev_parse(git_repo)

        # Mimic frozen_prefix_tip: head is 100's merge commit, already holding
        # the line-1 edit that 101 rewrites differently.
        seed_wt = await git_ops.create_throwaway_verify_worktree(main_sha)
        seeded = await git_ops.merge_branch_into_worktree(seed_wt, '100')
        assert seeded.success is True
        head = seeded.merge_commit
        assert head is not None
        await git_ops.cleanup_merge_worktree(seed_wt)

        snap = (_make_req('101', '101', config, git_repo),)

        res = await build_chain(git_ops, snap, head, cap=6, target_depth=1)

        assert res.links == []
        assert res.depth == 0
        assert res.tip == head
        assert res.truncated_at == '101'
        assert res.truncated_reason == 'conflict'
        # An empty result never HOLDS a lane, so a caller that skips the
        # release on the empty path cannot leak one.
        assert res.lane is None
        assert res.lane_warm is False
        assert git_ops.spec_warm_lane_pool is not None
        assert all(
            s == LaneState.FREE
            for s in git_ops.spec_warm_lane_pool._lanes.values()
        )
        assert not snap[0].result.done()
