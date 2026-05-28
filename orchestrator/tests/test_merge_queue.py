"""Tests for merge queue: MergeWorker, CAS update-ref, ghost-loop detection."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.merge_queue import (
    TRAIN_INCOMPLETE_REASON_PREFIX,
    TRAIN_PARTIAL_FLIP_REASON_PREFIX,
    TRAIN_REBASE_CONFLICT_REASON_PREFIX,
    TRANSIENT_INFRA_REASON_PREFIX,
    WORKTREE_MISSING_REASON_PREFIX,
    DropGuardResult,
    GroupMergeRequest,
    InFlightMergeRegistry,
    MergeOutcome,
    MergeRequest,
    MergeWorker,
    SpeculativeItem,
    SpeculativeMergeWorker,
    _check_plan_files_touched_in_branch,
    _check_plan_targets_in_tree,
    _check_post_merge_equivalence,
    _ensure_verify_disk_space,
    _verify_hit_enospc,
    coalesce_or_enqueue_merge_request,
)
from orchestrator.verify import VerifyResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


async def _setup_repo(repo: Path):
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        # Tests use a tmp repo with no real remote; disabling the push avoids
        # per-test subprocess noise. Push behavior is exercised explicitly in
        # test_git_ops.TestPushMain and TestPushHook below.
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
    pre_rebased: bool = False,
) -> MergeRequest:
    future: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=pre_rebased,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
    )


def _mock_verify_pass():
    """Return a mock that makes run_scoped_verification always pass."""
    return AsyncMock(return_value=MagicMock(passed=True, summary=''))


# ---------------------------------------------------------------------------
# TestCasUpdateRef — Phase A
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCasUpdateRef:
    async def test_advance_main_with_expected(self, git_ops: GitOps):
        """CAS succeeds when expected_main matches actual main."""
        worktree = (await git_ops.create_worktree('cas-ok')).path
        (worktree / 'file.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add file')

        result = await git_ops.merge_to_main(worktree, 'cas-ok')
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None

        main_sha = await git_ops.get_main_sha()
        advanced = await git_ops.advance_main(
            result.merge_commit,
            expected_main=main_sha,
        )
        assert advanced == 'advanced'

        await git_ops.cleanup_merge_worktree(result.merge_worktree)

    async def test_advance_main_cas_mismatch(self, git_ops: GitOps):
        """Main moved past merge commit with no worktree for retry → not_descendant.

        Note: the CAS check (update-ref expected_main) only runs AFTER the
        descendant check passes.  Without a merge_worktree to rebase onto
        the new main, advance_main cannot make the commit a descendant, so
        it returns 'not_descendant' before reaching the CAS step.
        """
        worktree = (await git_ops.create_worktree('cas-fail')).path
        (worktree / 'file.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add file')

        result = await git_ops.merge_to_main(worktree, 'cas-fail')
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None

        # Simulate external actor advancing main
        stale_sha = await git_ops.get_main_sha()
        (git_ops.project_root / 'external.py').write_text('ext = True\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'External commit'], cwd=git_ops.project_root)

        # Merge commit is no longer a descendant of (new) main and no
        # merge_worktree was passed for retry → not_descendant
        advanced = await git_ops.advance_main(
            result.merge_commit,
            expected_main=stale_sha,
        )
        assert advanced == 'not_descendant'

        await git_ops.cleanup_merge_worktree(result.merge_worktree)

    async def test_advance_main_none_expected(self, git_ops: GitOps):
        """Backward compat: no expected_main → unconditional update-ref."""
        worktree = (await git_ops.create_worktree('cas-none')).path
        (worktree / 'file.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add file')

        result = await git_ops.merge_to_main(worktree, 'cas-none')
        assert result.success
        assert result.merge_commit is not None

        # No expected_main — should work as before
        advanced = await git_ops.advance_main(result.merge_commit)
        assert advanced == 'advanced'

        if result.merge_worktree:
            await git_ops.cleanup_merge_worktree(result.merge_worktree)


# ---------------------------------------------------------------------------
# TestMergeWorker — Phase B
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCheckPlanTargetsInTree:
    """Unit tests for the plan-target drop-guard helper."""

    async def test_all_plan_targets_present(
        self, git_ops: GitOps,
    ):
        """Plan targets that exist in the merge commit → empty missing list."""
        worktree = (await git_ops.create_worktree('plan-all-present')).path
        (worktree / 'alpha.py').write_text('alpha = 1\n')
        (worktree / 'beta.py').write_text('beta = 2\n')
        await git_ops.commit(worktree, 'Add files')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t1', 'T1', 'desc')
        artifacts.write_plan({
            'files': ['alpha.py', 'beta.py'],
            'modules': [],
            'steps': [],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'plan-all-present')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
                await git_ops.get_main_sha(),
            )
            missing = result.dropped
            assert missing == []
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_plan_target_never_created_not_flagged(
        self, git_ops: GitOps,
    ):
        """Plan lists a file the task never created — not a merge drop.

        The file isn't on task HEAD either, so its absence from the merge
        commit reflects the task branch's own state, not conflict loss.
        Plan-delivery gaps (listed in plan.files, never produced by the
        task) are a different class of problem and are out of scope for
        the merge-time drop-guard; catching them belongs to review/verify.
        """
        worktree = (await git_ops.create_worktree('plan-one-missing')).path
        (worktree / 'present.py').write_text('present = 1\n')
        await git_ops.commit(worktree, 'Add present only')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t2', 'T2', 'desc')
        artifacts.write_plan({
            'files': ['present.py', 'absent.py'],
            'modules': [],
            'steps': [],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'plan-one-missing')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
                await git_ops.get_main_sha(),
            )
            missing = result.dropped
            assert missing == []
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_plan_target_added_then_deleted_on_branch_not_flagged(
        self, git_ops: GitOps,
    ):
        """File added then deleted on the task branch → intentional, not flagged.

        Real-world case (task/982): a plan adds a scaffold, a later reviewed
        step deletes it as an anti-pattern. The file is listed in
        plan.files, is absent from the merge commit, and is absent from
        task HEAD — that matches task intent, not conflict loss.
        """
        worktree = (
            await git_ops.create_worktree('plan-added-then-deleted')
        ).path
        (worktree / 'keep.py').write_text('keep = 1\n')
        (worktree / 'scratch.py').write_text('scratch = 1\n')
        await git_ops.commit(worktree, 'Add keep + scratch')
        (worktree / 'scratch.py').unlink()
        await git_ops.commit(worktree, 'Remove scratch per review')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t2b', 'T2b', 'desc')
        artifacts.write_plan({
            'files': ['keep.py', 'scratch.py'],
            'modules': [],
            'steps': [],
        })

        merge_result = await git_ops.merge_to_main(
            worktree, 'plan-added-then-deleted',
        )
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
                await git_ops.get_main_sha(),
            )
            missing = result.dropped
            assert missing == []
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_plan_target_on_head_dropped_by_merge_is_flagged(
        self, git_ops: GitOps,
    ):
        """File present on task HEAD but absent from merge commit → flagged.

        Simulates the real failure mode the guard was built for: conflict
        resolution accepts origin and drops a file the task branch
        produced. We synthesise the detector input by pointing the
        `merge_commit_sha` at an earlier task-branch commit that predates
        the addition of the dropped file — it has the retained file but
        not the dropped one, matching what a bad conflict resolution would
        have produced.
        """
        worktree = (await git_ops.create_worktree('plan-dropped')).path
        (worktree / 'retained.py').write_text('retained = 1\n')
        await git_ops.commit(worktree, 'Add retained')
        rc, pre_drop_sha, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=worktree,
        )
        assert rc == 0
        pre_drop_sha = pre_drop_sha.strip()

        (worktree / 'dropped.py').write_text('dropped = 1\n')
        await git_ops.commit(worktree, 'Add dropped')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t2c', 'T2c', 'desc')
        artifacts.write_plan({
            'files': ['retained.py', 'dropped.py'],
            'modules': [],
            'steps': [],
        })

        # pre_drop_sha has retained.py but not dropped.py, and task HEAD
        # has both — so only dropped.py should be flagged as a merge drop.
        # Pass the worktree's real main base: dropped.py is in branch_changed
        # (base..task_head AM), so it survives the main-side subtraction.
        result = await _check_plan_targets_in_tree(
            pre_drop_sha, worktree, git_ops, await git_ops.get_main_sha(),
        )
        missing = result.dropped
        assert missing == ['dropped.py']

    async def test_real_conflict_resolution_drop_is_flagged(
        self, git_ops: GitOps,
    ):
        """Genuine conflict-time drop (file removed during resolution) → flagged.

        Complements ``test_plan_target_on_head_dropped_by_merge_is_flagged``
        which synthesises the detector input from a pre-drop task commit.
        Here we build an actual merge commit produced by resolving a
        real conflict via `git rm`, which is the failure mode the guard
        was originally designed to catch.
        """
        # Task branch: adds contested.py + other.py
        worktree = (await git_ops.create_worktree('real-conflict-drop')).path
        full_branch = f'{git_ops.config.branch_prefix}real-conflict-drop'
        (worktree / 'contested.py').write_text('task_version = 1\n')
        (worktree / 'other.py').write_text('other = 1\n')
        await git_ops.commit(worktree, 'Task: add contested + other')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('t2d', 'T2d', 'desc')
        artifacts.write_plan({
            'files': ['contested.py', 'other.py'],
            'modules': [],
            'steps': [],
        })

        # Main: independently add contested.py with different content →
        # guaranteed conflict when task branch is merged in.
        (git_ops.project_root / 'contested.py').write_text(
            'main_version = 1\n'
        )
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Main: add contested'],
            cwd=git_ops.project_root,
        )

        # Manually build a merge commit that resolves the conflict by
        # dropping contested.py — emulating a human (or LLM) resolving
        # the merge by removing the contested file entirely.
        merge_wt = (
            git_ops.worktree_base / '_real-merge'
        )
        await _run(
            ['git', 'worktree', 'add', '--detach', str(merge_wt), 'main'],
            cwd=git_ops.project_root,
        )
        try:
            rc, out, err = await _run(
                ['git', 'merge', '--no-ff', '--no-commit', full_branch],
                cwd=merge_wt,
            )
            # Expect a real conflict
            assert 'CONFLICT' in out or 'CONFLICT' in err, (
                f'Expected conflict; got rc={rc} out={out!r} err={err!r}'
            )
            # Resolve by dropping contested.py entirely
            await _run(['git', 'rm', '-f', 'contested.py'], cwd=merge_wt)
            rc, _, err = await _run(
                ['git', 'commit', '--no-edit'], cwd=merge_wt,
            )
            assert rc == 0, f'merge commit failed: {err}'
            _, merge_sha, _ = await _run(
                ['git', 'rev-parse', 'HEAD'], cwd=merge_wt,
            )
            merge_sha = merge_sha.strip()

            # Detector must flag contested.py (on task HEAD, absent from merge)
            # but leave other.py (present on both) alone.  Pass the real main
            # tip: contested.py is in branch_changed (the branch added it), so
            # it survives the main-side subtraction and stays flagged.
            result = await _check_plan_targets_in_tree(
                merge_sha, worktree, git_ops, await git_ops.get_main_sha(),
            )
            missing = result.dropped
            assert missing == ['contested.py']
        finally:
            await _run(
                ['git', 'worktree', 'remove', str(merge_wt), '--force'],
                cwd=git_ops.project_root,
            )

    async def test_no_plan_json_returns_empty(
        self, git_ops: GitOps,
    ):
        """No plan.json (architect never ran) → empty missing list."""
        worktree = (await git_ops.create_worktree('no-plan')).path
        (worktree / 'file.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add file')

        # Deliberately NOT creating .task/plan.json
        merge_result = await git_ops.merge_to_main(worktree, 'no-plan')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
                await git_ops.get_main_sha(),
            )
            missing = result.dropped
            assert missing == []
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_check_plan_targets_emits_structured_warning_when_dropped_non_empty(
        self, git_ops: GitOps, caplog: pytest.LogCaptureFixture,
    ):
        """Structured WARNING fires when dropped is non-empty, silent otherwise.

        The WARNING text must include the task_id, the merge_commit_sha, and
        the dropped-file list — these are the fields ops grep for when
        diagnosing a fired drop-guard.  When dropped is empty the helper
        must stay silent so a normal merge produces no spurious WARNINGs.
        """
        worktree = (await git_ops.create_worktree('struct-warn-drop')).path
        (worktree / 'retained.py').write_text('retained = 1\n')
        await git_ops.commit(worktree, 'Add retained.py')
        rc, pre_drop_sha_out, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=worktree,
        )
        assert rc == 0
        pre_drop_sha = pre_drop_sha_out.strip()

        # Add the dropped file so it's on task HEAD but not on pre_drop_sha
        # — pre_drop_sha plays the role of a merge commit that lost the file.
        (worktree / 'dropped.py').write_text('dropped = 1\n')
        await git_ops.commit(worktree, 'Add dropped.py')

        # ── Sub-case 1: dropped is non-empty → structured WARNING ──────────
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = await _check_plan_targets_in_tree(
                pre_drop_sha, worktree, git_ops, await git_ops.get_main_sha(),
                task_id='warn-test',
            )
        assert result.dropped == ['dropped.py'], (
            f'Unexpected dropped: {result.dropped!r}'
        )

        warn_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warn_records, 'Expected at least one WARNING when dropped is non-empty'
        all_messages = ' '.join(r.getMessage() for r in warn_records)
        assert 'warn-test' in all_messages, (
            f'Expected task_id "warn-test" in WARNING; got: {all_messages!r}'
        )
        assert pre_drop_sha in all_messages or pre_drop_sha[:12] in all_messages, (
            f'Expected merge_commit_sha in WARNING; got: {all_messages!r}'
        )
        assert 'dropped.py' in all_messages, (
            f'Expected dropped file path in WARNING; got: {all_messages!r}'
        )

        # ── Sub-case 2: dropped is empty → no WARNING ──────────────────────
        worktree2 = (await git_ops.create_worktree('struct-warn-empty')).path
        (worktree2 / 'present.py').write_text('p = 1\n')
        await git_ops.commit(worktree2, 'Add present.py')

        merge_result2 = await git_ops.merge_to_main(worktree2, 'struct-warn-empty')
        assert merge_result2.success
        assert merge_result2.merge_commit is not None
        try:
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
                result2 = await _check_plan_targets_in_tree(
                    merge_result2.merge_commit, worktree2, git_ops,
                    await git_ops.get_main_sha(),
                    task_id='warn-test-empty',
                )
            assert result2.dropped == [], (
                f'Expected empty dropped, got {result2.dropped!r}'
            )
            new_warn = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert not new_warn, (
                f'Expected no WARNING when dropped is empty; '
                f'got: {[r.getMessage() for r in new_warn]!r}'
            )
        finally:
            if merge_result2.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result2.merge_worktree)

    async def test_drop_guard_no_op_for_gitignored_path(
        self, git_ops: GitOps,
    ):
        """Gitignored paths in plan['files'] must NEVER fire the drop-guard.

        Replicates reify task-3087: tree-sitter generated files
        (parser.c, grammar.json, node-types.json) are listed in the
        architect's plan for module-locking purposes but are explicitly
        gitignored at the repo level.  They are never on task HEAD and
        never on the merge commit, so the drop-guard must report no drops.

        Under the post-fix contract, the gate compares task HEAD to the
        merge commit directly — plan['files'] is no longer consulted —
        so gitignored entries cannot reach the dropped list.
        """
        worktree = (await git_ops.create_worktree('drop-guard-gitignore')).path
        # Add a real file plus a .gitignore that excludes the planned-but-generated path.
        (worktree / '.gitignore').write_text('generated/\n')
        (worktree / 'real.py').write_text('real = 1\n')
        await git_ops.commit(worktree, 'Add real.py + .gitignore')

        # Plan lists a gitignored path AND a structured step — under the OLD
        # heuristic this would skip the narrow-against-task-HEAD filter and
        # flag the gitignored file as dropped.
        artifacts = TaskArtifacts(worktree)
        artifacts.init('drop-gitignore', 'Drop gitignore', 'desc')
        artifacts.write_plan({
            'files': ['real.py', 'generated/parser.c'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-1',
                    'description': 'noop — anchors structured-steps branch',
                    'status': 'pending',
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'drop-guard-gitignore')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
                await git_ops.get_main_sha(),
                task_id='drop-gitignore',
            )
            assert result.dropped == [], (
                f'Gitignored path must not be flagged as dropped; '
                f'got {result.dropped!r}'
            )
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_drop_guard_no_op_for_branch_deleted_path(
        self, git_ops: GitOps,
    ):
        """File deleted by a commit not recorded in plan['steps'] is not flagged.

        Replicates reify task-3093: ``tests/common/mod.rs`` was deleted by
        a prereq commit (recorded in plan['prerequisites'], not in
        plan['steps']).  Under the OLD heuristic the structured-steps
        branch only iterates done plan-steps so the prereq deletion is
        invisible and the file is mis-flagged as a merger drop.

        Under the post-fix contract, the gate diffs task HEAD vs merge
        commit — the branch's own deletion means the file isn't on task
        HEAD, so it can't be in the diff-D output.
        """
        worktree = (await git_ops.create_worktree('drop-guard-branch-del')).path

        # Step 1: add the file (would be a "prereq" in production wiring).
        (worktree / 'tests_common.rs').write_text('// prereq\n')
        await git_ops.commit(worktree, 'Prereq: add tests_common.rs')

        # Step 2: delete the file as part of branch work.  This commit is NOT
        # recorded in plan['steps'] — it represents a prereq landing on the
        # branch outside the architect's tracked steps.
        (worktree / 'tests_common.rs').unlink()
        await _run(['git', 'add', '-A'], cwd=worktree)
        await _run(['git', 'commit', '-m', 'Prereq: delete tests_common.rs'], cwd=worktree)

        # Step 3: real work commit.
        (worktree / 'real.py').write_text('real = 1\n')
        real_sha = await git_ops.commit(worktree, 'Add real.py')
        assert real_sha

        artifacts = TaskArtifacts(worktree)
        artifacts.init('drop-branch-del', 'Drop branch del', 'desc')
        artifacts.write_plan({
            'files': ['tests_common.rs', 'real.py'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-real',
                    'description': 'add real.py',
                    'status': 'done',
                    'commit': real_sha,
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'drop-guard-branch-del')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
                await git_ops.get_main_sha(),
                task_id='drop-branch-del',
            )
            assert result.dropped == [], (
                f'Prereq-deleted file must not be flagged as dropped; '
                f'got {result.dropped!r}'
            )
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_drop_guard_no_op_for_amend_deleted_path(
        self, git_ops: GitOps,
    ):
        """File deleted by an amend commit not recorded in any plan-step is not flagged.

        Replicates reify task-3004: ``crates/reify-compiler/stdlib/fea.ri``
        was deleted by an ``amend(reviewer-suggestions)`` commit that
        isn't recorded in any plan-step's ``commit`` field.  Under the OLD
        heuristic this would flag the file because the structured-steps
        branch can't see the amend deletion.

        Under the post-fix contract, the gate diffs task HEAD vs merge
        commit — the amend rewrote HEAD without the file, so the diff-D
        output is empty.
        """
        worktree = (await git_ops.create_worktree('drop-guard-amend-del')).path

        # Step 1: add a file the architect intends to keep.
        (worktree / 'fea.ri').write_text('// stdlib\n')
        await git_ops.commit(worktree, 'Add fea.ri (planned)')

        # Step 2: a real work commit that the plan records.
        (worktree / 'real.py').write_text('real = 1\n')
        original_sha = await git_ops.commit(worktree, 'Add real.py')
        assert original_sha

        # Step 3: amend the *most recent* commit to also delete fea.ri —
        # simulating ``amend(reviewer-suggestions)``.  The amended SHA replaces
        # original_sha; the plan's recorded commit becomes orphaned/different.
        (worktree / 'fea.ri').unlink()
        await _run(['git', 'add', '-A'], cwd=worktree)
        await _run(['git', 'commit', '--amend', '--no-edit'], cwd=worktree)

        artifacts = TaskArtifacts(worktree)
        artifacts.init('drop-amend-del', 'Drop amend del', 'desc')
        artifacts.write_plan({
            'files': ['fea.ri', 'real.py'],
            'modules': [],
            'steps': [
                {
                    'id': 'step-real',
                    'description': 'add real.py (pre-amend SHA)',
                    'status': 'done',
                    'commit': original_sha,
                },
            ],
        })

        merge_result = await git_ops.merge_to_main(worktree, 'drop-guard-amend-del')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops,
                await git_ops.get_main_sha(),
                task_id='drop-amend-del',
            )
            assert result.dropped == [], (
                f'Amend-deleted file must not be flagged as dropped; '
                f'got {result.dropped!r}'
            )
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_sibling_moved_file_not_flagged(self, git_ops: GitOps):
        """A sibling git-mv on main must not flag the old path (esc-3861).

        A sibling task renamed the file on main after the victim forked.
        The victim carried the old path verbatim and never touched it, so a
        clean merge correctly drops it.  With ``--no-renames`` + the
        branch-changed intersection, the old path is absent from the branch's
        add/modify set and must NOT be flagged as a drop.
        """
        # Base state on main: the file at its old path (the fork point).
        (git_ops.project_root / 'old').mkdir()
        (git_ops.project_root / 'old' / 'ast.rs').write_text('// ast\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Main: add old/ast.rs'],
            cwd=git_ops.project_root,
        )

        # Victim forks here carrying old/ast.rs, and only adds unrelated.py.
        worktree = (await git_ops.create_worktree('sibling-move')).path
        (worktree / 'unrelated.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Victim: add unrelated.py')
        rc, task_head_out, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=worktree,
        )
        assert rc == 0
        task_head = task_head_out.strip()

        # Sibling moves the file on main AFTER the victim forked (modelled as
        # a delete + identical-content add, which git records as a rename).
        (git_ops.project_root / 'new').mkdir()
        (git_ops.project_root / 'new' / 'ast.rs').write_text('// ast\n')
        (git_ops.project_root / 'old' / 'ast.rs').unlink()
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Sibling: move old/ast.rs -> new/ast.rs'],
            cwd=git_ops.project_root,
        )
        main_sha = await git_ops.get_main_sha()

        merge_result = await git_ops.merge_to_main(worktree, 'sibling-move')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        try:
            # Sanity: the clean merge really did drop the old path — the raw
            # (unsubtracted) drop set contains it, so the intersection is what
            # prevents the false positive.
            rc, raw_out, _ = await _run(
                ['git', 'diff', '--name-only', '--no-renames',
                 '--diff-filter=D', task_head, merge_result.merge_commit],
                cwd=git_ops.project_root,
            )
            assert rc == 0
            assert 'old/ast.rs' in raw_out, (
                f'expected merge to drop old/ast.rs; raw drop set: {raw_out!r}'
            )

            result = await _check_plan_targets_in_tree(
                merge_result.merge_commit, worktree, git_ops, main_sha,
                task_id='sibling-move',
            )
            assert result.dropped == [], (
                f'sibling-moved file must not be flagged; got {result.dropped!r}'
            )
        finally:
            if merge_result.merge_worktree:
                await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_genuine_drop_of_branch_added_file_still_flagged(
        self, git_ops: GitOps,
    ):
        """A real drop of branch-only work is STILL flagged after subtraction.

        The branch adds feature.py; main never touches it; the merge drops it
        (resolution "accepted main").  Main is non-trivially ahead so the
        merge-base is a genuine ancestor — distinct from main's tip — which
        exercises the new merge-base step.  feature.py is in the branch's
        add/modify set and absent from main, so the subtraction leaves it
        flagged.
        """
        worktree = (await git_ops.create_worktree('genuine-drop')).path
        (worktree / 'feature.py').write_text('feature = 1\n')
        await git_ops.commit(worktree, 'Branch: add feature.py')

        # Main moves ahead AFTER the fork (unrelated file) so merge-base is a
        # real ancestor rather than main's tip.
        (git_ops.project_root / 'ahead.py').write_text('ahead = 1\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Main: add ahead.py'],
            cwd=git_ops.project_root,
        )
        main_sha = await git_ops.get_main_sha()

        # Synthetic merge commit = main's tip: the branch's only file never
        # landed (a resolution that dropped feature.py entirely).
        result = await _check_plan_targets_in_tree(
            main_sha, worktree, git_ops, main_sha, task_id='genuine-drop',
        )
        assert result.dropped == ['feature.py']

    async def test_drop_guard_fails_open_on_bad_main_sha(
        self, git_ops: GitOps,
    ):
        """A merge-base failure (bogus main_sha) fails open → no drops flagged."""
        worktree = (await git_ops.create_worktree('failopen-drop')).path
        (worktree / 'f.py').write_text('f = 1\n')
        await git_ops.commit(worktree, 'Add f.py')

        result = await _check_plan_targets_in_tree(
            await git_ops.get_main_sha(), worktree, git_ops,
            'definitely-not-a-ref', task_id='failopen-drop',
        )
        assert result.dropped == []


@pytest.mark.asyncio
class TestMergeWorker:
    async def test_basic_merge_through_queue(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Submit a merge request → worker merges → file appears on main."""
        worktree = (await git_ops.create_worktree('queue-basic')).path
        (worktree / 'queued.py').write_text('queued = True\n')
        await git_ops.commit(worktree, 'Add queued file')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req = _make_request('1', 'queue-basic', worktree, config)
            await queue.put(req)
            result = await asyncio.wait_for(req.result, timeout=30)

        assert result.status == 'done'

        # Verify file is on main
        _, content, _ = await _run(
            ['git', 'show', 'main:queued.py'], cwd=git_ops.project_root,
        )
        assert 'queued = True' in content

        # File should also be in the working tree (working tree synced)
        assert (git_ops.project_root / 'queued.py').exists()

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_blocks_when_merge_drops_plan_target(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """MergeWorker blocks and surfaces a clear reason on a real drop.

        Drop semantics: file was on task HEAD but is absent from the
        merge commit. Reproducing a real conflict-time drop in a unit
        test is awkward, so we mock the detector to simulate the drop
        and verify MergeWorker's handling (reason text, no advance).
        """
        worktree = (await git_ops.create_worktree('drop-guard-task')).path
        (worktree / 'kept.py').write_text('kept = True\n')
        (worktree / 'dropped.py').write_text('dropped = True\n')
        await git_ops.commit(worktree, 'Add kept + dropped')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('drop-guard', 'Drop guard', 'desc')
        artifacts.write_plan({
            'files': ['kept.py', 'dropped.py'],
            'modules': [],
            'steps': [],
        })

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        async def _fake_drop_check(*_args, **_kwargs):
            return DropGuardResult(dropped=['dropped.py'])

        with patch(
            'orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass(),
        ), patch(
            'orchestrator.merge_queue._check_plan_targets_in_tree',
            _fake_drop_check,
        ):
            req = _make_request('drop-guard', 'drop-guard-task', worktree, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked'
        assert 'dropped.py' in outcome.reason
        assert 'plan target' in outcome.reason.lower()

        # Main must NOT have advanced — drop-guard fires before advance_main
        _, main_files, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'kept.py' not in main_files

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_blocks_when_merge_drops_plan_target_real_detector(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """MergeWorker drives the real detector and blocks on a synthesised drop.

        Companion to ``test_blocks_when_merge_drops_plan_target``: that test
        mocks the detector to pin the reason-text contract; this one leaves
        the detector in place and mocks ``git_ops.merge_to_main`` so a
        future refactor of the wiring (argument order, return-value
        interpretation) cannot pass both tests while silently breaking the
        guard in production.
        """
        worktree = (await git_ops.create_worktree('drop-guard-real')).path
        (worktree / 'retained.py').write_text('retained = 1\n')
        await git_ops.commit(worktree, 'Add retained')
        rc, pre_drop_sha, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=worktree,
        )
        assert rc == 0
        pre_drop_sha = pre_drop_sha.strip()

        (worktree / 'dropped.py').write_text('dropped = 1\n')
        await git_ops.commit(worktree, 'Add dropped')

        artifacts = TaskArtifacts(worktree)
        artifacts.init('drop-guard-real', 'Drop guard real', 'desc')
        artifacts.write_plan({
            'files': ['retained.py', 'dropped.py'],
            'modules': [],
            'steps': [],
        })

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # Point the "merge commit" at a task-branch SHA that predates the
        # addition of dropped.py. The real detector sees dropped.py on task
        # HEAD but absent from that tree → flags it as a drop.
        async def _fake_merge_to_main(*_args: Any, **_kwargs: Any) -> MergeResult:
            return MergeResult(
                success=True,
                merge_commit=pre_drop_sha,
                pre_merge_sha=pre_drop_sha,
                merge_worktree=None,
            )

        with patch(
            'orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass(),
        ), patch.object(git_ops, 'merge_to_main', _fake_merge_to_main):
            req = _make_request(
                'drop-guard-real', 'drop-guard-real', worktree, config,
            )
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked'
        assert 'dropped.py' in outcome.reason
        assert 'plan target' in outcome.reason.lower()

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_already_merged_returns_done(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Branch that's already on main returns already_merged."""
        worktree = (await git_ops.create_worktree('already-merged')).path
        (worktree / 'merged.py').write_text('merged = True\n')
        await git_ops.commit(worktree, 'Add merged file')

        # Merge manually first
        result = await git_ops.merge_to_main(worktree, 'already-merged')
        assert result.success
        assert result.merge_commit is not None
        await git_ops.advance_main(result.merge_commit)
        if result.merge_worktree:
            await git_ops.cleanup_merge_worktree(result.merge_worktree)

        # Now submit to queue — should detect already merged
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        req = _make_request('2', 'already-merged', worktree, config)
        await queue.put(req)

        outcome = await asyncio.wait_for(req.result, timeout=10)
        assert outcome.status == 'already_merged'

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_conflict_returns_conflict(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Conflicting branch returns conflict status."""
        # Create worktree FIRST (from current main)
        worktree = (await git_ops.create_worktree('conflict-task')).path

        # THEN advance main with conflicting change to same file
        (git_ops.project_root / 'README.md').write_text('# Main version\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Main change'],
            cwd=git_ops.project_root,
        )

        # Now modify same file in worktree (divergent history)
        (worktree / 'README.md').write_text('# Task version\n')
        await git_ops.commit(worktree, 'Task change')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        req = _make_request('3', 'conflict-task', worktree, config)
        await queue.put(req)

        outcome = await asyncio.wait_for(req.result, timeout=10)
        assert outcome.status == 'conflict'
        assert outcome.conflict_details  # non-empty

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_cas_failure_reenqueues_at_front(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """CAS failure → re-enqueue at front → succeeds on retry."""
        worktree = (await git_ops.create_worktree('cas-retry')).path
        (worktree / 'retry.py').write_text('retry = True\n')
        await git_ops.commit(worktree, 'Add retry file')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # Monkey-patch advance_main to fail once, then succeed
        original = git_ops.advance_main
        call_count = 0

        async def _fail_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 'cas_failed'  # simulate CAS failure
            return await original(*args, **kwargs)

        with (
            patch.object(git_ops, 'advance_main', side_effect=_fail_then_succeed),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('4', 'cas-retry', worktree, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done'
        assert call_count == 2  # failed once, succeeded on retry

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_graceful_shutdown_drains(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """stop() resolves all pending futures as blocked."""
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)

        # Don't start worker yet — put items in queue, then stop
        worktree = (await git_ops.create_worktree('shutdown')).path
        req = _make_request('5', 'shutdown', worktree, config)
        await queue.put(req)

        # stop() should drain the queue and resolve the future
        await worker.stop()

        assert req.result.done()
        outcome = req.result.result()
        assert outcome.status == 'blocked'
        assert 'shutting down' in outcome.reason.lower()

    async def test_verify_failure_returns_blocked(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Post-merge verification failure → blocked."""
        worktree = (await git_ops.create_worktree('verify-fail')).path
        (worktree / 'bad.py').write_text('bad = True\n')
        await git_ops.commit(worktree, 'Add bad file')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # Mock verification to fail
        mock_verify = AsyncMock(return_value=MagicMock(passed=False, summary='tests failed'))

        with patch('orchestrator.merge_queue.run_scoped_verification', mock_verify):
            req = _make_request('6', 'verify-fail', worktree, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked'
        assert 'verification failed' in outcome.reason.lower()

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_cas_retry_limit_exhausted(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """CAS failures beyond MAX_CAS_RETRIES resolve as blocked."""
        worktree = (await git_ops.create_worktree('cas-limit')).path
        (worktree / 'limit.py').write_text('limit = True\n')
        await git_ops.commit(worktree, 'Add limit file')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # advance_main always returns cas_failed
        async def _always_cas_fail(*args, **kwargs):
            return 'cas_failed'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_always_cas_fail),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('7', 'cas-limit', worktree, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked'
        assert 'cas retry limit' in outcome.reason.lower()

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_not_descendant_returns_blocked_immediately(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Permanent not_descendant failure blocks without re-enqueue."""
        worktree = (await git_ops.create_worktree('perm-fail')).path
        (worktree / 'perm.py').write_text('perm = True\n')
        await git_ops.commit(worktree, 'Add perm file')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        call_count = 0

        async def _not_descendant(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return 'not_descendant'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_not_descendant),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('8', 'perm-fail', worktree, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=10)

        assert outcome.status == 'blocked'
        assert 'not_descendant' in outcome.reason
        # Should only be called once — no re-enqueue for permanent failures
        assert call_count == 1

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_merge_worker_emits_duration_ms_on_done(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """MergeWorker emits duration_ms on the 'done' outcome.

        Asserts that the merge_attempt event row for outcome='done' has a
        non-null integer duration_ms >= 0.
        """
        db_path = tmp_path / 'events.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        wt = await _make_branch_with_file(
            git_ops, 'dur-done', 'dur_done.py', 'dur = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=event_store)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req = _make_request('dur-done', 'dur-done', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done'

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT json_extract(data, '$.outcome') AS outcome, duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()

        done_rows = [r for r in rows if r[0] == 'done']
        assert len(done_rows) == 1, f'Expected 1 done row, got: {rows}'
        assert done_rows[0][1] is not None, 'duration_ms should not be NULL'
        assert isinstance(done_rows[0][1], int), f'duration_ms should be int, got {type(done_rows[0][1])}'
        assert done_rows[0][1] >= 0, f'duration_ms should be >= 0, got {done_rows[0][1]}'

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_merge_worker_emits_duration_ms_on_non_done_outcomes(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """Every MergeWorker emit site sets a non-null duration_ms.

        Covers already_merged, conflict, and cas_retry outcomes in addition
        to done (covered by test_merge_worker_emits_duration_ms_on_done).
        """
        # --- Scenario A: already_merged ---
        db_a = tmp_path / 'events_a.db'
        es_a = EventStore(db_path=db_a, run_id='run-a')

        wt_am = await _make_branch_with_file(
            git_ops, 'dur-am', 'dur_am.py', 'am = 1\n',
        )
        # Merge manually so it's already on main
        r = await git_ops.merge_to_main(wt_am, 'dur-am')
        assert r.success
        assert r.merge_commit is not None
        await git_ops.advance_main(r.merge_commit)
        if r.merge_worktree:
            await git_ops.cleanup_merge_worktree(r.merge_worktree)

        q_a: asyncio.Queue[MergeRequest] = asyncio.Queue()
        w_a = MergeWorker(git_ops, q_a, event_store=es_a)
        wt_a = asyncio.create_task(w_a.run())

        req_am = _make_request('dur-am', 'dur-am', wt_am, config)
        await q_a.put(req_am)
        out_am = await asyncio.wait_for(req_am.result, timeout=30)
        assert out_am.status == 'already_merged'
        await w_a.stop()
        wt_a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wt_a

        conn = sqlite3.connect(str(db_a))
        rows_a = conn.execute(
            "SELECT json_extract(data, '$.outcome'), duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()
        assert all(r[1] is not None for r in rows_a), f'NULL duration_ms in already_merged: {rows_a}'

        # --- Scenario B: conflict ---
        db_b = tmp_path / 'events_b.db'
        es_b = EventStore(db_path=db_b, run_id='run-b')

        wt_cfl = (await git_ops.create_worktree('dur-cfl')).path
        # Advance main with a conflicting change
        (git_ops.project_root / 'README.md').write_text('# conflict-source\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Conflict source'], cwd=git_ops.project_root)
        # Make conflicting change in worktree
        (wt_cfl / 'README.md').write_text('# conflict-task\n')
        await git_ops.commit(wt_cfl, 'Conflict task change')

        q_b: asyncio.Queue[MergeRequest] = asyncio.Queue()
        w_b = MergeWorker(git_ops, q_b, event_store=es_b)
        wt_b = asyncio.create_task(w_b.run())

        req_cfl = _make_request('dur-cfl', 'dur-cfl', wt_cfl, config)
        await q_b.put(req_cfl)
        out_cfl = await asyncio.wait_for(req_cfl.result, timeout=30)
        assert out_cfl.status == 'conflict'
        await w_b.stop()
        wt_b.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wt_b

        conn = sqlite3.connect(str(db_b))
        rows_b = conn.execute(
            "SELECT json_extract(data, '$.outcome'), duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()
        assert all(r[1] is not None for r in rows_b), f'NULL duration_ms in conflict: {rows_b}'

        # --- Scenario C: cas_retry ---
        db_c = tmp_path / 'events_c.db'
        es_c = EventStore(db_path=db_c, run_id='run-c')

        wt_cas = await _make_branch_with_file(
            git_ops, 'dur-cas', 'dur_cas.py', 'cas = 1\n',
        )

        q_c: asyncio.Queue[MergeRequest] = asyncio.Queue()
        w_c = MergeWorker(git_ops, q_c, event_store=es_c)
        wt_c = asyncio.create_task(w_c.run())

        original_advance = git_ops.advance_main
        call_count_c = 0

        async def _fail_once(*args, **kwargs):
            nonlocal call_count_c
            call_count_c += 1
            if call_count_c == 1:
                return 'cas_failed'
            return await original_advance(*args, **kwargs)

        with (
            patch.object(git_ops, 'advance_main', side_effect=_fail_once),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req_cas = _make_request('dur-cas', 'dur-cas', wt_cas, config)
            await q_c.put(req_cas)
            out_cas = await asyncio.wait_for(req_cas.result, timeout=30)

        assert out_cas.status == 'done'
        await w_c.stop()
        wt_c.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wt_c

        conn = sqlite3.connect(str(db_c))
        rows_c = conn.execute(
            "SELECT json_extract(data, '$.outcome'), duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()
        assert all(r[1] is not None for r in rows_c), f'NULL duration_ms in cas scenario: {rows_c}'

    async def test_merge_worker_success_returns_merge_sha(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """MergeWorker success path: MergeOutcome.merge_sha is the merge commit.

        Drives the full CAS-advance path through MergeWorker and asserts that
        the resulting MergeOutcome carries the real 40-char merge commit SHA.
        Fails initially because merge_queue.py:400 still constructs
        MergeOutcome('done') without the SHA (step-3 guard; impl in step-4).
        """
        worktree = await _make_branch_with_file(
            git_ops, 'sha-basic', 'sha_basic.py', 'sha = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req = _make_request('sha-task', 'sha-basic', worktree, config)
            await queue.put(req)
            result = await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

        assert result.status == 'done'
        assert result.merge_sha is not None, 'merge_sha must be set on done outcome'
        assert len(result.merge_sha) == 40, f'expected 40-char SHA, got: {result.merge_sha!r}'
        assert all(c in '0123456789abcdef' for c in result.merge_sha), (
            f'merge_sha is not a hex string: {result.merge_sha!r}'
        )


# ---------------------------------------------------------------------------
# Helpers for speculative tests
# ---------------------------------------------------------------------------

async def _make_branch_with_file(
    git_ops: GitOps,
    branch_name: str,
    filename: str,
    content: str,
) -> Path:
    """Create a worktree branch with one committed file and return its path."""
    worktree = (await git_ops.create_worktree(branch_name)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


# ---------------------------------------------------------------------------
# TestSpeculativeMergeWorker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSpeculativeMergeWorker:
    async def test_speculative_basic_throughput(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Submit 2 merge requests. Both complete as 'done', both files on main.

        N+1 is speculatively merged against N's merge SHA (not original main).
        Both complete without error.
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'spec-n', 'file_n.py', 'n = 1\n',
        )
        wt_n1 = await _make_branch_with_file(
            git_ops, 'spec-n1', 'file_n1.py', 'n1 = 2\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            _mock_verify_pass(),
        ):
            req_n = _make_request('spec-n', 'spec-n', wt_n, config)
            req_n1 = _make_request('spec-n1', 'spec-n1', wt_n1, config)

            # Submit both before the worker processes them
            await queue.put(req_n)
            await queue.put(req_n1)

            outcome_n = await asyncio.wait_for(req_n.result, timeout=60)
            outcome_n1 = await asyncio.wait_for(req_n1.result, timeout=60)

        assert outcome_n.status == 'done', f'N failed: {outcome_n}'
        assert outcome_n1.status == 'done', f'N+1 failed: {outcome_n1}'

        # Both files must appear on main
        _, out_n, _ = await _run(
            ['git', 'show', 'main:file_n.py'], cwd=git_ops.project_root,
        )
        assert 'n = 1' in out_n

        _, out_n1, _ = await _run(
            ['git', 'show', 'main:file_n1.py'], cwd=git_ops.project_root,
        )
        assert 'n1 = 2' in out_n1

        await worker.stop()
        await worker_task

    async def test_speculative_discard_on_failure(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """When N's verification fails, N+1's speculative merge is discarded
        and re-merged against actual main.  N returns 'blocked', N+1 returns
        'done' after the fresh re-merge.
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'disc-n', 'file_disc_n.py', 'disc_n = 1\n',
        )
        wt_n1 = await _make_branch_with_file(
            git_ops, 'disc-n1', 'file_disc_n1.py', 'disc_n1 = 2\n',
        )

        # Track how many times verify is called per task
        verify_calls: dict[str, int] = {}

        async def _verify_side_effect(
            merge_wt, cfg, module_configs, task_files=None, **_kwargs,
        ):
            # Determine which task by looking at which file is present
            n_file = merge_wt / 'file_disc_n.py'
            if n_file.exists():
                verify_calls['n'] = verify_calls.get('n', 0) + 1
                return MagicMock(passed=False, summary='N tests failed')
            else:
                verify_calls['n1'] = verify_calls.get('n1', 0) + 1
                return MagicMock(passed=True, summary='')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=_verify_side_effect,
        ):
            req_n = _make_request('disc-n', 'disc-n', wt_n, config)
            req_n1 = _make_request('disc-n1', 'disc-n1', wt_n1, config)
            await queue.put(req_n)
            await queue.put(req_n1)

            outcome_n = await asyncio.wait_for(req_n.result, timeout=60)
            outcome_n1 = await asyncio.wait_for(req_n1.result, timeout=60)

        assert outcome_n.status == 'blocked', f'N should be blocked: {outcome_n}'
        assert outcome_n1.status == 'done', f'N+1 should succeed after re-merge: {outcome_n1}'

        # N+1's file must appear on main (re-merged and advanced)
        _, out_n1, _ = await _run(
            ['git', 'show', 'main:file_disc_n1.py'], cwd=git_ops.project_root,
        )
        assert 'disc_n1 = 2' in out_n1

        # N's file must NOT be on main (N was blocked)
        rc, _, _ = await _run(
            ['git', 'cat-file', '-e', 'main:file_disc_n.py'],
            cwd=git_ops.project_root,
        )
        assert rc != 0, 'N file should not be on main'

        await worker.stop()
        await worker_task

    async def test_speculative_depth_cap(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """With depth-1 cap, N+2 is not speculatively merged until N+1's
        speculation resolves.  Submit N, N+1, N+2 — all complete as 'done'.

        Verified by tracking concurrent active merge worktrees: the count must
        never exceed 2 (N's worktree while N is being verified, plus N+1's
        speculative worktree).  N+2 is only started after N finishes.
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'cap-n', 'file_cap_n.py', 'cap_n = 1\n',
        )
        wt_n1 = await _make_branch_with_file(
            git_ops, 'cap-n1', 'file_cap_n1.py', 'cap_n1 = 2\n',
        )
        wt_n2 = await _make_branch_with_file(
            git_ops, 'cap-n2', 'file_cap_n2.py', 'cap_n2 = 3\n',
        )

        # Track maximum number of merge worktrees active at the same time.
        # Each merge worktree is created in _create_merge_worktree and removed
        # in cleanup_merge_worktree.  With depth-1, the peak must be ≤ 2.
        active_worktrees: set[str] = set()
        max_concurrent = 0
        original_create = git_ops._create_merge_worktree
        original_cleanup = git_ops.cleanup_merge_worktree

        async def _tracking_create(base_sha=None):
            wt, sha = await original_create(base_sha)
            active_worktrees.add(str(wt))
            nonlocal max_concurrent
            max_concurrent = max(max_concurrent, len(active_worktrees))
            return wt, sha

        async def _tracking_cleanup(merge_wt):
            active_worktrees.discard(str(merge_wt))
            await original_cleanup(merge_wt)

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with (
            patch.object(git_ops, '_create_merge_worktree', side_effect=_tracking_create),
            patch.object(git_ops, 'cleanup_merge_worktree', side_effect=_tracking_cleanup),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req_n = _make_request('cap-n', 'cap-n', wt_n, config)
            req_n1 = _make_request('cap-n1', 'cap-n1', wt_n1, config)
            req_n2 = _make_request('cap-n2', 'cap-n2', wt_n2, config)
            await queue.put(req_n)
            await queue.put(req_n1)
            await queue.put(req_n2)

            outcome_n = await asyncio.wait_for(req_n.result, timeout=60)
            outcome_n1 = await asyncio.wait_for(req_n1.result, timeout=60)
            outcome_n2 = await asyncio.wait_for(req_n2.result, timeout=60)

        assert outcome_n.status == 'done', f'N: {outcome_n}'
        assert outcome_n1.status == 'done', f'N+1: {outcome_n1}'
        assert outcome_n2.status == 'done', f'N+2: {outcome_n2}'

        # Depth-1 cap: at most 2 merge worktrees active simultaneously
        # (the item being verified + 1 speculative item)
        assert max_concurrent <= 2, (
            f'Depth-1 cap violated: {max_concurrent} concurrent merge worktrees '
            f'(expected ≤ 2)'
        )

        # All three files on main
        for fname in ('file_cap_n.py', 'file_cap_n1.py', 'file_cap_n2.py'):
            rc, _, _ = await _run(
                ['git', 'cat-file', '-e', f'main:{fname}'],
                cwd=git_ops.project_root,
            )
            assert rc == 0, f'{fname} not on main'

        await worker.stop()
        await worker_task

    async def test_speculative_single_item_degenerates(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Single merge request through SpeculativeMergeWorker completes as 'done'.

        Confirms the speculative pipeline degenerates to serial behavior when
        there is only one item in the queue (no look-ahead possible).
        """
        wt = await _make_branch_with_file(
            git_ops, 'single', 'single.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req = _make_request('single', 'single', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done'
        _, out, _ = await _run(['git', 'show', 'main:single.py'], cwd=git_ops.project_root)
        assert 'x = 1' in out

        await worker.stop()
        await worker_task

    async def test_speculative_verify_called_with_max_retries_zero(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Merge-queue post-merge verify must pass max_retries=0.

        A deterministic verify hang would otherwise be retried per
        ``config.verify_timeout_retries``, tripling queue-wide stall.
        This is the regression that caused the 2026-04-20 90-minute jam.
        """
        wt = await _make_branch_with_file(
            git_ops, 'retry0', 'retry0.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        captured_kwargs: list[dict] = []

        async def spy_verify(*args, **kwargs):
            captured_kwargs.append(kwargs)
            result = AsyncMock()
            result.passed = True
            result.summary = ''
            return result

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=spy_verify,
        ):
            req = _make_request('retry0', 'retry0', wt, config)
            await queue.put(req)
            await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        await worker_task

        assert captured_kwargs, 'run_scoped_verification was not invoked'
        assert captured_kwargs[0].get('max_retries') == 0, (
            f'merge-queue verify must pass max_retries=0; got {captured_kwargs[0]!r}'
        )

    async def test_speculative_verify_called_with_role_merge(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Merge-queue post-merge verify must pass role='merge' to run_scoped_verification.

        DF_VERIFY_ROLE=merge is injected so reify's verify.sh can apply
        the merge-role priority prefix (nice -n 5) for OCCT throttling.
        """
        wt = await _make_branch_with_file(
            git_ops, 'rolem', 'rolem.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        captured_kwargs: list[dict] = []

        async def spy_verify(*args, **kwargs):
            captured_kwargs.append(kwargs)
            result = AsyncMock()
            result.passed = True
            result.summary = ''
            return result

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=spy_verify,
        ):
            req = _make_request('rolem', 'rolem', wt, config)
            await queue.put(req)
            await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        await worker_task

        assert captured_kwargs, 'run_scoped_verification was not invoked'
        assert captured_kwargs[0].get('role') == 'merge', (
            f"merge-queue verify must pass role='merge'; got {captured_kwargs[0]!r}"
        )

    async def test_speculative_shutdown_drains_both(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """stop() resolves all pending Futures as 'blocked' with shutdown reason."""
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        # Don't start worker — just queue items and stop
        wt_a = (await git_ops.create_worktree('shut-a')).path
        wt_b = (await git_ops.create_worktree('shut-b')).path
        req_a = _make_request('shut-a', 'shut-a', wt_a, config)
        req_b = _make_request('shut-b', 'shut-b', wt_b, config)
        await queue.put(req_a)
        await queue.put(req_b)

        await worker.stop()

        assert req_a.result.done()
        assert req_b.result.done()
        outcome_a = req_a.result.result()
        outcome_b = req_b.result.result()
        assert outcome_a.status == 'blocked'
        assert outcome_b.status == 'blocked'
        assert 'shutting down' in outcome_a.reason.lower()
        assert 'shutting down' in outcome_b.reason.lower()

    async def test_speculative_conflict_n_plus_1(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """N merges cleanly; N+1 conflicts when speculatively merged.
        N completes as 'done'; N+1 returns 'conflict'.
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'cfl-n', 'file_cfl_n.py', 'cfl_n = 1\n',
        )

        # Create N+1 worktree from current main, then advance main via
        # a direct commit to cause a conflict on the same file.
        wt_n1 = (await git_ops.create_worktree('cfl-n1')).path
        # Write conflicting content to README.md in both main and wt_n1
        (git_ops.project_root / 'README.md').write_text('# Main conflict\n')
        await _run(['git', 'add', 'README.md'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Main side change'], cwd=git_ops.project_root)
        (wt_n1 / 'README.md').write_text('# N+1 conflict\n')
        await git_ops.commit(wt_n1, 'N+1 conflicting change')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req_n = _make_request('cfl-n', 'cfl-n', wt_n, config)
            req_n1 = _make_request('cfl-n1', 'cfl-n1', wt_n1, config)
            await queue.put(req_n)
            await queue.put(req_n1)

            outcome_n = await asyncio.wait_for(req_n.result, timeout=60)
            outcome_n1 = await asyncio.wait_for(req_n1.result, timeout=60)

        assert outcome_n.status == 'done', f'N: {outcome_n}'
        assert outcome_n1.status == 'conflict', f'N+1: {outcome_n1}'
        assert outcome_n1.conflict_details

        await worker.stop()
        await worker_task

    async def test_speculative_events_emitted(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """speculative_merge event emitted when N+1 is speculatively merged.
        speculative_discard event emitted when N fails and N+1 is discarded.
        """
        db_path = tmp_path / 'events.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        wt_n = await _make_branch_with_file(
            git_ops, 'ev-n', 'file_ev_n.py', 'ev_n = 1\n',
        )
        wt_n1 = await _make_branch_with_file(
            git_ops, 'ev-n1', 'file_ev_n1.py', 'ev_n1 = 2\n',
        )

        async def _fail_n_pass_n1(merge_wt, cfg, module_configs, task_files=None):
            n_present = (merge_wt / 'file_ev_n.py').exists()
            n1_present = (merge_wt / 'file_ev_n1.py').exists()
            # Speculative verify of N: N present, N+1 not yet merged → fail.
            # Any other shape means N+1 was re-verified after N failed (which
            # contradicts the discard-on-failure contract) — fail loudly
            # rather than silently returning pass.
            if n_present and not n1_present:
                return MagicMock(passed=False, summary='N failed')
            raise AssertionError(
                f'unexpected verify call: n_present={n_present}, '
                f'n1_present={n1_present} — N+1 should have been discarded '
                f'after N failed, not re-verified'
            )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=event_store)
        worker_task = asyncio.create_task(worker.run())

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=_fail_n_pass_n1,
        ):
            req_n = _make_request('ev-n', 'ev-n', wt_n, config)
            req_n1 = _make_request('ev-n1', 'ev-n1', wt_n1, config)
            await queue.put(req_n)
            await queue.put(req_n1)
            await asyncio.wait_for(req_n.result, timeout=60)
            await asyncio.wait_for(req_n1.result, timeout=60)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, task_id FROM events ORDER BY id"
        ).fetchall()
        conn.close()

        event_types = [r[0] for r in rows]
        assert 'speculative_merge' in event_types, f'No speculative_merge event: {event_types}'
        assert 'speculative_discard' in event_types, f'No speculative_discard event: {event_types}'

        await worker.stop()
        await worker_task

    async def test_speculative_already_merged_n_plus_1(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """N+1 branch already on main → returns 'already_merged' without
        attempting a speculative merge.
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'am-n', 'file_am_n.py', 'am_n = 1\n',
        )
        # Create N+1 as already-merged: merge it first, then submit it
        wt_n1 = (await git_ops.create_worktree('am-n1')).path
        (wt_n1 / 'file_am_n1.py').write_text('am_n1 = 2\n')
        await git_ops.commit(wt_n1, 'N+1 file')
        result = await git_ops.merge_to_main(wt_n1, 'am-n1')
        assert result.success
        assert result.merge_commit is not None
        await git_ops.advance_main(result.merge_commit)
        if result.merge_worktree:
            await git_ops.cleanup_merge_worktree(result.merge_worktree)

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req_n = _make_request('am-n', 'am-n', wt_n, config)
            req_n1 = _make_request('am-n1', 'am-n1', wt_n1, config)
            await queue.put(req_n)
            await queue.put(req_n1)

            outcome_n = await asyncio.wait_for(req_n.result, timeout=60)
            outcome_n1 = await asyncio.wait_for(req_n1.result, timeout=60)

        assert outcome_n.status == 'done', f'N: {outcome_n}'
        assert outcome_n1.status == 'already_merged', f'N+1: {outcome_n1}'

        await worker.stop()
        await worker_task

    async def test_verifier_exception_releases_speculation_slot(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """_verify_and_advance raising must resolve N's Future and release the slot.

        Without the try/except/finally fix: the verifier loop crashes before
        calling _speculation_slot.set() and before resolving N's Future, causing
        both a deadlock (merger blocked waiting for slot) and a hung Future.

        With the fix: except clause resolves N's Future as 'blocked' with a
        'Verifier error' reason; finally clause always sets _speculation_slot.
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'vex-n', 'file_vex_n.py', 'n = 1\n',
        )
        wt_n1 = await _make_branch_with_file(
            git_ops, 'vex-n1', 'file_vex_n1.py', 'n1 = 2\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # Capture original before replacing with mock
        original_vaa = worker._verify_and_advance
        call_count = 0

        async def mock_vaa(item):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError('Unexpected verifier error')
            return await original_vaa(item)

        worker._verify_and_advance = mock_vaa  # type: ignore[method-assign]

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req_n = _make_request('vex-n', 'vex-n', wt_n, config)
            req_n1 = _make_request('vex-n1', 'vex-n1', wt_n1, config)
            await queue.put(req_n)
            await queue.put(req_n1)

            # N must resolve as 'blocked' with 'Verifier error' (not hang forever)
            outcome_n = await asyncio.wait_for(req_n.result, timeout=30)
            assert outcome_n.status == 'blocked', f'N: {outcome_n}'
            assert 'Verifier error' in outcome_n.reason, (
                f'Expected Verifier error in reason, got: {outcome_n.reason}'
            )
            assert 'Unexpected verifier error' in outcome_n.reason

            # _speculation_slot must be set (not stuck cleared → deadlock)
            assert worker._speculation_slot.is_set(), (
                '_speculation_slot stuck cleared — merger will deadlock on next request'
            )

            # N+1 must also complete (not hang forever)
            outcome_n1 = await asyncio.wait_for(req_n1.result, timeout=30)
            assert outcome_n1.status in ('done', 'blocked'), f'N+1: {outcome_n1}'

        await worker.stop()
        await worker_task

    async def test_verifier_remerge_exception_releases_slot(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """_remerge raising must resolve N+1's Future and release the speculation slot.

        Scenario: N fails verification (n_failed=True), N+1 is speculative.
        The verifier calls _remerge(N+1) which raises unexpectedly.

        Without fix: exception propagates out of loop body, N+1's Future is never
        resolved, _speculation_slot may be left cleared → downstream deadlock.

        With fix: except clause resolves N+1's Future as 'blocked' with
        'Verifier error'; finally clause always sets _speculation_slot.
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'vre-n', 'file_vre_n.py', 'n = 1\n',
        )
        wt_n1 = await _make_branch_with_file(
            git_ops, 'vre-n1', 'file_vre_n1.py', 'n1 = 2\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # N fails verification → n_failed=True; _remerge then raises for N+1
        mock_verify = AsyncMock(return_value=MagicMock(passed=False, summary='tests failed'))

        async def raise_on_remerge(req, started_monotonic: float | None = None):  # type: ignore[no-untyped-def]
            raise RuntimeError('_remerge failed unexpectedly')

        worker._remerge = raise_on_remerge  # type: ignore[method-assign]

        with patch('orchestrator.merge_queue.run_scoped_verification', mock_verify):
            req_n = _make_request('vre-n', 'vre-n', wt_n, config)
            req_n1 = _make_request('vre-n1', 'vre-n1', wt_n1, config)
            await queue.put(req_n)
            await queue.put(req_n1)

            # N fails verification → blocked
            outcome_n = await asyncio.wait_for(req_n.result, timeout=30)
            assert outcome_n.status == 'blocked', f'N: {outcome_n}'

            # N+1: _remerge raised → 'blocked' with Verifier error (not hang)
            outcome_n1 = await asyncio.wait_for(req_n1.result, timeout=30)
            assert outcome_n1.status == 'blocked', f'N+1: {outcome_n1}'
            assert 'Verifier error' in outcome_n1.reason, (
                f'Expected Verifier error in N+1 reason, got: {outcome_n1.reason}'
            )
            assert '_remerge failed' in outcome_n1.reason

            # _speculation_slot must be released
            assert worker._speculation_slot.is_set(), (
                '_speculation_slot stuck cleared after _remerge exception'
            )

        await worker.stop()
        await worker_task

    async def test_run_cancels_subtasks_on_cancellation(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Cancelling run()'s outer task must cancel both _merger_task and _verifier_task.

        Without fix: run() only catches CancelledError and re-raises, leaving
        the subtasks running (orphaned). If one subtask raises RuntimeError,
        the other continues running forever.

        With fix: any BaseException cancels both subtasks before re-raising.
        """
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        # ── Part 1: outer task cancellation ──────────────────────────────
        worker_task = asyncio.create_task(worker.run())
        # Give merger and verifier tasks a chance to start
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert worker._merger_task is not None
        assert worker._verifier_task is not None

        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

        assert worker._merger_task.done(), 'merger_task not done after cancellation'
        assert worker._verifier_task.done(), 'verifier_task not done after cancellation'
        assert worker._merger_task.cancelled() or worker._merger_task.exception() is not None
        assert worker._verifier_task.cancelled() or worker._verifier_task.exception() is not None

        # ── Part 2: subtask RuntimeError cancels sibling ─────────────────
        worker2 = SpeculativeMergeWorker(git_ops, asyncio.Queue())

        async def crashing_merger():
            raise RuntimeError('Merger crashed unexpectedly')

        worker2._merger_loop = crashing_merger  # type: ignore[method-assign]

        worker_task2 = asyncio.create_task(worker2.run())
        # Allow merger to crash
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        with pytest.raises((RuntimeError, asyncio.CancelledError)):
            await asyncio.wait_for(worker_task2, timeout=5)

        assert worker2._verifier_task is not None
        assert worker2._verifier_task.done(), (
            'verifier_task not cancelled after merger RuntimeError'
        )

    async def test_merger_exception_sends_verifier_sentinel(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """_merger_loop must put None sentinel into verifier queue even when it crashes.

        The inner try/except (step-31) catches Exception, resolves the Future as
        'blocked', and continues the loop (rather than propagating the exception).
        The loop then exits cleanly via a shutdown sentinel. The try/finally wrapping
        the entire while-loop guarantees the verifier sentinel is always sent.

        We test _merger_loop() directly to isolate this from the run() subtask
        cancellation logic (step-24), which also terminates the verifier.
        """
        wt = await _make_branch_with_file(
            git_ops, 'mes-n', 'file_mes_n.py', 'n = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        # merge_to_main raises on the first call to simulate an unexpected crash
        async def crash_merge(worktree, branch, base_sha=None):  # type: ignore[no-untyped-def]
            raise RuntimeError('Unexpected error in merge_to_main')

        req = _make_request('mes-n', 'mes-n', wt, config)
        await queue.put(req)
        # Shutdown sentinel so the loop exits after handling the exception —
        # the inner except catches RuntimeError and continues, so without this
        # the loop would block on queue.get() forever.
        await queue.put(None)  # type: ignore[arg-type]

        with patch.object(git_ops, 'merge_to_main', new=crash_merge):
            await worker._merger_loop()

        # (1) Future must be resolved as 'blocked' by the inner exception handler.
        assert req.result.done(), (
            'Future must be resolved when merger catches an unexpected exception'
        )
        assert req.result.result().status == 'blocked'
        assert 'Merger error' in req.result.result().reason

        # (2) The verifier queue must contain the sentinel (None).
        # Without the try/finally fix, the queue would be empty here.
        assert not worker._verifier_queue.empty(), (
            'Verifier queue is empty — sentinel was never sent by dying merger'
        )
        sentinel = worker._verifier_queue.get_nowait()
        assert sentinel is None, f'Expected sentinel (None), got: {sentinel}'

    async def test_revparse_failure_produces_blocked(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """git rev-parse HEAD failure must resolve Future as blocked (not crash).

        Without fix: the return code from git rev-parse HEAD is not checked.
        A non-zero rc leaves branch_head as empty/garbage, and the subsequent
        is_ancestor() call may crash or behave incorrectly.

        With fix: rc != 0 triggers an immediate blocked outcome pushed to the
        verifier queue with reason 'rev-parse HEAD failed: <err>'. Subsequent
        requests are still processed normally.
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'rp-n', 'file_rp_n.py', 'n = 1\n',
        )
        wt_ok = await _make_branch_with_file(
            git_ops, 'rp-ok', 'file_rp_ok.py', 'ok = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # Mock _run to fail rev-parse for rp-n worktree only
        original_run = __import__(
            'orchestrator.merge_queue', fromlist=['_run']
        )._run
        # We patch the module-level _run used inside merge_queue
        call_log: list[tuple] = []

        async def mock_run(cmd, cwd=None, **kwargs):  # type: ignore[no-untyped-def]
            call_log.append(tuple(cmd))
            if cmd[:2] == ['git', 'rev-parse'] and cwd == wt_n:
                return (1, '', 'fatal: not a git repository')
            return await original_run(cmd, cwd=cwd, **kwargs)

        with (
            patch('orchestrator.merge_queue._run', new=mock_run),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req_n = _make_request('rp-n', 'rp-n', wt_n, config)
            req_ok = _make_request('rp-ok', 'rp-ok', wt_ok, config)
            await queue.put(req_n)
            await queue.put(req_ok)

            # rp-n must resolve as blocked with rev-parse reason
            outcome_n = await asyncio.wait_for(req_n.result, timeout=30)
            assert outcome_n.status == 'blocked', f'rp-n: {outcome_n}'
            assert 'rev-parse' in outcome_n.reason.lower(), (
                f'Expected rev-parse in reason: {outcome_n.reason}'
            )

            # rp-ok must still succeed (merger loop continues after the error)
            outcome_ok = await asyncio.wait_for(req_ok.result, timeout=30)
            assert outcome_ok.status == 'done', f'rp-ok: {outcome_ok}'

        await worker.stop()
        await worker_task

    async def test_merger_exception_resolves_inflight_future(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Unexpected exception in merger loop must resolve in-flight Future and continue.

        Without fix: if get_main_sha() raises after req is dequeued but before the
        SpeculativeItem is pushed to the verifier queue, the exception propagates to
        the outer try/finally which sends the sentinel but never resolves req.result.
        The caller hangs forever and the merger loop terminates.

        With fix: inner try/except Exception in the loop body resolves the in-flight
        req.result as 'blocked' (with the error message) and continues to the next
        request, keeping the merger loop alive.
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'mef-n', 'file_mef_n.py', 'n = 1\n',
        )
        wt_ok = await _make_branch_with_file(
            git_ops, 'mef-ok', 'file_mef_ok.py', 'ok = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # get_main_sha raises RuntimeError on the first call, succeeds after
        original_get_main_sha = git_ops.get_main_sha
        call_count = 0

        async def failing_get_main_sha():  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError('Simulated get_main_sha failure')
            return await original_get_main_sha()

        with (
            patch.object(git_ops, 'get_main_sha', new=failing_get_main_sha),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req_n = _make_request('mef-n', 'mef-n', wt_n, config)
            req_ok = _make_request('mef-ok', 'mef-ok', wt_ok, config)
            await queue.put(req_n)
            await queue.put(req_ok)

            # mef-n must resolve as 'blocked' with reason mentioning the error
            # (not hang forever — that's the regression without the fix)
            outcome_n = await asyncio.wait_for(req_n.result, timeout=30)
            assert outcome_n.status == 'blocked', f'mef-n: {outcome_n}'
            assert 'Simulated get_main_sha failure' in outcome_n.reason, (
                f'Expected error message in reason, got: {outcome_n.reason}'
            )

            # mef-ok must succeed — merger loop continues after the per-request error
            outcome_ok = await asyncio.wait_for(req_ok.result, timeout=30)
            assert outcome_ok.status == 'done', f'mef-ok: {outcome_ok}'

        await worker.stop()
        await worker_task

    async def test_stop_drain_survives_cleanup_exception(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Cleanup exception during stop() verifier-queue drain must not orphan Futures.

        Without fix: if cleanup_merge_worktree raises for item1, the exception
        propagates out of the drain loop body, so item2's Future is never resolved —
        the caller hangs forever.

        With fix: cleanup is wrapped in contextlib.suppress(Exception), so the drain
        loop continues to item2 and resolves both Futures as 'blocked'.
        Covers review issue [exception_aborts_drain] at stop() ~line 367-376.
        """
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        # Do NOT start worker.run() — we test stop()'s drain logic directly.

        # Build two requests whose Futures we will check after stop().
        req1 = _make_request('drain-1', 'drain-1', git_ops.project_root, config)
        req2 = _make_request('drain-2', 'drain-2', git_ops.project_root, config)

        dummy_wt1 = git_ops.project_root / '.worktrees' / 'dummy1'
        dummy_wt2 = git_ops.project_root / '.worktrees' / 'dummy2'

        item1 = SpeculativeItem(
            request=req1, merge_result=None, merge_wt=dummy_wt1,
            base_sha='aaa', speculative=False, skip_verify=False,
        )
        item2 = SpeculativeItem(
            request=req2, merge_result=None, merge_wt=dummy_wt2,
            base_sha='bbb', speculative=False, skip_verify=False,
        )
        await worker._verifier_queue.put(item1)
        await worker._verifier_queue.put(item2)

        # First cleanup raises OSError; second succeeds.
        cleanup_calls: list[object] = []

        async def mock_cleanup(wt: object) -> None:
            cleanup_calls.append(wt)
            if len(cleanup_calls) == 1:
                raise OSError('disk full')

        with patch.object(git_ops, 'cleanup_merge_worktree', new=mock_cleanup):
            await worker.stop()

        # Both Futures must be resolved — cleanup failure must not abort the drain.
        assert req1.result.done(), 'req1 Future not resolved despite cleanup exception'
        assert req2.result.done(), 'req2 Future orphaned because drain loop aborted'
        assert req1.result.result().status == 'blocked'
        assert req2.result.result().status == 'blocked'
        # Second cleanup was still attempted despite first failure.
        assert len(cleanup_calls) == 2, (
            f'Expected 2 cleanup calls, got {len(cleanup_calls)}'
        )

    async def test_stop_race_resolves_inflight_merger_future(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """stop() must resolve Future for a request the merger is currently processing.

        Race condition: stop() drains both queues (empty), sends sentinels,
        asyncio.wait() times out while merger is still blocked inside merge_to_main.
        Verifier received its sentinel and has already exited.  When the merger
        eventually resumes and pushes a SpeculativeItem, the verifier is gone —
        the caller's Future is never resolved.

        With fix (step-35): after asyncio.wait() returns, stop() checks
        self._inflight_req.  If set and Future not done, resolves it as 'blocked'.
        The caller's Future is guaranteed to be resolved even if the merger was
        mid-operation when stop() was called.

        Covers review issue [race_condition_unresolved_future] at stop() ~line 350.
        """
        block_event = asyncio.Event()   # released after stop() returns
        merge_started = asyncio.Event() # set when merger enters merge_to_main

        original_merge = git_ops.merge_to_main

        async def blocking_merge(worktree: Path, branch: str, **kwargs: Any) -> Any:
            merge_started.set()
            await block_event.wait()  # simulates long-running merge
            return await original_merge(worktree, branch, **kwargs)

        wt = await _make_branch_with_file(
            git_ops, 'race-1', 'race_file.py', 'race = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        # Use a very short shutdown timeout so the test doesn't take 5 seconds.
        worker._shutdown_timeout = 0.1  # type: ignore[attr-defined]
        worker_task = asyncio.create_task(worker.run())

        req = _make_request('race-1', 'race-1', wt, config)

        with (
            patch.object(git_ops, 'merge_to_main', new=blocking_merge),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            await queue.put(req)
            # Wait until the merger is definitely blocked inside merge_to_main.
            await asyncio.wait_for(merge_started.wait(), timeout=10)

            # stop() will time out (asyncio.wait) since merger is blocked.
            # Without fix: req.result is NOT done after stop() returns.
            # With fix: stop() checks _inflight_req and resolves it.
            await worker.stop()

        assert req.result.done(), (
            'Future must be resolved by stop() via _inflight_req check, '
            'even when merger was mid-operation'
        )
        assert req.result.result().status == 'blocked'

        # Release the merger so it can finish and worker_task can exit cleanly.
        block_event.set()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(worker_task, timeout=15)

    async def test_speculative_chain_invalidation_propagates(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Chain invalidation must propagate: if N fails and N+1 is re-merged,
        N+2 (built speculatively on N+1's stale commit) must ALSO be re-merged.

        Scenario (depth-1 cap):
          - Queue has N, N+1, N+2 pre-loaded before worker starts.
          - Merger: merges N (non-spec), speculatively merges N+1 against N's
            merge commit, then awaits spec slot.
          - Verifier: N fails (has file_chain_n.py) → n_failed=True, releases slot.
          - Merger: grabs N+2, speculatively merges against N+1's STALE commit.
          - Verifier: N+1 discarded (n_failed=True), re-merged against actual main
            (no file_chain_n.py) → passes.  n_failed=False.  remerge_occurred=True.
          - Verifier: N+2 (speculative=True).
              WITHOUT FIX: n_failed=False → no discard → verification sees
              file_chain_n.py in speculative worktree → blocked.
              WITH FIX: remerge_occurred=True → discard → re-merge against actual
              main (only N+1, no N) → passes → done.

        Covers review issue [correctness_bug_in_speculative_chain_invalidation].
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'chain-n', 'file_chain_n.py', 'n = 1\n',
        )
        wt_n1 = await _make_branch_with_file(
            git_ops, 'chain-n1', 'file_chain_n1.py', 'n1 = 2\n',
        )
        wt_n2 = await _make_branch_with_file(
            git_ops, 'chain-n2', 'file_chain_n2.py', 'n2 = 3\n',
        )

        # Pre-load all three so the Merger builds a 3-deep speculative chain:
        # N (non-spec), N+1 (spec against N's commit), N+2 (spec against N+1's
        # stale commit once the Verifier releases the slot after N fails).
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        req_n = _make_request('chain-n', 'chain-n', wt_n, config)
        req_n1 = _make_request('chain-n1', 'chain-n1', wt_n1, config)
        req_n2 = _make_request('chain-n2', 'chain-n2', wt_n2, config)
        await queue.put(req_n)
        await queue.put(req_n1)
        await queue.put(req_n2)

        worker = SpeculativeMergeWorker(git_ops, queue)

        # Fail verification whenever file_chain_n.py is present (N's tainted code).
        # N's merge is non-spec → fail; N+2's speculative worktree descends from
        # N's commit → also has file_chain_n.py → would fail unless discarded first.
        async def _verify_chain(
            merge_wt, cfg, module_configs, task_files=None, **_kwargs,
        ):
            if (merge_wt / 'file_chain_n.py').exists():
                return MagicMock(passed=False, summary='N tainted: file_chain_n.py present')
            return MagicMock(passed=True, summary='')

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=_verify_chain,
        ):
            worker_task = asyncio.create_task(worker.run())
            outcome_n = await asyncio.wait_for(req_n.result, timeout=60)
            outcome_n1 = await asyncio.wait_for(req_n1.result, timeout=60)
            outcome_n2 = await asyncio.wait_for(req_n2.result, timeout=60)

        assert outcome_n.status == 'blocked', (
            f'N: expected blocked, got {outcome_n}'
        )
        assert outcome_n1.status == 'done', (
            f'N+1: expected done after re-merge against actual main, got {outcome_n1}'
        )
        assert outcome_n2.status == 'done', (
            f'N+2: expected done after chain-invalidation re-merge, got {outcome_n2}. '
            f'Without fix, N+2 is blocked because it was speculatively built on '
            f"N+1's stale commit (which contains file_chain_n.py from N)."
        )

        # Verify git state: N's tainted file must not be on main; N+1 and N+2 must be.
        _, ls_files, _ = await _run(
            ['git', 'ls-tree', '--name-only', 'main'], cwd=git_ops.project_root,
        )
        assert 'file_chain_n.py' not in ls_files, (
            'N (tainted) must not appear on main'
        )
        assert 'file_chain_n1.py' in ls_files, (
            'N+1 must appear on main after re-merge'
        )
        assert 'file_chain_n2.py' in ls_files, (
            'N+2 must appear on main after re-merge'
        )

        await worker.stop()
        await worker_task

    async def test_speculative_cas_failure_retries_until_advanced(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """CAS failure in SpeculativeMergeWorker retries and eventually succeeds.

        Mirrors MergeWorker.test_cas_failure_reenqueues_at_front but exercises
        the _verify_and_advance CAS-retry loop (which rebuilds SpeculativeItem
        with updated base_sha and tracks cumulative retries in _cas_retries).
        """
        wt = await _make_branch_with_file(
            git_ops, 'scas-ok', 'file_scas_ok.py', 'cas_ok = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        original_advance = git_ops.advance_main
        call_count = 0

        async def _fail_twice_then_succeed(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return 'cas_failed'
            return await original_advance(*args, **kwargs)

        with (
            patch.object(git_ops, 'advance_main', side_effect=_fail_twice_then_succeed),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('scas-ok', 'scas-ok', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done', f'Expected done, got {outcome}'
        assert call_count == 3, f'Expected 3 advance_main calls (2 CAS fail + 1 success), got {call_count}'
        # _cas_retries should be cleaned up after success
        assert 'scas-ok' not in worker._cas_retries, (
            '_cas_retries not cleaned up after successful advance'
        )

        # File must appear on main
        _, content, _ = await _run(
            ['git', 'show', 'main:file_scas_ok.py'], cwd=git_ops.project_root,
        )
        assert 'cas_ok = 1' in content

        await worker.stop()
        await worker_task

    async def test_speculative_cas_retry_limit_exhausted(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """CAS failures beyond MAX_CAS_RETRIES resolve as blocked.

        Mirrors MergeWorker.test_cas_retry_limit_exhausted but exercises the
        SpeculativeMergeWorker's _verify_and_advance loop, which tracks retries
        in self._cas_retries (a per-task dict shared across calls).
        """
        wt = await _make_branch_with_file(
            git_ops, 'scas-lim', 'file_scas_lim.py', 'cas_lim = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        async def _always_cas_fail(*args: Any, **kwargs: Any):
            return 'cas_failed'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_always_cas_fail),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('scas-lim', 'scas-lim', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked', f'Expected blocked, got {outcome}'
        assert 'cas retry limit' in outcome.reason.lower(), (
            f'Expected CAS retry limit message, got: {outcome.reason}'
        )
        # _cas_retries should be cleaned up after exhaustion
        assert 'scas-lim' not in worker._cas_retries, (
            '_cas_retries not cleaned up after retry limit exhausted'
        )

        await worker.stop()
        await worker_task

    @pytest.mark.parametrize('failure_code', ['not_descendant', 'contaminated', 'stash_failed'])
    async def test_speculative_permanent_failure_returns_blocked(
        self, git_ops: GitOps, config: OrchestratorConfig, failure_code: str,
    ):
        """Permanent advance_main failure codes block without retry.

        Mirrors MergeWorker.test_not_descendant_returns_blocked_immediately but
        exercises the SpeculativeMergeWorker's _verify_and_advance path (lines
        816-824 of merge_queue.py), which also cleans up merge worktree and
        resolves the Future.  Parameterized over all three permanent codes.
        """
        branch_name = f'sperm-{failure_code}'
        filename = f'file_sperm_{failure_code}.py'
        wt = await _make_branch_with_file(
            git_ops, branch_name, filename, f'{failure_code} = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        call_count = 0

        async def _return_failure(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1
            return failure_code

        with (
            patch.object(git_ops, 'advance_main', side_effect=_return_failure),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
            patch.object(git_ops, 'cleanup_merge_worktree', wraps=git_ops.cleanup_merge_worktree) as mock_cleanup,
        ):
            req = _make_request(branch_name, branch_name, wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked', f'Expected blocked for {failure_code}, got {outcome}'
        assert failure_code in outcome.reason, (
            f'Expected {failure_code} in reason, got: {outcome.reason}'
        )
        # Should only be called once — no retry for permanent failures
        assert call_count == 1, (
            f'Expected 1 advance_main call for permanent failure, got {call_count}'
        )
        # Worktree must have been cleaned up
        assert mock_cleanup.call_count >= 1, (
            f'cleanup_merge_worktree not called for {failure_code}'
        )
        # _cas_retries should be clean
        assert branch_name not in worker._cas_retries

        await worker.stop()
        await worker_task

    async def test_merger_post_merge_exception_cleans_worktree(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Post-merge exception (inside the success path) must clean up merge worktree.

        After merge_to_main succeeds, a live merge worktree exists.  An exception
        raised between lines 568-583 (assert, skip_verify calc, verifier queue put)
        is caught by the inner except Exception handler.  Without the fix the handler
        resolves the Future but never calls cleanup_merge_worktree — the worktree leaks.

        With the fix: cleanup is called (guarded by contextlib.suppress(Exception))
        before the Future is resolved.

        Scenario A: merge_commit=None triggers AssertionError at line 570.
        Scenario B: valid merge_commit but verifier-queue put raises RuntimeError.

        Covers review issue [resource_leak] at _merger_loop lines 568-614.
        """
        wt_a = await _make_branch_with_file(
            git_ops, 'pme-a', 'file_pme_a.py', 'a = 1\n',
        )
        wt_b = await _make_branch_with_file(
            git_ops, 'pme-b', 'file_pme_b.py', 'b = 1\n',
        )
        wt_ok = await _make_branch_with_file(
            git_ops, 'pme-ok', 'file_pme_ok.py', 'ok = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        fake_wt_a = git_ops.project_root / '.worktrees' / '_merge-pme-a-fake'
        fake_wt_b = git_ops.project_root / '.worktrees' / '_merge-pme-b-fake'

        cleanup_calls: list[object] = []

        async def tracking_cleanup(wt: object) -> None:
            cleanup_calls.append(wt)

        # ── Scenario A: AssertionError from merge_commit=None ──────────────────
        a_result = MergeResult(success=True, merge_commit=None, merge_worktree=fake_wt_a)

        with (
            patch.object(git_ops, 'merge_to_main', AsyncMock(return_value=a_result)),
            patch.object(git_ops, 'cleanup_merge_worktree', new=tracking_cleanup),
        ):
            req_a = _make_request('pme-a', 'pme-a', wt_a, config)
            await queue.put(req_a)
            outcome_a = await asyncio.wait_for(req_a.result, timeout=30)

        assert outcome_a.status == 'blocked', f'Scenario A: expected blocked, got {outcome_a}'
        assert 'Merger error' in outcome_a.reason, (
            f'Scenario A: expected "Merger error" in reason, got: {outcome_a.reason!r}'
        )
        # The merge worktree must have been cleaned up despite the exception
        assert fake_wt_a in cleanup_calls, (
            f'Scenario A: cleanup_merge_worktree not called for fake_wt_a; '
            f'cleanup_calls={cleanup_calls}'
        )

        # ── Scenario B: RuntimeError from verifier-queue put ───────────────────
        # A valid merge_commit passes the assert; the put raises instead.
        b_merge_commit = 'ab' * 20  # 40-char fake SHA
        b_result = MergeResult(
            success=True, merge_commit=b_merge_commit, merge_worktree=fake_wt_b,
        )

        original_put = worker._verifier_queue.put
        b_put_count = 0

        async def sometimes_failing_put(item: SpeculativeItem | None) -> None:
            nonlocal b_put_count
            b_put_count += 1
            if b_put_count == 1 and isinstance(item, SpeculativeItem):
                raise RuntimeError('queue broken')
            await original_put(item)

        with (
            patch.object(git_ops, 'merge_to_main', AsyncMock(return_value=b_result)),
            patch.object(git_ops, 'cleanup_merge_worktree', new=tracking_cleanup),
            patch.object(worker._verifier_queue, 'put', new=sometimes_failing_put),
        ):
            req_b = _make_request('pme-b', 'pme-b', wt_b, config)
            await queue.put(req_b)
            outcome_b = await asyncio.wait_for(req_b.result, timeout=30)

        assert outcome_b.status == 'blocked', f'Scenario B: expected blocked, got {outcome_b}'
        assert 'Merger error' in outcome_b.reason, (
            f'Scenario B: expected "Merger error" in reason, got: {outcome_b.reason!r}'
        )
        assert fake_wt_b in cleanup_calls, (
            f'Scenario B: cleanup_merge_worktree not called for fake_wt_b; '
            f'cleanup_calls={cleanup_calls}'
        )

        # ── Merger loop continues after both exceptions ──────────────────────
        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req_ok = _make_request('pme-ok', 'pme-ok', wt_ok, config)
            await queue.put(req_ok)
            outcome_ok = await asyncio.wait_for(req_ok.result, timeout=30)

        assert outcome_ok.status == 'done', (
            f'Merger loop should continue after exceptions, got {outcome_ok}'
        )

        await worker.stop()
        await worker_task

    async def test_speculative_merge_worker_emits_duration_ms_on_done(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """SpeculativeMergeWorker emits duration_ms on the 'done' outcome.

        Exercises the verifier-phase emit at _verify_and_advance (done path).
        Asserts the merge_attempt event row has a non-null integer duration_ms.
        """
        db_path = tmp_path / 'events_spec_done.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        wt = await _make_branch_with_file(
            git_ops, 'sdur-done', 'sdur_done.py', 'sdur = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=event_store)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req = _make_request('sdur-done', 'sdur-done', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done', f'Expected done, got: {outcome}'

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT json_extract(data, '$.outcome') AS outcome, duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()

        done_rows = [r for r in rows if r[0] == 'done']
        assert len(done_rows) == 1, f'Expected 1 done row, got: {rows}'
        assert done_rows[0][1] is not None, 'duration_ms should not be NULL'
        assert isinstance(done_rows[0][1], int), f'duration_ms should be int, got {type(done_rows[0][1])}'
        assert done_rows[0][1] >= 0, f'duration_ms should be >= 0, got {done_rows[0][1]}'

        await worker.stop()
        await worker_task

    async def test_speculative_merger_phase_emits_duration_ms(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """Merger-phase emit sites set non-null duration_ms.

        Covers already_merged and conflict outcomes emitted from _merger_loop.
        """
        # --- Scenario A: already_merged (merger phase) ---
        db_a = tmp_path / 'events_sphase_a.db'
        es_a = EventStore(db_path=db_a, run_id='run-a')

        wt_n = await _make_branch_with_file(
            git_ops, 'sphase-n', 'sphase_n.py', 'sphase_n = 1\n',
        )
        # Create N+1 as already-merged: merge it first, then submit it
        wt_n1 = (await git_ops.create_worktree('sphase-n1')).path
        (wt_n1 / 'sphase_n1.py').write_text('sphase_n1 = 2\n')
        await git_ops.commit(wt_n1, 'Add sphase_n1.py')
        r = await git_ops.merge_to_main(wt_n1, 'sphase-n1')
        assert r.success
        assert r.merge_commit is not None
        await git_ops.advance_main(r.merge_commit)
        if r.merge_worktree:
            await git_ops.cleanup_merge_worktree(r.merge_worktree)

        q_a: asyncio.Queue[MergeRequest] = asyncio.Queue()
        w_a = SpeculativeMergeWorker(git_ops, q_a, event_store=es_a)
        wt_a = asyncio.create_task(w_a.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req_n = _make_request('sphase-n', 'sphase-n', wt_n, config)
            req_n1 = _make_request('sphase-n1', 'sphase-n1', wt_n1, config)
            await q_a.put(req_n)
            await q_a.put(req_n1)
            out_n = await asyncio.wait_for(req_n.result, timeout=30)
            out_n1 = await asyncio.wait_for(req_n1.result, timeout=30)

        assert out_n.status == 'done', f'N: {out_n}'
        assert out_n1.status == 'already_merged', f'N+1: {out_n1}'
        await w_a.stop()
        await wt_a

        conn = sqlite3.connect(str(db_a))
        rows_a = conn.execute(
            "SELECT json_extract(data, '$.outcome'), duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()
        assert all(r[1] is not None for r in rows_a), f'NULL duration_ms in already_merged scenario: {rows_a}'

        # --- Scenario B: conflict (merger phase) ---
        db_b = tmp_path / 'events_sphase_b.db'
        es_b = EventStore(db_path=db_b, run_id='run-b')

        wt_n2 = await _make_branch_with_file(
            git_ops, 'sphase-n2', 'sphase_n2.py', 'sphase_n2 = 1\n',
        )
        # Create a conflicting N+2 (README conflict)
        wt_cfl = (await git_ops.create_worktree('sphase-cfl')).path
        (git_ops.project_root / 'README.md').write_text('# conflict-src-sphase\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Conflict source sphase'], cwd=git_ops.project_root)
        (wt_cfl / 'README.md').write_text('# conflict-task-sphase\n')
        await git_ops.commit(wt_cfl, 'Conflict task sphase')

        q_b: asyncio.Queue[MergeRequest] = asyncio.Queue()
        w_b = SpeculativeMergeWorker(git_ops, q_b, event_store=es_b)
        wt_b = asyncio.create_task(w_b.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req_n2 = _make_request('sphase-n2', 'sphase-n2', wt_n2, config)
            req_cfl = _make_request('sphase-cfl', 'sphase-cfl', wt_cfl, config)
            await q_b.put(req_n2)
            await q_b.put(req_cfl)
            out_n2 = await asyncio.wait_for(req_n2.result, timeout=30)
            out_cfl = await asyncio.wait_for(req_cfl.result, timeout=30)

        assert out_n2.status == 'done', f'N2: {out_n2}'
        assert out_cfl.status == 'conflict', f'Conflict: {out_cfl}'
        await w_b.stop()
        await wt_b

        conn = sqlite3.connect(str(db_b))
        rows_b = conn.execute(
            "SELECT json_extract(data, '$.outcome'), duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()
        assert all(r[1] is not None for r in rows_b), f'NULL duration_ms in conflict scenario: {rows_b}'

    async def test_speculative_remerge_preserves_duration_ms(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """_remerge uses the original started_monotonic so duration_ms is realistic.

        The _remerge path is triggered when N fails verification and N+1 was
        speculatively merged. The verifier discards N+1's stale worktree and
        calls _remerge. If started_monotonic is correctly threaded through,
        the conflict emit inside _remerge yields a realistic duration (< 60s).
        If it falls back to the 0.0 default, duration would be huge (seconds
        since process start × 1000).

        Setup: N succeeds merge but fails verification, so N+1 is discarded
        and re-merged. We patch merge_to_main so the re-merge produces a
        conflict for N+1, causing a conflict emit inside _remerge.
        """
        db_path = tmp_path / 'events_remerge.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        wt_n = await _make_branch_with_file(
            git_ops, 'rmp-n', 'rmp_n.py', 'rmp_n = 1\n',
        )
        wt_n1 = await _make_branch_with_file(
            git_ops, 'rmp-n1', 'rmp_n1.py', 'rmp_n1 = 2\n',
        )

        # Track merge_to_main calls so we can return conflict on the re-merge
        original_merge = git_ops.merge_to_main
        merge_call_count = 0

        async def _controlled_merge(worktree, branch, **kwargs):
            nonlocal merge_call_count
            merge_call_count += 1
            # First two calls are speculative merges for N and N+1 (normal)
            # Third call is the re-merge for N+1 after N fails — return conflict
            if merge_call_count >= 3 and branch == 'rmp-n1':
                return MergeResult(
                    success=False,
                    conflicts=True,
                    details='simulated remerge conflict',
                    merge_commit=None,
                    merge_worktree=None,
                    pre_merge_sha=None,
                )
            return await original_merge(worktree, branch, **kwargs)

        async def _fail_n_pass_n1(merge_wt, cfg, module_configs, task_files=None):
            """Fail N's verification; N+1 re-merge won't reach verify (conflicts)."""
            n_present = (merge_wt / 'rmp_n.py').exists()
            if n_present:
                return MagicMock(passed=False, summary='N failed intentionally')
            return MagicMock(passed=True, summary='')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=event_store)
        worker_task = asyncio.create_task(worker.run())

        with (
            patch.object(git_ops, 'merge_to_main', side_effect=_controlled_merge),
            patch('orchestrator.merge_queue.run_scoped_verification', side_effect=_fail_n_pass_n1),
        ):
            req_n = _make_request('rmp-n', 'rmp-n', wt_n, config)
            req_n1 = _make_request('rmp-n1', 'rmp-n1', wt_n1, config)
            await queue.put(req_n)
            await queue.put(req_n1)
            out_n = await asyncio.wait_for(req_n.result, timeout=30)
            out_n1 = await asyncio.wait_for(req_n1.result, timeout=30)

        # N should be blocked (verify failed); N+1 should be conflict (from _remerge)
        assert out_n.status == 'blocked', f'N: {out_n}'
        assert out_n1.status == 'conflict', f'N+1: {out_n1}'

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT json_extract(data, '$.outcome'), duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()

        # All merge_attempt events must have non-null duration_ms
        assert all(r[1] is not None for r in rows), f'NULL duration_ms found: {rows}'

        # The conflict emit inside _remerge must have a realistic duration_ms
        # (0 to 60000 ms). If started_monotonic were 0.0 (default), the value
        # would be time-since-process-start * 1000 — many thousands of ms.
        conflict_rows = [r for r in rows if r[0] == 'conflict']
        assert len(conflict_rows) >= 1, f'Expected at least one conflict event: {rows}'
        for _outcome, dur in conflict_rows:
            assert 0 <= dur <= 60_000, (
                f'duration_ms={dur} is not realistic; '
                f'started_monotonic was likely not threaded through _remerge'
            )

        await worker.stop()
        await worker_task

    async def test_speculative_merge_worker_success_returns_merge_sha(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """SpeculativeMergeWorker success path: MergeOutcome.merge_sha is set.

        Submits one request, drives through the verify-and-advance path, and
        asserts the resulting MergeOutcome has status='done' with a 40-char
        merge commit SHA.  Fails initially because line ~1130 still constructs
        MergeOutcome('done') without merge_sha (step-5 guard; impl in step-6).
        """
        wt_n = await _make_branch_with_file(
            git_ops, 'sspec-n', 'file_sspec_n.py', 'sspec_n = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            _mock_verify_pass(),
        ):
            req_n = _make_request('sspec-task', 'sspec-n', wt_n, config)
            await queue.put(req_n)
            outcome_n = await asyncio.wait_for(req_n.result, timeout=60)

        await worker.stop()
        await worker_task

        assert outcome_n.status == 'done', f'Expected done, got: {outcome_n}'
        assert outcome_n.merge_sha is not None, 'merge_sha must be set on done outcome'
        assert len(outcome_n.merge_sha) == 40, f'Expected 40-char SHA, got: {outcome_n.merge_sha!r}'
        assert all(c in '0123456789abcdef' for c in outcome_n.merge_sha), (
            f'merge_sha is not a hex string: {outcome_n.merge_sha!r}'
        )

# ---------------------------------------------------------------------------
# TestMergeOutcomeDataclass — unit tests for MergeOutcome dataclass fields
# ---------------------------------------------------------------------------


class TestMergeOutcomeDataclass:
    def test_merge_outcome_has_merge_sha_field_default_none(self):
        """MergeOutcome.merge_sha defaults to None and can be set.

        Verifies that the field exists (step-1 / step-2 guard), that constructing
        MergeOutcome without the kwarg gives None, and that the field stores the
        value when supplied.
        """
        outcome_no_sha = MergeOutcome('done')
        assert outcome_no_sha.merge_sha is None  # type: ignore[attr-defined]

        outcome_with_sha = MergeOutcome('done', merge_sha='abc123')  # type: ignore[call-arg]
        assert outcome_with_sha.merge_sha == 'abc123'  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# TestSpeculativeItemDefaults — unit tests for SpeculativeItem field defaults
# ---------------------------------------------------------------------------


class TestSpeculativeItemDefaults:
    def test_started_monotonic_default_is_none(self):
        """SpeculativeItem.started_monotonic defaults to None when not passed.

        Ensures construction sites that omit started_monotonic produce NULL
        duration_ms (via _elapsed_ms) rather than a bogus time-since-process-start
        value derived from the 0.0 sentinel.

        Uses a MagicMock for request to avoid the asyncio.Future that
        MergeRequest.result requires (we only care about the dataclass default).
        """
        from unittest.mock import MagicMock

        from orchestrator.merge_queue import _elapsed_ms
        item = SpeculativeItem(
            request=MagicMock(),
            merge_result=None,
            merge_wt=None,
            base_sha='',
            speculative=False,
            skip_verify=False,
        )
        assert item.started_monotonic is None
        # Tie the default to the observability guarantee: None → NULL duration_ms
        assert _elapsed_ms(item.started_monotonic) is None


# ---------------------------------------------------------------------------
# TestEmitMergeAttemptHelper — unit tests for module-level _emit_merge_attempt
# ---------------------------------------------------------------------------


class TestEmitMergeAttemptHelper:
    def test_emit_merge_attempt_writes_row_without_attempt(
        self, tmp_path: Path,
    ):
        """Call with outcome and duration_ms — row has no 'attempt' key."""
        from orchestrator.merge_queue import _emit_merge_attempt

        db_path = tmp_path / 'eh_a.db'
        es = EventStore(db_path=db_path, run_id='eh-run')

        _emit_merge_attempt(es, 'task-1', 'conflict', duration_ms=42)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT json_extract(data, '$.outcome'), "
            "       json_extract(data, '$.attempt'), "
            "       duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()

        assert len(rows) == 1, f'Expected 1 row, got: {rows}'
        outcome, attempt, dur = rows[0]
        assert outcome == 'conflict'
        assert attempt is None, f'Expected no attempt key, got {attempt!r}'
        assert dur == 42

    def test_emit_merge_attempt_writes_row_with_attempt(
        self, tmp_path: Path,
    ):
        """Call with outcome, attempt, and duration_ms — row includes 'attempt'."""
        from orchestrator.merge_queue import _emit_merge_attempt

        db_path = tmp_path / 'eh_b.db'
        es = EventStore(db_path=db_path, run_id='eh-run')

        _emit_merge_attempt(es, 'task-2', 'cas_retry', attempt=3, duration_ms=500)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT json_extract(data, '$.outcome'), "
            "       json_extract(data, '$.attempt'), "
            "       duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()

        assert len(rows) == 1, f'Expected 1 row, got: {rows}'
        outcome, attempt, dur = rows[0]
        assert outcome == 'cas_retry'
        assert attempt == 3
        assert dur == 500

    def test_emit_merge_attempt_null_duration_when_none(
        self, tmp_path: Path,
    ):
        """Call with duration_ms=None — duration_ms column is NULL."""
        from orchestrator.merge_queue import _emit_merge_attempt

        db_path = tmp_path / 'eh_c.db'
        es = EventStore(db_path=db_path, run_id='eh-run')

        _emit_merge_attempt(es, 'task-3', 'done', duration_ms=None)

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT json_extract(data, '$.outcome'), duration_ms "
            "FROM events WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()

        assert len(rows) == 1, f'Expected 1 row, got: {rows}'
        outcome, dur = rows[0]
        assert outcome == 'done'
        assert dur is None, f'Expected NULL duration_ms, got {dur!r}'

    def test_emit_merge_attempt_calls_emit_when_store_provided(self):
        """Call with a real (mock) store — emit is invoked exactly once."""
        from unittest.mock import MagicMock

        from orchestrator.merge_queue import _emit_merge_attempt

        mock_es = MagicMock()
        _emit_merge_attempt(mock_es, 'task-check', 'done', duration_ms=1)
        mock_es.emit.assert_called_once()

    def test_emit_merge_attempt_noop_when_event_store_is_none(self):
        """Call with event_store=None — no exception, emit never invoked."""
        from unittest.mock import MagicMock

        from orchestrator.merge_queue import _emit_merge_attempt

        mock_es = MagicMock()
        _emit_merge_attempt(None, 'task-4', 'done', duration_ms=1)
        mock_es.emit.assert_not_called()


# ---------------------------------------------------------------------------
# TestSpeculativeBackwardCompat — step-17
# Run key MergeWorker scenarios through SpeculativeMergeWorker to confirm
# they behave identically with queue depth 1.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSpeculativeBackwardCompat:
    async def test_basic_merge(self, git_ops: GitOps, config: OrchestratorConfig):
        worktree = (await git_ops.create_worktree('compat-basic')).path
        (worktree / 'compat.py').write_text('compat = True\n')
        await git_ops.commit(worktree, 'Add compat file')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req = _make_request('compat-1', 'compat-basic', worktree, config)
            await queue.put(req)
            result = await asyncio.wait_for(req.result, timeout=30)

        assert result.status == 'done'
        _, content, _ = await _run(
            ['git', 'show', 'main:compat.py'], cwd=git_ops.project_root,
        )
        assert 'compat = True' in content

        await worker.stop()
        await worker_task

    async def test_already_merged(self, git_ops: GitOps, config: OrchestratorConfig):
        worktree = (await git_ops.create_worktree('compat-am')).path
        (worktree / 'am.py').write_text('am = True\n')
        await git_ops.commit(worktree, 'Add am file')

        result = await git_ops.merge_to_main(worktree, 'compat-am')
        assert result.success
        assert result.merge_commit is not None
        await git_ops.advance_main(result.merge_commit)
        if result.merge_worktree:
            await git_ops.cleanup_merge_worktree(result.merge_worktree)

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        req = _make_request('compat-am', 'compat-am', worktree, config)
        await queue.put(req)
        outcome = await asyncio.wait_for(req.result, timeout=10)
        assert outcome.status == 'already_merged'

        await worker.stop()
        await worker_task

    async def test_verify_failure(self, git_ops: GitOps, config: OrchestratorConfig):
        worktree = (await git_ops.create_worktree('compat-vf')).path
        (worktree / 'bad.py').write_text('bad = True\n')
        await git_ops.commit(worktree, 'Add bad file')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        mock_verify = AsyncMock(return_value=MagicMock(passed=False, summary='tests failed'))

        with patch('orchestrator.merge_queue.run_scoped_verification', mock_verify):
            req = _make_request('compat-vf', 'compat-vf', worktree, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked'
        assert 'verification failed' in outcome.reason.lower()

        await worker.stop()
        await worker_task


# ---------------------------------------------------------------------------
# TestMergeVerifyColdTimeout — Fix #1
# Merge worktrees are freshly created per merge (no warm cargo cache) but
# lack .task/ (only .taskmaster/), so _is_verify_cold mis-classifies them as
# warm.  The merge-queue call sites must pass is_merge_verify=True so the
# cold-track timeout applies.  These tests assert the kwarg is threaded
# through both MergeWorker and SpeculativeMergeWorker.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeVerifyColdTimeout:
    async def test_merge_worker_passes_is_merge_verify_true(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """MergeWorker's verify call must set is_merge_verify=True.

        The legacy serial worker is preserved for compat; even though
        SpeculativeMergeWorker is the default production path, this flag
        must still flow through here so tests/eval/debug harnesses that
        opt back into the serial worker also get the cold timeout.
        """
        worktree = (await git_ops.create_worktree('merge-cold-mw')).path
        (worktree / 'coldmw.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add coldmw')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        captured_kwargs: list[dict] = []

        async def spy_verify(*args, **kwargs):
            captured_kwargs.append(kwargs)
            result = AsyncMock()
            result.passed = True
            result.summary = ''
            result.timed_out = False
            return result

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=spy_verify,
        ):
            req = _make_request('cold-mw', 'merge-cold-mw', worktree, config)
            await queue.put(req)
            await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        await worker_task

        assert captured_kwargs, 'run_scoped_verification was not invoked'
        assert captured_kwargs[0].get('is_merge_verify') is True, (
            f'merge-queue verify must pass is_merge_verify=True; '
            f'got {captured_kwargs[0]!r}'
        )

    async def test_speculative_worker_passes_is_merge_verify_true(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """SpeculativeMergeWorker's verify call must set is_merge_verify=True.

        This is the production path — it's the call site that was
        mis-classifying merge worktrees as warm and blowing the 30-min
        timeout on each post-merge verify against reify.
        """
        wt = await _make_branch_with_file(
            git_ops, 'merge-cold-spec', 'coldspec.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        captured_kwargs: list[dict] = []

        async def spy_verify(*args, **kwargs):
            captured_kwargs.append(kwargs)
            result = AsyncMock()
            result.passed = True
            result.summary = ''
            result.timed_out = False
            return result

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=spy_verify,
        ):
            req = _make_request('cold-spec', 'merge-cold-spec', wt, config)
            await queue.put(req)
            await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        await worker_task

        assert captured_kwargs, 'run_scoped_verification was not invoked'
        assert captured_kwargs[0].get('is_merge_verify') is True, (
            f'speculative merge-queue verify must pass is_merge_verify=True; '
            f'got {captured_kwargs[0]!r}'
        )


# ---------------------------------------------------------------------------
# TestMergeVerifyTimeoutLoopBreaker — Fix #2
# After MAX_POST_MERGE_VERIFY_TIMEOUTS consecutive post-merge verify
# TIMEOUTS for the same task, the merge queue must stop running merge+verify
# and return a ``blocked`` outcome with the ABANDONED_REASON_PREFIX.  Real
# (non-timeout) verify failures must NOT feed the counter, and a successful
# merge must reset the counter.
# ---------------------------------------------------------------------------


def _mock_verify_timeout():
    """Return a mock that makes run_scoped_verification time out."""
    async def _fake(*args, **kwargs):
        result = AsyncMock()
        result.passed = False
        result.summary = 'Verification timed out'
        result.timed_out = True
        result.failure_report = lambda: '## Verify Timed Out\n\n(mock)'
        return result
    return _fake


@pytest.mark.asyncio
class TestMergeVerifyTimeoutLoopBreaker:
    async def test_merge_worker_abandons_after_threshold(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """N consecutive verify timeouts → next submission blocked without verify.

        Submits the same task_id MAX+1 times.  The first MAX submissions
        run merge+verify and surface a blocked/timeout outcome.  The next
        submission must short-circuit: no merge, no verify, blocked
        outcome with ABANDONED_REASON_PREFIX.
        """
        from orchestrator.merge_queue import ABANDONED_REASON_PREFIX

        wt = await _make_branch_with_file(
            git_ops, 'loop-break-mw', 'lb.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        assert worker.MAX_POST_MERGE_VERIFY_TIMEOUTS == 2
        worker_task = asyncio.create_task(worker.run())

        verify_call_count = 0

        async def counting_timeout_verify(*args, **kwargs):
            nonlocal verify_call_count
            verify_call_count += 1
            result = AsyncMock()
            result.passed = False
            result.summary = 'Verification timed out'
            result.timed_out = True
            result.failure_report = lambda: ''
            return result

        try:
            with patch(
                'orchestrator.merge_queue.run_scoped_verification',
                side_effect=counting_timeout_verify,
            ):
                # Submissions 1..MAX: run merge+verify, surface timeout.
                outcomes: list[MergeOutcome] = []
                for _ in range(worker.MAX_POST_MERGE_VERIFY_TIMEOUTS):
                    req = _make_request('lb-task', 'loop-break-mw', wt, config)
                    await queue.put(req)
                    outcomes.append(await asyncio.wait_for(req.result, timeout=30))

                # Every one of those must be blocked with the verify-failed reason.
                for o in outcomes:
                    assert o.status == 'blocked'
                    assert 'verification failed' in o.reason.lower()

                verify_calls_before_loopbreak = verify_call_count
                assert verify_calls_before_loopbreak == worker.MAX_POST_MERGE_VERIFY_TIMEOUTS

                # Submission MAX+1: must short-circuit BEFORE invoking verify.
                req_final = _make_request(
                    'lb-task', 'loop-break-mw', wt, config,
                )
                await queue.put(req_final)
                final = await asyncio.wait_for(req_final.result, timeout=10)

            assert final.status == 'blocked'
            assert final.reason.startswith(ABANDONED_REASON_PREFIX), (
                f'Expected abandoned reason prefix; got {final.reason!r}'
            )
            # Crucially: verify was NOT invoked again on the abandoned path.
            assert verify_call_count == verify_calls_before_loopbreak, (
                f'Abandoned submission must not invoke verify; '
                f'before={verify_calls_before_loopbreak} after={verify_call_count}'
            )
        finally:
            await worker.stop()
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task

    async def test_speculative_worker_abandons_after_threshold(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """SpeculativeMergeWorker loop-breaker — same contract as MergeWorker."""
        from orchestrator.merge_queue import ABANDONED_REASON_PREFIX

        wt = await _make_branch_with_file(
            git_ops, 'loop-break-spec', 'lbs.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        assert worker.MAX_POST_MERGE_VERIFY_TIMEOUTS == 2
        worker_task = asyncio.create_task(worker.run())

        verify_call_count = 0

        async def counting_timeout_verify(*args, **kwargs):
            nonlocal verify_call_count
            verify_call_count += 1
            result = AsyncMock()
            result.passed = False
            result.summary = 'Verification timed out'
            result.timed_out = True
            result.failure_report = lambda: ''
            return result

        try:
            with patch(
                'orchestrator.merge_queue.run_scoped_verification',
                side_effect=counting_timeout_verify,
            ):
                for _ in range(worker.MAX_POST_MERGE_VERIFY_TIMEOUTS):
                    req = _make_request('lbs-task', 'loop-break-spec', wt, config)
                    await queue.put(req)
                    outcome = await asyncio.wait_for(req.result, timeout=30)
                    assert outcome.status == 'blocked'

                verify_calls_before_loopbreak = verify_call_count
                assert verify_calls_before_loopbreak == worker.MAX_POST_MERGE_VERIFY_TIMEOUTS

                req_final = _make_request(
                    'lbs-task', 'loop-break-spec', wt, config,
                )
                await queue.put(req_final)
                final = await asyncio.wait_for(req_final.result, timeout=10)

            assert final.status == 'blocked'
            assert final.reason.startswith(ABANDONED_REASON_PREFIX)
            assert verify_call_count == verify_calls_before_loopbreak
        finally:
            await worker.stop()
            await worker_task

    async def test_non_timeout_failure_does_not_feed_counter(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Real (non-timeout) verify failures must NOT count toward the budget.

        Submits the same task twice with a real test-failure result and
        then a third time — the third submission must still run merge+verify
        (not abandon), because the counter only advances on timed_out=True.
        """
        wt = await _make_branch_with_file(
            git_ops, 'loop-real-fail', 'lrf.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        call_count = 0

        async def real_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            result = AsyncMock()
            result.passed = False
            result.summary = 'tests failed'
            result.timed_out = False
            result.failure_report = lambda: ''
            return result

        try:
            with patch(
                'orchestrator.merge_queue.run_scoped_verification',
                side_effect=real_failure,
            ):
                # Submit 3 times — more than the abandon threshold (2).
                # Real failures must not abandon; all 3 must run verify.
                for _ in range(worker.MAX_POST_MERGE_VERIFY_TIMEOUTS + 1):
                    req = _make_request('rf-task', 'loop-real-fail', wt, config)
                    await queue.put(req)
                    outcome = await asyncio.wait_for(req.result, timeout=30)
                    assert outcome.status == 'blocked'
                    assert 'verification failed' in outcome.reason.lower()
                    # Must NOT be the abandoned-reason prefix.
                    assert not outcome.reason.startswith(
                        'Post-merge verify timed out'
                    ), (
                        f'Real failure must not produce abandoned reason; '
                        f'got {outcome.reason!r}'
                    )

            assert call_count == worker.MAX_POST_MERGE_VERIFY_TIMEOUTS + 1, (
                f'Every submission must run verify when failures are real; '
                f'got {call_count} verify calls'
            )
            # Counter must be zero (real failures never bumped it).
            assert worker._post_merge_verify_timeouts.get('rf-task', 0) == 0
        finally:
            await worker.stop()
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task

    async def test_success_resets_counter(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """A successful merge clears the counter so future timeouts start fresh.

        Injects 1 timeout (under the threshold of 2), then a success, and
        asserts the counter is reset to zero.  A separate assert on the
        dict keeps this test decoupled from the ``_abandon_outcome`` path.
        """
        # First task: time out once.
        wt_fail = await _make_branch_with_file(
            git_ops, 'reset-fail', 'rf.py', 'x = 1\n',
        )
        # Second task: same task_id, but arrange verify to pass.
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        verify_pass_once_after_timeout = {'first_call': True}

        async def timeout_then_pass(*args, **kwargs):
            result = AsyncMock()
            if verify_pass_once_after_timeout['first_call']:
                verify_pass_once_after_timeout['first_call'] = False
                result.passed = False
                result.summary = 'Verification timed out'
                result.timed_out = True
                result.failure_report = lambda: ''
            else:
                result.passed = True
                result.summary = ''
                result.timed_out = False
            return result

        try:
            with patch(
                'orchestrator.merge_queue.run_scoped_verification',
                side_effect=timeout_then_pass,
            ):
                # First submission → verify times out → counter = 1.
                req1 = _make_request('reset-task', 'reset-fail', wt_fail, config)
                await queue.put(req1)
                r1 = await asyncio.wait_for(req1.result, timeout=30)
                assert r1.status == 'blocked'
                assert worker._post_merge_verify_timeouts.get('reset-task') == 1

                # Second submission for a *different* branch that merges cleanly,
                # same task_id → verify passes → counter cleared.
                wt_ok = await _make_branch_with_file(
                    git_ops, 'reset-ok', 'ro.py', 'x = 2\n',
                )
                req2 = _make_request('reset-task', 'reset-ok', wt_ok, config)
                await queue.put(req2)
                r2 = await asyncio.wait_for(req2.result, timeout=30)
                assert r2.status == 'done'

            # Counter must have been cleared by the successful merge.
            assert 'reset-task' not in worker._post_merge_verify_timeouts
        finally:
            await worker.stop()
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task


# ---------------------------------------------------------------------------
# TestWipHalt — WIP-safe merge queue halt mechanism
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWipHaltMergeWorker:
    async def test_wip_halted_blocks_subsequent_tasks(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """wip_overlap halts queue; second request stays pending until unhalt."""
        wt1 = await _make_branch_with_file(
            git_ops, 'halt-1', 'file_halt_1.py', 'halt1 = 1\n',
        )
        wt2 = await _make_branch_with_file(
            git_ops, 'halt-2', 'file_halt_2.py', 'halt2 = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # advance_main returns wip_overlap for first request
        call_count = 0
        original_advance = git_ops.advance_main

        async def _wip_overlap_then_normal(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                git_ops._last_overlap_files = ['file_halt_1.py']
                return 'wip_overlap'
            return await original_advance(*args, **kwargs)

        with (
            patch.object(git_ops, 'advance_main', side_effect=_wip_overlap_then_normal),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req1 = _make_request('halt-1', 'halt-1', wt1, config)
            await queue.put(req1)
            outcome1 = await asyncio.wait_for(req1.result, timeout=30)

        assert outcome1.status == 'wip_halted'
        assert outcome1.overlap_files == ['file_halt_1.py']
        assert worker.is_wip_halted

        # Second request: put it in queue, it should NOT resolve while halted
        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req2 = _make_request('halt-2', 'halt-2', wt2, config)
            await queue.put(req2)

            # Give the worker a chance to process (it shouldn't, it's halted)
            await asyncio.sleep(0.2)
            assert not req2.result.done(), 'Second request resolved while queue was halted'

            # Un-halt the queue
            worker.unhalt_wip()
            assert not worker.is_wip_halted

            # Now the second request should resolve
            outcome2 = await asyncio.wait_for(req2.result, timeout=30)

        assert outcome2.status == 'done'

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_done_wip_recovery_outcome(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """pop_conflict returns done_wip_recovery with recovery branch info."""
        wt = await _make_branch_with_file(
            git_ops, 'recov-1', 'file_recov.py', 'recov = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        async def _pop_conflict(*args, **kwargs):
            git_ops._last_recovery_branch = 'wip/recovery-recov-1-20260407T120000'
            return 'pop_conflict'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_pop_conflict),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('recov-1', 'recov-1', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done_wip_recovery'
        assert outcome.recovery_branch == 'wip/recovery-recov-1-20260407T120000'
        assert worker.is_wip_halted

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_done_wip_recovery_propagates_advanced_sha(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """pop_conflict outcome carries the post-rebase on-main SHA via merge_sha.

        Without this, workflow._handle_wip_recovery leaves self._merge_sha=None
        and the success-path set_task_status('done', done_provenance=None) fails
        fused-memory's "kind required" validation, leaving the task stuck
        in-progress despite the merge having landed.
        """
        wt = await _make_branch_with_file(
            git_ops, 'recov-sha', 'file_recov_sha.py', 'recov_sha = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # Simulate advance_main: main IS advanced (records _last_advanced_sha)
        # but stash pop conflicted (returns 'pop_conflict').
        async def _pop_conflict(*args, **kwargs):
            git_ops._last_recovery_branch = 'wip/recovery-recov-sha-20260428T000000'
            git_ops._last_advanced_sha = 'feedface' * 5  # 40-char fake on-main SHA
            return 'pop_conflict'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_pop_conflict),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('recov-sha', 'recov-sha', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done_wip_recovery'
        assert outcome.merge_sha == 'feedface' * 5

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_unmerged_state_returns_unmerged_state_and_halts(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """unmerged_state: MergeWorker returns 'unmerged_state' status and halts the queue."""
        wt = await _make_branch_with_file(
            git_ops, 'uu-mw-1', 'file_uu_mw.py', 'uu_mw = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        async def _unmerged_state(*args: Any, **kwargs: Any):
            return 'unmerged_state'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_unmerged_state),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('uu-mw-1', 'uu-mw-1', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'unmerged_state'
        assert 'unmerged' in outcome.reason.lower()
        assert worker.is_wip_halted

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

    async def test_pop_conflict_no_advance_returns_wip_recovery_no_advance_and_halts(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """pop_conflict_no_advance: MergeWorker returns wip_recovery_no_advance and halts."""
        wt = await _make_branch_with_file(
            git_ops, 'pcna-mw-1', 'file_pcna_mw.py', 'pcna_mw = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        async def _pop_conflict_no_advance(*args: Any, **kwargs: Any):
            git_ops._last_recovery_branch = 'wip/recovery-x-y'
            return 'pop_conflict_no_advance'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_pop_conflict_no_advance),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('pcna-mw-1', 'pcna-mw-1', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'wip_recovery_no_advance'
        assert outcome.recovery_branch == 'wip/recovery-x-y'
        assert 'did not advance' in outcome.reason.lower()
        assert worker.is_wip_halted

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task


@pytest.mark.asyncio
class TestWipHaltSpeculativeMergeWorker:
    async def test_wip_halted_blocks_subsequent_tasks(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """wip_overlap in speculative worker halts queue; unhalt resumes.

        Submit req1 alone (no speculative look-ahead) so the merger loop
        reaches _wip_halt.wait() before req2 enters the queue.
        """
        wt1 = await _make_branch_with_file(
            git_ops, 'shalt-1', 'file_shalt_1.py', 'shalt1 = 1\n',
        )
        wt2 = await _make_branch_with_file(
            git_ops, 'shalt-2', 'file_shalt_2.py', 'shalt2 = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        call_count = 0
        original_advance = git_ops.advance_main

        async def _wip_overlap_then_normal(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                git_ops._last_overlap_files = ['file_shalt_1.py']
                return 'wip_overlap'
            return await original_advance(*args, **kwargs)

        with (
            patch.object(git_ops, 'advance_main', side_effect=_wip_overlap_then_normal),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            # Submit req1 alone — no req2 in queue, so no speculative look-ahead
            req1 = _make_request('shalt-1', 'shalt-1', wt1, config)
            await queue.put(req1)
            outcome1 = await asyncio.wait_for(req1.result, timeout=30)

            assert outcome1.status == 'wip_halted'
            assert outcome1.overlap_files == ['file_shalt_1.py']
            assert worker.is_wip_halted

            # Now submit req2 — merger is blocked at _wip_halt.wait()
            req2 = _make_request('shalt-2', 'shalt-2', wt2, config)
            await queue.put(req2)
            await asyncio.sleep(0.3)
            assert not req2.result.done(), 'Second request resolved while queue was halted'

            # Un-halt and wait for req2
            worker.unhalt_wip()
            outcome2 = await asyncio.wait_for(req2.result, timeout=30)

        assert outcome2.status == 'done'

        await worker.stop()
        await worker_task

    async def test_done_wip_recovery_outcome(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """pop_conflict in speculative worker returns done_wip_recovery and halts."""
        wt = await _make_branch_with_file(
            git_ops, 'srecov-1', 'file_srecov.py', 'srecov = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        async def _pop_conflict(*args, **kwargs):
            git_ops._last_recovery_branch = 'wip/recovery-srecov-1-20260407T120000'
            return 'pop_conflict'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_pop_conflict),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('srecov-1', 'srecov-1', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done_wip_recovery'
        assert outcome.recovery_branch == 'wip/recovery-srecov-1-20260407T120000'
        assert worker.is_wip_halted

        await worker.stop()
        await worker_task

    async def test_speculative_done_wip_recovery_propagates_advanced_sha(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Speculative worker pop_conflict path also propagates merge_sha.

        Sister test to MergeWorker's test_done_wip_recovery_propagates_advanced_sha.
        """
        wt = await _make_branch_with_file(
            git_ops, 'srecov-sha', 'file_srecov_sha.py', 'srecov_sha = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        async def _pop_conflict(*args, **kwargs):
            git_ops._last_recovery_branch = 'wip/recovery-srecov-sha-20260428T000000'
            git_ops._last_advanced_sha = 'cafebabe' * 5
            return 'pop_conflict'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_pop_conflict),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('srecov-sha', 'srecov-sha', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done_wip_recovery'
        assert outcome.merge_sha == 'cafebabe' * 5

        await worker.stop()
        await worker_task

    async def test_speculative_unmerged_state_returns_unmerged_state_and_halts(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """unmerged_state in SpeculativeMergeWorker returns 'unmerged_state' status and halts."""
        wt = await _make_branch_with_file(
            git_ops, 'uu-sw-1', 'file_uu_sw.py', 'uu_sw = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        async def _unmerged_state(*args: Any, **kwargs: Any):
            return 'unmerged_state'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_unmerged_state),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('uu-sw-1', 'uu-sw-1', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'unmerged_state'
        assert 'unmerged' in outcome.reason.lower()
        assert worker.is_wip_halted

        await worker.stop()
        await worker_task

    async def test_speculative_pop_conflict_no_advance_outcome(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """pop_conflict_no_advance in SpeculativeMergeWorker returns wip_recovery_no_advance."""
        wt = await _make_branch_with_file(
            git_ops, 'pcna-sw-1', 'file_pcna_sw.py', 'pcna_sw = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        async def _pop_conflict_no_advance(*args: Any, **kwargs: Any):
            git_ops._last_recovery_branch = 'wip/recovery-x-y'
            return 'pop_conflict_no_advance'

        with (
            patch.object(git_ops, 'advance_main', side_effect=_pop_conflict_no_advance),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('pcna-sw-1', 'pcna-sw-1', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'wip_recovery_no_advance'
        assert outcome.recovery_branch == 'wip/recovery-x-y'
        assert 'did not advance' in outcome.reason.lower()
        assert worker.is_wip_halted

        await worker.stop()
        await worker_task


@pytest.mark.parametrize(
    'worker_cls', [MergeWorker, SpeculativeMergeWorker],
)
class TestHaltOwnerMechanics:
    """Halt-owner pointer: single source of truth for resolve-callback un-halt.

    Both MergeWorker and SpeculativeMergeWorker implement the same contract.
    These tests exercise the mechanics directly — no merge flow, just the
    halt-owner state machine. Integration is covered in test_workflow_e2e.
    """

    def test_fresh_worker_has_no_halt_owner(
        self, worker_cls, git_ops: GitOps,
    ):
        """Freshly constructed worker: not halted, owner is None, is_halt_owner is False."""
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = worker_cls(git_ops, queue)

        assert not worker.is_wip_halted
        assert worker.is_halt_owner('any-id') is False
        assert worker._halt_owner_esc_id is None

    def test_halt_for_wip_clears_owner(
        self, worker_cls, git_ops: GitOps,
    ):
        """halt_for_wip sets the halt flag and clears owner (workflow registers after)."""
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = worker_cls(git_ops, queue)

        worker.halt_for_wip('test reason')
        assert worker.is_wip_halted
        assert worker._halt_owner_esc_id is None
        assert worker.is_halt_owner('any-id') is False

    def test_set_halt_owner_registers_id(
        self, worker_cls, git_ops: GitOps,
    ):
        """set_halt_owner records the id; is_halt_owner matches on equality only."""
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = worker_cls(git_ops, queue)

        worker.halt_for_wip('test reason')
        worker.set_halt_owner('esc-42-1')

        assert worker.is_halt_owner('esc-42-1') is True
        assert worker.is_halt_owner('esc-42-2') is False
        assert worker.is_halt_owner('esc-99-1') is False

    def test_set_halt_owner_rejects_double_register(
        self, worker_cls, git_ops: GitOps,
    ):
        """set_halt_owner raises when owner is already set — catches double-halt bugs."""
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = worker_cls(git_ops, queue)

        worker.halt_for_wip('test reason')
        worker.set_halt_owner('esc-42-1')

        with pytest.raises(AssertionError, match='halt owner already set'):
            worker.set_halt_owner('esc-42-2')

    def test_unhalt_wip_clears_owner(
        self, worker_cls, git_ops: GitOps,
    ):
        """unhalt_wip releases the halt and clears the owner pointer."""
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = worker_cls(git_ops, queue)

        worker.halt_for_wip('test reason')
        worker.set_halt_owner('esc-42-1')
        worker.unhalt_wip()

        assert not worker.is_wip_halted
        assert worker._halt_owner_esc_id is None
        assert worker.is_halt_owner('esc-42-1') is False

    def test_halt_cycle_allows_reuse(
        self, worker_cls, git_ops: GitOps,
    ):
        """After a full halt→unhalt cycle, a new owner can be registered."""
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = worker_cls(git_ops, queue)

        worker.halt_for_wip('first')
        worker.set_halt_owner('esc-1-1')
        worker.unhalt_wip()

        worker.halt_for_wip('second')
        worker.set_halt_owner('esc-2-1')
        assert worker.is_halt_owner('esc-2-1') is True
        assert worker.is_halt_owner('esc-1-1') is False


# ---------------------------------------------------------------------------
# TestEnqueueMergeRequest — step-3
# ---------------------------------------------------------------------------


class TestEnqueueMergeRequest:
    """Tests for the module-level enqueue_merge_request helper."""

    @pytest.mark.asyncio
    async def test_enqueue_helper_emits_merge_queued_and_puts_on_queue(
        self, tmp_path: Path, config: OrchestratorConfig,
    ):
        """enqueue_merge_request emits merge_queued and places req on queue."""
        from orchestrator.merge_queue import enqueue_merge_request

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        db_path = tmp_path / 'runs.db'
        event_store = EventStore(db_path, 'run-1')

        wt = tmp_path / 'wt'
        wt.mkdir()
        req = _make_request('42', 'task/42', wt, config)

        await enqueue_merge_request(queue, req, event_store)

        # Queue has exactly one item which is our req
        assert queue.qsize() == 1
        dequeued = queue.get_nowait()
        assert dequeued is req

        # Exactly one merge_queued row in events
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, task_id, phase, "
            "json_extract(data, '$.branch') AS branch "
            "FROM events WHERE event_type = 'merge_queued'"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == 'merge_queued'
        assert rows[0][1] == '42'
        assert rows[0][2] == 'merge'
        assert rows[0][3] == 'task/42'

    @pytest.mark.asyncio
    async def test_enqueue_helper_with_none_event_store_still_enqueues(
        self, tmp_path: Path, config: OrchestratorConfig,
    ):
        """Passing event_store=None must still enqueue and not raise."""
        from orchestrator.merge_queue import enqueue_merge_request

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        wt = tmp_path / 'wt'
        wt.mkdir()
        req = _make_request('99', 'task/99', wt, config)

        await enqueue_merge_request(queue, req, None)

        assert queue.qsize() == 1
        dequeued = queue.get_nowait()
        assert dequeued is req


# ---------------------------------------------------------------------------
# TestMergeWorkerDequeueEvent — step-5
# ---------------------------------------------------------------------------


class TestMergeWorkerDequeueEvent:
    """MergeWorker emits merge_dequeued after dequeuing a request."""

    @pytest.mark.asyncio
    async def test_merge_worker_emits_merge_dequeued_after_dequeue(
        self, tmp_path: Path, config: OrchestratorConfig, git_ops: GitOps,
    ):
        """MergeWorker emits merge_dequeued after pulling request from queue.

        Timestamp of merge_dequeued must be >= merge_queued timestamp.
        """
        from orchestrator.merge_queue import enqueue_merge_request

        db_path = tmp_path / 'events.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=event_store)

        wt = tmp_path / 'wt'
        wt.mkdir()
        req = _make_request('42', 'task/42', wt, config)

        # Patch _do_merge so it immediately returns 'done' without git ops
        async def _fast_done(req):
            return MergeOutcome('done')

        worker_task = asyncio.create_task(worker.run())
        with patch.object(worker, '_do_merge', side_effect=_fast_done):
            await enqueue_merge_request(queue, req, event_store)
            outcome = await asyncio.wait_for(req.result, timeout=10)

        assert outcome.status == 'done'

        conn = sqlite3.connect(str(db_path))
        dequeued_rows = conn.execute(
            "SELECT event_type, task_id, timestamp FROM events "
            "WHERE event_type = 'merge_dequeued'"
        ).fetchall()
        queued_rows = conn.execute(
            "SELECT timestamp FROM events WHERE event_type = 'merge_queued'"
        ).fetchall()
        conn.close()

        assert len(dequeued_rows) == 1, f'Expected 1 merge_dequeued row, got: {dequeued_rows}'
        assert dequeued_rows[0][1] == '42'
        # merge_dequeued timestamp must be >= merge_queued timestamp
        assert len(queued_rows) == 1
        assert dequeued_rows[0][2] >= queued_rows[0][0]

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


# ---------------------------------------------------------------------------
# TestSpeculativeMergeWorkerDequeueEvent — step-7
# ---------------------------------------------------------------------------


class TestSpeculativeMergeWorkerDequeueEvent:
    """SpeculativeMergeWorker emits merge_dequeued after dequeuing a request."""

    @pytest.mark.asyncio
    async def test_speculative_worker_emits_merge_dequeued_after_dequeue(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """SpeculativeMergeWorker emits merge_dequeued after dequeuing.

        Uses an immediate conflict path (merge_to_main returns conflicts=True)
        so the test is fast and doesn't need real git merge work.
        """
        from orchestrator.git_ops import MergeResult
        from orchestrator.merge_queue import enqueue_merge_request

        db_path = tmp_path / 'events.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        # Use a real worktree so rev-parse HEAD works
        wt = await _make_branch_with_file(
            git_ops, 'spec-deq', 'spec_deq.py', 'x = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=event_store)
        worker._shutdown_timeout = 2.0

        # Force an immediate conflict so the merger resolves quickly
        conflict_result = MergeResult(
            success=False, conflicts=True, details='conflict',
            merge_worktree=None, merge_commit=None, pre_merge_sha=None,
        )

        worker_task = asyncio.create_task(worker.run())
        with patch.object(git_ops, 'merge_to_main', return_value=conflict_result):
            req = _make_request('spec-deq', 'spec-deq', wt, config)
            await enqueue_merge_request(queue, req, event_store)
            outcome = await asyncio.wait_for(req.result, timeout=10)

        assert outcome.status == 'conflict'

        conn = sqlite3.connect(str(db_path))
        dequeued_rows = conn.execute(
            "SELECT event_type, task_id, timestamp FROM events "
            "WHERE event_type = 'merge_dequeued'"
        ).fetchall()
        queued_rows = conn.execute(
            "SELECT timestamp FROM events WHERE event_type = 'merge_queued'"
        ).fetchall()
        conn.close()

        assert len(dequeued_rows) == 1, f'Expected 1 merge_dequeued row, got: {dequeued_rows}'
        assert dequeued_rows[0][1] == 'spec-deq'
        # merge_dequeued timestamp must be >= merge_queued timestamp
        assert len(queued_rows) == 1
        assert dequeued_rows[0][2] >= queued_rows[0][0]

        await worker.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


# ---------------------------------------------------------------------------
# TestMergeWorkerCasRetryEmitsMergeQueued — step-9
# ---------------------------------------------------------------------------


class TestMergeWorkerCasRetryEmitsMergeQueued:
    """MergeWorker emits merge_queued when re-enqueuing on CAS retry."""

    @pytest.mark.asyncio
    async def test_cas_retry_reenqueue_emits_merge_queued(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """CAS retry path emits a second merge_queued, then merge_dequeued, then done.

        Event sequence expected:
          merge_queued        (initial enqueue via helper)
          merge_dequeued      (worker picks up request the first time)
          merge_attempt(cas_retry)
          merge_queued        (re-enqueue on CAS failure)
          merge_dequeued      (worker picks up from _urgent)
          merge_attempt(done)
        """
        db_path = tmp_path / 'events.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        wt = await _make_branch_with_file(
            git_ops, 'cas-evt', 'cas_evt.py', 'x = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=event_store)
        worker_task = asyncio.create_task(worker.run())

        original_advance = git_ops.advance_main
        call_count = 0

        async def _fail_once(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 'cas_failed'
            return await original_advance(*args, **kwargs)

        with (
            patch.object(git_ops, 'advance_main', side_effect=_fail_once),
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
        ):
            req = _make_request('cas-evt', 'cas-evt', wt, config)
            from orchestrator.merge_queue import enqueue_merge_request
            await enqueue_merge_request(queue, req, event_store)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done'
        assert call_count == 2

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, json_extract(data, '$.outcome') AS outcome "
            "FROM events ORDER BY id"
        ).fetchall()
        conn.close()

        event_types = [(r[0], r[1]) for r in rows]

        # Count merge_queued rows for this task — expect exactly 2
        queued_count = sum(1 for et, _ in event_types if et == 'merge_queued')
        assert queued_count == 2, f'Expected 2 merge_queued rows, got: {event_types}'

        # Count merge_dequeued rows — expect exactly 2
        dequeued_count = sum(1 for et, _ in event_types if et == 'merge_dequeued')
        assert dequeued_count == 2, f'Expected 2 merge_dequeued rows, got: {event_types}'

        # Exactly one cas_retry and one done attempt
        attempt_outcomes = [out for et, out in event_types if et == 'merge_attempt']
        assert 'cas_retry' in attempt_outcomes, f'Expected cas_retry in attempts: {attempt_outcomes}'
        assert 'done' in attempt_outcomes, f'Expected done in attempts: {attempt_outcomes}'

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


# ---------------------------------------------------------------------------
# TestWorkflowSubmitUsesEnqueueHelper — step-11 test
# ---------------------------------------------------------------------------


class TestWorkflowSubmitUsesEnqueueHelper:
    """_submit_to_merge_queue delegates to enqueue_merge_request instead of put() directly."""

    @pytest.mark.asyncio
    async def test_submit_to_merge_queue_calls_enqueue_helper(self, tmp_path: Path):
        """_submit_to_merge_queue calls enqueue_merge_request with (queue, req, event_store).

        Before step-12 impl, the function calls self.merge_queue.put() directly and
        never calls enqueue_merge_request — so mock_helper.assert_called_once() fails.
        After step-12, the function calls enqueue_merge_request — assertion passes.
        """
        from orchestrator.merge_queue import MergeOutcome, MergeRequest
        from orchestrator.workflow import TaskWorkflow

        # Minimal assignment mock (mirrors test_workflow_escalation_warning pattern)
        assignment = MagicMock()
        assignment.task_id = '42'
        assignment.task = {'id': '42', 'title': 'T', 'description': 'desc'}
        assignment.modules = []

        _spec = pydantic_spec(OrchestratorConfig)
        wf_config = MagicMock(spec_set=_spec)
        wf_config.fused_memory.project_id = 'test'
        wf_config.fused_memory.url = 'http://localhost'
        wf_config.max_review_cycles = 2
        wf_config.max_amendment_rounds = 1
        wf_config.lock_depth = 2
        wf_config.steward_completion_timeout = 300.0

        workflow = TaskWorkflow(
            assignment=assignment,
            config=wf_config,
            git_ops=MagicMock(),
            scheduler=MagicMock(),
            briefing=MagicMock(),
            mcp=MagicMock(),
        )

        # Wire required attributes
        merge_queue_mock: AsyncMock = AsyncMock()
        event_store_mock = MagicMock()
        workflow.merge_queue = merge_queue_mock
        workflow.event_store = event_store_mock
        workflow.worktree = tmp_path / 'wt'
        workflow.worktree.mkdir()

        # Before step-12: merge_queue.put() is called directly → resolve future
        # so _submit_to_merge_queue doesn't hang.
        async def _put_resolves_future(req):
            if isinstance(req, MergeRequest) and not req.result.done():
                req.result.set_result(MergeOutcome('done'))

        merge_queue_mock.put.side_effect = _put_resolves_future

        # After step-12: enqueue_merge_request is called → resolve future via mock.
        async def _mock_enqueue(queue, req, es):
            if not req.result.done():
                req.result.set_result(MergeOutcome('done'))

        mock_helper = AsyncMock(side_effect=_mock_enqueue)

        # Patch the source module so both local and module-level imports get the mock.
        with patch('orchestrator.merge_queue.enqueue_merge_request', mock_helper):
            await workflow._submit_to_merge_queue('task/42')

        # KEY: enqueue_merge_request must have been called exactly once
        mock_helper.assert_called_once()
        call_queue, call_req, call_es = mock_helper.call_args.args
        assert call_queue is merge_queue_mock
        assert isinstance(call_req, MergeRequest)
        assert call_req.task_id == '42'
        assert call_req.branch == 'task/42'
        assert call_es is event_store_mock


# ---------------------------------------------------------------------------
# TestEscalationServerUsesEnqueueHelper — step-13 test
# ---------------------------------------------------------------------------


class TestEscalationServerUsesEnqueueHelper:
    """escalation server merge_request tool delegates to enqueue_merge_request."""

    @pytest.mark.asyncio
    async def test_escalation_server_merge_request_uses_enqueue_helper(
        self, tmp_path: Path,
    ):
        """merge_request tool calls enqueue_merge_request(queue, req, event_store).

        Must fail until escalation/server.py accepts the event_store kwarg (step-14)
        and replaces merge_queue.put() with the helper.
        """
        from escalation.server import create_server

        from orchestrator.event_store import EventStore
        from orchestrator.merge_queue import MergeOutcome, MergeRequest

        merge_queue: asyncio.Queue = asyncio.Queue()
        event_store = EventStore(db_path=tmp_path / 'test.db', run_id='test')

        # Stub orch_config — _module_configs is a PrivateAttr, not in model_fields
        _spec2 = pydantic_spec(OrchestratorConfig)
        stub_config = MagicMock(spec_set=_spec2)
        stub_config._module_configs = {}

        # Mock resolves the future so the tool doesn't hang
        async def _mock_enqueue(queue, req, es):
            if not req.result.done():
                req.result.set_result(MergeOutcome('done'))

        mock_helper = AsyncMock(side_effect=_mock_enqueue)

        # Patch the source module so local imports inside the tool get the mock
        with patch('orchestrator.merge_queue.enqueue_merge_request', mock_helper):
            # Before step-14: create_server raises TypeError (unexpected kwarg)
            mcp = create_server(
                MagicMock(),
                merge_queue=merge_queue,
                orch_config=stub_config,
                event_store=event_store,
            )
            from fastmcp.tools.function_tool import FunctionTool
            tool = await mcp.get_tool('merge_request')
            assert isinstance(tool, FunctionTool)
            await tool.fn(task_id='9', branch='task/9', worktree='/tmp/x')

        # Helper must have been called exactly once with the right args
        mock_helper.assert_called_once()
        call_queue, call_req, call_es = mock_helper.call_args.args
        assert call_queue is merge_queue
        assert isinstance(call_req, MergeRequest)
        assert call_req.task_id == '9'
        assert call_req.branch == 'task/9'
        assert call_es is event_store


# ---------------------------------------------------------------------------
# TestEscalationServerMergeRequestModuleConfigsNone — step-1 regression test
# ---------------------------------------------------------------------------


class TestEscalationServerMergeRequestModuleConfigsNone:
    """Regression: merge_request must not raise when _module_configs is None.

    Post-1405, OrchestratorConfig._module_configs defaults to None (PrivateAttr).
    Configs built by direct instantiation (e.g. build_eval_orch_config) keep the
    sentinel at None.  Pre-fix server.py:321 calls None.values() → AttributeError.
    """

    @pytest.mark.asyncio
    async def test_merge_request_with_none_module_configs_does_not_raise(
        self, tmp_path: Path,
    ):
        """merge_request tool must succeed and pass module_configs=[] when
        _module_configs is the post-1405 None sentinel (not {})."""
        from escalation.server import create_server

        from orchestrator.event_store import EventStore
        from orchestrator.merge_queue import MergeOutcome, MergeRequest

        merge_queue: asyncio.Queue = asyncio.Queue()
        event_store = EventStore(db_path=tmp_path / 'test.db', run_id='test')

        # Real OrchestratorConfig — _module_configs stays at its PrivateAttr default
        # (None), which is exactly the post-1405 direct-instantiation path exercised
        # by build_eval_orch_config in evals/runner.py.  Using a real instance means
        # the module_configs_or_empty property is actually invoked and the None→{}
        # normalization is genuinely tested (not shadowed by MagicMock's __iter__).
        stub_config = OrchestratorConfig(project_root=tmp_path)

        # Mock resolves the future so the tool doesn't hang
        async def _mock_enqueue(queue, req, es):
            if not req.result.done():
                req.result.set_result(MergeOutcome('done'))

        mock_helper = AsyncMock(side_effect=_mock_enqueue)

        with patch('orchestrator.merge_queue.enqueue_merge_request', mock_helper):
            mcp = create_server(
                MagicMock(),
                merge_queue=merge_queue,
                orch_config=stub_config,
                event_store=event_store,
            )
            from fastmcp.tools.function_tool import FunctionTool
            tool = await mcp.get_tool('merge_request')
            assert isinstance(tool, FunctionTool)
            # Pre-fix: raises AttributeError: 'NoneType' object has no attribute 'values'
            await tool.fn(task_id='9', branch='task/9', worktree='/tmp/x')

        # (a) no AttributeError — we reached here
        mock_helper.assert_called_once()
        call_queue, call_req, call_es = mock_helper.call_args.args
        assert isinstance(call_req, MergeRequest)
        # (c) None sentinel must collapse to [] via module_configs_or_empty property
        assert call_req.module_configs == []


# ---------------------------------------------------------------------------
# TestPushHook — main is mirrored to origin after every successful CAS advance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPushHook:
    """push_main fires once per successful CAS advance, in both worker paths.

    Push status is surfaced on MergeOutcome.push_status. A push failure must
    not change the merge outcome — local main has already been advanced.
    """

    async def test_merge_worker_invokes_push_main_on_success(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Normal MergeWorker success path calls push_main exactly once and
        propagates the result onto MergeOutcome.push_status."""
        worktree = await _make_branch_with_file(
            git_ops, 'push-hook-1', 'push_hook_1.py', 'x = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        push_mock = AsyncMock(return_value='pushed')
        with patch.object(git_ops, 'push_main', push_mock), \
             patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req = _make_request('push-hook-1', 'push-hook-1', worktree, config)
            await queue.put(req)
            result = await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

        assert result.status == 'done'
        assert result.push_status == 'pushed'
        assert push_mock.await_count == 1

    async def test_merge_worker_done_when_push_fails(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """A push 'error' must not change merge status — main was advanced."""
        worktree = await _make_branch_with_file(
            git_ops, 'push-hook-fail', 'push_hook_fail.py', 'y = 2\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        push_mock = AsyncMock(return_value='error')
        with patch.object(git_ops, 'push_main', push_mock), \
             patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req = _make_request('push-hook-fail', 'push-hook-fail', worktree, config)
            await queue.put(req)
            result = await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

        assert result.status == 'done'
        assert result.push_status == 'error'
        assert push_mock.await_count == 1

    async def test_speculative_worker_invokes_push_main_on_success(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """SpeculativeMergeWorker success path also calls push_main once."""
        worktree = await _make_branch_with_file(
            git_ops, 'spec-push', 'spec_push.py', 'z = 3\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        push_mock = AsyncMock(return_value='pushed')
        with patch.object(git_ops, 'push_main', push_mock), \
             patch(
                 'orchestrator.merge_queue.run_scoped_verification',
                 _mock_verify_pass(),
             ):
            req = _make_request('spec-push', 'spec-push', worktree, config)
            await queue.put(req)
            result = await asyncio.wait_for(req.result, timeout=30)

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

        assert result.status == 'done'
        assert result.push_status == 'pushed'
        assert push_mock.await_count == 1


# ---------------------------------------------------------------------------
# TestWorktreeMissing — surface as ``blocked`` with recognisable reason
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWorktreeMissingHandling:
    """Both merge workers tolerate a deleted task worktree.

    The plan: when a human marks a task ``done`` and removes its worktree
    while the merge queue is processing the request, the worker must NOT
    raise an unhandled exception or emit a generic ``blocked`` outcome that
    the workflow then re-escalates.  Instead, the outcome.reason starts with
    ``WORKTREE_MISSING_REASON_PREFIX`` so the workflow can re-check task
    status and short-circuit to DONE.
    """

    async def test_merge_worker_surfaces_worktree_missing(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        worktree = (await git_ops.create_worktree('worktree-missing')).path
        (worktree / 'f.py').write_text('x=1\n')
        await git_ops.commit(worktree, 'add f')
        # Remove worktree directory before submission to simulate the race
        # where the human has already deleted the worktree.
        import shutil as _shutil
        _shutil.rmtree(worktree)

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())
        try:
            req = _make_request('worktree-missing', 'worktree-missing', worktree, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=15)
        finally:
            await worker.stop()
            worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task

        assert outcome.status == 'blocked'
        assert outcome.reason.startswith(WORKTREE_MISSING_REASON_PREFIX), (
            f'unexpected reason: {outcome.reason!r}'
        )


@pytest.mark.asyncio
class TestCheckPlanFilesTouchedInBranch:
    """Unit tests for the pre-merge Decision-1 subset check."""

    async def test_empty_plan_files_returns_empty(self, git_ops: GitOps):
        """Empty plan.files → vacuously satisfied (no entries to check)."""
        result = await _check_plan_files_touched_in_branch(
            [], 'a' * 40, 'b' * 40, git_ops,
        )
        assert result.not_touched == []

    async def test_all_plan_files_touched_in_branch(self, git_ops: GitOps):
        """Every plan entry appears in the branch history → empty not_touched."""
        wt = (await git_ops.create_worktree('plan-touched-all')).path
        rc, base_out, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=wt,
        )
        assert rc == 0
        base = base_out.strip()
        (wt / 'src').mkdir()
        (wt / 'src' / 'a.py').write_text('a = 1\n')
        (wt / 'src' / 'b.py').write_text('b = 2\n')
        await git_ops.commit(wt, 'Add a.py + b.py')
        rc, head_out, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=wt,
        )
        head = head_out.strip()

        result = await _check_plan_files_touched_in_branch(
            ['src/a.py', 'src/b.py'], base, head, git_ops,
            task_id='all-touched',
        )
        assert result.not_touched == []

    async def test_single_plan_file_not_touched_flagged(
        self, git_ops: GitOps, caplog: pytest.LogCaptureFixture,
    ):
        """A plan entry the branch never touched → flagged + structured WARNING."""
        wt = (await git_ops.create_worktree('plan-touched-missing')).path
        rc, base_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        assert rc == 0
        base = base_out.strip()
        (wt / 'src').mkdir()
        (wt / 'src' / 'a.py').write_text('a = 1\n')
        await git_ops.commit(wt, 'Add a.py')
        rc, head_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        head = head_out.strip()

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = await _check_plan_files_touched_in_branch(
                ['src/a.py', 'src/never_touched.py'],
                base, head, git_ops, task_id='miss-test',
            )

        assert result.not_touched == ['src/never_touched.py']
        warn = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warn, 'expected a structured WARNING when not_touched non-empty'
        msg = ' '.join(r.getMessage() for r in warn)
        assert 'miss-test' in msg
        assert 'src/never_touched.py' in msg

    async def test_directory_prefix_match_passes(self, git_ops: GitOps):
        """A plan entry that resolves to a directory passes when any touched
        file has it as a path prefix.
        """
        wt = (await git_ops.create_worktree('plan-touched-dir')).path
        rc, base_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        assert rc == 0
        base = base_out.strip()
        (wt / 'src' / 'pkg').mkdir(parents=True)
        (wt / 'src' / 'pkg' / 'mod_a.py').write_text('a = 1\n')
        (wt / 'src' / 'pkg' / 'mod_b.py').write_text('b = 2\n')
        await git_ops.commit(wt, 'Add pkg/*.py')
        rc, head_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        head = head_out.strip()

        # Plan declares a directory; the touched-set has pkg/mod_a.py and
        # pkg/mod_b.py — directory entry must satisfy the gate.
        result = await _check_plan_files_touched_in_branch(
            ['src/pkg'], base, head, git_ops, task_id='dir-test',
        )
        assert result.not_touched == []

    async def test_unknown_entry_neither_file_nor_directory_flagged(
        self, git_ops: GitOps,
    ):
        """Plan entry that's neither a touched file nor a directory in the
        branch tree → flagged (it's a typo or stale plan reference)."""
        wt = (await git_ops.create_worktree('plan-touched-typo')).path
        rc, base_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        assert rc == 0
        base = base_out.strip()
        (wt / 'real.py').write_text('r = 1\n')
        await git_ops.commit(wt, 'Add real.py')
        rc, head_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        head = head_out.strip()

        result = await _check_plan_files_touched_in_branch(
            ['real.py', 'phantom/typo.py'], base, head, git_ops,
        )
        assert result.not_touched == ['phantom/typo.py']


@pytest.mark.asyncio
class TestCheckPostMergeEquivalence:
    """Unit tests for the post-merge Decision-2 content-equivalence check."""

    async def test_ff_merge_passes(self, git_ops: GitOps):
        """A clean merge whose advanced main matches branch HEAD → empty list."""
        wt = (await git_ops.create_worktree('equiv-ff')).path
        (wt / 'a.py').write_text('a = 1\n')
        await git_ops.commit(wt, 'Add a.py')

        # Capture the pre-merge main tip — the gate anchors its merge-base on
        # main_sha, not the post-advance SHA.
        pre_merge_main = await git_ops.get_main_sha()
        merge_result = await git_ops.merge_to_main(wt, 'equiv-ff')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None
        try:
            await git_ops.advance_main(
                merge_result.merge_commit, merge_result.merge_worktree,
                branch='equiv-ff', max_attempts=1,
            )
            advanced = (
                getattr(git_ops, '_last_advanced_sha', None)
                or merge_result.merge_commit
            )
            assert advanced is not None
            failed = await _check_post_merge_equivalence(
                wt, advanced, git_ops, pre_merge_main, task_id='equiv-ff-test',
            )
            assert failed == []
        finally:
            await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_diverging_tree_flags_files(
        self, git_ops: GitOps,
    ):
        """When advanced main has a file the branch lacks (or vice versa),
        the diverging path is reported.

        Synthesized by pointing the check at a SHA that doesn't match the
        branch — cheaper than orchestrating a real conflict resolution."""
        wt = (await git_ops.create_worktree('equiv-diverge')).path
        (wt / 'branch_only.py').write_text('b = 1\n')
        await git_ops.commit(wt, 'Add branch_only.py')
        rc, branch_head_out, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=wt,
        )
        branch_head = branch_head_out.strip()

        # Use the worktree's BASE SHA as the "advanced" reference — the
        # branch's commit isn't there, so the diff names branch_only.py.
        rc, base_out, _ = await _run(
            ['git', 'rev-parse', 'HEAD~1'], cwd=wt,
        )
        assert rc == 0
        synthetic_main = base_out.strip()

        # Pass synthetic_main as main_sha too: it's the shared baseline, so
        # main_touched is empty and the gate degrades to strict equivalence —
        # this guards that branch-only divergence is STILL flagged.
        failed = await _check_post_merge_equivalence(
            wt, synthetic_main, git_ops, synthetic_main,
            task_id='equiv-diverge-test',
        )
        assert failed == ['branch_only.py']
        assert branch_head != synthetic_main  # sanity

    async def test_dot_task_diff_excluded(self, git_ops: GitOps):
        """``.task/`` differences are excluded — they legitimately diverge
        post-cleanup and would false-positive otherwise."""
        wt = (await git_ops.create_worktree('equiv-task-only')).path
        (wt / 'real.py').write_text('r = 1\n')
        await git_ops.commit(wt, 'Add real.py')

        pre_merge_main = await git_ops.get_main_sha()
        merge_result = await git_ops.merge_to_main(wt, 'equiv-task-only')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None
        try:
            await git_ops.advance_main(
                merge_result.merge_commit, merge_result.merge_worktree,
                branch='equiv-task-only', max_attempts=1,
            )
            advanced = (
                getattr(git_ops, '_last_advanced_sha', None)
                or merge_result.merge_commit
            )
            assert advanced is not None

            # Modify .task/ on the branch — should be excluded from the diff.
            (wt / '.task').mkdir(exist_ok=True)
            (wt / '.task' / 'plan.json').write_text('{"local": true}\n')
            # Note: we don't commit .task/; it's untracked, so it doesn't
            # affect the branch HEAD commit's tree.  This test asserts the
            # exclusion pathspec doesn't fire spurious flags.

            failed = await _check_post_merge_equivalence(
                wt, advanced, git_ops, pre_merge_main, task_id='equiv-task-test',
            )
            assert failed == []
        finally:
            await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_sibling_also_touched_lockfile_not_flagged(
        self, git_ops: GitOps,
    ):
        """A shared lockfile both sides edited must not be flagged (esc-3843).

        The branch and a sibling each append a different dependency to
        Cargo.lock in non-adjacent regions; a clean 3-way merge combines
        both, so advanced main's Cargo.lock differs from the branch tip.  The
        path is in both branch_touched and main_touched, so it's subtracted
        and the gate reports no divergence.
        """
        # Base Cargo.lock with enough spacing that the two edits don't collide.
        base_lock = ''.join(f'line{i}\n' for i in range(20))
        (git_ops.project_root / 'Cargo.lock').write_text(base_lock)
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Main: add Cargo.lock'],
            cwd=git_ops.project_root,
        )

        # Victim forks here and adds its dep near the TOP of the file.
        wt = (await git_ops.create_worktree('lockfile-victim')).path
        (wt / 'Cargo.lock').write_text(
            base_lock.replace('line1\n', 'line1\nvictim-dep\n')
        )
        await git_ops.commit(wt, 'Victim: add victim-dep to Cargo.lock')
        rc, branch_head_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        assert rc == 0
        branch_head = branch_head_out.strip()

        # Sibling adds its dep near the BOTTOM of main's Cargo.lock.
        (git_ops.project_root / 'Cargo.lock').write_text(
            base_lock.replace('line18\n', 'line18\nsibling-dep\n')
        )
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Sibling: add sibling-dep to Cargo.lock'],
            cwd=git_ops.project_root,
        )
        main_sha = await git_ops.get_main_sha()

        merge_result = await git_ops.merge_to_main(wt, 'lockfile-victim')
        assert merge_result.success, (
            f'expected clean 3-way merge; details={merge_result.details!r}'
        )
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None
        try:
            await git_ops.advance_main(
                merge_result.merge_commit, merge_result.merge_worktree,
                branch='lockfile-victim', max_attempts=1,
            )
            advanced = (
                getattr(git_ops, '_last_advanced_sha', None)
                or merge_result.merge_commit
            )
            assert advanced is not None

            # Sanity: advanced main's Cargo.lock really does differ from the
            # branch tip (it carries the sibling's dep too).
            rc, diff_out, _ = await _run(
                ['git', 'diff', '--name-only', branch_head, advanced,
                 '--', 'Cargo.lock'],
                cwd=git_ops.project_root,
            )
            assert rc == 0
            assert 'Cargo.lock' in diff_out, (
                'expected advanced main Cargo.lock to differ from branch tip'
            )

            failed = await _check_post_merge_equivalence(
                wt, advanced, git_ops, main_sha, task_id='lockfile-victim',
            )
            assert failed == [], (
                f'shared lockfile must not be flagged; got {failed!r}'
            )
        finally:
            await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

    async def test_speculative_base_equals_main_sha_strict_equivalence(
        self, git_ops: GitOps,
    ):
        """base == main_sha (speculative) → main_touched empty → strict
        equivalence: a clean merge passes, a drop is still flagged."""
        wt = (await git_ops.create_worktree('spec-equiv')).path
        # The fork point is the speculative base; it doubles as main_sha.
        base_sha = await git_ops.get_main_sha()
        (wt / 'spec.py').write_text('s = 1\n')
        await git_ops.commit(wt, 'Add spec.py')
        rc, branch_head_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        assert rc == 0
        branch_head = branch_head_out.strip()

        # Clean: a real merge+advance preserves spec.py → [].
        merge_result = await git_ops.merge_to_main(wt, 'spec-equiv')
        assert merge_result.success
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None
        try:
            await git_ops.advance_main(
                merge_result.merge_commit, merge_result.merge_worktree,
                branch='spec-equiv', max_attempts=1,
            )
            advanced = (
                getattr(git_ops, '_last_advanced_sha', None)
                or merge_result.merge_commit
            )
            clean = await _check_post_merge_equivalence(
                wt, advanced, git_ops, base_sha, task_id='spec-equiv-clean',
            )
            assert clean == []
        finally:
            await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)

        # Drop: point advanced at base_sha (spec.py absent).  main_touched is
        # empty (base == main_sha), so strict equivalence flags spec.py.
        dropped = await _check_post_merge_equivalence(
            wt, base_sha, git_ops, base_sha, task_id='spec-equiv-drop',
        )
        assert dropped == ['spec.py']
        assert branch_head != base_sha  # sanity

    async def test_equivalence_fails_open_on_bad_main_sha(
        self, git_ops: GitOps,
    ):
        """A merge-base failure (bogus main_sha) fails open → no divergence."""
        wt = (await git_ops.create_worktree('failopen-equiv')).path
        (wt / 'g.py').write_text('g = 1\n')
        await git_ops.commit(wt, 'Add g.py')

        failed = await _check_post_merge_equivalence(
            wt, await git_ops.get_main_sha(), git_ops,
            'definitely-not-a-ref', task_id='failopen-equiv',
        )
        assert failed == []


@pytest.mark.asyncio
async def test_speculative_merger_surfaces_worktree_missing_after_plan_touched_class(
    git_ops, config,
):
    """Re-anchor the speculative-merger worktree-missing test after the
    plan-files-touched test class (the original function body was clipped
    when the new class was inserted)."""
    worktree = (await git_ops.create_worktree('spec-worktree-missing')).path
    (worktree / 'f.py').write_text('y=2\n')
    await git_ops.commit(worktree, 'add f')
    import shutil as _shutil
    _shutil.rmtree(worktree)

    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker = SpeculativeMergeWorker(git_ops, queue)
    worker_task = asyncio.create_task(worker.run())
    try:
        req = _make_request(
            'spec-worktree-missing', 'spec-worktree-missing', worktree, config,
        )
        await queue.put(req)
        outcome = await asyncio.wait_for(req.result, timeout=15)
    finally:
        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

    assert outcome.status == 'blocked'
    assert outcome.reason.startswith(WORKTREE_MISSING_REASON_PREFIX), (
        f'unexpected reason: {outcome.reason!r}'
    )


# ---------------------------------------------------------------------------
# A2: transient-ENOSPC detection, pruning, and prune-and-retry
# ---------------------------------------------------------------------------


def _enospc_verify_result() -> VerifyResult:
    """A failed VerifyResult whose captured output bears an ENOSPC signature."""
    return VerifyResult(
        passed=False,
        test_output='E   OSError: [Errno 28] No space left on device',
        lint_output='',
        type_output='',
        summary='disk full',
    )


class TestVerifyHitEnospc:
    """Unit tests for the _verify_hit_enospc string-match detector."""

    def test_detects_no_space_left_phrase(self):
        assert _verify_hit_enospc(_enospc_verify_result()) is True

    def test_detects_os_error_28_in_lint_output(self):
        v = VerifyResult(
            passed=False, test_output='', lint_output='build: os error 28',
            type_output='', summary='',
        )
        assert _verify_hit_enospc(v) is True

    def test_detects_bare_enospc_token_in_type_output(self):
        v = VerifyResult(
            passed=False, test_output='', lint_output='',
            type_output='write failed: ENOSPC', summary='',
        )
        assert _verify_hit_enospc(v) is True

    def test_ordinary_test_failure_is_not_enospc(self):
        v = VerifyResult(
            passed=False, test_output='2 failed, 3 passed',
            lint_output='', type_output='', summary='tests failed',
        )
        assert _verify_hit_enospc(v) is False

    def test_non_string_outputs_are_skipped_without_raising(self):
        # A bare MagicMock (the shape several existing verify tests use) must
        # not raise — non-string attributes are filtered out, yielding False.
        assert _verify_hit_enospc(
            MagicMock(passed=False, summary='tests failed'),
        ) is False


@pytest.mark.asyncio
class TestPruneStaleMergeWorktrees:
    """GitOps.prune_stale_merge_worktrees removes _merge-* worktrees only."""

    async def test_prunes_stale_keeps_active_and_never_touches_tasks(
        self, git_ops: GitOps,
    ):
        keep_wt, _ = await git_ops._create_merge_worktree()
        stale_a, _ = await git_ops._create_merge_worktree()
        stale_b, _ = await git_ops._create_merge_worktree()
        # A live task worktree must never be touched.
        task_wt = (await git_ops.create_worktree('live-task')).path

        removed = await git_ops.prune_stale_merge_worktrees(keep=keep_wt)

        assert len(removed) == 2
        assert not stale_a.exists()
        assert not stale_b.exists()
        assert keep_wt.exists()  # the active merge wt survives
        assert task_wt.exists()  # task worktrees are never pruned

    async def test_prune_with_no_keep_removes_all_merge_worktrees(
        self, git_ops: GitOps,
    ):
        a, _ = await git_ops._create_merge_worktree()
        b, _ = await git_ops._create_merge_worktree()
        task_wt = (await git_ops.create_worktree('keepme')).path

        removed = await git_ops.prune_stale_merge_worktrees(keep=None)

        assert len(removed) == 2
        assert not a.exists()
        assert not b.exists()
        assert task_wt.exists()


@pytest.mark.asyncio
class TestEnospcTransientInfraRetry:
    """SpeculativeMergeWorker prunes + retries once on ENOSPC, then escalates
    as transient infra if it persists."""

    async def test_persistent_enospc_prunes_retries_once_then_blocks(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        wt = await _make_branch_with_file(
            git_ops, 'enospc-task', 'enospc.py', 'x = 1\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # Verify always reports ENOSPC → worker should retry exactly once.
        mock_verify = AsyncMock(return_value=_enospc_verify_result())

        with (
            patch('orchestrator.merge_queue.run_scoped_verification', mock_verify),
            patch.object(
                git_ops, 'prune_stale_merge_worktrees',
                AsyncMock(return_value=['/x/_merge-stale']),
            ) as mock_prune,
        ):
            req = _make_request('enospc-task', 'enospc-task', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked'
        assert outcome.reason.startswith(TRANSIENT_INFRA_REASON_PREFIX), (
            f'expected transient-infra reason, got: {outcome.reason!r}'
        )
        # Verify ran twice (initial + one prune-and-retry); prune ran once.
        assert mock_verify.call_count == 2
        assert mock_prune.call_count == 1

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

    async def test_enospc_then_pass_on_retry_completes_done(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """If the post-prune retry passes, the merge proceeds to 'done' and
        no transient-infra escalation is raised."""
        wt = await _make_branch_with_file(
            git_ops, 'enospc-heals', 'heals.py', 'y = 2\n',
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        # First verify hits ENOSPC; the retry (after prune) passes.
        passing = MagicMock(passed=True, summary='', timed_out=False)
        mock_verify = AsyncMock(
            side_effect=[_enospc_verify_result(), passing],
        )

        with (
            patch('orchestrator.merge_queue.run_scoped_verification', mock_verify),
            patch.object(
                git_ops, 'prune_stale_merge_worktrees',
                AsyncMock(return_value=[]),
            ) as mock_prune,
        ):
            req = _make_request('enospc-heals', 'enospc-heals', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done', f'unexpected: {outcome}'
        assert mock_verify.call_count == 2
        assert mock_prune.call_count == 1

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


# ---------------------------------------------------------------------------
# A2 (cont.): pre-verify disk guard
# ---------------------------------------------------------------------------

_GIB = 1024**3


def _usage(free_bytes: int):
    """Fake ``shutil.disk_usage`` return value exposing only ``.free``."""
    return MagicMock(free=free_bytes)


@pytest.mark.asyncio
class TestEnsureVerifyDiskSpace:
    """Unit tests for the _ensure_verify_disk_space pre-verify guard helper."""

    def _git_ops(self, pruned: list[str] | None = None) -> MagicMock:
        gops = MagicMock()
        gops.prune_stale_merge_worktrees = AsyncMock(
            return_value=pruned if pruned is not None else [],
        )
        return gops

    async def test_sufficient_space_returns_none_without_pruning(
        self, tmp_path: Path,
    ):
        gops = self._git_ops()
        with patch(
            'orchestrator.merge_queue.shutil.disk_usage',
            return_value=_usage(20 * _GIB),
        ):
            reason = await _ensure_verify_disk_space(
                gops, tmp_path, 10 * _GIB, 't1',
            )
        assert reason is None
        gops.prune_stale_merge_worktrees.assert_not_called()

    async def test_low_then_prune_frees_enough_returns_none(
        self, tmp_path: Path,
    ):
        gops = self._git_ops(pruned=['/x/_merge-a'])
        with patch(
            'orchestrator.merge_queue.shutil.disk_usage',
            side_effect=[_usage(2 * _GIB), _usage(15 * _GIB)],
        ):
            reason = await _ensure_verify_disk_space(
                gops, tmp_path, 10 * _GIB, 't1',
            )
        assert reason is None
        gops.prune_stale_merge_worktrees.assert_awaited_once()

    async def test_persistent_low_returns_transient_infra_reason(
        self, tmp_path: Path,
    ):
        gops = self._git_ops()
        with patch(
            'orchestrator.merge_queue.shutil.disk_usage',
            side_effect=[_usage(1 * _GIB), _usage(1 * _GIB)],
        ):
            reason = await _ensure_verify_disk_space(
                gops, tmp_path, 10 * _GIB, 't1',
            )
        assert reason is not None
        assert reason.startswith(TRANSIENT_INFRA_REASON_PREFIX)
        gops.prune_stale_merge_worktrees.assert_awaited_once()

    async def test_oserror_on_first_stat_fails_open(self, tmp_path: Path):
        gops = self._git_ops()
        with patch(
            'orchestrator.merge_queue.shutil.disk_usage',
            side_effect=OSError('stat failed'),
        ):
            reason = await _ensure_verify_disk_space(
                gops, tmp_path, 10 * _GIB, 't1',
            )
        assert reason is None
        gops.prune_stale_merge_worktrees.assert_not_called()

    async def test_oserror_on_post_prune_stat_fails_open(self, tmp_path: Path):
        gops = self._git_ops()
        with patch(
            'orchestrator.merge_queue.shutil.disk_usage',
            side_effect=[_usage(1 * _GIB), OSError('stat failed')],
        ):
            reason = await _ensure_verify_disk_space(
                gops, tmp_path, 10 * _GIB, 't1',
            )
        # Pruned once (free was low), but the re-check stat failed → fail open.
        assert reason is None
        gops.prune_stale_merge_worktrees.assert_awaited_once()


@pytest.mark.asyncio
class TestPreVerifyDiskGuardWiring:
    """The guard is wired before the first verify in both merge workers."""

    async def test_merge_worker_proceeds_when_space_sufficient(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        wt = await _make_branch_with_file(
            git_ops, 'disk-ok', 'ok.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        mock_verify = _mock_verify_pass()
        with (
            patch('orchestrator.merge_queue.run_scoped_verification', mock_verify),
            patch(
                'orchestrator.merge_queue.shutil.disk_usage',
                return_value=_usage(50 * _GIB),
            ),
            patch.object(
                git_ops, 'prune_stale_merge_worktrees',
                AsyncMock(return_value=[]),
            ) as mock_prune,
        ):
            req = _make_request('disk-ok', 'disk-ok', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done', f'unexpected: {outcome}'
        assert mock_verify.call_count == 1
        mock_prune.assert_not_called()

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

    async def test_merge_worker_proceeds_when_prune_frees_enough(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        wt = await _make_branch_with_file(
            git_ops, 'disk-heals', 'heals.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        mock_verify = _mock_verify_pass()
        with (
            patch('orchestrator.merge_queue.run_scoped_verification', mock_verify),
            patch(
                'orchestrator.merge_queue.shutil.disk_usage',
                side_effect=[_usage(2 * _GIB), _usage(50 * _GIB)],
            ),
            patch.object(
                git_ops, 'prune_stale_merge_worktrees',
                AsyncMock(return_value=['/x/_merge-stale']),
            ) as mock_prune,
        ):
            req = _make_request('disk-heals', 'disk-heals', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done', f'unexpected: {outcome}'
        assert mock_verify.call_count == 1
        assert mock_prune.call_count == 1

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

    async def test_merge_worker_fails_open_on_disk_usage_oserror(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        wt = await _make_branch_with_file(
            git_ops, 'disk-stat-boom', 'boom.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        mock_verify = _mock_verify_pass()
        with (
            patch('orchestrator.merge_queue.run_scoped_verification', mock_verify),
            patch(
                'orchestrator.merge_queue.shutil.disk_usage',
                side_effect=OSError('stat boom'),
            ),
            patch.object(
                git_ops, 'prune_stale_merge_worktrees',
                AsyncMock(return_value=[]),
            ) as mock_prune,
        ):
            req = _make_request('disk-stat-boom', 'disk-stat-boom', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'done', f'unexpected: {outcome}'
        assert mock_verify.call_count == 1
        mock_prune.assert_not_called()

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

    async def test_merge_worker_short_circuits_on_persistent_low_disk(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        wt = await _make_branch_with_file(
            git_ops, 'disk-low', 'low.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        mock_verify = _mock_verify_pass()
        with (
            patch('orchestrator.merge_queue.run_scoped_verification', mock_verify),
            patch(
                'orchestrator.merge_queue.shutil.disk_usage',
                return_value=_usage(1 * _GIB),
            ),
            patch.object(
                git_ops, 'prune_stale_merge_worktrees',
                AsyncMock(return_value=[]),
            ) as mock_prune,
        ):
            req = _make_request('disk-low', 'disk-low', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked'
        assert outcome.reason.startswith(TRANSIENT_INFRA_REASON_PREFIX), (
            f'expected transient-infra reason, got: {outcome.reason!r}'
        )
        # Build must NOT run when the guard short-circuits.
        mock_verify.assert_not_called()
        assert mock_prune.call_count == 1
        # Main must not have advanced.
        _, main_files, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'low.py' not in main_files

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

    async def test_speculative_worker_short_circuits_on_persistent_low_disk(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        wt = await _make_branch_with_file(
            git_ops, 'spec-disk-low', 'low.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        mock_verify = _mock_verify_pass()
        with (
            patch('orchestrator.merge_queue.run_scoped_verification', mock_verify),
            patch(
                'orchestrator.merge_queue.shutil.disk_usage',
                return_value=_usage(1 * _GIB),
            ),
            patch.object(
                git_ops, 'prune_stale_merge_worktrees',
                AsyncMock(return_value=[]),
            ) as mock_prune,
        ):
            req = _make_request('spec-disk-low', 'spec-disk-low', wt, config)
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked'
        assert outcome.reason.startswith(TRANSIENT_INFRA_REASON_PREFIX), (
            f'expected transient-infra reason, got: {outcome.reason!r}'
        )
        mock_verify.assert_not_called()
        assert mock_prune.call_count == 1

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task


# ---------------------------------------------------------------------------
# TestGroupMergeRequestDataclass — step-1 structural contract
# ---------------------------------------------------------------------------


class TestGroupMergeRequestDataclass:
    """Structural contract tests for GroupMergeRequest."""

    def _make_instance(self, config: OrchestratorConfig, tmp_path: Path) -> GroupMergeRequest:
        """Build a minimal GroupMergeRequest for introspection."""
        # Use MagicMock instead of a real asyncio.Future to avoid creating (and
        # leaking) an event loop just to produce a placeholder field value.
        # These dataclass tests do not exercise async behaviour.
        future: asyncio.Future[MergeOutcome] = MagicMock(spec=asyncio.Future)
        status_check_mock = AsyncMock(return_value={})
        mark_done_mock = AsyncMock()
        return GroupMergeRequest(
            task_id='tip-task',
            branch='tip-branch',
            worktree=tmp_path,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=future,
            train_id='train-42',
            member_task_ids=['task-a', 'task-b', 'task-c'],
            tip_branch='tip-branch',
            tip_task_id='tip-task',
            status_check=status_check_mock,
            mark_member_done=mark_done_mock,
        )

    def test_is_subclass_of_merge_request(self):
        assert issubclass(GroupMergeRequest, MergeRequest)

    def test_instance_isinstance_merge_request(
        self, config: OrchestratorConfig, tmp_path: Path,
    ):
        req = self._make_instance(config, tmp_path)
        assert isinstance(req, MergeRequest)
        assert isinstance(req, GroupMergeRequest)

    def test_base_fields_accessible(
        self, config: OrchestratorConfig, tmp_path: Path,
    ):
        req = self._make_instance(config, tmp_path)
        assert req.task_id == 'tip-task'
        assert req.branch == 'tip-branch'
        assert req.worktree == tmp_path
        assert req.pre_rebased is False
        assert req.task_files is None
        assert req.module_configs == []
        assert req.config is config

    def test_train_fields(
        self, config: OrchestratorConfig, tmp_path: Path,
    ):
        req = self._make_instance(config, tmp_path)
        assert req.train_id == 'train-42'
        assert req.member_task_ids == ['task-a', 'task-b', 'task-c']
        assert req.tip_branch == 'tip-branch'
        assert req.tip_task_id == 'tip-task'

    def test_callback_fields_callable(
        self, config: OrchestratorConfig, tmp_path: Path,
    ):
        req = self._make_instance(config, tmp_path)
        assert callable(req.status_check)
        assert callable(req.mark_member_done)


# ---------------------------------------------------------------------------
# Train helpers (shared by all TestGroupMergeRequest* classes)
# ---------------------------------------------------------------------------


async def _setup_repo_with_config(repo: Path) -> None:
    """Set git identity in a repo (required for commits)."""
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)


async def _make_stacked_train(
    git_ops: GitOps,
    config: OrchestratorConfig,
    train_id: str = 'train-test',
    member_names: tuple[str, str, str] = ('trn-a', 'trn-b', 'trn-c'),
) -> GroupMergeRequest:
    """Build a 3-member stacked train in the tmp repo.

    Creates branch A off main, B stacked on A, C (tip) stacked on B.
    Each branch adds its own unique file (a.py, b.py, c.py).

    Returns a GroupMergeRequest with:
    - status_check: AsyncMock returning all 'merge-deferred'
    - mark_member_done: AsyncMock recording (task_id, sha) calls
    """
    a_name, b_name, c_name = member_names

    # Branch A: from main
    wt_a = (await git_ops.create_worktree(a_name)).path
    (wt_a / f'{a_name}.py').write_text(f'{a_name} = 1\n')
    await git_ops.commit(wt_a, f'Add {a_name}.py')
    rc, a_head, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt_a)
    a_head = a_head.strip()

    # Branch B: stacked on A's HEAD
    b_full = f'task/{b_name}'
    wt_b_path = git_ops.worktree_base / b_name
    wt_b_path.parent.mkdir(parents=True, exist_ok=True)
    await _run(
        ['git', 'worktree', 'add', '-b', b_full, str(wt_b_path), a_head],
        cwd=git_ops.project_root,
    )
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=wt_b_path)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=wt_b_path)
    (wt_b_path / f'{b_name}.py').write_text(f'{b_name} = 2\n')
    await git_ops.commit(wt_b_path, f'Add {b_name}.py')
    rc, b_head, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt_b_path)
    b_head = b_head.strip()

    # Branch C (tip): stacked on B's HEAD
    c_full = f'task/{c_name}'
    wt_c_path = git_ops.worktree_base / c_name
    wt_c_path.parent.mkdir(parents=True, exist_ok=True)
    await _run(
        ['git', 'worktree', 'add', '-b', c_full, str(wt_c_path), b_head],
        cwd=git_ops.project_root,
    )
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=wt_c_path)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=wt_c_path)
    (wt_c_path / f'{c_name}.py').write_text(f'{c_name} = 3\n')
    await git_ops.commit(wt_c_path, f'Add {c_name}.py')

    # Callbacks
    status_check = AsyncMock(return_value={
        a_name: 'merge-deferred',
        b_name: 'merge-deferred',
        c_name: 'merge-deferred',
    })
    mark_member_done = AsyncMock()

    loop = asyncio.get_event_loop()
    future: asyncio.Future[MergeOutcome] = loop.create_future()

    return GroupMergeRequest(
        task_id=c_name,
        branch=c_name,
        worktree=wt_c_path,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
        train_id=train_id,
        member_task_ids=[a_name, b_name, c_name],
        tip_branch=c_name,
        tip_task_id=c_name,
        status_check=status_check,
        mark_member_done=mark_member_done,
    )


# ---------------------------------------------------------------------------
# TestGroupMergeRequestHappyPath (MergeWorker) — step-3 RED / step-4 GREEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGroupMergeRequestHappyPath:
    """PRD scenario 1: 3-train merges atomically via MergeWorker."""

    async def test_single_merge_commit_all_members_done(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Happy path: 3-member train → 1 merge commit, 3 callbacks, same SHA."""
        req = await _make_stacked_train(git_ops, config)

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)

        # Count merge commits on main before the train lands
        _, before_log, _ = await _run(
            ['git', 'rev-list', '--merges', '--count', 'main'],
            cwd=git_ops.project_root,
        )
        merge_commits_before = int(before_log.strip())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            outcome = await worker._do_merge(req)

        assert outcome is not None
        assert outcome.status == 'done', f'expected done, got: {outcome!r}'
        assert outcome.merge_sha is not None

        # Exactly one new merge commit added to main
        _, after_log, _ = await _run(
            ['git', 'rev-list', '--merges', '--count', 'main'],
            cwd=git_ops.project_root,
        )
        merge_commits_after = int(after_log.strip())
        assert merge_commits_after == merge_commits_before + 1, (
            'expected exactly 1 new merge commit on main'
        )

        # All three member files present on main
        _, main_files, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'trn-a.py' in main_files
        assert 'trn-b.py' in main_files
        assert 'trn-c.py' in main_files

        # mark_member_done called exactly 3 times, all with the SAME merge SHA
        assert req.mark_member_done.call_count == 3, (  # type: ignore[reportFunctionMemberAccess]
            f'expected 3 mark_member_done calls, got {req.mark_member_done.call_count}'  # type: ignore[reportFunctionMemberAccess]
        )
        called_shas = {call.args[1] for call in req.mark_member_done.call_args_list}  # type: ignore[reportFunctionMemberAccess]
        assert len(called_shas) == 1, f'all callbacks must share one SHA, got: {called_shas}'
        called_sha = next(iter(called_shas))
        assert called_sha == outcome.merge_sha


# ---------------------------------------------------------------------------
# TestGroupMergeRequestTrainIncomplete (MergeWorker) — step-5 RED / step-6 GREEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGroupMergeRequestTrainIncomplete:
    """PRD spec §9.6 step 1: non-merge-deferred member → immediate block."""

    async def test_incomplete_member_blocks_before_git(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """One member not merge-deferred → blocked with TRAIN_INCOMPLETE prefix; no git work."""
        req = await _make_stacked_train(git_ops, config)
        # Override status_check to return one non-deferred member
        req.status_check = AsyncMock(return_value={
            'trn-a': 'merge-deferred',
            'trn-b': 'in-progress',  # not deferred
            'trn-c': 'merge-deferred',
        })

        # Capture main SHA before calling _do_merge
        _, main_before, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        main_before = main_before.strip()

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)

        # Spy on merge_to_main: should NOT be called
        with patch.object(git_ops, 'merge_to_main', wraps=git_ops.merge_to_main) as spy_merge:
            outcome = await worker._do_merge(req)

        assert outcome is not None
        assert outcome.status == 'blocked', f'expected blocked, got: {outcome!r}'
        assert outcome.reason.startswith(TRAIN_INCOMPLETE_REASON_PREFIX), (
            f'expected TRAIN_INCOMPLETE prefix, got: {outcome.reason!r}'
        )
        # Naming the offending member in the reason
        assert 'trn-b' in outcome.reason, (
            f'expected offending member in reason, got: {outcome.reason!r}'
        )
        assert 'in-progress' in outcome.reason, (
            f'expected offending status in reason, got: {outcome.reason!r}'
        )

        # Main unchanged (no new commits)
        _, main_after, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        assert main_after.strip() == main_before, 'main must not advance on incomplete train'

        # No git merge work done
        spy_merge.assert_not_called()

        # No member callbacks
        req.mark_member_done.assert_not_called()  # type: ignore[reportFunctionMemberAccess]


# ---------------------------------------------------------------------------
# TestGroupMergeRequestRebaseConflict (MergeWorker) — step-7 RED / step-8 GREEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGroupMergeRequestRebaseConflict:
    """PRD scenario 8 conflict variant: tip conflicts with advanced main."""

    async def test_rebase_conflict_blocks_without_merge(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Conflicting main advance → TRAIN_REBASE_CONFLICT, no merge commit, no callbacks."""
        req = await _make_stacked_train(git_ops, config)

        # Commit a conflicting change on main AFTER the train was built.
        # The tip branch (trn-c) touched trn-c.py; we clobber the same file
        # on main so the rebase of the tip will conflict.
        conflict_file = git_ops.project_root / 'trn-c.py'
        conflict_file.write_text('# main version — conflicts with tip\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Conflicting commit on main'], cwd=git_ops.project_root)

        # Record main SHA (includes the conflicting commit, tip is no longer a descendant)
        _, main_sha_after, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        main_sha_after = main_sha_after.strip()

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)

        with patch.object(git_ops, 'merge_to_main', wraps=git_ops.merge_to_main) as spy_merge:
            outcome = await worker._do_merge(req)

        assert outcome is not None
        assert outcome.status == 'blocked', f'expected blocked, got: {outcome!r}'
        assert outcome.reason.startswith(TRAIN_REBASE_CONFLICT_REASON_PREFIX), (
            f'expected TRAIN_REBASE_CONFLICT prefix, got: {outcome.reason!r}'
        )

        # Main must not have advanced beyond the conflicting commit
        _, main_after, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        assert main_after.strip() == main_sha_after, (
            'main must not advance when rebase conflicts'
        )

        # No merge work done
        spy_merge.assert_not_called()

        # No callbacks
        req.mark_member_done.assert_not_called()  # type: ignore[reportFunctionMemberAccess]

        # Tip worktree must be clean (rebase was aborted)
        _, status_out, _ = await _run(
            ['git', 'status', '--porcelain'], cwd=req.worktree,
        )
        assert status_out.strip() == '', f'tip worktree not clean: {status_out!r}'
        # Must not be mid-rebase
        assert not (req.worktree / '.git').exists() or not (req.worktree / '.git' / 'rebase-merge').exists(), (
            'rebase-merge dir exists — rebase was not aborted'
        )


# ---------------------------------------------------------------------------
# TestGroupMergeRequestMainAdvancedClean (MergeWorker) — step-9 RED / step-10 GREEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGroupMergeRequestMainAdvancedClean:
    """PRD scenario 8 clean variant: non-conflicting main advance; rebase + merge atomic."""

    async def test_clean_advance_includes_external_commit(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Non-conflicting external commit on main → rebase succeeds, train lands, all files present."""
        req = await _make_stacked_train(git_ops, config)

        # Commit an EXTERNAL (non-conflicting) change on main AFTER the train
        # was built (different file so no rebase conflict on the tip).
        external_file = git_ops.project_root / 'external.py'
        external_file.write_text('external = True\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'External non-conflicting commit'], cwd=git_ops.project_root)
        _, external_sha_out, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        external_sha = external_sha_out.strip()

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            outcome = await worker._do_merge(req)

        assert outcome is not None
        assert outcome.status == 'done', f'expected done, got: {outcome!r}'
        assert outcome.merge_sha is not None

        # Main must contain external commit AND all train member files
        _, main_files, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'external.py' in main_files
        assert 'trn-a.py' in main_files
        assert 'trn-b.py' in main_files
        assert 'trn-c.py' in main_files

        # The advanced SHA must be a descendant of the external commit
        rc, _, _ = await _run(
            ['git', 'merge-base', '--is-ancestor', external_sha, outcome.merge_sha],
            cwd=git_ops.project_root,
        )
        assert rc == 0, (
            f'merge_sha {outcome.merge_sha[:8]} is not a descendant of external commit {external_sha[:8]}'
        )

        # All members marked done with same SHA
        assert req.mark_member_done.call_count == 3  # type: ignore[reportFunctionMemberAccess]
        called_shas = {call.args[1] for call in req.mark_member_done.call_args_list}  # type: ignore[reportFunctionMemberAccess]
        assert len(called_shas) == 1
        assert next(iter(called_shas)) == outcome.merge_sha


# ---------------------------------------------------------------------------
# TestGroupMergeRequestVerifyGate (MergeWorker) — step-11 RED / step-12 GREEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGroupMergeRequestVerifyGate:
    """PRD scenario 5 red gate: verify failure → no advance, no callbacks."""

    async def test_red_verify_blocks_advance_and_callbacks(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Post-merge verify fails → blocked, advance_main not called, callbacks silent."""
        req = await _make_stacked_train(git_ops, config)

        # Record main SHA before the attempt
        _, main_before, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        main_before = main_before.strip()

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)

        # Patch verify to FAIL
        mock_verify_fail = AsyncMock(return_value=MagicMock(
            passed=False,
            summary='Tests failed: 3 errors',
            failure_report=MagicMock(return_value='Tests failed: 3 errors'),
        ))
        # Spy on advance_main to assert it's never called
        with (
            patch('orchestrator.merge_queue.run_scoped_verification', mock_verify_fail),
            patch.object(git_ops, 'advance_main', wraps=git_ops.advance_main) as spy_advance,
        ):
            outcome = await worker._do_merge(req)

        assert outcome is not None
        assert outcome.status == 'blocked', f'expected blocked, got: {outcome!r}'
        assert 'Post-merge verification failed' in outcome.reason, (
            f'expected verify-gate reason, got: {outcome.reason!r}'
        )

        # advance_main must NOT have been called
        spy_advance.assert_not_called()

        # Main must be unchanged
        _, main_after, _ = await _run(
            ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
        )
        assert main_after.strip() == main_before, 'main must not advance on red verify'

        # No member callbacks
        req.mark_member_done.assert_not_called()  # type: ignore[reportFunctionMemberAccess]


# ---------------------------------------------------------------------------
# TestGroupMergeRequestSpeculativeWorker — step-13 RED / step-14 GREEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGroupMergeRequestSpeculativeWorker:
    """GroupMergeRequest through SpeculativeMergeWorker (the harness's default worker)."""

    async def test_train_lands_via_speculative_worker(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """Happy-path 3-train via SpeculativeMergeWorker: done, 1 merge commit, 3 callbacks.

        Drives the same scenario as TestGroupMergeRequestHappyPath but through
        SpeculativeMergeWorker — the worker the harness actually instantiates.

        Fails before step-14: _merger_loop falls through to the regular single-task
        path, which does NOT call mark_member_done (no GroupMergeRequest dispatch).
        """
        req = await _make_stacked_train(git_ops, config)

        # Count merge commits on main before the train lands
        _, before_log, _ = await _run(
            ['git', 'rev-list', '--merges', '--count', 'main'],
            cwd=git_ops.project_root,
        )
        merge_commits_before = int(before_log.strip())

        db_path = tmp_path / 'events_spec_train.db'
        event_store = EventStore(db_path=db_path, run_id='test-spec-train')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=event_store)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=60)

        assert outcome.status == 'done', f'expected done, got: {outcome!r}'
        assert outcome.merge_sha is not None

        # Exactly one new merge commit added to main
        _, after_log, _ = await _run(
            ['git', 'rev-list', '--merges', '--count', 'main'],
            cwd=git_ops.project_root,
        )
        merge_commits_after = int(after_log.strip())
        assert merge_commits_after == merge_commits_before + 1, (
            'expected exactly 1 new merge commit on main'
        )

        # All three member files present on main
        _, main_files, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'trn-a.py' in main_files
        assert 'trn-b.py' in main_files
        assert 'trn-c.py' in main_files

        # mark_member_done called exactly 3 times, all with the SAME merge SHA
        assert req.mark_member_done.call_count == 3, (  # type: ignore[reportFunctionMemberAccess]
            f'expected 3 mark_member_done calls, got {req.mark_member_done.call_count}'  # type: ignore[reportFunctionMemberAccess]
        )
        called_shas = {call.args[1] for call in req.mark_member_done.call_args_list}  # type: ignore[reportFunctionMemberAccess]
        assert len(called_shas) == 1, f'all callbacks must share one SHA, got: {called_shas}'
        assert next(iter(called_shas)) == outcome.merge_sha

        # No speculative_merge event emitted for the train's task_id —
        # trains bypass the speculative look-ahead path entirely.
        conn = sqlite3.connect(str(db_path))
        spec_rows = conn.execute(
            "SELECT event_type, task_id FROM events "
            "WHERE event_type = 'speculative_merge' AND task_id = ?",
            (req.task_id,),
        ).fetchall()
        conn.close()
        assert len(spec_rows) == 0, (
            f'speculative_merge events must NOT be emitted for GroupMergeRequest; '
            f'got: {spec_rows}'
        )

        # merge_attempt event for the train carries train_id and member_task_ids
        # so downstream reconciliation can correlate rows with the specific train.
        conn2 = sqlite3.connect(str(db_path))
        import json as _json
        done_rows = conn2.execute(
            "SELECT data FROM events "
            "WHERE event_type = 'merge_attempt' AND task_id = ?",
            (req.task_id,),
        ).fetchall()
        conn2.close()
        assert done_rows, 'expected at least one merge_attempt event for the train'
        done_payloads = [_json.loads(row[0]) for row in done_rows]
        # The final 'done' event (or any train event) should carry train_id
        train_events = [p for p in done_payloads if p.get('outcome') == 'done']
        assert train_events, f'expected a done merge_attempt event; got: {done_payloads}'
        assert train_events[-1].get('train_id') == req.train_id, (
            f'done event missing train_id; payload: {train_events[-1]}'
        )
        assert train_events[-1].get('member_task_ids') == req.member_task_ids, (
            f'done event missing member_task_ids; payload: {train_events[-1]}'
        )

        await worker.stop()
        await worker_task

    async def test_regular_then_train_both_land(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """A regular MergeRequest followed by a GroupMergeRequest both land cleanly.

        Verifies the ordering contract: regular → train → both on main, no
        split-brain.  Also documents the pipeline-ordering concern (suggestion 1):
        the train may dequeue while the regular request's CAS is still pending;
        advance_main retries absorb the race.
        """
        # Build the stacked train (3 members)
        train_req = await _make_stacked_train(git_ops, config)

        # Build a separate single-task request using git_ops.create_worktree
        # (which creates branch task/<name>, matching how all other tests work).
        reg_task_id = 'reg-task'
        reg_file = 'reg-task.py'
        reg_wt_info = await git_ops.create_worktree(reg_task_id)
        reg_wt = reg_wt_info.path
        (reg_wt / reg_file).write_text('# reg\n')
        await git_ops.commit(reg_wt, f'add {reg_file}')
        reg_req = _make_request(reg_task_id, reg_task_id, reg_wt, config)

        db_path = tmp_path / 'events_reg_train.db'
        event_store = EventStore(db_path=db_path, run_id='test-reg-then-train')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=event_store)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            await queue.put(reg_req)
            reg_outcome = await asyncio.wait_for(reg_req.result, timeout=60)
            # Now enqueue the train (after regular has resolved, pipeline is idle)
            await queue.put(train_req)
            train_outcome = await asyncio.wait_for(train_req.result, timeout=60)

        assert reg_outcome.status == 'done', f'regular request failed: {reg_outcome!r}'
        assert train_outcome.status == 'done', f'train request failed: {train_outcome!r}'

        # Both the regular file and all train member files must be on main
        _, main_files, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert reg_file in main_files, f'{reg_file} missing from main'
        assert 'trn-a.py' in main_files
        assert 'trn-b.py' in main_files
        assert 'trn-c.py' in main_files

        # mark_member_done called 3 times for the train
        assert train_req.mark_member_done.call_count == 3  # type: ignore[reportFunctionMemberAccess]

        await worker.stop()
        await worker_task


# ---------------------------------------------------------------------------
# TestGroupMergeRequestPartialMemberFlipFailure (MergeWorker) — step-15 RED / step-16 GREEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGroupMergeRequestPartialMemberFlipFailure:
    """Guards atomicity invariant when a per-member mark_member_done callback raises
    AFTER main has advanced.  The worker must NOT produce a split-brain 'blocked'
    outcome and must attempt all callbacks even when one fails."""

    async def test_partial_flip_failure_after_advance(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """When one mark_member_done callback raises, worker continues all flips,
        returns 'done' with TRAIN_PARTIAL_FLIP_REASON_PREFIX reason, and main HAS advanced."""
        req = await _make_stacked_train(git_ops, config)

        # Override mark_member_done: 1st OK, 2nd raises, 3rd OK
        req.mark_member_done = AsyncMock(side_effect=[  # type: ignore[reportFunctionMemberAccess]
            None,                                    # trn-a: success
            RuntimeError('scheduler network blip'),  # trn-b: fails
            None,                                    # trn-c: success
        ])

        # Count merge commits before
        _, before_log, _ = await _run(
            ['git', 'rev-list', '--merges', '--count', 'main'],
            cwd=git_ops.project_root,
        )
        merge_commits_before = int(before_log.strip())

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            outcome = await worker._do_merge(req)

        # (a) _do_merge returns normally — no exception propagated
        assert outcome is not None

        # (b) main HAS advanced: one new merge commit, all 3 member files present
        _, after_log, _ = await _run(
            ['git', 'rev-list', '--merges', '--count', 'main'],
            cwd=git_ops.project_root,
        )
        merge_commits_after = int(after_log.strip())
        assert merge_commits_after == merge_commits_before + 1, (
            'expected exactly 1 new merge commit on main even when a callback fails'
        )

        _, main_files, _ = await _run(
            ['git', 'ls-tree', '-r', '--name-only', 'main'],
            cwd=git_ops.project_root,
        )
        assert 'trn-a.py' in main_files
        assert 'trn-b.py' in main_files
        assert 'trn-c.py' in main_files

        # (c) mark_member_done called 3 times (NOT aborted after second failure)
        assert req.mark_member_done.call_count == 3, (  # type: ignore[reportFunctionMemberAccess]
            f'expected 3 mark_member_done calls (loop must not abort on failure), '
            f'got {req.mark_member_done.call_count}'  # type: ignore[reportFunctionMemberAccess]
        )
        # All with the same advanced_sha
        called_shas = {call.args[1] for call in req.mark_member_done.call_args_list}  # type: ignore[reportFunctionMemberAccess]
        assert len(called_shas) == 1, f'all callbacks must share one SHA, got: {called_shas}'

        # (d) outcome.status is 'done' (not 'blocked') — main landed
        assert outcome.status == 'done', (
            f'expected done (main advanced), got: {outcome!r}'
        )
        assert outcome.merge_sha is not None
        assert outcome.merge_sha == next(iter(called_shas))

        # (e) reason starts with TRAIN_PARTIAL_FLIP_REASON_PREFIX and names
        #     the failed count and offending member
        assert outcome.reason is not None, 'partial-flip outcome must carry a reason'
        assert outcome.reason.startswith(TRAIN_PARTIAL_FLIP_REASON_PREFIX), (
            f'expected TRAIN_PARTIAL_FLIP prefix, got: {outcome.reason!r}'
        )
        assert '1/3' in outcome.reason, (
            f'expected failed-count ratio (1/3) in reason, got: {outcome.reason!r}'
        )
        assert 'trn-b' in outcome.reason, (
            f'expected offending member task_id (trn-b) in reason, got: {outcome.reason!r}'
        )


# ---------------------------------------------------------------------------
# TestGroupMergeRequestTrainVerifyRole — train-merge callsite role injection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGroupMergeRequestTrainVerifyRole:
    """_do_train_merge must pass role='merge' to run_scoped_verification.

    _do_train_merge (merge_queue.py:717) is the shared train-merge pipeline
    reached from both the deprecated MergeWorker._do_merge and the active
    SpeculativeMergeWorker.  Its post-merge verify (line ~811) sets
    is_merge_verify=True; under the invariant "every merge-queue verify
    carries role='merge'" it must also pass role='merge' so reify's
    verify.sh applies the merge-role priority prefix (nice -n 5).
    """

    async def test_train_merge_verify_called_with_role_merge(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Train-merge post-merge verify must pass role='merge'.

        Drive a GroupMergeRequest through MergeWorker._do_merge (which
        dispatches to _do_train_merge) and capture the kwargs passed to
        run_scoped_verification.  Assert role='merge' is present.
        RED: the run_scoped_verification call at merge_queue.py:811 inside
        _do_train_merge does not pass role='merge' yet, so the captured role
        is absent (None).
        """
        req = await _make_stacked_train(git_ops, config)

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)

        captured_kwargs: list[dict] = []

        async def spy_verify(*args, **kwargs):
            captured_kwargs.append(kwargs)
            result = AsyncMock()
            result.passed = True
            result.summary = ''
            return result

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=spy_verify,
        ):
            outcome = await worker._do_merge(req)

        assert outcome is not None
        assert outcome.status == 'done', f'expected done, got: {outcome!r}'
        assert captured_kwargs, 'run_scoped_verification was not invoked'
        assert captured_kwargs[0].get('role') == 'merge', (
            f"train-merge verify must pass role='merge'; got {captured_kwargs[0]!r}"
        )


# ---------------------------------------------------------------------------
# TestMergeFailureDiagnostic — task 1539
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeFailureDiagnostic:
    """Merge-failure diagnostic enrichment: base SHA, label, ref-resolution, git stderr."""

    async def test_remerge_ghost_branch_sets_failure_diagnostic(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """_remerge on a non-existent branch populates failure_diagnostic on the outcome.

        The 'task/ghost' ref does not exist → git fatal + branch_ref_in_worktree=<unresolved>.
        This is the smoking-gun for the 2026-05-28 'not something we can merge' incident.
        """
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        req = _make_request('ghost', 'ghost', git_ops.project_root, config)
        item = await worker._remerge(req, None)

        outcome = item.immediate_outcome
        assert outcome is not None
        assert outcome.status == 'blocked'

        # failure_diagnostic must be populated with all four keys
        assert outcome.failure_diagnostic is not None, (
            'failure_diagnostic must be set on blocked outcome from _remerge'
        )
        diag = outcome.failure_diagnostic
        assert len(diag['base_sha']) == 40, f'base_sha must be 40-char hex: {diag["base_sha"]!r}'
        assert all(c in '0123456789abcdef' for c in diag['base_sha'])
        assert diag['base_label'] == 'main_head'
        assert diag['branch_ref_in_worktree'] == '<unresolved>'
        assert isinstance(diag['git_stderr'], str) and diag['git_stderr'], (
            'git_stderr must be a non-empty string'
        )

        # item.failure_diagnostic must mirror outcome.failure_diagnostic
        assert item.failure_diagnostic == diag

        # rendered labels must appear in reason for backward compat
        assert 'base_sha=' in outcome.reason
        assert 'base_label=main_head' in outcome.reason
        assert 'branch_ref_in_worktree=<unresolved>' in outcome.reason

    async def test_merger_loop_ghost_branch_sets_failure_diagnostic(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Non-existent branch through the Merger loop (direct merge_request path) sets failure_diagnostic.

        The direct merge_request path flows through _merger_loop, not _remerge.
        Uses a real worktree (non-main HEAD) so the already-merged check passes,
        but the branch ref 'task/ghost-m' does not exist → merge fails.
        """
        # Create a real branch so the worktree HEAD is NOT an ancestor of main
        # (which would short-circuit to already_merged before the merge attempt).
        # The request branch name 'ghost-m' has no refs/heads/task/ghost-m ref.
        wt = await _make_branch_with_file(git_ops, 'phantom-1', 'ph1.py', 'x = 1\n')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        req = _make_request('ghost-m', 'ghost-m', wt, config)
        await queue.put(req)
        outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'blocked'
        assert outcome.failure_diagnostic is not None, (
            'failure_diagnostic must be set on blocked outcome from Merger loop'
        )
        diag = outcome.failure_diagnostic
        assert len(diag['base_sha']) == 40
        assert all(c in '0123456789abcdef' for c in diag['base_sha'])
        assert diag['base_label'] == 'main_head'
        assert diag['branch_ref_in_worktree'] == '<unresolved>'
        assert isinstance(diag['git_stderr'], str) and diag['git_stderr']
        assert 'base_sha=' in outcome.reason
        assert 'base_label=main_head' in outcome.reason
        assert 'branch_ref_in_worktree=<unresolved>' in outcome.reason

        await worker.stop()
        await worker_task

    async def test_merger_loop_speculative_ghost_sets_base_label_speculative(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """N+1 ghost branch speculatively merged gets failure_diagnostic['base_label']=='speculative'."""
        wt_n = await _make_branch_with_file(
            git_ops, 'spec-diag-n', 'diag_n.py', 'n = 1\n',
        )
        # N+1: real worktree (non-main HEAD) but ref 'task/ghost-spec' does not exist
        wt_n1 = await _make_branch_with_file(
            git_ops, 'phantom-n1', 'ph_n1.py', 'n1 = 2\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            req_n = _make_request('spec-diag-n', 'spec-diag-n', wt_n, config)
            req_n1 = _make_request('ghost-spec', 'ghost-spec', wt_n1, config)

            await queue.put(req_n)
            await queue.put(req_n1)

            outcome_n = await asyncio.wait_for(req_n.result, timeout=30)
            outcome_n1 = await asyncio.wait_for(req_n1.result, timeout=30)

        assert outcome_n.status == 'done', f'N should succeed: {outcome_n}'
        assert outcome_n1.status == 'blocked', f'N+1 ghost should be blocked: {outcome_n1}'
        assert outcome_n1.failure_diagnostic is not None
        # N+1 was speculatively merged against N's merge commit → base_label='speculative'
        assert outcome_n1.failure_diagnostic['base_label'] == 'speculative', (
            f"expected 'speculative', got {outcome_n1.failure_diagnostic['base_label']!r}"
        )
        assert outcome_n1.failure_diagnostic['branch_ref_in_worktree'] == '<unresolved>'

        await worker.stop()
        await worker_task

    async def test_escalation_server_merge_request_failure_diagnostic(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """merge_request tool response includes failure_diagnostic on non-conflict failure.

        Successful merges must NOT include failure_diagnostic (byte-identical response shape).
        """
        from escalation.queue import EscalationQueue
        from escalation.server import create_server

        esc_queue = EscalationQueue(tmp_path / 'esc')
        merge_q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, merge_q)
        worker_task = asyncio.create_task(worker.run())

        server = create_server(esc_queue, merge_queue=merge_q, orch_config=config)
        from fastmcp.tools.function_tool import FunctionTool
        tool = await server.get_tool('merge_request')
        assert isinstance(tool, FunctionTool)

        # Ghost branch: use a real worktree (non-main HEAD) but branch ref doesn't exist
        wt_ghost = await _make_branch_with_file(git_ops, 'e2e-phantom', 'phantom.py', 'x = 1\n')

        # Ghost branch → failure_diagnostic must appear in response
        resp = await asyncio.wait_for(
            tool.fn(task_id='ghost-e2e', branch='ghost-e2e', worktree=str(wt_ghost)),
            timeout=30,
        )
        assert resp['status'] == 'blocked'
        assert 'failure_diagnostic' in resp, (
            f"'failure_diagnostic' missing from blocked response: {resp}"
        )
        diag = resp['failure_diagnostic']
        assert diag['base_label'] == 'main_head'
        assert diag['branch_ref_in_worktree'] == '<unresolved>'
        assert len(diag['base_sha']) == 40

        # Valid branch → NO failure_diagnostic in successful response
        wt = await _make_branch_with_file(git_ops, 'e2e-valid', 'e2e.py', 'x = 1\n')
        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            resp_ok = await asyncio.wait_for(
                tool.fn(task_id='e2e-valid', branch='e2e-valid', worktree=str(wt)),
                timeout=30,
            )
        assert resp_ok['status'] == 'done', f'expected done: {resp_ok}'
        assert 'failure_diagnostic' not in resp_ok, (
            f"'failure_diagnostic' must not appear in successful merge response: {resp_ok}"
        )

        await worker.stop()
        await worker_task


# ---------------------------------------------------------------------------
# TestInFlightMergeRegistry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInFlightMergeRegistry:
    """Tests for InFlightMergeRegistry — per-branch in-flight de-dup slot."""

    def _make_future(self) -> asyncio.Future:
        return asyncio.get_event_loop().create_future()

    async def test_acquire_free_branch_returns_true(self):
        """(a) Acquiring a free branch returns True; is_inflight becomes True."""
        registry = InFlightMergeRegistry()
        fut = self._make_future()

        result = registry.acquire('101', 'task-101', fut)

        assert result is True
        assert registry.is_inflight('101') is True

    async def test_acquire_held_branch_returns_false(self):
        """(b) A second acquire for the same branch returns False while first holds."""
        registry = InFlightMergeRegistry()
        fut1 = self._make_future()
        fut2 = self._make_future()

        first = registry.acquire('202', 'task-202a', fut1)
        second = registry.acquire('202', 'task-202b', fut2)

        assert first is True
        assert second is False
        # Branch still held by the first
        assert registry.is_inflight('202') is True

    async def test_resolving_future_releases_slot(self):
        """(c) Resolving the future auto-releases the slot via add_done_callback."""
        registry = InFlightMergeRegistry()
        fut = self._make_future()
        registry.acquire('303', 'task-303', fut)
        assert registry.is_inflight('303') is True

        # Resolve the future — the callback should fire
        fut.set_result(None)
        # Yield control so the done_callback runs
        await asyncio.sleep(0)

        assert registry.is_inflight('303') is False

    async def test_cancelling_future_releases_slot(self):
        """(c) Cancelling the future also auto-releases the slot."""
        registry = InFlightMergeRegistry()
        fut = self._make_future()
        registry.acquire('404', 'task-404', fut)
        assert registry.is_inflight('404') is True

        fut.cancel()
        await asyncio.sleep(0)

        assert registry.is_inflight('404') is False

    async def test_different_branches_are_independent(self):
        """(d) Acquiring different branches is fully independent."""
        registry = InFlightMergeRegistry()
        futA = self._make_future()
        futB = self._make_future()

        a = registry.acquire('A', 'task-A', futA)
        b = registry.acquire('B', 'task-B', futB)

        assert a is True
        assert b is True
        assert registry.is_inflight('A') is True
        assert registry.is_inflight('B') is True

        # Releasing A does not affect B
        futA.set_result(None)
        await asyncio.sleep(0)

        assert registry.is_inflight('A') is False
        assert registry.is_inflight('B') is True

    async def test_entry_exposes_task_id_and_eta(self):
        """(e) entry(branch) exposes task_id; eta_seconds returns an int >= 0."""
        registry = InFlightMergeRegistry()
        fut = self._make_future()

        registry.acquire('505', 'task-505', fut)

        entry = registry.entry('505')
        assert entry is not None
        assert entry.task_id == 'task-505'

        eta = registry.eta_seconds('505')
        assert eta is not None
        assert isinstance(eta, int)
        assert eta >= 0

    async def test_entry_and_eta_none_for_free_branch(self):
        """entry() and eta_seconds() return None for a branch not in-flight."""
        registry = InFlightMergeRegistry()

        assert registry.entry('unknown') is None
        assert registry.eta_seconds('unknown') is None

    async def test_acquire_after_release_dispatches_again(self):
        """After release a new acquire succeeds for the same branch."""
        registry = InFlightMergeRegistry()
        fut = self._make_future()
        registry.acquire('606', 'task-606', fut)

        fut.set_result(None)
        await asyncio.sleep(0)

        fut2 = self._make_future()
        result = registry.acquire('606', 'task-606b', fut2)
        assert result is True
        assert registry.is_inflight('606') is True


# ---------------------------------------------------------------------------
# TestCoalesceOrEnqueue — registry-only path (git_ops=None)
# ---------------------------------------------------------------------------


def _count_events(db_path, event_type: str) -> int:
    """Query the EventStore SQLite DB for a specific event type count."""
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            'SELECT COUNT(*) FROM events WHERE event_type = ?', (event_type,),
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


@pytest.mark.asyncio
class TestCoalesceOrEnqueueRegistryOnly:
    """Tests for coalesce_or_enqueue_merge_request with git_ops=None.

    Exercises the registry-only fast-path: no disk scan, no worktree
    creation needed.
    """

    def _make_event_store(self, tmp_path: Path) -> EventStore:
        db = tmp_path / 'coalesce_events.db'
        return EventStore(db_path=db, run_id='coalesce-test')

    async def test_first_call_dispatches(
        self, config: OrchestratorConfig, tmp_path: Path,
    ):
        """(a) First call: dispatched=True, in_flight=False, queue has 1 item,
        registry.is_inflight(branch) is True."""
        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = self._make_event_store(tmp_path)
        req = _make_request('111', '111', tmp_path, config)

        result = await coalesce_or_enqueue_merge_request(
            queue, req, event_store, registry, git_ops=None,
        )

        assert result.dispatched is True
        assert result.in_flight is False
        assert queue.qsize() == 1
        assert registry.is_inflight('111') is True

    async def test_second_call_coalesces(
        self, config: OrchestratorConfig, tmp_path: Path,
    ):
        """(b) Second call for same branch: in_flight=True, no duplicate enqueue,
        merge_coalesced event emitted."""
        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = self._make_event_store(tmp_path)
        req1 = _make_request('222', '222', tmp_path, config)
        req2 = _make_request('222', '222', tmp_path, config)

        await coalesce_or_enqueue_merge_request(
            queue, req1, event_store, registry, git_ops=None,
        )
        result2 = await coalesce_or_enqueue_merge_request(
            queue, req2, event_store, registry, git_ops=None,
        )

        assert result2.in_flight is True
        assert result2.dispatched is False
        # No duplicate enqueue
        assert queue.qsize() == 1
        # Exactly one merge_coalesced event
        assert _count_events(event_store.db_path, 'merge_coalesced') == 1

    async def test_third_call_dispatches_after_release(
        self, config: OrchestratorConfig, tmp_path: Path,
    ):
        """(c) After the first future resolves, a third call dispatches again."""
        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = self._make_event_store(tmp_path)
        req1 = _make_request('333', '333', tmp_path, config)

        await coalesce_or_enqueue_merge_request(
            queue, req1, event_store, registry, git_ops=None,
        )

        # Resolve the first future — releases the registry slot
        req1.result.set_result(MergeOutcome(status='done'))
        await asyncio.sleep(0)
        assert registry.is_inflight('333') is False

        req3 = _make_request('333', '333', tmp_path, config)
        result3 = await coalesce_or_enqueue_merge_request(
            queue, req3, event_store, registry, git_ops=None,
        )

        assert result3.dispatched is True
        assert result3.in_flight is False
        assert queue.qsize() == 2  # both requests enqueued (queue not drained)

    async def test_different_branches_always_dispatch(
        self, config: OrchestratorConfig, tmp_path: Path,
    ):
        """(d) Different branches are always dispatched independently."""
        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = self._make_event_store(tmp_path)
        req_a = _make_request('A', 'branchA', tmp_path, config)
        req_b = _make_request('B', 'branchB', tmp_path, config)

        result_a = await coalesce_or_enqueue_merge_request(
            queue, req_a, event_store, registry, git_ops=None,
        )
        result_b = await coalesce_or_enqueue_merge_request(
            queue, req_b, event_store, registry, git_ops=None,
        )

        assert result_a.dispatched is True
        assert result_b.dispatched is True
        assert queue.qsize() == 2
        # No coalesce events
        assert _count_events(event_store.db_path, 'merge_coalesced') == 0


# ---------------------------------------------------------------------------
# TestCoalesceOrEnqueueWorktreePath — disk-scan coalesces alive worktrees
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCoalesceOrEnqueueWorktreePath:
    """Tests for coalesce_or_enqueue_merge_request disk-scan branch.

    Uses a real GitOps and _merge-* worktree to simulate the cross-actor
    scenario where the in-memory registry is empty (e.g. after restart) but
    an in-progress merger's worktree exists on disk.
    """

    def _make_event_store(self, tmp_path: Path) -> EventStore:
        db = tmp_path / 'wt_coalesce_events.db'
        return EventStore(db_path=db, run_id='wt-coalesce-test')

    async def test_alive_worktree_coalesces_without_enqueue(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """Disk-scan detects a recent _merge-* worktree → in_flight=True,
        source='worktree', queue stays empty, worktree NOT removed."""
        from orchestrator.merge_queue import INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS

        # Create a real _merge-* worktree via merge_to_main
        branch = 'wt-coalesce-branch'
        wt = await _make_branch_with_file(
            git_ops, branch, 'wt_coalesce.py', 'x = 42\n',
        )
        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success, f'merge_to_main failed: {merge_result}'
        merge_wt = merge_result.merge_worktree
        assert merge_wt is not None

        try:
            queue: asyncio.Queue = asyncio.Queue()
            registry = InFlightMergeRegistry()   # EMPTY — simulates restart
            event_store = self._make_event_store(tmp_path)
            req = _make_request('wt-coalesce', branch, tmp_path, config)

            result = await coalesce_or_enqueue_merge_request(
                queue, req, event_store, registry,
                git_ops=git_ops,
                liveness_secs=INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS,
            )

            assert result.in_flight is True, f'Expected in_flight=True, got {result}'
            assert result.dispatched is False
            assert result.source == 'worktree'

            # Queue must be EMPTY — no duplicate enqueue
            assert queue.qsize() == 0, f'Expected empty queue, got {queue.qsize()}'

            # Worktree must NOT have been removed (it is alive)
            assert merge_wt.exists(), f'Alive worktree {merge_wt} was unexpectedly removed'

            # merge_coalesced event must be emitted
            assert _count_events(event_store.db_path, 'merge_coalesced') == 1

        finally:
            # Clean up the merge worktree
            if merge_wt is not None and merge_wt.exists():
                await git_ops.cleanup_merge_worktree(merge_wt)


# ---------------------------------------------------------------------------
# TestCoalesceOrEnqueueStaleWorktreeReap — reap abandoned worktree then dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCoalesceOrEnqueueStaleWorktreeReap:
    """Tests for the stale-worktree reap-then-dispatch path.

    When a _merge-* worktree exists on disk but is older than liveness_secs,
    coalesce_or_enqueue_merge_request should reap it and dispatch a fresh merge.
    """

    def _make_event_store(self, tmp_path: Path) -> EventStore:
        db = tmp_path / 'reap_events.db'
        return EventStore(db_path=db, run_id='reap-test')

    async def test_stale_worktree_is_reaped_and_dispatched(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """Stale _merge-* worktree → reap + worktree_reaped event +
        dispatch new request (dispatched=True, queue size 1)."""
        import os

        branch = 'stale-reap-branch'
        wt = await _make_branch_with_file(
            git_ops, branch, 'stale_reap.py', 'x = 1\n',
        )
        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success, f'merge_to_main failed: {merge_result}'
        merge_wt = merge_result.merge_worktree
        assert merge_wt is not None

        # Force mtime to ancient past so liveness check fails
        ancient_mtime = 0  # 1970-01-01
        os.utime(str(merge_wt), (ancient_mtime, ancient_mtime))

        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = self._make_event_store(tmp_path)
        req = _make_request('stale-reap', branch, tmp_path, config)

        result = await coalesce_or_enqueue_merge_request(
            queue, req, event_store, registry,
            git_ops=git_ops,
            liveness_secs=1,  # tiny window so age=ancient >> 1
        )

        # (a) stale worktree should be reaped
        assert not merge_wt.exists(), (
            f'Stale worktree {merge_wt} should have been removed'
        )
        # Confirm reaped by checking find_inflight returns None
        found = await git_ops.find_inflight_merge_worktree(branch)
        assert found is None, f'Worktree still registered after reap: {found}'

        # (b) worktree_reaped event emitted
        assert _count_events(event_store.db_path, 'worktree_reaped') == 1

        # (c) new request dispatched
        assert result.dispatched is True, f'Expected dispatched=True, got {result}'
        assert result.in_flight is False
        assert queue.qsize() == 1, f'Expected queue size 1, got {queue.qsize()}'
        # Registry now holds the new request
        assert registry.is_inflight(branch) is True
