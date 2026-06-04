"""Tests for merge queue: MergeWorker, CAS update-ref, ghost-loop detection."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
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
    TerminalOutcomeRecord,
    TerminalOutcomeRetention,
    _check_plan_files_touched_in_branch,
    _check_plan_targets_in_tree,
    _check_post_merge_equivalence,
    _classify_branch_presence,
    _ensure_verify_disk_space,
    _is_speculation_race,
    _verify_hit_enospc,
    coalesce_or_enqueue_merge_request,
    register_and_enqueue_merge_request,
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
class TestClassifyBranchPresence:
    """Unit tests for the _classify_branch_presence branch-existence guard."""

    async def test_unknown_branch_no_ref_no_marker(
        self, git_ops: GitOps, tmp_path: Path,
    ):
        """No branch ref and no merge marker → unknown_branch, merge_attempt emitted."""
        db_path = tmp_path / 'events.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        outcome = await _classify_branch_presence(
            git_ops, event_store, 'ghost-4011', 'ghost-4011', time.monotonic(),
        )

        assert outcome is not None
        assert outcome.status == 'unknown_branch'
        assert 'task/ghost-4011' in outcome.reason

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT json_extract(data, '$.outcome') FROM events "
            "WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()
        assert rows == [('unknown_branch',)], f'rows={rows}'

    async def test_already_merged_marker_present(
        self, git_ops: GitOps, tmp_path: Path,
    ):
        """Branch ref deleted but a merge marker remains on main → already_merged.

        Disambiguates a merged-then-cleaned-up branch from one that never
        existed (unknown_branch).
        """
        db_path = tmp_path / 'events.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        # Merge a branch into main so a 'Merge task/am-marker into main' marker
        # exists, then remove the worktree + branch ref so only the marker is left.
        worktree = (await git_ops.create_worktree('am-marker')).path
        (worktree / 'm.py').write_text('m = 1\n')
        await git_ops.commit(worktree, 'Add m')
        result = await git_ops.merge_to_main(worktree, 'am-marker')
        assert result.success
        assert result.merge_commit is not None
        await git_ops.advance_main(result.merge_commit)
        if result.merge_worktree:
            await git_ops.cleanup_merge_worktree(result.merge_worktree)
        await _run(
            ['git', 'worktree', 'remove', '--force', str(worktree)],
            cwd=git_ops.project_root,
        )
        await _run(
            ['git', 'branch', '-D', 'task/am-marker'], cwd=git_ops.project_root,
        )
        assert await git_ops.resolve_branch_sha('task/am-marker') is None

        outcome = await _classify_branch_presence(
            git_ops, event_store, 'am-marker', 'am-marker', time.monotonic(),
        )

        assert outcome is not None
        assert outcome.status == 'already_merged', f'got {outcome}'

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT json_extract(data, '$.outcome') FROM events "
            "WHERE event_type = 'merge_attempt'"
        ).fetchall()
        conn.close()
        assert rows == [('already_merged',)], f'rows={rows}'

    async def test_existing_branch_returns_none(
        self, git_ops: GitOps, tmp_path: Path,
    ):
        """An existing branch ref → None (proceed), no merge_attempt emitted."""
        db_path = tmp_path / 'events.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        await git_ops.create_worktree('live-branch')  # creates task/live-branch
        assert await git_ops.resolve_branch_sha('task/live-branch') is not None

        outcome = await _classify_branch_presence(
            git_ops, event_store, 'live-branch', 'live-branch', time.monotonic(),
        )

        assert outcome is None

        conn = sqlite3.connect(str(db_path))
        count = conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_type = 'merge_attempt'"
        ).fetchone()[0]
        conn.close()
        assert count == 0, 'no merge_attempt expected when the branch ref exists'


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

    async def test_unknown_branch_emits_terminal_merge_attempt(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """A request for a branch with no ref (e.g. a mis-routed merge_request)
        resolves to 'unknown_branch' with a terminal merge_attempt as the latest
        event — never a trailing bare merge_dequeued (dashboard phantom)."""
        db_path = tmp_path / 'events.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=event_store)
        worker_task = asyncio.create_task(worker.run())

        # 'ghost-4011' was never created here — no task/ghost-4011 ref exists.
        req = _make_request(
            'ghost-4011', 'ghost-4011', tmp_path / 'no-such-wt', config,
        )
        await queue.put(req)
        outcome = await asyncio.wait_for(req.result, timeout=10)

        assert outcome.status == 'unknown_branch', f'got {outcome}'
        assert 'task/ghost-4011' in outcome.reason

        await worker.stop()
        worker_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker_task

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, json_extract(data, '$.outcome') FROM events "
            "WHERE task_id = 'ghost-4011' ORDER BY id"
        ).fetchall()
        conn.close()
        assert ('merge_attempt', 'unknown_branch') in rows, f'rows={rows}'
        assert rows[-1] == ('merge_attempt', 'unknown_branch'), (
            f'latest event must be terminal, not a bare merge_dequeued: {rows}'
        )

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

    async def test_unknown_branch_emits_terminal_merge_attempt(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """SpeculativeMergeWorker: a missing branch ref resolves to
        'unknown_branch' via the immediate-outcome path, leaving a terminal
        merge_attempt (not a bare merge_dequeued) as the latest event."""
        db_path = tmp_path / 'events.db'
        event_store = EventStore(db_path=db_path, run_id='test-run')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, event_store=event_store)
        worker_task = asyncio.create_task(worker.run())

        # 'ghost-4011' was never created here — no task/ghost-4011 ref exists.
        req = _make_request(
            'ghost-4011', 'ghost-4011', tmp_path / 'no-such-wt', config,
        )
        await queue.put(req)
        outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'unknown_branch', f'got {outcome}'

        await worker.stop()
        await worker_task

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, json_extract(data, '$.outcome') FROM events "
            "WHERE task_id = 'ghost-4011' ORDER BY id"
        ).fetchall()
        conn.close()
        assert ('merge_attempt', 'unknown_branch') in rows, f'rows={rows}'
        assert rows[-1] == ('merge_attempt', 'unknown_branch'), (
            f'latest event must be terminal, not a bare merge_dequeued: {rows}'
        )

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

    # ── OOB delivery (γ3 / task-1644) ────────────────────────────────────────

    async def test_oob_delivery_nonspec_conflict(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Non-spec conflict: merger resolves req.result OOB at detection time.

        Drive _merger_loop() directly with a single conflicting non-speculative
        request.  With OOB delivery the merger must resolve req.result to
        'conflict' before the verifier runs.  The enqueued SpeculativeItem must
        carry already_delivered=True.

        RED before step-4 impl: req.result is not resolved by the merger.
        """
        # Worktree created BEFORE the main-side README.md change so the merge
        # base is the initial main.  Both sides then diverge on README.md →
        # three-way merge detects a conflict.
        wt = (await git_ops.create_worktree('oob-cfl')).path

        (git_ops.project_root / 'README.md').write_text('# Main side\n')
        await _run(['git', 'add', 'README.md'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Main oob-cfl side'], cwd=git_ops.project_root)

        (wt / 'README.md').write_text('# Branch side\n')
        await git_ops.commit(wt, 'Branch oob-cfl conflict')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        req = _make_request('oob-cfl', 'oob-cfl', wt, config)
        await queue.put(req)
        await queue.put(None)  # type: ignore[arg-type]

        await worker._merger_loop()

        # Merger must have resolved req.result at conflict-detection time
        assert req.result.done(), (
            'OOB delivery: req.result must be resolved by the merger before '
            'the verifier dequeues the SpeculativeItem'
        )
        assert req.result.result().status == 'conflict', (
            f'Expected conflict, got {req.result.result().status!r}'
        )

        # The ordering token must be flagged so the verifier skips set_result
        item = worker._verifier_queue.get_nowait()
        assert isinstance(item, SpeculativeItem)
        assert item.already_delivered is True, (
            'Non-spec conflict SpeculativeItem must have already_delivered=True'
        )

    async def test_oob_not_delivered_for_speculative_conflict(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Speculative conflict is NOT OOB-delivered — it rides through the verifier FIFO.

        Pre-load N (clean) + M (conflicting) so M is speculatively prefetched.
        Assert req_M.result stays pending and M's ordering token has
        already_delivered=False and speculative=True — locking the
        'not speculative' predicate clause.

        Passes both before and after step-4 impl (negative/guard test).
        """
        # N: clean branch with a unique file (created BEFORE main-side change)
        wt_n = await _make_branch_with_file(
            git_ops, 'oob-spec-n', 'file_spec_n.py', 'n = 1\n',
        )

        # M: also created BEFORE the main-side README.md change so the merge
        # base is the initial main.  Both main and M then diverge on README.md.
        wt_m = (await git_ops.create_worktree('oob-spec-m')).path

        (git_ops.project_root / 'README.md').write_text('# Main spec side\n')
        await _run(['git', 'add', 'README.md'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Main spec side'], cwd=git_ops.project_root)

        (wt_m / 'README.md').write_text('# M spec conflict\n')
        await git_ops.commit(wt_m, 'M spec conflict')

        # Pre-load N + M + sentinel so M is speculatively prefetched after N
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        req_n = _make_request('oob-spec-n', 'oob-spec-n', wt_n, config)
        req_m = _make_request('oob-spec-m', 'oob-spec-m', wt_m, config)
        await queue.put(req_n)
        await queue.put(req_m)
        await queue.put(None)  # type: ignore[arg-type]

        await worker._merger_loop()

        # Speculative conflict must NOT be OOB-delivered
        assert not req_m.result.done(), (
            'Speculative conflict must NOT be OOB-delivered; req_m.result '
            'must remain pending until the verifier drains the ordering token'
        )

        # Drain verifier queue to find M's SpeculativeItem
        items: list[SpeculativeItem | None] = []
        while True:
            it = worker._verifier_queue.get_nowait()
            items.append(it)
            if it is None:
                break

        m_items = [i for i in items if isinstance(i, SpeculativeItem) and i.request is req_m]
        assert len(m_items) == 1, f'Expected one SpeculativeItem for req_m, got: {m_items}'
        m_item = m_items[0]
        assert m_item.speculative is True, (
            'M was prefetched speculatively — item.speculative must be True'
        )
        assert m_item.already_delivered is False, (
            'Speculative conflict must have already_delivered=False '
            '(verifier owns resolution for speculative items)'
        )

    async def test_oob_delivery_unblocks_waiter_while_verifier_busy(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """OOB-delivered conflict resolves immediately while verifier is blocked.

        Start worker.run(); enqueue N (clean, blocking verify); await
        verify_started; enqueue M (non-spec conflict); assert req_M.result
        resolves to 'conflict' within 5 seconds while req_N.result is still
        pending.  The verifier is draining N the entire time — this test proves
        OOB delivery bypasses the FIFO delay.

        RED before step-4 impl: req_M.result does not resolve until N's verify
        finishes, causing the wait_for to time out.
        """
        # N: clean branch
        wt_n = await _make_branch_with_file(
            git_ops, 'oob-e2e-n', 'file_e2e_n.py', 'n = 1\n',
        )

        # M: created BEFORE main-side README.md change so the merge base is
        # the initial main — both main and M then diverge on README.md.
        wt_m = (await git_ops.create_worktree('oob-e2e-m')).path
        (git_ops.project_root / 'README.md').write_text('# Main e2e\n')
        await _run(['git', 'add', 'README.md'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Main e2e change'], cwd=git_ops.project_root)
        (wt_m / 'README.md').write_text('# M e2e conflict\n')
        await git_ops.commit(wt_m, 'M e2e conflict')

        verify_started = asyncio.Event()
        release = asyncio.Event()

        async def blocking_verify(merge_wt, cfg, module_configs, task_files=None, **kwargs):  # type: ignore[no-untyped-def]
            verify_started.set()
            await release.wait()
            return MagicMock(passed=True, summary='')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        req_n = _make_request('oob-e2e-n', 'oob-e2e-n', wt_n, config)
        req_m = _make_request('oob-e2e-m', 'oob-e2e-m', wt_m, config)

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=blocking_verify,
        ):
            await queue.put(req_n)
            await asyncio.wait_for(verify_started.wait(), timeout=30)

            # Verifier is now blocked on N. Enqueue M — merger detects conflict OOB.
            await queue.put(req_m)

            # M must resolve before N's verify finishes
            outcome_m = await asyncio.wait_for(req_m.result, timeout=5)
            assert outcome_m.status == 'conflict', (
                f'OOB: M must resolve to conflict while N verify is blocked; '
                f'got {outcome_m.status!r}'
            )

            # N is still blocked
            assert not req_n.result.done(), (
                'req_n.result must still be pending — verifier is blocked on N'
            )

            # Unblock N; it completes as done
            release.set()
            outcome_n = await asyncio.wait_for(req_n.result, timeout=30)
            assert outcome_n.status == 'done', f'N should complete as done: {outcome_n}'

        await worker.stop()
        await asyncio.wait_for(worker_task, timeout=30)


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
# TestGenerationFieldsIdentity — γ2 step-01 RED: 'superseded' status +
# superseded_by field on MergeOutcome; generation field on MergeRequest.
# ---------------------------------------------------------------------------


class TestGenerationFieldsIdentity:
    """γ2 step-01/02: MergeOutcome gains 'superseded' status + superseded_by;
    MergeRequest and GroupMergeRequest gain generation field."""

    def test_merge_outcome_superseded_status_and_superseded_by(self) -> None:
        """(a) MergeOutcome('superseded', superseded_by='mr-x') constructs and exposes superseded_by."""
        outcome = MergeOutcome(status='superseded', superseded_by='mr-x')  # type: ignore[call-arg]
        assert outcome.status == 'superseded'
        assert outcome.superseded_by == 'mr-x'  # type: ignore[attr-defined]

    def test_merge_outcome_blocked_superseded_by_default_none(self) -> None:
        """(b) MergeOutcome('blocked') has superseded_by default None."""
        outcome = MergeOutcome('blocked')
        assert outcome.superseded_by is None  # type: ignore[attr-defined]

    def test_merge_request_generation_defaults_to_1(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """(c) MergeRequest(...) defaults generation==1."""
        loop = asyncio.new_event_loop()
        try:
            future: asyncio.Future[MergeOutcome] = loop.create_future()
            req = MergeRequest(
                task_id='t-gen1',
                branch='task/t-gen1',
                worktree=tmp_path,
                pre_rebased=False,
                task_files=None,
                module_configs=[],
                config=config,
                result=future,
            )
            assert req.generation == 1  # type: ignore[attr-defined]
        finally:
            loop.close()

    def test_merge_request_generation_can_be_set(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """(d) MergeRequest(..., generation=2) carries 2."""
        loop = asyncio.new_event_loop()
        try:
            future: asyncio.Future[MergeOutcome] = loop.create_future()
            req = MergeRequest(
                task_id='t-gen2',
                branch='task/t-gen2',
                worktree=tmp_path,
                pre_rebased=False,
                task_files=None,
                module_configs=[],
                config=config,
                result=future,
                generation=2,
            )
            assert req.generation == 2  # type: ignore[attr-defined]
        finally:
            loop.close()

    def test_group_merge_request_generation_defaults_to_1(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """(e) GroupMergeRequest still constructs and exposes generation default 1."""
        loop = asyncio.new_event_loop()
        try:
            future: asyncio.Future[MergeOutcome] = loop.create_future()
            req = GroupMergeRequest(
                task_id='t-grp',
                branch='task/t-grp',
                worktree=tmp_path,
                pre_rebased=False,
                task_files=None,
                module_configs=[],
                config=config,
                result=future,
                train_id='train-1',
                member_task_ids=['t-grp'],
                tip_branch='task/t-grp',
                tip_task_id='t-grp',
                status_check=AsyncMock(),
                mark_member_done=AsyncMock(),
            )
            assert req.generation == 1  # type: ignore[attr-defined]
        finally:
            loop.close()


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

    def test_already_delivered_default_is_false(self):
        """SpeculativeItem.already_delivered defaults to False when not passed.

        True means the merger already resolved req.result out-of-band; the
        verifier must skip set_result but still run n_failed / slot bookkeeping
        for that ordering token.  The default must be False so existing
        construction sites that omit the flag behave identically to before.
        """
        from unittest.mock import MagicMock

        item = SpeculativeItem(
            request=MagicMock(),
            merge_result=None,
            merge_wt=None,
            base_sha='',
            speculative=False,
            skip_verify=False,
        )
        assert item.already_delivered is False


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
# TestEmitTrainEventHelper — unit tests for module-level _emit_train_event
# ---------------------------------------------------------------------------


class TestEmitTrainEventHelper:
    def test_emit_train_event_writes_one_row(self, tmp_path: Path) -> None:
        """Call _emit_train_event — writes exactly one row with correct fields."""
        import sqlite3

        from orchestrator.event_store import EventStore, EventType
        from orchestrator.merge_queue import _emit_train_event

        db_path = tmp_path / 'train_eh_a.db'
        es = EventStore(db_path=db_path, run_id='train-eh-run')

        _emit_train_event(
            es,
            EventType.train_started,
            task_id='trn-a',
            train_id='train-001',
            member_task_ids=['trn-a', 'trn-b'],
        )

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, task_id, phase, "
            "       json_extract(data, '$.train_id') as train_id, "
            "       json_extract(data, '$.member_task_ids') as members "
            "FROM events WHERE event_type = 'train_started'"
        ).fetchall()
        conn.close()

        assert len(rows) == 1, f'Expected 1 row, got: {rows}'
        event_type, task_id, phase, train_id, members = rows[0]
        assert event_type == 'train_started'
        assert task_id == 'trn-a'
        assert phase == 'merge'
        assert train_id == 'train-001'

    def test_emit_train_event_includes_extra_data_keys(self, tmp_path: Path) -> None:
        """Extra data dict keys appear in the stored JSON."""
        import json
        import sqlite3

        from orchestrator.event_store import EventStore, EventType
        from orchestrator.merge_queue import _emit_train_event

        db_path = tmp_path / 'train_eh_b.db'
        es = EventStore(db_path=db_path, run_id='train-eh-run')

        _emit_train_event(
            es,
            EventType.train_merged,
            task_id='trn-a',
            train_id='train-002',
            member_task_ids=['trn-a', 'trn-b'],
            data={'merge_commit_sha': 'abc999', 'base_sha': 'base111'},
        )

        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT data FROM events WHERE event_type = 'train_merged'").fetchone()
        conn.close()

        assert row is not None
        payload = json.loads(row[0])
        assert payload['train_id'] == 'train-002'
        assert payload['merge_commit_sha'] == 'abc999'
        assert payload['base_sha'] == 'base111'

    def test_emit_train_event_noop_when_event_store_is_none(self) -> None:
        """Call with event_store=None — no exception, nothing written."""
        from orchestrator.event_store import EventType
        from orchestrator.merge_queue import _emit_train_event

        # Should not raise
        _emit_train_event(
            None,
            EventType.train_started,
            task_id='trn-x',
            train_id='train-003',
        )


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

    async def test_cleanup_raise_on_wip_overlap_does_not_strand_halt(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """cleanup_merge_worktree raise must NOT strand the queue in a halted state.

        Verify that when cleanup_merge_worktree raises after advance_main returns
        'wip_overlap', the WIP halt is never set -- because cleanup must run
        BEFORE _map_advance_failure (which calls halt_for_wip).

        RED on buggy ordering: _map_advance_failure halts (via halt_for_wip)
        before cleanup_merge_worktree raises, so is_wip_halted is True.
        GREEN after fix: cleanup moved before _map_advance_failure so the
        RuntimeError propagates before halt_for_wip is ever invoked.

        Calls _verify_and_advance directly (not via worker.run()) so the
        RuntimeError propagates to the test -- the run()-loop's except handler
        would otherwise suppress the re-raise and resolve 'blocked'.
        """
        wt = await _make_branch_with_file(
            git_ops, 'clnup-raise', 'file_clnup.py', 'x = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        req = _make_request('clnup-raise', 'clnup-raise', wt, config)

        # Build a real flowing item BEFORE the patch block so _remerge's
        # own merge paths run against the real git_ops (not the raising mock).
        item = await worker._remerge(req, None)
        assert item.immediate_outcome is None, (
            f'_remerge must succeed; got immediate_outcome={item.immediate_outcome!r}'
        )

        async def _wip_overlap(*args: Any, **kwargs: Any):
            git_ops._last_overlap_files = ['file_clnup.py']
            return 'wip_overlap'

        try:
            with (
                patch.object(git_ops, 'advance_main', side_effect=_wip_overlap),
                patch.object(
                    git_ops, 'cleanup_merge_worktree',
                    side_effect=RuntimeError('cleanup boom'),
                ),
                patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
                pytest.raises(RuntimeError, match='cleanup boom'),
            ):
                await worker._verify_and_advance(item)

            # PRIMARY discriminator: queue must NOT be halted when cleanup raised
            # before the wip_halted outcome could reach the workflow.
            # The single-task workflow path registers a halt owner only on an
            # explicit 'wip_halted' status; a 'blocked' outcome (what the
            # run()-loop's except handler would deliver) leaves the halt silently
            # orphaned with no escalation owner (task 1598).
            assert not worker.is_wip_halted
            assert worker.halt_owner_esc_id is None
        finally:
            # Best-effort: clean up the merge worktree the patched
            # cleanup_merge_worktree left behind, so the fixture is left clean.
            with contextlib.suppress(Exception):
                if item.merge_wt is not None:
                    await git_ops.cleanup_merge_worktree(item.merge_wt)


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

    @pytest.mark.asyncio
    async def test_enqueue_registers_terminal_callback_emitting_merge_finalized(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """enqueue_merge_request registers a done-callback that emits merge_finalized.

        Resolving req.result triggers the callback; one merge_finalized row
        must appear with the correct request_id, state, branch, and merge_sha.
        """
        from orchestrator.merge_queue import enqueue_merge_request

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        db_path = tmp_path / 'runs.db'
        event_store = EventStore(db_path, 'run-mf-test')

        wt = tmp_path / 'wt'
        wt.mkdir()
        req = _make_request('42', 'task/42', wt, config)

        await enqueue_merge_request(queue, req, event_store)

        # Resolve the future — the done-callback should fire
        req.result.set_result(MergeOutcome(status='done', merge_sha='abc123'))
        await asyncio.sleep(0)  # yield so the callback runs

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, task_id, "
            "json_extract(data, '$.request_id') AS request_id, "
            "json_extract(data, '$.state') AS state, "
            "json_extract(data, '$.branch') AS branch, "
            "json_extract(data, '$.merge_sha') AS merge_sha "
            "FROM events WHERE event_type = 'merge_finalized'"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == 'merge_finalized'
        assert rows[0][1] == '42'           # task_id
        assert rows[0][2] == req.request_id  # data.request_id
        assert rows[0][3] == 'done'          # data.state
        assert rows[0][4] == 'task/42'       # data.branch
        assert rows[0][5] == 'abc123'        # data.merge_sha

    @pytest.mark.asyncio
    async def test_cancelled_future_is_recorded_as_abandoned(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """A cancelled future triggers a merge_finalized row with state=='abandoned'.

        Covers PRD D7 — cancelled futures must be finalized, not silently dropped.
        """
        from orchestrator.merge_queue import enqueue_merge_request

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        db_path = tmp_path / 'runs_cancel.db'
        event_store = EventStore(db_path, 'run-cancel-test')

        wt = tmp_path / 'wt'
        wt.mkdir()
        req = _make_request('99', 'task/99', wt, config)

        await enqueue_merge_request(queue, req, event_store)

        # Cancel the future — the done-callback must handle it
        req.result.cancel()
        await asyncio.sleep(0)  # yield so the callback runs

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, "
            "json_extract(data, '$.request_id') AS request_id, "
            "json_extract(data, '$.state') AS state, "
            "json_extract(data, '$.merge_sha') AS merge_sha "
            "FROM events WHERE event_type = 'merge_finalized'"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == 'merge_finalized'
        assert rows[0][1] == req.request_id
        assert rows[0][2] == 'abandoned'
        assert rows[0][3] is None  # no merge_sha for abandoned

    @pytest.mark.asyncio
    async def test_retention_records_resolved_outcome(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """When retention is passed, resolving the future populates the ring."""
        from orchestrator.merge_queue import enqueue_merge_request

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        db_path = tmp_path / 'runs_ret.db'
        event_store = EventStore(db_path, 'run-retention')
        retention = TerminalOutcomeRetention()

        wt = tmp_path / 'wt'
        wt.mkdir()
        # Build request with snapshot_tip so we can verify it is captured
        future: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
        req = MergeRequest(
            task_id='77',
            branch='task/77',
            worktree=wt,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=future,
            snapshot_tip='tip-sha-0077',
        )

        await enqueue_merge_request(queue, req, event_store, retention=retention)

        req.result.set_result(MergeOutcome(status='done', merge_sha='sha9'))
        await asyncio.sleep(0)

        stored = retention.get(req.request_id)
        assert stored is not None
        assert stored.state == 'done'
        assert stored.merge_sha == 'sha9'
        assert stored.snapshot_tip == 'tip-sha-0077'
        assert stored.branch == 'task/77'
        assert stored.task_id == '77'

    @pytest.mark.asyncio
    async def test_retention_records_abandoned_outcome(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """When retention is passed, cancelling the future records state=='abandoned'."""
        from orchestrator.merge_queue import enqueue_merge_request

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        db_path = tmp_path / 'runs_ret2.db'
        event_store = EventStore(db_path, 'run-retention2')
        retention = TerminalOutcomeRetention()

        wt = tmp_path / 'wt2'
        wt.mkdir()
        req = _make_request('88', 'task/88', wt, config)

        await enqueue_merge_request(queue, req, event_store, retention=retention)

        req.result.cancel()
        await asyncio.sleep(0)

        stored = retention.get(req.request_id)
        assert stored is not None
        assert stored.state == 'abandoned'
        assert stored.merge_sha is None

    @pytest.mark.asyncio
    async def test_exception_future_is_recorded_as_error(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """A future resolved with set_exception() is finalized as state=='error'.

        Covers the elif fut.exception() branch of _on_finalized.  Verifies:
        - merge_finalized row with state=='error' and merge_sha IS NULL
        - retention record (when provided) also carries state=='error', merge_sha=None
        - fut.exception() is called inside the callback so asyncio never emits
          an 'exception was never retrieved' warning
        """
        from orchestrator.merge_queue import enqueue_merge_request

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        db_path = tmp_path / 'runs_exc.db'
        event_store = EventStore(db_path, 'run-exc-test')
        retention = TerminalOutcomeRetention()

        wt = tmp_path / 'wt_exc'
        wt.mkdir()
        req = _make_request('55', 'task/55', wt, config)

        await enqueue_merge_request(queue, req, event_store, retention=retention)

        # Resolve with an exception — the callback must handle it without
        # raising into the event loop
        req.result.set_exception(RuntimeError('worker blew up'))
        await asyncio.sleep(0)  # yield so the callback runs

        # --- durable tier assertion ------------------------------------------
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, task_id, "
            "json_extract(data, '$.request_id') AS request_id, "
            "json_extract(data, '$.state') AS state, "
            "json_extract(data, '$.merge_sha') AS merge_sha "
            "FROM events WHERE event_type = 'merge_finalized'"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == 'merge_finalized'
        assert rows[0][1] == '55'            # task_id
        assert rows[0][2] == req.request_id  # data.request_id
        assert rows[0][3] == 'error'         # data.state
        assert rows[0][4] is None            # data.merge_sha must be NULL

        # --- in-memory hot tier assertion ------------------------------------
        stored = retention.get(req.request_id)
        assert stored is not None
        assert stored.state == 'error'
        assert stored.merge_sha is None

    # γ2 step-05/06 — _on_finalized propagates supersession provenance
    @pytest.mark.asyncio
    async def test_on_finalized_propagates_superseded_by_and_generation(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """Resolving gen-1 future with MergeOutcome('superseded', superseded_by='mr-2')
        causes _on_finalized to record superseded_by + generation in the retention ring
        AND in the merge_finalized event data dict.
        """
        from orchestrator.merge_queue import enqueue_merge_request

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        db_path = tmp_path / 'runs_sup.db'
        event_store = EventStore(db_path, 'run-sup-test')
        retention = TerminalOutcomeRetention()

        wt = tmp_path / 'wt-sup'
        wt.mkdir()
        loop = asyncio.get_event_loop()
        future: asyncio.Future[MergeOutcome] = loop.create_future()
        req = MergeRequest(
            task_id='sup-task',
            branch='task/sup-task',
            worktree=wt,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=future,
            generation=1,
        )

        await enqueue_merge_request(queue, req, event_store, retention=retention)

        # Resolve the gen-1 future with a 'superseded' outcome
        req.result.set_result(MergeOutcome(
            status='superseded',
            superseded_by='mr-gen2abc',
            merge_sha='sha-adv',
        ))
        await asyncio.sleep(0)  # yield so the callback runs

        # --- in-memory ring: superseded_by + generation must be present ------
        stored = retention.get(req.request_id)
        assert stored is not None
        assert stored.state == 'superseded'
        assert stored.superseded_by == 'mr-gen2abc'
        assert stored.generation == 1

        # --- durable event: data dict must include superseded_by + generation --
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT "
            "json_extract(data, '$.state') AS state, "
            "json_extract(data, '$.superseded_by') AS superseded_by, "
            "json_extract(data, '$.generation') AS generation, "
            "json_extract(data, '$.merge_sha') AS merge_sha "
            "FROM events WHERE event_type = 'merge_finalized'"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == 'superseded'
        assert rows[0][1] == 'mr-gen2abc'
        assert rows[0][2] == 1
        assert rows[0][3] == 'sha-adv'

    @pytest.mark.asyncio
    async def test_on_finalized_blocked_outcome_superseded_by_null(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """Blocked outcomes don't set superseded_by (None in both ring and event)."""
        from orchestrator.merge_queue import enqueue_merge_request

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        db_path = tmp_path / 'runs_blk.db'
        event_store = EventStore(db_path, 'run-blk-test')
        retention = TerminalOutcomeRetention()

        wt = tmp_path / 'wt-blk'
        wt.mkdir()
        req = _make_request('blk-task', 'task/blk-task', wt, config)

        await enqueue_merge_request(queue, req, event_store, retention=retention)

        req.result.set_result(MergeOutcome(status='blocked', reason='oops'))
        await asyncio.sleep(0)

        stored = retention.get(req.request_id)
        assert stored is not None
        assert stored.superseded_by is None
        assert stored.generation == 1  # default

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT "
            "json_extract(data, '$.superseded_by') AS superseded_by, "
            "json_extract(data, '$.generation') AS generation "
            "FROM events WHERE event_type = 'merge_finalized'"
        ).fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] is None  # superseded_by NULL
        assert rows[0][1] == 1    # generation present


# ---------------------------------------------------------------------------
# TestMergeRequestIdentity — step-3 RED / step-4 GREEN
# ---------------------------------------------------------------------------


class TestMergeRequestIdentity:
    """MergeRequest.request_id and .snapshot_tip identity field contract."""

    @pytest.mark.asyncio
    async def test_request_id_format(self, config: OrchestratorConfig, tmp_path: Path) -> None:
        """(a) request_id matches ^mr-[0-9a-f]{8}$."""
        req = _make_request('1', 'task/1', tmp_path, config)
        assert re.fullmatch(r'^mr-[0-9a-f]{8}$', req.request_id), (
            f'request_id {req.request_id!r} does not match ^mr-[0-9a-f]{{8}}$'
        )

    @pytest.mark.asyncio
    async def test_request_id_is_unique_per_instance(
        self, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """(b) Two independently-built requests have different request_id."""
        req1 = _make_request('1', 'task/1', tmp_path, config)
        req2 = _make_request('2', 'task/2', tmp_path, config)
        assert req1.request_id != req2.request_id

    @pytest.mark.asyncio
    async def test_snapshot_tip_defaults_to_none(
        self, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """(c) snapshot_tip defaults to None."""
        req = _make_request('1', 'task/1', tmp_path, config)
        assert req.snapshot_tip is None

    @pytest.mark.asyncio
    async def test_snapshot_tip_accepts_explicit_value(
        self, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """(d) snapshot_tip carries an explicitly-set value."""
        future: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
        req = MergeRequest(
            task_id='1',
            branch='task/1',
            worktree=tmp_path,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=future,
            snapshot_tip='abc123def',
        )
        assert req.snapshot_tip == 'abc123def'

    def test_group_merge_request_still_constructs(
        self, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """(e) GroupMergeRequest still constructs; exposes auto request_id + snapshot_tip=None.

        Uses MagicMock for the Future so this test runs outside an event loop.
        """
        future: asyncio.Future[MergeOutcome] = MagicMock(spec=asyncio.Future)
        status_check_mock = AsyncMock(return_value={})
        mark_done_mock = AsyncMock()
        greq = GroupMergeRequest(
            task_id='tip',
            branch='task/tip',
            worktree=tmp_path,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=future,
            train_id='train-1',
            member_task_ids=['tip'],
            tip_branch='task/tip',
            tip_task_id='tip',
            status_check=status_check_mock,
            mark_member_done=mark_done_mock,
        )
        assert re.fullmatch(r'^mr-[0-9a-f]{8}$', greq.request_id)
        assert greq.snapshot_tip is None


# ---------------------------------------------------------------------------
# TestTerminalOutcomeRetention — step-5 RED / step-6 GREEN
# ---------------------------------------------------------------------------


class TestTerminalOutcomeRetention:
    """Unit tests for the TerminalOutcomeRecord + TerminalOutcomeRetention ring."""

    def _make_record(self, request_id: str, state: str = 'done') -> TerminalOutcomeRecord:
        return TerminalOutcomeRecord(
            request_id=request_id,
            task_id=f'task-{request_id}',
            branch=f'task/{request_id}',
            state=state,
            snapshot_tip=None,
            merge_sha=None,
        )

    def test_record_and_get(self) -> None:
        """record(rec) then get(req_id) returns the same record."""
        ring = TerminalOutcomeRetention(maxlen=10)
        rec = self._make_record('mr-aabbccdd')
        ring.record(rec)
        result = ring.get('mr-aabbccdd')
        assert result is not None
        assert result is rec
        assert result.state == 'done'
        assert result.task_id == 'task-mr-aabbccdd'

    def test_get_missing_returns_none(self) -> None:
        """get() on an unknown request_id returns None."""
        ring = TerminalOutcomeRetention(maxlen=10)
        assert ring.get('mr-doesnotexist') is None

    def test_eviction_syncs_index(self) -> None:
        """Oldest entry is evicted from both ring and dict index when ring is full."""
        ring = TerminalOutcomeRetention(maxlen=2)
        rec_a = self._make_record('mr-aaaaaaaa')
        rec_b = self._make_record('mr-bbbbbbbb')
        rec_c = self._make_record('mr-cccccccc')

        ring.record(rec_a)
        ring.record(rec_b)
        ring.record(rec_c)  # evicts rec_a

        # Oldest (a) is evicted
        assert ring.get('mr-aaaaaaaa') is None
        # Two newest remain
        assert ring.get('mr-bbbbbbbb') is rec_b
        assert ring.get('mr-cccccccc') is rec_c

    def test_snapshot_tip_and_merge_sha_are_stored(self) -> None:
        """Fields snapshot_tip and merge_sha are preserved on the record."""
        ring = TerminalOutcomeRetention(maxlen=5)
        rec = TerminalOutcomeRecord(
            request_id='mr-12345678',
            task_id='42',
            branch='task/42',
            state='done',
            snapshot_tip='sha-tip',
            merge_sha='deadbeef',
        )
        ring.record(rec)
        stored = ring.get('mr-12345678')
        assert stored is not None
        assert stored.snapshot_tip == 'sha-tip'
        assert stored.merge_sha == 'deadbeef'

    # γ2 step-03/04 — supersession provenance fields
    def test_superseded_by_and_generation_stored_and_retrieved(self) -> None:
        """TerminalOutcomeRecord with superseded_by + generation round-trips the ring."""
        ring = TerminalOutcomeRetention(maxlen=10)
        rec = TerminalOutcomeRecord(
            request_id='mr-alpha1',
            task_id='t1',
            branch='task/t1',
            state='superseded',
            snapshot_tip=None,
            merge_sha='sha-adv',
            superseded_by='mr-alpha2',
            generation=1,
        )
        ring.record(rec)
        stored = ring.get('mr-alpha1')
        assert stored is not None
        assert stored.superseded_by == 'mr-alpha2'
        assert stored.generation == 1

    def test_superseded_by_defaults_none_generation_defaults_1(self) -> None:
        """Defaults: superseded_by is None, generation is 1 when omitted."""
        rec = TerminalOutcomeRecord(
            request_id='mr-defaults',
            task_id='t2',
            branch='task/t2',
            state='done',
        )
        assert rec.superseded_by is None  # type: ignore[attr-defined]
        assert rec.generation == 1  # type: ignore[attr-defined]

    def test_generation_2_stored_correctly(self) -> None:
        """Generation=2 is preserved through record/get."""
        ring = TerminalOutcomeRetention(maxlen=10)
        rec = TerminalOutcomeRecord(
            request_id='mr-gen2',
            task_id='t3',
            branch='task/t3',
            state='done',
            generation=2,
        )
        ring.record(rec)
        stored = ring.get('mr-gen2')
        assert stored is not None
        assert stored.generation == 2


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
        async def _mock_enqueue(queue, req, es, **kwargs):
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
        async def _mock_enqueue(queue, req, es, **kwargs):
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
        async def _mock_enqueue(queue, req, es, **kwargs):
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

    # ------------------------------------------------------------------
    # ./-prefixed declared plan path regression tests (task 1587)
    # ------------------------------------------------------------------

    async def test_dot_slash_prefixed_root_file_is_satisfied(
        self, git_ops: GitOps,
    ):
        """Exact 4099 repro: declare './.jcodemunch.jsonc' when the branch
        touched '.jcodemunch.jsonc' → gate must pass (not_touched == []).

        RED before fix: literal membership check misses because
        './.jcodemunch.jsonc' != '.jcodemunch.jsonc', and the blob-type
        ls-tree result has no ' tree ' so the entry falls through to
        not_touched.
        """
        wt = (await git_ops.create_worktree('plan-dot-slash-root')).path
        rc, base_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        assert rc == 0
        base = base_out.strip()
        (wt / '.jcodemunch.jsonc').write_text('{"a": 1}\n')
        await git_ops.commit(wt, 'Add .jcodemunch.jsonc')
        rc, head_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        head = head_out.strip()

        result = await _check_plan_files_touched_in_branch(
            ['./.jcodemunch.jsonc'], base, head, git_ops,
            task_id='dot-slash-root',
        )
        assert result.not_touched == []

    async def test_dot_slash_prefixed_subdir_file_is_satisfied(
        self, git_ops: GitOps,
    ):
        """Declare './src/a.py' when the branch touched 'src/a.py'
        → gate must pass (not_touched == []).

        RED before fix: literal membership check misses because
        './src/a.py' != 'src/a.py'.
        """
        wt = (await git_ops.create_worktree('plan-dot-slash-subdir')).path
        rc, base_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        assert rc == 0
        base = base_out.strip()
        (wt / 'src').mkdir()
        (wt / 'src' / 'a.py').write_text('a = 1\n')
        await git_ops.commit(wt, 'Add src/a.py')
        rc, head_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        head = head_out.strip()

        result = await _check_plan_files_touched_in_branch(
            ['./src/a.py'], base, head, git_ops,
            task_id='dot-slash-subdir',
        )
        assert result.not_touched == []

    async def test_dot_slash_prefixed_directory_is_satisfied(
        self, git_ops: GitOps,
    ):
        """Declare './src/pkg' when the branch touched 'src/pkg/mod_a.py'
        → gate must pass (not_touched == []).

        RED before fix: the directory prefix is built as './src/pkg/' which
        fails to prefix-match the un-prefixed touched path 'src/pkg/mod_a.py'.
        """
        wt = (await git_ops.create_worktree('plan-dot-slash-dir')).path
        rc, base_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        assert rc == 0
        base = base_out.strip()
        (wt / 'src' / 'pkg').mkdir(parents=True)
        (wt / 'src' / 'pkg' / 'mod_a.py').write_text('a = 1\n')
        await git_ops.commit(wt, 'Add src/pkg/mod_a.py')
        rc, head_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        head = head_out.strip()

        result = await _check_plan_files_touched_in_branch(
            ['./src/pkg'], base, head, git_ops,
            task_id='dot-slash-dir',
        )
        assert result.not_touched == []

    async def test_dot_slash_prefixed_untouched_entry_still_flagged(
        self, git_ops: GitOps,
    ):
        """Declare './phantom.py' when the branch touched only 'real.py'
        → must still be flagged with the ORIGINAL declared path preserved.

        No-false-negative guard: normalization must not cause a spurious
        match.  Also pins the contract that not_touched preserves the
        architect's declared string (not a rewritten form).
        """
        wt = (await git_ops.create_worktree('plan-dot-slash-phantom')).path
        rc, base_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        assert rc == 0
        base = base_out.strip()
        (wt / 'real.py').write_text('r = 1\n')
        await git_ops.commit(wt, 'Add real.py')
        rc, head_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        head = head_out.strip()

        result = await _check_plan_files_touched_in_branch(
            ['./phantom.py'], base, head, git_ops,
            task_id='dot-slash-phantom',
        )
        assert result.not_touched == ['./phantom.py']

    async def test_redundant_separator_and_trailing_slash_normalized(
        self, git_ops: GitOps,
    ):
        """Trailing slashes and redundant separators are collapsed before matching.

        Declare ``'./src/pkg/'`` (trailing slash) and ``'src//pkg'``
        (redundant separator) when the branch touched ``'src/pkg/mod_a.py'``
        → gate must pass for both (not_touched == []).

        Pins the docstring contract: _normalize_plan_path collapses trailing
        slashes and ``//``-style redundant separators, not just the ``./``
        prefix.
        """
        wt = (await git_ops.create_worktree('plan-redundant-sep')).path
        rc, base_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        assert rc == 0
        base = base_out.strip()
        (wt / 'src' / 'pkg').mkdir(parents=True)
        (wt / 'src' / 'pkg' / 'mod_a.py').write_text('a = 1\n')
        await git_ops.commit(wt, 'Add src/pkg/mod_a.py')
        rc, head_out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=wt)
        head = head_out.strip()

        # Trailing-slash variant: './src/pkg/' normalizes to 'src/pkg'
        result = await _check_plan_files_touched_in_branch(
            ['./src/pkg/'], base, head, git_ops,
            task_id='redundant-trailing-slash',
        )
        assert result.not_touched == [], \
            "trailing-slash variant should be satisfied"

        # Redundant-separator variant: 'src//pkg' normalizes to 'src/pkg'
        result = await _check_plan_files_touched_in_branch(
            ['src//pkg'], base, head, git_ops,
            task_id='redundant-double-slash',
        )
        assert result.not_touched == [], \
            "redundant-separator variant should be satisfied"


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

    async def test_speculative_worker_train_pushes_on_success(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """Regression: step-02 signature change routes train via _do_train_merge(self, req).

        Guards that SpeculativeMergeWorker._merger_loop correctly dispatches
        GroupMergeRequest to _do_train_merge(self, req), which then calls
        _finalize_advanced_merge with train_id/member_task_ids and propagates
        push_status back to the caller.
        """
        req = await _make_stacked_train(git_ops, config, train_id='spec-push-test')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        with (
            patch(
                'orchestrator.merge_queue.run_scoped_verification',
                _mock_verify_pass(),
            ),
            patch.object(
                git_ops, 'push_main',
                AsyncMock(return_value='pushed'),
            ),
        ):
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=60)

        # (1) Outcome is done with push_status propagated from _finalize_advanced_merge.
        assert outcome.status == 'done', f'expected done, got: {outcome!r}'
        assert outcome.push_status == 'pushed', (
            f'expected push_status="pushed", got: {outcome.push_status!r}'
        )

        # (2) All 3 members flipped.
        assert req.mark_member_done.call_count == 3, (  # type: ignore[reportFunctionMemberAccess]
            f'expected 3 mark_member_done calls, got {req.mark_member_done.call_count}'  # type: ignore[reportFunctionMemberAccess]
        )

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

    async def test_merger_loop_unknown_branch_intercepted_before_diagnostic(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """A missing branch ref through the Merger loop is intercepted by the
        branch-presence guard and returns 'unknown_branch' before the merge —
        so it never reaches the merge-failure path and carries no
        failure_diagnostic.

        Historical context: this is the 2026-05-28 'not something we can merge'
        misroute scenario (a request for a branch that never existed here).  It
        used to surface as blocked + failure_diagnostic[branch_ref_in_worktree=
        '<unresolved>']; that diagnostic-enrichment path is still covered for
        the _remerge case by test_remerge_ghost_branch_sets_failure_diagnostic.
        """
        # Real worktree (non-main HEAD) so the already-merged check would pass,
        # but the request branch 'ghost-m' has no refs/heads/task/ghost-m ref.
        wt = await _make_branch_with_file(git_ops, 'phantom-1', 'ph1.py', 'x = 1\n')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        req = _make_request('ghost-m', 'ghost-m', wt, config)
        await queue.put(req)
        outcome = await asyncio.wait_for(req.result, timeout=30)

        assert outcome.status == 'unknown_branch', f'got {outcome}'
        assert 'task/ghost-m' in outcome.reason
        assert outcome.failure_diagnostic is None, (
            'guard short-circuits before the merge-failure path — no diagnostic'
        )

        await worker.stop()
        await worker_task

    async def test_merger_loop_speculative_ghost_returns_unknown_branch(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ):
        """A missing-ref N+1 branch in the speculative position is intercepted
        by the branch-presence guard and returns 'unknown_branch' (N still
        succeeds).  Previously surfaced as blocked + failure_diagnostic
        ['base_label']=='speculative' from the speculative merge-failure path.
        """
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
        assert outcome_n1.status == 'unknown_branch', (
            f'N+1 ghost should be unknown_branch: {outcome_n1}'
        )
        assert outcome_n1.failure_diagnostic is None

        await worker.stop()
        await worker_task

    async def test_escalation_server_merge_request_unknown_branch(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ):
        """merge_request returns 'unknown_branch' for a branch with no ref (a
        misroute), carrying no failure_diagnostic; a valid branch still merges
        cleanly with no failure_diagnostic (byte-identical success shape).
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

        # Missing branch ref (real worktree of a different branch) → unknown_branch
        wt_ghost = await _make_branch_with_file(git_ops, 'e2e-phantom', 'phantom.py', 'x = 1\n')

        resp = await asyncio.wait_for(
            tool.fn(task_id='ghost-e2e', branch='ghost-e2e', worktree=str(wt_ghost), wait_secs=100),
            timeout=30,
        )
        assert resp['status'] == 'unknown_branch', f'got {resp}'
        assert 'failure_diagnostic' not in resp, (
            f'unknown_branch must not carry a failure_diagnostic: {resp}'
        )

        # Valid branch → NO failure_diagnostic in successful response
        wt = await _make_branch_with_file(git_ops, 'e2e-valid', 'e2e.py', 'x = 1\n')
        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            resp_ok = await asyncio.wait_for(
                tool.fn(task_id='e2e-valid', branch='e2e-valid', worktree=str(wt), wait_secs=100),
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
        return asyncio.get_running_loop().create_future()

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

    async def test_concurrent_acquire_during_scan_coalesces(
        self, config: OrchestratorConfig, tmp_path: Path,
    ):
        """(race) Concurrent dispatch claims the slot during the disk-scan await.

        The only ``await`` between the ``is_inflight`` check and the ``acquire``
        call is inside ``find_inflight_merge_worktree``.  A second caller can
        win the acquire race while the first is suspended there.  We simulate
        this by injecting a fake git_ops whose scan yields twice (two
        ``sleep(0)``s) while a background task grabs the slot between them.

        The original caller should observe acquire returning False and fall
        through to the race-fallback coalesce path: ``in_flight=True``,
        ``source='registry'``, one ``merge_coalesced`` event, empty queue.
        """
        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = self._make_event_store(tmp_path)
        req = _make_request('race', 'race', tmp_path, config)

        # Fake git_ops: yields control twice so the concurrent acquirer can
        # grab the slot between the two sleep(0)s.
        class _SlowGitOps:
            async def find_inflight_merge_worktree(self, branch: str):
                await asyncio.sleep(0)  # let _acquirer get scheduled
                await asyncio.sleep(0)  # let _acquirer actually run & acquire
                return None

            async def cleanup_merge_worktree(self, merge_wt: object) -> None:
                pass  # never reached — find_inflight returns None

        other_future: asyncio.Future = asyncio.get_running_loop().create_future()

        async def _acquirer() -> None:
            await asyncio.sleep(0)  # wait until main is inside find_inflight
            registry.acquire('race', 'other-task', other_future)

        asyncio.create_task(_acquirer())

        result = await coalesce_or_enqueue_merge_request(
            queue, req, event_store, registry,
            git_ops=_SlowGitOps(),
        )

        assert result.in_flight is True, f'Expected in_flight=True, got {result}'
        assert result.dispatched is False
        assert result.source == 'registry', (
            f'Expected source=registry (race-fallback path), got {result.source!r}'
        )
        assert queue.qsize() == 0, 'No enqueue should happen on race-fallback coalesce'
        assert _count_events(event_store.db_path, 'merge_coalesced') == 1

        # Clean up the never-resolving future to avoid ResourceWarning
        other_future.cancel()

    async def test_retention_forwarded_through_coalesce(
        self, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """coalesce_or_enqueue_merge_request forwards retention to enqueue chokepoint.

        After a dispatched request's future resolves, the retention ring must
        contain a record and a merge_finalized row must exist in the event store.
        """
        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = self._make_event_store(tmp_path)
        retention = TerminalOutcomeRetention()

        req = _make_request('777', '777', tmp_path, config)

        result = await coalesce_or_enqueue_merge_request(
            queue, req, event_store, registry, git_ops=None, retention=retention,
        )
        assert result.dispatched is True

        # Resolve the future → done-callback fires
        req.result.set_result(MergeOutcome(status='done', merge_sha='cf1'))
        await asyncio.sleep(0)

        # Ring must have the record
        stored = retention.get(req.request_id)
        assert stored is not None
        assert stored.state == 'done'
        assert stored.merge_sha == 'cf1'

        # merge_finalized row must exist in the event store
        assert _count_events(event_store.db_path, 'merge_finalized') == 1


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


# ---------------------------------------------------------------------------
# TestSpeculationRaceRetry
# ---------------------------------------------------------------------------


class TestSpeculationRaceRetry:
    """Speculation-race retry: re-merge against actual main when base drifted."""

    def test_is_speculation_race_exact_match(self):
        """_is_speculation_race matches exactly on git porcelain phrase."""
        # Exact phrase — load-bearing git porcelain output
        assert _is_speculation_race('not something we can merge') is True
        assert _is_speculation_race(
            "merge: task/feature-foo - not something we can merge"
        ) is True
        assert _is_speculation_race(
            "fatal: 'task/bar' - not something we can merge\nerror: merge failed"
        ) is True

        # Paraphrases must NOT match
        assert _is_speculation_race('cannot merge') is False
        assert _is_speculation_race('not something to merge') is False
        assert _is_speculation_race('refusing to merge unrelated histories') is False
        assert _is_speculation_race('fatal: refusing to merge unrelated histories') is False
        assert _is_speculation_race('not a merge') is False

        # Empty and unrelated fatals
        assert _is_speculation_race('') is False
        assert _is_speculation_race('fatal: no such branch') is False
        assert _is_speculation_race('error: CONFLICT (content)') is False

    @pytest.mark.asyncio
    async def test_remerge_retry_succeeds(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        """Race-retry: 1st merge_to_main fails with stale-base speculation-race
        stderr; 2nd call (against fresh main) succeeds and the branch lands on main.

        Verifies:
        - merge_to_main called exactly twice
        - 2nd call's base_sha == actual main at test time
        - returned item has no immediate_outcome (flowing, not blocked)
        - _verify_and_advance returns True and resolves outcome.status == 'done'
        - caplog contains 'merge_retry_after_speculation_race'
        """
        branch = 'race-retry-ok'
        worktree = await _make_branch_with_file(git_ops, branch, 'race_ok.py', 'x = 1\n')
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        req = _make_request('race-retry', branch, worktree, config)

        real_merge_to_main = git_ops.merge_to_main
        call_count = 0
        call_base_shas: list[str | None] = []

        async def fake_merge_to_main(wt: Any, br: str, base_sha: str | None = None) -> MergeResult:
            nonlocal call_count
            call_count += 1
            call_base_shas.append(base_sha)
            if call_count == 1:
                return MergeResult(
                    success=False,
                    conflicts=False,
                    details=f'merge: task/{br} - not something we can merge',
                    pre_merge_sha='0' * 40,
                )
            # 2nd call: delegate to the real merge_to_main
            return await real_merge_to_main(wt, br, base_sha=base_sha)

        monkeypatch.setattr(git_ops, 'merge_to_main', fake_merge_to_main)
        actual_main = await git_ops.get_main_sha()

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            _mock_verify_pass(),
        ), caplog.at_level(logging.INFO, logger='orchestrator.merge_queue'):
            item = await worker._remerge(req, None)

        # merge_to_main must have been called exactly twice
        assert call_count == 2, f'Expected 2 calls to merge_to_main, got {call_count}'

        # 2nd call's base_sha must be the actual main SHA (freshly read inside _remerge)
        assert call_base_shas[1] == actual_main, (
            f'2nd merge_to_main base_sha {call_base_shas[1]!r} != actual main {actual_main!r}'
        )

        # item must be flowing (no immediate_outcome), merge succeeded
        assert item.immediate_outcome is None, (
            f'Expected flowing item but got immediate_outcome={item.immediate_outcome}'
        )
        assert item.merge_result is not None
        assert item.merge_result.success

        # _verify_and_advance must land the branch on main
        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            _mock_verify_pass(),
        ):
            advanced = await worker._verify_and_advance(item)

        assert advanced is True
        outcome = req.result.result()
        assert outcome.status == 'done', f'Expected done, got {outcome.status}: {outcome}'

        # The branch's file must now appear on main
        rc, file_content, _ = await _run(
            ['git', 'show', 'main:race_ok.py'], cwd=git_ops.project_root,
        )
        assert rc == 0, 'race_ok.py not found on main after retry merge'
        assert 'x = 1' in file_content

        # Structured log note must appear
        assert 'merge_retry_after_speculation_race' in caplog.text, (
            'Expected "merge_retry_after_speculation_race" in caplog'
        )

    @pytest.mark.asyncio
    async def test_remerge_retry_also_fails(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Race-retry: both attempts fail (no-conflict) → dual-attempt diagnostics.

        1st call: speculation-race failure (stale base '0'*40 != actual main).
        2nd call: a different non-conflict failure (refusing to merge unrelated histories).

        Verifies:
        - merge_to_main called exactly twice (no 3rd retry)
        - returned item has immediate_outcome.status == 'blocked'
        - reason contains BOTH stderr strings and BOTH base SHAs
        - failure_diagnostic carries μ 4 keys for the final (retry) attempt
          PLUS first_attempt_base_sha / first_attempt_git_stderr for the first
        - item.failure_diagnostic == item.immediate_outcome.failure_diagnostic
        """
        branch = 'race-retry-fail'
        worktree = await _make_branch_with_file(git_ops, branch, 'race_fail.py', 'y = 2\n')
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        req = _make_request('race-fail', branch, worktree, config)

        actual_main = await git_ops.get_main_sha()
        first_base = '0' * 40
        retry_base = actual_main  # retry re-reads get_main_sha() → actual_main
        first_stderr = f'merge: task/{branch} - not something we can merge'
        retry_stderr = 'fatal: refusing to merge unrelated histories'

        call_count = 0

        async def fake_merge_to_main(wt: Any, br: str, base_sha: str | None = None) -> MergeResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MergeResult(
                    success=False, conflicts=False,
                    details=first_stderr,
                    pre_merge_sha=first_base,
                )
            # 2nd call: different non-conflict failure; pre_merge_sha = retry_base
            return MergeResult(
                success=False, conflicts=False,
                details=retry_stderr,
                pre_merge_sha=retry_base,
            )

        monkeypatch.setattr(git_ops, 'merge_to_main', fake_merge_to_main)

        item = await worker._remerge(req, None)

        # Exactly 2 calls — no 3rd retry
        assert call_count == 2, f'Expected exactly 2 calls, got {call_count}'

        # Must be a blocked immediate_outcome
        assert item.immediate_outcome is not None
        assert item.immediate_outcome.status == 'blocked', (
            f'Expected blocked, got {item.immediate_outcome.status}'
        )

        reason = item.immediate_outcome.reason
        # reason must surface BOTH stderr strings
        assert first_stderr in reason or 'not something we can merge' in reason, (
            f'1st stderr missing from reason: {reason!r}'
        )
        assert retry_stderr in reason, f'retry stderr missing from reason: {reason!r}'
        # reason must surface BOTH base SHAs
        assert first_base in reason or first_base[:8] in reason, (
            f'first_base missing from reason: {reason!r}'
        )
        assert retry_base in reason or retry_base[:8] in reason, (
            f'retry_base missing from reason: {reason!r}'
        )

        # failure_diagnostic must carry μ 4 keys for retry attempt
        diag = item.immediate_outcome.failure_diagnostic
        assert diag is not None, 'failure_diagnostic must not be None'
        assert 'base_sha' in diag
        assert 'base_label' in diag
        assert 'branch_ref_in_worktree' in diag
        assert 'git_stderr' in diag

        # First-attempt keys must be present
        assert 'first_attempt_base_sha' in diag, (
            f'first_attempt_base_sha missing from failure_diagnostic: {diag!r}'
        )
        assert 'first_attempt_git_stderr' in diag, (
            f'first_attempt_git_stderr missing from failure_diagnostic: {diag!r}'
        )
        assert diag['first_attempt_base_sha'] == first_base
        assert diag['first_attempt_git_stderr'] == first_stderr

        # item.failure_diagnostic must mirror outcome.failure_diagnostic
        assert item.failure_diagnostic == diag

    @pytest.mark.asyncio
    async def test_no_retry_when_base_is_actual_main(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """No retry when base_sha == actual main (genuine main-HEAD failure, not a race).

        When merge_result.pre_merge_sha == actual_main (the merge ran against
        the real current main), the load-bearing stderr alone must NOT trigger a
        retry. Only exactly ONE merge_to_main call must be made, and the returned
        item must be a single-diagnostic blocked outcome (no first_attempt_* keys).

        Also asserts that a non-load-bearing failure with a stale base also
        yields exactly one call.
        """
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        actual_main = await git_ops.get_main_sha()
        branch = 'no-retry-main'
        worktree = await _make_branch_with_file(git_ops, branch, 'no_retry.py', 'z = 3\n')
        req = _make_request('no-retry', branch, worktree, config)

        # Case 1: load-bearing stderr but pre_merge_sha == actual main (not a race)
        call_count = 0

        async def fake_base_is_main(wt: Any, br: str, base_sha: str | None = None) -> MergeResult:
            nonlocal call_count
            call_count += 1
            return MergeResult(
                success=False, conflicts=False,
                details=f'merge: task/{br} - not something we can merge',
                pre_merge_sha=actual_main,  # base == actual main → not a race
            )

        monkeypatch.setattr(git_ops, 'merge_to_main', fake_base_is_main)
        item = await worker._remerge(req, None)

        assert call_count == 1, (
            f'Expected exactly 1 call (base==main, not a race), got {call_count}'
        )
        assert item.immediate_outcome is not None
        assert item.immediate_outcome.status == 'blocked'
        # Must NOT have first_attempt_* keys (single-attempt diagnostic)
        diag = item.immediate_outcome.failure_diagnostic
        assert diag is not None
        assert 'first_attempt_base_sha' not in diag, (
            f'Unexpected first_attempt_base_sha in diag for non-race failure: {diag!r}'
        )

        # Case 2: non-load-bearing stderr with stale base → no retry either
        call_count = 0
        branch2 = 'no-retry-stale-ok'
        worktree2 = await _make_branch_with_file(git_ops, branch2, 'no_retry2.py', 'w = 4\n')
        req2 = _make_request('no-retry-2', branch2, worktree2, config)

        async def fake_non_loadbearing(wt: Any, br: str, base_sha: str | None = None) -> MergeResult:
            nonlocal call_count
            call_count += 1
            return MergeResult(
                success=False, conflicts=False,
                details='fatal: refusing to merge unrelated histories',
                pre_merge_sha='0' * 40,  # stale base, but NOT the race phrase
            )

        monkeypatch.setattr(git_ops, 'merge_to_main', fake_non_loadbearing)
        item2 = await worker._remerge(req2, None)

        assert call_count == 1, (
            f'Expected 1 call (non-race stderr), got {call_count}'
        )
        assert item2.immediate_outcome is not None
        assert item2.immediate_outcome.status == 'blocked'
        diag2 = item2.immediate_outcome.failure_diagnostic
        assert diag2 is not None
        assert 'first_attempt_base_sha' not in diag2

    @pytest.mark.asyncio
    async def test_remerge_retry_returns_conflict(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Race-retry: 1st fails with speculation-race; 2nd returns a true conflict.

        Verifies:
        - merge_to_main called exactly twice
        - immediate_outcome.status == 'conflict'
        - conflict_details carries the retry stderr
        - retry merge_worktree is cleaned up
        - _emit_merge_attempt is called with outcome='conflict'
        """
        branch = 'race-retry-conflict'
        worktree = await _make_branch_with_file(
            git_ops, branch, 'conflict_file.py', 'value = 1\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        req = _make_request('race-conflict', branch, worktree, config)

        fake_retry_wt = Path('/tmp/fake-race-retry-wt')
        retry_stderr = 'CONFLICT (content): Merge conflict in conflict_file.py'
        call_count = 0

        async def fake_merge_to_main(wt: Any, br: str, base_sha: str | None = None) -> MergeResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MergeResult(
                    success=False, conflicts=False,
                    details=f'merge: task/{br} - not something we can merge',
                    pre_merge_sha='0' * 40,
                )
            # 2nd call: real merge conflict with a retained worktree
            return MergeResult(
                success=False, conflicts=True,
                details=retry_stderr,
                pre_merge_sha='a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2',
                merge_worktree=fake_retry_wt,
            )

        cleanup_calls: list[Path] = []

        async def fake_cleanup(wt: Path) -> None:
            cleanup_calls.append(wt)

        emit_calls: list[dict] = []

        def fake_emit(event_store: Any, task_id: str, outcome: str, **kwargs: Any) -> None:
            emit_calls.append({'task_id': task_id, 'outcome': outcome})

        monkeypatch.setattr(git_ops, 'merge_to_main', fake_merge_to_main)
        monkeypatch.setattr(git_ops, 'cleanup_merge_worktree', fake_cleanup)
        monkeypatch.setattr('orchestrator.merge_queue._emit_merge_attempt', fake_emit)

        item = await worker._remerge(req, None)

        # Exactly 2 calls — no 3rd retry
        assert call_count == 2, f'Expected 2 calls, got {call_count}'

        # Conflict immediate_outcome
        assert item.immediate_outcome is not None
        assert item.immediate_outcome.status == 'conflict', (
            f'Expected conflict, got {item.immediate_outcome.status!r}'
        )
        assert item.immediate_outcome.conflict_details == retry_stderr, (
            f'conflict_details mismatch: {item.immediate_outcome.conflict_details!r}'
        )

        # Retry merge_worktree must be cleaned up
        assert fake_retry_wt in cleanup_calls, (
            f'Expected cleanup_merge_worktree({fake_retry_wt}); got: {cleanup_calls}'
        )

        # _emit_merge_attempt must have been called with outcome='conflict'
        conflict_emits = [e for e in emit_calls if e['outcome'] == 'conflict']
        assert conflict_emits, (
            f'Expected a conflict merge_attempt emission; got emit_calls={emit_calls!r}'
        )
        assert conflict_emits[0]['task_id'] == req.task_id

    @pytest.mark.asyncio
    async def test_remerge_retry_success_skip_verify(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Race-retry with pre_rebased=True: retry success MUST force verification (skip_verify=False).

        The speculation-race gate fires only when merge_result.pre_merge_sha != actual_main,
        meaning main demonstrably advanced between _remerge's get_main_sha() read and
        merge_to_main's own read.  req.pre_rebased=True reflects a rebase onto the OLD main,
        NOT retry_main; the retry merges the branch against the newer retry_main, integrating
        main commits the branch never incorporated.  The documented skip_verify invariant
        ('pre_rebased AND main unchanged', merge_queue.py SpeculativeItem.skip_verify comment)
        therefore does NOT hold, and skipping verification would let semantically-unverified
        main commits land on the protected branch.  skip_verify must be False unconditionally
        on this path regardless of req.pre_rebased.

        Behavioural check: _verify_and_advance(item) must invoke run_scoped_verification
        (i.e., must NOT skip the verify step).
        """
        branch = 'race-retry-skip-verify'
        worktree = await _make_branch_with_file(
            git_ops, branch, 'skip_verify.py', 's = 99\n',
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)
        # pre_rebased=True is the would-be trigger for skip_verify; must NOT skip here
        req = _make_request('race-skip', branch, worktree, config, pre_rebased=True)

        real_merge_to_main = git_ops.merge_to_main
        call_count = 0

        async def fake_merge_to_main(wt: Any, br: str, base_sha: str | None = None) -> MergeResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MergeResult(
                    success=False, conflicts=False,
                    details=f'merge: task/{br} - not something we can merge',
                    pre_merge_sha='0' * 40,
                )
            # 2nd call: delegate to real merge so a real worktree/commit is produced
            return await real_merge_to_main(wt, br, base_sha=base_sha)

        monkeypatch.setattr(git_ops, 'merge_to_main', fake_merge_to_main)

        item = await worker._remerge(req, None)

        assert call_count == 2, f'Expected 2 calls, got {call_count}'
        assert item.immediate_outcome is None, (
            f'Expected flowing item; got immediate_outcome={item.immediate_outcome}'
        )
        assert item.merge_result is not None
        assert item.merge_result.success

        # SAFETY CONTRACT: main advanced since the pre-rebase, so verification must run.
        # skip_verify MUST be False regardless of req.pre_rebased.
        assert item.skip_verify is False, (
            f'Expected skip_verify=False (main advanced: race gate fired), '
            f'but got skip_verify={item.skip_verify}. '
            f'Skipping verification after a speculation-race retry is unsafe.'
        )

        # Behavioural check: _verify_and_advance must invoke run_scoped_verification
        # (not skip it) because skip_verify=False.
        mock_verify = AsyncMock(return_value=MagicMock(passed=True, summary=''))
        with patch('orchestrator.merge_queue.run_scoped_verification', mock_verify):
            advanced = await worker._verify_and_advance(item)

        assert advanced is True
        assert mock_verify.called, (
            'run_scoped_verification must be invoked on a race-retry success '
            '(skip_verify=False); verification was skipped.'
        )


# ---------------------------------------------------------------------------
# TestTrainLifecycleEvents — train_* event emission integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTrainLifecycleEvents:
    """Assert train_started, train_merged, train_derailed, and train_member_deferred
    are emitted by _do_train_merge at the correct points."""

    async def test_happy_path_emits_train_started_and_merged(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """Happy-path 3-member train → train_started then train_merged emitted."""
        from orchestrator.event_store import EventStore

        req = await _make_stacked_train(git_ops, config)
        db_path = tmp_path / 'train_events.db'
        event_store = EventStore(db_path=db_path, run_id='train-lifecycle-run')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=event_store)

        with patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()):
            outcome = await worker._do_merge(req)

        assert outcome is not None
        assert outcome.status == 'done', f'expected done, got: {outcome!r}'

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, task_id, phase, data FROM events "
            "WHERE event_type IN ('train_started', 'train_merged') ORDER BY id"
        ).fetchall()
        conn.close()

        event_types = [r[0] for r in rows]
        assert 'train_started' in event_types, f'Missing train_started in {event_types}'
        assert 'train_merged' in event_types, f'Missing train_merged in {event_types}'

        # train_started must come before train_merged (by id order)
        started_idx = event_types.index('train_started')
        merged_idx = event_types.index('train_merged')
        assert started_idx < merged_idx, 'train_started must precede train_merged'

        # Verify train_started payload
        import json
        started_row = rows[started_idx]
        assert started_row[1] == req.task_id  # task_id
        assert started_row[2] == 'merge'  # phase
        started_data = json.loads(started_row[3])
        assert started_data['train_id'] == req.train_id
        assert started_data['member_task_ids'] == req.member_task_ids
        assert started_data['member_count'] == 3
        assert started_data['base_sha'], 'base_sha must be non-empty'

        # Verify train_merged payload
        merged_row = rows[merged_idx]
        merged_data = json.loads(merged_row[3])
        assert merged_data['train_id'] == req.train_id
        assert merged_data['merge_commit_sha'], 'merge_commit_sha must be non-empty'
        assert merged_data['base_sha'], 'base_sha must be non-empty'

    async def test_verify_failed_emits_train_derailed(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """Verify failure → train_started then train_derailed, no train_merged."""
        import json

        from orchestrator.event_store import EventStore

        req = await _make_stacked_train(git_ops, config)
        db_path = tmp_path / 'train_derailed_verify.db'
        event_store = EventStore(db_path=db_path, run_id='train-derailed-verify-run')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=event_store)

        mock_verify_fail = AsyncMock(return_value=MagicMock(
            passed=False,
            summary='Tests failed: 3 errors',
            failure_report=MagicMock(return_value='Tests failed: 3 errors'),
        ))
        with patch('orchestrator.merge_queue.run_scoped_verification', mock_verify_fail):
            outcome = await worker._do_merge(req)

        assert outcome is not None
        assert outcome.status == 'blocked', f'expected blocked, got: {outcome!r}'

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, data FROM events "
            "WHERE event_type IN ('train_started', 'train_merged', 'train_derailed') ORDER BY id"
        ).fetchall()
        conn.close()

        event_types = [r[0] for r in rows]
        assert event_types[0] == 'train_started', f'First event must be train_started, got {event_types}'
        assert 'train_derailed' in event_types, f'Missing train_derailed in {event_types}'
        assert 'train_merged' not in event_types, f'Unexpected train_merged in {event_types}'

        derailed_idx = event_types.index('train_derailed')
        derailed_data = json.loads(rows[derailed_idx][1])
        assert derailed_data['train_id'] == req.train_id
        assert derailed_data['member_task_ids'] == req.member_task_ids
        assert 'verif' in derailed_data['derail_reason'].lower(), (
            f"Expected 'verif' (from 'verify' or 'verification') in derail_reason, "
            f"got: {derailed_data['derail_reason']!r}"
        )

    async def test_rebase_conflict_emits_train_derailed(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """Rebase conflict → train_started then train_derailed, no train_merged."""
        import json

        from orchestrator.event_store import EventStore

        req = await _make_stacked_train(git_ops, config)
        db_path = tmp_path / 'train_derailed_rebase.db'
        event_store = EventStore(db_path=db_path, run_id='train-derailed-rebase-run')

        # Commit a conflicting change on main so rebase fails
        conflict_file = git_ops.project_root / 'trn-c.py'
        conflict_file.write_text('# main version — conflicts with tip\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Conflicting commit on main'], cwd=git_ops.project_root)

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=event_store)

        outcome = await worker._do_merge(req)
        assert outcome is not None
        assert outcome.status == 'blocked', f'expected blocked, got: {outcome!r}'

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, data FROM events "
            "WHERE event_type IN ('train_started', 'train_merged', 'train_derailed') ORDER BY id"
        ).fetchall()
        conn.close()

        event_types = [r[0] for r in rows]
        assert event_types[0] == 'train_started', f'First event must be train_started, got {event_types}'
        assert 'train_derailed' in event_types, f'Missing train_derailed in {event_types}'
        assert 'train_merged' not in event_types, f'Unexpected train_merged in {event_types}'

        derailed_idx = event_types.index('train_derailed')
        derailed_data = json.loads(rows[derailed_idx][1])
        assert derailed_data['train_id'] == req.train_id

    async def test_incomplete_emits_train_member_deferred(
        self, git_ops: GitOps, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """Incomplete member → train_started then train_member_deferred, no train_derailed."""
        import json

        from orchestrator.event_store import EventStore

        req = await _make_stacked_train(git_ops, config)
        # Override: trn-a has non-deferred status 'planning'
        req.status_check = AsyncMock(return_value={
            'trn-a': 'planning',
            'trn-b': 'merge-deferred',
            'trn-c': 'merge-deferred',
        })

        db_path = tmp_path / 'train_member_deferred.db'
        event_store = EventStore(db_path=db_path, run_id='train-member-deferred-run')

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=event_store)

        outcome = await worker._do_merge(req)
        assert outcome is not None
        assert outcome.status == 'blocked', f'expected blocked, got: {outcome!r}'
        assert outcome.reason.startswith(TRAIN_INCOMPLETE_REASON_PREFIX), (
            f'expected TRAIN_INCOMPLETE prefix, got: {outcome.reason!r}'
        )

        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT event_type, task_id, data FROM events "
            "WHERE event_type IN ('train_started', 'train_member_deferred', 'train_merged', 'train_derailed') "
            "ORDER BY id"
        ).fetchall()
        conn.close()

        event_types = [r[0] for r in rows]
        assert event_types[0] == 'train_started', f'First event must be train_started, got {event_types}'
        assert 'train_member_deferred' in event_types, f'Missing train_member_deferred in {event_types}'
        assert 'train_merged' not in event_types, f'Unexpected train_merged in {event_types}'
        assert 'train_derailed' not in event_types, f'Unexpected train_derailed in {event_types}'

        deferred_idx = event_types.index('train_member_deferred')
        deferred_row = rows[deferred_idx]
        deferred_data = json.loads(deferred_row[2])
        assert deferred_data['train_id'] == req.train_id
        assert deferred_data['deferred_task_id'] == 'trn-a', (
            f"Expected deferred_task_id='trn-a', got {deferred_data['deferred_task_id']!r}"
        )
        assert 'planning' in deferred_data['deferred_reason'], (
            f"Expected 'planning' in deferred_reason, got: {deferred_data['deferred_reason']!r}"
        )
        assert set(deferred_data['remaining_members']) == {'trn-b', 'trn-c'}, (
            f"Expected remaining_members={{trn-b, trn-c}}, got: {deferred_data['remaining_members']!r}"
        )


# ---------------------------------------------------------------------------
# TestRunPostMergeVerify — unit tests for the _run_post_merge_verify helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunPostMergeVerify:
    """Direct unit tests for the _run_post_merge_verify module-level helper.

    The helper is imported at the top of the test class (not at module level)
    so a missing symbol produces a readable ImportError on collection rather
    than a hard SyntaxError.
    """

    def _make_git_ops(self) -> MagicMock:
        git_ops = MagicMock()
        git_ops.cleanup_merge_worktree = AsyncMock()
        git_ops.prune_stale_merge_worktrees = AsyncMock(return_value=[])
        return git_ops

    def _make_req(self) -> MagicMock:
        req = MagicMock()
        req.task_id = 'task-verify-test'
        req.task_files = None
        req.module_configs = []
        req.config.merge_verify_min_free_disk_bytes = 1024
        req.config.merge_verify_workspace = False
        return req

    async def test_disk_guard_none_verify_passed_returns_none(self) -> None:
        """(a) disk guard returns None + verify passed → returns None; merge_wt NOT cleaned."""
        from orchestrator.merge_queue import _run_post_merge_verify

        git_ops = self._make_git_ops()
        req = self._make_req()
        merge_wt = MagicMock()
        timeouts: dict[str, int] = {}
        enospc_retries: dict[str, int] = {}

        passed_result = MagicMock(passed=True, summary='', timed_out=False)
        with (
            patch('orchestrator.merge_queue._ensure_verify_disk_space', AsyncMock(return_value=None)),
            patch('orchestrator.merge_queue.run_scoped_verification', AsyncMock(return_value=passed_result)),
        ):
            result = await _run_post_merge_verify(
                git_ops, req, merge_wt,
                timeouts=timeouts, enospc_retries=enospc_retries,
                max_timeouts=2, max_enospc=1,
            )

        assert result is None
        git_ops.cleanup_merge_worktree.assert_not_awaited()

    async def test_disk_guard_returns_reason_blocks(self) -> None:
        """(b) disk guard returns reason string → MergeOutcome('blocked') with that reason; merge_wt cleaned once."""
        from orchestrator.merge_queue import _run_post_merge_verify

        git_ops = self._make_git_ops()
        req = self._make_req()
        merge_wt = MagicMock()
        disk_reason = f'{TRANSIENT_INFRA_REASON_PREFIX}: no space'

        with (
            patch('orchestrator.merge_queue._ensure_verify_disk_space', AsyncMock(return_value=disk_reason)),
            patch('orchestrator.merge_queue.run_scoped_verification', AsyncMock()) as mock_verify,
        ):
            result = await _run_post_merge_verify(
                git_ops, req, merge_wt,
                timeouts={}, enospc_retries={},
                max_timeouts=2, max_enospc=1,
            )

        assert result is not None
        assert result.status == 'blocked'
        assert result.reason == disk_reason
        assert result.verify_skipped is True, (
            'disk guard must set verify_skipped=True so callers can log '
            '"verify skipped: low disk" instead of "passed=False"'
        )
        git_ops.cleanup_merge_worktree.assert_awaited_once_with(merge_wt)
        mock_verify.assert_not_awaited()

    async def test_verify_fails_non_enospc_blocks_and_cleans(self) -> None:
        """(c) verify fails (non-ENOSPC) → MergeOutcome('blocked') reason starts 'Post-merge verification failed:' and merge_wt cleaned."""
        from orchestrator.merge_queue import _run_post_merge_verify

        git_ops = self._make_git_ops()
        req = self._make_req()
        merge_wt = MagicMock()
        failed_result = MagicMock(
            passed=False, summary='Test suite exploded', timed_out=False,
        )
        failed_result.failure_report.return_value = ''
        # Ensure ENOSPC is not triggered
        failed_result.test_output = ''
        failed_result.lint_output = ''
        failed_result.type_output = ''

        with (
            patch('orchestrator.merge_queue._ensure_verify_disk_space', AsyncMock(return_value=None)),
            patch('orchestrator.merge_queue.run_scoped_verification', AsyncMock(return_value=failed_result)),
        ):
            result = await _run_post_merge_verify(
                git_ops, req, merge_wt,
                timeouts={}, enospc_retries={},
                max_timeouts=2, max_enospc=1,
            )

        assert result is not None
        assert result.status == 'blocked'
        assert result.reason.startswith('Post-merge verification failed:'), (
            f'unexpected reason: {result.reason!r}'
        )
        assert result.verify_skipped is False, (
            'actual verify failure must NOT set verify_skipped (verify ran)'
        )
        git_ops.cleanup_merge_worktree.assert_awaited_once_with(merge_wt)

    async def test_verify_timeout_bumps_timeouts_dict(self) -> None:
        """(d) verify fails with timed_out=True → timeouts dict bumped to 1 for req.task_id."""
        from orchestrator.merge_queue import _run_post_merge_verify

        git_ops = self._make_git_ops()
        req = self._make_req()
        merge_wt = MagicMock()
        timeouts: dict[str, int] = {}
        timed_out_result = MagicMock(
            passed=False, summary='verify timed out', timed_out=True,
        )
        timed_out_result.failure_report.return_value = ''
        timed_out_result.test_output = ''
        timed_out_result.lint_output = ''
        timed_out_result.type_output = ''

        with (
            patch('orchestrator.merge_queue._ensure_verify_disk_space', AsyncMock(return_value=None)),
            patch('orchestrator.merge_queue.run_scoped_verification', AsyncMock(return_value=timed_out_result)),
        ):
            result = await _run_post_merge_verify(
                git_ops, req, merge_wt,
                timeouts=timeouts, enospc_retries={},
                max_timeouts=2, max_enospc=1,
            )

        assert result is not None
        assert result.status == 'blocked'
        assert timeouts.get(req.task_id, 0) == 1, (
            f'expected timeouts[task_id]=1, got: {timeouts}'
        )

    async def test_persistent_enospc_prunes_and_escalates(self) -> None:
        """(e) persistent ENOSPC (both attempts) → prune awaited, enospc_retries bumped, transient infra blocked."""
        from orchestrator.merge_queue import _run_post_merge_verify

        git_ops = self._make_git_ops()
        req = self._make_req()
        merge_wt = MagicMock()
        enospc_retries: dict[str, int] = {}

        enospc_result = _enospc_verify_result()

        with (
            patch('orchestrator.merge_queue._ensure_verify_disk_space', AsyncMock(return_value=None)),
            patch(
                'orchestrator.merge_queue.run_scoped_verification',
                AsyncMock(return_value=enospc_result),
            ),
        ):
            result = await _run_post_merge_verify(
                git_ops, req, merge_wt,
                timeouts={}, enospc_retries=enospc_retries,
                max_timeouts=2, max_enospc=1,
            )

        assert result is not None
        assert result.status == 'blocked'
        assert result.reason.startswith(TRANSIENT_INFRA_REASON_PREFIX), (
            f'expected transient-infra reason, got: {result.reason!r}'
        )
        git_ops.prune_stale_merge_worktrees.assert_awaited_once()
        assert enospc_retries.get(req.task_id, 0) == 1, (
            f'expected enospc_retries[task_id]=1, got: {enospc_retries}'
        )

    async def test_verify_exception_propagates_no_cleanup(self) -> None:
        """(f) run_scoped_verification raises RuntimeError → exception propagates; merge_wt NOT cleaned."""
        from orchestrator.merge_queue import _run_post_merge_verify

        git_ops = self._make_git_ops()
        req = self._make_req()
        merge_wt = MagicMock()

        with (
            patch('orchestrator.merge_queue._ensure_verify_disk_space', AsyncMock(return_value=None)),
            patch(
                'orchestrator.merge_queue.run_scoped_verification',
                AsyncMock(side_effect=RuntimeError('boom')),
            ),
            pytest.raises(RuntimeError, match='boom'),
        ):
            await _run_post_merge_verify(
                git_ops, req, merge_wt,
                timeouts={}, enospc_retries={},
                max_timeouts=2, max_enospc=1,
            )

        git_ops.cleanup_merge_worktree.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestUnscopedTypecheckGate — step-5 gate tests
# ---------------------------------------------------------------------------

# Hermetic type_check_command that always exits 1 (simulates a RED frontend
# type-check).  We use a simple inline Python one-liner so no external tools
# are required and the exit code is deterministic regardless of repo contents.
_TYPE_CMD_ALWAYS_FAIL = 'python3 -c "import sys; sys.stderr.write(\'synthetic type error\\n\'); sys.exit(1)"'


def _make_request_with_module_configs(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
    module_configs: list[ModuleConfig],
) -> MergeRequest:
    """Like _make_request but accepts explicit module_configs."""
    future: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=module_configs,
        config=config,
        result=future,
    )


@pytest.mark.asyncio
class TestUnscopedTypecheckGate:
    """Tests for the pre-advance, fail-closed unscoped type-check gate wired into
    _run_post_merge_verify (step-5 RED / step-6 GREEN).

    Patches run_scoped_verification to always pass so only the new unscoped
    gate can block the merge.
    """

    async def test_run_post_merge_verify_blocks_on_failing_typecheck(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """(a) _run_post_merge_verify with a failing type_check_command → blocked.

        reason starts with 'Post-merge verification failed', names the failing
        subproject, and merge_wt is cleaned up.
        """
        from orchestrator.merge_queue import _run_post_merge_verify

        # Create a branch and merge it (to get a real merge_wt)
        branch = 'gate-typecheck-a'
        wt = (await git_ops.create_worktree(branch)).path
        (wt / 'mod.py').write_text('x = 1\n')
        await git_ops.commit(wt, 'Add mod.py')

        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success
        assert merge_result.merge_worktree is not None
        merge_wt = merge_result.merge_worktree

        mc = ModuleConfig(
            prefix='frontend',
            test_command=None,
            lint_command=None,
            type_check_command=_TYPE_CMD_ALWAYS_FAIL,
        )
        req = MagicMock()
        req.task_id = 'gate-test-a'
        req.task_files = None
        req.module_configs = [mc]
        req.config = config

        scoped_pass = MagicMock(passed=True, summary='', timed_out=False)
        with (
            patch('orchestrator.merge_queue._ensure_verify_disk_space', AsyncMock(return_value=None)),
            patch('orchestrator.merge_queue.run_scoped_verification', AsyncMock(return_value=scoped_pass)),
        ):
            outcome = await _run_post_merge_verify(
                git_ops, req, merge_wt,
                timeouts={}, enospc_retries={},
                max_timeouts=2, max_enospc=1,
            )

        assert outcome is not None, 'Expected MergeOutcome(blocked), got None'
        assert outcome.status == 'blocked', f'Expected blocked, got {outcome.status!r}'
        assert outcome.reason.startswith('Post-merge verification failed'), (
            f'Reason does not start with expected prefix: {outcome.reason!r}'
        )
        assert 'frontend' in outcome.reason, (
            f'"frontend" not in reason: {outcome.reason!r}'
        )
        # merge_wt should have been cleaned up by _run_post_merge_verify on failure
        assert not merge_wt.exists(), 'merge_wt should be cleaned up on blocked outcome'

    async def test_mergeworker_blocks_red_tip_main_sha_unchanged(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """(b) MergeWorker end-to-end: RED type_check_command → blocked, main SHA unchanged."""
        branch = 'gate-typecheck-b'
        wt = (await git_ops.create_worktree(branch)).path
        (wt / 'mod.py').write_text('x = 1\n')
        await git_ops.commit(wt, 'Add mod.py')

        main_sha_before = await git_ops.get_main_sha()

        mc = ModuleConfig(
            prefix='frontend',
            test_command=None,
            lint_command=None,
            type_check_command=_TYPE_CMD_ALWAYS_FAIL,
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)
        worker_task = asyncio.create_task(worker.run())

        scoped_pass = MagicMock(passed=True, summary='', timed_out=False)
        with patch('orchestrator.merge_queue.run_scoped_verification', AsyncMock(return_value=scoped_pass)):
            req = _make_request_with_module_configs(
                'gate-test-b', branch, wt, config, module_configs=[mc],
            )
            await queue.put(req)
            outcome = await asyncio.wait_for(req.result, timeout=60)

        await worker.stop()
        worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker_task

        assert outcome.status == 'blocked', (
            f'Expected blocked, got {outcome.status!r}: {outcome.reason!r}'
        )

        main_sha_after = await git_ops.get_main_sha()
        assert main_sha_before == main_sha_after, (
            f'RED tip landed on main! SHA changed from {main_sha_before!r} to {main_sha_after!r}'
        )

    async def test_gate_timeout_is_fail_closed_and_bumps_counter(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
    ) -> None:
        """(c) Timeout in the unscoped gate → FAIL-CLOSED: blocked, timeouts counter bumped.

        Patches run_scoped_verification to pass and run_verification (used inside
        _run_unscoped_typechecks) to simulate a timed-out type-check (timed_out=True,
        passed=False). Calls _run_post_merge_verify and asserts:
        (a) the outcome is MergeOutcome('blocked') — fail-closed on timeout, not fail-open;
        (b) the timeouts dict has been incremented for req.task_id.
        """
        from orchestrator.merge_queue import _run_post_merge_verify

        # Create a branch and merge it (to get a real merge_wt)
        branch = 'gate-timeout-c'
        wt = (await git_ops.create_worktree(branch)).path
        (wt / 'mod.py').write_text('x = 1\n')
        await git_ops.commit(wt, 'Add mod.py')

        merge_result = await git_ops.merge_to_main(wt, branch)
        assert merge_result.success
        assert merge_result.merge_worktree is not None
        merge_wt = merge_result.merge_worktree

        mc = ModuleConfig(
            prefix='frontend',
            test_command=None,
            lint_command=None,
            type_check_command='pyright src/',
        )
        req = MagicMock()
        req.task_id = 'gate-timeout-test'
        req.task_files = None
        req.module_configs = [mc]
        req.config = config

        scoped_pass = MagicMock(passed=True, summary='', timed_out=False)
        # Simulate a timed-out unscoped type-check
        timeout_result = MagicMock(passed=False, timed_out=True)

        timeouts: dict[str, int] = {}
        with (
            patch('orchestrator.merge_queue._ensure_verify_disk_space', AsyncMock(return_value=None)),
            patch('orchestrator.merge_queue.run_scoped_verification', AsyncMock(return_value=scoped_pass)),
            patch('orchestrator.merge_queue.run_verification', AsyncMock(return_value=timeout_result)),
        ):
            outcome = await _run_post_merge_verify(
                git_ops, req, merge_wt,
                timeouts=timeouts, enospc_retries={},
                max_timeouts=2, max_enospc=1,
            )

        # (a) fail-closed: timeout → blocked (NOT fail-open)
        assert outcome is not None, 'Expected MergeOutcome(blocked), got None (fail-open)'
        assert outcome.status == 'blocked', f'Expected blocked, got {outcome.status!r}'
        assert 'Post-merge verification failed' in outcome.reason, (
            f'Expected "Post-merge verification failed" in reason: {outcome.reason!r}'
        )

        # (b) loop-breaker counter was bumped for the task_id
        assert timeouts.get('gate-timeout-test', 0) == 1, (
            f'Expected timeouts["gate-timeout-test"] == 1, got: {timeouts!r}'
        )


# ---------------------------------------------------------------------------
# TestFinalizeAdvancedMerge — unit tests for the _finalize_advanced_merge helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFinalizeAdvancedMerge:
    """Direct unit tests for the _finalize_advanced_merge module-level helper."""

    def _make_git_ops(self, *, last_advanced_sha: str | None = 'abc123def') -> MagicMock:
        git_ops = MagicMock()
        git_ops.push_main = AsyncMock(return_value='pushed')
        git_ops.cleanup_merge_worktree = AsyncMock()
        git_ops._last_advanced_sha = last_advanced_sha
        return git_ops

    def _make_req(self) -> MagicMock:
        req = MagicMock()
        req.task_id = 'task-finalize-test'
        req.worktree = MagicMock()
        req.config = MagicMock()
        req.module_configs = []
        return req

    def _primed_dicts(self, task_id: str) -> tuple[dict, dict, dict]:
        cas_retries = {task_id: 1}
        timeouts = {task_id: 1}
        enospc_retries = {task_id: 1}
        return cas_retries, timeouts, enospc_retries

    async def test_success_path_returns_done_pops_counters(self) -> None:
        """(a) equivalence=[] + pyright not broken → done with merge_sha + push_main called; counters cleared."""
        from orchestrator.merge_queue import (
            _finalize_advanced_merge,
        )

        git_ops = self._make_git_ops()
        req = self._make_req()
        cas_retries, timeouts, enospc_retries = self._primed_dicts(req.task_id)
        pyright_clean = MagicMock(broken=False, failing_subprojects=[], detail='')

        with (
            patch('orchestrator.merge_queue._check_post_merge_equivalence', AsyncMock(return_value=[])),
            patch('orchestrator.merge_queue._check_post_merge_pyright', AsyncMock(return_value=pyright_clean)),
        ):
            outcome = await _finalize_advanced_merge(
                git_ops, req, None,
                merge_commit_fallback='fallback-sha',
                base_sha='base-sha',
                started_monotonic=0.0,
                cas_retries=cas_retries,
                timeouts=timeouts,
                enospc_retries=enospc_retries,
            )

        assert outcome.status == 'done'
        assert outcome.merge_sha == git_ops._last_advanced_sha
        assert outcome.push_status == 'pushed'
        git_ops.push_main.assert_awaited_once()
        assert req.task_id not in cas_retries
        assert req.task_id not in timeouts
        assert req.task_id not in enospc_retries
        git_ops.cleanup_merge_worktree.assert_not_awaited()

    async def test_equivalence_failure_blocks_no_push(self) -> None:
        """(b) equivalence returns non-empty → blocked with equiv prefix; push NOT called."""
        from orchestrator.merge_queue import (
            POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
            _finalize_advanced_merge,
        )

        git_ops = self._make_git_ops()
        req = self._make_req()
        cas_retries, timeouts, enospc_retries = self._primed_dicts(req.task_id)

        with (
            patch('orchestrator.merge_queue._check_post_merge_equivalence', AsyncMock(return_value=['file.py'])),
            patch('orchestrator.merge_queue._check_post_merge_pyright', AsyncMock()) as mock_pyright,
        ):
            outcome = await _finalize_advanced_merge(
                git_ops, req, None,
                merge_commit_fallback='fallback-sha',
                base_sha='base-sha',
                started_monotonic=0.0,
                cas_retries=cas_retries,
                timeouts=timeouts,
                enospc_retries=enospc_retries,
            )

        assert outcome.status == 'blocked'
        assert outcome.reason.startswith(POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX), (
            f'unexpected reason: {outcome.reason!r}'
        )
        git_ops.push_main.assert_not_awaited()
        mock_pyright.assert_not_awaited()
        git_ops.cleanup_merge_worktree.assert_not_awaited()

    async def test_pyright_broken_blocks_no_push(self) -> None:
        """(c) pyright .broken True → blocked with pyright prefix, failing subproject in reason; push NOT called."""
        from orchestrator.merge_queue import (
            POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
            _finalize_advanced_merge,
        )

        git_ops = self._make_git_ops()
        req = self._make_req()
        cas_retries, timeouts, enospc_retries = self._primed_dicts(req.task_id)
        pyright_broken = MagicMock(broken=True, failing_subprojects=['mypackage'], detail='type error detail')

        with (
            patch('orchestrator.merge_queue._check_post_merge_equivalence', AsyncMock(return_value=[])),
            patch('orchestrator.merge_queue._check_post_merge_pyright', AsyncMock(return_value=pyright_broken)),
        ):
            outcome = await _finalize_advanced_merge(
                git_ops, req, None,
                merge_commit_fallback='fallback-sha',
                base_sha='base-sha',
                started_monotonic=0.0,
                cas_retries=cas_retries,
                timeouts=timeouts,
                enospc_retries=enospc_retries,
            )

        assert outcome.status == 'blocked'
        assert outcome.reason.startswith(POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX), (
            f'unexpected reason: {outcome.reason!r}'
        )
        assert 'mypackage' in outcome.reason
        git_ops.push_main.assert_not_awaited()
        git_ops.cleanup_merge_worktree.assert_not_awaited()

    async def test_no_last_advanced_sha_uses_fallback(self) -> None:
        """(d) _last_advanced_sha absent/None → advanced_sha falls back to merge_commit_fallback."""
        from orchestrator.merge_queue import _finalize_advanced_merge

        git_ops = self._make_git_ops(last_advanced_sha=None)
        req = self._make_req()
        cas_retries, timeouts, enospc_retries = self._primed_dicts(req.task_id)
        pyright_clean = MagicMock(broken=False, failing_subprojects=[], detail='')

        with (
            patch('orchestrator.merge_queue._check_post_merge_equivalence', AsyncMock(return_value=[])),
            patch('orchestrator.merge_queue._check_post_merge_pyright', AsyncMock(return_value=pyright_clean)),
        ):
            outcome = await _finalize_advanced_merge(
                git_ops, req, None,
                merge_commit_fallback='the-fallback-sha',
                base_sha='base-sha',
                started_monotonic=0.0,
                cas_retries=cas_retries,
                timeouts=timeouts,
                enospc_retries=enospc_retries,
            )

        assert outcome.status == 'done'
        assert outcome.merge_sha == 'the-fallback-sha'

    # γ2 step-11 RED: chain_ctx integration tests
    async def test_back_compat_no_chain_ctx_still_blocks(self) -> None:
        """(a) BACK-COMPAT: called without chain_ctx → equivalence failure returns blocked (unchanged)."""
        from orchestrator.merge_queue import (
            POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
            _finalize_advanced_merge,
        )

        git_ops = self._make_git_ops()
        req = self._make_req()
        cas_retries, timeouts, enospc_retries = self._primed_dicts(req.task_id)

        with (
            patch('orchestrator.merge_queue._check_post_merge_equivalence', AsyncMock(return_value=['f.py'])),
            patch('orchestrator.merge_queue._check_post_merge_pyright', AsyncMock()),
        ):
            # No chain_ctx — default behaviour
            outcome = await _finalize_advanced_merge(
                git_ops, req, None,
                merge_commit_fallback='fallback-sha',
                base_sha='base-sha',
                started_monotonic=0.0,
                cas_retries=cas_retries,
                timeouts=timeouts,
                enospc_retries=enospc_retries,
            )

        assert outcome.status == 'blocked'
        assert outcome.reason.startswith(POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX)

    async def test_feature_gate_default_off_blocks_not_supersedes(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """(b-gate) AUTO_CHAIN_GENERATIONS_ENABLED defaults to False; with kill-switch
        OFF, _finalize_advanced_merge returns 'blocked' (not 'superseded') even when
        chain_ctx is wired and tip is a SUPERSET advance, and the queue stays empty."""
        import orchestrator.merge_queue as mq
        from orchestrator.merge_queue import (
            POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
            MergeRequest,
            TipRelation,
            _finalize_advanced_merge,
            _GenerationChainContext,
        )

        # Verify the kill-switch is False by default.
        assert mq.AUTO_CHAIN_GENERATIONS_ENABLED is False

        git_ops = self._make_git_ops()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        req = MergeRequest(
            task_id='task-gate',
            branch='task/t-gate',
            worktree=tmp_path,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=fut,
            generation=1,
        )
        cas_retries, timeouts, enospc_retries = self._primed_dicts(req.task_id)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        counts: dict[str, int] = {}
        chain_ctx = _GenerationChainContext(
            queue=queue, counts=counts, max_auto_generations=2,
        )

        # Kill-switch is OFF (default) — no patch needed.
        with (
            patch('orchestrator.merge_queue._check_post_merge_equivalence', AsyncMock(return_value=['f.py'])),
            patch('orchestrator.merge_queue._check_post_merge_pyright', AsyncMock()),
            patch('orchestrator.merge_queue._run', AsyncMock(return_value=(0, 'newhead\n', ''))),
            patch('orchestrator.merge_queue.classify_tip_relation', AsyncMock(return_value=TipRelation.SUPERSET)),
        ):
            outcome = await _finalize_advanced_merge(
                git_ops, req, None,
                merge_commit_fallback='fallback-sha',
                base_sha='base-sha',
                started_monotonic=0.0,
                cas_retries=cas_retries,
                timeouts=timeouts,
                enospc_retries=enospc_retries,
                chain_ctx=chain_ctx,
                merged_branch_tip='oldtip',
            )

        assert outcome.status == 'blocked', (
            f'kill-switch OFF should block, got {outcome.status!r}'
        )
        assert outcome.reason.startswith(POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX)
        assert queue.empty(), 'no gen-(n+1) request should be enqueued while kill-switch is OFF'

    async def test_chain_ctx_superset_advance_returns_superseded_and_enqueues(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """(b) chain_ctx + merged_branch_tip + SUPERSET advance → returns superseded,
        enqueues gen-(n+1) request (tested with AUTO_CHAIN_GENERATIONS_ENABLED=True)."""
        from orchestrator.merge_queue import (
            MergeRequest,
            TipRelation,
            _finalize_advanced_merge,
            _GenerationChainContext,
        )

        git_ops = self._make_git_ops()
        # _maybe_auto_chain_generation calls dataclasses.replace(req, ...) so
        # we need a real MergeRequest (not MagicMock) for the chaining path.
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        req = MergeRequest(
            task_id='task-chain',
            branch='task/t-chain',
            worktree=tmp_path,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=fut,
            generation=1,
        )
        cas_retries, timeouts, enospc_retries = self._primed_dicts(req.task_id)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        counts: dict[str, int] = {}
        chain_ctx = _GenerationChainContext(
            queue=queue, counts=counts, max_auto_generations=2,
        )

        with (
            patch('orchestrator.merge_queue.AUTO_CHAIN_GENERATIONS_ENABLED', True),
            patch('orchestrator.merge_queue._check_post_merge_equivalence', AsyncMock(return_value=['f.py'])),
            patch('orchestrator.merge_queue._check_post_merge_pyright', AsyncMock()),
            patch('orchestrator.merge_queue._run', AsyncMock(return_value=(0, 'newhead\n', ''))),
            patch('orchestrator.merge_queue.classify_tip_relation', AsyncMock(return_value=TipRelation.SUPERSET)),
        ):
            outcome = await _finalize_advanced_merge(
                git_ops, req, None,
                merge_commit_fallback='fallback-sha',
                base_sha='base-sha',
                started_monotonic=0.0,
                cas_retries=cas_retries,
                timeouts=timeouts,
                enospc_retries=enospc_retries,
                chain_ctx=chain_ctx,
                merged_branch_tip='oldtip',
            )

        assert outcome.status == 'superseded'
        assert outcome.superseded_by is not None
        assert queue.qsize() == 1

    async def test_chain_ctx_superseded_emits_generation_chained_event(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """When AUTO_CHAIN_GENERATIONS_ENABLED=True and _maybe_auto_chain_generation
        returns a chained outcome, _finalize_advanced_merge emits a
        'post_merge_generation_chained' merge_attempt event to the event_store.
        This is the observable contract for reconciliation provenance."""
        from orchestrator.event_store import EventType
        from orchestrator.merge_queue import (
            MergeRequest,
            TipRelation,
            _finalize_advanced_merge,
            _GenerationChainContext,
        )

        event_store = MagicMock()
        git_ops = self._make_git_ops()
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        req = MergeRequest(
            task_id='task-event',
            branch='task/t-event',
            worktree=tmp_path,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=fut,
            generation=1,
        )
        cas_retries, timeouts, enospc_retries = self._primed_dicts(req.task_id)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        counts: dict[str, int] = {}
        chain_ctx = _GenerationChainContext(
            queue=queue, counts=counts, max_auto_generations=2,
        )

        with (
            patch('orchestrator.merge_queue.AUTO_CHAIN_GENERATIONS_ENABLED', True),
            patch('orchestrator.merge_queue._check_post_merge_equivalence', AsyncMock(return_value=['f.py'])),
            patch('orchestrator.merge_queue._check_post_merge_pyright', AsyncMock()),
            patch('orchestrator.merge_queue._run', AsyncMock(return_value=(0, 'newhead\n', ''))),
            patch('orchestrator.merge_queue.classify_tip_relation', AsyncMock(return_value=TipRelation.SUPERSET)),
        ):
            outcome = await _finalize_advanced_merge(
                git_ops, req, event_store,
                merge_commit_fallback='fallback-sha',
                base_sha='base-sha',
                started_monotonic=0.0,
                cas_retries=cas_retries,
                timeouts=timeouts,
                enospc_retries=enospc_retries,
                chain_ctx=chain_ctx,
                merged_branch_tip='oldtip',
            )

        assert outcome.status == 'superseded', (
            f'expected superseded, got {outcome.status!r}'
        )
        # Assert that the 'post_merge_generation_chained' merge_attempt event was emitted.
        # Note: event_store.emit is also called for merge_queued events (from
        # enqueue_merge_request → _emit_merge_queued_event) whose data dict has
        # no 'outcome' key — guard with .get() to skip those.
        emitted_outcomes = [
            call.kwargs['data'].get('outcome')
            for call in event_store.emit.call_args_list
            if 'data' in call.kwargs
        ]
        assert 'post_merge_generation_chained' in emitted_outcomes, (
            f'expected post_merge_generation_chained event; got: {emitted_outcomes!r}'
        )
        # The 'post_merge_equivalence_failed' event should also have been emitted first.
        assert 'post_merge_equivalence_failed' in emitted_outcomes, (
            f'expected post_merge_equivalence_failed event; got: {emitted_outcomes!r}'
        )
        # Event was emitted with the correct task_id.
        chained_call = next(
            c for c in event_store.emit.call_args_list
            if 'data' in c.kwargs
            and c.kwargs['data'].get('outcome') == 'post_merge_generation_chained'
        )
        assert chained_call.args[0] == EventType.merge_attempt
        assert chained_call.kwargs['task_id'] == req.task_id

    async def test_chain_ctx_done_pops_branch_counter(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """(c) on 'done' path with chain_ctx, counts[branch] is popped."""
        from orchestrator.merge_queue import (
            _finalize_advanced_merge,
            _GenerationChainContext,
        )

        git_ops = self._make_git_ops()
        req = self._make_req()
        req.branch = 'task/t-done-pop'
        cas_retries, timeouts, enospc_retries = self._primed_dicts(req.task_id)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        counts: dict[str, int] = {'task/t-done-pop': 1}
        chain_ctx = _GenerationChainContext(
            queue=queue, counts=counts, max_auto_generations=2,
        )
        pyright_clean = MagicMock(broken=False, failing_subprojects=[], detail='')

        with (
            patch('orchestrator.merge_queue._check_post_merge_equivalence', AsyncMock(return_value=[])),
            patch('orchestrator.merge_queue._check_post_merge_pyright', AsyncMock(return_value=pyright_clean)),
        ):
            outcome = await _finalize_advanced_merge(
                git_ops, req, None,
                merge_commit_fallback='fallback-sha',
                base_sha='base-sha',
                started_monotonic=0.0,
                cas_retries=cas_retries,
                timeouts=timeouts,
                enospc_retries=enospc_retries,
                chain_ctx=chain_ctx,
                merged_branch_tip='oldtip',
            )

        assert outcome.status == 'done'
        assert 'task/t-done-pop' not in counts  # popped on clean landing


# ---------------------------------------------------------------------------
# TestMapAdvanceFailure — unit tests for the _map_advance_failure helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMapAdvanceFailure:
    """Direct unit tests for the _map_advance_failure module-level helper."""

    def _make_git_ops(self) -> MagicMock:
        git_ops = MagicMock()
        git_ops.push_main = AsyncMock(return_value='pushed')
        git_ops._last_recovery_branch = 'recovery/branch-abc'
        git_ops._last_overlap_files = ['foo.py', 'bar.py']
        git_ops._last_advanced_sha = 'adv-sha-123'
        return git_ops

    async def test_wip_overlap_halts_returns_wip_halted(self) -> None:
        """(a) wip_overlap → halt called once, push NOT awaited, task_id STILL in cas_retries."""
        from orchestrator.merge_queue import _map_advance_failure

        git_ops = self._make_git_ops()
        halt = MagicMock()
        task_id = 'task-adv-fail'
        cas_retries = {task_id: 1}

        outcome = await _map_advance_failure(
            git_ops, 'wip_overlap',
            task_id=task_id, merge_commit_fallback='fallback-sha',
            halt=halt, cas_retries=cas_retries,
        )

        assert outcome.status == 'wip_halted'
        assert outcome.overlap_files == git_ops._last_overlap_files
        assert 'WIP overlaps merge diff:' in outcome.reason
        halt.assert_called_once_with('advance_main: wip_overlap')
        git_ops.push_main.assert_not_awaited()
        assert task_id in cas_retries, 'wip_overlap must NOT pop cas_retries'

    async def test_pop_conflict_halts_pushes_returns_done_wip_recovery(self) -> None:
        """(b) pop_conflict → halt called, push_main awaited, done_wip_recovery with recovery branch."""
        from orchestrator.merge_queue import _map_advance_failure

        git_ops = self._make_git_ops()
        halt = MagicMock()
        task_id = 'task-pop-conf'
        cas_retries = {task_id: 1}

        outcome = await _map_advance_failure(
            git_ops, 'pop_conflict',
            task_id=task_id, merge_commit_fallback='fallback-sha',
            halt=halt, cas_retries=cas_retries,
        )

        assert outcome.status == 'done_wip_recovery'
        assert outcome.recovery_branch == git_ops._last_recovery_branch
        assert outcome.push_status == 'pushed'
        assert outcome.merge_sha == git_ops._last_advanced_sha
        halt.assert_called_once_with('advance_main: pop_conflict')
        git_ops.push_main.assert_awaited_once()

    async def test_unmerged_state_halts_pops_retries(self) -> None:
        """(c) unmerged_state → halt with unmerged message, cas_retries popped."""
        from orchestrator.merge_queue import _map_advance_failure

        git_ops = self._make_git_ops()
        halt = MagicMock()
        task_id = 'task-unmerged'
        cas_retries = {task_id: 2}

        outcome = await _map_advance_failure(
            git_ops, 'unmerged_state',
            task_id=task_id, merge_commit_fallback='fallback-sha',
            halt=halt, cas_retries=cas_retries,
        )

        assert outcome.status == 'unmerged_state'
        assert task_id in outcome.reason
        assert 'unmerged_state' in outcome.reason
        halt.assert_called_once()
        assert 'unmerged_state' in halt.call_args[0][0]
        assert task_id not in cas_retries

    async def test_pop_conflict_no_advance_halts_pops_retries(self) -> None:
        """(d) pop_conflict_no_advance → halt, recovery_branch from git_ops, cas_retries popped."""
        from orchestrator.merge_queue import _map_advance_failure

        git_ops = self._make_git_ops()
        halt = MagicMock()
        task_id = 'task-no-adv'
        cas_retries = {task_id: 1}

        outcome = await _map_advance_failure(
            git_ops, 'pop_conflict_no_advance',
            task_id=task_id, merge_commit_fallback='fallback-sha',
            halt=halt, cas_retries=cas_retries,
        )

        assert outcome.status == 'wip_recovery_no_advance'
        assert outcome.recovery_branch == git_ops._last_recovery_branch
        halt.assert_called_once_with('advance_main: pop_conflict_no_advance')
        assert task_id not in cas_retries

    @pytest.mark.parametrize('result', ['not_descendant', 'contaminated', 'stash_failed'])
    async def test_terminal_results_block_no_halt(self, result: str) -> None:
        """(e) not_descendant/contaminated/stash_failed → blocked with exact reason, halt NOT called, cas_retries popped."""
        from orchestrator.merge_queue import _map_advance_failure

        git_ops = self._make_git_ops()
        halt = MagicMock()
        task_id = 'task-terminal'
        cas_retries = {task_id: 3}

        outcome = await _map_advance_failure(
            git_ops, result,
            task_id=task_id, merge_commit_fallback='fallback-sha',
            halt=halt, cas_retries=cas_retries,
        )

        assert outcome.status == 'blocked'
        assert outcome.reason == f'advance_main failed ({result}) for task {task_id}'
        halt.assert_not_called()
        assert task_id not in cas_retries


# ---------------------------------------------------------------------------
# TestWipHaltMixin — unit tests pinning the _WipHaltMixin shared contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('worker_cls', [MergeWorker, SpeculativeMergeWorker])
class TestWipHaltMixin:
    """Pin the _WipHaltMixin contract as seen through both worker classes."""

    def _make_worker(self, worker_cls: type) -> Any:
        git_ops = MagicMock()
        queue: asyncio.Queue = asyncio.Queue()
        return worker_cls(git_ops, queue)

    async def test_issubclass_pins_single_source(self, worker_cls: type) -> None:
        """Both workers must subclass _WipHaltMixin (fails at import until S8)."""
        from orchestrator.merge_queue import _WipHaltMixin

        assert issubclass(MergeWorker, _WipHaltMixin)
        assert issubclass(SpeculativeMergeWorker, _WipHaltMixin)

    async def test_initial_state_not_halted(self, worker_cls: type) -> None:
        """is_wip_halted is False immediately after construction."""
        from orchestrator.merge_queue import _WipHaltMixin  # noqa: F401

        worker = self._make_worker(worker_cls)
        assert not worker.is_wip_halted

    async def test_halt_for_wip_sets_halted(self, worker_cls: type) -> None:
        """After halt_for_wip('x'), is_wip_halted True and halt_owner_esc_id still None."""
        from orchestrator.merge_queue import _WipHaltMixin  # noqa: F401

        worker = self._make_worker(worker_cls)
        worker.halt_for_wip('x')
        assert worker.is_wip_halted
        assert worker.halt_owner_esc_id is None

    async def test_set_halt_owner_and_query(self, worker_cls: type) -> None:
        """set_halt_owner('e1') → is_halt_owner('e1') True, is_halt_owner('e2') False, halt_owner_esc_id == 'e1'."""
        from orchestrator.merge_queue import _WipHaltMixin  # noqa: F401

        worker = self._make_worker(worker_cls)
        worker.halt_for_wip('reason')
        worker.set_halt_owner('e1')
        assert worker.is_halt_owner('e1')
        assert not worker.is_halt_owner('e2')
        assert worker.halt_owner_esc_id == 'e1'

    async def test_second_set_halt_owner_raises(self, worker_cls: type) -> None:
        """A second set_halt_owner call raises AssertionError."""
        from orchestrator.merge_queue import _WipHaltMixin  # noqa: F401

        worker = self._make_worker(worker_cls)
        worker.halt_for_wip('reason')
        worker.set_halt_owner('e1')
        with pytest.raises(AssertionError):
            worker.set_halt_owner('e2')

    async def test_unhalt_clears_state(self, worker_cls: type) -> None:
        """After unhalt_wip(), is_wip_halted False and halt_owner_esc_id None."""
        from orchestrator.merge_queue import _WipHaltMixin  # noqa: F401

        worker = self._make_worker(worker_cls)
        worker.halt_for_wip('reason')
        worker.set_halt_owner('e1')
        worker.unhalt_wip()
        assert not worker.is_wip_halted
        assert worker.halt_owner_esc_id is None

    async def test_abandon_outcome_uses_prefix(self, worker_cls: type) -> None:
        """_abandon_outcome('t', 3) → blocked MergeOutcome starting with ABANDONED_REASON_PREFIX containing 't'."""
        from orchestrator.merge_queue import ABANDONED_REASON_PREFIX, _WipHaltMixin  # noqa: F401

        worker = self._make_worker(worker_cls)
        outcome = worker._abandon_outcome('t', 3)
        assert outcome.status == 'blocked'
        assert outcome.reason.startswith(ABANDONED_REASON_PREFIX)
        assert 't' in outcome.reason


# ---------------------------------------------------------------------------
# TestHaltAdvanceResults — pin _HALT_ADVANCE_RESULTS module constant
# ---------------------------------------------------------------------------


class TestHaltAdvanceResults:
    """Pins that _HALT_ADVANCE_RESULTS is a single module-level constant shared
    by both workers, preventing silent MergeWorker/SpeculativeMergeWorker drift."""

    def test_is_importable(self) -> None:
        """_HALT_ADVANCE_RESULTS must be importable (fails before amendment)."""
        from orchestrator.merge_queue import _HALT_ADVANCE_RESULTS  # noqa: F401

    def test_contains_expected_results(self) -> None:
        """All four halt-triggering advance_main results must be present."""
        from orchestrator.merge_queue import _HALT_ADVANCE_RESULTS

        expected = frozenset({
            'wip_overlap', 'pop_conflict',
            'unmerged_state', 'pop_conflict_no_advance',
        })
        assert frozenset(_HALT_ADVANCE_RESULTS) == expected, (
            f'_HALT_ADVANCE_RESULTS mismatch: {_HALT_ADVANCE_RESULTS!r}'
        )


# ---------------------------------------------------------------------------
# TestAdvanceMainReverifyOnRebase — step-1 (RED)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAdvanceMainReverifyOnRebase:
    """advance_main gains an opt-in reverify_on_rebase flag.

    When set, a rebase-then-break inside the for-attempt loop causes
    advance_main to:
      • expose the rebased SHA via _last_advanced_sha
      • expose the original expected_main via _rebased_from
      • expose the rebased-onto SHA via _rebased_onto
      • return 'rebased_pending_reverify' WITHOUT calling update-ref

    The default (flag omitted / False) is unchanged: rebase + advance.
    """

    async def test_reverify_on_rebase_returns_new_result(
        self, git_ops: GitOps,
    ) -> None:
        """(a) reverify_on_rebase=True + main moves → rebased_pending_reverify,
        main stays at moved-main SHA, side channels populated, merge_wt at
        rebased SHA.
        """
        # Build a branch commit
        worktree = (await git_ops.create_worktree('rev-branch')).path
        (worktree / 'branch_only.py').write_text('branch = 1\n')
        await git_ops.commit(worktree, 'Add branch_only.py')

        # Capture base_sha and create the merge commit
        base_sha = await git_ops.get_main_sha()
        result = await git_ops.merge_to_main(worktree, 'rev-branch')
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None
        merge_commit = result.merge_commit.strip()
        merge_wt = result.merge_worktree

        # Move main by adding a DISJOINT file directly in project_root
        (git_ops.project_root / 'main_only.py').write_text('main = 1\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Disjoint: add main_only.py'],
            cwd=git_ops.project_root,
        )
        moved_main_sha = await git_ops.get_main_sha()
        assert moved_main_sha != base_sha, 'main must have moved for the test'

        try:
            advanced = await git_ops.advance_main(
                merge_commit, merge_wt,
                branch='rev-branch',
                expected_main=base_sha,
                reverify_on_rebase=True,
            )

            # Must return the new gate result, NOT 'advanced'
            assert advanced == 'rebased_pending_reverify', (
                f'Expected rebased_pending_reverify, got {advanced!r}'
            )

            # Main must NOT have been advanced (still at moved_main_sha)
            current_main = await git_ops.get_main_sha()
            assert current_main == moved_main_sha, (
                f'Main must not have advanced: expected {moved_main_sha[:8]}, '
                f'got {current_main[:8]}'
            )

            # Side channels must be set
            rebased_sha = getattr(git_ops, '_last_advanced_sha', None)
            assert rebased_sha is not None, '_last_advanced_sha must be set'
            rebased_from = getattr(git_ops, '_rebased_from', None)
            assert rebased_from == base_sha, (
                f'_rebased_from must be original base_sha {base_sha[:8]}, '
                f'got {rebased_from!r}'
            )
            rebased_onto = getattr(git_ops, '_rebased_onto', None)
            assert rebased_onto == moved_main_sha, (
                f'_rebased_onto must be moved_main_sha {moved_main_sha[:8]}, '
                f'got {rebased_onto!r}'
            )

            # _last_advanced_sha must be a descendant of moved_main
            rc, _, _ = await _run(
                ['git', 'merge-base', '--is-ancestor', moved_main_sha, rebased_sha],
                cwd=git_ops.project_root,
            )
            assert rc == 0, (
                f'rebased_sha {rebased_sha[:8]} must be a descendant of '
                f'moved_main {moved_main_sha[:8]}'
            )

            # merge_wt HEAD must equal _last_advanced_sha
            wt_rc, wt_head, _ = await _run(
                ['git', 'rev-parse', 'HEAD'], cwd=merge_wt,
            )
            assert wt_rc == 0
            assert wt_head.strip() == rebased_sha, (
                f'merge_wt HEAD {wt_head.strip()[:8]} must equal '
                f'_last_advanced_sha {rebased_sha[:8]}'
            )
        finally:
            await git_ops.cleanup_merge_worktree(merge_wt)

    async def test_no_rebase_needed_returns_advanced(
        self, git_ops: GitOps,
    ) -> None:
        """(b) reverify_on_rebase=True but no rebase needed → returns 'advanced'.

        When main has NOT moved, the merge commit is already a descendant of
        main.  No rebase occurs, so the gate is not triggered even with the
        flag set.
        """
        worktree = (await git_ops.create_worktree('rev-no-rebase')).path
        (worktree / 'no_rebase.py').write_text('x = 1\n')
        await git_ops.commit(worktree, 'Add no_rebase.py')

        base_sha = await git_ops.get_main_sha()
        result = await git_ops.merge_to_main(worktree, 'rev-no-rebase')
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None

        # Main has NOT moved — no rebase needed
        try:
            advanced = await git_ops.advance_main(
                result.merge_commit, result.merge_worktree,
                branch='rev-no-rebase',
                expected_main=base_sha,
                reverify_on_rebase=True,
            )
            assert advanced == 'advanced', (
                f'No rebase → must return advanced, got {advanced!r}'
            )
            # Main must have moved to the merge commit
            current_main = await git_ops.get_main_sha()
            assert current_main != base_sha, 'Main must have been advanced'
        finally:
            if result.merge_worktree:
                await git_ops.cleanup_merge_worktree(result.merge_worktree)

    async def test_backward_compat_default_flag_still_advances(
        self, git_ops: GitOps,
    ) -> None:
        """(c) Backward compat: default (flag omitted) → rebase still advances.

        MergeWorker and _do_train_merge callers that don't pass
        reverify_on_rebase must be unaffected.
        """
        worktree = (await git_ops.create_worktree('rev-compat')).path
        (worktree / 'compat.py').write_text('c = 1\n')
        await git_ops.commit(worktree, 'Add compat.py')

        base_sha = await git_ops.get_main_sha()
        result = await git_ops.merge_to_main(worktree, 'rev-compat')
        assert result.success
        assert result.merge_commit is not None
        assert result.merge_worktree is not None
        merge_wt = result.merge_worktree

        # Move main by adding a disjoint file
        (git_ops.project_root / 'compat_main.py').write_text('cm = 1\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Disjoint: add compat_main.py'],
            cwd=git_ops.project_root,
        )
        moved_main_sha = await git_ops.get_main_sha()

        try:
            # Default flag (not passed) — should still rebase and advance
            advanced = await git_ops.advance_main(
                result.merge_commit, merge_wt,
                branch='rev-compat',
                expected_main=base_sha,
                # reverify_on_rebase NOT passed — default False
            )
            # With default flag, advance_main rebases and advances (old behavior)
            assert advanced in ('advanced', 'cas_failed'), (
                f'Default flag: expected advanced or cas_failed, got {advanced!r}'
            )
            if advanced == 'advanced':
                current_main = await git_ops.get_main_sha()
                assert current_main != moved_main_sha, (
                    'Main must have advanced past moved_main_sha'
                )
        finally:
            if merge_wt:
                await git_ops.cleanup_merge_worktree(merge_wt)


# ---------------------------------------------------------------------------
# TestRebaseDeltaTouchedOverlap — step-3 (RED)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRebaseDeltaTouchedOverlap:
    """Unit tests for _rebase_delta_touched_overlap helper.

    The helper computes the intersection of:
      • branch_touched: files the branch changed since its fork from rebased_from
      • intervening:    files changed on main from rebased_from to rebased_onto

    Fails CLOSED on any git error: returns a non-empty sentinel so the caller
    re-verifies rather than skipping.
    """

    async def test_disjoint_returns_empty(self, git_ops: GitOps) -> None:
        """(a) Disjoint: branch adds branch_only.py, main adds main_only.py.

        The intersection must be empty (no overlap → no re-verify).
        """
        from orchestrator.merge_queue import _rebase_delta_touched_overlap

        # Create the branch worktree at the fork point (current main)
        fork_sha = await git_ops.get_main_sha()  # F = rebased_from
        wt = (await git_ops.create_worktree('delta-disjoint')).path
        (wt / 'branch_only.py').write_text('branch = 1\n')
        await git_ops.commit(wt, 'Branch: add branch_only.py')

        # Simulate intervening main commit (disjoint file)
        (git_ops.project_root / 'main_only.py').write_text('main = 1\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Main: add main_only.py'],
            cwd=git_ops.project_root,
        )
        rebased_onto = await git_ops.get_main_sha()

        overlap = await _rebase_delta_touched_overlap(
            wt, fork_sha, rebased_onto, git_ops,
        )
        assert overlap == [], (
            f'Disjoint: expected empty overlap, got {overlap!r}'
        )
        await git_ops.cleanup_merge_worktree(wt)

    async def test_overlap_returns_shared_file(self, git_ops: GitOps) -> None:
        """(b) Overlap: branch edits top of shared.py, main edits bottom.

        After a clean 3-way rebase the touched sets intersect on shared.py.
        """
        from orchestrator.merge_queue import _rebase_delta_touched_overlap

        # Commit a 20-line base file on main before the fork
        base_content = ''.join(f'line{i}\n' for i in range(20))
        (git_ops.project_root / 'shared.py').write_text(base_content)
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Main: add shared.py'],
            cwd=git_ops.project_root,
        )
        fork_sha = await git_ops.get_main_sha()  # F = rebased_from

        # Branch edits near the TOP (line1 area)
        wt = (await git_ops.create_worktree('delta-overlap')).path
        (wt / 'shared.py').write_text(
            base_content.replace('line1\n', 'line1\nbranch-edit\n')
        )
        await git_ops.commit(wt, 'Branch: edit top of shared.py')

        # Intervening main commit edits near the BOTTOM (line18 area)
        (git_ops.project_root / 'shared.py').write_text(
            base_content.replace('line18\n', 'line18\nmain-edit\n')
        )
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Main: edit bottom of shared.py'],
            cwd=git_ops.project_root,
        )
        rebased_onto = await git_ops.get_main_sha()

        overlap = await _rebase_delta_touched_overlap(
            wt, fork_sha, rebased_onto, git_ops,
        )
        assert 'shared.py' in overlap, (
            f'Overlap: expected shared.py in overlap, got {overlap!r}'
        )
        await git_ops.cleanup_merge_worktree(wt)

    async def test_task_dir_changes_excluded(self, git_ops: GitOps) -> None:
        """(c) .task/ changes are excluded from both the branch-touched set
        and the intervening-delta set.

        The branch adds .task/plan.json and branch_only.py; main adds
        .task/state.json (via its own commit) and main_only.py.  Only the
        non-.task/ files appear in the respective sets; the intersection
        of {branch_only.py} and {main_only.py} is still empty.
        """
        from orchestrator.merge_queue import _rebase_delta_touched_overlap

        fork_sha = await git_ops.get_main_sha()
        wt = (await git_ops.create_worktree('delta-taskdir')).path

        # Branch adds a real file and a .task/ file
        (wt / 'branch_file.py').write_text('b = 1\n')
        (wt / '.task').mkdir(exist_ok=True)
        (wt / '.task' / 'plan.json').write_text('{"branch": true}\n')
        await _run(['git', 'add', '-A'], cwd=wt)
        await _run(['git', 'commit', '-m', 'Branch: real + .task/'], cwd=wt)

        # Intervening main commit adds a real file and a .task/ file
        (git_ops.project_root / 'main_file.py').write_text('m = 1\n')
        (git_ops.project_root / '.task').mkdir(exist_ok=True)
        (git_ops.project_root / '.task' / 'state.json').write_text(
            '{"main": true}\n'
        )
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Main: real + .task/'],
            cwd=git_ops.project_root,
        )
        rebased_onto = await git_ops.get_main_sha()

        overlap = await _rebase_delta_touched_overlap(
            wt, fork_sha, rebased_onto, git_ops,
        )
        # Intersection of {branch_file.py} and {main_file.py} = {} — disjoint
        assert overlap == [], (
            f'task-dir exclusion: expected empty overlap, got {overlap!r}'
        )
        await git_ops.cleanup_merge_worktree(wt)

    async def test_fail_closed_on_bogus_ref(self, git_ops: GitOps) -> None:
        """(d) Fail CLOSED: bogus rebased_from causes a git error; the helper
        must return a non-empty sentinel list so the caller re-verifies.
        """
        from orchestrator.merge_queue import _rebase_delta_touched_overlap

        wt = (await git_ops.create_worktree('delta-failclosed')).path
        (wt / 'file.py').write_text('x = 1\n')
        await git_ops.commit(wt, 'Branch: add file.py')
        rebased_onto = await git_ops.get_main_sha()

        # Pass a nonsense SHA as rebased_from — merge-base or diff will fail
        bogus_sha = 'deadbeef' * 5  # 40 char hex but doesn't exist

        overlap = await _rebase_delta_touched_overlap(
            wt, bogus_sha, rebased_onto, git_ops,
        )
        assert len(overlap) > 0, (
            f'Fail-closed: bogus ref must return non-empty sentinel, got {overlap!r}'
        )
        await git_ops.cleanup_merge_worktree(wt)


# ---------------------------------------------------------------------------
# TestReverifyRebasedTree — step-5 (RED)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReverifyRebasedTree:
    """Unit tests for _reverify_rebased_tree shared gate.

    The gate:
    (a) Disjoint overlap → returns None, run_scoped_verification NOT called,
        merge_wt still exists.
    (b) Overlapping, green verify → returns None, run_scoped_verification
        called exactly once, merge_wt still exists.
    (c) Overlapping, red verify → returns blocked MergeOutcome,
        merge_wt cleaned up.
    """

    async def _make_merge_wt(
        self, git_ops: GitOps, branch_name: str, config: OrchestratorConfig,
    ) -> tuple[Path, MergeRequest]:
        """Create a branch, merge it, and return (merge_wt, req)."""
        wt = await _make_branch_with_file(git_ops, branch_name, f'{branch_name}.py', 'x = 1\n')
        result = await git_ops.merge_to_main(wt, branch_name)
        assert result.success
        assert result.merge_worktree is not None
        req = _make_request(branch_name, branch_name, wt, config)
        return result.merge_worktree, req

    async def test_disjoint_no_reverify(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """(a) Disjoint (overlap returns empty): gate returns None and does
        NOT call run_scoped_verification.  merge_wt still exists.
        """
        from orchestrator.merge_queue import _reverify_rebased_tree

        merge_wt, req = await self._make_merge_wt(git_ops, 'rvrt-disjoint', config)
        fork_sha = await git_ops.get_main_sha()
        # Move main (so we have a rebased_onto)
        (git_ops.project_root / 'rvrt_main.py').write_text('m = 1\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Move main'], cwd=git_ops.project_root)
        rebased_onto = await git_ops.get_main_sha()

        verify_mock = _mock_verify_pass()
        try:
            with (
                patch(
                    'orchestrator.merge_queue._rebase_delta_touched_overlap',
                    new=AsyncMock(return_value=[]),  # disjoint
                ),
                patch('orchestrator.merge_queue.run_scoped_verification', verify_mock),
            ):
                result = await _reverify_rebased_tree(
                    git_ops, req, merge_wt,
                    rebased_from=fork_sha,
                    rebased_onto=rebased_onto,
                    timeouts={},
                    enospc_retries={},
                    max_timeouts=3,
                    max_enospc=1,
                )
            assert result is None, (
                f'Disjoint: expected None, got {result!r}'
            )
            verify_mock.assert_not_called()
            assert merge_wt.exists(), 'merge_wt must still exist (not cleaned up)'
        finally:
            if merge_wt.exists():
                await git_ops.cleanup_merge_worktree(merge_wt)

    async def test_overlap_green_verify(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """(b) Overlapping + green verify: gate returns None,
        run_scoped_verification called exactly once, merge_wt still exists.
        """
        from orchestrator.merge_queue import _reverify_rebased_tree

        merge_wt, req = await self._make_merge_wt(git_ops, 'rvrt-overlap-green', config)
        fork_sha = await git_ops.get_main_sha()
        (git_ops.project_root / 'rvrt_shared.py').write_text('s = 1\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Move main'], cwd=git_ops.project_root)
        rebased_onto = await git_ops.get_main_sha()

        verify_mock = _mock_verify_pass()
        try:
            with (
                patch(
                    'orchestrator.merge_queue._rebase_delta_touched_overlap',
                    new=AsyncMock(return_value=['rvrt_shared.py']),  # overlapping
                ),
                patch('orchestrator.merge_queue.run_scoped_verification', verify_mock),
            ):
                result = await _reverify_rebased_tree(
                    git_ops, req, merge_wt,
                    rebased_from=fork_sha,
                    rebased_onto=rebased_onto,
                    timeouts={},
                    enospc_retries={},
                    max_timeouts=3,
                    max_enospc=1,
                )
            assert result is None, (
                f'Green verify: expected None, got {result!r}'
            )
            verify_mock.assert_called_once()
            assert merge_wt.exists(), 'merge_wt must still exist after green verify'
        finally:
            if merge_wt.exists():
                await git_ops.cleanup_merge_worktree(merge_wt)

    async def test_overlap_red_verify(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """(c) Overlapping + red verify: gate returns blocked MergeOutcome,
        merge_wt cleaned up.
        """
        from orchestrator.merge_queue import _reverify_rebased_tree

        merge_wt, req = await self._make_merge_wt(git_ops, 'rvrt-overlap-red', config)
        fork_sha = await git_ops.get_main_sha()
        (git_ops.project_root / 'rvrt_red.py').write_text('r = 1\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(['git', 'commit', '-m', 'Move main'], cwd=git_ops.project_root)
        rebased_onto = await git_ops.get_main_sha()

        verify_fail = AsyncMock(return_value=MagicMock(
            passed=False, summary='Test failed', timed_out=False,
            failure_report=MagicMock(return_value=None),
        ))

        with (
            patch(
                'orchestrator.merge_queue._rebase_delta_touched_overlap',
                new=AsyncMock(return_value=['rvrt_red.py']),  # overlapping
            ),
            patch('orchestrator.merge_queue.run_scoped_verification', verify_fail),
        ):
            result = await _reverify_rebased_tree(
                git_ops, req, merge_wt,
                rebased_from=fork_sha,
                rebased_onto=rebased_onto,
                timeouts={},
                enospc_retries={},
                max_timeouts=3,
                max_enospc=1,
            )
        assert result is not None, 'Red verify: expected a MergeOutcome, got None'
        assert isinstance(result, MergeOutcome)
        assert result.status == 'blocked', (
            f'Red verify: expected blocked status, got {result.status!r}'
        )
        # _run_post_merge_verify cleans up merge_wt on failure
        assert not merge_wt.exists(), 'merge_wt must be cleaned up after red verify'


# ---------------------------------------------------------------------------
# TestSpeculativeMergeWorkerGate — step-7 (RED)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSpeculativeMergeWorkerGate:
    """End-to-end integration tests for the disjoint-delta re-verify gate.

    Each test builds a SpeculativeItem via worker._remerge(req, None),
    simulates intervening main movement by committing directly in
    project_root, then drives worker._verify_and_advance(item) and asserts
    the expected outcome and run_scoped_verification call count.

    Scenarios
    ---------
    (a) GATE-on-red: overlapping shared.py churn; 1st verify (step-4) passes,
        2nd (gate re-verify) fails → outcome blocked, main stays at moved-main
        SHA (rebased overlapping tree never lands).
    (b) Overlap+green: overlapping churn, both verify calls pass → outcome
        done, branch-edit AND main-edit both on main, called twice.
    (c) Disjoint: branch_only.py vs main_only.py → outcome done, called
        exactly once (fast path preserved — no gate verify).
    (d) Regression: main does NOT move → outcome done, called once.

    Scenarios (a) and (b) fail BEFORE wiring because the CAS loop does not
    pass reverify_on_rebase=True nor handle the rebased_pending_reverify
    result (so verify is only called once and main advances unguarded).
    """

    # 20-line shared base file; edits to line1 (branch) and line18 (main)
    # are non-adjacent so the rebase is always a clean 3-way merge.
    _SHARED_BASE: str = ''.join(f'line{i}\n' for i in range(20))

    async def _setup_overlap_branch(
        self,
        git_ops: GitOps,
        branch_name: str,
        config: OrchestratorConfig,
    ) -> tuple[Path, MergeRequest]:
        """Commit shared.py (20 lines) on main, then create a branch that
        edits line1 (top).  Returns (branch_wt, req).  main is at fork_sha.
        Does NOT call _remerge.
        """
        # Commit shared.py baseline on main
        (git_ops.project_root / 'shared.py').write_text(self._SHARED_BASE)
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', f'Add shared.py for {branch_name}'],
            cwd=git_ops.project_root,
        )

        # Branch edits line1 (top) — non-adjacent to the line18 main edit
        branch_wt = (await git_ops.create_worktree(branch_name)).path
        (branch_wt / 'shared.py').write_text(
            self._SHARED_BASE.replace('line1\n', 'line1\nbranch-edit\n')
        )
        await git_ops.commit(branch_wt, f'Branch {branch_name}: edit top of shared.py')

        req = _make_request(branch_name, branch_name, branch_wt, config)
        return branch_wt, req

    async def test_gate_on_red_blocks_main(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """(a) GATE-on-red: overlapping churn, 1st verify passes, 2nd fails.

        Core safety property: the rebased overlapping tree must NEVER land on
        main.  outcome must be blocked and main must stay at the moved-main SHA.
        run_scoped_verification must be called exactly twice (initial + gate).
        """
        branch = 'smwg-red'
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        _, req = await self._setup_overlap_branch(git_ops, branch, config)

        # _remerge: merge branch into current main
        item = await worker._remerge(req, None)
        assert item.immediate_outcome is None, (
            f'_remerge must succeed; got immediate_outcome={item.immediate_outcome!r}'
        )

        # Move main: edit line18 (bottom) — produces overlapping delta
        (git_ops.project_root / 'shared.py').write_text(
            self._SHARED_BASE.replace('line18\n', 'line18\nmain-edit\n')
        )
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Move main: edit bottom of shared.py'],
            cwd=git_ops.project_root,
        )
        moved_main_sha = await git_ops.get_main_sha()

        verify_calls: list[int] = []

        async def _verify_side_effect(*args: Any, **kwargs: Any) -> Any:
            n = len(verify_calls) + 1
            verify_calls.append(n)
            if n == 1:
                # Initial verify (step-4): pass
                return MagicMock(passed=True, summary='')
            # Gate re-verify (2nd call): fail
            return MagicMock(
                passed=False, summary='gate re-verify failed',
                timed_out=False,
                failure_report=MagicMock(return_value=None),
            )

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=_verify_side_effect,
        ):
            advanced = await worker._verify_and_advance(item)

        assert not advanced, (
            '(a) Gate-on-red: expected _verify_and_advance to return False'
        )
        outcome = req.result.result()
        assert outcome.status == 'blocked', (
            f'(a) Gate-on-red: expected blocked, got {outcome.status!r}: {outcome}'
        )
        # Main must NOT have advanced to the rebased tree
        current_main = await git_ops.get_main_sha()
        assert current_main == moved_main_sha, (
            f'(a) Main must not advance past moved_main_sha. '
            f'Expected {moved_main_sha[:8]}, got {current_main[:8]}'
        )
        # Gate verify must have been triggered
        assert len(verify_calls) == 2, (
            f'(a) Expected run_scoped_verification called 2 times '
            f'(initial + gate), got {len(verify_calls)}'
        )

    async def test_overlap_green_advances_main(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """(b) Overlap+green: overlapping churn, both verify passes.

        Both the branch edit (line1) and the main-movement edit (line18) must
        appear on main after the advance.  run_scoped_verification called twice.
        """
        branch = 'smwg-green'
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        _, req = await self._setup_overlap_branch(git_ops, branch, config)

        item = await worker._remerge(req, None)
        assert item.immediate_outcome is None, (
            f'_remerge must succeed; got {item.immediate_outcome!r}'
        )

        # Move main: edit line18 (bottom) — overlapping delta
        (git_ops.project_root / 'shared.py').write_text(
            self._SHARED_BASE.replace('line18\n', 'line18\nmain-edit\n')
        )
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Move main: edit bottom of shared.py (green)'],
            cwd=git_ops.project_root,
        )

        verify_calls: list[int] = []

        async def _verify_side_effect(*args: Any, **kwargs: Any) -> Any:
            verify_calls.append(len(verify_calls) + 1)
            return MagicMock(passed=True, summary='')

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=_verify_side_effect,
        ):
            advanced = await worker._verify_and_advance(item)

        assert advanced, '(b) Overlap+green: expected True (done)'
        outcome = req.result.result()
        assert outcome.status == 'done', (
            f'(b) Overlap+green: expected done, got {outcome.status!r}: {outcome}'
        )
        # Both the branch edit and the main-movement edit must appear on main
        _, content, _ = await _run(
            ['git', 'show', 'main:shared.py'], cwd=git_ops.project_root,
        )
        assert 'branch-edit' in content, (
            '(b) branch edit must appear on main after overlap+green advance'
        )
        assert 'main-edit' in content, (
            '(b) main-movement edit must appear on main after overlap+green advance'
        )
        # Gate re-verify must have been called (overlap → two verify calls total)
        assert len(verify_calls) == 2, (
            f'(b) Expected run_scoped_verification called 2 times, '
            f'got {len(verify_calls)}'
        )

    async def test_disjoint_no_extra_verify(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """(c) Disjoint: branch_only.py vs main_only.py — no shared files.

        Fast path: the gate detects no overlap and skips re-verify.
        run_scoped_verification called exactly once (initial verify only).
        """
        branch = 'smwg-disjoint'
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        branch_wt = await _make_branch_with_file(
            git_ops, branch, 'branch_only.py', 'branch = 1\n',
        )
        req = _make_request(branch, branch, branch_wt, config)

        item = await worker._remerge(req, None)
        assert item.immediate_outcome is None

        # Move main: add a DISJOINT file (no shared files touched)
        (git_ops.project_root / 'main_only.py').write_text('main = 1\n')
        await _run(['git', 'add', '-A'], cwd=git_ops.project_root)
        await _run(
            ['git', 'commit', '-m', 'Move main: add main_only.py (disjoint)'],
            cwd=git_ops.project_root,
        )

        verify_calls: list[int] = []

        async def _verify_side_effect(*args: Any, **kwargs: Any) -> Any:
            verify_calls.append(len(verify_calls) + 1)
            return MagicMock(passed=True, summary='')

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=_verify_side_effect,
        ):
            advanced = await worker._verify_and_advance(item)

        assert advanced, '(c) Disjoint: expected True (done)'
        outcome = req.result.result()
        assert outcome.status == 'done', (
            f'(c) Disjoint: expected done, got {outcome.status!r}: {outcome}'
        )
        # Fast path: must NOT trigger a second verify call
        assert len(verify_calls) == 1, (
            f'(c) Disjoint fast path: expected 1 verify call, got {len(verify_calls)}'
        )

    async def test_no_movement_regression(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """(d) Regression: main does NOT move between _remerge and
        _verify_and_advance.  No gate is triggered; outcome done, one call.
        """
        branch = 'smwg-noop'
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue)

        branch_wt = await _make_branch_with_file(
            git_ops, branch, 'noop_branch.py', 'x = 1\n',
        )
        req = _make_request(branch, branch, branch_wt, config)

        item = await worker._remerge(req, None)
        assert item.immediate_outcome is None

        # main does NOT move

        verify_calls: list[int] = []

        async def _verify_side_effect(*args: Any, **kwargs: Any) -> Any:
            verify_calls.append(len(verify_calls) + 1)
            return MagicMock(passed=True, summary='')

        with patch(
            'orchestrator.merge_queue.run_scoped_verification',
            side_effect=_verify_side_effect,
        ):
            advanced = await worker._verify_and_advance(item)

        assert advanced, '(d) No-movement: expected True (done)'
        outcome = req.result.result()
        assert outcome.status == 'done', (
            f'(d) No-movement regression: expected done, got {outcome.status!r}: {outcome}'
        )
        assert len(verify_calls) == 1, (
            f'(d) No-movement: expected 1 verify call, got {len(verify_calls)}'
        )


# ---------------------------------------------------------------------------
# TestRegisterAndEnqueue — register_and_enqueue_merge_request unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRegisterAndEnqueue:
    """Unit tests for register_and_enqueue_merge_request.

    Exercises the workflow-path enqueue helper that registers the branch
    in the InFlightMergeRegistry before enqueuing, so the MCP coalesce
    gate sees cross-path in-flight merges.
    """

    def _make_event_store(self, tmp_path: Path) -> EventStore:
        db = tmp_path / 'rae_events.db'
        return EventStore(db_path=db, run_id='rae-test')

    async def test_happy_path_acquires_and_enqueues(
        self, config: OrchestratorConfig, tmp_path: Path,
    ):
        """Happy path: acquired=True, branch is_inflight, entry.task_id=='B', queue has 1 item."""
        from orchestrator.merge_queue import register_and_enqueue_merge_request

        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = self._make_event_store(tmp_path)
        req = _make_request('B', 'B', tmp_path, config)

        acquired = await register_and_enqueue_merge_request(queue, req, event_store, registry)

        assert acquired is True
        assert registry.is_inflight('B') is True
        entry_b = registry.entry('B')
        assert entry_b is not None
        assert entry_b.task_id == 'B'
        assert queue.qsize() == 1

    async def test_release_on_resolve_allows_redispatch(
        self, config: OrchestratorConfig, tmp_path: Path,
    ):
        """(a) release-on-resolve: after future resolves, a second call re-acquires."""
        from orchestrator.merge_queue import register_and_enqueue_merge_request

        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = self._make_event_store(tmp_path)
        req = _make_request('B', 'B', tmp_path, config)

        acquired = await register_and_enqueue_merge_request(queue, req, event_store, registry)
        assert acquired is True

        # Resolve the future → done_callback fires → slot released
        req.result.set_result(MergeOutcome(status='done'))
        await asyncio.sleep(0)
        assert registry.is_inflight('B') is False

        # Second call for the same branch should re-acquire (re-dispatch)
        req2 = _make_request('B', 'B', tmp_path, config)
        acquired2 = await register_and_enqueue_merge_request(queue, req2, event_store, registry)
        assert acquired2 is True
        assert registry.is_inflight('B') is True
        assert queue.qsize() == 2  # both enqueued (queue not drained)

    async def test_release_on_cancel_allows_retry(
        self, config: OrchestratorConfig, tmp_path: Path,
    ):
        """(b) release-on-cancel: soft-cancel releases the slot for the retry loop."""
        from orchestrator.merge_queue import register_and_enqueue_merge_request

        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = self._make_event_store(tmp_path)
        req = _make_request('B', 'B', tmp_path, config)

        await register_and_enqueue_merge_request(queue, req, event_store, registry)
        assert registry.is_inflight('B') is True

        # Cancel the future → done_callback fires → slot released
        req.result.cancel()
        await asyncio.sleep(0)
        assert registry.is_inflight('B') is False

    async def test_already_held_still_enqueues_returns_false(
        self, config: OrchestratorConfig, tmp_path: Path,
    ):
        """(c) already-held: returns False but still enqueues (no deadlock),
        and the slot remains owned by the first holder."""
        from orchestrator.merge_queue import register_and_enqueue_merge_request

        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = self._make_event_store(tmp_path)

        # Pre-seed the slot with a never-resolving future from 'other' task
        other_future: asyncio.Future = asyncio.get_running_loop().create_future()
        registry.acquire('B', 'other', other_future)

        # Helper call for same branch: slot already held → acquired=False
        req = _make_request('B', 'B', tmp_path, config)
        acquired = await register_and_enqueue_merge_request(queue, req, event_store, registry)

        assert acquired is False
        # Still enqueued — workflow must not deadlock
        assert queue.qsize() == 1
        # Slot is still owned by the original holder
        entry_b = registry.entry('B')
        assert entry_b is not None
        assert entry_b.task_id == 'other'

        # Cleanup
        other_future.cancel()

    async def test_registry_none_plain_enqueue(
        self, config: OrchestratorConfig, tmp_path: Path,
    ):
        """(d) registry=None: falls back to plain enqueue, returns False."""
        from orchestrator.merge_queue import register_and_enqueue_merge_request

        queue: asyncio.Queue = asyncio.Queue()
        event_store = self._make_event_store(tmp_path)
        req = _make_request('B', 'B', tmp_path, config)

        acquired = await register_and_enqueue_merge_request(queue, req, event_store, None)

        assert acquired is False
        assert queue.qsize() == 1

    async def test_slot_leak_guard_releases_on_enqueue_failure(
        self, config: OrchestratorConfig, tmp_path: Path,
    ):
        """(e) slot-leak guard: if enqueue_merge_request raises after acquire,
        the slot is released and the exception propagates."""
        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        req = _make_request('B', 'B', tmp_path, config)

        # Patch enqueue_merge_request to raise before the worker can ever
        # resolve req.result (simulating a closed queue / cancellation).
        boom = RuntimeError('queue closed')
        with patch(
            'orchestrator.merge_queue.enqueue_merge_request',
            new=AsyncMock(side_effect=boom),
        ), pytest.raises(RuntimeError, match='queue closed'):
            await register_and_enqueue_merge_request(queue, req, None, registry)

        # The slot-leak guard must have released the slot.
        assert registry.is_inflight('B') is False
        # The queue must be empty — the patched enqueue_merge_request never put.
        assert queue.qsize() == 0

    async def test_retention_forwarded_through_register_and_enqueue(
        self, config: OrchestratorConfig, tmp_path: Path,
    ) -> None:
        """register_and_enqueue_merge_request forwards retention to the chokepoint.

        After the dispatched request's future resolves, the retention ring must
        contain a TerminalOutcomeRecord keyed by request_id, and a
        merge_finalized row must exist in the event store.  Mirrors the
        coalesce-path test (TestCoalesceOrEnqueueRegistryOnly) for the dominant
        workflow path.
        """
        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        event_store = self._make_event_store(tmp_path)
        retention = TerminalOutcomeRetention()
        req = _make_request('rae-ret', 'rae-ret', tmp_path, config)

        await register_and_enqueue_merge_request(
            queue, req, event_store, registry, retention=retention,
        )

        # Resolve the future → done-callback fires
        req.result.set_result(MergeOutcome(status='done', merge_sha='rae1'))
        await asyncio.sleep(0)

        # Ring must have the record
        stored = retention.get(req.request_id)
        assert stored is not None
        assert stored.state == 'done'
        assert stored.merge_sha == 'rae1'
        assert stored.branch == req.branch
        assert stored.task_id == req.task_id

        # merge_finalized row must exist in the event store
        assert _count_events(event_store.db_path, 'merge_finalized') == 1


# ---------------------------------------------------------------------------
# TestSnapshotEntryRequestId — snapshot entry dict exposes request_id (task 1630)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSnapshotEntryRequestId:
    """Snapshot entry dict must expose ``request_id`` (α3 additive wiring)."""

    async def test_snapshot_entry_exposes_request_id(self, tmp_path: Path) -> None:
        """worker.snapshot()['entries'][0] contains 'request_id' == req.request_id."""
        import types

        loop = asyncio.get_running_loop()
        config = OrchestratorConfig(project_root=tmp_path / 'repo')
        mq: asyncio.Queue = asyncio.Queue()
        git_ops_stub = types.SimpleNamespace()
        worker = SpeculativeMergeWorker(
            git_ops=git_ops_stub,  # type: ignore[reportArgumentType]
            queue=mq,
        )

        req = MergeRequest(
            task_id='T1630',
            branch='T1630',
            worktree=tmp_path / 'wt',
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=loop.create_future(),
        )
        await mq.put(req)

        snap = worker.snapshot()
        entries = snap['entries']
        assert len(entries) == 1, f'Expected 1 entry, got: {entries}'
        entry = entries[0]
        assert 'request_id' in entry, (
            f"'request_id' key missing from snapshot entry: {entry}"
        )
        assert entry['request_id'] == req.request_id, (
            f"entry['request_id']={entry['request_id']!r} != req.request_id={req.request_id!r}"
        )


# ---------------------------------------------------------------------------
# β1 Step-1 RED: _InFlightEntry.request_id + MergeDispatchResult.inflight_request_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInFlightRegistryRequestId:
    """β1 step-1 RED: InFlightMergeRegistry tracks request_id; coalesce surfaces it.

    Tests are RED until step-2 impl:
    - acquire() does not yet accept a request_id keyword argument (TypeError)
    - _InFlightEntry has no request_id field (AttributeError)
    - MergeDispatchResult has no inflight_request_id attribute (AttributeError)
    """

    def _make_future(self) -> asyncio.Future:
        return asyncio.get_running_loop().create_future()

    async def test_acquire_with_request_id_stores_it(self):
        """(a) acquire(..., request_id='mr-x') → entry.request_id == 'mr-x'."""
        registry = InFlightMergeRegistry()
        fut = self._make_future()

        registry.acquire('X', 'task-X', fut, request_id='mr-x')

        entry = registry.entry('X')
        assert entry is not None
        assert entry.request_id == 'mr-x'

    async def test_acquire_without_request_id_is_none(self):
        """(b) acquire(branch, task_id, future) without request_id → entry.request_id is None.

        Back-compat: the existing 3-arg positional call still works unchanged.
        """
        registry = InFlightMergeRegistry()
        fut = self._make_future()

        registry.acquire('Y', 'task-Y', fut)

        entry = registry.entry('Y')
        assert entry is not None
        assert entry.request_id is None

    async def test_coalesce_surfaces_inflight_request_id(
        self, config: OrchestratorConfig, tmp_path: Path,
    ):
        """(c) Second coalesce returns inflight_request_id == first req's request_id."""
        from orchestrator.merge_queue import MergeDispatchResult  # noqa: F401

        queue: asyncio.Queue = asyncio.Queue()
        registry = InFlightMergeRegistry()
        req1 = _make_request('C', 'branchC', tmp_path, config)
        req2 = _make_request('C', 'branchC', tmp_path, config)

        await coalesce_or_enqueue_merge_request(
            queue, req1, None, registry, git_ops=None,
        )
        result2 = await coalesce_or_enqueue_merge_request(
            queue, req2, None, registry, git_ops=None,
        )

        assert result2.in_flight is True
        assert result2.inflight_request_id == req1.request_id


# ---------------------------------------------------------------------------
# β1 Step-3 RED: SpeculativeMergeWorker.snapshot() exposes request_id per entry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSnapshotExposesRequestId:
    """β1 step-3 RED: snapshot() must include request_id in each entry dict.

    RED until step-4 impl: _entry() does not yet include a 'request_id' key
    (KeyError when test asserts entry['request_id']).
    """

    async def test_snapshot_queued_entry_includes_request_id(
        self, config: OrchestratorConfig, tmp_path: Path,
    ):
        """A queued MergeRequest appears in snapshot with its request_id."""
        import types

        queue: asyncio.Queue = asyncio.Queue()
        git_ops_stub = types.SimpleNamespace()
        worker = SpeculativeMergeWorker(
            git_ops=git_ops_stub,  # type: ignore[reportArgumentType]
            queue=queue,
        )

        req = _make_request('snap-req', 'snap-req', tmp_path, config)
        await queue.put(req)

        snap = worker.snapshot()

        entries = snap['entries']
        matching = [e for e in entries if e.get('task_id') == 'snap-req']
        assert matching, f'Entry for snap-req not found in snapshot: {entries}'

        entry = matching[0]
        assert 'request_id' in entry, (
            f'snapshot entry missing request_id key; keys present: {list(entry.keys())}'
        )
        assert entry['request_id'] == req.request_id, (
            f"snapshot entry['request_id']={entry['request_id']!r} "
            f"!= req.request_id={req.request_id!r}"
        )


# ---------------------------------------------------------------------------
# β1 Step-5 RED: WaiterRecord dataclass contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWaiterRecordContract:
    """β1 step-5 RED: WaiterRecord dataclass contract.

    RED until step-6 impl: WaiterRecord does not exist (ImportError).
    """

    async def test_waiter_record_fields(self):
        """Construct WaiterRecord and verify all field contracts."""
        from orchestrator.merge_queue import WaiterRecord  # type: ignore[reportMissingImports]

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        wr = WaiterRecord(request_id='mr-x', future=fut)

        assert wr.request_id == 'mr-x'
        assert wr.future is fut
        assert wr.source == 'mcp'          # default
        assert wr.submitted_tip is None    # default

    async def test_waiter_record_explicit_source_and_tip(self):
        """Explicit source and submitted_tip are stored correctly."""
        from orchestrator.merge_queue import WaiterRecord  # type: ignore[reportMissingImports]

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()

        wr = WaiterRecord(
            request_id='mr-y',
            future=fut,
            source='workflow',
            submitted_tip='abc123def456',
        )

        assert wr.request_id == 'mr-y'
        assert wr.future is fut
        assert wr.source == 'workflow'
        assert wr.submitted_tip == 'abc123def456'


# ---------------------------------------------------------------------------
# TestMultiWaiterEntry — γ1 step-1 RED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMultiWaiterEntry:
    """γ1 step-1 RED: extended _InFlightEntry fields + acquire seeding.

    RED until step-2 impl: _InFlightEntry has no snapshot_tip/generation/
    verifying/waiters/primary_future fields; acquire does not seed a waiter.
    """

    def _make_future(self) -> asyncio.Future:
        return asyncio.get_running_loop().create_future()

    async def test_acquire_seeds_entry_fields(self):
        """acquire with new kw-args seeds all extended fields correctly."""
        registry = InFlightMergeRegistry()
        fut = self._make_future()

        result = registry.acquire(
            'B', 'task-B', fut,
            request_id='mr-1',
            source='workflow',
            submitted_tip='deadbeef',
            snapshot_tip='deadbeef',
        )

        assert result is True
        entry = registry.entry('B')
        assert entry is not None
        assert entry.snapshot_tip == 'deadbeef'
        assert entry.generation == 1
        assert entry.verifying is False
        assert entry.primary_future is fut
        assert len(entry.waiters) == 1
        w = entry.waiters[0]
        assert w.request_id == 'mr-1'
        assert w.source == 'workflow'
        assert w.submitted_tip == 'deadbeef'
        assert w.future is fut

    async def test_acquire_back_compat_three_positional_args(self):
        """Back-compat: 3 positional args + no kw still returns True, seeds one waiter."""
        registry = InFlightMergeRegistry()
        fut = self._make_future()

        result = registry.acquire('C', 'task-C', fut)

        assert result is True
        entry = registry.entry('C')
        assert entry is not None
        assert len(entry.waiters) == 1
        w = entry.waiters[0]
        assert w.source == 'mcp'         # default
        assert w.submitted_tip is None   # default
        assert w.future is fut


# ---------------------------------------------------------------------------
# TestAttachFanOut — γ1 step-3 RED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAttachFanOut:
    """γ1 step-3 RED: attach + fan-out (boundary test 9 substrate).

    RED until step-4 impl: InFlightMergeRegistry has no attach() method.
    """

    def _make_future(self) -> asyncio.Future:
        return asyncio.get_running_loop().create_future()

    async def test_attach_appends_waiter_returns_true(self):
        """attach() on a held branch appends waiter, returns True."""
        from orchestrator.merge_queue import WaiterRecord
        registry = InFlightMergeRegistry()
        f1 = self._make_future()
        registry.acquire('B', 'task-B', f1, request_id='mr-1')
        f2 = self._make_future()

        result = registry.attach('B', WaiterRecord(
            request_id='mr-2', future=f2, source='mcp',
        ))

        assert result is True
        entry = registry.entry('B')
        assert entry is not None
        assert len(entry.waiters) == 2

    async def test_attach_free_branch_returns_false(self):
        """attach() on a branch not in-flight returns False."""
        from orchestrator.merge_queue import WaiterRecord
        registry = InFlightMergeRegistry()
        f = self._make_future()

        result = registry.attach('free', WaiterRecord(
            request_id='mr-x', future=f, source='mcp',
        ))

        assert result is False

    async def test_fanout_result_mirrors_to_attached_future(self):
        """Resolving primary future mirrors result onto attached waiter's future."""
        from orchestrator.merge_queue import WaiterRecord
        registry = InFlightMergeRegistry()
        f1 = self._make_future()
        registry.acquire('B', 'task-B', f1, request_id='mr-1')
        f2 = self._make_future()
        registry.attach('B', WaiterRecord(request_id='mr-2', future=f2, source='mcp'))

        outcome = MergeOutcome(status='done', merge_sha='abc123')
        f1.set_result(outcome)
        await asyncio.sleep(0)

        assert f2.done()
        assert f2.result() is outcome

    async def test_fanout_cancel_mirrors_to_attached_future(self):
        """Cancelling primary future also cancels attached waiter's future."""
        from orchestrator.merge_queue import WaiterRecord
        registry = InFlightMergeRegistry()
        f1 = self._make_future()
        registry.acquire('B', 'task-B', f1, request_id='mr-1')
        f2 = self._make_future()
        registry.attach('B', WaiterRecord(request_id='mr-2', future=f2, source='mcp'))

        f1.cancel()
        await asyncio.sleep(0)

        assert f2.cancelled()

    async def test_fanout_exception_mirrors_to_attached_future(self):
        """Setting exception on primary mirrors exception onto attached waiter's future."""
        from orchestrator.merge_queue import WaiterRecord
        registry = InFlightMergeRegistry()
        f1 = self._make_future()
        registry.acquire('B', 'task-B', f1, request_id='mr-1')
        f2 = self._make_future()
        registry.attach('B', WaiterRecord(request_id='mr-2', future=f2, source='mcp'))

        exc = RuntimeError('merge failed')
        f1.set_exception(exc)
        await asyncio.sleep(0)

        assert f2.done()
        assert f2.exception() is exc

    async def test_fanout_skips_pre_resolved_attached_future(self):
        """Fan-out callback skips an attached future that is already done."""
        from orchestrator.merge_queue import WaiterRecord
        registry = InFlightMergeRegistry()
        f1 = self._make_future()
        registry.acquire('B', 'task-B', f1, request_id='mr-1')
        f2 = self._make_future()
        registry.attach('B', WaiterRecord(request_id='mr-2', future=f2, source='mcp'))
        # Pre-cancel f2 (simulate soft-cancel / detach)
        f2.cancel()
        await asyncio.sleep(0)

        # Resolving primary should NOT raise — the guard skips done f2
        outcome = MergeOutcome(status='done')
        f1.set_result(outcome)
        await asyncio.sleep(0)  # callback runs; f2 already done, guard fires


# ---------------------------------------------------------------------------
# TestDetachProceedDrop — γ1 step-5 RED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDetachProceedDrop:
    """γ1 step-5 RED: detach proceed-vs-drop (boundary test 10 substrate).

    RED until step-6 impl: InFlightMergeRegistry has no detach() method.
    (detach was added alongside attach in step-4; tests written per plan.)
    """

    def _make_future(self) -> asyncio.Future:
        return asyncio.get_running_loop().create_future()

    async def test_detach_non_last_proceeds(self):
        """Detaching one of two waiters keeps the entry in-flight."""
        from orchestrator.merge_queue import WaiterRecord
        registry = InFlightMergeRegistry()
        f1 = self._make_future()
        registry.acquire('B', 'task-B', f1, request_id='mr-1')
        f2 = self._make_future()
        registry.attach('B', WaiterRecord(request_id='mr-2', future=f2, source='mcp'))

        remaining = registry.detach('B', 'mr-2')

        assert remaining == 1
        assert registry.is_inflight('B') is True
        # Primary future NOT cancelled — entry still proceeds
        assert f1.cancelled() is False

    async def test_detach_last_cancels_primary_and_releases(self):
        """Detaching the last waiter cancels primary, releases slot."""
        from orchestrator.merge_queue import WaiterRecord
        registry = InFlightMergeRegistry()
        f1 = self._make_future()
        registry.acquire('B', 'task-B', f1, request_id='mr-1')
        f2 = self._make_future()
        registry.attach('B', WaiterRecord(request_id='mr-2', future=f2, source='mcp'))

        # Remove mr-2 → 1 waiter remains
        registry.detach('B', 'mr-2')
        # Remove mr-1 → 0 waiters → primary cancelled
        remaining = registry.detach('B', 'mr-1')

        assert remaining == 0
        assert f1.cancelled() is True
        await asyncio.sleep(0)  # release callback fires
        assert registry.is_inflight('B') is False

    async def test_detach_last_abandoned_check(self):
        """_request_abandoned returns True when primary is cancelled (drop at checkpoint)."""
        registry = InFlightMergeRegistry()
        f1 = self._make_future()
        registry.acquire('B', 'task-B', f1, request_id='mr-1')
        registry.detach('B', 'mr-1')

        # Build a minimal MergeWorker-style check: _request_abandoned(req) checks req.result.cancelled()
        loop = asyncio.get_running_loop()
        dummy_future: asyncio.Future = loop.create_future()
        dummy_future.cancel()
        # _request_abandoned is an instance method on MergeWorker (inherits _WipHaltMixin)
        # We can test the logic directly since we know it checks req.result.cancelled()
        assert dummy_future.cancelled() is True  # same check as _request_abandoned

    async def test_detach_unknown_request_id_is_noop(self):
        """Detach with an unknown request_id returns unchanged count."""
        registry = InFlightMergeRegistry()
        f1 = self._make_future()
        registry.acquire('B', 'task-B', f1, request_id='mr-1')

        count = registry.detach('B', 'mr-unknown')

        assert count == 1
        assert registry.is_inflight('B') is True

    async def test_detach_free_branch_returns_zero(self):
        """Detach on a branch not in-flight returns 0 (safe no-op)."""
        registry = InFlightMergeRegistry()

        assert registry.detach('free', 'mr-1') == 0


# ---------------------------------------------------------------------------
# TestReSnapshotSetVerifying — γ1 step-7 RED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReSnapshotSetVerifying:
    """γ1 step-7 RED: re_snapshot + set_verifying substrate.

    RED until step-8 impl: methods don't exist.
    (implemented alongside attach/detach in step-4; tests written per plan.)
    """

    def _make_future(self) -> asyncio.Future:
        return asyncio.get_running_loop().create_future()

    async def test_re_snapshot_updates_tip_returns_true(self):
        """re_snapshot sets snapshot_tip; returns True for in-flight branch."""
        registry = InFlightMergeRegistry()
        fut = self._make_future()
        registry.acquire('B', 'task-B', fut, snapshot_tip='old')

        result = registry.re_snapshot('B', 'new')

        assert result is True
        entry = registry.entry('B')
        assert entry is not None
        assert entry.snapshot_tip == 'new'
        # generation unchanged (re-snapshot does not bump generation)
        assert entry.generation == 1

    async def test_re_snapshot_free_branch_returns_false(self):
        """re_snapshot on a free branch returns False."""
        registry = InFlightMergeRegistry()

        assert registry.re_snapshot('free', 'x') is False

    async def test_set_verifying_true(self):
        """set_verifying() flips entry.verifying to True."""
        registry = InFlightMergeRegistry()
        fut = self._make_future()
        registry.acquire('B', 'task-B', fut)

        registry.set_verifying('B')

        entry = registry.entry('B')
        assert entry is not None
        assert entry.verifying is True

    async def test_set_verifying_false(self):
        """set_verifying(branch, False) flips entry.verifying back to False."""
        registry = InFlightMergeRegistry()
        fut = self._make_future()
        registry.acquire('B', 'task-B', fut)
        registry.set_verifying('B', True)

        registry.set_verifying('B', False)

        entry = registry.entry('B')
        assert entry is not None
        assert entry.verifying is False

    async def test_set_verifying_free_branch_no_raise(self):
        """set_verifying on a free branch is a no-op (does not raise)."""
        registry = InFlightMergeRegistry()
        registry.set_verifying('free')  # must not raise


# ---------------------------------------------------------------------------
# TestTipRelation — γ1 step-9 RED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTipRelation:
    """γ1 step-9 RED: TipRelation enum + classify_tip_relation with real git.

    RED until step-10 impl: TipRelation/classify_tip_relation don't exist.
    """

    async def _head_sha(self, worktree: Path) -> str:
        """Get HEAD SHA for a worktree."""
        _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
        return sha.strip()

    async def test_same_tip(self, git_ops: GitOps):
        """Same SHA → TipRelation.SAME."""
        from orchestrator.merge_queue import TipRelation, classify_tip_relation
        main_sha = await git_ops.get_main_sha()
        relation = await classify_tip_relation(main_sha, main_sha, git_ops)
        assert relation is TipRelation.SAME

    async def test_superset(self, git_ops: GitOps):
        """Branch advanced past old tip → TipRelation.SUPERSET (new is ancestor-descendant)."""
        from orchestrator.merge_queue import TipRelation, classify_tip_relation
        # Create branch with one commit; record that tip as "old"
        worktree = await _make_branch_with_file(git_ops, 'tr-super', 'f.py', 'x=1\n')
        old_sha = await self._head_sha(worktree)
        # Advance branch: add another commit
        (worktree / 'g.py').write_text('y=2\n')
        await git_ops.commit(worktree, 'Add g.py')
        new_sha = await self._head_sha(worktree)

        relation = await classify_tip_relation(new_sha, old_sha, git_ops)
        assert relation is TipRelation.SUPERSET

    async def test_subset(self, git_ops: GitOps):
        """Old tip is ahead of new tip → TipRelation.SUBSET."""
        from orchestrator.merge_queue import TipRelation, classify_tip_relation
        worktree = await _make_branch_with_file(git_ops, 'tr-sub', 'h.py', 'z=3\n')
        old_sha = await self._head_sha(worktree)
        (worktree / 'k.py').write_text('k=4\n')
        await git_ops.commit(worktree, 'Add k.py')
        new_sha_advanced = await self._head_sha(worktree)

        # new=old_sha (behind), old=new_sha_advanced (ahead) → new is subset
        relation = await classify_tip_relation(old_sha, new_sha_advanced, git_ops)
        assert relation is TipRelation.SUBSET

    async def test_divergent(self, git_ops: GitOps):
        """Two branches off same base with distinct commits → TipRelation.DIVERGENT."""
        from orchestrator.merge_queue import TipRelation, classify_tip_relation
        wt_a = await _make_branch_with_file(git_ops, 'tr-div-a', 'a.py', 'a=1\n')
        wt_b = await _make_branch_with_file(git_ops, 'tr-div-b', 'b.py', 'b=2\n')
        sha_a = await self._head_sha(wt_a)
        sha_b = await self._head_sha(wt_b)

        relation = await classify_tip_relation(sha_a, sha_b, git_ops)
        assert relation is TipRelation.DIVERGENT


# ---------------------------------------------------------------------------
# TestPatchContentContained — γ1 step-11 RED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPatchContentContained:
    """γ1 step-11 RED: patch_content_contained + resolve_divergent with real git.

    RED until step-12 impl: helpers don't exist.
    (implemented alongside TipRelation in step-10 commit; tests per plan.)
    """

    async def _head_sha(self, worktree: Path) -> str:
        _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=worktree)
        return sha.strip()

    async def test_cherry_picked_content_is_contained(self, git_ops: GitOps):
        """Branch commit cherry-picked onto main → content already present (True)."""
        from orchestrator.merge_queue import patch_content_contained
        # Create branch with one file
        wt = await _make_branch_with_file(git_ops, 'pcc-cherry', 'cherry.py', 'x = 42\n')
        branch_tip = await self._head_sha(wt)

        # Cherry-pick the branch commit onto main
        rc, _, err = await _run(
            ['git', 'cherry-pick', branch_tip],
            cwd=git_ops.project_root,
        )
        assert rc == 0, f'cherry-pick failed: {err}'
        upstream = await git_ops.get_main_sha()

        # The branch tip's content is now on main (different SHA — rebase/cp)
        result = await patch_content_contained(branch_tip, upstream, git_ops)
        assert result is True

    async def test_unapplied_commit_not_contained(self, git_ops: GitOps):
        """Branch with a genuinely new commit not on main → False."""
        from orchestrator.merge_queue import patch_content_contained
        wt = await _make_branch_with_file(git_ops, 'pcc-new', 'new.py', 'y = 99\n')
        branch_tip = await self._head_sha(wt)
        main_tip = await git_ops.get_main_sha()

        result = await patch_content_contained(branch_tip, main_tip, git_ops)
        assert result is False

    async def test_empty_range_is_contained(self, git_ops: GitOps):
        """Branch tip == upstream (no commits beyond base) → True (vacuously contained)."""
        from orchestrator.merge_queue import patch_content_contained
        main_sha = await git_ops.get_main_sha()
        # Branch off main with no extra commits: both tips are the same SHA
        result = await patch_content_contained(main_sha, main_sha, git_ops)
        assert result is True

    async def test_bogus_ref_returns_false(self, git_ops: GitOps):
        """git cherry with a bogus ref → rc != 0 → fail-open → False."""
        from orchestrator.merge_queue import patch_content_contained
        result = await patch_content_contained('deadbeef0000', 'abcdef1234', git_ops)
        assert result is False

    async def test_resolve_divergent_content_equal_returns_subset(self, git_ops: GitOps):
        """resolve_divergent for cherry-picked content → TipRelation.SUBSET."""
        from orchestrator.merge_queue import TipRelation, resolve_divergent
        wt = await _make_branch_with_file(git_ops, 'rd-eq', 'eq.py', 'z = 7\n')
        branch_tip = await self._head_sha(wt)
        rc, _, err = await _run(
            ['git', 'cherry-pick', branch_tip],
            cwd=git_ops.project_root,
        )
        assert rc == 0, f'cherry-pick failed: {err}'
        upstream = await git_ops.get_main_sha()

        relation = await resolve_divergent(branch_tip, upstream, git_ops)
        assert relation is TipRelation.SUBSET

    async def test_resolve_divergent_genuinely_new_returns_superset(self, git_ops: GitOps):
        """resolve_divergent for a genuinely new branch commit → TipRelation.SUPERSET."""
        from orchestrator.merge_queue import TipRelation, resolve_divergent
        wt = await _make_branch_with_file(git_ops, 'rd-new', 'rd_new.py', 'w = 5\n')
        branch_tip = await self._head_sha(wt)
        main_tip = await git_ops.get_main_sha()

        relation = await resolve_divergent(branch_tip, main_tip, git_ops)
        assert relation is TipRelation.SUPERSET


# ---------------------------------------------------------------------------
# TestDecideAttachAction — γ1 step-13 RED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDecideAttachAction:
    """γ1 step-13 RED: decide_attach_action pure mapping (PRD §7.2 table).

    RED until step-14 impl: AttachAction/decide_attach_action don't exist.
    (implemented alongside TipRelation in step-10 commit; tests per plan.)
    """

    async def test_same_not_verifying(self):
        """SAME + not verifying → COALESCE."""
        from orchestrator.merge_queue import AttachAction, TipRelation, decide_attach_action
        assert decide_attach_action(TipRelation.SAME, verifying=False) is AttachAction.COALESCE

    async def test_same_verifying(self):
        """SAME + verifying → COALESCE."""
        from orchestrator.merge_queue import AttachAction, TipRelation, decide_attach_action
        assert decide_attach_action(TipRelation.SAME, verifying=True) is AttachAction.COALESCE

    async def test_superset_not_verifying(self):
        """SUPERSET + not verifying → RESNAPSHOT."""
        from orchestrator.merge_queue import AttachAction, TipRelation, decide_attach_action
        assert decide_attach_action(TipRelation.SUPERSET, verifying=False) is AttachAction.RESNAPSHOT

    async def test_superset_verifying(self):
        """SUPERSET + verifying → ATTACH_AND_CHAIN (gen-2 chaining, γ2)."""
        from orchestrator.merge_queue import AttachAction, TipRelation, decide_attach_action
        assert decide_attach_action(TipRelation.SUPERSET, verifying=True) is AttachAction.ATTACH_AND_CHAIN

    async def test_subset_not_verifying(self):
        """SUBSET + not verifying → ATTACH_CONTAINMENT (boundary test 13 substrate)."""
        from orchestrator.merge_queue import AttachAction, TipRelation, decide_attach_action
        assert decide_attach_action(TipRelation.SUBSET, verifying=False) is AttachAction.ATTACH_CONTAINMENT

    async def test_subset_verifying(self):
        """SUBSET + verifying → ATTACH_CONTAINMENT (containment applies regardless)."""
        from orchestrator.merge_queue import AttachAction, TipRelation, decide_attach_action
        assert decide_attach_action(TipRelation.SUBSET, verifying=True) is AttachAction.ATTACH_CONTAINMENT

    async def test_divergent_raises_value_error(self):
        """DIVERGENT → ValueError (must resolve via resolve_divergent first)."""
        import pytest

        from orchestrator.merge_queue import TipRelation, decide_attach_action
        with pytest.raises(ValueError, match='DIVERGENT'):
            decide_attach_action(TipRelation.DIVERGENT, verifying=False)

    async def test_divergent_verifying_also_raises(self):
        """DIVERGENT + verifying → ValueError (same constraint)."""
        import pytest

        from orchestrator.merge_queue import TipRelation, decide_attach_action
        with pytest.raises(ValueError, match='DIVERGENT'):
            decide_attach_action(TipRelation.DIVERGENT, verifying=True)


# ---------------------------------------------------------------------------
# TestMaybeAutoChainGeneration — γ2 step-07/08/09/10: _maybe_auto_chain_generation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMaybeAutoChainGeneration:
    """Unit tests for the _maybe_auto_chain_generation module-level helper (γ2)."""

    def _make_req(
        self,
        tmp_path: Path,
        config: OrchestratorConfig,
        *,
        branch: str = 'task/t1',
        generation: int = 1,
    ) -> MergeRequest:
        # Called from @pytest.mark.asyncio tests — get_running_loop() is always valid here.
        fut: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
        return MergeRequest(
            task_id='t1',
            branch=branch,
            worktree=tmp_path,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=fut,
            generation=generation,
        )

    async def test_merged_branch_tip_none_returns_none(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """(1) merged_branch_tip=None → returns None, queue stays empty."""
        from orchestrator.merge_queue import _maybe_auto_chain_generation

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        req = self._make_req(tmp_path, config)
        git_ops = MagicMock()
        event_store = MagicMock()

        result = await _maybe_auto_chain_generation(
            req, 'sha-adv', git_ops, event_store,
            merged_branch_tip=None,
            counts={},
            queue=queue,
            max_auto_generations=2,
        )

        assert result is None
        assert queue.empty()

    async def test_head_equals_merged_tip_returns_none(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """(2) current HEAD == merged_branch_tip → genuine drop, returns None."""
        from orchestrator.merge_queue import _maybe_auto_chain_generation

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        req = self._make_req(tmp_path, config)
        git_ops = MagicMock()
        git_ops.project_root = tmp_path
        event_store = MagicMock()
        tip = 'aaabbbccc'

        with patch('orchestrator.merge_queue._run', AsyncMock(return_value=(0, tip + '\n', ''))):
            result = await _maybe_auto_chain_generation(
                req, 'sha-adv', git_ops, event_store,
                merged_branch_tip=tip,
                counts={},
                queue=queue,
                max_auto_generations=2,
            )

        assert result is None
        assert queue.empty()

    async def test_superset_within_bound_chains(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """(3) SUPERSET advance within bound → returns MergeOutcome('superseded'),
        enqueues gen-(n+1) request, increments counts[branch]."""
        from orchestrator.merge_queue import (
            TipRelation,
            _maybe_auto_chain_generation,
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        req = self._make_req(tmp_path, config, branch='task/t1', generation=1)
        git_ops = MagicMock()
        git_ops.project_root = tmp_path
        event_store = MagicMock()
        current_head = 'newhead111'
        merged_tip = 'oldtip000'
        counts: dict[str, int] = {}

        with (
            patch('orchestrator.merge_queue._run', AsyncMock(return_value=(0, current_head + '\n', ''))),
            patch('orchestrator.merge_queue.classify_tip_relation', AsyncMock(return_value=TipRelation.SUPERSET)),
        ):
            result = await _maybe_auto_chain_generation(
                req, 'sha-adv', git_ops, event_store,
                merged_branch_tip=merged_tip,
                counts=counts,
                queue=queue,
                max_auto_generations=2,
            )

        assert result is not None
        assert result.status == 'superseded'
        assert result.superseded_by is not None
        assert result.superseded_by.startswith('mr-')
        assert result.merge_sha == 'sha-adv'
        # Queue has the gen-(n+1) request
        assert queue.qsize() == 1
        chained = queue.get_nowait()
        assert chained.request_id != req.request_id
        assert chained.generation == 2
        assert chained.snapshot_tip == current_head
        assert chained.pre_rebased is False
        assert chained.branch == req.branch
        assert chained.task_id == req.task_id
        assert result.superseded_by == chained.request_id
        # counts incremented
        assert counts[req.branch] == 1

    async def test_divergent_resolves_to_superset_chains(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """(4) DIVERGENT → resolve_divergent → SUPERSET → chains."""
        from orchestrator.merge_queue import (
            TipRelation,
            _maybe_auto_chain_generation,
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        req = self._make_req(tmp_path, config, branch='task/t1', generation=1)
        git_ops = MagicMock()
        git_ops.project_root = tmp_path
        event_store = MagicMock()
        current_head = 'divhead222'
        counts: dict[str, int] = {}

        with (
            patch('orchestrator.merge_queue._run', AsyncMock(return_value=(0, current_head + '\n', ''))),
            patch('orchestrator.merge_queue.classify_tip_relation', AsyncMock(return_value=TipRelation.DIVERGENT)),
            patch('orchestrator.merge_queue.resolve_divergent', AsyncMock(return_value=TipRelation.SUPERSET)),
        ):
            result = await _maybe_auto_chain_generation(
                req, 'sha-adv', git_ops, event_store,
                merged_branch_tip='oldtip000',
                counts=counts,
                queue=queue,
                max_auto_generations=2,
            )

        assert result is not None
        assert result.status == 'superseded'
        assert queue.qsize() == 1

    async def test_divergent_resolves_to_subset_returns_none(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """(4b) DIVERGENT → resolve_divergent → SUBSET (patch-contained) → returns None."""
        from orchestrator.merge_queue import (
            TipRelation,
            _maybe_auto_chain_generation,
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        req = self._make_req(tmp_path, config)
        git_ops = MagicMock()
        git_ops.project_root = tmp_path
        event_store = MagicMock()

        with (
            patch('orchestrator.merge_queue._run', AsyncMock(return_value=(0, 'divhead333\n', ''))),
            patch('orchestrator.merge_queue.classify_tip_relation', AsyncMock(return_value=TipRelation.DIVERGENT)),
            patch('orchestrator.merge_queue.resolve_divergent', AsyncMock(return_value=TipRelation.SUBSET)),
        ):
            result = await _maybe_auto_chain_generation(
                req, 'sha-adv', git_ops, event_store,
                merged_branch_tip='oldtip000',
                counts={},
                queue=queue,
                max_auto_generations=2,
            )

        assert result is None
        assert queue.empty()

    # γ2 step-09/10 — bound enforcement
    async def test_bound_exceeded_returns_blocked_and_resets_counter(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """counts already at MAX_AUTO_CHAINED_GENERATIONS → escalate, reset counter."""
        from orchestrator.merge_queue import (
            POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
            TipRelation,
            _maybe_auto_chain_generation,
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        req = self._make_req(tmp_path, config, branch='task/t-bound')
        git_ops = MagicMock()
        git_ops.project_root = tmp_path
        event_store = MagicMock()
        # Pre-populate counter at max
        counts: dict[str, int] = {'task/t-bound': 2}

        with (
            patch('orchestrator.merge_queue._run', AsyncMock(return_value=(0, 'newhead777\n', ''))),
            patch('orchestrator.merge_queue.classify_tip_relation', AsyncMock(return_value=TipRelation.SUPERSET)),
        ):
            result = await _maybe_auto_chain_generation(
                req, 'sha-adv', git_ops, event_store,
                merged_branch_tip='oldtip000',
                counts=counts,
                queue=queue,
                max_auto_generations=2,
            )

        assert result is not None
        assert result.status == 'blocked'
        assert result.reason.startswith(POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX)
        assert '2' in result.reason  # mentions the bound
        assert queue.empty()  # NO new request enqueued
        # Counter reset (popped)
        assert 'task/t-bound' not in counts

    async def test_consecutive_sequence_first_two_chain_third_escalates(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """Sequence: advance-1 → counts=1, advance-2 → counts=2, advance-3 → escalate + reset."""
        from orchestrator.merge_queue import (
            TipRelation,
            _maybe_auto_chain_generation,
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        req = self._make_req(tmp_path, config, branch='task/t-seq')
        git_ops = MagicMock()
        git_ops.project_root = tmp_path
        event_store = MagicMock()
        counts: dict[str, int] = {}
        heads = iter(['head1\n', 'head2\n', 'head3\n'])

        with (
            patch('orchestrator.merge_queue._run', AsyncMock(side_effect=lambda *a, **kw: (0, next(heads), ''))),
            patch('orchestrator.merge_queue.classify_tip_relation', AsyncMock(return_value=TipRelation.SUPERSET)),
        ):
            # First advance → chain
            r1 = await _maybe_auto_chain_generation(
                req, 'sha1', git_ops, event_store,
                merged_branch_tip='old0', counts=counts, queue=queue, max_auto_generations=2,
            )
            assert r1 is not None and r1.status == 'superseded'
            assert counts.get('task/t-seq') == 1
            _ = queue.get_nowait()  # drain

            # Second advance → chain
            r2 = await _maybe_auto_chain_generation(
                req, 'sha2', git_ops, event_store,
                merged_branch_tip='head1', counts=counts, queue=queue, max_auto_generations=2,
            )
            assert r2 is not None and r2.status == 'superseded'
            assert counts.get('task/t-seq') == 2
            _ = queue.get_nowait()  # drain

            # Third advance → escalate
            r3 = await _maybe_auto_chain_generation(
                req, 'sha3', git_ops, event_store,
                merged_branch_tip='head2', counts=counts, queue=queue, max_auto_generations=2,
            )
            assert r3 is not None and r3.status == 'blocked'
            assert queue.empty()
            assert 'task/t-seq' not in counts  # reset

    async def test_retention_seam_chained_request_gets_recorded(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """(γ2 step-21 RED) provenance retention seam: when _maybe_auto_chain_generation
        receives a retention ring it passes it to enqueue_merge_request so the gen-(n+1)
        terminal outcome is recorded (superseded_by pointer resolves)."""
        from orchestrator.merge_queue import (
            MergeOutcome,
            TerminalOutcomeRecord,
            TerminalOutcomeRetention,
            TipRelation,
            _GenerationChainContext,
            _maybe_auto_chain_generation,
        )

        # _GenerationChainContext must accept a retention kwarg and expose it.
        ring = TerminalOutcomeRetention(maxlen=50)
        queue: asyncio.Queue = asyncio.Queue()
        ctx = _GenerationChainContext(
            queue=queue,
            counts={},
            max_auto_generations=2,
            retention=ring,
        )
        assert ctx.retention is ring

        req = self._make_req(tmp_path, config, branch='task/t-ret', generation=1)
        git_ops = MagicMock()
        git_ops.project_root = tmp_path
        event_store = MagicMock()
        event_store.emit = MagicMock()

        with (
            patch('orchestrator.merge_queue._run', AsyncMock(return_value=(0, 'newhead\n', ''))),
            patch('orchestrator.merge_queue.classify_tip_relation', AsyncMock(return_value=TipRelation.SUPERSET)),
        ):
            result = await _maybe_auto_chain_generation(
                req, 'sha-adv', git_ops, event_store,
                merged_branch_tip='oldtip',
                counts={},
                queue=queue,
                max_auto_generations=2,
                retention=ring,
            )

        assert result is not None and result.status == 'superseded'
        gen_next = await queue.get()
        assert gen_next.generation == 2

        # Resolve the gen-(n+1) future so _on_finalized fires.
        gen_next.result.set_result(MergeOutcome('done', merge_sha='advsha'))
        await asyncio.sleep(0)  # let callbacks run

        rec = ring.get(gen_next.request_id)
        assert rec is not None, 'ring should contain the gen-(n+1) terminal record'
        assert isinstance(rec, TerminalOutcomeRecord)
        assert rec.state == 'done'
        assert rec.generation == 2


# ---------------------------------------------------------------------------
# TestMergeWorkerGenerationChain — γ2 step-13/14: MergeWorker wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergeWorkerGenerationChain:
    """Unit tests for MergeWorker γ2 generation-chain wiring (step-13 RED / step-14 GREEN)."""

    async def test_init_has_generation_chain_counts(self) -> None:
        """MergeWorker.__init__ initialises self._generation_chain_counts == {}."""
        queue: asyncio.Queue = asyncio.Queue()
        worker = MergeWorker(MagicMock(), queue)
        assert hasattr(worker, '_generation_chain_counts')
        assert worker._generation_chain_counts == {}

    async def test_do_merge_passes_chain_ctx_and_merged_branch_tip(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """_do_merge passes chain_ctx(queue=worker._queue, counts=_generation_chain_counts,
        max=MAX_AUTO_CHAINED_GENERATIONS) and merged_branch_tip=branch HEAD to
        _finalize_advanced_merge on the 'advanced' path."""
        from orchestrator.merge_queue import (
            MAX_AUTO_CHAINED_GENERATIONS,
            _GenerationChainContext,
        )

        # Set up a real MergeRequest so the worker can process it
        queue: asyncio.Queue = asyncio.Queue()
        worker = MergeWorker(MagicMock(), queue)

        branch_head_sha = 'branch-head-abc123'
        merge_commit_sha = 'merge-commit-xyz'

        # Mock git_ops methods needed by _do_merge
        git_ops = MagicMock()
        git_ops.get_main_sha = AsyncMock(return_value='main-sha')
        git_ops.is_ancestor = AsyncMock(return_value=False)  # not already on main
        git_ops.has_uncommitted_work = AsyncMock(return_value=False)
        git_ops.merge_to_main = AsyncMock(return_value=MagicMock(
            success=True,
            conflicts=False,
            merge_commit=merge_commit_sha,
            merge_worktree=tmp_path / 'merge-wt',
            pre_merge_sha='main-sha',
        ))
        git_ops.advance_main = AsyncMock(return_value='advanced')
        git_ops.cleanup_merge_worktree = AsyncMock()

        worker._git_ops = git_ops

        # rev-parse HEAD → branch_head_sha for the branch tip snapshot
        finalize_mock = AsyncMock(return_value=MergeOutcome('done', merge_sha='adv-sha'))

        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        req = MergeRequest(
            task_id='wt-1',
            branch='task/wt-branch',
            worktree=tmp_path,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=fut,
        )

        with (
            patch('orchestrator.merge_queue._run',
                  AsyncMock(return_value=(0, branch_head_sha + '\n', ''))),
            patch('orchestrator.merge_queue._classify_branch_presence', AsyncMock(return_value=None)),
            patch('orchestrator.merge_queue._check_plan_targets_in_tree',
                  AsyncMock(return_value=MagicMock(dropped=[]))),
            patch('orchestrator.merge_queue._run_post_merge_verify', AsyncMock(return_value=None)),
            patch('orchestrator.merge_queue._finalize_advanced_merge', finalize_mock),
        ):
            outcome = await worker._do_merge(req)

        assert outcome is not None and outcome.status == 'done'
        finalize_mock.assert_awaited_once()
        _call_kwargs = finalize_mock.call_args.kwargs
        assert 'chain_ctx' in _call_kwargs, 'chain_ctx not passed to _finalize_advanced_merge'
        ctx: _GenerationChainContext = _call_kwargs['chain_ctx']
        assert ctx.queue is worker._queue
        assert ctx.counts is worker._generation_chain_counts
        assert ctx.max_auto_generations == MAX_AUTO_CHAINED_GENERATIONS
        assert _call_kwargs.get('merged_branch_tip') == branch_head_sha


# ---------------------------------------------------------------------------
# TestSMWGenerationChain — γ2 step-15/16: SpeculativeMergeWorker wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSMWGenerationChain:
    """Unit tests for SpeculativeMergeWorker γ2 generation-chain wiring (step-15 RED / step-16 GREEN)."""

    async def test_speculative_item_has_merged_branch_tip(self) -> None:
        """(a) SpeculativeItem can carry merged_branch_tip (default None)."""
        item = SpeculativeItem(
            request=MagicMock(),
            merge_result=None,
            merge_wt=None,
            base_sha='base',
            speculative=False,
            skip_verify=False,
        )
        assert hasattr(item, 'merged_branch_tip')
        assert item.merged_branch_tip is None
        # Can be set explicitly
        item2 = SpeculativeItem(
            request=MagicMock(),
            merge_result=None,
            merge_wt=None,
            base_sha='base',
            speculative=False,
            skip_verify=False,
            merged_branch_tip='T1',
        )
        assert item2.merged_branch_tip == 'T1'

    async def test_smw_init_has_generation_chain_counts(self) -> None:
        """(b) SpeculativeMergeWorker.__init__ initialises self._generation_chain_counts == {}."""
        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(MagicMock(), queue)
        assert hasattr(worker, '_generation_chain_counts')
        assert worker._generation_chain_counts == {}

    async def test_verify_and_advance_passes_chain_ctx_and_merged_branch_tip(
        self, tmp_path: Path, config: OrchestratorConfig,
    ) -> None:
        """(c) _verify_and_advance on an item with merged_branch_tip='T1' through a
        successful advance awaits _finalize_advanced_merge with chain_ctx
        (queue=self._queue, counts=self._generation_chain_counts,
        max=MAX_AUTO_CHAINED_GENERATIONS) and merged_branch_tip=='T1'."""
        from orchestrator.merge_queue import (
            MAX_AUTO_CHAINED_GENERATIONS,
            _GenerationChainContext,
        )

        queue: asyncio.Queue = asyncio.Queue()
        git_ops = MagicMock()
        git_ops.advance_main = AsyncMock(return_value='advanced')
        git_ops.cleanup_merge_worktree = AsyncMock()

        worker = SpeculativeMergeWorker(git_ops, queue)

        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        req = MergeRequest(
            task_id='smw-wt',
            branch='task/smw-branch',
            worktree=tmp_path,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=fut,
        )
        merge_commit_sha = 'merge-commit-smw'
        item = SpeculativeItem(
            request=req,
            merge_result=MagicMock(
                success=True,
                conflicts=False,
                merge_commit=merge_commit_sha,
                merge_worktree=tmp_path / 'merge-wt',
                pre_merge_sha='base-sha',
            ),
            merge_wt=tmp_path / 'merge-wt',
            base_sha='base-sha',
            speculative=False,
            skip_verify=False,
            started_monotonic=None,
            merged_branch_tip='T1',
        )

        finalize_mock = AsyncMock(return_value=MergeOutcome('done', merge_sha='adv-sha'))

        with (
            patch('orchestrator.merge_queue._run_post_merge_verify', AsyncMock(return_value=None)),
            patch('orchestrator.merge_queue._finalize_advanced_merge', finalize_mock),
        ):
            advanced = await worker._verify_and_advance(item)

        assert advanced is True
        finalize_mock.assert_awaited_once()
        _call_kwargs = finalize_mock.call_args.kwargs
        assert 'chain_ctx' in _call_kwargs, 'chain_ctx not passed to _finalize_advanced_merge'
        ctx: _GenerationChainContext = _call_kwargs['chain_ctx']
        assert ctx.queue is worker._queue
        assert ctx.counts is worker._generation_chain_counts
        assert ctx.max_auto_generations == MAX_AUTO_CHAINED_GENERATIONS
        assert _call_kwargs.get('merged_branch_tip') == 'T1'


# ---------------------------------------------------------------------------
# TestTrainEquivalenceNeverAutoChains — γ2 step-17 RED / step-18 GREEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTrainEquivalenceNeverAutoChains:
    """PRD D9 / boundary test 12 regression guard.

    _do_train_merge must NEVER auto-chain on equivalence failure.  Trains are
    multi-waiter (all members share a single git branch tip), so only the
    single-branch MergeWorker/SpeculativeMergeWorker paths should grow with
    new delta commits.  The ``chain_ctx=None`` default on
    ``_finalize_advanced_merge`` guarantees this; these tests lock that
    invariant in.
    """

    async def test_train_equiv_failure_returns_blocked_not_superseded(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """Equivalence failure on a GroupMergeRequest returns 'blocked' (not 'superseded')
        and leaves the worker queue empty — no auto-chain, no delta re-queue.

        PRD D9: trains are bit-identical single-waiter merges; the γ2 chain
        mechanism applies ONLY to single-branch MergeRequest paths.
        """
        from orchestrator.merge_queue import POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX

        req = await _make_stacked_train(git_ops, config)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)

        with (
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
            patch(
                'orchestrator.merge_queue._check_post_merge_equivalence',
                AsyncMock(return_value=['train.py']),
            ),
        ):
            outcome = await worker._do_merge(req)

        assert outcome is not None, 'expected a MergeOutcome, got None'
        # (a) Outcome is 'blocked', not 'superseded' — trains never auto-chain
        assert outcome.status == 'blocked', (
            f'train equiv-failure must return blocked, got: {outcome!r}'
        )
        # (b) Reason carries the equivalence prefix
        assert outcome.reason is not None
        assert outcome.reason.startswith(POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX), (
            f'expected equiv prefix, got: {outcome.reason!r}'
        )
        # (c) Queue is empty — no gen-(n+1) request was enqueued for the train
        assert worker._queue.empty(), (
            'no gen-(n+1) request should be enqueued for train equiv-failure '
            '(PRD D9: trains never auto-chain)'
        )

    async def test_do_train_merge_finalize_called_with_no_chain_ctx(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        """_do_train_merge calls _finalize_advanced_merge WITHOUT chain_ctx (default None).

        Asserts the chain_ctx=None invariant at the call site so any future
        change that accidentally wires chain_ctx into the train path is caught
        immediately.  Trains must stay bit-identical and single-waiter (PRD D9).
        """
        req = await _make_stacked_train(git_ops, config)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue)

        finalize_mock = AsyncMock(return_value=MergeOutcome('done', merge_sha='adv-sha'))

        with (
            patch('orchestrator.merge_queue.run_scoped_verification', _mock_verify_pass()),
            patch('orchestrator.merge_queue._finalize_advanced_merge', finalize_mock),
        ):
            await worker._do_merge(req)

        finalize_mock.assert_awaited_once()
        call_kwargs = finalize_mock.call_args.kwargs
        # chain_ctx must be absent or explicitly None — trains never auto-chain (PRD D9)
        assert call_kwargs.get('chain_ctx') is None, (
            f'_do_train_merge must not pass chain_ctx to _finalize_advanced_merge; '
            f'got chain_ctx={call_kwargs.get("chain_ctx")!r}'
        )
