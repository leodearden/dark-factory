"""Tests for re-deriving plan.json step-completion status from branch state
after an inter-iteration rebase (task 2387).

Recurring failure mode: after a rebase, a plan step's implementation is
already committed on the branch (often folded into a WIP safety-commit) but
plan.json's status is stale "pending" — the EXECUTE loop then treats it as a
genuine remaining blocker and re-attempts already-done work. This suite
covers the new reconciliation machinery that fixes that:

  - TaskWorkflow._rederive_step_status_from_branch_state (workflow.py)
  - _inter_iteration_rebase wiring the re-derivation in right after
    update_base_commit

Complementary to (and non-overlapping with) task 2386's
``_reconcile_done_step_commits``, covered by test_reconcile_done_step_commits.py:
that method reconciles a *done* step's stale ``commit``; this suite covers
re-deriving a *pending* step's status back to ``done`` in the first place.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow

# ---------------------------------------------------------------------------
# Fixtures — mirrors test_harness_wip_step_detection.py's git_repo/config/
# git_ops/task_assignment/_make_workflow pattern (real temp git repo + real
# worktree via git_ops.create_worktree; heavy collaborators mocked).
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'lib.py').write_text('x = 1\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial'], cwd=repo)


@pytest.fixture
def config(git_repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


@pytest.fixture
def git_ops(config: OrchestratorConfig) -> GitOps:
    return GitOps(config.git, config.project_root)


@pytest.fixture
def task_assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id='42',
        task={
            'id': '42', 'title': 'X', 'description': '',
            'status': 'pending', 'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


def _make_workflow(
    config: OrchestratorConfig,
    git_ops: GitOps,
    assignment: TaskAssignment,
    worktree: Path,
) -> tuple[TaskWorkflow, TaskArtifacts]:
    """Wire a minimal TaskWorkflow with heavy collaborators mocked.

    Mirrors the pattern in test_harness_wip_step_detection.py._make_workflow.
    """
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=MagicMock(),  # type: ignore[arg-type]
        briefing=MagicMock(),  # type: ignore[arg-type]
        mcp=MagicMock(),  # type: ignore[arg-type]
    )
    workflow.worktree = worktree
    artifacts = TaskArtifacts(worktree)
    artifacts.init('42', 'X', 'desc', base_commit='base-sha-old')
    workflow.artifacts = artifacts
    workflow.plan = {'task_id': '42', 'steps': [], 'prerequisites': []}
    return workflow, artifacts


def _write_plan(
    artifacts: TaskArtifacts,
    workflow: TaskWorkflow,
    steps: list[dict],
    prerequisites: list[dict] | None = None,
) -> dict:
    """Persist a plan with the given step dicts and stamp provenance,
    returning the re-read plan so ``workflow.plan`` mirrors what's on disk —
    required because ``update_step_status`` reads/writes plan.json directly,
    independent of any in-memory ``workflow.plan`` the caller also sets.

    ``prerequisites`` defaults to ``[]`` — most callers only care about the
    'steps' collection; pass it explicitly to also seed 'prerequisites'."""
    plan = {
        'task_id': '42',
        'title': 'X',
        'analysis': 'A',
        'prerequisites': prerequisites or [],
        'steps': steps,
    }
    artifacts.write_plan(plan)
    artifacts.stamp_plan_provenance(workflow.session_id)
    return artifacts.read_plan()


# ---------------------------------------------------------------------------
# step-1 RED: happy-path re-derivation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRederiveStepStatusHappyPath:
    async def test_pending_step_completed_in_log_is_rederived_to_done(
        self, config, git_ops, task_assignment,
    ):
        """A step recorded 'pending' in plan.json but marked complete in the
        durable iteration log (with a real branch commit beyond base) is
        re-derived to 'done', while an unrelated pending step is untouched."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'impl.py').write_text('implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
            {'id': 'step-2', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        artifacts.append_iteration_log({
            'agent': 'implementer',
            'steps_completed': ['step-1'],
            'commit': step_commit,
        })

        result = await workflow._rederive_step_status_from_branch_state()

        assert result == ['step-1']
        plan = artifacts.read_plan()
        step_1 = next(s for s in plan['steps'] if s['id'] == 'step-1')
        step_2 = next(s for s in plan['steps'] if s['id'] == 'step-2')
        assert step_1['status'] == 'done'
        assert step_1['commit'], 'Expected a non-null commit recorded on the re-derived step'
        assert step_2['status'] == 'pending', (
            'step-2 was never in steps_completed and must stay pending'
        )


# ---------------------------------------------------------------------------
# Reviewer amendment: prerequisite re-derivation. The implementation walks
# both ('prerequisites', 'steps') collections; every other test in this file
# only populates 'steps', so a regression that dropped the 'prerequisites'
# iteration would pass the suite undetected. Pin it directly.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRederiveStepStatusPrerequisite:
    async def test_pending_prerequisite_completed_in_log_is_rederived_to_done(
        self, config, git_ops, task_assignment,
    ):
        """A *prerequisite* recorded 'pending' in plan.json but marked
        complete in the durable iteration log is re-derived to 'done', the
        same as a step — while an unrelated pending step (not in
        steps_completed) stays untouched."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'impl.py').write_text('implementation\n')
        prereq_commit = await git_ops.commit(wt, 'feat: GREEN — prereq-1')
        assert prereq_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_plan(
            artifacts, workflow,
            steps=[{'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None}],
            prerequisites=[
                {'id': 'prereq-1', 'type': 'impl', 'status': 'pending', 'commit': None},
            ],
        )
        artifacts.append_iteration_log({
            'agent': 'implementer',
            'steps_completed': ['prereq-1'],
            'commit': prereq_commit,
        })

        result = await workflow._rederive_step_status_from_branch_state()

        assert result == ['prereq-1']
        plan = artifacts.read_plan()
        prereq = next(p for p in plan['prerequisites'] if p['id'] == 'prereq-1')
        step = next(s for s in plan['steps'] if s['id'] == 'step-1')
        assert prereq['status'] == 'done'
        assert prereq['commit'], (
            'Expected a non-null commit recorded on the re-derived prerequisite'
        )
        assert step['status'] == 'pending', (
            'step-1 was never in steps_completed and must stay pending'
        )


# ---------------------------------------------------------------------------
# step-3 RED: contamination / no-work guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRederiveStepStatusNoWorkGuard:
    async def test_orphan_log_on_unadvanced_branch_is_not_rederived(
        self, config, git_ops, task_assignment,
    ):
        """An inherited/contaminated iterations.jsonl entry claiming step-1
        completed must NOT be trusted when the branch HEAD has not actually
        diverged from base (no real commit was made) — the SHA-primary
        `_has_prior_implementation` guard must reject it."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        artifacts.append_iteration_log({
            'agent': 'implementer',
            'steps_completed': ['step-1'],
            'commit': 'orphan-sha',
        })

        result = await workflow._rederive_step_status_from_branch_state()

        assert result == []
        plan = artifacts.read_plan()
        assert plan['steps'][0]['status'] == 'pending'


# ---------------------------------------------------------------------------
# step-5 RED: crash-safety / best-effort contract (mirrors
# TestDetectTipWipCommits/TestReconcileDoneStepCommits' defensive tests)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRederiveStepStatusDefensive:
    async def test_worktree_none_returns_empty_and_does_not_raise(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'impl.py').write_text('implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        artifacts.append_iteration_log({
            'agent': 'implementer', 'steps_completed': ['step-1'], 'commit': step_commit,
        })
        workflow.worktree = None

        result = await workflow._rederive_step_status_from_branch_state()

        assert result == []
        assert artifacts.read_plan()['steps'][0]['status'] == 'pending'

    async def test_git_ops_none_returns_empty_and_does_not_raise(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'impl.py').write_text('implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        artifacts.append_iteration_log({
            'agent': 'implementer', 'steps_completed': ['step-1'], 'commit': step_commit,
        })
        workflow.git_ops = None  # type: ignore[assignment]

        result = await workflow._rederive_step_status_from_branch_state()

        assert result == []
        assert artifacts.read_plan()['steps'][0]['status'] == 'pending'

    async def test_internal_failure_returns_empty_and_does_not_raise(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'impl.py').write_text('implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        artifacts.append_iteration_log({
            'agent': 'implementer', 'steps_completed': ['step-1'], 'commit': step_commit,
        })
        workflow._get_head_commit = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError('boom'),
        )

        result = await workflow._rederive_step_status_from_branch_state()

        assert result == []
        assert artifacts.read_plan()['steps'][0]['status'] == 'pending'


# ---------------------------------------------------------------------------
# step-7 RED: observability log entry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRederiveStepStatusLogEntry:
    async def test_rederive_emits_plan_step_rederive_log_entry(
        self, config, git_ops, task_assignment,
    ):
        """Positive: a genuine re-derivation writes exactly one
        plan_step_rederive iteration-log entry naming the re-derived id(s)."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'impl.py').write_text('implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        artifacts.append_iteration_log({
            'agent': 'implementer', 'steps_completed': ['step-1'], 'commit': step_commit,
        })

        result = await workflow._rederive_step_status_from_branch_state()
        assert result == ['step-1']

        entries, _ = artifacts.read_iteration_log()
        rederive_entries = [e for e in entries if e.get('event') == 'plan_step_rederive']
        assert len(rederive_entries) == 1, (
            f'Expected exactly one plan_step_rederive entry, got {rederive_entries}'
        )
        assert rederive_entries[0]['rederived_steps'] == ['step-1']

    async def test_no_rederive_emits_no_log_entry(
        self, config, git_ops, task_assignment,
    ):
        """Negative: reuses the no-work-guard setup (nothing re-derived) —
        no plan_step_rederive entry must be written."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        artifacts.append_iteration_log({
            'agent': 'implementer',
            'steps_completed': ['step-1'],
            'commit': 'orphan-sha',
        })

        result = await workflow._rederive_step_status_from_branch_state()
        assert result == []

        entries, _ = artifacts.read_iteration_log()
        assert not any(e.get('event') == 'plan_step_rederive' for e in entries), (
            f'No plan_step_rederive entry should be written when nothing was '
            f're-derived; entries: {entries}'
        )


# ---------------------------------------------------------------------------
# step-9 RED: wiring into the real _inter_iteration_rebase path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInterIterationRebaseRederivesStepStatus:
    async def test_rebase_rederives_pending_step_to_done(
        self, config, git_ops, task_assignment,
    ):
        """A real inter-iteration rebase must re-derive a stale-pending step
        (already completed per the durable iteration log, with a genuine
        branch commit) to 'done' — with its recorded commit reachable from
        the post-rebase HEAD. The rebase rewrites the branch's commits onto
        the new base, so the *pre-rebase* SHA the log reported (``step_commit``
        below) is expected to become unreachable; the re-derivation must
        record the post-rebase HEAD instead of that stale SHA (see
        ``_rederive_step_status_from_branch_state``'s docstring) — otherwise
        this would silently recreate the orphaned-commit condition task
        2386's ``_reconcile_done_step_commits`` exists to repair."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'impl.py').write_text('implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        artifacts.append_iteration_log({
            'agent': 'implementer', 'steps_completed': ['step-1'], 'commit': step_commit,
        })

        # Advance main so the rebase actually runs.
        repo = config.project_root
        (repo / 'sibling.txt').write_text('sibling fix\n')
        await _run(['git', 'add', 'sibling.txt'], cwd=repo)
        await _run(['git', 'commit', '-m', 'sibling fix'], cwd=repo)

        result = await workflow._inter_iteration_rebase()

        assert result is not None, 'Rebase should have happened (main advanced).'
        plan = artifacts.read_plan()
        step_1 = plan['steps'][0]
        assert step_1['status'] == 'done', (
            'Expected step-1 re-derived to done by the wired-in re-derivation'
        )
        post_rebase_head = await workflow._get_head_commit()
        assert await git_ops.is_ancestor(step_1['commit'], post_rebase_head), (
            f"Re-derived step's recorded commit {step_1['commit']!r} must be "
            f'reachable from the post-rebase HEAD {post_rebase_head!r} — '
            'recording the rewritten pre-rebase log commit instead would be '
            'orphaned and unreachable here'
        )

    async def test_no_rebase_when_main_unchanged_leaves_step_pending(
        self, config, git_ops, task_assignment,
    ):
        """When main has not advanced, _inter_iteration_rebase returns None
        and the re-derivation path is never reached — step-1 stays pending
        even though it would be re-derivable in principle."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'impl.py').write_text('implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1')
        assert step_commit, 'Setup: expected a real commit to be made'

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        artifacts.append_iteration_log({
            'agent': 'implementer', 'steps_completed': ['step-1'], 'commit': step_commit,
        })

        result = await workflow._inter_iteration_rebase()

        assert result is None
        plan = artifacts.read_plan()
        assert plan['steps'][0]['status'] == 'pending'

    # -- task 3651: per-step provenance, not one blanket HEAD --------------

    async def test_multiple_rederived_steps_get_distinct_per_step_commits(
        self, config, git_ops, task_assignment,
    ):
        """Two steps re-derived in ONE pass must each record their OWN
        replayed commit, not both share the single post-rebase HEAD.

        The blanket-HEAD stamp satisfies the reachability invariant the test
        above pins, but it collapses N steps onto one sha — the same
        provenance loss task 3651 fixes at
        ``workflow.py::_reconcile_done_step_commits``."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        # Same file, distinct hunks -> distinct patch-ids.
        (wt / 'impl.py').write_text('v1\n')
        commit_a = await git_ops.commit(wt, 'feat: GREEN — step-1')
        (wt / 'impl.py').write_text('v2\n')
        commit_b = await git_ops.commit(wt, 'feat: GREEN — step-2')
        assert commit_a and commit_b and commit_a != commit_b

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
            {'id': 'step-2', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        artifacts.append_iteration_log({
            'agent': 'implementer', 'steps_completed': ['step-1'], 'commit': commit_a,
        })
        artifacts.append_iteration_log({
            'agent': 'implementer', 'steps_completed': ['step-2'], 'commit': commit_b,
        })

        # Advance main so the rebase actually runs.
        repo = config.project_root
        (repo / 'sibling.txt').write_text('sibling fix\n')
        await _run(['git', 'add', 'sibling.txt'], cwd=repo)
        await _run(['git', 'commit', '-m', 'sibling fix'], cwd=repo)

        result = await workflow._inter_iteration_rebase()
        assert result is not None, 'Rebase should have happened (main advanced).'

        plan = artifacts.read_plan()
        by_id = {s['id']: s for s in plan['steps']}
        step_1, step_2 = by_id['step-1'], by_id['step-2']
        post_rebase_head = await workflow._get_head_commit()

        assert step_1['status'] == 'done'
        assert step_2['status'] == 'done'

        # The anti-collapse assertion.
        assert step_1['commit'] != step_2['commit'], (
            f'both steps collapsed onto one sha {step_1["commit"]!r} '
            f'(post-rebase HEAD is {post_rebase_head!r})'
        )

        # The existing reachability invariant still holds for both.
        for step in (step_1, step_2):
            assert await git_ops.is_ancestor(step['commit'], post_rebase_head), (
                f"{step['id']}'s recorded commit {step['commit']!r} must be "
                f'reachable from the post-rebase HEAD {post_rebase_head!r}'
            )

        # Each sha must be the replay of the RIGHT original, identified by
        # subject.
        for step, expected_subject in (
            (step_1, 'feat: GREEN — step-1'),
            (step_2, 'feat: GREEN — step-2'),
        ):
            _, subject, _ = await _run(
                ['git', 'log', '-1', '--format=%s', step['commit']], cwd=wt,
            )
            assert subject.strip() == expected_subject, (
                f"{step['id']} recorded {step['commit']!r}, whose subject is "
                f'{subject.strip()!r}, not {expected_subject!r}'
            )

        # At most one of them may legitimately be the tip.
        at_head = [s['id'] for s in (step_1, step_2) if s['commit'] == post_rebase_head]
        assert len(at_head) <= 1, f'more than one step stamped HEAD: {at_head}'

    async def test_rederive_falls_back_to_head_when_no_log_commit(
        self, config, git_ops, task_assignment,
    ):
        """FALLBACK RUNG: an iteration-log entry naming a completed step but
        carrying NO ``commit`` key at all must still re-derive the step to
        'done' against the post-rebase HEAD, without raising."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'impl.py').write_text('implementation\n')
        step_commit = await git_ops.commit(wt, 'feat: GREEN — step-1')
        assert step_commit

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        artifacts.append_iteration_log({
            'agent': 'implementer', 'steps_completed': ['step-1'],
        })

        repo = config.project_root
        (repo / 'sibling.txt').write_text('sibling fix\n')
        await _run(['git', 'add', 'sibling.txt'], cwd=repo)
        await _run(['git', 'commit', '-m', 'sibling fix'], cwd=repo)

        result = await workflow._inter_iteration_rebase()
        assert result is not None

        plan = artifacts.read_plan()
        step_1 = plan['steps'][0]
        assert step_1['status'] == 'done'
        assert step_1['commit'] == await workflow._get_head_commit()

    async def test_multi_step_log_entry_is_not_trusted_as_per_step_provenance(
        self, config, git_ops, task_assignment,
    ):
        """A ledger entry naming SEVERAL steps must not lend its sha to any of
        them — the honest post-rebase HEAD is recorded for all instead.

        An entry's ``commit`` is the POST-ITERATION HEAD
        (``workflow.py::_iteration_commit_provenance``) and ``steps_completed``
        is the LIST of every step that round finished (the ledger writer's
        ``newly_completed``), so a round that completes two steps records ONE
        sha that belongs to at most the last of them. Feeding that into the
        per-step ladder would reproduce the collapse this class's sibling test
        pins — and in a strictly WORSE form: step-1 would stop carrying an
        obvious coarse marker and start carrying a specific commit that is
        really step-2's. This asserts step-1 is never handed step-2's replay.
        """
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'impl.py').write_text('v1\n')
        commit_a = await git_ops.commit(wt, 'feat: GREEN — step-1')
        (wt / 'impl.py').write_text('v2\n')
        commit_b = await git_ops.commit(wt, 'feat: GREEN — step-2')
        assert commit_a and commit_b and commit_a != commit_b

        workflow.plan = _write_plan(artifacts, workflow, [
            {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
            {'id': 'step-2', 'type': 'impl', 'status': 'pending', 'commit': None},
        ])
        # ONE entry for BOTH steps, carrying the round's post-iteration HEAD —
        # exactly the shape the real implementer ledger writer emits when an
        # iteration commits more than one step.
        artifacts.append_iteration_log({
            'agent': 'implementer',
            'steps_completed': ['step-1', 'step-2'],
            'commit': commit_b,
        })

        repo = config.project_root
        (repo / 'sibling.txt').write_text('sibling fix\n')
        await _run(['git', 'add', 'sibling.txt'], cwd=repo)
        await _run(['git', 'commit', '-m', 'sibling fix'], cwd=repo)

        result = await workflow._inter_iteration_rebase()
        assert result is not None, 'Rebase should have happened (main advanced).'

        plan = artifacts.read_plan()
        by_id = {s['id']: s for s in plan['steps']}
        step_1, step_2 = by_id['step-1'], by_id['step-2']
        post_rebase_head = await workflow._get_head_commit()

        assert step_1['status'] == 'done'
        assert step_2['status'] == 'done'

        # THE POINT: step-1 must not be attributed step-2's work. Resolving
        # commit_b through the ladder would land exactly there.
        _, step_1_subject, _ = await _run(
            ['git', 'log', '-1', '--format=%s', step_1['commit']], cwd=wt,
        )
        assert step_1_subject.strip() != 'feat: GREEN — step-2', (
            f"step-1 recorded {step_1['commit']!r}, which is step-2's replay — "
            'a multi-step ledger entry was mistaken for per-step provenance'
        )

        # Both fall to the honest coarse marker instead.
        assert step_1['commit'] == post_rebase_head
        assert step_2['commit'] == post_rebase_head

        # And the reachability invariant still holds.
        for step in (step_1, step_2):
            assert await git_ops.is_ancestor(step['commit'], post_rebase_head)
