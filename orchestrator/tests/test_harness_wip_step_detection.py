"""Tests for surfacing already-committed WIP safety-commits to the implementer (task 2051).

The harness auto-commits uncommitted work as a safety net before several
rebase/requeue/reclaim operations (see git_ops.py and workflow.py). Any of
these can land a still-"pending" plan step's complete implementation at
branch HEAD *before* mark_step_done is called for that step, so every
session has had to re-discover the workaround ad-hoc. This suite covers the
new detection + prompt-surfacing machinery that fixes that:

  - is_wip_safety_commit / WIP_SAFETY_COMMIT_PREFIXES (git_ops.py)
  - GitOps.get_commit_subjects (git_ops.py)
  - TaskWorkflow._detect_tip_wip_commits (workflow.py)
  - BriefingAssembler.build_implementer_prompt's wip_notice rendering (briefing.py)
  - _execute_iterations wiring the detector into the prompt builder (workflow.py)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.agents.invoke import AgentResult
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import WIP_SAFETY_COMMIT_PREFIXES, GitOps, _run, is_wip_safety_commit
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

# ---------------------------------------------------------------------------
# step-1 RED: is_wip_safety_commit predicate + WIP_SAFETY_COMMIT_PREFIXES
# ---------------------------------------------------------------------------


class TestIsWipSafetyCommit:
    """Pins the predicate to the exact literal subjects the harness produces.

    Producing sites (must stay recognized):
      - workflow.py:4579  'chore: save WIP before inter-iteration rebase'
      - git_ops.py:1134/2496 'chore: save WIP before requeue rebase'
      - git_ops.py:2050  'chore: save WIP before warm-lane reclaim (task 1933)'
    """

    def test_recognizes_requeue_rebase_literal(self):
        assert is_wip_safety_commit('chore: save WIP before requeue rebase') is True

    def test_recognizes_inter_iteration_rebase_literal(self):
        assert is_wip_safety_commit('chore: save WIP before inter-iteration rebase') is True

    def test_recognizes_warm_lane_reclaim_literal(self):
        assert is_wip_safety_commit(
            'chore: save WIP before warm-lane reclaim (task 1933)',
        ) is True

    def test_rejects_normal_feat_commit(self):
        assert is_wip_safety_commit('feat: GREEN — x') is False

    def test_rejects_normal_test_commit(self):
        assert is_wip_safety_commit('test: RED — y') is False

    def test_rejects_empty_string(self):
        assert is_wip_safety_commit('') is False

    def test_rejects_near_miss_subject(self):
        assert is_wip_safety_commit('chore: save the world') is False


class TestProducingSiteSubjectsStayRecognized:
    """Pins the predicate against the exact literals the harness produces.

    The TestIsWipSafetyCommit cases above pin the predicate's behavior
    against literals *copy-pasted* into this test file — they say nothing
    about whether git_ops.py/workflow.py still actually emit those literals.
    This test asserts each known producing-site literal (workflow.py's
    inter-iteration-rebase commit, and git_ops.py's requeue-rebase /
    warm-lane-reclaim commits) is still recognized by is_wip_safety_commit,
    so a change to WIP_SAFETY_COMMIT_PREFIXES that stops matching one of them
    is caught by a failing test rather than quietly disabling detection.
    """

    PRODUCING_SUBJECTS = (
        'chore: save WIP before requeue rebase',
        'chore: save WIP before inter-iteration rebase',
        'chore: save WIP before warm-lane reclaim (task 1933)',
    )

    def test_producing_subjects_are_recognized(self):
        for subject in self.PRODUCING_SUBJECTS:
            assert is_wip_safety_commit(subject) is True, (
                f'{subject!r} no longer starts with {WIP_SAFETY_COMMIT_PREFIXES!r} '
                f'— TaskWorkflow._detect_tip_wip_commits would silently stop '
                f'recognizing this producing site.'
            )


# ---------------------------------------------------------------------------
# step-3 RED: GitOps.get_commit_subjects
# ---------------------------------------------------------------------------
#
# Real temp git repo pattern, mirroring test_rebase_verify_cost.py's git_repo
# fixture — a real repo is simplest and least brittle for exercising actual
# `git log` output.


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


@pytest.mark.asyncio
class TestGetCommitSubjects:
    async def test_head_first_order_and_subjects_match(self, git_repo, git_ops):
        """Two commits on top of base: HEAD-first (sha, subject) tuples, subjects match."""
        _, base_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        base_sha = base_sha.strip()

        (git_repo / 'a.txt').write_text('a\n')
        await _run(['git', 'add', '-A'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'second commit'], cwd=git_repo)
        _, second_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        second_sha = second_sha.strip()

        (git_repo / 'b.txt').write_text('b\n')
        await _run(['git', 'add', '-A'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'third commit'], cwd=git_repo)
        _, third_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        third_sha = third_sha.strip()

        result = await git_ops.get_commit_subjects(git_repo, base_sha)

        assert result == [
            (third_sha, 'third commit'),
            (second_sha, 'second commit'),
        ], f'Expected HEAD-first pairs for third/second commit, got {result}'

    async def test_excludes_commits_at_or_before_base(self, git_repo, git_ops):
        """Commits at/before base_sha (e.g. 'Initial') must not appear in the result."""
        _, base_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        base_sha = base_sha.strip()

        (git_repo / 'a.txt').write_text('a\n')
        await _run(['git', 'add', '-A'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'only new commit'], cwd=git_repo)

        result = await git_ops.get_commit_subjects(git_repo, base_sha)

        subjects = [subject for _sha, subject in result]
        assert subjects == ['only new commit']
        assert 'Initial' not in subjects

    async def test_empty_range_returns_empty_list(self, git_repo, git_ops):
        """base_sha == HEAD (no commits since base) must return []."""
        _, head_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=git_repo)
        head_sha = head_sha.strip()

        result = await git_ops.get_commit_subjects(git_repo, head_sha)

        assert result == []


# ---------------------------------------------------------------------------
# step-5 RED: TaskWorkflow._detect_tip_wip_commits
# ---------------------------------------------------------------------------
#
# Real-git _make_workflow pattern, mirroring test_rebase_verify_cost.py /
# test_verify_phase_rebase.py — real GitOps + a real worktree created via
# git_ops.create_worktree, heavy collaborators (scheduler/briefing/mcp) mocked.


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

    Mirrors the pattern in test_rebase_verify_cost.py._make_workflow.
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


@pytest.mark.asyncio
class TestDetectTipWipCommits:
    async def test_wip_commit_at_head_is_detected(self, config, git_ops, task_assignment):
        """A real WIP safety-commit sitting at HEAD is surfaced."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'wip.txt').write_text('uncommitted work\n')
        wip_sha = await git_ops.commit(wt, 'chore: save WIP before requeue rebase')
        assert wip_sha, 'Setup: expected a real commit to be made'

        result = await workflow._detect_tip_wip_commits()

        assert result == [
            {'sha': wip_sha, 'subject': 'chore: save WIP before requeue rebase'},
        ]

    async def test_normal_commit_at_head_returns_empty(self, config, git_ops, task_assignment):
        """A normal (non-WIP) commit at HEAD must not be surfaced."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'real.txt').write_text('real work\n')
        await git_ops.commit(wt, 'feat: GREEN — real implementation')

        result = await workflow._detect_tip_wip_commits()

        assert result == []

    async def test_wip_commit_already_recorded_on_done_step_is_deduped(
        self, config, git_ops, task_assignment,
    ):
        """A WIP commit already attributed to a done plan step is not re-surfaced."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'wip.txt').write_text('uncommitted work\n')
        wip_sha = await git_ops.commit(wt, 'chore: save WIP before requeue rebase')
        assert wip_sha

        workflow.plan = {
            'task_id': '42',
            'prerequisites': [],
            'steps': [{'id': 'step-1', 'status': 'done', 'commit': wip_sha}],
        }

        result = await workflow._detect_tip_wip_commits()

        assert result == []

    async def test_wip_commit_recorded_with_abbreviated_sha_is_deduped(
        self, config, git_ops, task_assignment,
    ):
        """A done step recorded with the 12-char SHA shown in the prompt notice
        (rather than the full SHA `git log` produces) must still dedup.

        build_implementer_prompt's wip_section displays `sha[:12]` and asks
        the implementer to pass "the WIP commit's SHA" to mark_step_done — an
        implementer may reasonably copy exactly what it saw. Exact-string
        dedup would never match the full SHA against that abbreviated form,
        re-surfacing the same already-attributed commit forever.
        """
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'wip.txt').write_text('uncommitted work\n')
        wip_sha = await git_ops.commit(wt, 'chore: save WIP before requeue rebase')
        assert wip_sha

        workflow.plan = {
            'task_id': '42',
            'prerequisites': [],
            'steps': [{'id': 'step-1', 'status': 'done', 'commit': wip_sha[:12]}],
        }

        result = await workflow._detect_tip_wip_commits()

        assert result == []

    async def test_wip_commit_not_at_tip_returns_empty(self, config, git_ops, task_assignment):
        """A normal commit on top of a WIP commit means the WIP is no longer at HEAD."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'wip.txt').write_text('uncommitted work\n')
        await git_ops.commit(wt, 'chore: save WIP before requeue rebase')
        (wt / 'real.txt').write_text('real work\n')
        await git_ops.commit(wt, 'feat: GREEN — real implementation')

        result = await workflow._detect_tip_wip_commits()

        assert result == []

    async def test_multiple_stacked_wip_commits_returned_head_first(
        self, config, git_ops, task_assignment,
    ):
        """A contiguous run of 2+ WIP safety-commits at HEAD are all surfaced.

        Exercises the walk-and-stop loop with a non-trivial "collect the
        rest" (2 commits) before hitting the boundary, distinct from the
        single-commit case covered by test_wip_commit_at_head_is_detected.
        """
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'real.txt').write_text('real work\n')
        await git_ops.commit(wt, 'feat: GREEN — real implementation')

        (wt / 'wip1.txt').write_text('uncommitted work 1\n')
        wip1_sha = await git_ops.commit(wt, 'chore: save WIP before requeue rebase')
        (wt / 'wip2.txt').write_text('uncommitted work 2\n')
        wip2_sha = await git_ops.commit(
            wt, 'chore: save WIP before inter-iteration rebase',
        )
        assert wip1_sha and wip2_sha

        result = await workflow._detect_tip_wip_commits()

        assert result == [
            {'sha': wip2_sha, 'subject': 'chore: save WIP before inter-iteration rebase'},
            {'sha': wip1_sha, 'subject': 'chore: save WIP before requeue rebase'},
        ]

    async def test_get_commit_subjects_error_returns_empty(
        self, config, git_ops, task_assignment,
    ):
        """A git failure inside get_commit_subjects must not propagate.

        Exercises the defensive `except Exception: return []` wrapper, which
        was previously untested.
        """
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'wip.txt').write_text('uncommitted work\n')
        await git_ops.commit(wt, 'chore: save WIP before requeue rebase')

        workflow.git_ops.get_commit_subjects = AsyncMock(  # type: ignore[method-assign]
            side_effect=RuntimeError('boom'),
        )

        result = await workflow._detect_tip_wip_commits()

        assert result == []

    async def test_worktree_none_returns_empty(self, config, git_ops, task_assignment):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt_info.path)
        artifacts.update_base_commit(wt_info.base_commit)
        workflow.worktree = None

        result = await workflow._detect_tip_wip_commits()

        assert result == []

    async def test_git_ops_none_returns_empty(self, config, git_ops, task_assignment):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt_info.path)
        artifacts.update_base_commit(wt_info.base_commit)
        workflow.git_ops = None  # type: ignore[assignment]

        result = await workflow._detect_tip_wip_commits()

        assert result == []

    async def test_base_commit_unset_returns_empty(self, config, git_ops, task_assignment):
        """artifacts.read_base_commit() falsy (no base_commit stamped) → []."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow = TaskWorkflow(
            assignment=task_assignment,
            config=config,
            git_ops=git_ops,
            scheduler=MagicMock(),  # type: ignore[arg-type]
            briefing=MagicMock(),  # type: ignore[arg-type]
            mcp=MagicMock(),  # type: ignore[arg-type]
        )
        workflow.worktree = wt
        artifacts = TaskArtifacts(wt)
        artifacts.init('42', 'X', 'desc')  # no base_commit
        workflow.artifacts = artifacts
        workflow.plan = {'task_id': '42', 'steps': [], 'prerequisites': []}

        result = await workflow._detect_tip_wip_commits()

        assert result == []


# ---------------------------------------------------------------------------
# step-7 RED: BriefingAssembler.build_implementer_prompt wip_notice rendering
# ---------------------------------------------------------------------------


@pytest.fixture
def briefing(tmp_path: Path) -> BriefingAssembler:
    config = OrchestratorConfig(
        project_root=tmp_path,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )
    return BriefingAssembler(config)


@pytest.fixture
def minimal_plan() -> dict:
    return {
        'task_id': '2051',
        'title': 'T',
        'analysis': 'A',
        'prerequisites': [],
        'steps': [{'id': 'step-1', 'status': 'pending'}],
    }


@pytest.mark.asyncio
class TestBuildImplementerPromptWipNotice:
    async def test_wip_notice_renders_header_sha_and_mark_step_done(
        self, briefing: BriefingAssembler, minimal_plan: dict,
    ):
        wip_notice = [
            {'sha': 'abcdef1234567890', 'subject': 'chore: save WIP before requeue rebase'},
        ]

        prompt = await briefing.build_implementer_prompt(
            minimal_plan, [], context='', wip_notice=wip_notice,
        )

        assert 'Verify Before Re-Implementing' in prompt
        assert 'abcdef123456' in prompt
        assert 'mark_step_done' in prompt

    async def test_wip_notice_none_omits_section(
        self, briefing: BriefingAssembler, minimal_plan: dict,
    ):
        prompt = await briefing.build_implementer_prompt(
            minimal_plan, [], context='', wip_notice=None,
        )

        assert 'Verify Before Re-Implementing' not in prompt

    async def test_wip_notice_empty_list_omits_section(
        self, briefing: BriefingAssembler, minimal_plan: dict,
    ):
        prompt = await briefing.build_implementer_prompt(
            minimal_plan, [], context='', wip_notice=[],
        )

        assert 'Verify Before Re-Implementing' not in prompt

    async def test_rebase_notice_still_renders_independently_of_wip_notice(
        self, briefing: BriefingAssembler, minimal_plan: dict,
    ):
        rebase_notice = {
            'old_base': 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
            'new_base': 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
            'changed_files': ['foo.py'],
        }

        prompt = await briefing.build_implementer_prompt(
            minimal_plan, [], context='', rebase_notice=rebase_notice, wip_notice=None,
        )

        assert 'Rebase Notice' in prompt
        assert 'Verify Before Re-Implementing' not in prompt


# ---------------------------------------------------------------------------
# task-2279 RED: mandatory git status/diff pre-flight always renders,
# independent of wip_notice — covers the broader "uncommitted, never
# committed at all" crash mode that 2051's WIP-safety-commit auto-detect
# (wip_section above) does not catch.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBuildImplementerPromptMandatoryPreflight:
    async def test_preflight_renders_with_wip_notice_none(
        self, briefing: BriefingAssembler, minimal_plan: dict,
    ):
        """Proves the pre-flight is not accidentally coupled to wip_section:
        it must render even when there is no WIP notice at all, which is the
        common case for most implementer invocations."""
        prompt = await briefing.build_implementer_prompt(
            minimal_plan, [], context='', wip_notice=None,
        )

        assert 'git diff HEAD' in prompt
        assert 'mark_step_done' in prompt

    async def test_preflight_still_renders_alongside_wip_notice(
        self, briefing: BriefingAssembler, minimal_plan: dict,
    ):
        wip_notice = [
            {'sha': 'abcdef1234567890', 'subject': 'chore: save WIP before requeue rebase'},
        ]

        prompt = await briefing.build_implementer_prompt(
            minimal_plan, [], context='', wip_notice=wip_notice,
        )

        assert 'git diff HEAD' in prompt
        assert 'Mandatory pre-flight' in prompt


# ---------------------------------------------------------------------------
# step-9 RED: _execute_iterations forwards the detected wip_notice
# ---------------------------------------------------------------------------
#
# Reproduces the cross-dispatch requeue-WIP-at-HEAD scenario: a real WIP
# safety-commit is already at HEAD when the workflow's FIRST iteration runs
# (execute_iterations == 0, so no inter-iteration rebase fires) — this is the
# path that requeue/warm-lane-reclaim commits (not inter-iteration rebase)
# would produce.


@pytest.mark.asyncio
class TestExecuteIterationsForwardsWipNotice:
    async def test_wip_notice_forwarded_to_build_implementer_prompt(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
        artifacts.update_base_commit(wt_info.base_commit)

        (wt / 'wip.txt').write_text('uncommitted work\n')
        wip_sha = await git_ops.commit(wt, 'chore: save WIP before requeue rebase')
        assert wip_sha, 'Setup: expected a real WIP commit at HEAD'

        plan = {
            'task_id': '42',
            'title': 'X',
            'analysis': 'A',
            'prerequisites': [],
            'steps': [{'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None}],
        }
        artifacts.write_plan(plan)
        artifacts.stamp_plan_provenance(workflow.session_id)
        workflow.plan = artifacts.read_plan()

        def _mark_step_done_side_effect(*args, **kwargs):
            artifacts.update_step_status('step-1', 'done', 'impl-commit-sha')
            return AgentResult(success=True, output='')

        workflow.briefing.build_implementer_prompt = AsyncMock(return_value='impl')
        workflow._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
        workflow._get_head_commit = AsyncMock(return_value='head-sha')  # type: ignore[method-assign]
        workflow._invoke = AsyncMock(  # type: ignore[method-assign]
            side_effect=_mark_step_done_side_effect,
        )

        outcome = await workflow._execute_iterations()

        assert outcome == WorkflowOutcome.DONE
        assert workflow.briefing.build_implementer_prompt.await_count == 1
        call_kwargs = workflow.briefing.build_implementer_prompt.call_args.kwargs
        assert call_kwargs.get('wip_notice'), (
            f'Expected non-empty wip_notice kwarg, got {call_kwargs}'
        )
        surfaced_shas = {n['sha'] for n in call_kwargs['wip_notice']}
        assert wip_sha in surfaced_shas, (
            f'Expected WIP commit {wip_sha} in surfaced wip_notice, got {surfaced_shas}'
        )
