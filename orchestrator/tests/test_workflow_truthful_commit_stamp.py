"""Truthful iteration-ledger commit stamping (task 2759).

All three orchestrator iteration-ledger writers (implementer in
``_execute_iterations``, debugger in ``_verify_debugfix_loop``, amender in
``_amend``) historically stamped ``commit: await self._get_head_commit()``
unconditionally — with no pre/post HEAD comparison. When an agent session
ended before committing (implementer died mid-background-wait, amendment left
uncommitted), HEAD was unchanged and the ledger recorded the round "at" an
unrelated pre-existing commit. Recovery guards then consumed that false
provenance (reify 5164 RCA).

The fix factors a single shared helper
``TaskWorkflow._iteration_commit_provenance(pre_head)`` that every writer
merges into its entry dict: HEAD advanced ⇒ ``{commit: <new sha>,
committed: True}``; HEAD unchanged ⇒ ``{commit: None, committed: False,
dirty: <porcelain-nonempty>}``. Two recovery guards
(``_iteration_entry_is_work`` and the ``_rederive_step_status_from_branch_state``
union) then treat ``committed is False`` as "no durable work this entry",
discriminating on the explicit ``False`` flag only so pre-2759 legacy entries
(no ``committed`` key) keep today's classification.

This file holds everything — the small real-git harness is replicated here
(mirroring test_harness_plan_step_rederive.py / test_workflow_verify_retry.py)
rather than editing any shared suite, keeping concurrency locks to workflow.py
plus this one new file.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.agents.invoke import AgentResult
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import (
    TaskWorkflow,
    WorkflowOutcome,
    _iteration_entry_is_work,
)

# ---------------------------------------------------------------------------
# Real-git harness — mirrors test_harness_plan_step_rederive.py /
# test_workflow_verify_retry.py (real temp git repo + real worktree via
# git_ops.create_worktree; heavy collaborators mocked).
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
    # Mirror the real repo: .task/ execution metadata (written by
    # TaskArtifacts.init here, since these harnesses construct in-worktree
    # artifacts) is gitignored, so it never surfaces in `git status
    # --porcelain` — exactly the invariant GitOps.has_uncommitted_work
    # (and thus _iteration_commit_provenance's dirty read) relies on.
    (repo / '.gitignore').write_text('.task/\n')
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

    Mirrors test_harness_plan_step_rederive.py._make_workflow.
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
    """Persist a plan with the given step dicts and stamp provenance so
    ``validate_plan_owner(workflow.session_id)`` passes, returning the re-read
    plan. Mirrors test_harness_plan_step_rederive.py._write_plan.
    """
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
# step-1 RED: the _iteration_commit_provenance helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestIterationCommitProvenanceHelper:
    """The shared helper compares pre/post HEAD and reports truthful
    provenance: advanced ⇒ {commit, committed:True} (no dirty read);
    unchanged ⇒ {commit:None, committed:False, dirty:<porcelain-nonempty>}.
    """

    async def test_head_advanced_records_new_commit_and_no_dirty(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, _artifacts = _make_workflow(config, git_ops, task_assignment, wt)

        pre_head = await workflow._get_head_commit()
        (wt / 'impl.py').write_text('implementation\n')
        new_commit = await git_ops.commit(wt, 'feat: real commit')
        assert new_commit, 'Setup: expected a real commit to be made'

        result = await workflow._iteration_commit_provenance(pre_head)

        post_head = await workflow._get_head_commit()
        assert result['commit'] == post_head
        assert result['commit'] != pre_head
        assert result['committed'] is True
        assert 'dirty' not in result, (
            'the committed (happy) path must not perform the extra dirty read'
        )

    async def test_head_unchanged_clean_tree_records_null_commit(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, _artifacts = _make_workflow(config, git_ops, task_assignment, wt)

        pre_head = await workflow._get_head_commit()  # == current HEAD, clean tree

        result = await workflow._iteration_commit_provenance(pre_head)

        assert result['commit'] is None
        assert result['committed'] is False
        assert result['dirty'] is False

    async def test_head_unchanged_dirty_tree_records_null_commit_dirty_true(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, _artifacts = _make_workflow(config, git_ops, task_assignment, wt)

        pre_head = await workflow._get_head_commit()
        # Uncommitted change: HEAD stays put but the tree is dirty.
        (wt / 'uncommitted.py').write_text('scratch\n')

        result = await workflow._iteration_commit_provenance(pre_head)

        assert result['commit'] is None
        assert result['committed'] is False
        assert result['dirty'] is True


# ---------------------------------------------------------------------------
# step-3 RED: the amend writer (_amend) — the RCA origin
# ---------------------------------------------------------------------------


def _amend_workflow(config, git_ops, task_assignment, wt):
    """A real-git workflow wired to drive _amend once: a plan on disk with
    stamped provenance (so validate_plan_owner passes) and a mocked amender
    briefing. The caller configures _invoke."""
    workflow, artifacts = _make_workflow(config, git_ops, task_assignment, wt)
    _write_plan(artifacts, workflow, [
        {'id': 'step-1', 'type': 'impl', 'status': 'done', 'commit': 'x'},
    ])
    workflow.plan = artifacts.read_plan()
    workflow.briefing.build_amender_prompt = AsyncMock(  # type: ignore[attr-defined]
        return_value='amend-prompt',
    )
    return workflow, artifacts


@pytest.mark.asyncio
class TestAmendWriterTruthfulCommit:
    """_amend must stamp truthful provenance: commit:null / committed:false
    when the amender left HEAD unchanged; the real new sha / committed:true
    when it committed."""

    async def test_amend_no_commit_records_null_and_uncommitted(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _amend_workflow(config, git_ops, task_assignment, wt)
        # No-op amender: returns success but makes no commit ⇒ HEAD unchanged.
        workflow._invoke = AsyncMock(  # type: ignore[method-assign]
            return_value=AgentResult(success=True, output=''),
        )

        in_scope = [{'file': 'lib.py', 'suggestion': 'tidy'}]
        ok = await workflow._amend(in_scope, amendment_round=1)
        assert ok is True

        entries, _ = artifacts.read_iteration_log()
        amendments = [e for e in entries if e.get('source') == 'amendment']
        assert len(amendments) == 1, f'expected one amendment entry, got {entries}'
        entry = amendments[0]
        assert entry['commit'] is None, (
            f'amender made no commit ⇒ commit must be null, got {entry.get("commit")!r}'
        )
        assert entry['committed'] is False

    async def test_amend_with_commit_records_new_head_and_committed(
        self, config, git_ops, task_assignment,
    ):
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, artifacts = _amend_workflow(config, git_ops, task_assignment, wt)

        async def _commit_effect(*_args, **_kwargs):
            (wt / 'amend.py').write_text('amended\n')
            await git_ops.commit(wt, 'feat: amendment commit')
            return AgentResult(success=True, output='')

        workflow._invoke = AsyncMock(side_effect=_commit_effect)  # type: ignore[method-assign]

        in_scope = [{'file': 'lib.py', 'suggestion': 'tidy'}]
        ok = await workflow._amend(in_scope, amendment_round=1)
        assert ok is True

        head = await workflow._get_head_commit()
        entries, _ = artifacts.read_iteration_log()
        amendments = [e for e in entries if e.get('source') == 'amendment']
        assert len(amendments) == 1, f'expected one amendment entry, got {entries}'
        entry = amendments[0]
        assert entry['commit'] == head, (
            f'amender committed ⇒ commit must be the new HEAD {head!r}, '
            f'got {entry.get("commit")!r}'
        )
        assert entry['committed'] is True


# ---------------------------------------------------------------------------
# step-5 RED: the implementer writer (_execute_iterations)
#
# Mocked-collaborator loop harness (mirrors
# test_workflow_resume_on_progress.py) but with a REAL TaskArtifacts so the
# implementer entry is genuinely written to iterations.jsonl (append/read are
# NOT mocked). git_ops/config are MagicMocks; _get_head_commit is pinned to a
# constant so pre==post (no HEAD advance) and has_uncommitted_work is forced
# True so the no-commit entry records dirty:True.
# ---------------------------------------------------------------------------


def _make_mock_workflow(*, tmp_path: Path, max_execute_iterations: int = 1) -> TaskWorkflow:
    """Resume-style MagicMock harness (mirrors
    test_workflow_resume_on_progress.py._make_workflow) with a REAL
    TaskArtifacts on a temp worktree."""
    assignment = MagicMock()
    assignment.task_id = '2759'
    assignment.task = {'id': '2759', 'title': 'T', 'description': 'd'}
    assignment.modules = []

    _spec = pydantic_spec(OrchestratorConfig)
    cfg = MagicMock(spec_set=_spec)
    cfg.fused_memory.project_id = 'dark_factory'
    cfg.fused_memory.url = 'http://localhost:8002'
    cfg.max_review_cycles = 2
    cfg.max_amendment_rounds = 1
    cfg.lock_depth = 2
    cfg.steward_completion_timeout = 300.0
    cfg.project_root = tmp_path / 'proj'
    cfg.merge_train_former_enabled = False
    cfg.merge_train_max_members = 3
    cfg.max_execute_iterations = max_execute_iterations
    cfg.max_consecutive_zero_output_timeouts = 2
    cfg.max_progress_resume_iterations = 20
    cfg.recycle_config_dir_on_zero_output = False
    cfg.judge_after_each_iteration = False
    cfg.inter_iteration_rebase = False

    wf = TaskWorkflow(
        assignment=assignment,
        config=cfg,
        git_ops=MagicMock(),
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    (worktree / '.task').mkdir(exist_ok=True)
    wf.artifacts = TaskArtifacts(worktree)
    wf.worktree = worktree
    wf.merge_queue = MagicMock()
    wf.plan = {
        'files': [], 'prerequisites': [],
        'steps': [{'id': 'step-1', 'status': 'pending'}],
    }
    wf._base_commit = 'base_sha'
    wf._module_configs = []
    return wf


def _stub_impl_loop(wf: TaskWorkflow, *, head_sha: str, dirty: bool) -> AsyncMock:
    """Stub the loop-driving artifacts/collaborators so _execute_iterations
    runs one no-op implementer iteration, WITHOUT mocking append_iteration_log
    / read_iteration_log (the entry must be genuinely written and read back).
    Returns the mocked _invoke."""
    wf.artifacts.get_pending_steps = MagicMock(  # type: ignore[method-assign]
        return_value=[{'id': 'step-1'}],
    )
    wf.artifacts.validate_plan_owner = MagicMock(return_value=True)  # type: ignore[method-assign]
    wf.artifacts.read_plan = MagicMock(  # type: ignore[method-assign]
        return_value={
            'prerequisites': [],
            'steps': [{'id': 'step-1', 'status': 'pending'}],
        },
    )
    wf.artifacts.stamp_plan_provenance = MagicMock()  # type: ignore[method-assign]

    mock_invoke = AsyncMock(  # no-op implementer: success, makes no commit
        return_value=AgentResult(
            success=True, output='ok', timed_out=False, turns=2, cost_usd=0.0,
        ),
    )
    wf._invoke = mock_invoke  # type: ignore[method-assign]
    wf._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
    wf._get_head_commit = AsyncMock(return_value=head_sha)  # type: ignore[method-assign]
    wf.git_ops.has_uncommitted_work = AsyncMock(return_value=dirty)  # type: ignore[attr-defined]
    wf.briefing.build_implementer_prompt = AsyncMock(return_value='impl-prompt')  # type: ignore[attr-defined]
    return mock_invoke


@pytest.mark.asyncio
class TestImplementerWriterTruthfulCommit:
    async def test_implementer_no_advance_records_null_committed_false_dirty(
        self, tmp_path: Path,
    ):
        """A no-op implementer iteration (HEAD unchanged, dirty tree) must
        record commit:None / committed:False / dirty:True — not the stale
        constant HEAD sha. Fails RED: the writer stamps the constant sha with
        no committed key."""
        wf = _make_mock_workflow(tmp_path=tmp_path, max_execute_iterations=1)
        _stub_impl_loop(wf, head_sha='const_sha', dirty=True)

        outcome = await wf._execute_iterations()
        assert outcome == WorkflowOutcome.BLOCKED  # cap fires after one write

        entries, _ = wf.artifacts.read_iteration_log()
        impl = [
            e for e in entries
            if e.get('agent') == 'implementer' and e.get('source') == 'orchestrator'
        ]
        assert len(impl) == 1, f'expected exactly one implementer entry, got {entries}'
        entry = impl[0]
        assert entry['commit'] is None, (
            f'no HEAD advance ⇒ commit must be null, got {entry.get("commit")!r}'
        )
        assert entry['committed'] is False
        assert entry['dirty'] is True


# ---------------------------------------------------------------------------
# step-7 RED: the debugger writer (_verify_debugfix_loop)
# ---------------------------------------------------------------------------


def _failing_verify_result() -> VerifyResult:
    return VerifyResult(
        passed=False,
        test_output='FAILED tests/test_x.py::test_y\n',
        lint_output='',
        type_output='',
        summary='Failures: tests failed',
        timed_out=False,
        cause_hint='FAILED tests/test_x.py::test_y',
        category='test_failure',
    )


def _passing_verify_result() -> VerifyResult:
    return VerifyResult(
        passed=True,
        test_output='',
        lint_output='',
        type_output='',
        summary='ok',
        timed_out=False,
    )


async def _make_verify_workflow(
    git_repo: Path, task_assignment: TaskAssignment,
) -> tuple[TaskWorkflow, TaskArtifacts, GitOps]:
    """Real-git workflow wired to drive _verify_debugfix_loop through exactly
    one debugger invocation: verify off-flags so the loop reaches the debugger,
    a stubbed verify (fail → pass), and a no-op debugger _invoke."""
    cfg = OrchestratorConfig(
        project_root=git_repo,
        rebase_before_verify=False,
        escalate_preexisting_main_break=False,
        max_verify_attempts=2,
        # Well above the one debugger iteration so the signature-repeat guard
        # never fires — this test is solely about the debugger writer.
        max_failure_signature_repeat=100,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees',
        ),
    )
    gops = GitOps(cfg.git, cfg.project_root)
    wt_info = await gops.create_worktree(task_assignment.task_id)
    wt = wt_info.path
    workflow, artifacts = _make_workflow(cfg, gops, task_assignment, wt)

    workflow._run_scoped_verification_with_infra_retry = AsyncMock(  # type: ignore[method-assign]
        side_effect=[_failing_verify_result(), _passing_verify_result()],
    )
    workflow._maybe_file_chronic_flakes = AsyncMock()  # type: ignore[method-assign]
    workflow._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
    workflow.briefing.build_debugger_prompt = AsyncMock(return_value='debug-prompt')  # type: ignore[attr-defined]
    workflow._invoke = AsyncMock(  # no-op debugger: success, makes no commit
        return_value=AgentResult(success=True, output=''),
    )  # type: ignore[method-assign]
    workflow._get_head_commit = AsyncMock(return_value='const_sha')  # type: ignore[method-assign]
    gops.has_uncommitted_work = AsyncMock(return_value=True)  # type: ignore[method-assign]
    return workflow, artifacts, gops


@pytest.mark.asyncio
class TestDebuggerWriterTruthfulCommit:
    async def test_debugger_no_advance_records_null_and_uncommitted(
        self, git_repo, task_assignment,
    ):
        """A no-op debugger pass (HEAD unchanged) must record
        commit:None / committed:False — not the stale constant HEAD. Fails
        RED: the writer stamps the constant sha with no committed key."""
        workflow, artifacts, _gops = await _make_verify_workflow(
            git_repo, task_assignment,
        )

        outcome = await workflow._verify_debugfix_loop()
        assert outcome == WorkflowOutcome.DONE  # fail → debug → pass

        entries, _ = artifacts.read_iteration_log()
        debug = [e for e in entries if e.get('agent') == 'debugger']
        assert len(debug) == 1, f'expected exactly one debugger entry, got {entries}'
        entry = debug[0]
        assert entry['commit'] is None, (
            f'no HEAD advance ⇒ commit must be null, got {entry.get("commit")!r}'
        )
        assert entry['committed'] is False


# ---------------------------------------------------------------------------
# step-9 RED: _iteration_entry_is_work treats committed:False as no work
# ---------------------------------------------------------------------------


class TestIterationEntryIsWorkCommittedGuard:
    """A committed:False entry recorded no durable commit (task 2759) and must
    not count as prior-implementation work, regardless of agent type. The guard
    keys strictly on the explicit False flag, so legacy entries lacking the key
    keep today's classification (backward compatible with all live logs)."""

    def test_committed_false_entries_are_never_work(self):
        # All of these classify as work TODAY (RED) but must be False once the
        # explicit-False guard lands.
        assert _iteration_entry_is_work({
            'agent': 'implementer', 'steps_completed': ['step-1'],
            'committed': False,
        }) is False
        assert _iteration_entry_is_work({
            'agent': 'debugger', 'steps_completed': [], 'committed': False,
        }) is False
        assert _iteration_entry_is_work({
            'agent': 'implementer', 'source': 'amendment', 'committed': False,
        }) is False

    def test_legacy_entries_without_committed_key_unchanged(self):
        # No 'committed' key ⇒ entry.get('committed') is None (≠ False) ⇒ the
        # guard falls through to today's classification.
        assert _iteration_entry_is_work({
            'agent': 'implementer', 'source': 'amendment',
        }) is True
        assert _iteration_entry_is_work({
            'agent': 'debugger', 'steps_completed': [],
        }) is True
        assert _iteration_entry_is_work({
            'agent': 'implementer', 'steps_completed': ['step-1'],
        }) is True
        assert _iteration_entry_is_work({
            'agent': 'implementer', 'steps_completed': [],
        }) is False

    def test_committed_true_entries_unchanged(self):
        assert _iteration_entry_is_work({
            'agent': 'implementer', 'steps_completed': ['step-1'],
            'committed': True,
        }) is True
        assert _iteration_entry_is_work({
            'agent': 'debugger', 'steps_completed': [], 'committed': True,
        }) is True
