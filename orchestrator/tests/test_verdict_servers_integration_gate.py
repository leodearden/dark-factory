"""θ — Verdict-servers integration gate (artifact-read boundary tests, task 2488).

TERMINAL integration gate for the mcp-verdict-servers PRD
(``plans/mcp-verdict-servers-prd.md`` §Boundary-test sketch). All five
dependencies are landed on main: the verdict-tools server +
``TaskArtifacts.write_verdict``/``read_verdict``/``clear_verdict`` (α), the
orchestrator's ``_inject_verdict_tools_mcp`` wiring (β), and the four role
migrations — merger (γ), reviewer (δ), triage (ε), judge (ζ→η). So θ adds NO
production code: every test here is a GREEN-on-write regression/integration
guard over already-landed behavior, not RED→GREEN feature TDD.

SEAM — the reason this module exists as more than a copy of the per-role
tests: production ``TaskWorkflow._invoke`` (workflow.py) calls
``invoke_with_cap_retry(invoke_fn=invoke_agent)``, and the steward's triage
path (``TaskSteward._pre_triage_suggestions``, steward.py) calls the same
with its own module-level ``invoke_agent``. The per-role tests
(``test_merger_disposition_verdict.py`` / ``test_reviewer_verdict_routing.py``
/ ``test_judge_completion_verdict.py``) mock ``TaskWorkflow._invoke`` WHOLESALE,
so they never exercise ``_inject_verdict_tools_mcp``, route resolution, or the
real clear→spawn→read sequencing. θ instead injects one level deeper —
``monkeypatch.setattr('orchestrator.workflow.invoke_agent', fake)`` (and
``'orchestrator.steward.invoke_agent'`` for triage) — the established pattern
from ``test_workflow_e2e.py`` / ``test_config_reload_integration_gate.py`` — so
the REAL ``_invoke`` / ``_pre_triage_suggestions`` runs and only the agent
subprocess is faked. The θ-local fakes emit their verdict by writing
``verdicts/<role>.json`` via ``verdict_tools._envelope``/``_submit_*``, exactly
as the real MCP tool server would.

Two scope corrections from premise verification (see plan.json
``design_decisions``):

1. The task's literal scenario 8 ("judge transition fallback via
   structured_output still gates") is DEAD post-η: task 2487 removed the ζ
   structured_output fallback, so ``_run_completion_judge`` reads
   ``verdicts/judge.json`` ONLY. ``TestJudgeStructuredOutputIgnoredPostEta``
   reframes it as the post-η regression guard: a judge returning legacy
   ``structured_output`` but no tool artifact ⇒ ``None`` (structured_output is
   actively ignored, not "still gating").
2. Scenario 14 (I-SHARED-INTACT) already has a full standing guard in
   ``test_shared_output_schema_intact.py`` (η/task 2487), whose third test is
   literally labeled "boundary test #14". ``TestSharedMachineryIntact`` here is
   a COMPACT re-assertion for this gate's own self-completeness — it does not
   duplicate that file's ``_drive_recon_like`` helper. See that module for the
   canonical, full-coverage guard.

Scenario → test class map:
    1-3   TestMergerBoundary                    (plan step S1)
    4-5   TestReviewerBoundary                  (plan step S2)
    6,7,9 TestJudgeBoundary                     (plan step S3)
    8     TestJudgeStructuredOutputIgnoredPostEta (plan step S4, reframed)
    10-11 TestTriageBoundary                    (plan step S5)
    12-13 TestFreshnessGuards                   (plan step S6, I-FRESH)
    14    TestSharedMachineryIntact             (plan step S7, I-SHARED-INTACT)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _workflow_helpers import (
    AgentStub,
    FakeBriefing,
    FakeMcp,
    _build_workflow,
    _init_repo,
)
from escalation.models import Escalation
from shared.cli_invoke import (
    _REAL_BUILTIN_TOOLS_DENYLIST,
    _SCHEMA_OUTPUT_TOOL,
    AgentResult,
    _SubprocessResult,
    invoke_claude_agent,
)

from orchestrator.agents.roles import REVIEWER_COMPREHENSIVE
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.mcp.verdict_tools import _envelope, _submit_review_verdict, _submit_triage
from orchestrator.scheduler import TaskAssignment
from orchestrator.steward import TaskSteward
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

# ---------------------------------------------------------------------------
# Fixtures — file-local, mirroring test_workflow_e2e.py:147-191. A REAL
# OrchestratorConfig is required (not _workflow_helpers._make's MagicMock)
# because the real _invoke calls resolve_route(route_inputs, self.config),
# and the real _pre_triage_suggestions calls resolve_and_record_route(...,
# config=self.config, ...) — both do real attribute/membership reads against
# config.routing.* that a MagicMock can't satisfy.
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A bare-minimum git repo with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


@pytest.fixture
def config(git_repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        max_execute_iterations=5,
        max_verify_attempts=3,
        max_review_cycles=2,
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
            'id': '42',
            'title': 'Verdict boundary task',
            'description': 'Exercise the real _invoke verdict-artifact path',
            'status': 'pending',
            'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


# ---------------------------------------------------------------------------
# Shared rig helpers
# ---------------------------------------------------------------------------


def _artifacts_for(worktree: Path) -> TaskArtifacts:
    """Real on-disk TaskArtifacts at the derived .task-meta sibling root for
    *worktree* — the SAME root ``_invoke``'s ``_inject_verdict_tools_mcp``
    derives (via ``_meta_root_for_worktree``) and ``TaskWorkflow.artifacts`` /
    ``TaskSteward``'s own internal ``TaskArtifacts`` read from. Mirrors
    ``AgentStub``'s explicit-meta_root construction in ``_workflow_helpers.py``
    (rather than relying on the autouse ``_derive_meta_root_like_production``
    shim, which exists for legacy bare-``TaskArtifacts(cwd)`` call sites) so
    this module is self-contained.
    """
    return TaskArtifacts(worktree, TaskArtifacts.meta_root_for(worktree.parent, worktree.name))


def _seed_workflow_artifacts(
    workflow: TaskWorkflow, *, tmp_path: Path, task_id: str = '42',
) -> TaskArtifacts:
    """Wire a real on-disk TaskArtifacts + worktree directory onto *workflow*.

    ``_build_workflow`` only constructs the TaskWorkflow — it does not create
    a worktree or set ``.worktree``/``.artifacts`` (those are normally set by
    the full ``run()`` cycle's ``_setup``). θ drives ``_resolve_and_resubmit``
    / ``_run_reviewer`` / ``_run_completion_judge`` directly, bypassing
    ``run()``, so this mirrors ``_workflow_helpers._make``'s worktree/artifacts
    wiring — but with a REAL config/git_ops (see module fixtures above),
    which ``_make``'s MagicMock config cannot support.
    """
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    artifacts = _artifacts_for(worktree)
    artifacts.init(task_id, 'Verdict boundary task', 'desc', base_commit='oldbase')
    workflow.artifacts = artifacts
    workflow.worktree = worktree
    return artifacts


def _setup_merger_workflow(
    config: OrchestratorConfig, git_ops: GitOps, task_assignment: TaskAssignment, tmp_path: Path,
) -> TaskWorkflow:
    """TaskWorkflow wired for ``_resolve_and_resubmit`` scenarios.

    Mocks the collaborators orthogonal to the verdict-artifact boundary under
    test (mirrors ``test_merger_disposition_verdict.py``'s ``_setup``), but
    leaves ``_invoke`` REAL — only ``orchestrator.workflow.invoke_agent`` (one
    level deeper) is faked by each test.
    """
    workflow, _scheduler = _build_workflow(config, git_ops, task_assignment, AgentStub())
    _seed_workflow_artifacts(workflow, tmp_path=tmp_path)
    workflow.git_ops.rebase_onto_main = AsyncMock()  # type: ignore[method-assign]
    workflow._submit_to_merge_queue = AsyncMock(  # type: ignore[method-assign]
        return_value=WorkflowOutcome.DONE,
    )
    workflow._mark_blocked = AsyncMock(  # type: ignore[method-assign]
        return_value=WorkflowOutcome.BLOCKED,
    )
    workflow._write_merge_failure_review = MagicMock()  # type: ignore[method-assign]
    return workflow


def _setup_reviewer_workflow(
    config: OrchestratorConfig, git_ops: GitOps, task_assignment: TaskAssignment, tmp_path: Path,
) -> TaskWorkflow:
    """TaskWorkflow wired for ``_run_reviewer`` scenarios — no extra
    collaborator mocks needed beyond the shared artifacts seeding.
    """
    workflow, _scheduler = _build_workflow(config, git_ops, task_assignment, AgentStub())
    _seed_workflow_artifacts(workflow, tmp_path=tmp_path)
    return workflow


def _setup_judge_workflow(
    config: OrchestratorConfig, git_ops: GitOps, task_assignment: TaskAssignment, tmp_path: Path,
) -> TaskWorkflow:
    """TaskWorkflow wired for ``_run_completion_judge`` scenarios.

    ``get_diff_from_base`` is mocked to return a non-empty diff (mirrors
    ``test_judge_completion_verdict.py``'s ``_setup``) so the judge is
    actually invoked rather than empty-diff-skipped — ``base_commit='oldbase'``
    is stamped at ``artifacts.init()`` time (via ``_seed_workflow_artifacts``),
    so ``_run_completion_judge`` takes the ``get_diff_from_base`` branch.
    """
    workflow, _scheduler = _build_workflow(config, git_ops, task_assignment, AgentStub())
    _seed_workflow_artifacts(workflow, tmp_path=tmp_path)
    workflow.git_ops.get_diff_from_base = AsyncMock(  # type: ignore[method-assign]
        return_value='diff --git a/x b/x\n+line',
    )
    return workflow


# ---------------------------------------------------------------------------
# θ-local fake-runner factories — build an invoke_agent-shaped double that
# optionally writes a verdict artifact exactly as the real verdict-tools MCP
# server would (via _envelope / _submit_review_verdict / _submit_triage),
# then returns a configurable AgentResult. Kept θ-local (not folded into
# _workflow_helpers.AgentStub) for precise per-scenario control — see plan.json
# design_decisions "θ-local fake-runner closures".
#
# Bound to invoke_agent's signature: invoke_with_cap_retry's fast path
# (no usage_gate) calls `invoke(**invoke_kwargs, config_dir=...)` — every
# argument arrives as a keyword, so `**kwargs` catches all of them robustly
# (immune to production signature changes) while individual factories pull
# out just the keyword(s) they need (typically `cwd`).
# ---------------------------------------------------------------------------


def _fake_invoke_writes_merger_verdict(
    *, blocked: bool | None, reason: str = '', output: str = '', success: bool = True,
    session_id: str = 'sid',
) -> Callable[..., Any]:
    """``blocked=None`` writes NO verdict at all (agent never called
    ``submit_merge_disposition``) — the absent/fail-safe scenario.
    """

    async def _fake(*, cwd: Path, **kwargs: Any) -> AgentResult:
        if blocked is not None:
            _artifacts_for(cwd).write_verdict(
                'merger', _envelope('merger', session_id, {'blocked': blocked, 'reason': reason}),
            )
        return AgentResult(success=success, output=output)

    return _fake


def _fake_invoke_writes_reviewer_verdict(
    *, verdict: str | None, issues: list | None = None, summary: str = '',
    output: str = '', success: bool = True, role_name: str = 'reviewer_comprehensive',
    session_id: str = 'sid',
) -> Callable[..., Any]:
    """``verdict=None`` writes NO verdict at all (agent never called
    ``submit_review_verdict``). Routes a real write through
    ``_submit_review_verdict`` (not a hand-built envelope), mirroring
    ``AgentStub``'s reviewer stub in ``_workflow_helpers.py`` — this fake
    always passes reviewer==role, so it does NOT exercise the
    reviewer==role identity-mismatch rejection (verdict_tools.py:86); that
    case belongs to the verdict-tools server's own unit tests.
    """

    async def _fake(*, cwd: Path, **kwargs: Any) -> AgentResult:
        if verdict is not None:
            _submit_review_verdict(
                _artifacts_for(cwd), role_name, session_id, role_name, verdict,
                issues or [], summary,
            )
        return AgentResult(success=success, output=output)

    return _fake


def _fake_invoke_writes_judge_verdict(
    *, artifact: dict | None, structured_output: dict | None = None,
    success: bool = True, output: str = '', session_id: str = 'sid',
) -> Callable[..., Any]:
    """``artifact=None`` writes NO verdict artifact at all — simulating a
    judge that never called ``submit_completion_verdict`` (or a pre-ζ judge
    relying solely on the now-removed ``--json-schema`` contract).
    ``structured_output``, when given, models the LEGACY channel — post-η
    this must be ignored whenever ``artifact`` is also ``None``.
    """

    async def _fake(*, cwd: Path, **kwargs: Any) -> AgentResult:
        if artifact is not None:
            _artifacts_for(cwd).write_verdict('judge', _envelope('judge', session_id, artifact))
        return AgentResult(success=success, output=output, structured_output=structured_output)

    return _fake


def _fake_invoke_writes_triage_verdict(
    *, worktree: Path, write: bool = True, accepted: list | None = None,
    skipped: list | None = None, proposed_task_groups: list | None = None,
    success: bool = True, output: str = '', session_id: str = 'sid',
) -> Callable[..., Any]:
    """Closes over *worktree* explicitly rather than reading ``cwd`` from
    kwargs: ``_pre_triage_suggestions`` invokes with
    ``cwd=self.config.project_root`` (so the agent can browse the whole repo)
    but derives its verdict meta_root from ``self.worktree`` — the two
    deliberately differ (steward.py:727/760/796), so the fake must target the
    worktree-derived root the steward actually reads back from, not the
    invocation cwd.
    """

    async def _fake(**kwargs: Any) -> AgentResult:
        if write:
            _submit_triage(
                _artifacts_for(worktree), 'triage', session_id,
                accepted or [], skipped or [], proposed_task_groups or [],
            )
        return AgentResult(success=success, output=output)

    return _fake


def _make_triage_escalation(
    suggestions: list[dict], *, escalation_id: str = 'esc-42-1',
) -> Escalation:
    return Escalation(
        id=escalation_id,
        task_id='42',
        agent_role='orchestrator',
        severity='blocking',
        category='review_suggestions',
        summary=f'{len(suggestions)} suggestions pending triage',
        detail=json.dumps(suggestions),
    )


def _build_steward_for_triage(
    config: OrchestratorConfig, worktree: Path, *, task_id: str = '42',
) -> TaskSteward:
    """A real TaskSteward against a real on-disk meta-root (pre-initialized
    so _pre_triage_suggestions's "meta-root missing" diagnostic branch never
    fires — mirrors production, where TaskWorkflow._setup always creates it
    first). usage_gate=None (default) takes invoke_with_cap_retry's no-gate
    fast path — a single invocation, no cap-retry machinery.
    """
    worktree.mkdir(parents=True, exist_ok=True)
    _artifacts_for(worktree).init(task_id, 'Verdict boundary task', 'exercise triage boundary')
    return TaskSteward(
        task_id=task_id,
        task={'id': task_id, 'title': 'Verdict boundary task', 'description': 'd'},
        worktree=worktree,
        config=config,
        mcp=FakeMcp(),  # type: ignore[arg-type]
        escalation_queue=MagicMock(),
        briefing=FakeBriefing(),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# S1 — Merger boundary (scenarios 1-3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMergerBoundary:
    """Scenarios 1-3: ``TaskWorkflow._resolve_and_resubmit`` driven through the
    REAL ``_invoke``, with only ``orchestrator.workflow.invoke_agent`` faked.
    Distinguishes θ from ``test_merger_disposition_verdict.py``, which mocks
    ``_invoke`` wholesale and never exercises ``_inject_verdict_tools_mcp``,
    route resolution, or the real clear→spawn→read sequencing.

    Plus one amendment scenario beyond the original 1-3: an invocation
    failure is untrusted even when a valid verdict was written before it
    failed (I-FAIL-SAFE's "success" leg, distinct from the absent/malformed
    leg already covered by scenario 3).
    """

    async def test_blocked_false_verdict_proceeds_despite_blocked_prose(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """(1) verdict blocked=False wins over stray "BLOCKED" prose in the
        agent's free-text output — the removed-substring-grep regression
        proof; a pre-γ implementation would false-block on this prose alone.
        """
        workflow = _setup_merger_workflow(config, git_ops, task_assignment, tmp_path)
        monkeypatch.setattr(
            'orchestrator.workflow.invoke_agent',
            _fake_invoke_writes_merger_verdict(
                blocked=False,
                output='Resolved cleanly. This looks BLOCKED but is not.',
            ),
        )

        outcome = await workflow._resolve_and_resubmit(
            'task/42', 'conflict in lib.py', merge_phase=True,
        )

        workflow._submit_to_merge_queue.assert_awaited_once()  # type: ignore[attr-defined]
        workflow._mark_blocked.assert_not_awaited()  # type: ignore[attr-defined]
        assert outcome == WorkflowOutcome.DONE

    async def test_blocked_true_verdict_blocks_with_tool_supplied_reason(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """(2) verdict blocked=True routes to _mark_blocked with its reason."""
        workflow = _setup_merger_workflow(config, git_ops, task_assignment, tmp_path)
        monkeypatch.setattr(
            'orchestrator.workflow.invoke_agent',
            _fake_invoke_writes_merger_verdict(
                blocked=True, reason='ARCH mismatch', output='Looks fine.',
            ),
        )

        outcome = await workflow._resolve_and_resubmit(
            'task/42', 'conflict in lib.py', merge_phase=True,
        )

        workflow._mark_blocked.assert_awaited_once()  # type: ignore[attr-defined]
        args, _kwargs = workflow._mark_blocked.await_args  # type: ignore[attr-defined]
        assert 'ARCH mismatch' in args[0]
        workflow._submit_to_merge_queue.assert_not_awaited()  # type: ignore[attr-defined]
        assert outcome == WorkflowOutcome.BLOCKED

    async def test_absent_verdict_is_failsafe_blocked(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """(3) no verdict written at all (agent never called
        submit_merge_disposition) => fail-safe blocked-equivalent
        (I-FAIL-SAFE); _submit_to_merge_queue never runs.
        """
        workflow = _setup_merger_workflow(config, git_ops, task_assignment, tmp_path)
        monkeypatch.setattr(
            'orchestrator.workflow.invoke_agent',
            _fake_invoke_writes_merger_verdict(blocked=None, output='no disposition emitted'),
        )

        outcome = await workflow._resolve_and_resubmit(
            'task/42', 'conflict in lib.py', merge_phase=True,
        )

        workflow._mark_blocked.assert_awaited_once()  # type: ignore[attr-defined]
        workflow._submit_to_merge_queue.assert_not_awaited()  # type: ignore[attr-defined]
        assert outcome == WorkflowOutcome.BLOCKED

    async def test_invocation_failure_is_untrusted_despite_valid_verdict(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """An invocation failure is untrusted even when a valid verdict was
        written before it failed (workflow.py: ``if not merger_result.success
        or verdict is None`` — "an invocation failure ... is untrusted even
        if it happened to write a verdict before failing"). A written
        blocked=False disposition must NOT override success=False; the
        merger still fails safe to BLOCKED and never reaches the merge queue.
        """
        workflow = _setup_merger_workflow(config, git_ops, task_assignment, tmp_path)
        monkeypatch.setattr(
            'orchestrator.workflow.invoke_agent',
            _fake_invoke_writes_merger_verdict(
                blocked=False, success=False, output='crashed after writing disposition',
            ),
        )

        outcome = await workflow._resolve_and_resubmit(
            'task/42', 'conflict in lib.py', merge_phase=True,
        )

        workflow._mark_blocked.assert_awaited_once()  # type: ignore[attr-defined]
        args, _kwargs = workflow._mark_blocked.await_args  # type: ignore[attr-defined]
        assert 'Merger invocation failed' in args[0]
        workflow._submit_to_merge_queue.assert_not_awaited()  # type: ignore[attr-defined]
        assert outcome == WorkflowOutcome.BLOCKED


# ---------------------------------------------------------------------------
# S2 — Reviewer boundary (scenarios 4-5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReviewerBoundary:
    """Scenarios 4-5: ``TaskWorkflow._run_reviewer`` driven through the REAL
    ``_invoke``, with only ``orchestrator.workflow.invoke_agent`` faked.

    Plus one amendment scenario beyond the original 4-5: an invocation
    failure is untrusted even when a valid PASS verdict was written before
    it failed (I-FAIL-SAFE's "success" leg, distinct from the absent-verdict
    leg already covered by scenario 5).
    """

    async def test_pass_verdict_is_aggregate_consumable_prose_ignored(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """(4) a written PASS verdict is returned verbatim — the
        aggregate_reviews-consumable payload shape, not an ERROR — even
        though the agent's free-text output separately claims issues.
        """
        workflow = _setup_reviewer_workflow(config, git_ops, task_assignment, tmp_path)
        monkeypatch.setattr(
            'orchestrator.workflow.invoke_agent',
            _fake_invoke_writes_reviewer_verdict(
                verdict='PASS', summary='Looks good.',
                output='On reflection I think this has ISSUES_FOUND actually...',
            ),
        )

        result = await workflow._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff --git a/x b/x')

        assert result == {
            'reviewer': 'reviewer_comprehensive',
            'verdict': 'PASS',
            'issues': [],
            'summary': 'Looks good.',
        }

    async def test_absent_verdict_synthesizes_error_fallback(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """(5) no verdict written at all => _run_reviewer synthesizes the
        fail-safe {'verdict': 'ERROR'} payload the existing retry/aggregate
        path already consumes; reviewer==role identity preserved.
        """
        workflow = _setup_reviewer_workflow(config, git_ops, task_assignment, tmp_path)
        monkeypatch.setattr(
            'orchestrator.workflow.invoke_agent',
            _fake_invoke_writes_reviewer_verdict(verdict=None, output='no verdict emitted'),
        )

        result = await workflow._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff --git a/x b/x')

        assert result['verdict'] == 'ERROR'
        assert result['reviewer'] == 'reviewer_comprehensive'

    async def test_invocation_failure_is_untrusted_despite_valid_verdict(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """An invocation failure is untrusted even when a valid PASS verdict
        was written before it failed (workflow.py: ``not result.success or
        ...`` — "an invocation failure ... is untrusted even if it happened
        to write a verdict before failing"). A written PASS verdict must NOT
        override success=False; the reviewer still degrades to the fail-safe
        ERROR disposition.
        """
        workflow = _setup_reviewer_workflow(config, git_ops, task_assignment, tmp_path)
        monkeypatch.setattr(
            'orchestrator.workflow.invoke_agent',
            _fake_invoke_writes_reviewer_verdict(
                verdict='PASS', summary='Looks good.', success=False,
                output='crashed after writing verdict',
            ),
        )

        result = await workflow._run_reviewer(REVIEWER_COMPREHENSIVE, 'diff --git a/x b/x')

        assert result['verdict'] == 'ERROR'
        assert result['reviewer'] == 'reviewer_comprehensive'


# ---------------------------------------------------------------------------
# S3 — Judge boundary (scenarios 6, 7, 9)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestJudgeBoundary:
    """Scenarios 6, 7, 9: ``TaskWorkflow._run_completion_judge`` driven
    through the REAL ``_invoke``, with only ``orchestrator.workflow.invoke_agent``
    faked. ``get_diff_from_base`` is mocked to a non-empty diff so the judge
    is actually invoked (not empty-diff-skipped).

    Plus one amendment scenario beyond the original 6, 7, 9: an invocation
    failure is untrusted even when a valid complete=True verdict was written
    before it failed — checked BEFORE the artifact is even read, distinct
    from scenario 9's absent-both leg.
    """

    async def test_complete_substantive_verdict_returned_verbatim(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """(6) complete=True, substantive_work=True => the verdict dict is
        returned verbatim — the value the caller uses for early-exit
        completion.
        """
        workflow = _setup_judge_workflow(config, git_ops, task_assignment, tmp_path)
        monkeypatch.setattr(
            'orchestrator.workflow.invoke_agent',
            _fake_invoke_writes_judge_verdict(artifact={
                'complete': True, 'reasoning': 'all plan steps covered',
                'uncovered_plan_steps': [], 'substantive_work': True,
            }),
        )

        verdict = await workflow._run_completion_judge([])

        assert verdict == {
            'complete': True, 'reasoning': 'all plan steps covered',
            'uncovered_plan_steps': [], 'substantive_work': True,
        }

    async def test_complete_non_substantive_verdict_withholds_completion(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """(7) complete=True, substantive_work=False => the verdict is still
        returned verbatim (this method does no substantive_work gating
        itself — it only requires the four keys be present), but carries
        substantive_work=False: the field the CALLER rejects a completion on
        ("Safety: reject complete=True if substantive_work=False",
        workflow.py ~6121-6126). Proves the artifact boundary faithfully
        passes the safety-relevant field through to that gate.
        """
        workflow = _setup_judge_workflow(config, git_ops, task_assignment, tmp_path)
        monkeypatch.setattr(
            'orchestrator.workflow.invoke_agent',
            _fake_invoke_writes_judge_verdict(artifact={
                'complete': True, 'reasoning': 'looks done but trivial',
                'uncovered_plan_steps': [], 'substantive_work': False,
            }),
        )

        verdict = await workflow._run_completion_judge([])

        assert verdict is not None
        assert verdict['complete'] is True
        assert verdict['substantive_work'] is False

    async def test_absent_verdict_and_no_structured_output_returns_none(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """(9) no verdict artifact and no legacy structured_output => None
        (I-FAIL-SAFE: keep iterating, never a false completion).
        """
        workflow = _setup_judge_workflow(config, git_ops, task_assignment, tmp_path)
        monkeypatch.setattr(
            'orchestrator.workflow.invoke_agent',
            _fake_invoke_writes_judge_verdict(artifact=None, structured_output=None),
        )

        verdict = await workflow._run_completion_judge([])

        assert verdict is None

    async def test_invocation_failure_is_untrusted_despite_valid_verdict(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """An invocation failure is untrusted even when a valid complete=True
        verdict was written before it failed (workflow.py: ``if not
        result.success: ... return None``, checked BEFORE the verdict
        artifact is even read). A written complete=True/substantive_work=True
        verdict must NOT override success=False; the judge still fails safe
        to None (keep iterating, never a false completion — I-FAIL-SAFE).
        """
        workflow = _setup_judge_workflow(config, git_ops, task_assignment, tmp_path)
        monkeypatch.setattr(
            'orchestrator.workflow.invoke_agent',
            _fake_invoke_writes_judge_verdict(
                artifact={
                    'complete': True, 'reasoning': 'all plan steps covered',
                    'uncovered_plan_steps': [], 'substantive_work': True,
                },
                success=False,
            ),
        )

        verdict = await workflow._run_completion_judge([])

        assert verdict is None


# ---------------------------------------------------------------------------
# S4 — Judge post-η structured_output-ignored (scenario 8, REFRAMED)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestJudgeStructuredOutputIgnoredPostEta:
    """Scenario 8, REFRAMED (design decision "Scenario 8 reframed post-η" in
    plan.json). The task's literal scenario 8 ("judge transition fallback via
    structured_output still gates") is dead: η (task 2487, a dependency of θ)
    removed the ζ structured_output fallback — workflow.py:6359-6371 reads
    ``verdicts/judge.json`` ONLY. This proves the removal instead: a judge
    that returns ONLY a legacy ``structured_output`` payload (no verdict-tools
    artifact) must NOT complete the task — a regression guard for η's cleanup.
    """

    async def test_structured_output_without_artifact_is_ignored(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        workflow = _setup_judge_workflow(config, git_ops, task_assignment, tmp_path)
        monkeypatch.setattr(
            'orchestrator.workflow.invoke_agent',
            _fake_invoke_writes_judge_verdict(
                artifact=None,
                structured_output={
                    'complete': True, 'reasoning': 'legacy path only',
                    'uncovered_plan_steps': [], 'substantive_work': True,
                },
            ),
        )

        verdict = await workflow._run_completion_judge([])

        assert verdict is None


# ---------------------------------------------------------------------------
# S5 — Triage boundary (scenarios 10-11)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTriageBoundary:
    """Scenarios 10-11: ``TaskSteward._pre_triage_suggestions`` driven through
    its REAL invoke seam — monkeypatch ``orchestrator.steward.invoke_agent``
    (NOT ``workflow.invoke_agent``; the steward binds its own module-level
    ``invoke_agent`` symbol, steward.py:33/790-793).
    """

    _SUGGESTIONS = [
        {
            'description': 'suggestion 0', 'location': 'file_0.py',
            'reviewer': 'bot', 'category': 'style',
        },
        {
            'description': 'suggestion 1', 'location': 'file_1.py',
            'reviewer': 'bot', 'category': 'style',
        },
    ]

    async def test_written_verdict_produces_modified_pretriaged_escalation(
        self, config, tmp_path, monkeypatch,
    ):
        """(10) fake writes verdicts/triage.json via submit_triage =>
        _pre_triage_suggestions returns a MODIFIED escalation whose detail
        carries the pre-triaged markdown (accepted/skipped counts, proposed
        task groups) — extract_triage_verdict consumed the real artifact.
        """
        worktree = tmp_path / 'wt'
        steward = _build_steward_for_triage(config, worktree)
        escalation = _make_triage_escalation(self._SUGGESTIONS)
        monkeypatch.setattr(
            'orchestrator.steward.invoke_agent',
            _fake_invoke_writes_triage_verdict(
                worktree=worktree,
                accepted=[{
                    'index': 0, 'suggestion': 'Add test for X', 'reason': 'missing coverage',
                    'files': ['orchestrator/foo.py'], 'proposed_task_title': 'Add coverage for X',
                }],
                skipped=[{
                    'index': 1, 'suggestion': 'Rename Y', 'reason': 'not worth it',
                }],
                proposed_task_groups=[{
                    'title': 'Add coverage for X', 'description': 'Add missing test coverage',
                    'accepted_indices': [0],
                }],
            ),
        )

        result = await steward._pre_triage_suggestions(escalation)

        assert result is not escalation
        assert '## Pre-Triaged Results' in result.detail
        assert 'Add coverage for X' in result.detail
        assert '1 accepted, 1 skipped' in result.summary

    async def test_absent_verdict_falls_back_to_original_escalation(
        self, config, tmp_path, monkeypatch,
    ):
        """(11) fake writes NO verdict (cleared slot) => read_verdict absent
        => returns the ORIGINAL escalation unchanged (inline-triage
        fallback). A stale verdict pre-seeded as if left over from a prior
        _pre_triage_suggestions invocation on this same worktree must be
        cleared before this spawn (I-FRESH) — proven here because it does
        NOT leak through as a modified escalation.
        """
        worktree = tmp_path / 'wt'
        steward = _build_steward_for_triage(config, worktree)
        escalation = _make_triage_escalation(self._SUGGESTIONS)
        _artifacts_for(worktree).write_verdict(
            'triage',
            _envelope('triage', 'stale-sid', {
                'accepted': [{
                    'index': 0, 'suggestion': 'stale', 'reason': 'stale',
                    'files': [], 'proposed_task_title': 'stale',
                }],
                'skipped': [], 'proposed_task_groups': [],
            }),
        )
        monkeypatch.setattr(
            'orchestrator.steward.invoke_agent',
            _fake_invoke_writes_triage_verdict(worktree=worktree, write=False),
        )

        result = await steward._pre_triage_suggestions(escalation)

        assert result is escalation


# ---------------------------------------------------------------------------
# S6 — Freshness / staleness (scenarios 12-13, I-FRESH)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFreshnessGuards:
    """Scenarios 12-13: a verdict pre-seeded on the real .task-meta root —
    simulating a prior iteration's (12) or a prior task's reused pooled
    lane's (13) leftover artifact — must never masquerade as the CURRENT
    spawn's output. Exercises the SAME pre-spawn clear_verdict mechanism the
    per-role tests cover at the mock-_invoke level, here through the real
    _invoke path.
    """

    async def test_cross_iteration_stale_judge_verdict_is_cleared(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """(12) a stale complete=True judge verdict from a "prior iteration"
        is cleared before this spawn; this spawn's fake writes nothing =>
        None, not the stale complete verdict.
        """
        workflow = _setup_judge_workflow(config, git_ops, task_assignment, tmp_path)
        assert workflow.worktree is not None  # set by _seed_workflow_artifacts above
        _artifacts_for(workflow.worktree).write_verdict(
            'judge',
            _envelope('judge', 'stale-sid', {
                'complete': True, 'reasoning': 'stale', 'uncovered_plan_steps': [],
                'substantive_work': True,
            }),
        )
        monkeypatch.setattr(
            'orchestrator.workflow.invoke_agent',
            _fake_invoke_writes_judge_verdict(artifact=None, structured_output=None),
        )

        verdict = await workflow._run_completion_judge([])

        assert verdict is None

    async def test_cross_task_stale_merger_verdict_is_cleared(
        self, config, git_ops, task_assignment, tmp_path, monkeypatch,
    ):
        """(13) a stale blocked=False merger verdict from a "prior task on a
        reused pooled lane" is cleared before this spawn; this spawn's fake
        writes nothing => fail-safe blocked (_mark_blocked), NOT the stale
        blocked=False that would have let it proceed.
        """
        workflow = _setup_merger_workflow(config, git_ops, task_assignment, tmp_path)
        assert workflow.worktree is not None  # set by _seed_workflow_artifacts above
        _artifacts_for(workflow.worktree).write_verdict(
            'merger', _envelope('merger', 'stale-sid', {'blocked': False, 'reason': ''}),
        )
        monkeypatch.setattr(
            'orchestrator.workflow.invoke_agent',
            _fake_invoke_writes_merger_verdict(blocked=None, output='nothing new'),
        )

        outcome = await workflow._resolve_and_resubmit(
            'task/42', 'conflict in lib.py', merge_phase=True,
        )

        workflow._mark_blocked.assert_awaited_once()  # type: ignore[attr-defined]
        workflow._submit_to_merge_queue.assert_not_awaited()  # type: ignore[attr-defined]
        assert outcome == WorkflowOutcome.BLOCKED


# ---------------------------------------------------------------------------
# S7 — Shared machinery intact (scenario 14, I-SHARED-INTACT)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSharedMachineryIntact:
    """Scenario 14 (I-SHARED-INTACT).

    CANONICAL GUARD: ``test_shared_output_schema_intact.py`` (η / task 2487)
    is the full standing guard for the shared ``shared.cli_invoke``
    ``output_schema`` path that fused-memory recon/curator/adjudicator ride —
    its third test is literally labeled "boundary test #14". This class is a
    COMPACT re-assertion of the same invariant for THIS gate's own
    self-completeness (the task mandates all 14 scenarios be covered here) —
    it deliberately does not duplicate that file's full ``_drive_recon_like``
    helper.
    """

    async def test_recon_like_output_schema_path_still_functions(self, tmp_path: Path) -> None:
        """A recon/curator-like invocation (non-trivial output_schema +
        disallowed_tools=['*']) still: (a) renders --json-schema with the
        schema JSON; (b) expands '*' to _REAL_BUILTIN_TOOLS_DENYLIST, sparing
        the synthetic StructuredOutput tool; (c) parses a StructuredOutput
        subprocess payload back into AgentResult.structured_output.
        """
        captured: list[list[str]] = []

        async def fake_run_subprocess(cmd: list[str], *args: Any, **kwargs: Any) -> _SubprocessResult:
            captured.append(list(cmd))
            stdout = json.dumps({
                'result': 'ok', 'is_error': False, 'subtype': 'success',
                'cost_usd': 0.01, 'duration_ms': 100, 'num_turns': 1,
                'session_id': 'test-session',
                'structured_output': {'summary': 'ok', 'findings': ['a', 'b']},
            })
            return _SubprocessResult(
                stdout=stdout, stderr='', returncode=0, duration_ms=100, timed_out=False,
            )

        schema = {
            'type': 'object',
            'properties': {
                'summary': {'type': 'string'},
                'findings': {'type': 'array', 'items': {'type': 'string'}},
            },
            'required': ['summary', 'findings'],
        }

        with patch('shared.cli_invoke._run_subprocess', side_effect=fake_run_subprocess):
            result = await invoke_claude_agent(
                prompt='hello', system_prompt='sys prompt', cwd=tmp_path,
                model='opus', max_turns=20, max_budget_usd=3.0,
                allowed_tools=['Read'], disallowed_tools=['*'],
                mcp_config=None, output_schema=schema,
                permission_mode='bypassPermissions',
            )

        assert len(captured) == 1, f'expected exactly one subprocess invocation; got {captured!r}'
        argv = captured[0]

        assert '--json-schema' in argv
        assert argv[argv.index('--json-schema') + 1] == json.dumps(schema)

        d_idx = argv.index('--disallowed-tools')
        js_idx = argv.index('--json-schema', d_idx)
        deny_values = argv[d_idx + 1:js_idx]
        assert deny_values == _REAL_BUILTIN_TOOLS_DENYLIST, f'got {deny_values!r}'
        assert _SCHEMA_OUTPUT_TOOL not in argv

        assert isinstance(result, AgentResult)
        assert result.structured_output == {'summary': 'ok', 'findings': ['a', 'b']}
