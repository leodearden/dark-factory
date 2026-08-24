"""Infra-hold resume never manufactures a claimant-less in-progress strand
(task 3538 / PRD γ3, D6; `plans/task-escalation-state-graph-prd.md`).

This is the task's headline user-observable signal.  Today
``Harness._cascade_unblock_member``'s infra-held branch writes ``'in-progress'``
with no claimant, which is EXACTLY the shape ``shared.task_claimant.is_stranded``
is defined to detect.  The row is then undispatchable (dispatch is pending-only
and status-first — ``Scheduler._eligible_for_dispatch`` returns early at
``status != 'pending'``) and its only route back to execution is the stranded
sweep's ``_RECOVERY`` row (c) REVERT_TO_PENDING, which is keyed on
``has_open_escalation is False`` — so any open record converts a transient infra
hold into permanent starvation (the 3465-shaped failure).

The comment defending that write claims re-pending would force the task to
re-compete for its implement footprint.  That premise was verified and the
conclusion refuted: the sweep re-pends anyway (deferred by ≤ one
``stranded_reconcile_interval_secs``, never avoided), so no-recompete was never
actually delivered.  Meanwhile the property that DOES matter — resume-at-verify,
i.e. not re-running the implementer — is branch-keyed, not status-keyed: it is
delivered by ``TaskWorkflow._has_prior_implementation`` (+ ``green_checkpoint_at_tip``),
which reads the worktree's artifacts and never consults the task row's status.

So: write ``'pending'``, and prove the skip survives.

RED before step-14 on (1) and (2); the branch-keyed class below is a
characterization test that must be GREEN both before and after — it is the
evidence that the status change costs nothing.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.models import Escalation
from shared.task_claimant import is_stranded

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.harness import Harness
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow

# ---------------------------------------------------------------------------
# Harness fixture — mirrors test_harness_infra_hold_repend.py:35-64
# ---------------------------------------------------------------------------


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config) -> Harness:
    """A Harness with mocked internals, wired for cascade-unblock unit tests."""
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.get_statuses = AsyncMock(return_value=({}, None))
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.update_task = AsyncMock(return_value=True)
    h.scheduler.get_task = AsyncMock(return_value=None)

    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()
    return h


def _make_infra_esc(
    task_id: str = '3538',
    status: str = 'resolved',
    level: int = 1,
    resolved_by: str | None = None,
) -> Escalation:
    """A minimal resolved L1 infra_issue escalation (mirrors the sibling suite)."""
    return Escalation(
        id=f'esc-{task_id}-99',
        task_id=task_id,
        agent_role='workflow',
        severity='blocking',
        category='infra_issue',
        summary='ENOSPC during verify warm marker write',
        level=level,
        status=status,
        resolved_by=resolved_by,
    )


async def _drive_orphan_resume(harness: Harness, esc: Escalation) -> None:
    """Fire the orphan resume path and drain it.

    ``_on_escalation_resolved`` is SYNC and schedules the cascade coroutine onto
    ``_background_tasks`` — without the gather the assertions race the write.
    """
    harness._escalation_events.pop(esc.task_id, None)
    harness._on_escalation_resolved(esc)
    await asyncio.gather(*list(harness._background_tasks))


def _written_statuses(harness: Harness, task_id: str) -> list[str]:
    return [
        c.args[1]
        for c in harness.scheduler.set_task_status.await_args_list  # type: ignore[attr-defined]
        if c.args[0] == task_id
    ]


# ---------------------------------------------------------------------------
# (1) + (2): the strand shape is never manufactured; the row is re-pended
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInfraResumeNeverManufacturesStrand:
    """The infra-hold resume writes 'pending', never a claimant-less 'in-progress'."""

    @staticmethod
    def _infra_row(harness: Harness, tid: str) -> None:
        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid,
            'status': 'infra-hold',
            'metadata': {},
        })
        harness.scheduler.get_status = AsyncMock(return_value='infra-hold')

    async def test_infra_resume_never_writes_in_progress(self, harness: Harness):
        """No 'in-progress' write — the cascade holds no claimant to stamp.

        The harness slot-``finally`` that would normally null the claimant is
        not even in play here: this is the ORPHAN branch, so there is no live
        workflow and nothing ever writes ``claimant_run_id``.  An 'in-progress'
        write therefore lands the row directly in ``is_stranded``'s target shape.
        """
        tid = '3538'
        self._infra_row(harness, tid)

        await _drive_orphan_resume(harness, _make_infra_esc(task_id=tid))

        assert 'in-progress' not in _written_statuses(harness, tid), (
            "cascade wrote 'in-progress' for an infra-held task. That row has no "
            'claimant, so it is stranded on the write, undispatchable under the '
            "pending-only dispatch gate, and recoverable only by the stranded "
            'sweep — which skips any task with an open escalation.'
        )

    async def test_resulting_row_shape_is_not_stranded(self, harness: Harness):
        """State the invariant in the predicate's own terms.

        Reconstruct the row as the cascade leaves it (written status + the
        claimant fields the orphan path never sets) and assert
        ``is_stranded`` is False.  Asserting through the real predicate rather
        than a hand-rolled status check means this test cannot drift from the
        definition the stranded sweep consumes.
        """
        tid = '3538'
        self._infra_row(harness, tid)

        await _drive_orphan_resume(harness, _make_infra_esc(task_id=tid))

        written = _written_statuses(harness, tid)
        assert written, 'cascade made no status write at all'
        row = {
            'id': tid,
            'status': written[-1],
            'claimant_run_id': None,   # orphan path: nothing stamps a claimant
            'heartbeat_at': None,      # …and nothing runs a heartbeat loop
            'metadata': {},
        }
        assert is_stranded(row, datetime.now(UTC), timedelta(seconds=300)) is False, (
            f'the resumed row {row["status"]!r} with a NULL claimant satisfies '
            'is_stranded — the cascade manufactured the exact strand the sweep '
            'exists to clean up'
        )

    async def test_infra_resume_writes_pending_exactly_once(self, harness: Harness):
        """'pending' is written exactly once — the row is dispatchable again.

        ``Scheduler._eligible_for_dispatch`` gates on ``status == 'pending'``
        BEFORE any lock work, so 'pending' is the only status from which the
        scheduler can pick this task back up.
        """
        tid = '3538'
        self._infra_row(harness, tid)

        await _drive_orphan_resume(harness, _make_infra_esc(task_id=tid))

        assert _written_statuses(harness, tid) == ['pending']

    async def test_non_infra_blocked_row_still_takes_table_b_path(
        self, harness: Harness,
    ):
        """Negative control: a plain blocked row is untouched by this change.

        Same target status, but reached through the ordinary Table B
        (``effect_for('resume', …)``) branch — proving the infra pre-gate did
        not swallow the general path.
        """
        tid = '3539'
        harness.scheduler.get_task = AsyncMock(return_value={
            'id': tid, 'status': 'blocked', 'metadata': {},
        })
        harness.scheduler.get_status = AsyncMock(return_value='blocked')

        await _drive_orphan_resume(harness, _make_infra_esc(task_id=tid))

        harness.scheduler.set_task_status.assert_awaited_once_with(tid, 'pending')  # type: ignore[attr-defined]
        # Table B consults get_status; the infra pre-gate only reads get_task.
        harness.scheduler.get_status.assert_awaited()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# (3): resume-at-verify is branch-keyed, so re-pending costs nothing
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'lib.py').write_text('x = 1\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


@pytest.fixture
def branch_config(git_repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
        git=GitConfig(
            main_branch='main', branch_prefix='task/',
            remote='origin', worktree_dir='.worktrees',
        ),
    )


@pytest.fixture
def branch_git_ops(branch_config: OrchestratorConfig) -> GitOps:
    return GitOps(branch_config.git, branch_config.project_root)


@pytest.mark.asyncio
class TestInfraResumeKeepsBranchKeyedSkip:
    """Characterization: the skip-re-implementation property does not read status.

    GREEN before AND after step-14 — that is the point.  If re-pending cost us
    resume-at-verify, this class would have to change; it does not, because
    ``_has_prior_implementation`` takes its evidence from the worktree
    (``base_commit`` + the durable iteration log), never from the task row.
    """

    async def _branch_with_prior_work(
        self, branch_config, branch_git_ops,
    ) -> tuple[TaskWorkflow, TaskArtifacts, str]:
        assignment = TaskAssignment(
            task_id='3538',
            task={
                'id': '3538', 'title': 'X', 'description': '',
                'status': 'infra-hold', 'metadata': {'files': ['lib']},
                'dependencies': [],
            },
            modules=['lib'],
        )
        wt_info = await branch_git_ops.create_worktree(assignment.task_id)
        wt = wt_info.path

        workflow = TaskWorkflow(
            assignment=assignment,
            config=branch_config,
            git_ops=branch_git_ops,
            scheduler=MagicMock(),  # type: ignore[arg-type]
            briefing=MagicMock(),  # type: ignore[arg-type]
            mcp=MagicMock(),  # type: ignore[arg-type]
        )
        workflow.worktree = wt
        artifacts = TaskArtifacts(wt)
        artifacts.init('3538', 'X', 'desc', base_commit=wt_info.base_commit)
        workflow.artifacts = artifacts

        # Prior implementation: a real commit beyond base + the durable log
        # entry.  _has_prior_implementation requires BOTH signals.
        (wt / 'impl.py').write_text('implementation\n')
        step_commit = await branch_git_ops.commit(wt, 'feat: GREEN — step-1')
        assert step_commit, 'Setup: expected a real commit beyond base'
        artifacts.append_iteration_log({
            'agent': 'implementer',
            'steps_completed': ['step-1'],
            'commit': step_commit,
        })
        return workflow, artifacts, step_commit

    @pytest.mark.parametrize('row_status', ['infra-hold', 'in-progress', 'pending'])
    async def test_prior_implementation_is_status_independent(
        self, branch_config, branch_git_ops, row_status,
    ):
        """has_work stays True whatever the task row says.

        The parametrization spans the status the cascade reads ('infra-hold'),
        the one it writes today ('in-progress'), and the one it writes after
        step-14 ('pending').  Identical answer in all three ⇒ the resume-side
        status write cannot cost us the skip.
        """
        workflow, _artifacts, _sha = await self._branch_with_prior_work(
            branch_config, branch_git_ops,
        )
        workflow.assignment.task['status'] = row_status

        head = await workflow._get_head_commit()
        assert workflow._has_prior_implementation(wt_head=head).has_work is True

    async def test_completed_plan_steps_are_rederived_after_repend(
        self, branch_config, branch_git_ops,
    ):
        """The re-dispatched workflow re-derives its completed steps.

        This is the mechanism that makes a re-pended task resume at verify
        rather than re-implement: EXECUTE sees step-1 already done.
        ``_rederive_step_status_from_branch_state`` is gated on
        ``_has_prior_implementation``, so it inherits the same
        status-independence.
        """
        workflow, artifacts, _sha = await self._branch_with_prior_work(
            branch_config, branch_git_ops,
        )
        workflow.assignment.task['status'] = 'pending'  # as step-14 leaves it

        plan = {
            'task_id': '3538', 'title': 'X', 'analysis': 'A',
            'prerequisites': [],
            'steps': [
                {'id': 'step-1', 'type': 'impl', 'status': 'pending', 'commit': None},
                {'id': 'step-2', 'type': 'impl', 'status': 'pending', 'commit': None},
            ],
        }
        artifacts.write_plan(plan)
        artifacts.stamp_plan_provenance(workflow.session_id)
        workflow.plan = artifacts.read_plan()

        rederived = await workflow._rederive_step_status_from_branch_state()

        assert rederived == ['step-1']
        steps = {s['id']: s for s in artifacts.read_plan()['steps']}
        assert steps['step-1']['status'] == 'done', (
            'step-1 was not re-derived — a re-pended task would re-implement it'
        )
        assert steps['step-2']['status'] == 'pending'
