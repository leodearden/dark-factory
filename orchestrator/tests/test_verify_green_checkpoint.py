"""End-to-end test for the durable verified-green VERIFY checkpoint (task 2752).

Drives ``TaskWorkflow._execute_verify_review_loop`` with a real worktree, a
fully-done plan, and a real ``EventStore`` pre-seeded — under a PRIOR run_id —
with a durable ``workflow_verify`` green at the branch's current tip. On a
fresh dispatch (a NEW run_id on the same db, the cross-restart scenario) the
loop must SKIP ``_verify_debugfix_loop`` entirely: the checkpoint fires, emits
``phase_skipped(verify)``, and (because verify did not run this cycle) does NOT
re-emit ``workflow_verify`` under the new run.

step-7: failing test, written before the checkpoint exists in
``_execute_verify_review_loop`` (RED — verify runs, no phase_skipped, a fresh
workflow_verify row is emitted). step-8 inserts the checkpoint and makes it green.

The redispatch boundary (task 5409) is driven through the real emitter: a green
recorded at a real HEAD lets the next run skip VERIFY, and a green whose tip
could not be read records why (``tip_sha_unresolved``) and lets it skip nothing.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from _git_fixtures import seed_repo
from _orch_helpers import pydantic_spec
from _unreadable_head import make_head_unreadable

from orchestrator.artifacts import ReviewAggregation, TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.verify import VerifyResult
from orchestrator.verify_checkpoint import green_checkpoint_at_tip
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

TASK_ID = '42'
GREEN_TIP = 'tip-xyz'


def _make_e2e_workflow(tmp_path: Path, event_store: EventStore | None) -> TaskWorkflow:
    """A workflow with a real worktree + fully-done plan, mirroring
    ``test_suggestion_triage``'s ``_make_e2e_workflow``, with *event_store*
    attached when given so the checkpoint has a durable history to read."""
    assignment = MagicMock()
    assignment.task_id = TASK_ID
    assignment.task = {'id': TASK_ID, 'title': 'Test Task', 'description': 'desc'}

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.git = GitConfig()  # branch_prefix defaults to 'task/'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.max_execute_iterations = 5
    config.inter_iteration_rebase = False
    config.steward_completion_timeout = 300.0

    mcp = MagicMock()
    mcp.url = 'http://localhost:8002'

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=MagicMock(),
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=mcp,
        event_store=event_store,
    )

    wt = tmp_path / 'wt'
    wt.mkdir()
    wf.worktree = wt
    wf.artifacts = TaskArtifacts(wt)
    wf.artifacts.init(TASK_ID, 'Test Task', 'desc', base_commit='deadbeef')

    plan = {
        'task_id': TASK_ID,
        'title': 'Test Task',
        'files': [],
        'analysis': 'Test analysis',
        'prerequisites': [],
        'steps': [
            {
                'id': 'step-1',
                'type': 'impl',
                'description': 'Write code',
                'status': 'done',
                'commit': 'abc123',
            },
        ],
    }
    wf.artifacts.write_plan(plan)
    wf.artifacts.stamp_plan_provenance(wf.session_id)
    return wf


def _stub_verify_and_review(
    wf: TaskWorkflow,
    monkeypatch: pytest.MonkeyPatch,
    during_verify: Callable[[], Awaitable[None]] | None = None,
) -> tuple[AsyncMock, AsyncMock]:
    """Verify passes (after awaiting *during_verify*) and review is
    non-blocking with no suggestions; returns the ``(verify, review)`` mocks."""
    async def verify() -> WorkflowOutcome:
        if during_verify is not None:
            await during_verify()
        return WorkflowOutcome.DONE

    verify_mock = AsyncMock(side_effect=verify)
    review_mock = AsyncMock(return_value=ReviewAggregation(
        has_blocking_issues=False,
        blocking_issues=[],
        suggestions=[],
        reviews={'analyst': {}},
    ))
    monkeypatch.setattr(wf, '_verify_debugfix_loop', verify_mock)
    monkeypatch.setattr(wf, '_review', review_mock)
    monkeypatch.setattr(wf, '_suggestions_in_scope', lambda s: [])
    monkeypatch.setattr(wf, '_write_suggestions_to_memory', AsyncMock())
    return verify_mock, review_mock


def _rev_parse_head(repo: Path) -> str:
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _dispatch(tmp_path: Path, run_id: str, db_path: Path, worktree: Path) -> tuple[TaskWorkflow, EventStore]:
    """One dispatch of the task: a fresh workflow and a new run on the shared *db_path*, at *worktree*."""
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    store = EventStore(db_path, run_id=run_id)
    wf = _make_e2e_workflow(run_dir, store)
    wf.worktree = worktree
    return wf, store


def _workflow_verify_rows(store: EventStore) -> list[dict[str, Any]]:
    return [row['data'] for row in store.fetch_events_by_type(EventType.workflow_verify)]


def _verify_skips(store: EventStore) -> list[dict[str, Any]]:
    return [row['data'] for row in store.fetch_events_by_type(EventType.phase_skipped) if row['phase'] == 'verify']


def _tip_fields(data: dict[str, Any]) -> tuple[object, ...]:
    return (data['passed'], data['tip_sha'], data['tip_sha_unresolved'], data['tip_sha_unresolved_reason'])


class TestVerifyGreenCheckpointSkipsVerify:
    @pytest.mark.asyncio
    async def test_durable_green_at_tip_skips_verify_and_suppresses_reemit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = tmp_path / 'runs.db'

        # Pre-seed a DURABLE green under a PRIOR run at the current tip.
        prior = EventStore(db_path, run_id='run-prior')
        prior.emit(
            EventType.workflow_verify,
            task_id=TASK_ID,
            data={'passed': True, 'tip_sha': GREEN_TIP, 'base_sha': 'deadbeef'},
        )

        # Fresh dispatch: a NEW run on the same db (the cross-restart scenario).
        store_now = EventStore(db_path, run_id='run-now')
        wf = _make_e2e_workflow(tmp_path, store_now)

        # Current branch tip matches the durable green's tip → checkpoint hits.
        wf._get_head_commit = AsyncMock(return_value=GREEN_TIP)  # type: ignore[method-assign]
        verify, _ = _stub_verify_and_review(wf, monkeypatch)

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE

        # (1) verify was skipped — the debugfix loop was never awaited.
        verify.assert_not_awaited()

        # (2) the honest skip signal was emitted: phase_skipped(verify).
        skipped = store_now.fetch_events_by_type_all_runs(
            EventType.phase_skipped, task_id=TASK_ID
        )
        assert any(row['phase'] == 'verify' for row in skipped), skipped

        # (3) NO new workflow_verify row under the current run (suppressed) —
        #     the run-scoped reader on run-now sees none.
        assert store_now.fetch_events_by_type(EventType.workflow_verify) == []


class TestVerifyCheckpointTipFetchFailSafe:
    @pytest.mark.asyncio
    async def test_raising_get_head_commit_falls_through_to_normal_verify(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Regression pin (reviewer robustness finding, task 2752 amendment): a
        # HEAD read that RAISES at the checkpoint site (here a missing
        # worktree cwd → FileNotFoundError at spawn) must NOT crash
        # _execute_verify_review_loop.  It must fall through to the normal
        # verify path (the tip cannot be read → checkpoint miss →
        # _verify_debugfix_loop runs), exactly as the checkpoint comment
        # promises — never propagate out of the loop.
        db_path = tmp_path / 'runs.db'

        # A durable prior-run green DOES exist at GREEN_TIP, but it must be
        # ignored: the checkpoint can never read the current tip to compare.
        prior = EventStore(db_path, run_id='run-prior')
        prior.emit(
            EventType.workflow_verify,
            task_id=TASK_ID,
            data={'passed': True, 'tip_sha': GREEN_TIP, 'base_sha': 'deadbeef'},
        )

        store_now = EventStore(db_path, run_id='run-now')
        wf = _make_e2e_workflow(tmp_path, store_now)
        wf.worktree = tmp_path / 'missing-worktree'
        verify, _ = _stub_verify_and_review(wf, monkeypatch)

        # Must NOT raise despite the checkpoint tip fetch failing.
        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE

        # verify RAN (the checkpoint missed because the tip could not be read).
        verify.assert_awaited_once()

        # NO phase_skipped(verify) — nothing was skipped this cycle.
        skipped = store_now.fetch_events_by_type_all_runs(
            EventType.phase_skipped, task_id=TASK_ID
        )
        assert not any(row['phase'] == 'verify' for row in skipped), skipped

    @pytest.mark.asyncio
    async def test_raising_post_verify_tip_capture_degrades_to_no_checkpoint_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # As test_raising_get_head_commit_falls_through_to_normal_verify, but
        # the checkpoint read succeeds and the worktree is removed during
        # verify, so the post-verify capture is the HEAD read that raises.
        store_now = EventStore(tmp_path / 'runs.db', run_id='run-now')
        wf = _make_e2e_workflow(tmp_path, store_now)
        repo = seed_repo(tmp_path / 'repo')
        head_before_verify = _rev_parse_head(repo)
        wf.worktree = repo

        async def remove_repo() -> None:
            shutil.rmtree(repo)

        verify, review = _stub_verify_and_review(wf, monkeypatch, during_verify=remove_repo)

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE
        verify.assert_awaited_once()
        review.assert_awaited()
        [green] = _workflow_verify_rows(store_now)
        assert _tip_fields(green) == (True, None, True, 'head_unreadable')
        assert not green_checkpoint_at_tip(
            store_now, EventType.workflow_verify, TASK_ID, head_before_verify,
        )

    @pytest.mark.asyncio
    async def test_raising_tip_capture_without_event_store_does_not_crash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # As test_raising_get_head_commit_falls_through_to_normal_verify, with no
        # event store: the post-verify capture is the only HEAD read.
        wf = _make_e2e_workflow(tmp_path, None)
        wf.worktree = tmp_path / 'missing-worktree'
        verify, review = _stub_verify_and_review(wf, monkeypatch)

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE
        verify.assert_awaited_once()
        review.assert_awaited()


class TestRedispatchBoundary:
    @pytest.mark.asyncio
    async def test_redispatch_at_a_recorded_green_tip_skips_verify(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = tmp_path / 'runs.db'
        repo = seed_repo(tmp_path / 'repo')
        head = _rev_parse_head(repo)

        first, first_store = _dispatch(tmp_path, 'run-1', db_path, repo)
        first_verify, _ = _stub_verify_and_review(first, monkeypatch)
        assert await first._execute_verify_review_loop() == WorkflowOutcome.DONE
        first_verify.assert_awaited_once()
        [green] = _workflow_verify_rows(first_store)
        assert _tip_fields(green) == (True, head, False, None)

        second, second_store = _dispatch(tmp_path, 'run-2', db_path, repo)
        second_verify, _ = _stub_verify_and_review(second, monkeypatch)
        assert await second._execute_verify_review_loop() == WorkflowOutcome.DONE

        second_verify.assert_not_awaited()
        assert [skip['tip_sha'] for skip in _verify_skips(second_store)] == [head]

    @pytest.mark.asyncio
    async def test_redispatch_after_an_unresolved_tip_green_reverifies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING)
        db_path = tmp_path / 'runs.db'
        repo = seed_repo(tmp_path / 'repo')
        branch_ref = subprocess.run(
            ['git', 'symbolic-ref', 'HEAD'], cwd=repo, check=True, capture_output=True, text=True,
        ).stdout.strip()

        first, first_store = _dispatch(tmp_path, 'run-1', db_path, repo)
        _stub_verify_and_review(first, monkeypatch, during_verify=lambda: make_head_unreadable(repo))
        assert await first._execute_verify_review_loop() == WorkflowOutcome.DONE

        [green] = _workflow_verify_rows(first_store)
        assert _tip_fields(green) == (True, None, True, 'head_unreadable')
        assert any(
            record.levelno == logging.WARNING
            and TASK_ID in record.getMessage()
            and 'tip_sha_unresolved' in record.getMessage()
            for record in caplog.records
        ), caplog.text

        subprocess.run(['git', 'symbolic-ref', 'HEAD', branch_ref], cwd=repo, check=True)
        second, second_store = _dispatch(tmp_path, 'run-2', db_path, repo)
        second_verify, _ = _stub_verify_and_review(second, monkeypatch)
        assert await second._execute_verify_review_loop() == WorkflowOutcome.DONE

        second_verify.assert_awaited_once()
        assert _verify_skips(second_store) == []

    @pytest.mark.asyncio
    async def test_post_verify_tip_read_retries_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = EventStore(tmp_path / 'runs.db', run_id='run-now')
        wf = _make_e2e_workflow(tmp_path, store)
        # No real fixture fails exactly once: checkpoint read, failed first capture, successful retry.
        monkeypatch.setattr(wf, '_get_head_commit', AsyncMock(side_effect=[GREEN_TIP, '', GREEN_TIP]))
        _stub_verify_and_review(wf, monkeypatch)

        assert await wf._execute_verify_review_loop() == WorkflowOutcome.DONE

        [green] = _workflow_verify_rows(store)
        assert _tip_fields(green) == (True, GREEN_TIP, False, None)


class TestPreMergeReverifyTip:
    @pytest.mark.asyncio
    async def test_an_unreadable_head_records_why_and_resolves_no_base(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = EventStore(tmp_path / 'runs.db', run_id='run-now')
        wf = _make_e2e_workflow(tmp_path, store)
        repo = seed_repo(tmp_path / 'repo')
        await make_head_unreadable(repo)
        wf.worktree = repo
        git_ops = MagicMock()
        git_ops.task_verify_lease.return_value = contextlib.nullcontext()
        git_ops.lane_in_use.return_value = contextlib.nullcontext()
        git_ops.merge_base_with_main = AsyncMock(return_value='b' * 40)
        wf.git_ops = git_ops
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification',
            AsyncMock(return_value=VerifyResult(
                passed=True, test_output='', lint_output='', type_output='', summary='ok',
            )),
        )

        assert await wf._post_rebase_reverify_passed() is True

        [row] = _workflow_verify_rows(store)
        assert (row['site'], row['tip_sha'], row['tip_sha_unresolved'], row['base_sha']) == (
            'pre_merge', None, True, None,
        )
        git_ops.merge_base_with_main.assert_not_awaited()
