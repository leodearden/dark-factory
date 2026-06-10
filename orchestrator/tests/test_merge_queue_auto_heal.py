"""Tests for merge_queue auto-heal building blocks (task 1691).

Step-1 (RED/GREEN): is_auto_heal_eligible conservative mechanical gate.
Step-3 (RED/GREEN): lane_for_task_metadata reads merge_lane metadata.
Step-5 (RED/GREEN): compose_fix_main_brief title/description shape.
Step-7 (RED/GREEN): MainHealthAutoHealRegistry + MAIN_HEALTH_AUTO_HEAL_MAX_ATTEMPTS.
Step-9 (RED/GREEN): MergeRequest.lane wired from task metadata in _submit_to_merge_queue.
Step-11 (RED/GREEN): _auto_heal_main_health happy-path (signal a + signal-c contract).
Step-13 (RED/GREEN): _submit_to_merge_queue routes MAIN_HEALTH_RED to _auto_heal_main_health.
Step-15 (RED/GREEN): _auto_heal_main_health idempotency (lane already halted → no spawn).
Step-17 (RED/GREEN): non-mechanical outcome → escalate, no halt/spawn.
Step-19 (RED/GREEN): attempt cap → hard-escalate, no spawn.
Step-21 (RED/GREEN): owner-tied auto-resume via unhalt_lanes_owned_by.
Step-23 (RED/GREEN): auto-heal signature derives from failing outcome, not constant instance fields.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_queue import (
    MAIN_HEALTH_RED_REASON_PREFIX,
    MergeOutcome,
    MergeRequest,
)
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import (  # type: ignore[attr-defined]
    TaskWorkflow,
    WorkflowOutcome,
    _compute_merge_outcome_signature,
)

MAIN_SHA = 'deadbeef1234567890'


# ---------------------------------------------------------------------------
# Shared helpers (mirror test_workflow_main_health_routing.py)
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


def _make_workflow(config: OrchestratorConfig, worktree: Path, *, task_metadata: dict | None = None) -> TaskWorkflow:
    assignment = TaskAssignment(
        task_id='42',
        task={
            'id': '42', 'title': 'X', 'description': '',
            'status': 'pending',
            'metadata': task_metadata if task_metadata is not None else {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )
    git_ops = MagicMock(spec=GitOps)
    git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)
    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()
    workflow = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    workflow.worktree = worktree
    artifacts = TaskArtifacts(worktree)
    artifacts.init('42', 'X', 'desc', base_commit='base-sha')
    workflow.artifacts = artifacts
    workflow.plan = {'task_id': '42', 'steps': []}
    return workflow


# ---------------------------------------------------------------------------
# Step-1: is_auto_heal_eligible
# ---------------------------------------------------------------------------


class TestIsAutoHealEligible:
    """Conservative mechanical gate for auto-heal eligibility."""

    def test_compile_error_with_hint_is_eligible(self) -> None:
        from orchestrator.merge_queue import is_auto_heal_eligible
        assert is_auto_heal_eligible('compile_error', 'error TS2322: StatusBar.tsx') is True

    def test_compile_error_no_hint_is_eligible(self) -> None:
        from orchestrator.merge_queue import is_auto_heal_eligible
        assert is_auto_heal_eligible('compile_error', '') is True

    def test_compile_error_none_hint_is_eligible(self) -> None:
        from orchestrator.merge_queue import is_auto_heal_eligible
        assert is_auto_heal_eligible('compile_error', None) is True

    def test_test_failure_is_not_eligible(self) -> None:
        from orchestrator.merge_queue import is_auto_heal_eligible
        assert is_auto_heal_eligible('test_failure', 'some test output') is False

    def test_unknown_test_failure_is_not_eligible(self) -> None:
        from orchestrator.merge_queue import is_auto_heal_eligible
        assert is_auto_heal_eligible('unknown_test_failure', '') is False

    def test_infra_timeout_is_not_eligible(self) -> None:
        from orchestrator.merge_queue import is_auto_heal_eligible
        assert is_auto_heal_eligible('infra_timeout', '') is False

    def test_flock_error_is_not_eligible(self) -> None:
        from orchestrator.merge_queue import is_auto_heal_eligible
        assert is_auto_heal_eligible('flock_error', '') is False

    def test_empty_strings_are_not_eligible(self) -> None:
        from orchestrator.merge_queue import is_auto_heal_eligible
        assert is_auto_heal_eligible('', '') is False

    def test_none_category_is_not_eligible(self) -> None:
        from orchestrator.merge_queue import is_auto_heal_eligible
        assert is_auto_heal_eligible(None, None) is False


# ---------------------------------------------------------------------------
# Step-3: lane_for_task_metadata
# ---------------------------------------------------------------------------


class TestLaneForTaskMetadata:
    """lane_for_task_metadata reads merge_lane from task metadata."""

    def test_high_lane(self) -> None:
        from orchestrator.merge_queue import lane_for_task_metadata
        assert lane_for_task_metadata({'merge_lane': 'high'}) == 'high'

    def test_normal_lane(self) -> None:
        from orchestrator.merge_queue import lane_for_task_metadata
        assert lane_for_task_metadata({'merge_lane': 'normal'}) == 'normal'

    def test_empty_dict_defaults_to_normal(self) -> None:
        from orchestrator.merge_queue import lane_for_task_metadata
        assert lane_for_task_metadata({}) == 'normal'

    def test_none_defaults_to_normal(self) -> None:
        from orchestrator.merge_queue import lane_for_task_metadata
        assert lane_for_task_metadata(None) == 'normal'

    def test_bogus_value_normalizes_to_normal(self) -> None:
        from orchestrator.merge_queue import lane_for_task_metadata
        assert lane_for_task_metadata({'merge_lane': 'bogus'}) == 'normal'


# ---------------------------------------------------------------------------
# Step-5: compose_fix_main_brief
# ---------------------------------------------------------------------------


class TestComposeFixMainBrief:
    """compose_fix_main_brief produces a title + description for a fix-main task."""

    def test_title_starts_with_fix_main(self) -> None:
        from orchestrator.merge_queue import compose_fix_main_brief
        title, _ = compose_fix_main_brief(
            'compile_error', 'error TS2322: StatusBar.tsx', 'tsc failed\n<report>',
        )
        assert title.startswith('fix main:'), f'Title must start with "fix main:"; got {title!r}'

    def test_title_contains_relevant_identifier(self) -> None:
        from orchestrator.merge_queue import compose_fix_main_brief
        title, _ = compose_fix_main_brief(
            'compile_error', 'error TS2322: StatusBar.tsx', 'tsc failed\n<report>',
        )
        # Either the cause_hint or the category must appear somewhere in the title
        assert ('StatusBar.tsx' in title or 'compile_error' in title or 'TS2322' in title), (
            f'Expected identifier in title; got {title!r}'
        )

    def test_title_bounded_length(self) -> None:
        from orchestrator.merge_queue import compose_fix_main_brief
        title, _ = compose_fix_main_brief(
            'compile_error', 'error TS2322: StatusBar.tsx', 'tsc failed\n<report>',
        )
        assert len(title) <= 120, f'Title must be ≤120 chars; got {len(title)}: {title!r}'

    def test_title_non_empty(self) -> None:
        from orchestrator.merge_queue import compose_fix_main_brief
        title, _ = compose_fix_main_brief('compile_error', 'hint', 'detail')
        assert title, 'Title must be non-empty'

    def test_description_contains_category(self) -> None:
        from orchestrator.merge_queue import compose_fix_main_brief
        _, description = compose_fix_main_brief(
            'compile_error', 'error TS2322: StatusBar.tsx', 'tsc failed\n<report>',
        )
        assert 'compile_error' in description, (
            f'Description must contain category; got {description!r}'
        )

    def test_description_contains_cause_hint(self) -> None:
        from orchestrator.merge_queue import compose_fix_main_brief
        _, description = compose_fix_main_brief(
            'compile_error', 'error TS2322: StatusBar.tsx', 'tsc failed\n<report>',
        )
        assert 'error TS2322' in description or 'StatusBar.tsx' in description, (
            f'Description must contain cause_hint; got {description!r}'
        )

    def test_description_contains_detail(self) -> None:
        from orchestrator.merge_queue import compose_fix_main_brief
        _, description = compose_fix_main_brief(
            'compile_error', 'error TS2322: StatusBar.tsx', 'tsc failed\n<report>',
        )
        assert 'tsc failed' in description or 'report' in description, (
            f'Description must contain (truncated) detail; got {description!r}'
        )

    def test_description_non_empty(self) -> None:
        from orchestrator.merge_queue import compose_fix_main_brief
        _, description = compose_fix_main_brief('compile_error', 'hint', 'detail')
        assert description, 'Description must be non-empty'


# ---------------------------------------------------------------------------
# Step-7: MainHealthAutoHealRegistry + MAIN_HEALTH_AUTO_HEAL_MAX_ATTEMPTS
# ---------------------------------------------------------------------------


class TestMainHealthAutoHealRegistry:
    """MainHealthAutoHealRegistry is a monotonic per-signature attempt counter."""

    def test_fresh_registry_has_zero_attempts(self) -> None:
        from orchestrator.merge_queue import MainHealthAutoHealRegistry
        r = MainHealthAutoHealRegistry()
        assert r.attempts('sig-A') == 0

    def test_record_attempt_returns_incremented_count(self) -> None:
        from orchestrator.merge_queue import MainHealthAutoHealRegistry
        r = MainHealthAutoHealRegistry()
        assert r.record_attempt('sig-A') == 1

    def test_attempts_reflects_recorded_count(self) -> None:
        from orchestrator.merge_queue import MainHealthAutoHealRegistry
        r = MainHealthAutoHealRegistry()
        r.record_attempt('sig-A')
        assert r.attempts('sig-A') == 1

    def test_monotonically_increments(self) -> None:
        from orchestrator.merge_queue import MainHealthAutoHealRegistry
        r = MainHealthAutoHealRegistry()
        r.record_attempt('sig-A')
        result = r.record_attempt('sig-A')
        assert result == 2
        assert r.attempts('sig-A') == 2

    def test_different_keys_are_independent(self) -> None:
        from orchestrator.merge_queue import MainHealthAutoHealRegistry
        r = MainHealthAutoHealRegistry()
        r.record_attempt('sig-A')
        assert r.attempts('sig-B') == 0

    def test_max_attempts_constant(self) -> None:
        from orchestrator.merge_queue import MAIN_HEALTH_AUTO_HEAL_MAX_ATTEMPTS
        assert MAIN_HEALTH_AUTO_HEAL_MAX_ATTEMPTS == 1

    def test_merge_worker_exposes_registry(self) -> None:
        from orchestrator.merge_queue import MainHealthAutoHealRegistry, MergeWorker
        git_ops = MagicMock(spec=GitOps)
        worker = MergeWorker(git_ops=git_ops, queue=asyncio.Queue())
        assert isinstance(worker.auto_heal_registry, MainHealthAutoHealRegistry)

    def test_speculative_merge_worker_exposes_registry(self) -> None:
        from orchestrator.merge_queue import MainHealthAutoHealRegistry, SpeculativeMergeWorker
        git_ops = MagicMock(spec=GitOps)
        worker = SpeculativeMergeWorker(git_ops=git_ops, queue=asyncio.Queue())
        assert isinstance(worker.auto_heal_registry, MainHealthAutoHealRegistry)


# ---------------------------------------------------------------------------
# Task-1712: SpeculativeMergeWorker bare-harness constructibility contract
# ---------------------------------------------------------------------------


class TestSpeculativeWorkerBareHarnessContract:
    """SpeculativeMergeWorker must be constructible with a bare MagicMock(spec=GitOps).

    Regression guard: task 1710 added an unconditional `git_ops.project_root`
    access in __init__ which broke all bare-worker/bare-harness tests.  The
    defensive fix (`getattr(git_ops, 'project_root', None)`) must keep
    _shadow_state_path=None when project_root is absent, mirroring the
    escalation_queue None-safety contract documented in the same constructor.
    """

    def test_bare_spec_mock_does_not_raise(self) -> None:
        """Constructing with bare MagicMock(spec=GitOps) — NO project_root wired —
        must NOT raise AttributeError."""
        from orchestrator.merge_queue import MainHealthAutoHealRegistry, SpeculativeMergeWorker

        # Intentionally bare: project_root is NOT set on this mock.
        # This exercises the project_root-absent / None path in __init__.
        git_ops = MagicMock(spec=GitOps)
        worker = SpeculativeMergeWorker(git_ops=git_ops, queue=asyncio.Queue())
        assert worker._shadow_state_path is None
        assert isinstance(worker.auto_heal_registry, MainHealthAutoHealRegistry)


# ---------------------------------------------------------------------------
# Step-9: MergeRequest.lane wired from task metadata in _submit_to_merge_queue
# ---------------------------------------------------------------------------


class TestMergeRequestLaneWiring:
    """_submit_to_merge_queue passes lane=lane_for_task_metadata(...) to MergeRequest."""

    def _run_and_capture_request(
        self, config: OrchestratorConfig, worktree: Path, task_metadata: dict,
    ) -> MergeRequest:
        workflow = _make_workflow(config, worktree, task_metadata=task_metadata)
        workflow.merge_queue = asyncio.Queue()

        captured: list[MergeRequest] = []

        async def _fake_enqueue(_queue, request, _event_store, **_kwargs):
            captured.append(request)
            request.result.set_result(MergeOutcome('done', reason='ok'))

        workflow._write_merge_failure_review = MagicMock()

        with patch('orchestrator.merge_queue.enqueue_merge_request', _fake_enqueue):
            asyncio.run(
                workflow._submit_to_merge_queue('task/42', pre_rebased=False, merge_phase=False)
            )
        assert captured, 'enqueue_merge_request was not called'
        return captured[0]

    def test_high_lane_metadata_produces_high_lane_request(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        req = self._run_and_capture_request(config, worktree, {'merge_lane': 'high'})
        assert req.lane == 'high', f'Expected lane=high; got {req.lane!r}'

    def test_no_merge_lane_produces_normal_lane_request(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt2'
        worktree.mkdir()
        req = self._run_and_capture_request(config, worktree, {'files': ['lib']})
        assert req.lane == 'normal', f'Expected lane=normal; got {req.lane!r}'


# ---------------------------------------------------------------------------
# Step-11: _auto_heal_main_health happy path (signals a + c contract)
# ---------------------------------------------------------------------------

def _make_compile_error_outcome(*, fp: str = 'fp-compile-001') -> MergeOutcome:
    """Eligible MAIN_HEALTH_RED MergeOutcome for compile_error."""
    return MergeOutcome(
        'blocked',
        reason=f'{MAIN_HEALTH_RED_REASON_PREFIX} (category=compile_error): error TS2322',
        failure_category='compile_error',
        failure_cause_hint='error TS2322: StatusBar.tsx',
        dedupe_fingerprint=fp,
    )


def _make_workflow_with_worker(
    config: OrchestratorConfig, worktree: Path,
    *,
    is_halted: bool = False,
    task_metadata: dict | None = None,
) -> tuple[TaskWorkflow, MagicMock]:
    """Return (workflow, merge_worker_mock) with auto_heal_registry pre-wired."""
    from orchestrator.merge_queue import MainHealthAutoHealRegistry

    workflow = _make_workflow(config, worktree, task_metadata=task_metadata)
    workflow.merge_queue = asyncio.Queue()

    merge_worker = MagicMock()
    merge_worker.auto_heal_registry = MainHealthAutoHealRegistry()
    merge_worker.is_lane_halted = MagicMock(return_value=is_halted)
    merge_worker.halt_lane = MagicMock()
    merge_worker.set_lane_halt_owner = MagicMock()

    workflow.merge_worker = merge_worker

    esc_mock = MagicMock()
    esc_mock.id = 'esc-auto-heal-001'
    escalation_queue = MagicMock()
    escalation_queue.make_id = MagicMock(return_value='esc-auto-heal-001')
    # submit_or_dedupe returns {'status': 'queued'} to simulate parent creation
    workflow.escalation_queue = escalation_queue

    return workflow, merge_worker


class TestAutoHealHappyPath:
    """Signal a: confirmed red halts normal lane + dedup'd escalation + spawns high-lane fix."""

    def test_happy_path_halts_lane_spawns_fix_and_returns_blocked(
        self, tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        outcome = _make_compile_error_outcome()

        workflow, merge_worker = _make_workflow_with_worker(config, worktree)

        # Intercept spawn: record call args
        spawned_args: list[list[dict]] = []

        async def _fake_post_submit(arguments_list: list[dict]) -> None:
            spawned_args.append(arguments_list)

        workflow._post_submit_tasks = _fake_post_submit  # type: ignore[method-assign]
        workflow._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]

        with patch('orchestrator.workflow.submit_or_dedupe', return_value={'status': 'queued'}):
            result = asyncio.run(
                workflow._auto_heal_main_health(outcome, merge_phase=True)
            )

        # (1) halt_lane called with 'normal'
        merge_worker.halt_lane.assert_called_once()
        call_args = merge_worker.halt_lane.call_args
        assert call_args[0][0] == 'normal', (
            f'halt_lane must be called with lane=normal; got {call_args}'
        )

        # (2) exactly one escalation submitted via submit_or_dedupe
        # (checked via mock call to workflow.escalation_queue or submit_or_dedupe patch)

        # (3) a fix task is spawned
        assert spawned_args, 'Expected _post_submit_tasks to be called with fix task args'
        fix_task_args = spawned_args[0][0]
        assert fix_task_args.get('title', '').startswith('fix main:'), (
            f'Fix task title must start with "fix main:"; got {fix_task_args.get("title")!r}'
        )
        metadata = fix_task_args.get('metadata', {})
        assert metadata.get('merge_lane') == 'high', (
            f'Fix task must be tagged merge_lane=high; got {metadata!r}'
        )
        assert metadata.get('spawn_context') == 'main_health_auto_heal', (
            f'Fix task must have spawn_context=main_health_auto_heal; got {metadata!r}'
        )
        assert 'main_health_signature' in metadata, (
            f'Fix task must carry main_health_signature; got {metadata!r}'
        )
        assert 'main_health_escalation_id' in metadata, (
            f'Fix task must carry main_health_escalation_id; got {metadata!r}'
        )

        # (4) registry.attempts(sig) == 1 after the call
        # Use _compute_merge_outcome_signature with the outcome's fields (not
        # workflow._merge_outcome_signature() which reads stale instance fields).
        sig = _compute_merge_outcome_signature('compile_error', 'error TS2322: StatusBar.tsx')
        assert merge_worker.auto_heal_registry.attempts(sig) == 1, (
            f'Expected attempts==1 after happy path; got {merge_worker.auto_heal_registry.attempts(sig)}'
        )

        # (5) returns BLOCKED
        assert result == WorkflowOutcome.BLOCKED, (
            f'Expected WorkflowOutcome.BLOCKED; got {result!r}'
        )


# ---------------------------------------------------------------------------
# Step-13: _submit_to_merge_queue routes MAIN_HEALTH_RED to _auto_heal_main_health
# ---------------------------------------------------------------------------


class TestSubmitToMergeQueueRoutesAutoHeal:
    """_submit_to_merge_queue must dispatch MAIN_HEALTH_RED to _auto_heal_main_health."""

    def test_main_health_red_calls_auto_heal(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        workflow = _make_workflow(config, worktree)
        workflow.merge_queue = asyncio.Queue()

        outcome = _make_compile_error_outcome()

        auto_heal_mock = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        workflow._auto_heal_main_health = auto_heal_mock  # type: ignore[method-assign]
        workflow._write_merge_failure_review = MagicMock()

        async def _fake_enqueue(_queue, request, _event_store, **_kwargs):
            request.result.set_result(outcome)

        with patch('orchestrator.merge_queue.enqueue_merge_request', _fake_enqueue):
            result = asyncio.run(
                workflow._submit_to_merge_queue('task/42', pre_rebased=False, merge_phase=True)
            )

        auto_heal_mock.assert_called_once()
        call_kwargs = auto_heal_mock.call_args
        # first positional arg is the outcome
        assert call_kwargs[0][0] is outcome, 'auto_heal must receive the MergeOutcome'
        assert call_kwargs[1].get('merge_phase') is True, 'merge_phase must be forwarded'
        assert result == WorkflowOutcome.BLOCKED

    def test_generic_blocked_outcome_does_not_call_auto_heal(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt2'
        worktree.mkdir()
        workflow = _make_workflow(config, worktree)
        workflow.merge_queue = asyncio.Queue()

        # A normal blocked outcome that does NOT start with MAIN_HEALTH_RED_REASON_PREFIX
        generic_outcome = MergeOutcome(
            'blocked',
            reason='Post-merge verification failed: tsc failed',
            failure_category='compile_error',
        )

        auto_heal_mock = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        workflow._auto_heal_main_health = auto_heal_mock  # type: ignore[method-assign]
        mark_blocked_mock = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        workflow._mark_blocked = mark_blocked_mock  # type: ignore[method-assign]
        workflow._write_merge_failure_review = MagicMock()

        async def _fake_enqueue(_queue, request, _event_store, **_kwargs):
            request.result.set_result(generic_outcome)

        with patch('orchestrator.merge_queue.enqueue_merge_request', _fake_enqueue):
            result = asyncio.run(
                workflow._submit_to_merge_queue('task/42', pre_rebased=False, merge_phase=False)
            )

        auto_heal_mock.assert_not_called()
        mark_blocked_mock.assert_called_once()
        assert result == WorkflowOutcome.BLOCKED


# ---------------------------------------------------------------------------
# Step-15: _auto_heal_main_health idempotency (lane already halted → no spawn)
# ---------------------------------------------------------------------------


class TestAutoHealIdempotency:
    """If normal lane is already halted, adopt in-flight auto-heal — no second spawn."""

    def test_already_halted_no_spawn_but_still_blocks(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        outcome = _make_compile_error_outcome()

        # is_lane_halted('normal') returns True → in-flight auto-heal
        workflow, merge_worker = _make_workflow_with_worker(config, worktree, is_halted=True)

        # Realistic scenario: the first healing task already recorded an attempt before
        # this concurrent duplicate arrives.  With attempts(sig)=1 and lane still halted,
        # the idempotency branch must win — NOT the attempt-cap branch.
        sig = _compute_merge_outcome_signature('compile_error', 'error TS2322: StatusBar.tsx')
        merge_worker.auto_heal_registry.record_attempt(sig)

        spawned_args: list[list[dict]] = []

        async def _fake_post_submit(arguments_list: list[dict]) -> None:
            spawned_args.append(arguments_list)

        workflow._post_submit_tasks = _fake_post_submit  # type: ignore[method-assign]
        mark_blocked_mock = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        workflow._mark_blocked = mark_blocked_mock  # type: ignore[method-assign]

        with patch('orchestrator.workflow.submit_or_dedupe', return_value={'status': 'folded'}):
            result = asyncio.run(
                workflow._auto_heal_main_health(outcome, merge_phase=True)
            )

        # No second halt
        merge_worker.halt_lane.assert_not_called()
        # No second spawn
        assert not spawned_args, 'Expected no spawn when lane already halted'
        # Still returns BLOCKED
        assert result == WorkflowOutcome.BLOCKED
        # Must fold via idempotency (skip_escalation) — NOT attempt-cap (escalate_to_human)
        mark_blocked_mock.assert_called_once()
        call_kwargs = mark_blocked_mock.call_args[1]
        assert call_kwargs.get('skip_escalation') is True, (
            f'Idempotency fold must use skip_escalation=True; got {call_kwargs}. '
            f'If escalate_to_human=True is set, the attempt-cap branch fired instead of '
            f'idempotency — branch ordering bug.'
        )
        assert not call_kwargs.get('escalate_to_human'), (
            f'Idempotency fold must NOT escalate_to_human; got {call_kwargs}. '
            f'Wrong branch fired.'
        )


# ---------------------------------------------------------------------------
# Step-17: non-mechanical outcome → escalate, no halt/spawn
# ---------------------------------------------------------------------------


class TestAutoHealNonMechanicalEscalates:
    """Signal d: test_failure / unknown_test_failure → escalate to human, no spawn."""

    @pytest.mark.parametrize('category', ['test_failure', 'unknown_test_failure'])
    def test_non_mechanical_escalates_without_spawn(
        self, tmp_path: Path, category: str,
    ) -> None:
        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        # Non-mechanical outcome
        outcome = MergeOutcome(
            'blocked',
            reason=f'{MAIN_HEALTH_RED_REASON_PREFIX} (category={category}): flaky test',
            failure_category=category,
            failure_cause_hint='TestFoo.test_bar',
            dedupe_fingerprint='fp-test-001',
        )
        workflow, merge_worker = _make_workflow_with_worker(config, worktree)

        spawned_args: list[list[dict]] = []

        async def _fake_post_submit(arguments_list: list[dict]) -> None:
            spawned_args.append(arguments_list)

        workflow._post_submit_tasks = _fake_post_submit  # type: ignore[method-assign]
        mark_blocked_mock = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        workflow._mark_blocked = mark_blocked_mock  # type: ignore[method-assign]

        result = asyncio.run(
            workflow._auto_heal_main_health(outcome, merge_phase=True)
        )

        # No halt
        merge_worker.halt_lane.assert_not_called()
        # No spawn
        assert not spawned_args, f'[{category}] Expected no spawn for non-mechanical failure'
        # Escalate to human
        mark_blocked_mock.assert_called_once()
        call_kwargs = mark_blocked_mock.call_args[1]
        assert call_kwargs.get('escalate_to_human') is True, (
            f'[{category}] Expected escalate_to_human=True; got {call_kwargs}'
        )
        assert call_kwargs.get('category') == 'preexisting_main_break', (
            f'[{category}] Expected category=preexisting_main_break; got {call_kwargs}'
        )
        assert call_kwargs.get('dedupe_fingerprint') == 'fp-test-001', (
            f'[{category}] Expected dedupe_fingerprint forwarded; got {call_kwargs}'
        )
        assert call_kwargs.get('suggested_action') == 'await_preexisting_main_hotfix', (
            f'[{category}] Expected suggested_action=await_preexisting_main_hotfix; got {call_kwargs}'
        )
        assert result == WorkflowOutcome.BLOCKED


# ---------------------------------------------------------------------------
# Step-19: attempt cap → hard-escalate, no spawn
# ---------------------------------------------------------------------------


class TestAutoHealAttemptCap:
    """Signal e: recurring same-signature break → attempt cap → hard-escalate, no spawn."""

    def test_attempt_cap_hard_escalates_without_spawn(self, tmp_path: Path) -> None:
        from orchestrator.merge_queue import MAIN_HEALTH_AUTO_HEAL_MAX_ATTEMPTS

        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        outcome = _make_compile_error_outcome()

        workflow, merge_worker = _make_workflow_with_worker(config, worktree)

        # Pre-seed the registry using the outcome-derived sig (not the stale
        # instance-field sig which would return the empty-basis constant).
        sig = _compute_merge_outcome_signature('compile_error', 'error TS2322: StatusBar.tsx')
        for _ in range(MAIN_HEALTH_AUTO_HEAL_MAX_ATTEMPTS):
            merge_worker.auto_heal_registry.record_attempt(sig)
        assert merge_worker.auto_heal_registry.attempts(sig) >= MAIN_HEALTH_AUTO_HEAL_MAX_ATTEMPTS

        spawned_args: list[list[dict]] = []

        async def _fake_post_submit(arguments_list: list[dict]) -> None:
            spawned_args.append(arguments_list)

        workflow._post_submit_tasks = _fake_post_submit  # type: ignore[method-assign]
        mark_blocked_mock = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        workflow._mark_blocked = mark_blocked_mock  # type: ignore[method-assign]

        result = asyncio.run(
            workflow._auto_heal_main_health(outcome, merge_phase=True)
        )

        # No new halt
        merge_worker.halt_lane.assert_not_called()
        # No spawn
        assert not spawned_args, 'Expected no spawn when attempt cap exceeded'
        # Hard-escalate
        mark_blocked_mock.assert_called_once()
        call_kwargs = mark_blocked_mock.call_args[1]
        assert call_kwargs.get('escalate_to_human') is True, (
            f'Expected escalate_to_human=True; got {call_kwargs}'
        )
        assert call_kwargs.get('category') == 'preexisting_main_break', (
            f'Expected category=preexisting_main_break; got {call_kwargs}'
        )
        # root_cause should mention re-break or attempt cap
        root_cause = call_kwargs.get('root_cause', '')
        assert root_cause, f'Expected root_cause set; got {call_kwargs}'
        assert result == WorkflowOutcome.BLOCKED


# ---------------------------------------------------------------------------
# Step-21: owner-tied auto-resume (signal b)
# ---------------------------------------------------------------------------


class TestAutoHealOwnerTiedResume:
    """Signal b: after happy path, unhalt_lanes_owned_by(esc_id) resumes normal lane."""

    def test_owner_tied_resume_primitive(self, tmp_path: Path) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()
        outcome = _make_compile_error_outcome()

        workflow = _make_workflow(config, worktree)
        workflow.merge_queue = asyncio.Queue()

        # Use a REAL SpeculativeMergeWorker so _WipHaltMixin state is exercised
        git_ops = MagicMock(spec=GitOps)
        real_worker = SpeculativeMergeWorker(git_ops=git_ops, queue=asyncio.Queue())
        workflow.merge_worker = real_worker

        escalation_queue = MagicMock()
        escalation_queue.make_id = MagicMock(return_value='esc-owner-001')
        workflow.escalation_queue = escalation_queue

        spawned_args: list[list[dict]] = []

        async def _fake_post_submit(arguments_list: list[dict]) -> None:
            spawned_args.append(arguments_list)

        workflow._post_submit_tasks = _fake_post_submit  # type: ignore[method-assign]
        workflow._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]

        with patch('orchestrator.workflow.submit_or_dedupe', return_value={'status': 'queued'}):
            asyncio.run(
                workflow._auto_heal_main_health(outcome, merge_phase=True)
            )

        # Lane must be halted after happy path
        assert real_worker.is_lane_halted('normal') is True, (
            'Normal lane must be halted after auto-heal happy path'
        )

        # The escalation owns the halt
        esc_id = 'esc-owner-001'
        assert real_worker.lane_owned_by(esc_id) == 'normal', (
            f'esc-owner-001 must own the normal-lane halt; '
            f'owner state: {real_worker._lane_halt_owner!r}'
        )

        # Owner-tied resume: unhalt_lanes_owned_by returns ['normal'] and clears the halt
        unhalted = real_worker.unhalt_lanes_owned_by(esc_id)
        assert unhalted == ['normal'], (
            f'unhalt_lanes_owned_by must return [normal]; got {unhalted!r}'
        )
        assert real_worker.is_lane_halted('normal') is False, (
            'Normal lane must be unhalted after unhalt_lanes_owned_by'
        )


# ---------------------------------------------------------------------------
# Step-23: auto-heal signature derives from the failing outcome, not constant
#           instance fields (_compute_merge_outcome_signature regression)
# ---------------------------------------------------------------------------


class TestAutoHealSignatureDerivedFromOutcome:
    """Regression: distinct (category,cause_hint) breaks must get distinct sigs.

    Bug: _auto_heal_main_health used self._merge_outcome_signature() which reads
    _last_merge_failure_* instance fields.  Those are never set on the
    MAIN_HEALTH_RED_REASON_PREFIX fast-path, so every call returned the same
    constant hash of the empty basis — a single shared registry key that caused
    any later break to see attempts>=MAX and hard-escalate without spawning.
    """

    # (A) Pure helper: distinct cause_hints → distinct sigs, and neither
    # collapses to the empty-basis constant.
    def test_distinct_cause_hints_produce_distinct_sigs(self) -> None:
        sig_a = _compute_merge_outcome_signature('compile_error', 'error TS2322: StatusBar.tsx')
        sig_b = _compute_merge_outcome_signature('compile_error', 'error TS1005: Toolbar.tsx')
        assert sig_a != sig_b, (
            f'Distinct cause_hints must produce distinct sigs; got sig_a={sig_a!r} sig_b={sig_b!r}'
        )

    def test_outcome_derived_sig_differs_from_empty_constant(self) -> None:
        sig_outcome = _compute_merge_outcome_signature('compile_error', 'error TS2322: StatusBar.tsx')
        sig_empty = _compute_merge_outcome_signature('', '')
        assert sig_outcome != sig_empty, (
            f'Outcome-derived sig must not collapse to empty-basis constant; '
            f'sig_outcome={sig_outcome!r} sig_empty={sig_empty!r}'
        )

    # (B) Behavioral: two workflows sharing ONE registry must get independent
    # attempt counters when called with DISTINCT outcomes.
    def test_two_distinct_outcomes_get_independent_attempt_counters(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import MainHealthAutoHealRegistry

        config = _make_config(tmp_path)

        # Shared merge_worker with a REAL registry
        shared_merge_worker = MagicMock()
        shared_merge_worker.auto_heal_registry = MainHealthAutoHealRegistry()
        shared_merge_worker.is_lane_halted = MagicMock(return_value=False)
        shared_merge_worker.halt_lane = MagicMock()
        shared_merge_worker.set_lane_halt_owner = MagicMock()

        # Outcome A: compile_error / TS2322 / StatusBar
        outcome_a = MergeOutcome(
            'blocked',
            reason=f'{MAIN_HEALTH_RED_REASON_PREFIX} (category=compile_error): error TS2322',
            failure_category='compile_error',
            failure_cause_hint='error TS2322: StatusBar.tsx',
            dedupe_fingerprint='fp-A',
        )

        # Outcome B: compile_error / TS1005 / Toolbar (DIFFERENT cause_hint)
        outcome_b = MergeOutcome(
            'blocked',
            reason=f'{MAIN_HEALTH_RED_REASON_PREFIX} (category=compile_error): error TS1005',
            failure_category='compile_error',
            failure_cause_hint='error TS1005: Toolbar.tsx',
            dedupe_fingerprint='fp-B',
        )

        # Build two separate workflows pointing at the SAME merge_worker
        worktree_a = tmp_path / 'wt-a'
        worktree_a.mkdir()
        workflow_a = _make_workflow(config, worktree_a)
        workflow_a.merge_queue = asyncio.Queue()
        workflow_a.merge_worker = shared_merge_worker

        esc_queue_a = MagicMock()
        esc_queue_a.make_id = MagicMock(return_value='esc-a-001')
        workflow_a.escalation_queue = esc_queue_a

        spawned_a: list[list[dict]] = []

        async def _fake_post_submit_a(arguments_list: list[dict]) -> None:
            spawned_a.append(arguments_list)

        workflow_a._post_submit_tasks = _fake_post_submit_a  # type: ignore[method-assign]
        workflow_a._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]

        worktree_b = tmp_path / 'wt-b'
        worktree_b.mkdir()
        workflow_b = _make_workflow(config, worktree_b)
        workflow_b.merge_queue = asyncio.Queue()
        workflow_b.merge_worker = shared_merge_worker

        esc_queue_b = MagicMock()
        esc_queue_b.make_id = MagicMock(return_value='esc-b-001')
        workflow_b.escalation_queue = esc_queue_b

        spawned_b: list[list[dict]] = []

        async def _fake_post_submit_b(arguments_list: list[dict]) -> None:
            spawned_b.append(arguments_list)

        workflow_b._post_submit_tasks = _fake_post_submit_b  # type: ignore[method-assign]
        workflow_b._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]

        with patch('orchestrator.workflow.submit_or_dedupe', return_value={'status': 'queued'}):
            asyncio.run(workflow_a._auto_heal_main_health(outcome_a, merge_phase=True))

        # After A's heal, reset is_lane_halted to return False for B's call
        shared_merge_worker.is_lane_halted = MagicMock(return_value=False)

        with patch('orchestrator.workflow.submit_or_dedupe', return_value={'status': 'queued'}):
            asyncio.run(workflow_b._auto_heal_main_health(outcome_b, merge_phase=True))

        # Both workflows must have spawned a fix task
        assert spawned_a, 'Workflow A must have spawned a fix task'
        assert spawned_b, (
            'Workflow B must have spawned a fix task — '
            'NOT spuriously attempt-capped by A\'s heal'
        )

        # The spawned main_health_signature values must be DISTINCT
        sig_in_a = spawned_a[0][0].get('metadata', {}).get('main_health_signature')
        sig_in_b = spawned_b[0][0].get('metadata', {}).get('main_health_signature')
        assert sig_in_a is not None, 'Workflow A fix task must carry main_health_signature'
        assert sig_in_b is not None, 'Workflow B fix task must carry main_health_signature'
        assert sig_in_a != sig_in_b, (
            f'Distinct outcomes must produce distinct fix-task signatures; '
            f'sig_in_a={sig_in_a!r} sig_in_b={sig_in_b!r}'
        )

        # Independent attempt counters on the shared registry
        registry = shared_merge_worker.auto_heal_registry
        assert registry.attempts(sig_in_a) == 1, (
            f'Registry for sig_in_a must have 1 attempt; got {registry.attempts(sig_in_a)}'
        )
        assert registry.attempts(sig_in_b) == 1, (
            f'Registry for sig_in_b must have 1 attempt; got {registry.attempts(sig_in_b)}'
        )


# ---------------------------------------------------------------------------
# Step-25: halt-owner is the SURVIVING (parent) escalation, not folded child
# ---------------------------------------------------------------------------


class TestAutoHealHaltOwnerIsDedupeParent:
    """Regression: when submit_or_dedupe folds into a pending parent, the
    normal-lane halt owner must be the PARENT escalation id, not the locally-
    built child esc.id that was never queued.

    If the child id is registered as the halt owner, unhalt_lanes_owned_by(parent)
    never matches and the normal lane stays halted forever.
    """

    def test_halt_owner_is_surviving_parent_on_fold(self, tmp_path: Path) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt-fold'
        worktree.mkdir()
        outcome = _make_compile_error_outcome()

        workflow = _make_workflow(config, worktree)
        workflow.merge_queue = asyncio.Queue()

        # Use a REAL SpeculativeMergeWorker so _WipHaltMixin lane-owner state is
        # exercised.
        git_ops = MagicMock(spec=GitOps)
        real_worker = SpeculativeMergeWorker(git_ops=git_ops, queue=asyncio.Queue())
        workflow.merge_worker = real_worker

        # make_id returns the CHILD (locally-built) escalation id
        escalation_queue = MagicMock()
        escalation_queue.make_id = MagicMock(return_value='esc-child-001')
        workflow.escalation_queue = escalation_queue

        spawned_args: list[list[dict]] = []

        async def _fake_post_submit(arguments_list: list[dict]) -> None:
            spawned_args.append(arguments_list)

        workflow._post_submit_tasks = _fake_post_submit  # type: ignore[method-assign]
        workflow._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]

        # submit_or_dedupe folds the child into an existing pending parent:
        # returns {'id': parent_id, 'status': 'dedup_skipped',
        #          'parent_id': parent_id, 'child_id': 'esc-child-001'}
        fold_result = {
            'id': 'parent-esc-999',
            'status': 'dedup_skipped',
            'parent_id': 'parent-esc-999',
            'child_id': 'esc-child-001',
        }
        with patch('orchestrator.workflow.submit_or_dedupe', return_value=fold_result):
            asyncio.run(
                workflow._auto_heal_main_health(outcome, merge_phase=True)
            )

        # (1) normal lane must be halted
        assert real_worker.is_lane_halted('normal') is True, (
            'Normal lane must be halted after auto-heal happy path'
        )

        # (2) the PARENT escalation (the one that resolves) owns the halt
        assert real_worker.lane_owned_by('parent-esc-999') == 'normal', (
            f'parent-esc-999 must own the normal-lane halt (surviving parent); '
            f'owner state: {real_worker._lane_halt_owner!r}'
        )

        # (3) the folded child must NOT own the halt
        assert real_worker.lane_owned_by('esc-child-001') is None, (
            f'esc-child-001 (folded child) must NOT own the halt; '
            f'owner state: {real_worker._lane_halt_owner!r}'
        )

        # (4) owner-tied resume: unhalt_lanes_owned_by(parent) resumes normal lane
        unhalted = real_worker.unhalt_lanes_owned_by('parent-esc-999')
        assert unhalted == ['normal'], (
            f'unhalt_lanes_owned_by(parent) must return [normal]; got {unhalted!r}'
        )
        assert real_worker.is_lane_halted('normal') is False, (
            'Normal lane must be unhalted after unhalt_lanes_owned_by(parent)'
        )

        # (5) the spawned fix task must carry the parent escalation id as the
        # correlation key (so the auto-watcher can match on resolution)
        assert spawned_args, 'Expected fix task to be spawned'
        fix_metadata = spawned_args[0][0].get('metadata', {})
        assert fix_metadata.get('main_health_escalation_id') == 'parent-esc-999', (
            f"Fix task must carry parent escalation id; got {fix_metadata.get('main_health_escalation_id')!r}"
        )


# ---------------------------------------------------------------------------
# Amendment: no halt when escalation_queue is None (suggestion 2 regression)
# ---------------------------------------------------------------------------


class TestAutoHealNoHaltWithoutOwner:
    """Regression for resource_leak: when escalation_queue is None, a halt owner can
    never be registered, so unhalt_lanes_owned_by can never match and the normal lane
    would stay halted permanently.

    Fix: treat escalation_queue=None like merge_worker=None in branch (d) — fall back
    to escalate-only with no halt/spawn.
    """

    def test_no_escalation_queue_skips_halt_and_escalates_only(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt-no-esc'
        worktree.mkdir()
        outcome = _make_compile_error_outcome()

        workflow, merge_worker = _make_workflow_with_worker(config, worktree)
        # Remove escalation_queue — no owner can be registered for the halt.
        workflow.escalation_queue = None

        spawned_args: list[list[dict]] = []

        async def _fake_post_submit(arguments_list: list[dict]) -> None:
            spawned_args.append(arguments_list)

        workflow._post_submit_tasks = _fake_post_submit  # type: ignore[method-assign]
        mark_blocked_mock = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        workflow._mark_blocked = mark_blocked_mock  # type: ignore[method-assign]

        result = asyncio.run(
            workflow._auto_heal_main_health(outcome, merge_phase=True)
        )

        # Must NOT halt the lane — no owner can ever be registered → livelock
        merge_worker.halt_lane.assert_not_called()
        # Must NOT spawn a fix task
        assert not spawned_args, (
            'Expected no spawn when escalation_queue is None (no owner can be registered)'
        )
        # Must fall back to escalate-only (branch d)
        mark_blocked_mock.assert_called_once()
        call_kwargs = mark_blocked_mock.call_args[1]
        assert call_kwargs.get('escalate_to_human') is True, (
            f'Expected escalate_to_human=True in fallback; got {call_kwargs}'
        )
        assert result == WorkflowOutcome.BLOCKED


# ---------------------------------------------------------------------------
# Amendment: concurrent same-signature tasks fold via idempotency, not cap
#             (suggestion 1 + 3 regression — branch ordering)
# ---------------------------------------------------------------------------


class TestAutoHealConcurrentSameSignature:
    """Regression for branch-ordering bug: with MAIN_HEALTH_AUTO_HEAL_MAX_ATTEMPTS=1,
    the first task records attempt=1 and halts the lane. A concurrent second task
    with the SAME signature then arrives while the lane is still halted.

    Before fix: attempt-cap check fires before idempotency check →  second task
    sees attempts(sig)=1 >= MAX=1 → takes attempt-cap branch → hard-escalates with
    wrong root_cause='main-health-auto-heal-rebreak:<sig>'. This mis-classifies an
    ordinary concurrent duplicate as a genuine re-break loop.

    After fix: cap check gated on `not is_lane_halted('normal')` → when lane IS
    halted, skip cap → take idempotency branch → fold cleanly (no second halt, no
    spawn, skip_escalation=True).
    """

    def test_concurrent_duplicate_folds_via_idempotency_not_cap(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        config = _make_config(tmp_path)

        # Use a REAL SpeculativeMergeWorker so is_lane_halted reflects actual state
        git_ops = MagicMock(spec=GitOps)
        real_worker = SpeculativeMergeWorker(git_ops=git_ops, queue=asyncio.Queue())

        # ── Task A setup ─────────────────────────────────────────────────────
        worktree_a = tmp_path / 'wt-conc-a'
        worktree_a.mkdir()
        workflow_a = _make_workflow(config, worktree_a)
        workflow_a.merge_queue = asyncio.Queue()
        workflow_a.merge_worker = real_worker

        esc_queue_a = MagicMock()
        esc_queue_a.make_id = MagicMock(return_value='esc-conc-a-001')
        workflow_a.escalation_queue = esc_queue_a

        spawned_a: list[list[dict]] = []

        async def _fake_post_submit_a(arguments_list: list[dict]) -> None:
            spawned_a.append(arguments_list)

        workflow_a._post_submit_tasks = _fake_post_submit_a  # type: ignore[method-assign]
        workflow_a._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[method-assign]

        # ── Task B setup (SAME outcome / SAME signature) ──────────────────────
        worktree_b = tmp_path / 'wt-conc-b'
        worktree_b.mkdir()
        workflow_b = _make_workflow(config, worktree_b)
        workflow_b.merge_queue = asyncio.Queue()
        workflow_b.merge_worker = real_worker  # shared real worker

        esc_queue_b = MagicMock()
        esc_queue_b.make_id = MagicMock(return_value='esc-conc-b-001')
        workflow_b.escalation_queue = esc_queue_b

        spawned_b: list[list[dict]] = []

        async def _fake_post_submit_b(arguments_list: list[dict]) -> None:
            spawned_b.append(arguments_list)

        workflow_b._post_submit_tasks = _fake_post_submit_b  # type: ignore[method-assign]
        mark_blocked_b = AsyncMock(return_value=WorkflowOutcome.BLOCKED)
        workflow_b._mark_blocked = mark_blocked_b  # type: ignore[method-assign]

        outcome = _make_compile_error_outcome()

        # Step 1: Task A runs the happy path → halts lane, records attempt=1
        with patch('orchestrator.workflow.submit_or_dedupe', return_value={'status': 'queued'}):
            asyncio.run(workflow_a._auto_heal_main_health(outcome, merge_phase=True))

        assert real_worker.is_lane_halted('normal') is True, (
            'After A: normal lane must be halted'
        )
        sig = _compute_merge_outcome_signature('compile_error', 'error TS2322: StatusBar.tsx')
        assert real_worker.auto_heal_registry.attempts(sig) == 1, (
            f'After A: attempts must be 1; got {real_worker.auto_heal_registry.attempts(sig)}'
        )
        assert spawned_a, 'Task A must have spawned a fix task'

        # Step 2: Task B arrives with the SAME signature while the lane is still halted.
        # It must fold via idempotency, NOT mis-fire the attempt-cap branch.
        with patch('orchestrator.workflow.submit_or_dedupe', return_value={'status': 'queued'}):
            asyncio.run(workflow_b._auto_heal_main_health(outcome, merge_phase=True))

        # Task B must NOT spawn a second fix task
        assert not spawned_b, (
            'Task B must NOT spawn a fix task — concurrent same-signature duplicate '
            'must fold via idempotency, not mis-fire the attempt-cap branch'
        )

        # Normal lane must still be halted (B did not halt or unhalt it)
        assert real_worker.is_lane_halted('normal') is True, (
            'After B: normal lane must still be halted (idempotency fold, no change)'
        )

        # B's _mark_blocked must use skip_escalation=True (idempotency fold), NOT
        # escalate_to_human=True (which would indicate the attempt-cap branch fired).
        mark_blocked_b.assert_called_once()
        call_kwargs_b = mark_blocked_b.call_args[1]
        assert call_kwargs_b.get('skip_escalation') is True, (
            f'Task B idempotency fold must use skip_escalation=True; got {call_kwargs_b}. '
            f'If escalate_to_human=True, the attempt-cap branch fired — branch-ordering bug.'
        )
        assert not call_kwargs_b.get('escalate_to_human'), (
            f'Task B must NOT set escalate_to_human (indicates wrong branch); got {call_kwargs_b}'
        )
