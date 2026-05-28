"""Tests for the verify-loop signature-repetition guard in ``_verify_debugfix_loop``.

After ``max_failure_signature_repeat`` consecutive verify failures whose
(category, normalised cause_hint) tuple is identical the loop must
short-circuit to ``WorkflowOutcome.BLOCKED`` rather than burning the
remaining ``max_verify_attempts`` budget.

``_normalize_cause_hint`` strips file:line numeric tails, ANSI colour escape
sequences, collapses whitespace, and lowercases the hint so superficial
textual variation (shifting line numbers, colour codes) does not defeat the
equality check.

Two test classes:
  TestNormalizeCauseHint  — pure-unit tests for the normaliser helper.
  TestSignatureRepeatLoopGuard — integration tests driving
    ``_verify_debugfix_loop`` with injected VerifyResult sequences, mirroring
    the test_workflow_verify_retry.py fixture structure exactly.
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
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome, _normalize_cause_hint

# ---------------------------------------------------------------------------
# Unit tests for _normalize_cause_hint
# ---------------------------------------------------------------------------

class TestNormalizeCauseHint:
    """Pure-unit tests for the _normalize_cause_hint helper."""

    def test_file_line_suffix_stripped_makes_two_hints_equal(self):
        """file:line tails are stripped so two hints differing only in line
        numbers normalise to the same string."""
        h1 = 'FAILED tests/test_x.py:42 — AssertionError'
        h2 = 'FAILED tests/test_x.py:99 — AssertionError'
        assert _normalize_cause_hint(h1) == _normalize_cause_hint(h2)

    def test_double_colon_line_col_stripped(self):
        """file:line:col form (double colon) is also stripped."""
        h1 = 'foo.py:42:7 some error'
        h2 = 'foo.py:99:3 some error'
        assert _normalize_cause_hint(h1) == _normalize_cause_hint(h2)

    def test_ansi_colour_escapes_stripped(self):
        """\x1b[31m...\x1b[0m ANSI codes are stripped before comparison."""
        raw = '\x1b[31mFAILED\x1b[0m'
        assert _normalize_cause_hint(raw) == 'failed'

    def test_contiguous_whitespace_collapses(self):
        """Spaces, tabs, and newlines all collapse to a single space."""
        raw = 'some\t  error\n  detail'
        assert _normalize_cause_hint(raw) == 'some error detail'

    def test_result_is_lowercased_and_trimmed(self):
        """Output is lowercase and stripped of leading/trailing whitespace."""
        raw = '  ASSERTION ERROR  '
        result = _normalize_cause_hint(raw)
        assert result == result.lower()
        assert result == result.strip()
        assert result == 'assertion error'

    def test_empty_string_returns_empty_string(self):
        """Empty string input returns empty string (no exception)."""
        assert _normalize_cause_hint('') == ''

    def test_none_returns_empty_string(self):
        """None input returns empty string (no exception)."""
        assert _normalize_cause_hint(None) == ''


# ---------------------------------------------------------------------------
# Fixtures and helpers for TestSignatureRepeatLoopGuard
# (mirrors test_workflow_verify_retry.py structure exactly)
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
        max_concurrent_tasks=1,
        # Generous attempt cap so the test asserts the signature guard,
        # not the global cap.
        max_verify_attempts=5,
        max_failure_signature_repeat=3,
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
    workflow.plan = {'task_id': '42', 'steps': []}
    workflow._check_escalations = MagicMock(return_value=[])  # type: ignore[method-assign]
    workflow.briefing.build_debugger_prompt = AsyncMock(return_value='debug')  # type: ignore[attr-defined]

    from orchestrator.agents.invoke import AgentResult

    workflow._invoke = AsyncMock(  # type: ignore[method-assign]
        return_value=AgentResult(success=True, output=''),
    )
    workflow._get_head_commit = AsyncMock(return_value='head-sha')  # type: ignore[method-assign]
    return workflow, artifacts


def _test_failure(
    category: str,
    cause_hint: str,
    *,
    summary: str = 'Failures',
) -> VerifyResult:
    """Factory for VerifyResult test failures with specific (category, cause_hint)."""
    return VerifyResult(
        passed=False,
        test_output='FAILED\n',
        lint_output='',
        type_output='',
        summary=summary,
        timed_out=False,
        cause_hint=cause_hint,
        category=category,
    )


# ---------------------------------------------------------------------------
# Integration tests for the loop-guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSignatureRepeatLoopGuard:
    """Integration tests driving _verify_debugfix_loop with mocked VerifyResults."""

    async def test_failure_signature_history_initialises_empty(
        self, config, git_ops, task_assignment,
    ):
        """_failure_signature_history must exist as an empty list on a fresh workflow."""
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, _ = _make_workflow(config, git_ops, task_assignment, wt)
        assert hasattr(workflow, '_failure_signature_history')
        assert workflow._failure_signature_history == []

    async def test_three_identical_signatures_triggers_blocked(
        self, config, git_ops, task_assignment, monkeypatch, caplog,
    ):
        """T1: three identical (category, cause_hint) results → BLOCKED at attempt 3.

        verify_mock.await_count must be exactly 3 (not 5 = max_verify_attempts).
        A WARNING containing 'consecutive identical verify failures' must appear
        with the task id and the signature tuple.
        """
        import logging
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, _ = _make_workflow(config, git_ops, task_assignment, wt)
        workflow._inter_iteration_rebase = AsyncMock(return_value=None)  # type: ignore[method-assign]

        failure = _test_failure(
            category='test_failure',
            cause_hint='FAILED tests/test_x.py::test_y',
        )
        verify_mock = AsyncMock(side_effect=[failure, failure, failure, failure, failure])
        monkeypatch.setattr('orchestrator.workflow.run_scoped_verification', verify_mock)

        with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'):
            outcome = await workflow._verify_debugfix_loop()

        assert outcome == WorkflowOutcome.BLOCKED
        assert verify_mock.await_count == 3, (
            f'expected guard at attempt 3, got {verify_mock.await_count}'
        )
        # WARNING must mention consecutive identical verify failures
        warning_text = ' '.join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
        assert 'consecutive identical verify failures' in warning_text, (
            f'Expected warning about consecutive identical failures; caplog: {warning_text!r}'
        )
        assert '42' in warning_text  # task id

    async def test_line_number_variants_normalise_to_same_signature(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """T2: three failures differing only in line numbers → BLOCKED at attempt 3.

        Proves the normaliser is wired into the guard, not just defined in isolation.
        """
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, _ = _make_workflow(config, git_ops, task_assignment, wt)
        workflow._inter_iteration_rebase = AsyncMock(return_value=None)  # type: ignore[method-assign]

        failures = [
            _test_failure('test_failure', 'FAILED tests/test_x.py:42 AssertionError'),
            _test_failure('test_failure', 'FAILED tests/test_x.py:99 AssertionError'),
            _test_failure('test_failure', 'FAILED tests/test_x.py:155 AssertionError'),
        ]
        verify_mock = AsyncMock(side_effect=failures)
        monkeypatch.setattr('orchestrator.workflow.run_scoped_verification', verify_mock)

        outcome = await workflow._verify_debugfix_loop()

        assert outcome == WorkflowOutcome.BLOCKED
        assert verify_mock.await_count == 3, (
            f'expected guard at attempt 3 (normaliser wired), got {verify_mock.await_count}'
        )

    async def test_mixed_categories_exhaust_global_cap(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """T3: five failures with mixed signatures → BLOCKED at max_verify_attempts (5).

        The signature guard must NOT fire; only the global cap fires.
        """
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, _ = _make_workflow(config, git_ops, task_assignment, wt)
        workflow._inter_iteration_rebase = AsyncMock(return_value=None)  # type: ignore[method-assign]

        mixed = [
            _test_failure('test_failure', 'assert x == 1'),
            _test_failure('infra_issue', 'port 5432 unavailable'),
            _test_failure('test_failure', 'assert x == 1'),
            _test_failure('lint_failure', 'E501 line too long'),
            _test_failure('type_failure', 'incompatible types'),
        ]
        verify_mock = AsyncMock(side_effect=mixed)
        monkeypatch.setattr('orchestrator.workflow.run_scoped_verification', verify_mock)

        outcome = await workflow._verify_debugfix_loop()

        assert outcome == WorkflowOutcome.BLOCKED
        assert verify_mock.await_count == config.max_verify_attempts, (
            f'expected global cap at {config.max_verify_attempts}, got {verify_mock.await_count}'
        )

    async def test_blip_then_repeat_uses_last_n_equal_window(
        self, config, git_ops, task_assignment, monkeypatch,
    ):
        """T4: two identical then one different then two more identical → global cap only.

        Sequence: [A, A, B, A, A] with max_failure_signature_repeat=3.
        The last-3 window is [B, A, A] and then [A, A, ?] — never 3× A in a
        row at the tail.  The guard must NOT fire at attempt 3; the global cap
        (max_verify_attempts=5) fires at attempt 5 instead.
        """
        wt_info = await git_ops.create_worktree(task_assignment.task_id)
        wt = wt_info.path
        workflow, _ = _make_workflow(config, git_ops, task_assignment, wt)
        workflow._inter_iteration_rebase = AsyncMock(return_value=None)  # type: ignore[method-assign]

        sig_a = _test_failure('test_failure', 'assert x == 1')
        sig_b = _test_failure('infra_issue', 'port 5432 unavailable')
        failures = [sig_a, sig_a, sig_b, sig_a, sig_a]
        verify_mock = AsyncMock(side_effect=failures)
        monkeypatch.setattr('orchestrator.workflow.run_scoped_verification', verify_mock)

        outcome = await workflow._verify_debugfix_loop()

        assert outcome == WorkflowOutcome.BLOCKED
        assert verify_mock.await_count == config.max_verify_attempts, (
            f'expected global cap at {config.max_verify_attempts} '
            f'(last-N window, not consecutive-from-start), got {verify_mock.await_count}'
        )
