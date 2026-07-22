"""Tests for the fail-closed sandbox-dispatch guard on TaskWorkflow (task 2908).

Covers the two workflow-side pieces of the OS-sandbox β1 fail-closed guard
(PRD plans/os-sandbox-worktree-containment-prd.md D4 / invariant INV-4):

- ``_escalate_sandbox_unavailable``: files a single blocking ``infra_issue``
  escalation mirroring the sibling ``_escalate_plan_overwrite`` shape.
- ``_guard_sandbox``: calls ``resolve_backend_or_refuse()``, files a DEDUPED
  escalation on ``SandboxUnavailable`` (only when ``exc.should_escalate``),
  and re-raises to refuse the invocation; the 'none' escape hatch and a healthy
  backend both return without raising or escalating.

Bare-workflow construction (``object.__new__(TaskWorkflow)`` + only the
attributes these methods touch) mirrors the pattern in
test_workflow_completion_memory.py — the full __init__ needs a config/git_ops/
scheduler/briefing/mcp we do not exercise here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from escalation.queue import EscalationQueue
from unittest.mock import patch

from orchestrator.agents import sandbox_dispatch
from orchestrator.workflow import TaskWorkflow
from orchestrator.workflow_types import WorkflowState, WorkflowStateMachine


@pytest.fixture(autouse=True)
def _reset_sandbox_state():
    """Snapshot/restore the process-global backend + re-arm the dedup per test."""
    saved = sandbox_dispatch.get_backend()
    sandbox_dispatch.reset_escalation_dedupe()
    yield
    sandbox_dispatch.set_backend(saved)
    sandbox_dispatch.reset_escalation_dedupe()


def _make_bare_workflow(
    tmp_path: Path,
    *,
    with_queue: bool = True,
    worktree: Path | None = None,
    task_id: str = '2908',
) -> TaskWorkflow:
    """Build a TaskWorkflow with only the attributes the guard methods touch."""
    wf = object.__new__(TaskWorkflow)
    wf.task_id = task_id
    wf.worktree = worktree
    wf.machine = WorkflowStateMachine(WorkflowState.EXECUTE)
    wf.event_store = None
    wf.escalation_queue = (
        EscalationQueue(tmp_path / 'queue') if with_queue else None
    )
    return wf


class TestEscalateSandboxUnavailable:
    """_escalate_sandbox_unavailable files the blocking infra_issue escalation."""

    def test_submits_one_escalation_matching_sibling_shape(self, tmp_path):
        wf = _make_bare_workflow(tmp_path, worktree=tmp_path)

        wf._escalate_sandbox_unavailable(
            'implementer', 'landlock', 'refusing to run unconfined',
        )

        assert wf.escalation_queue is not None
        pending = wf.escalation_queue.get_pending()
        assert len(pending) == 1
        esc = pending[0]
        assert esc.agent_role == 'orchestrator'
        assert esc.severity == 'blocking'
        assert esc.category == 'infra_issue'
        assert esc.task_id == '2908'
        # summary/detail name the role and backend state.
        assert 'implementer' in esc.summary
        assert 'landlock' in esc.detail
        # worktree carried through when present.
        assert esc.worktree == str(tmp_path)

    def test_no_queue_is_a_noop(self, tmp_path):
        wf = _make_bare_workflow(tmp_path, with_queue=False)
        # Must not raise when there is no escalation queue wired up.
        wf._escalate_sandbox_unavailable('debugger', 'auto', 'detail')


class TestGuardSandbox:
    """_guard_sandbox refuses fail-closed and files a single deduped escalation."""

    def test_refuse_files_single_escalation_across_n(self, tmp_path):
        """N refusals raise SandboxUnavailable each; the queue holds exactly one (INV-4)."""
        wf = _make_bare_workflow(tmp_path, worktree=tmp_path)
        sandbox_dispatch.set_backend('landlock')
        with patch(
            'orchestrator.agents.landlock.is_landlock_available',
            return_value=False,
        ):
            for _ in range(3):
                with pytest.raises(sandbox_dispatch.SandboxUnavailable):
                    wf._guard_sandbox('implementer')

        assert wf.escalation_queue is not None
        pending = wf.escalation_queue.get_pending()
        assert len(pending) == 1, (
            f'exactly one escalation expected across N refusals; got {len(pending)}'
        )
        assert 'implementer' in pending[0].summary
        assert 'landlock' in pending[0].detail

    def test_none_escape_returns_without_escalation(self, tmp_path, caplog):
        """backend='none' returns None (no raise), files nothing, logs a WARNING."""
        wf = _make_bare_workflow(tmp_path, worktree=tmp_path)
        sandbox_dispatch.set_backend('none')
        with caplog.at_level(
            logging.WARNING, logger='orchestrator.agents.sandbox_dispatch',
        ):
            assert wf._guard_sandbox('implementer') is None

        assert wf.escalation_queue is not None
        assert wf.escalation_queue.get_pending() == []
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_healthy_backend_returns_without_escalation(self, tmp_path):
        """A resolvable backend returns None, no raise, files nothing."""
        wf = _make_bare_workflow(tmp_path, worktree=tmp_path)
        sandbox_dispatch.set_backend('landlock')
        with patch(
            'orchestrator.agents.landlock.is_landlock_available',
            return_value=True,
        ):
            assert wf._guard_sandbox('implementer') is None

        assert wf.escalation_queue is not None
        assert wf.escalation_queue.get_pending() == []
