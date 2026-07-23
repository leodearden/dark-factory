"""Tests for the OS-sandbox β2 ``sandbox_applied`` wrap-path event emission.

``TaskWorkflow._emit_sandbox_applied`` (task 2909, PRD
plans/os-sandbox-worktree-containment-prd.md β2 §Goal 3 / INV-2) emits one
``sandbox_applied`` event per sandboxed invocation on the NON-refusing guard
path — a real backend (landlock/bwrap) OR the explicit ``backend=none`` escape
hatch — carrying the resolved backend and the stable ``WriteSet.digest()`` so
an operator can diff exactly what a given invocation could touch. Together with
``sandbox_unavailable`` (the refusing path, covered in
test_workflow_sandbox_refusal.py) every sandboxed invocation emits exactly one
of the two.

Bare-workflow construction (``object.__new__(TaskWorkflow)`` + only the
attributes the emit helper touches) mirrors test_workflow_sandbox_refusal.py —
the full ``__init__`` needs a config/git_ops/scheduler/briefing/mcp not
exercised here.
"""

from __future__ import annotations

from pathlib import Path

from orchestrator.agents.write_set import WriteSet
from orchestrator.event_store import EventStore, EventType
from orchestrator.workflow import TaskWorkflow
from orchestrator.workflow_types import WorkflowState, WorkflowStateMachine


def _make_bare_workflow(
    *, task_id: str = '2909', event_store: EventStore | None = None,
) -> TaskWorkflow:
    """Build a TaskWorkflow with only the attributes _emit_sandbox_applied touches."""
    wf = object.__new__(TaskWorkflow)
    wf.task_id = task_id
    wf.machine = WorkflowStateMachine(WorkflowState.EXECUTE)
    wf.event_store = event_store
    return wf


def _make_write_set(**overrides: Path) -> WriteSet:
    """A WriteSet from dummy Paths — digest() depends only on writable_paths()."""
    fields: dict[str, Path] = dict(
        worktree=Path('/wt'),
        task_meta=Path('/base/.task-meta/wt'),
        git_objects=Path('/main/.git/objects'),
        git_task_refs=Path('/main/.git/refs/heads/task'),
        git_task_reflogs=Path('/main/.git/logs/refs/heads/task'),
        git_worktree_admin=Path('/main/.git/worktrees/wt'),
        uv_cache=Path('/home/.cache/uv'),
        claude_fleet=Path('/home/.claude/fleet'),
        tmp=Path('/tmp'),
        dev=Path('/dev'),
    )
    fields.update(overrides)
    return WriteSet(**fields)


class TestEmitSandboxApplied:
    """_emit_sandbox_applied writes one sandbox_applied event per sandboxed
    invocation on the non-refusing guard path (β2)."""

    def test_real_backend_emits_one_row_with_backend_and_digest(self, tmp_path):
        store = EventStore(tmp_path / 'events.db', 'run')
        wf = _make_bare_workflow(event_store=store)
        write_set = _make_write_set()

        wf._emit_sandbox_applied('implementer', 'landlock', write_set)

        events = store.fetch_events_by_type(EventType.sandbox_applied)
        assert len(events) == 1
        ev = events[0]
        assert ev['data'] == {'backend': 'landlock', 'digest': write_set.digest()}
        assert ev['role'] == 'implementer'
        assert ev['task_id'] == '2909'
        assert ev['phase'] == 'execute'

    def test_none_escape_hatch_still_emits(self, tmp_path):
        # The explicit backend=none escape hatch is deliberately-unsandboxed but
        # STILL emits sandbox_applied (data.backend lets γ1 distinguish it from
        # real containment) — every sandboxed invocation emits one of the two.
        store = EventStore(tmp_path / 'events.db', 'run')
        wf = _make_bare_workflow(event_store=store)
        write_set = _make_write_set()

        wf._emit_sandbox_applied('debugger', 'none', write_set)

        events = store.fetch_events_by_type(EventType.sandbox_applied)
        assert len(events) == 1
        assert events[0]['data']['backend'] == 'none'
        assert events[0]['data']['digest'] == write_set.digest()
        assert events[0]['role'] == 'debugger'

    def test_no_event_store_is_a_noop(self, tmp_path):
        wf = _make_bare_workflow(event_store=None)
        write_set = _make_write_set()
        # Must not raise when there is no event store wired up.
        wf._emit_sandbox_applied('implementer', 'landlock', write_set)
