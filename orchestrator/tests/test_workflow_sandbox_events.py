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

import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _workflow_helpers import FakeBriefing, FakeScheduler, _init_git_repo

from orchestrator.agents import sandbox_dispatch
from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.write_set import WriteSet
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import IMPLEMENTER, TaskWorkflow
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


class TestEnsureSandboxDirs:
    """_ensure_sandbox_dirs pre-creates the load-bearing claude_fleet carve-out
    OUTSIDE the sandbox before dispatch (task 2996). On success the dir exists
    for both backends to grant; on failure it WARNs LOUDLY with the task_id +
    fleet path (loud-over-silent) and continues — best-effort, never raising."""

    def test_creates_claude_fleet_dir(self, tmp_path):
        # (a) Happy path: the claude_fleet dir is materialized and no exception
        # escapes.
        wf = _make_bare_workflow()
        claude_fleet = tmp_path / '.claude' / 'fleet'
        write_set = _make_write_set(claude_fleet=claude_fleet)
        assert not claude_fleet.exists()

        wf._ensure_sandbox_dirs(write_set)

        assert claude_fleet.is_dir()

    def test_failure_warns_loudly_with_task_id_and_path(self, tmp_path, caplog):
        # (b) Failure path: a regular FILE cannot be a parent dir, so mkdir
        # raises OSError and the helper returns False. _ensure_sandbox_dirs must
        # NOT raise (best-effort) and must emit a WARNING naming BOTH the task_id
        # (_make_bare_workflow's default '2909') and the fleet path.
        wf = _make_bare_workflow()
        blocker = tmp_path / 'blocker'
        blocker.write_text('i am a regular file, not a directory\n')
        claude_fleet = blocker / 'fleet'
        write_set = _make_write_set(claude_fleet=claude_fleet)

        with caplog.at_level(logging.WARNING):
            wf._ensure_sandbox_dirs(write_set)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, 'expected a loud WARNING when pre-creation fails'
        rendered = ' '.join(r.getMessage() for r in warnings)
        assert '2909' in rendered
        assert str(claude_fleet) in rendered


@pytest.mark.asyncio
class TestInvokeWiresEnsureSandboxDirs:
    """Pins the CALL-SITE wiring inside ``TaskWorkflow._invoke``: for a
    sandboxed role the claude_fleet carve-out is pre-created AFTER
    ``compute_write_set`` and BEFORE ``write_set.writable_paths()`` is consumed
    into ``sandbox_extras`` (task 2996).

    The isolated ``TestEnsureSandboxDirs`` unit tests above cover the helper's
    behaviour; this drives the REAL ``_invoke`` sandbox branch end-to-end so a
    future refactor that drops the call — or reorders it AFTER the
    ``writable_paths()`` consumption — fails loudly here (that ordering is the
    behaviour that actually matters: the dir must exist before the sandbox is
    built). Mirrors the full-``_invoke`` harness in
    test_workflow_status_on_resume.py: a real worktree + a fully-constructed
    TaskWorkflow with ``invoke_with_cap_retry`` mocked away.
    """

    async def test_claude_fleet_exists_before_sandbox_extras_built(
        self, tmp_path, monkeypatch,
    ):
        # A real git repo + linked worktree — the _invoke prologue needs
        # config/git_ops/scheduler/briefing, and cwd must be a worktree.
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_git_repo(repo)
        config = OrchestratorConfig(
            project_root=repo,
            max_concurrent_tasks=1,
            git=GitConfig(
                main_branch='main',
                branch_prefix='task/',
                remote='origin',
                worktree_dir='.worktrees',
            ),
        )
        git_ops = GitOps(config.git, config.project_root)
        wt_info = await git_ops.create_worktree('42')
        cwd = wt_info.path

        assignment = TaskAssignment(
            task_id='42',
            task={
                'id': '42',
                'title': 'X',
                'description': 'Y',
                'status': 'pending',
                'metadata': {'files': ['README']},
                'dependencies': [],
            },
            modules=['README'],
        )
        workflow = TaskWorkflow(
            assignment=assignment,
            config=config,
            git_ops=git_ops,
            scheduler=FakeScheduler(),  # type: ignore[arg-type]
            briefing=FakeBriefing(),  # type: ignore[arg-type]
            mcp=None,
        )
        workflow.artifacts = None  # skip write_agent_session

        # Hermetic write-set: ONLY claude_fleet is materialized on disk, under
        # tmp_path — NEVER the real ~/.claude. Patch compute_write_set so the
        # sandbox branch consumes THIS set (the other, dummy paths are only
        # stringified into the mocked-away sandbox_extras).
        claude_fleet = tmp_path / '.claude' / 'fleet'
        hermetic = _make_write_set(claude_fleet=claude_fleet)
        assert not claude_fleet.exists()
        monkeypatch.setattr(
            'orchestrator.workflow.compute_write_set', lambda _cwd: hermetic,
        )

        # Spy: record whether claude_fleet already exists on disk at the moment
        # writable_paths() is consumed into sandbox_extras. If pre-creation is
        # dropped or reordered AFTER this consumption, the FIRST observation is
        # False and the assertion below fails.
        observations: list[bool] = []
        orig_writable_paths = WriteSet.writable_paths

        def _recording_writable_paths(self):
            observations.append(self.claude_fleet.is_dir())
            return orig_writable_paths(self)

        monkeypatch.setattr(WriteSet, 'writable_paths', _recording_writable_paths)

        # Pin the backend to the 'none' escape hatch so _guard_sandbox returns
        # without raising, independent of host landlock/bwrap availability.
        saved_backend = sandbox_dispatch.get_backend()
        sandbox_dispatch.set_backend('none')
        try:
            with patch(
                'orchestrator.workflow.invoke_with_cap_retry',
                new_callable=AsyncMock,
                return_value=AgentResult(success=True, output=''),
            ) as mock_cap_retry:
                await workflow._invoke(IMPLEMENTER, 'PROMPT', cwd)
        finally:
            sandbox_dispatch.set_backend(saved_backend)

        # The sandboxed dispatch actually reached invoke_with_cap_retry (the
        # sandbox branch ran to completion).
        assert mock_cap_retry.await_count == 1
        # sandbox_extras was built from write_set.writable_paths()...
        assert observations, (
            'sandbox_extras must be built from write_set.writable_paths()'
        )
        # ...and claude_fleet ALREADY existed on disk at that first consumption,
        # i.e. _ensure_sandbox_dirs ran before it — the wiring under test.
        assert observations[0] is True, (
            'claude_fleet must be pre-created BEFORE sandbox_extras is built: '
            '_ensure_sandbox_dirs is wired after compute_write_set and before '
            'the write_set.writable_paths() consumption'
        )
        # And the dir is actually materialized on disk after dispatch.
        assert claude_fleet.is_dir()
