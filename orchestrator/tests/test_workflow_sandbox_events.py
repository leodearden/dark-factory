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

import dataclasses
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _workflow_helpers import FakeBriefing, FakeScheduler, _init_git_repo, fixed_prompt

from orchestrator.agents import sandbox_dispatch
from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.write_set import WriteSet
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import IMPLEMENTER, SIMPLE_TASK, TaskWorkflow
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


def _make_write_set(**overrides: Path | tuple[Path, ...]) -> WriteSet:
    """A WriteSet from dummy Paths — digest() depends only on writable_paths()."""
    base = WriteSet(
        worktree=Path('/wt'),
        task_meta=Path('/base/.task-meta/wt'),
        task_root=Path('/base/.task-meta/task-2909'),
        git_objects=Path('/main/.git/objects'),
        git_task_refs=Path('/main/.git/refs/heads/task'),
        git_task_reflogs=Path('/main/.git/logs/refs/heads/task'),
        git_worktree_admin=Path('/main/.git/worktrees/wt'),
        uv_cache=Path('/home/.cache/uv'),
        tree_sitter_cache=Path('/home/.cache/tree-sitter'),
        claude_fleet=Path('/home/.claude/fleet'),
        tmp=Path('/tmp'),
        dev=Path('/dev'),
    )
    return dataclasses.replace(base, **overrides)


def _docs_grant_write_set() -> WriteSet:
    return _make_write_set(
        docs_main_git=Path('/main/.git'),
        docs_main_paths=(Path('/main/docs/a.md'), Path('/main/CLAUDE.md')),
    )


def _sandbox_applied_events(
    tmp_path: Path, role: str, backend: str, write_set: WriteSet,
) -> list[dict]:
    """The sandbox_applied rows one ``_emit_sandbox_applied`` call stores."""
    store = EventStore(tmp_path / 'events.db', 'run')
    wf = _make_bare_workflow(event_store=store)
    wf._emit_sandbox_applied(role, backend, write_set)
    return store.fetch_events_by_type(EventType.sandbox_applied)


class TestEmitSandboxApplied:
    """_emit_sandbox_applied writes one sandbox_applied event per sandboxed
    invocation on the non-refusing guard path (β2)."""

    def test_real_backend_emits_one_row_with_backend_and_digest(self, tmp_path):
        write_set = _make_write_set()

        events = _sandbox_applied_events(tmp_path, 'implementer', 'landlock', write_set)

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
        write_set = _make_write_set()

        events = _sandbox_applied_events(tmp_path, 'debugger', 'none', write_set)

        assert len(events) == 1
        assert events[0]['data']['backend'] == 'none'
        assert events[0]['data']['digest'] == write_set.digest()
        assert events[0]['role'] == 'debugger'

    def test_docs_main_grant_adds_its_paths_to_the_payload(self, tmp_path):
        write_set = _docs_grant_write_set()

        events = _sandbox_applied_events(tmp_path, 'simple_task', 'landlock', write_set)

        assert len(events) == 1
        assert events[0]['data'] == {
            'backend': 'landlock',
            'digest': write_set.digest(),
            'docs_main_paths': ['/main/docs/a.md', '/main/CLAUDE.md'],
        }

    def test_no_event_store_is_a_noop(self, tmp_path):
        wf = _make_bare_workflow(event_store=None)
        write_set = _make_write_set()
        # Must not raise when there is no event store wired up.
        wf._emit_sandbox_applied('implementer', 'landlock', write_set)


def _blocking_file(tmp_path: Path) -> Path:
    """A regular FILE: mkdir of any path beneath it raises NotADirectoryError."""
    blocker = tmp_path / 'blocker'
    blocker.write_text('i am a regular file, not a directory\n')
    return blocker


def _warnings_during_ensure(
    wf: TaskWorkflow, write_set: WriteSet, caplog: pytest.LogCaptureFixture,
) -> list[str]:
    with caplog.at_level(logging.WARNING):
        wf._ensure_sandbox_dirs(write_set)
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


class TestEnsureSandboxDirs:
    """_ensure_sandbox_dirs pre-creates BOTH load-bearing carve-outs OUTSIDE the
    sandbox before dispatch: claude_fleet (task 2996) and tree_sitter_cache
    (task 3382). On success they exist for both backends to grant; for each
    path that fails it WARNs with the task_id and THAT path, and continues —
    best-effort, never raising.

    Every test overrides BOTH carve-outs under tmp_path: ``_make_write_set``
    defaults them to absolute dummies the wrapper would otherwise try to create.
    """

    def test_creates_both_carve_out_dirs(self, tmp_path):
        wf = _make_bare_workflow()
        claude_fleet = tmp_path / '.claude' / 'fleet'
        tree_sitter_cache = tmp_path / '.cache' / 'tree-sitter'
        write_set = _make_write_set(
            claude_fleet=claude_fleet, tree_sitter_cache=tree_sitter_cache,
        )
        assert not claude_fleet.exists()
        assert not tree_sitter_cache.exists()

        wf._ensure_sandbox_dirs(write_set)

        assert claude_fleet.is_dir()
        assert tree_sitter_cache.is_dir()

    def test_tree_sitter_failure_warns_with_task_id_and_that_path(
        self, tmp_path, caplog,
    ):
        wf = _make_bare_workflow()
        tree_sitter_cache = _blocking_file(tmp_path) / 'tree-sitter'
        write_set = _make_write_set(
            claude_fleet=tmp_path / '.claude' / 'fleet',
            tree_sitter_cache=tree_sitter_cache,
        )

        warnings = _warnings_during_ensure(wf, write_set, caplog)

        assert warnings, 'expected a loud WARNING when pre-creation fails'
        rendered = ' '.join(warnings)
        assert '2909' in rendered
        assert str(tree_sitter_cache) in rendered

    def test_only_the_failing_path_is_warned_about(self, tmp_path, caplog):
        wf = _make_bare_workflow()
        claude_fleet = tmp_path / '.claude' / 'fleet'
        write_set = _make_write_set(
            claude_fleet=claude_fleet,
            tree_sitter_cache=_blocking_file(tmp_path) / 'tree-sitter',
        )

        warnings = _warnings_during_ensure(wf, write_set, caplog)

        assert str(claude_fleet) not in ' '.join(warnings)
        assert claude_fleet.is_dir()

    def test_fleet_failure_warns_with_task_id_and_only_that_path(
        self, tmp_path, caplog,
    ):
        wf = _make_bare_workflow()
        claude_fleet = _blocking_file(tmp_path) / 'fleet'
        tree_sitter_cache = tmp_path / '.cache' / 'tree-sitter'
        write_set = _make_write_set(
            claude_fleet=claude_fleet, tree_sitter_cache=tree_sitter_cache,
        )

        warnings = _warnings_during_ensure(wf, write_set, caplog)

        rendered = ' '.join(warnings)
        assert '2909' in rendered
        assert str(claude_fleet) in rendered
        assert str(tree_sitter_cache) not in rendered
        assert tree_sitter_cache.is_dir()

    def test_each_failing_path_gets_its_own_warning(self, tmp_path, caplog):
        wf = _make_bare_workflow()
        blocker = _blocking_file(tmp_path)
        claude_fleet = blocker / 'fleet'
        tree_sitter_cache = blocker / 'tree-sitter'
        write_set = _make_write_set(
            claude_fleet=claude_fleet, tree_sitter_cache=tree_sitter_cache,
        )

        warnings = _warnings_during_ensure(wf, write_set, caplog)

        assert len(warnings) == 2, warnings
        assert str(claude_fleet) in warnings[0]
        assert str(tree_sitter_cache) in warnings[1]


@pytest.mark.asyncio
class TestInvokeWiresEnsureSandboxDirs:
    """Pins the CALL-SITE wiring inside ``TaskWorkflow._invoke``: for a
    sandboxed role the claude_fleet and tree_sitter_cache carve-outs are
    pre-created AFTER ``compute_write_set`` and BEFORE
    ``write_set.writable_paths()`` is consumed into ``sandbox_extras``
    (tasks 2996, 3382).

    The isolated ``TestEnsureSandboxDirs`` unit tests above cover the helper's
    behaviour; this drives the REAL ``_invoke`` sandbox branch end-to-end so a
    future refactor that drops the call — or reorders it AFTER the
    ``writable_paths()`` consumption — fails loudly here (that ordering is the
    behaviour that actually matters: the dir must exist before the sandbox is
    built). Mirrors the full-``_invoke`` harness in
    test_workflow_status_on_resume.py: a real worktree + a fully-constructed
    TaskWorkflow with ``invoke_with_cap_retry`` mocked away.
    """

    async def test_carve_outs_exist_before_sandbox_extras_built(
        self, tmp_path, monkeypatch,
    ):
        workflow, cwd = await _task42_workflow(tmp_path, metadata_files=['README'])

        # Hermetic write-set: ONLY the two pre-created carve-outs are
        # materialized on disk, under tmp_path — NEVER the real ~/.claude or
        # ~/.cache. Patch compute_write_set so the sandbox branch consumes THIS
        # set (the other, dummy paths are only stringified into the
        # mocked-away sandbox_extras).
        claude_fleet = tmp_path / '.claude' / 'fleet'
        tree_sitter_cache = tmp_path / '.cache' / 'tree-sitter'
        hermetic = _make_write_set(
            claude_fleet=claude_fleet, tree_sitter_cache=tree_sitter_cache,
        )
        assert not claude_fleet.exists()
        assert not tree_sitter_cache.exists()
        monkeypatch.setattr(
            'orchestrator.workflow.compute_write_set', lambda _cwd, **_kw: hermetic,
        )

        # Spy: record whether both carve-outs already exist on disk at the
        # moment writable_paths() is consumed into sandbox_extras. If
        # pre-creation is dropped or reordered AFTER this consumption, the
        # FIRST observation is not (True, True) and the assertion below fails.
        observations: list[tuple[bool, bool]] = []
        orig_writable_paths = WriteSet.writable_paths

        def _recording_writable_paths(self):
            observations.append(
                (self.claude_fleet.is_dir(), self.tree_sitter_cache.is_dir()),
            )
            return orig_writable_paths(self)

        monkeypatch.setattr(WriteSet, 'writable_paths', _recording_writable_paths)

        mock_cap_retry = await _invoke_unconfined(workflow, IMPLEMENTER, cwd)

        # The sandboxed dispatch actually reached invoke_with_cap_retry (the
        # sandbox branch ran to completion).
        assert mock_cap_retry.await_count == 1
        # sandbox_extras was built from write_set.writable_paths()...
        assert observations, (
            'sandbox_extras must be built from write_set.writable_paths()'
        )
        # ...and both carve-outs ALREADY existed on disk at that first
        # consumption, i.e. _ensure_sandbox_dirs ran before it — the wiring
        # under test.
        assert observations[0] == (True, True), (
            'claude_fleet and tree_sitter_cache must be pre-created BEFORE '
            'sandbox_extras is built: _ensure_sandbox_dirs is wired after '
            'compute_write_set and before the write_set.writable_paths() '
            'consumption'
        )
        # And the dirs are actually materialized on disk after dispatch.
        assert claude_fleet.is_dir()
        assert tree_sitter_cache.is_dir()


async def _task42_workflow(
    tmp_path: Path, *, metadata_files: list[str],
) -> tuple[TaskWorkflow, Path]:
    """A fully-constructed TaskWorkflow for task 42 over a real repo and
    linked worktree — the ``_invoke`` prologue needs config/git_ops/scheduler/
    briefing, and cwd must be a worktree. Returns the workflow and its cwd."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    _init_git_repo(repo)
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
    assignment = TaskAssignment(
        task_id='42',
        task={
            'id': '42',
            'title': 'X',
            'description': 'Y',
            'status': 'pending',
            'metadata': {'files': metadata_files},
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
    return workflow, wt_info.path


async def _invoke_unconfined(workflow: TaskWorkflow, role, cwd: Path) -> AsyncMock:
    """Run ``workflow._invoke`` with ``invoke_with_cap_retry`` mocked away and
    the backend pinned to the 'none' escape hatch, so ``_guard_sandbox``
    returns without raising, independent of host landlock/bwrap availability.
    Returns the ``invoke_with_cap_retry`` mock."""
    saved_backend = sandbox_dispatch.get_backend()
    sandbox_dispatch.set_backend('none')
    try:
        with patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            return_value=AgentResult(success=True, output=''),
        ) as mock_cap_retry:
            await workflow._invoke(role, fixed_prompt('PROMPT'), cwd)
    finally:
        sandbox_dispatch.set_backend(saved_backend)
    return mock_cap_retry


async def _declared_files_seen(
    tmp_path: Path, monkeypatch, *, role, plan: dict, metadata_files: list[str],
    write_set: WriteSet,
) -> list[object]:
    """The ``declared_files`` each ``compute_write_set`` call received while
    ``_invoke`` ran *role* for task 42; the stub returns *write_set*."""
    workflow, cwd = await _task42_workflow(tmp_path, metadata_files=metadata_files)
    workflow.plan = plan
    seen: list[object] = []

    def _record(_cwd: Path, **kwargs: object) -> WriteSet:
        seen.append(kwargs['declared_files'])
        return write_set

    monkeypatch.setattr('orchestrator.workflow.compute_write_set', _record)
    await _invoke_unconfined(workflow, role, cwd)
    return seen


@pytest.mark.asyncio
class TestInvokeForwardsDeclaredFiles:
    """``_invoke`` hands ``compute_write_set`` the task's declared files: the
    plan's ``files``, else ``metadata.files`` — the fallback the SIMPLE_TASK
    fast path (no plan yet) depends on for the docs-only main grant."""

    async def test_plan_files_win_over_metadata_files(self, tmp_path, monkeypatch):
        seen = await _declared_files_seen(
            tmp_path, monkeypatch, role=IMPLEMENTER,
            plan={'files': ['docs/a.md']}, metadata_files=['x.py'],
            write_set=_make_write_set(
                claude_fleet=tmp_path / 'fleet',
                tree_sitter_cache=tmp_path / 'tree-sitter',
            ),
        )
        assert seen == [['docs/a.md']]

    async def test_simple_task_without_plan_falls_back_to_metadata_files(
        self, tmp_path, monkeypatch, caplog,
    ):
        granted = dataclasses.replace(
            _docs_grant_write_set(),
            claude_fleet=tmp_path / 'fleet',
            tree_sitter_cache=tmp_path / 'tree-sitter',
        )
        with caplog.at_level(logging.INFO, logger='orchestrator.workflow'):
            seen = await _declared_files_seen(
                tmp_path, monkeypatch, role=SIMPLE_TASK,
                plan={}, metadata_files=['docs/a.md'], write_set=granted,
            )

        assert seen == [['docs/a.md']]
        grant_logs = [
            r.getMessage() for r in caplog.records
            if r.levelno == logging.INFO and '/main/docs/a.md' in r.getMessage()
        ]
        assert grant_logs, [r.getMessage() for r in caplog.records]
        assert '42' in grant_logs[0]
        assert SIMPLE_TASK.name in grant_logs[0]
