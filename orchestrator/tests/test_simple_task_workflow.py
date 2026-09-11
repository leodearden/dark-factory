"""Tests for ``TaskWorkflow._run_simple_task`` — Lever C integration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec
from _workflow_helpers import FakeMetadataBackend, wire_metadata_backend
from shared.cli_invoke import AgentResult

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


@dataclass
class _Fixture:
    wf: TaskWorkflow
    artifacts: TaskArtifacts
    handle_blast_radius_expansion: AsyncMock
    update_task: AsyncMock
    invocation_count: list[int]
    event_emit: MagicMock
    # Opt-in merge-faithful backend (task 3579); None unless the test passed one.
    metadata_backend: FakeMetadataBackend | None = None


def _make_result(
    *,
    success: bool = True,
    output: str = 'done',
    cost_usd: float = 0.10,
    turns: int = 5,
    duration_ms: int = 1000,
    subtype: str | None = None,
) -> AgentResult:
    return AgentResult(
        success=success,
        output=output,
        cost_usd=cost_usd,
        duration_ms=duration_ms,
        turns=turns,
        session_id='sid',
        structured_output=None,
        subtype=subtype if subtype is not None else ('success' if success else 'error'),
        stderr='',
        account_name='test',
        timed_out=False,
        schema_salvaged=False,
    )


def _make(
    *,
    worktree: Path,
    project_root: Path,
    task_id: str = '500',
    plan_to_write: dict | None = None,
    invoke_outcome: AgentResult | None = None,
    write_unactionable: bool = False,
    blast_radius_grants: bool = True,
    task_metadata: dict | None = None,
    modules: list[str] | None = None,
    metadata_backend: FakeMetadataBackend | None = None,
) -> _Fixture:
    worktree.mkdir(parents=True, exist_ok=True)
    project_root.mkdir(parents=True, exist_ok=True)

    artifacts = TaskArtifacts(worktree)
    artifacts.init(task_id, 'Document foo', 'd', base_commit='base123')

    # `files` first so a caller-supplied task_metadata CAN override it (task
    # 3579 needs to seed a dispatch-time files list); the three existing
    # task_metadata= callers pass routing/complexity only, never files.
    task_md = {'files': ['mod_a/foo.py'], **(task_metadata or {})}

    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id,
        'title': 'Document foo',
        'description': 'Add a docstring to foo.',
        'metadata': task_md,
    }
    assignment.modules = list(modules) if modules is not None else ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = project_root
    config.simple_task_enabled = True
    config.budgets.simple_task = 1.50
    config.max_turns.simple_task = 30

    handle_blast_radius_expansion = AsyncMock(return_value=blast_radius_grants)
    update_task = AsyncMock(return_value=True)
    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()
    scheduler.handle_blast_radius_expansion = handle_blast_radius_expansion
    scheduler.update_task = update_task
    if metadata_backend is not None:
        handle_blast_radius_expansion, update_task = wire_metadata_backend(
            scheduler, metadata_backend,
            seed=task_md, grants=blast_radius_grants,
        )

    git_ops = MagicMock()
    git_ops.is_ancestor = AsyncMock(return_value=True)
    git_ops.get_main_sha = AsyncMock(return_value='mainsha')

    briefing = MagicMock()
    briefing.build_simple_task_prompt = AsyncMock(return_value='simple-task-prompt')

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=briefing,
        mcp=MagicMock(),
    )
    wf.artifacts = artifacts
    wf.worktree = worktree

    invocation_count = [0]

    async def _fake_invoke(role, prompt, worktree_arg):
        invocation_count[0] += 1
        # Side-effects to simulate the agent: write plan + mark done.
        if plan_to_write is not None:
            (artifacts.root / 'plan.json').write_text(json.dumps(plan_to_write))
        if write_unactionable:
            artifacts.write_unactionable_task(reason='spec broken', evidence='nope')
        if invoke_outcome is not None:
            return invoke_outcome
        return _make_result()

    wf._invoke = _fake_invoke  # type: ignore[assignment]
    wf._handle_unactionable_task_report = AsyncMock(  # type: ignore[assignment]
        return_value=WorkflowOutcome.BLOCKED,
    )
    wf._mark_blocked = AsyncMock(return_value=WorkflowOutcome.BLOCKED)  # type: ignore[assignment]

    event_emit = MagicMock()
    wf.event_store = MagicMock()
    wf.event_store.emit = event_emit

    return _Fixture(
        wf=wf,
        artifacts=artifacts,
        handle_blast_radius_expansion=handle_blast_radius_expansion,
        update_task=update_task,
        invocation_count=invocation_count,
        event_emit=event_emit,
        metadata_backend=metadata_backend,
    )


def _good_plan() -> dict:
    return {
        'task_id': '500',
        'title': 'Document foo',
        'analysis': 'Add a docstring',
        'files': ['mod_a/foo.py'],
        'prerequisites': [],
        'steps': [
            {'id': 'step-1', 'type': 'impl', 'description': 'add docstring',
             'status': 'done', 'commit': 'abc123'},
        ],
    }


@pytest.mark.asyncio
async def test_simple_task_success_returns_planned(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        plan_to_write=_good_plan(),
    )

    outcome = await f.wf._run_simple_task()

    assert outcome == WorkflowOutcome.PLANNED
    assert f.invocation_count == [1]
    plan = f.artifacts.read_plan()
    assert plan['_session_id'] == f.wf.session_id

    skip_calls = [
        c for c in f.event_emit.call_args_list
        if c.args and c.args[0] == EventType.phase_skipped
    ]
    assert len(skip_calls) == 1
    data = skip_calls[0].kwargs['data']
    assert data['reason'] == 'architect_skipped_simple_task'
    assert data['classifier_signals']['title'] == 'Document foo'

    # optimistic_path stamped for auto-eval, as a narrow single-key merge
    # write (task 3579) — a positional payload, NOT the whole metadata blob.
    update_calls = [
        c for c in f.update_task.call_args_list
        if len(c.args) >= 2
        and isinstance(c.args[1], dict)
        and c.args[1].get('optimistic_path') == 'simple_task'
        and c.kwargs.get('metadata_mode') == 'merge'
    ]
    assert update_calls
    # Single-key payload: a regression that re-broadens it to a whole blob
    # would clobber sibling keys such as metadata.files.
    assert set(update_calls[-1].args[1]) == {'optimistic_path'}


@pytest.mark.asyncio
async def test_simple_task_no_plan_falls_through(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        plan_to_write=None,  # agent doesn't write plan.json
    )

    outcome = await f.wf._run_simple_task()
    assert outcome == WorkflowOutcome.REQUEUED


@pytest.mark.asyncio
async def test_simple_task_no_steps_done_falls_through(tmp_path: Path):
    plan = _good_plan()
    plan['steps'][0]['status'] = 'pending'  # nothing marked done
    plan['steps'][0]['commit'] = None

    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        plan_to_write=plan,
    )

    outcome = await f.wf._run_simple_task()
    assert outcome == WorkflowOutcome.REQUEUED


@pytest.mark.asyncio
async def test_simple_task_unactionable_returns_blocked(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        write_unactionable=True,
    )

    outcome = await f.wf._run_simple_task()
    assert outcome == WorkflowOutcome.BLOCKED
    f.wf._handle_unactionable_task_report.assert_awaited_once()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_simple_task_invoke_failure_falls_through(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        invoke_outcome=_make_result(success=False, output=''),
    )

    outcome = await f.wf._run_simple_task()
    assert outcome == WorkflowOutcome.REQUEUED


def _saturation_stamp_calls(update_task: AsyncMock) -> list:
    """Every scheduler.update_task call that is a merge-mode routing write
    flipping ``simple_saturated`` True — i.e. the ``_stamp_simple_saturated``
    write, positional payload ``{'routing': {..., 'simple_saturated': True}}``
    with ``metadata_mode='merge'``."""
    return [
        c for c in update_task.call_args_list
        if len(c.args) >= 2
        and isinstance(c.args[1], dict)
        and isinstance(c.args[1].get('routing'), dict)
        and c.args[1]['routing'].get('simple_saturated') is True
        and c.kwargs.get('metadata_mode') == 'merge'
    ]


@pytest.mark.asyncio
async def test_simple_task_max_turns_stamps_saturated(tmp_path: Path):
    """A SIMPLE_TASK invocation that ends at max_turns (subtype
    ``error_max_turns`` → classify MAX_TURNS) falls through to the architect
    path AND stamps ``metadata.routing.simple_saturated=True`` via a merge-mode
    routing write, so subsequent dispatches skip the simple path (task ν)."""
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        invoke_outcome=_make_result(
            success=False, subtype='error_max_turns', output='',
        ),
    )

    outcome = await f.wf._run_simple_task()

    assert outcome == WorkflowOutcome.REQUEUED
    stamp_calls = _saturation_stamp_calls(f.update_task)
    assert stamp_calls, (
        'expected a merge-mode routing stamp with simple_saturated=True'
    )
    # _invoke is stubbed so no _record_routing_decision write intervenes —
    # the stamp is the sole routing write.
    assert stamp_calls[0].args[0] == f.wf.task_id
    # The persistence-independent in-memory flip is what makes the very next
    # in-dispatch _should_run_simple_task read return False — assert it landed
    # regardless of the (here-succeeding) scheduler write.
    assert f.wf.task['metadata']['routing']['simple_saturated'] is True


@pytest.mark.asyncio
async def test_simple_task_max_turns_in_memory_stamp_survives_write_failure(
    tmp_path: Path,
):
    """The saturation stamp is best-effort: when the merge-mode scheduler write
    raises, ``_run_simple_task`` still returns REQUEUED (the exception is
    swallowed inside ``_stamp_simple_saturated``, never propagated to crash the
    fall-through to the architect) AND the persistence-independent in-memory
    ``self.task['metadata']['routing']['simple_saturated']`` flip still lands —
    so the very next in-dispatch gate read observes the retired label (task ν)."""
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        invoke_outcome=_make_result(
            success=False, subtype='error_max_turns', output='',
        ),
    )
    # On the max_turns failure path the ONLY update_task call is the stamp
    # write, so a blanket side_effect targets exactly that write.
    f.update_task.side_effect = RuntimeError('scheduler write failed')

    outcome = await f.wf._run_simple_task()

    assert outcome == WorkflowOutcome.REQUEUED
    assert f.wf.task['metadata']['routing']['simple_saturated'] is True


@pytest.mark.asyncio
async def test_simple_task_max_turns_already_saturated_no_double_write(
    tmp_path: Path,
):
    """``_stamp_simple_saturated`` is idempotent: when the task is ALREADY
    saturated, a fresh max_turns failure early-returns before touching the
    scheduler — no double merge-write occurs — and the pre-existing in-memory
    flag is preserved (task ν)."""
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        task_metadata={'routing': {'simple_saturated': True}},
        invoke_outcome=_make_result(
            success=False, subtype='error_max_turns', output='',
        ),
    )

    outcome = await f.wf._run_simple_task()

    assert outcome == WorkflowOutcome.REQUEUED
    # Idempotent early-return: no scheduler write at all on this failure path.
    f.update_task.assert_not_awaited()
    assert not _saturation_stamp_calls(f.update_task)
    assert f.wf.task['metadata']['routing']['simple_saturated'] is True


@pytest.mark.asyncio
async def test_simple_task_generic_failure_does_not_stamp(tmp_path: Path):
    """A non-max_turns SIMPLE_TASK failure (subtype ``error_empty_output`` →
    classify EMPTY_OUTPUT) falls through WITHOUT stamping saturation — only a
    demonstrated turn-cap exhaustion retires the simple label (task ν)."""
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        invoke_outcome=_make_result(
            success=False, subtype='error_empty_output', output='',
        ),
    )

    outcome = await f.wf._run_simple_task()

    assert outcome == WorkflowOutcome.REQUEUED
    assert not _saturation_stamp_calls(f.update_task)


@pytest.mark.asyncio
async def test_simple_task_invoke_exception_falls_through(tmp_path: Path):
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')

    async def _raising(*args, **kwargs):
        raise RuntimeError('transient')

    f.wf._invoke = _raising  # type: ignore[assignment]

    outcome = await f.wf._run_simple_task()
    assert outcome == WorkflowOutcome.REQUEUED


@pytest.mark.asyncio
async def test_blast_radius_expansion_invoked_when_files_grow(tmp_path: Path):
    plan = _good_plan()
    plan['files'] = ['mod_b/x.py']  # different module from mod_a

    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        plan_to_write=plan,
    )

    outcome = await f.wf._run_simple_task()
    assert outcome == WorkflowOutcome.PLANNED
    f.handle_blast_radius_expansion.assert_awaited_once()


@pytest.mark.asyncio
async def test_should_run_simple_task_true_for_clean_simple_task(tmp_path: Path):
    """The extracted SIMPLE_TASK gate returns True for a clean author-declared
    simple task: no initial_plan, simple_task_enabled, no auto_eval_redo /
    force_full_path / hard-blocker, and NOT saturated."""
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        task_metadata={'complexity': 'simple'},
    )
    f.wf.initial_plan = None

    assert f.wf._should_run_simple_task() is True


@pytest.mark.asyncio
async def test_should_run_simple_task_false_when_saturated(tmp_path: Path):
    """Once ``metadata.routing.simple_saturated`` is stamped (task ν), the gate
    returns False even for an otherwise-clean simple task — the demonstrated-
    wrong label is not re-trusted, so the full architect path runs."""
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        task_metadata={'complexity': 'simple', 'routing': {'simple_saturated': True}},
    )
    f.wf.initial_plan = None

    assert f.wf._should_run_simple_task() is False


@pytest.mark.asyncio
async def test_optimistic_stamp_preserves_narrowed_files(tmp_path: Path):
    """Task 3579: the optimistic_path stamp must not write back the stale
    dispatch-time metadata blob — at shallow-merge mode its ``files`` key
    overwrites the NARROWED set ``_reconcile_scope_locks`` just persisted."""
    plan = _good_plan()
    plan['files'] = ['mod_a/src/foo.py']
    backend = FakeMetadataBackend()

    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        plan_to_write=plan,
        task_metadata={'files': ['mod_a/src/foo.py', 'mod_b/src/bar.py']},
        modules=['mod_a/src', 'mod_b/src'],
        metadata_backend=backend,
    )

    outcome = await f.wf._run_simple_task()

    assert outcome == WorkflowOutcome.PLANNED
    # The reconcile ran and persisted the narrowed set.
    assert backend.blast_radius_calls
    assert backend.blast_radius_calls[-1][2] == ['mod_a/src/foo.py']
    # ...and the stamp did NOT revert it.
    assert backend.blob['files'] == ['mod_a/src/foo.py']
    # The stamp itself still landed, in the backend and in memory — so a
    # failure above is the clobber, not a missing stamp.
    assert backend.blob['optimistic_path'] == 'simple_task'
    assert f.wf.task['metadata']['optimistic_path'] == 'simple_task'


@pytest.mark.asyncio
async def test_optimistic_stamp_preserves_widened_files(tmp_path: Path):
    """Task 3579, the other direction: the stale dispatch-time blob written
    back at shallow-merge mode also re-NARROWS a scope the reconcile just
    widened."""
    plan = _good_plan()
    plan['files'] = ['mod_a/src/foo.py', 'mod_b/src/bar.py']
    backend = FakeMetadataBackend()

    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        plan_to_write=plan,
        task_metadata={'files': ['mod_a/src/foo.py']},
        modules=['mod_a/src'],
        metadata_backend=backend,
    )

    outcome = await f.wf._run_simple_task()

    assert outcome == WorkflowOutcome.PLANNED
    assert backend.blast_radius_calls
    assert backend.blast_radius_calls[-1][2] == [
        'mod_a/src/foo.py', 'mod_b/src/bar.py',
    ]
    assert backend.blob['files'] == ['mod_a/src/foo.py', 'mod_b/src/bar.py']
    assert backend.blob['optimistic_path'] == 'simple_task'
    assert f.wf.task['metadata']['optimistic_path'] == 'simple_task'


@pytest.mark.asyncio
async def test_optimistic_stamp_mirrors_in_memory_when_persist_fails(
    tmp_path: Path,
):
    """Task 3579: the in-memory mirror sits OUTSIDE the scheduler-write try, so
    a failed persist must still leave optimistic_path readable in-process (the
    harness auto-eval hook reads it there) and must not change the outcome."""
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        plan_to_write=_good_plan(),
    )
    f.wf.scheduler.update_task = AsyncMock(  # type: ignore[attr-defined]
        side_effect=RuntimeError('backend down'),
    )

    outcome = await f.wf._run_simple_task()

    # Non-vacuity: the raising write really was attempted.
    assert f.wf.scheduler.update_task.await_count >= 1  # type: ignore[attr-defined]
    assert outcome == WorkflowOutcome.PLANNED
    assert f.wf.task['metadata']['optimistic_path'] == 'simple_task'


@pytest.mark.asyncio
async def test_optimistic_stamp_survives_non_dict_metadata(tmp_path: Path):
    """Task 3579: the mirror normalises a None/non-dict metadata before
    assigning. ``setdefault`` would return the existing None and raise
    TypeError — which, sitting outside the try, would escape this
    fire-and-forget stamp into its caller."""
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')

    for corrupt in (None, '{"files": []}', ['not', 'a', 'dict']):
        f.wf.task['metadata'] = corrupt

        await f.wf._stamp_optimistic_path('simple_task')

        assert f.wf.task['metadata'] == {'optimistic_path': 'simple_task'}
