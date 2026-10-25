"""Tests for the Lever B revalidation skip in ``TaskWorkflow._plan``.

When the prior plan has provenance and the diff main has gained since the
plan was stamped does not overlap any plan files, the architect's
revalidation pass is short-circuited. The orchestrator stamps
``_revalidated_at`` and bumps ``base_commit`` directly (mirroring
``confirm_plan``), then returns PLANNED without invoking the architect.

Task 4031 adds a fifth precondition: the plan's ``_task_text_fp`` must equal
the fingerprint of the task's current description/details.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec
from _workflow_helpers import FakeMetadataBackend, wire_metadata_backend
from shared.cli_invoke import AgentResult
from shared.task_text_fingerprint import TaskTextDrift, task_text_fingerprint

from orchestrator.artifacts import PLAN_SCHEMA_VERSION, TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.scheduler import BlastRadiusResult, files_to_modules
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome


@dataclass
class _Fixture:
    wf: TaskWorkflow
    artifacts: TaskArtifacts
    handle_blast_radius_expansion: AsyncMock
    update_task: AsyncMock
    get_main_sha: AsyncMock
    get_changed_files: AsyncMock
    invoke_called: list[bool]
    event_emit: MagicMock
    build_architect_prompt: AsyncMock
    build_revalidation_prompt: AsyncMock
    # Opt-in merge-faithful backend (task 3579); None unless the test passed one.
    metadata_backend: FakeMetadataBackend | None = None


_FP_OF_D = task_text_fingerprint('d', None)


def _make(
    *,
    worktree: Path,
    project_root: Path,
    task_id: str = '101',
    plan_files: list[str] | None = None,
    plan_session_id: str = 'prior-session-aaa',
    plan_age_hours: float = 1.0,
    schema_version: int = PLAN_SCHEMA_VERSION,
    old_main_sha: str = 'oldmain123',
    new_main_sha: str = 'newmain456',
    changed_files: list[str] | None = None,
    revalidation_skip_enabled: bool = True,
    max_revalidation_age_hours: float = 24.0,
    blast_radius_grants: bool = True,
    task_metadata: dict | None = None,
    modules: list[str] | None = None,
    create_plan_files_on_disk: bool = True,
    metadata_backend: FakeMetadataBackend | None = None,
    task_description: str = 'd',
    task_details: str | None = None,
    plan_text_fp: str | None = _FP_OF_D,
    plan_text_fp_gate_enabled: bool = True,
) -> _Fixture:
    if plan_files is None:
        plan_files = ['mod_a/file.py']
    if changed_files is None:
        changed_files = ['unrelated/other.py']
    if modules is None:
        modules = ['mod_a']

    worktree.mkdir(parents=True, exist_ok=True)
    project_root.mkdir(parents=True, exist_ok=True)

    if create_plan_files_on_disk:
        for f in plan_files:
            full = worktree / f
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text('# stub\n')

    artifacts = TaskArtifacts(worktree)
    artifacts.init(task_id, 'T', 'd', base_commit=old_main_sha)

    plan = {
        'task_id': task_id,
        'title': 'T',
        'analysis': 'analysis',
        'files': plan_files,
        'prerequisites': [],
        'steps': [
            {'id': 'step-1', 'type': 'test', 'description': 'test',
             'status': 'pending', 'commit': None},
            {'id': 'step-2', 'type': 'impl', 'description': 'impl',
             'status': 'pending', 'commit': None},
        ],
        '_session_id': plan_session_id,
        '_created_at': (
            datetime.now(UTC) - timedelta(hours=plan_age_hours)
        ).isoformat(),
        '_schema_version': schema_version,
    }
    if plan_text_fp is not None:
        plan['_task_text_fp'] = plan_text_fp
    # Write directly (bypassing artifacts.write_plan, which clobbers
    # _schema_version) so the schema-mismatch test can stage stale plans.
    (artifacts.root / 'plan.json').write_text(json.dumps(plan))

    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id, 'title': 'T', 'description': task_description,
        'details': task_details,
        'metadata': task_metadata or {},
    }
    assignment.modules = list(modules)

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = project_root
    config.revalidation_skip_enabled = revalidation_skip_enabled
    config.max_revalidation_age_hours = max_revalidation_age_hours
    config.plan_text_fp_gate_enabled = plan_text_fp_gate_enabled

    handle_blast_radius_expansion = AsyncMock(
        return_value=BlastRadiusResult(applied=blast_radius_grants),
    )
    update_task = AsyncMock(return_value=True)
    scheduler = MagicMock()
    scheduler.set_task_status = AsyncMock()
    scheduler.handle_blast_radius_expansion = handle_blast_radius_expansion
    scheduler.update_task = update_task
    if metadata_backend is not None:
        handle_blast_radius_expansion, update_task = wire_metadata_backend(
            scheduler, metadata_backend,
            seed=task_metadata or {}, grants=blast_radius_grants,
        )

    get_main_sha = AsyncMock(return_value=new_main_sha)
    get_changed_files = AsyncMock(return_value=changed_files)
    git_ops = MagicMock()
    git_ops.get_main_sha = get_main_sha
    git_ops.get_changed_files = get_changed_files

    briefing = MagicMock()
    briefing.build_revalidation_prompt = AsyncMock(return_value='revalidate-prompt')
    briefing.build_architect_prompt = AsyncMock(return_value='architect-prompt')

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
    wf._old_plan_base = old_main_sha

    invoke_called: list[bool] = []

    async def _fail_invoke(_role, build_prompt, *args, **kwargs):
        await build_prompt()
        invoke_called.append(True)
        raise AssertionError(
            'architect must NOT be invoked when revalidation skip applies'
        )

    wf._invoke = _fail_invoke  # type: ignore[assignment]

    event_emit = MagicMock()
    wf.event_store = MagicMock()
    wf.event_store.emit = event_emit

    return _Fixture(
        wf=wf,
        artifacts=artifacts,
        handle_blast_radius_expansion=handle_blast_radius_expansion,
        update_task=update_task,
        get_main_sha=get_main_sha,
        get_changed_files=get_changed_files,
        invoke_called=invoke_called,
        event_emit=event_emit,
        build_architect_prompt=briefing.build_architect_prompt,
        build_revalidation_prompt=briefing.build_revalidation_prompt,
        metadata_backend=metadata_backend,
    )


@pytest.mark.asyncio
async def test_overlap_zero_short_circuits(tmp_path: Path):
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.PLANNED
    assert f.invoke_called == []  # architect never invoked

    plan = f.artifacts.read_plan()
    assert '_revalidated_at' in plan
    assert plan['_revalidated_by_session'] == f.wf.session_id

    # base_commit advanced to current main SHA
    assert f.artifacts.read_base_commit() == 'newmain456'

    # phase_skipped event emitted with the right reason
    skip_calls = [c for c in f.event_emit.call_args_list
                  if c.args and c.args[0] == EventType.phase_skipped]
    assert len(skip_calls) == 1
    data = skip_calls[0].kwargs['data']
    assert data['reason'] == 'revalidation_skipped_no_overlap'
    assert data['plan_session_id'] == 'prior-session-aaa'
    assert data['main_sha'] == 'newmain456'

    # optimistic_path metadata stamped for auto-eval, as a narrow single-key
    # merge write (task 3579) — a positional payload, NOT the whole blob.
    update_calls = [
        c for c in f.update_task.call_args_list
        if len(c.args) >= 2
        and isinstance(c.args[1], dict)
        and c.args[1].get('optimistic_path')
        and c.kwargs.get('metadata_mode') == 'merge'
    ]
    assert update_calls
    assert update_calls[-1].args[1]['optimistic_path'] == 'revalidation_skip'
    # Single-key payload: a regression that re-broadens it to a whole blob
    # would clobber sibling keys such as metadata.files.
    assert set(update_calls[-1].args[1]) == {'optimistic_path'}

    # Task 4031: a current fingerprint is silent and the skip re-stamps nothing.
    assert _drift_events(f) == []
    assert _mirror_writes(f) == []


@pytest.mark.asyncio
async def test_overlap_nonzero_falls_through(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        plan_files=['mod_a/file.py'],
        changed_files=['mod_a/file.py', 'other/x.py'],  # overlap with plan
    )

    # Architect not stubbed for success — falling through means it gets
    # invoked, which our _fail_invoke surfaces. Test expectation: assertion
    # error from _fail_invoke confirms fall-through behavior.
    with pytest.raises(AssertionError, match='architect must NOT'):
        await f.wf._plan()
    assert f.invoke_called == [True]


@pytest.mark.asyncio
async def test_revalidation_skip_disabled_falls_through(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        revalidation_skip_enabled=False,
    )

    with pytest.raises(AssertionError, match='architect must NOT'):
        await f.wf._plan()
    assert f.invoke_called == [True]


@pytest.mark.asyncio
async def test_missing_plan_file_falls_through(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        plan_files=['mod_a/file.py'],
        create_plan_files_on_disk=False,
    )

    with pytest.raises(AssertionError, match='architect must NOT'):
        await f.wf._plan()
    assert f.invoke_called == [True]


@pytest.mark.asyncio
async def test_schema_version_mismatch_falls_through(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        schema_version=PLAN_SCHEMA_VERSION + 99,
    )

    with pytest.raises(AssertionError, match='architect must NOT'):
        await f.wf._plan()
    assert f.invoke_called == [True]


@pytest.mark.asyncio
async def test_age_bound_exceeded_falls_through(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        plan_age_hours=72.0,
        max_revalidation_age_hours=24.0,
    )

    with pytest.raises(AssertionError, match='architect must NOT'):
        await f.wf._plan()
    assert f.invoke_called == [True]


@pytest.mark.asyncio
async def test_prior_review_block_falls_through(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        task_metadata={
            'last_block_phase': 'review',
            'last_block_outcome': 'blocked',
        },
    )

    with pytest.raises(AssertionError, match='architect must NOT'):
        await f.wf._plan()
    assert f.invoke_called == [True]


@pytest.mark.asyncio
async def test_blast_radius_denied_falls_through(tmp_path: Path):
    # Plan declares files in a *different* module from the held lock —
    # forces blast-radius expansion. Scheduler denies → fall through.
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        plan_files=['mod_b/x.py'],   # module mod_b
        modules=['mod_a'],            # held lock is mod_a
        blast_radius_grants=False,
    )

    with pytest.raises(AssertionError, match='architect must NOT'):
        await f.wf._plan()
    assert f.invoke_called == [True]


@pytest.mark.asyncio
async def test_post_skip_revalidating_session_owns_plan(tmp_path: Path):
    """After a Lever B skip, ``validate_plan_owner`` must accept the
    revalidating session — otherwise the very next ``_amend`` call inside
    the same run would fire a spurious ``_escalate_plan_overwrite``
    (the run-3 bug, esc-2911-39).

    The original planner remains a valid owner (audit trail), and a third
    party is still rejected.
    """
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')

    outcome = await f.wf._plan()
    assert outcome == WorkflowOutcome.PLANNED

    # The session driving _plan() is the revalidating session — it must
    # now own the plan even though _session_id was preserved.
    assert f.artifacts.validate_plan_owner(f.wf.session_id) is True

    # Original planner still recognised (the preserved _session_id).
    assert f.artifacts.validate_plan_owner('prior-session-aaa') is True

    # Foreign sessions still rejected.
    assert f.artifacts.validate_plan_owner('foreign-session') is False


@pytest.mark.asyncio
async def test_blast_radius_granted_succeeds(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        plan_files=['mod_b/x.py'],
        modules=['mod_a'],
        blast_radius_grants=True,
    )

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.PLANNED
    assert f.invoke_called == []
    f.handle_blast_radius_expansion.assert_awaited_once()
    # modules updated to the plan's modules (lock_depth=2 keeps file path)
    assert set(f.wf.modules) == {'mod_b/x.py'}


# ---------------------------------------------------------------------------
# task 2557 step-11: include_prior_proposals wiring on the `else` (architect)
# branch of _plan — fresh-vs-replan.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_blast_radius_dir_bearing_plan_files_alpha_stripped(tmp_path: Path):
    """Task 2373: ``_plan``'s plan-file module-lock re-derivation routes
    through ``module_charter.derive_modules``. A directory-shaped entry in
    plan.json's "files" (no recognised extension) must be stripped before
    coarsening, so it derives only the file-level sibling's module key
    instead of widening to a subtree-wide prefix lock (reify-3468 class).
    """
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        # 'mod_b/subdir' has no extension -> directory-shaped; the sibling
        # file is the only entry that should survive alpha-strip.
        plan_files=['mod_b/subdir', 'mod_b/x.py'],
        modules=['mod_a'],
        blast_radius_grants=True,
    )

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.PLANNED
    f.handle_blast_radius_expansion.assert_awaited_once()
    _task_id, _current, needed = f.handle_blast_radius_expansion.call_args.args
    assert needed == ['mod_b/x.py']
    assert set(f.wf.modules) == {'mod_b/x.py'}


@pytest.mark.asyncio
async def test_fresh_dispatch_excludes_prior_proposals(tmp_path: Path):
    """Truly-fresh dispatch (no existing plan at all) reaches the `else`
    branch with a falsy existing_plan — C-A1 anti-anchoring — so
    build_architect_prompt must be awaited with include_prior_proposals
    False (or omitted, relying on the parameter's own False default).
    """
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
    # Simulate a truly-fresh dispatch: no plan.json on disk at all.
    (f.artifacts.root / 'plan.json').unlink()

    with pytest.raises(AssertionError, match='architect must NOT'):
        await f.wf._plan()

    f.build_architect_prompt.assert_awaited_once()
    _, kwargs = f.build_architect_prompt.call_args
    assert kwargs.get('include_prior_proposals', False) is False


@pytest.mark.asyncio
async def test_replan_fallthrough_includes_prior_proposals(tmp_path: Path):
    """A fallen-through existing plan (has steps + _session_id, but no
    _old_plan_base to revalidate against) reaches the `else` branch with a
    truthy existing_plan — a genuine re-plan, not a first dispatch — so
    build_architect_prompt must be awaited with include_prior_proposals=True.
    """
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
    # Force fallthrough past the revalidation elif: no _old_plan_base means
    # its last condition is falsy even though the plan has steps + _session_id.
    f.wf._old_plan_base = None

    with pytest.raises(AssertionError, match='architect must NOT'):
        await f.wf._plan()

    f.build_architect_prompt.assert_awaited_once()
    _, kwargs = f.build_architect_prompt.call_args
    assert kwargs.get('include_prior_proposals') is True


@pytest.mark.asyncio
async def test_optimistic_stamp_preserves_narrowed_files(tmp_path: Path):
    """Task 3579: on the revalidation-skip path the optimistic_path stamp must
    not write back the stale dispatch-time metadata blob — at shallow-merge
    mode its ``files`` key overwrites the narrowed set
    ``_reconcile_scope_locks`` just persisted."""
    backend = FakeMetadataBackend()
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        plan_files=['mod_a/src/foo.py'],
        modules=['mod_a/src', 'mod_b/src'],
        task_metadata={'files': ['mod_a/src/foo.py', 'mod_b/src/bar.py']},
        changed_files=['unrelated/other.py'],
        metadata_backend=backend,
    )

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.PLANNED
    assert f.invoke_called == []  # architect never invoked
    # The reconcile ran and persisted the narrowed set.
    assert backend.blast_radius_calls
    assert backend.blast_radius_calls[-1][2] == ['mod_a/src/foo.py']
    # ...and the stamp did NOT revert it.
    assert backend.blob['files'] == ['mod_a/src/foo.py']
    # The stamp itself still landed — a failure above is the clobber, not a
    # missing stamp.
    assert backend.blob['optimistic_path'] == 'revalidation_skip'
    assert f.wf.task['metadata']['optimistic_path'] == 'revalidation_skip'


def _wiped_unchanged_lock_fixture(
    tmp_path: Path, *, changed_files: list[str] | None = None,
) -> tuple[_Fixture, FakeMetadataBackend]:
    """Task 3113: a wiped ``metadata.files`` whose plan modules equal the held lock set."""
    plan_files = ['mod_a/file.py']
    held = files_to_modules(plan_files, 2)
    backend = FakeMetadataBackend()
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        plan_files=plan_files, modules=held,
        task_metadata={'files': []}, metadata_backend=backend,
        changed_files=changed_files,
    )
    assert set(f.wf.modules) == set(
        files_to_modules(plan_files, f.wf.config.lock_depth),
    )
    return f, backend


@pytest.mark.asyncio
async def test_revalidation_skip_heals_wiped_metadata_files(tmp_path: Path):
    """The skip path re-establishes plan.files with the lock set unchanged, so
    the choke point's heal must restore the wiped record from plan.files."""
    f, backend = _wiped_unchanged_lock_fixture(tmp_path)

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.PLANNED
    assert f.invoke_called == []
    f.handle_blast_radius_expansion.assert_not_awaited()
    assert backend.blob['files'] == ['mod_a/file.py']
    assert backend.blob['optimistic_path'] == 'revalidation_skip'


@pytest.mark.asyncio
async def test_architect_plan_boundary_heals_wiped_metadata_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The architect ``_plan()`` boundary heals the same shape, keeping the
    sibling ``plan_text_fp`` key task 4031's mirror wrote just before."""
    f, backend = _wiped_unchanged_lock_fixture(
        tmp_path, changed_files=['mod_a/file.py'],
    )
    _confirm_plan_architect(f, monkeypatch)

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.PLANNED
    assert backend.blob['files'] == ['mod_a/file.py']
    assert backend.blob['plan_text_fp'] == task_text_fingerprint('d', None)
    f.handle_blast_radius_expansion.assert_not_awaited()


# ---------------------------------------------------------------------------
# Task 4031: the task-text fingerprint gates the skip.
# ---------------------------------------------------------------------------

def _events(f: _Fixture, event_type: EventType) -> list[dict]:
    return [c.kwargs['data'] for c in f.event_emit.call_args_list
            if c.args and c.args[0] == event_type]


def _drift_events(f: _Fixture) -> list[dict]:
    return _events(f, EventType.plan_text_drift)


def _mirror_writes(f: _Fixture) -> list:
    return [c for c in f.update_task.call_args_list
            if len(c.args) >= 2 and isinstance(c.args[1], dict)
            and 'plan_text_fp' in c.args[1]]


def _revalidation_drift(f: _Fixture) -> TaskTextDrift | None:
    f.build_revalidation_prompt.assert_awaited_once()
    return f.build_revalidation_prompt.call_args.kwargs['text_drift']


async def _plan_falls_through_to_architect(f: _Fixture) -> None:
    with pytest.raises(AssertionError, match='architect must NOT'):
        await f.wf._plan()
    assert f.invoke_called == [True]


def _confirm_plan_architect(f: _Fixture, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """An architect that only calls confirm_plan(): every plan key is kept."""
    async def _confirm(_role, build_prompt, *args, **kwargs) -> AgentResult:
        await build_prompt()
        plan_path = f.artifacts.root / 'plan.json'
        plan = json.loads(plan_path.read_text())
        plan['_finalized_at'] = datetime.now(UTC).isoformat()
        plan_path.write_text(json.dumps(plan))
        return AgentResult(
            success=True, output='confirmed', cost_usd=0.5,
            duration_ms=1_000, turns=3,
        )

    invoke = AsyncMock(side_effect=_confirm)
    monkeypatch.setattr(f.wf, '_invoke', invoke)
    return invoke


@pytest.mark.asyncio
async def test_amended_description_declines_skip_6820(tmp_path: Path):
    """The 6820 incident: items 1-3 were planned, then item 4 and an
    ACCEPTANCE clause were folded into the description."""
    original = 'WORK: items 1-3.'
    amended = 'WORK: items 1-3. Item 4: transcript secret scanner. ACCEPTANCE: scan clean.'
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        task_description=amended,
        plan_text_fp=task_text_fingerprint(original, None),
    )

    await _plan_falls_through_to_architect(f)

    f.build_architect_prompt.assert_not_awaited()
    assert 'Item 4: transcript secret scanner' in (
        f.build_revalidation_prompt.call_args.args[0]['description']
    )
    assert _revalidation_drift(f) is TaskTextDrift.AMENDED
    assert not [d for d in _events(f, EventType.phase_skipped)
                if d.get('reason') == 'revalidation_skipped_no_overlap']
    assert _drift_events(f) == [{
        'drift': 'amended',
        'plan_text_fp': task_text_fingerprint(original, None),
        'task_text_fp': task_text_fingerprint(amended, None),
        'plan_session_id': 'prior-session-aaa',
    }]


@pytest.mark.asyncio
async def test_details_corrected_after_finalize_replans_and_restamps_4822(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """The 4822 incident (ADDED SCOPE): details corrected after the plan was
    finalized. The architect only confirms, yet the plan leaves _plan()
    carrying the fingerprint of the text it was just shown."""
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        task_details='Original scope. ADDED SCOPE: a 4822-shaped regression test.',
        plan_text_fp=task_text_fingerprint('d', 'Original scope.'),
    )
    _confirm_plan_architect(f, monkeypatch)

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.PLANNED
    assert _events(f, EventType.plan_revalidated)
    assert _events(f, EventType.phase_skipped) == []
    plan = f.artifacts.read_plan()
    current_fp = task_text_fingerprint(
        'd', 'Original scope. ADDED SCOPE: a 4822-shaped regression test.',
    )
    assert plan['_task_text_fp'] == current_fp
    [mirror] = _mirror_writes(f)
    assert mirror.args[1] == {
        'plan_text_fp': plan['_task_text_fp'],
        'plan_authored_at': plan['_created_at'],
    }
    assert mirror.kwargs == {}  # update_task's default: a shallow merge
    assert f.wf.task['metadata']['plan_text_fp'] == current_fp


@pytest.mark.asyncio
async def test_kwargless_stamp_keeps_the_fingerprint_so_an_amendment_still_declines(
    tmp_path: Path,
):
    """The per-iteration stamp shape must never refresh the fingerprint, or a
    mid-EXECUTE amendment would self-clear before the next dispatch."""
    f = _make(worktree=tmp_path / 'wt', project_root=tmp_path / 'proj')
    stamped = task_text_fingerprint('d', None)
    f.artifacts.stamp_plan_provenance('prior-session-aaa', task_text_fp=stamped)
    f.wf.task['description'] = 'd, amended mid-execute'

    f.artifacts.stamp_plan_provenance('prior-session-aaa')

    assert f.artifacts.read_plan()['_task_text_fp'] == stamped
    await _plan_falls_through_to_architect(f)
    assert [d['drift'] for d in _drift_events(f)] == ['amended']


@pytest.mark.asyncio
async def test_unstamped_plan_declines_skip_once(tmp_path: Path):
    """A legacy plan with no fingerprint (Leo's LEO-A ruling, option i)."""
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        plan_text_fp=None,
    )

    await _plan_falls_through_to_architect(f)

    assert _revalidation_drift(f) is TaskTextDrift.UNSTAMPED
    [event] = _drift_events(f)
    assert event['drift'] == 'unstamped'
    assert event['plan_text_fp'] is None


@pytest.mark.asyncio
async def test_reflowed_description_still_skips(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        task_description='fix the  bug\n\nin foo  ',
        plan_text_fp=task_text_fingerprint('fix the bug in foo', None),
    )

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.PLANNED
    assert f.invoke_called == []
    assert _drift_events(f) == []


@pytest.mark.asyncio
async def test_case_change_declines_skip(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        task_description='Foo',
        plan_text_fp=task_text_fingerprint('foo', None),
    )

    await _plan_falls_through_to_architect(f)

    assert [d['drift'] for d in _drift_events(f)] == ['amended']


@pytest.mark.asyncio
async def test_gate_off_restores_the_pre_4031_skip(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        task_description='d, amended',
        plan_text_fp_gate_enabled=False,
    )

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.PLANNED
    assert f.invoke_called == []
    assert [d['reason'] for d in _events(f, EventType.phase_skipped)] == [
        'revalidation_skipped_no_overlap',
    ]
    assert f.wf.task['metadata']['optimistic_path'] == 'revalidation_skip'
    assert _drift_events(f) == []


@pytest.mark.asyncio
async def test_gate_off_still_stamps_authored_plans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        changed_files=['mod_a/file.py'],
        task_description='d, amended',
        plan_text_fp_gate_enabled=False,
    )
    _confirm_plan_architect(f, monkeypatch)

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.PLANNED
    assert f.artifacts.read_plan()['_task_text_fp'] == task_text_fingerprint(
        'd, amended', None,
    )
    assert _revalidation_drift(f) is None
    assert _drift_events(f) == []


@pytest.mark.asyncio
async def test_architect_is_told_of_drift_when_skip_is_disabled(tmp_path: Path):
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        revalidation_skip_enabled=False,
        task_description='d, amended',
    )

    await _plan_falls_through_to_architect(f)

    assert _revalidation_drift(f) is TaskTextDrift.AMENDED
    assert [d['drift'] for d in _drift_events(f)] == ['amended']


@pytest.mark.parametrize(
    'mirror_failure', [False, RuntimeError('fused-memory down')],
    ids=['refused', 'raises'],
)
@pytest.mark.asyncio
async def test_mirror_failure_never_fails_the_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mirror_failure,
):
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        task_description='d, amended',
    )

    async def _update_task(_task_id, payload, **_kwargs):
        if isinstance(payload, dict) and 'plan_text_fp' in payload:
            if isinstance(mirror_failure, Exception):
                raise mirror_failure
            return mirror_failure
        return True

    f.update_task.side_effect = _update_task
    _confirm_plan_architect(f, monkeypatch)

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.PLANNED
    assert f.artifacts.read_plan()['_task_text_fp'] == task_text_fingerprint(
        'd, amended', None,
    )


def _finalize_fixture_plan(f: _Fixture) -> None:
    plan_path = f.artifacts.root / 'plan.json'
    plan = json.loads(plan_path.read_text())
    plan['_finalized_at'] = '2026-10-01T00:00:00+00:00'
    plan_path.write_text(json.dumps(plan))


def _untouching_architect(
    f: _Fixture, monkeypatch: pytest.MonkeyPatch, *, success: bool,
) -> None:
    """An architect that never calls confirm_plan(): budget-capped, or done early."""
    async def _leave_plan(_role, build_prompt, *args, **kwargs) -> AgentResult:
        await build_prompt()
        return AgentResult(
            success=success, output='', cost_usd=3.0, duration_ms=60_000, turns=40,
        )

    monkeypatch.setattr(f.wf, '_invoke', AsyncMock(side_effect=_leave_plan))


@pytest.mark.parametrize('success', [False, True], ids=['capped', 'no_confirm'])
@pytest.mark.parametrize(
    'stale_fp', [task_text_fingerprint('d', None), None], ids=['stamped', 'unstamped'],
)
@pytest.mark.asyncio
async def test_unconfirmed_revalidation_never_mints_a_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, success: bool, stale_fp: str | None,
):
    """The amended text must not be certified onto a plan no architect re-read,
    or the drift would self-clear and the stale plan would skip forever."""
    f = _make(
        worktree=tmp_path / 'wt', project_root=tmp_path / 'proj',
        task_description='d, amended',
        plan_text_fp=stale_fp,
    )
    _finalize_fixture_plan(f)
    _untouching_architect(f, monkeypatch, success=success)

    outcome = await f.wf._plan()

    assert outcome == WorkflowOutcome.PLANNED
    assert f.artifacts.read_plan().get('_task_text_fp') == stale_fp
    assert _mirror_writes(f) == []
    assert 'plan_text_fp' not in f.wf.task['metadata']
