"""Tests for ``TaskWorkflow._reconcile_metadata_files_for_done``.

Refactored 2026-05-10 (Stage 2 of stuck-done recovery): truth source is
the merge-diff (``git diff base..merge_sha``), not the architect's
``plan.files``.  The architect's plan can include files that get squashed
or refactored away before merge; the merge-diff is the actual record of
what landed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec
from shared.locking import directory_locks, strip_directory_locks

from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow


def _make_workflow(
    *,
    project_root: Path,
    task_id: str = '101',
    backend_metadata: dict | None = None,
) -> tuple[TaskWorkflow, AsyncMock, AsyncMock]:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.project_root = project_root

    update_task = AsyncMock(return_value=True)
    scheduler = MagicMock()
    scheduler.update_task = update_task
    scheduler.get_task = AsyncMock(
        return_value={'id': task_id, 'metadata': backend_metadata or {}}
    )

    git_ops = MagicMock()
    get_merge_diff_files = AsyncMock(return_value=([], None))
    git_ops.get_merge_diff_files = get_merge_diff_files

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    return wf, update_task, get_merge_diff_files


def _persisted_payload(update_task: AsyncMock) -> dict:
    """Return the metadata dict passed to the most recent update_task call."""
    assert update_task.await_args is not None
    args, _ = update_task.await_args
    return args[1]


@pytest.mark.asyncio
async def test_writes_merge_diff_files_when_merge_sha_and_base_commit_set(
    tmp_path: Path,
):
    """Happy path: writes git-diff base..merge_sha (not plan.files)."""
    wf, update_task, get_merge_diff_files = _make_workflow(project_root=tmp_path)
    wf._base_commit = 'a' * 40
    wf._merge_sha = 'b' * 40
    # plan.files names what the architect said it would touch — but the
    # merge actually landed a different set of paths.
    wf.plan = {'files': ['old/path.py'], 'steps': []}
    get_merge_diff_files.return_value = (
        ['src/landed_a.py', 'src/landed_b.py'], None,
    )

    await wf._reconcile_metadata_files_for_done()

    get_merge_diff_files.assert_awaited_once_with('a' * 40, 'b' * 40)
    update_task.assert_awaited_once_with(
        '101', {'files': ['src/landed_a.py', 'src/landed_b.py']},
    )
    # Anchor the read half of the RMW — prevents silent regression where
    # get_task is skipped and only the bare {'files': ...} payload is sent.
    wf.scheduler.get_task.assert_awaited_once_with('101')  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_clears_when_merge_sha_missing(tmp_path: Path):
    """No merge_sha (already-on-main shortcuts) → write empty list.

    The gate-skip-when-verified-provenance branch in fused-memory's
    task_interceptor.py handles the missing-files case.
    """
    wf, update_task, get_merge_diff_files = _make_workflow(project_root=tmp_path)
    wf._base_commit = 'a' * 40
    wf._merge_sha = None
    wf.plan = {'files': ['anything.py']}

    await wf._reconcile_metadata_files_for_done()

    get_merge_diff_files.assert_not_awaited()
    update_task.assert_awaited_once_with('101', {'files': []})


@pytest.mark.asyncio
async def test_clears_when_base_commit_missing(tmp_path: Path):
    """No base_commit (eval mode without create_worktree) → empty list."""
    wf, update_task, get_merge_diff_files = _make_workflow(project_root=tmp_path)
    wf._base_commit = None
    wf._merge_sha = 'b' * 40

    await wf._reconcile_metadata_files_for_done()

    get_merge_diff_files.assert_not_awaited()
    update_task.assert_awaited_once_with('101', {'files': []})


@pytest.mark.asyncio
async def test_writes_empty_list_when_diff_returns_empty(tmp_path: Path):
    """Empty diff (e.g. revert merge) → empty list, no error."""
    wf, update_task, get_merge_diff_files = _make_workflow(project_root=tmp_path)
    wf._base_commit = 'a' * 40
    wf._merge_sha = 'b' * 40
    get_merge_diff_files.return_value = ([], None)

    await wf._reconcile_metadata_files_for_done()

    update_task.assert_awaited_once_with('101', {'files': []})


@pytest.mark.asyncio
async def test_reconcile_preserves_memory_hints_from_backend(tmp_path: Path):
    """Sibling keys added by Stage-2 reconciliation survive the files write.

    The in-memory task dict has no memory_hints; the backend (get_task)
    has memory_hints + _causation_id attached after the workflow loaded.
    The persisted payload must contain BOTH the recomputed files AND the
    preserved sibling keys.
    """
    wf, update_task, get_merge_diff_files = _make_workflow(
        project_root=tmp_path,
        backend_metadata={
            'memory_hints': {'entities': ['E1'], 'queries': ['q1']},
            '_causation_id': 'C1',
        },
    )
    wf._base_commit = 'a' * 40
    wf._merge_sha = 'b' * 40
    get_merge_diff_files.return_value = (['src/landed.py'], None)

    await wf._reconcile_metadata_files_for_done()

    payload = _persisted_payload(update_task)
    assert payload == {
        'memory_hints': {'entities': ['E1'], 'queries': ['q1']},
        '_causation_id': 'C1',
        'files': ['src/landed.py'],
    }, f'Sibling keys from backend must survive the files write; got {payload}'


@pytest.mark.asyncio
async def test_reconcile_explicit_files_override_backend_files(tmp_path: Path):
    """New 'files' value always wins over any pre-existing backend 'files'.

    Even when the backend already has a stale 'files' list, the freshly
    computed merge-diff replaces it.  Sibling keys are still preserved.
    """
    wf, update_task, get_merge_diff_files = _make_workflow(
        project_root=tmp_path,
        backend_metadata={
            'files': ['old/stale.py'],
            'memory_hints': {'entities': ['E1'], 'queries': ['q1']},
        },
    )
    wf._base_commit = 'a' * 40
    wf._merge_sha = 'b' * 40
    get_merge_diff_files.return_value = (['src/new.py'], None)

    await wf._reconcile_metadata_files_for_done()

    payload = _persisted_payload(update_task)
    assert payload.get('files') == ['src/new.py'], (
        f"New files must override stale backend files; got {payload.get('files')!r}"
    )
    assert payload.get('memory_hints') == {'entities': ['E1'], 'queries': ['q1']}, (
        f"memory_hints must survive when files key is overridden; got {payload!r}"
    )


@pytest.mark.asyncio
async def test_reconcile_writes_empty_files_on_git_error_fail_open(tmp_path: Path):
    """When get_merge_diff_files returns ([], error), files=[] is written (fail-open).

    Sibling keys from the backend are still preserved — the read-modify-write
    structure is unchanged even on the error path.
    """
    wf, update_task, get_merge_diff_files = _make_workflow(
        project_root=tmp_path,
        backend_metadata={
            'memory_hints': {'entities': ['E2'], 'queries': ['q2']},
        },
    )
    wf._base_commit = 'a' * 40
    wf._merge_sha = 'b' * 40
    get_merge_diff_files.return_value = ([], OSError('diff failed'))

    await wf._reconcile_metadata_files_for_done()

    payload = _persisted_payload(update_task)
    assert payload.get('files') == [], (
        f'files must be [] (fail-open) on git error; got {payload.get("files")!r}'
    )
    assert payload.get('memory_hints') == {'entities': ['E2'], 'queries': ['q2']}, (
        f'memory_hints must survive even on git error path; got {payload!r}'
    )


@pytest.mark.asyncio
async def test_reconcile_strips_directory_shaped_merge_diff_files(tmp_path: Path):
    """Directory-shaped entries from git diff are stripped before persisting.

    ``git diff --name-only`` can return extension-less files (e.g. Dockerfile)
    or non-allowlisted dotfiles (e.g. .gitignore).  Both are present in this
    repo.  ``is_file_path`` classifies them as directory-shaped, so they would
    cause the update_task lock-charter guard (changes #2/#3) to return a
    LockCharterViolation — silently rejected (return value ignored at line 1343)
    — leaving stale plan.files and potentially tripping the phantom-done gate.

    The fix applies ``strip_directory_locks`` to the diff output before writing,
    mirroring the change-#1 scheduler._persist_files_metadata fix.
    """
    wf, update_task, get_merge_diff_files = _make_workflow(project_root=tmp_path)
    wf._base_commit = 'a' * 40
    wf._merge_sha = 'b' * 40
    # Mixed: Dockerfile and .gitignore are directory-shaped; the .py files are file-level.
    get_merge_diff_files.return_value = (
        [
            'fused-memory/docker/Dockerfile',
            '.gitignore',
            'src/landed.py',
            'orchestrator/src/orchestrator/workflow.py',
        ],
        None,
    )

    await wf._reconcile_metadata_files_for_done()

    persisted = _persisted_payload(update_task)['files']
    # Only file-level entries must survive (order preserved).
    assert persisted == [
        'src/landed.py',
        'orchestrator/src/orchestrator/workflow.py',
    ], (
        f'directory-shaped entries must be stripped; got {persisted!r}'
    )
    # The persisted set passes the update_task lock-charter guard.
    assert directory_locks(persisted) == [], (
        f'persisted files must all be file-level (no directory locks); got {directory_locks(persisted)!r}'
    )
    # Optimistic in-memory copy must match the persisted (stripped) list.
    assert wf.task['metadata']['files'] == persisted, (
        f'in-memory task metadata.files must equal persisted files; got {wf.task["metadata"]["files"]!r}'
    )
