"""A diverged ``metadata.files`` is healed FROM plan.files at the plan boundary (task 3113).

Two reify shapes reached a dispatch whose ``metadata.files`` no longer
described the plan: 5367 with an EMPTY record (the sha-less done reconcile
wiped it) and 5602 with a non-empty but WRONG one.  Both re-dispatched with
the held lock set already equal to the plan's, so the scope-reconciliation
choke point took its unchanged-module branch and persisted nothing, and the
MERGE-entry tripwire fired.

The heal's predicate is module-set divergence between ``metadata.files`` and
plan.files, not emptiness, and it restores from plan.files, the authoritative
source.  This file imports no merge-lane module on purpose.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _orch_helpers import pydantic_spec
from _workflow_helpers import FakeScheduler
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.module_charter import derive_modules
from orchestrator.workflow import TaskWorkflow

_LOCK_DEPTH = 2
_TASK_ID = '5602'
_HINTS = {'entities': ['E1'], 'queries': ['q1']}
_PLAN_FILES = [
    'src/a/x.py', 'src/a/y.py', 'src/b/z.py', 'src/b/w.py',
    'tests/c/test_x.py', 'tests/c/test_z.py',
]


class _RefusingScheduler(FakeScheduler):
    async def update_task(
        self, task_id: str, metadata: str | dict, *, append: bool = False,
    ) -> bool:
        raise RuntimeError('fused-memory down')


def _modules(files: list[str]) -> list[str]:
    return derive_modules(files, _LOCK_DEPTH, task_id=_TASK_ID)


def _make(
    tmp_path: Path, *, metadata: dict, modules: list[str],
    scheduler: FakeScheduler | None = None,
) -> tuple[TaskWorkflow, FakeScheduler]:
    assignment = MagicMock()
    assignment.task_id = _TASK_ID
    assignment.task = {
        'id': _TASK_ID, 'title': 'T', 'description': 'd', 'metadata': dict(metadata),
    }
    assignment.modules = list(modules)

    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = _LOCK_DEPTH
    config.project_root = tmp_path / 'proj'

    fs = scheduler if scheduler is not None else FakeScheduler()
    fs.task_data[_TASK_ID] = {'id': _TASK_ID, 'metadata': dict(metadata)}
    wf = TaskWorkflow(
        assignment=assignment, config=config, git_ops=MagicMock(),
        scheduler=fs, briefing=MagicMock(), mcp=MagicMock(),
    )
    assert wf.config.lock_depth == _LOCK_DEPTH
    wf.modules = list(modules)
    return wf, fs


def _persisted(fs: FakeScheduler) -> dict:
    (call,) = fs.update_task_calls
    payload = call[1]
    assert isinstance(payload, dict)
    return payload


@pytest.mark.asyncio
async def test_unchanged_lock_set_heals_wiped_metadata_files(tmp_path: Path):
    wf, fs = _make(
        tmp_path, metadata={'files': [], 'memory_hints': _HINTS},
        modules=_modules(_PLAN_FILES),
    )

    result = await wf._reconcile_scope_locks(list(_PLAN_FILES))

    assert result.applied
    assert fs.blast_radius_calls == []
    payload = _persisted(fs)
    assert payload['files'] == _PLAN_FILES
    assert payload['memory_hints'] == _HINTS
    assert wf.task['metadata']['files'] == _PLAN_FILES


@pytest.mark.asyncio
async def test_unchanged_lock_set_heals_non_empty_but_divergent_files(tmp_path: Path):
    """The 5602 shape: a well-formed 4-file guess against a 6-file plan."""
    guess = ['src/a/x.py', 'src/a/y.py', 'src/b/z.py', 'docs/notes/guess.py']
    assert guess
    assert set(_modules(guess)) != set(_modules(_PLAN_FILES))
    wf, fs = _make(
        tmp_path, metadata={'files': guess, 'memory_hints': _HINTS},
        modules=_modules(_PLAN_FILES),
    )

    result = await wf._reconcile_scope_locks(list(_PLAN_FILES))

    assert result.applied
    assert fs.blast_radius_calls == []
    assert _persisted(fs)['files'] == _PLAN_FILES


@pytest.mark.asyncio
async def test_module_equivalent_metadata_is_not_rewritten(tmp_path: Path):
    subset = ['src/a/x.py', 'src/b/z.py', 'tests/c/test_x.py']
    assert set(_modules(subset)) == set(_modules(_PLAN_FILES))
    wf, fs = _make(
        tmp_path, metadata={'files': subset}, modules=_modules(_PLAN_FILES),
    )

    result = await wf._reconcile_scope_locks(list(_PLAN_FILES))

    assert result.applied
    assert fs.update_task_calls == []


@pytest.mark.asyncio
async def test_empty_plan_and_empty_metadata_writes_nothing(tmp_path: Path):
    assert _modules([]) == []
    wf, fs = _make(tmp_path, metadata={'files': []}, modules=[])

    result = await wf._reconcile_scope_locks([])

    assert result.applied
    assert fs.update_task_calls == []
    assert fs.blast_radius_calls == []


@pytest.mark.asyncio
async def test_module_change_still_routes_through_blast_radius_only(tmp_path: Path):
    held = _modules(['src/a/x.py'])
    assert set(held) != set(_modules(_PLAN_FILES))
    wf, fs = _make(tmp_path, metadata={'files': []}, modules=held)

    result = await wf._reconcile_scope_locks(list(_PLAN_FILES))

    assert result.applied
    assert [persist for _cur, _need, persist in fs.blast_radius_calls] == [_PLAN_FILES]
    assert fs.update_task_calls == []
    assert wf.modules == _modules(_PLAN_FILES)


@pytest.mark.asyncio
async def test_wiped_record_no_longer_trips_merge_entry_tripwire(tmp_path: Path):
    wiped = {'files': [], 'memory_hints': _HINTS}
    wf, fs = _make(tmp_path, metadata=wiped, modules=_modules(_PLAN_FILES))
    queue = EscalationQueue(tmp_path / 'escalations')
    wf.escalation_queue = queue
    wf.plan = {'files': list(_PLAN_FILES)}

    await wf._reconcile_scope_locks(list(_PLAN_FILES))
    fs.task_data[_TASK_ID] = {'id': _TASK_ID, 'metadata': _persisted(fs)}
    await wf._check_scope_invariant()

    assert queue.get_by_task(_TASK_ID) == []

    fs.task_data[_TASK_ID] = {'id': _TASK_ID, 'metadata': dict(wiped)}
    await wf._check_scope_invariant()

    (esc,) = queue.get_by_task(_TASK_ID)
    assert 'plan.files/metadata.files divergence detected' in esc.summary


@pytest.mark.asyncio
async def test_heal_write_failure_is_fail_open(tmp_path: Path):
    """A repair must never fail a plan; the MERGE-entry tripwire stays the backstop."""
    held = _modules(_PLAN_FILES)
    wf, _ = _make(
        tmp_path, metadata={'files': []}, modules=held,
        scheduler=_RefusingScheduler(),
    )

    result = await wf._reconcile_scope_locks(list(_PLAN_FILES))

    assert result.applied
    assert wf.modules == held
