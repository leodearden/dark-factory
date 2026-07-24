"""Durable merge-phase liveness stamp/clear + entry wiring (task 2991).

Task 2991 (successor to task 2931): a legitimately-live task in the
pre-enqueue MERGE phase runs a rebase/scoped-verify/queue-submit loop with NO
LLM calls, so it never refreshes ``metadata.routing.latest`` — the orphan-L0
divergence reaper's task-2931 ``_has_fresh_dispatch`` gate cannot see it and
false-promotes its scope-invariant L0 to a human-facing L1 (cluster
esc-2789-22). The fix is a DURABLE ``metadata.merge_phase_liveness =
{'entered_at': <iso>}`` stamp — the restart-survivable analog of
``routing.latest`` — written at merge entry and read by
``_has_fresh_merge_phase`` (see test_orphan_l0_reaper.py).

This module unit-tests the workflow producer side:

* ``_stamp_merge_phase_entered`` / ``_clear_merge_phase_entered`` —
  best-effort metadata stamp/clear helpers (mirroring
  ``_stamp_merge_retry_pending`` / ``_clear_merge_retry_pending``).
* the ``_run_merge_phase`` entry ordering — the stamp is written after
  ``_enter_phase(MERGE)`` and BEFORE ``_check_scope_invariant`` (so it always
  precedes the scope-invariant escalation filing and the gating bail).
* the symmetric clears at the durable-enqueue boundary
  (``_submit_to_merge_queue``) and on merge success (``_merge_and_finalise``).

Follows the mock-fixture style of ``test_workflow_merge_retry_pending.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow


@dataclass
class _Fixture:
    wf: TaskWorkflow
    update_task: AsyncMock
    scheduler: MagicMock
    git_ops: MagicMock
    queue: MagicMock


def _make(
    *,
    task_id: str = '77',
    metadata: dict | None = None,
    worktree: Path = Path('/tmp/wt-2991'),
    main_sha: str = 'BASE-SHA',
    backend_metadata: dict | None = None,
    update_task_raises: bool = False,
) -> _Fixture:
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {
        'id': task_id, 'title': 'T', 'description': 'd',
        'metadata': metadata or {},
    }
    assignment.modules = ['mod_a']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0
    config.project_root = Path('/tmp/non-existent-for-test')

    if update_task_raises:
        update_task = AsyncMock(side_effect=RuntimeError('mcp down'))
    else:
        update_task = AsyncMock(return_value=True)

    scheduler = MagicMock()
    scheduler.update_task = update_task
    scheduler.set_task_status = AsyncMock()
    scheduler.get_status = AsyncMock(return_value='in-progress')
    # _merge_fresh_metadata reads backend metadata before every stamp/clear.
    scheduler.get_task = AsyncMock(
        return_value={
            'metadata': backend_metadata if backend_metadata is not None else {},
        },
    )

    git_ops = MagicMock()
    git_ops.get_main_sha = AsyncMock(return_value=main_sha)

    queue = MagicMock()
    queue.get_by_task = MagicMock(return_value=[])

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
        escalation_queue=queue,  # type: ignore[arg-type]
    )
    wf.artifacts = MagicMock()
    wf.worktree = worktree

    return _Fixture(
        wf=wf, update_task=update_task, scheduler=scheduler,
        git_ops=git_ops, queue=queue,
    )


def _persisted_metadata(update_task: AsyncMock) -> dict:
    assert update_task.await_args is not None
    args, kwargs = update_task.await_args
    return kwargs.get('metadata') or args[1]


def _fake_run(*, head: str = 'HEAD-SHA', rc: int = 0):
    """Return an async stand-in for orchestrator.workflow._run (git rev-parse)."""

    async def _run(cmd, cwd=None, **kwargs):  # noqa: ARG001
        if rc == 0:
            return (0, head + '\n', '')
        return (rc, '', 'fatal: bad revision')

    return _run


# ---------------------------------------------------------------------------
# step-3: _stamp_merge_phase_entered helper
# ---------------------------------------------------------------------------


class TestStampHelper:
    @pytest.mark.asyncio
    async def test_stamp_persists_entered_at_iso(self):
        f = _make(metadata={'retry_ledger': {'x': 1}})

        await f.wf._stamp_merge_phase_entered()

        meta = _persisted_metadata(f.update_task)
        stamp = meta['merge_phase_liveness']
        # (a) persisted stamp is exactly {'entered_at': <non-empty iso str>}.
        assert set(stamp) == {'entered_at'}
        assert isinstance(stamp['entered_at'], str) and stamp['entered_at']
        # Parseable tz-aware ISO-8601 (what _has_fresh_merge_phase reads).
        parsed = datetime.fromisoformat(stamp['entered_at'])
        assert parsed.tzinfo is not None
        # (b) sibling metadata keys preserved (read-modify-write via
        # _merge_fresh_metadata, not a clobber).
        assert meta['retry_ledger'] == {'x': 1}
        # (c) in-memory task metadata mirrors the persisted stamp.
        assert f.wf.task['metadata']['merge_phase_liveness'] == stamp

    @pytest.mark.asyncio
    async def test_stamp_overlays_backend_sibling_keys(self):
        # A backend-only sibling key (present in get_task but NOT in the
        # in-memory copy) surviving into the persisted write proves the stamp
        # goes through _merge_fresh_metadata's backend overlay, not a raw
        # in-memory clobber — e.g. memory_hints re-attached by Stage-2
        # reconciliation after self.task was loaded.
        f = _make(
            metadata={'retry_ledger': {'x': 1}},
            backend_metadata={'memory_hints': ['h1']},
        )

        await f.wf._stamp_merge_phase_entered()

        meta = _persisted_metadata(f.update_task)
        assert meta['merge_phase_liveness']['entered_at']
        assert meta['retry_ledger'] == {'x': 1}   # in-memory sibling kept
        assert meta['memory_hints'] == ['h1']     # backend sibling overlaid

    @pytest.mark.asyncio
    async def test_stamp_is_best_effort_and_does_not_raise_on_persist_failure(self):
        f = _make(update_task_raises=True)

        # scheduler.update_task raises — the helper must swallow it (durability
        # is best-effort; a lost stamp is self-healing on the next entry).
        await f.wf._stamp_merge_phase_entered()

        # (d) In-memory metadata is still updated (persist is the last step).
        assert f.wf.task['metadata']['merge_phase_liveness']['entered_at']
