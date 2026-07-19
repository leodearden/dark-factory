"""Unit tests for ``TaskWorkflow._commit_amendment_wip`` — the post-amendment
dirty-tree / no-commit guard (task 2760).

``_amend`` invokes the implementer but never commits — it only appends the
iteration log and validates plan ownership.  If an amendment left file changes
UNCOMMITTED, the loop must never re-enter VERIFY/REVIEW (which run on the dirty
worktree) with that uncommitted work, because the merge queue lands only
COMMITTED branch state — so a reviewed, verified amendment could otherwise
silently fail to land.  The guard enforces "an amendment round's success ⟹ its
work is committed on the branch" by auto-committing the dirty tree as
``wip: amendment round N`` and emitting an ``amendment_uncommitted_recovered``
event.

These tests exercise the helper in isolation with an async-mocked ``git_ops``
and a :class:`_RecordingEventStore` double.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec
from _recording_event_store import _RecordingEventStore

from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import TaskWorkflow

# Sentinel distinguishing "event_store not supplied → default recorder" from an
# explicit ``event_store=None`` (the fire-and-forget None-guard case).
_DEFAULT = object()


def _make_workflow(
    tmp_path: Path,
    *,
    has_uncommitted_work: bool = True,
    commit_sha: str | None = 'wipsha01',
    event_store: object = _DEFAULT,
) -> TaskWorkflow:
    """Minimal TaskWorkflow instance for ``_commit_amendment_wip`` tests.

    Mirrors ``test_workflow_amendment.py::_make_workflow`` (pydantic_spec
    spec_set MagicMock config, ``max_amendment_rounds=1``) but wires an
    async-mocked ``git_ops`` (``has_uncommitted_work`` / ``commit`` as
    AsyncMock), a worktree Path, and a :class:`_RecordingEventStore`.
    """
    assignment = MagicMock()
    assignment.task_id = '42'
    assignment.task = {'id': '42', 'title': 'Test Task', 'description': 'd'}
    assignment.modules = ['crates/reify-types']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = 2
    config.max_amendment_rounds = 1
    config.lock_depth = 2
    config.steward_completion_timeout = 300.0

    git_ops = MagicMock()
    git_ops.has_uncommitted_work = AsyncMock(return_value=has_uncommitted_work)
    git_ops.commit = AsyncMock(return_value=commit_sha)

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    wf.worktree = tmp_path
    wf.event_store = (
        _RecordingEventStore() if event_store is _DEFAULT else event_store
    )
    return wf


def _recovery_events(wf: TaskWorkflow) -> list[dict]:
    """The ``amendment_uncommitted_recovered`` records captured by the store."""
    return [
        rec
        for etype, rec in wf.event_store.events  # type: ignore[union-attr]
        if str(etype).endswith('amendment_uncommitted_recovered')
    ]


@pytest.mark.asyncio
async def test_dirty_tree_commits_and_emits_event(tmp_path):
    """A dirty worktree is committed as ``wip: amendment round N`` and the
    recovery event is emitted with the round and the WIP sha."""
    wf = _make_workflow(
        tmp_path, has_uncommitted_work=True, commit_sha='wipsha01',
    )

    sha = await wf._commit_amendment_wip(2)

    # (a) commit awaited exactly once with the worktree + a round-2 message.
    wf.git_ops.commit.assert_awaited_once()
    (call_wt, call_msg), _kwargs = wf.git_ops.commit.await_args
    assert call_wt == tmp_path
    assert 'amendment round 2' in call_msg
    # (b) returns the WIP sha.
    assert sha == 'wipsha01'
    # (c) exactly one recovery event, carrying round 2 + the sha.
    recs = _recovery_events(wf)
    assert len(recs) == 1
    assert recs[0]['data']['amendment_round'] == 2
    assert recs[0]['data']['wip_sha'] == 'wipsha01'
    assert recs[0]['data']['recovery'] == 'auto_commit'


@pytest.mark.asyncio
async def test_clean_tree_is_noop_no_commit_no_event(tmp_path):
    """A clean worktree is a no-op: no commit, no event, returns None.

    Nothing was left uncommitted, so there is no data-loss risk to recover —
    the guard must not manufacture an empty WIP commit or a spurious event."""
    wf = _make_workflow(tmp_path, has_uncommitted_work=False)

    sha = await wf._commit_amendment_wip(1)

    assert sha is None
    wf.git_ops.commit.assert_not_awaited()
    assert _recovery_events(wf) == []
    assert wf.event_store.events == []


@pytest.mark.asyncio
async def test_dirty_tree_with_no_event_store_commits_without_raising(tmp_path):
    """Fire-and-forget None-guard: with ``event_store=None`` and a dirty tree,
    the guard still commits and returns the sha without raising (mirrors the
    ``_emit_rebase_verify_cost`` guarded-emit pattern)."""
    wf = _make_workflow(
        tmp_path,
        has_uncommitted_work=True,
        commit_sha='sha',
        event_store=None,
    )

    sha = await wf._commit_amendment_wip(1)

    assert sha == 'sha'
    wf.git_ops.commit.assert_awaited_once()
