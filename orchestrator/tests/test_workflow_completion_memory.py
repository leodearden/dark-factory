"""Tests for TaskWorkflow._write_completion_to_memory metadata stamping (task 2621).

Finding 2: the orchestrator's deterministic post-merge completion writer POSTed
``add_memory`` with NO ``metadata`` key at all — it embedded the id only in
``agent_id=f'orchestrator-task-{task_id}'``.  Qdrant payload filters and the
deterministic ``count_memories_by_metadata`` / ``get_memories_by_metadata``
audits match on ``metadata.task_id``, so those organic completion notes were
audit-invisible.  These tests pin that the write now stamps
``metadata={'task_id': str(self.task_id), 'source': 'orchestrator_completion'}``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.workflow import TaskWorkflow


class _CaptureClient:
    """Async-context httpx.AsyncClient stand-in that captures POST ``json=`` bodies."""

    def __init__(self, captured: list[dict]) -> None:
        self._captured = captured

    async def __aenter__(self) -> _CaptureClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def post(self, url: str, *, json: dict, timeout: float | None = None):
        self._captured.append({'url': url, 'json': json})
        return MagicMock()


def _make_workflow(*, tmp_path: Path, task_id: str = '2610') -> TaskWorkflow:
    """Minimal TaskWorkflow wired with only the attrs _write_completion_to_memory reads.

    Uses ``object.__new__`` to skip __init__ (the same lightweight pattern as
    test_eval_memory_isolation.py) so no worktree/git/scheduler wiring is needed.
    """
    wf = object.__new__(TaskWorkflow)
    wf.mcp = MagicMock()
    wf.mcp.url = 'http://memory.test:8002'
    wf.config = MagicMock()
    wf.config.fused_memory.project_id = 'dark_factory'
    wf.task = {'id': task_id, 'title': 'Widen done-task audit', 'description': 'does a thing'}
    wf.plan = {
        'analysis': 'because reasons',
        'design_decisions': [],
        'steps': [{'status': 'done'}, {'status': 'done'}, {'status': 'pending'}],
    }
    wf.modules = ['orchestrator/src/orchestrator']
    wf.task_id = task_id
    wf.worktree = tmp_path
    return wf


@pytest.mark.asyncio
async def test_completion_write_stamps_task_id_and_source_metadata(tmp_path):
    """The add_memory POST carries metadata.task_id (str) and metadata.source."""
    wf = _make_workflow(tmp_path=tmp_path, task_id='2610')

    captured: list[dict] = []
    with patch('httpx.AsyncClient', lambda *a, **k: _CaptureClient(captured)):
        await wf._write_completion_to_memory()

    assert len(captured) == 1, 'expected exactly one add_memory POST'
    body = captured[0]['json']
    assert body['params']['name'] == 'add_memory'
    arguments = body['params']['arguments']

    metadata = arguments.get('metadata')
    assert isinstance(metadata, dict), (
        'add_memory arguments must carry a metadata dict so the note is '
        'audit-visible via metadata.task_id'
    )
    assert metadata['task_id'] == '2610', 'metadata.task_id must be the task id (exact-string form)'
    assert metadata['task_id'] == str(wf.task_id), 'metadata.task_id must equal str(self.task_id)'
    assert metadata['source'] == 'orchestrator_completion', (
        "metadata.source must tag the organic completion writer"
    )


@pytest.mark.asyncio
async def test_completion_write_task_id_is_string_not_int(tmp_path):
    """metadata.task_id is stored as a string even when task_id is int-like.

    The deterministic count/get-by-metadata audits query on the exact-string
    form, so the tag must be str(...) regardless of the source type.
    """
    wf = _make_workflow(tmp_path=tmp_path, task_id='2610')
    # Deliberately simulate an int-like task_id (declared type is str) to pin
    # that the writer coerces via str(); setattr sidesteps the attribute-type
    # check for this intentional violation.
    setattr(wf, 'task_id', 2610)  # noqa: B010 — int form, deliberate type violation

    captured: list[dict] = []
    with patch('httpx.AsyncClient', lambda *a, **k: _CaptureClient(captured)):
        await wf._write_completion_to_memory()

    metadata = captured[0]['json']['params']['arguments']['metadata']
    assert metadata['task_id'] == '2610'
    assert isinstance(metadata['task_id'], str), 'metadata.task_id must be a str'
