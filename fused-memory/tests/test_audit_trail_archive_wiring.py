"""``server/main.py::_wire_audit_trail_archive`` arms the audit-trail rotation.

The rotation in ``TaskInterceptor.update_task`` is dormant until an archive is
wired, so this wiring is the only thing that turns it on in production. As in
``tests/test_consolidation_closure_seam.py::TestClosureCollaboratorWiring``,
arming is proved by awaiting the captured archive against a recording stub,
not by a call-args assertion on the wiring call.
"""

from __future__ import annotations

from typing import Any

import pytest

from fused_memory.models.memory import AddMemoryResponse
from fused_memory.reconciliation.audit_trail_rotation import ARCHIVE_AGENT_ID

_ARCHIVE_METADATA = {'source': 'audit_trail_rotation', 'task_id': '3708'}


class _RecordingInterceptor:
    """Mirrors ``TaskInterceptor.set_audit_trail_archive``'s real signature."""

    def __init__(self):
        self.wire_calls = 0
        self.archive: Any = None

    def set_audit_trail_archive(self, archive: Any) -> None:
        self.wire_calls += 1
        self.archive = archive


class _RecordingMemoryService:
    """The three ``MemoryService`` methods the archive uses, recording each call.

    ``get_memory_by_id`` takes ``project_id`` FIRST and positionally, so a
    transposition in the adapter would read the wrong scope as a miss.
    """

    def __init__(self, *, memory_ids=('mem-1',), record=None):
        self.memory_ids = list(memory_ids)
        self.record = record
        self.calls = []

    async def add_memory(self, content, category=None, project_id='main', **kwargs):
        self.calls.append(
            ('add_memory', (), {'content': content, 'category': category,
                                'project_id': project_id, **kwargs})
        )
        return AddMemoryResponse(memory_ids=self.memory_ids)

    async def get_memory_by_id(self, project_id, memory_id):
        self.calls.append(('get_memory_by_id', (project_id, memory_id), {}))
        return self.record

    async def delete_memory(self, memory_id, store, project_id='main', **kwargs):
        self.calls.append(
            ('delete_memory', (), {'memory_id': memory_id, 'store': store,
                                   'project_id': project_id, **kwargs})
        )
        return {'status': 'deleted'}


def _wire(memory_service):
    from fused_memory.server.main import _wire_audit_trail_archive  # noqa: PLC0415

    interceptor = _RecordingInterceptor()
    _wire_audit_trail_archive(interceptor, memory_service)
    return interceptor


class TestAuditTrailArchiveWiring:
    def test_the_archive_is_wired_exactly_once_and_non_none(self):
        interceptor = _wire(_RecordingMemoryService())
        assert interceptor.wire_calls == 1
        assert interceptor.archive is not None

    @pytest.mark.asyncio
    async def test_write_stores_an_observation_under_the_archive_agent(self):
        svc = _RecordingMemoryService(memory_ids=('mem-1', 'mem-2'))
        archive = _wire(svc).archive
        got = await archive.write(
            project_id='dark_factory', content='X', metadata=dict(_ARCHIVE_METADATA)
        )
        assert got == 'mem-1'
        [(method, args, kwargs)] = svc.calls
        assert (method, args) == ('add_memory', ())
        assert kwargs['content'] == 'X'
        assert kwargs['category'] == 'observations_and_summaries'
        assert kwargs['project_id'] == 'dark_factory'
        assert kwargs['agent_id'] == ARCHIVE_AGENT_ID
        assert kwargs['metadata'] == _ARCHIVE_METADATA

    @pytest.mark.asyncio
    async def test_write_reports_a_swallowed_mem0_failure_as_none(self):
        archive = _wire(_RecordingMemoryService(memory_ids=())).archive
        got = await archive.write(
            project_id='dark_factory', content='X', metadata=dict(_ARCHIVE_METADATA)
        )
        assert got is None

    @pytest.mark.asyncio
    async def test_read_scopes_by_project_first_and_returns_the_content(self):
        svc = _RecordingMemoryService(record={'id': 'm', 'content': 'X', 'metadata': {}})
        archive = _wire(svc).archive
        assert await archive.read(project_id='dark_factory', memory_id='m') == 'X'
        assert svc.calls == [('get_memory_by_id', ('dark_factory', 'm'), {})]

    @pytest.mark.asyncio
    async def test_read_of_a_missing_record_is_none(self):
        archive = _wire(_RecordingMemoryService(record=None)).archive
        assert await archive.read(project_id='dark_factory', memory_id='m') is None

    @pytest.mark.asyncio
    async def test_discard_deletes_the_mem0_record_in_the_project(self):
        svc = _RecordingMemoryService()
        archive = _wire(svc).archive
        await archive.discard(project_id='dark_factory', memory_id='m')
        [(method, args, kwargs)] = svc.calls
        assert (method, args) == ('delete_memory', ())
        assert kwargs['memory_id'] == 'm'
        assert kwargs['store'] == 'mem0'
        assert kwargs['project_id'] == 'dark_factory'
        assert kwargs['agent_id'] == ARCHIVE_AGENT_ID
