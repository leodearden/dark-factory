"""The γ2 optional ``project_root`` guard on five escalation read and write tools.

Pins the guard table of ``plans/escalation-store-ambiguity-prd.md`` §6.3 on a
real ``create_server``: a mismatched or reconciliation-store assertion is
refused with ``{'error': ...}`` before any read or write, and an omitted
assertion, a matching one, or an identity-less server behaves exactly as
before.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _pending_tool_fixtures import _file
from fastmcp import Client, FastMCP
from mcp.types import TextContent

from escalation.queue import EscalationQueue
from escalation.server import create_server
from escalation.store_identity import StoreIdentity

READS = ('get_pending_escalations', 'get_escalation', 'get_task_escalation_history')
WRITES = ('escalate_info', 'escalate_blocker')
TOOLS = READS + WRITES
FILED_TASK = '77'
_FILING = {
    'task_id': FILED_TASK,
    'agent_role': 'implementer',
    'category': 'design_concern',
    'summary': 's',
}


async def call(server, name: str, **kw: Any) -> Any:
    tool = await server.get_tool(name)
    result = tool.fn(**kw)
    if inspect.isawaitable(result):
        result = await result
    return result


@pytest.fixture
def proj(tmp_path: Path) -> Path:
    path = tmp_path / 'proj'
    path.mkdir()
    return path


@pytest.fixture
def other(tmp_path: Path) -> Path:
    path = tmp_path / 'other'
    path.mkdir()
    return path


@pytest.fixture
def queue(proj: Path) -> EscalationQueue:
    return EscalationQueue(proj / 'esc')


@pytest.fixture
def seeded_id(queue: EscalationQueue) -> str:
    return _file(queue, '42').id


@pytest.fixture
def kwargs_for(seeded_id: str) -> Callable[[str], dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {
        'get_pending_escalations': {'task_id': '42'},
        'get_escalation': {'escalation_id': seeded_id},
        'get_task_escalation_history': {'task_id': '42'},
        'escalate_info': dict(_FILING),
        'escalate_blocker': dict(_FILING),
    }
    return lambda name: dict(table[name])


@pytest.fixture
def project_server(queue: EscalationQueue, proj: Path) -> FastMCP:
    identity = StoreIdentity(
        kind='project', queue_dir=proj / 'esc', project_id='proj', project_root=proj,
    )
    return create_server(queue, startup_sweep=False, store_identity=identity)


@pytest.fixture
def bare_server(queue: EscalationQueue) -> FastMCP:
    return create_server(queue, startup_sweep=False, store_identity=None)


@pytest.fixture
def recon_queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'recon')


@pytest.fixture
def recon_server(recon_queue: EscalationQueue) -> FastMCP:
    identity = StoreIdentity(
        kind='reconciliation', queue_dir=recon_queue.queue_dir, project_id=None, project_root=None,
    )
    return create_server(recon_queue, startup_sweep=False, store_identity=identity)


@pytest.fixture
def matching_spellings(tmp_path: Path, proj: Path) -> list[str]:
    link = tmp_path / 'link'
    link.symlink_to(proj, target_is_directory=True)
    return [str(proj), str(proj) + '/', str(proj / 'x' / '..'), str(link)]


class TestProjectStoreMismatch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('name', TOOLS)
    async def test_mismatched_root_is_refused_before_any_read_or_write(
        self,
        name: str,
        project_server: FastMCP,
        queue: EscalationQueue,
        kwargs_for: Callable[[str], dict[str, Any]],
        seeded_id: str,
        proj: Path,
        other: Path,
    ) -> None:
        result = await call(project_server, name, **kwargs_for(name), project_root=str(other))

        assert set(result) == {'error'}
        assert 'project_root mismatch' in result['error']
        assert str(other) in result['error']
        assert str(proj) in result['error']
        assert seeded_id not in json.dumps(result)
        assert queue.get_by_task(FILED_TASK) == []


class TestProjectStoreMatch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('name', READS)
    async def test_matching_read_returns_the_omitted_result(
        self,
        name: str,
        project_server: FastMCP,
        kwargs_for: Callable[[str], dict[str, Any]],
        matching_spellings: list[str],
    ) -> None:
        omitted = await call(project_server, name, **kwargs_for(name))

        for spelling in matching_spellings:
            asserted = await call(project_server, name, **kwargs_for(name), project_root=spelling)
            assert asserted == omitted, spelling

    @pytest.mark.asyncio
    @pytest.mark.parametrize('name', WRITES)
    async def test_matching_write_is_filed(
        self,
        name: str,
        project_server: FastMCP,
        queue: EscalationQueue,
        kwargs_for: Callable[[str], dict[str, Any]],
        matching_spellings: list[str],
    ) -> None:
        for spelling in matching_spellings:
            result = await call(project_server, name, **kwargs_for(name), project_root=spelling)
            assert result['status'] == 'queued', spelling
            assert queue.get(result['id']) is not None


class TestReconciliationStore:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('name', TOOLS)
    async def test_any_assertion_is_refused_as_the_reconciliation_store(
        self,
        name: str,
        recon_server: FastMCP,
        recon_queue: EscalationQueue,
        kwargs_for: Callable[[str], dict[str, Any]],
        proj: Path,
        other: Path,
    ) -> None:
        for asserted in (str(other), str(proj)):
            result = await call(recon_server, name, **kwargs_for(name), project_root=asserted)

            assert set(result) == {'error'}
            assert 'reconciliation store' in result['error']
            assert str(recon_queue.queue_dir) in result['error']
            assert 'project_root mismatch' not in result['error']
        assert recon_queue.get_by_task(FILED_TASK) == []


class TestIdentitylessServer:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('name', READS)
    async def test_any_assertion_reads_as_if_omitted(
        self,
        name: str,
        bare_server: FastMCP,
        kwargs_for: Callable[[str], dict[str, Any]],
        other: Path,
    ) -> None:
        omitted = await call(bare_server, name, **kwargs_for(name))

        assert await call(bare_server, name, **kwargs_for(name), project_root=str(other)) == omitted

    @pytest.mark.asyncio
    @pytest.mark.parametrize('name', WRITES)
    async def test_any_assertion_files(
        self,
        name: str,
        bare_server: FastMCP,
        queue: EscalationQueue,
        kwargs_for: Callable[[str], dict[str, Any]],
        other: Path,
    ) -> None:
        result = await call(bare_server, name, **kwargs_for(name), project_root=str(other))

        assert result['status'] == 'queued'
        assert queue.get(result['id']) is not None


class TestOmittedAssertionIsUnchanged:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('name', READS)
    async def test_identity_does_not_change_an_omitted_read(
        self,
        name: str,
        project_server: FastMCP,
        bare_server: FastMCP,
        kwargs_for: Callable[[str], dict[str, Any]],
    ) -> None:
        assert await call(project_server, name, **kwargs_for(name)) == await call(
            bare_server, name, **kwargs_for(name)
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('name', READS)
    async def test_empty_assertion_is_omitted(
        self,
        name: str,
        project_server: FastMCP,
        kwargs_for: Callable[[str], dict[str, Any]],
    ) -> None:
        assert await call(project_server, name, **kwargs_for(name), project_root='') == await call(
            project_server, name, **kwargs_for(name)
        )


class TestWireShape:
    @pytest.mark.asyncio
    async def test_omitted_pending_read_keeps_its_list_shape(
        self, project_server: FastMCP, seeded_id: str,
    ) -> None:
        async with Client(project_server) as client:
            result = await client.call_tool('get_pending_escalations', {'task_id': '42'})

        assert result.structured_content is not None
        listed = result.structured_content['result']
        assert isinstance(listed, list)
        assert [e['id'] for e in listed] == [seeded_id]
        text = result.content[0]
        assert isinstance(text, TextContent)
        assert json.loads(text.text) == listed

    @pytest.mark.asyncio
    async def test_mismatched_pending_read_returns_the_error_over_the_wire(
        self, project_server: FastMCP, other: Path,
    ) -> None:
        async with Client(project_server) as client:
            result = await client.call_tool(
                'get_pending_escalations', {'task_id': '42', 'project_root': str(other)},
            )

        assert set(result.data) == {'error'}
        assert 'project_root mismatch' in result.data['error']

    @pytest.mark.asyncio
    async def test_reconciliation_write_returns_the_error_over_the_wire(
        self, recon_server: FastMCP, other: Path,
    ) -> None:
        async with Client(recon_server) as client:
            result = await client.call_tool(
                'escalate_info', {**_FILING, 'project_root': str(other)},
            )

        assert 'reconciliation store' in result.data['error']
