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
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import _project_root_stores as stores
import pytest
from _pending_tool_fixtures import _file
from _project_root_stores import storm_records
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


async def assert_refused(
    server, kwargs_for: Callable[[str], dict[str, Any]], root: str, names: Iterable[str],
) -> None:
    for name in names:
        result = await call(server, name, **kwargs_for(name), project_root=root)
        assert set(result) == {'error'}, name


@pytest.fixture
def proj(tmp_path: Path) -> Path:
    return stores.make_root(tmp_path, 'proj')


@pytest.fixture
def other(tmp_path: Path) -> Path:
    return stores.make_root(tmp_path, 'other')


@pytest.fixture
def queue(proj: Path) -> EscalationQueue:
    return stores.project_queue(proj)


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
def project_identity(queue: EscalationQueue, proj: Path) -> StoreIdentity:
    return stores.project_identity(queue, proj)


@pytest.fixture
def project_server(queue: EscalationQueue, project_identity: StoreIdentity) -> FastMCP:
    return create_server(queue, startup_sweep=False, store_identity=project_identity)


@pytest.fixture
def bare_server(queue: EscalationQueue) -> FastMCP:
    return create_server(queue, startup_sweep=False, store_identity=None)


@pytest.fixture
def recon_queue(tmp_path: Path) -> EscalationQueue:
    return stores.recon_queue(tmp_path)


@pytest.fixture
def recon_server(recon_queue: EscalationQueue) -> FastMCP:
    return create_server(
        recon_queue, startup_sweep=False, store_identity=stores.recon_identity(recon_queue),
    )


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


class TestMismatchStormEscape:
    @pytest.mark.asyncio
    async def test_fifth_mismatch_files_once_and_the_rest_of_the_burst_does_not_refile(
        self,
        project_server: FastMCP,
        queue: EscalationQueue,
        kwargs_for: Callable[[str], dict[str, Any]],
        other: Path,
    ) -> None:
        first_four = (
            'get_pending_escalations',
            'get_escalation',
            'escalate_info',
            'get_task_escalation_history',
        )
        await assert_refused(project_server, kwargs_for, str(other), first_four)
        assert storm_records(queue) == []

        await assert_refused(project_server, kwargs_for, str(other), ['escalate_blocker'])
        records = storm_records(queue)
        assert len(records) == 1
        assert str(other) in records[0].summary
        assert str(queue.queue_dir) in records[0].summary

        await assert_refused(project_server, kwargs_for, str(other), ['get_escalation'])
        records = storm_records(queue)
        assert len(records) == 1
        assert records[0].dedupe_count == 0

        await assert_refused(project_server, kwargs_for, str(other), TOOLS[1:])
        records = storm_records(queue)
        assert len(records) == 1
        assert records[0].dedupe_count == 0

    @pytest.mark.asyncio
    async def test_the_storm_lands_in_the_reconciliation_queue(
        self,
        recon_server: FastMCP,
        recon_queue: EscalationQueue,
        kwargs_for: Callable[[str], dict[str, Any]],
        other: Path,
    ) -> None:
        await assert_refused(recon_server, kwargs_for, str(other), TOOLS)

        records = storm_records(recon_queue)
        assert len(records) == 1
        assert 'reconciliation' in records[0].summary

    @pytest.mark.asyncio
    async def test_streaks_are_per_asserted_root(
        self,
        tmp_path: Path,
        project_server: FastMCP,
        queue: EscalationQueue,
        kwargs_for: Callable[[str], dict[str, Any]],
    ) -> None:
        for root in (tmp_path / 'other-a', tmp_path / 'other-b'):
            await assert_refused(project_server, kwargs_for, str(root), TOOLS[:4])

        assert storm_records(queue) == []

    @pytest.mark.asyncio
    async def test_threshold_is_tunable(
        self,
        queue: EscalationQueue,
        project_identity: StoreIdentity,
        kwargs_for: Callable[[str], dict[str, Any]],
        other: Path,
    ) -> None:
        server = create_server(
            queue, startup_sweep=False, store_identity=project_identity,
            mismatch_streak_threshold=3,
        )

        await assert_refused(server, kwargs_for, str(other), READS[:2])
        assert storm_records(queue) == []
        await assert_refused(server, kwargs_for, str(other), READS[2:])
        assert len(storm_records(queue)) == 1

    def test_invalid_threshold_fails_at_construction_without_an_identity(
        self, queue: EscalationQueue,
    ) -> None:
        with pytest.raises(ValueError, match='mismatch_streak'):
            create_server(
                queue, startup_sweep=False, store_identity=None, mismatch_streak_threshold=0,
            )

    @pytest.mark.asyncio
    async def test_matching_and_omitted_calls_never_feed_the_streak(
        self,
        project_server: FastMCP,
        queue: EscalationQueue,
        kwargs_for: Callable[[str], dict[str, Any]],
        proj: Path,
    ) -> None:
        for name in READS * 4:
            await call(project_server, name, **kwargs_for(name), project_root=str(proj))
            await call(project_server, name, **kwargs_for(name))

        assert storm_records(queue) == []

    @pytest.mark.asyncio
    async def test_identityless_server_never_feeds_the_streak(
        self,
        bare_server: FastMCP,
        queue: EscalationQueue,
        kwargs_for: Callable[[str], dict[str, Any]],
        other: Path,
    ) -> None:
        for name in READS * 4:
            await call(bare_server, name, **kwargs_for(name), project_root=str(other))

        assert storm_records(queue) == []
