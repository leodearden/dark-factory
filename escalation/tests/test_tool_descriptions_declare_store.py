"""Every escalation tool description declares which store answers it.

The γ3 contract of ``plans/escalation-store-ambiguity-prd.md`` §6.2 (boundary
rows B4/B5), read off the wire through ``fastmcp.Client``: a caller holding an
empty result can tell from the same session's tool listing which store said so.

``expected_block`` is this test's OWN copy of the §6.2 text, never imported
from the server, so the contract and its renderer cannot drift together.
Docstring prose is never pinned; it is compared against an identity-less server.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import Client, FastMCP

from escalation.queue import EscalationQueue
from escalation.server import create_server
from escalation.store_identity import StoreIdentity


def expected_block(queue_dir: Path, kind: str, project: str) -> str:
    return '\n'.join(
        [
            f'THIS ENDPOINT SERVES: {queue_dir} (kind={kind}, project={project}).',
            'It contains ONLY escalations for that store. A task id belonging to any other',
            'project returns an empty result because the record is not HERE — not because it',
            'does not exist. To ask whether an escalation was EVER filed for a task',
            '(including resolved/archived), use get_task_escalation_history.',
        ]
    )


def recon_identity(queue: EscalationQueue) -> StoreIdentity:
    return StoreIdentity(
        kind='reconciliation', queue_dir=queue.queue_dir, project_id=None, project_root=None
    )


def project_identity(queue: EscalationQueue, root: Path) -> StoreIdentity:
    return StoreIdentity(
        kind='project', queue_dir=queue.queue_dir, project_id='test-project', project_root=root
    )


def identity_for(kind: str, queue: EscalationQueue, root: Path) -> StoreIdentity:
    return recon_identity(queue) if kind == 'reconciliation' else project_identity(queue, root)


def identity_server(queue: EscalationQueue, identity: StoreIdentity | None) -> FastMCP:
    return create_server(queue, startup_sweep=False, store_identity=identity)


async def listed_descriptions(server: FastMCP) -> dict[str, str | None]:
    async with Client(server) as client:
        return {tool.name: tool.description for tool in await client.list_tools()}


KINDS = pytest.mark.parametrize(
    ('kind', 'rendered_project'), [('project', 'test-project'), ('reconciliation', 'n/a')]
)


@KINDS
@pytest.mark.asyncio
async def test_every_listed_tool_ends_with_the_store_identity_block(
    tmp_path: Path, kind: str, rendered_project: str
) -> None:
    queue = EscalationQueue(tmp_path / 'esc')
    identity = identity_for(kind, queue, tmp_path)
    block = expected_block(identity.queue_dir, kind, rendered_project)

    descriptions = await listed_descriptions(identity_server(queue, identity))

    assert descriptions
    for name, description in descriptions.items():
        assert description is not None, name
        assert description.endswith('\n\n' + block), name


@KINDS
@pytest.mark.asyncio
async def test_the_block_is_purely_additive_to_each_docstring(
    tmp_path: Path, kind: str, rendered_project: str
) -> None:
    queue = EscalationQueue(tmp_path / 'with-identity')
    identity = identity_for(kind, queue, tmp_path)
    block = expected_block(identity.queue_dir, kind, rendered_project)

    with_identity = await listed_descriptions(identity_server(queue, identity))
    without_identity = await listed_descriptions(
        identity_server(EscalationQueue(tmp_path / 'without-identity'), None)
    )

    assert set(with_identity) == set(without_identity)
    for name, description in with_identity.items():
        assert description == f'{without_identity[name]}\n\n{block}', name


@pytest.mark.asyncio
async def test_a_server_without_an_identity_renders_no_block(tmp_path: Path) -> None:
    omitted = create_server(EscalationQueue(tmp_path / 'omitted'), startup_sweep=False)
    explicit_none = identity_server(EscalationQueue(tmp_path / 'explicit-none'), None)

    for server in (omitted, explicit_none):
        descriptions = await listed_descriptions(server)
        assert descriptions
        for name, description in descriptions.items():
            assert 'THIS ENDPOINT SERVES' not in (description or ''), name


@pytest.mark.asyncio
async def test_an_empty_pending_read_is_attributable_to_the_store_that_answered(
    tmp_path: Path,
) -> None:
    queue = EscalationQueue(tmp_path / 'recon-esc')
    identity = recon_identity(queue)

    async with Client(identity_server(queue, identity)) as client:
        result = await client.call_tool('get_pending_escalations', {'task_id': '5534'})
        listed = {tool.name: tool.description for tool in await client.list_tools()}

    assert isinstance(result.data, list)
    assert result.data == []
    description = listed['get_pending_escalations'] or ''
    assert str(identity.queue_dir) in description
    assert 'kind=reconciliation' in description


@pytest.mark.asyncio
async def test_the_cited_history_tool_is_registered_on_the_same_server(
    tmp_path: Path,
) -> None:
    queue = EscalationQueue(tmp_path / 'esc')

    descriptions = await listed_descriptions(identity_server(queue, recon_identity(queue)))

    assert 'get_task_escalation_history' in descriptions
    assert 'get_task_escalation_history' in (descriptions['get_pending_escalations'] or '')


@pytest.mark.asyncio
async def test_tools_still_dispatch_on_an_identity_bearing_server(tmp_path: Path) -> None:
    queue = EscalationQueue(tmp_path / 'esc')

    async with Client(identity_server(queue, project_identity(queue, tmp_path))) as client:
        filed = await client.call_tool(
            'escalate_info',
            {
                'task_id': '3167',
                'agent_role': 'implementer',
                'category': 'cleanup_needed',
                'summary': 's',
            },
        )
        history = await client.call_tool('get_task_escalation_history', {'task_id': '3167'})

    assert filed.data['status'] == 'queued'
    assert len(queue.get_by_task('3167')) == 1
    assert history.data['count'] == 1
