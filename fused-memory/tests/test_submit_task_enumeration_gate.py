"""The enumeration liveness gate at the real MCP task-tool boundary (task 3056).

Reify gate 5537 shipped enumerating a Mem0 entry deleted about eleven minutes
earlier, alongside a truncated 8-char id. These tests drive ``submit_task`` and
``update_task`` through ``create_mcp_server`` so the gate is exercised with
the preceding recon hooks (``execution_class_error``,
``recurring_gate_guard_error``) in the loop, and assert what the task
interceptor actually receives. The gate's own rule matrix lives in
``tests/reconciliation/test_enumeration_guard.py``.

Harness modelled on ``test_delete_memory_citation_guard.py``:
``create_mcp_server`` + ``_tool_manager.call_tool`` + ``_parse_result``.
``TestUpdatePersistence`` forwards the interceptor's ``update_task`` to a real
``SqliteTaskBackend`` so the backend's own metadata merge decides what is
stored, and asserts on the stored row rather than on the forwarded payload.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from reconciliation.enumeration_fixtures import (
    DELETED_0D5,
    FINDING_IDS,
    LIVE_21F,
    LIVE_23E,
    LIVE_B3F,
    LIVE_IDS,
    VALID_SCOPE,
    fake_memory_service,
)

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.server.tools import create_mcp_server

RECON_AGENT = 'recon-stage-task_knowledge_sync'

GATE_DESCRIPTION = '\n'.join([
    'THE FOUR ENTRIES:',
    f'- {LIVE_B3F} (observations_and_summaries)',
    f'- {DELETED_0D5} (procedural_knowledge), the likeliest deletion target',
])


def gate_5537_metadata(**overrides: Any) -> dict[str, Any]:
    return {
        'mem0_falsified_entries': [LIVE_B3F, LIVE_21F, LIVE_23E, DELETED_0D5],
        'execution_class': 'operational',
        'operational_mode': 'gate',
        'x_enumeration_scope': VALID_SCOPE,
        **overrides,
    }


def _parse_result(result):
    """Parse a FastMCP call_tool result (list of TextContent) into a dict."""
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch):
    """The tmp project_root is not a git checkout; pass it through unchanged."""
    monkeypatch.setattr(
        'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
    )


@pytest.fixture
def project_root(tmp_path) -> str:
    root = tmp_path / 'reify'
    root.mkdir()
    return str(root)


@pytest.fixture
def memory_service() -> AsyncMock:
    """get_memory_by_id resolves the three live entries and misses the rest."""
    return fake_memory_service(live=LIVE_IDS)


@pytest.fixture
def task_interceptor() -> AsyncMock:
    interceptor = AsyncMock()
    interceptor.submit_task = AsyncMock(return_value={'ticket': 'tkt_x'})
    interceptor.update_task = AsyncMock(return_value={'success': True})
    interceptor.get_tasks = AsyncMock(return_value={'tasks': []})
    interceptor.get_task = AsyncMock(return_value={})
    return interceptor


@pytest.fixture
def server(memory_service, task_interceptor):
    return create_mcp_server(memory_service, task_interceptor=task_interceptor)


async def submit(
    server, project_root: str, metadata: dict[str, Any], *,
    agent_id: str = RECON_AGENT, description: str = GATE_DESCRIPTION,
) -> dict[str, Any]:
    return _parse_result(await server._tool_manager.call_tool('submit_task', {
        'project_root': project_root,
        'title': 'Human gate: decide the fate of the falsified entries',
        'description': description,
        'metadata': metadata,
        'agent_id': agent_id,
    }))


def forwarded(interceptor_method: AsyncMock) -> dict[str, Any]:
    interceptor_method.assert_awaited_once()
    assert interceptor_method.await_args is not None
    return dict(interceptor_method.await_args.kwargs)


class TestSubmitTask:

    @pytest.mark.asyncio
    async def test_gate_5537_is_filed_without_the_deleted_entry(
        self, server, project_root, task_interceptor,
    ):
        result = await submit(server, project_root, gate_5537_metadata())

        assert 'error' not in result
        kwargs = forwarded(task_interceptor.submit_task)
        metadata = kwargs['metadata']
        assert metadata['mem0_falsified_entries'] == list(LIVE_IDS)
        assert metadata['x_liveness_exceptions'] == [{
            'memory_id': DELETED_0D5,
            'path': 'mem0_falsified_entries[3]',
            'outcome': 'absent',
        }]
        assert 'x_liveness_reverified_at' in metadata
        deleted_line = kwargs['description'].split('\n')[2]
        assert 'already deleted, verified' in deleted_line
        assert result['enumeration_liveness']['dropped_absent'] == 1
        assert result['enumeration_liveness']['verified'] == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'reply',
        [{'success': False, 'error_type': 'CuratorRejected'}, {}],
        ids=['success-false', 'empty'],
    )
    async def test_stats_ride_back_only_on_a_write_that_landed(
        self, server, project_root, task_interceptor, reply,
    ):
        task_interceptor.submit_task.return_value = reply

        result = await submit(server, project_root, gate_5537_metadata())

        assert 'enumeration_liveness' not in result

    @pytest.mark.asyncio
    async def test_a_truncated_fragment_is_rejected_before_any_lookup(
        self, server, project_root, task_interceptor, memory_service,
    ):
        metadata = gate_5537_metadata(
            mem0_falsified_entries=[LIVE_B3F, LIVE_23E, DELETED_0D5, '21f78de1'],
        )

        result = await submit(server, project_root, metadata)

        assert result['error_type'] == 'EnumeratedIdTruncated'
        task_interceptor.submit_task.assert_not_awaited()
        memory_service.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('execution_class', ['operational', 'decision'])
    async def test_an_enumerating_gate_without_a_scope_is_rejected(
        self, server, project_root, task_interceptor, execution_class,
    ):
        metadata = gate_5537_metadata(execution_class=execution_class)
        del metadata['x_enumeration_scope']
        if execution_class == 'decision':
            del metadata['operational_mode']

        result = await submit(server, project_root, metadata)

        assert result['error_type'] == 'EnumerationScopeMissing'
        task_interceptor.submit_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_non_recon_caller_reaches_the_interceptor_unchanged(
        self, server, project_root, task_interceptor, memory_service,
    ):
        result = await submit(
            server, project_root, gate_5537_metadata(),
            agent_id='claude-interactive',
        )

        assert 'enumeration_liveness' not in result
        kwargs = forwarded(task_interceptor.submit_task)
        assert kwargs['description'] == GATE_DESCRIPTION
        assert kwargs['metadata']['mem0_falsified_entries'] == [
            LIVE_B3F, LIVE_21F, LIVE_23E, DELETED_0D5,
        ]
        assert 'x_liveness_reverified_at' not in kwargs['metadata']
        memory_service.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_lookups_use_the_project_id_of_project_root(
        self, server, project_root, memory_service,
    ):
        await submit(server, project_root, gate_5537_metadata())

        project_ids = {
            call.args[0] for call in memory_service.get_memory_by_id.await_args_list
        }
        assert project_ids == {'reify'}


MERGE: Mapping[str, Any] = {'metadata_mode': 'merge'}


async def update(
    server, project_root: str, *,
    metadata: dict[str, Any] | None = None,
    description: str | None = None,
    agent_id: str = RECON_AGENT,
    task_id: str = '5537',
    merge_flags: Mapping[str, Any] = MERGE,
) -> dict[str, Any]:
    """*merge_flags* ride along with *metadata*; ``{}`` omits both flags."""
    arguments: dict[str, Any] = {
        'id': task_id, 'project_root': project_root, 'agent_id': agent_id,
    }
    if metadata is not None:
        arguments |= {'metadata': metadata, **merge_flags}
    if description is not None:
        arguments['description'] = description
    return _parse_result(await server._tool_manager.call_tool('update_task', arguments))


def forwarded_update(task_interceptor: AsyncMock) -> tuple[dict[str, Any], str | None]:
    """The metadata (decoded from the JSON string the handler sends) and the
    description the interceptor's update_task received."""
    kwargs = forwarded(task_interceptor.update_task)
    return json.loads(kwargs['metadata']), kwargs['description']


class TestUpdateTask:

    @pytest.mark.asyncio
    async def test_a_re_asserted_enumeration_is_cleaned_stamped_and_annotated(
        self, server, project_root, task_interceptor,
    ):
        result = await update(
            server, project_root,
            metadata={'mem0_falsified_entries': [*LIVE_IDS, DELETED_0D5]},
            description=GATE_DESCRIPTION,
        )

        assert 'error' not in result
        metadata, description = forwarded_update(task_interceptor)
        assert metadata['mem0_falsified_entries'] == list(LIVE_IDS)
        assert [r['outcome'] for r in metadata['x_liveness_exceptions']] == ['absent']
        assert 'x_liveness_reverified_at' in metadata
        assert description is not None
        assert 'already deleted, verified' in description.split('\n')[2]
        assert result['enumeration_liveness']['dropped_absent'] == 1

    @pytest.mark.asyncio
    async def test_a_truncated_fragment_is_recorded_not_rejected(
        self, server, project_root, task_interceptor,
    ):
        result = await update(
            server, project_root,
            metadata={'mem0_falsified_entries': [LIVE_B3F, LIVE_23E, '21f78de1']},
        )

        assert 'error' not in result
        metadata, _ = forwarded_update(task_interceptor)
        assert metadata['mem0_falsified_entries'] == [LIVE_B3F, LIVE_23E, '21f78de1']
        assert {
            'memory_id': '21f78de1',
            'path': 'mem0_falsified_entries[2]',
            'outcome': 'truncated',
        } in metadata['x_liveness_exceptions']
        assert result['enumeration_liveness']['truncated'] == 1

    @pytest.mark.asyncio
    async def test_an_inconclusive_lookup_keeps_the_id_and_the_write(
        self, server, project_root, task_interceptor, memory_service,
    ):
        async def _flaky(_project_id: str, memory_id: str) -> dict[str, Any] | None:
            if memory_id == LIVE_21F:
                raise TimeoutError('qdrant timeout')
            return {'id': memory_id} if memory_id in LIVE_IDS else None

        memory_service.get_memory_by_id.side_effect = _flaky

        result = await update(
            server, project_root,
            metadata={'mem0_falsified_entries': [LIVE_B3F, LIVE_21F]},
        )

        assert 'error' not in result
        metadata, _ = forwarded_update(task_interceptor)
        assert metadata['mem0_falsified_entries'] == [LIVE_B3F, LIVE_21F]
        assert metadata['x_liveness_exceptions'] == [{
            'memory_id': LIVE_21F,
            'path': 'mem0_falsified_entries[1]',
            'outcome': 'inconclusive',
            'error_type': 'TimeoutError',
        }]

    @pytest.mark.asyncio
    async def test_scope_is_not_required_on_update(
        self, server, project_root, task_interceptor,
    ):
        metadata = gate_5537_metadata()
        del metadata['x_enumeration_scope']

        result = await update(server, project_root, metadata=metadata)

        assert 'error' not in result
        task_interceptor.update_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_malformed_legacy_scope_is_rejected_on_update(
        self, server, project_root, task_interceptor,
    ):
        metadata = gate_5537_metadata(
            x_enumeration_scope='enumerated over procedural_knowledge ONLY',
        )

        result = await update(server, project_root, metadata=metadata)

        assert result['error_type'] == 'EnumerationScopeMalformed'
        assert set(result['expected_shape']) == {
            'stores', 'categories', 'exhaustive', 'note',
        }
        task_interceptor.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_non_recon_update_bypasses_the_gate(
        self, server, project_root, task_interceptor, memory_service,
    ):
        result = await update(
            server, project_root,
            metadata={'mem0_falsified_entries': [LIVE_B3F, DELETED_0D5]},
            agent_id='claude-interactive',
        )

        assert 'enumeration_liveness' not in result
        metadata, _ = forwarded_update(task_interceptor)
        assert metadata == {'mem0_falsified_entries': [LIVE_B3F, DELETED_0D5]}
        memory_service.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_gate_filters_only_the_outgoing_payload(
        self, server, project_root, task_interceptor, memory_service,
    ):
        result = await update(server, project_root, description=GATE_DESCRIPTION)

        assert 'enumeration_liveness' not in result
        kwargs = forwarded(task_interceptor.update_task)
        assert kwargs['description'] == GATE_DESCRIPTION
        assert kwargs['metadata'] is None
        memory_service.get_memory_by_id.assert_not_awaited()
        task_interceptor.get_task.assert_not_awaited()


# --------------------------------------------------------------------------- #
# What an update actually persists under the backend's real metadata merge
# --------------------------------------------------------------------------- #

SEED_STAMP = '2026-01-01T00:00:00+00:00'

# Gate 5537 as stored after an earlier write: the four-entry enumeration, a
# stale stamp, a stale exception for an unrelated id, and attached hints.
SEED_METADATA: dict[str, Any] = {
    'mem0_falsified_entries': [*LIVE_IDS, DELETED_0D5],
    'x_liveness_reverified_at': SEED_STAMP,
    'x_liveness_exceptions': [{
        'memory_id': '9a4e568b-0000-4000-8000-000000000000',
        'path': 'x_old[0]',
        'outcome': 'absent',
    }],
    'memory_hints': {'entities': ['E1'], 'queries': ['Q1']},
}


@pytest_asyncio.fixture
async def backend(project_root) -> AsyncIterator[SqliteTaskBackend]:
    task_backend = SqliteTaskBackend(TaskmasterConfig(project_root=project_root))
    await task_backend.start()
    try:
        yield task_backend
    finally:
        await task_backend.close()


@pytest_asyncio.fixture
async def seeded_task_id(backend, project_root) -> str:
    added = await backend.add_task(
        project_root,
        title='Human gate: decide the fate of the falsified entries',
        description=GATE_DESCRIPTION,
        metadata=json.dumps(SEED_METADATA),
    )
    return str(added['id'])


@pytest.fixture
def persisting_server(backend, memory_service, task_interceptor):
    """The MCP server over an interceptor whose update_task writes through the
    real backend, so none of the TaskInterceptor's unrelated recon gates can
    reject and the stored row is the backend's own merge of the payload."""

    async def _write_through(**kwargs: Any) -> dict[str, Any]:
        rest = {
            key: value for key, value in kwargs.items()
            if key not in ('agent_id', 'client_op_id')
        }
        await backend.update_task(**rest)
        return {'success': True}

    task_interceptor.update_task.side_effect = _write_through
    return create_mcp_server(memory_service, task_interceptor=task_interceptor)


async def stored_metadata(
    backend: SqliteTaskBackend, project_root: str, task_id: str,
) -> dict[str, Any]:
    return (await backend.get_task(task_id, project_root))['metadata']


class TestUpdatePersistence:
    """The gate's report must describe what the backend actually stores."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'merge_flags',
        [{'append': True}, {'metadata_mode': 'additive'}, {'append': False}],
        ids=['append', 'additive', 'bare-append-false'],
    )
    async def test_a_non_overwriting_enumeration_update_is_refused_unwritten(
        self, persisting_server, backend, project_root, seeded_task_id,
        task_interceptor, merge_flags,
    ):
        seeded = await stored_metadata(backend, project_root, seeded_task_id)

        result = await update(
            persisting_server, project_root, task_id=seeded_task_id,
            metadata={'mem0_falsified_entries': [*LIVE_IDS, DELETED_0D5]},
            merge_flags=merge_flags,
        )

        assert result['error_type'] == 'EnumerationUpdateNotOverwriting'
        assert result['enumerations'] == ['mem0_falsified_entries']
        assert 'enumeration_liveness' not in result
        task_interceptor.update_task.assert_not_awaited()
        assert seeded == {**SEED_METADATA, 'pending_since': seeded['pending_since']}
        assert await stored_metadata(backend, project_root, seeded_task_id) == seeded

    @pytest.mark.asyncio
    async def test_a_contradictory_merge_append_pair_is_refused_by_the_backend(
        self, persisting_server, backend, project_root, seeded_task_id,
    ):
        seeded = await stored_metadata(backend, project_root, seeded_task_id)

        result = await update(
            persisting_server, project_root, task_id=seeded_task_id,
            metadata={'mem0_falsified_entries': [*LIVE_IDS, DELETED_0D5]},
            merge_flags={'metadata_mode': 'merge', 'append': True},
        )

        assert 'error' in result
        assert 'enumeration_liveness' not in result
        assert await stored_metadata(backend, project_root, seeded_task_id) == seeded

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'merge_flags', [MERGE, {}], ids=['merge', 'flags-omitted'],
    )
    async def test_an_overwriting_update_stores_what_the_gate_reports(
        self, persisting_server, backend, project_root, seeded_task_id,
        merge_flags,
    ):
        result = await update(
            persisting_server, project_root, task_id=seeded_task_id,
            metadata={'mem0_falsified_entries': [*LIVE_IDS, DELETED_0D5]},
            merge_flags=merge_flags,
        )

        assert 'error' not in result
        assert result['enumeration_liveness']['dropped_absent'] == 1
        stored = await stored_metadata(backend, project_root, seeded_task_id)
        assert stored['mem0_falsified_entries'] == list(LIVE_IDS)
        stamp = stored['x_liveness_reverified_at']
        assert stamp != SEED_STAMP
        assert datetime.fromisoformat(stamp).utcoffset() == timedelta(0)
        assert stored['x_liveness_exceptions'] == [{
            'memory_id': DELETED_0D5,
            'path': 'mem0_falsified_entries[3]',
            'outcome': 'absent',
        }]

    @pytest.mark.asyncio
    async def test_a_replace_read_modify_write_keeps_the_stored_exceptions(
        self, persisting_server, backend, project_root, seeded_task_id,
    ):
        resent = {
            **await stored_metadata(backend, project_root, seeded_task_id),
            'mem0_falsified_entries': list(LIVE_IDS),
        }

        result = await update(
            persisting_server, project_root, task_id=seeded_task_id,
            metadata=resent, merge_flags={'metadata_mode': 'replace'},
        )

        assert 'error' not in result
        stored = await stored_metadata(backend, project_root, seeded_task_id)
        assert stored['mem0_falsified_entries'] == list(LIVE_IDS)
        assert stored['x_liveness_exceptions'] == SEED_METADATA['x_liveness_exceptions']
        assert stored['x_liveness_reverified_at'] != SEED_STAMP
        assert stored['memory_hints'] == SEED_METADATA['memory_hints']

    @pytest.mark.asyncio
    async def test_a_stage2_memory_hints_attach_stays_additive(
        self, persisting_server, backend, project_root, seeded_task_id,
        memory_service,
    ):
        result = await update(
            persisting_server, project_root, task_id=seeded_task_id,
            metadata={'memory_hints': {'entities': ['E2'], 'queries': ['Q2']}},
            merge_flags={'append': True},
        )

        assert 'error' not in result
        memory_service.get_memory_by_id.assert_not_awaited()
        stored = await stored_metadata(backend, project_root, seeded_task_id)
        assert stored['memory_hints'] == {
            'entities': ['E1', 'E2'], 'queries': ['Q1', 'Q2'],
        }

    @pytest.mark.asyncio
    async def test_an_append_to_an_unproven_id_list_stays_legal(
        self, persisting_server, backend, project_root, seeded_task_id,
    ):
        result = await update(
            persisting_server, project_root, task_id=seeded_task_id,
            metadata={'origin_finding_ids': list(FINDING_IDS)},
            merge_flags={'append': True},
        )

        assert 'error' not in result
        stored = await stored_metadata(backend, project_root, seeded_task_id)
        assert stored['origin_finding_ids'] == FINDING_IDS
