"""Boundary row 1, minus the hold, for metadata.followup_items at the submit boundary (task 8414).

plans/follow-up-filing-shape-prd.md §4.1 and §3 decisions 1 and 3: a bundle's R4 keys are derived
server-side on both creation paths, and a bundle breaking a create invariant is refused before a
ticket or a row is written. Each row drives the public MCP tools against a real SQLite store, ticket
store and interceptor.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from shared.followup_items import (
    FOLLOWUP_ITEMS_METADATA_KEY,
    MAX_FOLLOWUP_ITEMS,
    UNPARENTED_ORIGIN,
    FollowupItem,
    item_key,
    with_derived_bundle_keys,
)

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig, TaskmasterConfig
from fused_memory.middleware.curator_types import CuratorDecision, PreparedCandidate
from fused_memory.middleware.task_curator import TaskCurator
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server

_HEX16 = re.compile(r'[0-9a-f]{16}')


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """This suite's project roots are tmp_path directories, not git checkouts."""
    monkeypatch.setattr('fused_memory.server.tools.resolve_main_checkout', lambda p: str(p))


@dataclass
class Stack:
    root: str
    backend: SqliteTaskBackend
    server: Any

    async def call(self, tool: str, **args: Any) -> Any:
        return await self.server._tool_manager.call_tool(tool, {'project_root': self.root, **args})

    async def task_count(self) -> int:
        return len((await self.call('get_tasks'))['tasks'])

    async def metadata(self, task_id: str) -> dict[str, Any]:
        result = await self.call('get_task', id=task_id)
        assert 'error' not in result, result
        return result['metadata']

    async def file_normal(self, **args: Any) -> str:
        submitted = await self.call('submit_task', **args)
        assert 'ticket' in submitted, submitted
        resolved = await self.call('resolve_ticket', ticket=submitted['ticket'], timeout_seconds=30)
        assert resolved.get('status') == 'created', resolved
        return str(resolved['task_id'])

    async def file_planned(self, **args: Any) -> str:
        result = await self.call('submit_task', planning_mode=True, **args)
        assert 'error' not in result, result
        return str(result['task_id'])


@pytest_asyncio.fixture
async def stack(tmp_path: Path) -> AsyncIterator[Stack]:
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)), task_metadata_enforce=True)
    await backend.start()
    event_buffer = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
    await event_buffer.initialize()
    ticket_store = TicketStore(tmp_path / 'tickets.db')
    await ticket_store.initialize()
    config = FusedMemoryConfig()
    config.taskmaster = None
    config.curator = CuratorConfig()
    interceptor = TaskInterceptor(backend, None, event_buffer, config, ticket_store=ticket_store)
    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
    try:
        yield Stack(root=str(tmp_path / 'proj'), backend=backend, server=server)
    finally:
        await interceptor.close()
        await event_buffer.close()
        await backend.close()


@pytest.fixture
def curated(monkeypatch: pytest.MonkeyPatch) -> list[PreparedCandidate]:
    """Every curated candidate is created as filed; the bundles the curator was handed are kept."""
    handed: list[PreparedCandidate] = []

    async def _create_each(self, prepared, project_id, project_root):
        handed.extend(prepared)
        return [CuratorDecision(action='create', justification='followup-items test') for _ in prepared]

    monkeypatch.setattr(TaskCurator, 'curate_prepared', _create_each)
    return handed


_AREA_FILES = ['src/app/a.py', 'src/app/b.py', 'src/app/c.py']
_FIRST = {'summary': 'The retry loop swallows the timeout', 'files': ['src/app/a.py']}
_SECOND = {'summary': 'The cache key ignores the tenant', 'files': ['src/app/c.py', 'src/app/b.py']}
_FORGED_IDENTITY = {'key': 'forged0000000000', 'origin_task': '1', 'origin_escalation_id': 'esc-forged'}


def _bundle(*items: dict[str, Any], **top: Any) -> dict[str, Any]:
    return {
        'source': 'agent-followup',
        'spawn_context': 'agent_followup',
        'spawned_from': '8414',
        'files': list(_AREA_FILES),
        FOLLOWUP_ITEMS_METADATA_KEY: list(items),
        **top,
    }


# (a) and (b): an implementer-shaped bundle lands as one task with server-derived keys.


@pytest.mark.asyncio
async def test_an_implementer_bundle_lands_with_server_derived_keys(
    stack: Stack, curated: list[PreparedCandidate],
) -> None:
    task_id = await stack.file_normal(
        title='Follow-ups in src/app', description='Two observations in one area.',
        metadata=_bundle(
            {**_FIRST, **_FORGED_IDENTITY}, {**_SECOND, **_FORGED_IDENTITY},
            escalation_id='esc-forged', suggestion_hash='forged',
        ),
    )

    metadata = await stack.metadata(task_id)
    stored = metadata[FOLLOWUP_ITEMS_METADATA_KEY]
    assert len(stored) == 2
    for item, sent in zip(stored, (_FIRST, _SECOND), strict=True):
        assert item['origin_task'] == '8414'
        assert item['key'] == item_key(origin_task='8414', files=sent['files'], summary=sent['summary'])
    assert metadata['escalation_id'] == 'agent-followup-8414'
    assert metadata['suggestion_hash'] != 'forged'
    assert _HEX16.fullmatch(metadata['suggestion_hash'])
    assert metadata['declared_scope']['basis'] == 'submit'

    assert [bundle.candidate.items for bundle in curated] == [
        tuple(FollowupItem.model_validate(item) for item in stored),
    ]


# (c): server-derived keys make a re-filed bundle an R4 hit, whatever hashes the caller sent.


@pytest.mark.asyncio
async def test_re_filing_the_same_observations_is_an_idempotency_hit(
    stack: Stack, curated: list[PreparedCandidate],
) -> None:
    first = await stack.file_normal(
        title='Follow-ups in src/app', description='Two observations in one area.',
        metadata=_bundle(_FIRST, _SECOND, suggestion_hash='forged-first'),
    )

    submitted = await stack.call(
        'submit_task', title='A later agent re-files them', description='The same two, reordered.',
        metadata=_bundle(_SECOND, _FIRST, escalation_id='esc-other', suggestion_hash='forged-second'),
    )
    assert 'ticket' in submitted, submitted
    resolved = await stack.call('resolve_ticket', ticket=submitted['ticket'], timeout_seconds=30)

    assert (resolved['status'], resolved['reason'], str(resolved['task_id'])) == (
        'combined', 'idempotency_hit', first,
    )
    assert await stack.task_count() == 1


# (d): the planning-mode path derives the same keys on its deferred row.


@pytest.mark.asyncio
async def test_a_planning_mode_bundle_carries_the_derived_keys(stack: Stack) -> None:
    task_id = await stack.file_planned(
        title='Planned follow-ups in src/app', description='Two observations in one area.',
        metadata=_bundle(_FIRST, _SECOND, escalation_id='esc-forged', suggestion_hash='forged'),
    )

    task = await stack.call('get_task', id=task_id)
    expected = with_derived_bundle_keys(_bundle(_FIRST, _SECOND))
    assert task['status'] == 'deferred'
    for key in (FOLLOWUP_ITEMS_METADATA_KEY, 'escalation_id', 'suggestion_hash'):
        assert task['metadata'][key] == expected[key]


# (e): an unparented bundle has the unparented origin and no top-level escalation_id.


@pytest.mark.asyncio
async def test_an_unparented_bundle_has_no_escalation_id(
    stack: Stack, curated: list[PreparedCandidate],
) -> None:
    metadata = _bundle(_FIRST, escalation_id='esc-forged')
    metadata.pop('spawned_from')

    task_id = await stack.file_normal(
        title='An unparented follow-up', description='Filed by no task.', metadata=metadata,
    )

    stored = await stack.metadata(task_id)
    assert [item['origin_task'] for item in stored[FOLLOWUP_ITEMS_METADATA_KEY]] == [UNPARENTED_ORIGIN]
    assert 'escalation_id' not in stored


# (f): a bundle breaking a create invariant is refused synchronously, on both paths.

_REFUSALS = [
    pytest.param(
        [{'summary': f'observation {index}', 'files': ['src/app/a.py']} for index in range(MAX_FOLLOWUP_ITEMS + 1)],
        'item_count', None, id='too-many-items',
    ),
    pytest.param(
        [{'summary': 'one', 'files': ['src/app/a.py', 'src/app/undeclared.py']}],
        'undeclared_files', None, id='undeclared-file',
    ),
    pytest.param([_FIRST, dict(_FIRST)], 'duplicate_key', None, id='identical-items'),
    pytest.param([_FIRST, {**_SECOND, 'title': 'not a field'}], 'invalid_item', 1, id='unknown-field'),
]


@pytest.mark.asyncio
@pytest.mark.parametrize('planning_mode', [False, True])
@pytest.mark.parametrize('as_json', [False, True], ids=['object', 'json-string'])
@pytest.mark.parametrize(('items', 'reason', 'index'), _REFUSALS)
async def test_a_bundle_breaking_a_create_invariant_is_refused(
    stack: Stack, planning_mode: bool, as_json: bool,
    items: list[dict[str, Any]], reason: str, index: int | None,
) -> None:
    metadata = _bundle(*items)
    before = await stack.task_count()

    result = await stack.call(
        'submit_task', title='A refused bundle', description='Breaks a create invariant.',
        planning_mode=planning_mode, metadata=json.dumps(metadata) if as_json else metadata,
    )

    assert result['success'] is False, result
    assert (result['error'], result['error_type'], result['path']) == (
        'followup_items_refused', 'FollowupItemsRefusedError', f'metadata.{FOLLOWUP_ITEMS_METADATA_KEY}',
    ), result
    assert (result['reason'], result['index']) == (reason, index), result
    assert result['detail'], result
    assert 'ticket' not in result
    assert await stack.task_count() == before


# (g): a submission without followup_items keeps its caller-supplied R4 keys.


@pytest.mark.asyncio
async def test_a_steward_shaped_submission_keeps_its_keys(
    stack: Stack, curated: list[PreparedCandidate],
) -> None:
    task_id = await stack.file_normal(
        title='A steward follow-up', description='Filed from an escalation.',
        metadata={'escalation_id': 'esc-1', 'suggestion_hash': 'h', 'files': ['src/app/a.py']},
    )

    stored = await stack.metadata(task_id)
    assert (stored['escalation_id'], stored['suggestion_hash']) == ('esc-1', 'h')
    assert FOLLOWUP_ITEMS_METADATA_KEY not in stored


# (h): the registration reaches the fused-memory process, so a bundle is never censused.


@pytest.mark.asyncio
async def test_filing_a_bundle_emits_no_census_line_for_it(
    stack: Stack, curated: list[PreparedCandidate], caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger='fused_memory.backends.sqlite_task_backend'):
        caplog.clear()
        await stack.file_normal(
            title='Follow-ups in src/app', description='Two observations in one area.',
            metadata=_bundle(_FIRST, _SECOND),
        )

    census = [
        record.getMessage() for record in caplog.records
        if 'task_metadata.schema_warning' in record.getMessage()
        and FOLLOWUP_ITEMS_METADATA_KEY in record.getMessage()
    ]
    assert census == []


def test_the_interceptor_import_chain_registers_the_slice() -> None:
    """``-I`` isolates the interceptor's own import closure from what this pytest process loaded."""
    probe = (
        'import fused_memory.middleware.task_interceptor\n'
        'from shared.task_metadata import is_known_metadata_key\n'
        f'print(is_known_metadata_key({FOLLOWUP_ITEMS_METADATA_KEY!r}))\n'
    )
    result = subprocess.run(
        [sys.executable, '-I', '-c', probe], capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'True'
