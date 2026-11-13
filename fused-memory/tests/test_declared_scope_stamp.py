"""Boundary rows for metadata.declared_scope at the submit boundary (task 8413).

The rows are plans/follow-up-filing-shape-prd.md §5 rows 6 and 20 and §3 decision 7. Each one
drives the public MCP tools against a real SQLite store, ticket store and interceptor.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from shared.declared_scope import derive_declared_scope

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig, TaskmasterConfig
from fused_memory.middleware.curator_types import CuratorDecision
from fused_memory.middleware.task_curator import TaskCurator
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server


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

    async def seed(self, title: str, metadata: dict[str, Any]) -> str:
        created = await self.backend.add_task(
            project_root=self.root, title=title, metadata=json.dumps(metadata),
        )
        return str(created['id'])

    async def file_normal(self, **args: Any) -> str:
        submitted = await self.call('submit_task', **args)
        assert 'ticket' in submitted, submitted
        resolved = await self.call('resolve_ticket', ticket=submitted['ticket'], timeout_seconds=30)
        assert resolved.get('status') == 'created', resolved
        return resolved['task_id']

    async def file_planned(self, **args: Any) -> str:
        result = await self.call('submit_task', planning_mode=True, **args)
        assert 'error' not in result, result
        return result['task_id']


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


_FORGED = derive_declared_scope(['pkg/a.py'], 'forged', '', basis='submit').model_dump()


def _assert_write_authority(
    result: Any, task_id: str | None, reason: str, stored: dict[str, Any] | None = None,
) -> None:
    assert result['success'] is False, result
    assert result['error'] == 'declared_scope_write_authority', result
    assert result['error_type'] == 'DeclaredScopeWriteAuthorityError', result
    assert result['path'] == 'metadata.declared_scope', result
    assert (result['task_id'], result['reason'], result['stored']) == (task_id, reason, stored), result


# Row 6: a submission that supplies declared_scope is refused and nothing is filed.


@pytest.mark.asyncio
@pytest.mark.parametrize('planning_mode', [False, True])
@pytest.mark.parametrize('as_json', [False, True], ids=['object', 'json-string'])
async def test_submit_task_refuses_a_caller_declared_scope(
    stack: Stack, planning_mode: bool, as_json: bool,
) -> None:
    metadata: dict[str, Any] = {'declared_scope': _FORGED}
    before = await stack.task_count()

    result = await stack.call(
        'submit_task', title='row 6 forged scope', description='forged declared_scope',
        planning_mode=planning_mode, metadata=json.dumps(metadata) if as_json else metadata,
    )

    _assert_write_authority(result, None, 'added')
    assert 'ticket' not in result
    assert await stack.task_count() == before


@pytest.fixture
def curator_creates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every curated candidate is created as filed, so the row carries what the ticket carried."""

    async def _create_each(self, prepared, project_id, project_root):
        return [CuratorDecision(action='create', justification='declared-scope test') for _ in prepared]

    monkeypatch.setattr(TaskCurator, 'curate_prepared', _create_each)


def _files(count: int) -> list[str]:
    return [f'src/app/module_{index}.py' for index in range(count)]


# Row 20 (stamp half): every task filed through submit_task carries the stamp, on both paths.


@pytest.mark.asyncio
async def test_a_fileless_normal_submission_is_stamped_undeclared(
    stack: Stack, curator_creates: None,
) -> None:
    description, details = 'A fileless follow-up.', 'Some details.'

    task_id = await stack.file_normal(
        title='row 20 fileless', description=description, details=details, metadata={'x_note': 'n'},
    )

    assert (await stack.metadata(task_id))['declared_scope'] == {
        'band': 'undeclared',
        'files': 0,
        'text_chars': len(description) + len(details),
        'basis': 'submit',
        'rule': 'v1',
    }


@pytest.mark.asyncio
async def test_a_normal_submission_counts_its_declared_files(
    stack: Stack, curator_creates: None,
) -> None:
    task_id = await stack.file_normal(
        title='row 20 three files', description='Touches three modules.', metadata={'files': _files(3)},
    )

    stamp = (await stack.metadata(task_id))['declared_scope']
    assert (stamp['band'], stamp['files'], stamp['basis']) == ('moderate', 3, 'submit')


@pytest.mark.asyncio
async def test_a_planning_mode_submission_is_stamped_on_the_deferred_row(stack: Stack) -> None:
    task_id = await stack.file_planned(
        title='row 20 planned', description='One short planned change.', metadata={'files': _files(1)},
    )

    stamp = (await stack.metadata(task_id))['declared_scope']
    assert (stamp['band'], stamp['files'], stamp['basis']) == ('narrow', 1, 'submit')


@pytest.mark.asyncio
async def test_a_submission_without_metadata_is_stamped_undeclared(
    stack: Stack, curator_creates: None,
) -> None:
    task_id = await stack.file_normal(title='row 20 no metadata', description='No metadata at all.')

    assert (await stack.metadata(task_id))['declared_scope']['band'] == 'undeclared'


@pytest.mark.asyncio
async def test_only_tasks_filed_after_the_stamp_carry_it(stack: Stack, curator_creates: None) -> None:
    pre_deploy = await stack.seed('filed before the stamp existed', {'files': _files(2)})
    filed = {
        await stack.file_normal(title='post-deploy normal', description='Normal path.'),
        await stack.file_planned(title='post-deploy planned', description='Planning path.'),
    }

    assert 'declared_scope' not in await stack.metadata(pre_deploy)
    found = await stack.call('find_tasks_by_metadata', key='declared_scope')
    assert {str(row['task_id']) for row in found['results']} == filed


# Decision 7: update_task may not add, alter or remove the stamp; an unchanged echo passes.

_MODES = ['merge', 'additive', 'replace']


async def _stamped(stack: Stack, title: str, files: list[str]) -> tuple[str, dict[str, Any]]:
    task_id = await stack.file_planned(title=title, description=f'{title} body', metadata={'files': files})
    return task_id, await stack.metadata(task_id)


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', _MODES)
@pytest.mark.parametrize(
    'alteration', [{'band': 'moderate', 'files': 3}, {'basis': 'combine'}], ids=['band', 'basis'],
)
async def test_update_task_cannot_alter_the_stamp(
    stack: Stack, mode: str, alteration: dict[str, Any],
) -> None:
    task_id, metadata = await _stamped(stack, f'alter {mode}', _files(1))
    altered = {**metadata['declared_scope'], **alteration}
    payload = {**metadata, 'declared_scope': altered} if mode == 'replace' else {'declared_scope': altered}

    result = await stack.call('update_task', id=task_id, metadata=payload, metadata_mode=mode)

    _assert_write_authority(result, task_id, 'altered', metadata['declared_scope'])
    assert (await stack.metadata(task_id))['declared_scope'] == metadata['declared_scope']


@pytest.mark.asyncio
async def test_a_replace_that_omits_the_stamp_is_refused(stack: Stack) -> None:
    task_id, metadata = await _stamped(stack, 'replace drops the stamp', _files(1))
    without = {key: value for key, value in metadata.items() if key != 'declared_scope'}

    result = await stack.call('update_task', id=task_id, metadata=without, metadata_mode='replace')

    _assert_write_authority(result, task_id, 'removed', metadata['declared_scope'])
    assert (await stack.metadata(task_id))['declared_scope'] == metadata['declared_scope']


@pytest.mark.asyncio
async def test_update_task_cannot_stamp_a_pre_deploy_task(stack: Stack) -> None:
    pre_deploy = await stack.seed('filed before the stamp existed', {'files': _files(1)})

    result = await stack.call(
        'update_task', id=pre_deploy, metadata={'declared_scope': _FORGED}, metadata_mode='merge',
    )

    _assert_write_authority(result, pre_deploy, 'added')
    assert 'declared_scope' not in await stack.metadata(pre_deploy)


@pytest.mark.asyncio
async def test_a_replace_carrying_the_stamp_verbatim_is_accepted(stack: Stack) -> None:
    task_id, metadata = await _stamped(stack, 'replace carries the stamp', _files(1))

    result = await stack.call(
        'update_task', id=task_id, metadata={**metadata, 'x_note': 'changed'}, metadata_mode='replace',
    )

    assert result.get('error') is None, result
    after = await stack.metadata(task_id)
    assert after['x_note'] == 'changed'
    assert after['declared_scope'] == metadata['declared_scope']


@pytest.mark.asyncio
async def test_a_merge_echoing_the_stamp_is_accepted(stack: Stack) -> None:
    task_id, metadata = await _stamped(stack, 'merge echoes the stamp', _files(1))

    result = await stack.call(
        'update_task', id=task_id,
        metadata={'declared_scope': metadata['declared_scope'], 'x_note': 'echoed'}, metadata_mode='merge',
    )

    assert result.get('error') is None, result
    after = await stack.metadata(task_id)
    assert after['x_note'] == 'echoed'
    assert after['declared_scope'] == metadata['declared_scope']


@pytest.mark.asyncio
async def test_a_recon_stage_write_is_judged_against_its_own_before_read(stack: Stack) -> None:
    task_id, metadata = await _stamped(stack, 'recon-stage writes', _files(1))
    stamp = metadata['declared_scope']

    altered = await stack.call(
        'update_task', id=task_id, agent_id='recon-stage-2',
        metadata={'declared_scope': {**stamp, 'basis': 'combine'}}, metadata_mode='merge',
    )
    echoed = await stack.call(
        'update_task', id=task_id, agent_id='recon-stage-2',
        metadata={**metadata, 'x_note': 'recon echo'}, metadata_mode='replace',
    )

    _assert_write_authority(altered, task_id, 'altered', stamp)
    assert echoed.get('error') is None, echoed
    after = await stack.metadata(task_id)
    assert (after['x_note'], after['declared_scope']) == ('recon echo', stamp)


@pytest.mark.asyncio
async def test_rewriting_files_does_not_re_derive_the_stamp(stack: Stack) -> None:
    task_id, metadata = await _stamped(stack, 'files refined at planning', [])
    assert metadata['declared_scope']['band'] == 'undeclared'

    result = await stack.call('update_task', id=task_id, metadata={'files': _files(8)}, metadata_mode='merge')

    assert result.get('error') is None, result
    after = await stack.metadata(task_id)
    assert after['files'] == _files(8)
    assert (after['declared_scope']['band'], after['declared_scope']['files']) == ('undeclared', 0)
