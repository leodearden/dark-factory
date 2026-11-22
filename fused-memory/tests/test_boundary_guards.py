"""The fused-memory boundary guard chain (task 4643).

``fused_memory.server.boundary_guards`` builds the ordered members and installs
them as ONE wrapping of ``ToolManager.call_tool``: markup guard outermost, then
the unknown-argument guard, then the uuid-prefix guard when a prefix resolver
is wired.

AUTHORING RULE: never write a raw MCP envelope sentinel literal into this file.
Specimens are BUILT from ``shared.toolcall_markup``'s own constants by
:func:`_leaked`, as ``tests/test_markup_guard_fused_memory.py`` does.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# The consolidate cluster harness, reused rather than re-modelled: a bare
# AsyncMock leaves every `mem0_update` config leaf a Mock, which the fail-closed
# authz resolver rejects (tests/test_markup_guard_fused_memory.py does the same).
import test_consolidate_memories_tool as _consolidate
from _hermetic_task_stack import hermetic_task_stack
from _id_resolver_fakes import FakeGraphiti, FakeMem0, RecordingTombstones
from fastmcp.exceptions import ToolError
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent
from shared.mcp_markup_middleware import MarkupGuardMiddleware, RepairPolicy
from shared.toolcall_markup import CANONICAL_OPENER_PREFIX, closer_for
from shared.uuid_prefix_guard import (
    FACT_UUID_PREFIX_DETECTED,
    Candidate,
    Resolution,
    ResolverUnavailable,
    UuidPrefixGuardMiddleware,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.server import markup_guard
from fused_memory.server.boundary_guards import boundary_guard_chain, install_boundary_guards
from fused_memory.server.main import _install_tool_dispatch_guards
from fused_memory.server.markup_residue_spool import MarkupResidueSpool
from fused_memory.server.tools import create_mcp_server, per_use_prefix_resolver
from fused_memory.server.unknown_argument_guard import UnknownArgumentMiddleware
from fused_memory.services.memory_service import MemoryService
from fused_memory.utils.validation import validate_full_uuid

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_PROJECT_ID = 'dark_factory'
_PROJECT_ROOT = '/project'
_KNOWN_PROJECTS = {_PROJECT_ID: _PROJECT_ROOT}
_AGENT_ID = 'claude-task-4643'


def _leaked(clean: str, param: str, value: str) -> str:
    """*clean* text that mis-closed and absorbed *param*, built without a raw sentinel."""
    return (
        clean
        + closer_for('parameter')
        + CANONICAL_OPENER_PREFIX
        + f'"{param}">'
        + value
        + closer_for('parameter')
    )


def _payload(exc_info: pytest.ExceptionInfo[ToolError]) -> dict[str, Any]:
    return json.loads(str(exc_info.value))


class TestTheChainWithoutAPrefixResolver:
    def test_the_members_are_markup_then_unknown_argument(self):
        chain = boundary_guard_chain(known_projects=_KNOWN_PROJECTS, prefix_resolver=None)

        assert [type(member) for member in chain] == [
            MarkupGuardMiddleware,
            UnknownArgumentMiddleware,
        ]
        markup_member = chain[0]
        assert isinstance(markup_member, MarkupGuardMiddleware)
        assert markup_member.policy is RepairPolicy.REJECT_WITH_REPAIR
        assert markup_member.exempt_tools == markup_guard.EXEMPT_TOOLS

    def test_installing_wraps_once_and_warns_once_that_the_uuid_member_is_absent(self, caplog):
        server = create_mcp_server(AsyncMock())

        with caplog.at_level('WARNING', logger='fused_memory.server.boundary_guards'):
            install_boundary_guards(server, known_projects=_KNOWN_PROJECTS, prefix_resolver=None)
            wrapped = server._tool_manager.call_tool
            install_boundary_guards(server, known_projects=_KNOWN_PROJECTS, prefix_resolver=None)

        assert server._tool_manager.call_tool is wrapped
        warnings = [
            record for record in caplog.records
            if record.name == 'fused_memory.server.boundary_guards'
            and record.levelname == 'WARNING'
        ]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert 'uuid-prefix' in message
        assert 'resolver' in message

    @pytest.mark.asyncio
    async def test_markup_sits_outside_the_unknown_argument_guard(self):
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)
        install_boundary_guards(server, known_projects=_KNOWN_PROJECTS, prefix_resolver=None)

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool(
                'add_memory',
                {
                    'content': _leaked('a clean note', 'agent_id', _AGENT_ID),
                    'project_id': _PROJECT_ID,
                    'categroy': 'observations_and_summaries',
                },
            )

        assert _payload(exc_info)['error_type'] == 'mcp_markup_detected'
        mock_service.add_memory.assert_not_awaited()


PREFIX = 'bff81530'
FULL = 'bff81530-70e5-4ab6-939e-507299d1bcd2'
SIBLING = 'bff81530-0000-4000-8000-000000000002'

UNIQUE = Resolution('unique', (Candidate('mem0', FULL, 'the dead repair'),))
AMBIGUOUS = Resolution('ambiguous', (
    Candidate('mem0', FULL, 'a memory'),
    Candidate('graphiti_node', SIBLING, 'a node'),
))
NONE = Resolution('none', ())

_CITING = f'see {PREFIX} for the dead repair'

#: The declared uuid-prefix exemptions on this server (D10), spelled out here
#: rather than imported, so a silent edit to the declaration fails this pin.
_EXPECTED_EXEMPT = frozenset({
    'delete_memory', 'delete_episode', 'delete_entity', 'update_edge',
    'reassign_edge', 'merge_entities', 'consolidate_memories', 'scan_memory_content',
})

#: update_task's passthrough-only metadata keys as exempt paths, spelled out
#: here so that adding such a key is a conscious edit of this pin.
_EXPECTED_EXEMPT_PATHS = {
    'update_task': frozenset({
        ('metadata', 'done_provenance'),
        ('metadata', 'defects'),
        ('metadata', 'defect_test_retirements'),
    }),
}

_SUBSTITUTION = {
    'field': 'content', 'path': ['content'], 'from': PREFIX, 'to': FULL, 'namespace': 'mem0',
}


class RecordingResolver:
    """A ``PrefixResolver`` answering one scripted Resolution, or raising one outage."""

    def __init__(
        self, answer: Resolution = NONE, *, outage: ResolverUnavailable | None = None
    ) -> None:
        self.answer = answer
        self.outage = outage
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, project_id: str, prefix: str) -> Resolution:
        self.calls.append((project_id, prefix))
        if self.outage is not None:
            raise self.outage
        return self.answer


def _pass_through(mock_service: AsyncMock, method: str) -> None:
    """Give *method*'s return value a real ``model_dump`` (tests/server/test_markup_tripwire_gate.py)."""
    result = MagicMock()
    result.model_dump.return_value = {'id': 'ok'}
    getattr(mock_service, method).return_value = result


def _memory_server(
    resolver: RecordingResolver, *, known_projects: dict[str, str] | None = None
) -> tuple[Any, AsyncMock]:
    mock_service = AsyncMock()
    _pass_through(mock_service, 'add_memory')
    _pass_through(mock_service, 'add_episode')
    server = create_mcp_server(mock_service)
    install_boundary_guards(server, known_projects=known_projects, prefix_resolver=resolver)
    return server, mock_service


@pytest.fixture
def task_server_for(monkeypatch):
    """Build a task-tool server whose '/project' root needs no real checkout."""
    monkeypatch.setattr('fused_memory.server.tools.resolve_main_checkout', lambda p: str(p))

    def build(
        resolver: RecordingResolver, *, known_projects: dict[str, str]
    ) -> tuple[Any, AsyncMock]:
        interceptor = AsyncMock()
        interceptor.submit_task.return_value = {'ticket': 'tkt_x'}
        interceptor.update_task.return_value = {'ok': True}
        server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
        install_boundary_guards(server, known_projects=known_projects, prefix_resolver=resolver)
        return server, interceptor

    return build


async def _wire(server: Any, tool: str, arguments: dict[str, Any]) -> CallToolResult:
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        return await session.call_tool(tool, arguments)


def _text_json(result: CallToolResult) -> dict[str, Any]:
    [block] = result.content
    assert isinstance(block, TextContent)
    return json.loads(block.text)


def _repair(result: CallToolResult) -> dict[str, Any]:
    assert result.meta is not None, 'the call returned no meta at all'
    return result.meta['uuid_prefix_repair']


def _prefix_facts(caplog: pytest.LogCaptureFixture) -> list[tuple[int, dict[str, Any]]]:
    """Every ``uuid_prefix_detected`` line the fact sink logged, parsed whole."""
    marker = f'{FACT_UUID_PREFIX_DETECTED} '
    return [
        (record.levelno, json.loads(record.getMessage()[len(marker):]))
        for record in caplog.records
        if record.name == 'fused_memory.server.boundary_guards'
        and record.getMessage().startswith(marker)
    ]


def _add_memory_citing(content: str = _CITING) -> dict[str, Any]:
    return {'project_id': 'reify', 'content': content, 'agent_id': _AGENT_ID}


def _uuid_member(resolver: RecordingResolver) -> UuidPrefixGuardMiddleware:
    member = boundary_guard_chain(known_projects=_KNOWN_PROJECTS, prefix_resolver=resolver)[-1]
    assert isinstance(member, UuidPrefixGuardMiddleware)
    return member


class TestTheUuidPrefixMemberIsRegistered:
    def test_it_is_third_after_markup_and_unknown_argument(self):
        chain = boundary_guard_chain(
            known_projects=_KNOWN_PROJECTS, prefix_resolver=RecordingResolver()
        )

        assert [type(member) for member in chain] == [
            MarkupGuardMiddleware,
            UnknownArgumentMiddleware,
            UuidPrefixGuardMiddleware,
        ]

    @pytest.mark.asyncio
    async def test_its_declared_tool_classes_name_live_tools_only(self):
        member = _uuid_member(RecordingResolver())
        server = create_mcp_server(AsyncMock(), task_interceptor=AsyncMock())
        live = {tool.name for tool in await server.list_tools()}

        assert member.exempt_tools == _EXPECTED_EXEMPT
        assert member.exempt_tools <= live
        assert 'remove_edge' not in member.exempt_tools
        assert 'remove_edge' not in live
        assert member.forward_on_ambiguity_tools == frozenset({'update_task'})
        assert member.forward_on_ambiguity_tools <= live

    @pytest.mark.asyncio
    async def test_its_declared_exempt_paths_are_update_tasks_passthrough_only_keys(self):
        member = _uuid_member(RecordingResolver())
        server = create_mcp_server(AsyncMock(), task_interceptor=AsyncMock())
        live = {tool.name for tool in await server.list_tools()}

        assert member.exempt_paths == _EXPECTED_EXEMPT_PATHS
        assert set(member.exempt_paths) <= live


_OBSERVATION = '18e5b292'
_OBSERVATION_FULL = '18e5b292-a32a-420d-8ecc-8a37fc9f3fb3'
_STORED_PROVENANCE = {
    'kind': 'merged', 'commit': 'a' * 40, 'note': f'per Mem0 observation {_OBSERVATION}',
}


class TestAStoredPassthroughValueIsNeverRepaired:
    """A verbatim done_provenance echo under replace survives the guard (task 11331).

    The stored note cites a bare prefix that resolves uniquely. Expanding it
    would make the echo differ from the stored row, and update_task would
    refuse the whole write as done_provenance_via_update_task.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('as_json', [False, True], ids=['dict', 'json-string'])
    async def test_the_echo_lands_and_new_text_is_still_repaired(self, tmp_path, as_json):
        root = str(tmp_path)
        async with hermetic_task_stack(tmp_path) as (server, interceptor):
            install_boundary_guards(
                server,
                known_projects={'scratch': root},
                prefix_resolver=RecordingResolver(
                    Resolution('unique', (Candidate('mem0', _OBSERVATION_FULL, ''),))
                ),
            )
            backend = interceptor.taskmaster
            created = await backend.add_task(
                project_root=root, title='x',
                metadata=json.dumps({'done_provenance': _STORED_PROVENANCE, 'retry_count': 2}),
            )
            task_id = str(created['id'])
            payload = {
                'done_provenance': dict(_STORED_PROVENANCE),
                'retry_ledger': {'retry_count': 2},
                'x_audit_note': f'nested by task 7135; see {_OBSERVATION}',
                'allow_mcp_markup': True,
            }

            result = await _wire(server, 'update_task', {
                'id': task_id,
                'project_root': root,
                'metadata_mode': 'replace',
                'metadata': json.dumps(payload) if as_json else payload,
            })

            assert 'error' not in _text_json(result), _text_json(result)
            stored = (await backend.get_task(task_id, root))['metadata']
            stored = json.loads(stored) if isinstance(stored, str) else stored
            assert stored['done_provenance'] == _STORED_PROVENANCE
            assert stored['retry_ledger'] == {'retry_count': 2}
            assert 'retry_count' not in stored
            assert stored['x_audit_note'] == f'nested by task 7135; see {_OBSERVATION_FULL}'
            assert [record['path'] for record in _repair(result)['substitutions']] == [
                ['metadata', 'x_audit_note']
            ]


class TestTheUuidPrefixMemberSitsInsideTheMarkupGuard:
    @pytest.mark.asyncio
    async def test_a_markup_leak_citing_a_prefix_is_refused_before_any_resolution(self):
        resolver = RecordingResolver(UNIQUE)
        server, mock_service = _memory_server(resolver)

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool(
                'add_memory',
                {
                    'project_id': 'reify',
                    'content': _leaked(_CITING, 'agent_id', _AGENT_ID),
                },
            )

        assert _payload(exc_info)['error_type'] == 'mcp_markup_detected'
        assert resolver.calls == []
        mock_service.add_memory.assert_not_awaited()


class TestAUniquePrefixIsExpanded:
    @pytest.mark.asyncio
    async def test_the_write_carries_the_full_id_and_the_caller_is_told(self, caplog):
        caplog.set_level(logging.INFO, logger='fused_memory.server.boundary_guards')
        resolver = RecordingResolver(UNIQUE)
        server, mock_service = _memory_server(resolver)

        result = await _wire(server, 'add_memory', _add_memory_citing())

        assert resolver.calls == [('reify', PREFIX)]
        mock_service.add_memory.assert_awaited_once()
        assert FULL in mock_service.add_memory.await_args.kwargs['content']
        report = _repair(result)
        assert report['substitutions'] == [_SUBSTITUTION]
        assert result.structuredContent is not None
        assert result.structuredContent['uuid_prefix_repair'] == report
        assert _text_json(result)['uuid_prefix_repair'] == report
        facts = _prefix_facts(caplog)
        assert [
            (level, fact['outcome'], fact['agent_id'], fact['project'])
            for level, fact in facts
        ] == [(logging.INFO, 'expanded', _AGENT_ID, 'reify')]


class TestAnAmbiguousPrefix:
    @pytest.mark.asyncio
    async def test_a_default_class_write_is_rejected_with_both_candidates(self):
        resolver = RecordingResolver(AMBIGUOUS)
        server, mock_service = _memory_server(resolver)
        arguments = _add_memory_citing()

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool('add_memory', dict(arguments))

        payload = _payload(exc_info)
        assert payload['error_type'] == 'ambiguous_uuid_prefix'
        assert {candidate['id'] for candidate in payload['candidates']} == {FULL, SIBLING}
        assert payload['original_call'] == arguments
        mock_service.add_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_task_creation_is_rejected_too(self, task_server_for):
        server, interceptor = task_server_for(
            RecordingResolver(AMBIGUOUS), known_projects={'scratch': _PROJECT_ROOT}
        )

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool(
                'submit_task',
                {
                    'project_root': _PROJECT_ROOT,
                    'title': 'Follow the dead repair',
                    'description': f'Repair cited as {PREFIX} in the review.',
                },
            )

        assert _payload(exc_info)['error_type'] == 'ambiguous_uuid_prefix'
        interceptor.submit_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_task_forwards_unchanged_scoped_through_its_project_root(
        self, task_server_for
    ):
        """Erratum 7: update_task carries only a project_root, translated by the registry."""
        resolver = RecordingResolver(AMBIGUOUS)
        server, interceptor = task_server_for(
            resolver, known_projects={'scratch': _PROJECT_ROOT}
        )
        description = f'Repair cited as {PREFIX} in the review.'

        result = await _wire(
            server,
            'update_task',
            {'id': '7', 'project_root': _PROJECT_ROOT, 'description': description},
        )

        assert resolver.calls == [('scratch', PREFIX)]
        interceptor.update_task.assert_awaited_once()
        assert interceptor.update_task.await_args.kwargs['description'] == description
        [ambiguity] = _repair(result)['ambiguous']
        assert {candidate['id'] for candidate in ambiguity['candidates']} == {FULL, SIBLING}

    @pytest.mark.asyncio
    async def test_an_unregistered_project_root_is_not_resolved_and_says_so(
        self, task_server_for
    ):
        resolver = RecordingResolver(AMBIGUOUS)
        server, _interceptor = task_server_for(
            resolver, known_projects={'scratch': '/elsewhere'}
        )

        result = await _wire(
            server,
            'update_task',
            {'id': '7', 'project_root': _PROJECT_ROOT, 'description': f'see {PREFIX}'},
        )

        assert resolver.calls == []
        assert _repair(result) == {'resolver': 'no_project'}


class TestAnUnmatchedTokenIsInert:
    @pytest.mark.asyncio
    async def test_nothing_is_changed_reported_or_logged(self, caplog):
        caplog.set_level(logging.INFO, logger='fused_memory.server.boundary_guards')
        server, mock_service = _memory_server(RecordingResolver(NONE))
        content = 'merged after recon-3f2a9c1e ran on 20260904 without incident'

        wire = await _wire(server, 'add_memory', _add_memory_citing(content))
        in_process = await server._tool_manager.call_tool(
            'add_memory', _add_memory_citing(content)
        )

        assert [call.kwargs['content'] for call in mock_service.add_memory.await_args_list] == [
            content, content,
        ]
        assert wire.meta is None or 'uuid_prefix_repair' not in wire.meta
        assert wire.structuredContent is not None
        assert 'uuid_prefix_repair' not in wire.structuredContent
        assert 'uuid_prefix_repair' not in in_process
        assert _prefix_facts(caplog) == []


class TestAnExemptToolIsNeverResolved:
    @pytest.mark.asyncio
    async def test_a_delete_by_prefix_meets_the_full_uuid_refusal(self):
        resolver = RecordingResolver(UNIQUE)
        server, mock_service = _memory_server(resolver)

        result = await server._tool_manager.call_tool(
            'delete_memory', {'store': 'mem0', 'memory_id': PREFIX, 'project_id': 'reify'}
        )

        assert resolver.calls == []
        mock_service.delete_memory.assert_not_awaited()
        assert result == validate_full_uuid(PREFIX, field_name='memory_id')


class TestAResolverOutageForwardsUnchanged:
    @pytest.mark.asyncio
    async def test_the_write_lands_unexpanded_and_the_store_is_named(self):
        server, mock_service = _memory_server(
            RecordingResolver(outage=ResolverUnavailable('mem0'))
        )

        result = await _wire(server, 'add_memory', _add_memory_citing())

        mock_service.add_memory.assert_awaited_once()
        assert mock_service.add_memory.await_args.kwargs['content'] == _CITING
        assert _repair(result) == {'resolver': 'unavailable', 'store': 'mem0', 'reason': 'unreachable'}


class TestMainInstallsTheUuidMember:
    @pytest.mark.asyncio
    async def test_the_dispatch_guards_helper_expands_a_unique_prefix(self, tmp_path):
        resolver = RecordingResolver(UNIQUE)
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_memory')
        server = create_mcp_server(mock_service)

        _install_tool_dispatch_guards(
            server,
            known_projects=_KNOWN_PROJECTS,
            prefix_resolver=resolver,
            residue_spool=MarkupResidueSpool(tmp_path / 'spool'),
        )
        result = await _wire(server, 'add_memory', _add_memory_citing())

        assert FULL in mock_service.add_memory.await_args.kwargs['content']
        assert _repair(result)['substitutions'] == [_SUBSTITUTION]

    @pytest.mark.asyncio
    async def test_the_dispatch_guards_helper_contains_a_tool_body_escape(self, tmp_path):
        mock_service = AsyncMock()
        mock_service.add_memory.side_effect = SystemExit('escaped the tool body')
        server = create_mcp_server(mock_service)

        _install_tool_dispatch_guards(
            server,
            known_projects=_KNOWN_PROJECTS,
            prefix_resolver=RecordingResolver(UNIQUE),
            residue_spool=MarkupResidueSpool(tmp_path / 'spool'),
        )
        result = await server._tool_manager.call_tool(
            'add_memory', _add_memory_citing('a note citing nothing')
        )

        assert result == {'error': 'escaped the tool body', 'error_type': 'SystemExit'}

    @pytest.mark.asyncio
    async def test_the_production_resolver_reads_the_services_config_at_each_use(
        self, mock_config, monkeypatch,
    ):
        """run_server's resolver is built per use, so a hot-reloaded collection prefix applies next call."""
        mem0 = FakeMem0({FULL: {'data': 'the dead repair', 'user_id': 'reify'}})
        service = MemoryService(mock_config)
        monkeypatch.setattr(service, 'mem0', mem0)
        monkeypatch.setattr(service, 'graphiti', FakeGraphiti())
        monkeypatch.setattr(service, 'find_mem0_deletion_tombstones', RecordingTombstones())
        resolve = per_use_prefix_resolver(service)
        loaded_prefix = mock_config.mem0.collection_prefix

        first = await resolve('reify', PREFIX)
        monkeypatch.setattr(mock_config.mem0, 'collection_prefix', 'reloaded')
        await resolve('reify', PREFIX)

        assert [candidate.id for candidate in first.candidates] == [FULL]
        assert [name for name, _ in mem0.scroll_calls] == [
            f'{loaded_prefix}_reify', 'reloaded_reify',
        ]


_OVERRIDE = 'allow_uuid_prefix'
_ORDINARY = {'x_probe': 'kept'}


def _as_dict(metadata: Any) -> dict[str, Any]:
    """Persisted task metadata arrives as a JSON string; memory metadata as a dict."""
    if isinstance(metadata, str):
        return json.loads(metadata)
    return dict(metadata or {})


def _assert_stripped_and_kept(persisted: Any) -> None:
    persisted = _as_dict(persisted)
    assert _OVERRIDE not in persisted, f'the write-time flag was persisted: {persisted!r}'
    assert persisted.get('x_probe') == 'kept', f"the caller's own key was lost: {persisted!r}"


def _overriding(*, as_json: bool = False) -> dict[str, Any] | str:
    metadata = {_OVERRIDE: True, **_ORDINARY}
    return json.dumps(metadata) if as_json else metadata


class TestTheUuidPrefixOverrideNeverReachesPersistence:
    """``allow_uuid_prefix`` is a WRITE-TIME control, stripped by each tool body.

    The guard forwards the flag UNCHANGED to a tool that declares ``metadata``
    (layer 1 is its schema-conditional strip for tools that do not), so these
    six bodies are the party that keeps it out of the stores. Each case keeps an
    ordinary caller key, so a wholesale metadata drop cannot pass.
    """

    @pytest.mark.asyncio
    async def test_add_memory_keeps_the_bare_prefix_and_drops_the_flag(self):
        resolver = RecordingResolver(UNIQUE)
        server, mock_service = _memory_server(resolver)

        result = await _wire(
            server, 'add_memory', {**_add_memory_citing(), 'metadata': _overriding()}
        )

        assert resolver.calls == []
        assert mock_service.add_memory.await_args.kwargs['content'] == _CITING
        assert result.meta is None or 'uuid_prefix_repair' not in result.meta
        _assert_stripped_and_kept(mock_service.add_memory.await_args.kwargs['metadata'])

    @pytest.mark.asyncio
    async def test_add_episode_keeps_the_bare_prefix_and_forwards_no_flag(self):
        """A regression pin for a future metadata pass-through, NOT proof of add_episode's strip.

        add_episode forwards no metadata to its service today, so the last
        assertion held before the strip was written; it fails the day a change
        starts forwarding metadata without stripping the flag.
        """
        resolver = RecordingResolver(UNIQUE)
        server, mock_service = _memory_server(resolver)

        result = await server._tool_manager.call_tool(
            'add_episode',
            {'project_id': 'reify', 'content': _CITING, 'metadata': _overriding()},
        )

        assert resolver.calls == []
        assert mock_service.add_episode.await_args.kwargs['content'] == _CITING
        assert 'uuid_prefix_repair' not in result
        assert _OVERRIDE not in repr(mock_service.add_episode.await_args)

    @pytest.mark.asyncio
    async def test_add_system_record_drops_the_flag(self):
        resolver = RecordingResolver(UNIQUE)
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_system_record')
        server = create_mcp_server(mock_service)
        install_boundary_guards(server, known_projects=_KNOWN_PROJECTS, prefix_resolver=resolver)

        await server._tool_manager.call_tool(
            'add_system_record',
            {
                'content': _CITING,
                'project_id': 'reify',
                'category': 'observations_and_summaries',
                'agent_id': 'recon-stage-9',
                'metadata': _overriding(),
            },
        )

        assert resolver.calls == []
        assert mock_service.add_system_record.await_args.kwargs['content'] == _CITING
        _assert_stripped_and_kept(mock_service.add_system_record.await_args.kwargs['metadata'])

    @pytest.mark.asyncio
    @pytest.mark.parametrize('as_json', [False, True], ids=['dict', 'json-string'])
    async def test_submit_task_drops_the_flag(self, task_server_for, as_json):
        resolver = RecordingResolver(UNIQUE)
        server, interceptor = task_server_for(resolver, known_projects={'scratch': _PROJECT_ROOT})
        description = f'Repair cited as {PREFIX} in the review.'

        await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': _PROJECT_ROOT,
                'title': 'Follow the dead repair',
                'description': description,
                'metadata': _overriding(as_json=as_json),
            },
        )

        assert resolver.calls == []
        interceptor.submit_task.assert_awaited_once()
        assert interceptor.submit_task.await_args.kwargs['description'] == description
        _assert_stripped_and_kept(interceptor.submit_task.await_args.kwargs['metadata'])

    @pytest.mark.asyncio
    @pytest.mark.parametrize('as_json', [False, True], ids=['dict', 'json-string'])
    async def test_update_task_drops_the_flag(self, task_server_for, as_json):
        resolver = RecordingResolver(UNIQUE)
        server, interceptor = task_server_for(resolver, known_projects={'scratch': _PROJECT_ROOT})
        description = f'Repair cited as {PREFIX} in the review.'

        await server._tool_manager.call_tool(
            'update_task',
            {
                'id': '7',
                'project_root': _PROJECT_ROOT,
                'description': description,
                'metadata': _overriding(as_json=as_json),
            },
        )

        assert resolver.calls == []
        assert interceptor.update_task.await_args.kwargs['description'] == description
        _assert_stripped_and_kept(interceptor.update_task.await_args.kwargs['metadata'])

    @pytest.mark.asyncio
    async def test_consolidate_memories_drops_the_flag_it_receives_untouched(self):
        """Exempt from resolution, so the flag reaches this body as the caller sent it."""
        resolver = RecordingResolver(UNIQUE)
        svc = _consolidate.make_service()
        server = create_mcp_server(svc)
        install_boundary_guards(server, known_projects=_KNOWN_PROJECTS, prefix_resolver=resolver)

        await server._tool_manager.call_tool(
            'consolidate_memories',
            {
                'canonical_content': f'{_consolidate.CONTENT} See {PREFIX}.',
                'topic': _consolidate.TOPIC,
                'project_id': _PROJECT_ID,
                'supersedes': list(_consolidate.SUPERSEDES),
                'run_id': _consolidate.RUN_ID,
                'agent_id': _consolidate.AGENT,
                'metadata': _overriding(),
            },
        )

        assert resolver.calls == []
        svc.add_memory.assert_awaited_once()
        _assert_stripped_and_kept(svc.add_memory.await_args.kwargs['metadata'])

    @pytest.mark.asyncio
    async def test_a_non_literal_true_flag_does_not_unlock_the_guard(self):
        resolver = RecordingResolver(UNIQUE)
        server, mock_service = _memory_server(resolver)

        await server._tool_manager.call_tool(
            'add_memory',
            {**_add_memory_citing(), 'metadata': {_OVERRIDE: 'yes', **_ORDINARY}},
        )

        assert resolver.calls == [('reify', PREFIX)]
        assert FULL in mock_service.add_memory.await_args.kwargs['content']
