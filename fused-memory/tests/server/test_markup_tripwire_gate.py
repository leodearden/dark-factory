"""Boundary tests for the MCP-markup write tripwire (task 3141).

Wiring tests for the four fused-memory write tools — add_memory, add_episode,
submit_task, update_task — asserting that a write carrying raw MCP envelope
markup is REJECTED before it reaches the service/interceptor. The pure helpers
and the storm counter are unit-tested in the sibling ``test_markup_tripwire.py``.

Harness: an AsyncMock service (plus AsyncMock task_interceptor for the task
tools) wired through create_mcp_server and invoked via
``server._tool_manager.call_tool(...)`` — the same shape as
``test_add_memory_near_duplicate_gate.py`` and
``test_tools_validation.py::TestSubmitTaskPremiseLintGuard``.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.tools import create_mcp_server

_PROJECT_ID = 'dark_factory'

# One specimen per literal in MCP_MARKUP_PATTERNS. The '</content>\n</invoke>'
# tail is the shape of the real corpus specimens (DF 3083 vector 1); the
# '<parameter name=' fragment is vector 2, the one that mis-parsed task 3210's
# priority silently.
_LEAKED_CONTENT = 'a real memory sentence\n</content>\n</invoke>'
_LEAKED_INVOKE = 'a real memory sentence </invoke>'
_LEAKED_PARAMETER = 'a real memory sentence <parameter name="priority">high</parameter>'
_CLEAN_CONTENT = 'a perfectly ordinary memory about the merge lane'


def _pass_through(mock_service: AsyncMock, method: str) -> None:
    """Give *method*'s return value a real ``model_dump``.

    An unspecced AsyncMock chains AsyncMock all the way down, so
    ``result.model_dump()`` would be an unawaited coroutine unless the return
    value is an explicit MagicMock (mirrors
    test_add_memory_near_duplicate_gate.py::_configure_pass_through_add_memory).
    """
    result = MagicMock()
    result.model_dump.return_value = {'id': 'ok'}
    getattr(mock_service, method).return_value = result


def _parse(result):
    """Extract the dict from a FastMCP TextContent result or pass-through dict.

    Mirrors test_tools_validation.py::_parse_tool_result — the task tools return
    through a different FastMCP path than the memory tools.
    """
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


@pytest.fixture
def task_server(monkeypatch):
    """A server whose task tools accept a synthetic project_root.

    '/project' is not a real git working tree, so resolve_main_checkout is
    stubbed to pass it through — the same passthrough
    TestSubmitTaskPremiseLintGuard uses.
    """
    monkeypatch.setattr('fused_memory.server.tools.resolve_main_checkout', lambda p: str(p))
    interceptor = AsyncMock()
    interceptor.submit_task.return_value = {'ticket': 'tkt_x'}
    interceptor.update_task.return_value = {'ok': True}
    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
    return server, interceptor


def _assert_markup_block(result: object, *, field: str, pattern: str, agent_id: str) -> None:
    """Assert *result* is the structured markup rejection (INV-1)."""
    assert isinstance(result, dict), f'Expected a dict, got {type(result)}: {result!r}'
    assert result.get('error') == 'mcp_markup_write_blocked', f'got: {result!r}'
    assert result.get('error_type') == 'McpEnvelopeMarkupWriteRejected', f'got: {result!r}'
    assert result.get('field') == field, f'expected field={field!r}, got: {result!r}'
    assert result.get('matched_pattern') == pattern, (
        f'expected matched_pattern={pattern!r}, got: {result!r}'
    )
    assert result.get('agent_id') == agent_id, f'expected agent_id echoed, got: {result!r}'
    hint = result.get('hint')
    assert hint, f'expected a non-empty hint, got: {result!r}'
    assert '3083' in hint, f'expected the hint to name DF 3083, got: {hint!r}'


class TestAddMemoryMarkupGate:
    """add_memory rejects leaked envelope markup in content."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('content', 'pattern'),
        [
            (_LEAKED_CONTENT, '</content>'),
            (_LEAKED_INVOKE, '</invoke>'),
            (_LEAKED_PARAMETER, '<parameter name='),
        ],
    )
    async def test_rejects_each_pattern_without_calling_the_service(self, content, pattern):
        """Every literal is rejected, and the write never reaches the store.

        assert_not_called is the load-bearing assertion: a block dict returned
        AFTER the memory landed would leave the specimen in the corpus anyway.
        """
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_memory')
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': content,
                'category': 'observations_and_summaries',
                'agent_id': 'claude-task-3141',
                'project_id': _PROJECT_ID,
            },
        )

        _assert_markup_block(
            result, field='content', pattern=pattern, agent_id='claude-task-3141'
        )
        mock_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_content_excerpt_echoes_the_rejected_content(self):
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_memory')
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _LEAKED_INVOKE,
                'category': 'observations_and_summaries',
                'agent_id': 'a',
                'project_id': _PROJECT_ID,
            },
        )
        assert result.get('content_excerpt') == _LEAKED_INVOKE[:200]

    @pytest.mark.asyncio
    async def test_clean_content_reaches_the_service_unchanged(self):
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_memory')
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CLEAN_CONTENT,
                'category': 'observations_and_summaries',
                'agent_id': 'a',
                'project_id': _PROJECT_ID,
            },
        )
        assert result == {'id': 'ok'}, f'expected the write to pass through, got: {result!r}'
        mock_service.add_memory.assert_called_once()
        assert mock_service.add_memory.call_args.kwargs['content'] == _CLEAN_CONTENT

    @pytest.mark.asyncio
    async def test_override_lets_deliberate_markup_through(self):
        """metadata={'allow_mcp_markup': True} is the documented escape hatch.

        Required for correctness, not convenience: DF 3083's own description
        quotes all three literals in prose, so without this the sibling task this
        containment leaf exists to feed could never be written about.
        """
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_memory')
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _LEAKED_INVOKE,
                'category': 'observations_and_summaries',
                'agent_id': 'a',
                'project_id': _PROJECT_ID,
                'metadata': {'allow_mcp_markup': True, 'source': 'doc'},
            },
        )
        assert result == {'id': 'ok'}, f'expected the override to allow the write, got: {result!r}'
        mock_service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_override_flag_is_stripped_before_persistence(self):
        """The flag is a write-time control, not corpus metadata.

        Mirrors how add_memory already pops 'allow_near_duplicate' — if it were
        persisted it would enter the stored metadata vocabulary and, worse, ride
        along on future reads of a memory that only ever needed it once.
        """
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_memory')
        server = create_mcp_server(mock_service)

        await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _LEAKED_INVOKE,
                'category': 'observations_and_summaries',
                'agent_id': 'a',
                'project_id': _PROJECT_ID,
                'metadata': {'allow_mcp_markup': True, 'source': 'doc'},
            },
        )
        forwarded = mock_service.add_memory.call_args.kwargs['metadata']
        assert 'allow_mcp_markup' not in (forwarded or {}), (
            f'the override flag must not be persisted, got: {forwarded!r}'
        )
        assert (forwarded or {}).get('source') == 'doc', (
            f'other metadata keys must survive the strip, got: {forwarded!r}'
        )

    @pytest.mark.asyncio
    async def test_a_non_true_override_value_does_not_bypass_the_gate(self):
        """Fail-closed at the boundary too, not just in the helper."""
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_memory')
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _LEAKED_INVOKE,
                'category': 'observations_and_summaries',
                'agent_id': 'a',
                'project_id': _PROJECT_ID,
                'metadata': {'allow_mcp_markup': 'yes'},
            },
        )
        _assert_markup_block(result, field='content', pattern='</invoke>', agent_id='a')
        mock_service.add_memory.assert_not_called()


class TestAddEpisodeMarkupGate:
    """add_episode rejects leaked envelope markup in content."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('content', 'pattern'),
        [
            (_LEAKED_CONTENT, '</content>'),
            (_LEAKED_INVOKE, '</invoke>'),
            (_LEAKED_PARAMETER, '<parameter name='),
        ],
    )
    async def test_rejects_each_pattern_without_calling_the_service(self, content, pattern):
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_episode')
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': content,
                'agent_id': 'claude-task-3141',
                'project_id': _PROJECT_ID,
            },
        )

        _assert_markup_block(
            result, field='content', pattern=pattern, agent_id='claude-task-3141'
        )
        mock_service.add_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_clean_content_reaches_the_service_unchanged(self):
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_episode')
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_episode',
            {'content': _CLEAN_CONTENT, 'agent_id': 'a', 'project_id': _PROJECT_ID},
        )
        assert result == {'id': 'ok'}, f'expected the write to pass through, got: {result!r}'
        mock_service.add_episode.assert_called_once()
        assert mock_service.add_episode.call_args.kwargs['content'] == _CLEAN_CONTENT

    @pytest.mark.asyncio
    async def test_override_lets_deliberate_markup_through(self):
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_episode')
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _LEAKED_INVOKE,
                'agent_id': 'a',
                'project_id': _PROJECT_ID,
                'metadata': {'allow_mcp_markup': True},
            },
        )
        assert result == {'id': 'ok'}, f'expected the override to allow the write, got: {result!r}'
        mock_service.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_markup_is_rejected_before_the_recon_stage_content_guards(self):
        """Ordering: the tripwire fires FIRST, so markup never reaches those guards.

        A recon-stage agent writing markup-bearing content must get the markup
        rejection, not a mixed-temporal-framing/snapshot verdict derived from
        text that is partly serialized envelope. Those heuristics read prose;
        feeding them a leaked tail would let the leak masquerade as a different
        (and misleading) violation.
        """
        mock_service = AsyncMock()
        _pass_through(mock_service, 'add_episode')
        server = create_mcp_server(mock_service)

        # Content that trips BOTH the markup tripwire and mixed-temporal framing.
        content = (
            'As of now there are 5 pending tasks; historically there were 9.\n'
            '</content>\n</invoke>'
        )
        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': content,
                'agent_id': 'recon-stage-1',
                'project_id': _PROJECT_ID,
            },
        )
        assert result.get('error') == 'mcp_markup_write_blocked', (
            f'expected the markup tripwire to win the ordering, got: {result!r}'
        )
        mock_service.add_episode.assert_not_called()


class TestSubmitTaskMarkupGate:
    """submit_task rejects leaked envelope markup in any of its four text fields."""

    @pytest.mark.asyncio
    async def test_rejects_markup_in_description_before_the_interceptor(self, task_server):
        """The DF 3083 vector-2 case, previously a SILENT mis-parse.

        A '<parameter name="priority">' fragment in a description reached the
        interceptor's description parser, which derived the wrong value from it
        without complaint (reify task 3210 was filed priority=high and stored as
        medium). Loud rejection ahead of that parser is the whole point.
        """
        server, interceptor = task_server

        result = _parse(await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': '/project',
                'prompt': 'Reconcile task 7',
                'description': f'Reconcile task 7.\n{_LEAKED_PARAMETER}',
                'agent_id': 'claude-task-3141',
            },
        ))

        _assert_markup_block(
            result, field='description', pattern='<parameter name=', agent_id='claude-task-3141'
        )
        interceptor.submit_task.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('field', ['title', 'description', 'details', 'prompt'])
    async def test_rejects_markup_in_each_scanned_field(self, task_server, field):
        """All four text fields are scanned, so no channel is left unchecked.

        premise_lint_guard already lints exactly these four at this boundary for
        the same reason: they all flow into the same description parser, so
        checking description alone would leave three ways in.
        """
        server, interceptor = task_server
        args = {
            'project_root': '/project',
            'prompt': 'a clean prompt',
            'description': 'a clean description',
            'agent_id': 'a',
        }
        args[field] = f'dirty {_LEAKED_INVOKE}'

        result = _parse(await server._tool_manager.call_tool('submit_task', args))

        _assert_markup_block(result, field=field, pattern='</invoke>', agent_id='a')
        interceptor.submit_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_clean_submit_task_reaches_the_interceptor(self, task_server):
        server, interceptor = task_server

        result = _parse(await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': '/project',
                'prompt': 'Reconcile task 7',
                'description': 'Reconcile task 7 status against the knowledge graph.',
                'agent_id': 'a',
            },
        ))
        assert result.get('error') != 'mcp_markup_write_blocked', f'got: {result!r}'
        interceptor.submit_task.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('as_json', [False, True])
    async def test_override_lets_deliberate_markup_through(self, task_server, as_json):
        """Both metadata shapes honour the override.

        submit_task accepts metadata as an object OR a JSON string; an author
        documenting the leak (as DF 3083's own description does) must get through
        whichever shape they used.
        """
        server, interceptor = task_server
        metadata = {'allow_mcp_markup': True, 'execution_class': 'code_tdd'}

        result = _parse(await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': '/project',
                'prompt': 'Document the leak',
                'description': f'The leak looks like {_LEAKED_INVOKE}',
                'agent_id': 'a',
                'metadata': json.dumps(metadata) if as_json else metadata,
            },
        ))
        assert result.get('error') != 'mcp_markup_write_blocked', f'got: {result!r}'
        interceptor.submit_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_override_flag_is_not_persisted_into_task_metadata(self, task_server):
        """'allow_mcp_markup' is a write-time control, not task metadata.

        It is deliberately absent from the task metadata vocabulary
        (docs/task-authoring.md), so letting it through would mint an
        unrecognised key on every task written with the override.
        """
        server, interceptor = task_server

        await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': '/project',
                'prompt': 'Document the leak',
                'description': f'The leak looks like {_LEAKED_INVOKE}',
                'agent_id': 'a',
                'metadata': {'allow_mcp_markup': True, 'execution_class': 'code_tdd'},
            },
        )
        forwarded = interceptor.submit_task.call_args.kwargs.get('metadata')
        assert 'allow_mcp_markup' not in json.dumps(forwarded), (
            f'the override flag must not reach the interceptor, got: {forwarded!r}'
        )
        assert 'code_tdd' in json.dumps(forwarded), (
            f'other metadata must survive the strip, got: {forwarded!r}'
        )


class TestUpdateTaskMarkupGate:
    """update_task rejects leaked envelope markup in any of its scanned fields."""

    @pytest.mark.asyncio
    async def test_rejects_markup_in_description_before_the_interceptor(self, task_server):
        server, interceptor = task_server

        result = _parse(await server._tool_manager.call_tool(
            'update_task',
            {
                'id': '3141',
                'project_root': '/project',
                'description': f'updated.\n{_LEAKED_CONTENT}',
                'agent_id': 'claude-task-3141',
            },
        ))

        _assert_markup_block(
            result, field='description', pattern='</content>', agent_id='claude-task-3141'
        )
        interceptor.update_task.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('field', ['title', 'description', 'details', 'prompt'])
    async def test_rejects_markup_in_each_scanned_field(self, task_server, field):
        server, interceptor = task_server
        args = {'id': '3141', 'project_root': '/project', 'agent_id': 'a'}
        args[field] = f'dirty {_LEAKED_INVOKE}'

        result = _parse(await server._tool_manager.call_tool('update_task', args))

        _assert_markup_block(result, field=field, pattern='</invoke>', agent_id='a')
        interceptor.update_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_clean_update_task_reaches_the_interceptor(self, task_server):
        server, interceptor = task_server

        result = _parse(await server._tool_manager.call_tool(
            'update_task',
            {
                'id': '3141',
                'project_root': '/project',
                'description': 'a clean updated description',
                'agent_id': 'a',
            },
        ))
        assert result.get('error') != 'mcp_markup_write_blocked', f'got: {result!r}'
        interceptor.update_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_override_lets_deliberate_markup_through(self, task_server):
        """The case that makes the hatch load-bearing: updating DF 3083 itself.

        3083's description quotes all three literals, so without this an
        update_task on the very sibling this leaf feeds would be permanently
        blocked.
        """
        server, interceptor = task_server

        result = _parse(await server._tool_manager.call_tool(
            'update_task',
            {
                'id': '3083',
                'project_root': '/project',
                'description': f'The leak emits {_LEAKED_CONTENT}',
                'agent_id': 'a',
                'metadata': {'allow_mcp_markup': True},
            },
        ))
        assert result.get('error') != 'mcp_markup_write_blocked', f'got: {result!r}'
        interceptor.update_task.assert_called_once()
