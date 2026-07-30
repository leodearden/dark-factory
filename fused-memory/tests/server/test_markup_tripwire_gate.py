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
