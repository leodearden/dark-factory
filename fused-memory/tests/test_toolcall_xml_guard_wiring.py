"""Task 3083 — the tool-call XML leak guard is LIVE at the MCP write boundary.

``scripts/scan_task_toolcall_leaks.py`` has been able to RECOGNIZE this
corruption since task 2939, and yet its own docstring records ~32 still-affected
tasks. The detector's weakness was never its pattern — it is that the pattern is
wired NOWHERE. Its only importer is its own test file, so it can report leaks
only out-of-band, long after the corrupted text was persisted. These tests pin
the remedy: the same shared detector (``fused_memory.utils.toolcall_xml_leak``,
the single source of truth) refusing the write at the one boundary where it can
still be refused.

The unit-level invariant matrix for ``toolcall_xml_error`` lives in
``tests/test_toolcall_xml_guard.py``. Every test HERE asserts WIRING instead:
that a leak in a real MCP tool call reaches the guard at all, and that a
rejection stops the write BEFORE the task interceptor / memory service is ever
reached — nothing is persisted, not even partially.

Harness mirrors ``tests/test_tools_validation.py::TestSubmitTaskPremiseLintGuard``
and ``tests/test_mcp_boundary_canonicalization.py``:
``create_mcp_server(AsyncMock(), task_interceptor=AsyncMock())`` + ``await
server._tool_manager.call_tool(name, kwargs)``, asserting on the downstream
mock's ``assert_awaited_once()`` / ``assert_not_called()``.

## Sentinel-literal hazard (task 3083, pre-1)

Every tool-call sentinel below is spelled with the ``\\x3c`` escape for ``<``.
Writing one verbatim would make THIS file's own authoring tool call terminate
early — reproducing the bug under test, truncating the file, and silently
dropping the sibling arguments of that same call. The escapes are
byte-identical at runtime; do not "helpfully" un-escape them.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tools import create_mcp_server

# The two live leak shapes, per fused_memory.utils.toolcall_xml_leak:
#   * the task-DB shape (tasks 992/1068/1067/2691) — a stray closing tag, REAL
#     whitespace, then a serialized opening `parameter` tag. Three of those four
#     recorded fixtures leak `priority` exactly like this, which is why the
#     rejection message names that parameter specifically.
#   * the Mem0 tail shape (record c759c53b) — a stray closing tag, real
#     whitespace, then a bare closing `invoke` tag and nothing after it. The
#     task-2939-era pattern required a `parameter` continuation and missed this.
_PRIORITY_LEAK = '\x3c/description>\n\x3cparameter name="priority">high'
_INVOKE_TAIL_LEAK = '\x3c/content>\n\x3c/invoke>'

# submit_task's and update_task's four caller-authored text channels. Scoped to
# match premise_lint_guard, whose docstring records why linting `description`
# alone is insufficient: `prompt`, `title`, and `details` are three further
# channels the same harness truncation can land in.
_CLEAN_TASK_FIELDS = {
    'title': 'Reconcile task 7 against the knowledge graph',
    'description': 'Reconcile task 7 status against the knowledge graph.',
    'prompt': 'Reconcile task 7.',
    'details': 'Implementation note: compare the ledger against the graph.',
}
_TASK_TEXT_FIELDS = tuple(_CLEAN_TASK_FIELDS)

_CLEAN_CONTENT = 'The merge worker consumes the stash stack in project_root.'


def _leaked(field: str) -> str:
    """The clean value for *field* with a live leak fragment appended."""
    return _CLEAN_TASK_FIELDS[field] + _PRIORITY_LEAK


def _parse_tool_result(result):
    """Extract the dict from a FastMCP TextContent result or pass-through dict."""
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


def _assert_leak_rejection(parsed, field: str) -> None:
    """Assert *parsed* is the leak rejection, naming *field* and the sibling risk.

    The sibling-parameter sentence is the root-cause payoff, not decoration:
    early termination silently swallows the arguments that FOLLOWED the
    truncated one, so `priority` never reaches this boundary and
    ``priority or 'medium'`` substitutes a plausible wrong value. Without that
    sentence the caller fixes the visible tail and never learns the invisible
    half of the same defect happened too.
    """
    assert parsed.get('error_type') == 'ToolCallXmlLeakError', f'got {parsed!r}'
    message = parsed.get('error', '')
    assert field in message, f'expected the offending field {field!r} named in: {message!r}'
    assert 'priority' in message, (
        f'expected the dropped-sibling warning naming `priority` in: {message!r}'
    )


@pytest.fixture
def passthrough_main_checkout(monkeypatch):
    """'/project' is not a real git working tree — pass it through unchanged.

    Mirrors test_task_tools.py's fixture of the same shape, which the
    premise-lint boundary tests reuse for the identical reason.
    """
    monkeypatch.setattr('fused_memory.server.tools.resolve_main_checkout', lambda p: str(p))


def _task_server():
    """An MCP server whose task interceptor is a mock, for assert_not_called()."""
    mock_task_interceptor = AsyncMock()
    mock_task_interceptor.submit_task.return_value = {'ticket': 'tkt_x'}
    mock_task_interceptor.update_task.return_value = {'id': '7', 'status': 'pending'}
    server = create_mcp_server(AsyncMock(), task_interceptor=mock_task_interceptor)
    return server, mock_task_interceptor


def _memory_server():
    """An MCP server with no known_projects registry (permissive project_id)."""
    mock_service = AsyncMock()
    return create_mcp_server(mock_service), mock_service


def _submit_kwargs(**overrides):
    kwargs = {
        'project_root': '/project',
        # A non-recon caller throughout: unlike premise_lint_error, this guard
        # is deliberately NOT scoped to recon-stage agents.
        'agent_id': 'claude-interactive',
        **_CLEAN_TASK_FIELDS,
    }
    kwargs.update(overrides)
    return kwargs


def _update_kwargs(**overrides):
    kwargs = {
        'id': '7',
        'project_root': '/project',
        'agent_id': 'claude-interactive',
        **_CLEAN_TASK_FIELDS,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.usefixtures('passthrough_main_checkout')
class TestSubmitTaskToolCallXmlGuardWiring:
    """submit_task rejects a leaked tool-call fragment before the curator."""

    @pytest.mark.asyncio
    async def test_clean_submit_reaches_the_interceptor(self):
        """All four text fields clean — the guard is invisible to normal traffic."""
        server, interceptor = _task_server()

        result = await server._tool_manager.call_tool('submit_task', _submit_kwargs())

        parsed = _parse_tool_result(result)
        assert 'error' not in parsed, f'clean submit must not be rejected; got {parsed!r}'
        interceptor.submit_task.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('field', _TASK_TEXT_FIELDS)
    async def test_leak_in_each_text_field_is_rejected_before_the_interceptor(self, field):
        """Each channel is guarded independently — nothing is persisted."""
        server, interceptor = _task_server()

        result = await server._tool_manager.call_tool(
            'submit_task', _submit_kwargs(**{field: _leaked(field)})
        )

        _assert_leak_rejection(_parse_tool_result(result), field)
        interceptor.submit_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_bare_invoke_tail_shape_is_rejected(self):
        """The Mem0 c759c53b tail shape (no `parameter` continuation) also rejects.

        The task-DB-era pattern required a serialized opening `parameter` tag
        after the stray closing tag and therefore missed this. Pinning it at the
        live boundary proves the generalization reached the wiring, not just the
        shared module's own unit tests.
        """
        server, interceptor = _task_server()

        result = await server._tool_manager.call_tool(
            'submit_task',
            _submit_kwargs(description=_CLEAN_TASK_FIELDS['description'] + _INVOKE_TAIL_LEAK),
        )

        parsed = _parse_tool_result(result)
        assert parsed.get('error_type') == 'ToolCallXmlLeakError', f'got {parsed!r}'
        interceptor.submit_task.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('field', _TASK_TEXT_FIELDS)
    async def test_allow_toolcall_xml_optout_reaches_the_interceptor(self, field):
        """The escape hatch is load-bearing: this task's own description quotes
        every sentinel verbatim and would otherwise be unfileable."""
        server, interceptor = _task_server()

        result = await server._tool_manager.call_tool(
            'submit_task',
            _submit_kwargs(
                **{field: _leaked(field)}, metadata={'allow_toolcall_xml': True}
            ),
        )

        parsed = _parse_tool_result(result)
        assert 'error' not in parsed, f'opt-out must let the write through; got {parsed!r}'
        interceptor.submit_task.assert_awaited_once()


@pytest.mark.usefixtures('passthrough_main_checkout')
class TestUpdateTaskToolCallXmlGuardWiring:
    """update_task is guarded identically — an edit can carry the leak too."""

    @pytest.mark.asyncio
    async def test_clean_update_reaches_the_interceptor(self):
        server, interceptor = _task_server()

        result = await server._tool_manager.call_tool('update_task', _update_kwargs())

        parsed = _parse_tool_result(result)
        assert 'error' not in parsed, f'clean update must not be rejected; got {parsed!r}'
        interceptor.update_task.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('field', _TASK_TEXT_FIELDS)
    async def test_leak_in_each_text_field_is_rejected_before_the_interceptor(self, field):
        server, interceptor = _task_server()

        result = await server._tool_manager.call_tool(
            'update_task', _update_kwargs(**{field: _leaked(field)})
        )

        _assert_leak_rejection(_parse_tool_result(result), field)
        interceptor.update_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_optout_is_read_before_metadata_is_json_encoded(self):
        """PLACEMENT PIN: update_task json.dumps-es a dict ``metadata`` on its way
        to the interceptor. The guard must run while ``metadata`` is still a dict,
        or the ``allow_toolcall_xml`` lookup would silently miss on a JSON string
        and the escape hatch would be unreachable through this tool.
        """
        server, interceptor = _task_server()

        result = await server._tool_manager.call_tool(
            'update_task',
            _update_kwargs(
                description=_leaked('description'),
                metadata={'allow_toolcall_xml': True},
            ),
        )

        parsed = _parse_tool_result(result)
        assert 'error' not in parsed, f'opt-out must let the write through; got {parsed!r}'
        interceptor.update_task.assert_awaited_once()


class TestAddMemoryToolCallXmlGuardWiring:
    """add_memory: the vector-1 vector — corrupted text persisted into Mem0.

    Records 9f2d2ae6 and c759c53b are exactly this write path, which is how the
    corpus acquired the leaks that task 3083's sweep now has to find.
    """

    @pytest.mark.asyncio
    async def test_clean_memory_reaches_the_service(self):
        server, service = _memory_server()

        result = await server._tool_manager.call_tool(
            'add_memory',
            {'content': _CLEAN_CONTENT, 'project_id': 'dark_factory', 'metadata': {}},
        )

        parsed = _parse_tool_result(result)
        assert not (isinstance(parsed, dict) and 'error' in parsed), f'got {parsed!r}'
        service.add_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_leak_in_content_is_rejected_before_the_service(self):
        server, service = _memory_server()

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CLEAN_CONTENT + _INVOKE_TAIL_LEAK,
                'project_id': 'dark_factory',
                'metadata': {},
            },
        )

        _assert_leak_rejection(_parse_tool_result(result), 'content')
        service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_allow_toolcall_xml_optout_reaches_the_service(self):
        """A memory DOCUMENTING this leak must remain writable."""
        server, service = _memory_server()

        result = await server._tool_manager.call_tool(
            'add_memory',
            {
                'content': _CLEAN_CONTENT + _INVOKE_TAIL_LEAK,
                'project_id': 'dark_factory',
                'metadata': {'allow_toolcall_xml': True},
            },
        )

        parsed = _parse_tool_result(result)
        assert not (isinstance(parsed, dict) and 'error' in parsed), f'got {parsed!r}'
        service.add_memory.assert_called_once()


class TestAddEpisodeToolCallXmlGuardWiring:
    """add_episode: the same corruption, but amplified by the extraction pipeline.

    An episode's content is fed to Graphiti's extractor, so a leaked fragment
    here does not merely sit in one payload — it seeds edges derived from
    truncated text. Graphiti residual episode d12b0eb4 is the recorded specimen.
    """

    @pytest.mark.asyncio
    async def test_clean_episode_reaches_the_service(self):
        server, service = _memory_server()

        result = await server._tool_manager.call_tool(
            'add_episode',
            {'content': _CLEAN_CONTENT, 'project_id': 'dark_factory'},
        )

        parsed = _parse_tool_result(result)
        assert not (isinstance(parsed, dict) and 'error' in parsed), f'got {parsed!r}'
        service.add_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_leak_in_content_is_rejected_before_the_service(self):
        server, service = _memory_server()

        result = await server._tool_manager.call_tool(
            'add_episode',
            {'content': _CLEAN_CONTENT + _INVOKE_TAIL_LEAK, 'project_id': 'dark_factory'},
        )

        _assert_leak_rejection(_parse_tool_result(result), 'content')
        service.add_episode.assert_not_called()

    @pytest.mark.asyncio
    async def test_allow_toolcall_xml_optout_reaches_the_service(self):
        server, service = _memory_server()

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _CLEAN_CONTENT + _INVOKE_TAIL_LEAK,
                'project_id': 'dark_factory',
                'metadata': {'allow_toolcall_xml': True},
            },
        )

        parsed = _parse_tool_result(result)
        assert not (isinstance(parsed, dict) and 'error' in parsed), f'got {parsed!r}'
        service.add_episode.assert_called_once()


@pytest.mark.usefixtures('passthrough_main_checkout')
class TestGuardOrderingInTheSubmitTaskPrologue:
    """The leak guard sits at the TOP of submit_task's guard chain.

    Each test first establishes a CONTROL — the same call WITHOUT a leak is
    rejected by the later guard under test — so the ordering assertion cannot
    pass vacuously against a guard that never fires.
    """

    @pytest.mark.asyncio
    async def test_leak_rejection_precedes_the_lock_charter_guard(self):
        """lock_charter_error is the first guard in the prologue today; the leak
        guard must land ahead of even that one."""
        dir_metadata = {'files': ['orchestrator/src']}

        server, interceptor = _task_server()
        control = _parse_tool_result(
            await server._tool_manager.call_tool(
                'submit_task', _submit_kwargs(metadata=dict(dir_metadata))
            )
        )
        assert control.get('error_type') == 'LockCharterViolation', (
            f'control must be rejected by the lock-charter guard; got {control!r}'
        )

        server, interceptor = _task_server()
        result = _parse_tool_result(
            await server._tool_manager.call_tool(
                'submit_task',
                _submit_kwargs(
                    description=_leaked('description'), metadata=dict(dir_metadata)
                ),
            )
        )

        assert result.get('error_type') == 'ToolCallXmlLeakError', (
            f'the leak must be reported ahead of the lock-charter violation; got {result!r}'
        )
        interceptor.submit_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_leak_rejection_precedes_the_execution_class_guard(self):
        """PLACEMENT RATIONALE, pinned: ``inject_execution_class`` and
        ``inject_operational_routing`` DERIVE ROUTING FROM THE TEXT FIELDS. If
        the rejection ran after them, routing would be coerced from a
        description the harness already truncated — layering a second wrong
        value on the first. The execution-class guard immediately precedes that
        derivation, so beating it proves the guard runs before any text-derived
        metadata mutation.
        """
        recon = 'recon-stage-task_knowledge_sync'

        server, _ = _task_server()
        control = _parse_tool_result(
            await server._tool_manager.call_tool(
                'submit_task', _submit_kwargs(agent_id=recon, metadata={})
            )
        )
        assert control.get('error_type') == 'ValidationError', (
            f'control must be rejected by the execution-class guard; got {control!r}'
        )
        assert 'execution_class' in control.get('error', ''), f'got {control!r}'

        server, interceptor = _task_server()
        result = _parse_tool_result(
            await server._tool_manager.call_tool(
                'submit_task',
                _submit_kwargs(
                    agent_id=recon, metadata={}, description=_leaked('description')
                ),
            )
        )

        assert result.get('error_type') == 'ToolCallXmlLeakError', (
            f'the leak must be reported before routing is derived from the '
            f'truncated text; got {result!r}'
        )
        interceptor.submit_task.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'agent_id', [None, 'claude-interactive', 'recon-stage-task_knowledge_sync']
    )
    async def test_guard_is_not_scoped_to_recon_callers(self, agent_id):
        """Unlike premise_lint_error, this leak is caller-agnostic — it is a
        property of the harness serialization boundary, not of any agent — so
        scoping it would leave every non-recon write path unguarded.
        """
        server, interceptor = _task_server()

        result = await server._tool_manager.call_tool(
            'submit_task',
            _submit_kwargs(
                agent_id=agent_id,
                description=_leaked('description'),
                metadata={'execution_class': 'code_tdd'},
            ),
        )

        assert _parse_tool_result(result).get('error_type') == 'ToolCallXmlLeakError', (
            f'got {_parse_tool_result(result)!r}'
        )
        interceptor.submit_task.assert_not_called()
