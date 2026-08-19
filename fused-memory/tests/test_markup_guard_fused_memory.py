"""Boundary markup-guard tests for fused-memory's BUNDLED FastMCP (task 4458).

PRD ``plans/toolcall-markup-containment-prd.md``, leaf gamma-3. Task 3689
(beta) delivered :class:`shared.mcp_markup_middleware.MarkupGuardMiddleware`
against the STANDALONE ``fastmcp`` package. fused-memory runs the FastMCP
BUNDLED inside the ``mcp`` SDK, whose ``FastMCP`` has no ``add_middleware`` and
no ``get_tool``, so the middleware cannot be attached the documented way.
:mod:`fused_memory.server.markup_guard` adapts it to the bundled
``mcp._tool_manager.call_tool`` chokepoint; these tests pin that adaptation.

What is asserted here that the retiring in-line ``_markup_gate`` could not do:

* ``add_system_record`` and ``update_memory`` — two write tools that had NO
  gate at all, so a leaked ``content`` that swallowed a trailing OPTIONAL
  parameter was stored with that parameter silently ``None``.
* ``repaired_call`` — the COMPLETE argument map with the absorbed sibling
  recovered verbatim, which a single-literal write-time guard cannot produce.

AUTHORING RULE, binding on this whole file: never write a raw MCP envelope
sentinel literal into a source, test or doc file — a file that contains one
becomes a specimen of the very corruption under test and trips the read-side
prefilter. Specimens are BUILT from ``shared.toolcall_markup``'s own constants
by :func:`_leaked`, exactly as ``shared/tests/test_toolcall_markup.py`` and the
middleware itself already do.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastmcp.exceptions import ToolError
from shared.mcp_markup_middleware import RepairPolicy
from shared.toolcall_markup import CANONICAL_OPENER_PREFIX, closer_for

from fused_memory.server.markup_guard import install_markup_guard
from fused_memory.server.tools import create_mcp_server

_PROJECT_ID = 'dark_factory'
_PROJECT_ROOT = '/project'

#: The guard's project-attribution map (bare id -> root), mirroring the
#: ``_known_projects_map`` main.py passes at the real registration site.
_KNOWN_PROJECTS = {_PROJECT_ID: _PROJECT_ROOT}

#: add_system_record is recon-stage-only by agent_id convention (its
#: authorization gate runs FIRST in the tool body), so the specimen's swallowed
#: agent_id is a recon-stage one. That keeps the negative control meaningful:
#: a clean call with this identity actually reaches the service instead of
#: bouncing off the authorization gate before the write.
_AGENT_ID = 'recon-stage-9'
_CLEAN_CONTENT = 'a deterministic cycle summary for the merge lane'


def _leaked(clean: str, param: str, value: str) -> str:
    """Build a specimen: *clean* text that mis-closed and absorbed *param*.

    The corpus shape (PRD section 2.1): the caller's text emission closed the
    wrong tag, so everything the envelope emitted AFTER that point — the next
    parameter's opener, name and value — landed inside this argument's string
    instead of being parsed as its own argument.

    Assembled from ``closer_for`` / ``CANONICAL_OPENER_PREFIX`` so no raw
    sentinel is authored here.
    """
    return (
        clean
        + closer_for('parameter')
        + CANONICAL_OPENER_PREFIX
        + f'"{param}">'
        + value
        + closer_for('parameter')
    )


def _pass_through(mock_service: AsyncMock, method: str) -> None:
    """Give *method*'s return value a real ``model_dump``.

    An unspecced AsyncMock chains AsyncMock all the way down, so
    ``result.model_dump()`` would be an unawaited coroutine unless the return
    value is an explicit MagicMock (mirrors
    tests/server/test_markup_tripwire_gate.py::_pass_through).
    """
    result = MagicMock()
    result.model_dump.return_value = {'id': 'ok'}
    getattr(mock_service, method).return_value = result


def _build_guarded_server(*methods: str) -> tuple[Any, AsyncMock]:
    """A real bundled-FastMCP server with the boundary guard installed.

    Shape copied from tests/test_tool_safe_wrapper.py::_build_server_with_tool:
    the guard is installed by ``main.py``, NOT by ``create_mcp_server``, so a
    test must install it explicitly — the same shape
    ``_install_safe_tool_wrapper`` already has.
    """
    mock_service = AsyncMock()
    for method in methods:
        _pass_through(mock_service, method)
    server = create_mcp_server(mock_service)
    install_markup_guard(
        server,
        policy=RepairPolicy.REJECT_WITH_REPAIR,
        known_projects=_KNOWN_PROJECTS,
    )
    return server, mock_service


def _payload(exc_info) -> dict:
    """The guard's structured rejection, parsed out of the raised ToolError.

    The guard RAISES rather than returning a dict (prototype P4: a returned
    dict is destroyed by the output schema of any tool annotated ``-> str``),
    so the payload travels as the exception's ``json.dumps`` message.
    """
    return json.loads(str(exc_info.value))


class TestAddSystemRecordBoundary:
    """add_system_record — the first of the two tools that had NO in-line gate.

    Its ``agent_id`` is OPTIONAL and trailing, which is the shape where the
    swallow is SILENT: pydantic defaults the eaten parameter to ``None`` and
    the write lands with a null identity and an isError=False response.
    """

    @pytest.mark.asyncio
    async def test_leaked_content_is_rejected_with_the_absorbed_agent_id_recovered(self):
        server, mock_service = _build_guarded_server('add_system_record')
        arguments = {
            'content': _leaked(_CLEAN_CONTENT, 'agent_id', _AGENT_ID),
            'project_id': _PROJECT_ID,
            'category': 'observations_and_summaries',
        }

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool('add_system_record', arguments)

        payload = _payload(exc_info)
        assert payload['error_type'] == 'mcp_markup_detected'
        assert payload['tool'] == 'add_system_record'
        assert payload['field'] == 'content'
        assert payload['recovered_params'] == ['agent_id']
        # The COMPLETE argument map, resubmittable verbatim: content restored
        # to the clean prefix and the swallowed agent_id recovered VERBATIM.
        assert payload['repaired_call'] == {
            'content': _CLEAN_CONTENT,
            'project_id': _PROJECT_ID,
            'category': 'observations_and_summaries',
            'agent_id': _AGENT_ID,
        }

    @pytest.mark.asyncio
    async def test_rejected_write_never_reaches_the_service(self):
        """Nothing written — asserted at the service, not by reading storage."""
        server, mock_service = _build_guarded_server('add_system_record')

        with pytest.raises(ToolError):
            await server._tool_manager.call_tool(
                'add_system_record',
                {
                    'content': _leaked(_CLEAN_CONTENT, 'agent_id', _AGENT_ID),
                    'project_id': _PROJECT_ID,
                    'category': 'observations_and_summaries',
                },
            )

        mock_service.add_system_record.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clean_content_reaches_the_service_unchanged(self):
        """Negative control: the guard sits on EVERY call, so a clean one must
        pass through untouched. Without this the suite above would also pass
        against a guard that rejected everything."""
        server, mock_service = _build_guarded_server('add_system_record')

        result = await server._tool_manager.call_tool(
            'add_system_record',
            {
                'content': _CLEAN_CONTENT,
                'project_id': _PROJECT_ID,
                'category': 'observations_and_summaries',
                'agent_id': _AGENT_ID,
            },
        )

        assert result == {'id': 'ok'}
        mock_service.add_system_record.assert_awaited_once()
        assert mock_service.add_system_record.await_args.kwargs['content'] == _CLEAN_CONTENT
        assert mock_service.add_system_record.await_args.kwargs['agent_id'] == _AGENT_ID
