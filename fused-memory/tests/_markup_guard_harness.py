"""Builders for fused-memory's markup-guard tests: one guarded server, its refusals.

Shared by ``tests/test_markup_guard_fused_memory.py``, which pins the boundary
guard, and ``tests/test_markup_residue_tiers.py``, which pins where a refused
payload is preserved. It lives outside both, and outside conftest.py for the
reason ``tests/_ast_guard.py`` gives, so neither file reaches into the other.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from shared.mcp_markup_middleware import RepairPolicy

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


def _build_guarded_server(
    *methods: str,
    known_projects: dict[str, str] | None = None,
    residue_floor: Any = None,
) -> tuple[Any, AsyncMock]:
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
        known_projects=_KNOWN_PROJECTS if known_projects is None else known_projects,
        residue_floor=residue_floor,
    )
    return server, mock_service


def _payload(exc_info) -> dict:
    """The guard's structured rejection, parsed out of the raised ToolError.

    The guard RAISES rather than returning a dict (prototype P4: a returned
    dict is destroyed by the output schema of any tool annotated ``-> str``),
    so the payload travels as the exception's ``json.dumps`` message.
    """
    return json.loads(str(exc_info.value))


def _escalations(root) -> list[dict]:
    """Every escalation record filed under *root*, parsed."""
    return [
        json.loads(path.read_text())
        for path in sorted((root / 'data' / 'escalations').glob('esc-*.json'))
    ]
