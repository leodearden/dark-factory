"""Pagination tests for the ``get_statuses`` MCP tool (task 3064).

``get_statuses`` is the reconciliation Stage 3 agent's PRIMARY task
enumerator.  On large projects the full ``{id: status}`` map exceeded the
MCP tool-response transport limit and the call failed CLOSED — the caller
got ZERO data (observed on reify at 5,603 / 5,680 / 5,845 tasks; payloads
of 80,795 and 84,638 chars against a ~62 KB documented-safe envelope).

These tests pin the fix at the MCP tool layer:
  * explicit ``page_size``/``offset`` pagination (shape-compatible with
    ``get_tasks``' pagination, tools.py, task 1727),
  * fail-OPEN auto-pagination on the un-paginated full-population path,
  * the opt-in ``pagination`` envelope (absent ⇒ response is complete),
  * deterministic page tiling — no gaps, no duplicates.

Fixtures mirror ``test_status_envelope_contract.py`` (the passthrough
main-checkout stub, the ids-filtering AsyncMock interceptor, and the
``_tool_manager.call_tool`` invocation pattern).

Runtime-behaviour assertions only — no docstring-wording assertions.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tools import create_mcp_server

# Mutable module-level population the fake interceptor serves.  Tests set it
# via _set_population() so a single fixture can back differently-sized cases.
_POPULATION: dict[str, str] = {}


def _make_statuses(n: int, *, start: int = 1, status: str = 'pending') -> dict[str, str]:
    """Build an ``{id: status}`` map of n entries with ids start..start+n-1."""
    return {str(i): status for i in range(start, start + n)}


def _set_population(mapping: dict[str, str]) -> dict[str, str]:
    """Replace the population the ``paging_task_interceptor`` fixture serves."""
    _POPULATION.clear()
    _POPULATION.update(mapping)
    return mapping


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch):
    """Stub resolve_main_checkout to pass its argument through unchanged.

    Mirrors test_status_envelope_contract.py's fixture of the same purpose —
    without this the real resolver rejects synthetic roots like '/project'
    because they are not real git working trees.
    """
    monkeypatch.setattr(
        'fused_memory.server.tools.resolve_main_checkout',
        lambda p: str(p),
    )


@pytest.fixture(autouse=True)
def _clean_population():
    """Each test starts from an empty population and leaves none behind."""
    _POPULATION.clear()
    yield
    _POPULATION.clear()


@pytest.fixture
def paging_task_interceptor():
    """AsyncMock task_interceptor whose get_statuses filters by ids like the
    real interceptor (mirrors test_status_envelope_contract.py's fixture)."""
    ti = AsyncMock()

    async def _get_statuses(project_root, ids=None, tag=None):
        if ids is None:
            return dict(_POPULATION)
        return {k: v for k, v in _POPULATION.items() if k in ids}

    ti.get_statuses = AsyncMock(side_effect=_get_statuses)
    return ti


@pytest.fixture
def paging_server(paging_task_interceptor):
    """MCP server wired with the paging interceptor."""
    mock_service = AsyncMock()
    return create_mcp_server(
        mock_service,
        task_interceptor=paging_task_interceptor,
    )
