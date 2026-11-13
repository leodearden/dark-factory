"""Test doubles for the bounded Cypher egress (``fused_memory/backends/graph_access.py``).

``FakeGraphAccess`` is the injected provider a test hands to
``GraphitiBackend.with_driver(..., graph_access=...)`` (or to
``make_backend(..., graph_access=...)``). It wraps the same raw graph doubles
``make_graph_mock`` builds in the production ``BudgetedGraph``, so the budget
under test is the real one. ``HangingRawGraph`` is the raw graph that never
answers, for proving a call is budgeted.

They live here rather than in ``conftest.py`` for the reason
``_graphiti_fake.py`` gives: a uniquely named ``tests/_*.py`` module cannot
collide with a sibling subproject's ``conftest`` when root-level pytest loads
several of them in one process.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from fused_memory.backends.graph_access import BudgetedGraph, GraphAccess, GraphHandle
from fused_memory.config.schema import QueueConfig

_DEFAULT_READ_BUDGET_S: float = QueueConfig.model_fields['backend_read_timeout_seconds'].default
_DEFAULT_WRITE_BUDGET_S: float = QueueConfig.model_fields['backend_write_timeout_seconds'].default


class FakeGraphAccess(GraphAccess):
    """A provider over a fixed mapping of group id to raw graph double."""

    def __init__(
        self,
        graphs: Mapping[str, Any],
        *,
        read_budget_s: float = _DEFAULT_READ_BUDGET_S,
        write_budget_s: float = _DEFAULT_WRITE_BUDGET_S,
    ) -> None:
        self._graphs = dict(graphs)
        self._read_budget_s = read_budget_s
        self._write_budget_s = write_budget_s

    def graph(self, group_id: str) -> GraphHandle:
        if group_id not in self._graphs:
            raise KeyError(f'no graph wired for {group_id!r}; wired: {sorted(self._graphs)}')
        return BudgetedGraph(
            self._graphs[group_id],
            name=group_id,
            read_budget_s=self._read_budget_s,
            write_budget_s=self._write_budget_s,
        )

    async def list_graphs(self) -> list[str]:
        return sorted(self._graphs)


async def wait_forever(*_args: Any, **_kwargs: Any) -> None:
    await asyncio.Event().wait()


class HangingRawGraph:
    """A raw falkordb-shaped graph whose every call waits forever."""

    def query(self, *args: Any) -> Any:
        return wait_forever(*args)

    def ro_query(self, *args: Any) -> Any:
        return wait_forever(*args)

    def execute_command(self, *args: Any) -> Any:
        return wait_forever(*args)
