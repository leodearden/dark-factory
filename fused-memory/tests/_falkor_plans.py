"""Pin the FalkorDB plan shape of the Cypher graphiti issues.

``spy_on_queries`` captures the Cypher a live driver issues, ``explain`` asks
FalkorDB how it would plan one captured query, and ``non_leaf_scans`` reads the
structured plan for scans that re-execute once per input row.
"""

from __future__ import annotations

from typing import Any

from _query_recorder import IssuedQuery
from falkordb.asyncio.graph import AsyncGraph
from falkordb.execution_plan import ExecutionPlan
from graphiti_core.driver.driver import GraphDriver


def spy_on_queries(driver: GraphDriver) -> list[IssuedQuery]:
    """Record every query *driver* issues while still executing it."""
    issued: list[IssuedQuery] = []
    execute = driver.execute_query

    async def spy(cypher: str, **params: Any):
        issued.append(IssuedQuery(cypher, params))
        return await execute(cypher, **params)

    driver.execute_query = spy  # pyright: ignore[reportAttributeAccessIssue]
    return issued


async def explain(graph: AsyncGraph, query: IssuedQuery) -> ExecutionPlan:
    params = {key: value for key, value in query.params.items() if key != 'routing_'}
    return await graph.explain(query.cypher, params)


def non_leaf_scans(plan: ExecutionPlan) -> list[str]:
    """Scan operators that have children; each is re-executed once per input row."""
    found: list[str] = []
    pending = [plan.structured_plan]
    while pending:
        operation = pending.pop()
        if operation.name.endswith('Scan') and operation.children:
            found.append(operation.name)
        pending.extend(operation.children)
    return found
