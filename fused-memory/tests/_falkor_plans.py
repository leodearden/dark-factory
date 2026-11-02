"""Pin the FalkorDB plan shape of the Cypher graphiti issues.

``spy_on_queries`` captures the Cypher a live driver issues, ``explain`` asks
FalkorDB how it would plan one captured query, ``non_leaf_scans`` reads the
structured plan for scans that re-execute once per input row, and
``scan_operators`` lists every scan the plan runs.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from _query_recorder import IssuedQuery
from falkordb.asyncio.graph import AsyncGraph
from falkordb.execution_plan import ExecutionPlan, Operation
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


def _operations(plan: ExecutionPlan) -> Iterator[Operation]:
    pending = [plan.structured_plan]
    while pending:
        operation = pending.pop()
        yield operation
        pending.extend(operation.children)


def non_leaf_scans(plan: ExecutionPlan) -> list[str]:
    """Scan operators that have children; each is re-executed once per input row."""
    return [
        operation.name
        for operation in _operations(plan)
        if operation.name.endswith('Scan') and operation.children
    ]


def scan_operators(plan: ExecutionPlan) -> list[str]:
    """Every scan operator in *plan*, leaf or not."""
    return [operation.name for operation in _operations(plan) if operation.name.endswith('Scan')]
