"""Pin the FalkorDB plan shape of the Cypher graphiti issues.

``spy_on_queries`` captures the Cypher a live driver issues, ``explain`` asks
FalkorDB how it would plan one captured query, ``profile`` runs a captured
query under GRAPH.PROFILE, ``non_leaf_scans`` reads the structured plan for
scans that re-execute once per input row, ``scan_operators`` lists every scan
the plan runs, and ``peak_records_produced`` reads the most records any
operator of a profiled plan produced. ``match_pattern`` and
``anchors_on_a_labelled_node`` are their static, connection-free counterparts:
they read the Cypher text itself for the pattern FalkorDB plans its scans from.
"""

from __future__ import annotations

import re
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


def _server_params(query: IssuedQuery) -> dict[str, Any]:
    """*query*'s params as FalkorDB takes them: graphiti's driver-side routing_ hint dropped."""
    return {key: value for key, value in query.params.items() if key != 'routing_'}


async def explain(graph: AsyncGraph, query: IssuedQuery) -> ExecutionPlan:
    return await graph.explain(query.cypher, _server_params(query))


async def profile(graph: AsyncGraph, query: IssuedQuery) -> ExecutionPlan:
    """Run *query* under GRAPH.PROFILE: its plan, with the records each operator produced."""
    return await graph.profile(query.cypher, _server_params(query))


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


def peak_records_produced(plan: ExecutionPlan) -> int:
    """The most records any one operator of a profiled *plan* produced."""
    return max(operation.records_produced for operation in _operations(plan))


def match_pattern(cypher: str) -> str:
    """The text between MATCH and the first WITH, from which FalkorDB plans its scans."""
    return cypher.split('MATCH', 1)[1].split('WITH', 1)[0]


def anchors_on_a_labelled_node(pattern: str) -> bool:
    """A labelled node such as ``(n:Entity)`` makes FalkorDB scan that label first."""
    return re.search(r'\(\s*\w*\s*:', pattern) is not None
