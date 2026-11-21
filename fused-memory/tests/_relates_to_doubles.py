"""Answer and inspect the hand-written RELATES_TO Cypher of
``fused_memory.maintenance.cross_graph_move`` by column/property NAME, so tests
pin which properties are read and written rather than RETURN-column order.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

_EDGE_PROPERTY_COLUMN = re.compile(r'e\.(\w+)')
_EDGE_SET_ASSIGNMENT = re.compile(r'\br\.(\w+)\s*=\s*\$(\w+)')


@dataclass(frozen=True)
class EdgeFixture:
    """A source-graph RELATES_TO edge: its stored properties plus its true direction."""

    src_uuid: str
    dst_uuid: str
    properties: Mapping[str, Any]


def answer_edge_read(cypher: str, *edges: EdgeFixture) -> MagicMock:
    """One row per edge, each cell answering the column *cypher* RETURNs in that position."""
    columns = [column.strip() for column in cypher.rsplit('RETURN', 1)[-1].split(',')]
    return MagicMock(result_set=[[_cell(edge, column, cypher) for column in columns] for edge in edges])


def _cell(edge: EdgeFixture, column: str, cypher: str) -> Any:
    if column == 's.uuid':
        return edge.src_uuid
    if column == 't.uuid':
        return edge.dst_uuid
    match = _EDGE_PROPERTY_COLUMN.fullmatch(column)
    if match is None:
        raise AssertionError(f'answer_edge_read cannot answer column {column!r} of {cypher!r}')
    return edge.properties.get(match.group(1))


def written_edge_properties(cypher: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Each ``r.<prop> = $<param>`` SET assignment in *cypher*, resolved to its bound value."""
    return {prop: params[param] for prop, param in _EDGE_SET_ASSIGNMENT.findall(cypher)}
