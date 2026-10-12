"""Graph drivers that record the Cypher graphiti issues, with no connection opened.

``recording_driver`` builds an uninitialised driver (``object.__new__``) whose
only stub is ``execute_query``.  ``provider`` and ``search_interface`` are class
attributes, so graphiti's real query assembly and dispatch run unchanged.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from graphiti_core.driver.driver import GraphDriver

DriverT = TypeVar('DriverT', bound=GraphDriver)


@dataclass(frozen=True)
class IssuedQuery:
    cypher: str
    params: dict[str, Any]


def recording_driver(
    driver_class: type[DriverT], records: Sequence[dict[str, Any]] = ()
) -> tuple[DriverT, list[IssuedQuery]]:
    """An unconnected *driver_class* whose ``execute_query`` records each call and returns *records*."""
    driver = object.__new__(driver_class)
    issued: list[IssuedQuery] = []

    async def record(cypher: str, **params: Any):
        issued.append(IssuedQuery(cypher, params))
        return copy.deepcopy(list(records)), [], None

    driver.execute_query = record  # pyright: ignore[reportAttributeAccessIssue]
    return driver, issued
