"""Mock-graph doubles shared by the FalkorDB index-provisioning suites.

Two suites drive ``GraphitiBackend``'s index provisioning against mock graphs:
``test_ensure_indices.py`` pins β's diff-and-create, and
``test_index_provisioning_wiring.py`` pins γ's wiring of it into the startup
sweep and the first write.  Both build the same ``CALL db.indexes()`` rows and
read back the same issued statements, so those doubles live here once.

Rows are DERIVED from α's normal form (``IndexSpec`` sets, folded back through
``parse_index_statement``), never hard-coded: a graphiti change to the index set
then moves every fixture with it instead of leaving a test passing against a
stale expectation (INV-5: single home, never restate).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from unittest.mock import MagicMock

import redis.exceptions
from test_falkor_indices import LIVE_HEADER

from fused_memory.backends.falkor_indices import IndexSpec, parse_index_statement

#: MEASURED 2026-08-09: what ``CALL db.indexes()`` raises against a graph KEY that
#: does not exist yet.  Reproduced verbatim so the test exercises the real shape —
#: but note the IMPLEMENTATION must not key on this wording (D2); it decides
#: structurally, via ``list_graphs()`` membership.
EMPTY_KEY_ERROR = redis.exceptions.ResponseError('Invalid graph operation on empty key')


def rows_for(specs: set[IndexSpec]) -> list[list]:
    """Invert the normal form back into ``CALL db.indexes()``-shaped rows.

    FalkorDB merges every index on a label into ONE record, so specs are grouped
    by ``(label, entity_type)`` and each property's index types are collected into
    the ``types`` column — the same merged shape measured live 2026-08-06, where
    ``Entity`` comes back once carrying
    ``{'group_id': ['RANGE', 'FULLTEXT'], 'summary': ['FULLTEXT'], ...}``.

    Columns are emitted in ``LIVE_HEADER`` order::

        [label, properties, types, options, language, stopwords,
         entitytype, status, info]
    """
    grouped: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for label, entity_type, field, index_type in sorted(specs):
        grouped[(label, entity_type)][field].append(index_type)

    return [
        [
            label,                              # label
            list(types_by_field),               # properties
            {f: list(ts) for f, ts in types_by_field.items()},  # types
            {},                                 # options
            'english',                          # language
            [],                                 # stopwords
            entity_type,                        # entitytype
            'OPERATIONAL',                      # status
            {},                                 # info
        ]
        for (label, entity_type), types_by_field in grouped.items()
    ]


def statements_written(graph) -> list[str]:
    """The statements actually sent on the WRITE path, in order."""
    return [call.args[0] for call in graph.query.call_args_list]


def statements_read(graph) -> list[str]:
    """The statements actually sent on the READ-ONLY path, in order."""
    return [call.args[0] for call in graph.ro_query.call_args_list]


class StatefulGraph:
    """A graph double whose ``db.indexes()`` reflects the CREATEs it has accepted.

    The plain ``make_graph_mock`` returns a FIXED row set, so a second concurrent
    call sees the same pre-write state no matter what the first one did — which
    makes the race under test invisible.  This double closes that: each accepted
    statement is folded back into the reported index set through α's own parser,
    so the read side is a real function of the write side.

    Both methods ``await asyncio.sleep(0)``, and that is LOAD-BEARING, not
    decoration: an ``AsyncMock`` resolves without ever suspending, so two gathered
    calls would run strictly one-after-the-other and the test would pass against
    an unserialized implementation.  The explicit yield is what lets the two
    interleave at all.
    """

    def __init__(self, present: set[IndexSpec]):
        self.present = set(present)
        self.issued: list[str] = []

    @staticmethod
    def _result(rows):
        result = MagicMock()
        result.result_set = rows
        result.header = LIVE_HEADER
        return result

    async def ro_query(self, statement, *args, **kwargs):
        await asyncio.sleep(0)
        return self._result(rows_for(self.present))

    async def query(self, statement, *args, **kwargs):
        await asyncio.sleep(0)
        self.issued.append(statement)
        self.present |= set(parse_index_statement(statement))
        return self._result([])
