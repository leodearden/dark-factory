"""The active-edge invariant for graphiti RELATES_TO edges, in the two forms its readers need.

``active_edge_filter()`` is graphiti SearchFilters for reads through graphiti's
search; ``ACTIVE_EDGE_PREDICATE`` is the same filter compiled to a Cypher
predicate over the edge alias ``e``.
"""

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    SearchFilters,
    edge_search_filter_query_constructor,
)


# An edge is ACTIVE iff `invalid_at` IS NULL.  `expired_at` is deliberately NOT
# part of it: graphiti_core never sets `expired_at` without also setting
# `invalid_at` (graphiti_core/utils/maintenance/edge_operations.py::
# resolve_extracted_edge and ::resolve_edge_contradictions), so it retires
# nothing `invalid_at` misses — while the restore hooks clear `invalid_at` and
# keep `expired_at` as the restored-edge signature (redirect_node_edges), so an
# `expired_at` clause would hide exactly the edges they restore (task 4714,
# esc-4714-9).
#
# Every read answering "what is currently true?" applies this invariant: reads
# through graphiti's own search pass `active_edge_filter()`, and Cypher written
# against RELATES_TO interpolates `ACTIVE_EDGE_PREDICATE`, which is compiled
# from that filter.  Maintenance, audit, and uuid-keyed paths deliberately do
# NOT — they must see retired edges;
# `graphiti_client.py::GraphitiBackend.count_foreign_relationships` most
# emphatically so.
def active_edge_filter() -> SearchFilters:
    """The active-edge invariant as graphiti SearchFilters, for reads through graphiti's own search."""
    return SearchFilters(invalid_at=[[DateFilter(comparison_operator=ComparisonOperator.is_null)]])


def _compile_active_edge_predicate() -> str:
    [predicate], params = edge_search_filter_query_constructor(
        active_edge_filter(), GraphProvider.FALKORDB,
    )
    if params:
        raise RuntimeError(f'the active-edge predicate must bind no parameters, got {params!r}')
    return predicate


ACTIVE_EDGE_PREDICATE = _compile_active_edge_predicate()
