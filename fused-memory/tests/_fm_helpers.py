"""Non-fixture test helpers for fused-memory tests.

Lives outside conftest.py to avoid the `sys.modules['conftest']` collision
that arises when root-level pytest loads multiple subprojects' conftests in
the same process.  Each subproject exports its helpers under a unique
module name so test files can `from _fm_helpers import X` without
colliding with sibling subprojects' helpers.
"""

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock


@dataclass
class MockNode:
    """Simulates a Graphiti entity node (source/target of an edge)."""

    name: str
    uuid: str = ''
    labels: list[str] = field(default_factory=list)


@dataclass
class MockEdge:
    """Simulates a Graphiti entity edge returned from add_episode or search."""

    fact: str
    uuid: str = ''
    source_node: MockNode | None = None
    target_node: MockNode | None = None
    source_node_uuid: str = ''
    target_node_uuid: str = ''
    episodes: list[str] = field(default_factory=list)
    valid_at: Any = None
    invalid_at: Any = None


@dataclass
class MockAddEpisodeResult:
    """Simulates the AddEpisodeResults returned by Graphiti's add_episode.

    The real AddEpisodeResults class uses 'edges' as the field name.
    We keep 'entity_edges' for backward compat with existing tests that
    construct MockAddEpisodeResult(entity_edges=[...]).
    """

    entity_edges: list[MockEdge] = field(default_factory=list)
    edges: list[MockEdge] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.edges == [] and self.entity_edges:
            self.edges = list(self.entity_edges)


async def assert_ro_query_only(
    backend,
    make_graph_mock_fn,
    rows: list[list],
    method_name: str,
    *args,
    **kwargs,
) -> MagicMock:
    """Assert that a backend method uses ro_query and never calls query.

    Creates a graph mock via *make_graph_mock_fn*, wires it into
    *backend._driver._get_graph*, invokes the named method, then asserts:
      - graph.ro_query was awaited exactly once
      - graph.query was not awaited at all

    Returns the graph mock so callers can add additional assertions.
    """
    graph = make_graph_mock_fn(rows)
    backend._driver._get_graph = MagicMock(return_value=graph)
    await getattr(backend, method_name)(*args, **kwargs)
    graph.ro_query.assert_awaited_once()
    graph.query.assert_not_awaited()
    return graph


_REBUILD_DETAIL_NO_ERROR = object()  # sentinel — distinguishes "error not provided" from None


def make_rebuild_detail(
    uuid: str,
    name: str,
    *,
    old_summary: str = '',
    new_summary: str = '',
    edge_count: int = 0,
    status: str = 'rebuilt',
    error: Any = _REBUILD_DETAIL_NO_ERROR,
) -> dict:
    """Return a rebuild-detail dict for use in rebuild pipeline tests.

    Pass ``error=None`` (or any value) to include an 'error' key in the
    returned dict.  When omitted, 'error' is absent from the dict.
    """
    d: dict = {
        'uuid': uuid,
        'name': name,
        'old_summary': old_summary,
        'new_summary': new_summary,
        'edge_count': edge_count,
        'status': status,
    }
    if error is not _REBUILD_DETAIL_NO_ERROR:
        d['error'] = error
    return d


def extract_cypher(call_args: Any) -> str:
    """Return the Cypher query string from a mock call_args object.

    Checks positional args[0] first, then falls back to the 'query' keyword
    argument. Returns '' if neither is present.
    """
    if call_args.args:
        return call_args.args[0]
    return call_args.kwargs.get('query', '')


def extract_params(call_args: Any) -> dict:
    """Return the Cypher params dict from a mock call_args object.

    Checks positional args[1] first, then falls back to the 'params' keyword
    argument. Returns {} if neither is present.
    """
    if len(call_args.args) > 1:
        return call_args.args[1]
    return call_args.kwargs.get('params', {})


async def submit_and_resolve(
    interceptor,
    project_root: str,
    *,
    timeout_seconds: float = 30.0,
    **kwargs,
) -> dict:
    """Submit a task ticket and wait for the worker to resolve it.

    Reconstructs the legacy facade result shape from ``result_json`` so that
    migrated test assertions (``result['id']``, ``result['action']``, etc.)
    remain verbatim.  Designed as a mechanical drop-in for the removed
    ``TaskInterceptor.add_task`` facade in test code.

    Returns:
        The parsed ``result_json`` dict on success (keys: ``id``, ``title``,
        ``action``, etc. — the legacy add_task shape).
        When ``submit_task`` rejects the request (e.g. backlog gate, closed
        server), returns the submit-error dict directly so callers can assert
        on ``result.get('error')`` / ``result.get('error_type')``.

    Raises:
        AssertionError: When ``submit_task`` returns a non-dict value (e.g.
            ``None``).  Helper is test-only; loud failure is preferred over
            silent pass-through of an invalid contract.
        AssertionError: When the ticket resolved but the worker never wrote a
            ``result_json`` (row is None or result_json is empty).  The message
            names the ticket id and dumps ``resolve_result`` so the failure is
            diagnosable without digging through logs.
        AssertionError: When ``result_json`` exists but is not valid JSON.

    Args:
        interceptor: A ``TaskInterceptor`` instance (or compatible).
        project_root: Absolute path to the project root.
        timeout_seconds: How long to wait for the worker to resolve the ticket.
            Defaults to 30 s — generous enough for heavy-concurrency tests on
            loaded CI without being an indefinite wait.  Pass a smaller value
            for tests that intentionally exercise timeout paths.
        **kwargs: Forwarded verbatim to ``submit_task``.
    """
    submit_result = await interceptor.submit_task(project_root, **kwargs)
    assert isinstance(submit_result, dict), (
        f'submit_and_resolve: submit_task returned non-dict: {submit_result!r}'
    )
    if 'ticket' not in submit_result:
        return submit_result
    ticket = submit_result['ticket']
    resolve_result = await interceptor.resolve_ticket(
        ticket, project_root, timeout_seconds=timeout_seconds,
    )
    # TODO: reaching into interceptor._ticket_store is private-attribute coupling.
    # If TaskInterceptor ever exposes a stable accessor (e.g. get_ticket_result(ticket))
    # or if resolve_ticket starts returning the parsed result_json directly, prefer that
    # and remove the _ticket_store access here.
    row = await interceptor._ticket_store.get(ticket)
    if row is not None and row.get('result_json'):
        try:
            return json.loads(row['result_json'])
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f'submit_and_resolve: malformed result_json for ticket {ticket!r}: '
                f'{row["result_json"]!r}'
            ) from exc
    raise AssertionError(
        f'submit_and_resolve: ticket {ticket!r} resolved with no result_json '
        f'(resolve_result={resolve_result!r})'
    )
