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


# ---------------------------------------------------------------------------
# Explicit pagination (task 3064)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_statuses_explicit_pagination_slices_and_reports_metadata(
    paging_server,
):
    """get_statuses with page_size slices the status map and attaches a
    pagination envelope — the get_statuses analogue of
    test_task_tools.py::test_get_tasks_pagination_slices_and_reports_metadata.

    Sub-scenarios over a 5-entry population (ids '1'..'5'):
      (a) first page  → ids 1-2, has_more=True
      (b) last page   → id 5 only, has_more=False
      (c) beyond end  → empty map, returned=0, has_more=False (NOT an error)
      (d) the wrapped 'statuses' envelope survives pagination
    """
    _set_population(_make_statuses(5))

    # (a) First page: offset=0, page_size=2 → the numerically-first two ids
    result = await paging_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project', 'page_size': 2, 'offset': 0},
    )
    assert result.get('statuses') == {'1': 'pending', '2': 'pending'}, (
        f'Expected first 2 statuses, got: {result.get("statuses")}'
    )
    assert result.get('pagination') == {
        'total': 5,
        'offset': 0,
        'page_size': 2,
        'returned': 2,
        'has_more': True,
        'auto_paginated': False,
    }, f'Unexpected pagination dict: {result.get("pagination")}'

    # (d) The wrapped envelope survives — the deliberate get_statuses vs
    # get_external_statuses asymmetry pinned by test_status_envelope_contract.py
    assert 'statuses' in result, f"Expected wrapped 'statuses' key, got: {result!r}"
    assert isinstance(result['statuses'], dict), f'Expected a dict, got: {result["statuses"]!r}'

    # (b) Last item: offset=4, page_size=2 → only id 5
    result2 = await paging_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project', 'page_size': 2, 'offset': 4},
    )
    assert result2.get('statuses') == {'5': 'pending'}, (
        f'Expected last status only, got: {result2.get("statuses")}'
    )
    assert result2['pagination']['returned'] == 1
    assert result2['pagination']['has_more'] is False
    assert result2['pagination']['total'] == 5

    # (c) Past end: offset=10 → empty map, not an error
    result3 = await paging_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project', 'page_size': 2, 'offset': 10},
    )
    assert result3.get('statuses') == {}, f'Expected empty map, got: {result3.get("statuses")}'
    assert result3['pagination']['returned'] == 0
    assert result3['pagination']['has_more'] is False


@pytest.mark.asyncio
async def test_get_statuses_pagination_validation_and_backward_compat(
    paging_server, paging_task_interceptor
):
    """get_statuses pagination: backward-compat + input validation.

    The get_statuses analogue of
    test_task_tools.py::test_get_tasks_pagination_validation_and_backward_compat.

    (a) Backward-compat: no page_size, no ids → full map, keys == {'statuses'}.
    (b) page_size=0    → ValidationError, interceptor NOT called.
    (c) page_size=-1   → ValidationError, interceptor NOT called.
    (d) page_size=True → ValidationError (bool is an int subclass), NOT called.
    (e) offset=-1 with page_size=2 → ValidationError, interceptor NOT called.

    Every invalid case must early-exit BEFORE the interceptor is touched.
    """
    population = _set_population(_make_statuses(5))

    # (a) Backward-compat: default call (no page_size, no ids).  This pins,
    # inside this file, the single-keyed envelope invariant that
    # test_status_envelope_contract.py:110 depends on.
    result = await paging_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project'},
    )
    assert result.get('statuses') == population, (
        f'Backward-compat: full map expected, got: {result.get("statuses")}'
    )
    assert set(result.keys()) == {'statuses'}, (
        f'Backward-compat: pagination key must be absent, got: {result}'
    )

    # (b) page_size=0 → ValidationError
    paging_task_interceptor.get_statuses.reset_mock()
    bad0 = await paging_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project', 'page_size': 0},
    )
    assert bad0.get('error_type') == 'ValidationError', f'Expected ValidationError for page_size=0, got: {bad0}'
    assert 'page_size' in bad0.get('error', '').lower(), f'Error message should mention page_size: {bad0}'
    paging_task_interceptor.get_statuses.assert_not_awaited()

    # (c) page_size=-1 → ValidationError
    paging_task_interceptor.get_statuses.reset_mock()
    bad_neg = await paging_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project', 'page_size': -1},
    )
    assert bad_neg.get('error_type') == 'ValidationError', f'Expected ValidationError for page_size=-1, got: {bad_neg}'
    paging_task_interceptor.get_statuses.assert_not_awaited()

    # (d) page_size=True / offset=True → ValidationError (bool is an int
    # subclass, so an unguarded `isinstance(x, int)` would accept a flag as a
    # page size).  Asserted against the tool FUNCTION rather than through
    # _tool_manager.call_tool: the MCP boundary declares `page_size: int | None`,
    # so pydantic coerces True → 1 before the guard can ever see a bool.  That
    # coercion is pre-existing and identical for get_tasks, whose guards these
    # are copied from verbatim — matching it is the point, so the bool rejection
    # is pinned at the layer where it is actually reachable (any in-process
    # caller of the tool function).
    paging_task_interceptor.get_statuses.reset_mock()
    get_statuses_fn = paging_server._tool_manager._tools['get_statuses'].fn

    bad_bool = await get_statuses_fn(project_root='/project', page_size=True)
    assert bad_bool.get('error_type') == 'ValidationError', f'Expected ValidationError for page_size=True, got: {bad_bool}'
    assert 'page_size' in bad_bool.get('error', '').lower(), f'Error message should mention page_size: {bad_bool}'

    bad_bool_off = await get_statuses_fn(project_root='/project', page_size=2, offset=True)
    assert bad_bool_off.get('error_type') == 'ValidationError', f'Expected ValidationError for offset=True, got: {bad_bool_off}'
    assert 'offset' in bad_bool_off.get('error', '').lower(), f'Error message should mention offset: {bad_bool_off}'

    paging_task_interceptor.get_statuses.assert_not_awaited()

    # (e) offset=-1 with page_size=2 → ValidationError
    paging_task_interceptor.get_statuses.reset_mock()
    bad_off = await paging_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project', 'page_size': 2, 'offset': -1},
    )
    assert bad_off.get('error_type') == 'ValidationError', f'Expected ValidationError for offset=-1, got: {bad_off}'
    assert 'offset' in bad_off.get('error', '').lower(), f'Error message should mention offset: {bad_off}'
    paging_task_interceptor.get_statuses.assert_not_awaited()


# ---------------------------------------------------------------------------
# Fail-open auto-pagination — the core defect (task 3064)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_statuses_auto_paginates_oversized_full_population(paging_server):
    """An un-paginated full-population call DEGRADES to a first page.

    This is the defect: on reify (5,603 / 5,680 / 5,845 tasks) the full map
    exceeded the MCP tool-response transport limit and the call failed CLOSED —
    the caller got ZERO data.  A caller that would happily accept a first page
    got nothing at all.

    The fix degrades fail-OPEN: real data plus a loud, structured continuation
    marker (``auto_paginated: True``, ``has_more: True``), never silent
    truncation that would look like a complete census.

    Asserted against the real ``_STATUSES_AUTO_PAGE_LIMIT`` constant, never a
    hard-coded magic number, so the test tracks the implementation's own bound.
    """
    from fused_memory.server.tools import _STATUSES_AUTO_PAGE_LIMIT as LIMIT

    population = _set_population(_make_statuses(LIMIT + 5))

    result = await paging_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project'},
    )

    # (a) Non-empty (the fail-closed regression) and not the whole population.
    assert result['statuses'] != {}, 'Fail-closed regression: auto-paged response must not be empty'
    assert len(result['statuses']) == LIMIT, (
        f'Expected a {LIMIT}-entry first page, got {len(result["statuses"])}'
    )

    # (b) A loud, structured continuation marker rather than silent truncation.
    assert result.get('pagination') == {
        'total': LIMIT + 5,
        'offset': 0,
        'page_size': LIMIT,
        'returned': LIMIT,
        'has_more': True,
        'auto_paginated': True,
    }, f'Unexpected pagination dict: {result.get("pagination")}'

    # (c) The page is exactly the numerically-first LIMIT ids.
    expected_first = {str(i) for i in range(1, LIMIT + 1)}
    assert set(result['statuses'].keys()) == expected_first, (
        'Auto-page must be the numerically-first ids, not an arbitrary dict slice'
    )

    # (d) Tiling property: page 2 completes the census with no gaps, no dupes.
    result2 = await paging_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project', 'offset': LIMIT, 'page_size': LIMIT},
    )
    assert len(result2['statuses']) == 5, f'Expected the remaining 5, got {len(result2["statuses"])}'
    assert result2['pagination']['has_more'] is False

    page1, page2 = set(result['statuses']), set(result2['statuses'])
    assert page1 & page2 == set(), f'Pages must not overlap, shared: {page1 & page2}'
    assert page1 | page2 == set(population), 'Pages must tile the full population with no gaps'


@pytest.mark.asyncio
async def test_get_statuses_ids_path_not_auto_capped_but_paginable(
    paging_server, paging_task_interceptor
):
    """The auto-cap fires ONLY on the ids-less path; ids-filtered stays complete.

    A caller that enumerated an explicit id set depends on getting an answer for
    each one — silently dropping the tail would trade a loud transport failure
    for a quiet correctness bug.  The ids-less enumeration is the unbounded one
    (it grows with the project) and is the path that actually fails closed.

    Guards against an implementation that caps every path indiscriminately.
    """
    from fused_memory.server.tools import _STATUSES_AUTO_PAGE_LIMIT as LIMIT

    population = _set_population(_make_statuses(LIMIT + 5))
    all_ids = list(population)

    # (a) Oversized ids list, no page_size → every named id present, no cap.
    result = await paging_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project', 'ids': all_ids},
    )
    assert len(result['statuses']) == LIMIT + 5, (
        f'ids path must return every named id, got {len(result["statuses"])} of {LIMIT + 5}'
    )
    assert 'pagination' not in result, (
        f'ids path must not be auto-capped, got: {result.get("pagination")}'
    )

    # (b) Explicit pagination still works on the ids path.
    result2 = await paging_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project', 'ids': all_ids, 'page_size': 2, 'offset': 0},
    )
    assert len(result2['statuses']) == 2, f'Expected a 2-entry page, got: {result2["statuses"]}'
    assert result2['pagination']['total'] == LIMIT + 5
    assert result2['pagination']['has_more'] is True
    assert result2['pagination']['auto_paginated'] is False

    # (c) ids is forwarded to the interceptor unchanged — pagination must not
    # alter what is asked of the backend.
    forwarded = paging_task_interceptor.get_statuses.await_args.kwargs
    assert forwarded['ids'] == all_ids, (
        f'ids must reach the interceptor unchanged, got: {forwarded["ids"]!r}'
    )


@pytest.mark.asyncio
async def test_auto_page_limit_fits_documented_safe_envelope(paging_server):
    """A WORST-CASE full auto-page must serialise inside the safe envelope.

    Derivation of the 62,000-char bound (auditable, not magic):
      * get_statuses failed closed on reify at payloads of 80,795 and 84,638
        chars — the MCP transport rejected them wholesale.
      * ~62 KB is the documented-safe envelope from the same incident record.
      * So the wall sits between 62 KB and ~80 KB; 62,000 chars is the
        conservative side of it.

    Worst case is modelled as 4-digit ids plus the longest realistic status
    string ('in-progress'), which is the densest shape a real project produces.

    This asserts on the REAL constant and the REAL serialised tool response, so
    a future bump to _STATUSES_AUTO_PAGE_LIMIT cannot silently re-cross the wall
    that cost reify three consecutive reconciliation cycles.
    """
    import json

    from fused_memory.server.tools import _STATUSES_AUTO_PAGE_LIMIT as LIMIT

    # 4-digit ids (start at 1000) + the longest realistic status string.
    _set_population(
        _make_statuses(LIMIT + 5, start=1000, status='in-progress')
    )

    result = await paging_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project'},
    )
    # Precondition: this really is a full auto-page, not a short one.
    assert result['pagination']['auto_paginated'] is True
    assert result['pagination']['returned'] == LIMIT

    serialised = json.dumps(result)
    assert len(serialised) < 62_000, (
        f'A worst-case full auto-page serialises to {len(serialised)} chars, at or '
        f'over the {62_000}-char documented-safe envelope. Lower '
        f'_STATUSES_AUTO_PAGE_LIMIT (currently {LIMIT}) — do NOT relax this bound.'
    )
