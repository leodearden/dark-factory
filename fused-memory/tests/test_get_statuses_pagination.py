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
    """An OPT-IN oversized full-population call DEGRADES to a first page.

    This is the defect: on reify (5,603 / 5,680 / 5,845 tasks) the full map
    exceeded the MCP tool-response transport limit and the call failed CLOSED —
    the caller got ZERO data.  A caller that would happily accept a first page
    got nothing at all.

    The fix degrades fail-OPEN: real data plus a loud, structured continuation
    marker (``auto_paginated: True``, ``has_more: True``), never silent
    truncation that would look like a complete census.

    The degradation is OPT-IN (``auto_paginate=True``) rather than automatic.
    Passing the flag IS the caller's assertion that it inspects
    ``pagination['has_more']`` and will page to completion.  Callers that cannot
    see a ``pagination`` marker must keep the complete map they have always
    received — see
    test_full_population_census_complete_for_programmatic_callers for the two
    live consumers that depend on that, and why truncating them by default
    would be a correctness regression rather than a fix.

    Asserted against the real ``_STATUSES_AUTO_PAGE_LIMIT`` constant, never a
    hard-coded magic number, so the test tracks the implementation's own bound.
    """
    from fused_memory.server.tools import _STATUSES_AUTO_PAGE_LIMIT as LIMIT

    population = _set_population(_make_statuses(LIMIT + 5))

    result = await paging_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project', 'auto_paginate': True},
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

    # (d) The two gates are INDEPENDENT: opting into auto-pagination must not be
    # a back-door that caps the caller-bounded ids path.  ``auto_paginate=True``
    # says "I can handle a page if you must truncate the unbounded
    # enumeration" — it does not license dropping ids the caller explicitly
    # named and is depending on an answer for.
    result3 = await paging_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project', 'ids': all_ids, 'auto_paginate': True},
    )
    assert 'pagination' not in result3, (
        f'auto_paginate must not cap the ids path, got: {result3.get("pagination")}'
    )
    assert len(result3['statuses']) == LIMIT + 5, (
        f'ids path must stay complete under auto_paginate=True, got '
        f'{len(result3["statuses"])} of {LIMIT + 5}'
    )


@pytest.mark.asyncio
async def test_full_population_census_complete_for_programmatic_callers(paging_server):
    """A >LIMIT population still yields a COMPLETE census by default.

    Auto-pagination is opt-in precisely because of the two live out-of-process
    programmatic consumers below.  Both call this MCP tool over plain HTTP with
    ``{'project_root': ...}`` and nothing else, both read ONLY the ``statuses``
    key, and neither can see a ``pagination`` marker.  Neither crosses the
    token-limited LLM tool-response channel that produced the original
    incident, so truncating them buys nothing and costs correctness.

    Invariant pinned here: **never make truncation the default for a caller that
    did not ask for it.**

    The argument shape asserted against is the exact one those consumers send —
    if a future change makes the auto branch fire without ``auto_paginate``,
    this test fails before the damage below can ship.
    """
    from fused_memory.server.tools import _STATUSES_AUTO_PAGE_LIMIT as LIMIT

    population = _set_population(_make_statuses(LIMIT + 500))

    # The EXACT argument shape both programmatic consumers send: project_root
    # only — no ids, no page_size, no auto_paginate.
    result = await paging_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project'},
    )

    # (a) The COMPLETE population, not a first page.
    assert len(result['statuses']) == LIMIT + 500, (
        f'Default (non-opted-in) call must return the complete census, got '
        f'{len(result["statuses"])} of {LIMIT + 500} — silent truncation'
    )

    # (b) The response shape stays byte-identical to what these callers have
    # always received: a single-keyed envelope with no pagination marker.
    assert set(result.keys()) == {'statuses'}, (
        f'Expected a bare {{\'statuses\'}} envelope, got keys: {sorted(result.keys())}'
    )

    # (c) Simulate Scheduler.get_statuses (orchestrator/src/orchestrator/
    # scheduler.py:2523 → parse_tool_result(result, 'statuses', dict) at :2545,
    # dispatched with NO ids from harness.py 2106/2111/2214/2326/3568/3738/4330).
    # It extracts the 'statuses' key alone and treats it as a COMPLETE census.
    #
    # Concrete damage a truncated page would cause: harness.py:3636 reads
    # `elif bare_id not in live:` as "task deleted" and calls
    # detach_lane_checkout (harness.py:3643) — it would detach the lane
    # checkouts of LIVE tasks.  The fail-safe does NOT cover this: the
    # `degraded` guard at harness.py:3569 is resolver_failed(statuses, err),
    # and a truncated page is non-empty with err is None, so degraded is False
    # and the guard never fires.
    scheduler_view = result['statuses']
    missing = set(population) - set(scheduler_view)
    assert missing == set(), (
        f'{len(missing)} live task ids absent from the scheduler view — '
        f'harness.py:3636 would misread these as deleted and detach their lane '
        f'checkouts (sample: {sorted(missing, key=int)[:5]})'
    )

    # (d) Simulate dashboard.data.tasks.fetch_statuses (dashboard/src/dashboard/
    # data/tasks.py:220, which rebuilds {int(id): status} at :239-244 and also
    # ignores 'pagination').  A truncated map permanently under-counts every
    # burndown snapshot.
    dashboard_view = {int(k): v for k, v in result.get('statuses', {}).items()}
    assert len(dashboard_view) == len(population), (
        f'Burndown under-count: dashboard view has {len(dashboard_view)} of '
        f'{len(population)} tasks'
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
        {'project_root': '/project', 'auto_paginate': True},
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


@pytest.mark.asyncio
async def test_get_statuses_non_dict_result_skips_pagination(
    paging_server, paging_task_interceptor
):
    """A non-dict interceptor result passes through untouched, no slicing error.

    Mirrors the documented rationale of get_tasks' ``isinstance(all_tasks, list)``
    guard: a non-standard backend returning the wrong shape must surface as the
    real failure at the caller, not be masked by a generic TypeError raised from
    inside the pagination code.
    """
    bogus = ['not', 'a', 'mapping']
    paging_task_interceptor.get_statuses = AsyncMock(return_value=bogus)

    # (a) No page_size — the pass-through path.
    result = await paging_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project'},
    )
    assert result == {'statuses': bogus}, f'Non-dict result must pass through, got: {result}'
    assert 'pagination' not in result

    # (b) With page_size — pagination is SKIPPED rather than attempting to sort
    # or slice a non-mapping.
    result2 = await paging_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project', 'page_size': 2},
    )
    assert result2 == {'statuses': bogus}, (
        f'Non-dict result must pass through even with page_size, got: {result2}'
    )
    assert 'pagination' not in result2


# ---------------------------------------------------------------------------
# auto_paginate must honour offset — no livelock (task 3064, review amendment)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_paginate_honours_offset_so_the_paging_loop_terminates(
    paging_server,
):
    """An ``auto_paginate`` paging loop must make forward progress and finish.

    Regression: the auto branch originally hard-coded ``offset`` 0 and silently
    dropped a caller-supplied offset.  A caller that opted in, saw
    ``auto_paginated: True`` / ``has_more: True``, and then continued paging as
    the contract instructs — but forgot to ALSO pass ``page_size``, an easy
    mistake for the LLM Stage 3 caller this fallback exists for — got the SAME
    first page back, again with ``has_more: True``, forever.  The loop never
    terminates and the census never completes: task 3064's LOUD transport
    failure converted into a SILENT livelock, which is the worse of the two.

    Driven as a real paging loop rather than two spot-checks, because the
    property under test is termination, not the content of any single page.
    """
    from fused_memory.server.tools import _STATUSES_AUTO_PAGE_LIMIT as LIMIT

    population = _set_population(_make_statuses(LIMIT + 500))

    # (a) The loop the docstring tells callers to write, with page_size OMITTED
    # on every call — the exact shape that used to livelock.  Bounded so a
    # regression fails the test instead of hanging the suite.
    merged: dict[str, str] = {}
    pages: list[dict] = []
    offset = 0
    max_pages = 10
    for _ in range(max_pages):
        page_result = await paging_server._tool_manager.call_tool(
            'get_statuses',
            {'project_root': '/project', 'auto_paginate': True, 'offset': offset},
        )
        merged.update(page_result['statuses'])
        meta = page_result.get('pagination')
        if meta is None or not meta['has_more']:
            break
        pages.append(meta)
        # Advance by what was actually SERVED, per the documented contract.
        offset += meta['page_size']
    else:
        pytest.fail(
            f'auto_paginate loop did not terminate within {max_pages} pages — '
            f'livelock: offsets served {[p["offset"] for p in pages]}'
        )

    # (b) It terminated having seen the COMPLETE census, no gaps, no dupes.
    assert merged.keys() == population.keys(), (
        f'Paging loop assembled {len(merged)} of {len(population)} ids — '
        f'missing {len(set(population) - set(merged))}'
    )

    # (c) Two pages exactly (LIMIT + 500 over a LIMIT-sized page), and the
    # second one really started where the first ended — the offset was honoured,
    # not silently reset to 0.
    second = await paging_server._tool_manager.call_tool(
        'get_statuses',
        {'project_root': '/project', 'auto_paginate': True, 'offset': LIMIT},
    )
    assert second['pagination'] == {
        'total': LIMIT + 500,
        'offset': LIMIT,
        'page_size': LIMIT,
        'returned': 500,
        'has_more': False,
        'auto_paginated': True,
    }, f'Unexpected second-page pagination dict: {second["pagination"]}'
    assert set(second['statuses']) == {str(i) for i in range(LIMIT + 1, LIMIT + 501)}, (
        'Second auto-page must be the ids AFTER the first page, not the first '
        'page served again'
    )


# ---------------------------------------------------------------------------
# Deterministic ordering across mixed key shapes (task 3064, review amendment)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pages_tile_a_mixed_numeric_and_non_numeric_population(paging_server):
    """Gap-free, duplicate-free tiling holds over MIXED id shapes too.

    ``_status_page``'s ordering key has two branches — numeric and
    non-numeric — and every other test in this file uses ``_make_statuses``,
    which emits purely numeric ids, so the second branch was never exercised.
    Deterministic tiling is the whole reason that sort exists (the backend's
    ``SELECT id, status FROM tasks WHERE tag = ?`` has no ``ORDER BY``), and the
    mixed population is precisely the case where an unstable or raising key
    would let two page calls interleave differently and silently drop entries.

    Not a synthetic shape: ``'3.1'`` is Taskmaster's subtask id form, and a
    negative or non-numeric id is exactly the sort of value a non-standard
    backend can hand back.
    """
    population = _set_population(
        {'10': 'pending', '2': 'done', '3.1': 'pending', 'abc': 'blocked', '-5': 'done'}
    )

    # (a) Page to exhaustion at page_size=2 and reassemble.
    merged: dict[str, str] = {}
    order: list[str] = []
    offset = 0
    for _ in range(10):
        page_result = await paging_server._tool_manager.call_tool(
            'get_statuses',
            {'project_root': '/project', 'page_size': 2, 'offset': offset},
        )
        page = page_result['statuses']
        overlap = set(page) & set(merged)
        assert overlap == set(), f'Pages must not overlap, duplicated: {overlap}'
        merged.update(page)
        order.extend(page)
        if not page_result['pagination']['has_more']:
            break
        offset += page_result['pagination']['page_size']
    else:
        pytest.fail('Mixed-population paging loop did not terminate')

    # (b) The union is the whole population — no gaps.
    assert merged == population, (
        f'Pages must tile the mixed population exactly, got {sorted(merged)} '
        f'vs {sorted(population)}'
    )

    # (c) Numeric ids come out in NUMERIC order, ahead of the non-numeric ones,
    # which are themselves deterministically ordered.  Pins the actual contract,
    # not just "some stable order": a lexicographic sort would emit 10 before 2.
    assert order == ['-5', '2', '10', '3.1', 'abc'], (
        f'Unexpected total order across the numeric/non-numeric branches: {order}'
    )


def test_status_page_tolerates_non_string_keys():
    """A non-str-keyed backend map must not raise from inside pagination.

    Direct unit test of ``_status_page`` (the MCP layer declares
    ``dict[str, Any]``, so an int-keyed map cannot be driven through the tool).
    Before the ``str(k)`` coercion this raised ``AttributeError: 'int' object
    has no attribute 'lstrip'`` from the sort key — masking the real backend
    shape problem behind a traceback from the pagination code, which is the same
    failure mode the ``isinstance`` guards at the call sites exist to avoid.
    """
    from fused_memory.server.tools import _status_page

    mixed = {3: 'done', 1: 'pending', '2': 'blocked', 'x': 'done'}

    page, total = _status_page(mixed, 0, 3)

    assert total == 4
    # Int-like keys still sort numerically alongside their str twins; the
    # non-numeric key sorts deterministically last and so falls off page 1.
    assert list(page) == [1, '2', 3], f'Unexpected order over mixed key types: {list(page)}'
    tail, _ = _status_page(mixed, 3, 3)
    assert list(tail) == ['x']


# ---------------------------------------------------------------------------
# Observability of the degraded response (task 3064, review amendment)
# ---------------------------------------------------------------------------


class _RecordingJournal:
    """Minimal WriteJournal stand-in that captures ``_log_read`` calls."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def log_write_op(self, **kwargs):
        self.calls.append(kwargs)


@pytest.fixture
def recording_journal():
    return _RecordingJournal()


@pytest.fixture
def journalled_paging_server(paging_task_interceptor, recording_journal):
    """Paging server wired with a read-log spy."""
    return create_mcp_server(
        AsyncMock(),
        task_interceptor=paging_task_interceptor,
        write_journal=recording_journal,
    )


@pytest.mark.asyncio
async def test_auto_pagination_is_visible_in_the_server_log_and_read_log(
    journalled_paging_server, recording_journal, caplog
):
    """A degraded response must be observable to an OPERATOR, not just the caller.

    This is the other half of loud degradation: the payload's ``pagination``
    marker only helps a caller that reads it, so the server must also say — in
    its own log — that it served a reduced census, and the read log must carry
    ``total`` so a paged response is distinguishable from a genuinely shrunken
    project rather than looking like one.

    Both behaviours are load-bearing design, but nothing asserted them, so a
    refactor could drop either without a single test failing.
    """
    import logging

    from fused_memory.server.tools import _STATUSES_AUTO_PAGE_LIMIT as LIMIT

    _set_population(_make_statuses(LIMIT + 500))

    with caplog.at_level(logging.WARNING, logger='fused_memory.server.tools'):
        result = await journalled_paging_server._tool_manager.call_tool(
            'get_statuses',
            {'project_root': '/project', 'auto_paginate': True},
        )
    assert result['pagination']['auto_paginated'] is True  # precondition

    # (a) A WARNING naming the degradation, carrying the structured facts an
    # operator needs to size it (not just a bare "something happened").
    degraded = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and 'auto-paginated' in r.getMessage()
    ]
    assert len(degraded) == 1, (
        f'Expected exactly one auto-pagination warning, got '
        f'{[r.getMessage() for r in caplog.records]}'
    )
    record = degraded[0]
    assert getattr(record, 'total', None) == LIMIT + 500
    assert getattr(record, 'returned', None) == LIMIT
    assert getattr(record, 'page_size', None) == LIMIT
    assert getattr(record, 'offset', None) == 0
    assert getattr(record, 'project_root', None) == '/project'

    # (b) The read log reports returned-vs-total, so a reduced response does not
    # read as a shrunken census in the journal.
    reads = [c for c in recording_journal.calls if c.get('operation') == 'get_statuses']
    assert len(reads) == 1, f'Expected one get_statuses read-log entry, got {len(reads)}'
    summary = reads[0]['result_summary']
    assert summary['count'] == LIMIT, f'Unexpected read-log count: {summary}'
    assert summary['total'] == LIMIT + 500, (
        f"Read log must carry 'total' for a paged response, got: {summary}"
    )


@pytest.mark.asyncio
async def test_complete_response_logs_no_degradation_marker(
    journalled_paging_server, recording_journal, caplog
):
    """The complementary half: a COMPLETE response must stay quiet.

    Without this, the observability assertions above would still pass against an
    implementation that warned on every call and stamped ``total`` on every read
    — which would make the degradation signal worthless.
    """
    import logging

    from fused_memory.server.tools import _STATUSES_AUTO_PAGE_LIMIT as LIMIT

    _set_population(_make_statuses(LIMIT + 500))

    with caplog.at_level(logging.WARNING, logger='fused_memory.server.tools'):
        result = await journalled_paging_server._tool_manager.call_tool(
            'get_statuses',
            {'project_root': '/project'},
        )
    assert set(result.keys()) == {'statuses'}  # precondition: complete response

    assert [r for r in caplog.records if 'auto-paginated' in r.getMessage()] == [], (
        'A complete response must not emit a degradation warning'
    )
    reads = [c for c in recording_journal.calls if c.get('operation') == 'get_statuses']
    assert len(reads) == 1
    assert 'total' not in reads[0]['result_summary'], (
        f"'total' marks a REDUCED response; a complete one must omit it, got: "
        f'{reads[0]["result_summary"]}'
    )
