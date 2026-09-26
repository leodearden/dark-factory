"""The one task snapshot unit — ``dashboard.data.task_snapshot``.

Every test here drives the unit through its PUBLIC seam: the canned
fused-memory substrate below is patched at ``dashboard.data.tasks.mcp_tool_call``
(a public name), so the real ``fetch_tasks`` / ``fetch_statuses`` / cache /
fan-out path runs underneath and no private name is reached into.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import httpx
import pytest

NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)
"""The one injected instant every test stamps against.

Resolved by the caller and threaded in, never read from a clock here: the
whole point of the envelope is that ``as_of`` names the instant a payload was
measured, and a test that cannot say what that instant is cannot check it.
"""

# ---------------------------------------------------------------------------
# CannedMCP — the fused-memory substrate, emulated faithfully enough that a
# test above it is not vacuous (task 5587 pre-1)
# ---------------------------------------------------------------------------


class CannedMCP:
    """A stand-in for :func:`dashboard.data.memory.mcp_tool_call`.

    Faithful to the two substrate contracts the unit above it depends on,
    because a paging loop tested against a non-paging fake tests nothing:

    * ``get_tasks`` applies ``statuses`` as a row filter SERVER-side, then
      slices ``page_size``/``offset`` over a list ordered by ascending ``id``
      — so reaching the high-id end requires a computed offset. No
      ``pagination`` envelope, matching ``test_active_tasks.py::_canned_mcp``:
      the unit's row read is unpaginated and the terminal window discards the
      envelope, so emitting one would fake a contract nothing here reads.
    * ``get_statuses`` slices a deterministic total order and answers with the
      five-key ``pagination`` envelope
      (``total``/``offset``/``page_size``/``returned``/``has_more``) exactly
      as ``fused_memory/server/tools.py::_pagination_meta`` builds it:
      ``page_size`` echoes the REQUESTED size verbatim, ``returned`` is the
      count actually SERVED, and ``has_more`` is ``offset + returned <
      total``. That includes serving a page SMALLER than requested, which is
      the case that tells a loop advancing by ``returned`` apart from one
      advancing by the size it asked for.

    Args:
        rows: Raw MCP ``get_tasks`` rows (string ids), as
            :func:`_raw_row` builds them.
        status_map: ``{int id: status}`` — the population ``get_statuses``
            serves.
        status_page_size: The server's own page cap. ``None`` (default) is the
            backward-compatible mode: the whole map in one response with NO
            ``pagination`` key, which the contract defines as COMPLETE.
        short_page_by: Serve this many FEWER statuses per page than the
            smaller of the requested and the server cap. The served count is
            reported in ``pagination['returned']``, like any other page. Zero
            disables it.

    Mutable after construction so one instance can change behaviour between a
    test's two acquisitions. :attr:`fail_when` is a predicate over the
    recorded call — ``{'tool', 'args', 'kwargs'}`` — and every call it accepts
    raises ``httpx.ReadTimeout``. A predicate rather than a set of tool names
    because the two failures worth injecting are not the same shape: "this
    half is unreachable now" keys on the tool, while "the walk breaks at its
    SECOND page" keys on the offset, and only the latter distinguishes an
    offline marker from a silently short map.
    """

    def __init__(
        self,
        rows=(),
        status_map=None,
        *,
        status_page_size: int | None = None,
        short_page_by: int = 0,
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.status_map = dict(status_map or {})
        self.status_page_size = status_page_size
        self.short_page_by = short_page_by
        self.fail_when: Callable[[dict], bool] = lambda call: False
        self.calls: list[dict] = []

    def calls_to(self, tool: str) -> list[dict]:
        """Every recorded call to *tool*, in order."""
        return [call for call in self.calls if call['tool'] == tool]

    async def __call__(self, client, url, tool, args, **kwargs):
        # ``kwargs`` is recorded too: the per-request budget rides as the
        # ``timeout=`` keyword and never inside ``args``, so it is only
        # assertable at the wire if it is kept.
        call = {'tool': tool, 'args': dict(args), 'kwargs': dict(kwargs)}
        self.calls.append(call)
        if self.fail_when(call):
            raise httpx.ReadTimeout(f'canned {tool} read timeout')
        if tool == 'get_statuses':
            return self._statuses(args)
        if tool == 'get_tasks':
            return self._tasks(args)
        raise AssertionError(f'unexpected tool {tool!r}')

    def _statuses(self, args: dict) -> dict:
        ordered = sorted(self.status_map.items(), key=lambda item: int(item[0]))
        if self.status_page_size is None:
            return {'statuses': {str(tid): status for tid, status in ordered}}

        requested = args.get('page_size') or self.status_page_size
        served = max(1, min(requested, self.status_page_size) - self.short_page_by)
        offset = args.get('offset') or 0
        window = ordered[offset:offset + served]
        return {
            'statuses': {str(tid): status for tid, status in window},
            'pagination': {
                'total': len(ordered),
                'offset': offset,
                'page_size': requested,
                'returned': len(window),
                'has_more': offset + len(window) < len(ordered),
            },
        }

    def _tasks(self, args: dict) -> dict:
        statuses = args.get('statuses')
        selected = [
            row for row in self.rows
            if statuses is None or row.get('status') in statuses
        ]
        selected.sort(key=lambda row: int(row['id']))  # ORDER BY id ASC
        page_size = args.get('page_size')
        if page_size is not None:
            start = args.get('offset') or 0
            selected = selected[start:start + page_size]
        return {'tasks': selected}


def _raw_row(task_id, status, **overrides) -> dict:
    """One raw MCP ``get_tasks`` row, distinguishable by id."""
    row = {
        'id': str(task_id),
        'title': f'task {task_id}',
        'status': status,
        'dependencies': [],
        'metadata': {},
        'updatedAt': '2026-01-01T00:00:00+00:00',
    }
    row.update(overrides)
    return row


class TestCannedMCP:
    """The emulation's own fidelity, since every test above it rests on it."""

    async def test_paged_statuses_carry_the_five_key_envelope(self):
        canned = CannedMCP(status_map={i: 'done' for i in range(1, 6)},
                           status_page_size=2)
        page = await canned(None, 'u', 'get_statuses',
                            {'project_root': '/p', 'page_size': 2, 'offset': 0})

        assert page['pagination'] == {
            'total': 5, 'offset': 0, 'page_size': 2, 'returned': 2, 'has_more': True,
        }
        assert page['statuses'] == {'1': 'done', '2': 'done'}

    async def test_returned_reports_what_was_served_and_page_size_what_was_asked(self):
        """The case that separates advancing by SERVED from advancing by requested.

        ``returned`` carries the served count and ``page_size`` echoes the
        request, because that is what ``_pagination_meta`` emits. A fake that
        put the served count in ``page_size`` would model the load-bearing
        field backwards, and a walker keyed on it would pass here and skip
        entries against the real server.
        """
        canned = CannedMCP(status_map={i: 'done' for i in range(1, 6)},
                           status_page_size=4, short_page_by=1)
        page = await canned(None, 'u', 'get_statuses',
                            {'project_root': '/p', 'page_size': 4, 'offset': 0})

        assert page['pagination']['returned'] == 3, 'served'
        assert page['pagination']['page_size'] == 4, 'requested, echoed verbatim'
        assert len(page['statuses']) == 3
        assert page['pagination']['has_more'] is True

    async def test_no_page_size_configured_means_no_pagination_key(self):
        """The absence of the key is the substrate's spelling of COMPLETE."""
        canned = CannedMCP(status_map={1: 'done', 2: 'pending'})
        page = await canned(None, 'u', 'get_statuses', {'project_root': '/p'})

        assert 'pagination' not in page
        assert page['statuses'] == {'1': 'done', '2': 'pending'}

    async def test_an_injected_failure_raises_read_timeout(self):
        canned = CannedMCP(status_map={1: 'done'})
        canned.fail_when = lambda call: call['tool'] == 'get_statuses'

        with pytest.raises(httpx.ReadTimeout):
            await canned(None, 'u', 'get_statuses', {'project_root': '/p'})

    async def test_get_tasks_filters_then_slices_by_ascending_id(self):
        canned = CannedMCP(rows=[_raw_row(i, 'done' if i % 2 else 'pending')
                                 for i in range(1, 8)])
        page = await canned(None, 'u', 'get_tasks',
                            {'project_root': '/p', 'statuses': ['done'],
                             'page_size': 2, 'offset': 1})

        assert [row['id'] for row in page['tasks']] == ['3', '5']


# ---------------------------------------------------------------------------
# TestAcquireSnapshotHappyPath — one unit, two halves, one TTL (step-3)
# ---------------------------------------------------------------------------


ALL_NINE = (
    (1, 'in-progress'), (2, 'blocked'), (3, 'merge-deferred'), (4, 'review'),
    (5, 'infra-hold'), (6, 'pending'), (7, 'deferred'), (8, 'done'),
    (9, 'cancelled'),
)
"""One id per ``TaskStatus`` member, so every census bucket is exercised."""


def _tree(pairs=ALL_NINE, overrides=None):
    """A ``(rows, status_map)`` pair agreeing with each other by construction.

    *overrides* maps a task id to extra raw-row fields, for the claimant
    columns the strand split reads.
    """
    extra = overrides or {}
    rows = [_raw_row(tid, status, **extra.get(tid, {})) for tid, status in pairs]
    return rows, {tid: status for tid, status in pairs}


@pytest.fixture()
def project_root(tmp_path):
    root = tmp_path / 'dark-factory'
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(autouse=True)
def _isolate_caches():
    """Neither the unit cache nor the fetch_tasks cache may cross a test."""
    import dashboard.data.task_snapshot as snapshot_mod
    import dashboard.data.tasks as tasks_mod

    snapshot_mod._snapshot_cache_clear()
    tasks_mod._fetch_tasks_cache_clear()
    yield
    snapshot_mod._snapshot_cache_clear()
    tasks_mod._fetch_tasks_cache_clear()


class TestAcquireSnapshotHappyPath:
    """The unit measures both halves, stamps them, and tells you how they relate.

    Driven end to end through ``mcp_tool_call``, so the real ``fetch_tasks`` /
    ``fetch_statuses`` / fan-out path runs underneath every assertion.
    """

    @staticmethod
    async def _acquire(canned, client, config, root, *, now):
        from dashboard.data.task_snapshot import acquire_snapshot

        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            return await acquire_snapshot(client, config, root, now=now)

    async def test_the_census_is_a_fresh_datum_stamped_with_the_injected_now(
        self, project_root, dashboard_config, dummy_client
    ):
        """(a) A ``Datum[TaskCensus]``, tz-aware, at the caller's own instant."""
        from dashboard.data.census import TaskView, build_census
        from dashboard.data.datum import DatumState

        rows, status_map = _tree()
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        snapshot = await self._acquire(canned, dummy_client, dashboard_config, project_root,
                                       now=NOW)

        census = snapshot.census
        assert census.state is DatumState.FRESH
        assert census.as_of == NOW and NOW.utcoffset() is not None
        tally = census.value
        assert tally == build_census(status_map)
        assert tally is not None
        assert len(tally.counts) == 9, 'every bucket present, none absent'
        assert sum(tally.counts.values()) == tally.total
        assert sum(tally.views.values()) == tally.total
        assert tally.sub_views[TaskView.RUNNING] <= tally.views[TaskView.IN_FLIGHT]
        assert tally.views[TaskView.IN_FLIGHT] == 5, (
            'review and infra-hold belong to in_flight, which the retired '
            f'five-member _ACTIVE_STATUSES omitted: {tally.views}'
        )

    async def test_the_row_read_asks_for_every_active_status(
        self, project_root, dashboard_config, dummy_client
    ):
        """(b) All seven of ``shared.task_statuses.ACTIVE``, review and infra-hold included."""
        from shared.task_statuses import ACTIVE

        from dashboard.data.datum import DatumState

        rows, status_map = _tree()
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        snapshot = await self._acquire(canned, dummy_client, dashboard_config, project_root,
                                       now=NOW)

        requested = [call['args'].get('statuses')
                     for call in canned.calls_to('get_tasks')]
        assert requested == [sorted(ACTIVE)], (
            f'the row read must ask for exactly the active vocabulary, got {requested}'
        )
        assert len(sorted(ACTIVE)) == 7
        assert snapshot.rows.state is DatumState.FRESH
        assert snapshot.rows.value is not None
        assert {row['id'] for row in snapshot.rows.value} == {1, 2, 3, 4, 5, 6, 7}

    async def test_the_live_stranded_split_counts_the_rows_not_the_census(
        self, project_root, dashboard_config, dummy_client
    ):
        """(c) Partitioned by ``tasks.task_is_stranded``, over the ROWS.

        The census counts a POPULATION; the split needs the claimant columns,
        which only the rows carry. Deriving it from the census would have to
        invent them.
        """
        from dashboard.data.census import TaskView

        beating = NOW.isoformat()
        stale = (NOW - timedelta(hours=2)).isoformat()
        rows, status_map = _tree(
            pairs=((1, 'in-progress'), (2, 'in-progress'), (6, 'pending')),
            overrides={
                1: {'claimant_run_id': 'run-1/sess-1/pid=42', 'heartbeat_at': beating},
                2: {'claimant_run_id': 'run-2/sess-2/pid=43', 'heartbeat_at': stale},
            },
        )
        # A THIRD in-progress id the map knows about and the rows do not, so
        # the split cannot pass by coincidentally agreeing with the census.
        status_map[99] = 'in-progress'
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        snapshot = await self._acquire(canned, dummy_client, dashboard_config,
                                       project_root, now=NOW)

        assert snapshot.in_progress_live == 1, 'the beating heartbeat'
        assert snapshot.in_progress_stranded == 1, 'the two-hour-old one'
        assert snapshot.rows.value is not None
        assert len([r for r in snapshot.rows.value
                    if r['status'] == 'in-progress']) == 2
        assert snapshot.census.value is not None
        assert snapshot.census.value.sub_views[TaskView.RUNNING] == 3, (
            'the census counts a population the rows do not carry, which is '
            'why the split may not be derived from it'
        )

    async def test_skew_is_the_measured_gap_between_the_two_halves(
        self, project_root, dashboard_config, dummy_client
    ):
        """(d) A non-negative int, or None when a half was never measured."""
        rows, status_map = _tree()
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        snapshot = await self._acquire(canned, dummy_client, dashboard_config, project_root,
                                       now=NOW)

        census_at, rows_at = snapshot.census.as_of, snapshot.rows.as_of
        assert census_at is not None and rows_at is not None
        assert snapshot.skew_seconds == int(
            abs((census_at - rows_at).total_seconds())
        ) == 0, 'both halves share the one injected instant'

        canned.fail_when = lambda call: call['tool'] == 'get_statuses'
        import dashboard.data.task_snapshot as snapshot_mod
        snapshot_mod._snapshot_cache_clear()
        degraded = await self._acquire(canned, dummy_client, dashboard_config, project_root,
                                       now=NOW)

        assert degraded.census.as_of is None
        assert degraded.skew_seconds is None, (
            'a gap between one measured instant and no instant is not zero'
        )

    async def test_both_halves_validate_against_a_served_at_inside_the_bound(
        self, project_root, dashboard_config, dummy_client
    ):
        """(e) Fresh up to the declared bound, and a contract error past it.

        The bound is twice the unit's own refresh TTL, because a unit served
        from that cache at age 14.9 s is legitimately fresh by its producer's
        contract — a tighter bound would make the access layer raise on its
        own correct output.
        """
        from dashboard.data.datum import DatumContractError, DatumInvariant, validate_datum
        from dashboard.data.task_snapshot import FRESHNESS_BOUND_SECONDS, SNAPSHOT_TTL_SECONDS

        assert FRESHNESS_BOUND_SECONDS == 2 * SNAPSHOT_TTL_SECONDS

        rows, status_map = _tree()
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        snapshot = await self._acquire(canned, dummy_client, dashboard_config, project_root,
                                       now=NOW)

        at_the_bound = NOW + timedelta(seconds=FRESHNESS_BOUND_SECONDS)
        validate_datum(snapshot.census, at_the_bound)
        validate_datum(snapshot.rows, at_the_bound)

        with pytest.raises(DatumContractError) as raised:
            validate_datum(snapshot.census, at_the_bound + timedelta(seconds=1))
        assert raised.value.invariant is DatumInvariant.FRESHNESS_BOUND

    async def test_a_unit_served_past_its_bound_is_stale_not_a_contract_break(
        self, project_root, dashboard_config, dummy_client
    ):
        """``as_served`` ages a cached unit at the instant a payload serves it.

        A unit shared through the cache can be served later than its bound: it
        sits in the cache for up to the TTL and then waits out the reading
        render's own fan-out. That is a stale value, the same one at the same
        ``as_of``, and it must reach the validator saying so.
        """
        from dashboard.data.datum import DatumState, validate_datum
        from dashboard.data.task_snapshot import (
            FRESHNESS_BOUND_SECONDS,
            SnapshotHealth,
            as_served,
            classify,
        )

        rows, status_map = _tree()
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        snapshot = await self._acquire(canned, dummy_client, dashboard_config, project_root,
                                       now=NOW)

        within = NOW + timedelta(seconds=FRESHNESS_BOUND_SECONDS)
        assert as_served(snapshot, within) == snapshot

        late = within + timedelta(seconds=5)
        served = as_served(snapshot, late)
        for half in (served.census, served.rows):
            validate_datum(half, late)
            assert half.state is DatumState.STALE
            assert half.as_of == NOW
            assert f'{FRESHNESS_BOUND_SECONDS}s freshness bound' in (half.reason or '')
        assert served.census.value == snapshot.census.value
        assert served.rows.value == snapshot.rows.value
        assert classify(served) is SnapshotHealth.COUNT_UNKNOWN

    async def test_to_wire_emits_the_contract_keys_and_not_the_raw_map(
        self, project_root, dashboard_config, dummy_client
    ):
        """(f) The map is the unit's raw material, not part of its wire shape.

        ``_resolve_deps`` needs it in-process as its only bounded fallback for
        a dependency outside the fetched rows, so it lives on the record and
        stops at the wire.
        """
        rows, status_map = _tree()
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        snapshot = await self._acquire(canned, dummy_client, dashboard_config, project_root,
                                       now=NOW)

        wire = snapshot.to_wire()
        assert set(wire) == {
            'census', 'rows', 'in_progress_live', 'in_progress_stranded',
            'skew_seconds',
        }
        assert snapshot.status_map == status_map, 'still reachable in-process'
        assert 'status_map' not in wire

        census_wire = wire['census']
        assert isinstance(census_wire, dict)
        rendered = census_wire['value']
        assert isinstance(rendered, dict)
        assert set(rendered) == {'counts', 'total', 'views', 'sub_views'}
        assert rendered['counts']['infra-hold'] == 1, (
            f'plain-string keys, not enum members: {rendered["counts"]}'
        )
        rows_wire = wire['rows']
        assert isinstance(rows_wire, dict)
        assert isinstance(rows_wire['value'], list)

    async def test_the_unit_holds_the_raw_rows_the_shaper_reads(
        self, project_root, dashboard_config, dummy_client
    ):
        """(f2) The unit's ``rows.value`` is raw material, not the wire's ``TaskRow`` list.

        The unit is what the 15 s cache holds, and each render shapes its rows
        again, so it keeps the raw integer ids, ``dependencies`` and
        ``metadata`` that shaping reads. The wire gets the shaped rows because
        ``collect_tasks_with_counts`` swaps them into the copy it returns. It
        never shapes the cached unit itself.
        """
        rows, status_map = _tree()
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        snapshot = await self._acquire(canned, dummy_client, dashboard_config, project_root,
                                       now=NOW)

        assert snapshot.rows.value is not None
        for row in snapshot.rows.value:
            assert isinstance(row['id'], int), row
            assert {'dependencies', 'metadata'} <= set(row), sorted(row)

    async def test_one_unit_one_ttl_and_an_uncached_row_read(
        self, project_root, dashboard_config, dummy_client, monkeypatch
    ):
        """(g) The unit owns the only TTL on this path.

        Inside it, nothing is read at all. Past it, BOTH halves are — including
        the rows, whose own 20 s cache would otherwise serve a value older than
        the ``as_of`` this unit is about to stamp on it.
        """
        import dashboard.data.task_snapshot as snapshot_mod

        rows, status_map = _tree()
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)

        first = await self._acquire(canned, dummy_client, dashboard_config, project_root, now=NOW)
        calls_after_first = len(canned.calls)
        second = await self._acquire(canned, dummy_client, dashboard_config, project_root, now=NOW)

        assert second is first, 'the second acquisition must be the same unit'
        assert len(canned.calls) == calls_after_first, (
            f'a warm unit issues no MCP call, got {canned.calls[calls_after_first:]}'
        )

        # Past the unit TTL. The fetch_tasks cache is NOT cleared and its 20 s
        # window is still open in real time, so a cached row read would serve
        # the first acquisition's rows here.
        monkeypatch.setattr(snapshot_mod, 'SNAPSHOT_TTL_SECONDS', 0.0)
        later = NOW + timedelta(seconds=16)
        third = await self._acquire(canned, dummy_client, dashboard_config, project_root, now=later)

        assert third is not first
        assert third.rows.as_of == later and third.census.as_of == later
        assert len(canned.calls_to('get_tasks')) == 2, (
            'the row read must be uncached, so the unit never stamps an as_of '
            'newer than the rows it stamps'
        )
        assert len(canned.calls_to('get_statuses')) == 2

    async def test_a_vocabulary_drift_degrades_the_census_and_spares_the_rows(
        self, project_root, dashboard_config, dummy_client
    ):
        """(h) The census carries the error verbatim; the rows stay measured.

        ``build_census`` refuses an off-vocabulary value rather than dropping
        the row, and this layer is where that refusal becomes visible instead
        of fatal.
        """
        from dashboard.data.census import build_census
        from dashboard.data.datum import DatumState

        rows, status_map = _tree()
        status_map[6] = 'not-a-status'
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        snapshot = await self._acquire(canned, dummy_client, dashboard_config, project_root,
                                       now=NOW)

        assert snapshot.census.state is not DatumState.FRESH
        with pytest.raises(Exception) as raised:
            build_census(status_map)
        assert snapshot.census.reason == str(raised.value), (
            'the producer\'s message must cross verbatim, not be reworded: '
            f'{snapshot.census.reason!r}'
        )
        assert snapshot.rows.state is DatumState.FRESH


# ---------------------------------------------------------------------------
# TestSnapshotDegradation — last-good, stale, unknown (step-5, sketch #13)
# ---------------------------------------------------------------------------


class TestSnapshotDegradation:
    """A failed refresh serves the PREVIOUS value, aged and explained.

    The fabricated zero is the failure this envelope exists to remove, so the
    three outcomes are kept distinct at every step: a value measured now
    (``fresh``), a value measured earlier and known to be old (``stale``), and
    no value at all (``unknown``). None of them is ever a zero.
    """

    @staticmethod
    async def _acquire(canned, client, config, root, *, now):
        from dashboard.data.task_snapshot import acquire_snapshot

        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            return await acquire_snapshot(client, config, root, now=now)

    @staticmethod
    def _expire_unit(monkeypatch):
        """Retire the cached unit without touching the last-good store."""
        import dashboard.data.task_snapshot as snapshot_mod

        monkeypatch.setattr(snapshot_mod, 'SNAPSHOT_TTL_SECONDS', 0.0)

    async def test_a_cold_failure_is_unknown_and_never_a_zero_census(
        self, project_root, dashboard_config, dummy_client
    ):
        """(a) Nothing was ever measured, so nothing is reported as measured."""
        from dashboard.data.datum import DatumState

        rows, status_map = _tree()
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        canned.fail_when = lambda call: True

        snapshot = await self._acquire(canned, dummy_client, dashboard_config,
                                       project_root, now=NOW)

        for half in (snapshot.census, snapshot.rows):
            assert half.state is DatumState.UNKNOWN
            assert half.value is None and half.as_of is None
            assert (half.reason or '').strip(), 'an unknown datum must say why'
        assert snapshot.skew_seconds is None
        assert snapshot.in_progress_live is None
        assert snapshot.in_progress_stranded is None

    async def test_a_failed_map_serves_the_previous_census_as_stale(
        self, project_root, dashboard_config, dummy_client, monkeypatch
    ):
        """(b) The badge ages, the rows do not, and the reason is verbatim."""
        from dashboard.data.datum import DatumState

        rows, status_map = _tree()
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        good = await self._acquire(canned, dummy_client, dashboard_config,
                                   project_root, now=NOW)

        self._expire_unit(monkeypatch)
        canned.fail_when = lambda call: call['tool'] == 'get_statuses'
        later = NOW + timedelta(seconds=20)
        degraded = await self._acquire(canned, dummy_client, dashboard_config,
                                       project_root, now=later)

        assert degraded.census.state is DatumState.STALE
        assert degraded.census.value == good.census.value, 'the PREVIOUS census'
        assert degraded.census.as_of == NOW, (
            'a stale datum keeps the instant it was measured, so its badge ages'
        )
        assert 'ReadTimeout' in (degraded.census.reason or ''), (
            'the fan-out\'s own marker text must cross verbatim, not a reworded '
            f'copy: {degraded.census.reason!r}'
        )
        assert degraded.rows.state is DatumState.FRESH
        assert degraded.rows.as_of == later
        assert degraded.skew_seconds == 20, 'the real gap, now that there is one'

    async def test_recovery_returns_the_census_to_fresh(
        self, project_root, dashboard_config, dummy_client, monkeypatch
    ):
        """(c) A stale badge is not a latch."""
        from dashboard.data.datum import DatumState

        rows, status_map = _tree()
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        await self._acquire(canned, dummy_client, dashboard_config,
                            project_root, now=NOW)

        self._expire_unit(monkeypatch)
        canned.fail_when = lambda call: call['tool'] == 'get_statuses'
        await self._acquire(canned, dummy_client, dashboard_config,
                            project_root, now=NOW + timedelta(seconds=20))

        canned.fail_when = lambda call: False
        recovered_at = NOW + timedelta(seconds=40)
        recovered = await self._acquire(canned, dummy_client, dashboard_config,
                                        project_root, now=recovered_at)

        assert recovered.census.state is DatumState.FRESH
        assert recovered.census.as_of == recovered_at
        assert recovered.census.reason is None
        assert recovered.skew_seconds == 0

    async def test_failed_rows_keep_their_split_judged_at_their_own_instant(
        self, project_root, dashboard_config, dummy_client, monkeypatch
    ):
        """(d) The mirror case — and the split ages with the rows, not the clock.

        A heartbeat beating at the instant the rows were measured was LIVE at
        that instant. Re-judging it against a later clock would manufacture a
        strand that never happened; the rows' own ``as_of`` is what discloses
        how old the claim is.
        """
        from dashboard.data.datum import DatumState

        rows, status_map = _tree(
            pairs=((1, 'in-progress'), (6, 'pending')),
            overrides={1: {'claimant_run_id': 'run-1/sess-1/pid=42',
                           'heartbeat_at': NOW.isoformat()}},
        )
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        good = await self._acquire(canned, dummy_client, dashboard_config,
                                   project_root, now=NOW)
        assert (good.in_progress_live, good.in_progress_stranded) == (1, 0)

        self._expire_unit(monkeypatch)
        canned.fail_when = lambda call: call['tool'] == 'get_tasks'
        much_later = NOW + timedelta(hours=1)
        degraded = await self._acquire(canned, dummy_client, dashboard_config,
                                       project_root, now=much_later)

        assert degraded.rows.state is DatumState.STALE
        assert degraded.rows.value == good.rows.value
        assert degraded.rows.as_of == NOW
        assert degraded.census.state is DatumState.FRESH
        assert (degraded.in_progress_live, degraded.in_progress_stranded) == (1, 0), (
            'the split reports what the stale rows measured, never a zero and '
            'never a strand invented by the passage of time'
        )

    async def test_the_last_good_store_does_not_rescue_another_root(
        self, project_root, dashboard_config, dummy_client, tmp_path
    ):
        """(e) Both halves fail with no prior value FOR THIS ROOT."""
        from dashboard.config import DashboardConfig
        from dashboard.data.datum import DatumState

        rows, status_map = _tree()
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        await self._acquire(canned, dummy_client, dashboard_config,
                            project_root, now=NOW)

        other = tmp_path / 'other-project'
        other.mkdir(parents=True, exist_ok=True)
        canned.fail_when = lambda call: call['args'].get('project_root') == str(other)
        starved = await self._acquire(
            canned, dummy_client, DashboardConfig(project_root=other), other, now=NOW,
        )

        for half in (starved.census, starved.rows):
            assert half.state is DatumState.UNKNOWN, (
                "one root's last good must never be served for another"
            )
            assert (half.reason or '').strip()

    async def test_a_last_good_past_retention_degrades_to_unknown(
        self, project_root, dashboard_config, dummy_client, monkeypatch
    ):
        """(f) Staleness is bounded: past the bound it is no longer evidence."""
        from dashboard.data.datum import DatumState
        from dashboard.data.task_snapshot import _RETENTION_BOUND_SECONDS

        rows, status_map = _tree()
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        await self._acquire(canned, dummy_client, dashboard_config,
                            project_root, now=NOW)

        self._expire_unit(monkeypatch)
        canned.fail_when = lambda call: call['tool'] == 'get_statuses'
        past_retention = NOW + timedelta(seconds=_RETENTION_BOUND_SECONDS + 1)
        expired = await self._acquire(canned, dummy_client, dashboard_config,
                                      project_root, now=past_retention)

        assert expired.census.state is DatumState.UNKNOWN
        assert expired.census.value is None and expired.census.as_of is None
        assert str(_RETENTION_BOUND_SECONDS) in (expired.census.reason or ''), (
            'the reason must name the bound that disqualified the last good: '
            f'{expired.census.reason!r}'
        )

    async def test_a_budget_expiry_and_a_read_failure_are_different_kinds(
        self, project_root, dashboard_config, dummy_client, monkeypatch
    ):
        """(g) Structured, so the banner routing never parses a message.

        Offline means a read demonstrably failed; degraded means this process
        ran out of budget and the server may be perfectly healthy. Rendering
        them identically is what made the 2026-07-30 event get misdiagnosed.
        """
        import dashboard.data.task_snapshot as snapshot_mod
        from dashboard.data.datum import DatumState
        from dashboard.data.task_snapshot import SnapshotFailure

        rows, status_map = _tree()
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        canned.fail_when = lambda call: call['tool'] == 'get_tasks'
        unreachable = await self._acquire(canned, dummy_client, dashboard_config,
                                          project_root, now=NOW)

        assert unreachable.failure is SnapshotFailure.UNREACHABLE
        assert unreachable.rows.state is not DatumState.FRESH

        snapshot_mod._snapshot_cache_clear()
        monkeypatch.setattr(snapshot_mod, 'PER_CALL_TIMEOUT', 0.01)
        slow = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)

        async def _slow_rows(client, url, tool, args, **kwargs):
            if tool == 'get_tasks':
                await asyncio.sleep(0.5)
            return await slow(client, url, tool, args, **kwargs)

        starved = await self._acquire(_slow_rows, dummy_client, dashboard_config,
                                      project_root, now=NOW)

        assert starved.failure is SnapshotFailure.BUDGET
        assert starved.rows.state is not DatumState.FRESH
        assert starved.failure is not unreachable.failure

    async def test_a_failed_unit_is_itself_held_for_the_ttl(
        self, project_root, dashboard_config, dummy_client
    ):
        """(h) Retry suppression, replacing the negative cache this path lost.

        A wedged root costs one attempt per TTL rather than one per 3 s poll.
        The price is up to a TTL of recovery latency, which is deliberate.
        """
        rows, status_map = _tree()
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        canned.fail_when = lambda call: True

        first = await self._acquire(canned, dummy_client, dashboard_config,
                                    project_root, now=NOW)
        attempts = len(canned.calls)
        second = await self._acquire(canned, dummy_client, dashboard_config,
                                     project_root, now=NOW + timedelta(seconds=1))

        assert second is first
        assert len(canned.calls) == attempts, (
            f'a failed unit must be held for the TTL, got '
            f'{canned.calls[attempts:]}'
        )

    async def test_an_unmeasured_unit_ages_its_census_but_holds_no_rows(
        self, project_root, dashboard_config, dummy_client
    ):
        """(i) A root the caller never reached keeps its last census, aged, and no rows.

        The census is a count, so the last good one is still evidence about
        this root. The rows are different. The caller pairs every unit with the
        rows IT shaped this render, and it shaped none for a root it never
        reached. So a last-good row list here can only reach the wire unshaped,
        or be swapped for an empty list that claims a measured zero at the last
        good's instant.
        """
        from dashboard.data.datum import DatumState
        from dashboard.data.task_snapshot import SnapshotFailure, unmeasured_snapshot

        rows, status_map = _tree()
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        good = await self._acquire(canned, dummy_client, dashboard_config,
                                   project_root, now=NOW)
        assert good.rows.value, 'the root must have a last good row list to refuse'

        reason = 'exceeded its share of the Tasks budget'
        unit = unmeasured_snapshot(
            project_root, now=NOW + timedelta(seconds=20), reason=reason,
            failure=SnapshotFailure.BUDGET,
        )

        assert unit.census.state is DatumState.STALE
        assert unit.census.value == good.census.value
        assert unit.census.as_of == NOW
        assert unit.rows.state is DatumState.UNKNOWN
        assert unit.rows.value is None and unit.rows.as_of is None
        assert unit.rows.reason == reason
        assert (unit.in_progress_live, unit.in_progress_stranded) == (None, None)
        assert unit.skew_seconds is None
        assert unit.failure is SnapshotFailure.BUDGET

    async def test_a_last_good_newer_than_this_render_is_served_stale(
        self, project_root, dashboard_config, dummy_client
    ):
        """(j) A sibling render refreshed this root at a LATER instant than this one's.

        The unit cache and the last-good store are shared across renders. A
        render whose share of the budget ran out while a render with a later
        instant refreshed the root finds a last good stamped after its own
        ``now``. That value is seconds old, so it is served stale at its own
        instant with the caller's reason verbatim. It used to be discarded as
        "past the retention bound", with a negative age in the reason.
        """
        from dashboard.data.datum import DatumState
        from dashboard.data.task_snapshot import (
            SnapshotFailure,
            SnapshotHealth,
            classify,
            unmeasured_snapshot,
        )

        rows, status_map = _tree()
        canned = CannedMCP(rows=rows, status_map=status_map, status_page_size=2000)
        sibling_at = NOW + timedelta(seconds=5)
        good = await self._acquire(canned, dummy_client, dashboard_config,
                                   project_root, now=sibling_at)

        reason = 'exceeded its share of the Tasks budget'
        unit = unmeasured_snapshot(
            project_root, now=NOW, reason=reason, failure=SnapshotFailure.BUDGET,
        )

        assert unit.census.state is DatumState.STALE, unit.census
        assert unit.census.value == good.census.value
        assert unit.census.as_of == sibling_at
        assert unit.census.reason == reason
        assert classify(unit) is SnapshotHealth.DEGRADED
