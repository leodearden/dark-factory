"""Integration tests for GraphitiBackend.drop_vector_indices() — requires a running FalkorDB.

Context: Task 3769.

``drop_vector_indices()`` was a PERMANENT NO-OP (its docstring records the three
defects), and the whole defect class here is "unit mocks pinned a fiction".  One
of the three — the old-style ``DROP INDEX`` verb, which FalkorDB applies to RANGE
indices only — is one NO unit test could have caught, because a mock accepts any
statement string.  That is what this module exists for.

So this module locks the fix in END TO END against live FalkorDB: seed a graph
carrying a NODE vector index, a RELATIONSHIP vector index and a RANGE index,
call ``drop_vector_indices()``, and assert both vector indices are really gone
while the RANGE index SURVIVES.

Skip automatically when FalkorDB is not reachable.

HAZARD (task description; open escalation esc-3375-1): the backend here is built
the way ``TestBackendListIndicesLive`` builds it — construct ``GraphitiBackend``
and inject a ``_MultiTenantFalkorDriver`` directly, NEVER calling
``initialize()`` — because ``FalkorDriver.__init__`` fire-and-forgets
``build_indices_and_constraints()``.  Every graph named here is a throwaway from
``unique_graph_name``; no real project graph is ever touched.  The absence of
indices on real graphs is protected evidence for esc-3375-1.

HAZARD (task 4748): the drop side has its own asynchrony, alongside the
already-documented index BUILD asynchrony (see ``live_vector_graph``'s
MANDATORY task-3377 barrier below).  ``DROP VECTOR INDEX`` against a label
whose merged index carries other surviving fields is NOT an in-place catalog
mutation — FalkorDB builds a REPLACEMENT index for those fields, and any
catalog read taken before that build finishes can still see the pre-drop row,
including the just-dropped VECTOR property.  Every post-drop
``list_indices()`` / ``drop_vector_indices()`` call in THIS MODULE is
therefore barriered by ``await_index_operational`` — these are TEST-only
barriers, load-bearing rather than defensive, and they do NOT mean production
is exposed to the same race:
``fused_memory/backends/graphiti_client.py::GraphitiBackend.drop_vector_indices``
issues its own ``list_indices()`` read exactly ONCE, BEFORE any drop, so a
single production call can never observe its own rebuild window.  Only a
rapid second call — what
``test_is_idempotent_and_reports_zero_on_a_second_run`` below exercises — or
a caller that re-reads the catalog after a drop, can land in it.
"""

from __future__ import annotations

import contextlib
import sys
import time

import pytest
import pytest_asyncio
from _fm_helpers import (
    FALKOR_HOST,
    FALKOR_PORT,
    IndexHeaderError,
    await_index_operational,
    falkor_skipif,
    poll_until,
    retry_until_observed,
    unique_graph_name,
)
from falkordb.asyncio import FalkorDB

from fused_memory.backends.graphiti_client import GraphitiBackend, _MultiTenantFalkorDriver

TEST_GRAPH: str = unique_graph_name('3769_drop_vector_indices')

pytestmark = [
    falkor_skipif(),
    # 30, not 15 (task 4748): pytest-timeout runs with func_only=False here, so
    # this budget covers fixture setup too, and each test below can now hold TWO
    # 10s barrier budgets -- live_vector_graph's build-side barrier (task 3377)
    # plus the post-drop barrier added by task 4748 -- so 15 would make that
    # pathological worst case (10s + 10s) a CERTAIN timeout instead of a rare
    # flake. fused-memory/pyproject.toml sets timeout_method='thread', whose
    # handler ends in os._exit(1) and kills the whole xdist worker, which is why
    # this headroom is not optional. 30 stays well under the pyproject default
    # of 60. Expected real cost of the added barrier is ~0.2ms on this module's
    # 1-node fixture graph (one CALL db.indexes() round-trip) -- this is
    # headroom, not spend.
    pytest.mark.timeout(30),
    pytest.mark.integration,
]


def _types_contain(raw, kind: str) -> bool:
    """True if a raw db.indexes() per-property type value (str or list) contains *kind*.

    FalkorDB reports a property's index type(s) as a bare string for a single
    type or a list for merged types (e.g. ``['RANGE', 'VECTOR']``). Every
    VECTOR/RANGE membership check in this module normalizes that the same
    way; this is the one place to change if that normalization ever needs to
    differ, rather than a fourth hand-rolled copy.
    """
    return kind in ([raw] if isinstance(raw, str) else raw)


def _vector_properties(records) -> set[tuple[str, str]]:
    """Every (label, property) in *records* whose type list contains 'VECTOR'.

    Deliberately hand-rolled rather than reusing
    ``falkor_indices.vector_index_properties``: the point of this module is to
    check the live index STATE independently of the helper under test, so a bug
    in that helper cannot make its own verification pass.
    """
    found: set[tuple[str, str]] = set()
    for record in records:
        types = record.get('type') or {}
        for prop, raw in types.items():
            if _types_contain(raw, 'VECTOR'):
                found.add((record['label'], prop))
    return found


@pytest_asyncio.fixture
async def live_vector_graph():
    """Seed a throwaway graph with NODE + RELATIONSHIP vector indices and a RANGE index."""
    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    # Best-effort delete any stale graph from a prior run.
    with contextlib.suppress(Exception):
        stale = client.select_graph(TEST_GRAPH)
        await stale.delete()

    graph = client.select_graph(TEST_GRAPH)
    await graph.query(
        'CREATE (a:Entity {name: $n, name_embedding: vecf32([1.0, 2.0, 3.0, 4.0])}) '
        '-[:RELATES_TO {fact_embedding: vecf32([1.0, 2.0, 3.0, 4.0])}]-> (a)',
        {'n': 'test'},
    )
    await graph.query(
        'CREATE VECTOR INDEX FOR (n:Entity) ON (n.name_embedding) '
        "OPTIONS {dimension: 4, similarityFunction: 'cosine'}"
    )
    # The survivor: a RANGE index that drop_vector_indices must NOT touch.
    await graph.query('CREATE INDEX FOR (n:Entity) ON (n.name)')
    await graph.query(
        'CREATE VECTOR INDEX FOR ()-[e:RELATES_TO]-() ON (e.fact_embedding) '
        "OPTIONS {dimension: 4, similarityFunction: 'cosine'}"
    )
    # MANDATORY (task 3377) — do NOT drop this when copying this fixture.
    # FalkorDB builds indices asynchronously, and querying an under-construction
    # index silently succeeds for queries the engine would otherwise reject, so
    # without this barrier these tests are a false-green generator.  On this
    # single-node graph the build finishes before the first poll, so it costs one
    # CALL db.indexes() round-trip in practice.  If this fixture is ever widened
    # to seed bulk data, raise the module-wide @pytest.mark.timeout(30) alongside
    # it — the barrier's own budget is 10s.
    await await_index_operational(graph)
    try:
        yield graph
    finally:
        with contextlib.suppress(Exception):
            await graph.delete()
        with contextlib.suppress(Exception):
            await client.aclose()


class TestDropVectorIndicesLive:
    """End-to-end: the vector indices really go away, and only they do.

    MEASURED 2026-08-16 on throwaway graph ``_impl3769_probe``, which is what
    these assertions encode:

    * ``DROP INDEX ON :Entity(emb)`` — the old-style form the pre-fix code issued
      via ``drop_index()`` — fails against a live VECTOR index with
      ``ERR Unable to drop index on :Entity(emb): no such index.``
    * ``DROP VECTOR INDEX FOR (n:Entity) ON (n.emb)`` reports ``Indices deleted: 1``.
    * The NODE form against a RELATIONSHIP vector index fails the same way as the
      first bullet, so ``entity_type`` is load-bearing.
    * ``DROP VECTOR INDEX`` on a property carrying ``['RANGE', 'VECTOR']`` leaves
      ``['RANGE']`` behind — it is surgical.

    MEASURED 2026-08-27 (task 4748), extending the above with the post-drop
    rebuild window that motivates every ``await_index_operational`` call in
    this class's tests:

    * After ``DROP VECTOR INDEX FOR (n:Entity) ON (n.name_embedding)`` on a
      graph whose Entity index carries both ``name_embedding`` VECTOR and
      ``name`` RANGE, one ``CALL db.indexes()`` can return two Entity rows —
      ``['name'] {'name': ['RANGE']}`` at ``'[Indexing] N/M: UNDER
      CONSTRUCTION'`` beside ``['name_embedding', 'name']
      {'name_embedding': ['VECTOR'], 'name': ['RANGE']}`` at ``'OPERATIONAL'``.
    * The window is ~4 ms on the 1-node fixture graph and 0.21-0.75s at 50,000
      nodes; it is NOT a read-path artifact (RO_QUERY and QUERY agree at every
      instant).
    * A relationship index with a single field opens no window — dropping it
      removes the whole index.
    """

    @pytest.mark.asyncio
    async def test_drops_both_vector_indices_and_spares_the_range_index(
        self, mock_config, live_vector_graph,
    ):
        backend = GraphitiBackend(mock_config)
        # HAZARD: inject the driver and never call initialize() —
        # FalkorDriver.__init__ fire-and-forgets build_indices_and_constraints(),
        # and the absence of indices on real graphs is esc-3375-1's protected
        # evidence.  This graph is a unique_graph_name throwaway either way.
        backend._driver = _MultiTenantFalkorDriver(host=FALKOR_HOST, port=FALKOR_PORT)
        try:
            before = await backend.list_indices(group_id=TEST_GRAPH)
            assert _vector_properties(before) == {
                ('Entity', 'name_embedding'),
                ('RELATES_TO', 'fact_embedding'),
            }, f'fixture did not seed the expected vector indices: {before!r}'

            dropped = await backend.drop_vector_indices(group_id=TEST_GRAPH)

            # Both entries come back, in the documented {'label', 'field'} shape
            # with `field` a per-property STRING (never the record's field LIST).
            assert {(d['label'], d['field']) for d in dropped} == {
                ('Entity', 'name_embedding'),
                ('RELATES_TO', 'fact_embedding'),
            }
            assert all(set(d) == {'label', 'field'} for d in dropped)
            assert all(isinstance(d['field'], str) for d in dropped)

            # MANDATORY (task 4748) — see the drop-side HAZARD in the module
            # docstring; do NOT delete as redundant with the fixture's
            # barrier. Without this, `after` can capture the stale row and
            # the `_vector_properties(after) == set()` assertion below fails
            # with "vector indices survived drop_vector_indices()".
            await await_index_operational(live_vector_graph)

            after = await backend.list_indices(group_id=TEST_GRAPH)
            # THE lock-in: no 'VECTOR' survives anywhere in any record's types.
            assert _vector_properties(after) == set(), (
                f'vector indices survived drop_vector_indices(): {after!r}'
            )
            # THE surgical-precision case: the RANGE index is untouched.
            assert ('Entity', 'name') in {
                (record['label'], prop)
                for record in after
                for prop, raw in (record.get('type') or {}).items()
                if _types_contain(raw, 'RANGE')
            }, f'the RANGE index on Entity.name was collateral damage: {after!r}'
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_is_idempotent_and_reports_zero_on_a_second_run(
        self, mock_config, live_vector_graph,
    ):
        """A second run finds nothing to drop and says so truthfully.

        Guards the inverse reading of the old defect: "Dropped 0 VECTOR
        index(es)" must mean the graph really has none, which it did not before
        task 3769.
        """
        backend = GraphitiBackend(mock_config)
        backend._driver = _MultiTenantFalkorDriver(host=FALKOR_HOST, port=FALKOR_PORT)
        try:
            first = await backend.drop_vector_indices(group_id=TEST_GRAPH)
            assert len(first) == 2

            # MANDATORY (task 4748) — see the drop-side HAZARD in the module
            # docstring; do NOT delete as redundant with the fixture's
            # barrier. drop_vector_indices() opens with its own
            # list_indices(), so without this barrier pass 2's internal read
            # can land in the rebuild window, see the stale
            # Entity{name_embedding: ['VECTOR']} row, and re-issue
            # `DROP VECTOR INDEX FOR (n:Entity) ON (n.name_embedding)` — which
            # FalkorDB then answers with:
            #   redis.exceptions.ResponseError: Unable to drop index on
            #   :Entity(name_embedding): no such index.
            # drop_vector_indices() deliberately does not absorb per-statement
            # failures, so that ERRORS the test rather than failing an
            # assertion.
            await await_index_operational(live_vector_graph)

            second = await backend.drop_vector_indices(group_id=TEST_GRAPH)
            assert second == []
        finally:
            await backend.close()


# ---------------------------------------------------------------------------
# TestDropRebuildWindow's real-time budgets (task 4748; re-derived task 4972,
# amendment pass). Named constants rather than inline literals because the
# class-level pytest.mark.timeout below is only sound as ARITHMETIC over them,
# and an invariant nobody can recompute is an invariant nobody maintains.
# ---------------------------------------------------------------------------

#: Per-barrier budget for every await_index_operational on the 10,000-node bulk
#: graph. 30, not await_index_operational's 10s default and not the 20 this
#: task first shipped: under 16-way FalkorDB contention the initial HNSW build
#: measured 2.87-24.45s (median 14.47s) and the post-drop rebuild 0.00-22.86s
#: (median 10.60s), so 20 sits BELOW both measured maxima and would trade this
#: task's observation flake for a barrier-timeout flake. 30 clears both with
#: ~20% headroom. The default itself is deliberately left alone: every OTHER
#: barrier in the suite gates a 1-node graph where the build finishes before
#: the first poll, and raising the default globally would buy them nothing
#: while slowing genuine failure reporting everywhere.
_BULK_BARRIER_S = 30.0

#: Deadline for ONE observation attempt. 2.0s is >2x the worst detect latency
#: ever measured (0.86s, under 16-way FalkorDB contention), and when the window
#: IS open it is caught on the very first read (measured first-read gap
#: 0.002-0.29s) -- so a shorter per-attempt deadline buys cheaper MISSES
#: without risking a truncated observation.
_OBSERVE_ATTEMPT_S = 2.0

#: Independent openings observed before giving up. Measured per-attempt
#: observation is 30/33 = 90.9% under the offline lane's own scheduling, so 5
#: puts the residual miss probability at ~5e-6, and still <=1e-3 under the
#: pessimistic 75% seen in one 8-trial batch (attempts are not perfectly
#: independent).
_OBSERVE_ATTEMPTS = 5

#: WALL-CLOCK ceiling on the whole retry phase, enforced by shrinking each
#: re-open barrier to the time actually left (see _reopen_rebuild_window).
#:
#: Bounding the retry in attempts ALONE is not enough, because the attempt
#: count multiplies the barrier budgets: 5 attempts x 4 re-opens x 2 barriers
#: is 240s of budget that a slow-but-SUCCEEDING barrier can walk through
#: without any of them ever raising. pytest-timeout runs with
#: timeout_method='thread' here, whose handler ends in os._exit(1) and kills
#: the whole xdist worker, so "walks past the class mark" is not a test
#: failure -- it takes unrelated tests down with it. A wall-clock bound is
#: what makes the class mark below checkable arithmetic instead of a guess.
#:
#: 100s against a measured retry-phase cost of median 0.10s / max 8.20s, i.e.
#: >12x the worst observed. It binds only when barriers are running tens of
#: seconds each -- and that is the regime where it costs least, because a
#: FalkorDB slow enough to spend 100s on re-opens is also one whose rebuild
#: window is WIDE (the 50,000-node measurements show a slower rebuild means a
#: longer window), so the first attempt is near-certain to have observed it
#: already. Exhausting this budget raises its OWN diagnosis, never job (a)'s
#: "FalkorDB stopped rebuilding" -- see _reopen_rebuild_window.
_RETRY_PHASE_BUDGET_S = 100.0


@pytest_asyncio.fixture
async def bulk_vector_graph():
    """Seed a throwaway 10,000-node graph with a mixed VECTOR+RANGE Entity index.

    Backs TestDropRebuildWindow below — see its docstring for the measurements
    that justify 10,000 as the seed size. 1,000 was tried first and rejected:
    re-measured empirically (task 4748 amendment pass, loadavg ~185 on 32
    cores) at only 12/15 phantom sightings even WITH polling, i.e. the window
    can fail to open at all at that size, which no poll interval can fix.
    50,000 was rejected too — it makes the window near-certain but costs
    8.5-9.2s to build UNCONTENDED against a measured 10-17x contention
    multiplier, trading an observation flake for a barrier-timeout flake.

    10,000 is NOT a size at which the window always opens, and the claim that
    it "measured 25/25 ... every one on the very first post-drop read" is
    RETRACTED by task 4972: that batch was taken at normal scheduling
    priority, and under the offline lane's own ``nice -n 19 ionice -c3``
    the first-attempt rate is 28/30 = 93.3%. What makes 10,000 the right size
    is that the class RE-OPENS the window up to five times rather than
    needing it on one try, which keeps the happy path at ~1.4s. The mixed index
    (``name_embedding`` VECTOR merged with ``name`` RANGE, both on
    ``:Entity``) is what makes the class's ``DROP VECTOR INDEX`` a REBUILD
    rather than a removal: FalkorDB keeps the label's index record alive to
    carry the surviving RANGE field, and that rebuild is what opens the
    window under test. A single-field vector index — like the module
    fixture's ``live_vector_graph`` RELATES_TO index — never opens this
    window, because dropping its only field removes the whole record
    outright.

    No stale-graph pre-delete here (unlike ``live_vector_graph`` above):
    ``graph_name`` is minted fresh from ``unique_graph_name`` on every call,
    so no prior run's graph can ever share this name to delete.
    """
    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    graph_name = unique_graph_name('4748_drop_rebuild_window')
    graph = client.select_graph(graph_name)
    await graph.query(
        'UNWIND range(1, 10000) AS i '
        'CREATE (:Entity {name: "n"+i, name_embedding: vecf32([1.0, 2.0, 3.0, 4.0])})'
    )
    await graph.query(
        'CREATE VECTOR INDEX FOR (n:Entity) ON (n.name_embedding) '
        "OPTIONS {dimension: 4, similarityFunction: 'cosine'}"
    )
    # The RANGE half of the merged index. Without this, dropping the VECTOR
    # field below removes the Entity index record entirely instead of
    # rebuilding it, and the window this fixture exists to open never opens.
    await graph.query('CREATE INDEX FOR (n:Entity) ON (n.name)')
    # (task 4748, re-budgeted task 4972) At this seed size the initial HNSW
    # build measured 1.3-2.8s across loadavg ~96-185 on 32 cores -- but that
    # is the UNCONTENDED figure. Under 16-way FalkorDB contention the same
    # build measured 2.87-24.45s (median 14.47s), i.e. a 10-17x multiplier
    # that blows straight through await_index_operational's 10s default, so
    # this call site carries an explicit budget above that measured maximum
    # (see _BULK_BARRIER_S for why 30 and not 20). If this fixture's seed
    # size is ever raised, re-measure before assuming even 30s covers it (see
    # TestDropRebuildWindow's measurement table; at 50,000 nodes the same
    # build measured 8.5-9.2s UNCONTENDED and needed timeout_s=60).
    await await_index_operational(graph, timeout_s=_BULK_BARRIER_S)
    try:
        yield graph
    finally:
        with contextlib.suppress(Exception):
            await graph.delete()
        with contextlib.suppress(Exception):
            await client.aclose()


def _entity_index_rows(result) -> list[tuple[dict, str]]:
    """Return (types, status) for every 'Entity'-labeled row in a raw db.indexes() result.

    Resolves the label/types/status columns BY NAME from ``result.header``,
    deliberately independent of ``list_indices()`` / ``resolve_header_positions``
    — the code path this module's barriers protect — for the same reason
    ``_vector_properties`` above hand-rolls its own read: a bug in the helper
    under test must not be able to make its own verification pass.

    Raises:
        IndexHeaderError: ``CALL db.indexes()`` no longer exposes one of the
            label/types/status columns. A DISTINCT TYPE, not a bare ``assert``,
            because ``_observe_phantom_window`` below has to tell this apart
            from an ordinary missed window — and a caller that discriminates
            by exception TYPE cannot be broken by someone rewording a message,
            which discriminating by message text can. This is the same
            two-modes-one-string collapse ``await_index_operational`` already
            uses ``IndexHeaderError`` to prevent, reused rather than re-forked
            so the blessed name a caller catches stays one name.
    """
    header_names = [col[1] for col in result.header]
    for col in ('label', 'types', 'status'):
        if col not in header_names:
            raise IndexHeaderError(
                f'CALL db.indexes() has no {col!r} column (header={header_names}); '
                'await_index_operational cannot be trusted.'
            )
    label_idx = header_names.index('label')
    types_idx = header_names.index('types')
    status_idx = header_names.index('status')
    return [
        (row[types_idx], row[status_idx])
        for row in result.result_set
        if row[label_idx] == 'Entity'
    ]


def _row_has_vector(types: dict) -> bool:
    """True if any property in a raw db.indexes() 'types' dict carries VECTOR."""
    return any(_types_contain(raw, 'VECTOR') for raw in types.values())


async def _reopen_rebuild_window(graph, deadline: float | None = None) -> None:
    """Re-open the post-DROP rebuild window by re-merging, then re-dropping, the VECTOR index.

    The ``reopen`` half of the ``retry_until_observed`` call in
    TestDropRebuildWindow below. The window it re-opens is a genuine RACE: on
    a miss FalkorDB had already finished the rebuild inside the ``DROP``
    round-trip, so there is nothing left for any client-side poll to catch and
    the ONLY remedy is a fresh opening (see the class docstring for the
    measurement).

    The four steps run in exactly this order, and the ORDER IS LOAD-BEARING:

    1. settle -- the missed attempt's own ``DROP`` may still have a rebuild in
       flight. Creating a vector index into an in-flight rebuild races it, so
       this barrier must come BEFORE the create, not after it;
    2. re-``CREATE VECTOR INDEX`` on ``(n:Entity) ON (n.name_embedding)`` --
       the module's exact statement, which re-merges the VECTOR field with the
       ``name`` RANGE index that ``DROP VECTOR INDEX`` deliberately spared;
    3. settle again -- the HNSW build is asynchronous (task 3377), and
       dropping an under-construction index is not the state under test;
    4. re-``DROP`` -- which is what actually opens the window the next attempt
       observes.

    Args:
        graph: The live bulk graph to re-open the window on.
        deadline: ``time.monotonic()`` value the whole RETRY PHASE must finish
            by, or ``None`` for no wall-clock bound. Each barrier below is
            shrunk to the time actually left, and a re-open that starts with
            none left raises immediately. This is what makes the class-level
            ``pytest.mark.timeout`` sound arithmetic rather than a guess: see
            :data:`_RETRY_PHASE_BUDGET_S`.
    """
    # Both barriers gate the SAME 10,000-node graph as the fixture's build
    # barrier and the test's final one, and carry the same _BULK_BARRIER_S
    # budget for the same measured reason (16-way FalkorDB contention pushed
    # the build to 2.87-24.45s and the post-drop rebuild to 0.00-22.86s, past
    # await_index_operational's 10s default).
    #
    # A barrier raising here aborts the WHOLE retry -- reopen failures
    # propagate; retry_until_observed only absorbs a falsy OBSERVATION (pinned
    # by test_fm_helpers.py::TestRetryUntilObserved
    # ::test_an_exception_from_reopen_aborts_the_retry_rather_than_being
    # _retried_around). That is deliberate: continuing to observe a window
    # that was never successfully re-opened would burn the budget and then
    # report the wrong diagnosis. It is also why the budget matters -- an
    # under-budgeted barrier would reintroduce the contention flake this task
    # removes.
    def _budget() -> float:
        """Seconds this barrier may spend: the full budget, or whatever is left."""
        if deadline is None:
            return _BULK_BARRIER_S
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            # Its OWN diagnosis, deliberately not job (a)'s. Running out of
            # wall clock while barriers crawl means FalkorDB is contended, not
            # that it stopped rebuilding the merged index in place -- and
            # reporting the latter for the former is exactly the
            # two-modes-one-string collapse this module keeps guarding
            # against.
            raise AssertionError(
                f'the {_RETRY_PHASE_BUDGET_S}s retry-phase budget was spent before this '
                'window could be re-opened. That is FalkorDB running slow (index '
                'barriers crawling under contention), NOT evidence the drop-side '
                'rebuild window is gone -- do not read this as the "stopped '
                'rebuilding in place" failure this class also reports.'
            )
        return min(_BULK_BARRIER_S, remaining)

    await await_index_operational(graph, timeout_s=_budget())
    await graph.query(
        'CREATE VECTOR INDEX FOR (n:Entity) ON (n.name_embedding) '
        "OPTIONS {dimension: 4, similarityFunction: 'cosine'}"
    )
    await await_index_operational(graph, timeout_s=_budget())
    await graph.query('DROP VECTOR INDEX FOR (n:Entity) ON (n.name_embedding)')


class TestReopenRebuildWindowBudget:
    """The wall-clock guard on ``_reopen_rebuild_window`` — deterministic, no FalkorDB.

    Needs no live graph because an expired *deadline* is refused BEFORE the
    first barrier runs, which is the whole point: the guard must fire without
    spending anything. ``graph=None`` is therefore load-bearing evidence, not
    a shortcut — a guard that touched the graph first could not use it.

    (The module-level ``falkor_skipif`` still skips this with the rest of the
    module when FalkorDB is down. That is deliberate: the budget it pins only
    exists to bound a live run, so pinning it in isolation would buy nothing.)
    """

    @pytest.mark.asyncio
    async def test_an_expired_deadline_is_refused_before_any_barrier_runs(self):
        """An exhausted retry-phase budget raises its OWN diagnosis, never job (a)'s.

        Two things are pinned, and the second matters more than the first:

        * it raises rather than starting a barrier it has no time for
          (``graph=None`` proves nothing was touched — a barrier call would
          have raised AttributeError instead);
        * the message says CONTENDED, not "stopped rebuilding". Running out of
          wall clock means FalkorDB is slow, and reporting that as the
          drop-side rebuild window being gone is exactly the
          two-modes-one-string collapse this module keeps guarding against.
        """
        expired = time.monotonic() - 1.0

        with pytest.raises(AssertionError, match='retry-phase budget was spent'):
            await _reopen_rebuild_window(None, deadline=expired)

    @pytest.mark.asyncio
    async def test_no_deadline_leaves_the_full_barrier_budget(self, monkeypatch):
        """``deadline=None`` means unbounded: the barrier gets the full _BULK_BARRIER_S.

        The default has to stay the un-shrunk budget so the helper is still
        usable outside the retry phase, and so a caller that forgets the
        deadline gets the SAFE behaviour (a full budget) rather than a
        silently truncated barrier.

        Asserted through the recorded ``timeout_s`` rather than wall clock, so
        this cannot itself become the next timing flake.
        """
        budgets: list[float] = []

        async def fake_barrier(_graph, timeout_s=10.0):
            budgets.append(timeout_s)

        async def fake_query(_statement):
            return None

        class _Graph:
            query = staticmethod(fake_query)

        monkeypatch.setattr(sys.modules[__name__], 'await_index_operational', fake_barrier)
        await _reopen_rebuild_window(_Graph(), deadline=None)

        assert budgets == [_BULK_BARRIER_S, _BULK_BARRIER_S], (
            f'expected both barriers to get the full budget, got {budgets!r}'
        )


# 240, overriding the module-level pytest.mark.timeout(30) for THIS CLASS ONLY
# (pytest-timeout resolves via get_closest_marker, so the module mark still
# correctly governs TestDropVectorIndicesLive above, which runs on the 1-node
# live_vector_graph and is fast).
#
# FAILURE MODE, which is why the number must be derived and not eyeballed.
# fused-memory/pyproject.toml sets timeout_method='thread', whose handler ends
# in os._exit(1) -- that kills the whole xdist worker rather than failing one
# test, taking unrelated tests with it. And pytest-timeout runs here with
# func_only=False (see the module pytestmark above), so FIXTURE SETUP counts
# against the mark too. So the mark must sit above the arithmetic worst case
# of every budget this class can spend, INCLUDING barriers that merely run
# long while still succeeding -- a 25s barrier inside a 30s budget never
# raises the clean AssertionError, it just walks the clock.
#
# THE ARITHMETIC, worst case, all constants defined above the bulk fixture:
#
#   fixture seed queries (unbarriered CREATEs)             <=  30s  (allowance)
#   fixture build barrier            _BULK_BARRIER_S       <=  30s
#   retry phase                      _RETRY_PHASE_BUDGET_S <= 100s  (wall-bounded)
#     + the last observation, which may start just inside
#       the deadline           _OBSERVE_ATTEMPT_S          <=   2s
#   final post-drop barrier          _BULK_BARRIER_S       <=  30s
#                                                          --------
#                                                             192s
#
# 240 leaves ~48s of slack over that ceiling. The retry phase is the term that
# used to be unbounded: 5 attempts x 4 re-opens x 2 barriers is 240s of budget
# on its own, which is why _reopen_rebuild_window shrinks each barrier to the
# wall clock actually left rather than letting the attempt count multiply the
# per-barrier budget. Whenever any of those constants changes, recompute this
# sum -- that is the invariant, not the literal 240.
#
# MEASURED COST, for contrast with the ceiling above: the realistic post-fix
# case is ~7s (5 x 2.0s poll + 4 x ~1.3s reopen, reopen measured 1.12-1.42s /
# median 1.27s, plus the fixture), and 30 consecutive lane-condition trials ran
# 2.73s min / 3.34s median / 7.64s max in-test. The headroom is for the
# contended tail, not the expected spend.
@pytest.mark.timeout(240)
class TestDropRebuildWindow:
    """PREMISE PIN for the post-DROP rebuild window — task 4748, de-flaked by task 4972.

    This is NOT a red-first TDD test; it is expected to PASS on introduction.
    A timing race cannot be reliably RED-tested: the phantom this test pins
    only shows up in a fraction of single-shot reads (see the measurements
    below), so a test built to fail without the fix would itself be a new
    flake — the exact defect task 4972 removed from this very test. This
    mirrors test_list_indices_integration.py::TestCallDbIndexesOverRoQuery
    .test_db_indexes_result_shape_matches_the_barriers_assumptions (task
    3377's build-side pin), which rejects the same idea for the same reason.
    What makes "expected to PASS" actually TRUE is not that the window is
    certain to open — it is not — but that the window is RE-OPENED up to five
    times and observed each time, and that exhausting all five still raises
    loudly (see ``retry_until_observed`` and ``_reopen_rebuild_window``).

    Its TWO durable jobs, both preserved by the retry:

    (a) fail loudly if a future FalkorDB upgrade removes the drop-side
        rebuild transient that every barrier task 4748 added exists to close,
        making them dead weight. Bounded retry is what keeps this job alive:
        the END STATE alone cannot tell "rebuilt too fast to see" from "no
        longer rebuilds asynchronously" — both land on one Entity row with no
        VECTOR — but five independent openings all failing to show the
        transient is strong evidence it is genuinely gone, and that still
        raises rather than passing quietly;
    (b) fail loudly if it changes the column names or the exact
        'OPERATIONAL' ready sentinel await_index_operational depends on. This
        job is why ``_observe_phantom_window`` re-raises ``IndexHeaderError``
        by TYPE instead of swallowing every AssertionError as a miss.

    MEASURED (task 4748, nice -n 19, loadavg ~96 on 32 cores, FalkorDB module
    v41800): after ``DROP VECTOR INDEX FOR (n:Entity) ON (n.name_embedding)``
    on a graph whose Entity index merges ``name_embedding`` VECTOR with
    ``name`` RANGE, the very next ``CALL db.indexes()`` can return TWO Entity
    rows — a fresh ``['name']`` row still ``'[Indexing] N/M: UNDER
    CONSTRUCTION'`` beside the stale ``['name_embedding', 'name']`` row still
    reporting ``'OPERATIONAL'`` and still advertising the just-dropped VECTOR
    type. Phantom rate on the FIRST (single-shot, un-polled) post-drop read,
    by seed size:

        nodes      phantom rate              window
        1          ~1 in 40-70 (tight loop)   ~4 ms
        1,000      1/3 (3/3 when polled)      40-64 ms
        10,000     1/3 (3/3 when polled)      52-135 ms
        50,000     7/7                        0.21-0.75s
        200,000    3/3                        ~0.73s

    RETRACTED (task 4972). This docstring used to claim 10,000 nodes measured
    "25/25 ... every one on the very first post-drop read", and concluded
    that 10,000 was "the smallest size this task could empirically confirm
    never drops the phantom". Both are wrong, and the second is what made
    this test a ~7% flake in the offline lane. Re-measured under the lane's
    ACTUAL conditions — ``DF_VERIFY_ROLE=offline nice -n 19 ionice -c3``,
    serial (``-p no:xdist -o addopts=``), loadavg ~92-471 on 32 cores — the
    FIRST-attempt observation rate at 10,000 nodes is 28/30 = 93.3%, with a
    second 8-trial batch at 6/8. The original 25/25 was taken at NORMAL
    scheduling priority, and that still reproduces (20/20 serial, un-niced).
    It is the NICENESS, not the seed size, that exposes the race.

    THE MECHANISM, stated as the measurement it is. On a miss the window is
    not merely hard to catch — it never opens at all. The first post-DROP
    read, taken 6.8 ms after ``DROP`` returned, already showed ONE Entity
    row, already ``'OPERATIONAL'``, and it stayed that way across all 167
    polls of a 5 s deadline; the ``DROP`` round-trip on that trial was
    49.7 ms against 0.6-35 ms typical. FalkorDB had completed the merged-index
    rebuild INSIDE the ``DROP`` call. This EXTENDS rather than contradicts
    the principle this docstring already stated correctly — "no poll interval
    fixes an absent window". What was wrong was inferring from it that a
    large enough seed makes the window CERTAIN. The only lever that acts on
    an absent window is to open a new one.

    WHY SEED SIZE IS NOT THE LEVER, so a future reader does not retry it.
    50,000 nodes does widen the window to 0.21-0.75s, far past any client
    scheduling jitter — but it costs 8.5-9.2s to build UNCONTENDED, against a
    measured 10-17x contention multiplier on the 10,000-node build (1.3s
    serial vs 2.87-24.45s under 16-way FalkorDB contention). At 50,000 that
    projects well past any sane barrier budget or pytest timeout, trading a
    ~7% observation flake for a barrier-timeout flake plus the
    ``os._exit(1)`` xdist-worker kill that ``timeout_method='thread'`` ends
    in. Retrying at 10,000 keeps the happy path at ~1.4s and pays only on a
    miss. 1,000 nodes is likewise rejected: 12/15 (80%) even WITH a
    5ms-interval/5s-deadline poll.

    VALIDATION of the task-4972 fix, all on this branch, FalkorDB module
    v41800, 32 cores:

    * pre-fix baseline, lane conditions, loadavg 141-471: 13/15 — the two
      failures both reported ``last saw [({'name': ['RANGE']},
      'OPERATIONAL')]``, i.e. exactly the absent-window signature above;
    * post-fix, same command, 30 consecutive trials at loadavg ~143:
      30/30, in-test wall time 2.73s min / 3.34s median / 7.64s max (the max
      is a trial that paid the retry);
    * the re-open path probed directly, 10 trials: the window was observed
      10/10 on the first attempt, 10/10 after re-open #1 and 10/10 after
      re-open #2 (30/30 per-attempt), proving a RE-OPENED window is caught as
      reliably as the original rather than the re-open merely burning budget.
      Re-open cost 1.12-1.42s, median 1.27s;
    * the lane's exact confirmation command
      (``DF_VERIFY_ROLE=offline nice -n 19 ionice -c3 pytest -m integration
      -p no:xdist -o addopts= -q``): 61 passed in 130.06s;
    * the default parallel run (``pytest -m integration``, ``-n auto --dist
      loadgroup``): 61 passed in 89.58s.
    """

    @pytest.mark.asyncio
    async def test_drop_vector_index_opens_a_rebuild_window_the_barrier_closes(
        self, bulk_vector_graph,
    ):
        """DROP opens the phantom window; await_index_operational closes it."""
        graph = bulk_vector_graph
        await graph.query('DROP VECTOR INDEX FOR (n:Entity) ON (n.name_embedding)')

        last_rows: list[tuple[dict, str]] = []

        async def _phantom_rows():
            nonlocal last_rows
            result = await graph.query('CALL db.indexes()')
            rows = _entity_index_rows(result)
            last_rows = rows
            # (a) the phantom is real: MORE THAN ONE Entity row, one of them
            # still listing name_embedding VECTOR; (b) await_index_operational
            # has a signal to wait on: at least one row is not 'OPERATIONAL'.
            is_phantom = (
                len(rows) > 1
                and any(_row_has_vector(types) for types, _ in rows)
                and any(status != 'OPERATIONAL' for _, status in rows)
            )
            return rows if is_phantom else None

        # ONE bounded attempt to catch the window. The 5ms interval is
        # deliberately tight against the measured 52-135ms window at 10,000
        # nodes -- a round trip costs ~0.2ms, so this buys many reads inside
        # the window rather than one. See _OBSERVE_ATTEMPT_S for why the
        # per-attempt deadline came down from 5.0s to 2.0s.
        async def _observe_phantom_window():
            try:
                return await poll_until(
                    _phantom_rows,
                    timeout=_OBSERVE_ATTEMPT_S,
                    interval=0.005,
                    message='phantom rebuild window not observed within this attempt',
                )
            except IndexHeaderError:
                # NOT a missed window: _entity_index_rows raises this when
                # FalkorDB no longer exposes the label/types/status columns,
                # which is durable job (b). Swallowing it as a miss would
                # spend the whole retry budget and then report job (a)'s
                # "stopped rebuilding" diagnosis for a completely different
                # defect -- the two-modes-one-string collapse
                # IndexHeaderError exists to prevent.
                #
                # Discriminated by TYPE, never by comparing str(exc) against
                # poll_until's pass-through message: poll_until happens to
                # re-raise the caller's message verbatim today, but that is
                # not a contract it documents, and its sibling
                # poll_until_stable already decorates one of its two failure
                # messages. A message-equality guard would flip to re-raising
                # EVERY ordinary miss the moment poll_until gained a prefix,
                # silently undoing this whole de-flake with no unit test
                # anywhere to catch it.
                raise
            except AssertionError:
                # poll_until's own deadline: the window did not open for this
                # attempt. Feed it to the retry rather than failing the test.
                return None

        # The window is a RACE, not merely a narrow target: on a miss FalkorDB
        # had already finished the rebuild INSIDE the DROP round-trip, so no
        # interval and no deadline on a single attempt can help (see the class
        # docstring). Re-open it and look again -- see _OBSERVE_ATTEMPTS for
        # why five times. Cost is paid only on a miss: measured retry-phase
        # cost median 0.10s, max 8.20s.
        #
        # The deadline is taken HERE, not inside the reopen, so it bounds the
        # whole phase rather than restarting per re-open. Attempts alone do
        # not bound wall clock (see _RETRY_PHASE_BUDGET_S and the class-level
        # timeout arithmetic above); this is what does.
        retry_deadline = time.monotonic() + _RETRY_PHASE_BUDGET_S
        try:
            await retry_until_observed(
                _observe_phantom_window,
                reopen=lambda: _reopen_rebuild_window(graph, deadline=retry_deadline),
                attempts=_OBSERVE_ATTEMPTS,
                message=(
                    'Expected the drop-side rebuild phantom (>1 Entity row, one '
                    'still VECTOR-typed, one not yet OPERATIONAL), each attempt '
                    'against an INDEPENDENTLY re-opened window (re-create then '
                    're-drop the merged VECTOR+RANGE index). If FalkorDB has '
                    'stopped rebuilding the merged index in place, every '
                    'post-drop barrier this task (4748) added is dead weight -- '
                    'see this class docstring before deleting them.'
                ),
            )
        except AssertionError as exc:
            # last_rows is only known once the attempts have run, so it cannot
            # ride in on `message`; chain it on instead.
            raise AssertionError(f'{exc} Last saw {last_rows!r}.') from exc

        # (c) The barrier is SUFFICIENT: once satisfied, the phantom is gone.
        # _BULK_BARRIER_S, not the 10s default, for the same measured reason
        # as the fixture's build barrier: under 16-way FalkorDB contention the
        # post-drop rebuild measured 0.00-22.86s (median 10.60s) on this
        # 10,000-node graph, so the default has no headroom here. This one is
        # NOT deadline-shrunk -- it is the assertion the test exists to make,
        # not part of the retry phase, and truncating it would turn a slow
        # rebuild into a false "the barrier is not sufficient" failure.
        await await_index_operational(graph, timeout_s=_BULK_BARRIER_S)

        result = await graph.query('CALL db.indexes()')
        entity_rows = _entity_index_rows(result)
        assert len(entity_rows) == 1, (
            f'expected exactly one Entity row once the barrier is satisfied but '
            f'got {len(entity_rows)}: {entity_rows!r}'
        )
        assert not _row_has_vector(entity_rows[0][0]), (
            f'expected no VECTOR type to survive the barrier: {entity_rows[0]!r}'
        )
