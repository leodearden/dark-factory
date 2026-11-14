"""What a test needs to drive a live FalkorDB safely: connection constants,
reachability gate, per-run graph names, and the index-readiness barrier.

The connection constants name the SCRATCH FalkorDB, never production
(OPERATIONS.md "The scratch FalkorDB"). _falkor_index_live.py builds the
index-provisioning fixtures on top of this module.

Lives outside conftest.py, imported by name, under the convention stated in
_fm_helpers' module docstring.
"""

import contextlib
import uuid
from collections.abc import MutableMapping
from pathlib import Path

import pytest
from _fm_polling import poll_until
from shared.lane_probe import runs_under_offline_lane

from fused_memory.backends.falkor_indices import IndexHeaderShapeError, resolve_header_positions
from fused_memory.config.falkordb_endpoints import scratch_falkordb_from_env

# ---------------------------------------------------------------------------
# Shared FalkorDB test scaffolding (task 3502)
# ---------------------------------------------------------------------------
#
# Connection constants, reachability probe and skip-marker factory for tests
# that drive a live FalkorDB. Shaped like the Qdrant scaffolding in
# _fm_qdrant_live.py so the two reachability-probe sets (constants → probe →
# skipif factory) read as one convention. tests/test_falkor_probe_routing_guard.py
# enforces that no test module forks this scaffolding back into itself.
# ---------------------------------------------------------------------------

# Resolved once at import, so a FALKOR_HOST/FALKOR_PORT aimed at production
# fails every live module loudly at collection (ProductionFalkorDBTarget).
_SESSION_FALKORDB = scratch_falkordb_from_env()
FALKOR_HOST: str = _SESSION_FALKORDB.host
FALKOR_PORT: int = _SESSION_FALKORDB.port

_COMPOSE_FILE = Path(__file__).resolve().parents[1] / 'docker' / 'docker-compose.yml'


class ScratchFalkorDBUnreachable(RuntimeError):
    """An offline-lane session found no scratch FalkorDB to run against."""


def prepare_scratch_falkordb_session(environ: MutableMapping[str, str]) -> None:
    """Aim config-built FalkorDB clients at the session's scratch server; under the lane, require it.

    Sets ``FALKORDB_URI`` in *environ* before collection, so module-scoped
    fixtures and backends built from the config see it too; a test-body
    ``monkeypatch.setenv`` still wins. Under the offline lane an unreachable
    server ends the session: a pass that skipped every FalkorDB test would
    read as green.

    Raises:
        ScratchFalkorDBUnreachable: under the offline lane, `_falkor_available()` is False.
    """
    environ['FALKORDB_URI'] = _SESSION_FALKORDB.uri
    if runs_under_offline_lane(environ) and not _falkor_available():
        raise ScratchFalkorDBUnreachable(
            f'the scratch FalkorDB at {_SESSION_FALKORDB.uri} is not reachable, and an '
            f'offline-lane session must not pass by skipping every FalkorDB test. Start it '
            f'with `docker compose -f {_COMPOSE_FILE} up -d --no-deps falkordb-scratch` '
            f'(OPERATIONS.md "The scratch FalkorDB").'
        )


def _falkor_available() -> bool:
    """Probe whether a FalkorDB instance is reachable at FALKOR_HOST:FALKOR_PORT.

    Returns False rather than raising for any failure — construct, query or
    close — and is bounded by ``socket_connect_timeout=2``, so an unreachable
    host costs ~2s at most.

    Uses the falkordb-native sync client rather than ``redis``: redis is not a
    declared dependency of fused-memory, only a transitive of
    graphiti-core[falkordb], so a redis-based probe would break silently if
    falkordb ever switched transport.

    Imports FalkorDB lazily so (a) importing this module does not drag
    falkordb into every importer, and (b) the name resolves at call time,
    which is what lets tests patch ``falkordb.FalkorDB`` deterministically.
    Same convention as ``_fm_qdrant_live.py::_qdrant_available``.

    Deliberately NOT cached: each consuming module pays its own probe at its
    own collection time. Memoising would break the call-time tracking
    ``falkor_skipif`` relies on, and test_integration_marker_real_service.py
    budgets for the per-module cost explicitly.
    """
    from falkordb import FalkorDB

    try:
        client = FalkorDB(
            host=FALKOR_HOST, port=FALKOR_PORT, socket_connect_timeout=2
        )
        try:
            client.select_graph('_probe').query('RETURN 1')
        finally:
            with contextlib.suppress(Exception):
                client.close()
        return True
    except Exception:
        return False


def falkor_skipif(reason: str = 'FalkorDB not reachable'):
    """Return a ``pytest.mark.skipif`` marker gated on `_falkor_available()`.

    A factory rather than a module-level marker constant, so the probe is not
    frozen at this module's first import. Called from a consuming module's
    ``pytestmark``, or as a class/function decorator, the probe instead fires
    at that module's own collection time.

    Resolves ``_falkor_available`` through the module global at call time
    rather than capturing it, so the marker tracks whatever the probe
    currently reports.
    """
    return pytest.mark.skipif(not _falkor_available(), reason=reason)


def unique_graph_name(slug: str) -> str:
    """Return a fresh per-run FalkorDB graph name, ``_test_<slug>_<8 hex>``.

    A fresh name is minted on every call so that concurrent test runs —
    pytest-xdist ``-n auto``, or a CI matrix pointed at one shared FalkorDB —
    never select the same graph and wipe each other's fixtures.

    The caller still owns the rest of the lifecycle: the best-effort delete of
    a stale graph left by a prior run, and the teardown ``graph.delete()`` /
    ``client.aclose()`` pair.

    Args:
        slug: Should embed the owning task id, by convention (e.g.
            ``2207_merge_entities``), so a graph orphaned by a killed run is
            traceable back to the test module that created it.
    """
    return f'_test_{slug}_{uuid.uuid4().hex[:8]}'


# ---------------------------------------------------------------------------
# Shared FalkorDB index-readiness barrier (task 3377; extracted from task 3334)
# ---------------------------------------------------------------------------
#
# Extracted from test_falkor_fulltext_integration.py so every live-index test
# module routes through ONE implementation instead of re-forking it. Public
# (no leading underscore) because it is now imported cross-module, matching
# poll_until / ensure_fresh_collection / collection_vector_size.
# tests/test_falkor_index_barrier_guard.py enforces that routing.
#
# IndexHeaderError is public for the same reason: it is the blessed name a
# caller catches to tell "FalkorDB changed shape" apart from "the index never
# became ready". Keeping it module-private would document a distinction across
# the module boundary that no caller could actually reach.
# ---------------------------------------------------------------------------

class IndexHeaderError(AssertionError):
    """``CALL db.indexes()`` returned a header the barrier cannot read.

    Two shapes: no ``status`` column, or a header entry that is not a
    ``(type, name)`` pair. Wraps the production
    ``falkor_indices.IndexHeaderShapeError`` and carries its message.

    A distinct type from the timeout AssertionError because the two failure
    modes mean different things and call for different operator actions: a
    timeout is a slow or wedged index build, whereas a header shape surprise
    means FalkorDB changed its result shape and the barrier cannot be
    trusted at all. Collapsing the second into a generic timeout message would
    hide a silent no-op behind a plausible-looking timeout.

    Subclasses ``AssertionError`` so a caller that only cares that the barrier
    failed can still catch the broad type, while a caller that needs to act on
    the difference has a name to catch. Callers asserting on the *timeout* path
    specifically should match its message (``'not OPERATIONAL'``) rather than
    the bare type, which this subclass also satisfies.
    """


async def await_index_operational(graph, timeout_s: float = 10.0) -> None:
    """Block until every index on ``graph`` reports ``OPERATIONAL``.

    FalkorDB builds indices **asynchronously**. Querying before the index is
    ready silently returns success for a query the engine would otherwise
    reject. Measured while characterising task 3334: a known-unparseable
    fulltext query issued immediately after
    ``CALL db.idx.fulltext.createNodeIndex(...)`` across six fresh graphs gave
    FAIL, OK, OK, FAIL, FAIL, FAIL — **2 of 6 runs falsely reported success**.
    Range indices are built asynchronously too (measured on a 200k-node graph:
    ``'[Indexing] 628/200000: UNDER CONSTRUCTION'``, never operational within
    40 polls), so this barrier is not fulltext-specific.

    The ``status`` column is resolved **by name** from ``result.header`` rather
    than hardcoded to its current position (7 of 9), so a FalkorDB column
    reorder fails loudly here instead of silently reading the wrong column and
    turning the barrier into a no-op. Resolution goes through
    ``falkor_indices.resolve_header_positions``, the helper shared with
    ``falkor_indices.index_records_from_rows``. The production barrier is
    ``falkor_indices.await_index_catalog_settled``, which deliberately differs
    on an empty catalog (see ``falkor_indices.unsettled_index_statuses``).

    Readiness is an **exact** match on ``'OPERATIONAL'`` for every row, never a
    substring test for ``'UNDER CONSTRUCTION'``: the live not-ready value
    carries a varying progress prefix (``'[Indexing] N/M: UNDER
    CONSTRUCTION'``), so matching the not-ready side would have to anticipate
    every status string FalkorDB might ever emit and would treat an
    unrecognised future status as ready. Testing the ready side is fail-closed
    — anything unfamiliar blocks and eventually raises. An empty
    ``result_set`` counts as NOT ready, since ``all()`` over an empty sequence
    is vacuously true and would otherwise return success having gated nothing.

    Polling is delegated to :func:`poll_until` (event-loop monotonic clock,
    check-before-sleep, raise-on-timeout) rather than hand-rolling a second
    deadline loop. An already-operational index
    therefore costs exactly one ``CALL db.indexes()`` round-trip.

    Raises on timeout — deliberately. A never-operational index must be a loud
    failure; skipping or passing would reintroduce exactly the false-green this
    barrier exists to remove.

    Measured against FalkorDB module v41800 (4.18.0), whose live header is
    ``['label', 'properties', 'types', 'options', 'language', 'stopwords',
    'entitytype', 'status', 'info']``.

    Args:
        graph: An async FalkorDB graph exposing ``await graph.query(...)``.
        timeout_s: Seconds to wait for every index to become operational.
            Callers with a module-wide ``pytest.mark.timeout`` must keep this
            budget inside it.

    Raises:
        IndexHeaderError: ``CALL db.indexes()`` exposed no ``status`` column,
            or a header entry that is not a ``(type, name)`` pair — raised on
            the first poll, never retried until the deadline.
        AssertionError: Not every index became ``OPERATIONAL`` within
            *timeout_s*; the message carries the last-seen status strings.
    """
    last_statuses: list[str] = []

    async def _all_operational() -> bool:
        nonlocal last_statuses
        result = await graph.query('CALL db.indexes()')
        try:
            positions = resolve_header_positions(result.header, {'status': 'status'})
        except IndexHeaderShapeError as exc:
            raise IndexHeaderError(
                f'{exc} The index-readiness barrier cannot be trusted until this '
                'is re-verified.'
            ) from exc
        status_idx = positions['status']
        last_statuses = [row[status_idx] for row in result.result_set]
        return bool(last_statuses) and all(s == 'OPERATIONAL' for s in last_statuses)

    try:
        await poll_until(_all_operational, timeout=timeout_s, interval=0.05)
    except IndexHeaderError:
        # Shape failure — propagate untouched so it is never reported as (or
        # masked by) a timeout.
        raise
    except AssertionError as exc:
        raise AssertionError(
            f'index not OPERATIONAL within {timeout_s}s '
            f'(last statuses={last_statuses!r})'
        ) from exc
