"""pytest fixtures for fused-memory tests.

Non-fixture helpers (MockEdge, make_rebuild_detail, extract_cypher, …)
live in `_fm_helpers.py` — a uniquely-named sibling module — so they can
be imported from test files without conflicting with sibling subprojects'
conftests under `sys.modules['conftest']`.

Testing a `scripts/` script? `scripts/` is not a package and is not on
PYTHONPATH, so import it with `from _fm_helpers import
load_script_module` rather than writing another local
`spec_from_file_location` loader: the shared one reuses an already-loaded
module for the same file instead of re-executing it under the same
`sys.modules` key, and refuses to shadow a module it did not install.
Many older test modules still carry their own copy (task 3895 migrates
them); don't add one (task 3738).
"""

import asyncio
import os
import re
import sys
import warnings
import weakref
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

# Make this file's directory importable by test modules so
# `from _fm_helpers import ...` resolves regardless of whether pytest
# is invoked from the subproject root or the workspace root.
_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

# Insert this worktree's shared/src at the front of sys.path so that
# `import shared` loads the local (possibly modified) code rather than
# whatever editable install the uv workspace has pinned to the main tree
# (mirrors orchestrator/tests/conftest.py's _SHARED_SRC insertion).
_shared_src = os.path.join(
    os.path.dirname(os.path.dirname(_tests_dir)),  # workspace root
    'shared', 'src',
)
if _shared_src not in sys.path:
    sys.path.insert(0, _shared_src)

# Make the sibling 'escalation' workspace package importable without installing it.
# curator_escalator.py uses a try/except guard (HAS_ESCALATION) — adding the src
# path here (before test files are collected) ensures the guard resolves to True so
# tests that exercise the escalation-routing branch can actually run.
_escalation_src = os.path.join(
    os.path.dirname(os.path.dirname(_tests_dir)),  # workspace root
    'escalation', 'src',
)
if _escalation_src not in sys.path:
    sys.path.insert(0, _escalation_src)

# Suite-wide git isolation (task 3355, incident esc-3072-3).  The verify lane
# runs `cd fused-memory && uv run pytest tests/`, which makes rootdir the
# SUBPROJECT — the repo-root conftest.py is never loaded, so each test-root
# conftest wires the defence itself.  APPEND the repo root, never insert(0, ...):
# at sys.path[0] it would make the subproject directories resolve as namespace
# packages pointing at the project folder instead of src/<pkg>/, beating the
# inserts above.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from _fm_helpers import (  # noqa: E402
    _leaked_async_httpx_clients,
    _warn_if_drain_closed_a_foreign_client,
    pydantic_spec,
    reap_leaked_async_httpx_clients,
    reap_leaked_ticket_workers,
    resolve_xdist_worker_id,
    track_async_httpx_clients,
)
from df_pytest_isolation import (  # noqa: E402
    _df_deploy_clocks_unwritten,  # noqa: F401  — the binding IS the wiring
    _df_git_ceiling_at_basetemp,  # noqa: F401  — the binding IS the wiring
    _df_git_env_hermetic,  # noqa: F401  — the binding IS the wiring
    reject_unsafe_basetemp,
)

from fused_memory.backends.graphiti_client import GraphitiBackend  # noqa: E402
from fused_memory.config.schema import (  # noqa: E402
    EmbedderConfig,
    EmbedderProvidersConfig,
    FusedMemoryConfig,
    LLMConfig,
    LLMProvidersConfig,
    OpenAIProviderConfig,
    QueueConfig,
    RoutingConfig,
)


def pytest_configure(config):
    """Session-start hooks: basetemp safety, and async-httpx leak tracking.

    ``track_async_httpx_clients()`` runs HERE — at session start, before
    collection — rather than lazily from a fixture, so clients constructed at
    module import time are tracked too and can be reaped like any other
    (task 4412).
    """
    reject_unsafe_basetemp(config)
    track_async_httpx_clients()


@pytest.fixture(scope='session')
def worker_id(request) -> str:
    """Per-worker id ('gw0', 'gw1', … or 'master') — supplied HERE, not only by xdist.

    This deliberately SHADOWS pytest-xdist's own `worker_id` fixture
    (xdist/plugin.py) for every fused-memory test.  Motivating caller: the
    offline-deep lane's serial confirm re-run, which appends
    `-p no:xdist -o addopts=` (orchestrator/src/orchestrator/verify_cmd.py).
    `-p no:xdist` unregisters the plugin along with its FIXTURES — not just its
    `-n`/`--dist` CLI options — so every test requesting `worker_id` ERRORED at
    setup with `fixture 'worker_id' not found`, and a developer typing
    `pytest -p no:xdist` locally hit the same wall.

    Shadowing is safe precisely because `resolve_xdist_worker_id` delegates to
    xdist's own `get_xdist_worker_id`: under a healthy `-n auto` run the value
    returned here is produced by xdist's own function and is therefore
    identical, while under `-p no:xdist` this is the only provider left.

    `scope='session'` MATCHES xdist's own scope.  Every current consumer is
    function-scoped, so any scope would work today — but pytest forbids a
    broader-scoped fixture depending on a narrower one, so a function-scoped
    shim would ScopeMismatch the first time a session- or module-scoped fixture
    requested `worker_id`.  Not autouse: consumers request it explicitly.
    """
    return resolve_xdist_worker_id(request)


@pytest.fixture(autouse=True)
def preserve_config_path():
    """Save and restore os.environ['CONFIG_PATH'] around every test.

    This is a safety net: if a test (or the code under test) modifies CONFIG_PATH,
    it won't leak into subsequent tests.  The fixture is autouse so all tests in this
    package are covered without needing to request it explicitly.
    """
    original = os.environ.get('CONFIG_PATH')
    yield
    if original is None:
        os.environ.pop('CONFIG_PATH', None)
    else:
        os.environ['CONFIG_PATH'] = original


@pytest_asyncio.fixture(autouse=True)
async def _reap_leaked_ticket_workers():
    """Drain any orphaned TaskInterceptor._curator_worker task at every
    test's teardown boundary.

    Runs while the per-test event loop is still open, so a worker leaked by
    a test that raised before its own cleanup (or by interceptor_with_store's
    teardown, which only cancels workers it can enumerate) cannot be
    destroyed-while-pending under a closing loop and surface as an
    order/xdist-dependent flake in a later, unrelated test (task 1907
    precedent for MergeWorker; see reap_leaked_ticket_workers in
    _fm_helpers.py for the _curator_worker case — task 2737). Cheap no-op
    for tests that leak nothing.
    """
    yield
    await reap_leaked_ticket_workers()


# ---------------------------------------------------------------------------
# Leaked async httpx client drain (task 4412) — TWO autouse arms.
#
# TEARDOWN ORDER IS LOAD-BEARING, and it is bought by an explicit FIXTURE
# DEPENDENCY, not by declaration order: the async arm REQUESTS the sync arm, so
# the sync arm is set up first and therefore torn down LAST, leaving the async
# arm to tear down FIRST. That matters because an async test's leaked clients
# must be closed inside that test's OWN still-open event loop — their
# connection pool has affinity to it.
#
# Declaration order does NOT decide this, despite the usual same-scope rule:
# pytest-asyncio 1.x async fixtures acquire an event-loop dependency that
# reorders them relative to plain autouse fixtures. MEASURED on the pinned
# toolchain before the dependency below was added: the sync arm tore down
# FIRST, the exact inverse of what declaration order predicts, and the
# real-I/O cohort's clients were aclose()d cross-loop from the sync arm's
# throwaway asyncio.run loop.
#
# Pinned behaviourally (not as source layout) by
# test_async_httpx_leak_isolation.py's
# test_aaa_leaked_client_records_its_closing_loop /
# test_aab_the_leak_was_closed_in_its_own_test_loop pair, which records which
# loop actually did the closing.
#
# Measured against: python 3.13.9, pytest 9.0.3, pytest-asyncio 1.3.0
# (asyncio_mode=strict), httpx 0.28.1, openai 2.31.0, anthropic 0.92.0. The
# ordering is a property of pytest-asyncio's fixture graph, so a bump to any of
# those is exactly when that pair should be re-run.
#
# ===========================================================================
# CONSTRAINT: NEVER build an async openai/anthropic client in a fixture scoped
# WIDER than `function` (module / package / session), or in a module-level
# cache a test then drops.
# ===========================================================================
# The drain has no notion of ownership or age: at EVERY test's teardown it
# closes every tracked client that is still open and resurrect-capable,
# whoever built it. A module- or session-scoped fixture's client would
# therefore be closed at the FIRST test's teardown, and its owner would fail
# much later with `Cannot send a request, as the client has been closed` — in
# an apparently unrelated test, which is the exact "blamed on an innocent
# test" shape this drain exists to REMOVE. Measured at time of writing: no
# fused-memory fixture holds such a client (the only module-scoped ones parse
# JSON fixtures), so the constraint costs nothing today.
#
# Scoping the reap to clients created during the current test was considered
# and REJECTED: it opens the inverse hole. A wider-scoped fixture's client
# dropped at ITS OWN teardown is then never drained, gets GC-finalised while
# some later test's loop is running, and resurrects as
# create_task(self.aclose()) — precisely the flake this task retires. Trading
# a real defence for a hypothetical one is the wrong way round, so the
# constraint is enforced by DISCOVERABILITY instead: the sync arm snapshots
# the already-leaked clients at SETUP and warns via
# _fm_helpers._warn_if_drain_closed_a_foreign_client whenever the drain closed
# one that pre-dated the current test — naming the trap at the teardown that
# sprang it, instead of leaving it to be diagnosed from the far-away symptom.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reap_leaked_async_httpx_clients_sync():
    """Close leaked openai/anthropic httpx clients left behind by a SYNC test.

    Task 4412. ``openai``/``anthropic`` ``AsyncHttpxClientWrapper`` defines a
    ``__del__`` that, when the object is GC-finalised while some event loop is
    running, RESURRECTS it as ``create_task(self.aclose())``. That task's
    ``aclose()`` hits a pool bound to an already-closed loop, raises
    ``RuntimeError('Event loop is closed')``, and — since nobody retrieves it —
    ``Task.__del__`` logs ``Task exception was never retrieved`` at ERROR on the
    root ``asyncio`` logger, landing in whichever unrelated test's ``caplog``
    window is open. Closing the client first makes ``__del__`` short-circuit on
    ``if self.is_closed: return``, so the record can never be emitted.

    THIS ARM IS DEFENCE IN DEPTH, NOT COVERAGE OF A GAP THE ASYNC ARM LEAVES.
    Measured on the pinned toolchain (with a spy installed at collection time,
    so it survives function-scoped teardown unlike a ``monkeypatch`` one): the
    ``pytest_asyncio`` arm below DOES fire for a plain ``def test_``, in a
    throwaway loop of its own. So this arm is not what makes the sync cohort —
    ``test_graphiti_llm_client_construction.py``'s ``test_returns_openai_client``,
    16 of 40 measured leaks — get drained. What it buys is a cheap,
    version-independent backstop that does not depend on pytest-asyncio's
    fixture graph continuing to behave that way across a bump, and that still
    runs if the async arm is skipped or errors. Sibling precedents:
    ``_reap_leaked_ticket_workers`` above (task 2737) and
    ``orchestrator/tests/conftest.py``'s ``_reap_leaked_aiosqlite_connections``
    (task 2413).

    Runs LAST (the async arm requests it, so it is set up first), by which
    point an async test's clients are already closed and this is a no-op. The
    empty-check comes first so the ~14100-of-14147 tests that leak nothing pay
    only a WeakSet scan and never spin up an event loop. A client that reaches
    this arm still open never had its pool exercised on a live loop, so a
    throwaway ``asyncio.run`` loop closes it correctly.

    THE ``asyncio.run`` IS GUARDED, because this is the arm that can close a
    client CROSS-LOOP (its throwaway loop is not the one the client's pool was
    opened on). ``reap_leaked_async_httpx_clients`` suppresses
    ``asyncio.TimeoutError`` and ``RuntimeError``, but a foreign-loop
    ``aclose()`` on an anyio-backed pool can surface other types — and this is
    a fallback path measured as never firing under the shipped ordering, so a
    regression in it would otherwise be discovered by it erroring some
    innocent test's teardown. A warning keeps it loud without letting the
    BACKSTOP arm become a flake source itself.
    """
    # Snapshot at SETUP (this arm is set up first, before the test body runs):
    # the clients already leaked by someone else. Weak refs, so the snapshot
    # never keeps a client alive or changes its GC timing.
    preexisting = weakref.WeakSet(_leaked_async_httpx_clients())
    yield
    if _leaked_async_httpx_clients():
        try:
            asyncio.run(reap_leaked_async_httpx_clients())
        except Exception as exc:  # noqa: BLE001 — a backstop must not fail the test
            warnings.warn(
                f'async-httpx drain fallback failed: {exc!r}. The sync arm '
                f'closes clients from a throwaway asyncio.run loop, so an '
                f'exception type beyond the suppressed TimeoutError/'
                f'RuntimeError reached it — widen the suppression in '
                f'_fm_helpers.reap_leaked_async_httpx_clients or fix the '
                f'ordering that sent this client to the fallback (task 4412).',
                stacklevel=2,
            )
    _warn_if_drain_closed_a_foreign_client(preexisting)


@pytest_asyncio.fixture(autouse=True)
async def _reap_leaked_async_httpx_clients(_reap_leaked_async_httpx_clients_sync):
    """Close leaked openai/anthropic httpx clients before this test's loop closes.

    Task 4412 — same mechanism as the sync arm above.

    THE ARGUMENT IS THE ORDERING, and it is the only reason it is there.
    Requesting the sync arm forces the sync arm to be SET UP first and so torn
    down LAST, which gives this arm teardown PRIORITY: an async test's clients
    are then closed inside that test's OWN still-open event loop, the
    correct-affinity path for ``test_local_endpoint_base_url_integration.py``,
    the only measured cohort that performs real I/O. Declaration order alone
    does not achieve this — see the block comment above for the measurement
    that showed the shipped order was the inverse — so do not "tidy" this
    parameter away. It is pinned by
    ``test_async_httpx_leak_isolation.py::test_aab_the_leak_was_closed_in_its_own_test_loop``.

    Best-effort and bounded — see ``reap_leaked_async_httpx_clients`` in
    ``_fm_helpers.py``: it never fails a test, and is a cheap no-op for the
    vast majority of tests that leak nothing.
    """
    yield
    await reap_leaked_async_httpx_clients()


@pytest.fixture
def standard_mock_config() -> MagicMock:
    """MagicMock config pre-configured with common 1536-dim embedder attributes.

    Used by run_* entrypoint tests (TestRunReindex, TestRunCleanup, etc.) that
    need a config mock but don't want to construct a full FusedMemoryConfig.
    Tests needing non-default values (e.g., 768-dim) can override in-place:

        def test_something(self, standard_mock_config):
            standard_mock_config.embedder.dimensions = 768

    Note: spec_set only constrains top-level attributes; nested attribute typos
    (e.g. cfg.embedder.dimensionz) are still silently accepted because cfg.embedder
    resolves to an unconstrained child MagicMock.
    """
    cfg = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
    cfg.embedder.dimensions = 1536
    cfg.embedder.providers.openai = None
    cfg.embedder.model = 'text-embedding-3-small'
    return cfg


@pytest.fixture
def make_backend():
    """Factory fixture: returns a callable(config) -> GraphitiBackend with mock client."""
    def _factory(config) -> GraphitiBackend:
        backend = GraphitiBackend(config)
        backend.client = MagicMock()
        backend._driver = MagicMock()
        return backend

    return _factory


@pytest.fixture
def make_graph_mock():
    """Factory fixture: returns a callable(rows, *, ro_rows, q_rows, header) -> MagicMock graph.

    The returned mock has both .query and .ro_query as AsyncMocks.

    ``header`` sets ``result.header`` on every returned result object, and
    defaults to ``[]`` rather than to the auto-``MagicMock`` attribute a bare
    ``MagicMock()`` would otherwise supply.  Code that resolves FalkorDB result
    columns BY NAME (``GraphitiBackend.list_indices``, and
    ``_fm_helpers.await_index_operational`` before it) iterates
    ``result.header``, and an auto-``MagicMock`` is not iterable — every mocked
    call would raise ``TypeError`` instead of exercising the code under test.
    The ``[]`` default is safe because no existing consumer of this fixture
    reads ``.header``; a by-name consumer must pass one explicitly.

    Header values are the measured live 2-tuples, e.g. (task 3706, measured
    2026-08-06 via ``GRAPH.RO_QUERY dark_factory "CALL db.indexes()"``)::

        [[1, 'label'], [1, 'properties'], [1, 'types'], [1, 'options'],
         [1, 'language'], [1, 'stopwords'], [1, 'entitytype'], [1, 'status'],
         [1, 'info']]

    CYPHER DISPATCH (task 4340).  Both mocks answer per the cypher they are
    given, rather than returning one static result for everything:

      - a cypher containing ``count(``      -> ``[[len(rows)]]``, a single row
      - a cypher containing ``SKIP n LIMIT m`` -> ``rows[n : n + m]``
      - anything else                       -> ``rows``, exactly as before

    This exists because two whole-graph reads are now paginated, and a
    paginated read issues a single-row ``count(*)`` census probe before its
    SKIP/LIMIT pages.  A static fixture would answer that census with a page
    of edge rows, whose first column is a uuid string — ``int('node-1')``
    raises, the count is unusable, and every caller would silently flip to
    ``complete=False`` plus a WARNING.  A shared fixture that lies about the
    read shape it stands in for is worse than a per-test patch: the next
    person to paginate something rediscovers the same trap.

    This fixture deliberately does NOT simulate the server's
    ``RESULTSET_SIZE`` truncation.  ONE double owns that behaviour —
    ``test_graph_read_pagination.FakeCappedGraph``, which also carries the
    stateful query log the truncation tests need — because two doubles that
    both claim to stand in for the same server drift, and the drift shows up
    as a test that passes against a fake nothing else agrees with.  A test
    that needs the cap should use that one.
    """
    skip_limit_re = re.compile(r'SKIP\s+(\d+)\s+LIMIT\s+(\d+)', re.IGNORECASE)
    # Deliberately NARROW: only a query whose entire projection is a bare row
    # count is a census probe. A loose `'count(' in cypher` test also captures
    # ordinary queries that return a count as one column among several — e.g.
    # find_duplicate_entity_nodes' `RETURN n.uuid, ..., count(e)` — and would
    # hand them a single-column [[n]] row, raising IndexError deep inside the
    # method under test rather than anywhere near the fixture.
    census_re = re.compile(r'RETURN\s+count\(\*\)\s*$', re.IGNORECASE)

    def _factory(
        rows: list[list] | None = None,
        *,
        ro_rows: list[list] | None = None,
        q_rows: list[list] | None = None,
        header: list | None = None,
    ) -> MagicMock:
        header_value = header if header is not None else []
        if ro_rows is not None or q_rows is not None:
            ro_row_data = ro_rows if ro_rows is not None else (rows or [])
            q_row_data = q_rows if q_rows is not None else (rows or [])
        else:
            ro_row_data = rows if rows is not None else []
            q_row_data = ro_row_data

        def _make_side_effect(row_data: list[list]):
            def _respond(cypher='', params=None, *args, **kwargs) -> MagicMock:
                text = cypher if isinstance(cypher, str) else ''
                if census_re.search(text.strip()):
                    # A single-row aggregate: never truncated by the row cap,
                    # and it agrees with the pages by construction.
                    result_set = [[len(row_data)]]
                else:
                    match = skip_limit_re.search(text)
                    if match:
                        skip, limit = int(match.group(1)), int(match.group(2))
                        result_set = row_data[skip: skip + limit]
                    else:
                        result_set = row_data
                result = MagicMock()
                result.result_set = result_set
                result.header = header_value
                return result

            return _respond

        graph_mock = MagicMock()
        graph_mock.query = AsyncMock(side_effect=_make_side_effect(q_row_data))
        graph_mock.ro_query = AsyncMock(side_effect=_make_side_effect(ro_row_data))
        return graph_mock

    return _factory


@pytest.fixture
def make_fake_maintenance_service():
    """Factory fixture: returns a callable(mock_cfg, mock_service) -> async context manager."""
    def _factory(mock_cfg, mock_service):
        @asynccontextmanager
        async def fake(config_path):
            yield mock_cfg, mock_service

        return fake

    return _factory


@pytest.fixture
def make_edge_backend():
    """Factory fixture: returns a callable(backend, *, nodes, edges) -> backend."""
    def _factory(backend, *, nodes, edges):
        backend.list_entity_nodes = AsyncMock(return_value=nodes)
        backend.get_all_valid_edges = AsyncMock(return_value=edges)
        return backend

    return _factory


@pytest.fixture
def mock_config(tmp_path) -> FusedMemoryConfig:
    """A FusedMemoryConfig that doesn't require real API keys or services."""
    return FusedMemoryConfig(
        llm=LLMConfig(
            provider='openai',
            model='gpt-4o-mini',
            providers=LLMProvidersConfig(
                openai=OpenAIProviderConfig(api_key='test-key'),
            ),
        ),
        embedder=EmbedderConfig(
            provider='openai',
            model='text-embedding-3-small',
            providers=EmbedderProvidersConfig(
                openai=OpenAIProviderConfig(api_key='test-key'),
            ),
        ),
        routing=RoutingConfig(
            use_heuristics=True,
            llm_fallback=False,
            confidence_threshold=0.7,
        ),
        queue=QueueConfig(
            semaphore_limit=5,
            workers_per_group=2,
            max_attempts=3,
            retry_base_seconds=0.05,
            write_timeout_seconds=2.0,
            data_dir=str(tmp_path / 'queue'),
        ),
    )
