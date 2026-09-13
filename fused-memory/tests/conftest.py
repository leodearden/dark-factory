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


#: The canonical fused-memory config, by ABSOLUTE path, for the autouse
#: ``_isolate_fm_config`` pin below.  Derived from this file's location and
#: NEVER from ``Path.cwd()`` — a CWD-derived path would reintroduce the exact
#: leak the pin exists to close.
FM_CONFIG_PATH = Path(_tests_dir).parent / 'config' / 'config.yaml'

#: The top-level fields ``FusedMemoryConfig`` exposes to the environment.
#: DERIVED from the model, never listed by hand: a field added later is then
#: covered with no edit here, which is the drift that let this leak class
#: survive in one subproject after being fixed in the other.
_FM_CONFIG_FIELDS = frozenset(FusedMemoryConfig.model_fields)


def _reads_as_a_config_override(env_name):
    """Whether pydantic-settings would read *env_name* as a config field.

    ``env_prefix=''`` makes the WHOLE name a top-level field name, and
    ``env_nested_delimiter='__'`` makes everything before the first ``__`` the
    top-level field of a nested override.  Both are matched case-insensitively
    because the model sets ``case_sensitive=False``.

    Nothing else matches: ``PATH`` is not ``path_scope_adjudicator``, and
    ``FOO__TASKMASTER`` has the head ``foo``.  That narrowness is deliberate —
    unrelated fixtures and the uv/venv machinery legitimately need the ambient
    environment, so a blanket clear would trade one nondeterminism for a worse
    one.
    """
    lowered = env_name.lower()
    return lowered in _FM_CONFIG_FIELDS or lowered.split('__', 1)[0] in _FM_CONFIG_FIELDS


@pytest.fixture(autouse=True)
def _isolate_fm_config(monkeypatch):
    """Pin ``CONFIG_PATH`` at the canonical config so config resolution does
    not depend on the process CWD (task 5444).

    ``FusedMemoryConfig`` is a pydantic-settings ``BaseSettings``, not a plain
    ``BaseModel``: ``fused_memory.config.schema::FusedMemoryConfig.settings_customise_sources``
    reads ``CONFIG_PATH`` with a default of the RELATIVE ``config/config.yaml``,
    and ``fused_memory.config.schema::YamlSettingsSource.__call__`` returns
    ``{}`` — silently — when that path does not exist.

    WITHOUT THIS PIN the YAML layer is present only when pytest happens to run
    from ``fused-memory/``.  Run the identical commit from the repo root and
    every field a test does not pass explicitly drops to its code default; the
    tracked ``taskmaster:`` section disappears and ``config.taskmaster``
    becomes ``None``.  That is not hypothetical: it is why
    ``test_referent_repair.py``'s taskmaster test read as green for both
    registered verify commands (both ``cd`` into ``fused-memory/``) and red for
    a human running it from the root — a phantom main-red that cost an
    investigation and produced a fix for a test that was never broken.  See
    ``plans/fused-memory-config-cwd-leak-rca-2026-09-13.md``.

    Pinning the CANONICAL file rather than an absent one reproduces the
    CWD=``fused-memory/`` semantics every currently-green test was written
    against, so the pin changes no test's meaning — it only makes the result
    the same from everywhere.  Tests that want pure schema defaults opt into
    ``code_default_config`` below.

    Mirrors ``orchestrator/tests/conftest.py::_isolate_orch_config``, whose
    docstring records the same reasoning for ``ORCH_CONFIG_PATH`` ("the
    absolute path is also CWD-independent, so the config no longer depends on
    running from ``orchestrator/``").  That hardening was applied
    subproject-locally and never propagated; this is the propagation.

    ``monkeypatch.setenv`` restores the pre-existing value at teardown, so this
    SUBSUMES the ``preserve_config_path`` fixture it replaced — one fixture
    owning ``CONFIG_PATH`` rather than two with no defined ordering between
    them.  A test that sets ``CONFIG_PATH`` itself still wins: a
    function-scoped ``monkeypatch.setenv`` in the test body runs after this
    autouse fixture.  That same ordering is why the scrub below cannot — and
    must not — stop a test setting ``SERVER__PORT`` deliberately; it removes
    only what pytest INHERITED.

    THE ENVIRONMENT HALF.  Pinning the file closes only half the leak.  The
    model sets ``env_prefix=''``, so a bare ambient variable named after any
    top-level field is an unprefixed override that outranks the YAML.
    Measured: ``TASKMASTER='{"project_root": "/pwned-by-env"}'`` rewrites
    ``config.taskmaster.project_root`` even with ``CONFIG_PATH`` pointing at a
    missing file, so whatever the shell, the CI runner or a parent process
    happens to export decides what a test reads.  The names are DERIVED from
    the model rather than listed, mirroring
    ``orchestrator/src/orchestrator/verify.py``'s reason for scrubbing the
    whole ``ORCH_`` prefix: so a variable added later cannot reintroduce the
    class.  The comprehension snapshots the names before the loop deletes any,
    since mutating ``os.environ`` while iterating it raises.
    """
    for inherited in [name for name in os.environ if _reads_as_a_config_override(name)]:
        monkeypatch.delenv(inherited, raising=False)

    monkeypatch.setenv('CONFIG_PATH', str(FM_CONFIG_PATH))


@pytest.fixture
def code_default_config(monkeypatch, tmp_path):
    """Resolve ``FusedMemoryConfig()`` from the SCHEMA alone, on request.

    Opt-in counterpart to the autouse ``_isolate_fm_config`` above.  That
    fixture deliberately keeps the tracked ``fused-memory/config/config.yaml``
    loaded, because that is what every currently-green test was written
    against — but it therefore leaves no way to ask what a field's CODE
    default is.  Pointing ``CONFIG_PATH`` at a guaranteed-absent file makes
    ``fused_memory.config.schema::YamlSettingsSource.__call__`` skip the YAML
    layer (its ``.exists()`` is False), so only the schema's own defaults
    remain.

    Request it explicitly, or via ``@pytest.mark.usefixtures``; NEVER autouse.
    It runs after the autouse pin and deliberately overrides it, so making it
    autouse would strip the YAML from the whole suite.

    THE MEASUREMENT THAT SIZES BOTH FIXTURES — taken at eb04f1d1c8, the whole
    suite with the YAML layer removed (``FM_CONFIG_PATH`` temporarily
    re-pointed at an absent file, less the two assertions in this branch that
    exist to pin the file's PRESENCE and so cannot survive its removal):
    ``1 failed, 19783 passed, 3 skipped`` in 265.69s.  The single failure is
    ``test_referent_repair.py::TestTheStormGateProjectRoot::test_the_taskmaster_project_root_is_never_used_as_a_fallback``
    — the trap this task exists because of.  Exactly one test in ~19.8k reads
    a value that only the ambient file supplies, which is why a CWD-dependent
    config could sit under this suite unnoticed.  Mirrors
    ``orchestrator/tests/conftest.py::code_default_config``, whose absent-file
    trick this copies.
    """
    monkeypatch.setenv('CONFIG_PATH', str(tmp_path / 'no-such-config.yaml'))


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

    Task 4412. For WHY an unclosed openai/anthropic client emits a ``Task
    exception was never retrieved`` ERROR into some unrelated test's ``caplog``
    window, and why closing it at teardown makes that record unreachable, see
    the ONE canonical write-up of the mechanism —
    ``_fm_helpers.reap_leaked_async_httpx_clients``. It is deliberately not
    restated here.

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

    Task 4412 — mechanism in ``_fm_helpers.reap_leaked_async_httpx_clients``,
    same as the sync arm above.

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
    """A FusedMemoryConfig that doesn't require real API keys or services.

    NOT hermetic, despite reading that way.  ``FusedMemoryConfig`` is a
    pydantic-settings ``BaseSettings``: only the sections passed explicitly
    below are fixed here.  Every other field — ``taskmaster`` among them —
    comes from the YAML that ``_isolate_fm_config`` pins and from the
    environment (the model sets ``env_prefix=''``).  A test asserting on a
    field this factory does not name is asserting on the tracked
    ``fused-memory/config/config.yaml``, not on a code default.
    """
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


# ---------------------------------------------------------------------------
# The integration lane's in-use collection lease (task 4775)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _integration_collection_lease(request):
    """Hold an in-use lease for the duration of every ``integration`` test.

    ``scripts/cleanup_test_collections.py`` runs from cron every six hours
    and deletes every collection under ``PREFIXES`` unconditionally.  A live
    integration test seeds under one of those prefixes, so an unguarded
    sweep landing between its seed and its assertions empties the corpus out
    from under it — and leaves nothing behind pointing at the reaper.

    MARKER-KEYED AND AUTOUSE, rather than a module-level
    ``pytest.mark.usefixtures``.  A module-level opt-in could not reach two
    of the three modules that need it: ``test_memory_eval_retrieval_probe.py``
    and ``test_memory_eval_staleness_sweep.py`` deliberately mark
    ``integration`` PER-TEST, because ``addopts = -m 'not integration'``
    means a module-level mark would deselect the ~170 pure tests each of
    them carries (both files say so in a comment).  More importantly, ANY
    per-module opt-in is a thing a future integration module can forget, and
    silently forgetting the guard is exactly the failure this exists to
    prevent — the same argument the reaper's own docstring makes about a
    prefix coined in one file and reaped by a constant in another.

    Costs an unmarked test nothing: it yields immediately, loads no script
    and creates no directory, so the merge lane (which runs under
    ``-m 'not integration'``) never writes into the machine-global lease
    directory the live cron reads.

    The reaper is loaded LAZILY here rather than at conftest import time, for
    the reason ``_fm_helpers.qdrant_skipif`` documents for deferring its own
    probe: a session that never runs an integration test must not pay for
    it.  Via ``load_script_module`` — do not add another local
    ``spec_from_file_location`` loader (tasks 3738/3895).
    """
    if request.node.get_closest_marker('integration') is None:
        yield
        return

    from _fm_helpers import load_script_module  # noqa: PLC0415

    reaper = load_script_module(
        Path(__file__).parent.parent / 'scripts' / 'cleanup_test_collections.py',
    )
    with reaper.hold_lease(owner=request.node.nodeid):
        yield
