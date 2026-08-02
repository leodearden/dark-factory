"""pytest configuration — ensure local src takes precedence over installed package.

Non-fixture helpers (build_usage_gate) live in `_orch_helpers.py` — a
uniquely-named sibling module — so they can be imported from test files
without conflicting with sibling subprojects' conftests under
`sys.modules['conftest']`.
"""
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

# Insert this worktree's src directories at the front of sys.path so that
# `import orchestrator` and `import shared` load the local (possibly modified)
# code rather than whatever editable install the uv workspace has pinned to
# the main tree.
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SHARED_SRC = Path(__file__).parent.parent.parent / "shared" / "src"
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))
_ESCALATION_SRC = Path(__file__).parent.parent.parent / "escalation" / "src"
if str(_ESCALATION_SRC) not in sys.path:
    sys.path.insert(0, str(_ESCALATION_SRC))
_TESTS_DIR = Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# Repo root of this worktree.  Used by the autouse ``_isolate_orch_config``
# fixture to pin ``ORCH_CONFIG_PATH`` at the canonical top-level
# ``dark-factory-orchestrator.yaml`` (task 2719: the transitional
# ``orchestrator/config.yaml`` symlink was retired, so the old CWD-relative
# ``Path('config.yaml')`` fallback no longer resolves to the operational config).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Suite-wide git isolation (task 3355, incident esc-3072-3).  The verify lane
# runs `cd orchestrator && uv run pytest tests/`, which makes rootdir the
# SUBPROJECT — the repo-root conftest.py is never loaded, so this suite wires
# the defence itself.  APPEND the repo root, never insert(0, ...): at sys.path[0]
# it would make orchestrator/, shared/ etc. resolve as namespace packages
# pointing at the project folder instead of src/<pkg>/, and the src dirs
# inserted above would lose.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Suite-wide single-writer debug asserts (task 1999 / MQ-invariants ξ, I7).
# Must be set BEFORE any `orchestrator.merge_queue` import so the module-level
# `_DEBUG_ASSERTS = os.environ.get(...)` seed picks it up.
os.environ.setdefault('ORCH_DEBUG_ASSERTS', '1')

from _orch_helpers import (  # noqa: E402
    drain_async_mock_coroutines,
    idle_psi_sample,
    pydantic_spec,
    reap_leaked_aiosqlite_connections,
    reap_leaked_claimant_heartbeats,
)
from df_pytest_isolation import (  # noqa: E402
    _df_git_ceiling_at_basetemp,  # noqa: F401  — the binding IS the wiring
    reject_unsafe_basetemp,
)
from shared.config_models import UsageCapConfig  # noqa: E402

from orchestrator import merge_queue  # noqa: E402
from orchestrator.config import (  # noqa: E402
    EscalationConfig,
    FusedMemoryConfig,
    GitConfig,
    OrchestratorConfig,
    ReviewConfig,
    SandboxConfig,
)

# Belt-and-braces direct assignment: defeats any import-order race where
# orchestrator.merge_queue was imported (by another conftest/plugin) before
# the os.environ.setdefault above took effect, which would have frozen its
# module-level _DEBUG_ASSERTS seed at False.
merge_queue._DEBUG_ASSERTS = True


@pytest_asyncio.fixture(autouse=True)
async def _reap_leaked_merge_workers():
    """Gracefully stop any MergeWorker orphaned onto the test event loop (task 1907).

    A merge-queue test that raises before its own ``await worker.stop()`` (e.g. an
    assertion fails partway through) leaks the worker's ``run()`` task and its
    four background loops, which do real ``git`` subprocess work. If
    pytest-asyncio's per-test loop teardown (``asyncio.runners._cancel_all_tasks``)
    then cancels a loop caught mid-subprocess-spawn
    (``BaseSubprocessTransport._connect_pipes``), the cancellation ``gather``
    deadlocks and the whole ``pytest tests/`` process HANGS forever at teardown
    (this is the remaining full-suite teardown stall once the worker-kill hang is
    fixed; there are 100+ ``create_task(worker.run())`` sites with inline-only
    cleanup, so per-test ``try/finally`` is not tractable).

    Reaping here — in the test's own loop, before it is closed — via the graceful
    ``worker.stop()`` (sets ``_running=False`` + sends sentinels + bounded drain)
    lets each loop FINISH its in-flight subprocess and exit cleanly, instead of
    being abruptly cancelled mid-spawn. Best-effort and bounded: it never fails a
    test and is a cheap no-op for the (vast majority of) tests that leak nothing.

    Works for sync and async tests alike: pytest-asyncio (strict mode) provides a
    loop for this async fixture even under a sync test, where ``all_tasks()`` is
    simply empty.
    """
    yield
    import asyncio
    import contextlib

    for task in list(asyncio.all_tasks()):
        if task.done():
            continue
        coro = task.get_coro()
        if not getattr(coro, "__qualname__", "").endswith("MergeWorker.run"):
            continue
        frame = getattr(coro, "cr_frame", None)
        worker = frame.f_locals.get("self") if frame is not None else None
        if worker is None or not hasattr(worker, "stop"):
            continue
        with contextlib.suppress(BaseException):
            await asyncio.wait_for(worker.stop(), timeout=15.0)
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await asyncio.wait_for(
                    asyncio.gather(task, return_exceptions=True), timeout=15.0
                )


@pytest_asyncio.fixture(autouse=True)
async def _reap_leaked_aiosqlite_connections():
    """Close any leaked aiosqlite connection before the per-test event loop closes.

    Task 2413 — fix for the scheduler-test xdist flake. ``aiosqlite.Connection``
    runs one background worker ``Thread`` per connection. If a test leaks a
    live connection (never calls ``await conn.close()``), that thread outlives
    the test: pytest-asyncio then closes the per-test event loop while the
    thread is still alive, and the next time the thread tries to resolve a
    future via ``future.get_loop().call_soon_threadsafe(...)`` it raises
    ``RuntimeError: Event loop is closed`` from inside the thread. pytest's
    threadexception plugin surfaces this as a
    ``PytestUnhandledThreadExceptionWarning`` (promoted to a hard error by this
    project's ``filterwarnings``) and attributes it to whatever unrelated test
    happens to be running when the thread fires — under ``-n auto`` on an
    oversubscribed host that is reliably a *different*, innocent test.

    Reaping here — in the test's own loop, before it is closed — closes and
    joins any live aiosqlite connection so its worker thread is guaranteed
    dead before this fixture (and therefore the test) returns; it can then
    never touch a closed loop. Best-effort and bounded (see
    ``reap_leaked_aiosqlite_connections`` in ``_orch_helpers.py``): it never
    fails a test and is a cheap no-op for the (vast majority of) tests that
    leak no connection.

    ASSUMPTION: every aiosqlite connection in this suite is function-scoped
    (verified at task-2413 authorship time — no test uses a
    module/session/package/class-scoped aiosqlite or ``CostStore`` fixture).
    This autouse reaper runs after *every* test, so a future higher-scoped
    shared aiosqlite connection meant to persist across tests would be
    force-closed the first time this fixture tears down within that scope,
    producing a confusing "Connection closed" failure in a later test rather
    than in the one that actually introduced the shared fixture. If such a
    fixture is ever added, it will need an explicit opt-out here (or its own
    teardown ordered ahead of this one).
    """
    yield
    await reap_leaked_aiosqlite_connections()


@pytest_asyncio.fixture(autouse=True)
async def _reap_leaked_claimant_heartbeats():
    """Cancel any leaked TaskWorkflow._claimant_heartbeat_loop before the loop closes.

    Task 2780 — fix for the orchestrator merge-queue/workflow xdist load-flake.
    ``TaskWorkflow._setup_worktree_and_artifacts`` starts a background
    ``_claimant_heartbeat_loop`` task; production ``run()``'s finally stops it
    via ``_stop_claimant_heartbeat``. A co-scheduled TaskWorkflow test that
    raises before its own inline ``_stop_claimant_heartbeat`` (or never starts
    one) orphans that loop onto the shared per-worker event loop, where it is
    destroyed-while-pending at teardown — or, under a fully-mocked config with
    a ``MagicMock`` ``claimant_heartbeat_interval_secs``, raises an
    un-retrieved ``TypeError`` from ``asyncio.sleep(<MagicMock>)`` — and
    pytest attributes the fallout to a later, innocent test (observed victim:
    ``TestB7SingleHostNewPath``'s serial-order assertion).

    Reaping here — in the test's own loop, before it is closed — cancels and
    bounded-drains any live heartbeat loop so it can never later touch a
    closed loop or pollute the shared worker. Best-effort and bounded (see
    ``reap_leaked_claimant_heartbeats`` in ``_orch_helpers.py``): it never
    fails a test and is a cheap no-op for the (vast majority of) tests that
    leak no loop. Mirrors the sibling autouse reapers ``_reap_leaked_merge_workers``
    (task 1907) and ``_reap_leaked_aiosqlite_connections`` (task 2413), and the
    fused-memory ``_reap_leaked_ticket_workers`` (task 2737).
    """
    yield
    await reap_leaked_claimant_heartbeats()


@pytest.fixture(scope="session")
def repo_root() -> Path | None:
    """Walk up from this conftest to find the repo root anchored by a .git entry.

    Scoped to orchestrator/tests.  If this fixture is ever needed by other test
    packages (reify, fused-memory, shared, …) it should be hoisted to a
    top-level conftest.py or a shared test-helpers module rather than duplicated.

    Returns the repo-root Path if found, or None when not running inside a git
    checkout (e.g. a packaged wheel, partial mirror, or isolated test run).

    Works for both normal checkouts (.git directory) and git worktrees (.git
    file) because ``Path.exists()`` matches either.

    Consumers should call ``pytest.skip(...)`` when None is returned rather than
    failing — the absence of a git sentinel means the repo-root-dependent tests
    are not applicable to this environment (e.g. CI running from a sdist).
    Consumers may also ``pytest.fail(...)`` when the sentinel is found but a
    required file within the repo is absent, so a genuinely missing file cannot
    silently hide behind a skip.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / '.git').exists():
            return parent
    return None


@pytest.fixture(autouse=True)
def _isolate_orch_config(monkeypatch, tmp_path):
    """Isolate OrchestratorConfig from the real project tree for every test.

    ORCH_CONFIG_PATH is pinned to the canonical top-level
    ``dark-factory-orchestrator.yaml`` (absolute path) so tests load the
    operational config deterministically, independent of the process CWD.

    Config source (task 2719): previously this fixture *deleted*
    ORCH_CONFIG_PATH and relied on ``settings_customise_sources`` falling back
    to the *relative* ``Path('config.yaml')`` — which, under ``cd orchestrator
    && pytest`` (how the per-subproject test command invokes us), resolved via
    the ``orchestrator/config.yaml`` symlink to the operational config.  That
    transitional symlink was retired, so we now pin the absolute canonical path
    explicitly.  It is byte-identical to the removed symlink's target, so the
    operational values every test relies on (e.g. ``lock_depth``,
    ``merge_verify_breadth='full'``) are unchanged; the absolute path is also
    CWD-independent, so the config no longer depends on running from
    ``orchestrator/``.

    project_root isolation (the other load-bearing part): any test that builds a
    bare ``OrchestratorConfig()`` and drives ``acquire_next`` writes
    ``<repo>/data/orchestrator/scheduler_state.json`` (and the overrides SQLite
    DB) into the live tree — which the dashboard reads back as real scheduler
    state.  Pin ``project_root`` (via the ``ORCH_`` env prefix) to this test's
    ``tmp_path`` so those writes land in tmp instead.

    Precedence keeps this safe: ``init_settings`` (explicit ``project_root=...``
    kwargs) still win over the env, and ``env_settings`` only overrides
    ``project_root`` — every other field still loads from config.yaml/defaults,
    so tests that depend on config values (e.g. lock_depth) are unaffected.
    Config-loading tests already use ``tmp_path`` as their project_root, so the
    env value agrees with the YAML they write.  The opt-in
    ``code_default_config`` fixture runs after this autouse fixture and still
    wins (re-pointing ORCH_CONFIG_PATH at a guaranteed-absent file) for the
    default-asserting / scoped-behaviour tests.
    """
    monkeypatch.setenv("ORCH_CONFIG_PATH", str(REPO_ROOT / "dark-factory-orchestrator.yaml"))
    monkeypatch.setenv("ORCH_PROJECT_ROOT", str(tmp_path))


@pytest.fixture
def code_default_config(monkeypatch, tmp_path):
    """Isolate ``OrchestratorConfig()`` from the ambient operational config.yaml.

    Opt-in counterpart to the autouse ``_isolate_orch_config`` above.  That
    fixture DELIBERATELY leaves ``config.yaml`` loading intact (only pinning
    ``project_root``) so the majority of tests keep the tracked
    ``orchestrator/config.yaml``'s operational values (``lock_depth`` etc.).
    But the merge-verify harness runs pytest from the ``orchestrator/`` cwd, so
    ``settings_customise_sources`` (config.py) falls back to the *relative*
    ``Path('config.yaml')`` -> the tracked ``orchestrator/config.yaml``, whose
    live operational overrides (commit b012af0e0e:
    ``merge_verify_breadth='full'``, ``merge_train_former_enabled=true``,
    ``merge_train_coalesce_enabled=true``) then bleed into any bare
    ``OrchestratorConfig()``.

    Tests that assert the CODE DEFAULTS (``defaults.yaml``:
    ``merge_verify_breadth='scoped'``, trains OFF) — or that exercise the
    default *scoped* merge-verify behaviour — must construct the config in
    isolation from those live overrides.  Pointing ``ORCH_CONFIG_PATH`` at a
    guaranteed-absent file makes ``YamlSettingsSource.__call__`` skip the
    layer-2 project file (its ``.exists()`` is False) and load ONLY the
    package-bundled ``defaults.yaml`` -> pure code defaults.  ``project_root``
    still resolves to ``tmp_path`` via ``ORCH_PROJECT_ROOT`` (set by the autouse
    fixture, which runs first), so bare-config tests still never write into the
    live tree.

    Opt-in via ``@pytest.mark.usefixtures("code_default_config")`` (or a direct
    request) on exactly the default-asserting / scoped-behaviour tests — NEVER
    autouse, which would strip config.yaml from the ~10k tests that legitimately
    rely on it.
    """
    monkeypatch.setenv("ORCH_CONFIG_PATH", str(tmp_path / "no-such-config.yaml"))


#: Guaranteed-absent path for the autouse warm-lane script-dir pin below.
#: A fixed literal rather than a ``tmp_path`` child ON PURPOSE: the directory is
#: only ever required NOT to exist and is never written to, so requesting
#: ``tmp_path`` would make pytest allocate a numbered temp dir for every test in
#: the orchestrator suite (thousands, times every xdist worker) purely to have a
#: name to point at.  The only consumer is
#: :meth:`GitOps._resolve_warm_lane_script`, which just calls ``.exists()``.
_ABSENT_WARM_LANE_SCRIPT_DIR = "/nonexistent/df-warm-lane-scripts"


@pytest.fixture(autouse=True)
def _isolate_warm_lane_script_dir(monkeypatch):
    """Pin dark-factory's warm-lane script directory ABSENT for every test.

    Task 3072 (PRD ``warm-lane-infra-repatriation-prd.md`` leaf α) gives
    ``GitOps`` a two-step resolution order for warm-lane scripts:
    ``<project_root>/scripts/<name>`` first, then dark-factory's own copies
    under ``orchestrator/scripts/warm-lane/``.

    That second step is a hazard for the existing suite.  About 200 tests
    across ``test_git_ops.py``, ``test_warm_lane_pool.py``,
    ``test_pool_storage_guard.py``, ``test_warm_lane_disk_guard.py``,
    ``test_warm_lane_soft_floor.py``, ``test_warm_lane_integration_gate.py``,
    ``test_prune_registrations_chokepoint.py`` and ``test_reconcile_stranded.py``
    build a synthetic ``tmp_path`` repo with no ``scripts/`` directory and
    assert the resulting "script absent → fail-soft sentinel" behaviour.  With
    an unconditional repo-relative fallback, every one of those tests would
    begin executing dark-factory's REAL warm-lane scripts (``rm -rf`` on lane
    target dirs, ``flock`` acquisition, ``df`` probes) against ``tmp_path`` —
    non-hermetic and destructive.

    Pinning the fallback root at a guaranteed-absent directory keeps all of
    them byte-identical.  Same shape and same reasoning as the autouse
    ``_isolate_orch_config`` above.

    **Test hermeticity only.**  Production never sets
    ``ORCH_WARM_LANE_SCRIPT_DIR``; the repo-relative default is what actually
    ships, and it is pinned by
    ``test_warm_lane_script_resolution.py::TestResolveWarmLaneScript::
    test_unset_env_falls_back_to_the_repo_relative_directory``.  Tests that
    genuinely exercise resolution opt in via ``df_warm_lane_script_dir`` below.
    """
    monkeypatch.setenv(
        "ORCH_WARM_LANE_SCRIPT_DIR", _ABSENT_WARM_LANE_SCRIPT_DIR,
    )


@pytest.fixture
def df_warm_lane_script_dir(monkeypatch, tmp_path: Path):
    """Opt-in: re-point the dark-factory warm-lane script root at a stub dir.

    Counterpart to the autouse ``_isolate_warm_lane_script_dir`` above, which
    runs first and pins the root absent; this fixture runs after it and wins
    for the tests that DO exercise resolution (task 3072's B7/B8 cases).

    Returns a factory.  Called with no argument it creates and pins a fresh
    ``<tmp_path>/df-warm-lane-scripts`` directory and returns it; called with a
    path it pins that path instead (which need not exist — pinning a
    nonexistent directory is how the "neither location" case is set up)::

        def test_something(df_warm_lane_script_dir):
            df_dir = df_warm_lane_script_dir()
            (df_dir / 'warm-lane-gc.sh').write_text(...)

    **Test hermeticity only** — production resolution is repo-relative and
    reads no environment variable.
    """
    def _pin(path: Path | None = None) -> Path:
        if path is None:
            path = tmp_path / "df-warm-lane-scripts"
            path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("ORCH_WARM_LANE_SCRIPT_DIR", str(path))
        return path

    return _pin


def pytest_configure(config: pytest.Config) -> None:
    """Refuse an unsafe ``--basetemp``, then register ``real_psi_reader`` (task 2418).

    The ``reject_unsafe_basetemp`` call comes FIRST and is unrelated to the
    marker below: it aborts collection when ``--basetemp`` points inside a live
    task worktree, where every git command this suite runs would resolve against
    the enclosing worktree's repo (task 3355, incident esc-3072-3).

    ``orchestrator/pyproject.toml``'s ``[tool.pytest.ini_options] markers``
    list is outside this task's locked scope, so the marker used by the
    ``_hermetic_psi_reader`` fixture below is registered here instead, via
    the standard ``addinivalue_line`` hook. Without one of the two
    registration paths, applying ``@pytest.mark.real_psi_reader`` would
    raise ``PytestUnknownMarkWarning`` under a future ``--strict-markers``
    addopts change, silently defeating the escape hatch — today's
    ``addopts`` doesn't set that flag, so this is precautionary rather than
    load-bearing yet.
    """
    reject_unsafe_basetemp(config)
    config.addinivalue_line(
        'markers',
        'real_psi_reader: opt a test OUT of the autouse `_hermetic_psi_reader` '
        'patch so it exercises the REAL `shared.psi.read_psi_sample` bound at '
        'Scheduler construction instead of the deterministic idle-sample stub. '
        'Default: autouse idle-PSI patch installed in conftest.py (task 2418).',
    )


@pytest.fixture(autouse=True)
def _hermetic_psi_reader(monkeypatch, request):
    """Default every Scheduler's PSI reader to a deterministic idle sample.

    Task 2418 — fix for the scheduler-test xdist flake where the dispatch-
    admission gate (``psi_admission.enabled`` defaults True,
    ``orchestrator.config.PsiAdmissionConfig``) reads REAL
    ``/proc/pressure/*`` via ``self._read_psi_sample()`` once per
    ``acquire_next()`` tick (scheduler.py).  Under host load (e.g. full
    parallel ``pytest -n auto --dist loadgroup``) ``mem_some_avg10`` can
    cross its configured threshold, non-deterministically deferring dispatch
    of non-deterministic candidates in any bare-config test that calls
    ``acquire_next()`` — flipping dispatch/skip/park/holder assertions.

    Monkeypatches the module-level ``orchestrator.scheduler.read_psi_sample``
    — the seam ``Scheduler.__init__`` binds to ``self._read_psi_sample`` by
    default — to ``idle_psi_sample`` (``_orch_helpers.py``), a non-saturating
    PsiSample (all metrics 0.0, ``read_ok=True``).  Every ``Scheduler``
    constructed during a test therefore defaults to a deterministic,
    non-saturating PSI reading regardless of host load. ``monkeypatch``
    auto-reverts at teardown, so production is never touched.

    Saturation tests (``test_scheduler_dispatch_admission.py``,
    ``test_scheduler_psi_saturation_transition.py``) reassign
    ``scheduler._read_psi_sample`` post-construction and are unaffected —
    that per-instance assignment overrides this constructor-time default.

    Mirrors ``_isolate_orch_config`` immediately above: both neutralize a
    host/global dependency (project_root / PSI) so the suite is hermetic
    under parallel execution.  Guard test:
    ``test_scheduler_hermetic_psi.py::TestAutouseFixtureDefaultsToIdlePsi``.

    **Suite-wide, not scheduler-scoped**: this fixture lives in the
    top-level orchestrator conftest (like its siblings above/below —
    ``_isolate_orch_config``, ``_no_dry_run_unblock``, etc.), so it patches
    every test, not just scheduler ones. Narrowing it to an allowlist would
    mean auditing every one of the ~28 files that call ``acquire_next()`` or
    construct a ``Scheduler`` — most of which this task does not hold locks
    for — so it stays broad-by-default with the escape hatches below; a full
    opt-in restructuring remains a follow-up outside this task's locked
    scope.

    **Opt-out via ``real_psi_reader`` marker**: tests that specifically need
    the real ``shared.psi.read_psi_sample`` bound at construction — e.g. to
    assert reader identity, or to exercise genuine ``/proc/pressure/*``
    reads — mark themselves with ``@pytest.mark.real_psi_reader`` to skip
    the monkeypatch, restoring the real binding. Mirrors the
    ``exercise_merge_verify`` opt-out pattern elsewhere in this file.
    Registered via this file's ``pytest_configure`` hook above (not in
    ``orchestrator/pyproject.toml``'s ``markers = [...]`` list, which
    remains outside this task's locked scope), so the marker survives a
    future ``--strict-markers`` addopts change instead of erroring at
    collection. Guard test for this opt-out branch:
    ``test_scheduler_hermetic_psi.py::test_real_psi_reader_marker_restores_real_reader``.
    A test may instead reassign ``scheduler._read_psi_sample``
    post-construction, as the saturation tests do.
    """
    if request.node.get_closest_marker('real_psi_reader'):
        return
    monkeypatch.setattr('orchestrator.scheduler.read_psi_sample', idle_psi_sample)


@pytest.fixture
def forbid_live_mcp(monkeypatch):
    """Opt-in guard: fail the test if a live fused-memory MCP round-trip is attempted.

    Task 2644 — de-flaking harness tests that build a real ``Scheduler`` with
    no injected ``mcp_session`` and no ``McpLifecycle.start()``.  In that
    configuration ``Scheduler.dispatch_tool`` (scheduler.py:1787-1794) falls
    through to the LIVE ``mcp_call(...)`` HTTP branch, which — because the
    module-global ``mcp_lifecycle._session`` is ``None`` in unit tests —
    constructs a one-shot ``McpSession`` and calls ``initialize()`` /
    ``_raw_call()`` against the real ``http://localhost:8002``.  Under
    ``pytest -n auto`` that shared server saturates under load (observed
    ~100x latency), and these multi-call paths can exceed the suite's 60s
    per-test timeout — which (``timeout_method='thread'``) ``os._exit()``s
    the xdist worker rather than just failing the one test.

    Patches the ``McpSession`` CLASS in ``orchestrator.mcp_lifecycle`` —
    NOT ``orchestrator.scheduler.mcp_call`` — because ``scheduler.py`` binds
    ``mcp_call`` via ``from ... import``, a distinct binding from the
    canonical ``orchestrator.mcp_lifecycle.mcp_call`` (and
    ``orchestrator.agents.briefing.mcp_call``).  Every alias, however,
    ultimately constructs/uses an ``McpSession`` and issues its first
    network I/O through ``McpSession.initialize`` -> ``_raw_call`` (the
    one-shot fallback path inside ``mcp_call``, since the module-global
    ``_session`` is ``None`` in unit tests) — patching the class methods is
    the single binding-independent network chokepoint every alias funnels
    through.

    RECORDS each attempt into the yielded list (rather than only raising)
    because the two dominant live seams here swallow exceptions:
    ``Scheduler.set_task_claimant`` wraps ``dispatch_tool`` in a bare
    ``try/except`` (and ``Harness._run_slot``'s ``finally`` additionally
    wraps that call in ``contextlib.suppress(Exception)``), and
    ``Scheduler.get_tasks`` wraps it in a ``try/except`` that returns
    ``[]``.  A raise-only spy would be silently absorbed by either path,
    reproducing a false "zero calls" result.  Recording the attempt at the
    seam — before/independent of the raise — and asserting the recorder is
    empty at teardown guarantees a swallowed live round-trip still fails
    the test.  The raised ``AssertionError`` still gives non-suppressed
    callers a fast, clear failure message.

    NOT autouse — request by name in tests that inject a hermetic
    ``mcp_session`` and need a guarantee that they actually did so.
    """
    recorder: list[tuple[str, str]] = []

    async def _fake_initialize(self) -> None:
        recorder.append((self.base_url, 'initialize'))
        raise AssertionError(
            f'live fused-memory MCP round-trip attempted: initialize at '
            f'{self.base_url!r} — inject a HermeticMcpSession into the '
            'Scheduler under test (see _orch_helpers.HermeticMcpSession)'
        )

    async def _fake_raw_call(self, method: str, params=None, timeout: float = 30) -> dict:
        recorder.append((self.base_url, method))
        raise AssertionError(
            f'live fused-memory MCP round-trip attempted: {method!r} at '
            f'{self.base_url!r} — inject a HermeticMcpSession into the '
            'Scheduler under test (see _orch_helpers.HermeticMcpSession)'
        )

    monkeypatch.setattr('orchestrator.mcp_lifecycle.McpSession.initialize', _fake_initialize)
    monkeypatch.setattr('orchestrator.mcp_lifecycle.McpSession._raw_call', _fake_raw_call)

    yield recorder

    assert recorder == [], (
        f'forbid_live_mcp: {len(recorder)} live fused-memory MCP round-trip(s) '
        f'were attempted (some may have been swallowed by caller exception '
        f'handling): {recorder!r}'
    )


@pytest.fixture(autouse=True)
def _no_dry_run_unblock(monkeypatch, request):
    """Replace ``orchestrator.workflow.run_dry_run_unblock`` with an async no-op.

    The real hook fires the Claude CLI fire-and-forget; in the test suite that
    causes pytest-timeout hangs (~60 s × ~8 tests) whenever ``_mark_blocked``
    runs.  Tests that genuinely need the real binding (e.g.
    ``test_workflow_dry_run_hook.py`` stacks its own ``patch(...)`` on top)
    can opt out with the ``exercise_dry_run_unblock`` marker.
    """
    if request.node.get_closest_marker('exercise_dry_run_unblock'):
        return

    async def _noop(**_):
        return None

    monkeypatch.setattr('orchestrator.workflow.run_dry_run_unblock', _noop)


@pytest.fixture(autouse=True)
def _restore_sandbox_backend():
    """Snapshot and restore sandbox_dispatch._preferred around every test.

    Mirrors the worktree-root conftest's fixture so that running this
    subproject in isolation (e.g. ``cd orchestrator && uv run pytest tests/``,
    which is exactly how Fix 2's per-subproject test_command invokes us)
    still restores backend state between tests.
    """
    from orchestrator.agents import sandbox_dispatch
    saved = sandbox_dispatch.get_backend()
    yield
    sandbox_dispatch.set_backend(saved)


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    """Clear verify._PROBE_CACHE (and its sibling _BASELINE_FAILING_IDS_CACHE)
    before and after every test.

    ``verify_failure_is_preexisting_on_main`` stores results in the
    process-global ``_PROBE_CACHE`` dict keyed by
    ``(main_sha, category, normalized_cause_hint)`` with a 300 s TTL.
    When two tests on the same xdist worker share an identical key (same
    MAIN_SHA + FAILING_RESULT), the earlier test's cached True entry
    short-circuits the probe path in the later test — causing it to skip
    worktree creation / cleanup and return the wrong result.

    ``_BASELINE_FAILING_IDS_CACHE`` (task μ, verify-scope-inversion-prd.md) is
    the sibling per-main-SHA failing-test-id baseline cache keyed on
    ``main_sha`` alone — the same cross-test-pollution hazard applies (an
    earlier test's seeded/probed baseline for a reused MAIN_SHA would
    silently serve a later test's ``main_baseline_failing_ids`` call from
    cache instead of exercising its own probe/seed path).

    Clearing the caches before *and* after each test ensures:
    - every test starts with empty caches regardless of prior teardown;
    - this suite's pollution cannot escape to any later consumer.

    Production is unaffected: main_sha advances on every merge so real
    keys never collide across separate runs.

    **Maintainer note — adding new verify.py process-globals:**
    If ``orchestrator/src/orchestrator/verify.py`` gains additional
    module-level caches (e.g. a sibling result cache), add a matching
    ``.clear()`` call here so they are also reset between tests.  The
    relevant globals are defined near ``_PROBE_CACHE``/``_PROBE_CACHE_TTL``
    and ``_BASELINE_FAILING_IDS_CACHE``.
    """
    from orchestrator import verify
    verify._PROBE_CACHE.clear()
    verify._BASELINE_FAILING_IDS_CACHE.clear()
    yield
    verify._PROBE_CACHE.clear()
    verify._BASELINE_FAILING_IDS_CACHE.clear()


@pytest.fixture(autouse=True)
def _mock_merge_queue_verification(monkeypatch, request):
    """Patch merge_queue's run_scoped_verification to return passed=True by default.

    MergeWorker hardcodes orchestrator.merge_queue.run_scoped_verification in its
    internal calls; tests that create a live MergeWorker need this patched or
    pytest/ruff/pyright (not in PATH in test environments) cause BLOCKED outcomes.
    Tests that need specific merge-verification behaviour override this with their
    own monkeypatch.setattr call in the test body.

    **Opt-out via ``exercise_merge_verify`` marker** (task 1829):
    Tests whose purpose IS to assert that a compile-broken member is CAUGHT by the
    post-merge verify gate must run the REAL ``run_scoped_verification`` (e.g. real
    cargo), not the passed=True stub — otherwise the gate always returns 'done' and
    the correctness-invariant assertion ``outcome.status != 'done'`` fails
    immediately without cargo ever running.  Marking such a test with
    ``@pytest.mark.exercise_merge_verify`` causes this fixture to return early
    (skip the monkeypatch), restoring the real binding.

    An Explore audit (task 1829) confirmed that only two tests are in this at-risk
    category — tests whose correctness assertion requires the blocked outcome and
    which use the marker as the SOLE restore mechanism (no in-body patch):
      - test_train_integration.py::TestTrainIntegrationB2::test_lower_member_break_blocks_train
      - test_atomic_train_merge.py::TestScenario5GroupMergeVerify::test_group_merge_workspace_verify_red
    Do NOT add a redundant in-body ``patch('orchestrator.merge_queue.run_scoped_verification',
    ...)`` to these tests — the marker is sufficient and is the authoritative mechanism
    (pinned by test_merge_verify_mock_autouse.py).
    Every other test that drives the real merge path either overrides
    run_scoped_verification itself, injects verify via a different seam
    (run_scoped= param / workflow-layer patch), tests a failure that occurs BEFORE
    verify (rebase conflict / incomplete-train), or expects the passed/happy path.

    Mirrors ``_no_dry_run_unblock`` (conftest.py:102-118) exactly.
    """
    if request.node.get_closest_marker('exercise_merge_verify'):
        return

    from orchestrator.verify import VerifyResult
    monkeypatch.setattr(
        'orchestrator.merge_queue.run_scoped_verification',
        AsyncMock(return_value=VerifyResult(
            passed=True, test_output='', lint_output='',
            type_output='', summary='mocked merge verify',
        )),
    )


@pytest.fixture(autouse=True)
def _neutralize_verify_admission(monkeypatch, request):
    """Patch verify._verify_admission_active to False for every test except
    those marked ``real_verify_admission``.

    Task 2390 (T2) wires ``shared.verify_admission``'s flock semaphore + role
    nice prefix into every test-leg ``verify.py`` pytest spawn, DEFAULT ON
    (``verify_admission_enabled=True``). Left unneutralized, the entire
    existing verify suite would route through the ``nice ... /bin/bash -c
    <shlex.quote(cmd)>`` wrap (mangling substring cmd assertions, e.g.
    env-transient tests matching ``-o addopts=''``) and contend on the
    shared ``/tmp/df-verify-slots-<uid>`` directory across xdist workers.
    This fixture keeps every legacy test byte-identical by forcing the
    module seam ``orchestrator.verify._verify_admission_active`` to return
    False, so admission is inactive regardless of
    ``config.verify_admission_enabled``.

    **Opt-out via ``real_verify_admission`` marker**: tests that specifically
    exercise the admission wiring (nice-wrap, acquire, fail-open, ...) mark
    themselves with ``@pytest.mark.real_verify_admission`` to restore the
    real seam, causing this fixture to return early (skip the monkeypatch).

    Guarded on ``hasattr`` so this fixture is a safe no-op until task 2390's
    verify.py wiring (step-6) actually defines ``_verify_admission_active`` —
    every commit before that stays green without this fixture touching
    anything.

    Mirrors ``_mock_merge_queue_verification`` (conftest.py:231-275).
    """
    if request.node.get_closest_marker('real_verify_admission'):
        return
    from orchestrator import verify
    if not hasattr(verify, '_verify_admission_active'):
        return
    monkeypatch.setattr('orchestrator.verify._verify_admission_active', lambda config: False)


@pytest.fixture(autouse=True)
def _drain_async_mock_coroutines():
    """Drain orphaned AsyncMock._execute_mock_call coroutines after every test.

    Task 1714 / esc-1702-13: prevents order-dependent orchestrator test failures
    caused by un-awaited AsyncMock coroutines surviving GC cycles into sibling
    tests.  CPython emits RuntimeWarning("coroutine '...' was never awaited")
    when GC finalizes such an orphan; orchestrator/pyproject.toml promotes this
    (and pytest's PytestUnraisableExceptionWarning wrapper) to hard errors via
    filterwarnings — failing whichever test the GC ran during.

    By closing every CORO_CREATED ``_execute_mock_call`` coroutine at each test's
    own teardown boundary, orphans are reclaimed before they can be promoted into
    a sibling.  Product coroutines (co_name != _execute_mock_call) are untouched,
    preserving the real-leak safety net.

    KNOWN LIMITATION — module/session-scoped fixture teardowns: pytest finalises
    fixtures in reverse setup order.  An orphaned AsyncMock coroutine created in
    the *teardown* of a fixture set up BEFORE this one (e.g. a module- or
    session-scoped fixture) will be finalised AFTER drain's teardown runs, so it
    is reclaimed at the *next* test's drain boundary rather than the current one.
    This is an edge case: function-scoped fixtures (the majority) tear down in
    definition order before this fixture's teardown, so they are covered.  If a
    module/session fixture teardown is found to create AsyncMock orphans, either
    add an explicit ``await`` there or register an additional
    ``pytest_runtest_teardown`` hook that fires after all finalizers.

    See drain_async_mock_coroutines() in _orch_helpers.py for full rationale and
    performance notes.
    """
    yield
    drain_async_mock_coroutines()


@pytest.fixture
def make_steward(tmp_path: Path):
    """Build a minimal ``TaskSteward`` on a fixture-OWNED, ``tmp_path``-rooted worktree.

    The single steward factory for the orchestrator suite (task 3461 merged
    ``test_suggestion_triage.py``'s and ``test_workflow_state_machine_boundary.py``'s
    two near-identical copies).  Lives in conftest.py — rather than
    ``_orch_helpers.py``, which is scoped to non-fixture helpers — because it
    must close over ``tmp_path`` to own the worktree directory; ``mock_orch_config``
    below is the same shape and the precedent for it.

    Worktree ownership — an ENFORCED invariant, not a convention: with no
    ``worktree=`` argument the fixture picks ``tmp_path / 'wt'`` and creates it
    (plus a ``.task`` subdir), so the common case is structurally safe and no
    caller needs to name a path.  A caller-supplied path (for the
    two-stewards-in-one-test case) is asserted to resolve *strictly below*
    ``tmp_path`` before anything is created, because the steward derives its
    artifacts root as a SIBLING of the worktree —
    ``<worktree.parent>/.task-meta/<worktree.name>``, see ``TASK_META_DIRNAME``
    in ``orchestrator/config.py`` and ``TaskArtifacts.meta_root_for`` in
    ``orchestrator/artifacts.py`` — so ``worktree=tmp_path`` would put that
    sibling outside the directory pytest's retention policy reclaims.

    Config defaults applied (union of what both former factories set; a caller
    overrides any of them via ``config_overrides``, applied last):
      - ``project_root`` = ``tmp_path / 'project'`` (created)
      - ``steward_lifetime_budget`` = 12.0, ``steward_max_attempts`` = 3
      - ``steward_max_timeouts_per_escalation`` = 3,
        ``steward_max_empty_outputs_per_escalation`` = 2
      - ``suggestion_triage_threshold`` = 10
      - triage + steward ``models`` / ``budgets`` / ``max_turns`` / ``effort`` / ``backends``
      - ``escalation.host`` / ``escalation.port``

    The config mock is ``spec_set``'d against ``OrchestratorConfig`` so a typo'd
    field name raises ``AttributeError`` on both read and write.
    """
    def _make(*, worktree: Path | None = None, config_overrides: dict | None = None):
        from orchestrator.steward import TaskSteward

        if worktree is None:
            worktree = tmp_path / 'wt'
        else:
            # Checked BEFORE any mkdir, so a rejected path is never created.
            resolved, root = worktree.resolve(), tmp_path.resolve()
            if resolved == root or not resolved.is_relative_to(root):
                raise AssertionError(
                    f'make_steward(worktree={worktree!r}) must be strictly below this '
                    f"test's tmp_path ({tmp_path}). The steward derives its .task-meta "
                    'artifacts root as <worktree.parent>/.task-meta/<worktree.name> '
                    '(orchestrator/config.py:TASK_META_DIRNAME, artifacts.meta_root_for), '
                    'so a worktree AT tmp_path lands them in tmp_path.parent — outside '
                    'the directory pytest reclaims. Pass a sub-path, or omit the kwarg '
                    'and let the fixture own the dir.'
                )
        worktree.mkdir(parents=True, exist_ok=True)
        (worktree / '.task').mkdir(exist_ok=True)

        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        project_root = tmp_path / 'project'
        project_root.mkdir(parents=True, exist_ok=True)
        config.project_root = project_root
        config.steward_lifetime_budget = 12.0
        config.steward_max_attempts = 3
        config.steward_max_timeouts_per_escalation = 3
        config.steward_max_empty_outputs_per_escalation = 2
        config.suggestion_triage_threshold = 10
        config.models.triage = 'sonnet'
        config.budgets.triage = 2.0
        config.max_turns.triage = 25
        config.effort.triage = 'medium'
        config.backends.triage = 'claude'
        config.models.steward = 'opus'
        config.budgets.steward = 5.0
        config.max_turns.steward = 100
        config.effort.steward = 'high'
        config.backends.steward = 'claude'
        config.escalation.host = '127.0.0.1'
        config.escalation.port = 8100
        for k, v in (config_overrides or {}).items():
            setattr(config, k, v)

        queue = MagicMock()
        queue.make_id.return_value = 'esc-42-1'
        queue.get_by_task.return_value = []
        queue.get.return_value = None

        briefing = AsyncMock()
        briefing.build_steward_initial_prompt = AsyncMock(return_value='initial prompt')

        mcp = MagicMock()
        mcp.mcp_config_json.return_value = {'mcpServers': {}}

        return TaskSteward(
            task_id='42',
            task={'id': '42', 'title': 'Test Task', 'description': 'desc'},
            worktree=worktree,
            config=config,
            mcp=mcp,
            escalation_queue=queue,
            briefing=briefing,
            usage_gate=None,
        )

    return _make


@pytest.fixture
def mock_orch_config(tmp_path: Path) -> MagicMock:
    """Return a MagicMock OrchestratorConfig with the standard harness defaults pre-applied.

    Defaults applied:
      - ``git`` = real ``GitConfig`` with main/task/origin/.worktrees
      - ``project_root`` = ``tmp_path``
      - ``usage_cap.enabled`` = False
      - ``review.enabled`` = False
      - ``sandbox.backend`` = 'auto'
      - ``fused_memory`` = pre-created sub-section mock (no default value)
      - ``escalation`` = pre-created sub-section mock (no default value)
      - ``overrides_db_path`` = ``tmp_path / 'overrides.db'``
      - ``park_eviction_requests_db_path`` = ``tmp_path / 'park_eviction_requests.db'``

    The top-level mock and each sub-section (usage_cap, review, sandbox,
    fused_memory, escalation) are spec_set'd against their pydantic model's
    fields so typos raise AttributeError on both get and set.

    Apply test-specific overrides directly on the returned object.

    Gotcha — pydantic_spec hides BaseModel methods
    -----------------------------------------------
    ``pydantic_spec`` (see ``_orch_helpers.py``) intentionally exposes
    ``model.model_fields`` names AND user-defined ``@property`` descriptors
    (e.g. ``overrides_db_path``) to ``MagicMock(spec_set=...)``.  BaseModel
    methods — ``model_dump``, ``model_validate``, ``model_copy``, etc. — are
    NOT in the proxy class, so ``spec_set`` rejects them on both *read* and
    *write*::

        mock_orch_config.model_dump()           # raises AttributeError
        mock_orch_config.model_dump = MagicMock(...)  # also raises AttributeError

    If a test genuinely needs BaseModel API access, use a real
    ``OrchestratorConfig`` (or the relevant sub-section model) instance
    instead of this fixture.  See ``_orch_helpers.pydantic_spec`` for the
    underlying rationale.
    """
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.git = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
    )
    config.project_root = tmp_path
    config.usage_cap = MagicMock(spec_set=pydantic_spec(UsageCapConfig))
    config.usage_cap.enabled = False
    config.review = MagicMock(spec_set=pydantic_spec(ReviewConfig))
    config.review.enabled = False
    config.sandbox = MagicMock(spec_set=pydantic_spec(SandboxConfig))
    config.sandbox.backend = 'auto'
    config.fused_memory = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
    config.escalation = MagicMock(spec_set=pydantic_spec(EscalationConfig))
    # Real numeric defaults for fields read by harness lifecycle loops —
    # MagicMock values would crash ``asyncio.sleep(<MagicMock>)``.
    # claimant_heartbeat_interval_secs mirrors the identical guard in
    # test_workflow_already_done.py:54 -- without it, a spec'd-MagicMock
    # config can leak into asyncio.sleep(MagicMock) in
    # _claimant_heartbeat_loop (workflow.py:2113) under load-exposed
    # teardown ordering, raising TypeError.
    config.claimant_heartbeat_interval_secs = 60.0
    config.orphan_l0_check_interval_secs = 60.0
    config.orphan_l0_reaper_enabled = False
    config.orphan_l0_timeout_secs = 600.0
    config.terminal_status_watcher_enabled = False
    config.terminal_status_poll_interval_secs = 30.0
    # Stranded-in-progress reconcile sweep — disabled by default in tests so
    # the background loop doesn't spin up under fixtures that don't expect
    # an additional asyncio.Task to manage.
    config.stranded_reconcile_enabled = False
    config.stranded_reconcile_interval_secs = 900.0
    # No-landings circuit-breaker (task 1918/θ-1893). Disabled by default, like
    # the other background sweep loops above, so its asyncio.Task doesn't spin
    # up under fixtures that don't expect it — and so its `while True` loop
    # can't churn on MagicMock config (a MagicMock interval crashes
    # asyncio.sleep, which the loop logs-and-retries forever). window_samples /
    # disk_free_floor_bytes are read synchronously in Harness.__init__ to size
    # collections.deque(maxlen=...), so they must be real ints regardless of
    # `enabled` (a MagicMock there raises "TypeError: an integer is required").
    # Values mirror the production defaults (config.py: 60 s, 30 samples,
    # 50 GiB floor); breaker-specific tests override these per-test.
    config.no_landings_breaker_enabled = False
    config.no_landings_breaker_interval_secs = 60.0
    config.no_landings_breaker_window_samples = 30
    config.no_landings_breaker_disk_free_floor_bytes = 50 * 1024 * 1024 * 1024
    # Main-tip integrity sweep — disabled by default in tests (task 1907).
    # Left enabled, the sweep loop starts under mocked git_ops/transport, its
    # pass fails immediately, and (absent a real interval) the loop can spin
    # logging exceptions — starving the xdist worker until the 60s per-test
    # timeout os._exit()s it.  A real interval is also set so any test that
    # explicitly re-enables the sweep gets a sane asyncio.sleep() argument.
    config.main_tip_sweep_enabled = False
    config.main_tip_sweep_interval_secs = 1800.0
    # Warm-lane GC cadence loop — disabled by default in tests (task 1927).
    # Defaults on (warm_lane_gc_enabled=True) in production; left enabled here,
    # the loop's asyncio.sleep() call fires under the monkeypatched sleep and
    # inflates idle-sleep counts in TestHarnessRunForever (observed: assert 3 == 1).
    # Real interval set so any test that explicitly re-enables gets a sane
    # asyncio.sleep() argument instead of a MagicMock crash.
    config.warm_lane_gc_enabled = False
    config.warm_lane_gc_interval_secs = 600.0
    # Real Path so OverrideStore.__init__ can call .parent.mkdir() and
    # sqlite3.connect(str(...)) without crashing — config.overrides_db_path
    # is a @property on OrchestratorConfig (see config.py) and Harness wires
    # OverrideStore.from_config(config) at construction (task 1313).
    config.overrides_db_path = tmp_path / 'overrides.db'
    # Real Path so ParkEvictionRequestStore.from_config(config) can call
    # .parent.mkdir() and sqlite3.connect(str(...)) without stringifying an
    # unpatched MagicMock into a stray on-disk file — park_eviction_requests_db_path
    # is a @property on OrchestratorConfig (see config.py) and Harness wires
    # ParkEvictionRequestStore.from_config(config) at construction (task 1871).
    # Same failure mode as overrides_db_path above (tasks 1313/1339); this
    # instance of it is task 2045.
    config.park_eviction_requests_db_path = tmp_path / 'park_eviction_requests.db'
    return config
