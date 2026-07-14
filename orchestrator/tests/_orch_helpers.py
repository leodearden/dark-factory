"""Non-fixture test helpers for orchestrator tests.

Lives outside conftest.py to avoid the `sys.modules['conftest']` collision
that arises when root-level pytest loads multiple subprojects' conftests in
the same process.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import inspect
import logging
import threading
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel
from shared.config_models import AccountConfig, UsageCapConfig
from shared.psi import PsiSample
from shared.usage_gate import AccountState, UsageGate

if TYPE_CHECKING:
    from orchestrator.harness import Harness

_log = logging.getLogger(__name__)


def mock_lock_table(held: dict[str, set[str]] | None = None) -> MagicMock:
    """MagicMock ``ModuleLockTable`` stand-in with a working ``is_held(tid)``.

    Task 2235: harness.py's liveness reach-ins now call
    ``scheduler.is_actively_held()``, which reads ``lock_table.is_held()``
    rather than ``lock_table._held`` directly. Tests that reassign
    ``scheduler.lock_table`` to simulate held/free modules must use this
    helper (not a bare ``MagicMock()``) so ``is_held`` reflects the
    ``_held`` state they set up rather than an auto-mocked (truthy) stub.
    """
    lt = MagicMock()
    lt._held = {} if held is None else held
    lt.is_held = lambda task_id: task_id in lt._held
    return lt


def wire_scheduler_liveness_mock(scheduler_mock: MagicMock) -> None:
    """Wire real (non-auto-mock) liveness-accessor behaviour onto a MagicMock
    Scheduler stand-in.

    Task 2235: harness.py now calls ``scheduler.is_dispatched`` /
    ``.is_actively_held`` / ``.workflow_cancel_recent`` /
    ``.note_workflow_cancelled`` / ``.clear_workflow_cancel`` instead of
    reaching into ``_dispatched`` / ``lock_table._held`` /
    ``_workflow_cancel_at`` directly. On a bare ``MagicMock()`` those methods
    auto-mock to a truthy return regardless of any raw state a test sets up,
    silently breaking tests that pre-date task 2235's accessor migration.
    This wires real implementations closing over the mock's own mutable
    state, so tests may still freely reassign ``scheduler_mock._dispatched``
    (a plain set) or ``scheduler_mock.lock_table`` (use :func:`mock_lock_table`)
    and have the accessors reflect it.
    """
    scheduler_mock._dispatched = set()
    scheduler_mock.lock_table = mock_lock_table()
    scheduler_mock._workflow_cancel_at = {}
    scheduler_mock._RECONCILE_CANCEL_GRACE_S = 30.0
    scheduler_mock.is_dispatched = lambda tid: tid in scheduler_mock._dispatched
    scheduler_mock.note_workflow_cancelled = (
        lambda tid: scheduler_mock._workflow_cancel_at.__setitem__(tid, time.monotonic())
    )
    scheduler_mock.clear_workflow_cancel = (
        lambda tid: scheduler_mock._workflow_cancel_at.pop(tid, None)
    )

    def _workflow_cancel_recent(tid: str) -> bool:
        stamp = scheduler_mock._workflow_cancel_at.get(tid)
        return (
            stamp is not None
            and time.monotonic() - stamp < scheduler_mock._RECONCILE_CANCEL_GRACE_S
        )

    scheduler_mock.workflow_cancel_recent = _workflow_cancel_recent
    scheduler_mock.is_actively_held = lambda tid: (
        tid in scheduler_mock._dispatched
        or scheduler_mock.lock_table.is_held(tid)
        or scheduler_mock.workflow_cancel_recent(tid)
    )

# task 2376: generous merge-pipeline result-wait ceiling that tolerates host
# oversubscription; never-narrow — only replaces literals <=15 across the
# merge-pipeline test files (test_merge_queue.py, test_merge_speculation.py,
# test_concurrent_verify_boundary.py, test_merge_queue_resolve_release.py,
# test_merge_queue_permit_conservation.py,
# test_merge_queue_invariant_integration_gate.py).
MERGE_RESULT_TIMEOUT = 45

# Constants for the process lifetime — lifted out of pydantic_spec (task 1426)
# to avoid re-computing BaseModel reflection on every call.
_BASEMODEL_PROPS: frozenset[str] = frozenset(
    name for name, v in inspect.getmembers(BaseModel) if isinstance(v, property)
)
_BASEMODEL_ATTRS: frozenset[str] = frozenset(dir(BaseModel))


def idle_psi_sample() -> PsiSample:
    """Return the hermetic default PSI reading: idle and read_ok.

    Task 2418 — fix for the scheduler-test xdist flake where the dispatch-
    admission gate (``psi_admission.enabled`` defaults True, scheduler.py)
    reads REAL ``/proc/pressure/*`` via ``self._read_psi_sample()`` once per
    ``acquire_next()`` tick.  Under host load (e.g. full parallel ``pytest -n
    auto``) ``mem_some_avg10`` can cross its configured threshold, so the
    gate non-deterministically defers dispatch of non-deterministic
    candidates in any bare-config test.

    This is a non-saturating sample (all four metrics 0.0, well under every
    configured threshold) with ``read_ok=True`` — it exercises the gate's
    normal "healthy host" path rather than the ``read_ok=False`` fail-open
    sentinel, so tests validate the real admission logic rather than a
    short-circuit.  Mirrors the admission suite's own idle default,
    ``test_scheduler_dispatch_admission.py``'s ``_psi(read_ok=True)`` (all
    metrics 0.0).

    Injected suite-wide as every ``Scheduler``'s default ``_read_psi_sample``
    by conftest's autouse ``_hermetic_psi_reader`` fixture, which monkeypatches
    the module-level ``orchestrator.scheduler.read_psi_sample`` (bound at
    ``__init__`` time to ``self._read_psi_sample``).  Callable with no
    arguments to match that call site (``self._read_psi_sample()``,
    scheduler.py) exactly — tests that need a specific (e.g. saturated)
    reading still reassign ``scheduler._read_psi_sample`` post-construction,
    which overrides this default same as before.
    """
    return PsiSample(
        cpu_some10=0.0,
        mem_some10=0.0,
        mem_full10=0.0,
        io_some10=0.0,
        read_ok=True,
    )


def drain_async_mock_coroutines() -> int:
    """Close all orphaned AsyncMock._execute_mock_call coroutines in the GC graph.

    Task 1714 / esc-1702-13 — fix for order-dependent orchestrator test failures.

    ROOT CAUSE: Calling an AsyncMock without awaiting the returned coroutine
    produces a ``_execute_mock_call`` coroutine (cr_code.co_name ==
    ``"_execute_mock_call"``, state CORO_CREATED).  When that orphan is held in
    a reference cycle (common in mock object graphs) it survives ref-count
    reclamation and is only finalized by a gc.collect() that may run during an
    arbitrary *later* test.  CPython then emits
    ``RuntimeWarning("coroutine '...' was never awaited")`` (or pytest's
    PytestUnraisableExceptionWarning wrapper).  orchestrator/pyproject.toml
    promotes BOTH to hard errors via ``filterwarnings``, failing whichever test
    the GC ran during — producing order-dependent suite failures.

    FIX: .close() on a CORO_CREATED coroutine finalizes it WITHOUT emitting the
    warning (verified under warnings.simplefilter("error") on CPython 3.13.9).
    Walking gc.get_objects() and closing every CORO_CREATED ``_execute_mock_call``
    coroutine before the test boundary ensures each test's orphans are reclaimed
    at its OWN boundary and cannot be promoted into a sibling.

    SAFETY NET preserved: only AsyncMock's internal ``_execute_mock_call``
    coroutines are closed; a genuinely forgotten ``await`` on product code yields
    a coroutine with a different co_name and still trips filterwarnings=error.
    Product coroutines are intentionally NOT reclaimed here, so a missing
    ``await`` in production code will still raise under filterwarnings=error.

    PERFORMANCE NOTE: ``gc.get_objects()`` materialises every live Python object
    (O(total live objects)) on every call.  The teardown ``gc.collect()`` is
    skipped when ``closed == 0`` to avoid a full-generation collection on the
    common no-orphan path.  If profiling shows the walk itself is a bottleneck
    across the full suite, consider a pytest opt-in marker for tests that use
    AsyncMock to scope the walk — the current unconditional path is a safe
    default.

    DIAGNOSTIC: a DEBUG-level log is emitted for each test that leaks AsyncMock
    orphans, so tests that routinely call AsyncMock without awaiting can be
    identified and fixed at source rather than being masked indefinitely.

    Returns the number of coroutines closed (useful for assertions in tests).
    """
    closed = 0
    for obj in gc.get_objects():
        if (
            inspect.iscoroutine(obj)
            and getattr(getattr(obj, 'cr_code', None), 'co_name', None)
            == '_execute_mock_call'
            and inspect.getcoroutinestate(obj) == inspect.CORO_CREATED
        ):
            obj.close()
            closed += 1
    if closed:
        _log.debug(
            'drain_async_mock_coroutines: closed %d orphaned AsyncMock '
            '_execute_mock_call coroutine(s) — consider auditing un-awaited '
            'AsyncMock calls in the preceding test',
            closed,
        )
        gc.collect()
    return closed


async def reap_leaked_aiosqlite_connections() -> int:
    """Close and join any aiosqlite Connection whose worker thread outlived its test.

    Task 2413 — fix for the scheduler-test xdist flake where a leaked aiosqlite
    connection's background worker thread survives past the end of the test
    that created it.

    ROOT CAUSE: ``aiosqlite.Connection`` runs one background ``Thread`` per
    connection (``aiosqlite.core._connection_worker_thread``). Each completed
    operation resolves its future via
    ``future.get_loop().call_soon_threadsafe(...)`` (core.py:66). If nothing
    closes the connection before the test's event loop is torn down
    (pytest-asyncio closes a fresh loop per test), the worker thread is still
    alive when a later operation tries to resolve a future against the
    now-CLOSED loop, raising ``RuntimeError: Event loop is closed`` from
    inside the thread. ``core.py``'s own exception handler re-invokes
    ``call_soon_threadsafe`` (which raises again, uncaught), so the thread
    dies loudly and pytest's threadexception plugin attributes the failure to
    whatever unrelated test happens to be executing when the thread fires —
    under CPU starvation (``pytest -n auto`` on an oversubscribed host) that
    is reliably a *different*, innocent test. orchestrator/pyproject.toml
    promotes the resulting ``PytestUnhandledThreadExceptionWarning`` to a
    hard error.

    FIX: at every test's teardown boundary — while that test's own event loop
    is STILL OPEN — find any live aiosqlite worker thread and gracefully
    ``await conn.close()`` then join its thread. ``close()`` drains the
    connection's queue and resolves any pending futures against the
    still-open loop before flipping ``_running`` to False and queuing the
    stop sentinel; joining guarantees the thread has actually exited before
    this fixture (and therefore the test) returns, so it can never later
    touch a closed loop.

    MAINTENANCE NOTE: this walk relies on aiosqlite ``Connection`` internals
    that are not part of its public API: the ``_running`` flag, the
    ``_thread`` attribute, and ``aiosqlite.core._connection_worker_thread`` as
    the pre-check thread target. If a future aiosqlite release renames or
    removes any of these, the guarded import below makes this helper a no-op
    (returns 0) rather than raising — but the leak protection this provides
    would then be silently lost, so re-verify this helper against aiosqlite's
    source whenever the pinned version changes.

    PERFORMANCE NOTE: mirrors ``drain_async_mock_coroutines``'s gc-walk
    shape, but gates the (relatively expensive) ``gc.get_objects()`` walk
    behind a cheap ``threading.enumerate()`` pre-check for a live aiosqlite
    worker thread, so the common no-leak path costs a single thread-list scan
    rather than a full object-graph walk on every test.

    Returns the number of connections reaped (useful for assertions in tests).
    """
    try:
        import aiosqlite
        from aiosqlite.core import _connection_worker_thread
    except Exception:
        return 0

    if not any(
        t.is_alive() and getattr(t, '_target', None) is _connection_worker_thread
        for t in threading.enumerate()
    ):
        return 0

    closed = 0
    for obj in gc.get_objects():
        if not (
            isinstance(obj, aiosqlite.Connection)
            and getattr(obj, '_running', False)
            and obj._thread.is_alive()
        ):
            continue
        # Exception (not BaseException): asyncio.CancelledError is a
        # BaseException subclass (since 3.8) and must propagate rather than
        # be silently absorbed if the test run itself is being cancelled
        # while this teardown is awaiting; SystemExit/KeyboardInterrupt are
        # likewise deliberately left unsuppressed. asyncio.TimeoutError
        # (raised by the wait_for below) is a plain Exception subclass, so
        # it — and any close()/thread-join failure — is still swallowed on
        # this best-effort path.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(obj.close(), timeout=15.0)
            obj._thread.join(timeout=15.0)
        if not obj._thread.is_alive():
            closed += 1
    if closed:
        _log.debug(
            'reap_leaked_aiosqlite_connections: closed %d leaked aiosqlite '
            'connection(s) — consider auditing missing `await conn.close()` '
            'in the preceding test',
            closed,
        )
    return closed


def _init_harness_state_for_test(h: Harness) -> None:
    """Initialise task-1327 AFK-hardening digest counters on a __new__-built Harness.

    Fixtures that construct ``Harness`` via ``Harness.__new__(Harness)`` and
    manually set only the attributes their tests exercise must call this helper
    to avoid silent ``AttributeError`` in ``_maybe_write_digest``.  The catch-all
    in that method (and in the supervisor wrapper) has been narrowed (task 1449,
    step-4) so ``AttributeError`` is re-raised rather than swallowed — missing
    state now surfaces as a test failure rather than a silent warning log.

    Delegates to ``Harness._init_digest_state()`` (task 1449 amend) so this
    helper calls the same canonical code as ``Harness.__init__`` rather than
    duplicating counter names by value.  When new state is added to
    ``_maybe_write_digest`` in the future, update ``_init_digest_state`` as the
    single fix-point — all seven ``Harness.__new__``-based fixtures pick it up
    automatically via this helper.
    """
    # task 1449: delegates to canonical method so no values are duplicated here
    h._init_digest_state()


def pydantic_spec(model: type[BaseModel]) -> type:
    """Return a proxy class exposing ``model``'s fields for ``MagicMock(spec_set=...)``.

    Pydantic v2 hides field names from ``dir()``, so passing a BaseModel subclass
    directly to ``spec_set=`` would only expose BaseModel/BaseSettings methods.
    The returned proxy has each field name as a class attribute, so MagicMock
    sees them and rejects typos on both get and set.

    User-defined ``@property`` descriptors (e.g. ``OrchestratorConfig.overrides_db_path``)
    are also included in the proxy so ``spec_set`` accepts both read and write.
    BaseModel-inherited properties (``model_extra``, ``model_fields_set``, …)
    are excluded to preserve the invariant that BaseModel API surface is NOT
    exposed.

    User-defined regular methods (callables on ``model`` not inherited from
    ``BaseModel``, not dunder names, not ``@property`` descriptors) are also
    included.  The canonical example is ``OrchestratorConfig.for_module``
    (config.py:991): a plain instance method absent from ``model_fields``.
    This removes the need for the ~18 ad-hoc ``_spec.for_module = None``
    patches that previously worked around the spec_set gap.  The method walk
    covers the full MRO down to (but not including) ``BaseModel``, so methods
    inherited from any intermediate base class (e.g. a shared mixin) are also
    included — spec_set should permit access to any non-BaseModel callable the
    real object exposes.

    Pydantic v2 ``PrivateAttr`` members (e.g. ``OrchestratorConfig._module_configs``
    at config.py:971) are also included by walking ``model.__private_attributes__``.
    PrivateAttr names begin with ``_`` and are therefore excluded by the regular
    method walk's underscore filter; this separate walk re-includes them because
    they are legitimate mock assignment targets.

    BaseModel API (``model_dump``, ``model_validate``, ``model_construct``,
    ``model_copy``, ``model_json_schema``, ``model_post_init``, …) remains
    explicitly excluded via the module-level ``_BASEMODEL_ATTRS`` frozenset
    (``frozenset(dir(BaseModel))``, computed once at import) applied to the
    method walk.  This preserves the invariant established by task 1064: writing
    ``mock.model_dump = ...`` must still raise ``AttributeError``.
    """
    members: dict[str, None] = {f: None for f in model.model_fields}
    # @property descriptors declared on the user's class (e.g.
    # OrchestratorConfig.overrides_db_path) — exclude properties inherited
    # from BaseModel (model_extra, model_fields_set, __fields_set__) so the
    # existing "BaseModel API is not exposed" invariant is preserved.
    for name, _ in inspect.getmembers(model, lambda v: isinstance(v, property)):
        if name in _BASEMODEL_PROPS:
            continue
        members[name] = None
    # User-defined regular methods — exclude dunders, BaseModel API surface, and
    # anything already collected as a @property above.
    # The walk covers the *full* MRO down to (but not including) BaseModel, so
    # methods inherited from any intermediate base class (e.g. a shared mixin
    # between BaseModel and the target model) are also included.  This is
    # intentional: spec_set should permit access to any non-BaseModel callable
    # the real object exposes, including helpers inherited from mixins.
    for name, _ in inspect.getmembers(model, callable):
        if name.startswith('_'):
            continue
        if name in _BASEMODEL_ATTRS:
            continue
        if isinstance(getattr(model, name, None), property):
            # Belt-and-braces: plain properties fail callable() and are already
            # excluded by the predicate above.  This guard catches exotic
            # descriptors that subclass property AND implement __call__, which
            # would slip through the callable() filter but belong in the
            # @property walk (already collected), not here.
            continue
        members[name] = None
    # Pydantic v2 PrivateAttr members — stored in __private_attributes__ dict,
    # NOT in model_fields.  The underscore-name filter above intentionally skips
    # these; walk them separately so PrivateAttrs bypass that filter.
    for name in getattr(model, '__private_attributes__', {}):
        members[name] = None
    return type(
        f'_{model.__name__}Spec',
        (),
        members,
    )


_GATE_PROPERTY_DEFAULTS: dict[str, object] = {
    # Known-good values for every UsageGate @property.
    # This dict is the *single* fix-point for property defaults: update here when
    # UsageGate gains a new @property, and every make_mock_gate() call site picks
    # it up automatically.  Motivation: the 122-error cascade (tasks 1313/1339)
    # where soonest_resets_at was added to UsageGate but not to every mock factory.
    'account_count': 1,
    'active_account_name': 'acct-a',
    'soonest_resets_at': None,
    'paused_reason': '',
    'cumulative_cost': 0.0,
    'total_pause_secs': 0.0,
    'is_paused': False,
    'project_id': None,
    'run_id': None,
}
# Guard: catch stale keys in _GATE_PROPERTY_DEFAULTS (e.g. renamed @property).
# Runs at import time so a typo or post-rename ghost key surfaces immediately.
_GATE_VALID_PROPS: frozenset[str] = frozenset(
    n for n, _ in inspect.getmembers(UsageGate, lambda v: isinstance(v, property))
)
assert set(_GATE_PROPERTY_DEFAULTS) <= _GATE_VALID_PROPS, (
    f'_GATE_PROPERTY_DEFAULTS contains keys that are not UsageGate @properties: '
    f'{set(_GATE_PROPERTY_DEFAULTS) - _GATE_VALID_PROPS!r}. '
    'Remove the stale entries from the dict.'
)


def make_mock_gate(**overrides) -> MagicMock:
    """Build a MagicMock UsageGate with all public @property defaults initialised.

    Property defaults are driven by introspecting UsageGate's @property surface via
    ``inspect.getmembers(UsageGate, property)`` so that new @property additions
    auto-propagate to every ``make_mock_gate()`` call without manual edits.  The
    canonical regression test (``TestMakeGateFactory.test_make_gate_covers_usage_gate_public_property_surface``
    in ``orchestrator/tests/test_invoke.py``) asserts that ``vars(make_mock_gate())``
    covers the full @property set, catching any future drift at test time.

    Known-good values for each property are stored in ``_GATE_PROPERTY_DEFAULTS``
    (the single fix-point).  Any future property not yet in that dict falls back
    to ``None``.

    Method mocks (``before_invoke``, ``on_agent_complete``, ``confirm_account_ok``,
    ``release_probe_slot``) are kept explicit because they require typed mock
    instances (``AsyncMock`` vs ``MagicMock``) that cannot be inferred from
    introspection alone.

    Accepts ``**overrides`` so callers can pin specific values without
    re-specifying the rest.

    Sister helper: ``shared/tests/test_cap_retry.py::_mock_gate`` — same shape
    but with extra ``invoke_slot()`` async-CM wiring for shared-layer tests.
    Cannot be unified here: ``shared`` cannot import from ``orchestrator/tests``
    (that would invert the package layering direction).

    Imported into ``orchestrator/tests/test_invoke.py`` and
    ``orchestrator/tests/test_steward.py`` as ``_make_gate`` via::

        from _orch_helpers import make_mock_gate as _make_gate

    so existing call sites (≈20 across both files) remain unchanged.
    """
    gate = MagicMock()
    # Set all known @property defaults; unknown future ones fall back to None.
    for prop_name, _ in inspect.getmembers(UsageGate, lambda v: isinstance(v, property)):
        default = _GATE_PROPERTY_DEFAULTS.get(prop_name)
        setattr(gate, prop_name, overrides.pop(prop_name, default))
    # Explicit method mocks (need typed mock instances).
    gate.before_invoke = overrides.pop('before_invoke', AsyncMock(return_value='tok-a'))
    gate.on_agent_complete = overrides.pop('on_agent_complete', MagicMock())
    gate.confirm_account_ok = overrides.pop('confirm_account_ok', MagicMock())
    gate.release_probe_slot = overrides.pop('release_probe_slot', MagicMock())
    gate.detect_cap_hit = overrides.pop('detect_cap_hit', MagicMock(return_value=False))
    for k, v in overrides.items():
        setattr(gate, k, v)
    return gate


def make_gate_yielding(slots, *, active_account_name=None) -> MagicMock:
    """Build a mock UsageGate whose successive invoke_slot() calls yield *slots* in order.

    Each element of *slots* is returned by ``gate.invoke_slot().__aenter__``, so
    production code that does ``async with gate.invoke_slot() as slot:`` gets a
    real slot object with a controlled ``detect_cap_hit`` on each iteration of the
    cap-retry while-loop.

    Without this helper, ``gate = MagicMock()`` yields an unconstrained slot whose
    ``detect_cap_hit`` returns a truthy coroutine — production's ``while`` loop
    never exits and the test hangs until pytest-timeout fires.

    Always routes through ``make_mock_gate`` so the gate carries the full property
    default surface (guards against bare-MagicMock drift).

    Imported into ``orchestrator/tests/test_invoke.py`` and
    ``orchestrator/tests/test_steward.py`` as ``_make_gate_yielding`` via::

        from _orch_helpers import make_gate_yielding as _make_gate_yielding

    Sister helper: ``shared/tests/test_cap_retry.py::_mock_gate`` — cannot be
    unified here due to package layering (shared cannot import from orchestrator).
    """
    slot_iter = iter(slots)

    def _new_cm(*args, **kwargs):
        slot = next(slot_iter)
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=slot)
        cm.__aexit__ = AsyncMock(return_value=False)
        return cm

    gate = make_mock_gate(
        account_count=len(slots),
        active_account_name=(
            active_account_name if active_account_name is not None
            else slots[0].account_name
        ),
        before_invoke=AsyncMock(return_value=slots[0].token),
    )
    gate.invoke_slot = MagicMock(side_effect=_new_cm)
    return gate


_PLACEHOLDER_LOOP: asyncio.AbstractEventLoop | None = None


def make_placeholder_future() -> asyncio.Future:
    """Return an asyncio.Future bound to a dedicated placeholder event loop.

    Use this in SYNC test bodies (``def test_*``) that build a ``MergeRequest``
    before entering ``asyncio.run()``, where the ``MergeRequest.result`` future
    is a structural placeholder that is *never* awaited or resolved on the test's
    run loop.

    The future is created on a module-global loop that is:

    * Lazily created (once) and reused across calls within a process.
    * NEVER installed as the thread-current loop via ``asyncio.set_event_loop()``,
      so this helper neither reads nor mutates the thread-current-loop state that
      ``asyncio.run()`` nulls on completion.
    * Persistently alive (not GC'd), so no ``ResourceWarning`` is emitted.

    **Do NOT use this helper where the future is awaited or resolved inside a
    running event loop** — in those contexts (``async def`` test bodies) use
    ``asyncio.get_running_loop().create_future()`` instead so the future stays
    bound to the actual running loop.
    """
    global _PLACEHOLDER_LOOP
    # `is_closed()` is defensive cover for an externally-closed loop — this
    # module never closes _PLACEHOLDER_LOOP itself, so the branch is not
    # exercised in normal use.
    if _PLACEHOLDER_LOOP is None or _PLACEHOLDER_LOOP.is_closed():
        _PLACEHOLDER_LOOP = asyncio.new_event_loop()
    return _PLACEHOLDER_LOOP.create_future()


def build_usage_gate(
    account_configs: list[AccountConfig],
    tokens: Sequence[str | None],
    *,
    wait_for_reset: bool = False,
    session_budget_usd: float | None = None,
    probe_interval_secs: int = 300,
    max_probe_interval_secs: int = 1800,
) -> UsageGate:
    """Create a UsageGate with tokens pre-injected (no os.environ lookup).

    Bypasses UsageGate.__init__ via __new__ and sets all private attrs directly,
    then injects AccountState entries from the parallel ``tokens`` list rather
    than reading from environment variables.  This is the canonical pattern for
    constructing test gates — both _make_gate (test_usage_gate.py) and
    _make_reify_gate (test_reify_multi_account.py) delegate to this helper.

    Parameters
    ----------
    account_configs:
        List of AccountConfig instances (same shape as UsageCapConfig.accounts).
    tokens:
        Parallel list of OAuth token strings (or None for default-credential
        accounts).  Must be the same length as ``account_configs``.
    wait_for_reset:
        Forwarded to UsageCapConfig.
    session_budget_usd:
        Forwarded to UsageCapConfig.
    probe_interval_secs:
        Forwarded to UsageCapConfig.
    max_probe_interval_secs:
        Forwarded to UsageCapConfig.

    Raises
    ------
    TypeError
        If ``tokens`` is a bare ``str`` instead of a list/tuple.
    ValueError
        If ``account_configs`` and ``tokens`` have different lengths.
    """
    if isinstance(tokens, str):
        raise TypeError(
            f'tokens must be a list/tuple of str|None, not a bare str; '
            f'got {tokens!r}. Did you forget to wrap it in a list?'
        )
    if len(account_configs) != len(tokens):
        raise ValueError(
            f'account_configs and tokens must have the same length; '
            f'got {len(account_configs)} account(s) and {len(tokens)} token(s)'
        )

    config = UsageCapConfig(
        wait_for_reset=wait_for_reset,
        session_budget_usd=session_budget_usd,
        probe_interval_secs=probe_interval_secs,
        max_probe_interval_secs=max_probe_interval_secs,
        accounts=account_configs,
    )

    gate = UsageGate.__new__(UsageGate)
    gate._config = config
    gate._open = asyncio.Event()
    gate._open.set()
    gate._lock = asyncio.Lock()
    gate._cumulative_cost = 0.0
    gate._paused_reason = ''
    gate._pause_started_at = None
    gate._total_pause_secs = 0.0
    gate._cost_store = None
    gate._project_id = None
    gate._run_id = None
    gate._last_account_name = None
    gate._background_tasks = set()
    gate._probe_config_dir = MagicMock()
    gate._run_probe = AsyncMock(return_value=True)
    gate._accounts = [
        AccountState(name=cfg.name, token=tok)
        for cfg, tok in zip(account_configs, tokens, strict=True)
    ]
    return gate


def assert_update_wire_mode(
    captured_payloads: list[dict],
    expected_mode: str,
) -> dict:
    """Assert exactly one update_task wire call with the expected metadata_mode.

    Shared by wire-mode tests across test_harness_prd_tagging,
    test_harness_module_tagging, test_auto_eval, and
    test_harness_reblock_signature_guard — each exercises a distinct caller
    path but asserts the same two invariants:

    1. Exactly one ``update_task`` call was captured.
    2. Its ``arguments`` carry ``metadata_mode == expected_mode`` and
       the legacy ``'append'`` key is absent (the wrapper must never
       forward ``append`` on the wire).

    Returns the ``arguments`` dict so callers can add further assertions
    (e.g. inspecting the serialised metadata blob).

    For tests that expect *two* update_task calls (e.g. the dry-run
    proposal-persist + trim test) use the raw filter idiom directly, since
    each call asserts a *different* expected_mode.
    """
    update_calls = [p for p in captured_payloads if p.get('name') == 'update_task']
    assert len(update_calls) == 1, (
        f'Expected exactly 1 update_task wire call; got {len(update_calls)} '
        f'(all calls: {[p.get("name") for p in captured_payloads]})'
    )
    arguments = update_calls[0]['arguments']
    assert arguments.get('metadata_mode') == expected_mode, (
        f'update_task wire call must forward metadata_mode={expected_mode!r}; '
        f'got: {arguments}'
    )
    assert 'append' not in arguments, (
        f"'append' key must not appear on the wire; got: {arguments}"
    )
    return arguments
