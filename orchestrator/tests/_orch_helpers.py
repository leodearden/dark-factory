"""Non-fixture test helpers for orchestrator tests.

Lives outside conftest.py to avoid the `sys.modules['conftest']` collision
that arises when root-level pytest loads multiple subprojects' conftests in
the same process.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Sequence
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

from pydantic import BaseModel
from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import AccountState, UsageGate

if TYPE_CHECKING:
    from orchestrator.harness import Harness

# Constants for the process lifetime — lifted out of pydantic_spec (task 1426)
# to avoid re-computing BaseModel reflection on every call.
_BASEMODEL_PROPS: frozenset[str] = frozenset(
    name for name, v in inspect.getmembers(BaseModel) if isinstance(v, property)
)
_BASEMODEL_ATTRS: frozenset[str] = frozenset(dir(BaseModel))


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
