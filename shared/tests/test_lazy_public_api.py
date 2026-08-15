"""Behavioural-parity pins for the lazy (PEP 562) `shared` package.

`shared/__init__.py` resolves its 47 public names through a module-level
``__getattr__`` instead of importing them eagerly (task 3896 — the eager block
made every stdlib-only leaf drag in pydantic/aiosqlite/yaml/dotenv/aiohttp).
That rewrite must be BEHAVIOUR-PRESERVING, and these tests pin the parts a
minimal symbol-only resolver would silently drop:

  - the 10 submodules the eager block used to bind as package attributes,
  - and, from the other side, that no OTHER real submodule became reachable,
  - ``dir(shared)`` reporting the public surface,
  - the AttributeError shape for an unknown name,
  - caching resolved values into the package globals,
  - every ``__all__`` name resolving to its submodule's own object.

Note the pair of submodule-reachability tests. A behaviour-preserving rewrite
has two failure directions, and pinning only the positive one leaves the
package free to grow a wider public surface than it had — so the set is held
open by PREVIOUSLY_EAGER_SUBMODULES and closed by NEVER_EXPORTED_SUBMODULES.

WHY THREE OF THESE RUN IN A SUBPROCESS. CPython's import machinery sets a
submodule as an attribute of its parent package, so once ANY test in the
session does ``import shared.agent_result``, ``getattr(shared,
'agent_result')`` succeeds regardless of what ``__getattr__`` does — measured
first-hand. The same warmth would mask both the negative reachability check
and the globals-caching check. All three are therefore asserted in a fresh
interpreter, via ``run_python_child`` imported from
shared/tests/test_pure_stdlib_leaves.py.
"""

from __future__ import annotations

import importlib

import pytest

# The child runner lives in its sibling guard module rather than being copied
# here — one home for the PYTHONPATH-must-beat-the-installed-copy invariant.
# Bare-module import: shared/tests is on sys.path via conftest, the same shape
# as test_usage_gate.py's `from test_config_dir import ...`.
from test_pure_stdlib_leaves import run_python_child

#: The submodules the pre-lazy eager block bound as package attributes. A frozen
#: HISTORICAL constant — do not extend it when a new exporting submodule is
#: added. Before task 3896 `import shared; shared.usage_gate` worked (as a side
#: effect of the eager imports) while `shared.psi` raised AttributeError, and
#: both halves of that must stay true.
PREVIOUSLY_EAGER_SUBMODULES = (
    'agent_result',
    'async_sqlite_base',
    'cli_invoke',
    'config_models',
    'cost_store',
    'locking',
    'mcp_idempotency',
    'safe_io',
    'sqlite_sync_base',
    'usage_gate',
)

#: Real `shared.*` modules that were NEVER package attributes. They exist on
#: disk, so an unrestricted `importlib.import_module(f'shared.{name}')` fallback
#: — or a careless addition to `_LAZY_SUBMODULES` — would resolve them and
#: silently widen the public surface. Sampled across the third-party-backed
#: (`task_metadata`), pure-leaf (`psi`, `task_statuses`) and test-support
#: (`testing`) kinds.
NEVER_EXPORTED_SUBMODULES = (
    'psi',
    'task_metadata',
    'task_statuses',
    'testing',
)


_SUBMODULE_ATTR_CHILD_SRC = """
import json
import sys
import types

import shared  # deliberately the ONLY import: importing shared.<x> here would
               # bind <x> on the package and mask what __getattr__ does.

out = {}
for name in sys.argv[1].split(','):
    try:
        value = getattr(shared, name)
    except AttributeError as exc:
        out[name] = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
    else:
        out[name] = {
            'ok': True,
            'name': getattr(value, '__name__', None),
            'is_module': isinstance(value, types.ModuleType),
        }
print(json.dumps(out))
"""


_CACHING_CHILD_SRC = """
import json

import shared

before = 'UsageGate' in vars(shared)
value = shared.UsageGate
after = 'UsageGate' in vars(shared)
print(json.dumps({
    'before': before,
    'after': after,
    'identity': after and vars(shared)['UsageGate'] is value,
}))
"""


def test_previously_eager_submodules_resolve_as_attributes():
    """`import shared; shared.usage_gate` must keep working.

    Before task 3896 this worked as a side effect of the eager imports. The
    lazy package must not silently drop it — so it is restored explicitly, for
    exactly the 10 submodules that used to be reachable, rather than by an
    unrestricted fallback that would silently widen the public surface.
    """
    out = run_python_child(_SUBMODULE_ATTR_CHILD_SRC, ','.join(PREVIOUSLY_EAGER_SUBMODULES))

    unresolved = {n: r['error'] for n, r in out.items() if not r['ok']}
    assert unresolved == {}, (
        'these submodules were package attributes before the lazy rewrite and '
        f'no longer resolve: {unresolved}'
    )
    wrong = {
        n: r for n, r in out.items() if not r['is_module'] or r['name'] != f'shared.{n}'
    }
    assert wrong == {}, f'resolved to something other than the expected module: {wrong}'


def test_submodules_outside_the_preserved_set_stay_unreachable():
    """`shared.psi` must keep raising AttributeError — the other half of the pin.

    ``_LAZY_SUBMODULES`` is narrow by design, but only the positive test above
    exists to hold it open; nothing held it CLOSED. Replacing the ``else: raise
    AttributeError`` branch with a blanket ``importlib.import_module(f'shared.
    {name}')``, or quietly adding a name to ``_LAZY_SUBMODULES``, would widen
    the package's public surface with a fully green suite — and a widened
    surface is a compatibility promise nobody meant to make.

    The names below are REAL modules, so an unrestricted fallback resolves them
    and this test goes red; a fabricated name (as in
    ``test_unknown_attribute_raises_attribute_error``) would raise
    AttributeError either way and could not tell the two designs apart.
    """
    out = run_python_child(_SUBMODULE_ATTR_CHILD_SRC, ','.join(NEVER_EXPORTED_SUBMODULES))

    reachable = sorted(name for name, result in out.items() if result['ok'])
    assert reachable == [], (
        f'`shared.<name>` now resolves for {len(reachable)} submodule(s) that '
        f'were never package attributes: {", ".join(reachable)}.\n'
        'This widens the public surface. If it is intended, it must be an '
        'explicit edit to _LAZY_SUBMODULES *and* to this list — not a side '
        'effect of loosening __getattr__.'
    )
    wrong_error = {
        name: result['error']
        for name, result in out.items()
        if not result['error'].startswith('AttributeError')
    }
    assert wrong_error == {}, (
        f'unreachable submodules must fail with AttributeError, not: {wrong_error}'
    )


def test_dir_lists_the_whole_public_surface():
    """dir(shared) must report the public surface, not just what is warm.

    Without an explicit __dir__, a lazy package reports only names already in
    its __dict__, so tab-completion and introspection would show a surface that
    shrinks and grows with import order.
    """
    import shared

    missing = sorted(set(shared.__all__) - set(dir(shared)))
    assert missing == [], (
        f'{len(missing)} public name(s) absent from dir(shared): {missing}\n'
        'shared/__init__.py needs a __dir__() covering the lazily-resolved names.'
    )


def test_unknown_attribute_raises_attribute_error():
    """A typo'd name must fail with the standard diagnostic.

    Loud over silent: AttributeError (never ImportError/KeyError), naming both
    the module and the attribute, is what `hasattr`/`getattr` callers and a
    mistyped `from shared import ...` expect to see.
    """
    import shared

    with pytest.raises(AttributeError) as excinfo:
        _ = shared.definitely_not_a_symbol

    message = str(excinfo.value)
    assert 'shared' in message, f'AttributeError should name the module: {message!r}'
    assert 'definitely_not_a_symbol' in message, (
        f'AttributeError should name the attribute: {message!r}'
    )


def test_resolved_symbol_is_cached_on_the_package():
    """A resolved symbol is written back into the package globals.

    This is what keeps repeated attribute access off the importlib path:
    once resolved, the name lives in the module __dict__ and normal attribute
    lookup finds it without ever calling __getattr__ again. Asserted in a fresh
    interpreter because pytest's already-warm `shared` would mask it.
    """
    out = run_python_child(_CACHING_CHILD_SRC)

    assert out['before'] is False, (
        "'UsageGate' was already in vars(shared) before first access — the "
        'package is not lazy, so this test cannot observe caching.'
    )
    assert out['after'] is True, (
        'accessing shared.UsageGate did not cache it into the package globals; '
        'every subsequent access pays another importlib.import_module lookup.'
    )
    assert out['identity'] is True, (
        'the cached object is not the object returned by attribute access'
    )


def test_every_public_name_resolves_and_is_the_submodule_object():
    """Every `__all__` name resolves to its owning submodule's own object.

    Guards the failure mode the lazy design introduces: a gap or typo in the
    symbol->module map used to be a loud ImportError at package import time,
    and now degrades to an AttributeError at first use — possibly deep inside
    an unrelated caller.

    The set of submodules to scan comes from `_SYMBOL_MODULE` (which tracks the
    package as it is today) but each OWNERSHIP claim is then checked against
    that submodule's own `__all__` and its own object — so the test stays
    non-tautological while surviving the addition of an eleventh exporting
    submodule. Scanning PREVIOUSLY_EAGER_SUBMODULES instead would fail such an
    addition with 'in shared.__all__ but in no submodule __all__', which is
    false and sends the next maintainer to the wrong file.
    """
    import shared

    owners: dict[str, list[str]] = {}
    for module_name in sorted(set(shared._SYMBOL_MODULE.values())):
        module = importlib.import_module(f'shared.{module_name}')
        for symbol in module.__all__:
            owners.setdefault(symbol, []).append(module_name)

    unowned = sorted(set(shared.__all__) - set(owners))
    assert unowned == [], f'in shared.__all__ but in no submodule __all__: {unowned}'

    failures: list[str] = []
    for symbol in shared.__all__:
        try:
            resolved = getattr(shared, symbol)
        except AttributeError as exc:
            failures.append(f'{symbol}: unresolvable ({exc})')
            continue
        # CheckpointResult is exported by two submodules, so accept any owner.
        candidates = [
            getattr(importlib.import_module(f'shared.{m}'), symbol) for m in owners[symbol]
        ]
        if not any(resolved is candidate for candidate in candidates):
            failures.append(
                f'{symbol}: shared.{symbol} is not the object exported by '
                f'{" or ".join(f"shared.{m}" for m in owners[symbol])}'
            )

    assert failures == [], 'lazy resolution disagrees with the submodules:\n  ' + '\n  '.join(
        failures
    )
