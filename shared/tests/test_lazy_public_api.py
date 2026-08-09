"""Behavioural-parity pins for the lazy (PEP 562) `shared` package.

`shared/__init__.py` resolves its 47 public names through a module-level
``__getattr__`` instead of importing them eagerly (task 3896 — the eager block
made every stdlib-only leaf drag in pydantic/aiosqlite/yaml/dotenv/aiohttp).
That rewrite must be BEHAVIOUR-PRESERVING, and these tests pin the parts a
minimal symbol-only resolver would silently drop:

  - the 10 submodules the eager block used to bind as package attributes,
  - ``dir(shared)`` reporting the public surface,
  - the AttributeError shape for an unknown name,
  - caching resolved values into the package globals,
  - every ``__all__`` name resolving to its submodule's own object.

WHY TWO OF THESE RUN IN A SUBPROCESS. CPython's import machinery sets a
submodule as an attribute of its parent package, so once ANY test in the
session does ``import shared.agent_result``, ``getattr(shared,
'agent_result')`` succeeds regardless of what ``__getattr__`` does — measured
first-hand. The same warmth would mask the globals-caching check. Both are
therefore asserted in a fresh interpreter, using the child-runner shape from
shared/tests/test_pure_stdlib_leaves.py.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Same src-root expression as shared/tests/conftest.py and
# test_public_api.py:255 — read the LOCAL tree, never an installed copy.
_SRC = Path(__file__).resolve().parent.parent / 'src'

#: The submodules the pre-lazy eager block bound as package attributes. This is
#: a behaviour-preservation set: before task 3896 `import shared;
#: shared.usage_gate` worked (as a side effect of the eager imports) while
#: `shared.psi` raised AttributeError, and that must stay true.
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


def _run_child(src: str, *argv: str) -> dict:
    """Run `src` in a fresh interpreter against the local src tree; parse its JSON."""
    env = {
        **os.environ,
        'PYTHONPATH': f'{_SRC}{os.pathsep}' + os.environ.get('PYTHONPATH', ''),
    }
    proc = subprocess.run(
        [sys.executable, '-c', src, *argv],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f'child exited {proc.returncode}\n'
        f'--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}'
    )
    return json.loads(proc.stdout)


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
    out = _run_child(_SUBMODULE_ATTR_CHILD_SRC, ','.join(PREVIOUSLY_EAGER_SUBMODULES))

    unresolved = {n: r['error'] for n, r in out.items() if not r['ok']}
    assert unresolved == {}, (
        'these submodules were package attributes before the lazy rewrite and '
        f'no longer resolve: {unresolved}'
    )
    wrong = {
        n: r for n, r in out.items() if not r['is_module'] or r['name'] != f'shared.{n}'
    }
    assert wrong == {}, f'resolved to something other than the expected module: {wrong}'


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
    out = _run_child(_CACHING_CHILD_SRC)

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
    an unrelated caller. Owners are derived independently, from the submodules'
    own __all__, so this cannot go tautological against the map it checks.
    """
    import shared

    owners: dict[str, list[str]] = {}
    for module_name in PREVIOUSLY_EAGER_SUBMODULES:
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
