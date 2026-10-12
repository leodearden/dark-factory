"""Every .jsx that persists a UI preference does so through persisted_state.js (task 5743).

persisted_state.js owns the storage policy and the one copy of the
``usePersistedState`` / ``useOpenSet`` hooks; their behaviour is pinned under
``node --test`` in ``tests/js/persisted_state.test.mjs``.  The consumers are
``type="text/babel"`` with no render harness, so this module checks the
structural half, one rule per test, over every consumer alike: each binds what
it calls from the module, keeps no hook copy of its own, and neither reaches
``localStorage`` nor swallows an error.

One place to change the policy, and one place a quota failure is observed: the
hook copies this replaced each wrote through ``try { localStorage... } catch (e) {}``.
"""

from __future__ import annotations

import re

import pytest
from _dashboard_helpers import strip_js_comments

CONSUMERS = (
    'tabs.jsx',
    'tab_escalations.jsx',
    'tab_escalation_analytics.jsx',
    'tab_memory_evals.jsx',
)
_POLICY_EXPORTS = ('readPersisted', 'writePersisted', 'createPersistedHooks')
_HOOKS = ('usePersistedState', 'useOpenSet')

_POLICY_BINDING = re.compile(r'const\s*\{([^}]*)\}\s*=\s*window\.DF_PERSISTED_STATE\s*;')
_HOOKS_BINDING = re.compile(r'const\s*\{([^}]*)\}\s*=\s*createPersistedHooks\(\s*React\s*\)\s*;')
_EMPTY_CATCH = re.compile(r'\bcatch\s*(?:\(\s*\w*\s*\))?\s*\{\s*\}')


@pytest.fixture(scope='module')
def consumer_code(_client) -> dict[str, str]:
    return {
        name: strip_js_comments(_client.get(f'/static/redux/{name}').text)
        for name in CONSUMERS
    }


def _called(code: str, names: tuple[str, ...]) -> set[str]:
    return {name for name in names if re.search(rf'\b{name}\s*\(', code)}


def _bound(binding: re.Pattern[str], code: str) -> set[str]:
    match = binding.search(code)
    return set(re.findall(r'\b\w+\b', match.group(1))) if match else set()


@pytest.mark.parametrize('consumer', CONSUMERS)
def test_consumer_binds_what_it_calls_from_the_policy_module(
    consumer_code: dict[str, str], consumer: str,
) -> None:
    """Each consumer destructures, at module scope, every persisted_state.js export it calls.

    A .jsx destructure compiles to a global ``var``, so a call left unbound
    would still resolve through some other file's binding, and break the day
    that file stops making it.
    """
    code = consumer_code[consumer]
    called = _called(code, _POLICY_EXPORTS)
    assert called, f'{consumer} calls nothing from persisted_state.js; drop it from CONSUMERS.'
    unbound = called - _bound(_POLICY_BINDING, code)
    assert not unbound, (
        f'{consumer} calls {sorted(unbound)} without destructuring them from '
        'window.DF_PERSISTED_STATE at module scope.'
    )


@pytest.mark.parametrize('consumer', CONSUMERS)
def test_consumer_takes_its_hooks_from_the_factory(
    consumer_code: dict[str, str], consumer: str,
) -> None:
    """Each persisted hook a consumer calls is bound from ``createPersistedHooks(React)``."""
    code = consumer_code[consumer]
    unbound = _called(code, _HOOKS) - _bound(_HOOKS_BINDING, code)
    assert not unbound, (
        f'{consumer} calls {sorted(unbound)} without binding them from '
        '`createPersistedHooks(React)` at module scope.'
    )


@pytest.mark.parametrize('consumer', CONSUMERS)
def test_consumer_keeps_no_hook_copy(consumer_code: dict[str, str], consumer: str) -> None:
    """No consumer defines its own ``usePersistedState`` / ``useOpenSet``."""
    code = consumer_code[consumer]
    for hook in _HOOKS:
        assert not re.search(rf'\bfunction\s+{hook}\s*\(', code), (
            f'{consumer} defines its own `{hook}`; persisted_state.js::createPersistedHooks '
            'holds the one copy.'
        )


@pytest.mark.parametrize('consumer', CONSUMERS)
def test_consumer_does_not_reach_local_storage(consumer_code: dict[str, str], consumer: str) -> None:
    assert not re.search(r'\blocalStorage\b', consumer_code[consumer]), (
        f'{consumer} reaches localStorage directly; go through persisted_state.js.'
    )


@pytest.mark.parametrize('consumer', CONSUMERS)
def test_consumer_swallows_no_error(consumer_code: dict[str, str], consumer: str) -> None:
    assert not _EMPTY_CATCH.search(consumer_code[consumer]), (
        f'{consumer} has an empty catch block, which swallows a storage failure unseen.'
    )
