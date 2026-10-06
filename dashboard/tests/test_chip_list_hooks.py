"""Structural contracts for the hooks behind tabs.jsx's shared ``ChipList`` (task 5743).

tabs.jsx is ``type="text/babel"`` with no render harness (task 4899 is
unlanded), so these read the served source, comment-stripped, scoped to one
function body at a time.

``ChipList`` used to pick its hook on ``persistKey``:
``persistKey ? usePersistedState(persistKey, false) : uS(false)``.
``usePersistedState`` calls two hooks and the other arm one, so the first caller
whose ``persistKey`` changed between renders would get React's "Rendered fewer
hooks than expected".  The fix calls ``usePersistedState`` unconditionally and
makes it tolerate a falsy key.
"""

from __future__ import annotations

import re

import pytest
from _dashboard_helpers import extract_function_body, strip_js_comments, walk_balanced

_HOOK_CALL = re.compile(r'\b(?:usePersistedState|uS)\s*\(')


@pytest.fixture(scope='module')
def tabs_jsx_code(tabs_jsx_body: str) -> str:
    return strip_js_comments(tabs_jsx_body)


@pytest.fixture(scope='module')
def chip_list_body(tabs_jsx_code: str) -> str:
    return extract_function_body(tabs_jsx_code, 'ChipList')


def _hook_argument(body: str, hook: str) -> str:
    """The parenthesised argument list of the first ``<hook>(`` call in *body*."""
    call = re.search(rf'\b{hook}\s*\(', body)
    assert call is not None, f'no `{hook}(` call in the body: {body!r}'
    return walk_balanced(body, call.end() - 1, '(', ')')


def test_chip_list_selects_no_hook_conditionally(chip_list_body: str) -> None:
    """No statement in ``ChipList`` chooses between hooks with a condition."""
    for statement in chip_list_body.split(';'):
        if _HOOK_CALL.search(statement):
            assert not re.search(r'\?|&&|\|\|', statement), (
                'ChipList selects a hook conditionally — the hook count would then '
                f'depend on its props, which React forbids: {statement.strip()!r}'
            )

    declaration = re.search(
        r'const\s*\[\s*expanded\s*,\s*setExpanded\s*\]\s*=([^;]*);', chip_list_body
    )
    assert declaration is not None, 'ChipList no longer declares `[expanded, setExpanded]`.'
    assert len(_HOOK_CALL.findall(declaration.group(1))) == 1, (
        f'`expanded` must come from exactly one hook call: {declaration.group(0)!r}'
    )


def test_chip_list_persists_expanded_unconditionally(chip_list_body: str) -> None:
    """The positive half: the persisted hook is called, so the checks above are not vacuous."""
    assert re.search(
        r'const\s*\[\s*expanded\s*,\s*setExpanded\s*\]\s*=\s*'
        r'usePersistedState\(\s*persistKey\s*,\s*false\s*\)\s*;',
        chip_list_body,
    ), 'ChipList must call `usePersistedState(persistKey, false)` unconditionally.'


_FALSY_KEY_GUARD = re.compile(
    r'!\s*storageKey\b|\bif\s*\(\s*storageKey\s*\)|\bstorageKey\s*(?:\?|&&)'
)


@pytest.mark.parametrize(
    'hook, delegate',
    [('uS', 'readPersisted'), ('uE', 'writePersisted')],
    ids=['lazy-initialiser', 'persist-effect'],
)
def test_use_persisted_state_tolerates_a_falsy_key(
    tabs_jsx_code: str, hook: str, delegate: str,
) -> None:
    """A falsy ``storageKey`` neither reads nor writes storage.

    Either an inline guard, or delegation to the persisted_state.js policy
    function whose own falsy-key handling is pinned behaviourally under
    ``node --test``.
    """
    body = extract_function_body(tabs_jsx_code, 'usePersistedState')
    argument = _hook_argument(body, hook)

    assert _FALSY_KEY_GUARD.search(argument) or re.search(rf'\b{delegate}\s*\(', argument), (
        f"usePersistedState's {hook}(...) touches storage with no falsy-key guard, "
        f'so ChipList without a persistKey would read or write the key "undefined": '
        f'{argument!r}'
    )
