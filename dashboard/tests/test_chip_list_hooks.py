"""Structural contract for the hooks behind tabs.jsx's shared ``ChipList`` (task 5743).

tabs.jsx is ``type="text/babel"`` with no render harness (task 4899 is
unlanded), so this reads the served source, comment-stripped, scoped to the
``ChipList`` body.

``ChipList`` used to pick its hook on ``persistKey``:
``persistKey ? usePersistedState(persistKey, false) : uS(false)``.
``usePersistedState`` calls two hooks and the other arm one, so the first caller
whose ``persistKey`` changed between renders would get React's "Rendered fewer
hooks than expected".  The fix calls ``usePersistedState`` unconditionally; that
it tolerates a falsy key is pinned behaviourally in
``tests/js/persisted_state.test.mjs``.  Once task 4899's render harness lands, a
render that changes ``persistKey`` between renders should replace this check.
"""

from __future__ import annotations

import re

import pytest
from _dashboard_helpers import extract_function_body, strip_js_comments

_HOOK_CALL = re.compile(r'\b(?:usePersistedState|uS)\s*\(')


@pytest.fixture(scope='module')
def chip_list_body(tabs_jsx_body: str) -> str:
    return extract_function_body(strip_js_comments(tabs_jsx_body), 'ChipList')


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
