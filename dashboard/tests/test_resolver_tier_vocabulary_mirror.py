"""The frontend resolver-tier whitelists mirror escalation.classify.RESOLVER_TIERS.

INV-10 tier 2 (docs/legibility/design-invariants.md): each JS array literal is
read out of its source file and compared with the live Python tuple. A tier
missing from a whitelist would fold into 'other-auto'
(tab_escalation_analytics.jsx::resolverTierKey) and render as the wrong signal.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from escalation.classify import RESOLVER_TIERS

_REDUX = Path(__file__).resolve().parents[1] / 'src' / 'dashboard' / 'static' / 'redux'
_ANALYTICS_TAB = _REDUX / 'tab_escalation_analytics.jsx'
_FLOW_LAYOUT = _REDUX / 'esc_flow_layout.js'
_OBJECT_KEY = re.compile(r"^\s*(?:'([^']+)'|([A-Za-z_$][\w$]*))\s*:", re.M)


def _declaration(path: Path, name: str, opener: str, closer: str) -> str:
    """The text between *name*'s opener and closer, failing unless declared exactly once."""
    pattern = re.compile(rf'^const {re.escape(name)} = {re.escape(opener)}(.*?){re.escape(closer)};', re.M | re.S)
    found = pattern.findall(path.read_text())
    assert len(found) == 1, f'{path.name}: expected one `const {name} = {opener}...{closer};`, found {len(found)}'
    return found[0]


def _array_literal(path: Path, name: str) -> list[str]:
    values = ast.literal_eval(f'[{_declaration(path, name, "[", "]")}]')
    assert values and all(isinstance(v, str) for v in values), f'{path.name}: {name} is not a list of strings'
    return values


@pytest.mark.parametrize(
    ('path', 'name'), [(_ANALYTICS_TAB, '_RESOLVER_TIERS'), (_FLOW_LAYOUT, 'TIER_ORDER')],
    ids=['analytics-tab', 'flow-layout'],
)
def test_the_whitelist_is_the_record_tier_vocabulary_in_order(path: Path, name: str):
    assert _array_literal(path, name) == list(RESOLVER_TIERS)


def test_every_tier_colour_names_a_record_tier():
    body = _declaration(_ANALYTICS_TAB, '_TIER_COLORS', '{', '}')
    keys = [quoted or bare for quoted, bare in _OBJECT_KEY.findall(body)]
    assert keys, '_TIER_COLORS has no keys'
    assert set(keys) <= set(RESOLVER_TIERS), sorted(set(keys) - set(RESOLVER_TIERS))
