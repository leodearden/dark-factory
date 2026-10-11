"""Contract: every ``census.TONES`` value resolves to a real ``charts.jsx::PALETTE`` key.

``census.py`` maps each status to a PALETTE tone KEY rather than a colour
literal, so the palette stays owned by charts.jsx alone. Nothing executed that
citation until this file, and a broken one is invisible: ``PALETTE[tone]``
evaluates to ``undefined``, which React renders as a missing
``style.color``/``fill`` — a test-GREEN regression. ``test_census.py``'s
``test_tones_cover_every_status_exactly_once`` only asserts the tone is
truthy, and ``tests/js/task_vocab.test.mjs`` only that it is a non-empty
string; neither can see that the key it names no longer exists. ``stranded``
is the live hazard — a recent, singly-used addition.

WHY A PYTHON SOURCE READ. charts.jsx is a Babel file node cannot load, so the
shipped artifact cannot be asked what its palette contains. This is the route
``test_charts_consumer_bindings.py`` already takes to the adjacent DF_CHARTS
bindings, over the same shared parser in ``_dashboard_helpers``.

AND WHY THAT IS NOT THE UNTESTED MATCHER PRD DECISION 17 REJECTS. That
decision rejects a SUBSTRING GUARD whose matcher is untested — a token grep
for an idiom, which goes vacuously GREEN the moment its pattern stops
matching. This matcher cannot: a failed extraction yields an empty key set,
and ``set(TONES.values()) <= set()`` is false, so a broken reader turns the
guard RED rather than quiet. It also carries the positive and negative
controls below, which is the remedy INV-10 asks for.
"""

from __future__ import annotations

import pathlib
import re

from _dashboard_helpers import strip_js_comments, walk_balanced

from dashboard.data.census import TONES

CHARTS_JSX = (
    pathlib.Path(__file__).parent.parent / 'src' / 'dashboard' / 'static' / 'redux' / 'charts.jsx'
)

PALETTE_RE = re.compile(r'const\s+PALETTE\s*=\s*\{')

# The key at the head of an object-literal entry, in all three spellings JS
# allows: bare identifier, single-quoted, double-quoted. Anchored on `{` or `,`
# so a colon inside a value cannot read as a key.
KEY_RE = re.compile(r"""(?:^|[{,])\s*(?:'([\w-]+)'|"([\w-]+)"|([A-Za-z_$][\w$]*))\s*:""")


def outer_level(object_literal):
    """*object_literal* with every NESTED object blanked, leaving its own keys.

    PALETTE nests two sub-objects, ``status`` and ``paths``, whose keys are
    not tone keys — ``PALETTE.status['in-progress']`` is a different lookup
    from ``PALETTE[tone]`` — so a flat key scan would wrongly accept them.
    Each nested character becomes a SPACE rather than vanishing, so two
    previously separated tokens can never splice into a new match (the
    ``strip_js_comments`` convention).
    """
    out, depth = [], 0
    for char in object_literal:
        if char == '{':
            depth += 1
            out.append(char if depth == 1 else ' ')
        elif char == '}':
            out.append(char if depth == 1 else ' ')
            depth -= 1
        else:
            out.append(char if depth <= 1 else ' ')
    return ''.join(out)


def palette_tone_keys():
    """The TOP-LEVEL keys of charts.jsx's ``PALETTE`` object literal.

    Comments are blanked first, so a key named only in prose cannot read as
    declared. ``walk_balanced`` does not skip braces inside string literals;
    that is acceptable here because every PALETTE value is a quoted
    ``oklch(...)`` colour or a nested object, and none embeds a brace.
    """
    source = strip_js_comments(CHARTS_JSX.read_text(encoding='utf-8'))
    match = PALETTE_RE.search(source)
    assert match is not None, f'{CHARTS_JSX.name} no longer declares `const PALETTE = {{`'
    literal = walk_balanced(source, match.end() - 1)
    assert literal, f'{CHARTS_JSX.name}: the `const PALETTE = {{` literal is never closed'
    return {
        name for entry in KEY_RE.finditer(outer_level(literal)) for name in entry.groups() if name
    }


def test_the_palette_reader_finds_top_level_keys_and_not_nested_ones():
    """A control on the matcher, so this file cannot pass by reading nothing.

    `status` and `paths` are keys OF the palette; the keys INSIDE them index a
    different lookup and must never be mistaken for tone keys.
    """
    keys = palette_tone_keys()
    assert {'accent', 'ok', 'warn', 'bad', 'info', 'fg3', 'stranded', 'status', 'paths'} <= keys
    assert 'in-progress' not in keys
    assert 'one-pass' not in keys


def test_every_tone_names_a_real_palette_key():
    """A renamed or typo'd key is `undefined` at the consumer — no colour at all."""
    keys = palette_tone_keys()
    unresolved = {member.value: tone for member, tone in TONES.items() if tone not in keys}
    assert not unresolved, (
        f'these census.TONES values are not top-level charts.jsx::PALETTE keys: '
        f'{unresolved}. PALETTE declares {sorted(keys)}'
    )
