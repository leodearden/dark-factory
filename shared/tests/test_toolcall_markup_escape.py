"""Contract of shared.toolcall_markup.escape_envelope_literals (task 5907).

Text that deliberately QUOTES tool-call envelope literals as data — the
legibility census's sighting evidence is the motivating case — is refused by
the boundary markup guard when it is written through a guarded MCP tool. The
helper neutralises exactly the envelope-shaped tokens, and nothing else, by
re-spelling each one's leading ``<`` as the four-character text ``\\x3c``.

## Sentinel-literal hazard — DO NOT "helpfully" un-escape these

Every envelope literal in this file is spelled with the ``\\x3c`` escape for
``<`` or built from the module's public constants; the rule and its rationale
are owned by ``shared/src/shared/toolcall_markup.py`` ("Sentinel-literal
hazard"). Expected ESCAPED outputs are spelled with a doubled backslash, so the
four-character escape text is what the source holds.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from shared.toolcall_markup import (
    CANONICAL_OPENER_PREFIX,
    ENVELOPE_LITERALS,
    INVOKE_CLOSER,
    closer_for,
    detect,
    detect_for,
    escape_envelope_literals,
)

_ESCAPE = '\\x3c'
_INCIDENT_SELF_NAMES = ('rationale', 'how', 'decision', 'what', 'title')


@pytest.mark.parametrize('literal', ENVELOPE_LITERALS)
def test_each_fixed_literal_escapes_only_its_leading_bracket(literal):
    text = 'before ' + literal + ' after'

    result = escape_envelope_literals(text)

    assert detect(result) is None
    assert result == 'before ' + _ESCAPE + literal[1:] + ' after'


def test_invoke_closer_escapes_to_the_house_spelling():
    assert escape_envelope_literals('x ' + INVOKE_CLOSER + ' y') == 'x \\x3c/invoke> y'


@pytest.mark.parametrize('name', _INCIDENT_SELF_NAMES)
def test_self_name_closers_pass_the_widened_gate(name):
    result = escape_envelope_literals('quoted ' + closer_for(name) + ' here')

    assert detect_for(result, name, (name,)) is None
    assert result == 'quoted ' + _ESCAPE + '/' + name + '> here'


def test_dialect_blend_closer_is_escaped():
    result = escape_envelope_literals('x \x3c/rationale' + '"' + '> y')

    assert result == 'x \\x3c/rationale"> y'
    assert detect_for(result, 'rationale', ('rationale',)) is None


def test_benign_brackets_survive_and_escape_round_trips():
    benign_comparison = 'a ' + chr(60) + ' b'
    echo_opener = '\x3cdiv>'
    spaced_slash = '\x3c/ spaced'
    text = ' '.join((
        benign_comparison,
        INVOKE_CLOSER,
        echo_opener,
        closer_for('content'),
        spaced_slash,
        closer_for('rationale'),
    ))

    result = escape_envelope_literals(text)

    assert result.replace(_ESCAPE, chr(60)) == text
    assert benign_comparison in result
    assert echo_opener in result
    assert spaced_slash in result
    assert detect(result) is None


@pytest.mark.parametrize('text', ['', 'plain prose', 'a ' + chr(60) + ' b > c'])
def test_text_without_envelope_literals_is_unchanged(text):
    assert escape_envelope_literals(text) == text


def test_escaping_is_idempotent():
    text = (
        'saw ' + INVOKE_CLOSER + ' then ' + closer_for('content')
        + ' and ' + closer_for('rationale') + ' and ' + CANONICAL_OPENER_PREFIX
        + '"what">'
    )

    once = escape_envelope_literals(text)

    assert escape_envelope_literals(once) == once


def test_canonical_opener_escapes_only_its_leading_bracket():
    text = CANONICAL_OPENER_PREFIX + '"description">'

    result = escape_envelope_literals(text)

    assert result == _ESCAPE + CANONICAL_OPENER_PREFIX[1:] + '"description">'
    assert detect(result) is None


def test_this_module_spells_no_raw_envelope_literal():
    """This file's own SOURCE must never contain a raw ``chr(60)`` + ``/``."""
    needle = chr(60) + '/'
    source = Path(__file__).read_text(encoding='utf-8')
    assert needle not in source, (
        'A raw envelope literal was written into this test file. Spell it with '
        'the \\x3c escape instead — see this module\'s docstring for why.'
    )
