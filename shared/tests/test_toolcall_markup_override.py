"""Contract of the ``allow_mcp_markup`` override lifecycle in shared.toolcall_markup (task 3689).

Contract C2 of ``plans/toolcall-markup-containment-prd.md``.
"""
from __future__ import annotations

import json

import pytest

from shared.toolcall_markup import (
    CANONICAL_OPENER_PREFIX,
    MARKUP_OVERRIDE_KEY,
    closer_for,
    markup_override_requested,
    strip_markup_override,
    swallowed_markup_override,
)


class TestMarkupOverrideLifecycle:
    """The ``allow_mcp_markup`` opt-in, at its new single home (task 3689).

    Promoted from ``fused_memory.server.markup_tripwire``. ``MarkupGuardMiddleware``
    (contract C2, boundary row B6) must PRESERVE today's markup_tripwire
    semantics, and a second implementation of the override is precisely the
    no-lockstep-duplication violation (INV-5) this PRD exists to end — so the
    middleware calls the code that DEFINES those semantics rather than
    re-describing them.

    These assertions therefore pin the EXISTING behaviour exactly; nothing here
    is new contract.
    """

    def test_the_key_is_unchanged(self):
        assert MARKUP_OVERRIDE_KEY == 'allow_mcp_markup'

    # -- (b) fail-closed: ONLY a literal boolean True ---------------------

    def test_literal_true_enables_the_override(self):
        assert markup_override_requested({MARKUP_OVERRIDE_KEY: True}) is True

    @pytest.mark.parametrize(
        'value',
        ['yes', 'True', 'true', 1, 1.0, [1], {'a': 1}, 'allow', False, 0, None, ''],
        ids=repr,
    )
    def test_truthy_but_not_true_does_not_enable_it(self, value):
        """Fail-closed, mirroring add_memory's ``allow_near_duplicate is True``.

        A truthy-but-not-``True`` value is far more likely to be unrelated data
        than a considered decision to write raw envelope markup — and the
        failure mode being contained, an accidental serialization leak, never
        sets an explicit flag at all.
        """
        assert markup_override_requested({MARKUP_OVERRIDE_KEY: value}) is False

    def test_an_absent_key_does_not_enable_it(self):
        assert markup_override_requested({}) is False
        assert markup_override_requested({'other': True}) is False

    # -- (c) dict OR JSON string, never raising ---------------------------

    def test_accepts_a_json_string(self):
        assert markup_override_requested('{"allow_mcp_markup": true}') is True
        assert markup_override_requested('{"allow_mcp_markup": "yes"}') is False

    @pytest.mark.parametrize(
        'metadata',
        [
            None,
            '',
            'not json at all',
            '{"unterminated": ',
            '[1, 2, 3]',           # valid JSON, not a dict
            '"a bare string"',     # valid JSON, not a dict
            '42',
            42,
            3.5,
            True,
            ['allow_mcp_markup'],
            # Bare sentinel: an explicit id, because repr(object()) embeds a heap
            # address that differs per process and makes pytest-xdist abort the
            # whole suite on a collection-consistency mismatch.
            pytest.param(object(), id='<bare object()>'),
        ],
        ids=repr,
    )
    def test_never_raises_and_fails_closed_on_any_other_input(self, metadata):
        assert markup_override_requested(metadata) is False

    # -- (d) strip is shape-preserving and NON-mutating --------------------

    def test_strips_from_a_dict_returning_a_dict(self):
        out = strip_markup_override({MARKUP_OVERRIDE_KEY: True, 'keep': 1})

        assert out == {'keep': 1}
        assert isinstance(out, dict)

    def test_strips_from_a_json_string_returning_a_json_string(self):
        out = strip_markup_override('{"allow_mcp_markup": true, "keep": 1}')

        assert isinstance(out, str)
        assert json.loads(out) == {'keep': 1}

    def test_does_not_mutate_the_callers_own_dict(self):
        """The handler may still need the original; quietly mutating
        caller-owned metadata is action-at-a-distance this guard must not
        introduce."""
        original = {MARKUP_OVERRIDE_KEY: True, 'keep': 1}

        strip_markup_override(original)

        assert original == {MARKUP_OVERRIDE_KEY: True, 'keep': 1}

    def test_stripping_is_the_only_change(self):
        original = {MARKUP_OVERRIDE_KEY: True, 'a': 1, 'b': [2], 'c': {'d': 3}}

        assert strip_markup_override(original) == {'a': 1, 'b': [2], 'c': {'d': 3}}

    # -- (e) unparseable input passes straight through ---------------------

    @pytest.mark.parametrize(
        'metadata',
        [None, '', 'not json at all', '[1, 2, 3]', 42, 3.5],
        ids=repr,
    )
    def test_unparseable_input_passes_through_unchanged(self, metadata):
        assert strip_markup_override(metadata) == metadata

    def test_a_dict_without_the_key_is_returned_as_is(self):
        original = {'keep': 1}

        assert strip_markup_override(original) is original

    def test_a_json_string_without_the_key_is_returned_byte_identical(self):
        """Not merely equivalent: an untouched string must not be re-serialized,
        or a caller round-tripping metadata would see spurious key reordering."""
        original = '{"keep": 1,   "other":  2}'

        assert strip_markup_override(original) is original


class TestMarkupOverrideReExport:
    """fused_memory.server.markup_tripwire keeps exposing all three names.

    The re-export contract, in the same shape task 3688 established for
    ``MCP_MARKUP_PATTERNS``: promote to ``shared``, re-export from the old
    home, public names unchanged, so ``server/tools.py``'s imports and
    fused-memory's own suite need no edit.

    Guarded by importorskip so shared's suite stays independent of fused-memory
    being installed — shared is the base layer and may not require it.
    """

    def test_the_old_home_still_exposes_all_three_names(self):
        tripwire = pytest.importorskip('fused_memory.server.markup_tripwire')

        assert tripwire.MARKUP_OVERRIDE_KEY is MARKUP_OVERRIDE_KEY
        assert tripwire.markup_override_requested is markup_override_requested
        assert tripwire.strip_markup_override is strip_markup_override


#: What the caller meant to send for the leaking parameter.
_INTENDED = 'The guard repairs at the boundary.'

#: The wrong-dialect closer that starts the leak: the parameter closed on its
#: own name.
_SELF_NAME_MISCLOSE = closer_for('rationale')


def _canonical(name: str, body: str) -> str:
    """A canonical-dialect pseudo-parameter, as the over-consume leaves it."""
    return CANONICAL_OPENER_PREFIX + '"' + name + '">' + body + closer_for('parameter')


def _echo(name: str, body: str) -> str:
    """A name-echo-dialect pseudo-parameter."""
    return '\x3c' + name + '>' + body + closer_for(name)


def _leaked(*tail: str) -> str:
    """A value that absorbed every pseudo-parameter in *tail* after a mis-close."""
    return _INTENDED + _SELF_NAME_MISCLOSE + '\n' + '\n'.join(tail)


class TestSwallowedMarkupOverride:
    """``swallowed_markup_override``: the override found INSIDE a leaked value.

    The leak swallows the parameters that follow the leaking one, so a
    ``metadata`` override sent after it arrives as text inside the value. This
    predicate reports that; it never honours it. Only a literal boolean
    ``True`` counts, as for :func:`markup_override_requested`.
    """

    def test_a_canonical_metadata_carrying_the_flag_is_found(self):
        value = _leaked(_canonical('metadata', json.dumps({MARKUP_OVERRIDE_KEY: True})))

        assert swallowed_markup_override(value) is True

    def test_a_name_echo_metadata_carrying_the_flag_is_found(self):
        value = _leaked(_echo('metadata', json.dumps({MARKUP_OVERRIDE_KEY: True})))

        assert swallowed_markup_override(value) is True

    def test_the_flag_beside_a_sibling_pseudo_parameter_is_found(self):
        value = _leaked(
            _canonical('alternatives', 'none'),
            _canonical('metadata', json.dumps({MARKUP_OVERRIDE_KEY: True})),
        )

        assert swallowed_markup_override(value) is True

    @pytest.mark.parametrize(
        'body',
        [
            json.dumps({MARKUP_OVERRIDE_KEY: False}),
            json.dumps({MARKUP_OVERRIDE_KEY: 'true'}),
            json.dumps({MARKUP_OVERRIDE_KEY: 1}),
            json.dumps({'other': True}),
            json.dumps({}),
        ],
        ids=repr,
    )
    def test_anything_but_a_literal_true_is_not_an_override(self, body):
        assert swallowed_markup_override(_leaked(_canonical('metadata', body))) is False

    def test_a_leak_without_a_metadata_pseudo_parameter_is_not_an_override(self):
        value = _leaked(_canonical('alternatives', json.dumps({MARKUP_OVERRIDE_KEY: True})))

        assert swallowed_markup_override(value) is False

    def test_a_clean_value_is_not_an_override(self):
        assert swallowed_markup_override(_INTENDED) is False

    @pytest.mark.parametrize(
        'body',
        ['not json at all', '{"unterminated": ', '[true]', 'true'],
        ids=repr,
    )
    def test_a_metadata_body_that_is_not_a_json_map_is_not_an_override(self, body):
        assert swallowed_markup_override(_leaked(_canonical('metadata', body))) is False

    @pytest.mark.parametrize(
        'value',
        [None, 42, 3.5, True, b'bytes', ['metadata'], {MARKUP_OVERRIDE_KEY: True}],
        ids=repr,
    )
    def test_a_non_str_value_is_not_an_override_and_never_raises(self, value):
        assert swallowed_markup_override(value) is False
