"""Unit tests for the MCP-markup write tripwire (task 3141).

Covers the pure matchers, the structured block builder, the rolling-window
storm counter (with an injected clock — no sleeping) and the best-effort
escalation emitter in
:mod:`fused_memory.server.markup_tripwire`.

Boundary/wiring tests for the four MCP write tools live in the sibling
``test_markup_tripwire_gate.py``.
"""

from __future__ import annotations

from fused_memory.server.markup_tripwire import (
    MCP_MARKUP_PATTERNS,
    find_markup_pattern,
    find_markup_violation,
)

# ---------------------------------------------------------------------------
# Tier-1 same-file drift guard.
#
# _CANONICAL_PATTERNS and MCP_MARKUP_PATTERNS must be updated TOGETHER.  The
# write-time pattern list is deliberately the SINGLE source of truth (INV-5);
# this guard makes an unannounced edit to it fail loudly here rather than
# silently widening/narrowing what the four MCP write boundaries reject.
#
# Pattern of tests/test_lock_charter_guard.py::_CANONICAL_EXTENSIONS.
# ---------------------------------------------------------------------------
_CANONICAL_PATTERNS = ('</content>', '<parameter name=', '</invoke>')


def test_pattern_list_drift_guard():
    """Tier-1 (same-file): MCP_MARKUP_PATTERNS must match _CANONICAL_PATTERNS.

    A same-file consistency check — update BOTH together.  The three literals
    are the envelope fragments observed leaking into the corpus (DF 3083
    vector-1 specimens are ``</content>``/``</invoke>`` tails; vector-2 is the
    ``<parameter name=`` fragment that mis-parsed task 3210's priority).
    """
    assert MCP_MARKUP_PATTERNS == _CANONICAL_PATTERNS, (
        f'Write-time pattern list drifted: {MCP_MARKUP_PATTERNS!r} != '
        f'{_CANONICAL_PATTERNS!r}. Update both together.'
    )


def test_pattern_list_is_an_immutable_tuple():
    """The pattern list is a tuple, so a caller cannot mutate the guard's rules."""
    assert isinstance(MCP_MARKUP_PATTERNS, tuple), (
        f'Expected a tuple, got {type(MCP_MARKUP_PATTERNS)}'
    )


class TestFindMarkupPattern:
    """The single-text matcher: first matching literal by position, else None."""

    def test_matches_each_canonical_pattern(self):
        """Each of the three literals is detected and returned verbatim.

        The matched literal is echoed back (rather than a bare bool) so the
        rejection can name WHICH pattern tripped — INV-1.
        """
        for pattern in _CANONICAL_PATTERNS:
            text = f'some prose then {pattern} and a tail'
            assert find_markup_pattern(text) == pattern, (
                f'Expected {pattern!r} to be detected in {text!r}'
            )

    def test_matches_a_bare_pattern_with_no_surrounding_prose(self):
        for pattern in _CANONICAL_PATTERNS:
            assert find_markup_pattern(pattern) == pattern

    def test_returns_first_match_by_position_not_list_order(self):
        """With several patterns present, the EARLIEST in the text wins.

        ``</invoke>`` is last in MCP_MARKUP_PATTERNS but first in this text, so
        a naive "iterate the list, return the first that appears" impl would
        report ``</content>`` instead and mislead the caller about where the
        leak starts.
        """
        text = 'head </invoke> middle </content> tail'
        assert find_markup_pattern(text) == '</invoke>'

        # And the mirror case, so the assertion above cannot pass by accident.
        text_reversed = 'head </content> middle </invoke> tail'
        assert find_markup_pattern(text_reversed) == '</content>'

    def test_returns_first_match_by_position_across_all_three(self):
        text = 'a <parameter name="x"> b </content> c </invoke> d'
        assert find_markup_pattern(text) == '<parameter name='

    def test_returns_none_for_clean_prose(self):
        assert find_markup_pattern('a perfectly ordinary sentence about tasks') is None

    def test_returns_none_for_similar_but_non_matching_markup(self):
        """Nearby-but-different markup does not trip the guard."""
        assert find_markup_pattern('<content> and <invoke> and <parameter>') is None

    def test_matching_is_case_sensitive(self):
        """Uppercase variants do NOT match — the harness emits lowercase tags.

        Case-folding would widen the guard to prose that shouts the tag name
        without buying any real recall: no observed specimen is uppercased.
        """
        assert find_markup_pattern('a leaked </INVOKE> tail') is None
        assert find_markup_pattern('a leaked </CONTENT> tail') is None
        assert find_markup_pattern('a leaked <PARAMETER NAME=') is None

    def test_returns_none_for_none_input(self):
        """None passes through without raising — callers hand us raw handler args."""
        assert find_markup_pattern(None) is None

    def test_returns_none_for_empty_string(self):
        assert find_markup_pattern('') is None

    def test_returns_none_for_non_str_input(self):
        """Non-str input never raises — an optional handler arg may be anything."""
        for value in (123, 4.5, True, [], {}, object()):
            assert find_markup_pattern(value) is None, f'Expected None for {value!r}'


class TestFindMarkupViolation:
    """The multi-field matcher: (field_name, matched_pattern) or None."""

    def test_returns_field_and_pattern_for_a_single_violating_field(self):
        result = find_markup_violation({'description': 'leaked </invoke> here'})
        assert result == ('description', '</invoke>')

    def test_returns_first_violating_field_in_insertion_order(self):
        """Dict insertion order decides, so call sites control reporting priority.

        Both fields are dirty; ``title`` was inserted first, so it is named.
        """
        fields = {
            'title': 'title with </content> in it',
            'description': 'description with </invoke> in it',
        }
        assert find_markup_violation(fields) == ('title', '</content>')

    def test_skips_clean_fields_to_reach_the_violating_one(self):
        fields = {
            'title': 'a clean title',
            'description': 'also clean',
            'details': 'dirty <parameter name="priority">',
            'prompt': 'clean too',
        }
        assert find_markup_violation(fields) == ('details', '<parameter name=')

    def test_returns_none_when_every_field_is_clean(self):
        fields = {
            'title': 'a clean title',
            'description': 'a clean description',
            'details': 'clean details',
            'prompt': 'a clean prompt',
        }
        assert find_markup_violation(fields) is None

    def test_returns_none_for_empty_and_none_field_values(self):
        """Absent/empty optional fields are not violations and never raise."""
        assert find_markup_violation({'title': None, 'description': '', 'details': None}) is None

    def test_returns_none_for_an_empty_field_map(self):
        assert find_markup_violation({}) is None

    def test_ignores_non_str_field_values(self):
        """A non-str value (e.g. a dict metadata blob) is skipped, not coerced."""
        fields = {'title': 12345, 'description': ['</invoke>'], 'details': 'clean'}
        assert find_markup_violation(fields) is None
