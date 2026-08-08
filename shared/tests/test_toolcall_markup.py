"""Tests for shared.toolcall_markup — envelope-markup detector and repairer.

PRD ``plans/toolcall-markup-containment-prd.md`` task alpha, contract C1.

TDD pair 1 (step 1/2): the SINGLE literal enumeration (INV-5) and ``detect``.
TDD pair 2 (step 3/4): ``repair`` over the four PRD section 2.1 specimens.
TDD pair 3 (step 5/6): the refusal boundary and the four C1 invariants.

## Sentinel-literal hazard — DO NOT "helpfully" un-escape these

Every envelope literal in this file is spelled with the ``\\x3c`` escape for
``<``, exactly as ``fused_memory/utils/toolcall_xml_leak.py`` lines 77-86
require. Writing ``<`` verbatim here would force any agent editing this file
to emit that literal inside its own tool-call envelope, reproducing the very
defect these tests pin. ``\\x3c`` is byte-identical at runtime and never
appears verbatim in the file text. Leave it escaped.
"""
from __future__ import annotations

import pytest

from shared.toolcall_markup import (
    CANONICAL_OPENER_PREFIX,
    ENVELOPE_LITERALS,
    INVOKE_CLOSER,
    MCP_MARKUP_PATTERNS,
    PARAMETER_CLOSER_NAMES,
    PREFILTER_NEEDLES,
    closer_for,
    detect,
)

# The two calibrations as they are spelled TODAY at their current owners. These
# are the byte-exact pins that make the promotion into shared a move rather than
# a rewrite; PRD section 7 puts re-litigating the write-time/read-time split out
# of scope, so neither tuple's value may change under this task.
_TRIPWIRE_TUPLE_TODAY = ('\x3c/content>', '\x3cparameter name=', '\x3c/invoke>')
_PREFILTER_TUPLE_TODAY = (
    '\x3c/description>',
    '\x3c/parameter>',
    '\x3c/details>',
    '\x3c/content>',
)


class TestSingleLiteralEnumeration:
    """INV-5: one literal set, two named predicates derived from it."""

    def test_one_enumeration_of_closer_names(self):
        assert PARAMETER_CLOSER_NAMES == ('description', 'parameter', 'details', 'content')
        assert INVOKE_CLOSER == '\x3c/invoke>'
        assert CANONICAL_OPENER_PREFIX == '\x3cparameter name='

    def test_write_time_tuple_is_byte_exact(self):
        """MCP_MARKUP_PATTERNS keeps markup_tripwire.py's value AND its order."""
        assert MCP_MARKUP_PATTERNS == _TRIPWIRE_TUPLE_TODAY

    def test_read_time_tuple_is_byte_exact_and_ordered(self):
        """PREFILTER_NEEDLES keeps toolcall_xml_leak.py's value AND its ORDER.

        Order is load-bearing, not cosmetic: fused-memory/tests/test_mem0_client.py
        zips this tuple against the Qdrant filter clauses with ``strict=True``.
        """
        assert PREFILTER_NEEDLES == _PREFILTER_TUPLE_TODAY

    def test_prefilter_is_derived_from_the_name_tuple(self):
        """Each needle is the closer built from the one name enumeration."""
        for name, needle in zip(PARAMETER_CLOSER_NAMES, PREFILTER_NEEDLES, strict=True):
            assert needle == closer_for(name)

    def test_write_time_tuple_is_derived_from_the_same_set(self):
        """The tripwire tuple re-uses the shared literals, it does not respell them."""
        assert MCP_MARKUP_PATTERNS[0] in PREFILTER_NEEDLES
        assert MCP_MARKUP_PATTERNS[1] == CANONICAL_OPENER_PREFIX
        assert MCP_MARKUP_PATTERNS[2] == INVOKE_CLOSER

    def test_both_predicates_are_tuples_of_str(self):
        for tup in (MCP_MARKUP_PATTERNS, PREFILTER_NEEDLES, PARAMETER_CLOSER_NAMES):
            assert isinstance(tup, tuple)
            assert all(isinstance(item, str) for item in tup)

    def test_envelope_literals_is_the_union(self):
        assert isinstance(ENVELOPE_LITERALS, tuple)
        for literal in (*MCP_MARKUP_PATTERNS, *PREFILTER_NEEDLES):
            assert literal in ENVELOPE_LITERALS
        assert len(ENVELOPE_LITERALS) == len(set(ENVELOPE_LITERALS))


class TestDetect:
    """detect() is find_markup_pattern generalised to the whole literal set."""

    def test_reports_the_earliest_literal_by_text_position(self):
        """First BY POSITION IN THE TEXT, not by position in the tuple.

        ``\\x3c/description>`` is PREFILTER_NEEDLES[0] but appears LATER here,
        so a tuple-order-first implementation would report the wrong offset.
        """
        value = 'lead \x3c/details> middle \x3c/description> tail'
        assert detect(value) == '\x3c/details>'

    def test_earliest_wins_across_the_two_calibrations(self):
        """The union is scanned as one set — a needle can beat a tripwire literal."""
        value = 'lead \x3c/invoke> middle \x3c/content> tail'
        assert detect(value) == INVOKE_CLOSER

    def test_reports_the_canonical_opener_prefix(self):
        value = 'body \x3cparameter name="priority">low'
        assert detect(value) == CANONICAL_OPENER_PREFIX

    def test_is_case_sensitive(self):
        """The harness emits lowercase tags; case-folding would only widen onto prose."""
        assert detect('body \x3c/CONTENT> tail') is None

    @pytest.mark.parametrize(
        'value',
        [
            None,
            '',
            0,
            17,
            {'content': 'x'},
            ['\x3c/invoke>'],
            b'\x3c/invoke>',
            'ordinary prose with no envelope markup at all',
            'prose mentioning a parameter and some content but no tags',
        ],
    )
    def test_returns_none_without_raising(self, value):
        assert detect(value) is None
