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
    repair,
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


# ---------------------------------------------------------------------------
# Tail-shape builders. Every envelope literal in this file goes through one of
# these, so the \x3c escape is written once per shape rather than once per use.
# ---------------------------------------------------------------------------


def _closer(name: str) -> str:
    """The name-echoing closing tag the model drifts into, e.g. description."""
    return '\x3c/' + name + '>'


def _opener(name: str) -> str:
    """The name-echoing opening tag, e.g. priority."""
    return '\x3c' + name + '>'


def _canonical_opener(name: str) -> str:
    """The canonical opening tag, e.g. parameter name="priority"."""
    return '\x3cparameter name="' + name + '">'


#: The canonical closing tag. Specimen 4's mis-close, and always a candidate.
_CANONICAL_CLOSER = '\x3c/parameter>'


def _blend_opener(name: str) -> str:
    """Specimen 1's literal DIALECT BLEND — a stray quote before the bracket.

    PRD section 2.1: ``metadata"`` is the model interpolating between the
    canonical ``parameter name="X"`` form and the name-echoing ``X`` form.
    """
    return '\x3c' + name + '">'


def _blend_closer(name: str) -> str:
    """The closing half of the dialect blend, e.g. ``/metadata"``."""
    return '\x3c/' + name + '">'


# Parameter names of the tools the four specimens were captured from. Real
# schemas, so the schema-validation accept condition is exercised honestly.
_SUBMIT_TASK_PARAMS = frozenset(
    {'project_root', 'title', 'description', 'priority', 'agent_id', 'metadata'}
)
_ADD_MEMORY_PARAMS = frozenset(
    {'content', 'project_id', 'agent_id', 'category', 'metadata', 'session_id'}
)
_UPDATE_MEMORY_PARAMS = frozenset({'memory_id', 'content', 'project_id', 'agent_id'})


class TestRepairSpecimens:
    """The four PRD section 2.1 specimens, hand-authored (step 3/4).

    These are the NON-CIRCULAR half of the correctness story: they are written
    from the PRD's parsed-input column before any corpus exists, so unlike the
    committed corpus (whose expectation column is machine-generated) they can
    actually falsify the repairer.
    """

    def test_s1_total_drift_recovers_all_three_dropped_params(self):
        """Specimen 1 / boundary row B2 — submit_task 07-30T16:47Z.

        The parser over-consumed to the trailing invoke closer, dumping three
        whole parameters into ``description``. B2 requires all three back.
        """
        clean = 'Investigate the divergence and report back on direction.'
        value = (
            clean
            + _closer('description') + '\n'
            + _opener('priority') + 'medium' + _closer('priority') + '\n'
            + _opener('agent_id') + 'claude-task-3688' + _closer('agent_id') + '\n'
            + _blend_opener('metadata')
            + '{"source": "agent-followup"}'
            + _blend_closer('metadata') + '\n'
            + INVOKE_CLOSER
        )

        result = repair(
            value,
            param='description',
            schema_params=_SUBMIT_TASK_PARAMS,
            supplied={'project_root', 'title', 'description'},
        )

        assert result is not None
        assert result.clean_value == clean
        assert result.recovered == {
            'priority': 'medium',
            'agent_id': 'claude-task-3688',
            'metadata': '{"source": "agent-followup"}',
        }
        assert result.pattern == _closer('description')
        assert result.misclose == _closer('description')

    def test_s2_partial_drift_recovers_the_unterminated_opener(self):
        """Specimen 2 / boundary row B1 — submit_task 08-04T12:13Z.

        The drift is only one parameter deep and its value is UNTERMINATED:
        the parser consumed the closer it was looking for, so ``low`` runs to
        end-of-string with nothing after it.
        """
        clean = 'The scheduler retries this automatically).'
        value = clean + _closer('description') + '\n' + _canonical_opener('priority') + 'low'

        result = repair(
            value,
            param='description',
            schema_params=_SUBMIT_TASK_PARAMS,
            supplied={'project_root', 'title', 'description'},
        )

        assert result is not None
        assert result.clean_value == clean
        assert result.recovered == {'priority': 'low'}
        assert result.pattern == _closer('description')
        assert result.misclose == _closer('description')

    def test_s3_last_parameter_drops_nothing(self):
        """Specimen 3 / boundary row B4 — add_memory 08-04T16:58Z.

        ``content`` was the LAST parameter, so nothing was dropped. The repair
        is a pure truncation: an empty ``recovered`` is a success, not a
        refusal, and the caller's text must come back untouched.
        """
        clean = 'The near-duplicate guard rejects this by design.'
        value = clean + _closer('content') + '\n' + INVOKE_CLOSER

        result = repair(
            value,
            param='content',
            schema_params=_ADD_MEMORY_PARAMS,
            supplied={'content', 'project_id'},
        )

        assert result is not None
        assert result.clean_value == clean
        assert result.recovered == {}
        assert result.pattern == _closer('content')
        assert result.misclose == _closer('content')

    def test_s4_canonical_closer_misclose(self):
        """Specimen 4 — update_memory 08-02T21:26Z, the UNGATED boundary.

        The mis-close is the CANONICAL closer, whose name (``parameter``) is
        not a parameter of update_memory. It is admitted as a candidate anyway
        — see the plan's first design decision — because otherwise one of the
        PRD's own four specimens would be unrepairable by construction.
        """
        clean = 'Only the escalation-watcher path is scoped).'
        value = (
            clean
            + _CANONICAL_CLOSER + '\n'
            + _canonical_opener('agent_id') + 'escalation-watcher-l2'
        )

        result = repair(
            value,
            param='content',
            schema_params=_UPDATE_MEMORY_PARAMS,
            supplied={'memory_id', 'content'},
        )

        assert result is not None
        assert result.clean_value == clean
        assert result.recovered == {'agent_id': 'escalation-watcher-l2'}
        assert result.pattern == _CANONICAL_CLOSER
        assert result.misclose == _CANONICAL_CLOSER

    def test_pattern_is_the_envelope_literal_misclose_is_the_wrong_tag(self):
        """The two fields differ whenever the mis-closed name is not a literal.

        PRD section 2.2's diagnostic ambiguity in miniature: ``/rationale`` is
        a real drift but is not in the literal set, so ``pattern`` reports the
        envelope literal that actually matched (the trailing invoke closer,
        earliest by text position among the literals) while ``misclose``
        reports the tag that actually went wrong.
        """
        clean = 'Because the split is calibrated in opposite directions.'
        value = (
            clean
            + _closer('rationale') + '\n'
            + _opener('agent_id') + 'claude-interactive' + _closer('agent_id') + '\n'
            + INVOKE_CLOSER
        )

        result = repair(
            value,
            param='rationale',
            schema_params={'rationale', 'agent_id'},
            supplied={'rationale'},
        )

        assert result is not None
        assert result.misclose == _closer('rationale')
        assert result.pattern == INVOKE_CLOSER
        assert result.recovered == {'agent_id': 'claude-interactive'}
