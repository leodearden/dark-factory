"""Tests for shared.toolcall_markup — envelope-markup detector and repairer.

PRD ``plans/toolcall-markup-containment-prd.md`` task alpha, contract C1.

TDD pair 1 (step 1/2): the SINGLE literal enumeration (INV-5) and ``detect``.
TDD pair 2 (step 3/4): ``repair`` over the four PRD section 2.1 specimens.
TDD pair 3 (step 5/6): the refusal boundary and the four C1 invariants.

Task **3689** (contract C2) appends the ``allow_mcp_markup`` override
lifecycle, promoted here from ``fused_memory.server.markup_tripwire`` so the
middleware's boundary row B6 preserves today's semantics by REUSING the code
that defines them rather than re-implementing it (INV-5).

## Sentinel-literal hazard — DO NOT "helpfully" un-escape these

Every envelope literal in this file is spelled with the ``\\x3c`` escape for
``<``, exactly as ``fused_memory/utils/toolcall_xml_leak.py`` lines 77-86
require. Writing ``<`` verbatim here would force any agent editing this file
to emit that literal inside its own tool-call envelope, reproducing the very
defect these tests pin. ``\\x3c`` is byte-identical at runtime and never
appears verbatim in the file text. Leave it escaped.
"""
from __future__ import annotations

import json

import pytest

from shared.toolcall_markup import (
    CANONICAL_OPENER_PREFIX,
    ENVELOPE_LITERALS,
    INVOKE_CLOSER,
    MARKUP_OVERRIDE_KEY,
    MCP_MARKUP_PATTERNS,
    PARAMETER_CLOSER_NAMES,
    PREFILTER_NEEDLES,
    Repair,
    closer_for,
    detect,
    markup_override_requested,
    repair,
    strip_markup_override,
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


def assert_repair_invariants(value: object, result: Repair | None) -> None:
    """Assert PRD section 4 C1's D5 structural invariants for one repair.

    NON-CIRCULAR BY CONSTRUCTION — which is why the committed-corpus replay
    (``test_toolcall_markup_corpus.py``) imports this exact function rather
    than re-deriving it. Both properties hold or fail independently of what any
    expectation column says, so unlike the machine-generated ``expected_outcome``
    they can actually falsify the repairer:

    * ``clean_value`` is a PREFIX of the input — the repairer never invents or
      reorders caller text;
    * every recovered value is a VERBATIM SUBSTRING of the input — the repairer
      never synthesises a value;
    * ``misclose`` sits exactly at the prefix boundary, so the three fields
      describe one consistent cut of the input rather than three guesses;
    * ``clean_value`` is ENVELOPE-FREE — ``detect(clean_value) is None``. A
      repair whose own output still trips the detector is a SILENT PARTIAL
      repair: contract C2's middleware forwards ``clean_value`` as the repaired
      argument, so a residual envelope there would re-trip the write-time
      tripwire downstream AND leave the arguments hiding in the residue
      permanently dropped, with no diagnostic. Refusing outright is the only
      honest answer; this clause is what makes that hold corpus-wide rather
      than for one hand-authored specimen.

    ``None`` (unrepairable) satisfies all four trivially and is accepted, so a
    caller can pass any result through unconditionally.
    """
    if result is None:
        return
    assert isinstance(value, str), 'a Repair can only come from a str input'
    assert value.startswith(result.clean_value), (
        'D5 violated: clean_value is not a prefix of the input'
    )
    for name, recovered_value in result.recovered.items():
        assert recovered_value in value, (
            f'D5 violated: recovered {name!r} is not a verbatim substring of the input'
        )
    assert value[len(result.clean_value):].startswith(result.misclose), (
        'D5 violated: misclose does not sit at the clean_value boundary'
    )
    residual = detect(result.clean_value)
    assert residual is None, (
        f'SILENT PARTIAL REPAIR: clean_value still carries {residual!r}; '
        'repair() must refuse rather than hand the middleware a value that '
        'still trips detect() and still swallows the caller arguments hiding '
        'in the residue'
    )


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
        assert_repair_invariants(value, result)
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
        assert_repair_invariants(value, result)
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
        assert_repair_invariants(value, result)
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
        assert_repair_invariants(value, result)
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
        assert_repair_invariants(value, result)
        assert result.misclose == _closer('rationale')
        assert result.pattern == INVOKE_CLOSER
        assert result.recovered == {'agent_id': 'claude-interactive'}


class TestRepairRefuses:
    """The refusal boundary (step 5/6). Never guess — PRD section 4 C2.

    Every case here parses far enough to be tempting. The point of the contract
    is that "far enough" is not the accept condition: a candidate that fails ANY
    of the three conditions is discarded whole and the scan advances, so there
    is no code path that emits a PARTIAL repair.
    """

    def test_b8_recovered_name_outside_the_schema_is_refused(self):
        """Boundary row B8 — the tail parses, but the name is not a parameter.

        A clean parse of a name the tool does not have means the drift was into
        something that is not a dropped argument. Recovering it would inject an
        argument the tool never declared.
        """
        clean = 'The reconciler re-reads the plan on every pass.'
        value = (
            clean
            + _closer('description') + '\n'
            + _opener('nonesuch') + 'whatever' + _closer('nonesuch')
        )

        assert repair(
            value,
            param='description',
            schema_params=_SUBMIT_TASK_PARAMS,
            supplied={'project_root', 'title', 'description'},
        ) is None

    def test_b9_collision_with_a_supplied_argument_is_refused(self):
        """Boundary row B9 — the caller's own argument is never overwritten.

        The recovered value may well be the better one, but the middleware has
        no way to know that, and silently replacing an argument the caller
        actually sent is a worse failure than refusing.
        """
        clean = 'Only the escalation-watcher path is scoped).'
        value = (
            clean
            + _CANONICAL_CLOSER + '\n'
            + _canonical_opener('agent_id') + 'escalation-watcher-l2'
        )

        assert repair(
            value,
            param='content',
            schema_params=_UPDATE_MEMORY_PARAMS,
            supplied={'memory_id', 'content', 'agent_id'},
        ) is None

    def test_leftover_text_after_the_last_pseudo_parameter_is_refused(self):
        """Zero leftover is the accept condition, not "mostly parsed".

        Trailing prose the harness never emitted means the mis-close reading is
        wrong, so the cut would take real caller text with it.
        """
        clean = 'The scheduler retries this automatically).'
        value = (
            clean
            + _closer('description') + '\n'
            + _opener('priority') + 'medium' + _closer('priority') + '\n'
            + 'and then some trailing prose no parser ever emitted'
        )

        assert repair(
            value,
            param='description',
            schema_params=_SUBMIT_TASK_PARAMS,
            supplied={'project_root', 'title', 'description'},
        ) is None

    def test_no_qualifying_candidate_closer_is_refused(self):
        """A closing tag for a name the tool does not have is not a mis-close.

        This is the ordinary-prose case that makes a naive substring scan
        useless (toolcall_xml_leak's module docstring makes the same point).
        """
        value = (
            'The audit quotes a closing tag '
            + _closer('rationale')
            + ' in prose and then just stops.'
        )

        assert repair(
            value,
            param='description',
            schema_params=_SUBMIT_TASK_PARAMS,
            supplied={'description'},
        ) is None

    def test_doubly_corrupted_tail_is_refused(self):
        """Boundary row B5 — a SECOND mis-close inside the recovered tail.

        ``agent_id`` opens but closes as ``/details``, so where its value ends
        is a guess. The one class-3 counterexample in the task-3654 sweep
        evidence (957 chars of real prose in a swallowed argument) is why this
        must refuse rather than take the longest plausible reading.
        """
        clean = 'The reconciler re-reads the plan on every pass.'
        value = (
            clean
            + _closer('description') + '\n'
            + _opener('priority') + 'medium' + _closer('priority') + '\n'
            + _opener('agent_id') + 'claude-interactive' + _closer('details') + '\n'
            + INVOKE_CLOSER
        )

        assert repair(
            value,
            param='description',
            schema_params=_SUBMIT_TASK_PARAMS,
            supplied={'project_root', 'title', 'description'},
        ) is None


# A hostile table for the totality and determinism invariants. Values only —
# the invariants are about what repair() does with a VALUE, so one fixed
# (param, schema, supplied) triple keeps the table readable.
_HOSTILE_VALUES = [
    None,
    '',
    0,
    17,
    {'content': 'x'},
    ['\x3c/invoke>'],
    b'\x3c/invoke>',
    'ordinary prose with no envelope markup at all',
    # a bare closer with no tail at all
    'a bare closer and then nothing ' + _closer('content'),
    # the value IS the invoke closer
    INVOKE_CLOSER,
    # an unterminated opener with an EMPTY name, in both dialects
    'lead ' + _closer('content') + '\x3c>orphan',
    'lead ' + _closer('content') + '\x3cparameter name="">low',
    # deeply nested pseudo-parameters that never close
    'lead ' + _closer('content') + ('\x3ca>' * 200) + 'deep',
    # the stray-quote blend appearing in the CLOSER position
    'lead ' + _closer('content') + _blend_closer('description'),
    # the blend as a well-formed pair, which SHOULD repair
    'lead ' + _closer('content') + _blend_opener('agent_id') + 'x' + _blend_closer('agent_id'),
    # 100 KB of plain text, and 100 KB followed by a real last-parameter drift
    'x' * 100_000,
    ('x' * 100_000) + _closer('content') + '\n' + INVOKE_CLOSER,
]


class TestRepairInvariants:
    """The four C1 invariants: totality, determinism, purity, D5."""

    @pytest.mark.parametrize('value', _HOSTILE_VALUES, ids=range(len(_HOSTILE_VALUES)))
    def test_never_raises_and_returns_none_or_a_repair(self, value):
        """C1: ``repair`` is pure, synchronous, and never raises for any input.

        Totality has to come from structure, not from a blanket except that
        returns None — that is signature (b) of the shared silent-fallthrough
        ratchet. A raise here is the only way this test can fail.
        """
        result = repair(
            value,
            param='content',
            schema_params=_ADD_MEMORY_PARAMS,
            supplied={'content', 'project_id'},
        )
        assert result is None or isinstance(result, Repair)

    @pytest.mark.parametrize('value', _HOSTILE_VALUES, ids=range(len(_HOSTILE_VALUES)))
    def test_is_deterministic(self, value):
        """C1: identical input ⇒ identical output. The corpus replay leans on this."""
        kwargs = {
            'param': 'content',
            'schema_params': _ADD_MEMORY_PARAMS,
            'supplied': {'content', 'project_id'},
        }
        assert repair(value, **kwargs) == repair(value, **kwargs)

    @pytest.mark.parametrize('value', _HOSTILE_VALUES, ids=range(len(_HOSTILE_VALUES)))
    def test_d5_holds_for_every_result(self, value):
        """C1 D5 over the hostile table, via the shared assertion helper."""
        result = repair(
            value,
            param='content',
            schema_params=_ADD_MEMORY_PARAMS,
            supplied={'content', 'project_id'},
        )
        assert_repair_invariants(value, result)

    def test_does_not_mutate_its_arguments(self):
        """C1 purity: the caller's schema and supplied collections are untouched."""
        schema = {'content', 'project_id', 'agent_id'}
        supplied = {'content', 'project_id'}
        value = 'lead ' + _closer('content') + '\n' + _opener('agent_id') + 'x' + _closer('agent_id')

        result = repair(value, param='content', schema_params=schema, supplied=supplied)

        assert result is not None
        assert schema == {'content', 'project_id', 'agent_id'}
        assert supplied == {'content', 'project_id'}

    def test_a_repaired_result_is_reachable_from_the_hostile_table(self):
        """Guards the table itself: an all-None table would make D5 vacuous."""
        results = [
            repair(
                value,
                param='content',
                schema_params=_ADD_MEMORY_PARAMS,
                supplied={'content', 'project_id'},
            )
            for value in _HOSTILE_VALUES
        ]
        assert any(result is not None for result in results)


class TestNoSilentPartialRepair:
    """The accept-time PREFIX-CLEAN post-condition (step 14/15).

    The candidate scan advances past a closer whose tail does not parse, which
    is right — but nothing stopped the ACCEPTED candidate's prefix from still
    containing the closer that was skipped over. When that skipped closer is
    itself an envelope literal, the returned ``clean_value`` is a value the
    middleware would forward while it still trips ``detect()``, and the
    arguments buried in the residue are dropped for good. The fix is a third
    accept-time condition, uniform with B8 and B9: reject the candidate and
    keep scanning.

    The pair below is the point. Both have the same SHAPE — an earlier
    qualifying closer whose tail leaves leftover text, then a later closer that
    parses cleanly. They must land differently, because only one of the two
    earlier closers is an envelope literal:

    * :meth:`test_earlier_candidate_rejected_then_a_later_one_accepted` — the
      earlier closer is ``priority``, a schema parameter but NOT one of
      ``PARAMETER_CLOSER_NAMES``, so the prefix stays clean and the scan's
      advance-and-accept behaviour must SURVIVE the new condition;
    * :meth:`test_the_same_shape_with_an_envelope_literal_prefix_is_refused` —
      the earlier closer is ``details``, which IS an envelope literal, so every
      later candidate's prefix is poisoned and the only honest answer is None.

    Without the first case the post-condition could be "fixed" by refusing
    every value with more than one qualifying closer, which would gut the scan.
    """

    def test_the_regression_returns_none_rather_than_a_partial_repair(self):
        """Reproduced verbatim from the review, then re-verified at this HEAD.

        ``nonesuch`` is not a submit-task-shaped parameter, so the FIRST
        ``description`` closer's tail fails the B8 schema check and the scan
        advances to the second one. Its tail parses and validates — but the
        prefix it would hand back still contains the first ``description``
        closer AND the whole ``nonesuch`` argument. Before the fix this
        returned ``Repair(clean_value='A\\x3c/description>\\x3cnonesuch>x...B',
        recovered={'priority': 'low'})``: a value that still trips ``detect``
        and has silently eaten a caller argument.
        """
        value = (
            'A'
            + _closer('description')
            + _opener('nonesuch') + 'x' + _closer('nonesuch')
            + 'B'
            + _closer('description')
            + _opener('priority') + 'low' + _closer('priority')
        )

        assert repair(
            value,
            param='description',
            schema_params={'description', 'priority'},
            supplied={'description'},
        ) is None

    def test_earlier_candidate_rejected_then_a_later_one_accepted(self):
        """The advance-and-accept path the review found untested ENTIRELY.

        ``priority`` is in the schema so its closer qualifies as a candidate,
        but its tail is the leftover text ``junk B...`` and the candidate is
        rejected. The scan advances to the ``description`` closer, whose tail
        parses. The prefix ``A\\x3c/priority>junk B`` carries a closing tag —
        but ``priority`` is not one of the four ``PARAMETER_CLOSER_NAMES``, so
        it is not an envelope literal and the prefix is clean. The repair must
        still be returned: the post-condition REFINES the scan, it does not
        kill it.
        """
        value = (
            'A'
            + _closer('priority') + 'junk B'
            + _closer('description')
            + _opener('task_id') + '7' + _closer('task_id')
        )

        result = repair(
            value,
            param='description',
            schema_params={'description', 'priority', 'task_id'},
            supplied={'description'},
        )

        assert result is not None
        assert result == Repair(
            clean_value='A' + _closer('priority') + 'junk B',
            recovered={'task_id': '7'},
            pattern=_closer('description'),
            misclose=_closer('description'),
        )
        assert detect(result.clean_value) is None
        assert_repair_invariants(value, result)

    def test_the_same_shape_with_an_envelope_literal_prefix_is_refused(self):
        """Same shape, one substitution: the skipped closer is now a literal.

        ``details`` is in the schema AND is one of ``PARAMETER_CLOSER_NAMES``,
        so once the scan steps over it every later candidate's prefix is
        poisoned — and stays poisoned, because candidate start positions only
        increase. Refuse.
        """
        value = (
            'A'
            + _closer('details') + 'junk B'
            + _closer('description')
            + _opener('task_id') + '7' + _closer('task_id')
        )

        assert repair(
            value,
            param='description',
            schema_params={'description', 'details', 'task_id'},
            supplied={'description'},
        ) is None


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
            object(),
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
