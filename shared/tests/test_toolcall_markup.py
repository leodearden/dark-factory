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
from pathlib import Path

import pytest

from shared import toolcall_markup
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
    detect_for,
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



class TestParameterAwareDetection:
    """``detect_for`` — the FOURTH named predicate, and the one the GATES need.

    Task **4696**. :func:`detect` scans a FIXED six-literal union that echoes
    no INVOKED TOOL'S OWN parameter names, while :func:`repair` has always
    qualified a candidate on ``X == param`` / ``X in schema_params``. That
    asymmetry — a schema-aware repairer behind a schema-blind detector — WAS
    the silent write path: a value mis-closed with its own parameter's tag
    matched no literal, every gate that asked ``detect`` first returned
    ``None``, and the corrupt value went straight to disk unrepaired even
    though the repairer standing behind the gate could already fix it.

    Measured over ``.worktrees/.task-meta/*/plan.json`` (2026-08-25): 444
    corrupted entries, of which **212 (48%) are invisible to the fixed literal
    set** — and **212 of 212** of those are caught by the SELF-NAME closer
    alone. That measurement is why the new predicate takes ``param``
    positionally and ``schema_params`` optionally: the self-name half is what
    the dominant population needs, and the schema half is free at the two
    call sites that already hold a schema.

    ``detect`` is deliberately NOT changed and NOT given a keyword argument:
    three call sites legitimately have no parameter in hand (``repair``'s own
    diagnostic, the sweep's bare list items, prose scans), and an optional
    keyword would make each site's blindness ungreppable. The predicates split
    by NAME so a reader can see which gates are parameter-aware.
    """

    #: The dominant real dialect, as it lands on disk today: plan-tools writes
    #: ``rationale``, and the model closes it with its own name-echoing tag.
    _SILENT_RATIONALE = (
        'Both mechanisms partition rather than race.' + closer_for('rationale') + '\n'
    )
    #: The second-commonest, 129 specimens: ``add_reuse_item``'s ``how``.
    _SILENT_HOW = 'Reuse the declared table directly.' + closer_for('how')

    def test_the_silent_write_specimen_is_invisible_to_detect(self):
        """The defect itself, pinned before the fix so it cannot be re-argued.

        ``rationale`` is a parameter of ``add_design_decision`` but not a
        member of ``PARAMETER_CLOSER_NAMES``, so its closer is in no literal
        the blanket predicate scans.
        """
        assert closer_for('rationale') not in ENVELOPE_LITERALS
        assert detect(self._SILENT_RATIONALE) is None

    def test_detect_for_sees_the_self_name_closer(self):
        assert detect_for(self._SILENT_RATIONALE, 'rationale') == closer_for('rationale')

    def test_the_how_dialect_is_the_same_shape(self):
        assert detect(self._SILENT_HOW) is None
        assert detect_for(self._SILENT_HOW, 'how') == closer_for('how')

    def test_a_foreign_param_does_not_widen_onto_the_self_name_closer(self):
        """Only the INVOKED parameter's own closer is added, not every name."""
        assert detect_for(self._SILENT_RATIONALE, 'title') is None

    @pytest.mark.parametrize('literal', ENVELOPE_LITERALS)
    def test_is_a_strict_superset_of_detect(self, literal):
        """Every fixed literal detect() reports is still reported, unchanged.

        Widening may only ADD needles. A parameter name unrelated to any
        literal must not shadow, reorder or suppress the existing set.
        """
        value = 'lead ' + literal + ' tail'
        assert detect(value) == literal
        assert detect_for(value, 'unrelated_param') == detect(value)
        assert detect_for(value, 'unrelated_param', ('other', 'names')) == detect(value)

    def test_earliest_by_text_position_when_the_self_name_closer_LEADS(self):
        """The added needle wins when it comes first, exactly as detect()'s rule says."""
        value = 'a ' + closer_for('rationale') + ' b ' + INVOKE_CLOSER
        assert detect(value) == INVOKE_CLOSER
        assert detect_for(value, 'rationale') == closer_for('rationale')

    def test_earliest_by_text_position_when_the_fixed_literal_LEADS(self):
        """...and loses when it comes second. Position, never tuple order."""
        value = 'a ' + INVOKE_CLOSER + ' b ' + closer_for('rationale')
        assert detect_for(value, 'rationale') == INVOKE_CLOSER

    def test_a_param_whose_closer_is_ALREADY_a_literal_is_unchanged(self):
        """``content`` is both a plan-tools field and PREFILTER_NEEDLES[3]."""
        value = 'body' + closer_for('content')
        assert detect_for(value, 'content') == detect(value) == closer_for('content')

    def test_schema_params_widens_the_set(self):
        """A CROSS-FIELD misclose: the closer names a SIBLING parameter."""
        value = 'Chose X.' + closer_for('decision')
        assert detect(value) is None
        assert detect_for(value, 'rationale') is None
        assert detect_for(value, 'rationale', ()) is None
        assert detect_for(
            value, 'rationale', ('task_id', 'decision', 'rationale')
        ) == closer_for('decision')

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
        ],
    )
    def test_totality_matches_detect_for_non_text(self, value):
        assert detect_for(value, 'rationale') is None
        assert detect_for(value, 'rationale', ('decision',)) is None

    @pytest.mark.parametrize('schema_params', [None, 0, 17, object(), b'decision'])
    def test_a_non_iterable_schema_params_degrades_to_the_param_alone(self, schema_params):
        assert detect_for(self._SILENT_RATIONALE, 'rationale', schema_params) == closer_for(
            'rationale'
        )
        assert detect_for('Chose X.' + closer_for('decision'), 'rationale', schema_params) is None

    def test_a_bare_str_schema_params_never_iterates_into_CHARACTERS(self):
        """The same fail-safe ``_as_name_set`` gives ``repair``.

        A caller passing the parameter NAME where a collection belongs is a
        bug; reading it as one-letter names would manufacture needles like the
        closer for ``r`` out of the string ``rationale``.
        """
        assert detect_for('text' + closer_for('r'), 'rationale', 'rationale') is None
        assert detect_for(self._SILENT_RATIONALE, 'rationale', 'rationale') == closer_for(
            'rationale'
        )

    @pytest.mark.parametrize('param', [None, '', 0, 17, b'rationale', {'a': 1}])
    def test_a_missing_or_non_string_param_degrades_to_exactly_detect(self, param):
        """Never to a DEGENERATE EMPTY-NAME TAG, which would match prose."""
        assert detect_for('\x3c/>', param) is None
        assert detect_for('body ' + INVOKE_CLOSER, param) == INVOKE_CLOSER
        assert detect_for('plain prose', param) is None
        assert detect_for(self._SILENT_RATIONALE, param) is None


class TestTheWidenedGateCostsWhatItClaims:
    """The COST contract of ``detect_for``, on the 99.7%-clean path.

    This class reaches module internals, which is normally an interface smell.
    It is the deliberate exception: cost is not observable through the public
    interface — ``detect_for`` returns the same answer whether it allocates
    three frozensets per call or none — so a contract about allocation and
    caching can only be stated against the mechanism. Everything about the
    ANSWER stays pinned through the public predicate in the class above.

    Why it is worth pinning at all: this predicate sits on a per-tool-call
    boundary and returns ``None`` for 99.7% of the values it sees, so the
    whole of its cost on the dominant path is setup that finds nothing. The
    rows below pin the three properties that keep that setup bounded.
    """

    #: Reused from the class above, so the cost rows and the answer rows are
    #: measured against the same specimen rather than two that could drift.
    _SILENT_RATIONALE = TestParameterAwareDetection._SILENT_RATIONALE

    def test_the_normalization_is_cached_across_identical_calls(self):
        """(1) Repeated identical calls normalize ONCE.

        Asserted on ``cache_info()`` deltas, never on wall-clock: a timing
        assertion on a shared machine is a flake generator, and the property
        that matters is "did it do the work again", which the counters answer
        exactly.
        """
        toolcall_markup._extra_names.cache_clear()

        for _ in range(20):
            detect_for(self._SILENT_RATIONALE, 'rationale', ('decision', 'rationale'))

        info = toolcall_markup._extra_names.cache_info()
        assert info.misses == 1, 'the normalization ran once for twenty calls'
        assert info.hits == 19

    def test_a_param_already_in_the_literal_set_reaches_the_module_pattern(self):
        """(2) The zero-allocation short-circuit, asserted by IDENTITY.

        ``content``'s closer is already in :data:`ENVELOPE_LITERALS`, so with
        no schema there is nothing to add and the widened set is empty. An
        empty set must resolve to the module-level ``_ENVELOPE_RE`` OBJECT —
        the identity guarantee ``_widened_re``'s own docstring already makes —
        so the widest-used call shape compiles nothing and allocates nothing
        beyond the two cache lookups.
        """
        names = toolcall_markup._extra_names('content', frozenset())

        assert names == frozenset(), 'a closer already in the set is not re-added'
        assert toolcall_markup._widened_re(names) is toolcall_markup._ENVELOPE_RE

        # ...and the public answer is unchanged by the short-circuit.
        value = 'body' + closer_for('content')
        assert detect_for(value, 'content') == detect(value) == closer_for('content')

    @pytest.mark.parametrize(
        'param',
        ['not-an-identifier', 'a b', '9lives', 'a/b', 'a.b', 'a-b', 'has\nnewline'],
    )
    def test_a_param_outside_the_tag_name_shape_never_becomes_a_needle(self, param):
        """(3) The cache-thrash bound, and the coherence argument behind it.

        ``param`` is CALLER-CONTROLLED: ``_first_markup_argument`` passes each
        key of the caller's ``arguments`` mapping straight through, so a caller
        sending unknown argument names would otherwise evict the bounded
        ``_widened_re`` cache with a fresh ``re.compile`` per name.

        The bound is the ``_TAG_NAME`` identifier shape rather than a bigger
        cache, because it answers something stronger: ``repair`` qualifies a
        mis-close candidate through ``_CLOSER_RE``, whose name group is
        ``[A-Za-z_]\\w*``. A needle built for a name outside that shape can be
        DETECTED and can never be QUALIFIED for repair, so spelling it would
        manufacture detections that are unrepairable by construction — routing
        authored text into the human queue for nothing.
        """
        value = 'prose ' + closer_for(param) + ' tail ' + INVOKE_CLOSER

        assert detect_for(value, param) == detect(value) == INVOKE_CLOSER
        assert param not in toolcall_markup._extra_names(param, frozenset())

    def test_an_identifier_param_is_still_widened_onto(self):
        """The bound's other side: a REAL parameter name is never dropped.

        MCP parameter names are Python function parameters and are therefore
        already identifiers, which is why the bound costs nothing in coverage.
        """
        assert detect_for(self._SILENT_RATIONALE, 'rationale') == closer_for('rationale')
        assert toolcall_markup._extra_names('rationale', frozenset()) == frozenset(
            {'rationale'}
        )

    def test_a_schema_name_outside_the_shape_is_dropped_too(self):
        """One rule, not two: the bound is on every name that becomes a needle.

        ``schema_params`` is not the caller-controlled vector — it is resolved
        from the invoked tool's own schema — but it lands in the same widened
        set and therefore the same cache key, and ``repair`` cannot qualify a
        non-identifier from it either. Filtering in one place keeps the gate's
        widening vocabulary exactly equal to the repairer's candidate grammar.
        """
        names = toolcall_markup._extra_names('rationale', frozenset({'a-b', 'decision'}))

        assert names == frozenset({'rationale', 'decision'})


def test_this_module_spells_no_raw_envelope_literal():
    """This file's own SOURCE must never contain a raw ``chr(60)`` + ``/``.

    The mechanical half of the authoring-hazard note in the module docstring
    above, promoted here from ``scripts/tests/test_sweep_toolcall_markup.py``
    by task **4696** so every file this containment work touches carries the
    same guard. Computed at runtime from :func:`chr` so the needle itself is
    not spelled here either — a test that had to write the literal to check
    for it would be the very hazard it guards.
    """
    needle = chr(60) + '/'
    source = Path(__file__).read_text(encoding='utf-8')
    assert needle not in source, (
        'A raw envelope literal was written into this test file. Spell it with '
        'the \\x3c escape instead — see this module\'s docstring for why.'
    )


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

#: Two MALFORMED closing sequences: a slash where the tag name should start,
#: and the degenerate empty-name tag. Both carry the two-character sequence the
#: repairer's cheap prefilter scans for, and NEITHER matches the closing-tag
#: grammar, whose name must be an identifier. That is the shape the narrowed
#: boundary row B5 rule deliberately does not widen onto — see
#: ``TestQuotedReportIsRepairable`` negative control (e).
_MALFORMED_CLOSERS = '\x3c/ note> and \x3c/>'


def _invoke_opener(tool: str) -> str:
    """The opening ``invoke`` tag that heads a whole tool-call block.

    Not an envelope LITERAL (only the closing half is), but the head of a
    following block is what distinguishes a genuinely doubly-corrupted tail
    from prose that merely quotes markup — see ``TestQuotedReportIsRepairable``
    negative control (a).
    """
    return '\x3cinvoke name="' + tool + '">'


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

    def test_pattern_names_the_HEAD_of_the_leak_not_the_literal_trailing_it(self):
        """PRD section 2.2's diagnostic ambiguity, and its resolution.

        ``/rationale`` is a real drift and is not in the FIXED literal set, so
        this specimen used to report ``pattern`` as the trailing invoke closer
        — earliest by text position among the fixed literals, and about 60
        characters downstream of where the envelope actually starts. That is
        section 2.2's complaint verbatim: a guard reporting whatever follows.

        MOVED BY TASK 5283 (expectation ``INVOKE_CLOSER`` -> the ``rationale``
        closer). ``pattern`` is now derived from ``detect_for`` on the same
        ``(value, param, schema_params)`` triple the candidate qualification
        above already uses, so it names the earliest needle over the literals
        WIDENED by those names — here the self-name closer that opens the leak.
        ``misclose`` is unchanged and still reports the tag that went wrong;
        the two coincide on this specimen because the head of the leak IS the
        mis-close, which is the common case rather than a special one.
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
        assert result.pattern == _closer('rationale')
        assert value.index(result.pattern) < value.index(INVOKE_CLOSER), (
            'the reported pattern must be the HEAD of the leak — the trailing '
            'invoke closer is what this row used to report'
        )
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


class TestQuotationIsNotATruncation:
    """The QUOTATION GUARD (task 4696 review). Prose that ENDS by quoting a
    sibling's tag pair must come back BYTE-IDENTICAL, never truncated.

    ``detect_for`` widened the gate with *param*'s own closer AND every
    ``schema_params`` sibling's, and ``repair`` accepts an EMPTY tail (a
    candidate closer at end-of-string recovers ``{}`` and still returns
    ``clean_value = value[:candidate.start()]``). Composed, those two facts made
    any value legitimately ending in a sibling's closing tag a silent
    TRUNCATION reported as ``repaired`` — in a repo whose plans and escalation
    records routinely quote this very markup.

    The discriminator is EVIDENCE, not breadth: an empty tail recovers nothing,
    so there is no absorbed argument and the "repair" is pure text loss. It
    stays legal for a SELF-NAME closer (PRD boundary row B4, and the whole
    212-of-212 population the 2026-08-25 census measured) and for the fixed
    literal set, which has always been repaired here. It is refused only for a
    name the WIDENING contributed — whose genuine cross-field population that
    same census puts at ZERO.
    """

    def test_prose_ending_in_a_sibling_closer_is_returned_unrepaired(self):
        """THE NEGATIVE CONTROL. A value quoting a sibling's pair is not a leak.

        Reproduced end-to-end before the fix: this returned a Repair whose
        ``clean_value`` dropped the trailing closer AND the closing half of the
        author's quotation, with ``recovered == {}`` — i.e. it destroyed text
        and recovered nothing, while reporting success.
        """
        value = (
            'The harness emits '
            + _opener('priority') + 'high' + _closer('priority')
        )

        assert repair(
            value,
            param='title',
            schema_params=_SUBMIT_TASK_PARAMS,
            supplied=frozenset(),
        ) is None

    def test_the_widened_gate_still_SEES_it_so_it_reaches_adjudication(self):
        """Refusing is not narrowing DETECTION. The value still trips the gate,
        so the caller reports it (plan-tools ``unrepairable``, the sweep
        ``refused``) into the human queue rather than silently rewriting it."""
        value = (
            'The harness emits '
            + _opener('priority') + 'high' + _closer('priority')
        )

        assert detect_for(value, 'title', _SUBMIT_TASK_PARAMS) is not None

    def test_a_self_name_closer_at_end_of_string_is_still_repaired(self):
        """PRD boundary row B4 is UNCHANGED — the guard is scoped to ``name !=
        param``. This is the dialect the whole task exists to repair."""
        clean = 'The reconciler re-reads the plan on every pass.'

        result = repair(
            clean + _closer('description'),
            param='description',
            schema_params=_SUBMIT_TASK_PARAMS,
            supplied=frozenset(),
        )

        assert result is not None
        assert result.clean_value == clean
        assert result.recovered == {}

    def test_a_fixed_literal_at_end_of_string_is_still_repaired(self):
        """The guard exempts :data:`ENVELOPE_LITERALS`, so nothing this task
        touched changed for the set ``detect`` already spelled.

        DELIBERATELY SCOPED. Prose ending in a fixed literal (``\x3c/content>``,
        ``\x3c/description>``, ...) has been truncated here since long before
        task 4696, under the blanket ``detect`` gate. Exempting the fixed set
        keeps this fix to the surface THIS task introduced; re-litigating the
        fixed set's calibration is PRD section 7 out-of-scope."""
        clean = 'Only the escalation-watcher path is scoped).'

        result = repair(
            clean + _CANONICAL_CLOSER,
            param='title',
            schema_params=_SUBMIT_TASK_PARAMS,
            supplied=frozenset(),
        )

        assert result is not None
        assert result.clean_value == clean

    def test_a_REAL_cross_field_leak_with_a_tail_is_still_repaired(self):
        """The guard keys on the EMPTY tail, not on cross-field-ness. A genuine
        absorbed argument carries a tail, parses, and is recovered as before —
        so this is not option (a)'s blanket narrowing of the gate."""
        clean = 'The reconciler re-reads the plan on every pass.'
        value = (
            clean
            + _closer('description') + '\n'
            + _opener('priority') + 'high' + _closer('priority')
        )

        result = repair(
            value,
            param='title',
            schema_params=_SUBMIT_TASK_PARAMS,
            supplied=frozenset(),
        )

        assert result is not None
        assert result.clean_value == clean
        assert result.recovered == {'priority': 'high'}



class TestQuotedReportIsRepairable:
    """Task **4502**: a report that QUOTES a leak pattern is still repairable.

    PRD boundary row B5 refuses a tail whose recovered item is "itself doubly
    corrupted, so its boundary is a guess". Its implementation was a BARE
    SUBSTRING test — any closing-tag opening sequence anywhere in a recovered
    item's value — which is strictly wider than that stated intent. The shape
    it over-refuses is a faithful REPORT of a markup leak: such a report
    necessarily quotes the pattern that tripped the tripwire (the
    ``matched_pattern=...`` field of an escalation record), so the quote lands
    inside the swallowed ``evidence`` argument and B5 fires on the caller's own
    prose.

    MEASURED POPULATION at this task's HEAD, so the carve-out's size is on the
    record rather than assumed small: committed-corpus record
    ``toolu_01XbCz5NFCA6pCvmseyqFgvy`` plus the two ``esc-3514`` specimens
    (``escalation/tests/fixtures/markup_specimens/``) — and those are the SAME
    underlying leaked call, so it is one call and its two filings.

    The specimen below is hand-authored from the parsed-input column the way
    ``TestRepairSpecimens``' S1-S4 are, NOT copied out of a fixture, so it
    documents the SHAPE rather than one captured byte string.

    Negative controls (a)-(c) pass BOTH before and after the narrowing. They
    exist because the naive rule — ambiguity alone, or schema membership alone
    — breaks exactly there, and it is far cheaper to read that as a red test
    than to rediscover it as a corpus surprise.

    Control (d) is different in kind and is NOT a both-ways control: it FAILED
    on the narrowing as first written (esc-4502-3) and passes only with the
    fix. It pins the dialect mirror of (a), which (a) does not reach. Keep the
    pair together — the gap existed precisely because one pairing was pinned
    and its mirror was not.
    """

    # escalate_info's eleven parameters, and the five the corrupted call
    # actually arrived with. Both siblings recovered below are disjoint from
    # the supplied set, which is what stops the assertion going vacuous:
    # repair() refuses any candidate whose recovered names intersect supplied.
    _ESCALATE_INFO_PARAMS = frozenset(
        {
            'task_id',
            'agent_role',
            'category',
            'summary',
            'detail',
            'suggested_action',
            'evidence',
            'severity',
            'terminal_state_is_the_bug',
            'workflow_state',
            'worktree',
        }
    )
    _SUPPLIED = frozenset({'task_id', 'agent_role', 'category', 'summary', 'detail'})

    # The prose the caller meant to send as `detail`, and the two arguments the
    # harness parser dropped into its tail. The evidence entry quotes the
    # content closer verbatim, exactly as a real leak report does.
    _CLEAN = (
        'The write-time tripwire fired on a memory body that had absorbed its '
        'siblings. Recording the raw observation here so the population is '
        'countable.'
    )
    _ACTION = 'Re-file the escalation once the guard is narrowed; no data was lost.'
    _EVIDENCE = (
        '[{"observation": "tripwire matched_pattern=' + _closer('content')
        + ', agent_id=claude-task-4502", "measured_at": "HEAD=b2035cf8c6", '
        '"ref": "rerun#1"}]'
    )

    def _specimen(self) -> str:
        """The leaked call as the harness delivered it, in one string.

        Canonical dialect throughout: ``detail``'s prose is mis-closed with the
        canonical ``parameter`` closer, a well-formed ``suggested_action`` pair
        follows, and ``evidence`` is a FINAL UNTERMINATED opener whose value
        runs to end-of-string — the parser consumed its closer as the
        terminator. That last value is the one that quotes a literal.
        """
        return (
            self._CLEAN
            + _CANONICAL_CLOSER + '\n'
            + _canonical_opener('suggested_action') + self._ACTION
            + _CANONICAL_CLOSER + '\n'
            + _canonical_opener('evidence') + self._EVIDENCE + '\n'
            + INVOKE_CLOSER
        )

    def _repair(self, value: str) -> Repair | None:
        return repair(
            value,
            param='detail',
            schema_params=self._ESCALATE_INFO_PARAMS,
            supplied=self._SUPPLIED,
        )

    def test_the_quoted_report_recovers_both_dropped_siblings(self):
        """THE RED ASSERTION. Returns None today; must return a Repair.

        Both dropped arguments are real caller text — 261 characters of
        recommendation and a full evidence array in the live specimens — that
        the current guard drops on the floor while reporting ``unrepairable``.
        """
        result = self._repair(self._specimen())

        assert result is not None
        assert set(result.recovered) == {'suggested_action', 'evidence'}
        assert result.recovered['suggested_action'] == self._ACTION
        assert result.recovered['evidence'] == self._EVIDENCE

    def test_the_recovered_evidence_still_QUOTES_the_literal_verbatim(self):
        """The point of the carve-out, stated as an assertion.

        A recovered value is the caller's OWN text — invariant D5 guarantees it
        is a verbatim substring of the input — so it may legitimately contain a
        literal. That is categorically different from ``clean_value``, which is
        the value the guard REWROTE and whose envelope-free post-condition is
        contract C1's and is unchanged by this task (pinned just below).
        """
        result = self._repair(self._specimen())

        assert result is not None
        assert _closer('content') in result.recovered['evidence']

    def test_clean_value_is_the_prose_prefix_and_stays_envelope_free(self):
        """C1's post-condition, UNCHANGED. Stated against ``detect_for``, the
        parameter-aware predicate the gates actually consume."""
        result = self._repair(self._specimen())

        assert result is not None
        assert result.clean_value == self._CLEAN
        assert detect_for(result.clean_value, 'detail', self._ESCALATE_INFO_PARAMS) is None

    def test_d5_structural_invariants_hold(self):
        """The same non-circular predicate the 504-record corpus replay uses,
        so the new carve-out cannot pass here while failing there."""
        value = self._specimen()

        assert_repair_invariants(value, self._repair(value))

    # -- negative controls -------------------------------------------------

    def test_a_cross_dialect_self_close_is_still_refused(self):
        """NEGATIVE CONTROL (a) — the shape of committed-corpus record 25.

        ``rationale`` opens in the CANONICAL dialect but closes with the
        name-echoing ``rationale`` closer, and is followed by an invoke closer
        and then the head of a whole NEXT invoke block ending in an
        unterminated opener. An ambiguity probe alone does not catch this (the
        residue does not itself parse as pseudo-parameters), so a narrowing
        that qualified inner closers only on ambiguity — or only on schema
        membership — would ACCEPT it and silently swallow the next tool call's
        fragment into the recovered ``rationale``. That is the
        no-silent-partial-repair failure this module exists to prevent, and a
        strictly worse outcome than the ``None`` returned here.

        An item's OWN closing tag appearing inside its value is a cross-dialect
        mis-close by definition, never prose about itself — which is why the
        rule may state that condition categorically.
        """
        clean = 'Recording the rationale for the routing change.'
        value = (
            clean
            + _CANONICAL_CLOSER + '\n'
            + _canonical_opener('rationale')
            + 'The scheduler indexes merge markers instead of shelling out.'
            + _closer('rationale')
            + INVOKE_CLOSER + '\n'
            + _invoke_opener('mcp__plan-tools__add_design_decision')
            + _canonical_opener('decision')
            + 'Index the markers.'
        )

        assert repair(
            value,
            param='decision',
            schema_params=frozenset({'decision', 'rationale', 'task_id'}),
            supplied=frozenset({'task_id', 'decision'}),
        ) is None

    def test_an_invoke_closer_inside_a_recovered_value_is_still_refused(self):
        """NEGATIVE CONTROL (b) — same reason, stated on ``invoke`` alone.

        ``_parse_tail`` strips ONE trailing invoke closer as the terminator it
        expects; a SECOND one inside an item's value means the tail spans a
        tool-call boundary, so the item's end is a guess and recovery would
        swallow whatever follows.
        """
        clean = 'The reconciler re-reads the plan on every pass.'
        value = (
            clean
            + _closer('description') + '\n'
            + _opener('priority') + 'high'
            + INVOKE_CLOSER
            + ' trailing text from the next block'
        )

        assert repair(
            value,
            param='description',
            schema_params=_SUBMIT_TASK_PARAMS,
            supplied=frozenset({'project_root', 'title', 'description'}),
        ) is None

    def test_boundary_row_b5s_own_ambiguous_case_is_still_refused(self):
        """NEGATIVE CONTROL (c) — B5's ORIGINAL case, unchanged.

        Referenced by SHAPE rather than duplicated: this is exactly
        ``TestRepairRefuses::test_doubly_corrupted_tail_is_refused`` — a
        name-echoing ``agent_id`` item closed by the ``details`` closer, where
        reading that closer as the terminator ALSO yields a valid parse of the
        remainder. That is the genuine ambiguity B5 was written for ("its
        boundary is a guess"), it is NOT quoted prose, and it stays refused.
        Asserted here too so the two halves of the narrowed rule — own-name and
        alternative-boundary — are both pinned inside this class.
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

    def test_an_echo_opened_item_closed_canonically_is_still_refused(self):
        """NEGATIVE CONTROL (d) — the MIRROR of (a), and a real regression.

        Control (a) pins canonical-opener/echo-closer. This pins the opposite
        pairing, echo-opener/CANONICAL-closer, which (a) does not reach. The
        asymmetry was live: condition (i) tested ``inner_name in (name,
        closer_name)``, and for an ECHO-dialect item ``closer_name`` IS the
        item's own name, so the tuple collapsed to one entry and the canonical
        ``parameter`` closer fell out of the block set entirely.

        MEASURED before the fix (esc-4502-3): this exact value recovered
        ``agent_id`` as ``'claude-interactive'`` + the canonical closer + the
        whole trailing paragraph, reported as ``outcome=repaired`` — so under
        FORWARD_REPAIR a corrupt ``agent_id`` carrying the head of the NEXT
        tool call went straight into the tool's arguments. Probe (ii) cannot
        save this: the trailing prose does not itself parse as pseudo-
        parameters, which is the same argument control (a) makes.

        The fix lists ``parameter`` categorically, independent of the item's
        opener dialect, because ``_parse_body`` treats the canonical closer as
        a UNIVERSAL terminator — a property of the parser, not of the opener.
        """
        clean = 'The reconciler re-reads the plan.'
        value = (
            clean
            + _closer('description') + '\n'
            + _opener('agent_id') + 'claude-interactive' + _CANONICAL_CLOSER
            + ' ...and then a whole paragraph of the NEXT tool call fragment'
            + ' that does not parse.'
        )

        assert repair(
            value,
            param='description',
            schema_params=_SUBMIT_TASK_PARAMS,
            supplied={'project_root', 'title', 'description'},
        ) is None

    # -- the narrowed rule's own BOUNDS -------------------------------------
    #
    # Three branches decide how far the narrowing does NOT reach: the
    # malformed-closer fallback, the inner-closer budget, and the probe's depth
    # bound. Measured with ``pytest --cov=shared.toolcall_markup`` before these
    # pins existed, all three were UNEXECUTED by the entire suite — so for a
    # change whose whole subject is loosening a safety guard, the parts that
    # bound the loosening carried no regression pin at all. Each specimen below
    # is mutation-verified: deleting the branch it names flips that specimen and
    # leaves every other test in this file green.

    def test_a_malformed_closing_sequence_alone_is_still_refused(self):
        """NEGATIVE CONTROL (e) — the prefilter fires, nothing is WELL-FORMED.

        The recovered value carries the two-character sequence the cheap
        prefilter scans for, but no closing tag that actually matches the
        grammar (a tag name must be an identifier). The alternative-boundary
        rule therefore has NOTHING to reason about, and it keeps B5's original
        answer — refuse — rather than widening the carve-out onto a shape it
        was never measured against.

        MUTATION-VERIFIED: replacing that fallback with a bare ``return False``
        flips this specimen from refused to RECOVERED, silently widening the
        carve-out in precisely the direction controls (a)-(d) exist to bound,
        while every other test in this file stays green.
        """
        value = (
            self._CLEAN
            + _CANONICAL_CLOSER + '\n'
            + _canonical_opener('evidence')
            + 'the sweep logged a malformed fragment ' + _MALFORMED_CLOSERS + ' here.'
            + '\n' + INVOKE_CLOSER
        )

        assert self._repair(value) is None

    def test_more_inner_closers_than_the_budget_is_still_refused(self):
        """NEGATIVE CONTROL (f) — past the bound the answer is BLOCK.

        At most ``_MAX_CANDIDATES`` inner closers are considered; beyond that
        the rule refuses rather than keeping a value under examination, which is
        the conservative direction and the one that makes the cost ceiling mean
        something. Every closer here is individually harmless — none names the
        item, either dialect's closer for it, or ``invoke``, and no remainder
        parses — so the budget is the ONLY thing refusing.

        The ceiling is READ from the module rather than restated: a test that
        hardcoded the number would pass vacuously the day the bound moved.

        MUTATION-VERIFIED: deleting the budget guard flips this specimen from
        refused to RECOVERED.
        """
        from shared.toolcall_markup import _MAX_CANDIDATES

        quoted = ' '.join(_closer(f'note{i}') for i in range(_MAX_CANDIDATES + 1))
        value = (
            self._CLEAN
            + _CANONICAL_CLOSER + '\n'
            + _canonical_opener('evidence')
            + 'the sweep quoted ' + quoted + ' and then prose that does not parse.'
            + '\n' + INVOKE_CLOSER
        )

        assert self._repair(value) is None

    def test_the_ambiguity_probe_does_not_recurse_and_that_is_VISIBLE(self):
        """The probe's DEPTH-1 bound, pinned by the shape whose answer it decides.

        Not a negative control: it asserts a RECOVERY, because the bound can
        only push the answer that way. The probe asks "does the remainder after
        this inner closer ALSO parse?", and at depth 1 it restores the blanket
        substring refusal — so a remainder that WOULD parse into an item whose
        own value quotes markup reads as "does not parse", condition (ii) stays
        silent, and the tail is recovered whole.

        This specimen is exactly that shape: the quoted report carries a closer
        immediately followed by a canonical opener, so the remainder is itself a
        quoting item. MUTATION-VERIFIED: deleting the ``probe`` short-circuit —
        i.e. letting the probe re-enter the narrowed rule — makes that remainder
        parse, fires condition (ii), and flips this specimen to ``None``.

        So the bound is a DESIGN CHOICE with an observable consequence rather
        than a free safety net, and moving it must be a decision rather than a
        tidy-up. What is delivered stays verbatim caller text under D5, which
        the invariant assertion states rather than assumes.
        """
        quoted_tail = (
            'the report quotes ' + _closer('foo') + ' '
            + _canonical_opener('suggested_action')
            + 'refile once the guard is narrowed ' + _closer('xyz') + ' done'
        )
        value = (
            self._CLEAN
            + _CANONICAL_CLOSER + '\n'
            + _canonical_opener('evidence') + quoted_tail
            + '\n' + INVOKE_CLOSER
        )

        result = self._repair(value)

        assert result is not None
        assert set(result.recovered) == {'evidence'}
        assert result.recovered['evidence'] == quoted_tail
        assert_repair_invariants(value, result)


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
      earlier closer is a BLEND-dialect ``priority``, which no predicate can
      spell, so the prefix stays clean and the scan's advance-and-accept
      behaviour must SURVIVE the condition;
    * :meth:`test_the_same_shape_with_an_envelope_literal_prefix_is_refused` —
      the earlier closer is ``details``, which IS an envelope literal, so every
      later candidate's prefix is poisoned and the only honest answer is None.

    Without the first case the post-condition could be "fixed" by refusing
    every value with more than one qualifying closer, which would gut the scan.

    NARROWED BY TASK **4696**, and the narrowing is the point of that task. The
    condition now asks :func:`detect_for`, so a skipped CANONICAL closer naming
    *param* or a *schema_params* member poisons the prefix too — which is
    exactly the double-self-name-misclose hole
    :class:`TestNoSilentPartialRepairOfSelfNameMisclose` below pins. The first
    case above was written with a canonical ``priority`` closer and passed only
    because ``priority`` sat outside the FIXED literal set — i.e. because of
    the same blindness 4696 exists to end, one layer in. It is preserved here
    in the blend dialect so the advance-and-accept path keeps a live pin
    instead of quietly ceasing to be exercised.
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
        parses. The prefix ``A\\x3c/priority">junk B`` carries a closing tag —
        but it is the DIALECT BLEND form, with the stray quote PRD section
        2.1's first specimen carries, and no predicate spells that: every
        needle :func:`detect_for` adds is built by ``closer_for``, which emits
        no quote. So the prefix is clean under both the fixed literal set and
        the widened one, and the repair must still be returned: the
        post-condition REFINES the scan, it does not kill it.

        The blend form is what keeps this pin ALIVE after task 4696 (see the
        class docstring). Written with a canonical ``priority`` closer it
        passed only because ``priority`` sat outside the fixed literal set —
        the very blindness that task closes — and would now, correctly, refuse.
        """
        value = (
            'A'
            + _blend_closer('priority') + 'junk B'
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
            clean_value='A' + _blend_closer('priority') + 'junk B',
            recovered={'task_id': '7'},
            pattern=_closer('description'),
            misclose=_closer('description'),
        )
        assert detect(result.clean_value) is None
        assert detect_for(
            result.clean_value, 'description', {'description', 'priority', 'task_id'}
        ) is None
        assert_repair_invariants(value, result)

    def test_a_canonical_schema_closer_in_the_prefix_is_now_refused(self):
        """The same shape with the stray quote removed. Task **4696**.

        The counterpart of the case above, kept beside it so the ONE-CHARACTER
        difference that separates accept from refuse is visible on one screen.
        ``\\x3c/priority>`` is a canonical closer for a real parameter of this
        tool, so under :func:`detect_for` the prefix is poisoned and the honest
        answer is ``None`` — the same verdict
        :meth:`test_the_same_shape_with_an_envelope_literal_prefix_is_refused`
        already reached for ``details``, now reached for the same STRUCTURAL
        reason rather than by the accident of set membership.
        """
        value = (
            'A'
            + _closer('priority') + 'junk B'
            + _closer('description')
            + _opener('task_id') + '7' + _closer('task_id')
        )

        assert repair(
            value,
            param='description',
            schema_params={'description', 'priority', 'task_id'},
            supplied={'description'},
        ) is None

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


class TestNoSilentPartialRepairOfSelfNameMisclose:
    """The DOUBLE SELF-NAME MISCLOSE — the last member of the 4696 family.

    :func:`repair`'s prefix-clean accept-time condition exists so an accepted
    ``clean_value`` can never still trip the detector; its own docstring calls
    a violation "the exact failure this module exists to end, reintroduced by
    its own repairer". But that guard called the PARAM-BLIND :func:`detect`, so
    it was blind in exactly the way every other gate was: a value mis-closed
    TWICE with its own parameter's name sailed straight through it.

    MEASURED CURRENT BEHAVIOUR on the specimen below, reproduced live at base
    ``dc5c9356``. :func:`repair` ACCEPTS it, returning::

        clean_value = 'Part one.' + closer_for('rationale') + 'GARBAGE PROSE'
        recovered   = {'decision': 'Chose X.'}

    The scan steps over the FIRST candidate, whose tail does not parse, which
    leaves that closer sitting inside the second candidate's prefix — and
    ``detect`` does not spell the ``rationale`` closer, so the poison passes.
    A value written back from that is STILL CORRUPT and has silently swallowed
    ``GARBAGE PROSE`` for good.

    THIS CANNOT REGRESS THE COMMITTED CORPUS, and that is measured rather than
    hoped. Replaying every record of
    ``shared/tests/fixtures/toolcall_markup_corpus.jsonl`` at TASK 4696's HEAD:
    **504 records, 443 accepted by repair(), and ZERO of those 443 produce a
    clean_value carrying a qualifying closer** — under the self-name-only
    widening AND under the full ``param + schema_params`` widening alike. So no
    per-specimen expectation flips and the corpus fixture was NOT edited by task
    4696.

    The accepted count is **444** as of task **4502**, which narrowed boundary
    row B5 and moved one record repaired-ward; 4696's 443 is left as it was
    measured rather than retyped, so the two figures stay attributable. The
    ZERO clause is the substantive half and was RE-VERIFIED at 4502, not merely
    restated: the newly-accepted record's ``clean_value`` carries no qualifying
    closer either, so all 444 still satisfy it.
    """

    #: The tool the specimen was captured against: ``add_design_decision``.
    _SCHEMA = ('task_id', 'decision', 'rationale')
    _SUPPLIED = ('task_id', 'rationale')

    #: Two ``rationale`` closers, prose stranded between them, and a canonical
    #: ``decision`` opener in the tail that DOES parse — so every accept-time
    #: condition except prefix-clean is satisfied and the guard is the only
    #: thing standing between this value and disk.
    _DOUBLE = (
        'Part one.'
        + _closer('rationale')
        + 'GARBAGE PROSE'
        + _closer('rationale')
        + '\n'
        + _canonical_opener('decision')
        + 'Chose X.'
    )

    def test_the_double_self_name_misclose_is_unrepairable(self):
        assert repair(self._DOUBLE, 'rationale', self._SCHEMA, self._SUPPLIED) is None

    def test_the_poisoned_prefix_is_what_makes_it_unrepairable(self):
        """Names the mechanism, so a future reader cannot mistake this for B8/B9.

        The tail parses, its one recovered name IS in the schema, and that name
        is NOT already supplied — so the candidate clears every other
        accept-time condition. Only the prefix disqualifies it.
        """
        poisoned_prefix = 'Part one.' + _closer('rationale') + 'GARBAGE PROSE'
        assert self._DOUBLE.startswith(poisoned_prefix)
        assert detect(poisoned_prefix) is None
        assert detect_for(poisoned_prefix, 'rationale') == _closer('rationale')
        assert 'decision' in self._SCHEMA
        assert 'decision' not in self._SUPPLIED

    def test_refusing_it_never_silently_swallows_the_stranded_prose(self):
        """The half of the defect that outlives the corruption itself.

        Accepting would have dropped ``GARBAGE PROSE`` permanently — it lands
        in neither ``clean_value`` nor ``recovered``. Refusing keeps the value
        byte-identical on disk, which is visible damage rather than invisible
        loss, and the unrepairable flag says so out loud.
        """
        assert repair(self._DOUBLE, 'rationale', self._SCHEMA, self._SUPPLIED) is None
        assert 'GARBAGE PROSE' in self._DOUBLE

    @pytest.mark.parametrize('param', ['rationale', 'how'])
    def test_the_SINGLE_misclose_population_still_repairs(self, param):
        """The tightening is SCOPED. This is the 212-specimen dominant class.

        A single self-name misclose has a clean prefix by construction, so the
        widened guard never fires on it and the value repairs exactly as it did
        before. If this ever went red, task 4696 would have turned the very
        population it exists to rescue into permanent damage.
        """
        value = 'The intended prose.' + _closer(param)
        result = repair(value, param, ('task_id', param), ('task_id', param))
        assert result is not None
        assert result == Repair(
            clean_value='The intended prose.',
            recovered={},
            pattern=_closer(param),
            misclose=_closer(param),
        )
        assert detect_for(result.clean_value, param, ('task_id', param)) is None
        assert_repair_invariants(value, result)

    def test_the_single_misclose_still_repairs_with_a_recovered_sibling(self):
        """The same, but with a tail that actually carries a dropped argument."""
        value = (
            'The intended prose.'
            + _closer('rationale')
            + '\n'
            + _canonical_opener('decision')
            + 'Chose X.'
        )
        result = repair(value, 'rationale', self._SCHEMA, self._SUPPLIED)
        assert result is not None
        assert result.clean_value == 'The intended prose.'
        assert result.recovered == {'decision': 'Chose X.'}
        assert detect_for(result.clean_value, 'rationale', self._SCHEMA) is None
        assert_repair_invariants(value, result)


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
