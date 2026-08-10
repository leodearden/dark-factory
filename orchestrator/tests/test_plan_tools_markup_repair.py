"""Lazy, atomic repair of envelope-markup corruption in a live plan.json.

Task 3692 (PRD ``plans/toolcall-markup-containment-prd.md``, task EPSILON,
boundary row B12). The subject under test is ``orchestrator.mcp.plan_tools``'s
read-time repair surface; the detection/repair mechanism itself is owned by
``shared.toolcall_markup`` (task 3688) and is NOT re-implemented here.

## Sentinel-literal hazard — every fixture is BUILT, never written verbatim

This module describes MCP tool-call envelope markup, so it is exactly the file
that must not contain any of that markup literally. The rationale is the one
recorded at ``shared/src/shared/toolcall_markup.py`` lines 52-62: an agent
editing a file that contains a raw envelope literal has to emit that literal
INSIDE its own tool-call argument, which reproduces the very over-consumption
defect under test — the Write/Edit argument terminates early, truncating this
file and silently dropping that call's sibling arguments.

So every specimen below is assembled at import time from :func:`_close`,
:func:`_open_param` and :data:`_INVOKE_CLOSER`, which build their angle bracket
from ``chr(60)``. The result is byte-identical at runtime and never appears
verbatim in the file text. :func:`_assert_no_raw_sentinels` enforces that on the
module's OWN BYTES at import, so a future editor cannot quietly reintroduce one
(it is a check on this file's source text, not on any docstring's wording).
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import tempfile
from pathlib import Path

import pytest
from shared.toolcall_markup import ENVELOPE_LITERALS, detect

from orchestrator.artifacts import PLAN_SCHEMA_VERSION, TaskArtifacts
from orchestrator.mcp import plan_tools

#: What a failed atomic write may raise. The helper's own declared error, plus
#: the serialization errors that fire before the temp is ever opened.
PLAN_WRITE_FAILURES = (OSError, TypeError, ValueError)

# ---------------------------------------------------------------------------
# Sentinel BUILDERS — the only way markup enters this module.
# ---------------------------------------------------------------------------

#: The opening angle bracket, spelled so it never appears verbatim in the file.
_LT = chr(60)


def _close(name: str) -> str:
    """Build the closing tag for *name* (the mis-close shape the harness emits)."""
    return _LT + '/' + name + '>'


def _open_param(name: str) -> str:
    """Build the canonical opening tag for parameter *name*."""
    return _LT + 'parameter name="' + name + '">'


#: The bare invoke closer — the terminator that trails a last-parameter leak.
_INVOKE_CLOSER = _close('invoke')


def _assert_no_raw_sentinels() -> None:
    """Fail at IMPORT if this file's own bytes carry a raw envelope literal.

    Checked against ``shared.toolcall_markup.ENVELOPE_LITERALS`` (the single
    owner of the literal set, INV-5) plus the two structural prefixes every
    built specimen uses, so a builder output spelled out by hand is caught even
    when it is not itself one of the enumerated literals.
    """
    source = Path(__file__).read_text(encoding='utf-8')
    forbidden = (*ENVELOPE_LITERALS, _LT + '/', _LT + 'parameter ')
    for sequence in forbidden:
        if sequence in source:
            raise AssertionError(
                f'{Path(__file__).name} contains a RAW envelope sentinel '
                f'({sequence!r}). Build it from _close()/_open_param() instead '
                '— a verbatim literal here corrupts the tool call that writes '
                'this file. See the module docstring.'
            )


_assert_no_raw_sentinels()


# ---------------------------------------------------------------------------
# The four REAL specimen shapes, measured on the 28 corrupted live plans.
# ---------------------------------------------------------------------------

#: Clean prose used as the intended value of whichever field a specimen poisons.
#: Deliberately free of any markup so ``detect()`` on a repaired prefix is None.
_RATIONALE_PROSE = (
    'Reusing the shared detector keeps one enumeration of the literals, so the '
    'write-time guard and the read-time repair can never drift apart.'
)
_HOW_PROSE = (
    'Imported directly; the helper does the grammar work and owns every '
    'accept/refuse decision, so this module adds no literal enumeration.'
)
_DECISION_PROSE = (
    'Repair the plan lazily on read rather than sweeping the fleet, because a '
    'sweep would have to quiesce every running task first.'
)

#: TRAILING RESIDUE on ``design_decisions[].rationale`` — the DOMINANT live
#: shape (97 of 118 corrupted strings): the parameter was last in the call, so
#: nothing was absorbed and only the mis-close plus the invoke closer trail it.
TRAILING_RATIONALE = _RATIONALE_PROSE + _close('rationale') + '\n' + _INVOKE_CLOSER + '\n'

#: The same trailing-residue shape on ``reuse[].how`` (27 of the 97).
TRAILING_HOW = _HOW_PROSE + _close('how') + '\n' + _INVOKE_CLOSER + '\n'

#: ABSORBED SIBLING on ``design_decisions[].decision``: the parser mis-closed
#: ``decision`` and then swallowed the whole ``rationale`` parameter into it, so
#: the rationale a later reader sees is another field's text (PRD section 2.4).
#: The final opener is UNTERMINATED — its closer was consumed as the terminator.
ABSORBED_RATIONALE = (
    _DECISION_PROSE + _close('decision') + '\n' + _open_param('rationale') + _RATIONALE_PROSE
)

#: PROSE FALSE POSITIVE, measured live in worktree 2939 — a plan ABOUT this leak,
#: whose authored text legitimately QUOTES the sentinels in ordinary sentences.
#: ``detect()`` fires, ``repair()`` correctly declines, and a trailing-only
#: sanitize contract would mutilate it. This is the specimen that makes the
#: repair-or-leave-byte-identical contract load-bearing rather than stylistic.
PROSE_QUOTED = (
    'The harness closes the argument with ' + _close('description') + ' or with '
    + _close('parameter') + ', and then re-opens with ' + _open_param('x')
    + ' before the next value, which is how the sibling arguments get lost.'
)


# ---------------------------------------------------------------------------
# Fixtures and plan factory.
# ---------------------------------------------------------------------------


@pytest.fixture()
def plan_artifacts(tmp_path):
    """TaskArtifacts over a temp worktree — mirrors ``test_plan_tools_server``."""
    a = TaskArtifacts(tmp_path)
    a.init('test-1', 'Test task', 'A test')
    return a


def corrupt_plan(**overrides) -> dict:
    """Return a complete, VALID plan dict whose fields can then be poisoned.

    Every call builds a fresh, independent document (no shared mutable state),
    so a test may poison ``plan['design_decisions'][0]['rationale']`` in place.
    Keyword *overrides* replace whole top-level keys, which is how a test swaps
    in its own collection (e.g. four decisions instead of the default two).

    The default document is entirely CLEAN: nothing here trips ``detect()``, so
    any fact a test observes came from the field it poisoned.
    """
    plan: dict = {
        'task_id': 'test-1',
        'title': 'A test plan',
        'analysis': 'Clean analysis prose describing the approach.',
        'files': ['orchestrator/src/orchestrator/mcp/plan_tools.py'],
        'prerequisites': [
            {
                'id': 'pre-1',
                'description': 'Clean prerequisite prose.',
                'status': 'pending',
                'commit': None,
                'tests': [],
            },
        ],
        'steps': [
            {
                'id': 'step-1',
                'type': 'test',
                'description': 'Clean step prose for the first step.',
                'status': 'pending',
                'commit': None,
            },
            {
                'id': 'step-2',
                'type': 'impl',
                'description': 'Clean step prose for the second step.',
                'status': 'pending',
                'commit': None,
            },
        ],
        'design_decisions': [
            {'decision': _DECISION_PROSE, 'rationale': 'Clean rationale prose.'},
            {'decision': 'A second clean decision.', 'rationale': 'A second clean rationale.'},
        ],
        'reuse': [
            {
                'what': 'The shared detector',
                'where': 'shared/src/shared/toolcall_markup.py',
                'how': 'Clean reuse prose.',
            },
            {
                'what': 'The plan artifact reader',
                'where': 'orchestrator/src/orchestrator/artifacts.py',
                'how': 'A second clean reuse prose.',
            },
        ],
    }
    plan.update(overrides)
    return plan


# ---------------------------------------------------------------------------
# step-1 — the repairable surface is DECLARED and machine-checked (INV-1).
# ---------------------------------------------------------------------------

#: The (collection, field) pairs measured as the corrupted surface across the
#: 28 live plans. ``collection is None`` means a top-level plan key. Held here
#: as the test's own independent statement of the contract — the point of the
#: table is that it cannot drift from this, so the two are deliberately
#: separate spellings rather than one imported from the other.
_EXPECTED_PAIRS = {
    (None, 'title'),
    (None, 'analysis'),
    ('prerequisites', 'description'),
    ('steps', 'description'),
    ('design_decisions', 'decision'),
    ('design_decisions', 'rationale'),
    ('reuse', 'what'),
    ('reuse', 'where'),
    ('reuse', 'how'),
}

#: Which plan-tools entry point AUTHORED each collection, i.e. whose parameter
#: names ``repair()`` must validate a recovery against.
_ORIGINATING_TOOL = {
    None: plan_tools._create_plan,
    'prerequisites': plan_tools._add_prerequisite,
    'steps': plan_tools._add_plan_step,
    'design_decisions': plan_tools._add_design_decision,
    'reuse': plan_tools._add_reuse_item,
}


def _tool_params(fn) -> tuple[str, ...]:
    """The tool's parameter names, minus the leading ``artifacts`` injection."""
    names = tuple(inspect.signature(fn).parameters)
    assert names[0] == 'artifacts', (
        f'{fn.__name__} no longer takes artifacts first — the table\'s '
        'schema_params derivation assumes it does'
    )
    return names[1:]


class TestRepairableFieldTable:
    """``_REPAIRABLE_PLAN_FIELDS`` is the single declared repairable surface."""

    def test_table_is_an_immutable_sequence_of_records(self):
        table = plan_tools._REPAIRABLE_PLAN_FIELDS
        assert isinstance(table, tuple), (
            'the table must be immutable — a list could be appended to at '
            'runtime, which would make the declared surface un-auditable'
        )
        assert table, 'the table must not be empty'
        for record in table:
            assert isinstance(record, tuple)
            for attr in ('collection', 'field', 'schema_params'):
                assert hasattr(record, attr), f'record {record!r} lacks {attr!r}'

    def test_covers_exactly_the_measured_corrupted_surface(self):
        table = plan_tools._REPAIRABLE_PLAN_FIELDS
        pairs = {(r.collection, r.field) for r in table}
        assert pairs == _EXPECTED_PAIRS
        # 'reuse' contributes three of the nine, so a set of pairs alone would
        # not catch a duplicated row: pin the record count too.
        assert len(table) == len(_EXPECTED_PAIRS) == 9

    def test_schema_params_match_the_live_tool_signatures(self):
        """INV-1: the table is checked against the tools, not against prose.

        A record's ``schema_params`` is what ``repair()`` validates a recovered
        name against, so a table that drifts from its tool's real signature
        would silently start refusing (or, worse, accepting) the wrong names.
        """
        for record in plan_tools._REPAIRABLE_PLAN_FIELDS:
            tool = _ORIGINATING_TOOL[record.collection]
            assert record.schema_params == _tool_params(tool), (
                f'{record.collection}.{record.field} declares '
                f'{record.schema_params!r} but {tool.__name__} takes '
                f'{_tool_params(tool)!r}'
            )

    def test_the_repaired_field_is_itself_a_parameter_of_its_tool(self):
        for record in plan_tools._REPAIRABLE_PLAN_FIELDS:
            assert record.field in record.schema_params, (
                f'{record.field!r} is not a parameter of '
                f'{_ORIGINATING_TOOL[record.collection].__name__}'
            )

    def test_schema_params_are_tuples_of_str(self):
        for record in plan_tools._REPAIRABLE_PLAN_FIELDS:
            assert isinstance(record.schema_params, tuple)
            assert all(isinstance(name, str) for name in record.schema_params)

    def test_files_is_not_repairable(self):
        """``files`` entries are paths, already recovered by ``_coerce_files``.

        A generic walk of the plan dict would sweep them in; the declared table
        is what keeps this surface to prose fields only.
        """
        fields = {r.field for r in plan_tools._REPAIRABLE_PLAN_FIELDS}
        assert 'files' not in fields


# ---------------------------------------------------------------------------
# step-3 — the pure repair pass over the DOMINANT trailing-residue shape.
# ---------------------------------------------------------------------------

#: The trailing shape on a step/prerequisite ``description``. Built here rather
#: than in the specimen block because the same field name serves two tools.
_TRAILING_DESCRIPTION = (
    'Clean step prose for the first step.' + _close('description') + '\n' + _INVOKE_CLOSER + '\n'
)
#: The trailing shape on the top-level ``analysis`` key (collection None).
_TRAILING_ANALYSIS = (
    'Clean analysis prose describing the approach.'
    + _close('analysis') + '\n' + _INVOKE_CLOSER + '\n'
)


def _all_strings(value, path=()):
    """Every str in a nested plan document, keyed by its full path."""
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _all_strings(item, (*path, key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _all_strings(item, (*path, i))


class TestRepairPlanFieldsTrailing:
    """The last-parameter / trailing-residue shape — PRD boundary row B4.

    97 of the 118 corrupted strings measured across the 28 live plans have this
    shape: the leaked parameter was LAST in the call, so nothing was absorbed
    and ``repair()`` returns ``recovered == {}``. That is a success, not a
    refusal.
    """

    def test_trailing_rationale_repairs_with_a_full_fact(self):
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = TRAILING_RATIONALE

        repaired, facts = plan_tools._repair_plan_fields(plan)

        decision = repaired['design_decisions'][0]
        assert decision['rationale'] == _RATIONALE_PROSE
        assert detect(decision['rationale']) is None
        # The sibling the tool authored in the same call is untouched.
        assert decision['decision'] == _DECISION_PROSE
        assert facts == [
            {
                'tool': 'add_design_decision',
                'param': 'rationale',
                'pattern': _INVOKE_CLOSER,
                'misclose': _close('rationale'),
                'outcome': 'repaired',
                'recovered_params': [],
                'collection': 'design_decisions',
                'index': 0,
                'field': 'rationale',
            }
        ]

    def test_is_pure__input_plan_is_not_mutated(self):
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = TRAILING_RATIONALE

        repaired, _facts = plan_tools._repair_plan_fields(plan)

        assert plan['design_decisions'][0]['rationale'] == TRAILING_RATIONALE, (
            'the caller\'s dict must not be mutated in place — the write-back '
            'decision belongs to the caller, not to the repair pass'
        )
        assert repaired is not plan
        assert repaired['design_decisions'] is not plan['design_decisions']

    def test_index_locator_is_the_items_real_position(self):
        plan = corrupt_plan()
        plan['reuse'][1]['how'] = TRAILING_HOW

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert repaired['reuse'][1]['how'] == _HOW_PROSE
        assert repaired['reuse'][0]['how'] == 'Clean reuse prose.'
        assert len(facts) == 1
        assert facts[0]['collection'] == 'reuse'
        assert facts[0]['index'] == 1
        assert facts[0]['field'] == 'how'
        assert facts[0]['tool'] == 'add_reuse_item'
        assert facts[0]['outcome'] == 'repaired'
        assert facts[0]['recovered_params'] == []
        assert facts[0]['misclose'] == _close('how')

    @pytest.mark.parametrize(
        ('collection', 'index', 'field', 'poisoned', 'clean', 'tool'),
        [
            (
                'steps', 0, 'description', _TRAILING_DESCRIPTION,
                'Clean step prose for the first step.', 'add_plan_step',
            ),
            (
                'prerequisites', 0, 'description',
                'Clean prerequisite prose.' + _close('description') + '\n' + _INVOKE_CLOSER + '\n',
                'Clean prerequisite prose.', 'add_prerequisite',
            ),
        ],
    )
    def test_trailing_shape_on_each_collection(
        self, collection, index, field, poisoned, clean, tool
    ):
        plan = corrupt_plan()
        plan[collection][index][field] = poisoned

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert repaired[collection][index][field] == clean
        assert detect(repaired[collection][index][field]) is None
        assert len(facts) == 1
        assert facts[0]['collection'] == collection
        assert facts[0]['index'] == index
        assert facts[0]['field'] == field
        assert facts[0]['tool'] == tool
        assert facts[0]['outcome'] == 'repaired'

    def test_top_level_field_reports_collection_and_index_none(self):
        plan = corrupt_plan(analysis=_TRAILING_ANALYSIS)

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert repaired['analysis'] == 'Clean analysis prose describing the approach.'
        assert len(facts) == 1
        assert facts[0]['collection'] is None
        assert facts[0]['index'] is None
        assert facts[0]['field'] == 'analysis'
        assert facts[0]['param'] == 'analysis'
        assert facts[0]['tool'] == 'create_plan'

    def test_four_corrupted_rationales_yield_four_indexed_facts(self):
        decisions = [
            {'decision': f'Decision number {i}.', 'rationale': TRAILING_RATIONALE}
            for i in range(4)
        ]
        plan = corrupt_plan(design_decisions=decisions)
        before = dict(_all_strings(plan))

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert len(facts) == 4
        assert [f['index'] for f in facts] == [0, 1, 2, 3]
        assert {f['outcome'] for f in facts} == {'repaired'}
        for item in repaired['design_decisions']:
            assert item['rationale'] == _RATIONALE_PROSE

        # Every OTHER string in the document is byte-identical.
        after = dict(_all_strings(repaired))
        assert set(after) == set(before)
        for path, value in after.items():
            if path[0] == 'design_decisions' and path[-1] == 'rationale':
                continue
            assert value == before[path], f'{path} changed but was not poisoned'

    def test_clean_plan_is_returned_unchanged_with_no_facts(self):
        plan = corrupt_plan()

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert facts == []
        assert repaired == plan

    @pytest.mark.parametrize(
        ('collection', 'index', 'field', 'poisoned'),
        [
            ('design_decisions', 0, 'rationale', TRAILING_RATIONALE),
            ('reuse', 1, 'how', TRAILING_HOW),
            ('steps', 0, 'description', _TRAILING_DESCRIPTION),
        ],
    )
    def test_d5_repaired_value_is_always_a_prefix_of_the_original(
        self, collection, index, field, poisoned
    ):
        """INVARIANT D5: recovery only ever SLICES; it never synthesises text."""
        plan = corrupt_plan()
        plan[collection][index][field] = poisoned

        repaired, facts = plan_tools._repair_plan_fields(plan)

        new_value = repaired[collection][index][field]
        assert poisoned.startswith(new_value)
        assert new_value != poisoned
        assert facts[0]['outcome'] == 'repaired'

    def test_fact_tool_names_a_real_plan_tools_entry_point(self):
        """The ``tool`` field is a live function name, not a prose label."""
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = TRAILING_RATIONALE
        plan['reuse'][0]['how'] = TRAILING_HOW
        plan['steps'][0]['description'] = _TRAILING_DESCRIPTION
        plan['analysis'] = _TRAILING_ANALYSIS

        _repaired, facts = plan_tools._repair_plan_fields(plan)

        assert len(facts) == 4
        for fact in facts:
            impl = getattr(plan_tools, '_' + fact['tool'], None)
            assert callable(impl), f'{fact["tool"]!r} is not a plan_tools function'
            assert fact['param'] in _tool_params(impl)


# ---------------------------------------------------------------------------
# step-5 — refusal is byte-identical and LOUD; never a partial or guessed fix.
# ---------------------------------------------------------------------------

#: DOUBLY CORRUPTED ``reuse[].how``: the recovered tail itself carries a second
#: mis-close, so the recovered value's own boundary would be a guess. Alpha's
#: ``_parse_tail`` refuses this by construction (PRD boundary row B5).
_DOUBLY_CORRUPT_HOW = (
    'Reuse prose that was cut short.' + _close('how') + '\n'
    + _open_param('what') + 'the absorbed value' + _close('description')
    + ' and then still more leftover text.'
)


class TestRepairPlanFieldsRefuses:
    """``repair()`` declines -> the field is left EXACTLY as it was.

    This is task 3685's open reject-vs-sanitize question, answered. There is NO
    trailing-only sanitize fallback, because the declining population is
    dominated by plans that legitimately QUOTE the sentinels in prose.
    """

    def test_prose_false_positive_is_left_byte_identical(self):
        """Worktree 2939's shape: a plan ABOUT the leak, quoting the sentinels.

        A trailing-only sanitize contract would mutilate this authored text —
        the same class of silent-wrong-value damage the PRD exists to end.
        """
        plan = corrupt_plan()
        plan['design_decisions'][0]['decision'] = PROSE_QUOTED

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert repaired['design_decisions'][0]['decision'] == PROSE_QUOTED
        # Nothing was shaved off the end: all three quoted sentinels survive.
        for quoted in (_close('description'), _close('parameter'), _open_param('x')):
            assert quoted in repaired['design_decisions'][0]['decision']
        assert len(facts) == 1
        fact = facts[0]
        assert fact['outcome'] == 'unrepairable'
        # The residue stays VISIBLE: the detected pattern is reported.
        assert fact['pattern'] == detect(PROSE_QUOTED) == _close('description')
        assert fact['recovered_params'] == []
        assert fact['misclose'] is None
        assert fact['collection'] == 'design_decisions'
        assert fact['index'] == 0
        assert fact['field'] == 'decision'
        assert fact['tool'] == 'add_design_decision'

    def test_doubly_corrupted_value_is_left_byte_identical(self):
        plan = corrupt_plan()
        plan['reuse'][0]['how'] = _DOUBLY_CORRUPT_HOW

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert repaired['reuse'][0]['how'] == _DOUBLY_CORRUPT_HOW
        assert len(facts) == 1
        assert facts[0]['outcome'] == 'unrepairable'
        assert facts[0]['pattern'] == detect(_DOUBLY_CORRUPT_HOW)
        assert facts[0]['recovered_params'] == []
        assert facts[0]['field'] == 'how'

    def test_refusal_never_partially_repairs_the_sibling_fields(self):
        plan = corrupt_plan()
        plan['design_decisions'][0]['decision'] = PROSE_QUOTED
        before = dict(_all_strings(plan))

        repaired, _facts = plan_tools._repair_plan_fields(plan)

        assert dict(_all_strings(repaired)) == before

    def _mixed_plan(self) -> dict:
        """One repairable field and one refusing field, in the same record."""
        plan = corrupt_plan()
        plan['design_decisions'][0]['decision'] = PROSE_QUOTED
        plan['design_decisions'][0]['rationale'] = TRAILING_RATIONALE
        plan['reuse'][0]['how'] = _DOUBLY_CORRUPT_HOW
        plan['reuse'][1]['how'] = TRAILING_HOW
        return plan

    def test_is_deterministic_across_repeated_runs(self):
        first_plan, first_facts = plan_tools._repair_plan_fields(self._mixed_plan())
        second_plan, second_facts = plan_tools._repair_plan_fields(self._mixed_plan())

        assert first_plan == second_plan
        assert first_facts == second_facts
        assert {f['outcome'] for f in first_facts} == {'repaired', 'unrepairable'}

    def test_converges__second_pass_repairs_nothing_and_still_refuses(self):
        """Idempotent CONVERGENCE, not oscillation.

        Feeding the result back must leave the already-repaired fields alone
        (zero further 'repaired' facts) while still reporting the residue that
        was, and remains, unrepairable.
        """
        once, first_facts = plan_tools._repair_plan_fields(self._mixed_plan())
        twice, second_facts = plan_tools._repair_plan_fields(once)

        assert twice == once
        assert [f for f in second_facts if f['outcome'] == 'repaired'] == []
        assert (
            [f for f in second_facts if f['outcome'] == 'unrepairable']
            == [f for f in first_facts if f['outcome'] == 'unrepairable']
        )

    @pytest.mark.parametrize('value', [None, 42, 3.5, [], {}, True])
    def test_non_str_values_produce_no_fact_and_no_crash(self, value):
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = value

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert facts == []
        assert repaired['design_decisions'][0]['rationale'] == value

    def test_missing_field_and_non_dict_item_produce_no_fact_and_no_crash(self):
        plan = corrupt_plan()
        del plan['design_decisions'][0]['rationale']
        del plan['analysis']
        plan['reuse'].append('not a dict at all')
        plan['steps'] = 'not a list at all'

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert facts == []
        assert 'rationale' not in repaired['design_decisions'][0]
        assert 'analysis' not in repaired
        assert repaired['reuse'][-1] == 'not a dict at all'
        assert repaired['steps'] == 'not a list at all'


# ---------------------------------------------------------------------------
# step-7 — the ABSORBED-SIBLING shape: fill a HOLE, never overwrite authored text.
# ---------------------------------------------------------------------------

#: A real, later-authored rationale, as the live worktree-3024 plans look after
#: the agent noticed the truncation and retried. At read time this and a
#: recovered tail are indistinguishable — which is precisely why one must win
#: by rule rather than by luck.
_AUTHORED_RATIONALE = (
    'The agent retried the call and this rationale is the one it actually meant '
    'to record, so nothing may clobber it.'
)


def _absorbed_decision(**record) -> dict:
    """A plan whose ``design_decisions[0]`` absorbed its rationale sibling."""
    return corrupt_plan(
        design_decisions=[{'decision': ABSORBED_RATIONALE, **record}],
    )


class TestRepairPlanFieldsAbsorbedSibling:
    """PRD section 2.4's headline damage, restated for the read-time surface.

    The rationale was ABSORBED INTO decision, so the design rationale a future
    architect reads is another field's text. Recovering it is the point of the
    task — but only ever into an EMPTY or ABSENT sibling (boundary row B9).
    """

    def test_recovers_into_an_empty_sibling(self):
        plan = _absorbed_decision(rationale='')

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['design_decisions'][0]
        assert item['decision'] == _DECISION_PROSE
        assert detect(item['decision']) is None
        assert item['rationale'] == _RATIONALE_PROSE
        # D5: the recovered text is a VERBATIM substring of the original value,
        # never synthesised.
        assert item['rationale'] in ABSORBED_RATIONALE
        assert ABSORBED_RATIONALE.startswith(item['decision'])
        assert len(facts) == 1
        assert facts[0]['outcome'] == 'repaired'
        assert facts[0]['recovered_params'] == ['rationale']
        assert facts[0]['field'] == 'decision'
        assert facts[0]['misclose'] == _close('decision')

    def test_recovers_into_a_missing_sibling__key_is_created(self):
        plan = _absorbed_decision()
        assert 'rationale' not in plan['design_decisions'][0]

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['design_decisions'][0]
        assert item['decision'] == _DECISION_PROSE
        assert item['rationale'] == _RATIONALE_PROSE
        assert len(facts) == 1
        assert facts[0]['outcome'] == 'repaired'
        assert facts[0]['recovered_params'] == ['rationale']

    def test_whitespace_only_sibling_counts_as_a_hole(self):
        plan = _absorbed_decision(rationale='   \n\t  ')

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['design_decisions'][0]
        assert item['decision'] == _DECISION_PROSE
        assert item['rationale'] == _RATIONALE_PROSE
        assert facts[0]['outcome'] == 'repaired'
        assert facts[0]['recovered_params'] == ['rationale']

    def test_never_overwrites_a_non_blank_authored_sibling(self):
        """The invariant: fill a hole, NEVER overwrite authored text.

        A recovered tail and a later-retried real value are indistinguishable
        at read time, so clobbering the authored one would be exactly the
        silent-wrong-value failure this surface exists to prevent.
        """
        plan = _absorbed_decision(rationale=_AUTHORED_RATIONALE)

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['design_decisions'][0]
        assert item['rationale'] == _AUTHORED_RATIONALE
        # The WHOLE decision field is left byte-identical — not truncated to
        # its clean prefix with the recovery merely suppressed.
        assert item['decision'] == ABSORBED_RATIONALE
        assert len(facts) == 1
        assert facts[0]['outcome'] == 'unrepairable'
        assert facts[0]['field'] == 'decision'
        assert facts[0]['recovered_params'] == []

    @pytest.mark.parametrize('sibling', ['', '   \n\t  ', None])
    def test_recovered_value_is_always_a_slice_of_the_original(self, sibling):
        plan = _absorbed_decision(rationale=sibling)

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['design_decisions'][0]
        assert facts[0]['outcome'] == 'repaired'
        assert item['rationale'] == _RATIONALE_PROSE
        assert item['decision'] in ABSORBED_RATIONALE
        assert item['rationale'] in ABSORBED_RATIONALE
        assert item['decision'] + item['rationale'] != ABSORBED_RATIONALE, (
            'the two slices must not simply re-concatenate to the input — the '
            'mis-close and the re-opened tag are DROPPED, not re-homed'
        )


# ---------------------------------------------------------------------------
# step-9 — the atomic write helper, contract C3 / boundary row B12.
# ---------------------------------------------------------------------------


class _AtomicSpies:
    """Delegating spies over the three syscalls ``_atomic_write_plan`` uses.

    Every spy DELEGATES to the real implementation, so the helper's behaviour
    is observed rather than simulated; only the call record is added.
    """

    def __init__(self, monkeypatch):
        self.mkstemp_dirs: list = []
        self.replace_calls: list = []
        self.fsync_calls: list = []
        self.order: list[str] = []
        self._real_mkstemp = tempfile.mkstemp
        self._real_replace = os.replace
        self._real_fsync = os.fsync

        monkeypatch.setattr(plan_tools.tempfile, 'mkstemp', self._mkstemp)
        monkeypatch.setattr(plan_tools.os, 'fsync', self._fsync)
        monkeypatch.setattr(plan_tools.os, 'replace', self._replace)

    def _mkstemp(self, *args, **kwargs):
        self.mkstemp_dirs.append(kwargs.get('dir'))
        self.order.append('mkstemp')
        return self._real_mkstemp(*args, **kwargs)

    def _fsync(self, fd):
        self.fsync_calls.append(fd)
        self.order.append('fsync')
        return self._real_fsync(fd)

    def _replace(self, src, dst):
        # Capture the temp file's state AT REPLACE TIME.
        src_path = Path(src)
        self.replace_calls.append({
            'src': src_path,
            'dst': Path(dst),
            'src_exists': src_path.exists(),
            'src_document': (
                json.loads(src_path.read_text()) if src_path.exists() else None
            ),
        })
        self.order.append('replace')
        return self._real_replace(src, dst)


@pytest.fixture()
def plan_dir(tmp_path):
    """A directory holding exactly one pre-existing plan.json."""
    directory = tmp_path / 'meta'
    directory.mkdir()
    (directory / 'plan.json').write_text(
        json.dumps({'task_id': 'test-1', 'steps': []}, indent=2) + '\n'
    )
    return directory


class TestAtomicWritePlan:
    """A concurrent reader must never observe a partial plan.json (B12)."""

    def test_temp_file_is_created_in_the_targets_own_directory(self, plan_dir, monkeypatch):
        """Same directory => os.replace is an intra-filesystem RENAME.

        A temp file in /tmp could land on a different filesystem, where
        os.replace raises EXDEV instead of atomically swapping.
        """
        spies = _AtomicSpies(monkeypatch)
        target = plan_dir / 'plan.json'

        plan_tools._atomic_write_plan(target, corrupt_plan())

        assert spies.mkstemp_dirs == [target.parent]

    def test_temp_is_written_and_verified_before_the_replace(self, plan_dir, monkeypatch):
        spies = _AtomicSpies(monkeypatch)
        target = plan_dir / 'plan.json'
        plan = corrupt_plan()

        plan_tools._atomic_write_plan(target, plan)

        assert len(spies.replace_calls) == 1
        call = spies.replace_calls[0]
        assert call['src_exists'], 'the temp must be fully written before replace'
        assert call['dst'] == target
        # The complete document is already on disk at replace time.
        assert call['src_document']['task_id'] == plan['task_id']
        assert call['src_document']['design_decisions'] == plan['design_decisions']

    def test_fsync_happens_before_the_replace(self, plan_dir, monkeypatch):
        spies = _AtomicSpies(monkeypatch)

        plan_tools._atomic_write_plan(plan_dir / 'plan.json', corrupt_plan())

        assert spies.fsync_calls, 'the temp must be fsynced, not just flushed'
        assert spies.order.index('fsync') < spies.order.index('replace')

    def test_writes_the_document_and_stamps_the_schema_version(self, plan_dir):
        target = plan_dir / 'plan.json'

        plan_tools._atomic_write_plan(target, corrupt_plan())

        written = json.loads(target.read_text())
        assert written['_schema_version'] == PLAN_SCHEMA_VERSION
        assert written['task_id'] == 'test-1'

    def test_byte_format_parity_with_task_artifacts_write_plan(self, tmp_path):
        """The repair write-back must not churn the file's formatting.

        Same bytes as the existing writer means a repaired plan diffs only
        where it was repaired — no reindentation, no key-order noise.
        """
        via_artifacts = tmp_path / 'artifacts'
        via_artifacts.mkdir()
        artifacts = TaskArtifacts(via_artifacts)
        artifacts.init('test-1', 'Test task', 'A test')
        artifacts.write_plan(copy.deepcopy(corrupt_plan()))

        via_atomic = tmp_path / 'atomic'
        via_atomic.mkdir()
        atomic_target = via_atomic / 'plan.json'
        plan_tools._atomic_write_plan(atomic_target, copy.deepcopy(corrupt_plan()))

        assert (
            atomic_target.read_bytes()
            == (artifacts.root / 'plan.json').read_bytes()
        )

    def test_verification_failure_leaves_the_original_untouched(self, plan_dir, monkeypatch):
        target = plan_dir / 'plan.json'
        before_bytes = target.read_bytes()
        before_entries = set(plan_dir.iterdir())

        def boom(_path):
            raise json.JSONDecodeError('injected', 'doc', 0)

        monkeypatch.setattr(plan_tools, '_verify_plan_json', boom)

        with pytest.raises(plan_tools.PlanWriteError) as excinfo:
            plan_tools._atomic_write_plan(target, corrupt_plan())

        assert str(target) in str(excinfo.value), (
            'the failure must name the path — loud, not a silent skip'
        )
        assert target.read_bytes() == before_bytes
        assert set(plan_dir.iterdir()) == before_entries, (
            'no .plan.json.*.tmp residue may be left behind'
        )

    def test_replace_failure_leaves_the_original_untouched(self, plan_dir, monkeypatch):
        target = plan_dir / 'plan.json'
        before_bytes = target.read_bytes()
        before_entries = set(plan_dir.iterdir())

        def boom(_src, _dst):
            raise OSError('injected replace failure')

        monkeypatch.setattr(plan_tools.os, 'replace', boom)

        with pytest.raises(plan_tools.PlanWriteError) as excinfo:
            plan_tools._atomic_write_plan(target, corrupt_plan())

        assert str(target) in str(excinfo.value)
        assert target.read_bytes() == before_bytes
        assert set(plan_dir.iterdir()) == before_entries

    def test_writing_a_plan_that_does_not_serialize_leaves_no_residue(self, plan_dir):
        """An unserializable payload must not strand a temp file either."""
        target = plan_dir / 'plan.json'
        before_bytes = target.read_bytes()
        before_entries = set(plan_dir.iterdir())

        with pytest.raises(PLAN_WRITE_FAILURES):
            plan_tools._atomic_write_plan(target, corrupt_plan(title=object()))

        assert target.read_bytes() == before_bytes
        assert set(plan_dir.iterdir()) == before_entries


# ---------------------------------------------------------------------------
# step-11 — the write-back must PRESERVE the lane plan symlink.
# ---------------------------------------------------------------------------


@pytest.fixture()
def lane_and_meta(tmp_path):
    """The real meta-root plan plus the absolute lane symlink onto it.

    Reproduces exactly what ``TaskArtifacts.ensure_lane_plan_symlink`` builds:
    ``<worktree>/.task/plan.json`` is an ABSOLUTE symlink to the durable
    meta-root plan, so the lane copy can never diverge from it.
    """
    meta = tmp_path / 'meta'
    meta.mkdir()
    real_plan = meta / 'plan.json'
    real_plan.write_text(json.dumps({'task_id': 'test-1'}, indent=2) + '\n')

    lane_task = tmp_path / 'worktree' / '.task'
    lane_task.mkdir(parents=True)
    lane_plan = lane_task / 'plan.json'
    os.symlink(real_plan, lane_plan)
    return lane_plan, real_plan


class TestAtomicWritePlanFollowsSymlink:
    """A naive os.replace onto the lane path would EAT the symlink.

    ``TaskArtifacts.ensure_lane_plan_symlink`` makes the lane plan an absolute
    symlink onto the meta-root copy, and ``_artifacts_from_args`` still
    supports ``meta_root=None`` where ``self.root`` IS ``<worktree>/.task`` —
    so ``self.root / 'plan.json'`` can be that symlink. Replacing it with a
    regular file would silently RE-FORK the lane and meta-root copies, which is
    the esc-5205-9 stale-plan divergence the symlink exists to make impossible.
    """

    def test_symlink_survives_and_the_real_file_receives_the_write(
        self, lane_and_meta, monkeypatch
    ):
        lane_plan, real_plan = lane_and_meta
        link_target_before = os.readlink(lane_plan)
        spies = _AtomicSpies(monkeypatch)

        plan_tools._atomic_write_plan(lane_plan, corrupt_plan(title='written'))

        assert lane_plan.is_symlink(), 'the lane symlink was replaced by a file'
        assert os.readlink(lane_plan) == link_target_before
        assert json.loads(real_plan.read_text())['title'] == 'written'
        # And the lane path still resolves to the same single source of truth.
        assert json.loads(lane_plan.read_text())['title'] == 'written'
        assert spies.replace_calls[0]['dst'] == real_plan

    def test_temp_lands_beside_the_REAL_file_not_beside_the_symlink(
        self, lane_and_meta, monkeypatch
    ):
        """The mechanism, not a side effect.

        Resolving first is what keeps the rename intra-filesystem (the lane and
        the meta root need not share one) AND what stops the replace from
        eating the link.
        """
        lane_plan, real_plan = lane_and_meta
        spies = _AtomicSpies(monkeypatch)

        plan_tools._atomic_write_plan(lane_plan, corrupt_plan())

        assert spies.mkstemp_dirs == [real_plan.parent]
        assert spies.mkstemp_dirs != [lane_plan.parent]

    def test_no_temp_residue_in_either_directory(self, lane_and_meta):
        lane_plan, real_plan = lane_and_meta

        plan_tools._atomic_write_plan(lane_plan, corrupt_plan())

        assert {p.name for p in real_plan.parent.iterdir()} == {'plan.json'}
        assert {p.name for p in lane_plan.parent.iterdir()} == {'plan.json'}

    def test_dangling_symlink_fails_loudly_without_materialising_a_file(self, tmp_path):
        """A broken link must NOT quietly become a stray regular file.

        Materialising one at the link path is the divergence itself: the lane
        would hold a plan the meta root has never seen.
        """
        lane_task = tmp_path / 'worktree' / '.task'
        lane_task.mkdir(parents=True)
        lane_plan = lane_task / 'plan.json'
        missing = tmp_path / 'gone' / 'plan.json'
        os.symlink(missing, lane_plan)
        assert lane_plan.is_symlink() and not lane_plan.exists()

        with pytest.raises(plan_tools.PlanWriteError) as excinfo:
            plan_tools._atomic_write_plan(lane_plan, corrupt_plan())

        message = str(excinfo.value)
        assert str(lane_plan) in message, 'the ORIGINAL path must be named'
        assert str(missing) in message, 'the RESOLVED path must be named too'
        assert lane_plan.is_symlink()
        assert not lane_plan.exists(), 'a stray regular file was materialised'
        assert {p.name for p in lane_task.iterdir()} == {'plan.json'}
