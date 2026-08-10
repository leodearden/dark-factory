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
import logging
import os
import tempfile
import threading
from collections.abc import Iterator, Mapping
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


#: Tool PARAMETERS that name no prose plan key and may therefore NEVER receive a
#: recovered string. Three of them are stored under a DIFFERENT plan key
#: (``prereq_id``/``step_id`` -> ``id``, ``step_type`` -> ``type``), and two are
#: not prose at all (``files`` is a list, ``task_id`` an identifier). Writing a
#: recovered tail into any of them is silent-wrong-value corruption of the
#: artifact that the lock charter and the merge gate both consume.
_NON_PROSE_PARAMS = frozenset({'task_id', 'files', 'prereq_id', 'step_id', 'step_type'})


def _observed_plan_keys(root) -> dict[str | None, set[str]]:
    """Build a plan through the REAL writers; return the keys they actually wrote.

    Keyed by collection (``None`` for the top-level document). This is the
    machine-check that keeps ``target_keys`` from drifting into a vocabulary the
    plan never uses — deriving the allowed key sets from the tools themselves,
    exactly as ``schema_params`` is derived from ``inspect.signature``, rather
    than restating them as a hardcoded literal a refactor could silently orphan.
    """
    artifacts = TaskArtifacts(root)
    artifacts.init('test-1', 'Test task', 'A test')
    plan_tools._create_plan(artifacts, 'test-1', 'A title.', 'An analysis.', ['a.py'])
    plan_tools._add_prerequisite(artifacts, 'pre-1', 'A prerequisite.')
    plan_tools._add_plan_step(artifacts, 'step-1', 'test', 'A step.')
    plan_tools._add_design_decision(artifacts, 'A decision.', 'A rationale.')
    plan_tools._add_reuse_item(artifacts, 'A thing', 'somewhere.py', 'By importing it.')
    plan = artifacts.read_plan()

    observed: dict[str | None, set[str]] = {None: set(plan)}
    for collection in ('prerequisites', 'steps', 'design_decisions', 'reuse'):
        items = plan[collection]
        assert items, f'{collection} came back empty — the writers did not run'
        observed[collection] = {key for item in items for key in item}
    return observed


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
            for attr in ('collection', 'field', 'schema_params', 'target_keys'):
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

    def test_files_is_neither_a_repaired_field_nor_a_recovery_target(self):
        """``files`` entries are paths, already recovered by ``_coerce_files``.

        BOTH directions matter, and only the first was ever asserted. Not being
        a repaired FIELD stops the walk from rewriting the list; not being a
        recovery TARGET stops an absorbed ``files`` tail from being written back
        as a bare ``str``, replacing the list the lock charter reads.
        """
        table = plan_tools._REPAIRABLE_PLAN_FIELDS
        assert 'files' not in {r.field for r in table}
        assert 'files' not in {name for r in table for name in r.target_keys}
        assert 'files' not in {key for r in table for key in r.target_keys.values()}

    def test_every_record_declares_an_immutable_target_keys_mapping(self):
        """A recovery target must be DECLARED, not inferred from the param name.

        ``schema_params`` is the tool's vocabulary; the plan document has its
        own. Conflating them is what let a recovered ``step_type`` land as a
        junk key on a step dict.
        """
        for record in plan_tools._REPAIRABLE_PLAN_FIELDS:
            targets = record.target_keys
            assert isinstance(targets, Mapping), (
                f'{record.collection}.{record.field}.target_keys is '
                f'{type(targets).__name__}, not a Mapping'
            )
            assert all(isinstance(k, str) and isinstance(v, str) for k, v in targets.items())
            assert not isinstance(targets, dict), (
                'target_keys must not be a plain dict — the declared surface '
                'would then be mutable at runtime and un-auditable; wrap it in '
                'types.MappingProxyType'
            )
            with pytest.raises(TypeError):
                targets['injected'] = 'anything'  # type: ignore[index]

    def test_every_target_key_names_a_real_parameter_of_its_tool(self):
        for record in plan_tools._REPAIRABLE_PLAN_FIELDS:
            invented = set(record.target_keys) - set(record.schema_params)
            assert invented == set(), (
                f'{record.collection}.{record.field} declares recovery targets '
                f'for parameters {sorted(invented)} that '
                f'{_ORIGINATING_TOOL[record.collection].__name__} never takes'
            )

    def test_every_target_value_is_a_key_the_real_writers_produce(self, tmp_path):
        """INV-1 again: the mapping is checked against a plan the TOOLS wrote.

        A target value that no writer ever produces is by definition a junk key
        — the exact defect this mapping exists to prevent — so the allowed set
        is derived, never hardcoded.
        """
        observed = _observed_plan_keys(tmp_path)

        for record in plan_tools._REPAIRABLE_PLAN_FIELDS:
            allowed = observed[record.collection]
            for param, key in record.target_keys.items():
                assert key in allowed, (
                    f'{record.collection}.{record.field} would recover {param!r} '
                    f'into {key!r}, which the writers never produce; real keys '
                    f'are {sorted(allowed)}'
                )

    def test_the_repaired_field_is_a_target_that_maps_to_itself(self):
        for record in plan_tools._REPAIRABLE_PLAN_FIELDS:
            assert record.field in record.target_keys
            assert record.target_keys[record.field] == record.field

    def test_no_non_prose_parameter_is_ever_a_recovery_target(self):
        """Identifiers, enums and lists may never receive a recovered string.

        ``prereq_id``/``step_id`` are stored under ``id`` and ``step_type``
        under ``type``, so a recovery keyed on the PARAMETER name could only
        ever create a junk key while leaving the real one corrupt.
        """
        for record in plan_tools._REPAIRABLE_PLAN_FIELDS:
            illegal = _NON_PROSE_PARAMS & set(record.target_keys)
            assert illegal == set(), (
                f'{record.collection}.{record.field} declares non-prose '
                f'recovery targets {sorted(illegal)}'
            )


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


#: A path to one string inside a plan document: dict keys and list indices.
_Path = tuple[str | int, ...]


def _all_strings(
    value, path: _Path = ()
) -> Iterator[tuple[_Path, str]]:
    """Every str in a nested plan document, keyed by its full path.

    Both the ``path`` parameter and the return are annotated rather than
    inferred. ``value`` is deliberately untyped (a plan document is arbitrary
    nested JSON), which puts this function on pyright's call-site return
    inference path, and the recursive ``yield from`` defeats that inference --
    it falls back to the bare ``()`` default's type, ``tuple[()]``. Callers
    then see an empty-tuple key, making every ``path[0]`` / ``path[-1]`` an
    out-of-range index error. The explicit return type pins the real key type:
    a mix of dict keys (str) and list indices (int).
    """
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
# step-19 — a recovery may only ever land on a REAL prose plan key.
# ---------------------------------------------------------------------------

#: An absorbed ``files`` argument on the top-level ``title``. This is the state
#: the harness leaves behind when it mis-closes ``title`` during ``create_plan``
#: and swallows the ``files`` argument: the plan's ``files`` is the empty list
#: (nothing was ever parsed into it), which is precisely what makes it look like
#: a HOLE a recovery may fill — and ``files`` is a LIST, so filling it with the
#: recovered ``str`` replaces the value the lock charter (``derive_modules`` /
#: ``files_to_modules``) and the merge gate (``plan_files_not_touched``) read.
_ABSORBED_FILES_TITLE = (
    'A real title.' + _close('title') + '\n' + _open_param('files') + 'orchestrator/src/a.py'
)

#: An absorbed ``step_type`` argument on a step ``description``. ``step_type``
#: is the TOOL's parameter name; the plan stores it as ``type``. Keyed on the
#: parameter name, the recovery creates a junk ``step_type`` key and leaves the
#: real ``type`` — the field actually corrupted — untouched.
_ABSORBED_STEP_TYPE = 'Do the thing.' + _close('description') + '\n' + _open_param('step_type') + 'test'

#: The same shape with ``prereq_id``, which the plan stores as ``id``.
_ABSORBED_PREREQ_ID = (
    'A prerequisite.' + _close('description') + '\n' + _open_param('prereq_id') + 'pre-99'
)


class TestRecoveryTargetsAreRealPlanKeys:
    """A recovered value lands on a DECLARED prose key, or nowhere at all.

    Both correctness findings share one root cause: the walk keyed recoveries
    on the TOOL's parameter names, which are not the PLAN's key names. The fix
    is not a new collision check — ``repair()`` still makes every accept/refuse
    decision (INV-5). It is that a parameter with no prose target is supplied
    UNCONDITIONALLY, so ``repair()``'s existing B9 disjointness condition
    refuses any candidate that would recover it.
    """

    def test_an_absorbed_files_argument_never_replaces_the_files_list(self):
        """Measured before the fix: ``plan['files']`` came back a ``str``."""
        plan = corrupt_plan(title=_ABSORBED_FILES_TITLE, files=[])

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert isinstance(repaired['files'], list), (
            'a recovered path string replaced the files LIST — the lock charter '
            'and the merge gate both read this value'
        )
        assert repaired['files'] == []
        assert repaired['title'] == _ABSORBED_FILES_TITLE
        assert len(facts) == 1
        assert facts[0]['outcome'] == 'unrepairable'
        assert facts[0]['field'] == 'title'
        assert facts[0]['recovered_params'] == []

    def test_a_non_empty_files_list_is_equally_untouched(self):
        original = ['orchestrator/src/orchestrator/mcp/plan_tools.py']
        plan = corrupt_plan(title=_ABSORBED_FILES_TITLE, files=list(original))

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert repaired['files'] == original
        assert repaired['title'] == _ABSORBED_FILES_TITLE
        assert facts[0]['outcome'] == 'unrepairable'

    def test_an_absorbed_step_type_creates_no_junk_key(self):
        """Measured before the fix: the step gained ``step_type: 'test'``.

        ...while ``type``, the field actually corrupted, stayed wrong — so the
        repair reported success having fixed nothing and added a key no reader
        of the plan schema expects.
        """
        step = {
            'id': 'step-1',
            'type': 'impl',
            'description': _ABSORBED_STEP_TYPE,
            'status': 'pending',
            'commit': None,
        }
        plan = corrupt_plan(steps=[step])

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['steps'][0]
        assert 'step_type' not in item, f'junk key written: {sorted(item)}'
        assert item['type'] == 'impl'
        assert item['description'] == _ABSORBED_STEP_TYPE
        assert len(facts) == 1
        assert facts[0]['outcome'] == 'unrepairable'
        assert facts[0]['recovered_params'] == []

    def test_an_absorbed_prereq_id_creates_no_junk_key(self):
        prereq = {
            'id': 'pre-1',
            'description': _ABSORBED_PREREQ_ID,
            'status': 'pending',
            'commit': None,
            'tests': [],
        }
        plan = corrupt_plan(prerequisites=[prereq])

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['prerequisites'][0]
        assert 'prereq_id' not in item, f'junk key written: {sorted(item)}'
        assert item['id'] == 'pre-1'
        assert item['description'] == _ABSORBED_PREREQ_ID
        assert len(facts) == 1
        assert facts[0]['outcome'] == 'unrepairable'

    def test_no_junk_key_survives_a_mixed_plan(self, tmp_path):
        """The invariant, over every specimen at once and against REAL keys.

        The allowed key sets are derived by building a plan through the actual
        writers, exactly as the table test does — a hardcoded list here would
        just be a second thing to keep in sync.
        """
        observed = _observed_plan_keys(tmp_path)
        decisions = [
            {'decision': _DECISION_PROSE, 'rationale': TRAILING_RATIONALE},
            {'decision': 'A clean decision.', 'rationale': 'A clean rationale.'},
        ]
        plan = corrupt_plan(
            title=_ABSORBED_FILES_TITLE,
            files=[],
            steps=[
                {
                    'id': 'step-1',
                    'type': 'impl',
                    'description': _ABSORBED_STEP_TYPE,
                    'status': 'pending',
                    'commit': None,
                },
            ],
            prerequisites=[
                {
                    'id': 'pre-1',
                    'description': _ABSORBED_PREREQ_ID,
                    'status': 'pending',
                    'commit': None,
                    'tests': [],
                },
            ],
            design_decisions=decisions,
        )

        repaired, _facts = plan_tools._repair_plan_fields(plan)

        assert set(repaired) <= observed[None], (
            f'top-level junk keys: {sorted(set(repaired) - observed[None])}'
        )
        for collection in ('prerequisites', 'steps', 'design_decisions', 'reuse'):
            for index, item in enumerate(repaired[collection]):
                junk = set(item) - observed[collection]
                assert junk == set(), f'{collection}[{index}] junk keys: {sorted(junk)}'

    def test_the_dominant_trailing_shapes_still_repair(self):
        """The load-bearing no-over-refusal guard.

        Supplying the non-target parameters unconditionally must not make the
        repair conservative in general: 97 of the 118 live corrupted strings
        have the trailing shape, and every one of them must still repair.
        """
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = TRAILING_RATIONALE
        plan['reuse'][1]['how'] = TRAILING_HOW
        plan['steps'][0]['description'] = _TRAILING_DESCRIPTION
        plan['analysis'] = _TRAILING_ANALYSIS

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert [f['outcome'] for f in facts] == ['repaired'] * 4
        assert all(f['recovered_params'] == [] for f in facts)
        assert repaired['design_decisions'][0]['rationale'] == _RATIONALE_PROSE
        assert repaired['reuse'][1]['how'] == _HOW_PROSE
        assert repaired['steps'][0]['description'] == 'Clean step prose for the first step.'
        assert repaired['analysis'] == 'Clean analysis prose describing the approach.'

    def test_the_absorbed_sibling_recovery_still_fills_an_empty_sibling(self):
        """``rationale`` IS a declared target, so step-7's recovery is intact."""
        plan = _absorbed_decision(rationale='')

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['design_decisions'][0]
        assert item['decision'] == _DECISION_PROSE
        assert item['rationale'] == _RATIONALE_PROSE
        assert facts[0]['outcome'] == 'repaired'
        assert facts[0]['recovered_params'] == ['rationale']


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


# ---------------------------------------------------------------------------
# step-13 — lazy write-back on read, its idempotence, and boundary row B12.
# ---------------------------------------------------------------------------

#: The stable event name every structured markup fact is logged under (INV-2).
MARKUP_FACT_EVENT = 'plan_markup_repaired'

#: C2's field names verbatim, plus the plan-specific locators this surface adds.
FACT_LOG_KEYS = frozenset({
    'tool', 'param', 'pattern', 'misclose', 'outcome', 'recovered_params',
    'collection', 'index', 'field', 'path',
})


def _fact_payloads(caplog) -> list[dict]:
    """Every structured fact plan-tools logged, already parsed.

    Consuming a fact must never require regex-scraping prose out of a log line:
    the WHOLE message is the payload, so ``json.loads`` is the only parser
    needed. A record from this logger that is not parseable JSON fails here,
    which is the point — that is what "structured" has to mean to be usable.
    """
    payloads = []
    for record in caplog.records:
        if record.name != plan_tools.__name__ or record.levelno < logging.WARNING:
            continue
        payloads.append(json.loads(record.getMessage()))
    return payloads


def _seed_mixed_plan(artifacts) -> dict:
    """Write a plan carrying one REPAIRABLE and one UNREPAIRABLE field."""
    plan = corrupt_plan()
    plan['design_decisions'][0]['rationale'] = TRAILING_RATIONALE
    plan['design_decisions'][1]['decision'] = PROSE_QUOTED
    artifacts.write_plan(copy.deepcopy(plan))
    return plan


class TestReadPlanRepaired:
    """The lazy write-back: repaired on read, in place, atomically."""

    def test_returns_the_repaired_document(self, plan_artifacts):
        _seed_mixed_plan(plan_artifacts)

        plan, facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert plan['design_decisions'][0]['rationale'] == _RATIONALE_PROSE
        assert detect(plan['design_decisions'][0]['rationale']) is None
        outcomes = sorted(fact['outcome'] for fact in facts)
        assert outcomes == ['repaired', 'unrepairable']

    def test_the_repair_is_persisted_to_disk(self, plan_artifacts):
        """The user-observable signal: the FILE is fixed, not just the reply.

        An in-memory-only repair would leave the next agent — and every later
        reader of this plan — looking at the corrupted rationale again.
        """
        _seed_mixed_plan(plan_artifacts)

        plan_tools._read_plan_repaired(plan_artifacts)

        on_disk = json.loads((plan_artifacts.root / 'plan.json').read_text())
        assert on_disk['design_decisions'][0]['rationale'] == _RATIONALE_PROSE
        # ...and the refusing field is BYTE-IDENTICAL on disk, not sanitized.
        assert on_disk['design_decisions'][1]['decision'] == PROSE_QUOTED

    def test_write_back_leaves_the_rest_of_the_document_intact(self, plan_artifacts):
        seeded = _seed_mixed_plan(plan_artifacts)

        plan_tools._read_plan_repaired(plan_artifacts)

        on_disk = json.loads((plan_artifacts.root / 'plan.json').read_text())
        assert on_disk['_schema_version'] == PLAN_SCHEMA_VERSION
        for key in ('task_id', 'title', 'analysis', 'files', 'prerequisites',
                    'steps', 'reuse'):
            assert on_disk[key] == seeded[key]
        assert on_disk['design_decisions'][0]['decision'] == _DECISION_PROSE
        assert on_disk['design_decisions'][1]['rationale'] == 'A second clean rationale.'

    # -- idempotence ------------------------------------------------------

    def test_second_read_repairs_nothing_and_writes_nothing(
        self, plan_artifacts, monkeypatch
    ):
        """A clean (or already-repaired) plan must NOT be rewritten per call.

        Rewriting on every tool call would churn mtimes under every watcher and
        turn a read into a write for the overwhelmingly common clean path.
        """
        _seed_mixed_plan(plan_artifacts)
        plan_path = plan_artifacts.root / 'plan.json'

        first, _ = plan_tools._read_plan_repaired(plan_artifacts)
        before_bytes = plan_path.read_bytes()
        before_mtime = plan_path.stat().st_mtime_ns

        writes: list = []
        real_write = plan_tools._atomic_write_plan
        monkeypatch.setattr(
            plan_tools,
            '_atomic_write_plan',
            lambda path, plan: (writes.append(path), real_write(path, plan))[1],
        )

        second, facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert second == first
        assert [f for f in facts if f['outcome'] == 'repaired'] == []
        assert writes == [], 'an already-repaired plan was rewritten'
        assert plan_path.read_bytes() == before_bytes
        assert plan_path.stat().st_mtime_ns == before_mtime

    def test_a_wholly_clean_plan_is_never_written(self, plan_artifacts, monkeypatch):
        plan_artifacts.write_plan(corrupt_plan())
        plan_path = plan_artifacts.root / 'plan.json'
        before_mtime = plan_path.stat().st_mtime_ns
        writes: list = []
        monkeypatch.setattr(
            plan_tools, '_atomic_write_plan', lambda path, plan: writes.append(path)
        )

        plan, facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert facts == []
        assert writes == []
        assert plan_path.stat().st_mtime_ns == before_mtime

    def test_an_unrepairable_only_plan_is_never_written(
        self, plan_artifacts, monkeypatch
    ):
        """Nothing changed, so nothing is written — the 2939 prose plan stays put."""
        plan = corrupt_plan()
        plan['design_decisions'][0]['decision'] = PROSE_QUOTED
        plan_artifacts.write_plan(plan)
        plan_path = plan_artifacts.root / 'plan.json'
        before_bytes = plan_path.read_bytes()
        writes: list = []
        monkeypatch.setattr(
            plan_tools, '_atomic_write_plan', lambda path, plan: writes.append(path)
        )

        _, facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert [f['outcome'] for f in facts] == ['unrepairable']
        assert writes == []
        assert plan_path.read_bytes() == before_bytes

    # -- the structured fact (INV-2) --------------------------------------

    def test_every_fact_is_logged_as_a_parseable_structured_payload(
        self, plan_artifacts, caplog
    ):
        _seed_mixed_plan(plan_artifacts)

        with caplog.at_level(logging.WARNING, logger=plan_tools.__name__):
            _, facts = plan_tools._read_plan_repaired(plan_artifacts)

        payloads = _fact_payloads(caplog)
        assert len(payloads) == len(facts) == 2
        for payload in payloads:
            assert payload['event'] == MARKUP_FACT_EVENT
            assert set(payload) >= FACT_LOG_KEYS
            assert payload['outcome'] in {'repaired', 'unrepairable'}
            assert payload['path'] == str(plan_artifacts.root / 'plan.json')
        assert sorted(p['outcome'] for p in payloads) == ['repaired', 'unrepairable']

    def test_the_logged_payload_carries_the_full_locator(self, plan_artifacts, caplog):
        """Enough to point at the exact field without reopening the plan."""
        plan = corrupt_plan()
        plan['reuse'][1]['how'] = TRAILING_HOW
        plan_artifacts.write_plan(plan)

        with caplog.at_level(logging.WARNING, logger=plan_tools.__name__):
            plan_tools._read_plan_repaired(plan_artifacts)

        payload = _fact_payloads(caplog)[0]
        assert payload['collection'] == 'reuse'
        assert payload['index'] == 1
        assert payload['field'] == 'how'
        assert payload['param'] == 'how'
        assert payload['tool'] == 'add_reuse_item'
        assert payload['outcome'] == 'repaired'
        assert payload['recovered_params'] == []

    def test_each_message_is_a_single_line(self, plan_artifacts, caplog):
        """One fact per log record — never a multi-line blob to reassemble."""
        _seed_mixed_plan(plan_artifacts)

        with caplog.at_level(logging.WARNING, logger=plan_tools.__name__):
            plan_tools._read_plan_repaired(plan_artifacts)

        messages = [
            r.getMessage() for r in caplog.records if r.name == plan_tools.__name__
        ]
        assert messages
        for message in messages:
            assert '\n' not in message

    # -- degradation ------------------------------------------------------

    def test_missing_plan_degrades_exactly_as_read_plan_does(
        self, plan_artifacts, monkeypatch
    ):
        """No plan.json -> ``{}``, no facts, no write. No NEW failure mode."""
        plan_path = plan_artifacts.root / 'plan.json'
        plan_path.unlink(missing_ok=True)
        assert plan_artifacts.read_plan() == {}
        writes: list = []
        monkeypatch.setattr(
            plan_tools, '_atomic_write_plan', lambda path, plan: writes.append(path)
        )

        plan, facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert plan == {}
        assert facts == []
        assert writes == []
        assert not plan_path.exists()

    def test_unparseable_plan_raises_exactly_what_read_plan_raises(
        self, plan_artifacts, monkeypatch
    ):
        plan_path = plan_artifacts.root / 'plan.json'
        plan_path.write_text('{not json at all')
        before_bytes = plan_path.read_bytes()
        with pytest.raises(json.JSONDecodeError):
            plan_artifacts.read_plan()
        writes: list = []
        monkeypatch.setattr(
            plan_tools, '_atomic_write_plan', lambda path, plan: writes.append(path)
        )

        with pytest.raises(json.JSONDecodeError):
            plan_tools._read_plan_repaired(plan_artifacts)

        assert writes == []
        assert plan_path.read_bytes() == before_bytes

    def test_a_failed_write_back_still_serves_the_repaired_plan(
        self, plan_artifacts, monkeypatch, caplog
    ):
        """Refusing to serve a repair we could not persist is strictly worse.

        The caller would then get the CORRUPTED text back — the exact failure
        this surface exists to end — because persistence happened to fail.
        """
        _seed_mixed_plan(plan_artifacts)

        def boom(_path, _plan):
            raise plan_tools.PlanWriteError('disk on fire')

        monkeypatch.setattr(plan_tools, '_atomic_write_plan', boom)

        with caplog.at_level(logging.WARNING, logger=plan_tools.__name__):
            plan, facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert plan['design_decisions'][0]['rationale'] == _RATIONALE_PROSE
        assert any(f['outcome'] == 'repaired' for f in facts)
        assert 'disk on fire' in caplog.text, 'the write failure must be LOUD'


# ---------------------------------------------------------------------------
# step-13(d) — boundary row B12: a concurrent reader never sees a partial file.
# ---------------------------------------------------------------------------


class TestConcurrentReadersNeverSeeAPartialPlan:
    """B12 stated as an executable property, not as an assertion about code.

    ``TaskArtifacts._write_json`` is ``path.write_text`` — truncate-then-write —
    so a reader racing it CAN observe a half-written file. The repair write-back
    must not inherit that window, and the only convincing evidence is to race it.
    """

    READERS = 4
    READS_PER_READER = 300
    WRITES = 50

    def test_readers_only_ever_observe_a_complete_document(self, tmp_path):
        directory = tmp_path / 'meta'
        directory.mkdir()
        plan_path = directory / 'plan.json'

        # Two documents of DIFFERENT sizes: a torn read of the larger one
        # cannot be mistaken for a clean read of the smaller.
        doc_a = corrupt_plan(title='A' * 4000)
        doc_b = corrupt_plan(title='B', analysis='B' * 40000)
        expected = []
        for doc in (doc_a, doc_b):
            stamped = copy.deepcopy(doc)
            stamped['_schema_version'] = PLAN_SCHEMA_VERSION
            expected.append(stamped)

        plan_path.write_text(json.dumps(expected[0], indent=2) + '\n')

        errors: list = []
        observed_unexpected: list = []
        start = threading.Barrier(self.READERS + 1)
        done = threading.Event()

        def reader():
            start.wait(timeout=30)
            for _ in range(self.READS_PER_READER):
                try:
                    document = json.loads(plan_path.read_text())
                except Exception as exc:  # noqa: BLE001 — recorded, then asserted on
                    errors.append(repr(exc))
                    continue
                if document not in expected:
                    observed_unexpected.append(sorted(document))
                if done.is_set():
                    break

        threads = [
            threading.Thread(target=reader, name=f'reader-{i}', daemon=True)
            for i in range(self.READERS)
        ]
        for thread in threads:
            thread.start()

        start.wait(timeout=30)
        for i in range(self.WRITES):
            plan_tools._atomic_write_plan(plan_path, copy.deepcopy(doc_b if i % 2 else doc_a))
        done.set()

        for thread in threads:
            thread.join(timeout=60)
            assert not thread.is_alive(), f'{thread.name} did not finish'

        assert errors == [], f'a reader observed an unparseable plan.json: {errors[:3]}'
        assert observed_unexpected == [], (
            'a reader observed a document that is neither the pre-write nor the '
            f'post-write plan: {observed_unexpected[:2]}'
        )

    def test_the_plan_path_is_never_absent_mid_write(self, tmp_path):
        """``os.replace`` swaps in place — it must never unlink-then-create."""
        directory = tmp_path / 'meta'
        directory.mkdir()
        plan_path = directory / 'plan.json'
        plan_path.write_text(json.dumps(corrupt_plan(), indent=2) + '\n')

        misses: list = []
        stop = threading.Event()

        def watcher():
            while not stop.is_set():
                if not plan_path.exists():
                    misses.append(1)

        thread = threading.Thread(target=watcher, daemon=True)
        thread.start()
        try:
            for i in range(self.WRITES):
                plan_tools._atomic_write_plan(plan_path, corrupt_plan(title=f't{i}'))
        finally:
            stop.set()
            thread.join(timeout=30)

        assert misses == [], 'plan.json vanished during an atomic write-back'


# ---------------------------------------------------------------------------
# step-15 — EVERY read path repairs, and the triggering tool reports it.
# ---------------------------------------------------------------------------

_NEW_SHA = 'a' * 40


def _seed_corrupt(artifacts) -> None:
    """Put a corrupted-but-otherwise-valid plan on disk for a tool to open."""
    plan = corrupt_plan()
    plan['design_decisions'][0]['rationale'] = TRAILING_RATIONALE
    artifacts.write_plan(plan)


def _on_disk(artifacts) -> dict:
    return json.loads((artifacts.root / 'plan.json').read_text())


def _added_step(plan: dict) -> None:
    assert [s['id'] for s in plan['steps']] == ['step-1', 'step-2', 'step-9']


def _added_prereq(plan: dict) -> None:
    assert [p['id'] for p in plan['prerequisites']] == ['pre-1', 'pre-9']


def _added_decision(plan: dict) -> None:
    assert plan['design_decisions'][2] == {
        'decision': 'A newly added decision.',
        'rationale': 'A newly added rationale.',
    }


def _added_reuse(plan: dict) -> None:
    assert plan['reuse'][2]['what'] == 'A newly reused thing'


def _step_marked_done(plan: dict) -> None:
    assert plan['steps'][0]['status'] == 'done'
    assert plan['steps'][0]['commit'] == _NEW_SHA


def _step_pre_committed(plan: dict) -> None:
    assert plan['steps'][0]['status'] == 'done'
    assert plan['steps'][0]['commit'] == _NEW_SHA
    assert plan['steps'][0]['description'].startswith(f'[COMMITTED {_NEW_SHA[:12]}]')


def _metadata_updated(plan: dict) -> None:
    assert plan['files'] == ['orchestrator/src/orchestrator/artifacts.py']
    assert plan['analysis'] == 'A newly written analysis.'


def _step_removed(plan: dict) -> None:
    assert [s['id'] for s in plan['steps']] == ['step-1']


def _step_replaced(plan: dict) -> None:
    assert plan['steps'][1]['type'] == 'impl'
    assert plan['steps'][1]['description'] == 'A newly written step description.'


def _plan_finalized(plan: dict) -> None:
    assert plan['_finalized_at']
    assert plan['_revalidated_at']


#: ``(id, call, expected-response-subset, on-disk-mutation-check)`` for every
#: plan-tools entry point that OPENS an existing plan. ``_create_plan`` is
#: absent on purpose: it overwrites the document wholesale, so it has no read to
#: repair — guarding its INBOUND arguments belongs to the write-time middleware.
READ_PATH_CASES = [
    (
        'add_plan_step',
        lambda a: plan_tools._add_plan_step(a, 'step-9', 'impl', 'A newly added step.'),
        {'status': 'ok', 'step_id': 'step-9', 'total_steps': 3},
        _added_step,
    ),
    (
        'add_prerequisite',
        lambda a: plan_tools._add_prerequisite(a, 'pre-9', 'A newly added prerequisite.'),
        {'status': 'ok', 'prereq_id': 'pre-9'},
        _added_prereq,
    ),
    (
        'add_design_decision',
        lambda a: plan_tools._add_design_decision(
            a, 'A newly added decision.', 'A newly added rationale.'
        ),
        {'status': 'ok', 'total_decisions': 3},
        _added_decision,
    ),
    (
        'add_reuse_item',
        lambda a: plan_tools._add_reuse_item(
            a, 'A newly reused thing', 'somewhere.py', 'By importing it.'
        ),
        {'status': 'ok', 'total_reuse': 3},
        _added_reuse,
    ),
    (
        'mark_step_done',
        lambda a: plan_tools._mark_step_done(a, 'step-1', _NEW_SHA),
        {'status': 'ok', 'step_id': 'step-1', 'new_status': 'done', 'commit': _NEW_SHA},
        _step_marked_done,
    ),
    (
        # Contract A1's DELIBERATELY divergent envelope — ``ok`` plus ``status``
        # meaning the step's NEW status. Adding markup_repairs must not disturb it.
        'mark_step_committed',
        lambda a: plan_tools._mark_step_committed(a, 'step-1', _NEW_SHA),
        {'ok': True, 'step_id': 'step-1', 'status': 'done'},
        _step_pre_committed,
    ),
    (
        'update_plan_metadata',
        lambda a: plan_tools._update_plan_metadata(
            a,
            files=['orchestrator/src/orchestrator/artifacts.py'],
            analysis='A newly written analysis.',
        ),
        {'status': 'ok', 'files': 1},
        _metadata_updated,
    ),
    (
        'remove_plan_step',
        lambda a: plan_tools._remove_plan_step(a, 'step-2'),
        {'status': 'ok', 'removed': 'step-2', 'collection': 'steps'},
        _step_removed,
    ),
    (
        'replace_plan_step',
        lambda a: plan_tools._replace_plan_step(
            a, 'step-2', 'impl', 'A newly written step description.'
        ),
        {'status': 'ok', 'replaced': 'step-2'},
        _step_replaced,
    ),
    (
        'confirm_plan',
        lambda a: plan_tools._confirm_plan(a),
        {'status': 'ok', 'finalized': True, 'steps': 2, 'files': 1},
        _plan_finalized,
    ),
]

READ_PATH_IDS = [case[0] for case in READ_PATH_CASES]


@pytest.fixture()
def on_branch(monkeypatch):
    """``mark_step_committed``'s reachability guard, satisfied.

    Stubbed rather than staged as a real commit: this module tests the repair
    surface, and the guard itself is covered by ``test_plan_tools_server``.
    """
    monkeypatch.setattr(plan_tools, '_sha_exists_on_branch', lambda worktree, sha: True)


@pytest.mark.parametrize(
    ('call', 'expected_response', 'check_mutation'),
    [case[1:] for case in READ_PATH_CASES],
    ids=READ_PATH_IDS,
)
class TestPlanToolsReadPathsRepair:
    """Opening the plan through ANY tool repairs it — no tool is a bypass.

    One unhooked read site is enough to leave a corrupted plan corrupted for
    the whole task, so coverage is asserted entry point by entry point rather
    than inferred from the one helper they are all supposed to share.
    """

    def test_the_documented_envelope_is_unchanged(
        self, plan_artifacts, on_branch, call, expected_response, check_mutation
    ):
        _seed_corrupt(plan_artifacts)

        response = call(plan_artifacts)

        for key, value in expected_response.items():
            assert response[key] == value, f'{key!r} changed shape'

    def test_the_on_disk_plan_is_repaired(
        self, plan_artifacts, on_branch, call, expected_response, check_mutation
    ):
        _seed_corrupt(plan_artifacts)

        call(plan_artifacts)

        rationale = _on_disk(plan_artifacts)['design_decisions'][0]['rationale']
        assert rationale == _RATIONALE_PROSE
        assert detect(rationale) is None

    def test_the_requested_mutation_lands_in_the_same_file(
        self, plan_artifacts, on_branch, call, expected_response, check_mutation
    ):
        """The repair must not eat the write it rode in on.

        A repair write-back followed by the tool's own write is two writes to
        one file; if they raced or ordered wrongly, one of them would vanish.
        """
        _seed_corrupt(plan_artifacts)

        call(plan_artifacts)

        plan = _on_disk(plan_artifacts)
        check_mutation(plan)
        assert plan['design_decisions'][0]['rationale'] == _RATIONALE_PROSE

    def test_the_response_reports_what_was_repaired(
        self, plan_artifacts, on_branch, call, expected_response, check_mutation
    ):
        """The agent that opened the plan must SEE what changed under it."""
        _seed_corrupt(plan_artifacts)

        response = call(plan_artifacts)

        repairs = response['markup_repairs']
        assert isinstance(repairs, list)
        assert len(repairs) == 1
        assert repairs[0]['outcome'] == 'repaired'
        assert repairs[0]['collection'] == 'design_decisions'
        assert repairs[0]['index'] == 0
        assert repairs[0]['field'] == 'rationale'

    def test_a_clean_plan_response_is_byte_identical_to_today(
        self, plan_artifacts, on_branch, call, expected_response, check_mutation
    ):
        """No key at all on the common path — not an empty list, not a null.

        Every existing caller and every existing assertion on these envelopes
        must be unable to tell this change happened.
        """
        plan_artifacts.write_plan(corrupt_plan())

        response = call(plan_artifacts)

        assert 'markup_repairs' not in response


class TestNoReadPathBypassesTheRepair:
    """A future read site must not be able to silently skip the repair."""

    @staticmethod
    def _module_functions() -> dict:
        return {
            name: obj
            for name, obj in vars(plan_tools).items()
            if inspect.isfunction(obj) and obj.__module__ == plan_tools.__name__
        }

    def test_no_function_calls_artifacts_read_plan_directly(self):
        offenders = sorted(
            name
            for name, fn in self._module_functions().items()
            if name != '_read_plan_repaired'
            and 'artifacts.read_plan(' in inspect.getsource(fn)
        )
        assert offenders == [], (
            'these read the plan directly and so skip the repair entirely — '
            f'route them through _read_plan_repaired: {offenders}'
        )

    def test_every_read_path_entry_point_goes_through_the_repair(self):
        """The non-vacuous half: the guard above passes trivially if nobody reads."""
        expected = {f'_{case_id}' for case_id in READ_PATH_IDS}
        callers = {
            name
            for name, fn in self._module_functions().items()
            if '_read_plan_repaired(' in inspect.getsource(fn)
        } - {'_read_plan_repaired'}
        assert expected <= callers, f'not routed through the repair: {expected - callers}'

    def test_create_plan_is_deliberately_not_hooked(self):
        """It overwrites the document wholesale, so it has no read to repair.

        Guarding its INBOUND arguments is the write-time middleware's job, not
        this surface's. Asserted so the omission reads as a decision.
        """
        source = inspect.getsource(plan_tools._create_plan)
        assert '_read_plan_repaired(' not in source
        assert 'artifacts.read_plan(' not in source
