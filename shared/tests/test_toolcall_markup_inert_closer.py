"""Contract of shared.toolcall_markup.only_inert_html_closers (task 10396).

Three of the closer names the envelope enumeration spells — ``description``,
``details`` and ``content`` — are also ordinary markup element names. A value
that QUOTES an HTML block such as a collapsible ``details`` section carries the
closer of that element, which the gate detects as an envelope literal. When
that closer names no parameter of the invoked tool, or names one but is
followed by prose rather than an item boundary, it terminates no envelope
boundary and :func:`repair` has nothing to accept.

The predicate answers exactly that question, and is asked by the boundary only
after :func:`repair` declines. These tests pin its contract over the value
alone: what counts as inert, what disqualifies a value, its totality, and that
no record of the committed leak corpus is ever classed inert.

## Sentinel-literal hazard — DO NOT "helpfully" un-escape these

Every tag in this file is built from the module's public constants,
:func:`closer_for` or the ``\\x3c`` escape for ``<``; the rule and its
rationale are owned by ``shared/src/shared/toolcall_markup.py``
("Sentinel-literal hazard").
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared import toolcall_markup
from shared.toolcall_markup import (
    CANONICAL_OPENER_PREFIX,
    HTML_COLLIDING_CLOSER_NAMES,
    INVOKE_CLOSER,
    PARAMETER_CLOSER_NAMES,
    closer_for,
    only_inert_html_closers,
)

_CORPUS_PATH = Path(__file__).resolve().parent / 'fixtures' / 'toolcall_markup_corpus.jsonl'

_STEP_PARAM = 'description'
_STEP_SCHEMA = ('step_id', 'step_type', 'description')
_TASK_SCHEMA = ('title', 'description', 'details')


def _opener(name: str) -> str:
    return '\x3c' + name + '>'


def _element(name: str, body: str) -> str:
    return _opener(name) + body + closer_for(name)


_DETAILS_BLOCK = _opener('details') + _element('summary', 'Evidence') + 'body text' + closer_for('details')
_PROSE_WITH_BLOCK = 'Before the block.\n' + _DETAILS_BLOCK + '\nAfter the block.'


class TestTheCollidingNames:

    def test_names_are_the_markup_element_names_of_the_enumeration(self):
        assert frozenset({'description', 'details', 'content'}) == HTML_COLLIDING_CLOSER_NAMES

    def test_every_name_is_already_enumerated_and_parameter_is_not_one(self):
        assert set(PARAMETER_CLOSER_NAMES) >= HTML_COLLIDING_CLOSER_NAMES
        assert 'parameter' not in HTML_COLLIDING_CLOSER_NAMES

    def test_both_names_are_exported(self):
        assert 'HTML_COLLIDING_CLOSER_NAMES' in toolcall_markup.__all__
        assert 'only_inert_html_closers' in toolcall_markup.__all__


class TestInertWhenTheNameIsOutsideTheSchema:

    def test_block_inside_prose(self):
        assert only_inert_html_closers(_PROSE_WITH_BLOCK, _STEP_PARAM, _STEP_SCHEMA) is True

    def test_block_at_end_of_string(self):
        assert only_inert_html_closers('Prose first.\n' + _DETAILS_BLOCK, _STEP_PARAM, _STEP_SCHEMA) is True

    def test_two_consecutive_blocks(self):
        value = 'Two blocks:\n' + _DETAILS_BLOCK + '\n' + _DETAILS_BLOCK + '\nDone.'
        assert only_inert_html_closers(value, _STEP_PARAM, _STEP_SCHEMA) is True

    def test_content_closer_in_a_summary_value(self):
        value = 'Rendered as\n' + _element('content', 'the body') + '\nin the page.'
        assert only_inert_html_closers(value, 'summary', ('task_id', 'verdict', 'summary')) is True


class TestInertWhenAnInSchemaCloserIsFollowedByProse:

    def test_own_closer_quoted_in_backticks(self):
        value = 'The `' + closer_for('description') + '` tag ends the field in the old dialect.'
        assert only_inert_html_closers(value, _STEP_PARAM, _STEP_SCHEMA) is True

    def test_schema_closer_followed_by_prose(self):
        value = 'Collapsed:\n' + _DETAILS_BLOCK + ' and then more prose.'
        assert only_inert_html_closers(value, _STEP_PARAM, _TASK_SCHEMA) is True


class TestNotInertWhenAnyOtherNeedleAppears:

    @pytest.mark.parametrize(
        'suffix',
        [
            pytest.param(INVOKE_CLOSER, id='invoke-closer'),
            pytest.param(CANONICAL_OPENER_PREFIX + '"x">', id='canonical-opener'),
            pytest.param(closer_for('parameter'), id='parameter-closer'),
        ],
    )
    def test_an_unambiguous_envelope_literal_disqualifies(self, suffix):
        value = _PROSE_WITH_BLOCK + suffix + ' trailing'
        assert only_inert_html_closers(value, _STEP_PARAM, _STEP_SCHEMA) is False

    def test_a_non_colliding_self_name_closer_disqualifies(self):
        value = _PROSE_WITH_BLOCK + '\nThe rationale ends here.' + closer_for('rationale')
        assert only_inert_html_closers(value, 'rationale', ('decision', 'rationale')) is False

    def test_a_non_colliding_schema_closer_disqualifies(self):
        value = _PROSE_WITH_BLOCK + closer_for('rationale') + ' tail prose'
        assert only_inert_html_closers(value, 'decision', ('decision', 'rationale')) is False


class TestNotInertWhenAnInSchemaCloserSitsAtAnItemBoundary:

    def test_own_closer_at_end_of_string(self):
        value = 'The step text.' + closer_for('description')
        assert only_inert_html_closers(value, _STEP_PARAM, _STEP_SCHEMA) is False

    def test_own_closer_followed_only_by_whitespace(self):
        value = 'The step text.' + closer_for('description') + '\n  \n'
        assert only_inert_html_closers(value, _STEP_PARAM, _STEP_SCHEMA) is False

    def test_schema_closer_at_end_of_string(self):
        value = 'Collapsed:\n' + _DETAILS_BLOCK
        assert only_inert_html_closers(value, _STEP_PARAM, _TASK_SCHEMA) is False

    def test_schema_closer_followed_by_an_echo_opener(self):
        value = 'Collapsed:\n' + _DETAILS_BLOCK + '\n' + _opener('title') + 'Next title'
        assert only_inert_html_closers(value, _STEP_PARAM, _TASK_SCHEMA) is False

    def test_one_boundary_closer_disqualifies_an_otherwise_inert_value(self):
        value = _PROSE_WITH_BLOCK + '\nFinal words.' + closer_for('description')
        assert only_inert_html_closers(value, _STEP_PARAM, _STEP_SCHEMA) is False


class TestTotalAndFailClosed:

    @pytest.mark.parametrize(
        'schema',
        [
            pytest.param((), id='empty-tuple'),
            pytest.param(frozenset(), id='empty-set'),
            pytest.param('step_id', id='bare-str'),
            pytest.param(None, id='none'),
            pytest.param(42, id='int'),
            pytest.param((1, 2), id='no-string-names'),
        ],
    )
    def test_no_usable_schema_is_never_inert(self, schema):
        assert only_inert_html_closers(_PROSE_WITH_BLOCK, _STEP_PARAM, schema) is False

    @pytest.mark.parametrize('value', [None, '', 0, 42, b'bytes', ['list'], {'k': 'v'}])
    def test_a_non_string_or_empty_value_is_never_inert(self, value):
        assert only_inert_html_closers(value, _STEP_PARAM, _STEP_SCHEMA) is False

    def test_a_value_with_no_needle_is_never_inert(self):
        value = 'Plain prose with ' + _opener('b') + 'bold' + ' and no closer at all.'
        assert only_inert_html_closers(value, _STEP_PARAM, _STEP_SCHEMA) is False

    @pytest.mark.parametrize('param', [None, '', 7, ('description',)])
    def test_an_unusable_param_still_answers(self, param):
        assert only_inert_html_closers(_PROSE_WITH_BLOCK, param, _STEP_SCHEMA) is True

    def test_a_schema_mixing_names_and_junk_uses_the_names(self):
        schema = ('step_id', None, 3, 'description')
        assert only_inert_html_closers(_PROSE_WITH_BLOCK, _STEP_PARAM, schema) is True


def _corpus_records() -> list[dict]:
    assert _CORPUS_PATH.is_file(), f'committed corpus missing at {_CORPUS_PATH}'
    with _CORPUS_PATH.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_no_committed_corpus_leak_is_classed_inert():
    records = _corpus_records()
    assert records
    inert = [
        record['tool_use_id']
        for record in records
        if only_inert_html_closers(record['value'], record['param'], record['schema_params'])
    ]
    assert inert == []
