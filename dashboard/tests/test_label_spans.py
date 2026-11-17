"""Contract of dashboard.data.label_spans: a label row drawn from a union row,
sent as half-open [start, stop) index spans over that union, and rebuilt."""

from __future__ import annotations

import json

import pytest

from dashboard.data.label_spans import label_spans, labels_from_spans

_UNION = ['a', 'b', 'c', 'd']


def test_a_full_row_is_one_span():
    assert label_spans(['a', 'b', 'c'], ['a', 'b', 'c']) == [[0, 3]]


def test_each_gap_starts_a_new_span():
    assert label_spans(['a', 'c', 'd'], _UNION) == [[0, 1], [2, 4]]


def test_an_empty_row_has_no_spans():
    assert label_spans([], _UNION) == []


def test_spans_are_two_int_lists_equal_to_their_json_round_trip():
    spans = label_spans(['b', 'd'], _UNION)

    assert spans == [[1, 2], [3, 4]]
    assert all(type(span) is list and len(span) == 2 for span in spans)
    assert all(type(bound) is int for span in spans for bound in span)
    assert json.loads(json.dumps(spans)) == spans


def test_a_row_out_of_union_order_keeps_its_own_order():
    spans = label_spans(['c', 'a'], ['a', 'b', 'c'])

    assert spans == [[2, 3], [0, 1]]
    assert labels_from_spans(spans, ['a', 'b', 'c']) == ['c', 'a']


@pytest.mark.parametrize('own', [
    ['b', 'd'],
    list(_UNION),
    [],
    ['c', 'a'],
    ['b', 'b', 'c'],
    ['d', 'a', 'b', 'c'],
])
def test_decoding_the_encoding_returns_the_row(own):
    assert labels_from_spans(label_spans(own, _UNION), _UNION) == own


def test_a_label_absent_from_the_union_is_refused_by_name():
    with pytest.raises(ValueError, match='zz'):
        label_spans(['a', 'zz'], _UNION)


@pytest.mark.parametrize('bad_span', [
    [2, 5],
    [-1, 2],
    [2, 2],
    [3, 1],
])
def test_an_out_of_range_span_is_refused_naming_it_and_the_union_length(bad_span):
    union = ['a', 'b', 'c']

    with pytest.raises(ValueError) as refused:
        labels_from_spans([[0, 1], bad_span], union)

    message = str(refused.value)
    assert str(bad_span) in message
    assert '3' in message.replace(str(bad_span), '')


@pytest.mark.parametrize('bad_span', [
    [1],
    [0, 1, 2],
    ['0', '1'],
    [0.0, 1.0],
    [True, 2],
    'ab',
    None,
])
def test_a_span_that_is_not_a_two_int_pair_is_refused_naming_it(bad_span):
    with pytest.raises(ValueError) as refused:
        labels_from_spans([bad_span], ['a', 'b', 'c'])

    message = str(refused.value)
    assert repr(bad_span) in message
    assert '3' in message.replace(repr(bad_span), '')
