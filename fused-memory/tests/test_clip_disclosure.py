"""Contract of fused_memory.utils.clip_disclosure: one vocabulary for disclosing a clip."""

from __future__ import annotations

import pytest

from fused_memory.middleware.task_curator import clip_for_prompt
from fused_memory.utils.clip_disclosure import (
    TRUNCATED_KEY,
    TextClip,
    clip_text,
    elide_text,
    elision_marker,
    truncation_facts,
)


def test_elision_marker_counts_the_elided_characters():
    assert elision_marker(1827) == '…[+1827 chars elided]'


@pytest.mark.parametrize('value', ['', 'short', 'x' * 10])
def test_elide_text_leaves_text_at_or_under_the_cap_unchanged(value):
    clip = elide_text(value, 10, field='desc')

    assert clip == TextClip(text=value, fact=None)


def test_elide_text_over_the_cap_keeps_the_head_and_counts_the_rest():
    value = 'a' * 10 + 'b' * 7

    clip = elide_text(value, 10, field='desc')

    assert clip.text == 'a' * 10 + elision_marker(7)
    assert clip.fact == {'field': 'desc', 'cap': 10, 'elided': 7}


@pytest.mark.parametrize(
    ('value', 'cap'),
    [('', 5), ('abc', 5), ('abcde', 5), ('abcdef', 5), ('x' * 4000, 300), ('é' * 12, 3)],
)
def test_elide_text_spells_the_marker_exactly_as_the_curator_prompt_does(value, cap):
    assert elide_text(value, cap, field='f').text == clip_for_prompt(value, cap)


def test_clip_text_reports_the_same_fact_but_leaves_the_text_unmarked():
    value = 'a' * 10 + 'b' * 7

    clip = clip_text(value, 10, field='content_excerpt')

    assert clip.text == 'a' * 10
    assert clip.fact == {'field': 'content_excerpt', 'cap': 10, 'elided': 7}


@pytest.mark.parametrize('clipper', [clip_text, elide_text])
@pytest.mark.parametrize('empty', [None, ''])
def test_empty_input_is_an_unclipped_empty_string(clipper, empty):
    assert clipper(empty, 3, field='f') == TextClip(text='', fact=None)


@pytest.mark.parametrize('clipper', [clip_text, elide_text])
def test_non_string_input_is_coerced_before_clipping(clipper):
    clip = clipper(1234567, 3, field='f')

    assert clip.text.startswith('123')
    assert clip.fact == {'field': 'f', 'cap': 3, 'elided': 4}


def test_truncation_facts_keeps_only_the_clips_that_cut_in_order():
    cut_a = clip_text('abcdef', 2, field='a')
    whole = clip_text('ab', 2, field='b')
    cut_c = elide_text('abcdef', 3, field='c')

    assert truncation_facts(cut_a, whole, cut_c) == [
        {'field': 'a', 'cap': 2, 'elided': 4},
        {'field': 'c', 'cap': 3, 'elided': 3},
    ]


def test_truncation_facts_is_empty_when_nothing_was_cut():
    assert truncation_facts(clip_text('ab', 2, field='a'), clip_text(None, 2, field='b')) == []


def test_truncated_key_is_the_one_dict_channel_spelling():
    assert TRUNCATED_KEY == 'truncated'
