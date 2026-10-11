"""Contract of fused_memory.utils.clip_disclosure: one vocabulary for disclosing a clip."""

from __future__ import annotations

import pytest

from fused_memory.middleware.task_curator import clip_for_prompt
from fused_memory.utils.clip_disclosure import (
    TRUNCATED_KEY,
    TextClip,
    clip_note,
    clip_text,
    elide_text,
    elision_marker,
    fetch_cap_note,
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


def _widget_note(omitted: int) -> str:
    return clip_note(
        omitted=omitted,
        omitted_noun='widget task(s)',
        cap_name='MAX_WIDGETS_RENDERED',
        cap=3,
        dropped_first='the newest widgets',
    )


@pytest.mark.parametrize('omitted', [0, -2])
def test_clip_note_is_empty_when_nothing_was_omitted(omitted):
    assert _widget_note(omitted) == ''


def test_clip_note_states_the_count_the_cap_and_the_dropped_end():
    assert _widget_note(2) == (
        '_NOTE: 2 additional widget task(s) were omitted from this render by the '
        'MAX_WIDGETS_RENDERED=3 cap. Coverage was clipped — NOT complete this cycle; '
        'the newest widgets were dropped first._'
    )


def _episode_fetch_note(fetched: int) -> str:
    return fetch_cap_note(fetched=fetched, noun='episode(s)', cap_name='episode_limit', cap=50)


@pytest.mark.parametrize('fetched', [0, 49])
def test_fetch_cap_note_is_empty_below_the_cap(fetched):
    assert _episode_fetch_note(fetched) == ''


@pytest.mark.parametrize('fetched', [50, 51])
def test_fetch_cap_note_at_the_cap_says_coverage_is_not_known_complete(fetched):
    note = _episode_fetch_note(fetched)

    assert note.startswith('_NOTE: ')
    assert note.endswith('_')
    assert '\n' not in note
    assert f'{fetched} episode(s)' in note
    assert 'episode_limit=50' in note
    assert 'not fetched' in note
    assert 'may be clipped' in note
    assert 'NOT known complete this cycle' in note
