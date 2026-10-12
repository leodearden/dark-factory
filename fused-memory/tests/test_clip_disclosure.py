"""Contract of fused_memory.utils.clip_disclosure: one vocabulary for disclosing a clip."""

from __future__ import annotations

import pytest

from fused_memory.middleware.task_curator import clip_for_prompt
from fused_memory.utils.clip_disclosure import (
    TRUNCATION_FACTS_KEY,
    TextClip,
    TruncationFact,
    clip_note,
    clip_text,
    elide_text,
    elision_marker,
    fetch_cap_note,
    fit_at_word,
    truncation_fields,
    window_note,
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
    assert clip.fact == TruncationFact(field='desc', cap=10, elided=7)


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
    assert clip.fact == TruncationFact(field='content_excerpt', cap=10, elided=7)


@pytest.mark.parametrize('clipper', [clip_text, elide_text])
@pytest.mark.parametrize('empty', [None, ''])
def test_empty_input_is_an_unclipped_empty_string(clipper, empty):
    assert clipper(empty, 3, field='f') == TextClip(text='', fact=None)


@pytest.mark.parametrize('clipper', [clip_text, elide_text])
def test_non_string_input_is_coerced_before_clipping(clipper):
    clip = clipper(1234567, 3, field='f')

    assert clip.text.startswith('123')
    assert clip.fact == TruncationFact(field='f', cap=3, elided=4)


def test_fit_at_word_leaves_text_at_or_under_the_limit_unchanged():
    """Text at or under the limit comes back verbatim, with no ellipsis."""
    short = 'x' * 20
    assert fit_at_word(short, 500) == short
    assert '…' not in fit_at_word(short, 500)
    exact = 'y' * 500
    assert fit_at_word(exact, 500) == exact


def test_fit_at_word_backs_up_to_the_last_word_boundary():
    """The cut drops the trailing word whole, so no fragment like '8679,868' (task 2080) reaches the ellipsis."""
    text = ('A' * 495) + ' ' + ('Z' * 10)

    assert fit_at_word(text, 500) == ('A' * 495) + '…'


def test_fit_at_word_hard_cuts_when_the_window_has_no_whitespace():
    """With no whitespace to back up to, the cut lands at limit - 1 so the ellipsis still fits the limit."""
    result = fit_at_word('N' * 600, 500)

    assert result == ('N' * 499) + '…'
    assert len(result) == 500


def test_fit_at_word_ignores_whitespace_that_would_keep_half_the_budget_or_less():
    """A boundary that would discard more than half the budget is passed over for a hard cut."""
    text = 'ab ' + ('N' * 600)

    assert fit_at_word(text, 500) == text[:499] + '…'


def test_fit_at_word_takes_a_boundary_well_past_half_the_budget():
    """A boundary comfortably past half the budget is taken, even far from the window's edge."""
    text = ('A' * 300) + ' ' + ('B' * 300)

    assert fit_at_word(text, 500) == ('A' * 300) + '…'


def test_fit_at_word_strips_whitespace_before_the_ellipsis():
    """A run of whitespace at the cut is stripped, so the result never reads 'word   …'."""
    text = ('A' * 300) + '   ' + ('B' * 300)

    assert fit_at_word(text, 500) == ('A' * 300) + '…'


_PROSE = 'the quick brown fox jumps over the lazy dog ' * 30


@pytest.mark.parametrize(
    ('text', 'limit'),
    [
        ('N' * 600, 1),
        ('N' * 600, 2),
        ('N' * 600, 120),
        ('N' * 600, 500),
        (_PROSE, 1),
        (_PROSE, 2),
        (_PROSE, 120),
        (_PROSE, 500),
        (' ' * 50, 2),
        ('short prose', 120),
    ],
)
def test_fit_at_word_never_exceeds_the_limit_and_marks_exactly_the_cut_text(text, limit):
    """The result fits the limit, and ends in an ellipsis exactly when the text was over it."""
    result = fit_at_word(text, limit)

    assert len(result) <= limit
    assert result.endswith('…') == (len(text) > limit)


def test_truncation_fields_serializes_only_the_present_facts_in_order():
    cut_a = clip_text('abcdef', 2, field='a')
    whole = clip_text('ab', 2, field='b')
    cut_c = elide_text('abcdef', 3, field='c')

    assert truncation_fields(cut_a.fact, whole.fact, cut_c.fact) == {
        TRUNCATION_FACTS_KEY: [
            {'field': 'a', 'cap': 2, 'elided': 4},
            {'field': 'c', 'cap': 3, 'elided': 3},
        ],
    }


def test_truncation_fields_serializes_an_unknown_elided_count_as_none():
    fact = TruncationFact(field='triage_candidates', cap=20, elided=None)

    assert truncation_fields(fact) == {
        TRUNCATION_FACTS_KEY: [{'field': 'triage_candidates', 'cap': 20, 'elided': None}],
    }


def test_truncation_fields_is_empty_when_nothing_was_cut():
    assert truncation_fields(clip_text('ab', 2, field='a').fact, clip_text(None, 2, field='b').fact) == {}
    assert truncation_fields() == {}


def test_the_dict_channel_key_is_not_the_bool_truncated_key():
    assert TRUNCATION_FACTS_KEY == 'truncation_facts'


@pytest.mark.parametrize('cap', [0, -1])
def test_a_fact_rejects_a_non_positive_cap(cap):
    with pytest.raises(ValueError, match='cap'):
        TruncationFact(field='f', cap=cap, elided=1)


@pytest.mark.parametrize('elided', [0, -3])
def test_a_fact_rejects_a_non_positive_elided_count(elided):
    with pytest.raises(ValueError, match='elided'):
        TruncationFact(field='f', cap=1, elided=elided)


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


def _done_window_note(*, shown: int, total: int) -> str:
    return window_note(
        shown=shown,
        total=total,
        noun='done task(s)',
        window='highest-id (most recent)',
        cap_name='MAX_DONE_TASKS_RETAINED',
        cap=30,
    )


@pytest.mark.parametrize(('shown', 'total'), [(0, 0), (3, 3), (30, 30)])
def test_window_note_is_empty_when_the_window_holds_everything(shown, total):
    assert _done_window_note(shown=shown, total=total) == ''


def test_window_note_names_the_window_rather_than_a_coverage_gap():
    note = _done_window_note(shown=30, total=1500)

    assert note == (
        '_NOTE: showing the 30 highest-id (most recent) of 1500 done task(s) '
        '(MAX_DONE_TASKS_RETAINED=30). This section is a window by design, not a complete list._'
    )
    assert 'this cycle' not in note
