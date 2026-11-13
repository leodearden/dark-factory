"""Contract tests for ``shared.elision``: the house marker and the word-safe cut every clip site shares."""

from __future__ import annotations

from os.path import commonprefix

import pytest

from shared.elision import elide_at_word, elision_marker, word_boundary_head

PROSE = 'the quick brown fox jumps over the lazy dog ' * 30


def test_the_marker_counts_the_elided_chars():
    assert elision_marker(1827) == '…[+1827 chars elided]'


class TestWordBoundaryHead:
    def test_backs_up_to_the_last_whitespace_and_strips_it(self):
        assert word_boundary_head('alpha beta gamma', 12) == 'alpha beta'

    def test_hard_cuts_when_the_only_whitespace_keeps_half_the_budget_or_less(self):
        text = 'a ' + 'N' * 50
        assert word_boundary_head(text, 20) == text[:20]

    def test_a_zero_budget_keeps_nothing(self):
        assert word_boundary_head('anything', 0) == ''


class TestElideAtWord:
    @pytest.mark.parametrize('text', ['', 'short', 'x' * 50])
    def test_text_that_fits_is_returned_unchanged(self, text):
        assert elide_at_word(text, 50) == text

    def test_a_tiny_cap_is_fine_when_nothing_is_cut(self):
        assert elide_at_word('ab', 3) == 'ab'

    def test_prose_is_cut_at_a_word_with_a_counted_marker(self):
        result = elide_at_word(PROSE, 120)
        head = commonprefix([result, PROSE])
        assert len(result) <= 120
        assert result == head + elision_marker(len(PROSE) - len(head))
        assert head
        assert PROSE[len(head)].isspace()

    def test_text_without_whitespace_is_hard_cut(self):
        text = 'N' * 600
        result = elide_at_word(text, 100)
        k = len(commonprefix([result, text]))
        assert result == 'N' * k + elision_marker(600 - k)
        assert len(result) <= 100

    @pytest.mark.parametrize(
        ('text', 'cap'),
        [
            ('N' * 600, 30),
            ('N' * 600, 100),
            ('N' * 600, 200),
            (PROSE, 30),
            (PROSE, 120),
            (PROSE, 500),
            ('short prose', 120),
        ],
    )
    def test_the_result_never_exceeds_the_cap_and_is_unchanged_only_when_it_fits(self, text, cap):
        result = elide_at_word(text, cap)
        assert len(result) <= cap
        assert (result == text) == (len(text) <= cap)

    def test_a_cap_that_cannot_hold_the_marker_is_refused(self):
        with pytest.raises(ValueError, match='cap'):
            elide_at_word('N' * 600, 10)
