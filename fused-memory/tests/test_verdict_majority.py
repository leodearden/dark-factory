"""Tests for maintenance/verdict_majority.py — the raters' majority rule."""
from __future__ import annotations

from typing import Any

import pytest

from fused_memory.maintenance.verdict_majority import VERDICT_WORDS, Resolution, resolve_verdicts


def _vote(
    entry: str, target: str, verdict: str, rater: str, batch: str = 'b1',
) -> dict[str, Any]:
    return {
        'entry_id': entry, 'target_id': target, 'verdict': verdict,
        'rater': rater, 'batch': batch,
    }


def _resolve_one(*verdicts: str) -> Resolution:
    rows = [_vote('e', 't', word, f'r{i}') for i, word in enumerate(verdicts)]
    return resolve_verdicts(rows)[('e', 't')]


class TestResolveVerdicts:
    def test_verdict_words_are_the_rater_briefs_seven_in_brief_order(self) -> None:
        assert VERDICT_WORDS == (
            'SAME', 'EXTENDS', 'SUBSUMED', 'CORRECTS', 'RELATED', 'UNRELATED', 'UNCLEAR',
        )

    @pytest.mark.parametrize(('word', 'expected'), [
        ('SAME', 'agreeing'),
        ('EXTENDS', 'agreeing'),
        ('SUBSUMED', 'agreeing'),
        ('CORRECTS', 'corrects'),
        ('RELATED', 'misfile'),
        ('UNRELATED', 'misfile'),
        ('UNCLEAR', 'unclear'),
    ])
    def test_a_single_rater_resolves_by_word_class(self, word: str, expected: str) -> None:
        assert _resolve_one(word) == Resolution(expected)

    @pytest.mark.parametrize(('votes', 'expected'), [
        (('CORRECTS', 'CORRECTS', 'RELATED'), 'corrects'),
        (('SAME', 'RELATED', 'EXTENDS'), 'agreeing'),
        (('RELATED', 'UNRELATED', 'SAME'), 'misfile'),
        (('CORRECTS', 'SAME', 'RELATED'), 'agreeing'),
    ])
    def test_three_raters_take_the_majority_on_each_binary(
        self, votes: tuple[str, ...], expected: str,
    ) -> None:
        assert _resolve_one(*votes) == Resolution(expected)

    @pytest.mark.parametrize('votes', [
        ('SAME', 'RELATED'),
        ('CORRECTS', 'SAME'),
    ])
    def test_two_raters_split_on_either_binary_is_a_tie(self, votes: tuple[str, ...]) -> None:
        assert _resolve_one(*votes) == Resolution.TIED

    def test_unclear_abstains_from_the_majority(self) -> None:
        assert _resolve_one('UNCLEAR', 'SAME') == Resolution.AGREEING

    def test_only_unclear_votes_resolve_unclear(self) -> None:
        assert _resolve_one('UNCLEAR', 'UNCLEAR') == Resolution.UNCLEAR

    def test_belongs_is_true_exactly_for_agreeing_and_corrects(self) -> None:
        belonging = {member for member in Resolution if member.belongs}
        assert belonging == {Resolution.AGREEING, Resolution.CORRECTS}

    def test_pairs_are_resolved_independently(self) -> None:
        rows = [
            _vote('e1', 't1', 'SAME', 'a'),
            _vote('e1', 't2', 'RELATED', 'a'),
            _vote('e2', 't1', 'CORRECTS', 'a'),
        ]
        assert resolve_verdicts(rows) == {
            ('e1', 't1'): Resolution.AGREEING,
            ('e1', 't2'): Resolution.MISFILE,
            ('e2', 't1'): Resolution.CORRECTS,
        }

    @pytest.mark.parametrize('word', ['DUPLICATE', 'same'])
    def test_a_word_outside_the_vocabulary_is_refused(self, word: str) -> None:
        with pytest.raises(ValueError, match=r"(?s)e9.*t9|t9.*e9"):
            resolve_verdicts([_vote('e9', 't9', word, 'a')])

    @pytest.mark.parametrize('missing', ['entry_id', 'target_id', 'verdict', 'rater', 'batch'])
    def test_a_row_missing_a_required_key_is_refused(self, missing: str) -> None:
        row = _vote('e9', 't9', 'SAME', 'a')
        del row[missing]
        with pytest.raises(ValueError, match=missing):
            resolve_verdicts([row])

    def test_one_rater_voting_twice_on_one_pair_is_refused(self) -> None:
        rows = [
            _vote('e9', 't9', 'SAME', 'opus-r3', batch='b1'),
            _vote('e9', 't9', 'SAME', 'opus-r3', batch='anchors'),
        ]
        with pytest.raises(ValueError, match=r"(?s)opus-r3.*e9.*t9|e9.*t9.*opus-r3"):
            resolve_verdicts(rows)

    def test_extra_keys_are_accepted(self) -> None:
        row = {**_vote('e', 't', 'SAME', 'a'), 'reason': 'same claim, reworded'}
        assert resolve_verdicts([row]) == {('e', 't'): Resolution.AGREEING}
