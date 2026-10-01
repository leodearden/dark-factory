"""Tests for score_write_triage_pairs.py — the C2'' write-triage flip-gate scorer.

The scorer judges each arm by the pair it attached a write to, read against a
blind verdict corpus (one or more raters per (entry, target) pair). These
tests reach it only through its public surface: ``resolve_verdicts``,
``score_pairs`` and the ``main`` CLI.

The script is loaded by path (``scripts/`` is not a package), lazily through
``_mod()``, exactly as ``test_eval_write_triage_judge.py`` does.

Seed fixtures, both committed under ``tests/fixtures/`` and both taken from the
gitignored bundle ``data/write-triage-jev-trial-2026-09-29/`` in the main
checkout:

- ``write_triage_pair_verdicts_seed.jsonl`` is a VERBATIM copy of that bundle's
  ``verdicts/verdict_cache.jsonl``: 155 blind Opus ratings, one rater per pair.
  It is the comparability anchor the rater brief names, so it is never edited.
- ``write_triage_pair_cases_seed.jsonl`` is derived from the bundle's
  ``inputs/cases_cosine@5.jsonl`` and ``inputs/cases_cosine@20.jsonl`` (the
  gpt-4o-mini 09-29 reference runs). It is reduced to the scorer's input-row
  contract and keeps only ids, band, outcome, usage and latency, with no memory
  text. In those runs ``attach_target_id`` equals ``x_judged_candidate_id`` on
  every judge-band row, and every judge-band attach pair is rated in the seed.

The seed-regression expectations were computed independently with the bundle's
reference definitions (``work/score.py`` wilson and mcnemar_exact,
``work/analyze_checks.py`` contested_decision_error and paired_test). They are
an oracle: if the scorer disagrees, the scorer is wrong.
"""
from __future__ import annotations

import functools
import types
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'score_write_triage_pairs.py'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, mod_name='score_write_triage_pairs')


def _vote(
    entry: str, target: str, verdict: str, rater: str, batch: str = 'b1',
) -> dict[str, Any]:
    return {
        'entry_id': entry, 'target_id': target, 'verdict': verdict,
        'rater': rater, 'batch': batch,
    }


def _resolve_one(*verdicts: str) -> Any:
    rows = [_vote('e', 't', word, f'r{i}') for i, word in enumerate(verdicts)]
    return _mod().resolve_verdicts(rows)[('e', 't')]


class TestResolveVerdicts:
    def test_verdict_words_are_the_rater_briefs_seven_in_brief_order(self) -> None:
        assert _mod().VERDICT_WORDS == (
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
        assert _resolve_one(word) == _mod().Resolution(expected)

    @pytest.mark.parametrize(('votes', 'expected'), [
        (('CORRECTS', 'CORRECTS', 'RELATED'), 'corrects'),
        (('SAME', 'RELATED', 'EXTENDS'), 'agreeing'),
        (('RELATED', 'UNRELATED', 'SAME'), 'misfile'),
        (('CORRECTS', 'SAME', 'RELATED'), 'agreeing'),
    ])
    def test_three_raters_take_the_majority_on_each_binary(
        self, votes: tuple[str, ...], expected: str,
    ) -> None:
        assert _resolve_one(*votes) == _mod().Resolution(expected)

    @pytest.mark.parametrize('votes', [
        ('SAME', 'RELATED'),
        ('CORRECTS', 'SAME'),
    ])
    def test_two_raters_split_on_either_binary_is_a_tie(self, votes: tuple[str, ...]) -> None:
        assert _resolve_one(*votes) == _mod().Resolution.TIED

    def test_unclear_abstains_from_the_majority(self) -> None:
        assert _resolve_one('UNCLEAR', 'SAME') == _mod().Resolution.AGREEING

    def test_only_unclear_votes_resolve_unclear(self) -> None:
        assert _resolve_one('UNCLEAR', 'UNCLEAR') == _mod().Resolution.UNCLEAR

    def test_belongs_is_true_exactly_for_agreeing_and_corrects(self) -> None:
        resolution = _mod().Resolution
        belonging = {member for member in resolution if member.belongs}
        assert belonging == {resolution.AGREEING, resolution.CORRECTS}

    def test_pairs_are_resolved_independently(self) -> None:
        rows = [
            _vote('e1', 't1', 'SAME', 'a'),
            _vote('e1', 't2', 'RELATED', 'a'),
            _vote('e2', 't1', 'CORRECTS', 'a'),
        ]
        resolution = _mod().Resolution
        assert _mod().resolve_verdicts(rows) == {
            ('e1', 't1'): resolution.AGREEING,
            ('e1', 't2'): resolution.MISFILE,
            ('e2', 't1'): resolution.CORRECTS,
        }

    @pytest.mark.parametrize('word', ['DUPLICATE', 'same'])
    def test_a_word_outside_the_vocabulary_is_refused(self, word: str) -> None:
        with pytest.raises(ValueError, match=r"(?s)e9.*t9|t9.*e9"):
            _mod().resolve_verdicts([_vote('e9', 't9', word, 'a')])

    @pytest.mark.parametrize('missing', ['entry_id', 'target_id', 'verdict', 'rater', 'batch'])
    def test_a_row_missing_a_required_key_is_refused(self, missing: str) -> None:
        row = _vote('e9', 't9', 'SAME', 'a')
        del row[missing]
        with pytest.raises(ValueError, match=missing):
            _mod().resolve_verdicts([row])

    def test_one_rater_voting_twice_on_one_pair_is_refused(self) -> None:
        rows = [
            _vote('e9', 't9', 'SAME', 'opus-r3', batch='b1'),
            _vote('e9', 't9', 'SAME', 'opus-r3', batch='anchors'),
        ]
        with pytest.raises(ValueError, match=r"(?s)opus-r3.*e9.*t9|e9.*t9.*opus-r3"):
            _mod().resolve_verdicts(rows)

    def test_extra_keys_are_accepted(self) -> None:
        row = {**_vote('e', 't', 'SAME', 'a'), 'reason': 'same claim, reworded'}
        assert _mod().resolve_verdicts([row]) == {('e', 't'): _mod().Resolution.AGREEING}
