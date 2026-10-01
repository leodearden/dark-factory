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
import json
import types
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'score_write_triage_pairs.py'
FIXTURES = Path(__file__).parent / 'fixtures'


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


_USAGE = {'prompt_tokens': 1000, 'completion_tokens': 100}


def _case(
    arm: str,
    memory_id: str,
    outcome: str,
    judged: str | None = None,
    *,
    band: str = 'judge',
    band_winner: str,
    parse_failure: bool = False,
    judge_seconds: float | None = 1.0,
    judge_model: str | None = 'gpt-4o-mini',
    usage: dict[str, int] | None = _USAGE,
) -> dict[str, Any]:
    return {
        'arm': arm, 'memory_id': memory_id, 'band': band, 'outcome': outcome,
        'judged_candidate_id': judged, 'band_winner_id': band_winner,
        'parse_failure': parse_failure, 'judge_seconds': judge_seconds,
        'judge_model': judge_model, 'usage': dict(usage) if usage is not None else None,
    }


BAND_WINNER = {'w1': 'p1', 'w2': 'p1', 'w3': 'p2', 'w4': 'p2', 'w5': 'p3', 'w6': 'p3'}


def _judged(arm: str, memory_id: str, outcome: str, judged: str | None = None,
            **fields: Any) -> dict[str, Any]:
    return _case(arm, memory_id, outcome, judged, band_winner=BAND_WINNER[memory_id], **fields)


def _deterministic(arm: str) -> dict[str, Any]:
    return _case(
        arm, 'w7', 'restated', band='restated', band_winner='p4',
        judge_seconds=None, judge_model=None, usage=None,
    )


SYNTH_VERDICTS = [
    _vote('w1', 't1', 'SAME', 'a'),
    _vote('w2', 't2', 'CORRECTS', 'a'),
    _vote('w2', 't2', 'CORRECTS', 'b'),
    _vote('w2', 't2', 'RELATED', 'c'),
    _vote('w3', 't3', 'RELATED', 'a'),
    _vote('w3', 't3', 'UNRELATED', 'b'),
    _vote('w4', 't4', 'SAME', 'a'),
    _vote('w4', 't4', 'RELATED', 'b'),
    _vote('w4', 't4', 'EXTENDS', 'c'),
    _vote('w5', 't5', 'UNCLEAR', 'a'),
    _vote('w6', 't6', 'EXTENDS', 'a'),
]

SYNTH_CASES = [
    _judged('ref', 'w1', 'amended', 't1'),
    _judged('ref', 'w2', 'amended', 't2'),
    _judged('ref', 'w3', 'restated', 't3'),
    _judged('ref', 'w4', 'contested', 't4'),
    _judged('ref', 'w5', 'amended', 't5'),
    _judged('ref', 'w6', 'stored'),
    _deterministic('ref'),
    _judged('cand', 'w1', 'restated', 't1'),
    _judged('cand', 'w2', 'contested', 't2'),
    _judged('cand', 'w3', 'stored', parse_failure=True),
    _judged('cand', 'w4', 'amended', 't4'),
    _judged('cand', 'w5', 'stored'),
    _judged('cand', 'w6', 'amended', 't6'),
    _deterministic('cand'),
]

QUALITY_KEYS = {
    'attaches', 'unclear_attaches', 'misfiles', 'misfile_rate_of_attaches',
    'misfile_rate_wilson95', 'duplicates_placed', 'contradictions_attached',
    'contradictions_contested', 'contradiction_recall', 'agreeing_attaches',
    'agreeing_contested', 'false_contested_rate', 'true_links',
    'true_links_answered_distinct', 'true_links_answered_distinct_rate',
    'contested_decision_errors', 'parse_failures', 'unrated_pairs', 'unrated_pair_ids',
}


def _score(cases: list[dict[str, Any]], verdicts: list[dict[str, Any]],
           reference_arm: str = 'ref') -> dict[str, Any]:
    return _mod().score_pairs(cases, verdicts, reference_arm=reference_arm)


@functools.cache
def _synth_result() -> dict[str, Any]:
    return _score(SYNTH_CASES, SYNTH_VERDICTS)


def _subset(block: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    return {key: block[key] for key in expected}


def _read_fixture(name: str) -> list[dict[str, Any]]:
    with (FIXTURES / name).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


SEED_REFERENCE = 'gpt-4o-mini@5'
SEED_CANDIDATE = 'gpt-4o-mini@20'


@functools.cache
def _seed_result() -> dict[str, Any]:
    return _score(
        _read_fixture('write_triage_pair_cases_seed.jsonl'),
        _read_fixture('write_triage_pair_verdicts_seed.jsonl'),
        reference_arm=SEED_REFERENCE,
    )


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


class TestQuality:
    def test_the_result_names_the_reference_and_every_arm(self) -> None:
        result = _synth_result()
        assert result['reference_arm'] == 'ref'
        assert set(result['arms']) == {'ref', 'cand'}

    def test_the_verdict_corpus_is_summarised(self) -> None:
        assert _synth_result()['verdict_corpus'] == {
            'pairs': 6,
            'ratings': 11,
            'raters': ['a', 'b', 'c'],
            'resolutions': {
                'agreeing': 3, 'corrects': 1, 'misfile': 1, 'unclear': 1, 'tied': 0,
            },
        }

    @pytest.mark.parametrize('arm', ['ref', 'cand'])
    def test_population_counts_every_row_but_only_the_judge_band_is_scored(
        self, arm: str,
    ) -> None:
        population = _synth_result()['arms'][arm]['population']
        assert population == {'n_cases': 7, 'n_judge_band': 6}

    def test_reference_arm_quality(self) -> None:
        quality = _synth_result()['arms']['ref']['quality']
        assert quality == {
            'attaches': 5,
            'unclear_attaches': 1,
            'misfiles': 1,
            'misfile_rate_of_attaches': 0.25,
            'misfile_rate_wilson95': [0.0456, 0.6994],
            'duplicates_placed': 3,
            'contradictions_attached': 1,
            'contradictions_contested': 0,
            'contradiction_recall': 0.0,
            'agreeing_attaches': 2,
            'agreeing_contested': 1,
            'false_contested_rate': 0.5,
            'true_links': 4,
            'true_links_answered_distinct': 1,
            'true_links_answered_distinct_rate': 0.25,
            'contested_decision_errors': 2,
            'parse_failures': 0,
            'unrated_pairs': 0,
            'unrated_pair_ids': [],
        }

    def test_candidate_arm_quality(self) -> None:
        expected = {
            'attaches': 4,
            'unclear_attaches': 0,
            'misfiles': 0,
            'misfile_rate_of_attaches': 0.0,
            'duplicates_placed': 4,
            'contradictions_attached': 1,
            'contradictions_contested': 1,
            'contradiction_recall': 1.0,
            'agreeing_attaches': 3,
            'agreeing_contested': 0,
            'false_contested_rate': 0.0,
            'true_links': 4,
            'true_links_answered_distinct': 0,
            'true_links_answered_distinct_rate': 0.0,
            'contested_decision_errors': 0,
            'parse_failures': 1,
            'unrated_pairs': 0,
        }
        assert _subset(_synth_result()['arms']['cand']['quality'], expected) == expected

    def test_empty_denominators_read_none_never_zero(self) -> None:
        result = _score(
            [_judged('solo', 'w1', 'amended', 't1')],
            [_vote('w1', 't1', 'UNCLEAR', 'a')],
            reference_arm='solo',
        )
        expected = {
            'misfile_rate_of_attaches': None,
            'misfile_rate_wilson95': None,
            'contradiction_recall': None,
            'false_contested_rate': None,
            'true_links_answered_distinct_rate': None,
        }
        assert _subset(result['arms']['solo']['quality'], expected) == expected

    @pytest.mark.parametrize('arm', ['ref', 'cand'])
    def test_every_arm_reports_every_quality_key(self, arm: str) -> None:
        assert set(_synth_result()['arms'][arm]['quality']) == QUALITY_KEYS


class TestSeedRegression:
    def test_verdict_corpus(self) -> None:
        assert _seed_result()['verdict_corpus'] == {
            'pairs': 155,
            'ratings': 155,
            'raters': [f'opus-r{i}' for i in range(6)],
            'resolutions': {
                'agreeing': 114, 'corrects': 21, 'misfile': 20, 'unclear': 0, 'tied': 0,
            },
        }

    @pytest.mark.parametrize('arm', [SEED_REFERENCE, SEED_CANDIDATE])
    def test_population(self, arm: str) -> None:
        assert _seed_result()['arms'][arm]['population'] == {
            'n_cases': 84, 'n_judge_band': 73,
        }

    @pytest.mark.parametrize(('arm', 'expected'), [
        (SEED_REFERENCE, {
            'attaches': 66,
            'unclear_attaches': 0,
            'misfiles': 6,
            'misfile_rate_of_attaches': 0.0909,
            'misfile_rate_wilson95': [0.0423, 0.1845],
            'duplicates_placed': 60,
            'contradictions_attached': 6,
            'contradictions_contested': 0,
            'contradiction_recall': 0.0,
            'agreeing_attaches': 54,
            'agreeing_contested': 3,
            'false_contested_rate': 0.0556,
            'true_links': 68,
            'true_links_answered_distinct': 3,
            'true_links_answered_distinct_rate': 0.0441,
            'contested_decision_errors': 9,
            'parse_failures': 0,
            'unrated_pairs': 0,
        }),
        (SEED_CANDIDATE, {
            'attaches': 69,
            'unclear_attaches': 0,
            'misfiles': 4,
            'misfile_rate_of_attaches': 0.058,
            'misfile_rate_wilson95': [0.0228, 0.1398],
            'duplicates_placed': 65,
            'contradictions_attached': 6,
            'contradictions_contested': 1,
            'contradiction_recall': 0.1667,
            'agreeing_attaches': 59,
            'agreeing_contested': 1,
            'false_contested_rate': 0.0169,
            'true_links': 68,
            'true_links_answered_distinct': 1,
            'true_links_answered_distinct_rate': 0.0147,
            'contested_decision_errors': 6,
            'parse_failures': 2,
            'unrated_pairs': 0,
        }),
    ])
    def test_quality(self, arm: str, expected: dict[str, Any]) -> None:
        assert _subset(_seed_result()['arms'][arm]['quality'], expected) == expected
