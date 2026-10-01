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

CAND_RUNTIME: dict[str, Any] = {
    'judge_seconds': 2.0,
    'judge_model': 'gpt-6.1-sol',
    'usage': {'prompt_tokens': 1000, 'completion_tokens': 500},
}

SYNTH_CASES = [
    _judged('ref', 'w1', 'amended', 't1', judge_seconds=1.0),
    _judged('ref', 'w2', 'amended', 't2', judge_seconds=2.0),
    _judged('ref', 'w3', 'restated', 't3', judge_seconds=3.0),
    _judged('ref', 'w4', 'contested', 't4', judge_seconds=4.0),
    _judged('ref', 'w5', 'amended', 't5', judge_seconds=5.0),
    _judged('ref', 'w6', 'stored', judge_seconds=6.0),
    _deterministic('ref'),
    _judged('cand', 'w1', 'restated', 't1', **CAND_RUNTIME),
    _judged('cand', 'w2', 'contested', 't2', **CAND_RUNTIME),
    _judged('cand', 'w3', 'stored', parse_failure=True, **CAND_RUNTIME),
    _judged('cand', 'w4', 'amended', 't4', **CAND_RUNTIME),
    _judged('cand', 'w5', 'stored', **CAND_RUNTIME),
    _judged('cand', 'w6', 'amended', 't6', **CAND_RUNTIME),
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


def _replaced(cases: list[dict[str, Any]], arm: str, memory_id: str,
              **changes: Any) -> list[dict[str, Any]]:
    return [
        {**row, **changes} if (row['arm'], row['memory_id']) == (arm, memory_id) else row
        for row in cases
    ]


PAIRED_ERRORS = {
    'misfile', 'contested_decision_error', 'missed_contradiction', 'false_contested',
    'true_link_answered_distinct',
}


def _paired_block(
    n_common: int, only_arm: int, only_reference: int, mcnemar_p: float,
    sign_test: tuple[int, int, float],
) -> dict[str, Any]:
    favouring_arm, favouring_reference, sign_p = sign_test
    return {
        'n_common': n_common,
        'only_arm': only_arm,
        'only_reference': only_reference,
        'mcnemar_p': mcnemar_p,
        'parent_sign_test': {
            'groups_favouring_arm': favouring_arm,
            'groups_favouring_reference': favouring_reference,
            'p': sign_p,
        },
    }


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


class TestRuntime:
    def test_reference_arm_runtime(self) -> None:
        assert _synth_result()['arms']['ref']['runtime'] == {
            'judge_calls': 6,
            'untimed_calls': 0,
            'p50_judge_seconds': 3.0,
            'p95_judge_seconds': 6.0,
            'judge_models': ['gpt-4o-mini'],
            'unpriced_calls': 0,
            'cost_per_write_usd': 0.00021,
        }

    def test_candidate_arm_runtime(self) -> None:
        expected = {
            'p50_judge_seconds': 2.0,
            'p95_judge_seconds': 2.0,
            'judge_models': ['gpt-6.1-sol'],
            'cost_per_write_usd': 0.007,
        }
        assert _subset(_synth_result()['arms']['cand']['runtime'], expected) == expected

    @pytest.mark.parametrize('arm', ['ref', 'cand'])
    def test_deterministic_band_rows_are_never_judge_calls(self, arm: str) -> None:
        report = _synth_result()['arms'][arm]
        assert report['runtime']['judge_calls'] == report['population']['n_judge_band']

    @pytest.mark.parametrize('degraded', [
        {'judge_model': 'gpt-9-unpriced'},
        {'usage': None},
    ])
    def test_an_unpriced_call_nulls_the_cost_rather_than_averaging_the_rest(
        self, degraded: dict[str, Any],
    ) -> None:
        cases = [
            _judged('solo', 'w1', 'amended', 't1', **degraded),
            _judged('solo', 'w6', 'stored'),
        ]
        runtime = _score(cases, SYNTH_VERDICTS, reference_arm='solo')['arms']['solo']['runtime']
        assert (runtime['unpriced_calls'], runtime['cost_per_write_usd']) == (1, None)

    def test_an_untimed_call_nulls_both_percentiles(self) -> None:
        cases = [
            _judged('solo', 'w1', 'amended', 't1', judge_seconds=None),
            _judged('solo', 'w6', 'stored'),
        ]
        runtime = _score(cases, SYNTH_VERDICTS, reference_arm='solo')['arms']['solo']['runtime']
        expected = {'untimed_calls': 1, 'p50_judge_seconds': None, 'p95_judge_seconds': None}
        assert _subset(runtime, expected) == expected

    def test_the_result_carries_the_dated_list_price_table(self) -> None:
        prices = _synth_result()['list_prices']
        assert set(prices) == {'as_of', 'source', 'usd_per_million_tokens'}
        assert (prices['as_of'], prices['source']) == (
            '2026-09-30', 'https://developers.openai.com/api/docs/pricing',
        )
        per_model = prices['usd_per_million_tokens']
        assert per_model['gpt-4o-mini'] == {'input': 0.15, 'output': 0.60}
        assert per_model['gpt-6.1-sol'] == {'input': 2.00, 'output': 10.00}


class TestPairedVsReference:
    def test_the_reference_arm_is_not_paired_with_itself(self) -> None:
        assert _synth_result()['arms']['ref']['paired_vs_reference'] is None

    def test_every_per_write_error_is_paired(self) -> None:
        assert set(_synth_result()['arms']['cand']['paired_vs_reference']) == PAIRED_ERRORS

    @pytest.mark.parametrize(('error', 'expected'), [
        ('misfile', _paired_block(6, 0, 1, 1.0, (1, 0, 1.0))),
        ('contested_decision_error', _paired_block(6, 0, 2, 0.5, (2, 0, 0.5))),
        ('missed_contradiction', _paired_block(6, 0, 1, 1.0, (1, 0, 1.0))),
        ('false_contested', _paired_block(6, 0, 1, 1.0, (1, 0, 1.0))),
        ('true_link_answered_distinct', _paired_block(6, 0, 1, 1.0, (1, 0, 1.0))),
    ])
    def test_paired_counts_and_exact_tests(self, error: str, expected: dict[str, Any]) -> None:
        assert _synth_result()['arms']['cand']['paired_vs_reference'][error] == expected

    def test_a_write_judged_in_only_one_arm_is_outside_the_pairing(self) -> None:
        cases = [
            *SYNTH_CASES,
            _case('cand', 'w8', 'amended', 't8', band_winner='p5', **CAND_RUNTIME),
        ]
        result = _score(cases, [*SYNTH_VERDICTS, _vote('w8', 't8', 'SAME', 'a')])
        cand = result['arms']['cand']
        assert cand['population']['n_judge_band'] == 7
        assert {block['n_common'] for block in cand['paired_vs_reference'].values()} == {6}

    def test_a_band_winner_that_differs_between_arms_is_refused(self) -> None:
        cases = _replaced(SYNTH_CASES, 'cand', 'w1', band_winner_id='p9')
        with pytest.raises(ValueError, match='w1'):
            _score(cases, SYNTH_VERDICTS)


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

    @pytest.mark.parametrize(('arm', 'p50', 'p95', 'cost'), [
        (SEED_REFERENCE, 0.872, 1.059, 0.000349),
        (SEED_CANDIDATE, 0.959, 1.263, 0.000954),
    ])
    def test_runtime(self, arm: str, p50: float, p95: float, cost: float) -> None:
        runtime = _seed_result()['arms'][arm]['runtime']
        counts = {
            'judge_calls': 73, 'untimed_calls': 0, 'unpriced_calls': 0,
            'judge_models': ['gpt-4o-mini'],
        }
        assert _subset(runtime, counts) == counts
        assert runtime['p50_judge_seconds'] == pytest.approx(p50, abs=5e-4)
        assert runtime['p95_judge_seconds'] == pytest.approx(p95, abs=5e-4)
        assert runtime['cost_per_write_usd'] == pytest.approx(cost, abs=5e-7)

    @pytest.mark.parametrize(('error', 'expected'), [
        ('misfile', _paired_block(73, 3, 5, 0.7266, (5, 3, 0.7266))),
        ('contested_decision_error', _paired_block(73, 1, 4, 0.375, (4, 1, 0.375))),
        ('missed_contradiction', _paired_block(73, 1, 2, 1.0, (2, 1, 1.0))),
        ('false_contested', _paired_block(73, 0, 2, 0.5, (2, 0, 0.5))),
        ('true_link_answered_distinct', _paired_block(73, 0, 2, 0.5, (2, 0, 0.5))),
    ])
    def test_paired(self, error: str, expected: dict[str, Any]) -> None:
        paired = _seed_result()['arms'][SEED_CANDIDATE]['paired_vs_reference']
        assert paired[error] == expected
