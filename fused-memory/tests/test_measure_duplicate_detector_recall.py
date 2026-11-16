"""Tests for measure_duplicate_detector_recall.py (task 10243).

The script replays the 2026-08-11 ANN calibration protocol on the task 3130
fixture and measures adjudicators over its ANN candidates. Every arm here is a
fake and every vector is hand-made, so each expected number is hand-computable
from the literals in the test. Nothing touches the network, torch or a store.
"""
from __future__ import annotations

import contextlib
import dataclasses
import datetime
import functools
import json
import types
from pathlib import Path

import pytest
from _fm_helpers import load_script_module

SCRIPTS = Path(__file__).parent.parent / 'scripts'

_PK = 'procedural_knowledge'
_OS = 'observations_and_summaries'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'measure_duplicate_detector_recall.py', 'measure_duplicate_detector_recall',
    )


@functools.cache
def _calibrate() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'calibrate_write_triage.py', 'calibrate_write_triage')


@functools.cache
def _arms() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'eval_write_triage_reranker_arms.py', 'eval_write_triage_reranker_arms',
    )


@functools.cache
def _audit_mod() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'audit_duplicate_memories.py', 'audit_duplicate_memories')


@functools.cache
def _report_schema() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'duplicate_cluster_report.py', 'duplicate_cluster_report')


def _context():
    return _arms().ArmContext(
        device='cpu', local_batch_size=1, vram_cap_gib=1.0, pairwise_concurrency=1,
    )


def _record(memory_id: str, cluster_id: str, label: str, category: str | None) -> dict:
    return {
        'memory_id': memory_id, 'cluster_id': cluster_id, 'label': label,
        'category': category, 'content': f'the text of {memory_id}',
    }


def _scan(records: list[dict], vectors: dict[str, list[float]]) -> list[dict]:
    return _mod().fixture_scan(records, lambda memory_id, content: vectors[memory_id])


class TestFixtureReplay:
    def test_fixture_scan_embeds_each_record_once_in_fixture_order(self):
        records = [_record('m2', 'c1', 'canonical', _PK), _record('m1', 'c1', 'duplicate', _OS)]
        calls: list[tuple[str, str]] = []

        def embed(memory_id: str, content: str) -> list[float]:
            calls.append((memory_id, content))
            return [float(len(calls)), 0.0]

        memories = _mod().fixture_scan(records, embed)

        assert calls == [('m2', 'the text of m2'), ('m1', 'the text of m1')]
        assert memories == [
            {'id': 'm2', 'content': 'the text of m2', 'category': _PK, 'vector': [1.0, 0.0]},
            {'id': 'm1', 'content': 'the text of m1', 'category': _OS, 'vector': [2.0, 0.0]},
        ]

    def test_fixture_neighbors_mirror_the_per_category_top_k_query(self):
        records = [
            _record('a', 'c1', 'canonical', _PK),
            _record('b', 'c1', 'duplicate', _PK),
            _record('c', 'c1', 'duplicate', _PK),
            _record('x', 'c2', 'canonical', _OS),
            _record('n', 'c3', 'canonical', None),
        ]
        memories = _scan(records, {
            'a': [1.0, 0.0], 'b': [0.8, 0.6], 'c': [0.6, 0.8],
            'x': [1.0, 0.0], 'n': [0.0, 1.0],
        })

        neighbors = _mod().fixture_neighbors(memories, top_k=1)

        assert neighbors == {
            'a': [{'id': 'b', 'score': pytest.approx(0.8)}],
            'b': [{'id': 'c', 'score': pytest.approx(0.96)}],
            'c': [{'id': 'b', 'score': pytest.approx(0.96)}],
            'x': [],
            'n': [{'id': 'c', 'score': pytest.approx(0.8)}],
        }

    def test_a_cosine_tie_is_broken_by_id(self):
        records = [
            _record('m', 'c1', 'canonical', _PK),
            _record('z', 'c1', 'duplicate', _PK),
            _record('y', 'c1', 'duplicate', _PK),
        ]
        memories = _scan(records, {'m': [1.0, 0.0], 'z': [0.6, 0.8], 'y': [0.6, 0.8]})

        neighbors = _mod().fixture_neighbors(memories, top_k=1)

        assert neighbors['m'] == [{'id': 'y', 'score': pytest.approx(0.6)}]

    def test_the_window_counts_the_self_hit_before_stripping_it(self):
        records = [
            _record('a', 'c1', 'canonical', _PK),
            _record('b', 'c1', 'duplicate', _PK),
            _record('c', 'c1', 'duplicate', _PK),
        ]
        memories = _scan(records, {'a': [1.0, 0.0], 'b': [0.8, 0.6], 'c': [0.6, 0.8]})

        neighbors = _mod().fixture_neighbors(memories, top_k=2)

        assert [hit['id'] for hit in neighbors['a']] == ['b', 'c']

    def test_labelled_pair_classes_agree_with_build_pair_sets(self):
        records = [
            _record('m0', 'C1', 'canonical', _PK),
            _record('m1', 'C1', 'duplicate', _PK),
            _record('m2', 'C1', 'distinct', _PK),
            _record('m3', 'C2', 'canonical', _OS),
        ]

        classes = _mod().labelled_pair_classes(records)

        assert classes == {
            (0, 1): 'true_dup_pairs',
            (0, 2): 'hard_negative_pairs',
            (1, 2): 'hard_negative_pairs',
            (0, 3): 'unrelated_pairs',
            (1, 3): 'unrelated_pairs',
            (2, 3): 'unrelated_pairs',
        }
        index = {r['memory_id']: i for i, r in enumerate(records)}
        expected = {
            tuple(sorted((index[p['a']], index[p['b']]))): name
            for name, pairs in _calibrate().build_pair_sets(records).items()
            for p in pairs
        }
        assert classes == expected

    def _generation_corpus(self) -> tuple[list[dict], list[dict]]:
        records = [
            _record('a', 'C1', 'canonical', _PK),
            _record('b', 'C1', 'duplicate', _OS),
            _record('c', 'C2', 'canonical', _PK),
            _record('d', 'C2', 'duplicate', _PK),
            _record('e', 'C2', 'distinct', _PK),
            _record('f', 'C3', 'canonical', _PK),
            _record('g', 'C3', 'duplicate', _PK),
        ]
        memories = _scan(records, {
            'a': [1.0, 0.0], 'b': [1.0, 0.0],
            'c': [0.0, 1.0], 'd': [0.6, 0.8], 'e': [0.28, 0.96],
            'f': [-1.0, 0.0], 'g': [-0.8, 0.6],
        })
        return records, memories

    def test_candidate_generation_counts_the_generator_ceiling(self):
        records, memories = self._generation_corpus()
        neighbors = _mod().fixture_neighbors(memories, top_k=1)
        classes = _mod().labelled_pair_classes(records)

        generation = _mod().candidate_generation(memories, neighbors, classes, top_k=1)

        assert generation.candidates == ((0, 3), (2, 4), (3, 4), (5, 6))
        assert generation.true_caught.hits == 1
        assert generation.true_caught.total == 3
        assert generation.true_caught.rate == pytest.approx(1 / 3)
        lo, hi = generation.true_caught.wilson_95
        assert 0.0 < lo < 1 / 3 < hi < 1.0
        assert generation.hard_negative_generated == 2
        assert generation.cross_category_true_pairs == 1
        assert generation.disclosure['top_k_saturated'] == 6
        assert generation.disclosure['missing_vector'] == 0
        assert generation.disclosure['below_threshold_dropped'] == 0


class TestOperatingPointAndIntervals:
    def test_hanley_mcneil_matches_the_hand_computed_interval(self):
        lo, hi = _mod().auc_interval_95(0.8, 301, 18)

        assert lo == pytest.approx(0.8 - 1.96 * 0.04158, abs=1e-3)
        assert hi == pytest.approx(0.8 + 1.96 * 0.04158, abs=1e-3)
        assert (lo, hi) == pytest.approx((0.7185, 0.8815), abs=1e-3)

    def test_the_interval_is_clamped_to_the_unit_range(self):
        assert _mod().auc_interval_95(1.0, 301, 18) == (1.0, 1.0)
        assert _mod().auc_interval_95(0.99, 5, 3)[1] == 1.0

    def test_an_empty_class_has_no_interval(self):
        assert _mod().auc_interval_95(0.8, 0, 18) is None
        assert _mod().auc_interval_95(0.8, 301, 0) is None

    # p1 0.9 T, p2 0.8 F, p3 0.7 T, p4 0.6 T, p5 0.5 F
    _SCORES = {(0, 1): 0.9, (0, 2): 0.8, (0, 3): 0.7, (0, 4): 0.6, (0, 5): 0.5}
    _POSITIVES = frozenset({(0, 1), (0, 3), (0, 4)})

    def _point(self, scores, positives, floor, n_positive=10):
        return _mod().operating_point(scores, positives, n_positive, floor)

    def test_the_most_caught_threshold_meeting_the_floor_wins(self):
        point = self._point(self._SCORES, self._POSITIVES, 0.75)

        assert (point.threshold, point.caught, point.flagged) == (0.6, 3, 4)

    def test_a_stricter_floor_keeps_only_the_qualifying_prefix(self):
        point = self._point(self._SCORES, self._POSITIVES, 0.8)

        assert (point.threshold, point.caught, point.flagged) == (0.9, 1, 1)

    def test_recall_divides_by_every_true_pair_not_the_generated_ones(self):
        point = self._point(self._SCORES, self._POSITIVES, 0.75, n_positive=10)

        assert point.recall.hits == 3
        assert point.recall.total == 10

    def test_no_qualifying_threshold_is_no_operating_point(self):
        point = self._point({(0, 2): 0.8, (0, 5): 0.5}, self._POSITIVES, 0.5)

        assert (point.threshold, point.caught, point.flagged) == (None, 0, 0)

    def test_tied_scores_are_flagged_together(self):
        point = self._point({(0, 1): 0.9, (0, 2): 0.9}, self._POSITIVES, 0.5)

        assert (point.threshold, point.caught, point.flagged) == (0.9, 1, 2)

    def test_a_tie_on_caught_goes_to_the_higher_threshold(self):
        point = self._point({(0, 1): 0.9, (0, 2): 0.5}, self._POSITIVES, 0.5)

        assert (point.threshold, point.caught, point.flagged) == (0.9, 1, 1)

    def test_recall_at_precision_reports_one_point_per_floor(self):
        points = _mod().recall_at_precision(self._SCORES, self._POSITIVES, 10)

        assert _mod().RECALL_AT_PRECISION_FLOORS == (0.80, 0.90, 0.95)
        assert points == {
            floor: _mod().operating_point(self._SCORES, self._POSITIVES, 10, floor)
            for floor in (0.80, 0.90, 0.95)
        }


class _TableScorer:
    """A fake arms Scorer: ``table[(entry, candidate)]`` per candidate, with per-call extras."""

    def __init__(self, table, *, costs=(), over=(), facts=None):
        self.table = table
        self.costs = list(costs)
        self.over = list(over)
        self._facts = facts
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score(self, entry, candidate_texts):
        call = len(self.calls)
        self.calls.append((entry, tuple(candidate_texts)))
        return _arms().SlateScores(
            scores=tuple(self.table[(entry, c)] for c in candidate_texts),
            cost_usd=self.costs[call] if call < len(self.costs) else 0.0,
            pairs_over_max_length=self.over[call] if call < len(self.over) else 0,
        )

    def facts(self):
        return self._facts


class TestClaimLevelArms:
    def test_split_claims_cuts_at_lines_and_sentence_ends(self):
        assert _mod().split_claims('A is true. B holds.\n- C\n\nD') == [
            'A is true.', 'B holds.', '- C', 'D',
        ]

    def test_a_dotted_path_is_not_a_sentence_end(self):
        assert _mod().split_claims('see foo.py::bar now') == ['see foo.py::bar now']

    def test_text_with_no_boundary_is_one_claim(self):
        assert _mod().split_claims('  one claim only  ') == ['one claim only']

    _TABLE = {
        ('E1.', 'C1.'): 0.2, ('E1.', 'D1.'): 0.1, ('E1.', 'D2.'): 0.9, ('E1.', 'D3'): 0.3,
        ('E2.', 'C1.'): 0.6, ('E2.', 'D1.'): 0.5, ('E2.', 'D2.'): 0.4, ('E2.', 'D3'): 0.0,
    }

    def test_a_candidate_scores_the_mean_over_entry_claims_of_its_best_claim(self):
        inner = _TableScorer(self._TABLE)

        slate = _mod().ClaimLevelScorer(inner).score('E1. E2.', ['C1.', 'D1. D2.\nD3'])

        assert slate.scores == pytest.approx(((0.2 + 0.6) / 2, (0.9 + 0.5) / 2))
        assert inner.calls == [
            ('E1.', ('C1.', 'D1.', 'D2.', 'D3')),
            ('E2.', ('C1.', 'D1.', 'D2.', 'D3')),
        ]

    def test_cost_and_truncation_sum_over_inner_calls(self):
        inner = _TableScorer(self._TABLE, costs=[0.01, 0.02], over=[1, 2])

        slate = _mod().ClaimLevelScorer(inner).score('E1. E2.', ['C1.'])

        assert slate.cost_usd == pytest.approx(0.03)
        assert slate.pairs_over_max_length == 3

    def test_an_unmeasured_inner_cost_or_truncation_is_unmeasured(self):
        inner = _TableScorer(self._TABLE, costs=[0.01, None], over=[None, 2])

        slate = _mod().ClaimLevelScorer(inner).score('E1. E2.', ['C1.'])

        assert slate.cost_usd is None
        assert slate.pairs_over_max_length is None

    def test_facts_delegate_to_the_inner_scorer(self):
        facts = _arms().ScorerFacts(device='cpu', vram_peak_mib=None, max_length=512)

        assert _mod().ClaimLevelScorer(_TableScorer({}, facts=facts)).facts() is facts

    def test_detector_arms_are_d1_minus_jev_plus_claims_over_local_cross_encoders(self):
        arms_mod = _arms()
        names = [arm.name for arm in _mod().DETECTOR_ARMS]
        full_text = [a for a in _mod().DETECTOR_ARMS if a.granularity == 'full_text']
        claims = [a for a in _mod().DETECTOR_ARMS if a.granularity == 'claims']
        local = [
            s for s in arms_mod.D1_ARMS
            if s.arm_class == arms_mod.ArmClass.local_cross_encoder
        ]

        assert [a.spec for a in full_text] == [
            s for s in arms_mod.D1_ARMS if s.arm_class != arms_mod.ArmClass.jev_choice
        ]
        assert [a.spec for a in claims] == local
        assert [a.name for a in claims] == [f'{s.name}+claims' for s in local]
        assert len(names) == len(set(names))
        assert 'jev-choice' not in names

    def test_a_claim_arm_opens_its_spec_and_wraps_the_scorer(self):
        sentinel = _TableScorer({})
        opened_with = []

        def open_(context):
            opened_with.append(context)
            return contextlib.nullcontext(sentinel)

        spec = _arms().ArmSpec(
            name='fake', arm_class=_arms().ArmClass.local_cross_encoder, model='m', open=open_,
        )
        full = _mod().DetectorArm(spec, _mod().Granularity.full_text)
        claims = _mod().DetectorArm(spec, _mod().Granularity.claims)
        context = _context()

        with _mod().open_detector_scorer(full, context) as scorer:
            assert scorer is sentinel
        with _mod().open_detector_scorer(claims, context) as scorer:
            assert isinstance(scorer, _mod().ClaimLevelScorer)
            assert scorer.inner is sentinel
        assert opened_with == [context, context]
        assert (full.name, claims.name) == ('fake', 'fake+claims')


def _fake_arm(open_, name='fake-arm', arm_class=None):
    spec = _arms().ArmSpec(
        name=name, arm_class=arm_class or _arms().ArmClass.local_cross_encoder,
        model='fake/model', open=open_,
    )
    return _mod().DetectorArm(spec, _mod().Granularity.full_text)


def _yielding(scorer):
    return lambda context: contextlib.nullcontext(scorer)


class TestMeasureDetectorArm:
    # Six records t0..t5. True pairs (0,1) (2,3) (4,5); hard negatives (0,2)
    # (1,3); every other pair unrelated. The candidates leave out the true
    # pair (4,5) and the hard negative (1,3), which must still be scored.
    _PAIR_SCORES = {
        (0, 1): 0.9, (2, 3): 0.7, (4, 5): 0.8,
        (0, 2): 0.75, (1, 3): 0.3, (1, 4): 0.6,
    }
    _CANDIDATES = ((0, 1), (0, 2), (1, 4), (2, 3))
    _FLOOR = 0.6

    def _memories(self):
        return [
            {'id': f'm{i}', 'content': f't{i}', 'category': _PK, 'vector': [1.0, 0.0]}
            for i in range(6)
        ]

    def _classes(self):
        classes = {(i, j): 'unrelated_pairs' for i in range(6) for j in range(i + 1, 6)}
        classes.update({p: 'true_dup_pairs' for p in ((0, 1), (2, 3), (4, 5))})
        classes.update({p: 'hard_negative_pairs' for p in ((0, 2), (1, 3))})
        return classes

    def _table(self):
        table = {}
        for (i, j), score in self._PAIR_SCORES.items():
            table[(f't{i}', f't{j}')] = score
            table[(f't{j}', f't{i}')] = score
        return table

    def _measure(self, arm, *, max_spend_usd=1.0, clock=None, memories=None):
        return _mod().measure_detector_arm(
            arm, memories or self._memories(), self._CANDIDATES, self._classes(),
            context=_context(), max_spend_usd=max_spend_usd,
            clock=clock or iter([10.0, 12.5]).__next__,
            proposal=types.SimpleNamespace(precision_floor=self._FLOOR),
        )

    def _scorer(self, **kwargs):
        facts = _arms().ScorerFacts(device='cpu', vram_peak_mib=None, max_length=512)
        return _TableScorer(self._table(), facts=facts, **kwargs)

    def test_auc_covers_every_labelled_pair_even_ungenerated_ones(self):
        scorer = self._scorer()
        row = self._measure(_fake_arm(_yielding(scorer)))

        assert row.status == 'measured'
        assert row.auc.value == pytest.approx(5 / 6)
        assert (row.auc.n_true, row.auc.n_hard_negative) == (3, 2)
        assert row.auc.hanley_mcneil_95 == _mod().auc_interval_95(row.auc.value, 3, 2)
        assert ('t4', ('t1', 't5')) in scorer.calls

    def test_the_operating_point_is_chosen_over_candidates_only(self):
        row = self._measure(_fake_arm(_yielding(self._scorer())))

        assert row.operating_rule == 'max_caught_at_precision_floor'
        assert row.operating_threshold == pytest.approx(0.7)
        assert (row.paraphrase_band_recall.hits, row.paraphrase_band_recall.total) == (2, 3)
        assert row.paraphrase_band_recall.rate == pytest.approx(2 / 3)
        assert (row.precision.hits, row.precision.total) == (2, 3)
        assert (row.hard_negative_flagged.hits, row.hard_negative_flagged.total) == (1, 2)

    def test_recall_at_each_precision_floor_is_reported(self):
        row = self._measure(_fake_arm(_yielding(self._scorer())))

        assert set(row.recall_at_precision) == {0.80, 0.90, 0.95}
        for point in row.recall_at_precision.values():
            assert (point.threshold, point.caught, point.flagged) == (pytest.approx(0.9), 1, 1)

    def test_cost_truncation_device_and_time_are_recorded(self):
        scorer = self._scorer(costs=[0.01] * 6, over=[0, 1, 0, 0, 2, 0])
        row = self._measure(_fake_arm(_yielding(scorer)))

        assert row.pairs_scored == 12
        assert row.unscored_pairs == 0
        assert row.cost_usd == pytest.approx(0.06)
        assert row.pairs_over_max_length == 3
        assert (row.device, row.vram_peak_mib, row.max_length) == ('cpu', None, 512)
        assert row.scoring_seconds == pytest.approx(2.5)

    def test_an_unmeasured_truncation_is_none(self):
        scorer = self._scorer(over=[0, None, 0, 0, 0, 0])

        assert self._measure(_fake_arm(_yielding(scorer))).pairs_over_max_length is None

    def test_an_empty_endpoint_is_counted_unscored(self):
        memories = self._memories()
        memories[5]['content'] = '  '

        row = self._measure(_fake_arm(_yielding(self._scorer())), memories=memories)

        assert row.unscored_pairs == 1
        assert row.auc.n_true == 2

    def _assert_skipped(self, row, reason, detail):
        assert row.status == 'skipped'
        assert row.skip_reason == reason
        assert row.skip_detail == detail
        assert row.arm == 'fake-arm'
        for metric in (
            row.auc, row.operating_threshold, row.paraphrase_band_recall, row.precision,
            row.hard_negative_flagged, row.recall_at_precision, row.pairs_scored,
            row.cost_usd, row.pairs_over_max_length, row.scoring_seconds,
        ):
            assert metric is None

    def test_an_unavailable_arm_is_a_skipped_row(self):
        def open_(context):
            raise _arms().ArmUnavailable(_arms().SkipReason.no_credential, 'KEY unset')

        row = self._measure(_fake_arm(open_))

        self._assert_skipped(row, 'no_credential', 'KEY unset')

    def test_crossing_the_spend_ceiling_is_a_skipped_row(self):
        row = self._measure(
            _fake_arm(_yielding(self._scorer(costs=[0.01] * 6))), max_spend_usd=0.015,
        )

        assert row.status == 'skipped'
        assert row.skip_reason == 'over_budget'
        assert row.auc is None

    def test_any_other_failure_is_a_skipped_error_row(self):
        class Boom:
            def score(self, entry, candidate_texts):
                raise RuntimeError('boom')

            def facts(self):
                return None

        row = self._measure(_fake_arm(_yielding(Boom())))

        self._assert_skipped(row, 'error', 'RuntimeError: boom')

    def test_the_ann_baseline_scores_the_existing_detector(self):
        records = [
            _record('a', 'C1', 'canonical', _PK),
            _record('b', 'C1', 'duplicate', _PK),
            _record('c', 'C2', 'canonical', _PK),
            _record('d', 'C2', 'distinct', _PK),
        ]
        memories = _scan(records, {
            'a': [1.0, 0.0], 'b': [0.8, 0.6], 'c': [0.6, 0.8], 'd': [0.0, 1.0],
        })
        classes = _mod().labelled_pair_classes(records)
        neighbors = _mod().fixture_neighbors(memories, top_k=1)

        row = _mod().measure_ann_baseline(memories, neighbors, {_PK: 0.79}, classes)

        # Hits at >= 0.79: a-b (0.8), b-c (0.96), c-d (0.8 from d's list).
        # True (a,b); hard negative (c,d); unrelated the rest.
        assert row.operating_rule == 'calibrated_ann_cutoffs'
        assert row.operating_threshold is None
        assert (row.paraphrase_band_recall.hits, row.paraphrase_band_recall.total) == (1, 1)
        assert (row.precision.hits, row.precision.total) == (1, 3)
        assert (row.hard_negative_flagged.hits, row.hard_negative_flagged.total) == (1, 1)
        assert row.auc.value == 0.5
        assert (row.auc.n_true, row.auc.n_hard_negative) == (1, 1)


def _row(name='arm-a', *, hits=3, total=4, flagged=3, auc: float | None = 0.85, auc_lo=0.81,
         recall_lo=0.76, precision_lo=0.91, threshold: float | None = 0.5):
    m = _mod()
    return m.DetectorRow(
        arm=name, arm_class='local_cross_encoder', model='fake/model', granularity='full_text',
        status='measured', operating_rule=m.OPERATING_RULE_FLOOR,
        auc=None if auc is None else m.AucBlock(
            value=auc, n_true=total, n_hard_negative=2, hanley_mcneil_95=(auc_lo, 1.0),
        ),
        operating_threshold=threshold,
        paraphrase_band_recall=m.Proportion(
            hits=hits, total=total, rate=hits / total, wilson_95=(recall_lo, 0.99),
        ),
        precision=None if threshold is None else m.Proportion(
            hits=hits, total=flagged, rate=hits / flagged, wilson_95=(precision_lo, 1.0),
        ),
        hard_negative_flagged=m.Proportion(hits=1, total=2, rate=0.5, wilson_95=(0.09, 0.91)),
    )


def _skipped(name='arm-skipped', reason='no_credential', detail='KEY unset'):
    return _mod().DetectorRow.skipped(_fake_arm(_yielding(None), name=name), reason, detail)


def _baseline(hits=10):
    row = _row(_mod().BASELINE_ARM, hits=hits, total=10, flagged=10)
    return dataclasses.replace(
        row, arm_class='baseline', operating_rule=_mod().OPERATING_RULE_BASELINE,
        operating_threshold=None,
    )


def _generation():
    m = _mod()
    return m.CandidateGeneration(
        top_k=3, candidates=((0, 1), (0, 2)),
        true_caught=m.Proportion(hits=1, total=4, rate=0.25, wilson_95=(0.05, 0.7)),
        hard_negative_generated=1, cross_category_true_pairs=37,
        disclosure={'top_k_saturated': 2, 'missing_vector': 0},
    )


_CLASSES = {
    (0, 1): 'true_dup_pairs', (0, 2): 'hard_negative_pairs',
    (1, 2): 'hard_negative_pairs', (0, 3): 'unrelated_pairs',
}
_PROVENANCE = {'measured_at': '2099-01-02', 'record_count': 17}


def _report(arm_rows):
    return _mod().build_report(
        generation=_generation(), pair_classes=_CLASSES, baseline=_baseline(),
        arm_rows=arm_rows, records=17,
        embedder={'model': 'fake-embedder', 'dimensions': 2}, provenance=_PROVENANCE,
    )


class TestReportAndProposal:
    def test_the_gate_proposal_is_the_pre_registered_one(self):
        proposal = _mod().GATE_PROPOSAL

        assert proposal.precision_floor == 0.90
        assert proposal.min_paraphrase_band_recall == 0.75
        assert proposal.min_auc == 0.80
        assert proposal.proposed_on == '2026-11-16'
        assert proposal.basis and all(proposal.basis)

    def _verdict(self, row):
        return _mod().gate_verdict(row.to_json(), _mod().GATE_PROPOSAL)

    def test_a_row_exactly_at_every_bar_passes_both_readings(self):
        row = _row(hits=3, total=4, auc=0.80, recall_lo=0.75, precision_lo=0.90, auc_lo=0.80)

        assert self._verdict(row) == {'on_point_estimates': True, 'on_lower_bounds': True}

    @pytest.mark.parametrize('changes', [
        {'hits': 74, 'total': 100}, {'auc': 0.79},
    ])
    def test_a_point_estimate_below_its_bar_fails_that_reading(self, changes):
        row = _row(**{'recall_lo': 0.75, 'precision_lo': 0.9, 'auc_lo': 0.8, **changes})

        assert self._verdict(row)['on_point_estimates'] is False

    @pytest.mark.parametrize('changes', [
        {'recall_lo': 0.749}, {'precision_lo': 0.899}, {'auc_lo': 0.799},
    ])
    def test_a_lower_bound_below_its_bar_fails_that_reading(self, changes):
        row = _row(**{'hits': 3, 'total': 4, 'auc': 0.9, **changes})

        assert self._verdict(row) == {'on_point_estimates': True, 'on_lower_bounds': False}

    def test_no_operating_point_fails_both_readings(self):
        row = _row(threshold=None)

        assert self._verdict(row) == {'on_point_estimates': False, 'on_lower_bounds': False}

    def _chosen(self, rows):
        chosen = _mod().choose_detector([row.to_json() for row in rows])
        return None if chosen is None else chosen['arm']

    def test_the_most_true_pairs_caught_wins_and_the_baseline_never_does(self):
        rows = [
            _baseline(hits=10), _skipped(),
            _row('a', hits=2), _row('b', hits=3, auc=0.7), _row('c', hits=1, auc=0.99),
        ]

        assert self._chosen(rows) == 'b'

    def test_a_tie_on_caught_goes_to_the_higher_auc_then_the_earlier_row(self):
        assert self._chosen([_row('a', auc=None), _row('b', auc=0.5)]) == 'b'
        assert self._chosen([_row('a', auc=0.6), _row('b', auc=0.6)]) == 'a'

    def test_nothing_measured_chooses_nothing(self):
        assert self._chosen([_baseline(), _skipped()]) is None

    def test_the_recall_caveat_entry_validates_against_the_config_schema(self):
        report = _report([_skipped(), _row('chosen-arm', hits=3, total=4)])

        entry = _mod().recall_caveat_entry(report)

        model = _report_schema().AdjudicatorMeasurement.model_validate(entry)
        assert model.measured_at == '2099-01-02'
        assert model.detector == 'chosen-arm'
        assert model.source == (
            'fused-memory/calibration/duplicate_detector_recall_report.json (task 10243)'
        )
        assert (model.paraphrase_band_recall.caught, model.paraphrase_band_recall.of) == (3, 4)
        assert (model.candidate_generation.caught, model.candidate_generation.of) == (1, 4)
        assert list(entry) == list(_report_schema().AdjudicatorMeasurement.model_fields)

    def test_no_chosen_arm_is_no_entry(self):
        assert _mod().recall_caveat_entry(_report([_skipped()])) is None

    def test_the_report_carries_setting_counts_rows_and_derived_caveats(self):
        report = _report([_skipped('arm-x', 'dependency_unavailable', 'no torch'), _row('a')])

        assert list(report) == [
            'setting', 'pair_counts', 'candidate_generation', 'baseline', 'arms', 'chosen',
            'gate_proposal', 'recall_caveat_entry', 'caveats', 'provenance',
        ]
        assert report['setting'] == {
            'scan_set': 'fixture_only', 'records': 17, 'top_k': 3, 'per_category': True,
            'embedder': {'model': 'fake-embedder', 'dimensions': 2},
        }
        assert report['pair_counts'] == {
            'true_dup_pairs': 1, 'hard_negative_pairs': 2, 'unrelated_pairs': 1,
        }
        assert report['chosen']['arm'] == 'a'
        assert set(report['chosen']['verdict']) == {'on_point_estimates', 'on_lower_bounds'}
        assert report['gate_proposal'] == _mod().gate_proposal_json(_mod().GATE_PROPOSAL)
        assert report['provenance'] == _PROVENANCE
        assert any(
            caveat.startswith('37 of the 1 true pairs are cross-category')
            for caveat in report['caveats']
        )
        caveats = ' '.join(report['caveats'])
        assert 'arm-x' in caveats and 'dependency_unavailable' in caveats
        assert 'in-sample' in caveats
        assert 'fixture_only' in caveats
        assert 'jev-choice' in caveats
        json.dumps(report)

    def test_the_markdown_is_a_function_of_the_json_alone(self):
        report = _report([_skipped(), _row('chosen-arm-x')])
        proposal = _mod().GATE_PROPOSAL

        markdown = _mod().render_markdown(report)

        assert markdown == _mod().render_markdown(json.loads(json.dumps(report)))
        assert '**Chosen:** chosen-arm-x' in markdown
        for bar in (proposal.precision_floor, proposal.min_paraphrase_band_recall, proposal.min_auc):
            assert f'{bar:.2f}' in markdown

    def test_write_report_writes_the_json_and_its_markdown_sibling(self, tmp_path):
        report = _report([_row('a')])
        path = tmp_path / 'out.json'

        _mod().write_report(report, path)

        assert json.loads(path.read_text()) == json.loads(json.dumps(report))
        assert path.with_suffix('.md').read_text() == _mod().render_markdown(report)

    def test_a_non_json_report_path_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match='.json'):
            _mod().guard_report_path(tmp_path / 'out.md', arm_filter=None)

    def test_a_filtered_run_is_refused_at_the_committed_path_however_spelled(self, monkeypatch):
        monkeypatch.chdir(SCRIPTS.parent)
        relative = 'calibration/duplicate_detector_recall_report.json'

        with pytest.raises(ValueError, match='committed'):
            _mod().guard_report_path(relative, arm_filter=['bge-reranker-v2-m3'])
        with pytest.raises(ValueError, match='committed'):
            _mod().guard_report_path(
                SCRIPTS / '..' / relative, arm_filter=['bge-reranker-v2-m3'],
            )
        assert _mod().guard_report_path(relative, arm_filter=None).name == (
            'duplicate_detector_recall_report.json'
        )

    def test_a_filtered_run_elsewhere_passes(self, tmp_path):
        path = tmp_path / 'smoke.json'

        assert _mod().guard_report_path(path, arm_filter=['bge-reranker-v2-m3']) == path


class _ConstantScorer:
    def score(self, entry, candidate_texts):
        return _arms().SlateScores(
            scores=(0.5,) * len(candidate_texts), cost_usd=0.0, pairs_over_max_length=0,
        )

    def facts(self):
        return _arms().ScorerFacts(device='cpu', vram_peak_mib=None, max_length=512)


class TestCli:
    _VECTORS = {'a': [1.0, 0.0], 'b': [0.8, 0.6], 'c': [0.6, 0.8], 'd': [0.0, 1.0]}

    def _fixture(self, tmp_path):
        records = [
            _record('a', 'C1', 'canonical', _PK), _record('b', 'C1', 'duplicate', _PK),
            _record('c', 'C2', 'canonical', _PK), _record('d', 'C2', 'distinct', _PK),
        ]
        path = tmp_path / 'fixture.jsonl'
        path.write_text(''.join(json.dumps(record) + '\n' for record in records))
        return path

    def _config(self):
        return types.SimpleNamespace(
            write_triage=types.SimpleNamespace(t_high=0.5, t_high_by_category={_PK: 0.79}),
            embedder=types.SimpleNamespace(model='fake-embedder'),
        )

    def _arms_under_test(self):
        def unavailable(context):
            raise _arms().ArmUnavailable(_arms().SkipReason.no_credential, 'KEY unset')

        return [_fake_arm(_yielding(_ConstantScorer()), name='arm-ok'),
                _fake_arm(unavailable, name='arm-unavailable')]

    def _run(self, tmp_path, argv, embed_calls):
        args = _mod().build_parser().parse_args(argv)

        def embed_factory(config):
            embed_calls.append(config)
            return lambda memory_id, content: self._VECTORS[memory_id]

        return _mod().run_measurement(
            args, embed_factory=embed_factory, arms=self._arms_under_test(),
            config_factory=self._config, today=datetime.date(2099, 1, 2),
        )

    def test_a_run_writes_the_report_with_its_provenance(self, tmp_path):
        fixture = self._fixture(tmp_path)
        report_path = tmp_path / 'out' / 'report.json'
        calls: list = []

        rc = self._run(
            tmp_path, ['--fixture', str(fixture), '--report-path', str(report_path)], calls,
        )

        assert rc == 0
        assert len(calls) == 1
        report = json.loads(report_path.read_text())
        assert report_path.with_suffix('.md').exists()
        provenance = report['provenance']
        assert provenance['fixture_path'] == _calibrate().package_relative(fixture)
        assert provenance['record_count'] == 4
        assert provenance['top_k'] == _audit_mod()._DEFAULT_ANN_TOP_K
        assert provenance['ann_cutoffs'] == {_PK: 0.79}
        assert provenance['project_id'] is None
        assert provenance['measured_at'] == '2099-01-02'
        assert provenance['arm_filter'] is None
        assert provenance['device'] == 'auto'
        assert provenance['local_batch_size'] == 4
        assert provenance['vram_cap_gib'] == 8.0
        assert provenance['pairwise_concurrency'] == 20
        assert provenance['max_arm_spend_usd'] == 2.0
        assert isinstance(provenance['libraries'], dict)
        assert report['setting']['embedder'] == {'model': 'fake-embedder', 'dimensions': 2}
        rows = {row['arm']: row for row in report['arms']}
        assert rows['arm-unavailable']['status'] == 'skipped'
        assert rows['arm-unavailable']['skip_reason'] == 'no_credential'
        assert rows['arm-ok']['status'] == 'measured'

    def test_an_unknown_arm_is_rejected_by_the_parser(self):
        with pytest.raises(SystemExit):
            _mod().build_parser().parse_args(['--arm', 'no-such-arm'])

    def test_a_filtered_run_at_the_committed_path_is_refused_before_embedding(self, tmp_path):
        fixture = self._fixture(tmp_path)
        calls: list = []
        arm = _mod().DETECTOR_ARMS[0].name

        with pytest.raises(ValueError, match='committed'):
            self._run(tmp_path, ['--fixture', str(fixture), '--arm', arm], calls)
        assert calls == []

    def test_the_parser_is_loud(self):
        from shared.cli_boundary import LoudArgumentParser

        assert isinstance(_mod().build_parser(), LoudArgumentParser)


_CALIBRATION = SCRIPTS.parent / 'calibration'
_COMMITTED_REPORT = _CALIBRATION / 'duplicate_detector_recall_report.json'
_COMMITTED_FIXTURE = SCRIPTS.parent / 'tests' / 'fixtures' / 'write_triage_calibration.jsonl'


@functools.cache
def _committed_report() -> dict:
    return json.loads(_COMMITTED_REPORT.read_text())


class TestTheCommittedReport:
    """STRUCTURE only: the measured values inform task 10244's ruling and gate no build."""

    def test_it_measured_the_whole_committed_fixture_with_every_arm(self):
        report = _committed_report()
        records = _calibrate().load_fixture(_COMMITTED_FIXTURE)

        assert report['provenance']['record_count'] == len(records)
        assert report['provenance']['fixture_path'] == 'tests/fixtures/write_triage_calibration.jsonl'
        assert report['provenance']['arm_filter'] is None
        assert report['setting']['scan_set'] == 'fixture_only'
        assert report['setting']['records'] == len(records)

    def test_its_pair_counts_are_the_fixtures(self):
        records = _calibrate().load_fixture(_COMMITTED_FIXTURE)
        expected = {
            name: len(pairs) for name, pairs in _calibrate().build_pair_sets(records).items()
        }

        assert _committed_report()['pair_counts'] == expected

    def test_every_detector_arm_has_exactly_one_row_beside_the_baseline(self):
        report = _committed_report()
        names = [row['arm'] for row in report['arms']]

        assert sorted(names) == sorted(arm.name for arm in _mod().DETECTOR_ARMS)
        assert len(names) == len(set(names))
        assert report['baseline']['arm'] == _mod().BASELINE_ARM

    def test_it_carries_the_pre_registered_proposal(self):
        assert _committed_report()['gate_proposal'] == _mod().gate_proposal_json(
            _mod().GATE_PROPOSAL,
        )

    def test_its_recall_caveat_entry_is_absent_or_schema_valid(self):
        entry = _committed_report()['recall_caveat_entry']

        if entry is not None:
            _report_schema().AdjudicatorMeasurement.model_validate(entry)

    def test_the_markdown_re_renders_from_the_json(self):
        markdown = _COMMITTED_REPORT.with_suffix('.md').read_text()

        assert _mod().render_markdown(_committed_report()) == markdown


_COMMITTED_AUDIT_CONFIG = SCRIPTS.parent / 'config' / 'duplicate-audit.yaml'


class TestTheRecallCaveatCarriesTheCommittedMeasurement:
    """The yaml is the single home the 10244 resolver reads; the report is its derivation."""

    def _caveat(self):
        return _report_schema().load_config(_COMMITTED_AUDIT_CONFIG).recall_caveat

    def test_its_single_adjudicator_entry_is_the_reports_entry_verbatim(self):
        reported = _committed_report()['recall_caveat_entry']

        (entry,) = self._caveat().adjudicator_measurements

        assert entry.source == reported['source']
        assert entry.model_dump(mode='json') == reported

    def test_gate_filing_stays_blocked_and_owned_by_4916(self):
        caveat = self._caveat()

        assert caveat.fitness.gate_filing == 'blocked'
        assert caveat.gate_filing_owner_task == 4916

    def test_the_2026_08_11_measurement_is_unchanged(self):
        caveat = self._caveat()

        assert caveat.measured_at == '2026-08-11'
        assert caveat.paraphrase_band_recall['ann'].model_dump() == {'caught': 25, 'of': 301}
        assert caveat.paraphrase_band_recall['lexical'].model_dump() == {'caught': 0, 'of': 301}
        assert caveat.auc_true_vs_hard_negative == 0.2534
