"""Tests for measure_duplicate_detector_recall.py (task 10243).

The script replays the 2026-08-11 ANN calibration protocol on the task 3130
fixture and measures adjudicators over its ANN candidates. Every arm here is a
fake and every vector is hand-made, so each expected number is hand-computable
from the literals in the test. Nothing touches the network, torch or a store.
"""
from __future__ import annotations

import contextlib
import functools
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
