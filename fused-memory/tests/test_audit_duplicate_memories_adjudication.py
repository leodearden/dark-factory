"""The adjudicated-ANN detector in audit_duplicate_memories.py (task 10243).

ANN generates candidate pairs with no similarity cutoff; an injected slate
scorer adjudicates each pair symmetrically; an inclusive threshold keeps the
pairs that feed the shared transitive closure. Every input is hand-built, so
no torch, network or store is involved.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest
from _fm_helpers import load_script_module

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'audit_duplicate_memories.py'
_mod = load_script_module(SCRIPT_PATH, mod_name='audit_duplicate_memories')

_VEC = [1.0, 0.0]


def _memory(memory_id: str, *, vector: list[float] | None = _VEC, content: str = '') -> dict:
    return {
        'id': memory_id, 'category': 'procedural_knowledge',
        'vector': vector, 'content': content,
    }


class TestAnnCandidatePairs:
    def _corpus(self) -> tuple[list[dict], dict[str, list[dict]]]:
        memories = [_memory('a'), _memory('b'), _memory('c'), _memory('d', vector=None)]
        neighbors = {
            'a': [
                {'id': 'a', 'score': 1.0},
                {'id': 'b', 'score': 0.95},
                {'id': 'c', 'score': 0.10},
                {'id': 'zz', 'score': 0.5},
            ],
            'b': [{'id': 'a', 'score': -0.30}, {'id': 'c', 'score': None}],
            'c': [{'id': 'd', 'score': 0.2}],
            'd': [{'id': 'a', 'score': 0.99}],
        }
        return memories, neighbors

    def test_every_hit_becomes_a_pair_with_no_cutoff(self):
        memories, neighbors = self._corpus()
        pairs, _ = _mod.ann_candidate_pairs(memories, neighbors, top_k=4)
        assert pairs == [(0, 1), (0, 2), (1, 2), (2, 3)]

    def test_a_pair_named_from_both_endpoints_appears_once(self):
        memories = [_memory('a'), _memory('b')]
        neighbors = {
            'a': [{'id': 'b', 'score': 0.4}],
            'b': [{'id': 'a', 'score': 0.4}],
        }
        pairs, _ = _mod.ann_candidate_pairs(memories, neighbors, top_k=10)
        assert pairs == [(0, 1)]

    def test_disclosure_counts_every_loss_and_admits_every_score(self):
        memories, neighbors = self._corpus()
        _, disclosure = _mod.ann_candidate_pairs(memories, neighbors, top_k=4)
        _, reference = _mod.ann_pairs_from_neighbors(memories, {}, 0.5)
        assert set(disclosure) == set(reference)
        assert disclosure['below_threshold_dropped'] == 0
        assert disclosure['unknown_neighbor_dropped'] == 1
        assert disclosure['missing_vector'] == 1
        assert disclosure['top_k_saturated'] == 1
        assert disclosure['uncalibrated_category_dropped'] == 0

    def test_a_vectorless_record_contributes_no_pairs_of_its_own(self):
        memories = [_memory('a', vector=None), _memory('b')]
        neighbors = {'a': [{'id': 'b', 'score': 0.9}]}
        pairs, disclosure = _mod.ann_candidate_pairs(memories, neighbors, top_k=10)
        assert pairs == []
        assert disclosure['missing_vector'] == 1

    def test_memories_are_not_mutated(self):
        memories, neighbors = self._corpus()
        before = copy.deepcopy(memories)
        _mod.ann_candidate_pairs(memories, neighbors, top_k=4)
        assert memories == before


class _RecordingScorer:
    """A directional slate scorer reading ``table[(entry, candidate)]``."""

    def __init__(self, table: dict[tuple[str, str], float], *, drop_one: bool = False):
        self.table = table
        self.drop_one = drop_one
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def __call__(self, entry: str, candidates):
        self.calls.append((entry, tuple(candidates)))
        scores = [self.table[(entry, candidate)] for candidate in candidates]
        return scores[:-1] if self.drop_one else scores


class TestAdjudicatePairs:
    _TABLE = {
        ('Alpha one', 'Beta two'): 0.8, ('Beta two', 'Alpha one'): 0.4,
        ('Alpha one', 'Gamma three'): 0.2, ('Gamma three', 'Alpha one'): 0.6,
        ('Beta two', 'Gamma three'): 1.0, ('Gamma three', 'Beta two'): 0.0,
    }

    def _corpus(self) -> list[dict]:
        return [
            _memory('alpha', content='  Alpha one  '),
            _memory('beta', content='Beta two'),
            _memory('gamma', content='Gamma three\n'),
            _memory('blank', content='   '),
            {'id': 'none', 'category': 'procedural_knowledge', 'vector': _VEC, 'content': None},
        ]

    _PAIRS = [(0, 1), (0, 2), (1, 2), (0, 3), (2, 4), (0, 1)]

    def test_each_pair_scores_the_mean_of_its_two_orientations(self):
        scorer = _RecordingScorer(self._TABLE)
        scores = _mod.adjudicate_pairs(self._corpus(), self._PAIRS, scorer)
        assert scores == {
            (0, 1): pytest.approx((0.8 + 0.4) / 2),
            (0, 2): pytest.approx((0.2 + 0.6) / 2),
            (1, 2): pytest.approx((1.0 + 0.0) / 2),
        }

    def test_the_scorer_is_called_once_per_record_with_partners_in_index_order(self):
        scorer = _RecordingScorer(self._TABLE)
        _mod.adjudicate_pairs(self._corpus(), self._PAIRS, scorer)
        assert scorer.calls == [
            ('Alpha one', ('Beta two', 'Gamma three')),
            ('Beta two', ('Alpha one', 'Gamma three')),
            ('Gamma three', ('Alpha one', 'Beta two')),
        ]

    def test_empty_content_never_reaches_the_scorer_and_never_pairs(self):
        scorer = _RecordingScorer(self._TABLE)
        scores = _mod.adjudicate_pairs(self._corpus(), self._PAIRS, scorer)
        assert (0, 3) not in scores
        assert (2, 4) not in scores
        sent = [entry for entry, _ in scorer.calls]
        sent += [text for _, candidates in scorer.calls for text in candidates]
        assert all(text.strip() for text in sent)

    def test_a_wrong_score_count_names_the_record_and_both_counts(self):
        scorer = _RecordingScorer(self._TABLE, drop_one=True)
        with pytest.raises(ValueError) as excinfo:
            _mod.adjudicate_pairs(self._corpus(), self._PAIRS, scorer)
        message = str(excinfo.value)
        assert 'alpha' in message
        assert re.search(r'\b2\b', message)
        assert re.search(r'\b1\b', message)

    def test_duplicate_input_pairs_are_scored_once_under_their_canonical_key(self):
        scorer = _RecordingScorer(self._TABLE)
        scores = _mod.adjudicate_pairs(self._corpus(), [(0, 1), (0, 1)], scorer)
        assert list(scores) == [(0, 1)]
        assert scorer.calls == [
            ('Alpha one', ('Beta two',)),
            ('Beta two', ('Alpha one',)),
        ]

    def test_adjudicated_pairs_keeps_scores_at_or_above_the_threshold_sorted(self):
        scores = {(1, 2): 0.5, (0, 2): 0.4, (0, 1): 0.6}
        assert _mod.adjudicated_pairs(scores, 0.5) == [(0, 1), (1, 2)]
        assert _mod.adjudicated_pairs(scores, 0.61) == []

    def test_the_detector_end_to_end_feeds_the_shared_closure(self):
        memories = [
            _memory(memory_id, content=text)
            for memory_id, text in (
                ('m1', 'one'), ('m2', 'two'), ('m3', 'three'),
                ('m4', 'four'), ('m5', 'five'),
            )
        ]
        symmetric = {
            ('one', 'two'): 0.9, ('two', 'three'): 0.7,
            ('four', 'five'): 0.8, ('one', 'four'): 0.3,
        }
        table = {**symmetric, **{(b, a): s for (a, b), s in symmetric.items()}}
        candidates = [(0, 1), (1, 2), (3, 4), (0, 3)]
        scores = _mod.adjudicate_pairs(memories, candidates, _RecordingScorer(table))
        groups = _mod.cluster_memories_by_pairs(
            memories, _mod.adjudicated_pairs(scores, 0.5),
        )
        assert [[m['id'] for m in group] for group in groups] == [
            ['m1', 'm2', 'm3'], ['m4', 'm5'],
        ]
