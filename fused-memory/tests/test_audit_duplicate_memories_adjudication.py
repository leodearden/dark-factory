"""The adjudicated-ANN detector in audit_duplicate_memories.py (task 10243).

ANN generates candidate pairs with no similarity cutoff; an injected slate
scorer adjudicates each pair symmetrically; an inclusive threshold keeps the
pairs that feed the shared transitive closure. Every input is hand-built, so
no torch, network or store is involved.
"""
from __future__ import annotations

import copy
from pathlib import Path

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
