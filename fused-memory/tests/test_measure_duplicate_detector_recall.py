"""Tests for measure_duplicate_detector_recall.py (task 10243).

The script replays the 2026-08-11 ANN calibration protocol on the task 3130
fixture and measures adjudicators over its ANN candidates. Every arm here is a
fake and every vector is hand-made, so each expected number is hand-computable
from the literals in the test. Nothing touches the network, torch or a store.
"""
from __future__ import annotations

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
