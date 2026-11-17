"""Tests for the ``kind_registry_drift`` block of scripts/census_memory_metadata.py.

The block scores the live ``kind`` population against the frozen
``memory_metadata.KIND_REGISTRY`` baseline (PRD §10 open question 1, ruled
option (b): ``kind`` only warns). Every input here is SYNTHETIC: a live
number would move nightly, so a test asserting one could never stay green.
Assertions against the real registry are relational only.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from _fm_helpers import load_script_module
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_mod = load_script_module(
    Path(__file__).parent.parent / 'scripts' / 'census_memory_metadata.py',
    mod_name='census_memory_metadata',
)


class TestBuildKindRegistryDrift:
    REGISTRY = frozenset({'alpha', 'beta', 'gone'})
    ENTRIES = [
        {'value': 'alpha', 'count': 7},
        {'value': 'yolo', 'count': 3},
        {'value': 'zeta', 'count': 3},
        {'value': 'beta', 'count': 1},
    ]

    def _drift(self, entries=None, registry=None):
        return _mod.build_kind_registry_drift(
            self.ENTRIES if entries is None else entries,
            self.REGISTRY if registry is None else registry,
        )

    def test_scalars_partition_and_reconcile(self):
        drift = self._drift()
        assert drift['registry_total'] == 3
        assert drift['measured_distinct'] == 4
        assert drift['measured_records'] == 14
        assert drift['registered_distinct'] == 2
        assert drift['unregistered_distinct'] == 2
        assert drift['unregistered_records'] == 6
        assert drift['absent_from_corpus_distinct'] == 1
        assert (
            drift['registered_distinct'] + drift['unregistered_distinct']
            == drift['measured_distinct']
        )

    def test_unregistered_rows_keep_the_input_order(self):
        assert self._drift()['unregistered'] == [
            {'value': 'yolo', 'count': 3},
            {'value': 'zeta', 'count': 3},
        ]

    def test_absent_from_corpus_is_a_sorted_list_of_plain_values(self):
        drift = self._drift(registry=frozenset({'gone', 'alpha', 'also_gone'}))
        assert drift['absent_from_corpus'] == ['also_gone', 'gone']

    def test_a_non_str_value_is_unregistered_and_never_raises(self):
        drift = self._drift(
            entries=[{'value': 'alpha', 'count': 1}, {'value': 42, 'count': 2}],
            registry=frozenset({'alpha', '42'}),
        )
        assert drift['unregistered'] == [{'value': 42, 'count': 2}]
        assert drift['registered_distinct'] == 1
        assert drift['absent_from_corpus'] == ['42']

    @pytest.mark.parametrize('entries', [[], None])
    def test_empty_input_is_a_measured_state_not_a_missing_one(self, entries):
        drift = _mod.build_kind_registry_drift(entries, self.REGISTRY)
        assert drift['measured_distinct'] == 0
        assert drift['measured_records'] == 0
        assert drift['registered_distinct'] == 0
        assert drift['unregistered_distinct'] == 0
        assert drift['unregistered_records'] == 0
        assert drift['unregistered'] == []
        assert drift['absent_from_corpus_distinct'] == drift['registry_total'] == 3
        assert drift['absent_from_corpus'] == sorted(self.REGISTRY)

    def test_an_entry_with_no_count_reads_as_zero(self):
        drift = self._drift(entries=[{'value': 'yolo'}])
        assert drift['measured_records'] == 0
        assert drift['unregistered'] == [{'value': 'yolo', 'count': 0}]

    def test_pure_deterministic_and_serialisable(self):
        entries = copy.deepcopy(self.ENTRIES)
        registry = frozenset(self.REGISTRY)
        first = _mod.build_kind_registry_drift(entries, registry)
        second = _mod.build_kind_registry_drift(entries, registry)
        assert first == second
        json.dumps(first)
        assert entries == self.ENTRIES
        assert registry == self.REGISTRY
