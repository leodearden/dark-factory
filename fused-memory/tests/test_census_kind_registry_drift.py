"""Tests for the ``kind_registry_drift`` block of scripts/census_memory_metadata.py.

The block scores the live ``kind`` population against the frozen
``memory_metadata.KIND_REGISTRY`` baseline (PRD §10 open question 1, ruled
option (b): ``kind`` membership only warns). Every input here is SYNTHETIC: a live
number would move nightly, so a test asserting one could never stay green.
Assertions against the real registry are relational only.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from _fm_helpers import load_script_module

from fused_memory import memory_metadata
from fused_memory.memory_metadata import KIND_REGISTRY
from fused_memory.models.enums import MemoryCategory

_mod = load_script_module(
    Path(__file__).parent.parent / 'scripts' / 'census_memory_metadata.py',
    mod_name='census_memory_metadata',
)

OBS = MemoryCategory.observations_and_summaries.value


def _cell(payloads):
    census = _mod.CategoryCensus()
    for payload in payloads:
        census.add(payload)
    return census


def _cov(collection, n):
    return {
        'collection': collection,
        'collection_points': n,
        'categories': {OBS: {'expected': n, 'scrolled': n, 'recount': n}},
    }


def _report(payloads_by_project, top_n=50):
    cells = {project: {OBS: _cell(p)} for project, p in payloads_by_project.items()}
    cov = {
        project: _cov(f'fused_{project}', len(p))
        for project, p in payloads_by_project.items()
    }
    return _mod.build_report(cells, cov, top_n=top_n)


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

    def test_empty_input_is_a_measured_state(self):
        drift = _mod.build_kind_registry_drift([], self.REGISTRY)
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


class TestKindRegistryDriftInReport:
    PAYLOADS = {
        'dark_factory': [{'kind': 'cycle_summary'}, {'kind': 'not_a_real_kind_xyz'}],
        'reify': [{'kind': 'cycle_summary'}],
    }

    def _block(self):
        return _report(self.PAYLOADS)['kind_registry_drift']

    def test_the_block_is_top_level_beside_registry_coverage(self):
        report = _report(self.PAYLOADS)
        assert isinstance(report['kind_registry_drift'], dict)
        assert 'registry_coverage' in report
        assert 'kind_registry_drift' not in report['topic_coverage']
        for project in report['projects'].values():
            assert 'kind_registry_drift' not in project

    def test_schema_version_records_the_new_block(self):
        assert _report(self.PAYLOADS)['schema_version'] >= 6

    def test_measured_at_grand_total_grain(self):
        block = _report({
            'dark_factory': [{'kind': 'cycle_summary'}],
            'reify': [{'kind': 'cycle_summary'}],
        })['kind_registry_drift']
        assert block['measured_distinct'] == 1
        assert block['measured_records'] == 2

    def test_the_registry_is_the_imported_object_not_a_copy(self):
        assert _mod.KIND_REGISTRY is memory_metadata.KIND_REGISTRY

    def test_the_real_registry_is_the_default(self):
        block = self._block()
        unregistered = [row['value'] for row in block['unregistered']]
        assert block['registry_total'] == len(KIND_REGISTRY)
        assert 'cycle_summary' in KIND_REGISTRY
        assert 'cycle_summary' not in unregistered
        assert 'not_a_real_kind_xyz' in unregistered

    def test_deterministic_and_serialisable(self):
        report = _report(self.PAYLOADS)
        assert json.loads(json.dumps(report)) == report
        assert self._block() == self._block()


def _section(markdown):
    start = markdown.index('## Kind vocabulary drift')
    end = markdown.find('\n## ', start + 1)
    return markdown[start:] if end == -1 else markdown[start:end]


class TestRenderKindRegistryDrift:
    INVENTED = ['invented_kind_a', 'invented_kind_b', 'invented_kind_c']

    def _report(self, top_n=2):
        return _report(
            {'dark_factory': [{'kind': 'cycle_summary'}] + [{'kind': k} for k in self.INVENTED]},
            top_n=top_n,
        )

    def test_the_section_sits_between_coverage_trend_and_record_counts(self):
        markdown = _mod.render_markdown(self._report())
        lines = markdown.splitlines()
        assert '## Kind vocabulary drift' in lines
        heading = markdown.index('\n## Kind vocabulary drift\n')
        assert markdown.index('\n## Coverage trend\n') < heading
        assert heading < markdown.index('\n## Record counts\n')

    def test_the_headline_carries_the_four_counts(self):
        section = _section(_mod.render_markdown(self._report()))
        assert f'**{len(KIND_REGISTRY):,}**' in section
        assert '**4**' in section
        assert '**3**' in section
        assert f'**{len(KIND_REGISTRY) - 1:,}**' in section

    def test_the_unregistered_table_is_cut_to_top_n_and_says_so(self):
        section = _section(_mod.render_markdown(self._report(top_n=2)))
        assert '| value | count |' in section
        assert '`invented_kind_a`' in section
        assert '`invented_kind_b`' in section
        assert '`invented_kind_c`' not in section
        assert 'truncated' in section

    def test_no_drift_renders_explicit_none_lines(self):
        report = self._report()
        report['kind_registry_drift'] = _mod.build_kind_registry_drift(
            [{'value': v, 'count': 1} for v in sorted(KIND_REGISTRY)],
        )
        assert report['kind_registry_drift']['unregistered'] == []
        assert report['kind_registry_drift']['absent_from_corpus'] == []
        section = _section(_mod.render_markdown(report))
        assert section.count('_(none)_') == 2

    def test_a_pre_v6_report_discloses_the_missing_block(self):
        report = self._report()
        del report['kind_registry_drift']
        section = _section(_mod.render_markdown(report))
        assert 'pre-schema-6' in section
        assert 'no drift was measured' in section
        assert '| value |' not in section

    def test_the_section_states_the_axis_only_warns(self):
        assert 'does not reject' in _section(_mod.render_markdown(self._report()))

    def test_rendering_is_deterministic(self):
        report = self._report()
        assert _mod.render_markdown(report) == _mod.render_markdown(report)
