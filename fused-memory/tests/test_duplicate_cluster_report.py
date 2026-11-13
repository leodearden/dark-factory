"""The weekly duplicate-cluster report (task 3136).

``scripts/duplicate_cluster_report.py`` runs the detector once per (project,
category), builds a report from the detector's stdout plans and diffs it
against that project's previous report. These tests drive its public
functions and its CLI. Plans come from the REAL detector's
``build_sweep_plan`` over synthetic records, loaded by path, so the
detector-to-report contract is exercised end to end rather than restated in a
fixture that could drift from it.
"""
from __future__ import annotations

from pathlib import Path

import pydantic
import pytest
import yaml
from _fm_helpers import load_script_module

_FM_ROOT = Path(__file__).resolve().parent.parent
REPORT_SCRIPT = _FM_ROOT / 'scripts' / 'duplicate_cluster_report.py'
DETECTOR_SCRIPT = _FM_ROOT / 'scripts' / 'audit_duplicate_memories.py'
COMMITTED_CONFIG = _FM_ROOT / 'config' / 'duplicate-audit.yaml'

report = load_script_module(REPORT_SCRIPT, mod_name='duplicate_cluster_report')
detector = load_script_module(DETECTOR_SCRIPT, mod_name='audit_duplicate_memories')


def _committed_yaml() -> dict:
    return yaml.safe_load(COMMITTED_CONFIG.read_text(encoding='utf-8'))


def _write_yaml(tmp_path: Path, data: dict, name: str = 'duplicate-audit.yaml') -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    return path


class TestCommittedConfig:
    def test_the_committed_yaml_loads_into_a_frozen_config(self) -> None:
        config = report.load_config(COMMITTED_CONFIG)

        assert isinstance(config, report.DuplicateAuditConfig)
        with pytest.raises(pydantic.ValidationError):
            config.scan_limit = 1

    def test_it_sweeps_both_projects_over_the_detectors_own_categories(self) -> None:
        config = report.load_config(COMMITTED_CONFIG)

        assert config.projects == ('dark_factory', 'reify')
        assert config.categories == tuple(detector._ALL_CATEGORIES)
        assert config.scan_limit > 0
        assert config.detector_timeout_secs > 0

    def test_the_resweep_cadence_sits_beside_the_timer_cadence(self) -> None:
        cadence = report.load_config(COMMITTED_CONFIG).cadence

        assert cadence.on_calendar == 'Sun *-*-* 06:30:00'
        assert cadence.resweep_interval_days == 7

    def test_cycle_summaries_are_declared_machine_written(self) -> None:
        assert 'cycle_summary' in report.load_config(COMMITTED_CONFIG).machine_written_kinds

    def test_the_recall_caveat_states_which_bar_each_number_clears(self) -> None:
        caveat = report.load_config(COMMITTED_CONFIG).recall_caveat

        assert caveat.fitness.regrowth_measurement == 'supported'
        assert caveat.fitness.gate_filing == 'blocked'
        assert caveat.gate_filing_owner_task == 4916
        assert caveat.paraphrase_band_recall['ann'].caught == 25
        assert caveat.paraphrase_band_recall['ann'].of == 301
        assert caveat.paraphrase_band_recall['lexical'].caught == 0
        assert caveat.paraphrase_band_recall['lexical'].of == 301
        assert caveat.auc_true_vs_hard_negative == 0.2534
        assert caveat.measured_at
        assert caveat.source


def _without(data: dict, key: str) -> dict:
    return {k: v for k, v in data.items() if k != key}


def _with(data: dict, **changes: object) -> dict:
    return {**data, **changes}


def _with_cadence(data: dict, **changes: object) -> dict:
    return {**data, 'cadence': {**data['cadence'], **changes}}


def _with_fitness(data: dict, **changes: object) -> dict:
    caveat = data['recall_caveat']
    return {**data, 'recall_caveat': {**caveat, 'fitness': {**caveat['fitness'], **changes}}}


_BAD_CONFIGS = {
    'missing_key': lambda d: _without(d, 'scan_limit'),
    'unknown_key': lambda d: _with(d, gate_cap=3),
    'empty_projects': lambda d: _with(d, projects=[]),
    'duplicate_projects': lambda d: _with(d, projects=['dark_factory', 'dark_factory']),
    'empty_categories': lambda d: _with(d, categories=[]),
    'duplicate_categories': lambda d: _with(
        d, categories=['procedural_knowledge', 'procedural_knowledge'],
    ),
    'zero_scan_limit': lambda d: _with(d, scan_limit=0),
    'zero_detector_timeout': lambda d: _with(d, detector_timeout_secs=0),
    'zero_resweep_interval': lambda d: _with_cadence(d, resweep_interval_days=0),
    'fitness_outside_its_vocabulary': lambda d: _with_fitness(d, gate_filing='maybe'),
}


class TestConfigRejections:
    @pytest.mark.parametrize('case', sorted(_BAD_CONFIGS))
    def test_an_invalid_declaration_names_the_file(self, tmp_path: Path, case: str) -> None:
        path = _write_yaml(tmp_path, _BAD_CONFIGS[case](_committed_yaml()))

        with pytest.raises(report.DuplicateAuditConfigError, match=str(path)):
            report.load_config(path)

    def test_an_unreadable_path_names_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / 'absent.yaml'

        with pytest.raises(report.DuplicateAuditConfigError, match=str(path)):
            report.load_config(path)

    def test_malformed_yaml_names_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / 'broken.yaml'
        path.write_text('projects: [dark_factory\n', encoding='utf-8')

        with pytest.raises(report.DuplicateAuditConfigError, match=str(path)):
            report.load_config(path)


# ---------------------------------------------------------------------------
# Plans: produced by the real detector over synthetic near-verbatim records
# ---------------------------------------------------------------------------

_PK = 'procedural_knowledge'
_PN = 'preferences_and_norms'
_OS = 'observations_and_summaries'
_STAMP = '20261115T063000Z'

# Two sentence families. Rewrites within a family are near-verbatim (pairwise
# SequenceMatcher ratio >= 0.95), across families about 0.23, so the lexical
# path at its production threshold clusters each family and never the two.
_FAMILIES = {
    'venv': (
        "Each git worktree keeps its own local virtualenv, so a uv sync inside "
        "the worktree never touches the main checkout's environment{}."
    ),
    'stash': (
        "Never run git stash in any dark-factory checkout because the stash ref "
        "is shared by every worktree and the merge worker{}."
    ),
}
_SUFFIXES = ('', ' at all', ' ever', ' today', ' here', ' again', ' now', ' too', ' still')


def _rec(record_id: str, family: str, rewrite: int, created_at: str | None = None,
         *, category: str = _PK, **metadata: object) -> dict:
    return {
        'id': record_id,
        'content': _FAMILIES[family].format(_SUFFIXES[rewrite]),
        'created_at': created_at,
        'category': category,
        'metadata': dict(metadata),
    }


def _plan(records: list[dict], category: str = _PK, *, scanned: int | None = None,
          truncated: int = 0, scan_limit: int = 5000, **kwargs: object) -> dict:
    coverage = {category: {
        'scanned': len(records) if scanned is None else scanned,
        'truncated': truncated,
        'scan_limit': scan_limit,
    }}
    return detector.build_sweep_plan(
        records, 0.85, categories=(category,), scan_coverage=coverage, **kwargs,
    )


def _legacy_plan(category: str, groups: list[dict], *, ann_cutoff: float = 0.8868,
                 threshold: float = 0.85, carved: list[dict] | None = None) -> dict:
    """The T0 baseline's shape: no members, scan_coverage, child_counts or retained_carveout."""
    return {
        'near_duplicate_groups': [
            {
                'member_ids': list(g['member_ids']),
                'survivor_id': g['member_ids'][0],
                'category': category,
                'found_by': ['ann'],
                'ann_max_score': 0.91,
                'lexical_max_ratio': 0.4,
                'delete_candidate_ids': list(g['member_ids'][1:]),
            }
            for g in groups
        ],
        'threshold': threshold,
        'ann_threshold_by_category': {category: ann_cutoff},
        'topic_carveout_groups': carved or [],
    }


@pytest.fixture(scope='module')
def config():
    return report.load_config(COMMITTED_CONFIG)


def _build(config, plans, previous=None, *, stamp=_STAMP, project_id='dark_factory'):
    return report.build_report(project_id, stamp, plans, config=config, previous=previous)


def _cluster_by_member(rep: dict, member_id: str) -> dict:
    (cluster,) = [c for c in rep['clusters'] if member_id in c['member_ids']]
    return cluster


class TestReportIdentity:
    def test_the_top_level_identifies_the_run_and_disclaims_authority(self, config) -> None:
        rep = _build(config, [('plan-pk.json', _plan([]))])

        assert rep['schema_version'] == 1
        assert rep['project_id'] == 'dark_factory'
        assert rep['run_stamp'] == _STAMP
        assert rep['detector'] == 'fused-memory/scripts/audit_duplicate_memories.py'
        assert rep['authoritative'] is False
        assert rep['recall_caveat'] == config.recall_caveat.model_dump(mode='json')
        assert rep['cadence'] == config.cadence.model_dump(mode='json')


class TestClusterInventory:
    def test_every_group_of_every_plan_becomes_one_cluster_entry(self, config) -> None:
        pk = _plan([
            _rec('a1', 'venv', 0, '2026-01-01T00:00:00+00:00'),
            _rec('a2', 'venv', 1, '2026-01-02T00:00:00+00:00'),
            _rec('b1', 'stash', 0, '2026-01-01T00:00:00+00:00'),
            _rec('b2', 'stash', 1, '2026-01-02T00:00:00+00:00'),
            _rec('b3', 'stash', 2, '2026-01-03T00:00:00+00:00'),
        ])
        pn = _plan([
            _rec('c1', 'venv', 0, '2026-01-01T00:00:00+00:00', category=_PN),
            _rec('c2', 'venv', 1, '2026-01-02T00:00:00+00:00', category=_PN),
        ], _PN)

        rep = _build(config, [('pk.json', pk), ('pn.json', pn)])

        groups = pk['near_duplicate_groups'] + pn['near_duplicate_groups']
        assert len(rep['clusters']) == len(groups) == 3
        for group in groups:
            cluster = _cluster_by_member(rep, group['member_ids'][0])
            assert cluster['category'] == group['category']
            assert cluster['size'] == len(group['member_ids'])
            for key in ('member_ids', 'survivor_id', 'found_by', 'ann_max_score',
                        'lexical_max_ratio', 'delete_candidate_ids', 'retained_member_ids'):
                assert cluster[key] == group[key], key

    def test_member_facts_give_topics_canonical_presence_and_machine_authorship(
        self, config,
    ) -> None:
        pk = _plan([
            _rec('a1', 'venv', 0, '2026-01-01T00:00:00+00:00', topic='venv', canonical=True),
            _rec('a2', 'venv', 1, '2026-01-02T00:00:00+00:00', topic='uv'),
            _rec('a3', 'venv', 2, '2026-01-03T00:00:00+00:00', topic='venv'),
            _rec('b1', 'stash', 0, '2026-01-01T00:00:00+00:00', kind='cycle_summary'),
            _rec('b2', 'stash', 1, '2026-01-02T00:00:00+00:00', kind='cycle_summary'),
        ])

        rep = _build(config, [('pk.json', pk)])

        stamped = _cluster_by_member(rep, 'a1')
        assert stamped['topics'] == ['uv', 'venv']
        assert stamped['canonical_present'] is True
        assert stamped['machine_written'] is False
        summaries = _cluster_by_member(rep, 'b1')
        assert summaries['topics'] == []
        assert summaries['canonical_present'] is False
        assert summaries['machine_written'] is True

    def test_one_agent_written_member_makes_a_cluster_not_machine_written(
        self, config,
    ) -> None:
        pk = _plan([
            _rec('b1', 'stash', 0, '2026-01-01T00:00:00+00:00', kind='cycle_summary'),
            _rec('b2', 'stash', 1, '2026-01-02T00:00:00+00:00'),
        ])
        assert _build(config, [('pk.json', pk)])['clusters'][0]['machine_written'] is False

    def test_a_legacy_plan_reports_unknown_member_facts(self, config) -> None:
        legacy = _legacy_plan(_OS, [{'member_ids': ['x1', 'x2', 'x3']}])

        (cluster,) = _build(config, [('t0-os.json', legacy)])['clusters']

        assert cluster['size'] == 3
        assert cluster['retained_member_ids'] == []
        assert cluster['topics'] == []
        assert cluster['canonical_present'] is None
        assert cluster['machine_written'] is None


class TestCategoryCoverage:
    def test_each_swept_category_records_its_coverage_cutoffs_and_source(
        self, config,
    ) -> None:
        pk = _plan(
            [_rec('a1', 'venv', 0), _rec('a2', 'venv', 1)],
            scanned=5000, truncated=1, ann_threshold_by_category={_PK: 0.839},
        )
        legacy = _legacy_plan(_OS, [{'member_ids': ['x1', 'x2']}], ann_cutoff=0.8868)

        categories = _build(config, [('pk.json', pk), ('t0-os.json', legacy)])['categories']

        assert categories[_PK] == {
            'scan_coverage': {'scanned': 5000, 'truncated': 1, 'scan_limit': 5000},
            'coverage_known': True,
            'lexical_threshold': 0.85,
            'ann_cutoff': 0.839,
            'source': 'pk.json',
        }
        assert categories[_OS] == {
            'scan_coverage': None,
            'coverage_known': False,
            'lexical_threshold': 0.85,
            'ann_cutoff': 0.8868,
            'source': 't0-os.json',
        }

    def test_a_swept_category_without_clusters_is_still_listed(self, config) -> None:
        rep = _build(config, [('pn.json', _plan([], _PN))])

        assert rep['clusters'] == []
        assert rep['categories'][_PN]['coverage_known'] is True
        assert rep['categories'][_PN]['ann_cutoff'] is None

    def test_two_plans_sweeping_one_category_are_refused_naming_both(self, config) -> None:
        with pytest.raises(report.DuplicatePlanCategoryError) as caught:
            _build(config, [('first.json', _plan([])), ('second.json', _plan([]))])

        assert 'first.json' in str(caught.value)
        assert 'second.json' in str(caught.value)


class TestCarveoutTotals:
    def test_topic_carveouts_concatenate_and_retained_counts_sum(self, config) -> None:
        pk = _plan([
            _rec('a1', 'venv', 0, '2026-01-01T00:00:00+00:00'),
            _rec('a2', 'venv', 1, '2026-01-02T00:00:00+00:00', topic='venv'),
            _rec('b1', 'stash', 0, '2026-01-01T00:00:00+00:00', topic='stash'),
            _rec('b2', 'stash', 1, '2026-01-02T00:00:00+00:00', topic='stash'),
        ])
        pn = _plan([
            _rec('c1', 'venv', 0, '2026-01-01T00:00:00+00:00', category=_PN),
            _rec('c2', 'venv', 1, '2026-01-02T00:00:00+00:00', category=_PN, parent_id='c1'),
        ], _PN)
        carved = [{'member_ids': ['x1', 'x2'], 'category': _OS, 'topic': 'old'}]
        legacy = _legacy_plan(_OS, [], carved=carved)

        rep = _build(config, [('pk.json', pk), ('pn.json', pn), ('t0-os.json', legacy)])

        assert rep['topic_carveout_groups'] == (
            pk['topic_carveout_groups'] + pn['topic_carveout_groups'] + carved
        )
        assert len(rep['topic_carveout_groups']) == 2
        assert rep['retained_carveout'] == {'topic_stamped': 1, 'child_record': 1}


class TestChildRankings:
    def test_children_are_ranked_per_kind_across_plans(self, config) -> None:
        pk = _plan([
            _rec('p', 'venv', 0, '2026-01-01T00:00:00+00:00'),
            _rec('s1', 'venv', 1, parent_id='p', kind='sighting'),
            _rec('s2', 'venv', 2, parent_id='p', kind='sighting'),
            _rec('s3', 'stash', 0, parent_id='q', kind='sighting'),
            _rec('m1', 'stash', 1, parent_id='q', kind='amendment'),
        ])
        pn = _plan([
            _rec('s4', 'venv', 0, category=_PN, parent_id='q', kind='sighting'),
            _rec('s5', 'venv', 1, category=_PN, parent_id='q', kind='sighting'),
            _rec('s6', 'stash', 0, category=_PN, parent_id='r', kind='sighting'),
        ], _PN)

        rep = _build(config, [('pk.json', pk), ('pn.json', pn)])

        assert rep['child_counts_known'] is True
        assert rep['child_rankings']['sighting'] == [
            {'parent_id': 'q', 'count': 3},
            {'parent_id': 'p', 'count': 2},
            {'parent_id': 'r', 'count': 1},
        ]
        assert rep['child_rankings']['amendment'] == [{'parent_id': 'q', 'count': 1}]

    def test_a_legacy_plan_makes_the_child_counts_unknown(self, config) -> None:
        rep = _build(config, [
            ('pk.json', _plan([])),
            ('t0-os.json', _legacy_plan(_OS, [])),
        ])

        assert rep['child_counts_known'] is False
        assert rep['child_rankings'] == {}
