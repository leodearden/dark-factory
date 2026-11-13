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

import json
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
          truncated: int = 0, scan_limit: int = 5000, threshold: float = 0.85,
          **kwargs: object) -> dict:
    coverage = {category: {
        'scanned': len(records) if scanned is None else scanned,
        'truncated': truncated,
        'scan_limit': scan_limit,
    }}
    return detector.build_sweep_plan(
        records, threshold, categories=(category,), scan_coverage=coverage, **kwargs,
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


def _project(entry: dict, keys: dict | set) -> dict:
    """*entry* cut down to *keys*, so an inventory assertion ignores the diff's fields."""
    return {k: entry[k] for k in keys if k in entry}


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

        expected_pk = {
            'scan_coverage': {'scanned': 5000, 'truncated': 1, 'scan_limit': 5000},
            'coverage_known': True,
            'lexical_threshold': 0.85,
            'ann_cutoff': 0.839,
            'source': 'pk.json',
        }
        expected_os = {
            'scan_coverage': None,
            'coverage_known': False,
            'lexical_threshold': 0.85,
            'ann_cutoff': 0.8868,
            'source': 't0-os.json',
        }
        assert _project(categories[_PK], expected_pk) == expected_pk
        assert _project(categories[_OS], expected_os) == expected_os

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

        rankings = {
            kind: [_project(row, {'parent_id', 'count'}) for row in rows]
            for kind, rows in rep['child_rankings'].items()
        }
        assert rep['child_counts_known'] is True
        assert rankings['sighting'] == [
            {'parent_id': 'q', 'count': 3},
            {'parent_id': 'p', 'count': 2},
            {'parent_id': 'r', 'count': 1},
        ]
        assert rankings['amendment'] == [{'parent_id': 'q', 'count': 1}]

    def test_a_legacy_plan_makes_the_child_counts_unknown(self, config) -> None:
        rep = _build(config, [
            ('pk.json', _plan([])),
            ('t0-os.json', _legacy_plan(_OS, [])),
        ])

        assert rep['child_counts_known'] is False
        assert rep['child_rankings'] == {}


# ---------------------------------------------------------------------------
# The consecutive-run regrowth diff (task 3136 amendment D)
# ---------------------------------------------------------------------------

_PREV = '20261108T063000Z'
_CUR = '20261115T063000Z'
_BEFORE_PREV = '2026-11-01T00:00:00+00:00'
_AFTER_PREV = '2026-11-10T00:00:00+00:00'


def _family(ids: list[str], family: str = 'venv', created_at: str = _BEFORE_PREV,
            *, category: str = _PK) -> list[dict]:
    return [_rec(i, family, n, created_at, category=category) for n, i in enumerate(ids)]


def _runs(config, prev_records, cur_records, *, prev_stamp=_PREV, cur_stamp=_CUR,
          prev_plan_kwargs=None, cur_plan_kwargs=None):
    prev = _build(config, [('prev.json', _plan(prev_records, **(prev_plan_kwargs or {})))],
                  stamp=prev_stamp)
    cur = _build(config, [('cur.json', _plan(cur_records, **(cur_plan_kwargs or {})))],
                 previous=prev, stamp=cur_stamp)
    return prev, cur


class TestBaselineLineage:
    def test_without_a_previous_report_every_cluster_is_a_baseline(self, config) -> None:
        rep = _build(config, [('pk.json', _plan(
            _family(['b', 'a']) + _family(['d', 'c'], 'stash'),
        ))])

        assert [c['lineage']['status'] for c in rep['clusters']] == ['baseline', 'baseline']
        assert [c['cluster_key'] for c in rep['clusters']] == ['a', 'c']
        assert rep['regrowth']['previous_run_stamp'] is None


class TestMatchedLineage:
    def test_an_unchanged_cluster_keeps_the_key_it_was_born_with(self, config) -> None:
        prev = _build(config, [('prev.json', _plan(_family(['a', 'b'])))], stamp=_PREV)
        prev['clusters'][0]['cluster_key'] = 'born-key'

        cur = _build(config, [('cur.json', _plan(_family(['a', 'b'])))],
                     previous=prev, stamp=_CUR)

        (cluster,) = cur['clusters']
        assert cluster['cluster_key'] == 'born-key'
        assert cluster['lineage']['status'] == 'unchanged'
        assert cluster['lineage']['size_delta'] == 0
        assert cur['regrowth']['unchanged'] == 1

    def test_a_grown_cluster_splits_its_added_members_by_age(self, config) -> None:
        _prev, cur = _runs(config, _family(['a', 'b']), [
            *_family(['a', 'b']),
            _rec('c', 'venv', 2, _AFTER_PREV),
            _rec('d', 'venv', 3, _BEFORE_PREV),
            _rec('e', 'venv', 4, None),
        ])

        (cluster,) = cur['clusters']
        lineage = cluster['lineage']
        assert cluster['cluster_key'] == 'a'
        assert lineage['status'] == 'grew'
        assert lineage['previous_key'] == 'a'
        assert lineage['added_member_ids'] == ['c', 'd', 'e']
        assert lineage['size_delta'] == 3
        assert lineage['added_by_age'] == {'re_emitted': 1, 'preexisting': 1, 'unknown': 1}
        assert lineage['rate_per_day'] == pytest.approx(1 / 7)
        regrowth = cur['regrowth']
        assert regrowth['grew'] == 1
        assert regrowth['re_emitted_members'] == 1
        assert regrowth['elapsed_days'] == pytest.approx(7.0)
        assert regrowth['previous_run_stamp'] == _PREV

    def test_the_rate_is_none_when_no_time_has_elapsed(self, config) -> None:
        _prev, cur = _runs(
            config, _family(['a', 'b']),
            [*_family(['a', 'b']), _rec('c', 'venv', 2, _AFTER_PREV)],
            cur_stamp=_PREV,
        )

        assert cur['clusters'][0]['lineage']['rate_per_day'] is None
        assert cur['regrowth']['rate_per_day'] is None

    def test_a_shrunk_cluster_lists_its_departed_members(self, config) -> None:
        _prev, cur = _runs(config, _family(['a', 'b', 'c']), _family(['a', 'b']))

        lineage = cur['clusters'][0]['lineage']
        assert lineage['status'] == 'shrank'
        assert lineage['departed_member_ids'] == ['c']
        assert lineage['size_delta'] == -1
        assert cur['regrowth']['shrank'] == 1


class TestAppearedAndDissolved:
    def test_a_cluster_overlapping_nothing_is_new_with_a_fresh_key(self, config) -> None:
        _prev, cur = _runs(
            config, _family(['a', 'b']),
            _family(['a', 'b']) + _family(['y', 'x'], 'stash', _AFTER_PREV),
        )

        fresh = _cluster_by_member(cur, 'x')
        assert fresh['lineage']['status'] == 'new'
        assert fresh['cluster_key'] == 'x'
        assert fresh['lineage']['added_by_age']['re_emitted'] == 2
        assert cur['regrowth']['appeared'] == 1

    def test_a_previous_cluster_overlapped_by_nothing_has_dissolved(self, config) -> None:
        _prev, cur = _runs(
            config, _family(['a', 'b']) + _family(['c', 'd', 'e'], 'stash'),
            _family(['a', 'b']),
        )

        assert cur['regrowth']['dissolved'] == [
            {'cluster_key': 'c', 'category': _PK, 'size': 3},
        ]

    def test_an_unswept_category_dissolves_nothing(self, config) -> None:
        prev = _build(config, [
            ('prev-pk.json', _plan(_family(['a', 'b']))),
            ('prev-pn.json', _plan(_family(['c', 'd'], category=_PN), _PN)),
        ], stamp=_PREV)

        cur = _build(config, [('cur-pk.json', _plan(_family(['a', 'b'])))],
                     previous=prev, stamp=_CUR)

        assert cur['regrowth']['dissolved'] == []


class TestMergeAndSplit:
    def test_a_merged_cluster_inherits_the_larger_overlap(self, config) -> None:
        prev = _family(['a', 'b', 'c']) + _family(['d', 'e'], 'stash')
        _prev, cur = _runs(config, prev, _family(['a', 'b', 'c', 'd', 'e']))

        (cluster,) = cur['clusters']
        assert cluster['cluster_key'] == 'a'
        assert cluster['lineage']['merged_previous_keys'] == ['a', 'd']
        assert cur['regrowth']['dissolved'] == []

    def test_an_overlap_tie_goes_to_the_lower_previous_key(self, config) -> None:
        prev = _family(['c', 'd']) + _family(['a', 'b'], 'stash')
        _prev, cur = _runs(config, prev, _family(['a', 'b', 'c', 'd']))

        assert cur['clusters'][0]['cluster_key'] == 'a'

    def test_a_split_keeps_the_key_on_the_larger_piece(self, config) -> None:
        _prev, cur = _runs(
            config, _family(['a', 'b', 'c', 'd', 'e']),
            _family(['a', 'b', 'c']) + _family(['d', 'e'], 'stash'),
        )

        larger, smaller = _cluster_by_member(cur, 'a'), _cluster_by_member(cur, 'd')
        assert larger['cluster_key'] == 'a'
        assert smaller['lineage']['split_from'] == 'a'
        assert smaller['cluster_key'] != 'a'

    def test_keys_stay_unique_when_a_split_piece_holds_the_old_key_member(
        self, config,
    ) -> None:
        _prev, cur = _runs(
            config, _family(['a', 'b', 'c']),
            _family(['b', 'c']) + _family(['a', 'z'], 'stash'),
        )

        keys = [c['cluster_key'] for c in cur['clusters']]
        assert _cluster_by_member(cur, 'b')['cluster_key'] == 'a'
        assert len(set(keys)) == len(keys)

    def test_matching_never_crosses_categories(self, config) -> None:
        prev = _build(config, [('prev.json', _plan(_family(['a', 'b'])))], stamp=_PREV)

        cur = _build(config, [
            ('cur-pk.json', _plan([])),
            ('cur-pn.json', _plan(_family(['a', 'b'], category=_PN), _PN)),
        ], previous=prev, stamp=_CUR)

        assert cur['clusters'][0]['lineage']['status'] == 'new'
        assert cur['regrowth']['dissolved'] == [
            {'cluster_key': 'a', 'category': _PK, 'size': 2},
        ]


class TestComparability:
    def _reasons(self, config, prev_kwargs=None, cur_kwargs=None, prev_plans=None):
        prev = _build(config, prev_plans or [
            ('prev.json', _plan(_family(['a', 'b']), **(prev_kwargs or {}))),
        ], stamp=_PREV)
        cur = _build(config, [
            ('cur.json', _plan(_family(['a', 'b']), **(cur_kwargs or {}))),
        ], previous=prev, stamp=_CUR)
        return cur['categories'][_PK]['incomparable_reasons'], cur

    def test_matching_runs_are_comparable(self, config) -> None:
        assert self._reasons(config)[0] == []

    @pytest.mark.parametrize(('prev_kwargs', 'cur_kwargs', 'reason'), [
        ({}, {'truncated': 1}, 'scan_truncated'),
        ({'truncated': 1}, {}, 'scan_truncated'),
        ({'scan_limit': 5000}, {'scan_limit': 4000}, 'scan_limit_changed'),
        ({'threshold': 0.85}, {'threshold': 0.8}, 'lexical_threshold_changed'),
        ({'ann_threshold_by_category': {_PK: 0.839}},
         {'ann_threshold_by_category': {_PK: 0.85}}, 'ann_cutoff_changed'),
    ])
    def test_each_changed_condition_is_disclosed(
        self, config, prev_kwargs, cur_kwargs, reason,
    ) -> None:
        assert self._reasons(config, prev_kwargs, cur_kwargs)[0] == [reason]

    def test_a_legacy_previous_run_makes_coverage_unknown(self, config) -> None:
        legacy = _legacy_plan(_PK, [{'member_ids': ['a', 'b']}], ann_cutoff=0.839)

        reasons, _cur = self._reasons(
            config, cur_kwargs={'ann_threshold_by_category': {_PK: 0.839}},
            prev_plans=[('t0-pk.json', legacy)],
        )

        assert reasons == ['coverage_unknown']

    def test_a_category_the_previous_run_lacks_is_disclosed(self, config) -> None:
        prev = _build(config, [('prev.json', _plan(_family(['a', 'b'])))], stamp=_PREV)
        cur = _build(config, [
            ('cur-pk.json', _plan(_family(['a', 'b']))),
            ('cur-pn.json', _plan(_family(['c', 'd'], category=_PN), _PN)),
        ], previous=prev, stamp=_CUR)

        assert cur['categories'][_PN]['incomparable_reasons'] == ['previous_missing_category']
        assert cur['categories'][_PK]['incomparable_reasons'] == []

    def test_reasons_are_sorted_and_the_diff_is_still_computed(self, config) -> None:
        reasons, cur = self._reasons(
            config, cur_kwargs={'truncated': 1, 'scan_limit': 4000, 'threshold': 0.8},
        )

        assert reasons == sorted(reasons) == [
            'lexical_threshold_changed', 'scan_limit_changed', 'scan_truncated',
        ]
        assert cur['clusters'][0]['lineage']['status'] == 'unchanged'


class TestOverdueAndChildDeltas:
    def test_a_previous_report_older_than_twice_the_cadence_is_overdue(self, config) -> None:
        _prev, late = _runs(config, _family(['a', 'b']), _family(['a', 'b']),
                            prev_stamp='20261018T063000Z')
        _prev, on_time = _runs(config, _family(['a', 'b']), _family(['a', 'b']))

        assert late['regrowth']['previous_overdue'] is True
        assert on_time['regrowth']['previous_overdue'] is False

    def test_each_child_ranking_row_carries_its_previous_count_and_delta(
        self, config,
    ) -> None:
        _prev, cur = _runs(
            config,
            [_rec('p', 'venv', 0, _BEFORE_PREV),
             _rec('s1', 'venv', 1, _BEFORE_PREV, parent_id='p', kind='sighting')],
            [_rec('p', 'venv', 0, _BEFORE_PREV),
             _rec('s1', 'venv', 1, _BEFORE_PREV, parent_id='p', kind='sighting'),
             _rec('s2', 'venv', 2, _AFTER_PREV, parent_id='p', kind='sighting'),
             _rec('s3', 'stash', 0, _AFTER_PREV, parent_id='q', kind='sighting')],
        )

        assert cur['child_rankings']['sighting'] == [
            {'parent_id': 'p', 'count': 2, 'previous_count': 1, 'delta': 1},
            {'parent_id': 'q', 'count': 1, 'previous_count': 0, 'delta': 1},
        ]


class TestWarmLaneShape:
    def test_an_unstamped_cluster_grows_beside_a_stamped_canonical_singleton(
        self, config,
    ) -> None:
        canonical = _rec('w', 'stash', 0, _BEFORE_PREV, topic='warm-lane', canonical=True)
        old = [f'r{n}' for n in range(7)]
        _prev, cur = _runs(
            config,
            [canonical, *_family(old)],
            [canonical, *_family(old),
             _rec('r7', 'venv', 7, _AFTER_PREV), _rec('r8', 'venv', 8, _AFTER_PREV)],
        )

        (cluster,) = cur['clusters']
        assert 'w' not in cluster['member_ids']
        assert cluster['topics'] == []
        assert cluster['size'] == 9
        assert cluster['lineage']['status'] == 'grew'
        assert cluster['lineage']['added_member_ids'] == ['r7', 'r8']
        assert cluster['lineage']['added_by_age']['re_emitted'] == 2
        assert cur['regrowth']['re_emitted_members'] == 2


# ---------------------------------------------------------------------------
# Artifacts and the store-free --plan CLI mode
# ---------------------------------------------------------------------------

_FAILED_OS = {
    'category': _OS, 'error_type': 'timeout', 'exit_code': None, 'stderr_tail': 'slow',
}


def _grown_report(config) -> dict:
    prev = _build(config, [('prev.json', _plan(_family(['a', 'b'])))], stamp=_PREV)
    return report.build_report(
        'dark_factory', _CUR,
        [('cur.json', _plan(
            [*_family(['a', 'b']), _rec('c', 'venv', 2, _AFTER_PREV)], truncated=1,
        ))],
        config=config, previous=prev, failed_categories=[_FAILED_OS],
    )


class TestWriteReport:
    def test_the_report_and_its_summary_land_under_the_project(
        self, config, tmp_path: Path,
    ) -> None:
        rep = _grown_report(config)

        json_path, summary_path = report.write_report(tmp_path, 'dark_factory', _CUR, rep)

        assert json_path == tmp_path / 'dark_factory' / f'report-{_CUR}.json'
        assert summary_path == tmp_path / 'dark_factory' / f'summary-{_CUR}.txt'
        text = json_path.read_text(encoding='utf-8')
        assert text.endswith('\n')
        assert json.loads(text) == json.loads(json.dumps(rep))
        assert text == json.dumps(rep, indent=2, sort_keys=True) + '\n'

    def test_the_summary_states_fitness_coverage_regrowth_and_failures(
        self, config, tmp_path: Path,
    ) -> None:
        _json_path, summary_path = report.write_report(
            tmp_path, 'dark_factory', _CUR, _grown_report(config),
        )
        summary = summary_path.read_text(encoding='utf-8')

        assert 'dark_factory' in summary
        assert _CUR in summary
        assert 'regrowth measurement: supported; gate filing: blocked (task 4916)' in summary
        (pk_line,) = [line for line in summary.splitlines() if line.strip().startswith(_PK)]
        assert '1 cluster' in pk_line
        assert 'scan_truncated' in pk_line
        assert 'grew 1' in summary
        assert 're-emitted members 1' in summary
        (top_line,) = [line for line in summary.splitlines() if '+1 added' in line]
        assert top_line.split()[0] == 'a'
        (failed_line,) = [line for line in summary.splitlines() if line.strip().startswith(_OS)]
        assert 'timeout' in failed_line


class TestFindPreviousReport:
    def _write(self, root: Path, project: str, stamp: str, project_id: str | None = None):
        path = root / project / f'report-{stamp}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'project_id': project_id or project, 'run_stamp': stamp}))
        return path

    def test_the_latest_report_strictly_before_the_stamp_is_returned(
        self, tmp_path: Path,
    ) -> None:
        for stamp in ('20261101T000000Z', '20261108T000000Z', '20261115T000000Z'):
            self._write(tmp_path, 'p', stamp)

        found = report.find_previous_report(tmp_path, 'p', '20261115T000000Z')

        assert found is not None
        stamp, prior = found
        assert stamp == '20261108T000000Z'
        assert prior['run_stamp'] == '20261108T000000Z'

    def test_malformed_names_and_other_projects_are_ignored(self, tmp_path: Path) -> None:
        self._write(tmp_path, 'p', '20261101T000000Z')
        self._write(tmp_path, 'p', '2026-11-10T00:00:00Z')
        self._write(tmp_path, 'p', '20261112T000000Z', project_id='q')

        found = report.find_previous_report(tmp_path, 'p', '20261115T000000Z')

        assert found is not None
        assert found[0] == '20261101T000000Z'

    def test_no_earlier_report_is_none(self, tmp_path: Path) -> None:
        self._write(tmp_path, 'p', '20261115T000000Z')
        assert report.find_previous_report(tmp_path, 'p', '20261115T000000Z') is None
        assert report.find_previous_report(tmp_path, 'absent', '20261115T000000Z') is None


def _dump(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    return _write_yaml(tmp_path, _committed_yaml(), name='config.yaml')


class TestPlanModeCli:
    def _argv(self, config_path: Path, out_root: Path, *plans: Path,
              stamp: str = '20260811T120000Z', project_id: str | None = 'p') -> list[str]:
        argv = ['--config', str(config_path), '--out-root', str(out_root), '--run-stamp', stamp]
        if project_id is not None:
            argv += ['--project-id', project_id]
        for plan in plans:
            argv += ['--plan', str(plan)]
        return argv

    def test_a_report_is_built_from_saved_plans_and_the_next_one_diffs_against_it(
        self, config_path: Path, tmp_path: Path,
    ) -> None:
        out_root = tmp_path / 'reports'
        first = [
            _dump(tmp_path / 'pk-1.json', _plan(_family(['a', 'b']))),
            _dump(tmp_path / 'pn-1.json', _plan([], _PN)),
        ]
        second = [
            _dump(tmp_path / 'pk-2.json', _plan(_family(['a', 'b', 'c']))),
            _dump(tmp_path / 'pn-2.json', _plan([], _PN)),
        ]

        assert report.main(self._argv(config_path, out_root, *first)) == report.EXIT_OK
        assert report.main(self._argv(
            config_path, out_root, *second, stamp='20260818T120000Z',
        )) == report.EXIT_OK

        rep = json.loads((out_root / 'p' / 'report-20260818T120000Z.json').read_text())
        assert rep['regrowth']['previous_run_stamp'] == '20260811T120000Z'
        assert rep['categories'][_PK]['source'] == str(second[0])
        assert rep['categories'][_PN]['source'] == str(second[1])
        assert rep['clusters'][0]['lineage']['status'] == 'grew'
        assert rep['failed_categories'] == []
        assert (out_root / 'p' / 'summary-20260818T120000Z.txt').exists()

    def test_a_missing_plan_file_is_bad_input_naming_it(
        self, config_path: Path, tmp_path: Path, caplog,
    ) -> None:
        missing = tmp_path / 'absent.json'

        code = report.main(self._argv(config_path, tmp_path / 'out', missing))

        assert code == report.EXIT_BAD_INPUT == 2
        assert str(missing) in caplog.text

    def test_a_plan_that_is_not_json_is_bad_input_naming_it(
        self, config_path: Path, tmp_path: Path, caplog,
    ) -> None:
        broken = tmp_path / 'broken.json'
        broken.write_text('{not json', encoding='utf-8')

        assert report.main(self._argv(config_path, tmp_path / 'out', broken)) == 2
        assert str(broken) in caplog.text

    def test_two_plans_sweeping_one_category_are_bad_input(
        self, config_path: Path, tmp_path: Path,
    ) -> None:
        plans = [_dump(tmp_path / f'pk-{n}.json', _plan([])) for n in (1, 2)]
        assert report.main(self._argv(config_path, tmp_path / 'out', *plans)) == 2

    def test_a_malformed_run_stamp_is_bad_input(self, config_path: Path, tmp_path: Path) -> None:
        plan = _dump(tmp_path / 'pk.json', _plan([]))
        code = report.main(self._argv(
            config_path, tmp_path / 'out', plan, stamp='2026-08-11T12:00:00Z',
        ))
        assert code == 2

    def test_a_config_error_is_bad_input(self, tmp_path: Path, caplog) -> None:
        bad = _write_yaml(tmp_path, _without(_committed_yaml(), 'projects'), name='bad.yaml')
        plan = _dump(tmp_path / 'pk.json', _plan([]))

        assert report.main(self._argv(bad, tmp_path / 'out', plan)) == 2
        assert str(bad) in caplog.text

    def test_plans_without_a_project_are_bad_input(
        self, config_path: Path, tmp_path: Path,
    ) -> None:
        plan = _dump(tmp_path / 'pk.json', _plan([]))
        code = report.main(self._argv(config_path, tmp_path / 'out', plan, project_id=None))
        assert code == 2

    def test_an_unwritable_out_root_aborts_the_run_without_a_traceback(
        self, config_path: Path, tmp_path: Path, caplog, capsys,
    ) -> None:
        not_a_dir = tmp_path / 'regular-file'
        not_a_dir.write_text('', encoding='utf-8')
        plan = _dump(tmp_path / 'pk.json', _plan([]))

        code = report.main(self._argv(config_path, not_a_dir, plan))

        assert code == report.EXIT_RUN_FAILED == 1
        assert 'ABORT: cannot write the report' in caplog.text
        assert 'Traceback' not in caplog.text + capsys.readouterr().err
