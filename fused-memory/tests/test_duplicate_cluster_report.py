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
