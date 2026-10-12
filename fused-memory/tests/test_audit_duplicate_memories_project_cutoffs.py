"""Project-aware ANN cutoff resolution in audit_duplicate_memories.py (task 5547).

A run's ``--project-id`` selects that project's entry in
``config.write_triage_project_calibrations``: cutoffs measured on the
project's OWN curator-labelled fixture. Per category the priority becomes
override -> project map -> global ``t_high_by_category`` -> pooled ``t_high``
(a disclosed fallback) -> disabled. The resolver and the ``--apply`` fence
predicate must agree on which categories ran on their own evidence.

The service double is a MagicMock whose project map holds REAL
``WriteTriageProjectCalibration`` instances, so the attribute reads are the
production ones. Every number is synthetic and distinctive.
"""
from __future__ import annotations

import logging
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _fm_helpers import load_script_module

from fused_memory.config.schema import WriteTriageProjectCalibration

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'audit_duplicate_memories.py'
_mod = load_script_module(SCRIPT_PATH, mod_name='audit_duplicate_memories')
resolve_ann_threshold = _mod.resolve_ann_threshold
categories_without_own_cutoff = _mod.categories_without_own_cutoff

_PK = 'procedural_knowledge'
_OS = 'observations_and_summaries'
_PN = 'preferences_and_norms'
_SWEPT = (_PK, _OS, _PN)

POOLED = 0.4141414141
GLOBAL_PK = 0.8686868686
GLOBAL_OS = 0.7272727272
PROJECT_OS = 0.9191919191
PROJECT_PK = 0.6363636363
REPORT = 'calibration/write_triage_calibration_report_dark_factory.json'
_UNSET = object()


def _entry(cutoffs: dict[str, float]) -> WriteTriageProjectCalibration:
    return WriteTriageProjectCalibration(
        calibration_report_path=REPORT, t_high_by_category=cutoffs,
    )


def _service(global_map: dict | None, projects=_UNSET) -> MagicMock:
    service = MagicMock()
    service.config.write_triage.t_high = POOLED
    service.config.write_triage.t_high_by_category = global_map
    if projects is not _UNSET:
        service.config.write_triage_project_calibrations = projects
    return service


def _resolve(service, project_id=None, override=None):
    return resolve_ann_threshold(service, override, _SWEPT, project_id=project_id)


class TestProjectCutoffResolution:
    def test_the_project_map_fills_the_category_it_calibrates(self) -> None:
        service = _service({_PK: GLOBAL_PK}, {'dark_factory': _entry({_OS: PROJECT_OS})})

        resolved, disclosure = _resolve(service, 'dark_factory')

        assert resolved == {_PK: GLOBAL_PK, _OS: PROJECT_OS, _PN: POOLED}
        assert disclosure == {
            'ann_pooled_fallback_categories': 1, 'ann_disabled_uncalibrated': 0,
        }

    def test_a_project_cutoff_wins_over_the_global_one_and_is_not_a_fallback(self) -> None:
        service = _service(
            {_PK: GLOBAL_PK}, {'dark_factory': _entry({_PK: PROJECT_PK, _OS: PROJECT_OS})},
        )

        resolved, disclosure = _resolve(service, 'dark_factory')

        assert resolved[_PK] == PROJECT_PK
        assert disclosure['ann_pooled_fallback_categories'] == 1

    @pytest.mark.parametrize('project_id', ['reify', 'unknown_project'])
    def test_a_project_without_an_entry_resolves_as_the_status_quo(
        self, project_id: str,
    ) -> None:
        service = _service({_PK: GLOBAL_PK}, {'dark_factory': _entry({_OS: PROJECT_OS})})
        assert _resolve(service, project_id) == _resolve(service, None)

    def test_the_default_ignores_every_project_entry(self) -> None:
        service = _service({_PK: GLOBAL_PK}, {'dark_factory': _entry({_OS: PROJECT_OS})})
        bare = _service({_PK: GLOBAL_PK}, {})

        assert resolve_ann_threshold(service, None, _SWEPT) == resolve_ann_threshold(
            bare, None, _SWEPT,
        )

    def test_an_override_still_wins_over_the_project_cutoff(self) -> None:
        service = _service({_PK: GLOBAL_PK}, {'dark_factory': _entry({_OS: PROJECT_OS})})

        resolved, _disclosure = _resolve(service, 'dark_factory', override=0.5)

        assert resolved == dict.fromkeys(_SWEPT, 0.5)

    def test_a_project_cutoff_is_logged_with_its_report(self, caplog) -> None:
        service = _service({_PK: GLOBAL_PK}, {'dark_factory': _entry({_OS: PROJECT_OS})})

        with caplog.at_level(logging.INFO):
            _resolve(service, 'dark_factory')

        records = [r for r in caplog.records
                   if r.levelno == logging.INFO and 'dark_factory' in r.getMessage()]
        assert records, 'a project cutoff must be named where it is used'
        assert REPORT in records[0].getMessage()
        assert _OS in records[0].getMessage()


class TestTheApplyFenceAgreesWithTheResolver:
    def test_a_project_calibrated_category_is_evidenced(self) -> None:
        service = _service({_PK: GLOBAL_PK}, {'dark_factory': _entry({_OS: PROJECT_OS})})

        unevidenced = categories_without_own_cutoff(
            service, _SWEPT, None, project_id='dark_factory',
        )

        assert unevidenced == frozenset({_PN})
        resolved, disclosure = _resolve(service, 'dark_factory')
        assert len(unevidenced) == (
            disclosure['ann_pooled_fallback_categories']
            + disclosure['ann_disabled_uncalibrated']
        )
        assert all(resolved.get(c) == POOLED for c in unevidenced)

    def test_an_override_fences_every_category(self) -> None:
        service = _service({_PK: GLOBAL_PK}, {'dark_factory': _entry({_OS: PROJECT_OS})})

        assert categories_without_own_cutoff(
            service, _SWEPT, 0.5, project_id='dark_factory',
        ) == frozenset(_SWEPT)

    def test_another_projects_entry_evidences_nothing(self) -> None:
        service = _service({_PK: GLOBAL_PK}, {'other_project': _entry({_OS: PROJECT_OS})})

        assert categories_without_own_cutoff(
            service, _SWEPT, None, project_id='dark_factory',
        ) == frozenset({_OS, _PN})


_JUNK = {
    'unspecced_mock_attribute': _UNSET,
    'scalar': 0.5,
    'entry_map_not_a_mapping': {'dark_factory': types.SimpleNamespace(
        calibration_report_path=REPORT, t_high_by_category=0.9,
    )},
    'bool_cutoff': {'dark_factory': types.SimpleNamespace(
        calibration_report_path=REPORT, t_high_by_category={_OS: True},
    )},
}


class TestJunkIsIgnoredWholesale:
    """Junk falls back EXACTLY as if there were no project map at all."""

    @pytest.mark.parametrize('projects', list(_JUNK.values()), ids=list(_JUNK))
    def test_resolution_falls_back_as_with_no_project_map(self, projects) -> None:
        global_map = {_PK: GLOBAL_PK, _OS: GLOBAL_OS}
        junk = _service(global_map, projects)
        bare = _service(global_map, {})

        assert _resolve(junk, 'dark_factory') == _resolve(bare, 'dark_factory')
        assert categories_without_own_cutoff(
            junk, _SWEPT, None, project_id='dark_factory',
        ) == categories_without_own_cutoff(bare, _SWEPT, None, project_id='dark_factory')
