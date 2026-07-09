"""T2: verify-admission config knobs + green-tier reload registration
(task 2390; PRD plans/verify-oversubscription-control-prd.md).

Fixtures are kept MODULE-LOCAL (not conftest.py) — a conftest.py edit trips
verify.py's has_conftest and forces the merge-time verify to fall back to
running the full owning-package suite instead of a scoped subset (mirrors
test_config_psi_admission_reload.py's stated rationale).
"""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    apply_reload,
    diff_config,
)


class TestVerifyAdmissionConfigDefaults:
    """Default-on (spec-mandated) with a conservative single-slot floor and
    empty per-role nice overrides (defer to shared.verify_admission.nice_prefix).
    """

    def test_defaults_on_bare_config(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig()
        assert cfg.verify_admission_enabled is True
        assert cfg.verify_admission_task_slots == 1
        assert cfg.verify_admission_slots_dir == f'/tmp/df-verify-slots-{os.getuid()}'
        assert cfg.verify_admission_nice_merge == ''
        assert cfg.verify_admission_nice_task == ''
        assert cfg.verify_admission_nice_background == ''


class TestVerifyAdmissionTaskSlotsValidation:
    """ge=1 floor: a non-positive slot count would admit nothing (or is
    nonsensical), so it is rejected at construction.
    """

    @pytest.mark.parametrize('bad_value', [0, -1])
    def test_sub_minimum_rejected(self, bad_value):
        with pytest.raises(ValidationError):
            OrchestratorConfig(verify_admission_task_slots=bad_value)

    @pytest.mark.parametrize('good_value', [1, 2, 4])
    def test_valid_values_accepted(self, good_value):
        cfg = OrchestratorConfig(verify_admission_task_slots=good_value)
        assert cfg.verify_admission_task_slots == good_value


class TestVerifyAdmissionReloadDisposition:
    """Every verify_admission knob is green-tier: hot-reloadable without a
    process restart, mirroring the psi_admission submodel-group pattern.
    """

    @pytest.mark.parametrize(
        'field',
        [
            'verify_admission_enabled',
            'verify_admission_task_slots',
            'verify_admission_slots_dir',
            'verify_admission_nice_merge',
            'verify_admission_nice_task',
            'verify_admission_nice_background',
        ],
    )
    def test_every_field_is_reloadable(self, field):
        assert field in RELOADABLE_FIELDS, (
            f'{field!r} is expected to be green-tier reloadable but is '
            f'missing from RELOADABLE_FIELDS'
        )

    def test_task_slots_edit_lands_in_applied_candidates_not_restart_required(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig(verify_admission_task_slots=1)
        fresh = OrchestratorConfig(verify_admission_task_slots=2)
        diff = diff_config(live, fresh)
        assert 'verify_admission_task_slots' in diff.applied_candidates
        assert 'verify_admission_task_slots' not in diff.restart_required

    def test_apply_reload_applies_in_place(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig(verify_admission_task_slots=1)
        fresh = OrchestratorConfig(verify_admission_task_slots=2)
        report = apply_reload(live, fresh)
        assert report['reloaded'] is True
        assert report['applied']['verify_admission_task_slots'] == {
            'old': 1, 'new': 2,
        }
        assert 'verify_admission_task_slots' not in report['restart_required']
        assert live.verify_admission_task_slots == 2
