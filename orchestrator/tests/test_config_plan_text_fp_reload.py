"""Green-tier registration + default for ``plan_text_fp_gate_enabled`` — the
task-4031 kill switch on the task-text amendment gate in front of the Lever-B
revalidation skip.

Fixtures are kept MODULE-LOCAL (not conftest.py) for the reason stated in
test_config_verify_admission_reload.py: a conftest.py edit trips verify.py's
has_conftest and forces the merge-time verify to run the full owning-package
suite instead of a scoped subset.
"""

from __future__ import annotations

from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    apply_reload,
    diff_config,
)


def test_defaults_on(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')
    assert OrchestratorConfig().plan_text_fp_gate_enabled is True


def test_field_is_reloadable():
    assert 'plan_text_fp_gate_enabled' in RELOADABLE_FIELDS


def test_skip_neighbours_stay_restart_only():
    assert 'revalidation_skip_enabled' not in RELOADABLE_FIELDS
    assert 'max_revalidation_age_hours' not in RELOADABLE_FIELDS


def test_edit_lands_in_applied_candidates_not_restart_required(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')
    diff = diff_config(
        OrchestratorConfig(plan_text_fp_gate_enabled=True),
        OrchestratorConfig(plan_text_fp_gate_enabled=False),
    )
    assert 'plan_text_fp_gate_enabled' in diff.applied_candidates
    assert 'plan_text_fp_gate_enabled' not in diff.restart_required


def test_apply_reload_applies_in_place(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')
    live = OrchestratorConfig(plan_text_fp_gate_enabled=True)
    report = apply_reload(live, OrchestratorConfig(plan_text_fp_gate_enabled=False))
    assert report['reloaded'] is True
    assert report['applied']['plan_text_fp_gate_enabled'] == {'old': True, 'new': False}
    assert live.plan_text_fp_gate_enabled is False
