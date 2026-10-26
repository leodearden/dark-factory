"""Default + green-tier registration for ``pre_merge_reverify`` — the task-6580
rollback switch for the merge phase's advisory post-rebase re-verify.

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


def test_code_default_is_off(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')
    assert OrchestratorConfig().pre_merge_reverify is False


def test_field_is_reloadable():
    assert 'pre_merge_reverify' in RELOADABLE_FIELDS


def test_retry_bound_stays_restart_only():
    assert 'max_pre_merge_retries' not in RELOADABLE_FIELDS


def test_edit_lands_in_applied_candidates_not_restart_required(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')
    diff = diff_config(
        OrchestratorConfig(pre_merge_reverify=False),
        OrchestratorConfig(pre_merge_reverify=True),
    )
    assert 'pre_merge_reverify' in diff.applied_candidates
    assert 'pre_merge_reverify' not in diff.restart_required


def test_apply_reload_flips_the_live_object_in_place(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')
    live = OrchestratorConfig()
    report = apply_reload(live, OrchestratorConfig(pre_merge_reverify=True))
    assert report['reloaded'] is True
    assert report['applied'] == {'pre_merge_reverify': {'old': False, 'new': True}}
    assert report['restart_required'] == {}
    assert live.pre_merge_reverify is True
