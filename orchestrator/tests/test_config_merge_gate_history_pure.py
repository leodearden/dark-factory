"""The same-tree gate keys: ``merge_gate_history_pure`` and its history leg (tasks 6647, 7248).

Fixtures are kept module-local for the reason given in
test_config_disjoint_skip_reload.py.
"""

from __future__ import annotations

from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    apply_reload,
    diff_config,
)


def test_code_default_is_off(monkeypatch, tmp_path):
    """Undeclared means the gate may read history, so an identical tree is re-verified."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')
    assert OrchestratorConfig().merge_gate_history_pure is False


def test_is_green_tier_and_applies_in_place(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')
    assert 'merge_gate_history_pure' in RELOADABLE_FIELDS
    live = OrchestratorConfig(merge_gate_history_pure=False)
    fresh = OrchestratorConfig(merge_gate_history_pure=True)
    diff = diff_config(live, fresh)
    assert 'merge_gate_history_pure' in diff.applied_candidates
    assert 'merge_gate_history_pure' not in diff.restart_required
    report = apply_reload(live, fresh)
    assert report['applied']['merge_gate_history_pure'] == {'old': False, 'new': True}
    assert live.merge_gate_history_pure is True


def test_history_leg_code_default_is_unset(monkeypatch, tmp_path):
    """Undeclared means no history leg, so a same-tree rebase under history_pure skips."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')
    assert OrchestratorConfig().merge_gate_history_test_command is None


def test_history_leg_is_green_tier_and_applies_in_place(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')
    leg = 'pytest -m reads_git_history'
    assert 'merge_gate_history_test_command' in RELOADABLE_FIELDS
    live = OrchestratorConfig(merge_gate_history_test_command=None)
    fresh = OrchestratorConfig(merge_gate_history_test_command=leg)
    diff = diff_config(live, fresh)
    assert 'merge_gate_history_test_command' in diff.applied_candidates
    assert 'merge_gate_history_test_command' not in diff.restart_required
    report = apply_reload(live, fresh)
    assert report['applied']['merge_gate_history_test_command'] == {'old': None, 'new': leg}
    assert live.merge_gate_history_test_command == leg
