"""T6: benchmark-tuned default `-n` for verify admission (task 2394; PRD
plans/verify-oversubscription-control-prd.md T6, follow-up to T2/task 2390).

Covers the new ``verify_admission_pytest_n`` knob end to end:

- step-2/3: config surface + green-tier reload registration (mirrors
  test_config_verify_admission_reload.py).
- step-4/5: the ``verify_cmd.apply_pytest_numprocesses`` mutator (mirrors
  test_verify_cmd.py::TestSerialPytest).
- step-6/7: wiring into ``orchestrator.verify._run_or_skip_timed`` — the
  test-leg-only, role-gated (task/background, not merge) ``-n`` rewrite
  (mirrors test_verify_admission_wiring.py).

The default (`'auto'`) is the T6 benchmark report's recommendation —
see plans/verify-oversubscription-benchmark-2026-07-14.md — not an
arbitrary placeholder; it is behavior-preserving (byte-identical to
pre-T6 `-n auto` addopts) because no clean idle-window measurement
supported a specific worker-count cap on this host.
"""

from __future__ import annotations

from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    apply_reload,
    diff_config,
)


class TestVerifyAdmissionPytestNConfigDefault:
    """Behavior-preserving default: 'auto' means no `-n` rewrite is injected
    (see the wiring tests below), so an untuned install keeps today's
    `-n auto` pyproject addopts byte-for-byte.
    """

    def test_default_is_str_auto(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig()
        assert isinstance(cfg.verify_admission_pytest_n, str)
        assert cfg.verify_admission_pytest_n == 'auto'


class TestVerifyAdmissionPytestNReloadDisposition:
    """Green-tier: hot-reloadable without a process restart, same tier as
    the other five verify_admission_* knobs.
    """

    def test_field_is_reloadable(self):
        assert 'verify_admission_pytest_n' in RELOADABLE_FIELDS, (
            "'verify_admission_pytest_n' is expected to be green-tier "
            'reloadable but is missing from RELOADABLE_FIELDS'
        )

    def test_edit_lands_in_applied_candidates_not_restart_required(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig(verify_admission_pytest_n='auto')
        fresh = OrchestratorConfig(verify_admission_pytest_n='16')
        diff = diff_config(live, fresh)
        assert 'verify_admission_pytest_n' in diff.applied_candidates
        assert 'verify_admission_pytest_n' not in diff.restart_required

    def test_apply_reload_applies_in_place(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig(verify_admission_pytest_n='auto')
        fresh = OrchestratorConfig(verify_admission_pytest_n='16')
        report = apply_reload(live, fresh)
        assert report['reloaded'] is True
        assert report['applied']['verify_admission_pytest_n'] == {'old': 'auto', 'new': '16'}
        assert live.verify_admission_pytest_n == '16'
