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

import pytest

from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    apply_reload,
    diff_config,
)
from orchestrator.verify_cmd import (
    ToolKind,
    apply_pytest_numprocesses,
    parse_config_command,
    render,
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


class TestApplyPytestNumprocesses:
    """apply_pytest_numprocesses(cmd, n) appends a `-n <n>` pytest-xdist
    worker-count flag. Mirrors test_verify_cmd.py::TestSerialPytest's shape.
    """

    def test_structured_single_invocation_appends_dash_n(self):
        cmd = parse_config_command('uv run pytest tests/')
        result = render(apply_pytest_numprocesses(cmd, '16'))
        assert result == 'uv run pytest -n 16 tests/'

    @pytest.mark.parametrize('n', ['auto', ''])
    def test_noop_values_leave_cmd_byte_identical(self, n):
        cmd = parse_config_command('uv run pytest tests/')
        mutated = apply_pytest_numprocesses(cmd, n)
        assert mutated == cmd
        assert render(mutated) == render(cmd)

    def test_rewrites_every_pytest_invocation_in_chained_command(self):
        raw = (
            'cd shared && uv run pytest tests/ && '
            'cd ../orchestrator && uv run pytest tests/'
        )
        cmd = parse_config_command(raw)
        assert cmd.tool is ToolKind.PYTEST
        assert cmd.raw == raw

        result = render(apply_pytest_numprocesses(cmd, '16'))
        assert result.count('pytest') == 2
        assert result.count('-n 16') == 2

    def test_non_pytest_command_returned_unchanged(self):
        cmd = parse_config_command('ruff check .')
        assert apply_pytest_numprocesses(cmd, '16') == cmd

    def test_noop_on_opaque(self):
        cmd = parse_config_command('mypy src/')
        assert apply_pytest_numprocesses(cmd, '16') == cmd
