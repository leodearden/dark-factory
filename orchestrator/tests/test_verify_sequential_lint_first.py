"""Tests for the sequential_lint_first knob — task 2832.

A per-project module-config bool ``sequential_lint_first`` (default False)
that, when enabled AND ``concurrent_verify`` is False AND role=='merge',
makes the sequential verify branch run LINT FIRST and short-circuit on a
lint failure (record test+type as ``CheckRun.skipped``, fail the attempt) —
so a lint-only-red merge no longer burns the ~50-min test phase before the
~4-min lint phase. Task/background roles and the concurrent branch stay
byte-unchanged.

Test coverage:
  test-config-field:      new config field schema (ModuleConfig +
                          OrchestratorConfig + _OVERRIDABLE_FIELDS membership)
  test-resolver:          verify._resolve_sequential_lint_first
                          (module-override-wins)
  test-lint-first-branch: run_verification merge-role lint-first phase order
                          + lint-red short-circuit (mocked _run_cmd)
"""

from __future__ import annotations

from pathlib import Path

from orchestrator.config import (
    _OVERRIDABLE_FIELDS,
    ModuleConfig,
    OrchestratorConfig,
)


class TestSequentialLintFirstConfig:
    """test-config-field: the new config field's schema.

    RED today because ``sequential_lint_first`` does not yet exist on
    ModuleConfig / OrchestratorConfig / _OVERRIDABLE_FIELDS. Mirrors the
    ``concurrent_verify`` field triple exactly (config.py:1925, :1959, :2421).
    """

    def test_orchestrator_config_default_is_false(self, tmp_path: Path):
        """OrchestratorConfig default is opt-in False (byte-unchanged for
        existing configs — the branch never activates unless enabled)."""
        assert OrchestratorConfig(project_root=tmp_path).sequential_lint_first is False

    def test_orchestrator_config_accepts_true(self, tmp_path: Path):
        """OrchestratorConfig(sequential_lint_first=True) round-trips."""
        config = OrchestratorConfig(project_root=tmp_path, sequential_lint_first=True)
        assert config.sequential_lint_first is True

    def test_module_config_default_is_none(self):
        """ModuleConfig default is None (fall through to top-level config),
        mirroring concurrent_verify's bool|None override semantics."""
        assert ModuleConfig(prefix='x').sequential_lint_first is None

    def test_module_config_accepts_true(self):
        """ModuleConfig(sequential_lint_first=True) round-trips."""
        assert ModuleConfig(prefix='x', sequential_lint_first=True).sequential_lint_first is True

    def test_is_overridable_field(self):
        """Membership in _OVERRIDABLE_FIELDS makes per-module orchestrator.yaml
        discovery (config.py:2075 kwargs comprehension) load it with zero extra
        loader code — same as concurrent_verify."""
        assert 'sequential_lint_first' in _OVERRIDABLE_FIELDS
