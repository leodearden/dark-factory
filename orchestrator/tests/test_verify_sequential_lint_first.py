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

from orchestrator import verify
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


class TestResolveSequentialLintFirst:
    """test-resolver: verify._resolve_sequential_lint_first(config, module_config).

    RED today because the helper does not exist yet (AttributeError). Mirrors
    _resolve_concurrent_verify (verify.py:3219): module override wins over the
    top-level config value.
    """

    def test_config_only_returns_config_value_true(self, tmp_path: Path):
        """module_config=None → returns config.sequential_lint_first (True)."""
        config = OrchestratorConfig(project_root=tmp_path, sequential_lint_first=True)
        assert verify._resolve_sequential_lint_first(config, None) is True

    def test_config_only_returns_config_value_false(self, tmp_path: Path):
        """module_config=None → returns config.sequential_lint_first (False)."""
        config = OrchestratorConfig(project_root=tmp_path, sequential_lint_first=False)
        assert verify._resolve_sequential_lint_first(config, None) is False

    def test_module_override_wins_true_over_false(self, tmp_path: Path):
        """config False, module True → module override wins → True."""
        config = OrchestratorConfig(project_root=tmp_path, sequential_lint_first=False)
        mc = ModuleConfig(prefix='m', sequential_lint_first=True)
        assert verify._resolve_sequential_lint_first(config, mc) is True

    def test_module_override_wins_false_over_true(self, tmp_path: Path):
        """config True, module False → module override wins → False."""
        config = OrchestratorConfig(project_root=tmp_path, sequential_lint_first=True)
        mc = ModuleConfig(prefix='m', sequential_lint_first=False)
        assert verify._resolve_sequential_lint_first(config, mc) is False

    def test_module_none_falls_back_to_config(self, tmp_path: Path):
        """module_config.sequential_lint_first is None → fall back to config."""
        config = OrchestratorConfig(project_root=tmp_path, sequential_lint_first=True)
        mc = ModuleConfig(prefix='m')  # sequential_lint_first defaults to None
        assert mc.sequential_lint_first is None
        assert verify._resolve_sequential_lint_first(config, mc) is True
