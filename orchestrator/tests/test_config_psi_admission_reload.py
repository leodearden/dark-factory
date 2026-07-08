"""L3b-DA2: PsiAdmissionConfig submodel + green-tier reload registration
(task 2327; PRD docs/prds/dispatch-admission-load-cap.md).

Fixtures are kept MODULE-LOCAL (not conftest.py) — a conftest.py edit trips
verify.py's has_conftest and forces the merge-time verify to fall back to
running the full owning-package suite instead of a scoped subset (mirrors
test_config_reload_integration_gate.py's stated rationale).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    PsiAdmissionConfig,
    apply_reload,
    diff_config,
)


class TestPsiAdmissionConfigDefaults:
    """DA-D1/DA-D7 defaults: enabled by default, conservative thresholds,
    memory ranked tighter than io.
    """

    def test_defaults_attached_on_orchestrator_config(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig()
        assert isinstance(cfg.psi_admission, PsiAdmissionConfig)
        assert cfg.psi_admission.enabled is True
        assert cfg.psi_admission.cpu_some_avg10 == 85.0
        assert cfg.psi_admission.mem_some_avg10 == 15.0
        assert cfg.psi_admission.mem_full_avg10 == 3.0
        assert cfg.psi_admission.io_some_avg10 == 40.0
        assert cfg.psi_admission.min_inflight_floor == 1

    def test_memory_threshold_ranked_tighter_than_io(self):
        """DA-D1: memory ranks above io, so its default threshold trips first."""
        cfg = PsiAdmissionConfig()
        assert cfg.mem_some_avg10 < cfg.io_some_avg10
