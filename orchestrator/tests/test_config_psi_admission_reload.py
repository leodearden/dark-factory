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


class TestMinInflightFloorValidation:
    """DA-D3 anti-deadlock floor: a min_inflight_floor < 1 would let the gate
    hold with nothing in flight and wedge the queue, so it must be rejected
    at construction/load.
    """

    @pytest.mark.parametrize('bad_value', [0, -1])
    def test_sub_minimum_rejected(self, bad_value):
        with pytest.raises(ValidationError):
            PsiAdmissionConfig(min_inflight_floor=bad_value)

    @pytest.mark.parametrize('good_value', [1, 2])
    def test_valid_values_accepted(self, good_value):
        cfg = PsiAdmissionConfig(min_inflight_floor=good_value)
        assert cfg.min_inflight_floor == good_value

    def test_rejected_at_load_via_orchestrator_config(self, monkeypatch, tmp_path):
        """Covers the 'rejected at load' signal: load_config constructs
        OrchestratorConfig, which builds the psi_admission submodel from a
        nested dict (as YAML would deserialize into) and must raise.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        with pytest.raises(ValidationError):
            OrchestratorConfig(psi_admission={'min_inflight_floor': 0})
