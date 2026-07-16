"""Tests for CuratorConfig + TaskCurator wiring of the recon claim-verification
guard (task 2438).

Mirrors TestCuratorReconPremiseRegistryConfig (test_task_curator.py) for the
config field, and _maybe_premise_refuted_drop's curator-hook tests for the
TaskCurator._maybe_flag_unverified_claims wiring — same shapes, new
(advisory, not a drop) guard. See recon_claim_verification_guard.py's module
docstring for the motivating task-2433 incident.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# task-2438 step-07 RED: TestCuratorReconClaimVerificationConfig
# ──────────────────────────────────────────────────────────────────────────────


class TestCuratorReconClaimVerificationConfig:
    """Tests that CuratorConfig has recon_claim_verification_enabled field,
    defaulting to False, and that FusedMemoryConfig round-trips it via YAML.

    Mirrors TestCuratorReconPremiseRegistryConfig in test_task_curator.py —
    same shape, new field.
    """

    def test_curator_config_has_field_default_false(self):
        """CuratorConfig has recon_claim_verification_enabled field, default False."""
        from fused_memory.config.schema import CuratorConfig

        cfg = CuratorConfig()
        assert hasattr(cfg, "recon_claim_verification_enabled")
        assert cfg.recon_claim_verification_enabled is False

    def test_curator_config_accepts_true(self):
        """CuratorConfig accepts recon_claim_verification_enabled=True."""
        from fused_memory.config.schema import CuratorConfig

        cfg = CuratorConfig(recon_claim_verification_enabled=True)
        assert cfg.recon_claim_verification_enabled is True

    def test_fused_memory_config_roundtrips_via_yaml(self, tmp_path, monkeypatch):
        """FusedMemoryConfig round-trips recon_claim_verification_enabled via YAML."""
        import yaml

        from fused_memory.config.schema import FusedMemoryConfig

        raw = {"curator": {"recon_claim_verification_enabled": True}}
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(raw), encoding="utf-8")

        monkeypatch.setenv("CONFIG_PATH", str(yaml_path))
        cfg = FusedMemoryConfig()
        assert cfg.curator.recon_claim_verification_enabled is True
