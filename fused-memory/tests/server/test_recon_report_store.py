"""Tests for SQLite-backed persistence of ReconReportState (task 2716).

Covers:
- Config schema field recon_report_persist_enabled (TestConfigSchema)
- ReconReportStore sync lifecycle (TestReconReportStore)
- _serialize_entry / _deserialize_entry round-trip (TestEntrySerialization)
- Write-through on every mutator (TestWriteThrough)
- hydrate_from_store readback after a simulated restart (TestHydrateReadback)
- Run-quiescence GC deletes a run's rows (TestStoreGC)
- Fresh in-process runs stay byte-identical with/without a store (TestByteIdenticalFreshRun)
- _build_recon_report_components / start_persistence / stop_persistence wiring
  (TestReconReportStoreWiredAtBoot)
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# step-1: Config schema — RED until step-2 adds the field
# ---------------------------------------------------------------------------


class TestConfigSchema:
    """Verify the new recon_report_persist_enabled config field."""

    def test_recon_report_persist_enabled_default_true(self):
        from fused_memory.config.schema import ReconciliationConfig

        cfg = ReconciliationConfig()
        assert cfg.recon_report_persist_enabled is True

    def test_recon_report_persist_enabled_roundtrips_false(self):
        from fused_memory.config.schema import ReconciliationConfig

        cfg = ReconciliationConfig(recon_report_persist_enabled=False)
        assert cfg.recon_report_persist_enabled is False
