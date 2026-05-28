"""Tests for the recon_report MCP namespace scaffold (task α).

Covers:
- Config schema fields (TestConfigSchema)
- ReconReportState happy path (TestReconReportStateHappyPath)
- In-run dedup (TestReconReportInRunDedup)
- complete() idempotence (TestReconReportCompleteIdempotence)
- Unknown run_id errors (TestReconReportRunIdMismatch)
- State isolation between runs (TestReconReportStateIsolation)
- TTL reaper (TestReconReportReaper)
- FastMCP server factory (TestCreateReconReportServer)
- Main.py boot helper (TestReconReportBoot)
- Reaper not started by _build_recon_report_components (TestReconReportReaperWiredAtBoot)
"""

import pytest


# ---------------------------------------------------------------------------
# step-1: Config schema — RED until step-2 adds the fields
# ---------------------------------------------------------------------------


class TestConfigSchema:
    """Verify the two new config fields exist with correct defaults."""

    def test_server_config_recon_report_port_default(self):
        from fused_memory.config.schema import ServerConfig

        cfg = ServerConfig()
        assert cfg.recon_report_port == 8003

    def test_reconciliation_config_ttl_default(self):
        from fused_memory.config.schema import ReconciliationConfig

        cfg = ReconciliationConfig()
        assert cfg.recon_report_state_ttl_seconds == 300
