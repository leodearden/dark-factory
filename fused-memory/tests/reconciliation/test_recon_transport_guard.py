"""RED tests for transport-guard helper in server/main.py (step-25).

Three cases mandated by the plan:
  (a) reconciliation.enabled=True + transport='stdio'  → ValueError naming both
      'reconciliation' and 'http'
  (b) reconciliation.enabled=True + transport='sse'    → same
  (c) transport='http' (or reconciliation.enabled=False) → no raise
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from fused_memory.server.main import _require_http_transport_for_reconciliation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(transport: str, recon_enabled: bool):
    """Minimal fake config sufficient for the guard function."""
    cfg = MagicMock()
    cfg.server.transport = transport
    cfg.reconciliation = MagicMock()
    cfg.reconciliation.enabled = recon_enabled
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRequireHttpTransportForReconciliation:
    """_require_http_transport_for_reconciliation guard contract."""

    # --- cases that must raise ---

    def test_stdio_raises_valueerror(self):
        cfg = _make_cfg('stdio', recon_enabled=True)
        with pytest.raises(ValueError) as exc_info:
            _require_http_transport_for_reconciliation(cfg)
        msg = str(exc_info.value)
        assert 'reconciliation' in msg, f"Expected 'reconciliation' in message: {msg!r}"
        assert 'http' in msg, f"Expected 'http' in message: {msg!r}"

    def test_sse_raises_valueerror(self):
        cfg = _make_cfg('sse', recon_enabled=True)
        with pytest.raises(ValueError) as exc_info:
            _require_http_transport_for_reconciliation(cfg)
        msg = str(exc_info.value)
        assert 'reconciliation' in msg, f"Expected 'reconciliation' in message: {msg!r}"
        assert 'http' in msg, f"Expected 'http' in message: {msg!r}"

    # --- cases that must NOT raise ---

    def test_http_transport_passes(self):
        cfg = _make_cfg('http', recon_enabled=True)
        # Must not raise
        _require_http_transport_for_reconciliation(cfg)

    def test_reconciliation_disabled_passes_even_with_stdio(self):
        cfg = _make_cfg('stdio', recon_enabled=False)
        # reconciliation disabled → guard is a no-op regardless of transport
        _require_http_transport_for_reconciliation(cfg)

    def test_reconciliation_none_passes(self):
        cfg = MagicMock()
        cfg.server.transport = 'stdio'
        cfg.reconciliation = None
        # reconciliation not configured → guard is a no-op
        _require_http_transport_for_reconciliation(cfg)
