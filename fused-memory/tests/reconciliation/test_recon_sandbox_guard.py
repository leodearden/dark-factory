"""Tests for the reconciliation sandbox guard (task 1935).

Covers:
  - ReconciliationConfig sandbox field defaults (S3/S4)
  - sandbox_guard.resolve_recon_sandbox_wrap shape and behaviour (S5/S6)
  - fail-closed and bwrap-fallback paths (S7/S8)
"""

from __future__ import annotations

import socket
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fused_memory.config.schema import ReconciliationConfig


# ── Config defaults (S3 / S4) ─────────────────────────────────────────────────


class TestReconciliationConfigSandboxDefaults:

    def test_sandbox_recon_agents_defaults_true(self) -> None:
        """ReconciliationConfig defaults sandbox_recon_agents to True (fail-safe on)."""
        cfg = ReconciliationConfig()
        assert cfg.sandbox_recon_agents is True

    def test_sandbox_recon_writable_extras_defaults_empty(self) -> None:
        """ReconciliationConfig defaults sandbox_recon_writable_extras to []."""
        cfg = ReconciliationConfig()
        assert cfg.sandbox_recon_writable_extras == []

    def test_sandbox_fields_round_trip(self) -> None:
        """ReconciliationConfig(sandbox_recon_agents=False, sandbox_recon_writable_extras=['/x'])
        round-trips correctly."""
        cfg = ReconciliationConfig(sandbox_recon_agents=False, sandbox_recon_writable_extras=['/x'])
        assert cfg.sandbox_recon_agents is False
        assert cfg.sandbox_recon_writable_extras == ['/x']
