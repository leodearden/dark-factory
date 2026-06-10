"""Tests for SpeculativeMergeWorker retroactive coalescing pass (task γ/1719).

Each TestClass maps to one plan step:
  step-1:  TestConfigAndEventType   — new config knob + new EventType
  step-3:  TestNoOpGuards           — early-return guards of _maybe_coalesce_waiting_singles
  step-5:  TestCoreFormation        — happy-path coalesce: 3 disjoint singles → 1 train
  step-7:  TestExclusionIdempotency — in-flight, detached, and GroupMergeRequest exclusions
  step-9:  TestPartialStackability  — overlap, stack-conflict eject, survivors<2
  step-11: TestDebounce             — signature-based deduplication
  step-13: TestEndToEndWiring       — merger-loop wiring: coalesced train dispatched end-to-end
"""

from __future__ import annotations

import asyncio
import collections
import sys
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import make_placeholder_future

# ─── Step 1 ─────────────────────────────────────────────────────────────────

class TestConfigAndEventType:
    """step-1 (RED): new config knob and EventType member exist with correct defaults."""

    def test_merge_train_coalesce_enabled_default_false(self):
        """OrchestratorConfig.merge_train_coalesce_enabled defaults to False (OFF by default,
        human-flips after soak — fold-the-decision norm).
        """
        from orchestrator.config import GitConfig, OrchestratorConfig
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cfg = OrchestratorConfig(
                project_root=Path(tmp),
                git=GitConfig(
                    main_branch='main',
                    branch_prefix='task/',
                    remote='origin',
                    worktree_dir='.worktrees',
                ),
            )
            assert cfg.merge_train_coalesce_enabled is False

    def test_event_type_train_coalesced(self):
        """EventType.train_coalesced has value 'train_coalesced'."""
        from orchestrator.event_store import EventType
        assert EventType.train_coalesced == 'train_coalesced'
