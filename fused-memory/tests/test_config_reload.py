"""Tests for fused-memory's green-tier config hot-reload engine + MCP tool.

Ports the orchestrator's proven config-hot-reload engine + tool contract
(orchestrator/config.py apply_reload/diff_config/_iter_leaves/_set_leaf and
escalation/server.py reload_config) into fused-memory, per task 2718 /
plans/fused-memory-restart-survey-2026-07-17.md task τ (finding A3).

The contract under test is the machine-checked disposition report
``{reloaded, config_path, applied, restart_required, unchanged, error}`` —
not prose.
"""

from __future__ import annotations

import pytest

from fused_memory.config.reload import RELOADABLE_FIELDS, _iter_leaves, diff_config
from fused_memory.config.schema import FusedMemoryConfig


@pytest.fixture(autouse=True)
def _pin_config_path(tmp_path, monkeypatch):
    """Pin CONFIG_PATH at a missing file so a bare ``FusedMemoryConfig()`` loads
    pure defaults deterministically (mirrors test_config_schema.py), independent
    of any ``config/config.yaml`` in the test cwd.

    Tool tests that need a real on-disk config re-point CONFIG_PATH themselves;
    a later ``monkeypatch.setenv`` in the test body wins over this default.
    """
    monkeypatch.setenv('CONFIG_PATH', str(tmp_path / 'missing.yaml'))


class TestDiffConfig:
    """The pure diff engine buckets every differing leaf by allowlist membership."""

    def test_reloadable_fields_are_all_real_leaves(self):
        """Every dotted path in RELOADABLE_FIELDS resolves to a real config leaf.

        Computes the leaf-path set from a freshly-built FusedMemoryConfig and
        asserts RELOADABLE_FIELDS is a subset — catches typos / stale paths.
        """
        leaf_paths = {path for path, _ in _iter_leaves(FusedMemoryConfig())}
        missing = RELOADABLE_FIELDS - leaf_paths
        assert not missing, f'RELOADABLE_FIELDS names non-existent leaves: {missing}'

    def test_allowlisted_leaf_lands_in_applied_candidates(self):
        """A differing ALLOWLISTED leaf lands in applied_candidates with {old,new}."""
        live = FusedMemoryConfig()
        fresh = FusedMemoryConfig()
        old = live.reconciliation.stale_run_recovery_seconds
        # Bypass validation on the submodel to stage a differing value.
        object.__setattr__(fresh.reconciliation, 'stale_run_recovery_seconds', old + 1)

        d = diff_config(live, fresh)

        assert d.applied_candidates['reconciliation.stale_run_recovery_seconds'] == {
            'old': old,
            'new': old + 1,
        }
        assert 'reconciliation.stale_run_recovery_seconds' not in d.restart_required

    def test_non_allowlisted_leaf_lands_in_restart_required(self):
        """A differing NON-allowlisted leaf lands in restart_required."""
        live = FusedMemoryConfig()
        fresh = FusedMemoryConfig()
        old = live.task_metadata.enforce
        object.__setattr__(fresh.task_metadata, 'enforce', not old)

        d = diff_config(live, fresh)

        assert d.restart_required['task_metadata.enforce'] == {'old': old, 'new': not old}
        assert 'task_metadata.enforce' not in d.applied_candidates

    def test_equal_leaves_tallied_unchanged(self):
        """Leaves equal between the two configs are counted in ``unchanged``."""
        live = FusedMemoryConfig()
        fresh = FusedMemoryConfig()

        d = diff_config(live, fresh)

        # Two identical default configs: nothing bucketed, everything unchanged.
        assert d.applied_candidates == {}
        assert d.restart_required == {}
        assert d.unchanged > 0
