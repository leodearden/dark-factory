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

import types

import pytest

from fused_memory.config.reload import (
    RELOADABLE_FIELDS,
    _iter_leaves,
    apply_reload,
    diff_config,
)
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.server.near_duplicate_guard import (
    resolve_near_dup_guard_enabled,
    resolve_near_dup_threshold,
)


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


class TestApplyReload:
    """apply_reload applies allowlisted leaves in place with hybrid re-validation."""

    def test_happy_applies_green_leaf_and_flags_red_leaf(self):
        """A green leaf is applied to ``live`` in place; a red leaf is reported
        restart_required and left untouched — both dispositions in one call."""
        live = FusedMemoryConfig()
        fresh = FusedMemoryConfig()
        old_green = live.reconciliation.stale_run_recovery_seconds
        old_red = live.task_metadata.enforce
        object.__setattr__(fresh.reconciliation, 'stale_run_recovery_seconds', old_green + 5)
        object.__setattr__(fresh.task_metadata, 'enforce', not old_red)

        report = apply_reload(live, fresh)

        assert report['reloaded'] is True
        assert report['applied'] == {
            'reconciliation.stale_run_recovery_seconds': {
                'old': old_green,
                'new': old_green + 5,
            }
        }
        assert report['restart_required'] == {
            'task_metadata.enforce': {'old': old_red, 'new': not old_red}
        }
        assert report['error'] is None
        assert isinstance(report['unchanged'], int) and report['unchanged'] > 0
        # Live green leaf mutated; live red leaf NOT mutated.
        assert live.reconciliation.stale_run_recovery_seconds == old_green + 5
        assert live.task_metadata.enforce == old_red

    def test_reconciliation_submodel_identity_preserved(self):
        """The live.reconciliation object is the SAME instance before and after
        apply, so held references (e.g. ReconciliationHarness.config) observe the
        in-place mutation with no reconstruction."""
        live = FusedMemoryConfig()
        fresh = FusedMemoryConfig()
        old_green = live.reconciliation.stale_run_recovery_seconds
        object.__setattr__(fresh.reconciliation, 'stale_run_recovery_seconds', old_green + 1)
        recon_before = live.reconciliation

        apply_reload(live, fresh)

        assert live.reconciliation is recon_before
        assert recon_before.stale_run_recovery_seconds == old_green + 1

    def test_rollback_on_hybrid_invariant(self):
        """A fresh whose applied leaf violates a field invariant only after the
        whole-config re-validation triggers a synchronous full rollback: the
        report is a fail-closed error and ``live`` is left byte-identical."""
        live = FusedMemoryConfig()
        fresh = FusedMemoryConfig()
        old_green = live.reconciliation.stale_run_recovery_seconds
        # 0 violates the field's gt=0, but only surfaces at whole-config
        # re-validation (object.__setattr__ bypasses per-write validation).
        object.__setattr__(fresh.reconciliation, 'stale_run_recovery_seconds', 0)

        report = apply_reload(live, fresh)

        assert report['reloaded'] is False
        assert report['applied'] == {}
        assert isinstance(report['error'], str)
        assert report['error'].startswith('hybrid-invariant')
        # Flagship leaf restored — live untouched after rollback.
        assert live.reconciliation.stale_run_recovery_seconds == old_green


class TestBehaviorChangesWithoutRestart:
    """The user-observable signal: a live consumer's behavior changes after an
    in-place apply_reload, with no reconstruction — proven on the real near-dup
    guard resolvers that read ``memory_service.config.reconciliation.*`` live."""

    def test_guard_enabled_flip_observed_by_live_resolver(self):
        memory_service = types.SimpleNamespace(config=FusedMemoryConfig())
        # Baseline: default True, and the live resolver agrees.
        assert (
            memory_service.config.reconciliation.procedural_knowledge_near_dup_guard_enabled
            is True
        )
        assert resolve_near_dup_guard_enabled(memory_service) is True

        fresh = FusedMemoryConfig()
        object.__setattr__(
            fresh.reconciliation, 'procedural_knowledge_near_dup_guard_enabled', False
        )

        report = apply_reload(memory_service.config, fresh)

        assert 'reconciliation.procedural_knowledge_near_dup_guard_enabled' in report['applied']
        # The live resolver observes the in-place mutation without a restart.
        assert resolve_near_dup_guard_enabled(memory_service) is False

    def test_threshold_flip_observed_by_live_resolver(self):
        memory_service = types.SimpleNamespace(config=FusedMemoryConfig())
        old = memory_service.config.reconciliation.procedural_knowledge_near_dup_threshold
        assert resolve_near_dup_threshold(memory_service) == old

        new_val = 0.5 if old != 0.5 else 0.7
        fresh = FusedMemoryConfig()
        object.__setattr__(
            fresh.reconciliation, 'procedural_knowledge_near_dup_threshold', new_val
        )

        report = apply_reload(memory_service.config, fresh)

        assert 'reconciliation.procedural_knowledge_near_dup_threshold' in report['applied']
        assert resolve_near_dup_threshold(memory_service) == new_val
