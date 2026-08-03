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

import asyncio
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import yaml

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
from fused_memory.server.tools import create_mcp_server


def _edit_yaml(path: Path, dotted_leaf: str, value: Any) -> None:
    """Rewrite one dotted leaf on the YAML file at *path*, preserving the rest.

    Ported from escalation/tests/test_reload_config_integration.py._edit_yaml.
    """
    data = yaml.safe_load(path.read_text()) or {}
    node = data
    parts = dotted_leaf.split('.')
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    path.write_text(yaml.safe_dump(data))


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

    def test_topic_guard_clusters_leaf_is_green_tier_applied_candidate(self):
        """The topic-guard clusters leaf is green-tier (task 2845): a changed value
        buckets as an applied_candidate, NOT restart_required — the whole
        list[ProceduralTopicCluster] is compared as ONE atomic leaf, mirroring the
        sibling near-dup knobs' hot-reload coverage."""
        path = 'reconciliation.procedural_knowledge_topic_guard_clusters'
        assert path in RELOADABLE_FIELDS, f'{path} must be allowlisted for hot-reload'

        live = FusedMemoryConfig()
        fresh = FusedMemoryConfig()
        old = live.reconciliation.procedural_knowledge_topic_guard_clusters
        assert old, 'default must seed a non-empty clusters list'
        # Operator clears the clusters -> a differing value on the same leaf.
        object.__setattr__(fresh.reconciliation, 'procedural_knowledge_topic_guard_clusters', [])

        d = diff_config(live, fresh)

        assert path in d.applied_candidates
        assert d.applied_candidates[path] == {'old': old, 'new': []}
        assert path not in d.restart_required


class TestDiffConfigOptionalSubmodels:
    """diff_config / apply_reload tolerate an OPTIONAL submodel field toggling
    between None and populated across a reload — the structural-asymmetry the
    reviewer flagged (task 2718 review esc-2718-1).

    ``taskmaster: TaskmasterConfig | None`` (default None) and
    ``usage_cap: UsageCapConfig | None`` (default a populated submodel) are the
    two nullable submodels on FusedMemoryConfig. Because ``_iter_leaves`` decides
    descent from the DECLARED annotation (descend only into a bare, non-Optional
    BaseModel), a nullable submodel is always compared WHOLE — so ``_iter_leaves``
    yields identical leaf-path sets for live and fresh regardless of nullability
    state, and a None<->populated toggle produces exactly ONE whole-object
    restart_required entry rather than a KeyError or a scatter of half-missing
    ``name.<sub>`` sub-paths.
    """

    def test_opposite_toggles_bucket_whole_without_crash(self):
        from fused_memory.config.schema import TaskmasterConfig

        live = FusedMemoryConfig()
        fresh = FusedMemoryConfig()
        assert live.taskmaster is None  # default
        old_usage_cap = live.usage_cap
        assert old_usage_cap is not None  # default-populated submodel

        # Toggle BOTH optional submodels in OPPOSITE directions in one diff, so a
        # leaf-path-set divergence is forced from both sides at once.
        new_tm = TaskmasterConfig()
        object.__setattr__(fresh, 'taskmaster', new_tm)  # None -> populated
        object.__setattr__(fresh, 'usage_cap', None)  # populated -> None

        # (a) NO-CRASH: the structural asymmetry no longer raises KeyError.
        d = diff_config(live, fresh)

        # (b) BOTH bucketed under restart_required as WHOLE atomic leaves.
        assert d.restart_required['taskmaster'] == {'old': None, 'new': new_tm}
        assert d.restart_required['usage_cap'] == {'old': old_usage_cap, 'new': None}
        # (c) Neither optional-submodel path is an applied candidate (restart-only).
        assert 'taskmaster' not in d.applied_candidates
        assert 'usage_cap' not in d.applied_candidates
        # (d) NO descended optional sub-paths leak as keys — compared whole.
        all_keys = set(d.restart_required) | set(d.applied_candidates)
        assert not any(
            k.startswith('taskmaster.') or k.startswith('usage_cap.') for k in all_keys
        ), f'optional submodel sub-paths leaked: {sorted(all_keys)}'

        # (e) END-TO-END: apply_reload reports both under restart_required, applies
        # nothing (neither is allowlisted), and leaves ``live`` untouched.
        report = apply_reload(live, fresh)
        assert report['reloaded'] is True
        assert report['applied'] == {}
        assert report['error'] is None
        assert 'taskmaster' in report['restart_required']
        assert 'usage_cap' in report['restart_required']
        assert live.taskmaster is None
        assert live.usage_cap is old_usage_cap


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


class TestReloadConfigTool:
    """The reload_config MCP tool re-reads the process's own CONFIG_PATH and
    returns the machine-checked disposition report, mutating only green leaves."""

    @pytest.mark.asyncio
    async def test_tool_applies_green_leaf_and_flags_red_leaf(self, tmp_path, monkeypatch):
        config_yaml = tmp_path / 'config.yaml'
        # Baseline on-disk config: an explicit green leaf; red leaf left default.
        config_yaml.write_text(
            yaml.safe_dump({'reconciliation': {'stale_run_recovery_seconds': 1800}})
        )
        monkeypatch.setenv('CONFIG_PATH', str(config_yaml))

        svc = AsyncMock()
        svc.config = FusedMemoryConfig()
        old_green = svc.config.reconciliation.stale_run_recovery_seconds
        assert svc.config.task_metadata.enforce is False  # baseline red leaf

        # Edit one green (allowlisted) + one red (restart-only) leaf on disk.
        _edit_yaml(config_yaml, 'reconciliation.stale_run_recovery_seconds', old_green + 1)
        _edit_yaml(config_yaml, 'task_metadata.enforce', True)

        server = create_mcp_server(svc)
        result = await server._tool_manager.call_tool('reload_config', {})

        assert result['reloaded'] is True
        assert result['config_path'] == str(config_yaml)
        assert result['applied'] == {
            'reconciliation.stale_run_recovery_seconds': {'old': old_green, 'new': old_green + 1}
        }
        assert result['restart_required'] == {
            'task_metadata.enforce': {'old': False, 'new': True}
        }
        assert result['error'] is None
        # Live config: green leaf mutated in place; red leaf NOT mutated.
        assert svc.config.reconciliation.stale_run_recovery_seconds == old_green + 1
        assert svc.config.task_metadata.enforce is False

    @pytest.mark.asyncio
    async def test_tool_fails_closed_on_invalid_config_reread(self, tmp_path, monkeypatch):
        """A CONFIG_PATH that re-reads as invalid yields a fail-closed report with
        the live config left completely untouched — never a half-applied reload."""
        # Build a VALID live config first (the autouse fixture points CONFIG_PATH
        # at a missing file → pure valid defaults), then capture the flagship leaf.
        svc = AsyncMock()
        svc.config = FusedMemoryConfig()
        flagship_before = svc.config.reconciliation.stale_run_recovery_seconds

        # Now point CONFIG_PATH at a yaml that RE-READS as invalid (gt=0 violated).
        bad_yaml = tmp_path / 'bad.yaml'
        bad_yaml.write_text(
            yaml.safe_dump({'reconciliation': {'stale_run_recovery_seconds': 0}})
        )
        monkeypatch.setenv('CONFIG_PATH', str(bad_yaml))

        server = create_mcp_server(svc)
        result = await server._tool_manager.call_tool('reload_config', {})

        assert result['reloaded'] is False
        assert isinstance(result['error'], str) and result['error']
        assert result['applied'] == {}
        assert result['config_path'] == str(bad_yaml)
        # Live config completely untouched — no apply on the load-failure path.
        assert svc.config.reconciliation.stale_run_recovery_seconds == flagship_before

    @pytest.mark.asyncio
    async def test_tool_fails_closed_on_load_timeout(self, monkeypatch):
        """A slow / hanging config re-read is bounded by ``asyncio.wait_for``: the
        tool returns a fail-closed 'timed out' report with the live config left
        untouched — never blocking the event loop or half-applying a reload.

        Exercises the REAL ``asyncio.wait_for`` timeout branch (distinct from the
        invalid-config ``except Exception`` branch above) by making the thread-off
        re-read hang and shrinking the tool's timeout so the bound trips fast and
        deterministically.
        """
        svc = AsyncMock()
        svc.config = FusedMemoryConfig()
        flagship_before = svc.config.reconciliation.stale_run_recovery_seconds

        # Replace the thread-off load with a hang (no real thread is spawned — the
        # coroutine is cancelled cleanly by wait_for), and shrink the tool's
        # timeout so the real asyncio.wait_for in reload_config trips quickly.
        async def _hang(*args, **kwargs):
            await asyncio.sleep(3600)

        monkeypatch.setattr(asyncio, 'to_thread', _hang)
        monkeypatch.setattr('fused_memory.server.tools._RELOAD_LOAD_TIMEOUT_SECS', 0.05)

        server = create_mcp_server(svc)
        # Outer guard: if the internal wait_for were ever removed (regression),
        # fail loudly here rather than hanging the whole test suite.
        result = await asyncio.wait_for(
            server._tool_manager.call_tool('reload_config', {}), timeout=5.0
        )

        assert result['reloaded'] is False
        assert result['applied'] == {}
        assert result['restart_required'] == {}
        assert isinstance(result['error'], str) and 'timed out' in result['error']
        # Live config completely untouched — no apply on the timeout path.
        assert svc.config.reconciliation.stale_run_recovery_seconds == flagship_before


class TestMem0UpdateLeavesAreGreenTier:
    """All five mem0_update.* leaves must hot-apply (task 3088).

    Modelled on TestWriteTriageLeavesAreGreenTier below — the direct precedent
    for registering TOP-LEVEL (non-reconciliation.*) leaves. The existing
    test_reloadable_fields_are_all_real_leaves guards these paths against typos
    automatically.

    The kill switch is the load-bearing one: mem0_update.enabled is what an
    operator flips to stop an in-flight rewrite incident, and a restart-only
    kill switch is no kill switch. The two storm leaves are only genuinely
    reload-safe because StormCounter takes threshold/window per record() call.
    """

    PATHS = (
        'mem0_update.enabled',
        'mem0_update.content_amend_allowed_agent_prefixes',
        'mem0_update.metadata_patch_allowed_agent_prefixes',
        'mem0_update.storm_threshold',
        'mem0_update.storm_window_seconds',
    )

    @pytest.mark.parametrize('path', PATHS)
    def test_leaf_is_allowlisted(self, path):
        assert path in RELOADABLE_FIELDS, f'{path} must be allowlisted for hot-reload'

    @pytest.mark.parametrize(
        ('path', 'new_value'),
        [
            ('mem0_update.enabled', False),
            ('mem0_update.content_amend_allowed_agent_prefixes', []),
            (
                'mem0_update.metadata_patch_allowed_agent_prefixes',
                ['recon-stage-', 'curator-'],
            ),
            ('mem0_update.storm_threshold', 5),
            ('mem0_update.storm_window_seconds', 600.0),
        ],
    )
    def test_changed_leaf_lands_in_applied_candidates(self, path, new_value):
        live = FusedMemoryConfig()
        fresh = FusedMemoryConfig()
        field = path.split('.', 1)[1]
        old = getattr(live.mem0_update, field)
        object.__setattr__(fresh.mem0_update, field, new_value)

        d = diff_config(live, fresh)

        assert path in d.applied_candidates, (
            f'{path} must hot-apply so an operator can retune without a restart'
        )
        assert d.applied_candidates[path] == {'old': old, 'new': new_value}
        assert path not in d.restart_required

    def test_kill_switch_flip_is_observed_live_without_a_restart(self):
        """A reload must be visible to the resolver through the SHARED config
        object — the precondition config/reload.py's reload-safety rule states
        before a leaf may be registered at all."""
        from types import SimpleNamespace

        from fused_memory.server.mem0_update_authz import resolve_mem0_update_authorization

        live = FusedMemoryConfig()
        svc = SimpleNamespace(config=live)
        assert resolve_mem0_update_authorization(
            svc, agent_id='recon-stage-1', content_amend=True, metadata_patch=False,
        ).allowed is True

        fresh = FusedMemoryConfig()
        object.__setattr__(fresh.mem0_update, 'enabled', False)
        result = apply_reload(live, fresh)

        assert result['reloaded'] is True, f'reload failed: {result.get("error")!r}'
        assert 'mem0_update.enabled' in result['applied']
        assert resolve_mem0_update_authorization(
            svc, agent_id='recon-stage-1', content_amend=True, metadata_patch=False,
        ).allowed is False, (
            'the resolver reads the same shared config object apply_reload '
            'mutated in place, so the flip takes effect with no restart'
        )

    def test_widened_metadata_bar_is_observed_live(self):
        """The operator story: admit a curator-gate metadata patch on a running
        server WITHOUT granting content-amend authority."""
        from types import SimpleNamespace

        from fused_memory.server.mem0_update_authz import resolve_mem0_update_authorization

        live = FusedMemoryConfig()
        svc = SimpleNamespace(config=live)

        fresh = FusedMemoryConfig()
        object.__setattr__(
            fresh.mem0_update,
            'metadata_patch_allowed_agent_prefixes',
            ['recon-stage-', 'curator-'],
        )
        apply_reload(live, fresh)

        assert resolve_mem0_update_authorization(
            svc, agent_id='curator-gate', content_amend=False, metadata_patch=True,
        ).allowed is True
        assert resolve_mem0_update_authorization(
            svc, agent_id='curator-gate', content_amend=True, metadata_patch=False,
        ).allowed is False, 'widening one bar must not widen the other'


class TestWriteTriageLeavesAreGreenTier:
    """The calibration script's config write must be hot-reloadable.

    Modelled on test_topic_guard_clusters_leaf_is_green_tier_applied_candidate.
    The existing test_reloadable_fields_are_all_real_leaves guards these
    paths against typos automatically.
    """

    PATHS = (
        'write_triage.t_high',
        'write_triage.t_low',
        'write_triage.calibration_report_path',
    )

    @pytest.mark.parametrize('path', PATHS)
    def test_leaf_is_allowlisted(self, path):
        assert path in RELOADABLE_FIELDS, f'{path} must be allowlisted for hot-reload'

    @pytest.mark.parametrize(
        ('path', 'new_value'),
        [
            ('write_triage.t_high', 0.87),
            ('write_triage.t_low', 0.61),
            ('write_triage.calibration_report_path', 'calibration/report.json'),
        ],
    )
    def test_changed_leaf_lands_in_applied_candidates(self, path, new_value):
        live = FusedMemoryConfig()
        fresh = FusedMemoryConfig()
        field = path.split('.', 1)[1]
        old = getattr(live.write_triage, field)
        assert old is None, 'the uncalibrated default is None'
        object.__setattr__(fresh.write_triage, field, new_value)

        d = diff_config(live, fresh)

        assert path in d.applied_candidates, (
            f'{path} must hot-apply so a calibration run is picked up without a restart'
        )
        assert d.applied_candidates[path] == {'old': old, 'new': new_value}
        assert path not in d.restart_required
