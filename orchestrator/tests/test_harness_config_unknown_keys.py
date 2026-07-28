"""Tests for Harness._file_config_unknown_keys_escalation (task 2989).

The startup filer surfaces the unknown-config-key census (config.py) as a
born-at-L2 escalation so a phantom key like the 2026-07-22 top-level
``spare_warm_lanes: 8`` (the field lives on ``git.``) can never again be
silently dropped for weeks.  It mirrors ``_file_dirty_tree_escalation``:
None-safe on the queue, self-closing on a clean census, and fail-open.

Signature dedup (via ``find_pending_l2_by_root_cause`` on a root_cause that
encodes the unknown-key-set signature) means an identical key-set files exactly
one L2 (storm escape) while a changed set re-files a distinct L2.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from _orch_helpers import _init_harness_state_for_test
from escalation.models import BORN_AT_L2_SEVERITIES
from escalation.queue import EscalationQueue
from escalation.server import CATEGORIES

from orchestrator.config import (
    ConfigIgnoredKey,
    ConfigUnknownKey,
    OrchestratorConfig,
    census_config_keys,
)
from orchestrator.harness import Harness


def _make_harness(tmp_path, monkeypatch, census, queue):
    """Bare Harness (Harness.__new__ + _init_harness_state_for_test) with a real
    config whose unknown_key_census is set directly and the given queue."""
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')
    monkeypatch.chdir(tmp_path)
    h = Harness.__new__(Harness)
    _init_harness_state_for_test(h)
    config = OrchestratorConfig(project_root=tmp_path)
    config._unknown_key_census = list(census)
    h.config = config
    h._escalation_queue = queue
    h.event_store = None
    h._run_id = 'run-test'
    return h


def _pending_config_l2s(queue):
    return [
        e
        for e in queue.get_pending()
        if e.level == 2 and e.root_cause.startswith('config_unknown_keys:')
    ]


@pytest.mark.asyncio
async def test_files_born_at_l2_for_unknown_keys(tmp_path: Path, monkeypatch):
    queue = EscalationQueue(tmp_path / 'esc')
    census = [ConfigUnknownKey('spare_warm_lanes', 'git.spare_warm_lanes')]
    h = _make_harness(tmp_path, monkeypatch, census, queue)

    await h._file_config_unknown_keys_escalation()

    l2s = _pending_config_l2s(queue)
    assert len(l2s) == 1
    esc = l2s[0]
    # Born-at-L2 CONTRACT (assert the contract, not exact string literals).
    assert esc.level == 2
    assert esc.severity in BORN_AT_L2_SEVERITIES
    assert esc.category in CATEGORIES
    assert esc.root_cause.startswith('config_unknown_keys:')
    assert esc.agent_role.startswith(('harness-', 'orchestrator-'))
    # Detail names both the bogus path and its placement hint.
    assert 'spare_warm_lanes' in esc.detail
    assert 'git.spare_warm_lanes' in esc.detail


@pytest.mark.asyncio
async def test_same_key_set_dedups_to_one(tmp_path: Path, monkeypatch):
    queue = EscalationQueue(tmp_path / 'esc')
    census = [ConfigUnknownKey('spare_warm_lanes', 'git.spare_warm_lanes')]
    h = _make_harness(tmp_path, monkeypatch, census, queue)

    await h._file_config_unknown_keys_escalation()
    await h._file_config_unknown_keys_escalation()

    assert len(_pending_config_l2s(queue)) == 1, 'identical key-set must file exactly one L2'


@pytest.mark.asyncio
async def test_changed_key_set_files_second_distinct_l2(tmp_path: Path, monkeypatch):
    queue = EscalationQueue(tmp_path / 'esc')
    h = _make_harness(
        tmp_path, monkeypatch,
        [ConfigUnknownKey('spare_warm_lanes', 'git.spare_warm_lanes')],
        queue,
    )
    await h._file_config_unknown_keys_escalation()

    # A different unknown-key set (e.g. operator fixed one key, introduced another).
    h.config._unknown_key_census = [ConfigUnknownKey('other_bogus', None)]
    await h._file_config_unknown_keys_escalation()

    l2s = _pending_config_l2s(queue)
    assert len(l2s) == 2, 'a changed key-set must re-file a distinct L2'
    assert len({e.root_cause for e in l2s}) == 2


@pytest.mark.asyncio
async def test_empty_census_self_heals_pending_l2(tmp_path: Path, monkeypatch):
    queue = EscalationQueue(tmp_path / 'esc')
    h = _make_harness(
        tmp_path, monkeypatch,
        [ConfigUnknownKey('spare_warm_lanes', 'git.spare_warm_lanes')],
        queue,
    )
    await h._file_config_unknown_keys_escalation()
    pending = _pending_config_l2s(queue)
    assert len(pending) == 1
    esc_id = pending[0].id

    # Operator fixed the config and restarted: the census is now empty.
    h.config._unknown_key_census = []
    await h._file_config_unknown_keys_escalation()

    assert _pending_config_l2s(queue) == []
    resolved = queue.get(esc_id)
    assert resolved is not None
    assert resolved.status == 'resolved'


@pytest.mark.asyncio
async def test_empty_census_no_pending_files_nothing(tmp_path: Path, monkeypatch):
    queue = EscalationQueue(tmp_path / 'esc')
    h = _make_harness(tmp_path, monkeypatch, [], queue)

    await h._file_config_unknown_keys_escalation()

    assert queue.get_pending() == []


@pytest.mark.asyncio
async def test_none_queue_is_noop(tmp_path: Path, monkeypatch):
    h = _make_harness(
        tmp_path, monkeypatch,
        [ConfigUnknownKey('spare_warm_lanes', 'git.spare_warm_lanes')],
        None,
    )
    # Must not raise even though the census is non-empty.
    await h._file_config_unknown_keys_escalation()


# --- reload_config surfaces the census ---------------------------------------


def _reload_harness(tmp_path):
    """Minimal Harness for driving reload_config() (uses self.config + event_store)."""
    h = Harness.__new__(Harness)
    _init_harness_state_for_test(h)
    h.config = OrchestratorConfig(project_root=tmp_path)
    h.event_store = None
    h._run_id = 'run-test'
    return h


@pytest.mark.asyncio
async def test_reload_config_surfaces_unknown_keys(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')
    h = _reload_harness(tmp_path)

    fresh = OrchestratorConfig(project_root=tmp_path)
    fresh._unknown_key_census = [ConfigUnknownKey('spare_warm_lanes', 'git.spare_warm_lanes')]

    monkeypatch.setenv('ORCH_CONFIG_PATH', str(tmp_path / 'orchestrator.yaml'))
    with patch('orchestrator.harness.load_config', return_value=fresh):
        report = await h.reload_config()

    assert 'unknown_config_keys' in report
    uck = report['unknown_config_keys']
    assert isinstance(uck, list)
    paths = {d['path'] for d in uck}
    assert 'spare_warm_lanes' in paths
    assert any(
        d.get('shadow_hint') and 'git.spare_warm_lanes' in d['shadow_hint'] for d in uck
    )


@pytest.mark.asyncio
async def test_reload_config_clean_yields_empty_unknown_keys(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', '')
    h = _reload_harness(tmp_path)

    fresh = OrchestratorConfig(project_root=tmp_path)  # census sentinel None → []

    monkeypatch.setenv('ORCH_CONFIG_PATH', str(tmp_path / 'orchestrator.yaml'))
    with patch('orchestrator.harness.load_config', return_value=fresh):
        report = await h.reload_config()

    assert report['unknown_config_keys'] == []


@pytest.mark.asyncio
async def test_reload_config_files_and_self_heals_l2(tmp_path: Path, monkeypatch):
    """reload_config is symmetric with startup (INV-5, one implementation): a
    phantom key introduced by a hot-reload files a born-at-L2, the live config's
    census is refreshed (not left at its stale startup value), and a subsequent
    clean reload self-heals the pending L2 — the same file/self-heal the startup
    filer performs."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(tmp_path / 'orchestrator.yaml'))
    queue = EscalationQueue(tmp_path / 'esc')
    h = _reload_harness(tmp_path)
    h._escalation_queue = queue

    # Hot-reload picks up a config carrying an unknown key → files an L2.
    dirty = OrchestratorConfig(project_root=tmp_path)
    dirty._unknown_key_census = [ConfigUnknownKey('spare_warm_lanes', 'git.spare_warm_lanes')]
    with patch('orchestrator.harness.load_config', return_value=dirty):
        report = await h.reload_config()

    assert report['unknown_config_keys']  # surfaced in the report
    # The live config's census is now in step with the report (no stale data).
    assert h.config.unknown_key_census == [
        ConfigUnknownKey('spare_warm_lanes', 'git.spare_warm_lanes')
    ]
    l2s = _pending_config_l2s(queue)
    assert len(l2s) == 1, 'a phantom key introduced via reload must file exactly one L2'
    assert l2s[0].level == 2
    assert l2s[0].severity in BORN_AT_L2_SEVERITIES
    esc_id = l2s[0].id

    # Operator fixes the config and hot-reloads → the pending L2 self-heals.
    clean = OrchestratorConfig(project_root=tmp_path)
    with patch('orchestrator.harness.load_config', return_value=clean):
        report = await h.reload_config()

    assert report['unknown_config_keys'] == []
    assert _pending_config_l2s(queue) == [], 'a clean reload must self-heal the pending L2'
    resolved = queue.get(esc_id)
    assert resolved is not None
    assert resolved.status == 'resolved'


# =============================================================================
# Escape hatches reach BOTH harness consumers (review round 1 — steps 15/16)
#
# The core regression guard for the reviewer's blocking finding: without a
# hatch, a project deliberately carrying keys for non-OrchestratorConfig
# consumers gets a PERMANENT, unresolvable born-at-L2, because the self-heal
# branch only fires on an EMPTY census and that census can never be empty
# without deleting working config.
# =============================================================================


# The measured reify shape: reify-owned knobs living in the SAME yaml blocks as
# real OrchestratorConfig fields (cpu_governance.enabled, fairness.skip_threshold).
_REIFY_HATCHED = {
    'cpu_governance': {
        'enabled': True,
        'weights': {'task': 100},
        'agent_admit': {'enabled': True},
    },
    'fairness': {'skip_threshold': 3, 'scheduler_v2': True},
}
_REIFY_ALLOWLIST = {
    'config_key_census': {'ignore': ['cpu_governance.*', 'fairness.scheduler_v2']}
}


def _census_harness(tmp_path, monkeypatch, data, queue, name='orchestrator.yaml'):
    """Harness whose config carries a census computed from a REAL YAML file.

    Unlike _make_harness (which injects a hand-built census), this drives the
    actual walk, so it proves the hatch reaches the harness rather than assuming
    the classification.
    """
    p = tmp_path / name
    p.write_text(yaml.dump(data))
    census = census_config_keys(p)
    h = _make_harness(tmp_path, monkeypatch, census.unknown, queue)
    h.config._ignored_key_census = census.ignored
    return h


@pytest.mark.asyncio
async def test_escape_hatched_only_config_files_nothing(tmp_path: Path, monkeypatch):
    """(a) NO FALSE L2 — the reviewer's finding.  A project whose only non-model
    keys are deliberate, allowlisted knobs must not be escalated at all."""
    queue = EscalationQueue(tmp_path / 'esc')
    h = _census_harness(
        tmp_path, monkeypatch, {**_REIFY_ALLOWLIST, **_REIFY_HATCHED}, queue
    )
    assert h.config.unknown_key_census == [], 'fixture precondition: census is clean'

    await h._file_config_unknown_keys_escalation()

    assert queue.get_pending() == []


@pytest.mark.asyncio
async def test_allowlist_resolves_a_previously_permanent_l2(tmp_path: Path, monkeypatch):
    """(b) RESOLVABILITY — the defect was that the L2 could never be cleared.

    File it from the pre-hatch census (those keys ARE unknown), then apply the
    operator's one-line remediation and re-run: the census goes empty and the
    EXISTING self-heal branch resolves it.  No manual resolve, no permanent L2.
    """
    queue = EscalationQueue(tmp_path / 'esc')
    # Before: no allowlist → the reify-owned keys are genuinely unknown.
    h = _census_harness(tmp_path, monkeypatch, dict(_REIFY_HATCHED), queue)
    assert len(h.config.unknown_key_census) == 3, h.config.unknown_key_census
    await h._file_config_unknown_keys_escalation()
    pending = _pending_config_l2s(queue)
    assert len(pending) == 1
    esc_id = pending[0].id

    # After: operator adds config_key_census.ignore and restarts.
    fixed = census_config_keys(
        _write_census_yaml(tmp_path, {**_REIFY_ALLOWLIST, **_REIFY_HATCHED}, 'fixed.yaml')
    )
    h.config._unknown_key_census = fixed.unknown
    h.config._ignored_key_census = fixed.ignored
    await h._file_config_unknown_keys_escalation()

    assert _pending_config_l2s(queue) == [], 'the documented remediation must self-heal'
    resolved = queue.get(esc_id)
    assert resolved is not None and resolved.status == 'resolved'


def _write_census_yaml(tmp_path: Path, data: dict, name: str) -> Path:
    p = tmp_path / name
    p.write_text(yaml.dump(data))
    return p


@pytest.mark.asyncio
async def test_genuine_signal_survives_the_hatch(tmp_path: Path, monkeypatch):
    """(c) SIGNAL PRESERVED — the hatch must not blind the census.

    A real unknown key alongside the allowlist still files exactly one born-at-L2,
    and the detail names ONLY that key (excused paths never appear, so the one
    genuine hit is not buried).
    """
    queue = EscalationQueue(tmp_path / 'esc')
    h = _census_harness(
        tmp_path, monkeypatch,
        {**_REIFY_ALLOWLIST, **_REIFY_HATCHED, 'warm_lane_pool': 6},
        queue,
    )

    await h._file_config_unknown_keys_escalation()

    l2s = _pending_config_l2s(queue)
    assert len(l2s) == 1
    esc = l2s[0]
    assert esc.level == 2
    assert esc.severity in BORN_AT_L2_SEVERITIES
    assert 'warm_lane_pool' in esc.detail
    assert 'git.warm_lane_pool' in esc.detail  # advisory placement hint
    for hatched in ('cpu_governance.weights', 'cpu_governance.agent_admit',
                    'fairness.scheduler_v2'):
        assert hatched not in esc.detail, f'excused key {hatched} must not be escalated'


@pytest.mark.asyncio
async def test_l2_detail_documents_the_escape_hatch(tmp_path: Path, monkeypatch):
    """The L2 must be self-describing: an operator hitting it needs the opt-out
    written down, else it still dead-ends them."""
    queue = EscalationQueue(tmp_path / 'esc')
    h = _make_harness(
        tmp_path, monkeypatch,
        [ConfigUnknownKey('some_project_knob', None)],
        queue,
    )

    await h._file_config_unknown_keys_escalation()

    detail = _pending_config_l2s(queue)[0].detail
    assert 'config_key_census.ignore' in detail
    assert 'x_' in detail


# --- (d) reload surfacing of the ignored census -------------------------------


@pytest.mark.asyncio
async def test_reload_config_surfaces_ignored_keys(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(tmp_path / 'orchestrator.yaml'))
    h = _reload_harness(tmp_path)

    fresh = OrchestratorConfig(project_root=tmp_path)
    fresh._unknown_key_census = [ConfigUnknownKey('warm_lane_pool', 'git.warm_lane_pool')]
    fresh._ignored_key_census = [
        ConfigIgnoredKey('cpu_governance.weights', 'allowlist'),
        ConfigIgnoredKey('x_custom', 'reserved_prefix'),
    ]

    with patch('orchestrator.harness.load_config', return_value=fresh):
        report = await h.reload_config()

    # Excused keys are reported separately and never inflate the unknown list.
    assert [d['path'] for d in report['unknown_config_keys']] == ['warm_lane_pool']
    ignored = {d['path']: d['reason'] for d in report['ignored_config_keys']}
    assert ignored == {
        'cpu_governance.weights': 'allowlist',
        'x_custom': 'reserved_prefix',
    }
    # The live config carries the fresh ignored census too (apply_reload copies
    # only model_fields, so the PrivateAttr needs an explicit hand-off).
    assert h.config.ignored_key_census == fresh._ignored_key_census


@pytest.mark.asyncio
async def test_reload_report_always_has_both_census_keys(tmp_path: Path, monkeypatch):
    """Both keys are ALWAYS present — including on the fail-closed load-failure
    shape — so callers can read them unconditionally."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(tmp_path / 'orchestrator.yaml'))
    h = _reload_harness(tmp_path)

    with patch('orchestrator.harness.load_config', side_effect=RuntimeError('boom')):
        report = await h.reload_config()

    assert report['reloaded'] is False
    assert report['unknown_config_keys'] == []
    assert report['ignored_config_keys'] == []
