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

import pytest
from escalation.models import BORN_AT_L2_SEVERITIES
from escalation.queue import EscalationQueue
from escalation.server import CATEGORIES

from _orch_helpers import _init_harness_state_for_test

from orchestrator.config import ConfigUnknownKey, OrchestratorConfig
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
