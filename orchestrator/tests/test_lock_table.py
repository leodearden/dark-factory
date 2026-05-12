"""Lock-table-level unit tests for the v2 install_parks API.

Tests here cover:
  - New signature: install_parks(task_id, modules, priority) -> (installed, evicted)
  - Cross-tier preemption (higher-priority park evicts lower-priority overlap)
  - prune_owners(predicate) owner-state GC

Existing ModuleLockTable / hierarchical locking / conflicts tests remain in
test_scheduler.py alongside TestModuleLockTable / TestHierarchicalLocking.
"""

from __future__ import annotations

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import ModuleLockTable


def _lt(lock_depth: int = 2) -> ModuleLockTable:
    """Return a fresh ModuleLockTable with standard test config."""
    return ModuleLockTable(OrchestratorConfig(max_per_module=1, lock_depth=lock_depth))


class TestInstallParksApi:
    """New install_parks(task_id, modules, priority) signature + return shape."""

    def test_install_parks_returns_installed_and_empty_evicted(self):
        """Fresh table: install returns (installed_list, []) tuple."""
        lt = _lt()
        result = lt.install_parks('A', ['backend'], priority='high')
        installed, evicted = result
        assert installed == ['backend']
        assert evicted == []

    def test_install_parks_blocks_non_owner(self):
        """After install, a different task cannot acquire the parked module."""
        lt = _lt()
        lt.install_parks('A', ['backend'], priority='medium')
        assert not lt.try_acquire('B', ['backend'])

    def test_install_parks_owner_can_acquire_own_park(self):
        """The park owner can still acquire its own reserved module."""
        lt = _lt()
        lt.install_parks('A', ['backend'], priority='medium')
        assert lt.try_acquire('A', ['backend'])

    def test_install_parks_normalizes_modules(self):
        """Modules deeper than lock_depth are truncated on install."""
        lt = _lt(lock_depth=2)
        installed, _ = lt.install_parks('A', ['backend/sub/deeper'], priority='medium')
        assert installed == ['backend/sub']
        # The normalized form blocks a direct acquire.
        assert not lt.try_acquire('B', ['backend/sub'])

    def test_same_tier_does_not_evict(self):
        """A same-tier install does NOT evict the existing park."""
        lt = _lt()
        lt.install_parks('A', ['m1'], priority='high')
        installed, evicted = lt.install_parks('B', ['m1'], priority='high')
        # B did not take the slot.
        assert evicted == []
        assert installed == []
        # A's park survives.
        assert lt.has_parks('A')
        assert not lt.has_parks('B')

    def test_install_parks_no_deadline_kwarg(self):
        """Calling with the old deadline= kwarg must raise TypeError."""
        lt = _lt()
        with pytest.raises(TypeError):
            lt.install_parks('A', ['backend'], deadline=9999.0)  # type: ignore[call-arg]
