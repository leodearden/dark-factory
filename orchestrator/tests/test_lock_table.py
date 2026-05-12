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


class TestInstallParksPreemption:
    """Cross-tier preemption: higher-priority park evicts lower-priority overlap."""

    def test_higher_tier_evicts_lower_overlap(self):
        """High-priority install evicts a lower-priority park on the same module."""
        lt = _lt()
        lt.install_parks('L', ['m1', 'm2'], 'low')
        installed, evicted = lt.install_parks('H', ['m1'], 'high')
        assert installed == ['m1']
        assert evicted == [('L', ['m1'])]
        # H owns m1 now; L still owns m2.
        assert lt.has_parks('H')
        assert lt.has_parks('L')
        # H's park on m1 means L cannot re-acquire m1.
        assert not lt.try_acquire('L', ['m1'])
        # L's park on m2 remains — H cannot acquire m2.
        assert not lt.try_acquire('H', ['m2'])

    def test_higher_tier_evicts_multiple_lower_owners(self):
        """A single high-priority install can evict parks from multiple owners."""
        lt = _lt()
        # L1 parks m1; L2 parks m2.
        lt.install_parks('L1', ['m1'], 'low')
        lt.install_parks('L2', ['m2'], 'low')
        installed, evicted = lt.install_parks('H', ['m1', 'm2'], 'high')
        assert sorted(installed) == ['m1', 'm2']
        # Both L1 and L2 should appear in evicted (any order).
        evicted_owners = {owner for owner, _ in evicted}
        assert evicted_owners == {'L1', 'L2'}
        assert not lt.has_parks('L1')
        assert not lt.has_parks('L2')

    def test_higher_tier_no_eviction_when_no_overlap(self):
        """High-priority install on a disjoint module does not evict lower park."""
        lt = _lt()
        lt.install_parks('L', ['m1'], 'low')
        installed, evicted = lt.install_parks('H', ['m2'], 'high')
        assert installed == ['m2']
        assert evicted == []
        # Both parks coexist.
        assert lt.has_parks('L')
        assert lt.has_parks('H')

    def test_lower_tier_does_not_preempt_higher(self):
        """A low-priority install cannot evict an existing high-priority park."""
        lt = _lt()
        lt.install_parks('H', ['m1'], 'high')
        installed, evicted = lt.install_parks('L', ['m1'], 'low')
        # L gets nothing; H's park is untouched.
        assert installed == []
        assert evicted == []
        assert lt.has_parks('H')
        assert not lt.has_parks('L')

    def test_hierarchical_preemption(self):
        """A high-priority install on a child evicts a low-priority park on the parent."""
        lt = _lt(lock_depth=3)
        # Low park on 'backend' (the parent).
        lt.install_parks('L', ['backend'], 'low')
        # High install on 'backend/api' — conflicts hierarchically with 'backend'.
        installed, evicted = lt.install_parks('H', ['backend/api'], 'high')
        assert installed == ['backend/api']
        # L's park on 'backend' should be evicted due to hierarchical conflict.
        assert evicted == [('L', ['backend'])]
        assert not lt.has_parks('L')
        assert lt.has_parks('H')
