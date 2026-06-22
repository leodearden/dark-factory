"""Lock-table-level unit tests for the v2 install_parks API.

Tests here cover:
  - New signature: install_parks(task_id, modules, priority) -> (installed, shadowed)
  - Cross-tier preemption (higher-priority park SHADOWS lower-priority overlap)
  - Park-stack invariants INV-1 through INV-7
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
    """Cross-tier preemption: higher-priority park SHADOWS lower-priority overlap."""

    def test_higher_tier_evicts_lower_overlap(self):
        """High-priority install SHADOWS (not destroys) a lower-priority park on the same module."""
        lt = _lt()
        lt.install_parks('L', ['m1', 'm2'], 'low')
        installed, shadowed = lt.install_parks('H', ['m1'], 'high')
        assert installed == ['m1']
        # 2nd slot reports the shadowed (retained) lower reservation — same shape as old evicted.
        assert shadowed == [('L', ['m1'])]
        # H is active top on m1; L is buried on m1 AND still active on m2.
        assert lt.has_parks('H')
        assert lt.has_parks('L')  # L is retained in the shadow stack (INV-5)
        # H's park on m1 means L cannot acquire m1 while H is on top (INV-2).
        assert not lt.try_acquire('L', ['m1'])
        # L's park on m2 remains — H cannot acquire m2.
        assert not lt.try_acquire('H', ['m2'])
        # After H completes (clear), L is RESTORED as the active top on m1 (INV-4).
        lt.clear_parks_for('H')
        assert lt.try_acquire('L', ['m1'])

    def test_higher_tier_evicts_multiple_lower_owners(self):
        """A single high-priority install shadows parks from multiple owners."""
        lt = _lt()
        # L1 parks m1; L2 parks m2.
        lt.install_parks('L1', ['m1'], 'low')
        lt.install_parks('L2', ['m2'], 'low')
        installed, shadowed = lt.install_parks('H', ['m1', 'm2'], 'high')
        assert sorted(installed) == ['m1', 'm2']
        # Both L1 and L2 should appear in shadowed (any order).
        shadowed_owners = {owner for owner, _ in shadowed}
        assert shadowed_owners == {'L1', 'L2'}
        # Shadow semantics: both L1 and L2 are RETAINED (buried), not destroyed (INV-5).
        assert lt.has_parks('L1')
        assert lt.has_parks('L2')
        # L1/L2 cannot acquire their shadowed modules — H is on top (INV-2).
        assert not lt.try_acquire('L1', ['m1'])
        assert not lt.try_acquire('L2', ['m2'])
        # After H clears, both are restored (INV-4).
        lt.clear_parks_for('H')
        assert lt.try_acquire('L1', ['m1'])
        assert lt.try_acquire('L2', ['m2'])

    def test_higher_tier_no_eviction_when_no_overlap(self):
        """High-priority install on a disjoint module does not shadow lower park."""
        lt = _lt()
        lt.install_parks('L', ['m1'], 'low')
        installed, shadowed = lt.install_parks('H', ['m2'], 'high')
        assert installed == ['m2']
        assert shadowed == []
        # Both parks coexist independently.
        assert lt.has_parks('L')
        assert lt.has_parks('H')

    def test_lower_tier_does_not_preempt_higher(self):
        """A low-priority install cannot push/shadow an existing high-priority park (INV-3)."""
        lt = _lt()
        lt.install_parks('H', ['m1'], 'high')
        installed, shadowed = lt.install_parks('L', ['m1'], 'low')
        # L gets nothing; H's park is untouched. No push (same/higher top blocks).
        assert installed == []
        assert shadowed == []
        assert lt.has_parks('H')
        assert not lt.has_parks('L')

    def test_hierarchical_preemption(self):
        """A high-priority install on a child SHADOWS a low-priority park on the parent."""
        lt = _lt(lock_depth=3)
        # Low park on 'backend' (the parent).
        lt.install_parks('L', ['backend'], 'low')
        # High install on 'backend/api' — conflicts hierarchically with 'backend'.
        installed, shadowed = lt.install_parks('H', ['backend/api'], 'high')
        assert installed == ['backend/api']
        # L's park on 'backend' is now shadowed (buried beneath H), not destroyed.
        assert shadowed == [('L', ['backend'])]
        # INV-5: L is buried but has_parks must return True.
        assert lt.has_parks('L')
        assert lt.has_parks('H')
        # L cannot acquire while H holds the top (INV-2).
        assert not lt.try_acquire('L', ['backend'])
        # After H clears, L is restored to the top (INV-4).
        lt.clear_parks_for('H')
        assert lt.try_acquire('L', ['backend'])


class TestParkStackInvariants:
    """Unit tests for the per-module park-stack invariants (INV-1 through INV-5)."""

    def test_inv1_push_on_preempt_retains_lower_owner(self):
        """INV-1: strictly-higher-priority install PUSHES (shadows) the lower park.

        The 2nd return slot lists the shadowed (retained) pair.
        The lower owner is still accessible via has_parks.
        """
        lt = _lt()
        lt.install_parks('low_task', ['m1'], 'low')
        installed, shadowed = lt.install_parks('high_task', ['m1'], 'high')
        assert installed == ['m1']
        # 2nd slot must identify the shadowed owner and its modules.
        assert shadowed == [('low_task', ['m1'])]
        # Shadow semantics: low_task is BURIED (not destroyed), has_parks still True.
        assert lt.has_parks('low_task'), 'low_task must be retained in the shadow stack (INV-1)'
        assert lt.has_parks('high_task'), 'high_task is the active top'

    def test_inv2_shadowed_owner_blocked_does_not_block_above(self):
        """INV-2: _is_parked_blocks keyed on TOP only.

        A shadowed (buried) owner cannot acquire its module.
        A buried reservation does NOT block the active-top owner from acquiring.
        """
        lt = _lt()
        lt.install_parks('L', ['m1'], 'low')
        lt.install_parks('H', ['m1'], 'high')  # H pushes, L is buried
        # INV-5: L is buried but has_parks must be True.
        assert lt.has_parks('L'), 'L is buried in the shadow stack — has_parks must be True'
        # L is blocked — cannot acquire the module it is shadowed on.
        assert not lt.try_acquire('L', ['m1']), 'Buried L must not acquire its shadowed module'
        # H (the active top) can acquire m1 freely — the buried L does not block it.
        assert lt.try_acquire('H', ['m1']), 'Active-top H must be able to acquire its module'

    def test_inv3_same_tier_blocks_install_no_push(self):
        """INV-3: same-priority existing top BLOCKS a new install (no push, no shadow)."""
        lt = _lt()
        lt.install_parks('A', ['m1'], 'medium')
        installed, shadowed = lt.install_parks('B', ['m1'], 'medium')
        assert installed == []
        assert shadowed == []  # No push for same tier
        assert lt.has_parks('A')
        assert not lt.has_parks('B')

    def test_inv3_higher_top_blocks_lower_install_no_push(self):
        """INV-3: higher-priority existing top BLOCKS a lower-priority install (no push)."""
        lt = _lt()
        lt.install_parks('H', ['m1'], 'high')
        installed, shadowed = lt.install_parks('L', ['m1'], 'low')
        assert installed == []
        assert shadowed == []  # No push — lower priority cannot shadow higher
        assert lt.has_parks('H')
        assert not lt.has_parks('L')

    def test_inv4_pop_and_restore_on_clear(self):
        """INV-4: clearing the top entry exposes and restores the next-lower entry."""
        lt = _lt()
        lt.install_parks('L', ['m1'], 'low')
        lt.install_parks('H', ['m1'], 'high')  # H on top, L buried
        # H dispatches — clear its parks.
        lt.clear_parks_for('H')
        # L is restored to the top of m1.
        assert lt.has_parks('L'), 'L must be restored to the top after H is cleared (INV-4)'
        # L can now acquire m1.
        assert lt.try_acquire('L', ['m1']), 'Restored L must be able to acquire m1'

    def test_inv4_pop_buried_entry_leaves_top_unchanged(self):
        """INV-4 buried clause: removing a non-top entry leaves the top unchanged."""
        lt = _lt()
        lt.install_parks('L', ['m1'], 'low')
        lt.install_parks('H', ['m1'], 'high')  # H on top, L buried
        # Remove L (the buried entry) — should not affect H's top position.
        lt.clear_parks_for('L')
        # H is still the active top.
        assert lt.has_parks('H'), 'H must remain the active top when buried L is cleared'
        assert not lt.has_parks('L'), 'L must be fully gone after clear_parks_for'

    def test_inv5_has_parks_true_for_buried_owner(self):
        """INV-5: has_parks returns True even for an owner buried in a shadow stack."""
        lt = _lt()
        lt.install_parks('low_task', ['m1'], 'low')
        lt.install_parks('high_task', ['m1'], 'high')  # low_task is now buried
        # Both owners: top and buried.
        assert lt.has_parks('low_task'), 'Buried owner must still report has_parks True (INV-5)'
        assert lt.has_parks('high_task'), 'Active-top owner must report has_parks True'


class TestPruneOwners:
    """prune_owners(predicate) owner-state GC and removal of prune_expired_parks."""

    def test_prune_owners_drops_matching(self):
        """prune_owners evicts owners for which the predicate returns True."""
        lt = _lt()
        lt.install_parks('A', ['m1'], 'medium')
        lt.install_parks('B', ['m2'], 'medium')
        evicted = lt.prune_owners(lambda tid: tid == 'A')
        assert evicted == ['A']
        assert not lt.has_parks('A')
        assert lt.has_parks('B')

    def test_prune_owners_returns_empty_when_no_match(self):
        """prune_owners returns [] when the predicate never fires."""
        lt = _lt()
        lt.install_parks('A', ['m1'], 'medium')
        evicted = lt.prune_owners(lambda tid: False)
        assert evicted == []
        assert lt.has_parks('A')

    def test_prune_owners_dedups_owners_by_first_seen(self):
        """Owner with multiple parks appears only once in the result."""
        lt = _lt()
        lt.install_parks('A', ['m1', 'm2'], 'medium')
        evicted = lt.prune_owners(lambda tid: True)
        assert evicted == ['A']

    def test_prune_owners_predicate_called_per_owner_not_per_park(self):
        """Predicate is called at most once per distinct owner (memoized)."""
        lt = _lt()
        lt.install_parks('A', ['m1', 'm2'], 'medium')
        calls: list[str] = []

        def predicate(tid: str) -> bool:
            calls.append(tid)
            return True

        lt.prune_owners(predicate)
        # A owns two parks but predicate should be called ≤ 1 time for 'A'.
        assert calls.count('A') <= 1

    def test_prune_expired_parks_does_not_exist(self):
        """prune_expired_parks must be gone (replaced by prune_owners)."""
        assert not hasattr(ModuleLockTable, 'prune_expired_parks')
