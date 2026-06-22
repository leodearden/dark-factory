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


class TestClearParksForRestoreContract:
    """clear_parks_for(task_id) must return restored (owner, modules) pairs (step-3)."""

    def test_clear_parks_for_returns_restored_pairs_on_top_removal(self):
        """Clearing the active top exposes the buried entry — returned as restored pair."""
        lt = _lt()
        lt.install_parks('L', ['m1'], 'low')
        lt.install_parks('H', ['m1'], 'high')  # H on top, L buried
        # Clear H (the top) — exposes L as the new active top.
        restored = lt.clear_parks_for('H')
        assert restored == [('L', ['m1'])], f'Expected L restored to m1, got: {restored}'
        assert lt.has_parks('L'), 'L must be the restored active top after H is cleared'

    def test_clear_parks_for_returns_empty_when_no_restore(self):
        """Clearing the only entry leaves no buried owner to restore — empty list."""
        lt = _lt()
        lt.install_parks('A', ['m1'], 'medium')
        restored = lt.clear_parks_for('A')
        assert restored == [], f'Expected empty restore list, got: {restored}'
        assert not lt.has_parks('A')

    def test_clear_parks_for_returns_empty_on_buried_removal(self):
        """INV-4 buried clause: clearing a buried (non-top) owner reports no restore."""
        lt = _lt()
        lt.install_parks('L', ['m1'], 'low')
        lt.install_parks('H', ['m1'], 'high')  # H on top, L buried
        # Clear L (the buried entry) — H stays on top; nothing is newly exposed.
        restored = lt.clear_parks_for('L')
        assert restored == [], f'Removing a buried entry must not report any restore, got: {restored}'
        # H is still the active top.
        assert lt.has_parks('H'), 'H must remain on top after buried L is cleared'

    def test_clear_parks_for_returns_multiple_modules_per_restored_owner(self):
        """When an owner is restored across multiple modules, all are in one pair."""
        lt = _lt()
        lt.install_parks('L', ['m1', 'm2'], 'low')
        lt.install_parks('H', ['m1', 'm2'], 'high')  # H shadows L on both
        restored = lt.clear_parks_for('H')
        # L should be restored on both m1 and m2.
        assert len(restored) == 1
        owner, mods = restored[0]
        assert owner == 'L'
        assert sorted(mods) == ['m1', 'm2'], f'Expected L restored on [m1,m2], got: {mods}'


class TestPruneOwners:
    """prune_owners(predicate) owner-state GC: widened return (evicted, restored_pairs)."""

    def test_prune_owners_drops_matching(self):
        """prune_owners evicts owners for which the predicate returns True."""
        lt = _lt()
        lt.install_parks('A', ['m1'], 'medium')
        lt.install_parks('B', ['m2'], 'medium')
        evicted, restored = lt.prune_owners(lambda tid: tid == 'A')
        assert evicted == ['A']
        assert restored == []  # A was the only owner of m1, no buried entry to expose
        assert not lt.has_parks('A')
        assert lt.has_parks('B')

    def test_prune_owners_returns_empty_when_no_match(self):
        """prune_owners returns ([], []) when the predicate never fires."""
        lt = _lt()
        lt.install_parks('A', ['m1'], 'medium')
        evicted, restored = lt.prune_owners(lambda tid: False)
        assert evicted == []
        assert restored == []
        assert lt.has_parks('A')

    def test_prune_owners_dedups_owners_by_first_seen(self):
        """Owner with multiple parks appears only once in the evicted list."""
        lt = _lt()
        lt.install_parks('A', ['m1', 'm2'], 'medium')
        evicted, restored = lt.prune_owners(lambda tid: True)
        assert evicted == ['A']
        assert restored == []

    def test_prune_owners_predicate_called_per_owner_not_per_park(self):
        """Predicate is called at most once per distinct owner (memoized)."""
        lt = _lt()
        lt.install_parks('A', ['m1', 'm2'], 'medium')
        calls: list[str] = []

        def predicate(tid: str) -> bool:
            calls.append(tid)
            return True

        evicted, restored = lt.prune_owners(predicate)
        # A owns two parks but predicate should be called ≤ 1 time for 'A'.
        assert calls.count('A') <= 1

    def test_prune_owners_returns_restored_when_top_is_pruned(self):
        """Pruning the active top exposes a buried entry — returned as restored pair."""
        lt = _lt()
        lt.install_parks('L', ['m1'], 'low')
        lt.install_parks('H', ['m1'], 'high')  # H on top, L buried
        evicted, restored = lt.prune_owners(lambda tid: tid == 'H')
        assert evicted == ['H']
        assert restored == [('L', ['m1'])], f'Expected L restored after H pruned, got: {restored}'

    def test_prune_owners_no_restore_when_buried_is_pruned(self):
        """INV-4 buried clause: pruning a buried (non-top) owner reports no restore."""
        lt = _lt()
        lt.install_parks('L', ['m1'], 'low')
        lt.install_parks('H', ['m1'], 'high')  # H on top, L buried
        evicted, restored = lt.prune_owners(lambda tid: tid == 'L')
        assert evicted == ['L']
        assert restored == [], f'Pruning a buried entry must not produce restores, got: {restored}'
        # H is still on top.
        assert lt.has_parks('H'), 'H must remain the active top after buried L is pruned'

    def test_prune_expired_parks_does_not_exist(self):
        """prune_expired_parks must be gone (replaced by prune_owners)."""
        assert not hasattr(ModuleLockTable, 'prune_expired_parks')


class TestHierarchicalPreemptionInvariants:
    """Pinned invariants for hierarchical preemption (step-11 RED tests).

    Both test cases are RED against the current re-key code:
    - case (a): the re-key collapses distinct child keys onto one parent key,
      leaving only one victim restorable after the preemptor clears.
    - case (b): the re-key pops only the TOP of the victim's key, leaving the
      buried entry exposed as a conflicting active top on the original key.
    """

    @staticmethod
    def _check_inv3(lt: ModuleLockTable) -> None:
        """Assert every park stack is strictly rank-decreasing top-ward (INV-3).

        For each stack, the rank of the active top (last element) must be
        strictly less than the rank of every entry beneath it, i.e., ranks
        strictly increase from the top (index -1) downward to the bottom
        (index 0).
        """
        for m, stack in lt._parked.items():
            for i in range(len(stack) - 1, 0, -1):
                top_rank = stack[i][1]
                below_rank = stack[i - 1][1]
                assert top_rank < below_rank, (
                    f"INV-3 violated on key {m!r}: "
                    f"entry[{i}] rank={top_rank} must be strictly less than "
                    f"entry[{i - 1}] rank={below_rank}. "
                    f"Full stack: {stack}"
                )

    def test_two_distinct_hierarchical_victims_no_collapse_no_inversion(self):
        """Two distinct child-key victims must shadow under their ORIGINAL keys.

        L1 parks ['backend/api'] (low), L2 parks ['backend/db'] (medium),
        then H installs ['backend'] (high).  The shadowed return must report
        BOTH victims under their original keys; INV-3 must hold; both victims
        must be blocked during shadow while H can acquire freely; and after
        clear_parks_for('H') BOTH victims must be independently restorable —
        neither stranded under the other on a collapsed key.

        RED under the current re-key code because clear_parks_for('H') exposes
        L2 as the active top of 'backend', blocking L1's try_acquire on
        'backend/api' permanently.
        """
        lt = _lt(lock_depth=3)
        lt.install_parks('L1', ['backend/api'], 'low')    # rank 3
        lt.install_parks('L2', ['backend/db'], 'medium')  # rank 2
        installed, shadowed = lt.install_parks('H', ['backend'], 'high')  # rank 1

        assert installed == ['backend']
        assert {(owner, tuple(mods)) for owner, mods in shadowed} == {
            ('L1', ('backend/api',)),
            ('L2', ('backend/db',)),
        }, f'Both victims must appear under their original keys; got: {shadowed}'

        # During shadow: victims blocked, H can acquire freely.
        assert not lt.try_acquire('L1', ['backend/api']), 'L1 must be blocked during H shadow'
        assert not lt.try_acquire('L2', ['backend/db']), 'L2 must be blocked during H shadow'
        assert lt.try_acquire('H', ['backend']), 'H must be able to acquire its reserved module'
        lt.release('H')  # release held lock; park stacks unchanged

        # INV-3: every stack strictly rank-decreasing top-ward.
        self._check_inv3(lt)

        # After H clears its park: BOTH victims must be independently restorable.
        lt.clear_parks_for('H')
        assert lt.try_acquire('L1', ['backend/api']), (
            'L1 must be restorable to backend/api after H clears; '
            'stranding L1 under L2 on the collapsed key is the regression'
        )
        assert lt.try_acquire('L2', ['backend/db']), (
            'L2 must be restorable to backend/db after H clears'
        )

    def test_hierarchical_preempt_over_victim_with_buried_entry(self):
        """Hierarchical preempt must not expose a buried entry as an active top.

        X parks ['backend/api'] (low), Y parks ['backend/api'] (medium) so Y
        exact-match-shadows X (stack [X,Y] with Y the active top).  Then H
        installs ['backend'] (high).

        After install:
        - The 'backend/api' stack must be untouched: Y still active top, X buried.
        - H can acquire 'backend' freely; Y cannot acquire 'backend/api'.
        - INV-3 holds across all stacks.

        After clear_parks_for('H'):
        - Y is restored as active top of 'backend/api'.
        - X remains correctly buried beneath Y.

        RED under the current re-key code: the re-key pops only Y (the top of
        'backend/api') and moves it to 'backend', leaving X as the sole
        (incorrectly exposed) active top of 'backend/api', which then blocks H
        from acquiring 'backend'.
        """
        lt = _lt(lock_depth=3)
        lt.install_parks('X', ['backend/api'], 'low')    # rank 3
        lt.install_parks('Y', ['backend/api'], 'medium') # rank 2; exact-match shadow: [X,Y]

        installed, shadowed = lt.install_parks('H', ['backend'], 'high')  # rank 1
        assert installed == ['backend']

        # 'backend/api' stack must be UNTOUCHED: Y still active top, X buried.
        assert 'backend/api' in lt._parked, (
            "'backend/api' key must still exist in _parked after hierarchical preempt"
        )
        assert lt._parked['backend/api'][-1][0] == 'Y', (
            f"Y must remain the active top of 'backend/api'; "
            f"got {lt._parked['backend/api'][-1][0]!r}. "
            f"Exposing X as the active top is the re-key bug."
        )

        # H can acquire 'backend' freely; Y (shadowed) cannot acquire 'backend/api'.
        assert lt.try_acquire('H', ['backend']), (
            'H must not be blocked by X (which the re-key bug exposes as a '
            'conflicting active top of backend/api)'
        )
        assert not lt.try_acquire('Y', ['backend/api']), 'Y must be blocked while H shadows it'
        lt.release('H')  # release held lock; park stacks unchanged

        # INV-3: every stack strictly rank-decreasing top-ward.
        self._check_inv3(lt)

        # After H clears its park: Y is restored; X remains correctly buried.
        lt.clear_parks_for('H')
        assert lt.try_acquire('Y', ['backend/api']), 'Y must be restored after H clears'
        assert not lt.try_acquire('X', ['backend/api']), (
            'X must remain buried beneath restored Y'
        )
