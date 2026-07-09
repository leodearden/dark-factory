"""Unit tests for orchestrator.merge_types.SpecPermit and
orchestrator.merge_speculation_controller.PermitLedger (MQ-refactor zeta /
task 2159).

Steps covered:
  step-1 RED   — PermitLedger acquire path: SpecPermit identity + slot/live
                 bookkeeping
  step-2 GREEN — implement SpecPermit (merge_types.py) + PermitLedger.acquire
                 (merge_speculation_controller.py)
  step-3 RED   — PermitLedger.release: idempotent double-release + live-assert
  step-4 GREEN — implement PermitLedger.release
  step-7 RED   — permit storage slot on RealMergeItem/DecidedItem/InflightEntry
  step-8 GREEN — add the permit storage field to the three dataclasses

This module imports orchestrator.merge_speculation_controller LOCALLY inside
each test — PermitLedger does not exist until step-2 — so a not-yet-implemented
symbol never breaks collection of the rest of the file during the RED steps
(mirrors test_merge_speculation_controller.py's convention). Neither
PermitLedger nor SpecPermit has any git/GitOps dependency, so this suite has
no git_ops/git_repo fixtures at all.
"""

from __future__ import annotations

import asyncio

import pytest

# ---------------------------------------------------------------------------
# step-1 RED / step-2 GREEN: PermitLedger acquire path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPermitLedgerAcquire:
    """PermitLedger.acquire() -> SpecPermit (task 2159 step-1).

    RED until step-2 GREEN adds SpecPermit (merge_types.py) and PermitLedger
    (merge_speculation_controller.py).
    """

    @pytest.mark.parametrize('depth', [1, 2, 3])
    async def test_acquire_returns_a_fresh_unreleased_permit_registered_live(
        self, depth: int,
    ) -> None:
        from orchestrator.merge_speculation_controller import PermitLedger

        ledger = PermitLedger(asyncio.Semaphore(depth), depth)

        p = await ledger.acquire()

        assert p.released is False
        assert p in ledger.live

    @pytest.mark.parametrize('depth', [1, 2, 3])
    async def test_n_acquires_track_live_count_and_slot_available(
        self, depth: int,
    ) -> None:
        from orchestrator.merge_speculation_controller import PermitLedger

        ledger = PermitLedger(asyncio.Semaphore(depth), depth)

        for n in range(1, depth + 1):
            permit = await ledger.acquire()

            assert permit.released is False
            assert permit in ledger.live
            assert len(ledger.live) == n
            assert ledger.slot_available == depth - n
            # Construction identity (PRD invariant P-1): holds after every
            # acquire, by construction — no intervening await between the
            # semaphore decrement and the `live` registration.
            assert ledger.slot_available + len(ledger.live) == ledger.depth


# ---------------------------------------------------------------------------
# step-3 RED / step-4 GREEN: PermitLedger.release
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPermitLedgerRelease:
    """PermitLedger.release() — idempotent + live-checked (task 2159 step-3).

    RED until step-4 GREEN adds PermitLedger.release.
    """

    @pytest.mark.parametrize('depth', [1, 2])
    async def test_release_restores_slot_and_marks_permit_released(
        self, depth: int,
    ) -> None:
        from orchestrator.merge_speculation_controller import PermitLedger

        ledger = PermitLedger(asyncio.Semaphore(depth), depth)
        p = await ledger.acquire()

        ledger.release(p)

        assert p.released is True
        assert p not in ledger.live
        assert ledger.slot_available == depth
        assert ledger.slot_available + len(ledger.live) == ledger.depth

    @pytest.mark.parametrize('depth', [1, 2])
    async def test_double_release_is_a_silent_noop(self, depth: int) -> None:
        from orchestrator.merge_speculation_controller import PermitLedger

        ledger = PermitLedger(asyncio.Semaphore(depth), depth)
        p = await ledger.acquire()
        ledger.release(p)

        ledger.release(p)  # second release — must not raise or over-release

        assert p.released is True
        assert p not in ledger.live
        assert ledger.slot_available == depth  # never exceeds depth
        assert ledger.slot_available + len(ledger.live) == ledger.depth

    async def test_release_of_a_never_acquired_permit_raises(self) -> None:
        from orchestrator.merge_speculation_controller import PermitLedger
        from orchestrator.merge_types import SpecPermit

        ledger = PermitLedger(asyncio.Semaphore(1), 1)
        stranger = SpecPermit()

        with pytest.raises(AssertionError):
            ledger.release(stranger)

    async def test_mixed_acquire_release_sequence_conserves_the_identity(
        self,
    ) -> None:
        """A realistic mixed sequence — acquire/acquire/release/acquire/
        double-release/release/release — keeps ``slot_available +
        len(live) == depth`` at every single step (the ledger-level B7, in
        isolation from any real verifier release).
        """
        from orchestrator.merge_speculation_controller import PermitLedger

        depth = 3
        ledger = PermitLedger(asyncio.Semaphore(depth), depth)

        def _assert_identity() -> None:
            assert ledger.slot_available + len(ledger.live) == ledger.depth

        _assert_identity()
        p1 = await ledger.acquire()
        _assert_identity()
        p2 = await ledger.acquire()
        _assert_identity()
        ledger.release(p1)
        _assert_identity()
        p3 = await ledger.acquire()
        _assert_identity()
        ledger.release(p1)  # double-release of an already-released token
        _assert_identity()
        ledger.release(p2)
        _assert_identity()
        ledger.release(p3)
        _assert_identity()
        assert ledger.slot_available == depth
        assert ledger.live == frozenset()


# ---------------------------------------------------------------------------
# step-1 RED / step-2 GREEN (task 2161, merge-queue-reliability PRD scope-3
# theta -- distinct from the older MQ-refactor theta/task 1993
# SpeculationController): CapPermit + PermitLedger's token_factory param
# ---------------------------------------------------------------------------


class TestCapPermit:
    """CapPermit is a distinct token type, byte-for-byte mirroring SpecPermit's
    identity semantics (task 2161 step-1).

    RED until step-2 GREEN adds CapPermit to merge_types.py.
    """

    def test_cap_permit_released_defaults_false(self) -> None:
        from orchestrator.merge_types import CapPermit

        p = CapPermit()

        assert p.released is False

    def test_two_cap_permits_compare_unequal(self) -> None:
        """eq=False -> identity, not field-value, comparison (mirrors
        SpecPermit): a PermitLedger's ``live`` set must treat every acquired
        CapPermit as a distinct member."""
        from orchestrator.merge_types import CapPermit

        p1 = CapPermit()
        p2 = CapPermit()

        assert p1 != p2
        assert p1 == p1


@pytest.mark.asyncio
class TestPermitLedgerTokenFactory:
    """PermitLedger generalized over its token type via a ``token_factory``
    constructor param (task 2161 step-1), so a second ledger instance can own
    ``_merge_ahead_cap`` and issue :class:`CapPermit` tokens through the
    identical acquire/release/conservation machinery
    ``_speculation_ledger`` uses for :class:`SpecPermit`.

    RED until step-2 GREEN adds the ``token_factory`` parameter to
    ``PermitLedger.__init__``.
    """

    @pytest.mark.parametrize('depth', [1, 2, 3])
    async def test_acquire_returns_a_fresh_unreleased_cap_permit_registered_live(
        self, depth: int,
    ) -> None:
        from orchestrator.merge_speculation_controller import PermitLedger
        from orchestrator.merge_types import CapPermit

        ledger = PermitLedger(asyncio.Semaphore(depth), depth, token_factory=CapPermit)

        p = await ledger.acquire()

        assert isinstance(p, CapPermit)
        assert p.released is False
        assert p in ledger.live

    @pytest.mark.parametrize('depth', [1, 2, 3])
    async def test_n_acquire_release_cycles_conserve_the_identity(
        self, depth: int,
    ) -> None:
        """Mirrors TestPermitLedgerAcquire/TestPermitLedgerRelease's identity
        assertions, but for a CapPermit-backed ledger -- proves the
        generalization didn't special-case SpecPermit anywhere in the
        acquire/release/conservation path."""
        from orchestrator.merge_speculation_controller import PermitLedger
        from orchestrator.merge_types import CapPermit

        ledger = PermitLedger(asyncio.Semaphore(depth), depth, token_factory=CapPermit)

        def _assert_identity() -> None:
            assert ledger.slot_available + len(ledger.live) == ledger.depth

        _assert_identity()
        permits = []
        for n in range(1, depth + 1):
            permit = await ledger.acquire()
            assert isinstance(permit, CapPermit)
            assert permit.released is False
            assert permit in ledger.live
            assert len(ledger.live) == n
            assert ledger.slot_available == depth - n
            _assert_identity()
            permits.append(permit)

        for permit in permits:
            ledger.release(permit)
            _assert_identity()

        assert ledger.slot_available == depth
        assert ledger.live == frozenset()

    async def test_release_is_idempotent_for_cap_permits_too(self) -> None:
        from orchestrator.merge_speculation_controller import PermitLedger
        from orchestrator.merge_types import CapPermit

        ledger = PermitLedger(asyncio.Semaphore(1), 1, token_factory=CapPermit)
        p = await ledger.acquire()

        ledger.release(p)
        ledger.release(p)  # second release -- must not raise or over-release

        assert p.released is True
        assert p not in ledger.live
        assert ledger.slot_available == 1
        assert ledger.slot_available + len(ledger.live) == ledger.depth

    async def test_release_of_a_never_acquired_cap_permit_raises(self) -> None:
        from orchestrator.merge_speculation_controller import PermitLedger
        from orchestrator.merge_types import CapPermit

        ledger = PermitLedger(asyncio.Semaphore(1), 1, token_factory=CapPermit)
        stranger = CapPermit()

        with pytest.raises(AssertionError):
            ledger.release(stranger)

    async def test_default_token_factory_still_issues_spec_permits(self) -> None:
        """The existing ``PermitLedger(slot, depth)`` call (no token_factory)
        must keep issuing SpecPermit unchanged -- the default factory is
        additive, not a behaviour change for the existing
        ``_speculation_ledger`` construction."""
        from orchestrator.merge_speculation_controller import PermitLedger
        from orchestrator.merge_types import SpecPermit

        ledger = PermitLedger(asyncio.Semaphore(1), 1)

        p = await ledger.acquire()

        assert isinstance(p, SpecPermit)


# ---------------------------------------------------------------------------
# step-7 RED / step-8 GREEN: permit storage slot on RealMergeItem / DecidedItem
# / InflightEntry
# ---------------------------------------------------------------------------


class TestPermitStorageSlot:
    """RealMergeItem, DecidedItem, and InflightEntry each accept an optional
    ``permit`` keyword (a :class:`SpecPermit` or ``None``) (task 2159 step-7).

    RED until step-8 GREEN adds ``permit: SpecPermit | None = None`` as the
    trailing field of all three dataclasses. eta populates/reads this slot;
    zeta only establishes the storage contract, exercised here via minimal,
    git-free construction (mirrors test_merge_types_invariants.py's
    _real_kwargs/_decided_kwargs convention).
    """

    def test_real_merge_item_carries_a_permit(self) -> None:
        from pathlib import Path
        from unittest.mock import MagicMock

        from orchestrator.git_ops import MergeResult
        from orchestrator.merge_types import RealMergeItem, SpecPermit

        p = SpecPermit()
        item = RealMergeItem(
            request=MagicMock(),
            merge_result=MergeResult(success=True, merge_commit='deadbeef'),
            merge_wt=Path('/fake/merge-wt'),
            base_sha='aabbccdd',
            speculative=False,
            permit=p,
        )

        assert item.permit is p

    def test_real_merge_item_permit_defaults_to_none(self) -> None:
        from pathlib import Path
        from unittest.mock import MagicMock

        from orchestrator.git_ops import MergeResult
        from orchestrator.merge_types import RealMergeItem

        item = RealMergeItem(
            request=MagicMock(),
            merge_result=MergeResult(success=True, merge_commit='deadbeef'),
            merge_wt=Path('/fake/merge-wt'),
            base_sha='aabbccdd',
            speculative=False,
        )

        assert item.permit is None

    def test_decided_item_carries_a_permit(self) -> None:
        from unittest.mock import MagicMock

        from orchestrator.merge_types import DecidedItem, MergeOutcome, SpecPermit

        p = SpecPermit()
        item = DecidedItem(
            request=MagicMock(),
            base_sha='aabbccdd',
            speculative=False,
            immediate_outcome=MergeOutcome('blocked', reason='test'),
            permit=p,
        )

        assert item.permit is p

    def test_decided_item_permit_defaults_to_none(self) -> None:
        from unittest.mock import MagicMock

        from orchestrator.merge_types import DecidedItem, MergeOutcome

        item = DecidedItem(
            request=MagicMock(),
            base_sha='aabbccdd',
            speculative=False,
            immediate_outcome=MergeOutcome('blocked', reason='test'),
        )

        assert item.permit is None

    def test_inflight_entry_carries_a_permit(self) -> None:
        from unittest.mock import MagicMock

        from orchestrator.merge_types import DecidedItem, InflightEntry, MergeOutcome, SpecPermit

        p = SpecPermit()
        decided_item = DecidedItem(
            request=MagicMock(),
            base_sha='aabbccdd',
            speculative=False,
            immediate_outcome=MergeOutcome('blocked', reason='test'),
        )
        entry = InflightEntry(
            item=decided_item,
            lease=None,
            verify_task=None,
            merge_wt=None,
            was_speculative=False,
            phase='passthrough',
            permit=p,
        )

        assert entry.permit is p

    def test_inflight_entry_permit_defaults_to_none(self) -> None:
        from unittest.mock import MagicMock

        from orchestrator.merge_types import DecidedItem, InflightEntry, MergeOutcome

        decided_item = DecidedItem(
            request=MagicMock(),
            base_sha='aabbccdd',
            speculative=False,
            immediate_outcome=MergeOutcome('blocked', reason='test'),
        )
        entry = InflightEntry(
            item=decided_item,
            lease=None,
            verify_task=None,
            merge_wt=None,
            was_speculative=False,
            phase='passthrough',
        )

        assert entry.permit is None
