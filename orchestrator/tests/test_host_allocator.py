"""Tests for HostAllocator + HostLease (β: per-host slots, prefer-local-when-free, cancel-aware release).

Steps covered in this file:
  step-1  RED  — construction + introspection
  step-3  RED  — prefer-local, remote overflow, ≤1/host, release
  step-5  RED  — quarantine_and_release
  step-7  RED  — cancel_and_release happy path
  step-9  RED  — cancel_and_release cancel-FAIL + PARK + probe
  1795/step-1  RED  — clear_quarantine + quarantined_remote_runners
  2369/step-1  RED  — cancel_and_release idempotency (double-release, null-lease)
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Fake runners (no real I/O, no real subprocess)
# ---------------------------------------------------------------------------


class _FakeRemoteRunner:
    """Minimal RemoteRunner-like for HostAllocator tests — no real I/O."""

    is_local: bool = False

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeLocalRunner:
    """Minimal LocalRunner-like returned by the fake local_factory."""

    is_local: bool = True
    name: str = 'local'


# ---------------------------------------------------------------------------
# step-1 RED: HostLease + HostAllocator construction & introspection
# ---------------------------------------------------------------------------


class TestHostAllocatorConstructionSync:
    """HostAllocator construction and sync introspection."""

    def _make_allocator(self, *, quarantine=None):
        from orchestrator.verify_runner import HostAllocator

        remote_a = _FakeRemoteRunner('remoteA')
        remote_b = _FakeRemoteRunner('remoteB')
        q = quarantine if quarantine is not None else set()
        return HostAllocator([remote_a, remote_b], quarantine=q)

    def test_host_names_includes_local_then_remotes(self):
        """host_names == ['local', 'remoteA', 'remoteB'] (local first)."""
        alloc = self._make_allocator()
        assert alloc.host_names == ['local', 'remoteA', 'remoteB']

    def test_free_host_count_is_three_after_construction(self):
        """All three slots are free at construction → free_host_count() == 3."""
        alloc = self._make_allocator()
        assert alloc.free_host_count() == 3

    def test_is_busy_false_for_all_hosts(self):
        """is_busy(name) returns False for all hosts at construction."""
        alloc = self._make_allocator()
        for name in ['local', 'remoteA', 'remoteB']:
            assert alloc.is_busy(name) is False

    def test_host_lease_is_frozen_dataclass(self):
        """HostLease is a frozen dataclass — mutation raises FrozenInstanceError."""
        import dataclasses

        from orchestrator.verify_runner import HostLease

        lease = HostLease(name='local', runner=_FakeLocalRunner(), is_local=True)
        with pytest.raises(dataclasses.FrozenInstanceError):
            lease.name = 'other'  # type: ignore[misc]

    def test_host_names_empty_remotes(self):
        """HostAllocator with no remotes: host_names == ['local']."""
        from orchestrator.verify_runner import HostAllocator
        alloc = HostAllocator([], quarantine=set())
        assert alloc.host_names == ['local']
        assert alloc.free_host_count() == 1


@pytest.mark.asyncio
class TestHostAllocatorConstructionAsync:
    """HostAllocator async acquire at construction time."""

    def _make_allocator(self, *, quarantine=None):
        from orchestrator.verify_runner import HostAllocator

        remote_a = _FakeRemoteRunner('remoteA')
        remote_b = _FakeRemoteRunner('remoteB')
        q = quarantine if quarantine is not None else set()
        return HostAllocator([remote_a, remote_b], quarantine=q)

    def _local_factory(self):
        return _FakeLocalRunner()

    async def test_acquire_when_all_free_returns_local_lease(self):
        """acquire(local_factory) with all slots free returns a local HostLease."""
        from orchestrator.verify_runner import HostLease
        alloc = self._make_allocator()
        lease = await alloc.acquire(self._local_factory)
        assert lease is not None
        assert isinstance(lease, HostLease)
        assert lease.is_local is True
        assert lease.name == 'local'

    async def test_acquired_local_lease_runner_from_factory(self):
        """The lease.runner is the object returned by local_factory."""
        sentinel = _FakeLocalRunner()

        def factory():
            return sentinel

        alloc = self._make_allocator()
        lease = await alloc.acquire(factory)
        assert lease is not None
        assert lease.runner is sentinel

    async def test_local_busy_after_acquire(self):
        """After acquire, the local slot is marked busy."""
        alloc = self._make_allocator()
        await alloc.acquire(self._local_factory)
        assert alloc.is_busy('local') is True
        assert alloc.free_host_count() == 2


# ---------------------------------------------------------------------------
# step-3 RED: prefer-local + remote overflow + ≤1/host + release
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHostAllocatorAcquireRelease:
    """Slot state machine: prefer-local → remote overflow → ≤1/host → release."""

    def _make_allocator(self):
        from orchestrator.verify_runner import HostAllocator
        remote_a = _FakeRemoteRunner('remoteA')
        remote_b = _FakeRemoteRunner('remoteB')
        return HostAllocator([remote_a, remote_b], quarantine=set())

    def _local_factory(self):
        return _FakeLocalRunner()

    async def test_second_acquire_returns_remote_when_local_busy(self):
        """With local busy, acquire() returns the first remote lease."""
        from orchestrator.verify_runner import HostLease
        alloc = self._make_allocator()
        local_lease = await alloc.acquire(self._local_factory)
        assert local_lease is not None and local_lease.is_local

        remote_lease = await alloc.acquire(self._local_factory)
        assert remote_lease is not None
        assert isinstance(remote_lease, HostLease)
        assert remote_lease.is_local is False
        assert remote_lease.name == 'remoteA'

    async def test_third_acquire_returns_second_remote(self):
        """local busy + remoteA busy → third acquire returns remoteB."""
        alloc = self._make_allocator()
        await alloc.acquire(self._local_factory)           # local
        await alloc.acquire(self._local_factory)           # remoteA
        third = await alloc.acquire(self._local_factory)
        assert third is not None
        assert third.name == 'remoteB'

    async def test_fourth_acquire_returns_none_all_busy(self):
        """All slots busy → acquire returns None (≤1 in-flight per host)."""
        alloc = self._make_allocator()
        await alloc.acquire(self._local_factory)           # local
        await alloc.acquire(self._local_factory)           # remoteA
        await alloc.acquire(self._local_factory)           # remoteB
        fourth = await alloc.acquire(self._local_factory)
        assert fourth is None

    async def test_free_host_count_tracks_busy_slots(self):
        """free_host_count decrements as slots are acquired."""
        alloc = self._make_allocator()
        assert alloc.free_host_count() == 3
        l1 = await alloc.acquire(self._local_factory)
        assert l1 is not None
        assert alloc.free_host_count() == 2
        await alloc.acquire(self._local_factory)
        assert alloc.free_host_count() == 1
        await alloc.acquire(self._local_factory)
        assert alloc.free_host_count() == 0

        # release local → local slot freed → prefer-local again
        await alloc.release(l1)
        assert alloc.free_host_count() == 1

    async def test_release_local_then_acquire_returns_local(self):
        """After releasing the local slot, the next acquire again returns local."""
        alloc = self._make_allocator()
        local_lease = await alloc.acquire(self._local_factory)
        assert local_lease is not None
        await alloc.acquire(self._local_factory)
        await alloc.release(local_lease)

        # Local slot freed — next acquire should prefer local
        new_lease = await alloc.acquire(self._local_factory)
        assert new_lease is not None
        assert new_lease.is_local is True
        assert new_lease.name == 'local'

    async def test_release_is_idempotent(self):
        """Double-release is a no-op — no error, slot stays FREE."""
        alloc = self._make_allocator()
        lease = await alloc.acquire(self._local_factory)
        assert lease is not None
        await alloc.release(lease)
        await alloc.release(lease)  # idempotent
        assert alloc.free_host_count() == 3


# ---------------------------------------------------------------------------
# step-5 RED: quarantine_and_release
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHostAllocatorQuarantine:
    """quarantine_and_release adds the host to the shared set + frees the slot."""

    def _make_allocator(self, *, shared_quarantine=None):
        from orchestrator.verify_runner import HostAllocator
        remote_a = _FakeRemoteRunner('remoteA')
        remote_b = _FakeRemoteRunner('remoteB')
        q = shared_quarantine if shared_quarantine is not None else set()
        return HostAllocator([remote_a, remote_b], quarantine=q)

    def _local_factory(self):
        return _FakeLocalRunner()

    async def test_quarantine_adds_to_shared_set(self):
        """quarantine_and_release(remoteA_lease) adds 'remoteA' to the shared quarantine set."""
        shared_q: set[str] = set()
        alloc = self._make_allocator(shared_quarantine=shared_q)
        # Acquire local first so we can get remoteA
        await alloc.acquire(self._local_factory)
        remote_lease = await alloc.acquire(self._local_factory)
        assert remote_lease is not None and remote_lease.name == 'remoteA'

        await alloc.quarantine_and_release(remote_lease)
        assert 'remoteA' in shared_q

    async def test_quarantine_frees_the_slot(self):
        """After quarantine_and_release, the remote slot is freed (count increments)."""
        alloc = self._make_allocator()
        await alloc.acquire(self._local_factory)           # local
        remote_lease = await alloc.acquire(self._local_factory)   # remoteA
        assert remote_lease is not None
        before = alloc.free_host_count()
        await alloc.quarantine_and_release(remote_lease)
        assert alloc.free_host_count() == before + 1

    async def test_quarantined_host_not_acquired(self):
        """A quarantined remote is skipped by acquire_remote()."""
        shared_q: set[str] = set()
        alloc = self._make_allocator(shared_quarantine=shared_q)
        await alloc.acquire(self._local_factory)           # local
        remote_a_lease = await alloc.acquire(self._local_factory)  # remoteA
        assert remote_a_lease is not None
        await alloc.quarantine_and_release(remote_a_lease)  # quarantine remoteA

        # Next acquire (local still busy) should return remoteB, not remoteA
        next_lease = await alloc.acquire(self._local_factory)
        assert next_lease is not None
        assert next_lease.name == 'remoteB'

    async def test_all_remotes_quarantined_local_busy_returns_none(self):
        """With all remotes quarantined and local busy, acquire returns None."""
        shared_q: set[str] = {'remoteA', 'remoteB'}
        alloc = self._make_allocator(shared_quarantine=shared_q)
        await alloc.acquire(self._local_factory)           # local BUSY
        result = await alloc.acquire(self._local_factory)
        assert result is None

    async def test_quarantine_local_lease_does_not_add_to_set(self):
        """quarantine_and_release on a LOCAL lease only frees the slot — 'local' NOT added to set."""
        shared_q: set[str] = set()
        alloc = self._make_allocator(shared_quarantine=shared_q)
        local_lease = await alloc.acquire(self._local_factory)
        assert local_lease is not None
        await alloc.quarantine_and_release(local_lease)

        assert 'local' not in shared_q
        assert alloc.free_host_count() == 3   # all slots freed

    async def test_preseeded_quarantine_name_not_acquirable(self):
        """A name pre-seeded in the shared quarantine set cannot be acquired."""
        shared_q: set[str] = {'remoteA'}
        alloc = self._make_allocator(shared_quarantine=shared_q)
        # local slot is free; remoteA quarantined; remoteB free
        local_lease = await alloc.acquire(self._local_factory)
        assert local_lease is not None and local_lease.is_local

        # With local busy, remoteA quarantined → next should be remoteB
        next_lease = await alloc.acquire(self._local_factory)
        assert next_lease is not None
        assert next_lease.name == 'remoteB'


# ---------------------------------------------------------------------------
# step-7 RED: cancel_and_release happy path
# ---------------------------------------------------------------------------


class _FakeRemoteRunnerCancellable(_FakeRemoteRunner):
    """Fake remote with async cancel_verify() + probe_clean() support."""

    def __init__(self, name: str, *, cancel_rc: int = 0, probe_sequence=None) -> None:
        super().__init__(name)
        self._cancel_rc = cancel_rc
        self._probe_sequence = list(probe_sequence) if probe_sequence else []
        self._probe_index = 0
        self.cancel_verify_called = 0
        self.probe_clean_called = 0

    async def cancel_verify(self) -> int:
        self.cancel_verify_called += 1
        return self._cancel_rc

    async def probe_clean(self) -> bool:
        self.probe_clean_called += 1
        if self._probe_index < len(self._probe_sequence):
            result = self._probe_sequence[self._probe_index]
            self._probe_index += 1
            return result
        return True   # default: clean


@pytest.mark.asyncio
class TestHostAllocatorCancelRelease:
    """cancel_and_release: happy path — cancel confirms, slot freed, host stays eligible."""

    def _local_factory(self):
        return _FakeLocalRunner()

    async def test_cancel_success_frees_slot(self):
        """cancel_verify() returns 0 → slot freed, returns True."""
        from orchestrator.verify_runner import HostAllocator

        remote_a = _FakeRemoteRunnerCancellable('remoteA', cancel_rc=0)
        alloc = HostAllocator([remote_a], quarantine=set())

        await alloc.acquire(self._local_factory)     # local busy
        remote_lease = await alloc.acquire(self._local_factory)   # remoteA
        assert remote_lease is not None

        result = await alloc.cancel_and_release(remote_lease)
        assert result is True
        # Slot freed — remoteA acquirable again
        assert alloc.is_busy('remoteA') is False

    async def test_cancel_success_host_stays_eligible(self):
        """After cancel+release, the remote host is still acquirable (not quarantined)."""
        from orchestrator.verify_runner import HostAllocator

        remote_a = _FakeRemoteRunnerCancellable('remoteA', cancel_rc=0)
        alloc = HostAllocator([remote_a], quarantine=set())

        await alloc.acquire(self._local_factory)    # local
        remote_lease = await alloc.acquire(self._local_factory)   # remoteA
        assert remote_lease is not None
        await alloc.cancel_and_release(remote_lease)

        # Still local-busy; remoteA should be acquirable
        new_lease = await alloc.acquire(self._local_factory)
        assert new_lease is not None
        assert new_lease.name == 'remoteA'

    async def test_cancel_local_lease_frees_slot(self):
        """cancel_and_release on a LOCAL lease frees the slot without calling ssh."""
        from orchestrator.verify_runner import HostAllocator

        # No cancel_verify on this runner — if called, AttributeError would surface
        alloc = HostAllocator([], quarantine=set())
        local_lease = await alloc.acquire(self._local_factory)
        assert local_lease is not None and local_lease.is_local

        result = await alloc.cancel_and_release(local_lease)
        assert result is True
        assert alloc.is_busy('local') is False


# ---------------------------------------------------------------------------
# step-9 RED: cancel_and_release cancel-FAIL → PARK → probe until clean
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHostAllocatorCancelFail:
    """cancel_and_release cancel-FAIL path: PARK slot, poll pgrep probe, un-park when clean."""

    def _local_factory(self):
        return _FakeLocalRunner()

    async def test_cancel_fail_returns_false(self):
        """cancel_verify() returns 1 → cancel_and_release returns False."""
        from orchestrator.verify_runner import HostAllocator

        remote_a = _FakeRemoteRunnerCancellable('remoteA', cancel_rc=1, probe_sequence=[True])
        alloc = HostAllocator([remote_a], quarantine=set())

        await alloc.acquire(self._local_factory)
        remote_lease = await alloc.acquire(self._local_factory)
        assert remote_lease is not None

        async def noop_sleep(_: float) -> None:
            pass

        result = await alloc.cancel_and_release(remote_lease, sleep=noop_sleep)
        assert result is False

    async def test_cancel_fail_slot_parked_during_probe(self):
        """While parked (probe not yet clean), the slot is non-acquirable."""
        from orchestrator.verify_runner import HostAllocator

        # probe_sequence=[False, True]: first poll not clean, second clean
        probe_calls: list[bool] = []

        class _ProbeTracker(_FakeRemoteRunnerCancellable):
            async def probe_clean(self) -> bool:
                call_count = len(probe_calls)
                result = [False, True][min(call_count, 1)]
                probe_calls.append(result)
                return result

        remote_a = _ProbeTracker('remoteA', cancel_rc=1)
        remote_b = _FakeRemoteRunner('remoteB')
        alloc = HostAllocator([remote_a, remote_b], quarantine=set())

        await alloc.acquire(self._local_factory)
        remote_lease = await alloc.acquire(self._local_factory)  # remoteA
        assert remote_lease is not None

        # We'll run cancel_and_release in a task so we can probe mid-flight
        # Use a sleep that lets us observe the parked state
        acquired_while_parked: list[str | None] = []

        async def checking_sleep(secs: float) -> None:
            # After first probe(False), the slot should be PARKED (not acquirable)
            lease = await alloc.acquire(self._local_factory)
            if lease is None:
                acquired_while_parked.append(None)
            else:
                acquired_while_parked.append(lease.name)
                await alloc.release(lease)

        result = await alloc.cancel_and_release(remote_lease, sleep=checking_sleep)
        assert result is False
        # While parked, remoteA was not acquirable → should have gotten remoteB or None
        assert all(n != 'remoteA' for n in acquired_while_parked)

    async def test_cancel_fail_slot_freed_after_probe_clean(self):
        """After probe returns True, the slot is un-parked and freed (acquirable again)."""
        from orchestrator.verify_runner import HostAllocator

        remote_a = _FakeRemoteRunnerCancellable('remoteA', cancel_rc=1, probe_sequence=[False, True])
        alloc = HostAllocator([remote_a], quarantine=set())

        await alloc.acquire(self._local_factory)
        remote_lease = await alloc.acquire(self._local_factory)
        assert remote_lease is not None

        async def noop_sleep(_: float) -> None:
            pass

        await alloc.cancel_and_release(remote_lease, sleep=noop_sleep)

        # After probe_clean returns True: un-parked + freed
        assert alloc.is_busy('remoteA') is False
        # And it should be acquirable
        new_lease = await alloc.acquire(self._local_factory)
        assert new_lease is not None
        assert new_lease.name == 'remoteA'

    async def test_cancel_fail_bounded_max_attempts_stays_parked(self):
        """probe_clean always False with bounded max_attempts → slot stays PARKED."""
        from orchestrator.verify_runner import HostAllocator

        remote_a = _FakeRemoteRunnerCancellable('remoteA', cancel_rc=1, probe_sequence=[False] * 20)
        alloc = HostAllocator([remote_a], quarantine=set())

        await alloc.acquire(self._local_factory)
        remote_lease = await alloc.acquire(self._local_factory)
        assert remote_lease is not None

        async def noop_sleep(_: float) -> None:
            pass

        result = await alloc.cancel_and_release(remote_lease, sleep=noop_sleep, max_attempts=2)
        assert result is False
        # Slot still held (PARKED) — not freed
        assert alloc.is_busy('remoteA') is True


# ---------------------------------------------------------------------------
# 1795/step-1 RED: clear_quarantine + quarantined_remote_runners
# ---------------------------------------------------------------------------


class TestHostAllocatorClearQuarantine:
    """HostAllocator.clear_quarantine and quarantined_remote_runners sync tests (task 1795)."""

    def _make_allocator(self, *, shared_quarantine=None):
        from orchestrator.verify_runner import HostAllocator
        remote_a = _FakeRemoteRunner('remoteA')
        remote_b = _FakeRemoteRunner('remoteB')
        q = shared_quarantine if shared_quarantine is not None else set()
        return HostAllocator([remote_a, remote_b], quarantine=q)

    # --- clear_quarantine ---

    def test_clear_quarantine_removes_name_from_shared_set(self):
        """clear_quarantine(name) discards the name from the shared quarantine set."""
        shared_q: set[str] = {'remoteA'}
        alloc = self._make_allocator(shared_quarantine=shared_q)
        alloc.clear_quarantine('remoteA')
        assert 'remoteA' not in shared_q

    def test_clear_quarantine_is_idempotent_absent_name(self):
        """clear_quarantine on a name not in quarantine is a no-op (no error)."""
        shared_q: set[str] = set()
        alloc = self._make_allocator(shared_quarantine=shared_q)
        # Should not raise
        alloc.clear_quarantine('remoteA')
        alloc.clear_quarantine('remoteA')  # second call — still no error
        assert shared_q == set()

    def test_clear_quarantine_is_idempotent_after_clearing(self):
        """clear_quarantine twice on the same name is safe."""
        shared_q: set[str] = {'remoteA'}
        alloc = self._make_allocator(shared_quarantine=shared_q)
        alloc.clear_quarantine('remoteA')
        alloc.clear_quarantine('remoteA')  # second call — idempotent
        assert 'remoteA' not in shared_q

    # --- quarantined_remote_runners ---

    def test_quarantined_remote_runners_empty_when_none_quarantined(self):
        """quarantined_remote_runners() returns [] when nothing is quarantined."""
        alloc = self._make_allocator()
        result = alloc.quarantined_remote_runners()
        assert result == []

    def test_quarantined_remote_runners_returns_quarantined_remotes(self):
        """quarantined_remote_runners() returns (name, runner) for each quarantined remote."""
        shared_q: set[str] = {'remoteA'}
        alloc = self._make_allocator(shared_quarantine=shared_q)
        result = alloc.quarantined_remote_runners()
        assert len(result) == 1
        name, runner = result[0]
        assert name == 'remoteA'
        assert runner.name == 'remoteA'  # the actual runner object

    def test_quarantined_remote_runners_excludes_non_quarantined(self):
        """quarantined_remote_runners() does not include non-quarantined remotes."""
        shared_q: set[str] = {'remoteA'}
        alloc = self._make_allocator(shared_quarantine=shared_q)
        result = alloc.quarantined_remote_runners()
        names = [n for n, _ in result]
        assert 'remoteB' not in names

    def test_quarantined_remote_runners_excludes_local(self):
        """quarantined_remote_runners() never includes the local host even if named in the set."""
        # The local host is not in _remote_runners so it should never appear
        shared_q: set[str] = {'local', 'remoteA'}
        alloc = self._make_allocator(shared_quarantine=shared_q)
        result = alloc.quarantined_remote_runners()
        names = [n for n, _ in result]
        assert 'local' not in names
        assert 'remoteA' in names

    def test_quarantined_remote_runners_returns_all_quarantined_when_multiple(self):
        """All quarantined remotes appear in the result when multiple are quarantined."""
        shared_q: set[str] = {'remoteA', 'remoteB'}
        alloc = self._make_allocator(shared_quarantine=shared_q)
        result = alloc.quarantined_remote_runners()
        names = sorted(n for n, _ in result)
        assert names == ['remoteA', 'remoteB']


@pytest.mark.asyncio
class TestHostAllocatorClearQuarantineAsync:
    """HostAllocator.clear_quarantine async tests (task 1795)."""

    def _make_allocator(self, *, shared_quarantine=None):
        from orchestrator.verify_runner import HostAllocator
        remote_a = _FakeRemoteRunner('remoteA')
        remote_b = _FakeRemoteRunner('remoteB')
        q = shared_quarantine if shared_quarantine is not None else set()
        return HostAllocator([remote_a, remote_b], quarantine=q)

    def _local_factory(self):
        return _FakeLocalRunner()

    async def test_clear_quarantine_re_enables_host_for_acquire_remote(self):
        """After quarantine_and_release then clear_quarantine, acquire_remote returns the host."""
        shared_q: set[str] = set()
        alloc = self._make_allocator(shared_quarantine=shared_q)
        # Fill local so overflow goes to remote
        await alloc.acquire(self._local_factory)       # local busy
        remote_lease = await alloc.acquire(self._local_factory)   # remoteA
        assert remote_lease is not None and remote_lease.name == 'remoteA'

        await alloc.quarantine_and_release(remote_lease)
        assert 'remoteA' in shared_q
        # remoteA is quarantined — slot is FREE but blocked
        assert alloc.is_busy('remoteA') is False

        alloc.clear_quarantine('remoteA')
        assert 'remoteA' not in shared_q

        # acquire_remote should now return remoteA (its slot is FREE + not quarantined)
        re_acquired = alloc.acquire_remote()
        assert re_acquired is not None
        assert re_acquired.name == 'remoteA'


# ---------------------------------------------------------------------------
# 2369/step-1 RED: cancel_and_release idempotency — already-released (FREE) slot guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHostAllocatorCancelReleaseIdempotent:
    """cancel_and_release must be idempotent: a repeat call on an already-FREE slot
    is a no-op that does NOT re-issue the remote cancel RPC (task 2369).
    """

    def _local_factory(self):
        return _FakeLocalRunner()

    async def test_double_release_does_not_reissue_cancel_rpc(self):
        """A second cancel_and_release on an already-released remote lease must not
        re-issue cancel_verify() — the slot is already FREE, so there is nothing to cancel.
        """
        from orchestrator.verify_runner import HostAllocator

        remote_a = _FakeRemoteRunnerCancellable('remoteA', cancel_rc=0)
        alloc = HostAllocator([remote_a], quarantine=set())

        await alloc.acquire(self._local_factory)                  # local busy (force overflow)
        remote_lease = await alloc.acquire(self._local_factory)   # remoteA
        assert remote_lease is not None

        first = await alloc.cancel_and_release(remote_lease)
        assert first is True
        assert remote_a.cancel_verify_called == 1
        assert alloc.is_busy('remoteA') is False

        # Second call on the same (now-stale) lease — slot is already FREE.
        second = await alloc.cancel_and_release(remote_lease)
        assert second is True
        assert remote_a.cancel_verify_called == 1   # RPC NOT re-issued
        assert alloc.is_busy('remoteA') is False

    async def test_double_release_nonzero_recancel_does_not_park(self):
        """Even if a re-issued cancel would fail (rc != 0), the already-FREE guard
        must prevent the re-issue entirely so a healthy, released slot is never PARKed.
        """
        from orchestrator.verify_runner import HostAllocator

        class _FirstCancelCleanThenFailing(_FakeRemoteRunnerCancellable):
            """cancel_verify() returns 0 on the first call, 1 on every call thereafter."""

            async def cancel_verify(self) -> int:
                self.cancel_verify_called += 1
                return 0 if self.cancel_verify_called == 1 else 1

        remote_a = _FirstCancelCleanThenFailing('remoteA', probe_sequence=[False] * 20)
        alloc = HostAllocator([remote_a], quarantine=set())

        await alloc.acquire(self._local_factory)
        remote_lease = await alloc.acquire(self._local_factory)   # remoteA
        assert remote_lease is not None

        async def noop_sleep(_: float) -> None:
            pass

        first = await alloc.cancel_and_release(remote_lease, sleep=noop_sleep)
        assert first is True
        assert alloc.is_busy('remoteA') is False

        # A real re-issue here would return rc=1 and PARK the slot. The guard must
        # short-circuit before the RPC, so the slot stays FREE and probe_clean is
        # never invoked.
        second = await alloc.cancel_and_release(remote_lease, sleep=noop_sleep)
        assert second is True
        assert alloc.is_busy('remoteA') is False
        assert remote_a.probe_clean_called == 0

    async def test_cancel_and_release_none_lease_is_noop(self):
        """cancel_and_release(None) is a defensive no-op: returns True, raises
        nothing, and never touches the RPC path.
        """
        from orchestrator.verify_runner import HostAllocator

        remote_a = _FakeRemoteRunnerCancellable('remoteA', cancel_rc=0)
        alloc = HostAllocator([remote_a], quarantine=set())

        result = await alloc.cancel_and_release(None)
        assert result is True
        assert remote_a.cancel_verify_called == 0

    async def test_double_release_local_lease_is_noop(self):
        """A second cancel_and_release on an already-released LOCAL lease is
        also a no-op: the FREE-slot guard sits above the is_local branch, so
        a repeat call on a FREE local slot returns True without raising.
        """
        from orchestrator.verify_runner import HostAllocator

        alloc = HostAllocator([], quarantine=set())
        local_lease = await alloc.acquire(self._local_factory)
        assert local_lease is not None and local_lease.is_local

        first = await alloc.cancel_and_release(local_lease)
        assert first is True
        assert alloc.is_busy('local') is False

        # Second call on the same (now-stale) local lease — slot already FREE.
        second = await alloc.cancel_and_release(local_lease)
        assert second is True
        assert alloc.is_busy('local') is False


# ---------------------------------------------------------------------------
# 3275/step-1 RED: host_states() + is_quarantined() read-only state accessors
# ---------------------------------------------------------------------------


_STATE_KEYS = {'name', 'is_local', 'slot_state', 'quarantined'}


@pytest.mark.asyncio
class TestHostAllocatorStateAccessors:
    """host_states() / is_quarantined(): the sanctioned read path for snapshot().

    Task 3275: SpeculativeMergeWorker.snapshot() must be able to report per-host
    slot state and quarantine membership WITHOUT reaching into the allocator's
    private ``_slots`` / ``_quarantine``.  Both accessors are pure reads.

    RED until 3275/step-2 adds the two methods (AttributeError before that).
    """

    def _make_allocator(self, *, quarantine=None):
        from orchestrator.verify_runner import HostAllocator

        remote_a = _FakeRemoteRunner('remoteA')
        remote_b = _FakeRemoteRunner('remoteB')
        q = quarantine if quarantine is not None else set()
        return HostAllocator([remote_a, remote_b], quarantine=q)

    def _local_factory(self):
        return _FakeLocalRunner()

    def _by_name(self, alloc) -> dict:
        return {h['name']: h for h in alloc.host_states()}

    # -- shape / order / schema ------------------------------------------------

    async def test_host_states_order_matches_host_names(self):
        """One dict per managed host, local first then remotes in declaration order."""
        alloc = self._make_allocator()
        states = alloc.host_states()
        assert [h['name'] for h in states] == ['local', 'remoteA', 'remoteB']
        # Same ordering contract as the pre-existing host_names property.
        assert [h['name'] for h in states] == alloc.host_names

    async def test_host_states_uniform_schema(self):
        """Every entry carries exactly {name, is_local, slot_state, quarantined}."""
        alloc = self._make_allocator()
        for entry in alloc.host_states():
            assert set(entry) == _STATE_KEYS, (
                f'Unexpected key set for {entry.get("name")!r}: {sorted(entry)}'
            )

    async def test_host_states_fresh_all_free_unquarantined(self):
        """Fresh allocator: every slot free, nothing quarantined."""
        alloc = self._make_allocator()
        for entry in alloc.host_states():
            assert entry['slot_state'] == 'free', entry
            assert entry['quarantined'] is False, entry

    async def test_host_states_is_local_only_for_local(self):
        """is_local is True for 'local' and False for every remote."""
        states = self._by_name(self._make_allocator())
        assert states['local']['is_local'] is True
        assert states['remoteA']['is_local'] is False
        assert states['remoteB']['is_local'] is False

    async def test_host_states_empty_remotes_single_local_entry(self):
        """An allocator with no remotes reports exactly one 'local' entry."""
        from orchestrator.verify_runner import HostAllocator

        alloc = HostAllocator([], quarantine=set())
        states = alloc.host_states()
        assert len(states) == 1
        assert states[0]['name'] == 'local'
        assert states[0]['is_local'] is True
        assert states[0]['slot_state'] == 'free'

    # -- slot_state transitions ------------------------------------------------

    async def test_slot_state_busy_after_acquire_remote(self):
        """acquire_remote() flips that host's slot_state to 'busy' (others stay free)."""
        alloc = self._make_allocator()
        lease = alloc.acquire_remote()
        assert lease is not None and lease.name == 'remoteA'

        states = self._by_name(alloc)
        assert states['remoteA']['slot_state'] == 'busy'
        assert states['local']['slot_state'] == 'free'
        assert states['remoteB']['slot_state'] == 'free'

    async def test_slot_state_free_again_after_release(self):
        """release(lease) returns the slot to 'free'."""
        alloc = self._make_allocator()
        lease = alloc.acquire_remote()
        assert lease is not None
        assert self._by_name(alloc)['remoteA']['slot_state'] == 'busy'

        await alloc.release(lease)
        assert self._by_name(alloc)['remoteA']['slot_state'] == 'free'

    async def test_slot_state_busy_after_acquire_local(self):
        """acquire_local(factory) flips 'local' to 'busy'."""
        alloc = self._make_allocator()
        lease = alloc.acquire_local(lambda: _FakeLocalRunner())
        assert lease is not None and lease.is_local

        assert self._by_name(alloc)['local']['slot_state'] == 'busy'

    async def test_slot_state_parked_after_cancel_fail(self):
        """A cancel-FAIL that never probes clean leaves the slot 'parked'.

        Drives the same path as TestHostAllocatorCancelFail's bounded-attempts
        case: cancel_verify() returns 1 and probe_clean() never returns True,
        so with max_attempts=2 the slot stays PARKED (held, non-acquirable).
        """
        from orchestrator.verify_runner import HostAllocator

        remote_a = _FakeRemoteRunnerCancellable(
            'remoteA', cancel_rc=1, probe_sequence=[False] * 20,
        )
        alloc = HostAllocator([remote_a], quarantine=set())

        lease = alloc.acquire_remote()
        assert lease is not None and lease.name == 'remoteA'

        async def noop_sleep(_: float) -> None:
            pass

        result = await alloc.cancel_and_release(lease, sleep=noop_sleep, max_attempts=2)
        assert result is False
        # Pre-existing introspection agrees the slot is still held...
        assert alloc.is_busy('remoteA') is True
        # ...and host_states() distinguishes PARKED from plain BUSY.
        assert self._by_name(alloc)['remoteA']['slot_state'] == 'parked'

    async def test_slot_state_uses_lowercase_wire_vocabulary(self):
        """slot_state is the lowercase wire spelling, never the _SLOT_* constants."""
        alloc = self._make_allocator()
        alloc.acquire_remote()
        values = {h['slot_state'] for h in alloc.host_states()}
        assert values <= {'free', 'busy', 'parked'}, values
        # Explicitly NOT the internal uppercase constants.
        assert not any(v.isupper() for v in values), values

    # -- is_quarantined --------------------------------------------------------

    async def test_is_quarantined_false_on_fresh_allocator(self):
        """Nothing is quarantined at construction."""
        alloc = self._make_allocator()
        for name in ['local', 'remoteA', 'remoteB']:
            assert alloc.is_quarantined(name) is False

    async def test_is_quarantined_true_after_quarantine_and_release(self):
        """quarantine_and_release(remote_lease) flips is_quarantined + host_states."""
        alloc = self._make_allocator()
        lease = alloc.acquire_remote()
        assert lease is not None and lease.name == 'remoteA'

        await alloc.quarantine_and_release(lease)

        assert alloc.is_quarantined('remoteA') is True
        assert self._by_name(alloc)['remoteA']['quarantined'] is True
        # The slot itself was freed by the same call.
        assert self._by_name(alloc)['remoteA']['slot_state'] == 'free'
        # Untouched hosts are unaffected.
        assert alloc.is_quarantined('remoteB') is False

    async def test_shared_quarantine_set_is_honoured_by_reference(self):
        """Mutating the CALLER's set flips both accessors — no allocator call needed.

        This is the DriftDetector / land-time-cross-check path: those writers add
        to the worker's ``_runner_quarantine`` directly, never through the
        allocator, so both read paths must see it.
        """
        q: set[str] = set()
        alloc = self._make_allocator(quarantine=q)
        assert alloc.is_quarantined('remoteB') is False

        q.add('remoteB')

        assert alloc.is_quarantined('remoteB') is True
        assert self._by_name(alloc)['remoteB']['quarantined'] is True
        assert self._by_name(alloc)['remoteA']['quarantined'] is False

    async def test_clear_quarantine_flips_back_to_false(self):
        """clear_quarantine(name) returns is_quarantined/host_states to False."""
        q: set[str] = {'remoteA'}
        alloc = self._make_allocator(quarantine=q)
        assert alloc.is_quarantined('remoteA') is True

        alloc.clear_quarantine('remoteA')

        assert alloc.is_quarantined('remoteA') is False
        assert self._by_name(alloc)['remoteA']['quarantined'] is False

    async def test_is_quarantined_unmanaged_name_returns_false(self):
        """An unmanaged host name returns False rather than raising."""
        alloc = self._make_allocator()
        assert alloc.is_quarantined('no-such-host') is False


# ---------------------------------------------------------------------------
# 3043/step-1 RED: is_parked() + remote_runner() strand-introspection accessors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHostAllocatorStrandAccessors:
    """is_parked() / remote_runner(): the primitives strand detection needs.

    Task 3043.  Two states the existing accessors structurally cannot answer:

    - A ``cancel_and_release`` against an unreachable host leaves the slot
      PARKED and non-acquirable, yet does NOT add the host to ``_quarantine``
      — so ``quarantined_remote_runners()`` never yields it and the auto-reprobe
      path cannot even consider it.  ``is_parked`` is the cheap per-host
      predicate the merge_queue strand check needs on the release hot path.
    - Reprobe candidacy becomes TRACKER-driven, so it must resolve a runner
      object for a host that is tracked but NOT quarantined —
      ``quarantined_remote_runners()`` structurally cannot supply that.

    RED until 3043/step-2 adds the two methods (AttributeError before that).
    """

    def _make_allocator(self, *, quarantine=None):
        from orchestrator.verify_runner import HostAllocator

        remote_a = _FakeRemoteRunner('remoteA')
        remote_b = _FakeRemoteRunner('remoteB')
        q = quarantine if quarantine is not None else set()
        return HostAllocator([remote_a, remote_b], quarantine=q)

    def _local_factory(self):
        return _FakeLocalRunner()

    def _by_name(self, alloc) -> dict:
        return {h['name']: h for h in alloc.host_states()}

    async def _make_parked_allocator(self):
        """Return (alloc, remote_a) with remoteA's slot left PARKED.

        Same construction as ``test_cancel_fail_bounded_max_attempts_stays_parked``
        (test_host_allocator.py:485): cancel_verify() returns rc != 0 and
        probe_clean() never returns True, so with bounded max_attempts the slot
        stays PARKED — held, non-acquirable, and NOT in the quarantine set.
        """
        from orchestrator.verify_runner import HostAllocator

        remote_a = _FakeRemoteRunnerCancellable(
            'remoteA', cancel_rc=1, probe_sequence=[False] * 20,
        )
        remote_b = _FakeRemoteRunner('remoteB')
        alloc = HostAllocator([remote_a, remote_b], quarantine=set())

        lease = alloc.acquire_remote()
        assert lease is not None and lease.name == 'remoteA'

        async def noop_sleep(_: float) -> None:
            pass

        result = await alloc.cancel_and_release(lease, sleep=noop_sleep, max_attempts=2)
        assert result is False
        return alloc, remote_a

    # -- is_parked -------------------------------------------------------------

    async def test_is_parked_true_for_cancel_fail_parked_slot(self):
        """A slot left PARKED by the bounded cancel-fail path reports is_parked True."""
        alloc, _remote_a = await self._make_parked_allocator()
        assert alloc.is_parked('remoteA') is True

    async def test_is_parked_false_for_free_slot(self):
        """A fresh FREE slot is not parked."""
        alloc = self._make_allocator()
        assert alloc.is_parked('remoteA') is False
        assert alloc.is_parked('remoteB') is False

    async def test_is_parked_false_for_busy_slot(self):
        """A BUSY slot (acquired, still verifying) is not parked."""
        alloc = self._make_allocator()
        lease = alloc.acquire_remote()
        assert lease is not None and lease.name == 'remoteA'
        assert alloc.is_parked('remoteA') is False

    async def test_is_parked_false_for_local_host(self):
        """The local slot is never parked — local is the trust anchor."""
        alloc = self._make_allocator()
        assert alloc.is_parked('local') is False
        alloc.acquire_local(self._local_factory)
        assert alloc.is_parked('local') is False

    async def test_is_parked_unknown_name_returns_false_not_keyerror(self):
        """An unmanaged host name returns False rather than raising KeyError."""
        alloc = self._make_allocator()
        assert alloc.is_parked('no-such-host') is False

    async def test_is_parked_agrees_with_host_states_slot_state(self):
        """is_parked(n) can never drift from host_states()'s 'parked' wire spelling.

        Checked over BOTH fixtures — one with a PARKED slot and one with none —
        so the two readers of ``_slots`` are pinned to agree for every host.
        """
        parked_alloc, _ = await self._make_parked_allocator()
        plain_alloc = self._make_allocator()
        plain_alloc.acquire_remote()                       # remoteA busy
        plain_alloc.acquire_local(self._local_factory)     # local busy

        # Fixture preconditions, asserted ONCE against the fixture each applies
        # to — so the agreement loop below is the test's only real content.
        assert parked_alloc.is_parked('remoteA') is True
        assert 'parked' not in {
            h['slot_state'] for h in self._by_name(plain_alloc).values()
        }

        for alloc in (parked_alloc, plain_alloc):
            states = self._by_name(alloc)
            for name, entry in states.items():
                assert alloc.is_parked(name) is (entry['slot_state'] == 'parked'), (
                    f'{name}: is_parked={alloc.is_parked(name)} vs '
                    f'slot_state={entry["slot_state"]!r}'
                )

    async def test_parked_host_is_not_quarantined(self):
        """The strand shape: PARKED yet absent from the quarantine set.

        This is why ``quarantined_remote_runners()`` cannot see a stranded host
        and why ``is_parked`` is needed at all.
        """
        alloc, _ = await self._make_parked_allocator()
        assert alloc.is_parked('remoteA') is True
        assert alloc.is_quarantined('remoteA') is False
        assert alloc.quarantined_remote_runners() == []

    # -- remote_runner ---------------------------------------------------------

    async def test_remote_runner_returns_identical_object_when_free(self):
        """A declared, FREE, unquarantined remote resolves to the same runner object.

        This is the case ``quarantined_remote_runners()`` structurally cannot
        answer — and precisely what a tracker-driven reprobe needs.
        """
        from orchestrator.verify_runner import HostAllocator

        remote_a = _FakeRemoteRunner('remoteA')
        alloc = HostAllocator([remote_a], quarantine=set())
        assert alloc.is_quarantined('remoteA') is False
        assert alloc.remote_runner('remoteA') is remote_a

    async def test_remote_runner_returns_identical_object_when_busy(self):
        """Slot state does not affect resolution — BUSY still resolves."""
        from orchestrator.verify_runner import HostAllocator

        remote_a = _FakeRemoteRunner('remoteA')
        alloc = HostAllocator([remote_a], quarantine=set())
        lease = alloc.acquire_remote()
        assert lease is not None
        assert alloc.remote_runner('remoteA') is remote_a

    async def test_remote_runner_returns_identical_object_when_parked(self):
        """A PARKED (stranded) host still resolves to its runner — the recovery handle."""
        alloc, remote_a = await self._make_parked_allocator()
        assert alloc.is_parked('remoteA') is True
        assert alloc.remote_runner('remoteA') is remote_a

    async def test_remote_runner_returns_identical_object_when_quarantined(self):
        """Quarantine membership does not affect resolution either."""
        from orchestrator.verify_runner import HostAllocator

        remote_a = _FakeRemoteRunner('remoteA')
        alloc = HostAllocator([remote_a], quarantine={'remoteA'})
        assert alloc.is_quarantined('remoteA') is True
        assert alloc.remote_runner('remoteA') is remote_a

    async def test_remote_runner_returns_none_for_local(self):
        """Local is not a remote — _remote_runners holds only remotes."""
        alloc = self._make_allocator()
        assert alloc.remote_runner('local') is None

    async def test_remote_runner_returns_none_for_unknown_name(self):
        """An unmanaged host name returns None rather than raising."""
        alloc = self._make_allocator()
        assert alloc.remote_runner('no-such-host') is None

    # -- local_name (task 3043 amend) -----------------------------------------

    async def test_local_name_defaults_to_local(self):
        """The O(1) read that replaces scanning host_states() for is_local."""
        alloc = self._make_allocator()
        assert alloc.local_name == 'local'

    async def test_local_name_honours_a_custom_local_name(self):
        """Not hard-coded: it reports whatever the allocator was built with."""
        from orchestrator.verify_runner import HostAllocator

        alloc = HostAllocator(
            [_FakeRemoteRunner('remoteA')], quarantine=set(), local_name='anchor-01',
        )
        assert alloc.local_name == 'anchor-01'

    async def test_local_name_agrees_with_host_states_is_local(self):
        """local_name can never drift from host_states()'s is_local flag.

        The two readers of ``_local_name`` are pinned to agree, over a custom
        name so a hard-coded ``'local'`` on either side would be caught.
        """
        from orchestrator.verify_runner import HostAllocator

        alloc = HostAllocator(
            [_FakeRemoteRunner('remoteA')], quarantine=set(), local_name='anchor-01',
        )
        flagged = [h['name'] for h in alloc.host_states() if h['is_local']]
        assert flagged == [alloc.local_name]

    async def test_local_name_is_read_only(self):
        """A property, not a settable attribute — callers cannot retarget the anchor."""
        alloc = self._make_allocator()
        with pytest.raises(AttributeError):
            alloc.local_name = 'somewhere-else'  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3043/step-3 RED: readmit() — the full re-engagement primitive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHostAllocatorReadmit:
    """readmit(name): clear quarantine AND un-PARK, so a recovered host is usable.

    Task 3043.  ``clear_quarantine`` only does ``_quarantine.discard(name)``, but
    ``acquire_remote`` additionally requires ``_SLOT_FREE`` — so the auto-reprobe
    recovery path can resolve the L1, pop the tracker and clear the quarantine
    while the slot stays PARKED, leaving a host that is unquarantined, untracked
    AND unusable: invisible to every recovery mechanism until a restart.

    RED until 3043/step-4 adds the method (AttributeError before that).
    """

    def _local_factory(self):
        return _FakeLocalRunner()

    def _by_name(self, alloc) -> dict:
        return {h['name']: h for h in alloc.host_states()}

    async def _make_parked_allocator(self, *, also_quarantine=False, extra_remote=False):
        """Return (alloc, quarantine_set) with remoteA's slot left PARKED.

        Same construction as ``TestHostAllocatorCancelFail`` /
        ``test_cancel_fail_bounded_max_attempts_stays_parked``: cancel rc != 0
        and probe_clean() never clean with bounded max_attempts.

        ``also_quarantine`` adds remoteA to the shared set AFTER parking — the
        host must be acquirable to reach the cancel path in the first place, so
        seeding the quarantine up front would make acquire_remote() refuse it.
        """
        from orchestrator.verify_runner import HostAllocator

        remote_a = _FakeRemoteRunnerCancellable(
            'remoteA', cancel_rc=1, probe_sequence=[False] * 20,
        )
        remotes = [remote_a] + ([_FakeRemoteRunner('remoteB')] if extra_remote else [])
        q: set[str] = set()
        alloc = HostAllocator(remotes, quarantine=q)

        lease = alloc.acquire_remote()
        assert lease is not None and lease.name == 'remoteA'

        async def noop_sleep(_: float) -> None:
            pass

        assert await alloc.cancel_and_release(lease, sleep=noop_sleep, max_attempts=2) is False
        assert alloc.is_parked('remoteA') is True
        if also_quarantine:
            q.add('remoteA')
        return alloc, q

    # -- (a) discards from the shared set --------------------------------------

    async def test_readmit_removes_name_from_shared_set(self):
        """readmit discards from the set passed BY REFERENCE at construction.

        Same assertion shape as test_clear_quarantine_removes_name_from_shared_set:
        the caller's own set must see the removal, since that is the worker's
        ``_runner_quarantine`` and is what makes re-engagement restart-free.
        """
        from orchestrator.verify_runner import HostAllocator

        shared_q: set[str] = {'remoteA'}
        alloc = HostAllocator([_FakeRemoteRunner('remoteA')], quarantine=shared_q)

        alloc.readmit('remoteA')

        assert 'remoteA' not in shared_q
        assert alloc.is_quarantined('remoteA') is False

    # -- (b) un-PARKs so the host is acquirable again ---------------------------

    async def test_readmit_unparks_slot_and_host_becomes_acquirable(self):
        """A PARKED slot is reset to FREE, so acquire_remote() hands the host out."""
        alloc, _q = await self._make_parked_allocator()

        alloc.readmit('remoteA')

        assert alloc.is_parked('remoteA') is False
        assert self._by_name(alloc)['remoteA']['slot_state'] == 'free'
        lease = alloc.acquire_remote()
        assert lease is not None and lease.name == 'remoteA'

    async def test_readmit_clears_quarantine_and_unparks_together(self):
        """Both halves in one call — the state a recovered host actually needs."""
        alloc, shared_q = await self._make_parked_allocator(also_quarantine=True)
        assert alloc.is_parked('remoteA') is True
        assert alloc.is_quarantined('remoteA') is True

        alloc.readmit('remoteA')

        assert 'remoteA' not in shared_q
        assert alloc.is_parked('remoteA') is False
        lease = alloc.acquire_remote()
        assert lease is not None and lease.name == 'remoteA'

    # -- (c) never steals a live verify ----------------------------------------

    async def test_readmit_leaves_busy_slot_busy(self):
        """A BUSY slot is never stolen — only the quarantine is cleared."""
        from orchestrator.verify_runner import HostAllocator

        shared_q: set[str] = set()
        alloc = HostAllocator([_FakeRemoteRunner('remoteA')], quarantine=shared_q)
        lease = alloc.acquire_remote()          # remoteA BUSY (verify in flight)
        assert lease is not None and lease.name == 'remoteA'
        # A drift-detector-style writer adds to the shared set directly while
        # the verify is still running — BUSY + quarantined is a reachable state.
        shared_q.add('remoteA')

        alloc.readmit('remoteA')

        assert 'remoteA' not in shared_q
        assert self._by_name(alloc)['remoteA']['slot_state'] == 'busy'
        assert alloc.acquire_remote() is None

    # -- (d) idempotent and safe ------------------------------------------------

    async def test_readmit_is_idempotent_on_already_free_host(self):
        """A second readmit on an already-FREE, unquarantined host is a no-op."""
        alloc, _q = await self._make_parked_allocator()
        alloc.readmit('remoteA')
        alloc.readmit('remoteA')

        assert alloc.is_parked('remoteA') is False
        assert alloc.is_quarantined('remoteA') is False
        lease = alloc.acquire_remote()
        assert lease is not None and lease.name == 'remoteA'

    async def test_readmit_local_host_is_safe_noop(self):
        """readmit('local') does not raise and does not disturb the local slot."""
        from orchestrator.verify_runner import HostAllocator

        alloc = HostAllocator([_FakeRemoteRunner('remoteA')], quarantine=set())
        local_lease = alloc.acquire_local(self._local_factory)
        assert local_lease is not None

        alloc.readmit('local')

        # BUSY local is not stolen; a second acquire_local still refuses.
        assert self._by_name(alloc)['local']['slot_state'] == 'busy'
        assert alloc.acquire_local(self._local_factory) is None

    async def test_readmit_unknown_name_does_not_create_a_slot(self):
        """An unmanaged name is a no-op — no raise, and no slot is fabricated."""
        from orchestrator.verify_runner import HostAllocator

        alloc = HostAllocator([_FakeRemoteRunner('remoteA')], quarantine=set())
        before = [h['name'] for h in alloc.host_states()]

        alloc.readmit('no-such-host')

        assert [h['name'] for h in alloc.host_states()] == before
        assert 'no-such-host' not in {h['name'] for h in alloc.host_states()}

    # -- (e) REGRESSION PIN: why clear_quarantine alone is not enough -----------

    async def test_clear_quarantine_alone_leaves_parked_host_unusable(self):
        """The strand this task fixes, stated executably.

        After the identical PARKED setup, clear_quarantine(name) ONLY leaves the
        host non-acquirable: unquarantined, (once the tracker entry is popped)
        untracked, AND unusable — invisible to every recovery mechanism.  This
        is precisely why the reprobe recovery path must call readmit instead.
        """
        alloc, shared_q = await self._make_parked_allocator(
            also_quarantine=True, extra_remote=True,
        )

        alloc.clear_quarantine('remoteA')

        assert 'remoteA' not in shared_q
        assert alloc.is_parked('remoteA') is True
        entry = self._by_name(alloc)['remoteA']
        assert entry['slot_state'] == 'parked'
        assert entry['quarantined'] is False
        # Still non-acquirable: acquire_remote skips it and falls through to remoteB.
        lease = alloc.acquire_remote()
        assert lease is not None and lease.name == 'remoteB'

        # readmit, by contrast, actually re-engages the host.
        alloc.readmit('remoteA')
        recovered = alloc.acquire_remote()
        assert recovered is not None and recovered.name == 'remoteA'
