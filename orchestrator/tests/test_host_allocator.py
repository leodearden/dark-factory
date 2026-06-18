"""Tests for HostAllocator + HostLease (β: per-host slots, prefer-local-when-free, cancel-aware release).

Steps covered in this file:
  step-1  RED  — construction + introspection
  step-3  RED  — prefer-local, remote overflow, ≤1/host, release
  step-5  RED  — quarantine_and_release
  step-7  RED  — cancel_and_release happy path
  step-9  RED  — cancel_and_release cancel-FAIL + PARK + probe
  1795/step-1  RED  — clear_quarantine + quarantined_remote_runners
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
