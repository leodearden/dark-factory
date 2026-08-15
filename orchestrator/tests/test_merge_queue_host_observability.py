"""Per-host observability in SpeculativeMergeWorker.snapshot() (task 3275).

Three related surfaces, all additive and backward-compatible:

  step-3/4  snapshot()['hosts']       — per-host slot state + quarantine class,
                                        so `verifying 1/2 hosts` is no longer
                                        four-way ambiguous.
  step-5/6  occupancy.inflight_by_host / inflight_total
                                      — lossless per-host occupant list; the
                                        historical by_host {host: task_id} map
                                        silently drops co-located occupants.
  step-7/8  heartbeat DEGRADED segment — a quarantined host is named inline in
                                        the log line and in the structured
                                        merge_heartbeat event.

Fixture block (_setup_repo, git_repo/git_config/git_ops/config, _make_req,
_make_item, _make_entry, _make_mock_allocator) is ported from
test_merge_queue_finalize_head_visibility.py:44-137 so these snapshot tests
build InflightEntry state exactly the way the existing ones do.

Steps covered:
  3275/step-3 RED — snapshot()['hosts'] block
  3275/step-5 RED — occupancy must not silently drop a co-located occupant
  3275/step-7 RED — heartbeat names a quarantined host inline
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import (
    DecidedItem,
    InflightEntry,
    ItemLifecycleState,
    MergeOutcome,
    MergeRequest,
    SpeculativeMergeWorker,
    _HostUnavailability,
)
from orchestrator.merge_types import QueuedBranch
from orchestrator.verify_runner import HostAllocator, HostLease

# ── fixtures (ported verbatim from test_merge_queue_finalize_head_visibility) ──


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


# ── shared helpers ────────────────────────────────────────────────────────────


def _make_req(
    task_id: str,
    branch: str,
    config: OrchestratorConfig,
    git_repo: Path,
) -> MergeRequest:
    """Build a minimal MergeRequest with a fresh event-loop future."""
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        worktree=git_repo,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
    )


def _make_item(req: MergeRequest) -> DecidedItem:
    return DecidedItem(
        request=req,
        immediate_outcome=MergeOutcome('blocked', reason='test-filler'),
        base_sha='abc123',
        speculative=False,
    )


def _make_entry(
    req: MergeRequest,
    lease: HostLease | None,
    started_at: float | None = None,
) -> InflightEntry:
    return InflightEntry(
        item=_make_item(req),
        lease=lease,
        verify_task=None,
        merge_wt=None,
        was_speculative=False,
        started_at=started_at,
    )


def _make_mock_allocator(host_names: list[str]) -> MagicMock:
    """The MagicMock allocator double used at 79 sites across 15 test files.

    Stubs only host_names/release/cancel_and_release.  Because a MagicMock
    answers EVERY attribute, duck-typing gives snapshot() no protection —
    hence the isinstance gate this file pins.
    """
    alloc = MagicMock()
    alloc.host_names = host_names
    alloc.release = AsyncMock()
    alloc.cancel_and_release = AsyncMock()
    return alloc


# ── host-observability helpers ────────────────────────────────────────────────


_HOST_KEYS = {
    'name', 'is_local', 'slot_state', 'quarantined',
    'quarantine_class', 'unavailable_since', 'streak', 'reason',
}


class _FakeRemoteRunner:
    """Minimal RemoteRunner-like for a REAL HostAllocator (no I/O)."""

    is_local: bool = False

    def __init__(self, name: str, *, healthy: bool = True) -> None:
        self.name = name
        self.health = AsyncMock(return_value=healthy)


class _FakeEscalationQueue:
    """Minimal escalation queue — no open L1, so recovery stays quiet."""

    def __init__(self) -> None:
        self.submitted: list = []
        self._seq = 0

    def has_open_l1(self, task_id: str) -> bool:  # noqa: ARG002
        return False

    def get_by_task(self, task_id: str, status: str | None = None, level: int | None = None) -> list:  # noqa: ARG002
        return []

    def make_id(self, task_id: str) -> str:  # noqa: ARG002
        self._seq += 1
        return f'esc-{self._seq}'

    def submit(self, esc) -> None:
        self.submitted.append(esc)


def _real_worker(git_ops: GitOps, *, remotes: list[str] | None = None):
    """Bare worker + a REAL HostAllocator sharing worker._runner_quarantine.

    Deliberately NOT a MagicMock: slot state and quarantine membership must be
    genuine for these tests to mean anything.
    """
    q: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker = SpeculativeMergeWorker(git_ops, q)
    runners = [_FakeRemoteRunner(n) for n in (remotes if remotes is not None else ['laptop'])]
    alloc = HostAllocator(runners, quarantine=worker._runner_quarantine)
    worker._host_allocator = alloc
    return worker, alloc, {r.name: r for r in runners}


def _seed_ru(
    worker,
    host: str,
    *,
    streak: int = 4,
    first_unavailable_at: float = 1000.0,
    reason: str = 'ssh: connect timed out',
) -> None:
    """Seed the RunnerUnavailable streak tracker directly."""
    worker._runner_unavailable[host] = _HostUnavailability(
        streak=streak,
        first_unavailable_at=first_unavailable_at,
        reason=reason,
    )


def _by_name(snap: dict) -> dict:
    return {h['name']: h for h in snap['hosts']}


# ── 3275/step-3 RED: snapshot()['hosts'] ──────────────────────────────────────


@pytest.mark.asyncio
class TestSnapshotHostsBlock:
    """snapshot()['hosts']: per-host slot state + quarantine class.

    Resolves the four-way ambiguity behind a `verifying 1/2 hosts` line:
      (a) RU-quarantined            → quarantine_class == 'ru'
      (b) divergence-quarantined    → quarantine_class == 'divergence'
      (c) leaked slot               → slot_state busy/parked, no occupant
      (d) free and never asked for  → slot_state == 'free', quarantined False

    RED until 3275/step-4 adds the key (KeyError before that).
    """

    async def test_hosts_shape_order_and_uniform_schema(self, git_ops: GitOps) -> None:
        """One entry per allocator host, local first, every key always present."""
        worker, _alloc, _ = _real_worker(git_ops)

        snap = worker.snapshot()
        hosts = snap['hosts']

        assert isinstance(hosts, list), f'hosts must be a list; got {type(hosts)}'
        assert [h['name'] for h in hosts] == ['local', 'laptop'], (
            f'hosts must mirror allocator order (local first); got {[h["name"] for h in hosts]}'
        )
        for entry in hosts:
            assert set(entry) == _HOST_KEYS, (
                f'Non-uniform schema for {entry.get("name")!r}: {sorted(entry)}'
            )
        assert _by_name(snap)['local']['is_local'] is True
        assert _by_name(snap)['laptop']['is_local'] is False

    async def test_case_d_free_and_never_asked_for(self, git_ops: GitOps) -> None:
        """(d) No quarantine, no RU tracking → free, and every N/A field is None."""
        worker, _alloc, _ = _real_worker(git_ops)

        laptop = _by_name(worker.snapshot())['laptop']

        assert laptop['slot_state'] == 'free'
        assert laptop['quarantined'] is False
        assert laptop['quarantine_class'] is None
        assert laptop['unavailable_since'] is None
        assert laptop['streak'] is None
        assert laptop['reason'] is None

    async def test_case_a_ru_quarantined(self, git_ops: GitOps) -> None:
        """(a) Quarantined AND RU-tracked → class 'ru', RU fields populated."""
        worker, _alloc, _ = _real_worker(git_ops)
        worker._runner_quarantine.add('laptop')
        _seed_ru(worker, 'laptop', streak=4, first_unavailable_at=1000.0,
                 reason='ssh: connect timed out')

        laptop = _by_name(worker.snapshot())['laptop']

        assert laptop['quarantined'] is True
        assert laptop['quarantine_class'] == 'ru'
        assert laptop['unavailable_since'] == 1000.0
        assert laptop['streak'] == 4
        assert laptop['reason'] == 'ssh: connect timed out'

    async def test_case_b_divergence_quarantined(self, git_ops: GitOps) -> None:
        """(b) Quarantined but NOT RU-tracked → class 'divergence', RU fields None.

        This is the exact discrimination _reprobe_quarantined_hosts makes
        (`entry = self._runner_unavailable.get(name); if entry is None: continue`
        — "Skip divergence-quarantined hosts, not tracked as RunnerUnavailable"),
        so the snapshot can never disagree with the reprobe path.
        """
        worker, _alloc, _ = _real_worker(git_ops)
        worker._runner_quarantine.add('laptop')
        assert 'laptop' not in worker._runner_unavailable

        laptop = _by_name(worker.snapshot())['laptop']

        assert laptop['quarantined'] is True
        assert laptop['quarantine_class'] == 'divergence'
        assert laptop['unavailable_since'] is None
        assert laptop['streak'] is None
        assert laptop['reason'] is None

    async def test_ru_tracked_but_not_quarantined_stays_visible(self, git_ops: GitOps) -> None:
        """Sub-threshold failures are visible: RU fields set, quarantine fields not."""
        worker, _alloc, _ = _real_worker(git_ops)
        _seed_ru(worker, 'laptop', streak=2, first_unavailable_at=555.0,
                 reason='ssh: connection reset')
        assert 'laptop' not in worker._runner_quarantine

        laptop = _by_name(worker.snapshot())['laptop']

        assert laptop['quarantined'] is False
        assert laptop['quarantine_class'] is None
        # RU enrichment is independent of quarantine.
        assert laptop['streak'] == 2
        assert laptop['unavailable_since'] == 555.0
        assert laptop['reason'] == 'ssh: connection reset'

    async def test_case_c_leaked_slot_vs_case_d(self, git_ops: GitOps) -> None:
        """(c) Busy slot with NO in-flight occupant — distinguishable from (d).

        A leaked slot and a never-asked-for host both leave `verifying N/M`
        under-full; only slot_state tells them apart.
        """
        worker, alloc, _ = _real_worker(git_ops)
        lease = alloc.acquire_remote()
        assert lease is not None and lease.name == 'laptop'
        assert not worker._inflight, 'the leak is: slot held, nothing in flight'

        snap = worker.snapshot()
        laptop = _by_name(snap)['laptop']
        occ = snap['occupancy']

        # Half 1: the allocator says the slot is held...
        assert laptop['slot_state'] == 'busy'
        assert laptop['quarantined'] is False
        assert laptop['quarantine_class'] is None
        # Half 2: ...while occupancy has no occupant for it. Asserted against the
        # lossless map once step-6 lands, and against the historical map always;
        # both must agree that nothing is in flight on 'laptop'.
        assert 'laptop' not in occ.get('inflight_by_host', {}), (
            f"leaked slot must have no in-flight occupant; got {occ.get('inflight_by_host')}"
        )
        assert 'laptop' not in occ['by_host'], (
            f"leaked slot must have no in-flight occupant; got {occ['by_host']}"
        )
        # And 'local' is the (d) contrast: free, unasked-for.
        assert _by_name(snap)['local']['slot_state'] == 'free'

    async def test_allocator_not_built_reports_empty_list(self, git_ops: GitOps) -> None:
        """No allocator yet (no verify dispatched) → [] — never a fabricated local."""
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        assert worker._host_allocator is None

        snap = worker.snapshot()

        assert snap['hosts'] == []
        # The pre-existing hosts_total fallback is untouched.
        assert snap['occupancy']['hosts_total'] == 1

    async def test_magicmock_allocator_double_reports_empty_list(self, git_ops: GitOps) -> None:
        """A MagicMock allocator yields [] rather than MagicMock garbage or a crash.

        Pins the documented isinstance gate: a MagicMock answers every
        attribute, so hasattr/duck-typing would silently inject junk into this
        read-only path.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)
        worker._host_allocator = _make_mock_allocator(['local', 'laptop'])

        snap = worker.snapshot()

        assert snap['hosts'] == [], f'MagicMock allocator must yield []; got {snap["hosts"]!r}'

    async def test_quarantine_to_recovery_round_trip(self, git_ops: GitOps) -> None:
        """End-to-end: quarantine → snapshot says 'ru' → reprobe → snapshot says free.

        The literal acceptance clause: `1/2 hosts` is never again ambiguous
        between "quarantined" and "not asked for".
        """
        worker, alloc, runners = _real_worker(git_ops)
        worker._escalation_queue = _FakeEscalationQueue()
        worker._unreachable_escalate_after_secs = 5.0
        worker._unreachable_escalate_after_n = 3

        now = 2000.0
        lease = alloc.acquire_remote()
        assert lease is not None and lease.name == 'laptop'
        await alloc.quarantine_and_release(lease)
        _seed_ru(worker, 'laptop', streak=5, first_unavailable_at=now - 60.0,
                 reason='ssh: connect timed out')

        before = _by_name(worker.snapshot())['laptop']
        assert before['quarantined'] is True
        assert before['quarantine_class'] == 'ru'
        assert before['streak'] == 5
        assert before['unavailable_since'] == now - 60.0

        # Host comes back: health() → True.
        runners['laptop'].health = AsyncMock(return_value=True)
        await worker._reprobe_quarantined_hosts(now)

        after = _by_name(worker.snapshot())['laptop']
        assert after['quarantined'] is False
        assert after['quarantine_class'] is None
        assert after['unavailable_since'] is None
        assert after['streak'] is None
        assert after['reason'] is None
        assert after['slot_state'] == 'free'


# ── 3275/step-5 RED: occupancy must not silently drop a co-located occupant ────


@pytest.mark.asyncio
class TestSnapshotOccupancyLossless:
    """occupancy.inflight_by_host / inflight_total: lossless per-host occupants.

    The historical by_host is a {host: task_id} dict built by comprehension, so
    two entries leased to ONE host collapse last-writer-wins, and the
    finalize-head prepend splat drops the head's task_id when it shares a host
    with an in-flight entry.  Both losses are silent.

    RED until 3275/step-6 adds the keys (KeyError before that).
    """

    def _leases(self):
        return (
            HostLease(name='local', runner=MagicMock(), is_local=True),
            HostLease(name='laptop', runner=MagicMock(), is_local=False),
        )

    def _assert_derivation_invariant(self, occ: dict) -> None:
        """hosts_busy is re-sourced from the lossless map, same value as before.

        len(inflight_by_host) == len(by_host) holds by construction (identical
        key sets), so re-sourcing is a no-op value change.
        """
        assert occ['hosts_busy'] == len(occ['inflight_by_host']) == len(occ['by_host']), (
            f'hosts_busy={occ["hosts_busy"]}, '
            f'len(inflight_by_host)={len(occ["inflight_by_host"])}, '
            f'len(by_host)={len(occ["by_host"])}'
        )

    async def test_two_inflight_on_one_host_keeps_both(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """The headline loss: two in-flight entries on ONE host — neither dropped."""
        worker, _alloc, _ = _real_worker(git_ops)
        _local, laptop_lease = self._leases()

        for tid in ('occ-a', 'occ-b'):
            req = _make_req(tid, f'task/{tid}', config, git_repo)
            worker._inflight.append(_make_entry(req, laptop_lease))

        occ = worker.snapshot()['occupancy']

        assert occ['inflight_by_host'] == {'laptop': ['occ-a', 'occ-b']}, (
            f'both occupants must survive in _inflight order; got '
            f'{occ["inflight_by_host"]}'
        )
        assert occ['inflight_total'] == 2
        # hosts_busy counts distinct busy HOSTS, not verifies in flight.
        assert occ['hosts_busy'] == 1
        # Backward compat: the lossy historical map is unchanged (task 3044 owns it).
        assert occ['by_host'] == {'laptop': 'occ-b'}
        self._assert_derivation_invariant(occ)

    async def test_finalize_head_colocated_with_inflight_keeps_both(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """The second loss path: the finalize-head prepend splat drops the head.

        `_by_host = {_fh_name: _fh_tid, **_by_host}` — when the head shares a
        host with an in-flight entry, the splat overwrites the head's task_id.
        """
        worker, _alloc, _ = _real_worker(git_ops)
        _local, laptop_lease = self._leases()

        req_tail = _make_req('occ-tail', 'task/occ-tail', config, git_repo)
        worker._inflight.append(_make_entry(req_tail, laptop_lease))

        req_head = _make_req('occ-head', 'task/occ-head', config, git_repo)
        head = _make_entry(req_head, laptop_lease)
        worker._register_item(head, initial=ItemLifecycleState.VERIFYING)

        occ = worker.snapshot()['occupancy']

        assert occ['inflight_by_host'] == {'laptop': ['occ-head', 'occ-tail']}, (
            f'finalize head must lead, and must not be overwritten; got '
            f'{occ["inflight_by_host"]}'
        )
        assert occ['inflight_total'] == 2
        assert occ['hosts_busy'] == 1
        self._assert_derivation_invariant(occ)

    async def test_distinct_hosts_unchanged(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """Multi-host, one occupant each — the non-colliding case is unaffected."""
        worker, _alloc, _ = _real_worker(git_ops)
        local_lease, laptop_lease = self._leases()

        req_local = _make_req('occ-local', 'task/occ-local', config, git_repo)
        req_laptop = _make_req('occ-laptop', 'task/occ-laptop', config, git_repo)
        worker._inflight.append(_make_entry(req_local, local_lease))
        worker._inflight.append(_make_entry(req_laptop, laptop_lease))

        occ = worker.snapshot()['occupancy']

        assert occ['inflight_by_host'] == {
            'local': ['occ-local'], 'laptop': ['occ-laptop'],
        }
        assert occ['inflight_total'] == 2
        assert occ['hosts_busy'] == 2
        self._assert_derivation_invariant(occ)

    async def test_empty_and_leaseless_are_excluded(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """Nothing in flight → empty; a lease=None entry is excluded from both."""
        worker, _alloc, _ = _real_worker(git_ops)

        occ = worker.snapshot()['occupancy']
        assert occ['inflight_by_host'] == {}
        assert occ['inflight_total'] == 0
        assert occ['hosts_busy'] == 0
        self._assert_derivation_invariant(occ)

        # A leaseless in-flight entry (awaiting host acquisition) is not an occupant.
        req = _make_req('occ-leaseless', 'task/occ-leaseless', config, git_repo)
        worker._inflight.append(_make_entry(req, None))

        occ = worker.snapshot()['occupancy']
        assert occ['inflight_by_host'] == {}
        assert occ['inflight_total'] == 0
        assert occ['hosts_busy'] == 0
        self._assert_derivation_invariant(occ)
