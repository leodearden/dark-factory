"""Tests for scripts/cgroup_weight_share.py: the weight-fair share a cgroup is entitled to."""
from __future__ import annotations

import dataclasses
import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest
from cgroup_weight_share import CgroupWindow, Claim, Snapshot, fair_shares, measure, read_tree


def test_every_claimant_oversubscribed_splits_by_weight():
    claims = {'a': Claim(100, 20.0), 'b': Claim(100, 20.0), 'c': Claim(100, 20.0)}
    assert fair_shares(12.0, claims) == pytest.approx({'a': 4.0, 'b': 4.0, 'c': 4.0})


def test_merge_and_task_scopes_split_100_to_33():
    claims = {'merge': Claim(100, 20.0), 'task': Claim(33, 20.0)}
    assert fair_shares(13.3, claims) == pytest.approx({'merge': 10.0, 'task': 3.3})


def test_satisfied_claimant_gets_its_demand_and_rest_is_resplit_by_weight():
    claims = {'a': Claim(100, 1.0), 'b': Claim(100, 20.0), 'c': Claim(100, 20.0)}
    assert fair_shares(12.0, claims) == pytest.approx({'a': 1.0, 'b': 5.5, 'c': 5.5})


@pytest.mark.parametrize('capacity', [10.0, 32.0])
def test_capacity_covering_total_demand_gives_everyone_their_demand(capacity):
    claims = {'a': Claim(100, 2.0), 'b': Claim(33, 3.0), 'c': Claim(10, 5.0)}
    assert fair_shares(capacity, claims) == pytest.approx({'a': 2.0, 'b': 3.0, 'c': 5.0})


def test_idle_claimant_gets_nothing_and_does_not_dilute_the_rest():
    claims = {'idle': Claim(100, 0.0), 'merge': Claim(100, 20.0), 'task': Claim(33, 20.0)}
    assert fair_shares(13.3, claims) == pytest.approx({'idle': 0.0, 'merge': 10.0, 'task': 3.3})


def test_any_unweighted_claimant_means_no_entitlement_exists():
    claims = {'a': Claim(100, 20.0), 'b': Claim(None, 20.0)}
    assert fair_shares(12.0, claims) is None


def test_no_claimants_yields_empty_shares():
    assert fair_shares(12.0, {}) == {}


def test_claim_is_frozen():
    claim = Claim(100, 1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        claim.weight = 33  # type: ignore[misc]


SERVICE = 'user@1000.service'
DF_SLICE = f'{SERVICE}/df.slice'
REIFY_SLICE = f'{DF_SLICE}/df-reify.slice'
MERGE = f'{REIFY_SLICE}/merge.scope'
TASK = f'{REIFY_SLICE}/task.scope'
APP_SLICE = f'{SERVICE}/app.slice'
DAEMON = f'{APP_SLICE}/orchestrator-reify.service'
WINDOW_S = 2.0
USEC_PER_CORE_WINDOW = int(WINDOW_S * 1e6)
NS_PER_CORE_WINDOW = int(WINDOW_S * 1e9)


@dataclasses.dataclass(frozen=True)
class FakeHost:
    cgroup_root: Path
    proc_root: Path

    def cgroup(self, rel: str, *, usage_usec: int, weight: int | None = None, tids: tuple[int, ...] = ()) -> None:
        directory = self.cgroup_root / rel
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'cpu.stat').write_text(
            f'usage_usec {usage_usec}\nuser_usec 0\nsystem_usec 0\nnr_periods 0\n'
        )
        if weight is not None:
            (directory / 'cpu.weight').write_text(f'{weight}\n')
        (directory / 'cgroup.threads').write_text(''.join(f'{tid}\n' for tid in tids))

    def thread(self, tid: int, *, wait_ns: int, run_ns: int = 0) -> None:
        directory = self.proc_root / str(tid)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'schedstat').write_text(f'{run_ns} {wait_ns} 7\n')

    def snapshot(self) -> Snapshot:
        return read_tree(self.cgroup_root, self.proc_root)


def write_reify_host(host: FakeHost, *, core_windows: Mapping[str, float], wait_windows: Mapping[int, float]) -> None:
    """One counter state: cumulative CPU per cgroup and wait per thread, in core-windows."""
    def usage(rel: str) -> int:
        return int(core_windows.get(rel, 0.0) * USEC_PER_CORE_WINDOW)

    host.cgroup('.', usage_usec=usage('.'))
    host.cgroup(SERVICE, usage_usec=usage(SERVICE), weight=100)
    host.cgroup(DF_SLICE, usage_usec=usage(DF_SLICE), weight=100)
    host.cgroup(REIFY_SLICE, usage_usec=usage(REIFY_SLICE), weight=100)
    host.cgroup(MERGE, usage_usec=usage(MERGE), weight=100, tids=(11, 12, 13))
    host.cgroup(TASK, usage_usec=usage(TASK), weight=33, tids=(21,))
    host.cgroup(APP_SLICE, usage_usec=usage(APP_SLICE), weight=100)
    host.cgroup(DAEMON, usage_usec=usage(DAEMON), weight=100, tids=(31,))
    for tid, waited in wait_windows.items():
        host.thread(tid, wait_ns=int(waited * NS_PER_CORE_WINDOW))


@pytest.fixture
def host(tmp_path: Path) -> FakeHost:
    return FakeHost(tmp_path / 'cgroup', tmp_path / 'proc')


@pytest.fixture
def windows(host: FakeHost) -> dict[str, CgroupWindow]:
    """Merge used 2 cores and waited 2; task used 1 and waited 1; daemon used 1 and waited 1.

    Thread 13 is listed in the merge scope but has no /proc entry (it exited).
    """
    write_reify_host(host, core_windows={}, wait_windows={11: 0.0, 12: 0.0, 21: 0.0, 31: 0.0})
    before = host.snapshot()
    write_reify_host(
        host,
        core_windows={
            '.': 6.0, SERVICE: 5.0, DF_SLICE: 3.0, REIFY_SLICE: 3.0,
            MERGE: 2.0, TASK: 1.0, APP_SLICE: 2.0, DAEMON: 1.0,
        },
        wait_windows={11: 0.5, 12: 1.5, 21: 1.0, 31: 1.0},
    )
    return measure(before, host.snapshot(), WINDOW_S)


def test_window_keys_are_paths_relative_to_the_cgroup_root(windows):
    assert set(windows) == {'.', SERVICE, DF_SLICE, REIFY_SLICE, MERGE, TASK, APP_SLICE, DAEMON}


def test_used_cores_reads_the_hierarchical_counter_without_summing(windows):
    assert windows[MERGE].used_cores == pytest.approx(2.0)
    assert windows[REIFY_SLICE].used_cores == pytest.approx(3.0)
    assert windows['.'].used_cores == pytest.approx(6.0)


def test_wait_cores_sums_own_and_descendant_threads(windows):
    assert windows[MERGE].wait_cores == pytest.approx(2.0)
    assert windows[TASK].wait_cores == pytest.approx(1.0)
    assert windows[REIFY_SLICE].wait_cores == pytest.approx(3.0)
    assert windows[APP_SLICE].wait_cores == pytest.approx(1.0)
    assert windows['.'].wait_cores == pytest.approx(4.0)


def test_wanted_cores_is_used_plus_wait(windows):
    assert windows[MERGE].wanted_cores == pytest.approx(4.0)
    assert windows['.'].wanted_cores == pytest.approx(10.0)


def test_weight_is_read_and_absent_file_means_none(windows):
    assert windows[MERGE].weight == 100
    assert windows[TASK].weight == 33
    assert windows['.'].weight is None


def test_thread_seen_in_only_one_snapshot_contributes_nothing(host):
    write_reify_host(host, core_windows={}, wait_windows={11: 0.0})
    before = host.snapshot()
    write_reify_host(host, core_windows={}, wait_windows={11: 0.5, 12: 9.0})
    assert measure(before, host.snapshot(), WINDOW_S)[MERGE].wait_cores == pytest.approx(0.5)


def test_thread_vanishing_between_listing_and_reading_is_skipped(host):
    write_reify_host(host, core_windows={}, wait_windows={})
    snapshot = host.snapshot()
    assert dict(snapshot.readings[MERGE].thread_wait_ns) == {}


def test_weight_comes_from_the_later_snapshot(host):
    write_reify_host(host, core_windows={}, wait_windows={})
    before = host.snapshot()
    (host.cgroup_root / TASK / 'cpu.weight').write_text('10\n')
    assert measure(before, host.snapshot(), WINDOW_S)[TASK].weight == 10


def test_cgroup_present_in_only_one_snapshot_is_dropped(host):
    write_reify_host(host, core_windows={}, wait_windows={})
    host.cgroup(f'{REIFY_SLICE}/gone.scope', usage_usec=0, weight=100)
    before = host.snapshot()
    shutil.rmtree(host.cgroup_root / REIFY_SLICE / 'gone.scope')
    host.cgroup(f'{REIFY_SLICE}/new.scope', usage_usec=5, weight=33)
    windows = measure(before, host.snapshot(), WINDOW_S)
    assert f'{REIFY_SLICE}/gone.scope' not in windows
    assert f'{REIFY_SLICE}/new.scope' not in windows


def test_readings_and_windows_are_frozen(host, windows):
    snapshot = host.snapshot()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.taken_at = 0.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.readings[MERGE].usage_usec = 0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        windows[MERGE].used_cores = 0.0  # type: ignore[misc]
