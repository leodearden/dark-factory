"""Tests for scripts/cgroup_weight_share.py: the weight-fair share a cgroup is entitled to."""
from __future__ import annotations

import dataclasses
import itertools
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path

import cgroup_weight_share
import pytest
from cgroup_weight_share import (
    CgroupWindow,
    Claim,
    Snapshot,
    fair_shares,
    host_share,
    main,
    measure,
    read_tree,
    share_chain,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


def assert_read_only(instance: object, field: str) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, getattr(instance, field))


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
    assert_read_only(Claim(100, 1.0), 'weight')


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
    """One counter state: cumulative CPU per cgroup and wait per thread, in core-windows.

    The service has no cpu.weight, as when its parent does not delegate the cpu controller.
    """
    def usage(rel: str) -> int:
        return round(core_windows.get(rel, 0.0) * USEC_PER_CORE_WINDOW)

    host.cgroup('.', usage_usec=usage('.'))
    host.cgroup(SERVICE, usage_usec=usage(SERVICE))
    host.cgroup(DF_SLICE, usage_usec=usage(DF_SLICE), weight=100)
    host.cgroup(REIFY_SLICE, usage_usec=usage(REIFY_SLICE), weight=100)
    host.cgroup(MERGE, usage_usec=usage(MERGE), weight=100, tids=(11, 12, 13))
    host.cgroup(TASK, usage_usec=usage(TASK), weight=33, tids=(21,))
    host.cgroup(APP_SLICE, usage_usec=usage(APP_SLICE), weight=100)
    host.cgroup(DAEMON, usage_usec=usage(DAEMON), weight=100, tids=(31,))
    for tid, waited in wait_windows.items():
        host.thread(tid, wait_ns=round(waited * NS_PER_CORE_WINDOW))


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


@pytest.mark.skipif(os.geteuid() == 0, reason='root reads a mode-000 file')
def test_unreadable_cgroup_is_an_error_not_a_removed_cgroup(host):
    write_reify_host(host, core_windows={}, wait_windows={})
    (host.cgroup_root / TASK / 'cpu.stat').chmod(0)
    with pytest.raises(PermissionError):
        host.snapshot()


def test_readings_and_windows_are_frozen(host, windows):
    snapshot = host.snapshot()
    assert_read_only(snapshot, 'taken_at')
    assert_read_only(snapshot.readings[MERGE], 'usage_usec')
    assert_read_only(windows[MERGE], 'used_cores')


def measure_reify_host(
    host: FakeHost,
    *,
    core_windows: Mapping[str, float],
    wait_windows: Mapping[int, float],
    idle_scopes: tuple[str, ...] = (),
) -> dict[str, CgroupWindow]:
    write_reify_host(host, core_windows={}, wait_windows={tid: 0.0 for tid in wait_windows})
    for rel in idle_scopes:
        host.cgroup(rel, usage_usec=0, weight=100)
    before = host.snapshot()
    write_reify_host(host, core_windows=core_windows, wait_windows=wait_windows)
    return measure(before, host.snapshot(), WINDOW_S)


CONTENDED_CORES = {
    '.': 26.6, SERVICE: 26.6, DF_SLICE: 13.3, REIFY_SLICE: 13.3,
    MERGE: 8.0, TASK: 5.3, APP_SLICE: 13.3, DAEMON: 13.3,
}
CONTENDED_WAITS = {11: 6.0, 12: 6.0, 21: 14.7, 31: 10.0}


@pytest.fixture
def contended(host: FakeHost) -> list:
    """Merge wants 8+12=20 and task wants 5.3+14.7=20 of a slice that got 13.3 cores."""
    windows = measure_reify_host(host, core_windows=CONTENDED_CORES, wait_windows=CONTENDED_WAITS)
    return share_chain(windows, MERGE)


def test_chain_walks_from_the_leaf_to_the_root_child(contended):
    assert [level.path for level in contended] == [MERGE, REIFY_SLICE, DF_SLICE, SERVICE]


def test_merge_scope_fair_share_is_100_to_33_against_its_task_sibling(contended):
    merge = contended[0]
    assert merge.used_cores == pytest.approx(8.0)
    assert merge.parent_used_cores == pytest.approx(13.3)
    assert merge.share_of_parent == pytest.approx(8.0 / 13.3)
    assert merge.fair_cores == pytest.approx(10.0)
    assert merge.used_vs_fair == pytest.approx(0.8)


def test_slice_level_splits_what_the_service_got_by_weight(contended):
    df_slice = contended[2]
    assert df_slice.share_of_parent == pytest.approx(0.5)
    assert df_slice.fair_cores == pytest.approx(13.3)
    assert df_slice.used_vs_fair == pytest.approx(1.0)


def test_idle_sibling_does_not_dilute_the_merge_scope(host):
    windows = measure_reify_host(
        host,
        core_windows=CONTENDED_CORES,
        wait_windows=CONTENDED_WAITS,
        idle_scopes=(f'{REIFY_SLICE}/idle.scope',),
    )
    assert share_chain(windows, MERGE)[0].fair_cores == pytest.approx(10.0)


def test_cpu_of_a_sibling_gone_by_the_window_end_is_unattributed_not_split(host):
    gone = f'{REIFY_SLICE}/gone.scope'
    write_reify_host(host, core_windows={}, wait_windows={tid: 0.0 for tid in CONTENDED_WAITS})
    host.cgroup(gone, usage_usec=0, weight=100)
    before = host.snapshot()
    shutil.rmtree(host.cgroup_root / gone)
    with_gone_cpu = {path: CONTENDED_CORES[path] + 4.0 for path in ('.', SERVICE, DF_SLICE, REIFY_SLICE)}
    write_reify_host(host, core_windows={**CONTENDED_CORES, **with_gone_cpu}, wait_windows=CONTENDED_WAITS)
    merge, reify_slice, *_ = share_chain(measure(before, host.snapshot(), WINDOW_S), MERGE)
    assert merge.unattributed_cores == pytest.approx(4.0)
    assert merge.fair_cores == pytest.approx(10.0)
    assert merge.used_vs_fair == pytest.approx(0.8)
    assert reify_slice.unattributed_cores == pytest.approx(0.0)


def test_siblings_read_ahead_of_their_parent_split_only_what_the_parent_used(host):
    windows = measure_reify_host(
        host, core_windows={**CONTENDED_CORES, REIFY_SLICE: 13.0}, wait_windows=CONTENDED_WAITS,
    )
    merge = share_chain(windows, MERGE)[0]
    assert merge.unattributed_cores == 0.0
    assert merge.fair_cores == pytest.approx(13.0 * 100 / 133)


def test_level_whose_siblings_lack_weight_has_no_entitlement(contended):
    service = contended[-1]
    assert service.fair_cores is None
    assert service.used_vs_fair is None
    assert service.share_of_parent == pytest.approx(1.0)


def test_unknown_leaf_raises_key_error_naming_it(windows):
    with pytest.raises(KeyError, match='nope.scope'):
        share_chain(windows, f'{REIFY_SLICE}/nope.scope')


def test_host_share_is_the_product_of_shares_down_the_chain(contended):
    assert host_share(contended) == pytest.approx((8.0 / 13.3) * 1.0 * 0.5 * 1.0)


def test_chain_level_is_frozen(contended):
    assert_read_only(contended[0], 'fair_cores')


@pytest.mark.parametrize('seconds', [0.0, -1.0])
def test_measure_refuses_a_window_without_positive_length(host, seconds):
    write_reify_host(host, core_windows={}, wait_windows={})
    snapshot = host.snapshot()
    with pytest.raises(ValueError, match=str(seconds)):
        measure(snapshot, snapshot, seconds)


def cli_args(host: FakeHost, *chains: str) -> list[str]:
    return [
        '--cgroup-root', str(host.cgroup_root), '--proc-root', str(host.proc_root), '--window', '0.01',
        *itertools.chain.from_iterable(('--chain', chain) for chain in chains),
    ]


def script_reads(monkeypatch: pytest.MonkeyPatch, *after_each_read: Callable[[], None]) -> None:
    """main's reads land WINDOW_S apart, and after the n-th read the host moves to its n-th next state."""
    real_read_tree = cgroup_weight_share.read_tree
    read_count = itertools.count()
    next_states = iter(after_each_read)

    def read_tree(cgroup_root: Path, proc_root: Path) -> Snapshot:
        snapshot = dataclasses.replace(real_read_tree(cgroup_root, proc_root), taken_at=next(read_count) * WINDOW_S)
        next(next_states, lambda: None)()
        return snapshot

    monkeypatch.setattr(cgroup_weight_share, 'read_tree', read_tree)


def sibling_table(out: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in out.splitlines()[1:]:
        if not line or line.startswith('chain '):
            break
        path, *cells = line.split()
        rows[path] = cells
    return rows


IDLE = f'{REIFY_SLICE}/idle.scope'


def run_contended_cli(host: FakeHost, monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    write_reify_host(host, core_windows={}, wait_windows={tid: 0.0 for tid in CONTENDED_WAITS})
    host.cgroup(IDLE, usage_usec=0, weight=100)
    script_reads(monkeypatch, lambda: write_reify_host(host, core_windows=CONTENDED_CORES, wait_windows=CONTENDED_WAITS))
    return main([*cli_args(host), *args])


def test_cli_sibling_table_puts_each_cgroup_against_its_weight_fair_share(host, monkeypatch, capsys):
    assert run_contended_cli(host, monkeypatch) == 0
    table = sibling_table(capsys.readouterr().out)
    assert table['path'] == [
        'weight', 'used', 'wait', 'wanted', 'unattributed', 'share_of_parent', 'fair', 'used/fair',
    ]
    assert table['.'] == ['-', '26.60', '36.70', '63.30', '-', '-', '-', '-']
    assert table[SERVICE] == ['-', '26.60', '36.70', '63.30', '0.00', '1.00', '-', '-']
    assert table[MERGE] == ['100', '8.00', '12.00', '20.00', '0.00', '0.60', '10.00', '0.80']


@pytest.mark.parametrize(('min_cores', 'hidden'), [
    ([], {IDLE}),
    (['--min-cores', '0'], set()),
    (['--min-cores', '21'], {IDLE, MERGE, TASK}),
])
def test_cli_hides_cgroups_wanting_fewer_than_min_cores(host, monkeypatch, capsys, min_cores, hidden):
    assert run_contended_cli(host, monkeypatch, *min_cores) == 0
    every_cgroup = {'.', SERVICE, DF_SLICE, REIFY_SLICE, MERGE, TASK, IDLE, APP_SLICE, DAEMON}
    assert set(sibling_table(capsys.readouterr().out)) == {'path'} | (every_cgroup - hidden)


def test_cli_prints_a_header_and_one_row_per_chain_level(host, capsys):
    write_reify_host(host, core_windows={}, wait_windows={})
    assert main(cli_args(host, MERGE)) == 0
    lines = capsys.readouterr().out.splitlines()
    assert 'procs_running/nproc ?' in lines[0]
    chain_table = lines[lines.index(f'chain {MERGE}') + 1:]
    assert chain_table[0].split()[:5] == ['path', 'weight', 'used', 'parent_used', 'unattributed']
    assert [row.split()[0] for row in chain_table[1:5]] == [MERGE, REIFY_SLICE, DF_SLICE, SERVICE]
    assert chain_table[5].startswith('host_share')


def test_cli_reports_a_chain_path_absent_from_the_tree_and_still_prints_the_window(host, capsys):
    write_reify_host(host, core_windows={}, wait_windows={})
    missing = f'{REIFY_SLICE}/nope.scope'
    assert main(cli_args(host, missing)) == 2
    captured = capsys.readouterr()
    assert missing in captured.err
    assert 'procs_running/nproc' in captured.out


def test_cli_keeps_window_one_chain_when_the_scope_ends_before_window_two(host, monkeypatch, capsys):
    write_reify_host(host, core_windows={}, wait_windows={})
    script_reads(monkeypatch, lambda: None, lambda: shutil.rmtree(host.cgroup_root / MERGE))
    assert main([*cli_args(host, MERGE), '--repeat', '2']) == 2
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    headers = [index for index, line in enumerate(lines) if line.startswith('# ')]
    chains = [index for index, line in enumerate(lines) if line == f'chain {MERGE}']
    assert len(headers) == 2
    assert len(chains) == 1
    assert headers[0] < chains[0] < headers[1]
    assert MERGE in captured.err


def test_cli_refuses_a_cgroup_root_with_no_readable_cgroup(tmp_path, capsys):
    missing = tmp_path / 'no-cgroup'
    assert main(['--cgroup-root', str(missing), '--proc-root', str(tmp_path), '--window', '0.01']) == 2
    captured = capsys.readouterr()
    assert str(missing) in captured.err
    assert captured.out == ''
