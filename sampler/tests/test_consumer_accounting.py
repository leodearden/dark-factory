"""D3 accounting over real fixture trees: the walk, per-cgroup state, increments, residuals.

PRD ``plans/per-consumer-resource-recording-prd.md`` D3 and the §7 boundary
scenarios. Every test reads a real directory tree through ``read_tree`` and
drives ``account`` through two collections, the way the recorder does.
"""

from __future__ import annotations

import pytest
from _cgroup_tree import MEMORY_FILES, CgroupTree

from sampler.consumers import (
    Accounting,
    Figure,
    TreeState,
    account,
    parse_consumer,
    read_tree,
)

_INTERVAL = 30
_T0 = 1_000_000
_S = 1_000_000  # one second of CPU, in µs
_SCOPE_S = 'app.slice/df-verify-proj-1234abcd-0123456789ab.scope'
_SCOPE_T = 'app.slice/df-verify-proj-1234abcd-ba9876543210.scope'


@pytest.fixture
def tree(tmp_path) -> CgroupTree:
    return CgroupTree(tmp_path / 'cgroup')


def _collect(tree: CgroupTree, previous: TreeState | None, now: int) -> Accounting:
    return account(previous, read_tree(tree.root), now=now, nominal_interval=_INTERVAL)


def _first(tree: CgroupTree) -> Accounting:
    return _collect(tree, None, _T0)


def _next(tree: CgroupTree, previous: Accounting, seconds: int = _INTERVAL) -> Accounting:
    return _collect(tree, previous.state, _T0 + seconds)


def _figures(accounting: Accounting, name: str):
    return accounting.per_consumer[parse_consumer(name)]


def _increment(accounting: Accounting, name: str, figure: Figure) -> int | None:
    increments = _figures(accounting, name).increments
    assert increments is not None, name
    return increments.get(figure)


def _cpu_total(accounting: Accounting) -> int:
    return sum(
        (figures.increments or {}).get(Figure.CPU_USEC, 0)
        for figures in accounting.per_consumer.values()
    )


def _no_negative(accounting: Accounting) -> None:
    for consumer, figures in accounting.per_consumer.items():
        values = [*figures.gauges.values(), *(figures.increments or {}).values()]
        assert all(v >= 0 for v in values), consumer


class TestAVerifyScopeExitsBetweenCollections:
    def test_increments_land_on_verify_exited_and_sum_to_the_root(self, tree):
        tree.set('', cpu_usec=100 * _S)
        tree.set('app.slice', cpu_usec=50 * _S)
        tree.set(_SCOPE_S, cpu_usec=10 * _S)
        first = _first(tree)

        tree.remove(_SCOPE_S)
        tree.set(_SCOPE_T, cpu_usec=3 * _S)
        tree.set('app.slice', cpu_usec=54 * _S)
        tree.set('', cpu_usec=104 * _S)
        second = _next(tree, first)

        assert _increment(second, 'verify@proj-1234abcd', Figure.CPU_USEC) == 3 * _S
        assert _increment(second, 'exited@app.slice', Figure.CPU_USEC) == 1 * _S
        assert _increment(second, 'kernel', Figure.CPU_USEC) == 0
        assert _cpu_total(second) == 4 * _S
        _no_negative(second)
        assert all(f.readable for f in second.per_consumer.values())


class TestStallIsNotAResidual:
    def test_each_consumer_stall_is_its_members_own(self, tree):
        tree.set('', memory_some=1000)
        tree.set('app.slice', memory_some=500)
        tree.set('app.slice/sccache.service', memory_some=100)
        tree.set('app.slice/fused-memory.service', memory_some=200)
        first = _first(tree)

        tree.set('', memory_some=1900)
        tree.set('app.slice', memory_some=900)
        tree.set('app.slice/sccache.service', memory_some=130)
        tree.set('app.slice/fused-memory.service', memory_some=250)
        second = _next(tree, first)

        stall = Figure.MEMORY_SOME_STALL_USEC
        assert _increment(second, 'shared@sccache.service', stall) == 30
        assert _increment(second, 'shared@fused-memory.service', stall) == 50
        assert _increment(second, 'exited@app.slice', stall) is None
        assert _increment(second, 'kernel', stall) is None


class TestARestartedService:
    def test_a_new_inode_at_an_old_path_is_a_reset(self, tree):
        tree.set('app.slice', cpu_usec=20 * _S)
        tree.set('app.slice/sccache.service', cpu_usec=10 * _S)
        first = _first(tree)
        before = tree.inode('app.slice/sccache.service')

        tree.recreate('app.slice/sccache.service', cpu_usec=2 * _S)
        tree.set('app.slice', cpu_usec=25 * _S)
        second = _next(tree, first)

        assert tree.inode('app.slice/sccache.service') != before
        assert _increment(second, 'shared@sccache.service', Figure.CPU_USEC) == 2 * _S
        assert _increment(second, 'exited@app.slice', Figure.CPU_USEC) == 3 * _S
        assert _figures(second, 'shared@sccache.service').readable

    def test_a_counter_that_fell_at_the_same_inode_is_a_per_counter_reset(self, tree):
        tree.set('app.slice', cpu_usec=20 * _S, pswpin=500)
        tree.set('app.slice/sccache.service', cpu_usec=10 * _S, pswpin=400)
        first = _first(tree)

        tree.set('app.slice/sccache.service', cpu_usec=12 * _S, pswpin=30)
        tree.set('app.slice', cpu_usec=22 * _S, pswpin=600)
        second = _next(tree, first)

        assert _increment(second, 'shared@sccache.service', Figure.CPU_USEC) == 2 * _S
        assert _increment(second, 'shared@sccache.service', Figure.SWAPIN_PAGES) == 30


class TestListedButUnreadableLastTime:
    def test_not_new_no_increment_use_lands_in_the_parent_both_flagged(self, tree):
        tree.set('app.slice', cpu_usec=20 * _S)
        tree.set('app.slice/sccache.service', cpu_usec=10 * _S, memory=4096)
        tree.set('app.slice/fused-memory.service', cpu_usec=5 * _S)
        tree.make_unreadable('app.slice/sccache.service', 'cpu.stat')
        first = _first(tree)

        tree.make_readable('app.slice/sccache.service', 'cpu.stat')
        tree.set('app.slice/sccache.service', cpu_usec=17 * _S)
        tree.set('app.slice/fused-memory.service', cpu_usec=6 * _S)
        tree.set('app.slice', cpu_usec=30 * _S)
        second = _next(tree, first)

        sccache = _figures(second, 'shared@sccache.service')
        assert not sccache.readable
        assert sccache.members == 1
        assert sccache.increments is not None
        assert Figure.CPU_USEC not in sccache.increments
        assert sccache.gauges[Figure.MEMORY] == 4096
        assert _increment(second, 'exited@app.slice', Figure.CPU_USEC) == 9 * _S
        assert not _figures(second, 'exited@app.slice').readable
        assert _figures(second, 'shared@fused-memory.service').readable

    def test_unreadable_now_flags_and_contributes_nothing(self, tree):
        tree.set('app.slice', cpu_usec=20 * _S)
        tree.set('app.slice/sccache.service', cpu_usec=10 * _S, memory=4096)
        first = _first(tree)

        tree.set('app.slice/sccache.service', cpu_usec=12 * _S)
        tree.set('app.slice', cpu_usec=23 * _S)
        tree.make_unreadable('app.slice/sccache.service', 'memory.current')
        second = _next(tree, first)

        sccache = _figures(second, 'shared@sccache.service')
        assert not sccache.readable
        assert sccache.members == 1
        assert sccache.gauges == {}
        assert sccache.increments == {}
        assert _increment(second, 'exited@app.slice', Figure.CPU_USEC) == 3 * _S


class TestRecorderGap:
    def test_four_intervals_old_writes_gauges_only_and_flags_everyone(self, tree):
        tree.set('app.slice', cpu_usec=20 * _S, memory=100)
        tree.set('app.slice/sccache.service', cpu_usec=10 * _S, memory=60)
        first = _first(tree)

        second = _next(tree, first, seconds=4 * _INTERVAL)

        assert second.interval_seconds == 4 * _INTERVAL
        for consumer, figures in second.per_consumer.items():
            assert figures.increments is None, consumer
            assert not figures.readable, consumer
        assert _figures(second, 'shared@sccache.service').gauges[Figure.MEMORY] == 60

    def test_two_intervals_is_not_a_gap(self, tree):
        tree.set('app.slice', cpu_usec=20 * _S)
        first = _first(tree)

        second = _next(tree, first, seconds=2 * _INTERVAL)

        assert second.interval_seconds == 2 * _INTERVAL
        assert _figures(second, 'other@app.slice').increments is not None

    def test_the_first_run_is_a_gap_with_no_interval(self, tree):
        tree.set('app.slice', cpu_usec=20 * _S, memory=100)

        first = _first(tree)

        assert first.interval_seconds is None
        assert all(f.increments is None for f in first.per_consumer.values())
        assert not any(f.readable for f in first.per_consumer.values())
        assert _figures(first, 'other@app.slice').gauges[Figure.MEMORY] == 100


class TestInnerNodesAreNotMembers:
    def test_members_are_the_leaves_and_inner_nodes_add_residuals(self, tree):
        service = 'app.slice/orchestrator-x.service'
        tree.set('app.slice', cpu_usec=10 * _S)
        tree.set(service, cpu_usec=8 * _S)
        tree.set(f'{service}/daemon', cpu_usec=4 * _S)
        tree.set(f'{service}/agent-1', cpu_usec=2 * _S)
        first = _first(tree)

        tree.set(f'{service}/daemon', cpu_usec=5 * _S)
        tree.set(f'{service}/agent-1', cpu_usec=4 * _S)
        tree.set(service, cpu_usec=12 * _S)
        tree.set('app.slice', cpu_usec=15 * _S)
        second = _next(tree, first)

        assert _figures(second, 'service@orchestrator-x.service').members == 1
        assert _figures(second, 'agents@orchestrator-x.service').members == 1
        assert _figures(second, 'exited@app.slice').members == 0
        assert _increment(second, 'service@orchestrator-x.service', Figure.CPU_USEC) == 1 * _S
        assert _increment(second, 'agents@orchestrator-x.service', Figure.CPU_USEC) == 3 * _S
        assert _increment(second, 'exited@app.slice', Figure.CPU_USEC) == 1 * _S


class TestBothVerifyPlacements:
    def test_both_scopes_are_verify_and_the_project_slice_adds_its_residual(self, tree):
        under_df = 'df.slice/df-x.slice/df-verify-proj-1234abcd-cccccccccccc.scope'
        tree.set('app.slice', cpu_usec=10 * _S)
        tree.set(_SCOPE_S, cpu_usec=1 * _S)
        tree.set('df.slice', cpu_usec=10 * _S)
        tree.set('df.slice/df-x.slice', cpu_usec=10 * _S)
        tree.set(under_df, cpu_usec=1 * _S)
        first = _first(tree)

        tree.set(_SCOPE_S, cpu_usec=3 * _S)
        tree.set('app.slice', cpu_usec=12 * _S)
        tree.set(under_df, cpu_usec=5 * _S)
        tree.set('df.slice/df-x.slice', cpu_usec=15 * _S)
        tree.set('df.slice', cpu_usec=15 * _S)
        second = _next(tree, first)

        verify = _figures(second, 'verify@proj-1234abcd')
        assert verify.members == 2
        assert _increment(second, 'verify@proj-1234abcd', Figure.CPU_USEC) == 6 * _S
        assert _figures(second, 'project@df-x.slice').members == 0
        assert _increment(second, 'project@df-x.slice', Figure.CPU_USEC) == 1 * _S


class TestAnUnreadableInnerCgroup:
    def test_children_count_once_and_the_grandparent_absorbs_its_own_residual(self, tree):
        service = 'app.slice/orchestrator-x.service'
        tree.set('app.slice', cpu_usec=20 * _S)
        tree.set(service, cpu_usec=10 * _S)
        tree.set(f'{service}/daemon', cpu_usec=4 * _S)
        tree.set(f'{service}/agent-1', cpu_usec=4 * _S)
        first = _first(tree)

        tree.set(f'{service}/daemon', cpu_usec=6 * _S)
        tree.set(f'{service}/agent-1', cpu_usec=7 * _S)
        tree.set(service, cpu_usec=16 * _S)
        tree.set('app.slice', cpu_usec=28 * _S)
        tree.make_unreadable(service, 'cpu.stat')
        second = _next(tree, first)

        assert _increment(second, 'service@orchestrator-x.service', Figure.CPU_USEC) == 2 * _S
        assert _increment(second, 'agents@orchestrator-x.service', Figure.CPU_USEC) == 3 * _S
        assert not _figures(second, 'agents@orchestrator-x.service').readable
        assert _increment(second, 'exited@app.slice', Figure.CPU_USEC) == 3 * _S
        assert not _figures(second, 'exited@app.slice').readable
        assert _figures(second, 'service@orchestrator-x.service').readable


class TestRootGauges:
    def test_root_memory_falls_back_and_the_kernel_has_no_swap_or_events(self, tree):
        tree.set('', anon=100, file=50, kernel=10, high=0)
        tree.set('app.slice', memory=100, swap=7, high=3)
        first = _first(tree)

        tree.set('app.slice', high=5)
        second = _next(tree, first)

        kernel = _figures(second, 'kernel')
        assert kernel.gauges[Figure.MEMORY] == 60
        assert Figure.SWAP not in kernel.gauges
        assert kernel.increments is not None
        assert Figure.MEMORY_HIGH_EVENTS not in kernel.increments
        assert Figure.MEMORY_MAX_EVENTS not in kernel.increments
        assert Figure.OOM_KILLS not in kernel.increments
        assert _increment(second, 'other@app.slice', Figure.MEMORY_HIGH_EVENTS) == 2


class TestAFileAbsentFromTheListing:
    def test_a_member_without_the_memory_controller_is_not_flagged(self, tree):
        udevd = 'system.slice/systemd-udevd.service'
        tree.set('system.slice', cpu_usec=10 * _S, memory=500)
        tree.set(udevd, cpu_usec=6 * _S, memory=300)
        tree.set(f'{udevd}/udev', cpu_usec=5 * _S, omit=MEMORY_FILES)
        first = _first(tree)

        tree.set(f'{udevd}/udev', cpu_usec=7 * _S)
        tree.set(udevd, cpu_usec=8 * _S)
        tree.set('system.slice', cpu_usec=12 * _S)
        second = _next(tree, first)

        udev = _figures(second, 'other@system.slice')
        assert udev.readable
        assert udev.members == 1
        assert _increment(second, 'other@system.slice', Figure.CPU_USEC) == 2 * _S
        assert Figure.MEMORY not in udev.gauges
        exited = _figures(second, 'exited@system.slice')
        assert exited.readable
        assert exited.gauges[Figure.MEMORY] == 500


class TestGauges:
    def test_sums_over_members_plus_residuals_clamped_at_zero(self, tree):
        tree.set('app.slice', memory=100, anon=40)
        tree.set('app.slice/a.scope', memory=70, anon=30)
        tree.set('app.slice/b.scope', memory=50, anon=5)
        tree.set('app.slice/sccache.service', memory=9)

        accounting = _first(tree)

        other = _figures(accounting, 'other@app.slice')
        assert other.members == 2
        assert other.gauges[Figure.MEMORY] == 120
        assert other.gauges[Figure.ANON] == 35
        exited = _figures(accounting, 'exited@app.slice')
        assert exited.gauges[Figure.MEMORY] == 0
        assert exited.gauges[Figure.ANON] == 5
        _no_negative(accounting)


class TestState:
    def test_the_state_round_trips_through_json(self, tree):
        tree.set('app.slice', cpu_usec=20 * _S)
        tree.set('app.slice/sccache.service', cpu_usec=10 * _S)
        tree.make_unreadable('app.slice/sccache.service', 'memory.stat')

        state = _first(tree).state

        assert TreeState.from_json(state.to_json()) == state

    @pytest.mark.parametrize(
        'text',
        ['', 'not json', '{}', '[]', '{"version": 999, "ts": 1, "listing": {}, "readings": {}}',
         '{"version": 1, "ts": "x", "listing": {}, "readings": {}}'],
    )
    def test_a_corrupt_or_foreign_state_is_no_previous_state(self, text):
        assert TreeState.from_json(text) is None

    def test_a_cgroup_that_left_the_listing_is_pruned(self, tree):
        tree.set('app.slice')
        tree.set(_SCOPE_S, cpu_usec=1)
        first = _first(tree)
        assert ('app.slice', _SCOPE_S.split('/')[1]) in first.state.listing

        tree.remove(_SCOPE_S)
        second = _next(tree, first)

        assert ('app.slice', _SCOPE_S.split('/')[1]) not in second.state.listing
        assert ('app.slice',) in second.state.listing


class TestReadTree:
    def test_children_are_read_before_their_parent(self, tree):
        tree.set('app.slice')
        tree.set('app.slice/orchestrator-x.service')
        tree.set('app.slice/orchestrator-x.service/daemon')
        tree.set('system.slice')

        order = [node.segments for node in read_tree(tree.root).nodes]

        for index, segments in enumerate(order):
            for child in order[index + 1 :]:
                assert child[: len(segments)] != segments, (segments, child)
        assert order[-1] == ()
