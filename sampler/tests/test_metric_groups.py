"""Tests for the metric-group table in sampler.metrics.

Every test builds its own fixture specs and groups unless it is explicitly
about the live table, ``METRIC_GROUPS``.
"""

from __future__ import annotations

import dataclasses

import pytest
from shared.psi import RunqueueReading

from sampler.metrics import (
    ARM_METRICS,
    METRIC_GROUPS,
    TICK_CLOCK,
    GroupName,
    MetricGroup,
    MetricSpec,
    SampleKey,
    arm_metrics,
    collect_load_metrics,
    exact_metric_names,
)


def _no_samples():
    return {}


def _group(name: GroupName, *metrics: MetricSpec, windowed: bool = True) -> MetricGroup:
    return MetricGroup(name, collect=_no_samples, metrics=metrics, windowed=windowed)


READ_OK = MetricSpec('fixture_read_ok')
RATIO = MetricSpec('fixture_ratio', arm='fixture_ratio_arm', readability=READ_OK)
LEAF_READ_OK = MetricSpec('fixture_leaf_read_ok', per_leaf=True)
LEAF_PRESSURE = MetricSpec(
    'fixture_leaf_pressure', per_leaf=True, arm='fixture_leaf_arm', readability=LEAF_READ_OK
)
LOAD = _group(GroupName.LOAD, RATIO, READ_OK, LEAF_PRESSURE, LEAF_READ_OK)

HOST_SOME = MetricSpec('fixture_host_some', arm='fixture_host_arm')
HOST_FULL = MetricSpec('fixture_host_full')
PSI = _group(GroupName.PSI, HOST_SOME, HOST_FULL, windowed=False)

DEPTH = MetricSpec('fixture_depth')
PROCESS = _group(GroupName.PROCESS, DEPTH)

TABLE = (PSI, PROCESS, LOAD)


class TestAdmission:
    """A group admits a host-level key by exact name, a per-leaf key by name plus a leaf."""

    def test_an_exact_name_with_no_leaf_is_admitted(self):
        assert LOAD.admits(RATIO.key())
        assert LOAD.admits(SampleKey('fixture_read_ok'))

    @pytest.mark.parametrize(
        'leaf',
        ['orchestrator-dark-factory.service', 'df-dark_factory.slice', 'weird:leaf.slice'],
    )
    def test_a_per_leaf_name_with_a_non_empty_leaf_is_admitted(self, leaf):
        assert LOAD.admits(LEAF_PRESSURE.key(leaf))
        assert LOAD.admits(LEAF_READ_OK.key(leaf))

    @pytest.mark.parametrize('leaf', [None, ''])
    def test_a_per_leaf_name_without_a_leaf_is_rejected(self, leaf):
        """A per-leaf metric without a leaf is evidence about no cgroup."""
        assert not LOAD.admits(SampleKey('fixture_leaf_pressure', leaf))

    def test_a_misspelled_name_is_rejected(self):
        assert not LOAD.admits(SampleKey('fixture_ratoi'))
        assert not LOAD.admits(SampleKey('fixture_leaf_pressur', 'leaf.service'))

    def test_an_exact_name_carrying_a_leaf_is_rejected(self):
        assert not LOAD.admits(SampleKey('fixture_ratio', 'leaf.service'))

    def test_a_group_with_no_per_leaf_spec_degenerates_to_exact_membership(self):
        assert PSI.admits(SampleKey('fixture_host_some'))
        assert not PSI.admits(SampleKey('fixture_host_somex'))
        assert not PSI.admits(SampleKey('fixture_host_some', 'leaf'))

    def test_exact_and_per_leaf_are_derived_from_the_specs(self):
        assert LOAD.exact == frozenset({'fixture_ratio', 'fixture_read_ok'})
        assert LOAD.per_leaf == frozenset({'fixture_leaf_pressure', 'fixture_leaf_read_ok'})
        assert PSI.per_leaf == frozenset()

    def test_a_spec_builds_its_own_sample_key(self):
        assert RATIO.key() == SampleKey('fixture_ratio')
        assert LEAF_PRESSURE.key('a.service') == SampleKey('fixture_leaf_pressure', 'a.service')


class TestConstructionInvariants:
    """A malformed spec or group fails where it is declared, naming what is wrong."""

    @pytest.mark.parametrize('name', ['', 'own_cpu:leaf'])
    def test_a_spec_name_must_be_non_empty_and_free_of_colons(self, name):
        with pytest.raises(ValueError, match="non-empty and contain no ':'") as raised:
            MetricSpec(name)
        assert repr(name) in str(raised.value)

    def test_spec_names_are_unique_within_a_group(self):
        with pytest.raises(ValueError, match='unique within a group') as raised:
            _group(GroupName.LOAD, MetricSpec('fixture_twice'), MetricSpec('fixture_twice'))
        assert 'fixture_twice' in str(raised.value)

    def test_a_readability_spec_belongs_to_the_same_group(self):
        orphan = MetricSpec('fixture_orphan_read_ok')
        reader = MetricSpec('fixture_reader', readability=orphan)
        with pytest.raises(ValueError, match='same group') as raised:
            _group(GroupName.LOAD, reader)
        assert 'fixture_reader' in str(raised.value)
        assert 'fixture_orphan_read_ok' in str(raised.value)

    @pytest.mark.parametrize('per_leaf', [True, False])
    def test_a_readability_spec_shares_its_metrics_per_leaf(self, per_leaf):
        readability = MetricSpec('fixture_read_ok', per_leaf=not per_leaf)
        with pytest.raises(ValueError, match='per_leaf') as raised:
            MetricSpec('fixture_value', per_leaf=per_leaf, readability=readability)
        assert 'fixture_value' in str(raised.value)
        assert 'fixture_read_ok' in str(raised.value)

    @pytest.mark.parametrize(('value', 'field'), [(RATIO, 'name'), (LOAD, 'windowed')])
    def test_specs_and_groups_are_frozen(self, value, field):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, field, None)


class TestArmMetrics:
    def test_each_arm_maps_to_the_spec_recording_it(self):
        assert dict(arm_metrics(TABLE)) == {
            'fixture_host_arm': HOST_SOME,
            'fixture_ratio_arm': RATIO,
            'fixture_leaf_arm': LEAF_PRESSURE,
        }

    def test_a_spec_without_an_arm_contributes_nothing(self):
        assert dict(arm_metrics((PROCESS,))) == {}

    def test_two_specs_carrying_one_arm_are_rejected(self):
        twin = MetricSpec('fixture_twin', arm='fixture_host_arm')
        table = (PSI, _group(GroupName.PROCESS, twin))
        with pytest.raises(ValueError, match='fixture_host_arm') as raised:
            arm_metrics(table)
        assert 'fixture_host_some' in str(raised.value)
        assert 'fixture_twin' in str(raised.value)

    @pytest.mark.parametrize('arms', [arm_metrics(TABLE), ARM_METRICS])
    def test_the_result_is_read_only(self, arms):
        with pytest.raises(TypeError):
            arms['fixture_new_arm'] = DEPTH


class TestExactMetricNames:
    def test_the_sorted_exact_names_of_exactly_the_named_groups(self):
        names = exact_metric_names(TABLE, {GroupName.PSI, GroupName.LOAD})
        assert names == ('fixture_host_full', 'fixture_host_some', 'fixture_ratio', 'fixture_read_ok')

    def test_per_leaf_names_never_appear(self):
        names = exact_metric_names(TABLE, {GroupName.LOAD})
        assert not set(names) & LOAD.per_leaf

    def test_an_unnamed_group_contributes_nothing(self):
        assert exact_metric_names(TABLE, {GroupName.PROCESS}) == ('fixture_depth',)
        assert exact_metric_names(TABLE, set()) == ()


class TestTheLiveTable:
    def test_one_group_per_group_name_in_a_stable_order(self):
        assert tuple(group.name for group in METRIC_GROUPS) == tuple(GroupName)

    def test_metric_names_are_distinct_across_the_whole_table(self):
        names = [spec.name for group in METRIC_GROUPS for spec in group.metrics]
        assert len(names) == len(set(names)), sorted(names)

    def test_only_the_kernel_windowed_psi_group_is_recorded_unwindowed(self):
        windowed = {group.name: group.windowed for group in METRIC_GROUPS}
        assert windowed == {GroupName.PSI: False, GroupName.PROCESS: True, GroupName.LOAD: True}

    def test_arm_metrics_is_the_projection_of_the_live_table(self):
        assert dict(ARM_METRICS) == dict(arm_metrics(METRIC_GROUPS))

    def test_the_tick_clock_is_an_exact_metric_of_the_load_group(self):
        (load,) = [group for group in METRIC_GROUPS if group.name is GroupName.LOAD]
        assert TICK_CLOCK in load.metrics
        assert not TICK_CLOCK.per_leaf


class TestTheTickClockIsEmittedOnEveryCompletedLoadCollection:
    """Every calibration coverage row divides by this metric's row count.

    So it must be written whenever the load collector completes: with no leaf
    discovered and with /proc/stat unreadable alike. Were it per-leaf or
    per-successful-read, coverage would be computed over the ticks something
    happened to be readable.
    """

    @pytest.mark.parametrize('read_ok', [True, False])
    def test_with_no_leaves_and_either_runqueue_outcome(self, tmp_path, read_ok):
        emitted = collect_load_metrics(
            read_runqueue=lambda **_kwargs: RunqueueReading(0.0, read_ok),
            own_cgroup_path='/system.slice/not-a-user-manager.service',
            cgroup_root=tmp_path,
        )

        assert TICK_CLOCK.key() in emitted, sorted(emitted)
