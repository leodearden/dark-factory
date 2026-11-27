"""Tests for shared.cgroup_stats — the one parser for cgroup v2 counter files.

PRD ``plans/per-consumer-resource-recording-prd.md`` D7: ``cpu.stat``,
``memory.stat`` and ``memory.events`` are key-value lines and have ONE parser;
pressure files come through ``shared.psi.parse_pressure_totals``. The fixture
texts below are copied from a live cgroup on the dev host (2026-11-17).
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import MutableMapping
from pathlib import Path

import pytest

from shared.cgroup_stats import (
    MEMORY_PEAK,
    SWAP_PEAK,
    CgroupReading,
    PressureResource,
    PressureTotals,
    Unreadable,
    parse_keyed_counters,
    parse_single_value,
    read_cgroup,
)

CPU_STAT_TEXT = (
    'usage_usec 160585967922369\n'
    'user_usec 127321389113066\n'
    'system_usec 33264578809302\n'
    'nice_usec 93252215050000\n'
    'core_sched.force_idle_usec 0\n'
    'nr_periods 0\n'
    'nr_throttled 0\n'
    'throttled_usec 0\n'
    'nr_bursts 0\n'
    'burst_usec 0\n'
)

MEMORY_STAT_TEXT = (
    'anon 44898074624\n'
    'file 24411680768\n'
    'kernel 5404991488\n'
    'shmem 142143488\n'
    'workingset_refault_anon 31593820556\n'
    'workingset_refault_file 257510567310\n'
    'pswpin 31416909893\n'
    'pswpout 22766278663\n'
    'pgsteal_kswapd 184713828355\n'
    'pgsteal_direct 312234084775\n'
)

MEMORY_EVENTS_TEXT = (
    'low 0\n'
    'high 532651168\n'
    'max 0\n'
    'oom 0\n'
    'oom_kill 0\n'
    'oom_group_kill 0\n'
    'sock_throttled 61489084\n'
)

CPU_PRESSURE_TEXT = (
    'some avg10=59.83 avg60=57.44 avg300=47.96 total=2564242951592\n'
    'full avg10=2.67 avg60=2.53 avg300=2.89 total=178949516551\n'
)
MEMORY_PRESSURE_TEXT = (
    'some avg10=0.18 avg60=0.25 avg300=0.89 total=89266459893\n'
    'full avg10=0.00 avg60=0.05 avg300=0.27 total=51877655090\n'
)
IO_PRESSURE_TEXT = (
    'some avg10=7.70 avg60=8.49 avg300=10.95 total=480697007206\n'
    'full avg10=0.00 avg60=0.18 avg300=0.65 total=82588805380\n'
)

_FULL_CGROUP = {
    'cpu.stat': CPU_STAT_TEXT,
    'memory.stat': MEMORY_STAT_TEXT,
    'memory.events': MEMORY_EVENTS_TEXT,
    'memory.current': '75084828672\n',
    'memory.swap.current': '45560156160\n',
    'memory.peak': '1073741824\n',
    'memory.swap.peak': '0\n',
    'cpu.pressure': CPU_PRESSURE_TEXT,
    'memory.pressure': MEMORY_PRESSURE_TEXT,
    'io.pressure': IO_PRESSURE_TEXT,
    'cgroup.procs': '',
}

_ROOT_ABSENT = (
    'memory.current',
    'memory.swap.current',
    'memory.events',
    'memory.peak',
    'memory.swap.peak',
)
_NO_MEMORY_CONTROLLER = tuple(name for name in _FULL_CGROUP if name.startswith('memory.'))


def _cgroup_dir(tmp_path: Path, files: dict[str, str]) -> Path:
    directory = tmp_path / 'cg'
    directory.mkdir()
    for name, text in files.items():
        (directory / name).write_text(text)
    return directory


def _read(directory: Path) -> CgroupReading | Unreadable:
    return read_cgroup(directory, present=frozenset(os.listdir(directory)))


class TestParseKeyedCounters:
    def test_cpu_stat_parses_every_key_to_int(self):
        counters = parse_keyed_counters(CPU_STAT_TEXT)
        assert counters['usage_usec'] == 160585967922369
        assert counters['user_usec'] == 127321389113066
        assert counters['system_usec'] == 33264578809302
        assert counters['core_sched.force_idle_usec'] == 0
        assert len(counters) == 10

    def test_memory_stat_carries_the_recorded_keys(self):
        counters = parse_keyed_counters(MEMORY_STAT_TEXT)
        assert counters == {
            'anon': 44898074624,
            'file': 24411680768,
            'kernel': 5404991488,
            'shmem': 142143488,
            'workingset_refault_anon': 31593820556,
            'workingset_refault_file': 257510567310,
            'pswpin': 31416909893,
            'pswpout': 22766278663,
            'pgsteal_kswapd': 184713828355,
            'pgsteal_direct': 312234084775,
        }

    def test_memory_events(self):
        counters = parse_keyed_counters(MEMORY_EVENTS_TEXT)
        assert counters == {
            'low': 0,
            'high': 532651168,
            'max': 0,
            'oom': 0,
            'oom_kill': 0,
            'oom_group_kill': 0,
            'sock_throttled': 61489084,
        }

    def test_empty_text_is_empty(self):
        assert parse_keyed_counters('') == {}

    @pytest.mark.parametrize(
        'bad_line',
        ['usage_usec', 'usage_usec twelve', 'usage_usec 1 2', 'usage_usec 1.5'],
    )
    def test_a_malformed_line_raises_naming_the_line(self, bad_line):
        text = f'user_usec 1\n{bad_line}\nsystem_usec 2\n'
        with pytest.raises(ValueError, match=bad_line):
            parse_keyed_counters(text)


class TestParseSingleValue:
    def test_parses_the_integer(self):
        assert parse_single_value('123\n') == 123

    @pytest.mark.parametrize('garbage', ['', 'max', '12 34', 'twelve'])
    def test_garbage_raises_naming_the_text(self, garbage):
        with pytest.raises(ValueError) as raised:
            parse_single_value(garbage)
        assert repr(garbage) in str(raised.value)


class TestReadCgroup:
    def test_a_full_cgroup_reads_every_file(self, tmp_path):
        reading = _read(_cgroup_dir(tmp_path, _FULL_CGROUP))

        assert isinstance(reading, CgroupReading)
        assert reading.cpu_stat is not None
        assert reading.cpu_stat['usage_usec'] == 160585967922369
        assert reading.memory_stat is not None
        assert reading.memory_stat['pgsteal_direct'] == 312234084775
        assert reading.memory_events is not None
        assert reading.memory_events['oom_kill'] == 0
        assert reading.memory_current == 75084828672
        assert reading.swap_current == 45560156160
        assert reading.memory_peak == 1073741824
        assert reading.swap_peak == 0
        assert reading.pressure == {
            PressureResource.CPU: PressureTotals(2564242951592, 178949516551),
            PressureResource.MEMORY: PressureTotals(89266459893, 51877655090),
            PressureResource.IO: PressureTotals(480697007206, 82588805380),
        }

    def test_the_root_lacks_five_files_and_is_still_a_reading(self, tmp_path):
        files = {k: v for k, v in _FULL_CGROUP.items() if k not in _ROOT_ABSENT}
        reading = _read(_cgroup_dir(tmp_path, files))

        assert isinstance(reading, CgroupReading)
        assert reading.memory_current is None
        assert reading.swap_current is None
        assert reading.memory_events is None
        assert reading.memory_peak is None
        assert reading.swap_peak is None
        assert reading.memory_stat is not None
        assert reading.cpu_stat is not None

    def test_a_cgroup_without_the_memory_controller_reads_cpu_only(self, tmp_path):
        files = {
            k: v for k, v in _FULL_CGROUP.items() if k not in _NO_MEMORY_CONTROLLER
        }
        reading = _read(_cgroup_dir(tmp_path, files))

        assert isinstance(reading, CgroupReading)
        assert reading.memory_stat is None
        assert reading.memory_events is None
        assert reading.memory_current is None
        assert reading.swap_current is None
        assert reading.memory_peak is None
        assert reading.swap_peak is None
        assert set(reading.pressure) == {PressureResource.CPU, PressureResource.IO}
        assert reading.cpu_stat is not None

    def test_a_file_outside_the_listing_is_never_opened(self, tmp_path):
        directory = _cgroup_dir(tmp_path, _FULL_CGROUP)
        (directory / 'memory.current').write_text('garbage\n')
        present = frozenset(os.listdir(directory)) - {'memory.current'}

        reading = read_cgroup(directory, present=present)

        assert isinstance(reading, CgroupReading)
        assert reading.memory_current is None

    @pytest.mark.parametrize(
        'name',
        [
            'cpu.stat',
            'memory.stat',
            'memory.current',
            'memory.pressure',
            'memory.peak',
            'memory.swap.peak',
        ],
    )
    def test_a_listed_file_that_cannot_be_opened_is_unreadable(self, tmp_path, name):
        files = {k: v for k, v in _FULL_CGROUP.items() if k != name}
        directory = _cgroup_dir(tmp_path, files)
        (directory / name).mkdir()

        reading = _read(directory)

        assert isinstance(reading, Unreadable)
        assert reading.file == name
        assert reading.reason

    @pytest.mark.parametrize(
        ('name', 'garbage'),
        [
            ('cpu.stat', 'usage_usec lots\n'),
            ('memory.events', 'high\n'),
            ('memory.swap.current', 'many\n'),
            ('io.pressure', 'some avg10=0.00\n'),
            ('memory.peak', 'garbage\n'),
            ('memory.swap.peak', 'max\n'),
        ],
    )
    def test_a_listed_file_that_does_not_parse_is_unreadable(
        self, tmp_path, name, garbage
    ):
        directory = _cgroup_dir(tmp_path, {**_FULL_CGROUP, name: garbage})

        reading = _read(directory)

        assert isinstance(reading, Unreadable)
        assert reading.file == name
        assert reading.reason

    def test_a_listed_file_that_vanished_is_unreadable(self, tmp_path):
        directory = _cgroup_dir(tmp_path, _FULL_CGROUP)
        present = frozenset(os.listdir(directory))
        (directory / 'cpu.stat').unlink()

        reading = read_cgroup(directory, present=present)

        assert isinstance(reading, Unreadable)
        assert reading.file == 'cpu.stat'

    def test_the_reading_is_immutable(self, tmp_path):
        reading = _read(_cgroup_dir(tmp_path, _FULL_CGROUP))
        assert isinstance(reading, CgroupReading)
        assert reading.cpu_stat is not None

        field = 'memory_current'
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(reading, field, 0)
        assert not isinstance(reading.cpu_stat, MutableMapping)
        assert not isinstance(reading.pressure, MutableMapping)


class TestFileNames:
    def test_the_peak_file_names_have_one_home(self):
        assert MEMORY_PEAK == 'memory.peak'
        assert SWAP_PEAK == 'memory.swap.peak'


class TestPressureResource:
    def test_each_resource_names_its_pressure_file(self):
        assert {r.file_name for r in PressureResource} == {
            'cpu.pressure',
            'memory.pressure',
            'io.pressure',
        }
