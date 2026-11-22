"""Host-level C1 figures (``sampler.host_memory``) over fixture /proc and cgroup trees.

PRD ``plans/per-consumer-resource-recording-prd.md`` C1 host-level metrics:
the background reclaim and compaction threads' CPU (µs increments from
``/proc/<pid>/stat``, summed per thread kind), ``MemTotal - MemAvailable``
from ``/proc/meminfo``, and host program memory and file cache from the
root's ``memory.stat``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sampler.host_memory import (
    HostMemoryStat,
    ReclaimState,
    ReclaimThreadKind,
    read_host_memory_stat,
    read_host_used_bytes,
    read_reclaim_threads,
    reclaim_increments,
)

_INTERVAL = 30
_T0 = 1_000_000
_TICK_USEC = 1_000_000 // os.sysconf('SC_CLK_TCK')


def _stat_line(pid: int, comm: str, utime: int, stime: int) -> str:
    """A real-format /proc/<pid>/stat line: comm in parentheses, then fields 3-52."""
    fields = ['S', '2', '0', '0', '0', '-1', '69238880', '0', '0', '0', '0']
    fields += [str(utime), str(stime), '0', '0', '20', '0', '1', '0', '79']
    fields += ['0'] * 32
    return f'{pid} ({comm}) {" ".join(fields)}\n'


class _Proc:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir()

    def thread(self, pid: int, comm: str, utime: int, stime: int = 0) -> None:
        directory = self.root / str(pid)
        directory.mkdir(exist_ok=True)
        (directory / 'stat').write_text(_stat_line(pid, comm, utime, stime))

    def exit(self, pid: int) -> None:
        (self.root / str(pid) / 'stat').unlink()
        (self.root / str(pid)).rmdir()


@pytest.fixture
def proc(tmp_path) -> _Proc:
    proc = _Proc(tmp_path / 'proc')
    (proc.root / 'self').mkdir()
    (proc.root / 'meminfo').write_text('MemTotal: 1 kB\n')
    return proc


def _increments(proc: _Proc, previous: ReclaimState | None, now: int):
    return reclaim_increments(
        previous, read_reclaim_threads(proc.root), now=now, nominal_interval=_INTERVAL
    )


class TestReclaimThreads:
    def test_kinds_by_comm_and_everything_else_ignored(self, proc):
        proc.thread(101, 'kswapd0', utime=0, stime=10)
        proc.thread(102, 'kswapd1', utime=1, stime=4)
        proc.thread(103, 'kcompactd0', utime=0, stime=3)
        proc.thread(104, 'kworker/0:1', utime=0, stime=50)
        proc.thread(105, 'kswapd', utime=0, stime=50)
        proc.thread(106, 'kswapd0x', utime=0, stime=50)
        proc.thread(107, 'x) kswapd0', utime=0, stime=50)
        _, state = _increments(proc, None, _T0)

        proc.thread(101, 'kswapd0', utime=0, stime=14)
        proc.thread(102, 'kswapd1', utime=1, stime=6)
        proc.thread(103, 'kcompactd0', utime=0, stime=4)
        for pid in (104, 105, 106, 107):
            proc.thread(pid, 'kworker/0:1', utime=0, stime=900)
        increments, _ = _increments(proc, state, _T0 + _INTERVAL)

        assert increments == {
            ReclaimThreadKind.KSWAPD: 6 * _TICK_USEC,
            ReclaimThreadKind.KCOMPACTD: 1 * _TICK_USEC,
        }

    def test_a_comm_holding_a_paren_and_space_is_split_at_the_last_paren(self, proc):
        proc.thread(201, 'kswapd0) (x', utime=0, stime=7)

        assert read_reclaim_threads(proc.root) == ()

    def test_a_new_thread_contributes_its_whole_value(self, proc):
        proc.thread(101, 'kswapd0', utime=0, stime=10)
        _, state = _increments(proc, None, _T0)

        proc.thread(301, 'kcompactd1', utime=2, stime=3)
        increments, _ = _increments(proc, state, _T0 + _INTERVAL)

        assert increments == {
            ReclaimThreadKind.KSWAPD: 0,
            ReclaimThreadKind.KCOMPACTD: 5 * _TICK_USEC,
        }

    def test_a_decreased_counter_is_a_reset(self, proc):
        proc.thread(101, 'kswapd0', utime=0, stime=10)
        _, state = _increments(proc, None, _T0)

        proc.thread(101, 'kswapd0', utime=0, stime=3)
        increments, _ = _increments(proc, state, _T0 + _INTERVAL)

        assert increments is not None
        assert increments[ReclaimThreadKind.KSWAPD] == 3 * _TICK_USEC

    def test_no_previous_state_and_a_gap_give_no_increments(self, proc):
        proc.thread(101, 'kswapd0', utime=0, stime=10)
        first, state = _increments(proc, None, _T0)

        late, _ = _increments(proc, state, _T0 + 4 * _INTERVAL)

        assert first is None
        assert late is None

    def test_the_state_round_trips_and_prunes_vanished_pids(self, proc):
        proc.thread(101, 'kswapd0', utime=0, stime=10)
        proc.thread(103, 'kcompactd0', utime=0, stime=3)
        _, first = _increments(proc, None, _T0)
        proc.exit(103)

        _, second = _increments(proc, first, _T0 + _INTERVAL)

        assert ReclaimState.from_json(second.to_json()) == second
        assert set(second.cpu_usec) == {101}
        assert second.ts == _T0 + _INTERVAL

    @pytest.mark.parametrize('text', ['', 'nope', '{}', '{"version": 1, "ts": 1, "cpu_usec": {"x": 1}}'])
    def test_a_corrupt_state_is_no_previous_state(self, text):
        assert ReclaimState.from_json(text) is None


class TestHostMemory:
    def test_used_is_total_minus_available(self, tmp_path):
        meminfo = tmp_path / 'meminfo'
        meminfo.write_text(
            'MemTotal:       131000000 kB\n'
            'MemFree:         2000000 kB\n'
            'MemAvailable:   31000000 kB\n'
            'SwapTotal:      67108860 kB\n'
        )

        assert read_host_used_bytes(meminfo) == 100_000_000 * 1024

    @pytest.mark.parametrize(
        'text', ['MemTotal: 131000000 kB\n', 'MemAvailable: 31000000 kB\n', '']
    )
    def test_a_missing_key_is_none_never_zero(self, tmp_path, text):
        meminfo = tmp_path / 'meminfo'
        meminfo.write_text(text)

        assert read_host_used_bytes(meminfo) is None

    def test_anon_and_file_come_from_the_root_memory_stat(self, tmp_path):
        (tmp_path / 'memory.stat').write_text('anon 44898074624\nfile 24411680768\nkernel 5\n')

        assert read_host_memory_stat(tmp_path) == HostMemoryStat(
            anon_bytes=44898074624, file_bytes=24411680768
        )

    def test_an_unreadable_root_memory_stat_is_absent(self, tmp_path):
        (tmp_path / 'memory.stat').write_text('anon lots\n')

        assert read_host_memory_stat(tmp_path) == HostMemoryStat(None, None)
