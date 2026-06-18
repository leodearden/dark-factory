"""Tests for sampler.metrics — pure PSI parser and process-metric counters.

All functions under test take injected data (fixture text, fake process objects,
fd9-exists predicates) so they are fully deterministic and safe to run in pytest.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# Realistic /proc/pressure/* text with both some and full lines
PSI_CPU_TEXT = (
    'some avg10=2.50 avg60=1.80 avg300=1.20 total=123456\n'
    'full avg10=0.30 avg60=0.20 avg300=0.10 total=9876\n'
)

# Memory pressure — only 'some' line present (full missing)
PSI_MEM_TEXT = 'some avg10=1.23 avg60=0.90 avg300=0.50 total=654321\n'

# IO pressure — both lines
PSI_IO_TEXT = (
    'some avg10=0.75 avg60=0.60 avg300=0.40 total=11111\n'
    'full avg10=0.45 avg60=0.30 avg300=0.20 total=2222\n'
)


# ---------------------------------------------------------------------------
# Step-1 tests: parse_pressure_file and collect_psi
# ---------------------------------------------------------------------------


class TestParsePressureFile:
    def test_both_lines_extracted(self):
        from sampler.metrics import parse_pressure_file

        result = parse_pressure_file(PSI_CPU_TEXT)
        assert result == {'some_avg10': 2.50, 'full_avg10': 0.30}

    def test_missing_full_defaults_to_zero(self):
        from sampler.metrics import parse_pressure_file

        result = parse_pressure_file(PSI_MEM_TEXT)
        assert result == {'some_avg10': 1.23, 'full_avg10': 0.0}

    def test_io_both_lines(self):
        from sampler.metrics import parse_pressure_file

        result = parse_pressure_file(PSI_IO_TEXT)
        assert result == {'some_avg10': 0.75, 'full_avg10': 0.45}

    def test_float_precision(self):
        from sampler.metrics import parse_pressure_file

        text = 'some avg10=99.99 avg60=0.00 avg300=0.00 total=0\n'
        result = parse_pressure_file(text)
        assert result['some_avg10'] == pytest.approx(99.99)
        assert result['full_avg10'] == 0.0

    def test_total_parse_miss_returns_none(self):
        """A total parse miss (no some/full avg10 line) must return None sentinel.

        Current code pre-seeds {some_avg10:0.0, full_avg10:0.0} and always
        returns that dict — these assertions FAIL (RED) until step-4 implements
        the sentinel.  The partial-miss boundary pin (PSI_MEM_TEXT has only
        'some') is included to verify None is NOT triggered for partial misses.
        """
        from sampler.metrics import parse_pressure_file

        # Total miss — garbage text with no recognisable avg10 fields
        assert parse_pressure_file('garbage line with no avg fields\n') is None
        # Empty string — also a total miss
        assert parse_pressure_file('') is None
        # Partial miss (some present, full absent) — NOT a total miss, must not be None
        assert parse_pressure_file(PSI_MEM_TEXT) is not None


class TestCollectPsi:
    def _fake_read(self, mapping: dict[str, str]):
        """Return a closure that looks up fixture text by name."""
        def read(name: str) -> str:
            return mapping[name]
        return read

    def test_returns_exactly_six_keys(self):
        from sampler.metrics import collect_psi

        mapping = {'cpu': PSI_CPU_TEXT, 'memory': PSI_MEM_TEXT, 'io': PSI_IO_TEXT}
        result = collect_psi(read=self._fake_read(mapping))
        expected_keys = {
            'psi_cpu_some_avg10', 'psi_cpu_full_avg10',
            'psi_mem_some_avg10', 'psi_mem_full_avg10',
            'psi_io_some_avg10', 'psi_io_full_avg10',
        }
        assert set(result.keys()) == expected_keys

    def test_values_match_parsed_text(self):
        from sampler.metrics import collect_psi

        mapping = {'cpu': PSI_CPU_TEXT, 'memory': PSI_MEM_TEXT, 'io': PSI_IO_TEXT}
        result = collect_psi(read=self._fake_read(mapping))

        assert result['psi_cpu_some_avg10'] == pytest.approx(2.50)
        assert result['psi_cpu_full_avg10'] == pytest.approx(0.30)
        assert result['psi_mem_some_avg10'] == pytest.approx(1.23)
        assert result['psi_mem_full_avg10'] == 0.0          # missing full -> 0.0
        assert result['psi_io_some_avg10'] == pytest.approx(0.75)
        assert result['psi_io_full_avg10'] == pytest.approx(0.45)

    def test_all_values_are_floats(self):
        from sampler.metrics import collect_psi

        mapping = {'cpu': PSI_CPU_TEXT, 'memory': PSI_MEM_TEXT, 'io': PSI_IO_TEXT}
        result = collect_psi(read=self._fake_read(mapping))
        for key, val in result.items():
            assert isinstance(val, float), f'{key} is not float: {val!r}'


# ---------------------------------------------------------------------------
# Step-3 tests: count_occt_queue_depth, count_verify_concurrency, sum_verify_rss
# ---------------------------------------------------------------------------


class FakeProc:
    """Lightweight process stub exposing the attributes the impl reads."""

    def __init__(
        self,
        pid: int,
        name: str = 'bash',
        cmdline: list[str] | None = None,
        rss: int = 0,
        children: list['FakeProc'] | None = None,
    ):
        self.pid = pid
        self._name = name
        self._cmdline = cmdline if cmdline is not None else []
        self._rss = rss
        self._children = children if children is not None else []

    def name(self) -> str:
        return self._name

    def cmdline(self) -> list[str]:
        return self._cmdline

    def memory_info(self):
        class _MI:
            def __init__(self, rss):
                self.rss = rss
        return _MI(self._rss)

    def children(self, recursive: bool = False) -> list['FakeProc']:
        if recursive:
            result = []
            for ch in self._children:
                result.append(ch)
                result.extend(ch.children(recursive=True))
            return result
        return list(self._children)


class TestCountOcctQueueDepth:
    def test_counts_only_occt_processes_waiting_on_semaphore(self):
        """Processes matching cargo-test-occt-gated bash without fd/9 are counted."""
        from sampler.metrics import count_occt_queue_depth

        procs = [
            FakeProc(1, 'bash', ['bash', 'cargo-test-occt-gated']),   # waiting (fd9 absent)
            FakeProc(2, 'bash', ['bash', 'cargo-test-occt-gated']),   # has fd9 → not waiting
            FakeProc(3, 'bash', ['bash', 'cargo-test-occt-gated']),   # waiting
            FakeProc(4, 'python', ['python', 'something']),            # not occt
        ]
        fd9_has = {2}  # pid 2 has fd/9

        def fd9_exists(pid: int) -> bool:
            return pid in fd9_has

        result = count_occt_queue_depth(procs, fd9_exists)
        assert result == 2

    def test_non_matching_processes_ignored(self):
        from sampler.metrics import count_occt_queue_depth

        procs = [
            FakeProc(10, 'python', ['python', 'script.py']),
            FakeProc(11, 'bash', ['bash', 'other-script']),
        ]
        result = count_occt_queue_depth(procs, lambda pid: False)
        assert result == 0

    def test_all_waiting(self):
        from sampler.metrics import count_occt_queue_depth

        procs = [
            FakeProc(i, 'bash', ['bash', 'cargo-test-occt-gated']) for i in range(5)
        ]
        result = count_occt_queue_depth(procs, lambda pid: False)
        assert result == 5

    def test_all_have_fd9_so_none_counted(self):
        from sampler.metrics import count_occt_queue_depth

        procs = [
            FakeProc(i, 'bash', ['bash', 'cargo-test-occt-gated']) for i in range(3)
        ]
        result = count_occt_queue_depth(procs, lambda pid: True)
        assert result == 0

    def test_realistic_path_with_extension_counted(self):
        """The real script is invoked with its .sh path — must be recognised.

        Realistic invocation: ['/bin/bash', 'reify/scripts/cargo-test-occt-gated.sh']
        The bare-token form ['bash', 'cargo-test-occt-gated'] was the only fixture
        used in prior tests, giving false confidence.  This test proves the
        path-with-extension form is also counted.

        Current _is_occt_gated uses list membership ('cargo-test-occt-gated' in cmdline),
        which requires an element to equal the bare token exactly.
        'reify/scripts/cargo-test-occt-gated.sh' != 'cargo-test-occt-gated', so count==0
        and this test FAILS against the current impl (confirming the RED phase).
        """
        from sampler.metrics import count_occt_queue_depth

        procs = [
            # Realistic form: bash running the script by its full relative path
            FakeProc(1, 'bash', ['/bin/bash', 'reify/scripts/cargo-test-occt-gated.sh']),
            # Another common form: bash + path with leading ./
            FakeProc(2, 'bash', ['bash', './scripts/cargo-test-occt-gated.sh', 'cargo', 'test']),
            # Unrelated process — must NOT be counted
            FakeProc(3, 'python', ['python', 'unrelated']),
        ]
        result = count_occt_queue_depth(procs, lambda _pid: False)
        # pids 1 and 2 match (path contains 'cargo-test-occt-gated'), pid 3 does not
        assert result == 2


class TestCountVerifyConcurrency:
    def test_counts_verify_sh_argv0(self):
        from sampler.metrics import count_verify_concurrency

        procs = [
            FakeProc(1, 'bash', ['verify.sh', 'arg1']),              # matches
            FakeProc(2, 'bash', ['/usr/local/bin/verify.sh', 'x']),  # basename matches
            FakeProc(3, 'bash', ['bash', 'verify.sh']),               # verify.sh NOT argv[0]
            FakeProc(4, 'python', ['python', 'app.py']),
        ]
        result = count_verify_concurrency(procs)
        assert result == 2

    def test_no_verify_processes(self):
        from sampler.metrics import count_verify_concurrency

        procs = [FakeProc(1, 'python', ['python', 'main.py'])]
        result = count_verify_concurrency(procs)
        assert result == 0

    def test_empty_cmdline_skipped(self):
        from sampler.metrics import count_verify_concurrency

        procs = [FakeProc(1, 'bash', [])]
        result = count_verify_concurrency(procs)
        assert result == 0


class TestSumVerifyRss:
    def test_sums_process_and_children_rss(self):
        from sampler.metrics import sum_verify_rss

        child1 = FakeProc(101, 'bash', [], rss=512)
        child2 = FakeProc(102, 'bash', [], rss=256)
        parent = FakeProc(100, 'bash', ['verify.sh'], rss=1024, children=[child1, child2])
        other = FakeProc(200, 'python', ['python'], rss=9999)

        result = sum_verify_rss([parent, other])
        # parent (1024) + child1 (512) + child2 (256) = 1792
        assert result == 1792

    def test_dedupes_shared_pids(self):
        """If a child appears in multiple process trees, count its RSS only once."""
        from sampler.metrics import sum_verify_rss

        shared_child = FakeProc(999, 'bash', [], rss=100)
        parent1 = FakeProc(1, 'bash', ['verify.sh'], rss=200, children=[shared_child])
        parent2 = FakeProc(2, 'bash', ['verify.sh'], rss=300, children=[shared_child])

        result = sum_verify_rss([parent1, parent2])
        # parent1 (200) + parent2 (300) + shared_child counted once (100) = 600
        assert result == 600

    def test_no_verify_processes_returns_zero(self):
        from sampler.metrics import sum_verify_rss

        procs = [FakeProc(1, 'python', ['python'], rss=1000)]
        result = sum_verify_rss(procs)
        assert result == 0

    def test_verify_without_children(self):
        from sampler.metrics import sum_verify_rss

        proc = FakeProc(1, 'bash', ['verify.sh'], rss=4096)
        result = sum_verify_rss([proc])
        assert result == 4096


# ---------------------------------------------------------------------------
# Step-1 (plan) tests: collect_process_metrics propagation on total failure
# ---------------------------------------------------------------------------


class TestCollectProcessMetricsDegrade:
    def test_total_proc_iter_failure_propagates(self):
        """A total proc_iter failure must propagate, not be swallowed into 0.0.

        Current code wraps proc_iter in `except Exception: procs = []` which
        converts a RuntimeError scan failure into a fabricated healthy-zero dict.
        This test confirms the RED premise: the exception is NOT raised.
        """
        from sampler.metrics import collect_process_metrics

        def raising(*_a, **_kw):
            raise RuntimeError('psutil scan down')

        with pytest.raises(RuntimeError):
            collect_process_metrics(proc_iter=raising, fd9_exists=lambda _pid: False)
