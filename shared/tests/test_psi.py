"""Tests for shared.psi — the re-homed PSI parser/reader (PRD DA1, DA-D9 reuse).

parse_pressure_file is re-homed VERBATIM from sampler/src/sampler/metrics.py
(behavior-identical) — see that module's sibling test suite
(sampler/tests/test_load_metrics.py) for the original regression guard. The
fixture text literals below reuse the same PSI_CPU_TEXT/PSI_MEM_TEXT/PSI_IO_TEXT
shapes so both suites pin the same parsing contract.
"""

from __future__ import annotations

import types

import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures (same shapes as sampler/tests/test_load_metrics.py)
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


class TestParsePressureFileRehome:
    def test_both_lines_extracted(self):
        from shared.psi import parse_pressure_file

        result = parse_pressure_file(PSI_CPU_TEXT)
        assert result == {'some_avg10': 2.50, 'full_avg10': 0.30}

    def test_missing_full_defaults_to_zero(self):
        from shared.psi import parse_pressure_file

        result = parse_pressure_file(PSI_MEM_TEXT)
        assert result == {'some_avg10': 1.23, 'full_avg10': 0.0}

    def test_io_both_lines(self):
        from shared.psi import parse_pressure_file

        result = parse_pressure_file(PSI_IO_TEXT)
        assert result == {'some_avg10': 0.75, 'full_avg10': 0.45}

    def test_float_precision(self):
        from shared.psi import parse_pressure_file

        text = 'some avg10=99.99 avg60=0.00 avg300=0.00 total=0\n'
        result = parse_pressure_file(text)
        assert result is not None
        assert result['some_avg10'] == pytest.approx(99.99)
        assert result['full_avg10'] == 0.0

    def test_total_miss_returns_none(self):
        """A total parse miss (no some/full avg10 line) must return None sentinel."""
        from shared.psi import parse_pressure_file

        # Total miss — garbage text with no recognisable avg10 fields
        assert parse_pressure_file('garbage line with no avg fields\n') is None
        # Empty string — also a total miss
        assert parse_pressure_file('') is None

    def test_partial_miss_is_not_none(self):
        """Partial miss (some present, full absent) is a legitimate kernel state, not None."""
        from shared.psi import parse_pressure_file

        assert parse_pressure_file(PSI_MEM_TEXT) is not None


def _saturation_cfg():
    """Duck-typed cfg stub shaped like DA2's PsiAdmissionConfig (sibling task, not landed)."""
    return types.SimpleNamespace(
        cpu_some_avg10=85.0,
        mem_some_avg10=15.0,
        mem_full_avg10=3.0,
        io_some_avg10=40.0,
    )


class TestPsiSampleSaturated:
    def test_frozen(self):
        import dataclasses

        from shared.psi import PsiSample

        sample = PsiSample(
            cpu_some10=10.0,
            mem_some10=5.0,
            mem_full10=1.0,
            io_some10=5.0,
            read_ok=True,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            sample.cpu_some10 = 99.0  # type: ignore[misc]

    def test_cpu_some_only_trips_saturation(self):
        from shared.psi import PsiSample

        sample = PsiSample(
            cpu_some10=85.0,
            mem_some10=10.0,
            mem_full10=2.0,
            io_some10=30.0,
            read_ok=True,
        )
        assert sample.saturated(_saturation_cfg()) is True

    def test_mem_some_only_trips_saturation(self):
        from shared.psi import PsiSample

        sample = PsiSample(
            cpu_some10=50.0,
            mem_some10=15.0,
            mem_full10=2.0,
            io_some10=30.0,
            read_ok=True,
        )
        assert sample.saturated(_saturation_cfg()) is True

    def test_mem_full_only_trips_saturation(self):
        from shared.psi import PsiSample

        sample = PsiSample(
            cpu_some10=50.0,
            mem_some10=10.0,
            mem_full10=3.0,
            io_some10=30.0,
            read_ok=True,
        )
        assert sample.saturated(_saturation_cfg()) is True

    def test_io_some_only_trips_saturation(self):
        from shared.psi import PsiSample

        sample = PsiSample(
            cpu_some10=50.0,
            mem_some10=10.0,
            mem_full10=2.0,
            io_some10=40.0,
            read_ok=True,
        )
        assert sample.saturated(_saturation_cfg()) is True

    def test_all_under_threshold_not_saturated(self):
        from shared.psi import PsiSample

        sample = PsiSample(
            cpu_some10=50.0,
            mem_some10=10.0,
            mem_full10=2.0,
            io_some10=30.0,
            read_ok=True,
        )
        assert sample.saturated(_saturation_cfg()) is False


class TestReadPsiSampleHappyPath:
    def _fake_read(self):
        # Note: the memory pressure file is read under the name 'memory', not 'mem'.
        sources = {'cpu': PSI_CPU_TEXT, 'memory': PSI_MEM_TEXT, 'io': PSI_IO_TEXT}

        def read(name):
            return sources[name]

        return read

    def test_maps_per_metric_fields(self):
        from shared.psi import read_psi_sample

        sample = read_psi_sample(read=self._fake_read())

        assert sample.read_ok is True
        assert sample.cpu_some10 == pytest.approx(2.50)
        assert sample.mem_some10 == pytest.approx(1.23)
        assert sample.mem_full10 == 0.0
        assert sample.io_some10 == pytest.approx(0.75)

    def test_saturated_reflects_injected_values(self):
        from shared.psi import read_psi_sample

        sample = read_psi_sample(read=self._fake_read())

        low_cpu_cfg = types.SimpleNamespace(
            cpu_some_avg10=1.0,
            mem_some_avg10=999.0,
            mem_full_avg10=999.0,
            io_some_avg10=999.0,
        )
        assert sample.saturated(low_cpu_cfg) is True

        all_high_cfg = types.SimpleNamespace(
            cpu_some_avg10=999.0,
            mem_some_avg10=999.0,
            mem_full_avg10=999.0,
            io_some_avg10=999.0,
        )
        assert sample.saturated(all_high_cfg) is False


class TestReadPsiSampleFailOpen:
    def _zero_threshold_cfg(self):
        """A maximally saturating cfg (all thresholds 0.0) — proves fail-open still wins."""
        return types.SimpleNamespace(
            cpu_some_avg10=0.0,
            mem_some_avg10=0.0,
            mem_full_avg10=0.0,
            io_some_avg10=0.0,
        )

    def _assert_sentinel(self, sample):
        assert sample.read_ok is False
        assert sample.cpu_some10 == 0.0
        assert sample.mem_some10 == 0.0
        assert sample.mem_full10 == 0.0
        assert sample.io_some10 == 0.0
        assert sample.saturated(self._zero_threshold_cfg()) is False

    def test_reader_raises_file_not_found_fails_open(self):
        from shared.psi import read_psi_sample

        def read(name):
            if name == 'cpu':
                raise FileNotFoundError('/proc/pressure/cpu')
            return PSI_MEM_TEXT if name == 'memory' else PSI_IO_TEXT

        self._assert_sentinel(read_psi_sample(read=read))

    def test_reader_raises_generic_oserror_fails_open(self):
        from shared.psi import read_psi_sample

        def read(name):
            if name == 'memory':
                raise PermissionError('/proc/pressure/memory')
            return PSI_CPU_TEXT if name == 'cpu' else PSI_IO_TEXT

        self._assert_sentinel(read_psi_sample(read=read))

    def test_unparseable_source_fails_open(self):
        from shared.psi import read_psi_sample

        def read(name):
            if name == 'io':
                return 'garbage line with no avg fields\n'
            return PSI_CPU_TEXT if name == 'cpu' else PSI_MEM_TEXT

        self._assert_sentinel(read_psi_sample(read=read))
