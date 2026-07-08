"""Tests for shared.psi — the re-homed PSI parser/reader (PRD DA1, DA-D9 reuse).

parse_pressure_file is re-homed VERBATIM from sampler/src/sampler/metrics.py
(behavior-identical) — see that module's sibling test suite
(sampler/tests/test_load_metrics.py) for the original regression guard. The
fixture text literals below reuse the same PSI_CPU_TEXT/PSI_MEM_TEXT/PSI_IO_TEXT
shapes so both suites pin the same parsing contract.
"""

from __future__ import annotations

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
