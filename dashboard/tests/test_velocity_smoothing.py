"""
Numeric unit tests for the velocity/smoothing helpers in charts.jsx.

Tests load charts.jsx in a node vm sandbox (stubbed React/window) and call the
helpers with JSON args, asserting numeric outputs.  They skip automatically when
node is absent from PATH (portability) but node v22 IS present in CI so RED→GREEN
is real.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest

CHARTS_PATH = str(
    pathlib.Path(__file__).parent.parent / 'src/dashboard/static/redux/charts.jsx'
)

# Node driver: loads charts.jsx in a vm sandbox, calls a named export as a
# function (when argsJson is provided) or reads it as a property (when absent).
_DRIVER = r"""
const vm = require('vm');
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
const sandbox = {
  React: { useRef: ()=>{}, useEffect: ()=>{}, useState: ()=>[], useMemo: ()=>{} },
  window: {},
  Math, console, Date,
};
vm.runInNewContext(src, sandbox);
const charts = sandbox.window.DF_CHARTS;
const name = process.argv[3];
const argsJson = process.argv[4];
const result = argsJson !== undefined
  ? charts[name](...JSON.parse(argsJson))
  : charts[name];
process.stdout.write(JSON.stringify(result) + '\n');
"""


def _node():
    path = shutil.which('node')
    if not path:
        pytest.skip('node not available')
    return path


def _eval_charts_fn(fn_name, *args):
    """Call window.DF_CHARTS[fn_name](*args) in a node vm and return decoded result."""
    result = subprocess.run(
        [_node(), '-e', _DRIVER, CHARTS_PATH, fn_name, json.dumps(list(args))],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip())


def _get_charts_const(const_name):
    """Read window.DF_CHARTS[const_name] in a node vm and return decoded value."""
    result = subprocess.run(
        [_node(), '-e', _DRIVER, CHARTS_PATH, const_name],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip())


# ---------------------------------------------------------------------------
# defaultSmoothingForWindow
# ---------------------------------------------------------------------------

class TestDefaultSmoothingForWindow:
    def test_24h(self):
        assert _eval_charts_fn('defaultSmoothingForWindow', '24h') == '2h'

    def test_7d(self):
        assert _eval_charts_fn('defaultSmoothingForWindow', '7d') == '8h'

    def test_30d(self):
        assert _eval_charts_fn('defaultSmoothingForWindow', '30d') == '1d'

    def test_90d(self):
        assert _eval_charts_fn('defaultSmoothingForWindow', '90d') == '3d'

    def test_unknown_falls_back_to_8h(self):
        assert _eval_charts_fn('defaultSmoothingForWindow', 'unknown') == '8h'


# ---------------------------------------------------------------------------
# smoothingLabelToSeconds
# ---------------------------------------------------------------------------

class TestSmoothingLabelToSeconds:
    def test_1h(self):
        assert _eval_charts_fn('smoothingLabelToSeconds', '1h') == 3600

    def test_2h(self):
        assert _eval_charts_fn('smoothingLabelToSeconds', '2h') == 7200

    def test_4h(self):
        assert _eval_charts_fn('smoothingLabelToSeconds', '4h') == 14400

    def test_8h(self):
        assert _eval_charts_fn('smoothingLabelToSeconds', '8h') == 28800

    def test_1d(self):
        assert _eval_charts_fn('smoothingLabelToSeconds', '1d') == 86400

    def test_3d(self):
        assert _eval_charts_fn('smoothingLabelToSeconds', '3d') == 259200


# ---------------------------------------------------------------------------
# SMOOTHING_OPTIONS constant
# ---------------------------------------------------------------------------

class TestSmoothingOptions:
    def test_non_empty_list(self):
        opts = _get_charts_const('SMOOTHING_OPTIONS')
        assert isinstance(opts, list)
        assert len(opts) > 0

    def test_contains_canonical_values(self):
        opts = _get_charts_const('SMOOTHING_OPTIONS')
        for v in ('2h', '8h', '1d', '3d'):
            assert v in opts


# ---------------------------------------------------------------------------
# deriveVelocitySeries — added in step-3 (these fail until step-4 GREEN)
# ---------------------------------------------------------------------------

class TestDeriveVelocitySeries:
    def test_empty_series_returns_empty(self):
        assert _eval_charts_fn('deriveVelocitySeries', [], [], 7200) == []

    def test_single_sample_returns_empty(self):
        assert _eval_charts_fn(
            'deriveVelocitySeries', [5], ['2026-01-01T00:00:00Z'], 7200
        ) == []

    def test_length_mismatch_returns_empty(self):
        assert _eval_charts_fn(
            'deriveVelocitySeries',
            [0, 1, 2],
            ['2026-01-01T00:00:00Z', '2026-01-01T01:00:00Z'],
            7200,
        ) == []

    def test_flat_series_all_zeros(self):
        labels = [
            '2026-01-01T00:00:00Z',
            '2026-01-01T01:00:00Z',
            '2026-01-01T02:00:00Z',
            '2026-01-01T03:00:00Z',
        ]
        out = _eval_charts_fn('deriveVelocitySeries', [5, 5, 5, 5], labels, 7200)
        assert out == [0.0, 0.0, 0.0, 0.0]

    def test_irregular_spacing_exact_value(self):
        # series=[0,1,5] at t=0,+1h,+2h, smoothing=7200s (2h)
        # At i=2: j=0 (t[2]-t[0]=7200<=7200), rate=(5-0)/(7200/86400)=60.0/day
        labels = [
            '2026-01-01T00:00:00Z',
            '2026-01-01T01:00:00Z',
            '2026-01-01T02:00:00Z',
        ]
        out = _eval_charts_fn('deriveVelocitySeries', [0, 1, 5], labels, 7200)
        assert abs(out[-1] - 60.0) < 1e-9

    def test_regular_10min_spacing_non_uniform_is_non_constant(self):
        # 145 samples at 10-min spacing; done rises in bursts → output must vary.
        import datetime
        base = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        labels = [
            (base + datetime.timedelta(minutes=10 * i)).strftime('%Y-%m-%dT%H:%M:%SZ')
            for i in range(145)
        ]
        # Done rises by 3 every 30 samples, 0 otherwise — non-uniform bursts.
        done = []
        v = 0
        for i in range(145):
            if i > 0 and i % 30 == 0:
                v += 3
            done.append(v)
        out = _eval_charts_fn('deriveVelocitySeries', done, labels, 7200)
        assert len({round(r, 6) for r in out}) > 1
