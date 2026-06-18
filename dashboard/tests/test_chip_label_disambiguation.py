"""
Set-aware label disambiguation — algorithm tests only.

Uses a node vm sandbox to execute disambiguateLabels from scheduler_utils.jsx
and verify its behaviour directly.  Tests skip when node is absent.

The 6 React render-site wiring changes (tabs.jsx, tab_tasks.jsx,
scheduler_drawer.jsx, scheduler_heatmap.jsx, tab_scheduler.jsx,
tab_curator.jsx) are implemented in steps 4/6/8/10/12/14 but have no
corresponding test here: the dashboard has no JS/DOM test runner (no
package.json/node_modules, React+babel are CDN-only with in-browser
transpilation and no build step), so a behavioural render test is out of
scope.  The only non-trivial logic — the disambiguation algorithm — has full
RED→GREEN coverage via the node-vm sandbox below.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import pytest

SCHED_UTILS_PATH = str(
    pathlib.Path(__file__).parent.parent / 'src/dashboard/static/redux/scheduler_utils.jsx'
)

# Node driver: extracts the pure-JS helper section from scheduler_utils.jsx
# (everything before the window.DF_SCHED_UTILS export) and runs it in a vm
# sandbox.  Returns a Map result as a plain object via Object.fromEntries.
_DRIVER = r"""
const vm = require('vm');
const fs = require('fs');
const src = fs.readFileSync(process.argv[1], 'utf8');

// Extract everything before the window.DF_SCHED_UTILS export line.
const endIdx = src.lastIndexOf('\nwindow.DF_SCHED_UTILS');
if (endIdx < 0) throw new Error('window.DF_SCHED_UTILS not found — check scheduler_utils.jsx');
const helpersSrc = src.slice(0, endIdx);

const name    = process.argv[2];
const argsJson = process.argv[3];

const sandbox = { Math, console };
if (argsJson !== undefined) sandbox.__args = JSON.parse(argsJson);

const scriptBody = argsJson !== undefined
  ? 'var r = ' + name + '(...__args); if (r instanceof Map) r = Object.fromEntries(r); var __result = r;'
  : 'var r = ' + name + '; if (r instanceof Map) r = Object.fromEntries(r); var __result = r;';

vm.runInNewContext(helpersSrc + '\n' + scriptBody, sandbox);
process.stdout.write(JSON.stringify(sandbox.__result) + '\n');
"""


def _node():
    path = shutil.which('node')
    if not path:
        pytest.skip('node not available')
    return path


def _eval_sched_utils_fn(fn_name, *args):
    """Call fn_name(*args) in a node vm sandbox and return the decoded result."""
    result = subprocess.run(
        [_node(), '-e', _DRIVER, SCHED_UTILS_PATH, fn_name, json.dumps(list(args))],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip())


# ---------------------------------------------------------------------------
# Algorithm tests for disambiguateLabels (node-vm runtime)
# ---------------------------------------------------------------------------

class TestDisambiguateLabels:
    def test_single_path_basename_only(self):
        # (a) single path -> basename only
        result = _eval_sched_utils_fn('disambiguateLabels', ['a/b/c.rs'])
        assert result == {'a/b/c.rs': 'c.rs'}

    def test_distinct_basenames_no_growth(self):
        # (b) distinct basenames don't grow beyond 1 segment
        result = _eval_sched_utils_fn(
            'disambiguateLabels', ['pkg-a/src/lib.rs', 'pkg-b/src/main.rs']
        )
        assert result == {'pkg-a/src/lib.rs': 'lib.rs', 'pkg-b/src/main.rs': 'main.rs'}

    def test_collision_grows_leftward_by_one(self):
        # (c) same basename in two different parents -> grow by exactly one
        result = _eval_sched_utils_fn(
            'disambiguateLabels', ['a/x/lib.rs', 'b/y/lib.rs']
        )
        assert result == {'a/x/lib.rs': 'x/lib.rs', 'b/y/lib.rs': 'y/lib.rs'}

    def test_full_suffix_collision_falls_back_to_full_path(self):
        # (d) identical trailing 2 segments -> fall back to full path
        result = _eval_sched_utils_fn(
            'disambiguateLabels', ['pkg-a/src/lib.rs', 'pkg-b/src/lib.rs']
        )
        assert result == {
            'pkg-a/src/lib.rs': 'pkg-a/src/lib.rs',
            'pkg-b/src/lib.rs': 'pkg-b/src/lib.rs',
        }

    def test_suffix_is_substring_edge(self):
        # (e) one path is a suffix of the other (lib.rs vs a/lib.rs)
        result = _eval_sched_utils_fn('disambiguateLabels', ['lib.rs', 'a/lib.rs'])
        assert result == {'lib.rs': 'lib.rs', 'a/lib.rs': 'a/lib.rs'}

    def test_dedupe_identical_inputs(self):
        # (f) duplicate paths should NOT be forced to full length
        result = _eval_sched_utils_fn('disambiguateLabels', ['a/lib.rs', 'a/lib.rs'])
        assert result == {'a/lib.rs': 'lib.rs'}

    def test_empty_input(self):
        # (g) empty input -> empty object
        result = _eval_sched_utils_fn('disambiguateLabels', [])
        assert result == {}

    def test_motivating_reify_case(self):
        # (h) all distinct basenames -> each stays at 1 segment
        result = _eval_sched_utils_fn(
            'disambiguateLabels',
            [
                'reify-eval/src/persistent_cache.rs',
                'reify-solver-elastic/src',
                'reify-eval/tests',
            ],
        )
        assert result == {
            'reify-eval/src/persistent_cache.rs': 'persistent_cache.rs',
            'reify-solver-elastic/src': 'src',
            'reify-eval/tests': 'tests',
        }

    def test_property_pairwise_unique_and_minimal(self):
        # (i) adversarial set with three crates' src/lib.rs — property assertions
        paths = [
            'crate-a/src/lib.rs',
            'crate-b/src/lib.rs',
            'crate-c/src/lib.rs',
            'crate-a/src/main.rs',
            'crate-b/src/util.rs',
        ]
        result = _eval_sched_utils_fn('disambiguateLabels', paths)
        labels = list(result.values())
        keys = list(result.keys())

        # Pairwise distinct
        assert len(labels) == len(set(labels)), f'Labels not pairwise distinct: {labels}'

        # Each label is a trailing-segment suffix of its key
        for key, label in result.items():
            assert key == label or key.endswith('/' + label), (
                f"Label '{label}' is not a trailing-segment suffix of key '{key}'"
            )

        # Minimality: if the label has >1 segment, the one-shorter suffix must
        # appear as a trailing-segment suffix in at least one other key.
        def is_path_suffix(suffix, k):
            return k == suffix or k.endswith('/' + suffix)

        for key, label in result.items():
            segs = label.split('/')
            if len(segs) > 1:
                shorter = '/'.join(segs[1:])
                other_keys = [k for k in keys if k != key]
                assert any(is_path_suffix(shorter, k) for k in other_keys), (
                    f"Label '{label}' for key '{key}' is not minimal: "
                    f"shorter suffix '{shorter}' doesn't appear in any other key"
                )
