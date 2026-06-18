"""
Set-aware label disambiguation tests for the lock/path chips.

Two test tiers:
 1. Algorithm tests — run disambiguateLabels in a node vm sandbox.
    Skip when node is absent.
 2. Source-structure tests — fetch JSX files via Starlette TestClient and
    assert structural contracts (naive shortener removed, helper routed,
    title attribute preserved).  Follow the idiom from test_tab_scheduler.py.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess

import pytest
from starlette.testclient import TestClient


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
# Shared Starlette client fixture (module-scoped for speed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Step-1: Algorithm RED tests for disambiguateLabels
# (RED because disambiguateLabels does not exist yet in scheduler_utils.jsx)
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


# ---------------------------------------------------------------------------
# Step-3: Source-structure tests for tabs.jsx (LockChip wiring)
# (RED because tabs.jsx still defines shortPath and never calls
# disambiguateLabels)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def tabs_jsx_body(_client):
    return _client.get('/static/redux/tabs.jsx').text


class TestTabsJsxLockChipWiring:
    def test_shortpath_function_removed(self, tabs_jsx_body):
        # The duplicate local shortener must be gone
        assert 'function shortPath' not in tabs_jsx_body, (
            'shortPath function still present in tabs.jsx — should be deleted'
        )

    def test_shortpath_call_removed(self, tabs_jsx_body):
        # No call site remains either
        assert 'shortPath(' not in tabs_jsx_body, (
            'shortPath( call still present in tabs.jsx — should be removed'
        )

    def test_disambiguate_labels_routed(self, tabs_jsx_body):
        # LocksCell must route the lockSet through the shared helper
        assert 'disambiguateLabels(' in tabs_jsx_body, (
            'disambiguateLabels( not found in tabs.jsx — wiring missing'
        )

    def test_full_key_title_preserved(self, tabs_jsx_body):
        # LockChip must still render a title containing the path
        assert re.search(r'title=\{[^}]*path', tabs_jsx_body), (
            'LockChip title={...path...} hover attribute not found in tabs.jsx'
        )


# ---------------------------------------------------------------------------
# Step-5: Source-structure tests for tab_tasks.jsx (Module-locks panel)
# (RED because tab_tasks.jsx still renders modPath.split('/').pop())
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def tab_tasks_jsx_body(_client):
    return _client.get('/static/redux/tab_tasks.jsx').text


class TestTabTasksJsxModuleLocksWiring:
    def test_modpath_basename_removed(self, tab_tasks_jsx_body):
        # The naive basename for modPath must be gone.
        # Guard: we target modPath.split only, not the task-id split('/T-').pop().
        assert "modPath.split('/').pop()" not in tab_tasks_jsx_body, (
            "modPath.split('/').pop() still present in tab_tasks.jsx — should be replaced"
        )

    def test_disambiguate_labels_routed(self, tab_tasks_jsx_body):
        assert 'disambiguateLabels(' in tab_tasks_jsx_body, (
            'disambiguateLabels( not found in tab_tasks.jsx — wiring missing'
        )

    def test_full_key_title_preserved(self, tab_tasks_jsx_body):
        # The chip must still have title={modPath}
        assert 'title={modPath}' in tab_tasks_jsx_body, (
            'title={modPath} not found in tab_tasks.jsx — full key hover removed'
        )


# ---------------------------------------------------------------------------
# Step-7: Source-structure tests for scheduler_drawer.jsx (HolderEtas)
# (RED because scheduler_drawer.jsx still renders m.path.split('/').pop())
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def scheduler_drawer_jsx_body(_client):
    return _client.get('/static/redux/scheduler_drawer.jsx').text


class TestSchedulerDrawerJsxHolderEtasWiring:
    def test_mpath_basename_removed(self, scheduler_drawer_jsx_body):
        assert "m.path.split('/').pop()" not in scheduler_drawer_jsx_body, (
            "m.path.split('/').pop() still present in scheduler_drawer.jsx — should be replaced"
        )

    def test_disambiguate_labels_destructured(self, scheduler_drawer_jsx_body):
        # disambiguateLabels must be added to the existing top-level destructure
        # of window.DF_SCHED_UTILS
        assert 'disambiguateLabels' in scheduler_drawer_jsx_body, (
            'disambiguateLabels not found in scheduler_drawer.jsx — wiring missing'
        )

    def test_full_key_title_preserved(self, scheduler_drawer_jsx_body):
        # The holder row must still have title={m.path}
        assert 'title={m.path}' in scheduler_drawer_jsx_body, (
            'title={m.path} not found in scheduler_drawer.jsx — full key hover removed'
        )


# ---------------------------------------------------------------------------
# Step-9: Source-structure tests for scheduler_heatmap.jsx (column headers)
# (RED because scheduler_heatmap.jsx still renders m.path.split('/').pop()
# for the sched-col-path span — the primary multi-collision site)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def scheduler_heatmap_jsx_body(_client):
    return _client.get('/static/redux/scheduler_heatmap.jsx').text


class TestSchedulerHeatmapJsxColumnHeaderWiring:
    def test_mpath_basename_removed(self, scheduler_heatmap_jsx_body):
        assert "m.path.split('/').pop()" not in scheduler_heatmap_jsx_body, (
            "m.path.split('/').pop() still present in scheduler_heatmap.jsx — should be replaced"
        )

    def test_disambiguate_labels_routed(self, scheduler_heatmap_jsx_body):
        assert 'disambiguateLabels(' in scheduler_heatmap_jsx_body, (
            'disambiguateLabels( not found in scheduler_heatmap.jsx — wiring missing'
        )

    def test_column_header_title_preserved(self, scheduler_heatmap_jsx_body):
        # The <th> must still have a title attribute containing m.path
        assert re.search(r'title=\{[^}]*m\.path', scheduler_heatmap_jsx_body), (
            'Column header title={...m.path...} not found in scheduler_heatmap.jsx'
        )


# ---------------------------------------------------------------------------
# Step-11: Source-structure tests for tab_scheduler.jsx (Modules view)
# (RED because tab_scheduler.jsx still renders m.path.split('/').pop()
# as the primary title line in the module card)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def tab_scheduler_jsx_body(_client):
    return _client.get('/static/redux/tab_scheduler.jsx').text


class TestTabSchedulerJsxModulesViewWiring:
    def test_mpath_basename_removed(self, tab_scheduler_jsx_body):
        assert "m.path.split('/').pop()" not in tab_scheduler_jsx_body, (
            "m.path.split('/').pop() still present in tab_scheduler.jsx — should be replaced"
        )

    def test_disambiguate_labels_destructured(self, tab_scheduler_jsx_body):
        # disambiguateLabels must be added to the existing top-level destructure
        assert 'disambiguateLabels' in tab_scheduler_jsx_body, (
            'disambiguateLabels not found in tab_scheduler.jsx — wiring missing'
        )

    def test_full_path_text_preserved(self, tab_scheduler_jsx_body):
        # The secondary full-path text line must still show m.path
        assert '{m.path}' in tab_scheduler_jsx_body, (
            '{m.path} not found in tab_scheduler.jsx — secondary full-path text removed'
        )


# ---------------------------------------------------------------------------
# Step-13: Source-structure tests for tab_curator.jsx (ticket-files ChipList)
# (RED because tab_curator.jsx still renders f.split('/').pop())
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def tab_curator_jsx_body(_client):
    return _client.get('/static/redux/tab_curator.jsx').text


class TestTabCuratorJsxFileChipWiring:
    def test_f_basename_removed(self, tab_curator_jsx_body):
        assert "f.split('/').pop()" not in tab_curator_jsx_body, (
            "f.split('/').pop() still present in tab_curator.jsx — should be replaced"
        )

    def test_disambiguate_labels_routed(self, tab_curator_jsx_body):
        assert 'disambiguateLabels(' in tab_curator_jsx_body, (
            'disambiguateLabels( not found in tab_curator.jsx — wiring missing'
        )

    def test_full_key_title_preserved(self, tab_curator_jsx_body):
        # The file chip must still have title={f}
        assert 'title={f}' in tab_curator_jsx_body, (
            'title={f} not found in tab_curator.jsx — full key hover removed'
        )


# ---------------------------------------------------------------------------
# Step-15: Cache-buster check — all ?v= values in index.html must be >= 24
# (RED because all assets are currently ?v=23)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def index_html_body(_client):
    return _client.get('/static/redux/index.html').text


class TestIndexHtmlCacheBuster:
    def test_at_least_one_versioned_asset(self, index_html_body):
        # Confirm there are versioned assets to check
        assert re.search(r'\?v=\d+', index_html_body), (
            'No ?v=<n> versioned assets found in index.html'
        )

    def test_all_version_numbers_at_least_24(self, index_html_body):
        # Every ?v=<n> cache-buster must be >= 24 (7 JSX files changed)
        versions = [int(m.group(1)) for m in re.finditer(r'\?v=(\d+)', index_html_body)]
        assert versions, 'No versioned assets found'
        for v in versions:
            assert v >= 24, (
                f'Asset with ?v={v} found — all cache-busters must be bumped to >= 24'
            )
