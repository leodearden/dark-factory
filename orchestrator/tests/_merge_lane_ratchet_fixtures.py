"""What the merge-lane ratchet suites share: the instrument and the one seed report.

The ``scripts/`` modules are imported HERE, once, so no suite carries its own
``sys.path`` bootstrap; orchestrator/pyproject.toml's [tool.pyright] extraPaths
already carries "../scripts", so this needs no type-checker config.

``synthetic_report`` is the ONE seed every suite measures against. A second
declaration would drift, and the suites would then disagree about what a raise
looks like. A helper lives here only when more than one suite uses it;
``_orch_helpers`` is the home for what the whole test tree shares.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).parents[2] / 'scripts'
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import merge_lane_metrics as metrics  # type: ignore[import-not-found]  # noqa: E402
import source_measures  # type: ignore[import-not-found]  # noqa: E402

__all__ = ['blank_file_entry', 'metrics', 'source_measures', 'synthetic_report']


def synthetic_report() -> dict:
    """Two cluster files, three functions, two test files -- a COMPLETE report.

    Every number is small and hand-chosen so a perturbation reads at a
    glance, and the cluster totals the ratchet compares are DERIVED from
    these entries by ``derive_totals`` rather than stated a second time.
    """
    return {
        'schema_version': 1,
        'params': {
            'complexipy_version': '6.2.0',
            'cluster_paths': ['a.py', 'b.py'],
            'file_line_ceiling': 1500,
            'new_function_cognitive_ceiling': 15,
        },
        'enumeration': {
            'requested': ['a.py', 'b.py'],
            'resolved': ['a.py', 'b.py'],
            # The LIVE shape: the test-tree half is two counts, and it is the one
            # key render_baseline drops on the way to the committed file.
            'test_tree': {'requested': 9, 'resolved': 2},
            'unreadable': [],
            'complete': True,
        },
        'files': {
            'a.py': {
                'lines': 1000,
                'prose_lines': 400,
                'cognitive': 120,
                'function_local_imports': 3,
                'reexport_names': 5,
            },
            'b.py': {
                'lines': 200,
                'prose_lines': 50,
                'cognitive': 30,
                'function_local_imports': 1,
                'reexport_names': 0,
            },
        },
        'functions': {'a.py::f': 40, 'a.py::C::m': 12, 'b.py::g': 7},
        'tests': {
            't1.py': {'patch_targets': ['foo', 'bar'], 'private_reads': 20},
            't2.py': {'patch_targets': ['bar', 'baz'], 'private_reads': 5},
        },
    }


def blank_file_entry(**overrides: int) -> dict:
    """A cluster-file entry with every measure zero, then *overrides* applied."""
    entry = {
        'lines': 0,
        'prose_lines': 0,
        'cognitive': 0,
        'function_local_imports': 0,
        'reexport_names': 0,
    }
    entry.update(overrides)
    return entry
