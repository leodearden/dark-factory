"""The seed report BOTH merge-lane ratchet suites measure against.

ONE synthetic report, never two. ``test_merge_lane_ratchet.py`` pins what the
instrument's pure comparators do with it, and
``test_merge_lane_ratchet_commit_gate.py`` commits it into throwaway repos to
pin what the commit-time auditor does with the resulting blobs. A second
declaration would drift, and the two suites would then disagree about what a
raise looks like.

It lives HERE rather than in either suite because a test module reaching into a
sibling test module for a private name is the shape this repo names an
interface smell -- pointedly, one this very instrument MEASURES. It also tied
collection of the gate suite to importing a 3700-line sibling. ``_orch_helpers``
is the home for helpers the WHOLE suite shares; this is the narrower one for a
single subject, matching the ``_*_fixtures.py`` modules already beside it.

Only what both suites use lives here. ``_blank_file_entry`` stays in
test_merge_lane_ratchet.py: it has one consumer, so moving it would widen a
scope rather than single-source a shared one.
"""
from __future__ import annotations


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
