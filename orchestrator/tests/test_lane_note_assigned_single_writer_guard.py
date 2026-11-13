"""Guard: the WarmLanePool is the sole caller of ``LaneLifecycle.note_assigned``.

The tree-wide half of invariant I2 (single-writer consolidation, W11), split
out of test_lane_lifecycle_gitops.py so its whole-tree sweep no longer widens
that module's unrelated tests to the guard-family ceiling. The named-file half,
git_ops.py writing no ASSIGNED/RELEASED lane state, stays there.

The sweep is a substring test that never parses, so it is cheap; it is in the
family because it reads the whole source tree.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from _tree_scan import ORCHESTRATOR_SRC, TREE_PASS_XDIST_GROUP, ParsedModule, TreeAnalysis, TreePass
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

# The family mark and shared group: shared/src/shared/whole_tree_guard_family.py::WHOLE_TREE_SCAN_TEST_TIMEOUT, _tree_scan.py.
pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP),
]


def _calls_note_assigned(module: ParsedModule) -> bool:
    return '.note_assigned(' in module.source


ANALYSIS = TreeAnalysis(name=Path(__file__).stem, files=ORCHESTRATOR_SRC, extract=_calls_note_assigned)
TREE_ANALYSES = (ANALYSIS,)


class TestSingleWriterSourceScan:
    """The ONLY orchestrator/src/orchestrator file containing a ``.note_assigned(``
    CALL is warm_lane_pool.py.

    lane_lifecycle.py DEFINES ``def note_assigned`` and its body is
    ``self.transition(lane, LaneState.ASSIGNED, ...)``; neither is a
    ``.note_assigned(`` call, so the primitive is naturally excluded.

    NOT asserted: "no ASSIGNED/RELEASED transition anywhere in the tree".
    harness.py's crash-recovery terminal-release finalizer
    (``_lane_lifecycle.transition(entry, LaneState.RELEASED)``) is a legitimate
    NON-pool lifecycle write the task does not consolidate, and the
    lane_lifecycle.py primitive body is the writer of record.
    """

    def test_note_assigned_called_only_from_warm_lane_pool(self, tree_pass: TreePass):
        callers = sorted(
            py.name for py, calls in tree_pass.facts(ANALYSIS).items() if calls
        )
        assert callers == ['warm_lane_pool.py'], (
            'LaneLifecycle.note_assigned() must be called ONLY from '
            'warm_lane_pool.py (the sole durable ASSIGNED writer, I2); '
            f'found callers in {callers}'
        )
