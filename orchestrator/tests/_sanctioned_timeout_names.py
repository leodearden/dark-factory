"""The timeout constant NAMES a ``pytest.mark.timeout(NAME)`` site may be spelled through.

:data:`SANCTIONED_TIMEOUT_NAMES` maps each name to the seconds it carries, so a
marker spelled as a name resolves to a number without importing the module that
defines it.  Two consumers read it: the tree-wide inversion sweep in
test_timeout_marker_inversion_guard.py, and the scene-marker pins in
test_spawn_budget_guard.py.  It is a cache of the real definitions, which
test_timeout_marker_inversion_guard.py::TestSanctionedNameMirrors re-reads out of
the tree and checks it against.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from _orch_helpers import (
    DEEP_GATE_SCENE_TEST_TIMEOUT,
    DEEP_LANDING_SCENE_TEST_TIMEOUT,
    VERIFY_CLI_PER_TEST_TIMEOUT,
)
from shared.whole_tree_guard_family import (
    WHOLE_TREE_SCAN_TEST_TIMEOUT,
    WHOLE_TREE_SCAN_TIMEOUT_NAME,
)

#: Names a ``pytest.mark.timeout(...)`` argument may resolve to, and the
#: seconds each one carries.  A literal MAP rather than an import, because none
#: of the three is importable from ``_orch_helpers``: two are defined
#: outside it -- ``HEAVY_BARRIER_TEST_TIMEOUT = 540`` in the shared support
#: module orchestrator/tests/_merge_worker_drive.py, and, FILE-LOCALLY,
#: ``PYTEST_TIMEOUT = 960``
#: (test_warm_lane_bash_suite.py -- test_pytest_marker_deselection.py's
#: identical line is inside a string FIXTURE, so it is not a binding, which
#: TestSanctionedNameMirrors measures rather than assumes).
#:
#: These are cross-module MIRRORS, and a stale one is NOT self-announcing.  A
#: mirror that reads too HIGH fails SILENTLY, which is the dangerous
#: direction: lower ``HEAVY_BARRIER_TEST_TIMEOUT`` by hand to 225 and every
#: ``timeout(HEAVY_BARRIER_TEST_TIMEOUT)`` site really pins 225s -- squarely
#: inside the band -- while this map still answers 540 and the sweep stays
#: green.  Matching on the TRAILING name only widens that hole: a new
#: file-local ``PYTEST_TIMEOUT = 120`` would be waved through at 960.
#:
#: So every entry carries the same EXECUTABLE link
#: test_timeout_marker_inversion_guard.py::TestVerifyCliBudgetConstant gives
#: the YAML -- TestSanctionedNameMirrors re-reads each name's REAL
#: definition out of the tree and fails if the two disagree.  This map is a
#: cache of those definitions, never a claim about them.
SANCTIONED_TIMEOUT_NAMES: Mapping[str, float] = MappingProxyType({
    WHOLE_TREE_SCAN_TIMEOUT_NAME: float(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    'HEAVY_BARRIER_TEST_TIMEOUT': 540.0,
    'PYTEST_TIMEOUT': 960.0,
    'VERIFY_CLI_PER_TEST_TIMEOUT': float(VERIFY_CLI_PER_TEST_TIMEOUT),
    'DEEP_GATE_SCENE_TEST_TIMEOUT': float(DEEP_GATE_SCENE_TEST_TIMEOUT),
    'DEEP_LANDING_SCENE_TEST_TIMEOUT': float(DEEP_LANDING_SCENE_TEST_TIMEOUT),
})

#: The sanctioned names defined OUTSIDE orchestrator/tests.  Each such entry
#: above is imported from its one home, so it cannot drift from it, and no binding
#: under the tests directory exists for TestSanctionedNameMirrors to find.
SANCTIONED_NAMES_DEFINED_ELSEWHERE = frozenset({WHOLE_TREE_SCAN_TIMEOUT_NAME})
