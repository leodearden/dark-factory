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
#: seconds each one carries.  Four entries are IMPORTED from their one home:
#: ``WHOLE_TREE_SCAN_TEST_TIMEOUT`` from ``shared.whole_tree_guard_family``,
#: and the three ``_orch_helpers`` constants imported above.  Two are LITERAL
#: MIRRORS, because importing their home costs more than mirroring it:
#: ``HEAVY_BARRIER_TEST_TIMEOUT = 540`` lives in the support module
#: orchestrator/tests/_merge_worker_drive.py, which imports the production
#: merge lane at module scope, and ``PYTEST_TIMEOUT = 960`` is FILE-LOCAL to
#: the test module test_warm_lane_bash_suite.py, which asserts at module scope
#: (test_pytest_marker_deselection.py's identical line is inside a string
#: FIXTURE, so it is not a binding, which TestSanctionedNameMirrors measures
#: rather than assumes).
#:
#: A literal mirror is NOT self-announcing when it goes stale.  One that reads
#: too HIGH fails SILENTLY, which is the dangerous direction: lower
#: ``HEAVY_BARRIER_TEST_TIMEOUT`` by hand to 225 and every
#: ``timeout(HEAVY_BARRIER_TEST_TIMEOUT)`` site really pins 225s -- squarely
#: inside the band -- while this map still answers 540 and the sweep stays
#: green.  Matching on the TRAILING name only widens that hole for EVERY
#: entry, imported or mirrored: a new file-local ``PYTEST_TIMEOUT = 120``
#: would be waved through at 960.
#:
#: So every entry carries the same EXECUTABLE link
#: test_timeout_marker_inversion_guard.py::TestVerifyCliBudgetConstant gives
#: the YAML -- TestSanctionedNameMirrors re-reads every binding of each name
#: under orchestrator/tests and fails if one disagrees.  This map is a cache
#: of those definitions, never a claim about them.
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
