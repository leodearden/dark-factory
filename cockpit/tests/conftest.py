"""pytest configuration for the cockpit suite — tests-dir import path, git isolation and the Hypothesis settings profile.

This subproject has no git-invoking test today.  It is wired anyway: task 3355
exists to close a defect CLASS, not today's instances, and the anti-drift guard
(``tests/scripts/test_basetemp_git_isolation.py``) requires uniform coverage
precisely so the first git-touching test added here inherits the defence instead
of having to remember it.

``cockpit/tests/smoke/conftest.py`` needs no wiring of its own: pytest loads
conftests hierarchically, so the smoke suite inherits everything below.
"""
import sys
from pathlib import Path

from hypothesis import settings

# Sibling helpers (`_cockpit_*.py`) import by bare name; enforced by
# tests/scripts/test_pytest_workspace_collection.py::_SUBPROJECTS_WITH_CONFTEST.
_TESTS_DIR = Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# Suite-wide git isolation (task 3355, incident esc-3072-3).  The verify lane
# runs `cd cockpit && uv run pytest tests/`, which makes rootdir the SUBPROJECT —
# the repo-root conftest.py is never loaded, so each test-root conftest wires the
# defence itself.  APPEND the repo root, never insert(0, ...): at sys.path[0] it
# would make the subproject directories resolve as namespace packages pointing at
# the project folder instead of src/<pkg>/.
if str(REPO_ROOT := Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from df_pytest_isolation import (
    arm_leaked_drain_process_guard,
    arm_test_root_isolation,
    reject_unsafe_basetemp,
)


def pytest_configure(config):
    """Refuse a --basetemp aimed inside a live task worktree (esc-3072-3); arm the suite-wide defences."""
    reject_unsafe_basetemp(config)
    arm_test_root_isolation(config)
    arm_leaked_drain_process_guard(config)


# The stock 200ms deadline measures host load, not these pure-arithmetic properties (esc-5449-2).
settings.register_profile("cockpit", deadline=None)
settings.load_profile("cockpit")
