"""pytest configuration for the cockpit suite — suite-wide git isolation.

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

# Suite-wide git isolation (task 3355, incident esc-3072-3).  The verify lane
# runs `cd cockpit && uv run pytest tests/`, which makes rootdir the SUBPROJECT —
# the repo-root conftest.py is never loaded, so each test-root conftest wires the
# defence itself.  APPEND the repo root, never insert(0, ...): at sys.path[0] it
# would make the subproject directories resolve as namespace packages pointing at
# the project folder instead of src/<pkg>/.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from df_pytest_isolation import (  # noqa: E402
    _df_deploy_clocks_unwritten,  # noqa: F401  — the binding IS the wiring
    _df_git_ceiling_at_basetemp,  # noqa: F401  — the binding IS the wiring
    reject_unsafe_basetemp,
)


def pytest_configure(config):
    """Refuse a --basetemp aimed inside a live task worktree (esc-3072-3)."""
    reject_unsafe_basetemp(config)
