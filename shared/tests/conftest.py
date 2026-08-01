"""pytest configuration — ensure local src takes precedence over installed package."""
import sys
from pathlib import Path

# Insert this worktree's src directory at the front of sys.path so that
# `import shared` loads the local (possibly modified) code rather than
# whatever editable install the shared .venv has pinned to the main tree.
_SRC = Path(__file__).parent.parent / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_TESTS_DIR = Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# Suite-wide git isolation (task 3355, incident esc-3072-3).  The verify lane
# runs `cd shared && uv run pytest tests/`, which makes rootdir the SUBPROJECT —
# the repo-root conftest.py is never loaded, so each test-root conftest wires
# the defence itself.  APPEND the repo root, never insert(0, ...): at sys.path[0]
# it would make the subproject directories resolve as namespace packages
# pointing at the project folder instead of src/<pkg>/, beating the insert above.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from df_pytest_isolation import (  # noqa: E402
    _df_git_ceiling_at_basetemp,  # noqa: F401  — the binding IS the wiring
    reject_unsafe_basetemp,
)


def pytest_configure(config):
    """Refuse a --basetemp aimed inside a live task worktree (esc-3072-3)."""
    reject_unsafe_basetemp(config)
