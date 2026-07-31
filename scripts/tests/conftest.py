"""conftest.py for scripts/ tests — inserts scripts/ onto sys.path.

Mirrors the repo root conftest.py sys.path-insertion pattern so that
`import reviewer_redundancy_diagnostic` resolves when pytest collects
scripts/tests/ under importlib import mode.  No package __init__.py is
needed (importlib mode does not require it).

Also inserts scripts/legibility/ so flat modules nested one level down
(e.g. `digest.py`, and its PRD-decomposition siblings — the sampler,
merger, trickle coder, etc. all planned to live under scripts/legibility/)
resolve via a bare `import digest` the same way top-level scripts/*.py
modules do. Without this, scripts/ on sys.path alone only makes
scripts/legibility/ importable as a namespace package (`import legibility`),
not its contents as bare top-level names.
"""
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent.parent  # scripts/tests/../ = scripts/
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_LEGIBILITY_DIR = _SCRIPTS_DIR / 'legibility'
if str(_LEGIBILITY_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGIBILITY_DIR))

# Suite-wide git isolation (task 3355, incident esc-3072-3).  A run rooted at
# this directory does not load the repo-root conftest.py, so each test-root
# conftest wires the defence itself.  APPEND the repo root, never
# insert(0, ...): at sys.path[0] it would make the subproject directories
# resolve as namespace packages pointing at the project folder instead of
# src/<pkg>/, defeating the inserts above.
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
