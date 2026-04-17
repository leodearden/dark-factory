"""pytest configuration — ensure local src takes precedence over installed package."""
import sys
from pathlib import Path

# Insert this worktree's src directory at the front of sys.path so that
# `import shared` loads the local (possibly modified) code rather than
# whatever editable install the shared .venv has pinned to the main tree.
_SRC = Path(__file__).parent.parent / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Cooperative jobserver: block at session start until a slot is free on the
# system-wide FIFO (pytest-jobserver.service).  No-op when PYTEST_JOBSERVER_FIFO
# is unset or the FIFO is absent.
pytest_plugins = ('shared.pytest_jobserver',)
