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
