"""pytest configuration — ensure local src takes precedence over installed package."""
import sys
from pathlib import Path

import pytest

# Insert this worktree's src directory at the front of sys.path so that
# `import orchestrator` loads the local (possibly modified) code rather than
# whatever editable install the shared .venv has pinned to the main tree.
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture(autouse=True)
def _clear_orch_config_path(monkeypatch):
    """Remove ORCH_CONFIG_PATH so tests don't inadvertently load the real config."""
    monkeypatch.delenv("ORCH_CONFIG_PATH", raising=False)
