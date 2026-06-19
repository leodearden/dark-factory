"""conftest.py for scripts/ tests — inserts scripts/ onto sys.path.

Mirrors the repo root conftest.py sys.path-insertion pattern so that
`import reviewer_redundancy_diagnostic` resolves when pytest collects
scripts/tests/ under importlib import mode.  No package __init__.py is
needed (importlib mode does not require it).
"""
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent.parent  # scripts/tests/../ = scripts/
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
