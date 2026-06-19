"""conftest.py for tests/scripts/ — inserts scripts/ onto sys.path.

Mirrors scripts/tests/conftest.py so that `import reviewer_redundancy_diagnostic`
resolves when pytest collects this directory.
"""
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / 'scripts'
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
