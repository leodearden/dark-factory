"""conftest.py for tests/scripts/ — inserts scripts/ and this directory onto sys.path.

Two insertions, both load-bearing for the same reason and neither redundant:

1. ``scripts/`` — mirrors scripts/tests/conftest.py so that
   `import reviewer_redundancy_diagnostic` (and the other module-under-test
   imports in this directory) resolves when pytest collects here.
2. ``this directory`` — so that shared, non-test helper modules living
   alongside the tests (e.g. `import systemd_unit_invariants`) resolve.

The second is not something pytest does for us: pyproject.toml sets
``addopts = "--import-mode=importlib ..."``, and under importlib mode pytest
deliberately does NOT insert a test file's own directory onto sys.path (that
sys.path mutation is a `prepend`/`append`-mode behaviour it exists to avoid).
So a sibling helper module is unimportable from these tests without this line,
and the failure surfaces at COLLECTION as a bare ModuleNotFoundError rather
than as anything resembling the invariant under test.
"""
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent.parent.parent / 'scripts'
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
