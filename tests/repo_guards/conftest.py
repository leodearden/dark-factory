"""conftest.py for tests/repo_guards/ — the sys.path insertions its guards import through.

The root pyproject runs pytest with ``--import-mode=importlib``, which keeps a
test file's own directory off sys.path, so each directory below is inserted
here or its imports fail at collection:

1. this directory — the guards' sibling helper modules;
2. ``scripts/`` — ``from legibility import invariants`` in
   test_design_invariants_consistency.py, and ``import governed_list_agreement``
   in test_repo_guards_governed_lists.py;
3. ``shared/tests/cross_suite_helpers/`` — helpers that stay beside shared's own
   unit tests, such as ``capability_manifest_corpus``. The dependency runs one
   way only: nothing in shared/tests imports from here.

Inserted in reverse priority, so this directory wins a basename collision.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

for _path in (
    REPO_ROOT / 'shared' / 'tests' / 'cross_suite_helpers',
    REPO_ROOT / 'scripts',
    Path(__file__).resolve().parent,
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
