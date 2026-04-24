"""Structural invariants ensuring root pytest can collect all subprojects.

These tests are file-content / filesystem checks (no subprocess, no pytest-in-pytest).
They guard the four conditions that together prevent the tests.conftest namespace
collision under --import-mode=importlib:

1. Each subproject's conftest.py adds its own tests/ dir to sys.path.
2. No test file uses `from tests.<module> import ...` (namespace-relative imports).
3. No subproject has tests/__init__.py (which causes the collision).
4. Root pyproject.toml does not set norecursedirs (the workaround is no longer needed).

See also:
  - tests/scripts/test_dashboard_service_template.py  — file-content pattern reference
  - pyproject.toml [tool.pytest.ini_options]          — addopts = --import-mode=importlib
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]

# The four subprojects whose conftests must insert their tests dir.
_SUBPROJECTS_WITH_CONFTEST = ['dashboard', 'fused-memory', 'orchestrator', 'shared']

# Regex matching either supported sys.path.insert pattern for the tests dir:
#   fused-memory idiom : sys.path.insert(0, _tests_dir)
#   new idiom          : sys.path.insert(0, str(_TESTS_DIR))
_TESTS_DIR_INSERT_RE = re.compile(
    r'sys\.path\.insert\(0,\s*(?:str\(_TESTS_DIR\)|_tests_dir)\)'
)

# Regex that matches `from tests.<module> import ...` imports (including indented).
_FROM_TESTS_RE = re.compile(r'^\s*from\s+tests\.')

# All test directories to scan for namespace-relative imports and __init__.py files.
# Vendored sub-libraries (graphiti, mem0, taskmaster-ai) inside fused-memory are excluded.
_ALL_TEST_DIRS = [
    REPO_ROOT / 'dashboard' / 'tests',
    REPO_ROOT / 'escalation' / 'tests',
    REPO_ROOT / 'fused-memory' / 'tests',
    REPO_ROOT / 'orchestrator' / 'tests',
    REPO_ROOT / 'shared' / 'tests',
    REPO_ROOT / 'tests',
]

# Parts of a path that indicate vendored sub-library code — excluded from scans.
_VENDORED_PARTS = frozenset({'graphiti', 'mem0', 'taskmaster-ai'})


def test_subproject_conftests_add_tests_dir_to_sys_path() -> None:
    """Each subproject's tests/conftest.py must insert its tests/ dir into sys.path.

    Without tests/__init__.py (dropped in the importlib-mode refactor), pytest
    adds only the *parent* of a test file to sys.path — not the tests/ directory
    itself.  The sys.path.insert in each conftest ensures that `from conftest import X`
    and `from _dt_helpers import X` etc. continue to resolve correctly whether pytest
    is invoked from the subproject root or from the repo root.

    Accepted patterns (both are in use across the codebase):
      sys.path.insert(0, _tests_dir)       — fused-memory idiom (os.path form)
      sys.path.insert(0, str(_TESTS_DIR))  — new idiom (Path form)
    """
    missing: list[str] = []
    for sub in _SUBPROJECTS_WITH_CONFTEST:
        conftest = REPO_ROOT / sub / 'tests' / 'conftest.py'
        source = conftest.read_text(encoding='utf-8')
        if not _TESTS_DIR_INSERT_RE.search(source):
            missing.append(str(conftest.relative_to(REPO_ROOT)))

    assert not missing, (
        'The following conftests are missing a sys.path.insert(0, <tests_dir>) call:\n'
        + '\n'.join(f'  {m}' for m in missing)
        + '\n\nAdd a block like:\n'
        + '  _TESTS_DIR = Path(__file__).parent\n'
        + '  if str(_TESTS_DIR) not in sys.path:\n'
        + '      sys.path.insert(0, str(_TESTS_DIR))\n'
        + 'positioned AFTER the existing _SRC insert and BEFORE the first non-stdlib import.'
    )


def test_no_tests_namespace_imports_in_subproject_tests() -> None:
    """No test file may use `from tests.<module> import ...` imports.

    Under --import-mode=importlib without tests/__init__.py, each conftest's
    sys.path.insert makes helper modules directly importable as
    `from <module> import ...`.  The `from tests.<module>` form creates a hard
    dependency on tests/ being a Python package, which is the root cause of the
    tests.conftest namespace collision.

    Indented occurrences (lazy imports inside test methods) are also caught.
    Vendored sub-libraries (graphiti/, mem0/, taskmaster-ai/) inside fused-memory
    are excluded from the scan.

    Offender format in the assertion message: path:lineno: line
    """
    offenders: list[str] = []
    for test_dir in _ALL_TEST_DIRS:
        if not test_dir.exists():
            continue
        for py_file in sorted(test_dir.rglob('*.py')):
            rel = py_file.relative_to(REPO_ROOT)
            if _VENDORED_PARTS & set(rel.parts):
                continue
            source = py_file.read_text(encoding='utf-8')
            for lineno, line in enumerate(source.splitlines(), start=1):
                if _FROM_TESTS_RE.match(line):
                    offenders.append(f'{rel}:{lineno}: {line.rstrip()}')

    assert not offenders, (
        f'Found {len(offenders)} `from tests.*` import(s) in test files.\n'
        + 'Rewrite to namespace-less form:\n'
        + '  `from tests.conftest import X` → `from conftest import X`\n'
        + '  `from tests._dt_helpers import X` → `from _dt_helpers import X`\n'
        + '  `from tests.test_costs_data import X` → `from test_costs_data import X`\n'
        + 'Offenders:\n'
        + '\n'.join(f'  {o}' for o in offenders)
    )
