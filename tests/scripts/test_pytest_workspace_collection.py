"""Structural invariants ensuring root pytest can collect all subprojects.

These tests are file-content / filesystem checks (no subprocess, no pytest-in-pytest).
They guard the conditions that together prevent the tests.conftest namespace
collision under --import-mode=importlib:

1. Each subproject's conftest.py adds its own tests/ dir to sys.path.
2. No test file uses `from tests.<module> import ...` (namespace-relative imports).
3. No subproject has tests/__init__.py (which causes the collision).
4. Root pyproject.toml does not set norecursedirs (the workaround is no longer needed).
5. No test file uses `from conftest import ...`.  The bare `conftest` module name
   collides in `sys.modules` across subprojects — each subproject exports its
   non-fixture helpers under a uniquely-named sibling module instead.

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

# Regex that matches `from conftest import ...` (including indented, lazy imports).
_FROM_CONFTEST_RE = re.compile(r'^\s*from\s+conftest\s+import\b')

# Regex that matches `from .xxx import ...` and `from ..xxx import ...` —
# package-relative imports. These fail under --import-mode=importlib once
# tests/__init__.py is gone (no parent package).
_RELATIVE_IMPORT_RE = re.compile(r'^\s*from\s+\.+\w')

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


def test_no_from_conftest_imports_in_subproject_tests() -> None:
    """No test file may use `from conftest import ...`.

    The bare name `conftest` is shared across every subproject's conftest.py,
    and Python's `sys.modules` can only cache one module under that name at a
    time.  When root-level pytest loads multiple subprojects in the same
    process, whichever conftest loads first wins the cache slot — subsequent
    `from conftest import X` calls in other subprojects resolve to the wrong
    module and fail with ImportError at test-execution time.

    Each subproject instead exports its non-fixture helpers under a uniquely-
    named sibling module (``_fm_helpers.py``, ``_orch_helpers.py``,
    ``_dashboard_helpers.py``).  Test files import from those modules.

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
                if _FROM_CONFTEST_RE.match(line):
                    offenders.append(f'{rel}:{lineno}: {line.rstrip()}')

    assert not offenders, (
        f'Found {len(offenders)} `from conftest import` call(s) in test files.\n'
        + 'Rewrite to the subproject-specific helper module:\n'
        + '  fused-memory/tests/*  → `from _fm_helpers import X`\n'
        + '  orchestrator/tests/*  → `from _orch_helpers import X`\n'
        + '  dashboard/tests/*     → `from _dashboard_helpers import X`\n'
        + 'Offenders:\n'
        + '\n'.join(f'  {o}' for o in offenders)
    )


def test_no_package_relative_imports_in_subproject_tests() -> None:
    """No test file may use `from .<module> import ...` package-relative imports.

    Once tests/__init__.py is dropped (step-6 of this refactor), tests/ is no
    longer a Python package — relative imports raise
    ImportError: attempted relative import with no known parent package at
    collection time.  Use flat imports resolved via the conftest's
    sys.path.insert instead: `from test_workflow_e2e import X`.

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
                if _RELATIVE_IMPORT_RE.match(line):
                    offenders.append(f'{rel}:{lineno}: {line.rstrip()}')

    assert not offenders, (
        f'Found {len(offenders)} package-relative import(s) in test files.\n'
        + 'These fail under --import-mode=importlib once tests/__init__.py is gone.\n'
        + 'Rewrite `from .test_foo import X` → `from test_foo import X`.\n'
        + 'Offenders:\n'
        + '\n'.join(f'  {o}' for o in offenders)
    )


def test_no_subproject_tests_init_py() -> None:
    """No subproject tests/ directory may contain an __init__.py file.

    Under --import-mode=importlib, pytest assigns each conftest.py a unique
    importlib-generated module name *only when* tests/ is NOT a Python package.
    Any __init__.py in tests/ causes pytest to treat it as a package, forcing
    the `tests.conftest` module name — which collides across subprojects.

    The seven paths below are the ones that existed before this refactor.
    If any are present, pytest root-collection will fail with:
      ValueError: Plugin already registered: conftest (tests.conftest)
    """
    _INIT_PY_PATHS = [
        REPO_ROOT / 'dashboard' / 'tests' / '__init__.py',
        REPO_ROOT / 'escalation' / 'tests' / '__init__.py',
        REPO_ROOT / 'fused-memory' / 'tests' / '__init__.py',
        REPO_ROOT / 'orchestrator' / 'tests' / '__init__.py',
        REPO_ROOT / 'shared' / 'tests' / '__init__.py',
        REPO_ROOT / 'tests' / '__init__.py',
        REPO_ROOT / 'tests' / 'scripts' / '__init__.py',
    ]

    present = [str(p.relative_to(REPO_ROOT)) for p in _INIT_PY_PATHS if p.exists()]

    assert not present, (
        'The following tests/__init__.py files must be removed '
        '(they cause tests.conftest namespace collision under --import-mode=importlib):\n'
        + '\n'.join(f'  {p}' for p in present)
        + '\n\nDelete them with: git rm ' + ' '.join(present)
    )


def test_root_pyproject_omits_norecursedirs() -> None:
    """Root pyproject.toml must not set norecursedirs to exclude subprojects.

    The norecursedirs workaround was needed to prevent pytest from collecting
    fused-memory, orchestrator, and shared when those subprojects had
    tests/__init__.py files that caused a `tests.conftest` namespace collision.
    Now that the __init__.py files are removed, root pytest can collect all
    subprojects without the workaround.

    Asserts that `[tool.pytest.ini_options].norecursedirs` is either absent
    or set to an empty list.  A non-empty list would re-introduce the asymmetry
    where some subprojects are silently excluded from root-level pytest runs.
    """
    pyproject = REPO_ROOT / 'pyproject.toml'
    data = tomllib.loads(pyproject.read_text(encoding='utf-8'))

    ini_options = data.get('tool', {}).get('pytest', {}).get('ini_options', {})
    norecursedirs = ini_options.get('norecursedirs', [])

    assert not norecursedirs, (
        f'[tool.pytest.ini_options].norecursedirs in pyproject.toml is set to '
        f'{norecursedirs!r}.\n'
        'This workaround was needed when subprojects had tests/__init__.py files '
        'causing a tests.conftest namespace collision. Now that those files are '
        'removed, delete the norecursedirs line and its associated comment.'
    )
