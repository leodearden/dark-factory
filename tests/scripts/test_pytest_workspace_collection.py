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
import shlex
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


# ---------------------------------------------------------------------------
# Root-vs-member marker deselection mirroring (task 3444)
# ---------------------------------------------------------------------------

# Matches a marker name used as a `not <name>` term inside a pytest -m
# expression: "not smoke and not integration" -> ['smoke', 'integration'].
# Only ever applied to an expression already accepted by _FLAT_DESELECT_RE.
_NOT_MARKER_RE = re.compile(r'\bnot\s+([A-Za-z_]\w*)')

# The ONLY -m expression shape this guard claims to understand: a pure flat
# conjunction of deselects -- `not X`, `not X and not Y`, ...
#
# Deliberately narrow. A pytest -m expression is a full boolean expression, and
# regexing `not <name>` out of a richer one silently gives the WRONG answer in
# both directions, which is the one outcome a vacuity-sensitive guard must not
# have:
#   * `-m 'not (smoke or integration)'` in a MEMBER yields no `not <name>`
#     matches at all -- the member's deselect becomes invisible to `owners` and
#     this guard passes on the exact config it exists to catch;
#   * the same expression in the ROOT empties `root_deselected` and fails the
#     guard on a config that is semantically correct;
#   * a selecting expression like `smoke and not slow` would be misread as a
#     deselect of `slow`.
# So anything but the flat form is REJECTED LOUDLY at parse time instead.
_FLAT_DESELECT_RE = re.compile(r'^not\s+[A-Za-z_]\w*(?:\s+and\s+not\s+[A-Za-z_]\w*)*$')


def _pytest_ini_options(pyproject: Path) -> dict:
    """Return [tool.pytest.ini_options] for *pyproject*, or {} if absent."""
    if not pyproject.exists():
        return {}
    data = tomllib.loads(pyproject.read_text(encoding='utf-8'))
    return data.get('tool', {}).get('pytest', {}).get('ini_options', {})


def _effective_marker_expression(addopts: str) -> str:
    """Return the -m expression pytest would actually apply for *addopts*.

    `-m` is a single argparse option, so a LATER `-m` silently REPLACES an
    earlier one — this returns the last occurrence, matching pytest's real
    behaviour (and so catching a two-`-m`-flags config that drops a deselect).
    shlex.split handles the quoted form actually in use, e.g.
    `-n auto --dist loadgroup -m 'not integration'`.
    """
    expression = ''
    tokens = shlex.split(addopts)
    for i, token in enumerate(tokens):
        if token.startswith('--'):
            continue
        if token == '-m':
            if i + 1 < len(tokens):
                expression = tokens[i + 1]
        elif token.startswith('-m='):
            expression = token[3:]
        elif token.startswith('-m'):
            expression = token[2:]
    return expression


def _deselected_markers(addopts: str, source: str) -> list[str]:
    """Marker names deselected by *addopts*, in the order they appear.

    *source* names the pyproject.toml the addopts came from; it appears in the
    failure message raised when the -m expression is not the flat deselect form
    (see _FLAT_DESELECT_RE for why anything else is rejected rather than
    approximated).
    """
    expression = _effective_marker_expression(addopts)
    if not expression:
        return []

    assert _FLAT_DESELECT_RE.match(expression), (
        f'the pytest -m expression in {source} is {expression!r}, which this guard '
        f'cannot parse.\n\n'
        f'Use the flat deselect form — "not X", or "not X and not Y and ..." — so '
        f'the root-vs-member mirroring invariant stays mechanically checkable.\n\n'
        f'Anything richer is rejected ON PURPOSE, not for style: this guard finds '
        f'deselects by matching `not <name>`, and on a parenthesised expression '
        f'(e.g. "not (smoke or integration)") that match returns NOTHING — which '
        f'would hide a member\'s deselect from the guard entirely (a silent pass on '
        f'the very config it exists to catch) or empty the root\'s deselect set (a '
        f'failure on a correct config). A selecting expression such as '
        f'"smoke and not slow" would likewise be misread as deselecting `slow`.\n\n'
        f'If a richer expression is genuinely needed, replace this shape check with '
        f'a real boolean parse of the -m expression. Do NOT loosen the regex.'
    )
    return _NOT_MARKER_RE.findall(expression)


def _registered_marker_names(ini_options: dict) -> set[str]:
    """Marker NAMES from a `markers` list whose entries are "<name>: <description>"."""
    return {str(entry).split(':', 1)[0].strip() for entry in ini_options.get('markers', [])}


def _mirroring_failure_message(
    unregistered: list[str],
    undeselected: list[str],
    owners: dict[str, list[str]],
    root_deselected: list[str],
    root_addopts: str,
) -> str:
    """Render the assertion message for a root config that ignores a member deselect."""
    # Keep the root's existing order and append what is missing, so the
    # suggested expression is the literal edit to make.
    expected = root_deselected + [m for m in owners if m not in root_deselected]
    sections = [
        'The ROOT pyproject.toml does not honour marker deselections declared by '
        'workspace members.',
        '',
        "pytest reads exactly ONE [tool.pytest.ini_options] — the rootdir's inifile — and "
        "never merges addopts/markers across pyproject.toml files. A member's `-m 'not X'` "
        'is therefore SILENTLY IGNORED whenever rootdir resolves to the repo root (an '
        'explicit `-c pyproject.toml`, a bare `pytest` from the repo root, or any arg set '
        'spanning two subprojects) — and those tests RUN.',
        '',
    ]
    if unregistered:
        sections.append('NOT REGISTERED in the root `markers` list:')
        sections += [f'  {m}  (deselected by: {", ".join(owners[m])})' for m in unregistered]
        sections.append('')
    if undeselected:
        sections.append('NOT DESELECTED by the root `addopts` -m expression:')
        sections += [f'  {m}  (deselected by: {", ".join(owners[m])})' for m in undeselected]
        sections.append('')
    sections += [
        'Required edit to the ROOT pyproject.toml [tool.pytest.ini_options]:',
        '  1. register each missing marker in `markers`:',
        '       "<name>: <description>; deselect: -m \'not <name>\'",',
        '  2. use ONE combined -m expression in `addopts`:',
        f"       -m '{' and '.join(f'not {m}' for m in expected)}'",
        '     The single -m flag is load-bearing: -m is one argparse option, so a second',
        '     -m would silently REPLACE the first and drop the earlier deselect.',
        '',
        f'root addopts today: {root_addopts!r}',
    ]
    return '\n'.join(sections)


def test_root_pyproject_mirrors_member_marker_deselections() -> None:
    """Every marker a workspace member DESELECTS must also be gated at the root.

    If a member decided a marker's tests must not run by default, a root-bound
    run has to honour that too — pytest will never merge the member's decision
    up.  Otherwise those tests run on any invocation whose rootdir resolves to
    the repo root, which is how 9 live-CLI/OAuth tests in shared/tests/ came to
    run (and go red) on ordinary runs before task 3444.

    Keyed on what a member DESELECTS, never on what it merely REGISTERS.
    orchestrator/pyproject.toml registers six markers and deliberately
    deselects none — its `warm_lane_bash` entry is documented there as
    "deliberately NOT deselected in addopts, because ... needs this coverage
    actually running on its verify leg".  A blanket "root must mirror every
    member marker" rule would demand deselecting it at root and silently drop
    that coverage.  Keying on the member's own explicit `-m 'not X'` keeps the
    invariant true by construction and the guard non-brittle.

    Every `-m` expression involved — root and member alike — must be the flat
    deselect form (`not X and not Y`); anything richer fails loudly here rather
    than being approximated by a regex.  See _FLAT_DESELECT_RE.

    Sibling behavioural guards for the same bug class (this one is the static,
    member-agnostic generalisation — it covers a NEW member with no edit):
      - cockpit/tests/test_root_config_smoke_deselection.py
      - shared/tests/test_root_config_integration_deselection.py
      - fused-memory/tests/test_integration_marker_config.py
    """
    root_pyproject = REPO_ROOT / 'pyproject.toml'
    root_data = tomllib.loads(root_pyproject.read_text(encoding='utf-8'))

    root_ini = root_data.get('tool', {}).get('pytest', {}).get('ini_options', {})
    root_addopts = root_ini.get('addopts', '')
    root_registered = _registered_marker_names(root_ini)
    root_deselected = _deselected_markers(root_addopts, 'the ROOT pyproject.toml')

    members = root_data.get('tool', {}).get('uv', {}).get('workspace', {}).get('members', [])
    # Anti-vacuity: a typo'd key here would make the loop below iterate nothing
    # and turn this guard into a silent pass.
    assert members, (
        '[tool.uv.workspace].members is empty or missing in the root pyproject.toml — '
        'this guard reads it to discover which subprojects to check, so it would '
        'otherwise pass vacuously.'
    )

    # marker name -> members that deselect it, in workspace order.
    owners: dict[str, list[str]] = {}
    for member in members:
        addopts = _pytest_ini_options(REPO_ROOT / member / 'pyproject.toml').get('addopts')
        if not addopts:
            continue
        for marker in _deselected_markers(addopts, f'{member}/pyproject.toml'):
            owners.setdefault(marker, []).append(member)

    unregistered = [m for m in owners if m not in root_registered]
    undeselected = [m for m in owners if m not in root_deselected]

    assert not unregistered and not undeselected, _mirroring_failure_message(
        unregistered, undeselected, owners, root_deselected, root_addopts
    )
