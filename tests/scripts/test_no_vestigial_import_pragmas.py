"""Guards on how the root pyright table resolves its targets' bare imports.

Two of them check that no import in these suites carries a suppression that
suppresses nothing, deliberately SEPARATE because the two classes of vestigial
pragma are vestigial for DIFFERENT reasons and have DIFFERENT remedies:

* ``test_no_missing_imports_pragma_on_installed_module_import`` — class (b),
  about INTERPRETER/venv selection. ``pytest``, ``yaml`` and every other
  declared dependency resolve from the worktree ``.venv`` that the root
  ``[tool.pyright]`` ``venvPath``/``venv`` keys pin, never from ``extraPaths``
  at all.
* ``test_no_missing_imports_pragma_on_resolvable_import`` — class (a), about
  static import RESOLUTION on pyright's search roots: the root ``extraPaths``,
  the importing file's own directory and the execution root. This is the
  task-3456 case proper.

A single blanket "no such pragma anywhere in this directory" assertion would
collapse that distinction, hand a reader whichever remedy happened to be
written down rather than the one that fits, and additionally forbid a
legitimate future suppression for a genuinely unresolvable import — turning a
hygiene guard into a blanket ban. Neither pragma guard forbids a pragma that
is doing real work.

The other two keep a same-directory sibling the module pyright checks for a
bare import of its name, under the search order
``_search_roots_in_pyright_order`` states (task 8507):

* ``test_no_root_extrapaths_entry_holds_a_test_module_or_conftest`` — the
  CONFIG: no ``extraPaths`` root holds the test modules and conftests that
  suites define in parallel.
* ``test_no_bare_import_in_a_root_pyright_target_resolves_past_its_same_directory_sibling``
  — the USE: no bare import in the root table's targets resolves elsewhere
  while a same-directory sibling holds that module.

WHY THESE EXIST (task 4516). Task 3456 added ``scripts``,
``scripts/legibility`` and ``scripts/local-model-serving`` to the root
``[tool.pyright] extraPaths``, which made this directory's flat ``scripts/``
imports resolve statically. The pragmas suppressing those imports became
vestigial at that moment, and TWELVE of them survived it in eight files for
four months because nothing detected a suppression that had outlived its
condition. In the week between task 4516 being filed and being worked, task
4793 added THIRTEEN more of the identical class. That is the drift these
guards close.

PYRIGHT'S NATIVE ``reportUnnecessaryTypeIgnoreComment`` WAS CONSIDERED, AND
MEASURED, before these guards were written — it is a one-line opt-in, not an
unavailable capability, and it is the checker-native mechanism for this exact
defect class. Measured on this tip (2026-09-04, pinned pyright 1.1.408, with
``reportUnnecessaryTypeIgnoreComment = "error"`` temporarily added to the root
``[tool.pyright]`` table and the tree restored byte-clean afterwards):

* It DOES fire under ``typeCheckingMode = "basic"``. A probe file carrying a
  vestigial suppression on a resolvable ``import systemd_unit_parity`` drew
  ``error: Unnecessary "# pyright: ignore" rule: "reportMissingImports"``.
* Enabling it TODAY turns both declared gates RED on pre-existing, unrelated
  suppressions: ``pyright tests/scripts/`` → 8 errors across 6 files (7 of them
  nothing to do with imports: four ``# type: ignore`` comments and three
  ``# pyright: ignore`` on ``reportArgumentType``/``reportAttributeAccessIssue``),
  and ``pyright scripts/`` → 1 error in ``scripts/run_vllm_eval.py``. Two of
  those files are outside task 4516's module locks, so the cleanup that a
  one-line enable requires cannot be done by this task at all.
* It is also NOT the repo-wide win it looks like from here. The root table
  governs exactly three targets — ``tests/scripts/``, ``scripts/`` and the root
  ``include`` — because every other workspace member carries its OWN
  ``[tool.pyright]`` table and its declared gate runs ``--directory <member>``
  against that table. Repo-wide coverage means editing eight tables and
  clearing eight independent fallout sets, not one line.

So the native rule is a real and better long-term mechanism, deferred as its
own task rather than smuggled into a cleanup, and these guards are not a
reimplementation of something unavailable — they are the part that can run in
the pytest gate today, with an import-specific remedy in the failure message
that a generic "unnecessary suppression" diagnostic cannot give. Task 5086
was that task: it measured every ``[tool.pyright]`` table and kept the ROOT
table off, because the root table carries import pragmas that are load-bearing
only in narrower environments; the per-table decision and the opt-in set live in
``tests/scripts/test_pyright_unnecessary_ignore_opt_in.py``. When the native
rule is enabled in the ROOT ``[tool.pyright]`` table, shrink or delete this
file.

NOT A DOCUMENTATION META-TEST. These assert on suppression DIRECTIVES —
load-bearing inputs to the type checker, in the same category as the
``extraPaths`` membership assertion in
``tests/scripts/test_scripts_module_config.py::
test_root_pyright_extrapaths_resolves_scripts_imports`` — and never on prose,
docstrings, comment contents, names or annotations. The surviving
backtick-quoted PROSE mentions of the pragma in this directory are correctly
invisible to both guards, because both gate on "the code part of this line
parses as an import statement" rather than on a substring scan.
"""
from __future__ import annotations

import ast
import functools
import importlib
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Collection, Iterable
from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).parents[2]
_THIS_DIR = Path(__file__).parent
_SIBLING_SUITE_DIR = REPO_ROOT / 'scripts' / 'tests'

# BOTH suites that resolve flat `scripts/` imports against the ROOT
# `[tool.pyright]` table, not just the one this file lives in. `tests/scripts/`
# is covered by `uv run --project shared pyright tests/scripts/` and
# `scripts/tests/` by `uv run --project shared pyright scripts/` — two declared
# gates, one shared table, so the drift task 4516 removed here is available in
# both places and would next appear in whichever one nobody guarded.
# `scripts/tests/conftest.py` performs the same three sys.path insertions the
# extraPaths entries mirror, so its files import the same flat modules by the
# same names. It carries zero such pragmas today (measured 2026-09-04); this
# guard is what keeps that true.
_SCANNED_DIRS = (_THIS_DIR, _SIBLING_SUITE_DIR)

# What the root table's two declared gates check, recursively: `pyright
# tests/scripts/` and `pyright scripts/`, the latter including scripts/tests/.
_ROOT_PYRIGHT_TARGET_DIRS = (_THIS_DIR, REPO_ROOT / 'scripts')

# The rule these guards forbid suppressing, ASSEMBLED FROM PARTS rather than
# spelled as one literal. A guard that forbids a token must not itself contain
# that token: `grep -rn 'pyright: ignore\[reportMissingImports\]' tests/scripts/`
# is the operator-facing way to audit this invariant by hand, and a literal here
# would plant a permanent false positive in every such audit — including the one
# task 4516 used to verify its own cleanup.
_PRAGMA_RULE = 'reportMissingImports'
_PRAGMA = f'pyright: ignore[{_PRAGMA_RULE}]'

# A suppression is matched by PARSING THE DIRECTIVE, not by substring-testing
# one of its spellings. Four spellings suppress reportMissingImports on a line
# and all four must count, or the guard reproduces one level up the exact gap it
# exists to close — a suppression nothing detects:
#
#   import x  # pyright: ignore[<rule>]        single rule
#   import x  # pyright: ignore[<rule>, <other>] comma list
#   import x  # pyright: ignore                  bare: ALL rules
#   import x  # type: ignore                     bare: ALL rules
#
# (<rule> is spelled out nowhere in this file, for the audit reason above.)
#
# The bare `# pyright: ignore` is the form most likely to appear next: it is
# what a reader reaches for when they do not know the rule name. `# type:
# ignore` counts because pyright honours it as a blanket line suppression
# (`enableTypeIgnoreComments`, on by default) and this repo runs no mypy, so
# pyright is its only consumer — a bracketed mypy error code in it changes
# nothing for pyright, which ignores the bracket contents entirely. Measured
# 2026-09-04: zero of either bare form sits on an import line in the two scanned
# directories, so widening the match to them is green today and purely
# forward-looking.
_PYRIGHT_DIRECTIVE_RE = re.compile(r'#\s*pyright:\s*ignore(?:\[(?P<rules>[^\]]*)\])?')
_TYPE_IGNORE_RE = re.compile(r'#\s*type:\s*ignore')


def _suppresses_missing_imports(comment: str) -> bool:
    """True when *comment* suppresses ``reportMissingImports`` in any spelling.

    A directive with no bracket suppresses every rule on the line, so it covers
    this one; a bracketed one covers it only if the rule is in the (possibly
    comma-separated) list.
    """
    if _TYPE_IGNORE_RE.search(comment):
        return True
    for match in _PYRIGHT_DIRECTIVE_RE.finditer(comment):
        rules = match.group('rules')
        if rules is None:
            return True
        if _PRAGMA_RULE in {rule.strip() for rule in rules.split(',')}:
            return True
    return False


def _parse_import_statement(code: str) -> ast.Import | ast.ImportFrom | None:
    """Return the import node *code* denotes, or None if it is not an import line.

    GATES ON "PARSES AS AN IMPORT", never on a substring or a backtick
    heuristic. A plain substring scan for the pragma would also match the
    backtick-quoted PROSE mentions in this directory —
    ``test_scripts_module_config.py``, ``test_root_py_type_gate.py`` and
    ``test_check_orchestrator_unit_parity.py`` all discuss these directives in
    docstrings and assertion messages — which must never be flagged. Gating on
    "the code part is an import" is a property of what the line IS; a backtick
    heuristic is incidental to how those happen to be written today and would
    silently stop working the moment one was reworded.

    The line is dedented before parsing because the pragma sites include
    function-body imports, and closed with a placeholder name when it is the
    opening line of a parenthesised ``from X import (`` — a real import
    statement that does not parse on its own. Only the module name is read off
    the result, so the placeholder never escapes this helper.
    """
    stripped = code.strip()
    if not stripped:
        return None

    candidates = [stripped]
    if stripped.endswith('('):
        candidates.append(stripped + '_placeholder)')

    for candidate in candidates:
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        if len(tree.body) != 1:
            continue
        node = tree.body[0]
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return node
    return None


def _pragma_carrying_imports(
    directories: Iterable[Path],
) -> list[tuple[Path, int, str, ast.Import | ast.ImportFrom]]:
    """Every ``(path, lineno, line, node)`` in *directories* where an import is suppressed.

    ONE scanner for both guards, so they cannot disagree about what counts as
    an import site — only about which sites are legitimate.

    Top level of each scanned directory. Splitting on the first ``#`` is safe
    for an import line specifically: an import statement cannot contain a string
    literal, so the first ``#`` on a line whose code part parses as an import is
    necessarily the comment delimiter.
    """
    sites: list[tuple[Path, int, str, ast.Import | ast.ImportFrom]] = []
    for directory in directories:
        assert directory.is_dir(), (
            f'{directory} does not exist, so this guard silently stopped covering it. '
            f'If that suite moved or was renamed, update _SCANNED_DIRS — a guard that '
            f'quietly narrows its own scope is the failure mode task 4516 removed'
        )
        for path in sorted(directory.glob('*.py')):
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                code, sep, comment = line.partition('#')
                if not sep or not _suppresses_missing_imports(sep + comment):
                    continue
                node = _parse_import_statement(code)
                if node is not None:
                    # The WHOLE line, directive included, because the failure
                    # message must show WHICH of the four matched spellings is
                    # there — "delete the pragma" is not actionable otherwise.
                    sites.append((path, lineno, line.strip(), node))
    return sites


def _load_root_pyright_config() -> dict:
    """Return the ``[tool.pyright]`` section of the ROOT pyproject.toml, or {}.

    Same shape as ``tests/scripts/test_scripts_module_config.py::
    _load_root_pyright_config``, COPIED rather than imported: task 4516's scope
    forbids touching that file, and a cross-import would couple this guard to a
    module it must leave alone — which is also the no-cross-import convention
    this directory's guard family already states about itself.

    The ROOT table is the one that governs, because both declared type gates
    (``uv run --project shared pyright tests/scripts/`` and
    ``uv run --project shared pyright scripts/``) run from the repo root with
    no ``--directory``.
    """
    toml_path = REPO_ROOT / 'pyproject.toml'
    assert toml_path.is_file(), f'pyproject.toml not found at {toml_path}'
    with open(toml_path, 'rb') as fh:
        config = tomllib.load(fh)
    return config.get('tool', {}).get('pyright', {})


@functools.cache
def _root_extra_paths() -> tuple[str, ...]:
    """The ROOT ``[tool.pyright] extraPaths`` entries, in config order."""
    return tuple(_load_root_pyright_config().get('extraPaths', []))


def _search_roots_in_pyright_order(
    execution_root: Path, extra_path_roots: Iterable[Path], importing_dir: Path
) -> list[Path]:
    """The roots pyright (measured, 1.1.408) resolves a bare top-level import against, in its order.

    Three sources, none of them redundant:

    1. The execution root, which pyright resolves local imports against. Here
       that is the repo root, holding ``conftest.py`` and
       ``df_pytest_isolation.py``.
    2. Every ``extraPaths`` root, in config order. The root table's entries are
       what make the flat ``scripts/`` imports resolve (task 3456).
    3. The importing file's OWN directory, pyright's last-resort local fallback.
       ``--import-mode=importlib`` means pytest does not put a test file's own
       directory on sys.path, so each suite's ``conftest.py`` inserts what it
       needs explicitly; pyright reaches the same modules because a bare import
       resolves against the importing file's own directory.
       ``setup_host_parsing.py`` lives in ``tests/scripts/``, not in
       ``scripts/``, so it resolves as a same-directory sibling and would
       otherwise be misjudged unresolvable. Taking it per-file rather than as a
       constant is what lets one scanner serve both scanned suites without
       either borrowing the other's siblings.

    So a module on any ``extraPaths`` root claims its name over a sibling of
    the importing file, which pyright then never checks for that import. This
    is the precedence the two task-8507 guards exist for.
    """
    return [execution_root, *extra_path_roots, importing_dir]


def _import_search_roots(importing_dir: Path) -> list[Path]:
    """The ROOT table's search roots for a bare import from *importing_dir*.

    DERIVED FROM CONFIG, never hardcoded — see this guard's docstring for why
    that is the whole point rather than a stylistic preference.
    """
    return _search_roots_in_pyright_order(
        REPO_ROOT, (REPO_ROOT / entry for entry in _root_extra_paths()), importing_dir
    )


def _imported_top_level_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
    """The top-level module name(s) *node* imports, or [] for a relative import.

    Covers all three spellings present in these directories — ``import X``,
    ``import X as Y`` and ``from X import (...)`` — and takes the first dotted
    component, since that is the name resolved against a search root.

    A relative import (``level > 0``) yields [] and is therefore never reported
    as resolvable: it is not resolved against these roots at all, so this guard
    has nothing to say about a suppression on one.
    """
    if isinstance(node, ast.ImportFrom):
        if node.level != 0 or not node.module:
            return []
        return [node.module.split('.')[0]]
    return [alias.name.split('.')[0] for alias in node.names]


def _holds_module_or_regular_package(directory: Path, module: str) -> bool:
    return (directory / f'{module}.py').is_file() or (
        directory / module / '__init__.py'
    ).is_file()


def _resolving_root(module: str, roots: list[Path]) -> Path | None:
    """The *root* pyright resolves *module* on as a bare top-level import, else None.

    The first root holding a module or a regular package wins; a bare namespace
    directory is taken only when no root holds either.
    """
    for root in roots:
        if _holds_module_or_regular_package(root, module):
            return root
    for root in roots:
        if (root / module).is_dir():
            return root
    return None


def _pyright_resolving_root(importing_dir: Path, module: str) -> Path | None:
    """The root table's search root on which pyright resolves a bare import of *module* from *importing_dir*, else None.

    The one resolver every guard here asks: the class-(a)/(b) pragma split, so
    those two cannot both claim a site nor both miss it, and the
    shadowed-sibling guard.
    """
    return _resolving_root(module, _import_search_roots(importing_dir))


def _resolvable_import_offenders(
    sites: Iterable[tuple[Path, int, str, ast.Import | ast.ImportFrom]],
) -> list[tuple[Path, int, str, str, Path]]:
    """The ``(path, lineno, line, module, root)`` among *sites* whose module resolves on *root*."""
    return [
        (path, lineno, code, module, root)
        for path, lineno, code, node in sites
        for module in _imported_top_level_modules(node)
        if (root := _pyright_resolving_root(path.parent, module)) is not None
    ]


def _shadowed_siblings(directory: Path) -> dict[str, Path]:
    """Each module or regular package in *directory* that pyright resolves on another root, mapped to that root."""
    siblings = {path.stem for path in directory.glob('*.py') if path.is_file()} | {
        path.parent.name for path in directory.glob('*/__init__.py') if path.is_file()
    }
    return {
        name: root
        for name in sorted(siblings)
        if name.isidentifier()
        and (root := _pyright_resolving_root(directory, name)) is not None
        and root != directory
    }


def _may_import_any(text: str, modules: Collection[str]) -> bool:
    """False only when *text* holds no import statement of any of *modules*.

    Once backslash continuations are joined, an import statement names its
    module on the same line as its ``import`` keyword.
    """
    return any(
        'import' in line and any(module in line for module in modules)
        for line in text.replace('\\\n', ' ').splitlines()
    )


def _bare_imports_naming(path: Path, modules: Collection[str]) -> list[tuple[int, str]]:
    """The ``(lineno, module)`` of each bare import in *path* of one of *modules*.

    Only a file that ``_may_import_any`` of them is parsed.
    """
    text = path.read_text()
    if not _may_import_any(text, modules):
        return []
    return [
        (node.lineno, module)
        for node in ast.walk(ast.parse(text, filename=str(path)))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for module in _imported_top_level_modules(node)
        if module in modules
    ]


def _shadowed_sibling_imports(paths: Iterable[Path]) -> list[tuple[Path, int, str, Path]]:
    """The ``(path, lineno, module, root)`` of each bare import in *paths* that pyright resolves on *root*.

    Reported only when *path*'s own directory also holds *module*: pyright
    never checks that sibling for the import, whichever of the two the runtime
    binds.
    """
    paths_by_dir: dict[Path, list[Path]] = {}
    for path in paths:
        paths_by_dir.setdefault(path.parent, []).append(path)
    shadowed: list[tuple[Path, int, str, Path]] = []
    for directory, dir_paths in paths_by_dir.items():
        root_by_sibling = _shadowed_siblings(directory)
        if not root_by_sibling:
            continue
        for path in dir_paths:
            shadowed.extend(
                (path, lineno, module, root_by_sibling[module])
                for lineno, module in _bare_imports_naming(path, root_by_sibling)
            )
    return shadowed


# The verdict goes on its own final line because site's `sitecustomize` and
# `.pth` lines run before this code and may print to stdout themselves.
_FIND_SPEC_PROBE = '''
import importlib.util
import json
import sys


def found(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


print()
print(json.dumps([name for name in sys.argv[1:] if found(name)]))
'''


def _installed_modules(modules: Iterable[str]) -> frozenset[str]:
    """Those of *modules* that a fresh ``-I`` interpreter of this environment can locate.

    That interpreter sees the venv's site-packages and ``.pth`` entries, but not
    the sys.path and sys.modules entries that collected test modules add.
    """
    probe = subprocess.run(
        [sys.executable, '-I', '-c', _FIND_SPEC_PROBE, *sorted(modules)],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, (
        f'the installed-module probe exited {probe.returncode} under '
        f'{sys.executable}:\n{probe.stderr}'
    )
    return frozenset(json.loads(probe.stdout.splitlines()[-1]))


def _installed_module_offenders(
    sites: Iterable[tuple[Path, int, str, ast.Import | ast.ImportFrom]],
) -> list[tuple[Path, int, str, str]]:
    """The ``(path, lineno, line, module)`` among *sites* whose module is installed.

    A module that resolves on class (a)'s search roots is excluded first, so it
    is never reported here as well.
    """
    candidates = [
        (path, lineno, code, module)
        for path, lineno, code, node in sites
        for module in _imported_top_level_modules(node)
        if _pyright_resolving_root(path.parent, module) is None
    ]
    installed = _installed_modules(module for *_, module in candidates)
    return [
        (path, lineno, code, module)
        for path, lineno, code, module in candidates
        if module in installed
    ]


def test_no_missing_imports_pragma_on_installed_module_import() -> None:
    """No import of an INSTALLED module may carry a ``reportMissingImports`` suppression.

    CLASS (b), and vestigial for a reason that has nothing to do with
    ``extraPaths``: a declared dependency resolves from the worktree ``.venv``
    that the root ``[tool.pyright]`` ``venvPath = "."`` / ``venv = ".venv"``
    keys pin. It is not on ``extraPaths`` and never could be.

    So an unresolved dependency is not an import problem at all — it means the
    ENVIRONMENT is broken, typically a worktree that has never been cold-synced
    and therefore has no ``.venv``, in which case pyright prints
    ``venv .venv subdirectory not found in venv path <worktree>`` and silently
    falls back to whatever ambient interpreter it can find. That fallback is
    precisely the condition the type gate must report LOUDLY, because every
    other diagnostic in the run was then measured against the wrong
    interpreter. A suppression on this line converts that env fault into
    silence.

    ASKS A FRESH INTERPRETER OF THIS ENVIRONMENT, rather than naming ``pytest``. The five pragmas task
    4516 deleted in this class all sat on ``import pytest``, but the rationale
    above is a property of "resolved from the venv", not of that one name:
    ``import yaml`` under the same suppression would be equally vestigial and
    would mask the same env fault. Naming pytest would leave that third case
    caught by neither guard, since a site-packages module resolves on none of
    class (a)'s search roots either.

    THE SPLIT IS PRESERVED by checking class (a) FIRST: a module that resolves
    on class (a)'s search roots belongs to the other guard and its other
    remedy, and is skipped here even when a fresh interpreter can also import
    it.

    These pragmas were provably copy-paste cargo rather than considered
    suppressions, which is visible without appeal to intent because they were
    internally inconsistent with their own neighbours: in
    ``test_backfill_decision_queue_stamp.py`` the pragma sat on ``import
    pytest`` while the genuine flat ``scripts/`` import on the very next line
    carried none; in ``test_dashboard_watchdog.py`` the ``import yaml`` just
    below carried none; and in ``test_trial_module_tagger_haiku.py`` the
    polarity was simply reversed.
    """
    offenders = _installed_module_offenders(_pragma_carrying_imports(_SCANNED_DIRS))

    assert not offenders, (
        f'{len(offenders)} import(s) of an installed module carry a '
        f'suppression covering `{_PRAGMA_RULE}` (canonically `{_PRAGMA}`, but '
        f'a comma list containing it, a bare pyright ignore or a type ignore '
        f'all count):\n'
        + '\n'.join(
            f'  {path.relative_to(REPO_ROOT)}:{lineno}: {code}\n'
            f'      `{module}` is importable by a fresh interpreter of this environment'
            for path, lineno, code, module in offenders
        )
        + f'\nDELETE the suppression; keep the import byte-identical. These '
        f'modules '
        f'are declared dependencies of the environment that runs this suite '
        f'and resolve from the worktree .venv that the root [tool.pyright] '
        f'venvPath="."/venv=".venv" keys pin — NOT from extraPaths, which they '
        f'are not on and could not be. An unresolved dependency therefore means '
        f'a BROKEN ENVIRONMENT (usually a worktree that was never cold-synced, '
        f'so pyright prints `venv .venv subdirectory not found in venv path` '
        f'and silently falls back to an ambient interpreter, making every '
        f'other diagnostic in that run a measurement of the wrong '
        f'interpreter). The type gate must report that LOUDLY; suppressing '
        f'{_PRAGMA_RULE} here converts a real env fault into silence. The '
        f'remedy is `uv sync --all-packages`, never a pragma'
    )


def test_no_missing_imports_pragma_on_resolvable_import() -> None:
    """No import that the search roots already resolve may carry the pragma.

    CLASS (a) — the task-3456 case proper. Task 3456 added ``scripts``,
    ``scripts/legibility`` and ``scripts/local-model-serving`` to the root
    ``[tool.pyright] extraPaths`` precisely so these suites' flat ``scripts/``
    imports would resolve statically. Every suppression on such an import
    became vestigial at that moment: it suppresses nothing, and it silently
    masks a REAL ``reportMissingImports`` if one ever appears on that line
    later.

    DERIVED FROM ``extraPaths`` AT TEST TIME, NOT HARDCODED, and that is the
    durable point of this guard rather than a stylistic preference. This task
    exists because a suppression outlived the condition that justified it and
    nothing detected the drift for four months. A guard that hardcoded "these
    eight files must not contain this string" would reproduce that exact
    failure mode one level up: it would freeze a single day's measurement as a
    permanent truth, go stale silently the moment ``extraPaths`` changed, and
    need hand-editing every time a test file was added. Reading the config
    instead makes the guard track the MECHANISM rather than a snapshot of its
    consequences — a future ``extraPaths`` entry that makes some pragma
    vestigial turns this red automatically and names the pragma to delete,
    while an import that genuinely does not resolve keeps it green and its
    suppression allowed.

    MEMBERSHIP/RESOLVABILITY, never list equality or a count pin, for the
    reason ``test_scripts_module_config.py::
    test_root_pyright_extrapaths_resolves_scripts_imports`` states about its
    own assertion: a future entry is a legitimate change, not a regression.

    THE REMEDY IS NEVER A PRAGMA. If a flat ``scripts/`` import ever stops
    resolving, the fix is restoring the ``extraPaths`` entries — removing one
    is a TWO-gate outage, ``pyright scripts/`` and ``pyright tests/scripts/``
    both — not re-adding suppressions to either suite.
    """
    offenders = _resolvable_import_offenders(_pragma_carrying_imports(_SCANNED_DIRS))

    assert not offenders, (
        f'{len(offenders)} import(s) carry a suppression covering '
        f'`{_PRAGMA_RULE}` (canonically `{_PRAGMA}`, but a comma list '
        f'containing it, a bare pyright ignore or a type ignore all count) '
        f'while already resolving on the search roots pyright uses for this '
        f'suite:\n'
        + '\n'.join(
            f'  {path.relative_to(REPO_ROOT)}:{lineno}: {code}\n'
            f'      `{module}` resolves at {root.relative_to(REPO_ROOT)}/'
            for path, lineno, code, module, root in offenders
        )
        + f'\nDELETE the suppression; keep the import byte-identical. A '
        f'suppression on a resolvable import suppresses NOTHING today, and '
        f'silently masks a '
        f'REAL {_PRAGMA_RULE} if one appears on that line later — which is '
        f'the whole defect task 4516 removed. These roots are read from the '
        f'ROOT [tool.pyright] extraPaths at test time (plus the importing '
        f'file\'s own directory, for same-directory sibling modules, and the '
        f'execution root, for root-level modules), so this '
        f'list is what pyright actually resolves against, not a snapshot of '
        f'it. IF AN IMPORT HERE EVER STOPS RESOLVING, THE FIX IS RESTORING THE '
        f'extraPaths ENTRIES, NOT RE-ADDING PRAGMAS: task 3456 added '
        f'`scripts`, `scripts/legibility` and `scripts/local-model-serving` '
        f'exactly so these resolve, and removing one is a TWO-gate outage '
        f'(`pyright scripts/` and `pyright tests/scripts/` both run from the '
        f'repo root against this same table)'
    )


def test_an_import_reachable_only_through_a_session_sys_path_entry_is_not_an_installed_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = '_vestigial_pragma_guard_session_only_helper'
    helpers_dir = tmp_path / 'helpers'
    helpers_dir.mkdir()
    (helpers_dir / f'{helper}.py').write_text('VALUE = 1\n')
    suite_dir = tmp_path / 'suite'
    suite_dir.mkdir()
    (suite_dir / 'test_probe.py').write_text(
        f'from {helper} import VALUE  # {_PRAGMA}\n'
        f'import pytest  # {_PRAGMA}\n'
        f'import sys  # {_PRAGMA}\n'
    )

    monkeypatch.syspath_prepend(str(helpers_dir))
    importlib.import_module(helper)
    # Re-registered through setitem so teardown deletes it: delitem on a key
    # that is not yet present records nothing to undo.
    monkeypatch.setitem(sys.modules, helper, sys.modules.pop(helper))

    offenders = _installed_module_offenders(_pragma_carrying_imports([suite_dir]))

    judged_installed = [module for *_, module in offenders]
    assert judged_installed == ['pytest', 'sys'], (
        f'expected exactly the third-party `pytest` and the builtin `sys` to be '
        f'judged installed, got {judged_installed}. `{helper}` must not be among '
        f'them: its directory {helpers_dir} reached sys.path only through this '
        f'test session, the way a collected test module puts one there, so the '
        f'environment pyright resolves from does not provide it'
    )


def test_a_suppressed_import_of_a_module_at_the_pyright_execution_root_is_reported_by_class_a(
    tmp_path: Path,
) -> None:
    root_modules = sorted(path.stem for path in REPO_ROOT.glob('*.py'))
    assert root_modules, (
        f'no *.py module found at {REPO_ROOT}, so this test would pass vacuously'
    )
    suite_dir = tmp_path / 'suite'
    suite_dir.mkdir()
    (suite_dir / 'test_probe.py').write_text(
        ''.join(f'import {module}  # {_PRAGMA}\n' for module in root_modules)
    )
    sites = _pragma_carrying_imports([suite_dir])

    reported_by_class_a = {
        module: root for *_, module, root in _resolvable_import_offenders(sites)
    }
    reported_by_class_b = [module for *_, module in _installed_module_offenders(sites)]

    assert reported_by_class_a == dict.fromkeys(root_modules, REPO_ROOT), (
        f'{root_modules} sit at the pyright execution root {REPO_ROOT}, the '
        f'directory holding the root pyproject.toml, and pyright resolves local '
        f'imports against it. So a suppression on an import of one is vestigial, '
        f'and class (a) must report it at that root; it reported '
        f'{reported_by_class_a}. A fresh interpreter of this environment cannot '
        f'import these modules, so without this root they would escape BOTH '
        f'guards'
    )
    assert not reported_by_class_b, (
        f'class (b) also reported {reported_by_class_b}, which class (a) already '
        f'claims; each site belongs to exactly one guard and its one remedy'
    )


def test_a_suppressed_import_of_a_package_is_reported_at_its_regular_package_root_not_a_namespace_directory(
    tmp_path: Path,
) -> None:
    package = 'orchestrator'
    namespace_dir = REPO_ROOT / package
    package_root = REPO_ROOT / package / 'src'
    assert namespace_dir.is_dir() and not (namespace_dir / '__init__.py').exists(), (
        f'{namespace_dir} is no longer a bare namespace directory, so this test '
        f'no longer pins the precedence it exists for'
    )
    assert (package_root / package / '__init__.py').is_file(), (
        f'{package_root} no longer holds the regular package `{package}`'
    )
    suite_dir = tmp_path / 'suite'
    suite_dir.mkdir()
    (suite_dir / 'test_probe.py').write_text(f'import {package}  # {_PRAGMA}\n')

    reported_by_class_a = {
        module: root
        for *_, module, root in _resolvable_import_offenders(
            _pragma_carrying_imports([suite_dir])
        )
    }

    assert reported_by_class_a == {package: package_root}, (
        f'class (a) reported {reported_by_class_a}. {namespace_dir} is only a '
        f'namespace directory, and pyright (measured, 1.1.408) resolves a '
        f'regular package on an extraPaths root ahead of a namespace directory '
        f'at the execution root, so `{package}` resolves at {package_root}. The '
        f'reported root must be the one pyright actually uses'
    )


def test_a_bare_import_resolves_on_the_execution_root_then_extrapaths_then_the_importing_directory(
    tmp_path: Path,
) -> None:
    module = '_search_order_probe'
    execution_root, extra_path_root, importing_dir = (
        tmp_path / name for name in ('execution_root', 'extra_path_root', 'suite')
    )
    roots = _search_roots_in_pyright_order(execution_root, [extra_path_root], importing_dir)

    winners = []
    for holder in (importing_dir, extra_path_root, execution_root):
        holder.mkdir()
        (holder / f'{module}.py').write_text('')
        winners.append(_resolving_root(module, roots))

    assert winners == [importing_dir, extra_path_root, execution_root], (
        f'adding `{module}` to the importing directory, then an extraPaths root, '
        f'then the execution root, resolved it at {winners} in turn. Pyright '
        f'(measured, 1.1.408) searches the execution root, then extraPaths, then '
        f'the importing file\'s own directory, so each addition must take over'
    )


def test_no_root_extrapaths_entry_holds_a_test_module_or_conftest() -> None:
    extra_paths = _root_extra_paths()
    assert extra_paths, (
        'the root [tool.pyright] table lists no extraPaths, so this test would '
        'pass vacuously'
    )
    suite_modules_by_entry = {
        entry: sorted(
            path.name
            for path in (REPO_ROOT / entry).glob('*.py')
            if path.name == 'conftest.py' or path.name.startswith('test_')
        )
        for entry in extra_paths
    }
    offending = {entry: names for entry, names in suite_modules_by_entry.items() if names}

    assert not offending, (
        'root [tool.pyright] extraPaths entries hold test modules or a conftest:\n'
        + '\n'.join(
            f'  {entry}/: {len(names)} file(s): {", ".join(names)}'
            for entry, names in offending.items()
        )
        + '\nA module on an extraPaths root claims its name over every '
        'same-directory sibling in this table\'s targets (see '
        '_search_roots_in_pyright_order), and test modules and conftests are '
        'exactly the names suites define in parallel. Put only the helpers '
        'these targets import into a dedicated directory, such as '
        'orchestrator/tests/cross_suite_helpers, and list that instead'
    )


def test_a_same_directory_sibling_that_a_root_extrapaths_module_shadows_is_reported(
    tmp_path: Path,
) -> None:
    root_modules = {path.stem for path in REPO_ROOT.glob('*.py')}
    shadowing = next(
        (
            (entry, stem)
            for entry in _root_extra_paths()
            for stem in sorted(path.stem for path in (REPO_ROOT / entry).glob('*.py'))
            if stem.isidentifier() and stem not in root_modules
        ),
        None,
    )
    assert shadowing is not None, (
        'no root extraPaths entry holds a top-level module that is not also at '
        f'{REPO_ROOT}, so this test would pass vacuously'
    )
    entry, module = shadowing
    sibling = '_shadow_guard_probe_sibling'
    suite_dir = tmp_path / 'suite'
    suite_dir.mkdir()
    (suite_dir / f'{module}.py').write_text('')
    (suite_dir / f'{sibling}.py').write_text('')
    probe = suite_dir / 'test_probe.py'
    probe.write_text(f'import {module}\nimport {sibling}\n')
    continued_probe = suite_dir / 'test_continued_probe.py'
    continued_probe.write_text(f'import {sibling}, \\\n    {module}\n')

    reported = [
        (path.name, reported_module, root)
        for path, _, reported_module, root in _shadowed_sibling_imports(
            [probe, continued_probe]
        )
    ]

    assert reported == [
        (probe.name, module, REPO_ROOT / entry),
        (continued_probe.name, module, REPO_ROOT / entry),
    ], (
        f'expected `{module}` reported at {REPO_ROOT / entry} once per probe, '
        f'the backslash-continued import included, got {reported}. Pyright '
        f'binds {entry}/{module}.py for that import, never the sibling '
        f'{suite_dir}/{module}.py; `{sibling}` exists only beside the importing '
        f'file, so pyright binds the sibling itself'
    )


def test_no_bare_import_in_a_root_pyright_target_resolves_past_its_same_directory_sibling() -> None:
    target_files = [
        path
        for directory in _ROOT_PYRIGHT_TARGET_DIRS
        for path in sorted(directory.rglob('*.py'))
        if not any(part.startswith('.') for part in path.relative_to(directory).parts)
    ]
    assert target_files, (
        f'no *.py found under {_ROOT_PYRIGHT_TARGET_DIRS}, so this test would '
        f'pass vacuously'
    )

    offenders = _shadowed_sibling_imports(target_files)

    assert not offenders, (
        f'{len(offenders)} bare import(s) resolve on a pyright search root '
        f'ahead of a same-directory sibling of the same name:\n'
        + '\n'.join(
            f'  {path.relative_to(REPO_ROOT)}:{lineno}: `{module}` resolves at '
            f'{root.relative_to(REPO_ROOT)}/ ahead of the sibling `{module}` in '
            f'{path.parent.relative_to(REPO_ROOT)}/'
            for path, lineno, module, root in offenders
        )
        + '\nPyright never checks the sibling for these lines. Which of the two '
        'the runtime binds depends on how the importing directory reaches '
        'sys.path: where it comes first (tests/scripts/conftest.py inserts it), '
        'the line type-checks against a file it does not run; where it comes '
        'last (scripts/tests/conftest.py appends it), the sibling is '
        'unreachable dead code. Rename the sibling either way, or take the '
        'colliding module off that extraPaths root (list a directory that holds '
        'only the helpers these targets import, as '
        'test_no_root_extrapaths_entry_holds_a_test_module_or_conftest '
        'requires). Never add a pragma'
    )
