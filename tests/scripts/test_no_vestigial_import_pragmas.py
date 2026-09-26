"""Guards that no import in this suite carries a suppression that suppresses nothing.

Two guards, deliberately SEPARATE because the two classes of vestigial pragma
are vestigial for DIFFERENT reasons and have DIFFERENT remedies:

* ``test_no_missing_imports_pragma_on_installed_module_import`` — class (b),
  about INTERPRETER/venv selection. ``pytest``, ``yaml`` and every other
  declared dependency resolve from the worktree ``.venv`` that the root
  ``[tool.pyright]`` ``venvPath``/``venv`` keys pin, never from ``extraPaths``
  at all.
* ``test_no_missing_imports_pragma_on_resolvable_import`` — class (a), about
  static import RESOLUTION via ``extraPaths``. This is the task-3456 case
  proper.

A single blanket "no such pragma anywhere in this directory" assertion would
collapse that distinction, hand a reader whichever remedy happened to be
written down rather than the one that fits, and additionally forbid a
legitimate future suppression for a genuinely unresolvable import — turning a
hygiene guard into a blanket ban. Neither guard here forbids a pragma that is
doing real work.

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
that a generic "unnecessary suppression" diagnostic cannot give. When the
native rule is enabled repo-wide, shrink or delete this file.

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
import importlib.util
import re
import tomllib
from pathlib import Path

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


def _pragma_carrying_imports() -> list[tuple[Path, int, str, ast.Import | ast.ImportFrom]]:
    """Every ``(path, lineno, line, node)`` in the scanned suites where an import is suppressed.

    ONE scanner for both guards, so they cannot disagree about what counts as
    an import site — only about which sites are legitimate.

    Top level of each scanned directory. Splitting on the first ``#`` is safe
    for an import line specifically: an import statement cannot contain a string
    literal, so the first ``#`` on a line whose code part parses as an import is
    necessarily the comment delimiter.
    """
    sites: list[tuple[Path, int, str, ast.Import | ast.ImportFrom]] = []
    for directory in _SCANNED_DIRS:
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


def _import_search_roots(importing_dir: Path) -> list[Path]:
    """The directories pyright resolves a bare top-level import against, from *importing_dir*.

    DERIVED FROM CONFIG, never hardcoded — see this guard's docstring for why
    that is the whole point rather than a stylistic preference.

    Two sources, and the second is not redundant:

    1. Every ROOT ``[tool.pyright] extraPaths`` entry, resolved relative to the
       repo root. This is what makes the flat ``scripts/`` imports here resolve
       (task 3456).
    2. The importing file's OWN directory. ``--import-mode=importlib`` means
       pytest does not put a test file's own directory on sys.path, so each
       suite's ``conftest.py`` inserts what it needs explicitly; pyright reaches
       the same modules because a bare import resolves against the importing
       file's own directory. ``setup_host_parsing.py`` lives in
       ``tests/scripts/``, not in ``scripts/``, so it resolves as a
       same-directory sibling and would otherwise be misjudged unresolvable.
       Taking it per-file rather than as a constant is what lets one scanner
       serve both scanned suites without either borrowing the other's siblings.
    """
    extra_paths = _load_root_pyright_config().get('extraPaths', [])
    return [REPO_ROOT / entry for entry in extra_paths] + [importing_dir]


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


def _resolving_root(module: str, roots: list[Path]) -> Path | None:
    """The first *root* on which *module* resolves as a bare top-level import, else None."""
    for root in roots:
        if (root / f'{module}.py').is_file() or (root / module).is_dir():
            return root
    return None


def _resolves_in_running_interpreter(module: str) -> bool:
    """True when *module* is importable in the interpreter running this test.

    Asks the interpreter rather than naming modules, so this covers ``pytest``,
    ``yaml`` and every other declared dependency with one rule and needs no edit
    when a new third-party import lands in either scanned suite.

    ``find_spec`` does not execute *module* — it only locates it — and a
    top-level name has no parent package to import as a side effect.
    """
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


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

    ASKS THE INTERPRETER, rather than naming ``pytest``. The five pragmas task
    4516 deleted in this class all sat on ``import pytest``, but the rationale
    above is a property of "resolved from the venv", not of that one name:
    ``import yaml`` under the same suppression would be equally vestigial and
    would mask the same env fault. Naming pytest would leave that third case
    caught by neither guard, since a site-packages module resolves on none of
    class (a)'s search roots either.

    THE SPLIT IS PRESERVED by checking class (a) FIRST: a module that resolves
    on the ``extraPaths`` search roots belongs to the other guard and its other
    remedy, and is skipped here even though the running interpreter can also
    import it (each suite's ``conftest.py`` puts those same directories on
    sys.path, so it always can).

    These pragmas were provably copy-paste cargo rather than considered
    suppressions, which is visible without appeal to intent because they were
    internally inconsistent with their own neighbours: in
    ``test_backfill_decision_queue_stamp.py`` the pragma sat on ``import
    pytest`` while the genuine flat ``scripts/`` import on the very next line
    carried none; in ``test_dashboard_watchdog.py`` the ``import yaml`` just
    below carried none; and in ``test_trial_module_tagger_haiku.py`` the
    polarity was simply reversed.
    """
    offenders = [
        (path, lineno, code, module)
        for path, lineno, code, node in _pragma_carrying_imports()
        for module in _imported_top_level_modules(node)
        if _resolving_root(module, _import_search_roots(path.parent)) is None
        and _resolves_in_running_interpreter(module)
    ]

    assert not offenders, (
        f'{len(offenders)} import(s) of an installed module carry a '
        f'suppression covering `{_PRAGMA_RULE}` (canonically `{_PRAGMA}`, but '
        f'a comma list containing it, a bare pyright ignore or a type ignore '
        f'all count):\n'
        + '\n'.join(
            f'  {path.relative_to(REPO_ROOT)}:{lineno}: {code}\n'
            f'      `{module}` is importable in the running interpreter'
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
    offenders = [
        (path, lineno, code, module, root)
        for path, lineno, code, node in _pragma_carrying_imports()
        for module in _imported_top_level_modules(node)
        if (root := _resolving_root(module, _import_search_roots(path.parent))) is not None
    ]

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
        f'file\'s own directory, for same-directory sibling modules), so this '
        f'list is what pyright actually resolves against, not a snapshot of '
        f'it. IF AN IMPORT HERE EVER STOPS RESOLVING, THE FIX IS RESTORING THE '
        f'extraPaths ENTRIES, NOT RE-ADDING PRAGMAS: task 3456 added '
        f'`scripts`, `scripts/legibility` and `scripts/local-model-serving` '
        f'exactly so these resolve, and removing one is a TWO-gate outage '
        f'(`pyright scripts/` and `pyright tests/scripts/` both run from the '
        f'repo root against this same table)'
    )
