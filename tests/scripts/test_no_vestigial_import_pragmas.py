"""Guards that no import in this directory carries a suppression that suppresses nothing.

Two guards, deliberately SEPARATE because the two classes of vestigial pragma
are vestigial for DIFFERENT reasons and have DIFFERENT remedies:

* ``test_no_missing_imports_pragma_on_pytest_import`` — class (b), about
  INTERPRETER/venv selection. ``pytest`` resolves from the worktree ``.venv``
  that the root ``[tool.pyright]`` ``venvPath``/``venv`` keys pin, never from
  ``extraPaths`` at all.
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
four months because nothing detects a suppression that has outlived its
condition: pyright's own ``reportUnnecessaryTypeIgnoreComment`` is set nowhere
in this repo and is off under ``typeCheckingMode = "basic"``. In the week
between task 4516 being filed and being worked, task 4793 added THIRTEEN more
of the identical class. That is the drift these guards close.

NOT A DOCUMENTATION META-TEST. These assert on suppression DIRECTIVES —
load-bearing inputs to the type checker, in the same category as the
``extraPaths`` membership assertion in
``tests/scripts/test_scripts_module_config.py::
test_root_pyright_extrapaths_resolves_scripts_imports`` — and never on prose,
docstrings, comment contents, names or annotations. The three surviving
backtick-quoted PROSE mentions of the pragma in this directory are correctly
invisible to both guards, because both gate on "the code part of this line
parses as an import statement" rather than on a substring scan.
"""
from __future__ import annotations

import ast
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
_THIS_DIR = Path(__file__).parent

# The suppression directive these guards forbid, ASSEMBLED FROM PARTS rather
# than spelled as one literal. A guard that forbids a token must not itself
# contain that token: `grep -rn 'pyright: ignore\[reportMissingImports\]'
# tests/scripts/` is the operator-facing way to audit this invariant by hand,
# and a literal here would plant a permanent false positive in every such
# audit — including the one task 4516 used to verify its own cleanup.
_PRAGMA_RULE = 'reportMissingImports'
_PRAGMA = f'pyright: ignore[{_PRAGMA_RULE}]'


def _parse_import_statement(code: str) -> ast.Import | ast.ImportFrom | None:
    """Return the import node *code* denotes, or None if it is not an import line.

    GATES ON "PARSES AS AN IMPORT", never on a substring or a backtick
    heuristic. A plain substring scan for the pragma would also match the
    backtick-quoted PROSE mentions in this directory —
    ``test_scripts_module_config.py`` (2) and
    ``test_check_orchestrator_unit_parity.py`` (1) — which must never be
    flagged. Gating on "the code part is an import" is a property of what the
    line IS; a backtick heuristic is incidental to how those three happen to be
    written today and would silently stop working the moment one was reworded.

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
    """Every ``(path, lineno, code, node)`` in this directory where an import carries the pragma.

    ONE scanner for both guards, so they cannot disagree about what counts as
    an import site — only about which sites are legitimate.

    Top level of this directory only, matching the scope of the two declared
    type gates that read the root ``[tool.pyright]`` table. Splitting on the
    first ``#`` is safe for an import line specifically: an import statement
    cannot contain a string literal, so the first ``#`` on a line whose code
    part parses as an import is necessarily the comment delimiter.
    """
    sites: list[tuple[Path, int, str, ast.Import | ast.ImportFrom]] = []
    for path in sorted(_THIS_DIR.glob('*.py')):
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            code, sep, comment = line.partition('#')
            if not sep or _PRAGMA not in comment:
                continue
            node = _parse_import_statement(code)
            if node is not None:
                sites.append((path, lineno, code.strip(), node))
    return sites


def _imports_pytest(node: ast.Import | ast.ImportFrom) -> bool:
    """True when *node* imports the ``pytest`` top-level module, in any of its spellings."""
    if isinstance(node, ast.ImportFrom):
        return node.level == 0 and (node.module or '').split('.')[0] == 'pytest'
    return any(alias.name.split('.')[0] == 'pytest' for alias in node.names)


def test_no_missing_imports_pragma_on_pytest_import() -> None:
    """No ``import pytest`` in this directory may carry a ``reportMissingImports`` pragma.

    CLASS (b), and vestigial for a reason that has nothing to do with
    ``extraPaths``: ``pytest`` is a declared dev dependency of every module that
    runs this suite, and it resolves from the worktree ``.venv`` that the root
    ``[tool.pyright]`` ``venvPath = "."`` / ``venv = ".venv"`` keys pin. It is
    not on ``extraPaths`` and never could be.

    So an unresolved ``pytest`` is not an import problem at all — it means the
    ENVIRONMENT is broken, typically a worktree that has never been cold-synced
    and therefore has no ``.venv``, in which case pyright prints
    ``venv .venv subdirectory not found in venv path <worktree>`` and silently
    falls back to whatever ambient interpreter it can find. That fallback is
    precisely the condition the type gate must report LOUDLY, because every
    other diagnostic in the run was then measured against the wrong
    interpreter. A suppression on this line converts that env fault into
    silence.

    These pragmas were provably copy-paste cargo rather than considered
    suppressions, which is visible without appeal to intent because they are
    internally inconsistent with their own neighbours: in
    ``test_backfill_decision_queue_stamp.py`` the pragma sat on ``import
    pytest`` while the genuine flat ``scripts/`` import on the very next line
    carried none; in ``test_dashboard_watchdog.py`` the ``import yaml`` just
    below carried none; and in ``test_trial_module_tagger_haiku.py`` the
    polarity was simply reversed.
    """
    offenders = [
        (path, lineno, code)
        for path, lineno, code, node in _pragma_carrying_imports()
        if _imports_pytest(node)
    ]

    assert not offenders, (
        f'{len(offenders)} pytest import(s) in {_THIS_DIR} carry a '
        f'`{_PRAGMA}` suppression:\n'
        + '\n'.join(
            f'  {path.relative_to(REPO_ROOT)}:{lineno}: {code}'
            for path, lineno, code in offenders
        )
        + f'\nDELETE the pragma; keep the import byte-identical. pytest is a '
        f'declared dev dependency of every module that runs this suite and '
        f'resolves from the worktree .venv that the root [tool.pyright] '
        f'venvPath="."/venv=".venv" keys pin — NOT from extraPaths, which it '
        f'is not on and could not be. An unresolved pytest therefore means a '
        f'BROKEN ENVIRONMENT (usually a worktree that was never cold-synced, '
        f'so pyright prints `venv .venv subdirectory not found in venv path` '
        f'and silently falls back to an ambient interpreter, making every '
        f'other diagnostic in that run a measurement of the wrong '
        f'interpreter). The type gate must report that LOUDLY; suppressing '
        f'{_PRAGMA_RULE} here converts a real env fault into silence. The '
        f'remedy for an unresolved pytest is `uv sync --all-packages`, never '
        f'a pragma'
    )


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


def _import_search_roots() -> list[Path]:
    """The directories pyright resolves a bare top-level import against, for this suite.

    DERIVED FROM CONFIG, never hardcoded — see this guard's docstring for why
    that is the whole point rather than a stylistic preference.

    Two sources, and the second is not redundant:

    1. Every ROOT ``[tool.pyright] extraPaths`` entry, resolved relative to the
       repo root. This is what makes the flat ``scripts/`` imports here resolve
       (task 3456).
    2. This directory itself. ``--import-mode=importlib`` means pytest does not
       put a test file's own directory on sys.path, so ``tests/scripts/
       conftest.py`` inserts it explicitly; pyright reaches the same modules
       because a bare import resolves against the importing file's own
       directory. ``setup_host_parsing.py`` lives HERE, not in ``scripts/``, so
       it resolves as a same-directory sibling and would otherwise be
       misjudged unresolvable.
    """
    extra_paths = _load_root_pyright_config().get('extraPaths', [])
    return [REPO_ROOT / entry for entry in extra_paths] + [_THIS_DIR]


def _imported_top_level_modules(node: ast.Import | ast.ImportFrom) -> list[str]:
    """The top-level module name(s) *node* imports, or [] for a relative import.

    Covers all three spellings present in this directory — ``import X``,
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


def test_no_missing_imports_pragma_on_resolvable_import() -> None:
    """No import that the search roots already resolve may carry the pragma.

    CLASS (a) — the task-3456 case proper. Task 3456 added ``scripts``,
    ``scripts/legibility`` and ``scripts/local-model-serving`` to the root
    ``[tool.pyright] extraPaths`` precisely so this directory's flat
    ``scripts/`` imports would resolve statically. Every suppression on such an
    import became vestigial at that moment: it suppresses nothing, and it
    silently masks a REAL ``reportMissingImports`` if one ever appears on that
    line later.

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

    THE REMEDY IS NEVER A PRAGMA. If a flat ``scripts/`` import here ever stops
    resolving, the fix is restoring the ``extraPaths`` entries — removing one
    is a TWO-gate outage, ``pyright scripts/`` and ``pyright tests/scripts/``
    both — not re-adding suppressions to this directory.
    """
    roots = _import_search_roots()

    offenders = [
        (path, lineno, code, module, root)
        for path, lineno, code, node in _pragma_carrying_imports()
        for module in _imported_top_level_modules(node)
        if (root := _resolving_root(module, roots)) is not None
    ]

    assert not offenders, (
        f'{len(offenders)} import(s) in {_THIS_DIR} carry a `{_PRAGMA}` '
        f'suppression while already resolving on the search roots pyright '
        f'uses for this suite:\n'
        + '\n'.join(
            f'  {path.relative_to(REPO_ROOT)}:{lineno}: {code}\n'
            f'      `{module}` resolves at {root.relative_to(REPO_ROOT)}/'
            for path, lineno, code, module, root in offenders
        )
        + f'\nDELETE the pragma; keep the import byte-identical. A pragma on a '
        f'resolvable import suppresses NOTHING today, and silently masks a '
        f'REAL {_PRAGMA_RULE} if one appears on that line later — which is '
        f'the whole defect task 4516 removed. These roots are read from the '
        f'ROOT [tool.pyright] extraPaths at test time (plus this directory '
        f'itself, for same-directory sibling modules), so this list is what '
        f'pyright actually resolves against, not a snapshot of it. '
        f'IF AN IMPORT HERE EVER STOPS RESOLVING, THE FIX IS RESTORING THE '
        f'extraPaths ENTRIES, NOT RE-ADDING PRAGMAS: task 3456 added '
        f'`scripts`, `scripts/legibility` and `scripts/local-model-serving` '
        f'exactly so these resolve, and removing one is a TWO-gate outage '
        f'(`pyright scripts/` and `pyright tests/scripts/` both run from the '
        f'repo root against this same table)'
    )
