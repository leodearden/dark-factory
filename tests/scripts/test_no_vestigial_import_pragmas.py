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
