"""The CLI and pragma contract shared by the fused-memory/scripts/check_*.py lints.

Each checker owns its rule logic, its rule code(s), its description and the globs
a directory argument expands to.  This module owns everything else, once:

  * ``Violation`` -- the one record type every checker emits.
  * The exemption pragma.  ``# noqa: <code> — <reason>`` on the nearest PRECEDING
    non-blank line exempts the node below it (``is_exempted``).  An em-dash or one
    or more ASCII hyphens separate the code from the reason, and the reason is
    mandatory: an unexplained suppression is not a suppression.  An inline
    trailing ``# noqa`` on the node's own line is deliberately NOT honoured, so a
    suppression always reads as a statement about the code below it.  Pragmas are
    keyed on the rule's own code (``exemption_pattern``) and codes are strictly
    separate: rules' remedies are unrelated, so a pragma written for one is not
    informed consent for another.  One grammar for every rule means an author
    learns it once.
  * The CLI driver (``run_cli``).  Paths are files or directories; a directory
    expands through the checker's globs (``discover_files``) while an explicit
    file is taken as given, because hooks/project-checks hands over staged files
    as-is.  A missing explicit path fails fast before anything is read.  A read
    failure on one file (OSError, or undecodable bytes) is reported on stderr
    without discarding violations found in other files.  Violations print to
    stdout as ``path:lineno:col: message`` (ruff-style), sorted across files by
    (filename, lineno, col_offset): ``ast.walk`` is breadth-first, so a checker's
    own emission order is not source order.
  * The exit ladder: 0 clean, 1 violations found, 2 fatal (a missing explicit
    path or any read failure, which outranks violations).

STDLIB ONLY.  hooks/project-checks runs the checkers with plain ``python3`` to
avoid uv environment resolution, and their suites prove it under
``python3 -I -S``.  A third-party import here would break every checker at once.

HOW A CHECKER IMPORTS THIS MODULE::

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from _lint_cli import ...
    finally:
        del sys.path[0]

A plain sibling import is not enough.  ``python3 -I`` implies ``-P``, and
``-P`` / ``PYTHONSAFEPATH`` drop the script's own directory from ``sys.path``, so
``from _lint_cli import ...`` fails with ModuleNotFoundError under the ``-I -S``
proofs.  Tests load a checker in-process by file path, with fused-memory/scripts
deliberately off ``sys.path``, and the import would fail there too.  Restoring
``sys.path`` in ``finally`` keeps scripts/ off it for those in-process loads,
while the module stays registered in ``sys.modules`` as ``_lint_cli``.  Ruff's
E402 exempts ``sys.path`` edits, so the bootstrap needs no suppression.

This module is imported, never run.
"""
from __future__ import annotations

import argparse
import functools
import re
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import NamedTuple


class Violation(NamedTuple):
    """A lint violation found by a checker."""

    filename: str
    lineno: int
    col_offset: int
    message: str


_EXEMPT_TEMPLATE = r'#\s*noqa:\s*{code}\s*[—\-]+\s*\S.*'


@functools.cache
def exemption_pattern(code: str) -> re.Pattern[str]:
    """Return the pattern a stripped line must match to exempt a *code* violation."""
    return re.compile(_EXEMPT_TEMPLATE.format(code=re.escape(code)))


def is_exempted(lines: Sequence[str], lineno: int, code: str) -> bool:
    """Return True if the node at *lineno* (1-based) carries a valid *code* exemption.

    Walks upward from the line above *lineno* over blank lines to the nearest
    non-blank line, which must match ``exemption_pattern(code)``.  Any other
    non-blank line breaks the exemption.
    """
    idx = lineno - 2  # 0-based index of the line immediately above the node
    while idx >= 0:
        stripped = lines[idx].strip()
        if stripped == '':
            idx -= 1
            continue
        return bool(exemption_pattern(code).match(stripped))
    return False


def discover_files(directory: Path, globs: Iterable[str]) -> list[Path]:
    """Return every file under *directory* matching any of *globs*, recursively and sorted."""
    found: set[Path] = set()
    for pattern in globs:
        found.update(directory.rglob(pattern))
    return sorted(found)


def _scan_every_file(filename: str) -> bool:
    return True


def run_cli(
    argv: Sequence[str] | None,
    *,
    description: str,
    discovery_globs: tuple[str, ...],
    find_violations: Callable[[str, str], list[Violation]],
    is_scannable: Callable[[str], bool] = _scan_every_file,
) -> int:
    """Run a checker over the paths in *argv* and return its exit code.

    Directories expand through *discovery_globs*; explicit files are taken as
    given.  Every collected file, explicit or discovered, must pass
    *is_scannable* before it is read.  See the module docstring for the output
    and exit-code contract.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('paths', nargs='+', help='Files or directories to check')
    args = parser.parse_args(argv)

    # Phase 1: discovery.  Glob results exist at discovery time, so only explicit
    # paths need the existence check, and it completes before anything is read.
    files_to_scan: list[Path] = []
    for path_str in args.paths:
        p = Path(path_str)
        if p.is_dir():
            files_to_scan.extend(discover_files(p, discovery_globs))
        elif not p.exists():
            print(f'error: {p}: No such file or directory', file=sys.stderr)
            return 2
        else:
            files_to_scan.append(p)
    files_to_scan = [f for f in files_to_scan if is_scannable(str(f))]

    # Phase 2: scan, accumulating read errors so one bad file never discards
    # violations already collected from the others.
    all_violations: list[Violation] = []
    read_errors: list[tuple[Path, Exception]] = []
    for file_path in files_to_scan:
        try:
            source = file_path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as exc:
            read_errors.append((file_path, exc))
            continue
        all_violations.extend(find_violations(source, str(file_path)))

    # Phase 3: report.
    all_violations.sort(key=lambda v: (v.filename, v.lineno, v.col_offset))
    for v in all_violations:
        print(f'{v.filename}:{v.lineno}:{v.col_offset}: {v.message}')
    for file_path, exc in read_errors:
        print(f'error reading {file_path}: {exc}', file=sys.stderr)

    if read_errors:
        return 2
    return 1 if all_violations else 0
