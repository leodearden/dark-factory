"""The CLI and pragma contract shared by the fused-memory/scripts/check_*.py lints.

A checker supplies its rule (``find_violations``, emitting ``Violation``), its
description and its ``discover`` callable; ``run_cli`` owns the rest.

Pragma: ``# noqa: <code> — <reason>`` on the nearest PRECEDING non-blank line
exempts the node below it.  The separator is an em-dash or ASCII hyphens, the
reason is mandatory, an inline trailing pragma is not honoured, and a pragma
exempts only its own code.

Output: ``path:lineno:col: message`` on stdout, sorted by (path, lineno, col).
Exit codes: 0 clean, 1 violations found, 2 a missing explicit path or any read
failure.

Stdlib only, because hooks/project-checks runs the checkers with plain
``python3``.  A checker imports this module through this bootstrap, which works
under ``python3 -I`` and leaves ``sys.path`` as it found it::

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from _lint_cli import ...
    finally:
        del sys.path[0]
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
    discover: Callable[[Path], Iterable[Path]],
    find_violations: Callable[[str, str], list[Violation]],
    is_scannable: Callable[[str], bool] = _scan_every_file,
) -> int:
    """Run a checker over the paths in *argv* and return its exit code.

    Directories expand through *discover*; explicit files are taken as given.
    Every collected file, explicit or discovered, must pass *is_scannable*
    before it is read.  See the module docstring for the output and exit-code
    contract.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('paths', nargs='+', help='Files or directories to check')
    args = parser.parse_args(argv)

    # Phase 1: discovery.  Discovered files exist, so only explicit paths need the
    # existence check, and it completes before anything is read.
    files_to_scan: list[Path] = []
    for path_str in args.paths:
        p = Path(path_str)
        if p.is_dir():
            files_to_scan.extend(discover(p))
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
