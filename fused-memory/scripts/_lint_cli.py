"""The CLI, pragma and per-file debt-budget contract shared by the check_*.py lints.

A checker supplies its rule (``find_violations``, emitting ``Violation``), its
description and its ``discover`` callable; ``run_cli`` owns the rest.

Pragma: ``# noqa: <code> — <reason>`` on the nearest PRECEDING non-blank line
exempts the node below it.  The separator is an em-dash or ASCII hyphens, the
reason is mandatory, an inline trailing pragma is not honoured, and a pragma
exempts only its own code.

Output: ``path:lineno:col: message`` on stdout, sorted by (path, lineno, col).
Exit codes: 0 clean, 1 violations found, 2 a missing explicit path or any read
failure.

Debt budget: a rule that grandfathers pre-existing violations keeps a
SHRINK-ONLY ``path -> count`` baseline, built once into a ``DebtBaseline``.
``apply_debt_budget`` compares one file's violation COUNT with its row, so a
debt file is silent only at exactly its budget.

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
import dataclasses
import functools
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import MappingProxyType
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


class DebtEntry(NamedTuple):
    """One file's row in a debt baseline.

    *key* is the baseline's OWN spelling of the path, which a remedy must quote: the
    scanned filename may be absolute, and only the key is an edit the author can apply
    verbatim.
    """

    key: str
    budget: int


def _path_parts(filename: str) -> tuple[str, ...]:
    return PurePosixPath(filename.replace('\\', '/')).parts


@dataclasses.dataclass(frozen=True)
class DebtBaseline:
    """A rule's per-file debt baseline, matched on TRAILING PATH COMPONENTS.

    ``a/b/c.py`` matches a row ``b/c.py`` because its last two components are
    exactly ``b`` and ``c.py``; ``b/not_c.py``, which merely contains the row's
    filename as a substring, does not.  Trailing components rather than an exact
    compare, because the lint call sites pass repo-relative paths while pytest
    passes absolutes, and both must reach the same verdict.

    Rows are indexed by their final component, so ``entry_for`` is one dict lookup
    plus a compare per row sharing that basename, not a scan of every row.  Where
    two rows both match, the first in the source mapping's order wins.
    """

    _rows_by_name: Mapping[str, tuple[tuple[tuple[str, ...], DebtEntry], ...]]

    @classmethod
    def from_mapping(cls, debt: Mapping[str, int]) -> DebtBaseline:
        """Build the baseline once from *debt*, a ``path -> budget`` snapshot."""
        rows: dict[str, list[tuple[tuple[str, ...], DebtEntry]]] = {}
        for key, budget in debt.items():
            parts = PurePosixPath(key).parts
            if not parts:
                raise ValueError(f'a debt baseline row needs a path, got {key!r}')
            rows.setdefault(parts[-1], []).append((parts, DebtEntry(key, budget)))
        return cls(MappingProxyType({name: tuple(found) for name, found in rows.items()}))

    def entry_for(self, filename: str) -> DebtEntry | None:
        """Return *filename*'s row, or None, never a zero budget, if it has none.

        None lets a caller tell "no budget recorded: report every site" from
        "a budget of zero".
        """
        parts = _path_parts(filename)
        if not parts:
            return None
        for row_parts, entry in self._rows_by_name.get(parts[-1], ()):
            if parts[-len(row_parts) :] == row_parts:
                return entry
        return None


def apply_debt_budget(
    found: list[Violation],
    entry: DebtEntry | None,
    *,
    filename: str,
    build_overrun_msg: Callable[[DebtEntry, int], str],
    build_slack_msg: Callable[[DebtEntry, int], str],
) -> list[Violation]:
    """Compare one rule's violation COUNT for *filename* against its debt *entry*.

    This is the single place a file's count is compared with its budget.

    Both builders are the calling rule's own ``(entry, found) -> str`` message
    builders, REQUIRED and keyword-only.  Each rule's report carries its OWN remedy,
    so there is deliberately no default: a rule that forgets one fails at the call
    rather than silently inheriting another rule's remedy.  They are builders rather
    than built strings so each message is computed only on its own branch.

    - Not a debt file (*entry* is None) → every violation is reported unchanged.
    - Exactly at budget → silence: this is the grandfathering the baseline exists for.
    - Over budget → report exactly ``found - budget`` violations, so the noise is
      proportional to the overrun.  The reported sites are the last in source
      order.  That choice is deterministic rather than diagnostic, since the
      checker cannot know which site is new.
    - Under budget → ONE file-level violation at *filename* line 1, col 0, naming
      the budget edit.  Slack has no site, and a budget above the measurement
      would silently licence that many new violations.
    """
    if entry is None:
        return found
    if len(found) == entry.budget:
        return []
    if len(found) < entry.budget:
        return [
            Violation(
                filename=filename,
                lineno=1,
                col_offset=0,
                message=build_slack_msg(entry, len(found)),
            )
        ]
    ordered = sorted(found, key=lambda v: (v.lineno, v.col_offset))
    message = build_overrun_msg(entry, len(ordered))
    return [v._replace(message=message) for v in ordered[entry.budget :]]


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
