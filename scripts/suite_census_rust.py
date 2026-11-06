"""Part 2 of the suite census for cargo workspaces (task 5414): duplication among #[test] fns, per crate.

Test fns are found by a small lexer that understands Rust comments (nested
block comments included), strings, raw strings and char literals, so a brace
or ``fn`` inside any of them is never mistaken for code. The output is a
count per crate, never a ranking.
"""
from __future__ import annotations

import re
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from source_measures import tracked_files

_NO_CRATE = '(none)'
_TOTAL = 'total'
_FAMILY_MIN = 3

_LITERAL = (
    r'b?r(?P<hashes>#*)".*?"(?P=hashes)'
    r'|b?"(?:\\.|[^"\\])*"'
    r"|b?'(?:\\(?:u\{[0-9A-Fa-f]+\}|x[0-9A-Fa-f]{2}|.)|[^'\\])'"
)
_COMMENT_OR_LITERAL = re.compile(rf'(?P<line>//[^\n]*)|(?P<block>/\*)|(?:{_LITERAL})', re.S)
_BLOCK_EDGE = re.compile(r'/\*|\*/')
_TOKEN = re.compile(
    rf'(?P<literal>{_LITERAL})'
    r'|(?P<attr>#!?\[\s*(?P<path>[A-Za-z_][\w:]*))'
    r'|(?P<fn>\bfn\s+(?P<name>[A-Za-z_]\w*))'
    r'|(?P<open>\{)|(?P<close>\})',
    re.S,
)
_SIGNIFICANT = re.compile(r'[A-Za-z0-9]')


@dataclass(frozen=True)
class RustTestFn:
    name: str
    body_lines: tuple[str, ...]
    start_line: int


def _blank(text: str) -> str:
    return re.sub(r'[^\n]', ' ', text)


def _block_end(source: str, start: int) -> int:
    depth, pos = 1, start
    while depth and (edge := _BLOCK_EDGE.search(source, pos)) is not None:
        depth += 1 if edge.group() == '/*' else -1
        pos = edge.end()
    return pos if not depth else len(source)


def _without_comments(source: str) -> str:
    """*source* with every comment character turned into a space; offsets are kept."""
    pieces: list[str] = []
    pos = 0
    while (match := _COMMENT_OR_LITERAL.search(source, pos)) is not None:
        end = _block_end(source, match.end()) if match['block'] else match.end()
        pieces.append(source[pos:match.start()])
        commented = match['line'] or match['block']
        pieces.append(_blank(source[match.start():end]) if commented else source[match.start():end])
        pos = end
    pieces.append(source[pos:])
    return ''.join(pieces)


def _body_span(code: str, start: int) -> tuple[int, int]:
    """Offsets just inside the first ``{`` after *start* and its matching ``}``."""
    depth, opened = 0, len(code)
    pos = start
    while (token := _TOKEN.search(code, pos)) is not None:
        pos = token.end()
        if token['open']:
            opened = min(opened, token.end())
            depth += 1
        elif token['close'] and depth:
            depth -= 1
            if not depth:
                return opened, token.start()
    return opened, len(code)


def _is_test_attribute(path: str) -> bool:
    return path.rsplit('::', 1)[-1] == 'test'


def rust_test_fns(source: str) -> tuple[RustTestFn, ...]:
    code = _without_comments(source)
    found: list[RustTestFn] = []
    pending = False
    pos = 0
    while (token := _TOKEN.search(code, pos)) is not None:
        pos = token.end()
        if token['attr']:
            pending = pending or _is_test_attribute(token['path'])
        elif token['fn'] and pending:
            opened, closed = _body_span(code, token.end())
            found.append(RustTestFn(
                name=token['name'], body_lines=tuple(code[opened:closed].splitlines()),
                start_line=code.count('\n', 0, token.start()) + 1,
            ))
            pos, pending = closed + 1, False
        elif token['open'] or token['close']:
            pending = False
    return tuple(found)


# ---------------------------------------------------------------------------
# Crates, then the per-crate measures.

def _crates(tree_root: Path, unreadable: set[str]) -> dict[str, str]:
    """Crate directory -> [package] name, for every tracked Cargo.toml that has one."""
    crates: dict[str, str] = {}
    for manifest in tracked_files(tree_root, 'Cargo.toml', '*/Cargo.toml'):
        try:
            parsed = tomllib.loads((tree_root / manifest).read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
            unreadable.add(manifest)
            continue
        package = parsed.get('package')
        if isinstance(package, dict) and isinstance(package.get('name'), str):
            crates[str(PurePosixPath(manifest).parent)] = package['name']
    return crates


def _crate_of(path: str, crates: Mapping[str, str]) -> str:
    for parent in PurePosixPath(path).parents:
        if str(parent) in crates:
            return crates[str(parent)]
    return _NO_CRATE


def _family(name: str) -> str:
    return name.rpartition('_')[0] or name


@dataclass(frozen=True)
class RustCrateRow:
    crate: str
    test_files: int
    test_fns: int
    non_trivial_lines: int
    duplicated_lines: int
    redundant_lines: int
    family_members: int
    largest_family: tuple[int, str, str] | None

    @property
    def duplicated_share(self) -> float:
        return self.duplicated_lines / self.non_trivial_lines if self.non_trivial_lines else 0.0

    @property
    def family_share(self) -> float:
        return self.family_members / self.test_fns if self.test_fns else 0.0


@dataclass(frozen=True)
class RustDuplicationCensus:
    rows: tuple[RustCrateRow, ...]
    totals: RustCrateRow
    unreadable: tuple[str, ...]
    complete: bool


def _largest(families: Iterable[tuple[int, str, str]]) -> tuple[int, str, str] | None:
    return min(families, key=lambda f: (-f[0], f[1], f[2]), default=None)


def _crate_row(crate: str, files: Mapping[str, Sequence[RustTestFn]]) -> RustCrateRow:
    lines = Counter(
        stripped
        for fns in files.values() for fn in fns for line in fn.body_lines
        if _SIGNIFICANT.search(stripped := line.strip())
    )
    families = [
        (size, path, key)
        for path, fns in files.items()
        for key, size in Counter(_family(fn.name) for fn in fns).items()
        if size >= _FAMILY_MIN
    ]
    return RustCrateRow(
        crate=crate, test_files=len(files), test_fns=sum(len(fns) for fns in files.values()),
        non_trivial_lines=sum(lines.values()),
        duplicated_lines=sum(count for count in lines.values() if count > 1),
        redundant_lines=sum(count - 1 for count in lines.values()),
        family_members=sum(size for size, _, _ in families),
        largest_family=_largest(families),
    )


def _totals(rows: Sequence[RustCrateRow]) -> RustCrateRow:
    return RustCrateRow(
        crate=_TOTAL,
        test_files=sum(row.test_files for row in rows),
        test_fns=sum(row.test_fns for row in rows),
        non_trivial_lines=sum(row.non_trivial_lines for row in rows),
        duplicated_lines=sum(row.duplicated_lines for row in rows),
        redundant_lines=sum(row.redundant_lines for row in rows),
        family_members=sum(row.family_members for row in rows),
        largest_family=_largest(row.largest_family for row in rows if row.largest_family),
    )


def measure_rust_tree(tree_root: Path) -> RustDuplicationCensus:
    unreadable: set[str] = set()
    crates = _crates(tree_root, unreadable)
    by_crate: defaultdict[str, dict[str, tuple[RustTestFn, ...]]] = defaultdict(dict)
    for path in tracked_files(tree_root, '*.rs'):
        try:
            fns = rust_test_fns((tree_root / path).read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError):
            unreadable.add(path)
            continue
        if fns:
            by_crate[_crate_of(path, crates)][path] = fns
    rows = tuple(_crate_row(crate, files) for crate, files in sorted(by_crate.items()))
    return RustDuplicationCensus(
        rows=rows, totals=_totals(rows), unreadable=tuple(sorted(unreadable)),
        complete=not unreadable,
    )


# ---------------------------------------------------------------------------
# Rendering.

_DEFINITIONS = """\
A test fn is a `fn` whose attributes include one whose path ends in `test`
(`#[test]`, `#[tokio::test(...)]`), in a tracked `.rs` file. Its crate is the
`[package]` name of the nearest enclosing tracked `Cargo.toml`. Rows are in
crate-name order and are not ranked.

- **non-trivial lines**: test-fn body lines, comments blanked, that contain a
  letter or digit after stripping (so `}` and `});` are trivial).
- **duplicated lines**: non-trivial lines whose stripped text occurs 2+ times
  among the crate's test-fn lines, counting every occurrence; **redundant**
  counts occurrences beyond the first.
- **family members**: test fns in a same-file family of 3+, where the family
  key is the fn name minus its last `_segment`."""

_HEADER = (
    'crate', 'test files', 'test fns', 'non-trivial lines', 'duplicated lines',
    'duplicated share', 'redundant lines', 'family members', 'family share',
    'largest family',
)


def _cells(row: RustCrateRow) -> tuple[object, ...]:
    largest = row.largest_family
    return (
        row.crate, row.test_files, row.test_fns, row.non_trivial_lines, row.duplicated_lines,
        f'{row.duplicated_share:.1%}', row.redundant_lines, row.family_members,
        f'{row.family_share:.1%}',
        f'{largest[0]}: `{largest[1]}` `{largest[2]}_*`' if largest else '—',
    )


def _row_line(cells: Iterable[object]) -> str:
    return '| ' + ' | '.join(str(cell).replace('|', '\\|') for cell in cells) + ' |'


def render_markdown(census: RustDuplicationCensus) -> str:
    table = [_row_line(_HEADER), _row_line('---' for _ in _HEADER)]
    table.extend(_row_line(_cells(row)) for row in (*census.rows, census.totals))
    unreadable = '\n'.join(f'- `{path}`' for path in census.unreadable) or '- none'
    return (
        f'### Rust test duplication\n\n{_DEFINITIONS}\n\n' + '\n'.join(table)
        + f'\n\nUnreadable files ({len(census.unreadable)}; the census is '
        + ('complete' if census.complete else 'INCOMPLETE') + f'):\n\n{unreadable}\n'
    )
