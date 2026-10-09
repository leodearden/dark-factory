"""INV-12's exception register: every governed exception, its disposition, and the gaps.

``plans/inv12-exceptions-owned-or-ratified-prd.md``, decisions **D3**, **D5**
and **D14**, Contract sections "Ratification table" and "CLIs".  One register
document is measured from the tree and rendered two ways:

* **Ratification table** (D3) — ``docs/legibility/exception-ratifications.yaml``,
  the closed world of operator rulings.  This module is its only loader.
* **Static discovery** (D5) — every ``governed_exceptions(...)`` call evaluated
  at import, read from the AST of the tracked tree.  Its literals are bound
  through the vocabulary's own signature and constructors; the key set and its
  count rules stay with the runtime call at collection.
* **Inline suppressions** — read from ``scripts/inline_suppressions.py --json``
  in a subprocess, and passed through verbatim.
* **Closed world** — every ``Policy`` in a declaration, an inline marker or a
  ratified class row must name a row of the table.
* **Unregistered candidates** (D14) — report-only: module-level collections
  whose names match the census pattern and that no declaration governs.
* **Sweep age** — the owner sweep's run stamp.

**Exit codes.**  ``--report``: 0 clean; 1 closed-world or malformed-disposition
violations, one per line on stderr; 2 an instrument failure or a stale or
unreadable sweep stamp.  ``--json``: 0 whenever a document could be built, with
violations and the sweep's status as data; 2 for an instrument failure.  An
``ImportError`` is 2, never 1.

**Consumers.**  The owner sweep reads ``--json`` as its only data source; the
declaration tasks, the day-one rulings, the rulings' application and the
integration gate read ``--report``.
"""

from __future__ import annotations

import ast
import dataclasses
import datetime
import inspect
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared' / 'src'))

try:
    import source_measures
    import yaml
    from shared.governed_exceptions import (
        DECLARATION_FORMS,
        Debt,
        Disposition,
        MalformedDisposition,
        Policy,
        TaskRef,
        TicketRef,
        governed_exceptions,
    )
except ImportError as exc:
    print(
        f'exception_register.py: {exc} -- run it as '
        '`uv run --project shared python scripts/exception_register.py`',
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

#: The ratification table, relative to the checkout root.
TABLE_PATH = 'docs/legibility/exception-ratifications.yaml'

_ROW_KEYS = ('id', 'date', 'source', 'covers')


class InstrumentFailure(Exception):
    """The register could not take a measurement — exit 2, never a finding.

    Every message names the file, path or key at fault.
    """


@dataclasses.dataclass(frozen=True)
class Ratification:
    """One operator ruling: a row of the ratification table."""

    id: str
    date: datetime.date
    source: str
    covers: str


def _row_label(path: Path, number: int, row: object) -> str:
    """How a refusal names one row: the table, its 1-based position, and its id."""
    row_id = row.get('id') if isinstance(row, dict) else None
    named = f' (id {row_id!r})' if isinstance(row_id, str) else ''
    return f'{path}: row {number}{named}'


def _check_layout(path: Path, text: str) -> None:
    """Refuse a non-block layout, or a row whose first key is not ``id``.

    Read off the parser's own node model: the rulings task's delivered check
    greps ``^- id: inv12-``, which only a block-style row with ``id`` first
    satisfies.
    """
    root = yaml.compose(text, Loader=yaml.SafeLoader)
    if not isinstance(root, yaml.SequenceNode) or not root.value:
        return
    if root.flow_style:
        raise InstrumentFailure(
            f'{path}: rows must be written block-style (one `- id: ...` row per ruling), '
            'not as a flow-style `[...]` list'
        )
    for number, node in enumerate(root.value, start=1):
        if not isinstance(node, yaml.MappingNode):
            raise InstrumentFailure(f'{path}: row {number} is not a mapping')
        if node.flow_style:
            raise InstrumentFailure(
                f'{path}: row {number} is written flow-style `{{...}}`; write it block-style'
            )
        keys = [key.value for key, _ in node.value]
        if keys[:1] != ['id']:
            raise InstrumentFailure(
                f'{path}: row {number} must have `id` as its first key, not {keys[:1]}'
            )
        if len(keys) != len(set(keys)):
            raise InstrumentFailure(f'{path}: row {number} repeats a key: {keys}')


def _ratification(label: str, row: object) -> Ratification:
    """One row, checked field by field against the schema."""
    if not isinstance(row, dict):
        raise InstrumentFailure(f'{label} is not a mapping')
    keys = set(row)
    if keys != set(_ROW_KEYS):
        raise InstrumentFailure(
            f'{label} must have exactly the keys {list(_ROW_KEYS)}; '
            f'missing {sorted(set(_ROW_KEYS) - keys)}, extra {sorted(map(str, keys - set(_ROW_KEYS)))}'
        )
    try:
        Policy(row['id'])
    except MalformedDisposition as exc:
        raise InstrumentFailure(f'{label}: id {exc}') from exc
    if type(row['date']) is not datetime.date:
        raise InstrumentFailure(
            f'{label}: date={row["date"]!r} must be an unquoted ISO date (YYYY-MM-DD)'
        )
    for key in ('source', 'covers'):
        if not isinstance(row[key], str) or not row[key].strip():
            raise InstrumentFailure(f'{label}: {key}={row[key]!r} must be a non-blank string')
    return Ratification(
        id=row['id'], date=row['date'], source=row['source'], covers=row['covers']
    )


def load_ratifications(path: Path) -> tuple[Ratification, ...]:
    """The ratification table at *path*, in file order.

    Raises:
        InstrumentFailure: the file is absent or unreadable, is not YAML, or any
            row breaks the schema its own header comment states.
    """
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise InstrumentFailure(f'{path}: the ratification table could not be read -- {exc}') from exc
    try:
        document = yaml.safe_load(text)
        _check_layout(path, text)
    except yaml.YAMLError as exc:
        raise InstrumentFailure(f'{path}: the ratification table is not valid YAML -- {exc}') from exc
    if not isinstance(document, list):
        raise InstrumentFailure(
            f'{path}: the ratification table must be a YAML list of rows (an empty table is '
            f'written `[]`), not {type(document).__name__}'
        )
    rows: list[Ratification] = []
    seen: dict[str, int] = {}
    for number, row in enumerate(document, start=1):
        label = _row_label(path, number, row)
        ratification = _ratification(label, row)
        if ratification.id in seen:
            raise InstrumentFailure(
                f'{label}: duplicates the id of row {seen[ratification.id]}; ids are unique'
            )
        seen[ratification.id] = number
        rows.append(ratification)
    return tuple(rows)


# ---------------------------------------------------------------------------
# D5 — static discovery of governed-list declarations.

_FUNCTION_NAME = governed_exceptions.__name__
_CALL_TEXT = f'{_FUNCTION_NAME}('.encode()
_SIGNATURE = inspect.signature(governed_exceptions)
_LITERAL_CLASSES: Mapping[str, type] = MappingProxyType(
    {cls.__name__: cls for cls in (Debt, Policy, TaskRef, TicketRef)}
)


@dataclasses.dataclass(frozen=True)
class Violation:
    """One exit-1 finding, located at a site and carrying the forms that fix it."""

    path: str
    line: int
    subject: str
    reason: str
    forms: tuple[str, ...]

    def render(self) -> str:
        """The one line ``--report`` prints to stderr."""
        return (
            f'{self.path}:{self.line}: {self.subject}: {self.reason} '
            f'Accepted forms: {" | ".join(self.forms)}'
        )

    def record(self) -> dict[str, object]:
        """The violation as ``--json`` publishes it."""
        return {
            'path': self.path,
            'line': self.line,
            'subject': self.subject,
            'reason': self.reason,
            'forms': list(self.forms),
        }


@dataclasses.dataclass(frozen=True)
class Declaration:
    """One governed list, as its literals read statically (D5).

    The key set is deliberately absent: only the runtime call sees it.
    """

    path: str
    line: int
    list_id: str
    rule: str
    default: Disposition | None
    default_covers: int | None
    overrides: Mapping[str, Disposition]
    keys: ast.expr = dataclasses.field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'overrides', MappingProxyType(dict(self.overrides)))


@dataclasses.dataclass(frozen=True)
class Discovery:
    """Every declaration in a tree, the work it took, and the faults at sites."""

    declarations: tuple[Declaration, ...]
    files_enumerated: int
    files_parsed: int
    faults: tuple[Violation, ...]


@dataclasses.dataclass(frozen=True)
class Corpus:
    """The tracked Python of one checkout, read once per run.

    Attributes:
        files: Raw bytes by repo-relative path, for every tracked file still
            present in the worktree.
        enumerated: How many tracked ``.py`` files git listed.
    """

    files: Mapping[str, bytes]
    enumerated: int


def read_corpus(root: Path) -> Corpus:
    """Read every tracked ``.py`` file under *root*.

    A file the index lists but the worktree has deleted is passed over, as the
    inline scanner does: it holds nothing to govern.
    """
    try:
        paths = source_measures.tracked_python_files(root)
    except source_measures.MetricsError as exc:
        raise InstrumentFailure(f'{root}: the tracked files could not be listed -- {exc}') from exc
    files: dict[str, bytes] = {}
    for path in paths:
        try:
            files[path] = (root / path).read_bytes()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise InstrumentFailure(f'{path}: could not be read -- {exc}') from exc
    return Corpus(files=MappingProxyType(files), enumerated=len(paths))


def parse_tracked(path: str, raw: bytes) -> ast.Module:
    """One tracked file's AST, every failure an :class:`InstrumentFailure`."""
    try:
        source = raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise InstrumentFailure(f'{path}: is not UTF-8 -- {exc}') from exc
    try:
        return source_measures.parse_source(source, path=path)
    except source_measures.MetricsError as exc:
        raise InstrumentFailure(str(exc)) from exc


class _ImportTimeCalls(ast.NodeVisitor):
    """Collect the calls to the declaring function that run at import.

    A call inside a ``def``, ``async def`` or ``lambda`` runs only when that code
    does, so it is a use of the function and not a declaration.
    """

    def __init__(self) -> None:
        self.calls: list[ast.Call] = []
        self._depth = 0

    def _nested(self, node: ast.AST) -> None:
        self._depth += 1
        self.generic_visit(node)
        self._depth -= 1

    visit_FunctionDef = visit_AsyncFunctionDef = visit_Lambda = _nested

    def visit_Call(self, node: ast.Call) -> None:
        if self._depth == 0 and _terminal_name(node.func) == _FUNCTION_NAME:
            self.calls.append(node)
        self.generic_visit(node)


def _terminal_name(node: ast.expr) -> str | None:
    """``f`` for ``f`` and for ``a.b.f``; ``None`` for anything else."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _bind(site: str, call: ast.Call) -> Mapping[str, ast.expr]:
    """*call*'s argument nodes by parameter name, bound by the function's own signature."""
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            raise InstrumentFailure(
                f'{site}: {_FUNCTION_NAME} argument `{ast.unparse(arg)}` is unpacked, so the '
                'static register cannot bind it; pass each argument by name or position'
            )
    keywords: dict[str, ast.expr] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise InstrumentFailure(
                f'{site}: {_FUNCTION_NAME} argument `**{ast.unparse(keyword.value)}` is unpacked, '
                'so the static register cannot bind it; pass each argument by name'
            )
        if keyword.arg not in _SIGNATURE.parameters:
            raise InstrumentFailure(
                f'{site}: {_FUNCTION_NAME} has no parameter `{keyword.arg}`'
            )
        keywords[keyword.arg] = keyword.value
    try:
        bound = _SIGNATURE.bind_partial(*call.args, **keywords)
    except TypeError as exc:
        raise InstrumentFailure(f'{site}: {_FUNCTION_NAME}(...) does not bind -- {exc}') from exc
    missing = [
        f'`{name}`'
        for name, parameter in _SIGNATURE.parameters.items()
        if parameter.default is inspect.Parameter.empty and name not in bound.arguments
    ]
    if missing:
        raise InstrumentFailure(
            f'{site}: {_FUNCTION_NAME} is missing its required argument(s) {", ".join(missing)}'
        )
    return bound.arguments


def _not_literal(site: str, parameter: str, node: ast.AST, expected: str) -> InstrumentFailure:
    return InstrumentFailure(
        f'{site}: {_FUNCTION_NAME} argument `{parameter}` is `{ast.unparse(node)}`, but the '
        f'static register reads every argument except `keys` from source (PRD D5), so it '
        f'must be {expected}'
    )


def _str_literal(site: str, parameter: str, node: ast.expr) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    raise _not_literal(site, parameter, node, 'a str literal')


def _count_literal(site: str, node: ast.expr | None) -> int | None:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise _not_literal(site, 'default_covers', node, 'an int literal or None') from exc
    if value is not None and type(value) is not int:
        raise _not_literal(site, 'default_covers', node, 'an int literal or None')
    return value


def _construct(site: str, parameter: str, node: ast.expr) -> object:
    """*node* built through the vocabulary's own classes, or a plain literal."""
    if isinstance(node, ast.Call):
        cls = _LITERAL_CLASSES.get(_terminal_name(node.func) or '')
        if cls is None:
            raise _not_literal(site, parameter, node, f'one of {", ".join(DECLARATION_FORMS)}')
        args = [_construct(site, parameter, arg) for arg in node.args]
        kwargs = {
            keyword.arg: _construct(site, parameter, keyword.value)
            for keyword in node.keywords
            if keyword.arg is not None
        }
        if len(kwargs) != len(node.keywords):
            raise _not_literal(site, parameter, node, f'one of {", ".join(DECLARATION_FORMS)}')
        try:
            return cls(*args, **kwargs)
        except TypeError as exc:
            raise InstrumentFailure(
                f'{site}: {_FUNCTION_NAME} argument `{parameter}` holds `{ast.unparse(node)}`, '
                f'which does not construct -- {exc}'
            ) from exc
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise _not_literal(site, parameter, node, 'a literal') from exc


def _disposition(
    path: str, site: str, parameter: str, node: ast.expr, faults: list[Violation], subject: str
) -> Disposition | None:
    """One disposition literal; ``None`` when it is malformed, after recording why."""
    try:
        value = _construct(site, parameter, node)
    except MalformedDisposition as exc:
        faults.append(
            Violation(
                path=path,
                line=node.lineno,
                subject=subject,
                reason=f'malformed disposition `{ast.unparse(node)}` -- {exc}',
                forms=DECLARATION_FORMS,
            )
        )
        return None
    if not isinstance(value, Debt | Policy):
        raise _not_literal(site, parameter, node, f'one of {", ".join(DECLARATION_FORMS)}')
    return value


def _overrides(
    path: str, site: str, node: ast.expr | None, faults: list[Violation], subject: str
) -> dict[str, Disposition | None]:
    if node is None:
        return {}
    if not isinstance(node, ast.Dict):
        raise _not_literal(site, 'dispositions', node, 'a dict literal')
    overrides: dict[str, Disposition | None] = {}
    for key, value in zip(node.keys, node.values, strict=True):
        if key is None:
            raise _not_literal(site, 'dispositions', node, 'a dict literal with no `**` spread')
        overrides[_str_literal(site, 'dispositions', key)] = _disposition(
            path, site, 'dispositions', value, faults, subject
        )
    return overrides


def _declaration(path: str, call: ast.Call, faults: list[Violation]) -> Declaration | None:
    """*call* read into a :class:`Declaration`, or ``None`` when a literal is malformed."""
    site = f'{path}:{call.lineno}'
    arguments = _bind(site, call)
    list_id = _str_literal(site, 'list_id', arguments['list_id'])
    rule = _str_literal(site, 'rule', arguments['rule'])
    default_covers = _count_literal(site, arguments.get('default_covers'))
    found = len(faults)
    default_node = arguments.get('default')
    default = (
        None
        if default_node is None or _is_none(default_node)
        else _disposition(path, site, 'default', default_node, faults, list_id)
    )
    overrides = _overrides(path, site, arguments.get('dispositions'), faults, list_id)
    if len(faults) > found:
        return None
    return Declaration(
        path=path,
        line=call.lineno,
        list_id=list_id,
        rule=rule,
        default=default,
        default_covers=default_covers,
        overrides={key: value for key, value in overrides.items() if value is not None},
        keys=arguments['keys'],
    )


def _is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _require_unique(declarations: list[Declaration]) -> None:
    sites: dict[str, list[str]] = {}
    for declaration in declarations:
        sites.setdefault(declaration.list_id, []).append(f'{declaration.path}:{declaration.line}')
    for list_id, where in sorted(sites.items()):
        if len(where) > 1:
            raise InstrumentFailure(
                f'list_id {list_id!r} is declared more than once, at {", ".join(where)}; a '
                'list_id is globally unique'
            )


def declarations_in(corpus: Corpus) -> Discovery:
    """Every import-time declaration in *corpus*, sorted by ``list_id``."""
    declarations: list[Declaration] = []
    faults: list[Violation] = []
    parsed = 0
    for path, raw in sorted(corpus.files.items()):
        if _CALL_TEXT not in raw:
            continue
        parsed += 1
        collector = _ImportTimeCalls()
        collector.visit(parse_tracked(path, raw))
        for call in collector.calls:
            declaration = _declaration(path, call, faults)
            if declaration is not None:
                declarations.append(declaration)
    _require_unique(declarations)
    return Discovery(
        declarations=tuple(sorted(declarations, key=lambda declaration: declaration.list_id)),
        files_enumerated=corpus.enumerated,
        files_parsed=parsed,
        faults=tuple(faults),
    )


def discover_declarations(root: Path) -> Discovery:
    """Every import-time declaration in *root*'s tracked Python (D5)."""
    return declarations_in(read_corpus(root))
