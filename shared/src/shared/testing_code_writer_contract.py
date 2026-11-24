"""Instruments for plans/live-task-facts-at-recall-prd.md C5: memory written by code names no task status, and every memory-write site is listed.

Pure stdlib plus shared leaves and pytest-free, like every ``shared.testing_*``
module, and registered as such in ``shared/tests/test_pure_stdlib_leaves.py``.
"""

from __future__ import annotations

import ast
import enum
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from shared.task_statuses import TaskStatus

__all__ = [
    'MCP_TOOL_CALLERS',
    'TESTED_DISPOSITIONS',
    'WRITE_TOOLS',
    'CensusKey',
    'CensusRow',
    'CensusVerdict',
    'Disposition',
    'RowProblem',
    'WriteSite',
    'census_failure_message',
    'census_row_problems',
    'judge_census',
    'missing_tests',
    'task_status_words_in',
    'write_sites',
]

WRITE_TOOLS: frozenset[str] = frozenset({'add_memory', 'add_episode', 'add_system_record', 'update_memory'})
MCP_TOOL_CALLERS: frozenset[str] = frozenset({'post_mcp_tool_call', 'call_mcp_tool', 'call_tool', '_call_tool'})

_STATUS_WORD = re.compile(
    r'(?<![\w-])('
    + '|'.join(re.escape(s.value) for s in sorted(TaskStatus, key=len, reverse=True))
    + r')(?![\w-])',
    re.IGNORECASE,
)

CensusKey = tuple[str, str, str]


def task_status_words_in(text: str) -> tuple[str, ...]:
    """Every task status value in ``text`` as a whole hyphen-aware word, lowercased, in order."""
    return tuple(match.group(1).lower() for match in _STATUS_WORD.finditer(text))


@dataclass(frozen=True)
class WriteSite:
    """One call that writes memory: the file, its enclosing qualname, and the write tool."""

    path: str
    function: str
    callee: str

    @property
    def key(self) -> CensusKey:
        return (self.path, self.function, self.callee)


def _callee_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _module_str_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        else:
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            for target in targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = value.value
    return constants


class _WriteSiteVisitor(ast.NodeVisitor):
    def __init__(self, relpath: str, constants: Mapping[str, str]) -> None:
        self._relpath = relpath
        self._constants = constants
        self._scope: list[str] = []
        self.sites: list[tuple[tuple[int, int], WriteSite]] = []

    def _visit_scope(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_ClassDef = _visit_scope

    def _tool_named_by(self, arg: ast.expr) -> str | None:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            name = arg.value
        elif isinstance(arg, ast.Name):
            name = self._constants.get(arg.id, '')
        else:
            return None
        return name if name in WRITE_TOOLS else None

    def _written_tool(self, node: ast.Call) -> str | None:
        callee = _callee_name(node.func)
        if callee in WRITE_TOOLS:
            return callee
        if callee in MCP_TOOL_CALLERS:
            for arg in [*node.args, *(kw.value for kw in node.keywords)]:
                tool = self._tool_named_by(arg)
                if tool is not None:
                    return tool
        return None

    def visit_Call(self, node: ast.Call) -> None:
        tool = self._written_tool(node)
        if tool is not None:
            function = '.'.join(self._scope) or '<module>'
            site = WriteSite(path=self._relpath, function=function, callee=tool)
            self.sites.append(((node.lineno, node.col_offset), site))
        self.generic_visit(node)


def write_sites(tree: ast.Module, relpath: str) -> tuple[WriteSite, ...]:
    """Every call in ``tree`` that writes memory, in source order.

    A site is a call to a :data:`WRITE_TOOLS` name on any receiver, or a call
    to a :data:`MCP_TOOL_CALLERS` name passing a write-tool name as a string
    literal or a module-level string constant. The detector is a floor: a
    write behind any other indirection is not seen.
    """
    visitor = _WriteSiteVisitor(relpath, _module_str_constants(tree))
    visitor.visit(tree)
    return tuple(site for _, site in sorted(visitor.sites, key=lambda pair: pair[0]))


@dataclass(frozen=True)
class CensusVerdict:
    """Keys found more often than listed (``unlisted``) and listed more often than found (``stale``)."""

    unlisted: tuple[CensusKey, ...]
    stale: tuple[CensusKey, ...]

    @property
    def agrees(self) -> bool:
        return not (self.unlisted or self.stale)


def judge_census(found: Iterable[WriteSite], listed: Mapping[CensusKey, int]) -> CensusVerdict:
    """Compare the write sites found with the census's expected count per key."""
    found_counts = Counter(site.key for site in found)
    keys = sorted(set(found_counts) | set(listed))
    return CensusVerdict(
        unlisted=tuple(key for key in keys if found_counts[key] > listed.get(key, 0)),
        stale=tuple(key for key in keys if listed.get(key, 0) > found_counts[key]),
    )


def census_failure_message(verdict: CensusVerdict, src_root: Path) -> str:
    """What a census test over ``src_root`` reports for ``verdict``: its unlisted sites, then its stale rows; '' when it agrees."""
    lines: list[str] = []
    if verdict.unlisted:
        lines.append(
            f'write sites under {src_root} that CENSUS does not list: {list(verdict.unlisted)}. '
            'List each with its count, disposition and reason; a CODE_WRITER or MODEL_PROSE row '
            "also names the behaviour test, in the writer's own module test file, that holds it to C5."
        )
    if verdict.stale:
        lines.append(f'CENSUS rows with fewer sites than listed: {list(verdict.stale)}')
    return '\n'.join(lines)


def _defines_test(tree: ast.Module, chain: list[str]) -> bool:
    body: list[ast.stmt] = tree.body
    *classes, function = chain
    for name in classes:
        cls = next((s for s in body if isinstance(s, ast.ClassDef) and s.name == name), None)
        if cls is None:
            return False
        body = cls.body
    return any(
        isinstance(s, ast.FunctionDef | ast.AsyncFunctionDef) and s.name == function
        for s in body
    )


class _TestModules:
    """The test files under one root, each read and parsed at most once."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._parsed: dict[str, ast.Module | None] = {}

    def _module(self, file_part: str) -> ast.Module | None:
        if file_part not in self._parsed:
            path = self.root / file_part
            self._parsed[file_part] = ast.parse(path.read_text(encoding='utf-8')) if path.is_file() else None
        return self._parsed[file_part]

    def _defines(self, node_id: str) -> bool:
        file_part, *chain = node_id.split('::')
        tree = self._module(file_part) if chain else None
        if tree is None:
            return False
        chain[-1] = chain[-1].split('[', 1)[0]
        return _defines_test(tree, chain)

    def missing(self, node_ids: Iterable[str]) -> tuple[str, ...]:
        return tuple(node_id for node_id in node_ids if not self._defines(node_id))


def missing_tests(node_ids: Iterable[str], tests_root: Path) -> tuple[str, ...]:
    """The pytest node ids, relative to ``tests_root``, whose file or test function does not exist."""
    return _TestModules(tests_root).missing(node_ids)


class Disposition(enum.StrEnum):
    """What a write site writes, which decides what its census row must carry."""

    CODE_WRITER = 'code_writer'
    MODEL_PROSE = 'model_prose'
    RELAY = 'relay'
    METADATA_ONLY = 'metadata_only'
    STORE_LAYER = 'store_layer'
    RETIRING = 'retiring'


#: Dispositions whose text this code writes or forwards, so a behaviour test must hold the site to C5.
TESTED_DISPOSITIONS: frozenset[Disposition] = frozenset({Disposition.CODE_WRITER, Disposition.MODEL_PROSE})

_NAMES_A_TASK = re.compile(r'\btask \d+\b')


@dataclass(frozen=True)
class CensusRow:
    """One census entry: how many sites the key has, what they write, why, and the tests that hold them to C5."""

    count: int
    disposition: Disposition
    reason: str
    tests: tuple[str, ...] = ()


@dataclass(frozen=True)
class RowProblem:
    """A census row that breaks a census rule."""

    key: CensusKey
    issue: str

    def __str__(self) -> str:
        return f'{self.key}: {self.issue}'


def _row_issues(row: CensusRow, tests: _TestModules) -> list[str]:
    issues: list[str] = []
    if row.disposition in TESTED_DISPOSITIONS:
        if not row.tests:
            issues.append(f'a {row.disposition} row names no test')
        if missing := tests.missing(row.tests):
            issues.append(f'names tests that do not exist under {tests.root}: {list(missing)}')
    if row.disposition is Disposition.RETIRING and not _NAMES_A_TASK.search(row.reason):
        issues.append("a retiring row's reason names no retiring 'task N'")
    return issues


def census_row_problems(census: Mapping[CensusKey, CensusRow], tests_root: Path) -> tuple[RowProblem, ...]:
    """Every rule a row breaks: a tested disposition names existing tests, and a retiring row names its retiring task."""
    tests = _TestModules(tests_root)
    return tuple(
        RowProblem(key, issue)
        for key, row in census.items()
        for issue in _row_issues(row, tests)
    )
