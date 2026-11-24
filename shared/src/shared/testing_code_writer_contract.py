"""Instruments for plans/live-task-facts-at-recall-prd.md C5: memory written by code names no task status, and every memory-write site is listed.

Pure stdlib plus shared leaves and pytest-free, like every ``shared.testing_*``
module, and registered as such in ``shared/tests/test_pure_stdlib_leaves.py``.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from shared.task_statuses import TaskStatus

__all__ = [
    'MCP_TOOL_CALLERS',
    'WRITE_TOOLS',
    'CensusKey',
    'CensusVerdict',
    'WriteSite',
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
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = stmt.value.value
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


def judge_census(found: Iterable[WriteSite], listed: Mapping[CensusKey, int]) -> CensusVerdict:
    """Compare the write sites found with the census's expected count per key."""
    found_counts = Counter(site.key for site in found)
    keys = sorted(set(found_counts) | set(listed))
    return CensusVerdict(
        unlisted=tuple(key for key in keys if found_counts[key] > listed.get(key, 0)),
        stale=tuple(key for key in keys if listed.get(key, 0) > found_counts[key]),
    )


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


def missing_tests(node_ids: Iterable[str], tests_root: Path) -> tuple[str, ...]:
    """The pytest node ids, relative to ``tests_root``, whose file or test function does not exist."""
    missing: list[str] = []
    for node_id in node_ids:
        file_part, *chain = node_id.split('::')
        path = tests_root / file_part
        if not chain or not path.is_file():
            missing.append(node_id)
            continue
        chain[-1] = chain[-1].split('[', 1)[0]
        if not _defines_test(ast.parse(path.read_text(encoding='utf-8')), chain):
            missing.append(node_id)
    return tuple(missing)
