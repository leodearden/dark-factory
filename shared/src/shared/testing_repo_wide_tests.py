"""Which test modules read the repository beyond their own imports, and whether each is a guard-family member.

A REPO-WIDE test module is one that walks the tree, shells out to git, anchors
on the repo root, or names a committed doc or config file.  Such a module's
outcome can change on a diff that touches nothing it imports, so task-role
verify must run it on every production diff in its leg.  That is what
membership of the whole-tree guard family buys
(``shared/src/shared/whole_tree_guard_family.py``).  The census here fails every
repo-wide test module that is not a member.

PROVENANCE.  The signature is the "sig09" one from the test-selection seat M2
of plans/portfolio-review-2026-11-01 (§2.1), adapted from a text regex to the
AST, so a mention in prose does not count: a comment, a docstring, an assert
message, or the value of a :data:`PROSE_KEYWORDS` keyword.

THE DETECTOR IS A FLOOR.  It sees only what a module spells itself: a tree read
behind a helper imported from elsewhere is not seen.

orchestrator/tests keeps its own narrower detector,
``orchestrator/tests/test_whole_tree_scan_timeout_guard.py::_scans_whole_tree_py``.
Its whole-tree glob rule is a subset of :attr:`RepoWideSignal.TREE_WALK`'s.

Pure stdlib plus shared leaves and pytest-free, like every ``shared.testing_*``
module, and registered as such in ``shared/tests/test_pure_stdlib_leaves.py``.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import shared.whole_tree_guard_family as whole_tree_guard_family
from shared.testing_timeout_markers import TreeScan, scan_python_tree
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TIMEOUT_NAME, is_whole_tree_guard_ast

__all__ = [
    'COMMITTED_CONFIG_NAMES',
    'COMMITTED_DOC_NAMES',
    'PROSE_KEYWORDS',
    'RepoWideFloors',
    'RepoWideModule',
    'RepoWideSignal',
    'RepoWideVerdict',
    'judge_repo_wide_family',
    'repo_wide_signals',
]


class RepoWideSignal(StrEnum):
    """One way a test module reaches past its imports into the repository."""

    TREE_WALK = 'tree-walk'
    GIT_SUBPROCESS = 'git-subprocess'
    REPO_ANCHOR = 'repo-anchor'
    COMMITTED_DOC_NAME = 'committed-doc-name'
    COMMITTED_CONFIG_NAME = 'committed-config-name'


#: Substrings of a string constant that name a committed document.
COMMITTED_DOC_NAMES = (
    'CONTRIBUTING.md', 'CLAUDE.md', 'ARCHITECTURE.md', 'OPERATIONS.md', 'README',
    'codebook', 'design-invariants', 'code-quality',
)

#: Substrings of a string constant that name a committed config or unit file.
COMMITTED_CONFIG_NAMES = (
    'dark-factory-orchestrator.yaml', 'defaults.yaml', 'orchestrator.yaml',
    'capability-manifest', '.service', 'systemd',
)

#: Keywords whose value is a message to a reader, like ``pytest.raises(match=...)``.
PROSE_KEYWORDS = ('match', 'msg', 'reason')

_REPO_ANCHOR_NAME = re.compile(r'_*(PROJECT_ROOT|REPO_ROOT)')


def _vocabulary_pattern(names: tuple[str, ...]) -> re.Pattern[str]:
    """Any of *names* as a substring; a name opening with '.' is an extension, so it must also end the word.

    Without that, ``.service`` would match every ``pkg.services.module`` path.
    """
    return re.compile('|'.join(
        re.escape(name) + (r'(?!\w)' if name.startswith('.') else '') for name in names
    ))


_COMMITTED_DOC_NAME = _vocabulary_pattern(COMMITTED_DOC_NAMES)
_COMMITTED_CONFIG_NAME = _vocabulary_pattern(COMMITTED_CONFIG_NAMES)
_GIT_LS_FILES = 'ls-files'


def _docstring(owner: ast.AST) -> ast.Constant | None:
    if not isinstance(owner, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        return None
    first = owner.body[0] if owner.body else None
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        return first.value if isinstance(first.value.value, str) else None
    return None


def _prose_roots(tree: ast.Module) -> Iterator[ast.AST]:
    for node in ast.walk(tree):
        docstring = _docstring(node)
        if docstring is not None:
            yield docstring
        if isinstance(node, ast.Assert) and node.msg is not None:
            yield node.msg
        if isinstance(node, ast.keyword) and node.arg in PROSE_KEYWORDS:
            yield node.value


def _prose_nodes(tree: ast.Module) -> set[int]:
    return {id(node) for root in _prose_roots(tree) for node in ast.walk(root)}


def _is_str_constant(node: ast.AST, value: str) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


def _is_whole_tree_glob(pattern: ast.expr) -> bool:
    if not (isinstance(pattern, ast.Constant) and isinstance(pattern.value, str)):
        return False
    text = pattern.value
    return '**' in text or ('*' in text and text.endswith('.py'))


def _leads_with_git(nodes: list[ast.expr]) -> bool:
    return bool(nodes) and _is_str_constant(nodes[0], 'git')


def _is_tree_walk_call(node: ast.Call) -> bool:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr == 'glob':
        return bool(node.args) and _is_whole_tree_glob(node.args[0])
    is_os_walk = func.attr == 'walk' and isinstance(func.value, ast.Name) and func.value.id == 'os'
    return func.attr == 'rglob' or is_os_walk


def _string_signals(text: str) -> Iterator[RepoWideSignal]:
    if text == _GIT_LS_FILES:
        yield RepoWideSignal.TREE_WALK
    if _COMMITTED_DOC_NAME.search(text):
        yield RepoWideSignal.COMMITTED_DOC_NAME
    if _COMMITTED_CONFIG_NAME.search(text):
        yield RepoWideSignal.COMMITTED_CONFIG_NAME


def _identifier(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _node_signals(node: ast.AST) -> Iterator[RepoWideSignal]:
    if isinstance(node, ast.Call):
        if _is_tree_walk_call(node):
            yield RepoWideSignal.TREE_WALK
        if _leads_with_git(node.args):
            yield RepoWideSignal.GIT_SUBPROCESS
    elif isinstance(node, ast.List | ast.Tuple) and _leads_with_git(node.elts):
        yield RepoWideSignal.GIT_SUBPROCESS
    elif (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == 'parents'
    ):
        yield RepoWideSignal.REPO_ANCHOR
    identifier = _identifier(node)
    if identifier is not None and _REPO_ANCHOR_NAME.fullmatch(identifier):
        yield RepoWideSignal.REPO_ANCHOR


def repo_wide_signals(tree: ast.Module) -> frozenset[RepoWideSignal]:
    """Every :class:`RepoWideSignal` the module *tree* spells, strings in prose excluded."""
    prose = _prose_nodes(tree)
    signals: set[RepoWideSignal] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in prose:
                signals.update(_string_signals(node.value))
            continue
        signals.update(_node_signals(node))
    return frozenset(signals)


class RepoWideModule(NamedTuple):
    """One collectable repo-wide test module: its path relative to the scanned root, its signals, its membership."""

    module: str
    signals: frozenset[RepoWideSignal]
    is_member: bool


@dataclass(frozen=True)
class RepoWideFloors:
    """One package's anti-vacuity floors: set BELOW its measured counts, never equal to them.

    ``test_files`` catches a sweep that stopped reading files;
    ``repo_wide_modules`` catches a detector that stopped matching.
    """

    test_files: int
    repo_wide_modules: int


@dataclass(frozen=True)
class RepoWideVerdict:
    """One package's census, judged once by :func:`judge_repo_wide_family`.

    ``examined`` and ``unreadable`` are the sweep's own counters, as
    ``shared.testing_timeout_markers.TreeScan`` defines them.  Each
    ``*_failure`` is the text its check prints, or None when it passes.
    """

    examined: int
    unreadable: tuple[str, ...]
    repo_wide: tuple[RepoWideModule, ...]
    offenders: tuple[RepoWideModule, ...]
    offender_failure: str | None
    census_failure: str | None


def _is_collectable_test_module(module: str) -> bool:
    name = PurePosixPath(module).name
    return name.startswith('test_') or name.endswith('_test.py')


def _repo_wide_module(module: str, tree: ast.Module) -> Iterator[RepoWideModule]:
    if not _is_collectable_test_module(module):
        return
    signals = repo_wide_signals(tree)
    if signals:
        yield RepoWideModule(module, signals, is_whole_tree_guard_ast(tree))


def judge_repo_wide_family(root: Path, floors: RepoWideFloors) -> RepoWideVerdict:
    """Judge every collectable test module under *root*: each repo-wide one must be a guard-family member.

    One read and parse per file, through
    ``shared.testing_timeout_markers.scan_python_tree``.  The check that asserts
    an ABSENCE -- no offender -- fails with the census text when the sweep
    read fewer files than the floor, because the absence would then prove
    nothing.
    """
    scan = scan_python_tree(root, _repo_wide_module)
    offenders = tuple(module for module in scan.items if not module.is_member)
    census_failure = _census_failure(scan, floors)
    unread = census_failure if scan.examined < floors.test_files else None
    return RepoWideVerdict(
        examined=scan.examined,
        unreadable=scan.unreadable,
        repo_wide=scan.items,
        offenders=offenders,
        offender_failure=unread or _offender_failure(scan.root, offenders),
        census_failure=census_failure,
    )


def _census_failure(scan: TreeScan[RepoWideModule], floors: RepoWideFloors) -> str | None:
    if scan.examined >= floors.test_files and len(scan.items) >= floors.repo_wide_modules:
        return None
    return (
        f'the sweep under {scan.root} examined {scan.examined} .py files '
        f'(floor {floors.test_files}; {len(scan.unreadable)} unreadable: '
        f'{sorted(scan.unreadable)}) and found {len(scan.items)} repo-wide test '
        f'module(s) (floor {floors.repo_wide_modules}). Below either floor the '
        'sweep or the detector is broken, and the guard would pass vacuously '
        'rather than because every repo-wide module is marked. Check the '
        'detector against shared/tests/test_testing_repo_wide_tests.py.'
    )


def _offender_failure(root: Path, offenders: tuple[RepoWideModule, ...]) -> str | None:
    if not offenders:
        return None
    listing = '\n  '.join(
        f'{offender.module} [{", ".join(sorted(offender.signals))}]' for offender in offenders
    )
    return (
        f'{len(offenders)} repo-wide test module(s) under {root} are not members '
        'of the whole-tree guard family. Each spells a sign, bracketed below, of '
        'reading the repository beyond its own imports, so task-role verify must '
        'run it on every production diff in its leg, and it runs only the family. '
        'Add, at module level:\n\n'
        f'    from {whole_tree_guard_family.__name__} import {WHOLE_TREE_SCAN_TIMEOUT_NAME}\n'
        f'    pytestmark = pytest.mark.timeout({WHOLE_TREE_SCAN_TIMEOUT_NAME})\n\n'
        'or append that timeout element to an existing pytestmark list. Spell it '
        'through the name: membership is decided by the name, so a literal at '
        'the same value does not admit. A module-level timeout already there is '
        'replaced; a deliberate tight bound moves onto its tests as decorators, '
        'which win over the module mark. Rationale: '
        'shared/src/shared/whole_tree_guard_family.py.'
        f'\n\nOffending modules [signals]:\n  {listing}'
    )
