"""Which host config fields can decide a merge verdict — a census-guarded table.

WHAT IS MEASURED.  Both merge-verify legs run one bundle,
``orchestrator/src/orchestrator/verify_runner.py::LocalRunner.run_merge_verify``.
The local leg runs it on the dispatcher's own ``OrchestratorConfig``.  The remote
leg runs it on the REMOTE host's config, after
``verify_runner.py::run_merge_verify_on_worktree`` has applied the dispatcher's
``MergeVerifySpec`` onto it.  Every config name the bundle reads is a place where
the two legs can disagree.  The census below walks the bundle's static call
graph and lists every ``OrchestratorConfig`` field, public property and public
method it names.  It over-approximates on purpose: a ``ModuleConfig`` attribute
with the same name counts too.

THE THREE CLASSES.

- ``SPEC_CARRIED``: the dispatcher's value rides the spec and wins on the
  remote leg.  Each such row is checked by a behavioural probe below, which goes
  through ``run_merge_verify_on_worktree`` and the wire codec.
- ``MERGE_NEUTRAL``: the host supplies the value, but no value of it can turn a
  failing verify into a passing one.  Scheduling, fan-out caps and timeouts that
  fail closed are in this class.
- ``HOST_DECIDING``: the host's value can decide the verdict.  This is a latent
  false green.  Each row names the follow-up work that owns closing it.

CLASSIFYING A NEW READ.  When the census reports a name the table lacks, add a
row to ``MERGE_PATH_CONFIG_READS``.  If the host's value can decide the verdict,
classify it ``HOST_DECIDING``, file a narrow follow-up task for it, and put the
filing's ``escalation_id`` key (or the owning task id) in ``follow_ups``.  Do not
fix the hole in the same change that classifies it.  When a read disappears,
delete its row: a stale row fails as loudly as a missing one.
"""
from __future__ import annotations

import ast
import functools
from collections import defaultdict, deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import FunctionType, MappingProxyType

from orchestrator import verify, verify_runner
from orchestrator.config import OrchestratorConfig
from orchestrator.merge_lane import gates


class Fidelity(Enum):
    SPEC_CARRIED = 'spec_carried'
    MERGE_NEUTRAL = 'merge_neutral'
    HOST_DECIDING = 'host_deciding'


@dataclass(frozen=True)
class Classified:
    fidelity: Fidelity
    why: str
    follow_ups: tuple[str, ...] = ()


MERGE_PATH_CONFIG_READS: Mapping[str, Classified] = MappingProxyType({})

UNREAD_CANDIDATES: Mapping[str, str] = MappingProxyType({})


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).resolve().parent.parent / 'src'
_PACKAGE = 'orchestrator'

# The bundle both legs run, plus the callables run_merge_verify_on_worktree wires into it.
_ENTRY_POINTS = (
    verify_runner.LocalRunner.run_merge_verify,
    verify.run_scoped_verification,
    verify.run_verification,
    gates._run_unscoped_typechecks,
)

CENSUS_NAMES: frozenset[str] = frozenset(OrchestratorConfig.model_fields) | frozenset(
    name
    for name, member in vars(OrchestratorConfig).items()
    if isinstance(member, property | FunctionType) and not name.startswith(('_', 'model_'))
)

_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
_FunctionKey = tuple[str, str]
_ImportTarget = tuple[str, str | None]


@dataclass(frozen=True)
class _ModuleIndex:
    defs: Mapping[str, _FunctionNode]
    imports: Mapping[str, frozenset[_ImportTarget]]


@dataclass(frozen=True)
class _Census:
    reads: Mapping[str, tuple[str, ...]]
    visited_modules: frozenset[str]


def _module_path(module: str) -> Path | None:
    base = _SRC_DIR.joinpath(*module.split('.'))
    for candidate in (base.with_suffix('.py'), base / '__init__.py'):
        if candidate.is_file():
            return candidate
    return None


def _in_package(module: str) -> bool:
    return module == _PACKAGE or module.startswith(f'{_PACKAGE}.')


def _import_source(node: ast.ImportFrom, package: str) -> str:
    if node.level == 0:
        return node.module or ''
    anchor = package.split('.')[: len(package.split('.')) - (node.level - 1)]
    return '.'.join([*anchor, *([node.module] if node.module else [])])


def _import_targets(tree: ast.Module, package: str) -> Iterator[tuple[str, _ImportTarget]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname and _in_package(alias.name):
                    yield alias.asname, (alias.name, None)
        elif isinstance(node, ast.ImportFrom):
            source = _import_source(node, package)
            if not _in_package(source):
                continue
            for alias in node.names:
                submodule = f'{source}.{alias.name}'
                target = (submodule, None) if _module_path(submodule) else (source, alias.name)
                yield alias.asname or alias.name, target


@functools.cache
def _index(module: str) -> _ModuleIndex:
    path = _module_path(module)
    if path is None:
        return _ModuleIndex(defs=MappingProxyType({}), imports=MappingProxyType({}))
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    defs: dict[str, _FunctionNode] = {}
    for node in tree.body:
        if isinstance(node, _FunctionNode):
            defs[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, _FunctionNode):
                    defs[f'{node.name}.{member.name}'] = member
    package = module if path.name == '__init__.py' else module.rpartition('.')[0]
    imports: defaultdict[str, set[_ImportTarget]] = defaultdict(set)
    for local_name, target in _import_targets(tree, package):
        imports[local_name].add(target)
    return _ModuleIndex(
        defs=MappingProxyType(defs),
        imports=MappingProxyType({name: frozenset(t) for name, t in imports.items()}),
    )


def _resolve(module: str, attr: str, seen: frozenset[_FunctionKey] = frozenset()) -> Iterator[_FunctionKey]:
    index = _index(module)
    if attr in index.defs:
        yield module, attr
        return
    for target_module, target_attr in index.imports.get(attr, ()):
        if target_attr is not None and (target_module, target_attr) not in seen:
            yield from _resolve(target_module, target_attr, seen | {(module, attr)})


def _callees(module: str, qualname: str, node: _FunctionNode) -> Iterator[_FunctionKey]:
    index = _index(module)
    enclosing_class = qualname.rpartition('.')[0]
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            if sub.id in index.defs:
                yield module, sub.id
            for target_module, attr in index.imports.get(sub.id, ()):
                if attr is not None:
                    yield from _resolve(target_module, attr)
        elif isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
            owner = sub.value.id
            method = f'{enclosing_class}.{sub.attr}'
            if owner in ('self', 'cls') and enclosing_class and method in index.defs:
                yield module, method
            for target_module, attr in index.imports.get(owner, ()):
                if attr is None:
                    yield from _resolve(target_module, sub.attr)


def _config_names_read(node: _FunctionNode) -> Iterator[str]:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and sub.attr in CENSUS_NAMES:
            yield sub.attr
        elif (
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == 'getattr'
            and len(sub.args) >= 2
            and isinstance(sub.args[1], ast.Constant)
            and sub.args[1].value in CENSUS_NAMES
        ):
            yield sub.args[1].value


@functools.cache
def _merge_path_census() -> _Census:
    entries = [(fn.__module__, fn.__qualname__) for fn in _ENTRY_POINTS]
    for module, qualname in entries:
        assert qualname in _index(module).defs, f'entry point {module}::{qualname} not found under {_SRC_DIR}'
    visited: set[_FunctionKey] = set()
    queue = deque(entries)
    sites: defaultdict[str, set[str]] = defaultdict(set)
    while queue:
        key = queue.popleft()
        if key in visited:
            continue
        visited.add(key)
        module, qualname = key
        node = _index(module).defs[qualname]
        for name in _config_names_read(node):
            sites[name].add(f'{module}::{qualname}')
        queue.extend(callee for callee in _callees(module, qualname, node) if callee not in visited)
    return _Census(
        reads=MappingProxyType({name: tuple(sorted(where)) for name, where in sites.items()}),
        visited_modules=frozenset(module for module, _ in visited),
    )


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_every_merge_path_config_read_is_classified() -> None:
    reads = _merge_path_census().reads
    unclassified = sorted(set(reads) - set(MERGE_PATH_CONFIG_READS))
    assert not unclassified, (
        'The merge path reads OrchestratorConfig names this file does not classify. '
        'Add a MERGE_PATH_CONFIG_READS row for each one. If the host value can decide '
        'the verdict, classify it HOST_DECIDING, file a narrow follow-up, and cite its '
        'key in follow_ups:\n'
        + '\n'.join(f'  {name}: read at {", ".join(reads[name])}' for name in unclassified)
    )


def test_every_classified_row_is_still_read() -> None:
    stale = sorted(set(MERGE_PATH_CONFIG_READS) - set(_merge_path_census().reads))
    assert not stale, (
        f'These MERGE_PATH_CONFIG_READS rows are no longer read on the merge path; delete them: {stale}'
    )


def test_census_resolves_reads_across_modules() -> None:
    census = _merge_path_census()
    assert len(census.reads) >= 30, sorted(census.reads)
    assert {'orchestrator.verify_plan', 'orchestrator.merge_lane.gates'} <= census.visited_modules, sorted(
        census.visited_modules
    )
