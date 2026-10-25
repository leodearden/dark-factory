"""The one place the orchestrator test suite enumerates and parses the repo for
whole-tree guards: a single streaming pass that keeps facts, never trees.

A guard module declares its :class:`TreeAnalysis` objects in a module-level
``TREE_ANALYSES`` and takes the session ``tree_pass`` fixture
(``conftest.py::tree_pass``).  The fixture discovers every analysis requested by
the collected items and runs them all in one :meth:`TreePass.run`: each file is
read and parsed at most once, lazily, handed to every interested extractor, and
dropped before the next, so at most one tree is ever live.  Extractors return
immutable, AST-free facts; anything else is recorded as a failure of that
analysis, because a retained node pins its whole tree.
"""
from __future__ import annotations

import ast
import dataclasses
import functools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import Any, Generic, TypeVar

T = TypeVar('T')

TESTS_DIR = Path(__file__).resolve().parent
ORCHESTRATOR_SRC_DIR = TESTS_DIR.parent / 'src' / 'orchestrator'
REPO_ROOT = TESTS_DIR.parents[1]

TREE_PASS_FIXTURE = 'tree_pass'
TREE_PASS_XDIST_GROUP = 'whole_tree_pass'


def python_files(*roots: Path, exclude_parts: frozenset[str] = frozenset()) -> tuple[Path, ...]:
    """Every ``*.py`` under *roots*, resolved, deduplicated and sorted.

    A path is dropped when any of its parts RELATIVE TO ITS ROOT is in
    *exclude_parts*.  A root that is not an existing directory raises, so a
    wrong root can never yield a vacuously empty census.
    """
    found: set[Path] = set()
    for root in roots:
        resolved_root = Path(root).resolve()
        if not resolved_root.is_dir():
            raise NotADirectoryError(f'python_files root is not an existing directory: {root}')
        for path in resolved_root.rglob('*.py'):
            if exclude_parts.isdisjoint(path.relative_to(resolved_root).parts):
                found.add(path.resolve())
    return tuple(sorted(found))


ORCHESTRATOR_TESTS = functools.partial(python_files, TESTS_DIR)
ORCHESTRATOR_SRC = functools.partial(python_files, ORCHESTRATOR_SRC_DIR)


class ParsedModule:
    """One file inside the pass: its source and tree are computed on first use."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @functools.cached_property
    def source(self) -> str:
        return self.path.read_text(encoding='utf-8')

    @functools.cached_property
    def tree(self) -> ast.Module:
        return ast.parse(self.source, filename=str(self.path))


@dataclass(frozen=True, eq=False)
class TreeAnalysis(Generic[T]):
    """A named per-file extraction over the file set *files* returns.

    Compared by identity, so two analyses are the same only if they are the
    same object.
    """

    name: str
    files: Callable[[], Iterable[Path]]
    extract: Callable[[ParsedModule], T]


@dataclass(frozen=True)
class ExtractionFailure:
    """Why one file produced no fact, as strings: an exception's frames would pin trees."""

    path: Path
    error_type: str
    message: str


_PLAIN_SCALARS = (str, bytes, int, float, bool, type(None), PurePath)


def _require_plain_fact(fact: object, analysis_name: str) -> None:
    if isinstance(fact, _PLAIN_SCALARS):
        return
    if isinstance(fact, tuple | frozenset):
        for item in fact:
            _require_plain_fact(item, analysis_name)
        return
    if dataclasses.is_dataclass(fact) and not isinstance(fact, type) and _is_frozen(fact):
        for field in dataclasses.fields(fact):
            _require_plain_fact(getattr(fact, field.name), analysis_name)
        return
    raise TypeError(
        f'analysis {analysis_name!r} returned a fact holding a {type(fact).__name__}; '
        'facts are shared across tests and must be immutable and AST-free: '
        'str, bytes, int, float, bool, None or Path, or a tuple, frozenset, '
        'NamedTuple or frozen dataclass of those'
    )


def _is_frozen(dataclass_instance: object) -> bool:
    params = getattr(type(dataclass_instance), '__dataclass_params__', None)
    return params is not None and params.frozen


def _dispatch_one_file(
    module: ParsedModule,
    interested: Iterable[TreeAnalysis[Any]],
    facts: dict[TreeAnalysis[Any], dict[Path, Any]],
    failures: dict[TreeAnalysis[Any], list[ExtractionFailure]],
) -> None:
    for analysis in interested:
        try:
            fact = analysis.extract(module)
            _require_plain_fact(fact, analysis.name)
        except Exception as error:
            failures[analysis].append(
                ExtractionFailure(module.path, type(error).__name__, str(error)),
            )
        else:
            facts[analysis][module.path] = fact


def _require_unique_names(analyses: tuple[TreeAnalysis[Any], ...]) -> None:
    names = [analysis.name for analysis in analyses]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f'tree analyses must have unique names; duplicated: {duplicates}')


def _resolved_file_sets(
    analyses: tuple[TreeAnalysis[Any], ...],
) -> dict[TreeAnalysis[Any], frozenset[Path]]:
    resolved: dict[Path, Path] = {}

    def resolve(path: Path) -> Path:
        if path not in resolved:
            resolved[path] = Path(path).resolve()
        return resolved[path]

    return {
        analysis: frozenset(resolve(path) for path in analysis.files())
        for analysis in analyses
    }


class TreePass:
    """The facts every declared analysis extracted in one streaming pass."""

    def __init__(
        self,
        facts: Mapping[TreeAnalysis[Any], Mapping[Path, Any]],
        failures: Mapping[TreeAnalysis[Any], tuple[ExtractionFailure, ...]],
    ) -> None:
        self._facts = MappingProxyType(dict(facts))
        self._failures = MappingProxyType(dict(failures))

    @classmethod
    def run(cls, analyses: Iterable[TreeAnalysis[Any]]) -> TreePass:
        declared = tuple(analyses)
        _require_unique_names(declared)
        file_sets = _resolved_file_sets(declared)
        facts: dict[TreeAnalysis[Any], dict[Path, Any]] = {a: {} for a in declared}
        failures: dict[TreeAnalysis[Any], list[ExtractionFailure]] = {a: [] for a in declared}
        for path in sorted(frozenset().union(*file_sets.values())):
            interested = [a for a in declared if path in file_sets[a]]
            _dispatch_one_file(ParsedModule(path), interested, facts, failures)
        return cls(
            {a: MappingProxyType(found) for a, found in facts.items()},
            {a: tuple(failed) for a, failed in failures.items()},
        )

    def facts(self, analysis: TreeAnalysis[T]) -> Mapping[Path, T]:
        """*analysis*'s facts keyed by resolved path, in sorted order.

        Raises if *analysis* was not part of this pass, or if any of its files
        failed, so a consumer never asserts over a partial census.
        """
        if analysis not in self._facts:
            raise LookupError(
                f'tree analysis {analysis.name!r} was not part of this tree pass: '
                "add it to its module's TREE_ANALYSES so the tree_pass fixture "
                'discovers it'
            )
        failed = self._failures[analysis]
        if failed:
            listing = '\n  '.join(f'{f.path}: {f.error_type}: {f.message}' for f in failed)
            raise AssertionError(
                f'tree analysis {analysis.name!r} failed on {len(failed)} file(s):\n  {listing}'
            )
        return self._facts[analysis]


def analyses_requested_by(items: Iterable[object]) -> tuple[TreeAnalysis[Any], ...]:
    """The ``TREE_ANALYSES`` of every module with an item requesting the tree pass.

    ``fixturenames`` is pytest's transitive closure, so an indirect request
    counts.  Deduplicated by identity, in first-seen order.
    """
    requesting_modules = (
        getattr(item, 'module', None)
        for item in items
        if TREE_PASS_FIXTURE in getattr(item, 'fixturenames', ())
    )
    return tuple(dict.fromkeys(
        analysis
        for module in requesting_modules
        if module is not None
        for analysis in getattr(module, 'TREE_ANALYSES', ())
    ))
