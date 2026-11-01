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
import enum
import functools
import subprocess
import unicodedata
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
    """Every ``*.py`` under each resolved root in *roots*, deduplicated and sorted.

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
                found.add(path)
    return tuple(sorted(found))


ORCHESTRATOR_TESTS = functools.partial(python_files, TESTS_DIR)
ORCHESTRATOR_SRC = functools.partial(python_files, ORCHESTRATOR_SRC_DIR)


def orchestrator_test_modules() -> tuple[Path, ...]:
    """Every ``test_*.py`` directly in the orchestrator tests directory, sorted: the modules pytest collects there."""
    return tuple(sorted(TESTS_DIR.glob('test_*.py')))

CHECKOUT_SKIP_PARTS = frozenset({'.worktrees', '.venv', 'node_modules', '__pycache__'})


def git_tracked_python_files(repo_root: Path) -> tuple[Path, ...]:
    """Every ``*.py`` git tracks in *repo_root*, under its resolved root, or ``()`` if git cannot answer.

    NUL-delimited (``-z``) so a path holding a space or newline cannot truncate
    the listing, which would make any completeness check built on it pass
    vacuously.
    """
    root = Path(repo_root).resolve()
    try:
        listing = subprocess.run(
            ['git', '-C', str(root), 'ls-files', '-z', '--', '*.py'],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    return tuple(root / relpath for relpath in listing.split('\0') if relpath)


def _package_root_python_files(root: Path) -> tuple[Path, ...]:
    package_roots = [
        *sorted(root.glob('*/src/*')),
        *sorted(root.glob('*/tests')),
        root / 'scripts',
        root / 'hooks',
    ]
    walkable = [
        package_root for package_root in package_roots
        if package_root.is_dir()
        and CHECKOUT_SKIP_PARTS.isdisjoint(package_root.relative_to(root).parts)
    ]
    return (
        *root.glob('*.py'),
        *python_files(*walkable, exclude_parts=CHECKOUT_SKIP_PARTS),
    )


def checkout_python_files(repo_root: Path) -> tuple[Path, ...]:
    """Every ``*.py`` in the checkout at *repo_root*: git-tracked UNION the on-disk
    package roots, deduplicated and sorted.

    Neither source alone is enough.  ``git ls-files`` is complete by
    construction for committed work but cannot see a file written and not yet
    added; the package roots (``*/src/*``, ``*/tests``, ``scripts``, ``hooks``
    and top-level ``*.py``) cover that in-progress work but, as a hand-kept
    list, miss whole tracked trees such as ``skills/**/scripts``.  Only files
    that exist are returned, since the index can list a deleted file.

    :data:`CHECKOUT_SKIP_PARTS` is judged on each path RELATIVE to *repo_root*:
    every task worktree lives under ``.worktrees/``, so an absolute-path filter
    would skip everything.  Never widen this into a whole-repo walk: the main
    checkout holds sibling copies of this repo at other revisions, which git
    never lists and these roots never reach.
    """
    root = Path(repo_root).resolve()
    candidates = (*git_tracked_python_files(root), *_package_root_python_files(root))
    return tuple(sorted({path for path in candidates if path.is_file()}))


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

    def may_name(self, *fragments: str, ignore_case: bool = False) -> bool:
        """Whether some identifier in this module could contain one of *fragments*.

        False proves none can, so an extractor whose every finding names one
        may return its empty fact without walking the tree.  The source is
        read as the parser compares identifiers, under NFKC, so ``ｇｌｏｂ``
        counts as ``glob``; *ignore_case* lowercases both sides, for a detector
        that lowercases names itself.  Comments and strings only add matches.
        """
        text = self._lowered_names_text if ignore_case else self._names_text
        return any((fragment.lower() if ignore_case else fragment) in text for fragment in fragments)

    @functools.cached_property
    def _names_text(self) -> str:
        source = self.source
        return source if source.isascii() else unicodedata.normalize('NFKC', source)

    @functools.cached_property
    def _lowered_names_text(self) -> str:
        return self._names_text.lower()


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

    def __str__(self) -> str:
        return f'{self.path}: {self.error_type}: {self.message}'


@dataclass(frozen=True)
class ListingFailure:
    """Why an analysis could not list its files, so it examined none."""

    error_type: str
    message: str

    def __str__(self) -> str:
        return f'listing its files: {self.error_type}: {self.message}'


AnalysisFailure = ExtractionFailure | ListingFailure


_PLAIN_SCALARS = (str, bytes, int, float, bool, type(None), PurePath, enum.Enum)


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
        'str, bytes, int, float, bool, None, Path or an Enum member, or a tuple, frozenset, '
        'NamedTuple or frozen dataclass of those'
    )


def _is_frozen(dataclass_instance: object) -> bool:
    params = getattr(type(dataclass_instance), '__dataclass_params__', None)
    return params is not None and params.frozen


def _dispatch_one_file(
    module: ParsedModule,
    interested: Iterable[TreeAnalysis[Any]],
    facts: dict[TreeAnalysis[Any], dict[Path, Any]],
    failures: dict[TreeAnalysis[Any], list[AnalysisFailure]],
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


def _anchored(path: Path) -> Path:
    """*path* with its directories resolved and its own name kept.

    A symlinked file therefore stays at its link, under the root it was listed
    from, rather than becoming a target that may lie outside that root.  Paths
    from :func:`python_files` and :func:`git_tracked_python_files` already have
    this form.
    """
    path = Path(path)
    return path.parent.resolve() / path.name


def _anchored_file_sets(
    analyses: tuple[TreeAnalysis[Any], ...],
    failures: dict[TreeAnalysis[Any], list[AnalysisFailure]],
) -> dict[TreeAnalysis[Any], frozenset[Path]]:
    """Each analysis's :func:`_anchored` file set; one that cannot list its files gets none and a failure."""
    anchored: dict[Path, Path] = {}

    def anchor(path: Path) -> Path:
        if path not in anchored:
            anchored[path] = _anchored(path)
        return anchored[path]

    file_sets: dict[TreeAnalysis[Any], frozenset[Path]] = {}
    for analysis in analyses:
        try:
            file_sets[analysis] = frozenset(anchor(path) for path in analysis.files())
        except Exception as error:
            failures[analysis].append(ListingFailure(type(error).__name__, str(error)))
            file_sets[analysis] = frozenset()
    return file_sets


class TreePass:
    """The facts every declared analysis extracted in one streaming pass."""

    def __init__(
        self,
        facts: Mapping[TreeAnalysis[Any], Mapping[Path, Any]],
        failures: Mapping[TreeAnalysis[Any], tuple[AnalysisFailure, ...]],
    ) -> None:
        self._facts = MappingProxyType(dict(facts))
        self._failures = MappingProxyType(dict(failures))

    @classmethod
    def run(cls, analyses: Iterable[TreeAnalysis[Any]]) -> TreePass:
        declared = tuple(analyses)
        _require_unique_names(declared)
        facts: dict[TreeAnalysis[Any], dict[Path, Any]] = {a: {} for a in declared}
        failures: dict[TreeAnalysis[Any], list[AnalysisFailure]] = {a: [] for a in declared}
        file_sets = _anchored_file_sets(declared, failures)
        for path in sorted(frozenset().union(*file_sets.values())):
            interested = [a for a in declared if path in file_sets[a]]
            _dispatch_one_file(ParsedModule(path), interested, facts, failures)
        return cls(
            {a: MappingProxyType(found) for a, found in facts.items()},
            {a: tuple(failed) for a, failed in failures.items()},
        )

    def facts(self, analysis: TreeAnalysis[T]) -> Mapping[Path, T]:
        """*analysis*'s facts keyed by :func:`_anchored` path, in sorted order.

        Raises if *analysis* was not part of this pass, or if it could not list
        its files or any of them failed, so a consumer never asserts over a
        partial census.
        """
        if analysis not in self._facts:
            raise LookupError(
                f'tree analysis {analysis.name!r} was not part of this tree pass: '
                "add it to its module's TREE_ANALYSES so the tree_pass fixture "
                'discovers it'
            )
        failed = self._failures[analysis]
        if failed:
            listing = '\n  '.join(map(str, failed))
            raise AssertionError(f'tree analysis {analysis.name!r} failed:\n  {listing}')
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
