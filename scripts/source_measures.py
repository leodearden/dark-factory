#!/usr/bin/env python3
"""Per-file source measures, and the listing of a repo's tracked files.

Each measure reads one Python source (or one tree parsed from it) and knows
nothing of any consumer: sizes and prose, function-local imports, re-export
names, cognitive complexity (complexipy), the maintainability index (radon),
patch targets through a given set of modules, and private-attribute reads.

INV-11 polarity: an input that cannot be parsed or measured raises
``MetricsError`` naming the path or the tool, never a zero or a skip. Whether
to record-and-skip such a file is the CALLER's decision, made by catching it.
complexipy and radon are imported lazily, inside the functions that need them.
"""
from __future__ import annotations

import ast
import dataclasses
import enum
import subprocess
import tokenize
import tomllib
from collections.abc import Collection, Mapping
from io import StringIO
from pathlib import Path
from typing import NamedTuple

# NOTE: module scope imports only the stdlib ON PURPOSE. complexipy and radon
# are imported lazily inside the functions that need them (see
# ``_import_complexipy``), so this module stays importable and type-checkable in
# environments whose dev group does not carry them -- notably the ``shared``
# project, which owns the ``ruff check scripts/`` and ``pyright scripts/`` gates
# for this file. Laziness is not softness: the moment a measure actually needs
# the tool, a missing or wrong-version one raises MetricsError with a named
# cause (INV-11).


class MetricsError(Exception):
    """A measurement that was asked for could not be taken.

    Always raised with the offending path / tool / version named in the message,
    so a caller can tell a broken input or tool apart from a measured result.
    """


# ---------------------------------------------------------------------------
# Source helpers shared by the per-file measures.


def parse_source(source: str, *, path: str) -> ast.Module:
    """Parse *source*, translating a SyntaxError into a named MetricsError.

    INV-11: an unparseable file is a named failure, never a skipped measure. A
    caller sweeping files it does not own catches this and records the path
    instead.

    Nothing is remembered between calls. A file's measures share ONE parse by
    its caller parsing once and handing the tree to each ``*_in_tree`` measure;
    the ``source``-taking functions beside them are thin wrappers that parse and
    delegate, for the callers that measure a single measure of a single snippet.
    """
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        raise MetricsError(
            f'{path}: could not be parsed -- SyntaxError: {exc}'
        ) from exc
    except ValueError as exc:  # e.g. source containing a null byte
        raise MetricsError(
            f'{path}: could not be parsed -- {exc.__class__.__name__}: {exc}'
        ) from exc


def read_source(root: Path, relpath: str) -> str:
    try:
        return (root / relpath).read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise MetricsError(
            f'{relpath}: could not be read -- {exc.__class__.__name__}: {exc}'
        ) from exc


def _comment_lines(source: str, *, path: str) -> set[int]:
    """Line numbers carrying a COMMENT token, via stdlib ``tokenize``.

    Token-based rather than regex-based on purpose: a string literal that merely
    mentions ``#`` is not a comment, and no regex over source text gets that
    right.
    """
    lines: set[int] = set()
    try:
        for token in tokenize.generate_tokens(StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                lines.add(token.start[0])
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise MetricsError(
            f'{path}: could not be tokenized -- {exc.__class__.__name__}: {exc}'
        ) from exc
    return lines


# ---------------------------------------------------------------------------
# Per-file size measures.


@dataclasses.dataclass(frozen=True)
class FileSizeMeasures:
    """Physical lines, and how many of them are docstring or comment."""

    lines: int
    prose_lines: int


_DOCSTRING_HOLDERS: tuple[type[ast.AST], ...] = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


def docstring_of(node: ast.AST) -> ast.Expr | None:
    """The docstring statement of *node*, or None when it has none.

    A docstring is the FIRST body element of a module, class or function when it
    is a bare string expression -- exactly Python's own rule, so a second string
    expression in the same body is code, not prose.
    """
    if not isinstance(node, _DOCSTRING_HOLDERS):
        return None
    body = getattr(node, 'body', None)
    if not body:
        return None
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first
    return None


def _docstrings(tree: ast.Module) -> list[ast.Expr]:
    """Every docstring statement in *tree*."""
    return [
        docstring
        for node in ast.walk(tree)
        if (docstring := docstring_of(node)) is not None
    ]


def _docstring_lines(tree: ast.Module) -> set[int]:
    """Line numbers spanned by every docstring in *tree*."""
    lines: set[int] = set()
    for docstring in _docstrings(tree):
        end = docstring.end_lineno if docstring.end_lineno is not None else docstring.lineno
        lines.update(range(docstring.lineno, end + 1))
    return lines


def file_size_measures(source: str, *, path: str) -> FileSizeMeasures:
    """Measure *source*'s physical and prose line counts.

    ``prose_lines`` is the UNION of two line-number sets -- docstring spans
    (from the AST) and COMMENT-token lines (from stdlib ``tokenize``) -- so a
    line that is both counts once. Token-based comment detection is what makes
    ``url = 'http://x/#frag'`` correctly zero prose lines; a regex over source
    text cannot.

    Raises ``MetricsError`` naming *path* when the source cannot be parsed or
    tokenized. INV-11: never a zero or None measure for a file we failed to read.
    """
    return file_size_measures_in_tree(source, parse_source(source, path=path), path=path)


def file_size_measures_in_tree(
    source: str, tree: ast.Module, *, path: str
) -> FileSizeMeasures:
    """``file_size_measures`` over a tree the caller already parsed from *source*."""
    prose = _docstring_lines(tree) | _comment_lines(source, path=path)
    return FileSizeMeasures(lines=len(source.splitlines()), prose_lines=len(prose))


# ---------------------------------------------------------------------------
# Structural import measures.

_FUNCTION_NODES: tuple[type[ast.AST], ...] = (ast.FunctionDef, ast.AsyncFunctionDef)


def function_local_imports(source: str, *, path: str) -> int:
    """Count import statements that live inside a function body.

    These are reach-back imports -- function-local imports that exist to break
    import cycles, plus every other deferred import in the same shape.

    Counted per STATEMENT, not per bound name, and deduped by node identity so a
    nested function's import is counted once rather than once per enclosing
    function. AST-based, so a docstring quoting an import statement -- which
    reach-back notes often do verbatim -- is never counted.
    """
    return function_local_imports_in_tree(parse_source(source, path=path))


def function_local_imports_in_tree(tree: ast.Module) -> int:
    """``function_local_imports`` over an already-parsed tree."""
    seen: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _FUNCTION_NODES):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Import | ast.ImportFrom):
                seen.add(id(inner))
    return len(seen)


def reexport_names(source: str, *, path: str) -> list[str]:
    """Names a module imports at module level and never itself references.

    This is the STRUCTURAL reading of a re-export shim, and deliberately not a
    scan for the ``# noqa: F401  re-export shim`` comment: comments do not exist
    in the AST at all, and a comment-based detector would zero out on a purely
    cosmetic edit. The structural predicate is exactly what ruff's F401 computes
    -- which is precisely why those blocks carry the suppression -- so it agrees
    with the annotated set while being ungameable.

    Scoped to MODULE-LEVEL ``from X import ...`` bindings: a bare ``import x``
    binds a module rather than re-exporting a name, a function-local import is
    the ``function_local_imports`` measure's business, and ``import *`` binds
    nothing nameable. ``from __future__ import ...`` is likewise excluded: it is
    a compiler directive, not a name a downstream module could import, and ruff
    explicitly never flags it under F401 -- counting it would both inflate every
    figure and make a ratchet on it REWARD deleting a future import, which
    silently changes runtime annotation semantics. Returns the bound names
    (``asname or name``) sorted and deduped.
    """
    return reexport_names_in_tree(parse_source(source, path=path))


def reexport_names_in_tree(tree: ast.Module) -> list[str]:
    """``reexport_names`` over an already-parsed tree."""
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    # An attribute chain rooted at the binding (`B.attr`) also uses it, and so
    # does an `__all__` listing -- but `__all__` entries are string constants,
    # not Names, and a module that re-exports via `__all__` is still a shim by
    # this measure's definition.
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module == '__future__':
            continue
        for alias in node.names:
            if alias.name == '*':
                continue
            bound = alias.asname or alias.name
            if bound not in used:
                names.add(bound)
    return sorted(names)


# ---------------------------------------------------------------------------
# The complexipy adapter, and the tool-availability half of INV-11.
#
# WHY BOTH BOUNDS EXIST, measured 2026-09-03 against merge_queue.py (21,550
# lines) in this worktree:
#
#   version | wall clock | file total | _verifier_loop | _run_post_merge_verify
#   3.0.0   |     4.63s  |      2031  |  186           |  183
#   4.0.0   |     4.31s  |      2124  |  188           |  183
#   5.0.0   |     4.81s  |      2092  |  192           |  188
#   6.0.0   |     5.18s  |      2133  |  245           |  175
#   6.2.0   |     4.75s  |      2133  |  245           |  175
#   7.0.1   |   247.00s  |      2133  |  245           |  175
#
# FLOOR (>=6.2) is CORRECTNESS: the algorithm changed across majors, so 3/4/5
# compute different numbers for the identical file. An unpinned complexipy would
# silently rewrite every baseline figure on upgrade, turning a ratchet into
# noise. 6.x and 7.0.1 agree, and reproduce exactly the numbers the PRD
# Background table quotes; 6.2.0 is the newest 6.x and the version every
# committed baseline number was measured with.
#
# CEILING (<7) is PERFORMANCE, and it is not hygiene -- it is what keeps this
# instrument from becoming a suite-truncating landmine. 7.0.1's cost grows
# roughly cubically in file size (2,800 lines = 0.31s; 21,550 lines = 247s), so
# the whole 22-path cluster costs ~330s+ at 7.0.1 against 9.52s at 6.2.0. The
# ratchet carries pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT) = 300s, and
# exceeding it does not merely fail the test: pytest-timeout's thread method
# os._exit()s the xdist worker and --max-worker-restart=0 then truncates the
# ENTIRE orchestrator suite, reporting against an innocent test (the esc-3980-1
# / esc-3787-1 mode documented at orchestrator/tests/_orch_helpers.py::
# WHOLE_TREE_SCAN_TEST_TIMEOUT).
#
# The tuples are the source of truth; the human-readable specifier is derived
# from them, so the message and the check can never disagree. The same string
# is pinned against orchestrator/pyproject.toml's dev-group entry by
# test_pyproject_pin_matches_the_scripts_requirement.
COMPLEXIPY_MIN: tuple[int, ...] = (6, 2)
COMPLEXIPY_MAX_EXCLUSIVE: tuple[int, ...] = (7,)
COMPLEXIPY_REQUIRED = '>={},<{}'.format(
    '.'.join(str(part) for part in COMPLEXIPY_MIN),
    '.'.join(str(part) for part in COMPLEXIPY_MAX_EXCLUSIVE),
)

_COMPLEXIPY_RANGE_REASON = (
    'complexipy majors compute DIFFERENT cognitive numbers for the same file '
    '(merge_queue.py totals 2031 at 3.0.0, 2092 at 5.0.0, 2133 at 6.x/7.x), so '
    'an unpinned engine would silently rewrite every baseline figure; and 7.x '
    'is ~48x slower on the monolith (247.0s vs 4.75s at 6.2.0, ~330s+ vs 9.52s '
    'cluster-wide) against the ratchet test\'s 300s timeout, which pytest-'
    'timeout enforces by os._exit()ing the xdist worker and truncating the '
    'whole suite. Install the pinned version: `uv sync --all-packages`.'
)


def _version_parts(version: str) -> tuple[int, ...]:
    """Leading numeric release segments of *version*, e.g. '6.2.0rc1' -> (6, 2, 0).

    Deliberately hand-rolled rather than reaching for ``packaging``: this module
    imports no third-party package at import time (see the note at the top), and a two-clause
    ``>=X,<Y`` range over release segments needs nothing more.
    """
    parts: list[int] = []
    for segment in version.split('.'):
        digits = ''
        for char in segment:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def satisfies_complexipy_requirement(version: str) -> bool:
    """True when *version* falls inside ``COMPLEXIPY_REQUIRED``."""
    parts = _version_parts(version)
    if not parts:
        return False
    return COMPLEXIPY_MIN <= parts < COMPLEXIPY_MAX_EXCLUSIVE


def complexipy_version() -> str:
    """The installed complexipy version, or ``MetricsError`` naming the tool."""
    import importlib.metadata

    try:
        return importlib.metadata.version('complexipy')
    except importlib.metadata.PackageNotFoundError as exc:
        raise MetricsError(
            'complexipy is not installed, so cognitive complexity cannot be '
            'measured. It belongs to orchestrator/pyproject.toml '
            '[dependency-groups] dev, pinned '
            f'{COMPLEXIPY_REQUIRED}. Run `uv sync --all-packages`.'
        ) from exc


def require_complexipy() -> str:
    """Assert the installed complexipy is inside ``COMPLEXIPY_REQUIRED``.

    Callers check before measuring, so a wrong-version environment fails
    immediately with a named cause rather than after a 250-second measurement
    whose numbers would be wrong anyway.
    """
    # Looked up through the module namespace on purpose, so a test can seed a
    # version without installing one.
    version = globals()['complexipy_version']()
    if not satisfies_complexipy_requirement(version):
        raise MetricsError(
            f'complexipy {version} is installed but this instrument requires '
            f'{COMPLEXIPY_REQUIRED}. {_COMPLEXIPY_RANGE_REASON}'
        )
    return version


def _import_complexipy():  # noqa: ANN202 - third-party module object
    """Import complexipy LAZILY, naming it in the failure.

    Lazy so this module stays importable and type-checkable under the
    ``shared`` project that owns its ruff/pyright gates, whose dev group
    carries neither complexipy nor radon. Lazy is not
    soft: the moment a measure actually needs the tool, a missing one is an
    instrument failure with a named cause.
    """
    try:
        import complexipy  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MetricsError(
            'complexipy could not be imported, so cognitive complexity cannot '
            'be measured. It belongs to orchestrator/pyproject.toml '
            f'[dependency-groups] dev, pinned {COMPLEXIPY_REQUIRED}. '
            f'Run `uv sync --all-packages`. ({exc})'
        ) from exc
    return complexipy


def _file_complexity(path: Path):  # noqa: ANN202 - complexipy.FileComplexity
    complexipy = _import_complexipy()
    try:
        return complexipy.file_complexity(str(path))
    except MetricsError:
        raise
    except Exception as exc:
        raise MetricsError(
            f'{path}: complexipy could not measure this file -- '
            f'{exc.__class__.__name__}: {exc}'
        ) from exc


@dataclasses.dataclass(frozen=True)
class FileCognitive:
    """Both cognitive projections of ONE complexipy measurement of one file."""

    total: int
    per_function: dict[str, int]


def file_cognitive_measures(path: Path) -> FileCognitive:
    """Measure *path* with complexipy ONCE and derive both projections from it.

    The file total and the per-function map are two VIEWS of a single
    ``FileComplexity``, so one call answers both and a caller asks once per
    file. There are deliberately no separate ``cognitive_complexity`` /
    ``file_cognitive_total`` accessors: a named half per projection would be
    surface kept alive by its own tests. Deliberately NOT a cache either: a memo
    keyed on a path would return the file as it WAS, would retain every result
    for the life of a long-running process, and would need a carve-out to keep
    ``_file_complexity``'s MetricsError from being swallowed.

    ``total`` is a projection of its own, beside ``per_function``, because it
    also counts module-level control flow belonging to no function. complexipy
    already emits ``Class::method`` for methods, so the keys need no
    post-processing, and a module with no functions yields an empty map -- a
    real measurement, not a skipped one.
    """
    result = _file_complexity(path)
    return FileCognitive(
        total=int(result.complexity),
        per_function={
            function.name: function.complexity for function in result.functions
        },
    )


def maintainability_index(source: str, *, path: str) -> float:
    """radon's maintainability index for *source*, in [0, 100].

    A derived composite (Halstead volume, cyclomatic complexity, SLOC, comment
    ratio) that already reads 0.00 for this repo's largest files, merge_queue.py
    and git_ops.py, so it has no headroom left to ratchet against and would only
    ever restate what the line and cognitive measures already say: it is for
    reporting, not for a ratchet.
    """
    try:
        from radon.metrics import mi_visit  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MetricsError(
            'radon could not be imported, so the maintainability index cannot '
            'be reported. It belongs to orchestrator/pyproject.toml '
            f'[dependency-groups] dev. Run `uv sync --all-packages`. ({exc})'
        ) from exc
    try:
        return float(mi_visit(source, True))
    except Exception as exc:
        raise MetricsError(
            f'{path}: radon could not compute a maintainability index -- '
            f'{exc.__class__.__name__}: {exc}'
        ) from exc


# ---------------------------------------------------------------------------
# Patch targets: the names a source patches through a given set of modules.
#
# The two detector shapes below were PORTED from the retired
# orchestrator/tests/test_merge_queue_reachback_patch_guard.py --
# `_merge_queue_module_aliases()` and the is_setattr / is_dotted_patch /
# is_bare_patch / is_patch_object call classification inside
# `_find_merge_queue_private_patches()` -- with two deliberate generalisations:
# the binding helper takes a SET of module paths rather than one hardcoded
# module, and that guard's `forbidden` filter is dropped so ALL leaves are
# counted rather than only the private ones.


@dataclasses.dataclass(frozen=True)
class PatchCall:
    """One patch-shaped call: ``target`` for ``patch('a.b._c')``, or
    ``receiver`` + ``attribute`` for ``patch.object(mod, '_c')``."""

    target: str | None
    receiver: ast.expr | None
    attribute: str | None


def _str_constant(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    return None


def _patch_call(node: ast.Call) -> PatchCall | None:
    func = node.func
    is_setattr = isinstance(func, ast.Attribute) and func.attr == 'setattr'
    is_dotted_patch = isinstance(func, ast.Attribute) and func.attr == 'patch'
    is_bare_patch = isinstance(func, ast.Name) and func.id == 'patch'
    is_patch_object = (
        isinstance(func, ast.Attribute)
        and func.attr == 'object'
        and (
            (isinstance(func.value, ast.Name) and func.value.id == 'patch')
            or (isinstance(func.value, ast.Attribute) and func.value.attr == 'patch')
        )
    )
    target = _str_constant(node.args[0])
    if target is not None:
        if is_setattr or is_dotted_patch or is_bare_patch:
            return PatchCall(target=target, receiver=None, attribute=None)
        return None
    if (is_setattr or is_patch_object) and len(node.args) >= 2:
        attribute = _str_constant(node.args[1])
        if attribute is not None:
            return PatchCall(target=None, receiver=node.args[0], attribute=attribute)
    return None


def patch_calls_in_tree(node: ast.AST) -> tuple[PatchCall, ...]:
    """Every patch-shaped call under *node*, in ``ast.walk`` order, duplicates kept."""
    calls = (
        _patch_call(call)
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and call.args
    )
    return tuple(call for call in calls if call is not None)


class PatchTarget(NamedTuple):
    """One name patched through a module of the set: ``patch('a.b._c')`` with
    ``a.b`` in the set is ``PatchTarget('a.b', '_c')``."""

    module: str
    leaf: str


def _dotted_spelling(expr: ast.expr) -> str | None:
    """``a.b.c`` for a Name/Attribute chain; None for any other expression."""
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        head = _dotted_spelling(expr.value)
        return None if head is None else f'{head}.{expr.attr}'
    return None


def _module_bindings(tree: ast.AST, modules: Collection[str]) -> dict[str, str]:
    """Names bound directly to a module of *modules* anywhere in *tree*, each
    mapped to that module.

    e.g. ``import a.b as x`` binds ``x``, and ``from a import b`` binds ``b``,
    when ``a.b`` is in *modules*. Used to recognise the object-path idiom, which
    targets the identical lookup site as the string-path form without
    embedding the module path as a string constant.
    """
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in modules and alias.asname:
                    bindings[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                module = f'{node.module}.{alias.name}'
                if module in modules:
                    bindings[alias.asname or alias.name] = module
    return bindings


def _receiver_module(
    receiver: ast.expr, bindings: Mapping[str, str], modules: Collection[str]
) -> str | None:
    """The module of *modules* that *receiver* IS, or None.

    A bound name resolves through its binding; anything else only when its own
    dotted spelling is a member. Never by prefix: ``a.b.sub`` is not ``a.b``.
    """
    if isinstance(receiver, ast.Name) and receiver.id in bindings:
        return bindings[receiver.id]
    spelling = _dotted_spelling(receiver)
    return spelling if spelling in modules else None


def _string_target(target: str, modules: Collection[str]) -> PatchTarget | None:
    """*target* split at the LONGEST member of *modules* it continues past."""
    owners = [module for module in modules if target.startswith(module + '.')]
    if not owners:
        return None
    module = max(owners, key=len)
    return PatchTarget(module, target[len(module) + 1:])


def _patch_target(
    call: PatchCall, bindings: Mapping[str, str], modules: Collection[str]
) -> PatchTarget | None:
    if call.target is not None:
        return _string_target(call.target, modules)
    if call.receiver is None or call.attribute is None:
        return None
    module = _receiver_module(call.receiver, bindings, modules)
    return None if module is None else PatchTarget(module, call.attribute)


def patch_targets(
    source: str, modules: Collection[str], *, path: str = '<source>'
) -> frozenset[PatchTarget]:
    """Distinct ``(module, leaf)`` pairs *source* patches through *modules*.

    A string target (``patch('a.b._c')``, ``monkeypatch.setattr('a.b._c', v)``)
    belongs to the LONGEST member it continues past, whatever order *modules*
    lists them in, and its leaf is the dotted remainder. An object target
    (``patch.object(m, '_c')``, ``monkeypatch.setattr(m, '_c', v)``) counts
    only when ``m`` IS a member: bound to one by an import, or spelled as one.

    Distinct pairs, not call sites. AST-based, so a docstring or comment quoting
    a dotted path is never mistaken for a real patch site.
    """
    return patch_targets_in_tree(parse_source(source, path=path), modules)


def patch_targets_in_tree(
    tree: ast.Module, modules: Collection[str]
) -> frozenset[PatchTarget]:
    """``patch_targets`` over an already-parsed tree."""
    bindings = _module_bindings(tree, modules)
    found = (_patch_target(call, bindings, modules) for call in patch_calls_in_tree(tree))
    return frozenset(target for target in found if target is not None and target.leaf)


# ---------------------------------------------------------------------------
# Private-attribute reads.
#
# THIS MEASURE IS DELIBERATELY RECEIVER-AGNOSTIC. It counts every `_x` attribute
# access except on the bare `self`/`cls`, and it does NOT maintain a list of
# blessed receiver variable names (`worker`, `mq`, ...). Two reasons, and the
# first is the decisive one:
#
# (1) A receiver-name allowlist is a single SHARED list that every parallel
#     task lowering the count would have to edit concurrently -- the same
#     rebase-conflict hazard a per-path baseline format exists to avoid.
# (2) It silently UNDER-counts the moment a test uses a receiver name nobody
#     listed, which is how a ratchet rots into a vacuous pass. Over-counting is
#     the safe direction here: a ratchet only ever refuses to let a number RISE,
#     so a superset costs a little extra friction and never lets a regression
#     through.
#
# The `self`/`cls` exclusion is a STRUCTURAL predicate, not a name list: a test
# class's own helpers are not another module's internals, which is the
# distinction the measure is actually about. Note it excludes only the BARE
# receiver, so `self.worker._x` still counts.

_SELF_RECEIVERS = frozenset({'self', 'cls'})


def private_reads(source: str, *, path: str) -> int:
    """Count accesses of a single-underscore attribute in *source*.

    Writes count too: both directions couple the test to an internal name, which
    is the coupling the measure exists to shrink. Dunders are excluded (Python
    protocol, not internals) and so is the bare ``self``/``cls`` receiver.
    """
    return private_reads_in_tree(parse_source(source, path=path))


def private_reads_in_tree(tree: ast.Module) -> int:
    """``private_reads`` over an already-parsed tree."""
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not node.attr.startswith('_') or node.attr.startswith('__'):
            continue
        receiver = node.value
        if isinstance(receiver, ast.Name) and receiver.id in _SELF_RECEIVERS:
            continue
        count += 1
    return count


# ---------------------------------------------------------------------------
# The tracked-file listing.


def _git_output(root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ['git', '-C', str(root), *args],
            capture_output=True,
            encoding='utf-8',
            errors='surrogateescape',
            check=False,
        )
    except OSError as exc:
        raise MetricsError(
            f'could not run git for `git {" ".join(args)}` in {root}: '
            f'{exc.__class__.__name__}: {exc} -- the tracked files are listed '
            'with git, and a listing that failed would read as an empty tree'
        ) from exc
    if completed.returncode != 0:
        raise MetricsError(
            f'`git {" ".join(args)}` failed in {root} (exit {completed.returncode}): '
            f'{completed.stderr.strip() or "no stderr"} -- a listing that failed '
            'would read as an empty tree'
        )
    return completed.stdout


def _require_work_tree_top(root: Path) -> None:
    above_root = _git_output(root, 'rev-parse', '--show-cdup').strip()
    if above_root:
        raise MetricsError(
            f'{root} is not the top of a git work tree ({above_root!r} leads up '
            'to it), so its tracked files cannot be listed as the repo; pass the '
            'repo root instead'
        )


def tracked_files(root: Path, *pathspecs: str) -> tuple[str, ...]:
    """Every file git tracks under *root* matching *pathspecs*, repo-relative and sorted.

    *root* must be the top of its work tree. Listing from a directory that is
    not a repository but sits inside one yields only that directory's tracked
    files -- usually none -- and that partial listing would read as the whole
    tree.
    """
    _require_work_tree_top(root)
    listing = _git_output(root, 'ls-files', '-z', '--', *pathspecs)
    return tuple(sorted(path for path in listing.split('\0') if path))


def tracked_python_files(root: Path) -> tuple[str, ...]:
    """Every ``.py`` file git tracks under *root*; see ``tracked_files``."""
    return tracked_files(root, '*.py')


# ---------------------------------------------------------------------------
# The workspace domain: each member's tracked .py files, classified.


class FileKind(enum.StrEnum):
    SRC = 'src'
    TESTS = 'tests'


@dataclasses.dataclass(frozen=True)
class MemberRoots:
    """Where one member's source and test files live, repo-relative.

    One rule for real and pseudo members alike: a path under ``tests_root`` is a
    test file, else a path under ``src_root`` is a source file named by its
    dotted path below that root, else it is not the member's.
    """

    name: str
    pseudo: bool
    src_root: str | None
    tests_root: str

    @classmethod
    def declared(cls, name: str) -> MemberRoots:
        return cls(name=name, pseudo=False, src_root=f'{name}/src', tests_root=f'{name}/tests')

    def kind_of(self, path: str) -> FileKind | None:
        if path.startswith(self.tests_root + '/'):
            return FileKind.TESTS
        if self.src_root is not None and path.startswith(self.src_root + '/'):
            return FileKind.SRC
        return None

    def import_name(self, path: str) -> str | None:
        """The dotted import name of source file *path*; None for anything else.

        A package's ``__init__.py`` is named by its package.
        """
        if (
            self.src_root is None
            or not path.endswith('.py')
            or self.kind_of(path) is not FileKind.SRC
        ):
            return None
        dotted = path[len(self.src_root) + 1:].removesuffix('.py').replace('/', '.')
        return dotted.removesuffix('.__init__')


def src_module_name(path: str) -> str | None:
    """The import name of *path* as a source file of the member its first
    segment names, by ``MemberRoots.import_name``; None outside a ``<member>/src/``."""
    return MemberRoots.declared(path.partition('/')[0]).import_name(path)


@dataclasses.dataclass(frozen=True)
class DomainFile:
    path: str
    blob: str
    kind: FileKind
    import_name: str | None


@dataclasses.dataclass(frozen=True)
class DomainMember:
    name: str
    pseudo: bool
    files: tuple[DomainFile, ...]


#: The repo's two Python roots that are not workspace members, described by the
#: same rule as a member: plans/quality-metrics-snapshot-prd.md decision 3.
PSEUDO_MEMBERS: tuple[MemberRoots, ...] = (
    MemberRoots(name='scripts', pseudo=True, src_root='scripts', tests_root='scripts/tests'),
    MemberRoots(name='tests', pseudo=True, src_root=None, tests_root='tests'),
)


_MEMBERS_KEY = '[tool.uv.workspace].members'


def _load_toml(path: Path) -> dict[str, object]:
    try:
        return tomllib.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError) as exc:
        raise MetricsError(
            f'{path}: could not be read -- {exc.__class__.__name__}: {exc}'
        ) from exc
    except tomllib.TOMLDecodeError as exc:
        raise MetricsError(f'{path}: is not valid TOML -- {exc}') from exc


def _member_list(path: Path, config: dict[str, object]) -> list[str]:
    found: object = config
    for key in ('tool', 'uv', 'workspace', 'members'):
        found = found.get(key) if isinstance(found, dict) else None
    if (
        not isinstance(found, list)
        or not found
        or not all(isinstance(member, str) for member in found)
    ):
        raise MetricsError(
            f'{path}: {_MEMBERS_KEY} must be a non-empty list of member directory '
            f'names; found {found!r}'
        )
    return found


def _declared_members(root: Path) -> tuple[str, ...]:
    """``[tool.uv.workspace].members`` of *root*'s pyproject.toml.

    Each must be a plain directory name (uv's glob expansion cannot be
    reproduced from the index), listed once, and not a pseudo-member's name.
    """
    path = root / 'pyproject.toml'
    members = _member_list(path, _load_toml(path))
    globbed = [member for member in members if any(char in member for char in '*?[')]
    if globbed:
        raise MetricsError(
            f'{path}: {_MEMBERS_KEY} holds glob member(s) {globbed!r}, which are not '
            'supported -- list each member directory by name'
        )
    duplicates = sorted({member for member in members if members.count(member) > 1})
    if duplicates:
        raise MetricsError(f'{path}: {_MEMBERS_KEY} lists duplicate member(s) {duplicates!r}')
    collisions = sorted(set(members) & {pseudo.name for pseudo in PSEUDO_MEMBERS})
    if collisions:
        raise MetricsError(
            f'{path}: {_MEMBERS_KEY} names {collisions!r}, which collide with the '
            'pseudo-member of the same name'
        )
    return tuple(members)


def _tracked_blobs(root: Path, *pathspecs: str) -> tuple[tuple[str, str], ...]:
    """``(path, blob sha)`` of every index entry matching *pathspecs*.

    An unmerged entry is refused: a conflicted path has one blob per stage, so it
    has no single content to measure and would be counted once per stage.
    """
    listing = _git_output(root, 'ls-files', '-s', '-z', '--', *pathspecs)
    entries: list[tuple[str, str]] = []
    unmerged: set[str] = set()
    for record in filter(None, listing.split('\0')):
        meta, path = record.split('\t', 1)
        _mode, blob, stage = meta.split(' ')
        if stage != '0':
            unmerged.add(path)
            continue
        entries.append((path, blob))
    if unmerged:
        raise MetricsError(
            f'{root}: the index holds unmerged path(s) {sorted(unmerged)!r}; resolve '
            'the merge before measuring'
        )
    return tuple(entries)


def _domain_member(
    member: MemberRoots, entries: tuple[tuple[str, str], ...]
) -> DomainMember:
    files = tuple(
        DomainFile(path=path, blob=blob, kind=kind, import_name=member.import_name(path))
        for path, blob in sorted(entries)
        if (kind := member.kind_of(path)) is not None
    )
    return DomainMember(name=member.name, pseudo=member.pseudo, files=files)


def _require_declared_members_populated(domain: tuple[DomainMember, ...]) -> None:
    """A DECLARED member with no files would read as measured and clean.

    A pseudo-member is a convention, not a declaration, so it may be empty.
    """
    empty = [member.name for member in domain if not member.pseudo and not member.files]
    if empty:
        raise MetricsError(
            '; '.join(
                f'{name} has no tracked .py under {name}/src or {name}/tests'
                for name in empty
            )
        )


def workspace_domain(root: Path) -> tuple[DomainMember, ...]:
    """Every workspace member's tracked ``.py`` files, classified src or tests.

    The member list's one home is the root pyproject.toml's
    ``[tool.uv.workspace].members``, followed by ``PSEUDO_MEMBERS``. Files come
    from the index, not the disk, so an untracked file is outside the domain, as
    is every file under no member's src or tests root (``hooks/`` included).
    """
    _require_work_tree_top(root)
    members = (
        *(MemberRoots.declared(name) for name in _declared_members(root)),
        *PSEUDO_MEMBERS,
    )
    entries = _tracked_blobs(root, '*.py')
    domain = tuple(_domain_member(member, entries) for member in members)
    _require_declared_members_populated(domain)
    return domain
