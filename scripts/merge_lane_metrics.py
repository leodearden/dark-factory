#!/usr/bin/env python3
"""Measure the merge-lane cluster, and ratchet every measure against a committed baseline.

PRD ``plans/merge-lane-quality-prd.md`` task alpha. The measures are the ones
the PRD's Background table quotes -- file lines and prose lines, per-function
cognitive complexity, function-local (reach-back) imports, re-export shim names,
distinct test patch targets into lane internals, and private-attribute reads
from tests. The committed baseline lives at
``orchestrator/tests/merge_lane_ratchet_baseline.json`` and the gate that
enforces it is ``orchestrator/tests/test_merge_lane_ratchet.py``.

Ratchet contract: the gate FAILS on any measure that rises above its baseline;
raising a measure is not allowed, only lowering one is. Equality is permitted --
this is a ratchet, not a day-one gate. The two ceilings (``FILE_LINE_CEILING``,
``NEW_FUNCTION_COGNITIVE_CEILING``) apply only to paths and qualnames ABSENT
from the baseline, so the grandfathering can only ever shrink.

INV-11, no silent fail-soft -- and note the polarity
----------------------------------------------------
The sibling guards under ``orchestrator/tests/`` all fail SOFT on an
unparseable file (``_find_merge_queue_private_patches`` returns ``[]``;
``_scans_whole_tree_py`` returns False), deliberately: they sweep the whole
tree, so a mid-edit or intentionally malformed fixture file must not redden a
guard about something else. This instrument does the OPPOSITE for its own
cluster, and the difference is not an oversight to be "fixed" by copying the
neighbours. Those guards measure OTHER people's files; this one measures a
fixed, named cluster where a path it cannot read IS the finding. Concretely:

* a ``CLUSTER_PATHS`` LITERAL that is missing, unreadable or unparseable ->
  ``MetricsError`` naming the path and the cause;
* a ``CLUSTER_PATHS`` GLOB expanding to zero -> fine (that is
  ``merge_lane/**`` until PRD task zeta1 lands);
* complexipy absent, or resolving outside ``COMPLEXIPY_REQUIRED`` ->
  ``MetricsError`` naming the tool and the version found;
* the whole-of-``orchestrator/tests`` sweep keeps the siblings' per-file
  fail-soft polarity, but records every skipped file in
  ``Enumeration.unreadable`` and marks the enumeration incomplete, and
  ``check_against_baseline`` REFUSES to compare an incomplete enumeration.

That last clause is what makes "a partial enumeration is distinguishable from a
complete one in the RESULT, not only in a log line" true rather than aspirational:
``{requested, resolved, unreadable, complete}`` travels in the report AND in the
committed baseline, and ``--report`` prints it as a row.

Exit codes
----------
0  clean -- measures taken, and (under ``--check``) no ratchet violation.
1  ratchet violations -- one or more measures rose above the baseline, or a new
   file/function exceeded a ceiling. Violations are printed to stderr.
2  instrument failure (``MetricsError``) -- an unparseable cluster file,
   complexipy missing or out of range, a missing/malformed baseline, or a
   baseline whose recorded parameters no longer match this tree. Deliberately
   distinct from 1 so a broken instrument is never mistaken either for a clean
   tree or for a real regression.
"""
from __future__ import annotations

import ast
import dataclasses
import tokenize
from io import StringIO
from pathlib import Path

# NOTE: module scope is stdlib-only ON PURPOSE. complexipy and radon are
# imported lazily inside the functions that need them (see
# ``_import_complexipy``), so this module stays importable and type-checkable in
# environments whose dev group does not carry them -- notably the ``shared``
# project, which owns the ``ruff check scripts/`` and ``pyright scripts/`` gates
# for this file. Laziness is not softness: the moment a measure actually needs
# the tool, a missing or wrong-version one raises MetricsError with a named
# cause (INV-11).


class MetricsError(Exception):
    """The instrument could not take a measurement it was asked for.

    Always raised with the offending path / tool / version named in the message.
    Mapped to exit code 2 at the ``main()`` boundary, categorically apart from
    the exit-1 "the tree regressed" outcome.
    """


# ---------------------------------------------------------------------------
# The cluster spec -- SPOT for PRD Appendix A.

#: Repo-relative literal paths and glob patterns naming the merge-lane cluster.
#: This tuple IS Appendix A; editing it forces a deliberate baseline
#: regeneration, because ``check_against_baseline`` compares the baseline's
#: recorded ``params.cluster_paths`` against it and raises on a mismatch.
CLUSTER_PATHS: tuple[str, ...] = (
    'orchestrator/src/orchestrator/merge_queue.py',
    'orchestrator/src/orchestrator/merge_gates.py',
    'orchestrator/src/orchestrator/merge_types.py',
    'orchestrator/src/orchestrator/merge_shadow.py',
    'orchestrator/src/orchestrator/merge_liveness.py',
    'orchestrator/src/orchestrator/merge_disposition.py',
    'orchestrator/src/orchestrator/merge_queue_store.py',
    'orchestrator/src/orchestrator/merge_completion.py',
    'orchestrator/src/orchestrator/merge_drift.py',
    'orchestrator/src/orchestrator/merge_speculation_controller.py',
    'orchestrator/src/orchestrator/merge_request_ledger.py',
    'orchestrator/src/orchestrator/merge_skew_tripwire.py',
    'orchestrator/src/orchestrator/lane_lifecycle.py',
    'orchestrator/src/orchestrator/offline_lane.py',
    'orchestrator/src/orchestrator/warm_lane_pool.py',
    'orchestrator/src/orchestrator/landing_evidence.py',
    'orchestrator/src/orchestrator/landed_outbox.py',
    'orchestrator/src/orchestrator/recover_main.py',
    # Measured (lines, cognitive) but exempt from the file-size ceiling -- see
    # SIZE_CEILING_EXEMPT.
    'orchestrator/src/orchestrator/git_ops.py',
    # Empty until PRD task zeta1 creates the package. A glob matching nothing is
    # NOT a failure; a literal matching nothing is.
    'orchestrator/src/orchestrator/merge_lane/**/*.py',
    'orchestrator/tests/_serial_merge_worker.py',
    'orchestrator/tests/_merge_queue_harness.py',
    'orchestrator/tests/conftest.py',
)

#: Paths measured and ratcheted, but exempt from ``FILE_LINE_CEILING``.
#: PRD decision 9: ``git_ops.py`` is the git engine for warm lanes, the
#: scheduler and recovery -- not only the merge lane -- and its split is a
#: follow-up PRD. The exemption is scoped to the CEILING alone: git_ops.py's
#: 14,721 lines are still frozen by the ratchet, so they cannot grow unwatched.
SIZE_CEILING_EXEMPT: frozenset[str] = frozenset(
    {'orchestrator/src/orchestrator/git_ops.py'}
)

#: A file ABSENT from the baseline may not exceed this many lines.
FILE_LINE_CEILING = 1500

#: A function ABSENT from the baseline may not exceed this cognitive complexity.
NEW_FUNCTION_COGNITIVE_CEILING = 15


# ---------------------------------------------------------------------------
# Enumeration -- completeness carried in the RESULT, per INV-11.


@dataclasses.dataclass(frozen=True)
class Enumeration:
    """Which paths were asked for, which were measured, and which were skipped.

    ``complete`` is the single signal every consumer checks. It is False
    whenever anything landed in ``unreadable``, and ``check_against_baseline``
    raises rather than compares when it is False -- so a degraded sweep can
    never masquerade as a clean tree.
    """

    requested: tuple[str, ...]
    resolved: tuple[str, ...]
    unreadable: tuple[str, ...]
    complete: bool

    def to_dict(self) -> dict[str, object]:
        return {
            'requested': list(self.requested),
            'resolved': list(self.resolved),
            'unreadable': list(self.unreadable),
            'complete': self.complete,
        }


def resolve_cluster_paths(root: Path) -> Enumeration:
    """Resolve ``CLUSTER_PATHS`` against *root* into a complete Enumeration.

    Raises ``MetricsError`` naming the path when a LITERAL entry is missing, is
    not a regular file, or cannot be read. A GLOB entry expanding to zero paths
    is accepted (that is ``merge_lane/**`` until PRD task zeta1).
    """
    resolved: list[str] = []
    for entry in CLUSTER_PATHS:
        if '*' in entry:
            for match in sorted(root.glob(entry)):
                if match.is_file():
                    resolved.append(match.relative_to(root).as_posix())
            continue
        candidate = root / entry
        if not candidate.exists():
            raise MetricsError(
                f'cluster path {entry!r} does not exist under {root} -- '
                'CLUSTER_PATHS is the SPOT source of PRD Appendix A; if the '
                'file was renamed or removed, update CLUSTER_PATHS and '
                'regenerate the baseline in the same commit'
            )
        if not candidate.is_file():
            raise MetricsError(
                f'cluster path {entry!r} is not a regular file under {root}'
            )
        try:
            candidate.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as exc:
            raise MetricsError(
                f'cluster path {entry!r} could not be read: {exc.__class__.__name__}: {exc}'
            ) from exc
        resolved.append(entry)
    return Enumeration(
        requested=CLUSTER_PATHS,
        resolved=tuple(resolved),
        unreadable=(),
        complete=True,
    )


# ---------------------------------------------------------------------------
# Source helpers shared by the per-file measures.


def _parse(source: str, *, path: str) -> ast.Module:
    """Parse *source*, translating a SyntaxError into a named MetricsError.

    INV-11: an unparseable CLUSTER file is the finding, never a skipped measure.
    Callers sweeping files OUTSIDE the cluster (the 559-file test tree) catch
    this and record the path in ``Enumeration.unreadable`` instead.
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


def _read_source(root: Path, relpath: str) -> str:
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


def _docstring_lines(tree: ast.Module) -> set[int]:
    """Line numbers spanned by every docstring in *tree*.

    A docstring is the FIRST body element of a module, class or function when it
    is a bare string expression -- exactly Python's own rule, so a second string
    expression in the same body is code, not prose.
    """
    lines: set[int] = set()
    holders: tuple[type[ast.AST], ...] = (
        ast.Module,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
    )
    for node in ast.walk(tree):
        if not isinstance(node, holders):
            continue
        body = getattr(node, 'body', None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            end = first.end_lineno if first.end_lineno is not None else first.lineno
            lines.update(range(first.lineno, end + 1))
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
    tree = _parse(source, path=path)
    prose = _docstring_lines(tree) | _comment_lines(source, path=path)
    return FileSizeMeasures(lines=len(source.splitlines()), prose_lines=len(prose))


# ---------------------------------------------------------------------------
# Structural import measures.

_FUNCTION_NODES: tuple[type[ast.AST], ...] = (ast.FunctionDef, ast.AsyncFunctionDef)


def function_local_imports(source: str, *, path: str) -> int:
    """Count import statements that live inside a function body.

    These are the lane's reach-back imports -- the function-local
    ``from orchestrator.merge_queue import ...`` sites that exist to break
    import cycles, plus every other deferred import in the same shape. The PRD's
    ceiling for this measure is zero.

    Counted per STATEMENT, not per bound name, and deduped by node identity so a
    nested function's import is counted once rather than once per enclosing
    function. AST-based, so a docstring quoting an import statement -- which the
    satellite modules' reach-back notes do verbatim -- is never counted.
    """
    tree = _parse(source, path=path)
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
    nothing nameable. Returns the bound names (``asname or name``) sorted and
    deduped.
    """
    tree = _parse(source, path=path)
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    # An attribute chain rooted at the binding (`B.attr`) also uses it, and so
    # does an `__all__` listing -- but `__all__` entries are string constants,
    # not Names, and a module that re-exports via `__all__` is still a shim by
    # this measure's definition, which is the reading the PRD's ceiling wants.
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
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
    is stdlib-only at import time (see the note at the top), and a two-clause
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

    Called once up front by ``build_report`` so a wrong-version environment
    fails immediately with a named cause rather than after a 250-second
    measurement whose numbers would be wrong anyway.
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

    Lazy so ``scripts/merge_lane_metrics.py`` stays importable and
    type-checkable under the ``shared`` project that owns its ruff/pyright
    gates, whose dev group carries neither complexipy nor radon. Lazy is not
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


def cognitive_complexity(path: Path) -> dict[str, int]:
    """Per-function cognitive complexity of *path*, keyed by complexipy qualname.

    complexipy already emits ``Class::method`` for methods, so the key needs no
    post-processing. A module with no functions yields an empty map -- that is a
    real measurement, not a skipped one.
    """
    result = _file_complexity(path)
    return {function.name: function.complexity for function in result.functions}


def file_cognitive_total(path: Path) -> int:
    """complexipy's whole-file cognitive total for *path*.

    Reported and ratcheted alongside the per-function map, because the file
    total also counts module-level control flow that belongs to no function.
    """
    return int(_file_complexity(path).complexity)


def maintainability_index(source: str, *, path: str) -> float:
    """radon's maintainability index for *source*, in [0, 100].

    REPORTED, never ratcheted: MI is a derived composite (Halstead volume,
    cyclomatic complexity, SLOC, comment ratio) that already reads 0.00 for both
    merge_queue.py and git_ops.py, so it has no headroom left to ratchet against
    and would only ever restate what the line and cognitive measures already
    say. It earns its place in ``--report`` as the PRD Background table's
    "Maintainability index (radon) | 0" row -- and it is what makes the `radon`
    dev-group entry genuinely exercised rather than dead weight.
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
