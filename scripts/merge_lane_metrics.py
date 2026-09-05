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

import argparse
import ast
import contextlib
import dataclasses
import json
import os
import sys
import tempfile
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
    nothing nameable. ``from __future__ import ...`` is likewise excluded: it is
    a compiler directive, not a name a downstream module could import, and ruff
    explicitly never flags it under F401 -- counting it would both inflate every
    baseline and make the ratchet REWARD deleting a future import, which silently
    changes runtime annotation semantics. Returns the bound names
    (``asname or name``) sorted and deduped.
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


# ---------------------------------------------------------------------------
# Test-suite patch targets into lane internals.
#
# The two detector shapes below are PORTED from
# orchestrator/tests/test_merge_queue_reachback_patch_guard.py --
# `_merge_queue_module_aliases()` and the is_setattr / is_dotted_patch /
# is_bare_patch / is_patch_object call classification inside
# `_find_merge_queue_private_patches()` -- with two deliberate generalisations:
# the alias helper takes a SET of lane module paths rather than the single
# hardcoded `orchestrator.merge_queue`, and that guard's `forbidden` filter is
# dropped so ALL leaves are counted rather than only the satellite-private ones.
# This measure therefore subsumes the guard's allowlist as a COUNT, which is
# what lets PRD task delta delete the guard once the count reaches zero.

#: Dotted module paths whose attributes count as lane internals when patched.
#: `merge_lane` is here from day one, before the package exists, precisely so
#: PRD task zeta2's `git mv` cannot make the count read 0 by relocation.
LANE_PATCH_MODULES: tuple[str, ...] = (
    'orchestrator.merge_queue',
    'orchestrator.merge_lane',
)


def _lane_module_aliases(tree: ast.AST) -> set[str]:
    """Names bound directly to a lane module object anywhere in *tree*.

    e.g. ``import orchestrator.merge_queue as mq`` or
    ``from orchestrator import merge_lane``. Used to recognise the object-path
    idiom, which targets the identical lookup site as the string-path form
    without embedding the module path as a string constant.
    """
    leaves = {path.rsplit('.', 1)[-1] for path in LANE_PATCH_MODULES}
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in LANE_PATCH_MODULES and alias.asname:
                    aliases.add(alias.asname)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module == 'orchestrator'
            and not node.level
        ):
            for alias in node.names:
                if alias.name in leaves:
                    aliases.add(alias.asname or alias.name)
    return aliases


def patch_targets(source: str, *, path: str = '<source>') -> set[str]:
    """Distinct leaf names patched through a lane module path in *source*.

    Distinct NAMES, not call sites: the PRD's measure is "79 distinct names"
    across "1,111 call sites", and the ratchet freezes the former.

    AST-based, so a docstring or comment quoting the dotted path -- which the
    satellite module docstrings and the reachback guard's own ALLOWLIST literal
    both do -- is never mistaken for a real patch site.
    """
    tree = _parse(source, path=path)
    aliases = _lane_module_aliases(tree)
    leaves = {module.rsplit('.', 1)[-1] for module in LANE_PATCH_MODULES}

    def _is_lane_ref(expr: ast.expr) -> bool:
        if isinstance(expr, ast.Name):
            return expr.id in aliases
        # The bare attribute chain `orchestrator.merge_queue`. Anchored on the
        # `orchestrator` root so an unrelated `workflow.merge_queue` attribute
        # that merely shares the leaf name is not counted.
        return (
            isinstance(expr, ast.Attribute)
            and expr.attr in leaves
            and isinstance(expr.value, ast.Name)
            and expr.value.id == 'orchestrator'
        )

    targets: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
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

        leaf: str | None = None

        # String-path form: the dotted path IS the first positional argument.
        if is_setattr or is_dotted_patch or is_bare_patch:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                for module in LANE_PATCH_MODULES:
                    prefix = module + '.'
                    if first.value.startswith(prefix):
                        leaf = first.value[len(prefix):]
                        break

        # Object-path form: first arg is a lane module reference, second is the
        # leaf name string.
        if leaf is None and (is_setattr or is_patch_object) and len(node.args) >= 2:
            target, name_arg = node.args[0], node.args[1]
            if (
                _is_lane_ref(target)
                and isinstance(name_arg, ast.Constant)
                and isinstance(name_arg.value, str)
            ):
                leaf = name_arg.value

        if leaf:
            targets.add(leaf)
    return targets


# ---------------------------------------------------------------------------
# Private-attribute reads from tests.
#
# THIS MEASURE IS DELIBERATELY RECEIVER-AGNOSTIC. It counts every `_x` attribute
# access in a lane-importing test file except on the bare `self`/`cls`, and it
# does NOT maintain a list of blessed receiver variable names (`worker`, `mq`,
# ...). Two reasons, and the first is the decisive one:
#
# (1) A receiver-name allowlist is a single SHARED list that all ten of PRD
#     gamma1..gamma10 would have to edit concurrently -- the same rebase-conflict
#     hazard the per-path baseline format exists to avoid.
# (2) It silently UNDER-counts the moment a test uses a receiver name nobody
#     listed, which is how a ratchet rots into a vacuous pass. Over-counting is
#     the safe direction here: a ratchet only ever refuses to let a number RISE,
#     so a superset costs a little extra friction and never lets a regression
#     through.
#
# The `self`/`cls` exclusion is a STRUCTURAL predicate, not a name list: a test
# class's own helpers are not lane internals, which is the distinction the
# measure is actually about. Note it excludes only the BARE receiver, so
# `self.worker._x` still counts.
#
# Measured magnitudes on this tree: 18,952 private attribute nodes across all
# 559 test files, 9,355 restricted to the 167 lane-importing ones. (The PRD
# Background table's 5,735 came from a narrower ad-hoc receiver set.)

_SELF_RECEIVERS = frozenset({'self', 'cls'})


def lane_module_names() -> frozenset[str]:
    """Dotted module names of the cluster, DERIVED from ``CLUSTER_PATHS``.

    Derived rather than hand-listed so the lane-importing predicate and
    Appendix A can never drift apart.
    """
    names: set[str] = set()
    for entry in CLUSTER_PATHS:
        if '*' in entry or not entry.endswith('.py'):
            continue
        parts = entry[: -len('.py')].split('/')
        if 'src' in parts:
            names.add('.'.join(parts[parts.index('src') + 1:]))
    return frozenset(names)


def imports_lane_module(source: str, *, path: str) -> bool:
    """True when *source* imports any cluster module, or anything under
    ``orchestrator.merge_lane``."""
    tree = _parse(source, path=path)
    lane_names = lane_module_names()
    lane_leaves = {name.rsplit('.', 1)[-1] for name in lane_names}

    def _is_lane(dotted: str) -> bool:
        return dotted in lane_names or dotted == 'orchestrator.merge_lane' or dotted.startswith(
            'orchestrator.merge_lane.'
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_is_lane(alias.name) for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            if _is_lane(node.module):
                return True
            # `from orchestrator import merge_gates` binds the module itself.
            if node.module == 'orchestrator' and any(
                alias.name in lane_leaves or alias.name == 'merge_lane'
                for alias in node.names
            ):
                return True
    return False


def private_reads(source: str, *, path: str) -> int:
    """Count accesses of a single-underscore attribute in *source*.

    Writes count too: both directions couple the test to an internal name, which
    is the coupling the measure exists to shrink. Dunders are excluded (Python
    protocol, not lane internals) and so is the bare ``self``/``cls`` receiver.
    """
    tree = _parse(source, path=path)
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


def test_file_measures(source: str, *, path: str) -> dict[str, object] | None:
    """The two test-suite measures for *source*, or None when it is not
    lane-importing.

    Returning None rather than a zeroed dict makes the exclusion a property of
    the MEASURE rather than of the caller, so a non-lane-importing file can
    never be summed into the report by accident.
    """
    if not imports_lane_module(source, path=path):
        return None
    return {
        'patch_targets': sorted(patch_targets(source, path=path)),
        'private_reads': private_reads(source, path=path),
    }


# ---------------------------------------------------------------------------
# Report assembly, and DERIVED totals.

SCHEMA_VERSION = 1

#: Subdirectory swept for the two test-suite measures.
TESTS_ROOT = 'orchestrator/tests'


def build_report(root: Path) -> dict[str, object]:
    """Measure the whole cluster plus the test suite under *root*.

    Note there is NO ``totals`` key: cluster-wide totals are DERIVED at check
    time by ``derive_totals`` from the stored per-path maps. PRD gamma1..gamma10
    run in parallel and each lowers only its own group's contribution, so a
    shared totals line would conflict on every one of ten rebases -- while
    deriving preserves the anti-rename-gaming property in full (see
    ``derive_totals``) and keeps one number in one place.
    """
    # Up front, before any measurement: a wrong-version engine must fail
    # immediately with a named cause rather than after a 250-second run whose
    # numbers would be wrong anyway.
    version = require_complexipy()
    enumeration = resolve_cluster_paths(root)

    files: dict[str, object] = {}
    functions: dict[str, int] = {}
    for relpath in enumeration.resolved:
        source = _read_source(root, relpath)
        size = file_size_measures(source, path=relpath)
        target = root / relpath
        files[relpath] = {
            'lines': size.lines,
            'prose_lines': size.prose_lines,
            'cognitive': file_cognitive_total(target),
            'function_local_imports': function_local_imports(source, path=relpath),
            'reexport_names': len(reexport_names(source, path=relpath)),
        }
        for qualname, score in cognitive_complexity(target).items():
            functions[f'{relpath}::{qualname}'] = score

    tests, test_enumeration = _sweep_test_tree(root)

    return {
        'schema_version': SCHEMA_VERSION,
        'params': {
            'complexipy_version': version,
            'cluster_paths': list(CLUSTER_PATHS),
            'file_line_ceiling': FILE_LINE_CEILING,
            'new_function_cognitive_ceiling': NEW_FUNCTION_COGNITIVE_CEILING,
        },
        'enumeration': _merge_enumerations(enumeration, test_enumeration).to_dict(),
        'files': dict(sorted(files.items())),
        'functions': dict(sorted(functions.items())),
        'tests': dict(sorted(tests.items())),
    }


def _sweep_test_tree(root: Path) -> tuple[dict[str, object], Enumeration]:
    """Measure every lane-importing file under ``orchestrator/tests``.

    THIS sweep keeps the sibling guards' per-file fail-SOFT polarity -- an
    unrelated mid-edit test file must not redden the ratchet, which is the
    misattribution every neighbouring guard exists to avoid. What is NOT soft is
    the record: each skipped file lands in ``Enumeration.unreadable``, the
    enumeration goes incomplete, and ``check_against_baseline`` then refuses to
    compare at all. That split is the INV-11 seam between "this cluster file is
    unmeasurable, which IS the finding" and "some unrelated test file is
    mid-edit".
    """
    tests: dict[str, object] = {}
    requested: list[str] = []
    unreadable: list[str] = []
    for path in sorted((root / TESTS_ROOT).rglob('*.py')):
        relpath = path.relative_to(root).as_posix()
        requested.append(relpath)
        try:
            source = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            unreadable.append(relpath)
            continue
        try:
            measures = test_file_measures(source, path=relpath)
        except MetricsError:
            unreadable.append(relpath)
            continue
        if measures is not None:
            tests[relpath] = measures
    return tests, Enumeration(
        requested=tuple(requested),
        resolved=tuple(sorted(tests)),
        unreadable=tuple(unreadable),
        complete=not unreadable,
    )


def _merge_enumerations(cluster: Enumeration, tests: Enumeration) -> Enumeration:
    return Enumeration(
        requested=cluster.requested + tests.requested,
        resolved=cluster.resolved + tests.resolved,
        unreadable=cluster.unreadable + tests.unreadable,
        complete=cluster.complete and tests.complete,
    )


#: Per-file measures summed into a cluster total by ``derive_totals``.
_SUMMED_FILE_MEASURES = (
    'lines',
    'prose_lines',
    'cognitive',
    'function_local_imports',
    'reexport_names',
)


def derive_totals(report: dict) -> dict[str, int]:
    """Cluster-wide totals, DERIVED by summing *report*'s per-path maps.

    Pure: *report* is never mutated.

    Deriving rather than storing is what makes the ten parallel gamma branches
    rebase cleanly, AND it preserves the anti-rename-gaming property exactly. A
    total computed by summing stored per-path baselines is unchanged when 500
    lines move from ``merge_queue.py`` to a brand-new path, so the ratchet still
    catches the move -- which a per-path-only comparison would not, since the
    new path is simply absent from the baseline.

    ``patch_targets`` is the size of the UNION across files, matching the PRD's
    "distinct names" measure rather than a sum of per-file counts.
    """
    files = report.get('files', {})
    totals = {
        measure: sum(int(entry.get(measure, 0)) for entry in files.values())
        for measure in _SUMMED_FILE_MEASURES
    }
    tests = report.get('tests', {})
    totals['private_reads'] = sum(
        int(entry.get('private_reads', 0)) for entry in tests.values()
    )
    union: set[str] = set()
    for entry in tests.values():
        union.update(entry.get('patch_targets', ()))
    totals['patch_targets'] = len(union)
    return totals


# ---------------------------------------------------------------------------
# Baseline serialization.
#
# WHY A HAND-ROLLED WRITER instead of json.dumps(indent=2). PRD gamma1..gamma10
# run in PARALLEL, each lowering only its own group's numbers and rebasing
# through the merge lane. indent=2 spreads one path's five measures across six
# lines, so two branches editing two unrelated paths land inside one diff hunk
# and conflict. Emitting each files/functions/tests entry on exactly ONE line
# makes those ten edits disjoint hunks that rebase cleanly. The property is
# asserted by a test rather than left to formatting habit -- see
# ``TestRenderBaseline::test_every_per_path_entry_occupies_exactly_one_line``.

#: Emitted as the baseline's leading key, so the rule is in the file a reader
#: is about to "fix" rather than only in a docstring they will not open.
BASELINE_README = (
    'This is a committed RATCHET BASELINE for the merge-lane cluster '
    '(PRD plans/merge-lane-quality-prd.md, task alpha). Every measure here is '
    'frozen at its value on the commit that recorded it: the gate '
    'orchestrator/tests/test_merge_lane_ratchet.py FAILS on any measure that '
    'RISES above these numbers. Equality is fine, lowering is the point. NEVER '
    'regenerate this file merely to make a test pass -- that silently widens '
    'the ratchet for every downstream task. A task that legitimately LOWERS a '
    'measure regenerates the baseline in the SAME commit: '
    'python scripts/merge_lane_metrics.py --write-baseline '
    'orchestrator/tests/merge_lane_ratchet_baseline.json'
)

#: The three per-path maps whose entries get one line each.
_PER_PATH_SECTIONS: tuple[str, ...] = ('files', 'functions', 'tests')


def _render_section(name: str, mapping: dict) -> str:
    """Render one per-path map with exactly one line per entry, key-sorted."""
    if not mapping:
        return f'  {json.dumps(name)}: {{}}'
    rows = ',\n'.join(
        f'    {json.dumps(key)}: {json.dumps(mapping[key], sort_keys=True)}'
        for key in sorted(mapping)
    )
    return f'  {json.dumps(name)}: {{\n{rows}\n  }}'


def render_baseline(report: dict) -> str:
    """Serialize *report* as the committed baseline's exact bytes.

    Idempotent: ``render_baseline(json.loads(render_baseline(r)))`` reproduces
    the same text, so regenerating a baseline from a baseline is a no-op rather
    than a churned file full of manufactured conflicts. The ``_README`` key is
    emitted from ``BASELINE_README`` and any inbound one is dropped, which is
    what makes that hold across a round trip.
    """
    entries = [f'  "_README": {json.dumps(BASELINE_README)}']
    for key, value in report.items():
        if key == '_README':
            continue
        if key in _PER_PATH_SECTIONS and isinstance(value, dict):
            entries.append(_render_section(key, value))
        else:
            # Top-level scalars and the small params/enumeration blocks are
            # ordinary pretty-printed JSON, shifted one level in.
            block = json.dumps(value, indent=2).replace('\n', '\n  ')
            entries.append(f'  {json.dumps(key)}: {block}')
    return '{\n' + ',\n'.join(entries) + '\n}\n'


def _atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* via a same-directory tempfile plus os.replace.

    Mirrors ``scripts/census_tagger_debris.py::_atomic_write_text``. A reader --
    or a concurrently running ratchet -- can never observe a half-written
    baseline, and a failed write leaves the previous file intact rather than
    truncated.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        suffix='.tmp', prefix=f'{path.name}.', dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def write_baseline(path: Path, report: dict) -> Path:
    """Render *report* and write it to *path* atomically.

    THE TEXT IS RENDERED BEFORE THE DESTINATION IS TOUCHED, so a rendering
    failure leaves the committed baseline byte-for-byte intact instead of
    truncated -- a truncated baseline would be a *widened* ratchet, the one
    failure mode this instrument must never produce silently.
    """
    text = render_baseline(report)
    target = Path(path)
    _atomic_write_text(target, text)
    return target


def load_baseline(path: Path) -> dict:
    """Load the committed baseline, failing HARD and by name (INV-11).

    A missing or malformed baseline is never an empty-baseline PASS. An empty
    baseline compares clean against every measure, which would disarm the
    ratchet for all twenty downstream PRD tasks while every gate stayed green --
    exactly the silent fail-soft INV-11 forbids.
    """
    target = Path(path)
    try:
        text = target.read_text(encoding='utf-8')
    except FileNotFoundError as exc:
        raise MetricsError(
            f'ratchet baseline {target} does not exist -- a missing baseline is '
            'a hard failure, never an empty-baseline pass; regenerate it with '
            '--write-baseline if this is the commit that introduces it'
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise MetricsError(
            f'ratchet baseline {target} could not be read: '
            f'{exc.__class__.__name__}: {exc}'
        ) from exc
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MetricsError(
            f'ratchet baseline {target} is not valid JSON: {exc}'
        ) from exc
    if not isinstance(loaded, dict):
        raise MetricsError(
            f'ratchet baseline {target} holds a '
            f'{type(loaded).__name__} at top level, expected a JSON object'
        )
    return loaded


# ---------------------------------------------------------------------------
# The ratchet comparator.
#
# PURE: two plain dicts in, a list of Violations out. No filesystem, no
# complexipy, no clock. That is what lets the pytest gate and the CLI's --check
# share ONE ratchet implementation (SPOT), and what lets the seeded-fixture
# self-test in test_merge_lane_ratchet.py pin every branch in microseconds
# instead of only ever reaching them through a 40-second measurement.

#: The synthetic key derived totals are reported under, so a totals violation
#: reads the same way a per-path one does.
CLUSTER_TOTAL_KEY = '<cluster>'


@dataclasses.dataclass(frozen=True, order=True)
class Violation:
    """One measure that rose above its baseline, or breached a ceiling."""

    measure: str
    key: str
    baseline: int
    current: int
    message: str


def _violation(measure: str, key: str, baseline: int, current: int) -> Violation:
    return Violation(
        measure=measure,
        key=key,
        baseline=baseline,
        current=current,
        message=(
            f'{measure} rose {baseline} -> {current} for {key} -- the merge-lane '
            'ratchet permits a measure to fall or hold, never to rise'
        ),
    )


def _total_violation(measure: str, baseline: int, current: int) -> Violation:
    name = f'total:{measure}'
    return Violation(
        measure=name,
        key=CLUSTER_TOTAL_KEY,
        baseline=baseline,
        current=current,
        message=(
            f'{name} rose {baseline} -> {current} for {CLUSTER_TOTAL_KEY} -- totals '
            'are DERIVED by summing the per-path baseline, so moving code to a '
            'new path does not lower them'
        ),
    )


def _section(report: dict, name: str) -> dict:
    value = report.get(name)
    return value if isinstance(value, dict) else {}


def _params(report: dict, which: str) -> dict:
    params = report.get('params')
    if not isinstance(params, dict):
        raise MetricsError(
            f'{which} report has no "params" block -- it was not produced by '
            'build_report, so there is nothing to state how it was measured'
        )
    return params


def _require_complete_enumeration(current: dict) -> None:
    enumeration = current.get('enumeration')
    if not isinstance(enumeration, dict):
        raise MetricsError(
            'current report has no "enumeration" block -- completeness must be '
            'legible in the RESULT, not only in a log line (INV-11)'
        )
    if enumeration.get('complete') is True:
        return
    unreadable = list(enumeration.get('unreadable', ()))
    raise MetricsError(
        'refusing to compare a PARTIAL measurement against the baseline: '
        f'{len(unreadable)} path(s) were skipped -- {unreadable}. A sweep that '
        'skipped files measures LOWER than the truth, so comparing it would '
        'read as a clean tree, or worse as an improvement worth writing into '
        'the baseline (INV-11).'
    )


def _require_matching_params(current: dict, baseline: dict) -> None:
    current_version = _params(current, 'current').get('complexipy_version')
    baseline_version = _params(baseline, 'baseline').get('complexipy_version')
    if current_version != baseline_version:
        raise MetricsError(
            f'complexipy version drift: the baseline was measured with '
            f'{baseline_version!r}, this run used {current_version!r}. Cognitive '
            'numbers are version-dependent -- the same merge_queue.py measures '
            '2031 at 3.0.0, 2092 at 5.0.0 and 2133 at 6.x/7.x -- so a silent '
            'drift rewrites every number at once and leaves the ratchet '
            'comparing two incomparable measurements. Pin the dev group '
            f'({COMPLEXIPY_REQUIRED}) or regenerate the baseline deliberately.'
        )

    recorded = list(_params(baseline, 'baseline').get('cluster_paths', ()))
    live = list(CLUSTER_PATHS)
    if recorded != live:
        added = [p for p in recorded if p not in live]
        removed = [p for p in live if p not in recorded]
        raise MetricsError(
            'the baseline\'s recorded cluster_paths no longer match '
            f'CLUSTER_PATHS. Recorded but no longer in the cluster: {added}. '
            f'In the cluster but not recorded: {removed}. Editing PRD Appendix A '
            'forces a deliberate baseline regeneration, so a path can never '
            'leave the cluster and take its frozen numbers with it.'
        )


def _check_files(current: dict, baseline: dict) -> list[Violation]:
    violations: list[Violation] = []
    current_files = _section(current, 'files')
    for path, base_entry in _section(baseline, 'files').items():
        entry = current_files.get(path)
        if entry is None:
            # The path is gone (deleted, or renamed by a task that moved code).
            # Not a per-path violation -- lowering is the point -- and the
            # derived totals are what catch a move that only relocated the mass.
            continue
        for measure in _SUMMED_FILE_MEASURES:
            was, now = int(base_entry.get(measure, 0)), int(entry.get(measure, 0))
            if now > was:
                violations.append(_violation(measure, path, was, now))
    return violations


def _check_functions(current: dict, baseline: dict) -> list[Violation]:
    current_functions = _section(current, 'functions')
    return [
        _violation('cognitive', key, int(was), int(current_functions[key]))
        for key, was in _section(baseline, 'functions').items()
        if key in current_functions and int(current_functions[key]) > int(was)
    ]


def _check_tests(current: dict, baseline: dict) -> list[Violation]:
    violations: list[Violation] = []
    current_tests = _section(current, 'tests')
    for path, base_entry in _section(baseline, 'tests').items():
        entry = current_tests.get(path)
        if entry is None:
            continue
        was = int(base_entry.get('private_reads', 0))
        now = int(entry.get('private_reads', 0))
        if now > was:
            violations.append(_violation('private_reads', path, was, now))
        # DISTINCT names, matching the PRD's measure: re-patching the same leaf
        # twice more in one file is not a new reach into lane internals.
        was_targets = len(set(base_entry.get('patch_targets', ())))
        now_targets = len(set(entry.get('patch_targets', ())))
        if now_targets > was_targets:
            violations.append(
                _violation('patch_targets', path, was_targets, now_targets)
            )
    return violations


def _check_totals(current: dict, baseline: dict) -> list[Violation]:
    current_totals = derive_totals(current)
    baseline_totals = derive_totals(baseline)
    return [
        _total_violation(measure, was, current_totals[measure])
        for measure, was in sorted(baseline_totals.items())
        if current_totals.get(measure, 0) > was
    ]


def _check_ceilings(current: dict, baseline: dict) -> list[Violation]:
    """The two ceilings, applied ONLY to keys absent from the baseline.

    Ceilings-on-new-only is what makes this a ratchet rather than a day-one
    gate: 62 lane functions already exceed cognitive 15 and merge_queue.py is
    fourteen times the line ceiling on the introducing commit. Grandfathering
    can only ever shrink, because every new key is held to the ceiling.
    """
    violations: list[Violation] = []
    baseline_files = _section(baseline, 'files')
    for path, entry in _section(current, 'files').items():
        if path in baseline_files or path in SIZE_CEILING_EXEMPT:
            continue
        lines = int(entry.get('lines', 0))
        if lines > FILE_LINE_CEILING:
            violations.append(
                Violation(
                    measure='new_file_over_ceiling',
                    key=path,
                    baseline=FILE_LINE_CEILING,
                    current=lines,
                    message=(
                        f'new_file_over_ceiling: {path} is {lines} lines, above '
                        f'the {FILE_LINE_CEILING}-line ceiling that applies to '
                        'paths absent from the baseline'
                    ),
                )
            )

    baseline_functions = _section(baseline, 'functions')
    for key, score in _section(current, 'functions').items():
        if key in baseline_functions:
            continue
        score = int(score)
        if score > NEW_FUNCTION_COGNITIVE_CEILING:
            violations.append(
                Violation(
                    measure='new_function_over_ceiling',
                    key=key,
                    baseline=NEW_FUNCTION_COGNITIVE_CEILING,
                    current=score,
                    message=(
                        f'new_function_over_ceiling: {key} has cognitive '
                        f'complexity {score}, above the '
                        f'{NEW_FUNCTION_COGNITIVE_CEILING} ceiling that applies '
                        'to functions absent from the baseline'
                    ),
                )
            )
    return violations


def check_against_baseline(current: dict, baseline: dict) -> list[Violation]:
    """Compare a fresh report against the committed baseline. Pure.

    Order of operations is deliberate. The three hard-failure preconditions run
    FIRST and raise ``MetricsError``, so a wrong-version or partial measurement
    reports its own named cause instead of a wall of downstream violations that
    would send the reader hunting a regression which does not exist.

    Returns violations sorted by ``(measure, key)`` so a failure message is
    diffable run to run.
    """
    _require_complete_enumeration(current)
    _require_matching_params(current, baseline)

    violations = [
        *_check_files(current, baseline),
        *_check_functions(current, baseline),
        *_check_tests(current, baseline),
        *_check_totals(current, baseline),
        *_check_ceilings(current, baseline),
    ]
    return sorted(violations, key=lambda v: (v.measure, v.key))


# ---------------------------------------------------------------------------
# CLI, in the house shape of scripts/scan_task_toolcall_leaks.py: _build_parser()
# / _render_table() / main(argv) -> int, with the exit ladder documented in the
# module docstring above (0 clean, 1 ratchet violations, 2 instrument failure).

#: Where the committed baseline lives, relative to the repo root.
BASELINE_RELPATH = 'orchestrator/tests/merge_lane_ratchet_baseline.json'


def repo_root() -> Path:
    """The repo root, derived from this file's location (``<root>/scripts/``)."""
    return Path(__file__).resolve().parents[1]


def _render_table(report: dict, root: Path) -> str:
    """The human-readable measure table printed by ``--report``.

    ``mi`` (radon's maintainability index) is computed HERE rather than stored in
    the report: it is reported, never ratcheted, so putting it in the baseline
    would freeze a number nothing enforces and churn the file whenever radon
    changed its formula.
    """
    files = report.get('files', {})
    totals = derive_totals(report)
    params = report.get('params', {})
    width = max([len(p) for p in files] + [len('TOTALS')])

    lines = [
        f'merge-lane metrics -- {len(files)} cluster paths, '
        f'complexipy {params.get("complexipy_version")}',
        '',
        f'{"path":<{width}}  {"lines":>7}  {"prose":>7}  {"cognitive":>9}  {"mi":>6}',
        '-' * (width + 36),
    ]
    for path, entry in sorted(files.items()):
        mi = maintainability_index(_read_source(root, path), path=path)
        lines.append(
            f'{path:<{width}}  {entry["lines"]:>7}  {entry["prose_lines"]:>7}  '
            f'{entry["cognitive"]:>9}  {mi:>6.2f}'
        )
    lines.extend(
        [
            '-' * (width + 36),
            f'{"TOTALS":<{width}}  {totals["lines"]:>7}  {totals["prose_lines"]:>7}  '
            f'{totals["cognitive"]:>9}',
            '',
            f'function_local_imports  {totals["function_local_imports"]:>7}',
            f'reexport_names          {totals["reexport_names"]:>7}',
            '',
            f'test suite -- {len(report.get("tests", {}))} lane-importing files '
            f'under {TESTS_ROOT}',
            f'  patch_targets (distinct)  {totals["patch_targets"]:>7}',
            f'  private_reads             {totals["private_reads"]:>7}',
            '',
        ]
    )

    # INV-11's user-observable signal: completeness is legible in the RESULT,
    # not only in a log line. A partial sweep measures LOW, so a reader who
    # cannot see this row cannot tell an improvement from a skipped file.
    enumeration = report.get('enumeration', {})
    unreadable = list(enumeration.get('unreadable', ()))
    lines.append(
        f'enumeration: complete={enumeration.get("complete")}  '
        f'requested={len(enumeration.get("requested", ()))}  '
        f'resolved={len(enumeration.get("resolved", ()))}  '
        f'unreadable={len(unreadable)}'
    )
    if unreadable:
        lines.extend(f'  UNREADABLE: {path}' for path in unreadable)
    return '\n'.join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Measure the merge-lane cluster (PRD plans/merge-lane-quality-prd.md '
            'task alpha) and ratchet every measure against the committed '
            'baseline. READ-ONLY except under --write-baseline.'
        ),
        epilog=(
            'Exit codes: 0 clean; 1 ratchet violations (a measure rose above '
            'the baseline, or a new file/function breached a ceiling); '
            '2 instrument failure (an unparseable cluster file, complexipy '
            f'missing or outside {COMPLEXIPY_REQUIRED}, a missing or malformed '
            'baseline, or a baseline whose recorded parameters no longer match '
            'this tree). 1 and 2 are deliberately distinct: a broken instrument '
            'must never read as a clean tree or as a real regression.'
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--report', action='store_true', help='Print the measure table.'
    )
    mode.add_argument(
        '--json', action='store_true', help='Print the report as JSON.'
    )
    mode.add_argument(
        '--check', action='store_true',
        help='Compare a fresh measurement against --baseline (the ratchet).',
    )
    mode.add_argument(
        '--write-baseline', metavar='PATH',
        help='Measure and write a baseline to PATH. Regenerating the committed '
        'baseline merely to make a test pass silently widens the ratchet for '
        'every downstream task -- see the file\'s own _README.',
    )
    parser.add_argument(
        '--root', default=str(repo_root()),
        help='Repo root to measure (default: this script\'s own checkout). '
        'Pair it with --baseline; on its own it would compare another tree '
        'against THIS checkout\'s baseline.',
    )
    parser.add_argument(
        '--baseline', default=str(repo_root() / BASELINE_RELPATH),
        help='Baseline JSON for --check (default: %(default)s).',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. See the module docstring for the exit-code contract."""
    args = _build_parser().parse_args(argv)
    root = Path(args.root)
    try:
        report = build_report(root)
        if args.json:
            print(json.dumps(report))
            return 0
        if args.report:
            print(_render_table(report, root))
            return 0
        if args.write_baseline:
            target = write_baseline(Path(args.write_baseline), report)
            print(f'wrote {target}')
            return 0
        violations = check_against_baseline(report, load_baseline(Path(args.baseline)))
        if not violations:
            return 0
        print(
            f'{len(violations)} merge-lane ratchet violation(s) against '
            f'{args.baseline}:',
            file=sys.stderr,
        )
        for violation in violations:
            print(f'  {violation.message}', file=sys.stderr)
        print(
            'A task that legitimately LOWERS a measure regenerates the baseline '
            'in the SAME commit. A task may never raise one.',
            file=sys.stderr,
        )
        return 1
    except MetricsError as exc:
        print(f'merge_lane_metrics: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
