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
