#!/usr/bin/env python3
"""Measure the merge-lane cluster, and ratchet its structure against a committed baseline.

PRD ``plans/merge-lane-quality-prd.md`` task alpha. The measures are the ones
the PRD's Background table quotes -- file lines and prose lines, per-function
cognitive complexity, function-local (reach-back) imports, re-export shim names,
distinct test patch targets into lane internals, and private-attribute reads
from tests. The two size measures are measured and reported but never
ratcheted (``REPORTED_ONLY_MEASURES``); every other one is. The committed
baseline lives at ``orchestrator/tests/merge_lane_ratchet_baseline.json`` and
the gate that enforces it is ``orchestrator/tests/test_merge_lane_ratchet.py``.

Ratchet contract, and it has two sides. An UNAUTHORIZED raise is refused: by
``--check``, and decisively by ``--write-baseline`` itself, so a raise cannot be
absorbed by regenerating the file that would have caught it. An AUTHORIZED one
is permitted and recorded -- ``--authorize-raise <task-id> --reason <text>``
writes the exact per-measure delta as one new file under ``LEDGER_DIR_RELPATH``,
so net-additive work lands as a reviewed diff rather than as a number nobody saw
move. Lowering needs
no ceremony and equality is permitted -- this is a ratchet, not a day-one gate.
The two ceilings (``FILE_LINE_CEILING``, ``NEW_FUNCTION_COGNITIVE_CEILING``)
apply only to paths and qualnames ABSENT from the baseline, so the
grandfathering can only ever shrink.

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
  ``RatchetError`` (missing, unreadable) or ``source_measures.MetricsError``
  (unparseable) naming the path and the cause;
* a ``CLUSTER_PATHS`` GLOB expanding to zero -> fine (that is
  ``merge_lane/**`` until PRD task zeta1 lands);
* complexipy absent, or resolving outside ``source_measures.COMPLEXIPY_REQUIRED`` ->
  ``MetricsError`` naming the tool and the version found;
* the whole-of-``orchestrator/tests`` sweep keeps the siblings' per-file
  fail-soft polarity, but records every skipped file in
  ``Enumeration.unreadable`` and marks the enumeration incomplete, and
  ``check_against_baseline`` REFUSES to compare an incomplete enumeration.

That last clause is what makes "a partial enumeration is distinguishable from a
complete one in the RESULT, not only in a log line" true rather than aspirational.
The block carrying it is split by KIND, and the two halves are stored
differently. ``unreadable`` (both halves, by name) and ``complete`` travel in
the report AND in the committed baseline, as do the cluster's ``requested`` and
``resolved`` paths -- that is the named manifest where a path the instrument
cannot read IS the finding. The test tree's breadth travels as ``test_tree``'s
two counts in the report ONLY: both reach ``--json`` verbatim and both are
legible under ``--report`` -- the numerator under its own ``test suite --``
label, since it is ``len(tests)`` and one number gets one place -- but neither
is ever ratcheted, because ``check_against_baseline`` reads completeness from
the CURRENT report and freezing a number nothing enforces would only churn the
file on every unrelated test file the repo gains (esc-5021-7).

Exit codes
----------
0  clean -- measures taken, and (under ``--check``) no ratchet violation.
1  ratchet violations -- one or more measures rose above the baseline, or a new
   file/function exceeded a ceiling. Violations are printed to stderr. This
   covers BOTH faces of the gate: ``--check`` finding a raise, and
   ``--write-baseline`` REFUSING to absorb one for want of an
   ``--authorize-raise`` (``UnauthorizedRaise``, which deliberately does not
   subclass ``MetricsError`` so it cannot be reclassified as a broken tool).
2  instrument failure (``source_measures.MetricsError``, or its ``RatchetError``
   subclass for the ratchet's own faults) -- an unparseable cluster file,
   complexipy missing or out of range, a missing/malformed baseline, or a
   baseline whose recorded parameters no longer match this tree. Deliberately
   distinct from 1 so a broken instrument is never mistaken either for a clean
   tree or for a real regression.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import re
import sys
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeGuard

import _checkout as _checkout
import source_measures

from shared import safe_io


class RatchetError(source_measures.MetricsError):
    """The ratchet could not do its job: its cluster spec, baseline, ledger,
    an authorization, or the params a comparison rests on is missing,
    malformed or inconsistent.

    A ``source_measures.MetricsError`` subclass because both are instrument
    failures (exit 2) and every boundary catches them together; a type of its
    own so the ratchet's faults are named by the ratchet, not by the layer it
    measures with.
    """


class AppendOnlyViolation(RatchetError):
    """The ledger's recorded history was REWRITTEN, or the one commit that splits
    the retired array into files broke its rule -- a verdict, not a fault.

    A ``RatchetError`` subclass, so nothing that already catches one changes
    behaviour; a DISTINCT type, because the two are read differently at a
    boundary. ``scripts/check_staged_ratchet_raise.py`` and
    ``scripts/check_ratchet_advance.py`` exit 1 for this (the change did
    something the ratchet forbids and can fix) and 2 for its siblings (the gate
    could not do its job). Collapsing them would leave a
    caller that reasonably treats 2 as "instrument down, retry or ignore"
    waving a rewritten ledger through.
    """


# ---------------------------------------------------------------------------
# The cluster spec -- SPOT for PRD Appendix A.

#: Repo-relative literal paths and glob patterns naming the merge-lane cluster.
#: This tuple IS Appendix A; editing it forces a deliberate baseline
#: regeneration, because ``check_against_baseline`` compares the baseline's
#: recorded ``params.cluster_paths`` against it and raises on a mismatch.
CLUSTER_PATHS: tuple[str, ...] = (
    'orchestrator/src/orchestrator/lane_lifecycle.py',
    'orchestrator/src/orchestrator/offline_lane.py',
    'orchestrator/src/orchestrator/warm_lane_pool.py',
    'orchestrator/src/orchestrator/recover_main.py',
    # Measured (lines, cognitive) but exempt from the file-size ceiling -- see
    # SIZE_CEILING_EXEMPT.
    'orchestrator/src/orchestrator/git_ops.py',
    # The package, holding the fourteen modules PRD task zeta2 moved into it. A
    # glob matching nothing is NOT a failure; a literal matching nothing is.
    'orchestrator/src/orchestrator/merge_lane/**/*.py',
    'orchestrator/tests/_merge_queue_harness.py',
    'orchestrator/tests/conftest.py',
)

#: Paths measured and ratcheted, but exempt from ``FILE_LINE_CEILING``.
#: PRD decision 9: ``git_ops.py`` is the git engine for warm lanes, the
#: scheduler and recovery -- not only the merge lane -- and its split is a
#: follow-up PRD. The exemption is scoped to the CEILING alone: git_ops.py's
#: structural measures are still ratcheted, so they cannot grow unwatched.
SIZE_CEILING_EXEMPT: frozenset[str] = frozenset(
    {'orchestrator/src/orchestrator/git_ops.py'}
)

#: A file ABSENT from the baseline may not exceed this many lines.
FILE_LINE_CEILING = 1500

#: A function ABSENT from the baseline may not exceed this cognitive complexity.
NEW_FUNCTION_COGNITIVE_CEILING = 15

#: The measures a breach of those two ceilings is reported under. A breach is
#: not a rise over a recorded value, so ``unrecorded_raises`` covers it by name.
NEW_FILE_OVER_CEILING = 'new_file_over_ceiling'
NEW_FUNCTION_OVER_CEILING = 'new_function_over_ceiling'
CEILING_BREACH_MEASURES: frozenset[str] = frozenset(
    {NEW_FILE_OVER_CEILING, NEW_FUNCTION_OVER_CEILING}
)

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


@dataclasses.dataclass(frozen=True)
class TestTreeCoverage:
    """How many .py files the test-tree sweep asked for, and how many it measured.

    The test tree's counterpart to ``Enumeration``, and deliberately NOT more
    fields on it: the cluster is a named manifest where an unresolved path IS
    the finding and must be shown by name, while this is an open sweep of other
    people's files where only the coverage ratio carries information. The
    ``unreadable`` paths of BOTH halves still travel by name -- that is INV-11's
    payload and it is never reduced to a count.
    """

    requested: int
    resolved: int

    def to_dict(self) -> dict[str, int]:
        return {'requested': self.requested, 'resolved': self.resolved}


def resolve_cluster_paths(root: Path) -> Enumeration:
    """Resolve ``CLUSTER_PATHS`` against *root* into a complete Enumeration.

    Raises ``RatchetError`` naming the path when a LITERAL entry is missing, is
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
            raise RatchetError(
                f'cluster path {entry!r} does not exist under {root} -- '
                'CLUSTER_PATHS is the SPOT source of PRD Appendix A; if the '
                'file was renamed or removed, update CLUSTER_PATHS and '
                'regenerate the baseline in the same commit'
            )
        if not candidate.is_file():
            raise RatchetError(
                f'cluster path {entry!r} is not a regular file under {root}'
            )
        try:
            candidate.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as exc:
            raise RatchetError(
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
# Test-suite patch targets into lane internals.
#
# The measure is ``source_measures.patch_targets_in_tree`` over
# ``LANE_PATCH_MODULES``, keeping the distinct leaves. It subsumes the retired
# reachback guard's allowlist as a COUNT, which is what lets PRD task delta
# delete the guard once the count reaches zero.

#: Dotted module paths whose attributes count as lane internals when patched:
#: the package, and through it every submodule (a leaf reads ``<sub>.<attr>``).
LANE_PATCH_MODULES: tuple[str, ...] = ('orchestrator.merge_lane',)


# ---------------------------------------------------------------------------
# Lane-importing test files, and their two test-suite measures.
#
# ``private_reads`` is applied to every lane-importing test file, receiver-
# agnostically (see ``source_measures.private_reads``). Measured magnitudes on
# this tree: 18,952 private attribute nodes across all 559 test files, 9,355
# restricted to the 167 lane-importing ones. (The PRD Background table's 5,735
# came from a narrower ad-hoc receiver set.)


def lane_module_names() -> frozenset[str]:
    """Dotted module names of the cluster, DERIVED from ``CLUSTER_PATHS``.

    Derived rather than hand-listed so the lane-importing predicate and
    Appendix A can never drift apart.
    """
    return frozenset(
        name
        for entry in CLUSTER_PATHS
        if '*' not in entry and (name := source_measures.src_module_name(entry)) is not None
    )


def imports_lane_module(source: str, *, path: str) -> bool:
    """True when *source* imports any cluster module, or anything under
    ``orchestrator.merge_lane``."""
    return imports_lane_module_in_tree(source_measures.parse_source(source, path=path))


def imports_lane_module_in_tree(tree: ast.Module) -> bool:
    """``imports_lane_module`` over an already-parsed tree."""
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


def test_file_measures(source: str, *, path: str) -> dict[str, object] | None:
    """The two test-suite measures for *source*, or None when it is not
    lane-importing.

    Returning None rather than a zeroed dict makes the exclusion a property of
    the MEASURE rather than of the caller, so a non-lane-importing file can
    never be summed into the report by accident.
    """
    return test_file_measures_in_tree(source_measures.parse_source(source, path=path))


def test_file_measures_in_tree(tree: ast.Module) -> dict[str, object] | None:
    """``test_file_measures`` over an already-parsed tree."""
    if not imports_lane_module_in_tree(tree):
        return None
    lane_targets = source_measures.patch_targets_in_tree(tree, LANE_PATCH_MODULES)
    return {
        'patch_targets': sorted({target.leaf for target in lane_targets}),
        'private_reads': source_measures.private_reads_in_tree(tree),
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

    The ``enumeration`` block obeys that same constraint, which it did not when
    the argument above was first written: the cluster half is a fixed manifest
    that moves only when ``CLUSTER_PATHS`` does, and the test tree's size is
    reported rather than stored, so nothing here is a shared line ten branches
    would each rewrite (see ``_stored_enumeration``).
    """
    # Up front, before any measurement: a wrong-version engine must fail
    # immediately with a named cause rather than after a 250-second run whose
    # numbers would be wrong anyway.
    version = source_measures.require_complexipy()
    enumeration = resolve_cluster_paths(root)
    sweep = sweep_repo(root)

    files: dict[str, object] = {}
    functions: dict[str, int] = {}
    for relpath in enumeration.resolved:
        source = source_measures.read_source(root, relpath)
        tree = source_measures.parse_source(source, path=relpath)
        size = source_measures.file_size_measures_in_tree(source, tree, path=relpath)
        target = root / relpath
        cognitive = source_measures.file_cognitive_measures(target)
        files[relpath] = {
            'lines': size.lines,
            'prose_lines': size.prose_lines,
            'cognitive': cognitive.total,
            'function_local_imports': source_measures.function_local_imports_in_tree(tree),
            'reexport_names': len(source_measures.reexport_names_in_tree(tree)),
        }
        for qualname, score in cognitive.per_function.items():
            functions[f'{relpath}::{qualname}'] = score

    return {
        'schema_version': SCHEMA_VERSION,
        'params': {
            'complexipy_version': version,
            'cluster_paths': list(CLUSTER_PATHS),
            'file_line_ceiling': FILE_LINE_CEILING,
            'new_function_cognitive_ceiling': NEW_FUNCTION_COGNITIVE_CEILING,
        },
        'enumeration': _report_enumeration(
            enumeration, sweep.unreadable, sweep.test_tree
        ),
        'files': dict(sorted(files.items())),
        'functions': dict(sorted(functions.items())),
        'tests': dict(sorted(sweep.tests.items())),
    }


@dataclasses.dataclass(frozen=True)
class RepoSweep:
    """What the ONE pass over the ``.py`` files under ``orchestrator/tests`` found.

    ``tests`` is every lane-importing file there with its two measures;
    ``unreadable`` names the paths the sweep had to skip, verbatim; ``test_tree``
    is how broad it was, as counts. The swept paths themselves are deliberately
    not accumulated: the caller has no use for a 568-entry manifest of other
    people's test files, and building one is how it ended up frozen in the
    committed baseline (esc-5021-7).
    """

    tests: dict[str, object]
    unreadable: tuple[str, ...]
    test_tree: TestTreeCoverage


def _test_tree_files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix() for path in (root / TESTS_ROOT).rglob('*.py')
    )


def sweep_repo(root: Path) -> RepoSweep:
    """Take the two test-suite measures in ONE pass over every ``.py`` on disk
    under ``orchestrator/tests``, reading and parsing each file once.

    THIS sweep keeps the sibling guards' per-file fail-SOFT polarity -- an
    unrelated mid-edit file must not redden the ratchet, which is the
    misattribution every neighbouring guard exists to avoid. What is NOT soft is
    the record: each skipped file lands in ``Enumeration.unreadable``, the
    enumeration goes incomplete, and ``check_against_baseline`` then refuses to
    compare at all. That split is the INV-11 seam between "this cluster file is
    unmeasurable, which IS the finding" and "some unrelated file is mid-edit".
    """
    test_tree = _test_tree_files(root)
    tests: dict[str, object] = {}
    unreadable: list[str] = []
    for relpath in test_tree:
        try:
            source = (root / relpath).read_text(encoding='utf-8')
            measures = test_file_measures(source, path=relpath)
        except (OSError, UnicodeDecodeError, source_measures.MetricsError):
            unreadable.append(relpath)
            continue
        if measures is not None:
            tests[relpath] = measures
    return RepoSweep(
        tests=tests,
        unreadable=tuple(unreadable),
        test_tree=TestTreeCoverage(requested=len(test_tree), resolved=len(tests)),
    )


def _report_enumeration(
    cluster: Enumeration,
    swept_unreadable: tuple[str, ...],
    coverage: TestTreeCoverage,
) -> dict[str, object]:
    """The report's enumeration block: cluster paths, test-tree counts, INV-11.

    *swept_unreadable* is every path the test-tree sweep had to skip.

    The two halves stay APART rather than being flattened into one pair of
    lists. Flattening cost information as well as bytes: the three CLUSTER_PATHS
    literals that live under ``orchestrator/tests`` appeared in both halves, so
    the merged lists carried three duplicate entries each, and any later attempt
    to separate them again by set membership against CLUSTER_PATHS would drop a
    ``merge_lane/**`` file once PRD task zeta1 makes that glob expand -- a
    silent completeness hole in the instrument whose whole job is completeness.

    Key order is fixed here, not incidental: ``render_baseline`` emits this block
    with ``json.dumps(..., indent=2)``, so insertion order IS the committed
    bytes.
    """
    return {
        'requested': list(cluster.requested),
        'resolved': list(cluster.resolved),
        'test_tree': coverage.to_dict(),
        'unreadable': list(cluster.unreadable) + list(swept_unreadable),
        'complete': cluster.complete and not swept_unreadable,
    }


#: Per-file measures summed into a cluster total by ``derive_totals``.
_SUMMED_FILE_MEASURES = (
    'lines',
    'prose_lines',
    'cognitive',
    'function_local_imports',
    'reexport_names',
)

#: Measured, stored and printed by --report, never ratcheted: raw size steers
#: toward deleting prose instead of rehoming it (docs/code-quality.md "Do not
#: steer by"; PRD plans/review-all-s25-scripts-instruments-prd.md D10). `lines`
#: stays stored because the new-file ceiling reads it.
REPORTED_ONLY_MEASURES: frozenset[str] = frozenset({'lines', 'prose_lines'})

_RATCHETED_FILE_MEASURES = tuple(
    measure for measure in _SUMMED_FILE_MEASURES if measure not in REPORTED_ONLY_MEASURES
)


def derive_totals(report: dict) -> dict[str, int]:
    """Cluster-wide totals, DERIVED by summing *report*'s per-path maps.

    Pure: *report* is never mutated.

    Deriving rather than storing is what makes the ten parallel gamma branches
    rebase cleanly, AND it preserves the anti-rename-gaming property exactly. A
    total computed by summing stored per-path baselines is unchanged when code
    carrying 500 cognitive moves from ``merge_queue.py`` to a brand-new path, so
    the ratchet still catches the move -- which a per-path-only comparison would
    not, since the new path is simply absent from the baseline.

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

# The ratchet's two COMMITTED ARTIFACTS, relative to the repo root. The
# baseline holds the frozen measures; its sidecar ledger -- a DIRECTORY holding
# one file per authorization -- holds the provenance of every raise that was
# ever authorized against them. Both paths are named here, above the README that
# cites them, because the remedy text an agent reads when the gate goes red is
# composed from them.
BASELINE_RELPATH = 'orchestrator/tests/merge_lane_ratchet_baseline.json'
LEDGER_DIR_RELPATH = 'orchestrator/tests/merge_lane_ratchet_raises'

#: The RETIRED single-array ledger, which every two authorizing branches
#: conflicted on at its tail. Read only by ``retire_legacy_ledger``, the rule for
#: the one commit that splits it into ``LEDGER_DIR_RELPATH`` (task 7298).
LEGACY_LEDGER_RELPATH = 'orchestrator/tests/merge_lane_ratchet_authorized_raises.json'

#: The commit-time auditor the remedy tells a blocked agent about, named here
#: rather than spelled into the prose. It is a REAL PATH a reader is sent to
#: grep, it is frozen into the committed baseline's ``_README`` verbatim, and
#: this instrument -- having no git dependency -- cannot notice it going stale.
#: As a constant, ``TestTheBlockMessageNamesTheAuthorizedPath`` can assert it
#: RESOLVES, which a bare string literal in the prose left unchecked.
COMMIT_GATE_RELPATH = 'scripts/check_staged_ratchet_raise.py'

#: The landing-time auditor ``hooks/reference-transaction`` runs on every move of
#: main, named here for the same reasons as ``COMMIT_GATE_RELPATH``: the remedy
#: sends a blocked reader to it, and a constant is what lets a test assert the
#: path RESOLVES.
ADVANCE_GATE_RELPATH = 'scripts/check_ratchet_advance.py'

#: THE ONE COPY of what to do when a measure rose. Composed into every site
#: that tells an agent it may not raise -- ``BASELINE_README`` (and so the
#: committed bytes a blocked reader opens), ``main()``'s ``--check`` trailer,
#: and the pytest gate's assertion message. Three near-copies used to say only
#: that raising was forbidden, and the copy that mattered sent an implementer to
#: escalation rather than to a sanctioned path (task 5342, esc-5342-1).
RAISE_REMEDY = (
    'Remedy: LOWER the measure. A task that legitimately lowers one regenerates '
    'the baseline in the SAME commit:\n'
    f'  python scripts/merge_lane_metrics.py --write-baseline {BASELINE_RELPATH}\n'
    'RAISING a measure is not forbidden -- it is never SILENT. Net-additive '
    'work (a bug fix that adds a guard, a widened CLUSTER_PATHS) records the '
    'raise as a reviewed diff:\n'
    f'  python scripts/merge_lane_metrics.py --write-baseline {BASELINE_RELPATH} '
    '--authorize-raise <task-id> '
    '--reason "<why this is net-additive work, not a refactor that failed>"\n'
    'That writes the exact per-measure delta as ONE NEW FILE under '
    f'{LEDGER_DIR_RELPATH}/, naming the task and the reason, so the raise lands '
    'as a diff a reviewer reads rather than as a number nobody saw move. A '
    "commit's raises are covered by the entry files THAT COMMIT adds: per "
    'measure and key, the rise may not exceed the sum of their deltas. Without '
    'those flags --write-baseline '
    'REFUSES to absorb a raise over an EXISTING baseline, so regenerating in '
    'place cannot widen the ratchet. Nor can going around it: '
    f'{COMMIT_GATE_RELPATH} runs in pre-commit on EVERY branch '
    'and compares the STAGED baseline against the one in git HEAD -- or, '
    'finishing a conflicted merge, against what merging both parents may '
    'hold -- so a baseline deleted first, written elsewhere and copied over, '
    'or hand-edited is refused on the same terms as a regeneration -- and so '
    'is a commit that '
    'modifies, deletes or renames a recorded entry file. The one thing it lets '
    'through deliberately is UNDOING THE LAST CHANGE to the baseline: staging '
    'the value this path held one commit ago, and nothing further back. That '
    'is the state the tree was just running with, so putting it back re-raises '
    'nothing new, and needing an authorization to undo a revert would make the '
    'honest move the expensive one. Reaching FURTHER back is refused, because '
    'every image in between is a value the ratchet moved through and returning '
    'behind them re-absorbs every measure they lowered.\n'
    'WHERE A RAISE LANDS: `git rebase` and a CLEAN merge -- including one the '
    'ratchet-baseline merge driver resolved -- skip pre-commit, so a raise they '
    'carry is caught where it LANDS instead. hooks/reference-transaction runs '
    f'{ADVANCE_GATE_RELPATH} on every move of main, the merge worker\'s `git '
    'update-ref` included, and refuses a move whose baseline rises over the '
    "merge base's beyond what the entry files the move adds cover. The remedy "
    "is to record the raise on the branch, against main's baseline, with the "
    'flags above.\n'
    'RESIDUAL, so you do not trust more than is true: creating main afresh, git '
    'run with core.hooksPath pointed elsewhere, a git older than 2.28, and a '
    'move made from a checkout whose hooks/ lacks the file all reach main '
    'unaudited. --authorize-raise is the way past.'
)

#: Emitted as the baseline's leading key, so the rule is in the file a reader
#: is about to "fix" rather than only in a docstring they will not open.
BASELINE_README = (
    'This is a committed RATCHET BASELINE for the merge-lane cluster '
    '(PRD plans/merge-lane-quality-prd.md, task alpha). Every measure here '
    f'except {" and ".join(sorted(REPORTED_ONLY_MEASURES))}, which are reported '
    'but never ratcheted, is frozen at its value on the commit that recorded it: '
    'the gate orchestrator/tests/test_merge_lane_ratchet.py FAILS on any such '
    'measure that RISES above these numbers. Equality is fine, lowering is the '
    'point. The enumeration block below is the CLUSTER half only: the '
    'instrument also sweeps every .py under '
    'orchestrator/tests for the measures in the tests section, and that sweep '
    'is sized by --report rather than ratcheted here, so an unrelated test file '
    'arriving never touches these bytes.\n'
    # Composed, never paraphrased: this is the one string in the instrument that
    # changes what a blocked agent does next, and the copy that mattered was
    # always the one in THIS file.
    + RAISE_REMEDY
)

#: The three per-path maps whose entries get one line each.
PER_PATH_SECTIONS: tuple[str, ...] = ('files', 'functions', 'tests')


def _stored_enumeration(enumeration: dict) -> dict:
    """Strip the live-only test-tree counts; the baseline stores the cluster half.

    WHY THE TEST TREE IS NOT STORED AT ALL, in paths OR in counts (esc-5021-7).
    ``sweep_repo`` walks every ``*.py`` under ``orchestrator/tests`` -- 568
    paths, of which only the lane-importing ones are ever measured. Freezing that
    manifest in the committed baseline made this gate a hair trigger: any task
    ANYWHERE in the repo that added, removed or renamed a single test file
    reddened ``test_baseline_matches_a_fresh_measurement`` even though no lane
    measure moved, and the failure it printed ("Regenerate it in this commit if
    you lowered a measure") invited a blind regeneration of a baseline that task
    had never inspected -- exactly the silent widening ``BASELINE_README`` and
    that test exist to prevent. Observed live: an unrelated file arriving via
    rebase (``test_roles_error_remedy_hint.py``, task 4964) produced a ONE-LINE
    baseline diff and a red gate.

    It also cut against this task's decompose-time baseline-format constraint,
    which rejected even a single shared ``totals`` line because ten parallel
    gamma branches "would conflict on every rebase". A frozen 568-entry list is
    strictly MORE rebase-sensitive than the line that constraint rejected, and it
    is sensitive to churn outside the lane entirely.

    STORING THE COUNTS INSTEAD WOULD BE WORSE, NOT BETTER, which is why this
    function drops them rather than reducing them. The denominator moves on any
    ``.py`` arriving anywhere under ``orchestrator/tests``, so it is the same
    hair trigger with a shorter diff. The numerator is exactly
    ``len(report['tests'])``, already in the file line-locally, so storing it is
    a second copy of one number (SPOT) -- and it is a single SHARED line that
    every gamma branch deleting a lane-importing test would rewrite, which is the
    guaranteed one-line conflict the ``totals`` constraint rejected. Neither is
    ratcheted by anything: ``check_against_baseline`` reads completeness from the
    CURRENT report, never from the stored block.

    INV-11 IS UNAFFECTED, and deliberately so: ``unreadable`` keeps every skipped
    path from BOTH halves verbatim and ``complete`` is untouched, so a skipped
    test-tree file is still named in the committed file and still makes
    ``check_against_baseline`` refuse to compare. The sweep's breadth is reported
    by ``--report`` and ``--json``, and a COLLAPSED sweep still fails loudly via
    ``test_the_live_sweep_denominator_covers_the_whole_test_tree`` rather than
    quietly shrinking anything.

    Idempotent, which ``render_baseline``'s round-trip contract requires:
    dropping an absent key is a no-op.
    """
    return {key: value for key, value in enumeration.items() if key != 'test_tree'}


def _render_section(name: str, mapping: dict) -> str:
    """Render one per-path map with exactly one line per entry, key-sorted."""
    if not mapping:
        return f'  {json.dumps(name)}: {{}}'
    rows = ',\n'.join(
        f'    {json.dumps(key)}: {json.dumps(mapping[key], sort_keys=True)}'
        for key in sorted(mapping)
    )
    return f'  {json.dumps(name)}: {{\n{rows}\n  }}'


def render_baseline(report: dict, *, readme: str = BASELINE_README) -> str:
    """Serialize *report* as the committed baseline's exact bytes.

    Idempotent: ``render_baseline(json.loads(render_baseline(r)))`` reproduces
    the same text, so regenerating a baseline from a baseline is a no-op rather
    than a churned file full of manufactured conflicts. The ``_README`` key is
    emitted from *readme* -- ``BASELINE_README`` for every measurement -- and
    any inbound one is dropped, which is what makes that hold across a round
    trip. The one caller passing another is the merge driver
    (``scripts/merge_lane_ratchet_merge.py``): it may run a different checkout's
    copy of this module than the tree it merges, so it carries the merged
    images' own README through rather than substituting its own.
    """
    entries = [f'  "_README": {json.dumps(readme)}']
    for key, value in report.items():
        if key == '_README':
            continue
        if key in PER_PATH_SECTIONS and isinstance(value, dict):
            entries.append(_render_section(key, value))
        else:
            # Top-level scalars and the small params/enumeration blocks are
            # ordinary pretty-printed JSON, shifted one level in. The
            # `enumeration` block sheds its live-only test-tree counts first --
            # see _stored_enumeration for why neither the test tree's paths nor
            # its numbers belong in a committed baseline (esc-5021-7).
            if key == 'enumeration' and isinstance(value, dict):
                value = _stored_enumeration(value)
            block = json.dumps(value, indent=2).replace('\n', '\n  ')
            entries.append(f'  {json.dumps(key)}: {block}')
    return '{\n' + ',\n'.join(entries) + '\n}\n'


class UnauthorizedRaise(Exception):
    """``write_baseline`` refused to absorb a raise nobody authorized.

    NOT a ``MetricsError``, deliberately. ``main()`` maps every MetricsError to
    exit 2 ("the instrument is broken"), and a refused regeneration is not a
    broken instrument -- it is a measure that rose, discovered at write time
    instead of at check time, which is squarely exit 1. Subclassing would
    silently reclassify a real regression as a broken tool, collapsing the one
    distinction the exit ladder exists to keep.

    Carries the measured ``violations`` so ``main()`` can print the same
    per-violation lines ``--check`` prints, and so the ledger record for an
    AUTHORIZED raise is derived from the same comparison that would have
    refused it.
    """

    def __init__(self, violations: Sequence[Violation]) -> None:
        self.violations: tuple[Violation, ...] = tuple(violations)
        super().__init__(
            f'refusing to write a baseline that RAISES {len(self.violations)} '
            'measure(s):\n'
            + '\n'.join(f'  {v.message}' for v in self.violations)
            + f'\n{RAISE_REMEDY}'
        )


@dataclasses.dataclass(frozen=True)
class BaselineWrite:
    """What one write actually did: where the bytes landed, and what it recorded.

    ``record`` is the ledger entry this write appended, or None when it appended
    none -- which is every unauthorized write, and an authorized one over a
    report that raised nothing. Returning it is what lets a caller REPORT the
    outcome instead of re-deriving it: the alternative is reading the ledger
    before and after and diffing the lengths, which is three reads of one file
    and misreports the moment anything else appends concurrently.
    """

    path: Path
    record: dict | None


def write_baseline(
    path: Path,
    report: dict,
    *,
    authorization: RaiseAuthorization | None = None,
    ledger: Path | None = None,
) -> BaselineWrite:
    """Render *report* and write it to *path* atomically, refusing silent raises.

    THE TEXT IS RENDERED BEFORE THE DESTINATION IS TOUCHED, so a rendering
    failure leaves the committed baseline byte-for-byte intact instead of
    truncated -- a truncated baseline would be a *widened* ratchet, the one
    failure mode this instrument must never produce silently.

    THE WRITE GATE, and why it lives here rather than in the comparator. The
    pytest gate pins the committed baseline to equal a fresh measurement of the
    tree, so in every COMMITTED state the comparator is trivially clean; a raise
    is red only between editing the code and regenerating, and regeneration used
    to absorb it silently because this function read nothing before overwriting.
    Regeneration is therefore the ratchet's real enforcement point. If the
    destination already holds a baseline, the raises this report would introduce
    are measured against it, and without an *authorization* the write is REFUSED
    with ``UnauthorizedRaise`` -- the destination untouched, exit 1 at the CLI.

    DELIBERATELY NOT APPLIED here: ``_require_matching_params``. Both it and
    ``resolve_cluster_paths`` instruct the reader to edit ``CLUSTER_PATHS`` and
    regenerate the baseline in one commit, so routing regeneration through that
    precondition would break the one workflow those messages prescribe. Widening
    the cluster still raises the derived totals, so it still needs an
    authorization -- which is honest, the totals really did rise.

    THE OTHER ARM OF THAT PRECONDITION, complexipy version drift, reaches this
    gate the same way and is easy to misread. A new version re-reads every
    cognitive number at once -- the same merge_queue.py measures 2031 at 3.0.0,
    2092 at 5.0.0 and 2133 at 6.x/7.x -- so a post-drift regeneration arrives as
    a wall of raises and needs one authorization. It is recorded as a raise
    because that is what the numbers did; the ledger reason is what tells a
    reviewer the INSTRUMENT moved rather than the code, which is why
    ``_require_matching_params`` asks for the version in it.

    With an *authorization*, the raises are recorded as one new entry file in
    the append-only ledger directory *ledger* (default ``repo_root() /
    LEDGER_DIR_RELPATH``) and the write goes ahead. WRITE ORDER PICKS THE
    SURVIVABLE FAILURE: the ledger entry lands FIRST, so
    a crash between the two writes leaves a recorded authorization over an
    unchanged baseline -- the gate stays red, nothing was widened, and the
    orphaned record is visible -- rather than an unrecorded raise sitting in the
    committed baseline, which is the exact silent widening this prevents. Both
    texts are rendered before either file is touched, extending the
    render-before-write property above across the pair.

    An authorization over a report that raises NOTHING is ignored and appends no
    record: the ledger holds raises, not intentions. The returned
    ``BaselineWrite`` says which of the two happened, so a caller reports the
    outcome from the write itself rather than re-reading the ledger to guess.

    A MISSING destination is a first write with nothing to compare against, and
    that is THIS function's exact limit. Deleting the committed baseline, or
    writing to a scratch path and copying it over, resets every frozen measure
    and re-grandfathers every ceiling, and no amount of reading the destination
    can see it.

    THAT HOLE IS NOW CLOSED ELSEWHERE, and deliberately elsewhere. Closing it
    means comparing against the copy git holds, which is a dependency on git
    this instrument does not have: ``scripts/ratchet_revisions.py`` holds every
    git invocation that reads history or the index, and the two gates built on
    it call ``compare_baseline_files`` here for the comparison itself, so there
    is one definition of "did a measure rise".
    ``scripts/check_staged_ratchet_raise.py`` runs in pre-commit on every
    branch; ``scripts/check_ratchet_advance.py`` runs from
    ``hooks/reference-transaction`` on every move of main, which is where a
    raise replayed by a rebase or landed by the merge worker's ``update-ref``
    is seen. The write gate stops a raise being ABSORBED, the commit gate one
    being STAGED, the advance gate one LANDING.

    A MALFORMED destination propagates ``load_baseline``'s ``RatchetError``
    rather than being overwritten: a previous baseline you cannot read is one
    whose raises you cannot see.
    """
    text = render_baseline(report)
    target = Path(path)
    raises = (
        _measure_raises(report, load_baseline(target)) if target.exists() else []
    )
    record = None
    if raises:
        if authorization is None:
            raise UnauthorizedRaise(raises)
        record = authorization_record(authorization, raises)
        append_authorization(
            Path(ledger) if ledger is not None else repo_root() / LEDGER_DIR_RELPATH,
            record,
        )
    safe_io.atomic_write_text(target, text, mkdir=True)
    return BaselineWrite(path=target, record=record)


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
        raise RatchetError(
            f'ratchet baseline {target} does not exist -- a missing baseline is '
            'a hard failure, never an empty-baseline pass; regenerate it with '
            '--write-baseline if this is the commit that introduces it'
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise RatchetError(
            f'ratchet baseline {target} could not be read: '
            f'{exc.__class__.__name__}: {exc}'
        ) from exc
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RatchetError(
            f'ratchet baseline {target} is not valid JSON: {exc}'
        ) from exc
    if not isinstance(loaded, dict):
        raise RatchetError(
            f'ratchet baseline {target} holds a '
            f'{type(loaded).__name__} at top level, expected a JSON object'
        )
    return loaded


# ---------------------------------------------------------------------------
# Authorized raises -- the ratchet's reviewed escape hatch.
#
# Placed here, immediately after the baseline's own rule, because a reader who
# opens this file to find out why a regeneration was refused must meet the rule
# and its escape hatch together. Splitting the enforcement away from the thing
# it enforces to keep a line count down is precisely the move this mechanism
# exists to stop rewarding.


def _require_non_blank(authorization: RaiseAuthorization, field: str) -> None:
    value = getattr(authorization, field)
    if not isinstance(value, str) or not value.strip():
        raise RatchetError(
            f'authorized raise is missing {field}: {value!r}. An authorization '
            f'with no {field} is not an authorization -- the ledger entry exists '
            'to tell a reviewer WHO raised a measure and WHY, and an entry that '
            'answers neither is worse than no entry at all.'
        )


@dataclasses.dataclass(frozen=True)
class RaiseAuthorization:
    """Permission to record ONE set of raises, granted by one invocation.

    An ACT, not a standing permission. It authorizes exactly the raises present
    in the report it accompanies and nothing else; the ledger entry it produces
    is provenance a reviewer reads, never an ACL this instrument consults. If a
    landed record could license a later raise, the natural next move -- pre-write
    a record, then regenerate -- would be indistinguishable from the silent
    widening the ratchet exists to prevent. Per-invocation means the default is
    always refusal, a stale record licenses nothing, and each raise costs one
    deliberate, attributed command.

    BOTH FIELDS ARE CHECKED AT CONSTRUCTION, so an unattributed authorization
    never exists to be passed anywhere. Enforcing it further down -- where the
    record is rendered -- would make the same value valid or invalid depending
    on whether the report it accompanied happened to raise anything, and a blank
    ``--reason`` over a lowering run would pass unremarked (heuristic 10).
    """

    task_id: str
    reason: str

    def __post_init__(self) -> None:
        _require_non_blank(self, 'task_id')
        _require_non_blank(self, 'reason')


#: Bumped only if an entry's SHAPE changes; entries are never rewritten otherwise.
LEDGER_SCHEMA_VERSION = 1

#: The one file in ``LEDGER_DIR_RELPATH`` that records no authorization.
LEDGER_README_NAME = '_README.json'

#: Emitted as ``LEDGER_README_NAME``'s leading key, for the same reason
#: BASELINE_README is: the rule belongs where a reader has the entries open.
LEDGER_README = (
    'This directory is the APPEND-ONLY provenance of every raise ever '
    'authorized against the merge-lane ratchet baseline '
    '(orchestrator/tests/merge_lane_ratchet_baseline.json): ONE FILE PER '
    'AUTHORIZATION, named <task-id>-<first 12 hex of the sha256 of the entry '
    'as sorted-key compact JSON>.json, so two branches authorizing at once add '
    'two different files and never conflict. A recorded entry is never '
    'modified, deleted or renamed. It is NOT a permission list: no entry here '
    'grants a future raise. A commit\'s raises are covered only by the entry '
    'files THAT COMMIT adds -- per measure and key, the rise may not exceed the '
    f'sum of their (current - baseline); a {NEW_FILE_OVER_CEILING} or '
    f'{NEW_FUNCTION_OVER_CEILING} row is covered when an added entry names it. '
    'An added entry must be ATTRIBUTED: its task_id is the task the branch '
    'works on -- the id in task/<id> or carry/<id>, or ending the branch name '
    'as -<id> -- or one an earlier commit on the branch since main cites, as '
    'its scope (fix(<id>): ...) or as "task <id>", which is how a carrier '
    'records the task it lands; and its reason is not blank. Pre-commit '
    '(scripts/check_staged_ratchet_raise.py) refuses any other, without '
    'reading the task store. '
    'Each entry\'s "measures" block is DERIVED from the measured raises the '
    'write gate computed, never hand-written, so it describes the BRANCH COMMIT '
    'it accompanied, not a state of main: when the ratchet-baseline merge '
    'driver composes two branches\' moves of one row, main holds ours + theirs '
    '- base, a value no single entry names. There is no timestamp: git carries '
    'the date. Written by scripts/merge_lane_metrics.py --write-baseline '
    'orchestrator/tests/merge_lane_ratchet_baseline.json --authorize-raise '
    '<task-id> --reason "<why>".'
)


def authorization_record(
    authorization: RaiseAuthorization, raises: Sequence[Violation]
) -> dict:
    """One ledger entry: who authorized it, why, and exactly what rose.

    A PURE PROJECTION -- it validates nothing, because a ``RaiseAuthorization``
    that exists is already attributed. ``measures`` is projected off the
    Violations the write gate measured, in their already-sorted order, reusing
    ``Violation``'s field names so the audit record and the failure message
    describe a raise in one vocabulary. Nothing here is typed by the agent, so
    the record cannot drift from the diff.
    """
    return {
        'task_id': authorization.task_id,
        'reason': authorization.reason,
        'measures': [
            {
                'measure': violation.measure,
                'key': violation.key,
                'baseline': violation.baseline,
                'current': violation.current,
            }
            for violation in raises
        ],
    }


def entry_filename(record: dict) -> str:
    """The file name *record* is recorded under: its task id, then its content digest.

    DERIVED FROM THE CONTENT, so a name can never hold two different entries:
    two branches adding the same name added the same bytes, which git merges
    clean, and an entry edited after it was written stops matching its own
    name -- which ``parse_entry`` refuses. The task id leads so an entry is
    findable by eye; it is reduced to filename-safe characters, because it is
    free text and the digest already carries uniqueness.
    """
    canonical = json.dumps(record, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:12]
    task = re.sub(r'[^A-Za-z0-9_-]', '_', str(record.get('task_id')))[:40]
    return f'{task}-{digest}.json'


def render_entry(record: dict) -> str:
    """One entry file's exact bytes."""
    return json.dumps(record, indent=2) + '\n'


def render_ledger_readme() -> str:
    """``LEDGER_README_NAME``'s exact bytes."""
    readme = {'_README': LEDGER_README, 'schema_version': LEDGER_SCHEMA_VERSION}
    return json.dumps(readme, indent=2) + '\n'


def parse_entry(name: str, text: str) -> dict:
    """The authorization recorded in the entry file *name* holding *text*.

    MALFORMED IS FATAL, as for every ratchet artifact: an entry that cannot be
    read is one whose provenance cannot be audited, and reading it as absent
    would hide history. So is an entry whose name is not ``entry_filename`` of
    its content -- it was edited after it was written, or written by hand under
    a guessed name, and either way the name no longer says what the file holds.
    """
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RatchetError(
            f'authorized-raise entry {name} is not valid JSON: {exc}'
        ) from exc
    if not isinstance(loaded, dict):
        raise RatchetError(
            f'authorized-raise entry {name} holds a {type(loaded).__name__} at '
            'top level, expected a JSON object'
        )
    expected = entry_filename(loaded)
    if name != expected:
        raise RatchetError(
            f'authorized-raise entry {name} is not named for its content '
            f'(its content is named {expected}). Entries are written by '
            '--authorize-raise and never edited afterwards; record a new one '
            'instead of changing a recorded one.'
        )
    return loaded


def load_ledger(directory: Path) -> list[dict]:
    """Every authorization recorded under *directory*. ABSENT is empty; MALFORMED is fatal.

    THE POLARITY IS THE OPPOSITE OF ``load_baseline``'s on absence, deliberately,
    because absence means opposite things. An absent BASELINE would compare clean
    against every measure -- a silent disarming, so it must fail hard (INV-11).
    An absent LEDGER means no raise was ever authorized, which is the fail-CLOSED
    state: nothing is permitted by it. Corruption is different in kind, and
    every file but ``LEDGER_README_NAME`` must parse as an entry
    (``parse_entry``). Ordered by file name, since git -- not the file -- carries
    the chronology.
    """
    target = Path(directory)
    if not target.exists():
        return []
    if not target.is_dir():
        raise RatchetError(
            f'authorized-raise ledger {target} is not a directory -- the ledger '
            'holds one file per authorization'
        )
    entries = []
    for path in sorted(target.iterdir()):
        if path.name == LEDGER_README_NAME:
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as exc:
            raise RatchetError(
                f'authorized-raise entry {path} could not be read: '
                f'{exc.__class__.__name__}: {exc}'
            ) from exc
        entries.append(parse_entry(path.name, text))
    return entries


def append_authorization(directory: Path, record: dict) -> Path:
    """Record *record* as a NEW file under *directory*; no existing file is touched.

    Append-only by construction: an authorization is a file of its own, so
    recording one cannot rewrite another, and the README is written only where
    it is absent. Writing the same record twice writes the same bytes to the
    same name. The promise still has to be CHECKED, because the directory is
    also edited by rebases, merges and hands -- ``own_entry_names`` is the
    reader that refuses a commit dropping or rewriting a recorded entry.
    """
    target = Path(directory)
    readme = target / LEDGER_README_NAME
    if not readme.exists():
        safe_io.atomic_write_text(readme, render_ledger_readme(), mkdir=True)
    entry = target / entry_filename(record)
    safe_io.atomic_write_text(entry, render_entry(record), mkdir=True)
    return entry


def own_entry_names(
    parents: Sequence[Mapping[str, str]], staged: Mapping[str, str]
) -> list[str]:
    """The entry files a commit adds, refusing any recorded one it drops or rewrites.

    Each mapping is entry file name -> content identity (a blob id, at the git
    boundary) for one parent's ledger and for the staged one. A commit's OWN
    entries are the staged files NO parent holds: for a merge, a file either
    parent recorded is HISTORY, never the merge's own, so it can cover nothing
    the merge raises (LEDGER_README: no entry grants a future raise).

    Append-only is a set comparison, not an ordering one: every file any parent
    holds must be staged unchanged. A drop, an edit and a rename (a drop plus an
    add) are each refused with an ``AppendOnlyViolation``. Pure; the caller
    decides where the listings came from.
    """
    for parent in parents:
        lost = sorted(name for name, identity in parent.items() if staged.get(name) != identity)
        if lost:
            raise AppendOnlyViolation(
                'the authorized-raise ledger is append-only, and this change '
                f'modifies, deletes or renames {len(lost)} of the '
                f'{len(parent)} recorded entr(ies) a parent holds: '
                f'{", ".join(lost[:5])}{" ..." if len(lost) > 5 else ""}. A '
                'recorded entry is never touched -- record a new one with '
                '--authorize-raise instead.'
            )
    return sorted(name for name in staged if not any(name in parent for parent in parents))


def legacy_ledger_entries(text: str) -> list[dict]:
    """The entries of the retired single-array ledger *text*. MALFORMED is fatal."""
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RatchetError(
            f'retired ledger {LEGACY_LEDGER_RELPATH} is not valid JSON: {exc}'
        ) from exc
    raises = loaded.get('raises') if isinstance(loaded, dict) else None
    if not isinstance(raises, list) or not all(isinstance(entry, dict) for entry in raises):
        raise RatchetError(
            f'retired ledger {LEGACY_LEDGER_RELPATH} has no "raises" list of '
            'entries, so the history it carries cannot be checked into files'
        )
    return raises


def retire_legacy_ledger(
    retired: Sequence[dict], staged_names: Collection[str], own: Sequence[dict]
) -> list[dict]:
    """The ONE-COMMIT RULE for retiring ``LEGACY_LEDGER_RELPATH``: what may cover a raise.

    *retired* is every entry of the array a parent held and this commit no
    longer stages; *staged_names* is every staged entry file; *own* is this
    commit's own entries (``own_entry_names``). Nothing retired -> *own*,
    unchanged. Otherwise the commit is the split, and it is SET-EQUAL: every
    retired entry must be staged as its ``entry_filename``, and every own entry
    must be a retired one -- so the split drops nothing, rewrites nothing, and
    authorizes nothing of its own. It returns no entry at all, because a
    retired entry is history and covers no raise. Any breach is an
    ``AppendOnlyViolation``.

    Uniform across a merge that retires the array (a branch still carrying it
    merging a main that already split it): the entries main already holds as
    files are staged as history, the branch's own become own files here, and
    the same two conditions hold.
    """
    if not retired:
        return list(own)
    retired_names = {entry_filename(entry) for entry in retired}
    dropped = sorted(retired_names - set(staged_names))
    fresh = sorted({entry_filename(entry) for entry in own} - retired_names)
    if dropped or fresh:
        raise AppendOnlyViolation(
            f'this commit retires {LEGACY_LEDGER_RELPATH}, so it must split '
            f'its {len(retired_names)} entr(ies) into {LEDGER_DIR_RELPATH}/ '
            'exactly: one file per entry, no entry dropped or rewritten, and no '
            f'authorization of its own. Missing: {len(dropped)} '
            f'({", ".join(dropped[:5])}); not from the retired array: '
            f'{len(fresh)} ({", ".join(fresh[:5])}). Re-run '
            'scripts/split_merge_lane_ratchet_ledger.py, and authorize any '
            'raise in a commit of its own.'
        )
    return []


def is_count(value: object) -> TypeGuard[int]:
    """A measured count: an int, and not a bool, which JSON would also load as one."""
    return isinstance(value, int) and not isinstance(value, bool)


def _authorized_amounts(records: Sequence[dict]) -> tuple[dict[tuple, int], set[tuple]]:
    """Per (measure, key): the summed (current - baseline) *records* authorize, and every pair named.

    A row whose numbers are not both ints adds nothing but is still NAMED; a
    record whose ``measures`` is missing or not a list names nothing. A badly
    hand-edited entry must REFUSE the commit, not crash the gate.
    """
    amounts: dict[tuple, int] = {}
    named: set[tuple] = set()
    for record in records:
        measures = record.get('measures')
        for row in measures if isinstance(measures, list) else ():
            if not isinstance(row, dict):
                continue
            pair = (row.get('measure'), row.get('key'))
            named.add(pair)
            baseline, current = row.get('baseline'), row.get('current')
            if is_count(baseline) and is_count(current):
                amounts[pair] = amounts.get(pair, 0) + current - baseline
    return amounts, named


def unrecorded_raises(
    raises: Sequence[Violation], records: Sequence[dict]
) -> list[Violation]:
    """The raises in *raises* that *records* -- a commit's OWN entries -- do not cover. Pure.

    THE COVERAGE RULE, one rule at every enforcement point (heuristic 10): per
    ``(measure, key)``, a raise is covered when its rise (``current -
    baseline``) is at most the SUM of ``current - baseline`` over every row of
    *records* naming that pair. A ceiling breach (``CEILING_BREACH_MEASURES``)
    is not a rise over a recorded value, so it is covered when a record names
    its measure and key.

    A SUM, because a commit -- and a branch -- carries several entries for one
    key: an agent running ``--authorize-raise`` twice records ``b -> m`` and
    ``m -> f`` while the commit's delta reads ``b -> f``, and two raises of one
    key composed by a merge add up the same way. Matching any one entry's
    absolute ``current`` refused 9 of 27 legitimately authorized landings when
    task 7298's design replayed the json-touching landings of 2026-10-06..20; the
    sum refused none, and still refuses an unrecorded rise, because only the
    commit's own entries are summed and a stale entry claims less than the rise.

    THE SLACK THE SUM ADMITS, stated so nobody trusts more than is true: two
    entries for ONE raise (``--authorize-raise`` rerun after ``git checkout --
    <baseline>``, with a different reason, so a different file) both count, and
    double that key's allowance in that commit. The write gate still refuses
    every rise it measures, so the slack is reachable only by a baseline staged
    past it by hand.

    The inverse of ``authorization_record``, and placed beside it so the writer
    and the auditor read a raise out of one field vocabulary (SPOT).
    """
    amounts, named = _authorized_amounts(records)

    def covered(violation: Violation) -> bool:
        pair = (violation.measure, violation.key)
        if violation.measure in CEILING_BREACH_MEASURES:
            return pair in named
        return violation.current - violation.baseline <= amounts.get(pair, 0)

    return [violation for violation in raises if not covered(violation)]


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

    @classmethod
    def rose(
        cls, measure: str, key: str, baseline: int, current: int
    ) -> Violation:
        """THE canonical constructor for a measured raise -- public, and on the type.

        ``message`` is composed HERE and nowhere else (SPOT), which is what
        makes it the only honest way to build one: the ledger's ``measures``
        vocabulary, ``unrecorded_raises``' coverage triple and the gate's
        refusal lines all describe a raise in these terms, so a caller that
        could not reach this would re-derive the wording and drift from it.
        It was ``_violation`` until task 5722 -- a leading underscore on the
        only constructor of a type this module hands out, which made every
        test of the auditor reach past the module's public face.

        A CLASSMETHOD rather than a module-level ``violation()``: ``main`` and
        several siblings bind ``violation`` as a loop variable over a list of
        them, so a module-level factory of that name would be shadowed inside
        exactly the scopes most likely to want it.
        """
        return cls(
            measure=measure,
            key=key,
            baseline=baseline,
            current=current,
            message=(
                f'{measure} rose {baseline} -> {current} for {key} -- the '
                'merge-lane ratchet permits a measure to fall or hold, never '
                'to rise'
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
        raise RatchetError(
            f'{which} report has no "params" block -- it was not produced by '
            'build_report, so there is nothing to state how it was measured'
        )
    return params


def _require_complete_enumeration(current: dict) -> None:
    enumeration = current.get('enumeration')
    if not isinstance(enumeration, dict):
        raise RatchetError(
            'current report has no "enumeration" block -- completeness must be '
            'legible in the RESULT, not only in a log line (INV-11)'
        )
    if enumeration.get('complete') is True:
        return
    unreadable = list(enumeration.get('unreadable', ()))
    raise RatchetError(
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
        raise RatchetError(
            f'complexipy version drift: the baseline was measured with '
            f'{baseline_version!r}, this run used {current_version!r}. Cognitive '
            'numbers are version-dependent -- the same merge_queue.py measures '
            '2031 at 3.0.0, 2092 at 5.0.0 and 2133 at 6.x/7.x -- so a silent '
            'drift rewrites every number at once and leaves the ratchet '
            'comparing two incomparable measurements. Pin the dev group '
            f'({source_measures.COMPLEXIPY_REQUIRED}), or regenerate the baseline deliberately '
            '-- which is a RE-MEASUREMENT, not code growth, and the write gate '
            'cannot tell the difference: every number the new version reads '
            'higher is a raise it will refuse. Pass --authorize-raise with a '
            f'--reason naming the version change, so the {LEDGER_DIR_RELPATH} entry '
            'reads as a change of instrument rather than as a task that grew '
            'the cluster.'
        )

    recorded = list(_params(baseline, 'baseline').get('cluster_paths', ()))
    live = list(CLUSTER_PATHS)
    if recorded != live:
        added = [p for p in recorded if p not in live]
        removed = [p for p in live if p not in recorded]
        raise RatchetError(
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
        for measure in _RATCHETED_FILE_MEASURES:
            was, now = int(base_entry.get(measure, 0)), int(entry.get(measure, 0))
            if now > was:
                violations.append(Violation.rose(measure, path, was, now))
    return violations


def _check_functions(current: dict, baseline: dict) -> list[Violation]:
    current_functions = _section(current, 'functions')
    return [
        Violation.rose('cognitive', key, int(was), int(current_functions[key]))
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
            violations.append(Violation.rose('private_reads', path, was, now))
        # DISTINCT names, matching the PRD's measure: re-patching the same leaf
        # twice more in one file is not a new reach into lane internals.
        was_targets = len(set(base_entry.get('patch_targets', ())))
        now_targets = len(set(entry.get('patch_targets', ())))
        if now_targets > was_targets:
            violations.append(
                Violation.rose('patch_targets', path, was_targets, now_targets)
            )
    return violations


def _check_totals(current: dict, baseline: dict) -> list[Violation]:
    current_totals = derive_totals(current)
    baseline_totals = derive_totals(baseline)
    return [
        _total_violation(measure, was, current_totals[measure])
        for measure, was in sorted(baseline_totals.items())
        if measure not in REPORTED_ONLY_MEASURES and current_totals.get(measure, 0) > was
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
                    measure=NEW_FILE_OVER_CEILING,
                    key=path,
                    baseline=FILE_LINE_CEILING,
                    current=lines,
                    message=(
                        f'{NEW_FILE_OVER_CEILING}: {path} is {lines} lines, above '
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
                    measure=NEW_FUNCTION_OVER_CEILING,
                    key=key,
                    baseline=NEW_FUNCTION_COGNITIVE_CEILING,
                    current=score,
                    message=(
                        f'{NEW_FUNCTION_OVER_CEILING}: {key} has cognitive '
                        f'complexity {score}, above the '
                        f'{NEW_FUNCTION_COGNITIVE_CEILING} ceiling that applies '
                        'to functions absent from the baseline'
                    ),
                )
            )
    return violations


def _measure_raises(current: dict, previous: dict) -> list[Violation]:
    """Every RAISE *current* introduces over *previous*, sorted by (measure, key).

    A "raise" is any state the baseline may not silently acquire. That is the
    four rise arms, and it is also a CEILING BREACH -- a new path over
    ``FILE_LINE_CEILING`` or a new function over
    ``NEW_FUNCTION_COGNITIVE_CEILING``. The two belong together because
    regeneration is *how* a ceiling breach would be acquired: ceilings apply
    only to keys ABSENT from the baseline, so writing an oversized new file into
    the baseline exempts it from that moment on. Absorbed by a blind
    regeneration, the two are indistinguishable.

    The comparison arms only -- no preconditions, no policy. TWO callers run
    this one implementation (SPOT): ``check_against_baseline``, which is what
    ``--check`` and the pytest gate compare with, and ``write_baseline``'s gate,
    which is what stops a regeneration absorbing a raise nobody reviewed. Two
    implementations of "did a measure rise" would eventually disagree, and the
    one that disagreed quietly would be the write path.
    """
    violations = [
        *_check_files(current, previous),
        *_check_functions(current, previous),
        *_check_tests(current, previous),
        *_check_totals(current, previous),
        *_check_ceilings(current, previous),
    ]
    return sorted(violations, key=lambda v: (v.measure, v.key))


def check_against_baseline(current: dict, baseline: dict) -> list[Violation]:
    """Compare a fresh report against the committed baseline. Pure.

    Order of operations is deliberate. The three hard-failure preconditions run
    FIRST and raise ``RatchetError``, so a wrong-version or partial measurement
    reports its own named cause instead of a wall of downstream violations that
    would send the reader hunting a regression which does not exist.

    Returns violations sorted by ``(measure, key)`` so a failure message is
    diffable run to run.
    """
    _require_complete_enumeration(current)
    _require_matching_params(current, baseline)
    return _measure_raises(current, baseline)


def compare_baseline_files(previous: Path, current: Path) -> list[Violation]:
    """Every raise the move from the *previous* baseline image to *current* makes.

    THE ENFORCEMENT POINT THE WRITER CANNOT BE. ``write_baseline``'s gate
    compares a fresh report against whatever sits at the destination, so a
    baseline deleted first, or rendered elsewhere and copied over, leaves it
    nothing to compare and resets every frozen measure. This face compares two
    committed IMAGES instead, which is the shape a raise actually arrives in:
    a diff. It is what ``scripts/check_staged_ratchet_raise.py`` audits a staged
    commit with, and ``scripts/check_ratchet_advance.py`` a move of main -- and
    being two images rather than a measurement, it runs no complexipy, reads no
    tree, and still consults no git.

    Both images load through ``load_baseline``, so a missing or malformed one is
    that function's named hard failure rather than a silent empty-baseline pass.

    DELIBERATELY NOT APPLIED here, as in ``write_baseline`` and for the same
    stated reason: ``_require_matching_params``. A commit that widens
    ``CLUSTER_PATHS`` legitimately changes the recorded params, and that is the
    one workflow ``resolve_cluster_paths`` prescribes.
    """
    return _measure_raises(load_baseline(Path(current)), load_baseline(Path(previous)))


def _magnitude(value: Any) -> int:
    """One measure's size: a name list by its DISTINCT names, as ``_check_tests`` counts."""
    return len(set(value)) if isinstance(value, list) else int(value)


def _merged_value(base: Any, ours: Any, theirs: Any) -> Any:
    """What a 3-way merge of one measure may hold, where None means absent.

    Each side's own move stands, up or down, and a deletion stands unless the
    other side moved the entry (a modify/delete keeps the modification). Where
    both sides moved one measure, :func:`_both_moved_bound` bounds it.
    """
    if ours == base:
        return theirs
    if theirs in (base, ours):
        return ours
    if ours is None or theirs is None:
        return theirs if ours is None else ours
    if isinstance(ours, dict) and isinstance(theirs, dict):
        return _merged_mapping(base if isinstance(base, dict) else {}, ours, theirs)
    return _both_moved_bound(base, ours, theirs)


def _both_moved_bound(base: Any, ours: Any, theirs: Any) -> Any:
    """The most a measure both sides moved can honestly hold.

    That is the higher side, or both sides' moves composed over *base* where
    that is higher: ``ours + theirs - base``, or for a name list the 3-way set
    merge, whose absent *base* is the empty list. A count with no *base* has
    nothing to compose over.
    """
    if isinstance(ours, list):
        before, mine, other = set(base or ()), set(ours), set(theirs)
        composed = sorted((mine & other) | (mine - before) | (other - before))
    elif base is None:
        return max(ours, theirs, key=_magnitude)
    else:
        composed = ours + theirs - base
    return max(ours, theirs, composed, key=_magnitude)


def _merged_mapping(base: dict, ours: dict, theirs: dict) -> dict:
    """:func:`_merged_value` key by key, dropping every key the merge leaves absent."""
    merged = {
        key: _merged_value(base.get(key), ours.get(key), theirs.get(key))
        for key in base.keys() | ours.keys() | theirs.keys()
    }
    return {key: value for key, value in merged.items() if value is not None}


def _merge_bound(base: dict, ours: dict, theirs: dict) -> dict:
    """The per-path sections a merge of *ours* and *theirs* over *base* may hold."""
    return {
        name: _merged_mapping(
            _section(base, name), _section(ours, name), _section(theirs, name)
        )
        for name in PER_PATH_SECTIONS
    }


def compare_merged_baseline_files(
    *, base: Path | None, ours: Path | None, theirs: Path | None, current: Path
) -> list[Violation]:
    """Every raise a MERGE commit's *current* baseline makes over its parents' bound.

    The bound is what git does to the file text, applied per measure against the
    merge *base*: a measure only one side moved stands at that side's value, up
    or down, and one both sides moved is bounded by :func:`_both_moved_bound`.
    An absent *base* is the empty image git merges unrelated histories against,
    and an absent parent moved nothing, so it stands in as the base.

    NOT "clean against either parent": that admits keeping one side's stale
    value over the other side's lowering, which is an unrecorded widening on
    whichever branch the merge lands on.

    KEYWORD-ONLY: four same-typed paths make a crossed call -- one that reads
    CLEAN -- the hazard ``test_the_argument_order_cannot_read_clean_when_crossed``
    pins for two.
    """
    base_image = load_baseline(Path(base)) if base is not None else {}
    ours_image = load_baseline(Path(ours)) if ours is not None else base_image
    theirs_image = load_baseline(Path(theirs)) if theirs is not None else base_image
    return _measure_raises(
        load_baseline(Path(current)), _merge_bound(base_image, ours_image, theirs_image)
    )



# ---------------------------------------------------------------------------
# CLI, in the house shape of scripts/scan_task_toolcall_leaks.py: _build_parser()
# / _render_table() / main(argv) -> int, with the exit ladder documented in the
# module docstring above (0 clean, 1 ratchet violations, 2 instrument failure).


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
        source = source_measures.read_source(root, path)
        mi = source_measures.maintainability_index(source, path=path)
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
    #
    # THE TEST-TREE DENOMINATOR IS ONLY HERE. It is reported, never ratcheted --
    # the same call this function already makes for `mi` -- so the committed
    # baseline carries no test-tree number at all and this row is the one place
    # the sweep's breadth is visible. Its NUMERATOR is deliberately absent: that
    # is ``len(report['tests'])``, already rendered under a better label by the
    # "test suite --" line above, and two copies of one number in one table only
    # invite a reader to wonder whether they can disagree (SPOT). `.get` is
    # tolerant on purpose: a stored baseline block has no `test_tree` key, and
    # rendering one must not raise.
    enumeration = report.get('enumeration', {})
    unreadable = list(enumeration.get('unreadable', ()))
    test_tree = enumeration.get('test_tree', {})
    lines.append(
        f'enumeration: complete={enumeration.get("complete")}  '
        f'cluster requested={len(enumeration.get("requested", ()))} '
        f'resolved={len(enumeration.get("resolved", ()))}  '
        f'test tree requested={test_tree.get("requested")}  '
        f'unreadable={len(unreadable)}'
    )
    if unreadable:
        lines.extend(f'  UNREADABLE: {path}' for path in unreadable)
    return '\n'.join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Measure the merge-lane cluster (PRD plans/merge-lane-quality-prd.md '
            'task alpha) and ratchet its structural measures against the '
            'committed baseline. READ-ONLY except under --write-baseline.'
        ),
        epilog=(
            'Exit codes: 0 clean; 1 ratchet violations (a measure rose above '
            'the baseline, or a new file/function breached a ceiling); '
            '2 instrument failure (an unparseable cluster file, complexipy '
            f'missing or outside {source_measures.COMPLEXIPY_REQUIRED}, a missing or malformed '
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
        help='Measure and write a baseline to PATH. REFUSES (exit 1) to absorb '
        'a measure that rose unless --authorize-raise names the task doing it, '
        'so a regeneration can never silently widen the ratchet for every '
        'downstream task -- see the file\'s own _README.',
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
    parser.add_argument(
        '--authorize-raise', metavar='TASK_ID',
        help='Permit --write-baseline to absorb the raises in THIS measurement, '
        'recording them in --ledger against this task id. Requires --reason. '
        'Authorization is an act, not a standing entry: a recorded raise never '
        'permits a later one.',
    )
    parser.add_argument(
        '--reason', metavar='TEXT',
        help='Why this raise is net-additive work rather than a refactor that '
        'failed. Requires --authorize-raise; recorded verbatim in the ledger.',
    )
    parser.add_argument(
        '--ledger', default=str(repo_root() / LEDGER_DIR_RELPATH),
        help='Authorized-raise ledger DIRECTORY --authorize-raise records one new '
        'entry file in (default: %(default)s). Pair it with --write-baseline when '
        'writing outside this checkout; on its own it would record in THIS '
        'checkout\'s ledger.',
    )
    return parser


def _resolve_authorization(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> RaiseAuthorization | None:
    """Validate the authorization flags against each other, or exit 2 saying why.

    argparse cannot express "these two imply each other, and both require that
    mode", so it is done here rather than left unchecked. A flag that silently
    did nothing would read, to the agent who typed it, exactly like an
    authorization that was granted.

    The value object's own non-blank invariant lands here as the THIRD usage
    error rather than as a broken instrument: ``--reason ""`` is a flag typed
    wrong, so it prints like the two above and exits on the same code, before
    anything is measured.
    """
    if (args.authorize_raise is None) != (args.reason is None):
        parser.error(
            '--authorize-raise and --reason require each other: a raise with no '
            'recorded reason is not an authorized raise, it is an unattributed '
            'one'
        )
    if args.authorize_raise is None:
        return None
    if not args.write_baseline:
        parser.error(
            '--authorize-raise is a modifier of --write-baseline, which is the '
            'only mode that can absorb a raise. --check reports raises; it '
            'never records them'
        )
    try:
        return RaiseAuthorization(task_id=args.authorize_raise, reason=args.reason)
    except RatchetError as exc:
        parser.error(str(exc))


def _write(
    args: argparse.Namespace, report: dict, authorization: RaiseAuthorization | None
) -> int:
    """The --write-baseline arm, reporting what this invocation actually did.

    Without an authorization the ledger is neither read nor written -- an
    unauthorized raise is refused before it is reached -- so a plain
    regeneration never depends on that file's health, and --ledger can be passed
    through unconditionally to keep one call site rather than two arms.

    WHAT IS PRINTED COMES FROM THE WRITE, never from re-reading the ledger, so
    the operator's log cannot claim a raise the committed file does not carry.
    """
    ledger = Path(args.ledger)
    written = write_baseline(
        Path(args.write_baseline), report, authorization=authorization, ledger=ledger
    )
    print(f'wrote {written.path}')
    if authorization is None:
        return 0
    if written.record is None:
        # An authorization that recorded nothing must SAY so. Printing only
        # "wrote ..." would read, to the agent who typed --authorize-raise,
        # exactly like an authorization that was granted and recorded -- the
        # very failure _resolve_authorization rejects flag shapes to avoid
        # (INV-11, no silent fail-soft).
        print(f'no measure rose; nothing recorded in {ledger}')
        return 0
    print(
        f'recorded {len(written.record["measures"])} authorized raise(s) for '
        f'task {written.record["task_id"]} in {ledger}'
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. See the module docstring for the exit-code contract."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    authorization = _resolve_authorization(parser, args)
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
            return _write(args, report, authorization)
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
        print(RAISE_REMEDY, file=sys.stderr)
        return 1
    except UnauthorizedRaise as exc:
        # Exit 1, not 2: a refused regeneration is a measure that ROSE, found at
        # write time instead of at check time. Exit 2 stays reserved for a
        # broken instrument, and both faces of the gate print the same lines.
        print(
            f'{len(exc.violations)} merge-lane ratchet violation(s) would be '
            f'absorbed by writing {args.write_baseline}:',
            file=sys.stderr,
        )
        for violation in exc.violations:
            print(f'  {violation.message}', file=sys.stderr)
        print(RAISE_REMEDY, file=sys.stderr)
        return 1
    except source_measures.MetricsError as exc:
        print(f'merge_lane_metrics: {exc}', file=sys.stderr)
        return 2




if __name__ == '__main__':
    sys.exit(main())
