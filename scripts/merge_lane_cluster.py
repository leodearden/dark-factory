"""Measure the merge-lane cluster and its lane-importing tests into one report.

PRD ``plans/merge-lane-quality-prd.md`` task alpha. The measures are the ones
the PRD's Background table quotes -- file lines and prose lines, per-function
cognitive complexity, function-local (reach-back) imports, re-export shim names,
distinct test patch targets into lane internals, and private-attribute reads
from tests. ``scripts/merge_lane_ratchet.py`` judges a report against the
committed baseline, and ``scripts/merge_lane_metrics.py`` is the command line.

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
  ``ClusterSpecError`` (missing, unreadable) or ``source_measures.MetricsError``
  (unparseable) naming the path and the cause;
* a ``CLUSTER_PATHS`` GLOB expanding to zero -> fine (that is
  ``merge_lane/**`` until PRD task zeta1 lands);
* complexipy absent, or resolving outside ``source_measures.COMPLEXIPY_REQUIRED`` ->
  ``MetricsError`` naming the tool and the version found;
* the whole-of-``orchestrator/tests`` sweep keeps the siblings' per-file
  fail-soft polarity, but records every skipped file in
  ``Enumeration.unreadable`` and marks the enumeration incomplete (a file
  that cannot be read, or that names a lane module and cannot be parsed --
  see ``sweep_repo``), and
  ``scripts/merge_lane_ratchet.py::check_against_baseline`` REFUSES to
  compare an incomplete enumeration.

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
is ever ratcheted, because ``scripts/merge_lane_ratchet.py::check_against_baseline``
reads completeness from the CURRENT report and freezing a number nothing
enforces would only churn the file on every unrelated test file the repo gains
(esc-5021-7).
"""
from __future__ import annotations

import ast
import dataclasses
import unicodedata
from collections.abc import Collection
from pathlib import Path

import source_measures


class ClusterSpecError(source_measures.MetricsError):
    """A ``CLUSTER_PATHS`` literal is missing, not a regular file, or unreadable.

    A ``source_measures.MetricsError`` subclass because it is an instrument
    failure (exit 2) like every other, and every boundary catches that base.
    """


# ---------------------------------------------------------------------------
# The cluster spec -- SPOT for PRD Appendix A.

#: The merge-lane package. Every module under it is a lane module, and a patched
#: name's leaf is its dotted path below it: ``<attr>`` on the package itself,
#: ``<sub>.<attr>`` on a submodule.
LANE_PACKAGE = 'orchestrator.merge_lane'

#: Where ``LANE_PACKAGE``'s modules live on disk, derived so the two cannot disagree.
LANE_PACKAGE_GLOB = f'orchestrator/src/{LANE_PACKAGE.replace(".", "/")}/**/*.py'

#: Repo-relative literal paths and glob patterns naming the merge-lane cluster.
#: This tuple IS Appendix A; editing it forces a deliberate baseline
#: regeneration, because ``scripts/merge_lane_ratchet.py::check_against_baseline``
#: compares the baseline's recorded ``params.cluster_paths`` against it and
#: raises on a mismatch.
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
    LANE_PACKAGE_GLOB,
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

# ---------------------------------------------------------------------------
# Enumeration -- completeness carried in the RESULT, per INV-11.


@dataclasses.dataclass(frozen=True)
class Enumeration:
    """Which paths were asked for, which were measured, and which were skipped.

    ``complete`` is the single signal every consumer checks. It is False
    whenever anything landed in ``unreadable``, and
    ``scripts/merge_lane_ratchet.py::check_against_baseline`` raises rather
    than compares when it is False -- so a degraded sweep can never masquerade
    as a clean tree.
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

    Raises ``ClusterSpecError`` naming the path when a LITERAL entry is missing, is
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
            raise ClusterSpecError(
                f'cluster path {entry!r} does not exist under {root} -- '
                'CLUSTER_PATHS is the SPOT source of PRD Appendix A; if the '
                'file was renamed or removed, update CLUSTER_PATHS and '
                'regenerate the baseline in the same commit'
            )
        if not candidate.is_file():
            raise ClusterSpecError(
                f'cluster path {entry!r} is not a regular file under {root}'
            )
        try:
            candidate.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError) as exc:
            raise ClusterSpecError(
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
# The measure is ``source_measures.patch_targets_in_tree`` over the package
# plus the patch modules the caller passes, keeping the distinct leaves. It
# subsumes the retired reachback guard's allowlist as a COUNT, which is what
# lets PRD task delta delete the guard once the count reaches zero.


def lane_patch_modules(root: Path) -> frozenset[str]:
    """``LANE_PACKAGE`` plus the dotted name of every module under it on disk in
    *root*, so a patch receiver bound to a submodule is resolved and a class
    imported from the package is not."""
    on_disk = (
        source_measures.src_module_name(match.relative_to(root).as_posix())
        for match in root.glob(LANE_PACKAGE_GLOB)
    )
    return frozenset({LANE_PACKAGE, *(name for name in on_disk if name is not None)})


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
    ``LANE_PACKAGE``."""
    return imports_lane_module_in_tree(source_measures.parse_source(source, path=path))


def imports_lane_module_in_tree(tree: ast.Module) -> bool:
    """``imports_lane_module`` over an already-parsed tree."""
    lane_names = lane_module_names()

    def _is_lane(dotted: str) -> bool:
        return dotted in lane_names or _is_in_lane_package(dotted)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_is_lane(alias.name) for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            if _is_lane(node.module):
                return True
            # `from orchestrator import git_ops` binds the module itself.
            if any(_is_lane(f'{node.module}.{alias.name}') for alias in node.names):
                return True
    return False


def _is_in_lane_package(module: str) -> bool:
    return module == LANE_PACKAGE or module.startswith(LANE_PACKAGE + '.')


def lane_import_fragments() -> frozenset[str]:
    """Identifiers of which every lane-importing source names at least one.

    The last component of each cluster module and of ``LANE_PACKAGE``, derived
    from the same two sources as ``imports_lane_module_in_tree``. A match there
    is a dotted name that is a cluster module or lies under the package, so one
    of its components is such a leaf -- written as an identifier, which no line
    continuation can split (a dotted name's dots can be: ``import
    orchestrator.\\<newline>git_ops`` parses).
    """
    return frozenset(
        name.rpartition('.')[2] for name in (*lane_module_names(), LANE_PACKAGE)
    )


def may_import_lane_module(source: str, fragments: Collection[str]) -> bool:
    """False proves *source* imports no lane module, without parsing it.

    True is only "perhaps": comments and strings add matches, and relative
    imports (which ``imports_lane_module_in_tree`` ignores) are admitted too.
    The source is read as the parser compares identifiers, under NFKC, so a
    full-width spelling of a leaf counts (``_tree_scan.ParsedModule.may_name``
    in ``orchestrator/tests`` is the precedent).
    """
    text = source if source.isascii() else unicodedata.normalize('NFKC', source)
    return any(fragment in text for fragment in fragments)


def test_file_measures(
    source: str, *, path: str, patch_modules: Collection[str]
) -> dict[str, object] | None:
    """The two test-suite measures for *source*, or None when it is not
    lane-importing.

    Returning None rather than a zeroed dict makes the exclusion a property of
    the MEASURE rather than of the caller, so a non-lane-importing file can
    never be summed into the report by accident.

    *patch_modules* names the modules under ``LANE_PACKAGE`` that a patch
    receiver may be bound to, as ``lane_patch_modules`` derives them; with the
    package alone, ``from orchestrator.merge_lane import gates`` would patch
    ``gates`` unseen.
    """
    return test_file_measures_in_tree(
        source_measures.parse_source(source, path=path), patch_modules
    )


def test_file_measures_in_tree(
    tree: ast.Module, patch_modules: Collection[str]
) -> dict[str, object] | None:
    """``test_file_measures`` over an already-parsed tree."""
    if not imports_lane_module_in_tree(tree):
        return None
    lane_targets = source_measures.patch_targets_in_tree(
        tree, {LANE_PACKAGE, *patch_modules}
    )
    return {
        'patch_targets': sorted({_lane_leaf(target) for target in lane_targets}),
        'private_reads': source_measures.private_reads_in_tree(tree),
    }


def _lane_leaf(target: source_measures.PatchTarget) -> str:
    return f'{target.module}.{target.leaf}'.removeprefix(LANE_PACKAGE + '.')


# ---------------------------------------------------------------------------
# Report assembly.

SCHEMA_VERSION = 1

#: Subdirectory swept for the two test-suite measures.
TESTS_ROOT = 'orchestrator/tests'


def build_report(root: Path) -> dict[str, object]:
    """Measure the whole cluster plus the test suite under *root*.

    Note there is NO ``totals`` key: cluster-wide totals are DERIVED at check
    time by ``scripts/merge_lane_ratchet.py::derive_totals`` from the stored
    per-path maps. PRD gamma1..gamma10 run in parallel and each lowers only its
    own group's contribution, so a shared totals line would conflict on every
    one of ten rebases -- while deriving preserves the anti-rename-gaming
    property in full (see that function) and keeps one number in one place.

    The ``enumeration`` block obeys that same constraint, which it did not when
    the argument above was first written: the cluster half is a fixed manifest
    that moves only when ``CLUSTER_PATHS`` does, and the test tree's size is
    reported rather than stored, so nothing here is a shared line ten branches
    would each rewrite (see ``scripts/merge_lane_ratchet.py::_stored_enumeration``).
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
    under ``orchestrator/tests``, reading each file once and parsing it once
    when its text names a lane module (``may_import_lane_module``).

    THIS sweep keeps the sibling guards' per-file fail-SOFT polarity -- an
    unrelated mid-edit file must not redden the ratchet, which is the
    misattribution every neighbouring guard exists to avoid. What is NOT soft is
    the record: each skipped file lands in ``Enumeration.unreadable``, the
    enumeration goes incomplete, and
    ``scripts/merge_lane_ratchet.py::check_against_baseline`` then refuses to
    compare at all. That split is the INV-11 seam between "this cluster file is
    unmeasurable, which IS the finding" and "some unrelated file is mid-edit".

    A file that cannot be read or decoded is recorded whatever it says. A file
    that names no lane module is never parsed, so a syntax error in one is not
    recorded: it cannot be lane-importing, and an unparseable file the sweep
    has no measure for is the unrelated mid-edit file above.
    """
    patch_modules = lane_patch_modules(root)
    fragments = lane_import_fragments()
    test_tree = _test_tree_files(root)
    tests: dict[str, object] = {}
    unreadable: list[str] = []
    for relpath in test_tree:
        try:
            source = (root / relpath).read_text(encoding='utf-8')
            if not may_import_lane_module(source, fragments):
                continue
            measures = test_file_measures(source, path=relpath, patch_modules=patch_modules)
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

    Key order is fixed here, not incidental:
    ``scripts/merge_lane_ratchet.py::render_baseline`` emits this block with
    ``json.dumps(..., indent=2)``, so insertion order IS the committed bytes.
    """
    return {
        'requested': list(cluster.requested),
        'resolved': list(cluster.resolved),
        'test_tree': coverage.to_dict(),
        'unreadable': list(cluster.unreadable) + list(swept_unreadable),
        'complete': cluster.complete and not swept_unreadable,
    }
