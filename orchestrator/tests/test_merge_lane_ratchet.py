"""The merge-lane quality ratchet, and the unit tests of the instrument behind it.

PRD ``plans/merge-lane-quality-prd.md`` task alpha. The instrument itself lives
in ``scripts/merge_lane_cluster.py`` (measurement), ``scripts/merge_lane_ratchet.py``
(verdict) and ``scripts/merge_lane_metrics.py`` (command line); this module hosts
the measurement-stratum unit tests, the derived-totals and stored-enumeration
tests, the command-line tests and the real tree-scanning gate. The synthetic
unit tests of the write gate, of the rest of the baseline rendering, and of the
comparator it shares with ``--check``, are in
``test_merge_lane_ratchet_write_gate.py``.

Ratchet contract
----------------
``orchestrator/tests/merge_lane_ratchet_baseline.json`` freezes every measure at
its value on the commit that introduced it, except the two size measures
(``ratchet.REPORTED_ONLY_MEASURES``), which are reported but never ratcheted.
The gate FAILS on any other measure that rises above its baseline (ratchet
semantics). Equality is permitted -- this is a
ratchet, not a day-one gate, and on the introducing commit 62 lane functions
already exceed cognitive 15 and ``merge_queue.py`` is 21,550 lines. The two
CEILINGS (1,500 lines per file, cognitive 15 per function) therefore apply only
to paths and qualnames ABSENT from the baseline, so the grandfathering can only
ever shrink.

A task that legitimately LOWERS a measure regenerates the baseline in the SAME
commit (``python scripts/merge_lane_metrics.py --write-baseline
orchestrator/tests/merge_lane_ratchet_baseline.json``). A raise is a different
matter but not a forbidden one: it must be AUTHORIZED, never silent. Adding
``--authorize-raise <task-id> --reason <text>`` to that same command records the
per-measure delta as one new file under ``merge_lane_ratchet_raises/``; without
those flags ``--write-baseline`` refuses to absorb a raise over an EXISTING
baseline, so regenerating cannot make this test pass by widening the ratchet.
Deleting the baseline first would -- there is then nothing to compare against --
but a staged deletion is refused outright by ``scripts/check_staged_ratchet_raise.py``
in pre-commit on every branch, and the wholesale reset it would otherwise be is
read in the diff by a reviewer besides.
``ratchet.RAISE_REMEDY`` is the one copy of that rule, and every failure message
here composes it.

Why the instrument fails HARD where its neighbours fail soft
------------------------------------------------------------
Every sibling guard in this directory fails SOFT on an unparseable file
(``_find_merge_queue_private_patches`` returns ``[]``; ``_scans_whole_tree_py``
returns False) because they sweep the WHOLE tree, so a mid-edit or deliberately
malformed fixture file must not redden a guard about something else. This
instrument inverts that polarity for its own named cluster: an Appendix A file
the script cannot parse is not noise, it IS the finding (INV-11, no silent
fail-soft). The whole-test-tree sweep keeps the siblings' per-file fail-soft
polarity, but every skipped file lands in ``enumeration.unreadable`` and the
ratchet refuses to compare a partial enumeration at all -- so a degraded sweep
can never masquerade as a clean tree.
"""
from __future__ import annotations

import ast
import copy
import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from _git_fixtures import RepoSeed, build_repo
from _merge_lane_ratchet_fixtures import (
    GIT_OPS_RELPATH,
    WORKER_RELPATH,
    corrupt_ledger,
    lane_cluster,
    ledger_files,
    metrics,
    ratchet,
    source_measures,
    synthetic_report,
)
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

# This module AST-parses the Appendix A cluster (8 CLUSTER_PATHS entries, one
# of them a glob), plus every *.py under orchestrator/tests/ (one pass,
# `sweep_repo`, one parse per file), and runs complexipy over the cluster. All
# of that is ONE measurement per session, taken by the module-scoped
# `live_measurement` fixture below.
#
# WALL CLOCK ON THIS HOST IS NOT A RELIABLE MEASURE of this module: repeated
# runs over ONE unchanged tree spread as widely as the effect of task 5101,
# which cut the work here fourfold. So every guard added by that task counts
# WORK -- ast.parse / Path.read_text / complexipy calls -- and none asserts
# elapsed time; see `no_private_tree_scan` and
# TestBuildReport::test_each_cluster_file_is_measured_by_complexipy_once. The
# before/after measurement history is recorded once, dated, in
# plans/merge-lane-quality-prd.md Open Question 7; per-run seconds restated here
# would only drift.
#
# A HIGH RUNTIME HERE IS NOT A COMPLEXIPY VERSION SIGNAL. An earlier version of
# this comment estimated "MEASURED end to end at 35-65s" and the module then ran
# several times that, with 6.2.0 correctly resolved and require_complexipy()
# passing on every one of those runs. A wrong version fails immediately and by
# name, before any measurement.
#
# The 60s ini default would be a coin flip, and pytest-timeout's thread method
# enforces it by os._exit()ing the xdist worker -- which with
# --max-worker-restart=0 truncates the whole session and reports against an
# innocent test (esc-3980-1). See
# shared/src/shared/whole_tree_guard_family.py::WHOLE_TREE_SCAN_TEST_TIMEOUT. That mark is PER ITEM and the slowest single item has always
# sat far below it, so it was never the thing at risk here.
#
# NOTE this mark is deliberate and NOT compelled by the family guard
# test_whole_tree_scan_timeout_guard.py: its ``_scans_whole_tree_py`` detector
# looks for a literal ``rglob('*.py')`` in the TEST file's own source, and this
# module's sweep lives in scripts/merge_lane_cluster.py instead. Adding a marked
# module cannot break that guard either -- its family invariant uses ``>=``
# floors.
#
# The xdist_group pins the WHOLE module to one worker. Without it, --dist
# loadgroup distributes these items individually and every worker that draws one
# pays build_report over again -- a complexipy pass over the cluster plus a
# whole-tree AST sweep -- for a single cached result. Grouped, the module takes
# that cost exactly ONCE (module-scoped `live_measurement` below) while running
# in parallel with the rest of the suite. `xdist_group` is a registered marker;
# see test_marker_registration_drift.py's allowlist.
pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group('merge_lane_ratchet'),
]

_REPO_ROOT = Path(__file__).parents[2]


@dataclasses.dataclass(frozen=True)
class LiveMeasurement:
    """The one real ``build_report``, and the complexipy work taking it cost."""

    report: dict
    complexipy_calls: tuple[Path, ...]


@pytest.fixture(scope='module')
def live_measurement() -> LiveMeasurement:
    """THE single real measurement this module takes, instrumented.

    build_report is the module's whole remaining cost: one complexipy run per
    cluster file and one AST sweep of orchestrator/tests. EVERY test that needs
    live numbers shares this one result -- task 5101 folded on the last three
    stragglers, and `no_private_tree_scan` is what keeps them folded. The
    module's xdist_group keeps them on one worker so the cost is paid once per
    session rather than once per worker.

    The recording wrapper RECORDS AND DELEGATES, so the report is a real
    measurement and the call list is real work rather than a simulation of it.
    It wraps the public `source_measures.file_cognitive_measures`, which takes
    ONE complexipy measurement per call (tests/scripts/test_source_measures.py
    pins that), so each recorded call is one complexipy run.
    `test_each_cluster_file_is_measured_by_complexipy_once` reads that list, so
    the per-file-once guarantee costs no measurement of its own. The builtin
    `monkeypatch` fixture is function-scoped and cannot be requested here,
    hence `pytest.MonkeyPatch.context()`.
    """
    calls: list[Path] = []
    original = source_measures.file_cognitive_measures

    def recording(path: Path):
        calls.append(path)
        return original(path)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(source_measures, 'file_cognitive_measures', recording)
        report = lane_cluster.build_report(_REPO_ROOT)
    return LiveMeasurement(report=report, complexipy_calls=tuple(calls))


@pytest.fixture(scope='module')
def live_report(live_measurement: LiveMeasurement) -> dict:
    """The report half of `live_measurement` -- what every live-number test reads."""
    return live_measurement.report


@pytest.fixture()
def no_private_tree_scan(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Forbid the requesting test from sweeping the test tree a second time.

    Task 4520's discipline verbatim (shared/tests/test_tree_scan_sharing.py::
    _WorkCounter): count WORK, never wall-clock. An `assert elapsed < N` guard
    on this host would be a new flake planted by an anti-flake task -- repeated
    `--check` runs over ONE unchanged tree spread more than twofold. Zero work
    is also the sharper claim: it proves the test CONSUMED the shared
    measurement, where a fast wall clock only proves the box was quiet.

    The counters RECORD AND DELEGATE -- never raise, never stub -- so the test
    still exercises real behaviour and a regrown sweep reads as a count.

    Reads are scoped to `*.py` under orchestrator/tests because the claim is
    "this test swept no tree of its own", not "this test opened no file":
    `main(['--check'])` legitimately reads the baseline .json, which lives in
    the swept directory and must not trip the guard.

    Parses are NOT scoped that way and cannot be: `source_measures.parse_source` calls
    `ast.parse(source)` with no filename, so every parse a sweep makes records
    as '<unknown>' (measured) and a path-scoped parse counter would witness
    nothing at all for the regrowth this fixture exists to catch. Broad is also
    what makes it a backstop -- it still catches a sweep that reads through
    `open()` and so slips the read counter. The price is that a guarded item
    parsing anything of its own trips it, so reads (the precise counter) are
    asserted FIRST and the parse message names both readings.
    """
    swept = _REPO_ROOT / 'orchestrator' / 'tests'
    parses: list[str] = []
    reads: list[str] = []
    real_parse = ast.parse
    real_read_text = Path.read_text

    def counting_parse(source, filename='<unknown>', *args, **kwargs):
        parses.append(str(filename))
        return real_parse(source, filename, *args, **kwargs)

    def counting_read_text(self_path: Path, *args, **kwargs):
        if self_path.suffix == '.py' and swept in self_path.parents:
            reads.append(str(self_path))
        return real_read_text(self_path, *args, **kwargs)

    monkeypatch.setattr(ast, 'parse', counting_parse)
    monkeypatch.setattr(Path, 'read_text', counting_read_text)
    yield
    _SWEPT_ONCE = (
        'The orchestrator/tests tree is swept exactly ONCE per session, by the '
        'module-scoped `live_report` fixture. Derive this anchor from that '
        'report instead of re-walking the tree.'
    )
    assert reads == [], (
        f'This test read {len(reads)} test-tree *.py file(s) of its own '
        f'(first: {reads[:3]}). {_SWEPT_ONCE}'
    )
    assert parses == [], (
        f'This test called ast.parse {len(parses)} time(s) '
        f'(filenames: {parses[:3]}; sweeps record <unknown>). The read '
        f'assertion above passed, so EITHER this is a sweep that read through '
        f'`open()` -- {_SWEPT_ONCE} -- OR the parse was of something else '
        f'entirely (a tmp_path fixture, a lazily imported module rewritten by '
        f'pytest), in which case drop this fixture from this item rather than '
        f'narrowing the counter; see the fixture docstring.'
    )

# The 4 orchestrator/src literal paths of PRD Appendix A left outside the
# merge_lane package. The other 14 moved into it (task zeta2), their aliases
# were deleted (task eta), and the glob below measures them now. git_ops.py is
# listed separately because Appendix A adds it under a different rule
# (measured, but exempt from the file-size ceiling -- PRD decision 9).
_APPENDIX_A_SRC = (
    'orchestrator/src/orchestrator/lane_lifecycle.py',
    'orchestrator/src/orchestrator/offline_lane.py',
    'orchestrator/src/orchestrator/warm_lane_pool.py',
    'orchestrator/src/orchestrator/recover_main.py',
)
_APPENDIX_A_TESTS = (
    'orchestrator/tests/_merge_queue_harness.py',
    'orchestrator/tests/conftest.py',
)
_MERGE_LANE_GLOB = 'orchestrator/src/orchestrator/merge_lane/**/*.py'


# ---------------------------------------------------------------------------
# CLUSTER_PATHS -- the SPOT source of Appendix A.


class TestClusterPathSpec:
    def test_cluster_paths_covers_every_appendix_a_src_module(self) -> None:
        assert set(_APPENDIX_A_SRC) <= set(lane_cluster.CLUSTER_PATHS)

    def test_cluster_paths_includes_git_ops(self) -> None:
        # Measured but not gated by the file-size ceiling (PRD decision 9).
        assert GIT_OPS_RELPATH in lane_cluster.CLUSTER_PATHS

    def test_cluster_paths_includes_the_three_test_paths(self) -> None:
        assert set(_APPENDIX_A_TESTS) <= set(lane_cluster.CLUSTER_PATHS)

    def test_cluster_paths_carries_one_merge_lane_glob(self) -> None:
        globs = [p for p in lane_cluster.CLUSTER_PATHS if '*' in p]
        assert globs == [_MERGE_LANE_GLOB]

    def test_cluster_paths_has_exactly_the_appendix_a_entries(self) -> None:
        expected = (
            set(_APPENDIX_A_SRC)
            | {GIT_OPS_RELPATH}
            | set(_APPENDIX_A_TESTS)
            | {_MERGE_LANE_GLOB}
        )
        assert set(lane_cluster.CLUSTER_PATHS) == expected

    def test_cluster_paths_is_an_ordered_tuple(self) -> None:
        # Order is part of the spec: `params.cluster_paths` is compared against
        # it in check_against_baseline, and a stable order keeps that diff
        # readable when Appendix A is deliberately edited.
        assert isinstance(lane_cluster.CLUSTER_PATHS, tuple)

    def test_size_ceiling_exempt_is_exactly_git_ops(self) -> None:
        # PRD decision 9: git_ops.py is the git engine for warm lanes, the
        # scheduler and recovery -- not only the merge lane -- and its split is
        # a follow-up PRD. The exemption is scoped to the CEILING; git_ops.py is
        # still ratcheted (pinned by TestSizeCeilingExemptionIsScoped).
        assert set(lane_cluster.SIZE_CEILING_EXEMPT) == {GIT_OPS_RELPATH}

    def test_ceiling_constants(self) -> None:
        assert lane_cluster.FILE_LINE_CEILING == 1500
        assert lane_cluster.NEW_FUNCTION_COGNITIVE_CEILING == 15


# ---------------------------------------------------------------------------
# resolve_cluster_paths -- INV-11: a partial enumeration is never returned
# silently, and completeness is legible in the RESULT.


class TestResolveClusterPaths:
    def test_real_repo_enumeration_is_complete(self) -> None:
        enumeration = lane_cluster.resolve_cluster_paths(_REPO_ROOT)
        assert enumeration.complete is True
        assert enumeration.unreadable == ()

    def test_real_repo_enumeration_resolves_every_literal_path(self) -> None:
        enumeration = lane_cluster.resolve_cluster_paths(_REPO_ROOT)
        literals = {p for p in lane_cluster.CLUSTER_PATHS if '*' not in p}
        assert literals <= set(enumeration.resolved)

    def test_resolved_paths_are_repo_relative_forward_slash(self) -> None:
        enumeration = lane_cluster.resolve_cluster_paths(_REPO_ROOT)
        for path in enumeration.resolved:
            assert not path.startswith('/'), path
            assert '\\' not in path, path

    def test_glob_expanding_to_zero_keeps_the_enumeration_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # This is orchestrator/src/orchestrator/merge_lane/** today: the package
        # arrives in PRD task zeta1. A glob that matches nothing is fine; a
        # LITERAL that matches nothing is the finding (next test).
        (tmp_path / 'a.py').write_text('x = 1\n', encoding='utf-8')
        monkeypatch.setattr(lane_cluster, 'CLUSTER_PATHS', ('a.py', 'nowhere/**/*.py'))
        enumeration = lane_cluster.resolve_cluster_paths(tmp_path)
        assert enumeration.complete is True
        assert enumeration.resolved == ('a.py',)

    def test_missing_literal_path_raises_naming_the_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(lane_cluster, 'CLUSTER_PATHS', ('gone/missing.py',))
        with pytest.raises(lane_cluster.ClusterSpecError) as excinfo:
            lane_cluster.resolve_cluster_paths(tmp_path)
        assert 'gone/missing.py' in str(excinfo.value)

    def test_missing_literal_path_never_returns_a_partial_enumeration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The INV-11 seam: a present sibling must not let the missing path be
        # silently dropped from an otherwise plausible-looking result.
        (tmp_path / 'a.py').write_text('x = 1\n', encoding='utf-8')
        monkeypatch.setattr(lane_cluster, 'CLUSTER_PATHS', ('a.py', 'gone/missing.py'))
        with pytest.raises(lane_cluster.ClusterSpecError):
            lane_cluster.resolve_cluster_paths(tmp_path)

    def test_literal_path_that_is_a_directory_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / 'adir.py').mkdir()
        monkeypatch.setattr(lane_cluster, 'CLUSTER_PATHS', ('adir.py',))
        with pytest.raises(lane_cluster.ClusterSpecError) as excinfo:
            lane_cluster.resolve_cluster_paths(tmp_path)
        assert 'adir.py' in str(excinfo.value)


class TestEnumerationRoundTrip:
    def test_to_dict_carries_the_four_completeness_keys(self) -> None:
        enumeration = lane_cluster.Enumeration(
            requested=('a.py', 'b/**/*.py'),
            resolved=('a.py',),
            unreadable=(),
            complete=True,
        )
        assert enumeration.to_dict() == {
            'requested': ['a.py', 'b/**/*.py'],
            'resolved': ['a.py'],
            'unreadable': [],
            'complete': True,
        }

    def test_to_dict_is_json_serialisable(self) -> None:
        import json

        enumeration = lane_cluster.Enumeration(
            requested=('a.py',), resolved=('a.py',), unreadable=('c.py',), complete=False
        )
        assert json.loads(json.dumps(enumeration.to_dict()))['complete'] is False

    def test_enumeration_is_frozen(self) -> None:
        import dataclasses

        enumeration = lane_cluster.Enumeration(
            requested=(), resolved=(), unreadable=(), complete=True
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            enumeration.complete = False  # type: ignore[misc]

    def test_unreadable_entries_make_the_enumeration_incomplete(self) -> None:
        # `complete` is the single legible signal a caller checks; it must
        # never read True while files were skipped.
        enumeration = lane_cluster.Enumeration(
            requested=('a.py',), resolved=(), unreadable=('a.py',), complete=False
        )
        assert enumeration.complete is False
        unreadable = enumeration.to_dict()['unreadable']
        assert isinstance(unreadable, list)
        assert 'a.py' in unreadable


# ---------------------------------------------------------------------------
# The per-file measures over the REAL cluster.
#
# Their unit tests live in tests/scripts/test_source_measures.py, pinned on
# inline synthetic snippets. What stays here is the anti-vacuity half: each
# measure, run over the real cluster, returns a real number. None of these pins
# the tree's current state.


class TestTheMeasuresOverTheRealCluster:
    def test_real_merge_queue_line_count_anchor(self) -> None:
        # Anti-vacuity anchor: proves the measure is wired to the REAL tree
        # and returns a real number. It deliberately does NOT pin the tree's
        # current state -- shrinking this file is the whole point of
        # plans/merge-lane-quality-prd.md, so the floor sits far below the
        # 21,550 measured on the introducing commit (the PRD Background
        # table's figure).
        #
        # A FALL is not a ratchet violation: the ratchet reports RISES only and
        # stays green. What reports a fall is
        # test_baseline_matches_a_fresh_measurement, which asserts the committed
        # baseline is byte-identical to a fresh measurement -- and its remedy,
        # regenerating the baseline with --write-baseline in the same commit,
        # actually applies. That is where exactness lives; here, only the
        # anti-vacuity property.
        source = (_REPO_ROOT / WORKER_RELPATH).read_text(encoding='utf-8')
        measures = source_measures.file_size_measures(source, path='merge_queue.py')
        assert measures.lines >= 10000
        assert measures.prose_lines > 0


    def test_cluster_total_is_positive_anchor(self) -> None:
        # Anti-vacuity: the PRD's ceilings row counts the function-local imports
        # inside the package; on this tree the count is well above zero.
        total = 0
        for relpath in lane_cluster.resolve_cluster_paths(_REPO_ROOT).resolved:
            source = (_REPO_ROOT / relpath).read_text(encoding='utf-8')
            total += source_measures.function_local_imports(source, path=relpath)
        assert total > 0


    def test_merge_queue_anchor_reproduces_the_prd_background_numbers(self) -> None:
        # Anti-vacuity anchor: complexipy really ran over the real file and
        # attributed scores to real qualnames. Measured on the introducing
        # commit, and the exact figures the PRD Background table quotes:
        # _verifier_loop 245, stop 109, file total 2133.
        #
        # FLOORS, not equalities. Lowering exactly these numbers is what the
        # gamma/theta-pi tasks in plans/merge-lane-quality-prd.md exist to do,
        # and a fall is permitted by the ratchet contract in this module's
        # docstring. Exactness is preserved by
        # test_baseline_matches_a_fresh_measurement (byte-identity against the
        # committed baseline), whose --write-baseline remedy actually applies.
        #
        # A complexipy algorithm change would also shift these (this file
        # measures 2031 at 3.0.0, 2124 at 4.0.0, 2092 at 5.0.0, and 2133 at both
        # 6.x and 7.0.1). Catching THAT here would be a redundant backstop, so
        # these floors are deliberately not sized for it: COMPLEXIPY_REQUIRED
        # ('>=6.2,<7'), require_complexipy(),
        # test_pyproject_pin_matches_the_scripts_requirement and the
        # comparator's params.complexipy_version check already hard-block every
        # other major with a named failure.
        target = _REPO_ROOT / WORKER_RELPATH
        measures = source_measures.file_cognitive_measures(target)
        assert measures.per_function['SpeculativeMergeWorker::_verifier_loop'] >= 100
        assert measures.per_function['SpeculativeMergeWorker::stop'] >= 50
        assert measures.total >= 1000


    def test_merge_queue_anchor_is_zero(self) -> None:
        # The PRD Background table's "Maintainability index (radon) | 0" row.
        # Reported, never ratcheted -- but genuinely exercised, so the `radon`
        # dev-group entry is not dead weight.
        source = (_REPO_ROOT / WORKER_RELPATH).read_text(encoding='utf-8')
        assert source_measures.maintainability_index(source, path='merge_queue.py') == 0.0


# ---------------------------------------------------------------------------
# Test-suite patch targets into lane internals: the ratchet's binding of the
# patch-target measure to the lane package and its submodules, driven through
# its public seam. Every snippet imports a lane module, so it is in the
# measure's domain.

_LANE_IMPORT = 'from orchestrator.merge_lane.worker import W\n'

#: A synthetic module universe, so these rows do not depend on the real package.
#: SpeculativeMergeWorker is deliberately absent: a class is not a submodule.
_PATCH_MODULES = frozenset(
    {
        'orchestrator.merge_lane',
        'orchestrator.merge_lane.worker',
        'orchestrator.merge_lane.gates',
        'orchestrator.merge_lane.pkg',
        'orchestrator.merge_lane.pkg.leaf',
    }
)


class TestLanePatchTargets:
    @pytest.mark.parametrize(
        ('body', 'expected'),
        [
            pytest.param(
                "patch('orchestrator.merge_lane.worker.run_scoped_verification', x)\n",
                ['worker.run_scoped_verification'],
                id='worker-string-path',
            ),
            # The pre-package path task eta deleted is no lane module any more:
            # a string still naming it patches nothing that exists.
            pytest.param(
                "patch('orchestrator.merge_queue.run_scoped_verification', x)\n",
                [],
                id='deleted-merge-queue-string-path-not-counted',
            ),
            # Measured before the package existed, deliberately: without it, PRD
            # task zeta2's `git mv` would move every patch target out from under
            # the measure and the count would read 0 by RELOCATION rather than by
            # the migration gamma1..gamma10 actually performs.
            pytest.param(
                "monkeypatch.setattr('orchestrator.merge_lane.foo', x)\n",
                ['foo'],
                id='merge-lane-prefix',
            ),
            pytest.param(
                "patch('orchestrator.merge_lane.worker.spin', x)\n",
                ['worker.spin'],
                id='merge-lane-submodule-leaf',
            ),
            pytest.param(
                "from orchestrator import merge_lane\nmonkeypatch.setattr(merge_lane, 'x', 1)\n",
                ['x'],
                id='from-import-binding',
            ),
            pytest.param(
                "import orchestrator.merge_lane as ml\nmonkeypatch.setattr(ml, 'y', 1)\n",
                ['y'],
                id='import-as-binding',
            ),
            pytest.param(
                "from orchestrator import merge_lane\np = patch.object(merge_lane, 'z', 1)\n",
                ['z'],
                id='merge-lane-binding',
            ),
            pytest.param(
                "patch.object(orchestrator.merge_lane, 'x', 1)\n",
                ['x'],
                id='bare-attribute-chain',
            ),
            # Already repointed to the defining satellite -- that is the END
            # STATE this measure is driving towards, so counting it would
            # penalise the fix.
            pytest.param(
                "from orchestrator import merge_gates\np = patch.object(merge_gates, 'x', 1)\n",
                [],
                id='satellite-not-counted',
            ),
            pytest.param(
                "patch.object(workflow.merge_lane, 'x', 1)\n",
                [],
                id='unrelated-attribute-sharing-the-leaf-not-counted',
            ),
            pytest.param(
                "_GATE = 'orchestrator.merge_lane.worker.gate_mark_done_on_delivered_checks'\n"
                'patch(_GATE, x)\n',
                ['worker.gate_mark_done_on_delivered_checks'],
                id='module-constant-indirection',
            ),
            pytest.param(
                "_WORKER = 'orchestrator.merge_lane.worker'\npatch(f'{_WORKER}.spin', x)\n",
                ['worker.spin'],
                id='f-string-over-module-constant',
            ),
            pytest.param(
                'from orchestrator.merge_lane import gates\n'
                "monkeypatch.setattr(gates, 'POST_ADVANCE_GATES', x)\n",
                ['gates.POST_ADVANCE_GATES'],
                id='from-import-of-a-submodule',
            ),
            pytest.param(
                'from orchestrator.merge_lane import worker\n'
                "p = patch.object(worker, '_debug', 1)\n",
                ['worker._debug'],
                id='from-import-of-a-submodule-private-name',
            ),
            pytest.param(
                'import orchestrator.merge_lane.worker as mq\n'
                "monkeypatch.setattr(mq, '_x', 1)\n",
                ['worker._x'],
                id='import-as-of-a-submodule',
            ),
            pytest.param(
                "patch.object(orchestrator.merge_lane.gates, 'x', 1)\n",
                ['gates.x'],
                id='bare-attribute-chain-to-a-submodule',
            ),
            pytest.param(
                'from orchestrator.merge_lane.pkg import leaf\n'
                "monkeypatch.setattr(leaf, '_y', 1)\n",
                ['pkg.leaf._y'],
                id='from-import-of-a-nested-submodule',
            ),
            pytest.param(
                'from orchestrator.merge_lane import gates\n'
                "monkeypatch.setattr(gates, 'X', 1)\n"
                "patch('orchestrator.merge_lane.gates.X', y)\n",
                ['gates.X'],
                id='object-and-string-forms-share-one-leaf',
            ),
            pytest.param(
                'from orchestrator.merge_lane import SpeculativeMergeWorker\n'
                "patch.object(SpeculativeMergeWorker, '_x', 1)\n",
                [],
                id='non-module-name-from-the-package-not-counted',
            ),
        ],
    )
    def test_the_lane_binding(self, body: str, expected: list[str]) -> None:
        measures = lane_cluster.test_file_measures(
            _LANE_IMPORT + body, path='t.py', patch_modules=_PATCH_MODULES
        )
        assert measures is not None
        assert measures['patch_targets'] == expected

    def test_real_tree_union_anchor(
        self, live_report: dict, no_private_tree_scan: None
    ) -> None:
        # Anti-vacuity, read off the ONE live sweep rather than a private
        # re-walk of the same tree. The PRD Background table's "79 distinct
        # names", and the 78 the string-path form alone measured, are DAY-ONE
        # figures and stay written as they were.
        #
        # The report's union covers only
        # lane-importing files, where the old body unioned every *.py in
        # orchestrator/tests. In principle the whole-tree union could be
        # larger, since a string-path patch needs no import -- MEASURED on this
        # tree it is not: both unions are equal and their set difference is
        # EMPTY. The whole-tree DENOMINATOR stays witnessed by
        # TestBaselineIsNotVacuous::test_the_live_sweep_denominator_covers_the_whole_test_tree,
        # which floors `enumeration.requested` at 400 orchestrator/tests paths
        # on this same live report. So this anchor now floors the measure the
        # gate actually gates -- the same number `derive_totals` computes and
        # the committed baseline ratchets.
        #
        # The FLOOR is deliberately slack, and stays that way. The gamma wave
        # retires patch targets by design, so the union falls as the wave lands
        # and a floor pinned near the current magnitude becomes a countdown
        # timer against the work it measures. It first bit on gamma4 (task
        # 5027) as pure INTEGRATION SKEW -- gamma1 and gamma4 each measured 71
        # alone and 69 merged, because each retires names the other still
        # counts, so neither branch could see it before the merge. The
        # substantive anti-vacuity check is the membership assertion below,
        # which a broken detector cannot satisfy at any magnitude; task 5446
        # retires the numeric floor outright when it drives both measures to 0.
        union: set[str] = set()
        for entry in live_report['tests'].values():
            union.update(entry['patch_targets'])
        assert len(union) >= 40, len(union)
        assert 'worker.run_scoped_verification' in union


_LANE_SRC = 'orchestrator/src/orchestrator/merge_lane'


class TestLanePatchModules:
    def test_the_package_and_every_module_under_it_on_disk(self, tmp_path: Path) -> None:
        root = _committed_tree(
            tmp_path,
            {
                f'{_LANE_SRC}/__init__.py': '',
                f'{_LANE_SRC}/gates.py': '',
                f'{_LANE_SRC}/pkg/__init__.py': '',
                f'{_LANE_SRC}/pkg/leaf.py': '',
            },
        )
        assert lane_cluster.lane_patch_modules(root) == {
            'orchestrator.merge_lane',
            'orchestrator.merge_lane.gates',
            'orchestrator.merge_lane.pkg',
            'orchestrator.merge_lane.pkg.leaf',
        }

    def test_a_tree_without_the_package_still_names_the_package(self, tmp_path: Path) -> None:
        root = _committed_tree(tmp_path, {'README.md': 'x\n'})
        assert lane_cluster.lane_patch_modules(root) == {lane_cluster.LANE_PACKAGE}


# ---------------------------------------------------------------------------
# Lane-importing test files.


class TestImportsLaneModule:
    def test_import_from_a_cluster_module_is_lane_importing(self) -> None:
        source = 'from orchestrator.offline_lane import OfflineLaneWorker\n'
        assert lane_cluster.imports_lane_module(source, path='t.py') is True

    def test_plain_import_of_merge_lane_is_lane_importing(self) -> None:
        assert lane_cluster.imports_lane_module('import orchestrator.merge_lane\n', path='t.py') is True

    def test_import_from_a_merge_lane_submodule_is_lane_importing(self) -> None:
        source = 'from orchestrator.merge_lane.worker import W\n'
        assert lane_cluster.imports_lane_module(source, path='t.py') is True

    def test_from_orchestrator_import_of_a_cluster_leaf_is_lane_importing(self) -> None:
        assert lane_cluster.imports_lane_module('from orchestrator import offline_lane\n', path='t.py') is True

    def test_from_orchestrator_import_of_the_package_is_lane_importing(self) -> None:
        assert lane_cluster.imports_lane_module('from orchestrator import merge_lane\n', path='t.py') is True

    @pytest.mark.parametrize(
        'source',
        [
            'import orchestrator.moved_lane\n',
            'from orchestrator.moved_lane.worker import W\n',
            'from orchestrator import moved_lane\n',
        ],
    )
    def test_the_package_is_named_by_lane_package_alone(
        self, source: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(lane_cluster, 'LANE_PACKAGE', 'orchestrator.moved_lane')
        assert lane_cluster.imports_lane_module(source, path='t.py') is True

    def test_an_unrelated_orchestrator_module_is_not_lane_importing(self) -> None:
        assert lane_cluster.imports_lane_module('import orchestrator.workflow\n', path='t.py') is False

    def test_a_module_with_no_imports_is_not_lane_importing(self) -> None:
        assert lane_cluster.imports_lane_module('x = 1\n', path='t.py') is False

    def test_the_module_name_set_is_derived_from_cluster_paths(self) -> None:
        # Derived, never hand-listed, so the two can't drift when Appendix A is
        # edited.
        assert 'orchestrator.offline_lane' in lane_cluster.lane_module_names()
        assert 'orchestrator.git_ops' in lane_cluster.lane_module_names()
        assert 'orchestrator.merge_queue' not in lane_cluster.lane_module_names()
        assert 'orchestrator.workflow' not in lane_cluster.lane_module_names()


class TestTheTreeAndSourceFormsAgree:
    """Each lane measure has a source-taking form and a tree-taking one.

    The sweeps parse once and call the tree form; everything else calls the
    source form, which is a parse plus a call of the tree form. They are one
    implementation, and this is what notices a change that makes them two.
    """

    _SOURCE = (
        '"""Docstring."""\n'
        'from orchestrator.merge_lane import worker\n'
        'from unittest.mock import patch\n'
        'from pathlib import Path\n\n'
        '# a comment\n'
        'def test_x():\n'
        '    from json import dumps\n'
        "    with patch('orchestrator.merge_lane.worker.helper'):\n"
        '        assert worker._worker\n'
    )

    @pytest.mark.parametrize(
        ('from_source', 'from_tree'),
        [
            pytest.param(
                lambda src: lane_cluster.imports_lane_module(src, path='t.py'),
                lambda src, tree: lane_cluster.imports_lane_module_in_tree(tree),
                id='imports_lane_module',
            ),
            pytest.param(
                lambda src: lane_cluster.test_file_measures(
                    src, path='t.py', patch_modules=_PATCH_MODULES
                ),
                lambda src, tree: lane_cluster.test_file_measures_in_tree(tree, _PATCH_MODULES),
                id='test_file_measures',
            ),
        ],
    )
    def test_the_two_forms_return_the_same_measure(self, from_source, from_tree) -> None:
        tree = source_measures.parse_source(self._SOURCE, path='t.py')
        expected = from_source(self._SOURCE)
        assert from_tree(self._SOURCE, tree) == expected
        # ANTI-VACUITY: the snippet exercises every measure, so none is trivially
        # equal as an empty value.
        assert expected not in (0, [], set(), frozenset(), None)


class TestTestFileMeasures:
    def test_a_non_lane_importing_file_yields_none(self) -> None:
        # The exclusion is a property of the MEASURE, not of the caller: a
        # non-lane-importing file can never be summed into the report by
        # accident.
        source = 'import orchestrator.workflow\n\nworker._x\n'
        assert lane_cluster.test_file_measures(source, path='t.py', patch_modules=_PATCH_MODULES) is None

    def test_a_lane_importing_file_yields_both_measures(self) -> None:
        source = (
            'from orchestrator.merge_lane.worker import W\n'
            '\n'
            "patch('orchestrator.merge_lane.worker.foo', x)\n"
            'worker._a._b\n'
        )
        assert lane_cluster.test_file_measures(source, path='t.py', patch_modules=_PATCH_MODULES) == {
            'patch_targets': ['worker.foo'],
            'private_reads': 2,
        }

    def test_patch_targets_are_sorted(self) -> None:
        source = (
            'from orchestrator.merge_lane.worker import W\n'
            "patch('orchestrator.merge_lane.worker.z', x)\n"
            "patch('orchestrator.merge_lane.worker.a', x)\n"
        )
        measures = lane_cluster.test_file_measures(source, path='t.py', patch_modules=_PATCH_MODULES)
        assert measures is not None
        assert measures['patch_targets'] == ['worker.a', 'worker.z']

    def test_real_tree_anchors(
        self, live_report: dict, no_private_tree_scan: None
    ) -> None:
        # Anti-vacuity, read off the ONE live sweep rather than a private
        # re-walk of the same tree: >= 150 lane-importing files (measured 229)
        # and a cluster-wide private-read total > 5000 (measured 7,847).
        #
        # `derive_totals` is the single encoding of the cluster-wide sum, so
        # this floors the very number the ratchet compares rather than a
        # second summation of the same per-file entries.
        lane_files = len(live_report['tests'])
        total_private = ratchet.derive_totals(live_report)['private_reads']
        assert lane_files >= 150, lane_files
        assert total_private > 5000, total_private


# ---------------------------------------------------------------------------
# Report assembly, and DERIVED totals.


class TestDeriveTotals:
    def test_sums_every_file_measure(self) -> None:
        totals = ratchet.derive_totals(synthetic_report())
        assert totals['lines'] == 1200
        assert totals['prose_lines'] == 450
        assert totals['cognitive'] == 150
        assert totals['function_local_imports'] == 4
        assert totals['reexport_names'] == 5

    def test_private_reads_are_summed_over_tests(self) -> None:
        assert ratchet.derive_totals(synthetic_report())['private_reads'] == 25

    def test_patch_targets_total_is_the_union_size_not_the_sum(self) -> None:
        # The PRD's measure is DISTINCT names: foo/bar/baz across two files that
        # both patch `bar` is 3, not 4.
        assert ratchet.derive_totals(synthetic_report())['patch_targets'] == 3

    def test_moving_code_to_a_new_path_leaves_every_total_identical(self) -> None:
        # THE ANTI-RENAME-GAMING PROPERTY. Totals are DERIVED by summing the
        # stored per-path map rather than stored as their own key, so moving 500
        # lines and 40 cognitive from a.py to a brand-new path cannot lower the
        # cluster figure -- the ratchet still catches the move.
        before = synthetic_report()
        after = synthetic_report()
        after['files']['a.py']['lines'] -= 500
        after['files']['a.py']['cognitive'] -= 40
        after['files']['new_module.py'] = {
            'lines': 500,
            'prose_lines': 0,
            'cognitive': 40,
            'function_local_imports': 0,
            'reexport_names': 0,
        }
        assert ratchet.derive_totals(after) == ratchet.derive_totals(before)

    def test_is_a_pure_function_of_the_report(self) -> None:
        report = synthetic_report()
        snapshot = json.dumps(report, sort_keys=True)
        ratchet.derive_totals(report)
        assert json.dumps(report, sort_keys=True) == snapshot


class TestBuildReport:
    @pytest.fixture()
    def report(self, live_report: dict) -> dict:
        return live_report

    def test_top_level_keys(self, report: dict) -> None:
        assert set(report) == {
            'schema_version',
            'params',
            'enumeration',
            'files',
            'functions',
            'tests',
        }

    def test_no_stored_totals_key(self, report: dict) -> None:
        # A stored totals block is the ONE shared line all ten of PRD
        # gamma1..gamma10 would each rewrite, conflicting on every rebase. It is
        # also two representations of one number, free to disagree (SPOT).
        assert 'totals' not in report

    def test_schema_version(self, report: dict) -> None:
        assert report['schema_version'] == 1

    def test_params_records_how_the_measurement_was_taken(self, report: dict) -> None:
        params = report['params']
        assert params['complexipy_version'] == source_measures.complexipy_version()
        assert params['cluster_paths'] == list(lane_cluster.CLUSTER_PATHS)
        assert params['file_line_ceiling'] == lane_cluster.FILE_LINE_CEILING
        assert params['new_function_cognitive_ceiling'] == (
            lane_cluster.NEW_FUNCTION_COGNITIVE_CEILING
        )

    def test_enumeration_is_complete(self, report: dict) -> None:
        assert report['enumeration']['complete'] is True
        assert report['enumeration']['unreadable'] == []

    def test_file_entries_carry_the_five_measures(self, report: dict) -> None:
        entry = report['files'][WORKER_RELPATH]
        assert set(entry) == {
            'lines',
            'prose_lines',
            'cognitive',
            'function_local_imports',
            'reexport_names',
        }
        # Floors, not equalities -- same reasoning as
        # test_real_merge_queue_line_count_anchor, on the same two numbers
        # (21,550 lines / cognitive 2,133 when this landed). Byte-exactness is
        # asserted by test_baseline_matches_a_fresh_measurement.
        assert entry['lines'] >= 10000
        assert entry['cognitive'] >= 1000

    def test_functions_is_a_flat_path_qualname_map(self, report: dict) -> None:
        key = f'{WORKER_RELPATH}::SpeculativeMergeWorker::_verifier_loop'
        # Floor, not equality (measured 245): a fall is permitted by the
        # ratchet contract -- see test_real_merge_queue_line_count_anchor.
        assert report['functions'][key] >= 100
        assert all(isinstance(value, int) for value in report['functions'].values())

    def test_tests_entries_carry_both_measures_and_only_lane_files(
        self, report: dict
    ) -> None:
        assert report['tests']
        for path, entry in report['tests'].items():
            assert set(entry) == {'patch_targets', 'private_reads'}
            assert entry['patch_targets'] == sorted(entry['patch_targets']), path
        assert len(report['tests']) >= 150

    def test_every_path_key_is_repo_relative_forward_slash(self, report: dict) -> None:
        for key in list(report['files']) + list(report['tests']):
            assert not key.startswith('/'), key
            assert '\\' not in key, key

    def test_cluster_cognitive_total_anchor(self, report: dict) -> None:
        # Anti-vacuity: measured 4,607 on this tree.
        assert ratchet.derive_totals(report)['cognitive'] >= 4000

    def test_each_cluster_file_is_measured_by_complexipy_once(
        self, live_measurement: LiveMeasurement
    ) -> None:
        # build_report wants two cognitive projections per cluster file, and
        # used to fetch each from its own complexipy run -- 50 runs
        # over the 25 files CLUSTER_PATHS' 23 entries resolve to (two are
        # globs), half of them redundant.
        #
        # A MULTISET equality, not a length check: `len(recorded) == len(files)`
        # stays green when one file is measured twice and another is skipped,
        # which is precisely the shape a regrown second call site would have.
        cluster = sorted(_REPO_ROOT / relpath for relpath in live_measurement.report['files'])
        # Anti-vacuity: an empty cluster would satisfy the equality below
        # without witnessing a single measurement.
        assert len(cluster) >= 20, len(cluster)
        assert sorted(live_measurement.complexipy_calls) == cluster


# ---------------------------------------------------------------------------
# The two enumeration halves, which are different KINDS of thing: the cluster is
# a named manifest where an unresolved path IS the finding and must be shown by
# name; the test tree is an open sweep of other people's files where only the
# coverage ratio is meaningful. Keeping them apart in the data structure is what
# lets the baseline store one and merely report the other.


def _test_tree_strays(block: dict) -> dict[str, list[str]]:
    """Per-path test-tree entries left in an enumeration block, keyed by list.

    THE HEADLINE PREDICATE, in ONE place. Three tests assert it -- on the live
    report, on a freshly rendered baseline and on the committed bytes -- and a
    copy each would drift the moment CLUSTER_PATHS grows (PRD task zeta1 expands
    `merge_lane/**` to real files), leaving two of the three silently weaker.
    An empty dict is clean; a non-empty one names the offending list and entries.
    """
    cluster = set(lane_cluster.CLUSTER_PATHS)
    strays = {}
    for key, value in block.items():
        if not isinstance(value, list):
            continue
        found = [
            entry for entry in value
            if isinstance(entry, str)
            and entry.startswith(f'{lane_cluster.TESTS_ROOT}/')
            and entry not in cluster
        ]
        if found:
            strays[key] = found
    return strays


class TestReportEnumerationSplitsTheTwoHalves:
    @pytest.fixture()
    def enumeration(self, live_report: dict) -> dict:
        return live_report['enumeration']

    def test_the_key_set_is_exactly_the_five(self, enumeration: dict) -> None:
        # EXACT, not a floor: a sixth key reaching the report is a deliberate
        # act, because it also has to be decided whether it reaches the file.
        assert set(enumeration) == {
            'requested',
            'resolved',
            'test_tree',
            'unreadable',
            'complete',
        }

    def test_the_cluster_half_stays_verbatim_paths(self, enumeration: dict) -> None:
        # The cluster half is PATHS, glob literals included -- those literals are
        # the SPOT record of PRD Appendix A and an unexpectedly-missing one IS
        # the finding, so a count could not express it.
        assert enumeration['requested'] == list(lane_cluster.CLUSTER_PATHS)

    def test_no_test_tree_path_list_survives_anywhere_in_the_block(
        self, enumeration: dict
    ) -> None:
        # THE HEADLINE PROPERTY, executably: the only orchestrator/tests paths
        # left in any list are CLUSTER_PATHS literals. Measured today: exactly
        # the 2 that live there (conftest.py, _merge_queue_harness.py).
        assert _test_tree_strays(enumeration) == {}, (
            'the live report still carries per-path test-tree entries'
        )

    def test_the_test_tree_half_is_two_integer_counts(self, enumeration: dict) -> None:
        test_tree = enumeration['test_tree']
        assert set(test_tree) == {'requested', 'resolved'}
        for name, value in test_tree.items():
            # bool is an int subclass, so exclude it explicitly -- a count that
            # collapsed to True/False would otherwise pass as an integer.
            assert isinstance(value, int) and not isinstance(value, bool), name

    def test_the_numerator_is_exactly_the_measured_test_count(
        self, live_report: dict
    ) -> None:
        # THE SPOT RELATION that makes storing this number redundant: it is
        # len(report['tests']), already in the baseline line-locally.
        assert live_report['enumeration']['test_tree']['resolved'] == len(
            live_report['tests']
        )

    def test_completeness_is_carried_verbatim(self, enumeration: dict) -> None:
        # INV-11: the split narrows nothing about completeness.
        assert enumeration['unreadable'] == []
        assert enumeration['complete'] is True


def _committed_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    """A throwaway tree holding *files* under ``tmp_path``, built by the suite's own repo recipe."""
    return build_repo(
        tmp_path / 'repo', RepoSeed(files=tuple(files.items()), message='seed')
    )


class TestTestTreeSweep:
    """`sweep_repo` in isolation -- no complexipy, so ms."""

    @staticmethod
    def _tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setattr(lane_cluster, 'TESTS_ROOT', 'orchestrator/tests')
        return _committed_tree(
            tmp_path,
            {
                'orchestrator/tests/test_lane.py': (
                    'from orchestrator.merge_lane import worker\n\n'
                    'def test_x():\n'
                    '    assert worker._worker is None\n'
                ),
                'orchestrator/tests/test_unrelated.py': (
                    'import json\n\ndef test_y():\n    assert json\n'
                ),
                'orchestrator/tests/test_broken.py': 'def (:\n',
            },
        )

    def test_counts_are_integers_not_path_lists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The sweep's contract: 3 files asked for, 1 measured, as COUNTS.
        sweep = lane_cluster.sweep_repo(self._tree(tmp_path, monkeypatch))
        assert sweep.test_tree.requested == 3
        assert sweep.test_tree.resolved == 1
        assert list(sweep.tests) == ['orchestrator/tests/test_lane.py']

    def test_the_unparseable_file_is_named_and_only_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # INV-11 POLARITY at the sweep boundary: the skip is fail-SOFT per file
        # (an unrelated mid-edit test must not redden the lane's ratchet) but the
        # RECORD is not -- the path is named verbatim.
        sweep = lane_cluster.sweep_repo(self._tree(tmp_path, monkeypatch))
        assert list(sweep.unreadable) == ['orchestrator/tests/test_broken.py']

    def test_a_skipped_file_marks_the_composed_block_incomplete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The seam that makes check_against_baseline REFUSE to compare: the
        # sweep's soft skip must still reach `complete` in the report block.
        sweep = lane_cluster.sweep_repo(self._tree(tmp_path, monkeypatch))
        cluster = lane_cluster.Enumeration(
            requested=('a.py',), resolved=('a.py',), unreadable=(), complete=True
        )
        block = lane_cluster._report_enumeration(cluster, sweep.unreadable, sweep.test_tree)
        assert block['complete'] is False
        assert block['unreadable'] == ['orchestrator/tests/test_broken.py']
        with pytest.raises(ratchet.RatchetError) as excinfo:
            ratchet._require_complete_enumeration({'enumeration': block})
        assert 'orchestrator/tests/test_broken.py' in str(excinfo.value)

    def test_a_clean_sweep_composes_a_complete_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The other polarity, so the test above cannot pass vacuously.
        monkeypatch.setattr(lane_cluster, 'TESTS_ROOT', 'orchestrator/tests')
        root = _committed_tree(
            tmp_path, {'orchestrator/tests/test_ok.py': 'import json\n'}
        )
        sweep = lane_cluster.sweep_repo(root)
        cluster = lane_cluster.Enumeration(
            requested=('a.py',), resolved=('a.py',), unreadable=(), complete=True
        )
        block = lane_cluster._report_enumeration(cluster, sweep.unreadable, sweep.test_tree)
        assert block['complete'] is True
        assert block['unreadable'] == []
        assert block['test_tree'] == {'requested': 1, 'resolved': 0}

    def test_a_patch_through_a_submodule_binding_is_counted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``only_in_tmp`` exists in no real checkout, so a module set read from
        # anywhere but the swept root would leave the receiver unbound.
        monkeypatch.setattr(lane_cluster, 'TESTS_ROOT', 'orchestrator/tests')
        root = _committed_tree(
            tmp_path,
            {
                f'{_LANE_SRC}/__init__.py': '',
                f'{_LANE_SRC}/only_in_tmp.py': '',
                'orchestrator/tests/test_g.py': (
                    'from orchestrator.merge_lane import only_in_tmp\n\n'
                    'def test_g(monkeypatch):\n'
                    "    monkeypatch.setattr(only_in_tmp, 'X', [])\n"
                ),
            },
        )
        sweep = lane_cluster.sweep_repo(root)
        assert sweep.tests == {
            'orchestrator/tests/test_g.py': {
                'patch_targets': ['only_in_tmp.X'],
                'private_reads': 0,
            }
        }

    def test_an_unreadable_cluster_path_is_not_masked_by_a_clean_sweep(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Both halves' unreadable paths travel verbatim in ONE top-level list --
        # the cluster half's finding must not be diluted or dropped.
        monkeypatch.setattr(lane_cluster, 'TESTS_ROOT', 'orchestrator/tests')
        sweep = lane_cluster.sweep_repo(_committed_tree(tmp_path, {'README.md': 'x\n'}))
        cluster = lane_cluster.Enumeration(
            requested=('a.py',), resolved=(), unreadable=('a.py',), complete=False
        )
        block = lane_cluster._report_enumeration(cluster, sweep.unreadable, sweep.test_tree)
        assert block['unreadable'] == ['a.py']
        assert block['complete'] is False


# ---------------------------------------------------------------------------
# Baseline serialization of the enumeration block.


class TestStoredEnumerationIsChurnFree:
    """The committed enumeration moves on lane churn ONLY, never on test-tree churn.

    esc-5021-7: freezing all 583 orchestrator/tests paths made ANY unrelated
    test-file addition redden test_baseline_matches_a_fresh_measurement, and the
    message it printed ("Regenerate it in this commit if you lowered a measure")
    invited a blind regeneration of a baseline that task had never inspected --
    exactly the silent widening BASELINE_README exists to prevent. Observed live:
    test_roles_error_remedy_hint.py arriving via rebase (task 4964) produced a
    one-line baseline diff and a red gate.

    The counts alternative is provably worse, which is why the numbers are not
    stored either: a shared count line is the shape this PRD's decompose-time
    constraint already rejected for `totals` ("would conflict on every one of ten
    rebases"), and a guaranteed one-line conflict beats a probabilistic
    multi-line one only in the wrong direction.
    """

    @staticmethod
    def _report(**enumeration: object) -> dict:
        report = synthetic_report()
        report['enumeration'].update(enumeration)
        return report

    @staticmethod
    def _stored(report: dict) -> dict:
        return json.loads(ratchet.render_baseline(report))['enumeration']

    def test_the_stored_key_set_omits_the_test_tree_counts(self) -> None:
        # No test-tree NUMBER reaches the file, only the cluster half plus
        # INV-11's two completeness keys.
        assert set(self._stored(synthetic_report())) == {
            'requested',
            'resolved',
            'unreadable',
            'complete',
        }

    def test_no_test_tree_path_reaches_the_stored_block(self) -> None:
        # THE BUG, stated over the real measurement rather than a synthetic one:
        # a swept-but-never-measured file must not reach the baseline, and the
        # only orchestrator/tests paths left are CLUSTER_PATHS literals.
        report = self._report(
            requested=list(lane_cluster.CLUSTER_PATHS),
            resolved=['orchestrator/tests/conftest.py'],
        )
        assert _test_tree_strays(self._stored(report)) == {}, (
            'a swept-but-never-measured file reached the stored block'
        )

    def test_cluster_paths_are_kept_including_the_glob_literals(self) -> None:
        # CLUSTER_PATHS entries are the SPOT record of PRD Appendix A. The glob
        # LITERALS never appear in `resolved`, so a resolved-only stored list
        # would silently delete them -- and a glob expanding to zero would stop
        # being legible in the RESULT.
        globs = [entry for entry in lane_cluster.CLUSTER_PATHS if '*' in entry]
        assert globs, 'expected at least one glob entry in CLUSTER_PATHS'
        report = self._report(requested=list(lane_cluster.CLUSTER_PATHS))
        kept = self._stored(report)['requested']
        assert kept == list(lane_cluster.CLUSTER_PATHS)
        for glob in globs:
            assert glob in kept

    def test_an_unreadable_path_is_kept_so_inv_11_still_reads(self) -> None:
        # INV-11: a partial enumeration must stay distinguishable in the RESULT.
        # The stored `requested` is the cluster manifest and no longer re-admits
        # a skipped path, so `unreadable` carries that payload alone -- and it is
        # a TEST-TREE path here, the half that is otherwise reduced to counts.
        bad = 'orchestrator/tests/test_broken.py'
        stored = self._stored(self._report(unreadable=[bad], complete=False))
        assert stored['unreadable'] == [bad]
        assert stored['complete'] is False

    def test_the_transform_is_idempotent(self) -> None:
        # render_baseline's round-trip contract: regenerating a baseline FROM a
        # baseline must be a no-op, not a second round of deletions.
        once = self._stored(synthetic_report())
        assert ratchet._stored_enumeration(once) == once

    def test_the_rendered_baseline_is_stable_across_an_unrelated_new_test(
        self,
    ) -> None:
        # THE REGRESSION, end to end and in the units that bit (esc-5021-7):
        # adding one non-lane test file to the sweep must not change a single
        # byte. Now STRENGTHENED to both counts, because storing either one would
        # reintroduce it -- `requested` moves on ANY .py arriving anywhere under
        # orchestrator/tests, and `resolved` moves whenever a lane-importing one
        # is added or deleted, which is a shared line every gamma branch rewrites.
        report = synthetic_report()
        before = ratchet.render_baseline(report)
        report['enumeration']['test_tree']['requested'] += 1
        assert ratchet.render_baseline(report) == before
        report['enumeration']['test_tree']['resolved'] += 1
        assert ratchet.render_baseline(report) == before


# ---------------------------------------------------------------------------
# CLI surface.
#
# --check is the face twenty downstream PRD tasks will actually run, so its exit
# ladder is pinned as carefully as the comparator itself: 0 clean, 1 ratchet
# violations, 2 instrument failure. Collapsing 1 and 2 would let a broken
# instrument read as a real regression, or a real regression as a broken
# instrument -- both send the reader to the wrong file.


@pytest.fixture()
def stub_measurement(monkeypatch: pytest.MonkeyPatch, live_report: dict) -> dict:
    """Hand the CLI this module's one cached measurement.

    The CLI's own job is dispatch, rendering and the exit ladder; build_report
    is already pinned by TestBuildReport against the real tree. Re-measuring
    once per CLI test would add ~7 x build_report to every orchestrator verify
    leg to re-prove something already proven -- several times what the whole
    module now costs.

    It hands back the module's REAL live measurement, not a synthetic one, so a
    CLI test using this fixture still compares live numbers.
    """
    monkeypatch.setattr(
        lane_cluster, 'build_report', lambda root: copy.deepcopy(live_report)
    )
    return live_report


class TestReportCli:
    def test_report_returns_zero_and_names_every_cluster_path(
        self, stub_measurement: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert metrics.main(['--report']) == 0
        out = capsys.readouterr().out
        for path in stub_measurement['files']:
            assert path in out, path

    def test_report_carries_the_per_file_column_set(
        self, stub_measurement: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        metrics.main(['--report'])
        out = capsys.readouterr().out
        for column in ('lines', 'prose', 'cognitive', 'mi'):
            assert column in out, column

    def test_report_carries_the_derived_cluster_totals(
        self, stub_measurement: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        metrics.main(['--report'])
        out = capsys.readouterr().out
        assert 'TOTALS' in out
        totals = ratchet.derive_totals(stub_measurement)
        assert str(totals['cognitive']) in out
        assert str(totals['lines']) in out

    def test_report_still_prints_both_size_columns_and_totals(
        self, stub_measurement: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Size leaves the RATCHET, never the report: --report is where a reader
        # still sees how big each file is.
        assert metrics.main(['--report']) == 0
        rows = capsys.readouterr().out.splitlines()
        totals = ratchet.derive_totals(stub_measurement)

        header = next(row for row in rows if row.startswith('path ')).split()
        assert header[1:3] == ['lines', 'prose']
        totals_row = next(row for row in rows if row.startswith('TOTALS ')).split()
        assert totals_row[1:3] == [str(totals['lines']), str(totals['prose_lines'])]
        for path, entry in stub_measurement['files'].items():
            row = next(row for row in rows if row.startswith(f'{path} ')).split()
            assert row[1] == str(entry['lines']), path

    def test_report_carries_the_test_suite_measures(
        self, stub_measurement: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        metrics.main(['--report'])
        out = capsys.readouterr().out
        totals = ratchet.derive_totals(stub_measurement)
        assert 'private_reads' in out
        assert 'patch_targets' in out
        assert str(totals['private_reads']) in out

    def test_report_carries_an_explicit_enumeration_completeness_row(
        self, stub_measurement: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # INV-11's user-observable signal: completeness is legible in the
        # RESULT, not only in a log line.
        metrics.main(['--report'])
        out = capsys.readouterr().out
        assert 'enumeration' in out
        assert 'complete' in out

    def test_report_carries_the_test_tree_sweep_breadth(
        self, stub_measurement: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The test-tree denominator is deliberately NOT ratcheted into the
        # baseline (it is reported, never enforced -- same call this file already
        # made for `mi`), so --report is the ONLY place a reader can see how
        # broad the sweep was. If it is not here it exists nowhere in the RESULT.
        assert metrics.main(['--report']) == 0
        out = capsys.readouterr().out
        test_tree = stub_measurement['enumeration']['test_tree']
        # Anchored to the enumeration ROW, not to the whole stdout blob: `str(N)
        # in out` passes on any other line that happens to carry those digits, so
        # it would stay green with the field deleted. Read from the report, never
        # hard-coded: the numbers churn with the tree, and pinning them would
        # rebuild the hair trigger this removes.
        row = next(
            line for line in out.splitlines() if line.startswith('enumeration:')
        )
        assert f'test tree requested={test_tree["requested"]}' in row
        # The NUMERATOR is deliberately NOT in that row -- it is len(tests), and
        # one number gets one place -- so its own better-labelled line is where
        # the RESULT has to show it. That this len IS the sweep numerator is the
        # SPOT relation, pinned once by
        # test_the_numerator_is_exactly_the_measured_test_count.
        assert f'test suite -- {len(stub_measurement["tests"])} lane-importing' in out

    def test_json_returns_zero_and_stdout_parses_as_the_report(
        self, stub_measurement: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert metrics.main(['--json']) == 0
        assert json.loads(capsys.readouterr().out) == stub_measurement

    def test_json_carries_the_test_tree_counts_untouched(
        self, stub_measurement: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The count/path reduction is a STORAGE concern and must never reach
        # --json, which is the machine-readable face of the measurement itself.
        assert metrics.main(['--json']) == 0
        emitted = json.loads(capsys.readouterr().out)
        assert emitted['enumeration']['test_tree'] == (
            stub_measurement['enumeration']['test_tree']
        )


class TestCheckCli:
    def test_check_is_clean_against_a_baseline_of_the_same_measurement(
        self, stub_measurement: dict, tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        baseline = tmp_path / 'baseline.json'
        ratchet.write_baseline(baseline, stub_measurement)
        assert metrics.main(['--check', '--baseline', str(baseline)]) == 0
        assert capsys.readouterr().err == ''

    def test_check_returns_one_and_prints_violations_for_a_doctored_baseline(
        self, stub_measurement: dict, tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        doctored = copy.deepcopy(stub_measurement)
        doctored['files'][WORKER_RELPATH]['cognitive'] -= 10
        baseline = tmp_path / 'baseline.json'
        ratchet.write_baseline(baseline, doctored)
        assert metrics.main(['--check', '--baseline', str(baseline)]) == 1
        err = capsys.readouterr().err
        assert WORKER_RELPATH in err
        assert 'cognitive' in err

    def test_check_passes_a_tree_whose_size_alone_rose(
        self, stub_measurement: dict, tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        doctored = copy.deepcopy(stub_measurement)
        doctored['files'][WORKER_RELPATH]['lines'] -= 10
        doctored['files'][WORKER_RELPATH]['prose_lines'] -= 10
        baseline = tmp_path / 'baseline.json'
        ratchet.write_baseline(baseline, doctored)

        assert metrics.main(['--check', '--baseline', str(baseline)]) == 0
        assert capsys.readouterr().err == ''

    def test_check_resolves_its_defaults_and_exits_clean(
        self, stub_measurement: dict, no_private_tree_scan: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # No --baseline and no --root: the argv the twenty downstream PRD tasks
        # will run, exercising the CLI's half of it -- default root resolution,
        # default baseline resolution, `load_baseline`, the comparator and the
        # exit ladder -- against the module's REAL live numbers, since
        # `stub_measurement` hands back the live measurement rather than a
        # fixture's. NOT end to end: `build_report` is stubbed here, as it is
        # at every `main(['--check'])` call site in this module.
        #
        # What moved elsewhere is the half this test used to re-prove: the
        # MEASUREMENT is TestBuildReport's, and the live-tree-versus-committed-
        # baseline RATCHET is test_merge_lane_ratchet_holds'. Both run against
        # this same tree and this same baseline, so the second unstubbed
        # `build_report` here bought nothing but a repeat of the whole sweep.
        #
        # The stub replaces `build_report` ONLY, so the baseline read is real
        # -- which is what makes `no_private_tree_scan` a live check that the
        # counter's `*.py` scoping does not count that .json.
        assert metrics.main(['--check']) == 0
        assert capsys.readouterr().err == ''

    def test_check_hands_build_report_the_resolved_repo_root(
        self, live_report: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The ONE seam `stub_measurement` stops witnessing above: that `main`
        # threads its resolved `args.root` INTO build_report, since the stub
        # ignores its argument. `test_root_defaults_to_the_repo_root` pins only
        # the PARSER default, so without this the threading would be untested.
        roots: list[Path] = []

        def recording(root: Path) -> dict:
            roots.append(root)
            return copy.deepcopy(live_report)

        monkeypatch.setattr(lane_cluster, 'build_report', recording)
        assert metrics.main(['--check']) == 0
        assert [r.resolve() for r in roots] == [_REPO_ROOT.resolve()]


class TestWriteBaselineCli:
    def test_write_baseline_returns_zero_and_writes_the_rendered_bytes(
        self, stub_measurement: dict, tmp_path: Path
    ) -> None:
        target = tmp_path / 'b.json'
        assert metrics.main(['--write-baseline', str(target)]) == 0
        assert target.read_text(encoding='utf-8') == ratchet.render_baseline(
            stub_measurement
        )

    def test_a_plain_regeneration_survives_a_corrupt_ledger(
        self, stub_measurement: dict, tmp_path: Path
    ) -> None:
        # --write-baseline now hands --ledger to write_baseline on every call,
        # so "a plain regeneration never depends on that file's health" has to
        # be pinned rather than read off a call site that no longer says it.
        target = tmp_path / 'b.json'
        ledger = corrupt_ledger(tmp_path)

        assert metrics.main([
            '--write-baseline', str(target), '--ledger', str(ledger)
        ]) == 0
        assert target.read_text(encoding='utf-8') == ratchet.render_baseline(
            stub_measurement
        )
        assert ledger_files(ledger) == {'corrupt.json': '{ not json at all'}


class TestAuthorizeRaiseCli:
    """The authorized-raise flags, and the exit ladder the refusal lands on.

    Every case drives `stub_measurement`, so the new surface adds zero
    build_report calls to the orchestrator verify leg.
    """

    @staticmethod
    def _seeded(stub: dict, tmp_path: Path) -> tuple[Path, Path, dict]:
        """A destination holding the stubbed report, and a report that RAISES."""
        target = tmp_path / 'b.json'
        ratchet.write_baseline(target, stub)
        raised = copy.deepcopy(stub)
        raised['files'][WORKER_RELPATH]['cognitive'] += 15
        return target, tmp_path / 'raises', raised

    def test_an_authorized_raise_is_written_and_recorded(
        self, stub_measurement: dict, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        target, ledger, raised = self._seeded(stub_measurement, tmp_path)
        monkeypatch.setattr(lane_cluster, 'build_report', lambda root: copy.deepcopy(raised))

        assert metrics.main([
            '--write-baseline', str(target),
            '--authorize-raise', '5342',
            '--reason', 'net-additive bug fix',
            '--ledger', str(ledger),
        ]) == 0

        assert target.read_text(encoding='utf-8') == ratchet.render_baseline(raised)
        entries = ratchet.load_ledger(ledger)
        assert len(entries) == 1
        assert entries[0]['task_id'] == '5342'
        out = capsys.readouterr().out
        assert str(ledger) in out
        assert '5342' in out
        assert str(len(entries[0]['measures'])) in out

    def test_a_refusal_exits_one_not_two_and_leaves_the_bytes(
        self, stub_measurement: dict, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        # BOTH codes in one test, because the split is the thing: 1 is "a
        # measure rose" -- at write time here rather than at check time -- and 2
        # stays reserved for a broken instrument. Collapsing them would send a
        # reader to the wrong file.
        target, ledger, raised = self._seeded(stub_measurement, tmp_path)
        before = target.read_text(encoding='utf-8')
        monkeypatch.setattr(lane_cluster, 'build_report', lambda root: copy.deepcopy(raised))

        assert metrics.main(['--write-baseline', str(target)]) == 1

        err = capsys.readouterr().err
        assert WORKER_RELPATH in err
        assert ratchet.RAISE_REMEDY in err
        assert target.read_text(encoding='utf-8') == before
        assert not ledger.exists()

        def explode(root: Path) -> dict:
            raise source_measures.MetricsError('complexipy 7.0.1 is outside >=6.2,<7')

        monkeypatch.setattr(lane_cluster, 'build_report', explode)
        assert metrics.main(['--write-baseline', str(target)]) == 2

    @pytest.mark.parametrize(
        ('argv', 'rule'),
        [
            pytest.param(
                ['--write-baseline', 'x.json', '--authorize-raise', '5342'],
                'require each other',
                id='authorize-without-reason',
            ),
            pytest.param(
                ['--write-baseline', 'x.json', '--reason', 'because'],
                'require each other',
                id='reason-without-authorize',
            ),
            pytest.param(
                ['--check', '--authorize-raise', '5342', '--reason', 'because'],
                'modifier of --write-baseline',
                id='with-check',
            ),
            pytest.param(
                ['--report', '--authorize-raise', '5342', '--reason', 'because'],
                'modifier of --write-baseline',
                id='with-report',
            ),
            pytest.param(
                ['--json', '--authorize-raise', '5342', '--reason', 'because'],
                'modifier of --write-baseline',
                id='with-json',
            ),
            pytest.param(
                [
                    '--write-baseline', 'x.json',
                    '--authorize-raise', '5342', '--reason', '   ',
                ],
                'missing reason',
                id='blank-reason',
            ),
        ],
    )
    def test_authorization_is_a_modifier_of_the_write_and_nothing_else(
        self, argv: list[str], rule: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A flag that silently does nothing is worse than a rejected one: it
        # reads, to the agent who typed it, exactly like an authorization that
        # was granted.
        #
        # THE RULE THAT FIRED IS ASSERTED, not merely that argparse exited.
        # Deleting the two flags outright would make argparse exit on
        # "unrecognized arguments" for every case here, so a bare
        # pytest.raises(SystemExit) would stay green over a vanished feature.
        with pytest.raises(SystemExit) as excinfo:
            metrics.main(argv)

        assert excinfo.value.code == 2
        assert rule in capsys.readouterr().err

    def test_a_blank_reason_is_rejected_before_anything_is_measured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The value object refuses to exist unattributed, and the CLI resolves
        # the flags before it measures -- so a typo costs a usage error, not a
        # 71-second measurement whose result is thrown away.
        def explode(root: Path) -> dict:
            raise AssertionError('measured despite a blank --reason')

        monkeypatch.setattr(lane_cluster, 'build_report', explode)
        target = tmp_path / 'b.json'

        with pytest.raises(SystemExit) as excinfo:
            metrics.main([
                '--write-baseline', str(target),
                '--authorize-raise', '5342', '--reason', '   ',
            ])

        assert excinfo.value.code == 2
        assert not target.exists()

    def test_an_authorization_that_recorded_nothing_says_so(
        self, stub_measurement: dict, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        # SILENCE IS THE FAILURE. An authorization over a lowering report
        # records nothing by design, and printing only "wrote ..." would read,
        # to the agent who typed the flags, exactly like one that was granted
        # and recorded (INV-11).
        target = tmp_path / 'b.json'
        ledger = tmp_path / 'raises'
        ratchet.write_baseline(target, stub_measurement)
        lowered = copy.deepcopy(stub_measurement)
        lowered['files'][WORKER_RELPATH]['cognitive'] -= 10
        monkeypatch.setattr(lane_cluster, 'build_report', lambda root: copy.deepcopy(lowered))

        assert metrics.main([
            '--write-baseline', str(target),
            '--authorize-raise', '5342',
            '--reason', 'a lowering run that carried the flags',
            '--ledger', str(ledger),
        ]) == 0

        out = capsys.readouterr().out
        assert 'no measure rose' in out
        assert str(ledger) in out
        assert not ledger.exists()

    def test_ledger_defaults_to_the_committed_path(self) -> None:
        args = metrics._build_parser().parse_args(['--check'])
        assert Path(args.ledger).name == 'merge_lane_ratchet_raises'


class TestCliContract:
    @pytest.mark.parametrize(
        'argv',
        [
            ['--report', '--json'],
            ['--check', '--report'],
            ['--json', '--write-baseline', 'x.json'],
            ['--check', '--write-baseline', 'x.json'],
        ],
    )
    def test_the_four_modes_are_mutually_exclusive(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit):
            metrics.main(argv)

    def test_an_instrument_failure_exits_two_with_a_named_cause(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Exit 2 is deliberately DISTINCT from the exit-1 ratchet violation, so
        # a broken instrument is never mistaken for a clean tree or for a real
        # regression.
        def explode(root: Path) -> dict:
            raise source_measures.MetricsError('complexipy 7.0.1 is outside >=6.2,<7')

        monkeypatch.setattr(lane_cluster, 'build_report', explode)
        assert metrics.main(['--report']) == 2
        err = capsys.readouterr().err
        assert 'complexipy 7.0.1' in err

    def test_a_missing_baseline_exits_two_not_one(
        self, stub_measurement: dict, tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The failure INV-11 exists to prevent: a missing baseline must not read
        # as "no violations".
        missing = tmp_path / 'absent.json'
        assert metrics.main(['--check', '--baseline', str(missing)]) == 2
        assert 'absent.json' in capsys.readouterr().err

    def test_root_defaults_to_the_repo_root(self) -> None:
        args = metrics._build_parser().parse_args(['--report'])
        assert Path(args.root).resolve() == _REPO_ROOT.resolve()

    def test_baseline_defaults_to_the_committed_path(self) -> None:
        args = metrics._build_parser().parse_args(['--check'])
        assert Path(args.baseline).name == 'merge_lane_ratchet_baseline.json'


class TestTheBlockMessageNamesTheAuthorizedPath:
    """The task's also-fix, asserted as runtime behaviour, not as a prose pin.

    Every site that tells an agent a measure may not rise composes the ONE
    RAISE_REMEDY constant, so the mechanism can never be documented in two
    places out of three. The site that actually mattered was the committed
    baseline's own _README -- that is what a blocked agent opens -- and it said
    only that raising was forbidden, which is what sent task 5342's implementer
    to escalation instead of to a sanctioned path.
    """

    def test_the_remedy_names_the_flags_and_both_artifacts(self) -> None:
        for fragment in (
            '--authorize-raise',
            '--reason',
            ratchet.LEDGER_DIR_RELPATH,
            ratchet.BASELINE_RELPATH,
            # The MECHANISM NAME an agent greps for after being blocked. Until
            # task 5722 this paragraph told them "what no gate can see is a
            # baseline deleted first" -- which a gate now does see, so the
            # sentence a blocked reader most needs was the one that was false.
            ratchet.COMMIT_GATE_RELPATH,
            ratchet.ADVANCE_GATE_RELPATH,
        ):
            assert fragment in ratchet.RAISE_REMEDY, fragment

    def test_the_auditor_the_remedy_names_actually_resolves(self) -> None:
        # THE SUBSTRING PIN ABOVE CANNOT CATCH A RENAME. Every other fragment
        # there is a real constant, so it moves with what it names; this one
        # points at a FILE, is frozen verbatim into the committed baseline's
        # _README, and the instrument has no git dependency with which to
        # notice it going stale. Renaming or moving the auditor would leave a
        # blocked agent grepping for nothing, with the pin still green.
        assert (ratchet.repo_root() / ratchet.COMMIT_GATE_RELPATH).is_file()
        assert (ratchet.repo_root() / ratchet.ADVANCE_GATE_RELPATH).is_file()

    def test_the_committed_baseline_bytes_state_the_mechanism(self) -> None:
        # THE SITE THAT MATTERED. Not the docstring, not the CLI -- the file a
        # reader has open when the gate goes red.
        #
        # Read off the COMMITTED bytes, so this goes red until the baseline is
        # regenerated with the new README rather than passing on the constant
        # alone. The remedy carries newlines (the sanctioned commands belong on
        # their own lines at a terminal) and JSON escapes those, so the verbatim
        # match is against the decoded _README while the flag and the ledger
        # path -- which a reader greps for -- are matched in the raw text.
        committed = (_REPO_ROOT / ratchet.BASELINE_RELPATH).read_text(
            encoding='utf-8'
        )
        assert '--authorize-raise' in committed
        assert ratchet.LEDGER_DIR_RELPATH in committed
        assert json.loads(committed)['_README'].endswith(ratchet.RAISE_REMEDY)

    def test_the_readme_composes_the_remedy_rather_than_paraphrasing_it(
        self,
    ) -> None:
        assert ratchet.BASELINE_README.endswith(ratchet.RAISE_REMEDY)

    def test_the_check_trailer_is_the_same_constant(
        self, stub_measurement: dict, tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        doctored = copy.deepcopy(stub_measurement)
        doctored['files'][WORKER_RELPATH]['cognitive'] -= 10
        baseline = tmp_path / 'baseline.json'
        ratchet.write_baseline(baseline, doctored)

        assert metrics.main(['--check', '--baseline', str(baseline)]) == 1
        assert ratchet.RAISE_REMEDY in capsys.readouterr().err


# ---------------------------------------------------------------------------
# THE RATCHET ITSELF.
#
# Everything above pins the instrument. This is the gate: the live tree, the
# committed baseline, and the comparator the CLI's --check runs verbatim.

_BASELINE_PATH = _REPO_ROOT / ratchet.BASELINE_RELPATH


@pytest.fixture(scope='module')
def committed_baseline() -> dict:
    return ratchet.load_baseline(_BASELINE_PATH)


def test_merge_lane_ratchet_holds(
    live_report: dict, committed_baseline: dict
) -> None:
    violations = ratchet.check_against_baseline(live_report, committed_baseline)
    assert not violations, (
        'The merge-lane quality ratchet has been breached by '
        f'{len(violations)} measure(s):\n'
        + '\n'.join(f'  - {v.message}' for v in violations)
        + f'\n\n{ratchet.RAISE_REMEDY}'
    )


class TestBaselineIsNotVacuous:
    """Anti-vacuity floors, the shape every sibling guard here carries.

    Without them the ratchet passes just as happily against a baseline that
    measured NOTHING -- an empty files map compares clean against every path,
    and a green gate would certify a cluster nobody looked at.
    """

    def test_the_baseline_names_the_whole_cluster(
        self, committed_baseline: dict
    ) -> None:
        assert len(committed_baseline['files']) >= 20

    def test_the_baseline_cognitive_total_is_real(
        self, committed_baseline: dict
    ) -> None:
        # Measured 4,607 across 724 lane functions on the introducing commit.
        assert ratchet.derive_totals(committed_baseline)['cognitive'] >= 4000

    def test_the_baseline_line_total_is_real(self, committed_baseline: dict) -> None:
        # Measured 54,048 lines across the 22-path cluster.
        assert ratchet.derive_totals(committed_baseline)['lines'] >= 50000

    def test_the_baseline_represents_the_lane_importing_test_files(
        self, committed_baseline: dict
    ) -> None:
        # Measured 226 lane-importing files under orchestrator/tests.
        assert len(committed_baseline['tests']) >= 150

    def test_the_baseline_enumeration_was_complete_when_recorded(
        self, committed_baseline: dict
    ) -> None:
        assert committed_baseline['enumeration']['complete'] is True
        assert committed_baseline['enumeration']['unreadable'] == []

    def test_the_committed_enumeration_holds_no_test_tree_path_list(
        self, committed_baseline: dict
    ) -> None:
        # THE HEADLINE SIGNAL, asserted against the real committed bytes rather
        # than a synthetic report: no per-path test-tree manifest and no
        # test-tree number reach the file, so churn under orchestrator/tests
        # cannot move a single byte of it.
        enumeration = committed_baseline['enumeration']
        assert 'test_tree' not in enumeration
        assert _test_tree_strays(enumeration) == {}, (
            'the committed baseline froze test-tree path(s)'
        )

    def test_the_committed_enumeration_still_names_the_whole_cluster(
        self, committed_baseline: dict
    ) -> None:
        # ANTI-VACUITY for the test above, which an empty block would satisfy
        # just as happily. Measured 22 resolved paths from 23 CLUSTER_PATHS
        # entries (merge_lane/** expands to zero until PRD task zeta1).
        enumeration = committed_baseline['enumeration']
        assert enumeration['requested'] == list(lane_cluster.CLUSTER_PATHS)
        assert len(enumeration['resolved']) >= 15

    def test_the_live_sweep_denominator_covers_the_whole_test_tree(
        self, live_report: dict
    ) -> None:
        """The FLOOR that stands in for the manifest the baseline never stores.

        The committed file carries the CLUSTER half of the enumeration only, so
        nothing in it witnesses that the sweep walked the whole test tree
        (esc-5021-7, and see `_stored_enumeration` for why the counts are not
        stored either). This is the SOLE remaining guard against a collapsed
        sweep quietly disarming the two test-suite measures: if the sweep stopped
        walking, `tests` would shrink, every per-path measure in it would vanish,
        and the ratchet would go green on a cluster nobody looked at -- a fall
        being permitted is exactly what makes that invisible to the comparator.

        A floor on a COUNT, not an equality, is the whole point: it catches the
        collapse while staying green when an unrelated task adds one test file,
        which is the churn that made this gate a hair trigger. Measured 568 .py
        files under orchestrator/tests; the floor sits far below that so it
        survives ordinary attrition rather than tracking the tree -- the idiom
        this module uses for every live-tree anchor (commit d54acca456,
        "live-tree anchors become floors, not exact pins").
        """
        assert live_report['enumeration']['test_tree']['requested'] >= 400


def test_baseline_matches_a_fresh_measurement(live_report: dict) -> None:
    """The committed bytes must equal what measuring this tree produces now.

    Without this, a baseline could sit ABOVE the truth -- recording numbers
    nobody ever lowered it to match -- and every one of those slack measures
    would be headroom a later task could silently grow into while the ratchet
    stayed green.
    """
    assert _BASELINE_PATH.read_text(encoding='utf-8') == ratchet.render_baseline(
        live_report
    ), (
        'The committed baseline is not what measuring this tree produces. '
        'Regenerate it in this commit -- which the instrument itself will '
        'refuse to do if that would absorb a raise you have not authorized:\n'
        f'\n{ratchet.RAISE_REMEDY}'
    )
