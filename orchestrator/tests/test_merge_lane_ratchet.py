"""The merge-lane quality ratchet, and the unit tests of the instrument behind it.

PRD ``plans/merge-lane-quality-prd.md`` task alpha. The instrument itself lives
in ``scripts/merge_lane_metrics.py``; this module hosts both its detector unit
tests and the real tree-scanning gate, in the same single-file shape as the
guard it supersedes (``test_merge_queue_reachback_patch_guard.py`` carries its
``_find_merge_queue_private_patches`` fixtures and its tree sweep in one file).

Ratchet contract
----------------
``orchestrator/tests/merge_lane_ratchet_baseline.json`` freezes every measure at
its value on the commit that introduced it. The gate FAILS on any measure that
rises above its baseline (ratchet semantics): raising a measure is not allowed;
only lowering one is. Equality is permitted -- this is a ratchet, not a day-one
gate, and on the introducing commit 62 lane functions already exceed cognitive
15 and ``merge_queue.py`` is 21,550 lines. The two CEILINGS (1,500 lines per
file, cognitive 15 per function) therefore apply only to paths and qualnames
ABSENT from the baseline, so the grandfathering can only ever shrink.

A task that legitimately LOWERS a measure regenerates the baseline in the SAME
commit (``python scripts/merge_lane_metrics.py --write-baseline
orchestrator/tests/merge_lane_ratchet_baseline.json``). A task may never raise
one. Do not regenerate the baseline merely to make this test pass.

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

import copy
import dataclasses
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT

# This module AST-parses the Appendix A cluster (23 CLUSTER_PATHS entries, two
# of them globs) plus every *.py under orchestrator/tests/, and runs complexipy
# over the cluster. All of that is ONE measurement per session, taken by the
# module-scoped `live_measurement` fixture below.
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
# innocent test (esc-3980-1). See WHOLE_TREE_SCAN_TEST_TIMEOUT in
# _orch_helpers.py. That mark is PER ITEM and the slowest single item has always
# sat far below it, so it was never the thing at risk here.
#
# NOTE this mark is deliberate and NOT compelled by the family guard
# test_whole_tree_scan_timeout_guard.py: its ``_scans_whole_tree_py`` detector
# looks for a literal ``rglob('*.py')`` in the TEST file's own source, and this
# module's sweep lives in scripts/merge_lane_metrics.py instead. Adding a marked
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

# Make scripts/merge_lane_metrics.py importable. This is the sanctioned
# precedent for an orchestrator/tests/ module importing a scripts/ one -- see
# test_run_vllm_eval.py -- and needs no new pyright config: orchestrator/
# pyproject.toml's [tool.pyright] extraPaths already carries "../scripts".
_SCRIPTS = Path(__file__).parents[2] / 'scripts'
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import merge_lane_metrics as metrics  # type: ignore[import-not-found]  # noqa: E402

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
    `test_each_cluster_file_is_measured_by_complexipy_once` reads that list, so
    the per-file-once guarantee costs no measurement of its own. The builtin
    `monkeypatch` fixture is function-scoped and cannot be requested here,
    hence `pytest.MonkeyPatch.context()`.
    """
    calls: list[Path] = []
    original = metrics._file_complexity

    def recording(path: Path):
        calls.append(path)
        return original(path)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(metrics, '_file_complexity', recording)
        report = metrics.build_report(_REPO_ROOT)
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

    Parses are NOT scoped that way and cannot be: `metrics._parse` calls
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
    real_parse = metrics.ast.parse
    real_read_text = metrics.Path.read_text

    def counting_parse(source, filename='<unknown>', *args, **kwargs):
        parses.append(str(filename))
        return real_parse(source, filename, *args, **kwargs)

    def counting_read_text(self_path: Path, *args, **kwargs):
        if self_path.suffix == '.py' and swept in self_path.parents:
            reads.append(str(self_path))
        return real_read_text(self_path, *args, **kwargs)

    monkeypatch.setattr(metrics.ast, 'parse', counting_parse)
    monkeypatch.setattr(metrics.Path, 'read_text', counting_read_text)
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

# The 18 orchestrator/src literal paths of PRD Appendix A, verbatim. git_ops.py
# is listed separately below because Appendix A adds it under a different rule
# (measured, but exempt from the file-size ceiling -- PRD decision 9).
_APPENDIX_A_SRC = (
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
)
_GIT_OPS = 'orchestrator/src/orchestrator/git_ops.py'
_APPENDIX_A_TESTS = (
    'orchestrator/tests/_serial_merge_worker.py',
    'orchestrator/tests/_merge_queue_harness.py',
    'orchestrator/tests/conftest.py',
)
_MERGE_LANE_GLOB = 'orchestrator/src/orchestrator/merge_lane/**/*.py'


# ---------------------------------------------------------------------------
# CLUSTER_PATHS -- the SPOT source of Appendix A.


class TestClusterPathSpec:
    def test_cluster_paths_covers_every_appendix_a_src_module(self) -> None:
        assert set(_APPENDIX_A_SRC) <= set(metrics.CLUSTER_PATHS)

    def test_cluster_paths_includes_git_ops(self) -> None:
        # Measured but not gated by the file-size ceiling (PRD decision 9).
        assert _GIT_OPS in metrics.CLUSTER_PATHS

    def test_cluster_paths_includes_the_three_test_paths(self) -> None:
        assert set(_APPENDIX_A_TESTS) <= set(metrics.CLUSTER_PATHS)

    def test_cluster_paths_carries_one_merge_lane_glob(self) -> None:
        globs = [p for p in metrics.CLUSTER_PATHS if '*' in p]
        assert globs == [_MERGE_LANE_GLOB]

    def test_cluster_paths_has_exactly_the_appendix_a_entries(self) -> None:
        expected = (
            set(_APPENDIX_A_SRC)
            | {_GIT_OPS}
            | set(_APPENDIX_A_TESTS)
            | {_MERGE_LANE_GLOB}
        )
        assert set(metrics.CLUSTER_PATHS) == expected

    def test_cluster_paths_is_an_ordered_tuple(self) -> None:
        # Order is part of the spec: `params.cluster_paths` is compared against
        # it in check_against_baseline, and a stable order keeps that diff
        # readable when Appendix A is deliberately edited.
        assert isinstance(metrics.CLUSTER_PATHS, tuple)

    def test_size_ceiling_exempt_is_exactly_git_ops(self) -> None:
        # PRD decision 9: git_ops.py is the git engine for warm lanes, the
        # scheduler and recovery -- not only the merge lane -- and its split is
        # a follow-up PRD. The exemption is scoped to the CEILING; git_ops.py is
        # still ratcheted (pinned by TestSizeCeilingExemptionIsScoped).
        assert set(metrics.SIZE_CEILING_EXEMPT) == {_GIT_OPS}

    def test_ceiling_constants(self) -> None:
        assert metrics.FILE_LINE_CEILING == 1500
        assert metrics.NEW_FUNCTION_COGNITIVE_CEILING == 15


# ---------------------------------------------------------------------------
# resolve_cluster_paths -- INV-11: a partial enumeration is never returned
# silently, and completeness is legible in the RESULT.


class TestResolveClusterPaths:
    def test_real_repo_enumeration_is_complete(self) -> None:
        enumeration = metrics.resolve_cluster_paths(_REPO_ROOT)
        assert enumeration.complete is True
        assert enumeration.unreadable == ()

    def test_real_repo_enumeration_resolves_every_literal_path(self) -> None:
        enumeration = metrics.resolve_cluster_paths(_REPO_ROOT)
        literals = {p for p in metrics.CLUSTER_PATHS if '*' not in p}
        assert literals <= set(enumeration.resolved)

    def test_resolved_paths_are_repo_relative_forward_slash(self) -> None:
        enumeration = metrics.resolve_cluster_paths(_REPO_ROOT)
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
        monkeypatch.setattr(metrics, 'CLUSTER_PATHS', ('a.py', 'nowhere/**/*.py'))
        enumeration = metrics.resolve_cluster_paths(tmp_path)
        assert enumeration.complete is True
        assert enumeration.resolved == ('a.py',)

    def test_missing_literal_path_raises_naming_the_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(metrics, 'CLUSTER_PATHS', ('gone/missing.py',))
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.resolve_cluster_paths(tmp_path)
        assert 'gone/missing.py' in str(excinfo.value)

    def test_missing_literal_path_never_returns_a_partial_enumeration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The INV-11 seam: a present sibling must not let the missing path be
        # silently dropped from an otherwise plausible-looking result.
        (tmp_path / 'a.py').write_text('x = 1\n', encoding='utf-8')
        monkeypatch.setattr(metrics, 'CLUSTER_PATHS', ('a.py', 'gone/missing.py'))
        with pytest.raises(metrics.MetricsError):
            metrics.resolve_cluster_paths(tmp_path)

    def test_literal_path_that_is_a_directory_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / 'adir.py').mkdir()
        monkeypatch.setattr(metrics, 'CLUSTER_PATHS', ('adir.py',))
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.resolve_cluster_paths(tmp_path)
        assert 'adir.py' in str(excinfo.value)


class TestEnumerationRoundTrip:
    def test_to_dict_carries_the_four_completeness_keys(self) -> None:
        enumeration = metrics.Enumeration(
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

        enumeration = metrics.Enumeration(
            requested=('a.py',), resolved=('a.py',), unreadable=('c.py',), complete=False
        )
        assert json.loads(json.dumps(enumeration.to_dict()))['complete'] is False

    def test_enumeration_is_frozen(self) -> None:
        import dataclasses

        enumeration = metrics.Enumeration(
            requested=(), resolved=(), unreadable=(), complete=True
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            enumeration.complete = False  # type: ignore[misc]

    def test_unreadable_entries_make_the_enumeration_incomplete(self) -> None:
        # `complete` is the single legible signal a caller checks; it must
        # never read True while files were skipped.
        enumeration = metrics.Enumeration(
            requested=('a.py',), resolved=(), unreadable=('a.py',), complete=False
        )
        assert enumeration.complete is False
        unreadable = enumeration.to_dict()['unreadable']
        assert isinstance(unreadable, list)
        assert 'a.py' in unreadable


# ---------------------------------------------------------------------------
# file_size_measures -- lines and prose lines.
#
# Pinned against INLINE synthetic snippets, never against a real repo file, so
# these tests cannot drift when the cluster changes. The one real-tree
# assertion below is an explicit anti-vacuity anchor, not a definition.


class TestFileSizeMeasures:
    def test_lines_counts_physical_lines(self) -> None:
        source = 'a = 1\nb = 2\nc = 3\n'
        assert metrics.file_size_measures(source, path='t.py').lines == 3

    def test_lines_counts_a_final_line_without_a_trailing_newline(self) -> None:
        assert metrics.file_size_measures('a = 1\nb = 2', path='t.py').lines == 2

    def test_blank_lines_are_lines_but_not_prose(self) -> None:
        measures = metrics.file_size_measures('a = 1\n\n\nb = 2\n', path='t.py')
        assert measures.lines == 4
        assert measures.prose_lines == 0

    def test_module_docstring_counts_as_prose(self) -> None:
        measures = metrics.file_size_measures('"""Doc."""\na = 1\n', path='t.py')
        assert measures.prose_lines == 1

    def test_function_and_class_docstrings_count_as_prose(self) -> None:
        source = (
            'class C:\n'
            '    """Class doc."""\n'
            '\n'
            '    def m(self):\n'
            '        """Method doc."""\n'
            '        return 1\n'
        )
        assert metrics.file_size_measures(source, path='t.py').prose_lines == 2

    def test_async_function_docstring_counts_as_prose(self) -> None:
        source = 'async def f():\n    """Doc."""\n    return 1\n'
        assert metrics.file_size_measures(source, path='t.py').prose_lines == 1

    def test_multiline_docstring_counts_once_per_line_it_spans(self) -> None:
        source = '"""Line one.\n\nLine three.\n"""\na = 1\n'
        measures = metrics.file_size_measures(source, path='t.py')
        assert measures.lines == 5
        assert measures.prose_lines == 4

    def test_standalone_comment_lines_count_as_prose(self) -> None:
        source = '# one\n# two\na = 1\n'
        assert metrics.file_size_measures(source, path='t.py').prose_lines == 2

    def test_trailing_inline_comment_makes_its_code_line_prose(self) -> None:
        source = 'a = 1  # why\nb = 2\n'
        assert metrics.file_size_measures(source, path='t.py').prose_lines == 1

    def test_a_line_that_is_both_docstring_and_comment_counts_once(self) -> None:
        # The two line-number sets are unioned, not summed.
        source = '"""Doc."""  # trailing\na = 1\n'
        assert metrics.file_size_measures(source, path='t.py').prose_lines == 1

    def test_a_string_literal_mentioning_hash_is_not_a_comment(self) -> None:
        # THE tokenize-vs-regex discriminator: no regex over source text gets
        # this right, and getting it wrong would inflate prose_lines on any file
        # that formats a '#'-bearing string.
        source = "url = 'http://x/#frag'\nheading = '# not a comment'\n"
        assert metrics.file_size_measures(source, path='t.py').prose_lines == 0

    def test_a_non_docstring_string_expression_is_not_prose(self) -> None:
        # Only the FIRST body element of a module/class/function is a docstring.
        source = '"""Doc."""\n"""Not a docstring."""\na = 1\n'
        assert metrics.file_size_measures(source, path='t.py').prose_lines == 1

    def test_unparseable_source_raises_naming_the_path(self) -> None:
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.file_size_measures('def (:\n', path='broken.py')
        message = str(excinfo.value)
        assert 'broken.py' in message
        assert 'SyntaxError' in message

    def test_unparseable_source_never_returns_a_zero_measure(self) -> None:
        # INV-11: an unmeasurable cluster file is the finding, not a 0 that
        # silently satisfies every ratchet comparison.
        with pytest.raises(metrics.MetricsError):
            metrics.file_size_measures('class ???:\n', path='broken.py')

    def test_measures_are_frozen(self) -> None:
        import dataclasses

        measures = metrics.file_size_measures('a = 1\n', path='t.py')
        with pytest.raises(dataclasses.FrozenInstanceError):
            measures.lines = 99  # type: ignore[misc]

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
        source = (_REPO_ROOT / 'orchestrator/src/orchestrator/merge_queue.py').read_text(
            encoding='utf-8'
        )
        measures = metrics.file_size_measures(source, path='merge_queue.py')
        assert measures.lines >= 10000
        assert measures.prose_lines > 0


# ---------------------------------------------------------------------------
# Structural import measures: function-local (reach-back) imports, and
# re-export shim names.


class TestFunctionLocalImports:
    def test_import_from_inside_a_def_is_counted(self) -> None:
        source = 'def f():\n    from x import y\n    return y\n'
        assert metrics.function_local_imports(source, path='t.py') == 1

    def test_plain_import_inside_a_def_is_counted(self) -> None:
        source = 'def f():\n    import x\n    return x\n'
        assert metrics.function_local_imports(source, path='t.py') == 1

    def test_import_inside_an_async_def_is_counted(self) -> None:
        source = 'async def f():\n    from x import y\n    return y\n'
        assert metrics.function_local_imports(source, path='t.py') == 1

    def test_import_inside_a_nested_def_is_counted_once(self) -> None:
        # Walking from each function node would visit the inner import twice
        # (once from the outer function, once from the inner); dedupe by node
        # identity keeps this at 1.
        source = (
            'def outer():\n'
            '    def inner():\n'
            '        from x import y\n'
            '        return y\n'
            '    return inner\n'
        )
        assert metrics.function_local_imports(source, path='t.py') == 1

    def test_module_level_import_is_not_counted(self) -> None:
        source = 'from x import y\nimport z\n\n\ndef f():\n    return y\n'
        assert metrics.function_local_imports(source, path='t.py') == 0

    def test_import_in_a_class_body_outside_any_function_is_not_counted(self) -> None:
        source = 'class C:\n    from x import y\n'
        assert metrics.function_local_imports(source, path='t.py') == 0

    def test_import_in_a_method_body_is_counted(self) -> None:
        source = 'class C:\n    def m(self):\n        from x import y\n        return y\n'
        assert metrics.function_local_imports(source, path='t.py') == 1

    def test_a_docstring_quoting_an_import_is_not_counted(self) -> None:
        # AST, not regex. The satellite modules' reach-back notes quote exactly
        # this string in prose; counting them would inflate the measure.
        source = (
            'def f():\n'
            '    """Resolves via ``from orchestrator.merge_queue import X``."""\n'
            '    return 1\n'
        )
        assert metrics.function_local_imports(source, path='t.py') == 0

    def test_each_import_statement_counts_once_regardless_of_names(self) -> None:
        source = 'def f():\n    from x import a, b, c\n    return a\n'
        assert metrics.function_local_imports(source, path='t.py') == 1

    def test_unparseable_source_raises_naming_the_path(self) -> None:
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.function_local_imports('def (:\n', path='broken.py')
        assert 'broken.py' in str(excinfo.value)

    def test_cluster_total_is_positive_anchor(self) -> None:
        # Anti-vacuity: the PRD's ceilings row counts the function-local imports
        # inside the package; on this tree the count is well above zero.
        total = 0
        for relpath in metrics.resolve_cluster_paths(_REPO_ROOT).resolved:
            source = (_REPO_ROOT / relpath).read_text(encoding='utf-8')
            total += metrics.function_local_imports(source, path=relpath)
        assert total > 0


class TestReexportNames:
    def test_unreferenced_module_level_import_from_names_are_reexports(self) -> None:
        source = 'from a import (B, C)\n'
        assert metrics.reexport_names(source, path='t.py') == ['B', 'C']

    def test_a_referenced_binding_is_not_a_reexport(self) -> None:
        source = 'from a import B\n\nx = B()\n'
        assert metrics.reexport_names(source, path='t.py') == []

    def test_an_alias_is_reported_under_the_bound_name(self) -> None:
        source = 'from a import B as C\n'
        assert metrics.reexport_names(source, path='t.py') == ['C']

    def test_a_used_alias_is_not_a_reexport(self) -> None:
        source = 'from a import B as C\n\nx = C()\n'
        assert metrics.reexport_names(source, path='t.py') == []

    def test_a_name_referenced_only_in_a_docstring_still_counts(self) -> None:
        # AST, not regex: prose mentioning the name does not make it used.
        source = '"""Re-exports B for consumers."""\nfrom a import B\n# B lives here\n'
        assert metrics.reexport_names(source, path='t.py') == ['B']

    def test_a_name_referenced_inside_a_nested_function_counts_as_used(self) -> None:
        source = (
            'from a import B\n'
            '\n'
            '\n'
            'def outer():\n'
            '    def inner():\n'
            '        return B\n'
            '    return inner\n'
        )
        assert metrics.reexport_names(source, path='t.py') == []

    def test_a_function_local_import_from_is_not_a_reexport(self) -> None:
        # Only MODULE-LEVEL ImportFrom bindings form the module's public
        # surface; a deferred import inside a function is the reach-back
        # measure's business, not this one's.
        source = 'def f():\n    from a import B\n    return 1\n'
        assert metrics.reexport_names(source, path='t.py') == []

    def test_a_plain_import_is_not_counted(self) -> None:
        # The PRD's measure is the `from X import (...)` shim block; a bare
        # `import x` binds a module, not a re-exported name.
        source = 'import a\n'
        assert metrics.reexport_names(source, path='t.py') == []

    def test_a_future_import_is_not_a_reexport(self) -> None:
        # `from __future__ import annotations` is a compiler directive, not a
        # name a downstream module could import, and ruff's F401 -- the
        # predicate this measure claims to agree with -- explicitly never flags
        # it. Counting it inflated 19 of 22 baseline paths by one and made the
        # ratchet REWARD deleting a future import, which silently changes
        # runtime annotation semantics (esc-5021-6).
        source = 'from __future__ import annotations\nimport os\nx = 1\n'
        assert metrics.reexport_names(source, path='t.py') == []

    def test_a_future_import_does_not_mask_a_real_shim(self) -> None:
        # Excluding __future__ must not swallow the genuine shims beside it.
        source = 'from __future__ import annotations\nfrom a import B\n'
        assert metrics.reexport_names(source, path='t.py') == ['B']

    def test_star_import_is_not_reported_as_a_name(self) -> None:
        source = 'from a import *\n'
        assert metrics.reexport_names(source, path='t.py') == []

    def test_names_are_returned_sorted_and_deduped(self) -> None:
        source = 'from a import Z\nfrom b import A\n'
        assert metrics.reexport_names(source, path='t.py') == ['A', 'Z']

    def test_unparseable_source_raises_naming_the_path(self) -> None:
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.reexport_names('def (:\n', path='broken.py')
        assert 'broken.py' in str(excinfo.value)

    def test_merge_queue_structural_reading_overlaps_the_annotated_shims(self) -> None:
        # Anti-vacuity anchor AND the evidence for the STRUCTURAL-not-comment
        # decision: the structural predicate (a module-level ImportFrom binding
        # never referenced elsewhere -- exactly what ruff's F401 computes, which
        # is why those blocks carry the suppression) must land squarely on the
        # nine annotated `noqa: F401  re-export shim` blocks at lines
        # 57/64/89/112/118/150/154/188/198.
        #
        # OVERLAP, not containment, and the asymmetry is the interesting part:
        # a `noqa: F401` suppresses a whole BLOCK, so a name that merge_queue.py
        # both re-exports AND uses internally sits inside an annotated block
        # while being perfectly F401-clean. MEASURED on this tree: 127 names
        # across the nine blocks, 63 of them structurally unused. So the
        # structural set is a proper subset per block (8 of the 9 blocks
        # contribute at least one; merge_speculation_controller's two names are
        # both used internally), and a proper SUPERSET cluster-wide, because it
        # also catches pure re-exports nobody annotated.
        import ast as _ast

        path = _REPO_ROOT / 'orchestrator/src/orchestrator/merge_queue.py'
        source = path.read_text(encoding='utf-8')
        reported = set(metrics.reexport_names(source, path=str(path)))
        assert reported

        # The annotated blocks are DERIVED, never line-pinned. This test used to
        # carry `annotated_linenos = {57, 64, ...}` -- nine hard line numbers
        # into merge_queue.py, which is both the hottest file in the repo and
        # the explicit refactor target of this very PRD (gamma1..gamma10,
        # zeta1/zeta2). Inserting one line above 198 shifted every pin and
        # reddened the whole orchestrator suite on a change that had nothing to
        # do with shims -- a false-failure generator inside a gate, and against
        # CLAUDE.md's `path::symbol` citation convention.
        #
        # `_comment_lines` (stdlib tokenize) decides what is REALLY a comment,
        # so a string literal merely mentioning the marker cannot qualify; the
        # marker text is then matched on those lines only. Matching anywhere in
        # the node's line SPAN, not just `node.lineno`, keeps a block annotated
        # on a continuation line in the set.
        #
        # Verified equivalent when it replaced the pins: this derivation
        # reproduced exactly {57, 64, 89, 112, 118, 150, 154, 188, 198} -- same
        # nine blocks, 127 names, 63 structurally unused, 8 blocks hit.
        source_lines = source.splitlines()
        marked = {
            lineno
            for lineno in metrics._comment_lines(source, path=str(path))
            if 'noqa: F401' in source_lines[lineno - 1]
        }
        assert marked, 'no `noqa: F401` annotated blocks found in merge_queue.py'
        tree = _ast.parse(source)
        blocks = [
            {a.asname or a.name for a in node.names}
            for node in tree.body
            if isinstance(node, _ast.ImportFrom)
            and marked & set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
        ]
        # FLOORS, not exact counts -- the PRD exists to lower these, and this
        # module's live-tree anchors are one-sided by convention (d54acca456).
        assert len(blocks) >= 6, f'only {len(blocks)} annotated shim blocks found'
        hit_blocks = [names for names in blocks if names & reported]
        assert len(hit_blocks) >= 6, f'{len(hit_blocks)}/{len(blocks)} annotated blocks hit'
        annotated_names = set().union(*blocks)
        assert len(annotated_names & reported) >= 50

    def test_deleting_the_noqa_comment_does_not_change_the_measure(self) -> None:
        # THE ungameable property, pinned directly. A comment-scanning detector
        # would zero out on this purely cosmetic edit; the structural one cannot
        # see comments at all.
        source = 'from a import (  # noqa: F401  re-export shim\n    B,\n    C,\n)\n'
        stripped = source.replace('  # noqa: F401  re-export shim', '')
        assert metrics.reexport_names(source, path='t.py') == ['B', 'C']
        assert metrics.reexport_names(stripped, path='t.py') == ['B', 'C']


# ---------------------------------------------------------------------------
# The complexipy adapter, and its version contract (INV-11 for tools).

_TINY_SOURCE = (
    'def f(a):\n'
    '    if a:\n'
    '        for i in range(3):\n'
    '            if i:\n'
    '                return i\n'
    '    return 0\n'
    '\n'
    '\n'
    'class C:\n'
    '    def m(self):\n'
    '        return 1\n'
)


class TestCognitiveComplexity:
    def test_keyed_by_complexipy_qualname(self, tmp_path: Path) -> None:
        target = tmp_path / 'tiny.py'
        target.write_text(_TINY_SOURCE, encoding='utf-8')
        scores = metrics.file_cognitive_measures(target).per_function
        assert scores == {'f': 6, 'C::m': 0}

    def test_module_with_no_functions_returns_an_empty_map(self, tmp_path: Path) -> None:
        target = tmp_path / 'empty.py'
        target.write_text('X = 1\n', encoding='utf-8')
        assert metrics.file_cognitive_measures(target).per_function == {}

    def test_file_total_is_reported_separately(self, tmp_path: Path) -> None:
        target = tmp_path / 'tiny.py'
        target.write_text(_TINY_SOURCE, encoding='utf-8')
        assert metrics.file_cognitive_measures(target).total == 6

    def test_both_projections_come_from_one_complexipy_measurement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The file total and the per-function map are two VIEWS of one
        # complexipy result, not two measurements. build_report asked each
        # cluster file for both and so paid complexipy 50 times over the 25
        # resolved files; this counter forbids that shape from growing back.
        #
        # The counter RECORDS AND DELEGATES rather than stubbing, so the
        # equalities below are still checked against a real measurement.
        target = tmp_path / 'tiny.py'
        target.write_text(_TINY_SOURCE, encoding='utf-8')
        raw = metrics._file_complexity(target)
        expected_total = int(raw.complexity)
        expected_per_function = {f.name: f.complexity for f in raw.functions}
        # Anti-vacuity: a module measuring 0 with no functions would satisfy
        # the equalities below while witnessing nothing.
        assert expected_total > 0
        assert expected_per_function

        calls: list[Path] = []
        original = metrics._file_complexity

        def counting(path: Path):
            calls.append(path)
            return original(path)

        monkeypatch.setattr(metrics, '_file_complexity', counting)
        measures = metrics.file_cognitive_measures(target)

        assert calls == [target], calls
        assert measures.total == expected_total
        assert measures.per_function == expected_per_function

    def test_missing_complexipy_raises_naming_the_tool(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Never a skipped measure and never a 0: a tool the instrument cannot
        # run is an instrument failure with a named cause (INV-11).
        monkeypatch.setitem(sys.modules, 'complexipy', None)
        target = tmp_path / 'tiny.py'
        target.write_text(_TINY_SOURCE, encoding='utf-8')
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.file_cognitive_measures(target)
        message = str(excinfo.value)
        assert 'complexipy' in message
        assert 'dev' in message

    def test_unparseable_file_raises_naming_the_path(self, tmp_path: Path) -> None:
        # INV-11's polarity: a file the instrument cannot measure is the
        # FINDING, never a silent 0 and never a skip.
        target = tmp_path / 'broken.py'
        target.write_text('def (:\n', encoding='utf-8')
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.file_cognitive_measures(target)
        assert 'broken.py' in str(excinfo.value)

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
        target = _REPO_ROOT / 'orchestrator/src/orchestrator/merge_queue.py'
        measures = metrics.file_cognitive_measures(target)
        assert measures.per_function['SpeculativeMergeWorker::_verifier_loop'] >= 100
        assert measures.per_function['SpeculativeMergeWorker::stop'] >= 50
        assert measures.total >= 1000


class TestComplexipyVersionContract:
    def test_required_specifier_is_the_measured_range(self) -> None:
        assert metrics.COMPLEXIPY_REQUIRED == '>=6.2,<7'

    def test_installed_version_satisfies_the_requirement(self) -> None:
        # The executable form of the measured pin.
        metrics.require_complexipy()
        assert metrics.satisfies_complexipy_requirement(metrics.complexipy_version())

    def test_pyproject_pin_matches_the_scripts_requirement(self) -> None:
        # Goes RED the moment someone relaxes the dependency to 7.x. The pin and
        # the runtime check are two halves of one contract; letting them drift
        # would leave the ratchet silently measuring with the wrong engine.
        pyproject = (_REPO_ROOT / 'orchestrator/pyproject.toml').read_text(encoding='utf-8')
        assert f'"complexipy{metrics.COMPLEXIPY_REQUIRED}"' in pyproject

    def test_version_out_of_range_raises_naming_both_versions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(metrics, 'complexipy_version', lambda: '7.0.1')
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.require_complexipy()
        message = str(excinfo.value)
        assert '7.0.1' in message
        assert metrics.COMPLEXIPY_REQUIRED in message

    def test_version_below_the_floor_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(metrics, 'complexipy_version', lambda: '6.1.0')
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.require_complexipy()
        assert '6.1.0' in str(excinfo.value)

    @pytest.mark.parametrize(
        ('version', 'ok'),
        [
            ('6.2.0', True),
            ('6.2', True),
            ('6.9.9', True),
            ('6.1.9', False),
            ('6.0.0', False),
            ('5.0.0', False),
            ('7.0.0', False),
            ('7.0.1', False),
            ('8.0.0', False),
        ],
    )
    def test_specifier_boundaries(self, version: str, ok: bool) -> None:
        assert metrics.satisfies_complexipy_requirement(version) is ok

    def test_missing_complexipy_distribution_raises_naming_the_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib.metadata

        def _boom(name: str) -> str:
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, 'version', _boom)
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.complexipy_version()
        assert 'complexipy' in str(excinfo.value)


class TestMaintainabilityIndex:
    def test_returns_a_float_for_a_tiny_module(self) -> None:
        value = metrics.maintainability_index(_TINY_SOURCE, path='tiny.py')
        assert isinstance(value, float)
        assert 0.0 <= value <= 100.0

    def test_merge_queue_anchor_is_zero(self) -> None:
        # The PRD Background table's "Maintainability index (radon) | 0" row.
        # Reported, never ratcheted -- but genuinely exercised, so the `radon`
        # dev-group entry is not dead weight.
        source = (_REPO_ROOT / 'orchestrator/src/orchestrator/merge_queue.py').read_text(
            encoding='utf-8'
        )
        assert metrics.maintainability_index(source, path='merge_queue.py') == 0.0

    def test_missing_radon_raises_naming_the_tool(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, 'radon.metrics', None)
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.maintainability_index(_TINY_SOURCE, path='tiny.py')
        assert 'radon' in str(excinfo.value)

    def test_unparseable_source_raises_naming_the_path(self) -> None:
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.maintainability_index('def (:\n', path='broken.py')
        assert 'broken.py' in str(excinfo.value)


# ---------------------------------------------------------------------------
# Test-suite patch targets into lane internals.
#
# The two detector shapes are ported from
# test_merge_queue_reachback_patch_guard.py -- this measure subsumes that
# guard's allowlist as a COUNT, so PRD task delta can delete the guard once the
# count reaches zero. Two deliberate generalisations there: the prefix set is
# `orchestrator.merge_queue.` AND `orchestrator.merge_lane.`, and the guard's
# `forbidden` filter is dropped so all leaves count, not only the private ones.


class TestPatchTargets:
    def test_string_path_patch_is_detected(self) -> None:
        source = "patch('orchestrator.merge_queue.run_scoped_verification', x)\n"
        assert metrics.patch_targets(source) == {'run_scoped_verification'}

    def test_dotted_patch_is_detected(self) -> None:
        source = "mock.patch('orchestrator.merge_queue.advance_main', x)\n"
        assert metrics.patch_targets(source) == {'advance_main'}

    def test_string_path_setattr_is_detected(self) -> None:
        source = "monkeypatch.setattr('orchestrator.merge_queue.foo', x)\n"
        assert metrics.patch_targets(source) == {'foo'}

    def test_merge_lane_prefix_is_measured_too(self) -> None:
        # Measured before the package exists, deliberately: without it, PRD task
        # zeta2's `git mv` would move every patch target out from under the
        # measure and the count would read 0 by RELOCATION rather than by the
        # migration gamma1..gamma10 actually performs.
        source = "monkeypatch.setattr('orchestrator.merge_lane.foo', x)\n"
        assert metrics.patch_targets(source) == {'foo'}

    def test_merge_lane_submodule_path_is_measured(self) -> None:
        source = "patch('orchestrator.merge_lane.worker.spin', x)\n"
        assert metrics.patch_targets(source) == {'worker.spin'}

    def test_object_path_setattr_via_from_import_alias(self) -> None:
        source = (
            'from orchestrator import merge_queue\n'
            '\n'
            '\n'
            'def t(monkeypatch):\n'
            "    monkeypatch.setattr(merge_queue, 'x', 1)\n"
        )
        assert metrics.patch_targets(source) == {'x'}

    def test_object_path_setattr_via_import_as_alias(self) -> None:
        source = (
            'import orchestrator.merge_queue as mq\n'
            '\n'
            '\n'
            'def t(monkeypatch):\n'
            "    monkeypatch.setattr(mq, 'y', 1)\n"
        )
        assert metrics.patch_targets(source) == {'y'}

    def test_patch_object_on_the_bare_attribute_chain(self) -> None:
        source = "patch.object(orchestrator.merge_queue, 'x', 1)\n"
        assert metrics.patch_targets(source) == {'x'}

    def test_patch_object_on_a_merge_lane_alias(self) -> None:
        source = (
            'from orchestrator import merge_lane\n'
            '\n'
            '\n'
            "p = patch.object(merge_lane, 'z', 1)\n"
        )
        assert metrics.patch_targets(source) == {'z'}

    def test_patch_object_on_a_satellite_is_not_counted(self) -> None:
        # Already repointed to the defining satellite -- that is the END STATE
        # this measure is driving towards, so counting it would penalise the fix.
        source = (
            'from orchestrator import merge_gates\n'
            '\n'
            '\n'
            "p = patch.object(merge_gates, 'x', 1)\n"
        )
        assert metrics.patch_targets(source) == set()

    def test_an_unrelated_attribute_named_merge_queue_is_not_counted(self) -> None:
        source = "patch.object(workflow.merge_queue, 'x', 1)\n"
        assert metrics.patch_targets(source) == set()

    def test_a_docstring_quoting_the_dotted_path_is_not_counted(self) -> None:
        source = '"""Patches orchestrator.merge_queue.foo at the lookup site."""\n'
        assert metrics.patch_targets(source) == set()

    def test_a_comment_quoting_the_dotted_path_is_not_counted(self) -> None:
        source = "# patch('orchestrator.merge_queue.foo')\nx = 1\n"
        assert metrics.patch_targets(source) == set()

    def test_distinct_names_not_call_sites(self) -> None:
        source = (
            "patch('orchestrator.merge_queue.foo', a)\n"
            "patch('orchestrator.merge_queue.foo', b)\n"
            "patch('orchestrator.merge_queue.bar', c)\n"
        )
        assert metrics.patch_targets(source) == {'foo', 'bar'}

    def test_a_bare_module_patch_with_no_leaf_is_not_counted(self) -> None:
        source = "patch('orchestrator.merge_queue', x)\n"
        assert metrics.patch_targets(source) == set()

    def test_a_non_string_second_arg_is_not_counted(self) -> None:
        source = (
            'from orchestrator import merge_queue\n'
            '\n'
            '\n'
            'p = patch.object(merge_queue, NAME, 1)\n'
        )
        assert metrics.patch_targets(source) == set()

    def test_unparseable_source_raises_naming_the_path(self) -> None:
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.patch_targets('def (:\n', path='broken.py')
        assert 'broken.py' in str(excinfo.value)

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
        assert 'run_scoped_verification' in union


# ---------------------------------------------------------------------------
# Private-attribute reads from tests.


class TestImportsLaneModule:
    def test_import_from_a_cluster_module_is_lane_importing(self) -> None:
        source = 'from orchestrator.merge_queue import SpeculativeMergeWorker\n'
        assert metrics.imports_lane_module(source, path='t.py') is True

    def test_plain_import_of_merge_lane_is_lane_importing(self) -> None:
        assert metrics.imports_lane_module('import orchestrator.merge_lane\n', path='t.py') is True

    def test_import_from_a_merge_lane_submodule_is_lane_importing(self) -> None:
        source = 'from orchestrator.merge_lane.worker import W\n'
        assert metrics.imports_lane_module(source, path='t.py') is True

    def test_from_orchestrator_import_satellite_is_lane_importing(self) -> None:
        assert metrics.imports_lane_module('from orchestrator import merge_gates\n', path='t.py') is True

    def test_an_unrelated_orchestrator_module_is_not_lane_importing(self) -> None:
        assert metrics.imports_lane_module('import orchestrator.workflow\n', path='t.py') is False

    def test_a_module_with_no_imports_is_not_lane_importing(self) -> None:
        assert metrics.imports_lane_module('x = 1\n', path='t.py') is False

    def test_the_module_name_set_is_derived_from_cluster_paths(self) -> None:
        # Derived, never hand-listed, so the two can't drift when Appendix A is
        # edited.
        assert 'orchestrator.merge_queue' in metrics.lane_module_names()
        assert 'orchestrator.git_ops' in metrics.lane_module_names()
        assert 'orchestrator.workflow' not in metrics.lane_module_names()


class TestPrivateReads:
    def test_a_single_private_attribute_read_counts_one(self) -> None:
        assert metrics.private_reads('worker._inflight\n', path='t.py') == 1

    def test_a_chained_private_read_counts_each_hop(self) -> None:
        # Both hops reach into internals.
        assert metrics.private_reads('worker._a._b\n', path='t.py') == 2

    def test_a_dunder_is_not_a_private_read(self) -> None:
        # Dunders are Python protocol, not lane internals.
        assert metrics.private_reads('obj.__dict__\nobj.__class__\n', path='t.py') == 0

    def test_self_and_cls_receivers_are_excluded(self) -> None:
        # A test class's own helpers are not lane internals. This is a
        # STRUCTURAL predicate, not a name list.
        source = (
            'class T:\n'
            '    def t(self):\n'
            '        self._helper()\n'
            '        return cls._x\n'
        )
        assert metrics.private_reads(source, path='t.py') == 0

    def test_a_public_attribute_is_not_counted(self) -> None:
        assert metrics.private_reads('worker.snapshot()\n', path='t.py') == 0

    def test_a_docstring_quoting_a_private_read_is_not_counted(self) -> None:
        assert metrics.private_reads('"""Reads worker._inflight."""\n', path='t.py') == 0

    def test_a_private_read_off_self_dot_something_is_counted(self) -> None:
        # `self.worker._x` reaches into a lane object even though the chain
        # starts at self -- only the BARE `self`/`cls` receiver is excluded.
        assert metrics.private_reads('self.worker._x\n', path='t.py') == 1

    def test_a_private_write_counts_too(self) -> None:
        # Writes are reads of the internal surface for this measure's purpose:
        # both couple the test to an internal name.
        assert metrics.private_reads('worker._inflight = 1\n', path='t.py') == 1

    def test_unparseable_source_raises_naming_the_path(self) -> None:
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.private_reads('def (:\n', path='broken.py')
        assert 'broken.py' in str(excinfo.value)


class TestTestFileMeasures:
    def test_a_non_lane_importing_file_yields_none(self) -> None:
        # The exclusion is a property of the MEASURE, not of the caller: a
        # non-lane-importing file can never be summed into the report by
        # accident.
        source = 'import orchestrator.workflow\n\nworker._x\n'
        assert metrics.test_file_measures(source, path='t.py') is None

    def test_a_lane_importing_file_yields_both_measures(self) -> None:
        source = (
            'from orchestrator.merge_queue import W\n'
            '\n'
            "patch('orchestrator.merge_queue.foo', x)\n"
            'worker._a._b\n'
        )
        assert metrics.test_file_measures(source, path='t.py') == {
            'patch_targets': ['foo'],
            'private_reads': 2,
        }

    def test_patch_targets_are_sorted(self) -> None:
        source = (
            'from orchestrator.merge_queue import W\n'
            "patch('orchestrator.merge_queue.z', x)\n"
            "patch('orchestrator.merge_queue.a', x)\n"
        )
        measures = metrics.test_file_measures(source, path='t.py')
        assert measures is not None
        assert measures['patch_targets'] == ['a', 'z']

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
        total_private = metrics.derive_totals(live_report)['private_reads']
        assert lane_files >= 150, lane_files
        assert total_private > 5000, total_private


# ---------------------------------------------------------------------------
# Report assembly, and DERIVED totals.


def _synthetic_report() -> dict:
    return {
        'schema_version': 1,
        'params': {
            'complexipy_version': '6.2.0',
            'cluster_paths': ['a.py', 'b.py'],
            'file_line_ceiling': 1500,
            'new_function_cognitive_ceiling': 15,
        },
        'enumeration': {
            'requested': ['a.py', 'b.py'],
            'resolved': ['a.py', 'b.py'],
            # The LIVE shape: the test-tree half is two counts, and it is the one
            # key render_baseline drops on the way to the committed file.
            'test_tree': {'requested': 9, 'resolved': 2},
            'unreadable': [],
            'complete': True,
        },
        'files': {
            'a.py': {
                'lines': 1000,
                'prose_lines': 400,
                'cognitive': 120,
                'function_local_imports': 3,
                'reexport_names': 5,
            },
            'b.py': {
                'lines': 200,
                'prose_lines': 50,
                'cognitive': 30,
                'function_local_imports': 1,
                'reexport_names': 0,
            },
        },
        'functions': {'a.py::f': 40, 'a.py::C::m': 12, 'b.py::g': 7},
        'tests': {
            't1.py': {'patch_targets': ['foo', 'bar'], 'private_reads': 20},
            't2.py': {'patch_targets': ['bar', 'baz'], 'private_reads': 5},
        },
    }


class TestDeriveTotals:
    def test_sums_every_file_measure(self) -> None:
        totals = metrics.derive_totals(_synthetic_report())
        assert totals['lines'] == 1200
        assert totals['prose_lines'] == 450
        assert totals['cognitive'] == 150
        assert totals['function_local_imports'] == 4
        assert totals['reexport_names'] == 5

    def test_private_reads_are_summed_over_tests(self) -> None:
        assert metrics.derive_totals(_synthetic_report())['private_reads'] == 25

    def test_patch_targets_total_is_the_union_size_not_the_sum(self) -> None:
        # The PRD's measure is DISTINCT names: foo/bar/baz across two files that
        # both patch `bar` is 3, not 4.
        assert metrics.derive_totals(_synthetic_report())['patch_targets'] == 3

    def test_moving_code_to_a_new_path_leaves_every_total_identical(self) -> None:
        # THE ANTI-RENAME-GAMING PROPERTY. Totals are DERIVED by summing the
        # stored per-path map rather than stored as their own key, so moving 500
        # lines and 40 cognitive from a.py to a brand-new path cannot lower the
        # cluster figure -- the ratchet still catches the move.
        before = _synthetic_report()
        after = _synthetic_report()
        after['files']['a.py']['lines'] -= 500
        after['files']['a.py']['cognitive'] -= 40
        after['files']['new_module.py'] = {
            'lines': 500,
            'prose_lines': 0,
            'cognitive': 40,
            'function_local_imports': 0,
            'reexport_names': 0,
        }
        assert metrics.derive_totals(after) == metrics.derive_totals(before)

    def test_is_a_pure_function_of_the_report(self) -> None:
        report = _synthetic_report()
        snapshot = json.dumps(report, sort_keys=True)
        metrics.derive_totals(report)
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
        assert params['complexipy_version'] == metrics.complexipy_version()
        assert params['cluster_paths'] == list(metrics.CLUSTER_PATHS)
        assert params['file_line_ceiling'] == metrics.FILE_LINE_CEILING
        assert params['new_function_cognitive_ceiling'] == (
            metrics.NEW_FUNCTION_COGNITIVE_CEILING
        )

    def test_enumeration_is_complete(self, report: dict) -> None:
        assert report['enumeration']['complete'] is True
        assert report['enumeration']['unreadable'] == []

    def test_file_entries_carry_the_five_measures(self, report: dict) -> None:
        entry = report['files']['orchestrator/src/orchestrator/merge_queue.py']
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
        key = 'orchestrator/src/orchestrator/merge_queue.py::SpeculativeMergeWorker::_verifier_loop'
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
        assert metrics.derive_totals(report)['cognitive'] >= 4000

    def test_each_cluster_file_is_measured_by_complexipy_once(
        self, live_measurement: LiveMeasurement
    ) -> None:
        # build_report wants two cognitive projections per cluster file, and
        # used to fetch each from its own `_file_complexity` call -- 50 runs
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
    cluster = set(metrics.CLUSTER_PATHS)
    strays = {}
    for key, value in block.items():
        if not isinstance(value, list):
            continue
        found = [
            entry for entry in value
            if isinstance(entry, str)
            and entry.startswith(f'{metrics.TESTS_ROOT}/')
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
        assert enumeration['requested'] == list(metrics.CLUSTER_PATHS)

    def test_no_test_tree_path_list_survives_anywhere_in_the_block(
        self, enumeration: dict
    ) -> None:
        # THE HEADLINE PROPERTY, executably: the only orchestrator/tests paths
        # left in any list are CLUSTER_PATHS literals. Measured today: exactly
        # the 3 that live there (conftest.py, _merge_queue_harness.py,
        # _serial_merge_worker.py).
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


class TestTestTreeSweep:
    """`_sweep_test_tree` in isolation -- no complexipy, so this runs in ms."""

    @staticmethod
    def _tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        tests = tmp_path / 'orchestrator' / 'tests'
        tests.mkdir(parents=True)
        (tests / 'test_lane.py').write_text(
            'from orchestrator import merge_queue\n\n'
            'def test_x():\n'
            '    assert merge_queue._worker is None\n',
            encoding='utf-8',
        )
        (tests / 'test_unrelated.py').write_text(
            'import json\n\ndef test_y():\n    assert json\n', encoding='utf-8'
        )
        (tests / 'test_broken.py').write_text('def (:\n', encoding='utf-8')
        monkeypatch.setattr(metrics, 'TESTS_ROOT', 'orchestrator/tests')
        return tmp_path

    def test_counts_are_integers_not_path_lists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The sweep's contract: 3 files asked for, 1 measured, as COUNTS.
        root = self._tree(tmp_path, monkeypatch)
        tests, unreadable, coverage = metrics._sweep_test_tree(root)
        assert coverage.requested == 3
        assert coverage.resolved == 1
        assert list(tests) == ['orchestrator/tests/test_lane.py']

    def test_the_unparseable_file_is_named_and_only_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # INV-11 POLARITY at the sweep boundary: the skip is fail-SOFT per file
        # (an unrelated mid-edit test must not redden the lane's ratchet) but the
        # RECORD is not -- the path is named verbatim.
        root = self._tree(tmp_path, monkeypatch)
        _tests, unreadable, _coverage = metrics._sweep_test_tree(root)
        assert list(unreadable) == ['orchestrator/tests/test_broken.py']

    def test_a_skipped_file_marks_the_composed_block_incomplete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The seam that makes check_against_baseline REFUSE to compare: the
        # sweep's soft skip must still reach `complete` in the report block.
        root = self._tree(tmp_path, monkeypatch)
        _tests, unreadable, coverage = metrics._sweep_test_tree(root)
        cluster = metrics.Enumeration(
            requested=('a.py',), resolved=('a.py',), unreadable=(), complete=True
        )
        block = metrics._report_enumeration(cluster, unreadable, coverage)
        assert block['complete'] is False
        assert block['unreadable'] == ['orchestrator/tests/test_broken.py']
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics._require_complete_enumeration({'enumeration': block})
        assert 'orchestrator/tests/test_broken.py' in str(excinfo.value)

    def test_a_clean_sweep_composes_a_complete_block(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The other polarity, so the test above cannot pass vacuously.
        tests_dir = tmp_path / 'orchestrator' / 'tests'
        tests_dir.mkdir(parents=True)
        (tests_dir / 'test_ok.py').write_text('import json\n', encoding='utf-8')
        monkeypatch.setattr(metrics, 'TESTS_ROOT', 'orchestrator/tests')
        _tests, unreadable, coverage = metrics._sweep_test_tree(tmp_path)
        cluster = metrics.Enumeration(
            requested=('a.py',), resolved=('a.py',), unreadable=(), complete=True
        )
        block = metrics._report_enumeration(cluster, unreadable, coverage)
        assert block['complete'] is True
        assert block['unreadable'] == []
        assert block['test_tree'] == {'requested': 1, 'resolved': 0}

    def test_an_unreadable_cluster_path_is_not_masked_by_a_clean_sweep(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Both halves' unreadable paths travel verbatim in ONE top-level list --
        # the cluster half's finding must not be diluted or dropped.
        tests_dir = tmp_path / 'orchestrator' / 'tests'
        tests_dir.mkdir(parents=True)
        monkeypatch.setattr(metrics, 'TESTS_ROOT', 'orchestrator/tests')
        _tests, unreadable, coverage = metrics._sweep_test_tree(tmp_path)
        cluster = metrics.Enumeration(
            requested=('a.py',), resolved=(), unreadable=('a.py',), complete=False
        )
        block = metrics._report_enumeration(cluster, unreadable, coverage)
        assert block['unreadable'] == ['a.py']
        assert block['complete'] is False


# ---------------------------------------------------------------------------
# Baseline serialization, and the per-path LINE LOCALITY that makes ten
# parallel gamma branches rebase without conflicting.


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
        report = _synthetic_report()
        report['enumeration'].update(enumeration)
        return report

    @staticmethod
    def _stored(report: dict) -> dict:
        return json.loads(metrics.render_baseline(report))['enumeration']

    def test_the_stored_key_set_omits_the_test_tree_counts(self) -> None:
        # No test-tree NUMBER reaches the file, only the cluster half plus
        # INV-11's two completeness keys.
        assert set(self._stored(_synthetic_report())) == {
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
            requested=list(metrics.CLUSTER_PATHS),
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
        globs = [entry for entry in metrics.CLUSTER_PATHS if '*' in entry]
        assert globs, 'expected at least one glob entry in CLUSTER_PATHS'
        report = self._report(requested=list(metrics.CLUSTER_PATHS))
        kept = self._stored(report)['requested']
        assert kept == list(metrics.CLUSTER_PATHS)
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
        once = self._stored(_synthetic_report())
        assert metrics._stored_enumeration(once) == once

    def test_the_rendered_baseline_is_stable_across_an_unrelated_new_test(
        self,
    ) -> None:
        # THE REGRESSION, end to end and in the units that bit (esc-5021-7):
        # adding one non-lane test file to the sweep must not change a single
        # byte. Now STRENGTHENED to both counts, because storing either one would
        # reintroduce it -- `requested` moves on ANY .py arriving anywhere under
        # orchestrator/tests, and `resolved` moves whenever a lane-importing one
        # is added or deleted, which is a shared line every gamma branch rewrites.
        report = _synthetic_report()
        before = metrics.render_baseline(report)
        report['enumeration']['test_tree']['requested'] += 1
        assert metrics.render_baseline(report) == before
        report['enumeration']['test_tree']['resolved'] += 1
        assert metrics.render_baseline(report) == before


class TestRenderBaseline:
    def test_round_trips_without_dropping_a_measure(self) -> None:
        # A hand-rolled writer's failure mode is a silently omitted measure, so
        # the round-trip is asserted structurally rather than eyeballed.
        report = _synthetic_report()
        loaded = json.loads(metrics.render_baseline(report))
        assert loaded.pop('_README') == metrics.BASELINE_README
        # The ONE measure deliberately not stored, popped here the same way
        # _README is: the test-tree denominator is reported by --report, never
        # ratcheted, so freezing it would churn the file on every unrelated test
        # file the repo gains. Everything else must survive verbatim.
        assert report['enumeration'].pop('test_tree') == {'requested': 9, 'resolved': 2}
        assert loaded == report

    def test_every_per_path_entry_occupies_exactly_one_line(self) -> None:
        # THE PARALLELISM PROPERTY. PRD gamma1..gamma10 run concurrently and each
        # lowers only its own group's numbers, rebasing through the merge lane.
        # One path per LINE makes those ten edits disjoint hunks; json.dumps(
        # indent=2) would spread merge_queue.py's five measures over six lines
        # and put two branches' unrelated edits inside one conflicting hunk.
        report = _synthetic_report()
        rendered = metrics.render_baseline(report)
        lines = rendered.splitlines()
        for section in ('files', 'functions', 'tests'):
            for key, value in report[section].items():
                prefix = json.dumps(key) + ':'
                hits = [line for line in lines if line.lstrip().startswith(prefix)]
                assert len(hits) == 1, f'{section}.{key} is not on exactly one line'
                # The whole value must parse from that ONE line -- proving it was
                # not merely started there and continued on the next.
                tail = hits[0].lstrip()[len(prefix):].strip().rstrip(',')
                assert json.loads(tail) == value, f'{section}.{key} value spans lines'

    def test_per_path_keys_are_emitted_in_sorted_order(self) -> None:
        report = _synthetic_report()
        rendered = metrics.render_baseline(report)
        for section in ('files', 'functions', 'tests'):
            positions = [
                rendered.index(json.dumps(key) + ':') for key in sorted(report[section])
            ]
            assert positions == sorted(positions), section

    def test_ends_with_exactly_one_trailing_newline(self) -> None:
        rendered = metrics.render_baseline(_synthetic_report())
        assert rendered.endswith('\n')
        assert not rendered.endswith('\n\n')

    def test_rendering_is_idempotent(self) -> None:
        # Regenerating a baseline from a baseline must be a no-op, or every
        # regeneration would churn the file and manufacture conflicts.
        once = metrics.render_baseline(_synthetic_report())
        twice = metrics.render_baseline(json.loads(once))
        assert twice == once

    def test_leads_with_a_readme_key(self) -> None:
        # STRUCTURE only. This used to also assert three substrings of
        # BASELINE_README ('ratchet baseline', 'regenerate', 'same commit'),
        # which pinned the constant's PROSE: rewording it to "refresh the frozen
        # numbers in the landing commit" would have reddened the suite with
        # nothing executable changed. Those bought no coverage either --
        # TestRenderBaseline::test_round_trips_without_dropping_a_measure
        # already proves BASELINE_README is exactly what gets emitted. If the
        # README's guidance matters, fix the constant; do not lock its phrasing.
        rendered = metrics.render_baseline(_synthetic_report())
        first_key_line = rendered.splitlines()[1]
        assert first_key_line.lstrip().startswith('"_README":')
        # Non-vacuity only, in this module's floor idiom: the key must not be
        # emitted empty. No wording is constrained.
        assert len(metrics.BASELINE_README) >= 100


class TestBaselineIO:
    def test_write_then_load_round_trips(self, tmp_path: Path) -> None:
        report = _synthetic_report()
        target = tmp_path / 'baseline.json'
        metrics.write_baseline(target, report)
        assert target.read_text(encoding='utf-8') == metrics.render_baseline(report)
        loaded = metrics.load_baseline(target)
        assert loaded['files'] == report['files']
        assert loaded['functions'] == report['functions']

    def test_write_is_atomic_leaving_no_debris(self, tmp_path: Path) -> None:
        target = tmp_path / 'baseline.json'
        metrics.write_baseline(target, _synthetic_report())
        assert [p.name for p in tmp_path.iterdir()] == ['baseline.json']

    def test_missing_baseline_is_a_named_hard_failure(self, tmp_path: Path) -> None:
        # INV-11: a missing baseline is never an empty-baseline PASS, which
        # would silently disarm the ratchet for every downstream task.
        missing = tmp_path / 'nope.json'
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.load_baseline(missing)
        assert 'nope.json' in str(excinfo.value)

    def test_malformed_baseline_is_a_named_hard_failure(self, tmp_path: Path) -> None:
        target = tmp_path / 'baseline.json'
        target.write_text('{"files": ', encoding='utf-8')
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.load_baseline(target)
        assert 'baseline.json' in str(excinfo.value)

    def test_non_object_baseline_is_a_named_hard_failure(self, tmp_path: Path) -> None:
        target = tmp_path / 'baseline.json'
        target.write_text('[]', encoding='utf-8')
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.load_baseline(target)
        assert 'baseline.json' in str(excinfo.value)


class TestWriteBaselineRefusesAnUnauthorizedRaise:
    """The write path is where the ratchet is actually enforceable.

    ``test_baseline_matches_a_fresh_measurement`` forces the committed baseline
    to equal the tree in every committed state, so the COMPARATOR is trivially
    clean on every commit: a raise is red only in the window between editing the
    code and regenerating. Regeneration then absorbs it, because
    ``--write-baseline`` reads nothing before it overwrites. So the only place a
    gate can bite is the write itself, and that is what these pin.

    Synthetic dicts throughout, driven through the public ``write_baseline``:
    microseconds, and every branch reached directly rather than only through a
    71-second measurement.
    """

    @staticmethod
    def _seeded(tmp_path: Path) -> Path:
        """A destination already holding the seed baseline, byte-for-byte."""
        target = tmp_path / 'b.json'
        metrics.write_baseline(target, _ratchet_baseline())
        return target

    def test_a_per_file_rise_is_refused_naming_the_measure_and_both_numbers(
        self, tmp_path: Path
    ) -> None:
        target = self._seeded(tmp_path)
        current = copy.deepcopy(_ratchet_baseline())
        current['files'][_MQ]['lines'] += 103  # the real task-5342 shape

        with pytest.raises(metrics.UnauthorizedRaise) as excinfo:
            metrics.write_baseline(target, current)

        message = str(excinfo.value)
        assert 'lines' in message
        assert _MQ in message
        assert '21550' in message and '21653' in message
        # The refusal must name the sanctioned path, not merely forbid the move.
        assert '--authorize-raise' in message

    def test_the_refusal_carries_the_measured_violations(
        self, tmp_path: Path
    ) -> None:
        target = self._seeded(tmp_path)
        current = copy.deepcopy(_ratchet_baseline())
        current['files'][_MQ]['lines'] += 103

        with pytest.raises(metrics.UnauthorizedRaise) as excinfo:
            metrics.write_baseline(target, current)

        violations = excinfo.value.violations
        assert isinstance(violations, tuple)
        assert violations
        assert all(isinstance(v, metrics.Violation) for v in violations)
        # Every one is a genuine rise: the refusal reports what it measured, so
        # the ledger record derived from it cannot disagree with what landed.
        assert all(v.current > v.baseline for v in violations)

    def test_a_refused_write_leaves_the_destination_byte_for_byte(
        self, tmp_path: Path
    ) -> None:
        target = self._seeded(tmp_path)
        before = target.read_text(encoding='utf-8')
        current = copy.deepcopy(_ratchet_baseline())
        current['files'][_MQ]['lines'] += 103

        with pytest.raises(metrics.UnauthorizedRaise):
            metrics.write_baseline(target, current)

        # A truncated or half-written baseline would be a WIDENED ratchet, the
        # one failure mode this instrument must never produce.
        assert target.read_text(encoding='utf-8') == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ['b.json']

    @pytest.mark.parametrize(
        ('mutate', 'measure'),
        [
            pytest.param(
                lambda cur: cur['files'][_MQ].__setitem__(
                    'cognitive', cur['files'][_MQ]['cognitive'] + 1
                ),
                'cognitive',
                id='files',
            ),
            pytest.param(
                lambda cur: cur['functions'].__setitem__(
                    _VERIFIER_LOOP, cur['functions'][_VERIFIER_LOOP] + 1
                ),
                'cognitive',
                id='functions',
            ),
            pytest.param(
                lambda cur: cur['tests'][_TEST_FILE].__setitem__(
                    'private_reads', cur['tests'][_TEST_FILE]['private_reads'] + 1
                ),
                'private_reads',
                id='tests',
            ),
        ],
    )
    def test_every_comparison_arm_reaches_the_write_gate(
        self, tmp_path: Path, mutate, measure: str
    ) -> None:
        # One arm left out of the write gate is one section a task could widen
        # by regenerating -- the same hole, one measure narrower.
        target = self._seeded(tmp_path)
        current = copy.deepcopy(_ratchet_baseline())
        mutate(current)

        with pytest.raises(metrics.UnauthorizedRaise) as excinfo:
            metrics.write_baseline(target, current)

        assert measure in {v.measure for v in excinfo.value.violations}

    def test_rewriting_an_identical_report_is_not_a_raise(
        self, tmp_path: Path
    ) -> None:
        # ANTI-VACUITY: a gate that refused everything would pass every test
        # above while making the sanctioned workflow impossible.
        target = self._seeded(tmp_path)
        assert metrics.write_baseline(target, _ratchet_baseline()) == target

    def test_lowering_a_measure_stays_frictionless(self, tmp_path: Path) -> None:
        # Lowering is the POINT. It must cost nothing extra.
        target = self._seeded(tmp_path)
        current = copy.deepcopy(_ratchet_baseline())
        current['files'][_MQ]['lines'] -= 4000

        assert metrics.write_baseline(target, current) == target
        assert target.read_text(encoding='utf-8') == metrics.render_baseline(current)

    def test_unauthorized_raise_is_not_a_metrics_error(self) -> None:
        # The exit ladder: 1 is "a measure rose", 2 is "the instrument broke".
        # main() maps every MetricsError to 2, so a refusal that subclassed it
        # would report a real regression as a broken tool.
        assert not issubclass(metrics.UnauthorizedRaise, metrics.MetricsError)


# ---------------------------------------------------------------------------
# The ratchet comparator, and INV-10 tier 1.
#
# Every assertion below drives check_against_baseline with SYNTHETIC dicts, so
# each branch is pinned directly rather than only ever reached through a real
# measurement that happens to walk over it. The comparator is a pure function of
# two plain dicts -- no filesystem, no complexipy -- which is what makes the
# seeded-fixture self-test run in microseconds AND lets the CLI's --check reuse
# the identical code path this test asserts on (SPOT: one ratchet, not two).

_MQ = 'orchestrator/src/orchestrator/merge_queue.py'
_GIT_OPS = 'orchestrator/src/orchestrator/git_ops.py'
_VERIFIER_LOOP = f'{_MQ}::SpeculativeMergeWorker::_verifier_loop'
_TEST_FILE = 'orchestrator/tests/test_merge_queue.py'


def _ratchet_baseline() -> dict:
    """A miniature but REAL-SHAPED baseline: real paths, real measured numbers."""
    return {
        'schema_version': 1,
        'params': {
            'complexipy_version': '6.2.0',
            'cluster_paths': list(metrics.CLUSTER_PATHS),
            'file_line_ceiling': metrics.FILE_LINE_CEILING,
            'new_function_cognitive_ceiling': (
                metrics.NEW_FUNCTION_COGNITIVE_CEILING
            ),
        },
        # REAL-SHAPED means the STORED shape, which has no `test_tree` key: the
        # test-tree counts are live-only, and this fixture stands in for a
        # committed baseline that the comparator reads.
        'enumeration': {
            'requested': [_MQ, _GIT_OPS],
            'resolved': [_MQ, _GIT_OPS],
            'unreadable': [],
            'complete': True,
        },
        'files': {
            _MQ: {
                'lines': 21550,
                'prose_lines': 9000,
                'cognitive': 2133,
                'function_local_imports': 12,
                'reexport_names': 9,
            },
            _GIT_OPS: {
                'lines': 14721,
                'prose_lines': 6000,
                'cognitive': 1361,
                'function_local_imports': 5,
                'reexport_names': 2,
            },
        },
        'functions': {_VERIFIER_LOOP: 245, f'{_GIT_OPS}::GitOps::advance_main': 108},
        'tests': {
            _TEST_FILE: {
                'patch_targets': ['run_scoped_verification', 'time'],
                'private_reads': 300,
            },
        },
    }


def _blank_file_entry(**overrides: int) -> dict:
    entry = {
        'lines': 0,
        'prose_lines': 0,
        'cognitive': 0,
        'function_local_imports': 0,
        'reexport_names': 0,
    }
    entry.update(overrides)
    return entry


def _bump_file(measure: str):
    def mutate(current: dict) -> None:
        current['files'][_MQ][measure] += 1

    return mutate


def _add_file(measure: str):
    def mutate(current: dict) -> None:
        # A brand-new path carrying +1 of exactly ONE measure. It is absent from
        # the baseline, so no per-path comparison fires -- only the derived
        # cluster total can catch it, which is precisely the rename-gaming shape.
        current['files']['orchestrator/src/orchestrator/merge_new.py'] = (
            _blank_file_entry(**{measure: 1})
        )

    return mutate


#: (test id, mutation, expected violation measure, expected violation key).
#: THE SEEDED-FIXTURE SELF-TEST (INV-10 tier 1): every ratcheted measure and
#: every derived total gets its own +1, and the ratchet must EXECUTE to a named
#: violation for it. A measure missing from this table is a measure nothing
#: proves is actually compared.
_SEEDS = [
    ('files.lines', _bump_file('lines'), 'lines', _MQ),
    ('files.prose_lines', _bump_file('prose_lines'), 'prose_lines', _MQ),
    ('files.cognitive', _bump_file('cognitive'), 'cognitive', _MQ),
    (
        'files.function_local_imports',
        _bump_file('function_local_imports'),
        'function_local_imports',
        _MQ,
    ),
    ('files.reexport_names', _bump_file('reexport_names'), 'reexport_names', _MQ),
    (
        'functions',
        lambda cur: cur['functions'].__setitem__(
            _VERIFIER_LOOP, cur['functions'][_VERIFIER_LOOP] + 1
        ),
        'cognitive',
        _VERIFIER_LOOP,
    ),
    (
        'tests.private_reads',
        lambda cur: cur['tests'][_TEST_FILE].__setitem__(
            'private_reads', cur['tests'][_TEST_FILE]['private_reads'] + 1
        ),
        'private_reads',
        _TEST_FILE,
    ),
    (
        'tests.patch_targets',
        lambda cur: cur['tests'][_TEST_FILE]['patch_targets'].append('_new_leaf'),
        'patch_targets',
        _TEST_FILE,
    ),
    ('total.lines', _add_file('lines'), 'total:lines', metrics.CLUSTER_TOTAL_KEY),
    (
        'total.prose_lines',
        _add_file('prose_lines'),
        'total:prose_lines',
        metrics.CLUSTER_TOTAL_KEY,
    ),
    (
        'total.cognitive',
        _add_file('cognitive'),
        'total:cognitive',
        metrics.CLUSTER_TOTAL_KEY,
    ),
    (
        'total.function_local_imports',
        _add_file('function_local_imports'),
        'total:function_local_imports',
        metrics.CLUSTER_TOTAL_KEY,
    ),
    (
        'total.reexport_names',
        _add_file('reexport_names'),
        'total:reexport_names',
        metrics.CLUSTER_TOTAL_KEY,
    ),
    (
        'total.private_reads',
        lambda cur: cur['tests'].__setitem__(
            'orchestrator/tests/test_brand_new.py',
            {'patch_targets': [], 'private_reads': 1},
        ),
        'total:private_reads',
        metrics.CLUSTER_TOTAL_KEY,
    ),
    (
        'total.patch_targets',
        lambda cur: cur['tests'].__setitem__(
            'orchestrator/tests/test_brand_new.py',
            {'patch_targets': ['_brand_new_leaf'], 'private_reads': 0},
        ),
        'total:patch_targets',
        metrics.CLUSTER_TOTAL_KEY,
    ),
]


class TestCheckAgainstBaseline:
    @pytest.mark.parametrize(
        ('mutate', 'measure', 'key'),
        [pytest.param(m, meas, k, id=name) for name, m, meas, k in _SEEDS],
    )
    def test_a_plus_one_on_every_ratcheted_measure_is_caught(
        self, mutate, measure: str, key: str
    ) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        mutate(current)

        violations = metrics.check_against_baseline(current, baseline)

        matching = [v for v in violations if v.measure == measure and v.key == key]
        assert len(matching) == 1, f'{measure} at {key} not reported once: {violations}'
        assert measure in matching[0].message
        assert key in matching[0].message
        # The ONLY collateral a per-path rise may produce is the cluster total
        # that mechanically follows from it. Anything else means the comparator
        # is firing on a measure nobody touched.
        others = [v for v in violations if v is not matching[0]]
        assert all(v.measure == f'total:{measure}' for v in others), others

    def test_an_identical_current_is_clean(self) -> None:
        baseline = _ratchet_baseline()
        assert metrics.check_against_baseline(copy.deepcopy(baseline), baseline) == []

    def test_equality_is_permitted_this_is_a_ratchet_not_a_gate(self) -> None:
        # 62 lane functions already exceed cognitive 15 on the introducing
        # commit. A day-one gate would be red on arrival and get disabled.
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        assert current['functions'][_VERIFIER_LOOP] == 245
        assert metrics.check_against_baseline(current, baseline) == []

    def test_lowering_a_measure_is_clean(self) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['files'][_MQ]['lines'] -= 4000
        current['files'][_MQ]['cognitive'] -= 400
        current['functions'][_VERIFIER_LOOP] = 12
        current['tests'][_TEST_FILE]['private_reads'] = 0
        assert metrics.check_against_baseline(current, baseline) == []


class TestCeilingsApplyOnlyToNewKeys:
    def test_a_new_file_over_the_line_ceiling_is_a_violation(self) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        new_path = 'orchestrator/src/orchestrator/merge_new.py'
        current['files'][new_path] = _blank_file_entry(lines=1501)
        ceilings = [
            v
            for v in metrics.check_against_baseline(current, baseline)
            if v.measure == 'new_file_over_ceiling'
        ]
        assert len(ceilings) == 1
        assert new_path in ceilings[0].message

    def test_a_new_file_exactly_at_the_line_ceiling_is_allowed(self) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['files']['orchestrator/src/orchestrator/merge_new.py'] = (
            _blank_file_entry(lines=1500)
        )
        assert not [
            v
            for v in metrics.check_against_baseline(current, baseline)
            if v.measure == 'new_file_over_ceiling'
        ]

    def test_grandfathered_merge_queue_is_not_a_ceiling_violation(self) -> None:
        # 21,550 lines, fourteen times the ceiling, and clean -- because the
        # ceiling applies only to paths ABSENT from the baseline. That is what
        # makes this a ratchet whose grandfathering can only ever shrink.
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        assert current['files'][_MQ]['lines'] == 21550
        assert metrics.check_against_baseline(current, baseline) == []

    def test_size_ceiling_exemption_covers_git_ops_even_when_new(self) -> None:
        # PRD decision 9. Exercised with git_ops.py ABSENT from the baseline,
        # because that is the only configuration in which the ceiling would
        # otherwise fire -- an exemption never reached is an exemption never
        # tested.
        baseline = _ratchet_baseline()
        del baseline['files'][_GIT_OPS]
        current = copy.deepcopy(baseline)
        current['files'][_GIT_OPS] = _blank_file_entry(lines=14721)
        assert not [
            v
            for v in metrics.check_against_baseline(current, baseline)
            if v.measure == 'new_file_over_ceiling'
        ]

    def test_a_non_exempt_path_at_the_same_size_is_a_violation(self) -> None:
        # The contrast that proves the previous test measured the EXEMPTION and
        # not merely a comparator that never fires.
        baseline = _ratchet_baseline()
        del baseline['files'][_GIT_OPS]
        current = copy.deepcopy(baseline)
        current['files']['orchestrator/src/orchestrator/merge_new.py'] = (
            _blank_file_entry(lines=14721)
        )
        assert [
            v
            for v in metrics.check_against_baseline(current, baseline)
            if v.measure == 'new_file_over_ceiling'
        ]

    def test_the_exemption_is_scoped_to_the_ceiling_not_to_the_ratchet(self) -> None:
        # git_ops.py's 14,721 lines are exempt from the CEILING and still frozen
        # by the RATCHET, so they cannot grow unwatched.
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['files'][_GIT_OPS]['lines'] += 1
        assert [
            v
            for v in metrics.check_against_baseline(current, baseline)
            if v.measure == 'lines' and v.key == _GIT_OPS
        ]

    def test_a_new_function_over_the_cognitive_ceiling_is_a_violation(self) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['functions'][f'{_MQ}::brand_new'] = 16
        ceilings = [
            v
            for v in metrics.check_against_baseline(current, baseline)
            if v.measure == 'new_function_over_ceiling'
        ]
        assert len(ceilings) == 1
        assert 'brand_new' in ceilings[0].message

    def test_a_new_function_exactly_at_the_cognitive_ceiling_is_allowed(self) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['functions'][f'{_MQ}::brand_new'] = 15
        assert metrics.check_against_baseline(current, baseline) == []

    def test_a_grandfathered_function_at_245_is_clean(self) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['functions'][_VERIFIER_LOOP] = 245
        assert metrics.check_against_baseline(current, baseline) == []


class TestComparatorPreconditions:
    def test_a_complexipy_version_drift_is_a_hard_failure(self) -> None:
        # Cognitive numbers are version-dependent: the SAME merge_queue.py
        # measures 2031 at complexipy 3.0.0, 2092 at 5.0.0 and 2133 at 6.x/7.x.
        # A silent version drift would rewrite every number at once and leave a
        # green ratchet comparing two incomparable measurements.
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['params']['complexipy_version'] = '7.0.1'
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.check_against_baseline(current, baseline)
        assert '7.0.1' in str(excinfo.value)
        assert '6.2.0' in str(excinfo.value)

    def test_a_cluster_path_edit_forces_a_deliberate_regeneration(self) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        baseline['params']['cluster_paths'] = [
            p for p in metrics.CLUSTER_PATHS if p != _GIT_OPS
        ] + ['orchestrator/src/orchestrator/ghost.py']
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.check_against_baseline(current, baseline)
        message = str(excinfo.value)
        assert _GIT_OPS in message
        assert 'ghost.py' in message

    def test_an_incomplete_enumeration_is_never_silently_compared(self) -> None:
        # INV-11. A sweep that skipped files measures LOWER than the truth, so
        # comparing it would read as a clean tree -- or worse, as an
        # improvement worth writing into the baseline.
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['enumeration']['complete'] = False
        current['enumeration']['unreadable'] = ['orchestrator/tests/test_broken.py']
        with pytest.raises(metrics.MetricsError) as excinfo:
            metrics.check_against_baseline(current, baseline)
        assert 'test_broken.py' in str(excinfo.value)

    def test_preconditions_are_checked_before_any_comparison(self) -> None:
        # Order matters: a wrong-version or partial measurement must fail with
        # its own named cause, not with a wall of downstream violations that
        # sends the reader hunting a regression that does not exist.
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['enumeration']['complete'] = False
        current['files'][_MQ]['lines'] += 5000
        with pytest.raises(metrics.MetricsError):
            metrics.check_against_baseline(current, baseline)


class TestViolationShape:
    def test_a_violation_carries_both_numbers_and_a_stable_order(self) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['files'][_MQ]['lines'] += 1
        current['files'][_GIT_OPS]['cognitive'] += 1
        violations = metrics.check_against_baseline(current, baseline)
        per_path = [v for v in violations if v.key in (_MQ, _GIT_OPS)]
        assert len(per_path) == 2
        for violation in per_path:
            assert violation.current == violation.baseline + 1
        # Sorted, so a failure message is diffable run to run.
        assert violations == sorted(
            violations, key=lambda v: (v.measure, v.key)
        )

    def test_the_comparator_mutates_neither_input(self) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['files'][_MQ]['lines'] += 1
        before = (json.dumps(current, sort_keys=True), json.dumps(baseline, sort_keys=True))
        metrics.check_against_baseline(current, baseline)
        after = (json.dumps(current, sort_keys=True), json.dumps(baseline, sort_keys=True))
        assert after == before


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
        metrics, 'build_report', lambda root: copy.deepcopy(live_report)
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
        totals = metrics.derive_totals(stub_measurement)
        assert str(totals['cognitive']) in out
        assert str(totals['lines']) in out

    def test_report_carries_the_test_suite_measures(
        self, stub_measurement: dict, capsys: pytest.CaptureFixture[str]
    ) -> None:
        metrics.main(['--report'])
        out = capsys.readouterr().out
        totals = metrics.derive_totals(stub_measurement)
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
        metrics.write_baseline(baseline, stub_measurement)
        assert metrics.main(['--check', '--baseline', str(baseline)]) == 0
        assert capsys.readouterr().err == ''

    def test_check_returns_one_and_prints_violations_for_a_doctored_baseline(
        self, stub_measurement: dict, tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        doctored = copy.deepcopy(stub_measurement)
        doctored['files'][_MQ]['lines'] -= 10
        baseline = tmp_path / 'baseline.json'
        metrics.write_baseline(baseline, doctored)
        assert metrics.main(['--check', '--baseline', str(baseline)]) == 1
        err = capsys.readouterr().err
        assert _MQ in err
        assert 'lines' in err

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

        monkeypatch.setattr(metrics, 'build_report', recording)
        assert metrics.main(['--check']) == 0
        assert [r.resolve() for r in roots] == [_REPO_ROOT.resolve()]


class TestWriteBaselineCli:
    def test_write_baseline_returns_zero_and_writes_the_rendered_bytes(
        self, stub_measurement: dict, tmp_path: Path
    ) -> None:
        target = tmp_path / 'b.json'
        assert metrics.main(['--write-baseline', str(target)]) == 0
        assert target.read_text(encoding='utf-8') == metrics.render_baseline(
            stub_measurement
        )


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
            raise metrics.MetricsError('complexipy 7.0.1 is outside >=6.2,<7')

        monkeypatch.setattr(metrics, 'build_report', explode)
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


# ---------------------------------------------------------------------------
# THE RATCHET ITSELF.
#
# Everything above pins the instrument. This is the gate: the live tree, the
# committed baseline, and the comparator the CLI's --check runs verbatim.

_BASELINE_PATH = _REPO_ROOT / metrics.BASELINE_RELPATH


@pytest.fixture(scope='module')
def committed_baseline() -> dict:
    return metrics.load_baseline(_BASELINE_PATH)


def test_merge_lane_ratchet_holds(
    live_report: dict, committed_baseline: dict
) -> None:
    violations = metrics.check_against_baseline(live_report, committed_baseline)
    assert not violations, (
        'The merge-lane quality ratchet has been breached by '
        f'{len(violations)} measure(s):\n'
        + '\n'.join(f'  - {v.message}' for v in violations)
        + '\n\nRemedy: LOWER the measure. A task that legitimately lowers one '
        'regenerates the baseline in the SAME commit:\n'
        '  python scripts/merge_lane_metrics.py --write-baseline '
        f'{metrics.BASELINE_RELPATH}\n'
        'A task may never RAISE a measure, and regenerating the baseline to '
        'make this test go green silently widens the ratchet for every task '
        'that follows.'
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
        assert metrics.derive_totals(committed_baseline)['cognitive'] >= 4000

    def test_the_baseline_line_total_is_real(self, committed_baseline: dict) -> None:
        # Measured 54,048 lines across the 22-path cluster.
        assert metrics.derive_totals(committed_baseline)['lines'] >= 50000

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
        assert enumeration['requested'] == list(metrics.CLUSTER_PATHS)
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
    assert _BASELINE_PATH.read_text(encoding='utf-8') == metrics.render_baseline(
        live_report
    ), (
        'The committed baseline is not what measuring this tree produces. '
        'Regenerate it in this commit if you lowered a measure:\n'
        '  python scripts/merge_lane_metrics.py --write-baseline '
        f'{metrics.BASELINE_RELPATH}'
    )
