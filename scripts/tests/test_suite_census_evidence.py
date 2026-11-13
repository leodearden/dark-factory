"""The evidence readers turn retained verify artefacts into typed census records."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
import suite_census_evidence as ev
import suite_census_outcomes as oc
from orchestrator.flake_ledger import UNKNOWN_TEST_ID, FlakeCallSite
from shared.project_layout import runs_db
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from suite_census_fixtures import (
    flake_db,
    git_tree,
    junit_xml,
    nextest_project,
    pytest_project,
    write_gz,
)

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

PASSED, FAILED, SKIPPED = oc.Outcome.PASSED, oc.Outcome.FAILED, oc.Outcome.SKIPPED

TEST_A = 'orchestrator/tests/test_a.py'


def _observations(records, source: str) -> list[oc.Observation]:
    return [r for r in records if isinstance(r, oc.Observation) and r.source == source]


def _unresolved(records, source: str) -> list[str]:
    return [r.raw_id for r in records if isinstance(r, oc.Unresolved) and r.source == source]


def _outcomes(observations, name: str) -> list[tuple[oc.Outcome, float | None]]:
    return [(o.outcome, o.seconds) for o in observations if o.test.name == name]


class TestPytestEvidence:
    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        return pytest_project(tmp_path)

    @pytest.fixture
    def records(self, root: Path) -> list[oc.Record]:
        return list(ev.pytest_evidence(root, root).records())

    def test_test_id_is_the_repo_relative_function(self, records):
        archived = _observations(records, 'archived-junit')
        assert oc.TestId('orchestrator', f'{TEST_A}::TestX::test_p') in {o.test for o in archived}
        assert oc.TestId('orchestrator', f'{TEST_A}::test_q') in {o.test for o in archived}

    def test_junit_outcomes_and_seconds(self, records):
        archived = _observations(records, 'archived-junit')
        assert _outcomes(archived, f'{TEST_A}::TestX::test_p') == [(PASSED, 2.0), (PASSED, 3.0)]
        assert _outcomes(archived, f'{TEST_A}::test_q') == [(FAILED, 0.5)]
        assert _outcomes(archived, f'{TEST_A}::test_e') == [(FAILED, 0.2)]
        assert _outcomes(archived, f'{TEST_A}::test_s') == [(SKIPPED, 0.01)]

    def test_a_live_report_of_an_archived_run_is_read_once(self, records):
        live = _observations(records, 'live-junit')
        assert {o.test.name for o in live} == {f'{TEST_A}::test_live'}

    def test_an_untracked_classname_is_unresolved(self, records):
        assert [raw for raw in _unresolved(records, 'archived-junit') if 'test_gone' in raw]

    def test_a_function_or_class_its_file_does_not_define_is_unresolved(self, records):
        names = {o.test.name for o in _observations(records, 'archived-junit')}
        assert not names & {f'{TEST_A}::test_renamed', f'{TEST_A}::TestGone::test_p'}
        unresolved = set(_unresolved(records, 'archived-junit'))
        assert {'tests.test_a::test_renamed', 'tests.test_a.TestGone::test_p'} <= unresolved

    def test_a_root_relative_classname_falls_back_to_the_repo_root(self, records):
        archived = _observations(records, 'archived-junit')
        (b1,) = [o for o in archived if o.test.name.endswith('test_b1')]
        assert b1.test == oc.TestId('tests', 'tests/scripts/test_b.py::test_b1')

    def test_log_failures_resolve_under_the_module_dir(self, records):
        logs = _observations(records, 'pytest-logs')
        assert sorted((o.test.name, o.outcome, o.seconds) for o in logs) == [
            (f'{TEST_A}::TestX::test_p', FAILED, None),
            (f'{TEST_A}::test_r', FAILED, None),
        ]

    def test_a_log_failure_of_an_undefined_function_is_unresolved(self, records):
        assert _unresolved(records, 'pytest-logs') == ['tests/test_a.py::TestX::test_removed']

    def test_an_ambiguous_flake_id_marks_every_candidate(self, records):
        flakes = _observations(records, 'flake_occurrence')
        assert sorted(
            (o.test.package, o.test.name, o.outcome)
            for o in flakes if o.test.name.endswith('::test_q')
        ) == [
            ('fused-memory', 'fused-memory/tests/test_a.py::test_q', FAILED),
            ('orchestrator', f'{TEST_A}::test_q', FAILED),
        ]
        assert sorted(_unresolved(records, 'flake_occurrence')) == [
            '<unknown>', 'tests/test_deleted.py::test_x',
        ]

    def test_an_ambiguous_flake_id_credits_only_the_candidates_that_define_it(self, records):
        flakes = _observations(records, 'flake_occurrence')
        assert [
            (o.test.package, o.test.name) for o in flakes if o.test.name.endswith('::test_flaky')
        ] == [('orchestrator', f'{TEST_A}::TestX::test_flaky')]

    def test_every_run_is_contiguous(self, records):
        keys = [(o.source, o.run) for o in records if isinstance(o, oc.Observation)]
        starts = [key for i, key in enumerate(keys) if i == 0 or keys[i - 1] != key]
        assert len(starts) == len(set(starts))

    def test_windows(self, root):
        windows = {w.source: w for w in ev.pytest_evidence(root, root).windows}
        assert list(windows) == ['archived-junit', 'live-junit', 'pytest-logs', 'flake_occurrence']
        observed = {
            source: (w.present, w.artefacts, w.first, w.last) for source, w in windows.items()
        }
        assert observed == {
            'archived-junit': (True, 2, '2026-09-30T13:15:43Z', '2026-10-01T00:00:00Z'),
            'live-junit': (True, 2, '2026-10-02T12:00:00Z', '2026-10-03T12:00:00Z'),
            'pytest-logs': (True, 1, '2026-09-20T10:10:10Z', '2026-09-20T10:10:10Z'),
            'flake_occurrence': (True, 4, '2026-09-01T00:00:00Z', '2026-09-04T00:00:00Z'),
        }

    def test_an_absent_runs_db_is_a_window_not_an_error(self, root):
        (root / 'data' / 'orchestrator' / 'runs.db').unlink()
        evidence = ev.pytest_evidence(root, root)
        (flake,) = [w for w in evidence.windows if w.source == 'flake_occurrence']
        assert (flake.present, flake.artefacts) == (False, 0)
        assert _observations(list(evidence.records()), 'flake_occurrence') == []

    def test_a_corrupt_archive_is_unresolved_and_the_rest_still_read(self, root):
        corrupt = (
            root / 'data' / 'verify-logs' / 'T0'
            / 'attempt-1.orchestrator.junit-20260929T000000_1Z.xml.gz'
        )
        whole = write_gz(corrupt, '<testsuites>' * 2000).read_bytes()
        corrupt.write_bytes(whole[: len(whole) // 2])
        records = list(ev.pytest_evidence(root, root).records())
        assert [raw for raw in _unresolved(records, 'archived-junit') if raw.endswith(corrupt.name)]
        assert _outcomes(_observations(records, 'archived-junit'), f'{TEST_A}::test_q')

    def test_pytest_has_no_uncosted_universe(self, root):
        assert ev.pytest_evidence(root, root).uncosted_universe == frozenset()


TEST_D = 'orchestrator/tests/test_d.py'


def _junit_records(
    root: Path, files: Mapping[str, str], cases: Sequence[tuple[str, str]],
) -> list[oc.Record]:
    git_tree(root, {'orchestrator/orchestrator.yaml': '', **files})
    write_gz(
        root / 'data' / 'verify-logs' / 'T1'
        / 'attempt-1.orchestrator.junit-20261001T000000_1Z.xml.gz',
        junit_xml(
            '2026-10-01T00:00:00+00:00',
            [(classname, name, 0.1, 'pass') for classname, name in cases],
        ),
    )
    return list(ev.pytest_evidence(root, root).records())


def _resolve(
    root: Path, files: Mapping[str, str], cases: Sequence[tuple[str, str]],
) -> tuple[set[str], set[str]]:
    """The archived-junit test names *cases* resolve to, and the raw ids left unresolved."""
    records = _junit_records(root, files, cases)
    return (
        {o.test.name for o in _observations(records, 'archived-junit')},
        set(_unresolved(records, 'archived-junit')),
    )


class TestPytestDefinitionScopes:
    def test_the_class_path_must_lead_to_the_function(self, tmp_path):
        source = 'def test_mod():\n    pass\n\n\nclass TestX:\n    def test_meth(self):\n        pass\n'
        resolved, unresolved = _resolve(tmp_path, {TEST_D: source}, [
            ('tests.test_d', 'test_mod'), ('tests.test_d.TestX', 'test_meth'),
            ('tests.test_d.TestX', 'test_mod'), ('tests.test_d', 'test_meth'),
        ])
        assert resolved == {f'{TEST_D}::test_mod', f'{TEST_D}::TestX::test_meth'}
        assert unresolved == {'tests.test_d.TestX::test_mod', 'tests.test_d::test_meth'}

    def test_a_nested_class_resolves_only_through_its_full_class_path(self, tmp_path):
        source = 'class TestOuter:\n    class TestInner:\n        def test_n(self):\n            pass\n'
        resolved, unresolved = _resolve(tmp_path, {TEST_D: source}, [
            ('tests.test_d.TestOuter.TestInner', 'test_n'), ('tests.test_d.TestOuter', 'test_n'),
        ])
        assert resolved == {f'{TEST_D}::TestOuter::TestInner::test_n'}
        assert unresolved == {'tests.test_d.TestOuter::test_n'}

    def test_an_async_function_resolves(self, tmp_path):
        source = 'async def test_a():\n    pass\n'
        resolved, unresolved = _resolve(tmp_path, {TEST_D: source}, [('tests.test_d', 'test_a')])
        assert (resolved, unresolved) == ({f'{TEST_D}::test_a'}, set())

    def test_an_unparseable_file_defines_nothing(self, tmp_path):
        resolved, unresolved = _resolve(
            tmp_path, {TEST_D: 'def test_ok(:\n'}, [('tests.test_d', 'test_ok')],
        )
        assert (resolved, unresolved) == (set(), {'tests.test_d::test_ok'})

    def test_a_module_file_lacking_the_function_yields_to_the_repo_root(self, tmp_path):
        records = _junit_records(tmp_path, {
            TEST_D: 'def test_other():\n    pass\n',
            'tests/test_d.py': 'def test_root_only():\n    pass\n',
        }, [('tests.test_d', 'test_root_only')])
        assert [o.test for o in _observations(records, 'archived-junit')] == [
            oc.TestId('tests', 'tests/test_d.py::test_root_only'),
        ]

    def test_a_module_log_failure_credits_the_first_base_whose_file_defines_it(self, tmp_path):
        git_tree(tmp_path, {
            'orchestrator/orchestrator.yaml': '',
            TEST_D: 'def test_both():\n    pass\n',
            'tests/test_d.py': 'def test_both():\n    pass\n\n\ndef test_root_only():\n    pass\n',
        })
        log = (
            tmp_path / 'data' / 'verify-logs' / 'T1'
            / 'attempt-1.orchestrator.test-20261001T000000_1Z.log'
        )
        log.parent.mkdir(parents=True)
        log.write_text('FAILED tests/test_d.py::test_both\nFAILED tests/test_d.py::test_root_only\n')
        records = list(ev.pytest_evidence(tmp_path, tmp_path).records())
        assert [o.test for o in _observations(records, 'pytest-logs')] == [
            oc.TestId('orchestrator', f'{TEST_D}::test_both'),
            oc.TestId('tests', 'tests/test_d.py::test_root_only'),
        ]

    def test_an_inherited_method_resolves_through_in_file_and_unknown_bases(self, tmp_path):
        source = (
            'from contracts import Contract\n\n\n'
            'class _Harness:\n    def test_h(self):\n        pass\n\n\n'
            'class TestChild(_Harness):\n    pass\n\n\n'
            'class TestGrandchild(TestChild):\n    pass\n\n\n'
            'class TestSplice(Contract):\n    pass\n\n\n'
            'class TestLoopA(TestLoopB):\n    pass\n\n\n'
            'class TestLoopB(TestLoopA):\n    pass\n\n\n'
            'class TestOuter:\n    class TestInner(_Harness):\n        pass\n'
        )
        resolved, unresolved = _resolve(tmp_path, {TEST_D: source}, [
            ('tests.test_d.TestChild', 'test_h'),
            ('tests.test_d.TestGrandchild', 'test_h'),
            ('tests.test_d.TestSplice', 'test_contract'),
            ('tests.test_d.TestChild', 'test_absent'),
            ('tests.test_d.TestLoopA', 'test_h'),
            ('tests.test_d.TestOuter.TestInner', 'test_h'),
            ('tests.test_d.TestOuter.TestInner', 'test_absent'),
        ])
        assert resolved == {
            f'{TEST_D}::TestChild::test_h', f'{TEST_D}::TestGrandchild::test_h',
            f'{TEST_D}::TestSplice::test_contract', f'{TEST_D}::TestOuter::TestInner::test_h',
        }
        assert unresolved == {
            'tests.test_d.TestChild::test_absent', 'tests.test_d.TestLoopA::test_h',
            'tests.test_d.TestOuter.TestInner::test_absent',
        }


ALPHA = oc.TestId('tests/infra', 'tests/infra/test_alpha.sh')
BETA = oc.TestId('tests/infra', 'tests/infra/test_beta.sh')
LOG_9 = 'data/verify-logs/9/attempt-1.test-20260920T101010_5Z.log'


class TestNextestEvidence:
    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        return nextest_project(tmp_path)

    @pytest.fixture
    def records(self, root: Path) -> list[oc.Record]:
        return list(ev.nextest_evidence(root, root).records())

    def test_a_debug_and_release_pass_sum_into_one_run_cost(self, root, records):
        case_one = oc.TestId('reify-compiler', 'reify-compiler::harness_types mod_a::case_one')
        logs = _observations(records, 'nextest-logs')
        assert [(o.run, o.outcome, o.seconds) for o in logs if o.test == case_one] == [
            (LOG_9, PASSED, 1.348), (LOG_9, PASSED, 0.652),
        ]
        census = oc.census_outcomes(ev.nextest_evidence(root, root))
        (cost,) = [cost for cost in census.ranking if cost.test == case_one]
        assert (cost.runs, cost.total_s) == (1, pytest.approx(2.0))

    def test_status_vocabulary(self, records):
        logs = _observations(records, 'nextest-logs')
        assert _outcomes(logs, 'reify-eval mod::leaky') == [(PASSED, 0.1)]
        assert _outcomes(logs, 'reify-eval::e2e t_fail') == [(FAILED, 2.0)]
        assert _outcomes(logs, 'reify-eval::solve x') == [(FAILED, 1200.048)]
        assert [raw for raw in _unresolved(records, 'nextest-logs') if 'WEIRD' in raw]

    def test_run_all_failures_count_once_per_log(self, records):
        logs = _observations(records, 'nextest-logs')
        assert [(o.run, o.outcome) for o in logs if o.test == ALPHA] == [(LOG_9, FAILED)]
        assert _unresolved(records, 'nextest-logs').count('test_gamma.sh') == 1

    def test_ledger_entries_fail_their_run(self, records):
        ledger = _observations(records, 'flaky-ledger')
        assert [(o.test, o.run, o.outcome) for o in ledger] == [
            (BETA, 'run-1', FAILED), (BETA, 'run-2', FAILED),
        ]
        assert _unresolved(records, 'flaky-ledger') == ['not json at all']

    def test_an_unknown_flake_row_is_unresolved(self, records):
        assert _observations(records, 'flake_occurrence') == []
        assert _unresolved(records, 'flake_occurrence') == ['<unknown>']

    def test_every_run_is_contiguous(self, records):
        keys = [(o.source, o.run) for o in records if isinstance(o, oc.Observation)]
        starts = [key for i, key in enumerate(keys) if i == 0 or keys[i - 1] != key]
        assert len(starts) == len(set(starts))

    def test_windows(self, root):
        windows = {w.source: w for w in ev.nextest_evidence(root, root).windows}
        assert list(windows) == ['nextest-logs', 'flaky-ledger', 'flake_occurrence']
        observed = {
            source: (w.present, w.artefacts, w.first, w.last) for source, w in windows.items()
        }
        assert observed == {
            'nextest-logs': (True, 2, '2026-09-20T10:10:10Z', '2026-10-01T00:00:00Z'),
            'flaky-ledger': (True, 3, '2026-09-05T01:02:03Z', '2026-09-06T01:02:03Z'),
            'flake_occurrence': (True, 1, '2026-09-07T00:00:00Z', '2026-09-07T00:00:00Z'),
        }

    def test_uncosted_universe_is_the_tracked_infra_tests(self, root):
        assert ev.nextest_evidence(root, root).uncosted_universe == frozenset({ALPHA, BETA})

    def test_an_absent_ledger_is_a_window_not_an_error(self, root):
        (root / 'data' / 'verify-logs' / 'flaky-ledger.jsonl').unlink()
        evidence = ev.nextest_evidence(root, root)
        (ledger,) = [w for w in evidence.windows if w.source == 'flaky-ledger']
        assert (ledger.present, ledger.artefacts) == (False, 0)
        assert _observations(list(evidence.records()), 'flaky-ledger') == []


LANE_FILE = 'orchestrator/tests/test_lane.py'
LANE_TESTS = ('test_gate', 'test_reproduced', 'test_cleared', 'test_inconclusive', 'test_steady')
LANE_PASS_AT = '2026-10-03T00:00:00+00:00'


def _lane(name: str) -> oc.TestId:
    return oc.TestId('orchestrator', f'{LANE_FILE}::{name}')


def _lane_tree(root: Path) -> Path:
    """Five tests costed by one junit run, and a merge-gate flake row for test_gate."""
    git_tree(root, {
        'orchestrator/orchestrator.yaml': '',
        LANE_FILE: ''.join(f'def {name}():\n    pass\n\n\n' for name in LANE_TESTS),
    })
    write_gz(
        root / 'data' / 'verify-logs' / 'T1'
        / 'attempt-1.orchestrator.junit-20261001T000000_1Z.xml.gz',
        junit_xml(
            '2026-10-01T00:00:00+00:00',
            [('tests.test_lane', name, 1.0, 'pass') for name in LANE_TESTS],
        ),
    )
    flake_db(runs_db(root), [
        ('2026-10-02T00:00:00+00:00', 'tests/test_lane.py::test_gate', 'fails_in_isolation'),
    ])
    return root


def _add_lane_rows(root: Path) -> None:
    """One offline-lane pass's suppressions: one test per verdict, and one naming no test."""
    flake_db(runs_db(root), [
        (LANE_PASS_AT, 'tests/test_lane.py::test_reproduced', 'fails_in_isolation'),
        (LANE_PASS_AT, 'tests/test_lane.py::test_cleared', 'passes_in_isolation'),
        (LANE_PASS_AT, 'tests/test_lane.py::test_inconclusive', 'unconfirmable'),
        (LANE_PASS_AT, None, 'unconfirmable'),
    ], call_site=FlakeCallSite.offline_lane)


def _flake_tally(census: oc.OutcomeCensus) -> oc.SourceTally:
    (tally,) = [tally for tally in census.sources if tally.source == 'flake_occurrence']
    return tally


def _flake_artefacts(census: oc.OutcomeCensus) -> int:
    (window,) = [window for window in census.windows if window.source == 'flake_occurrence']
    return window.artefacts


class TestFlakeOccurrenceCallSites:
    """Every flake_occurrence row fails the test it names, whatever its call site or verdict."""

    @pytest.fixture
    def root(self, tmp_path: Path) -> Path:
        return _lane_tree(tmp_path)

    @staticmethod
    def _census(root: Path) -> oc.OutcomeCensus:
        return oc.census_outcomes(ev.pytest_evidence(root, root))

    def test_offline_lane_rows_fail_the_tests_they_name(self, root):
        before = self._census(root)
        _add_lane_rows(root)
        after = self._census(root)
        assert before.failed_floor == {_lane('test_gate')}
        assert after.failed_floor - before.failed_floor == {
            _lane('test_reproduced'), _lane('test_cleared'), _lane('test_inconclusive'),
        }
        assert before.failed_floor <= after.failed_floor
        assert [cost.test for cost in after.ranking] == [
            cost.test for cost in before.ranking if cost.test not in after.failed_floor
        ] == [_lane('test_steady')]

    def test_offline_lane_rows_enter_the_flake_tally_and_window(self, root):
        before = self._census(root)
        _add_lane_rows(root)
        after = self._census(root)
        tallies = [
            (tally.runs, tally.red_runs, tally.unresolved)
            for tally in (_flake_tally(before), _flake_tally(after))
        ]
        assert tallies == [(1, 1, 0), (4, 4, 1)]
        assert _flake_tally(after).unresolved_samples == (UNKNOWN_TEST_ID,)
        assert (_flake_artefacts(before), _flake_artefacts(after)) == (1, 5)

    def test_a_lane_observation_naming_no_test_credits_no_test(self, root):
        _add_lane_rows(root)
        records = list(ev.pytest_evidence(root, root).records())
        assert _unresolved(records, 'flake_occurrence') == [UNKNOWN_TEST_ID]
        assert not [
            o for o in _observations(records, 'flake_occurrence')
            if UNKNOWN_TEST_ID in o.test.name
        ]

    def test_nextest_counts_an_offline_lane_row_of_an_infra_script(self, tmp_path):
        root = git_tree(tmp_path, {
            'tests/infra/test_lane.sh': '#!/bin/sh\n',
            'tests/infra/test_steady.sh': '#!/bin/sh\n',
        })
        flake_db(runs_db(root), [
            (LANE_PASS_AT, 'test_lane.sh', 'fails_in_isolation'),
            (LANE_PASS_AT, None, 'unconfirmable'),
        ], call_site=FlakeCallSite.offline_lane)
        census = oc.census_outcomes(ev.nextest_evidence(root, root))
        assert census.failed_floor == {oc.TestId('tests/infra', 'tests/infra/test_lane.sh')}
        assert census.never_failed_uncosted == (
            oc.TestId('tests/infra', 'tests/infra/test_steady.sh'),
        )
        assert _flake_tally(census).unresolved_samples == (UNKNOWN_TEST_ID,)
