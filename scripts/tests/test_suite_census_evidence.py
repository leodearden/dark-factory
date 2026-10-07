"""The evidence readers turn retained verify artefacts into typed census records."""
from __future__ import annotations

from pathlib import Path

import pytest
import suite_census_evidence as ev
import suite_census_outcomes as oc
from suite_census_fixtures import nextest_project, pytest_project, write_gz

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

    def test_an_ambiguous_flake_id_marks_every_candidate(self, records):
        flakes = _observations(records, 'flake_occurrence')
        assert sorted((o.test.package, o.test.name, o.outcome) for o in flakes) == [
            ('fused-memory', 'fused-memory/tests/test_a.py::test_q', FAILED),
            ('orchestrator', f'{TEST_A}::test_q', FAILED),
        ]
        assert sorted(_unresolved(records, 'flake_occurrence')) == [
            '<unknown>', 'tests/test_deleted.py::test_x',
        ]

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
            'flake_occurrence': (True, 3, '2026-09-01T00:00:00Z', '2026-09-03T00:00:00Z'),
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
