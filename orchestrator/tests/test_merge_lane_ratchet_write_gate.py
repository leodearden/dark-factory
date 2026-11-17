"""Unit tests of the merge-lane ratchet's write gate, over synthetic reports only.

The baseline ``write_baseline`` renders and reads, its refusal of an unauthorized
raise, the authorized-raise ledger it appends to, and the comparator that both
the write gate and ``--check`` run. The live gate, the measurement stratum, the
derived totals, the stored enumeration block and the command line are in
``test_merge_lane_ratchet.py``; the commit-time gate is in
``test_merge_lane_ratchet_commit_delta.py`` and ``test_merge_lane_ratchet_commit_gate.py``.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest
from _merge_lane_ratchet_fixtures import (
    GIT_OPS_RELPATH,
    WORKER_RELPATH,
    blank_file_entry,
    corrupt_ledger,
    lane_cluster,
    ledger_files,
    ratchet,
    source_measures,
    synthetic_report,
)

# ---------------------------------------------------------------------------
# The seed baseline the write-gate, authorized-raise and comparator tests share:
# each one mutates a deep copy of it.

_VERIFIER_LOOP = f'{WORKER_RELPATH}::SpeculativeMergeWorker::_verifier_loop'
_TEST_FILE = 'orchestrator/tests/test_merge_queue.py'


def _ratchet_baseline() -> dict:
    """A miniature but REAL-SHAPED baseline: real paths, real measured numbers."""
    return {
        'schema_version': 1,
        'params': {
            'complexipy_version': '6.2.0',
            'cluster_paths': list(lane_cluster.CLUSTER_PATHS),
            'file_line_ceiling': lane_cluster.FILE_LINE_CEILING,
            'new_function_cognitive_ceiling': (
                lane_cluster.NEW_FUNCTION_COGNITIVE_CEILING
            ),
        },
        # REAL-SHAPED means the STORED shape, which has no `test_tree` key: the
        # test-tree counts are live-only, and this fixture stands in for a
        # committed baseline that the comparator reads.
        'enumeration': {
            'requested': [WORKER_RELPATH, GIT_OPS_RELPATH],
            'resolved': [WORKER_RELPATH, GIT_OPS_RELPATH],
            'unreadable': [],
            'complete': True,
        },
        'files': {
            WORKER_RELPATH: {
                'lines': 21550,
                'prose_lines': 9000,
                'cognitive': 2133,
                'function_local_imports': 12,
                'reexport_names': 9,
            },
            GIT_OPS_RELPATH: {
                'lines': 14721,
                'prose_lines': 6000,
                'cognitive': 1361,
                'function_local_imports': 5,
                'reexport_names': 2,
            },
        },
        'functions': {_VERIFIER_LOOP: 245, f'{GIT_OPS_RELPATH}::GitOps::advance_main': 108},
        'tests': {
            _TEST_FILE: {
                'patch_targets': ['run_scoped_verification', 'time'],
                'private_reads': 300,
            },
        },
    }


def _grow_worker_size(current: dict) -> None:
    """The real task-5342 size shape: worker.py 103 lines and 50 prose lines longer."""
    current['files'][WORKER_RELPATH]['lines'] += 103
    current['files'][WORKER_RELPATH]['prose_lines'] += 50


class TestRenderBaseline:
    def test_round_trips_without_dropping_a_measure(self) -> None:
        # A hand-rolled writer's failure mode is a silently omitted measure, so
        # the round-trip is asserted structurally rather than eyeballed.
        report = synthetic_report()
        loaded = json.loads(ratchet.render_baseline(report))
        assert loaded.pop('_README') == ratchet.BASELINE_README
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
        # indent=2) would spread one file's five measures over six lines
        # and put two branches' unrelated edits inside one conflicting hunk.
        report = synthetic_report()
        rendered = ratchet.render_baseline(report)
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
        report = synthetic_report()
        rendered = ratchet.render_baseline(report)
        for section in ('files', 'functions', 'tests'):
            positions = [
                rendered.index(json.dumps(key) + ':') for key in sorted(report[section])
            ]
            assert positions == sorted(positions), section

    def test_ends_with_exactly_one_trailing_newline(self) -> None:
        rendered = ratchet.render_baseline(synthetic_report())
        assert rendered.endswith('\n')
        assert not rendered.endswith('\n\n')

    def test_rendering_is_idempotent(self) -> None:
        # Regenerating a baseline from a baseline must be a no-op, or every
        # regeneration would churn the file and manufacture conflicts.
        once = ratchet.render_baseline(synthetic_report())
        twice = ratchet.render_baseline(json.loads(once))
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
        rendered = ratchet.render_baseline(synthetic_report())
        first_key_line = rendered.splitlines()[1]
        assert first_key_line.lstrip().startswith('"_README":')
        # Non-vacuity only, in the ratchet suites' floor idiom: the key must not be
        # emitted empty. No wording is constrained.
        assert len(ratchet.BASELINE_README) >= 100


class TestBaselineIO:
    def test_write_then_load_round_trips(self, tmp_path: Path) -> None:
        report = synthetic_report()
        target = tmp_path / 'baseline.json'
        ratchet.write_baseline(target, report)
        assert target.read_text(encoding='utf-8') == ratchet.render_baseline(report)
        loaded = ratchet.load_baseline(target)
        assert loaded['files'] == report['files']
        assert loaded['functions'] == report['functions']

    def test_write_is_atomic_leaving_no_debris(self, tmp_path: Path) -> None:
        target = tmp_path / 'baseline.json'
        ratchet.write_baseline(target, synthetic_report())
        assert [p.name for p in tmp_path.iterdir()] == ['baseline.json']

    def test_missing_baseline_is_a_named_hard_failure(self, tmp_path: Path) -> None:
        # INV-11: a missing baseline is never an empty-baseline PASS, which
        # would silently disarm the ratchet for every downstream task.
        missing = tmp_path / 'nope.json'
        with pytest.raises(ratchet.RatchetError) as excinfo:
            ratchet.load_baseline(missing)
        assert 'nope.json' in str(excinfo.value)

    def test_malformed_baseline_is_a_named_hard_failure(self, tmp_path: Path) -> None:
        target = tmp_path / 'baseline.json'
        target.write_text('{"files": ', encoding='utf-8')
        with pytest.raises(ratchet.RatchetError) as excinfo:
            ratchet.load_baseline(target)
        assert 'baseline.json' in str(excinfo.value)

    def test_non_object_baseline_is_a_named_hard_failure(self, tmp_path: Path) -> None:
        target = tmp_path / 'baseline.json'
        target.write_text('[]', encoding='utf-8')
        with pytest.raises(ratchet.RatchetError) as excinfo:
            ratchet.load_baseline(target)
        assert 'baseline.json' in str(excinfo.value)


class TestWriteBaselineRefusesAnUnauthorizedRaise:
    """The write path is where the ratchet is actually enforceable.

    ``test_merge_lane_ratchet.py::test_baseline_matches_a_fresh_measurement`` forces
    the committed baseline to equal the tree in every committed state, so the
    COMPARATOR is trivially clean on every commit: a raise is red only in the
    window between editing the code and regenerating. Regeneration then absorbs it,
    because ``--write-baseline`` reads nothing before it overwrites. So the only
    place a gate can bite is the write itself, and that is what these pin.

    Synthetic dicts throughout, driven through the public ``write_baseline``:
    microseconds, and every branch reached directly rather than only through a
    71-second measurement.
    """

    _AUTHORIZATION = ratchet.RaiseAuthorization(
        task_id='5406', reason='net-additive: a new guard, not a failed refactor'
    )

    @staticmethod
    def _seeded(tmp_path: Path) -> Path:
        """A destination already holding the seed baseline, byte-for-byte."""
        target = tmp_path / 'b.json'
        ratchet.write_baseline(target, _ratchet_baseline())
        return target

    def test_a_per_file_rise_is_refused_naming_the_measure_and_both_numbers(
        self, tmp_path: Path
    ) -> None:
        target = self._seeded(tmp_path)
        current = copy.deepcopy(_ratchet_baseline())
        current['files'][WORKER_RELPATH]['cognitive'] += 15

        with pytest.raises(ratchet.UnauthorizedRaise) as excinfo:
            ratchet.write_baseline(target, current)

        message = str(excinfo.value)
        assert 'cognitive' in message
        assert WORKER_RELPATH in message
        assert '2133' in message and '2148' in message
        # The refusal must name the sanctioned path, not merely forbid the move.
        assert '--authorize-raise' in message

    def test_the_refusal_carries_the_measured_violations(
        self, tmp_path: Path
    ) -> None:
        target = self._seeded(tmp_path)
        current = copy.deepcopy(_ratchet_baseline())
        current['files'][WORKER_RELPATH]['cognitive'] += 15

        with pytest.raises(ratchet.UnauthorizedRaise) as excinfo:
            ratchet.write_baseline(target, current)

        violations = excinfo.value.violations
        assert isinstance(violations, tuple)
        assert violations
        assert all(isinstance(v, ratchet.Violation) for v in violations)
        # Every one is a genuine rise: the refusal reports what it measured, so
        # the ledger record derived from it cannot disagree with what landed.
        assert all(v.current > v.baseline for v in violations)

    def test_a_refused_write_leaves_the_destination_byte_for_byte(
        self, tmp_path: Path
    ) -> None:
        target = self._seeded(tmp_path)
        before = target.read_text(encoding='utf-8')
        current = copy.deepcopy(_ratchet_baseline())
        current['files'][WORKER_RELPATH]['cognitive'] += 15

        with pytest.raises(ratchet.UnauthorizedRaise):
            ratchet.write_baseline(target, current)

        # A truncated or half-written baseline would be a WIDENED ratchet, the
        # one failure mode this instrument must never produce.
        assert target.read_text(encoding='utf-8') == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ['b.json']

    @pytest.mark.parametrize(
        ('mutate', 'measure'),
        [
            pytest.param(
                lambda cur: cur['files'][WORKER_RELPATH].__setitem__(
                    'cognitive', cur['files'][WORKER_RELPATH]['cognitive'] + 1
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

        with pytest.raises(ratchet.UnauthorizedRaise) as excinfo:
            ratchet.write_baseline(target, current)

        assert measure in {v.measure for v in excinfo.value.violations}

    def test_rewriting_an_identical_report_is_not_a_raise(
        self, tmp_path: Path
    ) -> None:
        # ANTI-VACUITY: a gate that refused everything would pass every test
        # above while making the sanctioned workflow impossible.
        target = self._seeded(tmp_path)
        written = ratchet.write_baseline(target, _ratchet_baseline())
        assert written.path == target
        assert written.record is None

    def test_lowering_a_measure_stays_frictionless(self, tmp_path: Path) -> None:
        # Lowering is the POINT. It must cost nothing extra.
        target = self._seeded(tmp_path)
        current = copy.deepcopy(_ratchet_baseline())
        current['files'][WORKER_RELPATH]['cognitive'] -= 400

        written = ratchet.write_baseline(target, current)
        assert written.path == target
        assert written.record is None
        assert target.read_text(encoding='utf-8') == ratchet.render_baseline(current)

    def test_a_size_only_rise_is_written_without_authorization(
        self, tmp_path: Path
    ) -> None:
        target = self._seeded(tmp_path)
        ledger = tmp_path / 'raises'
        current = copy.deepcopy(_ratchet_baseline())
        _grow_worker_size(current)
        current['files'][self._NEW_PATH] = blank_file_entry(lines=1400, prose_lines=600)

        written = ratchet.write_baseline(target, current, ledger=ledger)

        assert written.record is None
        assert target.read_text(encoding='utf-8') == ratchet.render_baseline(current)
        assert not ledger.exists()

    def test_a_recorded_raise_authorizes_no_later_one(self, tmp_path: Path) -> None:
        # THE BYPASS THE DESIGN EXISTS TO CLOSE: pre-write a record, then
        # regenerate. Authorization is the ACT of passing the flags, never a
        # standing entry, so a landed record -- even one for this very measure
        # -- licenses nothing. Every other refusal test seeds no ledger at all,
        # which leaves this branch green whether the gate consults the file or
        # not.
        target = self._seeded(tmp_path)
        ledger = tmp_path / 'raises'
        first = copy.deepcopy(_ratchet_baseline())
        first['files'][WORKER_RELPATH]['cognitive'] += 15
        ratchet.write_baseline(
            target, first, authorization=self._AUTHORIZATION, ledger=ledger
        )
        recorded = ledger_files(ledger)
        assert ratchet.load_ledger(ledger)

        further = copy.deepcopy(first)
        further['files'][WORKER_RELPATH]['cognitive'] += 5
        with pytest.raises(ratchet.UnauthorizedRaise):
            ratchet.write_baseline(target, further, ledger=ledger)

        assert ledger_files(ledger) == recorded
        assert target.read_text(encoding='utf-8') == ratchet.render_baseline(first)

    def test_a_plain_regeneration_never_reads_the_ledger(
        self, tmp_path: Path
    ) -> None:
        # The other face of the same property: with nothing to record, the
        # ledger is not consulted, so a lowering regeneration cannot be blocked
        # by the health of a file it does not touch. load_ledger fails HARD on
        # a malformed entry, so any read at all would redden this.
        target = self._seeded(tmp_path)
        ledger = corrupt_ledger(tmp_path)
        lowered = copy.deepcopy(_ratchet_baseline())
        lowered['files'][WORKER_RELPATH]['cognitive'] -= 400

        assert ratchet.write_baseline(target, lowered, ledger=ledger).record is None
        assert ledger_files(ledger) == {'corrupt.json': '{ not json at all'}

    def test_unauthorized_raise_is_not_a_metrics_error(self) -> None:
        # The exit ladder: 1 is "a measure rose", 2 is "the instrument broke".
        # main() maps every MetricsError to 2, so a refusal that subclassed it
        # would report a real regression as a broken tool.
        assert not issubclass(ratchet.UnauthorizedRaise, source_measures.MetricsError)

    def test_a_ratchet_fault_is_an_instrument_failure(self) -> None:
        # Each stratum names its own faults, but they are still exit 2: every
        # boundary catches the measure layer's MetricsError for all of them.
        assert issubclass(lane_cluster.ClusterSpecError, source_measures.MetricsError)
        assert issubclass(ratchet.RatchetError, source_measures.MetricsError)
        assert issubclass(ratchet.AppendOnlyViolation, ratchet.RatchetError)

    # CEILINGS ARE A HOLE OF THE SAME SHAPE, and the reason is subtle: the two
    # ceilings apply only to keys ABSENT from the baseline. So a blind
    # regeneration puts the oversized new file INTO the baseline, and every run
    # after that exempts it forever. Regeneration absorbed ceiling breaches
    # exactly as silently as it absorbed rises, which is why the write gate runs
    # the full comparator rather than the four rise arms.

    _NEW_PATH = 'orchestrator/src/orchestrator/merge_new.py'

    def test_a_new_file_over_the_line_ceiling_is_refused(
        self, tmp_path: Path
    ) -> None:
        target = self._seeded(tmp_path)
        current = copy.deepcopy(_ratchet_baseline())
        current['files'][self._NEW_PATH] = blank_file_entry(
            lines=lane_cluster.FILE_LINE_CEILING + 1
        )

        with pytest.raises(ratchet.UnauthorizedRaise) as excinfo:
            ratchet.write_baseline(target, current)

        breaches = [
            v for v in excinfo.value.violations
            if v.measure == 'new_file_over_ceiling'
        ]
        assert [v.key for v in breaches] == [self._NEW_PATH]

    def test_a_new_function_over_the_cognitive_ceiling_is_refused(
        self, tmp_path: Path
    ) -> None:
        target = self._seeded(tmp_path)
        current = copy.deepcopy(_ratchet_baseline())
        current['functions'][f'{WORKER_RELPATH}::brand_new'] = (
            lane_cluster.NEW_FUNCTION_COGNITIVE_CEILING + 1
        )

        with pytest.raises(ratchet.UnauthorizedRaise) as excinfo:
            ratchet.write_baseline(target, current)

        breaches = [
            v for v in excinfo.value.violations
            if v.measure == 'new_function_over_ceiling'
        ]
        assert [v.key for v in breaches] == [f'{WORKER_RELPATH}::brand_new']

    def test_a_new_file_exactly_at_the_line_ceiling_is_written_without_authorization(
        self, tmp_path: Path
    ) -> None:
        # A ceiling is a ceiling, not a floor -- mirrors
        # TestCeilingsApplyOnlyToNewKeys::test_a_new_file_exactly_at_the_line_ceiling_is_allowed.
        target = self._seeded(tmp_path)
        current = copy.deepcopy(_ratchet_baseline())
        current['files'][self._NEW_PATH] = blank_file_entry(
            lines=lane_cluster.FILE_LINE_CEILING
        )

        assert ratchet.write_baseline(target, current).record is None

    def test_the_git_ops_size_exemption_holds_on_the_write_path(
        self, tmp_path: Path
    ) -> None:
        # SIZE_CEILING_EXEMPT is scoped to the CEILING, not to the ratchet, so
        # re-adding git_ops.py as a new key at its real size may still be
        # refused for a totals rise -- but never as a ceiling breach.
        target = self._seeded(tmp_path)
        seed = _ratchet_baseline()
        current = copy.deepcopy(seed)
        del current['files'][GIT_OPS_RELPATH]
        ratchet.write_baseline(target, current)  # dropping it only LOWERS
        restored = copy.deepcopy(seed)

        with pytest.raises(ratchet.UnauthorizedRaise) as excinfo:
            ratchet.write_baseline(target, restored)

        assert not [
            v for v in excinfo.value.violations
            if v.measure == 'new_file_over_ceiling' and v.key == GIT_OPS_RELPATH
        ]

    def test_an_authorized_ceiling_breach_is_recorded(self, tmp_path: Path) -> None:
        # The discriminator pair, completed for ceilings.
        target = self._seeded(tmp_path)
        ledger = tmp_path / 'raises'
        current = copy.deepcopy(_ratchet_baseline())
        current['files'][self._NEW_PATH] = blank_file_entry(
            lines=lane_cluster.FILE_LINE_CEILING + 1
        )

        ratchet.write_baseline(
            target, current, authorization=self._AUTHORIZATION, ledger=ledger
        )

        measures = ratchet.load_ledger(ledger)[0]['measures']
        breach = next(
            row for row in measures if row['measure'] == 'new_file_over_ceiling'
        )
        assert breach['key'] == self._NEW_PATH
        assert breach['baseline'] == lane_cluster.FILE_LINE_CEILING
        assert breach['current'] == lane_cluster.FILE_LINE_CEILING + 1


class TestAuthorizedRaise:
    """The other half of the gate: a raise that WAS authorized lands, recorded.

    The record is what makes the mechanism reviewable rather than an honour
    system. In commit bbfbf1059e a steward hand-wrote the key-by-key delta and a
    separate audit confirming no unrelated drift came along; hand-writing is the
    step that can disagree with what landed, and the audit is the step that gets
    skipped. Deriving the record from the same comparison that would have
    refused the write makes both mechanical.
    """

    @staticmethod
    def _seeded(tmp_path: Path) -> tuple[Path, Path]:
        target = tmp_path / 'b.json'
        ratchet.write_baseline(target, _ratchet_baseline())
        return target, tmp_path / 'raises'

    @staticmethod
    def _raised() -> dict:
        """Several measures up across two sections, as task 5342's raise was."""
        current = copy.deepcopy(_ratchet_baseline())
        current['files'][WORKER_RELPATH]['function_local_imports'] += 2
        current['files'][WORKER_RELPATH]['cognitive'] += 15
        current['functions'][_VERIFIER_LOOP] += 7
        return current

    @staticmethod
    def _authorization() -> ratchet.RaiseAuthorization:
        return ratchet.RaiseAuthorization(
            task_id='5342',
            reason='rename-aware equivalence fix: net-additive bug fix, not a refactor',
        )

    def test_an_authorized_raise_is_written(self, tmp_path: Path) -> None:
        target, ledger = self._seeded(tmp_path)
        current = self._raised()

        written = ratchet.write_baseline(
            target, current, authorization=self._authorization(), ledger=ledger
        )
        assert written.path == target
        assert target.read_text(encoding='utf-8') == ratchet.render_baseline(current)

    def test_exactly_one_ledger_entry_is_appended_verbatim(
        self, tmp_path: Path
    ) -> None:
        target, ledger = self._seeded(tmp_path)
        authorization = self._authorization()

        written = ratchet.write_baseline(
            target, self._raised(), authorization=authorization, ledger=ledger
        )

        entries = ratchet.load_ledger(ledger)
        assert len(entries) == 1
        assert entries[0]['task_id'] == authorization.task_id
        assert entries[0]['reason'] == authorization.reason
        # What the write HANDS BACK is the entry the file gained, not a
        # reconstruction of it -- so what the CLI prints cannot drift from what
        # a reviewer reads in the ledger.
        assert written.record == entries[0]

    def test_the_recorded_measures_are_derived_not_typed(
        self, tmp_path: Path
    ) -> None:
        # THE HEADLINE. An audit record that CAN disagree with the diff it
        # describes is the failure mode this mechanism replaces, so the record
        # is projected off the same Violations the refusal would have carried.
        target, ledger = self._seeded(tmp_path)
        current = self._raised()
        expected = [
            {
                'measure': v.measure,
                'key': v.key,
                'baseline': v.baseline,
                'current': v.current,
            }
            for v in ratchet._measure_raises(current, _ratchet_baseline())
        ]

        ratchet.write_baseline(
            target, current, authorization=self._authorization(), ledger=ledger
        )

        assert ratchet.load_ledger(ledger)[0]['measures'] == expected

    def test_the_derived_totals_travel_with_the_per_path_rises(
        self, tmp_path: Path
    ) -> None:
        # ANTI-VACUITY for the test above, which an empty list would satisfy:
        # a real multi-measure raise carries derived total:* rows, and they are
        # how a reviewer sees that moving mass around did not lower anything.
        target, ledger = self._seeded(tmp_path)
        ratchet.write_baseline(
            target, self._raised(), authorization=self._authorization(), ledger=ledger
        )

        measures = ratchet.load_ledger(ledger)[0]['measures']
        totals = [row for row in measures if row['measure'].startswith('total:')]
        assert {row['measure'] for row in totals} == {
            'total:cognitive', 'total:function_local_imports'
        }
        assert all(row['key'] == ratchet.CLUSTER_TOTAL_KEY for row in totals)
        assert {row['key'] for row in measures} >= {WORKER_RELPATH, _VERIFIER_LOOP}

    def test_a_size_rise_riding_with_an_authorized_raise_is_not_recorded(
        self, tmp_path: Path
    ) -> None:
        target, ledger = self._seeded(tmp_path)
        current = self._raised()
        _grow_worker_size(current)

        ratchet.write_baseline(
            target, current, authorization=self._authorization(), ledger=ledger
        )

        recorded = {row['measure'] for row in ratchet.load_ledger(ledger)[0]['measures']}
        assert not recorded & {'lines', 'prose_lines', 'total:lines', 'total:prose_lines'}
        assert {'cognitive', 'function_local_imports'} <= recorded

    @pytest.mark.parametrize(
        ('task_id', 'reason', 'field'),
        [
            pytest.param('5342', '   ', 'reason', id='blank-reason'),
            pytest.param('5342', '', 'reason', id='empty-reason'),
            pytest.param('  ', 'a real reason', 'task_id', id='blank-task-id'),
            pytest.param('', 'a real reason', 'task_id', id='empty-task-id'),
        ],
    )
    def test_an_unattributed_authorization_cannot_be_constructed(
        self, task_id: str, reason: str, field: str
    ) -> None:
        # AT CONSTRUCTION, not at the point the record is rendered. Enforced
        # further down, the same value would be valid or invalid depending on
        # whether the report it accompanied happened to raise anything -- a
        # blank --reason over a lowering run would pass unremarked, and the
        # ledger's whole purpose is to answer WHO and WHY (heuristic 10).
        with pytest.raises(ratchet.RatchetError) as excinfo:
            ratchet.RaiseAuthorization(task_id=task_id, reason=reason)

        assert field in str(excinfo.value)

    def test_a_rejected_authorization_moves_neither_file(
        self, tmp_path: Path
    ) -> None:
        target, ledger = self._seeded(tmp_path)
        before = target.read_text(encoding='utf-8')

        with pytest.raises(ratchet.RatchetError):
            ratchet.write_baseline(
                target,
                self._raised(),
                authorization=ratchet.RaiseAuthorization(task_id='5342', reason='  '),
                ledger=ledger,
            )

        # A rejected authorization is not a partial one: refusing at
        # construction means the write never starts.
        assert target.read_text(encoding='utf-8') == before
        assert not ledger.exists()

    def test_an_authorization_over_a_non_raising_report_records_nothing(
        self, tmp_path: Path
    ) -> None:
        # The ledger records RAISES, not intentions. A routine lowering
        # regeneration that happens to carry the flag must not manufacture a
        # record -- a record nobody could point at a diff is noise that makes
        # the real ones harder to read.
        target, ledger = self._seeded(tmp_path)
        lowered = copy.deepcopy(_ratchet_baseline())
        lowered['files'][WORKER_RELPATH]['cognitive'] -= 400

        written = ratchet.write_baseline(
            target, lowered, authorization=self._authorization(), ledger=ledger
        )
        assert written.path == target
        # The write SAYS it recorded nothing, so the CLI can tell the agent who
        # typed --authorize-raise rather than leaving silence to read as assent.
        assert written.record is None
        assert target.read_text(encoding='utf-8') == ratchet.render_baseline(lowered)
        assert not ledger.exists()

    def test_raise_authorization_is_frozen(self) -> None:
        authorization = self._authorization()
        with pytest.raises(dataclasses.FrozenInstanceError):
            authorization.task_id = '9999'  # type: ignore[misc]


class TestAuthorizedRaiseLedger:
    """The ledger directory's own contract, and the polarity split it makes deliberate.

    One file per authorization, named for its content, so two branches
    authorizing at once add two different files and never conflict -- the tail
    of the retired single array conflicted on every such pair.
    """

    @staticmethod
    def _record(task_id: str, rise: int = 103) -> dict:
        return ratchet.authorization_record(
            ratchet.RaiseAuthorization(task_id=task_id, reason=f'reason {task_id}'),
            [ratchet.Violation.rose('lines', WORKER_RELPATH, 21550, 21550 + rise)],
        )

    def test_absence_is_empty_here_and_fatal_for_the_baseline(
        self, tmp_path: Path
    ) -> None:
        # THE ASYMMETRY, asserted side by side so it reads as deliberate rather
        # than as one of the two having been overlooked. An absent BASELINE would
        # compare clean against every measure -- a silent disarming, so it is a
        # hard failure (INV-11). An absent LEDGER means no raise was ever
        # authorized, which is the fail-CLOSED state: it permits nothing.
        missing = tmp_path / 'nope.json'

        assert ratchet.load_ledger(missing) == []

        with pytest.raises(ratchet.RatchetError) as excinfo:
            ratchet.load_baseline(missing)
        assert 'nope.json' in str(excinfo.value)

    @pytest.mark.parametrize(
        ('name', 'content', 'why'),
        [
            pytest.param('x.json', '{"measures": ', 'invalid JSON', id='truncated'),
            pytest.param('x.json', '[]', 'a JSON array at top level', id='array'),
            pytest.param(
                'x.json', '{"task_id": "5342"}', 'a name not derived from the content',
                id='misnamed',
            ),
        ],
    )
    def test_a_malformed_entry_is_a_named_hard_failure(
        self, tmp_path: Path, name: str, content: str, why: str
    ) -> None:
        # An entry that cannot be read is provenance that cannot be audited.
        # Reading it as absent would hide history, so corruption -- unlike
        # absence -- fails hard and names the file.
        ledger = tmp_path / 'raises'
        ledger.mkdir()
        (ledger / name).write_text(content, encoding='utf-8')
        with pytest.raises(ratchet.RatchetError) as excinfo:
            ratchet.load_ledger(ledger)
        assert name in str(excinfo.value), why

    def test_a_ledger_path_that_is_a_file_is_a_named_hard_failure(
        self, tmp_path: Path
    ) -> None:
        # The retired shape at the new path: a single file is not a ledger.
        target = tmp_path / 'raises'
        target.write_text('{"raises": []}', encoding='utf-8')
        with pytest.raises(ratchet.RatchetError) as excinfo:
            ratchet.load_ledger(target)
        assert 'not a directory' in str(excinfo.value)

    def test_an_entry_round_trips_through_its_file(self) -> None:
        record = self._record('5342')
        text = ratchet.render_entry(record)

        assert ratchet.parse_entry(ratchet.entry_filename(record), text) == record
        assert ratchet.render_entry(json.loads(text)) == text

    def test_the_name_leads_with_the_task_and_ends_with_a_content_digest(self) -> None:
        name = ratchet.entry_filename(self._record('5342'))

        assert name.startswith('5342-') and name.endswith('.json')
        assert name != ratchet.entry_filename(self._record('5342', rise=104))

    def test_a_free_text_task_id_still_makes_a_plain_file_name(self) -> None:
        name = ratchet.entry_filename(self._record('3790 carry/../x'))

        assert '/' not in name and ' ' not in name

    def test_each_authorization_is_its_own_file_and_none_is_rewritten(
        self, tmp_path: Path
    ) -> None:
        # APPEND-ONLY by construction: recording one entry cannot touch another.
        ledger = tmp_path / 'raises'
        first = ratchet.append_authorization(ledger, self._record('5342'))
        before = first.read_text(encoding='utf-8')
        for task_id in ('5500', '5600'):
            ratchet.append_authorization(ledger, self._record(task_id))

        assert first.read_text(encoding='utf-8') == before
        assert sorted(entry['task_id'] for entry in ratchet.load_ledger(ledger)) == [
            '5342', '5500', '5600'
        ]

    def test_recording_the_same_entry_twice_is_one_file(self, tmp_path: Path) -> None:
        ledger = tmp_path / 'raises'
        ratchet.append_authorization(ledger, self._record('5342'))
        ratchet.append_authorization(ledger, self._record('5342'))

        assert len(ratchet.load_ledger(ledger)) == 1

    def test_the_first_append_writes_a_legible_readme(self, tmp_path: Path) -> None:
        ledger = tmp_path / 'raises'
        ratchet.append_authorization(ledger, self._record('5342'))

        readme = (ledger / ratchet.LEDGER_README_NAME).read_text(encoding='utf-8')
        assert readme == ratchet.render_ledger_readme()
        assert '--authorize-raise' in json.loads(readme)['_README']

    def test_the_committed_ledger_is_present_and_legible(self) -> None:
        # ANTI-VACUITY for the artifact a human actually opens. EXISTENCE is
        # asserted on the PATH, not inferred from load_ledger: absence reads as
        # the empty ledger by design (the polarity pinned above), so every
        # other assertion here would pass just as happily with nothing
        # committed at all.
        ledger = ratchet.repo_root() / ratchet.LEDGER_DIR_RELPATH
        assert ledger.is_dir(), ledger
        readme = (ledger / ratchet.LEDGER_README_NAME).read_text(encoding='utf-8')
        assert readme == ratchet.render_ledger_readme()
        # Every entry was written by the module's own writer and never edited:
        # its name is derived from its content, and its bytes are a no-op
        # round trip rather than a hand-edited shape.
        entries = ratchet.load_ledger(ledger)
        assert entries
        for entry in entries:
            path = ledger / ratchet.entry_filename(entry)
            assert path.read_text(encoding='utf-8') == ratchet.render_entry(entry), path
        # The retired single array is gone: one home for one fact.
        assert not (ratchet.repo_root() / ratchet.LEGACY_LEDGER_RELPATH).exists()


# ---------------------------------------------------------------------------
# The ratchet comparator, and INV-10 tier 1.
#
# Every assertion below drives check_against_baseline with SYNTHETIC dicts, so
# each branch is pinned directly rather than only ever reached through a real
# measurement that happens to walk over it. The comparator is a pure function of
# two plain dicts -- no filesystem, no complexipy -- which is what makes the
# seeded-fixture self-test run in microseconds AND lets the CLI's --check reuse
# the identical code path this test asserts on (SPOT: one ratchet, not two).


def _bump_file(measure: str):
    def mutate(current: dict) -> None:
        current['files'][WORKER_RELPATH][measure] += 1

    return mutate


def _add_file(measure: str):
    def mutate(current: dict) -> None:
        # A brand-new path carrying +1 of exactly ONE measure. It is absent from
        # the baseline, so no per-path comparison fires -- only the derived
        # cluster total can catch it, which is precisely the rename-gaming shape.
        current['files']['orchestrator/src/orchestrator/merge_new.py'] = (
            blank_file_entry(**{measure: 1})
        )

    return mutate


#: (test id, mutation, expected violation measure, expected violation key).
#: THE SEEDED-FIXTURE SELF-TEST (INV-10 tier 1): every RATCHETED measure and
#: every derived total of one gets its own +1, and the ratchet must EXECUTE to a
#: named violation for it. A measure missing from this table is a measure
#: nothing proves is actually compared.
_SEEDS = [
    ('files.cognitive', _bump_file('cognitive'), 'cognitive', WORKER_RELPATH),
    (
        'files.function_local_imports',
        _bump_file('function_local_imports'),
        'function_local_imports',
        WORKER_RELPATH,
    ),
    ('files.reexport_names', _bump_file('reexport_names'), 'reexport_names', WORKER_RELPATH),
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
    (
        'total.cognitive',
        _add_file('cognitive'),
        'total:cognitive',
        ratchet.CLUSTER_TOTAL_KEY,
    ),
    (
        'total.function_local_imports',
        _add_file('function_local_imports'),
        'total:function_local_imports',
        ratchet.CLUSTER_TOTAL_KEY,
    ),
    (
        'total.reexport_names',
        _add_file('reexport_names'),
        'total:reexport_names',
        ratchet.CLUSTER_TOTAL_KEY,
    ),
    (
        'total.private_reads',
        lambda cur: cur['tests'].__setitem__(
            'orchestrator/tests/test_brand_new.py',
            {'patch_targets': [], 'private_reads': 1},
        ),
        'total:private_reads',
        ratchet.CLUSTER_TOTAL_KEY,
    ),
    (
        'total.patch_targets',
        lambda cur: cur['tests'].__setitem__(
            'orchestrator/tests/test_brand_new.py',
            {'patch_targets': ['_brand_new_leaf'], 'private_reads': 0},
        ),
        'total:patch_targets',
        ratchet.CLUSTER_TOTAL_KEY,
    ),
]

#: (test id, mutation): a rise in a SIZE measure alone, per path and in the
#: derived total. Size is measured and reported, never ratcheted (PRD
#: plans/review-all-s25-scripts-instruments-prd.md D10).
_SIZE_ONLY_SEEDS = [
    ('files.lines', _bump_file('lines')),
    ('files.prose_lines', _bump_file('prose_lines')),
    ('total.lines', _add_file('lines')),
    ('total.prose_lines', _add_file('prose_lines')),
    ('files.task_5342_shape', _grow_worker_size),
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

        violations = ratchet.check_against_baseline(current, baseline)

        matching = [v for v in violations if v.measure == measure and v.key == key]
        assert len(matching) == 1, f'{measure} at {key} not reported once: {violations}'
        assert measure in matching[0].message
        assert key in matching[0].message
        # The ONLY collateral a per-path rise may produce is the cluster total
        # that mechanically follows from it. Anything else means the comparator
        # is firing on a measure nobody touched.
        others = [v for v in violations if v is not matching[0]]
        assert all(v.measure == f'total:{measure}' for v in others), others

    @pytest.mark.parametrize(
        'mutate', [pytest.param(m, id=name) for name, m in _SIZE_ONLY_SEEDS]
    )
    def test_a_rise_in_a_size_measure_alone_is_no_violation(self, mutate) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        mutate(current)

        assert ratchet.check_against_baseline(current, baseline) == []

    def test_a_structural_rise_beside_a_size_rise_is_reported_alone(self) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        _grow_worker_size(current)
        current['files'][WORKER_RELPATH]['function_local_imports'] += 1

        violations = ratchet.check_against_baseline(current, baseline)

        assert [(v.measure, v.key, v.baseline, v.current) for v in violations] == [
            ('function_local_imports', WORKER_RELPATH, 12, 13),
            ('total:function_local_imports', ratchet.CLUSTER_TOTAL_KEY, 17, 18),
        ]

    def test_an_identical_current_is_clean(self) -> None:
        baseline = _ratchet_baseline()
        assert ratchet.check_against_baseline(copy.deepcopy(baseline), baseline) == []

    def test_equality_is_permitted_this_is_a_ratchet_not_a_gate(self) -> None:
        # 62 lane functions already exceed cognitive 15 on the introducing
        # commit. A day-one gate would be red on arrival and get disabled.
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        assert current['functions'][_VERIFIER_LOOP] == 245
        assert ratchet.check_against_baseline(current, baseline) == []

    def test_lowering_a_measure_is_clean(self) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['files'][WORKER_RELPATH]['lines'] -= 4000
        current['files'][WORKER_RELPATH]['cognitive'] -= 400
        current['functions'][_VERIFIER_LOOP] = 12
        current['tests'][_TEST_FILE]['private_reads'] = 0
        assert ratchet.check_against_baseline(current, baseline) == []


class TestCeilingsApplyOnlyToNewKeys:
    def test_a_new_file_over_the_line_ceiling_is_a_violation(self) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        new_path = 'orchestrator/src/orchestrator/merge_new.py'
        current['files'][new_path] = blank_file_entry(lines=1501)
        ceilings = [
            v
            for v in ratchet.check_against_baseline(current, baseline)
            if v.measure == 'new_file_over_ceiling'
        ]
        assert len(ceilings) == 1
        assert new_path in ceilings[0].message

    def test_a_new_file_exactly_at_the_line_ceiling_is_allowed(self) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['files']['orchestrator/src/orchestrator/merge_new.py'] = (
            blank_file_entry(lines=1500)
        )
        assert not [
            v
            for v in ratchet.check_against_baseline(current, baseline)
            if v.measure == 'new_file_over_ceiling'
        ]

    def test_a_grandfathered_oversized_file_is_not_a_ceiling_violation(self) -> None:
        # 21,550 lines, fourteen times the ceiling, and clean -- because the
        # ceiling applies only to paths ABSENT from the baseline. That is what
        # makes this a ratchet whose grandfathering can only ever shrink.
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        assert current['files'][WORKER_RELPATH]['lines'] == 21550
        assert ratchet.check_against_baseline(current, baseline) == []

    def test_size_ceiling_exemption_covers_git_ops_even_when_new(self) -> None:
        # PRD decision 9. Exercised with git_ops.py ABSENT from the baseline,
        # because that is the only configuration in which the ceiling would
        # otherwise fire -- an exemption never reached is an exemption never
        # tested.
        baseline = _ratchet_baseline()
        del baseline['files'][GIT_OPS_RELPATH]
        current = copy.deepcopy(baseline)
        current['files'][GIT_OPS_RELPATH] = blank_file_entry(lines=14721)
        assert not [
            v
            for v in ratchet.check_against_baseline(current, baseline)
            if v.measure == 'new_file_over_ceiling'
        ]

    def test_a_non_exempt_path_at_the_same_size_is_a_violation(self) -> None:
        # The contrast that proves the previous test measured the EXEMPTION and
        # not merely a comparator that never fires.
        baseline = _ratchet_baseline()
        del baseline['files'][GIT_OPS_RELPATH]
        current = copy.deepcopy(baseline)
        current['files']['orchestrator/src/orchestrator/merge_new.py'] = (
            blank_file_entry(lines=14721)
        )
        assert [
            v
            for v in ratchet.check_against_baseline(current, baseline)
            if v.measure == 'new_file_over_ceiling'
        ]

    def test_the_exemption_is_scoped_to_the_ceiling_not_to_the_ratchet(self) -> None:
        # git_ops.py is exempt from the CEILING, and its structural measures
        # are still frozen by the RATCHET, so they cannot grow unwatched.
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['files'][GIT_OPS_RELPATH]['cognitive'] += 1
        assert [
            v
            for v in ratchet.check_against_baseline(current, baseline)
            if v.measure == 'cognitive' and v.key == GIT_OPS_RELPATH
        ]

    def test_a_new_function_over_the_cognitive_ceiling_is_a_violation(self) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['functions'][f'{WORKER_RELPATH}::brand_new'] = 16
        ceilings = [
            v
            for v in ratchet.check_against_baseline(current, baseline)
            if v.measure == 'new_function_over_ceiling'
        ]
        assert len(ceilings) == 1
        assert 'brand_new' in ceilings[0].message

    def test_a_new_function_exactly_at_the_cognitive_ceiling_is_allowed(self) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['functions'][f'{WORKER_RELPATH}::brand_new'] = 15
        assert ratchet.check_against_baseline(current, baseline) == []

    def test_a_grandfathered_function_at_245_is_clean(self) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['functions'][_VERIFIER_LOOP] = 245
        assert ratchet.check_against_baseline(current, baseline) == []


class TestComparatorPreconditions:
    def test_a_complexipy_version_drift_is_a_hard_failure(self) -> None:
        # Cognitive numbers are version-dependent: the SAME source file
        # measured 2031 at complexipy 3.0.0, 2092 at 5.0.0 and 2133 at 6.x/7.x.
        # A silent version drift would rewrite every number at once and leave a
        # green ratchet comparing two incomparable measurements.
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['params']['complexipy_version'] = '7.0.1'
        with pytest.raises(ratchet.RatchetError) as excinfo:
            ratchet.check_against_baseline(current, baseline)
        assert '7.0.1' in str(excinfo.value)
        assert '6.2.0' in str(excinfo.value)

    def test_a_cluster_path_edit_forces_a_deliberate_regeneration(self) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        baseline['params']['cluster_paths'] = [
            p for p in lane_cluster.CLUSTER_PATHS if p != GIT_OPS_RELPATH
        ] + ['orchestrator/src/orchestrator/ghost.py']
        with pytest.raises(ratchet.RatchetError) as excinfo:
            ratchet.check_against_baseline(current, baseline)
        message = str(excinfo.value)
        assert GIT_OPS_RELPATH in message
        assert 'ghost.py' in message

    def test_an_incomplete_enumeration_is_never_silently_compared(self) -> None:
        # INV-11. A sweep that skipped files measures LOWER than the truth, so
        # comparing it would read as a clean tree -- or worse, as an
        # improvement worth writing into the baseline.
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['enumeration']['complete'] = False
        current['enumeration']['unreadable'] = ['orchestrator/tests/test_broken.py']
        with pytest.raises(ratchet.RatchetError) as excinfo:
            ratchet.check_against_baseline(current, baseline)
        assert 'test_broken.py' in str(excinfo.value)

    def test_preconditions_are_checked_before_any_comparison(self) -> None:
        # Order matters: a wrong-version or partial measurement must fail with
        # its own named cause, not with a wall of downstream violations that
        # sends the reader hunting a regression that does not exist.
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['enumeration']['complete'] = False
        current['files'][WORKER_RELPATH]['cognitive'] += 5000
        with pytest.raises(ratchet.RatchetError):
            ratchet.check_against_baseline(current, baseline)


class TestViolationShape:
    def test_a_violation_carries_both_numbers_and_a_stable_order(self) -> None:
        baseline = _ratchet_baseline()
        current = copy.deepcopy(baseline)
        current['files'][WORKER_RELPATH]['function_local_imports'] += 1
        current['files'][GIT_OPS_RELPATH]['cognitive'] += 1
        violations = ratchet.check_against_baseline(current, baseline)
        per_path = [v for v in violations if v.key in (WORKER_RELPATH, GIT_OPS_RELPATH)]
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
        current['files'][WORKER_RELPATH]['cognitive'] += 1
        before = (json.dumps(current, sort_keys=True), json.dumps(baseline, sort_keys=True))
        ratchet.check_against_baseline(current, baseline)
        after = (json.dumps(current, sort_keys=True), json.dumps(baseline, sort_keys=True))
        assert after == before
