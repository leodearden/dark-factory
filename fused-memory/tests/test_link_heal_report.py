"""Rendering one link-heal sweep run's records as a report (task 7159).

The renderer is pure: a :class:`SweepRunView` built by hand renders to a JSON
document and a Markdown page, and the same view always renders the same bytes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from fused_memory.maintenance.link_heal import (
    BasisSource,
    HealAction,
    Report,
    RunCounts,
    Verdict,
)
from fused_memory.maintenance.link_heal_executor import undo_command
from fused_memory.maintenance.link_heal_ledger import (
    NO_ACTION_TAKEN,
    ActionState,
    CandidateRecord,
    Disposition,
    RecordSnapshot,
)
from fused_memory.maintenance.link_heal_report import (
    CalibrationStatus,
    LedgerRun,
    SweepRunView,
    WritesWithheld,
    render_run_report,
    report_paths,
)
from fused_memory.maintenance.link_heal_store import CreatedWindow

DF = 'dark_factory'
REIFY = 'reify'
PLAN_RUN = 'a1' * 16
APPLY_RUN = 'b2' * 16
CALIBRATION_RUN = 'c3' * 16
LEDGER = Path('/var/lib/fused-memory/link_heal.db')
WINDOW = CreatedWindow(
    since=datetime(2026, 11, 1, 2, 15, tzinfo=UTC), until=datetime(2026, 11, 2, 2, 15, tzinfo=UTC),
)
CALIBRATED = CalibrationStatus(
    model='opus', prompt_sha256='9' * 64, calibration_run_id=CALIBRATION_RUN, passed=True,
)
UNCALIBRATED = CalibrationStatus(
    model='opus', prompt_sha256='9' * 64, calibration_run_id=None, passed=False,
)
MISFILE_SHARE = 'link-heal-misfile-share'
STORM = 'link-heal-adjudicator'


def _snapshot(snapshot_id: int, memory_id: str, parent_id: str | None) -> RecordSnapshot:
    metadata = {} if parent_id is None else {'parent_id': parent_id, 'kind': 'sighting'}
    return RecordSnapshot(
        snapshot_id=snapshot_id,
        project_id=DF,
        memory_id=memory_id,
        content=f'text of {memory_id}',
        content_sha256='0' * 64,
        metadata=metadata,
        metadata_sha256='1' * 64,
        created_at='2026-11-01T12:00:00+00:00',
        updated_at=None,
    )


def _record(
    n: int,
    *,
    project: str,
    decision: HealAction | Report,
    verdict: Verdict | None,
    applied: bool = False,
    disposition: Disposition = Disposition.REPORTED,
) -> CandidateRecord:
    child_id = f'child-{n}'
    before = _snapshot(n * 10, child_id, f'parent-{n}')
    after = _snapshot(n * 10 + 1, child_id, None) if applied else before
    return CandidateRecord(
        run_id=PLAN_RUN,
        project_id=project,
        child_id=child_id,
        parent_id=f'parent-{n}',
        before_child=before,
        before_parent=_snapshot(n * 10 + 2, f'parent-{n}', None),
        basis_source=None if verdict is None else BasisSource.ADJUDICATOR,
        basis_key=None if verdict is None else str(n),
        decision=decision,
        disposition=disposition,
        action_id=n if isinstance(decision, HealAction) else None,
        action_taken=decision.value if applied else NO_ACTION_TAKEN,
        action_state=ActionState.APPLIED if applied else None,
        after_child=after,
        at='2026-11-02T02:16:00+00:00',
        adjudication_id=None if verdict is None else n,
        verdict=verdict,
    )


RECORDS = (
    _record(1, project=DF, decision=HealAction.DETACH, verdict=Verdict.RELATED,
            applied=True, disposition=Disposition.STAGED),
    _record(2, project=DF, decision=HealAction.DETACH, verdict=Verdict.RELATED,
            applied=True, disposition=Disposition.STAGED),
    _record(3, project=DF, decision=HealAction.RELABEL, verdict=Verdict.EXTENDS,
            applied=True, disposition=Disposition.STAGED),
    _record(4, project=DF, decision=Report.NO_ACTION, verdict=Verdict.SAME),
    _record(5, project=REIFY, decision=Report.HAS_CHILDREN, verdict=None),
    _record(6, project=REIFY, decision=HealAction.FLAG, verdict=Verdict.CORRECTS,
            disposition=Disposition.NOT_ENABLED),
)

PLAN_COUNTS = RunCounts(examined=6, planned=3, not_enabled=1, would_escape=(MISFILE_SHARE,))
APPLY_COUNTS = RunCounts(applied=3, escaped=({'anchor': STORM, 'escalation_id': 'esc-7159-9'},))
APPLIED = LedgerRun(run_id=APPLY_RUN, counts=APPLY_COUNTS)


def _view(
    *,
    apply: LedgerRun | None = APPLIED,
    calibration: CalibrationStatus = CALIBRATED,
    withheld: WritesWithheld | None = None,
    records: tuple[CandidateRecord, ...] = RECORDS,
) -> SweepRunView:
    return SweepRunView(
        plan=LedgerRun(run_id=PLAN_RUN, counts=PLAN_COUNTS),
        apply=apply,
        records=records,
        window=WINDOW,
        calibration=calibration,
        withheld=withheld,
        ledger_path=LEDGER,
    )


def _document(view: SweepRunView) -> dict[str, Any]:
    json_text, _ = render_run_report(view)
    return json.loads(json_text)


def _markdown(view: SweepRunView) -> str:
    _, markdown = render_run_report(view)
    return markdown


class TestCounts:
    def test_by_project_counts_each_decision(self):
        counts = _document(_view())['counts']
        assert counts['by_project'] == {
            DF: {'detach': 2, 'relabel': 1, 'no_action': 1},
            REIFY: {'has_children': 1, 'flag': 1},
        }

    def test_by_verdict_counts_an_unadjudicated_record_as_none(self):
        assert _document(_view())['counts']['by_verdict'] == {
            'RELATED': 2, 'EXTENDS': 1, 'SAME': 1, 'CORRECTS': 1, 'none': 1,
        }

    def test_by_action_counts_what_was_taken(self):
        assert _document(_view())['counts']['by_action'] == {
            'detach': 2, 'relabel': 1, NO_ACTION_TAKEN: 3,
        }

    def test_by_action_state_tells_an_applied_heal_from_no_heal(self):
        assert _document(_view())['counts']['by_action_state'] == {
            ActionState.APPLIED.value: 3, NO_ACTION_TAKEN: 3,
        }

    def test_the_cross_tab_is_one_row_per_project_decision_verdict_action_and_state(self):
        cross_tab = _document(_view())['counts']['cross_tab']
        applied = ActionState.APPLIED.value
        assert cross_tab == [
            {'project': DF, 'decision': 'detach', 'verdict': 'RELATED',
             'action_taken': 'detach', 'action_state': applied, 'count': 2},
            {'project': DF, 'decision': 'no_action', 'verdict': 'SAME',
             'action_taken': NO_ACTION_TAKEN, 'action_state': NO_ACTION_TAKEN, 'count': 1},
            {'project': DF, 'decision': 'relabel', 'verdict': 'EXTENDS',
             'action_taken': 'relabel', 'action_state': applied, 'count': 1},
            {'project': REIFY, 'decision': 'flag', 'verdict': 'CORRECTS',
             'action_taken': NO_ACTION_TAKEN, 'action_state': NO_ACTION_TAKEN, 'count': 1},
            {'project': REIFY, 'decision': 'has_children', 'verdict': 'none',
             'action_taken': NO_ACTION_TAKEN, 'action_state': NO_ACTION_TAKEN, 'count': 1},
        ]

    def test_each_run_discloses_its_own_counts(self):
        runs = _document(_view())['runs']
        assert runs['plan'] == {'run_id': PLAN_RUN, 'counts': PLAN_COUNTS.as_json()}
        assert runs['apply'] == {'run_id': APPLY_RUN, 'counts': APPLY_COUNTS.as_json()}

    def test_a_run_with_no_apply_discloses_none(self):
        view = _view(apply=None, withheld=WritesWithheld.WRITES_OFF)
        assert _document(view)['runs']['apply'] is None


class TestEscapes:
    def test_filed_and_would_escape_are_both_disclosed_with_their_run(self):
        escapes = _document(_view())['escapes']
        assert escapes['filed'] == [
            {'run_id': APPLY_RUN, 'anchor': STORM, 'escalation_id': 'esc-7159-9'},
        ]
        assert escapes['would_escape'] == [{'run_id': PLAN_RUN, 'anchor': MISFILE_SHARE}]

    def test_a_run_with_no_escapes_discloses_empty_lists(self):
        view = SweepRunView(
            plan=LedgerRun(run_id=PLAN_RUN, counts=RunCounts()),
            apply=None,
            records=(),
            window=WINDOW,
            calibration=UNCALIBRATED,
            withheld=WritesWithheld.DRY_RUN,
            ledger_path=LEDGER,
        )
        assert _document(view)['escapes'] == {'filed': [], 'would_escape': []}


class TestUndo:
    def test_undo_names_the_apply_run_when_a_heal_applied(self):
        assert _document(_view())['undo_run_id'] == APPLY_RUN

    def test_undo_is_null_when_the_apply_run_applied_nothing(self):
        view = _view(apply=LedgerRun(run_id=APPLY_RUN, counts=RunCounts(stopped_by='share')))
        assert _document(view)['undo_run_id'] is None

    def test_undo_is_null_without_an_apply_run(self):
        view = _view(apply=None, withheld=WritesWithheld.UNCALIBRATED, calibration=UNCALIBRATED)
        assert _document(view)['undo_run_id'] is None


class TestIdentity:
    def test_the_window_is_iso_strings(self):
        assert _document(_view())['window'] == {
            'since': '2026-11-01T02:15:00+00:00', 'until': '2026-11-02T02:15:00+00:00',
        }

    def test_the_calibration_status_is_present(self):
        assert _document(_view())['calibration'] == {
            'model': 'opus',
            'prompt_sha256': '9' * 64,
            'calibration_run_id': CALIBRATION_RUN,
            'passed': True,
        }

    def test_the_writes_disclose_mode_and_reason(self):
        assert _document(_view())['writes'] == {'mode': 'applied', 'withheld': None}
        withheld = _view(apply=None, withheld=WritesWithheld.UNCALIBRATED, calibration=UNCALIBRATED)
        assert _document(withheld)['writes'] == {'mode': 'withheld', 'withheld': 'uncalibrated'}
        dry = _view(apply=None, withheld=WritesWithheld.DRY_RUN)
        assert _document(dry)['writes'] == {'mode': 'dry_run', 'withheld': 'dry_run'}

    def test_the_records_ref_names_the_ledger_and_the_plan_run(self):
        assert _document(_view())['records_ref'] == {
            'ledger': str(LEDGER), 'plan_run_id': PLAN_RUN,
        }


class TestMarkdown:
    def test_one_table_per_project(self):
        markdown = _markdown(_view())
        assert f'## {DF}' in markdown
        assert f'## {REIFY}' in markdown
        assert markdown.count('| decision | verdict | action | action state | count |') == 2
        assert '| detach | RELATED | detach | applied | 2 |' in markdown
        assert '| has_children | none | none | none | 1 |' in markdown

    def test_the_writes_line_says_applied(self):
        assert '- Writes: applied' in _markdown(_view())

    def test_the_writes_line_says_withheld_with_its_reason(self):
        view = _view(apply=None, withheld=WritesWithheld.UNCALIBRATED, calibration=UNCALIBRATED)
        assert '- Writes: withheld (uncalibrated)' in _markdown(view)

    @pytest.mark.parametrize('withheld', [WritesWithheld.WRITES_OFF, WritesWithheld.DRY_RUN])
    def test_the_writes_line_says_dry_run(self, withheld):
        assert '- Writes: dry-run' in _markdown(_view(apply=None, withheld=withheld))

    def test_the_undo_command_is_given_when_undo_is_set(self):
        assert f'link_heal.py undo --run {APPLY_RUN}' in _markdown(_view())

    def test_the_undo_command_is_the_one_operators_are_shown(self):
        view = _view()
        assert view.undo_run_id is not None
        assert undo_command(view.undo_run_id) in _markdown(view)

    def test_no_undo_command_when_nothing_applied(self):
        view = _view(apply=None, withheld=WritesWithheld.DRY_RUN)
        assert 'link_heal.py undo' not in _markdown(view)

    def test_a_run_that_examined_nothing_says_so(self):
        view = _view(apply=None, withheld=WritesWithheld.DRY_RUN, records=())
        assert 'No links were examined.' in _markdown(view)


def test_the_same_view_renders_the_same_bytes():
    assert render_run_report(_view()) == render_run_report(_view())


class TestViewInvariants:
    def test_a_view_without_an_apply_run_says_why(self):
        with pytest.raises(ValueError, match='withheld'):
            _view(apply=None, withheld=None)

    def test_a_view_with_an_apply_run_withheld_nothing(self):
        with pytest.raises(ValueError, match='withheld'):
            _view(withheld=WritesWithheld.DRY_RUN)

    def test_a_passed_calibration_names_its_run(self):
        with pytest.raises(ValueError, match='calibration run'):
            CalibrationStatus(model='opus', prompt_sha256='9' * 64, calibration_run_id=None, passed=True)


class TestReportPaths:
    def test_paths_carry_the_date_and_the_plan_run_prefix(self, tmp_path):
        now = datetime(2026, 11, 2, 2, 15, tzinfo=UTC)
        assert report_paths(tmp_path, now, PLAN_RUN) == (
            tmp_path / f'link-heal-sweep-2026-11-02-{PLAN_RUN[:8]}.json',
            tmp_path / f'link-heal-sweep-2026-11-02-{PLAN_RUN[:8]}.md',
        )

    def test_a_naive_now_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match='timezone'):
            report_paths(tmp_path, datetime(2026, 11, 2, 2, 15), PLAN_RUN)
