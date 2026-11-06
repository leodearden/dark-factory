"""DecisionRecord.audit_outcomes: the sitting's verdicts on its executor closes, and their one writer."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from orchestrator import session_registry as sr

_PROJECT = 'dark_factory'
_QUEUE = '/srv/df/data/escalations'
_AUDITED_AT = '2026-09-26T09:15:00+00:00'
_EVIDENCE = 'row landing_provenance: merge sha abc123 found on main'
_CLASS = 'landing_provenance'


def _decision(
    decision_id: str = 'dec-audit', audit_outcomes: tuple[sr.AuditOutcome, ...] = (),
) -> sr.DecisionRecord:
    return sr.DecisionRecord(
        id=decision_id,
        project=_PROJECT,
        text='esc-1-1: landing provenance',
        filed_at='2026-09-20T00:00:00+00:00',
        escalation_id='esc-1-1',
        escalations_dir=_QUEUE,
        audit_outcomes=audit_outcomes,
    )


def _seed_closed(root: Path, decision_id: str = 'dec-audit') -> Path:
    assert sr.write_decision(_decision(decision_id), root=root)
    assert sr.close_decision_with_evidence(
        decision_id, sr.DecisionState.ANSWERED, _EVIDENCE, root=root,
        expected_project=_PROJECT, expected_escalations_dir=_QUEUE,
    ) is not None
    return sr.decision_path_for_id(decision_id, root=root)


def _closed_at(root: Path, decision_id: str = 'dec-audit') -> str:
    """The close *decision_id* carries now, as an auditor drawing it reads it; '' when there is no record."""
    path = sr.decision_path_for_id(decision_id, root=root)
    return sr.DecisionRecord.from_json(path.read_text()).closed_at if path.exists() else ''


def _audit(
    root: Path,
    decision_id: str = 'dec-audit',
    *,
    verdict: sr.AuditVerdict = sr.AuditVerdict.WRONG,
    expected_project: str = _PROJECT,
    expected_escalations_dir: str = _QUEUE,
    policy_class: str = _CLASS,
    closed_at: str | None = None,
) -> sr.DecisionRecord | None:
    return sr.record_audit_outcome(
        decision_id, verdict, 'note', root,
        expected_project=expected_project,
        expected_escalations_dir=expected_escalations_dir,
        audited_at=_AUDITED_AT,
        policy_class=policy_class,
        closed_at=_closed_at(root, decision_id) if closed_at is None else closed_at,
    )


def _reopen(root: Path) -> None:
    assert sr.reopen_decision(
        'dec-audit', root, expected_project=_PROJECT, expected_escalations_dir=_QUEUE,
    ) is not None


def _close_again(root: Path) -> None:
    assert sr.close_decision_with_evidence(
        'dec-audit', sr.DecisionState.ANSWERED, f'{_EVIDENCE}, closed again', root=root,
        expected_project=_PROJECT, expected_escalations_dir=_QUEUE,
    ) is not None


def _only(root: Path) -> sr.DecisionRecord:
    (record,) = sr.list_decisions(root)
    return record


class TestSerialization:
    def test_defaults_to_empty_and_round_trips(self) -> None:
        record = _decision()

        assert record.audit_outcomes == ()
        assert record.to_dict()['audit_outcomes'] == []
        assert sr.DecisionRecord.from_json(record.to_json()) == record

    def test_a_legacy_body_without_the_key_loads(self) -> None:
        body = _decision().to_dict()
        del body['audit_outcomes']

        assert sr.DecisionRecord.from_dict(body).audit_outcomes == ()

    def test_an_outcome_round_trips(self) -> None:
        outcome = sr.AuditOutcome(
            verdict=sr.AuditVerdict.OK, note='', audited_at=_AUDITED_AT, policy_class=_CLASS,
            closed_at='2026-09-25T00:00:00+00:00',
        )
        record = _decision(audit_outcomes=(outcome,))

        assert sr.DecisionRecord.from_json(record.to_json()) == record
        assert record.to_dict()['audit_outcomes'][0]['policy_class'] == _CLASS
        assert record.to_dict()['audit_outcomes'][0]['closed_at'] == '2026-09-25T00:00:00+00:00'

    @pytest.mark.parametrize('closed_at', ['yesterday', 7])
    def test_an_outcome_naming_a_malformed_close_is_refused(self, closed_at: object) -> None:
        body = {'verdict': 'ok', 'note': '', 'audited_at': _AUDITED_AT, 'policy_class': _CLASS,
                'closed_at': closed_at}

        with pytest.raises(ValueError):
            sr.AuditOutcome.from_dict(body)

    def test_the_current_audit_is_the_outcome_that_judged_the_close_carried_now(self) -> None:
        judged = '2026-09-25T00:00:00+00:00'
        outcome = sr.AuditOutcome(sr.AuditVerdict.OK, '', _AUDITED_AT, _CLASS, closed_at=judged)

        assert dataclasses.replace(_decision(audit_outcomes=(outcome,)), closed_at=judged).current_audit == outcome
        later = dataclasses.replace(_decision(audit_outcomes=(outcome,)), closed_at='2026-09-27T00:00:00+00:00')
        assert later.current_audit is None
        assert later.audit_outcomes == (outcome,)
        assert _decision().current_audit is None

    @pytest.mark.parametrize('verdict', ['maybe', '', None])
    def test_an_unknown_verdict_is_refused(self, verdict: object) -> None:
        body = _decision().to_dict()
        body['audit_outcomes'] = [{
            'verdict': verdict, 'note': '', 'audited_at': _AUDITED_AT, 'policy_class': _CLASS,
        }]

        with pytest.raises(ValueError):
            sr.DecisionRecord.from_dict(body)

    def test_a_non_iso_audited_at_is_refused(self) -> None:
        body = _decision().to_dict()
        body['audit_outcomes'] = [{
            'verdict': 'ok', 'note': '', 'audited_at': 'yesterday', 'policy_class': _CLASS,
        }]

        with pytest.raises(ValueError):
            sr.DecisionRecord.from_dict(body)

    def test_list_decisions_skips_a_body_with_an_unknown_verdict(self, tmp_path: Path) -> None:
        path = _seed_closed(tmp_path)
        body = json.loads(path.read_text())
        body['audit_outcomes'] = [{
            'verdict': 'maybe', 'note': '', 'audited_at': _AUDITED_AT, 'policy_class': _CLASS,
        }]
        path.write_text(json.dumps(body))

        assert sr.list_decisions(tmp_path) == []

    @pytest.mark.parametrize('stored', [{'verdict': 'ok'}, 'ok', [7]])
    def test_a_history_that_is_not_a_list_of_outcomes_is_refused(self, stored: object) -> None:
        body = _decision().to_dict()
        body['audit_outcomes'] = stored

        with pytest.raises(ValueError, match='audit_outcomes'):
            sr.DecisionRecord.from_dict(body)

    def test_an_outcome_with_an_empty_class_is_refused(self) -> None:
        with pytest.raises(ValueError, match='policy_class'):
            sr.AuditOutcome(verdict=sr.AuditVerdict.OK, note='', audited_at=_AUDITED_AT, policy_class='')

    @pytest.mark.parametrize('policy_class', [7, None, ['landing_provenance']])
    def test_an_outcome_with_a_non_text_class_is_refused(self, policy_class: object) -> None:
        body = {'verdict': 'ok', 'note': '', 'audited_at': _AUDITED_AT, 'policy_class': policy_class}

        with pytest.raises(ValueError, match='policy_class'):
            sr.AuditOutcome.from_dict(body)

    def test_a_stored_outcome_without_a_class_fails_to_load(self) -> None:
        body = _decision().to_dict()
        body['audit_outcomes'] = [{'verdict': 'ok', 'note': '', 'audited_at': _AUDITED_AT}]

        with pytest.raises((KeyError, ValueError)):
            sr.DecisionRecord.from_dict(body)

    def test_list_decisions_skips_a_body_whose_outcome_names_no_class(self, tmp_path: Path) -> None:
        path = _seed_closed(tmp_path)
        body = json.loads(path.read_text())
        body['audit_outcomes'] = [{'verdict': 'ok', 'note': '', 'audited_at': _AUDITED_AT}]
        path.write_text(json.dumps(body))

        assert sr.list_decisions(tmp_path) == []


class TestRecordAuditOutcome:
    def test_records_the_outcome_on_a_closed_record(self, tmp_path: Path) -> None:
        _seed_closed(tmp_path)

        returned = _audit(tmp_path)

        closed_at = _closed_at(tmp_path)
        expected = sr.AuditOutcome(
            verdict=sr.AuditVerdict.WRONG, note='note', audited_at=_AUDITED_AT, policy_class=_CLASS,
            closed_at=closed_at,
        )
        assert returned is not None and returned.audit_outcomes == (expected,)
        reread = _only(tmp_path)
        assert reread.audit_outcomes == (expected,)
        assert reread.current_audit == expected
        assert reread.to_dict()['audit_outcomes'] == [{
            'verdict': 'wrong', 'note': 'note', 'audited_at': _AUDITED_AT, 'policy_class': _CLASS,
            'closed_at': closed_at,
        }]
        assert sr.DecisionRecord.from_json(reread.to_json()) == reread
        assert reread.closing_evidence == _EVIDENCE

    def test_an_empty_class_is_refused_before_the_lock(self, tmp_path: Path) -> None:
        path = _seed_closed(tmp_path)
        before = path.read_bytes()

        with pytest.raises(ValueError, match='policy_class'):
            _audit(tmp_path, policy_class='')

        assert path.read_bytes() == before

    def test_an_open_record_is_refused_untouched(self, tmp_path: Path) -> None:
        assert sr.write_decision(_decision(), root=tmp_path)
        path = sr.decision_path_for_id('dec-audit', root=tmp_path)
        before = path.read_bytes()

        with pytest.raises(sr.DecisionAuditRefused):
            _audit(tmp_path)

        assert path.read_bytes() == before

    def test_an_existing_outcome_is_never_overwritten(self, tmp_path: Path) -> None:
        path = _seed_closed(tmp_path)
        assert _audit(tmp_path) is not None
        before = path.read_bytes()

        with pytest.raises(sr.DecisionAuditRefused):
            _audit(tmp_path, verdict=sr.AuditVerdict.OK)

        assert path.read_bytes() == before

    @pytest.mark.parametrize(
        ('project', 'queue'),
        [('reify', _QUEUE), (_PROJECT, '/srv/other/data/escalations')],
        ids=['project', 'queue'],
    )
    def test_another_records_identity_is_refused_untouched(
        self, tmp_path: Path, project: str, queue: str,
    ) -> None:
        path = _seed_closed(tmp_path)
        before = path.read_bytes()

        with pytest.raises(sr.DecisionAuditRefused):
            _audit(tmp_path, expected_project=project, expected_escalations_dir=queue)

        assert path.read_bytes() == before

    def test_a_missing_record_is_fail_soft(self, tmp_path: Path) -> None:
        assert _audit(tmp_path, decision_id='dec-absent') is None

    def test_reopen_keeps_the_audited_close(self, tmp_path: Path) -> None:
        _seed_closed(tmp_path)
        audited = _audit(tmp_path)
        assert audited is not None

        _reopen(tmp_path)

        assert _only(tmp_path).audit_outcomes == audited.audit_outcomes
        assert _only(tmp_path).current_audit is None


class TestAReopenedRecordClosedAgain:
    """Each outcome names the close it judged, so a later close of the same record is audited in its own right."""

    def _closed_twice(self, root: Path) -> tuple[str, str]:
        _seed_closed(root)
        first = _closed_at(root)
        assert _audit(root) is not None
        _reopen(root)
        _close_again(root)
        return first, _closed_at(root)

    def test_its_new_close_is_unaudited_while_the_earlier_audit_is_kept(self, tmp_path: Path) -> None:
        first, second = self._closed_twice(tmp_path)

        record = _only(tmp_path)
        assert second != first
        assert record.current_audit is None
        assert [outcome.closed_at for outcome in record.audit_outcomes] == [first]

    def test_auditing_the_new_close_appends_and_keeps_the_earlier_audit(self, tmp_path: Path) -> None:
        first, second = self._closed_twice(tmp_path)

        returned = _audit(tmp_path, verdict=sr.AuditVerdict.OK)

        assert returned is not None
        record = _only(tmp_path)
        assert record.current_audit is not None
        assert (record.current_audit.verdict, record.current_audit.closed_at) == (sr.AuditVerdict.OK, second)
        assert [(o.verdict, o.closed_at) for o in record.audit_outcomes] == [
            (sr.AuditVerdict.WRONG, first), (sr.AuditVerdict.OK, second),
        ]

    def test_a_verdict_on_the_earlier_close_is_refused_untouched(self, tmp_path: Path) -> None:
        first, _ = self._closed_twice(tmp_path)
        path = sr.decision_path_for_id('dec-audit', root=tmp_path)
        before = path.read_bytes()

        with pytest.raises(sr.DecisionAuditRefused, match='closed again'):
            _audit(tmp_path, closed_at=first)

        assert path.read_bytes() == before
