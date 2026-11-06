"""DecisionRecord.audit_outcome: the sitting's verdict on an executor close, and its one writer."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator import session_registry as sr

_PROJECT = 'dark_factory'
_QUEUE = '/srv/df/data/escalations'
_AUDITED_AT = '2026-09-26T09:15:00+00:00'
_EVIDENCE = 'row landing_provenance: merge sha abc123 found on main'


def _decision(
    decision_id: str = 'dec-audit', audit_outcome: sr.AuditOutcome | None = None,
) -> sr.DecisionRecord:
    return sr.DecisionRecord(
        id=decision_id,
        project=_PROJECT,
        text='esc-1-1: landing provenance',
        filed_at='2026-09-20T00:00:00+00:00',
        escalation_id='esc-1-1',
        escalations_dir=_QUEUE,
        audit_outcome=audit_outcome,
    )


def _seed_closed(root: Path, decision_id: str = 'dec-audit') -> Path:
    assert sr.write_decision(_decision(decision_id), root=root)
    assert sr.close_decision_with_evidence(
        decision_id, sr.DecisionState.ANSWERED, _EVIDENCE, root=root,
        expected_project=_PROJECT, expected_escalations_dir=_QUEUE,
    ) is not None
    return sr.decision_path_for_id(decision_id, root=root)


def _audit(
    root: Path,
    decision_id: str = 'dec-audit',
    *,
    expected_project: str = _PROJECT,
    expected_escalations_dir: str = _QUEUE,
) -> sr.DecisionRecord | None:
    return sr.record_audit_outcome(
        decision_id, sr.AuditVerdict.WRONG, 'note', root,
        expected_project=expected_project,
        expected_escalations_dir=expected_escalations_dir,
        audited_at=_AUDITED_AT,
    )


def _only(root: Path) -> sr.DecisionRecord:
    (record,) = sr.list_decisions(root)
    return record


class TestSerialization:
    def test_defaults_to_none_and_round_trips(self) -> None:
        record = _decision()

        assert record.audit_outcome is None
        assert record.to_dict()['audit_outcome'] is None
        assert sr.DecisionRecord.from_json(record.to_json()) == record

    def test_a_legacy_body_without_the_key_loads(self) -> None:
        body = _decision().to_dict()
        del body['audit_outcome']

        assert sr.DecisionRecord.from_dict(body).audit_outcome is None

    def test_an_outcome_round_trips(self) -> None:
        outcome = sr.AuditOutcome(verdict=sr.AuditVerdict.OK, note='', audited_at=_AUDITED_AT)
        record = _decision(audit_outcome=outcome)

        assert sr.DecisionRecord.from_json(record.to_json()) == record

    @pytest.mark.parametrize('verdict', ['maybe', '', None])
    def test_an_unknown_verdict_is_refused(self, verdict: object) -> None:
        body = _decision().to_dict()
        body['audit_outcome'] = {'verdict': verdict, 'note': '', 'audited_at': _AUDITED_AT}

        with pytest.raises(ValueError):
            sr.DecisionRecord.from_dict(body)

    def test_a_non_iso_audited_at_is_refused(self) -> None:
        body = _decision().to_dict()
        body['audit_outcome'] = {'verdict': 'ok', 'note': '', 'audited_at': 'yesterday'}

        with pytest.raises(ValueError):
            sr.DecisionRecord.from_dict(body)

    def test_list_decisions_skips_a_body_with_an_unknown_verdict(self, tmp_path: Path) -> None:
        path = _seed_closed(tmp_path)
        body = json.loads(path.read_text())
        body['audit_outcome'] = {'verdict': 'maybe', 'note': '', 'audited_at': _AUDITED_AT}
        path.write_text(json.dumps(body))

        assert sr.list_decisions(tmp_path) == []


class TestRecordAuditOutcome:
    def test_records_the_outcome_on_a_closed_record(self, tmp_path: Path) -> None:
        _seed_closed(tmp_path)

        returned = _audit(tmp_path)

        expected = sr.AuditOutcome(verdict=sr.AuditVerdict.WRONG, note='note', audited_at=_AUDITED_AT)
        assert returned is not None and returned.audit_outcome == expected
        reread = _only(tmp_path)
        assert reread.audit_outcome == expected
        assert reread.to_dict()['audit_outcome'] == {
            'verdict': 'wrong', 'note': 'note', 'audited_at': _AUDITED_AT,
        }
        assert reread.closing_evidence == _EVIDENCE

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
            sr.record_audit_outcome(
                'dec-audit', sr.AuditVerdict.OK, '', tmp_path,
                expected_project=_PROJECT, expected_escalations_dir=_QUEUE, audited_at=_AUDITED_AT,
            )

        assert path.read_bytes() == before

    @pytest.mark.parametrize(
        'overrides',
        [{'expected_project': 'reify'}, {'expected_escalations_dir': '/srv/other/data/escalations'}],
        ids=['project', 'queue'],
    )
    def test_another_records_identity_is_refused_untouched(
        self, tmp_path: Path, overrides: dict[str, object],
    ) -> None:
        path = _seed_closed(tmp_path)
        before = path.read_bytes()

        with pytest.raises(sr.DecisionAuditRefused):
            _audit(tmp_path, **overrides)

        assert path.read_bytes() == before

    def test_a_missing_record_is_fail_soft(self, tmp_path: Path) -> None:
        assert _audit(tmp_path, decision_id='dec-absent') is None

    def test_reopen_keeps_the_audited_close(self, tmp_path: Path) -> None:
        _seed_closed(tmp_path)
        audited = _audit(tmp_path)
        assert audited is not None

        reopened = sr.reopen_decision(
            'dec-audit', tmp_path, expected_project=_PROJECT, expected_escalations_dir=_QUEUE,
        )

        assert reopened is not None
        assert _only(tmp_path).audit_outcome == audited.audit_outcome
