"""``before_done.recheck_secs``: the re-arm declaration's shape, kind and milestone rules.

PRD plans/completion-driven-triggers-prd.md C1. A predicate that may answer
"not yet" (exit 75) declares ``recheck_secs``; the runner then advances the
dated ``milestone.at`` by that many seconds. The shape and kind half live on
``BeforeDone``; the milestone half is ``recheck_secs_milestone_conflict``,
which ``TaskMetadata`` calls, so both hold at every write that parses metadata.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.task_metadata import (
    BeforeDone,
    Milestone,
    TaskMetadata,
    parse_metadata,
    recheck_secs_milestone_conflict,
)

DATED = {'mode': 'dated', 'at': '2026-11-01T00:00:00+00:00'}
DELAYED = {'mode': 'delayed', 'after_secs': 60}


def _before_done(**fields: object) -> dict:
    return {'script': 's', 'timeout_secs': 60, 'kind': 'predicate', **fields}


def _predicate_blob(*, milestone: dict | None, **before_done_fields: object) -> dict:
    blob: dict = {
        'task_kind': 'deterministic',
        'before_done': _before_done(**before_done_fields),
    }
    if milestone is not None:
        blob['milestone'] = milestone
    return blob


class TestBeforeDoneRecheckSecs:
    def test_declared_recheck_secs_is_exposed(self) -> None:
        assert BeforeDone.model_validate(_before_done(recheck_secs=86400)).recheck_secs == 86400

    def test_absent_recheck_secs_is_none(self) -> None:
        assert BeforeDone.model_validate(_before_done()).recheck_secs is None

    def test_minimum_is_three_hundred_seconds(self) -> None:
        assert BeforeDone.model_validate(_before_done(recheck_secs=300)).recheck_secs == 300
        with pytest.raises(ValidationError, match='recheck_secs'):
            BeforeDone.model_validate(_before_done(recheck_secs=299))

    @pytest.mark.parametrize('lax_value', [True, 300.0, '300'], ids=['bool', 'float', 'str'])
    def test_only_a_strict_int_is_accepted(self, lax_value: object) -> None:
        with pytest.raises(ValidationError, match='recheck_secs'):
            BeforeDone.model_validate(_before_done(recheck_secs=lax_value))

    @pytest.mark.parametrize('kind_fields', [{'kind': 'deploy'}, {}], ids=['explicit', 'default'])
    def test_recheck_secs_requires_a_predicate(self, kind_fields: dict) -> None:
        raw = {'script': 's', 'timeout_secs': 60, 'recheck_secs': 86400, **kind_fields}
        with pytest.raises(ValidationError) as excinfo:
            BeforeDone.model_validate(raw)
        message = str(excinfo.value)
        assert 'recheck_secs' in message
        assert "kind='predicate'" in message


class TestTaskMetadataMilestoneHalf:
    def test_dated_milestone_passes(self) -> None:
        model, warnings = parse_metadata(
            _predicate_blob(milestone=DATED, recheck_secs=86400), direction='write', enforce=True,
        )
        assert model.before_done is not None
        assert model.before_done.recheck_secs == 86400
        assert warnings == []

    def test_delayed_milestone_is_refused(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            parse_metadata(
                _predicate_blob(milestone=DELAYED, recheck_secs=86400),
                direction='write',
                enforce=True,
            )
        message = str(excinfo.value)
        assert 'recheck_secs' in message
        assert "mode='dated'" in message
        assert 'delayed' in message

    def test_absent_milestone_is_refused(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            parse_metadata(
                _predicate_blob(milestone=None, recheck_secs=86400), direction='write', enforce=True,
            )
        message = str(excinfo.value)
        assert 'recheck_secs' in message
        assert 'milestone' in message
        assert 'absent' in message

    def test_raw_dict_milestone_is_read_too(self) -> None:
        with pytest.raises(ValidationError, match='recheck_secs'):
            TaskMetadata(**_predicate_blob(milestone=DELAYED, recheck_secs=86400))

    def test_read_direction_warns_on_the_whole_blob(self) -> None:
        _model, warnings = parse_metadata(
            _predicate_blob(milestone=DELAYED, recheck_secs=86400), direction='read',
        )
        whole = [w for w in warnings if w.field == '<metadata>']
        assert len(whole) == 1
        assert whole[0].code == 'invalid_metadata'
        assert 'recheck_secs' in whole[0].message


class TestRecheckSecsMilestoneConflict:
    @pytest.mark.parametrize(
        'milestone',
        [None, DELAYED, Milestone(mode='delayed', after_secs=60)],
        ids=['absent', 'delayed-dict', 'delayed-model'],
    )
    def test_non_dated_milestones_conflict(self, milestone: object) -> None:
        reason = recheck_secs_milestone_conflict(milestone)
        assert isinstance(reason, str)
        assert reason

    @pytest.mark.parametrize(
        'milestone',
        [DATED, Milestone(mode='dated', at='2026-11-01T00:00:00+00:00')],
        ids=['dated-dict', 'dated-model'],
    )
    def test_dated_milestones_do_not_conflict(self, milestone: object) -> None:
        assert recheck_secs_milestone_conflict(milestone) is None


class TestPredicatesWithoutRecheckSecsUnchanged:
    @pytest.mark.parametrize('milestone', [DELAYED, None], ids=['delayed', 'absent'])
    def test_still_validates(self, milestone: dict | None) -> None:
        model, warnings = parse_metadata(
            _predicate_blob(milestone=milestone), direction='write', enforce=True,
        )
        assert model.before_done is not None
        assert model.before_done.recheck_secs is None
        assert warnings == []
