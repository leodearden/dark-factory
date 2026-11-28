"""The ``metadata.followup_holder``, ``metadata.followup_hold`` and ``metadata.finding_set_hold`` records and their registration."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from shared.followup_hold_records import (
    FINDING_SET_HOLD_METADATA_KEY,
    FOLLOWUP_HOLD_METADATA_KEY,
    FOLLOWUP_HOLDER_METADATA_KEY,
    FindingSetHold,
    FollowupHold,
    FollowupHolder,
    PromotionCause,
    PromotionRecord,
)
from shared.task_metadata import parse_metadata, registered_metadata_submodel

_AT = '2026-11-19T04:30:00+00:00'
_RULING = 'Leo 2026-11-12 on fk-c0932f0b7df1'


def _records() -> list[BaseModel]:
    return [
        FollowupHolder(project_id='dark_factory', minted_at=_AT),
        FollowupHold(holder=7, held_at=_AT),
        PromotionRecord(by=PromotionCause.CITATION, at=_AT, ref='fk-1'),
        FindingSetHold(ruling=_RULING),
    ]


def test_the_metadata_keys() -> None:
    assert FOLLOWUP_HOLDER_METADATA_KEY == 'followup_holder'
    assert FOLLOWUP_HOLD_METADATA_KEY == 'followup_hold'
    assert FINDING_SET_HOLD_METADATA_KEY == 'finding_set_hold'


def test_promotion_causes_are_the_six_enumerated() -> None:
    assert {cause.value for cause in PromotionCause} == {
        'dependent', 'citation', 'combine', 'pack', 'rider', 'human',
    }


@pytest.mark.parametrize('record', _records(), ids=lambda record: type(record).__name__)
def test_records_are_frozen(record: BaseModel) -> None:
    field = next(iter(type(record).model_fields))

    with pytest.raises(ValidationError):
        setattr(record, field, 'x')


@pytest.mark.parametrize('record', _records(), ids=lambda record: type(record).__name__)
def test_records_forbid_extra_keys(record: BaseModel) -> None:
    with pytest.raises(ValidationError):
        type(record).model_validate({**record.model_dump(mode='json'), 'surplus': 1})


def test_an_unknown_promotion_cause_is_refused() -> None:
    with pytest.raises(ValidationError):
        PromotionRecord.model_validate({'by': 'age_out', 'at': _AT})


@pytest.mark.parametrize('blob', [{}, {'ruling': ''}], ids=['missing', 'empty'])
def test_a_finding_set_hold_needs_a_ruling(blob: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        FindingSetHold.model_validate(blob)


def test_an_unpromoted_hold_dumps_without_promoted() -> None:
    record = FollowupHold(holder=7, held_at=_AT)

    assert record.model_dump(mode='json', exclude_none=True) == {'holder': 7, 'held_at': _AT}


def test_a_promoted_hold_round_trips() -> None:
    record = FollowupHold(
        holder=7, held_at=_AT, promoted=PromotionRecord(by=PromotionCause.HUMAN, at=_AT),
    )

    dumped = record.model_dump(mode='json', exclude_none=True)

    assert dumped == {'holder': 7, 'held_at': _AT, 'promoted': {'by': 'human', 'at': _AT}}
    assert FollowupHold.model_validate(dumped) == record


@pytest.mark.parametrize(
    ('key', 'model'),
    [
        ('followup_holder', FollowupHolder),
        ('followup_hold', FollowupHold),
        ('finding_set_hold', FindingSetHold),
    ],
)
def test_every_hold_key_is_a_registered_submodel(key: str, model: type[BaseModel]) -> None:
    assert registered_metadata_submodel(key) is model


def test_valid_records_parse_without_unknown_key_warnings() -> None:
    blob = {
        FOLLOWUP_HOLDER_METADATA_KEY: {'project_id': 'dark_factory', 'minted_at': _AT},
        FOLLOWUP_HOLD_METADATA_KEY: {'holder': 7, 'held_at': _AT},
        FINDING_SET_HOLD_METADATA_KEY: {'ruling': _RULING},
    }

    metadata, warnings = parse_metadata(blob, direction='write', enforce=True)

    assert warnings == []
    assert metadata.model_extra is not None
    assert isinstance(metadata.model_extra[FOLLOWUP_HOLDER_METADATA_KEY], FollowupHolder)
    assert isinstance(metadata.model_extra[FOLLOWUP_HOLD_METADATA_KEY], FollowupHold)
    assert isinstance(metadata.model_extra[FINDING_SET_HOLD_METADATA_KEY], FindingSetHold)


def test_a_hold_whose_holder_is_not_an_int_warns_naming_the_key() -> None:
    blob = {FOLLOWUP_HOLD_METADATA_KEY: {'holder': 'the-holder', 'held_at': _AT}}

    _, warnings = parse_metadata(blob, direction='read')

    assert [(w.field, w.code) for w in warnings] == [(FOLLOWUP_HOLD_METADATA_KEY, 'invalid_submodel')]


def test_a_finding_set_hold_with_a_misspelt_field_warns_naming_the_key() -> None:
    _, warnings = parse_metadata({FINDING_SET_HOLD_METADATA_KEY: {'ruled': 'x'}}, direction='read')

    assert [(w.field, w.code) for w in warnings] == [(FINDING_SET_HOLD_METADATA_KEY, 'invalid_submodel')]
