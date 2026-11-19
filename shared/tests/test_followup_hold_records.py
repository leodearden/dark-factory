"""The ``metadata.followup_holder`` and ``metadata.followup_hold`` records and their registration."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from shared.followup_hold_records import (
    FOLLOWUP_HOLD_METADATA_KEY,
    FOLLOWUP_HOLDER_METADATA_KEY,
    FollowupHold,
    FollowupHolder,
    PromotionCause,
    PromotionRecord,
)
from shared.task_metadata import parse_metadata, registered_metadata_submodel

_AT = '2026-11-19T04:30:00+00:00'


def _records() -> list[BaseModel]:
    return [
        FollowupHolder(project_id='dark_factory', minted_at=_AT),
        FollowupHold(holder=7, held_at=_AT),
        PromotionRecord(by=PromotionCause.CITATION, at=_AT, ref='fk-1'),
    ]


def test_the_metadata_keys() -> None:
    assert FOLLOWUP_HOLDER_METADATA_KEY == 'followup_holder'
    assert FOLLOWUP_HOLD_METADATA_KEY == 'followup_hold'


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
    [('followup_holder', FollowupHolder), ('followup_hold', FollowupHold)],
)
def test_both_keys_are_registered_submodels(key: str, model: type[BaseModel]) -> None:
    assert registered_metadata_submodel(key) is model


def test_valid_records_parse_without_unknown_key_warnings() -> None:
    blob = {
        FOLLOWUP_HOLDER_METADATA_KEY: {'project_id': 'dark_factory', 'minted_at': _AT},
        FOLLOWUP_HOLD_METADATA_KEY: {'holder': 7, 'held_at': _AT},
    }

    metadata, warnings = parse_metadata(blob, direction='write', enforce=True)

    assert warnings == []
    assert metadata.model_extra is not None
    assert isinstance(metadata.model_extra[FOLLOWUP_HOLDER_METADATA_KEY], FollowupHolder)
    assert isinstance(metadata.model_extra[FOLLOWUP_HOLD_METADATA_KEY], FollowupHold)


def test_a_hold_whose_holder_is_not_an_int_warns_naming_the_key() -> None:
    blob = {FOLLOWUP_HOLD_METADATA_KEY: {'holder': 'the-holder', 'held_at': _AT}}

    _, warnings = parse_metadata(blob, direction='read')

    assert [(w.field, w.code) for w in warnings] == [(FOLLOWUP_HOLD_METADATA_KEY, 'invalid_submodel')]
