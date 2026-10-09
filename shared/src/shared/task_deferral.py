"""The one definition of a deferral hold: what ends it, who ends it, and what bounds it.

Contract: plans/deferral-flip-condition-prd.md §5.1 and decisions 1, 2, 5, 9 and 10.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, LiteralString, cast

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    RootModel,
    StringConstraints,
    TypeAdapter,
    ValidationError,
)
from pydantic_core import ErrorDetails, PydanticCustomError

from shared.process_identity import ProcessIdentity
from shared.task_metadata import register_metadata_submodel

__all__ = [
    'CALLER_KINDS',
    'CONDITION_MAX_CHARS',
    'DEFAULT_GRACE_SECS',
    'DEFAULT_SWEEP_INTERVAL_SECS',
    'GRACE_BOUNDS',
    'LAPSE_OVERDUE_AFTER_SECS',
    'REASON_MAX_CHARS',
    'STALE_CONDITION_AFTER_SECS',
    'STALE_PLANNING_AFTER_SECS',
    'CarriedByDeferral',
    'CarriedByRequest',
    'Deferral',
    'DeferralInvalid',
    'DeferralInvalidReason',
    'DeferralKind',
    'DeferralRecord',
    'DeferralRequest',
    'HeldBySessionDeferral',
    'HeldBySessionRequest',
    'InvalidRecord',
    'LegacyUnknownDeferral',
    'LegacyUnknownRequest',
    'PlanningDeferral',
    'UntilConditionDeferral',
    'UntilConditionRequest',
    'parse_record',
    'parse_request',
    'parse_utc_timestamp',
    'record_to_json',
    'stamp',
]


class DeferralKind(StrEnum):
    CARRIED_BY = 'carried_by'
    UNTIL_CONDITION = 'until_condition'
    HELD_BY_SESSION = 'held_by_session'
    PLANNING = 'planning'
    LEGACY_UNKNOWN = 'legacy_unknown'


CALLER_KINDS: Final = frozenset(
    {DeferralKind.CARRIED_BY, DeferralKind.UNTIL_CONDITION, DeferralKind.HELD_BY_SESSION}
)
_SERVER_ONLY_KINDS: Final = frozenset(DeferralKind) - CALLER_KINDS - {DeferralKind.LEGACY_UNKNOWN}


class DeferralInvalidReason(StrEnum):
    """PRD §5.3's deferral_invalid codes, plus INVALID_VALUE for a wrong-typed or unparseable value."""

    UNKNOWN_KIND = 'unknown_kind'
    MISSING_FIELD = 'missing_field'
    FIELD_NOT_ALLOWED = 'field_not_allowed'
    SERVER_ONLY_KIND = 'server_only_kind'
    HOLDER_NOT_LIVE = 'holder_not_live'
    HOLDER_NOT_SESSION = 'holder_not_session'
    HOLDER_UNVERIFIABLE = 'holder_unverifiable'
    CARRIER_NOT_FOUND = 'carrier_not_found'
    CARRIER_IS_SELF = 'carrier_is_self'
    EXPIRES_AT_NOT_FUTURE = 'expires_at_not_future'
    GRACE_OUT_OF_RANGE = 'grace_out_of_range'
    TEXT_TOO_LONG = 'text_too_long'
    STATUS_NOT_DEFERRED = 'status_not_deferred'
    NOT_PLANNING_HOLD = 'not_planning_hold'
    CAS_NEEDS_SINGLE_ID = 'cas_needs_single_id'
    INVALID_VALUE = 'invalid_value'


DEFAULT_GRACE_SECS: Final = 7200
GRACE_BOUNDS: Final = (600, 86400)
STALE_PLANNING_AFTER_SECS: Final = 86400
STALE_CONDITION_AFTER_SECS: Final = 30 * 86400
DEFAULT_SWEEP_INTERVAL_SECS: Final = 60
LAPSE_OVERDUE_AFTER_SECS: Final = 2 * DEFAULT_SWEEP_INTERVAL_SECS
CONDITION_MAX_CHARS: Final = 500
REASON_MAX_CHARS: Final = 2000


def parse_utc_timestamp(value: str | datetime) -> datetime:
    """ISO-8601 text or a datetime, as an aware UTC datetime; a naive value is read as UTC."""
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(parsed, datetime):
        raise ValueError(f'expected an ISO-8601 string or a datetime, got {value!r}')
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _field_error(
    reason: DeferralInvalidReason, message: LiteralString, **context: object
) -> PydanticCustomError:
    return PydanticCustomError(cast(LiteralString, reason.value), message, context)


def _utc_timestamp_field(value: Any) -> datetime:
    try:
        return parse_utc_timestamp(value)
    except ValueError as exc:
        raise _field_error(
            DeferralInvalidReason.INVALID_VALUE, '{error}', error=str(exc)
        ) from exc


def _non_blank(text: str) -> str:
    if not text.strip():
        raise _field_error(DeferralInvalidReason.MISSING_FIELD, 'must not be blank')
    return text


def _within_grace_bounds(secs: int) -> int:
    low, high = GRACE_BOUNDS
    if not low <= secs <= high:
        raise _field_error(
            DeferralInvalidReason.GRACE_OUT_OF_RANGE,
            'must be within {low}..{high} seconds',
            low=low,
            high=high,
        )
    return secs


def _decimal_text_as_int(value: Any) -> Any:
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        return int(value)
    return value


UtcTimestamp = Annotated[datetime, BeforeValidator(_utc_timestamp_field)]
ConditionText = Annotated[
    str, StringConstraints(max_length=CONDITION_MAX_CHARS), AfterValidator(_non_blank)
]
ReasonText = Annotated[str, StringConstraints(max_length=REASON_MAX_CHARS)]
GraceSecs = Annotated[int, Field(strict=True), AfterValidator(_within_grace_bounds)]
PositiveId = Annotated[int, Field(strict=True, gt=0)]
TaskId = Annotated[PositiveId, BeforeValidator(_decimal_text_as_int)]


class _DeferralRequestBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    reason: ReasonText | None = None


class CarriedByRequest(_DeferralRequestBase):
    kind: Literal[DeferralKind.CARRIED_BY] = DeferralKind.CARRIED_BY
    carrier_task_id: TaskId


class UntilConditionRequest(_DeferralRequestBase):
    kind: Literal[DeferralKind.UNTIL_CONDITION] = DeferralKind.UNTIL_CONDITION
    condition: ConditionText
    expires_at: UtcTimestamp | None = None


class HeldBySessionRequest(_DeferralRequestBase):
    kind: Literal[DeferralKind.HELD_BY_SESSION] = DeferralKind.HELD_BY_SESSION
    holder_pid: PositiveId
    grace_secs: GraceSecs = DEFAULT_GRACE_SECS


class LegacyUnknownRequest(_DeferralRequestBase):
    kind: Literal[DeferralKind.LEGACY_UNKNOWN] = DeferralKind.LEGACY_UNKNOWN


DeferralRequest = Annotated[
    CarriedByRequest | UntilConditionRequest | HeldBySessionRequest | LegacyUnknownRequest,
    Field(discriminator='kind'),
]
_REQUEST_ADAPTER: Final = TypeAdapter(DeferralRequest)


class DeferralInvalid(Exception):
    """A deferral the caller may not set, with the reason code, field and value that say why."""

    def __init__(self, reason: DeferralInvalidReason, field: str, value: object) -> None:
        super().__init__(f'deferral_invalid {reason.value}: {field}={value!r}')
        self.reason = reason
        self.field = field
        self.value = value


_ERROR_TYPE_REASONS: Final[Mapping[str, DeferralInvalidReason]] = {
    'union_tag_invalid': DeferralInvalidReason.UNKNOWN_KIND,
    'union_tag_not_found': DeferralInvalidReason.MISSING_FIELD,
    'missing': DeferralInvalidReason.MISSING_FIELD,
    'extra_forbidden': DeferralInvalidReason.FIELD_NOT_ALLOWED,
    'string_too_long': DeferralInvalidReason.TEXT_TOO_LONG,
    **{reason.value: reason for reason in DeferralInvalidReason},
}


def _refusal(error: ErrorDetails, raw: Mapping[Any, Any]) -> DeferralInvalid:
    reason = _ERROR_TYPE_REASONS.get(error['type'], DeferralInvalidReason.INVALID_VALUE)
    path_below_tag = error['loc'][1:]
    if not path_below_tag:
        return DeferralInvalid(reason, 'kind', raw.get('kind'))
    value = None if error['type'] == 'missing' else error['input']
    return DeferralInvalid(reason, '.'.join(map(str, path_below_tag)), value)


def parse_request(raw: object, *, now: datetime) -> DeferralRequest:
    """Validate a caller's `deferral` argument; every pure refusal raises DeferralInvalid."""
    if not isinstance(raw, Mapping):
        raise DeferralInvalid(DeferralInvalidReason.INVALID_VALUE, 'deferral', raw)
    kind = raw.get('kind')
    if isinstance(kind, str) and kind in _SERVER_ONLY_KINDS:
        raise DeferralInvalid(DeferralInvalidReason.SERVER_ONLY_KIND, 'kind', kind)
    try:
        request = _REQUEST_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        raise _refusal(exc.errors()[0], raw) from exc
    if isinstance(request, UntilConditionRequest) and _not_after(request.expires_at, now):
        raise DeferralInvalid(
            DeferralInvalidReason.EXPIRES_AT_NOT_FUTURE, 'expires_at', raw['expires_at']
        )
    return request


def _not_after(moment: datetime | None, now: datetime) -> bool:
    return moment is not None and moment <= now


class _DeferralRecordBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    reason: ReasonText | None = None
    deferred_at: UtcTimestamp | None
    stamped_at: UtcTimestamp
    stamped_by: str | None = None


class CarriedByDeferral(_DeferralRecordBase):
    kind: Literal[DeferralKind.CARRIED_BY] = DeferralKind.CARRIED_BY
    carrier_task_id: PositiveId


class UntilConditionDeferral(_DeferralRecordBase):
    kind: Literal[DeferralKind.UNTIL_CONDITION] = DeferralKind.UNTIL_CONDITION
    condition: ConditionText
    expires_at: UtcTimestamp | None = None


class HeldBySessionDeferral(_DeferralRecordBase):
    kind: Literal[DeferralKind.HELD_BY_SESSION] = DeferralKind.HELD_BY_SESSION
    holder: ProcessIdentity
    grace_secs: GraceSecs = DEFAULT_GRACE_SECS


class PlanningDeferral(_DeferralRecordBase):
    kind: Literal[DeferralKind.PLANNING] = DeferralKind.PLANNING


class LegacyUnknownDeferral(_DeferralRecordBase):
    kind: Literal[DeferralKind.LEGACY_UNKNOWN] = DeferralKind.LEGACY_UNKNOWN


_DeferralVariant = (
    CarriedByDeferral
    | UntilConditionDeferral
    | HeldBySessionDeferral
    | PlanningDeferral
    | LegacyUnknownDeferral
)
Deferral = Annotated[_DeferralVariant, Field(discriminator='kind')]


class DeferralRecord(RootModel[Deferral]):
    model_config = ConfigDict(frozen=True)


register_metadata_submodel('deferral', DeferralRecord, cardinality='dict')


def _required_holder(
    request: HeldBySessionRequest, holder: ProcessIdentity | None
) -> ProcessIdentity:
    if holder is None:
        raise ValueError(
            f'a held_by_session deferral needs the holder identity of pid '
            f'{request.holder_pid}, got holder=None'
        )
    if holder.pid != request.holder_pid:
        raise ValueError(
            f'holder pid {holder.pid} does not match the requested holder_pid '
            f'{request.holder_pid}'
        )
    return holder


def stamp(
    request: DeferralRequest,
    *,
    stamped_at: datetime,
    stamped_by: str | None,
    deferred_at: datetime | None,
    holder: ProcessIdentity | None = None,
) -> Deferral:
    """The one request-to-record translation; only held_by_session takes, and needs, a holder."""
    if holder is not None and not isinstance(request, HeldBySessionRequest):
        raise ValueError(f'a {request.kind.value} deferral has no holder, got {holder!r}')
    stamped: dict[str, Any] = {
        'reason': request.reason,
        'deferred_at': deferred_at,
        'stamped_at': stamped_at,
        'stamped_by': stamped_by,
    }
    match request:
        case CarriedByRequest():
            return CarriedByDeferral(carrier_task_id=request.carrier_task_id, **stamped)
        case UntilConditionRequest():
            return UntilConditionDeferral(
                condition=request.condition, expires_at=request.expires_at, **stamped
            )
        case HeldBySessionRequest():
            return HeldBySessionDeferral(
                holder=_required_holder(request, holder), grace_secs=request.grace_secs, **stamped
            )
        case LegacyUnknownRequest():
            return LegacyUnknownDeferral(**stamped)


@dataclass(frozen=True)
class InvalidRecord:
    """A metadata.deferral slice that does not parse; detail says why."""

    detail: str


def parse_record(raw: object) -> Deferral | InvalidRecord | None:
    """Any metadata.deferral slice as its variant: None when absent, InvalidRecord when malformed."""
    if raw is None:
        return None
    if isinstance(raw, DeferralRecord):
        return raw.root
    if isinstance(raw, _DeferralVariant):
        return raw
    try:
        return DeferralRecord.model_validate(raw).root
    except ValidationError as exc:
        return InvalidRecord(str(exc))


def record_to_json(record: Deferral) -> dict[str, Any]:
    """The one serialization of a record into metadata.deferral."""
    return DeferralRecord(record).model_dump(mode='json')
