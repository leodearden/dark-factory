"""shared.task_deferral: the caller-input contract, the stored record and its predicates.

Contract: plans/deferral-flip-condition-prd.md §5.1. The clock is a fixed NOW
moved by timedelta offsets; everything goes through the public names.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from shared.task_deferral import (
    CALLER_KINDS,
    CONDITION_MAX_CHARS,
    DEFAULT_GRACE_SECS,
    GRACE_BOUNDS,
    REASON_MAX_CHARS,
    CarriedByRequest,
    DeferralInvalid,
    DeferralInvalidReason,
    DeferralKind,
    HeldBySessionRequest,
    LegacyUnknownRequest,
    UntilConditionRequest,
    parse_request,
    parse_utc_timestamp,
)

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)

_MINIMAL_REQUEST_FIELDS = {
    DeferralKind.CARRIED_BY: {'carrier_task_id': 6524},
    DeferralKind.UNTIL_CONDITION: {'condition': 'x'},
    DeferralKind.HELD_BY_SESSION: {'holder_pid': 1234},
    DeferralKind.LEGACY_UNKNOWN: {},
}


def _until(**fields: object) -> dict[str, object]:
    return {'kind': 'until_condition', 'condition': 'x', **fields}


def _held(**fields: object) -> dict[str, object]:
    return {'kind': 'held_by_session', 'holder_pid': 1234, **fields}


def _carried(**fields: object) -> dict[str, object]:
    return {'kind': 'carried_by', 'carrier_task_id': 6524, **fields}


def _until_request(**fields: object) -> UntilConditionRequest:
    request = parse_request(_until(**fields), now=NOW)
    assert isinstance(request, UntilConditionRequest)
    return request


def _held_request(**fields: object) -> HeldBySessionRequest:
    request = parse_request(_held(**fields), now=NOW)
    assert isinstance(request, HeldBySessionRequest)
    return request


# --- vocabulary --------------------------------------------------------------


def test_caller_kinds_are_the_three_kinds_a_caller_may_choose():
    caller_kinds = CALLER_KINDS

    assert caller_kinds == {
        DeferralKind.CARRIED_BY,
        DeferralKind.UNTIL_CONDITION,
        DeferralKind.HELD_BY_SESSION,
    }
    assert isinstance(caller_kinds, frozenset)
    assert caller_kinds <= set(DeferralKind)


# --- accepted requests -------------------------------------------------------


@pytest.mark.parametrize('carrier', [6524, '6524'])
def test_carried_by_names_its_carrier_as_an_int(carrier):
    request = parse_request(_carried(carrier_task_id=carrier), now=NOW)

    assert isinstance(request, CarriedByRequest)
    assert request.carrier_task_id == 6524
    assert type(request.carrier_task_id) is int


def test_until_condition_without_expiry():
    request = parse_request(_until(), now=NOW)

    assert isinstance(request, UntilConditionRequest)
    assert request.condition == 'x'
    assert request.expires_at is None


def test_held_by_session_defaults_its_grace():
    request = parse_request(_held(), now=NOW)

    assert isinstance(request, HeldBySessionRequest)
    assert request.holder_pid == 1234
    assert request.grace_secs == DEFAULT_GRACE_SECS == 7200


def test_legacy_unknown_parses_because_its_row_shape_rule_is_the_gates():
    request = parse_request({'kind': 'legacy_unknown'}, now=NOW)

    assert isinstance(request, LegacyUnknownRequest)


@pytest.mark.parametrize('kind', sorted(_MINIMAL_REQUEST_FIELDS))
def test_reason_at_its_limit_is_accepted_on_every_kind(kind):
    reason = 'r' * REASON_MAX_CHARS
    raw = {'kind': kind.value, **_MINIMAL_REQUEST_FIELDS[kind], 'reason': reason}

    assert parse_request(raw, now=NOW).reason == reason
    assert REASON_MAX_CHARS == 2000


def test_condition_at_its_limit_is_accepted():
    condition = 'c' * CONDITION_MAX_CHARS

    assert _until_request(condition=condition).condition == condition
    assert CONDITION_MAX_CHARS == 500


@pytest.mark.parametrize('grace_secs', GRACE_BOUNDS)
def test_grace_bounds_are_inclusive(grace_secs):
    assert _held_request(grace_secs=grace_secs).grace_secs == grace_secs
    assert GRACE_BOUNDS == (600, 86400)


@pytest.mark.parametrize(
    'expires_at',
    ['2026-10-10T00:00:00', '2026-10-10T02:00:00+02:00', '2026-10-10T00:00:00Z'],
)
def test_expires_at_is_normalised_to_utc(expires_at):
    expiry = _until_request(expires_at=expires_at).expires_at

    assert expiry == datetime(2026, 10, 10, tzinfo=UTC)
    assert expiry is not None and expiry.tzinfo is UTC


def test_expires_at_one_microsecond_ahead_is_accepted():
    soonest = NOW + timedelta(microseconds=1)

    assert _until_request(expires_at=soonest.isoformat()).expires_at == soonest


def test_requests_are_frozen():
    request = _until_request()

    with pytest.raises(ValidationError):
        request.condition = 'y'


# --- refusals ----------------------------------------------------------------

_R = DeferralInvalidReason
_PAST = (NOW - timedelta(seconds=1)).isoformat()

_REFUSALS = [
    pytest.param({'kind': 'planning'}, _R.SERVER_ONLY_KIND, 'kind', 'planning', id='planning'),
    pytest.param({'kind': 'until_task'}, _R.UNKNOWN_KIND, 'kind', 'until_task', id='unknown'),
    pytest.param({'kind': 5}, _R.UNKNOWN_KIND, 'kind', 5, id='kind-not-a-str'),
    pytest.param({}, _R.MISSING_FIELD, 'kind', None, id='no-kind'),
    pytest.param(
        {'kind': 'carried_by'}, _R.MISSING_FIELD, 'carrier_task_id', None, id='no-carrier'
    ),
    pytest.param(_until(condition='  '), _R.MISSING_FIELD, 'condition', '  ', id='blank'),
    pytest.param(
        _until(holder_pid=1), _R.FIELD_NOT_ALLOWED, 'holder_pid', 1, id='foreign-field'
    ),
    pytest.param(
        _until(condition='c' * 501), _R.TEXT_TOO_LONG, 'condition', 'c' * 501, id='condition'
    ),
    pytest.param(
        _until(reason='r' * 2001), _R.TEXT_TOO_LONG, 'reason', 'r' * 2001, id='reason'
    ),
    pytest.param(
        _held(grace_secs=599), _R.GRACE_OUT_OF_RANGE, 'grace_secs', 599, id='grace-short'
    ),
    pytest.param(
        _held(grace_secs=86401), _R.GRACE_OUT_OF_RANGE, 'grace_secs', 86401, id='grace-long'
    ),
    *(
        pytest.param(
            _held(holder_pid=pid), _R.INVALID_VALUE, 'holder_pid', pid, id=f'pid-{pid!r}'
        )
        for pid in (True, '1234', 0, -5)
    ),
    *(
        pytest.param(
            _carried(carrier_task_id=carrier),
            _R.INVALID_VALUE,
            'carrier_task_id',
            carrier,
            id=f'carrier-{carrier!r}',
        )
        for carrier in (True, 'abc', 0)
    ),
    pytest.param(
        _until(expires_at='tomorrow'), _R.INVALID_VALUE, 'expires_at', 'tomorrow', id='prose'
    ),
    pytest.param(
        _until(expires_at=1760000000),
        _R.INVALID_VALUE,
        'expires_at',
        1760000000,
        id='epoch-int',
    ),
    pytest.param(
        _until(expires_at=NOW.isoformat()),
        _R.EXPIRES_AT_NOT_FUTURE,
        'expires_at',
        NOW.isoformat(),
        id='expires-now',
    ),
    pytest.param(
        _until(expires_at=_PAST), _R.EXPIRES_AT_NOT_FUTURE, 'expires_at', _PAST, id='past'
    ),
    pytest.param([], _R.INVALID_VALUE, 'deferral', [], id='list'),
    pytest.param('carried_by', _R.INVALID_VALUE, 'deferral', 'carried_by', id='str'),
]


@pytest.mark.parametrize(('raw', 'reason', 'field', 'value'), _REFUSALS)
def test_refusal_names_its_reason_field_and_value(raw, reason, field, value):
    with pytest.raises(DeferralInvalid) as refused:
        parse_request(raw, now=NOW)

    assert refused.value.reason is reason
    assert refused.value.field == field
    assert refused.value.value == value
    assert type(refused.value.value) is type(value)


def test_refusal_message_names_reason_field_and_value():
    with pytest.raises(DeferralInvalid) as refused:
        parse_request(_held(grace_secs=599), now=NOW)

    message = str(refused.value)
    assert 'grace_out_of_range' in message
    assert 'grace_secs' in message
    assert '599' in message


# --- the one timestamp parser ------------------------------------------------


def test_naive_timestamp_reads_as_utc():
    assert parse_utc_timestamp('2026-10-09T12:00:00') == NOW


def test_timestamp_parser_refuses_a_number():
    with pytest.raises(ValueError):
        parse_utc_timestamp(1760000000)  # type: ignore[arg-type]
