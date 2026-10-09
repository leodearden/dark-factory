"""shared.task_deferral: the caller-input contract, the stored record and its predicates.

Contract: plans/deferral-flip-condition-prd.md §5.1. The clock is a fixed NOW
moved by timedelta offsets; everything goes through the public names.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from test_pure_stdlib_leaves import run_python_child

from shared.process_identity import ProcessIdentity
from shared.task_deferral import (
    CALLER_KINDS,
    CONDITION_MAX_CHARS,
    DEFAULT_GRACE_SECS,
    GRACE_BOUNDS,
    REASON_MAX_CHARS,
    CarriedByDeferral,
    CarriedByRequest,
    Deferral,
    DeferralInvalid,
    DeferralInvalidReason,
    DeferralKind,
    DeferralRecord,
    HeldBySessionDeferral,
    HeldBySessionRequest,
    InvalidRecord,
    LegacyUnknownDeferral,
    LegacyUnknownRequest,
    PlanningDeferral,
    UntilConditionDeferral,
    UntilConditionRequest,
    parse_record,
    parse_request,
    parse_utc_timestamp,
    record_to_json,
    stamp,
)
from shared.task_metadata import is_known_metadata_key, parse_metadata

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


# --- the stored record -------------------------------------------------------

HOLDER = ProcessIdentity(pid=4242, start_ticks=99, boot_id='b' * 8, host='h')
STAMPED_AT = NOW + timedelta(microseconds=123456)
_AGENT = 'claude-task-6524'


def _stamped(
    raw: dict[str, object],
    *,
    holder: ProcessIdentity | None = None,
    deferred_at: datetime | None = NOW,
    stamped_by: str | None = _AGENT,
) -> Deferral:
    return stamp(
        parse_request(raw, now=NOW),
        stamped_at=STAMPED_AT,
        stamped_by=stamped_by,
        deferred_at=deferred_at,
        holder=holder,
    )


def _held_record(**fields: object) -> HeldBySessionDeferral:
    record = _stamped(_held(holder_pid=HOLDER.pid, **fields), holder=HOLDER)
    assert isinstance(record, HeldBySessionDeferral)
    return record


def test_stamped_carried_by_keeps_the_request_and_the_stamp():
    record = _stamped(_carried(reason='rides on 6524'))

    assert isinstance(record, CarriedByDeferral)
    assert record.kind is DeferralKind.CARRIED_BY
    assert record.carrier_task_id == 6524
    assert record.reason == 'rides on 6524'
    assert record.deferred_at == NOW
    assert record.stamped_at == STAMPED_AT
    assert record.stamped_by == _AGENT


def test_stamped_until_condition_keeps_condition_and_expiry():
    expires_at = NOW + timedelta(days=1)

    record = _stamped(_until(condition='CI green', expires_at=expires_at.isoformat()))

    assert isinstance(record, UntilConditionDeferral)
    assert record.kind is DeferralKind.UNTIL_CONDITION
    assert record.condition == 'CI green'
    assert record.expires_at == expires_at


def test_stamped_held_by_session_stores_the_holder_identity_not_its_pid():
    record = _held_record()

    assert record.kind is DeferralKind.HELD_BY_SESSION
    assert record.holder == HOLDER
    assert record.grace_secs == DEFAULT_GRACE_SECS
    assert 'holder_pid' not in record_to_json(record)


def test_stamped_held_by_session_keeps_a_custom_grace():
    assert _held_record(grace_secs=900).grace_secs == 900


def test_legacy_unknown_may_be_unattributed_with_no_known_entry_time():
    record = _stamped({'kind': 'legacy_unknown'}, deferred_at=None, stamped_by=None)

    assert isinstance(record, LegacyUnknownDeferral)
    assert record.deferred_at is None
    assert record.stamped_by is None


@pytest.mark.parametrize(
    ('raw', 'holder', 'named'),
    [
        pytest.param(_held(holder_pid=4242), None, ['held_by_session', 'holder'], id='none'),
        pytest.param(_held(holder_pid=1234), HOLDER, ['1234', '4242'], id='other-pid'),
        pytest.param(_carried(), HOLDER, ['carried_by', '4242'], id='carried-by'),
        pytest.param(_until(), HOLDER, ['until_condition', '4242'], id='until-condition'),
    ],
)
def test_stamp_refuses_a_holder_that_does_not_fit_the_kind(raw, holder, named):
    request = parse_request(raw, now=NOW)

    with pytest.raises(ValueError) as refused:
        stamp(request, stamped_at=NOW, stamped_by=_AGENT, deferred_at=NOW, holder=holder)

    assert type(refused.value) is ValueError
    for text in named:
        assert text in str(refused.value)


_EVERY_KIND_OF_RECORD = [
    _stamped(_carried()),
    _stamped(_until(expires_at=(NOW + timedelta(days=1)).isoformat())),
    _stamped(_until()),
    _stamped(_held(holder_pid=HOLDER.pid, grace_secs=900), holder=HOLDER),
    _stamped({'kind': 'legacy_unknown'}, deferred_at=None, stamped_by=None),
    PlanningDeferral(deferred_at=STAMPED_AT, stamped_at=STAMPED_AT, stamped_by='orchestrator'),
]


@pytest.mark.parametrize('record', _EVERY_KIND_OF_RECORD, ids=lambda r: r.kind.value)
def test_record_round_trips_through_json(record):
    wire = json.loads(json.dumps(record_to_json(record)))

    restored = parse_record(wire)

    assert restored == record
    assert not isinstance(restored, InvalidRecord | None)
    assert restored.stamped_at.microsecond == STAMPED_AT.microsecond == 123456


@pytest.mark.parametrize('record', _EVERY_KIND_OF_RECORD, ids=lambda r: r.kind.value)
def test_records_are_frozen(record):
    with pytest.raises(ValidationError):
        record.reason = 'changed'


def test_no_record_is_none():
    assert parse_record(None) is None


def test_a_wrapped_record_unwraps_to_its_variant():
    record = _stamped(_carried())

    assert parse_record(DeferralRecord(record)) == record


def test_a_variant_is_already_parsed():
    record = _stamped(_carried())

    assert parse_record(record) is record


_HELD_JSON = record_to_json(_stamped(_held(holder_pid=HOLDER.pid), holder=HOLDER))


@pytest.mark.parametrize(
    'raw',
    [
        pytest.param({'kind': 'carried_by', 'carrier_task_id': 1}, id='never-stamped'),
        pytest.param({**_HELD_JSON, 'surprise': 1}, id='unknown-key'),
        pytest.param([_HELD_JSON], id='list'),
        pytest.param({**_HELD_JSON, 'holder': {**_HELD_JSON['holder'], 'pid': 0}}, id='pid-0'),
        pytest.param({**_HELD_JSON, 'grace_secs': 5}, id='grace-5'),
    ],
)
def test_malformed_record_is_invalid_with_a_detail(raw):
    parsed = parse_record(raw)

    assert isinstance(parsed, InvalidRecord)
    assert parsed.detail


# --- registration as metadata.deferral ---------------------------------------


def test_deferral_is_a_known_metadata_key():
    assert is_known_metadata_key('deferral')


def test_metadata_read_types_the_deferral_slice():
    held = _held_record()

    metadata, warnings = parse_metadata({'deferral': record_to_json(held)}, direction='read')

    assert warnings == []
    assert metadata.model_extra is not None
    assert parse_record(metadata.model_extra['deferral']) == held


def test_malformed_slice_reads_as_an_invalid_submodel():
    _, warnings = parse_metadata({'deferral': {'kind': 'carried_by'}}, direction='read')

    assert [(w.field, w.code) for w in warnings] == [('deferral', 'invalid_submodel')]


def test_malformed_slice_is_refused_on_an_enforced_write():
    with pytest.raises(ValidationError):
        parse_metadata({'deferral': {'kind': 'carried_by'}}, direction='write', enforce=True)


_TASK_METADATA_ONLY_CHILD = """
import json, sys
from shared.task_metadata import is_known_metadata_key, parse_metadata
metadata, warnings = parse_metadata(json.loads(sys.argv[1]), direction='read')
deferral = metadata.model_extra['deferral']
print(json.dumps([
    is_known_metadata_key('deferral'),
    [warning.code for warning in warnings],
    type(deferral).__name__,
    type(deferral.root).__name__,
]))
"""

_TASK_DEFERRAL_FIRST_CHILD = """
import json, sys
import shared.task_deferral
print(json.dumps(sys.modules['shared.task_metadata'].is_known_metadata_key('deferral')))
"""


def test_a_process_importing_only_task_metadata_types_the_deferral_slice():
    blob = json.dumps({'deferral': record_to_json(_held_record())})

    seen = run_python_child(_TASK_METADATA_ONLY_CHILD, blob)

    assert seen == [True, [], 'DeferralRecord', 'HeldBySessionDeferral']


def test_a_process_importing_task_deferral_first_knows_the_key():
    assert run_python_child(_TASK_DEFERRAL_FIRST_CHILD) is True
