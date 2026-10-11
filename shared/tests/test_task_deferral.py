"""shared.task_deferral: the caller-input contract, the stored record and its predicates.

Contract: plans/deferral-flip-condition-prd.md §5.1. The clock is a fixed NOW
moved by timedelta offsets; everything goes through the public names.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from _child_processes import spawned_children
from pydantic import ValidationError
from test_pure_stdlib_leaves import run_python_child

from shared.process_identity import Liveness, ProcessIdentity, capture, liveness
from shared.task_deferral import (
    CALLER_KINDS,
    CONDITION_MAX_CHARS,
    DEFAULT_GRACE_SECS,
    GRACE_BOUNDS,
    LAPSE_OVERDUE_AFTER_SECS,
    MIGRATION_ONLY_KINDS,
    REASON_MAX_CHARS,
    STALE_CONDITION_AFTER_SECS,
    STALE_PLANNING_AFTER_SECS,
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
    LapseCause,
    LegacyUnknownDeferral,
    LegacyUnknownRequest,
    NeedsHumanReason,
    PlanningDeferral,
    UntilConditionDeferral,
    UntilConditionRequest,
    lapse_cause,
    needs_human,
    parse_record,
    parse_request,
    parse_utc_timestamp,
    record_to_json,
    stamp,
)
from shared.task_metadata import is_known_metadata_key, parse_metadata
from shared.task_statuses import TERMINAL, TaskStatus

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


def test_migration_only_kinds_are_legacy_unknown_alone_and_no_caller_kind():
    migration_only_kinds = MIGRATION_ONLY_KINDS

    assert migration_only_kinds == {DeferralKind.LEGACY_UNKNOWN}
    assert isinstance(migration_only_kinds, frozenset)
    assert not migration_only_kinds & CALLER_KINDS


@pytest.mark.parametrize('kind', sorted(set(DeferralKind) - CALLER_KINDS - MIGRATION_ONLY_KINDS))
def test_every_kind_neither_caller_nor_migration_only_is_server_only(kind):
    with pytest.raises(DeferralInvalid) as refused:
        parse_request({'kind': kind}, now=NOW)

    assert refused.value.reason is DeferralInvalidReason.SERVER_ONLY_KIND


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


@pytest.mark.parametrize(
    ('model', 'raw', 'expected'),
    [
        pytest.param(
            UntilConditionRequest, {'condition': '  '}, _R.MISSING_FIELD, id='blank'
        ),
        pytest.param(
            HeldBySessionRequest,
            {'holder_pid': 1, 'grace_secs': 599},
            _R.GRACE_OUT_OF_RANGE,
            id='grace-short',
        ),
        pytest.param(
            HeldBySessionRequest,
            {'holder_pid': 1, 'grace_secs': 86401},
            _R.GRACE_OUT_OF_RANGE,
            id='grace-long',
        ),
        pytest.param(
            UntilConditionRequest,
            {'condition': 'c', 'expires_at': 'tomorrow'},
            _R.INVALID_VALUE,
            id='request-timestamp',
        ),
        pytest.param(
            UntilConditionDeferral,
            {'condition': 'c', 'deferred_at': None, 'stamped_at': 'tomorrow'},
            _R.INVALID_VALUE,
            id='record-timestamp',
        ),
    ],
)
def test_every_custom_field_error_is_a_deferral_invalid_reason(model, raw, expected):
    with pytest.raises(ValidationError) as refused:
        model.model_validate(raw)

    error_type = refused.value.errors()[0]['type']
    assert DeferralInvalidReason(error_type) is expected


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


# --- the stored record -------------------------------------------------------

HOLDER = ProcessIdentity(pid=4242, start_ticks=99, boot_id='b' * 8, host='h')
STAMPED_AT = NOW + timedelta(microseconds=123456)
_AGENT = 'claude-task-6524'


def _stamped(
    raw: dict[str, object],
    *,
    holder: ProcessIdentity | None = None,
    stamped_at: datetime = STAMPED_AT,
    deferred_at: datetime | None = NOW,
    stamped_by: str | None = _AGENT,
) -> Deferral:
    return stamp(
        parse_request(raw, now=stamped_at),
        stamped_at=stamped_at,
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
        *(
            pytest.param({**_HELD_JSON, 'holder': {**_HELD_JSON['holder'], name: value}}, id=label)
            for name, value, label in (
                ('pid', True, 'holder-pid-bool'),
                ('pid', str(HOLDER.pid), 'holder-pid-text'),
                ('start_ticks', float(HOLDER.start_ticks), 'holder-ticks-float'),
                ('boot_id', 7, 'holder-boot-id-int'),
            )
        ),
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
loaded_metadata = 'shared.task_metadata' in sys.modules
from shared.task_metadata import is_known_metadata_key
print(json.dumps([loaded_metadata, is_known_metadata_key('deferral')]))
"""


def test_a_process_importing_only_task_metadata_types_the_deferral_slice():
    blob = json.dumps({'deferral': record_to_json(_held_record())})

    seen = run_python_child(_TASK_METADATA_ONLY_CHILD, blob)

    assert seen == [True, [], 'DeferralRecord', 'HeldBySessionDeferral']


def test_task_deferral_loads_without_task_metadata_and_the_key_is_known_after():
    assert run_python_child(_TASK_DEFERRAL_FIRST_CHILD) == [False, True]



# --- the pure predicates -----------------------------------------------------

_US = timedelta(microseconds=1)
_EXPIRES_AT = NOW + timedelta(hours=1)
_STALE_PLANNING = timedelta(seconds=STALE_PLANNING_AFTER_SECS)
_STALE_CONDITION = timedelta(seconds=STALE_CONDITION_AFTER_SECS)
_OVERDUE = timedelta(seconds=LAPSE_OVERDUE_AFTER_SECS)

_CARRIED = _stamped(_carried(), stamped_at=NOW)
_EXPIRING = _stamped(_until(expires_at=_EXPIRES_AT.isoformat()), stamped_at=NOW)
_OPEN_ENDED = _stamped(_until(), stamped_at=NOW)
_HELD_600 = _stamped(_held(holder_pid=HOLDER.pid, grace_secs=600), holder=HOLDER, stamped_at=NOW)
_HELD_DEFAULT = _stamped(_held(holder_pid=HOLDER.pid), holder=HOLDER, stamped_at=NOW)
_PLANNING = PlanningDeferral(deferred_at=NOW, stamped_at=NOW, stamped_by='orchestrator')
_LEGACY = _stamped({'kind': 'legacy_unknown'}, stamped_at=NOW, deferred_at=None, stamped_by=None)
_TRIAGED_YESTERDAY = _stamped(_until(), stamped_at=NOW - timedelta(days=1), deferred_at=None)
_FAR_EXPIRY_STAMPED_LONG_AGO = _stamped(
    _until(expires_at=(NOW + timedelta(days=1)).isoformat()), stamped_at=NOW - timedelta(days=60)
)
_LONG_DEAD = NOW - timedelta(days=1)

_LAPSE_CASES = [
    pytest.param(_EXPIRING, _EXPIRES_AT, None, LapseCause.EXPIRED, id='expires-now'),
    pytest.param(_EXPIRING, _EXPIRES_AT - _US, None, None, id='expires-in-1us'),
    pytest.param(_OPEN_ENDED, NOW + timedelta(days=365), None, None, id='no-expiry'),
    pytest.param(_HELD_600, NOW + timedelta(days=1), None, None, id='holder-never-seen-dead'),
    pytest.param(
        _HELD_600, NOW + timedelta(seconds=600), NOW, LapseCause.HOLDER_LOST, id='grace-600-spent'
    ),
    pytest.param(_HELD_600, NOW + timedelta(seconds=600) - _US, NOW, None, id='grace-600-left'),
    pytest.param(
        _HELD_DEFAULT,
        NOW + timedelta(seconds=DEFAULT_GRACE_SECS),
        NOW,
        LapseCause.HOLDER_LOST,
        id='grace-default-spent',
    ),
    pytest.param(
        _HELD_DEFAULT,
        NOW + timedelta(seconds=DEFAULT_GRACE_SECS) - _US,
        NOW,
        None,
        id='grace-default-left',
    ),
    pytest.param(_CARRIED, NOW + timedelta(days=365), _LONG_DEAD, None, id='carried-by'),
    pytest.param(_PLANNING, NOW + timedelta(days=365), _LONG_DEAD, None, id='planning'),
    pytest.param(_LEGACY, NOW + timedelta(days=365), _LONG_DEAD, None, id='legacy-unknown'),
]


@pytest.mark.parametrize(('record', 'now', 'dead_since', 'expected'), _LAPSE_CASES)
def test_lapse_cause(record, now, dead_since, expected):
    assert lapse_cause(record, now, dead_since) is expected


def test_lapse_cases_cover_every_cause():
    covered = {case.values[-1] for case in _LAPSE_CASES} - {None}

    assert covered == set(LapseCause)


_NEEDS_HUMAN_CASES = [
    pytest.param(None, NOW, None, None, NeedsHumanReason.MISSING_RECORD, id='missing'),
    pytest.param(InvalidRecord('x'), NOW, None, None, NeedsHumanReason.INVALID_RECORD, id='bad'),
    pytest.param(_LEGACY, NOW, None, None, NeedsHumanReason.LEGACY_UNKNOWN, id='legacy'),
    pytest.param(
        _CARRIED, NOW, TaskStatus.DONE, None, NeedsHumanReason.CLOSABLE, id='carrier-done'
    ),
    pytest.param(
        _CARRIED,
        NOW,
        TaskStatus.CANCELLED,
        None,
        NeedsHumanReason.ORPHANED,
        id='carrier-cancelled',
    ),
    *(
        pytest.param(_CARRIED, NOW, status, None, None, id=f'carrier-{status}')
        for status in TaskStatus
        if status not in TERMINAL
    ),
    pytest.param(
        _CARRIED, NOW, None, None, NeedsHumanReason.CARRIER_MISSING, id='carrier-missing'
    ),
    pytest.param(_PLANNING, NOW + _STALE_PLANNING, None, None, None, id='planning-at-limit'),
    pytest.param(
        _PLANNING,
        NOW + _STALE_PLANNING + _US,
        None,
        None,
        NeedsHumanReason.STALE_PLANNING,
        id='planning-stale',
    ),
    pytest.param(_OPEN_ENDED, NOW + _STALE_CONDITION, None, None, None, id='condition-at-limit'),
    pytest.param(
        _OPEN_ENDED,
        NOW + _STALE_CONDITION + _US,
        None,
        None,
        NeedsHumanReason.STALE_CONDITION,
        id='condition-stale',
    ),
    pytest.param(_TRIAGED_YESTERDAY, NOW, None, None, None, id='condition-triaged-yesterday'),
    pytest.param(_EXPIRING, _EXPIRES_AT + _OVERDUE, None, None, None, id='expiry-at-limit'),
    pytest.param(
        _EXPIRING,
        _EXPIRES_AT + _OVERDUE + _US,
        None,
        None,
        NeedsHumanReason.EXPIRY_OVERDUE,
        id='expiry-overdue',
    ),
    pytest.param(_FAR_EXPIRY_STAMPED_LONG_AGO, NOW, None, None, None, id='expiry-ahead-old-stamp'),
    pytest.param(
        _HELD_DEFAULT, NOW, None, None, NeedsHumanReason.LIVENESS_UNKNOWN, id='holder-unknown'
    ),
    pytest.param(
        _HELD_DEFAULT, NOW, None, Liveness.DEAD, NeedsHumanReason.HOLDER_DEAD, id='holder-dead'
    ),
    pytest.param(
        _HELD_DEFAULT,
        NOW,
        None,
        Liveness.OTHER_HOST,
        NeedsHumanReason.HOLDER_OTHER_HOST,
        id='holder-other-host',
    ),
    pytest.param(
        _HELD_DEFAULT, NOW + timedelta(days=7), None, Liveness.ALIVE, None, id='alive-a-week'
    ),
    pytest.param(
        _HELD_DEFAULT, NOW + timedelta(days=365), None, Liveness.ALIVE, None, id='alive-a-year'
    ),
    pytest.param(_OPEN_ENDED, NOW, None, None, None, id='condition-ignores-liveness'),
    pytest.param(
        _CARRIED, NOW, TaskStatus.PENDING, Liveness.DEAD, None, id='carried-ignores-liveness'
    ),
]


@pytest.mark.parametrize(
    ('record', 'now', 'carrier_status', 'holder_liveness', 'expected'), _NEEDS_HUMAN_CASES
)
def test_needs_human(record, now, carrier_status, holder_liveness, expected):
    assert needs_human(record, now, carrier_status, holder_liveness) is expected


def test_needs_human_cases_cover_every_reason():
    covered = {case.values[-1] for case in _NEEDS_HUMAN_CASES} - {None}

    assert covered == set(NeedsHumanReason)


@pytest.fixture
def sleeping_child() -> Iterator[subprocess.Popen]:
    with spawned_children() as spawn:
        yield spawn(['sleep', '600'])


def test_a_live_holder_is_silent_and_a_killed_one_needs_a_human(sleeping_child):
    identity = capture(sleeping_child.pid)
    assert identity is not None
    record = _stamped(_held(holder_pid=identity.pid), holder=identity, stamped_at=NOW)
    a_week_later = NOW + timedelta(days=7)

    assert needs_human(record, a_week_later, None, liveness(identity)) is None

    sleeping_child.kill()
    sleeping_child.wait()

    assert needs_human(record, a_week_later, None, liveness(identity)) is (
        NeedsHumanReason.HOLDER_DEAD
    )
