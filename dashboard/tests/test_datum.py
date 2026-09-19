"""Tests for dashboard.data.datum — the Datum envelope and its contract invariants.

Pins the envelope declared in ``plans/dashboard-one-datum-one-path-prd.md``
("The `Datum` envelope"): the five wire keys, the four states, and the
machine-checked invariants. Every timestamp here is a fixed tz-aware literal —
these tests read no clock, exactly as the code under test does not.
"""

from __future__ import annotations

import dataclasses
import enum
import json
from datetime import UTC, datetime, timedelta, timezone

import pytest
from shared.task_statuses import TaskStatus

from dashboard.data.census import build_census
from dashboard.data.datum import (
    Datum,
    DatumContractError,
    DatumInvariant,
    DatumState,
    validate_datum,
)

# A fixed measurement instant and a serving instant 30 s later, named rather
# than inlined so a reader can see the age each test intends at a glance.
AS_OF = datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC)
SERVED_AT = datetime(2026, 9, 18, 12, 0, 30, tzinfo=UTC)

WIRE_KEYS = {'value', 'as_of', 'state', 'reason', 'freshness_bound_seconds'}


def fresh_datum(value: object = 7):
    """A conforming ``fresh`` Datum carrying *value*, measured at ``AS_OF``."""
    return Datum(
        value=value,
        as_of=AS_OF,
        state=DatumState.FRESH,
        reason=None,
        freshness_bound_seconds=60,
    )


def unknown_datum(reason: str | None = 'not yet fetched'):
    """A conforming ``unknown`` Datum — no value, no measurement instant."""
    return Datum(
        value=None,
        as_of=None,
        state=DatumState.UNKNOWN,
        reason=reason,
        freshness_bound_seconds=60,
    )


def test_datum_state_is_a_str_enum():
    """DatumState members are genuine strings, so the wire needs no conversion table."""
    assert issubclass(DatumState, enum.StrEnum)
    assert DatumState.FRESH == 'fresh'


def test_datum_state_vocabulary_is_exactly_the_contract_four():
    """The PRD declares four states; a fifth would be a silent contract change."""
    assert {member.value for member in DatumState} == {
        'fresh',
        'stale',
        'unknown',
        'lower_bound',
    }


def test_datum_is_frozen():
    """A served envelope is immutable — the SPA stores and renders, never mutates."""
    datum = fresh_datum()
    with pytest.raises(dataclasses.FrozenInstanceError):
        datum.value = 8  # type: ignore[misc]


def test_to_wire_emits_exactly_the_five_contract_keys():
    """The wire shape is closed: five keys, no more and no fewer."""
    assert set(fresh_datum().to_wire()) == WIRE_KEYS


def test_to_wire_renders_state_as_the_plain_status_string():
    """A consumer reads the state without importing the Python enum."""
    wire = fresh_datum().to_wire()
    assert wire['state'] == 'fresh'
    assert type(wire['state']) is str


def test_to_wire_renders_as_of_as_an_iso_8601_utc_string():
    """`as_of` crosses the wire as text; the datetime stays server-side."""
    assert fresh_datum().to_wire()['as_of'] == '2026-09-18T12:00:00+00:00'


def test_to_wire_renders_an_unknown_datums_as_of_as_none():
    """An unknown datum has no measurement instant to render."""
    wire = unknown_datum().to_wire()
    assert wire['as_of'] is None
    assert wire['value'] is None


def test_to_wire_passes_a_plain_payload_through_unchanged():
    """A payload with no wire shape of its own is emitted verbatim."""
    assert fresh_datum(value=42).to_wire()['value'] == 42
    assert fresh_datum(value=[1, 2, 3]).to_wire()['value'] == [1, 2, 3]


def test_to_wire_carries_reason_and_freshness_bound_verbatim():
    """The producer's reason and declared bound survive the boundary unedited."""
    datum = Datum(
        value=None,
        as_of=None,
        state=DatumState.UNKNOWN,
        reason='fused-memory unreachable',
        freshness_bound_seconds=120,
    )
    wire = datum.to_wire()
    assert wire['reason'] == 'fused-memory unreachable'
    assert wire['freshness_bound_seconds'] == 120


# ---------------------------------------------------------------------------
# Invariant 1 — the unknown triad: state == 'unknown' iff value is None iff
# as_of is None. Each violation names all three values, so a reader of the
# message never has to re-derive which of the three disagreed.
# ---------------------------------------------------------------------------


def test_validate_accepts_a_conforming_fresh_datum():
    """All three of the triad agree that a measurement exists."""
    validate_datum(fresh_datum(), SERVED_AT)


def test_validate_accepts_a_conforming_unknown_datum():
    """All three of the triad agree that no measurement exists."""
    validate_datum(unknown_datum(), SERVED_AT)


def test_validate_rejects_an_unknown_datum_carrying_a_value():
    """An unknown state with a value is the exact ambiguity the envelope removes."""
    datum = Datum(
        value=17,
        as_of=None,
        state=DatumState.UNKNOWN,
        reason='partial read',
        freshness_bound_seconds=60,
    )
    with pytest.raises(DatumContractError) as excinfo:
        validate_datum(datum, SERVED_AT)
    assert excinfo.value.invariant is DatumInvariant.UNKNOWN_TRIAD
    assert repr(17) in str(excinfo.value)


def test_validate_rejects_an_unknown_datum_carrying_an_as_of():
    """A measurement instant with no measurement is equally incoherent."""
    datum = Datum(
        value=None,
        as_of=AS_OF,
        state=DatumState.UNKNOWN,
        reason='partial read',
        freshness_bound_seconds=60,
    )
    with pytest.raises(DatumContractError) as excinfo:
        validate_datum(datum, SERVED_AT)
    assert excinfo.value.invariant is DatumInvariant.UNKNOWN_TRIAD
    assert repr(AS_OF) in str(excinfo.value)


@pytest.mark.parametrize('state', [DatumState.FRESH, DatumState.STALE])
def test_validate_rejects_a_known_state_with_no_value(state):
    """A non-unknown state promises a value; None would render as a silent zero."""
    datum = Datum(
        value=None,
        as_of=AS_OF,
        state=state,
        reason='cache miss',
        freshness_bound_seconds=60,
    )
    with pytest.raises(DatumContractError) as excinfo:
        validate_datum(datum, SERVED_AT)
    assert excinfo.value.invariant is DatumInvariant.UNKNOWN_TRIAD
    assert repr(state.value) in str(excinfo.value)


@pytest.mark.parametrize('state', [DatumState.FRESH, DatumState.STALE])
def test_validate_rejects_a_known_state_with_no_as_of(state):
    """A value with no measurement instant cannot have its freshness checked."""
    datum = Datum(
        value=3,
        as_of=None,
        state=state,
        reason='cache miss',
        freshness_bound_seconds=60,
    )
    with pytest.raises(DatumContractError) as excinfo:
        validate_datum(datum, SERVED_AT)
    assert excinfo.value.invariant is DatumInvariant.UNKNOWN_TRIAD
    assert repr(None) in str(excinfo.value)


# ---------------------------------------------------------------------------
# Invariant 2 — state != 'fresh' implies a non-empty reason. A number that is
# anything but freshly measured owes the reader an explanation.
# ---------------------------------------------------------------------------

NON_FRESH_STATES = [DatumState.STALE, DatumState.LOWER_BOUND, DatumState.UNKNOWN]


def datum_in_state(state, reason):
    """A Datum in *state* carrying *reason*, conforming to the unknown triad."""
    known = state is not DatumState.UNKNOWN
    return Datum(
        value=5 if known else None,
        as_of=AS_OF if known else None,
        state=state,
        reason=reason,
        freshness_bound_seconds=60,
    )


@pytest.mark.parametrize('state', NON_FRESH_STATES)
@pytest.mark.parametrize('reason', [None, '', '   ', '\t\n'])
def test_validate_rejects_a_non_fresh_datum_without_a_reason(state, reason):
    """An empty or whitespace-only reason explains nothing, so it is not a reason."""
    with pytest.raises(DatumContractError) as excinfo:
        validate_datum(datum_in_state(state, reason), SERVED_AT)
    assert excinfo.value.invariant is DatumInvariant.REASON_REQUIRED
    assert repr(state.value) in str(excinfo.value)
    assert repr(reason) in str(excinfo.value)


@pytest.mark.parametrize('state', NON_FRESH_STATES)
def test_validate_accepts_a_non_fresh_datum_with_a_reason(state):
    """A real explanation satisfies the invariant in every non-fresh state."""
    validate_datum(datum_in_state(state, 'fused-memory unreachable'), SERVED_AT)


def test_validate_accepts_a_fresh_datum_without_a_reason():
    """A freshly measured value needs no excuse."""
    validate_datum(datum_in_state(DatumState.FRESH, None), SERVED_AT)


# ---------------------------------------------------------------------------
# Invariant 3 — state == 'fresh' implies served_at - as_of <= the declared
# bound. Scoped to 'fresh' alone: a stale or lower_bound datum is ALLOWED to
# be older than its bound, which is precisely what those states announce.
# ---------------------------------------------------------------------------

BOUND_SECONDS = 60


def datum_measured_at(as_of, state=DatumState.FRESH, reason=None):
    """A Datum in *state* measured at *as_of*, bounded at ``BOUND_SECONDS``."""
    return Datum(
        value=5,
        as_of=as_of,
        state=state,
        reason=reason,
        freshness_bound_seconds=BOUND_SECONDS,
    )


@pytest.mark.parametrize('age_seconds', [0, 1, BOUND_SECONDS - 1])
def test_validate_accepts_a_fresh_datum_inside_its_bound(age_seconds):
    """An age strictly inside the declared bound is fresh."""
    as_of = SERVED_AT - timedelta(seconds=age_seconds)
    validate_datum(datum_measured_at(as_of), SERVED_AT)


def test_validate_accepts_a_fresh_datum_exactly_at_its_bound():
    """The contract's bound is inclusive; pinned so it cannot silently become '<'."""
    as_of = SERVED_AT - timedelta(seconds=BOUND_SECONDS)
    validate_datum(datum_measured_at(as_of), SERVED_AT)


def test_validate_rejects_a_fresh_datum_one_second_past_its_bound():
    """One second over the bound is no longer fresh, whatever the producer claimed."""
    as_of = SERVED_AT - timedelta(seconds=BOUND_SECONDS + 1)
    with pytest.raises(DatumContractError) as excinfo:
        validate_datum(datum_measured_at(as_of), SERVED_AT)
    assert excinfo.value.invariant is DatumInvariant.FRESHNESS_BOUND
    message = str(excinfo.value)
    assert repr(float(BOUND_SECONDS + 1)) in message
    assert repr(BOUND_SECONDS) in message


def test_validate_rejects_a_fresh_datum_measured_after_it_was_served():
    """A negative age is a producer/clock defect and is refused, not clamped."""
    as_of = SERVED_AT + timedelta(seconds=1)
    with pytest.raises(DatumContractError) as excinfo:
        validate_datum(datum_measured_at(as_of), SERVED_AT)
    assert excinfo.value.invariant is DatumInvariant.FRESHNESS_BOUND
    assert repr(-1.0) in str(excinfo.value)


@pytest.mark.parametrize('state', [DatumState.STALE, DatumState.LOWER_BOUND])
def test_validate_accepts_a_non_fresh_datum_older_than_its_bound(state):
    """Being past the bound is what `stale` and `lower_bound` exist to say."""
    as_of = SERVED_AT - timedelta(seconds=BOUND_SECONDS * 100)
    validate_datum(datum_measured_at(as_of, state=state, reason='refresh failed'), SERVED_AT)


# ---------------------------------------------------------------------------
# Invariant 4 — every instant is tz-aware. A naive datetime is a wall-clock
# reading, not an instant: it makes the freshness subtraction raise a bare
# TypeError (which sails past the access layer's `except DatumContractError`)
# and it crosses the wire with no offset, where `new Date(...)` reads it as
# the BROWSER's local time. Both are the silent lie the envelope removes.
# ---------------------------------------------------------------------------

NAIVE = datetime(2026, 9, 18, 12, 0, 0)
BERLIN = timezone(timedelta(hours=2))


def test_validate_rejects_a_naive_as_of_as_a_contract_error_not_a_type_error():
    """`datetime.utcnow()` reaches here as a defect the caller can catch, not a crash."""
    with pytest.raises(DatumContractError) as excinfo:
        validate_datum(datum_measured_at(NAIVE), SERVED_AT)
    assert excinfo.value.invariant is DatumInvariant.TZ_AWARE
    assert 'as_of' in str(excinfo.value)


def test_validate_rejects_a_naive_served_at():
    """The serving instant is half of the same subtraction, so it is held to the same rule."""
    with pytest.raises(DatumContractError) as excinfo:
        validate_datum(fresh_datum(), NAIVE)
    assert excinfo.value.invariant is DatumInvariant.TZ_AWARE
    assert 'served_at' in str(excinfo.value)


def test_validate_names_every_naive_instant_at_once():
    """Both halves wrong is one raise, not two runs."""
    with pytest.raises(DatumContractError) as excinfo:
        validate_datum(datum_measured_at(NAIVE), NAIVE)
    message = str(excinfo.value)
    assert 'as_of' in message
    assert 'served_at' in message


def test_validate_accepts_an_aware_as_of_in_another_offset():
    """Aware is the rule, not UTC: `parse_timestamp_or_warn` preserves the source offset."""
    validate_datum(datum_measured_at(SERVED_AT.astimezone(BERLIN)), SERVED_AT)


def test_to_wire_renders_an_aware_non_utc_as_of_in_utc():
    """The PRD spells `as_of` as ISO-8601 UTC; a `+02:00` offset would contradict it."""
    wire = datum_measured_at(AS_OF.astimezone(BERLIN)).to_wire()
    assert wire['as_of'] == '2026-09-18T12:00:00+00:00'


# ---------------------------------------------------------------------------
# The nesting seam beta consumes: Datum[TaskCensus]. The envelope delegates to
# a payload that knows its own wire shape, and passes anything else through.
# ---------------------------------------------------------------------------

NINE_MEMBER_MAP = {index: member.value for index, member in enumerate(TaskStatus)}


def census_datum():
    """A `fresh` Datum wrapping a census of the nine-member fixture."""
    return Datum(
        value=build_census(NINE_MEMBER_MAP),
        as_of=AS_OF,
        state=DatumState.FRESH,
        reason=None,
        freshness_bound_seconds=60,
    )


def test_to_wire_delegates_to_a_payload_that_knows_its_own_wire_shape():
    """`value` is the census's OWN wire dict, not the dataclass object."""
    value = census_datum().to_wire()['value']
    assert isinstance(value, dict)
    assert value['views']['in_flight'] == 5
    assert value['total'] == 9


def test_a_wrapped_census_survives_a_json_round_trip():
    """The whole envelope is serialisable — the seam beta serves over HTTP."""
    wire = census_datum().to_wire()
    assert json.loads(json.dumps(wire)) == wire


def test_validate_accepts_an_envelope_wrapping_a_census():
    """Nesting changes nothing about the invariants; the payload is opaque to them."""
    validate_datum(census_datum(), SERVED_AT)


@pytest.mark.parametrize('payload', [42, [1, 2, 3], {'a': 1}], ids=['int', 'list', 'dict'])
def test_to_wire_still_passes_a_payload_with_no_wire_shape_through(payload):
    """The pass-through arm is unchanged: no to_wire means emitted verbatim."""
    assert fresh_datum(value=payload).to_wire()['value'] == payload
