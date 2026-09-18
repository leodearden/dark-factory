"""Tests for dashboard.data.datum — the Datum envelope and its contract invariants.

Pins the envelope declared in ``plans/dashboard-one-datum-one-path-prd.md``
("The `Datum` envelope"): the five wire keys, the four states, and the three
machine-checked invariants. Every timestamp here is a fixed tz-aware literal —
these tests read no clock, exactly as the code under test does not.
"""

from __future__ import annotations

import dataclasses
import enum
from datetime import UTC, datetime

import pytest

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
