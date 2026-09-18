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

from dashboard.data.datum import Datum, DatumState

# A fixed measurement instant, named rather than inlined so a reader can see
# the instant each test intends at a glance.
AS_OF = datetime(2026, 9, 18, 12, 0, 0, tzinfo=UTC)

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
