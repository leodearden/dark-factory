"""The ``Datum`` envelope — every dashboard number carries its own provenance.

Declared by ``plans/dashboard-one-datum-one-path-prd.md``, section "The
``Datum`` envelope": a value travels with the instant it was measured, how
fresh that makes it, and — whenever it is anything but ``fresh`` — the
producer's verbatim reason. A consumer therefore never has to infer from a
zero whether a number is measured, stale or simply unavailable.

A ``Datum`` is produced SERVER-SIDE ONLY. The SPA stores and renders one; it
never constructs or mutates one, which is why the dataclass is frozen and why
``to_wire()`` is the single place the wire spelling of each field is known.

This module reads no clock. ``validate_datum`` takes the serving instant as a
parameter (see its docstring), so the freshness invariant is checked against
the one ``served_at`` the payload actually carries rather than against whatever
the clock says when the validator happens to run.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Generic, Protocol, TypeVar, runtime_checkable

T = TypeVar('T')


@runtime_checkable
class WireShaped(Protocol):
    """A payload that knows its own wire shape.

    :meth:`Datum.to_wire` delegates to this when the payload satisfies it. A
    declared Protocol rather than a bare ``hasattr`` check because it is
    self-documenting and pyright-visible.
    """

    def to_wire(self) -> dict[str, object]: ...


class DatumState(enum.StrEnum):
    """How much the accompanying value can be trusted.

    ``LOWER_BOUND`` is a measured value known to under-report — a windowed
    count, for instance, where rows outside the window were never read.
    """

    FRESH = 'fresh'
    STALE = 'stale'
    UNKNOWN = 'unknown'
    LOWER_BOUND = 'lower_bound'


@dataclass(frozen=True, slots=True)
class Datum(Generic[T]):
    """A value plus the provenance a consumer needs to render it honestly.

    Attributes:
        value: The payload, or ``None`` when the state is ``UNKNOWN``.
        as_of: When the payload was measured; ``None`` when ``UNKNOWN``.
            Held as a ``datetime`` so arithmetic stays exact — the ISO-8601
            text exists only on the wire. Must be tz-aware, which
            :func:`validate_datum` enforces rather than assuming.
        state: How fresh the measurement is.
        reason: The producer's verbatim explanation; required for every state
            but ``FRESH``.
        freshness_bound_seconds: The age past which this datum's producer
            declares the value no longer fresh. Declared beside the access
            implementation that shapes the datum, not here.
    """

    value: T | None
    as_of: datetime | None
    state: DatumState
    reason: str | None
    freshness_bound_seconds: int

    def to_wire(self) -> dict[str, object]:
        """Render the five contract keys as JSON-serialisable values.

        A payload satisfying :class:`WireShaped` renders itself; anything else
        passes through unchanged.

        ``as_of`` is normalised to UTC, because the contract spells it as
        ISO-8601 UTC and ``shared.timestamps.parse_timestamp_or_warn``
        explicitly preserves a source offset instead of converting it — so a
        producer can hand us an honest ``+02:00`` instant that must still
        cross the wire as ``+00:00``. This presumes a datum that passed
        :func:`validate_datum`: on a NAIVE datetime ``astimezone`` would
        silently read the SERVER's local zone, which is why tz-awareness is a
        checked invariant rather than a docstring promise.

        Note that ``datum.py`` imports NOTHING from ``census.py`` — the
        dependency runs the other way. The Protocol is what keeps the envelope
        and its payload families orthogonal: a new payload family is added by
        writing a ``to_wire()`` beside its own type, never by adding a branch
        here.
        """
        return {
            'value': self.value.to_wire() if isinstance(self.value, WireShaped) else self.value,
            'as_of': None if self.as_of is None else self.as_of.astimezone(UTC).isoformat(),
            'state': self.state.value,
            'reason': self.reason,
            'freshness_bound_seconds': self.freshness_bound_seconds,
        }


class DatumInvariant(enum.StrEnum):
    """The contract invariant a :class:`DatumContractError` reports.

    Carried as a structured field so a caller recovers WHICH rule broke
    without parsing the message — the message is free to be reworded.
    """

    UNKNOWN_TRIAD = 'unknown_triad'
    TZ_AWARE = 'tz_aware'
    REASON_REQUIRED = 'reason_required'
    FRESHNESS_BOUND = 'freshness_bound'


class DatumContractError(ValueError):
    """A ``Datum`` violates one of the envelope's declared invariants.

    Subclasses ``ValueError`` so an unprepared caller's ``except ValueError``
    still catches it, while a prepared one reads :attr:`invariant`.
    """

    def __init__(self, invariant: DatumInvariant, message: str) -> None:
        super().__init__(message)
        self.invariant = invariant


def validate_datum(datum: Datum, served_at: datetime) -> None:
    """Raise :class:`DatumContractError` if *datum* breaks a declared invariant.

    Args:
        datum: The envelope to check.
        served_at: The instant the payload carrying *datum* is being shaped.
            Required and injected rather than read from the clock here: the
            contract's freshness invariant is about the one ``served_at`` the
            payload actually carries, not about whenever this runs. Only the
            freshness invariant reads it.

    Every instant is required to be TZ-AWARE. A naive one is a wall-clock
    reading rather than an instant: it makes the freshness subtraction below
    raise a bare ``TypeError``, which sails straight past the access layer's
    ``except DatumContractError`` and crashes payload shaping instead of
    degrading it, and it crosses the wire with no offset, where the SPA's
    ``new Date(...)`` reads it as the BROWSER's local time. Note that
    ``datetime.utcnow()`` — the usual source of one — is NOT caught by
    ``test_clock_discipline.py``'s ``.now(...)`` matcher.

    A NEGATIVE age — a measurement instant after *served_at*, which means a
    producer or a clock is wrong — is REFUSED under the freshness invariant
    rather than clamped to zero. Clamping would let a skewed producer's value
    render as freshly measured, which is the class of silent lie the envelope
    exists to remove.

    Raises:
        DatumContractError: Naming the invariant and the offending values.
    """
    is_unknown = datum.state is DatumState.UNKNOWN
    if is_unknown != (datum.value is None) or is_unknown != (datum.as_of is None):
        expectation = (
            "state is 'unknown', so both value and as_of must be None"
            if is_unknown
            else "state is not 'unknown', so neither value nor as_of may be None"
        )
        raise DatumContractError(
            DatumInvariant.UNKNOWN_TRIAD,
            f'state, value and as_of disagree on whether a measurement exists: '
            f'state={datum.state.value!r}, value={datum.value!r}, '
            f'as_of={datum.as_of!r} ({expectation})',
        )

    naive = [
        (name, moment)
        for name, moment in (('as_of', datum.as_of), ('served_at', served_at))
        if moment is not None and moment.utcoffset() is None
    ]
    if naive:
        raise DatumContractError(
            DatumInvariant.TZ_AWARE,
            f'every instant on a datum must be tz-aware, so it names one moment '
            f'rather than a local-clock reading: '
            f'{", ".join(f"{name}={moment!r}" for name, moment in naive)}',
        )

    if datum.state is not DatumState.FRESH and not (datum.reason or '').strip():
        raise DatumContractError(
            DatumInvariant.REASON_REQUIRED,
            f'a datum in state {datum.state.value!r} must carry a non-empty reason, '
            f'got reason={datum.reason!r}',
        )

    if datum.state is DatumState.FRESH and datum.as_of is not None:
        age_seconds = (served_at - datum.as_of).total_seconds()
        if not 0 <= age_seconds <= datum.freshness_bound_seconds:
            raise DatumContractError(
                DatumInvariant.FRESHNESS_BOUND,
                f'a fresh datum must be no older than the bound its producer '
                f'declared: age={age_seconds!r}s, '
                f'freshness_bound_seconds={datum.freshness_bound_seconds!r}, '
                f'as_of={datum.as_of!r}, served_at={served_at!r}',
            )
