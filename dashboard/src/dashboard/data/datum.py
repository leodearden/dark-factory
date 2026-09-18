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
from datetime import datetime
from typing import Generic, TypeVar

T = TypeVar('T')


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
        as_of: When the payload was measured, tz-aware; ``None`` when
            ``UNKNOWN``. Held as a ``datetime`` so arithmetic stays exact —
            the ISO-8601 text exists only on the wire.
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
        """Render the five contract keys as JSON-serialisable values."""
        return {
            'value': self.value,
            'as_of': None if self.as_of is None else self.as_of.isoformat(),
            'state': self.state.value,
            'reason': self.reason,
            'freshness_bound_seconds': self.freshness_bound_seconds,
        }
