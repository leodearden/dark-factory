"""The ``?window=`` vocabulary, its one parser, and the served-window echo.

Several ``/api/v2/dashboard/*`` routes accept a ``?window=`` parameter
naming a trailing time range. This module owns what those labels mean
(:data:`_WINDOW_DAYS`), the one parse that applies a vocabulary to a request
(:func:`_parse_window`), and the echo that tells the SPA which window the
payload was actually computed over (:func:`with_window`). A label the
vocabulary does not know is declined to the caller's default, and the echo
reports both the request and what was served, so a declined request is
visible on the page instead of silently relabelled.

The parser is parameterised by vocabulary because the vocabularies differ in
exactly one respect: ``dashboard.api.burndown`` passes its own
``_BURNDOWN_WINDOWS``, whose chip offers ``90d`` where this one offers
``all``. Everything else — the decline rule, the default check, the echo —
is shared.

``_WINDOW_DAYS`` and ``_parse_window`` carry a leading underscore from when
they were private to ``app.py``; read it as vestigial, not as a private-use
signal — the move that created this module was a pure extraction, with
renaming outside its scope.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

_WINDOW_DAYS: dict[str, int] = {
    '24h': 1,
    '7d': 7,
    '30d': 30,
    'all': 3650,
}


@dataclass(frozen=True, slots=True)
class ServedWindow:
    """The window a request asked for, the window it got, and that window's days.

    ``requested`` is the raw label (or the default when none was sent);
    ``served`` is always a member of the vocabulary that parsed it, and
    ``days`` is ``served``'s length. ``requested != served`` means the request
    was declined.
    """

    requested: str
    served: str
    days: int

    def to_wire(self) -> dict[str, object]:
        return {'requested': self.requested, 'served': self.served, 'days': self.days}


def _parse_window(
    query_params: Mapping[str, str],
    default: str = '30d',
    vocabulary: Mapping[str, int] = _WINDOW_DAYS,
) -> ServedWindow:
    """Parse ``?window=`` against ``vocabulary``, declining unknown labels to ``default``.

    Raises:
        ValueError: ``default`` is not in ``vocabulary`` — the served window
            must always be one the vocabulary can serve, or the echo would lie.
    """
    if default not in vocabulary:
        raise ValueError(
            f'default window {default!r} is not in the vocabulary {sorted(vocabulary)}'
        )
    requested = query_params.get('window') or default
    served = requested if requested in vocabulary else default
    return ServedWindow(requested=requested, served=served, days=vocabulary[served])


def with_window(payload: Mapping[str, object], window: ServedWindow) -> dict[str, object]:
    """Return ``payload`` plus the ``WINDOW`` echo, leaving ``payload`` untouched."""
    return {**payload, 'WINDOW': window.to_wire()}
