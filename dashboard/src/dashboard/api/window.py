"""The ``?window=`` query-parameter vocabulary shared by windowed endpoints.

Several ``/api/v2/dashboard/*`` routes accept a ``?window=`` parameter
naming a trailing time range, and each turns that label into a number of
days before handing it to the data layer. This module owns that one
mapping and the parse that applies it, so every windowed endpoint agrees on
what ``7d`` means and an unknown label degrades the same way everywhere
(to the caller's own default) rather than per-handler.

``dashboard.api.burndown`` deliberately does NOT use this vocabulary: the
burndown chart offers ``90d`` where this one offers ``all``, and the two
sets are pinned as distinct by ``static/redux/app.jsx``.

Both names below carry a leading underscore from when they were private to
``app.py``, and both are now this module's entire interface: ``app.py`` and
``dashboard/api/merge_queue.py`` import them. Read the underscore as
vestigial, not as a private-use signal — the move that created this module
was a pure extraction, with renaming outside its scope.
"""

from __future__ import annotations

from collections.abc import Mapping

_WINDOW_DAYS: dict[str, int] = {
    '24h': 1,
    '7d': 7,
    '30d': 30,
    'all': 3650,
}


def _parse_window(query_params: Mapping[str, str], default: int = 30) -> int:
    """Parse the ``?window=`` query parameter and return the corresponding days int."""
    window = query_params.get('window', f'{default}d')
    return _WINDOW_DAYS.get(window, default)
