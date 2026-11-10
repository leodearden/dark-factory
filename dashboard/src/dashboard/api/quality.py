"""`/api/v2/dashboard/quality` — one project's per-landing quality series.

The ``?window=`` chip counts landings, not days: ``_QUALITY_WINDOWS`` is this
route's vocabulary, parsed by ``dashboard.api.window._parse_landing_window``.
"""

from __future__ import annotations

_QUALITY_WINDOWS: dict[str, int | None] = {
    '100': 100,
    '300': 300,
    'all': None,
}
