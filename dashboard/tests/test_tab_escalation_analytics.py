"""Wiring tests for the escalation-analytics endpoint + data.js registration.

Follows the source-assertion idiom established in test_tab_escalations.py:
static text checks against data.js (no JS runtime in this project) plus a
TestClient-driven route test against the real FastAPI app.
"""

from __future__ import annotations

import re

import pytest
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Module-scoped fixtures (static data.js content) — mirrors test_tab_escalations.py
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def data_js_body(_client):
    return _client.get('/static/redux/data.js').text


# ---------------------------------------------------------------------------
# Helper: extract a named seed block from window.DF_DATA (brace-aware).
# Copied from test_tab_escalations.py — see that module for the full rationale
# (brace-depth walk; does not skip braces inside JS string literals).
# ---------------------------------------------------------------------------


def _extract_df_data_block(src: str, key: str) -> str:
    """Return the body of the ``<key>: { ... }`` seed object, braces included."""
    m = re.search(rf'{re.escape(key)}\s*:\s*\{{', src)
    if m is None:
        return ''
    start = m.end() - 1  # index of the opening `{`
    depth = 0
    for i in range(start, len(src)):
        c = src[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
    return ''


# ---------------------------------------------------------------------------
# step-17: data.js registers the escalation-analytics endpoint
# ---------------------------------------------------------------------------


def test_data_js_registers_escalation_analytics_endpoint(data_js_body: str) -> None:
    """data.js must register /api/v2/dashboard/escalation-analytics -> ['ESCALATION_ANALYTICS'].

    The entry must be present in the static (unwindowed) section of endpointsFor,
    and the empty-defaults block must initialise ESCALATION_ANALYTICS with the
    Seam-2 contract shape: generated_at/parse_failures/regime_markers/per_project
    (so applyKey has a target before the first fetch resolves).
    """
    assert '/api/v2/dashboard/escalation-analytics' in data_js_body, (
        "data.js does not contain the literal URL '/api/v2/dashboard/escalation-analytics' — "
        'add it to the unwindowed entries in endpointsFor.'
    )
    assert (
        "'ESCALATION_ANALYTICS'" in data_js_body or '"ESCALATION_ANALYTICS"' in data_js_body
    ), (
        "data.js does not reference 'ESCALATION_ANALYTICS' — add it as the mapped key "
        "for '/api/v2/dashboard/escalation-analytics' in endpointsFor."
    )
    assert '/api/v2/dashboard/escalation-analytics?window=' not in data_js_body, (
        'escalation-analytics must stay unwindowed (no ?window= query param) — the '
        'frontend windows client-side over samples/flow_daily (per the PRD).'
    )
    seed_block = _extract_df_data_block(data_js_body, 'ESCALATION_ANALYTICS')
    assert seed_block, (
        'data.js does not contain an `ESCALATION_ANALYTICS: { ... }` seed block — '
        'add the initializer to the window.DF_DATA assignment so applyKey has '
        'something to replace on each poll.'
    )
    for field_name in ('generated_at', 'parse_failures', 'regime_markers', 'per_project'):
        assert re.search(rf'\b{field_name}\s*:', seed_block), (
            f"ESCALATION_ANALYTICS seed missing key '{field_name}:' — "
            'add it to the window.DF_DATA ESCALATION_ANALYTICS initializer in data.js.'
        )
