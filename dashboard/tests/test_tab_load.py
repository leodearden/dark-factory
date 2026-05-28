"""Structural tests for the HostLoadCard component in tab_overview.jsx.

Tests parse JSX source files as text and assert structural contracts
(component definition, metric rendering, formatting, polling).
Follows the idiom established in test_tab_escalations.py and test_tab_curator.py.
"""

from __future__ import annotations

import re

import pytest
from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def tab_overview_jsx_body(_client):
    return _client.get('/static/redux/tab_overview.jsx').text


# ---------------------------------------------------------------------------
# Helper: extract a named JS/JSX function body (brace-aware)
# ---------------------------------------------------------------------------


def _extract_function_body(src: str, fn_name: str) -> str:
    """Return the body block of a ``function <fn_name>(`` declaration, braces included.

    Uses a brace-depth walk.  Only matches named ``function`` declarations —
    not arrow functions or class methods.
    Returns the empty string if the function is not found.

    This is used to scope token-presence checks to a specific function body
    rather than searching the entire file (which would give false confidence
    when a token appears in an unrelated context).
    """
    m = re.search(rf'\bfunction\s+{re.escape(fn_name)}\s*\(', src)
    if m is None:
        return ''
    start = src.find('{', m.end())
    if start == -1:
        return ''
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
# step-1 test: tab_overview.jsx is served and defines HostLoadCard
# ---------------------------------------------------------------------------


def test_tab_overview_serves_and_defines_host_load_card(tab_overview_jsx_body: str) -> None:
    """GET /static/redux/tab_overview.jsx returns 200, defines HostLoadCard,
    renders <HostLoadCard, includes the card title 'Host load', and still
    exports window.DF_OVERVIEW = { OverviewTab } (regression guard).

    Asserts:
    (a) 200 HTTP status (implicitly — fixture would fail on non-200 body retrieval).
    (b) 'function HostLoadCard(' is declared.
    (c) '<HostLoadCard' is rendered somewhere.
    (d) The literal card title text 'Host load' is present.
    (e) 'window.DF_OVERVIEW = { OverviewTab }' export is still present.
    """
    body = tab_overview_jsx_body

    # (b) HostLoadCard function declaration
    assert 'function HostLoadCard(' in body, (
        "tab_overview.jsx does not define `function HostLoadCard(` — add a named "
        'HostLoadCard function declaration to the file.'
    )

    # (c) <HostLoadCard rendered
    assert '<HostLoadCard' in body, (
        "tab_overview.jsx does not render `<HostLoadCard` — add `<HostLoadCard />` "
        'to the OverviewTab grid.'
    )

    # (d) Card title text
    assert 'Host load' in body, (
        "tab_overview.jsx does not contain the card title text 'Host load' — add it "
        "to the HostLoadCard panel-head."
    )

    # (e) Regression guard: original DF_OVERVIEW export unchanged
    assert 'window.DF_OVERVIEW = { OverviewTab }' in body, (
        "tab_overview.jsx lost the `window.DF_OVERVIEW = { OverviewTab }` export — "
        'the OverviewTab export must remain unchanged.'
    )


# ---------------------------------------------------------------------------
# step-3 test: HostLoadCard renders all nine metric rows
# ---------------------------------------------------------------------------


def test_host_load_card_renders_all_nine_metric_rows(tab_overview_jsx_body: str) -> None:
    """tab_overview.jsx must reference all 9 metric keys and (scoped to the
    HostLoadCard body) must map over a metrics list and reuse <Sparkline with
    a values prop bound to a sparkline field.

    Asserts:
    (a) All 9 KNOWN_METRICS keys appear in the file.
    (b) Scoped to HostLoadCard body: `.map(` is present (iterating over metrics).
    (c) Scoped to HostLoadCard body: `<Sparkline` with a `values=` prop bound to
        a sparkline field appears.
    """
    body = tab_overview_jsx_body

    # (a) All 9 metric keys present
    KNOWN_METRICS = [
        'psi_cpu_some_avg10',
        'psi_cpu_full_avg10',
        'psi_mem_some_avg10',
        'psi_mem_full_avg10',
        'psi_io_some_avg10',
        'psi_io_full_avg10',
        'occt_queue_depth',
        'verify_concurrency',
        'verify_rss_total_bytes',
    ]
    for key in KNOWN_METRICS:
        assert key in body, (
            f"tab_overview.jsx does not reference metric key '{key}' — add it to "
            "the METRICS config array in HostLoadCard."
        )

    # (b) and (c) Scoped to HostLoadCard body
    card_body = _extract_function_body(body, 'HostLoadCard')
    assert card_body, (
        'Could not extract HostLoadCard function body — ensure it is a named function declaration.'
    )

    assert '.map(' in card_body, (
        'HostLoadCard body does not call `.map(` — add a metrics.map(...) to render '
        'one row per metric.'
    )

    assert re.search(r'<Sparkline\b[^>]*values\s*=', card_body), (
        'HostLoadCard body does not render `<Sparkline` with a `values=` prop — '
        'add `<Sparkline values={datum.sparkline} ... />` to each metric row.'
    )


# ---------------------------------------------------------------------------
# step-5 test: HostLoadCard empty-state renders em-dash gated on null check
# ---------------------------------------------------------------------------


def test_host_load_card_empty_state_renders_em_dash(tab_overview_jsx_body: str) -> None:
    """HostLoadCard body must contain the em-dash literal — (U+2014) AND
    gate it on a null check of the current value.

    The em-dash is the sole empty representation — never a fabricated 0 or
    placeholder number (per feedback_redux_no_synthetic_data).

    Asserts (scoped to HostLoadCard body):
    (a) The em-dash literal '—' is present.
    (b) It is gated on a null check: one of `current == null`,
        `current === null`, or `?? '—'` appears nearby.
    """
    card_body = _extract_function_body(tab_overview_jsx_body, 'HostLoadCard')
    assert card_body, (
        'Could not extract HostLoadCard function body.'
    )

    # (a) Em-dash present
    assert '—' in card_body, (
        "HostLoadCard body does not contain the em-dash '—' (U+2014) — "
        "add `datum.current == null ? '—' : formatLoadValue(...)` to the value cell."
    )

    # (b) Gated on a null check
    null_check = re.search(
        r'current\s*==\s*null|current\s*===\s*null|\?\?\s*[\'\"]—[\'\"]',
        card_body,
    )
    assert null_check, (
        "HostLoadCard em-dash is not gated on a null check — "
        "use `datum.current == null ? '—' : ...` (or `?? '—'`) to guard it."
    )


# ---------------------------------------------------------------------------
# step-7 test: HostLoadCard formats each metric type correctly
# ---------------------------------------------------------------------------


def test_host_load_card_formats_each_metric_type(tab_overview_jsx_body: str) -> None:
    """tab_overview.jsx (or the formatLoadValue function body) must implement
    all three format contracts:
    - PSI: .toFixed(2) + '%' literal + some/full label derivation
    - verify_rss_total_bytes: 'GiB' literal + .toFixed(1) + divisor 1073741824
    - occt_queue_depth / verify_concurrency integers: Math.round(

    This encodes 'card renders with populated data'.

    Asserts:
    (a) PSI format: `.toFixed(2)` and `%` appear in the file.
    (b) PSI some/full: both substrings 'some' and 'full' appear in the formatting/label code.
    (c) Bytes format: 'GiB' and `.toFixed(1)` and `1073741824` appear.
    (d) Integer format: `Math.round(` appears in the file.
    """
    body = tab_overview_jsx_body

    # (a) PSI format
    assert '.toFixed(2)' in body, (
        "tab_overview.jsx does not use `.toFixed(2)` for PSI metric formatting — "
        "add `v.toFixed(2)` to the PSI format branch in formatLoadValue."
    )
    assert '%' in body, (
        "tab_overview.jsx does not include a '%' literal for PSI metric formatting — "
        "add `'%'` to the PSI format branch output."
    )

    # (b) PSI some/full labels
    assert 'some' in body, (
        "tab_overview.jsx does not reference 'some' in the formatting/label code — "
        "derive some/full from the metric key for PSI rows."
    )
    assert 'full' in body, (
        "tab_overview.jsx does not reference 'full' in the formatting/label code — "
        "derive some/full from the metric key for PSI rows."
    )

    # (c) Bytes format
    assert 'GiB' in body, (
        "tab_overview.jsx does not include 'GiB' for verify_rss_total_bytes formatting — "
        "add `' GiB'` to the bytes format branch in formatLoadValue."
    )
    assert '.toFixed(1)' in body, (
        "tab_overview.jsx does not use `.toFixed(1)` for GiB formatting — "
        "add `(v / 1073741824).toFixed(1)` to the bytes format branch."
    )
    assert '1073741824' in body, (
        "tab_overview.jsx does not include the GiB divisor `1073741824` — "
        "add `v / 1073741824` to the bytes format branch in formatLoadValue."
    )

    # (d) Integer format
    assert 'Math.round(' in body, (
        "tab_overview.jsx does not use `Math.round(` for integer metric formatting — "
        "add `Math.round(v)` to the int format branch in formatLoadValue."
    )


# ---------------------------------------------------------------------------
# step-9 test: HostLoadCard polls /api/load every 5s
# ---------------------------------------------------------------------------


def test_host_load_card_polls_api_load_every_5s(tab_overview_jsx_body: str) -> None:
    """HostLoadCard body must call fetch('/api/load'), set up setInterval with
    literal 5000, and return a cleanup that calls clearInterval.

    This is the structural realization of 'polling triggers re-fetch (mocked)'
    given no JS runtime is available.

    Asserts (scoped to HostLoadCard body):
    (a) fetch('/api/load' or "/api/load") appears.
    (b) setInterval( with literal 5000 appears.
    (c) clearInterval( appears (in the cleanup return).
    """
    card_body = _extract_function_body(tab_overview_jsx_body, 'HostLoadCard')
    assert card_body, (
        'Could not extract HostLoadCard function body.'
    )

    # (a) fetch call to /api/load
    assert re.search(r'fetch\([\'\"]/api/load', card_body), (
        "HostLoadCard body does not call `fetch('/api/load'` or `fetch(\"/api/load\"` — "
        "add a useEffect that fetches /api/load on mount and every 5s."
    )

    # (b) setInterval with 5000
    assert re.search(r'setInterval\s*\(', card_body), (
        "HostLoadCard body does not call `setInterval(` — "
        "add `setInterval(fetchLoad, 5000)` to poll every 5 seconds."
    )
    assert '5000' in card_body, (
        "HostLoadCard body does not contain the interval literal `5000` — "
        "use `setInterval(fetchLoad, 5000)` for the 5-second poll cadence."
    )

    # (c) clearInterval cleanup
    assert re.search(r'clearInterval\s*\(', card_body), (
        "HostLoadCard body does not call `clearInterval(` — "
        "add `return () => { clearInterval(id); }` as the useEffect cleanup."
    )
