"""asset_skew.js is served, and loads before the app.jsx that reads it.

app.jsx destructures window.DF_ASSET_SKEW at module scope with no fallback,
so a 404 or a late load throws at app.jsx's evaluation and takes the whole
page down. The order guard alone cannot see a file that is registered but not
served, so each guard has its pair.
"""

from __future__ import annotations

from _jsx_source_helpers import assert_script_loads_before

_ASSET_SKEW_PREFIX = '/static/redux/asset_skew.js'
_APP_JSX_PREFIX = '/static/redux/app.jsx'


def test_asset_skew_js_is_served(client) -> None:
    resp = client.get(_ASSET_SKEW_PREFIX)

    assert resp.status_code == 200, f'expected 200 for {_ASSET_SKEW_PREFIX}, got {resp.status_code}'


def test_asset_skew_js_loads_before_app_jsx(index_html_body: str) -> None:
    assert_script_loads_before(
        index_html_body,
        _ASSET_SKEW_PREFIX,
        _APP_JSX_PREFIX,
        before_label='asset_skew.js',
        after_label='app.jsx',
        consumer_note='app.jsx destructures window.DF_ASSET_SKEW at top level; asset_skew.js must define it first.',
    )
