"""Source-text contract on app.jsx's per-tab staleness render site (task 4884, #4791).

WHY SOURCE TEXT AND NOT A RENDER. `app.jsx` is served as `type="text/babel"`
and transpiled in the browser; there is no JS runtime in this project that can
import it, so the established idiom for a `.jsx` render-site contract is a
brace-aware extraction plus comment-stripped substring probes — see
`test_tab_escalation_analytics.py` and `test_charts_null_samples.py`, whose
helpers (`extract_function_body`, `strip_js_comments`) this module reuses
verbatim. Stripping comments first is load-bearing: without it these probes
also match PROSE, which has deformed production source before (see
`strip_js_comments`' docstring).

WHAT IS BEING PINNED. The 2026-08-27 incident ran 19.8h with every tab
rendering a fully-populated table whose numbers had stopped advancing. The
decision content lives in `endpoint_staleness.js` (covered by
`dashboard/tests/js/endpoint_staleness.test.mjs`) and the recorded facts in
`data.js` (covered by `data_poll.test.mjs`); what neither can see is whether
`App` actually reads them, renders them, and — the acceptance that matters
most — renders them ALONGSIDE the tab body rather than in place of it.
"""

from __future__ import annotations

import re

from _dashboard_helpers import extract_function_body, strip_js_comments

_TESTID = 'endpoint-stale-banner'


def _app_body(app_jsx_body: str) -> str:
    """`App`'s comment-stripped body."""
    body = extract_function_body(app_jsx_body, 'App')
    assert body, 'could not extract App()\'s body from app.jsx'
    return strip_js_comments(body)


def test_app_reads_notices_through_the_shared_module(app_jsx_body: str) -> None:
    """`App` gets its notices from `window.DF_ENDPOINT_STALENESS`, not from
    a second copy of the threshold arithmetic.

    Destructured at MODULE scope, the way `tab_tasks.jsx` destructures
    `window.DF_TASKS_OFFLINE_BANNER` — index.html's load order is the enforced
    contract (`test_index_html.py::test_endpoint_staleness_js_loads_before_app_jsx`),
    so no `|| {}` fallback is wanted here: a silently-absent indicator is the
    exact failure this task closes.
    """
    src = strip_js_comments(app_jsx_body)
    assert re.search(
        r'const\s*\{[^}]*staleNoticesForTab[^}]*\}\s*=\s*window\.DF_ENDPOINT_STALENESS',
        src,
    ), (
        'app.jsx must destructure staleNoticesForTab from '
        'window.DF_ENDPOINT_STALENESS at module scope, next to the existing '
        'window.DF_SHELL / window.DF_TABS / window.DF_TASKS destructures.'
    )


def test_app_passes_the_current_tab_and_the_published_stale_map(app_jsx_body: str) -> None:
    """The call is parameterised on the ACTIVE tab and on `DF_DATA.__stale`.

    A call that hardcoded a tab, or that read some other source of truth,
    would render notices for endpoints the operator is not looking at — the
    global-banner mistake `tasks_offline_banner.js` already exists to undo.
    """
    body = _app_body(app_jsx_body)
    call = re.search(r'staleNoticesForTab\(\s*\{(.*?)\}\s*\)', body, re.S)
    assert call, 'App() must call staleNoticesForTab({...})'
    args = call.group(1)

    assert re.search(r'\btab\b', args), (
        f'the call must pass the CURRENT tab id; got: {args!r}'
    )
    assert '__stale' in args, (
        'the call must read the published map DF_DATA.__stale — that is where '
        f'refreshOne records failures/lastSuccessAt; got: {args!r}'
    )


def test_notices_render_with_a_stable_testid(app_jsx_body: str) -> None:
    """Each notice renders with a stable `data-testid`.

    Follows the existing banner precedent in `tab_tasks.jsx`
    (`tasks-runtime-probe-banner`, `memory-eval-storm-banner`): a testid is
    what makes the indicator addressable from a browser check
    (#4791 acceptance 3) rather than only from a source grep.
    """
    body = _app_body(app_jsx_body)
    assert _TESTID in body, (
        f'App() must render each staleness notice with data-testid="{_TESTID}"'
    )
    assert re.search(r'staleNotices\s*\.\s*map\s*\(', body), (
        'App() must render ONE element PER notice (a .map over the notices), '
        'not a single merged banner — a tab can have several failing endpoints '
        'and each names a different one.'
    )


def test_the_tab_body_is_never_replaced_by_a_staleness_notice(app_jsx_body: str) -> None:
    """`renderTab()` stays UNCONDITIONAL, and the notices render beside it.

    THE ACCEPTANCE THAT MATTERS MOST. `refreshOne` deliberately keeps the
    prior values "so the UI does not blank out"; an indicator that replaced
    the tab body with a "data is stale" panel would throw away exactly the
    last-good payload that decision preserved. Show the stale data, marked.
    """
    body = _app_body(app_jsx_body)

    assert 'renderTab()' in body, 'App() must still call renderTab()'

    # The notices and the tab body must both live inside `.body`.
    section = re.search(
        r'className="body".*?renderTab\(\)', body, re.S,
    )
    assert section, (
        'renderTab() must still be rendered inside the .body container'
    )
    assert _TESTID in section.group(0), (
        f'the staleness notices (data-testid="{_TESTID}") must render inside '
        '.body ALONGSIDE renderTab(), not in a separate region and not '
        'instead of it.'
    )

    # No early return may sit between the notice render and renderTab().
    assert not re.search(r'return\b', section.group(0)), (
        'no early `return` may separate the staleness notices from '
        'renderTab() — the tab body is never short-circuited by a notice.'
    )


def test_no_new_timer_is_introduced(app_jsx_body: str) -> None:
    """Re-rendering rides the EXISTING refresh signals.

    `App` already re-renders on the `df-data-refresh` event the loader
    dispatches every cycle, and on the 1s `now` tick while unpaused. A second
    listener or a third timer would add load to a page whose connection budget
    is already the subject of this task's `STALE_TIMEOUT_MS` change.
    """
    body = _app_body(app_jsx_body)

    assert body.count('setInterval') == 1, (
        'App() must keep exactly ONE setInterval (the 1s `now` tick); the '
        f'staleness indicator must not add another. Found {body.count("setInterval")}.'
    )
    # Counted on addEventListener alone: the existing effect both adds and
    # removes the listener, so the bare event name legitimately appears twice.
    adds = len(re.findall(r"addEventListener\(\s*'df-data-refresh'", body))
    assert adds == 1, (
        'App() must keep exactly ONE df-data-refresh listener; the staleness '
        f'indicator rides the existing one. Found {adds}.'
    )


def test_stale_is_never_read_by_identity_comparison(app_jsx_body: str) -> None:
    """Staleness is derived from RECORDED TIMESTAMPS, never from `!==` against
    a captured seed.

    Pins the `__loaded` comment block's warning at the CONSUMER as well as the
    producer. A `text/babel` module is transpiled and evaluated after
    DOMContentLoaded while the loader's immediate first fetch can resolve
    BEFORE that, so a module-scope capture can freeze a REAL payload as the
    "seed" — and a verdict derived from comparing against it then reports
    "current" for a payload that has not moved in hours, which is precisely
    the 19.8h blind spot.
    """
    src = strip_js_comments(app_jsx_body)

    assert not re.search(r'__stale\s*(!==|===)', src), (
        'app.jsx must not compare DF_DATA.__stale by identity — derive '
        'staleness from the recorded lastSuccessAt timestamps instead.'
    )
    assert not re.search(r'(!==|===)\s*[A-Za-z_$][\w$]*__stale', src, re.I), (
        'app.jsx must not compare against a captured __stale seed.'
    )
