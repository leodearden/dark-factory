"""Comment-stripped views of app.jsx's served source, for its structural contract tests.

app.jsx is ``type="text/babel"`` and cannot run under node, so its wiring is
asserted over source text. Every view here strips comments first, so prose that
mentions an identifier never satisfies or breaks a probe.
"""

from __future__ import annotations

import re

from _dashboard_helpers import destructure_bindings, extract_function_body, strip_js_comments

_MODULE_DESTRUCTURE = re.compile(r'const\s*\{([^{}]*)\}\s*=\s*window\.(DF_\w+)\s*;')


def app_code(app_jsx_body: str) -> str:
    return strip_js_comments(app_jsx_body)


def app_body(app_jsx_body: str) -> str:
    """``App()``'s body."""
    body = extract_function_body(app_code(app_jsx_body), 'App')
    assert body, "could not extract App()'s body from app.jsx"
    return body


def app_module_scope(app_jsx_body: str) -> str:
    """Everything in app.jsx outside ``App()``'s body."""
    return app_code(app_jsx_body).replace(app_body(app_jsx_body), '')


def app_module_destructures(app_jsx_body: str) -> list[tuple[str, str]]:
    """``(brace body, global name)`` for each module-scope ``const {...} = window.DF_*;``."""
    return _MODULE_DESTRUCTURE.findall(app_module_scope(app_jsx_body))


def assert_app_destructures_without_fallback(
    app_jsx_body: str, global_name: str, binding: str, *, fallback_consequence: str,
) -> None:
    """app.jsx binds *binding* from ``window.<global_name>`` at module scope, with no fallback.

    *fallback_consequence* says what a fallback would hide, for the failure message.
    """
    destructures = [
        bindings for bindings, name in app_module_destructures(app_jsx_body)
        if name == global_name
    ]
    assert destructures, f'app.jsx must destructure window.{global_name} at module scope'
    assert any(
        (binding, binding) in destructure_bindings(bindings) for bindings in destructures
    ), f'app.jsx does not bind {binding} from window.{global_name}: {destructures}'
    assert not re.search(
        rf'window\.{re.escape(global_name)}\s*(\|\||&&|\?\?)', app_code(app_jsx_body),
    ), f'a fallback on window.{global_name} turns a load-order regression into {fallback_consequence}'
