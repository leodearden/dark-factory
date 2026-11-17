"""Contract for ``_browser_abort.browser_abort_ms``, the budget tests' ceiling.

It reports the delay after which data.js's production poll aborts one
endpoint's fetch, by running ``dashboard/tests/js/_abort_delay_probe.mjs``.
"""

from __future__ import annotations

import pytest
from _browser_abort import browser_abort_ms


def test_a_windowed_endpoint_reports_a_positive_abort_delay() -> None:
    abort_ms = browser_abort_ms('/api/v2/dashboard/merge-queue')

    assert isinstance(abort_ms, int)
    assert abort_ms > 0


def test_an_endpoint_data_js_does_not_poll_fails_naming_the_polled_paths() -> None:
    with pytest.raises(AssertionError) as excinfo:
        browser_abort_ms('/api/v2/dashboard/no-such-endpoint')

    message = str(excinfo.value)
    assert '/api/v2/dashboard/no-such-endpoint' in message
    assert '/api/v2/dashboard/tasks' in message, 'the failure must list the paths data.js does poll'
