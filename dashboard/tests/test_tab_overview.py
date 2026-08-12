"""Wiring tests for the Overview tab UI in tab_overview.jsx.

Tests parse JSX source as text and assert structural contracts.
Follows the idiom established in test_tab_curator.py / test_tab_orchestrators.py /
test_index_html.py.
"""

from __future__ import annotations

import re

import pytest
from starlette.testclient import TestClient


@pytest.fixture(scope='module')
def _client():
    from dashboard.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope='module')
def tab_overview_jsx_body(_client):
    return _client.get('/static/redux/tab_overview.jsx').text


class TestOverviewTabCurrentTaskRemoved:
    """The 'Current task' column must not appear in the Overview tab."""

    def test_tab_overview_jsx_served(self, _client):
        resp = _client.get('/static/redux/tab_overview.jsx')
        assert resp.status_code == 200

    def test_overview_tab_positive_anchor_function(self, tab_overview_jsx_body):
        """File must still export OverviewTab — guards against a renamed/empty file."""
        assert 'function OverviewTab(' in tab_overview_jsx_body

    def test_current_task_td_render_ref_removed(self, tab_overview_jsx_body):
        """The JSX expression {o.current_task} must NOT appear in tab_overview.jsx."""
        assert not re.search(r'\{\s*o\.current_task\s*\}', tab_overview_jsx_body)

    def test_current_task_th_removed(self, tab_overview_jsx_body):
        """The <th>Current task</th> column header must NOT appear in tab_overview.jsx."""
        assert '<th>Current task</th>' not in tab_overview_jsx_body


class TestHostLoadCardStaleness:
    """HostLoadCard must surface a stale/offline badge when /api/load fails.

    step-19 RED: The test below fails today because the panel-head renders no
    badge.  After step-20 GREEN it passes.
    """

    def test_host_load_card_stale_badge_rendered(self, tab_overview_jsx_body):
        """HostLoadCard panel-head must render a stale/offline badge element.

        Accepts either a className containing 'stale' (e.g. className="badge stale")
        or an element with visible text '>stale<' / '>offline<'.
        This is the key behavioral contract: a failed /api/load must surface a
        visible stale/offline marker rather than frozen/blank values.
        """
        jsx = tab_overview_jsx_body
        has_stale_class = bool(re.search(r'className=["\'][^"\']*stale', jsx))
        has_stale_text = bool(re.search(r'>stale<|>offline<', jsx))
        assert has_stale_class or has_stale_text, (
            'HostLoadCard panel-head must render a stale/offline badge '
            '(e.g. <span className="badge stale">stale</span> or equivalent); '
            'currently no stale badge is rendered in tab_overview.jsx'
        )


@pytest.fixture(scope='module')
def tab_overview_jsx_code(tab_overview_jsx_body):
    """`tab_overview.jsx` with every comment stripped.

    Copied from `test_tab_memory_evals.py`'s `tab_overview_jsx_code` idiom, for
    the same reason it was created there: the assertions in this file are
    whole-file substring greps, which a MENTION in a comment satisfies just as
    well as a render site.  That false-pass mode is not hypothetical — the
    memory-evals file's `alarmed_open` / `clear` assertions once passed while
    matching only explanatory prose.

    Because the phantom branch below is necessarily accompanied by a comment
    explaining what a phantom IS (and that comment names `is_phantom` and
    `unreviewed`), greping the raw body here would assert nothing at all.

    Safe to strip naively: the source contains no `//` inside a string literal
    (no URLs) and no regex literals, so no `/`-bearing code is eaten.
    """
    return re.sub(r'/\*[\s\S]*?\*/|//[^\n]*', '', tab_overview_jsx_body)


class TestReconciliationHealthRowPhantom:
    """The Reconciliation System-health row must not paint a PHANTOM verdict red.

    A phantom is the placeholder the reconciliation judge fabricates when it
    could not parse its own model output — the run went UNREVIEWED.  It is
    stored with `severity='serious'`, so the row's `ok: sev !== 'serious'`
    renders a red `bad` dot reading `verdict: serious · halt`: a serious
    finding no judge ever made.  `get_latest_verdict` now ships `is_phantom`
    (task 3287 step-8); this is the renderer that has to consume it, or the
    user-visible defect stays in place and the payload field is dead weight.

    Measured caveat, recorded honestly: the newest verdict on the live DB
    today is an ordinary `ok`, so this is a LATENT mis-render.  It was visible
    across the 2026-07-20..29 stretch and returns the next time a phantom is
    the newest row.
    """

    def test_is_phantom_is_read_in_code_not_only_in_prose(self, tab_overview_jsx_code):
        """(a) The branch exists in render code, not just in an explanatory comment."""
        assert 'is_phantom' in tab_overview_jsx_code

    def test_phantom_row_renders_an_unreviewed_label(self, tab_overview_jsx_code):
        """(b) The operator-facing text must say the run was unreviewed.

        `verdict: serious` is the exact lie being fixed — the substitute label
        has to name what actually happened.
        """
        assert 'unreviewed' in tab_overview_jsx_code

    def test_ordinary_severity_path_survives(self, tab_overview_jsx_code):
        """(c) Positive anchor: the non-phantom branch is unchanged.

        A genuine `severity=serious` verdict must still paint red — suppressing
        that would be strictly worse than the over-report being fixed.
        """
        assert "sev !== 'serious'" in tab_overview_jsx_code

    def test_phantom_branch_does_not_paint_red(self, tab_overview_jsx_code):
        """(d) Negative guard: the phantom branch must not set `ok: false`.

        Scoped to the phantom branch body rather than the whole file, so an
        `ok: false` elsewhere in the health rows cannot satisfy or break it.
        A phantom is `warn` (yellow): the run genuinely went unreviewed, which
        is degraded, but no judge found anything serious.
        """
        match = re.search(
            r'is_phantom[\s\S]{0,400}?\}', tab_overview_jsx_code
        )
        assert match, 'no is_phantom branch found in tab_overview.jsx render code'
        branch = match.group(0)
        assert not re.search(r'\bok:\s*false\b', branch), (
            'the is_phantom branch must not render a red `bad` row — a phantom '
            f'is a fabricated placeholder, not a serious finding; got: {branch!r}'
        )
        assert re.search(r'\bwarn:\s*true\b', branch), (
            'the is_phantom branch must render `warn: true` (yellow) — the run '
            f'genuinely went unreviewed, which is degraded; got: {branch!r}'
        )
