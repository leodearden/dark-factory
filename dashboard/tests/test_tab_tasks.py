"""Wiring tests for the Tasks tab JSX and its CSS.

Tests fetch tab_tasks.jsx and styles.css as text via TestClient and assert
structural contracts:  literal status strings, regex on className blocks,
presence of the train-badge selector.  Follows the idiom established in
test_tab_curator.py / test_index_html.py — no JS test runner needed.
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
def tab_tasks_jsx_body(_client):
    return _client.get('/static/redux/tab_tasks.jsx').text


@pytest.fixture(scope='module')
def styles_css_body(_client):
    return _client.get('/static/redux/styles.css').text


# ---------------------------------------------------------------------------
# step-3 tests: structural contracts on tab_tasks.jsx
# ---------------------------------------------------------------------------


def test_tab_tasks_jsx_contains_merge_deferred_literal(tab_tasks_jsx_body: str) -> None:
    """tab_tasks.jsx must contain the literal string 'merge-deferred'.

    This proves the JSX has a code path that recognises the status — e.g.
    in statusMatches, fmtAge, or the TaskDetail badge classifier.  Without
    this, a merge-deferred task that reaches the wire is either invisible
    (filtered out) or misclassified (shown with the wrong badge/pip color).
    """
    assert "'merge-deferred'" in tab_tasks_jsx_body, (
        "tab_tasks.jsx does not contain the literal string 'merge-deferred' — "
        "add it to statusMatches (active filter), fmtAge, and the TaskDetail "
        "badge classifier so the status is handled on the frontend."
    )


def test_tab_tasks_jsx_references_t_dot_train(tab_tasks_jsx_body: str) -> None:
    r"""tab_tasks.jsx must reference t\.train for the annotation render.

    The Python data layer now emits `train: {id, order} | None` on every
    active-task dict.  The JSX must reference `t.train` to surface the
    train-badge annotation.  A bare substring check is insufficient because
    `t.train` could appear in a comment; we require a word-boundary match
    (\b) to catch the real property access.
    """
    assert re.search(r'\bt\.train\b', tab_tasks_jsx_body), (
        r"tab_tasks.jsx does not contain a word-boundary match for 't.train' — "
        "add `{t.train && <span className=\"train-badge\">...</span>}` inside "
        "the TaskGraph node meta block to render the annotation."
    )


def test_tab_tasks_jsx_has_train_badge_classname(tab_tasks_jsx_body: str) -> None:
    """tab_tasks.jsx must emit a 'train-badge' className for the annotation.

    The CSS step will add a `.train-badge` rule; the JSX must produce a
    matching className so the styles are applied.  Both kebab-case variants
    (train-badge, train_badge) are accepted in case the implementation uses
    an underscore.
    """
    assert re.search(r'train[-_]badge', tab_tasks_jsx_body), (
        "tab_tasks.jsx does not contain a 'train-badge' or 'train_badge' "
        "className reference — add <span className=\"train-badge\">...</span> "
        "inside the TaskGraph node meta block."
    )


# ---------------------------------------------------------------------------
# step-5 tests: structural contracts on styles.css
# (added here in the same module; fixtures are module-scoped so re-use is free)
# ---------------------------------------------------------------------------


def test_styles_css_defines_merge_deferred_token(styles_css_body: str) -> None:
    """styles.css :root block must define a --merge-deferred CSS custom property.

    The token is referenced by the status-pip, badge, and train-badge rules.
    Without it, all three rules would fall back to the browser default (black),
    and the amber-orange PRD requirement would not be met.
    """
    assert re.search(r'--merge-deferred\s*:', styles_css_body), (
        "styles.css does not define a '--merge-deferred:' token in :root — "
        "add `--merge-deferred: oklch(0.74 0.15 60);` next to the other "
        "status-color tokens (--ok, --warn, --bad, --info)."
    )


def test_styles_css_has_merge_deferred_pip_rule(styles_css_body: str) -> None:
    """styles.css must have a .taskgraph .node.s-merge-deferred .status-pip rule.

    The class is emitted automatically by the `s-${t.status}` template in
    the TaskGraph node className.  Without a matching CSS rule the pip renders
    as the default --bg-1 (invisible on dark backgrounds).
    """
    assert re.search(
        r'\.taskgraph\s+\.node\.s-merge-deferred\s+\.status-pip',
        styles_css_body,
    ), (
        "styles.css has no '.taskgraph .node.s-merge-deferred .status-pip' rule — "
        "add it after the existing .s-deferred pip rule."
    )


def test_styles_css_pip_uses_merge_deferred_token_not_bad_or_ok(styles_css_body: str) -> None:
    """The .s-merge-deferred .status-pip rule must use var(--merge-deferred), not --bad/--ok.

    This enforces the PRD ζ₂ requirement: the merge-deferred pill is amber,
    not red (blocked) and not green (done).  We locate the rule body and
    assert it contains var(--merge-deferred) and does NOT contain var(--bad)
    or var(--ok).
    """
    # Find the rule: selector line followed by the opening brace and rule body.
    m = re.search(
        r'\.taskgraph\s+\.node\.s-merge-deferred\s+\.status-pip\s*\{([^}]*)\}',
        styles_css_body,
    )
    assert m, (
        "Could not locate the rule body for '.taskgraph .node.s-merge-deferred .status-pip' — "
        "confirm the rule is present and has a non-empty block."
    )
    body = m.group(1)
    assert 'var(--merge-deferred)' in body, (
        "The .s-merge-deferred .status-pip rule body does not contain "
        "var(--merge-deferred) — the pip will render the wrong color."
    )
    assert 'var(--bad)' not in body, (
        "The .s-merge-deferred .status-pip rule body contains var(--bad) — "
        "merge-deferred must be amber, not red."
    )
    assert 'var(--ok)' not in body, (
        "The .s-merge-deferred .status-pip rule body contains var(--ok) — "
        "merge-deferred must be amber, not green."
    )


def test_styles_css_has_train_badge_selector(styles_css_body: str) -> None:
    """styles.css must define a .train-badge selector with a non-empty rule body."""
    m = re.search(r'\.train-badge\s*\{([^}]*)\}', styles_css_body)
    assert m, (
        "styles.css has no '.train-badge { ... }' rule — "
        "add the train-badge chip style after the .badge.merge-deferred rule."
    )
    assert m.group(1).strip(), (
        "styles.css has a '.train-badge' rule but its body is empty — "
        "add display/color/border declarations for the annotation chip."
    )


def test_styles_css_has_badge_merge_deferred_rule(styles_css_body: str) -> None:
    """styles.css must define a .badge.merge-deferred rule for the TaskDetail badge."""
    assert re.search(r'\.badge\.merge-deferred\s*\{', styles_css_body), (
        "styles.css has no '.badge.merge-deferred { ... }' rule — "
        "add it after the existing .badge.muted rule so the TaskDetail status "
        "badge renders in amber instead of the default accent (blue)."
    )
