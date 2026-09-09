"""Wiring tests for the Recon tab and its rail badge (task 5320).

THE DEFECT. `runs.status` is written by exactly one place —
fused-memory/src/fused_memory/reconciliation/journal.py::ReconciliationJournal
— which defaults the column to 'running' and completes a run as 'completed',
'failed' or 'interrupted'. The dashboard tested for two spellings the store
has never written: the rail badge counted `'failed' || 'partial'` (a dead
disjunct, and 'interrupted' uncounted), and the success-rate tile counted
`x.status === 'success'`, which is always zero — so the tile read 0% forever
while reconciliation was in fact succeeding.

THE SPLIT. The counting is pure and lives in recon_status.js, covered
executably by dashboard/tests/js/recon_status.test.mjs. What THAT suite
structurally cannot see is whether app.jsx and tabs.jsx actually CALL it, or
whether they quietly go on restating status literals of their own. That is
what this module asserts — the same division test_tab_tasks_status_counts.py
documents for task_status_counts.js.

Deliberately a NEW module. test_app.py is scoped to app-shell concerns and
test_tab_orchestrators.py to the Orchestrators tab; neither covers the Recon
surface, and this task edits app.jsx and tabs.jsx, both of which are shared
merge surfaces already. A separate module keeps the recon assertions from
being entangled with either.

Tests parse JSX/CSS source as text and assert structural contracts. Actual
rendering must be visually verified — these are source-structure assertions.
"""

from __future__ import annotations

import re

import pytest
from _dashboard_helpers import extract_function_body, strip_js_comments

# The store's vocabulary plus the two spellings it has never written. Any
# `status === '<one of these>'` left in the recon UI means a call site is
# restating the vocabulary instead of consuming recon_status.js — the
# condition that let two sites in one file disagree with each other.
RECON_STATUS_LITERALS = (
    'running',
    'completed',
    'failed',
    'interrupted',
    'success',
    'partial',
)


@pytest.fixture(scope='module')
def app_jsx_code(app_jsx_body):
    """app.jsx with comments blanked — what the assertions run on.

    A comment can satisfy or falsify an assertion either way: an absence
    assertion ("'partial' must not appear") is broken by the very comment
    explaining why it must not, and a presence assertion can be met by prose
    mentioning the token instead of by code doing it.
    """
    return strip_js_comments(app_jsx_body)


def _extract_object_literal(code: str, name: str) -> str:
    """Return the brace-delimited body of ``const <name> = { ... }``.

    Walks forward from the opening ``{`` counting brace depth, mirroring
    test_tab_scheduler.py::_extract_css_rule_block. Does not skip braces
    inside string literals — acceptable here because the rail-count literal
    embeds no brace characters in quoted values. Raises rather than returning
    '' on a miss: an empty slice would make every assertion over it pass
    vacuously, the permanent false GREEN extract_function_body also refuses.
    """
    match = re.search(r'const\s+' + re.escape(name) + r'\s*=\s*\{', code)
    assert match is not None, (
        f'no `const {name} = {{` object literal in this source — it was '
        f'renamed or restructured, and an assertion over an empty slice '
        f'would pass vacuously.'
    )
    start = match.end() - 1
    depth = 0
    for i in range(start, len(code)):
        if code[i] == '{':
            depth += 1
        elif code[i] == '}':
            depth -= 1
            if depth == 0:
                return code[start:i + 1]
    raise AssertionError(f'`const {name} = {{` is never closed')


class TestReconRailBadgeWiring:
    """DEFECT 1 — the rail badge must count the store's real terminal-failure
    states ('failed' + 'interrupted') via recon_status.js, not 'failed' plus a
    spelling that is never written."""

    def test_app_jsx_served(self, _client):
        resp = _client.get('/static/redux/app.jsx')
        assert resp.status_code == 200

    def test_destructures_recon_status_at_top_level(self, app_jsx_body):
        """app.jsx must destructure window.DF_RECON_STATUS.

        Whole-file scope on purpose: top-level destructures sit outside every
        component function, so scoping this to a component body would never
        match.
        """
        match = re.search(
            r'const\s*\{([^}]*)\}\s*=\s*window\.DF_RECON_STATUS\s*;',
            app_jsx_body,
        )
        assert match is not None, (
            'app.jsx does not destructure window.DF_RECON_STATUS at top level '
            '— recon_status.js is loaded by index.html but unused, so the rail '
            'badge is still counting a status vocabulary of its own.'
        )
        names = {n.strip() for n in match.group(1).split(',') if n.strip()}
        assert 'reconRunCounts' in names

    def test_rail_recon_count_comes_from_the_pure_module(self, app_jsx_code):
        """The railCounts `recon:` entry must call reconRunCounts and read
        `.unsuccessful`.

        Scoped to the railCounts literal so a call elsewhere in app.jsx cannot
        satisfy it. `.unsuccessful` specifically, not any bucket: the badge
        carries an ATTENTION signal, so folding in-flight runs into it would
        make a busy healthy system and a failing one render the same digit.
        """
        rail = _extract_object_literal(app_jsx_code, 'railCounts')

        recon_entry = re.search(r'\brecon\s*:\s*([^\n]*)', rail)
        assert recon_entry is not None, 'railCounts has no `recon:` entry'
        entry = recon_entry.group(1)

        assert 'reconRunCounts(' in entry, (
            f'railCounts.recon does not call reconRunCounts(...): {entry!r} — '
            'the badge must derive from the module the node suite covers.'
        )
        assert '.unsuccessful' in entry, (
            f'railCounts.recon does not read `.unsuccessful`: {entry!r} — the '
            'badge counts terminal FAILURES (failed + interrupted); any other '
            'bucket changes what the number means to an operator.'
        )

    def test_dead_partial_disjunct_is_gone(self, app_jsx_code):
        """`'partial'` must not appear anywhere in app.jsx.

        Absence, not bypass: a status the journal has never written is a
        literal no reader can evaluate, and leaving it in place is what made
        the badge look like it already handled more than 'failed'.
        """
        assert "'partial'" not in app_jsx_code, (
            "app.jsx still contains the literal 'partial' — the "
            'reconciliation journal has never written that status.'
        )

    def test_no_recon_status_literal_is_restated_in_app_jsx(self, app_jsx_code):
        """app.jsx must not compare a status against any recon run-status
        literal.

        The vocabulary is consumed from recon_status.js, never restated. Note
        this deliberately does NOT forbid `status === 'in-progress'` and the
        other TASK statuses on the neighbouring railCounts line: those belong
        to a different vocabulary with a different writer, and collapsing the
        two would be a change this task was not asked to make.
        """
        restated = [
            lit for lit in RECON_STATUS_LITERALS
            if re.search(r'status\s*===\s*' + re.escape(f"'{lit}'"), app_jsx_code)
        ]
        assert restated == [], (
            f'app.jsx compares a status against {restated} — the recon run '
            'vocabulary must come from recon_status.js so the rail badge and '
            'the Recon tab cannot drift apart from each other.'
        )
