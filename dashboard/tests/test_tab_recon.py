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
documents for task_status_counts.js. The stylesheet half of the contract
(TestBadgeTonesAreStyled) lives here for the same reason: styles.css is
reachable from the test client and not from a node process.

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

# The burst-AGENT states, a separate vocabulary rendered by ReconTab's Burst
# state table (`b.state`, not a run's status). 'running' belongs to both, so a
# whole-body ban on the string would forbid correct code this task has no
# business touching; the overlap is checked by USE instead — see
# TestReconStatusLiteralsAreGone.
BURST_STATE_LITERALS = ('bursting', 'cooling', 'running', 'idle')


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
    """DEFECT 1 — the rail badge must count the rows an operator should act on,
    via recon_status.js, instead of 'failed' plus a spelling the store has
    never written."""

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
        assert {'reconRunCounts', 'reconAttentionCount'} <= names, (
            f'app.jsx destructures {sorted(names)} from window.DF_RECON_STATUS '
            '— the rail badge needs both: the counts bucket the window, and '
            'reconAttentionCount says which of those buckets is an alarm.'
        )

    def test_rail_recon_count_comes_from_the_pure_module(self, app_jsx_code):
        """The railCounts `recon:` entry must be reconAttentionCount's answer.

        Scoped to the railCounts literal so a call elsewhere in app.jsx cannot
        satisfy it. The badge asks "how many rows should an operator go and
        look at?", which is NOT the same question as the rate tile's "how many
        did not succeed?": journal.py writes 'interrupted' whenever a run's
        process died, and this fleet is restarted routinely, so an interrupted
        row is a healthy-fleet artefact. Counting it here would sit the badge
        permanently nonzero — the old under-count traded for a standing false
        alarm. recon_status.js answers both questions separately; the badge
        must consume the one it means rather than arithmetic of its own.
        """
        rail = _extract_object_literal(app_jsx_code, 'railCounts')

        recon_entry = re.search(r'\brecon\s*:\s*([^\n]*)', rail)
        assert recon_entry is not None, 'railCounts has no `recon:` entry'
        entry = recon_entry.group(1)

        assert 'reconRunCounts(' in entry, (
            f'railCounts.recon does not call reconRunCounts(...): {entry!r} — '
            'the badge must derive from the module the node suite covers.'
        )
        assert 'reconAttentionCount(' in entry, (
            f'railCounts.recon does not call reconAttentionCount(...): '
            f'{entry!r} — the badge is an attention signal, and what deserves '
            'attention is the module\'s call, not a bucket picked here.'
        )
        assert '.unsuccessful' not in entry, (
            f'railCounts.recon reads `.unsuccessful`: {entry!r} — that bucket '
            "includes 'interrupted', which this fleet produces routinely, so "
            'the badge would never return to zero on a healthy system.'
        )

    def test_dead_partial_disjunct_is_gone(self, app_jsx_code):
        """`'partial'` must not appear in railCounts.

        Absence, not bypass: a status the journal has never written is a
        literal no reader can evaluate, and leaving it in place is what made
        the badge look like it already handled more than 'failed'.
        """
        rail = _extract_object_literal(app_jsx_code, 'railCounts')
        assert "'partial'" not in rail, (
            "railCounts still contains the literal 'partial' — the "
            'reconciliation journal has never written that status.'
        )

    def test_no_recon_status_literal_is_restated_in_rail_counts(self, app_jsx_code):
        """railCounts must not compare a status against a recon run-status
        literal.

        The vocabulary is consumed from recon_status.js, never restated.
        Scoped to the railCounts literal, NOT to app.jsx as a whole: these
        words are not exclusive to reconciliation — tabs.jsx already spells an
        ORCHESTRATOR run 'running' / 'completed' — and app.jsx is the shell
        that holds a count for every tab, so a future merge or escalation
        count legitimately writing `d.status === 'completed'` would otherwise
        fail a RECON test with a message blaming the recon vocabulary. Even
        within this scope the TASK statuses on the neighbouring line
        (`'in-progress'`, `'blocked'`, `'pending'`) are deliberately spared:
        different vocabulary, different writer.
        """
        rail = _extract_object_literal(app_jsx_code, 'railCounts')
        restated = [
            lit for lit in RECON_STATUS_LITERALS
            if re.search(r'status\s*===\s*' + re.escape(f"'{lit}'"), rail)
        ]
        assert restated == [], (
            f'railCounts compares a status against {restated} — the recon run '
            'vocabulary must come from recon_status.js so the rail badge and '
            'the Recon tab cannot drift apart from each other.'
        )


@pytest.fixture(scope='module')
def recon_tab_code(tabs_jsx_body):
    """ReconTab's comment-stripped body — the scope every tabs.jsx assertion
    below runs in.

    tabs.jsx is ~1400 lines and defines nine tab components; unscoped, a
    token from MemoryTab or MergeTab could satisfy a presence assertion and
    any of them could falsify an absence one. extract_function_body raises on
    a miss rather than returning '', so this fixture cannot hand the tests an
    empty slice that passes everything vacuously.
    """
    return strip_js_comments(extract_function_body(tabs_jsx_body, 'ReconTab'))


def _extract_stat_tile(code: str, label: str) -> str:
    """Return the source of the self-closing ``<ST label="<label>" ... />``
    element.

    Scopes a tile assertion to ONE tile: the Recon strip renders several, and
    an unscoped search for `reconSuccessPct(` would be satisfied by any of
    them. Raises on a miss for the same reason _extract_object_literal does.
    """
    match = re.search(
        r'<ST\s+label="' + re.escape(label) + r'"(.*?)/>',
        code,
        re.DOTALL,
    )
    assert match is not None, (
        f'no self-closing <ST label="{label}" ... /> tile in ReconTab — it was '
        f'renamed or restructured, and an assertion over an empty slice would '
        f'pass vacuously.'
    )
    return match.group(0)


class TestReconSuccessRateWiring:
    """DEFECT 2 — the success-rate tile counted `x.status === 'success'`, a
    spelling the journal never writes, so it read 0% forever."""

    def test_tabs_jsx_destructures_recon_status_at_top_level(self, tabs_jsx_body):
        """Whole-file scope: top-level destructures sit outside every
        component body, so scoping this to ReconTab would never match."""
        match = re.search(
            r'const\s*\{([^}]*)\}\s*=\s*window\.DF_RECON_STATUS\s*;',
            tabs_jsx_body,
        )
        assert match is not None, (
            'tabs.jsx does not destructure window.DF_RECON_STATUS at top level.'
        )
        names = {n.strip() for n in match.group(1).split(',') if n.strip()}
        assert {'reconRunCounts', 'reconSuccessPct', 'reconStatusTone'} <= names, (
            f'tabs.jsx destructures {sorted(names)} from window.DF_RECON_STATUS '
            '— ReconTab needs all three: the counts drive the tiles, the pct '
            'drives the rate, the tone drives the Recent Runs badges.'
        )

    def test_counts_are_derived_once_over_the_unfiltered_window(self, recon_tab_code):
        """ReconTab must call reconRunCounts once, bind it to `counts`, and
        hand it the UNFILTERED `r.runs`.

        ONE derivation feeding every tile is the structural point: this file
        previously held two independent status filters that disagreed with
        each other (the tile's 'success' and the badge ternary's
        'success' || 'completed'), and a second call site is how that returns.

        The window is `r.runs`, not the `runs` local — which is r.runs
        narrowed by projectFilter and search. The tile strip reports the
        store, matching its three untouched neighbours (Buffered events,
        Active agents, Last full run); the Recent runs table below reports
        the filter the operator set and headlines itself "N matching". Two
        deliberately different questions, which is exactly why the ARGUMENT
        has to be pinned here: swapping it switches every tile to the
        filtered population in one word, invisibly to every other assertion
        in this module. Measured, not inferred — with the call mutated to
        `reconRunCounts(runs)`, all 20 tests passed.
        """
        calls = re.findall(r'reconRunCounts\s*\(', recon_tab_code)
        assert len(calls) == 1, (
            f'ReconTab calls reconRunCounts {len(calls)} times — every tile '
            'must read the SAME counts object, or two tiles can disagree '
            'about the same window.'
        )
        bound = re.search(
            r'const\s+counts\s*=\s*reconRunCounts\s*\(\s*(.*?)\s*\)', recon_tab_code
        )
        assert bound is not None, (
            'ReconTab does not bind reconRunCounts(...) to `const counts`.'
        )
        window = bound.group(1)
        assert re.search(r'\br\.runs\b', window), (
            f'the counts are derived over {window!r}, not the store-wide '
            'r.runs — the tile strip would report a narrowed population while '
            'standing beside three neighbours that report the store.'
        )
        assert not re.search(r'(?<![\w.])runs\b', window), (
            f'the counts are derived over the filtered `runs` local '
            f'({window!r}) — that is the window of the Recent runs table, '
            'headlined "N matching", not the window of the tile strip.'
        )

    def test_success_rate_tile_value_comes_from_recon_success_pct(self, recon_tab_code):
        """The rate the tile renders must be the module's, traced in two hops.

        The tile reads a local, and that local is bound to reconSuccessPct —
        deliberately not inlined into the tile, because `datum` and `unit`
        both read it and inlining would call it twice. Asserting the two
        hops separately pins the provenance without dictating which of the
        two spellings the component uses.

        The prop is `datum=` since task 5588: the tile is handed the value
        wrapped in its provenance (plainDatum) rather than bare, so that a
        rate which has not been fetched renders an em-dash instead of a zero.
        """
        tile = _extract_stat_tile(recon_tab_code, 'Run success rate')

        bound = re.search(
            r'const\s+(\w+)\s*=\s*reconSuccessPct\s*\(\s*counts\s*\)', recon_tab_code
        )
        assert bound is not None, (
            'ReconTab does not bind reconSuccessPct(counts) — the rate must '
            'come from the module the node suite covers, computed over the '
            'same counts object every other tile reads.'
        )
        assert re.search(r'datum=\{[^}]*\b' + bound.group(1) + r'\b', tile), (
            f'the Run success rate tile does not render {bound.group(1)!r}, the '
            f'local bound to reconSuccessPct(counts): {tile!r}'
        )

    def test_the_success_tally_is_not_recomputed_by_hand(self, recon_tab_code):
        """`successCount` — the local that filtered for 'success' — must be gone.

        Only the LOCAL is checked here. That no comparison against 'success'
        (or any other status literal) survives is the strictly stronger rule
        asserted once, by
        TestReconStatusLiteralsAreGone::test_no_run_status_literal_survives_anywhere_in_recon_tab;
        restating it here would be a second copy that can never fail while
        that one passes.
        """
        assert 'successCount' not in recon_tab_code, (
            'ReconTab still computes `successCount` by hand — the success '
            'tally belongs to reconRunCounts.'
        )

    def test_rate_denominator_is_terminal_runs_not_the_whole_window(
        self, recon_tab_code
    ):
        """The rate must not be divided by the window size.

        Fixing only the 'success' literal would leave a rate that DIPS every
        time reconciliation gets busy, because an in-flight run would sit in
        the denominator having neither succeeded nor failed — a second,
        subtler version of the same bug. `totalRuns` was the old local for it.
        """
        assert 'totalRuns' not in recon_tab_code, (
            'ReconTab still computes `totalRuns` — the success rate denominator '
            'must be the TERMINAL count (counts.terminal), not the window size.'
        )
        assert not re.search(r'/\s*(r\.)?runs\.length', recon_tab_code), (
            'ReconTab still divides by the window size — an in-flight run '
            'would depress the success rate.'
        )

    def test_success_rate_hint_surfaces_the_denominator_and_the_residue(
        self, recon_tab_code
    ):
        """The tile's hint must report what the rate was computed over.

        `counts.terminal` and `counts.inFlight` make the denominator legible
        on the tile itself, so an operator can tell "no run has finished yet"
        from "everything failed". `counts.unknown` is what keeps THIS defect
        from recurring silently: a status the store grows later shows up as a
        number in the UI instead of vanishing into a filter that matches
        nothing.
        """
        tile = _extract_stat_tile(recon_tab_code, 'Run success rate')
        hint = re.search(r'hint=\{(.*?)\}\s*\n', tile, re.DOTALL)
        assert hint is not None, f'the Run success rate tile has no hint: {tile!r}'
        hint_src = hint.group(1)

        for field in ('counts.terminal', 'counts.inFlight', 'counts.unknown'):
            assert field in hint_src, (
                f'the Run success rate hint does not surface {field}: '
                f'{hint_src!r}'
            )


@pytest.fixture(scope='module')
def styles_css_body(_client):
    return _client.get('/static/redux/styles.css').text


def _extract_css_rule_block(css: str, selector: str) -> str:
    """Return the body of the first matching CSS rule block, braces included.

    Copied from test_tab_scheduler.py rather than imported: a cross-test-module
    import would couple two otherwise-independent suites. Walks forward from
    the opening ``{`` counting depth; does not skip braces inside string
    literals, which this stylesheet does not use. Returns '' if absent — the
    caller below asserts on that explicitly rather than on a slice.
    """
    match = re.search(re.escape(selector) + r'\s*\{', css)
    if match is None:
        return ''
    start = match.end() - 1
    depth = 0
    for i in range(start, len(css)):
        if css[i] == '{':
            depth += 1
        elif css[i] == '}':
            depth -= 1
            if depth == 0:
                return css[start:i + 1]
    return ''


class TestReconInProgressTile:
    """DEFECT 3 — none of the four tiles counted 'running', so a tab watching
    a reconciliation actively in flight read as idle."""

    def test_in_progress_tile_reads_the_shared_counts(self, recon_tab_code):
        """The tile's value must come off the same `counts` object.

        Re-deriving it with a local filter would put a second status
        vocabulary back into this file — the exact condition that let the tile
        and the badge disagree.

        Read through `datum=` since task 5588: the count is handed over wrapped
        in its provenance (plainDatum) rather than bare, so a tab whose recon
        endpoint has not answered shows an em-dash rather than a confident 0.
        """
        tile = _extract_stat_tile(recon_tab_code, 'In progress')
        assert re.search(r'datum=\{\s*plainDatum\(\s*counts\.inFlight\s*,', tile), (
            f'the In progress tile does not render counts.inFlight: {tile!r}'
        )

    def test_tile_row_declares_five_columns(self, recon_tab_code):
        """Five tiles, one row.

        Left at cols-4 the fifth tile wraps onto a second row at quarter
        width, which reads as a rendering fault rather than a layout choice.
        """
        assert 'col-span-12 grid cols-5' in recon_tab_code, (
            "ReconTab's tile row is not a five-column grid — it still declares "
            'cols-4 (or was restructured), so the new tile wraps.'
        )
        assert 'col-span-12 grid cols-4' not in recon_tab_code, (
            'ReconTab still declares a cols-4 tile row.'
        )

    def test_cols_5_grid_rule_exists(self, styles_css_body):
        """`.cols-5` must actually declare its columns.

        Not incidental to this tile: `cols-5` is ALREADY referenced by
        MemoryTab with no matching rule, so `.grid`'s default applies and
        those five tiles stack in one column. Adding the rule completes a
        contract the markup has been asserting unbacked, and repairs that tab
        as well as this one.
        """
        block = _extract_css_rule_block(styles_css_body, '.cols-5')
        assert block, (
            '.cols-5 has no rule in styles.css, but the markup references it '
            '— the grid silently falls back to a single column and the tiles '
            'stack.'
        )
        assert 'grid-template-columns' in block, (
            f'.cols-5 declares no grid-template-columns: {block!r}'
        )
        assert re.search(r'repeat\(\s*5\s*,', block), (
            f'.cols-5 does not lay out FIVE columns: {block!r} — the sibling '
            'rules are all repeat(N, minmax(0, 1fr)) with N matching the class '
            'name, and a mismatch here would silently mis-size every tile.'
        )


@pytest.fixture(scope='module')
def recon_status_js_code(_client):
    """recon_status.js as the page serves it, comments blanked."""
    resp = _client.get('/static/redux/recon_status.js')
    assert resp.status_code == 200, (
        'recon_status.js is not served — index.html loads it, so a non-200 '
        'here means the tones read below come from a file no browser ever '
        'receives.'
    )
    return strip_js_comments(resp.text)


@pytest.fixture(scope='module')
def recon_badge_tones(recon_status_js_code):
    """Every tone reconStatusTone can return, read out of the module itself.

    Deliberately NOT a hand-written list. A second copy of the tones would
    reduce the test below to "these two lists agree", which is precisely the
    failure it exists to catch: a tone added to the module AND to the copy
    but not to the stylesheet renders an unstyled badge and stays green.
    Both halves of the return set are parsed — the mapped tones and the
    unrecognised-status fallback — and each is asserted non-empty, so a
    restructuring that defeats the parse fails loudly instead of shrinking
    the check to nothing.
    """
    mapped = set(re.findall(
        r":\s*'([\w-]+)'",
        _extract_object_literal(recon_status_js_code, 'RECON_STATUS_TONES'),
    ))
    fallback = set(re.findall(
        r"'([\w-]+)'",
        extract_function_body(recon_status_js_code, 'reconStatusTone'),
    ))
    assert mapped, (
        'no tones parsed out of RECON_STATUS_TONES — the map was restructured '
        'and this fixture no longer sees what a badge can be given.'
    )
    assert fallback, (
        'no fallback tone parsed out of reconStatusTone — the '
        'unrecognised-status branch was restructured and is no longer checked.'
    )
    return mapped | fallback


class TestBadgeTonesAreStyled:
    """Every tone recon_status.js can hand a badge must be styled.

    dashboard/tests/js/recon_status.test.mjs pins reconStatusTone's RETURN
    SET; what a node process structurally cannot see is whether styles.css
    defines those classes. Without this, a tone added to the module renders
    as an unstyled badge — every run row looking alike — and nothing fails.
    """

    def test_every_tone_the_module_returns_has_a_badge_rule(
        self, recon_badge_tones, styles_css_body
    ):
        unstyled = sorted(
            tone for tone in recon_badge_tones
            if not _extract_css_rule_block(styles_css_body, f'.badge.{tone}')
        )
        assert unstyled == [], (
            f'reconStatusTone can return {unstyled}, for which styles.css has '
            'no `.badge.<tone>` rule — those badges render with the base '
            '.badge styling alone, so a failed run reads exactly like a '
            'completed one.'
        )


class TestReconStatusLiteralsAreGone:
    """The invariant that closes the task: after the Recent Runs badge is
    converted, ReconTab CONSUMES the run vocabulary and RESTATES none of it.

    This is the property that would have prevented the original defect. Two
    call sites in this one file disagreed about the same statuses — the rate
    tile accepted only 'success' while the badge accepted 'success' OR
    'completed' — and neither matched what journal.py writes. With every
    literal sourced from recon_status.js, they cannot drift apart again.
    """

    def test_scope_is_really_recon_tab(self, recon_tab_code):
        """Vacuity guard for the absence assertions below.

        extract_function_body raises on a miss, so the body cannot be empty —
        but it could in principle be the WRONG function's, and absence
        assertions over an unrelated body would pass while proving nothing.
        """
        assert recon_tab_code.strip(), 'ReconTab body is empty'
        assert 'reconRunCounts' in recon_tab_code, (
            'the extracted body does not call reconRunCounts — this is not '
            "ReconTab's body, so the absence assertions below prove nothing."
        )

    def test_recent_runs_badge_tone_comes_from_the_module(self, recon_tab_code):
        """The per-row badge class must be computed by reconStatusTone."""
        assert re.search(r'badge \$\{reconStatusTone\(', recon_tab_code), (
            'the Recent Runs status badge does not derive its class from '
            'reconStatusTone(...).'
        )

    def test_no_run_status_literal_survives_anywhere_in_recon_tab(
        self, recon_tab_code
    ):
        """No vocabulary member, and neither retired spelling, may appear.

        Deliberately stronger than "no comparison": a literal in a label, a
        className or a filter is the same duplication with a different shape.
        The rendered badge TEXT is `{rn.status}` — the raw store value — so an
        unrecognised status still shows its own name under a muted tone
        rather than being hidden or relabelled, which is why no literal is
        needed here at all.
        """
        survivors = [
            lit for lit in RECON_STATUS_LITERALS
            if lit not in BURST_STATE_LITERALS and f"'{lit}'" in recon_tab_code
        ]
        assert survivors == [], (
            f'ReconTab still hard-codes the run-status literal(s) {survivors} '
            '— every one of them must come from recon_status.js, or two call '
            'sites in this file can disagree about the same window again.'
        )

    def test_no_run_row_is_compared_against_a_status_literal(self, recon_tab_code):
        """The overlapping literals, checked by USE rather than by presence.

        'running' is a member of BOTH the run vocabulary and the burst-AGENT
        vocabulary (bursting / cooling / running / idle), and the Burst state
        table legitimately branches on `b.state === 'running'`. That is a
        different vocabulary with a different writer — the same reason the
        rail-badge test above spares app.jsx's TASK statuses — so a
        whole-body ban on the string would forbid correct, in-scope-adjacent
        code. What must hold instead is that no RUN row's status is ever
        compared against a literal.
        """
        compared = re.findall(
            r'\b(\w+)\.status\s*===\s*\'([^\']*)\'', recon_tab_code
        )
        assert compared == [], (
            f'ReconTab compares a run status against a literal: {compared} — '
            'the run vocabulary belongs to recon_status.js.'
        )
