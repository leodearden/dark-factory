"""Contract: each shared component renders THROUGH a decision module.

The three shared components of PRD leaf gamma1 — ``charts.jsx::StatTile``,
``shell.jsx::Pip`` and ``tabs.jsx::LocksCell`` — must ask a plain-JS decision
function what to draw and then draw it.  None may carry a render branch of its
own.  That is the entire point of the leaf: 43 StatTile call sites came to
hand-roll 14 different null guards precisely because the component answered
nothing and every caller answered for itself.

WHY THE ASSERTIONS ARE STRUCTURAL, AND WHAT THAT BUYS.  These files are
``type="text/babel"`` JSX behind CDN Babel with no node_modules, so no harness
here can execute a component body (the rationale block at charts.jsx:6-11 states
it outright; ``test_charts_axis_labels.py``:1-20 and ``test_charts_null_samples.py``
carry the same one).  The BEHAVIOUR is therefore asserted where it is
executable — ``dashboard/tests/js/datum.test.mjs`` for ``datumView`` and
``dashboard/tests/js/task_row_cells.test.mjs`` for ``locksCellState`` — and what
is left for this file is the WIRING: that the component reaches the decision at
all, and that it kept no second answer beside it.  Those two questions are
exactly what source text can settle and behaviour cannot.

EVERY PROBE RUNS OVER COMMENT-STRIPPED CODE, via
``extract_function_body(strip_js_comments(body), name)``.  Without the strip, a
substring probe also matches PROSE, and that coupling has already deformed this
repo's production source once: charts.jsx carried a comment that was an apology
for what it could not say, because naming the expression the component had just
stopped using would fail CI with a message pointing at the comment
(``strip_js_comments``' own docstring records it).  A presence probe satisfied by
a comment is a vacuous pass; an absence probe broken by one is a false red.

The extractors RAISE on a miss rather than returning ''.  An absence assertion
over an empty slice passes vacuously, which would make a deleted component read
as a migrated one.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from _dashboard_helpers import (
    extract_function_body,
    find_function_params,
    strip_js_comments,
)


def _params_of(body: str, name: str) -> str:
    """The parameter-list text of ``function <name>(``, comments already gone."""
    masked, start, end = find_function_params(body, name)
    return masked[start:end]


# The decision function each component must route through, and the name it is
# bound to in that file.  Stated as one table because "every shared component
# renders through a decision module" is the actual claim, and a per-test literal
# is how three copies of one rule come to disagree.
_COMPONENT_DECISIONS = (
    ('charts.jsx', 'StatTile', 'datumView'),
    ('shell.jsx', 'Pip', 'datumView'),
    ('tabs.jsx', 'LocksCell', 'locksCellState'),
)


def test_stat_tile_renders_through_datum_view(charts_jsx_body: str) -> None:
    """charts.jsx::StatTile asks datumView what to draw, and keeps no second answer.

    The absence half is the load-bearing one.  A tile that still rendered a bare
    ``{value}`` would be a second authority on the hole question — the exact
    shape the census found 14 copies of — and one that still bound ``spark``
    would mean the rename to ``history`` happened in the signature only, leaving
    the old prop silently ignored at 43 call sites.
    """
    body = extract_function_body(strip_js_comments(charts_jsx_body), 'StatTile')

    assert 'datumView' in body, (
        'charts.jsx::StatTile does not call datumView. The tile must ask for the '
        'render decision (text, title, age) rather than formatting a value '
        'itself — see datum.js::datumView.'
    )
    assert not re.search(r'\{\s*value\s*\}', body), (
        'charts.jsx::StatTile still renders a bare {value}. The value cell must '
        "come from datumView's `text`, so the hole decision is made in one "
        'place instead of once per call site.'
    )
    # An IDENTIFIER named spark, not the CSS class of the same name: the
    # sparkline's container is `className="spark"` and always was, so a probe
    # that swallowed the string literal too would be unsatisfiable by any
    # correct source.  The lookarounds reject a quoted occurrence and nothing
    # else — `{spark && ...}` and `values={spark}` both still match.
    assert not re.search(r'''(?<!['"])\bspark\b(?!['"])''', body), (
        "charts.jsx::StatTile still references `spark`. The prop is renamed to "
        '`history` so the sparkline series is named for what it is and cannot be '
        'confused with the datum-backed value; a lingering `spark` read means '
        'call sites keep passing a prop the component ignores.'
    )


def test_stat_tile_signature_names_datum_and_history(charts_jsx_body: str) -> None:
    """The signature is the half a call site can see, so it is pinned separately.

    A body that used `datum` while the signature still accepted `value` would
    make every unmigrated call site render undefined rather than fail — the
    silent degradation assertDatum exists to prevent.
    """
    params = _params_of(strip_js_comments(charts_jsx_body), 'StatTile')

    for required in ('datum', 'history'):
        assert re.search(rf'\b{required}\b', params), (
            f'charts.jsx::StatTile does not accept `{required}`. Its signature is '
            '{label, datum, history, format, unit, delta, deltaDir, sparkColor, '
            f'hint}} — got: {params.strip()}'
        )
    for removed in ('value', 'spark'):
        assert not re.search(rf'\b{removed}\b', params), (
            f'charts.jsx::StatTile still accepts `{removed}`. Keeping it means a '
            'call site can hand the tile an unprovenanced value and never hear '
            f'about it — got: {params.strip()}'
        )


def test_pip_is_a_shared_component_rendering_through_datum_view(shell_jsx_body: str) -> None:
    """shell.jsx::Pip exists and renders through datumView.

    Pip is NEW.  ProjectGroup could not be "the shared component the pips render
    through": it takes `summary` as an opaque node, and the word "pip" does not
    occur in shell.jsx at all — every pip is a caller-built JSX fragment sharing
    only the CSS at styles.css:381-384.  So the six fragments had no shared home
    to migrate INTO, and this creates it beside ProjectGroup, which is the
    closest honest reading of the PRD's "shell.jsx::ProjectGroup pips".
    """
    body = extract_function_body(strip_js_comments(shell_jsx_body), 'Pip')

    assert 'datumView' in body, (
        'shell.jsx::Pip does not call datumView. A pip rendering a MEASURED '
        'number must make the same hole/age decision a tile does, or the two '
        'surfaces disagree about what an unmeasured value looks like.'
    )
    assert 'pip-dot' in body, (
        'shell.jsx::Pip does not render the `pip-dot` span. The component must '
        'carry the EXISTING markup the six caller-side fragments share '
        '(styles.css:381-384 styles `.proj-head .summary .pip`/`.pip-dot`), not '
        'a new shape that styles differently.'
    )

    params = _params_of(strip_js_comments(shell_jsx_body), 'Pip')
    assert re.search(r'\bdatum\b', params), (
        f'shell.jsx::Pip does not accept `datum` — got: {params.strip()}'
    )


def test_pip_is_exported_on_df_shell(shell_jsx_body: str) -> None:
    """Pip must be reachable by the tabs that render it.

    tabs.jsx and tab_escalations.jsx destructure window.DF_SHELL at top level
    with no fallback, so a component defined but not exported is a component
    those files cannot name — and the migration would have to hand-copy the
    markup back, which is the drift this component removes.
    """
    stripped = strip_js_comments(shell_jsx_body)
    match = re.search(r'window\.DF_SHELL\s*=\s*\{([^}]*)\}', stripped)
    assert match, 'shell.jsx no longer assigns window.DF_SHELL = { ... }'
    assert re.search(r'\bPip\b', match.group(1)), (
        'shell.jsx does not export Pip on window.DF_SHELL. Exported members: '
        f'{match.group(1).strip()}'
    )


def test_locks_cell_renders_through_locks_cell_state(tabs_jsx_body: str) -> None:
    """tabs.jsx::LocksCell asks locksCellState, and does NOT re-derive it.

    The decision lives in task_row_cells.js where it is executable under
    `node --test`; a copy of its `state === 'unknown'` test inside this JSX
    would be a second authority on the question, un-assertable by construction
    and free to drift from the one the node suite covers.
    """
    body = extract_function_body(strip_js_comments(tabs_jsx_body), 'LocksCell')

    assert 'locksCellState' in body, (
        'tabs.jsx::LocksCell does not call locksCellState. The offline-scheduler '
        'arm — an em-dash with its reason, rather than an empty chip list that '
        'reads as "no locks held" — is decided in task_row_cells.js.'
    )
    assert not re.search(r"""state\s*===\s*['"]unknown['"]""", body), (
        "tabs.jsx::LocksCell re-derives the unknown test itself. That decision "
        'belongs to task_row_cells.js::locksCellState, which the node suite can '
        'actually execute.'
    )

    params = _params_of(strip_js_comments(tabs_jsx_body), 'LocksCell')
    assert re.search(r'\bdatum\b', params), (
        f'tabs.jsx::LocksCell does not accept `datum` — got: {params.strip()}'
    )


def test_each_shared_component_reaches_its_decision_module(
    charts_jsx_body: str,
    shell_jsx_body: str,
    tabs_jsx_body: str,
) -> None:
    """The table above, swept — so a fourth shared component cannot be added
    un-wired and a third file cannot quietly drop its decision call.

    Parametrized by data rather than by three more test functions: the claim is
    "every shared component routes through a decision", and a per-component test
    states that claim three times.
    """
    bodies = {
        'charts.jsx': charts_jsx_body,
        'shell.jsx': shell_jsx_body,
        'tabs.jsx': tabs_jsx_body,
    }
    for filename, component, decision in _COMPONENT_DECISIONS:
        body = extract_function_body(strip_js_comments(bodies[filename]), component)
        assert decision in body, (
            f'{filename}::{component} does not reach {decision}. Every shared '
            'component in this leaf renders through a plain-JS decision module; '
            'a component that decides for itself is un-executable by any test '
            'here and is how 43 tiles came to hand-roll 14 null guards.'
        )


def test_every_shared_component_file_destructures_df_datum(
    charts_jsx_body: str,
    shell_jsx_body: str,
    tabs_jsx_body: str,
) -> None:
    """Each file binds window.DF_DATUM at TOP level, with no fallback.

    The DF_SPARK_PATH convention (charts.jsx:14-20): a missing dependency throws
    at load with a clear message rather than deferring to a TypeError inside a
    render or degrading into a component that draws nothing.  A per-render
    `window.DF_DATUM && ...` read would be the silent-degradation shape instead,
    and would also hide a load-order regression that test_index_html.py pins.
    """
    for filename, body in (
        ('charts.jsx', charts_jsx_body),
        ('shell.jsx', shell_jsx_body),
        ('tabs.jsx', tabs_jsx_body),
    ):
        stripped = strip_js_comments(body)
        assert re.search(r'=\s*window\.DF_DATUM\s*;', stripped), (
            f'{filename} does not destructure window.DF_DATUM at module scope. '
            'Bind the names it needs with `const { ... } = window.DF_DATUM;` and '
            'no `|| {}` fallback, so a missing or mis-ordered datum.js throws at '
            'load instead of inside a render.'
        )
        assert not re.search(r'window\.DF_DATUM\s*(\|\||&&|\?\?)', stripped), (
            f'{filename} guards its window.DF_DATUM read with a fallback. That '
            'turns a load-order regression into components that silently render '
            'nothing — the degradation this convention exists to refuse.'
        )


# ---------------------------------------------------------------------------
# THE CALL-SITE CENSUS.
#
# A migrated component is only half the contract: a tile that still hands over a
# bare `value=` renders whatever it is given with no provenance at all, and the
# components above cannot detect it — `datum` simply arrives undefined and
# assertDatum throws inside a render nobody runs in CI.  So the sites are counted
# here, by hand-free enumeration of the served source.
#
# THE COUNTS ARE THE LOAD-BEARING HALF.  An absence-only probe ("no site carries
# `value=`") passes vacuously if a whole file's tiles are deleted, and passes
# just as happily if the grep spelling silently stops matching.  Pinning the
# per-file totals means a deletion, a new un-migrated site, and a broken matcher
# each fail with a message naming the file.
#
# THE PRD'S OWN LIST IS STALE IN BOTH DIRECTIONS, which is why this table was
# re-measured rather than transcribed.  It names "36 in tabs.jsx/tab_overview.jsx"
# — correct, 32 + 4 — but omits the seven `<C.StatTile` sites in
# tab_escalations.jsx and tab_escalation_analytics.jsx, and it names four files
# (scheduler_drawer.jsx, tab_curator.jsx, tab_memory_evals.jsx, tab_scheduler.jsx)
# that carry no StatTile site at all — they do not even import the component.
# Those four are asserted at zero below, so the table cannot quietly lose a file.
#
# THREE LOCAL SPELLINGS, ONE COMPONENT.  tabs.jsx aliases it `ST` in its
# window.DF_CHARTS destructure, tab_overview.jsx imports it under its own name,
# and the two escalation tabs reach it through the `C` namespace binding as
# `C.StatTile`.  A probe that knew only one spelling would report a clean sweep
# over a third of the sites.
_STAT_TILE_SITES = {
    'tabs.jsx': 32,
    'tab_overview.jsx': 4,
    'tab_escalations.jsx': 5,
    'tab_escalation_analytics.jsx': 2,
}

_NO_STAT_TILE_FILES = (
    'scheduler_drawer.jsx',
    'tab_curator.jsx',
    'tab_memory_evals.jsx',
    'tab_scheduler.jsx',
)

# Pips that render a MEASURED NUMBER, which is what makes them this leaf's
# business: a frozen count with no age beside it reads as a current one.  The
# status-word pips below are not measurements and stay hand-built.
_PIP_SITES = {
    'tabs.jsx': 13,
    'tab_escalations.jsx': 4,
}

# The pips that legitimately remain hand-built `className="pip"` spans, and the
# word each one renders.  All three are derived FLAGS — a boolean read off an
# orchestrator row — with no measurement instant of their own, so wrapping one in
# an envelope whose contract is value + as_of + state would be a category error;
# PRD leaves gamma2/theta own those surfaces.
#
# tab_tasks.jsx is deliberately absent from both tables.  Its pips are fed by
# task_status_counts.js::activityPips, which PRD leaf gamma3 DELETES — migrating
# them here would be work gamma3 has to undo.
_HAND_BUILT_PIPS = {
    'tabs.jsx': ('running', 'offline', 'state unknown'),
    'tab_escalations.jsx': (),
}

_STAT_TILE_TAG_RE = re.compile(r'<(?:C\.)?(?:ST|StatTile)(?=[\s/>])')
_PIP_TAG_RE = re.compile(r'<Pip(?=[\s/>])')
_LOCKS_CELL_TAG_RE = re.compile(r'<LocksCell(?=[\s/>])')
_HAND_BUILT_PIP_RE = re.compile(r'className="pip"')

# `(?<![\w$])` and not `\b`: `sparkColor=` starts with the letters of `spark`,
# and `\bspark\s*=` would not match it — but `\bvalue\s*=` DOES match a
# hypothetical `x.value =`, and the lookbehind is the spelling that says why.
_DATUM_PROP_RE = re.compile(r'(?<![\w$])datum\s*=')
_BANNED_TILE_PROPS = (
    ('value', 'renders an unprovenanced number: no measurement instant, no state, no reason'),
    ('spark', 'is renamed `history`, so the series is named for what it is'),
)


def _tag_end(source: str, index: int, site: str) -> int:
    """Index just past the ``>`` closing the opening tag that starts at *index*.

    Brace-depth aware and quote aware, because a JSX prop value is an arbitrary
    expression: ``hint={x > 0 ? 'a' : 'b'}`` carries a ``>`` that does not close
    the tag, and ``label="a > b"`` carries one inside a literal.  A regex ending
    at the first ``>`` would truncate a third of this census's spans mid-prop,
    and every truncated span would then read as a site carrying no ``datum=``.

    RAISES rather than returning a best effort.  A span that runs off the end of
    the file means the walker lost the tag, and an absence assertion over a
    runaway span is a silent false GREEN.

    A SIBLING COPY OF THIS WALK EXISTS at
    ``test_tab_memory_evals.py::_jsx_open_tag_end``, written for the same
    documented trap (mem0 b412a877/86bc64c0: a `[^<>]*` tag span is truncated by
    a bare `<`/`>` inside an attribute expression, so an ordinary
    ``format={n => …}`` turns a correct file red).  The two differ in contract —
    that one returns ``-1`` on a miss, this one raises, because an absence probe
    over a lost span is a false green here — but the scan is the same primitive
    and belongs in ``_dashboard_helpers.py`` beside ``walk_balanced``.  Hoisting
    it touches a module outside this task's scope; filed as follow-up work
    rather than done inline.
    """
    depth = 0
    quote = None
    while index < len(source):
        char = source[index]
        if quote is not None:
            if char == '\\':
                index += 2
                continue
            if char == quote:
                quote = None
        elif char in '\'"`':
            quote = char
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
        elif char == '>' and depth == 0:
            return index + 1
        index += 1
    raise AssertionError(
        f'{site}: an opening tag is never closed — the census walker ran off the '
        'end of the file, so no assertion over its span would mean anything.'
    )


def _tag_spans(source: str, pattern: re.Pattern[str], site: str) -> list[str]:
    """Each opening tag matching *pattern*, as its own balanced text."""
    return [
        source[match.start():_tag_end(source, match.end(), site)]
        for match in pattern.finditer(source)
    ]


_SPAN_TOKEN_RE = re.compile(r'</span\s*>|<span(?=[\s/>])')


def _hand_built_pip_spans(source: str, site: str) -> list[str]:
    """Each hand-built ``className="pip"`` element, CHILDREN INCLUDED.

    ``_tag_spans`` above stops at the opening tag's ``>``, which is the right
    span for a prop assertion and the wrong one for a word: a pip renders its
    word as a TEXT CHILD (``…></span>offline</span>``), so an opening tag never
    contains it.  Searching the whole file for the word instead is what made the
    assertion below vacuous — 'running' occurs four times in tabs.jsx and
    'offline' nine, almost all of them outside any pip.

    Nesting-aware because every pip wraps a ``<span>`` dot, and self-closing
    aware because ``<span … />`` opens no depth.  RAISES on an unclosed element,
    for ``_tag_end``'s reason: a probe over a runaway span means nothing.
    """
    spans = []
    for match in _HAND_BUILT_PIP_RE.finditer(source):
        start = source.rfind('<span', 0, match.start())
        assert start != -1, (
            f'{site}: a `className="pip"` with no opening <span> before it — the '
            'walker cannot bound the element, so no assertion over it would mean '
            'anything.'
        )
        depth = 0
        index = start
        while True:
            token = _SPAN_TOKEN_RE.search(source, index)
            assert token is not None, (
                f'{site}: a hand-built pip element is never closed.'
            )
            if token.group().startswith('</'):
                depth -= 1
                index = token.end()
                if depth == 0:
                    spans.append(source[start:index])
                    break
            else:
                index = _tag_end(source, token.end(), site)
                if source[index - 2:index] != '/>':
                    depth += 1
    return spans


@pytest.fixture(scope='module')
def census_bodies(_client):
    """Comment-stripped served bodies for every file the census names.

    One fixture over a table rather than eight per-file fixtures: the tables
    above are the list of files this leaf claims to have swept, and deriving the
    fetches from them is what stops a file being added to a table and then never
    actually read.

    Comment-stripped for the reason the module docstring gives — a substring
    probe over raw source also matches prose, and `<ST` is quoted in this repo's
    own comments.  The served body, not the file on disk, because that is what a
    browser executes and it is what every sibling wiring suite reads.
    """
    bodies = {}
    for name in (*_STAT_TILE_SITES, *_NO_STAT_TILE_FILES, *_PIP_SITES):
        if name in bodies:
            continue
        resp = _client.get(f'/static/redux/{name}')
        assert resp.status_code == 200, (
            f'{name} is not served (HTTP {resp.status_code}) — the census names '
            'it, so a 404 here is a renamed or deleted file, not a test bug.'
        )
        bodies[name] = strip_js_comments(resp.text)
    return bodies


def test_stat_tile_census_counts_are_exact(census_bodies):
    """43 tiles, over four files, under three local spellings.

    The census is stated as counts rather than as "at least one" so the
    migration assertions below cannot pass by deletion.
    """
    measured = {
        name: len(_STAT_TILE_TAG_RE.findall(census_bodies[name]))
        for name in _STAT_TILE_SITES
    }
    assert measured == _STAT_TILE_SITES, (
        'the StatTile call-site census moved. Expected '
        f'{_STAT_TILE_SITES} (43 total), measured {measured}. If a tile was '
        'legitimately added or removed, update _STAT_TILE_SITES in the same '
        'commit — the count is what stops a migration passing by deletion.'
    )
    assert sum(measured.values()) == 43


def test_files_the_prd_named_carry_no_stat_tile(census_bodies):
    """The four files the PRD listed that hold no tile — pinned, not assumed.

    Measured, because the PRD's list was measured STALE: these four do not even
    destructure StatTile.  Asserting zero is what makes the table above a census
    rather than a sample — a tile appearing in one of them would otherwise be
    swept up by nothing.
    """
    for name in _NO_STAT_TILE_FILES:
        found = _STAT_TILE_TAG_RE.findall(census_bodies[name])
        assert not found, (
            f'{name} now carries {len(found)} StatTile call site(s). Move it into '
            '_STAT_TILE_SITES with its count so the migration probes cover it.'
        )


def test_every_stat_tile_site_hands_over_a_datum(census_bodies):
    """Each of the 43 sites carries ``datum=`` and neither ``value=`` nor ``spark=``.

    Matched within the tag's OWN balanced span, so a neighbouring element
    carrying `datum=` cannot satisfy a site that does not — the failure mode a
    line-oriented grep has, since a dozen of these sites span four or more lines
    and sit inside dense JSX.
    """
    for name in _STAT_TILE_SITES:
        for span in _tag_spans(census_bodies[name], _STAT_TILE_TAG_RE, name):
            assert _DATUM_PROP_RE.search(span), (
                f'{name}: a StatTile site hands over no `datum=`. Wrap the value '
                "with plainDatum(value, '<endpoint path>') so the tile knows when "
                f'it was measured:\n{span}'
            )
            for prop, why in _BANNED_TILE_PROPS:
                assert not re.search(rf'(?<![\w$]){prop}\s*=', span), (
                    f'{name}: a StatTile site still passes `{prop}=`, which {why}. '
                    f'StatTile no longer accepts it, so it is silently ignored:\n{span}'
                )


def test_measured_pips_render_through_the_shared_component(census_bodies):
    """Every pip that renders a measured number is a ``<Pip datum={…}>``.

    Two halves, and both are needed.  The COUNT of `<Pip` sites stops the
    migration passing by deleting a pip; the count of hand-built
    `className="pip"` spans stops a new un-migrated one being added beside them,
    which is the drift that produced six copies of this fragment in the first
    place.
    """
    for name, expected in _PIP_SITES.items():
        body = census_bodies[name]

        spans = _tag_spans(body, _PIP_TAG_RE, name)
        assert len(spans) == expected, (
            f'{name} renders {len(spans)} <Pip> sites, expected {expected}. A pip '
            'showing a measured number must go through the shared component so '
            'its age and its hole are decided once.'
        )
        for span in spans:
            assert _DATUM_PROP_RE.search(span), (
                f'{name}: a <Pip> site hands over no `datum=`:\n{span}'
            )

        remaining = _hand_built_pip_spans(body, name)
        allowed = _HAND_BUILT_PIPS[name]
        assert len(remaining) == len(allowed), (
            f'{name} still hand-builds {len(remaining)} `className="pip"` span(s); '
            f'only {len(allowed)} may remain {allowed}. Every other pip renders a '
            'measured number and belongs in <Pip>.'
        )
        # Inside the pip's OWN element, never anywhere in the file. The count
        # above already catches a deletion; what this catches is a SWAP — one
        # allowed flag replaced by a different hand-built pip, which leaves the
        # count untouched. Searched file-wide it caught neither: 'running'
        # occurs four times in tabs.jsx and 'offline' nine, almost all outside
        # any pip, so both assertions passed whatever the markup said.
        for word in allowed:
            holders = [span for span in remaining if word in span]
            assert len(holders) == 1, (
                f'{name} renders the `{word}` status pip in {len(holders)} of its '
                f'{len(remaining)} hand-built pips, expected exactly 1. It is a '
                'derived FLAG, not a measurement — it must stay hand-built, and '
                f'it must stay:\n{remaining}'
            )


def test_the_locks_cell_site_hands_over_a_datum(census_bodies):
    """LocksCell's single call site says whether anything is known about locks.

    Pinned at exactly one site: the cell's whole reason for existing is that an
    empty chip list and an unreadable scheduler snapshot render identically, and
    a second call site that forgot the datum would reintroduce that conflation
    in the one column built to remove it.
    """
    body = census_bodies['tabs.jsx']
    spans = _tag_spans(body, _LOCKS_CELL_TAG_RE, 'tabs.jsx')

    assert len(spans) == 1, (
        f'tabs.jsx renders {len(spans)} <LocksCell> sites, expected 1.'
    )
    assert _DATUM_PROP_RE.search(spans[0]), (
        'tabs.jsx: <LocksCell> is handed no `datum=`, so locksCellState throws '
        'inside the render rather than drawing the unknown arm:\n'
        f'{spans[0]}'
    )


# ---------------------------------------------------------------------------
# The endpoint paths that key DF_DATA.__receipt
#
# plainDatum's provenance is endpoint-granular until PRD leaf beta puts a served
# Datum on the wire, so each tile hands it the path its number arrived on — and
# that path is a LOOKUP KEY into a map data.js wrote.  Four JSX files now declare
# those paths as hand-typed copies (tabs.jsx::EP, tab_overview.jsx::EP_OVERVIEW,
# tab_escalations.jsx::EP_ESCALATIONS, tab_escalation_analytics.jsx::EP_ANALYTICS)
# of paths already written once in data.js::endpointsFor.
#
# WHAT A MISTYPED COPY DOES, and why it needs a test rather than care.  It is the
# one failure the wrapper cannot report: the lookup finds no receipt, plainDatum
# takes its "not yet fetched" arm, and every tile on that endpoint reads as a
# permanent em-dash with a never-fetched tooltip while the endpoint is in fact
# healthy.  That is the confident lie this PRD exists to remove, inverted — and
# the four copies are all correct today, which is exactly when a drift guard is
# worth installing.
#
# THE SAME GUARD ALREADY EXISTS ONE LAYER DOWN.  endpoint_staleness.js carries
# its own copy of this path list and endpoint_staleness.test.mjs:222 ("every
# mapped endpoint is a real endpointsFor() path") pins it against the registry.
# This is that test's JSX analogue; the registry stays the one place a path is
# declared (heuristic 11, SPOT).
# ---------------------------------------------------------------------------

# Every asset index.html serves, discovered rather than listed: a fifth copy
# added by a later leaf is caught only if nothing has to remember to enrol it.
_REDUX_DIR = pathlib.Path(__file__).resolve().parent.parent / 'src' / 'dashboard' / 'static' / 'redux'

# `/api/load` and the other non-dashboard routes are deliberately out of range —
# the claim is about the receipt map's keys, which are dashboard endpoints.
_DASHBOARD_PATH_RE = re.compile(r'/api/v2/dashboard/[A-Za-z0-9-]+')


@pytest.fixture(scope='module')
def registry_paths(_client) -> set[str]:
    """Every endpoint path data.js::endpointsFor declares, query already stripped.

    Read out of the registry FUNCTION rather than the whole file, so a path
    mentioned anywhere else in data.js (ON_DEMAND_KEYS builds one by
    interpolation) cannot make an undeclared copy look declared.
    """
    resp = _client.get('/static/redux/data.js')
    assert resp.status_code == 200, f'data.js is not served (HTTP {resp.status_code})'
    body = extract_function_body(strip_js_comments(resp.text), 'endpointsFor')
    paths = set(_DASHBOARD_PATH_RE.findall(body))
    assert paths, 'no endpoint paths found in endpointsFor — the extractor lost the registry'
    return paths


def test_every_endpoint_path_a_redux_asset_names_is_one_the_registry_polls(_client, registry_paths):
    """No served asset may name a dashboard endpoint endpointsFor does not declare.

    Counted over every ``.js``/``.jsx`` under ``static/redux`` except data.js
    itself, so the guard covers a file this leaf never touched and a file a later
    leaf adds.  Comment-stripped, so a path quoted in prose — and the four EP
    blocks each quote one — cannot satisfy or break the probe.
    """
    offenders = {}
    for source in sorted(_REDUX_DIR.glob('*.js*')):
        if source.name == 'data.js':
            continue
        resp = _client.get(f'/static/redux/{source.name}')
        assert resp.status_code == 200, (
            f'{source.name} is on disk but not served (HTTP {resp.status_code}).'
        )
        undeclared = set(_DASHBOARD_PATH_RE.findall(strip_js_comments(resp.text))) - registry_paths
        if undeclared:
            offenders[source.name] = sorted(undeclared)

    assert not offenders, (
        f'these assets name dashboard endpoints data.js::endpointsFor does not '
        f'declare: {offenders}. A path that is not a registry key is not a key of '
        'DF_DATA.__receipt either, so plainDatum finds no receipt and every tile '
        'on it reads as a permanent em-dash while the endpoint is healthy. Fix '
        'the spelling, or add the row to endpointsFor.'
    )


def test_the_four_datum_files_key_their_receipts_off_real_paths(census_bodies, registry_paths):
    """Each file that calls plainDatum/derivedDatum names at least one real path.

    The sibling above is an ABSENCE assertion, which passes just as happily over
    a file that names no path at all — including one whose EP block was deleted
    and whose tiles now key their receipts off `undefined`.  This is the presence
    half, and it is stated per file so a deletion cannot hide behind the other
    three.
    """
    for name in _STAT_TILE_SITES:
        body = census_bodies[name]
        named = set(_DASHBOARD_PATH_RE.findall(body))
        assert named, (
            f'{name} calls plainDatum but names no /api/v2/dashboard path, so every '
            'receipt lookup in it resolves to undefined and every tile reads as '
            'never-fetched.'
        )
        assert named <= registry_paths, (
            f'{name} names {sorted(named - registry_paths)}, which endpointsFor '
            'does not declare.'
        )
