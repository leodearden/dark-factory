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

import re

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
    assert not re.search(r'\bspark\b', body), (
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
