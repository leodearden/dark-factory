"""Contract: no consumer destructures a ``window.DF_CHARTS`` name it never uses.

A tab that binds a chart component it does not render is not merely untidy —
it is a false statement about what that tab draws.  It survives greps, it makes
``find_references`` on the component report a consumer that does not exist, and
it is exactly what made HBarChart look like it had five call sites when it had
two (task 3681, which closed out that component's null-sample defects and found
the stale imports while measuring its real blast radius).

SCOPED TO THE DESTRUCTURE SHAPE ONLY — deliberate, not an oversight.  Nine .jsx
files under static/redux/ touch ``window.DF_CHARTS``, but only five reach it by
a ``const { ... } = window.DF_CHARTS`` destructure.  The others use:

  - a NAMESPACE binding (``const C = window.DF_CHARTS;`` —
    esc_flow_diagram.jsx, tab_escalation_analytics.jsx, tab_escalations.jsx)
  - a single MEMBER read (``const SP_SHELL = window.DF_CHARTS.Sparkline;`` —
    shell.jsx)

Do NOT widen the regex to swallow those.  Widening is actively wrong, not just
extra work: the namespace binding's own name (``C``) IS used, so a naive
extension flags it as a false positive and turns a green sweep red for no
defect; and member reads off a namespace object are not statically enumerable
the way a destructure list is.  The defect this file exists to catch — an
imported-but-never-rendered component — can only exist in the destructure shape
anyway.

The parser is the one already proven in test_tab_burndown.py
(``_DF_CHARTS_DESTRUCTURE_RE`` and its ``Canonical: alias`` splitting), copied
rather than re-invented so this is not a third dialect of the same line.
"""

from __future__ import annotations

import pathlib
import re

_REDUX_DIR = pathlib.Path(__file__).parent.parent / 'src' / 'dashboard' / 'static' / 'redux'

# Copied verbatim from
# dashboard/tests/test_tab_burndown.py::_DF_CHARTS_DESTRUCTURE_RE.
# Matches ONLY the destructure
# shape (see the module docstring for the three shapes it deliberately skips).
_DF_CHARTS_DESTRUCTURE_RE = re.compile(r'const\s*\{([^{}]*)\}\s*=\s*window\.DF_CHARTS')

# Measured after the sweep above was made green (task 3681): tab_curator 3,
# tab_memory_evals 3, tab_overview 4, tab_tasks 1, tabs 12.  It was 29 across
# the same five files before that — six of those bindings were the dead ones
# this file was written to find (tab_overview's five, tabs' one).
#
# The floors sit well UNDER the current census on purpose, and specifically
# under it by more than one tab's worth of imports: deleting dead bindings is a
# NORMAL outcome of this suite passing, so a floor set snugly against today's
# count would be failed by the very next cleanup it prompts.  See the vacuity
# guard for why the floors exist at all.
_MEASURED_FILES = 5
_MEASURED_BINDINGS = 23
_MIN_FILES = 4
_MIN_BINDINGS = 18


def _local_names(destructured: str) -> list[str]:
    """The LOCAL names a destructure list binds, resolving ``Canonical: alias``.

    ``{ StackedAreaChart, HistBar: HB }`` binds ``StackedAreaChart`` and ``HB``
    — the alias is what the file actually renders by, so the alias is what must
    be referenced.  Same splitting as test_tab_burndown.py's
    ``_chart_component_aliases``.
    """
    names = []
    for part in destructured.split(','):
        part = part.strip()
        if not part:
            continue
        canonical, _, alias = part.partition(':')
        names.append(alias.strip() or canonical.strip())
    return names


def _unused_bindings(src: str) -> list[str]:
    """The names ``src`` pulls off window.DF_CHARTS and then never mentions again.

    Every destructure statement in the file is cut out first, so the binding
    site itself does not count as a use.  What remains is searched for a
    whole-word occurrence of each name.

    That remainder still includes comments and strings, so a name mentioned only
    in prose reads as used.  The bias is deliberate and one-directional: this
    sweep would rather MISS a dead binding than flag a live one, because acting
    on a false positive means deleting an import the tab actually renders
    through.
    """
    spans = [m.span() for m in _DF_CHARTS_DESTRUCTURE_RE.finditer(src)]
    if not spans:
        return []

    remainder, cursor = [], 0
    for start, end in spans:
        remainder.append(src[cursor:start])
        cursor = end
    remainder.append(src[cursor:])
    outside = ' '.join(remainder)

    names = [n for m in _DF_CHARTS_DESTRUCTURE_RE.finditer(src) for n in _local_names(m.group(1))]
    return [n for n in names if not re.search(rf'\b{re.escape(n)}\b', outside)]


def _destructure_consumers() -> dict[str, str]:
    """``{filename: source}`` for every .jsx under redux/ using the destructure shape."""
    return {
        p.name: src
        for p in sorted(_REDUX_DIR.glob('*.jsx'))
        if _DF_CHARTS_DESTRUCTURE_RE.search(src := p.read_text(encoding='utf-8'))
    }


def test_no_consumer_binds_a_chart_component_it_never_renders() -> None:
    """Every destructured DF_CHARTS name is referenced somewhere else in its file.

    Reported as ONE assertion listing every offender.  A per-file assert would
    stop at the first and hide the rest, which is how a sweep like this ends up
    being run five times to find five things.
    """
    offenders = {
        name: unused
        for name, src in _destructure_consumers().items()
        if (unused := _unused_bindings(src))
    }

    assert not offenders, (
        'these consumers destructure window.DF_CHARTS components they never '
        'render:\n'
        + '\n'.join(f'  {f}: {", ".join(names)}' for f, names in sorted(offenders.items()))
        + '\n\nA binding with no use is a false claim about what the tab draws. '
        'Delete the binding — NOT the component: charts.jsx keeps every '
        'primitive and its window.DF_CHARTS registration regardless of what '
        'renders it today (test_charts_null_samples.py pins that separately).'
    )


def test_the_sweep_actually_found_the_destructure_shaped_consumers() -> None:
    """Vacuity guard: the sweep above must have had something to sweep.

    Every other assertion in this file is an ABSENCE assertion, so a path typo
    or a regex change that silently matched NOTHING would read as a clean green
    forever — the same permanent-false-GREEN failure mode
    test_charts_null_samples.py keeps a frozen pre-fix negative control to
    prevent.

    Floors, not exact pins.  The job here is to catch a sweep that matched
    nothing at all; pinning the measured census would additionally freeze it, so
    the next tab that legitimately adds or drops a chart import would fail an
    unrelated test and teach the next reader to edit the number rather than
    read it.  The measurement is stated in the message instead, so a real
    regression is still distinguishable from ordinary drift.
    """
    assert _REDUX_DIR.is_dir(), f'the redux static directory does not exist at {_REDUX_DIR}'

    consumers = _destructure_consumers()
    bindings = {
        name: [n for m in _DF_CHARTS_DESTRUCTURE_RE.finditer(src) for n in _local_names(m.group(1))]
        for name, src in consumers.items()
    }
    total = sum(len(v) for v in bindings.values())

    assert len(consumers) >= _MIN_FILES and total >= _MIN_BINDINGS, (
        f'the window.DF_CHARTS destructure sweep matched {len(consumers)} '
        f'file(s) and {total} binding(s), below the floor of {_MIN_FILES}/'
        f'{_MIN_BINDINGS}. Measured at task 3681: {_MEASURED_FILES} files, '
        f'{_MEASURED_BINDINGS} bindings (tab_curator 3, tab_memory_evals 3, '
        f'tab_overview 4, tab_tasks 1, tabs 12). A sweep this far below that '
        f'is almost certainly a broken path or regex rather than real drift, '
        f'and it would make the assertions in this file vacuously green. '
        f'Found: { {k: len(v) for k, v in sorted(bindings.items())} }'
    )


# ---------------------------------------------------------------------------
# Negative control — the detector, run against the defect it was written for
#
# The vacuity guard above proves the sweep MATCHED something.  It does not
# prove ``_unused_bindings`` can still SEE a dead binding, and that is a
# separate failure: widen the excised span by one character, drop the
# ``\b`` word boundary, or let ``_local_names`` return the canonical name where
# an alias is bound, and every consumer reads as clean forever.  The sweep is a
# pure absence assertion, so it would go permanently green rather than red.
#
# So the detector is pinned against the real pre-fix sources, frozen verbatim
# from the commit before task 3681 deleted them (15889b4777^) — the same
# frozen-pre-fix-body shape test_charts_null_samples.py uses for its arithmetic
# probes.  Both directions are asserted: the six dead names ARE reported, and
# nothing that the body actually references is.
# ---------------------------------------------------------------------------

# tab_overview.jsx:2, verbatim.  Nine bindings, five of them dead.
_PRE_FIX_TAB_OVERVIEW = """\
/* Overview tab — command-center grid */
const { Sparkline, LineChart, StackedAreaChart, BarChart, HBarChart, Donut, StatTile, HistBar, PALETTE: P } = window.DF_CHARTS;
const { Glyph, LiveFeed } = window.DF_SHELL;

function Overview() {
  return (
    <div>
      <StatTile label="tasks" />
      <Sparkline values={[1, 2]} color={P.accent} />
      <LineChart series={[]} />
      <Glyph name="dot" />
    </div>
  );
}
"""

# tabs.jsx:2, verbatim.  Thirteen bindings, one dead (``HistBar: HB`` — the
# ALIAS is the live name, which is exactly what a canonical-vs-alias slip in
# ``_local_names`` would get wrong in both directions at once).
_PRE_FIX_TABS = """\
/* Remaining tabs: orchestrators, performance, memory, recon, merge, costs, burndown */
const { Sparkline: SP, LineChart: LC, StackedAreaChart: SA, BarChart: BC, HBarChart: HBC, Donut: DN, StatTile: ST, HistBar: HB, PALETTE: CP, deriveVelocitySeries, defaultSmoothingForWindow, smoothingLabelToSeconds, SMOOTHING_OPTIONS } = window.DF_CHARTS;

function CostsTab() {
  const series = deriveVelocitySeries([]);
  const smoothing = defaultSmoothingForWindow(SMOOTHING_OPTIONS[0]);
  return (
    <div>
      <SP values={series} /><LC series={[]} /><SA stacks={[]} /><BC values={[]} />
      <HBC rows={[]} /><DN data={[]} /><ST label="spend" />
      <span style={{ color: CP.fg2 }}>{smoothingLabelToSeconds(smoothing)}</span>
    </div>
  );
}
"""

_PRE_FIX_SOURCES: dict[str, tuple[str, list[str]]] = {
    'tab_overview.jsx': (
        _PRE_FIX_TAB_OVERVIEW,
        ['StackedAreaChart', 'BarChart', 'HBarChart', 'Donut', 'HistBar'],
    ),
    'tabs.jsx': (_PRE_FIX_TABS, ['HB']),
}


def test_the_detector_reports_the_dead_bindings_it_was_written_to_find() -> None:
    """``_unused_bindings`` finds the exact bindings task 3681 deleted.

    Without this, the sweep above could be satisfied by a detector that
    reports nothing at all — the permanent-false-GREEN failure mode.
    """
    for filename, (src, expected_dead) in _PRE_FIX_SOURCES.items():
        found = _unused_bindings(src)

        assert found == expected_dead, (
            f'the pre-fix {filename} bound {len(expected_dead)} DF_CHARTS '
            f'name(s) it never rendered — {", ".join(expected_dead)} — but '
            f'the detector reported {found or "nothing"}. Whatever the sweep '
            f'above is doing, it is no longer what deleting these was proven '
            f'to require, so its green is not evidence of anything.'
        )


def test_the_detector_does_not_report_a_binding_the_file_actually_renders() -> None:
    """The other direction: a rendered name must never be reported.

    Acting on a false positive means deleting an import a tab renders
    through, which is why ``_unused_bindings`` is deliberately biased toward
    missing a dead binding over flagging a live one.  Asserted on the same
    frozen sources so both directions are pinned by one pair of fixtures.
    """
    for filename, (src, expected_dead) in _PRE_FIX_SOURCES.items():
        live = [
            n
            for m in _DF_CHARTS_DESTRUCTURE_RE.finditer(src)
            for n in _local_names(m.group(1))
            if n not in expected_dead
        ]
        falsely_flagged = sorted(set(_unused_bindings(src)) & set(live))

        assert not falsely_flagged, (
            f'the detector flagged {", ".join(falsely_flagged)} in the frozen '
            f'{filename}, but that source renders through those names. A false '
            f'positive here gets a live import deleted.'
        )
