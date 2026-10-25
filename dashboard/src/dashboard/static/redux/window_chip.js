// window_chip.js — the pure decisions behind the global window chip: which
// tabs carry it, which windows each offers, whose served-window echo each
// reports, how a served window is labelled and highlighted, and which chosen
// window is still pending (app.jsx's Toolbar wiring, tabs.jsx's panel headers
// and Recent-merges caption).
//
// LOAD CONTRACT. It destructures window.DF_DATUM at module scope with no
// fallback, so index.html loads it after datum.js and before the Babel JSX
// tags, so the global exists before tabs.jsx and app.jsx run their top-level
// destructures of it (test_index_html.py pins all three orders). A browser
// classic `<script>` assigns `window.DF_WINDOW_CHIP`; node requires the same
// file as CommonJS once its test has put DF_DATUM on a window shim.
//
// ── THE SHARED SUBSTRATE DECISION IS NOT RESTATED HERE ────────────────────
// Why pure helpers rather than a DOM harness: written out ONCE, in
// pins_recovery.js's header (the block marked CANONICAL). Coverage is
// behavioural, in dashboard/tests/js/window_chip.test.mjs.
//
// ── HONEST SCOPING ────────────────────────────────────────────────────────
// The chip appears only on tabs whose endpoints actually consume ?window=, and
// each offers only the windows its server vocabulary maps:
//   - Overview's cost spark, Performance, Merge and Costs obey
//     dashboard/src/dashboard/api/window.py::_WINDOW_DAYS — no 1h, no 90d.
//   - Burndown obeys dashboard/src/dashboard/api/burndown.py::_BURNDOWN_WINDOWS
//     — 90d, no all.
// WINDOW_SETS mirrors those two tables by hand; no parity test crosses the
// language boundary. Drift is VISIBLE rather than silent because every
// windowed payload echoes `WINDOW: {requested, served, days}`: a chip the
// server does not map reads "30d (Xd not available)" in the panel header and
// lights no chip, instead of quietly relabelling the server's default.
//
// A chip tab re-validates the window on every switch: a window it does not
// offer resets to DEFAULT_WINDOW. A chip-less tab keeps the user's last
// choice, since it has no set to validate against and its windowed reads are
// labelled from the echo.

// Module scope, no fallback, RENAMED — see the CANONICAL note in datum.js's
// header.
const { EM_DASH: WINDOW_LABEL_UNKNOWN } = window.DF_DATUM;

const WINDOW_SETS = Object.freeze({
  standard: Object.freeze(['24h', '7d', '30d', 'all']),
  burndown: Object.freeze(['24h', '7d', '30d', '90d']),
});

const DEFAULT_WINDOW = '24h';

// Each chip tab's offered windows, and the endpoint whose WINDOW echo its
// headers report. A tab absent here carries no global chip.
const TAB_WINDOWS = Object.freeze({
  overview: Object.freeze({ windows: WINDOW_SETS.standard, endpoint: '/api/v2/dashboard/costs' }),
  perf: Object.freeze({ windows: WINDOW_SETS.standard, endpoint: '/api/v2/dashboard/performance' }),
  merge: Object.freeze({ windows: WINDOW_SETS.standard, endpoint: '/api/v2/dashboard/merge-queue' }),
  cost: Object.freeze({ windows: WINDOW_SETS.standard, endpoint: '/api/v2/dashboard/costs' }),
  burn: Object.freeze({ windows: WINDOW_SETS.burndown, endpoint: '/api/v2/dashboard/burndown' }),
});

function windowForTab(tab, win) {
  const chip = TAB_WINDOWS[tab];
  return chip && !chip.windows.includes(win) ? DEFAULT_WINDOW : win;
}

// The echo recorded in data.js's per-endpoint receipt, or null when there is
// none yet or it is malformed. data.js transports it verbatim; this is where
// it is validated.
function windowEcho(receipts, path) {
  const echo = ((receipts || {})[path] || {}).window;
  if (!echo) return null;
  const { requested, served, days } = echo;
  const wellFormed = typeof requested === 'string'
    && typeof served === 'string'
    && Number.isFinite(days);
  return wellFormed ? echo : null;
}

function isHonoured(echo) {
  return echo.requested === echo.served;
}

function windowLabel(echo) {
  if (!echo) return WINDOW_LABEL_UNKNOWN;
  return isHonoured(echo) ? echo.served : `${echo.served} (${echo.requested} not available)`;
}

// The Toolbar lights the window actually SERVED, so a declined request lights
// none, and so does a window this tab does not offer.
function highlightedWindow(echo, windows) {
  return echo && isHonoured(echo) && windows.includes(echo.served) ? echo.served : null;
}

// A click is taken the moment the chosen window changes, but no chip lights
// for it until the endpoint serves it — which a slow fetch, a backoff or a
// paused poll can delay indefinitely. Until the echo answers the chosen window
// it is pending, so the click never reads as ignored.
function pendingWindow(win, echo, windows) {
  const answered = echo && echo.requested === win;
  return windows.includes(win) && !answered ? win : null;
}

function recentMergesCaption(shown, total, echo) {
  const of = Number.isFinite(total) ? total : WINDOW_LABEL_UNKNOWN;
  return `showing ${shown} of ${of} in ${windowLabel(echo)}`;
}

// Module-unique export const, never a bare `API` — see the
// shared-classic-script-scope note in graph_layout.js's header, enforced at
// runtime by dashboard/tests/js/classic_script_scope.test.mjs.
const WINDOW_CHIP_API = {
  WINDOW_SETS,
  DEFAULT_WINDOW,
  TAB_WINDOWS,
  windowForTab,
  windowEcho,
  windowLabel,
  highlightedWindow,
  pendingWindow,
  recentMergesCaption,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = WINDOW_CHIP_API;
}
if (typeof window !== 'undefined') {
  window.DF_WINDOW_CHIP = WINDOW_CHIP_API;
}
