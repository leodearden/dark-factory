/* Escalations tab — read-only queue view.
 *
 * No JS test runner in this project (see scheduler_drawer.jsx comment).
 * Wiring contracts are verified via Python source-assertion tests in
 * dashboard/tests/test_tab_escalations.py.
 *
 * Load order: tabs.jsx → tab_escalations.jsx → app.jsx
 * Export:     window.DF_TABS.EscalationsTab  (additive mutation of the object
 *             created by tabs.jsx; app.jsx destructures it last)
 */
const { useState: uS, useEffect: uE } = React;
const { ProjectGroup, taskId } = window.DF_SHELL;
const DF = window.DF_DATA;
const C = window.DF_CHARTS;

// ── Cross-tab focus helpers (module scope) ──
//
// Stable empty payload for the `DF.ESCALATIONS` falsy case.  Hoisted rather
// than written inline as `|| { subsections: [], ... }` because `escalations` is
// a dependency of the focus effect below: a fresh literal per render would make
// an effect that is meant to fire on payload ARRIVAL re-run on every paint.
// data.js's applyKey refuses null/undefined (data.js:153) so the fallback is
// unreachable today; hoisting keeps it stable if a change ever makes it
// reachable.
const ESC_EMPTY = { subsections: [], summary: { by_level: {}, by_status: {} } };

// Has the escalations payload ARRIVED, as opposed to still being data.js's
// pre-fetch seed?  Read from data.js's per-key first-success marker, which is
// the only sound signal:
//
//   * Not derivable from the payload's contents — the seed and a loaded but
//     genuinely EMPTY queue are structurally identical by design (ESCALATIONS
//     is deliberately not in STABLE_ARRAY_KEYS, data.js:96-105).
//   * Not derivable from object identity either.  Capturing `DF.ESCALATIONS`
//     at module-eval time and testing `payload !== SEED` races Babel: this is
//     a `type="text/babel"` module evaluated after DOMContentLoaded, so
//     startPolling()'s immediate first fetch can land BEFORE the capture,
//     freezing a real payload as the "seed" — and nothing unfreezes it while
//     polling is paused or the endpoint is in backoff, so the tab would claim
//     "still loading" above a fully-populated queue, indefinitely.
//   * Not derivable from the `df-data-refresh` event, which fires after
//     Promise.all even when the escalations fetch FAILED — that reports
//     arrival on a failure, producing a false miss.
//
// The failure direction is chosen on purpose: an endpoint stuck in backoff
// never sets the marker, so the UI keeps WAITING (and says so) rather than
// asserting a false "no longer in the queue" — the stronger claim, on evidence
// that cannot support it.
function escalationsLoaded() {
  return !!(DF.__loaded && DF.__loaded.ESCALATIONS);
}

function findEscalationRow(escalations, id) {
  for (const sub of (escalations.subsections || [])) {
    for (const row of (sub.escalations || [])) {
      if (row.id === id) return row;
    }
  }
  return null;
}

// ── Local helpers (tabs.jsx-compatible copies; not exported from any namespace) ──

function useOpenSet(ids, defaultOpen = true, storageKey = null) {
  const [openMap, setOpenMap] = uS(() => {
    let stored = {};
    if (storageKey) {
      try { stored = JSON.parse(localStorage.getItem(storageKey) || '{}') || {}; } catch (e) {}
    }
    const init = {};
    for (const id of ids) init[id] = id in stored ? !!stored[id] : defaultOpen;
    return init;
  });
  // Backfill ids that arrive after mount (ESCALATIONS.subsections starts [] and
  // is populated by the first poll, so groups would otherwise render collapsed).
  const idsKey = ids.join('\0');
  uE(() => {
    setOpenMap(m => {
      let patch = null;
      for (const id of ids) {
        if (!(id in m)) { if (!patch) patch = {}; patch[id] = defaultOpen; }
      }
      return patch ? { ...m, ...patch } : m;
    });
  }, [idsKey]); // eslint-disable-line react-hooks/exhaustive-deps
  uE(() => {
    if (storageKey) {
      try { localStorage.setItem(storageKey, JSON.stringify(openMap)); } catch (e) {}
    }
  }, [storageKey, openMap]);
  const toggle = id => setOpenMap(m => ({ ...m, [id]: !m[id] }));
  const setAll = v => setOpenMap(Object.fromEntries(ids.map(id => [id, v])));
  return [openMap, toggle, setAll];
}

function usePersistedState(storageKey, defaultValue) {
  const [v, setV] = uS(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      return raw === null ? defaultValue : JSON.parse(raw);
    } catch (e) { return defaultValue; }
  });
  uE(() => { try { localStorage.setItem(storageKey, JSON.stringify(v)); } catch (e) {} }, [storageKey, v]);
  return [v, setV];
}

function GroupAllToggle({ allOpen, onSetAll }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 6 }}>
      <button className="seg" onClick={() => onSetAll(!allOpen)}
        style={{ cursor: 'pointer', padding: '4px 10px', fontSize: 11, color: 'var(--fg-2)' }}>
        {allOpen ? '⌃ collapse all' : '⌄ expand all'}
      </button>
    </div>
  );
}

// ── Level / severity badge helpers ──

function levelClass(level) {
  if (level === 0) return 'esc-level-0';
  if (level === 1) return 'esc-level-1';
  return 'esc-level-2';
}

function sevClass(sev) {
  if (!sev) return '';
  const s = String(sev).toLowerCase();
  if (s === 'critical' || s === 'high') return 'esc-sev-high';
  if (s === 'medium') return 'esc-sev-medium';
  return 'esc-sev-low';
}

// ── Unreadable-queue-file notice ──
//
// Modelled on tab_memory_evals.jsx::IssuesNotice.  Expanded by default, on
// purpose: collapsing a degraded-state notice reproduces the silent degradation
// it exists to prevent (INV-2/INV-4, the 2658 parse_failures precedent).  Each
// entry names its path and the parse error — a bare count tells the operator
// something is wrong but not what.
//
// These records are deliberately NOT filtered by the level/status chips: a file
// that could not be parsed has neither a `level` nor a `status`, so there is
// nothing for matchesFilter to test, and routing them through the chips would
// let an arbitrary default decide whether the operator is told about corruption.
//
// At most SKIPPED_ROW_CAP paths are listed, with an overflow line for the rest.
// The normal case is one or two files, but a truncated write, a permission
// fault, or a half-synced mount degrades a whole directory at once — hundreds of
// always-expanded rows would push the actual escalation table far below the fold
// and degrade the same operator view this notice exists to serve.  The headline
// and the collapsed-group badge always state the TRUE total, so the cap shortens
// the list without ever understating the loss.
const SKIPPED_ROW_CAP = 20;

function SkippedNotice({ skipped }) {
  const rows = skipped || [];
  if (!(rows.length > 0)) return null;
  const shown = rows.slice(0, SKIPPED_ROW_CAP);
  const hidden = rows.length - shown.length;
  return (
    <div
      data-testid="escalation-skipped"
      style={{
        padding: '8px 12px',
        marginBottom: 8,
        border: '1px solid var(--line)',
        borderRadius: 4,
        background: 'var(--bg-2)',
        color: 'var(--fg-2)',
        fontFamily: 'var(--mono)',
        fontSize: 11,
      }}
    >
      <div style={{ color: 'var(--warn)', marginBottom: 4 }}>
        {rows.length} queue file(s) unreadable — the counts for this queue are short by that many
      </div>
      {shown.map((s, i) => (
        <div key={`${s.path}-${i}`} style={{ color: 'var(--fg-3)' }}>
          {s.path || '—'} — {s.error || '—'}
        </div>
      ))}
      {hidden > 0 && (
        <div style={{ color: 'var(--fg-3)', marginTop: 4 }}>
          …and {hidden} more (listing the first {SKIPPED_ROW_CAP})
        </div>
      )}
    </div>
  );
}

// ── Window slicing (trailing 7d, anchored to the payload's own generated_at
//    clock — never Date.now(), so the window stays consistent with the
//    server's clock and immune to browser-clock skew; same discipline as
//    tab_escalation_analytics.jsx) ──

// Returns the inclusive cutoff date (YYYY-MM-DD) `days` before `generatedAt`,
// or null when generatedAt is missing/unparseable (callers fall back to all
// rows via the slice helpers below — no crash before the first poll
// resolves).
function windowCutoffDate(generatedAt, days = 7) {
  if (!generatedAt) return null;
  const end = new Date(generatedAt);
  if (isNaN(end.getTime())) return null;
  return new Date(end.getTime() - days * 86400000).toISOString().slice(0, 10);
}

// Filters an array of `{date, ...}` rows (flow_daily / esc_per_done_daily)
// down to rows on/after the window cutoff. A null cutoff (unresolvable
// generatedAt) returns rows unchanged.
function sliceRowsByWindow(rows, cutoff) {
  if (!cutoff || !rows) return rows || [];
  return rows.filter(row => row.date >= cutoff);
}

// Filters a `{date: value}` object (churn_daily) down to keys on/after the
// window cutoff. A null cutoff returns the object unchanged.
function sliceDailyByWindow(dailyObj, cutoff) {
  if (!cutoff || !dailyObj) return dailyObj || {};
  const out = {};
  for (const date of Object.keys(dailyObj)) {
    if (date >= cutoff) out[date] = dailyObj[date];
  }
  return out;
}

// ── EscalationStatStrip — four-tile summary (benign rate, 6h breaches,
//    esc/done, churn), reading the ESCALATION_ANALYTICS payload already
//    wired into DF_DATA by the analytics tab (no duplicated computation) ──

function EscalationStatStrip({ analytics, projectFilter }) {
  const a = analytics || DF.ESCALATION_ANALYTICS;
  const projects = (a.per_project || []).filter(p => {
    if (!projectFilter || projectFilter.length === 0) return true;
    return projectFilter.includes(p.project);
  });

  // Trailing-7d window anchored to the payload's own generated_at clock. A
  // null cutoff (generated_at missing/unparseable, e.g. pre-first-poll)
  // falls back to all rows via the slice helpers' pass-through.
  const cutoff = windowCutoffDate(a.generated_at, 7);

  // (a) benign rate — workflow.flow_daily is the only per-day benign/
  // actionable series in the payload; sum n by class across every filtered
  // project's WINDOWED rows (cross-project rollup), also building a
  // per-date map (summed across projects) for the trend sparkline.
  let benignN = 0, actionableN = 0;
  const benignByDate = {}; // date -> { benign, actionable }
  for (const p of projects) {
    const flowDaily = sliceRowsByWindow((p.workflow || {}).flow_daily || [], cutoff);
    for (const row of flowDaily) {
      const bucket = benignByDate[row.date] || (benignByDate[row.date] = { benign: 0, actionable: 0 });
      if (row.class === 'benign') { benignN += row.n; bucket.benign += row.n; }
      else if (row.class === 'actionable') { actionableN += row.n; bucket.actionable += row.n; }
    }
  }
  const benignDenom = benignN + actionableN;
  const benignRate = benignDenom > 0 ? benignN / benignDenom : null;
  // Per-day benign rate, one point per date that appears in flow_daily (i.e.
  // had at least one row that day). Windowed dates with NO flow_daily rows
  // are OMITTED here, not zero-filled, so the sparkline is not fully
  // window-aligned when flow_daily has gaps — it only covers dates the
  // payload actually reported.
  const benignSpark = Object.keys(benignByDate).sort().map(d => {
    const { benign, actionable } = benignByDate[d];
    const denom = benign + actionable;
    return denom > 0 ? benign / denom : 0;
  });

  // Stamped-share hint — NO per-day provenance series exists in the payload,
  // so this is an ALL-TIME aggregate from origin.sources[], weighted by each
  // source's classified count (benign + actionable). A coarse adoption
  // indicator, not a windowed figure.
  let stampedWeighted = 0, classifiedTotal = 0;
  for (const p of projects) {
    for (const s of (p.origin || {}).sources || []) {
      const classified = (s.benign || 0) + (s.actionable || 0);
      stampedWeighted += (s.stamped_share || 0) * classified;
      classifiedTotal += classified;
    }
  }
  const stampedPct = classifiedTotal > 0 ? Math.round((stampedWeighted / classifiedTotal) * 100) : null;

  // (b) 6h-breach — live pending queue (lifespan.open_items), NOT windowed;
  // the payload carries no historical breach series.
  let openItems = [];
  for (const p of projects) {
    openItems = openItems.concat((p.lifespan || {}).open_items || []);
  }
  const breachCount = openItems.filter(item => item.breach_6h).length;

  // (c) esc-per-done — aggregate ratio sum(filings)/sum(done) over the
  // WINDOWED rows, NOT a mean of daily ratios (undefined/biased on
  // low-volume or done==0 days). Also builds a per-date map (summed across
  // projects — dates can repeat across per_project entries) for the trend
  // sparkline; re-derived from filings/done rather than the payload's
  // per-project row.ratio, since that field isn't valid post-rollup.
  let filingsSum = 0, doneSum = 0;
  const epdByDate = {}; // date -> { filings, done }
  for (const p of projects) {
    const epd = sliceRowsByWindow((p.workflow || {}).esc_per_done_daily || [], cutoff);
    for (const row of epd) {
      filingsSum += row.filings || 0;
      doneSum += row.done || 0;
      const bucket = epdByDate[row.date] || (epdByDate[row.date] = { filings: 0, done: 0 });
      bucket.filings += row.filings || 0;
      bucket.done += row.done || 0;
    }
  }
  const escPerDone = doneSum > 0 ? filingsSum / doneSum : null;
  // Null ratio (done == 0 that day) is OMITTED rather than plotted as a
  // misleading zero — same precedent as tab_escalation_analytics.jsx.
  const epdSpark = Object.keys(epdByDate).sort()
    .map(d => epdByDate[d])
    .filter(b => b.done > 0)
    .map(b => b.filings / b.done);

  // (d) churn-24h rate — sum(WINDOWED churn_daily)/sum(WINDOWED
  // esc_per_done_daily filings); both are keyed by filed-date
  // (date(timestamp)), so they reconcile. Also builds a per-date map (summed
  // across projects) for the trend sparkline.
  let churnSum = 0;
  const churnByDate = {};
  for (const p of projects) {
    const churnDaily = sliceDailyByWindow((p.workflow || {}).churn_daily || {}, cutoff);
    for (const [d, n] of Object.entries(churnDaily)) {
      churnSum += n;
      churnByDate[d] = (churnByDate[d] || 0) + n;
    }
  }
  const churnRate = filingsSum > 0 ? churnSum / filingsSum : null;
  // Per-day churn RATE (that day's churn / that day's filings), matching the
  // quantity the tile displays — plotting raw per-day counts here would show
  // a different quantity than the tile value and mislead when daily filing
  // volume varies. Reuses epdByDate's per-day filings (already collected for
  // the esc-per-done tile, same filed-date keying). Days with zero filings
  // that day are OMITTED (undefined rate), mirroring epdSpark's done==0
  // omission above.
  const churnSpark = Object.keys(churnByDate).sort()
    .filter(d => (epdByDate[d] || {}).filings > 0)
    .map(d => churnByDate[d] / epdByDate[d].filings);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 10, marginBottom: 10 }}>
      <C.StatTile
        label="benign rate"
        value={benignRate != null ? `${Math.round(benignRate * 100)}%` : '—'}
        hint={stampedPct != null ? `${stampedPct}% stamped` : undefined}
        spark={benignSpark}
        sparkColor={C.PALETTE.ok}
      />
      <C.StatTile
        label="6h breaches"
        value={breachCount}
        hint={`of ${openItems.length} pending`}
      />
      <C.StatTile
        label="esc / done"
        value={escPerDone != null ? escPerDone.toFixed(2) : '—'}
        spark={epdSpark}
        sparkColor={C.PALETTE.accent}
      />
      <C.StatTile
        label="churn 24h"
        value={churnRate != null ? `${Math.round(churnRate * 100)}%` : '—'}
        spark={churnSpark}
        sparkColor={C.PALETTE.bad}
      />
    </div>
  );
}

// ── EscalationsTab ──

function EscalationsTab({ projectFilter, focusId, onFocusConsumed }) {
  // ESC_EMPTY is hoisted to module scope; see the note at its declaration.
  // Note it is NOT a "loaded" signal: arrival is read from data.js's
  // first-success marker (escalationsLoaded), so falling back to this
  // placeholder cannot be mistaken for an arrived payload.
  const escalations = DF.ESCALATIONS || ESC_EMPTY;

  // Filter subsections by project when projectFilter is active.
  // Orchestrator subsections: filter by subsection label (label == project name).
  // Reconciliation subsections: always include — their rows are filtered per-row
  // by row.project below, because the subsection label is 'fused-memory' (not a
  // project name) and rows may belong to different owning projects.
  const subsections = escalations.subsections.filter(s => {
    if (!projectFilter || projectFilter.length === 0) return true;
    if (s.kind === 'reconciliation') return true;
    return projectFilter.includes(s.label);
  });
  const subsectionIds = subsections.map(s => s.id);

  const [openMap, toggle, setAll] = useOpenSet(subsectionIds, true, 'df.open.esc');
  const allOpen = subsectionIds.length > 0 && subsectionIds.every(id => openMap[id]);

  // Sort state (key: 'task' | 'timestamp', dir: 'asc' | 'desc')
  const [sort, setSort] = usePersistedState('df.esc.sort', { key: 'task', dir: 'asc' });
  const flipDir = () => setSort(s => ({ ...s, dir: s.dir === 'asc' ? 'desc' : 'asc' }));

  // Filter state (levels 0/1/2, statuses pending/resolved/dismissed)
  const [filter, setFilter] = usePersistedState('df.esc.filter', {
    levels: { 0: true, 1: true, 2: true },
    statuses: { pending: true, resolved: false, dismissed: false },
  });
  const toggleLevel = lv => setFilter(f => ({ ...f, levels: { ...f.levels, [lv]: !f.levels[lv] } }));
  const toggleStatus = st => setFilter(f => ({ ...f, statuses: { ...f.statuses, [st]: !f.statuses[st] } }));

  function matchesFilter(row) {
    const lvKey = row.level != null ? row.level : 0;
    const stKey = row.status || 'pending';
    return !!(filter.levels[lvKey] && filter.statuses[stKey]);
  }

  function sortRows(rows) {
    const mul = sort.dir === 'asc' ? 1 : -1;
    return [...rows].sort((a, b) => {
      // Primary: numeric task_id (NaN/null sorts last regardless of direction)
      const aId = Number(a.task_id);
      const bId = Number(b.task_id);
      const aValid = !isNaN(aId);
      const bValid = !isNaN(bId);
      if (!aValid && !bValid) {
        // Both invalid: secondary tie-break by timestamp, respecting sort direction
        const ts = (a.timestamp || '') < (b.timestamp || '') ? -1 : (a.timestamp || '') > (b.timestamp || '') ? 1 : 0;
        return mul * ts;
      }
      if (!aValid) return 1;  // invalid always last, regardless of direction
      if (!bValid) return -1; // invalid always last, regardless of direction
      if (aId !== bId) return mul * (aId - bId);
      // Tie-break by timestamp, respecting sort direction
      const ts = (a.timestamp || '') < (b.timestamp || '') ? -1 : (a.timestamp || '') > (b.timestamp || '') ? 1 : 0;
      return mul * ts;
    });
  }

  // Selected row for sidebar
  const [selected, setSelected] = uS(null);

  // A focus id that matched no row. Held rather than dropped: the operator
  // clicked a link and is owed an answer either way.
  const [focusMiss, setFocusMiss] = uS(null);

  // Cross-tab focus handoff (from the memory-eval escalation links in
  // tab_memory_evals.jsx, lifted through app.jsx). Search every subsection for
  // the row and open the existing detail sidebar with it.
  //
  // The focus is consumed once a DECISION is REACHABLE, never before. Keyed on
  // `escalations` as well as `focusId`: while the payload is still the pre-fetch
  // seed the effect declines to decide, and the dep's per-poll identity change
  // (applyKey replaces the reference) is what re-runs it — so a cold load, or an
  // endpoint in backoff, no longer silently eats the focus.
  //
  // Once the payload is in, both outcomes are reported. A miss is real: the
  // escalation may have resolved between the poll that produced the link and
  // the click. It is still CONSUMED on a miss, because leaving it set would
  // reopen a stale drawer on every later visit to this tab — but it is now
  // recorded and rendered rather than dropped in silence.
  uE(() => {
    if (!focusId) return;
    if (!escalationsLoaded()) return;
    const found = findEscalationRow(escalations, focusId);
    if (found) { setSelected(found); setFocusMiss(null); }
    else { setFocusMiss(focusId); }
    if (onFocusConsumed) onFocusConsumed();
  }, [focusId, escalations]);

  // Global summary from top-level data
  const gs = escalations.summary || {};
  const byLevel = gs.by_level || {};
  const byStatus = gs.by_status || {};

  return (
    <div style={{ position: 'relative' }}>
      <EscalationStatStrip analytics={DF.ESCALATION_ANALYTICS} projectFilter={projectFilter} />

      {/* Cross-tab focus feedback. Two states, not one: a click that lands
          before the escalations payload does is WAITING, not a miss, and
          saying "no longer in the queue" while the endpoint is still in
          backoff would be a false claim. Both name the id, so the operator
          can see which link they followed. */}
      {focusId && !escalationsLoaded() && (
        <div className="badge" data-testid="esc-focus-pending" style={{ marginBottom: 8 }}>
          waiting for the escalation queue to load — will open{' '}
          <span className="mono">{focusId}</span> when it arrives
        </div>
      )}
      {focusMiss && (
        <div className="badge warn" data-testid="esc-focus-miss" style={{ marginBottom: 8 }}>
          <span className="mono">{focusMiss}</span> is not in the queue — it was
          likely resolved between the poll that produced the link and the click
          <button
            type="button"
            className="chip"
            style={{ marginLeft: 8 }}
            onClick={() => setFocusMiss(null)}
          >
            dismiss
          </button>
        </div>
      )}

      {/* Controls header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10, flexWrap: 'wrap' }}>
        {/* Level filter chips */}
        <span style={{ fontSize: 11, color: 'var(--fg-3)' }}>Level:</span>
        {[0, 1, 2].map(lv => (
          <button key={lv} className={`chip${filter.levels[lv] ? ' on' : ''}`}
            onClick={() => toggleLevel(lv)}
            title={`Level ${lv}`}>
            L{lv}
          </button>
        ))}
        <span style={{ marginLeft: 6, fontSize: 11, color: 'var(--fg-3)' }}>Status:</span>
        {['pending', 'resolved', 'dismissed'].map(st => (
          <button key={st} className={`chip${filter.statuses[st] ? ' on' : ''}`}
            onClick={() => toggleStatus(st)}
            title={st}>
            {st}
          </button>
        ))}
        {/* Sort toggle */}
        <button className="seg" onClick={flipDir}
          style={{ marginLeft: 8, cursor: 'pointer', padding: '4px 10px', fontSize: 11, color: 'var(--fg-2)' }}>
          task {sort.dir === 'asc' ? '↑' : '↓'}
        </button>
        {/* Summary pills */}
        <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--fg-3)' }}>
          {byStatus.pending || 0} pending · {byLevel[1] || 0} L1 · {byLevel[2] || 0} L2
          {/* Global, like the pips beside it: read from the unfiltered top-level
              summary, so with a project filter active this can count files in
              queues that are not rendered below.  Titled rather than re-derived
              from the filtered subsections — a corruption signal that shrinks
              when you narrow the view would understate the fleet's actual
              degraded state, which is the failure this whole notice exists to
              prevent.  The title states the mismatch instead of leaving the
              operator to infer it from a count with no visible source. */}
          {(gs.skipped_count || 0) > 0 && (
            <span title="Across all queues, including any hidden by the project filter">
              {' '}· {gs.skipped_count} unreadable
            </span>
          )}
        </span>
      </div>

      {/* Expand / collapse all */}
      <GroupAllToggle allOpen={allOpen} onSetAll={setAll} />

      {/* Subsection groups */}
      {subsections.map(sec => {
        const secSummary = sec.summary || {};
        const secByLevel = secSummary.by_level || {};
        const secByStatus = secSummary.by_status || {};
        const filteredRows = sortRows((sec.escalations || []).filter(row => {
          if (!matchesFilter(row)) return false;
          // Reconciliation subsections: filter by row.project (not subsection label).
          if (sec.kind === 'reconciliation' && projectFilter && projectFilter.length > 0) {
            return !row.project || projectFilter.includes(row.project);
          }
          return true;
        }));

        // Queue files the reader could not parse.  Read unfiltered — see the
        // SkippedNotice comment: a corrupt file has no level or status to chip-filter on.
        const skipped = sec.skipped || [];

        const summary = (
          <>
            <span className="pip" style={{ fontSize: 10 }}>{secByStatus.pending || 0} pending</span>
            {(secByLevel[1] || 0) > 0 && (
              <span className="pip"><span className="badge warn" style={{ fontSize: 9 }}>L1 · {secByLevel[1]}</span></span>
            )}
            {(secByLevel[2] || 0) > 0 && (
              <span className="pip"><span className="badge bad" style={{ fontSize: 9 }}>L2 · {secByLevel[2]}</span></span>
            )}
            {skipped.length > 0 && (
              <span className="pip"><span className="badge bad" style={{ fontSize: 9 }}>{skipped.length} unreadable</span></span>
            )}
            <span className="mono" style={{ color: 'var(--fg-3)', fontSize: 10 }}>{sec.kind}</span>
          </>
        );

        return (
          <div key={sec.id} style={{ marginBottom: 8 }}>
            <ProjectGroup
              id={sec.id}
              label={sec.label}
              open={!!openMap[sec.id]}
              onToggle={() => toggle(sec.id)}
              summary={summary}
            >
              {/* Above and outside the empty-state ternary on purpose: a group
                  rendering "No escalations match current filters" while holding an
                  unreadable file is precisely the looks-empty-but-isn't case. */}
              <SkippedNotice skipped={skipped} />
              {filteredRows.length === 0 ? (
                <div style={{ fontSize: 12, color: 'var(--fg-3)', padding: '8px 0' }}>
                  No escalations match current filters.
                </div>
              ) : (
                <table className="tbl" style={{ tableLayout: 'fixed', width: '100%' }}>
                  <colgroup>
                    <col style={{ width: 70 }} />
                    <col style={{ width: 60 }} />
                    <col style={{ width: 80 }} />
                    <col style={{ width: 80 }} />
                    <col />
                    <col style={{ width: 100 }} />
                  </colgroup>
                  <thead>
                    <tr>
                      <th style={{ cursor: 'pointer', userSelect: 'none' }} onClick={flipDir}>
                        Task {sort.dir === 'asc' ? '↑' : '↓'}
                      </th>
                      <th>Level</th>
                      <th>Status</th>
                      <th>Severity</th>
                      <th>Summary</th>
                      <th>Project</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredRows.map(row => (
                      <tr key={row.id} style={{ cursor: 'pointer' }}
                        onClick={() => setSelected(row)}>
                        <td className="mono">{row.task_id ? `T-${taskId(String(row.task_id))}` : '—'}</td>
                        <td>
                          <span className={`badge ${levelClass(row.level)}`}
                            style={{ fontSize: 10 }}>
                            L{row.level ?? 0}
                          </span>
                        </td>
                        <td>
                          <span className={`badge ${row.status === 'pending' ? 'warn' : row.status === 'resolved' ? 'ok' : ''}`}
                            style={{ fontSize: 10 }}>
                            {row.status || '—'}
                          </span>
                        </td>
                        <td>
                          <span className={`badge ${sevClass(row.severity)}`}
                            style={{ fontSize: 10 }}>
                            {row.severity || '—'}
                          </span>
                        </td>
                        <td style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                          title={row.summary}>
                          {row.summary || '—'}
                        </td>
                        <td style={{ fontSize: 11, color: 'var(--fg-3)' }}>
                          {row.project || '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </ProjectGroup>
          </div>
        );
      })}

      {/* Detail sidebar */}
      {selected && (
        <EscalationSidebar row={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}

// ── EscalationSidebar — read-only detail panel ──

function EscalationSidebar({ row, onClose }) {
  if (!row) return null;
  const task = row.task || null;
  return (
    <div className="sched-drawer" role="dialog" aria-label={`Escalation detail for ${row.id}`}>
      {/* Header */}
      <div className="sched-drawer-head">
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span className={`badge ${levelClass(row.level)}`} style={{ fontSize: 10 }}>L{row.level ?? 0}</span>
            <span className={`badge ${row.status === 'pending' ? 'warn' : row.status === 'resolved' ? 'ok' : ''}`}
              style={{ fontSize: 10 }}>{row.status || '—'}</span>
            <span className={`badge ${sevClass(row.severity)}`} style={{ fontSize: 10 }}>{row.severity || '—'}</span>
          </div>
          <div style={{ fontSize: 13, color: 'var(--fg-0)', fontWeight: 500,
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
            title={row.summary}>
            {row.summary || '—'}
          </div>
          <div style={{ display: 'flex', gap: 12, marginTop: 4, fontSize: 10, color: 'var(--fg-3)' }}>
            {row.category && <span>category: <span className="mono">{row.category}</span></span>}
            {row.agent_role && <span>agent: <span className="mono">{row.agent_role}</span></span>}
            {row.worktree && <span>worktree: <span className="mono">{row.worktree}</span></span>}
          </div>
        </div>
        <button className="sched-drawer-close" onClick={onClose} title="Close" aria-label="Close drawer">×</button>
      </div>

      {/* Body */}
      <div className="sched-drawer-body">
        {/* Detail */}
        {row.detail && (
          <div className="sched-drawer-section">
            <div style={{ fontSize: 10, color: 'var(--fg-3)', marginBottom: 4 }}>Detail</div>
            <div style={{ fontSize: 12, color: 'var(--fg-1)', whiteSpace: 'pre-wrap' }}>{row.detail}</div>
          </div>
        )}

        {/* Suggested action */}
        {row.suggested_action && (
          <div className="sched-drawer-section">
            <div style={{ fontSize: 10, color: 'var(--fg-3)', marginBottom: 4 }}>Suggested Action</div>
            <div style={{ fontSize: 12, color: 'var(--fg-1)' }}>{row.suggested_action}</div>
          </div>
        )}

        {/* Resolution */}
        {row.resolution && (
          <div className="sched-drawer-section">
            <div style={{ fontSize: 10, color: 'var(--fg-3)', marginBottom: 4 }}>Resolution</div>
            <div style={{ fontSize: 12, color: 'var(--fg-1)' }}>{row.resolution}</div>
          </div>
        )}

        {/* Workflow state */}
        {row.workflow_state && (
          <div className="sched-drawer-section">
            <div style={{ display: 'flex', gap: 16, fontSize: 11 }}>
              <span style={{ color: 'var(--fg-3)' }}>Workflow:</span>
              <span className="mono">{row.workflow_state}</span>
            </div>
          </div>
        )}

        {/* Project */}
        {row.project && (
          <div className="sched-drawer-section">
            <div style={{ fontSize: 10, color: 'var(--fg-3)', marginBottom: 2 }}>Project</div>
            <div style={{ fontSize: 12, color: 'var(--fg-1)' }}>{row.project}</div>
          </div>
        )}

        {/* Linked task */}
        <div className="sched-drawer-section">
          <div style={{ fontSize: 10, color: 'var(--fg-3)', marginBottom: 4 }}>Linked Task</div>
          {row.task_id && row.task_unresolved ? (
            // task_id present but could not be resolved — distinct from "no task linked"
            <div style={{ fontSize: 12, color: 'var(--fg-3)' }}>
              Task ID <span className="mono">{row.task_id}</span> could not be resolved
              {row.worktree && <span> (worktree: <span className="mono">{row.worktree}</span>)</span>}.
            </div>
          ) : task ? (
            <div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 4 }}>
                <span className="mono" style={{ fontSize: 11, color: 'var(--fg-3)' }}>T-{taskId(String(task.id || ''))}</span>
                <span className={`badge ${task.status === 'in-progress' ? 'ok' : task.status === 'blocked' ? 'bad' : ''}`}
                  style={{ fontSize: 10 }}>{task.status || '—'}</span>
              </div>
              <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--fg-0)', marginBottom: 4 }}>{task.title || '—'}</div>
              {task.description && (
                <div style={{ fontSize: 11, color: 'var(--fg-2)', whiteSpace: 'pre-wrap', maxHeight: 120, overflow: 'auto' }}>
                  {task.description}
                </div>
              )}
            </div>
          ) : (
            <div style={{ fontSize: 12, color: 'var(--fg-3)' }}>No linked task.</div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Additive export (mutates the object created by tabs.jsx) ──
window.DF_TABS.EscalationsTab = EscalationsTab;
