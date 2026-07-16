/* Escalation Analytics tab — Origin / Lifespan / Workflow lifecycle panels.
 *
 * No JS test runner in this project (see scheduler_drawer.jsx comment).
 * Wiring contracts are verified via Python source-assertion tests in
 * dashboard/tests/test_tab_escalation_analytics.py.
 *
 * Load order: tabs.jsx → tab_escalation_analytics.jsx → app.jsx
 * Export:     window.DF_TABS.EscalationAnalyticsTab  (additive mutation of
 *             the object created by tabs.jsx; app.jsx destructures it last)
 */
const { useState: uS, useEffect: uE, useMemo } = React;
const DF = window.DF_DATA;
const { ProjectGroup, Segmented, fmtUptime, taskId } = window.DF_SHELL;
const C = window.DF_CHARTS;

// ── Local helpers (tab_escalations.jsx-compatible copies; not exported from
//    any namespace) ──

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
  // Backfill ids that arrive after mount (per_project starts [] and is
  // populated by the first poll, so groups would otherwise render collapsed).
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

// ── Window slicing ──
//
// The endpoint is deliberately unwindowed (no ?window= query param — the
// payload always carries the full daily-bucket history); this tab slices
// client-side over the 7d/28d/all toggle below. The cutoff is anchored to
// the payload's own `generated_at` clock, NEVER Date.now() — that keeps the
// frontend window consistent with the server's clock (which threads a
// single resolve_now() through the whole aggregation) and immune to
// browser-clock skew. 'all' means no cutoff at all.

const _WINDOW_DAYS = { '7d': 7, '28d': 28 };

// Returns the inclusive cutoff date (YYYY-MM-DD) for `win` relative to
// `generatedAt`, or null when the window is 'all' (no cutoff) or
// `generatedAt` is missing/unparseable.
function windowCutoffDate(generatedAt, win) {
  const days = _WINDOW_DAYS[win];
  if (!days || !generatedAt) return null;
  const end = new Date(generatedAt);
  if (isNaN(end.getTime())) return null;
  return new Date(end.getTime() - days * 86400000).toISOString().slice(0, 10);
}

// Filters a `{date: value}` object (e.g. daily_by_source, churn_daily) down
// to keys on/after the window cutoff. 'all' (or an unresolvable cutoff)
// returns the object unchanged.
function sliceDailyByWindow(dailyObj, generatedAt, win) {
  const cutoff = windowCutoffDate(generatedAt, win);
  if (!cutoff || !dailyObj) return dailyObj || {};
  const out = {};
  for (const date of Object.keys(dailyObj)) {
    if (date >= cutoff) out[date] = dailyObj[date];
  }
  return out;
}

// Filters an array of date-bearing rows (samples/flow_daily/esc_per_done_daily
// — shapes vary, so the caller supplies `dateOf` to extract the date string)
// down to rows on/after the window cutoff.
function sliceRowsByWindow(rows, generatedAt, win, dateOf) {
  const cutoff = windowCutoffDate(generatedAt, win);
  if (!cutoff || !rows) return rows || [];
  return rows.filter(row => dateOf(row) >= cutoff);
}

// ── Regime-marker overlay ──
//
// charts.jsx's primitives (LineChart/StackedAreaChart) plot points at evenly
// spaced x (index-based) with fixed padding padL=38/padR=12, and have no
// vertical-marker support — charts.jsx is not modified (see design
// decisions). RegimeMarkers renders vertical lines + labels as an
// absolutely-positioned overlay keyed off the same `labels` array the chart
// itself renders, so the two line up regardless of the chart's
// ResizeObserver-driven width.
const _CHART_PAD_L = 38, _CHART_PAD_R = 12;

function RegimeMarkers({ labels, markers }) {
  if (!markers || markers.length === 0 || !labels || labels.length < 2) return null;
  const n = labels.length;
  return (
    <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
      {markers.map((m, i) => {
        const idx = labels.indexOf(m.date);
        if (idx === -1) return null;
        const frac = idx / (n - 1);
        const left = `calc(${_CHART_PAD_L}px + ${frac} * (100% - ${_CHART_PAD_L + _CHART_PAD_R}px))`;
        return (
          <div key={i} style={{ position: 'absolute', left, top: 0, bottom: 20 }}>
            <div style={{ height: '100%', borderLeft: '1px dashed var(--fg-3)' }} />
            <span style={{
              position: 'absolute', top: 0, left: 3, fontSize: 9, color: 'var(--fg-3)',
              whiteSpace: 'nowrap', transform: 'translateY(-2px)',
            }}>{m.label}</span>
          </div>
        );
      })}
    </div>
  );
}

// TimeChart wraps a chart primitive with the RegimeMarkers overlay. Every
// time-axis chart in the panels below (Origin's filings StackedAreaChart,
// the Lifespan ECDF, Workflow's churn/esc-per-done LineCharts) renders
// through this wrapper so regime markers appear consistently everywhere
// without charts.jsx itself needing marker support.
function TimeChart({ labels, markers, children }) {
  return (
    <div style={{ position: 'relative' }}>
      {children}
      <RegimeMarkers labels={labels} markers={markers} />
    </div>
  );
}

// ── EscalationAnalyticsTab ──

function EscalationAnalyticsTab({ projectFilter }) {
  const analytics = DF.ESCALATION_ANALYTICS || { generated_at: null, parse_failures: 0, regime_markers: [], per_project: [] };

  const projects = (analytics.per_project || []).filter(p => {
    if (!projectFilter || projectFilter.length === 0) return true;
    return projectFilter.includes(p.project);
  });
  const projectIds = projects.map(p => p.project);

  const [openMap, toggle] = useOpenSet(projectIds, true, 'df.open.escanalytics');
  const [win, setWin] = usePersistedState('df.escanalytics.window', '28d');
  // Shared across every project's panels via TimeChart (Origin/Lifespan/
  // Workflow, added in later steps) — regime markers are a single
  // cross-project timeline, not per-project data.
  const regimeMarkers = analytics.regime_markers || [];

  return (
    <div style={{ position: 'relative' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10, flexWrap: 'wrap', gap: 8 }}>
        <Segmented
          options={[{ value: '7d', label: '7d' }, { value: '28d', label: '28d' }, { value: 'all', label: 'All' }]}
          value={win}
          onChange={setWin}
        />
        {analytics.parse_failures > 0 && (
          <span className="badge warn" title="Escalation archive records that failed to parse (skipped, not silently dropped)">
            ⚠ {analytics.parse_failures} parse failure{analytics.parse_failures !== 1 ? 's' : ''}
          </span>
        )}
      </div>
      {projects.map(p => (
        <div key={p.project} style={{ marginBottom: 8 }}>
          <ProjectGroup
            id={p.project}
            label={p.project}
            open={!!openMap[p.project]}
            onToggle={() => toggle(p.project)}
          >
            {/* Origin / Lifespan / Workflow panels land in later steps */}
          </ProjectGroup>
        </div>
      ))}
    </div>
  );
}

// ── Additive export (mutates the object created by tabs.jsx) ──
window.DF_TABS.EscalationAnalyticsTab = EscalationAnalyticsTab;
