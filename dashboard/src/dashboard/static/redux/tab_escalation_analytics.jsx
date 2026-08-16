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
const { useState: uS, useEffect: uE } = React;
const DF = window.DF_DATA;
const { ProjectGroup, Segmented, fmtUptime, fmtDateTime, taskId } = window.DF_SHELL;
const C = window.DF_CHARTS;
const { LifecycleFlowDiagram } = window.DF_ESC_FLOW || {};

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

// tier_weekly is keyed by ISO-8601 week ("YYYY-Www", matching Python's
// `isocalendar()`), NOT a calendar date — comparing week keys against a
// "YYYY-MM-DD" cutoff would compare apples to oranges, so the cutoff date is
// first converted to its own week key (nearest-Thursday algorithm) and week
// keys are then compared as strings (valid since both sides are zero-padded
// "YYYY-Www").
function dateToIsoWeekKey(dateStr) {
  const d = new Date(dateStr + 'T00:00:00Z');
  if (isNaN(d.getTime())) return null;
  const isoDayNum = (d.getUTCDay() + 6) % 7 + 1; // Mon=1..Sun=7
  const thursday = new Date(d.getTime());
  thursday.setUTCDate(d.getUTCDate() + 4 - isoDayNum);
  const isoYear = thursday.getUTCFullYear();
  const jan1 = new Date(Date.UTC(isoYear, 0, 1));
  const week = Math.ceil(((thursday.getTime() - jan1.getTime()) / 86400000 + 1) / 7);
  return `${isoYear}-W${String(week).padStart(2, '0')}`;
}

// Filters a `{week: value}` object (tier_weekly) down to weeks on/after the
// window cutoff's own week. 'all' (or an unresolvable cutoff) returns the
// object unchanged.
function sliceWeeklyByWindow(weeklyObj, generatedAt, win) {
  const cutoffDate = windowCutoffDate(generatedAt, win);
  const cutoffWeek = cutoffDate && dateToIsoWeekKey(cutoffDate);
  if (!cutoffWeek || !weeklyObj) return weeklyObj || {};
  const out = {};
  for (const week of Object.keys(weeklyObj)) {
    if (week >= cutoffWeek) out[week] = weeklyObj[week];
  }
  return out;
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

// ── Origin panel ──
//
// Top-N-by-source filings/day (long tail folded into an 'other' stack) + a
// benign-rate table (one row per origin.sources[] entry).
const _TOP_N_SOURCES = 6;
// Categorical color cycling — same convention as tabs.jsx's per-account/
// per-status donut/legend coloring ([CP.accent, CP.ok, CP.warn, CP.info,
// CP.accent2, CP.bad][i % 6]).
const _CATEGORY_COLORS = [
  C.PALETTE.accent, C.PALETTE.ok, C.PALETTE.warn, C.PALETTE.info, C.PALETTE.accent2, C.PALETTE.bad,
];

function OriginPanel({ origin, win, generatedAt, regimeMarkers }) {
  const daily = sliceDailyByWindow(origin.daily_by_source, generatedAt, win);
  const dates = Object.keys(daily).sort();

  // Rank sources by total windowed filings; top-N get their own stack, the
  // long tail folds into a single 'other' band so the chart stays readable
  // regardless of how many distinct agent_role values the archive has seen.
  const totalsBySource = {};
  for (const d of dates) {
    for (const [src, n] of Object.entries(daily[d] || {})) {
      totalsBySource[src] = (totalsBySource[src] || 0) + n;
    }
  }
  const ranked = Object.entries(totalsBySource).sort((a, b) => b[1] - a[1]).map(([src]) => src);
  const topSources = ranked.slice(0, _TOP_N_SOURCES);
  const otherSources = ranked.slice(_TOP_N_SOURCES);

  const stacks = topSources.map((src, i) => ({
    key: src,
    color: _CATEGORY_COLORS[i % _CATEGORY_COLORS.length],
    values: dates.map(d => (daily[d] || {})[src] || 0),
  }));
  if (otherSources.length > 0) {
    stacks.push({
      key: 'other',
      color: C.PALETTE.fg3,
      values: dates.map(d => otherSources.reduce((sum, src) => sum + ((daily[d] || {})[src] || 0), 0)),
    });
  }

  // Benign-rate table: sorted DESC by benign COUNT (volume × rate) — a
  // source with a high rate but tiny volume sorts below a high-volume,
  // slightly-less-benign source, which is the more actionable ordering for
  // "where should triage attention go".
  const rows = [...(origin.sources || [])].sort((a, b) => b.benign - a.benign);

  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ fontSize: 11, color: 'var(--fg-3)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
        Origin — filings by source
      </div>
      {stacks.length > 0 && (
        <TimeChart labels={dates} markers={regimeMarkers}>
          <C.StackedAreaChart stacks={stacks} labels={dates} formatX={fmtDateTime} />
        </TimeChart>
      )}
      <table className="tbl" style={{ marginTop: 10 }}>
        <thead>
          <tr>
            <th>Source</th>
            <th className="num">Filings</th>
            <th>Benign / actionable</th>
            <th>Stamped / inferred</th>
            <th>Trend</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {rows.map(s => {
            const benignPct = Math.round((s.benign_rate || 0) * 100);
            const stampedPct = Math.round((s.stamped_share || 0) * 100);
            return (
              <tr key={s.source}>
                <td>{s.source}</td>
                <td className="num mono">{s.filings}</td>
                <td>
                  <div style={{ display: 'flex', height: 6, width: 90, background: 'var(--bg-2)', borderRadius: 3, overflow: 'hidden' }}>
                    <div style={{ width: `${benignPct}%`, background: C.PALETTE.ok }} title={`benign ${s.benign}`} />
                    <div style={{ width: `${100 - benignPct}%`, background: C.PALETTE.bad }} title={`actionable ${s.actionable}`} />
                  </div>
                </td>
                <td className="mono" style={{ fontSize: 10, color: 'var(--fg-3)' }}>{stampedPct}% / {100 - stampedPct}%</td>
                <td style={{ width: 90, height: 22 }}><C.Sparkline values={s.daily_spark || []} /></td>
                <td>{s.predictably_benign && <span className="badge ok">predictably benign</span>}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Lifespan panel ──
//
// Percentile tiles by level + a resolver-tier-overlaid ECDF of resolution
// time (log-x, since escalation lifetimes span seconds to weeks) with a
// vertical 6h freshness marker, an open-items list ranked by pending age,
// and a render-when-present filed→triaged→resolved segment block (2555
// forward-compat).
const _RESOLVER_TIERS = ['human', 'cascade', 'auto-watcher', 'steward', 'reaper-sweep', 'unknown', 'other-auto'];
const _TIER_COLORS = {
  human: C.PALETTE.ok,
  cascade: C.PALETTE.accent,
  'auto-watcher': C.PALETTE.info,
  steward: C.PALETTE.accent2,
  'reaper-sweep': C.PALETTE.warn,
  unknown: C.PALETTE.fg3,
  'other-auto': C.PALETTE.bad,
};

// Any resolver tier the backend emits that isn't in the whitelist above
// (e.g. a future addition to escalation.classify.classify_resolver_tier, or
// a parse anomaly) folds into 'other-auto' rather than being dropped.
// LifespanPanel's ECDF and WorkflowPanel's 100%-normalized stack both
// iterate _RESOLVER_TIERS only — an un-folded unknown tier would silently
// vanish from the ECDF, and would make the normalized stack's bands sum to
// less than 1.0 for any week that touched it.
function resolverTierKey(tier) {
  return _RESOLVER_TIERS.includes(tier) ? tier : 'other-auto';
}

function LifespanPanel({ lifespan, win, generatedAt }) {
  const levels = Object.keys(lifespan.percentiles_by_level || {}).sort();
  // lifespan.samples rows are [date, tier, level, secs] — date is date(resolved_at).
  const samples = sliceRowsByWindow(lifespan.samples || [], generatedAt, win, row => row[0]);

  // Log-spaced ECDF threshold grid: ~60s .. ~30d (2,592,000s). Even spacing in
  // log-space lets the (index-spaced) LineChart primitive read as a log-x
  // axis without charts.jsx needing log-scale support.
  const GRID_STEPS = 24;
  const logLo = Math.log(60), logHi = Math.log(2592000);
  const grid = Array.from({ length: GRID_STEPS }, (_, i) => Math.exp(logLo + ((logHi - logLo) * i) / (GRID_STEPS - 1)));
  const gridLabels = grid.map(fmtUptime);

  // Vertical 6h (21600s) freshness marker — the grid index whose threshold
  // lands nearest 21600, rendered via the same overlay technique as
  // RegimeMarkers/TimeChart (a synthetic single-marker array).
  const BREACH_SECS = 21600;
  let breachIdx = 0;
  for (let i = 1; i < grid.length; i++) {
    if (Math.abs(grid[i] - BREACH_SECS) < Math.abs(grid[breachIdx] - BREACH_SECS)) breachIdx = i;
  }
  const breachMarker = [{ date: gridLabels[breachIdx], label: '6h' }];

  const secsByTier = {};
  for (const row of samples) {
    const tier = resolverTierKey(row[1]), secs = row[3];
    (secsByTier[tier] = secsByTier[tier] || []).push(secs);
  }
  const series = _RESOLVER_TIERS
    .filter(t => (secsByTier[t] || []).length > 0)
    .map(t => {
      const sorted = [...secsByTier[t]].sort((a, b) => a - b);
      const n = sorted.length;
      return {
        key: t,
        color: _TIER_COLORS[t] || C.PALETTE.fg3,
        values: grid.map(threshold => sorted.filter(s => s <= threshold).length / n),
      };
    });

  const openItems = [...(lifespan.open_items || [])].sort((a, b) => b.age_secs - a.age_secs);

  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ fontSize: 11, color: 'var(--fg-3)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
        Lifespan — resolution time
      </div>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 10 }}>
        {levels.map(level => {
          const pct = lifespan.percentiles_by_level[level];
          return (
            <C.StatTile
              key={level}
              label={`L${level} resolution time`}
              value={fmtUptime(pct.p50)}
              hint={`p50 · p90 ${fmtUptime(pct.p90)}`}
            />
          );
        })}
        {lifespan.l1_to_l2_promotion && lifespan.l1_to_l2_promotion.count > 0 && (
          <C.StatTile
            label="L1→L2 promotion"
            value={fmtUptime(lifespan.l1_to_l2_promotion.p50_secs)}
            hint={`p50 · p90 ${fmtUptime(lifespan.l1_to_l2_promotion.p90_secs)}`}
          />
        )}
      </div>
      {series.length > 0 && (
        <>
          <div style={{ fontSize: 10, color: 'var(--fg-3)', marginBottom: 4 }}>
            Resolution-time ECDF by resolver tier (log-x; dashed line = 6h freshness)
          </div>
          <TimeChart labels={gridLabels} markers={breachMarker}>
            <C.LineChart series={series} labels={gridLabels} formatY={v => `${Math.round(v * 100)}%`} formatX={v => v} />
          </TimeChart>
        </>
      )}
      {lifespan.samples_downsampled && (
        <div style={{ fontSize: 10, color: 'var(--fg-3)', marginTop: 4 }}>
          showing {samples.length} of {lifespan.samples_total} resolution samples (downsampled)
        </div>
      )}
      <div style={{ marginTop: 10 }}>
        <div style={{ fontSize: 10, color: 'var(--fg-3)', marginBottom: 4 }}>Open items — ranked by age</div>
        <table className="tbl">
          <thead>
            <tr><th>ID</th><th>Task</th><th>Level</th><th className="num">Age</th></tr>
          </thead>
          <tbody>
            {openItems.map(item => (
              <tr key={item.id}>
                <td className="mono">{item.id}</td>
                <td className="mono">{taskId(item.task_id)}</td>
                <td>
                  <span className={`badge esc-level-${item.level === 2 ? 2 : item.level === 1 ? 1 : 0}`}>
                    L{item.level}
                  </span>
                </td>
                <td className="num mono">
                  {fmtUptime(item.age_secs)}
                  {item.breach_6h && <span className="badge bad" style={{ marginLeft: 6, fontSize: 9 }}>6h+</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {lifespan.triage_segments && (
        <div style={{ marginTop: 10, fontSize: 11, color: 'var(--fg-3)' }}>
          Filed→triaged {fmtUptime(lifespan.triage_segments.filed_to_triaged.p50)} p50 / {fmtUptime(lifespan.triage_segments.filed_to_triaged.p90)} p90
          {' · '}Triaged→resolved {fmtUptime(lifespan.triage_segments.triaged_to_resolved.p50)} p50 / {fmtUptime(lifespan.triage_segments.triaged_to_resolved.p90)} p90
          {' '}(n={lifespan.triage_segments.count})
        </div>
      )}
    </div>
  );
}

// ── Workflow panel ──
//
// 100%-normalized weekly tier-absorption chart (+ a total-volume sparkline),
// an action-mix donut, churn/throughput time charts, and a reserved mount
// seam for ζ's lifecycle-flow diagram (depends on δ).

function WorkflowPanel({ workflow, win, generatedAt, regimeMarkers }) {
  const tierWeekly = sliceWeeklyByWindow(workflow.tier_weekly, generatedAt, win);
  const weeks = Object.keys(tierWeekly).sort();
  const weekTotals = weeks.map(w => Object.values(tierWeekly[w] || {}).reduce((s, n) => s + n, 0));

  // Fold any tier outside the known whitelist into 'other-auto' BEFORE
  // building the per-week stacks below — weekTotals (the normalization
  // denominator, just above) already counts every tier the backend sent,
  // whitelisted or not, so an un-folded unknown tier would have no band of
  // its own and the stack would visibly fall short of the 100% line.
  const foldedWeekly = {};
  for (const w of weeks) {
    const folded = {};
    for (const [t, n] of Object.entries(tierWeekly[w] || {})) {
      const key = resolverTierKey(t);
      folded[key] = (folded[key] || 0) + n;
    }
    foldedWeekly[w] = folded;
  }

  // 100%-normalized: every tier is stacked every week (even at 0) so each
  // week's band heights always sum to exactly 1.0 — StackedAreaChart
  // auto-scales its y-axis to the max stacked total, which is then 1.0.
  const stacks = _RESOLVER_TIERS.map(t => ({
    key: t,
    color: _TIER_COLORS[t] || C.PALETTE.fg3,
    values: weeks.map((w, wi) => (weekTotals[wi] > 0 ? ((foldedWeekly[w] || {})[t] || 0) / weekTotals[wi] : 0)),
  }));

  const actionEntries = Object.entries(workflow.action_mix || {}).sort((a, b) => b[1] - a[1]);
  const donutData = actionEntries.map(([action, count], i) => ({
    label: action,
    value: count,
    color: _CATEGORY_COLORS[i % _CATEGORY_COLORS.length],
  }));
  const totalActions = actionEntries.reduce((s, [, count]) => s + count, 0);

  const churnDaily = sliceDailyByWindow(workflow.churn_daily, generatedAt, win);
  const churnDates = Object.keys(churnDaily).sort();

  const escPerDoneDaily = sliceRowsByWindow(workflow.esc_per_done_daily || [], generatedAt, win, row => row.date);
  // A null ratio means done == 0 that day: no task completed, so escalations
  // per done is undefined rather than zero. It is passed straight through as a
  // hole, and LineChart breaks the line across it (task 3489). These rows used
  // to be FILTERED OUT, which dropped the day from this label row too — that
  // compacted the x-axis and silently redated every surviving sample, the exact
  // hazard spark_path.js's header calls out.
  const epdDates = escPerDoneDaily.map(row => row.date);

  const flowDaily = sliceRowsByWindow(workflow.flow_daily || [], generatedAt, win, row => row.date);

  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ fontSize: 11, color: 'var(--fg-3)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
        Workflow — resolver mix &amp; throughput
      </div>
      {weeks.length > 0 && (
        <div style={{ marginBottom: 10 }}>
          <div style={{ height: 22, marginBottom: 4 }}><C.Sparkline values={weekTotals} /></div>
          <C.StackedAreaChart stacks={stacks} labels={weeks} formatY={v => `${Math.round(v * 100)}%`} />
        </div>
      )}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'center', marginBottom: 10 }}>
        {donutData.length > 0 && (
          <C.Donut data={donutData} centerLabel="actions" centerValue={totalActions} />
        )}
      </div>
      {churnDates.length > 0 && (
        <>
          <div style={{ fontSize: 10, color: 'var(--fg-3)', marginBottom: 4 }}>Churn — same-task re-filings within 24h</div>
          <TimeChart labels={churnDates} markers={regimeMarkers}>
            <C.LineChart
              series={[{ key: 'churn', color: C.PALETTE.bad, values: churnDates.map(d => churnDaily[d] || 0) }]}
              labels={churnDates}
              formatX={fmtDateTime}
            />
          </TimeChart>
        </>
      )}
      {epdDates.length > 0 && (
        <>
          <div style={{ fontSize: 10, color: 'var(--fg-3)', margin: '10px 0 4px' }}>Escalations filed per task done</div>
          <TimeChart labels={epdDates} markers={regimeMarkers}>
            <C.LineChart
              series={[{ key: 'ratio', color: C.PALETTE.accent, values: escPerDoneDaily.map(row => row.ratio) }]}
              labels={epdDates}
              formatX={fmtDateTime}
            />
          </TimeChart>
        </>
      )}
      {/* ζ lifecycle flow diagram / mini-Sankey — origin → level → tier →
          class, fed the windowed `flowDaily` computed above (dep on δ). */}
      <div className="esc-flow-slot">
        <LifecycleFlowDiagram flowDaily={flowDaily} />
      </div>
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
  // Shared across every project's date-axis charts via TimeChart (Origin's
  // filings chart, Workflow's churn/esc-per-done charts) — regime markers
  // are a single cross-project timeline, not per-project data. Lifespan's
  // only chart is the threshold-axis ECDF, so it doesn't consume these.
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
            <OriginPanel
              origin={p.origin}
              win={win}
              generatedAt={analytics.generated_at}
              regimeMarkers={regimeMarkers}
            />
            <LifespanPanel
              lifespan={p.lifespan}
              win={win}
              generatedAt={analytics.generated_at}
            />
            <WorkflowPanel
              workflow={p.workflow}
              win={win}
              generatedAt={analytics.generated_at}
              regimeMarkers={regimeMarkers}
            />
          </ProjectGroup>
        </div>
      ))}
    </div>
  );
}

// ── Additive export (mutates the object created by tabs.jsx) ──
window.DF_TABS.EscalationAnalyticsTab = EscalationAnalyticsTab;
