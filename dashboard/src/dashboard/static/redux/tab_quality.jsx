/* Quality tab — one project's per-landing quality series (quality-dashboard ε).
 *
 * Every figure arrives in one on-demand read of /api/v2/dashboard/quality
 * (data.js::ON_DEMAND_KEYS.quality), addressed by the canonical query
 * data.js::qualityQuery builds from the chosen project, landing window and
 * module. It is never polled, so nothing here reads a static DF_DATA.QUALITY.
 * The decisions — layout, headline, rule, alarm and before/after text, chart
 * arrays and mark placement — are quality_view.js's; this file only wires
 * them to charts.jsx and does no arithmetic on a served series.
 *
 * The landing window is the tab's own (window_chip.js::WINDOW_SETS.landings),
 * not the global chip: an on-demand receipt is keyed by its query, so the
 * served echo is read with onDemandReceipt.
 *
 * Render decisions are exercised in a jsdom window by
 * dashboard/tests/js/jsx_render_quality.test.mjs.
 *
 * Load order: charts.jsx, shell.jsx → tab_quality.jsx → app.jsx
 * Export:     window.DF_QUALITY = { QualityTab }
 *
 * Babel compiles every top-level binding here to a global `var`, so a name a
 * classic script declares with const is renamed in its destructure.
 */
const QUALITY_DF = window.DF_DATA;
const { useState: useQualityState } = React;
const {
  LineChart: QualityLineChart,
  MarkedChart: QualityMarkedChart,
  StatTile: QualityStatTile,
  PALETTE: QUALITY_PALETTE,
} = window.DF_CHARTS;
const { Segmented: QualitySegmented, DatumReading: QualityDatumReading } = window.DF_SHELL;
const {
  QUALITY_LAYOUT: QUALITY_TAB_LAYOUT,
  metricTitle,
  headlineText,
  ruleCaption,
  chartSeries,
  chartAxis,
  indexMarkers,
  beforeAfterText,
  costModelReadings,
  qualityMarks,
} = window.DF_QUALITY_VIEW;
const {
  createOnDemandHook,
  qualityQuery,
  ON_DEMAND_KEYS: QUALITY_ON_DEMAND_KEYS,
  ON_DEMAND_VIEWS: QUALITY_ON_DEMAND_VIEWS,
  datumFor,
  onDemandReceipt,
} = window.DF_DATA_LOADER;
const {
  WINDOW_SETS: QUALITY_WINDOW_SETS,
  LANDING_DEFAULT_WINDOW: QUALITY_DEFAULT_WINDOW,
  receiptWindow,
  windowLabel,
  pendingWindow,
} = window.DF_WINDOW_CHIP;
const { datumView, EM_DASH: QUALITY_EM_DASH } = window.DF_DATUM;
const { createPersistedHooks } = window.DF_PERSISTED_STATE;
const { usePersistedState: useQualityPersisted } = createPersistedHooks(React);
const { useOnDemand: useQualityOnDemand } = createOnDemandHook(React);

const QUALITY_NO_SRC = '(no src)';
const QUALITY_NOTE_STYLE = Object.freeze({ fontSize: 11, color: 'var(--fg-3)' });
const QUALITY_HINT_STYLE = Object.freeze({ color: 'var(--fg-3)', textTransform: 'none', letterSpacing: 0, fontSize: 10 });
const QUALITY_SERIES_COLORS = Object.freeze({
  center: QUALITY_PALETTE.accent,
  ewma: QUALITY_PALETTE.fg2,
  ucl: QUALITY_PALETTE.bad,
});

function qualityFieldDatum(query, field) {
  return datumFor(QUALITY_ON_DEMAND_KEYS.quality.key(query, field));
}

function QualityHeader({ projects, project, onProject, landings, onLandings, query, moduleName, onClearModule }) {
  const echo = receiptWindow(onDemandReceipt('quality', query));
  const pending = pendingWindow(landings, echo, QUALITY_WINDOW_SETS.landings);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
      <select aria-label="project" value={project} onChange={e => onProject(e.target.value)}>
        {projects.map(name => <option key={name} value={name}>{name}</option>)}
      </select>
      <QualitySegmented options={QUALITY_WINDOW_SETS.landings} value={landings} onChange={onLandings} />
      <span style={QUALITY_NOTE_STYLE} title="the landing window the server served">
        landings: <span data-quality-window="">{windowLabel(echo)}</span>
      </span>
      {pending && <span style={QUALITY_NOTE_STYLE} title="chosen, not yet served">{pending} pending</span>}
      {moduleName !== null && (
        <span style={QUALITY_NOTE_STYLE}>
          module {moduleName} <button className="multi" onClick={onClearModule}>clear</button>
        </span>
      )}
    </div>
  );
}

function QualityHeadlines({ metrics }) {
  const entries = [...QUALITY_TAB_LAYOUT.outcomes, ...QUALITY_TAB_LAYOUT.changeCost].filter(e => metrics[e.metric]);
  return (
    <div data-quality-section="headlines" className="grid cols-5" style={{ gap: 8, marginBottom: 12 }}>
      {entries.map(entry => {
        const metric = metrics[entry.metric];
        return (
          <QualityStatTile
            key={entry.metric}
            label={entry.title}
            datum={metric.headline}
            format={value => headlineText(value, metric.unit)}
            history={metric.rolling.center}
            title={ruleCaption(metric.rule)}
          />
        );
      })}
    </div>
  );
}

function QualityBeside({ metric }) {
  return (
    <div style={QUALITY_NOTE_STYLE} title={ruleCaption(metric.rule)}>
      {metric.rule.counts}: <QualityDatumReading datum={metric.headline} format={value => headlineText(value, metric.unit)} />
    </div>
  );
}

function QualityMetricPanel({ entry, quality, markers }) {
  const metric = quality.metrics[entry.metric];
  if (!metric) {
    return <div className="panel" data-quality-metric={entry.metric} style={QUALITY_NOTE_STYLE}>{entry.title}: not served</div>;
  }
  const { series, band } = chartSeries(metric);
  const { formatY, minAxisMax } = chartAxis(metric);
  const marks = qualityMarks({
    axis: quality.axis, markers, alarms: metric.alarms, beforeAfter: quality.before_after,
    metricId: entry.metric, kind: metric.kind,
  });
  const beside = entry.beside === null ? null : quality.metrics[entry.beside];
  return (
    <div className="panel" data-quality-metric={entry.metric}>
      <div className="panel-head"><span className="title">{entry.title}</span></div>
      <div className="panel-body">
        <div style={{ ...QUALITY_NOTE_STYLE, fontSize: 10, marginBottom: 14 }}>{ruleCaption(metric.rule)}</div>
        <QualityMarkedChart marks={marks} legend>
          <QualityLineChart
            series={series.map(s => ({ ...s, color: QUALITY_SERIES_COLORS[s.key], fill: false }))}
            band={band}
            labels={quality.axis.map(point => `#${point.ordinal}`)}
            formatY={formatY}
            minAxisMax={minAxisMax}
            height={180}
          />
        </QualityMarkedChart>
        {beside && <QualityBeside metric={beside} />}
      </div>
    </div>
  );
}

function QualityMetricSection({ name, title, entries, quality, markers, children }) {
  return (
    <div data-quality-section={name} style={{ marginBottom: 12 }}>
      <div className="panel-head"><span className="title">{title}</span></div>
      {children}
      <div className="grid cols-2" style={{ gap: 12 }}>
        {entries.map(entry => <QualityMetricPanel key={entry.metric} entry={entry} quality={quality} markers={markers} />)}
      </div>
    </div>
  );
}

function QualityCostModel({ costModel }) {
  return (
    <div data-quality-section="cost-model" className="grid cols-2" style={{ gap: 8, marginBottom: 12 }}>
      {costModelReadings(costModel).map(reading => (
        <div key={reading.key} className="kpi">
          <div className="lbl">{reading.label}{reading.hint && <span style={QUALITY_HINT_STYLE}> · {reading.hint}</span>}</div>
          <div className="val" title={reading.text === null ? 'no cost model is in force yet' : undefined}>
            {reading.text ?? QUALITY_EM_DASH}
          </div>
        </div>
      ))}
    </div>
  );
}

function QualityBeforeAfter({ quality, markers }) {
  const byId = indexMarkers(markers);
  const entries = quality.before_after.filter(entry => quality.metrics[entry.metric]);
  const markerIds = [...new Set(entries.map(entry => entry.marker_id))];
  return (
    <div data-quality-section="before-after" className="panel" style={{ marginTop: 12 }}>
      <div className="panel-head"><span className="title">Before / after each marker</span></div>
      <div className="panel-body">
        {markerIds.length === 0 && <div style={QUALITY_NOTE_STYLE}>No marker has a before/after comparison.</div>}
        {markerIds.map(id => (
          <div key={id} style={{ marginBottom: 8 }}>
            <div style={{ fontSize: 12 }}>{byId.has(id) ? byId.get(id).label : id}</div>
            {entries.filter(entry => entry.marker_id === id).map(entry => (
              <div key={entry.metric} style={QUALITY_NOTE_STYLE}>
                {metricTitle(entry.metric, quality.metrics[entry.metric])}: <span className="mono">{beforeAfterText(entry, quality.metrics[entry.metric].kind)}</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function QualityModules({ rows, onModule }) {
  return (
    <div data-quality-section="modules" className="panel" style={{ marginBottom: 12 }}>
      <div className="panel-head"><span className="title">Modules</span></div>
      <table className="tbl">
        <thead>
          <tr><th>module</th><th>landings</th><th>tasks</th><th>lines</th><th>cost</th><th>$ / 100 lines</th></tr>
        </thead>
        <tbody>
          {rows.map(row => {
            const drillable = row.module !== null;
            return (
              <tr
                key={row.module ?? QUALITY_NO_SRC}
                onClick={drillable ? () => onModule(row.module) : undefined}
                title={row.ranked ? undefined : 'unranked: too few lines or tasks, or no src module'}
                style={{ cursor: drillable ? 'pointer' : 'default', opacity: row.ranked ? 1 : 0.6 }}
              >
                <td>{row.module ?? QUALITY_NO_SRC}</td>
                <td className="mono">{row.landings}</td>
                <td className="mono">{row.tasks}</td>
                <td className="mono">{row.allocated_lines.toFixed(0)}</td>
                <td className="mono">${row.allocated_usd.toFixed(2)}</td>
                <td className="mono">{row.usd_per_100_lines === null ? QUALITY_EM_DASH : `$${row.usd_per_100_lines.toFixed(2)}`}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function QualitySnapshots({ datum }) {
  const reading = datumView(datum);
  return (
    <div data-quality-section="snapshots" className="panel">
      <div className="panel-head"><span className="title">Snapshots</span></div>
      <div className="panel-body" style={QUALITY_NOTE_STYLE}>
        {reading.isHole ? reading.title : 'The snapshot series is served; its view arrives with quality-dashboard leaf ι.'}
      </div>
    </div>
  );
}

function QualitySeries({ query, quality, onModule }) {
  const markersDatum = qualityFieldDatum(query, 'QUALITY_MARKERS');
  const markersReading = datumView(markersDatum);
  const markers = markersReading.isHole ? [] : markersDatum.value.markers;
  return (
    <>
      {markersReading.isHole && <div style={QUALITY_NOTE_STYLE}>Markers unavailable: {markersReading.title}</div>}
      <QualityHeadlines metrics={quality.metrics} />
      <QualityMetricSection name="outcomes" title="Outcomes" entries={QUALITY_TAB_LAYOUT.outcomes} quality={quality} markers={markers} />
      <QualityMetricSection name="change-cost" title="Change cost" entries={QUALITY_TAB_LAYOUT.changeCost} quality={quality} markers={markers}>
        <QualityCostModel costModel={quality.cost_model} />
      </QualityMetricSection>
      <QualityBeforeAfter quality={quality} markers={markers} />
      <QualityModules rows={quality.modules} onModule={onModule} />
      <QualitySnapshots datum={qualityFieldDatum(query, 'QUALITY_SNAPSHOTS')} />
    </>
  );
}

function QualityRead({ view, value, query, onModule }) {
  if (view === QUALITY_ON_DEMAND_VIEWS.loading) return <div style={QUALITY_NOTE_STYLE}>loading…</div>;
  if (view !== QUALITY_ON_DEMAND_VIEWS.ready) {
    return <div style={{ ...QUALITY_NOTE_STYLE, color: 'var(--warn)' }}>The quality series could not be fetched.</div>;
  }
  const reading = datumView(value);
  if (reading.isHole) return <div style={QUALITY_NOTE_STYLE}>{reading.title}</div>;
  return <QualitySeries query={query} quality={value.value} onModule={onModule} />;
}

// The read's own outcome re-renders the header too, so the served-window echo
// it shows is the receipt of the request that just settled.
function QualityProject({ projects, project, onProject, landings, onLandings, moduleName, onModule }) {
  const query = qualityQuery({ project, window: landings, module: moduleName });
  const { view, value } = useQualityOnDemand('quality', query);
  return (
    <div data-quality-tab="">
      <QualityHeader
        projects={projects}
        project={project}
        onProject={onProject}
        landings={landings}
        onLandings={onLandings}
        query={query}
        moduleName={moduleName}
        onClearModule={() => onModule(null)}
      />
      <QualityRead view={view} value={value} query={query} onModule={onModule} />
    </div>
  );
}

function QualityTab() {
  const projects = QUALITY_DF.PROJECTS.map(row => row.name);
  const [chosenProject, setChosenProject] = useQualityPersisted('df.quality.project', null);
  const [landings, setLandings] = useQualityPersisted('df.quality.window', QUALITY_DEFAULT_WINDOW);
  const [moduleName, setModuleName] = useQualityState(null);
  if (projects.length === 0) {
    return <div data-quality-tab="" style={QUALITY_NOTE_STYLE}>No project is configured.</div>;
  }
  const chooseProject = next => {
    setChosenProject(next);
    setModuleName(null);
  };
  return (
    <QualityProject
      projects={projects}
      project={projects.includes(chosenProject) ? chosenProject : projects[0]}
      onProject={chooseProject}
      landings={QUALITY_WINDOW_SETS.landings.includes(landings) ? landings : QUALITY_DEFAULT_WINDOW}
      onLandings={setLandings}
      moduleName={moduleName}
      onModule={setModuleName}
    />
  );
}

window.DF_QUALITY = { QualityTab };
