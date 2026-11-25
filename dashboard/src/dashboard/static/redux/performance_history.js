// performance_history.js — the CLIENT twin of the history half of
// dashboard/src/dashboard/data/performance.py. Each PERFORMANCE[label] entry
// serves its series on one history_labels axis, aligned as
// performance.py::PerformanceHistory guarantees and re-checked here. PerfTab
// reads one project's trend through projectHistory, and averages the in-scope
// projects hour by hour for its header sparks through meanHistorySpark.
//
// It has no module-scope dependency. index.html loads it before tabs.jsx,
// which destructures window.DF_PERFORMANCE_HISTORY at module scope; node
// requires the same file as CommonJS (performance_history.test.mjs).

// Each series: the served block and key it is read from, and the decimals its
// header spark rounds to.
const PERF_HISTORY_SERIES = {
  p50: { block: 'time_centiles_history', key: 'p50', decimals: 0 },
  p95: { block: 'time_centiles_history', key: 'p95', decimals: 0 },
  one_pass: { block: 'one_pass_history', key: 'values', decimals: 1 },
  escalation: { block: 'escalation_history', key: 'values', decimals: 1 },
};

function projectHistory(perf, project) {
  const entry = perf[project];
  if (entry == null) {
    return Object.fromEntries(['labels', ...Object.keys(PERF_HISTORY_SERIES)].map(name => [name, []]));
  }
  const labels = entry.history_labels;
  if (!Array.isArray(labels)) {
    throw new Error(`PERFORMANCE[${project}] serves no history_labels axis`);
  }
  const history = { labels };
  for (const [name, { block, key }] of Object.entries(PERF_HISTORY_SERIES)) {
    const values = entry[block]?.[key];
    const length = Array.isArray(values) ? values.length : 'no';
    if (length !== labels.length) {
      throw new Error(`PERFORMANCE[${project}] series ${name} has ${length} values for ${labels.length} history_labels`);
    }
    history[name] = values;
  }
  return history;
}

function meanHistorySpark(perf, series) {
  const byHour = new Map();
  for (const project of Object.keys(perf)) {
    const history = projectHistory(perf, project);
    history.labels.forEach((label, i) => {
      const hour = byHour.get(label) || { sum: 0, n: 0 };
      hour.sum += history[series][i];
      hour.n += 1;
      byHour.set(label, hour);
    });
  }
  const scale = 10 ** PERF_HISTORY_SERIES[series].decimals;
  return [...byHour.keys()].sort().map(label => {
    const { sum, n } = byHour.get(label);
    return Math.round((sum / n) * scale) / scale;
  });
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced at runtime by classic_script_scope.test.mjs.
const PERFORMANCE_HISTORY_API = { projectHistory, meanHistorySpark };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = PERFORMANCE_HISTORY_API;
}
if (typeof window !== 'undefined') {
  window.DF_PERFORMANCE_HISTORY = PERFORMANCE_HISTORY_API;
}
