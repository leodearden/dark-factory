/* Real-data loader for Dark Factory dashboard.
 *
 * Replaces the mockup's synthetic fixtures.  Polls the server-side JSON API
 * exposed by dashboard.app under /api/v2/dashboard/* and merges results into
 * window.DF_DATA.
 *
 * Reference-stability matters: shell.jsx captures DF_DATA.PROJECTS and
 * DF_DATA.AGENTS at module-load (`const SHELL_PROJECTS = window.DF_DATA.PROJECTS`),
 * so we MUTATE those arrays in place rather than replacing the references.
 * Other DF_DATA.* values are read through the DF_DATA object reference each
 * render and can be replaced freely.
 */

const ENDPOINTS = {
  '/api/v2/dashboard/orchestrators': ['ORCHESTRATORS', 'PROJECTS', 'ORCHESTRATORS_SPARK'],
  '/api/v2/dashboard/tasks':         ['ACTIVE_TASKS', 'FILE_LOCKS'],
  '/api/v2/dashboard/memory':        ['MEMORY_STATUS'],
  '/api/v2/dashboard/memory-graphs': ['MEMORY_TIMESERIES', 'MEMORY_OPS_BREAKDOWN'],
  '/api/v2/dashboard/recon':         ['RECON_STATE', 'AGENTS'],
  '/api/v2/dashboard/merge-queue':   ['MERGE_QUEUE'],
  '/api/v2/dashboard/costs':         ['COSTS'],
  '/api/v2/dashboard/performance':   ['PERFORMANCE'],
  '/api/v2/dashboard/burndown':      ['BURNDOWN', 'BURNDOWN_BY_PROJECT'],
};

// Keys whose array reference is captured at module-load by shell.jsx — mutate
// in place rather than reassigning, so cached references stay valid.
const STABLE_ARRAY_KEYS = new Set(['PROJECTS', 'AGENTS']);

// ── Empty defaults — all keys initialised so the first render before fetch
//    completes does not crash any component reading DF_DATA.* ──
window.DF_DATA = {
  PROJECTS: [],
  AGENTS: [],
  ORCHESTRATORS: [],
  ORCHESTRATORS_SPARK: { labels: [], values: [] },
  ACTIVE_TASKS: [],
  FILE_LOCKS: {},
  PERFORMANCE: {},
  MEMORY_STATUS: {
    graphiti: { connected: false, node_count: 0, edge_count: 0, episode_count: 0 },
    mem0: { connected: false, memory_count: 0 },
    taskmaster: { connected: false },
    queue: { counts: { pending: 0, retry: 0, dead: 0 }, oldest_pending_age_seconds: null },
    projects: {},
  },
  MEMORY_TIMESERIES: { labels: [], reads: [], writes: [] },
  MEMORY_OPS_BREAKDOWN: [],
  RECON_STATE: {
    buffer: { buffered_count: 0, oldest_event_age_seconds: null },
    burst_state: [],
    watermarks: {},
    verdict: null,
    runs: [],
  },
  MERGE_QUEUE: {},
  COSTS: {
    summary: { total: 0, runs: 0, today: 0, tokens: null, p95_run_cost: null, delta_pct: null, delta_hint: null },
    by_project: [],
    by_account: [],
    by_role: [],
    trend: { labels: [], values: [] },
    events: [],
  },
  BURNDOWN: { labels: [], done: [], in_progress: [], blocked: [], pending: [], forecast_low: null, forecast_high: null },
  BURNDOWN_BY_PROJECT: {},
};

function applyKey(key, value) {
  if (value === undefined || value === null) return;
  if (STABLE_ARRAY_KEYS.has(key) && Array.isArray(window.DF_DATA[key]) && Array.isArray(value)) {
    window.DF_DATA[key].length = 0;
    window.DF_DATA[key].push(...value);
  } else {
    window.DF_DATA[key] = value;
  }
}

async function refreshOne(url, keys) {
  try {
    const resp = await fetch(url, { credentials: 'same-origin' });
    if (!resp.ok) return;
    const body = await resp.json();
    keys.forEach(k => applyKey(k, body[k]));
  } catch (err) {
    // Network blip — keep the prior values so the UI does not blank out.
    console.warn('DF_DATA fetch failed', url, err);
  }
}

async function refreshDFData() {
  await Promise.all(Object.entries(ENDPOINTS).map(([url, keys]) => refreshOne(url, keys)));
  window.dispatchEvent(new CustomEvent('df-data-refresh'));
}

window.DF_REFRESH = refreshDFData;
window.__DF_PAUSE = false;

refreshDFData();
setInterval(() => { if (!window.__DF_PAUSE) refreshDFData(); }, 3000);
