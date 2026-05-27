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

// Endpoint → DF_DATA keys map, parameterised on the active window chip.
// Only the four windowed endpoints append ?window=; the rest stay static.
function endpointsFor(win) {
  const w = encodeURIComponent(win);
  return {
    '/api/v2/dashboard/orchestrators':                ['ORCHESTRATORS', 'PROJECTS', 'ORCHESTRATORS_SPARK'],
    '/api/v2/dashboard/tasks':                        ['ACTIVE_TASKS', 'FILE_LOCKS', 'TASKS_OFFLINE', 'TASKS_OFFLINE_PROJECTS'],
    '/api/v2/dashboard/memory':                       ['MEMORY_STATUS'],
    '/api/v2/dashboard/memory-graphs':                ['MEMORY_TIMESERIES', 'MEMORY_OPS_BREAKDOWN'],
    '/api/v2/dashboard/recon':                        ['RECON_STATE', 'AGENTS'],
    [`/api/v2/dashboard/merge-queue?window=${w}`]:    ['MERGE_QUEUE'],
    [`/api/v2/dashboard/costs?window=${w}`]:          ['COSTS'],
    [`/api/v2/dashboard/performance?window=${w}`]:    ['PERFORMANCE'],
    [`/api/v2/dashboard/burndown?window=${w}`]:       ['BURNDOWN', 'BURNDOWN_BY_PROJECT'],
    '/api/v2/dashboard/curator':                      ['CURATOR_STATE'],
    '/api/v2/dashboard/scheduler':                    ['SCHEDULER'],
  };
}

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
  TASKS_OFFLINE: false,
  TASKS_OFFLINE_PROJECTS: [],
  PERFORMANCE: {},
  MEMORY_STATUS: {
    graphiti: { connected: false, node_count: 0, edge_count: 0, episode_count: 0 },
    mem0: { connected: false, memory_count: 0 },
    taskmaster: { connected: false },
    queue: { counts: { pending: 0, retry: 0, dead: 0 }, oldest_pending_age_seconds: null },
    projects: {},
    wal: { status: 'offline', reason: null, rows: [] },
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
  // CURATOR_STATE is an object (not a captured top-level array), so it is NOT
  // added to STABLE_ARRAY_KEYS. applyKey replaces the reference on each poll;
  // tab_curator.jsx reads through DF_DATA.CURATOR_STATE per render.
  CURATOR_STATE: {
    pending: [],
    latency_spark: { labels: [], p50: [], p90: [], p99: [] },
    pending_spark: { labels: [], values: [] },
    capped_spark: { labels: [], values: [] },
    state: { capped_now: 0, paused_reason: null, pending_total: 0, accounts_summary: { total: 0, capped: 0, available: 0, capped_accounts: [] } },
  },
  // SCHEDULER is read through DF_DATA.SCHEDULER per render (not captured at
  // module-load), so reference replacement on each poll is safe.
  SCHEDULER: {
    rows: [],
    modules: [],
    pin_queue: [],
    events_by_task: {},
    snapshot_at: null,
    offline: false,
    offline_projects: [],
    paused: false,
    paused_projects: [],
  },
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

// Module-scope window — updated by DF_REFRESH(win); 3 s polling reads from it
// so chip changes take effect on the next tick without restarting the loop.
let currentWin = '24h';

async function refreshDFData(win) {
  if (typeof win === 'string' && win) currentWin = win;
  await Promise.all(Object.entries(endpointsFor(currentWin)).map(([url, keys]) => refreshOne(url, keys)));
  window.dispatchEvent(new CustomEvent('df-data-refresh'));
}

window.DF_REFRESH = refreshDFData;
window.__DF_PAUSE = false;

refreshDFData();
setInterval(() => { if (!window.__DF_PAUSE) refreshDFData(); }, 3000);
