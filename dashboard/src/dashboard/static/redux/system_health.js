// system_health.js — the System health panel's status decisions: each row's
// tone from a served field, and the header those rows imply. tab_overview.jsx
// spreads these rows into its panel and decides none of them itself.
//
// A PLAIN-JS CLASSIC SCRIPT, NOT A .jsx MODULE, so its decisions are
// EXECUTABLE: dashboard/tests/js/system_health.test.mjs runs them.
// pins_recovery.js's header holds the CANONICAL statement of why.
//
// LOAD CONTRACT. It destructures window.DF_DATUM and
// window.DF_TASKS_OFFLINE_BANNER at module scope with no fallback, so
// index.html loads it after datum.js and tasks_offline_banner.js and before
// tab_overview.jsx (test_index_html.py pins the order). Receipts come from
// `data.__receipt` only: no function here reads a browser global.

// Module scope, no fallback, RENAMED — see the CANONICAL note in datum.js's
// header.
const {
  plainDatum: plainHealthDatum,
  EM_DASH: HEALTH_EM_DASH,
} = window.DF_DATUM;
const { tasksBannerNoticesFor: taskStoreNotices } = window.DF_TASKS_OFFLINE_BANNER;

const HEALTH_MEMORY_ENDPOINT = '/api/v2/dashboard/memory';
const HEALTH_TASKS_ENDPOINT = '/api/v2/dashboard/tasks';

// Red first: the header names the worst news first.
const HEALTH_FAULT_TONES = Object.freeze(['bad', 'warn']);

// "Has this endpoint delivered?" is ASKED of datum.js, as derivedDatum asks
// it: a probe value plainDatum cannot call absent comes back a hole only when
// there is no receipt, and that hole's reason is the row's title.
function undeliveredHole(data, endpointKey) {
  const probe = plainHealthDatum(0, endpointKey, data.__receipt || {});
  return probe.state === 'unknown' ? probe : null;
}

// Unmeasured is amber: never green, since nothing was measured, and never red,
// since nothing was measured to be wrong — queueHealth's rule.
function unmeasuredTone(title) {
  return { ok: true, warn: true, title };
}

function unmeasuredRow(title) {
  return { sub: HEALTH_EM_DASH, ...unmeasuredTone(title) };
}

function healthMemoryStatus(data) {
  return data.MEMORY_STATUS || {};
}

// ── Graphiti / Mem0: connectivity as fused-memory's get_status measured it ──

function connectionHealth(data, storeKey) {
  const hole = undeliveredHole(data, HEALTH_MEMORY_ENDPOINT);
  if (hole) return unmeasuredRow(hole.reason);

  const status = healthMemoryStatus(data);
  if (status.offline) {
    return unmeasuredRow(status.error ? 'fused-memory unreachable: ' + status.error : 'fused-memory unreachable');
  }

  const store = status[storeKey] || {};
  if (store.connected === true) return { sub: 'connected', ok: true, warn: false };
  if (store.connected === false) {
    return { sub: 'not connected', ok: false, warn: false, title: store.error || 'not connected; no error reported' };
  }
  return unmeasuredRow('fused-memory did not report ' + storeKey + ' connectivity');
}

function graphitiHealth(data) {
  return connectionHealth(data, 'graphiti');
}

function mem0Health(data) {
  return connectionHealth(data, 'mem0');
}

// ── The task store, through the notices the Tasks-tab banner shows ──

function projectsAnswering(count) {
  return count + (count === 1 ? ' project' : ' projects') + ' answering';
}

function taskStoreHealth(data) {
  const hole = undeliveredHole(data, HEALTH_TASKS_ENDPOINT);
  if (hole) return unmeasuredRow(hole.reason);

  const notices = taskStoreNotices(data);
  const outage = notices.find(notice => notice.kind === 'global');
  if (outage) return { sub: 'unreachable', ok: false, warn: false, title: outage.text };
  if (notices.length > 0) {
    return {
      sub: notices.map(notice => notice.kind).join(' · '),
      ok: true,
      warn: true,
      title: notices.map(notice => notice.text).join('\n'),
    };
  }

  const count = data.TASKS_PROJECT_COUNT || 0;
  if (count === 0) {
    return { sub: 'no task projects', ok: true, warn: true, title: 'the /tasks fan-out reached no task project' };
  }
  return { sub: projectsAnswering(count), ok: true, warn: false };
}

// ── fused-memory itself: tone and title only ──
// The row's uptime sub needs DF_SHELL.fmtUptime, so the JSX keeps it; with no
// `sub` key here, spreading this after the JSX's own sub never overwrites it.

function fusedMemoryHealth(data) {
  const hole = undeliveredHole(data, HEALTH_MEMORY_ENDPOINT);
  if (hole) return unmeasuredTone(hole.reason);

  const status = healthMemoryStatus(data);
  if (status.offline) return { ok: false, warn: false, title: status.error || 'fused-memory unreachable' };
  return { ok: true, warn: false, title: status.started_at };
}

// ── The one row -> tone mapping, read by the dot, the badge and the header ──

function healthTone(row) {
  if (!row.ok) return 'bad';
  return row.warn ? 'warn' : 'ok';
}

function healthSummary(rows) {
  if (rows.length === 0) return HEALTH_EM_DASH;
  const tones = rows.map(healthTone);
  const faults = HEALTH_FAULT_TONES
    .map(tone => [tone, tones.filter(t => t === tone).length])
    .filter(([, count]) => count > 0)
    .map(([tone, count]) => count + ' ' + tone);
  return faults.length === 0 ? 'all ok' : faults.join(' · ');
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced at runtime by classic_script_scope.test.mjs.
const SYSTEM_HEALTH_API = {
  graphitiHealth,
  mem0Health,
  taskStoreHealth,
  fusedMemoryHealth,
  healthTone,
  healthSummary,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = SYSTEM_HEALTH_API;
}
if (typeof window !== 'undefined') {
  window.DF_SYSTEM_HEALTH = SYSTEM_HEALTH_API;
}
