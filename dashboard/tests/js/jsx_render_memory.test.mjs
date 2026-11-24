// MemoryTab's JSX render seams (task 9944): the plan route's line under the
// stat tiles. The real .jsx renders in a jsdom window (dom_harness.mjs); every
// expected value comes from calling the classic-script helpers
// (memory_readings.js, system_health.js), never from source text.
//
// A separate file from jsx_render.test.mjs, which is past the 1,500-line
// ceiling; each file pays the Babel compile once, in its own process.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { openWindow } from './_render_helpers.mjs';
import { seedDashboardData } from './dom_harness.mjs';
import { receiptsFor, wireDatum } from './wire_fixtures.mjs';

const PLAN_ROUTE_TESTID = 'memory-plan-route';

const CAPPED_ROUTE = Object.freeze({
  state: 'capped',
  proxy_host: 'plan-proxy.internal',
  capped_since: '2026-10-14T09:40:00+00:00',
  resets_at: '2026-10-14T14:00:00+00:00',
  next_probe_at: '2026-10-14T10:10:00+00:00',
  parked_items: 3,
  judge_fail_opens_since_latch: 2,
  escalation_id: 'esc-plan-route-capped-1',
});
const OK_ROUTE = Object.freeze({
  ...CAPPED_ROUTE,
  state: 'ok',
  capped_since: null,
  resets_at: null,
  next_probe_at: null,
  parked_items: 0,
  judge_fail_opens_since_latch: 0,
  escalation_id: null,
});

const EMPTY_SERIES = Object.freeze({ labels: [], values: [] });

// MEMORY_STATUS as redux_api.py::shape_memory serves it, holding only what
// MemoryTab reads.
function memoryStatus(planRoute) {
  return {
    graphiti: { connected: true, node_count: 12, edge_count: 30, episode_count: 4, spark: EMPTY_SERIES },
    mem0: { connected: true, memory_count: 7, spark: EMPTY_SERIES },
    queue: {
      stats: wireDatum({ pending: 0, retry: 0, dead: 0, oldest_pending_age_seconds: null }),
      spark: EMPTY_SERIES,
    },
    plan_route: planRoute,
    projects: {},
    uptime_seconds: 60,
    started_at: '2026-10-14T09:59:00+00:00',
  };
}

// MEMORY_OPS as redux_api.py::shape_memory_graphs serves it for an idle window.
const MEMORY_OPS = Object.freeze({
  labels: [], reads: [], writes: [], other: [], total: [], by_operation: [],
  totals: wireDatum({ reads: 0, writes: 0, other: 0, total: 0 }),
  newest_hour_total: wireDatum(0),
});

function mountMemoryTab(harness, planRoute) {
  const { window } = harness;
  seedDashboardData(window, {
    MEMORY_STATUS: memoryStatus(planRoute),
    MEMORY_OPS,
    __receipt: receiptsFor(window.DF_DATA_LOADER, ['MEMORY_STATUS', 'MEMORY_OPS']),
  });
  const { container } = harness.mount(window.DF_TABS.MemoryTab, { projectFilter: [], onNavigate: () => {} });
  const line = container.querySelector(`[data-testid="${PLAN_ROUTE_TESTID}"]`);
  assert.ok(line, `MemoryTab renders a [data-testid="${PLAN_ROUTE_TESTID}"] line`);
  return line;
}

// The reading and tone the shared helpers give the seeded payload.
function expectedLine(window) {
  const { planRoute, planRouteText, planRouteHealth } = window.DF_MEMORY_READINGS;
  const route = planRoute(window.DF_DATA);
  const tone = window.DF_SYSTEM_HEALTH.healthTone(planRouteHealth(route));
  return { route, tone, text: window.DF_DATUM.datumView(route, { format: planRouteText }).text };
}

for (const [name, value, tone] of [['a capped', CAPPED_ROUTE, 'bad'], ['an ok', OK_ROUTE, 'ok']]) {
  test(`tabs.jsx::MemoryTab reads ${name} plan route through planRouteText, coloured by planRouteHealth`, t => {
    const harness = openWindow(t);
    const { window } = harness;
    const line = mountMemoryTab(harness, wireDatum(value));

    const expected = expectedLine(window);
    assert.equal(expected.route.state, 'fresh', 'the seeded route reaches the tab as a fresh Datum');
    assert.equal(line.textContent, window.DF_MEMORY_READINGS.planRouteText(expected.route.value));
    assert.equal(line.textContent, expected.text);
    assert.equal(expected.tone, tone);
    assert.equal(line.style.color, window.DF_CHARTS.PALETTE[tone]);
  });
}

test('tabs.jsx::MemoryTab renders an unread plan route as a hole titled by its reason, in amber', t => {
  const harness = openWindow(t);
  const { window } = harness;
  const reason = 'get_status reports no plan_route: fused-memory predates task 9944';
  const line = mountMemoryTab(
    harness,
    wireDatum(null, { as_of: null, state: 'unknown', reason, freshness_bound_seconds: 30 }),
  );

  const expected = expectedLine(window);
  assert.equal(line.textContent, window.DF_DATUM.EM_DASH);
  assert.equal(line.textContent, expected.text);
  assert.equal(line.querySelector('span').getAttribute('title'), reason);
  assert.equal(expected.tone, 'warn');
  assert.equal(line.style.color, window.DF_CHARTS.PALETTE.warn);
});
