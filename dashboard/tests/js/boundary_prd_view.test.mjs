// Boundary rows 20–21 of plans/dashboard-prd-lanes-prd.md: the PRD lanes page.
//
// Every body here is one the REAL /prds and /tasks routes served
// (test_boundary_js.py builds them through _boundary_payloads.prd_lanes_family
// and prd_view_family). The landing test lands /prds through data.js's real
// refreshOne; the render tests seed the same bodies into a dom_harness window
// and mount the real components, taking every expected text from
// prd_lanes_view.js's helpers and the served values.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { applyServed, loadClient, payload } from './_boundary_client.mjs';

const PRDS = '/api/v2/dashboard/prds';
const PROJECT = 'dark-factory';

test('the served /prds body lands through refreshOne and reads back per project, stamped', async () => {
  const client = loadClient();
  const body = payload('prd_lanes');
  const receivedAt = Date.parse(body.served_at);
  assert.equal(await applyServed(client, PRDS, body, { receivedAt }), 'applied');

  const win = client.window;
  const { projectPrdView, projectPrdContention } = win.DF_PRD_LANES_VIEW;
  const readings = [
    [projectPrdView(win.DF_DATA, PROJECT), body.PRD_VIEW[PROJECT]],
    [projectPrdContention(win.DF_DATA, PROJECT), body.PRD_CONTENTION[PROJECT]],
  ];
  for (const [read, served] of readings) {
    assert.deepEqual(read.value, served.value);
    assert.equal(read.state, served.state);
    assert.equal(read.as_of, served.as_of);
    assert.equal(read.reason, served.reason);
    assert.equal(read._received_at, receivedAt, 'the reader must stamp the /prds receipt');
  }

  const { TAB_ENDPOINTS } = win.DF_POLL_SCOPE;
  assert.ok(TAB_ENDPOINTS.tasks.includes(PRDS), 'the Tasks tab must poll /prds');
  const others = Object.entries(TAB_ENDPOINTS).filter(([tab, paths]) => tab !== 'tasks' && paths.includes(PRDS));
  assert.deepEqual(others.map(([tab]) => tab), [], 'no other tab polls /prds (§4.4)');
});
