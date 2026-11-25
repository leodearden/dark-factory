// OverviewTab's Host load card render seam (task 6987): a row for every key
// /api/load serves. The real .jsx renders in a jsdom window (dom_harness.mjs);
// every expected value comes from calling host_load_card.js::hostLoadRows,
// never from source text.
//
// A separate file from jsx_render.test.mjs, which is past the 1,500-line
// ceiling; each file pays the Babel compile once, in its own process.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { allowAsyncAct, openWindow, settle, stubFetch } from './_render_helpers.mjs';

function served(current, sparkline) {
  return { current, sparkline, window_mean: null, window_max: null };
}

function loadResponse(body) {
  return { ok: true, status: 200, json: async () => body };
}

async function mountOverview(t, answers) {
  const harness = openWindow(t);
  const { window } = harness;
  allowAsyncAct(window);
  stubFetch(window, answers);
  const { container } = harness.mount(window.DF_OVERVIEW.OverviewTab, { paused: false });
  await settle(window);
  return { window, container };
}

function hostLoadPanel(container) {
  const panel = [...container.querySelectorAll('.panel')]
    .find(candidate => candidate.querySelector('.panel-head .title')?.textContent === 'Host load');
  assert.ok(panel, 'OverviewTab renders a panel titled "Host load"');
  return panel;
}

function renderedRows(panel) {
  return [...panel.querySelectorAll('tbody tr')]
    .map(row => [...row.querySelectorAll('td')].slice(0, 2).map(cell => cell.textContent));
}

test("tab_overview.jsx::HostLoadCard renders a row for every key /api/load serves, in hostLoadRows's words", async t => {
  const body = {
    psi_cpu_some_avg10: served(12.5, [1, 2]),
    fixture_unlabelled_metric: served(2.5, [2.5]),
    verify_scopes_live: served(null, []),
  };
  const { window, container } = await mountOverview(t, { '/api/load': loadResponse(body) });

  const expected = window.DF_HOST_LOAD_CARD.hostLoadRows(body);
  assert.equal(
    expected.find(row => row.key === 'fixture_unlabelled_metric').label,
    'fixture_unlabelled_metric',
    'precondition: the fixture exercises the fallback for an unlabelled key',
  );
  assert.deepEqual(renderedRows(hostLoadPanel(container)), expected.map(row => [row.label, row.text]));
});

test('tab_overview.jsx::HostLoadCard renders no metric row until /api/load answers, and says so', async t => {
  const { container } = await mountOverview(t, {});

  const panel = hostLoadPanel(container);
  assert.deepEqual(renderedRows(panel), []);
  assert.ok(panel.textContent.includes('no load data'), panel.textContent);
});
