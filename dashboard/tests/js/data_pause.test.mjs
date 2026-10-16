// The __DF_PAUSE contract for data.js: while window.__DF_PAUSE is true, NO
// poll-set fetch runs — not the 3s timer, not a window-chip change, not a tab
// switch. What the user changed while paused is recorded and takes effect on
// the first tick after resume. requestOnDemand alone still fetches.
//
// The measured defect: six window-chip clicks under `__DF_PAUSE = true`
// produced two extra full 14-endpoint refreshes, because only pollTick
// consulted the flag and a chip change reaches the same fetch loop through
// refreshDFData directly.
//
// Run via `node --test`; dashboard/tests/test_graph_layout_js.py globs this
// directory's *.test.mjs into the pytest run.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { loadDataJs, drain } from './_data_loader.mjs';

const { pollKey } = loadDataJs().api;

// Any endpoint the unscoped poll set fetches without a ?window= parameter.
const UNWINDOWED_ENDPOINT_PATH = '/api/v2/dashboard/curator';

// Two tabs with TAB_ENDPOINTS entries, where the second needs an endpoint the
// first's poll set does not fetch, so switching between them has something
// newly needed to fetch at once.
const FIRST_TAB = 'scheduler';
const SWITCHED_TAB = 'curator';

const ON_DEMAND_PROJECT = 'dark-factory';

// A fresh data.js (currentWin '24h', currentTab null) whose every fetch is
// recorded, with isolated flow-control state and no jitter.
function pauseLoad() {
  const requested = [];
  const fetchImpl = url => {
    requested.push(url);
    return Promise.resolve({ ok: true, json: async () => ({}) });
  };
  const loaded = loadDataJs({ fetchStub: fetchImpl });
  const opts = { state: loaded.api.createPollState(), deps: { fetchImpl }, jitterMaxMs: 0 };
  const refreshEvents = () => loaded.events.filter(ev => ev.type === 'df-data-refresh').length;
  return { ...loaded, opts, requested, refreshEvents, clear: () => { requested.length = 0; } };
}

function uniqueSorted(urls) {
  return [...new Set(urls)].sort();
}

test('preserved behaviour: window.__DF_PAUSE = true stops pollTick from fetching; false resumes on the next tick', async () => {
  const callCount = new Map();
  const fetchImpl = url => {
    const path = pollKey(url);
    callCount.set(path, (callCount.get(path) || 0) + 1);
    return Promise.resolve({ ok: true, json: async () => ({}) });
  };
  const { api, window: win } = loadDataJs({ fetchStub: fetchImpl });
  const opts = { state: api.createPollState(), deps: { fetchImpl }, jitterMaxMs: 0 };

  win.__DF_PAUSE = true;
  api.pollTick(opts);
  await drain();
  assert.equal(callCount.size, 0, 'pollTick must issue zero fetches while __DF_PAUSE is true');

  win.__DF_PAUSE = false;
  api.pollTick(opts);
  await drain();
  assert.equal(
    callCount.get(UNWINDOWED_ENDPOINT_PATH),
    1,
    'pollTick must resume fetching once __DF_PAUSE is set back to false, on the very next tick',
  );
});

test('paused: a window-chip change (app.jsx [win] effect -> DF_REFRESH) fetches nothing and dispatches nothing', async () => {
  const load = pauseLoad();
  load.window.__DF_PAUSE = true;

  await load.api.refreshDFData('7d', load.opts);
  await drain();

  assert.deepEqual(load.requested, [], 'a chip change while paused must not fetch');
  assert.equal(load.refreshEvents(), 0, 'a chip change while paused must not dispatch df-data-refresh');
});

test('paused: the chip change is still recorded, and the first tick after resume fetches at that window', async () => {
  const load = pauseLoad();
  load.window.__DF_PAUSE = true;
  await load.api.refreshDFData('7d', load.opts);
  await drain();

  load.window.__DF_PAUSE = false;
  load.api.pollTick(load.opts);
  await drain();

  const windowed = load.requested.filter(url => url.includes('?window='));
  const expected = Object.keys(load.api.endpointsFor('7d')).filter(url => url.includes('?window='));
  assert.ok(expected.length > 0, 'endpointsFor must declare windowed endpoints for this test to mean anything');
  assert.deepEqual(uniqueSorted(windowed), uniqueSorted(expected),
    `the resumed tick must fetch every windowed endpoint at ?window=7d; fetched ${windowed.join(', ')}`);
});

test('paused: a tab switch fetches nothing and dispatches nothing', async () => {
  const load = pauseLoad();
  await load.api.scopePollingToTab(FIRST_TAB, undefined, load.opts);
  const newlyNeeded = Object.keys(load.api.pollSetFor(SWITCHED_TAB, '24h'))
    .filter(url => !(url in load.api.pollSetFor(FIRST_TAB, '24h')));
  assert.ok(newlyNeeded.length > 0,
    `${SWITCHED_TAB} must need an endpoint ${FIRST_TAB} does not poll, or this test proves nothing`);
  load.clear();
  const eventsBefore = load.refreshEvents();

  load.window.__DF_PAUSE = true;
  await load.api.scopePollingToTab(SWITCHED_TAB, undefined, load.opts);
  await drain();

  assert.deepEqual(load.requested, [], 'a tab switch while paused must not fetch');
  assert.equal(load.refreshEvents(), eventsBefore, 'a tab switch while paused must not dispatch df-data-refresh');
});

test('paused: the tab switch is still recorded, and the first tick after resume polls that tab\'s set', async () => {
  const load = pauseLoad();
  await load.api.scopePollingToTab(FIRST_TAB, undefined, load.opts);
  load.window.__DF_PAUSE = true;
  await load.api.scopePollingToTab(SWITCHED_TAB, undefined, load.opts);
  await drain();
  load.clear();

  load.window.__DF_PAUSE = false;
  load.api.pollTick(load.opts);
  await drain();

  assert.deepEqual(uniqueSorted(load.requested), uniqueSorted(Object.keys(load.api.pollSetFor(SWITCHED_TAB, '24h'))));
});

// A deliberate exemption, pinned so that changing it is a decision: an
// on-demand row is one user-requested fetch, not the poll loop, and pausing
// it would leave the asking pane waiting on a request that never starts.
test('paused: requestOnDemand still fetches its row', async () => {
  const load = pauseLoad();
  load.window.__DF_PAUSE = true;

  await load.api.requestOnDemand('terminal', ON_DEMAND_PROJECT, load.opts);

  assert.deepEqual(load.requested, [load.api.ON_DEMAND_KEYS.terminal.url(ON_DEMAND_PROJECT)]);
});
