// The app shell's render tests: app.jsx's poll-scope announcements, the
// endpoints each tab reads, and the tab slot's error boundary. The real page renders in a jsdom window
// (dom_harness.mjs); every expected value is taken from the shipped helpers'
// public globals, never from script text.
import { after, test } from 'node:test';
import assert from 'node:assert/strict';

import { caughtFrom } from './_caught.mjs';
import {
  assertReportsOnlyContainedThrows,
  click,
  failureNotices,
  noticeLines,
  openContainingWindow,
  openTab,
  railButton,
} from './_render_helpers.mjs';
import { buildDashboardWindow, seedDashboardData } from './dom_harness.mjs';
import {
  everyPolledKey,
  orchestratorRow,
  projectRow,
  receiptsFor,
  servingEndpoint,
  taskCensus,
  taskRow,
  tasksSnapshotEntry,
} from './wire_fixtures.mjs';

const PROJECT = 'p';

// ── App tells data.js which tab is open, and at which window ──────────────
// app.jsx binds scopePollingToTab off window.DF_DATA_LOADER when it loads,
// which is inside mountPage(), and calls window.DF_REFRESH when it refreshes.
// So both are wrapped on the window first, and each wrapper calls through.

function recordPollCalls(window) {
  const calls = [];
  const loader = window.DF_DATA_LOADER;
  const refresh = window.DF_REFRESH;
  window.DF_DATA_LOADER = {
    ...loader,
    scopePollingToTab(tab, win, opts) {
      calls.push({ scope: tab, win });
      return loader.scopePollingToTab(tab, win, opts);
    },
  };
  window.DF_REFRESH = (win, opts) => {
    calls.push({ refresh: win });
    return refresh(win, opts);
  };
  return calls;
}

function openRecordedPage(t) {
  const harness = openContainingWindow(t);
  const calls = recordPollCalls(harness.window);
  return { ...harness, calls, root: harness.mountPage() };
}

// The calls `action` makes.
function callsDuring(page, action) {
  const before = page.calls.length;
  action();
  return page.calls.slice(before);
}

function chooseWindow(page, win) {
  const button = [...page.root.querySelectorAll('.toolbar .chips button')].find(chip => chip.textContent === win);
  assert.ok(button, `the toolbar offers no "${win}" window chip`);
  click(page.window, button);
}

const scopeCalls = calls => calls.filter(call => 'scope' in call);

test('app.jsx announces the open tab at mount, at its resolved window, before its first DF_REFRESH', t => {
  const page = openRecordedPage(t);
  const { windowForTab, DEFAULT_WINDOW } = page.window.DF_WINDOW_CHIP;

  const win = windowForTab('overview', DEFAULT_WINDOW);
  assert.deepEqual(page.calls[0], { scope: 'overview', win }, `the page's first poll call: ${JSON.stringify(page.calls)}`);
  const firstRefresh = page.calls.findIndex(call => 'refresh' in call);
  assert.ok(firstRefresh > 0, `the page never refreshed after announcing its tab: ${JSON.stringify(page.calls)}`);
  assert.deepEqual(page.calls[firstRefresh], { refresh: win });
});

test('a window-chip click refreshes at the chosen window and re-announces no tab', t => {
  const page = openRecordedPage(t);
  const { TAB_WINDOWS } = page.window.DF_WINDOW_CHIP;
  assert.ok(TAB_WINDOWS.overview.windows.includes('7d'), 'Overview, the page\'s first tab, must offer 7d');

  const calls = callsDuring(page, () => chooseWindow(page, '7d'));

  assert.deepEqual(scopeCalls(calls), [], 'a chip click re-announced the tab');
  assert.ok(calls.some(call => call.refresh === '7d'), `a chip click must refresh at 7d: ${JSON.stringify(calls)}`);
});

test('a rail switch announces the new tab exactly once, at the window windowForTab resolves', t => {
  const page = openRecordedPage(t);
  const { windowForTab } = page.window.DF_WINDOW_CHIP;
  chooseWindow(page, '7d');

  const calls = callsDuring(page, () => openTab(page, 'Performance'));

  assert.deepEqual(scopeCalls(calls), [{ scope: 'perf', win: windowForTab('perf', '7d') }]);
});

test('a rail switch to a tab that does not offer the chosen window announces it at the reset window', t => {
  const page = openRecordedPage(t);
  const { TAB_WINDOWS, windowForTab } = page.window.DF_WINDOW_CHIP;
  assert.ok(TAB_WINDOWS.burn.windows.includes('90d') && !TAB_WINDOWS.cost.windows.includes('90d'),
    'the reset arm needs a window Burndown offers and Costs does not');
  openTab(page, 'Burndown');
  chooseWindow(page, '90d');

  const calls = callsDuring(page, () => openTab(page, 'Costs'));

  const resetWin = windowForTab('cost', '90d');
  assert.notEqual(resetWin, '90d');
  assert.deepEqual(scopeCalls(calls), [{ scope: 'cost', win: resetWin }]);
  const announced = calls.findIndex(call => 'scope' in call);
  const refreshed = calls.findIndex(call => call.refresh === resetWin);
  assert.ok(announced < refreshed, `the reset window must be announced before it is refreshed: ${JSON.stringify(calls)}`);
});

// ── app.jsx has no fallback for the modules it binds when it loads ────────
// The browser's Babel compiles each destructure to member reads, so the
// TypeError names the binding read off the missing module, not the module:
// the throw is identified by its frame in app.jsx instead.

for (const global of ['DF_DATA_LOADER', 'DF_RENDER_FAILURE']) {
  test(`app.jsx refuses to load without window.${global}`, t => {
    const harness = openContainingWindow(t);
    delete harness.window[global];

    assert.throws(
      () => harness.mountPage(),
      err => err instanceof harness.window.TypeError && err.stack.includes('app.jsx'),
      `a page bootstrapped without window.${global} must fail loudly, in app.jsx, rather than load on a fallback`,
    );
  });
}

// ── Every tab polls what it reads, and the chrome reads its endpoints everywhere ──
// Tab modules alias window.DF_DATA when they load, and shell.jsx reads
// PROJECTS/AGENTS then, so the recorder is installed before any script runs:
// an accessor whose setter wraps data.js's DF_DATA in a recording Proxy.

function readRecorder() {
  const atLoad = { keys: new Set(), rows: false };
  let measuring = atLoad;
  return {
    atLoad,
    beforeScripts(window) {
      let recorded;
      Object.defineProperty(window, 'DF_DATA', {
        configurable: true,
        enumerable: true,
        get: () => recorded,
        set(data) {
          recorded = new window.Proxy(data, {
            get(target, key, receiver) {
              if (measuring && typeof key === 'string' && !key.startsWith('__')) measuring.keys.add(key);
              return Reflect.get(target, key, receiver);
            },
          });
        },
      });
    },
    loaded() {
      measuring = null;
    },
    rowsRead() {
      if (measuring) measuring.rows = true;
    },
    during(action) {
      const reads = { keys: new Set(), rows: false };
      measuring = reads;
      try {
        action();
      } finally {
        measuring = null;
      }
      return reads;
    },
  };
}

// The keys data.js's on-demand rows build. requestOnDemand fetches those, not
// the poll, so they are no poll set's to serve.
function recordOnDemandKeys(loader) {
  const built = new Set();
  for (const row of Object.values(loader.ON_DEMAND_KEYS)) {
    const buildKey = row.key;
    row.key = (...args) => {
      const key = buildKey(...args);
      built.add(key);
      return key;
    };
  }
  return built;
}

// One running orchestrator and project, holding one in-progress task whose
// rows Datum records being read, and a receipt for every polled endpoint, so
// every reader gets past its not-yet-fetched guard.
function seedReadSetPage(window, recorder) {
  const task = taskRow({ project: PROJECT, agent: 'impl' });
  const entry = tasksSnapshotEntry([task], taskCensus({ [task.status]: 1 }, window.DF_TASK_VOCAB));
  const rows = entry.rows;
  Object.defineProperty(entry, 'rows', {
    enumerable: true,
    get() {
      recorder.rowsRead();
      return rows;
    },
  });
  seedDashboardData(window, {
    ORCHESTRATORS: [orchestratorRow({ pid: 1, project: PROJECT, running: true })],
    PROJECTS: [projectRow(PROJECT)],
    TASKS_SNAPSHOT: { [PROJECT]: entry },
    __receipt: receiptsFor(window.DF_DATA_LOADER, everyPolledKey(window.DF_DATA_LOADER)),
  });
}

// What each tab the rail opens reads, by tab id, with the window it was
// announced at: Overview while the page mounts, every other tab while its rail
// button is clicked. Keys read at load count on every tab.
function measureEveryTab() {
  const recorder = readRecorder();
  const harness = buildDashboardWindow({ beforeScripts: recorder.beforeScripts });
  try {
    const { window } = harness;
    recorder.loaded();
    seedReadSetPage(window, recorder);
    const onDemand = recordOnDemandKeys(window.DF_DATA_LOADER);
    const calls = recordPollCalls(window);
    const tabs = new Map();
    const measure = action => {
      const before = calls.length;
      const reads = recorder.during(action);
      const announced = scopeCalls(calls.slice(before));
      assert.equal(announced.length, 1, `one rail action must announce one tab: ${JSON.stringify(announced)}`);
      const [{ scope: tab, win }] = announced;
      const keys = [...recorder.atLoad.keys, ...reads.keys].filter(key => !onDemand.has(key));
      tabs.set(tab, { win, keys: new Set(keys), rows: reads.rows });
    };
    let root;
    measure(() => {
      root = harness.mountPage();
    });
    const [overviewButton, ...otherButtons] = root.querySelectorAll('.rail-nav button');
    assert.equal(overviewButton, railButton(root, 'Overview'), "the rail's first button must be the page's first tab");
    for (const button of otherButtons) measure(() => click(window, button));
    return { harness, loader: window.DF_DATA_LOADER, pollScope: window.DF_POLL_SCOPE, tabs };
  } catch (err) {
    harness.close();
    throw err;
  }
}

let everyTab = null;

function readsOfEveryTab() {
  everyTab ??= measureEveryTab();
  return everyTab;
}

after(() => everyTab?.harness.close());

function pathsRead(loader, { keys, win }) {
  return new Set([...keys].map(key => servingEndpoint(loader, key, win).stateKey));
}

function polledPaths(loader, { tab, win }) {
  return new Set(Object.keys(loader.pollSetFor(tab, win)).map(url => loader.pollKey(url)));
}

const TASKS_PATH = '/api/v2/dashboard/tasks';

test('every tab the rail opens has a TAB_ENDPOINTS entry', () => {
  const { tabs, pollScope } = readsOfEveryTab();
  for (const tab of tabs.keys()) {
    assert.ok(Array.isArray(pollScope.TAB_ENDPOINTS[tab]), `the rail opens tab ${tab}, which has no TAB_ENDPOINTS entry`);
  }
});

test('every key a tab reads, at load or while it renders, is served by an endpoint polled while it is open', () => {
  const { tabs, loader } = readsOfEveryTab();
  for (const [tab, reads] of tabs) {
    const polled = polledPaths(loader, { tab, win: reads.win });
    for (const key of reads.keys) {
      const { stateKey } = servingEndpoint(loader, key, reads.win);
      assert.ok(
        polled.has(stateKey),
        `tab ${tab} reads ${key}, served by ${stateKey}, which data.js does not poll while ${tab} is open, ` +
          `so it would sit frozen at its last value. List it in poll_scope.js::TAB_ENDPOINTS.${tab}`,
      );
    }
  }
});

test('a tab that reads task rows polls the full /tasks render, not its census form', () => {
  const { tabs, loader } = readsOfEveryTab();
  for (const [tab, reads] of tabs) {
    if (!reads.rows) continue;
    const fullUrl = Object.keys(loader.endpointsFor(reads.win)).find(url => loader.pollKey(url) === TASKS_PATH);
    assert.ok(
      fullUrl in loader.pollSetFor(tab, reads.win),
      `tab ${tab} renders task rows, which only ${fullUrl} serves, but polls ${Object.keys(loader.pollSetFor(tab, reads.win))}`,
    );
  }
});

test('every CHROME_ENDPOINTS path is read on every tab', () => {
  const { tabs, loader, pollScope } = readsOfEveryTab();
  for (const [tab, reads] of tabs) {
    const read = pathsRead(loader, reads);
    for (const path of pollScope.CHROME_ENDPOINTS) {
      assert.ok(read.has(path), `nothing on tab ${tab} reads ${path}, so CHROME_ENDPOINTS polls it on every tab for nothing`);
    }
  }
});

// Reads the measurement must see, as literals: a non-vacuity floor, not the
// contract. orch's Locks cells read SCHEDULER and overview's live feed reads
// MERGE_QUEUE; both were missing from TAB_ENDPOINTS until found by hand.
const CHROME_KEY_FLOOR = [
  'TASKS_SNAPSHOT', 'MERGE_QUEUE', 'MEMORY_STATUS', 'ORCHESTRATORS', 'RECON_STATE', 'ESCALATIONS', 'COSTS', 'PROJECTS', 'AGENTS',
];
const DASH = '/api/v2/dashboard';
const TAB_READ_FLOOR = {
  orch: { paths: [`${DASH}/scheduler`], rows: true },
  overview: { paths: [`${DASH}/merge-queue`], rows: false },
  tasks: { paths: [`${DASH}/tasks`, `${DASH}/invariants`, `${DASH}/prds`], rows: true },
};

test('the measured read sets hold what was found by hand', () => {
  const { tabs, loader } = readsOfEveryTab();
  for (const [tab, reads] of tabs) {
    const missing = CHROME_KEY_FLOOR.filter(key => !reads.keys.has(key));
    assert.deepEqual(missing, [], `tab ${tab} no longer reads the chrome's keys ${missing}`);
  }
  for (const [tab, floor] of Object.entries(TAB_READ_FLOOR)) {
    const reads = tabs.get(tab);
    const read = pathsRead(loader, reads);
    assert.deepEqual(floor.paths.filter(path => !read.has(path)), [], `tab ${tab} lost reads it is known to make`);
    assert.equal(reads.rows, floor.rows, `tab ${tab} ${floor.rows ? 'no longer reads' : 'now reads'} task rows`);
  }
});

// ── A tab's render throw is contained to the tab slot ─────────────────────
// app.jsx binds window.DF_TABS when it loads inside mountPage(), so a probe
// substituted there renders in the real tab slot. The probe hands the real
// assertDatum a bare number, as a tab handed one by its caller would.

const PROBE = 'CostsTab probe';
const HOUR_MS = 3_600_000;

// Fails one of `paths`' endpoints through data.js's own refreshOne until its
// streak reaches the staleness threshold, which publishes the genuine
// DF_DATA.__stale record. Returns that endpoint's path.
async function makeStale(window, paths) {
  const loader = window.DF_DATA_LOADER;
  const [path] = paths;
  const [url, keySpecs] = Object.entries(loader.endpointsFor('24h')).find(([candidate]) => loader.pollKey(candidate) === path);
  const state = loader.createPollState();
  let now = Date.now();
  const deps = {
    fetchImpl: () => Promise.reject(new Error('down')),
    now: () => now,
    setTimeoutImpl: () => 0,
    clearTimeoutImpl: () => {},
  };
  for (let attempt = 0; attempt < window.DF_ENDPOINT_STALENESS.STALE_FAILURE_THRESHOLD; attempt++) {
    await loader.refreshOne(url, keySpecs, state, deps);
    now += HOUR_MS;
  }
  return path;
}

test("app.jsx replaces a tab whose child throws with the tab's notice, and the stale banner and rail render on", async t => {
  const harness = openContainingWindow(t);
  const { window } = harness;
  const { DF_DATUM, DF_POLL_SCOPE, DF_ENDPOINT_STALENESS, DF_RENDER_FAILURE } = window;
  window.DF_TABS = { ...window.DF_TABS, CostsTab: () => DF_DATUM.assertDatum(42, PROBE) };
  const stalePath = await makeStale(window, DF_POLL_SCOPE.TAB_ENDPOINTS.cost);
  const page = { ...harness, root: harness.mountPage() };

  openTab(page, 'Costs');

  const notices = failureNotices(page.root);
  assert.deepEqual(notices.map(notice => notice.dataset.testid), ['tab-render-failure-banner']);
  const expected = DF_RENDER_FAILURE.tabRenderFailureNotice('Costs', caughtFrom(() => DF_DATUM.assertDatum(42, PROBE)));
  assert.deepEqual(noticeLines(notices[0]), [expected.text, expected.detail]);

  const staleBanners = [...page.root.querySelectorAll('.body [data-testid="endpoint-stale-banner"]')];
  const [staleNotice] = DF_ENDPOINT_STALENESS.staleNoticesForTab({
    paths: [stalePath],
    stale: window.DF_DATA.__stale,
    now: Date.now(),
  });
  assert.deepEqual(staleBanners.map(banner => banner.textContent), [staleNotice.text]);
  assert.ok(page.root.querySelectorAll('.rail-nav button').length > 0, 'the rail stopped rendering');
  assertReportsOnlyContainedThrows(page.consoleErrors);
});
