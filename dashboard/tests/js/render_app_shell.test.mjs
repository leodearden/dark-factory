// The app shell's render tests: app.jsx's poll-scope announcements, the
// endpoints each tab reads, the tab slot's error boundary, the shell's Datum
// components (charts.jsx::StatTile, shell.jsx::Pip and DatumReading), and
// OrchTab's per-project Locks column. The real page renders in a jsdom window
// (dom_harness.mjs); every expected value is taken from the shipped helpers'
// public globals, never from script text.
import { after, test } from 'node:test';
import assert from 'node:assert/strict';

import { caughtFrom } from './_caught.mjs';
import {
  UNIT_NOTICE_TESTID,
  assertReportsOnlyContainedThrows,
  assertSiblingRenders,
  click,
  failureNotices,
  graphNode,
  mountInsideBoundary,
  noticeLines,
  openContainingWindow,
  openTab,
  openWindow,
  railButton,
} from './_render_helpers.mjs';
import { buildDashboardWindow, seedDashboardData } from './dom_harness.mjs';
import {
  everyPolledKey,
  orchestratorRow,
  projectRow,
  receiptsFor,
  schedulerLockRow,
  schedulerModule,
  schedulerSnapshot,
  servingEndpoint,
  taskCensus,
  taskRow,
  tasksSnapshotEntry,
  wireDatum,
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
// the poll, so they are no poll set's to serve. Each row's key builder is
// wrapped in place, so this relies on ON_DEMAND_KEYS rows staying unfrozen.
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
// every reader gets past its not-yet-fetched guard. Returns the task.
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
  return task;
}

// The Tasks detail pane renders only once a task is selected. The project is
// listed flat through its own toggle, so the rail click before this still
// measured the default By-PRD render. `body` holds the Tasks tab and nothing else.
function openTaskDetail(window, body, task) {
  const listToggle = [...body.querySelectorAll('.seg button')].find(button => button.textContent === 'list');
  assert.ok(listToggle, 'the Tasks tab offers no "list" toggle');
  click(window, listToggle);
  const node = graphNode(body, task);
  assert.ok(node, 'the seeded task never reached the Tasks graph, so it cannot be selected');
  click(window, node);
  const sub = body.querySelector('.task-detail')?.querySelector('.sub');
  assert.ok(
    sub?.textContent.includes(window.DF_SHELL.taskId(task.id)),
    'the Tasks detail pane did not open on the seeded task, so its readers go unmeasured',
  );
}

// Views a tab renders only behind a click, by tab id: each action opens its
// view on the open tab, and what that view reads counts as the tab's.
const DEEPER_VIEWS = {
  tasks: openTaskDetail,
};

// The globals app.jsx binds its tab components off when it loads.
const TAB_MODULES = ['DF_OVERVIEW', 'DF_TABS', 'DF_TASKS', 'DF_CURATOR', 'DF_SCHEDULER', 'DF_QUALITY'];

// Wraps every *Tab component of TAB_MODULES, before mountPage binds them, so
// each render the page's tab slot makes is recorded with its props. take()
// returns the components rendered since the last take().
function recordTabBodies(window) {
  let rendered = new Map();
  for (const name of TAB_MODULES) {
    const module = window[name];
    assert.ok(module, `app.jsx binds tabs off window.${name}, which the page does not define`);
    const wrapped = { ...module };
    for (const [key, Component] of Object.entries(module)) {
      if (!key.endsWith('Tab') || typeof Component !== 'function') continue;
      wrapped[key] = props => {
        rendered.set(Component, props);
        return window.React.createElement(Component, props);
      };
    }
    window[name] = wrapped;
  }
  return {
    take() {
      const taken = rendered;
      rendered = new Map();
      return taken;
    },
  };
}

// What each tab the rail opens reads, by tab id, with the window it was
// announced at. Overview is measured while the page mounts, every other tab
// while its rail button is clicked, plus whatever DEEPER_VIEWS opens on it.
// Keys read at load count on every tab. `body` is what the tab's component
// reads mounted on its own, with the props the page gave it, and its
// DEEPER_VIEWS opened there: the tab's reads without the chrome's.
function measureEveryTab() {
  const recorder = readRecorder();
  const harness = buildDashboardWindow({ beforeScripts: recorder.beforeScripts });
  try {
    const { window } = harness;
    recorder.loaded();
    const task = seedReadSetPage(window, recorder);
    const onDemand = recordOnDemandKeys(window.DF_DATA_LOADER);
    const calls = recordPollCalls(window);
    const bodies = recordTabBodies(window);
    const readKeys = views => new Set(views.flatMap(view => [...view.keys]).filter(key => !onDemand.has(key)));
    const tabs = new Map();
    let root;
    const readsDuring = action => {
      const before = calls.length;
      const reads = recorder.during(action);
      return { reads, announced: scopeCalls(calls.slice(before)) };
    };
    const measureBody = (tab, rendered) => {
      assert.equal(
        rendered.size,
        1,
        `tab ${tab} must render one tab component in its slot, rendered ${rendered.size}; ` +
          'a tab bound off a global TAB_MODULES does not list renders none',
      );
      const [[Component, props]] = rendered;
      // The page's DEEPER_VIEWS persisted their toggles (Tasks' list view), and
      // the tab must be measured in its default view.
      window.localStorage.clear();
      const views = [];
      let mounted;
      views.push(recorder.during(() => {
        mounted = harness.mount(Component, props);
      }));
      try {
        const openDeeperView = DEEPER_VIEWS[tab];
        if (openDeeperView) views.push(recorder.during(() => openDeeperView(window, mounted.container, task)));
      } finally {
        mounted.unmount();
      }
      return { keys: readKeys(views), rows: views.some(view => view.rows) };
    };
    const measure = action => {
      bodies.take();
      const { reads, announced } = readsDuring(action);
      assert.equal(announced.length, 1, `one rail action must announce one tab: ${JSON.stringify(announced)}`);
      const [{ scope: tab, win }] = announced;
      const rendered = bodies.take();
      const views = [recorder.atLoad, reads];
      const openDeeperView = DEEPER_VIEWS[tab];
      if (openDeeperView) {
        const deeper = readsDuring(() => openDeeperView(window, root.querySelector('.body'), task));
        assert.deepEqual(deeper.announced, [], `opening a view on tab ${tab} announced a tab`);
        views.push(deeper.reads);
      }
      tabs.set(tab, {
        win,
        keys: readKeys(views),
        rows: views.some(view => view.rows),
        body: measureBody(tab, rendered),
      });
    };
    measure(() => {
      root = harness.mountPage();
    });
    const [overviewButton, ...otherButtons] = root.querySelectorAll('.rail-nav button');
    assert.ok(overviewButton === railButton(root, 'Overview'), "the rail's first button must be the page's first tab");
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

// The census every tab's chrome reads is polled as /tasks?projection=census,
// so a tab's entry owes /tasks only when the tab renders task rows.
test("every endpoint a tab's own component reads is listed in that tab's TAB_ENDPOINTS entry", () => {
  const { tabs, loader, pollScope } = readsOfEveryTab();
  for (const [tab, { win, body }] of tabs) {
    const listed = new Set(pollScope.TAB_ENDPOINTS[tab]);
    const owed = [...pathsRead(loader, { keys: body.keys, win })].filter(path => path !== TASKS_PATH || body.rows);
    const missing = owed.filter(path => !listed.has(path)).sort();
    assert.deepEqual(
      missing,
      [],
      `tab ${tab} reads ${missing}, which TAB_ENDPOINTS.${tab} does not list, so the tab shows no stale or loading ` +
        'notice for them, and stops polling them if the chrome stops reading them',
    );
  }
});

test('the measurement sees each tab component read on its own', () => {
  const { tabs } = readsOfEveryTab();
  for (const [tab, { body }] of tabs) {
    assert.ok(body.keys.size > 0, `tab ${tab}'s component, mounted on its own, read nothing off DF_DATA`);
  }
  assert.equal(tabs.get('tasks').body.rows, true, "the Tasks tab's component no longer reads task rows");
  assert.equal(tabs.get('overview').body.rows, false, "Overview's component now reads task rows");
  assert.ok(tabs.get('tasks').body.keys.has('PRD_VIEW'), "the Tasks tab's component was not measured in its default By-PRD view");
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
// MERGE_QUEUE; both were missing from TAB_ENDPOINTS until found by hand. The
// Tasks detail pane's Module locks read SCHEDULER.
const CHROME_KEY_FLOOR = [
  'TASKS_SNAPSHOT', 'MERGE_QUEUE', 'MEMORY_STATUS', 'ORCHESTRATORS', 'RECON_STATE', 'ESCALATIONS', 'COSTS', 'PROJECTS', 'AGENTS',
];
const DASH = '/api/v2/dashboard';
const TAB_READ_FLOOR = {
  orch: { paths: [`${DASH}/scheduler`], rows: true },
  overview: { paths: [`${DASH}/merge-queue`], rows: false },
  tasks: { paths: [`${DASH}/tasks`, `${DASH}/invariants`, `${DASH}/prds`, `${DASH}/scheduler`], rows: true },
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

// A path already stale gets no loading notice (endpoint_staleness.js::
// loadingNoticeForTab), so the loading half is its own page: nothing has
// answered Costs yet and nothing has failed.
test("app.jsx keeps the open tab's loading banner beside the tab's notice when the tab throws", t => {
  const harness = openContainingWindow(t);
  const { window } = harness;
  const { DF_DATUM, DF_POLL_SCOPE, DF_ENDPOINT_STALENESS } = window;
  window.DF_TABS = { ...window.DF_TABS, CostsTab: () => DF_DATUM.assertDatum(42, PROBE) };
  const page = { ...harness, root: harness.mountPage() };

  openTab(page, 'Costs');

  assert.deepEqual(failureNotices(page.root).map(notice => notice.dataset.testid), ['tab-render-failure-banner']);
  const loading = DF_ENDPOINT_STALENESS.loadingNoticeForTab({
    paths: DF_POLL_SCOPE.TAB_ENDPOINTS.cost,
    receipt: window.DF_DATA.__receipt || {},
    stale: window.DF_DATA.__stale || {},
  });
  assert.ok(loading, 'the page must start with Costs not yet delivered, or there is no loading banner to keep');
  const banners = [...page.root.querySelectorAll('.body [data-testid="endpoint-loading-banner"]')];
  assert.deepEqual(banners.map(banner => banner.textContent), [loading.text]);
  assertReportsOnlyContainedThrows(page.consoleErrors);
});

// ── The shell's Datum components draw datumView's answer ──────────────────
// Every expected value comes from window.DF_DATUM, never a literal of the
// markup.

const MINUTE_MS = 60_000;

// A Datum whose displayed age is 90 minutes, far from any boundary of
// formatAge's minute granularity, as a poll that received it now would stamp it.
function agedDatum(value, overrides) {
  const asOf = Date.parse(wireDatum(value).as_of);
  return wireDatum(value, {
    _served_at: new Date(asOf + 90 * MINUTE_MS).toISOString(),
    _received_at: Date.now(),
    ...overrides,
  });
}

const DRAWN_DATUMS = [
  { arm: 'a fresh Datum', datum: agedDatum(5) },
  { arm: 'a fresh measured zero', datum: agedDatum(0) },
  { arm: 'a stale Datum', datum: agedDatum(5, { state: 'stale', reason: 'the scheduler stopped answering' }) },
  { arm: 'a lower_bound Datum', datum: agedDatum(5, { state: 'lower_bound', reason: 'the runs ledger starts later' }) },
  { arm: 'an unknown Datum', datum: wireDatum(null, { state: 'unknown', reason: 'the map could not be read' }) },
];

const callerFormat = value => `${value} widgets`;

for (const { arm, datum } of DRAWN_DATUMS) {
  test(`charts.jsx::StatTile draws datumView's text, title and age over ${arm}`, t => {
    const harness = openWindow(t);
    const { DF_CHARTS, DF_DATUM } = harness.window;
    const { container } = harness.mount(DF_CHARTS.StatTile, { label: 'L', datum, format: callerFormat });

    const view = DF_DATUM.datumView(datum, { format: callerFormat, now: Date.now() });
    const val = container.querySelector('.kpi .val');
    assert.ok(val.textContent.startsWith(view.text), `"${val.textContent}" does not start with datumView's ${view.text}`);
    if (view.title === null) assert.equal(val.hasAttribute('title'), false);
    else assert.equal(val.title, view.title);
    const showsAge = [...container.querySelectorAll('.kpi span')].some(span => span.textContent === view.age);
    assert.equal(showsAge, !!view.age, view.age ? `the tile shows no age ${view.age}` : 'the tile shows an age datumView gives none');
  });
}

test('charts.jsx::StatTile draws a sparkline of its history', t => {
  const harness = openWindow(t);
  const { container } = harness.mount(harness.window.DF_CHARTS.StatTile, { label: 'L', datum: wireDatum(3), history: [1, 2, 3] });

  assert.ok(container.querySelector('.kpi .spark svg'), 'a tile given a history must draw its sparkline in .spark');
});

test('charts.jsx::StatTile handed a bare value and no Datum fails over through assertDatum', t => {
  const harness = openContainingWindow(t);
  const { React, DF_CHARTS, DF_DATUM } = harness.window;
  const { container, describe } = mountInsideBoundary(harness, React.createElement(DF_CHARTS.StatTile, { label: 'L', value: 5 }));

  const expected = describe(caughtFrom(() => DF_DATUM.datumView(undefined)));
  const notices = container.querySelectorAll(`[data-testid="${UNIT_NOTICE_TESTID}"]`);
  assert.equal(notices.length, 1);
  assert.deepEqual(noticeLines(notices[0]), [expected.text, expected.detail]);
  assertSiblingRenders(container);
  assertReportsOnlyContainedThrows(harness.consoleErrors);
});

test('shell.jsx::Pip handed a bare number fails over through assertDatum', t => {
  const harness = openContainingWindow(t);
  const { React, DF_SHELL, DF_DATUM } = harness.window;
  const { container, describe } = mountInsideBoundary(harness, React.createElement(DF_SHELL.Pip, { datum: 42 }));

  const expected = describe(caughtFrom(() => DF_DATUM.datumView(42)));
  const notices = container.querySelectorAll(`[data-testid="${UNIT_NOTICE_TESTID}"]`);
  assert.equal(notices.length, 1);
  assert.deepEqual(noticeLines(notices[0]), [expected.text, expected.detail]);
  assertSiblingRenders(container);
  assertReportsOnlyContainedThrows(harness.consoleErrors);
});

test("shell.jsx::Pip draws its dot and datumView's text through the caller's format", t => {
  const harness = openWindow(t);
  const { DF_SHELL, DF_DATUM } = harness.window;
  const datum = wireDatum(7);
  const { container } = harness.mount(DF_SHELL.Pip, { datum, format: callerFormat, color: 'red' });

  const pip = container.querySelector('.pip');
  assert.ok(pip.querySelector('.pip-dot'), 'a Pip given a colour draws no .pip-dot');
  const { text } = DF_DATUM.datumView(datum, { format: callerFormat });
  assert.ok(pip.textContent.startsWith(text), `"${pip.textContent}" does not start with datumView's ${text}`);
});

test("shell.jsx::DatumReading draws datumView's text through the caller's format", t => {
  const harness = openWindow(t);
  const { DF_SHELL, DF_DATUM } = harness.window;
  const datum = wireDatum(7);
  const { container } = harness.mount(DF_SHELL.DatumReading, { datum, format: callerFormat });

  assert.equal(container.textContent, DF_DATUM.datumView(datum, { format: callerFormat }).text);
});

// ── OrchTab decides the Locks column per project ──────────────────────────

const OFFLINE_PROJECT = 'pa';
const MEASURED_PROJECT = 'pb';
const TAB_PROPS = { projectFilter: [], search: '' };

function seedTwoProjectsWithScheduler(window) {
  const tasks = [taskRow({ project: OFFLINE_PROJECT, id: '41' }), taskRow({ project: MEASURED_PROJECT, id: '42' })];
  const lockSet = ['src/held', 'src/free'];
  seedDashboardData(window, {
    ORCHESTRATORS: tasks.map((task, i) => orchestratorRow({ pid: i + 1, project: task.project, running: true })),
    PROJECTS: tasks.map(task => projectRow(task.project)),
    TASKS_SNAPSHOT: Object.fromEntries(tasks.map(task => [
      task.project,
      tasksSnapshotEntry([task], taskCensus({ [task.status]: 1 }, window.DF_TASK_VOCAB)),
    ])),
    SCHEDULER: schedulerSnapshot({
      offlineProjects: [OFFLINE_PROJECT],
      rows: [schedulerLockRow({ project: MEASURED_PROJECT, taskId: '42', lockSet })],
      modules: [schedulerModule({ project: MEASURED_PROJECT, path: 'src/held', holder: '42', holderProject: MEASURED_PROJECT })],
    }),
    __receipt: receiptsFor(window.DF_DATA_LOADER, ['ORCHESTRATORS', 'TASKS_SNAPSHOT', 'SCHEDULER']),
  });
  return { tasks, lockSet };
}

// The Locks cell of `task`'s row in OrchTab's task table.
function locksCell(window, container, task) {
  const idText = window.DF_SHELL.taskId(task.id);
  const row = [...container.querySelectorAll('tbody tr')].find(tr => tr.querySelector('td')?.textContent === idText);
  assert.ok(row, `OrchTab draws no row for task ${task.id}`);
  const column = [...row.closest('table').querySelectorAll('thead th')].findIndex(th => th.textContent === 'Locks');
  assert.ok(column >= 0, 'the task table has no Locks column');
  return row.children[column];
}

test("tabs.jsx::OrchTab draws one project's offline Locks em-dash without blanking another project's chips", t => {
  const harness = openWindow(t);
  const { window } = harness;
  const { tasks: [offlineTask, measuredTask], lockSet } = seedTwoProjectsWithScheduler(window);
  const { container } = harness.mount(window.DF_TABS.OrchTab, TAB_PROPS);
  const D = window.DF_DATA;
  const { locksCellState, schedulerLocksDatum } = window.DF_TASK_ROW_CELLS;
  const expectedFor = task =>
    locksCellState(schedulerLocksDatum(D.SCHEDULER, task.project, D.__receipt), window.DF_SCHED_UTILS.buildSchedLockInfo(task, D.SCHEDULER));

  const offline = expectedFor(offlineTask);
  assert.ok(offline.placeholder && offline.title, 'locksCellState draws no reasoned placeholder offline, so this fixture tests nothing');
  const offlineCell = locksCell(window, container, offlineTask);
  assert.equal(offlineCell.querySelector('.chip-empty')?.textContent, offline.placeholder);
  assert.equal(offlineCell.querySelector('.chip-empty').title, offline.title);

  assert.equal(expectedFor(measuredTask).placeholder, null, 'locksCellState reads the measured project as unknown');
  const measuredCell = locksCell(window, container, measuredTask);
  assert.ok(!measuredCell.querySelector('.chip-empty'), "the offline project blanked the measured project's locks");
  const lockInfo = window.DF_SCHED_UTILS.buildSchedLockInfo(measuredTask, D.SCHEDULER);
  const classes = lockSet.map(path => window.DF_SCHED_UTILS.lockChipStateFor(lockInfo.moduleByPath.get(path), measuredTask.id, MEASURED_PROJECT).cls);
  assert.equal(new Set(classes).size, lockSet.length, 'each lock must take its own class, or the check below cannot tell the chips apart');
  const chips = [...measuredCell.querySelectorAll('.chip')];
  assert.equal(chips.length, lockSet.length, `the measured project draws ${chips.length} lock chips for ${lockSet.length} locks`);
  for (const cls of classes) assert.ok(chips.some(chip => chip.classList.contains(cls)), `no lock chip is ${cls}`);
});

// ── Every tab renders its Datum sites from Datums, and keys them off real receipts ──

// What each rail tab shows while it is open: its render-failure banners and
// its not-yet-fetched readings.
function openEveryRailTab(page) {
  return [...page.root.querySelectorAll('.rail-nav button')].map(button => {
    click(page.window, button);
    return {
      tab: page.root.querySelector('.crumbs .here')?.textContent,
      failures: failureNotices(page.root).map(notice => notice.dataset.testid),
      notYetFetched: notYetFetchedReadings(page),
    };
  });
}

// The reason every reader gives a reading whose endpoint has no receipt yet
// (datum.js, task_snapshot.js, merge_queue.js, escalation_views.js).
const NOT_YET_FETCHED = 'not yet fetched';

function notYetFetchedReadings(page) {
  return [...page.root.querySelectorAll('[title]')].filter(element => element.title === NOT_YET_FETCHED);
}

test('on the pre-fetch page no tab renders a render-failure banner, and its readings say not yet fetched', t => {
  const harness = openContainingWindow(t);
  const page = { ...harness, root: harness.mountPage() };

  const opened = openEveryRailTab(page);

  for (const { tab, failures } of opened) assert.deepEqual(failures, [], `tab ${tab} threw on the pre-fetch page`);
  assert.ok(opened.some(({ notYetFetched }) => notYetFetched.length > 0), 'no pre-fetch reading says not yet fetched, so the sweep below proves nothing');
  assert.deepEqual(page.consoleErrors, []);
});

test('once every polled endpoint has a receipt, no tab renders a not-yet-fetched reading', t => {
  const harness = openContainingWindow(t);
  const loader = harness.window.DF_DATA_LOADER;
  seedDashboardData(harness.window, { __receipt: receiptsFor(loader, everyPolledKey(loader)) });
  const page = { ...harness, root: harness.mountPage() };

  const opened = openEveryRailTab(page);

  for (const { tab, failures, notYetFetched } of opened) {
    assert.deepEqual(failures, [], `tab ${tab} threw once every endpoint had answered`);
    assert.deepEqual(notYetFetched.map(element => element.outerHTML), [], `tab ${tab} still reads an endpoint as not yet fetched`);
  }
});
