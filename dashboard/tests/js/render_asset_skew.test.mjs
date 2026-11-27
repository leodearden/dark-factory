// The page's asset-skew wiring in app.jsx, rendered in a jsdom window
// (dom_harness.mjs): the banner above the tab body, the one reload of a stale
// page, the guarded page's manual reload, and the tab render-failure notice,
// which says to reload only when that reaches the server's build. asset_skew.js's own contract
// is asset_skew.test.mjs; every expected text here is taken from its public
// global, never a literal.
//
// The harness builds a blank document, so a test boots the page with a build
// by appending the df-asset-build meta app.py::index would have served.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { caughtFrom } from './_caught.mjs';
import {
  assertReportsOnlyContainedThrows,
  failureNotices,
  noticeLines,
  openContainingWindow,
  openTab,
} from './_render_helpers.mjs';
import { buildDashboardWindow, seedDashboardData } from './dom_harness.mjs';

const A = '0123456789abcdef';
const B = 'fedcba9876543210';
const BANNER = '[data-testid="asset-skew-banner"]';
const PROBE_RECEIPT = 'asset-skew probe';

function bootWithBuild(window, pageBuild) {
  const meta = window.document.createElement('meta');
  meta.setAttribute('name', 'df-asset-build');
  meta.setAttribute('content', pageBuild);
  window.document.head.appendChild(meta);
}

// app.jsx creates its one watcher off window.DF_ASSET_SKEW when it loads,
// which is inside mountPage(). So the wrapper goes on the window first, and
// calls through with only `reload` replaced by a recorder.
function recordReloads(window) {
  const reloads = [];
  const skew = window.DF_ASSET_SKEW;
  window.DF_ASSET_SKEW = {
    ...skew,
    createAssetSkewWatcher: options => skew.createAssetSkewWatcher({ ...options, reload: () => reloads.push('reload') }),
  };
  return reloads;
}

// The whole page lands one async App update outside act() after it mounts
// (jsx_render_quality.test.mjs measured the same), so this window tolerates
// exactly that warning and nothing else on console.error.
const PAGE_ACT_WARNING = 'Warning: An update to %s inside a test was not wrapped in act(...)';

function openPageWindow(t) {
  const harness = buildDashboardWindow();
  t.after(() => {
    harness.close();
    assert.deepEqual(harness.consoleErrors.filter(entry => !entry.startsWith(PAGE_ACT_WARNING)), []);
  });
  return harness;
}

function openSkewPage(t, { pageBuild, open = openPageWindow, beforeMount = () => {} } = {}) {
  const harness = open(t);
  const { window } = harness;
  if (pageBuild !== undefined) bootWithBuild(window, pageBuild);
  const reloads = recordReloads(window);
  beforeMount(window);
  return { ...harness, reloads, root: harness.mountPage() };
}

// One poll set: a receipt served by a process started on `started` with
// `onDisk` on disk, newer than any before it, then the event data.js
// dispatches once per poll set.
let receivedAt = Date.now();
function poll(page, started, onDisk) {
  const { window } = page;
  receivedAt += 1000;
  const receipt = { servedAt: null, receivedAt, window: null, pacingHoldMs: 0, assetBuild: { started, onDisk } };
  seedDashboardData(window, { __receipt: { [PROBE_RECEIPT]: receipt } });
  window.React.act(() => window.dispatchEvent(new window.CustomEvent('df-data-refresh')));
}

// The state a fresh watcher reaches for the same builds, so the expected
// banner comes from the module rather than a literal.
function noticeFor(window, pageBuild, started, onDisk, storage = null) {
  const { createAssetSkewWatcher, assetSkewNotice } = window.DF_ASSET_SKEW;
  const watcher = createAssetSkewWatcher({ pageBuild, storage, now: () => Date.now(), reload: () => {} });
  let state = null;
  for (let i = 1; i <= 2; i += 1) {
    state = watcher.observe({ [PROBE_RECEIPT]: { receivedAt: i, assetBuild: { started, onDisk } } });
  }
  return assetSkewNotice(state);
}

function banners(page) {
  return [...page.root.querySelectorAll(BANNER)];
}

test('a server older than the code on disk shows the restart-pending banner above the tab body, and never reloads', t => {
  const page = openSkewPage(t, { pageBuild: A });

  for (let i = 0; i < 4; i += 1) poll(page, A, B);

  const shown = banners(page);
  assert.equal(shown.length, 1);
  const expected = noticeFor(page.window, A, A, B);
  assert.equal(expected.offersReload, false);
  assert.equal(shown[0].textContent, expected.text);
  assert.equal(page.root.querySelector('.body').firstElementChild, shown[0], 'the banner sits above the tab body');
  assert.equal(shown[0].querySelector('button'), null);
  assert.deepEqual(page.reloads, []);
});

test('a page older than the server reloads exactly once, after two distinct polls', t => {
  const page = openSkewPage(t, { pageBuild: A });

  poll(page, B, B);
  assert.deepEqual(page.reloads, [], 'one poll only confirms');
  poll(page, B, B);
  poll(page, B, B);

  assert.deepEqual(page.reloads, ['reload']);
});

test('a page that reloaded moments ago holds the reload and offers one instead', t => {
  let guardKey;
  const page = openSkewPage(t, {
    pageBuild: A,
    beforeMount: window => {
      guardKey = window.DF_ASSET_SKEW.ASSET_SKEW_GUARD_KEY;
      window.sessionStorage.setItem(guardKey, String(Date.now()));
    },
  });

  poll(page, B, B);
  poll(page, B, B);

  assert.deepEqual(page.reloads, []);
  const shown = banners(page);
  assert.equal(shown.length, 1);
  const expected = noticeFor(page.window, A, B, B, page.window.sessionStorage);
  assert.equal(expected.offersReload, true);
  assert.ok(shown[0].textContent.startsWith(expected.text), shown[0].textContent);
  assert.ok(shown[0].querySelector('button'), 'the guarded banner offers a reload control');
});

// A reload helps only a page older than a server that matches the disk. On the
// reverse skew (P = D = B, S = A) a reload serves B again, so the tab keeps
// the switch-tabs hint and the banner says a restart is pending.
for (const [label, pageBuild, started, onDisk, reloadReachesServerBuild] of [
  ['a page older than the server says to reload it', A, B, B, true],
  ['a page on the disk build, ahead of a stale server, does not say to reload it', B, A, B, false],
]) {
  test(`a tab that throws on ${label}`, t => {
    const PROBE = 'CostsTab probe';
    const page = openSkewPage(t, {
      pageBuild,
      open: openContainingWindow,
      beforeMount: window => {
        const { DF_DATUM } = window;
        window.DF_TABS = { ...window.DF_TABS, CostsTab: () => DF_DATUM.assertDatum(42, PROBE) };
      },
    });
    const { DF_DATUM, DF_RENDER_FAILURE } = page.window;

    poll(page, started, onDisk);
    openTab(page, 'Costs');

    const notices = failureNotices(page.root);
    assert.deepEqual(notices.map(notice => notice.dataset.testid), ['tab-render-failure-banner']);
    const thrown = caughtFrom(() => DF_DATUM.assertDatum(42, PROBE));
    const expected = DF_RENDER_FAILURE.tabRenderFailureNotice('Costs', thrown, reloadReachesServerBuild);
    assert.deepEqual(noticeLines(notices[0]), [expected.text, expected.detail]);
    assert.deepEqual(page.reloads, []);
    assertReportsOnlyContainedThrows(page.consoleErrors);
  });
}

test('a page on the disk build, ahead of a stale server, shows the restart-pending banner for a page off the server build', t => {
  const page = openSkewPage(t, { pageBuild: B });

  for (let i = 0; i < 4; i += 1) poll(page, A, B);

  const shown = banners(page);
  assert.equal(shown.length, 1);
  const expected = noticeFor(page.window, B, A, B);
  assert.notEqual(expected.text, noticeFor(page.window, A, A, B).text);
  assert.equal(shown[0].textContent, expected.text);
  assert.equal(shown[0].querySelector('button'), null);
  assert.deepEqual(page.reloads, []);
});

for (const [label, pageBuild, started, onDisk] of [
  ['a page with no df-asset-build meta', undefined, B, B],
  ['a page in step with the server and the disk', A, A, A],
]) {
  test(`${label} shows no banner and never reloads`, t => {
    const page = openSkewPage(t, { pageBuild });

    for (let i = 0; i < 4; i += 1) poll(page, started, onDisk);

    assert.deepEqual(banners(page), []);
    assert.deepEqual(page.reloads, []);
  });
}
