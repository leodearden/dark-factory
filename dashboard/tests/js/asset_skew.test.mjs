// Module-contract tests for asset_skew.js: whether an open page, the server
// process that answers it, and the code on disk are the same static/redux
// build, and what the page does when they are not.
//
// Three identities: P, the build the page booted with (its df-asset-build
// meta); S, the build the serving process started with; D, the build on disk
// now. S and D arrive on every receipt as assetBuild {started, onDisk}.
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const SPECIFIER = '../../src/dashboard/static/redux/asset_skew.js';

function freshRequire(win) {
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  delete require.cache[require.resolve(SPECIFIER)];
  return require(SPECIFIER);
}

const loadedWindow = {};
const skew = freshRequire(loadedWindow);
const {
  ASSET_SKEW_STATES,
  ASSET_SKEW_RELOAD_GUARD_MS,
  ASSET_SKEW_GUARD_KEY,
  createAssetSkewWatcher,
  assetSkewNotice,
  latestAssetBuild,
  pageAssetBuild,
  assetSkewGuardStorage,
} = skew;

const A = '0123456789abcdef';
const B = 'fedcba9876543210';
const C = 'a1b2c3d4e5f60718';
const START = 1_800_000_000_000;

function memoryStorage(initial = {}) {
  const items = new Map(Object.entries(initial));
  return {
    items,
    getItem: key => (items.has(key) ? items.get(key) : null),
    setItem: (key, value) => {
      items.set(key, String(value));
    },
  };
}

function receiptsAt(receivedAt, started, onDisk, path = '/api/v2/dashboard/memory') {
  return { [path]: { receivedAt, assetBuild: { started, onDisk } } };
}

// A watcher over injected storage, clock and reload. `observe(started, onDisk)`
// lands one new poll, each with a newer receivedAt. An explicit `pageBuild:
// undefined` is kept, not defaulted, so the garbled-page cases can pass it.
function harness(options = {}) {
  const { storage = memoryStorage(), guardMs } = options;
  const pageBuild = 'pageBuild' in options ? options.pageBuild : A;
  const clock = { now: START };
  const reloads = [];
  const watcher = createAssetSkewWatcher({
    pageBuild,
    storage,
    now: () => clock.now,
    reload: () => reloads.push(clock.now),
    guardMs,
  });
  return {
    clock,
    reloads,
    storage,
    watcher,
    observe(started, onDisk) {
      clock.now += 1000;
      return watcher.observe(receiptsAt(clock.now, started, onDisk));
    },
  };
}

// ---------------------------------------------------------------------------
// In step
// ---------------------------------------------------------------------------

test('in step: P = S = D is in-step over many polls, with no reload and no notice', () => {
  const h = harness();

  for (let i = 0; i < 5; i += 1) {
    const state = h.observe(A, A);
    assert.equal(state.kind, ASSET_SKEW_STATES.inStep);
    assert.equal(state.pageDiffersFromServer, false);
    assert.equal(assetSkewNotice(state), null);
  }
  assert.deepEqual(h.reloads, []);
});

// ---------------------------------------------------------------------------
// Stale page: P differs from a server that matches the disk
// ---------------------------------------------------------------------------

test('stale page: the first observation only confirms, writing no stamp', () => {
  const h = harness();

  const state = h.observe(B, B);

  assert.equal(state.kind, ASSET_SKEW_STATES.confirming);
  assert.equal(state.pageDiffersFromServer, true);
  assert.deepEqual(h.reloads, []);
  assert.equal(h.storage.items.size, 0);
});

test('stale page: a second distinct observation reloads exactly once and stamps the guard at now()', () => {
  const h = harness();
  h.observe(B, B);

  const state = h.observe(B, B);

  assert.equal(state.kind, ASSET_SKEW_STATES.reloading);
  assert.equal(state.pageDiffersFromServer, true);
  assert.deepEqual(h.reloads, [h.clock.now]);
  assert.equal(h.storage.items.get(ASSET_SKEW_GUARD_KEY), String(h.clock.now));
});

test('stale page: later observations never reload again', () => {
  const h = harness();
  h.observe(B, B);
  h.observe(B, B);

  for (let i = 0; i < 5; i += 1) {
    h.clock.now += ASSET_SKEW_RELOAD_GUARD_MS * 2;
    assert.equal(h.observe(B, B).kind, ASSET_SKEW_STATES.reloading);
  }
  assert.equal(h.reloads.length, 1);
});

test('stale page: the same receipt seen twice is one observation, not a confirmation', () => {
  const h = harness();
  const receipts = receiptsAt(START + 1000, B, B);

  h.watcher.observe(receipts);
  const state = h.watcher.observe(receipts);

  assert.equal(state.kind, ASSET_SKEW_STATES.confirming);
  assert.deepEqual(h.reloads, []);
});

test('stale page: an in-step or unknown observation between two stale ones resets the count', () => {
  for (const [label, started, onDisk] of [
    ['in step', A, A],
    ['unknown', null, null],
  ]) {
    const h = harness();
    h.observe(B, B);

    h.observe(started, onDisk);
    const state = h.observe(B, B);

    assert.equal(state.kind, ASSET_SKEW_STATES.confirming, label);
    assert.deepEqual(h.reloads, [], label);
  }
});

test('stale page: a different target server build restarts the confirmation', () => {
  const h = harness();
  h.observe(B, B);

  const state = h.observe(C, C);

  assert.equal(state.kind, ASSET_SKEW_STATES.confirming);
  assert.deepEqual(h.reloads, []);
});

// ---------------------------------------------------------------------------
// The reload loop guard
// ---------------------------------------------------------------------------

test('guard: a stamp younger than the guard window holds the reload and says to reload', () => {
  const storage = memoryStorage({ [ASSET_SKEW_GUARD_KEY]: String(START) });
  const h = harness({ storage });
  h.observe(B, B);

  const state = h.observe(B, B);

  assert.ok(ASSET_SKEW_RELOAD_GUARD_MS > 2000, 'the harness polls stay inside the guard window');
  assert.equal(state.kind, ASSET_SKEW_STATES.reloadGuarded);
  assert.equal(state.pageDiffersFromServer, true);
  assert.deepEqual(h.reloads, []);
  const notice = assetSkewNotice(state);
  assert.ok(notice, 'a guarded page tells the operator to reload');
  assert.match(notice.text, /reload/i);
  assert.equal(notice.offersReload, true);
});

test('guard: a guarded page stays guarded on later observations', () => {
  const storage = memoryStorage({ [ASSET_SKEW_GUARD_KEY]: String(START) });
  const h = harness({ storage });
  h.observe(B, B);
  h.observe(B, B);

  for (let i = 0; i < 5; i += 1) {
    h.clock.now += ASSET_SKEW_RELOAD_GUARD_MS * 2;
    assert.equal(h.observe(B, B).kind, ASSET_SKEW_STATES.reloadGuarded);
  }
  assert.deepEqual(h.reloads, []);
});

test('guard: a stamp older than the guard window lets the reload through', () => {
  const storage = memoryStorage({ [ASSET_SKEW_GUARD_KEY]: String(START - ASSET_SKEW_RELOAD_GUARD_MS - 1) });
  const h = harness({ storage });
  h.observe(B, B);

  const state = h.observe(B, B);

  assert.equal(state.kind, ASSET_SKEW_STATES.reloading);
  assert.equal(h.reloads.length, 1);
});

test('guard: the window is the injected guardMs when one is given', () => {
  const storage = memoryStorage({ [ASSET_SKEW_GUARD_KEY]: String(START) });
  const h = harness({ storage, guardMs: 500 });
  h.observe(B, B);

  assert.equal(h.observe(B, B).kind, ASSET_SKEW_STATES.reloading);
});

test('guard: no storage, or a stamp write that throws, holds the reload', () => {
  const throwingWrite = { getItem: () => null, setItem: () => { throw new Error('QuotaExceededError'); } };
  const throwingRead = { getItem: () => { throw new Error('SecurityError'); }, setItem: () => {} };
  for (const [label, storage] of [
    ['null storage', null],
    ['setItem throws', throwingWrite],
    ['getItem throws', throwingRead],
  ]) {
    const h = harness({ storage });
    h.observe(B, B);

    const state = h.observe(B, B);

    assert.equal(state.kind, ASSET_SKEW_STATES.reloadGuarded, label);
    assert.deepEqual(h.reloads, [], label);
  }
});

test('guard: a garbled stamp counts as no stamp, a future one as recent', () => {
  for (const garbled of ['', 'yesterday', 'NaN', '{}']) {
    const h = harness({ storage: memoryStorage({ [ASSET_SKEW_GUARD_KEY]: garbled }) });
    h.observe(B, B);
    assert.equal(h.observe(B, B).kind, ASSET_SKEW_STATES.reloading, garbled);
  }

  const future = harness({ storage: memoryStorage({ [ASSET_SKEW_GUARD_KEY]: String(START + 10 * ASSET_SKEW_RELOAD_GUARD_MS) }) });
  future.observe(B, B);
  assert.equal(future.observe(B, B).kind, ASSET_SKEW_STATES.reloadGuarded);
  assert.deepEqual(future.reloads, []);
});

// ---------------------------------------------------------------------------
// Stale server: the process predates the disk. A banner, never a reload.
// ---------------------------------------------------------------------------

test('stale server: S differs from D shows a restart-pending notice and never reloads', () => {
  for (const [label, pageBuild, differs] of [
    ['the page matches the server', A, false],
    ['the page matches the disk', B, true],
  ]) {
    const h = harness({ pageBuild });

    for (let i = 0; i < 6; i += 1) {
      const state = h.observe(A, B);
      assert.equal(state.kind, ASSET_SKEW_STATES.staleServer, label);
      assert.equal(state.pageDiffersFromServer, differs, label);
      const notice = assetSkewNotice(state);
      assert.ok(notice, label);
      assert.match(notice.text, /restart/i, label);
      assert.equal(notice.offersReload, false, label);
    }
    assert.deepEqual(h.reloads, [], label);
  }
});

// ---------------------------------------------------------------------------
// Missing or garbled identities: unknown, and never a reload storm
// ---------------------------------------------------------------------------

test('unknown: a missing or garbled page build takes no action', () => {
  for (const pageBuild of [null, undefined, '', 'null', 'XYZ', A.toUpperCase(), A.slice(0, 12), 12, {}]) {
    const h = harness({ pageBuild });
    for (let i = 0; i < 6; i += 1) {
      const state = h.observe(B, B);
      assert.equal(state.kind, ASSET_SKEW_STATES.unknown, String(pageBuild));
      assert.equal(assetSkewNotice(state), null);
    }
    assert.deepEqual(h.reloads, [], String(pageBuild));
  }
});

test('unknown: missing or garbled served builds take no action', () => {
  const garbled = [null, undefined, '', 'null', 'XYZ', B.toUpperCase(), B.slice(0, 12), 12, {}, ['x']];
  const cases = [
    ['no receipts', () => ({})],
    ['null receipts', () => null],
    ['assetBuild null', at => ({ '/api/v2/dashboard/memory': { receivedAt: at, assetBuild: null } })],
    ['assetBuild absent', at => ({ '/api/v2/dashboard/memory': { receivedAt: at } })],
    ...garbled.flatMap(value => [
      [`started ${JSON.stringify(value)}`, at => receiptsAt(at, value, B)],
      [`onDisk ${JSON.stringify(value)}`, at => receiptsAt(at, B, value)],
    ]),
  ];
  for (const [label, receiptsFor] of cases) {
    const h = harness();
    for (let i = 1; i <= 6; i += 1) {
      const state = h.watcher.observe(receiptsFor(START + i * 1000));
      assert.equal(state.kind, ASSET_SKEW_STATES.unknown, label);
      assert.equal(state.pageDiffersFromServer, false, label);
      assert.equal(assetSkewNotice(state), null, label);
    }
    assert.deepEqual(h.reloads, [], label);
  }
});

test('notice: no state, an in-step and a confirming state carry no notice', () => {
  assert.equal(assetSkewNotice(null), null);
  assert.equal(assetSkewNotice(undefined), null);
  const h = harness();
  assert.equal(assetSkewNotice(h.observe(A, A)), null);
  assert.equal(assetSkewNotice(h.observe(B, B)), null);
});

// ---------------------------------------------------------------------------
// Readers
// ---------------------------------------------------------------------------

test('latestAssetBuild picks the newest receipt by receivedAt', () => {
  const receipts = {
    '/api/v2/dashboard/memory': { receivedAt: 300, assetBuild: { started: B, onDisk: C } },
    '/api/v2/dashboard/tasks': { receivedAt: 900, assetBuild: { started: A, onDisk: B } },
    '/api/v2/dashboard/costs': { receivedAt: 600, assetBuild: { started: C, onDisk: C } },
  };

  assert.deepEqual(latestAssetBuild(receipts), { started: A, onDisk: B, receivedAt: 900 });
  assert.equal(latestAssetBuild({}), null);
  assert.equal(latestAssetBuild(null), null);
  assert.equal(latestAssetBuild({ x: { receivedAt: 1, assetBuild: { started: 'XYZ', onDisk: A } } }), null);
});

test('pageAssetBuild reads the df-asset-build meta, and is null when it is missing or empty', () => {
  const docWith = content => ({
    querySelector: selector => (selector === 'meta[name="df-asset-build"]' && content !== undefined
      ? { getAttribute: name => (name === 'content' ? content : null) }
      : null),
  });

  assert.equal(pageAssetBuild(docWith(A)), A);
  assert.equal(pageAssetBuild(docWith('')), null);
  assert.equal(pageAssetBuild(docWith('XYZ')), null);
  assert.equal(pageAssetBuild(docWith(undefined)), null);
  assert.equal(pageAssetBuild(null), null);
  assert.equal(pageAssetBuild({}), null);
});

test('assetSkewGuardStorage is the session storage, or null when reading it throws', () => {
  const session = memoryStorage();
  globalThis.window = { sessionStorage: session };
  assert.equal(assetSkewGuardStorage(), session);

  globalThis.window = Object.defineProperty({}, 'sessionStorage', {
    get() { throw new Error('SecurityError'); },
  });
  assert.equal(assetSkewGuardStorage(), null);

  globalThis.window = {};
  assert.equal(assetSkewGuardStorage(), null);
});

test('exports are published identically on module.exports and window.DF_ASSET_SKEW', () => {
  assert.equal(loadedWindow.DF_ASSET_SKEW, skew);
  for (const name of ['createAssetSkewWatcher', 'assetSkewNotice', 'latestAssetBuild', 'pageAssetBuild', 'assetSkewGuardStorage']) {
    assert.equal(typeof skew[name], 'function', name);
  }
  assert.ok(Object.isFrozen(ASSET_SKEW_STATES));
  assert.equal(typeof ASSET_SKEW_GUARD_KEY, 'string');
});
