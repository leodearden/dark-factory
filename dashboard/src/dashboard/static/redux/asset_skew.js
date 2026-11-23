// asset_skew.js — whether this open page, the server process answering it and
// the code on disk are one static/redux build, and what the page does when
// they are not (task 10720).
//
// THREE IDENTITIES. P is the build the page booted with: the df-asset-build
// meta that app.py::index fills in when it serves `/`. S is the build the
// serving process started with, and D the build on disk now. The server stamps
// both on every /api/v2/dashboard/ response (dashboard/src/dashboard/asset_build.py),
// and data.js copies them raw onto each receipt as assetBuild {started, onDisk}.
// An identity is opaque: it is validated here and compared only for equality.
//
// THE VERDICT, in precedence order:
//   - any identity missing or garbled: unknown, and no action;
//   - S != D: the server predates the code on disk. A banner says a restart is
//     pending and the page NEVER reloads: a reload serves D, which still
//     differs from S, so it would create the reverse skew rather than cure it;
//   - P != S: the page is stale. It reloads, landing on the build the server
//     serves;
//   - otherwise in step.
//
// Each state says whether the page differs from the server, and separately
// whether a reload would bring it onto the server's build. The second is what
// a retry hint may act on: it is false on a stale server even when the first
// is true.
//
// A RELOAD needs two consecutive distinct observations of the same stale
// server build, and a sessionStorage stamp bounds it. A page that reloaded
// within the guard window, or cannot read or write the stamp, shows a banner
// asking for a reload instead. Once reloading or guarded, the page stays so
// until an observation that is not stale-page; a later stale run confirms
// afresh and may claim the stamp again. So the stamp spaces reloads at least
// the guard window apart, and one watcher calls reload at most once.
//
// LOAD CONTRACT. It reads no window.DF_* global at load. index.html loads it
// before the Babel JSX tags, so window.DF_ASSET_SKEW exists before app.jsx
// destructures it at module scope (test_asset_skew_registration.py pins the
// order). node requires the same file as CommonJS. Coverage is behavioural, in
// dashboard/tests/js/asset_skew.test.mjs.
//
// Every top-level name is module-unique, and every exported helper is a
// function declaration, not a const: a .jsx destructure of it compiles to a
// global `var`, which may share the classic scripts' scope with a `function`
// but not with a `const`.

const ASSET_SKEW_STATES = Object.freeze({
  unknown: 'unknown',
  inStep: 'in-step',
  confirming: 'confirming',
  reloading: 'reloading',
  reloadGuarded: 'reload-guarded',
  staleServer: 'stale-server',
});

const ASSET_SKEW_VERDICTS = Object.freeze({
  unknown: ASSET_SKEW_STATES.unknown,
  inStep: ASSET_SKEW_STATES.inStep,
  staleServer: ASSET_SKEW_STATES.staleServer,
  stalePage: 'stale-page',
});

const ASSET_SKEW_RELOAD_GUARD_MS = 5 * 60 * 1000;
const ASSET_SKEW_GUARD_KEY = 'df-asset-skew-reloaded-at';
const ASSET_SKEW_IDENTITY_PATTERN = /^[0-9a-f]{16,64}$/;
const ASSET_SKEW_CONFIRMATIONS = 2;

const ASSET_SKEW_RESTART_PENDING = 'The dashboard server is running an older build than the code on disk: a restart is pending. ';

// The stale-server banner, by whether the page matches the server.
const ASSET_SKEW_STALE_SERVER_NOTICES = Object.freeze({
  pageMatchesServer: Object.freeze({
    text: ASSET_SKEW_RESTART_PENDING + 'This page stays on the build the server is serving until then.',
    offersReload: false,
  }),
  pageDiffersFromServer: Object.freeze({
    text: ASSET_SKEW_RESTART_PENDING
      + 'This page is running a different build from the server, so parts of it may fail until then; '
      + 'reloading will not help.',
    offersReload: false,
  }),
});

const ASSET_SKEW_NOTICES = Object.freeze({
  [ASSET_SKEW_STATES.reloadGuarded]: Object.freeze({
    text: 'The dashboard has been updated since this page loaded. Reload the page to pick up the new build.',
    offersReload: true,
  }),
  [ASSET_SKEW_STATES.reloading]: Object.freeze({
    text: 'The dashboard has been updated since this page loaded. Reloading the page to pick up the new build.',
    offersReload: true,
  }),
});

function isAssetBuildIdentity(value) {
  return typeof value === 'string' && ASSET_SKEW_IDENTITY_PATTERN.test(value);
}

// P: the build this page booted with, or null when the page carries none.
function pageAssetBuild(doc) {
  if (!doc || typeof doc.querySelector !== 'function') return null;
  const meta = doc.querySelector('meta[name="df-asset-build"]');
  const content = meta ? meta.getAttribute('content') : null;
  return isAssetBuildIdentity(content) ? content : null;
}

// S and D from the newest receipt, or null when either is missing or garbled.
function latestAssetBuild(receipts) {
  let newest = null;
  for (const receipt of Object.values(receipts || {})) {
    if (receipt && Number.isFinite(receipt.receivedAt) && (!newest || receipt.receivedAt > newest.receivedAt)) {
      newest = receipt;
    }
  }
  const build = newest && newest.assetBuild;
  if (!build || !isAssetBuildIdentity(build.started) || !isAssetBuildIdentity(build.onDisk)) return null;
  return Object.freeze({ started: build.started, onDisk: build.onDisk, receivedAt: newest.receivedAt });
}

function assetSkewVerdict(pageBuild, served) {
  if (!isAssetBuildIdentity(pageBuild) || !served) return ASSET_SKEW_VERDICTS.unknown;
  if (served.started !== served.onDisk) return ASSET_SKEW_VERDICTS.staleServer;
  if (pageBuild !== served.started) return ASSET_SKEW_VERDICTS.stalePage;
  return ASSET_SKEW_VERDICTS.inStep;
}

function assetSkewState(kind, pageDiffersFromServer) {
  const reloadReachesServerBuild = pageDiffersFromServer && kind !== ASSET_SKEW_STATES.staleServer;
  return Object.freeze({ kind, pageDiffersFromServer, reloadReachesServerBuild });
}

// Whether this page may reload now. True only once the stamp is written, so a
// page that cannot record its reload never gets one.
function assetSkewClaimReload(storage, now, guardMs) {
  if (!storage) return false;
  let raw;
  try {
    raw = storage.getItem(ASSET_SKEW_GUARD_KEY);
  } catch (e) {
    return false;
  }
  const stampedAt = raw === null || raw === '' ? NaN : Number(raw);
  if (Number.isFinite(stampedAt) && now - stampedAt < guardMs) return false;
  try {
    storage.setItem(ASSET_SKEW_GUARD_KEY, String(now));
  } catch (e) {
    return false;
  }
  return true;
}

// One watcher per page. observe(receipts) takes DF_DATA.__receipt once per poll
// set and returns the page's state: {kind, pageDiffersFromServer,
// reloadReachesServerBuild}. It calls `reload` at most once, ever.
function createAssetSkewWatcher({ pageBuild, storage, now, reload, guardMs = ASSET_SKEW_RELOAD_GUARD_MS }) {
  let confirmation = null;
  let latched = null;
  let reloadRequested = false;

  function confirmStalePage(served) {
    if (confirmation && confirmation.target === served.started) {
      if (confirmation.receivedAt !== served.receivedAt) {
        confirmation = { target: served.started, receivedAt: served.receivedAt, count: confirmation.count + 1 };
      }
    } else {
      confirmation = { target: served.started, receivedAt: served.receivedAt, count: 1 };
    }
    return confirmation.count >= ASSET_SKEW_CONFIRMATIONS;
  }

  function observe(receipts) {
    const served = latestAssetBuild(receipts);
    const verdict = assetSkewVerdict(pageBuild, served);
    if (verdict !== ASSET_SKEW_VERDICTS.stalePage) {
      confirmation = null;
      latched = null;
      return assetSkewState(verdict, verdict === ASSET_SKEW_VERDICTS.staleServer && pageBuild !== served.started);
    }
    if (latched) return latched;
    if (!confirmStalePage(served)) return assetSkewState(ASSET_SKEW_STATES.confirming, true);
    if (!reloadRequested && assetSkewClaimReload(storage, now(), guardMs)) {
      reloadRequested = true;
      latched = assetSkewState(ASSET_SKEW_STATES.reloading, true);
      reload();
    } else {
      latched = assetSkewState(ASSET_SKEW_STATES.reloadGuarded, true);
    }
    return latched;
  }

  return { observe };
}

// The banner for a state, as {text, offersReload}, or null when there is none.
function assetSkewNotice(state) {
  if (!state) return null;
  if (state.kind === ASSET_SKEW_STATES.staleServer) {
    return state.pageDiffersFromServer === true
      ? ASSET_SKEW_STALE_SERVER_NOTICES.pageDiffersFromServer
      : ASSET_SKEW_STALE_SERVER_NOTICES.pageMatchesServer;
  }
  return Object.hasOwn(ASSET_SKEW_NOTICES, state.kind) ? ASSET_SKEW_NOTICES[state.kind] : null;
}

// The browser's sessionStorage, or null when there is none. Merely reading
// window.sessionStorage throws a SecurityError where storage is disabled.
function assetSkewGuardStorage() {
  try {
    return (typeof window !== 'undefined' && window.sessionStorage) || null;
  } catch (e) {
    return null;
  }
}

const ASSET_SKEW_API = {
  ASSET_SKEW_STATES,
  ASSET_SKEW_RELOAD_GUARD_MS,
  ASSET_SKEW_GUARD_KEY,
  pageAssetBuild,
  latestAssetBuild,
  createAssetSkewWatcher,
  assetSkewNotice,
  assetSkewGuardStorage,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = ASSET_SKEW_API;
}
if (typeof window !== 'undefined') {
  window.DF_ASSET_SKEW = ASSET_SKEW_API;
}
