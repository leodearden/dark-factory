// The jsdom render harness's own contract (dom_harness.mjs). Its identity guard holds the installed
// react and react-dom to the packages the browser loads (their copy of index.html's build must hash
// to the tag's integrity); the harness renders with each package's development build.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {
  REDUX_DIR,
  REPO_ROOT,
  babelScriptSrcs,
  cdnScriptTag,
  classicScriptSrcs,
  readIndexHtml,
  sha384Digest,
} from './_served_bundle.mjs';
import { NPM_CI, REACT_BUILDS, buildDashboardWindow, seedDashboardData, toolchainPackageDir } from './dom_harness.mjs';
import { escalationAnalytics, receiptsFor } from './wire_fixtures.mjs';

const ROOT_PACKAGE_JSON = path.join(REPO_ROOT, 'package.json');
const EXACT_VERSION_RE = /^\d+\.\d+\.\d+$/;

function devDependencies() {
  return JSON.parse(fs.readFileSync(ROOT_PACKAGE_JSON, 'utf8')).devDependencies ?? {};
}

function reactInstallCommand() {
  const specs = REACT_BUILDS.map(pkg => `${pkg}@${cdnScriptTag(pkg).version}`);
  return `npm install --save-dev --save-exact ${specs.join(' ')}`;
}

for (const pkg of REACT_BUILDS) {
  test(`package.json pins ${pkg} to exactly the version index.html loads`, () => {
    assert.equal(
      devDependencies()[pkg],
      cdnScriptTag(pkg).version,
      `${ROOT_PACKAGE_JSON} must pin devDependencies.${pkg} to the bare version index.html loads from the CDN. ` +
        `Remedy: ${reactInstallCommand()}`,
    );
  });

  test(`the installed ${pkg} build is byte-identical to the one index.html loads`, () => {
    const { url, integrity, pathInPackage } = cdnScriptTag(pkg);
    const installed = path.join(toolchainPackageDir(pkg), pathInPackage);
    assert.equal(
      sha384Digest(installed),
      integrity,
      `${installed} is not the build index.html loads (${url}), so the harness would render with a ` +
        `different React than the browser. Remedy: ${reactInstallCommand()}, then \`${NPM_CI}\``,
    );
  });
}

test('package.json pins jsdom to a bare exact version', () => {
  assert.match(
    devDependencies().jsdom ?? 'absent',
    EXACT_VERSION_RE,
    `${ROOT_PACKAGE_JSON} must pin devDependencies.jsdom to a bare X.Y.Z. ` +
      'Remedy: npm install --save-dev --save-exact jsdom@<version>',
  );
});

// ── The window the render tests mount into ────────────────────────────────

function openWindow(t, options) {
  const harness = buildDashboardWindow(options);
  t.after(() => harness.close());
  return harness;
}

function everyLocalScriptInExecutionOrder() {
  const html = readIndexHtml();
  return [...classicScriptSrcs(html), ...babelScriptSrcs(html)];
}

test('the harness evaluates every local script in browser execution order: classic tags, then text/babel', t => {
  const { loaded, excluded } = openWindow(t);
  assert.deepEqual(
    loaded,
    everyLocalScriptInExecutionOrder().filter(src => !excluded.includes(src)),
    'Babel-standalone runs the text/babel tags after every classic script, each group in document order',
  );
});

test('the harness leaves out exactly app.jsx, the page bootstrap', t => {
  const { loaded, excluded } = openWindow(t);
  assert.deepEqual(excluded, ['app.jsx']);
  assert.deepEqual(
    new Set([...loaded, ...excluded]),
    new Set(everyLocalScriptInExecutionOrder()),
    'every local script index.html loads must be either evaluated or deliberately excluded',
  );
});

test('a script that fails to evaluate fails the build, naming the file (negative control)', () => {
  const readSource = src =>
    src === 'spark_path.js' ? 'throw new Error("boom");' : fs.readFileSync(path.join(REDUX_DIR, src), 'utf8');
  assert.throws(
    () => buildDashboardWindow({ readSource }),
    err => err.message.includes('spark_path.js') && err.message.includes('boom'),
    'a script that throws on load must fail the build loudly, never return a partially loaded window',
  );
});

test('the window exposes the helpers and tabs the render tests mount', t => {
  const { window } = openWindow(t);
  for (const name of ['DF_TASK_ROW_CELLS', 'DF_BURNDOWN_BANDS', 'DF_PINS_RECOVERY']) {
    assert.ok(window[name], `window.${name} is not set after the load`);
  }
  for (const tab of ['OrchTab', 'BurnTab', 'EscalationsTab', 'EscalationAnalyticsTab']) {
    assert.equal(typeof window.DF_TABS?.[tab], 'function', `window.DF_TABS.${tab} is not a component`);
  }
  assert.equal(typeof window.DF_TASKS?.TasksTab, 'function', 'window.DF_TASKS.TasksTab is not a component');
});

test('the window runs React at the version index.html loads', t => {
  const { window } = openWindow(t);
  assert.equal(window.React.version, cdnScriptTag('react').version);
  assert.equal(window.ReactDOM.version, cdnScriptTag('react-dom').version);
});

test("the window runs React's development build, so a broken React rule reaches consoleErrors", t => {
  const { window, consoleErrors, mount } = openWindow(t);
  const { React } = window;
  const items = withKeys => ['a', 'b'].map(id => React.createElement('li', withKeys ? { key: id } : null, id));
  const Keyed = () => React.createElement('ul', null, items(true));
  const Keyless = () => React.createElement('ul', null, items(false));

  mount(Keyed);
  assert.deepEqual(consoleErrors, [], 'a keyed list must render silently, or the check below proves nothing');

  mount(Keyless);
  assert.ok(
    consoleErrors.some(entry => entry.includes('unique "key" prop')),
    'a keyless list rendered without React\'s key warning. The browser loads the production build, which is ' +
      "silent about key, prop and hook misuse, so jsx_render.test.mjs's consoleErrors assertion is the one place " +
      'those warnings still fail a change, and only the development build supports the synchronous act() that ' +
      'mount() renders through.',
  );
});

test('the load writes nothing to console.error', t => {
  const { consoleErrors } = openWindow(t);
  assert.deepEqual(consoleErrors, []);
});

test('consecutive windows are isolated from each other', t => {
  const first = openWindow(t);
  first.window.DF_DATA.PROJECTS.push({ id: 'leaked-from-first-window', name: 'leak' });
  const second = openWindow(t);
  assert.ok(
    !second.window.DF_DATA.PROJECTS.some(p => p.id === 'leaked-from-first-window'),
    "a mutation to one window's DF_DATA is visible in the next",
  );
});

function liveTimerCount() {
  return process.getActiveResourcesInfo().filter(kind => kind === 'Timeout').length;
}

test('close() leaves no live timer behind, though data.js starts polling on load', () => {
  const before = liveTimerCount();
  const harness = buildDashboardWindow();
  try {
    assert.ok(
      liveTimerCount() > before,
      'data.js polling armed no timer in the open window, so the check below would prove nothing',
    );
  } finally {
    harness.close();
  }
  assert.equal(liveTimerCount(), before, 'a timer outlived close(); it would hold `node --test` open');
});

// ── Seeding lands only what a poll could land ─────────────────────────────

test('seeding refuses a key that no endpoint data.js polls serves', t => {
  const { window } = openWindow(t);
  assert.throws(() => seedDashboardData(window, { NOT_A_POLLED_KEY: 1 }), /polls no endpoint that serves NOT_A_POLLED_KEY/);
});

test("seeding refuses a key without its endpoint's receipt, and lands it with one", t => {
  const { window } = openWindow(t);
  const patch = { ESCALATION_ANALYTICS: escalationAnalytics([]) };
  assert.throws(() => seedDashboardData(window, patch), /records a receipt for \/api\/v2\/dashboard\/escalation-analytics/);
  assert.ok(!window.DF_DATA.__loaded.ESCALATION_ANALYTICS, 'a refused seed must land nothing');

  seedDashboardData(window, { ...patch, __receipt: receiptsFor(window.DF_DATA_LOADER, ['ESCALATION_ANALYTICS']) });
  assert.equal(window.DF_DATA.__loaded.ESCALATION_ANALYTICS, true);
});

// ── A missing or stale toolchain fails with its remedy, never a skip ──────

function scratchRoot(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'dom-harness-root-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  fs.copyFileSync(path.join(REPO_ROOT, 'package-lock.json'), path.join(root, 'package-lock.json'));
  return root;
}

test('a checkout with no node_modules fails with the npm ci remedy, and its sandbox-safe form', t => {
  const root = scratchRoot(t);
  assert.throws(
    () => toolchainPackageDir('jsdom', root),
    err =>
      err.message.includes('jsdom') &&
      err.message.includes(NPM_CI) &&
      err.message.includes(`npm_config_cache="$(mktemp -d)" ${NPM_CI}`) &&
      err.message.includes(root) &&
      !/stale/.test(err.message),
  );
});

test('a node_modules installed from an older package-lock.json is named as stale', t => {
  const root = scratchRoot(t);
  const pyrightOnly = { name: 'dark-factory', lockfileVersion: 3, packages: { 'node_modules/pyright': { version: '1.1.408' } } };
  fs.mkdirSync(path.join(root, 'node_modules', 'pyright'), { recursive: true });
  fs.writeFileSync(path.join(root, 'node_modules', '.package-lock.json'), JSON.stringify(pyrightOnly));
  const locked = JSON.parse(fs.readFileSync(path.join(root, 'package-lock.json'), 'utf8')).packages['node_modules/jsdom'].version;
  assert.throws(
    () => toolchainPackageDir('jsdom', root),
    err => /stale/.test(err.message) && err.message.includes(locked) && err.message.includes(NPM_CI),
  );
});
