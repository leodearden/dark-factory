// A browser page for the dashboard's .jsx, in node: each call builds a fresh
// jsdom window and loads index.html's local scripts into it the way the
// browser does, on the browser's exact React bytes and Babel build. Render
// tests mount one real tab into it and query the DOM it produces.
//
// A helper module, not a test: the `*.test.mjs` glob skips it.
// dom_harness.test.mjs holds its contract.
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createRequire } from 'node:module';

import {
  REDUX_DIR,
  REPO_ROOT,
  babelScriptSrcs,
  cdnScriptTag,
  classicScriptSrcs,
  compileAsScriptTag,
  readIndexHtml,
  sha384Digest,
} from './_served_bundle.mjs';

const NPM_CI = 'npm ci --no-audit --no-fund';

// app.jsx is the page bootstrap: at load it mounts the whole <App/> into
// #root, which would bring every tab's effects into every window. Render tests
// mount one tab each instead.
const EXCLUDED_SCRIPTS = ['app.jsx'];

const REACT_BUILDS = ['react', 'react-dom'];

function lockfileVersion(lockfile, pkg) {
  if (!fs.existsSync(lockfile)) return null;
  return JSON.parse(fs.readFileSync(lockfile, 'utf8')).packages?.[`node_modules/${pkg}`]?.version ?? null;
}

function missingToolchainMessage(pkg, root) {
  const nodeModules = path.join(root, 'node_modules');
  const sentences = [`${pkg} is not installed in ${nodeModules}.`];
  const installedRecord = path.join(nodeModules, '.package-lock.json');
  if (fs.existsSync(installedRecord)) {
    const installed = lockfileVersion(installedRecord, pkg);
    const locked = lockfileVersion(path.join(root, 'package-lock.json'), pkg);
    if (installed !== locked) {
      sentences.push(
        `That node_modules is stale: it records ${pkg} as ${installed ?? 'absent'}, but package-lock.json ` +
          `pins ${locked ?? 'nothing'}. A worktree that cold-verified before the lockfile changed keeps ` +
          'its old node_modules until npm ci is re-run.',
      );
    }
  }
  sentences.push(`Remedy: run \`${NPM_CI}\` in ${root}.`);
  return sentences.join(' ');
}

// The directory `pkg` is installed in, under exactly `root`'s node_modules.
// Not require's resolution, which walks up the tree: a task worktree sits
// inside the main checkout, and would silently borrow its node_modules.
export function toolchainPackageDir(pkg, root = REPO_ROOT) {
  const dir = path.join(root, 'node_modules', pkg);
  if (!fs.existsSync(path.join(dir, 'package.json'))) throw new Error(missingToolchainMessage(pkg, root));
  return dir;
}

let jsdomModule = null;

function loadJsdom() {
  jsdomModule ??= createRequire(import.meta.url)(toolchainPackageDir('jsdom'));
  return jsdomModule;
}

// The installed copy of the build index.html loads for `pkg`, compiled once,
// and only after its sha384 equals the tag's integrity attribute.
const reactScripts = new Map();

function reactScript(pkg) {
  if (!reactScripts.has(pkg)) {
    const { url, pathInPackage, integrity } = cdnScriptTag(pkg);
    const file = path.join(toolchainPackageDir(pkg), pathInPackage);
    const digest = sha384Digest(file);
    if (digest !== integrity) {
      throw new Error(
        `${file} (${digest}) is not the build index.html loads (${url}, ${integrity}), so the harness ` +
          `would render with a different React than the browser. Remedy: run \`${NPM_CI}\` in ${REPO_ROOT}.`,
      );
    }
    reactScripts.set(pkg, new vm.Script(fs.readFileSync(file, 'utf8'), { filename: url }));
  }
  return reactScripts.get(pkg);
}

// Compiled once per (file, source text), so an injected mutant compiles fresh.
// A vm.Script is not bound to a context: each window runs the same one.
const servedScripts = new Map();

function servedScript(src, source, toScriptCode) {
  const key = `${src}\0${source}`;
  if (!servedScripts.has(key)) servedScripts.set(key, new vm.Script(toScriptCode(source, src), { filename: src }));
  return servedScripts.get(key);
}

function readServedSource(src) {
  return fs.readFileSync(path.join(REDUX_DIR, src), 'utf8');
}

// Babel-standalone runs the text/babel tags after every classic script, so
// that is the order here, each group in document order.
function scriptsInExecutionOrder(readSource) {
  const html = readIndexHtml();
  const classic = classicScriptSrcs(html).map(src => ({ src, script: servedScript(src, readSource(src), code => code) }));
  const babelSrcs = babelScriptSrcs(html);
  const babel = babelSrcs
    .filter(src => !EXCLUDED_SCRIPTS.includes(src))
    .map(src => ({ src, script: servedScript(src, readSource(src), compileAsScriptTag) }));
  return { scripts: [...classic, ...babel], excluded: babelSrcs.filter(src => EXCLUDED_SCRIPTS.includes(src)) };
}

// The browser APIs the dashboard calls that jsdom lacks, and index.html's two
// CDN globals that are not React. Nothing else is stubbed: the components run
// for real.
function installBrowserGaps(window) {
  // charts.jsx and tab_tasks.jsx construct one in layout effects.
  window.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  // data.js starts polling as soon as `document` exists. A fetch that never
  // settles means the poll never writes over the data a test seeds.
  window.fetch = () => new Promise(() => {});
  // No render decision under test renders markdown.
  window.marked = { parse: text => String(text ?? '') };
  window.DOMPurify = { sanitize: html => String(html ?? '') };
  // Declares that renders go through act(). Only the synchronous form works:
  // async act needs MessageChannel, which jsdom lacks.
  window.IS_REACT_ACT_ENVIRONMENT = true;
}

function capturedConsoleErrors(VirtualConsole) {
  const errors = [];
  const virtualConsole = new VirtualConsole();
  virtualConsole.on('error', (...args) => errors.push(args.map(String).join(' ')));
  virtualConsole.on('jsdomError', err => errors.push(String(err.type === 'unhandled-exception' ? err.cause?.stack : err.message)));
  return { errors, virtualConsole };
}

// `readSource(src)` supplies each local script's text; it defaults to the
// served file. Returns the window and what was loaded, plus:
//   consoleErrors — everything the window wrote to console.error, live;
//   mount(Component, props) — renders into a fresh container under act();
//   close() — unmounts every root, then closes the window and its timers.
// Any script that fails to evaluate fails the build.
export function buildDashboardWindow({ readSource = readServedSource } = {}) {
  const { JSDOM, VirtualConsole } = loadJsdom();
  const react = REACT_BUILDS.map(pkg => ({ src: pkg, script: reactScript(pkg) }));
  const { scripts, excluded } = scriptsInExecutionOrder(readSource);

  const { errors: consoleErrors, virtualConsole } = capturedConsoleErrors(VirtualConsole);
  const dom = new JSDOM('<!doctype html><html><body></body></html>', {
    runScripts: 'outside-only',
    pretendToBeVisual: true,
    url: 'http://localhost/',
    virtualConsole,
  });
  const { window } = dom;

  // vm.Script in jsdom's own context, NOT window.eval. Classic scripts share
  // ONE global lexical scope in the browser (a top-level const in one is
  // visible to, and collides with, every later one); an indirect eval would
  // scope each script's const to that eval instead.
  try {
    installBrowserGaps(window);
    const context = dom.getInternalVMContext();
    for (const { src, script } of [...react, ...scripts]) {
      try {
        script.runInContext(context);
      } catch (err) {
        throw new Error(`${src}: ${err}`);
      }
    }
  } catch (err) {
    window.close();
    throw err;
  }

  const mounted = new Set();

  function mount(Component, props = {}) {
    const { React, ReactDOM, document } = window;
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = ReactDOM.createRoot(container);
    const render = nextProps => React.act(() => root.render(React.createElement(Component, nextProps)));
    const handle = {
      container,
      rerender: render,
      unmount() {
        mounted.delete(handle);
        React.act(() => root.unmount());
        container.remove();
      },
    };
    mounted.add(handle);
    render(props);
    return handle;
  }

  let closed = false;

  function close() {
    if (closed) return;
    closed = true;
    try {
      for (const handle of [...mounted]) handle.unmount();
    } finally {
      window.close();
    }
  }

  return { window, loaded: scripts.map(({ src }) => src), excluded, consoleErrors, mount, close };
}

// Lands `patch` in the window's DF_DATA as a poll would: each key through
// data.js's own applyKey, which fills PROJECTS and AGENTS in place (shell.jsx
// captured those arrays at load) and marks the key __loaded. `__receipt`
// entries are merged into data.js's map. Call it after the load, because
// data.js creates DF_DATA itself.
export function seedDashboardData(window, patch) {
  const { __receipt: receipts = {}, ...keys } = patch;
  Object.assign(window.DF_DATA.__receipt, receipts);
  for (const [key, value] of Object.entries(keys)) window.DF_DATA_LOADER.applyKey(key, value);
}
