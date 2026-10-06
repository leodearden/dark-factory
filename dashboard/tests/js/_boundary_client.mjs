// The real client, loaded as index.html loads it, for the boundary suite.
//
// A helper module, not a test: the `*.test.mjs` glob skips it. The bodies it
// applies are never hand-built: test_boundary_js.py drives the real routes and
// writes each served body into DF_BOUNDARY_PAYLOADS, and payload() reads them.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';

import { REDUX_DIR, classicScriptSrcs, readIndexHtml } from './_served_bundle.mjs';

const OWNER = 'dashboard/tests/test_boundary_js.py';

export function payload(name) {
  const dir = process.env.DF_BOUNDARY_PAYLOADS;
  if (!dir) {
    throw new Error(
      `DF_BOUNDARY_PAYLOADS is unset: the boundary suite applies the bodies the real routes serve, ` +
        `which only ${OWNER} builds. Run it through that file.`,
    );
  }
  const file = path.join(dir, `${name}.json`);
  if (!fs.existsSync(file)) {
    throw new Error(`no served body ${file}: ${OWNER} (_boundary_payloads.build_all) builds no scenario '${name}'`);
  }
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

// Every classic script index.html loads, in its order, into ONE fresh window
// shim. With no `document` defined data.js stays inert: it never starts
// polling (data_poll.test.mjs::loadDataJs). Each call replaces the global
// window, so one client is live at a time.
export function loadClient() {
  const win = { dispatchEvent() {} };
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  const files = classicScriptSrcs(readIndexHtml()).map(src => path.join(REDUX_DIR, src));
  for (const file of files) delete require.cache[require.resolve(file)];
  for (const file of files) require(file);
  return { window: win, loader: win.DF_DATA_LOADER };
}

// Lands `body` as the response to `path` through data.js's real refreshOne, at
// the instant `receivedAt`, and resolves to its REFRESH_OUTCOMES value. `path`
// is the endpointsFor(win) key, so a windowed endpoint carries its
// `?window=` and names the same `win`.
export function applyServed(client, path, body, { receivedAt = Date.now(), win = '24h', url = path } = {}) {
  if (globalThis.window !== client.window) {
    throw new Error('a later loadClient() replaced this client\'s window; data.js would write into that one');
  }
  const keySpecs = client.loader.endpointsFor(win)[path];
  if (keySpecs === undefined) throw new Error(`data.js registers no endpoint ${path} at window ${win}`);
  return client.loader.refreshOne(url, keySpecs, client.loader.createPollState(), {
    fetchImpl: () => Promise.resolve({ ok: true, json: async () => body }),
    now: () => receivedAt,
    setTimeoutImpl: () => 0,
    clearTimeoutImpl: () => {},
  });
}
