// Prints the delay, in ms, after which data.js's poll aborts the fetch of one
// endpoint: `node _abort_delay_probe.mjs <endpoint path>`. The server-side
// budget tests read their ceiling through it (dashboard/tests/_browser_abort.py).
//
// It drives refreshDFData, the DEFAULT_POLL_DEPS merge production takes, not
// refreshOne alone: a refreshOne given partial deps once hid a deadline only
// the production merge armed (data.js's comment above DEFAULT_POLL_DEPS). Each
// armed timer is paired with the fetch its firing aborts through the real
// AbortController, so the pairing relies on neither call order nor the
// timeout error's text.
//
// Exits 2, listing the polled paths on stderr, when data.js polls no such
// path, and 1 when that path's timer aborts nothing.
//
// A helper module, not a test: the `*.test.mjs` glob skips it.
import { drain, loadDataJs, quietly } from './_data_loader.mjs';

const [endpointPath] = process.argv.slice(2);

const { api } = loadDataJs();
const fetches = [];
const timers = [];
const deps = {
  fetchImpl: (url, init) => {
    fetches.push({ url, signal: init.signal });
    return new Promise(() => {});
  },
  setTimeoutImpl: (fire, ms) => {
    timers.push({ fire, ms });
    return timers.length;
  },
  clearTimeoutImpl: () => {},
};
api.refreshDFData(undefined, { state: api.createPollState(), jitterMaxMs: 0, deps });

// The armed delay of each fetch, by its url.
const abortDelays = new Map();
await quietly(async () => {
  for (const { fire, ms } of timers) {
    const live = fetches.filter(({ signal }) => !signal.aborted);
    fire();
    for (const { url } of live.filter(({ signal }) => signal.aborted)) abortDelays.set(url, ms);
  }
  await drain();
});

const fetched = fetches.find(({ url }) => api.pollKey(url) === endpointPath);
if (!fetched) {
  const polled = fetches.map(({ url }) => api.pollKey(url));
  process.stderr.write(`data.js polls no endpoint ${endpointPath}; it polls: ${polled.join(', ')}\n`);
  process.exitCode = 2;
} else if (!abortDelays.has(fetched.url)) {
  process.stderr.write(`no timer data.js armed aborts the fetch of ${fetched.url}\n`);
  process.exitCode = 1;
} else {
  process.stdout.write(`${abortDelays.get(fetched.url)}\n`);
}
