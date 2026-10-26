// Stated-literal expectations and stub responses that more than one data.js
// node suite shares. A helper module, not a test: the `*.test.mjs` glob skips
// it.
//
// Every value is written out rather than read off data.js, so a suite pins the
// module against an expectation instead of against itself.

// Number of rows in endpointsFor(). Several tests assert that a cycle touched
// EVERY endpoint — "all of them" is the actual claim, and a literal is the only
// way to state it without deriving the expectation from the same map under
// test. Bump this whenever endpointsFor gains or loses a row; it is
// deliberately one constant rather than a literal repeated per test, because
// scattered copies is what went stale when the memory-evals endpoint was added.
export const EXPECTED_ENDPOINT_COUNT = 15;

// The two shared registry specs.
export const PLAIN_SPEC = Object.freeze({ kind: 'plain' });
export const DATUM_SPEC = Object.freeze({ kind: 'datum' });

export const CURATOR_PATH = '/api/v2/dashboard/curator';
export const COSTS_PATH = '/api/v2/dashboard/costs';
export const TASKS_PATH = '/api/v2/dashboard/tasks';

export const TERMINAL_PROJECT = 'dark-factory';

// Stated as a literal rather than read back from endpoint_staleness.js.
// data.js prefers window.DF_ENDPOINT_STALENESS's threshold when that module has
// loaded and falls back to its own literal otherwise;
// endpoint_staleness.test.mjs's "data.js's fallback threshold agrees with
// STALE_FAILURE_THRESHOLD" asserts the two agree.
export const STALE_FAILURE_THRESHOLD = 3;

export function okResponse(body) {
  return () => Promise.resolve({ ok: true, json: async () => body });
}
