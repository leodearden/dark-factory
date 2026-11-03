// The value a call throws, for suites that describe a real throw rather than a
// hand-built one.
//
// A helper module, not a test: the `*.test.mjs` glob skips it.

export function caughtFrom(fn) {
  try {
    fn();
  } catch (thrown) {
    return thrown;
  }
  throw new Error('expected the call to throw');
}
