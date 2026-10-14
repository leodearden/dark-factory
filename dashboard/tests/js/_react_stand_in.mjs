// Enough of React's hook contract for node to drive a hook factory and observe
// what its hooks hold across renders. The .jsx that call those hooks are
// `type="text/babel"` behind CDN Babel and cannot run under node, so a factory
// that takes React from its caller is the executable seam
// (task_snapshot.js::createTerminalWindowHooks).
//
// A helper module, not a test: the `*.test.mjs` glob skips it.
//
// State, refs and effect dependencies live in per-call slots. A setter whose
// value is unchanged schedules nothing. An effect runs after its render only
// when a dependency changed, and first runs the cleanup its previous run
// returned, so a cleanup that cancels work is observable. render() repeats
// while an effect set state, as React's commit loop does, and returns the last
// render's result. A setter called between renders, by a settled promise, is
// seen by the next render().
export function reactStandIn() {
  const slots = [];
  let cursor = 0;
  let dirty = false;
  let due = [];
  function useState(initial) {
    const at = cursor++;
    if (!(at in slots)) slots[at] = typeof initial === 'function' ? initial() : initial;
    const setState = next => {
      const value = typeof next === 'function' ? next(slots[at]) : next;
      if (!Object.is(value, slots[at])) {
        slots[at] = value;
        dirty = true;
      }
    };
    return [slots[at], setState];
  }
  function useEffect(effect, deps) {
    const at = cursor++;
    const previous = slots[at];
    if (previous && deps.every((dep, i) => Object.is(dep, previous.deps[i]))) return;
    const entry = { deps, cleanup: undefined };
    slots[at] = entry;
    due.push(() => {
      if (previous && typeof previous.cleanup === 'function') previous.cleanup();
      entry.cleanup = effect();
    });
  }
  function useRef(initial) {
    const at = cursor++;
    if (!(at in slots)) slots[at] = { current: initial };
    return slots[at];
  }
  function render(component) {
    for (let pass = 0; pass < 10; pass += 1) {
      cursor = 0;
      dirty = false;
      due = [];
      const result = component();
      for (const effect of due) effect();
      if (!dirty) return result;
    }
    throw new Error('the hook never settled');
  }
  return { useState, useEffect, useRef, render };
}
