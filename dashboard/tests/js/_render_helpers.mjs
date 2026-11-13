// Plumbing shared by the jsdom render suites (jsx_render.test.mjs,
// jsx_render_quality.test.mjs): opening a dashboard window that must stay
// console.error-free, clicking, letting React's async act() run, and
// answering on-demand fetches by url.
import assert from 'node:assert/strict';

import { buildDashboardWindow } from './dom_harness.mjs';

// React reports a broken render (a missing key, an update outside act(), a bad
// prop) on console.error and renders on, so every test also asserts that
// nothing reached it, through the unmount.
export function openWindow(t) {
  const harness = buildDashboardWindow();
  t.after(() => {
    harness.close();
    assert.deepEqual(harness.consoleErrors, [], 'the window wrote to console.error');
  });
  return harness;
}

export function click(window, element) {
  window.React.act(() => element.dispatchEvent(new window.MouseEvent('click', { bubbles: true })));
}

// React 18's async act() ends with one MessageChannel task, which jsdom lacks;
// a window timer is the same macrotask hop.
export function allowAsyncAct(window) {
  window.MessageChannel = class {
    constructor() {
      this.port1 = { onmessage: null };
      this.port2 = { postMessage: () => window.setTimeout(() => this.port1.onmessage(), 0) };
    }
  };
}

export async function settle(window) {
  await window.React.act(async () => {
    await new Promise(resolve => window.setTimeout(resolve, 0));
  });
}

// Answers each url named in `answers` with its value (a fetch Response or a
// promise of one) and leaves every other fetch pending, as the harness's own
// stub does. Returns every url asked for, in order.
export function stubFetch(window, answers) {
  const asked = [];
  window.fetch = url => {
    asked.push(url);
    if (!Object.prototype.hasOwnProperty.call(answers, url)) return new Promise(() => {});
    return Promise.resolve(answers[url]);
  };
  return asked;
}
