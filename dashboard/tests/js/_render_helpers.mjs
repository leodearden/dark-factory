// Plumbing shared by the jsdom render suites (jsx_render.test.mjs,
// jsx_render_quality.test.mjs, render_app_shell.test.mjs): opening a dashboard
// window that must stay console.error-free, or one whose contained throws the
// test checks itself; clicking, and switching the page's tab from its rail;
// reading render-failure notices; letting React's async act() run; and
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

// React's development build reports every throw a boundary catches on
// console.error: jsdom's report of the replayed throw (its stack, or
// 'undefined' for a non-Error), then a note naming the boundary that caught
// it. So a test of a contained throw cannot assert an empty console. It
// asserts that every report is of a throw RenderFailureBoundary caught, and
// nothing else.
const CAUGHT_NOTE = 'using the error boundary you provided, ';
const UNCAUGHT_NOTE = 'Consider adding an error boundary';

export function assertReportsOnlyContainedThrows(consoleErrors) {
  const caughtNotes = consoleErrors.filter(entry => entry.includes(CAUGHT_NOTE));
  assert.ok(caughtNotes.length > 0, `no throw was reported as caught by a boundary: ${JSON.stringify(consoleErrors)}`);
  for (const note of caughtNotes) {
    assert.ok(note.includes(`${CAUGHT_NOTE}RenderFailureBoundary.`), `a boundary other than RenderFailureBoundary caught: ${note}`);
  }
  for (const entry of consoleErrors) {
    assert.ok(!entry.includes(UNCAUGHT_NOTE), `a throw escaped every boundary: ${entry}`);
    assert.ok(!entry.startsWith('Warning:'), `React warned beside the contained throw: ${entry}`);
  }
}

// Like openWindow, but the test asserts the window's reports itself.
export function openContainingWindow(t, options) {
  const harness = buildDashboardWindow(options);
  t.after(() => harness.close());
  return harness;
}

export const NOTICE_SELECTOR = '[data-testid$="-render-failure-banner"]';

export function failureNotices(root) {
  return [...root.querySelectorAll(NOTICE_SELECTOR)];
}

export function noticeLines(notice) {
  return [...notice.children].map(line => line.textContent);
}

export function railButton(root, label) {
  const button = [...root.querySelectorAll('.rail-nav button')].find(candidate =>
    [...candidate.children].some(part => part.textContent === label),
  );
  assert.ok(button, `the rail has no "${label}" button`);
  return button;
}

// `page` is a harness plus the #root its mountPage() returned, as `root`.
export function openTab(page, label) {
  click(page.window, railButton(page.root, label));
}
