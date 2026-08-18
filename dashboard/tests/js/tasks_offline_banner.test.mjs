// Module-contract tests for tasks_offline_banner.js — a plain-JS (no
// JSX/Babel) module holding the pure banner-copy DECISION for the Tasks tab
// (tab_tasks.jsx). Run via `node --test` (see
// dashboard/tests/test_graph_layout_js.py for the pytest wrapper that
// surfaces this suite in CI via its `**/*.test.mjs` glob — no wrapper change
// needed for this new file).
//
// tasks_offline_banner.js has no package.json in the repo, so it resolves as
// CommonJS (`module.exports = <object>`). Node's cjs-module-lexer cannot
// statically detect named exports assigned from a variable, so
// `import { tasksBannerNotices } from '...'` would come back undefined. We
// therefore default-import the module and destructure instead (mirrors
// task_status_counts.test.mjs / prd_grouping.test.mjs).
//
// WHY THIS IS A TESTED MODULE AND NOT JSX. The defect being closed is a copy
// decision: the Tasks tab rendered ONE global banner reading "fused-memory
// offline — task data unavailable" whenever ANY project's fetch failed, which
// on a nine-root config meant a total-outage claim printed directly above
// eight healthy projects' rows. Left in JSX, that decision is only greppable;
// as a pure function it is executable, so "1 of 9 must not say fused-memory
// offline" is a checked assertion rather than a comment.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import banner from '../../src/dashboard/static/redux/tasks_offline_banner.js';

const { tasksBannerNotices } = banner;

// The real-outage wording, preserved VERBATIM from the pre-change JSX
// (tab_tasks.jsx). The global case was never the bug — it was the only case,
// applied to states it did not describe. Rewriting it here would lose the one
// piece of copy that was already correct.
const GLOBAL_COPY = 'fused-memory offline — task data unavailable';

// The substring that must NOT appear on a partial failure: fused-memory is
// demonstrably reachable in that state (other projects' rows came back
// through it), so any wording that names it as offline is false.
const OFFLINE_CLAIM = 'fused-memory offline';

function kinds(notices) {
  return notices.map(n => n.kind);
}

test('module exposes tasksBannerNotices as a function', () => {
  assert.equal(typeof tasksBannerNotices, 'function');
});

test('nothing failed -> no banner at all', () => {
  const notices = tasksBannerNotices({
    offline: false,
    offlineProjects: [],
    degradedProjects: [],
    totalProjects: 9,
  });
  assert.deepEqual(notices, []);
});

test('global outage -> exactly one notice carrying the verbatim outage copy', () => {
  const notices = tasksBannerNotices({
    offline: true,
    offlineProjects: ['a', 'b', 'c'],
    degradedProjects: [],
    totalProjects: 3,
  });

  assert.equal(notices.length, 1);
  assert.equal(notices[0].kind, 'global');
  assert.ok(
    notices[0].text.includes(GLOBAL_COPY),
    `global notice must preserve the existing copy verbatim, got: ${notices[0].text}`
  );
});

test('1 of 9 offline -> a partial notice that does NOT claim fused-memory is offline', () => {
  // THE regression assertion. Today's render prints the global copy here.
  const notices = tasksBannerNotices({
    offline: false,
    offlineProjects: ['reify'],
    degradedProjects: [],
    totalProjects: 9,
  });

  assert.equal(notices.length, 1);
  assert.equal(notices[0].kind, 'partial');
  assert.ok(
    notices[0].text.includes('1 of 9'),
    `partial notice must name the count as "1 of 9", got: ${notices[0].text}`
  );
  assert.ok(
    notices[0].text.includes('reify'),
    `partial notice must name the failing project, got: ${notices[0].text}`
  );
  assert.ok(
    !notices[0].text.includes(OFFLINE_CLAIM),
    'a partial failure must not claim fused-memory is offline — 8 of 9 ' +
      `projects' rows came back through it: ${notices[0].text}`
  );
});

test('several of many offline -> the count reflects the real k of N', () => {
  const notices = tasksBannerNotices({
    offline: false,
    offlineProjects: ['a', 'b', 'c'],
    degradedProjects: [],
    totalProjects: 7,
  });

  assert.equal(kinds(notices), kinds(notices)); // shape sanity
  assert.equal(notices.length, 1);
  assert.equal(notices[0].kind, 'partial');
  assert.ok(
    notices[0].text.includes('3 of 7'),
    `expected "3 of 7", got: ${notices[0].text}`
  );
});

test('global and partial are mutually exclusive across a truth table', () => {
  const table = [
    { offline: false, offlineProjects: [],            totalProjects: 4 },
    { offline: false, offlineProjects: ['a'],         totalProjects: 4 },
    { offline: false, offlineProjects: ['a', 'b'],    totalProjects: 4 },
    { offline: true,  offlineProjects: ['a','b','c','d'], totalProjects: 4 },
    { offline: true,  offlineProjects: [],            totalProjects: 4 },
    { offline: true,  offlineProjects: ['a'],         totalProjects: 4 },
    { offline: false, offlineProjects: ['a','b','c','d'], totalProjects: 4 },
  ];

  for (const row of table) {
    for (const degradedProjects of [[], ['z']]) {
      const ks = kinds(tasksBannerNotices({ ...row, degradedProjects }));
      assert.ok(
        !(ks.includes('global') && ks.includes('partial')),
        `global and partial must never co-occur for ${JSON.stringify({ ...row, degradedProjects })}, got ${JSON.stringify(ks)}`
      );
    }
  }
});

test('degraded projects get their own notice, naming them and the partial view', () => {
  const notices = tasksBannerNotices({
    offline: false,
    offlineProjects: [],
    degradedProjects: ['reify', 'sidecar'],
    totalProjects: 5,
  });

  assert.deepEqual(kinds(notices), ['degraded']);
  assert.ok(notices[0].text.includes('reify'), notices[0].text);
  assert.ok(notices[0].text.includes('sidecar'), notices[0].text);
  assert.ok(
    /timed out|partial/i.test(notices[0].text),
    `degraded notice must say the data is partial / timed out, got: ${notices[0].text}`
  );
  assert.ok(
    !notices[0].text.includes(OFFLINE_CLAIM),
    'a budget expiry proved nothing unreachable — it must not claim an outage'
  );
});

test('degraded composes with partial: both notices appear', () => {
  const notices = tasksBannerNotices({
    offline: false,
    offlineProjects: ['a'],
    degradedProjects: ['b'],
    totalProjects: 6,
  });

  // Both facts are true simultaneously and neither implies the other, so the
  // operator must see both — one project failed, a different one timed out.
  assert.deepEqual(kinds(notices).sort(), ['degraded', 'partial']);
});

test('a global outage subsumes the degraded notice', () => {
  const notices = tasksBannerNotices({
    offline: true,
    offlineProjects: ['a', 'b'],
    degradedProjects: ['c'],
    totalProjects: 3,
  });

  // "task data unavailable" already covers everything a degraded marker could
  // add; stacking a second banner under a total outage is noise.
  assert.deepEqual(kinds(notices), ['global']);
});

test('DF_DATA pre-fetch defaults and missing inputs never throw', () => {
  // This module runs on window.DF_DATA's defaults during the very first
  // render, before any fetch resolves, so every input must be optional.
  assert.deepEqual(tasksBannerNotices({}), []);
  assert.deepEqual(tasksBannerNotices(), []);
  assert.deepEqual(tasksBannerNotices({ offline: false }), []);
  assert.deepEqual(
    tasksBannerNotices({ offline: false, totalProjects: 0 }),
    []
  );
  assert.deepEqual(
    tasksBannerNotices({ offline: false, offlineProjects: undefined, degradedProjects: undefined, totalProjects: 3 }),
    []
  );

  // A global outage with nothing else known is still a well-formed notice.
  const globalOnly = tasksBannerNotices({ offline: true });
  assert.deepEqual(kinds(globalOnly), ['global']);
  assert.equal(typeof globalOnly[0].text, 'string');
  assert.ok(globalOnly[0].text.length > 0);
});

test('offline projects with a zero/unknown total still produce a well-formed notice', () => {
  // totalProjects can lag offlineProjects by a render (they arrive from
  // different sources), so k > N and N === 0 must degrade to sane copy rather
  // than emitting "1 of 0" nonsense or throwing.
  for (const totalProjects of [0, undefined, 1]) {
    const notices = tasksBannerNotices({
      offline: false,
      offlineProjects: ['a'],
      degradedProjects: [],
      totalProjects,
    });
    assert.equal(notices.length, 1, JSON.stringify({ totalProjects, notices }));
    assert.ok(['partial', 'global'].includes(notices[0].kind));
    assert.equal(typeof notices[0].text, 'string');
    assert.ok(notices[0].text.length > 0);
  }
});

test('every notice is a {kind, text} pair with a known kind', () => {
  const notices = tasksBannerNotices({
    offline: false,
    offlineProjects: ['a'],
    degradedProjects: ['b'],
    totalProjects: 4,
  });

  for (const notice of notices) {
    assert.deepEqual(Object.keys(notice).sort(), ['kind', 'text']);
    assert.ok(['global', 'partial', 'degraded'].includes(notice.kind), notice.kind);
    assert.equal(typeof notice.text, 'string');
  }
});
