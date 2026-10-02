// Behavioural tests for the GENERATED task_vocab.js — the SPA-side half of
// that artifact's coverage. Run via `node --test` (see
// dashboard/tests/test_graph_layout_js.py for the pytest wrapper, which
// globs `**/*.test.mjs` under this directory — no wrapper change needed).
//
// WHY THIS EXISTS BESIDE THE BYTE-EQUALITY PARITY TEST. That test
// (tests/scripts/test_dashboard_task_vocab.py) compares generated output
// against generated output, so it cannot see a RENDERING defect — a payload
// that is faithfully rendered into JS the browser reads differently, or not
// at all. This file loads the shipped file the way the browser does and
// asserts on the object that comes out.
//
// task_vocab.js has no package.json in the repo, so it resolves as CommonJS
// (`module.exports = <object>`). Node's cjs-module-lexer cannot statically
// detect named exports assigned from a variable, so we default-import and
// destructure (mirrors recon_status.test.mjs).
import { test } from 'node:test';
import assert from 'node:assert/strict';

import taskVocab from '../../src/dashboard/static/redux/task_vocab.js';

const { MEMBERS, VIEWS, SUB_VIEWS, TONES } = taskVocab;

test('MEMBERS lists nine distinct statuses', () => {
  assert.equal(MEMBERS.length, 9);
  assert.equal(new Set(MEMBERS).size, 9);
});

test('the three views partition MEMBERS', () => {
  const views = [VIEWS.in_flight, VIEWS.backlog, VIEWS.terminal];
  assert.deepEqual(Object.keys(VIEWS).sort(), ['backlog', 'in_flight', 'terminal']);

  for (let i = 0; i < views.length; i += 1) {
    for (let j = i + 1; j < views.length; j += 1) {
      const shared = views[i].filter((member) => views[j].includes(member));
      assert.deepEqual(shared, [], `views ${i} and ${j} overlap on ${shared}`);
    }
  }

  const union = views.flat();
  assert.equal(union.length, MEMBERS.length);
  assert.deepEqual(union.slice().sort(), MEMBERS.slice().sort());
});

test('SUB_VIEWS.running is a subset of VIEWS.in_flight', () => {
  assert.ok(SUB_VIEWS.running.length > 0);
  for (const member of SUB_VIEWS.running) {
    assert.ok(VIEWS.in_flight.includes(member), `${member} is not in_flight`);
  }
});

test('TONES carries exactly one non-empty entry per member', () => {
  assert.deepEqual(Object.keys(TONES).sort(), MEMBERS.slice().sort());
  for (const member of MEMBERS) {
    assert.equal(typeof TONES[member], 'string');
    assert.ok(TONES[member].length > 0, `${member} has an empty tone`);
  }
});
