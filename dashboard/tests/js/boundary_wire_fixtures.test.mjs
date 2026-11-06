// The render tests' hand-built DF_DATA payloads (wire_fixtures.mjs), checked
// against the bodies the real routes serve (test_boundary_js.py builds them).
// A builder may leave out a key a route serves, but never emit one it does
// not, so a server-side rename fails here rather than leaving the render
// tests green on a shape no route serves. Where a served shape is a fixed
// contract, the builder must match it exactly.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { loadClient, payload } from './_boundary_client.mjs';
import {
  burndownBlock,
  escalationAnalytics,
  escalationRow,
  escalationsPayload,
  mergeQueueBlock,
  openEscalationItem,
  taskCensus,
  tasksSnapshotEntry,
  wireDatum,
} from './wire_fixtures.mjs';

function isRecord(x) {
  return typeof x === 'object' && x !== null;
}

// Every key `built` carries, at every depth, is a key `served` carries at the
// same place, holding an object exactly where `served` holds one. Lists are
// compared by their first elements; a built null stands for any value.
function assertBuiltKeysServed(built, served, where) {
  if (!isRecord(built)) {
    assert.ok(built === null || !isRecord(served), `${where} is built as ${built}, but served as an object`);
    return;
  }
  assert.ok(isRecord(served), `${where} is built as ${Array.isArray(built) ? 'a list' : 'an object'}, but served as ${served}`);
  assert.equal(Array.isArray(built), Array.isArray(served), `${where}: built and served disagree on being a list`);
  if (Array.isArray(built)) {
    if (built.length > 0 && served.length > 0) assertBuiltKeysServed(built[0], served[0], `${where}[0]`);
    return;
  }
  for (const key of Object.keys(built)) {
    assert.ok(key in served, `${where}.${key} is built, but the route serves only ${Object.keys(served).join(', ')}`);
    assertBuiltKeysServed(built[key], served[key], `${where}.${key}`);
  }
}

function assertSameKeys(built, served, where) {
  assert.deepEqual(Object.keys(built).sort(), Object.keys(served).sort(), `${where}: the builder's keys are not the served ones`);
}

test('a TASKS_SNAPSHOT entry is built with exactly the keys /tasks serves for one', () => {
  const vocab = loadClient().window.DF_TASK_VOCAB;
  const served = payload('census_fresh').TASKS_SNAPSHOT['dark-factory'];
  const built = tasksSnapshotEntry([], taskCensus({}, vocab));

  assertBuiltKeysServed(built, served, 'TASKS_SNAPSHOT[project]');
  assertSameKeys(built, served, 'TASKS_SNAPSHOT[project]');
  assertSameKeys(wireDatum(null), served.census, 'the census Datum');
  for (const part of ['counts', 'views', 'sub_views']) {
    assertSameKeys(built.census.value[part], served.census.value[part], `census.value.${part}`);
  }
});

test('a burndown block is built only with keys /burndown serves, in the aggregate and per project', () => {
  const vocab = loadClient().window.DF_TASK_VOCAB;
  const served = payload('burndown_ragged');
  const verdict = { alarm: true, cap: 4, peak: 6, breachCount: 1 };

  assertBuiltKeysServed(burndownBlock(vocab, 2, { ...verdict, projects: ['A'] }), served.BURNDOWN, 'BURNDOWN');
  for (const [project, block] of Object.entries(served.BURNDOWN_BY_PROJECT)) {
    assertBuiltKeysServed(burndownBlock(vocab, 2, verdict), block, `BURNDOWN_BY_PROJECT.${project}`);
  }
});

test('the escalation-analytics payload is built only with keys /escalation-analytics serves', () => {
  const served = payload('analytics_a').ESCALATION_ANALYTICS;
  const item = openEscalationItem({ id: 'esc-1-1', taskId: '1' });

  assertBuiltKeysServed(escalationAnalytics([{ project: 'p', openItems: [item] }]), served, 'ESCALATION_ANALYTICS');
  const servedItem = served.per_project[0].lifespan.open_items[0];
  assert.ok(!('pins_recovery' in servedItem), 'this check expects the served item to carry no pins annotation');
  assertSameKeys(item, servedItem, 'an open item with no pins annotation');
});

test('a MERGE_QUEUE block is built only with keys /merge-queue serves', () => {
  const served = payload('merge_7d').MERGE_QUEUE['dark-factory'];

  assertBuiltKeysServed(mergeQueueBlock({ bucketMinutes: 60 }), served, 'MERGE_QUEUE[project]');
});

test('an /escalations payload, a check-owner row included, is built only with keys /escalations serves', () => {
  const served = payload('escalations_check_owner').ESCALATIONS;
  const row = escalationRow({
    id: 'esc-3-1',
    taskId: '3',
    summary: 's',
    project: 'dark-factory',
    deliveredCheckOwner: { task_id: '4', check_name: 'c', sidecar: { path: 'p', on_main: 'declares' } },
  });

  assertBuiltKeysServed(escalationsPayload([{ project: 'dark-factory', rows: [row] }]), served, 'ESCALATIONS');
});

test("an /escalations payload's counts and views agree with the served ones for its one pending row", () => {
  const served = payload('escalations_check_owner').ESCALATIONS;
  const servedQueue = served.subsections.find(subsection => subsection.kind === 'orchestrator');
  const servedRow = servedQueue.escalations[0];
  const row = escalationRow({
    id: servedRow.id,
    taskId: servedRow.task_id,
    level: servedRow.level,
    status: servedRow.status,
    summary: servedRow.summary,
    project: 'dark-factory',
  });
  const built = escalationsPayload([{ project: 'dark-factory', rows: [row] }]);
  const viewValues = views => Object.fromEntries(Object.entries(views).map(([name, datum]) => [name, datum.value]));

  for (const [where, builtPart, servedPart] of [
    ['ESCALATIONS', built, served],
    ['the orchestrator subsection', built.subsections[0], servedQueue],
  ]) {
    assert.deepEqual(builtPart.summary, servedPart.summary, `${where}.summary`);
    assert.deepEqual(viewValues(builtPart.views), viewValues(servedPart.views), `${where}.views`);
  }
});
