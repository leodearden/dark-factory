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
  qualityBody,
  qualityComparison,
  qualityMarker,
  qualityMarkers,
  qualityMetric,
  qualityModuleRow,
  qualityPayload,
  schedulerLockRow,
  schedulerModule,
  schedulerSnapshot,
  taskCensus,
  taskRow,
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

test('a TASKS_SNAPSHOT row is built only with keys /tasks serves on one', () => {
  const served = payload('census_fresh').TASKS_SNAPSHOT['dark-factory'].rows.value;
  assert.ok(served.length > 0, 'the census_fresh body serves no task rows to check the builder against');

  assertBuiltKeysServed(taskRow(), served[0], 'TASKS_SNAPSHOT[project].rows.value[0]');
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

test('a SCHEDULER snapshot, a lock row and a module included, is built only with keys /scheduler serves', () => {
  const served = payload('scheduler_offline').SCHEDULER;
  const built = schedulerSnapshot({
    offlineProjects: ['P'],
    rows: [schedulerLockRow({ project: 'H', taskId: '1', lockSet: ['src/m'] })],
    modules: [schedulerModule({ project: 'H', path: 'src/m', holder: '1', holderProject: 'H', parkedBy: '2', parkedOwnerLive: true })],
  });

  assertBuiltKeysServed(built, served, 'SCHEDULER');
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

test('a QUALITY series, a measured and a no-producer metric included, is built only with keys /quality serves', () => {
  const served = payload('quality_fresh').QUALITY;
  const measured = qualityMetric({
    kind: 'rate',
    points: [0, 1, 0],
    headlineForm: 'rate',
    ewma: { line: [0.2, 0.25, 0.22], ucl: [0.4, 0.4, 0.4] },
  });
  const unproduced = qualityMetric({ kind: 'count', points: [null, null, null], measuredSince: null });
  const built = qualityPayload({
    axis: [{}, { git_first_parent: false }, { late_claim: true }],
    metrics: { rejected: measured, reverts: unproduced },
  });
  const servedMetrics = served.value.metrics;

  assertSameKeys(wireDatum(null), served, 'the QUALITY Datum');
  assertBuiltKeysServed(built, served.value, 'QUALITY.value');
  assertSameKeys(built, served.value, 'QUALITY.value');
  assertSameKeys(built.axis[0], served.value.axis[0], 'QUALITY.value.axis[0]');
  for (const [id, metric] of Object.entries(built.metrics)) {
    assertSameKeys(metric, servedMetrics[id], `QUALITY.value.metrics.${id}`);
    assertSameKeys(metric.rule, servedMetrics[id].rule, `QUALITY.value.metrics.${id}.rule`);
    assertSameKeys(metric.headline, servedMetrics[id].headline, `QUALITY.value.metrics.${id}.headline`);
  }
  assertSameKeys(measured.headline.value, servedMetrics.rejected.headline.value, 'a measured headline value');
  assert.equal(servedMetrics.reverts.rule.measured_since, null, 'this check expects reverts to have no producer');
  assert.equal(unproduced.headline.state, servedMetrics.reverts.headline.state);
});

test('a whole /quality body, its cost model and a before/after comparison are built with exactly the keys /quality serves', () => {
  const served = payload('quality_alarm_explained');
  const [servedComparison] = served.QUALITY.value.before_after;
  const comparison = qualityComparison({ markerId: 'routing:reviewer_comprehensive:1', metric: 'rejected', state: 'not_yet' });
  const body = qualityBody({
    quality: qualityPayload({ axis: [{}], metrics: {}, costModel: { reference_cost_100: 1.5 }, beforeAfter: [comparison] }),
  });

  assertSameKeys(body, served, 'the /quality body');
  assertSameKeys(body.WINDOW, served.WINDOW, 'the /quality WINDOW echo');
  assertSameKeys(body.QUALITY_SNAPSHOTS, served.QUALITY_SNAPSHOTS, 'the QUALITY_SNAPSHOTS Datum');
  assert.equal(body.QUALITY_SNAPSHOTS.state, served.QUALITY_SNAPSHOTS.state);
  assertSameKeys(body.QUALITY.value.cost_model, served.QUALITY.value.cost_model, 'QUALITY.value.cost_model');
  assert.ok(servedComparison, 'this check expects quality_alarm_explained to serve a before/after comparison');
  assertSameKeys(comparison, servedComparison, 'QUALITY.value.before_after[0]');
  assert.equal(comparison.state, servedComparison.state);
});

// No boundary scenario serves a module row yet (each one's modules is []), so
// qualityModuleRow is held to quality_series.py::ModuleRow.to_wire's field
// list only; this pins that the builder's keys stay that one list.
test('a module row is built with the seven keys ModuleRow.to_wire serves', () => {
  assert.deepEqual(Object.keys(qualityModuleRow({ module: 'dashboard' })).sort(), [
    'allocated_lines', 'allocated_usd', 'landings', 'module', 'ranked', 'tasks', 'usd_per_100_lines',
  ]);
});

test('a QUALITY_MARKERS stream and its derived marker are built with exactly the keys /quality serves', () => {
  const served = payload('quality_alarm_explained').QUALITY_MARKERS;
  const [servedMarker] = served.value.markers.filter(marker => marker.source.type === 'routing');
  const built = qualityMarkers([qualityMarker({ id: 'routing:reviewer_comprehensive:1', ordinal: 3 })]);

  assertSameKeys(wireDatum(null), served, 'the QUALITY_MARKERS Datum');
  assertSameKeys(built, served.value, 'QUALITY_MARKERS.value');
  assertSameKeys(built.prompt_coverage, served.value.prompt_coverage, 'QUALITY_MARKERS.value.prompt_coverage');
  assertSameKeys(built.markers[0], servedMarker, 'QUALITY_MARKERS.value.markers[0]');
  assertSameKeys(built.markers[0].source, servedMarker.source, 'QUALITY_MARKERS.value.markers[0].source');
});
