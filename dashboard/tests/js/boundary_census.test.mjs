// Boundary family 1, the census: sketch #1-#5 and #13.
//
// Each test lands a body the REAL /tasks route served (test_boundary_js.py
// builds them) through data.js's real refreshOne, then reads it the way every
// census surface does: a datumView over projectCensus or censusOver with the
// surface's own format, plus the Progress bar's censusSegments.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { applyServed, loadClient, payload } from './_boundary_client.mjs';

const TASKS = '/api/v2/dashboard/tasks';
const HOUR_MS = 60 * 60 * 1000;

async function served(name) {
  const client = loadClient();
  const body = payload(name);
  const receivedAt = Date.parse(body.served_at);
  assert.equal(await applyServed(client, TASKS, body, { receivedAt }), 'applied');
  return { client, body, receivedAt };
}

// What every census surface renders for `scope` (a project, or null for the
// fleet) at `now`, each named for the surface that calls it.
function surfaces(client, scope, now) {
  const win = client.window;
  const { censusOver, projectCensus, CENSUS_VIEWS, CENSUS_TILES } = win.DF_TASK_SNAPSHOT;
  const { censusSegments, terminalOfTotal, runningOfInFlight, inFlightCount } = win.DF_TASK_SNAPSHOT;
  const census = scope === null ? censusOver(win.DF_DATA, null) : projectCensus(win.DF_DATA, scope);
  const view = format => win.DF_DATUM.datumView(census, { now, format });
  return {
    census,
    readings: {
      ...Object.fromEntries(CENSUS_VIEWS.map(v => [`pip:${v.key}`, view(v.reading)])),
      ...Object.fromEntries(CENSUS_VIEWS.map(v => [`filter:${v.key}`, view(v.count)])),
      ...Object.fromEntries(CENSUS_TILES.map(t => [`tile:${t.key}`, view(t.reading)])),
      progress: view(terminalOfTotal),
      topbar: view(runningOfInFlight),
      rail: view(inFlightCount),
    },
    segments: censusSegments(census),
  };
}

function servedNumbers(value) {
  return new Set(
    [value.total, ...Object.values(value.views), ...Object.values(value.sub_views), ...Object.values(value.counts)]
      .map(String),
  );
}

function sharesOf(value) {
  return Object.fromEntries(Object.entries(value.views).map(([key, n]) => [key, (n / value.total) * 100]));
}

test('sketch #1: every census surface renders the served view values and no other number', async () => {
  const { client, body, receivedAt } = await served('census_fresh');
  const value = body.TASKS_SNAPSHOT['dark-factory'].census.value;
  const { readings } = surfaces(client, 'dark-factory', receivedAt);

  assert.deepEqual(
    Object.fromEntries(Object.entries(readings).map(([surface, view]) => [surface, view.text])),
    {
      'pip:in_flight': '25 running of 43 in-flight',
      'pip:backlog': '1310 backlog',
      'pip:terminal': '4106 terminal',
      'filter:in_flight': '43',
      'filter:backlog': '1310',
      'filter:terminal': '4106',
      'tile:running': '25 / 43',
      'tile:blocked': '10',
      'tile:pending': '1200',
      progress: '4106/5459',
      topbar: '25 running of 43 in-flight',
      rail: '43',
    },
  );
  const allowed = servedNumbers(value);
  for (const [surface, view] of Object.entries(readings)) {
    assert.equal(view.age, null, `${surface} is fresh and must not badge`);
    for (const n of view.text.match(/\d+/g)) {
      assert.ok(allowed.has(n), `${surface} renders ${n}, which the served census does not carry`);
    }
  }
});

test('sketch #2: an unknown census renders the placeholder and its reason on every surface, never a zero', async () => {
  const before = surfaces(loadClient(), null, Date.now());
  for (const [surface, view] of Object.entries(before.readings)) {
    assert.equal(view.text, '—', `${surface} before the first fetch`);
    assert.equal(view.title, 'not yet fetched', `${surface} before the first fetch`);
  }

  const { client, body, receivedAt } = await served('census_unknown');
  const reason = body.TASKS_SNAPSHOT['dark-factory'].census.reason;
  for (const scope of ['dark-factory', null]) {
    const { readings, segments } = surfaces(client, scope, receivedAt);
    assert.deepEqual(segments, [], `${scope ?? 'the fleet'} draws no bar over a hole`);
    for (const [surface, view] of Object.entries(readings)) {
      assert.equal(view.text, '—', `${surface} over ${scope ?? 'the fleet'}`);
      assert.notEqual(view.text, '0');
      assert.notEqual(view.text, '0/1');
      assert.ok(view.title.includes(reason), `${surface} over ${scope ?? 'the fleet'} is titled ${view.title}`);
    }
  }

  const D = client.window.DF_DATA;
  const notices = client.window.DF_TASKS_OFFLINE_BANNER.tasksBannerNotices({
    offline: !!D.TASKS_OFFLINE,
    offlineProjects: D.TASKS_OFFLINE_PROJECTS || [],
    degradedProjects: D.TASKS_DEGRADED_PROJECTS || [],
    countUnknownProjects: D.TASKS_COUNT_UNKNOWN_PROJECTS || [],
    totalProjects: D.TASKS_PROJECT_COUNT || 0,
  });
  assert.deepEqual(notices.map(n => n.kind), ['count-unknown']);
  assert.ok(notices[0].text.includes('dark-factory'), notices[0].text);
});

test('sketch #3: a census served three hours stale renders its value with an age badge that keeps growing', async () => {
  const { client, body, receivedAt } = await served('census_stale_3h');
  const census = body.TASKS_SNAPSHOT['dark-factory'].census;

  const atReceipt = surfaces(client, 'dark-factory', receivedAt);
  assert.equal(atReceipt.readings.progress.text, '4106/5459');
  for (const [surface, view] of Object.entries(atReceipt.readings)) {
    assert.equal(view.age, '3h', surface);
    assert.equal(view.title, census.reason, surface);
  }
  const shares = sharesOf(census.value);
  for (const segment of atReceipt.segments) assert.equal(segment.share, shares[segment.key], segment.key);

  const anHourOn = surfaces(client, 'dark-factory', receivedAt + HOUR_MS);
  for (const [surface, view] of Object.entries(anHourOn.readings)) {
    assert.equal(view.age, '4h', `${surface} an hour later, with no new payload`);
  }
});

test('sketch #4: the client partitions and sums the served census without re-counting it', async () => {
  const { client, body, receivedAt } = await served('census_fresh');
  const perProject = Object.values(body.TASKS_SNAPSHOT).map(entry => entry.census.value);
  const { census, segments } = surfaces(client, null, receivedAt);
  const fleet = census.value;

  assert.equal(census.state, 'fresh');
  assert.equal(fleet.total, perProject.reduce((sum, v) => sum + v.total, 0));
  for (const key of Object.keys(fleet.views)) {
    assert.equal(fleet.views[key], perProject.reduce((sum, v) => sum + v.views[key], 0), key);
  }
  assert.equal(fleet.sub_views.running, perProject.reduce((sum, v) => sum + v.sub_views.running, 0));
  assert.equal(Object.values(fleet.views).reduce((a, b) => a + b, 0), fleet.total, 'the views partition the total');
  assert.ok(Math.abs(segments.reduce((sum, s) => sum + s.share, 0) - 100) < 1e-9, 'the bar is the whole');
});

test('sketch #5: nine statuses, each counted once, review and infra-hold in flight', async () => {
  const { client, receivedAt } = await served('census_nine');
  const { MEMBERS, VIEWS } = client.window.DF_TASK_VOCAB;
  const { census, readings } = surfaces(client, 'dark-factory', receivedAt);

  assert.equal(MEMBERS.length, 9);
  assert.deepEqual(census.value.counts, Object.fromEntries(MEMBERS.map(member => [member, 1])));
  assert.ok(VIEWS.in_flight.includes('review') && VIEWS.in_flight.includes('infra-hold'));
  assert.equal(readings['pip:in_flight'].text, '1 running of 5 in-flight');
  assert.equal(readings.progress.text, '2/9');
});

test('sketch #13: a transient census failure renders fresh, then stale with its ReadTimeout, then fresh', async () => {
  const client = loadClient();
  const rendered = [];
  for (const phase of ['before', 'during', 'after']) {
    const body = payload(`census_transient_${phase}`);
    const receivedAt = Date.parse(body.served_at);
    assert.equal(await applyServed(client, TASKS, body, { receivedAt }), 'applied');
    const { census, readings } = surfaces(client, 'dark-factory', receivedAt);
    rendered.push({ state: census.state, title: readings.progress.title, text: readings.progress.text });
  }

  const [before, during, after] = rendered;
  assert.deepEqual(before, { state: 'fresh', title: null, text: '4106/5459' });
  assert.equal(during.state, 'stale');
  assert.equal(during.text, '4106/5459');
  assert.match(during.title, /ReadTimeout/);
  assert.deepEqual(after, { state: 'fresh', title: null, text: '4106/5459' });
});
