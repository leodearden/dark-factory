// Boundary family 2, burndown: sketch #6 and #7.
//
// Each test lands bodies the REAL /burndown and /tasks routes served over a
// store the REAL sampler wrote (test_boundary_js.py builds them) through
// data.js's real refreshOne, then reads them the way BurnTab and the OrchTab
// tiles do.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { applyServed, loadClient, payload } from './_boundary_client.mjs';

const TASKS = '/api/v2/dashboard/tasks';
const burndownPath = win => `/api/v2/dashboard/burndown?window=${win}`;

async function land(client, path, name, opts = {}) {
  const body = payload(name);
  const receivedAt = Date.parse(body.served_at);
  assert.equal(await applyServed(client, path, body, { receivedAt, ...opts }), 'applied', name);
  return { body, receivedAt };
}

// The OrchTab running tile over the fleet: its headline and its spark.
function runningTile(client, now) {
  const win = client.window;
  const { censusOver, censusHistory, CENSUS_TILES } = win.DF_TASK_SNAPSHOT;
  const tile = CENSUS_TILES.find(t => t.key === 'running');
  const census = censusOver(win.DF_DATA, null);
  return {
    census,
    text: win.DF_DATUM.datumView(census, { now, format: tile.reading }).text,
    history: censusHistory(win.DF_DATA, null, tile),
  };
}

test('sketch #6: the running tile and its spark read one sampled datum, and only the tile moves after', async () => {
  const client = loadClient();
  await land(client, burndownPath('24h'), 'burndown_at_t');
  const { receivedAt: t } = await land(client, TASKS, 'census_at_t');

  const atT = runningTile(client, t);
  const sampled = atT.history.at(-1);
  assert.equal(atT.census.state, 'fresh');
  assert.equal(atT.census.value.sub_views.running, sampled, 'the tile headline is its spark endpoint');
  assert.equal(atT.text, `${sampled} / ${atT.census.value.views.in_flight}`);

  const { receivedAt: later } = await land(client, TASKS, 'census_after_t');
  const after = runningTile(client, later);
  assert.notEqual(after.census.value.sub_views.running, sampled, 'the census moved');
  assert.equal(after.text, `${after.census.value.sub_views.running} / ${after.census.value.views.in_flight}`);
  assert.deepEqual(after.history, atT.history, 'no new sample, so the spark is unchanged');
  assert.equal(after.history.at(-1), sampled);
});

test('sketch #7: a project whose newest sample is a gap renders stale, carried into the aggregate and judged once', async () => {
  const client = loadClient();
  const { receivedAt } = await land(client, burndownPath('30d'), 'burndown_ragged', { win: '30d' });
  const win = client.window;
  const D = win.DF_DATA;
  const { burndownDatum, burndownStacks, parityBannerState } = win.DF_BURNDOWN_BANDS;
  const { MEMBERS, SERIES_KEYS } = win.DF_TASK_VOCAB;
  const { formatAge } = win.DF_ENDPOINT_STALENESS;
  const view = datum => win.DF_DATUM.datumView(datum, { now: receivedAt, format: v => String(v.counts.pending) });

  const a = burndownDatum(D, D.BURNDOWN_BY_PROJECT.A, 'latest');
  const b = burndownDatum(D, D.BURNDOWN_BY_PROJECT.B, 'latest');
  const aggregate = burndownDatum(D, D.BURNDOWN, 'latest');
  assert.equal(a.state, 'fresh');
  assert.equal(b.state, 'stale');
  const bAge = Date.parse(D.BURNDOWN.labels.at(-1)) - Date.parse(b.as_of);
  assert.equal(view(b).age, formatAge(bAge), 'B ages by t2 - t1');
  assert.equal(view(b).age, '1d');
  assert.equal(view(b).title, b.reason);

  for (const member of MEMBERS) {
    const key = SERIES_KEYS[member];
    assert.equal(aggregate.value.counts[key], a.value.counts[key] + b.value.counts[key], `${key}: A(t2) + B(t1)`);
  }
  assert.equal(aggregate.state, 'stale', 'the aggregate is as old as its carried part');
  assert.equal(aggregate.as_of, b.as_of);

  const stacks = burndownStacks(D.BURNDOWN, {});
  assert.equal(stacks.length, 9);
  assert.deepEqual(new Set(stacks.map(s => s.member)), new Set(MEMBERS));
  for (const band of stacks) {
    assert.equal(band.values, D.BURNDOWN[band.key], `${band.member} draws the served series itself`);
    assert.equal(band.values.at(-1), aggregate.value.counts[band.key], `${band.member} ends at the latest reading`);
  }

  assert.deepEqual(parityBannerState(D.BURNDOWN, D.BURNDOWN.parity_projects), {
    peak: 30, cap: 24, text: ' · 1 snapshot over · B',
  });
  assert.deepEqual(parityBannerState(D.BURNDOWN_BY_PROJECT.B, null), { peak: 30, cap: 24, text: ' · 1 snapshot over' });
  assert.equal(parityBannerState(D.BURNDOWN_BY_PROJECT.A, null), null);
});
