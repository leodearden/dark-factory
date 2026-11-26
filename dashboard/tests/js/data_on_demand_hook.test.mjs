// data.js::createOnDemandHook and data.js::taskUid — the on-demand protocol as
// a component reads it, and the uid the taskProse row is addressed by.
//
// The hook is driven through reactStandIn over a freshly loaded data.js whose
// fetch answers each url only when a case says so, so what the hook returns is
// observed between a request leaving and its answer landing.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { drain, loadDataJs } from './_data_loader.mjs';
import { reactStandIn } from './_react_stand_in.mjs';

const FIRST = 'dark-factory/T-19';
const SECOND = 'dark-factory/T-20';

// A data.js whose fetch parks every request until answer(url, response).
function deferredFetchLoad() {
  const pending = new Map();
  const load = loadDataJs({
    fetchStub: url => new Promise(resolve => { pending.set(url, resolve); }),
  });
  const { ON_DEMAND_KEYS } = load.api;
  return {
    ...load,
    answer: async (uid, response) => {
      pending.get(ON_DEMAND_KEYS.taskProse.url(uid))(response);
      await drain();
    },
    served: (uid, prose) => ({
      ok: true, status: 200, json: async () => ({ [ON_DEMAND_KEYS.taskProse.key(uid)]: prose }),
    }),
  };
}

const FAILED = { ok: false, status: 503, json: async () => ({}) };
const PROSE = { description: 'why it exists', details: 'how' };

function proseSurface(load) {
  const react = reactStandIn();
  const { useOnDemand } = load.api.createOnDemandHook(react);
  return uid => react.render(() => useOnDemand('taskProse', uid));
}

function proseRequests(load) {
  return load.fetchCalls.map(call => call.url).filter(url => url.startsWith('/api/v2/dashboard/task/'));
}

test('taskUid: `<project>/T-<id>`, the param taskProse encodes segment by segment', () => {
  const { api } = loadDataJs();
  assert.equal(api.taskUid('dark-factory', 19), FIRST);
  assert.equal(api.ON_DEMAND_KEYS.taskProse.url(api.taskUid('my proj', '7')), '/api/v2/dashboard/task/my%20proj/T-7');
});

test('useOnDemand: a mount requests once and reads loading, then ready with the value that landed', async () => {
  const load = deferredFetchLoad();
  const show = proseSurface(load);

  assert.deepEqual(show(FIRST), { view: load.api.ON_DEMAND_VIEWS.loading, value: undefined });
  await load.answer(FIRST, load.served(FIRST, PROSE));

  assert.deepEqual(show(FIRST), { view: load.api.ON_DEMAND_VIEWS.ready, value: PROSE });
  assert.deepEqual(proseRequests(load), [load.api.ON_DEMAND_KEYS.taskProse.url(FIRST)]);
});

test('useOnDemand: a failed request reads unavailable', async () => {
  const load = deferredFetchLoad();
  const show = proseSurface(load);

  show(FIRST);
  await load.answer(FIRST, FAILED);

  assert.deepEqual(show(FIRST), { view: load.api.ON_DEMAND_VIEWS.unavailable, value: undefined });
});

test("useOnDemand: a new param is requested, and the superseded request's answer does not settle it", async () => {
  const load = deferredFetchLoad();
  const show = proseSurface(load);
  const { loading, unavailable } = load.api.ON_DEMAND_VIEWS;

  show(FIRST);
  show(SECOND);
  await load.answer(FIRST, FAILED);
  assert.equal(show(SECOND).view, loading, "the first uid's failure settled the second's view");

  await load.answer(SECOND, FAILED);
  assert.equal(show(SECOND).view, unavailable);
  assert.deepEqual(
    proseRequests(load),
    [FIRST, SECOND].map(uid => load.api.ON_DEMAND_KEYS.taskProse.url(uid)),
  );
});
