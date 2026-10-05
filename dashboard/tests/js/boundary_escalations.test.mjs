// Boundary family 4a, the escalation corpus: sketch #10.
//
// The REAL /escalations and /escalation-analytics routes served each body
// over one live queue, in two corpus-TTL generations (test_boundary_js.py
// builds them). Each pair lands through data.js's real refreshOne and is read
// the way the Escalations pill and the analytics strip read it.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { applyServed, loadClient, payload } from './_boundary_client.mjs';

const ESCALATIONS = '/api/v2/dashboard/escalations';
const ANALYTICS = '/api/v2/dashboard/escalation-analytics';

for (const generation of ['a', 'b']) {
  test(`sketch #10: in TTL generation ${generation} the pill reads 2 and the strip 5, as of one walk`, async () => {
    const client = loadClient();
    let receivedAt;
    for (const [path, name] of [[ESCALATIONS, `escalations_${generation}`], [ANALYTICS, `analytics_${generation}`]]) {
      const body = payload(name);
      receivedAt = Date.parse(body.served_at);
      assert.equal(await applyServed(client, path, body, { receivedAt }), 'applied', name);
    }
    const D = client.window.DF_DATA;
    const { queuePending, openInHistoryOver } = client.window.DF_ESCALATION_VIEWS;
    const read = datum => client.window.DF_DATUM.datumView(datum, { now: receivedAt });

    const pill = queuePending(D);
    const strip = openInHistoryOver(D, null);
    assert.deepEqual([read(pill).text, read(strip).text], ['2', '5']);
    assert.deepEqual([pill.state, strip.state], ['fresh', 'fresh']);
    assert.equal(pill.as_of, strip.as_of, 'the pill and the strip must count one walk');
  });
}
