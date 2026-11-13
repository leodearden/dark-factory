import asyncio, json, sys
sys.path.insert(0, 'plans/edge-bm25-leg-ranking-9542')
import measure_leg as m
from falkordb import FalkorDB
from graphiti_core.search import search_utils
from graphiti_core.search.search_filters import SearchFilters

async def ranks(ro, query, limit):
    ro.issued.clear()
    driver = ro.driver()
    edges = await search_utils.edge_fulltext_search(driver, query, SearchFilters(), [m.GRAPH], limit)
    issued = list(ro.issued)
    if not issued:
        return {}, []
    cy, params, _ = issued[-1]
    a_query = m.build_query(driver.sanitize(query), [m.GRAPH], search_utils.MAX_QUERY_LENGTH)
    promoted = list(params['promoted_uuids'])
    out = {'C': [e.uuid for e in edges]}
    if a_query:
        out['A'] = [r['uuid'] for r in ro.rows(cy, {**params, 'query': a_query, 'promoted_uuids': []})]
        out['A+tier'] = [r['uuid'] for r in ro.rows(cy, {**params, 'query': a_query})]
        if promoted:
            live = [r['uuid'] for r in ro.rows('MATCH ()-[e:RELATES_TO]->() WHERE e.uuid IN $u AND e.invalid_at IS NULL RETURN e.uuid AS uuid', {'u': promoted})]
            out['A+livetier'] = [r['uuid'] for r in ro.rows(cy, {**params, 'query': a_query, 'promoted_uuids': live})]
            out['C+livetier'] = [r['uuid'] for r in ro.rows(cy, {**params, 'promoted_uuids': live})]
        else:
            out['A+livetier'] = out['A+tier']; out['C+livetier'] = out['C']
    return out, promoted

async def main():
    ro = m.ReadOnlyGraph(FalkorDB(host=m.HOST, port=m.PORT).select_graph(m.GRAPH))
    live = m.live_edges(ro)
    stride = len(live) // m.M1_SAMPLE
    sample = live[::stride][:m.M1_SAMPLE]
    found = {}; rk = {}; fired = 0
    for e in sample:
        r, p = await ranks(ro, e['fact'], m.WRITE_PATH_LIMIT)
        fired += bool(p)
        for k, v in r.items():
            if e['uuid'] in v:
                found[k] = found.get(k, 0) + 1
                rk.setdefault(k, []).append(v.index(e['uuid']) + 1)
    print('live', len(live), 'stride', stride, 'n', len(sample), 'fired', fired)
    for k in sorted(found):
        print('M1', k, 'recall@20', found[k] / len(sample), 'mean rank', sum(rk[k]) / len(rk[k]))
    r, p = await ranks(ro, m.TICKET_877_QUERY, m.READ_PATH_LIMIT)
    inv = {x['u']: x['i'] for x in ro.rows('MATCH ()-[e:RELATES_TO]->() WHERE e.uuid IN $u RETURN e.uuid AS u, e.invalid_at IS NOT NULL AS i', {'u': sum(r.values(), [])})}
    for k, v in r.items():
        print('877', k, [(u[:8], 'INV' if inv.get(u) else 'live') for u in v[:10]])
        print('    named ranks', {pfx: m.rank_of_prefix(pfx, v) for pfx in m.TICKET_877_EDGES})

asyncio.run(main())
