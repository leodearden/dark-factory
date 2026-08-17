import json, pytest
from test_consolidate_memories_tool import make_service, call_consolidate, S1, S2

@pytest.mark.asyncio
async def test_duplicate_supersedes():
    svc = make_service(gone=[S1, S2])
    result = await call_consolidate(svc, supersedes=[S1, S1, S2])
    out = {
        'status': result['status'],
        'deleted': result['deleted'],
        'failed': result['failed_deletes'],
        'tw': result['tombstones_written'],
        'te': result['tombstones_expected'],
        'calls': [c.kwargs['memory_id'] for c in svc.delete_memory.await_args_list],
        'canon_sup': result['canonical_supersedes'],
    }
    open('/tmp/dup_out.json','w').write(json.dumps(out, indent=2))
