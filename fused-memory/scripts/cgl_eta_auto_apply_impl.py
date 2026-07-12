#!/usr/bin/env python3
"""CGL-η auto-apply IMPL (committed before_done=predicate action for the
deterministic bulk-apply task; see cgl_eta_auto_apply.sh wrapper).

Runs the Phase-1 live cross-graph migration over the WHOLE real census and
decides task success by exit code (predicate contract): exit 0 == clean apply
(deterministic task -> done); non-zero == a real problem (born-at-L2 escalation
for a human/L2 session to review). It is safe to run ONLY after task 2451 lands
(null-embedding tolerance + per-item Phase-B isolation) -- the deterministic
task depends on 2451, so dispatch is gated on that.

Design (idempotent + create-before-delete throughout, so re-run on resume is
safe -- matching the predicate re-run-on-resume contract):
  1. Lean GraphitiBackend shim (identity-scan + index build stubbed) -- NO
     second MemoryService, no Graphiti maintenance writes on init.
  2. Fresh dry-run census -> manifest (written to a dated durable file).
  3. Pre-apply recovery dump: full content (fact + best-effort embedding +
     endpoints) of every cross-target edge that WILL be dropped, so the
     dropped facts remain recoverable (the run() report records only edge
     identity, not fact/embedding). Read-only.
  4. FalkorDB backup (BGSAVE + copy out). ABORTS (exit 3) if the backup fails
     -- never migrates irreversibly without a fresh restore point.
  5. Apply the fresh manifest via the shipped three-phase run(apply=True).
  6. Post-verify: 0 foreign :Entity nodes must remain (every displaced entity
     moved home); 0 unexpected-blocked MOVE/MERGE. Exit 0 iff both hold.

All artifacts land under fused-memory/data/cgl-eta/ with a run-stamp suffix
passed by the wrapper via $CGL_RUN_STAMP (predicate re-runs get a fresh stamp
so an earlier run's artifacts are never clobbered).
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

REPO = Path('/home/leo/src/dark-factory')
FM = REPO / 'fused-memory'
SCRIPT = FM / 'scripts' / 'migrate_cross_graph_leak.py'
OUTDIR = FM / 'data' / 'cgl-eta'
STAMP = os.environ.get('CGL_RUN_STAMP', 'autorun')
FALKOR_CONTAINER = os.environ.get('FALKORDB_CONTAINER', 'docker-falkordb-1')


def _load_migrate():
    spec = importlib.util.spec_from_file_location('migrate_cross_graph_leak', SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def _noop_idx(self, g):
    return None


async def _noop_scan(self):
    return {'stubbed': True}


def _log(msg: str) -> None:
    print(f'[cgl-auto-apply] {msg}', flush=True)


def backup_falkordb() -> Path:
    """BGSAVE the live FalkorDB and copy the RDB out. Raises on any failure."""
    dest = OUTDIR / f'falkordb-backup-{STAMP}.rdb'
    before = subprocess.run(['docker', 'exec', FALKOR_CONTAINER, 'redis-cli', 'LASTSAVE'],
                            capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(['docker', 'exec', FALKOR_CONTAINER, 'redis-cli', 'BGSAVE'],
                   capture_output=True, text=True, check=True)
    # poll for completion (bounded)
    for _ in range(120):
        inprog = subprocess.run(['docker', 'exec', FALKOR_CONTAINER, 'redis-cli', 'INFO', 'persistence'],
                                capture_output=True, text=True).stdout
        after = subprocess.run(['docker', 'exec', FALKOR_CONTAINER, 'redis-cli', 'LASTSAVE'],
                               capture_output=True, text=True).stdout.strip()
        if 'rdb_bgsave_in_progress:0' in inprog and after != before:
            break
        subprocess.run(['sleep', '1'])
    status = subprocess.run(['docker', 'exec', FALKOR_CONTAINER, 'redis-cli', 'INFO', 'persistence'],
                            capture_output=True, text=True).stdout
    if 'rdb_last_bgsave_status:ok' not in status:
        raise RuntimeError('FalkorDB BGSAVE did not report ok')
    subprocess.run(['docker', 'cp', f'{FALKOR_CONTAINER}:/var/lib/falkordb/data/dump.rdb', str(dest)],
                   capture_output=True, text=True, check=True)
    size = dest.stat().st_size
    if size < 1_000_000:
        raise RuntimeError(f'FalkorDB backup suspiciously small ({size} bytes)')
    _log(f'FalkorDB backup OK -> {dest} ({size} bytes)')
    return dest


async def _compact_vec(backend, gkey, edge_uuid):
    """Best-effort raw --compact fact_embedding read; None if null/absent."""
    q = f"MATCH ()-[e:RELATES_TO {{uuid: '{edge_uuid}'}}]-() RETURN e.fact_embedding"
    try:
        client = backend._require_falkor_client()
        reply = await client.execute_command('GRAPH.RO_QUERY', gkey, q, '--compact')
        rows = reply[1]
        if not rows:
            return None
        cell = rows[0][0]
        stype, value = int(cell[0]), cell[1]
        if stype != 12:
            return None
        toks = [v.decode() if isinstance(v, bytes) else str(v) for v in value]
        return '[' + ', '.join(toks) + ']'
    except Exception:
        return None


async def dump_cross_target(backend, migrate, manifest) -> Path:
    """Read-only recovery dump of every cross-target edge's full content."""
    nodes = manifest['nodes']
    target_of = {n['uuid']: n['target_graph'] for n in nodes if n['disposition'] in ('MOVE', 'MERGE')}
    src_of = {n['uuid']: n['source_graph'] for n in nodes if n['disposition'] in ('MOVE', 'MERGE')}
    present_cache: dict = {}

    async def present(g, u):
        k = (g, u)
        if k not in present_cache:
            r = await backend._graph_for(g).ro_query('MATCH (n:Entity {uuid: $u}) RETURN n.uuid LIMIT 1', {'u': u})
            present_cache[k] = bool(r.result_set)
        return present_cache[k]

    from collections import defaultdict
    by_src = defaultdict(list)
    for u, s in src_of.items():
        by_src[s].append(u)
    seen: set = set()
    dropped = []
    for src_graph, uuids in by_src.items():
        graph = backend._graph_for(src_graph)
        for uuid in uuids:
            res = await graph.ro_query(
                'MATCH (n:Entity {uuid: $u})-[e:RELATES_TO]-(m:Entity) '
                'WITH DISTINCT e, startNode(e) AS s, endNode(e) AS t '
                'RETURN e.uuid, e.name, e.fact, s.uuid, s.name, t.uuid, t.name', {'u': uuid})
            for row in (res.result_set or []):
                e_uuid, e_name, fact, s_uuid, s_name, t_uuid, t_name = row
                if e_uuid in seen:
                    continue
                seen.add(e_uuid)
                s_tgt, t_tgt = target_of.get(s_uuid), target_of.get(t_uuid)
                if s_tgt is None and t_tgt is not None and await present(t_tgt, s_uuid):
                    s_tgt = t_tgt
                elif t_tgt is None and s_tgt is not None and await present(s_tgt, t_uuid):
                    t_tgt = s_tgt
                if s_tgt is not None and t_tgt is not None and s_tgt == t_tgt:
                    continue  # preserved, not dropped
                dropped.append({
                    'edge_uuid': e_uuid, 'edge_name': e_name, 'fact': fact,
                    'lives_in_graph': src_graph, 'src_uuid': s_uuid, 'src_name': s_name,
                    'dst_uuid': t_uuid, 'dst_name': t_name, 'src_target': s_tgt, 'dst_target': t_tgt,
                    'fact_embedding_compact': await _compact_vec(backend, src_graph, e_uuid),
                })
    dest = OUTDIR / f'cross-target-edges-recovery-{STAMP}.json'
    dest.write_text(json.dumps({'count': len(dropped), 'edges': dropped}, indent=2, default=str))
    _log(f'cross-target recovery dump: {len(dropped)} edges -> {dest}')
    return dest


async def foreign_entity_residual(backend, migrate) -> dict:
    """Per-graph count of foreign nodes that ARE :Entity (should be 0 post-apply)."""
    out = {}
    for g in await backend.list_graphs():
        foreign = await migrate.census_foreign_nodes(backend, g, page_size=1000)
        ent = [n for n in foreign if 'Entity' in (n.get('labels') or [])]
        if ent:
            out[g] = len(ent)
    return out


async def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    from fused_memory.backends.graphiti_client import GraphitiBackend
    from fused_memory.config.schema import FusedMemoryConfig

    backend = GraphitiBackend(FusedMemoryConfig())
    backend._ensure_indices = types.MethodType(_noop_idx, backend)
    backend._run_startup_identity_scan = types.MethodType(_noop_scan, backend)
    await backend.initialize()
    migrate = _load_migrate()
    shim = types.SimpleNamespace(graphiti=backend)

    try:
        # 2. fresh census -> manifest
        dry_args = types.SimpleNamespace(apply=False, manifest=None, page_size=1000, config=None)
        manifest = await migrate.run(dry_args, shim)
        man_path = OUTDIR / f'apply-manifest-{STAMP}.json'
        man_path.write_text(json.dumps(manifest, indent=2, default=str))
        summ = manifest.get('summary', {})
        _log(f"census: {summ}  (UNRESOLVED={summ.get('UNRESOLVED')})")
        if summ.get('UNRESOLVED'):
            _log('ABORT: UNRESOLVED nodes present -- needs human alias-map review, not auto-apply.')
            return 2

        # 3. recovery dump (read-only, before any drop)
        await dump_cross_target(backend, migrate, manifest)

        # 4. backup (abort if it fails)
        try:
            backup_falkordb()
        except Exception as exc:
            _log(f'ABORT: FalkorDB backup failed ({exc}) -- refusing to migrate without a restore point.')
            return 3

        # 5. apply
        apply_args = types.SimpleNamespace(apply=True, manifest=str(man_path), page_size=1000, config=None)
        report = await migrate.run(apply_args, shim)
        rep_path = OUTDIR / f'apply-report-{STAMP}.json'
        rep_path.write_text(json.dumps(report, indent=2, default=str))
        ar = report.get('apply_results', [])
        applied = sum(1 for r in ar if r.get('applied'))
        unexpected = [r for r in ar if r.get('blocked') and r.get('disposition') in ('MOVE', 'MERGE') and r.get('error')]
        _log(f"apply: {len(ar)} results, {applied} applied, edges_recreated={report.get('edges_recreated')} "
             f"edges_skipped={report.get('edges_skipped')} dropped_cross_target={len(report.get('dropped_cross_target_edges', []))} "
             f"unexpected_blocked={len(unexpected)}")

        # 6. post-verify
        residual = await foreign_entity_residual(backend, migrate)
        _log(f'post-verify foreign :Entity residual (want empty): {residual}')

        clean = (not residual) and (not unexpected)
        if clean:
            _log('RESULT: CLEAN -- every displaced :Entity moved home; no unexpected blocks. exit 0 (done).')
            return 0
        _log('RESULT: NOT CLEAN -- residual foreign :Entity and/or unexpected-blocked MOVE/MERGE. '
             f'exit 1 (escalate). report={rep_path}')
        for r in unexpected[:10]:
            _log(f"  blocked {r.get('uuid','?')[:12]} {r.get('disposition')}: {r.get('error')}")
        return 1
    finally:
        if hasattr(backend, 'close'):
            await backend.close()


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
