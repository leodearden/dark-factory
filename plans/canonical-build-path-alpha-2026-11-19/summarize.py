#!/usr/bin/env python3
"""summarize.py <results-dir> <run-id>: one run's counters, CPU and missed units, as JSON.

Counters are the private server's end-of-leg stats (sccache-end.json). CPU is the sum over the
run's steps of the unit cgroup's usage_usec delta (snap.py before/after). Missed units are the
`[crate]: Cache miss` verdict lines of the server's debug log, split by step window when possible.
"""
import json
import os
import re
import sys

res, run = sys.argv[1], sys.argv[2]
d = os.path.join(res, run)


def load(p):
    with open(p) as f:
        return json.load(f)


try:
    end = load(os.path.join(d, 'sccache-end.json'))
    end_source = 'sccache-end.json'
except (OSError, ValueError):   # W1: lost to a harness bug; the last step's after-snapshot is cumulative
    last = [l.split('\t')[2] for l in open(os.path.join(res, 'steps.tsv')) if l.split('\t')[1] == run][-1]
    end = load(os.path.join(d, f'{last}.after.json'))['sccache']
    end_source = f'{last}.after.json'
st = end['stats']
hits = sum(st['cache_hits']['counts'].values())
misses = sum(st['cache_misses']['counts'].values())
steps = []
cpu_total = 0.0
for line in open(os.path.join(res, 'steps.tsv')):
    host, rid, name, rc, wall = line.rstrip('\n').split('\t')
    if rid != run:
        continue
    b = load(os.path.join(d, f'{name}.before.json'))
    a = load(os.path.join(d, f'{name}.after.json'))
    cpu = (a['cpu_usec'] - b['cpu_usec']) / 6e7 if a['cpu_usec'] is not None else None
    sb, sa = b['sccache'].get('stats', {}), a['sccache'].get('stats', {})
    def delta(k):
        return sum(sa[k]['counts'].values()) - sum(sb[k]['counts'].values()) if sa and sb else None
    steps.append({'step': name, 'rc': int(rc), 'wall_s': float(wall), 'cpu_min': round(cpu, 1) if cpu else cpu,
                  'hits': delta('cache_hits'), 'misses': delta('cache_misses')})
    if name != 'tests_in_namespace' and cpu:
        cpu_total += cpu

miss_re = re.compile(r'\[([^\]]+)\]: Cache miss')
missed = []
log = os.path.join(d, 'server.log')
if os.path.exists(log):
    with open(log, errors='replace') as f:
        for line in f:
            m = miss_re.search(line)
            if m:
                missed.append(m.group(1))

out = {
    'run': run, 'counters_from': end_source, 'host': steps and host, 'cache_location': end.get('cache_location'),
    'hits': hits, 'misses': misses, 'hits_by_lang': st['cache_hits']['counts'],
    'misses_by_lang': st['cache_misses']['counts'],
    'miss_share': round(misses / (hits + misses), 4) if hits + misses else None,
    'writes': st['cache_writes'], 'write_errors': st['cache_write_errors'],
    'read_errors': st['cache_read_errors'], 'timeouts': st['cache_timeouts'],
    'not_cached': st['not_cached'], 'compile_fails': st['compile_fails'],
    'build_cpu_min': round(cpu_total, 1), 'steps': steps,
    'missed_units': sorted(set(missed)), 'missed_unit_lines': len(missed),
}
print(json.dumps(out, indent=1))
