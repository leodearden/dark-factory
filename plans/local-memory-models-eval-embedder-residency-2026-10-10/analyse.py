"""Summarise one run's logs: per phase, device memory peak, per-pid peaks, whisper latency."""
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path


def parse_ts(s):
    return datetime.strptime(s.strip(), '%Y/%m/%d %H:%M:%S.%f').timestamp()


def load_gpu(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        rows.append((parse_ts(parts[0]), int(parts[1]), int(parts[2])))
    return rows


def load_apps(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 4 or not parts[3].isdigit():
            continue
        rows.append((parse_ts(parts[0]), int(parts[1]), parts[2], int(parts[3])))
    return rows


def main(logdir):
    logdir = Path(logdir)
    events = [json.loads(line) for line in (logdir / 'events.jsonl').read_text().splitlines()]
    gpu = load_gpu(logdir / 'gpu.csv')
    apps = load_apps(logdir / 'apps.csv')
    out = {'samples_gpu': len(gpu), 'samples_apps': len(apps),
           'window_device_peak_mib': max(r[1] for r in gpu),
           'window_device_min_mib': min(r[1] for r in gpu),
           'first_device_mib': gpu[0][1], 'last_device_mib': gpu[-1][1]}
    pid_peak = {}
    for _, pid, _, mib in apps:
        pid_peak[pid] = max(pid_peak.get(pid, 0), mib)
    out['pid_peak_mib'] = pid_peak
    out['pid_first_mib'] = {}
    out['pid_last_mib'] = {}
    for _, pid, _, mib in apps:
        out['pid_first_mib'].setdefault(pid, mib)
        out['pid_last_mib'][pid] = mib
    starts = {e['phase']: e['t'] for e in events if e['event'] == 'phase_start'}
    ends = {e['phase']: e['t'] for e in events if e['event'] == 'phase_end'}
    phases = {}
    for ph, t0 in starts.items():
        t1 = ends.get(ph, t0)
        win = [r for r in gpu if t0 <= r[0] <= t1]
        aw = [r for r in apps if t0 <= r[0] <= t1]
        tr = [e for e in events if e['event'] == 'transcribe' and e['phase'] == ph]
        short = [e['secs'] for e in tr if e['clip'] == 'jfk11s']
        long_ = [e['secs'] for e in tr if e['clip'] == 'jfk30s']
        pp = {}
        for _, pid, _, mib in aw:
            pp[pid] = max(pp.get(pid, 0), mib)
        phases[ph] = {
            'duration_s': round(t1 - t0, 1),
            'device_peak_mib': max((r[1] for r in win), default=None),
            'device_median_mib': statistics.median([r[1] for r in win]) if win else None,
            'gpu_util_median': statistics.median([r[2] for r in win]) if win else None,
            'pid_peak_mib': pp,
            'jfk11s_median_s': round(statistics.median(short), 4) if short else None,
            'jfk11s_all_s': short,
            'jfk11s_min_match': min((e['match'] for e in tr if e['clip'] == 'jfk11s'), default=None),
            'jfk30s_median_s': round(statistics.median(long_), 4) if long_ else None,
            'load': next((e for e in events if e['event'] == 'load_summary' and e['phase'] == ph), None),
        }
    out['phases'] = phases
    out['embed_errors'] = sum(1 for e in events if e['event'] == 'embed_error')
    out['warmup'] = next((e for e in events if e['event'] == 'warmup'), None)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
