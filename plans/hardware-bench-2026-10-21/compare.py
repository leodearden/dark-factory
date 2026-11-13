"""Assemble the cross-host comparison from bench.sh/phase2/phase3 results dirs.

    python3 compare.py <laptop dir> <pod dir> [<pod dir> ...]

Every ratio is laptop seconds / host seconds: above 1 means the host is faster.
Each figure names the rule it was computed under; see RULES at the end of the output.
"""

import json
import sqlite3
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

DF_RUNS_DB = Path('/home/leo/src/dark-factory/data/orchestrator/runs.db')
REIFY_RUNS_DB = Path('/home/leo/src/reify/data/orchestrator/runs.db')
HERE = Path(__file__).resolve().parent
LAPTOP_REIFY_STEPS = HERE / 'laptop-reify-verify-runs.jsonl'
# The laptop's whole reify verify ledger (copied 2026-10-22): its uncontended baseline.
LAPTOP_REIFY_LEDGER = HERE / 'laptop-reify-ledger-2026-10-22.jsonl'
DF_SHA = '71bc561086e5116c5ef65bc91d5b4e44a77ef211'
REIFY_WARM_SHA = '57be3b937f8cddfb77408603f46a798710644d89'
# verify.sh omits these on the pods (REIFY_INFRA_SUITE_ACTIVE=1); the laptop ran them.
REIFY_EXCLUDED_PHASES = {'infra_pool', 'release_prebuild'}


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def median_wall(batches: list[dict], probe: str, k: int) -> float | None:
    walls = [r['wall_s'] for b in batches if b['probe'] == probe and b['k'] == k for r in b['runs'] if r['rc'] == 0]
    return statistics.median(walls) if walls else None


def top_k(batches: list[dict], probe: str) -> int | None:
    ks = [b['k'] for b in batches if b['probe'] == probe]
    return max(ks) if ks else None


def batch_field(batches: list[dict], probe: str, k: int, field: str) -> float | None:
    vals = [b.get(field) for b in batches if b['probe'] == probe and b['k'] == k and b.get(field) is not None]
    return statistics.mean(vals) if vals else None


class Host:
    def __init__(self, path: Path):
        self.path = path
        self.info = json.loads((path / 'host.json').read_text())
        self.first = jsonl(path / 'probes.jsonl')
        self.rerun = jsonl(path / 'probes.rerun.jsonl')
        self.diag = jsonl(path / 'diag.jsonl')
        self.gates = jsonl(path / 'gates.jsonl') + jsonl(path / 'user-gates' / 'gates.jsonl')
        self.reify = jsonl(path / 'reify_gates.jsonl')
        self.reify_steps = jsonl(path / 'reify-verify-runs.jsonl')
        self.samples = jsonl(path / 'st-samples.jsonl')

    @property
    def name(self) -> str:
        model = self.info['cpu_model'].replace('AMD ', '').replace(' Processor', '').replace(' w/ Radeon 780M Graphics', '')
        return f"{model} ({self.info.get('cpu_budget', self.info['threads'])}T)"

    def probe(self, probe: str, k: int | None = 1) -> float | None:
        """Single-threaded probe: the quiet-disk rerun when there is one, else the first set."""
        for batches in (self.rerun, self.first):
            kk = top_k(batches, probe) if k is None else k
            if kk is not None and (value := median_wall(batches, probe, kk)) is not None:
                return value
        return None

    def sample_median(self, probe: str) -> float | None:
        vals = [d['wall_s'] for d in self.samples if d['probe'] == probe]
        return statistics.median(vals) if vals else None

    def diag_median(self, probe: str, field: str = 'wall_s') -> float | None:
        vals = [d[field] for d in self.diag if d['probe'] == probe]
        return statistics.median(vals) if vals else None


def ratio(laptop: float | None, host: float | None) -> str:
    return f'{laptop / host:.2f}×' if laptop and host else '-'


def fmt(seconds: float | None) -> str:
    return f'{seconds:.1f} s' if seconds is not None else '-'


def laptop_df_gate() -> float:
    db = sqlite3.connect(f'file:{DF_RUNS_DB}?mode=ro', uri=True)
    rows = db.execute("select data from events where event_type = 'merge_verify' and data like ?", (f'%{DF_SHA}%',))
    walls = [e['duration_ms'] / 1000 for e in (json.loads(d) for (d,) in rows) if e.get('runner') == 'laptop' and e.get('passed')]
    return statistics.median(walls)


def laptop_df_gate_uncontended() -> tuple[float, int]:
    """Median passed laptop DF gate from 2026-10-12 to 2026-10-22 that overlapped no laptop reify gate."""
    reify = [(b, e) for b, e in merge_verify_intervals(REIFY_RUNS_DB)]
    walls = [e - b for b, e in merge_verify_intervals(DF_RUNS_DB, passed_only=True)
             if not any(min(e, re) > max(b, rb) for rb, re in reify)]
    return statistics.median(walls), len(walls)


def merge_verify_intervals(db_path: Path, passed_only: bool = False) -> list[tuple[float, float]]:
    db = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    rows = db.execute("select timestamp, data from events where event_type = 'merge_verify' "
                      "and data like '%\"runner\": \"laptop\"%' and timestamp > '2026-10-12' and timestamp < '2026-10-23'")
    out = []
    for ts, data in rows:
        d = json.loads(data)
        if passed_only and not d.get('passed'):
            continue
        end = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        end = (end if end.tzinfo else end.replace(tzinfo=UTC)).timestamp()
        out.append((end - d['duration_ms'] / 1000, end))
    return out


def single_thread_table(laptop: Host, pods: list[Host]) -> list[str]:
    hosts = [laptop, *pods]
    rows = [
        ('pytest collection, orchestrator tests (CPython imports + AST)', lambda h: h.probe('py_collect')),
        ('same, first set (pods: right after GBs of writes)', lambda h: median_wall(h.first, 'py_collect', 1)),
        ('pyright, orchestrator/src (Node, single-threaded)', lambda h: h.probe('pyright')),
        ('same, first set', lambda h: median_wall(h.first, 'pyright', 1)),
        ('pure CPython loop (no I/O)', lambda h: h.diag_median('interp_loop')),
        ('pytest collection, 10 interleaved rounds (laptop: its probe median)',
         lambda h: h.sample_median('py_collect') if h is not laptop else h.probe('py_collect')),
        ('pyright, 10 interleaved rounds (laptop: its probe median)',
         lambda h: h.sample_median('pyright') if h is not laptop else h.probe('pyright')),
        ('cold reify-compiler build, -j1, no sccache', lambda h: h.probe('rustc_j1')),
    ]
    out = ['| one thread, median seconds | ' + ' | '.join(h.name for h in hosts) + ' |',
           '|---|' + '---|' * len(hosts)]
    for label, get in rows:
        lap = get(laptop)
        cells = [fmt(lap)] + [f'{fmt(get(p))} ({ratio(lap, get(p))})' for p in pods]
        out.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    return out


def full_load_table(laptop: Host, pods: list[Host]) -> list[str]:
    hosts = [laptop, *pods]
    out = ['| every thread busy | ' + ' | '.join(h.name for h in hosts) + ' |', '|---|' + '---|' * len(hosts)]
    for probe in ('py_collect', 'pyright'):
        per_copy = {h.name: h.probe(probe, k=None) for h in hosts}
        out.append(f'| {probe}: seconds per copy | ' + ' | '.join(fmt(per_copy[h.name]) for h in hosts) + ' |')
        out.append(f'| {probe}: copies/hour per hardware thread | ' + ' | '.join(
            f'{3600 / per_copy[h.name]:.0f}' if per_copy[h.name] else '-' for h in hosts) + ' |')
    rows = [('cold 5-crate reify build, -j<threads>, no sccache', lambda h: h.probe('rustc_jn'))]
    for label, get in rows:
        out.append(f'| {label} | ' + ' | '.join(fmt(get(h)) for h in hosts) + ' |')
    for label, field in (('max MHz of a busy core (1-copy batches)', 'mhz_max_allowed'),
                         ('host 1-min load during 1-copy batches (pods: whole host, other tenants)', 'host_load1_mean')):
        values = [batch_field(h.rerun or h.first, 'py_collect', 1, field) for h in hosts]
        out.append(f'| {label} | ' + ' | '.join(f'{v:.0f}' if v else '-' for v in values) + ' |')
    return out


def gate_lines(laptop_gate: float, pods: list[Host]) -> list[str]:
    median, n = laptop_df_gate_uncontended()
    out = [f'- laptop, same SHA, production record (n=1, no reify gate alongside): {laptop_gate:.0f} s, '
           f'{3600 / laptop_gate:.1f} gates/hour (one at a time, 16 threads, fsync 0.55 ms)',
           f'- laptop, all SHAs 10-12..10-22 with no reify gate alongside: median {median:.0f} s (n={n})']
    for pod in pods:
        for batch in pod.gates:
            walls = [g['wall_s'] for g in batch['gates']]
            partial = any('partial' in (g.get('summary') or '') for g in batch['gates'])
            wrapper = batch.get('wrapper')
            label = batch.get('label') or f"gate-c{batch['concurrency']}" + (f'-{wrapper}' if wrapper else '')
            out.append(
                f"- {pod.name} [{label}{', PARTIAL tally' if partial else ''}]: {batch['concurrency']} at once, "
                f"mean {statistics.mean(walls):.0f} s per gate (ratio {laptop_gate / statistics.mean(walls):.2f}×), "
                f"{batch['concurrency'] * 3600 / batch['makespan_s']:.1f} gates/hour on {pod.info.get('cpu_budget')} vCPU"
            )
    return out


def reify_steps(records: list[dict], head: str) -> dict[str, float]:
    """Wall seconds per (phase, step index) of the LAST test+lint record pair for <head>."""
    runs = [r for r in records if r.get('head') == head]
    totals: dict[str, float] = {}
    for run in runs[-2:]:
        for step in run.get('steps', []):
            key = step['phase']
            totals[key] = totals.get(key, 0.0) + (step.get('wall_secs') or 0.0)
    return totals


def laptop_df_gate_intervals() -> list[tuple[float, float]]:
    """(start, end) epoch seconds of every DF merge gate the laptop ran."""
    db = sqlite3.connect(f'file:{DF_RUNS_DB}?mode=ro', uri=True)
    rows = db.execute("select timestamp, data from events where event_type = 'merge_verify' and data like '%\"runner\": \"laptop\"%'")
    intervals = []
    for ts, data in rows:
        end = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        end = (end if end.tzinfo else end.replace(tzinfo=UTC)).timestamp()
        intervals.append((end - json.loads(data)['duration_ms'] / 1000, end))
    return intervals


def laptop_uncontended_reify() -> tuple[dict[str, float], int, int]:
    """Per-phase median wall over the laptop's merge-role reify records (test and lint legs,
    every commit in the ledger) whose whole span overlapped no laptop DF merge gate."""
    df = laptop_df_gate_intervals()
    per_leg_phase: dict[tuple[str, str], list[float]] = {}
    clean = {'test': 0, 'lint': 0}
    for run in jsonl(LAPTOP_REIFY_LEDGER):
        steps = run.get('steps', [])
        if run.get('role') != 'merge' or run.get('action') not in clean or not steps:
            continue
        if any(s.get('rc') not in (0, None) for s in steps) or any(s.get('rc') is None and s.get('wall_secs') for s in steps):
            continue
        timed = [s for s in steps if s.get('began_at') and s.get('ended_at')]
        lo, hi = min(s['began_at'] for s in timed), max(s['ended_at'] for s in timed)
        if any(min(hi, e) > max(lo, b) for b, e in df):
            continue
        clean[run['action']] += 1
        sums: dict[str, float] = {}
        for s in steps:
            sums[s['phase']] = sums.get(s['phase'], 0.0) + (s.get('wall_secs') or 0.0)
        for phase, secs in sums.items():
            per_leg_phase.setdefault((run['action'], phase), []).append(secs)
    baseline: dict[str, float] = {}
    for (_, phase), values in per_leg_phase.items():
        baseline[phase] = baseline.get(phase, 0.0) + statistics.median(values)
    return baseline, clean['test'], clean['lint']


def reify_table(pods: list[Host]) -> list[str]:
    contended = reify_steps(jsonl(LAPTOP_REIFY_STEPS), REIFY_WARM_SHA)
    laptop, n_test, n_lint = laptop_uncontended_reify()
    pod_steps = {p.name: reify_steps(p.reify_steps, REIFY_WARM_SHA) for p in pods}
    phases = [ph for ph in laptop if ph not in REIFY_EXCLUDED_PHASES and ph != 'bookkeeping']
    out = [f'| reify warm gate phase (seconds) | laptop, uncontended median (n={n_test} test + {n_lint} lint legs, other SHAs) '
           f'| laptop at 57be3b937f (DF gate running alongside) | ' + ' | '.join(pod_steps) + ' |',
           '|---|---|---|' + '---|' * len(pods)]
    for ph in (ph for ph in phases if max(laptop[ph], contended.get(ph, 0)) >= 1):
        out.append(f'| {ph} | {laptop[ph]:.0f} | {contended.get(ph, 0):.0f} | ' + ' | '.join(
            f"{pod_steps[n].get(ph, 0):.0f} ({ratio(laptop[ph], pod_steps[n].get(ph))})" for n in pod_steps) + ' |')
    lap_total = sum(laptop[ph] for ph in phases)
    out.append(f'| **sum of compared phases** | {lap_total:.0f} | {sum(contended.get(ph, 0) for ph in phases):.0f} | ' + ' | '.join(
        f"{sum(pod_steps[n].get(ph, 0) for ph in phases):.0f} ({ratio(lap_total, sum(pod_steps[n].get(ph, 0) for ph in phases))})"
        for n in pod_steps) + ' |')
    out.append(f"| excluded on pods (laptop only: {', '.join(sorted(REIFY_EXCLUDED_PHASES))}) | "
               f"{sum(v for k, v in laptop.items() if k in REIFY_EXCLUDED_PHASES):.0f} | "
               f"{sum(v for k, v in contended.items() if k in REIFY_EXCLUDED_PHASES):.0f} | " + ' | '.join('-' for _ in pods) + ' |')
    for pod in pods:
        for run in pod.reify:
            out.append(f"- {pod.name} reify {run['run']} gate at {run['sha'][:10]}: {run['wall_s']:.0f} s wall, "
                       f"passed={run['passed']}, wrapper={run.get('wrapper') or 'none'}")
    return out


RULES = """
RULES
- one thread: median wall of single-copy batches with rc 0; pods use probes.rerun.jsonl (quiet disk) when present,
  else the first set; the laptop's set ran under wait-for-quiet. diag rows: median of diag.jsonl reps.
  '10 interleaved rounds': st-samples.jsonl, py_collect/pyright/interp alternated 10x on the Milan pod, host load
  28-74 (the first set ran at host load ~100, the rerun at ~130: other tenants on the 256-thread host).
  pyright is not strictly single-threaded (Node GC threads: user CPU ~1.7x wall).
- every thread busy: per-copy median wall at the top K level (laptop K=16, pods K=32 = their cpuset);
  copies/hour per hardware thread = 3600 / per-copy seconds.
- DF gates: wall of `orchestrator verify-merge` per gate; gates/hour = concurrency * 3600 / makespan.
  'eatmydata' = fsync turned into a no-op (pod disks differ: Milan 2.2 ms, Turin 126 ms per 4 KiB dsync);
  'deselect' = the live spec minus one fused-memory test that hangs 300 s on some hosts.
  Pod gates run as root: ~65 permission/host-fixture tests fail fast there and pass on the laptop.
- reify: per-phase sums of verify.sh's own step records (test + lint legs). Pods: their warm gate at 57be3b937f.
  Laptop baseline: per-leg, per-phase medians over every merge-role record in its ledger (other SHAs, 10-14..10-22)
  whose whole span overlapped no laptop DF merge gate (runs.db); the laptop's own 57be3b937f record is shown beside
  it but ran alongside a DF gate (~1.5x slower nextest). Phases under 1 s are summed but not listed. The pods omit
  the infra pool and its release prebuilds (REIFY_INFRA_SUITE_ACTIVE=1, verify.sh's nested-suite guard).
"""


def main(laptop_dir: str, *pod_dirs: str) -> None:
    laptop = Host(Path(laptop_dir))
    pods = [Host(Path(d)) for d in pod_dirs]
    lines = ['## Per-thread speed', '', *single_thread_table(laptop, pods), '',
             '## With every thread busy', '', *full_load_table(laptop, pods), '',
             '## Real dark-factory merge gates', '', *gate_lines(laptop_df_gate(), pods), '',
             '## Real reify merge gate, phase by phase', '', *reify_table(pods), '', RULES]
    print('\n'.join(lines))


if __name__ == '__main__':
    main(*sys.argv[1:])
