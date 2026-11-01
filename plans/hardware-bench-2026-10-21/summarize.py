"""Summarise a bench.sh results dir, against the latest reference run if one exists.

    python3 summarize.py <results dir> [<reference results dir>]

Ratios are reference time / this host's time: above 1 means this host is faster.
"""

import json
import sqlite3
import statistics
import sys
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parent / 'results'
RUNS_DB = Path('/home/leo/src/dark-factory/data/orchestrator/runs.db')
PROBES = ('py_collect', 'pyright')
COMPILES = ('rustc_j1', 'rustc_jn')


def load_run(run_dir: Path) -> dict:
    host = json.loads((run_dir / 'host.json').read_text())
    probes = [json.loads(line) for line in (run_dir / 'probes.jsonl').read_text().splitlines()]
    gates_file = run_dir / 'gates.jsonl'
    gates = [json.loads(line) for line in gates_file.read_text().splitlines()] if gates_file.exists() else []
    return {'host': host, 'probes': probes, 'gates': gates}


def latest_reference(exclude: Path) -> Path | None:
    candidates = [
        d for d in sorted(RESULTS_ROOT.glob('*'))
        if d != exclude and (d / 'DONE').exists()
        and json.loads((d / 'host.json').read_text())['mode'] == 'reference'
    ]
    return candidates[-1] if candidates else None


def per_copy_seconds(run: dict, probe: str, k: int) -> tuple[float, int, bool] | None:
    """Median wall of one copy at concurrency k, the batch's mean MHz, and whether a foreign verify ran."""
    batches = [b for b in run['probes'] if b['probe'] == probe and b['k'] == k]
    if not batches:
        return None
    walls = [r['wall_s'] for b in batches for r in b['runs']]
    mhz = [b['mhz_mean'] for b in batches if b['mhz_mean']]
    return (
        statistics.median(walls),
        round(statistics.mean(mhz)) if mhz else 0,
        any(b['foreign_verify_seen'] for b in batches),
    )


def full_load_k(run: dict, probe: str) -> int:
    return max(b['k'] for b in run['probes'] if b['probe'] == probe)


def cell(value: tuple[float, int, bool] | None) -> str:
    if value is None:
        return '-'
    seconds, mhz, contaminated = value
    return f'{seconds:.1f} s @ {mhz} MHz' + (' *' if contaminated else '')


def ratio(ref: tuple | None, this: tuple | None) -> str:
    return f'{ref[0] / this[0]:.2f}×' if ref and this else '-'


def describe(host: dict) -> str:
    return (f"{host['hostname']}: {host['cpu_model']}, {host['sockets']} socket(s), "
            f"{host['cores']} cores / {host['threads']} threads, {host['mem_gib']} GiB")


def probe_table(run: dict, ref: dict | None) -> list[str]:
    lines = [
        '| probe | 1 copy | per copy, k = cores | per copy, k = threads |'
        + (' ratio 1 copy | ratio all threads busy |' if ref else ''),
        '|---|---|---|---|' + ('---|---|' if ref else ''),
    ]
    for probe in PROBES:
        if not any(b['probe'] == probe for b in run['probes']):
            continue
        one = per_copy_seconds(run, probe, 1)
        cores = per_copy_seconds(run, probe, min(run['host']['cores'], full_load_k(run, probe)))
        full = per_copy_seconds(run, probe, full_load_k(run, probe))
        row = f'| {probe} | {cell(one)} | {cell(cores)} | {cell(full)} (k={full_load_k(run, probe)}) |'
        if ref:
            ref_one = per_copy_seconds(ref, probe, 1)
            ref_full = per_copy_seconds(ref, probe, full_load_k(ref, probe))
            row += f' {ratio(ref_one, one)} | {ratio(ref_full, full)} |'
        lines.append(row)
    return lines


def box_throughput(run: dict, probe: str) -> float:
    """Copies per hour with every thread busy: the whole-box capacity measure."""
    k = full_load_k(run, probe)
    batch = next(b for b in run['probes'] if b['probe'] == probe and b['k'] == k)
    return k * 3600 / batch['makespan_s']


def compile_lines(run: dict, ref: dict | None) -> list[str]:
    lines = []
    for probe in COMPILES:
        this = per_copy_seconds(run, probe, 1)
        if this is None:
            continue
        text = f'- {probe}: {cell(this)}'
        if ref:
            text += f'; reference {cell(per_copy_seconds(ref, probe, 1))}; ratio {ratio(per_copy_seconds(ref, probe, 1), this)}'
        lines.append(text)
    return lines


def laptop_gate_seconds(sha: str) -> list[float]:
    if not RUNS_DB.exists():
        return []
    db = sqlite3.connect(f'file:{RUNS_DB}?mode=ro', uri=True)
    rows = db.execute(
        "select data from events where event_type = 'merge_verify' and data like ?", (f'%{sha}%',),
    ).fetchall()
    events = [json.loads(data) for (data,) in rows]
    return [e['duration_ms'] / 1000 for e in events if e.get('runner') == 'laptop' and e.get('passed')]


def gate_lines(run: dict) -> list[str]:
    if not run['gates']:
        return []
    lines = []
    for batch in run['gates']:
        n = batch['concurrency']
        walls = [g['wall_s'] for g in batch['gates']]
        verdicts = ', '.join('pass' if g['passed'] else f"FAIL (rc {g['rc']})" for g in batch['gates'])
        lines.append(
            f'- {n} at once: mean {statistics.mean(walls):.0f} s per gate, makespan {batch["makespan_s"]:.0f} s, '
            f'{n * 3600 / batch["makespan_s"]:.1f} gates/hour; verdicts: {verdicts}'
        )
    if any(not g['passed'] for b in run['gates'] for g in b['gates']):
        lines.append('- a gate FAILED: its wall time is not a full gate; read gate-c*/stderr.*.log before comparing')
    laptop = laptop_gate_seconds(run['host']['df_sha'])
    solo = next((b for b in run['gates'] if b['concurrency'] == 1), None)
    if laptop and solo and not run['host'].get('quick'):
        single = solo['gates'][0]['wall_s']
        lines.append(
            f'- laptop, same sha (production record): {statistics.median(laptop):.0f} s, '
            f'{3600 / statistics.median(laptop):.1f} gates/hour; one-gate ratio {statistics.median(laptop) / single:.2f}×'
        )
    return lines


REIFY_RUNS_DB = Path('/home/leo/src/reify/data/orchestrator/runs.db')
LAPTOP_REIFY_STEPS = Path(__file__).resolve().parent / 'laptop-reify-verify-runs.jsonl'


def laptop_reify_gate_seconds(sha: str) -> list[float]:
    """The laptop's passed production merge gates of reify at *sha*, from reify's runs.db."""
    if not REIFY_RUNS_DB.exists():
        return []
    db = sqlite3.connect(f'file:{REIFY_RUNS_DB}?mode=ro', uri=True)
    rows = db.execute(
        "select data from events where event_type = 'merge_verify' and data like ?", (f'%{sha}%',),
    ).fetchall()
    events = [json.loads(data) for (data,) in rows]
    return [
        e['duration_ms'] / 1000 for e in events
        if e.get('runner') == 'laptop' and e.get('passed') and e.get('merge_sha') == sha
    ]


def phase_seconds(verify_runs: Path, sha: str) -> dict[str, float]:
    """Wall seconds per verify.sh plan phase, summed over the test and lint actions of the gate at *sha*."""
    if not verify_runs.exists():
        return {}
    totals: dict[str, float] = {}
    for line in verify_runs.read_text().splitlines():
        record = json.loads(line)
        if record.get('head') != sha or record.get('role') != 'merge':
            continue
        for step in record['steps']:
            totals[step['phase']] = totals.get(step['phase'], 0.0) + step['wall_secs']
    return totals


def reify_phase_lines(run_dir: Path, sha: str) -> list[str]:
    pod, laptop = phase_seconds(run_dir / 'reify-verify-runs.jsonl', sha), phase_seconds(LAPTOP_REIFY_STEPS, sha)
    if not pod or not laptop:
        return []
    lines = ['', '| verify.sh phase (warm gate) | this host | laptop | ratio |', '|---|---|---|---|']
    for phase in sorted(set(pod) | set(laptop), key=lambda p: -laptop.get(p, 0.0)):
        if max(pod.get(phase, 0.0), laptop.get(phase, 0.0)) < 1:
            continue
        this, ref = pod.get(phase), laptop.get(phase)
        ratio_text = f'{ref / this:.2f}×' if this and ref else '-'
        lines.append(f'| {phase} | {this or 0:.0f} s | {ref or 0:.0f} s | {ratio_text} |')
    return lines


def reify_gate_lines(run_dir: Path) -> list[str]:
    gates_file = run_dir / 'reify_gates.jsonl'
    if not gates_file.exists():
        return []
    gates = [json.loads(line) for line in gates_file.read_text().splitlines()]
    lines = []
    for g in gates:
        stats = g.get('sccache_stats') or {}
        hits, misses = stats.get('cache_hits', {}).get('counts', {}), stats.get('cache_misses', {}).get('counts', {})
        sccache = f'sccache {sum(hits.values())} hits / {sum(misses.values())} misses' if stats else 'no sccache stats'
        verdict = 'pass' if g['passed'] else f"FAIL (rc {g['rc']}, timed_out {g['timed_out']})"
        lines.append(f"- {g['run']} at {g['sha'][:10]}: {g['wall_s']:.0f} s, {verdict}, {sccache}")
    if any(not g['passed'] for g in gates):
        lines.append('- a gate FAILED: its wall time is not a full gate; read reify-gate-*.stderr.log before comparing')
    warm = next((g for g in reversed(gates) if g['run'] == 'warm'), None)
    laptop = laptop_reify_gate_seconds(warm['sha']) if warm else []
    if warm and laptop:
        lines.append(
            f"- laptop, same sha (production record): {statistics.median(laptop):.0f} s; "
            f"warm-gate ratio {statistics.median(laptop) / warm['wall_s']:.2f}× (above 1: this host is faster)"
        )
    if warm:
        lines += reify_phase_lines(run_dir, warm['sha'])
    return lines


def main(run_dir: str, reference_dir: str | None = None) -> None:
    run_path = Path(run_dir).resolve()
    run = load_run(run_path)
    ref_path = Path(reference_dir).resolve() if reference_dir else latest_reference(run_path)
    ref = load_run(ref_path) if ref_path and run['host']['mode'] != 'reference' else None

    out = [f'## {describe(run["host"])}' + (' (smoke run)' if run['host'].get('quick') else ''), '']
    if ref:
        out += [f'Reference: {describe(ref["host"])}', '']
    out += ['### Per-thread probes (median wall per copy)', '', *probe_table(run, ref), '']
    out += ['`*` a foreign verify ran during the batch (reference hosts only).', '']
    if ref:
        out += ['### Whole-box capacity (copies per hour, every thread busy)', '']
        for probe in PROBES:
            this, other = box_throughput(run, probe), box_throughput(ref, probe)
            out.append(f'- {probe}: {this:.0f} vs reference {other:.0f} ({this / other:.1f}×)')
        out.append('')
    out += ['### Cold reify compiles', '', *compile_lines(run, ref), '']
    gates = gate_lines(run)
    if gates:
        out += ['### Real dark-factory merge gates', '', *gates, '']
    reify_gates = reify_gate_lines(run_path)
    if reify_gates:
        out += ['### Real reify merge gates', '', *reify_gates, '']
    print('\n'.join(out))


if __name__ == '__main__':
    main(*sys.argv[1:])
