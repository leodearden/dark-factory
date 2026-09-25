"""Tests for scripts/check_write_triage_readiness_gate.py — the flip-readiness gate predicate.

Assertions are on exit codes and on the two trailing JSON lines (full verdict,
then compact verdict), never on the human lines. Fixtures are stdlib-only JSON
files written per test; sub-checks run the test interpreter via shlex-tokenized argv, never a shell.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'check_write_triage_readiness_gate.py'
NOTE_CAP = 400


def run(tmp_path: Path, *args: str) -> tuple[int, dict, dict]:
    proc = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, cwd=tmp_path)
    lines = proc.stdout.strip().splitlines()
    assert lines, proc.stderr
    compact, verdict = json.loads(lines[-1]), json.loads(lines[-2])
    assert compact['verdict'] == verdict['verdict'] and compact['gate'] == verdict['gate']
    assert len(lines[-1]) <= NOTE_CAP
    return proc.returncode, verdict, compact


def write(tmp_path: Path, name: str, doc: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return str(path)


REPORT = {
    'provenance': {'slate_mode': 'retrieved', 'flag': True},
    'production_shape': {'duplicate_attach': {'n': 75, 'strict_rate': 0.16}, 'text': '0.9'},
    'recall_at_k': {'per_k': [{'k': 5, 'recall': 0.38}, {'k': 20, 'recall': 0.7857}]},
}


def test_every_requirement_holding_exits_zero_with_on_pass_actions(tmp_path):
    p = write(tmp_path, 'r.json', REPORT)
    rc, v, c = run(tmp_path, '--gate', 'G', '--report', 'j', p,
                   '--require', 'j', 'production_shape.duplicate_attach.strict_rate', '>=', '0.1',
                   '--require', 'j', 'recall_at_k.per_k[k=20].recall', '>=', '0.7',
                   '--require', 'j', 'recall_at_k.per_k[1].k', '==', '20',
                   '--equals', 'j', 'provenance.slate_mode', 'retrieved',
                   '--on-pass', 'dispatch rho2', '--on-fail', 'cancel rho2')
    assert rc == 0 and v['verdict'] == 'pass' and v['actions'] == 'dispatch rho2'
    assert len(v['checks']) == 4 and all(x['ok'] for x in v['checks']) and c['failed'] == []


def test_one_failing_threshold_exits_one_with_on_fail_actions(tmp_path):
    p = write(tmp_path, 'r.json', REPORT)
    rc, v, c = run(tmp_path, '--gate', 'G', '--report', 'j', p,
                   '--require', 'j', 'production_shape.duplicate_attach.strict_rate', '>=', '0.5',
                   '--equals', 'j', 'provenance.slate_mode', 'retrieved',
                   '--on-pass', 'x', '--on-fail', 'remove the mu<-rho2 edge, cancel rho2')
    assert rc == 1 and v['actions'] == 'remove the mu<-rho2 edge, cancel rho2'
    failed = [x for x in v['checks'] if not x['ok']]
    assert len(failed) == 1 and failed[0]['actual'] == 0.16 and c['failed'] == [failed[0]['check']]


def test_an_equals_mismatch_fails_closed(tmp_path):
    p = write(tmp_path, 'r.json', {**REPORT, 'provenance': {'slate_mode': 'seeded'}})
    rc, v, _ = run(tmp_path, '--gate', 'G', '--report', 'j', p, '--equals', 'j', 'provenance.slate_mode', 'retrieved')
    assert rc == 1 and v['checks'][0]['actual'] == 'seeded' and not v['checks'][0]['ok']


def test_missing_key_and_missing_selector_fail_closed_and_name_the_gap(tmp_path):
    p = write(tmp_path, 'r.json', REPORT)
    rc, v, _ = run(tmp_path, '--gate', 'G', '--report', 'j', p,
                   '--require', 'j', 'production_shape.wrong_record_attach.rate_of_attaches', '<=', '0.25',
                   '--require', 'j', 'recall_at_k.per_k[k=50].recall', '>=', '0',
                   '--require', 'j', 'recall_at_k.per_k[7].recall', '>=', '0',
                   '--require', 'j', 'provenance.slate_mode[0]', '>=', '0')
    assert rc == 1 and not any(x['ok'] for x in v['checks'])
    notes = [x['note'] for x in v['checks']]
    assert 'wrong_record_attach' in notes[0] and 'k=50' in notes[1] and 'out of range' in notes[2] and 'not a list' in notes[3]


def test_bool_string_bad_operator_and_bad_value_never_satisfy_a_numeric_requirement(tmp_path):
    p = write(tmp_path, 'r.json', REPORT)
    rc, v, _ = run(tmp_path, '--gate', 'G', '--report', 'j', p,
                   '--require', 'j', 'provenance.flag', '>=', '0',
                   '--require', 'j', 'production_shape.text', '>=', '0',
                   '--require', 'j', 'production_shape.duplicate_attach.n', '=>', '0',
                   '--require', 'j', 'production_shape.duplicate_attach.n', '>=', 'seventy')
    assert rc == 1 and not any(x['ok'] for x in v['checks'])
    assert v['checks'][0]['note'] == 'bool is not a number' and v['checks'][1]['note'] == 'not a number'
    assert 'operator' in v['checks'][2]['note'] and 'seventy' in v['checks'][3]['note']


def test_an_unreadable_report_and_an_empty_check_list_fail_closed(tmp_path):
    rc, v, c = run(tmp_path, '--gate', 'G', '--report', 'j', str(tmp_path / 'absent.json'),
                   '--require', 'j', 'a', '>=', '0', '--on-fail', 'fix the artifact')
    assert rc == 1 and 'unreadable' in v['error'] and c['error'] == v['error'] and v['actions'] == 'fix the artifact'
    p = write(tmp_path, 'r.json', REPORT)
    rc, v, _ = run(tmp_path, '--gate', 'G', '--report', 'j', p)
    assert rc == 1 and v['error'] == 'no checks given'


def test_a_subcheck_is_argv_and_its_exit_code_is_a_requirement(tmp_path):
    p = write(tmp_path, 'r.json', REPORT)
    rc, _, _ = run(tmp_path, '--gate', 'G', '--report', 'j', p, '--subcheck', f'{sys.executable} -c pass')
    assert rc == 0
    rc, v, _ = run(tmp_path, '--gate', 'G', '--report', 'j', p,
                   '--subcheck', f'{sys.executable} -c "print(\'boom\'); raise SystemExit(3)"')
    assert rc == 1 and v['checks'][0]['actual'] == 3 and 'boom' in v['checks'][0]['note']
    rc, v, _ = run(tmp_path, '--gate', 'G', '--report', 'j', p, '--subcheck', str(tmp_path / 'no-such-program'))
    assert rc == 1 and v['checks'][0]['actual'] == 'unrunnable'
    rc, v, _ = run(tmp_path, '--gate', 'G', '--report', 'j', p, '--subcheck', 'echo "unterminated')
    assert rc == 1 and v['checks'][0]['actual'] == 'unrunnable'


def test_the_compact_line_stays_under_the_note_cap_however_many_checks_fail(tmp_path):
    p = write(tmp_path, 'r.json', REPORT)
    args = ['--gate', 'Gamma-3', '--report', 'j', p, '--on-fail', 'y' * 500]
    for i in range(8):
        args += ['--require', 'j', f'production_shape.some_very_long_metric_name_number_{i}.rate_of_attaches', '<=', '0.25']
    args += ['--subcheck', f'{sys.executable} -c "raise SystemExit(1)"']
    rc, v, c = run(tmp_path, *args)
    assert rc == 1 and len(v['checks']) == 9 and len(c['actions']) == 160
    assert c['failed'][-1] == '…' and c['gate'] == 'Gamma-3'
