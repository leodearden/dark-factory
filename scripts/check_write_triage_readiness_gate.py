#!/usr/bin/env python3
"""Predicate for the write-triage flip-readiness gates (plans/write-triage-flip-readiness-prd.md).

Reads JSON reports, evaluates requirements against dotted keys, optionally runs
sub-check programs, and decides by exit code alone: 0 when every requirement
holds, 1 otherwise. The exit code is the machine contract; the branch actions
(`--on-pass`, `--on-fail`) are advisory prose for the operator. The last line
of stdout is a compact JSON verdict (gate, verdict, failed checks, actions)
bounded to the DeterministicRunner's 400-char provenance note; the line before
it is the full verdict with every observed value, which the runner logs and a
milestone_check_failed escalation carries in its stdout tail.

Every input is structured argv, never a mini-language:
  --report  NAME PATH
  --require NAME PATH OP VALUE        OP in >= <= > < ==   (numbers only; a bool never satisfies)
  --equals  NAME PATH VALUE           string equality
  --subcheck 'PROGRAM ARG ...'        one string, shlex-tokenized to argv, no shell; exit 0 required
A PATH is dotted; a segment may select a list element by index (`per_k[3]`)
or by key match (`per_k[k=20]`). Every malformed or missing input fails closed.
"""
from __future__ import annotations

import argparse
import json
import operator
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

NOTE_CAP = 400
_OPS = {'>=': operator.ge, '<=': operator.le, '>': operator.gt, '<': operator.lt, '==': operator.eq}
_SEGMENT = re.compile(r'^(?P<key>[^\[\]]+)(?:\[(?P<sel>[^\]]+)\])?$')


class Missing(Exception):
    pass


def resolve(doc: Any, path: str) -> Any:
    node = doc
    for raw in path.split('.'):
        seg = _SEGMENT.match(raw)
        if seg is None:
            raise Missing(f'malformed path segment {raw!r}')
        key, sel = seg.group('key'), seg.group('sel')
        if not isinstance(node, dict) or key not in node:
            raise Missing(f'key {key!r} absent')
        node = node[key]
        if sel is None:
            continue
        if not isinstance(node, list):
            raise Missing(f'{key!r} is not a list')
        node = _select(node, sel, key)
    return node


def _select(items: list, sel: str, key: str) -> Any:
    if sel.isdigit():
        index = int(sel)
        if index >= len(items):
            raise Missing(f'{key}[{index}] out of range (len {len(items)})')
        return items[index]
    field, _, wanted = sel.partition('=')
    for item in items:
        if isinstance(item, dict) and str(item.get(field)) == wanted:
            return item
    raise Missing(f'no element of {key!r} has {field}={wanted}')


def numeric_reason(actual: Any) -> str | None:
    if isinstance(actual, bool):
        return 'bool is not a number'
    if not isinstance(actual, (int, float)):
        return 'not a number'
    return None


def check_require(reports: dict[str, Any], handle: str, path: str, op: str, value: str) -> dict[str, Any]:
    label = f'{handle}:{path} {op} {value}'
    if op not in _OPS:
        return {'check': label, 'actual': None, 'ok': False, 'note': f'unknown operator {op!r}'}
    try:
        wanted = float(value)
        actual = resolve(reports[handle], path)
    except (ValueError, KeyError, Missing) as exc:
        return {'check': label, 'actual': None, 'ok': False, 'note': str(exc)}
    reason = numeric_reason(actual)
    ok = reason is None and _OPS[op](actual, wanted)
    return {'check': label, 'actual': actual, 'ok': bool(ok), 'note': reason}


def check_equals(reports: dict[str, Any], handle: str, path: str, value: str) -> dict[str, Any]:
    label = f'{handle}:{path} = {value}'
    try:
        actual = resolve(reports[handle], path)
    except (KeyError, Missing) as exc:
        return {'check': label, 'actual': None, 'ok': False, 'note': str(exc)}
    return {'check': label, 'actual': actual, 'ok': str(actual) == value, 'note': None}


def check_subcheck(command: str, timeout: int) -> dict[str, Any]:
    label = f'subcheck {command}'
    try:
        proc = subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {'check': label, 'actual': 'timeout', 'ok': False, 'note': f'exceeded {timeout}s'}
    except (OSError, ValueError) as exc:
        return {'check': label, 'actual': 'unrunnable', 'ok': False, 'note': str(exc)}
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
    return {'check': label, 'actual': proc.returncode, 'ok': proc.returncode == 0, 'note': ' | '.join(tail)}


def load_reports(specs: list[list[str]]) -> dict[str, Any]:
    return {handle: json.loads(Path(path).read_text()) for handle, path in specs}


def compact_verdict(verdict: dict[str, Any]) -> dict[str, Any]:
    failed = [c['check'] for c in verdict['checks'] if not c['ok']]
    compact = {'gate': verdict['gate'], 'verdict': verdict['verdict'], 'failed': failed,
               'actions': verdict['actions'][:160], **({'error': verdict['error']} if 'error' in verdict else {})}
    while len(json.dumps(compact, sort_keys=True)) > NOTE_CAP and compact['failed']:
        compact['failed'] = compact['failed'][:-1] + ['…'] if compact['failed'][-1] != '…' else compact['failed'][:-2] + ['…']
    return compact


def emit(verdict: dict[str, Any]) -> int:
    for c in verdict['checks']:
        extra = f' ({c["note"]})' if c['note'] else ''
        print(f'{"PASS" if c["ok"] else "FAIL"}  {c["check"]}  actual={c["actual"]!r}{extra}')
    print(json.dumps(verdict, sort_keys=True))
    print(json.dumps(compact_verdict(verdict), sort_keys=True))
    return 0 if verdict['verdict'] == 'pass' else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gate', required=True)
    parser.add_argument('--report', action='append', default=[], nargs=2, metavar=('NAME', 'PATH'))
    parser.add_argument('--require', action='append', default=[], nargs=4, metavar=('NAME', 'PATH', 'OP', 'VALUE'))
    parser.add_argument('--equals', action='append', default=[], nargs=3, metavar=('NAME', 'PATH', 'VALUE'))
    parser.add_argument('--subcheck', action='append', default=[], metavar='COMMAND')
    parser.add_argument('--subcheck-timeout', type=int, default=90)
    parser.add_argument('--on-pass', default='')
    parser.add_argument('--on-fail', default='')
    args = parser.parse_args(argv)

    try:
        reports = load_reports(args.report)
    except (OSError, ValueError) as exc:
        return emit({'gate': args.gate, 'verdict': 'fail', 'checks': [], 'error': f'report unreadable: {exc}', 'actions': args.on_fail})

    checks = [check_require(reports, *r) for r in args.require]
    checks += [check_equals(reports, *e) for e in args.equals]
    checks += [check_subcheck(s, args.subcheck_timeout) for s in args.subcheck]
    passed = bool(checks) and all(c['ok'] for c in checks)
    return emit({'gate': args.gate, 'verdict': 'pass' if passed else 'fail', 'checks': checks,
                 'actions': args.on_pass if passed else args.on_fail,
                 **({} if checks else {'error': 'no checks given'})})


if __name__ == '__main__':
    sys.exit(main())
