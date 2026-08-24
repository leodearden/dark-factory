#!/usr/bin/env python3
"""Member-chain sweep: find pending L2 escalations whose members were already ruled.

The answered-but-unrecorded class (INV-9 `one-fact-one-home`; measured
2026-08-22..24: five records, 30.2 answered-yet-open days): one L1 gets
promoted to L2 more than once, one promotion is ruled, `queue.resolve()`
cascades DOWN to the shared member but never SIDEWAYS to the other L2s
built on it — so an answered question keeps a task pinned indefinitely.
Every measured instance shared this fingerprint: the surviving record and
the record carrying the ruling had the identical single member id.

For each pending level-2 record with members, this script locates every
member ACROSS THE ARCHIVE (a ruled member is archived by definition) and
reports members in a terminal status, classified by their resolution TEXT
(status alone is useless -- both shapes below read `dismissed`):

- RULED        -- substantive ruling text: the L2 is probably ANSWERED.
                  Recover the ruling from the record named in the member's
                  `resolved_by` (`l2-cascade:<id>`) rather than re-deriving.
- DEDUP-MARKER -- "DUPLICATE of esc-X (survivor, stays open)": the
                  OPPOSITE -- a dedup pass deliberately kept this record
                  live; the question may be genuinely open.

REPORT-ONLY, strictly read-only. Never close anything off this output:
a record can be a deliberately-preserved PIN whose value is its existence
(esc-3105-3 scores 15/15 ruled members and must NOT be closed -- see
skills/escalation-watcher/SKILL.md "Ruled-elsewhere check"). Until task
4377 lands `pin_declared_by`, pins and answered questions are
indistinguishable from the member chain alone.

Usage:
    member-chain-sweep.py                 # sweep the default fleet queues
    member-chain-sweep.py <queue-dir>...  # sweep only the named queue dirs
"""

from __future__ import annotations

import glob
import json
import os
import sys

DEDUP_MARKER = 'survivor, stays open'
TERMINAL_STATUSES = ('resolved', 'dismissed')


def _load(path: str) -> dict | None:
    try:
        with open(path, encoding='utf-8') as f:
            record = json.load(f)
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def sweep_queue(queue_dir: str) -> int:
    """Report terminal members of pending L2s in one queue. Returns hit count."""
    pending: list[tuple[str, dict]] = []
    for path in sorted(glob.glob(os.path.join(queue_dir, 'esc-*.json'))):
        record = _load(path)
        if (
            record is not None
            and record.get('status') == 'pending'
            and record.get('level') == 2
            and record.get('members')
        ):
            pending.append((path, record))

    findings = 0
    lines: list[str] = []
    for path, record in pending:
        hits: list[str] = []
        for member in record['members']:
            candidates = glob.glob(os.path.join(queue_dir, f'{member}.json')) + glob.glob(
                os.path.join(queue_dir, 'archive', '*', f'{member}.json')
            )
            for member_path in candidates:
                member_record = _load(member_path)
                if member_record is None or member_record.get('status') not in TERMINAL_STATUSES:
                    continue
                resolution = member_record.get('resolution') or ''
                kind = 'DEDUP-MARKER' if DEDUP_MARKER in resolution else 'RULED'
                excerpt = ' '.join(resolution.split())[:140]
                hits.append(
                    f'     member {member} status={member_record.get("status")}'
                    f' [{kind}] {excerpt}'
                )
        if hits:
            findings += 1
            lines.append(
                f'  {os.path.basename(path)}  category={record.get("category")}'
                f'  task={record.get("task_id")}  severity={record.get("severity")}'
            )
            lines.extend(hits)

    print(f'== {queue_dir}: {len(pending)} pending L2s with members, {findings} with terminal members')
    for line in lines:
        print(line)
    return findings


def default_queues() -> list[str]:
    src = os.path.expanduser('~/src')
    queues = sorted(glob.glob(os.path.join(src, '*', 'data', 'escalations')))
    recon = os.path.join(src, 'dark-factory', 'data', 'reconciliation', 'escalations')
    if os.path.isdir(recon):
        queues.append(recon)
    return queues


def main(argv: list[str]) -> int:
    queues = [q for q in (argv or default_queues()) if os.path.isdir(q)]
    if not queues:
        print('no escalation queue directories found', file=sys.stderr)
        return 2
    total = sum(sweep_queue(q) for q in queues)
    print(f'== total: {total} pending L2(s) with terminal members across {len(queues)} queue(s)')
    print('== REPORT-ONLY: never close a record off this output alone --'
          ' see skills/escalation-watcher/SKILL.md "Ruled-elsewhere check" (pins!)')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
