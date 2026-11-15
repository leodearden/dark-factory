#!/usr/bin/env python3
"""Split the retired single-array raise ledger into one file per entry, then delete it.

The one-shot migration of task 7298 (Leo's ruling 2 of 2026-10-21). It writes
every entry of ``merge_lane_ratchet.LEGACY_LEDGER_RELPATH`` as its own file under
``merge_lane_ratchet.LEDGER_DIR_RELPATH`` through the instrument's own writer,
and deletes the array file. The commit staging the result is then held to
``merge_lane_ratchet.retire_legacy_ledger``'s set-equal rule by the pre-commit
auditor: every entry split, none rewritten, nothing else authorized.

RE-RUNNABLE: an entry already present as a file is rewritten to identical bytes,
so it can run again after a rebase brings in more entries, and on a branch that
still carries the array while merging a main that has already split it. With no
array present it reports so and changes nothing.

Exit codes: 0 split (or nothing to split); 2 the array or an entry is unreadable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import merge_lane_cluster as lane_cluster
import merge_lane_ratchet as ratchet
import source_measures


def split(root: Path) -> str:
    """Split *root*'s legacy array into entry files and delete it; the outcome, as a line."""
    legacy = root / ratchet.LEGACY_LEDGER_RELPATH
    if not legacy.exists():
        return f'{ratchet.LEGACY_LEDGER_RELPATH} is absent; nothing to split'
    entries = ratchet.legacy_ledger_entries(legacy.read_text(encoding='utf-8'))
    directory = root / ratchet.LEDGER_DIR_RELPATH
    for entry in entries:
        ratchet.append_authorization(directory, entry)
    recorded = {ratchet.entry_filename(entry) for entry in ratchet.load_ledger(directory)}
    missing = {ratchet.entry_filename(entry) for entry in entries} - recorded
    if missing:
        raise lane_cluster.RatchetError(f'entries not recorded as files: {sorted(missing)}')
    legacy.unlink()
    return (
        f'split {len(entries)} entr(ies) of {ratchet.LEGACY_LEDGER_RELPATH} into '
        f'{ratchet.LEDGER_DIR_RELPATH}/ ({len(recorded)} entry file(s) now) and deleted it'
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument('--root', type=Path, default=ratchet.repo_root())
    args = parser.parse_args(argv)
    try:
        print(split(args.root))
    except source_measures.MetricsError as exc:
        print(f'split_merge_lane_ratchet_ledger: {exc}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
