#!/usr/bin/env python3
"""One-shot backfill: record each legacy shard spec's kind and basis in a rating-rounds ``key.json``.

Every shard spec a :class:`write_triage_rating_rounds.RatingScratch` keys now
records its kind (:class:`write_triage_rating_rounds.ShardKind`) and basis, and
``add_shards`` refuses a key with a spec lacking either. Keys rated before then
carry neither. This script derives both from the key's own structured fields:

- kind: a ``third`` shard is THIRD; a double-rated (primary or second) shard is
  RERATE when a re-rate in ``rerates`` lists it, else INITIAL at round 0 and
  REASK at any later round;
- basis: the one basis every item of the shard shares.

It reads no shard name, so every value stays re-derivable from the key alone.
It refuses a spec whose items span more than one basis, whose role is neither
double-rated nor third, or whose recorded kind or basis disagrees with its
derivation. A recorded value that agrees is kept, so a second run is a no-op.

Usage, from ``fused-memory/``::

    python scripts/backfill_rating_shard_kinds.py calibration/write_triage_eval_set_v2_rounds/key.json
    python scripts/backfill_rating_shard_kinds.py --check calibration/*_rounds/key.json

``--check`` writes nothing, names each key needing a change on stderr, and exits 1 if any does.
A key whose derivation is refused is named on stderr with the refusal and left
as it is; every other key is still handled, and the run exits 1.
"""
from __future__ import annotations

import argparse
import copy
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fused_memory.utils.script_loader import load_script

_SCRIPTS = Path(__file__).resolve().parent

if TYPE_CHECKING:
    import write_triage_rating_rounds as _rounds
else:
    _rounds = load_script(_SCRIPTS / 'write_triage_rating_rounds.py', 'write_triage_rating_rounds')

ShardKind = _rounds.ShardKind
Role = _rounds.Role


def derived_kind(key: Mapping[str, Any], name: str) -> _rounds.ShardKind:
    """Shard *name*'s kind, from its role, its round and the re-rates' shard lists."""
    spec = key['shards'][name]
    if spec['role'] == Role.THIRD:
        return ShardKind.THIRD
    if spec['role'] not in _rounds.DOUBLE_RATED:
        raise ValueError(f'shard {name}: role {spec["role"]!r} is neither double-rated nor third')
    if any(name in rerate['shards'] for rerate in key.get('rerates', [])):
        return ShardKind.RERATE
    return ShardKind.INITIAL if spec['round'] == 0 else ShardKind.REASK


def derived_basis(key: Mapping[str, Any], name: str) -> str:
    """The one basis shared by every item of shard *name*."""
    bases = sorted({key['items'][item]['basis'] for item in key['shards'][name]['items']})
    if len(bases) != 1:
        raise ValueError(f'shard {name}: its items must share exactly one basis, not {bases}')
    return bases[0]


def backfilled(key: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of *key* with each shard spec's kind and basis set to their derivation."""
    result = copy.deepcopy(dict(key))
    for name, spec in result['shards'].items():
        derived = {'kind': str(derived_kind(key, name)), 'basis': derived_basis(key, name)}
        for field, value in derived.items():
            if spec.get(field, value) != value:
                raise ValueError(f'shard {name}: records {field} {spec[field]!r},'
                                 f' but its fields derive {value!r}')
        spec.update(derived)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record each legacy shard spec's kind and basis in rating-rounds key.json files.")
    parser.add_argument('keys', nargs='+', type=Path, help='a rating-rounds key.json, rewritten in place')
    parser.add_argument('--check', action='store_true',
                        help='write nothing; exit 1 naming each key that needs a change')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    not_keys = [str(path) for path in args.keys if path.name != 'key.json' or not path.is_file()]
    if not_keys:
        parser.error(f'not a rating-rounds key.json: {", ".join(not_keys)}')
    needing_change = refused = 0
    for path in args.keys:
        scratch = _rounds.RatingScratch(path.parent)
        key = scratch.load_key()
        try:
            result = backfilled(key)
        except ValueError as refusal:
            print(f'{path}: {refusal}', file=sys.stderr)
            refused += 1
            continue
        if result == key:
            continue
        needing_change += 1
        if args.check:
            print(f'{path}: shard specs lack a kind or basis', file=sys.stderr)
            continue
        scratch.save_key(result)
        counts = Counter((spec['kind'], spec['basis']) for spec in result['shards'].values())
        print(f'{path}: {len(result["shards"])} shards: '
              + ', '.join(f'{count} {kind}/{basis}' for (kind, basis), count in sorted(counts.items())))
    return 1 if refused or (args.check and needing_change) else 0


if __name__ == '__main__':
    sys.exit(main())
