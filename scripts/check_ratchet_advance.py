#!/usr/bin/env python3
"""Refuse a move of main that RAISES the merge-lane ratchet baseline without recording it.

``scripts/check_staged_ratchet_raise.py`` audits a COMMIT, so a raise that
reaches main without one -- a ``git rebase`` replay, a clean merge, a hook-less
commit -- passes it unseen. Each of those still ends in a move of
``refs/heads/main``. ``hooks/reference-transaction`` runs this auditor on that
move, before it happens, with main's current value as OLD and the proposed one
as NEW.

The move is audited as ONE change, merge-base(OLD, NEW) -> NEW, under the commit
gate's own rules: the baseline may not be deleted, the ledger is append-only,
and a raise is covered by the sum of the entry files the move adds. A rewind
adds nothing and passes. A sideways move is a rewind to the merge base plus an
advance from it, so what only OLD holds -- an entry main recorded, a lowering
main took -- is dropped unrefused, exactly as a rewind drops it. Same exit
ladder as the commit gate: 0 clean, 1 a policy refusal, 2 the auditor could not
do its job.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    import merge_lane_metrics as metrics
    import ratchet_revisions as revisions
    import source_measures
except ImportError as exc:
    # LOUD, never a silent skip (INV-11), exactly as the commit gate: an
    # unbindable checkout's CheckoutBindingError is an ImportError too.
    print(
        'ratchet advance gate: cannot import its instrument '
        f'({exc.__class__.__name__}: {exc}). This gate does not skip itself -- '
        "the instrument binds this checkout's workspace members through "
        'scripts/_checkout.py, so the fault is the file or module named above '
        "(e.g. the checkout's pyproject.toml or a member's src/), not which "
        'virtualenv the shell has active.',
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

_PREFIX = 'ratchet advance gate:'


@dataclass(frozen=True)
class _Move:
    """main's move OLD -> NEW over their merge base, and the baseline blob at each end.

    ``base`` is None for histories with nothing in common, and a baseline oid is
    None where that commit holds no baseline.
    """

    old: str
    new: str
    base: str | None
    base_baseline_oid: str | None
    new_baseline_oid: str | None


def _say(*lines: str, refusal: bool = False) -> None:
    """Print *lines*, each prefixed, to stderr for a refusal and stdout otherwise."""
    text = '\n'.join(f'{_PREFIX} {line}' for line in lines)
    print(text, file=sys.stderr if refusal else sys.stdout)


def _commit(root: Path, revision: str) -> str:
    """The commit *revision* names; a ``RatchetError`` when it names none."""
    oid = revisions.object_id(root, f'{revision}^{{commit}}')
    if oid is None:
        raise metrics.RatchetError(f'{revision!r} names no commit, so there is no move to audit')
    return oid


def _changed_artifacts(root: Path, base: str | None, new: str) -> list[str]:
    """The ratchet artifacts *new* changes over *base*, or over the empty tree."""
    since = base or revisions.git(root, 'hash-object', '-t', 'tree', '--stdin', stdin='').strip()
    listing = revisions.git(
        root, 'diff-tree', '-r', '--name-only', since, new, '--', *revisions.ARTIFACTS
    )
    return [line for line in listing.splitlines() if line]


def _baseline_oid(root: Path, revision: str) -> str | None:
    return revisions.object_id(root, f'{revision}:{metrics.BASELINE_RELPATH}')


def _raises(root: Path, base_oid: str, new_oid: str) -> list[metrics.Violation]:
    """Every raise NEW's baseline image makes over BASE's, read as two blobs."""
    with tempfile.TemporaryDirectory() as scratch_dir:
        scratch = Path(scratch_dir)
        return metrics.compare_baseline_files(
            revisions.write_blob(root, base_oid, scratch / 'base.json'),
            revisions.write_blob(root, new_oid, scratch / 'new.json'),
        )


def _refuse_unrecorded(move: _Move, unrecorded: list[metrics.Violation]) -> int:
    baseline = metrics.BASELINE_RELPATH
    _say(
        f"main's move {move.old[:10]}..{move.new[:10]} RAISES {len(unrecorded)} measure(s) of "
        f'{baseline} that the entry files it adds do not record:',
        *(f'  {violation.message}' for violation in unrecorded),
        metrics.RAISE_REMEDY,
        'This raise reached main without passing the commit gate (a rebase, a '
        'clean merge or a hook-less commit). Record it on the branch against '
        f"main's baseline: `git checkout {move.base} -- {baseline}`, then "
        f'`python scripts/merge_lane_metrics.py --write-baseline {baseline} '
        '--authorize-raise <task-id> --reason "<why this is net-additive>"`, '
        'then commit both artifacts.',
        refusal=True,
    )
    return 1


def _restores_mains_previous_image(
    root: Path, base: str, new_oid: str, raises: list[metrics.Violation]
) -> bool:
    """True, and said out loud, when NEW's baseline undoes main's LAST baseline change.

    The commit gate's carve-out, with main's history -- the merge base's -- as
    "one commit ago". Covering entries are not consulted on this path.
    """
    restored = revisions.restores_previous_image(root, new_oid, history_tip=base)
    if restored is None:
        return False
    _say(
        f'{metrics.BASELINE_RELPATH} restores the value this path held before '
        f'commit {restored}, reabsorbing {len(raises)} measure(s). Allowed -- this '
        'is the state main was running with one baseline change ago:',
        *(f'  {violation.message}' for violation in raises),
    )
    return True


def _audit_baseline(root: Path, move: _Move, own: list[dict]) -> int:
    """Audit NEW's baseline against BASE's, covered by the move's own entries."""
    base_oid, new_oid = move.base_baseline_oid, move.new_baseline_oid
    if new_oid is None or new_oid == base_oid:
        return 0
    if move.base is None or base_oid is None:
        _say(
            f'{metrics.BASELINE_RELPATH} is absent from the merge base, so this is '
            'a first write with nothing to compare against.'
        )
        return 0
    raises = _raises(root, base_oid, new_oid)
    if not raises or _restores_mains_previous_image(root, move.base, new_oid, raises):
        return 0
    unrecorded = metrics.unrecorded_raises(raises, own)
    if not unrecorded:
        return 0
    return _refuse_unrecorded(move, unrecorded)


def _audit(root: Path, old_revision: str, new_revision: str) -> int:
    old, new = _commit(root, old_revision), _commit(root, new_revision)
    base = revisions.merge_base(root, old, new)
    if base is None:
        _say(
            f'{old[:10]} and {new[:10]} share no history, so the audit covers '
            f'everything {new[:10]} holds.'
        )
    elif base == new:
        _say(
            'main moves back to an ancestor; nothing reaches main that it has '
            'not already held.'
        )
        return 0
    elif base != old:
        _say(
            f'main does not fast-forward from {old[:10]} to {new[:10]}, so the '
            f'audit covers what it adds over merge base {base}.'
        )

    if not _changed_artifacts(root, base, new):
        return 0
    move = _Move(
        old,
        new,
        base,
        base_baseline_oid=None if base is None else _baseline_oid(root, base),
        new_baseline_oid=_baseline_oid(root, new),
    )
    if move.base_baseline_oid and not move.new_baseline_oid:
        _say(
            f'{metrics.BASELINE_RELPATH} may not be DELETED. With nothing to '
            'compare against, every frozen measure resets and every ceiling is '
            're-grandfathered, unrefused and unrecorded.',
            refusal=True,
        )
        return 1
    try:
        own = revisions.own_entries(root, [base] if base else [], tip=new)
    except metrics.AppendOnlyViolation as exc:
        # A VERDICT, so the POLICY rung, uniform with the commit gate.
        _say(str(exc), refusal=True)
        return 1
    return _audit_baseline(root, move, own)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Refuse a move of main that RAISES the merge-lane ratchet baseline '
            'without recording it.'
        )
    )
    parser.add_argument(
        '--root',
        default=str(metrics.repo_root()),
        help='repository root whose main is moving (default: this checkout)',
    )
    parser.add_argument('old', help="main's current value")
    parser.add_argument('new', help="main's proposed value")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = Path(args.root)
    try:
        return _audit(root, args.old, args.new)
    except source_measures.MetricsError as exc:
        # A named cause on one line, never a traceback.
        _say(str(exc), refusal=True)
        return 2
    except subprocess.CalledProcessError as exc:
        _say(
            f'git {" ".join(exc.cmd[3:])} failed (rc={exc.returncode}): '
            f'{(exc.stderr or "").strip()}',
            refusal=True,
        )
        return 2


if __name__ == '__main__':
    sys.exit(main())
