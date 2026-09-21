#!/usr/bin/env python3
"""Refuse a commit that RAISES the merge-lane ratchet baseline without recording it.

``scripts/merge_lane_metrics.py``'s write gate is the enforcement point for a
*regeneration*: it refuses to absorb a raise into a baseline it can read first.
What it cannot see is the baseline deleted before the write, or rendered at a
scratch path and copied over -- with nothing at the destination to compare
against, every frozen measure resets and every ceiling is re-grandfathered,
unrefused and unrecorded. Two raises reached main exactly that way
(``merge_lane/ports.py`` at 0b04534c7b, ``orchestrator/tests/conftest.py`` at
52d98220ad) while the ledger still read ``"raises": []``.

THE QUESTION THIS ASKS IS DIFFERENT, and that is the whole point. Not "does the
tree match the baseline" -- which is trivially true in every committed state,
because the pytest gate forces it -- but "does this COMMIT raise anything".
Regenerate, delete-then-write, write-elsewhere-and-copy, hand-edit: every route
to a widened baseline lands as the same staged diff against HEAD's blob, so one
comparison covers all of them.

WHY THIS IS A SEPARATE FILE from the instrument it calls. ``merge_lane_metrics``
deliberately has no git dependency -- ``repo_root()`` derives from ``__file__``,
and ``write_baseline``'s docstring declines to consult git HEAD in as many
words. Every git invocation lives here and every comparison lives there, so the
policy stays pure and testable on dicts in microseconds while the plumbing stays
testable against a real repo. Same-directory import: running this file puts
``scripts/`` on ``sys.path[0]``, so no path surgery is needed.

Exit ladder, matching the instrument's own: 0 clean, 1 a raise this commit did
not record, 2 the gate could not do its job.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import merge_lane_metrics as metrics  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - exercised by the hook, not pytest
    # LOUD, never a silent skip (INV-11). A gate that disarms itself when its
    # environment is wrong reproduces in a new place the exact defect it exists
    # to close: `merge_lane_metrics` imports `shared.safe_io` at module scope,
    # so plain python3 needs `shared` importable.
    print(
        'ratchet commit gate: cannot import merge_lane_metrics '
        f'({exc.__class__.__name__}: {exc}). This gate does not skip itself -- '
        'run the commit from a shell whose python3 can import `shared` (the '
        "checkout's .venv), or repair that environment.",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

#: The two committed artifacts this gate is about, named once.
_ARTIFACTS = (metrics.BASELINE_RELPATH, metrics.LEDGER_RELPATH)


def _git(root: Path, *args: str, stdin: str | None = None) -> str:
    """Every git invocation this gate makes.

    ``check=True``: a plumbing call that fails is an instrument failure, and
    ``main`` reports it as one. The single exception -- a revision that names
    nothing, which is legitimate data rather than a fault -- goes through
    :func:`_blob_oid`, whose ``--quiet`` makes absence an exit code instead of a
    message. ``encoding`` is pinned so a blob round trip does not depend on the
    committing shell's locale.
    """
    return subprocess.run(
        ['git', '-C', str(root), *args],
        input=stdin,
        check=True,
        text=True,
        encoding='utf-8',
        capture_output=True,
    ).stdout


def _blob_oid(root: Path, revision: str) -> str | None:
    """The blob *revision* names, or None when it names nothing.

    Absence is an ANSWER here, not a fault: an unborn HEAD, the commit that
    introduces the baseline, a ledger not yet committed. This is the one place a
    non-zero git exit is read as data.
    """
    try:
        return _git(root, 'rev-parse', '--verify', '--quiet', revision).strip()
    except subprocess.CalledProcessError:
        return None


def _staged(root: Path, diff_filter: str) -> set[str]:
    """Which ratchet artifacts this commit stages, under *diff_filter*."""
    listing = _git(
        root,
        'diff',
        '--cached',
        '--name-only',
        f'--diff-filter={diff_filter}',
        '--',
        *_ARTIFACTS,
    )
    return {line for line in listing.splitlines() if line}


def _image(root: Path, oid: str, target: Path) -> Path:
    """Write the blob *oid* to *target* so a path-taking loader can read it."""
    target.write_text(_git(root, 'cat-file', 'blob', oid), encoding='utf-8')
    return target


def _ledger_image(root: Path, revision: str, scratch: Path, name: str) -> dict:
    """The ledger at *revision*, or the fail-CLOSED empty ledger when absent.

    ``load_ledger`` owns both polarities -- absent is empty, malformed is fatal
    -- so neither is re-implemented here.
    """
    oid = _blob_oid(root, revision)
    if oid is None:
        return metrics.empty_ledger()
    return metrics.load_ledger(_image(root, oid, scratch / name))


def _refuse_unrecorded(
    raises: list[metrics.Violation], appended: list[dict]
) -> list[str]:
    """The refusal lines for every raise this commit's ledger entries do not name."""
    unrecorded = metrics.unrecorded_raises(raises, appended)
    if not unrecorded:
        return []
    return [
        f'{metrics.BASELINE_RELPATH} RAISES {len(unrecorded)} measure(s) that '
        'this commit does not record:',
        *(f'  {violation.message}' for violation in unrecorded),
        metrics.RAISE_REMEDY,
    ]


def _restored_from(root: Path, staged_oid: str) -> str | None:
    """The commit whose baseline blob *staged_oid* restores, or None.

    RESTORING BYTES THIS PATH ALREADY CARRIED IS NOT A NEW RAISE. Reverting a
    revert puts back an image the repository already reviewed and recorded, and
    demanding a fresh authorization for it would make the honest move the
    expensive one. Verified exact on the real revert: `3e7d55ce47:<baseline>`,
    `5f577b9613^:<baseline>` and `52d98220ad:<baseline>` are all blob
    a0fb5cc8e0fb1e9ce363c81538a350dbfc12c191.

    BLOB IDENTITY, not a per-measure high-water mark. A high-water rule also
    exempts that revert, but it stays permissively open afterwards, so a later
    re-raise back to an old high-water would go unrecorded.

    ``--full-history`` is load-bearing rather than decorative: git's default
    history simplification omits commits from a path's log, which would silently
    NARROW the carve-out and refuse a legitimate restore. And the enumeration
    resolves ``<commit>:<baseline>`` rather than asking whether the object
    exists, which is what scopes identity to this path -- a blob that happens to
    sit elsewhere in the tree must not license a baseline.
    """
    baseline = metrics.BASELINE_RELPATH
    commits = _git(
        root, 'rev-list', '--full-history', 'HEAD', '--', baseline
    ).split()
    if not commits:
        return None
    resolved = _git(
        root,
        'cat-file',
        '--batch-check',
        stdin='\n'.join(f'{commit}:{baseline}' for commit in commits),
    )
    # One line per revision, in the order fed in: "<oid> <type> <size>", or
    # "<revision> missing" where the path did not exist at that commit -- whose
    # first field is the revision string, so it can never match an oid.
    for commit, line in zip(commits, resolved.splitlines(), strict=False):
        if line.split()[:1] == [staged_oid]:
            return commit
    return None


def _audit_baseline(root: Path, scratch: Path) -> list[str]:
    """Audit the staged baseline against HEAD's. Empty list means clean."""
    baseline = metrics.BASELINE_RELPATH
    head_oid = _blob_oid(root, f'HEAD:{baseline}')
    if head_oid is None:
        # A first write has nothing to compare against -- write_baseline's
        # already-documented limit, and said out loud rather than passed over.
        print(
            f'ratchet commit gate: {baseline} is absent from HEAD, so this is a '
            'first write with nothing to compare against.'
        )
        return []

    staged_oid = _blob_oid(root, f':{baseline}')
    if staged_oid is None:
        return []

    raises = metrics.compare_baseline_files(
        _image(root, head_oid, scratch / 'previous.json'),
        _image(root, staged_oid, scratch / 'current.json'),
    )
    if not raises:
        return []

    restored = _restored_from(root, staged_oid)
    if restored is not None:
        print(
            f'ratchet commit gate: {baseline} restores the blob recorded at '
            f'this path by commit {restored}, absorbing {len(raises)} '
            'measure(s) this repository has already reviewed. Allowed.'
        )
        return []

    appended = metrics.ledger_appended_entries(
        _ledger_image(root, f'HEAD:{metrics.LEDGER_RELPATH}', scratch, 'head.json'),
        _ledger_image(root, f':{metrics.LEDGER_RELPATH}', scratch, 'staged.json'),
    )
    return _refuse_unrecorded(raises, appended)


def _audit(root: Path) -> int:
    if not _staged(root, 'ACMRD'):
        return 0

    if metrics.BASELINE_RELPATH in _staged(root, 'D'):
        # The first half of "delete the destination first", closed before any
        # comparison is attempted. Deleting the baseline is never legitimate:
        # the freshness gate fails hard without it.
        print(
            f'ratchet commit gate: {metrics.BASELINE_RELPATH} may not be '
            'DELETED. With nothing to compare against, every frozen measure '
            'resets and every ceiling is re-grandfathered, unrefused and '
            'unrecorded.',
            file=sys.stderr,
        )
        return 1

    with tempfile.TemporaryDirectory() as scratch:
        refusals = _audit_baseline(root, Path(scratch))

    if not refusals:
        return 0
    print('\n'.join(refusals), file=sys.stderr)
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Refuse a commit that RAISES the merge-lane ratchet baseline '
            'without recording it.'
        )
    )
    parser.add_argument(
        '--root',
        default=str(metrics.repo_root()),
        help='repository root whose staged diff is audited (default: this checkout)',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = Path(args.root)
    try:
        return _audit(root)
    except metrics.MetricsError as exc:
        # A named cause on one line, never a traceback: a traceback reads as a
        # broken tool and sends the committer hunting the wrong thing.
        print(f'ratchet commit gate: {exc}', file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(
            f'ratchet commit gate: git {" ".join(exc.cmd[3:])} failed '
            f'(rc={exc.returncode}): {(exc.stderr or "").strip()}',
            file=sys.stderr,
        )
        return 2


if __name__ == '__main__':
    sys.exit(main())
