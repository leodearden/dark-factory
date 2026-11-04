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
to a widened baseline lands as the same staged diff against the commit's
parent's blob -- HEAD's, or both parents' when finishing a merge -- so one
comparison covers all of them.

A CONFLICTED MERGE has two parents, and pre-commit runs when one is finished
with `git commit`, with MERGE_HEAD set; only a clean merge skips it. Read
against HEAD alone, every measure MERGE_HEAD moved is a "raise" -- 18 of them
refused esc-3620-11's resolver, all main's own. So a merge's artifacts are
audited against BOTH parents: each measure is held to the 3-way bound defined
by merge_lane_metrics.py::compare_merged_baseline_files, and only the ledger
entry files the merge itself adds cover anything above that bound. The restore
carve-out is a single-parent rule and is not consulted.

THE LEDGER IS A DIRECTORY, one file per authorization
(``metrics.LEDGER_DIR_RELPATH``), so append-only is a comparison of file
listings: every entry file a parent holds must be staged with the same blob,
and a commit's own entries are the files no parent holds. The retired
single-array file (``metrics.LEGACY_LEDGER_RELPATH``) appears here for one
reason: the commit that splits it into files is held to
``metrics.retire_legacy_ledger``'s set-equal rule, and nothing may stage it
back once a parent has dropped it.

AN ENTRY IS ATTRIBUTED, NOT MERELY NAMED (Leo, 2026-10-21, task 7298). A NEW
entry file's ``task_id`` must be a task this BRANCH works on: the id in its
name (``task/<id>``, the orchestrator's branches; ``carry/<id>`` or a name
ending ``-<id>``, a hand carry's), or an id an earlier commit on the branch
since main cites -- as its conventional scope, ``fix(<id>): ...``, or as
``task <id>``; the commit being made has no message yet. That admits a
carrier landing another task's work, whose commits cite the carried id. Its
``reason`` may not be blank. Before this, an entry with a blank or foreign
``task_id`` covered a raise whenever its measures matched. The task store is
deliberately not consulted: a git hook does not depend on ``tasks.db``.

WHY THIS IS A SEPARATE FILE from the instrument it calls. ``merge_lane_metrics``
deliberately has no git dependency -- ``repo_root()`` derives from ``__file__``,
and ``write_baseline``'s docstring declines to consult git HEAD in as many
words. Every comparison lives there; reading the artifacts out of git at a
revision or the index lives in ``ratchet_revisions``, which this gate shares
with ``scripts/check_ratchet_advance.py``; what stays here is commit-specific --
the staged filter, the merge 3-way arm, attribution and the exit ladder. So the
policy stays pure and testable on dicts in microseconds while the plumbing stays
testable against a real repo. Same-directory import: running this file puts
``scripts/`` on ``sys.path[0]``, so no path surgery is needed.

EXIT LADDER: 0 clean; 1 a POLICY refusal this commit must fix -- an unrecorded
raise, a staged deletion of the baseline, a rewritten ledger history, a retired
ledger split wrongly or staged back, or an entry this branch may not record; 2 the
gate could not do its job -- its environment is wrong, git failed, or a staged
artifact is unreadable. The split is the only machine-readable thing this gate
says, and it is what lets a caller -- ``hooks/project-checks`` here, and
``hooks/reference-transaction`` for ``scripts/check_ratchet_advance.py``, which
keeps the same ladder -- tell "instrument down" apart without ever waving a
policy refusal through under it.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import merge_lane_metrics as metrics  # type: ignore[import-not-found]
    import ratchet_revisions as revisions
    import source_measures  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - exercised by the hook, not pytest
    # LOUD, never a silent skip (INV-11). The instrument binds this checkout's
    # workspace members through scripts/_checkout.py, whose CheckoutBindingError
    # is an ImportError, so an unbindable checkout lands here as exit 2 too.
    print(
        'ratchet commit gate: cannot import its instrument '
        f'({exc.__class__.__name__}: {exc}). This gate does not skip itself -- '
        "the instrument binds this checkout's workspace members through "
        'scripts/_checkout.py, so the fault is the file or module named above '
        "(e.g. the checkout's pyproject.toml or a member's src/), not which "
        'virtualenv the shell has active.',
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

#: A branch naming the task it works on: the orchestrator's ``task/<id>``, or a
#: hand carry's ``carry/<id>``, the id leading (``task/5722-gate``) or ending the
#: name after a dash (``carry/A4-4614``).
_BRANCH_TASK = re.compile(r'(?:task|carry)/(?:(\d+)(?!\d).*|.*-(\d+))$')

#: How a commit subject cites a task: a conventional scope of ids
#: (``fix(3774): ...``), or ``task <id>`` in its text (CONTRIBUTING.md section 7).
_SUBJECT_SCOPE = re.compile(r'[a-z]+\(([\d, ]+)\)!?:')
_SUBJECT_TASK = re.compile(r'\btask (\d+)\b', re.IGNORECASE)


def _staged(root: Path, diff_filter: str, merge_head: str | None) -> set[str]:
    """Which ratchet artifacts this commit stages, under *diff_filter*.

    Against HEAD and, finishing a merge, against MERGE_HEAD too. A resolution
    that keeps HEAD's artifacts byte-for-byte while dropping MERGE_HEAD's moves
    or entries differs from MERGE_HEAD alone, and it is exactly the resolution
    that breaks a merge's invariant (heuristic 10, uniformly).
    """
    staged = _index_changes(root, diff_filter)
    if merge_head is not None:
        staged |= _index_changes(root, diff_filter, merge_head)
    return staged


def _index_changes(root: Path, diff_filter: str, *commit: str) -> set[str]:
    """The ratchet artifacts the index changes against *commit*, or else HEAD.

    HEAD stays IMPLICIT rather than named: ``git diff --cached HEAD`` fails on
    an unborn HEAD, where the implicit form compares against the empty tree.
    """
    listing = revisions.git(
        root,
        'diff',
        '--cached',
        *commit,
        '--name-only',
        f'--diff-filter={diff_filter}',
        '--',
        *revisions.ARTIFACTS,
    )
    return {line for line in listing.splitlines() if line}


def _refuse_unrecorded(
    raises: list[metrics.Violation], appended: list[dict]
) -> list[str]:
    """The refusal lines for every raise this commit's own ledger entries do not cover."""
    unrecorded = metrics.unrecorded_raises(raises, appended)
    if not unrecorded:
        return []
    return [
        f'{metrics.BASELINE_RELPATH} RAISES {len(unrecorded)} measure(s) that '
        'this commit does not record:',
        *(f'  {violation.message}' for violation in unrecorded),
        metrics.RAISE_REMEDY,
    ]


def _branch_task_ids(root: Path) -> set[str]:
    """The task ids this branch works on: its name's, and every one its own commits cite.

    "Its own commits" are those reachable from HEAD and not from main, so an id
    only main's history cites licenses nothing here. A detached HEAD has no
    name, an unborn one no commits, and a repository with no main no fork
    point to measure from -- each just contributes nothing.
    """
    ids: set[str] = set()
    try:
        branch = revisions.git(root, 'symbolic-ref', '--quiet', '--short', 'HEAD').strip()
    except subprocess.CalledProcessError:
        branch = ''
    if named := _BRANCH_TASK.match(branch):
        ids.update(group for group in named.groups() if group)
    if (
        revisions.object_id(root, 'HEAD') is None
        or revisions.object_id(root, 'refs/heads/main') is None
    ):
        return ids
    subjects = revisions.git(root, 'log', '--format=%s', 'refs/heads/main..HEAD')
    for subject in subjects.splitlines():
        if scope := _SUBJECT_SCOPE.match(subject):
            ids.update(re.findall(r'\d+', scope.group(1)))
        ids.update(_SUBJECT_TASK.findall(subject))
    return ids


def _refuse_unattributed(own_entries: list[dict], task_ids: set[str]) -> list[str]:
    """The refusal lines for every own entry this branch may not record."""
    lines = []
    for entry in own_entries:
        task_id, reason = entry.get('task_id'), entry.get('reason')
        name = metrics.entry_filename(entry)
        if not isinstance(reason, str) or not reason.strip():
            lines.append(f'  {name} records no reason')
        if not isinstance(task_id, str) or task_id.strip() not in task_ids:
            lines.append(f'  {name} records task_id {task_id!r}')
    if not lines:
        return []
    return [
        f'ratchet commit gate: {len(lines)} new {metrics.LEDGER_DIR_RELPATH} '
        'entr(ies) are not attributed to a task this branch works on:',
        *lines,
        "An entry's task_id must be this branch's own task (task/<id>, "
        'carry/<id>, or a name ending -<id>) or one an EARLIER commit on this '
        'branch cites -- `fix(<id>): ...` or `task <id>`; this branch names or '
        f'cites: {sorted(task_ids) or "none"}. The commit being made cannot cite '
        'itself, because pre-commit runs before its message exists: commit the '
        'code first and the entry after it, or name the branch carry/<id>. Its '
        'reason may not be blank. Record the raise with --authorize-raise <that id>.',
    ]


def _audit_baseline(
    root: Path, scratch: Path, appended: list[dict]
) -> list[str]:
    """Audit the staged baseline against HEAD's. Empty list means clean."""
    baseline = metrics.BASELINE_RELPATH
    head_oid = revisions.object_id(root, f'HEAD:{baseline}')
    if head_oid is None:
        return _first_write('HEAD')

    staged_oid = revisions.object_id(root, f':{baseline}')
    if staged_oid is None:
        return []

    raises = metrics.compare_baseline_files(
        revisions.write_blob(root, head_oid, scratch / 'previous.json'),
        revisions.write_blob(root, staged_oid, scratch / 'current.json'),
    )
    if not raises:
        return []

    restored = revisions.restores_previous_image(root, staged_oid, 'HEAD')
    if restored is not None:
        # NAME WHAT CAME BACK, not just how much of it. A count tells a reviewer
        # nothing about whether undoing the last change was the right move; the
        # measures and keys are what they judge it on.
        print(
            '\n'.join([
                f'ratchet commit gate: {baseline} restores the value this path '
                f'held before commit {restored}, reabsorbing '
                f'{len(raises)} measure(s). Allowed -- this is the state the '
                'tree was running with one commit ago:',
                *(f'  {violation.message}' for violation in raises),
            ])
        )
        return []

    return _refuse_unrecorded(raises, appended)


def _audit_merged_baseline(
    root: Path, scratch: Path, merge_head: str, own_entries: list[dict]
) -> list[str]:
    """Audit a MERGE's staged baseline against its parents' 3-way bound.

    The restore carve-out is NOT consulted. "Undoing the last change" names ONE
    parent's history, and in a merge HEAD's previous value is exactly the stale
    image that discards HEAD's own last move.
    """
    baseline = metrics.BASELINE_RELPATH
    ours_oid = revisions.object_id(root, f'HEAD:{baseline}')
    theirs_oid = revisions.object_id(root, f'{merge_head}:{baseline}')
    if ours_oid is None and theirs_oid is None:
        return _first_write('both parents')

    staged_oid = revisions.object_id(root, f':{baseline}')
    if staged_oid is None:
        return []

    base = revisions.merge_base(root, 'HEAD', merge_head)
    base_oid = None if base is None else revisions.object_id(root, f'{base}:{baseline}')
    over = f'merge base {base}' if base is not None else 'no common ancestor'
    print(
        f'ratchet commit gate: finishing a merge, so {baseline} is held to the '
        f'3-way bound of HEAD and MERGE_HEAD over {over}.'
    )

    def image(oid: str | None, name: str) -> Path | None:
        return None if oid is None else revisions.write_blob(root, oid, scratch / name)

    raises = metrics.compare_merged_baseline_files(
        base=image(base_oid, 'base.json'),
        ours=image(ours_oid, 'ours.json'),
        theirs=image(theirs_oid, 'theirs.json'),
        current=revisions.write_blob(root, staged_oid, scratch / 'current.json'),
    )
    refusals = _refuse_unrecorded(raises, own_entries)
    if not refusals:
        return []
    return [
        *refusals,
        'FINISHING A MERGE: a value above that bound is this merge\'s own raise '
        '-- and --write-baseline records nothing over a file that already holds '
        'it. Run `git checkout '
        f'HEAD -- {metrics.BASELINE_RELPATH}` (ours; `--ours` is a no-op once the '
        'ratchet-baseline driver resolved the file), then the --write-baseline --authorize-raise '
        "command above: it measures from ours and records the merge's own entry.",
    ]


def _first_write(absent_from: str) -> list[str]:
    """Pass a baseline no parent carried -- and say so out loud.

    A first write has nothing to compare against: ``write_baseline``'s
    already-documented limit, announced rather than passed over.
    """
    print(
        f'ratchet commit gate: {metrics.BASELINE_RELPATH} is absent from '
        f'{absent_from}, so this is a first write with nothing to compare against.'
    )
    return []


def _refuse(lines: list[str]) -> int:
    if not lines:
        return 0
    print('\n'.join(lines), file=sys.stderr)
    return 1


def _audit(root: Path) -> int:
    # One quiet rev-parse that reads no history, so it may precede the filter.
    merge_head = revisions.object_id(root, 'MERGE_HEAD')
    staged = _staged(root, 'ACMRD', merge_head)
    if not staged:
        # The cheap filter, and it decides before anything consults HEAD: an
        # ordinary commit is never ambushed, and never pays for archaeology.
        return 0

    if metrics.BASELINE_RELPATH in _staged(root, 'D', merge_head):
        # The first half of "delete the destination first", closed before any
        # comparison is attempted. Deleting the baseline is never legitimate:
        # the freshness gate fails hard without it.
        return _refuse([
            f'ratchet commit gate: {metrics.BASELINE_RELPATH} may not be '
            'DELETED. With nothing to compare against, every frozen measure '
            'resets and every ceiling is re-grandfathered, unrefused and '
            'unrecorded.'
        ])

    with tempfile.TemporaryDirectory() as scratch_dir:
        scratch = Path(scratch_dir)
        # The ledger audit runs on BOTH paths; the baseline audit only when the
        # baseline is staged. Neither can excuse the other: a covering entry
        # does not launder a rewritten history, and an untouched history does
        # not excuse an unrecorded raise.
        parents = ['HEAD'] if merge_head is None else ['HEAD', merge_head]
        try:
            own_entries = revisions.own_entries(root, parents)
        except metrics.AppendOnlyViolation as exc:
            # A VERDICT, so the POLICY rung -- never the instrument-failure one
            # its ``MetricsError`` siblings take at ``main``. It returns here
            # rather than joining the baseline arm's lines because nothing else
            # in the commit can excuse it.
            return _refuse([f'ratchet commit gate: {exc}'])
        if own_entries:
            unattributed = _refuse_unattributed(own_entries, _branch_task_ids(root))
            if unattributed:
                return _refuse(unattributed)
        if metrics.BASELINE_RELPATH not in staged:
            return 0
        refusals = (
            _audit_baseline(root, scratch, own_entries)
            if merge_head is None
            else _audit_merged_baseline(root, scratch, merge_head, own_entries)
        )
    return _refuse(refusals)


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
    except source_measures.MetricsError as exc:
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
