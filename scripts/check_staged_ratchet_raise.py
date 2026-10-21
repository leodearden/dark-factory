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
audited against BOTH parents: each measure is bounded by the 3-way rule git
applies to the text (a side's own move stands, up or down; where both sides
moved, the higher one bounds it), and only the ledger entry files the merge
itself adds cover anything above that bound. The restore carve-out is a
single-parent rule and is not consulted.

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
words. Every git invocation lives here and every comparison lives there, so the
policy stays pure and testable on dicts in microseconds while the plumbing stays
testable against a real repo. Same-directory import: running this file puts
``scripts/`` on ``sys.path[0]``, so no path surgery is needed.

EXIT LADDER: 0 clean; 1 a POLICY refusal this commit must fix -- an unrecorded
raise, a staged deletion of the baseline, a rewritten ledger history, a retired
ledger split wrongly or staged back, or an entry this branch may not record; 2 the
gate could not do its job -- its environment is wrong, git failed, or a staged
artifact is unreadable. The split is the only machine-readable thing this gate
says, and it is what lets a later caller (CI, a merge-lane gate, the residual
``RAISE_REMEDY`` already names) treat 2 as "instrument down, retry or ignore"
without ever waving a policy refusal through under it.
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

#: The committed artifacts this gate is about, named once: the baseline, the
#: ledger directory, and the retired ledger whose split and burial it audits.
_ARTIFACTS = (
    metrics.BASELINE_RELPATH,
    metrics.LEDGER_DIR_RELPATH,
    metrics.LEGACY_LEDGER_RELPATH,
)


def _git(root: Path, *args: str, stdin: str | None = None) -> str:
    """Every git invocation this gate makes.

    ``check=True``: a plumbing call that fails is an instrument failure, and
    ``main`` reports it as one. The two exceptions are answers rather than
    faults, and each is read in exactly one place: a revision that names
    nothing (:func:`_object_id`, whose ``--quiet`` makes absence an exit code
    instead of a message), and two histories with no common ancestor
    (:func:`_merge_base`). ``encoding`` is pinned so a blob round trip does not
    depend on the committing shell's locale.
    """
    return subprocess.run(
        ['git', '-C', str(root), *args],
        input=stdin,
        check=True,
        text=True,
        encoding='utf-8',
        capture_output=True,
    ).stdout


def _object_id(root: Path, revision: str) -> str | None:
    """The object *revision* names, or None when it names nothing.

    Absence is an ANSWER here, not a fault: an unborn HEAD, the commit that
    introduces the baseline, a ledger not yet committed, a commit that is not
    finishing a merge and so has no MERGE_HEAD. This is the one rev-parse site
    where a non-zero git exit is read as data.
    """
    try:
        return _git(root, 'rev-parse', '--verify', '--quiet', revision).strip()
    except subprocess.CalledProcessError:
        return None


def _merge_base(root: Path, merge_head: str) -> str | None:
    """The commit HEAD and *merge_head* are merged over, or None when none exists.

    None is DATA: unrelated histories have no common ancestor, and git merges
    them against the empty tree. ``git merge-base`` says so with rc 1 and no
    output, so exactly that is read as absence; any other failure is the
    instrument failure ``main`` reports.
    """
    try:
        return _git(root, 'merge-base', 'HEAD', merge_head).strip()
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 1 and not (exc.stdout or exc.stderr):
            return None
        raise


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
    listing = _git(
        root,
        'diff',
        '--cached',
        *commit,
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


def _restores_previous_image(root: Path, staged_oid: str) -> str | None:
    """The commit whose baseline blob *staged_oid* restores, if it is the LAST one.

    UNDOING THE LAST CHANGE TO THE BASELINE IS NOT A NEW RAISE. The path's
    immediately-previous value is the state this repository held one commit ago,
    so putting it back re-raises nothing the tree has not just been running
    with, and demanding a fresh authorization to undo a revert would make the
    honest move the expensive one. Verified on the real revert: `3e7d55ce47`
    restored blob a0fb5cc8e0, which was the value at that path immediately
    before 5f577b9613 replaced it with 0a42c2ee7d.

    ONE STEP BACK IS THE WHOLE RULE, and the bound is the point. An earlier
    version of this function admitted any blob the path had EVER carried, which
    is a wholesale ratchet reset wearing a carve-out's clothes: staging the
    baseline from 60e954b608 was waved through "absorbing 97 measure(s)". A
    per-measure high-water-mark rule was rejected for the SAME defect at a
    different granularity -- blob identity was not the cure for it, merely the
    same unboundedness at whole-image resolution. Both stay permissively open to
    every value the ratchet has moved through, and returning to one re-absorbs
    every measure the images in between lowered. One step back is the only form
    of this rule that is bounded by construction.

    ``--full-history`` is load-bearing rather than decorative: git's default
    history simplification omits commits from a path's log, so it could report
    some older image as the previous one and WIDEN this rule. And the
    enumeration resolves ``<commit>:<baseline>`` rather than asking whether the
    object exists, which is what scopes identity to this path -- a blob that
    happens to sit elsewhere in the tree must not license a baseline.
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
    ).splitlines()
    head_oid = _object_id(root, f'HEAD:{baseline}')

    # Newest first. Skip the run of entries still holding HEAD's blob -- those
    # are commits that touched the path without changing its value -- and let
    # the FIRST remaining entry answer, whatever it is. A `missing` line (the
    # path was deleted, or did not yet exist) has the revision string as its
    # first field, so it can never equal an oid: a deletion therefore ENDS the
    # walk rather than being skipped, with no special case for it.
    for commit, line in zip(commits, resolved, strict=False):
        oid = line.split()[:1]
        if oid == [head_oid]:
            continue
        return commit if oid == [staged_oid] else None
    return None


def _blob_text(root: Path, oid: str) -> str:
    return _git(root, 'cat-file', 'blob', oid)


def _entry_listing(root: Path, revision: str | None) -> dict[str, str]:
    """The ledger's entry files at *revision* (None: the index), as name -> blob id.

    ``_README.json`` is left out: it records no authorization, and it is the
    one file there that may change. A revision that names nothing -- an unborn
    HEAD -- holds no entries. An index entry still unmerged is a fault, not an
    answer: the gate never guesses which stage a resolution meant.
    """
    ledger = metrics.LEDGER_DIR_RELPATH
    if revision is None:
        listing = _git(root, 'ls-files', '--stage', '-z', '--', ledger)
    elif _object_id(root, revision) is None:
        return {}
    else:
        listing = _git(root, 'ls-tree', '-r', '-z', '--full-tree', revision, '--', ledger)
    entries: dict[str, str] = {}
    for record in filter(None, listing.split('\0')):
        meta, path = record.split('\t', 1)
        fields = meta.split()
        if revision is None and fields[2] != '0':
            raise metrics.RatchetError(f'{path} is unmerged in the index')
        name = path.removeprefix(f'{ledger}/')
        if name != metrics.LEDGER_README_NAME:
            entries[name] = fields[1] if revision is None else fields[2]
    return entries


def _own_entries(root: Path, parents: list[str]) -> list[dict]:
    """This commit's OWN ledger entries, refusing any rewrite of a parent's.

    Resolved whenever a ratchet artifact is staged, not only when the baseline
    moved: the append-only promise is broken precisely by a commit that touches
    the ledger ALONE, so an audit conditioned on the baseline would leave it
    unenforced in the one case that breaks it (heuristic 10, uniformly). The
    refusals are ``metrics``' own ``AppendOnlyViolation`` verdicts, which
    :func:`_audit` puts on the policy rung.
    """
    staged = _entry_listing(root, None)
    own_names = metrics.own_entry_names(
        [_entry_listing(root, parent) for parent in parents], staged
    )
    own = [metrics.parse_entry(name, _blob_text(root, staged[name])) for name in own_names]
    return metrics.retire_legacy_ledger(_retired_entries(root, parents), staged, own)


def _retired_entries(root: Path, parents: list[str]) -> list[dict]:
    """The retired array's entries in every parent that held it, if this commit drops it.

    Staging it while some parent lacks it -- a merge resolution that kept a
    branch's copy of the array -- buries an entry where nothing reads it, so
    that is refused outright rather than retired.
    """
    legacy = metrics.LEGACY_LEDGER_RELPATH
    held = {parent: _object_id(root, f'{parent}:{legacy}') for parent in parents}
    if _object_id(root, f':{legacy}') is not None:
        if None in held.values():
            raise metrics.AppendOnlyViolation(
                f'{legacy} was retired: a raise is recorded as one file under '
                f'{metrics.LEDGER_DIR_RELPATH}/. Delete it from this commit and '
                'split any entry only it carries with '
                'scripts/split_merge_lane_ratchet_ledger.py.'
            )
        return []
    return [
        entry
        for oid in held.values()
        if oid is not None
        for entry in metrics.legacy_ledger_entries(_blob_text(root, oid))
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
        branch = _git(root, 'symbolic-ref', '--quiet', '--short', 'HEAD').strip()
    except subprocess.CalledProcessError:
        branch = ''
    if named := _BRANCH_TASK.match(branch):
        ids.update(group for group in named.groups() if group)
    if _object_id(root, 'HEAD') is None or _object_id(root, 'refs/heads/main') is None:
        return ids
    for subject in _git(root, 'log', '--format=%s', 'refs/heads/main..HEAD').splitlines():
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
    head_oid = _object_id(root, f'HEAD:{baseline}')
    if head_oid is None:
        return _first_write('HEAD')

    staged_oid = _object_id(root, f':{baseline}')
    if staged_oid is None:
        return []

    raises = metrics.compare_baseline_files(
        _image(root, head_oid, scratch / 'previous.json'),
        _image(root, staged_oid, scratch / 'current.json'),
    )
    if not raises:
        return []

    restored = _restores_previous_image(root, staged_oid)
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
    ours_oid = _object_id(root, f'HEAD:{baseline}')
    theirs_oid = _object_id(root, f'{merge_head}:{baseline}')
    if ours_oid is None and theirs_oid is None:
        return _first_write('both parents')

    staged_oid = _object_id(root, f':{baseline}')
    if staged_oid is None:
        return []

    base = _merge_base(root, merge_head)
    base_oid = None if base is None else _object_id(root, f'{base}:{baseline}')
    over = f'merge base {base}' if base is not None else 'no common ancestor'
    print(
        f'ratchet commit gate: finishing a merge, so {baseline} is held to the '
        f'3-way bound of HEAD and MERGE_HEAD over {over}.'
    )

    def image(oid: str | None, name: str) -> Path | None:
        return None if oid is None else _image(root, oid, scratch / name)

    raises = metrics.compare_merged_baseline_files(
        base=image(base_oid, 'base.json'),
        ours=image(ours_oid, 'ours.json'),
        theirs=image(theirs_oid, 'theirs.json'),
        current=_image(root, staged_oid, scratch / 'current.json'),
    )
    refusals = _refuse_unrecorded(raises, own_entries)
    if not refusals:
        return []
    return [
        *refusals,
        'FINISHING A MERGE where both sides raised one key: the bound is the '
        "higher side, so the composed value (the ratchet-baseline driver's, or "
        'a regeneration\'s) is this merge\'s own raise -- and --write-baseline '
        'records nothing over a file that already holds it. Run `git checkout '
        f'HEAD -- {metrics.BASELINE_RELPATH}` (ours; `--ours` is a no-op once the '
        'driver resolved the file), then the --write-baseline --authorize-raise '
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
    merge_head = _object_id(root, 'MERGE_HEAD')
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
            own_entries = _own_entries(root, parents)
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
