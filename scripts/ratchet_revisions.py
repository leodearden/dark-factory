"""Read the merge-lane ratchet's committed artifacts out of git, at a revision or the index.

``merge_lane_metrics`` deliberately has no git dependency: every comparison
lives there, pure and testable on dicts. The two gates that enforce it need
the artifacts as git holds them -- ``scripts/check_staged_ratchet_raise.py`` at
a commit (HEAD, MERGE_HEAD, the index), ``scripts/check_ratchet_advance.py`` at a
move of main (its merge base, the new tip). Every git invocation either gate
makes lives here, so neither gate reaches into the other, and the policy each
applies stays the instrument's.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import merge_lane_metrics as metrics

#: The committed artifacts the gates are about, named once: the baseline, the
#: ledger directory, and the retired ledger whose split and burial they audit.
ARTIFACTS = (
    metrics.BASELINE_RELPATH,
    metrics.LEDGER_DIR_RELPATH,
    metrics.LEGACY_LEDGER_RELPATH,
)


def git(root: Path, *args: str, stdin: str | None = None) -> str:
    """Every git invocation the ratchet gates make.

    ``check=True``: a plumbing call that fails is an instrument failure, and
    each gate's ``main`` reports it as one. The two exceptions are answers
    rather than faults, and each is read in exactly one place: a revision that
    names nothing (:func:`object_id`, whose ``--quiet`` makes absence an exit
    code instead of a message), and two histories with no common ancestor
    (:func:`merge_base`). ``encoding`` is pinned so a blob round trip does not
    depend on the calling shell's locale.
    """
    return subprocess.run(
        ['git', '-C', str(root), *args],
        input=stdin,
        check=True,
        text=True,
        encoding='utf-8',
        capture_output=True,
    ).stdout


def object_id(root: Path, revision: str) -> str | None:
    """The object *revision* names, or None when it names nothing.

    Absence is an ANSWER here, not a fault: an unborn HEAD, the commit that
    introduces the baseline, a ledger not yet committed, a commit that is not
    finishing a merge and so has no MERGE_HEAD. This is the one rev-parse site
    where a non-zero git exit is read as data.
    """
    try:
        return git(root, 'rev-parse', '--verify', '--quiet', revision).strip()
    except subprocess.CalledProcessError:
        return None


def merge_base(root: Path, a: str, b: str) -> str | None:
    """The commit *a* and *b* are merged over, or None when none exists.

    None is DATA: unrelated histories have no common ancestor, and git merges
    them against the empty tree. ``git merge-base`` says so with rc 1 and no
    output, so exactly that is read as absence; any other failure is the
    instrument failure a gate's ``main`` reports.
    """
    try:
        return git(root, 'merge-base', a, b).strip()
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 1 and not (exc.stdout or exc.stderr):
            return None
        raise


def write_blob(root: Path, oid: str, target: Path) -> Path:
    """Write the blob *oid* to *target* so a path-taking loader can read it."""
    target.write_text(git(root, 'cat-file', 'blob', oid), encoding='utf-8')
    return target


def blob_text(root: Path, oid: str) -> str:
    return git(root, 'cat-file', 'blob', oid)


def restores_previous_image(root: Path, staged_oid: str, history_tip: str) -> str | None:
    """The commit whose baseline blob *staged_oid* restores, if it is the LAST one.

    *history_tip* names the history the rule reads: HEAD for a commit, the merge
    base for a move of main.

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
    commits = git(
        root, 'rev-list', '--full-history', history_tip, '--', baseline
    ).split()
    if not commits:
        return None
    resolved = git(
        root,
        'cat-file',
        '--batch-check',
        stdin='\n'.join(f'{commit}:{baseline}' for commit in commits),
    ).splitlines()
    tip_oid = object_id(root, f'{history_tip}:{baseline}')

    # Newest first. Skip the run of entries still holding the tip's blob -- those
    # are commits that touched the path without changing its value -- and let
    # the FIRST remaining entry answer, whatever it is. A `missing` line (the
    # path was deleted, or did not yet exist) has the revision string as its
    # first field, so it can never equal an oid: a deletion therefore ENDS the
    # walk rather than being skipped, with no special case for it.
    for commit, line in zip(commits, resolved, strict=False):
        oid = line.split()[:1]
        if oid == [tip_oid]:
            continue
        return commit if oid == [staged_oid] else None
    return None


def entry_listing(root: Path, revision: str | None) -> dict[str, str]:
    """The ledger's entry files at *revision* (None: the index), as name -> blob id.

    ``_README.json`` is left out: it records no authorization, and it is the
    one file there that may change. A revision that names nothing -- an unborn
    HEAD -- holds no entries. An index entry still unmerged is a fault, not an
    answer: the gate never guesses which stage a resolution meant.
    """
    ledger = metrics.LEDGER_DIR_RELPATH
    if revision is None:
        listing = git(root, 'ls-files', '--stage', '-z', '--', ledger)
    elif object_id(root, revision) is None:
        return {}
    else:
        listing = git(root, 'ls-tree', '-r', '-z', '--full-tree', revision, '--', ledger)
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


def own_entries(root: Path, parents: list[str], tip: str | None = None) -> list[dict]:
    """The OWN ledger entries of *tip* (None: the index) over *parents*, refusing any rewrite.

    Resolved whenever a ratchet artifact changes, not only when the baseline
    moved: the append-only promise is broken precisely by a change that touches
    the ledger ALONE, so an audit conditioned on the baseline would leave it
    unenforced in the one case that breaks it (heuristic 10, uniformly). The
    refusals are ``metrics``' own ``AppendOnlyViolation`` verdicts, which each
    gate puts on its policy rung.
    """
    staged = entry_listing(root, tip)
    own_names = metrics.own_entry_names(
        [entry_listing(root, parent) for parent in parents], staged
    )
    own = [metrics.parse_entry(name, blob_text(root, staged[name])) for name in own_names]
    return metrics.retire_legacy_ledger(retired_entries(root, parents, tip), staged, own)


def retired_entries(root: Path, parents: list[str], tip: str | None = None) -> list[dict]:
    """The retired array's entries in every parent that held it, if *tip* drops it.

    Holding it while some parent lacks it -- a merge resolution that kept a
    branch's copy of the array -- buries an entry where nothing reads it, so
    that is refused outright rather than retired.
    """
    legacy = metrics.LEGACY_LEDGER_RELPATH
    held = {parent: object_id(root, f'{parent}:{legacy}') for parent in parents}
    if object_id(root, f'{tip or ""}:{legacy}') is not None:
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
        for entry in metrics.legacy_ledger_entries(blob_text(root, oid))
    ]
