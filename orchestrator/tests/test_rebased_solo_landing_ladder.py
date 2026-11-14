"""Real-git pins for the ladder's rebased-solo-landing arm (task 6936).

When main moves under a solo merge request's verify,
``orchestrator/src/orchestrator/git_ops.py::GitOps.advance_main`` takes its
rebased path: it rebases the branch onto main and fast-forwards main to the
rebased tip.  No merge commit is written, every branch commit is rewritten,
and the task ref stays at its pre-rebase tip.  ``skills/_shared/deriving-landed-sha.md``
reads that shape on its rc=1 arm through three checks, given the request's
recorded landing sha M: (a) M is an ancestor of main; (b) M is '='-marked on
main's side of ``git log --cherry-mark --left-only --no-merges main...task/<ID>``,
i.e. attribution by patch-equivalence; (c) ``git cherry main task/<ID>`` is
non-empty and every line starts with '-'.  Each premise is pinned here:

* **Reachability.** The shape lands on rc=1 with no marker and a retained ref.
* **Content proof.** Check (c) holds, and production's
  ``orchestrator/src/orchestrator/merge_lane/landing_evidence.py::patch_content_contained``
  agrees with it.
* **Attribution.** Check (b) marks M, an uncited ``amend:`` tip, and never an
  unrelated direct-to-main commit that is equally an ancestor of main.
* **A later commit.** Work added to the branch after the landing shows as '+'.
* **Context drift.** A GENUINE landing can still show '+', because patch-id
  hashes context lines.  That is why a '+' held against a request-scoped
  ``done`` reads as landing-unattributable, never as not-landed.
* **Empty output.** ``git cherry`` skips merges, so a branch whose only commit
  beyond main is a sync merge prints nothing.  ``patch_content_contained``
  reads that as contained; the ladder deliberately does NOT, and requires
  non-empty output (the same rule as ``skills/merge-queue/SKILL.md`` rule 2b).

These are characterization pins, expected to pass on write: a failure means
git's behaviour moved and skills/_shared/deriving-landed-sha.md's
rebased-solo-landing arm must be re-litigated.  Prose greps of the runbook
are deliberately absent; see ``orchestrator/tests/test_roles_ancestry_check.py``'s
module docstring.  Every fixture repo carries the esc-3072-3 isolation guards
through ``_ladder_git.py::new_repo``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from _ladder_git import commit, git, git_rc, new_repo

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.landing_evidence import patch_content_contained

TID = 'T'
BRANCH = f'task/{TID}'


@dataclass(frozen=True)
class RebasedLanding:
    """The two main-side commits a rebased solo landing's checks discriminate."""

    merge_sha: str
    direct_sha: str


def _rebase_onto_main_and_fast_forward(root: Path) -> str:
    """Land ``task/T`` the way the worker's rebased path does; return main's new tip.

    A scratch branch carries the rebase so ``task/T`` keeps its pre-rebase
    tip, as the worker's retained ref does.
    """
    git(root, 'checkout', '-b', 'scratch', BRANCH)
    git(root, 'rebase', 'main')
    git(root, 'checkout', 'main')
    git(root, 'merge', '--ff-only', 'scratch')
    git(root, 'branch', '-D', 'scratch')
    return git(root, 'rev-parse', 'main')


def _build_rebased_solo_landing(root: Path) -> RebasedLanding:
    """A two-commit task branch, landed rebased past one direct-to-main commit.

    The tip deliberately cites no task: only 117 of the last 300 real
    task-branch tips cite their task, so an uncited tip is the common case.
    """
    new_repo(root)
    commit(root, 'base', 'base.txt')
    git(root, 'checkout', '-b', BRANCH)
    commit(root, f'impl: first (task {TID})', 'a.txt')
    commit(root, 'amend: second', 'b.txt')
    git(root, 'checkout', 'main')
    direct_sha = commit(root, 'docs: direct-to-main', 'd.md')
    return RebasedLanding(
        merge_sha=_rebase_onto_main_and_fast_forward(root), direct_sha=direct_sha,
    )


def _marker_search(root: Path) -> list[str]:
    """Step 1's selecting search: every sha on main whose SUBJECT is the marker."""
    subject = f'Merge {BRANCH} into main'
    out = git(root, 'log', 'main', '--fixed-strings', f'--grep={subject}', '--format=%H%x09%s')
    rows = (line.split('\t', 1) for line in out.splitlines())
    return [sha for sha, row_subject in rows if row_subject == subject]


def _cherry_lines(root: Path) -> list[str]:
    """Check (c)'s ``git cherry main task/T`` output, one entry per line."""
    return git(root, 'cherry', 'main', BRANCH).splitlines()


def _cherry_marked_main_side(root: Path) -> list[str]:
    """Check (b)'s listing: main's side of ``main...task/T``, '=' marking equivalents."""
    return git(
        root, 'log', '--cherry-mark', '--left-only', '--no-merges', '--format=%m %H',
        f'main...{BRANCH}',
    ).splitlines()


def _production_contained(root: Path) -> bool:
    """Drive the merge worker's ``patch_content_contained`` for ``task/T`` against main."""
    git_ops = GitOps(GitConfig(main_branch='main', branch_prefix='task/'), root)
    return asyncio.run(patch_content_contained(BRANCH, 'main', git_ops))


def test_a_rebased_solo_landing_reaches_rc1_with_no_marker_and_a_retained_ref(
    tmp_path: Path,
) -> None:
    """The shape the arm exists for: the ladder arrives at rc=1 empty-handed."""
    root = tmp_path / 'repo'
    _build_rebased_solo_landing(root)

    assert git_rc(root, 'merge-base', '--is-ancestor', BRANCH, 'main') == 1, (
        'every branch commit was rewritten and the ref kept its pre-rebase tip, so the '
        'branch is not an ancestor of main: the ladder lands on its rc=1 arm'
    )
    assert _marker_search(root) == [], (
        'a rebased fast-forward writes no merge commit, so step 1 finds no marker'
    )
    assert int(git(root, 'rev-list', '--count', f'main..{BRANCH}')) > 0, (
        'main..<branch> counts by sha, and the pre-rebase shas are not on main; that '
        'count is why GitOps._delete_branch_if_on_main retains the ref'
    )


def test_every_rebased_commit_is_patch_equivalent_on_main(tmp_path: Path) -> None:
    """Check (c) holds on a clean rebase, and production's predicate agrees."""
    root = tmp_path / 'repo'
    _build_rebased_solo_landing(root)

    lines = _cherry_lines(root)
    assert lines, 'the content proof requires NON-EMPTY git cherry output'
    assert all(line.startswith('-') for line in lines), (
        f'every branch commit must have a patch-equivalent image on main: {lines}'
    )
    assert _production_contained(root) is True, (
        "patch_content_contained must agree with the ladder's git cherry reading"
    )


def test_the_recorded_merge_sha_is_attributed_by_patch_equivalence(tmp_path: Path) -> None:
    """Check (b) marks the worker's merge_sha '=' although its subject cites no task."""
    root = tmp_path / 'repo'
    landing = _build_rebased_solo_landing(root)

    assert git_rc(root, 'merge-base', '--is-ancestor', landing.merge_sha, 'main') == 0
    assert git(root, 'log', '-1', '--format=%s', landing.merge_sha) == 'amend: second', (
        'merge_sha must be the rebased image of the UNCITED tip, so the pin shows '
        'attribution needs no subject citation'
    )
    listing = _cherry_marked_main_side(root)
    assert listing.count(f'= {landing.merge_sha}') == 1, (
        f'merge_sha must be listed exactly once, marked patch-equivalent: {listing}'
    )


def test_an_unrelated_main_commit_is_never_attributed(tmp_path: Path) -> None:
    """Check (b) refuses a commit that is merely on main, which check (a) cannot."""
    root = tmp_path / 'repo'
    landing = _build_rebased_solo_landing(root)

    assert git_rc(root, 'merge-base', '--is-ancestor', landing.direct_sha, 'main') == 0, (
        'the direct-to-main commit passes ancestry, exactly like merge_sha'
    )
    listing = _cherry_marked_main_side(root)
    direct_rows = [row for row in listing if row.endswith(landing.direct_sha)]
    assert len(direct_rows) == 1, f"the direct commit must be listed on main's side: {listing}"
    assert not direct_rows[0].startswith('='), (
        f'an unrelated main commit must never be marked patch-equivalent: {direct_rows[0]!r}'
    )


def test_a_commit_added_after_the_landing_shows_as_plus(tmp_path: Path) -> None:
    """New branch work after the landing is the one '+' line, and production agrees."""
    root = tmp_path / 'repo'
    _build_rebased_solo_landing(root)
    git(root, 'checkout', BRANCH)
    late = commit(root, 'late: third', 'c.txt')

    plus_lines = [line for line in _cherry_lines(root) if line.startswith('+')]
    assert plus_lines == [f'+ {late}'], (
        f'exactly the post-landing commit must be reported as not on main: {plus_lines}'
    )
    assert _production_contained(root) is False


def _commit_text(root: Path, subject: str, filename: str, text: str) -> str:
    """Write *text* to *filename*, commit it with *subject*, and return the new sha."""
    (root / filename).write_text(text)
    git(root, 'add', filename)
    git(root, 'commit', '-m', subject)
    return git(root, 'rev-parse', 'HEAD')


def _numbered_lines(edits: dict[int, str]) -> str:
    """An 8-line file of ``line N``, with the 1-based line numbers in *edits* replaced."""
    return ''.join(f'{edits.get(n, f"line {n}")}\n' for n in range(1, 9))


def test_context_drift_shows_a_genuinely_landed_commit_as_plus(tmp_path: Path) -> None:
    """A '+' does not prove not-landed: patch-id hashes the hunk's context lines.

    The branch edits line 5 and main's direct commit edits line 3, inside
    the branch hunk's three lines of context.  The rebase applies cleanly and
    the work is on main, yet the rebased image's patch-id differs.
    """
    root = tmp_path / 'repo'
    new_repo(root)
    _commit_text(root, 'base', 'shared.txt', _numbered_lines({}))
    git(root, 'checkout', '-b', BRANCH)
    landed = _commit_text(
        root, f'impl: edit line 5 (task {TID})', 'shared.txt', _numbered_lines({5: 'branch 5'}),
    )
    git(root, 'checkout', 'main')
    _commit_text(root, 'docs: edit line 3', 'shared.txt', _numbered_lines({3: 'main 3'}))

    merge_sha = _rebase_onto_main_and_fast_forward(root)

    assert git(root, 'show', 'main:shared.txt') == _numbered_lines(
        {3: 'main 3', 5: 'branch 5'},
    ).rstrip('\n'), 'the rebase applied cleanly: the branch edit is genuinely on main'
    assert git(root, 'log', '-1', '--format=%s', merge_sha) == f'impl: edit line 5 (task {TID})'
    assert _cherry_lines(root) == [f'+ {landed}'], (
        'git cherry must report the GENUINELY landed commit as not on main; if this '
        'stops holding, patch-id stopped hashing context lines'
    )


def test_a_sync_merge_only_branch_prints_nothing_where_production_reads_contained(
    tmp_path: Path,
) -> None:
    """The one deliberate divergence: empty cherry output is no content proof.

    The branch's only commit beyond main is a ``--no-ff`` merge of main into
    it.  ``git cherry`` skips merges, so it prints nothing, and
    ``patch_content_contained`` (no '+' line) answers contained.  The ladder
    requires NON-EMPTY output, so it does not read this as landed.
    """
    root = tmp_path / 'repo'
    new_repo(root)
    commit(root, 'base', 'base.txt')
    git(root, 'branch', BRANCH)
    commit(root, 'docs: direct-to-main', 'd.md')
    git(root, 'checkout', BRANCH)
    git(root, 'merge', '--no-ff', 'main', '-m', f'Merge main into {BRANCH}')

    assert git_rc(root, 'merge-base', '--is-ancestor', BRANCH, 'main') == 1, (
        'the sync merge is not on main, so this branch is on the rc=1 arm'
    )
    assert _cherry_lines(root) == [], 'git cherry skips merges, so it prints NOTHING'
    assert _production_contained(root) is True, (
        'production reads the empty output as contained; the ladder deliberately does not'
    )
