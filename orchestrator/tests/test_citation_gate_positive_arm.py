"""Root-cause-as-spec pin for the citation gate's POSITIVE arm (task 4924).

Task 4924 fixed the runbooks' citation-gate positive arm: on the rc=0 arm
of the landed-sha ladder, a subject-matching row on main used to license
stamping ``done_provenance.commit`` = the BRANCH TIP
(``git rev-parse task/<TASK_ID>``).  That is wrong whenever the branch
never advanced past its creation point: its tip is main's own OLD BASE
commit, carrying none of this task's work, yet
``git merge-base --is-ancestor`` -- the server's only backstop -- passes
it trivially.  The gate can still fire POSITIVE there, because
``DEFAULT_COMMIT_CITATION_PATTERN`` carries an UNANCHORED ``\\(#?{tid}\\)``
alternative, so a *sibling* task's landing commit whose subject mentions
``(#<TASK_ID>)`` satisfies it.  The fix stamps the CITING sha instead --
byte-identical to what
``escalation/src/escalation/server.py::_found_on_main_response`` returns
on the same evidence.

This module pins that MECHANISM, not the runbooks' wording.  Both tests
are expected to PASS on write: they are characterization pins that make
the failure scenario executable, so a future change to the citation
pattern or to git's behaviour re-litigates the fix's premises instead of
silently invalidating them.

A test that grepped ``skills/_shared/deriving-landed-sha.md`` or
``skills/*/SKILL.md`` for the corrected stamp shape is deliberately
ABSENT.  Per ``orchestrator/tests/test_roles_ancestry_check.py``'s module
docstring, such a test "exercises no runtime behaviour; it only pins
prose, which pressures the wording toward whatever passes the assertion
rather than toward what is clearest to the reader", and that docstring
names grepping ``skills/merge-queue/SKILL.md`` and
``skills/unblock/SKILL.md`` as the specific anti-pattern to avoid because
it "would couple this suite to prose in two skill docs that legitimately
get rewritten" -- the same docs task 4924 edits.  The real-git tests here
are what actually protect the change.

Every citation-gate claim here is made TWICE against the same repo, and
the two are asserted to AGREE:

1. through the REAL
   ``orchestrator/src/orchestrator/git_ops.py::GitOps.find_task_citation_commit``
   (``_real_citation_walk``), so the pin tracks production rather than a
   frozen copy of it; and
2. through ``_citation_walk``, a transcription of what
   ``skills/_shared/deriving-landed-sha.md``'s step-4 citation gate tells
   an agent to TYPE -- ``git log main --extended-regexp --grep=<pattern>``
   as a coarse full-message pre-filter, then the same pattern re-applied
   to each candidate's SUBJECT alone, most-recent-first, first match wins.

The shell form is deliberately NOT byte-identical to the method: the
method reads ``-z`` NUL-separated records, while the runbook prescribes a
human-readable ``--format='%H %s'`` listing an agent eyeballs. That gap is
exactly why the two are asserted to agree instead of the transcription
standing alone. Should the walk's semantics move under either of them --
the subject-anchoring work that
``fused-memory/scripts/audit_found_on_main_provenance.py``'s header tracks
as ``plans/found-on-main-provenance-integrity-prd.md`` label delta, a
``--max-count`` cap, a narrowed pattern -- the disagreement fails HERE,
where the runbook edit that must follow is one file away, instead of both
copies passing quietly while the behaviour this fix rests on has moved.

``GitOps`` is constructed directly from a default ``GitConfig``, the way
``orchestrator/tests/test_git_ops.py::TestFindTaskCitationCommit`` does
through its ``git_ops``/``git_repo`` fixtures; those fixtures are not
reused here because each test builds its own purpose-shaped history. The
async method is driven with ``asyncio.run`` so these stay ordinary sync
tests.

No sleeps, no network, no skips: if git is unavailable these fail loudly
rather than silently skipping (the repo's no-silent-fail-soft invariant).
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path

from orchestrator.config import GitConfig
from orchestrator.git_ops import DEFAULT_COMMIT_CITATION_PATTERN, GitOps

TASK_ID = '4924'

# Deterministic identity + no signing, so subjects and shas are stable and
# the tests never depend on the developer's git config.
_GIT_ENV_ARGS = [
    '-c', 'user.name=Test',
    '-c', 'user.email=test@test.com',
    '-c', 'commit.gpgsign=false',
]


def _git(root: Path, *args: str) -> str:
    """Run git in *root* and return stripped stdout (raises on non-zero)."""
    return subprocess.run(
        ['git', *_GIT_ENV_ARGS, *args],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _git_rc(root: Path, *args: str) -> int:
    """Run git in *root* and return only its exit code."""
    return subprocess.run(
        ['git', *_GIT_ENV_ARGS, *args], cwd=root, capture_output=True, text=True,
    ).returncode


def _commit(root: Path, subject: str, content: str) -> str:
    """Write `f.txt`, commit it with *subject*, and return the new sha."""
    (root / 'f.txt').write_text(content)
    _git(root, 'add', 'f.txt')
    _git(root, 'commit', '-m', subject)
    return _git(root, 'rev-parse', 'HEAD')


def _new_repo(root: Path) -> None:
    root.mkdir()
    _git(root, 'init', '-b', 'main')


def _real_citation_walk(root: Path, tid: str) -> str | None:
    """Drive the REAL `GitOps.find_task_citation_commit` over *root*.

    A default `GitConfig` (main_branch='main', branch_prefix='task/',
    commit_citation_pattern=None -> DEFAULT_COMMIT_CITATION_PATTERN) is
    everything the method reads, so no fixture plumbing is needed.
    `asyncio.run` keeps the callers ordinary sync tests.
    """
    git_ops = GitOps(GitConfig(main_branch='main', branch_prefix='task/'), root)
    return asyncio.run(git_ops.find_task_citation_commit(tid))


def _citation_walk(root: Path, tid: str) -> str | None:
    """The citation gate as `skills/_shared/deriving-landed-sha.md` spells it in SHELL.

    Coarse `--grep` full-message pre-filter, then re-test each candidate's
    SUBJECT with the same pattern compiled as a Python `re`, walking
    most-recent-first; the first subject match wins.

    NOT a copy of the method -- that one reads `-z` NUL-separated records,
    while the runbook prescribes a readable listing. Callers assert this
    AGREES with `_real_citation_walk` on the same repo, which is what
    keeps the runbook's transcription honest as production moves.
    """
    pattern_str = DEFAULT_COMMIT_CITATION_PATTERN.format(tid=re.escape(tid))
    compiled = re.compile(pattern_str)
    out = _git(
        root, 'log', 'main', '--extended-regexp',
        f'--grep={pattern_str}', '--format=%H%x1f%s',
    )
    for record in out.split('\n'):
        if not record:
            continue
        sha, _, subject = record.partition('\x1f')
        if compiled.search(subject):
            return sha.strip()
    return None


def _marker_search(root: Path, tid: str) -> str:
    """Step 1's exact-subject merge-marker search (empty == no marker)."""
    return _git(
        root, 'log', 'main', '--fixed-strings',
        f'--grep=Merge task/{tid} into main', '--max-count=1', '--format=%H',
    )


def test_sibling_covered_phantom_branch_fires_the_gate_with_a_workless_tip(
    tmp_path: Path,
) -> None:
    """A branch that never advanced reaches the rc=0 arm AND fires the gate.

    This is the exact repo state task 4924's fix exists for: the ladder
    falls through steps 1-3 (no marker, no rev-list candidate) to step 4's
    rc=0 citation gate, the gate fires POSITIVE off a *sibling's* landing
    commit, and the tip the pre-fix ladder prescribed as
    `done_provenance.commit` is a commit that neither cites this task nor
    carries any of its work.
    """
    root = tmp_path / 'repo'
    _new_repo(root)

    # B: a realistic branch base -- some *other* task's landing on main.
    base = _commit(root, 'Merge task/4700 into main', 'one\n')

    # The task branch is created at B and never advances.
    _git(root, 'branch', f'task/{TASK_ID}', base)

    # S: a sibling's landing commit on main, citing this task via the
    # unanchored `\(#?{tid}\)` alternative.
    _git(root, 'checkout', 'main')
    sibling = _commit(root, f'impl(4701): sweep provenance (#{TASK_ID})', 'two\n')

    # Step 4: degenerate ancestry passes trivially.
    assert _git_rc(root, 'merge-base', '--is-ancestor', f'task/{TASK_ID}', 'main') == 0, (
        'a branch that never advanced must still pass --is-ancestor against main; '
        'if this stops holding, the rc=0 arm is no longer reachable in this state'
    )

    # Step 1: no merge marker, so the ladder does not stop there.
    assert _marker_search(root, TASK_ID) == '', (
        'expected no `Merge task/<ID> into main` marker on main for this task'
    )

    # Step 3: the group/train-merge probe yields no candidate either.
    assert _git(
        root, 'rev-list', '--ancestry-path', '--merges', f'task/{TASK_ID}..main',
    ) == '', 'expected no ancestry-path merge candidate for a branch that never advanced'

    # Step 4's citation gate nevertheless fires POSITIVE -- off the sibling.
    # Claimed through the REAL method, then through the runbook's shell
    # transcription, then that the two agree (see this module's docstring).
    assert _real_citation_walk(root, TASK_ID) == sibling, (
        'expected GitOps.find_task_citation_commit to select the sibling commit; the '
        'whole defect is that this arm cannot tell a sibling-covered landing from a '
        'fast-forward'
    )
    assert _citation_walk(root, TASK_ID) == _real_citation_walk(root, TASK_ID), (
        "the runbook's shell citation gate and GitOps.find_task_citation_commit must "
        'agree; a disagreement means production moved and '
        'skills/_shared/deriving-landed-sha.md step 4 needs the same change'
    )

    # ...and the sha the PRE-FIX ladder would have stamped is worthless.
    tip = _git(root, 'rev-parse', f'task/{TASK_ID}')
    assert tip == base, 'the phantom branch tip must be main\'s own old base commit'
    assert tip != sibling, 'the tip and the citing sha must be distinct commits'
    compiled = re.compile(DEFAULT_COMMIT_CITATION_PATTERN.format(tid=re.escape(TASK_ID)))
    base_subject = _git(root, 'log', '-1', '--format=%s', base)
    assert not compiled.search(base_subject), (
        f'the branch tip subject {base_subject!r} must NOT cite task {TASK_ID} -- that is '
        'why stamping the tip earns `misattributed` from '
        'fused-memory/scripts/audit_found_on_main_provenance.py::classify'
    )


def test_genuine_fast_forward_citing_commit_is_on_main_and_may_precede_the_tip(
    tmp_path: Path,
) -> None:
    """The safety proof for the fix: the citing sha is right in the HONEST case too.

    On a genuine fast-forward the citing commit is on main and cites the
    task, so switching `commit` from the tip to the citing sha loses
    nothing.  It also shows the two shas diverge ordinarily -- a trailing
    non-citing commit on the branch is enough -- so a fix that preserved
    the tip form "just for real fast-forwards" would still disagree with
    `escalation/src/escalation/server.py::_found_on_main_response`.
    """
    root = tmp_path / 'repo'
    _new_repo(root)

    _commit(root, 'Merge task/4700 into main', 'one\n')

    _git(root, 'checkout', '-b', f'task/{TASK_ID}')
    citing = _commit(root, f'impl({TASK_ID}): real work', 'two\n')
    trailing = _commit(root, 'docs: tweak', 'three\n')

    # Fast-forward main to the branch tip.
    _git(root, 'checkout', 'main')
    _git(root, 'merge', '--ff-only', f'task/{TASK_ID}')
    assert _git(root, 'rev-parse', 'main') == trailing

    assert _real_citation_walk(root, TASK_ID) == citing, (
        'GitOps.find_task_citation_commit must select this branch\'s own citing '
        'commit, not the trailing non-citing tip'
    )
    assert _citation_walk(root, TASK_ID) == _real_citation_walk(root, TASK_ID), (
        "the runbook's shell citation gate and GitOps.find_task_citation_commit must "
        'agree; a disagreement means production moved and '
        'skills/_shared/deriving-landed-sha.md step 4 needs the same change'
    )

    tip = _git(root, 'rev-parse', f'task/{TASK_ID}')
    assert tip == trailing
    assert citing != tip, (
        'citing sha and branch tip differ even on an honest fast-forward whenever the '
        'last commit\'s subject does not cite the task -- an ordinary shape, not an '
        'exotic one'
    )

    assert _git_rc(root, 'merge-base', '--is-ancestor', citing, 'main') == 0, (
        'the citing sha must itself be on main -- it is on main by construction '
        '(`git log main` found it), which is what makes the stamp auditable'
    )
