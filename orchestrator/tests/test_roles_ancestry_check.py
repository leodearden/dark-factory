"""Behavioral + regression-guard tests for `ANCESTRY_CHECK_INSTRUCTIONS`.

Task 4107: the rc=1 arm of the canonical `git merge-base --is-ancestor`
check (spliced into both STEWARD "Marking tasks done" call sites) told the
steward that if this checkout's `main` might be behind, run
`git -C <project_root> fetch --all` and re-run before concluding the SHA is
off main. That remedy cannot work: `fetch` writes only `refs/remotes/*` and
the object store -- it never moves the local `refs/heads/main` this check
actually compares against. `refs/heads/main` is a single ref in the shared
common `.git` dir, advanced only by the merge worker's own local
`update-ref` CAS. So during the merge worker's lag window, the prescribed
fetch is a strict no-op: the steward re-runs into the same rc=1 and reaches
exactly the false "the SHA really is off main" verdict the block exists to
prevent. The correct remedy is temporal (confirm via
`mcp__escalation__merge_status` and re-run), not sync-based.

`test_fetch_all_cannot_advance_local_main_ref` pins that root cause against
real git, mirroring the "root-cause-as-spec" pattern in
`test_roles_staging_command.py`: it drives real git in a throwaway
upstream+clone pair and asserts on real refs and a real exit code. The
prompt prose itself -- the wording of the rc=1 arm in
`ANCESTRY_CHECK_INSTRUCTIONS` -- is deliberately NOT pinned by a test here.
A test that asserts on substrings of a comment block exercises no runtime
behaviour; it only pins prose, which pressures the wording toward whatever
passes the assertion rather than toward what is clearest to the reader. The
real-git test above is what actually protects this change.

A second defect in the same comment block -- a false claim that the
SKILL.md ancestry-check blocks still carry the silent-rc gap this block's
`echo "ancestry rc=$rc"` fixes -- is a pure source-comment correction with
no runtime behaviour. It is fixed alongside this module but gets no test
here: the only possible test would grep skills/merge-queue/SKILL.md and
skills/unblock/SKILL.md for the echo form, which is a documentation
meta-test that would couple this suite to prose in two skill docs that
legitimately get rewritten.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def _init_upstream_repo(repo: Path) -> None:
    subprocess.run(['git', 'init', '-b', 'main'], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'test@test.com'], cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=repo, check=True, capture_output=True)
    (repo / 'f.txt').write_text('one\n')
    subprocess.run(['git', 'add', 'f.txt'], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ['git', '-c', 'commit.gpgsign=false', 'commit', '-m', 'initial'],
        cwd=repo, check=True, capture_output=True,
    )


def test_fetch_all_cannot_advance_local_main_ref(tmp_path: Path) -> None:
    """Root-cause-as-spec: `git fetch --all` cannot move a clone's local `main`.

    Mirrors `test_legacy_exclusion_form_exits_one_root_cause` in
    test_roles_staging_command.py: runs the exact remedy the (unfixed)
    rc=1 arm prescribes against real git, so a future git behavior change
    (not just intuition) would catch it if this premise ever stopped
    holding. Validated empirically during planning: local `main` stayed at
    its pre-fetch SHA while `refs/remotes/origin/main` advanced, and
    `--is-ancestor` against the new tip still exited 1.
    """
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    _init_upstream_repo(upstream)

    clone = tmp_path / 'clone'
    clone_result = subprocess.run(
        ['git', 'clone', str(upstream), str(clone)], capture_output=True, text=True,
    )
    assert clone_result.returncode == 0, (
        f'git clone failed: stdout={clone_result.stdout!r} stderr={clone_result.stderr!r}'
    )

    local_main_before = subprocess.run(
        ['git', 'rev-parse', 'refs/heads/main'],
        cwd=clone, check=True, capture_output=True, text=True,
    ).stdout.strip()

    # Advance upstream's main with a second commit -- simulates the merge
    # worker landing a merge while this checkout's `main` sits behind.
    (upstream / 'f.txt').write_text('two\n')
    subprocess.run(['git', 'add', 'f.txt'], cwd=upstream, check=True, capture_output=True)
    subprocess.run(
        ['git', '-c', 'commit.gpgsign=false', 'commit', '-m', 'second'],
        cwd=upstream, check=True, capture_output=True,
    )
    upstream_new_tip = subprocess.run(
        ['git', 'rev-parse', 'main'], cwd=upstream, check=True, capture_output=True, text=True,
    ).stdout.strip()

    fetch_result = subprocess.run(['git', 'fetch', '--all'], cwd=clone, capture_output=True, text=True)
    assert fetch_result.returncode == 0, (
        f'git fetch --all failed: stdout={fetch_result.stdout!r} stderr={fetch_result.stderr!r}'
    )

    local_main_after = subprocess.run(
        ['git', 'rev-parse', 'refs/heads/main'],
        cwd=clone, check=True, capture_output=True, text=True,
    ).stdout.strip()
    remote_main_after = subprocess.run(
        ['git', 'rev-parse', 'refs/remotes/origin/main'],
        cwd=clone, check=True, capture_output=True, text=True,
    ).stdout.strip()

    assert local_main_after == local_main_before, (
        f'git fetch --all moved the local main ref: {local_main_before} -> {local_main_after} -- '
        'if this ever changes, the roles.py rc=1 remedy this test protects needs revisiting'
    )
    assert remote_main_after == upstream_new_tip, (
        f'expected refs/remotes/origin/main to advance to {upstream_new_tip}, got {remote_main_after}'
    )

    ancestor_result = subprocess.run(
        ['git', 'merge-base', '--is-ancestor', upstream_new_tip, 'main'],
        cwd=clone, capture_output=True, text=True,
    )
    assert ancestor_result.returncode == 1, (
        'expected --is-ancestor to still exit 1 against the un-advanced local main after '
        f'fetch --all; got returncode={ancestor_result.returncode} '
        f'stdout={ancestor_result.stdout!r} stderr={ancestor_result.stderr!r}'
    )
