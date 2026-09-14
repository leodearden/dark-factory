"""Tests for ``orchestrator.rebase_recovery`` — guarded rebase/merge abort (task 4797).

Two defects motivate the module under test, and both were reproduced
first-hand in throwaway repos on git 2.43.0 before any code was written:

* A MERGE_RR record naming a conflict id whose ``rr-cache/<id>`` directory is
  absent makes ``git rebase --abort`` die in ``rerere_clear()`` — measured
  ``Segmentation fault (core dumped)``, rc 139, with the rebase state left
  fully in place and a fresh ``MERGE_RR.lock`` created.
* A stale ``MERGE_RR.lock`` with a perfectly INTACT rr-cache makes the same
  abort fail rc 128 with git's "Another git process seems to be running"
  advice.  This is an independent failure, not the crash's residue.

Nothing here asserts the segfault.  It is an upstream git bug that this task
puts out of scope, so pinning rc 139 would turn a future git patch into a red
suite — the fix would read as the regression.  What is asserted is the
version-independent post-condition: the preflight DETECTS the dangling id, and
the guarded abort then returns rc 0 with the evidence preserved.

The paired negative control matters as much as the positive case.  An "abort
works" assertion against a HEALTHY worktree passes vacuously — the unguarded
abort works there too — so every behavioural claim below is made against a
deliberately dangling fixture with an intact-fixture control beside it.

ISOLATION (esc-3072-3).  This module builds real conflicted repositories and
deliberately deletes ``rr-cache`` directories, which is precisely the shape of
write that once landed in a live task worktree: git's repository discovery
walks UP, and pytest's basetemp can sit inside one.  Both layers of the house
defence are applied to every git subprocess spawned here —
:func:`assert_isolated_git_repo` first (pure filesystem, so a rejected call
writes nothing anywhere), then ``env=`` :func:`git_env_with_ceiling` (so escape
is impossible at the git level even if the pre-flight is refactored away).
"""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import re
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

from orchestrator import git_ops as git_ops_module
from orchestrator import rebase_recovery
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps

# ---------------------------------------------------------------------------
# Real-git fixture scaffolding
# ---------------------------------------------------------------------------

_BASE = 'line1\nline2\nline3\n'
_FEATURE = 'line1\nFEATURE\nline3\n'
_MAIN = 'line1\nMAIN\nline3\n'


def _run_argv(repo: Path, argv) -> subprocess.CompletedProcess:
    """Run an arbitrary command vector inside *repo*, under both isolation layers.

    Takes the whole vector rather than trailing arguments so a caller can pass
    ``rebase_recovery.RECOVERY_GIT`` itself, instead of re-spelling the prefix
    the production code already defines.
    """
    assert_isolated_git_repo(repo)
    return subprocess.run(
        list(argv),
        cwd=str(repo),
        capture_output=True,
        text=True,
        env=git_env_with_ceiling(repo),
    )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Run one plain git command inside *repo*."""
    return _run_argv(repo, ['git', *args])


def _git_ok(repo: Path, *args: str) -> str:
    proc = _git(repo, *args)
    assert proc.returncode == 0, f'git {args!r} failed: {proc.stderr}'
    return proc.stdout


def build_mid_rebase_repo(root: Path, name: str = 'repo') -> tuple[Path, str]:
    """Build an isolated repo left mid-rebase with a populated MERGE_RR.

    Creates a deterministic two-branch content conflict on one line of one
    file, then rebases ``feature`` onto ``main`` so the rebase stops on that
    conflict.  rerere is enabled, so git writes ``MERGE_RR`` naming the
    conflict id and a backing ``rr-cache/<id>/`` directory — the substrate the
    dangling-ref cases manufacture by deleting.

    Returns ``(repo, conflict_id)``.  The conflict id is the LITERAL token from
    MERGE_RR, carrying any rerere variant suffix verbatim.
    """
    repo = root / name
    repo.mkdir(parents=True)

    # The one call that cannot pre-assert: the directory is not a repository
    # until this command makes it one.  ``git init`` creates a repo at the path
    # given rather than reusing an enclosing one, and the ceiling is applied
    # regardless; the assertion inside ``_git`` covers every call after it.
    init = subprocess.run(
        ['git', 'init', '-q', '-b', 'main', '.'],
        cwd=str(repo), capture_output=True, text=True,
        env=git_env_with_ceiling(repo),
    )
    assert init.returncode == 0, f'git init failed: {init.stderr}'
    assert_isolated_git_repo(repo)

    _git_ok(repo, 'config', 'user.email', 'rebase-recovery@test.invalid')
    _git_ok(repo, 'config', 'user.name', 'Rebase Recovery Test')
    _git_ok(repo, 'config', 'commit.gpgsign', 'false')
    _git_ok(repo, 'config', 'rerere.enabled', 'true')
    _git_ok(repo, 'config', 'rerere.autoupdate', 'true')

    (repo / 'f.txt').write_text(_BASE)
    _git_ok(repo, 'add', 'f.txt')
    _git_ok(repo, 'commit', '-qm', 'base')

    _git_ok(repo, 'checkout', '-qb', 'feature')
    (repo / 'f.txt').write_text(_FEATURE)
    _git_ok(repo, 'commit', '-qam', 'feature edit')

    _git_ok(repo, 'checkout', '-q', 'main')
    (repo / 'f.txt').write_text(_MAIN)
    _git_ok(repo, 'commit', '-qam', 'main edit')

    _git_ok(repo, 'checkout', '-q', 'feature')
    rebase = _git(repo, 'rebase', 'main')
    assert rebase.returncode != 0, 'fixture expected a rebase conflict'

    git_dir = repo / '.git'
    assert (git_dir / 'rebase-merge').is_dir(), 'fixture expected mid-rebase state'

    merge_rr = (git_dir / 'MERGE_RR').read_bytes()
    conflict_id = merge_rr.split(b'\x00')[0].split(b'\t')[0].decode()
    assert (git_dir / 'rr-cache' / conflict_id).is_dir(), (
        'fixture expected a backing rr-cache entry'
    )
    return repo, conflict_id


# ---------------------------------------------------------------------------
# The MERGE_RR grammar
# ---------------------------------------------------------------------------

class TestParseMergeRr:
    """MERGE_RR's real on-disk grammar, measured rather than assumed.

    Records are NUL-TERMINATED ``<id>\\t<path>\\0``, not newline-separated, and
    the id is ``<40-hex>[.<variant>]`` — git appends a ``.N`` suffix when one
    conflict has several rerere variants.  Both details are load-bearing: a
    parser that splits on newlines finds nothing in a real file, and one that
    normalizes the id away reports a dangling ref as intact.

    The byte literals below are taken verbatim from live files in this repo's
    ``.git/worktrees/*/MERGE_RR`` (read-only observation).
    """

    def test_empty_file_yields_no_records(self) -> None:
        parsed = rebase_recovery.parse_merge_rr(b'')
        assert parsed.records == ()
        assert parsed.unparsable == ()

    def test_single_nul_terminated_record(self) -> None:
        data = b'd233fdd99096e62540dc6cb96cae57c25398fa57\tshared/src/shared/mcp_markup_middleware.py\x00'
        parsed = rebase_recovery.parse_merge_rr(data)
        assert parsed.unparsable == ()
        assert len(parsed.records) == 1
        assert parsed.records[0].conflict_id == 'd233fdd99096e62540dc6cb96cae57c25398fa57'
        assert parsed.records[0].path == 'shared/src/shared/mcp_markup_middleware.py'

    def test_variant_suffix_is_retained_verbatim(self) -> None:
        """The ``.1`` is part of the rr-cache directory NAME, never a decoration.

        Live state of worktree 29171: MERGE_RR cites ``...648.1`` while
        ``rr-cache/...648.1`` is ABSENT and the bare ``rr-cache/...648`` is
        PRESENT.  A parser that strips the suffix here hands the classifier a
        token that resolves, turning a genuinely dangling ref into a false
        "intact" — the exact false negative that lets the crash through.
        """
        data = (
            b'd932b0e1e48d84453c25373f569e77581b8cc648.1\t'
            b'fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py\x00'
        )
        parsed = rebase_recovery.parse_merge_rr(data)
        assert parsed.unparsable == ()
        assert parsed.records[0].conflict_id == (
            'd932b0e1e48d84453c25373f569e77581b8cc648.1'
        )

    def test_multiple_records_including_a_path_with_a_space(self) -> None:
        data = (
            b'a' * 40 + b'\tsrc/one.py\x00'
            + b'b' * 40 + b'\tdocs/a file with spaces.md\x00'
            + b'c' * 40 + b'.12\tsrc/three.py\x00'
        )
        parsed = rebase_recovery.parse_merge_rr(data)
        assert parsed.unparsable == ()
        assert [r.conflict_id for r in parsed.records] == [
            'a' * 40, 'b' * 40, 'c' * 40 + '.12',
        ]
        assert parsed.records[1].path == 'docs/a file with spaces.md'

    def test_corrupt_records_surface_as_unparsable_without_raising(self) -> None:
        """``read_rr()`` calls ``die("corrupt MERGE_RR")`` on these shapes.

        A malformed record is a second way recovery fails hard, so it is
        reported rather than raised: this helper decorates a RECOVERY path and
        must never itself become the reason recovery fails.
        """
        data = (
            b'deadbeef\tsrc/short-id.py\x00'          # id too short
            + b'e' * 40 + b'src/no-tab.py\x00'        # missing the tab
            + b'f' * 40 + b'\tsrc/good.py\x00'        # still parsed
        )
        parsed = rebase_recovery.parse_merge_rr(data)
        assert [r.conflict_id for r in parsed.records] == ['f' * 40]
        assert parsed.unparsable == (
            b'deadbeef\tsrc/short-id.py',
            b'e' * 40 + b'src/no-tab.py',
        )

    def test_trailing_bytes_without_a_nul_are_not_dropped(self) -> None:
        """A truncated final record is evidence of damage, not something to skip."""
        parsed = rebase_recovery.parse_merge_rr(b'a' * 40 + b'\tsrc/one.py')
        assert [r.conflict_id for r in parsed.records] == ['a' * 40]


# ---------------------------------------------------------------------------
# Dangling-vs-intact classification
# ---------------------------------------------------------------------------

_HEX = 'd932b0e1e48d84453c25373f569e77581b8cc648'


def _plant_merge_rr(git_dir: Path, *records: bytes) -> None:
    git_dir.mkdir(parents=True, exist_ok=True)
    (git_dir / 'MERGE_RR').write_bytes(b''.join(r + b'\x00' for r in records))


def _record(conflict_id: str, path: str = 'src/one.py') -> bytes:
    return f'{conflict_id}\t{path}'.encode()


class TestScanMergeRr:
    """Classify each MERGE_RR record against its backing rr-cache directory.

    The scan is filesystem-only and takes ``git_dir`` and ``common_dir`` as
    explicit arguments, so these cases need plain directories rather than real
    repositories.  That separation is deliberate: it keeps the classifier
    testable without a repo, and it makes the common-dir resolution an
    assertable property rather than an implementation detail.
    """

    def test_variant_suffix_dangling_while_bare_id_exists(self, tmp_path: Path) -> None:
        """THE load-bearing case — the measured live state of worktree 29171.

        MERGE_RR cites ``<hex>.1``; ``rr-cache/<hex>`` EXISTS and
        ``rr-cache/<hex>.1`` does NOT.  Verdict must be DANGLING.  A checker
        that strips or splits the suffix resolves the bare directory, reports
        intact, and lets exactly the crash this module guards against through.
        """
        git_dir = tmp_path / 'gitdir'
        _plant_merge_rr(git_dir, _record(f'{_HEX}.1'))
        (git_dir / 'rr-cache' / _HEX).mkdir(parents=True)

        scan = rebase_recovery.scan_merge_rr(git_dir=git_dir, common_dir=git_dir)

        assert [r.conflict_id for r in scan.dangling] == [f'{_HEX}.1']

    def test_present_directory_classifies_intact(self, tmp_path: Path) -> None:
        git_dir = tmp_path / 'gitdir'
        _plant_merge_rr(git_dir, _record(_HEX))
        (git_dir / 'rr-cache' / _HEX).mkdir(parents=True)

        scan = rebase_recovery.scan_merge_rr(git_dir=git_dir, common_dir=git_dir)

        assert scan.dangling == ()
        assert [r.conflict_id for r in scan.records] == [_HEX]

    def test_absent_directory_classifies_dangling(self, tmp_path: Path) -> None:
        git_dir = tmp_path / 'gitdir'
        _plant_merge_rr(git_dir, _record(_HEX))
        (git_dir / 'rr-cache').mkdir(parents=True)

        scan = rebase_recovery.scan_merge_rr(git_dir=git_dir, common_dir=git_dir)

        assert [r.conflict_id for r in scan.dangling] == [_HEX]

    def test_rr_cache_entry_that_is_a_file_classifies_dangling(
        self, tmp_path: Path,
    ) -> None:
        """git wants a DIRECTORY holding preimage/postimage; a file is not one."""
        git_dir = tmp_path / 'gitdir'
        _plant_merge_rr(git_dir, _record(_HEX))
        (git_dir / 'rr-cache').mkdir(parents=True)
        (git_dir / 'rr-cache' / _HEX).write_text('not a directory')

        scan = rebase_recovery.scan_merge_rr(git_dir=git_dir, common_dir=git_dir)

        assert [r.conflict_id for r in scan.dangling] == [_HEX]

    def test_rr_cache_resolves_under_the_common_dir_not_the_worktree_git_dir(
        self, tmp_path: Path,
    ) -> None:
        """A linked worktree's MERGE_RR is per-worktree; its rr-cache is shared.

        Resolving rr-cache under the per-worktree git dir would find nothing
        for EVERY record in a linked worktree — reporting the whole file
        dangling and quarantining healthy state on every run.
        """
        git_dir = tmp_path / 'worktrees' / '4797'
        common_dir = tmp_path / 'common'
        _plant_merge_rr(git_dir, _record(_HEX))
        (common_dir / 'rr-cache' / _HEX).mkdir(parents=True)
        (git_dir / 'rr-cache').mkdir(parents=True)

        scan = rebase_recovery.scan_merge_rr(git_dir=git_dir, common_dir=common_dir)

        assert scan.dangling == ()

    def test_unparsable_record_is_reported_as_suspect(self, tmp_path: Path) -> None:
        git_dir = tmp_path / 'gitdir'
        _plant_merge_rr(git_dir, b'deadbeef\tsrc/short.py', _record(_HEX))
        (git_dir / 'rr-cache' / _HEX).mkdir(parents=True)

        scan = rebase_recovery.scan_merge_rr(git_dir=git_dir, common_dir=git_dir)

        assert scan.dangling == ()
        assert scan.unparsable == (b'deadbeef\tsrc/short.py',)
        assert scan.suspect is True

    def test_missing_merge_rr_is_the_healthy_case(self, tmp_path: Path) -> None:
        """Absence is normal, not an error: most worktrees have no MERGE_RR."""
        git_dir = tmp_path / 'gitdir'
        git_dir.mkdir()

        scan = rebase_recovery.scan_merge_rr(git_dir=git_dir, common_dir=git_dir)

        assert scan.records == ()
        assert scan.dangling == ()
        assert scan.unparsable == ()
        assert scan.suspect is False

    def test_dangling_records_make_the_scan_suspect(self, tmp_path: Path) -> None:
        git_dir = tmp_path / 'gitdir'
        _plant_merge_rr(git_dir, _record(_HEX))

        scan = rebase_recovery.scan_merge_rr(git_dir=git_dir, common_dir=git_dir)

        assert scan.suspect is True


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------

class TestQuarantineMergeRr:
    """MERGE_RR is MOVED aside, never unlinked.

    The file is the only record of which conflict ids a wedged worktree was
    carrying, and a SUCCESSFUL abort DELETES it (measured).  So the quarantine
    is not crash-avoidance — the ``RECOVERY_GIT`` prefix already covers that —
    it is what stops the repair from destroying its own evidence.
    """

    def test_dangling_scan_moves_the_file_preserving_its_bytes(
        self, tmp_path: Path,
    ) -> None:
        git_dir = tmp_path / 'gitdir'
        _plant_merge_rr(git_dir, _record(_HEX))
        original = (git_dir / 'MERGE_RR').read_bytes()

        scan = rebase_recovery.scan_merge_rr(git_dir=git_dir, common_dir=git_dir)
        backup = rebase_recovery.quarantine_merge_rr(scan)

        assert backup is not None
        assert not (git_dir / 'MERGE_RR').exists()
        assert backup.read_bytes() == original
        assert backup.parent == git_dir

    def test_a_second_quarantine_does_not_clobber_the_first(
        self, tmp_path: Path,
    ) -> None:
        """Two wedged runs leave two backups; the earlier evidence survives."""
        git_dir = tmp_path / 'gitdir'
        _plant_merge_rr(git_dir, _record(_HEX, 'src/first.py'))
        first = rebase_recovery.quarantine_merge_rr(
            rebase_recovery.scan_merge_rr(git_dir=git_dir, common_dir=git_dir),
        )

        _plant_merge_rr(git_dir, _record(_HEX, 'src/second.py'))
        second = rebase_recovery.quarantine_merge_rr(
            rebase_recovery.scan_merge_rr(git_dir=git_dir, common_dir=git_dir),
        )

        assert first is not None and second is not None
        assert first != second
        assert b'src/first.py' in first.read_bytes()
        assert b'src/second.py' in second.read_bytes()

    def test_intact_scan_leaves_the_file_exactly_where_it_was(
        self, tmp_path: Path,
    ) -> None:
        git_dir = tmp_path / 'gitdir'
        _plant_merge_rr(git_dir, _record(_HEX))
        (git_dir / 'rr-cache' / _HEX).mkdir(parents=True)
        original = (git_dir / 'MERGE_RR').read_bytes()

        scan = rebase_recovery.scan_merge_rr(git_dir=git_dir, common_dir=git_dir)
        backup = rebase_recovery.quarantine_merge_rr(scan)

        assert backup is None
        assert (git_dir / 'MERGE_RR').read_bytes() == original
        assert list(git_dir.glob('MERGE_RR.*')) == []

    def test_unparsable_record_also_quarantines(self, tmp_path: Path) -> None:
        git_dir = tmp_path / 'gitdir'
        _plant_merge_rr(git_dir, b'deadbeef\tsrc/short.py')

        backup = rebase_recovery.quarantine_merge_rr(
            rebase_recovery.scan_merge_rr(git_dir=git_dir, common_dir=git_dir),
        )

        assert backup is not None
        assert not (git_dir / 'MERGE_RR').exists()

    def test_warning_names_every_dangling_conflict_id(
        self, tmp_path: Path, caplog,
    ) -> None:
        """Assert on the IDS, not on sentence wording — the ids are the payload."""
        git_dir = tmp_path / 'gitdir'
        other = 'a' * 40
        _plant_merge_rr(
            git_dir, _record(f'{_HEX}.1', 'src/one.py'), _record(other, 'src/two.py'),
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.rebase_recovery'):
            backup = rebase_recovery.quarantine_merge_rr(
                rebase_recovery.scan_merge_rr(git_dir=git_dir, common_dir=git_dir),
            )

        logged = '\n'.join(r.getMessage() for r in caplog.records)
        assert f'{_HEX}.1' in logged
        assert other in logged
        assert backup is not None and str(backup) in logged


# ---------------------------------------------------------------------------
# Stale-lock sweep
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 9, 11, 12, 0, 0, tzinfo=UTC)
_STALE = 3600


def _plant_lock(git_dir: Path, name: str, *, age_seconds: float) -> Path:
    git_dir.mkdir(parents=True, exist_ok=True)
    lock = git_dir / name
    lock.touch()
    stamp = (_NOW - timedelta(seconds=age_seconds)).timestamp()
    os.utime(lock, (stamp, stamp))
    return lock


def _swept(git_dir: Path):
    return rebase_recovery.sweep_stale_locks(
        git_dir=git_dir, now=_NOW, stale_after_seconds=_STALE,
    )


class TestSweepStaleLocks:
    """Removal requires the CONJUNCTION of no holder AND age past threshold.

    Age alone must never authorise removal: a legitimately held lock can be
    arbitrarily old, and deleting it corrupts whatever still holds it.
    """

    def test_old_and_unheld_is_removed(self, tmp_path: Path) -> None:
        lock = _plant_lock(tmp_path / 'gitdir', 'MERGE_RR.lock', age_seconds=_STALE * 2)

        swept = _swept(tmp_path / 'gitdir')

        assert [s.path for s in swept.removed] == [lock]
        assert swept.retained == ()
        assert not lock.exists()

    def test_old_but_HELD_is_retained(self, tmp_path: Path) -> None:
        """The case age alone gets wrong, and the reason the conjunction exists."""
        lock = _plant_lock(tmp_path / 'gitdir', 'MERGE_RR.lock', age_seconds=_STALE * 2)

        with lock.open('a'):
            swept = _swept(tmp_path / 'gitdir')

        assert swept.removed == ()
        assert [s.path for s in swept.retained] == [lock]
        assert swept.retained[0].holder_pids == (os.getpid(),)
        assert lock.exists()

    def test_young_and_unheld_is_retained(self, tmp_path: Path) -> None:
        lock = _plant_lock(tmp_path / 'gitdir', 'MERGE_RR.lock', age_seconds=5)

        swept = _swept(tmp_path / 'gitdir')

        assert swept.removed == ()
        assert [s.path for s in swept.retained] == [lock]
        assert swept.retained[0].holder_pids == ()
        assert lock.exists()

    def test_young_and_held_is_retained(self, tmp_path: Path) -> None:
        lock = _plant_lock(tmp_path / 'gitdir', 'MERGE_RR.lock', age_seconds=5)

        with lock.open('a'):
            swept = _swept(tmp_path / 'gitdir')

        assert swept.removed == ()
        assert lock.exists()

    def test_verdict_ignores_the_mtime_of_the_operation_the_lock_blocks(
        self, tmp_path: Path,
    ) -> None:
        """THE ordering trap from incident 3517, pinned so it cannot come back.

        That lock's mtime (16:52:47) was OLDER than the rebase-merge directory
        it blocked (17:33:50).  So "newer than the operation" would have
        cleared a lock it must keep, and "older than the operation" would have
        kept one it must clear — the relative comparison is wrong in BOTH
        directions and is banned outright.  With a rebase-merge dir planted
        NEWER than each lock, both verdicts must be unchanged from the cases
        above: the blocked operation's mtime is never consulted.
        """
        git_dir = tmp_path / 'gitdir'
        old = _plant_lock(git_dir, 'MERGE_RR.lock', age_seconds=_STALE * 2)
        young = _plant_lock(git_dir, 'index.lock', age_seconds=5)
        rebase_merge = git_dir / 'rebase-merge'
        rebase_merge.mkdir()
        fresh = _NOW.timestamp()
        os.utime(rebase_merge, (fresh, fresh))

        swept = _swept(git_dir)

        assert [s.path for s in swept.removed] == [old]
        assert [s.path for s in swept.retained] == [young]

    def test_zero_byte_lock_is_not_treated_specially(self, tmp_path: Path) -> None:
        """Size is not a liveness signal: git's locks are empty while held.

        The incident's lock was 0 bytes AND stale; a 0-byte lock held right now
        is indistinguishable by size and must still be retained.
        """
        git_dir = tmp_path / 'gitdir'
        lock = _plant_lock(git_dir, 'MERGE_RR.lock', age_seconds=_STALE * 2)
        assert lock.stat().st_size == 0

        with lock.open('a'):
            held = _swept(git_dir)
        unheld = _swept(git_dir)

        assert held.removed == ()
        assert [s.path for s in unheld.removed] == [lock]

    def test_removal_logs_both_the_age_and_the_holder_finding(
        self, tmp_path: Path, caplog,
    ) -> None:
        """Both halves of the conjunction are evidence, so both are reported."""
        git_dir = tmp_path / 'gitdir'
        lock = _plant_lock(git_dir, 'MERGE_RR.lock', age_seconds=7200)

        with caplog.at_level(logging.WARNING, logger='orchestrator.rebase_recovery'):
            _swept(git_dir)

        logged = '\n'.join(r.getMessage() for r in caplog.records)
        assert str(lock) in logged
        assert '7200' in logged
        assert 'holder' in logged.lower()

    def test_only_lock_files_directly_under_the_git_dir_are_considered(
        self, tmp_path: Path,
    ) -> None:
        git_dir = tmp_path / 'gitdir'
        _plant_lock(git_dir, 'MERGE_RR.lock', age_seconds=_STALE * 2)
        nested = git_dir / 'refs'
        plain = _plant_lock(git_dir, 'ORIG_HEAD', age_seconds=_STALE * 2)
        deep = _plant_lock(nested, 'heads.lock', age_seconds=_STALE * 2)

        swept = _swept(git_dir)

        assert plain.exists()
        assert deep.exists()
        assert [s.path.name for s in swept.removed] == ['MERGE_RR.lock']


# ---------------------------------------------------------------------------
# End-to-end: the dangling-ref recovery this task exists for
# ---------------------------------------------------------------------------

def _make_dangling(repo: Path, conflict_id: str) -> None:
    """Delete the rr-cache directory MERGE_RR points at, leaving the ref dangling."""
    shutil.rmtree(repo / '.git' / 'rr-cache' / conflict_id)


class TestPreflightEndToEnd:
    """The full guarded-recovery path against a real wedged repository.

    What is deliberately NOT asserted: that an UNGUARDED ``git rebase --abort``
    segfaults here.  It does on git 2.43.0 — measured rc 139 — but that is an
    upstream ``rerere_clear()`` bug this task puts out of scope, and pinning it
    would turn the day the host's git is patched into a red suite, with the fix
    reading as the regression.  The guarded post-condition asserted instead is
    version-independent and still fails loudly if the preflight regresses.
    """

    def test_dangling_ref_is_detected_quarantined_and_the_abort_recovers(
        self, tmp_path: Path,
    ) -> None:
        repo, conflict_id = build_mid_rebase_repo(tmp_path)
        _make_dangling(repo, conflict_id)
        pre_rebase_tip = _git_ok(repo, 'rev-parse', 'feature').strip()

        result = rebase_recovery.preflight_rebase_recovery(repo)

        assert [r.conflict_id for r in result.dangling] == [conflict_id]
        assert result.verdict == 'repaired'

        assert result.merge_rr_backup is not None
        assert not (repo / '.git' / 'MERGE_RR').exists()
        assert conflict_id.encode() in result.merge_rr_backup.read_bytes()

        abort = _run_argv(repo, [*rebase_recovery.RECOVERY_GIT, 'rebase', '--abort'])
        assert abort.returncode == 0, abort.stderr

        assert not (repo / '.git' / 'rebase-merge').exists()
        assert _git_ok(repo, 'status', '--porcelain') == ''
        assert _git_ok(repo, 'rev-parse', 'HEAD').strip() == pre_rebase_tip
        assert _git_ok(repo, 'rev-parse', '--abbrev-ref', 'HEAD').strip() == 'feature'
        assert result.merge_rr_backup.exists(), 'evidence must outlive the abort'

    def test_healthy_mid_rebase_worktree_is_left_untouched(
        self, tmp_path: Path,
    ) -> None:
        """The negative control that stops the case above passing vacuously.

        Same fixture, rr-cache INTACT.  If the preflight reported dangling refs
        here it would quarantine healthy state on every wedged rebase in the
        fleet, and the assertion above would hold no matter what the classifier
        did.
        """
        repo, conflict_id = build_mid_rebase_repo(tmp_path)
        original = (repo / '.git' / 'MERGE_RR').read_bytes()

        result = rebase_recovery.preflight_rebase_recovery(repo)

        assert result.dangling == ()
        assert result.merge_rr_backup is None
        assert result.verdict == 'clean'
        assert (repo / '.git' / 'MERGE_RR').read_bytes() == original
        assert list((repo / '.git').glob('MERGE_RR.quarantined-*')) == []


# ---------------------------------------------------------------------------
# git_ops wiring
# ---------------------------------------------------------------------------

async def _isolated_run(cmd, cwd=None, **kwargs) -> tuple[int, str, str]:
    """An :data:`rebase_recovery.AbortRunner` that keeps both isolation layers.

    ``guarded_abort`` takes its runner as a PARAMETER, so a caller supplies one
    rather than reaching into git_ops for the private ``_run`` it happens to
    pass.  Here that parameter earns its keep twice over: these cases run real
    aborts against real conflicted repos, and routing them through
    :func:`_run_argv` keeps them inside the module's isolation contract.
    """
    assert cwd is not None
    proc = _run_argv(Path(cwd), cmd)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _recording_run(recorded: list[list[str]]):
    """A runner that records the command vector and spawns nothing.

    Lets a case assert on command STRUCTURE — token order and membership —
    rather than on a rendered line a harmless reflow would break.
    """
    async def run(cmd, cwd=None, **kwargs) -> tuple[int, str, str]:
        recorded.append(list(cmd))
        return 0, '', ''

    return run


@contextlib.contextmanager
def _guard_spy():
    """Record every ``(verb, cwd)`` git_ops routes through the public guard.

    Patches :func:`rebase_recovery.guarded_abort` — the seam the two modules
    genuinely share — so the wiring cases are phrased in the vocabulary of that
    interface instead of reaching through git_ops' private ``_run``.  The real
    guard still runs underneath, so the abort these cases observe is the one
    production issues.
    """
    recorded: list[tuple[str, Path]] = []
    real_guard = rebase_recovery.guarded_abort

    async def recording_guard(verb, cwd, run):
        recorded.append((verb, Path(cwd)))
        return await real_guard(verb, cwd, run)

    with patch.object(rebase_recovery, 'guarded_abort', side_effect=recording_guard):
        yield recorded


def _make_git_ops(repo: Path):
    config = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )
    return GitOps(config, repo)


@pytest.mark.asyncio
class TestGitOpsGuardedAbort:
    """Every abort git_ops issues on a recovery path carries the guard."""

    async def test_guarded_abort_recovers_a_dangling_mid_rebase_worktree(
        self, tmp_path: Path,
    ) -> None:
        """The behavioural arm: real repo, real dangling ref, real recovery."""
        repo, conflict_id = build_mid_rebase_repo(tmp_path)
        _make_dangling(repo, conflict_id)

        rc, _, err = await rebase_recovery.guarded_abort(
            'rebase', repo, _isolated_run,
        )

        assert rc == 0, err
        assert not (repo / '.git' / 'rebase-merge').exists()
        assert _git_ok(repo, 'status', '--porcelain') == ''
        backups = list((repo / '.git').glob('MERGE_RR.quarantined-*'))
        assert len(backups) == 1
        assert conflict_id.encode() in backups[0].read_bytes()

    async def test_guarded_abort_emits_the_rerere_disabling_prefix(
        self, tmp_path: Path,
    ) -> None:
        """Asserted by token ORDER, never by matching a rendered command line."""
        repo, _ = build_mid_rebase_repo(tmp_path)
        recorded: list[list[str]] = []

        await rebase_recovery.guarded_abort('rebase', repo, _recording_run(recorded))

        assert recorded == [[*rebase_recovery.RECOVERY_GIT, 'rebase', '--abort']]
        assert recorded[0].index('rerere.enabled=false') < recorded[0].index('rebase')

    async def test_preflight_runs_BEFORE_the_abort(self, tmp_path: Path) -> None:
        """Ordering is the contract: a preflight after the abort guards nothing.

        The abort DELETES MERGE_RR, so a preflight that ran afterwards would
        find an empty worktree, report clean, and quarantine nothing — passing
        every state assertion while preserving no evidence at all.
        """
        repo, conflict_id = build_mid_rebase_repo(tmp_path)
        _make_dangling(repo, conflict_id)
        observed: list[str] = []

        real_preflight = rebase_recovery.preflight_rebase_recovery

        def spy_preflight(worktree, **kwargs):
            observed.append('preflight')
            return real_preflight(worktree, **kwargs)

        async def spy_run(cmd, cwd=None, **kwargs):
            observed.append('abort')
            return await _isolated_run(cmd, cwd=cwd, **kwargs)

        with patch.object(
            rebase_recovery, 'preflight_rebase_recovery', side_effect=spy_preflight,
        ):
            await rebase_recovery.guarded_abort('rebase', repo, spy_run)

        assert observed == ['preflight', 'abort']
        backups = list((repo / '.git').glob('MERGE_RR.quarantined-*'))
        assert len(backups) == 1, 'the preflight that ran first kept the evidence'
        assert conflict_id.encode() in backups[0].read_bytes()

    async def test_rebase_onto_main_failure_path_aborts_through_the_guard(
        self, tmp_path: Path,
    ) -> None:
        repo, _ = build_mid_rebase_repo(tmp_path)
        _git_ok(repo, 'rebase', '--abort')
        ops = _make_git_ops(repo)

        with _guard_spy() as recorded:
            landed = await ops.rebase_onto_main(repo)

        assert landed is False, 'fixture expected the rebase to conflict'
        assert recorded == [('rebase', repo)]

    async def test_abort_merge_aborts_through_the_same_guard(
        self, tmp_path: Path,
    ) -> None:
        """Included for UNIFORMITY, and the docstring says why.

        ``git merge --abort`` was measured NOT to crash on a dangling ref
        (rc 0), so this site needs no crash-avoidance.  It still consumes
        MERGE_RR and can still hit the stale-lock rc 128, and a per-site
        carve-out is a rule a future reader has to re-derive before they can
        safely touch any of the four.
        """
        repo, _ = build_mid_rebase_repo(tmp_path)
        _git_ok(repo, 'rebase', '--abort')
        ops = _make_git_ops(repo)

        with _guard_spy() as recorded:
            await ops.abort_merge(repo)

        assert recorded == [('merge', repo)]


class TestGitOpsAbortUniformity:
    """SPOT, enforced against the FILE rather than against known call sites."""

    def test_no_unguarded_abort_vector_survives_anywhere_in_git_ops(self) -> None:
        """SPOT, enforced against the file rather than against known call sites.

        Four sites route through one guard precisely so a future edit cannot
        fix three and miss the fourth.  A per-site spy cannot see that: it
        asserts about the sites it already knows, so a newly ADDED fifth
        unguarded abort passes it silently.  Scanning the source closes that,
        and it is the only assertion here that gets stronger as the file grows.

        The guard lives in ``rebase_recovery``, so git_ops should now spell an
        abort ONLY as a call to it and carry no ``--abort`` literal of its own.
        Both halves are asserted: a bare literal is the regression, and the
        count of routed sites is what stops the scan passing vacuously if a
        future edit deletes the calls rather than guarding them.
        """
        source = Path(git_ops_module.__file__).read_text()
        quoted_abort = re.compile(r"""['"]--abort['"]""")
        offenders = [
            line.strip()
            for line in source.splitlines()
            if quoted_abort.search(line) and 'RECOVERY_GIT' not in line
        ]
        assert offenders == []
        routed = [
            line for line in source.splitlines()
            if 'rebase_recovery.guarded_abort(' in line
        ]
        assert len(routed) == 4, routed


# ---------------------------------------------------------------------------
# Fail-safe: the guard never becomes the reason recovery fails
# ---------------------------------------------------------------------------

class TestVanishedWorktreeKeepsTheTypedException:
    """A worktree deleted out-of-band must still surface as ``WorktreeMissing``.

    The orchestrator races humans who delete a task worktree mid-flight, and
    two consumers pattern-match the typed exception to recover: merge_queue's
    ``except WorktreeMissing`` logs ``exc.path``, cleans up the merge worktree
    and surfaces the task ``blocked``; steward's auto-escalates naming
    ``exc.path``.  Both read ``.path``, and a bare ``FileNotFoundError``
    carries neither the type nor the attribute — it escapes to a broader
    handler with a different disposition and no worktree cleanup.

    Inserting a preflight AHEAD of the abort put a second subprocess spawn in
    front of ``_run``'s own typed pre-flight check, so the generic error now
    wins the race.  ``WorktreeMissing`` subclasses ``FileNotFoundError``, so
    this discriminates: the parent is not an instance of the subclass.

    :meth:`TestPreflightCli.test_an_unresolvable_worktree_does_not_crash_the_cli`
    does not cover this.  It points at a directory that EXISTS but is not a
    repository — git runs and exits non-zero — whereas here git never spawns
    at all.
    """

    @pytest.mark.asyncio
    async def test_abort_on_a_vanished_worktree_raises_the_typed_exception(
        self, tmp_path: Path,
    ) -> None:
        """Through ``GitOps.abort_merge``: the production wiring, unmocked."""
        vanished = tmp_path / 'deleted-out-of-band'
        ops = _make_git_ops(tmp_path)

        with pytest.raises(git_ops_module.WorktreeMissing) as caught:
            await ops.abort_merge(vanished)

        assert caught.value.path == vanished

    def test_preflight_on_a_vanished_worktree_reports_unresolved(
        self, tmp_path: Path,
    ) -> None:
        """The unit arm: an unspawnable cwd degrades, it does not raise."""
        vanished = tmp_path / 'deleted-out-of-band'

        result = rebase_recovery.preflight_rebase_recovery(vanished)

        assert result.resolved is False
        assert result.verdict == rebase_recovery.VERDICT_CLEAN

def _quarantine_rename_fails(monkeypatch, error: OSError) -> None:
    """Make the quarantine's ``rename`` of MERGE_RR fail with a chosen errno.

    Monkeypatched rather than ``chmod``ed.  ``chmod`` is a no-op for root, so a
    permission-bit fixture asserts nothing wherever CI runs as root — the same
    vacuous pass this module's own header warns about for "abort works" on a
    healthy worktree.  A monkeypatch is deterministic and root-independent, and
    it models the likelier race more directly anyway: a concurrent process
    unlinking MERGE_RR between the scan's ``read_bytes`` and the quarantine's
    ``rename`` produces an errno, not a permission change.

    Scoped to the MERGE_RR name so every other rename in the process — pytest's
    own bookkeeping included — still works.
    """
    real_rename = Path.rename

    def rename(self: Path, target):
        if self.name == 'MERGE_RR':
            raise error
        return real_rename(self, target)

    monkeypatch.setattr(Path, 'rename', rename)


class TestQuarantineFailureDoesNotSwallowTheAbort:
    """A repair this module cannot perform degrades into the RESULT, not an exception.

    The module exists to stop a recovery path failing hard, so a preflight that
    raises makes it the NEW reason recovery fails — strictly worse than having
    no preflight at all.  Measured on this branch with a read-only git dir:
    the quarantine's unguarded ``rename`` raised ``PermissionError`` out through
    ``guarded_abort``, THE ABORT NEVER RAN, and the worktree was left wedged.

    ``sweep_stale_locks`` already had the right shape — ``except OSError``
    around ``unlink``, counting the lock as retained — so the module's two
    mutating repairs degraded differently for no reason a reader could derive.
    """

    @pytest.mark.asyncio
    async def test_the_abort_is_still_issued_and_the_evidence_survives(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Nothing escapes, the guarded vector is still emitted, MERGE_RR stays.

        The runner records instead of spawning, so the abort that would
        otherwise DELETE MERGE_RR does not run — which is what lets the same
        case assert both that the abort was issued and that a failed move
        destroyed no evidence.
        """
        repo, conflict_id = build_mid_rebase_repo(tmp_path)
        _make_dangling(repo, conflict_id)
        merge_rr = repo / '.git' / 'MERGE_RR'
        original = merge_rr.read_bytes()
        _quarantine_rename_fails(
            monkeypatch, PermissionError(errno.EACCES, 'Permission denied'),
        )
        recorded: list[list[str]] = []

        rc, _, _ = await rebase_recovery.guarded_abort(
            'rebase', repo, _recording_run(recorded),
        )

        assert rc == 0
        assert recorded == [[*rebase_recovery.RECOVERY_GIT, 'rebase', '--abort']]
        assert recorded[0].index('rerere.enabled=false') < recorded[0].index('rebase')
        assert merge_rr.read_bytes() == original

    def test_the_failure_is_reported_as_unrepaired_and_logged(
        self, tmp_path: Path, monkeypatch, caplog,
    ) -> None:
        """``guarded_abort`` discards the result, so the report is asserted here.

        No new reporting machinery is needed for this: a suspect scan with no
        backup is already rendered by ``unrepaired`` and already turns the
        verdict ``blocked``, so the operator is told precisely what was left
        un-repaired while the abort proceeds regardless.
        """
        repo, conflict_id = build_mid_rebase_repo(tmp_path)
        _make_dangling(repo, conflict_id)
        merge_rr = repo / '.git' / 'MERGE_RR'
        _quarantine_rename_fails(
            monkeypatch, PermissionError(errno.EACCES, 'Permission denied'),
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.rebase_recovery'):
            result = rebase_recovery.preflight_rebase_recovery(repo)

        assert result.merge_rr_backup is None
        assert result.verdict == rebase_recovery.VERDICT_BLOCKED
        assert conflict_id in ' '.join(result.unrepaired)
        assert merge_rr.exists()
        logged = '\n'.join(r.getMessage() for r in caplog.records)
        assert str(merge_rr) in logged

def _merge_rr_read_fails(monkeypatch, error: OSError) -> None:
    """Make reading MERGE_RR fail with a chosen errno, by the same means as above.

    Monkeypatched for the reasons :func:`_quarantine_rename_fails` gives, plus
    one this case adds: it pins ``IsADirectoryError`` and ``PermissionError``
    as DISTINCT states, where a real fixture would hand back whichever errno
    the filesystem and euid happened to produce.

    Scoped to the MERGE_RR name, so the quarantined backup — a different name —
    is still readable, and a case can assert the evidence survived.
    """
    real_read_bytes = Path.read_bytes

    def read_bytes(self: Path):
        if self.name == 'MERGE_RR':
            raise error
        return real_read_bytes(self)

    monkeypatch.setattr(Path, 'read_bytes', read_bytes)


def vanished_worktree(tmp_path: Path, monkeypatch) -> Path:
    """Deleted out-of-band: ``git`` cannot be spawned, so there is no exit code."""
    return tmp_path / 'deleted-out-of-band'


def worktree_is_a_file(tmp_path: Path, monkeypatch) -> Path:
    """Same spawn failure, different errno (``ENOTDIR``) — a distinct code path in."""
    path = tmp_path / 'a-file'
    path.write_text('not a directory\n')
    return path


def merge_rr_is_a_directory(tmp_path: Path, monkeypatch) -> Path:
    """A valid repo whose MERGE_RR cannot be read as a file (``EISDIR``)."""
    repo, _ = build_mid_rebase_repo(tmp_path)
    _merge_rr_read_fails(
        monkeypatch, IsADirectoryError(errno.EISDIR, 'Is a directory'),
    )
    return repo


def merge_rr_is_unreadable(tmp_path: Path, monkeypatch) -> Path:
    """A valid repo whose MERGE_RR cannot be read at all (``EACCES``)."""
    repo, _ = build_mid_rebase_repo(tmp_path)
    _merge_rr_read_fails(
        monkeypatch, PermissionError(errno.EACCES, 'Permission denied'),
    )
    return repo


#: Hostile states the preflight must survive.  Every one was MEASURED to raise
#: on this branch before the guards landed, so none of them is a hypothetical.
HOSTILE_STATES = (
    vanished_worktree,
    worktree_is_a_file,
    merge_rr_is_a_directory,
    merge_rr_is_unreadable,
)


class TestPreflightIsTotal:
    """The fail-safe contract is an INVARIANT over the entry point, not two patches.

    Both docstrings in the module already assert it — ``guarded_abort``'s "It
    never raises", ``preflight_rebase_recovery``'s "returns an
    unresolved-but-clean result rather than raising" — and what review found is
    that the contract did not hold.  The two defects it named were instances;
    pinning only those leaves the defect class live, and leaves the prose
    untrue for the next reader who relies on it.

    So the battery is over STATES, not over the call sites that happened to be
    found.  ``survey_locks`` is deliberately absent: ``Path.glob`` on an
    unreadable directory was measured to yield ``[]`` rather than raise, so a
    glob arm would assert a hole that does not exist.
    """

    @pytest.mark.parametrize('report_only', [False, True])
    @pytest.mark.parametrize(
        'make_worktree', HOSTILE_STATES, ids=lambda f: f.__name__,
    )
    def test_no_hostile_state_makes_the_preflight_raise(
        self, make_worktree, report_only: bool, tmp_path: Path, monkeypatch,
    ) -> None:
        """Both arms of the one public flag: report-only takes different branches."""
        worktree = make_worktree(tmp_path, monkeypatch)

        result = rebase_recovery.preflight_rebase_recovery(
            worktree, report_only=report_only,
        )

        assert isinstance(result, rebase_recovery.PreflightResult)
        assert result.verdict in {
            rebase_recovery.VERDICT_CLEAN,
            rebase_recovery.VERDICT_REPAIRED,
            rebase_recovery.VERDICT_BLOCKED,
        }

    def test_a_merge_rr_that_could_not_be_read_is_never_called_clean(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Surviving is not enough — the survivor must not report a lie.

        Folding an unreadable MERGE_RR into the absent/healthy branch would
        make the preflight say ``clean`` about a file it never managed to
        inspect, which is exactly the dishonesty ``PreflightResult.verdict``'s
        own docstring argues against for ``report_only``.  Absent means
        healthy; unreadable means unknown, and unknown is not healthy.
        """
        repo = merge_rr_is_unreadable(tmp_path, monkeypatch)

        repaired = rebase_recovery.preflight_rebase_recovery(repo)
        assert repaired.verdict != rebase_recovery.VERDICT_CLEAN
        assert repaired.merge_rr_backup is not None, 'evidence is still preserved'
        assert not (repo / '.git' / 'MERGE_RR').exists()

    def test_an_unreadable_merge_rr_is_reported_unrepaired_when_nothing_moves(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """Report-only leaves it in place, so the caller must be told it is there."""
        repo = merge_rr_is_unreadable(tmp_path, monkeypatch)

        reported = rebase_recovery.preflight_rebase_recovery(repo, report_only=True)

        assert reported.verdict == rebase_recovery.VERDICT_BLOCKED
        assert reported.unrepaired
        assert (repo / '.git' / 'MERGE_RR').exists()

# ---------------------------------------------------------------------------
# The CLI the skills invoke
# ---------------------------------------------------------------------------

class TestPreflightCli:
    """The contract both SKILL.md files already use for ``b3_gate check``.

    They invoke it, parse JSON from stdout, and branch on a ``verdict`` string.
    Matching that shape verbatim means the skills edit reuses a sentence
    pattern already in those files rather than inventing a second convention.
    """

    def _run_cli(self, capsys, *argv: str) -> tuple[int, dict]:
        code = rebase_recovery.main(list(argv))
        out = capsys.readouterr().out
        return code, json.loads(out)

    def test_dangling_fixture_reports_repaired_with_the_id_and_backup(
        self, tmp_path: Path, capsys,
    ) -> None:
        repo, conflict_id = build_mid_rebase_repo(tmp_path)
        _make_dangling(repo, conflict_id)

        code, payload = self._run_cli(capsys, 'preflight', '--worktree', str(repo))

        assert code == 0
        assert payload['verdict'] == 'repaired'
        assert [d['conflict_id'] for d in payload['dangling']] == [conflict_id]
        assert payload['merge_rr_backup'] is not None
        assert Path(payload['merge_rr_backup']).exists()

    def test_healthy_fixture_reports_clean_with_no_backup(
        self, tmp_path: Path, capsys,
    ) -> None:
        repo, _ = build_mid_rebase_repo(tmp_path)

        code, payload = self._run_cli(capsys, 'preflight', '--worktree', str(repo))

        assert code == 0
        assert payload['verdict'] == 'clean'
        assert payload['dangling'] == []
        assert payload['merge_rr_backup'] is None

    def test_report_only_detects_without_moving_anything(
        self, tmp_path: Path, capsys,
    ) -> None:
        """Detection and reporting with zero mutation, so an operator can look first."""
        repo, conflict_id = build_mid_rebase_repo(tmp_path)
        _make_dangling(repo, conflict_id)
        merge_rr = repo / '.git' / 'MERGE_RR'
        original = merge_rr.read_bytes()

        code, payload = self._run_cli(
            capsys, 'preflight', '--worktree', str(repo), '--report-only',
        )

        assert code == 0
        assert [d['conflict_id'] for d in payload['dangling']] == [conflict_id]
        assert payload['merge_rr_backup'] is None
        assert merge_rr.read_bytes() == original
        assert list((repo / '.git').glob('MERGE_RR.quarantined-*')) == []

    def test_stdout_is_exactly_one_json_object(
        self, tmp_path: Path, capsys,
    ) -> None:
        """A caller does ``json.loads(stdout)``; a second line or a log breaks it."""
        repo, conflict_id = build_mid_rebase_repo(tmp_path)
        _make_dangling(repo, conflict_id)

        rebase_recovery.main(['preflight', '--worktree', str(repo)])

        out = capsys.readouterr().out
        assert len([line for line in out.splitlines() if line.strip()]) == 1
        assert isinstance(json.loads(out), dict)

    def test_verdict_is_always_one_of_the_documented_enum(
        self, tmp_path: Path, capsys,
    ) -> None:
        repo, conflict_id = build_mid_rebase_repo(tmp_path)
        _make_dangling(repo, conflict_id)

        _, repaired = self._run_cli(capsys, 'preflight', '--worktree', str(repo))
        _, again = self._run_cli(capsys, 'preflight', '--worktree', str(repo))

        allowed = {
            rebase_recovery.VERDICT_CLEAN,
            rebase_recovery.VERDICT_REPAIRED,
            rebase_recovery.VERDICT_BLOCKED,
        }
        assert repaired['verdict'] in allowed
        assert again['verdict'] in allowed

    def test_lock_stale_after_seconds_is_a_flag_not_a_config_knob(
        self, tmp_path: Path, capsys,
    ) -> None:
        """The threshold varies per invocation, so it is an argument, not config.

        A young unheld lock is retained at the default and swept once the
        caller lowers the threshold below its age — which is the only
        observable difference the flag is supposed to make.
        """
        repo, _ = build_mid_rebase_repo(tmp_path)
        _git_ok(repo, 'rebase', '--abort')
        lock = repo / '.git' / 'MERGE_RR.lock'
        lock.touch()

        _, default = self._run_cli(capsys, 'preflight', '--worktree', str(repo))
        assert default['locks_removed'] == []
        assert lock.exists()

        _, lowered = self._run_cli(
            capsys, 'preflight', '--worktree', str(repo),
            '--lock-stale-after-seconds', '0',
        )
        assert [Path(f['path']).name for f in lowered['locks_removed']] == [
            'MERGE_RR.lock',
        ]
        assert not lock.exists()

    def test_an_unresolvable_worktree_does_not_crash_the_cli(
        self, tmp_path: Path, capsys,
    ) -> None:
        """Fail-safe all the way out: this decorates recovery, never blocks it."""
        not_a_repo = tmp_path / 'plain'
        not_a_repo.mkdir()

        code, payload = self._run_cli(
            capsys, 'preflight', '--worktree', str(not_a_repo),
        )

        assert code == 0
        assert payload['resolved'] is False
        assert payload['verdict'] == 'clean'

    def test_report_only_never_calls_detected_damage_clean(
        self, tmp_path: Path, capsys,
    ) -> None:
        """`clean` must mean "nothing needs attention", not "I changed nothing".

        The two coincide everywhere EXCEPT here, and this is the case a caller
        acts on: report-only deliberately leaves the damage in place, so a
        verdict derived purely from what was MUTATED reports `clean` for a
        worktree it just described as dangling.  A skill branching on the
        verdict — which is the whole point of the enum — would then proceed
        unguarded into exactly the state the preflight exists to catch.
        """
        repo, conflict_id = build_mid_rebase_repo(tmp_path)
        _make_dangling(repo, conflict_id)

        _, payload = self._run_cli(
            capsys, 'preflight', '--worktree', str(repo), '--report-only',
        )

        assert payload['dangling'], 'fixture expected a detected dangling ref'
        assert payload['verdict'] == rebase_recovery.VERDICT_BLOCKED
        assert conflict_id in ' '.join(payload['unrepaired'])

    def test_a_held_lock_blocks_even_when_everything_else_was_repaired(
        self, tmp_path: Path, capsys,
    ) -> None:
        """The other unrepaired arm: a live holder is a human's decision."""
        repo, conflict_id = build_mid_rebase_repo(tmp_path)
        _make_dangling(repo, conflict_id)
        lock = repo / '.git' / 'MERGE_RR.lock'
        lock.touch()

        with lock.open('a'):
            _, payload = self._run_cli(
                capsys, 'preflight', '--worktree', str(repo),
                '--lock-stale-after-seconds', '0',
            )

        assert payload['merge_rr_backup'] is not None, 'MERGE_RR was still repaired'
        assert payload['verdict'] == rebase_recovery.VERDICT_BLOCKED
        assert str(os.getpid()) in ' '.join(payload['unrepaired'])
