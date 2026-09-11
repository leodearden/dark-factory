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

import logging
import os
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from _orch_helpers import assert_isolated_git_repo, git_env_with_ceiling

from orchestrator import rebase_recovery

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
